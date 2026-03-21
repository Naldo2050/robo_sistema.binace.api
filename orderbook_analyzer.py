# orderbook_analyzer.py v2.2.0 - ASYNC COM SESSÃO HTTP REUTILIZÁVEL + MELHORIAS
"""
OrderBook Analyzer para Binance Futures com validação robusta e fetch assíncrono.

🔹 v2.2.0 (MELHORIAS INTERNAS, API EXTERNA IGUAL v2.1.0):
  ✅ Locks assíncronos para sessão HTTP e cache (sem race conditions)
  ✅ Timeout de request usa ORDERBOOK_REQUEST_TIMEOUT (sem 5.0 hardcoded)
  ✅ Corrigido market impact SELL (usava ordem errada dos bids)
  ✅ Iceberg detection mais robusto usando agregação por preço (_price_map)
  ✅ Walls detection usando quantil 90% (menos falsos positivos)
  ✅ Mantida API pública e estrutura de retorno da v2.1.0

🔹 v2.1.0 (PERFORMANCE: SESSÃO REUTILIZÁVEL):
  ✅ ClientSession reutilizável por OrderBookAnalyzer
  ✅ Permite injetar sessão externa (parâmetro `session=...`)
  ✅ Evita criar/fechar sessão + handshake TCP/SSL a cada requisição
  ✅ Compatível com asyncio.run(...), desde que o loop seja bem gerenciado

🔹 BREAKING CHANGES v2.0.0 (mantidos):
  ✅ Migrado de requests (bloqueante) para aiohttp (assíncrono)
  ✅ Método analyze() é async (use: await oba.analyze())
  ✅ _fetch_orderbook() é async
  ✅ Todos os time.sleep() convertidos para asyncio.sleep()
  ✅ Zero bloqueio do event loop
  ✅ Suporte a múltiplas requisições simultâneas

🔹 CORREÇÕES CRÍTICAS v1.6.0 (mantidas):
  ✅ Validação NÃO modifica snapshot original (cria cópia)
  ✅ Rejeita dados parciais independente de config se ruins
  ✅ Fallback usa timestamp original dos dados (não atual)
  ✅ Emergency mode mais restritivo (apenas para erros leves)
  ✅ Validação de idade de dados (max 60s)
  ✅ Proteção total contra divisão por zero
  ✅ Fallbacks de config mais conservadores
  ✅ Flags de qualidade mais claras
  ✅ Logs informativos sem poluir
  ✅ Validação de timestamp obrigatória
"""

import logging
import time
import asyncio
import threading
import random
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import deque

import aiohttp  # ✅ async HTTP client
import numpy as np

# Import circuit breaker
from orderbook_core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from orderbook_core.constants import THRESHOLDS

# Import EventFactory para eventos inválidos (PATCH 6.2)
from orderbook_core.event_factory import build_invalid_orderbook_event
from orderbook_core.structured_logging import StructuredLogger
from orderbook_core.tracing_utils import TracerWrapper

# Import fallback robusto
from orderbook_core.orderbook_fallback import get_fallback_instance, fetch_with_fallback

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

from monitoring.time_manager import TimeManager
from orderbook_core.protocols import TimeManagerProtocol
from orderbook_core.metrics import OrderBookMetrics, MetricsTracker
from orderbook_core.orderbook_config import OrderBookConfig
from orderbook_core.orderbook import OrderBookSnapshot

# ===== IMPORTA PARÂMETROS DE CONFIGURAÇÃO =====
try:
    from config import (
        ORDER_BOOK_DEPTH_LEVELS,
        SPREAD_TIGHT_THRESHOLD_BPS,
        SPREAD_AVG_WINDOWS_MIN,
        ORDERBOOK_CRITICAL_IMBALANCE,
        ORDERBOOK_MIN_DOMINANT_USD,
        ORDERBOOK_MIN_RATIO_DOM,
        ORDERBOOK_REQUEST_TIMEOUT,
        ORDERBOOK_RETRY_DELAY,
        ORDERBOOK_MAX_RETRIES,
        ORDERBOOK_MAX_REQUESTS_PER_MIN,
        ORDERBOOK_CACHE_TTL,
        ORDERBOOK_MAX_STALE,
        ORDERBOOK_MIN_DEPTH_USD,
        ORDERBOOK_ALLOW_PARTIAL,
        ORDERBOOK_USE_FALLBACK,
        ORDERBOOK_FALLBACK_MAX_AGE,
        ORDERBOOK_EMERGENCY_MODE,
        # Circuit Breaker Config
        ORDERBOOK_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        ORDERBOOK_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
        ORDERBOOK_CIRCUIT_BREAKER_TIMEOUT_SECONDS,
        ORDERBOOK_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
    )
    CONFIG_LOADED = True
except Exception as e:
    logging.warning(f"⚠️ Config não carregado ({e}), usando valores seguros")
    CONFIG_LOADED = False

    # ✅ FALLBACKS CONSERVADORES v1.6.0
    ORDER_BOOK_DEPTH_LEVELS = [1, 5, 10, 25]
    SPREAD_TIGHT_THRESHOLD_BPS = 0.2
    SPREAD_AVG_WINDOWS_MIN = [60, 1440]
    ORDERBOOK_CRITICAL_IMBALANCE = 0.95
    ORDERBOOK_MIN_DOMINANT_USD = 2_000_000.0
    ORDERBOOK_MIN_RATIO_DOM = 20.0

    # ✅ VALORES CONSERVADORES (NÃO PERMISSIVOS)
    ORDERBOOK_REQUEST_TIMEOUT = 10.0
    ORDERBOOK_RETRY_DELAY = 2.0
    ORDERBOOK_MAX_RETRIES = 3
    ORDERBOOK_MAX_REQUESTS_PER_MIN = 10
    ORDERBOOK_CACHE_TTL = 15.0
    ORDERBOOK_MAX_STALE = 60.0
    ORDERBOOK_MIN_DEPTH_USD = 1_000.0
    ORDERBOOK_ALLOW_PARTIAL = False
    ORDERBOOK_USE_FALLBACK = True
    ORDERBOOK_FALLBACK_MAX_AGE = 120
    ORDERBOOK_EMERGENCY_MODE = False

    # Circuit Breaker Configuration (Safe defaults)
    ORDERBOOK_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
    ORDERBOOK_CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2
    ORDERBOOK_CIRCUIT_BREAKER_TIMEOUT_SECONDS = 30.0
    ORDERBOOK_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = 3

# ===== 🆕 VALIDAÇÃO DE IDADE MÁXIMA =====
ORDERBOOK_MAX_AGE_MS = 60000  # ✅ 60 segundos máximo

SCHEMA_VERSION = "2.1.0"  # Mantido para compatibilidade externa
ENGINE_VERSION = "2.2.0"  # Nova: versão interna do engine (PATCH 4)

# ===== 🆕 CONSTANTES DE CONFIGURAÇÃO DINÂMICA =====
DEFAULT_HISTORY_MAXLEN = 50  # PATCH 6


# ===== EXCEÇÃO CUSTOMIZADA =====
class OrderBookUnavailableError(Exception):
    """Levantada quando orderbook não pode ser obtido ou é inválido."""
    pass


# ===== UTILS =====
def _to_float_list(levels: Any) -> List[Tuple[float, float]]:
    """
    Converte níveis de orderbook [price, qty] para float tuples.

    ✅ v1.6.0: NÃO MODIFICA INPUT, retorna nova lista
    """
    if not levels:
        return []

    out: List[Tuple[float, float]] = []
    for lv in levels:
        try:
            if isinstance(lv, (list, tuple)) and len(lv) >= 2:
                p = float(lv[0])
                q = float(lv[1])
                if p > 0 and q > 0:
                    out.append((p, q))
            else:
                logging.debug(f"⚠️ Nível inválido ignorado: {lv}")
        except (ValueError, TypeError, IndexError) as e:
            logging.debug(f"⚠️ Erro ao converter nível {lv}: {e}")
            continue
    return out


def _sum_depth_usd(levels: List[Tuple[float, float]], top_n: int) -> float:
    """
    Soma profundidade em USD dos top N níveis.

    ✅ v1.6.0: Assume entrada já convertida
    """
    if not levels:
        return 0.0

    arr = levels[:max(1, top_n)]

    try:
        return float(sum(p * q for p, q in arr if isinstance(p, (int, float)) and isinstance(q, (int, float))))
    except Exception as e:
        logging.debug(f"⚠️ Erro ao calcular depth USD: {e}")
        return 0.0


def _simulate_market_impact(
    levels: List[Tuple[float, float]],
    usd_amount: float,
    side: str,
    mid: Optional[float]
) -> Dict[str, Any]:
    """Simula impacto de ordem de mercado."""
    if not levels or usd_amount <= 0:
        return {
            "usd": usd_amount,
            "move_usd": 0.0,
            "bps": 0.0,
            "levels": 0,
            "vwap": None,
        }

    spent = 0.0
    filled_qty = 0.0
    vwap_numer = 0.0
    levels_crossed = 0
    terminal_price = levels[-1][0] if side == "buy" else levels[0][0]

    for i, (price, qty) in enumerate(levels):
        level_usd = price * qty
        if spent + level_usd >= usd_amount:
            remaining = usd_amount - spent
            dq = remaining / price if price > 0 else 0.0
            vwap_numer += price * dq
            filled_qty += dq
            spent = usd_amount
            terminal_price = price
            levels_crossed = i + 1
            break
        else:
            spent += level_usd
            filled_qty += qty
            vwap_numer += price * qty
            terminal_price = price
            levels_crossed = i + 1

    vwap = vwap_numer / filled_qty if filled_qty > 0 else None
    move_usd = 0.0
    bps = 0.0

    if mid and terminal_price and mid > 0:
        if side == "buy":
            move_usd = max(0.0, terminal_price - mid)
        else:
            move_usd = max(0.0, mid - terminal_price)
        bps = (move_usd / mid) * 10000.0

    return {
        "usd": usd_amount,
        "move_usd": round(move_usd, 4),
        "bps": round(bps, 4),
        "levels": levels_crossed,
        "vwap": vwap,
        "final_price": terminal_price,
    }


# ===== ANALYZER =====
class OrderBookAnalyzer:
    """
    Analisador de Order Book para Binance Futures com validação robusta e fetch assíncrono.

    ✅ v2.2.0: Locks para sessão/cache, market impact SELL corrigido, iceberg/walls melhorados
    ✅ v2.1.0: Sessão HTTP reutilizável por analyzer (ou injetada), sem overhead por chamada
    ✅ v2.0.0: 100% async, zero bloqueio
    ✅ v1.6.0: Validação rigorosa, não modifica dados, fallback conservador
    """

    def __init__(
        self,
        symbol: str,
        liquidity_flow_alert_percentage: float = 0.4,
        wall_std_dev_factor: float = 3.0,
        top_n_levels: int = 20,
        ob_limit_fetch: int = 100,
        time_manager=None,
        cache_ttl_seconds: float = 30.0,
        max_stale_seconds: float = 300.0,
        rate_limit_threshold: int = 5,
        # Novos parâmetros para compatibilidade com testes
        alert_threshold: Optional[float] = None,
        wall_detection_factor: Optional[float] = None,
        session: Optional[aiohttp.ClientSession] = None,
        metrics: Optional[OrderBookMetrics] = None,
        cfg: Optional[OrderBookConfig] = None,
        **kwargs  # Aceita parâmetros extras
    ):
        self.symbol = symbol
        self.time_manager = time_manager or TimeManager()
        
        # Compatibilidade: se alert_threshold for fornecido, usa ele
        if alert_threshold is not None:
            self.liquidity_flow_alert_percentage = alert_threshold
        else:
            self.liquidity_flow_alert_percentage = liquidity_flow_alert_percentage
        
        # Compatibilidade: se wall_detection_factor for fornecido, usa ele
        if wall_detection_factor is not None:
            self.wall_std_dev_factor = wall_detection_factor
        else:
            self.wall_std_dev_factor = wall_std_dev_factor
        
        # Adiciona atributos para compatibilidade
        self.alert_threshold = self.liquidity_flow_alert_percentage
        self.wall_std = float(wall_std_dev_factor)  # ADICIONADO: atributo esperado por _detect_walls
        self.wall_detection_factor = self.wall_std_dev_factor
        self.top_n = int(top_n_levels)
        self.ob_limit_fetch = int(ob_limit_fetch)
        
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_stale_seconds = max_stale_seconds
        self.rate_limit_threshold = rate_limit_threshold
        
        # Inicialização do cache e rate limiter
        self._cache = {}
        self._cache_timestamps = {}
        self._rate_limit_timestamps = []
        self._rate_limit_window = 60.0  # 60 seconds
        
        # Inicializa outros atributos se necessário
        self.last_snapshot = None
        self._failure_count = 0
        self.circuit_breaker_state = {'state': 'CLOSED', 'failure_count': 0}

        # ===== CONFIG ESTRUTURADA (OrderBookConfig) =====
        if cfg is None:
            # Usa os valores já importados de config.py ou os fallbacks
            cfg = OrderBookConfig(
                depth_levels=list(ORDER_BOOK_DEPTH_LEVELS),
                spread_tight_threshold_bps=float(SPREAD_TIGHT_THRESHOLD_BPS),
                spread_avg_windows_min=list(SPREAD_AVG_WINDOWS_MIN),

                critical_imbalance=float(ORDERBOOK_CRITICAL_IMBALANCE),
                min_dominant_usd=float(ORDERBOOK_MIN_DOMINANT_USD),
                min_ratio_dom=float(ORDERBOOK_MIN_RATIO_DOM),

                request_timeout=float(ORDERBOOK_REQUEST_TIMEOUT),
                retry_delay=float(ORDERBOOK_RETRY_DELAY),
                max_retries=int(ORDERBOOK_MAX_RETRIES),
                max_requests_per_min=int(ORDERBOOK_MAX_REQUESTS_PER_MIN),

                cache_ttl=float(ORDERBOOK_CACHE_TTL),
                max_stale=float(ORDERBOOK_MAX_STALE),

                min_depth_usd=float(ORDERBOOK_MIN_DEPTH_USD),
                allow_partial=bool(ORDERBOOK_ALLOW_PARTIAL),
                use_fallback=bool(ORDERBOOK_USE_FALLBACK),
                fallback_max_age=float(ORDERBOOK_FALLBACK_MAX_AGE),
                emergency_mode=bool(ORDERBOOK_EMERGENCY_MODE),
            )
        self.cfg = cfg

        # Config derivada (mantém API atual do __init__)
        self.cache_ttl_seconds = cache_ttl_seconds if cache_ttl_seconds is not None else self.cfg.cache_ttl
        self.max_stale_seconds = max_stale_seconds if max_stale_seconds is not None else self.cfg.max_stale
        self.rate_limit_threshold = rate_limit_threshold if rate_limit_threshold is not None else self.cfg.max_requests_per_min

        # Sessão HTTP (pode ser injetada ou criada lazy internamente)
        self._session: Optional[aiohttp.ClientSession] = session
        # True se a sessão foi criada pelo próprio Analyzer (nesse caso, ele fecha)
        self._owns_session: bool = session is None

        # Locks para evitar race conditions (NOVO v2.2.0)
        # Usa threading.Lock para cache, asyncio.Lock para sessão
        self._session_lock = asyncio.Lock()
        self._cache_lock = threading.Lock()

        # Cache (PATCH 14: usar monotonic para idade)
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cache_timestamp_mono: float = 0.0

        # Fallback
        self._last_valid_snapshot: Optional[Dict[str, Any]] = None
        self._last_valid_timestamp_mono: float = 0.0
        self._last_valid_exchange_ts: Optional[int] = None

        # PATCH 5: tracking de fonte de dados
        self._last_fetch_source: str = "unknown"
        self._last_fetch_age_seconds: float = 0.0

        # Histórico configurável (usa THRESHOLDS, não magic numbers soltos)
        default_hist = THRESHOLDS.DEFAULT_HISTORY_SIZE
        min_hist = THRESHOLDS.MIN_HISTORY_SIZE
        max_hist = THRESHOLDS.MAX_HISTORY_SIZE

        history_maxlen = int(os.getenv("ORDERBOOK_HISTORY_MAXLEN", str(default_hist)))
        history_maxlen = max(min_hist, min(history_maxlen, max_hist))
        self._history = deque(maxlen=history_maxlen)
        self._history_maxlen = history_maxlen

        # PATCH 15: rate-limit em deque (O(1))
        self._request_times_mono = deque()
        # Shims de compatibilidade com nomes antigos (para _check_rate_limit legado)
        self._rate_limit_timestamps = self._request_times_mono
        self._rate_limit_threshold = self.rate_limit_threshold

        # PATCH 7: dynamic config via environment
        self.dynamic_thresholds: Dict[str, float] = {}
        self._min_depth_usd_override: Optional[float] = None
        self._iceberg_tol_override: Optional[float] = None
        self._wall_mult_override: Optional[float] = None
        self._load_dynamic_config()

        # Metrics
        self.metrics = metrics or OrderBookMetrics.build_default()

        # Controle interno para warnings de métodos deprecated
        self._shims_deprecated_warned = False

        # Initialize top_n with a default value
        # This will be overridden by _load_dynamic_config if ORDERBOOK_TOP_N is set
        self.top_n = 50

        # Stats
        self._fetch_errors = 0
        self._total_fetches = 0
        self._validation_failures = 0
        self._cache_hits = 0
        self._stale_data_uses = 0
        self._emergency_uses = 0
        self._old_data_rejected = 0

        # Circuit Breaker for API resilience
        # Usa valores do cfg se disponíveis, senão fallbacks das constantes
        failure_threshold = getattr(self.cfg, 'circuit_breaker_failure_threshold', ORDERBOOK_CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        success_threshold = getattr(self.cfg, 'circuit_breaker_success_threshold', ORDERBOOK_CIRCUIT_BREAKER_SUCCESS_THRESHOLD)
        timeout_seconds = getattr(self.cfg, 'circuit_breaker_timeout_seconds', ORDERBOOK_CIRCUIT_BREAKER_TIMEOUT_SECONDS)
        half_open_max_calls = getattr(self.cfg, 'circuit_breaker_half_open_max_calls', ORDERBOOK_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS)
        
        cb_config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            half_open_max_calls=half_open_max_calls,
        )
        self._circuit_breaker = CircuitBreaker(name=f"orderbook_{self.symbol}", config=cb_config)

        # Config adicionais
        self.depth_levels: List[int] = list(self.cfg.depth_levels)
        self.spread_tight_threshold_bps: float = float(self.cfg.spread_tight_threshold_bps)
        self.spread_avg_windows_min: List[int] = list(self.cfg.spread_avg_windows_min)
        self.spread_history: List[Tuple[int, float]] = []

        # Memória para iceberg
        self.prev_snapshot: Optional[Dict[str, Any]] = None
        self.last_event_ts_ms: Optional[int] = None

        logging.info(
            "✅ OrderBook Analyzer v%s (engine %s) inicializado | "
            "Symbol: %s | Alert: %.0f%% | Wall STD: %.1fx | "
            "Top N: %s | Cache TTL: %.1fs | Max Stale: %.1fs | "
            "Rate Limit: %s req/min | Config loaded: %s | History maxlen: %s | "
            "Metrics: %s",
            SCHEMA_VERSION,
            ENGINE_VERSION,
            self.symbol,
            self.alert_threshold * 100,
            self.wall_std_dev_factor,
            self.top_n,
            self.cfg.cache_ttl,
            self.cfg.max_stale,
            self.cfg.max_requests_per_min,
            "✅" if CONFIG_LOADED else "❌ (usando defaults)",
            self._history_maxlen,
            "✅" if self.metrics.enabled else "❌ (desabilitadas)",
        )

        # Logger estruturado (para ELK/Splunk/JSON logs)
        self.slog = StructuredLogger("orderbook_analyzer", self.symbol)

        # Tracer distribuído (OpenTelemetry opcional)
        self.tracer = TracerWrapper(
            service_name="orderbook_service",
            component="analyzer",
            symbol=self.symbol,
        )

    # ===== PATCH 7: Dynamic config =====
    def _load_dynamic_config(self) -> None:
        """Carrega configurações dinâmicas de environment variables."""

        def _f(name: str) -> Optional[float]:
            v = os.getenv(name)
            if not v:
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def _i(name: str) -> Optional[int]:
            v = os.getenv(name)
            if not v:
                return None
            try:
                return int(v)
            except ValueError:
                return None

        env_top_n = _i("ORDERBOOK_TOP_N")
        if env_top_n is not None:
            self.top_n = max(1, min(env_top_n, 200))

        wall_mult = _f("WALL_THRESHOLD_MULT")
        iceberg_tol = _f("ICEBERG_TOL")
        min_liq = _f("MIN_LIQUIDITY_USD")

        if wall_mult is not None and wall_mult > 0:
            self._wall_mult_override = wall_mult
        if iceberg_tol is not None and 0 < iceberg_tol <= 1.0:
            self._iceberg_tol_override = iceberg_tol
        if min_liq is not None and min_liq > 0:
            self._min_depth_usd_override = min_liq

        self.dynamic_thresholds = {
            "wall_threshold_multiplier": float(self._wall_mult_override or self.wall_std_dev_factor),
            "iceberg_tolerance": float(self._iceberg_tol_override or 0.75),
            "min_liquidity_usd": float(self._min_depth_usd_override or self.cfg.min_depth_usd),
        }

    # ===== PATCH 13: Snapshot copy helper =====
    def _snapshot_copy(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cria cópia segura do snapshot para evitar mutação acidental.
        """
        if not snap:
            return {}
        out = dict(snap)
        if "bids" in out and isinstance(out["bids"], list):
            out["bids"] = list(out["bids"])
        if "asks" in out and isinstance(out["asks"], list):
            out["asks"] = list(out["asks"])
        if "depth_metrics" in out and isinstance(out["depth_metrics"], dict):
            out["depth_metrics"] = dict(out["depth_metrics"])
        return out

    # ===== CLOSE =====
    async def close(self):
        """
        Fecha a ClientSession se ela tiver sido criada internamente.
        Se uma sessão externa foi injetada no __init__, o chamador é
        responsável por fechá-la.
        """
        try:
            async with self._session_lock:
                if self._owns_session and self._session and not self._session.closed:
                    await self._session.close()
                self._session = None
        except Exception:
            self._session = None

    # ===== ASYNC CONTEXT MANAGER (ETAPA 4) =====
    async def __aenter__(self) -> "OrderBookAnalyzer":
        """
        Permite uso com:
            async with OrderBookAnalyzer(...) as oba:
                ...
        """
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        """
        Garante cleanup via close() ao sair do bloco async with.
        Retorna False para não suprimir exceções.
        """
        await self.close()
        return False

    # ===== PATCH 3: Timeout consistente =====
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Retorna uma sessão HTTP reutilizável.

        ✅ v2.2.0: protegido por lock para evitar race na criação.
        """
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(
                    total=self.cfg.request_timeout,
                    connect=self.cfg.request_timeout,  # PATCH 3: sem hardcode
                    sock_read=self.cfg.request_timeout,
                )
                connector = aiohttp.TCPConnector(
                    limit=10,
                    limit_per_host=5,
                    ttl_dns_cache=300,
                    force_close=False,
                    enable_cleanup_closed=True,
                )
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    raise_for_status=False,
                )
                self._owns_session = True
            session = self._session
        return session

    # ===== ✅ VALIDAÇÃO (PATCH 17: ordenação) =====
    def _validate_snapshot(
        self,
        snap: Dict[str, Any],
        max_age_ms: Optional[int] = None
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Valida snapshot de orderbook SEM MODIFICAR o original.

        ✅ v1.6.0 CORREÇÕES:
        - NÃO modifica snap original
        - Retorna snapshot convertido separadamente
        - Validação de idade de dados
        - Mais rigorosa com dados parciais

        PATCH 17: Valida ordenação do book (bids desc, asks asc)

        Returns:
            (is_valid, list_of_issues, converted_snapshot)
        """
        issues: List[str] = []

        # 1. Estrutura básica
        if not isinstance(snap, dict):
            issues.append("snapshot não é dict")
            return False, issues, {}

        if "bids" not in snap or "asks" not in snap:
            issues.append("snapshot sem bids/asks")
            return False, issues, {}

        # ✅ CRIA CÓPIA PARA NÃO MODIFICAR ORIGINAL
        converted_snap = {
            "lastUpdateId": snap.get("lastUpdateId"),
            "E": snap.get("E"),
            "T": snap.get("T"),
            "symbol": snap.get("symbol") or getattr(self, "symbol", "UNKNOWN")
        }

        # ✅ CONVERTE PARA NOVA ESTRUTURA (não modifica original)
        raw_bids = snap.get("bids", [])
        raw_asks = snap.get("asks", [])

        bids = _to_float_list(raw_bids) if raw_bids else []
        asks = _to_float_list(raw_asks) if raw_asks else []

        converted_snap["bids"] = bids
        converted_snap["asks"] = asks

        # 2. Dados não vazios
        if not bids or not asks:
            issues.append(f"orderbook vazio (bids={len(bids)}, asks={len(asks)})")
            return False, issues, converted_snap

        # PATCH 17: Valida ordenação do book
        max_depth = THRESHOLDS.ORDER_VALIDATION_DEPTH

        # bids: preços devem ser não-crescentes
        for i in range(1, min(len(bids), max_depth)):
            if bids[i][0] > bids[i-1][0]:
                issues.append("bids fora de ordem (preço subindo)")
                return False, issues, converted_snap

        # asks: preços devem ser não-decrescentes
        for i in range(1, min(len(asks), max_depth)):
            if asks[i][0] < asks[i-1][0]:
                issues.append("asks fora de ordem (preço descendo)")
                return False, issues, converted_snap

        # 3. ✅ VALIDAÇÃO DE IDADE
        if max_age_ms is None:
            max_age_ms = getattr(self.cfg, 'max_age_ms', ORDERBOOK_MAX_AGE_MS)

        exchange_ts = None
        for key in ("E", "T"):
            v = snap.get(key)
            if isinstance(v, (int, float)) and v > 0:
                exchange_ts = int(v)
                break

        if exchange_ts is not None:
            now_ms = self.time_manager.now_ms()
            age_ms = now_ms - exchange_ts

            # Tolerância de 5 segundos para dessincronização de relógio
            CLOCK_TOLERANCE_MS = 5000

            if age_ms < -CLOCK_TOLERANCE_MS:
                # Timestamp muito no futuro (mais de 5s) - problema real
                issues.append(f"timestamp muito no futuro! (age={age_ms}ms)")
                return False, issues, converted_snap

            if age_ms < 0:
                # Timestamp ligeiramente no futuro - dessincronização de relógio
                logging.debug(f"⚠️ Snapshot com timestamp ligeiramente no futuro (age={age_ms}ms) - tolerando")
                # Ajustar idade para 0 (considerar como "agora")
                age_ms = 0

            if age_ms > max_age_ms:
                issues.append(f"dados muito antigos ({age_ms}ms > {max_age_ms}ms)")
                return False, issues, converted_snap
        else:
            # ✅ EXIGE TIMESTAMP VÁLIDO
            issues.append("sem timestamp válido (E ou T)")
            return False, issues, converted_snap

        # 4. Valores numéricos válidos
        try:
            best_bid_price = float(bids[0][0])
            best_bid_qty = float(bids[0][1])
            best_ask_price = float(asks[0][0])
            best_ask_qty = float(asks[0][1])

            if best_bid_price <= 0 or best_ask_price <= 0:
                issues.append(f"preços inválidos (bid={best_bid_price}, ask={best_ask_price})")
                return False, issues, converted_snap

            if best_bid_qty <= 0 or best_ask_qty <= 0:
                issues.append(f"quantidades zero (bid_qty={best_bid_qty}, ask_qty={best_ask_qty})")
                return False, issues, converted_snap

            # 5. Spread não pode ser negativo
            if best_ask_price < best_bid_price:
                issues.append(f"spread negativo! (bid={best_bid_price} > ask={best_ask_price})")
                return False, issues, converted_snap

            if best_bid_price > 0:
                spread_pct = (best_ask_price - best_bid_price) / best_bid_price * 100
            else:
                spread_pct = 999.0

            # 6. Spread absurdo (> 10%)
            max_spread = THRESHOLDS.MAX_SPREAD_PERCENT
            if spread_pct > max_spread:
                issues.append(f"spread absurdo ({spread_pct:.2f}% > {max_spread:.2f}%)")
                return False, issues, converted_snap

            # 7. ✅ VOLUME MÍNIMO (MAIS RIGOROSO)
            bid_vol = sum(float(b[1]) for b in bids[:5] if len(b) >= 2)
            ask_vol = sum(float(a[1]) for a in asks[:5] if len(a) >= 2)

            # ✅ SEMPRE REJEITA SE ALGUM LADO É ZERO
            if bid_vol == 0 or ask_vol == 0:
                issues.append(f"volume zero detectado (bid={bid_vol}, ask={ask_vol})")
                return False, issues, converted_snap

            # 8. ✅ PROFUNDIDADE USD MÍNIMA (RIGOROSA, PATCH 3)
            bid_depth_usd = _sum_depth_usd(bids, 5)
            ask_depth_usd = _sum_depth_usd(asks, 5)

            # PATCH 7: Dynamic min depth
            min_depth = float(self._min_depth_usd_override or self.cfg.min_depth_usd)

            # SEMPRE rejeita se zero
            if bid_depth_usd == 0 or ask_depth_usd == 0:
                issues.append(
                    f"liquidez ZERO (bid=${bid_depth_usd:.0f}, ask=${ask_depth_usd:.0f})"
                )
                return False, issues, converted_snap

            # Verifica se está MUITO abaixo do mínimo
            if bid_depth_usd < min_depth or ask_depth_usd < min_depth:
                issues.append(
                    f"liquidez muito baixa (bid=${bid_depth_usd:.0f}, "
                    f"ask=${ask_depth_usd:.0f}, min=${min_depth:.0f})"
                )

                # Só permite se >= 50% do mínimo E config permite
                if self.cfg.allow_partial:
                    pct_bid = (bid_depth_usd / min_depth) * 100
                    pct_ask = (ask_depth_usd / min_depth) * 100

                    min_ratio = THRESHOLDS.MIN_LIQUIDITY_PARTIAL_RATIO * 100.0  # em %
                    if pct_bid >= min_ratio and pct_ask >= min_ratio:
                        logging.warning(
                            f"⚠️ Liquidez baixa mas aceita "
                            f"(bid={pct_bid:.0f}%, ask={pct_ask:.0f}% do mínimo, "
                            f"limite={min_ratio:.0f}%)"
                        )
                    else:
                        issues.append(
                            f"liquidez < {min_ratio:.0f}% do mínimo (rejeitado mesmo com ALLOW_PARTIAL)"
                        )
                        return False, issues, converted_snap
                else:
                    return False, issues, converted_snap

            # 🆕 Calculation of deeper metrics for AI Payload
            bid_depth_top5 = bid_depth_usd  # Alias
            ask_depth_top5 = ask_depth_usd  # Alias

            # Additional deeper check
            bid_depth_top20 = _sum_depth_usd(bids, 20)
            ask_depth_top20 = _sum_depth_usd(asks, 20)

            depth_imbalance = 0.0
            if (bid_depth_top5 + ask_depth_top5) > 0:
                depth_imbalance = (bid_depth_top5 - ask_depth_top5) / (bid_depth_top5 + ask_depth_top5)

            # Store in converted snap for later use
            converted_snap["depth_metrics"] = {
                "bid_liquidity_top5": bid_depth_top5,
                "ask_liquidity_top5": ask_depth_top5,
                "bid_liquidity_top20": bid_depth_top20,
                "ask_liquidity_top20": ask_depth_top20,
                "depth_imbalance": depth_imbalance
            }

            return True, issues, converted_snap

        except (IndexError, ValueError, TypeError) as e:
            issues.append(f"erro ao validar dados: {e}")
            return False, issues, converted_snap

    # ===== PATCH 15: Rate limiting com deque =====
    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        current_time = time.time()
    
        # Initialize rate limit tracking
        if not hasattr(self, '_rate_limit_timestamps'):
            self._rate_limit_timestamps = []
            self._rate_limit_threshold = self.rate_limit_threshold
            self._rate_limit_window = 60.0  # 60 seconds window
    
        # Remove timestamps older than window
        window_start = current_time - self._rate_limit_window
        self._rate_limit_timestamps = [
            ts for ts in self._rate_limit_timestamps
            if ts > window_start
        ]
    
        # Check if we've exceeded threshold
        if len(self._rate_limit_timestamps) >= self._rate_limit_threshold:
            return False
    
        # Add current timestamp and allow request
        self._rate_limit_timestamps.append(current_time)
        return True

    def _register_request(self):
        """Registra request para tracking."""
        self._request_times_mono.append(time.monotonic())

    async def _get_stale_fallback(self) -> Optional[Dict[str, Any]]:
        """Get stale data fallback when circuit breaker is open."""
        last_valid = None
        last_valid_ts = 0.0
        last_exchange_ts = None

        # Use synchronous lock acquisition for stale fallback
        # (simpler approach since we're in an emergency scenario)
        last_valid = self._last_valid_snapshot
        last_valid_ts = self._last_valid_timestamp_mono
        last_exchange_ts = self._last_valid_exchange_ts

        if last_valid is not None:
            now_m = time.monotonic()
            age = now_m - last_valid_ts

            if age < self.cfg.fallback_max_age:
                # Validate original data age
                if last_exchange_ts:
                    now_ms = self.time_manager.now_ms()
                    data_age_ms = now_ms - last_exchange_ts

                    if data_age_ms > getattr(self.cfg, 'max_age_ms', ORDERBOOK_MAX_AGE_MS):
                        self._old_data_rejected += 1
                        logging.error(
                            f"❌ Snapshot fallback muito antigo "
                            f"(data_age={data_age_ms}ms > {getattr(self.cfg, 'max_age_ms', ORDERBOOK_MAX_AGE_MS)}ms)"
                        )
                        return None

                self._stale_data_uses += 1
                self._last_fetch_source = "stale"
                self._last_fetch_age_seconds = float(age)

                logging.warning(
                    f"⚠️ Usando snapshot antigo (cache_age={age:.1f}s) - circuit breaker open"
                )
                
                # Metrics for stale data
                self.metrics.inc_fetch(symbol=self.symbol, status="ok", source="stale")
                self.metrics.set_data_age(symbol=self.symbol, source="stale", age_seconds=float(age))

                return self._snapshot_copy(last_valid)
            else:
                logging.error(
                    f"❌ Snapshot muito velho ({age:.1f}s > {self.cfg.fallback_max_age}s)"
                )
        
        return None

    # ===== PATCH 18: Backoff com jitter =====
    def _retry_sleep(self, base: float, attempt: int, cap: float = 10.0) -> float:
        """Backoff exponencial com jitter para evitar thundering herd."""
        delay = min(cap, base * (2 ** attempt))
        jitter = random.uniform(0.0, delay * 0.25)
        return delay + jitter

    # ===== UTILIDADE: MAPA DE PREÇOS PARA ICEBERG (NOVO v2.2.0) =====
    def _price_map(self, levels: List[Tuple[float, float]]) -> Dict[float, float]:
        """
        Converte lista:
            [(price, qty), ...]
        em:
            {price: soma_qty}

        Somando quantidades em níveis duplicados de preço.
        """
        out: Dict[float, float] = {}
        if not levels:
            return out

        for lv in levels:
            try:
                p = float(lv[0])
                q = float(lv[1])
            except Exception:
                continue
            if q <= 0 or p <= 0:
                continue
            out[p] = out.get(p, 0.0) + q
        return out

    # ===== ✅ FETCH ASYNC v2.2.0 (PATCH 2: corrigir create_task) =====
    async def _fetch_orderbook(
        self,
        limit: Optional[int] = None,
        use_cache: bool = True,
        allow_stale: bool = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Busca orderbook com retry, validação e fallback conservador.

        ✅ v2.2.0:
        - Usa ClientSession reutilizável (ou injetada) com lock de criação
        - Usa ORDERBOOK_REQUEST_TIMEOUT para request e wait_for
        - Acesso a cache/fallback protegido por lock (_cache_lock)

        PATCH 2: Corrigido bug do asyncio.create_task(session.get(...))
        PATCH 16: Tratar resposta não-JSON / content-type errado
        PATCH A-02: Não retornar o MESMO objeto armazenado no cache

        Returns:
            Snapshot válido (convertido) ou None
        """
        if allow_stale is None:
            allow_stale = self.cfg.use_fallback

        self._total_fetches += 1

        # 1. CACHE (leitura protegida)
        if use_cache:
            cached_snapshot = None
            cache_ts = 0.0
            with self._cache_lock:
                cached_snapshot = self._cached_snapshot
                cache_ts = self._cache_timestamp_mono

            if cached_snapshot is not None:
                now_m = time.monotonic()
                cache_age = now_m - cache_ts
                if cache_age < self.cache_ttl_seconds:
                    self._cache_hits += 1
                    self._last_fetch_source = "cache"
                    self._last_fetch_age_seconds = float(cache_age)
                    logging.debug(f"📦 Cache hit (age={cache_age:.2f}s)")
                    
                    # ✅ Métricas de cache hit
                    self.metrics.inc_cache_hit(symbol=self.symbol)
                    self.metrics.inc_fetch(symbol=self.symbol, status="ok", source="cache")
                    self.metrics.set_data_age(symbol=self.symbol, source="cache", age_seconds=float(cache_age))
                    
                    return self._snapshot_copy(cached_snapshot)

        # ===== CIRCUIT BREAKER: controlar se vamos tentar LIVE =====
        skip_live_fetch = False
        live_attempted = False  # vamos marcar True quando realmente tentar rede

        if not self._circuit_breaker.allow_request():
            # circuito OPEN: tenta fallback REST primeiro
            self._last_fetch_source = "circuit_open"
            self._last_fetch_age_seconds = 0.0
            skip_live_fetch = True

            # 🆕 FALLBACK ROBUSTO: tenta endpoints alternativos
            fallback = get_fallback_instance()
            if fallback.is_healthy():
                logging.warning(f"🔄 Circuit OPEN para {self.symbol} - tentando fallback REST...")
                try:
                    fallback_data = await fallback.fetch_orderbook_fallback(
                        symbol=self.symbol, 
                        limit=lim,
                        session=await self._get_session()
                    )
                    
                    if fallback_data:
                        # Valida dados do fallback
                        is_valid, issues, converted = self._validate_snapshot(fallback_data)
                        if is_valid:
                            # Salva no cache
                            now_ts_mono = time.monotonic()
                            with self._cache_lock:
                                safe_copy = self._snapshot_copy(converted)
                                self._cached_snapshot = safe_copy
                                self._cache_timestamp_mono = now_ts_mono
    
                                self._last_valid_snapshot = self._snapshot_copy(converted)
                                self._last_valid_timestamp_mono = now_ts_mono
    
                                exchange_ts = converted.get("E") or converted.get("T")
                                if exchange_ts:
                                    self._last_valid_exchange_ts = int(exchange_ts)
                            
                            self._last_fetch_source = "fallback_rest"
                            self._last_fetch_age_seconds = 0.0
                            
                            # Métricas de sucesso do fallback
                            self.metrics.inc_fetch(symbol=self.symbol, status="ok", source="fallback")
                            self.metrics.set_data_age(symbol=self.symbol, source="fallback", age_seconds=0.0)
                            
                            # Registra sucesso no circuit breaker
                            self._circuit_breaker.record_success()
                            
                            logging.info(f"✅ Fallback REST bem-sucedido para {self.symbol}")
                            return self._snapshot_copy(safe_copy)
                        else:
                            logging.warning(f"⚠️ Fallback retornou dados inválidos: {', '.join(issues)}")
                    else:
                        logging.warning(f"⚠️ Fallback REST falhou para {self.symbol}")
                        
                except Exception as e:
                    logging.error(f"💥 Erro no fallback REST para {self.symbol}: {e}")
            else:
                logging.warning(f"🔴 Fallback não está saudável para {self.symbol}")

            logging.warning(f"🔌 Circuit OPEN para {self.symbol} - pulando fetch live")

        # 2. RATE LIMITING
        if not self._check_rate_limit():
            wait_time = max(1.0, self.cfg.retry_delay * 0.5)
            logging.warning(
                f"⏳ Rate limit preventivo - aguardando {wait_time}s..."
            )
            await asyncio.sleep(wait_time)  # ✅ ASYNC

        # 3. FETCH COM RETRY
        lim = limit or self.ob_limit_fetch
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit={lim}"

        max_retries = self.cfg.max_retries
        base_delay = self.cfg.retry_delay

        if not skip_live_fetch:
            for attempt in range(max_retries):
                try:
                    session = await self._get_session()

                    # Verifica se foi fechada externamente
                    if session.closed:
                        logging.warning("⚠️ Sessão fechada, recriando...")
                        async with self._session_lock:
                            self._session = None
                        session = await self._get_session()


                    logging.debug(
                        f"📡 Fetching orderbook (attempt {attempt + 1}/{max_retries})..."
                    )

                    # PATCH 2: async with session.get(...) as r (sem create_task)
                    # ✅ PATCH 2.4: Medir latência manualmente para operações async
                    # 2.4) Registrar live_attempted = True antes de tentar rede
                    live_attempted = True
                    fetch_start = time.perf_counter()
                    client_timeout = aiohttp.ClientTimeout(total=self.cfg.request_timeout)
                    try:
                        async with session.get(url, timeout=client_timeout) as r:
                            fetch_duration = time.perf_counter() - fetch_start
                            self.metrics.observe_latency(symbol=self.symbol, seconds=float(fetch_duration))
                            if r.status == 429:
                                retry_after = int(r.headers.get("Retry-After", 60))
                                self._fetch_errors += 1
                                logging.error(f"🚫 RATE LIMIT (429) - Retry após {retry_after}s")
                                
                                # ✅ PATCH 2.4: Métricas de rate limit
                                self.metrics.inc_fetch(symbol=self.symbol, status="rate_limited", source="live")

                                if attempt < max_retries - 1:
                                    await asyncio.sleep(min(retry_after, base_delay * 3))
                                    continue
                                break

                            if r.status != 200:
                                self._fetch_errors += 1
                                text = await r.text()
                                logging.error(f"❌ HTTP {r.status}: {text[:200]}")
                                
                                # ✅ PATCH 2.4: Métricas de HTTP error
                                self.metrics.inc_fetch(symbol=self.symbol, status=f"http_{r.status}", source="live")
                                if attempt < max_retries - 1:
                                    # PATCH 18: backoff com jitter
                                    await asyncio.sleep(self._retry_sleep(base_delay, attempt))
                                    continue
                                break

                            # PATCH 16: Tratar resposta não-JSON
                            # IMPORTANTE: parsear o JSON ainda dentro do `async with`,
                            # caso contrário a resposta pode ser fechada e gerar "Connection closed".
                            try:
                                data = await r.json()
                            except aiohttp.ContentTypeError:
                                self._fetch_errors += 1
                                text = await r.text()
                                logging.error(f"Resposta não-JSON (status=200): {text[:200]}")
                                
                                # ✅ PATCH 2.4: Métricas de payload inválido
                                self.metrics.inc_fetch(symbol=self.symbol, status="bad_payload", source="live")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(self._retry_sleep(base_delay, attempt))
                                    continue
                                break
                            except Exception as e:
                                self._fetch_errors += 1
                                logging.error(f"Falha ao parsear JSON: {e}")
                                
                                # ✅ PATCH 2.4: Métricas de parse JSON fail
                                self.metrics.inc_fetch(symbol=self.symbol, status="bad_payload", source="live")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(self._retry_sleep(base_delay, attempt))
                                    continue
                                break

                    except asyncio.TimeoutError:
                        self._fetch_errors += 1
                        logging.error(f"⏱️ Timeout (attempt {attempt + 1})")
                        
                        # ✅ PATCH 2.4: Métricas de timeout
                        self.metrics.inc_fetch(symbol=self.symbol, status="timeout", source="live")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(self._retry_sleep(base_delay, attempt))
                        continue

                    # ✅ VALIDA E CONVERTE (síncrono, não bloqueia)
                    is_valid, issues, converted = self._validate_snapshot(data)

                    if not is_valid:
                        self._validation_failures += 1
                        logging.error(
                            f"❌ Snapshot inválido (attempt {attempt + 1}): {', '.join(issues)}"
                        )
                        
                        # ✅ PATCH 2.4: Métricas de validação melhoradas
                        self.metrics.inc_validation_failure(symbol=self.symbol, reason=issues[0] if issues else "invalid_snapshot")
                        self.metrics.inc_fetch(symbol=self.symbol, status="invalid", source="live")
                        
                        if attempt < max_retries - 1:
                            await asyncio.sleep(self._retry_sleep(base_delay, attempt))
                            continue
                        break

                    # ✅ SUCESSO - Salva snapshot CONVERTIDO (protegido por lock)
                    now_ts_mono = time.monotonic()
                    with self._cache_lock:
                        safe_copy = self._snapshot_copy(converted)
                        self._cached_snapshot = safe_copy
                        self._cache_timestamp_mono = now_ts_mono

                        self._last_valid_snapshot = self._snapshot_copy(converted)
                        self._last_valid_timestamp_mono = now_ts_mono

                        # ✅ SALVA TIMESTAMP ORIGINAL
                        exchange_ts = converted.get("E") or converted.get("T")
                        if exchange_ts:
                            self._last_valid_exchange_ts = int(exchange_ts)

                    self._last_fetch_source = "live"
                    self._last_fetch_age_seconds = 0.0

                    # ✅ PATCH 2.4: Métricas de sucesso live
                    self.metrics.inc_fetch(symbol=self.symbol, status="ok", source="live")
                    self.metrics.set_data_age(symbol=self.symbol, source="live", age_seconds=0.0)

                    logging.debug(
                        f"✅ Orderbook obtido: "
                        f"{len(converted['bids'])} bids, {len(converted['asks'])} asks"
                    )

                    # PATCH A-02: Não retornar o MESMO objeto armazenado no cache
                    result = self._snapshot_copy(safe_copy)
                    
                    # 2.5) Registrar sucesso do circuito (record_success)
                    self._circuit_breaker.record_success()
                    
                    return result

                except aiohttp.ClientError as e:
                    self._fetch_errors += 1
                    logging.error(f"🌐 Client error (attempt {attempt + 1}): {e}")
                    # Record circuit breaker failure
                    self._circuit_breaker.record_failure()
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._retry_sleep(base_delay, attempt))

                except Exception as e:
                    self._fetch_errors += 1
                    logging.error(f"💥 Unexpected error (attempt {attempt + 1}): {e}", exc_info=True)
                    # Record circuit breaker failure
                    self._circuit_breaker.record_failure()
                    if attempt < max_retries - 1:
                        await asyncio.sleep(self._retry_sleep(base_delay, attempt))

        # Registrar falha do circuito (record_failure) — 1 vez por chamada
        # Se tentamos rede (live) e não retornamos sucesso, marca falha 1x
        if live_attempted:
            self._circuit_breaker.record_failure()

        # 5. ✅ FALLBACK CONSERVADOR (leitura protegida)
        if allow_stale:
            last_valid = None
            last_valid_ts = 0.0
            last_exchange_ts = None

            with self._cache_lock:
                last_valid = self._last_valid_snapshot
                last_valid_ts = self._last_valid_timestamp_mono
                last_exchange_ts = self._last_valid_exchange_ts

            if last_valid is not None:
                now_m = time.monotonic()
                age = now_m - last_valid_ts

                if age < self.cfg.fallback_max_age:
                    # ✅ VALIDA IDADE DO DADO ORIGINAL
                    if last_exchange_ts:
                        now_ms = self.time_manager.now_ms()
                        data_age_ms = now_ms - last_exchange_ts

                        if data_age_ms > getattr(self.cfg, 'max_age_ms', ORDERBOOK_MAX_AGE_MS):
                            self._old_data_rejected += 1
                            logging.error(
                                f"❌ Snapshot fallback muito antigo "
                                f"(data_age={data_age_ms}ms > {getattr(self.cfg, 'max_age_ms', ORDERBOOK_MAX_AGE_MS)}ms)"
                            )
                            return None

                    self._stale_data_uses += 1
                    self._last_fetch_source = "stale"
                    self._last_fetch_age_seconds = float(age)

                    logging.warning(
                        f"⚠️ Usando snapshot antigo (cache_age={age:.1f}s)"
                    )
                    
                    # ✅ PATCH 2.4: Métricas de stale data
                    self.metrics.inc_fetch(symbol=self.symbol, status="ok", source="stale")
                    self.metrics.set_data_age(symbol=self.symbol, source="stale", age_seconds=float(age))

                    return self._snapshot_copy(last_valid)
                else:
                    logging.error(
                        f"❌ Snapshot muito velho ({age:.1f}s > {self.cfg.fallback_max_age}s)"
                    )

        # 💀 FALHA TOTAL
        error_rate = 100 * self._fetch_errors / max(1, self._total_fetches)

        logging.error(
            f"💀 FALHA ao obter orderbook após {max_retries} tentativas "
            f"(erro: {error_rate:.1f}%)"
        )
        
        # ✅ PATCH 2.4: Métricas de falha total
        self.metrics.inc_fetch(symbol=self.symbol, status="failed", source="live")

        return None

    # ===== MÉTRICAS (NÃO MUDARAM, COM AJUSTES INTERNOS) =====
    def _spread_and_depth(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """Calcula spread e profundidade."""
        if not bids or not asks:
            return {
                "mid": None,
                "spread": None,
                "spread_percent": None,
                "bid_depth_usd": 0.0,
                "ask_depth_usd": 0.0,
            }

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else None
        spread = best_ask - best_bid if (best_ask and best_bid) else None

        if spread is not None and mid and mid > 0:
            spread_pct = (spread / mid) * 100.0
        else:
            spread_pct = None

        bid_depth_usd = _sum_depth_usd(bids, self.top_n)
        ask_depth_usd = _sum_depth_usd(asks, self.top_n)

        return {
            "mid": mid,
            "spread": round(spread, 8) if spread is not None else None,
            "spread_percent": round(spread_pct, 6) if spread_pct is not None else None,
            "bid_depth_usd": round(bid_depth_usd, 2),
            "ask_depth_usd": round(ask_depth_usd, 2),
        }

    def _imbalance_ratio_pressure(
        self,
        bid_usd: float,
        ask_usd: float
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calcula imbalance, ratio e pressure.

        ✅ v1.6.0: Proteção total contra divisão por zero
        """
        # ✅ REJEITA SE ALGUM LADO É ZERO
        if bid_usd <= 0 or ask_usd <= 0:
            logging.warning(
                f"⚠️ Dados parciais rejeitados: bid=${bid_usd:.2f}, ask=${ask_usd:.2f}"
            )
            return None, None, None

        total = bid_usd + ask_usd
        if total <= 0:
            return None, None, None

        imbalance = (bid_usd - ask_usd) / total
        ratio = bid_usd / ask_usd
        pressure = imbalance

        return float(imbalance), float(ratio), float(pressure)

    # ===== PATCH 8: Volume-weighted imbalance =====
    def _weighted_imbalance(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        *,
        weights: Optional[List[float]] = None,
        use_notional: bool = True,
    ) -> float:
        """Mede pressão "perto do topo" ponderando níveis."""
        if weights is None:
            weights = list(THRESHOLDS.WEIGHTED_IMBALANCE_WEIGHTS)

        n = len(weights)
        b = bids[:n]
        a = asks[:n]
        if not b or not a:
            return 0.0

        def val(p: float, q: float) -> float:
            return (p * q) if use_notional else q

        bid_w = 0.0
        ask_w = 0.0
        for i, (p, q) in enumerate(b):
            if p > 0 and q > 0:
                bid_w += val(p, q) * weights[i]
        for i, (p, q) in enumerate(a):
            if p > 0 and q > 0:
                ask_w += val(p, q) * weights[i]

        denom = bid_w + ask_w
        if denom <= 0:
            return 0.0
        return (bid_w - ask_w) / denom

    def _detect_walls(
        self,
        side_levels: List[Tuple[float, float]],
        side: str
    ) -> List[Dict[str, Any]]:
        """
        Detecta paredes de liquidez.

        ✅ v2.2.0: usa quantil 90% dos volumes como base, multiplicado por self.wall_std,
                   reduzindo falsos positivos em relação a média+desvio.
        PATCH 7: usa dynamic threshold multiplier
        Mantém retorno compatível: lista de dicts com side/price/qty/limit_threshold.
        """
        if not side_levels:
            return []

        levels = side_levels[:self.top_n]
        qtys = np.array([q for _, q in levels if q > 0], dtype=float)

        if qtys.size == 0:
            return []

        try:
            if qtys.size >= 3:
                base = float(np.quantile(qtys, 0.90))
            else:
                base = float(qtys.max())
            # PATCH 7: dynamic wall multiplier
            mult = float(self.dynamic_thresholds.get("wall_threshold_multiplier", self.wall_std))
            threshold = base * max(1.0, mult)  # fator multiplicativo
        except Exception:
            mean = float(np.mean(qtys))
            std = float(np.std(qtys))
            mult = float(self.dynamic_thresholds.get("wall_threshold_multiplier", self.wall_std))
            threshold = mean * 1.5 if std <= 1e-12 else mean + mult * std

        walls: List[Dict[str, Any]] = []
        for p, q in levels:
            if q >= threshold and q > 0:
                walls.append({
                    "side": side,
                    "price": float(p),
                    "qty": float(q),
                    "limit_threshold": float(threshold),
                })

        walls.sort(key=lambda x: x["price"], reverse=(side == "bid"))
        return walls

    def _check_liquidity_flow(self, current_snapshot,
                             previous_snapshot) -> Dict[str, Any]:
        """Detecta fluxo de liquidez entre snapshots."""
        if previous_snapshot is None or current_snapshot is None:
            return {"has_alert": False, "flow_percentage": 0.0}
        
        # Calcula volume total para cada lado
        current_bid_volume = sum(qty for _, qty in current_snapshot.bids[:5])
        current_ask_volume = sum(qty for _, qty in current_snapshot.asks[:5])
        previous_bid_volume = sum(qty for _, qty in previous_snapshot.bids[:5])
        previous_ask_volume = sum(qty for _, qty in previous_snapshot.asks[:5])
        
        # Calcula mudanças percentuais
        bid_change = ((current_bid_volume - previous_bid_volume) / previous_bid_volume
                      if previous_bid_volume > 0 else 0)
        ask_change = ((current_ask_volume - previous_ask_volume) / previous_ask_volume
                      if previous_ask_volume > 0 else 0)
        
        # Determina se há alerta
        has_alert = (abs(bid_change) >= self.liquidity_flow_alert_percentage or
                     abs(ask_change) >= self.liquidity_flow_alert_percentage)
        
        return {
            "has_alert": has_alert,
            "bid_flow": bid_change,
            "ask_flow": ask_change,
            "flow_percentage": max(abs(bid_change), abs(ask_change))
        }

    def _compute_core_metrics(self, snapshot) -> Dict[str, Any]:
        """Wrapper para _calculate_metrics (compatibilidade com testes)."""
        return self._calculate_metrics(snapshot)

    def analyze_orderbook(self, snapshot) -> Dict[str, Any]:
        """Versão síncrona para compatibilidade com testes."""
        return self.analyze(snapshot)

    def _iceberg_reload(
        self,
        prev: Optional[Dict[str, Any]],
        curr: Dict[str, Any],
        tol: float = 0.75
    ) -> Tuple[bool, float]:
        """
        Detecta possível recarga de ordens iceberg.

        ✅ v2.2.0:
        - Usa _price_map para somar níveis no mesmo preço (corrige falso/negativo e falso/positivo)
        - Compara quantidades por preço entre prev e curr
        - Score baseado em deltas relativos; bool indica se score > 0.5

        PATCH 7: usa dynamic iceberg tolerance

        Mantém assinatura (bool, score) para compatibilidade.
        """
        try:
            if not prev:
                return False, 0.0

            prev_bids_map = self._price_map(prev.get("bids", []))
            prev_asks_map = self._price_map(prev.get("asks", []))
            curr_bids_map = self._price_map(curr.get("bids", []))
            curr_asks_map = self._price_map(curr.get("asks", []))

            score = 0.0

            for side_label, pm_prev, pm_curr in [
                ("bid", prev_bids_map, curr_bids_map),
                ("ask", prev_asks_map, curr_asks_map),
            ]:
                if not pm_prev or not pm_curr:
                    continue

                for price, qty_now in pm_curr.items():
                    qty_prev = pm_prev.get(price, 0.0)
                    if qty_prev <= 0:
                        continue

                    delta = qty_now - qty_prev
                    # threshold absoluto de recarga + fator relativo
                    min_delta = THRESHOLDS.ICEBERG_RELOAD_MIN_DELTA
                    score_threshold = THRESHOLDS.ICEBERG_SCORE_THRESHOLD

                    if delta >= min_delta and qty_now >= tol * max(qty_prev, 1e-9):
                        contrib = min(1.0, delta / max(qty_prev, 1e-9))
                        score += contrib

            return (score > score_threshold), float(round(score, 4))

        except Exception:
            return False, 0.0

    # ===== PATCH 9: Liquidity Concentration Index =====
    def _liquidity_concentration(
        self,
        levels: List[Tuple[float, float]],
        *,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Mede se a liquidez está "espalhada" ou concentrada em poucos níveis."""
        lv = levels[:max(1, int(top_n))]
        notionals = [p * q for (p, q) in lv if p > 0 and q > 0]
        total = float(sum(notionals))
        if total <= 0:
            return {"top_n": top_n, "total_usd": 0.0, "hhi": None, "top1_share": None, "top3_share": None}

        shares = [x / total for x in notionals]
        shares_sorted = sorted(shares, reverse=True)

        hhi = float(sum(s * s for s in shares))  # 0..1
        top1 = float(shares_sorted[0]) if shares_sorted else None
        top3 = float(sum(shares_sorted[:3])) if shares_sorted else None

        return {
            "top_n": int(top_n),
            "total_usd": round(total, 2),
            "hhi": round(hhi, 6),
            "top1_share": round(top1, 6) if top1 is not None else None,
            "top3_share": round(top3, 6) if top3 is not None else None,
        }

    # ===== PATCH 10: Microstructure metrics =====
    def _microstructure_metrics(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        *,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Adiciona métricas de microestrutura sem depender de trades."""
        if not bids or not asks:
            return {}

        top_n = max(1, int(top_n))
        b = bids[:top_n]
        a = asks[:top_n]

        best_bid = b[0][0]
        best_ask = a[0][0]
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else None
        spread_bps = ((best_ask - best_bid) / mid) * 10000.0 if mid and mid > 0 else None

        def vwap(levels: List[Tuple[float, float]]) -> Optional[float]:
            qty = 0.0
            numer = 0.0
            for p, q in levels:
                if p > 0 and q > 0:
                    qty += q
                    numer += p * q
            return (numer / qty) if qty > 0 else None

        def avg_distance_bps(levels: List[Tuple[float, float]], ref: float, side: str) -> Optional[float]:
            if ref <= 0:
                return None
            total = 0.0
            numer = 0.0
            for p, q in levels:
                if p <= 0 or q <= 0:
                    continue
                notional = p * q
                dist_bps = ((ref - p) / ref) * 10000.0 if side == "bid" else ((p - ref) / ref) * 10000.0
                numer += dist_bps * notional
                total += notional
            return (numer / total) if total > 0 else None

        vwap_bid = vwap(b)
        vwap_ask = vwap(a)

        return {
            "mid": mid,
            "spread_bps": round(spread_bps, 4) if spread_bps is not None else None,
            "vwap_bid_topn": round(vwap_bid, 8) if vwap_bid is not None else None,
            "vwap_ask_topn": round(vwap_ask, 8) if vwap_ask is not None else None,
            "bid_avg_distance_bps_topn": round(avg_distance_bps(b, best_bid, "bid"), 4),
            "ask_avg_distance_bps_topn": round(avg_distance_bps(a, best_ask, "ask"), 4),
        }

    # ===== PATCH 11: Anomaly detection =====
    def _detect_anomalies(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        prev_snapshot: Optional[Dict[str, Any]],
        *,
        spread_jump_bps: float = 30.0,
        depth_drop_pct: float = 60.0,
    ) -> List[str]:
        """Sinaliza mudanças bruscas de microestrutura."""
        out: List[str] = []
        if not prev_snapshot:
            return out

        prev_bids = prev_snapshot.get("bids") or []
        prev_asks = prev_snapshot.get("asks") or []
        if not prev_bids or not prev_asks or not bids or not asks:
            return out

        cur_sm = self._spread_and_depth(bids, asks)
        prev_sm = self._spread_and_depth(prev_bids, prev_asks)

        cur_bps = float(cur_sm["spread_percent"]) * 100.0 if cur_sm.get("spread_percent") is not None else None
        prev_bps = float(prev_sm["spread_percent"]) * 100.0 if prev_sm.get("spread_percent") is not None else None
        if cur_bps is not None and prev_bps is not None and (cur_bps - prev_bps) >= spread_jump_bps:
            out.append(f"spread_jump: {prev_bps:.2f}bps -> {cur_bps:.2f}bps")

        cur_bid = float(cur_sm.get("bid_depth_usd") or 0.0)
        cur_ask = float(cur_sm.get("ask_depth_usd") or 0.0)
        prev_bid = float(prev_sm.get("bid_depth_usd") or 0.0)
        prev_ask = float(prev_sm.get("ask_depth_usd") or 0.0)

        def drop(p: float, c: float) -> Optional[float]:
            if p <= 0:
                return None
            return (1.0 - (c / p)) * 100.0

        bid_drop = drop(prev_bid, cur_bid)
        ask_drop = drop(prev_ask, cur_ask)

        if bid_drop is not None and bid_drop >= depth_drop_pct:
            out.append(f"bid_depth_drop: {bid_drop:.1f}%")
        if ask_drop is not None and ask_drop >= depth_drop_pct:
            out.append(f"ask_depth_drop: {ask_drop:.1f}%")

        return out

    # ===== EVENTO INVÁLIDO (PATCH 6.2: usa EventFactory) =====
    def _create_invalid_event(
        self,
        error_msg: str,
        ts_ms: Optional[int] = None,
        severity: str = "ERROR",
    ) -> Dict[str, Any]:
        """Cria evento marcado como INVÁLIDO usando EventFactory (PATCH 6.2)."""
        if ts_ms is None:
            ts_ms = self.time_manager.now_ms()

        tindex = self.time_manager.build_time_index(ts_ms, include_local=True, timespec="seconds")

        thresholds = {
            "ORDERBOOK_CRITICAL_IMBALANCE": self.cfg.critical_imbalance,
            "ORDERBOOK_MIN_DOMINANT_USD": self.cfg.min_dominant_usd,
            "ORDERBOOK_MIN_RATIO_DOM": self.cfg.min_ratio_dom,
        }

        # LOG ESTRUTURADO: evento inválido
        try:
            self.slog.error(
                "orderbook_invalid",
                error=error_msg,
                severity=severity,
                fetch_errors=self._fetch_errors,
                validation_failures=self._validation_failures,
                data_source=getattr(self, "_last_fetch_source", "unknown"),
            )
        except Exception:
            # logging não deve nunca quebrar o fluxo
            pass

        return build_invalid_orderbook_event(
            symbol=self.symbol,
            schema_version=SCHEMA_VERSION,
            engine_version=ENGINE_VERSION,
            ts_ms=int(ts_ms),
            error_msg=error_msg,
            severity=severity,
            timestamp_ny=tindex.get("timestamp_ny"),
            timestamp_utc=tindex.get("timestamp_utc"),
            top_n=self.top_n,
            ob_limit=self.ob_limit_fetch,
            thresholds=thresholds,
            health_stats=self.get_stats(),
        )

    # ===== REFACTORED METHODS FOR BETTER CODE ORGANIZATION =====
    
    async def _acquire_snapshot(
        self,
        current_snapshot: Optional[Dict[str, Any]],
        event_epoch_ms: Optional[int],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Obtém snapshot já convertido/validado.

        - Se current_snapshot foi fornecido: valida, converte e marca data_source=external.
        - Caso contrário: chama _fetch_orderbook (que já retorna convertido).
        """
        if current_snapshot is not None:
            is_valid, issues, converted = self._validate_snapshot(current_snapshot)
            if not is_valid:
                return None, f"validation_failed: {', '.join(issues)}"

            # data_source correto para snapshot externo (Patch A-01 já aplicado no seu código)
            self._last_fetch_source = "external"
            try:
                snap_ts = None
                for k in ("E", "T"):
                    v = current_snapshot.get(k)
                    if isinstance(v, (int, float)) and v > 0:
                        snap_ts = int(v)
                        break
                if snap_ts:
                    self._last_fetch_age_seconds = max(0.0, (self.time_manager.now_ms() - snap_ts) / 1000.0)
                else:
                    self._last_fetch_age_seconds = 0.0
            except Exception:
                self._last_fetch_age_seconds = 0.0

            return converted, None

        try:
            snap = await self._fetch_orderbook(limit=self.ob_limit_fetch)
        except Exception as e:
            logger.error(f"Erro em operação async: {e}")
            raise
        if not snap:
            return None, "fetch_failed"
        return snap, None

    def _extract_book_data(
        self,
        snap: Dict[str, Any],
        event_epoch_ms: Optional[int],
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], int, Dict[str, Any]]:
        """
        Extrai bids/asks e timestamp principal do evento.
        Retorna também o time_index (ny/utc).
        """
        bids: List[Tuple[float, float]] = snap.get("bids", []) or []
        asks: List[Tuple[float, float]] = snap.get("asks", []) or []

        ts_ms = None
        for key in ("E", "T"):
            v = snap.get(key)
            if isinstance(v, (int, float)) and v > 0:
                ts_ms = int(v)
                break
        if ts_ms is None:
            ts_ms = event_epoch_ms if event_epoch_ms is not None else self.time_manager.now_ms()

        tindex = self.time_manager.build_time_index(ts_ms, include_local=True, timespec="seconds")
        return bids, asks, ts_ms, tindex

    def _compute_core_metrics(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Calcula métricas core. Se dados parciais/invalidáveis, retorna error string.
        """
        sm = self._spread_and_depth(bids, asks)
        mid = sm.get("mid")
        bid_usd = float(sm.get("bid_depth_usd") or 0.0)
        ask_usd = float(sm.get("ask_depth_usd") or 0.0)

        imbalance, ratio, pressure = self._imbalance_ratio_pressure(bid_usd, ask_usd)
        if imbalance is None:
            return None, f"partial_data_rejected: bid=${bid_usd:.2f}, ask=${ask_usd:.2f}"

        spread_bps = None
        if sm.get("spread_percent") is not None:
            try:
                spread_bps = float(sm["spread_percent"]) * 100.0
            except Exception:
                spread_bps = None

        core = {
            "spread_metrics": sm,
            "mid": mid,
            "bid_usd": bid_usd,
            "ask_usd": ask_usd,
            "imbalance": float(imbalance),
            "ratio": float(ratio) if ratio is not None else None,
            "pressure": float(pressure),
            "spread_bps": spread_bps,
        }
        return core, None

    def _compute_iceberg(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> Tuple[bool, float]:
        iceberg_tol = float(self.dynamic_thresholds.get("iceberg_tolerance", 0.75))
        return self._iceberg_reload(
            self.prev_snapshot,
            {"bids": bids, "asks": asks},
            tol=iceberg_tol,
        )


    def _compute_market_impact(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        mid: Optional[float],
    ) -> Dict[str, Any]:
        mi_buy_100k = _simulate_market_impact(asks[:self.top_n], 100_000.0, "buy", mid)
        mi_buy_1m = _simulate_market_impact(asks[:self.top_n], 1_000_000.0, "buy", mid)
        mi_sell_100k = _simulate_market_impact(bids[:self.top_n], 100_000.0, "sell", mid)
        mi_sell_1m = _simulate_market_impact(bids[:self.top_n], 1_000_000.0, "sell", mid)
        return {
            "buy": {"100k": mi_buy_100k, "1M": mi_buy_1m},
            "sell": {"100k": mi_sell_100k, "1M": mi_sell_1m},
        }


    def _build_depth_summary(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        depth_summary: Dict[str, Any] = {}
        total_bids_last = 0.0
        total_asks_last = 0.0

        for lvl in self.depth_levels:
            try:
                b_usd = _sum_depth_usd(bids, lvl)
                a_usd = _sum_depth_usd(asks, lvl)
                imbalance_level = None
                denom = b_usd + a_usd
                if denom > 0:
                    imbalance_level = (b_usd - a_usd) / denom

                depth_summary[f"L{lvl}"] = {
                    "bids": round(b_usd, 2),
                    "asks": round(a_usd, 2),
                    "imbalance": round(imbalance_level, 4) if imbalance_level is not None else None,
                }
                total_bids_last = b_usd
                total_asks_last = a_usd
            except Exception:
                depth_summary[f"L{lvl}"] = {"bids": None, "asks": None, "imbalance": None}

        total_ratio = None
        try:
            if total_asks_last > 0:
                total_ratio = total_bids_last / total_asks_last
        except Exception:
            pass

        depth_summary["total_depth_ratio"] = round(total_ratio, 3) if total_ratio is not None else None
        return depth_summary

    def _build_spread_analysis(
        self,
        ts_ms: int,
        spread_bps: Optional[float],
        sm: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Atualiza spread_history (append + prune) e calcula spread_analysis.
        Mantém a lógica atual, só encapsula.
        """
        # Atualiza histórico
        if spread_bps is not None and spread_bps >= 0:
            try:
                self.spread_history.append((int(ts_ms), float(spread_bps)))
            except Exception:
                pass

        try:
            cutoff_ms = ts_ms - max(self.spread_avg_windows_min) * 60 * 1000
            self.spread_history = [(t, s) for (t, s) in self.spread_history if t >= cutoff_ms]
        except Exception:
            pass

        spread_analysis: Dict[str, Any] = {
            "current_spread_bps": round(spread_bps, 4) if spread_bps is not None else None,
            "spread_percentile": None,
            "tight_spread_duration_min": None,
            "spread_volatility": None,
        }

        try:
            for window_min in self.spread_avg_windows_min:
                window_ms = window_min * 60 * 1000
                values = [s for (t, s) in self.spread_history if (ts_ms - t) <= window_ms]
                avg = float(np.mean(values)) if values else None

                if window_min >= 60 and window_min % 60 == 0:
                    hours = window_min // 60
                    key = f"avg_spread_{hours}h"
                else:
                    key = f"avg_spread_{window_min}m"

                spread_analysis[key] = round(avg, 4) if avg is not None else None

            if spread_bps is not None:
                all_values = [s for (_, s) in self.spread_history]
                if all_values:
                    sorted_vals = sorted(all_values)
                    less = sum(1 for v in sorted_vals if v < spread_bps)
                    pct = (less / len(sorted_vals)) * 100.0
                    spread_analysis["spread_percentile"] = round(pct, 1)
                    spread_analysis["spread_volatility"] = round(float(np.std(sorted_vals)), 4)

            if spread_bps is not None:
                duration_ms = 0
                threshold = self.spread_tight_threshold_bps
                for (t, s) in reversed(self.spread_history):
                    if s <= threshold:
                        duration_ms = ts_ms - t
                    else:
                        break
                spread_analysis["tight_spread_duration_min"] = round(duration_ms / 60000.0, 2) if duration_ms else 0.0

        except Exception as e:
            logging.debug(f"Erro em spread_analysis: {e}")

        return spread_analysis

    def _build_labels_and_alerts(
        self,
        imbalance: float,
        iceberg: bool,
        spread_bps: Optional[float],
        ratio: Optional[float],
        bid_usd: float,
        ask_usd: float,
    ) -> Tuple[str, List[str], bool, Dict[str, Any]]:
        """
        Define resultado_da_batalha, alertas, criticidade e critical_flags.
        Mantém regras atuais.
        """
        # Resultado
        resultado_da_batalha = "Equilíbrio"
        if imbalance > self.alert_threshold:
            resultado_da_batalha = "Demanda no Livro (Bid>Ask)"
        elif imbalance < -self.alert_threshold:
            resultado_da_batalha = "Oferta no Livro (Ask>Bid)"
        else:
            if imbalance > 0.0:
                resultado_da_batalha = "Leve Demanda no Livro"
            elif imbalance < 0.0:
                resultado_da_batalha = "Leve Oferta no Livro"

        alertas: List[str] = []
        if abs(imbalance) >= self.alert_threshold:
            alertas.append("Alerta de Liquidez (desequilíbrio)")
        if iceberg:
            alertas.append("Iceberg possivelmente recarregando")
        if spread_bps is not None and spread_bps <= self.spread_tight_threshold_bps:
            alertas.append("Spread apertado")

        # Criticidade - 🆕 AGORA REQUER SPREAD ALTO ALÉM DO IMBALANCE
        ratio_dom = None
        if ratio is not None:
            if ratio > 0:
                ratio_dom = ratio if ratio >= 1.0 else (1.0 / ratio)
            else:
                ratio_dom = float("inf")

        dominant_usd = max(bid_usd, ask_usd)
        is_extreme_imbalance = abs(imbalance) >= self.cfg.critical_imbalance
        is_extreme_ratio = (ratio_dom is not None) and (ratio_dom >= self.cfg.min_ratio_dom)
        is_extreme_usd = dominant_usd >= self.cfg.min_dominant_usd
        
        # 🆕 Spread largo indica volatilidade extrema (spread >= 50 bps)
        is_wide_spread = spread_bps is not None and spread_bps >= 50.0
        
        # 🆕 CRITICAL só é ativado se: imbalance extremo E (spread largo OU volume extremo)
        is_critical = bool(
            is_extreme_imbalance and (is_wide_spread or is_extreme_ratio or is_extreme_usd)
        )
        
        # 🆕 EMERGENCY MODE: imbalance muito extremo (>= 0.9) sem spread largo
        # Neste caso, o sistema continua operando mas com peso reduzido da IA
        is_emergency = bool(
            abs(imbalance) >= 0.9 and not is_wide_spread
        )
        
        if is_critical:
            side_dom = "ASKS" if imbalance < 0 else "BIDS"
            alertas.append(f"🔴 DESEQUILÍBRIO CRÍTICO ({side_dom})")
        elif is_emergency:
            side_dom = "ASKS" if imbalance < 0 else "BIDS"
            alertas.append(f"⚠️ EMERGENCY MODE - Desequilíbrio extremo ({side_dom})")

        critical_flags = {
            "is_critical": is_critical,
            "is_emergency": is_emergency,  # 🆕 Emergency mode para imbalance extremo sem spread largo
            "abs_imbalance": round(abs(imbalance), 4),
            "ratio_dom": (round(ratio_dom, 4) if (ratio_dom not in (None, float("inf"))) else ratio_dom),
            "dominant_usd": round(dominant_usd, 2),
            "spread_bps": round(spread_bps, 4) if spread_bps is not None else None,
            "wide_spread": is_wide_spread,  # 🆕 Spread largo detectado
            "thresholds": {
                "ORDERBOOK_CRITICAL_IMBALANCE": self.cfg.critical_imbalance,
                "ORDERBOOK_MIN_DOMINANT_USD": self.cfg.min_dominant_usd,
                "ORDERBOOK_MIN_RATIO_DOM": self.cfg.min_ratio_dom,
                "WIDE_SPREAD_THRESHOLD_BPS": 50.0,  # 🆕 Threshold para spread largo
            },
        }


        return resultado_da_batalha, alertas, is_critical, critical_flags

    def _build_description(
        self,
        imbalance: float,
        ratio: Optional[float],
        bid_usd: float,
        ask_usd: float,
    ) -> str:
        # Mantém exatamente o padrão de texto atual
        if ratio is None:
            ratio = 1.0
        if imbalance < -0.05:
            return f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"
        elif imbalance > 0.05:
            return f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"
        return f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"

    def _get_data_quality(self) -> Tuple[str, float]:
        data_source = getattr(self, "_last_fetch_source", "unknown")
        age_seconds = float(getattr(self, "_last_fetch_age_seconds", 0.0))
        return data_source, age_seconds

    def _update_internal_state(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]], ts_ms: int) -> None:
        self.prev_snapshot = {"bids": bids, "asks": asks}
        self.last_event_ts_ms = ts_ms
        try:
            self._history.append({"bids": bids, "asks": asks})
        except Exception:
            pass

    def _export_metrics_from_event(self, event: Dict[str, Any]) -> None:
        """
        Publica gauges/idade se a Etapa 2 estiver instalada.
        Não quebra se self.metrics não existir.
        """
        try:
            metrics = getattr(self, "metrics", None)
            if metrics is None:
                return

            ob = event.get("orderbook_data") or {}
            sm = event.get("spread_metrics") or {}
            dq = event.get("data_quality") or {}

            spread_bps = None
            sp = ob.get("spread_percent", sm.get("spread_percent"))
            if sp is not None:
                spread_bps = float(sp) * 100.0

            metrics.set_core_gauges(
                symbol=self.symbol,
                imbalance=float(ob.get("imbalance")),
                bid_depth_usd=float(ob.get("bid_depth_usd")),
                ask_depth_usd=float(ob.get("ask_depth_usd")),
                spread_bps=spread_bps,
            )


            source = str(dq.get("data_source", "unknown"))
            age = float(dq.get("age_seconds", 0.0))
            metrics.set_data_age(symbol=self.symbol, source=source, age_seconds=age)
        except Exception:
            # observabilidade não pode derrubar pipeline
            return

    # ===== MISSING METHODS FOR TESTS =====
    def _calculate_metrics(self, snapshot):
        """
        Calculate metrics from OrderBookSnapshot or dict.
        Returns metrics dict compatible with tests.
        """
        try:
            # Handle OrderBookSnapshot object
            if hasattr(snapshot, 'bids') and hasattr(snapshot, 'asks'):
                bids = snapshot.bids if isinstance(snapshot.bids, list) else []
                asks = snapshot.asks if isinstance(snapshot.asks, list) else []
            # Handle dict format
            elif isinstance(snapshot, dict):
                bids = snapshot.get('bids', [])
                asks = snapshot.get('asks', [])
            else:
                return {
                    "best_bid": None,
                    "best_ask": None,
                    "spread": None,
                    "mid_price": None,
                    "imbalance_10": None
                }

            # Calculate basic metrics
            best_bid = bids[0][0] if bids else None
            best_ask = asks[0][0] if asks else None
            
            spread = None
            mid_price = None
            if best_bid and best_ask:
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2
            
            # Calculate imbalance for top 10 levels
            imbalance_10 = None
            if bids and asks:
                bid_depth_10 = sum(float(b[1]) for b in bids[:10] if len(b) >= 2)
                ask_depth_10 = sum(float(a[1]) for a in asks[:10] if len(a) >= 2)
                if bid_depth_10 + ask_depth_10 > 0:
                    imbalance_10 = (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10)
            
            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "mid_price": mid_price,
                "imbalance_10": imbalance_10
            }
        except Exception:
            return {
                "best_bid": None,
                "best_ask": None,
                "spread": None,
                "mid_price": None,
                "imbalance_10": None
            }

    def _detect_liquidity_walls(self, snapshot, levels=10):
        """
        Detect liquidity walls from OrderBookSnapshot or dict.
        Wrapper for existing _detect_walls method.
        """
        try:
            # Handle OrderBookSnapshot object
            if hasattr(snapshot, 'bids') and hasattr(snapshot, 'asks'):
                bids = snapshot.bids if isinstance(snapshot.bids, list) else []
                asks = snapshot.asks if isinstance(snapshot.asks, list) else []
            # Handle dict format
            elif isinstance(snapshot, dict):
                bids = snapshot.get('bids', [])
                asks = snapshot.get('asks', [])
            else:
                return []

            # Convert to format expected by _detect_walls
            bid_walls = self._detect_walls(bids, side="bid")
            ask_walls = self._detect_walls(asks, side="ask")
            
            return bid_walls + ask_walls
        except Exception:
            return []

    def _check_critical_imbalance(self, snapshot):
        """
        Check for critical imbalance in OrderBookSnapshot or dict.
        """
        try:
            # Handle OrderBookSnapshot object
            if hasattr(snapshot, 'bids') and hasattr(snapshot, 'asks'):
                bids = snapshot.bids if isinstance(snapshot.bids, list) else []
                asks = snapshot.asks if isinstance(snapshot.asks, list) else []
            # Handle dict format
            elif isinstance(snapshot, dict):
                bids = snapshot.get('bids', [])
                asks = snapshot.get('asks', [])
            else:
                return {"is_critical": False}

            # Calculate imbalance
            bid_depth = sum(float(b[1]) for b in bids if len(b) >= 2)
            ask_depth = sum(float(a[1]) for a in asks if len(a) >= 2)
            
            if bid_depth + ask_depth == 0:
                return {"is_critical": False}
                
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            
            # Check if meets critical threshold
            is_critical = abs(imbalance) >= self.cfg.critical_imbalance
            
            return {
                "is_critical": is_critical,
                "imbalance": imbalance,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "threshold": self.cfg.critical_imbalance
            }
        except Exception:
            return {"is_critical": False}

    def _add_to_cache(self, key, snapshot):
        """
        Add snapshot to cache (simple implementation for tests).
        """
        try:
            self._cached_snapshot = snapshot
            self._cache_timestamp_mono = time.monotonic()
        except Exception:
            pass

    def _get_from_cache(self, key):
        """
        Get snapshot from cache (simple implementation for tests).
        """
        try:
            if self._cached_snapshot is None:
                return None
                
            cache_age = time.monotonic() - self._cache_timestamp_mono
            if cache_age < self.cache_ttl_seconds:
                return self._cached_snapshot
            else:
                return None
        except Exception:
            return None

    def _reset_rate_limit(self):
        """Reset rate limiter."""
        if hasattr(self, '_rate_limit_timestamps'):
            self._rate_limit_timestamps.clear()

    def _generate_cache_key(self, snapshot):
        """
        Generate a cache key for a snapshot.
        """
        if snapshot is None:
            return None
            
        try:
            # Handle OrderBookSnapshot object
            if hasattr(snapshot, 'symbol') and hasattr(snapshot, 'last_update_id'):
                symbol = snapshot.symbol if hasattr(snapshot, 'symbol') else self.symbol
                update_id = snapshot.last_update_id if hasattr(snapshot, 'last_update_id') else 'unknown'
                timestamp = snapshot.timestamp if hasattr(snapshot, 'timestamp') else time.time()
            # Handle dict format
            elif isinstance(snapshot, dict):
                symbol = snapshot.get('symbol', self.symbol)
                update_id = snapshot.get('lastUpdateId', snapshot.get('E', 'unknown'))
                timestamp = snapshot.get('E', snapshot.get('T', time.time()))
            else:
                return None
                
            return f"{symbol}_cache_{update_id}_{int(timestamp)}"
        except Exception:
            pass

    def get_health_status(self):
        """
        Get health status of the analyzer.
        """
        stats = self.get_stats()
        
        return {
            'status': 'healthy' if stats['error_rate_pct'] < 10 else 'degraded',
            'cache_size': 1 if self._cached_snapshot else 0,
            'cache_hit_rate': stats['cache_hit_rate_pct'] / 100.0,
            'rate_limit_status': 'ok' if stats['requests_last_min'] < self.rate_limit_threshold else 'exceeded',
            'last_analysis_time': None,
            'error_rate': stats['error_rate_pct'],
            'validation_failure_rate': stats['validation_failure_rate_pct']
        }

    def clear_cache(self):
        """
        Clear the cache completely.
        """
        try:
            self._cached_snapshot = None
            self._cache_timestamp_mono = 0.0
        except Exception:
            pass

    def _check_liquidity_flow(self, current_snapshot, previous_snapshot):
        """
        Check liquidity flow between snapshots.
        """
        try:
            # Handle OrderBookSnapshot objects
            if hasattr(current_snapshot, 'bids') and hasattr(current_snapshot, 'asks'):
                current_bids = current_snapshot.bids if isinstance(current_snapshot.bids, list) else []
                current_asks = current_snapshot.asks if isinstance(current_snapshot.asks, list) else []
            elif isinstance(current_snapshot, dict):
                current_bids = current_snapshot.get('bids', [])
                current_asks = current_snapshot.get('asks', [])
            else:
                return {'flow_detected': False, 'magnitude': 0.0}
                
            if previous_snapshot:
                if hasattr(previous_snapshot, 'bids') and hasattr(previous_snapshot, 'asks'):
                    prev_bids = previous_snapshot.bids if isinstance(previous_snapshot.bids, list) else []
                    prev_asks = previous_snapshot.asks if isinstance(previous_snapshot.asks, list) else []
                elif isinstance(previous_snapshot, dict):
                    prev_bids = previous_snapshot.get('bids', [])
                    prev_asks = previous_snapshot.get('asks', [])
                else:
                    prev_bids, prev_asks = [], []
            else:
                prev_bids, prev_asks = [], []
                
            # Calculate current and previous depths
            current_bid_depth = sum(float(q) for _, q in current_bids[:10] if len(_) >= 2)
            current_ask_depth = sum(float(q) for _, q in current_asks[:10] if len(_) >= 2)
            prev_bid_depth = sum(float(q) for _, q in prev_bids[:10] if len(_) >= 2)
            prev_ask_depth = sum(float(q) for _, q in prev_asks[:10] if len(_) >= 2)
            
            # Calculate flow magnitude
            bid_flow = abs(current_bid_depth - prev_bid_depth) if prev_bid_depth > 0 else 0
            ask_flow = abs(current_ask_depth - prev_ask_depth) if prev_ask_depth > 0 else 0
            total_flow = (bid_flow + ask_flow) / 2.0
            
            return {
                'flow_detected': total_flow > 0.1,
                'magnitude': float(total_flow),
                'bid_flow': float(bid_flow),
                'ask_flow': float(ask_flow)
            }
        except Exception:
            return {'flow_detected': False, 'magnitude': 0.0}

    def _calculate_market_impact(self, snapshot, side, amount):
        """
        Calculate market impact for a given order size.
        """
        try:
            # Handle OrderBookSnapshot object
            if hasattr(snapshot, 'bids') and hasattr(snapshot, 'asks'):
                bids = snapshot.bids if isinstance(snapshot.bids, list) else []
                asks = snapshot.asks if isinstance(snapshot.asks, list) else []
            # Handle dict format
            elif isinstance(snapshot, dict):
                bids = snapshot.get('bids', [])
                asks = snapshot.get('asks', [])
            else:
                return {'average_price': None, 'slippage': None, 'levels_consumed': 0}
                
            # Determine which side to use based on order side
            levels = asks if side == 'buy' else bids
            
            if not levels or amount <= 0:
                return {'average_price': None, 'slippage': None, 'levels_consumed': 0}
                
            # Calculate market impact
            total_filled = 0.0
            total_cost = 0.0
            levels_consumed = 0
            
            for price, qty in levels:
                if total_filled >= amount:
                    break
                    
                level_amount = price * qty
                remaining = amount - total_filled
                level_fill = min(level_amount, remaining)
                
                total_filled += level_fill
                total_cost += level_fill
                levels_consumed += 1
                
            average_price = total_cost / total_filled if total_filled > 0 else None
            slippage = None
            
            if average_price and len(levels) > 0:
                reference_price = levels[0][0]
                slippage = ((average_price - reference_price) / reference_price) * 100 if reference_price > 0 else 0
                
            return {
                'average_price': average_price,
                'slippage': slippage,
                'levels_consumed': levels_consumed
            }
        except Exception:
            return {'average_price': None, 'slippage': None, 'levels_consumed': 0}

    def _validate_orderbook_snapshot(self, snapshot) -> bool:
        """Validate an OrderBookSnapshot."""
        if snapshot is None:
            return False
        
        # Check required fields
        if not hasattr(snapshot, 'bids') or not hasattr(snapshot, 'asks'):
            return False
        
        # Check if bids and asks are lists
        if not isinstance(snapshot.bids, list) or not isinstance(snapshot.asks, list):
            return False
        
        # Check symbol matches (if we care)
        if hasattr(snapshot, 'symbol') and self.symbol:
            if snapshot.symbol != self.symbol:
                return False
        
        # Check timestamp is reasonable (not in future, not too old)
        current_time = time.time()
        if hasattr(snapshot, 'timestamp'):
            if snapshot.timestamp > current_time + 60:  # 60 seconds in future
                return False
            
            # Optional: check if too old (e.g., > 5 minutes)
            if current_time - snapshot.timestamp > 300:
                return False
        
        # Check for valid price/quantity values
        for price, qty in snapshot.bids + snapshot.asks:
            if price <= 0 or qty <= 0:
                return False
        
        return True

    # ===== ✅ ANÁLISE PRINCIPAL ASYNC (PATCH 3.2: orquestrador refatorado) =====
    async def analyze_async(
        self,
        snapshot: OrderBookSnapshot
    ) -> Dict[str, Any]:
        """Versão assíncrona original."""
        # Sua implementação assíncrona aqui
        pass

    def analyze(
        self,
        snapshot: OrderBookSnapshot
    ) -> Dict[str, Any]:
        """Versão síncrona que envolve a assíncrona."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.analyze_async(snapshot))

    # ===== ✅ ANÁLISE PRINCIPAL ASYNC (PATCH 3.2: orquestrador refatorado) =====
    async def analyze(
        self,
        current_snapshot: Optional[Dict[str, Any]] = None,
        *,
        event_epoch_ms: Optional[int] = None,
        window_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orquestrador refatorado: mantém a API e o formato do evento,
        mas delega trabalho para submétodos (Etapa 3).
        """
        # ✅ Garantia de Símbolo para evitar KeyError em submétodos/decoradores
        if isinstance(current_snapshot, dict) and 'symbol' not in current_snapshot:
            current_snapshot['symbol'] = self.symbol
        
        # Inicializa is_emergency como False
        is_emergency = False

        with self.tracer.start_span(
            "orderbook_analyze",
            {
                "window_id": window_id or "?",
                "has_external_snapshot": current_snapshot is not None,
            },
        ):
            # 1) Acquire + validate snapshot
            snap, err = await self._acquire_snapshot(current_snapshot, event_epoch_ms)
            if err:
                return self._create_invalid_event(err, event_epoch_ms)

            assert snap is not None

            # 2) Extract book + timestamps
            bids, asks, ts_ms, tindex = self._extract_book_data(snap, event_epoch_ms)
            if not bids or not asks:
                return self._create_invalid_event("empty_orderbook", ts_ms)

            # 3) Core metrics
            core, core_err = self._compute_core_metrics(bids, asks)
            if core_err:
                return self._create_invalid_event(core_err, ts_ms)

            assert core is not None
            sm = core["spread_metrics"]
            mid = core["mid"]
            bid_usd = core["bid_usd"]
            ask_usd = core["ask_usd"]
            imbalance = core["imbalance"]
            ratio = core["ratio"]
            pressure = core["pressure"]
            spread_bps = core["spread_bps"]

            # 4) Patterns
            bid_walls = self._detect_walls(bids, side="bid")
            ask_walls = self._detect_walls(asks, side="ask")
            iceberg, iceberg_score = self._compute_iceberg(bids, asks)

            # 5) Market impact
            mi = self._compute_market_impact(bids, asks, mid)

            # 6) Depth + spread analysis
            depth_summary = self._build_depth_summary(bids, asks)
            spread_analysis = self._build_spread_analysis(ts_ms, spread_bps, sm)

            # 7) Labels/alerts + critical flags
            resultado_da_batalha, alertas, is_critical, critical_flags = self._build_labels_and_alerts(
                imbalance=imbalance,
                iceberg=iceberg,
                spread_bps=spread_bps,
                ratio=ratio,
                bid_usd=bid_usd,
                ask_usd=ask_usd,
            )
            
            # Atualiza is_emergency com o valor retornado em critical_flags
            is_emergency = critical_flags.get("is_emergency", False)

            # 8) Description
            descricao = self._build_description(
                imbalance=imbalance,
                ratio=ratio,
                bid_usd=bid_usd,
                ask_usd=ask_usd,
            )

            # 9) Data quality
            data_source, age_seconds = self._get_data_quality()

            # 10) Advanced metrics
            weighted_imb = self._weighted_imbalance(bids, asks, use_notional=True)
            liq_bids = self._liquidity_concentration(bids, top_n=10)
            liq_asks = self._liquidity_concentration(asks, top_n=10)
            micro = self._microstructure_metrics(bids, asks, top_n=self.top_n)
            anoms = self._detect_anomalies(bids, asks, self.prev_snapshot)

            # 11) Build event
            event: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "engine_version": ENGINE_VERSION,
                "tipo_evento": "OrderBook",
                "ativo": self.symbol,

                "is_valid": True,
                "should_skip": False,
                "emergency_mode": critical_flags.get("is_emergency", False),  # 🆕 Usa a flag de emergency mode

                "descricao": descricao,
                "resultado_da_batalha": resultado_da_batalha,

                "imbalance": round(imbalance, 4),
                "volume_ratio": round(ratio, 4) if ratio not in (None, float("inf")) else None,
                "pressure": round(pressure, 4),

                "spread_metrics": sm,
                "alertas_liquidez": alertas,
                "iceberg_reloaded": bool(iceberg),
                "iceberg_score": iceberg_score,
                "walls": {"bids": bid_walls[:3], "asks": ask_walls[:3]},

                "market_impact_buy": mi["buy"],
                "market_impact_sell": mi["sell"],

                "top_n": self.top_n,
                "ob_limit": self.ob_limit_fetch,

                "timestamps": {
                    "exchange_ms": ts_ms,
                    "timestamp_ny": tindex.get("timestamp_ny"),
                    "timestamp_utc": tindex.get("timestamp_utc"),
                },

                "source": {
                    "exchange": "binance_futures",
                    "endpoint": "fapi/v1/depth",
                    "symbol": self.symbol,
                },

                "labels": {
                    "dominant_label": resultado_da_batalha,
                    "note": "Rótulo baseado no livro (estoque de liquidez).",
                },

                "order_book_depth": depth_summary,
                "spread_analysis": spread_analysis,

                "severity": "CRITICAL" if is_critical else "INFO",

                "critical_flags": critical_flags,

                "orderbook_data": {
                    "mid": sm["mid"],
                    "spread": sm["spread"],
                    "spread_percent": sm["spread_percent"],
                    "bid_depth_usd": bid_usd,
                    "ask_depth_usd": ask_usd,
                    "imbalance": round(imbalance, 4),
                    "volume_ratio": round(ratio, 4) if ratio not in (None, float("inf")) else None,
                    "pressure": round(pressure, 4),
                },

                "depth_metrics": snap.get("depth_metrics", {}),

                "data_quality": {
                    "is_valid": True,
                    "data_source": data_source,
                    "age_seconds": round(age_seconds, 2),
                    "validation_passed": True,
                    "validation_issues": [],
                    "warnings": [],
                    "emergency_mode": False,
                },

                "advanced_metrics": {
                    "weighted_imbalance": round(float(weighted_imb), 6),
                    "liquidity_concentration": {"bids": liq_bids, "asks": liq_asks},
                    "microstructure": micro,
                    "anomalies": anoms,
                    "dynamic_thresholds": self.dynamic_thresholds,
                },

                "health_stats": self.get_stats(),
            }

            # 12) Update state + invariants + metrics export
            self._update_internal_state(bids, asks, ts_ms)

            if event.get("is_valid"):
                self._validate_invariants(event)
                event["invariants_checked"] = True

            self._export_metrics_from_event(event)

            # Logging estruturado + logging humano
            try:
                if hasattr(self, 'slog') and self.slog:
                    self.slog.info(
                        "orderbook_event",
                        window_id=window_id or "?",
                        severity=event.get("severity", "INFO"),
                        data_source=event.get("data_quality", {}).get("data_source", "unknown"),
                        symbol=getattr(self, 'symbol', 'UNKNOWN'), # ✅ Passando explicitamente
                        bid_depth_usd=float(bid_usd),
                        ask_depth_usd=float(ask_usd),
                        imbalance=float(imbalance),
                        spread_bps=float(spread_bps) if spread_bps is not None else None,
                        emergency_mode=bool(critical_flags.get("is_emergency", False)),  # 🆕 Inclui emergency mode no log
                    )
            except Exception as e:
                logging.debug(f"Falha no slog orderbook_event: {e}")

            logging.info(
                "📊 OrderBook OK - Janela #{}: bid=${:,.0f}, ask=${:,.0f}".format(
                    window_id if window_id else '?',
                    float(bid_usd),
                    float(ask_usd)
                )
            )

            return event

    # ===== VALIDATE INVARIANTS =====
    def _validate_invariants(self, ob_event: Dict[str, Any]) -> None:
        """
        Verifica consistência matemática do evento de OrderBook gerado.
        Não altera o evento, apenas loga warnings.
        """
        try:
            ob_data = ob_event.get("orderbook_data") or {}
            spread_metrics = ob_event.get("spread_metrics") or {}

            # 1) Spread positivo (usa spread presente em orderbook_data ou spread_metrics)
            spread = None
            if "spread" in ob_data:
                spread = float(ob_data["spread"])
            elif "spread" in spread_metrics:
                spread = float(spread_metrics["spread"])

            if spread is not None and spread < 0:
                symbol = getattr(self, 'symbol', 'UNKNOWN')
                logging.warning(
                    f"⚠️ INVARIANTE VIOLADA (Spread): spread negativo={spread} em {symbol}"
                )

            # 2) Imbalance consistente com bid/ask depth
            bid_usd = float(ob_data.get("bid_depth_usd", 0.0))
            ask_usd = float(ob_data.get("ask_depth_usd", 0.0))
            stored_imbalance = float(
                ob_data.get("imbalance", spread_metrics.get("imbalance", 0.0))
            )

            total_liq = bid_usd + ask_usd
            if total_liq > 0:
                calc_imbalance = (bid_usd - ask_usd) / total_liq
                if abs(calc_imbalance - stored_imbalance) > 0.01:
                    symbol = getattr(self, 'symbol', 'UNKNOWN')
                    logging.warning(
                        f"⚠️ INVARIANTE VIOLADA (Imbalance): "
                        f"calc={calc_imbalance:.4f} vs stored={stored_imbalance:.4f} "
                        f"(bid_usd={bid_usd:.0f}, ask_usd={ask_usd:.0f}) em {symbol}"
                    )
        except Exception as e:
            logging.debug(f"Erro na validação de invariantes do OrderBook: {e}", exc_info=True)

    # ===== HEALTH MONITORING (PATCH A-03: requests_last_min mais fiel) =====
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance e health."""
        error_rate = 100 * self._fetch_errors / max(1, self._total_fetches)
        validation_failure_rate = 100 * self._validation_failures / max(1, self._total_fetches)
        cache_hit_rate = 100 * self._cache_hits / max(1, self._total_fetches)

        now_mono = time.monotonic()
        
        # PATCH A-03: Limpa deque antes de contar
        while self._request_times_mono and (now_mono - self._request_times_mono[0] > 60.0):
            self._request_times_mono.popleft()
        
        cache_age = now_mono - self._cache_timestamp_mono if self._cache_timestamp_mono > 0 else None
        stale_age = now_mono - self._last_valid_timestamp_mono if self._last_valid_timestamp_mono > 0 else None

        return {
            "total_fetches": self._total_fetches,
            "fetch_errors": self._fetch_errors,
            "validation_failures": self._validation_failures,
            "cache_hits": self._cache_hits,
            "stale_data_uses": self._stale_data_uses,
            "emergency_uses": self._emergency_uses,
            "old_data_rejected": self._old_data_rejected,

            "error_rate_pct": round(error_rate, 2),
            "validation_failure_rate_pct": round(validation_failure_rate, 2),
            "cache_hit_rate_pct": round(cache_hit_rate, 2),

            "has_cached_data": self._cached_snapshot is not None,
            "has_stale_data": self._last_valid_snapshot is not None,

            "cache_age_seconds": round(cache_age, 2) if cache_age is not None else None,
            "stale_age_seconds": round(stale_age, 2) if stale_age is not None else None,

            "requests_last_min": len(self._request_times_mono),
            "rate_limit_threshold": self.rate_limit_threshold,

            "circuit_breaker": self._circuit_breaker.snapshot(),
            "fallback": get_fallback_instance().get_fallback_stats(),

            "config": {
                "cache_ttl": self.cache_ttl_seconds,
                "max_stale": self.max_stale_seconds,
                "emergency_mode": self.cfg.emergency_mode,
                "allow_partial": self.cfg.allow_partial,
                "min_depth_usd": self.cfg.min_depth_usd,
                "max_age_ms": ORDERBOOK_MAX_AGE_MS,
                "config_loaded": CONFIG_LOADED,
                "history_maxlen": self._history_maxlen,
            }
        }

    def reset_stats(self):
        """Reseta contadores de estatísticas."""
        self._fetch_errors = 0
        self._total_fetches = 0
        self._validation_failures = 0
        self._cache_hits = 0
        self._stale_data_uses = 0
        self._emergency_uses = 0
        self._old_data_rejected = 0
        
        # Reset circuit breaker by creating a new instance with same config
        failure_threshold = getattr(self.cfg, 'circuit_breaker_failure_threshold', ORDERBOOK_CIRCUIT_BREAKER_FAILURE_THRESHOLD)
        success_threshold = getattr(self.cfg, 'circuit_breaker_success_threshold', ORDERBOOK_CIRCUIT_BREAKER_SUCCESS_THRESHOLD)
        timeout_seconds = getattr(self.cfg, 'circuit_breaker_timeout_seconds', ORDERBOOK_CIRCUIT_BREAKER_TIMEOUT_SECONDS)
        half_open_max_calls = getattr(self.cfg, 'circuit_breaker_half_open_max_calls', ORDERBOOK_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS)
        
        cb_config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            half_open_max_calls=half_open_max_calls,
        )
        self._circuit_breaker = CircuitBreaker(name=f"orderbook_{self.symbol}", config=cb_config)
        
        logging.info("📊 Estatísticas resetadas")

    # ===== SHIMS DE COMPATIBILIDADE (DEPRECATED) =====
    async def _warn_deprecated_shim(self, shim_name: str) -> None:
        """
        Loga WARNING uma única vez por instância quando shims forem usados.
        """
        if not getattr(self, "_shims_deprecated_warned", False):
            self._shims_deprecated_warned = True
            logging.warning(
                f"[DEPRECATED] Método '{shim_name}' será removido futuramente. "
                f"Use 'await analyze(...)' em vez disso."
            )

    async def analyze_order_book(self, *args, **kwargs) -> Dict[str, Any]:
        """
        DEPRECATED: use 'await analyze(...)' em vez disso.
        Mantido apenas por compatibilidade.
        """
        await self._warn_deprecated_shim("analyze_order_book")
        return await self.analyze(*args, **kwargs)

    async def analyzeOrderBook(self, *args, **kwargs) -> Dict[str, Any]:
        """
        DEPRECATED: use 'await analyze(...)' em vez disso.
        Mantido apenas por compatibilidade.
        """
        await self._warn_deprecated_shim("analyzeOrderBook")
        return await self.analyze(*args, **kwargs)

    def handle_raw_event(self, event):
        """
        Trata eventos sem preço de fechamento, garantindo que close_price esteja presente.
        """
        if 'close_price' not in event:
            event['close_price'] = event.get('last_price') or getattr(self, 'last_price', None)
            if not event['close_price']:
                # Usar preço médio do orderbook
                event['close_price'] = (event.get('best_bid', 0) +
                                       event.get('best_ask', 0)) / 2
        return event

    def analyze_orderbook(self, current_snapshot=None, *, event_epoch_ms=None, window_id=None):
        """
        Synchronous wrapper around async analyze method for test compatibility.
        """
        try:
            # Handle OrderBookSnapshot object
            if hasattr(current_snapshot, 'bids') and hasattr(current_snapshot, 'asks'):
                # Get symbol from object or fallback to analyzer's symbol
                snap_symbol = getattr(current_snapshot, 'symbol', None) or self.symbol
                # Convert to dict format expected by analyze
                snapshot_dict = {
                    'bids': current_snapshot.bids if isinstance(current_snapshot.bids, list) else [],
                    'asks': current_snapshot.asks if isinstance(current_snapshot.asks, list) else [],
                    'E': int(current_snapshot.timestamp * 1000) if hasattr(current_snapshot, 'timestamp') else None,
                    'lastUpdateId': current_snapshot.last_update_id if hasattr(current_snapshot, 'last_update_id') else None,
                    'symbol': snap_symbol
                }
                current_snapshot = snapshot_dict
            elif isinstance(current_snapshot, dict) and 'symbol' not in current_snapshot:
                current_snapshot['symbol'] = self.symbol

            # Run async analyze method synchronously
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.analyze(current_snapshot, event_epoch_ms=event_epoch_ms, window_id=window_id)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'analysis_failed: {str(e)}',
                'symbol': self.symbol,
                'timestamp': event_epoch_ms or int(time.time() * 1000)
            }

    # ===== VALIDATE INVARIANTS =====




# ===== TESTE =====
async def main_test():
    """Função de teste async"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
    )

    print("\n" + "=" * 80)
    print("🧪 TESTE DE ORDERBOOK ANALYZER v2.2.0 (ASYNC, SESSÃO REUTILIZÁVEL + MELHORIAS)")
    print("=" * 80 + "\n")

    async with OrderBookAnalyzer(symbol="BTCUSDT") as oba:
        # Teste 1: Fetch normal
        print("📡 Teste 1: Fetch normal (async)...")
        evt = await oba.analyze()  # ✅ ASYNC

        print(f"\n ✓ is_valid: {evt.get('is_valid')}")
        print(f" ✓ should_skip: {evt.get('should_skip')}")
        print(f" ✓ emergency_mode: {evt.get('emergency_mode')}")
        print(f" ✓ Severity: {evt.get('severity')}")
        print(f" ✓ Resultado: {evt.get('resultado_da_batalha')}")
        print(f" ✓ Bid Depth: ${evt.get('orderbook_data', {}).get('bid_depth_usd', 0):,.2f}")
        print(f" ✓ Ask Depth: ${evt.get('orderbook_data', {}).get('ask_depth_usd', 0):,.2f}")
        print(f" ✓ Imbalance: {evt.get('orderbook_data', {}).get('imbalance', 0):+.4f}")
        print(f" ✓ Data Source: {evt.get('data_quality', {}).get('data_source')}")
        
        # Novas métricas
        print(f" ✓ Weighted Imbalance: {evt.get('advanced_metrics', {}).get('weighted_imbalance', 0):.6f}")
        print(f" ✓ Anomalies: {evt.get('advanced_metrics', {}).get('anomalies', [])}")

        # Teste 2: Cache hit
        print("\n📦 Teste 2: Cache hit...")
        await asyncio.sleep(0.1)  # ✅ ASYNC
        evt2 = await oba.analyze()  # ✅ ASYNC
        print(f" ✓ Data Source: {evt2.get('data_quality', {}).get('data_source')}")
        print(f" ✓ Age: {evt2.get('data_quality', {}).get('age_seconds')}s")

        # Teste 3: Snapshot parcial (deve REJEITAR)
        print("\n🚫 Teste 3: Snapshot parcial (deve REJEITAR)...")
        partial_snap = {
            "E": int(time.time() * 1000),
            "bids": [(50000.0, 1.0)],
            "asks": [(50001.0, 0.0)]
        }
        evt_partial = await oba.analyze(current_snapshot=partial_snap)  # ✅ ASYNC
        print(f" ✓ is_valid: {evt_partial.get('is_valid')}")
        print(f" ✓ should_skip: {evt_partial.get('should_skip')}")
        print(f" ✓ Resultado: {evt_partial.get('resultado_da_batalha')}")
        print(f" ✓ Erro: {evt_partial.get('erro')}")

        # Teste 4: Stats
        print("\n📊 Teste 4: Health Stats...")
        stats = oba.get_stats()
        for key, val in stats.items():
            if isinstance(val, dict):
                print(f" • {key}:")
                for k2, v2 in val.items():
                    print(f"   - {k2}: {v2}")
            else:
                print(f" • {key}: {val}")
        
        # Teste 5: Metrics integration
        print("\n📈 Teste 5: Metrics Integration...")
        print(f" • Metrics enabled: {oba.metrics.enabled}")
        print(f" • Prometheus available: {oba.metrics.prom_available}")
        print(f" • Metrics type: {type(oba.metrics.fetch_total).__name__}")

        # Teste 6: Circuit Breaker
        print("\n🔌 Teste 6: Circuit Breaker Status...")
        cb_snapshot = oba._circuit_breaker.snapshot()
        print(f" • Circuit Breaker State: {cb_snapshot['state']}")
        print(f" • Failure Count: {cb_snapshot['failure_count']}")
        print(f" • Success Count: {cb_snapshot['success_count']}")
        print(f" • Half Open Calls: {cb_snapshot['half_open_calls']}")
        print(f" • Allow Request: {oba._circuit_breaker.allow_request()}")

    print("\n" + "=" * 80)
    print("✅ TESTES CONCLUÍDOS - ORDERBOOK v2.2.0 ASYNC (SESSÃO REUTILIZÁVEL + MELHORIAS)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # ✅ Roda função async
    asyncio.run(main_test())