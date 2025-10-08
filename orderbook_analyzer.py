# orderbook_analyzer.py (VERSÃO CORRIGIDA COMPLETA - v1.4.0)
"""
OrderBook Analyzer para Binance Futures com validação robusta.

🔹 CORREÇÕES v1.4.0:
  ✅ Validação rigorosa de snapshots (rejeita zeros)
  ✅ Rate limiting preventivo (evita 429)
  ✅ Fallback para dados antigos (stale data)
  ✅ Timeouts aumentados (10/15s)
  ✅ Flags de qualidade (is_valid, data_source)
  ✅ Logs completos (sem supressão)
  ✅ Detecção de spread negativo/absurdo
  ✅ Validação de volumes mínimos
  ✅ Retry com backoff exponencial melhorado
  ✅ Exceção customizada para erros críticos
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

import requests
import numpy as np

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

from time_manager import TimeManager

# Importa parâmetros de configuração
try:
    from config import (
        ORDER_BOOK_DEPTH_LEVELS,
        SPREAD_TIGHT_THRESHOLD_BPS,
        SPREAD_AVG_WINDOWS_MIN,
        ORDERBOOK_CRITICAL_IMBALANCE,
        ORDERBOOK_MIN_DOMINANT_USD,
        ORDERBOOK_MIN_RATIO_DOM,
    )
except Exception:
    ORDER_BOOK_DEPTH_LEVELS = [1, 5, 10, 25]
    SPREAD_TIGHT_THRESHOLD_BPS = 0.2
    SPREAD_AVG_WINDOWS_MIN = [60, 1440]
    ORDERBOOK_CRITICAL_IMBALANCE = 0.95
    ORDERBOOK_MIN_DOMINANT_USD = 2_000_000.0
    ORDERBOOK_MIN_RATIO_DOM = 20.0

SCHEMA_VERSION = "1.4.0"

# 🆕 Exceção customizada
class OrderBookUnavailableError(Exception):
    """Levantada quando orderbook não pode ser obtido ou é inválido."""
    pass


# ------------------------- Utils -------------------------

def _to_float_list(levels: List[List[str]]) -> List[Tuple[float, float]]:
    """Converte níveis de orderbook [price, qty] para float tuples."""
    out: List[Tuple[float, float]] = []
    for lv in levels or []:
        try:
            p = float(lv[0])
            q = float(lv[1])
            if p > 0 and q > 0:  # 🆕 Valida que são positivos
                out.append((p, q))
        except (ValueError, TypeError, IndexError):
            continue
    return out


def _sum_depth_usd(levels: List[Tuple[float, float]], top_n: int) -> float:
    """Soma profundidade em USD dos top N níveis."""
    if not levels:
        return 0.0
    arr = levels[: max(1, top_n)]
    return float(sum(p * q for p, q in arr))


def _simulate_market_impact(
    levels: List[Tuple[float, float]], usd_amount: float, side: str, mid: Optional[float]
) -> Dict[str, Any]:
    """
    Simula impacto de ordem de mercado caminhando pelo livro.
    
    Args:
        levels: Níveis do livro (ASKs para buy, BIDs invertidos para sell)
        usd_amount: Valor em USD a ser executado
        side: 'buy' ou 'sell'
        mid: Preço mid para calcular deslocamento
    
    Returns:
        Dict com move_usd, bps, levels_crossed, vwap
    """
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
            dq = remaining / price
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
    
    if mid and terminal_price:
        if side == "buy":
            move_usd = max(0.0, terminal_price - mid)
        else:
            move_usd = max(0.0, mid - terminal_price)
        if mid > 0:
            bps = (move_usd / mid) * 10000.0

    return {
        "usd": usd_amount,
        "move_usd": round(move_usd, 4),
        "bps": round(bps, 4),
        "levels": levels_crossed,
        "vwap": vwap,
    }


# ------------------------- Analyzer -------------------------

class OrderBookAnalyzer:
    """
    Analisador de Order Book para Binance Futures com validação robusta.
    
    Features:
      - Validação rigorosa de dados (rejeita zeros/inválidos)
      - Rate limiting preventivo
      - Fallback para dados antigos (stale data)
      - Retry com backoff exponencial
      - Flags de qualidade de dados
      - Métricas detalhadas de health
    """

    def __init__(
        self,
        symbol: str,
        liquidity_flow_alert_percentage: float = 0.40,
        wall_std_dev_factor: float = 3.0,
        top_n_levels: int = 20,
        ob_limit_fetch: int = 100,
        time_manager: Optional[TimeManager] = None,
        cache_ttl_seconds: float = 1.0,
        max_stale_seconds: float = 30.0,  # 🆕
        rate_limit_threshold: int = 10,    # 🆕 Max req/min
    ):
        self.symbol = symbol.upper()
        self.alert_threshold = float(liquidity_flow_alert_percentage)
        self.wall_std = float(wall_std_dev_factor)
        self.top_n = int(top_n_levels)
        self.ob_limit_fetch = int(ob_limit_fetch)
        self.tz_ny = ZoneInfo("America/New_York") if ZoneInfo else None
        self.tm = time_manager or TimeManager()

        # Cache e validação
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_stale_seconds = max_stale_seconds
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cache_timestamp: float = 0.0
        
        # 🆕 Fallback para dados antigos
        self._last_valid_snapshot: Optional[Dict[str, Any]] = None
        self._last_valid_timestamp: float = 0.0
        
        # 🆕 Rate limiting preventivo
        self._request_times: List[float] = []
        self._rate_limit_threshold = rate_limit_threshold
        
        # Estatísticas
        self._fetch_errors = 0
        self._total_fetches = 0
        self._validation_failures = 0
        self._cache_hits = 0
        self._stale_data_uses = 0

        # Configurações adicionais
        self.depth_levels: List[int] = list(ORDER_BOOK_DEPTH_LEVELS)
        self.spread_tight_threshold_bps: float = float(SPREAD_TIGHT_THRESHOLD_BPS)
        self.spread_avg_windows_min: List[int] = list(SPREAD_AVG_WINDOWS_MIN)
        self.spread_history: List[Tuple[int, float]] = []

        # Memória para heurística de iceberg
        self.prev_snapshot: Optional[Dict[str, Any]] = None
        self.last_event_ts_ms: Optional[int] = None

        logging.info(
            "✅ OrderBook Analyzer v%s inicializado | "
            "Symbol: %s | Alert: %.0f%% | Wall STD: %.1fx | "
            "Top N: %s | Cache TTL: %.1fs | Max Stale: %.1fs | Rate Limit: %s req/min",
            SCHEMA_VERSION,
            self.symbol,
            self.alert_threshold * 100,
            self.wall_std,
            self.top_n,
            self.cache_ttl_seconds,
            self.max_stale_seconds,
            self._rate_limit_threshold,
        )

    # -------- 🆕 VALIDAÇÃO RIGOROSA --------

    def _validate_snapshot(self, snap: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valida snapshot de orderbook com múltiplas checagens.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues: List[str] = []
        
        # 1. Estrutura básica
        if not isinstance(snap, dict):
            issues.append("snapshot não é dict")
            return False, issues
        
        if "bids" not in snap or "asks" not in snap:
            issues.append("snapshot sem bids/asks")
            return False, issues
        
        bids = snap.get("bids", [])
        asks = snap.get("asks", [])
        
        # 2. Dados não vazios
        if not bids or not asks:
            issues.append(f"orderbook vazio (bids={len(bids)}, asks={len(asks)})")
            return False, issues
        
        # 3. Valores numéricos válidos
        try:
            best_bid_price = float(bids[0][0])
            best_bid_qty = float(bids[0][1])
            best_ask_price = float(asks[0][0])
            best_ask_qty = float(asks[0][1])
            
            # 🆕 Valida que são positivos
            if best_bid_price <= 0 or best_ask_price <= 0:
                issues.append(f"preços inválidos (bid={best_bid_price}, ask={best_ask_price})")
                return False, issues
            
            if best_bid_qty <= 0 or best_ask_qty <= 0:
                issues.append(f"quantidades zero (bid_qty={best_bid_qty}, ask_qty={best_ask_qty})")
                return False, issues
            
            # 4. 🆕 Spread não pode ser negativo
            if best_ask_price < best_bid_price:
                issues.append(f"spread negativo! (bid={best_bid_price} > ask={best_ask_price})")
                return False, issues
            
            spread_pct = (best_ask_price - best_bid_price) / best_bid_price * 100
            
            # 5. 🆕 Spread absurdo (> 10%)
            if spread_pct > 10:
                issues.append(f"spread absurdo ({spread_pct:.2f}%)")
                # Não invalida mas loga
            
            # 6. 🆕 Volume mínimo nos top 5 níveis
            bid_vol = sum(float(b[1]) for b in bids[:5] if len(b) >= 2)
            ask_vol = sum(float(a[1]) for a in asks[:5] if len(a) >= 2)
            
            if bid_vol == 0 or ask_vol == 0:
                issues.append(f"volume zero nos top 5 níveis (bid={bid_vol}, ask={ask_vol})")
                return False, issues
            
            # 7. 🆕 Profundidade USD mínima
            bid_depth_usd = _sum_depth_usd(bids, 5)
            ask_depth_usd = _sum_depth_usd(asks, 5)
            
            if bid_depth_usd == 0 or ask_depth_usd == 0:
                issues.append(f"profundidade USD zero (bid=${bid_depth_usd}, ask=${ask_depth_usd})")
                return False, issues
            
            # Mínimo de $1000 nos top 5 níveis para ser válido
            if bid_depth_usd < 1000 or ask_depth_usd < 1000:
                issues.append(f"liquidez muito baixa (bid=${bid_depth_usd:.0f}, ask=${ask_depth_usd:.0f})")
                # Aceita mas loga warning
            
            return True, issues
            
        except (IndexError, ValueError, TypeError) as e:
            issues.append(f"erro ao validar dados: {e}")
            return False, issues

    # -------- 🆕 RATE LIMITING PREVENTIVO --------

    def _check_rate_limit(self) -> bool:
        """
        Verifica se está próximo do rate limit.
        
        Returns:
            True se deve aguardar antes de fazer request
        """
        now = time.time()
        
        # Remove requests > 60s atrás
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        # Verifica limite
        if len(self._request_times) >= self._rate_limit_threshold:
            return True
        
        return False
    
    def _register_request(self):
        """Registra request para tracking de rate limit."""
        self._request_times.append(time.time())

    # -------- 🆕 FETCH COM VALIDAÇÃO E FALLBACK --------

    def _fetch_orderbook(
        self, 
        limit: Optional[int] = None, 
        use_cache: bool = True,
        allow_stale: bool = True,  # 🆕
    ) -> Optional[Dict[str, Any]]:
        """
        Busca orderbook com retry, validação e fallback.
        
        Args:
            limit: Número de níveis a buscar
            use_cache: Se True, usa cache se disponível
            allow_stale: Se True, usa dados antigos em caso de falha
        
        Returns:
            Snapshot válido ou None
        """
        self._total_fetches += 1
        
        # 🆕 1. VERIFICA CACHE
        if use_cache and self._cached_snapshot is not None:
            cache_age = time.time() - self._cache_timestamp
            if cache_age < self.cache_ttl_seconds:
                self._cache_hits += 1
                logging.debug(f"📦 Cache hit (age={cache_age:.2f}s)")
                return self._cached_snapshot
        
        # 🆕 2. RATE LIMITING PREVENTIVO
        if self._check_rate_limit():
            wait_time = 1.0
            logging.warning(
                f"⏳ Rate limit preventivo ativado "
                f"({len(self._request_times)} req/min) - aguardando {wait_time}s..."
            )
            time.sleep(wait_time)
        
        # 🆕 3. TENTA FETCH COM RETRY
        lim = limit or self.ob_limit_fetch
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit={lim}"
        
        max_retries = 5  # 🆕 Era 3
        base_delay = 1.0  # 🆕 Era 0.5
        
        for attempt in range(max_retries):
            try:
                # 🆕 Timeout aumentado
                timeout = 10.0 if attempt == 0 else 15.0  # Era 5/10
                
                self._register_request()
                
                logging.debug(
                    f"📡 Fetching orderbook (attempt {attempt + 1}/{max_retries}, "
                    f"timeout={timeout}s)..."
                )
                
                r = requests.get(url, timeout=timeout)
                
                # 🆕 Detecta rate limiting
                if r.status_code == 429:
                    retry_after = int(r.headers.get('Retry-After', 60))
                    self._fetch_errors += 1
                    
                    logging.error(
                        f"🚫 RATE LIMIT (429) - Retry após {retry_after}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    
                    if attempt < max_retries - 1:
                        time.sleep(retry_after)
                        continue
                    else:
                        break
                
                r.raise_for_status()
                data = r.json()
                
                # 🆕 Parse
                parsed = {
                    "lastUpdateId": data.get("lastUpdateId"),
                    "E": data.get("E"),
                    "T": data.get("T"),
                    "bids": _to_float_list(data.get("bids", [])),
                    "asks": _to_float_list(data.get("asks", [])),
                }
                
                # 🆕 4. VALIDA SNAPSHOT
                is_valid, issues = self._validate_snapshot(parsed)
                
                if not is_valid:
                    self._validation_failures += 1
                    
                    logging.error(
                        f"❌ Snapshot inválido (attempt {attempt + 1}): {', '.join(issues)}"
                    )
                    
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logging.debug(f"   Retry em {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        break
                
                # ✅ SUCESSO - Atualiza caches
                self._cached_snapshot = parsed
                self._cache_timestamp = time.time()
                
                self._last_valid_snapshot = parsed.copy()
                self._last_valid_timestamp = time.time()
                
                logging.debug(
                    f"✅ Orderbook obtido e validado: "
                    f"{len(parsed['bids'])} bids, {len(parsed['asks'])} asks"
                )
                
                return parsed
                
            except requests.exceptions.Timeout as e:
                self._fetch_errors += 1
                logging.error(f"⏱️ Timeout (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    
            except requests.exceptions.RequestException as e:
                self._fetch_errors += 1
                logging.error(f"🌐 Request error (attempt {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    
            except Exception as e:
                self._fetch_errors += 1
                logging.error(f"💥 Unexpected error (attempt {attempt + 1}): {e}", exc_info=True)
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
        
        # 🆕 5. TODAS AS TENTATIVAS FALHARAM - TENTA FALLBACK
        if allow_stale and self._last_valid_snapshot is not None:
            age = time.time() - self._last_valid_timestamp
            
            if age < self.max_stale_seconds:
                self._stale_data_uses += 1
                
                logging.warning(
                    f"⚠️ Fetch falhou - usando snapshot antigo (age={age:.1f}s)"
                )
                
                return self._last_valid_snapshot.copy()
            else:
                logging.error(
                    f"❌ Snapshot antigo muito velho ({age:.1f}s > {self.max_stale_seconds}s) "
                    f"- descartado"
                )
        
        # 💀 FALHA TOTAL
        error_rate = 100 * self._fetch_errors / max(1, self._total_fetches)
        
        logging.error(
            f"💀 FALHA TOTAL ao obter orderbook após {max_retries} tentativas. "
            f"Taxa de erro: {self._fetch_errors}/{self._total_fetches} ({error_rate:.1f}%)"
        )
        
        return None

    # -------- MÉTRICAS (MANTÉM CÓDIGO ORIGINAL) --------

    def _spread_and_depth(
        self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]
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
        spread_pct = ((spread / mid) * 100.0) if (spread is not None and mid) else None

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
        self, bid_usd: float, ask_usd: float
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calcula imbalance, ratio e pressure."""
        total = bid_usd + ask_usd
        if total <= 0:
            return None, None, None
        
        imbalance = (bid_usd - ask_usd) / total  # [-1, +1]
        ratio = (bid_usd / ask_usd) if ask_usd > 0 else float("inf")
        pressure = imbalance
        
        return float(imbalance), float(ratio), float(pressure)

    def _detect_walls(
        self, side_levels: List[Tuple[float, float]], side: str
    ) -> List[Dict[str, Any]]:
        """Detecta paredes de liquidez (qty > média + k*std)."""
        if not side_levels:
            return []
        
        levels = side_levels[: self.top_n]
        qtys = np.array([q for _, q in levels], dtype=float)
        
        if qtys.size == 0:
            return []
        
        mean = float(np.mean(qtys))
        std = float(np.std(qtys))
        threshold = mean * 1.5 if std <= 1e-12 else mean + self.wall_std * std

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

    def _iceberg_reload(
        self, prev: Optional[Dict[str, Any]], curr: Dict[str, Any], tol: float = 0.75
    ) -> Tuple[bool, float]:
        """Detecta possível recarga de ordens iceberg."""
        try:
            if not prev:
                return False, 0.0
            
            prev_bids = dict(prev.get("bids", []))
            prev_asks = dict(prev.get("asks", []))
            curr_bids = dict(curr.get("bids", []))
            curr_asks = dict(curr.get("asks", []))

            score = 0.0
            
            for side_label, pbook_prev, pbook_curr in [
                ("bid", prev_bids, curr_bids),
                ("ask", prev_asks, curr_asks),
            ]:
                if not pbook_prev or not pbook_curr:
                    continue
                
                p_prev = max(pbook_prev.keys()) if side_label == "bid" else min(pbook_prev.keys())
                p_curr = max(pbook_curr.keys()) if side_label == "bid" else min(pbook_curr.keys())
                
                if p_prev == p_curr:
                    q_prev = float(pbook_prev[p_prev])
                    q_curr = float(pbook_curr[p_curr])
                    
                    if q_curr >= tol * max(q_prev, 1e-9) and q_curr > q_prev:
                        score += min(1.0, (q_curr - q_prev) / max(q_prev, 1e-9))
            
            return (score > 0.5), float(round(score, 4))
            
        except Exception:
            return False, 0.0

    # -------- 🆕 EVENTO INVÁLIDO COM FLAGS CLARAS --------

    def _create_invalid_event(
        self, 
        error_msg: str, 
        ts_ms: Optional[int] = None,
        severity: str = "ERROR",
    ) -> Dict[str, Any]:
        """
        Cria evento marcado como INVÁLIDO com flags claras.
        
        🔹 DIFERENÇA DA VERSÃO ANTIGA:
          - Adiciona `is_valid: False`
          - Adiciona `should_skip: True`
          - Severity claramente ERROR
          - data_quality com detalhes
        """
        if ts_ms is None:
            ts_ms = self.tm.now_ms()
        
        tindex = self.tm.build_time_index(ts_ms, include_local=True, timespec="seconds")
        
        return {
            "schema_version": SCHEMA_VERSION,
            "tipo_evento": "OrderBook",
            "ativo": self.symbol,
            
            # 🆕 FLAGS DE VALIDADE
            "is_valid": False,
            "should_skip": True,
            "erro": error_msg,
            
            "descricao": f"❌ Order book indisponível: {error_msg}",
            "resultado_da_batalha": "INDISPONÍVEL",
            
            "imbalance": 0.0,
            "volume_ratio": 1.0,
            "pressure": 0.0,
            
            "spread_metrics": {
                "mid": 0.0,
                "spread": 0.0,
                "spread_percent": 0.0,
                "bid_depth_usd": 0.0,
                "ask_depth_usd": 0.0,
            },
            
            "alertas_liquidez": [f"🚫 ERRO: {error_msg}"],
            "iceberg_reloaded": False,
            "iceberg_score": 0.0,
            "walls": {"bids": [], "asks": []},
            
            "market_impact_buy": {
                "100k": {"usd": 100000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
                "1M": {"usd": 1000000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
            },
            "market_impact_sell": {
                "100k": {"usd": 100000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
                "1M": {"usd": 1000000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
            },
            
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
                "error": error_msg,
            },
            
            "labels": {
                "dominant_label": "INDISPONÍVEL",
                "note": "Order book não pôde ser obtido ou validado.",
            },
            
            "order_book_depth": {},
            "spread_analysis": {},
            
            "severity": severity,
            
            "critical_flags": {
                "is_critical": False,
                "abs_imbalance": 0.0,
                "ratio_dom": 1.0,
                "dominant_usd": 0.0,
                "thresholds": {
                    "ORDERBOOK_CRITICAL_IMBALANCE": ORDERBOOK_CRITICAL_IMBALANCE,
                    "ORDERBOOK_MIN_DOMINANT_USD": ORDERBOOK_MIN_DOMINANT_USD,
                    "ORDERBOOK_MIN_RATIO_DOM": ORDERBOOK_MIN_RATIO_DOM,
                },
            },
            
            "orderbook_data": {
                "mid": 0.0,
                "spread": 0.0,
                "spread_percent": 0.0,
                "bid_depth_usd": 0.0,
                "ask_depth_usd": 0.0,
                "imbalance": 0.0,
                "volume_ratio": 1.0,
                "pressure": 0.0,
            },
            
            # 🆕 QUALITY METRICS
            "data_quality": {
                "is_valid": False,
                "data_source": "error",
                "age_seconds": 0.0,
                "validation_passed": False,
                "validation_issues": [error_msg],
                "warnings": [],
            },
            
            # 🆕 HEALTH STATS
            "health_stats": self.get_stats(),
        }

    # -------- 🆕 ANÁLISE PRINCIPAL COM VALIDAÇÃO --------

    def analyze(
        self,
        current_snapshot: Optional[Dict[str, Any]] = None,
        *,
        event_epoch_ms: Optional[int] = None,
        window_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analisa orderbook e retorna evento padronizado.
        
        🔹 CORREÇÕES v1.4.0:
          - Valida snapshot antes de processar
          - Retorna evento com is_valid=False se dados ruins
          - Adiciona data_quality metrics
          - Usa fallback para dados antigos
          - Logging completo de erros
        
        Returns:
            Dict com evento. Sempre checar event['is_valid'] antes de usar!
        """
        # 1. Obtém snapshot (usa cache/fallback se necessário)
        snap = current_snapshot or self._fetch_orderbook(limit=self.ob_limit_fetch)
        
        # 2. 🆕 VALIDA SNAPSHOT
        if not snap:
            return self._create_invalid_event("fetch_failed", event_epoch_ms)
        
        is_valid, issues = self._validate_snapshot(snap)
        
        if not is_valid:
            error_msg = f"validation_failed: {', '.join(issues)}"
            logging.error(f"❌ Snapshot inválido: {error_msg}")
            return self._create_invalid_event(error_msg, event_epoch_ms)
        
        # 3. Extrai dados
        bids: List[Tuple[float, float]] = snap["bids"]
        asks: List[Tuple[float, float]] = snap["asks"]
        
        # 4. Timestamp
        ts_ms = None
        for key in ("E", "T"):
            v = snap.get(key)
            if isinstance(v, (int, float)) and v > 0:
                ts_ms = int(v)
                break
        if ts_ms is None:
            ts_ms = event_epoch_ms if event_epoch_ms is not None else self.tm.now_ms()
        
        tindex = self.tm.build_time_index(ts_ms, include_local=True, timespec="seconds")
        timestamp_ny = tindex.get("timestamp_ny")
        timestamp_utc = tindex.get("timestamp_utc")
        
        # 5. Calcula métricas
        sm = self._spread_and_depth(bids, asks)
        mid = sm.get("mid")
        bid_usd = float(sm.get("bid_depth_usd") or 0.0)
        ask_usd = float(sm.get("ask_depth_usd") or 0.0)
        imbalance, ratio, pressure = self._imbalance_ratio_pressure(bid_usd, ask_usd)
        
        bid_walls = self._detect_walls(bids, side="bid")
        ask_walls = self._detect_walls(asks, side="ask")
        
        iceberg, iceberg_score = self._iceberg_reload(
            self.prev_snapshot, 
            {"bids": bids, "asks": asks}
        )
        
        # 6. Market impact
        mi_buy_100k = _simulate_market_impact(asks[: self.top_n], 100_000.0, "buy", mid)
        mi_buy_1m = _simulate_market_impact(asks[: self.top_n], 1_000_000.0, "buy", mid)
        mi_sell_100k = _simulate_market_impact(bids[: self.top_n][::-1], 100_000.0, "sell", mid)
        mi_sell_1m = _simulate_market_impact(bids[: self.top_n][::-1], 1_000_000.0, "sell", mid)
        
        # 7. Histórico de spread
        if sm.get("spread_percent") is not None and sm["spread_percent"] >= 0:
            try:
                spread_bps = float(sm["spread_percent"]) * 100.0
                now_ms = ts_ms if ts_ms is not None else self.tm.now_ms()
                self.spread_history.append((int(now_ms), spread_bps))
            except Exception:
                pass
        
        # Remove spreads antigos
        try:
            cutoff_ms = (ts_ms or self.tm.now_ms()) - max(self.spread_avg_windows_min) * 60 * 1000
            self.spread_history = [(t, s) for (t, s) in self.spread_history if t >= cutoff_ms]
        except Exception:
            pass
        
        # 8. Rótulos e alertas
        resultado_da_batalha = "Equilíbrio"
        if imbalance is not None:
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
        if imbalance is not None and abs(imbalance) >= self.alert_threshold:
            alertas.append("Alerta de Liquidez (desequilíbrio)")
        if iceberg:
            alertas.append("Iceberg possivelmente recarregando")
        if sm.get("spread") is not None and sm["spread"] <= 0.5:
            alertas.append("Spread apertado")
        
        # 9. Depth summary
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
        
        # 10. Spread analysis
        spread_analysis: Dict[str, Any] = {
            "current_spread_bps": None,
            "spread_percentile": None,
            "tight_spread_duration_min": None,
            "spread_volatility": None,
        }
        
        try:
            current_bps = None
            if sm.get("spread_percent") is not None:
                current_bps = float(sm["spread_percent"]) * 100.0
                spread_analysis["current_spread_bps"] = round(current_bps, 4)
            
            for window_min in self.spread_avg_windows_min:
                window_ms = window_min * 60 * 1000
                now_ms = ts_ms if ts_ms is not None else self.tm.now_ms()
                values = [s for (t, s) in self.spread_history if (now_ms - t) <= window_ms]
                avg = float(np.mean(values)) if values else None
                
                if window_min >= 60 and window_min % 60 == 0:
                    hours = window_min // 60
                    key = f"avg_spread_{hours}h"
                else:
                    key = f"avg_spread_{window_min}m"
                
                spread_analysis[key] = round(avg, 4) if avg is not None else None
            
            if current_bps is not None:
                all_values = [s for (_, s) in self.spread_history]
                if all_values:
                    sorted_vals = sorted(all_values)
                    less = sum(1 for v in sorted_vals if v < current_bps)
                    pct = (less / len(sorted_vals)) * 100.0
                    spread_analysis["spread_percentile"] = round(pct, 1)
                    spread_analysis["spread_volatility"] = round(float(np.std(sorted_vals)), 4)
            
            if current_bps is not None:
                now_ms = ts_ms if ts_ms is not None else self.tm.now_ms()
                duration_ms = 0
                threshold = self.spread_tight_threshold_bps
                
                for (t, s) in reversed(self.spread_history):
                    if s <= threshold:
                        duration_ms = now_ms - t
                    else:
                        break
                
                spread_analysis["tight_spread_duration_min"] = round(duration_ms / 60000.0, 2) if duration_ms else 0.0
        
        except Exception as e:
            logging.debug(f"Erro em spread_analysis: {e}")
        
        # 11. Criticidade
        ratio_dom = None
        if ratio is not None:
            if ratio > 0:
                ratio_dom = ratio if ratio >= 1.0 else (1.0 / ratio)
            else:
                ratio_dom = float("inf")
        
        dominant_usd = max(bid_usd, ask_usd)
        is_extreme_imbalance = (imbalance is not None) and (abs(imbalance) >= ORDERBOOK_CRITICAL_IMBALANCE)
        is_extreme_ratio = (ratio_dom is not None) and (ratio_dom >= ORDERBOOK_MIN_RATIO_DOM)
        is_extreme_usd = dominant_usd >= ORDERBOOK_MIN_DOMINANT_USD
        
        is_critical = bool(
            is_extreme_imbalance and (is_extreme_ratio or is_extreme_usd) or
            (ratio_dom is not None and ratio_dom >= max(50.0, ORDERBOOK_MIN_RATIO_DOM))
        )
        
        if is_critical:
            side_dom = "ASKS" if (imbalance is not None and imbalance < 0) else "BIDS"
            alertas.append(f"🔴 DESEQUILÍBRIO CRÍTICO ({side_dom})")
        
        # 12. Descrição
        if imbalance is None:
            batalha = "INDISPONÍVEL"
            descricao = "Livro: dados insuficientes"
        elif imbalance < -0.05:
            batalha = "Oferta domina"
            descricao = f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"
        elif imbalance > 0.05:
            batalha = "Demanda domina"
            descricao = f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"
        else:
            batalha = "Equilíbrio"
            descricao = f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"
        
        # 13. 🆕 DATA QUALITY
        data_source = "live"
        age_seconds = 0.0
        
        if snap == self._cached_snapshot:
            data_source = "cache"
            age_seconds = time.time() - self._cache_timestamp
        elif snap == self._last_valid_snapshot:
            data_source = "stale"
            age_seconds = time.time() - self._last_valid_timestamp
        
        validation_warnings = []
        if issues:
            validation_warnings = issues
        
        # 14. Log
        if imbalance is not None and ratio is not None:
            logging.debug(
                f"📊 OrderBook OK: bid=${bid_usd:,.2f}, ask=${ask_usd:,.2f}, "
                f"Δ={imbalance:+.4f}, ratio={ratio:.4f}, source={data_source}"
            )
        
        # 15. Monta evento
        event: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tipo_evento": "OrderBook",
            "ativo": self.symbol,
            
            # 🆕 FLAGS DE VALIDADE
            "is_valid": True,
            "should_skip": False,
            
            "descricao": descricao,
            "resultado_da_batalha": batalha,
            
            "imbalance": round(imbalance, 4) if imbalance is not None else None,
            "volume_ratio": round(ratio, 4) if ratio not in (None, float("inf")) else None,
            "pressure": round(pressure, 4) if pressure is not None else None,
            
            "spread_metrics": sm,
            "alertas_liquidez": alertas,
            "iceberg_reloaded": bool(iceberg),
            "iceberg_score": iceberg_score,
            "walls": {"bids": bid_walls[:3], "asks": ask_walls[:3]},
            
            "market_impact_buy": {"100k": mi_buy_100k, "1M": mi_buy_1m},
            "market_impact_sell": {"100k": mi_sell_100k, "1M": mi_sell_1m},
            
            "top_n": self.top_n,
            "ob_limit": self.ob_limit_fetch,
            
            "timestamps": {
                "exchange_ms": ts_ms,
                "timestamp_ny": timestamp_ny,
                "timestamp_utc": timestamp_utc,
            },
            
            "source": {
                "exchange": "binance_futures",
                "endpoint": "fapi/v1/depth",
                "symbol": self.symbol,
            },
            
            "labels": {
                "dominant_label": resultado_da_batalha,
                "note": "Rótulo baseado no livro (estoque de liquidez), não na fita executada (delta).",
            },
            
            "order_book_depth": depth_summary,
            "spread_analysis": spread_analysis,
            
            "severity": "CRITICAL" if is_critical else "INFO",
            
            "critical_flags": {
                "is_critical": is_critical,
                "abs_imbalance": round(abs(imbalance), 4) if imbalance is not None else None,
                "ratio_dom": (round(ratio_dom, 4) if (ratio_dom not in (None, float("inf"))) else ratio_dom),
                "dominant_usd": round(dominant_usd, 2),
                "thresholds": {
                    "ORDERBOOK_CRITICAL_IMBALANCE": ORDERBOOK_CRITICAL_IMBALANCE,
                    "ORDERBOOK_MIN_DOMINANT_USD": ORDERBOOK_MIN_DOMINANT_USD,
                    "ORDERBOOK_MIN_RATIO_DOM": ORDERBOOK_MIN_RATIO_DOM,
                },
            },
            
            "orderbook_data": {
                "mid": sm["mid"],
                "spread": sm["spread"],
                "spread_percent": sm["spread_percent"],
                "bid_depth_usd": bid_usd,
                "ask_depth_usd": ask_usd,
                "imbalance": round(imbalance, 4) if imbalance is not None else None,
                "volume_ratio": round(ratio, 4) if ratio not in (None, float("inf")) else None,
                "pressure": round(pressure, 4) if pressure is not None else None,
            },
            
            # 🆕 DATA QUALITY METRICS
            "data_quality": {
                "is_valid": True,
                "data_source": data_source,
                "age_seconds": round(age_seconds, 2),
                "validation_passed": True,
                "validation_issues": [],
                "warnings": validation_warnings,
            },
            
            # 🆕 HEALTH STATS
            "health_stats": self.get_stats(),
        }
        
        # 16. Atualiza memória
        self.prev_snapshot = {"bids": bids, "asks": asks}
        self.last_event_ts_ms = ts_ms
        
        return event

    # -------- 🆕 HEALTH MONITORING --------

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance e health."""
        error_rate = 100 * self._fetch_errors / max(1, self._total_fetches)
        validation_failure_rate = 100 * self._validation_failures / max(1, self._total_fetches)
        cache_hit_rate = 100 * self._cache_hits / max(1, self._total_fetches)
        
        return {
            "total_fetches": self._total_fetches,
            "fetch_errors": self._fetch_errors,
            "validation_failures": self._validation_failures,
            "cache_hits": self._cache_hits,
            "stale_data_uses": self._stale_data_uses,
            
            "error_rate_pct": round(error_rate, 2),
            "validation_failure_rate_pct": round(validation_failure_rate, 2),
            "cache_hit_rate_pct": round(cache_hit_rate, 2),
            
            "has_cached_data": self._cached_snapshot is not None,
            "has_stale_data": self._last_valid_snapshot is not None,
            
            "cache_age_seconds": round(time.time() - self._cache_timestamp, 2) if self._cache_timestamp > 0 else None,
            "stale_age_seconds": round(time.time() - self._last_valid_timestamp, 2) if self._last_valid_timestamp > 0 else None,
            
            "requests_last_min": len(self._request_times),
            "rate_limit_threshold": self._rate_limit_threshold,
        }
    
    def reset_stats(self):
        """Reseta contadores de estatísticas."""
        self._fetch_errors = 0
        self._total_fetches = 0
        self._validation_failures = 0
        self._cache_hits = 0
        self._stale_data_uses = 0
        logging.info("📊 Estatísticas resetadas")

    # -------- Shims de compatibilidade --------
    
    def analyze_order_book(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze(*args, **kwargs)

    def analyzeOrderBook(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze(*args, **kwargs)

    def analyze_orderbook(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze(*args, **kwargs)


# -------- TESTE --------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
    )
    
    print("\n" + "="*80)
    print("🧪 TESTE DE ORDERBOOK ANALYZER v1.4.0")
    print("="*80 + "\n")
    
    oba = OrderBookAnalyzer(
        symbol="BTCUSDT",
        cache_ttl_seconds=2.0,
        max_stale_seconds=30.0,
        rate_limit_threshold=10,
    )
    
    # Teste 1: Fetch normal
    print("📡 Teste 1: Fetch normal...")
    evt = oba.analyze()
    
    print(f"\n  ✓ is_valid: {evt.get('is_valid')}")
    print(f"  ✓ should_skip: {evt.get('should_skip')}")
    print(f"  ✓ Severity: {evt.get('severity')}")
    print(f"  ✓ Resultado: {evt.get('resultado_da_batalha')}")
    print(f"  ✓ Bid Depth: ${evt.get('orderbook_data', {}).get('bid_depth_usd', 0):,.2f}")
    print(f"  ✓ Ask Depth: ${evt.get('orderbook_data', {}).get('ask_depth_usd', 0):,.2f}")
    print(f"  ✓ Imbalance: {evt.get('orderbook_data', {}).get('imbalance', 0):+.4f}")
    print(f"  ✓ Data Source: {evt.get('data_quality', {}).get('data_source')}")
    print(f"  ✓ Alertas: {evt.get('alertas_liquidez')}")
    
    # Teste 2: Cache hit
    print("\n📦 Teste 2: Cache hit (imediato)...")
    time.sleep(0.1)
    evt2 = oba.analyze()
    print(f"  ✓ Data Source: {evt2.get('data_quality', {}).get('data_source')}")
    print(f"  ✓ Age: {evt2.get('data_quality', {}).get('age_seconds')}s")
    
    # Teste 3: Stats
    print("\n📊 Teste 3: Health Stats...")
    stats = oba.get_stats()
    for key, val in stats.items():
        print(f"  • {key}: {val}")
    
    # Teste 4: Validação com snapshot inválido
    print("\n❌ Teste 4: Snapshot inválido (bids vazios)...")
    invalid_snap = {"bids": [], "asks": [(50000.0, 1.0)]}
    evt_invalid = oba.analyze(current_snapshot=invalid_snap)
    print(f"  ✓ is_valid: {evt_invalid.get('is_valid')}")
    print(f"  ✓ should_skip: {evt_invalid.get('should_skip')}")
    print(f"  ✓ Erro: {evt_invalid.get('erro')}")
    print(f"  ✓ Issues: {evt_invalid.get('data_quality', {}).get('validation_issues')}")
    
    print("\n" + "="*80)
    print("✅ TESTES CONCLUÍDOS")
    print("="*80 + "\n")