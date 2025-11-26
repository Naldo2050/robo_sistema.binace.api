# orderbook_analyzer.py v2.1.0 - ASYNC COM SESSÃO HTTP REUTILIZÁVEL
"""
OrderBook Analyzer para Binance Futures com validação robusta e fetch assíncrono.

🔹 v2.1.0 (PERFORMANCE: SESSÃO REUTILIZÁVEL):
  ✅ Reintroduz ClientSession reutilizável por OrderBookAnalyzer
  ✅ Permite injetar uma sessão externa (parâmetro `session=...`)
  ✅ Evita criar/fechar sessão + handshake TCP/SSL a cada requisição
  ✅ Continua compatível com asyncio.run(...), desde que o loop seja bem gerenciado

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
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import aiohttp  # ✅ async HTTP client
import numpy as np

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

from time_manager import TimeManager

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

# ===== 🆕 VALIDAÇÃO DE IDADE MÁXIMA =====
ORDERBOOK_MAX_AGE_MS = 60000  # ✅ 60 segundos máximo

SCHEMA_VERSION = "2.1.0"  # ✅ ATUALIZADO


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
    }


# ===== ANALYZER =====
class OrderBookAnalyzer:
    """
    Analisador de Order Book para Binance Futures com validação robusta e fetch assíncrono.

    ✅ v2.1.0: Sessão HTTP reutilizável por analyzer (ou injetada), sem overhead por chamada
    ✅ v2.0.0: 100% async, zero bloqueio
    ✅ v1.6.0: Validação rigorosa, não modifica dados, fallback conservador
    """

    def __init__(
        self,
        symbol: str,
        liquidity_flow_alert_percentage: float = 0.40,
        wall_std_dev_factor: float = 3.0,
        top_n_levels: int = 20,
        ob_limit_fetch: int = 100,
        time_manager: Optional[TimeManager] = None,
        cache_ttl_seconds: float = None,
        max_stale_seconds: float = None,
        rate_limit_threshold: int = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.symbol = symbol.upper()
        self.alert_threshold = float(liquidity_flow_alert_percentage)
        self.wall_std = float(wall_std_dev_factor)
        self.top_n = int(top_n_levels)
        self.ob_limit_fetch = int(ob_limit_fetch)
        self.tz_ny = ZoneInfo("America/New_York") if ZoneInfo else None
        self.tm = time_manager or TimeManager()

        # Config
        self.cache_ttl_seconds = cache_ttl_seconds if cache_ttl_seconds is not None else ORDERBOOK_CACHE_TTL
        self.max_stale_seconds = max_stale_seconds if max_stale_seconds is not None else ORDERBOOK_MAX_STALE
        self.rate_limit_threshold = rate_limit_threshold if rate_limit_threshold is not None else ORDERBOOK_MAX_REQUESTS_PER_MIN

        # Sessão HTTP (pode ser injetada ou criada lazy internamente)
        self._session: Optional[aiohttp.ClientSession] = session
        # True se a sessão foi criada pelo próprio Analyzer (nesse caso, ele fecha)
        self._owns_session: bool = session is None

        # Cache
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cache_timestamp: float = 0.0

        # Fallback
        self._last_valid_snapshot: Optional[Dict[str, Any]] = None
        self._last_valid_timestamp: float = 0.0
        self._last_valid_exchange_ts: Optional[int] = None

        # Rate limiting
        self._request_times: List[float] = []

        # Stats
        self._fetch_errors = 0
        self._total_fetches = 0
        self._validation_failures = 0
        self._cache_hits = 0
        self._stale_data_uses = 0
        self._emergency_uses = 0
        self._old_data_rejected = 0

        # Config adicionais
        self.depth_levels: List[int] = list(ORDER_BOOK_DEPTH_LEVELS)
        self.spread_tight_threshold_bps: float = float(SPREAD_TIGHT_THRESHOLD_BPS)
        self.spread_avg_windows_min: List[int] = list(SPREAD_AVG_WINDOWS_MIN)
        self.spread_history: List[Tuple[int, float]] = []

        # Memória para iceberg
        self.prev_snapshot: Optional[Dict[str, Any]] = None
        self.last_event_ts_ms: Optional[int] = None

        logging.info(
            "✅ OrderBook Analyzer v%s inicializado (ASYNC) | "
            "Symbol: %s | Alert: %.0f%% | Wall STD: %.1fx | "
            "Top N: %s | Cache TTL: %.1fs | Max Stale: %.1fs | "
            "Rate Limit: %s req/min | Config loaded: %s",
            SCHEMA_VERSION,
            self.symbol,
            self.alert_threshold * 100,
            self.wall_std,
            self.top_n,
            self.cache_ttl_seconds,
            self.max_stale_seconds,
            self.rate_limit_threshold,
            "✅" if CONFIG_LOADED else "❌ (usando defaults)"
        )

    # ===== CLOSE =====
    async def close(self):
        """
        Fecha a ClientSession se ela tiver sido criada internamente.
        Se uma sessão externa foi injetada no __init__, o chamador é
        responsável por fechá-la.
        """
        try:
            if self._owns_session and self._session and not self._session.closed:
                await self._session.close()
        except Exception:
            pass
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Retorna uma sessão HTTP reutilizável.

        - Se o usuário injetou uma sessão no __init__, ela é usada.
        - Caso contrário, criamos uma ClientSession própria (lazy) e
          reaproveitamos até close().
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=ORDERBOOK_REQUEST_TIMEOUT,
                connect=5.0,
                sock_read=ORDERBOOK_REQUEST_TIMEOUT,
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
        return self._session

    # ===== ✅ VALIDAÇÃO (NÃO MUDOU, COM PATCH 3) =====
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

        # 3. ✅ VALIDAÇÃO DE IDADE
        if max_age_ms is None:
            max_age_ms = ORDERBOOK_MAX_AGE_MS

        exchange_ts = None
        for key in ("E", "T"):
            v = snap.get(key)
            if isinstance(v, (int, float)) and v > 0:
                exchange_ts = int(v)
                break

        if exchange_ts is not None:
            now_ms = self.tm.now_ms()
            age_ms = now_ms - exchange_ts

            if age_ms < 0:
                issues.append(f"timestamp futuro! (age={age_ms}ms)")
                return False, issues, converted_snap

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
            if spread_pct > 10:
                issues.append(f"spread absurdo ({spread_pct:.2f}%)")
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

            min_depth = ORDERBOOK_MIN_DEPTH_USD

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
                if ORDERBOOK_ALLOW_PARTIAL:
                    pct_bid = (bid_depth_usd / min_depth) * 100
                    pct_ask = (ask_depth_usd / min_depth) * 100

                    if pct_bid >= 50.0 and pct_ask >= 50.0:
                        logging.warning(
                            f"⚠️ Liquidez baixa mas aceita "
                            f"(bid={pct_bid:.0f}%, ask={pct_ask:.0f}% do mínimo)"
                        )
                    else:
                        issues.append(
                            f"liquidez < 50% do mínimo (rejeitado mesmo com ALLOW_PARTIAL)"
                        )
                        return False, issues, converted_snap
                else:
                    return False, issues, converted_snap

            return True, issues, converted_snap

        except (IndexError, ValueError, TypeError) as e:
            issues.append(f"erro ao validar dados: {e}")
            return False, issues, converted_snap

    # ===== RATE LIMITING =====
    def _check_rate_limit(self) -> bool:
        """Verifica se está próximo do rate limit."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        buffer = 1
        return len(self._request_times) >= (self.rate_limit_threshold - buffer)

    def _register_request(self):
        """Registra request para tracking."""
        self._request_times.append(time.time())

    # ===== ✅ FETCH ASYNC v2.1.0 (SESSÃO REUTILIZÁVEL, PATCHES 1 e 2) =====
    async def _fetch_orderbook(
        self,
        limit: Optional[int] = None,
        use_cache: bool = True,
        allow_stale: bool = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Busca orderbook com retry, validação e fallback conservador.

        ✅ v2.1.0:
        - Usa uma ClientSession reutilizável por OrderBookAnalyzer (ou injetada)
        - Evita overhead de criar/fechar sessão + handshake a cada chamada
        - Continua compatível com asyncio.run(...) (desde que o loop não seja recriado o tempo todo)

        ✅ v2.0.0:
        - Usa aiohttp (não bloqueia event loop)
        - await asyncio.sleep() ao invés de time.sleep()

        ✅ Patches:
        - Verificação de sessão antes de cada uso
        - Timeout com cancelamento explícito da request (evita requests órfãs)

        Returns:
            Snapshot válido (convertido) ou None
        """
        if allow_stale is None:
            allow_stale = ORDERBOOK_USE_FALLBACK

        self._total_fetches += 1

        # 1. CACHE
        if use_cache and self._cached_snapshot is not None:
            cache_age = time.time() - self._cache_timestamp
            if cache_age < self.cache_ttl_seconds:
                self._cache_hits += 1
                logging.debug(f"📦 Cache hit (age={cache_age:.2f}s)")
                return self._cached_snapshot

        # 2. RATE LIMITING
        if self._check_rate_limit():
            wait_time = max(1.0, ORDERBOOK_RETRY_DELAY * 0.5)
            logging.warning(
                f"⏳ Rate limit preventivo - aguardando {wait_time}s..."
            )
            await asyncio.sleep(wait_time)  # ✅ ASYNC

        # 3. FETCH COM RETRY
        lim = limit or self.ob_limit_fetch
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit={lim}"

        max_retries = ORDERBOOK_MAX_RETRIES
        base_delay = ORDERBOOK_RETRY_DELAY

        for attempt in range(max_retries):
            task: Optional[asyncio.Task] = None

            try:
                session = await self._get_session()

                # Verifica se foi fechada externamente
                if session.closed:
                    logging.warning("⚠️ Sessão fechada, recriando...")
                    self._session = None
                    session = await self._get_session()

                self._register_request()

                logging.debug(
                    f"📡 Fetching orderbook (attempt {attempt + 1}/{max_retries})..."
                )

                # Cria task para poder cancelar
                task = asyncio.create_task(
                    session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=5.0)  # Timeout no nível da request
                    )
                )

                try:
                    # Timeout com cancelamento explícito
                    r = await asyncio.wait_for(task, timeout=5.0)

                    async with r:
                        # Rate limiting
                        if r.status == 429:
                            retry_after = int(r.headers.get('Retry-After', 60))
                            self._fetch_errors += 1

                            logging.error(
                                f"🚫 RATE LIMIT (429) - Retry após {retry_after}s"
                            )

                            if attempt < max_retries - 1:
                                await asyncio.sleep(min(retry_after, base_delay * 3))
                                continue
                            else:
                                break

                        # Outros erros HTTP
                        if r.status != 200:
                            self._fetch_errors += 1
                            text = await r.text()
                            logging.error(
                                f"❌ HTTP {r.status}: {text[:200]}"
                            )
                            if attempt < max_retries - 1:
                                await asyncio.sleep(base_delay * (attempt + 1))
                                continue
                            else:
                                break

                        # ✅ PARSE JSON ASYNC
                        data = await r.json()

                        # ✅ VALIDA E CONVERTE (síncrono, não bloqueia)
                        is_valid, issues, converted = self._validate_snapshot(data)

                        if not is_valid:
                            self._validation_failures += 1

                            logging.error(
                                f"❌ Snapshot inválido (attempt {attempt + 1}): {', '.join(issues)}"
                            )

                            if attempt < max_retries - 1:
                                delay = base_delay * (attempt + 1)
                                logging.debug(f"  Retry em {delay:.1f}s...")
                                await asyncio.sleep(delay)
                                continue
                            else:
                                break

                        # ✅ SUCESSO - Salva snapshot CONVERTIDO
                        self._cached_snapshot = converted
                        self._cache_timestamp = time.time()

                        self._last_valid_snapshot = converted.copy()
                        self._last_valid_timestamp = time.time()

                        # ✅ SALVA TIMESTAMP ORIGINAL
                        exchange_ts = converted.get("E") or converted.get("T")
                        if exchange_ts:
                            self._last_valid_exchange_ts = int(exchange_ts)

                        logging.debug(
                            f"✅ Orderbook obtido: "
                            f"{len(converted['bids'])} bids, {len(converted['asks'])} asks"
                        )

                        return converted

                except asyncio.TimeoutError:
                    # Cancela request pendente
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # Repassa para o bloco externo tratar retry
                    raise

            except asyncio.TimeoutError:
                self._fetch_errors += 1
                logging.error(f"⏱️ Timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (attempt + 1))

            except aiohttp.ClientError as e:
                self._fetch_errors += 1
                logging.error(f"🌐 Client error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (attempt + 1))

            except Exception as e:
                self._fetch_errors += 1
                logging.error(f"💥 Unexpected error (attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (attempt + 1))

            finally:
                # Garante cancelamento mesmo em exceção
                if task and not task.done():
                    task.cancel()

        # 5. ✅ FALLBACK CONSERVADOR
        if allow_stale and self._last_valid_snapshot is not None:
            age = time.time() - self._last_valid_timestamp

            if age < ORDERBOOK_FALLBACK_MAX_AGE:
                # ✅ VALIDA IDADE DO DADO ORIGINAL
                if self._last_valid_exchange_ts:
                    now_ms = self.tm.now_ms()
                    data_age_ms = now_ms - self._last_valid_exchange_ts

                    if data_age_ms > ORDERBOOK_MAX_AGE_MS:
                        self._old_data_rejected += 1
                        logging.error(
                            f"❌ Snapshot fallback muito antigo "
                            f"(data_age={data_age_ms}ms > {ORDERBOOK_MAX_AGE_MS}ms)"
                        )
                        return None

                self._stale_data_uses += 1

                logging.warning(
                    f"⚠️ Usando snapshot antigo (cache_age={age:.1f}s)"
                )

                return self._last_valid_snapshot.copy()
            else:
                logging.error(
                    f"❌ Snapshot muito velho ({age:.1f}s > {ORDERBOOK_FALLBACK_MAX_AGE}s)"
                )

        # 💀 FALHA TOTAL
        error_rate = 100 * self._fetch_errors / max(1, self._total_fetches)

        logging.error(
            f"💀 FALHA ao obter orderbook após {max_retries} tentativas "
            f"(erro: {error_rate:.1f}%)"
        )

        return None

    # ===== MÉTRICAS (NÃO MUDARAM) =====
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

    def _detect_walls(
        self,
        side_levels: List[Tuple[float, float]],
        side: str
    ) -> List[Dict[str, Any]]:
        """Detecta paredes de liquidez."""
        if not side_levels:
            return []

        levels = side_levels[:self.top_n]
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
        self,
        prev: Optional[Dict[str, Any]],
        curr: Dict[str, Any],
        tol: float = 0.75
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

    # ===== EVENTO INVÁLIDO =====
    def _create_invalid_event(
        self,
        error_msg: str,
        ts_ms: Optional[int] = None,
        severity: str = "ERROR",
    ) -> Dict[str, Any]:
        """Cria evento marcado como INVÁLIDO."""
        if ts_ms is None:
            ts_ms = self.tm.now_ms()

        tindex = self.tm.build_time_index(ts_ms, include_local=True, timespec="seconds")

        return {
            "schema_version": SCHEMA_VERSION,
            "tipo_evento": "OrderBook",
            "ativo": self.symbol,

            "is_valid": False,
            "should_skip": True,
            "emergency_mode": False,
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

            "data_quality": {
                "is_valid": False,
                "data_source": "error",
                "age_seconds": 0.0,
                "validation_passed": False,
                "validation_issues": [error_msg],
                "warnings": [],
            },

            "health_stats": self.get_stats(),
        }

    # ===== ✅ ANÁLISE PRINCIPAL ASYNC v2.0.0 =====
    async def analyze(
        self,
        current_snapshot: Optional[Dict[str, Any]] = None,
        *,
        event_epoch_ms: Optional[int] = None,
        window_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analisa orderbook e retorna evento padronizado.

        ✅ v2.0.0:
        - async (use: await oba.analyze())

        Returns:
            Dict com evento. Sempre checar event['is_valid'] antes de usar!
        """
        # 1. Obtém snapshot (já convertido) - ✅ ASYNC
        snap = current_snapshot or await self._fetch_orderbook(limit=self.ob_limit_fetch)

        if not snap:
            return self._create_invalid_event("fetch_failed", event_epoch_ms)

        # 2. ✅ SNAPSHOT JÁ DEVE ESTAR CONVERTIDO
        if current_snapshot is not None:
            is_valid, issues, converted = self._validate_snapshot(snap)

            if not is_valid:
                error_msg = f"validation_failed: {', '.join(issues)}"
                logging.error(f"❌ Snapshot externo inválido: {error_msg}")
                return self._create_invalid_event(error_msg, event_epoch_ms)

            snap = converted

        # 3. ✅ GARANTE QUE ESTÁ CONVERTIDO
        bids: List[Tuple[float, float]] = snap.get("bids", [])
        asks: List[Tuple[float, float]] = snap.get("asks", [])

        if not bids or not asks:
            return self._create_invalid_event("empty_orderbook", event_epoch_ms)

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

        # ✅ PROTEÇÃO TOTAL - Rejeita se dados parciais
        imbalance, ratio, pressure = self._imbalance_ratio_pressure(bid_usd, ask_usd)

        if imbalance is None:
            return self._create_invalid_event(
                f"partial_data_rejected: bid=${bid_usd:.2f}, ask=${ask_usd:.2f}",
                ts_ms
            )

        bid_walls = self._detect_walls(bids, side="bid")
        ask_walls = self._detect_walls(asks, side="ask")

        iceberg, iceberg_score = self._iceberg_reload(
            self.prev_snapshot,
            {"bids": bids, "asks": asks}
        )

        # 6. Market impact
        mi_buy_100k = _simulate_market_impact(asks[:self.top_n], 100_000.0, "buy", mid)
        mi_buy_1m = _simulate_market_impact(asks[:self.top_n], 1_000_000.0, "buy", mid)
        mi_sell_100k = _simulate_market_impact(bids[:self.top_n][::-1], 100_000.0, "sell", mid)
        mi_sell_1m = _simulate_market_impact(bids[:self.top_n][::-1], 1_000_000.0, "sell", mid)

        # 7. Histórico de spread
        if sm.get("spread_percent") is not None and sm["spread_percent"] >= 0:
            try:
                spread_bps = float(sm["spread_percent"]) * 100.0
                now_ms = ts_ms if ts_ms is not None else self.tm.now_ms()
                self.spread_history.append((int(now_ms), spread_bps))
            except Exception:
                pass

        try:
            cutoff_ms = (ts_ms or self.tm.now_ms()) - max(self.spread_avg_windows_min) * 60 * 1000
            self.spread_history = [(t, s) for (t, s) in self.spread_history if t >= cutoff_ms]
        except Exception:
            pass

        # 8. Rótulos e alertas
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
        is_extreme_imbalance = abs(imbalance) >= ORDERBOOK_CRITICAL_IMBALANCE
        is_extreme_ratio = (ratio_dom is not None) and (ratio_dom >= ORDERBOOK_MIN_RATIO_DOM)
        is_extreme_usd = dominant_usd >= ORDERBOOK_MIN_DOMINANT_USD

        is_critical = bool(
            is_extreme_imbalance and (is_extreme_ratio or is_extreme_usd) or
            (ratio_dom is not None and ratio_dom >= max(50.0, ORDERBOOK_MIN_RATIO_DOM))
        )

        if is_critical:
            side_dom = "ASKS" if imbalance < 0 else "BIDS"
            alertas.append(f"🔴 DESEQUILÍBRIO CRÍTICO ({side_dom})")

        # 12. Descrição
        if imbalance < -0.05:
            batalha = "Oferta domina"
            descricao = f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"
        elif imbalance > 0.05:
            batalha = "Demanda domina"
            descricao = f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"
        else:
            batalha = "Equilíbrio"
            descricao = f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"

        # 13. ✅ DATA QUALITY
        data_source = "live"
        age_seconds = 0.0

        if snap == self._cached_snapshot:
            data_source = "cache"
            age_seconds = time.time() - self._cache_timestamp
        elif snap == self._last_valid_snapshot:
            data_source = "stale"
            age_seconds = time.time() - self._last_valid_timestamp

        # 14. Log
        logging.info(
            f"📊 OrderBook OK - Janela #{window_id if window_id else '?'}: "
            f"bid=${bid_usd:,.0f}, ask=${ask_usd:,.0f}"
        )

        # 15. Monta evento
        event: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tipo_evento": "OrderBook",
            "ativo": self.symbol,

            "is_valid": True,
            "should_skip": False,
            "emergency_mode": False,

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
                "note": "Rótulo baseado no livro (estoque de liquidez).",
            },

            "order_book_depth": depth_summary,
            "spread_analysis": spread_analysis,

            "severity": "CRITICAL" if is_critical else "INFO",

            "critical_flags": {
                "is_critical": is_critical,
                "abs_imbalance": round(abs(imbalance), 4),
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
                "imbalance": round(imbalance, 4),
                "volume_ratio": round(ratio, 4) if ratio not in (None, float("inf")) else None,
                "pressure": round(pressure, 4),
            },

            "data_quality": {
                "is_valid": True,
                "data_source": data_source,
                "age_seconds": round(age_seconds, 2),
                "validation_passed": True,
                "validation_issues": [],
                "warnings": [],
                "emergency_mode": False,
            },

            "health_stats": self.get_stats(),
        }

        # 16. Atualiza memória
        self.prev_snapshot = {"bids": bids, "asks": asks}
        self.last_event_ts_ms = ts_ms

        return event

    # ===== HEALTH MONITORING =====
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
            "emergency_uses": self._emergency_uses,
            "old_data_rejected": self._old_data_rejected,

            "error_rate_pct": round(error_rate, 2),
            "validation_failure_rate_pct": round(validation_failure_rate, 2),
            "cache_hit_rate_pct": round(cache_hit_rate, 2),

            "has_cached_data": self._cached_snapshot is not None,
            "has_stale_data": self._last_valid_snapshot is not None,

            "cache_age_seconds": round(time.time() - self._cache_timestamp, 2) if self._cache_timestamp > 0 else None,
            "stale_age_seconds": round(time.time() - self._last_valid_timestamp, 2) if self._last_valid_timestamp > 0 else None,

            "requests_last_min": len(self._request_times),
            "rate_limit_threshold": self.rate_limit_threshold,

            "config": {
                "cache_ttl": self.cache_ttl_seconds,
                "max_stale": self.max_stale_seconds,
                "emergency_mode": ORDERBOOK_EMERGENCY_MODE,
                "allow_partial": ORDERBOOK_ALLOW_PARTIAL,
                "min_depth_usd": ORDERBOOK_MIN_DEPTH_USD,
                "max_age_ms": ORDERBOOK_MAX_AGE_MS,
                "config_loaded": CONFIG_LOADED,
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
        logging.info("📊 Estatísticas resetadas")

    # ===== SHIMS DE COMPATIBILIDADE =====
    async def analyze_order_book(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.analyze(*args, **kwargs)

    async def analyzeOrderBook(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.analyze(*args, **kwargs)

    async def analyze_orderbook(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.analyze(*args, **kwargs)


# ===== TESTE =====
async def main_test():
    """Função de teste async"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
    )

    print("\n" + "=" * 80)
    print("🧪 TESTE DE ORDERBOOK ANALYZER v2.1.0 (ASYNC, SESSÃO REUTILIZÁVEL)")
    print("=" * 80 + "\n")

    oba = OrderBookAnalyzer(symbol="BTCUSDT")

    try:
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

    finally:
        # Fecha sessão apenas se criada internamente
        await oba.close()

    print("\n" + "=" * 80)
    print("✅ TESTES CONCLUÍDOS - ORDERBOOK v2.1.0 ASYNC (SESSÃO REUTILIZÁVEL)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # ✅ Roda função async
    asyncio.run(main_test())