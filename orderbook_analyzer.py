# orderbook_analyzer.py (VERSÃO CORRIGIDA COMPLETA - v1.3.1)
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

# Importa parâmetros adicionais de configuração (inclui limiares CRÍTICOS)
try:
    from config import (
        ORDER_BOOK_DEPTH_LEVELS,
        SPREAD_TIGHT_THRESHOLD_BPS,
        SPREAD_AVG_WINDOWS_MIN,
        # novos limiares p/ elevação a CRITICAL
        ORDERBOOK_CRITICAL_IMBALANCE,
        ORDERBOOK_MIN_DOMINANT_USD,
        ORDERBOOK_MIN_RATIO_DOM,
    )
except Exception:
    # Valores padrão se não definidos
    ORDER_BOOK_DEPTH_LEVELS = [1, 5, 10, 25]
    SPREAD_TIGHT_THRESHOLD_BPS = 0.2
    SPREAD_AVG_WINDOWS_MIN = [60, 1440]
    ORDERBOOK_CRITICAL_IMBALANCE = 0.95
    ORDERBOOK_MIN_DOMINANT_USD = 2_000_000.0
    ORDERBOOK_MIN_RATIO_DOM = 20.0

SCHEMA_VERSION = "1.3.1"


# ------------------------- Utils -------------------------

def _to_float_list(levels: List[List[str]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for lv in levels or []:
        try:
            p = float(lv[0]); q = float(lv[1])
            out.append((p, q))
        except Exception:
            continue
    return out


def _sum_depth_usd(levels: List[Tuple[float, float]], top_n: int) -> float:
    if not levels:
        return 0.0
    arr = levels[: max(1, top_n)]
    return float(sum(p * q for p, q in arr))


def _simulate_market_impact(
    levels: List[Tuple[float, float]], usd_amount: float, side: str, mid: Optional[float]
) -> Dict[str, Any]:
    """
    Caminhada determinística no livro:
      - BUY: anda pelos ASKS (asc.)
      - SELL: anda pelos BIDS (desc.) -> passe levels[::-1]
    Retorna deslocamento terminal vs mid em USD e bps.
    """
    if not levels or usd_amount <= 0:
        return {"usd": usd_amount, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None}

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
    Extrai métricas do livro (Binance Futures):
      - spread, mid, profundidade USD (Top-N)
      - imbalance / ratio / pressure
      - paredes (walls) por desvio-padrão
      - iceberg reload (heurístico)
      - market impact (100k / 1M) determinístico
      - promoção a CRITICAL quando há desequilíbrio extremo
    
    🔹 CORREÇÕES v1.3.1:
      - Retry logic com backoff exponencial
      - Cache de dados para evitar rate limiting
      - Logging detalhado de erros
      - Retorno completo mesmo em caso de falha
      - Fix de formatação de log
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
    ):
        self.symbol = symbol.upper()
        self.alert_threshold = float(liquidity_flow_alert_percentage)
        self.wall_std = float(wall_std_dev_factor)
        self.top_n = int(top_n_levels)
        self.ob_limit_fetch = int(ob_limit_fetch)
        self.tz_ny = ZoneInfo("America/New_York") if ZoneInfo else None
        self.tm = time_manager or TimeManager()

        # Cache de order book
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cached_snapshot: Optional[Dict[str, Any]] = None
        self._cache_timestamp: float = 0.0
        
        # Contadores de erro
        self._fetch_errors = 0
        self._total_fetches = 0
        self._last_error_logged = 0.0

        # Configurações adicionais
        self.depth_levels: List[int] = list(ORDER_BOOK_DEPTH_LEVELS)
        self.spread_tight_threshold_bps: float = float(SPREAD_TIGHT_THRESHOLD_BPS)
        self.spread_avg_windows_min: List[int] = list(SPREAD_AVG_WINDOWS_MIN)
        self.spread_history: List[Tuple[int, float]] = []

        # memória leve para heurística de recarga
        self.prev_snapshot: Optional[Dict[str, Any]] = None
        self.last_event_ts_ms: Optional[int] = None

        logging.info(
            "✅ OrderBook Analyzer inicializado para %s | Alerta fluxo: %s | Wall STD: %s | Top N: %s | Cache TTL: %.1fs",
            self.symbol,
            f"{self.alert_threshold*100:.0f}%",
            f"{self.wall_std:.1f}x",
            self.top_n,
            self.cache_ttl_seconds,
        )

    # -------- Data (COM RETRY E CACHE) --------

    def _fetch_orderbook(self, limit: Optional[int] = None, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Busca order book da Binance com retry logic e cache.
        """
        self._total_fetches += 1
        
        # Verifica cache
        if use_cache and self._cached_snapshot is not None:
            cache_age = time.time() - self._cache_timestamp
            if cache_age < self.cache_ttl_seconds:
                logging.debug(f"📦 Usando cache de order book (age: {cache_age:.2f}s)")
                return self._cached_snapshot
        
        lim = limit or self.ob_limit_fetch
        url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit={lim}"
        
        max_retries = 3
        base_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                timeout = 5.0 if attempt == 0 else 10.0
                
                logging.debug(f"📡 Buscando order book (tentativa {attempt + 1}/{max_retries})...")
                
                r = requests.get(url, timeout=timeout)
                
                # Detecta rate limiting
                if r.status_code == 429:
                    retry_after = int(r.headers.get('Retry-After', 60))
                    logging.warning(f"⚠️ RATE LIMIT detectado! Retry após {retry_after}s")
                    self._fetch_errors += 1
                    
                    if attempt < max_retries - 1:
                        time.sleep(retry_after)
                        continue
                    else:
                        return None
                
                r.raise_for_status()
                data = r.json()
                
                # Valida estrutura básica
                if not isinstance(data, dict):
                    logging.error(f"❌ Resposta inválida da API (não é dict): {type(data)}")
                    continue
                
                if 'bids' not in data or 'asks' not in data:
                    logging.error(f"❌ Resposta sem bids/asks: {list(data.keys())}")
                    continue
                
                # Parse dos dados
                parsed = {
                    "lastUpdateId": data.get("lastUpdateId"),
                    "E": data.get("E"),
                    "T": data.get("T"),
                    "bids": _to_float_list(data.get("bids", [])),
                    "asks": _to_float_list(data.get("asks", [])),
                }
                
                # Valida que tem dados
                if not parsed["bids"] or not parsed["asks"]:
                    logging.warning(f"⚠️ Order book vazio! bids={len(parsed['bids'])}, asks={len(parsed['asks'])}")
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                    return None
                
                # Sucesso - atualiza cache
                self._cached_snapshot = parsed
                self._cache_timestamp = time.time()
                
                logging.debug(f"✅ Order book obtido: {len(parsed['bids'])} bids, {len(parsed['asks'])} asks")
                
                return parsed
                
            except requests.exceptions.Timeout as e:
                self._fetch_errors += 1
                logging.warning(f"⏱️ Timeout ao buscar order book (tentativa {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logging.debug(f"   Aguardando {delay:.1f}s antes de retry...")
                    time.sleep(delay)
                    
            except requests.exceptions.RequestException as e:
                self._fetch_errors += 1
                logging.error(f"❌ Erro HTTP ao buscar order book (tentativa {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    
            except Exception as e:
                self._fetch_errors += 1
                logging.error(f"❌ Erro inesperado ao buscar order book: {e}", exc_info=True)
                
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        
        # Todas as tentativas falharam
        now = time.time()
        if now - self._last_error_logged > 60:
            logging.error(
                f"💀 FALHA TOTAL ao buscar order book após {max_retries} tentativas. "
                f"Taxa de erro: {self._fetch_errors}/{self._total_fetches} "
                f"({100 * self._fetch_errors / max(1, self._total_fetches):.1f}%)"
            )
            self._last_error_logged = now
        
        return None

    # -------- Métricas --------

    def _spread_and_depth(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Dict[str, Any]:
        if not bids or not asks:
            return {"mid": None, "spread": None, "spread_percent": None, "bid_depth_usd": 0.0, "ask_depth_usd": 0.0}

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
        total = bid_usd + ask_usd
        if total <= 0:
            return None, None, None
        imbalance = (bid_usd - ask_usd) / total  # [-1, +1]
        ratio = (bid_usd / ask_usd) if ask_usd > 0 else float("inf")
        pressure = imbalance
        return float(imbalance), float(ratio), float(pressure)

    def _detect_walls(self, side_levels: List[Tuple[float, float]], side: str) -> List[Dict[str, Any]]:
        """
        Parede = nível do Top-N cujo qty >= média + k*desvio.
        """
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
                walls.append({"side": side, "price": float(p), "qty": float(q), "limit_threshold": float(threshold)})

        walls.sort(key=lambda x: x["price"], reverse=(side == "bid"))
        return walls

    def _iceberg_reload(self, prev: Optional[Dict[str, Any]], curr: Dict[str, Any], tol: float = 0.75) -> Tuple[bool, float]:
        """
        Heurística: reaparecimento de qty no melhor nível >= tol * qty anterior ⇒ possível recarga.
        """
        try:
            if not prev:
                return False, 0.0
            prev_bids = dict(prev.get("bids", []))
            prev_asks = dict(prev.get("asks", []))
            curr_bids = dict(curr.get("bids", []))
            curr_asks = dict(curr.get("asks", []))

            score = 0.0
            for side_label, pbook_prev, pbook_curr in (("bid", prev_bids, curr_bids), ("ask", prev_asks, curr_asks)):
                if not pbook_prev or not pbook_curr:
                    continue
                p_prev = max(pbook_prev.keys()) if side_label == "bid" else min(pbook_prev.keys())
                p_curr = max(pbook_curr.keys()) if side_label == "bid" else min(pbook_curr.keys())
                if p_prev == p_curr:
                    q_prev = float(pbook_prev[p_prev]); q_curr = float(pbook_curr[p_curr])
                    if q_curr >= tol * max(q_prev, 1e-9) and q_curr > q_prev:
                        score += min(1.0, (q_curr - q_prev) / max(q_prev, 1e-9))
            return (score > 0.5), float(round(score, 4))
        except Exception:
            return False, 0.0

    def _create_empty_event(self, error_msg: str, ts_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Cria um evento completo mesmo quando não há dados.
        """
        if ts_ms is None:
            ts_ms = self.tm.now_ms()
        
        tindex = self.tm.build_time_index(ts_ms, include_local=True, timespec="seconds")
        
        return {
            "schema_version": SCHEMA_VERSION,
            "tipo_evento": "OrderBook",
            "ativo": self.symbol,
            "erro": error_msg,
            "descricao": f"Order book indisponível: {error_msg}",
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
            
            "alertas_liquidez": [f"ERRO: {error_msg}"],
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
                "note": "Order book não pôde ser obtido.",
            },
            
            "order_book_depth": {},
            "spread_analysis": {},
            
            "severity": "ERROR",
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
        }

    # -------- Pública --------

    def analyze(
        self,
        current_snapshot: Optional[Dict[str, Any]] = None,
        *,
        event_epoch_ms: Optional[int] = None,
        window_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analisa o livro e devolve o evento padronizado.
        """
        snap = current_snapshot or self._fetch_orderbook(limit=self.ob_limit_fetch)
        
        if not snap or not snap.get("bids") or not snap.get("asks"):
            error_msg = "snapshot_vazio" if snap else "fetch_failed"
            logging.warning(f"⚠️ Order book indisponível: {error_msg}")
            return self._create_empty_event(error_msg, event_epoch_ms)

        bids: List[Tuple[float, float]] = snap["bids"]
        asks: List[Tuple[float, float]] = snap["asks"]

        # timestamp
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

        sm = self._spread_and_depth(bids, asks)
        mid = sm.get("mid")
        bid_usd = float(sm.get("bid_depth_usd") or 0.0)
        ask_usd = float(sm.get("ask_depth_usd") or 0.0)
        imbalance, ratio, pressure = self._imbalance_ratio_pressure(bid_usd, ask_usd)

        bid_walls = self._detect_walls(bids, side="bid")
        ask_walls = self._detect_walls(asks, side="ask")

        iceberg, iceberg_score = self._iceberg_reload(self.prev_snapshot, {"bids": bids, "asks": asks})

        # impactos determinísticos
        mi_buy_100k = _simulate_market_impact(asks[: self.top_n], usd_amount=100_000.0, side="buy", mid=mid)
        mi_buy_1m = _simulate_market_impact(asks[: self.top_n], usd_amount=1_000_000.0, side="buy", mid=mid)
        mi_sell_100k = _simulate_market_impact(bids[: self.top_n][::-1], usd_amount=100_000.0, side="sell", mid=mid)
        mi_sell_1m = _simulate_market_impact(bids[: self.top_n][::-1], usd_amount=1_000_000.0, side="sell", mid=mid)

        # Atualiza histórico de spread
        if sm.get("spread_percent") is not None and sm["spread_percent"] >= 0:
            try:
                spread_bps = float(sm["spread_percent"]) * 100.0
                now_ms = ts_ms if ts_ms is not None else self.tm.now_ms()
                self.spread_history.append((int(now_ms), spread_bps))
            except Exception:
                pass

        # rótulo
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

        # Remove spreads antigos
        try:
            cutoff_ms = (ts_ms if ts_ms is not None else self.tm.now_ms()) - max(self.spread_avg_windows_min) * 60 * 1000
            self.spread_history = [(t, s) for (t, s) in self.spread_history if t >= cutoff_ms]
        except Exception:
            pass

        # Depth summary
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

        # Spread analysis
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

        # PROMOÇÃO A CRITICAL
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
        is_critical = bool(is_extreme_imbalance and (is_extreme_ratio or is_extreme_usd) or
                           (ratio_dom is not None and ratio_dom >= max(50.0, ORDERBOOK_MIN_RATIO_DOM)))

        if is_critical:
            side_dom = "ASKS" if (imbalance is not None and imbalance < 0) else "BIDS"
            alertas.append(f"DESEQUILÍBRIO CRÍTICO ({side_dom})")

        # Resultado da batalha
        if imbalance is None:
            batalha = "INDISPONÍVEL"
        elif imbalance < -0.05:
            batalha = "Oferta domina"
        elif imbalance > 0.05:
            batalha = "Demanda domina"
        else:
            batalha = "Equilíbrio"

        # Descrição
        descricao = (
            f"Livro: Δ={imbalance:+.4f} | ratio={ratio:.4f} | "
            f"bids=${bid_usd:,.2f} vs asks=${ask_usd:,.2f}"
            if imbalance is not None and ratio is not None
            else "Livro: dados insuficientes"
        )

        # 🔹 LOG DE DEBUG (CORRIGIDO)
        if imbalance is not None and ratio is not None:
            logging.debug(
                f"📊 Order Book: bid_usd=${bid_usd:,.2f}, ask_usd=${ask_usd:,.2f}, "
                f"imbalance={imbalance:.4f}, ratio={ratio:.4f}"
            )
        else:
            logging.debug(
                f"📊 Order Book: bid_usd=${bid_usd:,.2f}, ask_usd=${ask_usd:,.2f}, "
                f"imbalance=None, ratio=None"
            )

        # Payload principal
        event: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tipo_evento": "OrderBook",
            "ativo": self.symbol,
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
            "source": {"exchange": "binance_futures", "endpoint": "fapi/v1/depth", "symbol": self.symbol},
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
            "fetch_stats": {
                "total_fetches": self._total_fetches,
                "fetch_errors": self._fetch_errors,
                "error_rate_pct": round(100 * self._fetch_errors / max(1, self._total_fetches), 2),
                "cache_hit": (current_snapshot is None and self._cached_snapshot is not None),
            },
        }

        # memória
        self.prev_snapshot = {"bids": bids, "asks": asks}
        self.last_event_ts_ms = ts_ms
        return event

    # -------- Shims de compatibilidade --------
    def analyze_order_book(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze(*args, **kwargs)

    def analyzeOrderBook(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze(*args, **kwargs)

    def analyze_orderbook(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze(*args, **kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance."""
        return {
            "total_fetches": self._total_fetches,
            "fetch_errors": self._fetch_errors,
            "error_rate_pct": round(100 * self._fetch_errors / max(1, self._total_fetches), 2),
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "has_cached_data": self._cached_snapshot is not None,
        }


if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.DEBUG)
    oba = OrderBookAnalyzer(symbol="BTCUSDT", cache_ttl_seconds=2.0)
    evt = oba.analyze()
    
    print("\n" + "="*80)
    print("📊 TESTE DE ORDER BOOK ANALYZER")
    print("="*80)
    print(f"Severity: {evt.get('severity')}")
    print(f"Resultado: {evt.get('resultado_da_batalha')}")
    print(f"Bid Depth: ${evt.get('orderbook_data', {}).get('bid_depth_usd', 0):,.2f}")
    print(f"Ask Depth: ${evt.get('orderbook_data', {}).get('ask_depth_usd', 0):,.2f}")
    print(f"Imbalance: {evt.get('orderbook_data', {}).get('imbalance', 0):+.4f}")
    print(f"Alertas: {evt.get('alertas_liquidez')}")
    print(f"\nFetch Stats: {oba.get_stats()}")
    print("="*80)