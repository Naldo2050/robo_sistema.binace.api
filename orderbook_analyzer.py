# orderbook_analyzer.py
import logging
from typing import List, Dict, Any, Tuple, Optional

import requests
import numpy as np

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

from time_manager import TimeManager

SCHEMA_VERSION = "1.2.1"


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
    """

    def __init__(
        self,
        symbol: str,
        liquidity_flow_alert_percentage: float = 0.40,
        wall_std_dev_factor: float = 3.0,
        top_n_levels: int = 20,
        ob_limit_fetch: int = 100,
        time_manager: Optional[TimeManager] = None,
    ):
        self.symbol = symbol.upper()
        self.alert_threshold = float(liquidity_flow_alert_percentage)
        self.wall_std = float(wall_std_dev_factor)
        self.top_n = int(top_n_levels)
        self.ob_limit_fetch = int(ob_limit_fetch)
        self.tz_ny = ZoneInfo("America/New_York") if ZoneInfo else None
        self.tm = time_manager or TimeManager()

        # memória leve para heurística de recarga
        self.prev_snapshot: Optional[Dict[str, Any]] = None
        self.last_event_ts_ms: Optional[int] = None

        logging.info(
            "✅ OrderBook Analyzer inicializado para %s | Alerta fluxo: %s | Wall STD: %s | Top N: %s",
            self.symbol,
            f"{self.alert_threshold*100:.0f}%",
            f"{self.wall_std:.1f}x",
            self.top_n,
        )

    # -------- Data --------

    def _fetch_orderbook(self, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        try:
            lim = limit or self.ob_limit_fetch
            url = f"https://fapi.binance.com/fapi/v1/depth?symbol={self.symbol}&limit={lim}"
            r = requests.get(url, timeout=2.5)
            r.raise_for_status()
            data = r.json()
            return {
                "lastUpdateId": data.get("lastUpdateId"),
                "E": data.get("E"),
                "T": data.get("T"),
                "bids": _to_float_list(data.get("bids", [])),
                "asks": _to_float_list(data.get("asks", [])),
            }
        except Exception as e:
            logging.error(f"Erro ao buscar orderbook: {e}")
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

        # bids: maior preço primeiro | asks: menor preço primeiro
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
            return {"tipo_evento": "OrderBook", "ativo": self.symbol, "erro": "snapshot_vazio"}

        bids: List[Tuple[float, float]] = snap["bids"]
        asks: List[Tuple[float, float]] = snap["asks"]

        # timestamp (preferir tempos do exchange)
        ts_ms = None
        for key in ("E", "T"):
            v = snap.get(key)
            if isinstance(v, (int, float)) and v > 0:
                ts_ms = int(v)
                break
        if ts_ms is None:
            ts_ms = event_epoch_ms if event_epoch_ms is not None else self.tm.now_ms()

        tindex = self.tm.build_time_index(ts_ms, include_local=True, timespec="seconds")
        ny_time = tindex.get("ny_time")
        utc_iso = tindex.get("utc_iso")

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

        # rótulo claro (estoque de liquidez)
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

        event: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "tipo_evento": "OrderBook",
            "ativo": self.symbol,
            "descricao": f"{resultado_da_batalha} | TopN={self.top_n}",
            "resultado_da_batalha": resultado_da_batalha,
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
            "timestamps": {"exchange_ms": ts_ms, "ny_time": ny_time, "utc_iso": utc_iso},
            "source": {"exchange": "binance_futures", "endpoint": "fapi/v1/depth", "symbol": self.symbol},
            "labels": {
                "dominant_label": resultado_da_batalha,
                "note": "Rótulo baseado no livro (estoque de liquidez), não na fita executada (delta).",
            },
        }

        # memória
        self.prev_snapshot = {"bids": bids, "asks": asks}
        self.last_event_ts_ms = ts_ms
        return event

    # -------- Shims de compatibilidade --------
    def analyze_order_book(self, *args, **kwargs) -> Dict[str, Any]:
        """Compat: nome legado usado no pipeline antigo."""
        return self.analyze(*args, **kwargs)

    def analyzeOrderBook(self, *args, **kwargs) -> Dict[str, Any]:
        """Compat camelCase."""
        return self.analyze(*args, **kwargs)

    def analyze_orderbook(self, *args, **kwargs) -> Dict[str, Any]:
        """Compat sem sublinhado."""
        return self.analyze(*args, **kwargs)
