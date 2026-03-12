# market_impact.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Any, Optional

Number = float
Ladder = Iterable[Tuple[Number, Number]]  # [(price, qty_base), ...]

class ImpactError(Exception):
    pass

def _assert_sorted(ladder: List[Tuple[Number, Number]], ascending: bool) -> bool:
    if not ladder:
        return True
    last = ladder[0][0]
    for p, _ in ladder[1:]:
        if ascending:
            if p < last:  # deve ser crescente (asks)
                return False
        else:
            if p > last:  # deve ser decrescente (bids)
                return False
        last = p
    return True

def _consume(ladder: List[Tuple[Number, Number]], qty_base: Number, side: str) -> Tuple[Number, Number, Number, bool]:
    """
    Consome a ladder até atender qty_base.
    Retorna: (avg_filled_price, final_price, filled_base, partial_fill)
    """
    if qty_base <= 0:
        raise ImpactError("qty_base deve ser > 0")
    filled = 0.0
    notional = 0.0
    final_price = None

    for price, avail in ladder:
        if avail <= 0:
            continue
        take = min(qty_base - filled, avail)
        if take <= 0:
            break
        notional += take * price
        filled += take
        final_price = price
        if filled >= qty_base:
            break

    if filled <= 0:
        # sem liquidez
        return (0.0, 0.0, 0.0, True)

    avg_price = notional / filled
    partial = filled < qty_base
    return (avg_price, final_price or avg_price, filled, partial)

def compute_market_impact(
    bids: Ladder,
    asks: Ladder,
    mid_price: Optional[Number],
    notional_usd: Number,
) -> Dict[str, Any]:
    """
    Calcula impacto para uma compra (consome ASKS) e uma venda (consome BIDS),
    simulando a execução nível a nível (walk-the-book).
    - bids: lista decrescente [(price, qty_base)]
    - asks: lista crescente  [(price, qty_base)]
    - notional_usd: tamanho da ordem em USD
    """
    bids_l = list(bids or [])
    asks_l = list(asks or [])
    if not _assert_sorted(bids_l, ascending=False):
        return {"error": "bids_not_sorted", "quality_flags": ["structure:ladder_unsorted_bids"]}
    if not _assert_sorted(asks_l, ascending=True):
        return {"error": "asks_not_sorted", "quality_flags": ["structure:ladder_unsorted_asks"]}

    # mid para slippage percentual
    # se não veio mid, tenta aproximar
    if mid_price is None:
        if bids_l and asks_l:
            mid_price = (bids_l[0][0] + asks_l[0][0]) / 2.0
        elif bids_l:
            mid_price = bids_l[0][0]
        elif asks_l:
            mid_price = asks_l[0][0]
        else:
            mid_price = 0.0

    quality_flags: List[str] = []

    # Define qty em base para BUY (consome asks)
    # Aproximação: usa o melhor preço disponível como referência.
    ref_ask = asks_l[0][0] if asks_l else (mid_price or 0.0)
    ref_bid = bids_l[0][0] if bids_l else (mid_price or 0.0)
    qty_buy_base = (notional_usd / ref_ask) if ref_ask > 0 else 0.0
    qty_sell_base = (notional_usd / ref_bid) if ref_bid > 0 else 0.0

    # BUY: consome ASKS (ascending)
    buy_avg, buy_final, buy_filled, buy_partial = _consume(asks_l, qty_buy_base, "buy")
    # SELL: consome BIDS (descending)
    sell_avg, sell_final, sell_filled, sell_partial = _consume(bids_l, qty_sell_base, "sell")

    out: Dict[str, Any] = {
        "buy": {
            "avg_filled_price": buy_avg if buy_filled > 0 else None,
            "final_price": buy_final if buy_filled > 0 else None,
            "filled_base": buy_filled,
            "partial_fill": buy_partial,
            "impact_usd": None,
            "slippage_percent": None,
        },
        "sell": {
            "avg_filled_price": sell_avg if sell_filled > 0 else None,
            "final_price": sell_final if sell_filled > 0 else None,
            "filled_base": sell_filled,
            "partial_fill": sell_partial,
            "impact_usd": None,
            "slippage_percent": None,
        },
        "quality_flags": quality_flags,
    }

    # Slippage/impact
    if mid_price and mid_price > 0:
        if buy_filled > 0 and buy_final:
            # compra -> preço pior (>= mid)
            slip_buy = (buy_final - mid_price) / mid_price
            out["buy"]["slippage_percent"] = float(slip_buy * 100.0)
            out["buy"]["impact_usd"] = float((buy_final - mid_price))
        else:
            out["buy"]["slippage_percent"] = None
            out["buy"]["impact_usd"] = None

        if sell_filled > 0 and sell_final:
            # venda -> preço pior (<= mid)
            slip_sell = (mid_price - sell_final) / mid_price
            out["sell"]["slippage_percent"] = float(slip_sell * 100.0)
            out["sell"]["impact_usd"] = float((mid_price - sell_final))
        else:
            out["sell"]["slippage_percent"] = None
            out["sell"]["impact_usd"] = None
    else:
        quality_flags.append("structure:mid_missing")

    # Flags de liquidez insuficiente
    if buy_filled <= 0:
        quality_flags.append("liquidity:insufficient_on_ask")
    if sell_filled <= 0:
        quality_flags.append("liquidity:insufficient_on_bid")
    if buy_partial:
        quality_flags.append("liquidity:partial_fill_buy")
    if sell_partial:
        quality_flags.append("liquidity:partial_fill_sell")

    return out
