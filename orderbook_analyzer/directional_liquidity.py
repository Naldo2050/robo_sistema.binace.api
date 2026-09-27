# orderbook_analyzer/directional_liquidity.py
"""
P2-B2 — Directional Liquidity & Market Impact Contract v1.

Separa explicitamente:
1. Métricas físicas direcionais de execução (buy vs sell em USD e bps);
2. Qualidade agregada legada (preservada para backward compatibility);
3. Unidades físicas canônicas (USD/BTC vs bps);
4. Disponibilidade/fillability e assimetria raw matemática sem heurística direcional.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _finite_or_none(val: Any) -> Optional[float]:
    """Converte para float se finito, senão retorna None."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _extract_side_metrics(
    mi_side_dict: Optional[Dict[str, Any]],
    notional_key: str,
    mid: Optional[float],
) -> Dict[str, Any]:
    """Extrai métricas direcionais canônicas para um lado e notional."""
    d = (mi_side_dict or {}).get(notional_key, {}) or {}

    insufficient = bool(d.get("insufficient_liquidity", False))
    raw_fill_ratio = d.get("fill_ratio")
    fill_ratio = _finite_or_none(raw_fill_ratio) if raw_fill_ratio is not None else (0.0 if insufficient else 1.0)
    if fill_ratio is None:
        fill_ratio = 0.0

    # Slippage VWAP
    raw_exec_usd = d.get("execution_slippage_usd")
    if raw_exec_usd is None and "move_usd" in d and not insufficient:
        # Fallback legado se schema antigo
        raw_exec_usd = d.get("move_usd")
    exec_slip_usd = _finite_or_none(raw_exec_usd)

    raw_exec_bps = d.get("execution_slippage_bps")
    exec_slip_bps = _finite_or_none(raw_exec_bps)

    # Terminal Move
    raw_term_usd = d.get("terminal_move_usd")
    if raw_term_usd is None and "move_usd" in d and not insufficient:
        raw_term_usd = d.get("move_usd")
    term_move_usd = _finite_or_none(raw_term_usd)

    raw_term_bps = d.get("terminal_move_bps")
    if raw_term_bps is None and "bps" in d and not insufficient:
        raw_term_bps = d.get("bps")
    term_move_bps = _finite_or_none(raw_term_bps)

    # Observed variants (para partial fills)
    obs_slip_usd = _finite_or_none(d.get("observed_execution_slippage_usd"))
    obs_slip_bps = _finite_or_none(d.get("observed_execution_slippage_bps"))
    obs_term_usd = _finite_or_none(d.get("observed_terminal_move_usd"))
    obs_term_bps = _finite_or_none(d.get("observed_terminal_move_bps"))

    # Derivar bps se ausente mas USD e mid finitos disponíveis
    mid_finite = _finite_or_none(mid)
    if mid_finite and mid_finite > 0:
        if exec_slip_bps is None and exec_slip_usd is not None:
            exec_slip_bps = round((exec_slip_usd / mid_finite) * 10000.0, 4)
        if term_move_bps is None and term_move_usd is not None:
            term_move_bps = round((term_move_usd / mid_finite) * 10000.0, 4)
        if obs_slip_bps is None and obs_slip_usd is not None:
            obs_slip_bps = round((obs_slip_usd / mid_finite) * 10000.0, 4)
        if obs_term_bps is None and obs_term_usd is not None:
            obs_term_bps = round((obs_term_usd / mid_finite) * 10000.0, 4)

    # Validity / Fillability
    is_fillable = (not insufficient) and (fill_ratio >= 1.0 - 1e-4) and (exec_slip_usd is not None)

    if not d:
        validity = "NON_VOTING_MISSING"
        reason = "DATA_MISSING"
    elif insufficient:
        validity = "INSUFFICIENT_LIQUIDITY"
        reason = "PARTIAL_FILL"
    elif exec_slip_usd is None or exec_slip_bps is None:
        validity = "NON_VOTING_INVALID_INPUT"
        reason = "NON_FINITE_SLIPPAGE"
    else:
        validity = "VALID"
        reason = None

    res: Dict[str, Any] = {
        "execution_slippage_usd": exec_slip_usd if is_fillable else None,
        "execution_slippage_bps": exec_slip_bps if is_fillable else None,
        "terminal_move_usd": term_move_usd if is_fillable else None,
        "terminal_move_bps": term_move_bps if is_fillable else None,
        "observed_execution_slippage_usd": obs_slip_usd,
        "observed_execution_slippage_bps": obs_slip_bps,
        "observed_terminal_move_usd": obs_term_usd,
        "observed_terminal_move_bps": obs_term_bps,
        "fill_ratio": round(fill_ratio, 4),
        "is_fillable": is_fillable,
        "validity": validity,
    }
    if reason:
        res["reason"] = reason
    return res


def _calculate_asymmetry(
    buy_item: Dict[str, Any],
    sell_item: Dict[str, Any],
) -> Dict[str, Any]:
    """Calcula métricas matemáticas puras de assimetria para um notional."""
    b_bps = buy_item.get("execution_slippage_bps")
    if b_bps is None:
        b_bps = buy_item.get("observed_execution_slippage_bps")

    s_bps = sell_item.get("execution_slippage_bps")
    if s_bps is None:
        s_bps = sell_item.get("observed_execution_slippage_bps")

    b_fill = buy_item.get("fill_ratio", 0.0)
    s_fill = sell_item.get("fill_ratio", 0.0)
    fill_diff = round(b_fill - s_fill, 4) if (b_fill is not None and s_fill is not None) else None

    abs_diff = None
    ratio = None
    buy_to_sell = None
    sell_to_buy = None
    ratio_reason = None

    if b_bps is not None and s_bps is not None:
        abs_diff = round(abs(b_bps - s_bps), 4)
        if s_bps > 0:
            buy_to_sell = round(b_bps / s_bps, 4)
            ratio = buy_to_sell
        elif s_bps == 0:
            ratio_reason = "ZERO_DENOMINATOR_SELL"
        if b_bps > 0:
            sell_to_buy = round(s_bps / b_bps, 4)
        elif b_bps == 0 and ratio_reason is None:
            ratio_reason = "ZERO_DENOMINATOR_BUY"
    else:
        ratio_reason = "NON_FINITE_OR_MISSING_SLIPPAGE"

    return {
        "buy_slippage_bps": b_bps,
        "sell_slippage_bps": s_bps,
        "absolute_difference_bps": abs_diff,
        "slippage_ratio": ratio,
        "buy_to_sell_ratio": buy_to_sell,
        "sell_to_buy_ratio": sell_to_buy,
        "buy_fill_ratio": b_fill,
        "sell_fill_ratio": s_fill,
        "fill_ratio_difference": fill_diff,
        "ratio_reason": ratio_reason,
    }


def build_directional_liquidity(
    mi_buy: Optional[Dict[str, Any]],
    mi_sell: Optional[Dict[str, Any]],
    mid: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Constrói o contrato direcional de liquidez e impacto de mercado v1 (P2-B2).

    Responde objetivamente:
      - 'quanto custa comprar?'
      - 'quanto custa vender?'
    sem mascarar assimetria em score agregado ou rótulo EXCELLENT.
    """
    notionals = ["100k", "1M"]
    key_map = {"100k": "100k_usd", "1M": "1m_usd", "1m": "1m_usd"}

    buy_dict: Dict[str, Any] = {}
    sell_dict: Dict[str, Any] = {}
    asym_dict: Dict[str, Any] = {}

    any_valid = False
    all_missing = True

    for n in notionals:
        out_key = key_map.get(n, f"{n.lower()}_usd")
        b_res = _extract_side_metrics(mi_buy, n, mid)
        s_res = _extract_side_metrics(mi_sell, n, mid)

        buy_dict[out_key] = b_res
        sell_dict[out_key] = s_res
        asym_dict[out_key] = _calculate_asymmetry(b_res, s_res)

        if b_res["validity"] != "NON_VOTING_MISSING" or s_res["validity"] != "NON_VOTING_MISSING":
            all_missing = False
        if b_res["validity"] == "VALID" or s_res["validity"] == "VALID":
            any_valid = True

    if all_missing:
        status = "NON_VOTING_MISSING"
    elif any_valid:
        status = "VALID"
    else:
        status = "FAIL_CLOSED"

    return {
        "buy": buy_dict,
        "sell": sell_dict,
        "asymmetry": asym_dict,
        "source_type": "POINT_IN_TIME_L2",
        "capability": "POINT_IN_TIME_L2",
        "status": status,
        "execution_gate": "NOT_DIRECTIONAL_EXECUTION_GATE",
        "notes": (
            "Directional metrics represent point-in-time L2 execution conditions. "
            "No directional bias, recommendation or gate is inferred."
        ),
    }
