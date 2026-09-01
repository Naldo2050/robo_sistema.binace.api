"""
flow_summary — Resume microestrutura e fluxo de ordens em conclusão operacional.

Transforma:
    flow.pa, flow.abs_*, flow.imb, flow.d1/d5/d15,
    flow.sf_w/r, flow.ti, flow.trs

Em:
    flow_summary: {
        "bias":    "BUY" | "SELL" | "NEUTRAL",
        "type":    "absorption" | "aggressive" | "passive" | "mixed",
        "actor":   "whale" | "retail" | "mixed" | "unknown",
        "conf":    "H" | "M" | "L",
        "note":    str  # frase curta interpretada
    }
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_PA_SIGNAL_BIAS: dict[str, str] = {
    "buy_absorp":       "BUY",
    "sell_absorp":      "SELL",
    "buy_aggres":       "BUY",
    "sell_aggre":       "SELL",
    "buy_passiv":       "BUY",
    "sell_passi":       "SELL",
    "buy_absorption":   "BUY",
    "sell_absorption":  "SELL",
    "buy_aggressive":   "BUY",
    "sell_aggressive":  "SELL",
    "buy_passive":      "BUY",
    "sell_passive":     "SELL",
    "neutral":          "NEUTRAL",
}

_CONVICTION_MAP: dict[str, str] = {
    "HIGH":   "H",
    "MEDIUM": "M",
    "LOW":    "L",
    "H":      "H",
    "M":      "M",
    "L":      "L",
}

_TYPE_FROM_SIGNAL: dict[str, str] = {
    "buy_absorp":      "absorption",
    "sell_absorp":     "absorption",
    "buy_aggres":      "aggressive",
    "sell_aggre":      "aggressive",
    "buy_passiv":      "passive",
    "sell_passi":      "passive",
    "buy_absorption":  "absorption",
    "sell_absorption": "absorption",
    "buy_aggressive":  "aggressive",
    "sell_aggressive": "aggressive",
    "buy_passive":     "passive",
    "sell_passive":    "passive",
    "neutral":         "mixed",
}


def build_flow_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Gera resumo interpretado do fluxo a partir do payload compacto.

    Args:
        payload: payload já construído pelo build_compact_payload()

    Returns:
        dict com bias, type, actor, conf e note
    """
    flow = payload.get("flow", {})
    if not flow:
        return {
            "bias": "NEUTRAL",
            "type": "mixed",
            "actor": "unknown",
        }

    # --- Bias principal via pa signal ---
    pa_signal = str(flow.get("pa", "neutral")).lower()
    bias = _PA_SIGNAL_BIAS.get(pa_signal, "NEUTRAL")

    # --- Confirmar bias via imbalance ---
    imb = flow.get("imb", 0.0) or 0.0
    if abs(imb) >= 0.25 and bias == "NEUTRAL":
        bias = "BUY" if imb > 0 else "SELL"

    # --- Tipo de fluxo ---
    flow_type = _TYPE_FROM_SIGNAL.get(pa_signal, "mixed")

    # --- Actor: whale vs retail (acumulado 4h, ver legend cvd_4h/sf_*_4h) ---
    sf_w = flow.get("sf_w_4h") or 0.0
    sf_r = flow.get("sf_r_4h") or 0.0

    whale_active = abs(sf_w) > 0.1
    retail_active = abs(sf_r) > 0.1

    if whale_active and not retail_active:
        actor = "whale"
    elif retail_active and not whale_active:
        actor = "retail"
    elif whale_active and retail_active:
        if abs(sf_w) > abs(sf_r) * 1.5:
            actor = "whale"
        elif abs(sf_r) > abs(sf_w) * 1.5:
            actor = "retail"
        else:
            actor = "mixed"
    else:
        actor = "unknown"

    # --- Absorção: forçar bias se sinal forte ---
    abs_buy = flow.get("abs_buy_str") or 0.0
    abs_sell = flow.get("abs_sell_exh") or 0.0

    if flow_type == "absorption":
        if abs_buy > 5.0 and abs_sell < 2.0:
            bias = "BUY"
        elif abs_buy < 2.0 and abs_sell > 5.0:
            bias = "SELL"

    # --- Divergência curto prazo vs fluxo dominante ---
    d1_str = str(flow.get("d1", "0"))
    d5_str = str(flow.get("d5", "0"))

    d1_positive = d1_str.startswith("+") and d1_str != "+0"
    d1_negative = d1_str.startswith("-")
    d5_positive = d5_str.startswith("+") and d5_str != "+0"
    d5_negative = d5_str.startswith("-")

    # Janelas degradadas (trunc/warm) não podem sustentar comparação
    # temporal entre timeframes — tratar valor parcial como completo
    # seria inferência sobre dado incompleto.
    degraded_windows = _degraded_flow_windows(flow.get("q"))
    temporal_comparison_valid = not ({"1m", "5m"} & set(degraded_windows))

    short_term_reversal = (
        temporal_comparison_valid
        and (
            (d1_positive and d5_negative)
            or (d1_negative and d5_positive)
        )
    )

    result: dict[str, Any] = {
        "bias":  bias,
        "type":  flow_type,
        "actor": actor,
    }

    if short_term_reversal:
        result["reversal_signal"] = True

    return result


def _degraded_flow_windows(quality: Any) -> list:
    """Janelas com cobertura temporal degradada, na ordem 1m/5m/15m."""
    if not isinstance(quality, dict):
        return []
    ordered = ("1m", "5m", "15m")
    return [
        window for window in ordered
        if isinstance(quality.get(window), dict)
        and quality[window].get("s") in ("trunc", "warm")
    ]


def _flow_quality_note(quality: dict, windows: list) -> str:
    fragments: list[str] = []
    for window in windows:
        info = quality.get(window) or {}
        status = info.get("s")
        if status == "trunc":
            label = "parcial"
        elif status == "warm":
            label = "em aquecimento"
        else:
            label = "degradada"
        coverage = info.get("c")
        if isinstance(coverage, (int, float)) and not isinstance(coverage, bool):
            fragments.append(f"{window} {label} ({round(float(coverage), 1)}% cobertura)")
        else:
            fragments.append(f"{window} {label}")
    return "Cobertura temporal parcial do fluxo: " + ", ".join(fragments)


def _build_note(
    bias: str,
    flow_type: str,
    actor: str,
    conf: str,
    pa_signal: str,
    abs_buy: float,
    abs_sell: float,
    abs_cont: float,
    imb: float,
    short_term_reversal: bool,
    sf_w: float,
    sf_r: float,
    buy_pct: float = 50.0,
    bsr: float = 1.0,
) -> str:

    parts: list[str] = []

    # Tipo e direção
    if flow_type == "absorption":
        if bias == "BUY":
            parts.append("Absorção compradora ativa")
            if abs_buy > 7.0:
                parts.append("com força de compra elevada")
            if abs_sell < 1.0:
                parts.append("e vendedores esgotados")
        elif bias == "SELL":
            parts.append("Absorção vendedora ativa")
            if abs_sell > 7.0:
                parts.append("com pressão vendedora elevada")
        else:
            parts.append("Absorção neutra")

    elif flow_type == "aggressive":
        direction = "compradores" if bias == "BUY" else "vendedores"
        parts.append(f"Fluxo agressivo de {direction}")

    elif flow_type == "passive":
        direction = "compra" if bias == "BUY" else "venda"
        parts.append(f"Pressão passiva de {direction}")

    else:
        parts.append("Fluxo misto sem dominância clara")

    # Actor
    # sf_w é acumulado de sessão — usar buy_pct da janela corrente para direção real
    if actor == "whale":
        direction_w = "comprando" if buy_pct > 50 else "vendendo"
        parts.append(f"(whales {direction_w})")
    elif actor == "retail":
        direction_r = "comprador" if sf_r > 0 else "vendedor"
        parts.append(f"(varejo {direction_r})")
    elif actor == "mixed":
        parts.append("(whales e varejo divergindo)")

    # Probabilidade de continuação
    if abs_cont > 0.3:
        parts.append(f"prob. continuação {round(abs_cont * 100)}%")

    # Reversão de curto prazo
    if short_term_reversal:
        parts.append("— divergência 1m vs 5m detectada")

    # Imbalance extremo
    if abs(imb) > 0.5:
        direction_imb = "compra" if imb > 0 else "venda"
        parts.append(f"[imbalance extremo de {direction_imb}]")

    return ". ".join(parts) if parts else "Fluxo sem sinal definido."
