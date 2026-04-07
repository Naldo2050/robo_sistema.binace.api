"""
institutional_summary — Resume análise institucional em conclusão acionável.

Transforma:
    price.sh (profile shape), price.auc (auction bias)
    price.ph / price.pl (poor extremes)
    price.brk_risk (breakout risk do Value Area)
    w.s, w.c (whale score e classificação)
    flow.pa, flow.conv

Em:
    institutional_summary: {
        "auction_state": str,
        "whale_bias":    "ACCUMULATING" | "DISTRIBUTING" | "NEUTRAL",
        "profile_bias":  "BULLISH" | "BEARISH" | "NEUTRAL",
        "unfinished":    list[str],   # "low" | "high"
        "note":          str
    }
"""

from __future__ import annotations

from typing import Any


_SHAPE_BIAS: dict[str, str] = {
    "b":  "BULLISH",  # b-shape: long liquidation, bullish after
    "p":  "BEARISH",  # p-shape: short covering, bearish after
    "D":  "NEUTRAL",  # double distribution
    "I":  "NEUTRAL",  # thin, indeterminate
}

_WHALE_CLS_BIAS: dict[str, str] = {
    "MA": "ACCUMULATING",
    "SA": "ACCUMULATING",
    "MD": "DISTRIBUTING",
    "SD": "DISTRIBUTING",
    "N":  "NEUTRAL",
}

_AUCTION_MAP: dict[str, str] = {
    "expect_retest_low":  "Leilão incompleto — mínima deve ser revisitada",
    "expect_retest_high": "Leilão incompleto — máxima deve ser revisitada",
    "expect_retest_both": "Leilão incompleto em ambos os extremos",
    "balanced":           "Leilão equilibrado",
    "accept_higher":      "Mercado aceitando preços mais altos",
    "accept_lower":       "Mercado aceitando preços mais baixos",
}


def build_institutional_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Gera resumo interpretado da análise institucional.

    Args:
        payload: payload já construído pelo build_compact_payload()

    Returns:
        dict com auction_state, whale_bias, profile_bias, unfinished e note
    """
    price = payload.get("price", {})
    whale = payload.get("w", {})
    flow = payload.get("flow", {})

    shape = price.get("sh", "")
    auction_raw = str(price.get("auc", "")).lower()
    ph = price.get("ph", 0)
    pl = price.get("pl", 0)
    brk_risk = price.get("brk_risk", "")

    whale_score = whale.get("s", 0) or 0
    whale_cls = whale.get("c", "N")
    whale_bias = _WHALE_CLS_BIAS.get(str(whale_cls).upper(), "NEUTRAL")

    profile_bias = _SHAPE_BIAS.get(shape, "NEUTRAL")

    auction_state = _AUCTION_MAP.get(
        auction_raw,
        auction_raw.replace("_", " ").capitalize() if auction_raw else "Estado de leilão desconhecido",
    )

    # --- Extremos incompletos ---
    unfinished: list[str] = []
    if pl:
        unfinished.append("low")
    if ph:
        unfinished.append("high")

    # --- Consistência whale vs profile ---
    alignment = _check_alignment(whale_bias, profile_bias, flow)

    # --- Nota interpretada ---
    note = _build_note(
        auction_state=auction_state,
        profile_bias=profile_bias,
        whale_bias=whale_bias,
        whale_score=whale_score,
        unfinished=unfinished,
        brk_risk=brk_risk,
        alignment=alignment,
    )

    return {
        "auction_state": auction_state,
        "whale_bias":    whale_bias,
        "profile_bias":  profile_bias,
        "unfinished":    unfinished,
        "alignment":     alignment,
        "note":          note,
    }


def _check_alignment(
    whale_bias: str,
    profile_bias: str,
    flow: dict,
) -> str:
    """Verifica se whales, profile e fluxo apontam na mesma direção."""
    pa = str(flow.get("pa", "")).lower()

    bull_signals = 0
    bear_signals = 0

    if whale_bias == "ACCUMULATING":
        bull_signals += 1
    elif whale_bias == "DISTRIBUTING":
        bear_signals += 1

    if profile_bias == "BULLISH":
        bull_signals += 1
    elif profile_bias == "BEARISH":
        bear_signals += 1

    if "buy" in pa:
        bull_signals += 1
    elif "sell" in pa:
        bear_signals += 1

    if bull_signals >= 2 and bear_signals == 0:
        return "BULL_ALIGNED"
    elif bear_signals >= 2 and bull_signals == 0:
        return "BEAR_ALIGNED"
    elif bull_signals > 0 and bear_signals > 0:
        return "CONFLICTED"
    return "NEUTRAL"


def _build_note(
    auction_state: str,
    profile_bias: str,
    whale_bias: str,
    whale_score: int,
    unfinished: list[str],
    brk_risk: str,
    alignment: str,
) -> str:

    parts: list[str] = []

    parts.append(auction_state)

    if profile_bias != "NEUTRAL":
        direction = "bullish" if profile_bias == "BULLISH" else "bearish"
        parts.append(f"Profile com viés {direction}")

    if whale_score > 0:
        action = "acumulando" if whale_bias == "ACCUMULATING" else "distribuindo"
        parts.append(f"Whales {action} (score {whale_score})")

    if unfinished:
        extremos = " e ".join(unfinished)
        parts.append(f"Extremo(s) incompleto(s): {extremos} — reteste esperado")

    if brk_risk in ("HI", "V_HI"):
        risk_label = "alto" if brk_risk == "HI" else "muito alto"
        parts.append(f"Risco de breakout da Value Area {risk_label}")

    if alignment == "BULL_ALIGNED":
        parts.append("Sinais institucionais alinhados para alta")
    elif alignment == "BEAR_ALIGNED":
        parts.append("Sinais institucionais alinhados para baixa")
    elif alignment == "CONFLICTED":
        parts.append("Sinais institucionais conflitantes — cautela")

    return ". ".join(parts) + "." if parts else "Análise institucional indisponível."
