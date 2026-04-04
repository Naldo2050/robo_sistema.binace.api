"""
quality_summary — Resume qualidade dos dados e impacto na confiança da análise.

Transforma:
    qual.lat, qual.liq, qual.ms, qual.holiday
    ctx.cached

Em:
    quality_summary: {
        "reliable":       bool,
        "confidence_cap": float,
        "issues":         list[str],
        "note":           str
    }
"""

from __future__ import annotations

from typing import Any


_LATENCY_CAPS: dict[str, float] = {
    "OK":   1.0,
    "NEAR": 0.9,
    "DEGR": 0.7,
    "CRIT": 0.4,
}

_LIQUIDITY_CAPS: dict[str, float] = {
    "NORMAL":   1.0,
    "RED":      0.8,
    "LOW":      0.7,
    "VERY_LOW": 0.5,
    "VERY":     0.5,   # fallback para truncamento
    "VER":      0.5,   # fallback para truncamento [:3]
}

_LIQUIDITY_LABELS: dict[str, str] = {
    "NORMAL":   "normal",
    "RED":      "reduzida",
    "LOW":      "baixa",
    "VERY_LOW": "muito baixa",
    "VERY":     "muito baixa",
    "VER":      "muito baixa",
}


def _resolve_liquidity(liq_raw: str) -> tuple[float, str]:
    """Resolve cap e label de liquidez independente de truncamento."""
    key = liq_raw.upper()
    cap = _LIQUIDITY_CAPS.get(key)
    label = _LIQUIDITY_LABELS.get(key)

    if cap is None:
        # tenta match parcial
        for k in _LIQUIDITY_CAPS:
            if key.startswith(k) or k.startswith(key):
                cap = _LIQUIDITY_CAPS[k]
                label = _LIQUIDITY_LABELS.get(k, key.lower())
                break

    return (cap or 1.0), (label or key.lower())


def build_quality_summary(compact_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Gera resumo interpretado da qualidade dos dados.

    Args:
        compact_payload: payload já construído pelo build_compact_payload()

    Returns:
        dict com reliable, confidence_cap, issues e note
    """
    qual = compact_payload.get("qual", {})
    ctx = compact_payload.get("ctx", {})

    issues: list[str] = []
    caps: list[float] = [1.0]

    # --- Latência ---
    lat_cat = str(qual.get("lat", "OK")).upper()
    lat_ms = qual.get("ms")
    lat_cap = _LATENCY_CAPS.get(lat_cat, 1.0)
    caps.append(lat_cap)

    if lat_cat == "DEGR":
        msg = f"Latência degradada ({lat_ms}ms)" if lat_ms else "Latência degradada"
        issues.append(msg)
    elif lat_cat == "CRIT":
        msg = f"Latência crítica ({lat_ms}ms)" if lat_ms else "Latência crítica"
        issues.append(msg)

    # --- Liquidez ---
    liq_raw = str(qual.get("liq", "NORMAL"))
    liq_cap, liq_label = _resolve_liquidity(liq_raw)
    caps.append(liq_cap)

    if liq_raw.upper() not in ("NORMAL",):
        issues.append(f"Liquidez {liq_label}")

    # --- Feriado ---
    holiday = qual.get("holiday")
    if holiday:
        issues.append(f"Feriado: {holiday} — liquidez muito reduzida")
        caps.append(0.6)

    # --- Contexto cacheado ---
    ctx_cached = ctx.get("cached", False)
    if ctx_cached:
        issues.append("Contexto estático em cache (até 5 min desatualizado)")
        caps.append(0.9)

    # --- Confidence cap final ---
    confidence_cap = round(min(caps), 2)
    reliable = confidence_cap >= 0.7 and len(issues) == 0

    note = _build_note(
        reliable=reliable,
        confidence_cap=confidence_cap,
        issues=issues,
        lat_cat=lat_cat,
        liq_raw=liq_raw,
        ctx_cached=ctx_cached,
        holiday=holiday,
    )

    return {
        "reliable":       reliable,
        "confidence_cap": confidence_cap,
        "issues":         issues,
        "note":           note,
    }


def _build_note(
    reliable: bool,
    confidence_cap: float,
    issues: list[str],
    lat_cat: str,
    liq_raw: str,
    ctx_cached: bool,
    holiday: str | None = None,
) -> str:

    if reliable:
        return "Dados em tempo real sem anomalias. Análise com confiança plena."

    parts: list[str] = []

    if lat_cat in ("DEGR", "CRIT"):
        severity = "crítica" if lat_cat == "CRIT" else "degradada"
        parts.append(f"Latência {severity} compromete freshness dos dados")

    if holiday:
        parts.append(f"Feriado ({holiday}) reduz liquidez severamente")
    elif liq_raw.upper() not in ("NORMAL",):
        parts.append("Liquidez reduzida pode distorcer sinais de fluxo e orderbook")

    if ctx_cached:
        parts.append("Contexto macro/derivativos pode estar desatualizado")

    cap_pct = round(confidence_cap * 100)
    parts.append(f"Confiança máxima desta análise: {cap_pct}%")

    return ". ".join(parts) + "."
