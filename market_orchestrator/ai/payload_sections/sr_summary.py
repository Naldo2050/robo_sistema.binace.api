"""
sr_summary — Resume suportes e resistências em contexto operacional.

Transforma:
    sr.r1, sr.r1_dist, sr.r1_conf
    sr.s1, sr.s1_dist, sr.s1_conf
    sr.def_bias

Em:
    sr_summary: {
        "nearest":    "support" | "resistance" | "equidistant",
        "compressed": bool,       # preço entre níveis muito próximos
        "conf_bias":  "BUY" | "SELL" | "NEUTRAL",
        "r1_dist_atr": float,     # distância da resistência em ATRs
        "s1_dist_atr": float,     # distância do suporte em ATRs
        "note":       str
    }
"""

from __future__ import annotations

from typing import Any


_BIAS_MAP: dict[str, str] = {
    "buyers":  "BUY",
    "sellers": "SELL",
    "neutral": "NEUTRAL",
    "buy":     "BUY",
    "sell":    "SELL",
}

# Compressão: preço está entre S/R com gap < 0.5% do preço
_COMPRESSION_THRESHOLD_PCT = 0.005

# Muito próximo: < 0.15% do preço
_NEAR_THRESHOLD_PCT = 0.0015


def build_sr_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Gera resumo interpretado de suporte/resistência.

    Args:
        payload: payload já construído pelo build_compact_payload()

    Returns:
        dict com nearest, compressed, conf_bias, distâncias em ATR e note
    """
    sr = payload.get("sr", {})
    price_section = payload.get("price", {})
    regime = payload.get("regime", {})

    close = price_section.get("c") or 0
    atr_1h = _extract_atr(payload, "1h")

    if not sr or "_" in sr:
        return {
            "nearest":    "unknown",
            "compressed": False,
            "conf_bias":  "NEUTRAL",
            "note":       "Sem dados de S/R disponíveis.",
        }

    r1 = sr.get("r1")
    s1 = sr.get("s1")
    r1_p = r1[0] if r1 and isinstance(r1, list) else "?"
    s1_p = s1[0] if s1 and isinstance(s1, list) else "?"
    r1_dist = sr.get("r1_dist")
    s1_dist = sr.get("s1_dist")
    r1_conf = sr.get("r1_conf", 0)
    s1_conf = sr.get("s1_conf", 0)
    def_bias_raw = str(sr.get("def_bias", "neutral")).lower()
    conf_bias = _BIAS_MAP.get(def_bias_raw, "NEUTRAL")

    # --- Nearest ---
    nearest = "equidistant"
    if r1_dist is not None and s1_dist is not None:
        if r1_dist < s1_dist * 0.7:
            nearest = "resistance"
        elif s1_dist < r1_dist * 0.7:
            nearest = "support"

    # --- Compressão ---
    compressed = False
    gap_total = None
    if r1_dist is not None and s1_dist is not None and close > 0:
        gap_total = r1_dist + s1_dist
        gap_pct = gap_total / close
        compressed = gap_pct < _COMPRESSION_THRESHOLD_PCT

    # --- Distâncias em ATR ---
    r1_dist_atr = None
    s1_dist_atr = None
    if atr_1h and atr_1h > 0:
        if r1_dist is not None:
            r1_dist_atr = round(r1_dist / atr_1h, 2)
        if s1_dist is not None:
            s1_dist_atr = round(s1_dist / atr_1h, 2)

    # --- Levels muito próximos (alerta de teste iminente) ---
    r1_near = r1_dist is not None and close > 0 and r1_dist / close < _NEAR_THRESHOLD_PCT
    s1_near = s1_dist is not None and close > 0 and s1_dist / close < _NEAR_THRESHOLD_PCT

    # --- Nota interpretada ---
    note = _build_note(
        nearest=nearest,
        compressed=compressed,
        conf_bias=conf_bias,
        r1=r1,
        s1=s1,
        r1_p=r1_p,
        s1_p=s1_p,
        r1_dist=r1_dist,
        s1_dist=s1_dist,
        r1_conf=r1_conf,
        s1_conf=s1_conf,
        r1_dist_atr=r1_dist_atr,
        s1_dist_atr=s1_dist_atr,
        r1_near=r1_near,
        s1_near=s1_near,
        gap_total=gap_total,
    )

    result: dict[str, Any] = {
        "nearest":    nearest,
        "compressed": compressed,
        "conf_bias":  conf_bias,
        "note":       note,
    }

    if r1_dist_atr is not None:
        result["r1_dist_atr"] = r1_dist_atr
    if s1_dist_atr is not None:
        result["s1_dist_atr"] = s1_dist_atr
    if r1_near:
        result["r1_near"] = True
    if s1_near:
        result["s1_near"] = True
    if compressed:
        result["gap_usd"] = gap_total

    return result


def _extract_atr(compact_payload: dict, tf: str = "1h") -> float | None:
    tf_data = compact_payload.get("tf", {}).get(tf, {})
    atr = tf_data.get("atr")
    if atr and float(atr) > 0:
        return float(atr)
    return None


def _build_note(
    nearest: str,
    compressed: bool,
    conf_bias: str,
    r1: list | None,
    s1: list | None,
    r1_p: Any,
    s1_p: Any,
    r1_dist: int | None,
    s1_dist: int | None,
    r1_conf: int,
    s1_conf: int,
    r1_dist_atr: float | None,
    s1_dist_atr: float | None,
    r1_near: bool,
    s1_near: bool,
    gap_total: int | None,
) -> str:

    parts: list[str] = []

    # Compressão
    if compressed and gap_total is not None:
        parts.append(
            f"Preço comprimido entre S:{s1_p} e R:{r1_p} (gap total {gap_total} pts) "
            f"— risco de breakout elevado"
        )
    elif nearest == "resistance":
        dist_str = f"{r1_dist} pts"
        if r1_dist_atr is not None:
            dist_str += f" ({r1_dist_atr:.1f} ATR)"
        strength = r1[1] if r1 and len(r1) > 1 else "?"
        conf_str = f", confluência {r1_conf} fontes" if r1_conf >= 3 else ""
        parts.append(
            f"Resistência mais próxima em {r1[0] if r1 else '?'} "
            f"(força {strength}{conf_str}, dist {dist_str})"
        )
    elif nearest == "support":
        dist_str = f"{s1_dist} pts"
        if s1_dist_atr is not None:
            dist_str += f" ({s1_dist_atr:.1f} ATR)"
        strength = s1[1] if s1 and len(s1) > 1 else "?"
        conf_str = f", confluência {s1_conf} fontes" if s1_conf >= 3 else ""
        parts.append(
            f"Suporte mais próximo em {s1_p} "
            f"(força {strength}{conf_str}, dist {dist_str})"
        )
    else:
        parts.append("Preço equidistante entre suporte e resistência")

    # Testes iminentes
    if r1_near:
        if s1_near:
            parts.append(f"⚠️ Resistência acima ({r1_p}) — teste iminente. ⚠️ Suporte abaixo ({s1_p}) — teste iminente")
        else:
            parts.append(f"⚠️ Testando resistência importante em {r1_p}. Rompimento pode gerar aceleração.")
    elif s1_near:
        parts.append(f"⚠️ Testando suporte em {s1_p}. Defesa institucional ativa ou risco de quebra.")

    # Bias de defesa
    if conf_bias == "BUY":
        parts.append("Defesa compradora mais forte")
    elif conf_bias == "SELL":
        parts.append("Defesa vendedora mais forte")

    return ". ".join(parts) if parts else "Contexto de S/R indeterminado."
