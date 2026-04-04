"""
regime_summary — Resume regime de mercado com estratégias recomendadas e proibidas.

Transforma:
    regime.cs, regime.cf, regime.v, regime.mode
    regime.bbw, regime.atr%
    tf.*.r (regime por timeframe)

Em:
    regime_summary: {
        "label":      str,    # descrição legível
        "strategies": list,   # o que fazer neste regime
        "avoid":      list,   # o que evitar
        "duration":   str,    # expectativa de duração
        "note":       str
    }
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Constantes por modo de mercado
# ---------------------------------------------------------------------------

_MODE_LABEL: dict[str, str] = {
    "MR":  "Mean Reversion",
    "RB":  "Range Bound",
    "TRD": "Trending",
    "BRK": "Breakout",
}

_MODE_STRATEGIES: dict[str, list[str]] = {
    "MR": [
        "fade extremos",
        "vender resistência",
        "comprar suporte",
        "aguardar absorção nos extremos",
    ],
    "RB": [
        "operar no range",
        "comprar VAL, vender VAH",
        "reduzir exposição fora do POC",
        "aguardar catalisador para breakout",
    ],
    "TRD": [
        "seguir tendência dominante",
        "comprar pullbacks no uptrend",
        "vender rallies no downtrend",
        "usar trailing stop",
    ],
    "BRK": [
        "aguardar confirmação de rompimento",
        "entrar no reteste do nível rompido",
        "usar stop apertado acima/abaixo do nível",
        "monitorar volume de confirmação",
    ],
}

_MODE_AVOID: dict[str, list[str]] = {
    "MR": [
        "perseguir momentum",
        "entrar no meio do range",
        "operar breakouts sem confirmação",
    ],
    "RB": [
        "comprar topo do range",
        "vender fundo do range",
        "usar alvos distantes",
    ],
    "TRD": [
        "operar contra a tendência",
        "mean reversion trades",
        "entradas no topo/fundo sem pullback",
    ],
    "BRK": [
        "entrar antes da confirmação",
        "ignorar falsos rompimentos",
        "operar range enquanto houver BRK ativo",
    ],
}

_MODE_DURATION: dict[str, str] = {
    "MR":  "15m – 2h tipicamente",
    "RB":  "horas a dias",
    "TRD": "horas a dias",
    "BRK": "minutos a horas — confirmar rápido",
}

_CONSENSUS_LABEL: dict[str, str] = {
    "BULL": "alta",
    "BEAR": "baixa",
    "MIX":  "indefinida",
}


def build_regime_summary(compact_payload: dict[str, Any]) -> dict[str, Any]:
    """
    Gera resumo interpretado do regime de mercado.

    Args:
        compact_payload: payload já construído pelo build_compact_payload()

    Returns:
        dict com label, strategies, avoid, duration e note
    """
    regime = compact_payload.get("regime", {})

    if not regime:
        return {
            "label":      "Indeterminado",
            "strategies": [],
            "avoid":      [],
            "duration":   "desconhecida",
            "note":       "Regime não disponível.",
        }

    mode = regime.get("mode", "RB")
    cs = regime.get("cs", "MIX")
    cf = regime.get("cf", 0.0) or 0.0
    vol = regime.get("v", "")
    bbw = regime.get("bbw")
    atr_pct = regime.get("atr%")
    dom_tf = regime.get("dom", "")

    label = _MODE_LABEL.get(mode, mode)
    strategies = _MODE_STRATEGIES.get(mode, [])
    avoid = _MODE_AVOID.get(mode, [])
    duration = _MODE_DURATION.get(mode, "indefinida")

    # --- Filtrar estratégias por consenso direcional ---
    cs_label = _CONSENSUS_LABEL.get(cs, "indefinida")

    if cs == "BEAR" and mode in ("TRD", "MR"):
        strategies = [s for s in strategies if "comprar" not in s.lower()]
    elif cs == "BULL" and mode in ("TRD", "MR"):
        strategies = [s for s in strategies if "vender" not in s.lower()]

    # --- Ajustar por volatilidade ---
    vol_note = ""
    if vol == "H":
        vol_note = "Volatilidade alta — stops maiores necessários"
    elif vol == "L":
        vol_note = "Volatilidade baixa — aguardar expansão antes de entrar"

    # --- Compressão via BBW ---
    bbw_note = ""
    if bbw is not None:
        if bbw < 0.10:
            bbw_note = "Bandas muito comprimidas — breakout iminente provável"
        elif bbw > 0.40:
            bbw_note = "Bandas expandidas — momentum ativo"

    # --- Confiança do consenso ---
    conf_label = "alta" if cf >= 0.7 else "média" if cf >= 0.4 else "baixa"

    note = _build_note(
        label=label,
        cs_label=cs_label,
        conf_label=conf_label,
        dom_tf=dom_tf,
        vol_note=vol_note,
        bbw_note=bbw_note,
        mode=mode,
        cf=cf,
    )

    result: dict[str, Any] = {
        "label":      label,
        "strategies": strategies,
        "avoid":      avoid,
        "duration":   duration,
        "note":       note,
    }

    if vol_note:
        result["vol_note"] = vol_note
    if bbw_note:
        result["bbw_note"] = bbw_note

    return result


def _build_note(
    label: str,
    cs_label: str,
    conf_label: str,
    dom_tf: str,
    vol_note: str,
    bbw_note: str,
    mode: str,
    cf: float,
) -> str:

    parts: list[str] = []

    parts.append(
        f"Regime {label} com consenso de {cs_label} "
        f"(confiança {conf_label}: {round(cf * 100)}%)"
    )

    if dom_tf:
        parts.append(f"dominado pelo {dom_tf}")

    if bbw_note:
        parts.append(bbw_note)

    if vol_note:
        parts.append(vol_note)

    if mode == "MR" and cs_label in ("alta", "baixa"):
        parts.append(
            f"Favorece reversão para equilíbrio — "
            f"não perseguir {'baixas' if cs_label == 'baixa' else 'altas'}"
        )

    return ". ".join(parts) + "."
