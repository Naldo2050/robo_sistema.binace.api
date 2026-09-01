"""
quality_summary — Resume qualidade dos dados e impacto na confiança da análise.

Transforma:
    qual.lat, qual.liq, qual.ms, qual.holiday, qual.src
    ctx.cached

Em:
    quality_summary: {
        "reliable":       bool,
        "confidence_cap": float,
        "issues":         list[str],
        "note":           str
    }

POLÍTICA EXPLÍCITA DE CAPS DE CONFIANÇA (ETAPA 3):

  - lat NORMAL/EXCE/GOOD/OK          = 1.0   (existente, ETAPA 2)
  - lat NEAR/ACCEPTABLE/ACCE         = 0.9   (existente, ETAPA 2)
  - lat DEGRADED/DEGR                = 0.7   (existente, ETAPA 2)
  - lat POOR/CRITICAL/CRIT           = 0.4   (existente, ETAPA 2)
  - lat desconhecida                 = 0.3   (existente, ETAPA 2 — fail-closed)
  - liq NORMAL                       = 1.0   (existente)
  - liq RED/REDUCED                  = 0.8   (existente; alias REDUCED completo
                                              adicionado na ETAPA 3)
  - liq LOW                          = 0.7   (existente)
  - liq VERY_LOW                     = 0.5   (existente)
  - liq desconhecida                 = 0.7   (NOVO na ETAPA 3 — política
                                              conservadora, equivalente a LOW)
  - src cache                        = 0.9   (NOVO na ETAPA 3)
  - src stale/fallback_rest/circuit_open/external/unknown/error
                                    = 0.8   (NOVO na ETAPA 3)
  - src emergency                    = 0.5   (NOVO no summary na ETAPA 3;
                                              peso -3.0 já existia no enricher)
  - feriado (qual.holiday)           = 0.6   (existente, ETAPA 2)
  - ctx.cached                       = 0.9   (existente, ETAPA 2)

Estes valores são constantes de código, não vêm de configuração externa;
representam política de risco definida na ETAPA 3 e podem ser revisados
futuramente.
"""

from __future__ import annotations

from typing import Any


_LATENCY_CAPS: dict[str, float] = {
    "OK":   1.0,
    "NEAR": 0.9,
    "DEGR": 0.7,
    "POOR": 0.4,
    "CRIT": 0.4,
    # FIX (ETAPA 2): categorias canônicas de time_manager.track_data_latency
    # (EXCELLENT/GOOD/ACCEPTABLE/DEGRADED/POOR/CRITICAL) e seus truncamentos
    # de 4 chars (payload_builder_compact envia cat[:4]).
    # Antes: EXCE/GOOD/ACCE caíam no fallback 0.3 — latência boa/aceitável
    # penalizada como desconhecida (falso negativo). Mapeamento usa os caps
    # existentes (OK/NEAR/DEGR/POOR/CRIT) por equivalência de freshness.
    "EXCELLENT": 1.0,
    "GOOD":      1.0,
    "ACCEPTABLE": 0.9,
    "DEGRADED":  0.7,
    "CRITICAL":  0.4,
    "EXCE":      1.0,
    "ACCE":      0.9,
}

_LIQUIDITY_CAPS: dict[str, float] = {
    "NORMAL":   1.0,
    # FIX (ETAPA 3): "REDUCED" é o valor canônico do produtor
    # (time_manager.get_market_calendar_context). Antes só existia "RED",
    # e "REDUCED" dependia de match parcial frágil por prefixo.
    "REDUCED":  0.8,
    "RED":      0.8,
    "LOW":      0.7,
    "VERY_LOW": 0.5,
    "VERY":     0.5,   # fallback para truncamento
    "VER":      0.5,   # fallback para truncamento [:3]
}

_LIQUIDITY_LABELS: dict[str, str] = {
    "NORMAL":   "normal",
    "REDUCED":  "reduzida",
    "RED":      "reduzida",
    "LOW":      "baixa",
    "VERY_LOW": "muito baixa",
    "VERY":     "muito baixa",
    "VER":      "muito baixa",
}

# FIX (ETAPA 3): origem do orderbook propagada em qual["src"] pelo
# payload_builder_compact (valores do orderbook_analyzer/core.py
# _last_fetch_source). Caps conservadores adicionados no padrão existente
# (ctx.cached=0.9, feriado=0.6): cache ~live (0.9), demais origens não-live
# degradadas (0.8), emergency com dados placeholder (0.5).
_SRC_CAPS: dict[str, float] = {
    "cache":         0.9,
    "stale":         0.8,
    "fallback_rest": 0.8,
    "circuit_open":  0.8,
    "external":      0.8,
    "unknown":       0.8,
    "error":         0.8,
    "emergency":     0.5,
}

_SRC_LABELS: dict[str, str] = {
    "cache":         "servido de cache (pode estar levemente defasado)",
    "stale":         "stale (fallback de dados antigos)",
    "fallback_rest": "via fallback REST",
    "circuit_open":  "com circuit breaker aberto",
    "external":      "de origem externa (não-live)",
    "unknown":       "de origem desconhecida",
    "error":         "com erro de fetch",
    "emergency":     "indisponível (emergency mode)",
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

    if cap is None:
        # FIX (ETAPA 3): categoria de liquidez desconhecida -> fail-closed.
        # Antes retornava (cap or 1.0) = 1.0 — dado desconhecido virava
        # "liquidez plena" no resumo da IA.
        return 0.7, "desconhecida"

    return cap, (label or key.lower())


def build_quality_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Gera resumo interpretado da qualidade dos dados.

    Args:
        payload: payload já construído pelo build_compact_payload()

    Returns:
        dict com reliable, confidence_cap, issues e note
    """
    qual = payload.get("qual", {})
    ctx = payload.get("ctx", {})

    issues: list[str] = []
    caps: list[float] = [1.0]

    # --- Latência ---
    # FIX (ETAPA 2): ausência de `lat` NÃO é "OK". O default
    # `str(qual.get("lat", "OK"))` promovia dado ausente a
    # "Dados em tempo real sem anomalias... confiança plena" (JANELA 1).
    # Novo contrato: latência desconhecida -> cap conservador (0.3, o mesmo
    # fallback já documentado para categoria desconhecida) + issue explícito.
    lat_raw = qual.get("lat") if isinstance(qual, dict) else None
    if lat_raw:
        lat_cat = str(lat_raw).upper()
        lat_ms = qual.get("ms")
        lat_cap = _LATENCY_CAPS.get(lat_cat, 0.3)  # fallback conservador
        caps.append(lat_cap)

        if lat_cat == "DEGR":
            msg = f"Latência degradada ({lat_ms}ms)" if lat_ms else "Latência degradada"
            issues.append(msg)
        elif lat_cat == "CRIT":
            msg = f"Latência crítica ({lat_ms}ms)" if lat_ms else "Latência crítica"
            issues.append(msg)
    else:
        lat_cat = None
        lat_ms = None
        caps.append(0.3)  # fallback conservador para dados ausentes
        issues.append("Latência desconhecida (freshness não confirmada)")

    # --- Liquidez ---
    # FIX (ETAPA 3): ausência de `liq` NÃO é "NORMAL". O default
    # `str(qual.get("liq", "NORMAL"))` promovia dado ausente a liquidez
    # plena (cap 1.0, sem issue) no resumo da IA. Novo contrato:
    # liquidez desconhecida -> cap conservador (0.7) + issue explícito.
    liq_raw = qual.get("liq") if isinstance(qual, dict) else None
    if liq_raw:
        liq_key = str(liq_raw).upper()
        liq_cap, liq_label = _resolve_liquidity(liq_key)
        caps.append(liq_cap)
        if liq_key not in ("NORMAL",):
            issues.append(f"Liquidez {liq_label}")
    else:
        liq_cap = 0.7
        liq_label = "desconhecida"
        caps.append(liq_cap)
        issues.append("Liquidez desconhecida (sem dado de calendário)")

    # --- Fonte do orderbook (stale/cache/fallback/emergency) ---
    # FIX (ETAPA 3): origem não-live do orderbook propagada em qual["src"].
    # Antes a IA nunca sabia que o orderbook veio de stale/fallback/cache/
    # emergency e o resumo podia afirmar "dados em tempo real... plena".
    src_raw = qual.get("src") if isinstance(qual, dict) else None
    if src_raw:
        src_key = str(src_raw).lower()
        src_cap = _SRC_CAPS.get(src_key)
        if src_cap is not None:
            caps.append(src_cap)
            issues.append(f"Orderbook {_SRC_LABELS.get(src_key, 'de origem desconhecida')}")

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

    return {
        "reliable":       reliable,
        "confidence_cap": confidence_cap,
        "issues":         issues,
    }


def _build_note(
    reliable: bool,
    confidence_cap: float,
    issues: list[str],
    lat_cat: str,
    liq_raw: str | None,
    src_raw: str | None = None,
    ctx_cached: bool = False,
    holiday: str | None = None,
) -> str:

    if reliable:
        return "Dados em tempo real sem anomalias. Análise com confiança plena."

    parts: list[str] = []

    if lat_cat in ("DEGR", "CRIT"):
        severity = "crítica" if lat_cat == "CRIT" else "degradada"
        parts.append(f"Latência {severity} compromete freshness dos dados")
    elif lat_cat is None:
        parts.append("Latência desconhecida compromete freshness dos dados")

    if holiday:
        parts.append(f"Feriado ({holiday}) reduz liquidez severamente")
    elif not liq_raw:
        # FIX (ETAPA 3): liq_raw agora pode ser None — nunca crashar.
        parts.append("Liquidez desconhecida pode distorcer sinais de fluxo e orderbook")
    elif liq_raw.upper() not in ("NORMAL",):
        parts.append("Liquidez reduzida pode distorcer sinais de fluxo e orderbook")

    # FIX (ETAPA 3): origem não-live do orderbook em nota.
    if src_raw and str(src_raw).lower() != "live":
        label = _SRC_LABELS.get(str(src_raw).lower())
        if label:
            parts.append(f"Orderbook {label}")

    if ctx_cached:
        parts.append("Contexto macro/derivativos pode estar desatualizado")

    cap_pct = round(confidence_cap * 100)
    parts.append(f"Confiança máxima desta análise: {cap_pct}%")

    return ". ".join(parts) + "."
