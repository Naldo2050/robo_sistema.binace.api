# common/signal_direction.py
"""
Módulo canônico e centralizado para inferência de direção de sinais de trading,
classificação de outcomes (Win/Loss/Flat) e resolução de confiança direcional.

Evita duplicação de regras e mapas de strings entre OutcomeTracker,
EventSimilarity, EventMemory e MarketOrchestrator.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Dict, Literal, Optional

# Tipos literais estritos
SignalSide = Literal["LONG", "SHORT", "NEUTRAL", "UNKNOWN"]
OutcomeResult = Literal["WIN", "LOSS", "FLAT", "UNKNOWN"]


def normalize_signal_label(text: str | None) -> str:
    """
    Normaliza texto removendo acentos/diacríticos, espaços extras e convertendo
    para letras maiúsculas.
    """
    if not text:
        return ""
    text_str = str(text).strip()
    # Decompor caracteres acentuados e descartar diacríticos (NFKD)
    normalized = unicodedata.normalize("NFKD", text_str)
    ascii_only = "".join(c for c in normalized if not unicodedata.combining(c))
    # Normalizar múltiplos espaços internos para um único espaço
    return " ".join(ascii_only.upper().split())


# ==============================================================================
# MAPAS CANÔNICOS DE RESULTADOS (Strings Normalizadas e Variantes Acentuadas)
# ==============================================================================

BULLISH_RESULTS = frozenset({
    "ABSORCAO DE VENDA",
    "ABSORÇÃO DE VENDA",
    "ABSORCAO DE VENDA (BULLISH)",
    "ABSORÇÃO DE VENDA (BULLISH)",
    "EXAUSTAO DE VENDA",
    "EXAUSTÃO DE VENDA",
    "DEMANDA NO LIVRO (BID>ASK)",
    "LEVE DEMANDA NO LIVRO",
    "DEMANDA FORTE",
    "COMPRA",
    "BULLISH",
    "SUPPLY_EXHAUSTION",
})

BEARISH_RESULTS = frozenset({
    "ABSORCAO DE COMPRA",
    "ABSORÇÃO DE COMPRA",
    "ABSORCAO DE COMPRA (BEARISH)",
    "ABSORÇÃO DE COMPRA (BEARISH)",
    "EXAUSTAO DE COMPRA",
    "EXAUSTÃO DE COMPRA",
    "OFERTA NO LIVRO (ASK>BID)",
    "LEVE OFERTA NO LIVRO",
    "VENCEDORES: VENDEDORES",
    "VENDEDOR_VENCEDOR",
    "VENDEDORES",
    "VENDA",
    "BEARISH",
    "DEMAND_EXHAUSTION",
})

NEUTRAL_RESULTS = frozenset({
    "SEM ABSORCAO",
    "SEM ABSORÇÃO",
    "SEM EXAUSTAO",
    "SEM EXAUSTÃO",
    "EQUILIBRIO",
    "EQUILÍBRIO",
    "NEUTRAL",
    "N/A",
    "NONE",
    "INDISPONIVEL",
    "INDISPONÍVEL",
    "DADOS INVALIDOS",
    "DADOS INVÁLIDOS",
    "JANELA VAZIA",
    "PRECOS INVALIDOS",
    "PREÇOS INVÁLIDOS",
    "ERRO",
    "EMERGENCIA",
    "EMERGÊNCIA",
    "VOLATILITY_EXPANSION",
    "VOLATILITY_SQUEEZE",
    "TEST",
    "ABS_TEST",
    "EXH_TEST",
    "OK",
})


def infer_signal_side(
    event_type: str | None = None,
    battle_result: str | None = None,
    explicit_side: str | None = None,
) -> SignalSide:
    """
    Infere a direção do sinal (LONG, SHORT, NEUTRAL ou UNKNOWN) segundo
    precedência estrita.

    Precedência:
    1. battle_result canônico normalizado (mais específico do detector);
    2. event_type normalizado (se battle_result estiver ausente/neutro e o tipo codificar direção);
    3. explicit_side normalizado (fallback secundário);
    4. UNKNOWN (nunca assume LONG por omissão).

    Args:
        event_type: Tipo do evento (ex: "Absorção", "Exaustão", "OrderBook").
        battle_result: Rótulo da batalha (ex: "Absorção de Venda", "Absorção de Compra").
        explicit_side: Campo side explícito opcional (ex: "buy", "sell", "long", "short").

    Returns:
        "LONG", "SHORT", "NEUTRAL" ou "UNKNOWN".
    """
    # 1. Precedência: battle_result canônico
    norm_battle = normalize_signal_label(battle_result)
    if norm_battle:
        if norm_battle in BULLISH_RESULTS:
            return "LONG"
        if norm_battle in BEARISH_RESULTS:
            return "SHORT"
        if norm_battle in NEUTRAL_RESULTS:
            return "NEUTRAL"

    # 2. Precedência: event_type (se codificar direção diretamente)
    norm_event = normalize_signal_label(event_type)
    if norm_event:
        if norm_event in BULLISH_RESULTS:
            return "LONG"
        if norm_event in BEARISH_RESULTS:
            return "SHORT"
        if norm_event in NEUTRAL_RESULTS:
            return "NEUTRAL"

    # 3. Precedência: explicit_side (fallback secundário)
    norm_side = normalize_signal_label(explicit_side)
    if norm_side in ("BUY", "LONG"):
        return "LONG"
    if norm_side in ("SELL", "SHORT"):
        return "SHORT"
    if norm_side in ("NEUTRAL", "NONE", "UNKNOWN"):
        return "NEUTRAL" if norm_side == "NEUTRAL" else "UNKNOWN"

    return "UNKNOWN"


def classify_outcome(
    signal_side: SignalSide | str,
    outcome_direction: str | None,
) -> OutcomeResult:
    """
    Classifica o resultado de um sinal com base na direção prevista e no movimento
    posterior do preço.

    Matriz de Classificação:
    - LONG + UP     -> WIN
    - LONG + DOWN   -> LOSS
    - SHORT + DOWN  -> WIN
    - SHORT + UP    -> LOSS
    - (LONG | SHORT) + FLAT -> FLAT
    - UNKNOWN / NEUTRAL / Outros -> UNKNOWN

    Args:
        signal_side: "LONG", "SHORT", "NEUTRAL" ou "UNKNOWN".
        outcome_direction: "UP", "DOWN" ou "FLAT".

    Returns:
        "WIN", "LOSS", "FLAT" ou "UNKNOWN".
    """
    side_clean = str(signal_side).strip().upper()
    dir_clean = str(outcome_direction).strip().upper() if outcome_direction else ""

    if side_clean not in ("LONG", "SHORT"):
        return "UNKNOWN"

    if dir_clean == "FLAT":
        return "FLAT"

    if side_clean == "LONG":
        if dir_clean == "UP":
            return "WIN"
        if dir_clean == "DOWN":
            return "LOSS"
        return "UNKNOWN"

    if side_clean == "SHORT":
        if dir_clean == "DOWN":
            return "WIN"
        if dir_clean == "UP":
            return "LOSS"
        return "UNKNOWN"

    return "UNKNOWN"


def get_directional_confidence(
    historical_confidence: Optional[Dict[str, Any]],
    signal_direction: str | None,
    default_fallback: float = 0.5,
) -> float:
    """
    Extrai a confiança estatística correspondente à direção do sinal.

    Garante que sinais SHORT utilizem short_prob e sinais LONG utilizem long_prob,
    com fallback seguro para sinais neutros, desconhecidos ou valores non-finite (NaN/Inf).

    Args:
        historical_confidence: Dicionário contendo "long_prob", "short_prob", etc.
        signal_direction: "long", "short", "neutral", etc.
        default_fallback: Valor padrão caso a probabilidade não esteja disponível (default 0.5).

    Returns:
        float representando a probabilidade/confiança direcional finita e segura.
    """
    if not historical_confidence or not isinstance(historical_confidence, dict):
        return default_fallback

    dir_clean = str(signal_direction or "").strip().lower()

    if dir_clean == "long":
        raw = historical_confidence.get("long_prob")
        if raw is not None:
            try:
                val = float(raw)
                if math.isfinite(val) and 0.0 <= val <= 1.0:
                    return val
            except (ValueError, TypeError):
                pass
        return default_fallback

    if dir_clean == "short":
        raw = historical_confidence.get("short_prob")
        if raw is not None:
            try:
                val = float(raw)
                if math.isfinite(val) and 0.0 <= val <= 1.0:
                    return val
            except (ValueError, TypeError):
                pass
        return default_fallback

    return default_fallback
