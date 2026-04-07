"""
Tipos e helpers para o payload compacto enviado ao pipeline de IA.

Esta etapa centraliza o contrato mínimo do formato flat usado por:
- build_compact_payload.py
- market_orchestrator.ai.llm_payload_guardrail
- ai_analyzer_qwen.py
"""

from __future__ import annotations

from typing import Any, Dict, TypedDict

PayloadSection = Dict[str, Any]


class CompactAIPayload(TypedDict, total=False):
    # Identidade
    symbol: str
    epoch_ms: int
    trigger: str
    tipo_evento: str
    descricao: str
    ativo: str
    window: int

    # Seções principais
    price: PayloadSection
    regime: PayloadSection
    flow: PayloadSection
    ob: PayloadSection
    tf: PayloadSection
    sr: PayloadSection

    # Seções opcionais
    qual: PayloadSection
    w: PayloadSection
    ctx: PayloadSection
    ext: PayloadSection
    alerts: PayloadSection
    quant: PayloadSection
    summary: PayloadSection

    # Gaps opcionais
    ofi: PayloadSection
    vwap: PayloadSection
    liq: PayloadSection
    sm: PayloadSection
    cvd_div: PayloadSection
    mr: PayloadSection
    iceberg: PayloadSection

    # Metadados internos
    _v: int
    _compacted: str


class WrappedCompactAIPayload(TypedDict, total=False):
    ai_payload: CompactAIPayload
    tipo_evento: str
    descricao: str
    symbol: str
    ativo: str
    epoch_ms: int
    janela_numero: int


COMPACT_AI_PRIMARY_SECTION_KEYS: frozenset[str] = frozenset(
    {"price", "flow", "ob", "tf", "sr"}
)

COMPACT_AI_ALLOWED_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "symbol",
        "epoch_ms",
        "trigger",
        "tipo_evento",
        "descricao",
        "ativo",
        "window",
        "price",
        "regime",
        "flow",
        "ob",
        "tf",
        "sr",
        "qual",
        "w",
        "ctx",
        "ext",
        "alerts",
        "quant",
        "ofi",
        "vwap",
        "liq",
        "sm",
        "cvd_div",
        "mr",
        "iceberg",
        "summary",
        "_v",
        "_compacted",
    }
)

COMPACT_AI_COMPRESSED_MARKERS: frozenset[str] = frozenset(
    {"_v", "epoch_ms", "price", "ob", "flow"}
)


def compact_primary_section_count(payload: object) -> int:
    """Conta quantas seções primárias de payload compacto estão presentes."""
    if not isinstance(payload, dict):
        return 0
    return sum(1 for key in COMPACT_AI_PRIMARY_SECTION_KEYS if key in payload)


def is_compact_ai_payload(
    payload: object,
    *,
    require_identity: bool = False,
    minimum_primary_sections: int = 3,
) -> bool:
    """
    Detecta se um dict parece ser um payload compacto flat.

    `require_identity=True` é útil no guardrail.
    `require_identity=False` é útil em caminhos de compatibilidade.
    """
    if not isinstance(payload, dict):
        return False

    if "price" not in payload:
        return False

    if require_identity:
        has_epoch = isinstance(payload.get("epoch_ms"), (int, float))
        has_symbol = isinstance(payload.get("symbol"), str)
        if not (has_epoch and has_symbol):
            return False

    return compact_primary_section_count(payload) >= minimum_primary_sections


def is_wrapped_compact_ai_payload(payload: object) -> bool:
    """Detecta o wrapper `{'ai_payload': <compact_payload>}`."""
    if not isinstance(payload, dict):
        return False
    inner = payload.get("ai_payload")
    return is_compact_ai_payload(inner, require_identity=False)
