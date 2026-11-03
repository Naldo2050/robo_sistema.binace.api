# common/financial_validator.py
"""
Validador Semântico Financeiro Puro.

Fase 5.11 - Gate H1-B2-B1:
Componente funcional determinístico, imutável, sem I/O, sem efeitos colaterais
e sem dependências externas, projetado para validar dados financeiros e de
preço antes de qualquer sanitização destrutiva (como sanitize_json_safe).
"""

from __future__ import annotations

import decimal
import math
from dataclasses import dataclass
from typing import Any, Dict, Final, FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class FinancialValidationResult:
    """
    Resultado imutável e estruturado da validação semântica financeira.

    Distingue três estados lógicos estritos:
    - VALID: Evento classificado com sucesso e aprovado em todas as regras.
    - INVALID: Evento classificado que violou uma regra semântica financeira.
    - UNCLASSIFIED: Evento cujo tipo/contrato não possui mapeamento seguro
      para atestar validade financeira (fail-closed defensivo).
    """

    is_valid: bool
    status: str  # "VALID", "INVALID", "UNCLASSIFIED"
    category: str
    rejected_field: Optional[str] = None
    rejected_value_repr: Optional[str] = None
    rejection_reason: Optional[str] = None
    requires_classification: bool = False

    def __post_init__(self) -> None:
        """
        Validação defensiva de invariantes lógicas estritas (Gate H1-B2-Q1).
        Impede combinações contraditórias de atributos no resultado imutável.
        """
        if self.status == "VALID":
            if not self.is_valid:
                raise ValueError("Invariante violada: status 'VALID' exige is_valid=True")
            if self.requires_classification:
                raise ValueError("Invariante violada: status 'VALID' exige requires_classification=False")
            if self.rejected_field is not None:
                raise ValueError("Invariante violada: status 'VALID' exige rejected_field=None")
            if self.rejected_value_repr is not None:
                raise ValueError("Invariante violada: status 'VALID' exige rejected_value_repr=None")
            if self.rejection_reason is not None:
                raise ValueError("Invariante violada: status 'VALID' exige rejection_reason=None")
        elif self.status == "INVALID":
            if self.is_valid:
                raise ValueError("Invariante violada: status 'INVALID' exige is_valid=False")
            if self.requires_classification:
                raise ValueError("Invariante violada: status 'INVALID' exige requires_classification=False")
            if self.rejected_field is not None and not isinstance(self.rejected_field, str):
                raise ValueError("Invariante violada: status 'INVALID' exige rejected_field como None ou str")
            if self.rejection_reason is None or not isinstance(self.rejection_reason, str) or not self.rejection_reason.strip():
                raise ValueError("Invariante violada: status 'INVALID' exige rejection_reason como string não vazia")
        elif self.status == "UNCLASSIFIED":
            if self.is_valid:
                raise ValueError("Invariante violada: status 'UNCLASSIFIED' exige is_valid=False")
            if not self.requires_classification:
                raise ValueError("Invariante violada: status 'UNCLASSIFIED' exige requires_classification=True")
            if self.rejection_reason is None or not isinstance(self.rejection_reason, str) or not self.rejection_reason.strip():
                raise ValueError("Invariante violada: status 'UNCLASSIFIED' exige rejection_reason como string não vazia")
        else:
            raise ValueError(
                f"Invariante violada: status desconhecido '{self.status}'. "
                "Deve ser 'VALID', 'INVALID' ou 'UNCLASSIFIED'"
            )

    @property
    def is_accepted(self) -> bool:
        """Retorna True apenas se o evento foi classificado e considerado válido."""
        return self.status == "VALID" and self.is_valid

    @property
    def is_rejected(self) -> bool:
        """Retorna True se o evento foi classificado e reprovado semanticamente."""
        return self.status == "INVALID"

    @property
    def is_unclassified(self) -> bool:
        """Retorna True se o evento não pôde ser classificado por falta de contrato."""
        return self.status == "UNCLASSIFIED"


# Conjuntos imutáveis de tipos conhecidos de eventos do robô
KNOWN_SIGNAL_TYPES: Final[FrozenSet[str]] = frozenset(
    {"Absorção", "Absorcao", "Exaustão", "Exaustao"}
)
KNOWN_TRIGGER_TYPES: Final[FrozenSet[str]] = frozenset({"ANALYSIS_TRIGGER"})
KNOWN_ZONE_TYPES: Final[FrozenSet[str]] = frozenset({"Zona", "Zone"})
KNOWN_ALERT_TYPES: Final[FrozenSet[str]] = frozenset(
    {"Alerta", "Alert", "institutional_alert"}
)
KNOWN_AI_TYPES: Final[FrozenSet[str]] = frozenset({"AI_ANALYSIS"})

# Chaves reservadas de preços e valores financeiros
FINANCIAL_PRICE_KEYS: Final[FrozenSet[str]] = frozenset(
    {
        "preco_fechamento",
        "price",
        "preco",
        "fill_price",
        "raw_price",
        "entry_price",
        "exit_price",
        "anchor_price",
        "bid_price",
        "ask_price",
        "notional",
        "notional_usdt",
    }
)

# Tokens textuais degenerados de números não finitos
DEGENERATE_NUMERIC_TOKENS: Final[FrozenSet[str]] = frozenset(
    {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
        "__nan__",
        "__infinity__",
        "__-infinity__",
        "none",
        "null",
        "n/a",
    }
)


def _safe_repr(val: Any, max_len: int = 80) -> str:
    """Produz uma representação textual defensiva e limitada sem invocar reprs hostis."""
    try:
        if isinstance(val, (int, float, bool, str, decimal.Decimal)):
            s = str(val)
        elif val is None:
            s = "None"
        else:
            s = f"<{type(val).__name__}>"
    except Exception:
        s = "<unrepresentable>"
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _validate_numeric_value(
    value: Any,
    field_name: str,
    must_be_positive: bool = True,
    allow_zero: bool = False,
    allow_negative: bool = False,
    is_optional: bool = False,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Inspeciona um valor numérico sem converter ou mutar o payload.

    Retorna: (is_ok, error_reason, value_repr)
    """
    # 1. Tratamento de None
    if value is None:
        if is_optional:
            return True, None, None
        return False, "MANDATORY_FIELD_IS_NONE", "None"

    # 2. Booleanos não são números financeiros válidos (bool é subclasse de int em Python)
    if isinstance(value, bool):
        return False, "BOOLEAN_IN_NUMERIC_FIELD", _safe_repr(value)

    # 3. Containers complexos não são valores numéricos escalares
    if isinstance(value, (list, dict, tuple, set)):
        return (
            False,
            f"UNSUPPORTED_CONTAINER_TYPE_{type(value).__name__.upper()}",
            _safe_repr(value),
        )

    # 4. Floats e Ints nativos
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if math.isnan(value):
                return False, "NAN_IN_NUMERIC_FIELD", "nan"
            if math.isinf(value):
                reason = "POS_INF_IN_NUMERIC_FIELD" if value > 0 else "NEG_INF_IN_NUMERIC_FIELD"
                return False, reason, _safe_repr(value)

        # Checagens de sinal e magnitude relativa
        if must_be_positive:
            if value <= 0:
                reason = (
                    "NUMERIC_VALUE_MUST_BE_POSITIVE_ZERO"
                    if value == 0
                    else "NUMERIC_VALUE_MUST_BE_POSITIVE_NEGATIVE"
                )
                return False, reason, _safe_repr(value)
            return True, None, None

        if not allow_negative and value < 0:
            return False, "NEGATIVE_VALUE_NOT_ALLOWED", _safe_repr(value)

        if not allow_zero and value == 0:
            return False, "ZERO_VALUE_NOT_ALLOWED", _safe_repr(value)

        return True, None, None

    # 5. Objetos Decimal da biblioteca padrão
    if isinstance(value, decimal.Decimal):
        if value.is_nan():
            return False, "NAN_IN_NUMERIC_FIELD", "nan"
        if value.is_infinite():
            reason = "POS_INF_IN_NUMERIC_FIELD" if value > 0 else "NEG_INF_IN_NUMERIC_FIELD"
            return False, reason, _safe_repr(value)

        if must_be_positive:
            if value <= 0:
                reason = (
                    "NUMERIC_VALUE_MUST_BE_POSITIVE_ZERO"
                    if value == 0
                    else "NUMERIC_VALUE_MUST_BE_POSITIVE_NEGATIVE"
                )
                return False, reason, _safe_repr(value)
            return True, None, None

        if not allow_negative and value < 0:
            return False, "NEGATIVE_VALUE_NOT_ALLOWED", _safe_repr(value)

        if not allow_zero and value == 0:
            return False, "ZERO_VALUE_NOT_ALLOWED", _safe_repr(value)

        return True, None, None

    # 6. Strings numéricas (incluindo representações decimais e notação científica)
    if isinstance(value, str):
        s_clean = value.strip()
        s_lower = s_clean.lower()

        if s_lower in DEGENERATE_NUMERIC_TOKENS:
            return False, "DEGENERATE_STRING_TOKEN", _safe_repr(s_clean)

        # Validação de formato decimal sem mutar o objeto de entrada
        try:
            d = decimal.Decimal(s_clean)
        except (decimal.InvalidOperation, TypeError, ValueError):
            return False, "MALFORMED_NUMERIC_STRING", _safe_repr(s_clean)

        if d.is_nan():
            return False, "NAN_IN_NUMERIC_FIELD", "nan"
        if d.is_infinite():
            reason = "POS_INF_IN_NUMERIC_FIELD" if d > 0 else "NEG_INF_IN_NUMERIC_FIELD"
            return False, reason, _safe_repr(s_clean)

        if must_be_positive:
            if d <= 0:
                reason = (
                    "NUMERIC_VALUE_MUST_BE_POSITIVE_ZERO"
                    if d == 0
                    else "NUMERIC_VALUE_MUST_BE_POSITIVE_NEGATIVE"
                )
                return False, reason, _safe_repr(s_clean)
            return True, None, None

        if not allow_negative and d < 0:
            return False, "NEGATIVE_VALUE_NOT_ALLOWED", _safe_repr(s_clean)

        if not allow_zero and d == 0:
            return False, "ZERO_VALUE_NOT_ALLOWED", _safe_repr(s_clean)

        return True, None, None

    # 7. Tipos arbitrários não suportados
    return (
        False,
        f"UNSUPPORTED_NUMERIC_TYPE_{type(value).__name__.upper()}",
        _safe_repr(value),
    )


def validate_financial_event_semantics(event: Any) -> FinancialValidationResult:
    """
    Valida a semântica financeira de um evento de forma estritamente pura.

    Não produz I/O, não muta o dicionário recebido e não possui dependências externas.
    """
    # 1. Validação de tipo estrutural básico
    if not isinstance(event, dict):
        return FinancialValidationResult(
            is_valid=False,
            status="INVALID",
            category="MALFORMED",
            rejection_reason="EVENT_NOT_A_DICT",
            rejected_value_repr=_safe_repr(event),
        )

    # 2. Extração determinística do discriminador de evento
    raw_type = event.get("tipo_evento") or event.get("event_type") or event.get("type")
    if raw_type is None:
        has_price_key = any(k in event for k in FINANCIAL_PRICE_KEYS)
        return FinancialValidationResult(
            is_valid=False,
            status="UNCLASSIFIED",
            category="UNCLASSIFIED",
            requires_classification=True,
            rejected_field="tipo_evento",
            rejection_reason=(
                "MISSING_EVENT_TYPE_WITH_FINANCIAL_FIELDS"
                if has_price_key
                else "MISSING_EVENT_TYPE_NO_CONTRACT"
            ),
        )

    event_type = str(raw_type).strip()

    # -------------------------------------------------------------
    # CATEGORIA 1: SINAIS FINANCEIROS INSTITUCIONAIS (Absorção / Exaustão)
    # -------------------------------------------------------------
    if event_type in KNOWN_SIGNAL_TYPES:
        cat = "FINANCIAL_SIGNAL"

        # Campo obrigatório comprovado no histórico: preco_fechamento
        if "preco_fechamento" not in event:
            return FinancialValidationResult(
                is_valid=False,
                status="INVALID",
                category=cat,
                rejected_field="preco_fechamento",
                rejection_reason="MANDATORY_FIELD_MISSING",
                rejected_value_repr="None",
            )

        ok, reason, val_repr = _validate_numeric_value(
            event["preco_fechamento"],
            "preco_fechamento",
            must_be_positive=True,
            is_optional=False,
        )
        if not ok:
            return FinancialValidationResult(
                is_valid=False,
                status="INVALID",
                category=cat,
                rejected_field="preco_fechamento",
                rejected_value_repr=val_repr,
                rejection_reason=reason,
            )

        # Campos opcionais de volume e microestrutura
        if "volume_total" in event:
            ok, reason, val_repr = _validate_numeric_value(
                event["volume_total"],
                "volume_total",
                must_be_positive=False,
                allow_zero=True,
                allow_negative=False,
                is_optional=True,
            )
            if not ok:
                return FinancialValidationResult(
                    is_valid=False,
                    status="INVALID",
                    category=cat,
                    rejected_field="volume_total",
                    rejected_value_repr=val_repr,
                    rejection_reason=reason,
                )

        if "delta" in event:
            ok, reason, val_repr = _validate_numeric_value(
                event["delta"],
                "delta",
                must_be_positive=False,
                allow_zero=True,
                allow_negative=True,
                is_optional=True,
            )
            if not ok:
                return FinancialValidationResult(
                    is_valid=False,
                    status="INVALID",
                    category=cat,
                    rejected_field="delta",
                    rejected_value_repr=val_repr,
                    rejection_reason=reason,
                )

        for price_opt in ("poc", "val", "vah"):
            if price_opt in event:
                ok, reason, val_repr = _validate_numeric_value(
                    event[price_opt],
                    price_opt,
                    must_be_positive=True,
                    is_optional=True,
                )
                if not ok:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category=cat,
                        rejected_field=price_opt,
                        rejected_value_repr=val_repr,
                        rejection_reason=reason,
                    )

        # Inspeção de subestrutura aninhada comprovada: raw_event
        raw_evt = event.get("raw_event")
        if isinstance(raw_evt, dict) and raw_evt is not event:
            if "preco_fechamento" in raw_evt:
                ok, reason, val_repr = _validate_numeric_value(
                    raw_evt["preco_fechamento"],
                    "raw_event.preco_fechamento",
                    must_be_positive=True,
                    is_optional=False,
                )
                if not ok:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category=cat,
                        rejected_field="raw_event.preco_fechamento",
                        rejected_value_repr=val_repr,
                        rejection_reason=reason,
                    )

        return FinancialValidationResult(is_valid=True, status="VALID", category=cat)

    # -------------------------------------------------------------
    # CATEGORIA 2: GATILHOS ANALÍTICOS (ANALYSIS_TRIGGER)
    # -------------------------------------------------------------
    if event_type in KNOWN_TRIGGER_TYPES:
        cat = "ANALYTICAL_TRIGGER"

        # Campos de preço de referência, se presentes, devem ser finitos e positivos
        for price_field in ("preco_fechamento", "bid_price", "ask_price", "anchor_price"):
            if price_field in event:
                ok, reason, val_repr = _validate_numeric_value(
                    event[price_field],
                    price_field,
                    must_be_positive=True,
                    is_optional=True,
                )
                if not ok:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category=cat,
                        rejected_field=price_field,
                        rejected_value_repr=val_repr,
                        rejection_reason=reason,
                    )

        if "volume_total" in event:
            ok, reason, val_repr = _validate_numeric_value(
                event["volume_total"],
                "volume_total",
                must_be_positive=False,
                allow_zero=True,
                allow_negative=False,
                is_optional=True,
            )
            if not ok:
                return FinancialValidationResult(
                    is_valid=False,
                    status="INVALID",
                    category=cat,
                    rejected_field="volume_total",
                    rejected_value_repr=val_repr,
                    rejection_reason=reason,
                )

        if "delta" in event:
            ok, reason, val_repr = _validate_numeric_value(
                event["delta"],
                "delta",
                must_be_positive=False,
                allow_zero=True,
                allow_negative=True,
                is_optional=True,
            )
            if not ok:
                return FinancialValidationResult(
                    is_valid=False,
                    status="INVALID",
                    category=cat,
                    rejected_field="delta",
                    rejected_value_repr=val_repr,
                    rejection_reason=reason,
                )

        # Indicadores técnicos (rsi, macd, etc.) são analíticos e admitem NaN/None.
        # Não são inspecionados para rejeição financeira.
        return FinancialValidationResult(is_valid=True, status="VALID", category=cat)

    # -------------------------------------------------------------
    # CATEGORIA 3: ZONAS DE MICROESTRUTURA (Zona)
    # -------------------------------------------------------------
    if event_type in KNOWN_ZONE_TYPES:
        cat = "ZONE_EVENT"

        for price_field in ("level", "price"):
            if price_field in event:
                ok, reason, val_repr = _validate_numeric_value(
                    event[price_field],
                    price_field,
                    must_be_positive=True,
                    is_optional=True,
                )
                if not ok:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category=cat,
                        rejected_field=price_field,
                        rejected_value_repr=val_repr,
                        rejection_reason=reason,
                    )

        if "volume_total" in event:
            ok, reason, val_repr = _validate_numeric_value(
                event["volume_total"],
                "volume_total",
                must_be_positive=False,
                allow_zero=True,
                allow_negative=False,
                is_optional=True,
            )
            if not ok:
                return FinancialValidationResult(
                    is_valid=False,
                    status="INVALID",
                    category=cat,
                    rejected_field="volume_total",
                    rejected_value_repr=val_repr,
                    rejection_reason=reason,
                )

        if "delta" in event:
            ok, reason, val_repr = _validate_numeric_value(
                event["delta"],
                "delta",
                must_be_positive=False,
                allow_zero=True,
                allow_negative=True,
                is_optional=True,
            )
            if not ok:
                return FinancialValidationResult(
                    is_valid=False,
                    status="INVALID",
                    category=cat,
                    rejected_field="delta",
                    rejected_value_repr=val_repr,
                    rejection_reason=reason,
                )

        return FinancialValidationResult(is_valid=True, status="VALID", category=cat)

    # -------------------------------------------------------------
    # CATEGORIA 4: ALERTAS OPERACIONAIS (Alerta)
    # -------------------------------------------------------------
    if event_type in KNOWN_ALERT_TYPES:
        cat = "ALERT_EVENT"

        for price_field in ("price", "level"):
            if price_field in event:
                ok, reason, val_repr = _validate_numeric_value(
                    event[price_field],
                    price_field,
                    must_be_positive=True,
                    is_optional=True,
                )
                if not ok:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category=cat,
                        rejected_field=price_field,
                        rejected_value_repr=val_repr,
                        rejection_reason=reason,
                    )

        if "volume" in event:
            ok, reason, val_repr = _validate_numeric_value(
                event["volume"],
                "volume",
                must_be_positive=False,
                allow_zero=True,
                allow_negative=False,
                is_optional=True,
            )
            if not ok:
                return FinancialValidationResult(
                    is_valid=False,
                    status="INVALID",
                    category=cat,
                    rejected_field="volume",
                    rejected_value_repr=val_repr,
                    rejection_reason=reason,
                )

        # Inspeciona subestrutura aninhada contextual
        ctx = event.get("context")
        if isinstance(ctx, dict) and ctx is not event:
            if "price" in ctx:
                ok, reason, val_repr = _validate_numeric_value(
                    ctx["price"],
                    "context.price",
                    must_be_positive=True,
                    is_optional=True,
                )
                if not ok:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category=cat,
                        rejected_field="context.price",
                        rejected_value_repr=val_repr,
                        rejection_reason=reason,
                    )
            if "volume" in ctx:
                ok, reason, val_repr = _validate_numeric_value(
                    ctx["volume"],
                    "context.volume",
                    must_be_positive=False,
                    allow_zero=True,
                    allow_negative=False,
                    is_optional=True,
                )
                if not ok:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category=cat,
                        rejected_field="context.volume",
                        rejected_value_repr=val_repr,
                        rejection_reason=reason,
                    )

        return FinancialValidationResult(is_valid=True, status="VALID", category=cat)

    # -------------------------------------------------------------
    # CATEGORIA 5: DIAGNÓSTICO DE IA (AI_ANALYSIS)
    # -------------------------------------------------------------
    if event_type in KNOWN_AI_TYPES:
        cat = "AI_DIAGNOSTIC"

        if "anchor_price" in event:
            ok, reason, val_repr = _validate_numeric_value(
                event["anchor_price"],
                "anchor_price",
                must_be_positive=True,
                is_optional=True,
            )
            if not ok:
                return FinancialValidationResult(
                    is_valid=False,
                    status="INVALID",
                    category=cat,
                    rejected_field="anchor_price",
                    rejected_value_repr=val_repr,
                    rejection_reason=reason,
                )

        return FinancialValidationResult(is_valid=True, status="VALID", category=cat)

    # -------------------------------------------------------------
    # EVENTOS NÃO CLASSIFICADOS (UNCLASSIFIED)
    # -------------------------------------------------------------
    has_price_field = any(k in event for k in FINANCIAL_PRICE_KEYS)
    return FinancialValidationResult(
        is_valid=False,
        status="UNCLASSIFIED",
        category="UNCLASSIFIED",
        requires_classification=True,
        rejected_field="tipo_evento",
        rejected_value_repr=_safe_repr(event_type),
        rejection_reason=(
            "UNKNOWN_EVENT_TYPE_WITH_FINANCIAL_FIELDS"
            if has_price_field
            else "UNKNOWN_EVENT_TYPE_NO_CONTRACT"
        ),
    )
