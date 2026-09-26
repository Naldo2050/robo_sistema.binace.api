# institutional/evidence.py
"""
Evidence Contract v1.0.0 — Contrato formal e versionado de evidências.

Escopo P1-A (somente contrato, sem fiação produtiva):
- Representa UMA evidência observada com proveniência, validade e calibração
  explícitas. Não decide, não pondera, não agrega.
- Coexiste com o `Signal` legado (`institutional/base.py`): os 17 analisadores
  atuais NÃO são tocados nesta fase. Conversão legada somente via
  `evidence_from_signal` (pura, sem inferência semântica).
- `counts_as_vote` default FALSE: nada conta como voto independente sem
  declaração explícita (P1-B/P2 decidem agregação, TTL, caps e dedup).

Regras fail-closed:
- `source` obrigatório e não vazio (ValueError caso contrário).
- Enums aceitam o membro ou a string exata do valor; string desconhecida
  levanta ValueError (nunca chuta UNKNOWN silenciosamente a partir de lixo).
- `value` non-finite (NaN/±Inf) NUNCA vira número: coagido para None com
  `validity=INVALID` e `reason` preservado + `NONFINITE_VALUE`.
- `from_dict` exige `feature_contract_version == "1.0.0"`.
- Sem import de capabilities/confluence/payload: este módulo é folha
  (só stdlib + `institutional.base` para tipagem do adapter). Sem ciclo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from institutional.base import Signal

FEATURE_CONTRACT_VERSION = "1.0.0"


class EvidenceType(str, Enum):
    """Origem física da observação."""
    CONTINUOUS_TRADES = "continuous_trades"
    POINT_IN_TIME_L2 = "point_in_time_l2"
    SLOW_CONTEXT = "slow_context"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class EvidenceFamily(str, Enum):
    """Família analítica (sem peso; P1-B define agrupamento, nunca aqui)."""
    EXECUTED_FLOW = "executed_flow"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    PRICE_RESPONSE = "price_response"
    MARKET_STRUCTURE = "market_structure"
    DERIVATIVES = "derivatives"
    CROSS_ASSET = "cross_asset"
    MACRO = "macro"
    UNKNOWN = "unknown"


class EvidenceDirection(str, Enum):
    """Direção explícita. Nunca LONG/SHORT; nunca inferida de rótulo."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class EvidenceValidity(str, Enum):
    """Validade. UNKNOWN nunca conta como VALID (contrato, sem TTL aqui)."""
    VALID = "valid"
    PARTIAL = "partial"
    INVALID = "invalid"
    STALE = "stale"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class EvidenceCalibration(str, Enum):
    """Estado de calibração. Nada é CALIBRATED sem evidência documentada."""
    CALIBRATED = "calibrated"
    UNCALIBRATED_HEURISTIC = "uncalibrated_heuristic"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


def _json_safe(value: Any) -> Any:
    """Recursivo: floats non-finite => None (convenção do repo: json_safe
    NaN/±Inf => null; nunca 0). Dicts têm chaves ordenadas (determinismo)."""
    if isinstance(value, dict):
        return {k: _json_safe(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _coerce_enum(enum_cls: type, value: Any, field_name: str):
    """Membro do enum ou string exata; ValueError em string desconhecida."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            raise ValueError(
                f"Evidence.{field_name}: valor desconhecido {value!r} "
                f"(use um membro de {enum_cls.__name__})"
            ) from None
    raise ValueError(
        f"Evidence.{field_name}: esperado {enum_cls.__name__} ou str, "
        f"recebido {type(value).__name__}"
    )


@dataclass
class Evidence:
    """Uma evidência observada (v1.0.0)."""
    source: str
    family: Any = EvidenceFamily.UNKNOWN
    evidence_type: Any = EvidenceType.UNKNOWN
    direction: Any = EvidenceDirection.UNKNOWN
    value: Optional[float] = None
    observed_at_ms: Optional[int] = None
    horizon_ms: Optional[int] = None
    provenance: dict[str, Any] = field(default_factory=dict)
    validity: Any = EvidenceValidity.UNKNOWN
    reason: Optional[str] = None
    calibration: Any = EvidenceCalibration.UNKNOWN
    counts_as_vote: bool = False
    derived_from: tuple = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_contract_version: str = FEATURE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Evidence.source obrigatório e não vazio")
        self.family = _coerce_enum(EvidenceFamily, self.family, "family")
        self.evidence_type = _coerce_enum(EvidenceType, self.evidence_type,
                                          "evidence_type")
        self.direction = _coerce_enum(EvidenceDirection, self.direction,
                                       "direction")
        self.validity = _coerce_enum(EvidenceValidity, self.validity,
                                     "validity")
        self.calibration = _coerce_enum(EvidenceCalibration, self.calibration,
                                        "calibration")
        if not isinstance(self.derived_from, (tuple, list)):
            raise ValueError("Evidence.derived_from: esperado tuple/list")
        self.derived_from = tuple(str(d) for d in self.derived_from)
        # value non-finite nunca vira número: null + INVALID + rastro.
        if self.value is not None:
            if isinstance(self.value, bool):
                raise ValueError("Evidence.value: bool não é evidência numérica")
            try:
                v = float(self.value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Evidence.value: não numérico {self.value!r}") from None
            if not math.isfinite(v):
                self.value = None
                self.validity = EvidenceValidity.INVALID
                self.reason = (f"{self.reason};NONFINITE_VALUE"
                               if self.reason else "NONFINITE_VALUE")
            else:
                self.value = v

    def to_dict(self) -> dict[str, Any]:
        """Serialização determinística, RFC8259-safe, enums como strings."""
        return {
            "feature_contract_version": self.feature_contract_version,
            "source": self.source,
            "family": self.family.value,
            "evidence_type": self.evidence_type.value,
            "direction": self.direction.value,
            "value": self.value,
            "observed_at_ms": self.observed_at_ms,
            "horizon_ms": self.horizon_ms,
            "provenance": _json_safe(dict(self.provenance)),
            "validity": self.validity.value,
            "reason": self.reason,
            "calibration": self.calibration.value,
            "counts_as_vote": bool(self.counts_as_vote),
            "derived_from": list(self.derived_from),
            "metadata": _json_safe(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        """Round-trip estrito: exige version 1.0.0 presente e igual."""
        if not isinstance(data, dict):
            raise ValueError("Evidence.from_dict: esperado dict")
        version = data.get("feature_contract_version")
        if version != FEATURE_CONTRACT_VERSION:
            raise ValueError(
                "Evidence.from_dict: feature_contract_version ausente ou "
                f"incompatível ({version!r}); esperado "
                f"{FEATURE_CONTRACT_VERSION!r}")
        return cls(
            source=data.get("source", ""),
            family=data.get("family", EvidenceFamily.UNKNOWN),
            evidence_type=data.get("evidence_type", EvidenceType.UNKNOWN),
            direction=data.get("direction", EvidenceDirection.UNKNOWN),
            value=data.get("value"),
            observed_at_ms=data.get("observed_at_ms"),
            horizon_ms=data.get("horizon_ms"),
            provenance=dict(data.get("provenance") or {}),
            validity=data.get("validity", EvidenceValidity.UNKNOWN),
            reason=data.get("reason"),
            calibration=data.get("calibration", EvidenceCalibration.UNKNOWN),
            counts_as_vote=bool(data.get("counts_as_vote", False)),
            derived_from=tuple(data.get("derived_from") or ()),
            metadata=dict(data.get("metadata") or {}),
        )


def evidence_from_signal(signal: Signal) -> Evidence:
    """Adapter puro Signal legado -> Evidence v1. Sem inferência semântica.

    - `source` vem do signal (vazio => ValueError, nunca inventado).
    - family/type/provenance/direction: sempre UNKNOWN/{} (o Signal não os
      fornece; mapear Side.BUY->BULLISH seria inferência — proibido aqui,
      inclusive para rótulos tipo "COMPRA").
    - `observed_at_ms`: carrega `signal.timestamp` (dado direto, não inferência).
    - Campos legados preservados crus em metadata (tipagem original do Signal).
    - validity UNKNOWN, calibration UNKNOWN, counts_as_vote False.
    - Nenhuma leitura de capability: `source="iceberg"` NÃO implica suporte
      (verificação cabe ao adapter/contexto futuro, nunca aqui).
    """
    if not isinstance(signal, Signal):
        raise ValueError("evidence_from_signal: esperado institutional.base.Signal")
    try:
        observed = int(signal.timestamp) if signal.timestamp is not None else None
    except (TypeError, ValueError, OverflowError):
        observed = None
    return Evidence(
        source=signal.source,
        observed_at_ms=observed,
        metadata={
            "legacy_signal_type": signal.signal_type,
            "legacy_direction": (signal.direction.value
                                 if isinstance(signal.direction, Enum)
                                 else str(signal.direction)),
            "legacy_strength": (signal.strength.value
                                if isinstance(signal.strength, Enum)
                                else str(signal.strength)),
            "legacy_confidence": signal.confidence,
            "legacy_price": signal.price,
            "legacy_description": signal.description,
        },
    )
