# institutional/positioning_contract.py
# -*- coding: utf-8 -*-
"""
P2-C — Open Interest & Positioning Evidence Contract v1.

Contrato formal, tipado e auditável para dados de posicionamento de mercado e
derivativos (Binance USD-M Futures):
- Open Interest (contratos base e nocional USD)
- Long/Short Accounts Ratio (mercado geral / todas as contas)
- Long/Short Positions Ratio (top traders por nocional)
- Funding Rate (taxa canônica per período de 8h)

Garante:
- Status de integridade explícito: AVAILABLE | PARTIAL | STALE | MISSING | INVALID.
- Nomes, unidades e proveniência explícitos (sem ambiguidade contracts vs USD).
- Tratamento estrito de missing vs zero:
  * missing OI != 0
  * missing LSR != 0
  * missing funding != 0
  * funding 0.0 observado real é VALID_ZERO_OBSERVED.
- Proteção non-finite: NaN/±Inf coagidos para None com validity=INVALID.
- TTL e Freshness com base nos tempos existentes (cache 300s, max stale 900s).
- Adapters Evidence v1 com counts_as_vote=False, direction=UNKNOWN, calibration=NOT_APPLICABLE.
- Compatibilidade RFC 8259 para serialização.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from institutional.evidence import (
    Evidence,
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
    _json_safe,
)

POSITIONING_CONTRACT_VERSION = "1.0.0"

# Freshness e TTLs canônicos já adotados em fetchers/binance_positioning_fetcher.py
# 300 segundos (5 minutos) é o bucket de publicação nativo da Binance Futures
DEFAULT_CACHE_TTL_MS = 300_000
# 900 segundos (15 minutos = 3 barras de 5m perdidas) é o limiar de obsolescência
DEFAULT_MAX_STALE_MS = 900_000


class PositioningFieldValidity(str, Enum):
    """Validade granular por métrica observada."""
    VALID = "VALID"
    VALID_ZERO_OBSERVED = "VALID_ZERO_OBSERVED"
    STALE = "STALE"
    MISSING = "MISSING"
    INVALID = "INVALID"


class PositioningSnapshotStatus(str, Enum):
    """Status agregado da integridade do snapshot de posicionamento."""
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    STALE = "STALE"
    MISSING = "MISSING"
    INVALID = "INVALID"


def _finite_or_none(val: Any) -> Optional[float]:
    """Coage número para float estritamente finito. Rejeita bool, strings e non-finite."""
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class PositioningField:
    """Campo individual de métrica de posicionamento com unidade e rastreabilidade."""
    value: Optional[float]
    unit: str
    observed_at_ms: Optional[int] = None
    age_ms: Optional[int] = None
    validity: PositioningFieldValidity = PositioningFieldValidity.MISSING
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Dicionário serializável RFC 8259 com chaves determinísticas."""
        val = self.value
        if val is not None and not math.isfinite(val):
            val = None
        return {
            "value": val,
            "unit": self.unit,
            "observed_at_ms": self.observed_at_ms,
            "age_ms": self.age_ms,
            "validity": (
                self.validity.value
                if isinstance(self.validity, PositioningFieldValidity)
                else str(self.validity)
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PositioningSnapshot:
    """Snapshot completo de posicionamento e derivativos com garantia de integridade."""
    symbol: str
    open_interest: PositioningField
    open_interest_usd: PositioningField
    long_short_accounts_ratio: PositioningField
    long_short_positions_ratio: PositioningField
    funding_rate: PositioningField
    source: str = "binance_usdm"
    status: PositioningSnapshotStatus = PositioningSnapshotStatus.MISSING
    reasons: List[str] = field(default_factory=list)
    contract_version: str = POSITIONING_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialização RFC 8259 sem floats não finitos."""
        return {
            "contract_version": self.contract_version,
            "symbol": self.symbol,
            "source": self.source,
            "status": (
                self.status.value
                if isinstance(self.status, PositioningSnapshotStatus)
                else str(self.status)
            ),
            "reasons": sorted(self.reasons),
            "open_interest": self.open_interest.to_dict(),
            "open_interest_usd": self.open_interest_usd.to_dict(),
            "long_short_accounts_ratio": self.long_short_accounts_ratio.to_dict(),
            "long_short_positions_ratio": self.long_short_positions_ratio.to_dict(),
            "funding_rate": self.funding_rate.to_dict(),
        }


def _extract_observed_at_ms(
    data: Any,
    now_ms: int,
) -> tuple[Optional[int], Optional[int], bool]:
    """
    Extrai observed_at_ms, age_ms e flag de stale a partir de snapshot ou dict.
    Preserva granularidade em milissegundos se source_timestamp estiver disponível.
    """
    if data is None:
        return None, None, False

    source_ts: Optional[int] = None
    observed_ts: Optional[float] = None
    is_stale_flag: bool = False

    if isinstance(data, dict):
        source_ts = data.get("source_timestamp") or data.get("timestamp") or data.get("time")
        observed_ts = data.get("observed_at")
        is_stale_flag = bool(data.get("is_stale", False))
    elif hasattr(data, "source_timestamp"):
        source_ts = getattr(data, "source_timestamp", None)
        observed_ts = getattr(data, "observed_at", None)
        is_stale_flag = bool(getattr(data, "is_stale", False))

    obs_ms: Optional[int] = None
    if source_ts is not None:
        try:
            s_int = int(source_ts)
            # Se for em segundos epoch (10 dígitos), converte para ms
            if s_int < 100_000_000_000:
                s_int *= 1000
            obs_ms = s_int
        except (ValueError, TypeError):
            pass

    if obs_ms is None and observed_ts is not None:
        try:
            o_flt = float(observed_ts)
            if math.isfinite(o_flt):
                if o_flt < 100_000_000_000:
                    obs_ms = int(o_flt * 1000)
                else:
                    obs_ms = int(o_flt)
        except (ValueError, TypeError):
            pass

    age_ms: Optional[int] = None
    if obs_ms is not None:
        age_ms = max(0, now_ms - obs_ms)

    return obs_ms, age_ms, is_stale_flag


def build_positioning_snapshot(
    positioning_data: Optional[Any] = None,
    derivatives_data: Optional[Dict[str, Any]] = None,
    funding_data: Optional[Any] = None,
    now_ms: Optional[int] = None,
    symbol: str = "BTCUSDT",
    cache_ttl_ms: int = DEFAULT_CACHE_TTL_MS,
    max_stale_ms: int = DEFAULT_MAX_STALE_MS,
) -> PositioningSnapshot:
    """
    Constrói snapshot canônico e tipado a partir de fontes heterogêneas da Binance.

    Fontes aceitas:
    - positioning_data: BinancePositioningSnapshot ou dict (de context_collector/enricher)
    - derivatives_data: dict sob 'BTCUSDT' contendo open_interest, long_short_ratio, etc.
    - funding_data: valor float direto ou dict de funding rates.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    # 1. Determinação de disponibilidade de posicionamento
    is_available = True
    if positioning_data is not None:
        if isinstance(positioning_data, dict):
            if "is_available" in positioning_data and not positioning_data.get("is_available"):
                is_available = False
        elif hasattr(positioning_data, "is_available"):
            if not getattr(positioning_data, "is_available"):
                is_available = False

    obs_ms, age_ms, is_stale_flag = _extract_observed_at_ms(positioning_data, now_ms)

    # Se obs_ms não foi encontrado no positioning_data, tentar no derivatives_data
    if obs_ms is None and derivatives_data is not None and isinstance(derivatives_data, dict):
        obs_ms, age_ms, _ = _extract_observed_at_ms(derivatives_data, now_ms)

    is_stale_by_age = (age_ms is not None and age_ms > max_stale_ms)
    is_stale = is_stale_flag or is_stale_by_age

    # Helper de extração com fallback
    def _get_val(key_pos: str, key_deriv: Optional[str] = None) -> Any:
        v = None
        if is_available and positioning_data is not None:
            if isinstance(positioning_data, dict):
                v = positioning_data.get(key_pos)
            elif hasattr(positioning_data, key_pos):
                v = getattr(positioning_data, key_pos, None)
        if v is None and derivatives_data and isinstance(derivatives_data, dict):
            dk = key_deriv or key_pos
            v = derivatives_data.get(dk)
        return v

    snapshot_reasons: List[str] = []

    if not is_available:
        snapshot_reasons.append("POSITIONING_UNAVAILABLE")

    # ── 1. Open Interest (Contratos) ──────────────────────────────────────────
    raw_oi = _get_val("open_interest")
    oi_field: PositioningField
    if raw_oi is None:
        oi_field = PositioningField(
            value=None,
            unit="contracts",
            observed_at_ms=obs_ms,
            age_ms=age_ms,
            validity=PositioningFieldValidity.MISSING,
            reason="OI_MISSING",
        )
    else:
        v_oi = _finite_or_none(raw_oi)
        if v_oi is None:
            oi_field = PositioningField(
                value=None,
                unit="contracts",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.INVALID,
                reason="OI_NONFINITE_OR_MALFORMED",
            )
        elif v_oi <= 0.0:
            oi_field = PositioningField(
                value=v_oi,
                unit="contracts",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.INVALID,
                reason="NON_POSITIVE_OI",
            )
        elif is_stale:
            oi_field = PositioningField(
                value=round(v_oi, 4),
                unit="contracts",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.STALE,
                reason="STALE_DATA",
            )
        else:
            oi_field = PositioningField(
                value=round(v_oi, 4),
                unit="contracts",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.VALID,
            )

    # ── 2. Open Interest USD ───────────────────────────────────────────────────
    raw_oi_usd = _get_val("open_interest_usd")
    oi_usd_field: PositioningField
    if raw_oi_usd is None:
        oi_usd_field = PositioningField(
            value=None,
            unit="USD",
            observed_at_ms=obs_ms,
            age_ms=age_ms,
            validity=PositioningFieldValidity.MISSING,
            reason="OI_USD_MISSING",
        )
    else:
        v_oi_usd = _finite_or_none(raw_oi_usd)
        if v_oi_usd is None:
            oi_usd_field = PositioningField(
                value=None,
                unit="USD",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.INVALID,
                reason="OI_USD_NONFINITE_OR_MALFORMED",
            )
        elif v_oi_usd <= 0.0:
            oi_usd_field = PositioningField(
                value=v_oi_usd,
                unit="USD",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.INVALID,
                reason="NON_POSITIVE_OI_USD",
            )
        elif is_stale:
            oi_usd_field = PositioningField(
                value=round(v_oi_usd, 2),
                unit="USD",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.STALE,
                reason="STALE_DATA",
            )
        else:
            oi_usd_field = PositioningField(
                value=round(v_oi_usd, 2),
                unit="USD",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.VALID,
            )

    # ── 3. Long/Short Accounts Ratio (Global / Todas as Contas) ───────────────
    raw_ls_acc = _get_val("global_account_ratio", "long_short_ratio")
    ls_acc_field: PositioningField
    if raw_ls_acc is None:
        ls_acc_field = PositioningField(
            value=None,
            unit="ratio",
            observed_at_ms=obs_ms,
            age_ms=age_ms,
            validity=PositioningFieldValidity.MISSING,
            reason="LSR_ACCOUNTS_MISSING",
        )
    else:
        v_ls_acc = _finite_or_none(raw_ls_acc)
        if v_ls_acc is None:
            ls_acc_field = PositioningField(
                value=None,
                unit="ratio",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.INVALID,
                reason="LSR_ACCOUNTS_NONFINITE_OR_MALFORMED",
            )
        elif v_ls_acc <= 0.0:
            ls_acc_field = PositioningField(
                value=v_ls_acc,
                unit="ratio",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.INVALID,
                reason="NON_POSITIVE_RATIO",
            )
        elif is_stale:
            ls_acc_field = PositioningField(
                value=round(v_ls_acc, 4),
                unit="ratio",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.STALE,
                reason="STALE_DATA",
            )
        else:
            ls_acc_field = PositioningField(
                value=round(v_ls_acc, 4),
                unit="ratio",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.VALID,
            )

    # ── 4. Long/Short Positions Ratio (Top Traders) ───────────────────────────
    raw_ls_pos = _get_val("top_position_ratio", "top_positions_ratio")
    ls_pos_field: PositioningField
    if raw_ls_pos is None:
        ls_pos_field = PositioningField(
            value=None,
            unit="ratio",
            observed_at_ms=obs_ms,
            age_ms=age_ms,
            validity=PositioningFieldValidity.MISSING,
            reason="LSR_POSITIONS_MISSING",
        )
    else:
        v_ls_pos = _finite_or_none(raw_ls_pos)
        if v_ls_pos is None:
            ls_pos_field = PositioningField(
                value=None,
                unit="ratio",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.INVALID,
                reason="LSR_POSITIONS_NONFINITE_OR_MALFORMED",
            )
        elif v_ls_pos <= 0.0:
            ls_pos_field = PositioningField(
                value=v_ls_pos,
                unit="ratio",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.INVALID,
                reason="NON_POSITIVE_RATIO",
            )
        elif is_stale:
            ls_pos_field = PositioningField(
                value=round(v_ls_pos, 4),
                unit="ratio",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.STALE,
                reason="STALE_DATA",
            )
        else:
            ls_pos_field = PositioningField(
                value=round(v_ls_pos, 4),
                unit="ratio",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.VALID,
            )

    # ── 5. Funding Rate (Canônico por Período 8h) ─────────────────────────────
    # Precedência: funding_data explícito -> positioning_data -> derivatives_data
    raw_funding = None
    if funding_data is not None:
        if isinstance(funding_data, dict):
            raw_funding = funding_data.get(symbol) or funding_data.get("funding_rate") or funding_data.get("rate")
        else:
            raw_funding = funding_data
    if raw_funding is None and positioning_data is not None:
        if isinstance(positioning_data, dict):
            raw_funding = positioning_data.get("funding_rate")
        elif hasattr(positioning_data, "funding_rate"):
            raw_funding = getattr(positioning_data, "funding_rate", None)
    if raw_funding is None and derivatives_data and isinstance(derivatives_data, dict):
        if "funding_rate_percent" in derivatives_data:
            pct_val = _finite_or_none(derivatives_data.get("funding_rate_percent"))
            if pct_val is not None:
                raw_funding = pct_val / 100.0
            else:
                raw_funding = derivatives_data.get("funding_rate_percent")
        elif "funding_rate" in derivatives_data:
            raw_funding = derivatives_data.get("funding_rate")

    funding_field: PositioningField
    if raw_funding is None:
        funding_field = PositioningField(
            value=None,
            unit="rate_8h",
            observed_at_ms=obs_ms,
            age_ms=age_ms,
            validity=PositioningFieldValidity.MISSING,
            reason="FUNDING_MISSING",
        )
    else:
        v_fr = _finite_or_none(raw_funding)
        if v_fr is None:
            funding_field = PositioningField(
                value=None,
                unit="rate_8h",
                observed_at_ms=obs_ms,
                age_ms=age_ms,
                validity=PositioningFieldValidity.INVALID,
                reason="FUNDING_NONFINITE_OR_MALFORMED",
            )
        elif v_fr == 0.0:
            if is_stale:
                funding_field = PositioningField(
                    value=0.0,
                    unit="rate_8h",
                    observed_at_ms=obs_ms,
                    age_ms=age_ms,
                    validity=PositioningFieldValidity.STALE,
                    reason="STALE_DATA",
                )
            else:
                funding_field = PositioningField(
                    value=0.0,
                    unit="rate_8h",
                    observed_at_ms=obs_ms,
                    age_ms=age_ms,
                    validity=PositioningFieldValidity.VALID_ZERO_OBSERVED,
                    reason="VALID_ZERO_OBSERVED",
                )
        else:
            if is_stale:
                funding_field = PositioningField(
                    value=round(v_fr, 8),
                    unit="rate_8h",
                    observed_at_ms=obs_ms,
                    age_ms=age_ms,
                    validity=PositioningFieldValidity.STALE,
                    reason="STALE_DATA",
                )
            else:
                funding_field = PositioningField(
                    value=round(v_fr, 8),
                    unit="rate_8h",
                    observed_at_ms=obs_ms,
                    age_ms=age_ms,
                    validity=PositioningFieldValidity.VALID,
                )

    # ── Determinação do Status Global do Snapshot ─────────────────────────────
    all_fields = [oi_field, oi_usd_field, ls_acc_field, ls_pos_field, funding_field]
    valid_count = sum(
        1 for f in all_fields
        if f.validity in (PositioningFieldValidity.VALID, PositioningFieldValidity.VALID_ZERO_OBSERVED)
    )
    stale_count = sum(1 for f in all_fields if f.validity == PositioningFieldValidity.STALE)
    invalid_count = sum(1 for f in all_fields if f.validity == PositioningFieldValidity.INVALID)
    missing_count = sum(1 for f in all_fields if f.validity == PositioningFieldValidity.MISSING)

    status: PositioningSnapshotStatus

    if not is_available and valid_count == 0:
        status = PositioningSnapshotStatus.MISSING
    elif missing_count == len(all_fields):
        status = PositioningSnapshotStatus.MISSING
        snapshot_reasons.append("ALL_FIELDS_MISSING")
    elif invalid_count == len(all_fields) or (invalid_count > 0 and valid_count == 0 and stale_count == 0):
        status = PositioningSnapshotStatus.INVALID
        snapshot_reasons.append("ALL_PRESENT_FIELDS_INVALID")
    elif stale_count > 0 and valid_count == 0:
        status = PositioningSnapshotStatus.STALE
        snapshot_reasons.append("STALE_DATA")
    elif valid_count == len(all_fields):
        status = PositioningSnapshotStatus.AVAILABLE
    else:
        # Mistura de válidos com missing, stale ou invalid
        status = PositioningSnapshotStatus.PARTIAL
        if missing_count > 0:
            snapshot_reasons.append(f"{missing_count}_FIELDS_MISSING")
        if invalid_count > 0:
            snapshot_reasons.append(f"{invalid_count}_FIELDS_INVALID")
        if stale_count > 0:
            snapshot_reasons.append(f"{stale_count}_FIELDS_STALE")

    return PositioningSnapshot(
        symbol=symbol,
        open_interest=oi_field,
        open_interest_usd=oi_usd_field,
        long_short_accounts_ratio=ls_acc_field,
        long_short_positions_ratio=ls_pos_field,
        funding_rate=funding_field,
        source="binance_usdm",
        status=status,
        reasons=snapshot_reasons,
    )


def positioning_snapshot_to_evidence(
    snapshot: PositioningSnapshot,
) -> List[Evidence]:
    """
    Adapter que converte campos observados de um PositioningSnapshot em instâncias
    formais de Evidence v1 (institutional/evidence.py).

    Contrato estrito:
    - Family: DERIVATIVES
    - Type: SLOW_CONTEXT
    - counts_as_vote: False (estritamente não-votante)
    - direction: UNKNOWN (estritamente sem inferência de viés direcional ou regime)
    - calibration: NOT_APPLICABLE (métricas físicas observadas da exchange)
    """
    field_mapping = [
        ("derivatives.open_interest", snapshot.open_interest),
        ("derivatives.open_interest_usd", snapshot.open_interest_usd),
        ("derivatives.long_short_accounts_ratio", snapshot.long_short_accounts_ratio),
        ("derivatives.long_short_positions_ratio", snapshot.long_short_positions_ratio),
        ("derivatives.funding_rate", snapshot.funding_rate),
    ]

    evidences: List[Evidence] = []
    base_provenance = {
        "exchange": "binance",
        "market": "usdm_futures",
        "symbol": snapshot.symbol,
        "contract_version": snapshot.contract_version,
        "snapshot_status": snapshot.status.value,
    }

    for field_id, fld in field_mapping:
        ev_validity: EvidenceValidity
        ev_reason = fld.reason

        if fld.validity in (PositioningFieldValidity.VALID, PositioningFieldValidity.VALID_ZERO_OBSERVED):
            ev_validity = EvidenceValidity.VALID
            if fld.validity == PositioningFieldValidity.VALID_ZERO_OBSERVED:
                ev_reason = "VALID_ZERO_OBSERVED"
        elif fld.validity == PositioningFieldValidity.STALE:
            ev_validity = EvidenceValidity.STALE
        elif fld.validity == PositioningFieldValidity.INVALID:
            ev_validity = EvidenceValidity.INVALID
        else:
            ev_validity = EvidenceValidity.UNKNOWN

        ev_val: Optional[float] = None
        if ev_validity in (EvidenceValidity.VALID, EvidenceValidity.STALE) and fld.value is not None:
            ev_val = fld.value

        evidence = Evidence(
            source="binance_usdm",
            family=EvidenceFamily.DERIVATIVES,
            evidence_type=EvidenceType.SLOW_CONTEXT,
            direction=EvidenceDirection.UNKNOWN,
            value=ev_val,
            observed_at_ms=fld.observed_at_ms,
            horizon_ms=DEFAULT_CACHE_TTL_MS,
            provenance={
                **base_provenance,
                "field_id": field_id,
                "unit": fld.unit,
            },
            validity=ev_validity,
            reason=ev_reason,
            calibration=EvidenceCalibration.NOT_APPLICABLE,
            counts_as_vote=False,
            derived_from=(),
            metadata={
                "field_id": field_id,
                "unit": fld.unit,
                "age_ms": fld.age_ms,
                "raw_validity": (
                    fld.validity.value
                    if isinstance(fld.validity, PositioningFieldValidity)
                    else str(fld.validity)
                ),
            },
        )
        evidences.append(evidence)

    return evidences


def positioning_from_event(
    event: Dict[str, Any],
    now_ms: Optional[int] = None,
    symbol: str = "BTCUSDT",
) -> PositioningSnapshot:
    """Helper que extrai dados de positioning e derivativos de um event dict completo."""
    ia = event.get("institutional_analytics", {}) or {}
    pos_data = ia.get("positioning") or event.get("positioning")

    deriv_root = event.get("derivatives", {}) or {}
    btc_deriv = deriv_root.get(symbol, deriv_root) if isinstance(deriv_root, dict) else {}

    funding_data = event.get("funding")

    return build_positioning_snapshot(
        positioning_data=pos_data,
        derivatives_data=btc_deriv if isinstance(btc_deriv, dict) else None,
        funding_data=funding_data,
        now_ms=now_ms,
        symbol=symbol,
    )
