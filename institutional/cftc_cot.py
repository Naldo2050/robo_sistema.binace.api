# institutional/cftc_cot.py
# -*- coding: utf-8 -*-
"""
Intérprete oficial CFTC/CME COT — Traders in Financial Futures (Futures Only).
Fase P3 (Arquitetura Context-Only, fora do hot path).

NÃO confundir com `institutional/crypto_cot.py` (posicionamento Binance
intraday). Este módulo trata exclusivamente do relatório semanal oficial
da CFTC (snapshot de terça, publicado sexta 15:30 ET).

Terminologia oficial TFF (sem renomear, sem equivalências automáticas):
  dealer        = Dealer/Intermediary
  asset_manager = Asset Manager/Institutional
  leveraged     = Leveraged Funds
  other         = Other Reportables
  nonreportable = Nonreportable Positions

É PROIBIDO chamar automaticamente:
  leveraged de "smart money", nonreportable de "varejo",
  asset_manager de direção futura de preço.

Sem thresholds direcionais, sem sinais BUY/SELL, sem score.
Apenas normalização + qualidade + freshness calendar-aware + WoW bruto.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - fallback sem DST
    _ET = timezone.utc

logger = logging.getLogger("CftcCOT")

SCHEMA_VERSION = 1

STATUSES = ("AVAILABLE", "PARTIAL", "STALE", "UNAVAILABLE", "UNSUPPORTED", "INVALID")

# Estado exclusivo de replay sem proveniência de disponibilidade.
NO_POINT_IN_TIME = "UNAVAILABLE_FOR_POINT_IN_TIME"

# Feriados US que deslocam a publicação de sexta (mesma base do enricher).
_US_HOLIDAYS_2025_2026 = {
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25", "2026-06-19",
    "2026-07-04", "2026-09-07", "2026-11-26", "2026-12-25",
}

# Freshness calendar-aware (documentado; sem TTL cego):
# fresh  = último relatório esperado disponível (≤ 9 dias de referência:
#          terça→sexta 15:30 ET + margem de feriado/grace);
# stale_usable = exatamente 1 relatório perdido (≤ 16 dias); além = UNAVAILABLE.
_FRESH_REFERENCE_S = float(os.getenv("CFTC_FRESH_REFERENCE_S", "777600"))   # 9 dias
_STALE_REFERENCE_S = float(os.getenv("CFTC_STALE_REFERENCE_S", "1382400"))  # 16 dias

_CATEGORY_FIELDS = (
    ("dealer", "dealer_positions_long_all", "dealer_positions_short_all",
     "dealer_positions_spread_all"),
    ("asset_manager", "asset_mgr_positions_long", "asset_mgr_positions_short",
     "asset_mgr_positions_spread"),
    ("leveraged", "lev_money_positions_long", "lev_money_positions_short",
     "lev_money_positions_spread"),
    ("other", "other_rept_positions_long", "other_rept_positions_short",
     "other_rept_positions_spread"),
    ("nonreportable", "nonrept_positions_long_all", "nonrept_positions_short_all",
     None),
)


def _safe_int(val: Any) -> Optional[int]:
    if val is None or isinstance(val, bool):
        return None
    try:
        if isinstance(val, float):
            if not math.isfinite(val) or val != int(val):
                return None
            return int(val)
        s = str(val).strip().replace(",", "")
        if not s:
            return None
        f = float(s)
        if not math.isfinite(f):
            return None
        if f < 0:
            return None
        return int(f)
    except (ValueError, TypeError):
        return None


def _parse_asof(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        s = str(value).strip()
        if "T" in s:
            s = s.split("T")[0]
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _utc(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


@dataclass
class CftcCotSnapshot:
    """Snapshot normalizado CFTC COT (contrato P2, schema_version=1)."""
    schema_version: int = SCHEMA_VERSION
    source: str = "cftc_cot"
    source_kind: str = "cftc_socrata"
    report_family: str = "TFF"
    report_scope: str = "futures_only"
    symbol: str = ""
    market_and_exchange_name: Optional[str] = None
    cftc_contract_market_code: Optional[str] = None
    report_as_of_date: Optional[str] = None
    first_seen_at: Optional[str] = None
    retrieved_at: Optional[str] = None
    analyzed_at: Optional[str] = None
    age_reference_seconds: Optional[float] = None
    age_available_seconds: Optional[float] = None
    status: str = "UNAVAILABLE"
    is_available: bool = False
    is_stale: bool = False
    positions: Dict[str, Any] = field(default_factory=dict)
    open_interest: Dict[str, Any] = field(default_factory=dict)
    derived_metrics: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CftcCot:
    """Normalização/interpretação do relatório oficial (stateless)."""

    def unsupported(self, symbol: str, now_iso: Optional[str] = None) -> CftcCotSnapshot:
        now = now_iso or datetime.now(timezone.utc).isoformat()
        return CftcCotSnapshot(
            symbol=symbol,
            retrieved_at=now,
            analyzed_at=now,
            status="UNSUPPORTED",
            is_available=False,
            is_stale=False,
            quality={"missing_fields": [], "validation_errors": [],
                     "revision": 0, "is_revision": False,
                     "weeks_missing_in_window": None},
            provenance={},
            error_code="unsupported_symbol",
        )

    def invalid(self, symbol: str, reason: str,
                now_iso: Optional[str] = None) -> CftcCotSnapshot:
        now = now_iso or datetime.now(timezone.utc).isoformat()
        return CftcCotSnapshot(
            symbol=symbol,
            retrieved_at=now,
            analyzed_at=now,
            status="INVALID",
            is_available=False,
            is_stale=False,
            quality={"missing_fields": [], "validation_errors": [reason],
                     "revision": 0, "is_revision": False,
                     "weeks_missing_in_window": None},
            provenance={},
            error_code="schema_error",
        )

    def analyze(
        self,
        raw_row: Optional[Dict[str, Any]],
        symbol: str = "BTCUSDT",
        contract_code: Optional[str] = None,
        first_seen_at: Optional[str] = None,
        retrieved_at: Optional[str] = None,
        prev_row: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> CftcCotSnapshot:
        """Normaliza uma linha Socrata TFF em snapshot P2 (puro, sem I/O)."""
        now_dt = now or datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        ret_iso = retrieved_at or now_iso

        if raw_row is None:
            snap = CftcCotSnapshot(symbol=symbol, retrieved_at=ret_iso,
                                   analyzed_at=now_iso, status="UNAVAILABLE",
                                   error_code="fetch_error")
            snap.quality = {"missing_fields": [], "validation_errors": [],
                            "revision": 0, "is_revision": False,
                            "weeks_missing_in_window": None}
            return snap

        errors: List[str] = []
        missing: List[str] = []

        asof = _parse_asof(raw_row.get("report_date_as_yyyy_mm_dd"))
        if asof is None:
            return self.invalid(symbol, "report_as_of_date ausente/inválido", now_iso)
        if asof.weekday() != 1:
            errors.append(f"report_as_of_date {asof.isoformat()} não é terça-feira")

        code = str(raw_row.get("cftc_contract_market_code", "")).strip()
        if contract_code and code != contract_code:
            return self.invalid(symbol, "contract_code divergente da consulta", now_iso)

        oi = _safe_int(raw_row.get("open_interest_all"))
        if oi is None:
            missing.append("open_interest_all")

        positions: Dict[str, Any] = {}
        for cat, f_long, f_short, f_spread in _CATEGORY_FIELDS:
            long = _safe_int(raw_row.get(f_long))
            short = _safe_int(raw_row.get(f_short))
            spread = _safe_int(raw_row.get(f_spread)) if f_spread else None
            if long is None:
                missing.append(f_long)
            if short is None:
                missing.append(f_short)
            if f_spread and spread is None:
                missing.append(f_spread)
            net = (long - short) if (long is not None and short is not None) else None
            positions[cat] = {
                "long": long,
                "short": short,
                "spreading": spread,
                "net": net,
                "share_oi_long": (long / oi) if (long is not None and oi) else None,
                "share_oi_short": (short / oi) if (short is not None and oi) else None,
            }

        # WoW bruto, mesmo contrato, sem forward-fill.
        prev_oi = _safe_int((prev_row or {}).get("open_interest_all"))
        change_wow = (oi - prev_oi) if (oi is not None and prev_oi is not None) else None
        wow_net: Dict[str, Optional[int]] = {}
        for cat, f_long, f_short, _ in _CATEGORY_FIELDS:
            pl = _safe_int((prev_row or {}).get(f_long))
            ps = _safe_int((prev_row or {}).get(f_short))
            cl = positions[cat]["long"]
            cs = positions[cat]["short"]
            if None in (pl, ps, cl, cs):
                wow_net[cat] = None
            else:
                wow_net[cat] = (cl - cs) - (pl - ps)

        # Freshness calendar-aware sobre a DATA DE REFERÊNCIA.
        asof_utc = datetime(asof.year, asof.month, asof.day, tzinfo=timezone.utc)
        age_ref = (now_dt - asof_utc).total_seconds()
        fs_dt = _utc(first_seen_at)
        age_avail = (now_dt - fs_dt).total_seconds() if fs_dt else None

        if errors:
            snap = self.invalid(symbol, "; ".join(errors), now_iso)
            snap.market_and_exchange_name = raw_row.get("market_and_exchange_names")
            snap.cftc_contract_market_code = code or None
            snap.report_as_of_date = asof.isoformat()
            return snap

        essential_missing = [m for m in missing if m in (
            "open_interest_all", "dealer_positions_long_all",
            "asset_mgr_positions_long", "lev_money_positions_long",
            "other_rept_positions_long", "nonrept_positions_long_all")]
        partial = bool(missing)
        if oi is None:
            snap = CftcCotSnapshot(symbol=symbol, retrieved_at=ret_iso,
                                   analyzed_at=now_iso, status="INVALID",
                                   error_code="invalid_number")
            snap.report_as_of_date = asof.isoformat()
            snap.quality = {"missing_fields": missing, "validation_errors": [],
                            "revision": 0, "is_revision": False,
                            "weeks_missing_in_window": None}
            return snap

        if age_ref <= _FRESH_REFERENCE_S:
            status, stale = ("AVAILABLE", False) if not partial else ("PARTIAL", False)
        elif age_ref <= _STALE_REFERENCE_S:
            status, stale = "STALE", True
        else:
            status, stale = "UNAVAILABLE", False

        return CftcCotSnapshot(
            symbol=symbol,
            market_and_exchange_name=raw_row.get("market_and_exchange_names"),
            cftc_contract_market_code=code or None,
            report_as_of_date=asof.isoformat(),
            first_seen_at=first_seen_at,
            retrieved_at=ret_iso,
            analyzed_at=now_iso,
            age_reference_seconds=round(age_ref, 1),
            age_available_seconds=round(age_avail, 1) if age_avail is not None else None,
            status=status,
            is_available=status in ("AVAILABLE", "PARTIAL", "STALE"),
            is_stale=stale,
            positions=positions,
            open_interest={"total": oi, "change_wow": change_wow},
            derived_metrics={
                "wow_net_change": wow_net,
                "week_over_week_report": {
                    "prev_report_as_of_date": (
                        _parse_asof(
                            (prev_row or {}).get("report_date_as_yyyy_mm_dd")
                        ).isoformat()
                        if _parse_asof(
                            (prev_row or {}).get("report_date_as_yyyy_mm_dd")
                        ) is not None else None
                    ),
                },
            },
            quality={"missing_fields": sorted(set(missing)),
                     "validation_errors": [],
                     "revision": 0, "is_revision": False,
                     "weeks_missing_in_window": None},
            provenance={
                "dataset_id": "gpe5-46if",
                "source_row_id": str(raw_row.get("id", "")),
                "contract_units": raw_row.get("contract_units"),
                "cache_hit": False,
            },
            error_code=None if status in ("AVAILABLE", "PARTIAL", "STALE") else "stale_cache",
        )


# =====================================================================
# Replay point-in-time / anti-look-ahead (P5). Funções puras, sem I/O.
#
# Regra: em instante t, só é visível o relatório com disponibilidade
# comprovada em ou antes de t. `report_as_of_date` NUNCA é available_at.
# Prioridade: 1) first_seen_at real; 2) calendário oficial (sexta 15:30 ET
# + feriados) + grace documentado; nunca horário inventado silencioso.
# =====================================================================

def _asof_to_et(asof: date) -> datetime:
    return datetime(asof.year, asof.month, asof.day, tzinfo=timezone.utc)


def expected_publication_utc(asof: date) -> datetime:
    """Sexta 15:30 ET da semana do asof (terça), +1 dia por feriado US.

    Aproximação conservadora documentada (P1 §17-18): desloca +1 dia útil
    se a sexta cair em feriado da base 2025-2026; fora da base, sem
    deslocamento e o chamador deve exigir first_seen_at real.
    """
    # dias da terça (weekday 1) até sexta (weekday 4)
    days_to_friday = (4 - asof.weekday()) % 7
    friday = date(asof.year, asof.month, asof.day)
    from datetime import timedelta as _td
    friday = friday + _td(days=days_to_friday)
    pub_et = datetime(friday.year, friday.month, friday.day, 15, 30, tzinfo=_ET)
    if friday.isoformat() in _US_HOLIDAYS_2025_2026:
        pub_et = pub_et + _td(days=1)
    return pub_et.astimezone(timezone.utc)


def select_point_in_time(
    records: List[Dict[str, Any]],
    at: datetime,
    allow_calendar_fallback: bool = True,
    calendar_grace_s: float = 7200.0,
) -> Dict[str, Any]:
    """Seleciona o registro visível em `at` ou declara indisponibilidade.

    Cada record: {report_as_of_date: YYYY-MM-DD, first_seen_at?: ISO,
    ...}. Retorna {"record": dict|None, "reason": str}. Revisões do mesmo
    asof competem por first_seen_at: a revisão só vale após ser vista.
    """
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    best = None
    for rec in records:
        asof = _parse_asof(rec.get("report_as_of_date"))
        if asof is None or asof.weekday() != 1:
            continue
        fs = _utc(rec.get("first_seen_at"))
        if fs is not None:
            visible = fs <= at
        elif allow_calendar_fallback:
            visible = (expected_publication_utc(asof).timestamp()
                       + calendar_grace_s) <= at.timestamp()
        else:
            continue
        if not visible:
            continue
        # Revisão posterior não retroage: desempatar por first_seen_at.
        key = (asof.isoformat(), (fs or datetime.max.replace(tzinfo=timezone.utc)).isoformat())
        if best is None or key > best[0]:
            best = (key, rec)
    if best is None:
        has_unproven = any(rec.get("report_as_of_date") and not rec.get("first_seen_at")
                           for rec in records)
        if has_unproven and not allow_calendar_fallback:
            return {"record": None, "reason": NO_POINT_IN_TIME}
        return {"record": None, "reason": "no_report_visible_at_t"}
    return {"record": best[1], "reason": "ok"}
