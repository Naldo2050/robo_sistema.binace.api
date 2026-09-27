# institutional/macro_calendar.py
# -*- coding: utf-8 -*-
"""
P2-E — Scheduled Macro Events Context Contract v1.

Representa eventos macroeconômicos AGENDADOS e previamente conhecidos
com event-time correto em epoch UTC, conversão precisa de timezone (America/New_York -> UTC),
tratamento de DST (EST vs EDT), proveniência, importância factual declarada pela fonte,
status temporal puro e adaptação para Evidence NON_VOTING.

Escopo inicial:
- FOMC rate decision / statement
- US CPI
- US Core CPI
- US NFP (Employment Situation)
- US PCE

Restrições estritas do contrato:
- NÃO criar trade gate.
- NÃO criar blackout windows ("não operar X minutos antes").
- NÃO criar threshold temporal ou danger zones.
- NÃO criar signal, score direcional ou confidence.
- NÃO integrar confluence como voto (counts_as_vote=False, direction=UNKNOWN).
- NÃO alterar regime, whale score ou risk manager.
"""

from __future__ import annotations

import json
import logging
import math
import time
import zoneinfo
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from institutional.evidence import (
    Evidence,
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
    _json_safe,
)

logger = logging.getLogger(__name__)

MACRO_CALENDAR_CONTRACT_VERSION = "1.0.0"

# Timezone de referência para anúncios macroeconômicos federais norte-americanos
NY_ZONE = zoneinfo.ZoneInfo("America/New_York")


class MacroEventType(str, Enum):
    """Tipos canônicos de eventos macroeconômicos agendados."""
    FOMC_RATE_DECISION = "FOMC_RATE_DECISION"
    FOMC_STATEMENT = "FOMC_STATEMENT"
    US_CPI = "US_CPI"
    US_CORE_CPI = "US_CORE_CPI"
    US_NFP = "US_NFP"
    US_PCE = "US_PCE"
    UNKNOWN = "UNKNOWN"


class MacroEventImportance(str, Enum):
    """Classificação factual de importância/impacto declarada pela fonte."""
    HIGH_IMPACT_TARGET_SET = "HIGH_IMPACT_TARGET_SET"  # Target set factual canônico (CPI/FOMC/NFP)
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class MacroEventPhase(str, Enum):
    """Fase temporal puramente factual em relação ao tempo de referência."""
    UPCOMING = "UPCOMING"  # scheduled_at_ms > reference_time_ms
    AT_EVENT = "AT_EVENT"  # scheduled_at_ms == reference_time_ms
    PAST = "PAST"          # scheduled_at_ms < reference_time_ms


class CalendarProviderStatus(str, Enum):
    """Status de disponibilidade do provedor ou cache do calendário."""
    AVAILABLE = "AVAILABLE"  # Calendário carregado com sucesso (pode conter 0 ou N eventos)
    PARTIAL = "PARTIAL"      # Dados parciais ou em degradação
    STALE = "STALE"          # Calendário desatualizado / cache expirado
    MISSING = "MISSING"      # Provedor indisponível / não configurado
    INVALID = "INVALID"      # Erro de parsing / payload malformado


def parse_ny_datetime_to_utc_ms(
    date_str: str,
    time_str: str = "08:30",
) -> Tuple[int, str]:
    """
    Converte data ("YYYY-MM-DD") e horário ("HH:MM" ou "HH:MM:SS") em America/New_York
    para timestamp epoch em milissegundos UTC e string formatada ISO-8601 UTC.

    Trata rigorosamente:
    - EST (UTC-5) no inverno (ex: janeiro, fevereiro, novembro, dezembro).
    - EDT (UTC-4) no verão (ex: abril a outubro).
    - Transições exatas de Daylight Saving Time (DST).

    Exemplo:
        08:30 America/New_York em EDT -> 12:30:00Z
        08:30 America/New_York em EST -> 13:30:00Z
    """
    date_clean = date_str.strip()
    time_clean = time_str.strip()

    # Normaliza formato HH:MM[:SS]
    time_parts = time_clean.split(":")
    if len(time_parts) not in (2, 3):
        raise ValueError(f"Formato de horário inválido: '{time_str}' (esperado HH:MM ou HH:MM:SS)")
    try:
        hour, minute = int(time_parts[0]), int(time_parts[1])
        second = int(time_parts[2]) if len(time_parts) == 3 else 0
    except ValueError as e:
        raise ValueError(f"Formato de horário inválido: '{time_str}' (esperado HH:MM ou HH:MM:SS)") from e

    date_parts = date_clean.split("-")
    if len(date_parts) != 3:
        raise ValueError(f"Formato de data inválido: '{date_str}' (esperado YYYY-MM-DD)")
    try:
        year, month, day = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
    except ValueError as e:
        raise ValueError(f"Formato de data inválido: '{date_str}' (esperado YYYY-MM-DD)") from e

    # Constrói datetime com timezone America/New_York
    local_dt = datetime(year, month, day, hour, minute, second, tzinfo=NY_ZONE)

    # Converte para UTC canônico
    utc_dt = local_dt.astimezone(timezone.utc)
    epoch_ms = int(utc_dt.timestamp() * 1000)
    iso_utc = utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return epoch_ms, iso_utc


def build_stable_macro_event_id(
    country: str,
    currency: str,
    event_type: Union[MacroEventType, str],
    target_date: str,
) -> str:
    """
    Gera event_id estável determinístico independente de revisões pontuais de horário.

    Composição canônica: {COUNTRY}_{CURRENCY}_{EVENT_TYPE}_{YYYY-MM-DD}
    Exemplo: US_USD_FOMC_RATE_DECISION_2026-03-18
    """
    c = str(country).upper().strip()
    curr = str(currency).upper().strip()
    ev_t = event_type.value if isinstance(event_type, MacroEventType) else str(event_type).upper().strip()
    d = str(target_date).strip()
    return f"{c}_{curr}_{ev_t}_{d}"


@dataclass(frozen=True)
class ScheduledMacroEvent:
    """
    Contrato imutável de evento macroeconômico agendado.
    """
    event_id: str
    country: str
    currency: str
    event_type: MacroEventType
    title: str
    scheduled_at_ms: int
    scheduled_at_utc: str
    importance: MacroEventImportance
    source: str
    source_event_id: Optional[str] = None
    published_schedule_at_ms: Optional[int] = None
    last_updated_at_ms: Optional[int] = None
    previous_scheduled_at_ms: Optional[int] = None
    status: str = "SCHEDULED"  # "SCHEDULED", "REVISED", "CANCELLED", etc.
    reason: Optional[str] = None
    contract_version: str = MACRO_CALENDAR_CONTRACT_VERSION

    def time_to_event_ms(self, reference_time_ms: int) -> int:
        """Tempo restante até o evento em ms. Positivo se futuro, zero se contemporâneo, negativo se passado."""
        return self.scheduled_at_ms - reference_time_ms

    def event_phase(self, reference_time_ms: int) -> MacroEventPhase:
        """Classificação factual de fase temporal sem thresholds heurísticos."""
        delta = self.time_to_event_ms(reference_time_ms)
        if delta > 0:
            return MacroEventPhase.UPCOMING
        elif delta == 0:
            return MacroEventPhase.AT_EVENT
        else:
            return MacroEventPhase.PAST

    def to_dict(self) -> Dict[str, Any]:
        """Serialização determinística RFC 8259."""
        return {
            "contract_version": self.contract_version,
            "event_id": self.event_id,
            "country": self.country,
            "currency": self.currency,
            "event_type": self.event_type.value if isinstance(self.event_type, MacroEventType) else str(self.event_type),
            "title": self.title,
            "scheduled_at_ms": self.scheduled_at_ms,
            "scheduled_at_utc": self.scheduled_at_utc,
            "importance": self.importance.value if isinstance(self.importance, MacroEventImportance) else str(self.importance),
            "source": self.source,
            "source_event_id": self.source_event_id,
            "published_schedule_at_ms": self.published_schedule_at_ms,
            "last_updated_at_ms": self.last_updated_at_ms,
            "previous_scheduled_at_ms": self.previous_scheduled_at_ms,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MacroCalendarSnapshot:
    """
    Snapshot pontual do calendário macroeconômico em um instante de referência.
    """
    reference_time_ms: int
    nearest_upcoming_event: Optional[ScheduledMacroEvent]
    upcoming_events: List[ScheduledMacroEvent]
    recent_events: List[ScheduledMacroEvent]
    provider_status: CalendarProviderStatus
    source: str
    last_refresh_ms: int
    age_ms: int
    validity: EvidenceValidity
    reason: Optional[str] = None
    contract_version: str = MACRO_CALENDAR_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialização determinística RFC 8259."""
        return {
            "contract_version": self.contract_version,
            "reference_time_ms": self.reference_time_ms,
            "provider_status": self.provider_status.value if isinstance(self.provider_status, CalendarProviderStatus) else str(self.provider_status),
            "source": self.source,
            "last_refresh_ms": self.last_refresh_ms,
            "age_ms": self.age_ms,
            "validity": self.validity.value if isinstance(self.validity, EvidenceValidity) else str(self.validity),
            "reason": self.reason,
            "nearest_upcoming_event": self.nearest_upcoming_event.to_dict() if self.nearest_upcoming_event else None,
            "upcoming_events_count": len(self.upcoming_events),
            "upcoming_events": [e.to_dict() for e in self.upcoming_events],
            "recent_events_count": len(self.recent_events),
            "recent_events": [e.to_dict() for e in self.recent_events],
        }


class BaseMacroCalendarProvider(ABC):
    """Interface abstrata pura para provedores de calendário macroeconômico."""

    @abstractmethod
    def get_snapshot(
        self,
        reference_time_ms: int,
        upcoming_horizon_ms: Optional[int] = None,
        recent_horizon_ms: Optional[int] = None,
    ) -> MacroCalendarSnapshot:
        """Retorna snapshot do calendário relativo a reference_time_ms."""
        pass


class LocalJsonMacroCalendarProvider(BaseMacroCalendarProvider):
    """
    Provedor auditável de calendário baseado em armazenamento JSON local ou lista em memória.

    Suporta:
    - Armazenamento determinístico.
    - Atualização com preservação de event_id e registro de previous_scheduled_at_ms (re-scheduling).
    - Distinção estrita entre AVAILABLE (com zero eventos) vs MISSING vs STALE.
    """

    def __init__(
        self,
        events: Optional[List[ScheduledMacroEvent]] = None,
        source_name: str = "local_audit_schedule",
        stale_threshold_ms: int = 7 * 86400 * 1000,  # 7 dias por default
    ):
        self.source_name = source_name
        self.stale_threshold_ms = stale_threshold_ms
        self._events_by_id: Dict[str, ScheduledMacroEvent] = {}
        self._last_refresh_ms: int = int(time.time() * 1000)
        self._is_available: bool = True
        self._is_stale_flag: bool = False

        if events:
            for ev in events:
                self.add_or_update_event(ev)

    def set_availability(self, is_available: bool) -> None:
        """Controla status de disponibilidade para testes e telemetria de falha."""
        self._is_available = is_available

    def set_stale(self, is_stale: bool) -> None:
        """Seta explicitamente o estado STALE."""
        self._is_stale_flag = is_stale

    def add_or_update_event(self, event: ScheduledMacroEvent) -> None:
        """
        Insere ou atualiza evento mantendo estabilidade de ID e registrando revisões de horário.
        """
        existing = self._events_by_id.get(event.event_id)
        if existing is not None:
            # Se o horário agendado mudou, anota previous_scheduled_at_ms e status REVISED
            if existing.scheduled_at_ms != event.scheduled_at_ms:
                updated = ScheduledMacroEvent(
                    event_id=event.event_id,
                    country=event.country,
                    currency=event.currency,
                    event_type=event.event_type,
                    title=event.title,
                    scheduled_at_ms=event.scheduled_at_ms,
                    scheduled_at_utc=event.scheduled_at_utc,
                    importance=event.importance,
                    source=event.source,
                    source_event_id=event.source_event_id or existing.source_event_id,
                    published_schedule_at_ms=event.published_schedule_at_ms or existing.published_schedule_at_ms,
                    last_updated_at_ms=int(time.time() * 1000),
                    previous_scheduled_at_ms=existing.scheduled_at_ms,
                    status="REVISED",
                    reason=f"RESCHEDULED_FROM_{existing.scheduled_at_utc}",
                    contract_version=event.contract_version,
                )
                self._events_by_id[event.event_id] = updated
                self._last_refresh_ms = int(time.time() * 1000)
                return

        self._events_by_id[event.event_id] = event
        self._last_refresh_ms = int(time.time() * 1000)

    def load_from_dict_list(self, raw_items: List[Dict[str, Any]]) -> int:
        """Carrega lista de dicionários brutos e popula o repositório."""
        loaded = 0
        for item in raw_items:
            try:
                c = item.get("country", "US")
                curr = item.get("currency", "USD")
                ev_type_str = item.get("event_type", "UNKNOWN")
                try:
                    ev_type = MacroEventType(ev_type_str)
                except ValueError:
                    ev_type = MacroEventType.UNKNOWN

                target_date = item.get("target_date") or item.get("date")
                event_id = item.get("event_id") or build_stable_macro_event_id(c, curr, ev_type, target_date)

                # Resolução de timestamp
                if "scheduled_at_ms" in item:
                    sched_ms = int(item["scheduled_at_ms"])
                    sched_utc = item.get("scheduled_at_utc") or datetime.fromtimestamp(sched_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                elif target_date and "time_ny" in item:
                    sched_ms, sched_utc = parse_ny_datetime_to_utc_ms(target_date, item["time_ny"])
                else:
                    logger.warning(f"Item macro descartado por ausência de horário: {item}")
                    continue

                imp_str = item.get("importance", "HIGH")
                try:
                    imp = MacroEventImportance(imp_str)
                except ValueError:
                    imp = MacroEventImportance.UNKNOWN

                ev = ScheduledMacroEvent(
                    event_id=event_id,
                    country=c,
                    currency=curr,
                    event_type=ev_type,
                    title=item.get("title", ev_type.value),
                    scheduled_at_ms=sched_ms,
                    scheduled_at_utc=sched_utc,
                    importance=imp,
                    source=item.get("source", self.source_name),
                    source_event_id=item.get("source_event_id"),
                    published_schedule_at_ms=item.get("published_schedule_at_ms"),
                    last_updated_at_ms=item.get("last_updated_at_ms"),
                    previous_scheduled_at_ms=item.get("previous_scheduled_at_ms"),
                    status=item.get("status", "SCHEDULED"),
                    reason=item.get("reason"),
                )
                self.add_or_update_event(ev)
                loaded += 1
            except Exception as e:
                logger.warning(f"Falha ao carregar item de calendário: {e}")

        return loaded

    def get_snapshot(
        self,
        reference_time_ms: int,
        upcoming_horizon_ms: Optional[int] = None,
        recent_horizon_ms: Optional[int] = None,
    ) -> MacroCalendarSnapshot:
        """
        Computa snapshot contemporâneo.
        """
        age_ms = max(0, reference_time_ms - self._last_refresh_ms)

        # 1. Checagem de disponibilidade
        if not self._is_available:
            return MacroCalendarSnapshot(
                reference_time_ms=reference_time_ms,
                nearest_upcoming_event=None,
                upcoming_events=[],
                recent_events=[],
                provider_status=CalendarProviderStatus.MISSING,
                source=self.source_name,
                last_refresh_ms=self._last_refresh_ms,
                age_ms=age_ms,
                validity=EvidenceValidity.UNKNOWN,
                reason="CALENDAR_PROVIDER_UNAVAILABLE",
            )

        # 2. Checagem de staleness
        is_stale = self._is_stale_flag or (age_ms > self.stale_threshold_ms)
        provider_status = CalendarProviderStatus.STALE if is_stale else CalendarProviderStatus.AVAILABLE
        validity = EvidenceValidity.PARTIAL if is_stale else EvidenceValidity.VALID
        reason = "CALENDAR_DATA_STALE" if is_stale else None

        # 3. Partição temporal ordenada
        all_events = sorted(self._events_by_id.values(), key=lambda e: e.scheduled_at_ms)

        upcoming: List[ScheduledMacroEvent] = []
        recent: List[ScheduledMacroEvent] = []

        for e in all_events:
            dt = e.time_to_event_ms(reference_time_ms)
            if dt > 0:
                if upcoming_horizon_ms is None or dt <= upcoming_horizon_ms:
                    upcoming.append(e)
            else:
                # Eventos passados recentes (dt <= 0, dt >= -recent_horizon_ms)
                if recent_horizon_ms is None or abs(dt) <= recent_horizon_ms:
                    recent.append(e)

        nearest = upcoming[0] if upcoming else None

        if not upcoming and not is_stale:
            reason = "NO_SCHEDULED_EVENTS_IN_HORIZON"

        return MacroCalendarSnapshot(
            reference_time_ms=reference_time_ms,
            nearest_upcoming_event=nearest,
            upcoming_events=upcoming,
            recent_events=recent,
            provider_status=provider_status,
            source=self.source_name,
            last_refresh_ms=self._last_refresh_ms,
            age_ms=age_ms,
            validity=validity,
            reason=reason,
        )


def macro_calendar_to_evidence(
    snapshot: MacroCalendarSnapshot,
) -> List[Evidence]:
    """
    Adapter canônico para Evidence v1 (P2-E1).

    Taxonomy fields:
    - macro.scheduled_event.time_to_event
    - macro.scheduled_event.importance
    - macro.scheduled_event.type

    Propriedades estritas de conformidade:
    - Family: MACRO
    - Type: SLOW_CONTEXT
    - counts_as_vote: False (estritamente não-votante)
    - direction: UNKNOWN (estritamente sem inferência de sinal)
    - calibration: NOT_APPLICABLE
    """
    observed_at = snapshot.reference_time_ms
    base_provenance = {
        "source": snapshot.source,
        "provider_status": snapshot.provider_status.value if isinstance(snapshot.provider_status, CalendarProviderStatus) else str(snapshot.provider_status),
        "last_refresh_ms": snapshot.last_refresh_ms,
        "age_ms": snapshot.age_ms,
        "contract_version": snapshot.contract_version,
    }

    nearest = snapshot.nearest_upcoming_event

    # 1. Campo: time_to_event (ms)
    time_to_event_val: Optional[float] = None
    if snapshot.validity in (EvidenceValidity.VALID, EvidenceValidity.PARTIAL) and nearest is not None:
        time_to_event_val = float(nearest.time_to_event_ms(snapshot.reference_time_ms))

    ev_time = Evidence(
        source=snapshot.source,
        family=EvidenceFamily.MACRO,
        evidence_type=EvidenceType.SLOW_CONTEXT,
        direction=EvidenceDirection.UNKNOWN,
        value=time_to_event_val,
        observed_at_ms=observed_at,
        horizon_ms=int(time_to_event_val) if time_to_event_val is not None and time_to_event_val > 0 else 0,
        provenance={**base_provenance, "field_id": "macro.scheduled_event.time_to_event", "unit": "ms"},
        validity=snapshot.validity,
        reason=snapshot.reason,
        calibration=EvidenceCalibration.NOT_APPLICABLE,
        counts_as_vote=False,
        derived_from=(),
        metadata={
            "field_id": "macro.scheduled_event.time_to_event",
            "unit": "ms",
            "nearest_event_id": nearest.event_id if nearest else None,
            "scheduled_at_utc": nearest.scheduled_at_utc if nearest else None,
        },
    )

    # 2. Campo: importance
    importance_val: Optional[float] = None
    if snapshot.validity in (EvidenceValidity.VALID, EvidenceValidity.PARTIAL) and nearest is not None:
        # Factual: HIGH_IMPACT_TARGET_SET=3.0, HIGH=3.0, MEDIUM=2.0, LOW=1.0, UNKNOWN=0.0
        imp_map = {
            MacroEventImportance.HIGH_IMPACT_TARGET_SET: 3.0,
            MacroEventImportance.HIGH: 3.0,
            MacroEventImportance.MEDIUM: 2.0,
            MacroEventImportance.LOW: 1.0,
        }
        importance_val = imp_map.get(nearest.importance, 0.0)

    ev_importance = Evidence(
        source=snapshot.source,
        family=EvidenceFamily.MACRO,
        evidence_type=EvidenceType.SLOW_CONTEXT,
        direction=EvidenceDirection.UNKNOWN,
        value=importance_val,
        observed_at_ms=observed_at,
        horizon_ms=0,
        provenance={**base_provenance, "field_id": "macro.scheduled_event.importance", "unit": "category"},
        validity=snapshot.validity,
        reason=snapshot.reason,
        calibration=EvidenceCalibration.NOT_APPLICABLE,
        counts_as_vote=False,
        derived_from=(),
        metadata={
            "field_id": "macro.scheduled_event.importance",
            "unit": "category",
            "importance_label": nearest.importance.value if nearest else None,
            "nearest_event_id": nearest.event_id if nearest else None,
        },
    )

    # 3. Campo: type
    ev_type = Evidence(
        source=snapshot.source,
        family=EvidenceFamily.MACRO,
        evidence_type=EvidenceType.SLOW_CONTEXT,
        direction=EvidenceDirection.UNKNOWN,
        value=None,  # Categórico (não-numérico)
        observed_at_ms=observed_at,
        horizon_ms=0,
        provenance={**base_provenance, "field_id": "macro.scheduled_event.type", "unit": "type"},
        validity=snapshot.validity,
        reason=snapshot.reason,
        calibration=EvidenceCalibration.NOT_APPLICABLE,
        counts_as_vote=False,
        derived_from=(),
        metadata={
            "field_id": "macro.scheduled_event.type",
            "unit": "type",
            "event_type": nearest.event_type.value if nearest else None,
            "title": nearest.title if nearest else None,
            "nearest_event_id": nearest.event_id if nearest else None,
        },
    )

    return [ev_time, ev_importance, ev_type]
