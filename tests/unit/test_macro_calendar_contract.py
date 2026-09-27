# tests/unit/test_macro_calendar_contract.py
# -*- coding: utf-8 -*-
"""
Suíte de testes para P2-E — Scheduled Macro Events Context Contract v1.

Cobre:
- Event types canônicos
- IDs estáveis determinísticos
- Revisão de schedule (mantém event_id, anota previous_scheduled_at_ms)
- Conversão EST vs EDT vs transição DST (America/New_York -> UTC)
- Epoch UTC canônico
- Fases temporais factuais (UPCOMING, AT_EVENT, PAST) e time_to_event_ms
- Identificação do nearest upcoming event
- Provider status: AVAILABLE com lista vazia != MISSING
- Detecção de STALE calendar
- Tratamento de tempo malformado
- Deduplicação e re-scheduling sem eventos duplicados
- Evidence v1 adapter: counts_as_vote=False, direction=UNKNOWN, calibration=NOT_APPLICABLE
- Confluence Shadow ignora macro evidências em contagem de votos
- Serialização determinística RFC 8259
- Ausência estrita de trade gates, blackout windows, danger zones ou thresholds heurísticos
- Independência de fim de semana (weekend vs macro calendar)
"""

import json
from datetime import datetime, timezone
import pytest

from institutional.evidence import (
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
)
from institutional.confluence_shadow import (
    ReconcilerStatus,
    reconcile as confluence_reconcile,
)
from institutional.macro_calendar import (
    MACRO_CALENDAR_CONTRACT_VERSION,
    BaseMacroCalendarProvider,
    CalendarProviderStatus,
    LocalJsonMacroCalendarProvider,
    MacroEventImportance,
    MacroEventPhase,
    MacroEventType,
    ScheduledMacroEvent,
    MacroCalendarSnapshot,
    build_stable_macro_event_id,
    macro_calendar_to_evidence,
    parse_ny_datetime_to_utc_ms,
)


# ==============================================================================
# 1. EVENT TYPES & IMPORTANCE ENUMS
# ==============================================================================

def test_canonical_event_types():
    """Valida que todos os tipos canônicos exigidos pelo escopo P2-E estão presentes."""
    assert MacroEventType.FOMC_RATE_DECISION == "FOMC_RATE_DECISION"
    assert MacroEventType.FOMC_STATEMENT == "FOMC_STATEMENT"
    assert MacroEventType.US_CPI == "US_CPI"
    assert MacroEventType.US_CORE_CPI == "US_CORE_CPI"
    assert MacroEventType.US_NFP == "US_NFP"
    assert MacroEventType.US_PCE == "US_PCE"
    assert MacroEventType.UNKNOWN == "UNKNOWN"


def test_canonical_importance_enums():
    """Valida classificações factuais de importância sem inferência de impacto histórico."""
    assert MacroEventImportance.HIGH_IMPACT_TARGET_SET == "HIGH_IMPACT_TARGET_SET"
    assert MacroEventImportance.HIGH == "HIGH"
    assert MacroEventImportance.MEDIUM == "MEDIUM"
    assert MacroEventImportance.LOW == "LOW"
    assert MacroEventImportance.UNKNOWN == "UNKNOWN"


# ==============================================================================
# 2. STABLE EVENT IDS
# ==============================================================================

def test_stable_event_id_generation():
    """Gera ID estável por {country}_{currency}_{event_type}_{target_date}."""
    event_id = build_stable_macro_event_id(
        country="US",
        currency="USD",
        event_type=MacroEventType.FOMC_RATE_DECISION,
        target_date="2026-03-18",
    )
    assert event_id == "US_USD_FOMC_RATE_DECISION_2026-03-18"

    # Não deve variar se passado como string
    event_id_str = build_stable_macro_event_id(
        country="us ",
        currency=" usd",
        event_type="fomc_rate_decision",
        target_date="2026-03-18",
    )
    assert event_id_str == "US_USD_FOMC_RATE_DECISION_2026-03-18"


# ==============================================================================
# 3. TIMEZONE & DST CONVERSIONS (EST vs EDT)
# ==============================================================================

def test_est_winter_conversion():
    """
    Em janeiro (EST = UTC-5):
    08:30 America/New_York -> 13:30:00Z.
    """
    epoch_ms, iso_utc = parse_ny_datetime_to_utc_ms("2026-01-14", "08:30")
    assert iso_utc == "2026-01-14T13:30:00Z"
    
    # 13:30 UTC em epoch ms
    dt_expected = datetime(2026, 1, 14, 13, 30, 0, tzinfo=timezone.utc)
    expected_ms = int(dt_expected.timestamp() * 1000)
    assert epoch_ms == expected_ms


def test_edt_summer_conversion():
    """
    Em junho (EDT = UTC-4):
    08:30 America/New_York -> 12:30:00Z.
    """
    epoch_ms, iso_utc = parse_ny_datetime_to_utc_ms("2026-06-10", "08:30")
    assert iso_utc == "2026-06-10T12:30:00Z"
    
    dt_expected = datetime(2026, 6, 10, 12, 30, 0, tzinfo=timezone.utc)
    expected_ms = int(dt_expected.timestamp() * 1000)
    assert epoch_ms == expected_ms


def test_dst_transition_boundary():
    """
    Em 2026, DST nos EUA inicia no domingo 8 de março de 2026 (às 02:00 local).
    Sexta-feira 6 de março de 2026: EST (UTC-5) -> 08:30 local = 13:30Z.
    Terça-feira 10 de março de 2026: EDT (UTC-4) -> 08:30 local = 12:30Z.
    """
    _, utc_before = parse_ny_datetime_to_utc_ms("2026-03-06", "08:30")
    assert utc_before == "2026-03-06T13:30:00Z"

    _, utc_after = parse_ny_datetime_to_utc_ms("2026-03-10", "08:30")
    assert utc_after == "2026-03-10T12:30:00Z"


def test_fomc_afternoon_time_conversion():
    """FOMC rate decision às 14:00 America/New_York em EDT (julho) -> 18:00:00Z."""
    epoch_ms, iso_utc = parse_ny_datetime_to_utc_ms("2026-07-29", "14:00")
    assert iso_utc == "2026-07-29T18:00:00Z"


def test_malformed_time_raises_error():
    """Horários malformados devem levantar ValueError claro."""
    with pytest.raises(ValueError, match="Formato de horário inválido"):
        parse_ny_datetime_to_utc_ms("2026-01-14", "invalid_time")

    with pytest.raises(ValueError, match="Formato de data inválido"):
        parse_ny_datetime_to_utc_ms("2026/01/14", "08:30")


# ==============================================================================
# 4. TEMPORAL PHASES & TIME TO EVENT
# ==============================================================================

def test_temporal_phases_pure_facts():
    """Verifica fases UPCOMING (>0), AT_EVENT (==0), PAST (<0) sem heurísticas."""
    sched_ms = 1_700_000_000_000
    ev = ScheduledMacroEvent(
        event_id="US_USD_US_CPI_2026-01-14",
        country="US",
        currency="USD",
        event_type=MacroEventType.US_CPI,
        title="Consumer Price Index",
        scheduled_at_ms=sched_ms,
        scheduled_at_utc="2023-11-14T13:30:00Z",
        importance=MacroEventImportance.HIGH_IMPACT_TARGET_SET,
        source="test_source",
    )

    # 10 segundos antes do evento
    t_before = sched_ms - 10_000
    assert ev.time_to_event_ms(t_before) == 10_000
    assert ev.event_phase(t_before) == MacroEventPhase.UPCOMING

    # Exato momento do evento
    assert ev.time_to_event_ms(sched_ms) == 0
    assert ev.event_phase(sched_ms) == MacroEventPhase.AT_EVENT

    # 10 segundos após o evento
    t_after = sched_ms + 10_000
    assert ev.time_to_event_ms(t_after) == -10_000
    assert ev.event_phase(t_after) == MacroEventPhase.PAST


# ==============================================================================
# 5. REVISIONS / SCHEDULE CHANGES (RE-SCHEDULING)
# ==============================================================================

def test_schedule_revision_preserves_id_and_tracks_previous_time():
    """
    Ao mudar o horário de um evento existente:
    - O event_id é estritamente preservado
    - previous_scheduled_at_ms registra o horário anterior
    - status torna-se REVISED
    - Não duplica eventos no repositório.
    """
    provider = LocalJsonMacroCalendarProvider(source_name="fed_calendar")

    ev_original = ScheduledMacroEvent(
        event_id="US_USD_FOMC_RATE_DECISION_2026-03-18",
        country="US",
        currency="USD",
        event_type=MacroEventType.FOMC_RATE_DECISION,
        title="FOMC Rate Decision",
        scheduled_at_ms=1_773_856_800_000,
        scheduled_at_utc="2026-03-18T18:00:00Z",
        importance=MacroEventImportance.HIGH_IMPACT_TARGET_SET,
        source="fed_calendar",
    )
    provider.add_or_update_event(ev_original)

    # Novo anúncio revisa horário para 18:30Z (+30min)
    ev_revised = ScheduledMacroEvent(
        event_id="US_USD_FOMC_RATE_DECISION_2026-03-18",
        country="US",
        currency="USD",
        event_type=MacroEventType.FOMC_RATE_DECISION,
        title="FOMC Rate Decision",
        scheduled_at_ms=1_773_858_600_000,
        scheduled_at_utc="2026-03-18T18:30:00Z",
        importance=MacroEventImportance.HIGH_IMPACT_TARGET_SET,
        source="fed_calendar",
    )
    provider.add_or_update_event(ev_revised)

    # Verifica que não duplicou
    snapshot = provider.get_snapshot(reference_time_ms=1_773_850_000_000)
    assert len(snapshot.upcoming_events) == 1

    stored = snapshot.upcoming_events[0]
    assert stored.event_id == "US_USD_FOMC_RATE_DECISION_2026-03-18"
    assert stored.scheduled_at_ms == 1_773_858_600_000
    assert stored.scheduled_at_utc == "2026-03-18T18:30:00Z"
    assert stored.previous_scheduled_at_ms == 1_773_856_800_000
    assert stored.status == "REVISED"
    assert "RESCHEDULED_FROM" in (stored.reason or "")


# ==============================================================================
# 6. PROVIDER HEALTH: AVAILABLE (EMPTY) != MISSING != STALE
# ==============================================================================

def test_available_empty_vs_missing_vs_stale():
    """
    Distinção crucial:
    - AVAILABLE com 0 eventos = VALID, provider_status=AVAILABLE, reason=NO_SCHEDULED_EVENTS_IN_HORIZON
    - MISSING = UNKNOWN, provider_status=MISSING, reason=CALENDAR_PROVIDER_UNAVAILABLE
    - STALE = PARTIAL, provider_status=STALE, reason=CALENDAR_DATA_STALE
    """
    now_ms = 1_700_000_000_000
    provider = LocalJsonMacroCalendarProvider(source_name="bls_calendar")

    # 1. AVAILABLE com 0 eventos
    snap_empty = provider.get_snapshot(reference_time_ms=now_ms)
    assert snap_empty.provider_status == CalendarProviderStatus.AVAILABLE
    assert snap_empty.validity == EvidenceValidity.VALID
    assert snap_empty.reason == "NO_SCHEDULED_EVENTS_IN_HORIZON"
    assert snap_empty.nearest_upcoming_event is None
    assert len(snap_empty.upcoming_events) == 0

    # 2. MISSING (provedor indisponível)
    provider.set_availability(False)
    snap_missing = provider.get_snapshot(reference_time_ms=now_ms)
    assert snap_missing.provider_status == CalendarProviderStatus.MISSING
    assert snap_missing.validity == EvidenceValidity.UNKNOWN
    assert snap_missing.reason == "CALENDAR_PROVIDER_UNAVAILABLE"
    assert snap_missing.nearest_upcoming_event is None

    # 3. STALE (marcado ou tempo expirado)
    provider.set_availability(True)
    provider.set_stale(True)
    snap_stale = provider.get_snapshot(reference_time_ms=now_ms)
    assert snap_stale.provider_status == CalendarProviderStatus.STALE
    assert snap_stale.validity == EvidenceValidity.PARTIAL
    assert snap_stale.reason == "CALENDAR_DATA_STALE"


# ==============================================================================
# 7. NEAREST UPCOMING EVENT & HORIZON FILTERING
# ==============================================================================

def test_nearest_upcoming_and_ordering():
    """Valida ordenação cronológica e seleção precisa do nearest upcoming event."""
    provider = LocalJsonMacroCalendarProvider(source_name="us_calendar")

    # Evento A: daqui a 2 horas
    # Evento B: daqui a 1 hora
    # Evento C: daqui a 5 horas
    now_ms = 1_700_000_000_000
    t_ev_a = now_ms + 2 * 3600 * 1000
    t_ev_b = now_ms + 1 * 3600 * 1000
    t_ev_c = now_ms + 5 * 3600 * 1000

    provider.load_from_dict_list([
        {"event_type": "US_CPI", "date": "2023-11-14", "scheduled_at_ms": t_ev_a, "title": "CPI"},
        {"event_type": "US_NFP", "date": "2023-11-14", "scheduled_at_ms": t_ev_b, "title": "NFP"},
        {"event_type": "FOMC_RATE_DECISION", "date": "2026-11-14", "scheduled_at_ms": t_ev_c, "title": "FOMC"},
    ])

    snapshot = provider.get_snapshot(reference_time_ms=now_ms)
    assert snapshot.nearest_upcoming_event is not None
    assert snapshot.nearest_upcoming_event.event_type == MacroEventType.US_NFP
    assert snapshot.nearest_upcoming_event.scheduled_at_ms == t_ev_b

    # Filtro de horizonte: horizonte de 3 horas só deve retornar B e A, excluindo C
    snap_horizon = provider.get_snapshot(reference_time_ms=now_ms, upcoming_horizon_ms=3 * 3600 * 1000)
    assert len(snap_horizon.upcoming_events) == 2
    assert snap_horizon.upcoming_events[0].event_type == MacroEventType.US_NFP
    assert snap_horizon.upcoming_events[1].event_type == MacroEventType.US_CPI


# ==============================================================================
# 8. EVIDENCE V1 ADAPTER CONFORMANCE
# ==============================================================================

def test_macro_calendar_to_evidence_compliance():
    """
    Garante conformidade estrita com Evidence v1:
    - Family: MACRO
    - Type: SLOW_CONTEXT
    - counts_as_vote: False (NÃO-VOTANTE)
    - direction: UNKNOWN (SEM VIÉS DIRECIONAL)
    - calibration: NOT_APPLICABLE
    - IDs canônicos registrados na taxonomia.
    """
    provider = LocalJsonMacroCalendarProvider(source_name="official_test_feed")
    now_ms = 1_700_000_000_000
    sched_ms = now_ms + 1_800_000  # 30 minutos à frente

    provider.load_from_dict_list([
        {
            "event_type": "US_CPI",
            "date": "2023-11-14",
            "scheduled_at_ms": sched_ms,
            "title": "US Consumer Price Index",
            "importance": "HIGH_IMPACT_TARGET_SET",
        }
    ])

    snapshot = provider.get_snapshot(reference_time_ms=now_ms)
    evidences = macro_calendar_to_evidence(snapshot)

    assert len(evidences) == 3
    ev_time, ev_importance, ev_type = evidences

    # Verificações universais obrigatórias
    for ev in evidences:
        assert ev.family == EvidenceFamily.MACRO
        assert ev.evidence_type == EvidenceType.SLOW_CONTEXT
        assert ev.direction == EvidenceDirection.UNKNOWN
        assert ev.counts_as_vote is False
        assert ev.calibration == EvidenceCalibration.NOT_APPLICABLE
        assert ev.validity == EvidenceValidity.VALID

    # Campo 1: time_to_event
    assert ev_time.metadata["field_id"] == "macro.scheduled_event.time_to_event"
    assert ev_time.value == 1_800_000.0
    assert ev_time.provenance["unit"] == "ms"

    # Campo 2: importance
    assert ev_importance.metadata["field_id"] == "macro.scheduled_event.importance"
    assert ev_importance.value == 3.0  # HIGH_IMPACT_TARGET_SET mapeado para 3.0 factual
    assert ev_importance.metadata["importance_label"] == "HIGH_IMPACT_TARGET_SET"

    # Campo 3: type
    assert ev_type.metadata["field_id"] == "macro.scheduled_event.type"
    assert ev_type.value is None  # Categórico puro
    assert ev_type.metadata["event_type"] == "US_CPI"


# ==============================================================================
# 9. CONFLUENCE SHADOW NEVER VOTES ON MACRO EVIDENCE
# ==============================================================================

def test_confluence_shadow_ignores_macro_voting():
    """
    Valida que o Confluence Shadow v1 (P2-A) ignora rigorosamente as evidências macro:
    - Não entram como valid_directional nem participam de votação
    - active_directional_evidences permanece vazio
    - status permanece INSUFFICIENT_EVIDENCES
    - directions_present permanece vazio.
    """
    provider = LocalJsonMacroCalendarProvider()
    now_ms = 1_700_000_000_000
    provider.load_from_dict_list([
        {
            "event_type": "FOMC_RATE_DECISION",
            "date": "2023-11-14",
            "scheduled_at_ms": now_ms + 60_000,
            "title": "FOMC",
            "importance": "HIGH",
        }
    ])
    snapshot = provider.get_snapshot(reference_time_ms=now_ms)
    macro_evidences = macro_calendar_to_evidence(snapshot)

    result = confluence_reconcile(
        evidences=macro_evidences,
        symbol="BTCUSDT",
        observation_open_ms=now_ms - 60_000,
        observation_close_ms=now_ms,
        causal_anchor_ms=now_ms,
    )

    assert result.status == ReconcilerStatus.INSUFFICIENT_DATA
    assert len(result.valid_directional_evidence) == 0
    assert len(result.directions_present) == 0
    assert result.evidence_count == 0


# ==============================================================================
# 10. STRICT BOUNDARIES: NO RISK, NO TRADE GATES, NO BLACKOUT WINDOWS
# ==============================================================================

def test_no_forbidden_fields_in_contracts():
    """
    Garante que os dataclasses não possuem campos de trade gate, blackout window,
    danger zones, cooldowns, confidence ou sinais de compra/venda.
    """
    forbidden_terms = [
        "trade", "blackout", "danger", "cooldown", "gate", "buy", "sell",
        "sizing", "risk_reduction", "kill_switch", "block", "confidence"
    ]

    event_fields = ScheduledMacroEvent.__dataclass_fields__.keys()
    snapshot_fields = MacroCalendarSnapshot.__dataclass_fields__.keys()

    for term in forbidden_terms:
        for f in event_fields:
            assert term not in f.lower(), f"Campo proibido '{f}' encontrado em ScheduledMacroEvent"
        for f in snapshot_fields:
            assert term not in f.lower(), f"Campo proibido '{f}' encontrado em MacroCalendarSnapshot"


# ==============================================================================
# 11. WEEKEND INDEPENDENCE (J1/J2 SCENARIOS)
# ==============================================================================

def test_weekend_independence():
    """
    Verifica que o calendário macro opera independentemente de ser final de semana (ex: domingo).
    Não assume automaticamente que domingo = sem eventos se o calendário for MISSING.
    """
    provider = LocalJsonMacroCalendarProvider()
    # Se provedor está MISSING no domingo, relata MISSING, não 'AVAILABLE vazio'
    provider.set_availability(False)
    snap = provider.get_snapshot(reference_time_ms=1_700_000_000_000)
    assert snap.provider_status == CalendarProviderStatus.MISSING
    assert snap.validity == EvidenceValidity.UNKNOWN


# ==============================================================================
# 12. RFC 8259 SERIALIZATION
# ==============================================================================

def test_rfc8259_serialization():
    """Valida serialização JSON pura de ScheduledMacroEvent e MacroCalendarSnapshot."""
    ev = ScheduledMacroEvent(
        event_id="US_USD_US_CPI_2026-01-14",
        country="US",
        currency="USD",
        event_type=MacroEventType.US_CPI,
        title="CPI",
        scheduled_at_ms=1_768_483_800_000,
        scheduled_at_utc="2026-01-14T13:30:00Z",
        importance=MacroEventImportance.HIGH_IMPACT_TARGET_SET,
        source="audit_fixture",
    )

    ev_dict = ev.to_dict()
    ev_json = json.dumps(ev_dict)
    assert json.loads(ev_json) == ev_dict

    snap = MacroCalendarSnapshot(
        reference_time_ms=1_768_480_000_000,
        nearest_upcoming_event=ev,
        upcoming_events=[ev],
        recent_events=[],
        provider_status=CalendarProviderStatus.AVAILABLE,
        source="audit_fixture",
        last_refresh_ms=1_768_480_000_000,
        age_ms=0,
        validity=EvidenceValidity.VALID,
        reason=None,
    )

    snap_dict = snap.to_dict()
    snap_json = json.dumps(snap_dict)
    assert json.loads(snap_json) == snap_dict


# ==============================================================================
# 13. IDEMPOTENT DUPLICATE INGESTION
# ==============================================================================

def test_duplicate_events_idempotent():
    """Ingerir o mesmo evento idêntico múltiplas vezes não duplica registros."""
    provider = LocalJsonMacroCalendarProvider()
    ev = ScheduledMacroEvent(
        event_id="US_USD_US_NFP_2026-02-06",
        country="US",
        currency="USD",
        event_type=MacroEventType.US_NFP,
        title="Non-Farm Payrolls",
        scheduled_at_ms=1_770_384_600_000,
        scheduled_at_utc="2026-02-06T13:30:00Z",
        importance=MacroEventImportance.HIGH_IMPACT_TARGET_SET,
        source="bls_feed",
    )

    provider.add_or_update_event(ev)
    provider.add_or_update_event(ev)
    provider.add_or_update_event(ev)

    snapshot = provider.get_snapshot(reference_time_ms=1_770_300_000_000)
    assert len(snapshot.upcoming_events) == 1
    assert snapshot.upcoming_events[0].status == "SCHEDULED"


# ==============================================================================
# 14. NO ARBITRARY BEFORE/AFTER THRESHOLDS ("NEAR" / "DANGER")
# ==============================================================================

def test_no_arbitrary_thresholds_in_snapshot():
    """
    O snapshot expõe time_to_event_ms e nearest_upcoming_event factualmente.
    Não classifica eventos em 'danger_zone', 'near', ou 'pre_event_risk'.
    """
    now_ms = 1_700_000_000_000
    ev_1m = ScheduledMacroEvent(
        event_id="US_USD_US_CPI_2023-11-14",
        country="US",
        currency="USD",
        event_type=MacroEventType.US_CPI,
        title="CPI",
        scheduled_at_ms=now_ms + 60_000,  # 1 minuto
        scheduled_at_utc="2023-11-14T13:30:00Z",
        importance=MacroEventImportance.HIGH_IMPACT_TARGET_SET,
        source="test",
    )

    ev_10d = ScheduledMacroEvent(
        event_id="US_USD_FOMC_RATE_DECISION_2023-11-24",
        country="US",
        currency="USD",
        event_type=MacroEventType.FOMC_RATE_DECISION,
        title="FOMC",
        scheduled_at_ms=now_ms + 10 * 86400 * 1000,  # 10 dias
        scheduled_at_utc="2023-11-24T18:00:00Z",
        importance=MacroEventImportance.HIGH_IMPACT_TARGET_SET,
        source="test",
    )

    # Ambos são UPCOMING factualmente, sem labels arbitrários
    assert ev_1m.event_phase(now_ms) == MacroEventPhase.UPCOMING
    assert ev_10d.event_phase(now_ms) == MacroEventPhase.UPCOMING
    assert ev_1m.time_to_event_ms(now_ms) == 60_000
    assert ev_10d.time_to_event_ms(now_ms) == 10 * 86400 * 1000


# ==============================================================================
# 15. DETERMINISTIC SYNTHETIC FIXTURES (FOMC, CPI, NFP)
# ==============================================================================

def test_synthetic_fixtures_deterministic():
    """
    Cria fixtures sintéticas determinísticas claramente marcadas.
    Testa FOMC (tarde), CPI (manhã) e NFP (manhã) em EST e EDT.
    """
    raw_synthetic_schedule = [
        # Inverno (EST: UTC-5)
        {"country": "US", "currency": "USD", "event_type": "US_CPI", "date": "2026-01-14", "time_ny": "08:30", "source": "synthetic_audit_fixture", "title": "SYNTHETIC US CPI Jan 2026"},
        {"country": "US", "currency": "USD", "event_type": "US_NFP", "date": "2026-02-06", "time_ny": "08:30", "source": "synthetic_audit_fixture", "title": "SYNTHETIC US NFP Feb 2026"},
        {"country": "US", "currency": "USD", "event_type": "FOMC_RATE_DECISION", "date": "2026-01-28", "time_ny": "14:00", "source": "synthetic_audit_fixture", "title": "SYNTHETIC FOMC Jan 2026"},
        # Verão (EDT: UTC-4)
        {"country": "US", "currency": "USD", "event_type": "US_CPI", "date": "2026-06-10", "time_ny": "08:30", "source": "synthetic_audit_fixture", "title": "SYNTHETIC US CPI Jun 2026"},
        {"country": "US", "currency": "USD", "event_type": "US_NFP", "date": "2026-06-05", "time_ny": "08:30", "source": "synthetic_audit_fixture", "title": "SYNTHETIC US NFP Jun 2026"},
        {"country": "US", "currency": "USD", "event_type": "FOMC_RATE_DECISION", "date": "2026-06-17", "time_ny": "14:00", "source": "synthetic_audit_fixture", "title": "SYNTHETIC FOMC Jun 2026"},
    ]

    provider = LocalJsonMacroCalendarProvider(source_name="synthetic_audit_fixture")
    loaded_count = provider.load_from_dict_list(raw_synthetic_schedule)
    assert loaded_count == 6

    # Referência: antes do primeiro evento
    ref_ms, _ = parse_ny_datetime_to_utc_ms("2026-01-01", "00:00")
    snap = provider.get_snapshot(reference_time_ms=ref_ms)

    assert len(snap.upcoming_events) == 6
    assert snap.nearest_upcoming_event.event_type == MacroEventType.US_CPI
    assert snap.nearest_upcoming_event.scheduled_at_utc == "2026-01-14T13:30:00Z"  # EST (UTC-5)

    # Verifica conversão do FOMC em janeiro (EST: 14:00 + 5h = 19:00Z)
    fomc_jan = [e for e in snap.upcoming_events if e.event_id == "US_USD_FOMC_RATE_DECISION_2026-01-28"][0]
    assert fomc_jan.scheduled_at_utc == "2026-01-28T19:00:00Z"

    # Verifica conversão do FOMC em junho (EDT: 14:00 + 4h = 18:00Z)
    fomc_jun = [e for e in snap.upcoming_events if e.event_id == "US_USD_FOMC_RATE_DECISION_2026-06-17"][0]
    assert fomc_jun.scheduled_at_utc == "2026-06-17T18:00:00Z"

