# tests/unit/test_cftc_cot_point_in_time.py
# -*- coding: utf-8 -*-
"""
Anti-look-ahead CFTC COT (P5). Puro e offline: nenhuma rede, relógio injetado.

Semana canônica: asof terça 2026-09-08, publicação sexta 2026-09-11 15:30 ET
(19:30 UTC, EDT). Prova que report_as_of_date nunca é available_at.
"""
from datetime import datetime, timezone

from institutional.cftc_cot import (
    NO_POINT_IN_TIME,
    expected_publication_utc,
    select_point_in_time,
)

TUE = "2026-09-08"
PUB_UTC = datetime(2026, 9, 11, 19, 30, tzinfo=timezone.utc)  # sexta 15:30 EDT


def _rec(asof=TUE, first_seen="2026-09-11T19:45:00+00:00", **over):
    r = {"report_as_of_date": asof, "first_seen_at": first_seen,
         "source_row_id": "260908133741F"}
    r.update(over)
    return r


def _at(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def test_tuesday_before_publication_invisible():
    sel = select_point_in_time([_rec()], _at("2026-09-08T10:00:00-04:00"))
    assert sel["record"] is None


def test_friday_before_1530et_invisible():
    sel = select_point_in_time([_rec()], _at("2026-09-11T14:00:00-04:00"))
    assert sel["record"] is None


def test_after_publication_visible():
    sel = select_point_in_time([_rec()], _at("2026-09-11T20:00:00+00:00"))
    assert sel["record"] is not None


def test_weekend_sees_previous_week():
    sel = select_point_in_time([_rec()], _at("2026-09-13T12:00:00+00:00"))
    assert sel["record"]["report_as_of_date"] == TUE


def test_holiday_delayed_publication():
    # Semana do feriado US 2026-07-04 (sábado; release remarcado 06* = segunda):
    # calendário cai na sexta 03/07; aqui o atraso real empurra para 06/07.
    # Com first_seen real, vale o first_seen, não o calendário.
    rec = _rec(asof="2026-06-30", first_seen="2026-07-06T19:30:00+00:00")
    assert select_point_in_time([rec], _at("2026-07-03T20:00:00+00:00"))["record"] is None
    assert select_point_in_time([rec], _at("2026-07-06T20:00:00+00:00"))["record"] is not None


def test_late_publication_not_backfilled():
    rec = _rec(first_seen="2026-09-14T10:00:00+00:00")  # 3 dias de atraso
    assert select_point_in_time([rec], _at("2026-09-12T00:00:00+00:00"))["record"] is None
    assert select_point_in_time([rec], _at("2026-09-14T11:00:00+00:00"))["record"] is not None


def test_later_revision_does_not_rewrite_history():
    v0 = _rec(first_seen="2026-09-11T19:45:00+00:00", rev=0)
    v1 = _rec(first_seen="2026-09-20T10:00:00+00:00", rev=1)  # revisão 9 dias depois
    recs = [v0, v1]
    early = select_point_in_time(recs, _at("2026-09-12T00:00:00+00:00"))
    assert early["record"]["rev"] == 0  # replay antigo vê a versão original
    late = select_point_in_time(recs, _at("2026-09-21T00:00:00+00:00"))
    assert late["record"]["rev"] == 1


def test_missing_first_seen_without_fallback_is_unavailable():
    rec = {"report_as_of_date": TUE, "source_row_id": "260908133741F"}
    sel = select_point_in_time([rec], _at("2026-09-20T00:00:00+00:00"),
                               allow_calendar_fallback=False)
    assert sel["record"] is None and sel["reason"] == NO_POINT_IN_TIME


def test_missing_first_seen_with_calendar_fallback():
    rec = {"report_as_of_date": TUE, "source_row_id": "260908133741F"}
    assert select_point_in_time([rec], _at("2026-09-11T19:00:00+00:00"))["record"] is None
    sel = select_point_in_time([rec], _at("2026-09-11T22:00:00+00:00"))
    assert sel["record"] is not None  # calendário + grace 2h


def test_dst_transition_nov2026():
    # Fim do DST: 2026-11-01 02:00 ET. Publicação sexta 30/10 ainda é EDT.
    pub = expected_publication_utc(__import__("datetime").date(2026, 10, 27))
    assert pub == datetime(2026, 10, 30, 19, 30, tzinfo=timezone.utc)
    # Semana seguinte (asof 03/11, terça) publica 06/11 já em EST (20:30 UTC).
    pub2 = expected_publication_utc(__import__("datetime").date(2026, 11, 3))
    assert pub2 == datetime(2026, 11, 6, 20, 30, tzinfo=timezone.utc)


def test_utc_et_conversion_boundary():
    # 19:29 UTC de sexta = 15:29 ET (antes); 19:31 UTC = depois (com first_seen).
    rec = _rec(first_seen="2026-09-11T19:30:30+00:00")
    assert select_point_in_time([rec], _at("2026-09-11T19:29:00+00:00"))["record"] is None
    assert select_point_in_time([rec], _at("2026-09-11T19:31:00+00:00"))["record"] is not None


def test_duplicate_identical_rows_single_visibility():
    recs = [_rec(first_seen="2026-09-11T19:45:00+00:00"),
            _rec(first_seen="2026-09-11T19:45:00+00:00")]
    sel = select_point_in_time(recs, _at("2026-09-12T00:00:00+00:00"))
    assert sel["record"] is not None and sel["reason"] == "ok"


def test_expected_publication_is_friday_1530et():
    assert expected_publication_utc(
        __import__("datetime").date(2026, 9, 8)) == PUB_UTC
