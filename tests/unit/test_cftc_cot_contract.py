# tests/unit/test_cftc_cot_contract.py
# -*- coding: utf-8 -*-
"""
Contrato CFTC COT (P2/P3) — testes offline com fixtures reais minimizadas.

Sem rede: usa tests/fixtures/cftc_cot/*.json (linha real 260908133741F).
Relógio injetado (analyze aceita `now`); fetcher com cache temporário.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fetchers.cftc_cot_fetcher import CftcCotFetcher, SYMBOL_TO_CONTRACT
from institutional.cftc_cot import CftcCot

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "cftc_cot"
NOW_OK = datetime(2026, 9, 12, 0, 5, 0, tzinfo=timezone.utc)  # sexta pós-publicação


def _row(name="tff_btc_futonly_2026-09-08.json"):
    with open(FIX / name, encoding="utf-8") as fh:
        return json.load(fh)


def test_symbol_map_explicit_no_fuzzy():
    assert SYMBOL_TO_CONTRACT["BTCUSDT"][0] == "133741"
    assert SYMBOL_TO_CONTRACT["ETHUSDT"][0] == "146021"
    assert "DOGEUSDT" not in SYMBOL_TO_CONTRACT


def test_parse_real_row_available():
    cot = CftcCot()
    snap = cot.analyze(_row(), symbol="BTCUSDT", contract_code="133741",
                       first_seen_at="2026-09-11T19:45:00+00:00",
                       retrieved_at="2026-09-12T00:05:00+00:00", now=NOW_OK)
    assert snap.status == "AVAILABLE"
    assert snap.is_available is True and snap.is_stale is False
    assert snap.report_as_of_date == "2026-09-08"
    assert snap.open_interest["total"] == 21083
    assert snap.positions["leveraged"]["long"] == 5146
    assert snap.positions["leveraged"]["short"] == 13038
    assert snap.positions["leveraged"]["net"] == 5146 - 13038
    assert snap.positions["nonreportable"]["spreading"] is None
    assert snap.error_code is None
    # terminologia oficial preservada, sem smart money
    assert set(snap.positions) == {"dealer", "asset_manager", "leveraged",
                                   "other", "nonreportable"}


def test_string_numbers_bool_nan_rejected():
    cot = CftcCot()
    row = _row()
    row["dealer_positions_long_all"] = True
    row["lev_money_positions_short"] = "NaN"
    row["open_interest_all"] = "Infinity"
    snap = cot.analyze(row, symbol="BTCUSDT", contract_code="133741", now=NOW_OK)
    assert snap.status == "INVALID"
    assert snap.is_available is False


def test_non_tuesday_invalid():
    cot = CftcCot()
    row = _row()
    row["report_date_as_yyyy_mm_dd"] = "2026-09-09T00:00:00.000"  # quarta
    snap = cot.analyze(row, symbol="BTCUSDT", contract_code="133741", now=NOW_OK)
    assert snap.status == "INVALID"


def test_none_row_unavailable_fail_closed():
    snap = CftcCot().analyze(None, symbol="BTCUSDT", now=NOW_OK)
    assert snap.status == "UNAVAILABLE"
    assert snap.is_available is False


def test_unsupported_symbol():
    snap = CftcCot().unsupported("DOGEUSDT")
    assert snap.status == "UNSUPPORTED"
    assert snap.is_available is False
    assert snap.error_code == "unsupported_symbol"
    assert snap.positions == {}


def test_partial_missing_category():
    cot = CftcCot()
    row = _row()
    del row["asset_mgr_positions_long"]
    snap = cot.analyze(row, symbol="BTCUSDT", contract_code="133741",
                       retrieved_at="2026-09-12T00:05:00+00:00", now=NOW_OK)
    assert snap.status == "PARTIAL"
    assert snap.is_available is True  # parcial explícito, não silencioso
    assert "asset_mgr_positions_long" in snap.quality["missing_fields"]
    assert snap.positions["asset_manager"]["net"] is None


def test_wow_needs_prev():
    cot = CftcCot()
    snap = cot.analyze(_row(), symbol="BTCUSDT", contract_code="133741",
                       retrieved_at="2026-09-12T00:05:00+00:00", now=NOW_OK)
    assert snap.open_interest["change_wow"] is None  # sem prev => null, nunca 0
    prev = dict(_row())
    prev["open_interest_all"] = "20000"
    prev["lev_money_positions_long"] = "5000"
    prev["lev_money_positions_short"] = "13000"
    snap2 = cot.analyze(_row(), symbol="BTCUSDT", contract_code="133741",
                        retrieved_at="2026-09-12T00:05:00+00:00",
                        prev_row=prev, now=NOW_OK)
    assert snap2.open_interest["change_wow"] == 1083
    assert snap2.derived_metrics["wow_net_change"]["leveraged"] == (5146 - 13038) - (5000 - 13000)


def test_ingest_versioning_append_only(tmp_path):
    fetcher = CftcCotFetcher(cache_path=tmp_path / "cftc_cache.json")
    row = _row()
    rec1, err1 = fetcher.ingest_row("133741", row, "2026-09-11T19:45:00+00:00")
    assert err1 is None and rec1.revision == 0
    rec_dup, _ = fetcher.ingest_row("133741", dict(row), "2026-09-11T20:00:00+00:00")
    assert rec_dup.revision == 0  # duplicata idêntica: sem nova revisão
    rev = _row("tff_btc_revision.json")
    rec2, err2 = fetcher.ingest_row("133741", rev, "2026-09-12T00:05:00+00:00")
    assert err2 is None and rec2.revision == 1
    # P5.1 item 3C: revisão tem first_seen próprio (nunca herda o da v0).
    assert rec2.first_seen_at == "2026-09-12T00:05:00+00:00"
    assert rec1.first_seen_at == "2026-09-11T19:45:00+00:00"
    assert len(fetcher._records[("133741", "2026-09-08")]) == 2


def test_snapshot_rfc8259_strict():
    snap = CftcCot().analyze(_row(), symbol="BTCUSDT", contract_code="133741",
                             first_seen_at="2026-09-11T19:45:00+00:00",
                             retrieved_at="2026-09-12T00:05:00+00:00", now=NOW_OK)
    payload = json.dumps(snap.to_dict(), allow_nan=False)
    assert '"NaN"' not in payload and '"Infinity"' not in payload

    def _walk(v):
        if isinstance(v, float):
            assert math.isfinite(v)
        elif isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, list):
            for x in v:
                _walk(x)

    _walk(snap.to_dict())
