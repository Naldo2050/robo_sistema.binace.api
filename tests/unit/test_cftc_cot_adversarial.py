# tests/unit/test_cftc_cot_adversarial.py
# -*- coding: utf-8 -*-
"""
Adversariais CFTC COT (P5.1 item 7). 100% offline: fixtures, fakes e tmp_path.
Sem rede, sem relógio real (relógio injetado onde há tempo).
"""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fetchers import cftc_cot_fetcher as fetcher_mod
from fetchers.cftc_cot_fetcher import CftcCotFetcher
from institutional.cftc_cot import (
    BASIS_CALENDAR_FALLBACK,
    BASIS_FIRST_SEEN,
    CftcCot,
    select_point_in_time,
    select_research_history,
)

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "cftc_cot"


def _row(name="tff_btc_futonly_2026-09-08.json"):
    with open(FIX / name, encoding="utf-8") as fh:
        return json.load(fh)


def _at(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# -- HIGH-2: fallback OFF por default ------------------------------------

def test_fallback_off_by_default_strict():
    rec = {"report_as_of_date": "2026-09-08"}
    sel = select_point_in_time([rec], _at("2026-09-20T00:00:00+00:00"))
    assert sel["record"] is None
    assert sel["availability_basis"] != BASIS_CALENDAR_FALLBACK


def test_holiday_without_first_seen_strict_ineligible():
    # Semana do 04/07/2026: calendário ingênuo preveria 03/07; strict barra.
    rec = {"report_as_of_date": "2026-06-30"}
    sel = select_point_in_time([rec], _at("2026-07-05T12:00:00+00:00"))
    assert sel["record"] is None


def test_fallback_on_is_marked_never_evidence():
    rec = {"report_as_of_date": "2026-09-08"}
    sel = select_point_in_time([rec], _at("2026-09-12T00:00:00+00:00"),
                               allow_calendar_fallback=True)
    assert sel["record"] is not None
    assert sel["availability_basis"] == BASIS_CALENDAR_FALLBACK
    assert sel["first_seen_at"] is None
    assert sel["calendar_estimated_at"] is not None


def test_late_first_seen_preserved_as_effective():
    rec = {"report_as_of_date": "2026-09-08",
           "first_seen_at": "2026-09-14T10:00:00+00:00"}
    assert select_point_in_time([rec], _at("2026-09-12T00:00:00+00:00"))["record"] is None
    sel = select_point_in_time([rec], _at("2026-09-14T11:00:00+00:00"))
    assert sel["record"] is not None
    assert sel["availability_basis"] == BASIS_FIRST_SEEN
    assert sel["effective_available_at"] == "2026-09-14T10:00:00+00:00"


# -- ingestão fora de ordem / isolamento standard-micro -------------------

def test_socrata_rows_out_of_order_still_sorted(tmp_path):
    f = CftcCotFetcher(cache_path=tmp_path / "c.json")
    rows = []
    for asof, oi in (("2026-09-08", "21083"), ("2026-08-25", "19000"), ("2026-09-01", "19697")):
        r = _row()
        r["report_date_as_yyyy_mm_dd"] = asof + "T00:00:00.000"
        r["open_interest_all"] = oi
        r["id"] = asof.replace("-", "") + "133741F"
        rows.append(r)
    for r in (rows[0], rows[2], rows[1]):  # embaralhado
        rec, err = f.ingest_row("133741", r, "2026-09-12T00:00:00+00:00")
        assert err is None
    assert [x.report_as_of_date for x in f.history("133741")] == [
        "2026-08-25", "2026-09-01", "2026-09-08"]


def test_standard_micro_never_cross(tmp_path):
    f = CftcCotFetcher(cache_path=tmp_path / "c.json")
    micro = _row()
    micro["cftc_contract_market_code"] = "133742"
    micro["market_and_exchange_names"] = "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE"
    rec, err = f.ingest_row("133741", micro, "2026-09-12T00:00:00+00:00")
    assert rec is None and err == "schema_error"
    assert f.history("133741") == []


def test_required_field_renamed_or_missing(tmp_path):
    f = CftcCotFetcher(cache_path=tmp_path / "c.json")
    bad = _row()
    bad["REPORT_DATE"] = bad.pop("report_date_as_yyyy_mm_dd")
    rec, err = f.ingest_row("133741", bad, "2026-09-12T00:00:00+00:00")
    assert rec is None and err == "schema_error"


# -- HTTP 200 heterogêneo (sessão fake, sem rede) --------------------------

class _FakeResp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp

    def get(self, url, params=None, timeout=None):
        return self._resp


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_http_200_dict_instead_of_list(tmp_path):
    f = CftcCotFetcher(cache_path=tmp_path / "c.json")
    rows, err = _run(f._get_json(_FakeSession(_FakeResp(200, {"not": "a list"})), {}))
    assert rows is None and err == "schema_error"


def test_http_200_heterogeneous_list(tmp_path):
    f = CftcCotFetcher(cache_path=tmp_path / "c.json")
    rows, err = _run(f._get_json(
        _FakeSession(_FakeResp(200, ["junk", 42, None])), {}))
    assert err is None and rows == ["junk", 42, None]
    row, err2 = _run(f.fetch_latest("133741", _FakeSession(
        _FakeResp(200, ["junk"]))))
    assert row is None and err2 is not None  # linha inválida nunca vira snapshot


# -- cache: tmp órfão, restart, dois writers, contenção ---------------------

def test_orphan_tmp_ignored(tmp_path):
    cache = tmp_path / "c.json"
    cache.with_suffix(".tmp").write_text("{quebrado", encoding="utf-8")
    f = CftcCotFetcher(cache_path=cache)
    # tmp órfão nunca é lido como cache (somente o caminho canônico).
    assert f._records == {}
    rec, _ = f.ingest_row("133741", _row(), "2026-09-11T19:45:00+00:00")
    assert rec.revision == 0
    # após save, o canônico é JSON válido com o registro.
    fresh = CftcCotFetcher(cache_path=cache)
    assert ("133741", "2026-09-08") in fresh._records


def test_shadow_first_seen_lifecycle(tmp_path):
    cache = tmp_path / "c.json"
    f = CftcCotFetcher(cache_path=cache)
    # A. primeira observação: first_seen == retrieved aproximado
    rec1, _ = f.ingest_row("133741", _row(), "2026-09-11T19:45:00+00:00")
    assert rec1.first_seen_at == "2026-09-11T19:45:00+00:00"
    # B. mesma versão de novo: first_seen original preservado
    rec_dup, _ = f.ingest_row("133741", _row(), "2026-09-12T08:00:00+00:00")
    assert rec_dup.first_seen_at == "2026-09-11T19:45:00+00:00"
    assert rec_dup.revision == 0
    # C. revisão: novo hash, nova revision, novo first_seen
    rev = json.loads((FIX / "tff_btc_revision.json").read_text(encoding="utf-8"))
    rec2, _ = f.ingest_row("133741", rev, "2026-09-12T09:00:00+00:00")
    assert rec2.revision == 1
    assert rec2.first_seen_at == "2026-09-12T09:00:00+00:00"
    assert rec2.content_hash != rec1.content_hash
    # D. restart: first_seen carregado do disco, não recalculado
    f2 = CftcCotFetcher(cache_path=cache)
    loaded = f2._records[("133741", "2026-09-08")]
    assert [r.revision for r in loaded] == [0, 1]
    assert loaded[0].first_seen_at == "2026-09-11T19:45:00+00:00"
    assert loaded[1].first_seen_at == "2026-09-12T09:00:00+00:00"


def test_two_writers_merge_without_loss(tmp_path):
    cache = tmp_path / "c.json"
    fa = CftcCotFetcher(cache_path=cache)
    fb = CftcCotFetcher(cache_path=cache)
    r1 = _row()
    r2 = _row()
    r2["report_date_as_yyyy_mm_dd"] = "2026-09-01T00:00:00.000"
    r2["id"] = "260901133741F"
    fa.ingest_row("133741", r1, "2026-09-11T19:45:00+00:00")
    fb.ingest_row("133741", r2, "2026-09-11T19:46:00+00:00")
    fresh = CftcCotFetcher(cache_path=cache)
    asofs = sorted(k[1] for k in fresh._records)
    assert asofs == ["2026-09-01", "2026-09-08"]  # nenhum writer apagou o outro


def test_lock_contention_fail_closed_memory_intact(tmp_path, monkeypatch):
    f = CftcCotFetcher(cache_path=tmp_path / "c.json")
    monkeypatch.setattr(fetcher_mod, "_try_interprocess_lock",
                        lambda path: (None, True))
    rec, err = f.ingest_row("133741", _row(), "2026-09-11T19:45:00+00:00")
    assert err is None  # ingestão em memória sucede
    assert f.cache_write_errors == 1 and f.cache_lock_contended == 1
    assert ("133741", "2026-09-08") in f._records  # memória intacta
    assert not (tmp_path / "c.json").exists()  # nada escrito sob contenção


# -- naive datetime documentado ---------------------------------------------

def test_naive_datetime_assumed_utc_documented():
    naive = datetime(2026, 9, 12, 0, 0, 0)  # sem tz -> UTC por contrato
    rec = {"report_as_of_date": "2026-09-08",
           "first_seen_at": "2026-09-11T19:45:00+00:00"}
    assert select_point_in_time([rec], naive)["record"] is not None


# -- MEDIUM-3: estimated_availability ----------------------------------------

def test_estimated_availability_flag():
    cot = CftcCot()
    no_fs = cot.analyze(_row(), symbol="BTCUSDT", contract_code="133741",
                        retrieved_at="2026-09-12T00:05:00+00:00",
                        now=_at("2026-09-12T00:05:00+00:00"))
    assert no_fs.status in ("AVAILABLE", "PARTIAL")
    assert no_fs.quality["estimated_availability"] is True
    assert no_fs.age_available_seconds is None
    fs = cot.analyze(_row(), symbol="BTCUSDT", contract_code="133741",
                     first_seen_at="2026-09-11T19:45:00+00:00",
                     retrieved_at="2026-09-12T00:05:00+00:00",
                     now=_at("2026-09-12T00:05:00+00:00"))
    assert fs.quality["estimated_availability"] is False
    assert fs.age_available_seconds is not None


def _at(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# -- research vs strict -------------------------------------------------------

def test_research_history_no_guarantee():
    recs = [{"report_as_of_date": "2026-09-08"},
            {"report_as_of_date": "2026-09-01"}]
    out = select_research_history(recs)
    assert [r["report_as_of_date"] for r in out["rows"]] == [
        "2026-09-01", "2026-09-08"]
    assert out["point_in_time_guarantee"] is False


# -- MEDIUM-2: timestamps Binance ----------------------------------------------

def _binance_rows(now_ms):
    g = [{"symbol": "BTCUSDT", "longAccount": "0.55", "shortAccount": "0.45",
          "longShortRatio": "1.2222", "timestamp": now_ms}]
    t = [{"symbol": "BTCUSDT", "longAccount": "0.58", "shortAccount": "0.42",
          "longShortRatio": "1.3810", "timestamp": now_ms}]
    p = [{"symbol": "BTCUSDT", "longPosition": "0.60", "shortPosition": "0.40",
          "longShortRatio": "1.5000", "timestamp": now_ms}]
    oi = [{"symbol": "BTCUSDT", "sumOpenInterest": "100000",
           "sumOpenInterestValue": "7000000000", "timestamp": now_ms}]
    return g, t, p, oi


def test_binance_snapshot_retrieved_at_set():
    import time as _time
    from fetchers.binance_positioning_fetcher import BinancePositioningFetcher
    now_ms = int(_time.time() * 1000)
    g, t, p, oi = _binance_rows(now_ms)
    fetcher = BinancePositioningFetcher()
    fetcher._fetch_single_endpoint = AsyncMock(side_effect=[g, t, p, oi])

    async def _go():
        return await fetcher.fetch_positioning("BTCUSDT", force_refresh=True)

    snap = asyncio.new_event_loop().run_until_complete(_go())
    assert snap.is_available is True
    assert snap.retrieved_at is not None
    assert snap.source_as_of is not None
    # retrieved_at >= source_as_of (recebido depois da fonte)
    assert snap.retrieved_at >= snap.source_as_of


def test_binance_analysis_carries_three_timestamps():
    from institutional.crypto_cot import CryptoCOT
    d = {"global_account_ratio": 1.10, "top_account_ratio": 1.15,
         "top_position_ratio": 1.20, "is_available": True, "is_stale": False,
         "source_as_of": "2026-09-14T12:00:00+00:00",
         "retrieved_at": "2026-09-14T12:00:05+00:00"}
    res = CryptoCOT().analyze(d)
    assert res.source_as_of == "2026-09-14T12:00:00+00:00"
    assert res.retrieved_at == "2026-09-14T12:00:05+00:00"  # nunca = analyzed_at
    assert res.analyzed_at is not None and res.analyzed_at != res.retrieved_at
    assert res.regime is not None  # decisão inalterada
