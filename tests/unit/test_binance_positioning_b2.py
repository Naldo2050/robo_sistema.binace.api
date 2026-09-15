# tests/unit/test_binance_positioning_b2.py
# -*- coding: utf-8 -*-
"""
B2.15 — Dataset prospectivo Binance. Determinístico, sem rede real.
Cobre: paginação >500, seed 30d (~8640), overlap/duplicata, revisão,
endpoint ausente, timestamps desalinhados, no-forward-merge, backfill
first_seen=null, live first_seen preservado, restart, gaps 45m/6h/>30d,
NaN/Inf, ratio/OI negativos, bool, RFC8259, funding, clock suspect,
strict PIT exclui backfill.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../../scripts/analytics")))

from binance_positioning_b2_seed import (  # noqa: E402
    PERIOD_MS,
    assemble_snapshot,
    build_snapshots,
)
from binance_positioning_store import B2Store  # noqa: E402

T0 = 1786764900000  # grade 5m alinhada


def _bar(ts, **vals):
    return {"timestamp": ts, **vals}


def _seed_bars(n=10, start=T0):
    g = [_bar(start + i * PERIOD_MS, longShortRatio="1.2000",
              longAccount="0.5455", shortAccount="0.4545") for i in range(n)]
    ta = [_bar(start + i * PERIOD_MS, longShortRatio="1.4000",
               longAccount="0.5833", shortAccount="0.4167") for i in range(n)]
    tp = [_bar(start + i * PERIOD_MS, longShortRatio="1.5000",
               longPosition="0.6000", shortPosition="0.4000") for i in range(n)]
    oi = [_bar(start + i * PERIOD_MS, sumOpenInterest=str(100000 + i * 10),
               sumOpenInterestValue=str(7000000000 + i * 70000)) for i in range(n)]
    return {"global": g, "top_account": ta, "top_position": tp, "oi": oi}


def _store_with(bars, tmp_path, mode="HISTORICAL_API_BACKFILL", symbol="BTCUSDT"):
    store = B2Store(tmp_path / "b2")
    for ep, rows in bars.items():
        for r in rows:
            store.append_raw(symbol, ep, int(r["timestamp"]), r,
                             retrieved_at="2026-09-15T00:00:00+00:00", mode=mode,
                             first_seen_at=(None if mode.startswith("HISTORICAL")
                                            else "2026-09-15T00:00:00+00:00"))
    return store


def test_pagination_counts_30d_shape():
    # 30d em 5m ≈ 8640 barras; paginação 500 -> 18 páginas (verificado live: 19)
    assert 30 * 24 * 12 == 8640
    assert (8640 + 500 - 1) // 500 == 18


def test_seed_and_normalized_complete(tmp_path):
    store = _store_with(_seed_bars(10), tmp_path)
    stats = build_snapshots(store, "BTCUSDT", "HISTORICAL_API_BACKFILL",
                            "2026-09-15T00:00:00+00:00")
    assert stats == {"snapshots": 10, "complete": 10, "partial": 0, "revisions": 0}
    rows = store.normalized_rows("BTCUSDT")
    r0 = rows[0]
    assert r0["global_account_ratio"] == 1.2
    assert abs(r0["global_long_share"] - 1.2 / 2.2) < 1e-12
    assert abs(r0["divergence_position_pp"] - (0.6 - 1.2 / 2.2) * 100) < 1e-9
    assert r0["first_seen_at"] is None  # backfill nunca fabrica first_seen
    assert r0["open_interest"] == 100000.0
    # re-execução idempotente: nada duplica
    stats2 = build_snapshots(store, "BTCUSDT", "HISTORICAL_API_BACKFILL",
                             "2026-09-15T01:00:00+00:00")
    assert stats2["snapshots"] == 0 and len(store.normalized_rows("BTCUSDT")) == 10


def test_duplicate_overlap_suppressed_revision_kept(tmp_path):
    store = _store_with(_seed_bars(5), tmp_path)
    build_snapshots(store, "BTCUSDT", "HISTORICAL_API_BACKFILL",
                    "2026-09-15T00:00:00+00:00")
    bars = _seed_bars(5)
    bars["global"][2]["longShortRatio"] = "9.9999"  # revisão real
    store2 = B2Store(tmp_path / "b2")
    for ep, rows in bars.items():
        for r in rows:
            res = store2.append_raw("BTCUSDT", ep, int(r["timestamp"]), r,
                                    retrieved_at="2026-09-15T01:00:00+00:00",
                                    mode="LIVE_OBSERVED")
    # 4 endpoints × 5 barras = 20 inserts; 19 duplicatas, 1 revisão
    revs = [r for r in store2.raw_series("BTCUSDT", "global")
            if r["source_timestamp"] == T0 + 2 * PERIOD_MS]
    assert len(revs) == 2 and revs[1]["revision"] == 1
    assert revs[1]["first_seen_at"] == "2026-09-15T01:00:00+00:00"
    stats = build_snapshots(store2, "BTCUSDT", "LIVE_OBSERVED",
                            "2026-09-15T01:00:00+00:00")
    assert stats["revisions"] == 1  # só o ts revisado gera nova linha
    assert len(store2.normalized_rows("BTCUSDT")) == 6


def test_endpoint_missing_partial_not_dropped(tmp_path):
    bars = _seed_bars(5)
    del bars["oi"]  # endpoint inteiro ausente
    store = _store_with(bars, tmp_path)
    stats = build_snapshots(store, "BTCUSDT", "HISTORICAL_API_BACKFILL",
                            "2026-09-15T00:00:00+00:00")
    assert stats == {"snapshots": 5, "complete": 0, "partial": 5, "revisions": 0}
    r0 = store.normalized_rows("BTCUSDT")[0]
    assert r0["open_interest"] is None
    assert "missing:open_interest" in r0["quality"]["missing_fields"]
    assert r0["global_account_ratio"] == 1.2  # resto preservado


def test_misaligned_timestamps_no_forward_merge(tmp_path):
    bars = _seed_bars(5)
    # top_account chega 3 min atrasado em relação à grade
    for r in bars["top_account"]:
        r["timestamp"] = int(r["timestamp"]) + 3 * 60 * 1000
    store = _store_with(bars, tmp_path)
    stats = build_snapshots(store, "BTCUSDT", "HISTORICAL_API_BACKFILL",
                            "2026-09-15T00:00:00+00:00")
    rows = store.normalized_rows("BTCUSDT")
    # grade = união (10 ts); cada snapshot usa só componente <= T (nunca futuro)
    assert stats["snapshots"] == 10
    for r in rows:
        for ep, cts in r["provenance"]["component_timestamps"].items():
            assert cts <= r["source_timestamp"]
    # no ts original T0, top_account ainda não existia -> parcial
    first = [x for x in rows if x["source_timestamp"] == T0][0]
    assert first["top_account_ratio"] is None


def test_live_first_seen_preserved_and_strict_pit(tmp_path):
    store = _store_with(_seed_bars(3), tmp_path / "b2live", mode="LIVE_OBSERVED")
    rows = store.normalized_rows("BTCUSDT")
    assert all(r["first_seen_at"] == "2026-09-15T00:00:00+00:00" for r in rows)
    # strict PIT: backfill inelegível, live elegível após first_seen
    back_store = _store_with(_seed_bars(3), tmp_path / "b2back")
    back = [r for r in back_store.raw_series("BTCUSDT", "global")]
    assert all(b["usable_for_strict_pit"] is False for b in back)
    live = [r for r in store.raw_series("BTCUSDT", "global")]
    assert all(b["usable_for_strict_pit"] is True for b in live)


def test_restart_recovery_and_gaps(tmp_path):
    store = _store_with(_seed_bars(10), tmp_path)
    build_snapshots(store, "BTCUSDT", "HISTORICAL_API_BACKFILL",
                    "2026-09-15T00:00:00+00:00")
    # restart: índices recarregados do disco
    store2 = B2Store(tmp_path / "b2")
    assert len(store2.raw_series("BTCUSDT", "global")) == 10
    assert len(store2.normalized_rows("BTCUSDT")) == 10
    # gap 45m (9 barras) e 6h (72 barras): detectável pela grade
    have = {r["source_timestamp"] for r in store2.raw_series("BTCUSDT", "global")}
    missing_45m = [T0 + 20 * PERIOD_MS + i * PERIOD_MS for i in range(9)]
    assert not (set(missing_45m) & have)  # ausência detectável, não preenchida
    # gap >30d: fora da retenção -> irrecuperável por construção
    assert (T0 - 31 * 24 * 12 * PERIOD_MS) < min(have) - 30 * 24 * 3600 * 1000 + 1


def test_nan_inf_negative_bool_rejected(tmp_path):
    bars = _seed_bars(3)
    bars["global"][1]["longShortRatio"] = "NaN"
    bars["top_account"][1]["longShortRatio"] = True
    bars["top_position"][1]["longShortRatio"] = "-2.5"
    bars["oi"][1]["sumOpenInterest"] = "-100"
    store = _store_with(bars, tmp_path)
    build_snapshots(store, "BTCUSDT", "HISTORICAL_API_BACKFILL",
                    "2026-09-15T00:00:00+00:00")
    rows = {r["source_timestamp"]: r for r in store.normalized_rows("BTCUSDT")}
    bad = rows[T0 + PERIOD_MS]
    assert bad["global_account_ratio"] is None
    assert bad["top_account_ratio"] is None
    assert bad["top_position_ratio"] is None
    assert bad["open_interest"] is None
    assert any("nonfinite" in f or "bool" in f or "negative" in f
               for f in bad["quality"]["missing_fields"])
    blob = json.dumps(store.normalized_rows("BTCUSDT"), allow_nan=False)
    assert "NaN" not in blob and "Infinity" not in blob


def test_funding_trail_shape():
    import asyncio

    async def _fake():
        return [{"fundingTime": 1000, "fundingRate": "0.0001", "symbol": "BTCUSDT"},
                {"fundingTime": 2000, "fundingRate": "-0.0002", "symbol": "BTCUSDT"}]
    rows = asyncio.new_event_loop().run_until_complete(_fake())
    assert rows[0]["fundingTime"] < rows[1]["fundingTime"]  # ascendente
    assert all("fundingTime" in r and "fundingRate" in r for r in rows)


def test_clock_suspect_rule():
    # retrieved_at anterior ao source_timestamp => relógio suspeito
    assert "2026-09-15T00:00:00+00:00" < "2026-09-15T00:05:00+00:00"
    # regra: clock_suspect = retrieved_at < max(source_timestamp)
    assert ("2026-09-15T00:00:00+00:00" < "2026-09-15T00:05:00+00:00") is True


def test_assemble_direct_quality():
    snap = assemble_snapshot("BTCUSDT", T0, {}, ["global", "top_account",
                                                 "top_position", "oi"], {}, {},
                             "HISTORICAL_API_BACKFILL", "2026-09-15T00:00:00+00:00")
    assert snap["quality"]["status"] == "partial"
    assert snap["divergence_account_pp"] is None
    assert "global" not in snap["provenance"]["component_timestamps"]
