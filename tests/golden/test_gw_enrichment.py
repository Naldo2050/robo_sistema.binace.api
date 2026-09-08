# tests/golden/test_gw_enrichment.py — L2: snapshots onchain/cross reais.
#
# GW4 congela SOMENTE contratos já corrigidos (lista fechada no teste);
# P1 (aggressive 50/50, RSI 50, returns 0, prob_up) em quarentena: intocados.
# GW6: shared_session real sobre closes da fixture (n exato) + gate de versão.

import pandas as pd
import pytest

from common.ml_features import _map_correlations_to_features
from fetchers.onchain_updater import OnchainSnapshot, OnchainUpdater
from market_analysis.cross_asset_correlations import shared_session_corr
from market_analysis.cross_asset_updater import CrossAssetUpdater

from .conftest import FakeMonotonic, assert_contract_version, load_fixture


def _onchain_updater(spec, mono):
    oc = spec["onchain"]
    updater = OnchainUpdater(monotonic_fn=mono, fetcher=object())
    if oc["mode"] == "unavailable":
        # snapshot ancestral: idade além do usable => unavailable (nada elegível)
        updater._snapshot = OnchainSnapshot(
            fast={"fees_fastest_sat_vb": 99}, slow={"difficulty": 1.0},
            fast_fetched_at_ms=0, slow_fetched_at_ms=0,
            fast_fetched_monotonic=mono() - 100000.0,
            slow_fetched_monotonic=mono() - 100000.0,
        )
        return updater
    updater._snapshot = OnchainSnapshot(
        fast=dict(oc["fast"]), slow=dict(oc["slow"]),
        fast_fetched_at_ms=spec["frozen_now_ms"] - oc["age_s"] * 1000,
        slow_fetched_at_ms=spec["frozen_now_ms"] - oc["age_s"] * 1000,
        fast_fetched_monotonic=mono() - oc["age_s"],
        slow_fetched_monotonic=mono() - oc["age_s"],
    )
    return updater


def _cross_updater(spec, mono):
    cx = spec["cross"]
    updater = CrossAssetUpdater(monotonic_fn=mono)
    if cx["mode"] == "warming":
        return updater  # sem snapshot => warming_up
    updater._store_snapshot_for_test(dict(cx["values"]), age_s=cx["age_s"])
    return updater


def test_gw1_enrichment_fresh(frozen_state):
    spec = load_fixture("gw1_balanced.json")
    mono = FakeMonotonic()
    oc_view = _onchain_updater(spec, mono).read_view()
    assert oc_view["fast"]["status"] == "fresh"
    assert oc_view["slow"]["status"] == "fresh"
    assert oc_view["fast"]["age_seconds"] == pytest.approx(10.0)
    assert oc_view["fast"]["values"]["fees_fastest_sat_vb"] == 12
    cx_view = _cross_updater(spec, mono).read_view()
    assert cx_view["status"] == "fresh"
    feats = _map_correlations_to_features(cx_view["values"])
    assert feats["btc_eth_corr_7d"] == 0.92
    assert feats["btc_dxy_corr_30d"] == -0.41
    assert_contract_version(feats, spec["expected_contract_version"], "gw1 cross")


def test_gw4_external_unavailable_contracts(frozen_state):
    """Somente contratos JÁ corrigidos. P1 quarentena: nenhum assert aqui."""
    spec = load_fixture("gw4_external_unavailable.json")
    mono = FakeMonotonic()
    mono.advance(100000.0)  # snapshot jamais publicado => unavailable/warming
    updater = _onchain_updater(spec, mono)
    oc_view = updater.read_view()
    assert oc_view["fast"]["status"] == "unavailable"
    assert oc_view["slow"]["status"] == "unavailable"
    # downstream real: grupos unavailable contribuem com NADA
    from data_processing.data_enricher import DataEnricher
    metrics = DataEnricher({}, onchain_updater=updater)._build_onchain_metrics()
    assert metrics == {}
    cx_view = _cross_updater(spec, mono).read_view()
    assert cx_view["status"] == "warming_up"
    assert cx_view["values"] == {}
    feats = _map_correlations_to_features(cx_view["values"])
    # mapping puro preenche NaN/None (missing); o gate fresh/stale (E3-B) é quem
    # zera o dict no ml — aqui congela-se: NaN/None, nunca 0.0/0.
    assert pd.isna(feats["btc_dxy_corr_30d"])
    assert pd.isna(feats["btc_dxy_corr_90d"])
    assert feats["btc_ndx_corr_30d"] is None
    assert feats["cross_asset_method"] is None  # legado => positional_v1
    assert feats["cross_asset_contract_version"] == 1
    # dominance ausente != 0 (B-P0-4)
    assert pd.isna(feats["btc_dominance_change_7d"])
    from core.window_state import WindowState
    ws = WindowState()
    assert ws.flow.flow_imbalance is None
    assert ws.flow.buy_sell_ratio is None
    assert ws.flow.pressure_label is None
    assert ws.derivatives.btc_long_short_ratio is None
    assert ws.derivatives.btc_open_interest is None
    assert ws.flow.validate() == [] and ws.derivatives.validate() == []


def _gw6_series(fx):
    s, e = fx["btc_calendar"]["start"], fx["btc_calendar"]["end"]
    excl = set(fx["tradfi_weekdays"]["exclusions"])
    f = fx["closes_formula"]
    cal = pd.date_range(s, e, freq="D")
    biz = [d for d in cal if d.weekday() < 5 and d.date().isoformat() not in excl]
    bmap, tmap = {}, {}
    bv, tv = f["btc_start"], f["tradfi_start"]
    j = 0
    denom = max(len(biz) - 1, 1)
    for d in cal:
        iso = d.date().isoformat()
        if d.weekday() >= 5:
            bmap[iso] = bv
            continue
        drift = f["daily_drift"] * (0.5 + j / denom)
        if iso in excl:
            bv = bv * (1 + drift)
            bmap[iso] = bv
            j += 1
            continue
        bv = bv * (1 + drift)
        tv = tv * (1 + drift)
        bmap[iso] = bv
        tmap[iso] = tv
        j += 1
    btc = pd.Series([bmap[d.date().isoformat()] for d in cal],
                    index=pd.DatetimeIndex(cal, tz="UTC"))
    tra = pd.Series([tmap[d.date().isoformat()] for d in biz],
                    index=pd.DatetimeIndex(biz, tz="UTC"))
    return btc, tra


def test_gw6_shared_session_exact_n(frozen_state):
    fx = load_fixture("gw6_weekend_holiday.json")
    btc, tra = _gw6_series(fx)
    dec = pd.Timestamp(fx["decision_ms"], unit="ms", tz="UTC").date()
    out = shared_session_corr(btc, tra, fx["target_returns"], decision_date=dec)
    assert out["n"] == 30  # EXATO pela fixture (31 sessões)
    assert out["corr"] == pytest.approx(fx["expected_corr_full"], abs=1e-6)
    assert out["first"] == "2026-07-27" and out["last"] == "2026-09-08"
    pairs = out["pairs"]
    assert pairs, "sem pares?"
    for a, b in pairs:
        assert a == b  # mesma data nos dois lados, sempre
        wd = pd.Timestamp(a).weekday()
        assert wd < 5 and a != "2026-09-07"  # sem fds/feriado


def test_gw6_insufficient_exact_n(frozen_state):
    fx = load_fixture("gw6_weekend_holiday.json")
    short = fx["short_case"]
    btc = pd.Series(
        [78600.0 * (1.0008) ** i for i in range(len(short["shared_dates"]))],
        index=pd.DatetimeIndex(short["shared_dates"], tz="UTC"))
    tra = pd.Series(
        [98.9 * (1.0008) ** i for i in range(len(short["shared_dates"]))],
        index=pd.DatetimeIndex(short["shared_dates"], tz="UTC"))
    out = shared_session_corr(
        btc, tra, 30,
        decision_date=pd.Timestamp("2026-09-15", tz="UTC").date())
    assert out["n"] == short["expected_n"] == 8  # EXATO
    assert pd.isna(out["corr"])  # insufficient: NaN, nunca 0.0
    assert out["corr"] != 0.0
