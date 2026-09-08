# tests/golden/test_gw_system.py — L4: process_window_snapshot real end-to-end.
#
# REGRESSÃO CRÍTICA (ajuste 7): nenhum shape top-level alternativo é
# construído à mão. FlowAnalyzer/DataPipeline REAIS produzem o evento, e o
# MESMO objeto alimenta _populate_window_state (via process_window_snapshot).
# Bordas mockadas: fetch_orderbook, context_collector, inferência ML (P1),
# rede/clock (harness). DataPipeline/FlowAnalyzer/WindowState: reais.

import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pytest

from core.state_manager import StateManager
from flow_analyzer.core import FlowAnalyzer
from market_analysis.cross_asset_updater import CrossAssetUpdater
from market_orchestrator.windows import window_processor as wp

from .conftest import (
    FakeClock,
    FakeMonotonic,
    assert_contract_version,
    expected_volumes,
    gen_trades,
    load_fixture,
)


@dataclass
class _LevelsStub:
    def update_from_vp(self, vp):
        pass


@dataclass
class _HealthStub:
    def heartbeat(self, name):
        pass


@dataclass
class _SlogStub:
    def info(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass


class _SpanStub:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@dataclass
class _TracerStub:
    def start_span(self, *a, **k):
        return _SpanStub()


@dataclass
class _StoreStub:
    saved: List[Dict[str, Any]] = field(default_factory=list)

    def save_features(self, window_id, features):
        self.saved.append({"window_id": window_id})


@dataclass
class GoldenBot:
    symbol: str = "BTCUSDT"
    window_data: List[Dict[str, Any]] = field(default_factory=list)
    should_stop: bool = False
    _warmup_lock: Any = field(default_factory=threading.Lock)
    warming_up: bool = False
    warmup_windows_remaining: int = 0
    warmup_windows_required: int = 0
    min_trades_for_pipeline: int = 3
    window_count: int = 0
    window_end_ms: int = 0
    time_manager: Any = None
    delta_history: Any = field(default_factory=lambda: deque(maxlen=100))
    delta_std_dev_factor: float = 2.0
    volume_history: Any = field(default_factory=lambda: deque(maxlen=100))
    health_monitor: Any = field(default_factory=_HealthStub)
    levels: Any = field(default_factory=_LevelsStub)
    flow_analyzer: Any = None
    context_collector: Any = None
    cross_asset_updater: Any = None
    onchain_updater: Any = None
    feature_calc: Any = None
    feature_store: Any = field(default_factory=_StoreStub)
    last_valid_vp: Any = None
    last_valid_vp_time: float = 0.0
    ny_tz: Any = field(default_factory=lambda: ZoneInfo("America/New_York"))
    slog: Any = field(default_factory=_SlogStub)
    tracer: Any = field(default_factory=_TracerStub)
    recorded: Any = None

    def _process_signals(self, signals, pipeline, flow_metrics,
                         historical_profile, macro_context, ob_event, enriched,
                         close_ms, total_buy_volume, total_sell_volume,
                         valid_window_data):
        self.recorded = {"signals": signals, "pipeline": pipeline,
                         "flow_metrics": flow_metrics,
                         "macro_context": macro_context, "ob_event": ob_event,
                         "enriched": enriched, "close_ms": close_ms}


class _CtxStub:
    def __init__(self, ctx):
        self._ctx = ctx

    def get_context(self):
        return self._ctx


def _macro_ctx(spec):
    return {
        "external": spec.get("macro_external", {}),
        "derivatives": spec.get("derivatives", {}),
        "historical_vp": {"daily": {"val": 66000.0, "vah": 67000.0,
                                    "poc": 66500.0}},
        "mtf_trends": {"1d": {"realized_vol": 0.03}},
        "market_context": {},
        "market_environment": {},
    }


def _ob_event(spec):
    ob = spec["orderbook"]
    return {"bid_depth_usd": ob["bid_depth_usd"],
            "ask_depth_usd": ob["ask_depth_usd"],
            "imbalance": ob["imbalance"],
            "spread_bps": ob["spread_bps"],
            "is_valid": ob["is_valid"],
            "data_quality": {"data_source": ob["data_source"]}}


def _run_window(spec, monkeypatch, cross_values=None, macro=None, ob=None):
    clock = FakeClock()
    flow = FlowAnalyzer(time_manager=clock)
    trades = gen_trades(spec)
    for t in trades:
        clock._now = t["T"]
        flow.process_trade(t)
    macro = macro if macro is not None else _macro_ctx(spec)
    ob = ob if ob is not None else _ob_event(spec)
    monkeypatch.setattr(wp, "fetch_orderbook_with_retry", lambda bot, ms: ob)
    import ml.model_inference as _mli
    # P1 quarentena: prob sentinelada só p/ pass-through, nunca correção
    monkeypatch.setattr(_mli, "predict_up_probability", lambda feats: 0.62)
    bot = GoldenBot(window_data=[dict(t) for t in trades],
                    window_end_ms=trades[-1]["T"],
                    time_manager=clock, flow_analyzer=flow,
                    context_collector=_CtxStub(macro))
    if cross_values is not None:
        mono = FakeMonotonic()
        up = CrossAssetUpdater(monotonic_fn=mono)
        up._store_snapshot_for_test(dict(cross_values), age_s=10.0)
        bot.cross_asset_updater = up
    else:
        up = CrossAssetUpdater(monotonic_fn=FakeMonotonic())
        bot.cross_asset_updater = up  # sem snapshot => warming_up
    wp.process_window_snapshot(bot, [dict(t) for t in trades], trades[-1]["T"])
    ws = StateManager.instance().current
    return bot, ws, trades


def test_gw1_system_shape_real_end_to_end(frozen_state, monkeypatch, project_sleeps):
    spec = load_fixture("gw1_balanced.json")
    bot, ws, trades = _run_window(spec, monkeypatch,
                                  cross_values=spec["cross"]["values"])
    rec = bot.recorded
    assert rec is not None
    # MESMO objeto: nested order_flow real alimenta a WindowState
    of = rec["flow_metrics"]["order_flow"]
    assert of["buy_sell_ratio"]["buy_sell_ratio"] == pytest.approx(1.0)
    assert ws is not None and ws.window_number == 1
    assert ws.flow.buy_sell_ratio == pytest.approx(1.0)
    assert ws.flow.flow_imbalance == pytest.approx(
        of["flow_imbalance"], abs=1e-9)
    assert ws.flow.pressure_label == of["buy_sell_ratio"]["pressure"]
    buy, sell = expected_volumes(spec)
    assert ws.volume.buy == pytest.approx(buy)
    assert ws.volume.sell == pytest.approx(sell)
    # ml via snapshot real no pipeline
    ml = rec["pipeline"].get_final_features()["ml_features"]
    assert ml["cross_asset"]["btc_eth_corr_7d"] == 0.92
    assert ml["cross_asset"]["cross_asset_method"] == "shared_session_returns_v2"
    assert_contract_version(ml["cross_asset"], spec["expected_contract_version"],
                            "gw1 system")
    # pass-through P1 (sentinela, não correção)
    assert rec["macro_context"]["quant_model"]["prob_up"] == 0.62
    assert project_sleeps.calls == []


def test_gw4_system_missing_end_to_end(frozen_state, monkeypatch, project_sleeps):
    spec = load_fixture("gw4_external_unavailable.json")
    bot, ws, trades = _run_window(spec, monkeypatch, cross_values=None)
    assert ws.macro.dxy is None
    assert ws.derivatives.btc_long_short_ratio is None
    assert ws.derivatives.btc_open_interest is None
    ml = bot.recorded["pipeline"].get_final_features()["ml_features"]
    assert ml["cross_asset"] == {}
    assert ml["cross_asset_status"] == "warming_up"
    assert project_sleeps.calls == []


def test_gw6_system_cross_values_reach_pipeline(frozen_state, monkeypatch):
    import pandas as pd

    from common.ml_features import _map_correlations_to_features
    from market_analysis.cross_asset_correlations import shared_session_corr

    fx = load_fixture("gw6_weekend_holiday.json")
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
    dec = pd.Timestamp(fx["decision_ms"], unit="ms", tz="UTC").date()
    out = shared_session_corr(btc, tra, fx["target_returns"], decision_date=dec)
    assert out["n"] == 30
    values = {"status": "ok", "btc_dxy_corr_30d": out["corr"],
              "btc_dxy_corr_30d_n": out["n"],
              "correlation_method": "shared_session_returns_v2",
              "correlation_contract_version": 2,
              "btc_dxy_instrument": "DX-Y.NYB"}
    spec = load_fixture("gw1_balanced.json")
    bot, ws, trades = _run_window(spec, monkeypatch, cross_values=values)
    ml = bot.recorded["pipeline"].get_final_features()["ml_features"]
    assert ml["cross_asset"]["btc_dxy_corr_30d_n"] == 30
    assert ml["cross_asset"]["cross_asset_method"] == "shared_session_returns_v2"
    assert_contract_version(ml["cross_asset"], fx["expected_contract_version"],
                            "gw6 system")
