# tests/unit/test_crossasset_history_sizing.py
"""
F5-C7 — aquisição suficiente (sizing derivado, sem magic number).

Sem rede: _fetch_with_instrument/_fetch_binance_klines mockados.
Contrato: n = min(target, available); insuficiente => ausente, nunca inventar.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import market_analysis.cross_asset_correlations as ca


UTC = timezone.utc
DEC = datetime(2026, 9, 9, 1, 0, tzinfo=UTC)


def _frame(dates, closes):
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dates])
    return pd.DataFrame({"close": [float(c) for c in closes]}, index=idx)


def _cal(n, end="2026-09-08"):
    return [(pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=i)).date().isoformat()
            for i in range(n - 1, -1, -1)]


def _closes(n, start=100.0, slope=0.001, seed=0):
    rng = np.random.RandomState(seed)
    rets = slope + rng.randn(n) * 0.005
    px = [start]
    for r in rets:
        px.append(px[-1] * (1 + r))
    return px[1:], rets


def test_lookback_derived_not_magic():
    assert ca._fetch_calendar_lookback(30) == "57d"   # 32 sessões*7/5=45+12
    assert ca._fetch_calendar_lookback(90) == "141d"  # 92 sessões*7/5=129+12
    assert ca.TRADING_WEEK_RATIO == 7 / 5
    assert ca.HOLIDAY_MARGIN_DAYS == 12


def test_period_parser():
    se = ca._period_to_start_end("141d")
    assert se is not None
    start, end = se
    assert (pd.Timestamp(end) - pd.Timestamp(start)).days == 141
    assert ca._period_to_start_end("3mo") is None  # passthrough legado
    assert ca._period_to_start_end("90d") is not None


def _mock_three(monkeypatch, btc_f, dxy_f, ndx_f, seen):
    def _fake(name, period="90d", interval="1d"):
        seen.append((name, period))
        return {"BTC-USD": btc_f, "DXY": dxy_f, "NDX": ndx_f}[name]
    monkeypatch.setattr(ca, "_fetch_with_instrument", _fake)


def test_single_fetch_feeds_30_and_90(monkeypatch):
    dates = _cal(141)  # calendário cheio: BTC todos os dias
    biz = [d for d in dates if pd.Timestamp(d).weekday() < 5]
    bc, _ = _closes(len(dates), seed=1)
    dc, _ = _closes(len(biz), seed=2)
    nc, _ = _closes(len(biz), seed=3)
    seen = []
    _mock_three(monkeypatch,
                (_frame(dates, bc), "BTC-USD"),
                (_frame(biz, dc), "DX-Y.NYB"),
                (_frame(biz, nc), "QQQ"), seen)
    out = ca.get_btc_macro_correlations(DEC)
    assert out["status"] == "ok"
    # UM fetch por ativo, todos na profundidade derivada (sem rede dupla)
    assert seen == [("BTC-USD", "141d"), ("DXY", "141d"), ("NDX", "141d")]
    assert out["btc_dxy_corr_30d_n"] == 30
    assert out["btc_dxy_corr_90d_n"] == 90
    assert out["btc_ndx_corr_30d_n"] == 30
    assert out["correlation_method"] == "shared_session_returns_v2"


def test_90_with_weekends_holidays_reaches_target(monkeypatch):
    dates = _cal(141)
    biz = [d for d in dates if pd.Timestamp(d).weekday() < 5]
    # remove 5 "feriados" no meio
    hol = set(biz[40:45])
    biz2 = [d for d in biz if d not in hol]
    bc, _ = _closes(len(dates), seed=11)
    dc, _ = _closes(len(biz2), seed=12)
    seen = []
    _mock_three(monkeypatch,
                (_frame(dates, bc), "BTC-USD"),
                (_frame(biz2, dc), "DX-Y.NYB"),
                (_frame(biz2, dc), "QQQ"), seen)
    out = ca.get_btc_macro_correlations(DEC)
    assert out["btc_dxy_corr_90d_n"] == 90
    assert out["btc_dxy_corr_30d_n"] == 30


def test_insufficient_history_reports_real_n(monkeypatch):
    dates = _cal(30)
    biz = [d for d in dates if pd.Timestamp(d).weekday() < 5]
    bc, _ = _closes(len(dates), seed=21)
    dc, _ = _closes(len(biz), seed=22)
    seen = []
    _mock_three(monkeypatch,
                (_frame(dates, bc), "BTC-USD"),
                (_frame(biz, dc), "DX-Y.NYB"),
                (_frame(biz, dc), "QQQ"), seen)
    out = ca.get_btc_macro_correlations(DEC)
    # ~22 úteis - decisão => ~20 retornos: emite n real (< 30), sem inventar
    assert out["btc_dxy_corr_30d_n"] < 30
    assert out["btc_dxy_corr_30d_n"] >= ca.CORR_MIN_POINTS
    assert not pd.isna(out["btc_dxy_corr_30d"])
    assert out["btc_dxy_corr_90d_n"] < 90  # longe do alvo: n real, não 90


def test_severe_shortage_is_missing_not_zero(monkeypatch):
    dates = _cal(6)
    biz = [d for d in dates if pd.Timestamp(d).weekday() < 5]
    bc, _ = _closes(len(dates), seed=31)
    dc, _ = _closes(len(biz), seed=32)
    seen = []
    _mock_three(monkeypatch,
                (_frame(dates, bc), "BTC-USD"),
                (_frame(biz, dc), "DX-Y.NYB"),
                (_frame(biz, dc), "QQQ"), seen)
    out = ca.get_btc_macro_correlations(DEC)
    assert pd.isna(out["btc_dxy_corr_30d"])
    assert out["btc_dxy_corr_30d"] != 0.0


def test_long_holiday_gap_excluded_and_tail_only(monkeypatch):
    dates = _cal(141)
    biz = [d for d in dates if pd.Timestamp(d).weekday() < 5]
    gap = set(biz[50:65])  # 15 sessões seguidas fora (feriado prolongado/falha)
    biz2 = [d for d in biz if d not in gap]
    bc, _ = _closes(len(dates), seed=41)
    dc, _ = _closes(len(biz2), seed=42)
    seen = []
    _mock_three(monkeypatch,
                (_frame(dates, bc), "BTC-USD"),
                (_frame(biz2, dc), "DX-Y.NYB"),
                (_frame(biz2, dc), "QQQ"), seen)
    out = ca.get_btc_macro_correlations(DEC)
    # buraco de 15 sessões: 90 inalcançável em 141d -> n real (85), sem inventar
    assert out["btc_dxy_corr_90d_n"] == 85
    assert out["btc_dxy_corr_30d_n"] == 30
    # tail-only: helper sobre a história inteira == manual nos últimos 31 closes
    a = pd.Series(bc, index=pd.DatetimeIndex(dates, tz="UTC"))
    b = pd.Series(dc, index=pd.DatetimeIndex(biz2, tz="UTC"))
    got = ca.shared_session_corr(a, b, 30, decision_date=DEC.date())
    ma = ca._daily_by_date(a)
    mb = ca._daily_by_date(b)
    shared = sorted(set(ma) & set(mb))
    shared = [d for d in shared if d < DEC.date()][-31:]
    ra = ca._log_returns(pd.Series([ma[d] for d in shared]).reset_index(drop=True))
    rb = ca._log_returns(pd.Series([mb[d] for d in shared]).reset_index(drop=True))
    assert got["corr"] == pytest.approx(round(float(ra.corr(rb)), 4))


def test_no_future_values_with_deep_history(monkeypatch):
    dates = _cal(141)
    biz = [d for d in dates if pd.Timestamp(d).weekday() < 5]
    bc, _ = _closes(len(dates), seed=51)
    dc, _ = _closes(len(biz), seed=52)
    seen = []
    _mock_three(monkeypatch,
                (_frame(dates, bc), "BTC-USD"),
                (_frame(biz, dc), "DX-Y.NYB"),
                (_frame(biz, dc), "QQQ"), seen)
    out = ca.get_btc_macro_correlations(DEC)
    assert out["status"] == "ok"
    r = ca.shared_session_corr(
        pd.Series(bc, index=pd.DatetimeIndex(dates, tz="UTC")),
        pd.Series(dc, index=pd.DatetimeIndex(biz, tz="UTC")),
        90, decision_date=DEC.date())
    for a, b in r["pairs"]:
        assert a < DEC.date().isoformat() and b < DEC.date().isoformat()
    assert r["n"] == 90


def _klines(n, start="2026-08-01 00:00+00:00", seed=0):
    rng = np.random.RandomState(seed)
    rets = 0.0005 + rng.randn(n) * 0.002
    px = [100.0]
    for r in rets:
        px.append(px[-1] * (1 + r))
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    ct = [(t + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)) for t in idx]
    return pd.DataFrame(
        {"close": px[1:], "close_time": [int(t.timestamp() * 1000) for t in ct]},
        index=idx)


def test_klines_single_fetch_reaches_720(monkeypatch):
    seen = {}

    def _fake(sym, interval="1h", limit=720):
        seen[sym] = limit
        return _klines(limit, seed=7 if sym == "BTCUSDT" else 8)

    monkeypatch.setattr(ca, "_fetch_binance_klines", _fake)
    dec = datetime(2026, 9, 9, 1, 0, tzinfo=UTC)
    out = ca.get_btc_eth_correlations(dec)
    assert out["status"] == "ok"
    # UM fetch por símbolo, limite derivado (720 + close + aberto + folga)
    assert seen == {"BTCUSDT": 724, "ETHUSDT": 724}
    assert out["btc_eth_corr_7d_n"] == 168
    assert out["btc_eth_corr_30d_n"] == 720
