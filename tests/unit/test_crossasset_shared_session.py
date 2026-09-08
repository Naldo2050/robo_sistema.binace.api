# tests/unit/test_crossasset_shared_session.py
"""
F5-C fix — shared-session returns (contrato temporal v2).

Sem rede: fetchers mockados via monkeypatch. Produção em
market_analysis/cross_asset_correlations.py (shared_session_corr,
intraday_join_corr e get_btc_*_correlations com decision_time + metadata).
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import market_analysis.cross_asset_correlations as ca


UTC = timezone.utc
DEC = datetime(2026, 9, 9, 1, 0, tzinfo=UTC)  # decisão 09-09: exclui dia 09-09


def _daily(dates, closes, tz="UTC"):
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz=tz) for d in dates])
    return pd.Series([float(c) for c in closes], index=idx)


def _closes_from_rets(start, rets):
    px = [float(start)]
    for r in rets:
        px.append(px[-1] * (1 + float(r)))
    return px[1:]


def _biz():
    return ["2026-08-31", "2026-09-01", "2026-09-02",
            "2026-09-03", "2026-09-04", "2026-09-08"]


def _biz15():
    # 15 úteis p/ testes de VALOR (14 retornos >= CORR_MIN_POINTS).
    return [d.date().isoformat()
            for d in pd.bdate_range("2026-08-19", periods=15, tz="UTC")]


def test_weekend_naturally_excluded():
    biz = _biz15()
    cal = ["2026-08-15", "2026-08-16"] + biz  # sáb/dom antes do período
    rets = list(np.linspace(-0.02, 0.02, len(biz)))
    btc = _daily(cal, [100.0, 100.0] + _closes_from_rets(100.0, rets))
    tra = _daily(biz, _closes_from_rets(200.0, rets))
    out = ca.shared_session_corr(btc, tra, 30, decision_date=DEC.date())
    assert out["n"] == len(biz) - 1
    assert out["corr"] == pytest.approx(1.0, abs=1e-6)
    assert out["first"] == biz[0] and out["last"] == biz[-1]


def test_holiday_midweek_excluded():
    full = ["2026-09-07", "2026-09-08", "2026-09-09",
            "2026-09-10", "2026-09-11"]  # 09 = feriado TradFi
    open_days = [d for d in full if d != "2026-09-09"]
    rets = [0.01, -0.02, 0.015, 0.005]
    btc = _daily(full, [100.0] + _closes_from_rets(100.0, rets))
    tra = _daily(open_days, [200.0] + _closes_from_rets(200.0, [rets[0], rets[2], rets[3]]))
    # 4 dias comuns => 3 retornos < 10 => NaN
    out = ca.shared_session_corr(btc, tra, 30,
                                 decision_date=datetime(2026, 9, 12, tzinfo=UTC).date())
    assert out["n"] == 3
    assert pd.isna(out["corr"])  # insuficiente => NaN, nunca 0.0
    # com janela maior de dados o feriado some da conta sem quebrar a matemática
    many = [f"2026-08-{d:02d}" for d in range(10, 32)]
    rets_many = list(np.linspace(-0.01, 0.01, len(many)))
    b2 = _daily(many, _closes_from_rets(100.0, rets_many))
    t2 = _daily([d for d in many if d != "2026-08-19"],
                _closes_from_rets(200.0, [r for d, r in zip(many, rets_many)
                                          if d != "2026-08-19"]))
    out2 = ca.shared_session_corr(b2, t2, 30,
                                  decision_date=datetime(2026, 9, 1, tzinfo=UTC).date())
    assert out2["n"] == len(many) - 2  # 22 closes - feriado => 21 retornos


def test_shared_session_monday_endpoints():
    # Segunda usa sex->seg nos DOIS lados (nunca dom->seg no BTC).
    btc = _daily(["2026-09-04", "2026-09-06", "2026-09-07"],
                 [100.0, 102.0, 103.0])
    tra = _daily(["2026-09-04", "2026-09-07"], [200.0, 206.0])
    out = ca.shared_session_corr(
        btc, tra, 30, decision_date=datetime(2026, 9, 8, tzinfo=UTC).date())
    assert out["n"] == 1  # 2 closes => 1 retorno (insuficiente p/ Pearson)
    assert pd.isna(out["corr"])
    # com 11+ sessões perfeitamente correlacionadas => +1 exato
    days = [f"2026-08-{d:02d}" for d in range(3, 18)]
    rets = list(np.linspace(-0.02, 0.02, len(days)))
    b3 = _daily(days, _closes_from_rets(100.0, rets))
    t3 = _daily(days, _closes_from_rets(50.0, rets))
    out3 = ca.shared_session_corr(
        b3, t3, 30, decision_date=datetime(2026, 8, 20, tzinfo=UTC).date())
    assert out3["corr"] == pytest.approx(1.0, abs=1e-9)


def test_open_daily_bar_excluded_by_decision():
    biz = _biz()
    rets = list(np.linspace(-0.01, 0.01, len(biz)))
    btc = _daily(biz, _closes_from_rets(100.0, rets))
    tra = _daily(biz, _closes_from_rets(200.0, rets))
    # decisão no meio do dia 09-08: barra de 09-08 proibida nos dois lados
    out = ca.shared_session_corr(
        btc, tra, 30, decision_date=datetime(2026, 9, 8, 12, 0, tzinfo=UTC).date())
    assert out["last"] == "2026-09-04"
    assert out["n"] == 4


def test_decision_before_and_after_close():
    biz = _biz()
    rets = list(np.linspace(-0.01, 0.01, len(biz)))
    btc = _daily(biz, _closes_from_rets(100.0, rets))
    tra = _daily(biz, _closes_from_rets(200.0, rets))
    before = ca.shared_session_corr(
        btc, tra, 30, decision_date=datetime(2026, 9, 4, tzinfo=UTC).date())
    assert before["last"] == "2026-09-03"
    assert before["n"] == 3  # 4 closes => 3 retornos
    after = ca.shared_session_corr(
        btc, tra, 30, decision_date=datetime(2026, 9, 9, tzinfo=UTC).date())
    assert after["last"] == "2026-09-08"
    assert after["n"] == 5  # +09-04 e +09-08
    assert after["n"] == before["n"] + 2


def test_tz_naive_and_aware_mix():
    biz = _biz15()
    rets = list(np.linspace(-0.01, 0.01, len(biz)))
    a = _daily(biz, _closes_from_rets(100.0, rets), tz="UTC")
    naive_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in biz])  # naive
    b = pd.Series(_closes_from_rets(200.0, rets), index=naive_idx)
    out = ca.shared_session_corr(a, b, 30, decision_date=DEC.date())
    assert out["n"] == len(biz) - 1
    assert out["corr"] == pytest.approx(1.0, abs=1e-9)


def test_duplicates_keep_last_and_unordered():
    biz = _biz15()
    rets = list(np.linspace(-0.01, 0.01, len(biz)))
    closes = _closes_from_rets(200.0, rets)
    dup_idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in biz]).insert(2, pd.Timestamp(biz[2], tz="UTC"))
    dup_vals = closes[:2] + [999.0] + closes[2:]
    b = pd.Series(dup_vals, index=dup_idx)
    a = _daily(biz, _closes_from_rets(100.0, rets))
    out = ca.shared_session_corr(a, b, 30, decision_date=DEC.date())
    # duplicata conta 1x (keep-last = valor correto, não 999.0)
    assert out["n"] == len(biz) - 1
    assert out["corr"] == pytest.approx(1.0, abs=1e-9)
    # fora de ordem: mesmo resultado
    shuf = b.sample(frac=1.0, random_state=7)
    out2 = ca.shared_session_corr(a, shuf, 30, decision_date=DEC.date())
    assert out2["corr"] == pytest.approx(out["corr"], abs=1e-12)


def test_insufficient_is_nan_never_zero():
    btc = _daily(["2026-09-01", "2026-09-02"], [100.0, 101.0])
    tra = _daily(["2026-09-01", "2026-09-02"], [200.0, 202.0])
    out = ca.shared_session_corr(btc, tra, 30, decision_date=DEC.date())
    assert out["n"] == 1
    assert pd.isna(out["corr"])
    assert out["corr"] != 0.0


def _klines(prices, start="2026-09-08 00:00+00:00"):
    idx = pd.date_range(start, periods=len(prices), freq="h", tz="UTC")
    ct = [(t + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)) for t in idx]
    return pd.DataFrame({"close": list(prices), "close_time": [int(t.timestamp() * 1000) for t in ct]}, index=idx)


def test_btc_eth_gap_and_open_candle_excluded():
    px_b = [100.0 + i * 0.1 for i in range(30)]
    px_e = [50.0 + i * 0.05 for i in range(30)]
    btc = _klines(px_b)
    eth = _klines(px_e).drop(pd.Timestamp("2026-09-08 05:00+00:00"))  # gap
    # decisão 09-10: todos os 30 candles fechados; só o gap de 05:00 exclui 1
    dec_ms = int(pd.Timestamp("2026-09-10 00:00+00:00").timestamp() * 1000)
    out = ca.intraday_join_corr(btc, eth, 30, decision_ms=dec_ms)
    assert out["n"] == 28  # 29 timestamps compartilhados => 28 retornos
    assert not pd.isna(out["corr"])


def test_btc_eth_open_candle_excluded():
    px = [100.0 + i for i in range(10)]
    btc = _klines(px)
    eth = _klines([p / 2 for p in px])
    # decisão no meio do último candle (09:30): último open 09:00 ainda aberto
    dec_ms = int(pd.Timestamp("2026-09-08 09:30+00:00").timestamp() * 1000)
    out = ca.intraday_join_corr(btc, eth, 30, decision_ms=dec_ms)
    assert out["last"] == "2026-09-08T08:00:00+00:00"
    assert out["n"] == 8  # 9 closes => 8 retornos


def test_metadata_method_n_instrument(monkeypatch):
    btc = _daily(_biz(), _closes_from_rets(100.0, np.linspace(-0.01, 0.01, 6)))
    dxy = _daily(_biz(), _closes_from_rets(99.0, np.linspace(-0.005, 0.005, 6)))
    ndx = _daily(_biz(), _closes_from_rets(700.0, np.linspace(-0.01, 0.01, 6)))
    monkeypatch.setattr(ca, "_fetch_with_instrument", lambda name, period="90d", interval="1d", stop_event=None: {
        "BTC-USD": (btc.to_frame("close"), "BTC-USD"),
        "DXY": (dxy.to_frame("close"), "DX-Y.NYB"),
        "NDX": (ndx.to_frame("close"), "QQQ"),
    }[name])
    out = ca.get_btc_macro_correlations(DEC)
    assert out["status"] == "ok"
    assert out["correlation_method"] == "shared_session_returns_v2"
    assert out["correlation_contract_version"] == 2
    assert out["btc_dxy_instrument"] == "DX-Y.NYB"
    assert out["nasdaq_instrument"] == "QQQ"
    assert out["nasdaq_role"] == "nasdaq_proxy"
    assert isinstance(out["btc_dxy_corr_30d_n"], int)
    assert out["btc_ndx_corr_30d_n"] >= 0


def test_no_pair_uses_availability_after_decision(monkeypatch):
    # golden: nenhum par usa informação com availability > decision_time
    btc = _daily(_biz(), _closes_from_rets(100.0, np.linspace(-0.01, 0.01, 6)))
    dxy = _daily(_biz(), _closes_from_rets(99.0, np.linspace(-0.005, 0.005, 6)))
    ndx = _daily(_biz(), _closes_from_rets(700.0, np.linspace(-0.01, 0.01, 6)))
    monkeypatch.setattr(ca, "_fetch_with_instrument", lambda name, period="90d", interval="1d", stop_event=None: {
        "BTC-USD": (btc.to_frame("close"), "BTC-USD"),
        "DXY": (dxy.to_frame("close"), "DX-Y.NYB"),
        "NDX": (ndx.to_frame("close"), "QQQ"),
    }[name])
    out = ca.get_btc_macro_correlations(DEC)
    dec_d = DEC.date().isoformat()
    for key in ("btc_dxy_corr_30d", "btc_dxy_corr_90d", "btc_ndx_corr_30d"):
        assert key in out
    # reconfere via helper: pares sempre < decision
    r = ca.shared_session_corr(btc, dxy, 30, decision_date=DEC.date())
    for a, b in r["pairs"]:
        assert a < dec_d and b < dec_d
