# tests/unit/test_cftc_r2_information.py
# -*- coding: utf-8 -*-
"""
R2.16 — Estudo de informação futura. Determinístico, sem rede.
Séries sintéticas: 60 terças de COT + 500 dias de preço.
"""
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../../scripts/analytics")))

import cftc_r2_forward_information as r2


def _tuesdays(n, start=date(2024, 1, 2)):
    return [(start + timedelta(days=7 * i)).isoformat() for i in range(n)]


def _feats(n=60, code="133741", slope=0.0):
    rows = []
    for i, asof in enumerate(_tuesdays(n)):
        rows.append({"report_as_of_date": asof,
                     "leveraged_net_share_oi": 0.1 + slope * i,
                     "leveraged_net_change_1w": 0.01,
                     "leveraged_net_share_pct52": (i / 59) if n == 60 else 0.5})
    return pd.DataFrame(rows)


def _prices(n=500, start=date(2023, 12, 1), drift=0.0):
    dates, closes = [], []
    px = 40000.0
    for i in range(n):
        dates.append((start + timedelta(days=i)).isoformat())
        px *= (1.0 + drift)
        closes.append(px)
    return pd.DataFrame({"date": dates, "close": closes})


HOLIDAYS = set()


def test_availability_alignment_abc():
    asof = date(2024, 1, 2)  # terça
    scen = r2.availability_scenarios(asof, HOLIDAYS)
    a_at, a_est = scen["A"]
    assert a_at.date().isoformat() == "2024-01-05"  # sexta
    assert a_est is False
    assert scen["B"][0].date().isoformat() == "2024-01-08"
    assert scen["C"][0].date().isoformat() == "2024-01-09"
    # sexta publica 15:30 ET = 20:30 UTC (EST, janeiro)
    assert (a_at.hour, a_at.minute) == (20, 30)


def test_holiday_marks_a_estimated():
    asof = date(2024, 7, 2)  # semana do 04/07 (quinta)
    hol = {d.isoformat() for d in r2.us_federal_holidays(2024)}
    assert "2024-07-04" in hol
    scen = r2.availability_scenarios(asof, hol)
    # sexta 05/07 não é feriado, mas existe feriado na semana? regra: só sexta/asof
    assert scen["A"][0].date().isoformat() == "2024-07-05"
    # asof segunda (deslocado) => estimado
    scen2 = r2.availability_scenarios(date(2024, 7, 1), hol)
    assert scen2["A"][1] is True


def test_entry_after_availability_no_leak():
    feats = _feats(4)
    prices = _prices()
    rows = r2.build_rows(feats, prices, 7, HOLIDAYS)
    assert not rows.empty
    for _, r in rows.iterrows():
        assert r["entry_date"] > r["assumed_available_at"][:10]
        assert r["exit_date"] > r["entry_date"]
        assert r["asof"] <= r["assumed_available_at"][:10]
    # sem merge_asof forward: entry é o primeiro close APÓS availability
    first = rows[(rows["asof"] == _tuesdays(4)[0]) & (rows["availability"] == "A")].iloc[0]
    assert first["entry_date"] == "2024-01-06"  # sábado após sexta 05/01


def test_forward_returns_math():
    feats = _feats(2)
    dates = ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09"]
    prices = pd.DataFrame({"date": dates, "close": [100.0, 110.0, 121.0, 130.0, 140.0]})
    rows = r2.build_rows(feats, prices, 1, HOLIDAYS)
    one = rows[(rows["asof"] == _tuesdays(2)[0]) & (rows["availability"] == "A")]
    assert len(one) == 1
    assert one.iloc[0]["entry_date"] == "2024-01-06"
    assert abs(one.iloc[0]["forward_return"] - (121.0 / 110.0 - 1.0)) < 1e-12


def test_quintiles_known_distribution():
    x = np.arange(100, dtype=float)
    y = np.arange(100, dtype=float)  # monotônico
    q = r2.quintile_stats(x, y)
    assert q["n"] == 100 and len(q["q"]) == 5
    assert q["q5_q1"] > 0
    assert [b["n"] for b in q["q"]] == [20, 20, 20, 20, 20]
    assert r2.quintile_stats(x[:10], y[:10])["q"] == []  # N insuficiente


def test_bootstrap_deterministic_and_blocked():
    rng = np.random.default_rng(7)
    x = rng.normal(size=120)
    y = 0.5 * x + rng.normal(size=120)
    ci1 = r2.block_bootstrap_ci(x, y, r2._pearson_only, block=8)
    ci2 = r2.block_bootstrap_ci(x, y, r2._pearson_only, block=8)
    assert ci1 == ci2 and ci1["lo"] > 0  # relação real detectada
    assert r2.block_bootstrap_ci(x[:10], y[:10], r2._pearson_only, block=8) is None


def test_fdr_bh():
    p = np.array([0.001, 0.01, 0.05, 0.5, np.nan])
    q = r2.bh_fdr(p)
    assert q[0] < q[1] < q[2] < q[3]
    assert np.isnan(q[4]) and q[0] <= 0.005


def test_subperiod_split_contracts_separate():
    feats = _feats(60)
    prices = _prices()
    ev_a = r2.analyze_combo(feats, prices, HOLIDAYS, "133741", "leveraged",
                            "net_share_oi", 7)
    ev_b = r2.analyze_combo(feats, prices, HOLIDAYS, "133742", "leveraged",
                            "net_share_oi", 7)
    assert ev_a["contract"] != ev_b["contract"]
    assert ev_a["n"] == ev_b["n"] == 60
    assert set(ev_a["subperiod_signs"]) <= {"2018-2020", "2021-2023", "2024-2026"}


def test_nan_inf_insufficient():
    x = np.array([1.0, np.nan, np.inf, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    st = r2.ic_stats(x, y)
    assert st["n"] == 3  # só finitos
    assert r2.ic_stats(np.array([1.0, 1.0]), np.array([1.0, 2.0]))["pearson"] is None
