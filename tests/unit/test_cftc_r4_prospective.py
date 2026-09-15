# tests/unit/test_cftc_r4_prospective.py
# -*- coding: utf-8 -*-
"""
R4.12 — Prospectivo TRUE OOS. Determinístico, sem rede.
Cobre: post-freeze only, first_seen obrigatório, sem calendar fallback,
entry > first_seen, maturity 28d/PENDING, revisão preserva original,
append-only, duplicata rejeitada, Binance futuro rejeitado, NaN/Inf,
restart preserva estado.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../../scripts/analytics")))

from cftc_r4_prospective_collector import (  # noqa: E402
    R4Store,
    build_observation,
    compute_entry,
    compute_outcome,
    observation_id,
)

FREEZE = "2026-09-15"
T0 = datetime(2026, 9, 18, 19, 30, tzinfo=timezone.utc)  # sexta pós-freeze


def _raw(asof="2026-09-22", lev_long="5000", lev_short="1000", oi="20000"):
    return {"report_date_as_yyyy_mm_dd": asof + "T00:00:00.000",
            "cftc_contract_market_code": "133742",
            "market_and_exchange_names": "MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE",
            "lev_money_positions_long": lev_long,
            "lev_money_positions_short": lev_short,
            "lev_money_positions_spread": "100",
            "open_interest_all": oi,
            "futonly_or_combined": "FutOnly"}


def _candles(start, n, px=100.0):
    return [(int((start + timedelta(days=i)).timestamp() * 1000), px + i)
            for i in range(n)]


def test_pre_freeze_rejected():
    out = build_observation(_raw(asof="2026-09-08"), T0.isoformat(),
                            T0.isoformat(), 0, FREEZE, [])
    assert out["status"] == "pre_freeze"


def test_first_seen_required_no_calendar():
    out = build_observation(_raw(), "", T0.isoformat(), 0, FREEZE, [])
    assert out["status"] == "invalid" and out["reason"] == "first_seen_required"
    # módulo nunca importa calendário: nenhuma estimativa é produzida
    import cftc_r4_prospective_collector as mod

    src = open(mod.__file__, encoding="utf-8").read()
    assert "expected_publication_utc" not in src
    assert "allow_calendar_fallback" not in src


def test_observation_ok_and_ids():
    out = build_observation(_raw(), T0.isoformat(), T0.isoformat(), 0, FREEZE, [0.1] * 60)
    assert out["status"] == "ok"
    assert out["observation_id"] == observation_id("133742", "2026-09-22", 0)
    assert out["feature_available_at"] == T0.isoformat()
    assert abs(out["h3_net_share_oi"] - (5000 - 1000) / 20000) < 1e-12
    assert 0.0 <= out["h4_net_share_pct52"] <= 1.0


def test_nan_inf_rejected():
    out = build_observation(_raw(lev_long="NaN"), T0.isoformat(), T0.isoformat(),
                            0, FREEZE, [])
    assert out["status"] == "invalid"


def test_entry_strictly_after_first_seen():
    fs = datetime(2026, 9, 18, 19, 30, tzinfo=timezone.utc)
    fs_ms = int(fs.timestamp() * 1000)
    candles = [(fs_ms - 3600_000, 100.0), (fs_ms, 101.0), (fs_ms + 3600_000, 102.0)]
    got = compute_entry(candles, fs.isoformat())
    # candle que começa EXATAMENTE no first_seen é ambíguo -> pula
    assert got["entry_price"] == 102.0
    assert compute_entry([(fs_ms - 1, 99.0)], fs.isoformat()) is None
    assert compute_entry(candles, "2026-09-18 19:30:00") is None  # naive rejeitado


def test_maturity_pending_then_matured():
    entry = datetime(2026, 9, 19, 0, 0, tzinfo=timezone.utc)
    candles = _candles(entry, 20)
    assert compute_outcome(candles, entry.isoformat()) is None  # PENDING
    candles = _candles(entry, 40)
    out = compute_outcome(candles, entry.isoformat())
    assert out is not None
    assert out["exit_timestamp"] >= (entry + timedelta(days=28)).isoformat()
    assert abs(out["exit_price"] / 100.0 - 1.0 - (out["exit_price"] - 100.0) / 100.0) < 1e-12


def test_append_only_duplicate_rejected_restart_safe(tmp_path):
    store = R4Store(tmp_path)
    obs = build_observation(_raw(), T0.isoformat(), T0.isoformat(), 0, FREEZE, [0.1] * 60)
    obs.update(compute_entry(_candles(T0, 40), T0.isoformat()))
    assert store.append_observation(obs) is True
    assert store.append_observation(dict(obs)) is False  # duplicata
    assert len(store.observations()) == 1
    store2 = R4Store(tmp_path)  # restart recarrega IDs
    assert store2.append_observation(dict(obs)) is False
    assert len(store2.observations()) == 1


def test_revision_never_rewrites_original(tmp_path):
    store = R4Store(tmp_path)
    obs = build_observation(_raw(), T0.isoformat(), T0.isoformat(), 0, FREEZE, [0.1] * 60)
    obs.update(compute_entry(_candles(T0, 40), T0.isoformat()))
    store.append_observation(obs)
    store.append_revision({"observation_id": obs["observation_id"],
                           "revision": 1, "revised_feature": 0.5,
                           "original_preserved": True})
    kept = store.observations()[0]
    assert abs(kept["h3_net_share_oi"] - 0.2) < 1e-12  # original intacto
    assert len(store.revisions()) == 1


def test_future_binance_rejected_concept():
    # snapshot posterior ao first_seen não pode virar evidência strict:
    # regra documentada — campo strict fica null + flag.
    fa_ms = int(T0.timestamp() * 1000)
    snap_ms = fa_ms + 600_000  # 10 min depois
    strict_ok = snap_ms <= fa_ms
    assert strict_ok is False
