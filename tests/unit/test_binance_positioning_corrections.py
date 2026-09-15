# tests/unit/test_binance_positioning_corrections.py
# -*- coding: utf-8 -*-
"""
Correções paralelas Binance positioning (CFTC P-missão, itens 1-10).

Regressão: legado preservado (diff bruto, regimes, thresholds) + novos
contratos (divergence_pp, squeeze_side, oi_direction, quality, validações,
PARTIAL no payload, nomenclatura sem claims).
"""
import json

from fetchers.binance_positioning_fetcher import _safe_level, _safe_ratio
from institutional.crypto_cot import CryptoCOT, PositioningRegime
from market_orchestrator.ai.payload_builder_compact import _build_positioning


def _dict(**over):
    base = {
        "global_account_ratio": 1.10,
        "top_account_ratio": 1.15,
        "top_position_ratio": 1.20,
        "global_long_account_pct": 52.0,
        "open_interest": 100000.0,
        "oi_delta_1h": 0.01,
        "funding_rate": 0.0001,
        "is_available": True,
        "is_stale": False,
    }
    base.update(over)
    return base


def test_divergence_pp_canonical():
    res = CryptoCOT().analyze(_dict(global_account_ratio=1.0, top_position_ratio=2.0))
    # long-share: 50% vs 66.67% => +16.6667pp (diff bruta 1.0 perde a escala)
    assert res.top_position_vs_global == 1.0  # legado preservado
    assert abs(res.top_position_vs_global_pp - 16.6667) < 1e-3
    assert res.top_account_vs_global_pp is not None


def test_squeeze_side_structured():
    long_sq = CryptoCOT().analyze(_dict(global_account_ratio=2.5,
                                        top_position_ratio=2.4, funding_rate=0.0005))
    assert long_sq.regime == PositioningRegime.SQUEEZE_RISK
    assert long_sq.squeeze_side == "LONG"
    short_sq = CryptoCOT().analyze(_dict(global_account_ratio=0.4,
                                         top_position_ratio=0.4, funding_rate=-0.0005))
    assert short_sq.squeeze_side == "SHORT"
    assert CryptoCOT().analyze(_dict()).squeeze_side is None


def test_oi_direction_structured():
    up = CryptoCOT().analyze(_dict(oi_delta_1h=0.06))
    assert up.regime == PositioningRegime.OI_EXPANSION
    assert up.oi_direction == "EXPANSION"
    down = CryptoCOT().analyze(_dict(oi_delta_1h=-0.06))
    assert down.oi_direction == "CONTRACTION"
    assert CryptoCOT().analyze(_dict()).oi_direction is None


def test_funding_bool_and_range_rejected():
    assert CryptoCOT().analyze(_dict(funding_rate=True)).funding_rate is None
    r = CryptoCOT().analyze(_dict(funding_rate=5.0))
    assert r.funding_rate is None
    assert any("funding_rate" in w for w in r.quality["warnings"])


def test_negative_ratio_and_oi_rejected():
    r = CryptoCOT().analyze(_dict(global_account_ratio=-1.5))
    assert r.regime == PositioningRegime.PARTIAL  # essencial ausente
    assert r.global_account_ratio is None
    r2 = CryptoCOT().analyze(_dict(open_interest=-100.0))
    assert r2.open_interest is None


def test_pct_inconsistency_warns_without_regime_change():
    ok = CryptoCOT().analyze(_dict())  # 52% vs share(1.10)=52.38%: consistente
    assert not [w for w in ok.quality["warnings"] if "inconsistente" in w]
    bad = CryptoCOT().analyze(_dict(global_long_account_pct=10.0))
    assert any("inconsistente" in w for w in bad.quality["warnings"])
    assert bad.regime == ok.regime


def test_quality_and_source_as_of_propagated():
    r = CryptoCOT().analyze(_dict(source_as_of="2026-09-14T12:00:00+00:00"))
    assert r.source_as_of == "2026-09-14T12:00:00+00:00"
    assert "missing_fields" in r.quality and "warnings" in r.quality
    json.dumps(r.to_dict(), allow_nan=False)


def test_safe_ratio_level_helpers():
    assert _safe_ratio("2.5") == 2.5
    assert _safe_ratio(-1.0) is None
    assert _safe_ratio(True) is None
    assert _safe_level(0.0) == 0.0
    assert _safe_level(-5.0) is None


def test_partial_reaches_payload_as_status_only():
    ev = {"institutional_analytics": {"positioning": {
        "global_account_ratio": None, "top_position_ratio": 1.5,
        "regime": "PARTIAL", "is_available": True}}}
    out = _build_positioning(ev)
    assert out == {"rg": "PARTIAL"}  # sem números parciais como completos


def test_no_smart_money_retail_claims_in_reasons():
    r = CryptoCOT().analyze(_dict(global_account_ratio=2.5, top_position_ratio=2.4,
                                  funding_rate=0.0005))
    blob = " ".join(r.reasons).lower()
    assert "smart money" not in blob
    assert "varejo" not in blob
    assert "institucional" not in blob
