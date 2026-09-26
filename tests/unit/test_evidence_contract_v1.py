# tests/unit/test_evidence_contract_v1.py — P1-A: Evidence Contract v1.0.0.
#
# Contrato puro, sem fiação produtiva: nenhuma confluence, payload, prompt,
# risk ou execução é tocada aqui. counts_as_vote default FALSE sempre.

import json

import pytest

from institutional.base import Side, Signal, SignalStrength
from institutional.evidence import (
    FEATURE_CONTRACT_VERSION,
    Evidence,
    EvidenceCalibration,
    EvidenceDirection,
    EvidenceFamily,
    EvidenceType,
    EvidenceValidity,
    evidence_from_signal,
)


def _signal(**kw):
    base = {"timestamp": 1700000000000, "signal_type": "test",
            "direction": Side.BUY, "strength": SignalStrength.STRONG,
            "price": 65000.0, "confidence": 0.9, "source": "cvd_analyzer"}
    base.update(kw)
    return Signal(**base)


# ── versão / defaults ────────────────────────────────────────────────────────

def test_version_is_1_0_0():
    assert FEATURE_CONTRACT_VERSION == "1.0.0"
    assert Evidence(source="x").to_dict()["feature_contract_version"] == "1.0.0"


def test_defaults_never_vote_nor_claim():
    e = Evidence(source="x")
    d = e.to_dict()
    assert d["counts_as_vote"] is False
    assert d["family"] == "unknown"
    assert d["evidence_type"] == "unknown"
    assert d["direction"] == "unknown"
    assert d["validity"] == "unknown"
    assert d["calibration"] == "unknown"
    assert d["value"] is None
    assert d["observed_at_ms"] is None
    assert d["horizon_ms"] is None
    assert d["derived_from"] == []


def test_empty_source_rejected():
    for bad in ("", "   ", None, 123):
        with pytest.raises(ValueError):
            Evidence(source=bad)


def test_unknown_enum_string_rejected_not_silenced():
    with pytest.raises(ValueError):
        Evidence(source="x", family="FAMILIA_INEXISTENTE")
    with pytest.raises(ValueError):
        Evidence(source="x", validity="quase_valid")


def test_enum_members_and_exact_strings_accepted():
    e = Evidence(source="x", family="executed_flow",
                 evidence_type=EvidenceType.CONTINUOUS_TRADES,
                 direction="bearish", validity="partial",
                 calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC)
    assert e.family is EvidenceFamily.EXECUTED_FLOW
    assert e.direction is EvidenceDirection.BEARISH


# ── serialização ─────────────────────────────────────────────────────────────

def test_enum_serialization_and_determinism():
    e = Evidence(source="s", family=EvidenceFamily.DERIVATIVES,
                 evidence_type=EvidenceType.SLOW_CONTEXT,
                 direction=EvidenceDirection.NEUTRAL,
                 value=1.5, observed_at_ms=1700000000000, horizon_ms=3600000,
                 provenance={"symbol": "BTCUSDT", "exchange": "binance",
                             "stream": "aggTrades"},
                 validity=EvidenceValidity.PARTIAL, reason="WARMING_UP",
                 calibration=EvidenceCalibration.UNCALIBRATED_HEURISTIC,
                 counts_as_vote=False,
                 derived_from=("aggtrades.buy_notional",),
                 metadata={"note": "explicabilidade"})
    d1 = e.to_dict()
    d2 = Evidence(source="s", family="derivatives", evidence_type="slow_context",
                  direction="neutral", value=1.5, observed_at_ms=1700000000000,
                  horizon_ms=3600000,
                  provenance={"stream": "aggTrades", "symbol": "BTCUSDT",
                              "exchange": "binance"},
                  validity="partial", reason="WARMING_UP",
                  calibration="uncalibrated_heuristic", counts_as_vote=False,
                  derived_from=["aggtrades.buy_notional"],
                  metadata={"note": "explicabilidade"}).to_dict()
    assert d1 == d2  # determinístico (ordens de dict irrelevantes)
    assert all(isinstance(v, str) for v in (
        d1["family"], d1["evidence_type"], d1["direction"], d1["validity"],
        d1["calibration"]))


def test_provenance_and_derived_from_round_trip():
    e = Evidence(source="s",
                 provenance={"exchange": "binance", "stream": "depth",
                             "symbol": "BTCUSDT"},
                 derived_from=("orderbook.snapshot.bid_depth",
                               "orderbook.snapshot.ask_depth"))
    back = Evidence.from_dict(e.to_dict())
    assert back.provenance == e.provenance
    assert back.derived_from == e.derived_from
    assert back.source == "s"


def test_observed_at_horizon_none_accepted():
    e = Evidence(source="s")
    assert e.observed_at_ms is None and e.horizon_ms is None


def test_from_dict_rejects_missing_or_wrong_version():
    good = Evidence(source="s").to_dict()
    with pytest.raises(ValueError):
        Evidence.from_dict({k: v for k, v in good.items()
                            if k != "feature_contract_version"})
    tampered = dict(good, feature_contract_version="0.9.0")
    with pytest.raises(ValueError):
        Evidence.from_dict(tampered)
    with pytest.raises(ValueError):
        Evidence.from_dict("nao-dict")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_value_becomes_null_invalid(bad):
    e = Evidence(source="s", value=bad, validity="partial",
                 reason="alguma_origem")
    assert e.value is None
    assert e.validity is EvidenceValidity.INVALID
    assert "NONFINITE_VALUE" in (e.reason or "")
    assert "alguma_origem" in (e.reason or "")  # reason preservado


def test_json_rfc8259():
    e = Evidence(source="s", value=float("nan"),
                 metadata={"x": float("inf"), "y": [1.0, float("nan")]},
                 provenance={"p": float("-inf")})
    text = json.dumps(e.to_dict(), allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    back = json.loads(text)
    assert back["value"] is None
    assert back["metadata"] == {"x": None, "y": [1.0, None]}


# ── P0 representável (sem mudar produtores) ──────────────────────────────────

def test_p0_absorption_bearish_non_voting():
    e = Evidence(source="whale_score.absorption",
                 family="executed_flow", evidence_type="continuous_trades",
                 direction="bearish", value=None,
                 validity="partial", reason="NON_VOTING_UNVALIDATED_MAGNITUDE",
                 calibration="unknown", counts_as_vote=False,
                 metadata={"buyer_strength": 8.6, "seller_exhaustion": 1.4,
                           "label": "Absorção de Compra"})
    d = e.to_dict()
    assert d["direction"] == "bearish" and d["counts_as_vote"] is False


def test_p0_flow_5m_warming_partial():
    e = Evidence(source="flow_analyzer.imbalance_5m",
                 family="executed_flow", evidence_type="continuous_trades",
                 direction="unknown", value=8.416,
                 validity="partial", reason="WARMING_UP")
    assert e.to_dict()["validity"] == "partial"


def test_p0_invalid_imbalance_null():
    e = Evidence(source="flow_analyzer.imbalance_5m", value=None,
                 validity="invalid", reason="INVARIANT_VIOLATION")
    assert e.to_dict() == {**e.to_dict(), "value": None}
    assert e.value is None


def test_p0_iceberg_unsupported():
    e = Evidence(source="orderbook_analyzer.iceberg",
                 family="orderbook_snapshot",
                 evidence_type="point_in_time_l2",
                 direction="unknown", value=1.0,
                 validity="unsupported", reason="CONTINUOUS_L2_UNAVAILABLE",
                 counts_as_vote=False)
    d = e.to_dict()
    assert d["validity"] == "unsupported" and d["counts_as_vote"] is False


def test_p0_regime_heuristic_non_voting_with_lineage():
    e = Evidence(source="enricher.regime",
                 evidence_type="derived", direction="unknown", value=1.0,
                 validity="partial",
                 calibration="uncalibrated_heuristic", counts_as_vote=False,
                 derived_from=("flow.trend", "profile.shape",
                               "orderbook.imbalance", "whale.score"))
    d = e.to_dict()
    assert d["counts_as_vote"] is False
    assert d["derived_from"] == ["flow.trend", "profile.shape",
                                 "orderbook.imbalance", "whale.score"]


# ── adapter legado: sem provenance/family/type inventados ────────────────────

def test_adapter_no_invented_provenance_family_type():
    sig = _signal(direction=Side.BUY, source="whale_detector")
    e = evidence_from_signal(sig)
    assert e.source == "whale_detector"
    assert e.family is EvidenceFamily.UNKNOWN
    assert e.evidence_type is EvidenceType.UNKNOWN
    assert e.provenance == {}
    assert e.direction is EvidenceDirection.UNKNOWN  # BUY não vira BULLISH
    assert e.validity is EvidenceValidity.UNKNOWN  # nunca promovido a VALID
    assert e.calibration is EvidenceCalibration.UNKNOWN
    assert e.counts_as_vote is False
    assert e.derived_from == ()
    assert e.observed_at_ms == 1700000000000  # dado direto, não inferência


def test_adapter_compra_label_never_infers_direction():
    sig = _signal(signal_type="Absorção de Compra", direction=Side.SELL,
                  source="absorption_detector")
    e = evidence_from_signal(sig)
    assert e.direction is EvidenceDirection.UNKNOWN


def test_adapter_empty_source_rejected_not_invented():
    with pytest.raises(ValueError):
        evidence_from_signal(_signal(source=""))
    with pytest.raises(ValueError):
        evidence_from_signal("nao-signal")


def test_adapter_iceberg_source_implies_no_capability():
    # source="iceberg" NÃO implica suporte: adapter não lê capability nenhuma.
    e = evidence_from_signal(_signal(source="iceberg_detector"))
    assert e.validity is EvidenceValidity.UNKNOWN
    assert e.counts_as_vote is False


# ── regressão arquitetural P1-A ──────────────────────────────────────────────

def test_architecture_default_never_votes():
    for kwargs in ({}, {"validity": "partial"}, {"validity": "invalid"},
                   {"validity": "stale"}, {"validity": "unsupported"},
                   {"validity": "unknown"}):
        e = Evidence(source="qualquer", **kwargs)
        assert e.counts_as_vote is False, kwargs


def test_architecture_no_implicit_promotion_to_valid():
    sig = _signal(confidence=0.99, strength=SignalStrength.STRONG)
    e = evidence_from_signal(sig)
    assert e.validity is not EvidenceValidity.VALID
    # Nem mesmo validade explícita parcial pode escalar sozinha:
    e2 = Evidence(source="s", validity="partial")
    assert e2.validity is not EvidenceValidity.VALID
