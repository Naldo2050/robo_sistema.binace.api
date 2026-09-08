# tests/unit/test_ml_eligibility.py — P1-C3: elegibilidade fail-closed.
#
# Previsão só vira evidência com status ok + valid_for_futures True +
# ml_stale False (+ frozen/hybrid via fontes reais). Metadata ausente != ok.
# prob 0.5 elegível = neutra real, distinguível (helper None + status ok).

import pytest

from market_orchestrator.ai.payload_builder_compact import _build_quant
from ml.hybrid_decision import eligibility_reason, is_eligible_prediction


def _ok(**over):
    base = {"status": "ok", "prob_up": 0.75, "confidence": 0.8,
            "valid_for_futures": True, "ml_stale": False}
    base.update(over)
    return base


def test_eligible_ok_emits():
    assert is_eligible_prediction(_ok()) is True
    assert eligibility_reason(_ok()) is None


def test_eligible_neutral_is_real_and_distinguishable():
    pred = _ok(prob_up=0.5, confidence=0.0)
    assert is_eligible_prediction(pred) is True
    q = _build_quant({"ml_prediction": pred})
    assert q["pu"] == 0.5 and q["ml_stale"] is False


def test_stale_is_not_evidence():
    assert is_eligible_prediction(_ok(ml_stale=True)) is False


def test_spot_model_is_not_evidence():
    assert is_eligible_prediction(_ok(valid_for_futures=False)) is False


def test_missing_metadata_is_not_evidence():
    # EXTRA exigido: ok + prob válido, mas SEM metadata => NÃO emitido.
    pred = {"status": "ok", "prob_up": 0.62, "confidence": 0.7}
    assert is_eligible_prediction(pred) is False
    assert eligibility_reason(pred) == "valid_for_futures_not_true"
    q = _build_quant({"ml_prediction": pred})
    assert "pu" not in q and "c" not in q
    assert q["ml_stale"] is True


@pytest.mark.parametrize("pred", [
    {"status": "hybrid_disabled", "prob_up": 0.5},
    {"status": "model_not_loaded", "prob_up": 0.5},
    {"status": "error", "prob_up": 0.5},
    {"status": "neutralized", "prob_up": 0.5},
    {"prob_up": 0.9},
    {},
    None,
    "ok",
])
def test_ineligible_shapes(pred):
    assert is_eligible_prediction(pred) is False
    assert eligibility_reason(pred) is not None


def test_frozen_filtered_is_not_evidence():
    assert is_eligible_prediction(
        _ok(frozen_filtered=True)) is False


def test_quant_omits_without_artificial_half():
    q = _build_quant({"ml_prediction": _ok(ml_stale=True, prob_up=0.9)})
    assert "pu" not in q and "c" not in q
    assert q == {"ml_stale": True, "reason": "ml_stale_not_false"}
    q2 = _build_quant({})
    assert q2 == {"ml_stale": True, "reason": "missing_prediction"}
