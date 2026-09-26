# tests/unit/test_p0final_nonfinite_whale_flow.py — P0-FINAL-CLOSE item 3.
#
# Contrato (só integridade numérica; fórmula finita bit-equivalente):
# - CVD NaN/±Inf: não produz flow_score NaN/Inf; non-voting com status;
#   distinguível de zero observado via component status.
# - Sector deltas NaN/±Inf: campo ruim vira 0.0 (como ausente) com rastro;
#   se NENHUM delta for utilizável => NON_VOTING_NONFINITE; se houver
#   delta válido restante, ele vota normalmente (fórmula intacta).

import json
import math

import pytest

from common.json_safe import json_dumps_rfc8259, sanitize_json_safe
from flow_analyzer.whale_score import WhaleAccumulationCalculator


def _flow_only(**kwargs):
    calc = WhaleAccumulationCalculator()
    out = calc.calculate(derivatives_data={}, onchain_data={}, **kwargs)
    return out["components"]["flow"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_cvd_nonfinite_is_non_voting(bad):
    comp = _flow_only(sector_flow={}, orderbook_data={}, absorption_data=None,
                      cvd=bad)
    assert comp["score"] == 0.0
    assert math.isfinite(comp["score"])
    assert comp["detail"].get("cvd_status") == "NON_VOTING_NONFINITE"
    assert "cvd_used" not in comp["detail"]
    json_dumps_rfc8259(sanitize_json_safe(comp))


@pytest.mark.parametrize("cvd,expected", [(2.0, 10.0), (-4.0, -15.0),
                                          (0.5, 2.5), (0.0, 0.0),
                                          (10.0, 15.0), (-10.0, -15.0)])
def test_cvd_finite_bit_equivalent(cvd, expected):
    comp = _flow_only(sector_flow={}, orderbook_data={}, absorption_data=None,
                      cvd=cvd)
    assert comp["score"] == expected
    assert comp["detail"].get("cvd_used") is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_sector_all_nonfinite_is_non_voting(bad):
    comp = _flow_only(
        sector_flow={"whale": {"delta": bad}, "mid": {"delta": bad},
                     "retail": {"delta": bad}},
        orderbook_data={}, absorption_data=None, cvd=0.0)
    assert comp["score"] == 0.0
    assert math.isfinite(comp["score"])
    assert comp["status"] == "NON_VOTING_NONFINITE"
    assert comp["reason"] == "SECTOR_DELTA_NONFINITE"
    assert comp["detail"]["nonfinite_ignored"] == ["mid", "retail", "whale"]
    json_dumps_rfc8259(sanitize_json_safe(comp))


def test_sector_partial_nonfinite_falls_back_to_valid():
    # whale NaN + mid 2.0 => mid vota como se whale estivesse ausente (2.0*10).
    comp = _flow_only(
        sector_flow={"whale": {"delta": float("nan")},
                     "mid": {"delta": 2.0}, "retail": {"delta": 0.0}},
        orderbook_data={}, absorption_data=None, cvd=0.0)
    assert comp["score"] == 20.0
    assert "status" not in comp  # componente votou validamente
    assert comp["detail"]["nonfinite_ignored"] == ["whale"]


def test_sector_finite_unchanged_no_status_keys():
    comp = _flow_only(
        sector_flow={"whale": {"delta": 3.0}, "mid": {"delta": 0.0},
                     "retail": {"delta": 0.0}},
        orderbook_data={}, absorption_data=None, cvd=0.0)
    assert comp["score"] == 30.0
    assert "status" not in comp
    assert "nonfinite_ignored" not in comp["detail"]
    assert "cvd_status" not in comp["detail"]


def test_no_sector_no_cvd_legacy_shape():
    comp = _flow_only(sector_flow={}, orderbook_data={}, absorption_data=None,
                      cvd=None)
    assert comp["score"] == 0.0
    assert comp["detail"] == {}
    assert "status" not in comp
    text = json_dumps_rfc8259(sanitize_json_safe(comp))
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["score"] == 0.0
