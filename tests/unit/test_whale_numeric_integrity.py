# tests/unit/test_whale_numeric_integrity.py
"""
P2-B1 — Numeric Integrity Hardening: Whale Depth + Derivatives.

Contrato:
- DEPTH:
  - inputs finitos >= 0 e sum > 0 -> fórmula bit-equivalent;
  - NaN/Inf/malformed/negative -> NON_VOTING_INVALID_INPUT, score=0;
  - missing/None/empty -> NON_VOTING_MISSING, score=0;
  - zero/zero -> NON_VOTING_ZERO_DEPTH (contrato auditado do event_factory);
  - max=20 inalterado.
- DERIVATIVES:
  - LSR finito e > 0 -> fórmula bit-equivalent;
  - NaN/Inf/<=0 LSR -> non-voting, contribution=0;
  - funding finito -> preservado;
  - NaN funding não contamina LSR válido;
  - Inf netflow não contamina outros;
  - requires_paid_api / unavailable -> NON_VOTING_MISSING, não 0 observado;
  - todos missing -> component non-voting;
  - combinação parcial válida soma somente válidos;
  - detail per-input: observed, validity, contribution (reason opcional);
  - max=25 inalterado.
- GLOBAL:
  - score total sempre int finito em [-100, 100];
  - classificação sempre válida;
  - JSON RFC8259 estrito (nenhum NaN/Inf/Infinity);
  - nenhum TypeError/ValueError escapa de calculate();
  - P0 absorption continua NON_VOTING_UNVALIDATED_MAGNITUDE;
  - flow non-finite hardening intacto.
"""
from __future__ import annotations

import json
import math
import pytest

from common.json_safe import json_dumps_rfc8259, sanitize_json_safe
from flow_analyzer.whale_score import WhaleAccumulationCalculator


@pytest.fixture
def calc() -> WhaleAccumulationCalculator:
    return WhaleAccumulationCalculator()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DEPTH TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_depth_finite_normal_bit_equivalent(calc: WhaleAccumulationCalculator):
    """Inputs finitos válidos produzem resultados bit-equivalent."""
    # Caso simétrico
    res = calc.calculate(orderbook_data={"bid_depth_usd": 500_000.0, "ask_depth_usd": 500_000.0})
    depth = res["components"]["depth"]
    assert depth["score"] == 0.0
    assert depth["detail"]["ratio"] == 0.0
    assert "status" not in depth

    # Caso assimétrico normal
    res = calc.calculate(orderbook_data={"bid_depth_usd": 600_000.0, "ask_depth_usd": 400_000.0})
    depth = res["components"]["depth"]
    # (600k - 400k) / 1000k = 0.2 -> 0.2 * 20 = 4.0
    assert depth["score"] == 4.0
    assert depth["detail"]["ratio"] == 0.2
    assert "status" not in depth

    # Caso J2 real: 100k vs 1274.5k
    res = calc.calculate(orderbook_data={"bid_depth_usd": 100_000.0, "ask_depth_usd": 1_274_500.0})
    depth = res["components"]["depth"]
    assert depth["score"] == pytest.approx(-17.09, abs=0.01)

    # Book unilateral (ask = 0, bid > 0)
    res = calc.calculate(orderbook_data={"bid_depth_usd": 1_000_000.0, "ask_depth_usd": 0.0})
    depth = res["components"]["depth"]
    assert depth["score"] == 20.0
    assert depth["detail"]["ratio"] == 1.0


def test_depth_deep_confirmation_boost(calc: WhaleAccumulationCalculator):
    """Confirmação profunda de desequilíbrio aplica boost de 1.2x quando alinhada."""
    # ratio > 0 e deep_imb > 0 -> boost 1.2x
    ob = {
        "bid_depth_usd": 600_000.0,
        "ask_depth_usd": 400_000.0,
        "depth_metrics": {"depth_imbalance": 0.3},
    }
    res = calc.calculate(orderbook_data=ob)
    depth = res["components"]["depth"]
    assert depth["score"] == pytest.approx(4.8, abs=0.01)
    assert depth["detail"]["deep_confirmation"] is True


@pytest.mark.parametrize("bad_val", [float("nan"), "nan", "NaN"])
def test_depth_nan_bid_is_non_voting(calc: WhaleAccumulationCalculator, bad_val):
    """NaN em bid_depth não contamina e resulta em NON_VOTING_INVALID_INPUT."""
    ob = {"bid_depth_usd": bad_val, "ask_depth_usd": 500_000.0}
    res = calc.calculate(orderbook_data=ob)
    depth = res["components"]["depth"]
    assert depth["score"] == 0.0
    assert depth["status"] == "NON_VOTING_INVALID_INPUT"
    assert depth["reason"] == "BID_OR_ASK_DEPTH_NONFINITE_OR_MALFORMED"


@pytest.mark.parametrize("bad_val", [float("inf"), float("-inf"), "inf", "-inf", "Infinity"])
def test_depth_inf_ask_is_non_voting(calc: WhaleAccumulationCalculator, bad_val):
    """Infinito em ask_depth não contamina e resulta em NON_VOTING_INVALID_INPUT."""
    ob = {"bid_depth_usd": 500_000.0, "ask_depth_usd": bad_val}
    res = calc.calculate(orderbook_data=ob)
    depth = res["components"]["depth"]
    assert depth["score"] == 0.0
    assert depth["status"] == "NON_VOTING_INVALID_INPUT"
    assert depth["reason"] == "BID_OR_ASK_DEPTH_NONFINITE_OR_MALFORMED"


@pytest.mark.parametrize("malformed", ["abc", "error", "", {}, []])
def test_depth_malformed_is_non_voting(calc: WhaleAccumulationCalculator, malformed):
    """Entradas malformadas em depth resultam em NON_VOTING_INVALID_INPUT sem TypeError."""
    ob = {"bid_depth_usd": malformed, "ask_depth_usd": 500_000.0}
    res = calc.calculate(orderbook_data=ob)
    depth = res["components"]["depth"]
    assert depth["score"] == 0.0
    assert depth["status"] == "NON_VOTING_INVALID_INPUT"


def test_depth_none_is_non_voting_missing(calc: WhaleAccumulationCalculator):
    """Valores None em chaves de depth são tratados como NON_VOTING_MISSING."""
    ob = {"bid_depth_usd": None, "ask_depth_usd": 500_000.0}
    res = calc.calculate(orderbook_data=ob)
    depth = res["components"]["depth"]
    assert depth["score"] == 0.0
    assert depth["status"] == "NON_VOTING_MISSING"
    assert depth["reason"] == "BID_OR_ASK_DEPTH_MISSING"


def test_depth_missing_orderbook_data(calc: WhaleAccumulationCalculator):
    """orderbook_data ausente ou vazio resulta em NON_VOTING_MISSING."""
    for ob in [None, {}, "not_a_dict"]:
        res = calc.calculate(orderbook_data=ob)
        depth = res["components"]["depth"]
        assert depth["score"] == 0.0
        assert depth["status"] == "NON_VOTING_MISSING"
        assert depth["reason"] == "ORDERBOOK_DATA_MISSING"


def test_depth_zero_zero_fail_closed_contract(calc: WhaleAccumulationCalculator):
    """Contrato provado: bid=0 e ask=0 (event_factory emite em erro/indisponibilidade) -> fail-closed."""
    ob = {"bid_depth_usd": 0.0, "ask_depth_usd": 0.0}
    res = calc.calculate(orderbook_data=ob)
    depth = res["components"]["depth"]
    assert depth["score"] == 0.0
    assert depth["status"] == "NON_VOTING_ZERO_DEPTH"
    assert depth["reason"] == "ZERO_TOTAL_DEPTH"


def test_depth_negative_is_invalid(calc: WhaleAccumulationCalculator):
    """Profundidade negativa é fisicamente inválida e vira NON_VOTING_INVALID_INPUT."""
    ob = {"bid_depth_usd": -100.0, "ask_depth_usd": 500_000.0}
    res = calc.calculate(orderbook_data=ob)
    depth = res["components"]["depth"]
    assert depth["score"] == 0.0
    assert depth["status"] == "NON_VOTING_INVALID_INPUT"
    assert depth["reason"] == "NEGATIVE_DEPTH"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DERIVATIVES & ONCHAIN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_derivatives_finite_lsr_bit_equivalent(calc: WhaleAccumulationCalculator):
    """LSR finito produz score bit-equivalent ao engine original."""
    # LSR > 1: (lsr - 1) * 15
    res = calc.calculate(derivatives_data={"BTCUSDT": {"long_short_ratio": 2.0}})
    deriv = res["components"]["derivatives"]
    assert deriv["score"] == 15.0
    assert deriv["detail"]["inputs"]["lsr"]["validity"] == "VALID"
    assert deriv["detail"]["inputs"]["lsr"]["contribution"] == 15.0

    # LSR clamp em 20.0
    res = calc.calculate(derivatives_data={"BTCUSDT": {"long_short_ratio": 3.0}})
    assert res["components"]["derivatives"]["score"] == 20.0

    # LSR < 1: (lsr - 1) * 20
    res = calc.calculate(derivatives_data={"BTCUSDT": {"long_short_ratio": 0.5}})
    assert res["components"]["derivatives"]["score"] == -10.0

    # LSR = 1: neutro
    res = calc.calculate(derivatives_data={"BTCUSDT": {"long_short_ratio": 1.0}})
    assert res["components"]["derivatives"]["score"] == 0.0


@pytest.mark.parametrize("bad_lsr", [float("nan"), float("inf"), float("-inf"), "nan", "inf", "abc"])
def test_derivatives_nan_inf_lsr_non_voting(calc: WhaleAccumulationCalculator, bad_lsr):
    """LSR não-finito ou malformado não vira score extremo (-20/20) e fica non-voting."""
    res = calc.calculate(derivatives_data={"BTCUSDT": {"long_short_ratio": bad_lsr}})
    deriv = res["components"]["derivatives"]
    assert deriv["score"] == 0.0
    assert deriv["detail"]["inputs"]["lsr"]["validity"] == "NON_VOTING_INVALID_INPUT"
    assert deriv["detail"]["inputs"]["lsr"]["contribution"] == 0.0


@pytest.mark.parametrize("invalid_lsr", [0.0, -1.0, -0.01])
def test_derivatives_non_positive_lsr_non_voting(calc: WhaleAccumulationCalculator, invalid_lsr):
    """LSR menor ou igual a zero é semanticamente impossível em ratio longs/shorts."""
    res = calc.calculate(derivatives_data={"BTCUSDT": {"long_short_ratio": invalid_lsr}})
    deriv = res["components"]["derivatives"]
    assert deriv["score"] == 0.0
    assert deriv["detail"]["inputs"]["lsr"]["validity"] == "NON_VOTING_INVALID_INPUT"
    assert deriv["detail"]["inputs"]["lsr"]["reason"] == "LSR_NON_POSITIVE"


def test_derivatives_finite_funding_preserved(calc: WhaleAccumulationCalculator):
    """Funding rates válidos contribuem corretamente para o componente."""
    # 0.0001 * 10000 = 1.0
    res = calc.calculate(onchain_data={"funding_rates": {"binance": 0.0001}})
    deriv = res["components"]["derivatives"]
    assert deriv["score"] == 1.0
    assert deriv["detail"]["inputs"]["funding"]["validity"] == "VALID"
    assert deriv["detail"]["inputs"]["funding"]["contribution"] == 1.0

    # Funding negativo: -0.0002 * 10000 = -2.0
    res = calc.calculate(onchain_data={"funding_rates": {"binance": -0.0002}})
    assert res["components"]["derivatives"]["score"] == -2.0

    # Funding clamp em max 5.0
    res = calc.calculate(onchain_data={"funding_rates": {"binance": 0.001}})
    assert res["components"]["derivatives"]["score"] == 5.0


def test_derivatives_nan_funding_does_not_contaminate_valid_lsr(calc: WhaleAccumulationCalculator):
    """Funding NaN em onchain_data não contamina LSR válido em derivatives_data."""
    res = calc.calculate(
        derivatives_data={"BTCUSDT": {"long_short_ratio": 2.0}},
        onchain_data={"funding_rates": {"binance": float("nan")}},
    )
    deriv = res["components"]["derivatives"]
    # LSR contribui 15.0; funding contribui 0.0 (non-voting)
    assert deriv["score"] == 15.0
    assert deriv["detail"]["inputs"]["lsr"]["contribution"] == 15.0
    assert deriv["detail"]["inputs"]["funding"]["validity"] == "NON_VOTING_INVALID_INPUT"
    assert deriv["detail"]["inputs"]["funding"]["contribution"] == 0.0


def test_derivatives_inf_netflow_does_not_contaminate_others(calc: WhaleAccumulationCalculator):
    """Netflow Infinito não contamina LSR nem funding válidos."""
    res = calc.calculate(
        derivatives_data={"BTCUSDT": {"long_short_ratio": 2.0}},
        onchain_data={
            "funding_rates": {"binance": 0.0001},
            "exchange_netflow": float("inf"),
        },
    )
    deriv = res["components"]["derivatives"]
    # LSR=15.0 + funding=1.0 = 16.0. Netflow inválido = 0.0
    assert deriv["score"] == 16.0
    assert deriv["detail"]["inputs"]["netflow"]["validity"] == "NON_VOTING_INVALID_INPUT"
    assert deriv["detail"]["inputs"]["netflow"]["contribution"] == 0.0


def test_derivatives_all_missing_is_non_voting(calc: WhaleAccumulationCalculator):
    """Quando todos os inputs de derivativos estão ausentes, componente vira NON_VOTING_MISSING."""
    res = calc.calculate(derivatives_data={}, onchain_data={})
    deriv = res["components"]["derivatives"]
    assert deriv["score"] == 0.0
    assert deriv["status"] == "NON_VOTING_MISSING"
    assert deriv["reason"] == "ALL_INPUTS_MISSING"


def test_derivatives_partial_valid_combination(calc: WhaleAccumulationCalculator):
    """Combinação parcial válida soma somente as contribuições válidas."""
    res = calc.calculate(
        derivatives_data={"BTCUSDT": {"long_short_ratio": 2.0}},  # 15.0
        onchain_data={
            "funding_rates": {"binance": 0.0001, "bybit": float("nan")},  # média só dos finitos -> 0.0001 -> 1.0
            "exchange_netflow": -50.0,  # -(-50) * 0.02 = 1.0
        },
    )
    deriv = res["components"]["derivatives"]
    # 15.0 + 1.0 + 1.0 = 17.0
    assert deriv["score"] == 17.0
    assert deriv["detail"]["inputs"]["lsr"]["contribution"] == 15.0
    assert deriv["detail"]["inputs"]["funding"]["contribution"] == 1.0
    assert deriv["detail"]["inputs"]["netflow"]["contribution"] == 1.0


def test_onchain_netflow_requires_paid_api_distinguished_from_zero(calc: WhaleAccumulationCalculator):
    """requires_paid_api em netflow não vira 0 observado, vira NON_VOTING_MISSING."""
    # Caso 1: requires_paid_api explícito
    onch_paid = {
        "exchange_netflow": 0.0,
        "requires_paid_api": ["exchange_netflow"],
    }
    res = calc.calculate(onchain_data=onch_paid)
    nf_detail = res["components"]["derivatives"]["detail"]["inputs"]["netflow"]
    assert nf_detail["validity"] == "NON_VOTING_MISSING"
    assert nf_detail["reason"] == "REQUIRES_PAID_API"
    assert nf_detail["observed"] is None
    assert nf_detail["contribution"] == 0.0

    # Caso 2: status="requires_paid_api"
    onch_status = {
        "exchange_netflow": 0.0,
        "status": "requires_paid_api",
    }
    res2 = calc.calculate(onchain_data=onch_status)
    nf_detail2 = res2["components"]["derivatives"]["detail"]["inputs"]["netflow"]
    assert nf_detail2["validity"] == "NON_VOTING_MISSING"
    assert nf_detail2["reason"] == "REQUIRES_PAID_API"

    # Caso 3: 0.0 observado de verdade (sem requires_paid_api)
    onch_real_zero = {"exchange_netflow": 0.0}
    res3 = calc.calculate(onchain_data=onch_real_zero)
    nf_detail3 = res3["components"]["derivatives"]["detail"]["inputs"]["netflow"]
    assert nf_detail3["validity"] == "VALID_ZERO_OBSERVED"
    assert nf_detail3["observed"] == 0.0
    assert nf_detail3["contribution"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GLOBAL, STRING/MALFORMED & RFC8259 PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bad_val", ["nan", "inf", "-inf", "abc", None, {}, []])
def test_no_type_error_escapes_calculate(calc: WhaleAccumulationCalculator, bad_val):
    """Nenhum TypeError ou ValueError escapa de calculate() com inputs hostis."""
    # Testar campos individuais com tipos bizarros
    res = calc.calculate(
        sector_flow={"whale": {"delta": bad_val}},
        orderbook_data={"bid_depth_usd": bad_val, "ask_depth_usd": bad_val},
        absorption_data={"current_absorption": {"buyer_strength": bad_val, "seller_exhaustion": bad_val}},
        derivatives_data={"BTCUSDT": {"long_short_ratio": bad_val, "open_interest": bad_val}},
        onchain_data={"exchange_netflow": bad_val, "funding_rates": bad_val},
        cvd=bad_val,
    )
    assert isinstance(res["score"], int)
    assert math.isfinite(res["score"])
    assert res["classification"] in (
        "STRONG_ACCUMULATION", "MILD_ACCUMULATION", "NEUTRAL",
        "MILD_DISTRIBUTION", "STRONG_DISTRIBUTION",
    )


def test_rfc8259_json_compliance(calc: WhaleAccumulationCalculator):
    """Resultado é 100% serializável conforme RFC 8259 (sem NaN/Infinity literais)."""
    res = calc.calculate(
        sector_flow={"whale": {"delta": float("nan")}},
        orderbook_data={"bid_depth_usd": float("nan"), "ask_depth_usd": float("inf")},
        absorption_data={"current_absorption": {"buyer_strength": float("nan")}},
        derivatives_data={"BTCUSDT": {"long_short_ratio": float("nan")}},
        onchain_data={"exchange_netflow": float("nan")},
        cvd=float("nan"),
    )
    safe = sanitize_json_safe(res)
    text = json_dumps_rfc8259(safe)
    assert "NaN" not in text
    assert "Infinity" not in text
    loaded = json.loads(text)
    assert isinstance(loaded["score"], int)


def test_p0_absorption_remains_non_voting(calc: WhaleAccumulationCalculator):
    """P0-A2 continua estritamente intacto: score sempre 0.0, NON_VOTING_UNVALIDATED_MAGNITUDE."""
    res = calc.calculate(
        absorption_data={
            "current_absorption": {
                "buyer_strength": 9.5,
                "seller_exhaustion": 1.0,
                "label": "Absorção de Compra",
            }
        }
    )
    comp = res["components"]["absorption"]
    assert comp["score"] == 0.0
    assert comp["status"] == "NON_VOTING_UNVALIDATED_MAGNITUDE"
    assert comp["canonical_direction"] == "BEARISH"


def test_flow_nonfinite_hardening_intact(calc: WhaleAccumulationCalculator):
    """Hardening de flow P0-FINAL continua operando e se integrando ao cálculo."""
    # Sector delta NaN vira non-voting
    res = calc.calculate(
        sector_flow={"whale": {"delta": float("nan")}, "mid": {"delta": float("nan")}},
        cvd=0.0,
    )
    assert res["components"]["flow"]["score"] == 0.0
    assert res["components"]["flow"]["status"] == "NON_VOTING_NONFINITE"

    # CVD NaN vira non-voting
    res2 = calc.calculate(
        sector_flow={},
        cvd=float("nan"),
    )
    assert res2["components"]["flow"]["score"] == 0.0
    assert res2["components"]["flow"]["detail"]["cvd_status"] == "NON_VOTING_NONFINITE"
