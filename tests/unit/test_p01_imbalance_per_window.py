"""P01: imbalance normalizado por janela usa o total da PRÓPRIA janela.

Contrato: imbalance_X = (buy_X - sell_X) / (buy_X + sell_X) = net_X / total_X,
domínio [-1, +1]. Sem clamp de erro, sem fallback cruzado net_5m/total_1m.
Aplica-se às duas implementações (metrics + aggregates, paridade).
"""
import pytest

from flow_analyzer.aggregates import (
    calculate_buy_sell_ratios as aggregates_calc,
)
from flow_analyzer.metrics import calculate_buy_sell_ratios as metrics_calc

CALCS = [metrics_calc, aggregates_calc]


def _flow(buy, sell, net1=None, tot1=None, net5=None, tot5=None, net15=None, tot15=None):
    d = {"buy_volume_btc": buy, "sell_volume_btc": sell}
    if net1 is not None:
        d["net_flow_1m"] = net1
    if tot1 is not None:
        d["total_volume"] = tot1
        d["total_volume_1m"] = tot1
    if net5 is not None:
        d["net_flow_5m"] = net5
    if tot5 is not None:
        d["total_volume_5m"] = tot5
    if net15 is not None:
        d["net_flow_15m"] = net15
    if tot15 is not None:
        d["total_volume_15m"] = tot15
    return d


@pytest.mark.parametrize("calc", CALCS)
def test_a_buy_only_plus_one(calc):
    out = calc(_flow(100.0, 0.0, net1=100.0, tot1=100.0, net5=100.0, tot5=100.0, net15=100.0, tot15=100.0))
    assert out["ratios"]["imbalance_1m"] == 1.0
    assert out["ratios"]["imbalance_5m"] == 1.0
    assert out["ratios"]["imbalance_15m"] == 1.0


@pytest.mark.parametrize("calc", CALCS)
def test_b_sell_only_minus_one(calc):
    out = calc(_flow(0.0, 100.0, net1=-100.0, tot1=100.0, net5=-100.0, tot5=100.0, net15=-100.0, tot15=100.0))
    assert out["ratios"]["imbalance_1m"] == -1.0
    assert out["ratios"]["imbalance_5m"] == -1.0
    assert out["ratios"]["imbalance_15m"] == -1.0


@pytest.mark.parametrize("calc", CALCS)
def test_c_equal_volumes_zero(calc):
    out = calc(_flow(50.0, 50.0, net1=0.0, tot1=100.0, net5=0.0, tot5=100.0, net15=0.0, tot15=100.0))
    assert out["ratios"]["imbalance_1m"] == 0.0
    assert out["ratios"]["imbalance_5m"] == 0.0
    assert out["ratios"]["imbalance_15m"] == 0.0


@pytest.mark.parametrize("calc", CALCS)
def test_d_75_25_plus_half(calc):
    out = calc(_flow(75.0, 25.0, net1=50.0, tot1=100.0, net5=50.0, tot5=100.0, net15=50.0, tot15=100.0))
    assert out["ratios"]["imbalance_1m"] == 0.5
    assert out["ratios"]["imbalance_5m"] == 0.5
    assert out["ratios"]["imbalance_15m"] == 0.5


@pytest.mark.parametrize("calc", CALCS)
def test_e_25_75_minus_half(calc):
    out = calc(_flow(25.0, 75.0, net1=-50.0, tot1=100.0, net5=-50.0, tot5=100.0, net15=-50.0, tot15=100.0))
    assert out["ratios"]["imbalance_1m"] == -0.5
    assert out["ratios"]["imbalance_5m"] == -0.5
    assert out["ratios"]["imbalance_15m"] == -0.5


@pytest.mark.parametrize("calc", CALCS)
def test_f_different_horizons_each_own_denominator(calc):
    # 1m total=10; 5m buy=60/sell=40 (net +20/total 100 -> +0.2);
    # 15m buy=120/sell=80 (net +40/total 200 -> +0.2). Nunca net_5m/total_1m.
    out = calc(_flow(6.0, 4.0, net1=2.0, tot1=10.0, net5=20.0, tot5=100.0, net15=40.0, tot15=200.0))
    assert out["ratios"]["imbalance_1m"] == pytest.approx(0.2)
    assert out["ratios"]["imbalance_5m"] == pytest.approx(0.2)
    assert out["ratios"]["imbalance_15m"] == pytest.approx(0.2)


@pytest.mark.parametrize("calc", CALCS)
def test_g_zero_total_no_exception_key_omitted(calc):
    out = calc(_flow(0.0, 0.0, net1=0.0, tot1=0.0, net5=0.0, tot5=0.0, net15=0.0, tot15=0.0))
    assert "imbalance_1m" not in out["ratios"]
    assert "imbalance_5m" not in out["ratios"]
    assert "imbalance_15m" not in out["ratios"]


@pytest.mark.parametrize("calc", CALCS)
def test_h_tiny_volumes_still_bounded(calc):
    out = calc(_flow(1e-9, 2e-9, net1=-1e-9, tot1=3e-9, net5=-1e-9, tot5=3e-9, net15=-1e-9, tot15=3e-9))
    for key in ("imbalance_1m", "imbalance_5m", "imbalance_15m"):
        assert -1.0 <= out["ratios"][key] <= 1.0


@pytest.mark.parametrize("calc", CALCS)
def test_no_cross_window_fallback(calc):
    # Só total 1m presente: 5m/15m NÃO podem usar denominador 1m (P01).
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 2.0, "net_flow_5m": 20.0, "net_flow_15m": 40.0,
                "total_volume": 5.0})
    assert out["ratios"]["imbalance_1m"] == 0.4
    assert "imbalance_5m" not in out["ratios"]
    assert "imbalance_15m" not in out["ratios"]


@pytest.mark.parametrize(
    "buy,sell",
    [(100.0, 0.0), (0.0, 100.0), (50.0, 50.0), (75.0, 25.0), (25.0, 75.0),
     (60.0, 40.0), (1.0, 999.0), (0.001, 0.002), (123.456, 78.9)],
)
@pytest.mark.parametrize("calc", CALCS)
def test_invariant_bounded(calc, buy, sell):
    total = buy + sell
    net = buy - sell
    out = calc({"buy_volume_btc": buy, "sell_volume_btc": sell,
                "net_flow_1m": net, "net_flow_5m": net, "net_flow_15m": net,
                "total_volume_1m": total, "total_volume_5m": total, "total_volume_15m": total})
    for key in ("imbalance_1m", "imbalance_5m", "imbalance_15m"):
        assert -1.0 <= out["ratios"][key] <= 1.0
        assert out["ratios"][key] == pytest.approx(round(net / total, 4))
