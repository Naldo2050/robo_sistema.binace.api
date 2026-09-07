# tests/unit/test_ratio_nonfinite_contract.py
"""
B-P0-1: contrato serializável do buy/sell ratio (sem sentinela, sem neutro).

Contrato:
  - buy=0, sell>0  -> ratio 0.0 (válido) + state sell_only
  - buy>0, sell=0  -> ratio None (infinito não serializa) + state buy_only
  - buy=sell=0     -> ratio None + state no_volume
  - missing/None/NaN/Inf -> ratio None + state invalid/insufficient
  - buy>0, sell>0  -> ratio buy/sell normal + state two_sided
  - zero legítimo preservado; nenhum NaN/Infinity/99 em JSON final.

Aplica-se às duas implementações (metrics + aggregates, paridade de teste).
"""

import json
import math

import pytest

from flow_analyzer.aggregates import (
    calculate_buy_sell_ratios as aggregates_calc,
)
from flow_analyzer.metrics import calculate_buy_sell_ratios as metrics_calc

CALCS = [metrics_calc, aggregates_calc]


def _no_sentinels(obj):
    text = json.dumps(obj, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    return text


@pytest.mark.parametrize("calc", CALCS)
def test_sell_only_ratio_zero_valid(calc):
    out = calc({"buy_volume_btc": 0.0, "sell_volume_btc": 4.0})
    assert out["buy_sell_ratio"] == 0.0
    assert out["ratio_state"] == "sell_only"
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
def test_buy_only_ratio_none_with_state(calc):
    out = calc({"buy_volume_btc": 4.0, "sell_volume_btc": 0.0})
    assert out["buy_sell_ratio"] is None, "infinito não serializa"
    assert out["ratio_state"] == "buy_only"
    assert out.get("pressure") is None
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
def test_both_zero_ratio_none(calc):
    out = calc({"buy_volume_btc": 0.0, "sell_volume_btc": 0.0})
    assert out["buy_sell_ratio"] is None, "vazio não é equilíbrio"
    assert out["ratio_state"] == "no_volume"
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"),
                                 float("-inf")])
def test_nonfinite_inputs_ratio_none(calc, bad):
    out = calc({"buy_volume_btc": bad, "sell_volume_btc": bad})
    assert out["buy_sell_ratio"] is None
    assert out["ratio_state"] in ("invalid", "insufficient")
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
def test_missing_keys_ratio_none(calc):
    out = calc({})
    assert out["buy_sell_ratio"] is None
    assert out["ratio_state"] in ("invalid", "insufficient")
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
def test_normal_ratio_unchanged(calc):
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 10.0, "net_flow_5m": 5.0,
                "total_volume": 5.0})
    assert out["buy_sell_ratio"] == 1.5
    assert out["ratio_state"] == "two_sided"
    assert out["ratios"]["imbalance_1m"] == 2.0
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
def test_equal_volumes_is_one(calc):
    out = calc({"buy_volume_btc": 2.5, "sell_volume_btc": 2.5})
    assert out["buy_sell_ratio"] == 1.0
    assert out["ratio_state"] == "two_sided"
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
def test_sector_empty_not_one(calc):
    out = calc({"buy_volume_btc": 1.0, "sell_volume_btc": 1.0,
                "sector_flow": {"retail": {"buy": 0.0, "sell": 0.0}}})
    assert out["sector_ratios"]["retail"] is None, "setor vazio não é 1.0"
    _no_sentinels(out)
