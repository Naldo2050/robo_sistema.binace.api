# tests/unit/test_orderbook_market_impact_insufficient_liquidity.py
"""
Teste unitário para validação da flag insufficient_liquidity em _simulate_market_impact.
Garante que ordens que esgotam todos os níveis disponíveis do book recebam
insufficient_liquidity=True, fill_ratio correto e levels == len(levels).
"""

import pytest
from orderbook_analyzer.core import _simulate_market_impact


def test_market_impact_insufficient_liquidity():
    """
    Quando o book possui profundidade total inferior ao notional solicitado:
    - insufficient_liquidity deve ser True
    - fill_ratio deve ser round(spent / usd_amount, 4)
    - usd_filled deve ser igual ao total gasto
    - levels deve ser igual a len(book)
    """
    # Book com 2 níveis: 100 * 1 = $100, 101 * 2 = $202 -> Total = $302
    levels = [(100.0, 1.0), (101.0, 2.0)]
    usd_amount = 500.0  # Notional maior que o total disponível ($302)
    mid = 100.0

    res = _simulate_market_impact(levels, usd_amount, side="buy", mid=mid)

    assert res["insufficient_liquidity"] is True
    assert res["usd_filled"] == 302.0
    expected_fill_ratio = round(302.0 / 500.0, 4)  # 0.604
    assert res["fill_ratio"] == expected_fill_ratio
    assert res["levels"] == len(levels)
    assert res["usd"] == 500.0
    assert res["final_price"] == 101.0
    assert res["vwap"] == pytest.approx(302.0 / 3.0, rel=1e-5)


def test_market_impact_sufficient_liquidity():
    """
    Quando o book possui profundidade suficiente para o notional solicitado:
    - insufficient_liquidity deve ser False
    - fill_ratio deve ser 1.0
    - usd_filled deve ser igual ao usd_amount solicitado
    - levels deve refletir apenas os níveis consumidos
    """
    # Book com 3 níveis: 100 * 2 = $200, 101 * 5 = $505, 102 * 10 = $1020 -> Total = $1725
    levels = [(100.0, 2.0), (101.0, 5.0), (102.0, 10.0)]
    usd_amount = 400.0  # Notional atendido nos 2 primeiros níveis ($200 + $200 de $505)
    mid = 100.0

    res = _simulate_market_impact(levels, usd_amount, side="buy", mid=mid)

    assert res["insufficient_liquidity"] is False
    assert res["fill_ratio"] == 1.0
    assert res["usd_filled"] == 400.0
    assert res["levels"] == 2
    assert res["usd"] == 400.0
    assert res["final_price"] == 101.0


def test_market_impact_empty_levels_insufficient():
    """Book vazio com usd_amount > 0 deve retornar insufficient_liquidity=True."""
    res = _simulate_market_impact([], 100.0, side="buy", mid=100.0)
    assert res["insufficient_liquidity"] is True
    assert res["fill_ratio"] == 0.0
    assert res["usd_filled"] == 0.0
    assert res["levels"] == 0
