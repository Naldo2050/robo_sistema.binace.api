import pytest
from flow_analyzer.aggregates import analyze_passive_aggressive_flow


def test_passive_aggressive_flow_initialization():
    """Testa a função analyze_passive_aggressive_flow."""
    result = analyze_passive_aggressive_flow(flow_data={})
    assert isinstance(result, dict)
    assert result["status"] == "no_data"
    assert "aggressive" in result
    assert "passive" in result
    assert "composite" in result


def test_passive_aggressive_flow_with_minimal_data():
    """Testa a análise com dados mínimos de fluxo."""
    result = analyze_passive_aggressive_flow(
        flow_data={
            "aggressive_buy_pct": 55,
            "aggressive_sell_pct": 45,
            "buy_volume_btc": 10.5,
            "sell_volume_btc": 8.2,
            "flow_imbalance": 0.1,
            "net_flow_1m": 2.3
        },
        orderbook_data={
            "bid_depth_usd": 1000000,
            "ask_depth_usd": 500000,
            "imbalance": 0.15
        }
    )
    
    assert result["status"] == "success"
    assert result["aggressive"]["dominance"] == "buyers"
    assert result["passive"]["dominance"] == "buyers"
    assert result["composite"]["agreement"] == True
    assert result["composite"]["signal"] == "strong_bullish"
    assert "Both aggressive and passive buyers active" in result["composite"]["interpretation"]
    assert result["composite"]["conviction"] == "HIGH"


def test_passive_aggressive_flow_strong_bearish():
    """Testa cenário de tendência forte bearish."""
    result = analyze_passive_aggressive_flow(
        flow_data={
            "aggressive_buy_pct": 40,
            "aggressive_sell_pct": 60,
            "buy_volume_btc": 5.2,
            "sell_volume_btc": 12.8,
            "flow_imbalance": -0.2,
            "net_flow_1m": -7.6
        },
        orderbook_data={
            "bid_depth_usd": 300000,
            "ask_depth_usd": 1200000,
            "imbalance": -0.18
        }
    )
    
    assert result["status"] == "success"
    assert result["aggressive"]["dominance"] == "sellers"
    assert result["passive"]["dominance"] == "sellers"
    assert result["composite"]["agreement"] == True
    assert result["composite"]["signal"] == "strong_bearish"
    assert "Both aggressive and passive sellers active" in result["composite"]["interpretation"]
    assert result["composite"]["conviction"] == "HIGH"


def test_passive_aggressive_flow_buy_absorption():
    """Testa cenário de absorção de compra."""
    result = analyze_passive_aggressive_flow(
        flow_data={
            "aggressive_buy_pct": 58,
            "aggressive_sell_pct": 42,
            "buy_volume_btc": 15.3,
            "sell_volume_btc": 10.8,
            "flow_imbalance": 0.16,
            "net_flow_1m": 4.5
        },
        orderbook_data={
            "bid_depth_usd": 400000,
            "ask_depth_usd": 1500000,
            "imbalance": -0.25
        }
    )
    
    assert result["status"] == "success"
    assert result["aggressive"]["dominance"] == "buyers"
    assert result["passive"]["dominance"] == "sellers"
    assert result["composite"]["agreement"] == False
    assert result["composite"]["signal"] == "buy_absorption"
    assert "Aggressive buyers hitting passive sell walls" in result["composite"]["interpretation"]
    assert result["composite"]["conviction"] == "MEDIUM"


def test_passive_aggressive_flow_sell_absorption():
    """Testa cenário de absorção de venda."""
    result = analyze_passive_aggressive_flow(
        flow_data={
            "aggressive_buy_pct": 45,
            "aggressive_sell_pct": 55,
            "buy_volume_btc": 8.7,
            "sell_volume_btc": 13.2,
            "flow_imbalance": -0.1,
            "net_flow_1m": -4.5
        },
        orderbook_data={
            "bid_depth_usd": 1800000,
            "ask_depth_usd": 600000,
            "imbalance": 0.22
        }
    )
    
    assert result["status"] == "success"
    assert result["aggressive"]["dominance"] == "sellers"
    assert result["passive"]["dominance"] == "buyers"
    assert result["composite"]["agreement"] == False
    assert result["composite"]["signal"] == "sell_absorption"
    assert "Aggressive sellers hitting passive buy walls" in result["composite"]["interpretation"]
    assert result["composite"]["conviction"] == "MEDIUM"


def test_passive_aggressive_flow_no_orderbook_data():
    """Testa análise sem dados do order book."""
    result = analyze_passive_aggressive_flow(
        flow_data={
            "aggressive_buy_pct": 52,
            "aggressive_sell_pct": 48,
            "buy_volume_btc": 10.2,
            "sell_volume_btc": 9.8,
            "flow_imbalance": 0.04,
            "net_flow_1m": 0.4
        },
        orderbook_data=None
    )
    
    assert result["status"] == "success"
    assert result["passive"]["inference"] == "no_orderbook_data"
    assert result["composite"]["signal"] == "passive_unknown"


def test_passive_aggressive_flow_balanced():
    """Testa cenário de equilíbrio entre agressivo e passivo."""
    result = analyze_passive_aggressive_flow(
        flow_data={
            "aggressive_buy_pct": 50,
            "aggressive_sell_pct": 50,
            "buy_volume_btc": 7.5,
            "sell_volume_btc": 7.5,
            "flow_imbalance": 0,
            "net_flow_1m": 0
        },
        orderbook_data={
            "bid_depth_usd": 800000,
            "ask_depth_usd": 850000,
            "imbalance": 0.02
        }
    )
    
    assert result["status"] == "success"
    assert result["aggressive"]["dominance"] == "balanced"
    assert result["passive"]["dominance"] == "balanced"
    assert result["composite"]["agreement"] == True
    assert result["composite"]["signal"] == "neutral_balanced"
    assert "Both sides balanced" in result["composite"]["interpretation"]
    assert result["composite"]["conviction"] == "LOW"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])