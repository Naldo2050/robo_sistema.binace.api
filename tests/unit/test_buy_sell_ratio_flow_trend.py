import pytest

from flow_analyzer.aggregates import calculate_buy_sell_ratios as aggregates_calc
from flow_analyzer.metrics import calculate_buy_sell_ratios as metrics_calc


@pytest.mark.parametrize("calc", [metrics_calc, aggregates_calc], ids=["metrics", "aggregates"])
class TestFlowTrendImbalanceNormalizado:
    def test_caso1_decelerating_selling(self, calc):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": -74.0,
            "net_flow_5m": -296.0,
            "net_flow_15m": 0.0,
            "total_volume": 1000.0,
        }
        result = calc(flow_data)
        assert result["flow_trend"] == "decelerating_selling"

    def test_caso2_accelerating_selling(self, calc):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": -450.0,
            "net_flow_5m": -200.0,
            "net_flow_15m": 0.0,
            "total_volume": 1000.0,
        }
        result = calc(flow_data)
        assert result["flow_trend"] == "accelerating_selling"

    def test_caso3_stable_selling(self, calc):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": -210.0,
            "net_flow_5m": -200.0,
            "net_flow_15m": 0.0,
            "total_volume": 1000.0,
        }
        result = calc(flow_data)
        assert result["flow_trend"] == "stable_selling"

    def test_caso4_decelerating_buying(self, calc):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": 100.0,
            "net_flow_5m": 350.0,
            "net_flow_15m": 0.0,
            "total_volume": 1000.0,
        }
        result = calc(flow_data)
        assert result["flow_trend"] == "decelerating_buying"

    def test_limiar_exato_0_05_e_stable(self, calc):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": -250.0,
            "net_flow_5m": -200.0,
            "net_flow_15m": 0.0,
            "total_volume": 1000.0,
        }
        result = calc(flow_data)
        assert result["flow_trend"] == "stable_selling"

    def test_insufficient_data_quando_fluxos_zero(self, calc):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": 0.0,
            "net_flow_5m": 0.0,
            "net_flow_15m": 0.0,
            "total_volume": 1000.0,
        }
        result = calc(flow_data)
        assert result["flow_trend"] == "insufficient_data"

    def test_insufficient_data_quando_sem_total_volume(self, calc):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": -74.0,
            "net_flow_5m": -296.0,
            "net_flow_15m": 0.0,
            "total_volume": 0,
        }
        result = calc(flow_data)
        assert result["flow_trend"] == "insufficient_data"

    def test_imbalances_expostos_no_resultado(self, calc):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": -74.0,
            "net_flow_5m": -296.0,
            "net_flow_15m": -444.0,
            "total_volume": 1000.0,
        }
        result = calc(flow_data)
        assert result["ratios"]["imbalance_1m"] == -0.074
        assert result["ratios"]["imbalance_5m"] == -0.296
        assert result["ratios"]["imbalance_15m"] == -0.444


class TestPressureNeutralZoneComFlowTrend:
    def test_neutral_ratio_com_accelerating_buying(self):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": 450.0,
            "net_flow_5m": 200.0,
            "net_flow_15m": 0.0,
            "total_volume": 1000.0,
        }
        result = metrics_calc(flow_data)
        assert result["pressure"] == "SLIGHT_BUY"

    def test_neutral_ratio_com_accelerating_selling(self):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": -450.0,
            "net_flow_5m": -200.0,
            "net_flow_15m": 0.0,
            "total_volume": 1000.0,
        }
        result = metrics_calc(flow_data)
        assert result["pressure"] == "SLIGHT_SELL"

    def test_neutral_ratio_com_stable_fica_neutral(self):
        flow_data = {
            "buy_volume_btc": 500.0,
            "sell_volume_btc": 500.0,
            "net_flow_1m": 10.0,
            "net_flow_5m": 20.0,
            "net_flow_15m": 0.0,
            "total_volume": 1000.0,
        }
        result = metrics_calc(flow_data)
        assert result["pressure"] == "NEUTRAL"
