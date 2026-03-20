"""Rodar: python -m pytest tests/test_window_state.py -v"""

from core.window_state import WindowState
from core.state_manager import StateManager
from market_orchestrator.windows.window_processor import (
    _populate_window_state,
    _populate_window_state_indicators,
)


def test_volume_validation_catches_mismatch():
    state = WindowState(symbol="BTCUSDT", window_number=1)
    state.volume.total = 5.986
    state.volume.buy = 0.0  # Bug que queremos detectar
    state.volume.sell = 0.0
    state.volume.delta = 0.463

    errors = state.volume.validate()
    assert len(errors) >= 1
    assert "VOLUME_SUM_MISMATCH" in errors[0]


def test_volume_validation_passes_correct():
    state = WindowState(symbol="BTCUSDT", window_number=1)
    state.volume.total = 5.986
    state.volume.buy = 3.225
    state.volume.sell = 2.761
    state.volume.delta = 0.464

    errors = state.volume.validate()
    assert len(errors) == 0


def test_rsi_validation():
    state = WindowState(symbol="BTCUSDT", window_number=1)
    state.indicators.rsi = 95.0  # Válido mas extremo
    state.indicators.realized_vol = 0.03  # Precisa ser > 0 para validação
    errors = state.indicators.validate()
    assert len(errors) == 0

    state.indicators.rsi = 150.0  # Inválido
    errors = state.indicators.validate()
    assert any("RSI_OUT_OF_RANGE" in e for e in errors)


def test_macro_validation_catches_etf_price():
    state = WindowState(symbol="BTCUSDT", window_number=1)
    state.macro.sp500 = 670.0  # Preço de ETF, não índice
    errors = state.macro.validate()
    assert any("SP500_SUSPICIOUS" in e for e in errors)


def test_negative_fees_auto_corrected():
    state = WindowState(symbol="BTCUSDT", window_number=1)
    state.onchain.total_fees_btc_24h = -381.25
    errors = state.onchain.validate()
    assert any("NEGATIVE_FEES" in e for e in errors)
    assert state.onchain.total_fees_btc_24h is None  # Auto-corrigido


def test_ml_features_uses_correct_field_names():
    state = WindowState(symbol="BTCUSDT", window_number=1)
    state.price.close = 70544.7
    state.price.open = 70530.0
    state.indicators.rsi = 64.65
    state.indicators.bb_width = 0.000185
    state.volume.buy = 3.225
    state.volume.sell = 2.761

    features = state.get_ml_features()

    # Verificar nomes corretos
    assert 'bb_width' in features    # underscore SIMPLES
    assert 'bb__width' not in features  # NÃO duplo
    assert 'rsi' in features
    assert features['rsi'] == 64.65  # Valor REAL, não default
    assert features['bb_width'] == 0.000185  # Valor REAL


def test_state_manager_history():
    mgr = StateManager()

    s1 = mgr.new_window(1)
    s1.price.close = 70500.0
    s1.volume.total = 3.0
    s1.volume.buy = 2.0
    s1.volume.sell = 1.0
    s1.volume.delta = 1.0
    s1.indicators.rsi = 64.7
    s1.indicators.realized_vol = 0.03
    s1.indicators.bb_width = 0.001

    s2 = mgr.new_window(2)
    s2.price.close = 70544.7

    prev = mgr.get_previous_state(1)
    assert prev is not None
    assert prev.price.close == 70500.0


def test_writer_tracking():
    state = WindowState(symbol="BTCUSDT", window_number=1)

    state.mark_written('pipeline')
    state.mark_written('indicators')
    state.mark_written('orderbook')

    # Módulos críticos escreveram — sem erro
    state.volume.total = 1.0
    state.volume.buy = 0.6
    state.volume.sell = 0.4
    state.volume.delta = 0.2
    state.indicators.rsi = 50.0
    state.indicators.realized_vol = 0.03
    state.indicators.bb_width = 0.001

    errors = state.validate_all()
    missing = [e for e in errors if "MISSING_WRITER" in e]
    assert len(missing) == 0


def test_populate_window_state_from_pipeline():
    """Testa que _populate_window_state preenche corretamente a partir de dados do pipeline."""
    ws = WindowState(symbol="BTCUSDT", window_number=5)

    enriched = {
        "ohlc": {"open": 70530.0, "high": 70600.0, "low": 70480.0, "close": 70544.7, "vwap": 70540.0},
        "volume_total": 5.986,
        "num_trades": 1200,
    }
    flow_metrics = {
        "cvd": 0.464,
        "flow_imbalance": 0.15,
        "buy_sell_ratio": 1.17,
        "pressure_label": "SLIGHT_BUY",
    }
    ob_event = {
        "bid_depth_usd": 500000.0,
        "ask_depth_usd": 450000.0,
        "imbalance": 0.05,
        "spread_bps": 1.2,
        "is_valid": True,
        "data_quality": {"data_source": "live"},
    }
    macro_context = {
        "external": {"dxy": 104.5, "sp500": 5800.0, "vix": 14.2, "fear_greed": 72},
        "derivatives": {"btc_funding_rate": 0.0001, "btc_long_short_ratio": 1.15},
    }

    _populate_window_state(ws, enriched, flow_metrics, ob_event, macro_context, 3.225, 2.761)

    # Price
    assert ws.price.close == 70544.7
    assert ws.price.vwap == 70540.0

    # Volume (buy + sell = total within tolerance)
    assert ws.volume.buy == 3.225
    assert ws.volume.sell == 2.761
    assert abs(ws.volume.delta - 0.464) < 0.001

    # OrderBook
    assert ws.orderbook.bid_depth_usd == 500000.0
    assert ws.orderbook.data_source == "live"

    # Flow
    assert ws.flow.cvd == 0.464
    assert ws.flow.pressure_label == "SLIGHT_BUY"

    # Macro
    assert ws.macro.dxy == 104.5
    assert ws.macro.vix == 14.2

    # Derivatives
    assert ws.derivatives.btc_funding_rate == 0.0001

    # Writers: pipeline, orderbook, flow, macro, derivatives marcados
    assert ws._writers["pipeline"] is True
    assert ws._writers["orderbook"] is True
    assert ws._writers["flow"] is True
    assert ws._writers["macro"] is True
    assert ws._writers["derivatives"] is True


def test_populate_indicators_from_feature_calc():
    """Testa que _populate_window_state_indicators preenche indicadores."""
    ws = WindowState(symbol="BTCUSDT", window_number=5)

    computed = {
        "rsi": 64.65,
        "bb_upper": 70600.0,
        "bb_lower": 70400.0,
        "bb_width": 0.000285,
        "atr": 150.0,
        "realized_vol": 0.032,
    }
    mtf = {"1d": {"realized_vol": 0.035}}

    _populate_window_state_indicators(ws, computed, mtf)

    assert ws.indicators.rsi == 64.65
    assert ws.indicators.bb_width == 0.000285
    # realized_vol prefers 1d multi_tf
    assert ws.indicators.realized_vol == 0.035
    assert ws.indicators.realized_vol_source == "1d"
    assert ws._writers["indicators"] is True


def test_full_window_state_validates_after_populate():
    """Testa que WindowState valida corretamente após populate completo."""
    ws = WindowState(symbol="BTCUSDT", window_number=10)

    enriched = {
        "ohlc": {"open": 70530.0, "high": 70600.0, "low": 70480.0, "close": 70544.7, "vwap": 70540.0},
        "volume_total": 5.986,
        "num_trades": 1200,
    }
    _populate_window_state(
        ws, enriched,
        flow_metrics={"cvd": 0.464},
        ob_event={"bid_depth_usd": 500000, "ask_depth_usd": 450000, "imbalance": 0.05},
        macro_context={"external": {"dxy": 104.5, "sp500": 5800.0}, "derivatives": {}},
        total_buy_volume=3.225, total_sell_volume=2.761,
    )
    _populate_window_state_indicators(
        ws, {"rsi": 64.65, "bb_width": 0.000285}, {"1d": {"realized_vol": 0.035}},
    )

    errors = ws.validate_all()
    # volume.total != buy+sell (5.986 != 3.225+2.761=5.986) — should be OK
    assert not any("MISSING_WRITER" in e for e in errors)
    # All critical writers present
    assert ws._writers["pipeline"] is True
    assert ws._writers["indicators"] is True
    assert ws._writers["orderbook"] is True
