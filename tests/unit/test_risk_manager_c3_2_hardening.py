# tests/unit/test_risk_manager_c3_2_hardening.py
import math
from datetime import date, datetime, timezone, timedelta
from unittest.mock import patch
import pytest

from risk_management.risk_manager import (
    RiskConfig,
    RiskManager,
    TradeRequest,
    Position,
)

@pytest.fixture
def risk_mgr() -> RiskManager:
    config = RiskConfig(
        max_position_size=100_000.0,
        max_daily_loss=0.05,
        initial_capital=100_000.0,
        max_loss_per_trade=0.02,
        max_open_positions=5,
    )
    return RiskManager(config)

def _make_req(**kwargs) -> TradeRequest:
    defaults = {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'size': 1.0,
        'price': 50_000.0,
        'stop_loss': 49_500.0,
        'take_profit': 51_000.0,
        'strategy': 'momentum',
        'confidence': 0.8,
    }
    defaults.update(kwargs)
    return TradeRequest(**defaults)

def test_01_price_zero_rejected_without_exception(risk_mgr: RiskManager) -> None:
    req = _make_req(price=0.0)
    res = risk_mgr.check_trade_request(req)
    assert res['approved'] is False
    assert res['reason'] == 'invalid price'
    assert res['max_size'] == 0

def test_02_price_negative_rejected(risk_mgr: RiskManager) -> None:
    req = _make_req(price=-50_000.0)
    res = risk_mgr.check_trade_request(req)
    assert res['approved'] is False
    assert res['reason'] == 'invalid price'
    assert res['max_size'] == 0

def test_03_price_nan_rejected(risk_mgr: RiskManager) -> None:
    req = _make_req(price=float('nan'))
    res = risk_mgr.check_trade_request(req)
    assert res['approved'] is False
    assert res['reason'] == 'invalid price'
    assert res['max_size'] == 0

def test_04_price_inf_rejected(risk_mgr: RiskManager) -> None:
    req_pos = _make_req(price=float('inf'))
    res_pos = risk_mgr.check_trade_request(req_pos)
    assert res_pos['approved'] is False
    assert res_pos['reason'] == 'invalid price'
    assert res_pos['max_size'] == 0

    req_neg = _make_req(price=float('-inf'))
    res_neg = risk_mgr.check_trade_request(req_neg)
    assert res_neg['approved'] is False
    assert res_neg['reason'] == 'invalid price'
    assert res_neg['max_size'] == 0

def test_04b_price_bool_rejected(risk_mgr: RiskManager) -> None:
    req = _make_req(price=True)
    res = risk_mgr.check_trade_request(req)
    assert res['approved'] is False
    assert res['reason'] == 'invalid price'
    assert res['max_size'] == 0

def test_05_size_zero_rejected(risk_mgr: RiskManager) -> None:
    req = _make_req(size=0.0)
    res = risk_mgr.check_trade_request(req)
    assert res['approved'] is False
    assert res['reason'] == 'invalid size'
    assert res['max_size'] == 0

def test_06_size_negative_rejected(risk_mgr: RiskManager) -> None:
    req = _make_req(size=-1.5)
    res = risk_mgr.check_trade_request(req)
    assert res['approved'] is False
    assert res['reason'] == 'invalid size'
    assert res['max_size'] == 0

def test_07_size_nan_inf_rejected(risk_mgr: RiskManager) -> None:
    for bad_size in (float('nan'), float('inf'), float('-inf'), True, False):
        req = _make_req(size=bad_size)
        res = risk_mgr.check_trade_request(req)
        assert res['approved'] is False
        assert res['reason'] == 'invalid size'
        assert res['max_size'] == 0

def test_08_stop_loss_nan_inf_negative_rejected(risk_mgr: RiskManager) -> None:
    for bad_sl in (float('nan'), float('inf'), float('-inf'), -100.0, True):
        req = _make_req(stop_loss=bad_sl)
        res = risk_mgr.check_trade_request(req)
        assert res['approved'] is False
        assert res['reason'] == 'invalid stop loss'
        assert res['max_size'] == 0

def test_09_take_profit_nan_inf_negative_rejected(risk_mgr: RiskManager) -> None:
    for bad_tp in (float('nan'), float('inf'), float('-inf'), -500.0, True):
        req = _make_req(take_profit=bad_tp)
        res = risk_mgr.check_trade_request(req)
        assert res['approved'] is False
        assert res['reason'] == 'invalid take profit'
        assert res['max_size'] == 0

def test_10_side_invalid_rejected(risk_mgr: RiskManager) -> None:
    for bad_side in ('HOLD', 'LONG', 'SHORT', 'NEUTRAL', '', None, 123, True):
        req = _make_req(side=bad_side)
        res = risk_mgr.check_trade_request(req)
        assert res['approved'] is False
        assert res['reason'] == 'invalid side'
        assert res['max_size'] == 0

def test_11_valid_buy_approved(risk_mgr: RiskManager) -> None:
    req = _make_req(side='BUY', size=1.0, price=50_000.0, stop_loss=49_500.0)
    res = risk_mgr.check_trade_request(req)
    assert res['approved'] is True
    assert res['reason'] == 'approved'
    assert res['max_size'] == pytest.approx(100_000.0 / 50_000.0)

def test_12_valid_sell_approved(risk_mgr: RiskManager) -> None:
    req = _make_req(side='SELL', size=1.0, price=50_000.0, stop_loss=50_500.0)
    res = risk_mgr.check_trade_request(req)
    assert res['approved'] is True
    assert res['reason'] == 'approved'
    assert res['max_size'] == pytest.approx(100_000.0 / 50_000.0)

def test_13_stop_loss_zero_and_none_preserved(risk_mgr: RiskManager) -> None:
    req_zero = _make_req(stop_loss=0.0)
    res_zero = risk_mgr.check_trade_request(req_zero)
    assert res_zero['approved'] is True
    assert res_zero['reason'] == 'approved'

    req_none = _make_req(stop_loss=None)
    res_none = risk_mgr.check_trade_request(req_none)
    assert res_none['approved'] is True
    assert res_none['reason'] == 'approved'

    req_tp_zero = _make_req(take_profit=0.0)
    res_tp_zero = risk_mgr.check_trade_request(req_tp_zero)
    assert res_tp_zero['approved'] is True

    req_tp_none = _make_req(take_profit=None)
    res_tp_none = risk_mgr.check_trade_request(req_tp_none)
    assert res_tp_none['approved'] is True

def test_14_utc_daily_reset_on_date_boundary(risk_mgr: RiskManager) -> None:
    risk_mgr.daily_pnl = -2500.0
    initial_utc_date = risk_mgr._last_reset_date

    with patch.object(RiskManager, '_get_utc_date', return_value=initial_utc_date):
        risk_mgr._check_daily_reset()
        assert risk_mgr.daily_pnl == -2500.0
        assert risk_mgr._last_reset_date == initial_utc_date

    next_utc_date = initial_utc_date + timedelta(days=1)
    with patch.object(RiskManager, '_get_utc_date', return_value=next_utc_date):
        risk_mgr._check_daily_reset()
        assert risk_mgr.daily_pnl == 0.0
        assert risk_mgr._last_reset_date == next_utc_date

def test_15_utc_reset_independent_of_local_machine_timezone(risk_mgr: RiskManager) -> None:
    local_fake_date = date(2026, 10, 3)
    utc_fake_date = date(2026, 10, 4)

    risk_mgr._last_reset_date = local_fake_date
    risk_mgr.daily_pnl = -1200.0

    fake_utc_now = datetime(2026, 10, 4, 0, 5, 0, tzinfo=timezone.utc)
    with patch('risk_management.risk_manager.datetime') as mock_dt:
        mock_dt.now.side_effect = lambda tz=None: fake_utc_now if tz == timezone.utc else datetime(2026, 10, 3, 21, 5, 0)
        computed_date = RiskManager._get_utc_date()
        assert computed_date == utc_fake_date

        req = _make_req()
        risk_mgr.check_trade_request(req)

        assert risk_mgr.daily_pnl == 0.0
        assert risk_mgr._last_reset_date == utc_fake_date

def test_15b_reset_daily_pnl_method_uses_utc(risk_mgr: RiskManager) -> None:
    risk_mgr.daily_pnl = 5000.0
    target_utc_date = date(2026, 12, 31)
    with patch.object(RiskManager, '_get_utc_date', return_value=target_utc_date):
        risk_mgr.reset_daily_pnl()
        assert risk_mgr.daily_pnl == 0.0
        assert risk_mgr._last_reset_date == target_utc_date

def test_16_max_size_never_divides_by_zero_on_invalid_input(risk_mgr: RiskManager) -> None:
    bad_inputs = [
        {'price': 0.0},
        {'price': -1.0},
        {'price': float('nan')},
        {'price': float('inf')},
        {'size': 0.0},
        {'size': -5.0},
        {'side': 'INVALID'},
    ]
    for kwargs in bad_inputs:
        req = _make_req(**kwargs)
        res = risk_mgr.check_trade_request(req)
        assert res['approved'] is False
        assert res['max_size'] == 0
        assert not math.isinf(res['max_size'])
        assert not math.isnan(res['max_size'])

def test_17_position_pnl_percentage_safe_against_zero_division() -> None:
    pos_zero_price = Position(
        symbol='BTCUSDT',
        side='BUY',
        size=1.0,
        entry_price=0.0,
        current_price=50_000.0,
    )
    assert pos_zero_price.pnl_percentage == 0.0

    pos_zero_size = Position(
        symbol='BTCUSDT',
        side='BUY',
        size=0.0,
        entry_price=50_000.0,
        current_price=51_000.0,
    )
    assert pos_zero_size.pnl_percentage == 0.0


def test_18_malformed_object_handled_gracefully(risk_mgr: RiskManager) -> None:
    class DummyMissingAttrs:
        pass
    dummy = DummyMissingAttrs()
    res = risk_mgr.check_trade_request(dummy)
    assert res['approved'] is False
    assert res['reason'] in ('invalid side', 'invalid price', 'invalid size')
    assert res['max_size'] == 0
