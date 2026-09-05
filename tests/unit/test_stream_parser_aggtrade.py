import pytest
from market_orchestrator.market_orchestrator import parse_trade_message


def test_stream_parser_aggtrade_vs_spot():
    msg_futures = {
        "e": "aggTrade",
        "E": 1788306540123,
        "s": "BTCUSDT",
        "a": 123456789,
        "p": "77550.30",
        "q": "0.512",
        "f": 900000001,
        "l": 900000004,
        "T": 1788306540120,
        "m": False,
    }
    msg_spot = {
        "e": "trade",
        "E": 1788306540123,
        "s": "BTCUSDT",
        "t": 555555,
        "p": "77550.30",
        "q": "0.512",
        "b": 111,
        "a": 222,
        "T": 1788306540120,
        "m": False,
        "M": True,
    }

    res_fut = parse_trade_message(msg_futures)
    res_spot = parse_trade_message(msg_spot)

    assert res_fut is not None
    assert res_spot is not None

    # - ambas produzem o mesmo (price, qty, ts_ms, is_buyer_maker)
    assert (
        res_fut["price"],
        res_fut["qty"],
        res_fut["ts_ms"],
        res_fut["is_buyer_maker"],
    ) == (
        res_spot["price"],
        res_spot["qty"],
        res_spot["ts_ms"],
        res_spot["is_buyer_maker"],
    )

    # - fixture 1: source == "fut_agg", trade_id == 123456789
    assert res_fut["source"] == "fut_agg"
    assert res_fut["trade_id"] == 123456789

    # - fixture 2: source == "spot_trade", trade_id == 555555 (e NÃO 222)
    assert res_spot["source"] == "spot_trade"
    assert res_spot["trade_id"] == 555555
    assert res_spot["trade_id"] != 222


def test_stream_parser_rejections():
    # - mensagem com q="0" ou sem T é rejeitada
    msg_zero_q = {
        "e": "aggTrade",
        "E": 1788306540123,
        "s": "BTCUSDT",
        "a": 123456789,
        "p": "77550.30",
        "q": "0",
        "T": 1788306540120,
        "m": False,
    }
    assert parse_trade_message(msg_zero_q) is None

    msg_missing_t = {
        "e": "aggTrade",
        "E": 1788306540123,
        "s": "BTCUSDT",
        "a": 123456789,
        "p": "77550.30",
        "q": "0.512",
        "m": False,
    }
    assert parse_trade_message(msg_missing_t) is None

    msg_zero_t = {
        "e": "aggTrade",
        "E": 1788306540123,
        "s": "BTCUSDT",
        "a": 123456789,
        "p": "77550.30",
        "q": "0.512",
        "T": 0,
        "m": False,
    }
    assert parse_trade_message(msg_zero_t) is None
