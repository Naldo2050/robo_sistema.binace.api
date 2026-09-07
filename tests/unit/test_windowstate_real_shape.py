# tests/unit/test_windowstate_real_shape.py
"""
B-P0-3: WindowState lê o schema REAL produzido (semav defaults fantasmas).

Antes: _populate_window_state lia fm.get("flow_imbalance"/"buy_sell_ratio"/
"pressure_label") no nível TOP de flow_metrics, mas o produtor emite
aninhado em order_flow (e pressure dentro de buy_sell_ratio) => ws.flow
permanentemente 0/1.0/NEUTRAL. Idem derivatives (por símbolo) e macro
-external (dicts {"preco_atual": ...}).

Contrato:
  - shape produtivo real => valores corretos no WindowState;
  - ratio None (P0-1) não quebra o leitor (guarda finita);
  - aninhado ausente => assignment pulado (sem crash).
"""

from core.window_state import WindowState
from market_orchestrator.windows.window_processor import (
    _populate_window_state,
)


def _enriched():
    return {"ohlc": {"close": 70544.7, "open": 70500.0, "high": 70600.0,
                     "low": 70400.0, "vwap": 70540.0},
            "volume_total": 5.986, "volume_compra": 3.225,
            "volume_venda": 2.761}


def _real_flow_metrics():
    return {
        "cvd": 0.464,
        "order_flow": {
            "flow_imbalance": 0.35,
            "buy_sell_ratio": {"buy_sell_ratio": 1.5,
                               "pressure": "MODERATE_BUY",
                               "flow_trend": "accelerating_buying"},
            "aggressive_buy_pct": 60.0,
            "aggressive_sell_pct": 40.0,
        },
        "sector_flow": {
            "retail": {"buy": 1.5, "sell": 1.0, "delta": 0.5},
            "mid": {"buy": 0.0, "sell": 0.0, "delta": 0.0},
            "whale": {"buy": 2.0, "sell": 0.0, "delta": 2.0},
        },
    }


def _real_macro():
    return {
        "external": {
            "DXY": {"preco_atual": 98.919, "source": "yfinance"},
            "SP500": {"preco_atual": 5800.0, "source": "yfinance"},
        },
        "derivatives": {
            "BTCUSDT": {"funding_rate_percent": 0.0001,
                        "open_interest": 1000000,
                        "long_short_ratio": 1.15},
        },
    }


def test_real_shape_flow_reaches_windowstate():
    ws = WindowState()
    _populate_window_state(ws, _enriched(), _real_flow_metrics(), {},
                           _real_macro(), 3.225, 2.761)
    assert ws.flow.cvd == 0.464
    assert ws.flow.flow_imbalance == 0.35, "imbalance real perdido"
    assert ws.flow.buy_sell_ratio == 1.5, "ratio real perdido"
    assert ws.flow.pressure_label == "MODERATE_BUY", "pressure real perdido"
    assert ws.flow.retail_buy == 1.5
    assert ws.flow.whale_delta == 2.0


def test_real_shape_macro_derivatives():
    ws = WindowState()
    _populate_window_state(ws, _enriched(), _real_flow_metrics(), {},
                           _real_macro(), 3.225, 2.761)
    assert ws.macro.dxy == 98.919, "dxy real perdido (era dict)"
    assert ws.macro.dxy_source == "yfinance"
    assert ws.derivatives.btc_long_short_ratio == 1.15
    assert ws.derivatives.btc_open_interest == 1000000


def test_none_ratio_does_not_crash_reader():
    """Ratio None (P0-1, janela buy-only) não pode quebrar float(None)."""
    ws = WindowState()
    fm = _real_flow_metrics()
    fm["order_flow"]["buy_sell_ratio"] = {"buy_sell_ratio": None,
                                          "pressure": None,
                                          "ratio_state": "buy_only"}
    _populate_window_state(ws, _enriched(), fm, {}, _real_macro(), 3.0, 2.0)
    assert ws.flow.flow_imbalance == 0.35


def test_missing_nested_skips_assignment():
    """Aninhado ausente => sem crash (defaults do dataclass permanecem)."""
    ws = WindowState()
    _populate_window_state(ws, _enriched(), {"cvd": 0.1}, {}, {}, 1.0, 1.0)
    assert ws.flow.cvd == 0.1
