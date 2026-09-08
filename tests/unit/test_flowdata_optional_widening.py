# tests/unit/test_flowdata_optional_widening.py
"""
P0-3b: FlowData Optional widening (commit separado).

Autorizado:
  flow_imbalance: Optional[float]
  buy_sell_ratio: Optional[float]
  pressure_label: Optional[str]
  + equivalentes Deriv (LSR/OI) se mesmo padrão.

Condições:
  - validate() aceita None explicitamente;
  - default do dataclass passa a ser None, não 0.0/1.0/NEUTRAL;
  - todo leitor de ws.flow/ws.deriv tolera None sem crash;
  - teste: aninhado ausente -> None, não defaults fabricados;
  - teste: valores reais (inclusive 0.0 e 1.0 legítimos) preservados.
  - sem ampliar para outros campos.
"""

from core.window_state import DerivativesData, FlowData, WindowState
from market_orchestrator.windows.window_processor import _populate_window_state


def _enriched():
    return {"ohlc": {"close": 70544.7, "open": 70500.0, "high": 70600.0,
                     "low": 70400.0, "vwap": 70540.0},
            "volume_total": 5.986, "volume_compra": 3.225,
            "volume_venda": 2.761}


def test_defaults_are_none_not_fabricated():
    f = FlowData()
    assert f.flow_imbalance is None, "default fabricado 0.0"
    assert f.buy_sell_ratio is None, "default fabricado 1.0"
    assert f.pressure_label is None, "default fabricado NEUTRAL"
    d = DerivativesData()
    assert d.btc_long_short_ratio is None, "LSR default fabricado 1.0"
    assert d.eth_long_short_ratio is None, "LSR default fabricado 1.0"
    assert d.btc_open_interest is None, "OI default fabricado 0.0"
    assert d.eth_open_interest is None, "OI default fabricado 0.0"
    # funding já era Optional None (precedente)
    assert d.btc_funding_rate is None
    assert d.eth_funding_rate is None


def test_validate_accepts_none_explicitly():
    assert FlowData().validate() == []
    assert DerivativesData().validate() == []
    # None explícito também válido quando outros campos preenchidos
    f = FlowData(flow_imbalance=None, buy_sell_ratio=None,
                 pressure_label=None)
    assert f.validate() == []
    d = DerivativesData(btc_open_interest=None,
                        btc_long_short_ratio=None,
                        eth_open_interest=None,
                        eth_long_short_ratio=None)
    assert d.validate() == []


def test_validate_rejects_nonfinite_out_of_range():
    assert FlowData(flow_imbalance=2.0).validate() != []
    assert FlowData(flow_imbalance=float("nan")).validate() != []
    assert FlowData(buy_sell_ratio=-0.5).validate() != []
    assert FlowData(buy_sell_ratio=float("inf")).validate() != []
    assert FlowData(pressure_label="").validate() != []
    assert DerivativesData(btc_open_interest=-1.0).validate() != []
    assert DerivativesData(btc_long_short_ratio=float("nan")).validate() != []
    # ...mas legítimos passam
    assert FlowData(flow_imbalance=0.0, buy_sell_ratio=1.0,
                    pressure_label="NEUTRAL").validate() == []
    assert DerivativesData(btc_open_interest=0.0,
                           btc_long_short_ratio=1.0).validate() == []


def test_missing_nested_yields_none():
    ws = WindowState()
    _populate_window_state(ws, _enriched(), {"cvd": 0.1}, {}, {}, 1.0, 1.0)
    assert ws.flow.flow_imbalance is None, "ausente virou 0.0 fabricado"
    assert ws.flow.buy_sell_ratio is None, "ausente virou 1.0 fabricado"
    assert ws.flow.pressure_label is None, "ausente virou NEUTRAL fabricado"
    assert ws.derivatives.btc_long_short_ratio is None
    assert ws.derivatives.btc_open_interest is None
    assert ws.derivatives.eth_long_short_ratio is None
    assert ws.derivatives.eth_open_interest is None
    # validate_all não deve reclamar de None (ausência explícita)
    assert ws.flow.validate() == []
    assert ws.derivatives.validate() == []


def test_none_ratio_stays_none():
    """buy_only (ratio None + pressure None) não fabrica 1.0/NEUTRAL."""
    ws = WindowState()
    fm = {"cvd": 0.2,
          "order_flow": {"flow_imbalance": 0.35,
                         "buy_sell_ratio": {"buy_sell_ratio": None,
                                            "pressure": None,
                                            "ratio_state": "buy_only"}}}
    _populate_window_state(ws, _enriched(), fm, {}, {}, 3.0, 2.0)
    assert ws.flow.flow_imbalance == 0.35
    assert ws.flow.buy_sell_ratio is None
    assert ws.flow.pressure_label is None


def test_real_values_preserved_including_zero_and_one():
    # 0.0 legítimo (sell_only / equilíbrio) e 1.0 legítimo (equilibrado)
    for fi, r, pl in [(0.0, 0.0, "STRONG_SELL"),
                      (0.0, 1.0, "NEUTRAL"),
                      (1.0, 1.0, "NEUTRAL"),
                      (-1.0, 2.5, "STRONG_BUY")]:
        ws = WindowState()
        fm = {"cvd": 0.1,
              "order_flow": {"flow_imbalance": fi,
                             "buy_sell_ratio": {"buy_sell_ratio": r,
                                                "pressure": pl}}}
        _populate_window_state(ws, _enriched(), fm, {}, {}, 1.0, 1.0)
        assert ws.flow.flow_imbalance == fi, f"{fi} legítimo perdido"
        assert ws.flow.buy_sell_ratio == r, f"{r} legítimo perdido"
        assert ws.flow.pressure_label == pl, f"{pl} legítimo perdido"
        assert ws.flow.validate() == []


def test_deriv_real_zero_one_preserved():
    ws = WindowState()
    macro = {"external": {},
             "derivatives": {
                 "BTCUSDT": {"funding_rate_percent": 0.0001,
                             "open_interest": 0.0,
                             "long_short_ratio": 1.0},
                 "ETHUSDT": {"funding_rate_percent": -0.0002,
                             "open_interest": 12345.0,
                             "long_short_ratio": 0.0},
             }}
    _populate_window_state(ws, _enriched(), {"cvd": 0.1}, {}, macro,
                           1.0, 1.0)
    assert ws.derivatives.btc_open_interest == 0.0, "OI 0.0 legítimo perdido"
    assert ws.derivatives.btc_long_short_ratio == 1.0, "LSR 1.0 legítimo perdido"
    assert ws.derivatives.eth_open_interest == 12345.0
    assert ws.derivatives.eth_long_short_ratio == 0.0, "LSR 0.0 legítimo perdido"
    assert ws.derivatives.validate() == []


def test_readers_tolerate_none_without_crash():
    """Mapeamento: únicos leitores produtivos são to_summary/get_ml_features/
    validate_all — nenhum dereferencia Optional sem guarda."""
    ws = WindowState()
    _populate_window_state(ws, _enriched(), {"cvd": 0.1}, {}, {}, 1.0, 1.0)
    # Nenhum desses pode dar TypeError com None
    s = ws.to_summary()
    assert isinstance(s, dict)
    feats = ws.get_ml_features()
    assert isinstance(feats, dict)
    # validate_all inclui flow/deriv e aceita None
    errs = ws.validate_all()
    assert not any("FLOW_IMBALANCE" in e or "BUY_SELL_RATIO" in e
                   or "PRESSURE_LABEL" in e for e in errs)
    assert not any("OPEN_INTEREST" in e or "LONG_SHORT_RATIO" in e
                   for e in errs)
    # logging-style formatting manual deve usar guarda, não crash:
    # (prova que formatação direta de None quebraria — leitores devem evitar)
    try:
        _ = f"{ws.flow.flow_imbalance:.2f}"
        direct_formats = True
    except TypeError:
        direct_formats = False
    assert direct_formats is False, "None não formata como float (leitores devem guardar)"
