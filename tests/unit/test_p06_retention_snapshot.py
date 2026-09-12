# tests/unit/test_p06_retention_snapshot.py
"""P06: retenção precisa cobrir (reference - max_window, reference] no cálculo.

Caso real: close=1789171020000, delay de pipeline ~1273ms; o prune ancorado
na chegada amputava os 4 trades do prefixo (-$133.637,47) antes do snapshot
ancorado no close. Escopo estrito P06 (CVD/setores/fórmulas intactos).
"""
import pytest

from flow_analyzer.core import FlowAnalyzer

CLOSE = 1789171020000
T0 = CLOSE - 15 * 60000
# Prefixo real do caso (offsets e lados; qtys normalizados p/ teste sintético)
PREFIX_OFFSETS = [316, 509, 851, 964, 1246]


def _analyzer(wall):
    fa = FlowAnalyzer()
    fa._get_synced_timestamp_ms = lambda: wall[0]  # type: ignore[method-assign]
    fa.last_reset_ms = T0 - 3600000
    return fa


def _feed(fa, wall, trades):
    """trades: lista (T, price, qty, m). arrival wall = max(wall, T+200ms)."""
    for T, p, q, m in trades:
        wall[0] = max(wall[0], T + 200)
        fa.process_trade({"p": p, "q": q, "T": T, "T_raw": T, "m": m,
                          "source": "fut_agg", "trade_id": T})


def _independent_net(trades, start, end):
    net = 0.0
    for T, p, q, m in trades:
        if start < T <= end:
            net += -q * p if m else q * p
    return net


def _case_trades():
    """Prefixo nos offsets reais + bulk distribuído + 1 trade pós-close."""
    trades = [
        (T0 + 316, 77189.9, 0.497, True),
        (T0 + 509, 77190.0, 0.162, False),
        (T0 + 851, 77190.0, 0.001, False),
        (T0 + 964, 77190.0, 1.398, True),
        (T0 + 1246, 77190.0, 0.002, False),
    ]
    t = T0 + 5000
    i = 0
    while t <= CLOSE - 1000:
        trades.append((t, 77100.0 + (i % 50), 0.05 + (i % 7) * 0.01, bool(i % 2)))
        t += 4500
        i += 1
    trades.append((CLOSE + 424, 77167.0, 0.016, False))  # pós-close, antes do snapshot
    return trades


def test_exact_case_1273ms_prefix_preserved():
    wall = [T0]
    fa = _analyzer(wall)
    trades = _case_trades()
    # alimenta até o close; snapshot ocorre com delay de 1273ms (1 trade pós-close chega antes)
    pre = [t for t in trades if t[0] <= CLOSE]
    post = [t for t in trades if t[0] > CLOSE]
    _feed(fa, wall, pre)
    wall[0] = CLOSE + 1273  # pipeline delay real do caso
    _feed(fa, wall, post)
    m = fa.get_flow_metrics(reference_epoch_ms=CLOSE)
    of = m["order_flow"]
    assert of["net_flow_15m"] == pytest.approx(_independent_net(trades, T0, CLOSE), abs=0.05)
    integ = m["flow_window_integrity"]["15m"]
    assert integ["status"] == "FULL"
    assert integ["is_temporal_coverage_valid"] is True


def _history_trades():
    """Histórico anterior à janela (estabelece first_trade_ts <= start)."""
    return [(T0 - 600000 + i * 30000, 77000.0, 0.05, bool(i % 2)) for i in range(20)]


def test_without_grace_prefix_is_lost():
    """Sensibilidade: sem margem, o delay amputa o prefixo (mecanismo do bug)."""
    wall = [T0 - 600000]
    fa = _analyzer(wall)
    fa.flow_retention_grace_ms = 0
    trades = _case_trades()
    _feed(fa, wall, _history_trades() + [t for t in trades if t[0] <= CLOSE])
    wall[0] = CLOSE + 1273
    _feed(fa, wall, [t for t in trades if t[0] > CLOSE])
    m = fa.get_flow_metrics(reference_epoch_ms=CLOSE)
    of = m["order_flow"]
    # sem margem o prefixo (<1273ms) é podado: diverge do independente
    assert of["net_flow_15m"] != pytest.approx(_independent_net(trades, T0, CLOSE), abs=0.05)
    assert m["flow_window_integrity"]["15m"]["status"] != "FULL"


@pytest.mark.parametrize("delay_ms", [0, 500, 1273, 2000, 5000, 30000, 120000])
def test_pipeline_lag_within_grace_stays_exact(delay_ms):
    wall = [T0]
    fa = _analyzer(wall)
    trades = _case_trades()
    _feed(fa, wall, [t for t in trades if t[0] <= CLOSE])
    wall[0] = CLOSE + delay_ms
    _feed(fa, wall, [t for t in trades if CLOSE < t[0] <= CLOSE + delay_ms])
    m = fa.get_flow_metrics(reference_epoch_ms=CLOSE)
    of = m["order_flow"]
    assert of["net_flow_15m"] == pytest.approx(_independent_net(trades, T0, CLOSE), abs=0.05)
    assert m["flow_window_integrity"]["15m"]["status"] == "FULL"


def test_lag_beyond_grace_fails_closed_not_full():
    wall = [T0 - 600000]
    fa = _analyzer(wall)
    trades = _case_trades()
    _feed(fa, wall, _history_trades() + [t for t in trades if t[0] <= CLOSE])
    wall[0] = CLOSE + 300000  # >> grace de 120s
    _feed(fa, wall, [t for t in trades if CLOSE < t[0] <= CLOSE + 300000]
          + [(CLOSE + 300000, 77100.0, 0.1, True)])
    m = fa.get_flow_metrics(reference_epoch_ms=CLOSE)
    integ = m["flow_window_integrity"]["15m"]
    assert integ["status"] == "TRUNCATED"
    assert integ["is_temporal_coverage_valid"] is False


@pytest.mark.parametrize("delay_ms", [0, 1273, 5000, 30000])
def test_all_timeframes_match_independent_selection(delay_ms):
    wall = [T0]
    fa = _analyzer(wall)
    trades = _case_trades()
    _feed(fa, wall, [t for t in trades if t[0] <= CLOSE])
    wall[0] = CLOSE + delay_ms
    _feed(fa, wall, [t for t in trades if CLOSE < t[0] <= CLOSE + delay_ms])
    m = fa.get_flow_metrics(reference_epoch_ms=CLOSE)
    of = m["order_flow"]
    for xm, key in ((1, "net_flow_1m"), (5, "net_flow_5m"), (15, "net_flow_15m")):
        assert of[key] == pytest.approx(
            _independent_net(trades, CLOSE - xm * 60000, CLOSE), abs=0.05)


def test_high_rate_separates_time_from_capacity():
    wall = [T0]
    fa = _analyzer(wall)
    trades = []
    t = T0 + 10
    i = 0
    while t <= CLOSE and len(trades) < 12000:
        trades.append((t, 77000.0, 0.01, bool(i % 2)))
        t += 75  # ~13.3 tps -> 12k trades em 15m
        i += 1
    _feed(fa, wall, trades)
    wall[0] = CLOSE + 1500
    _feed(fa, wall, [(CLOSE + 500, 77000.0, 0.01, True)])
    m = fa.get_flow_metrics(reference_epoch_ms=CLOSE)
    of = m["order_flow"]
    assert of["net_flow_15m"] == pytest.approx(_independent_net(trades, T0, CLOSE), abs=1.0)
    assert m["flow_window_integrity"]["15m"]["status"] == "FULL"


def test_capacity_eviction_still_detected():
    wall = [T0]
    fa = _analyzer(wall)
    fa.flow_trades_maxlen = 100
    #тики densos: 3000 trades x 300ms cobrem a janela, mas o cap 100 só retém ~30s
    trades = [(T0 + i * 300, 77000.0, 0.01, False) for i in range(0, 3000)]
    _feed(fa, wall, trades)
    m = fa.get_flow_metrics(reference_epoch_ms=CLOSE)
    statuses = {k: v["status"] for k, v in m["flow_window_integrity"].items()}
    assert statuses["15m"] == "CAPACITY_TRUNCATED"


def test_cvd_and_sectors_untouched_by_retention():
    wall = [T0]
    fa = _analyzer(wall)
    trades = _case_trades()
    _feed(fa, wall, trades)
    wall[0] = CLOSE + 1273
    _feed(fa, wall, [(CLOSE + 424, 77167.0, 0.016, False)])
    m = fa.get_flow_metrics(reference_epoch_ms=CLOSE)
    exp_cvd = sum((q if not mm else -q) for _, _, q, mm in trades) + 0.016
    assert float(m["cvd"]) == pytest.approx(exp_cvd, abs=1e-9)
