# tests/golden/test_gw5_slow.py — GW5: consolidação high-activity (SLOW).
#
# 1500 trades em faixa estreita via FlowAnalyzer REAL (throttle 100ms +
# step 100ms = cenário adversarial O(N³) original, ~80s pré-fix).
# Semântica == golden do heatmap + budget GENEROSO (detecta retorno aos ~80s,
# não micro-performance). Sem duplicar o scaling 4000/1000 (já coberto em
# tests/unit/test_heatmap_perf_regression.py).

import time

import pytest

from flow_analyzer.core import FlowAnalyzer

from .conftest import FakeClock, gen_trades, load_fixture

pytestmark = pytest.mark.slow

# Budget folgado: pós-fix ~6-12s nesta máquina; O(N³) de volta => ~80s
# (ou estouro do timeout de 60s do pytest — ambos detectam a regressão).
INGEST_BUDGET_S = 45.0


def test_gw5_high_activity_semantics_and_budget(frozen_state):
    spec = load_fixture("gw5_high_activity.json")
    clock = FakeClock()
    flow = FlowAnalyzer(time_manager=clock)  # defaults prod (janela 2000)
    trades = gen_trades(spec)
    assert len(trades) == 1500
    t0 = time.perf_counter()
    for t in trades:
        clock._now = t["T"]
        flow.process_trade(t)
    dt = time.perf_counter() - t0
    m = flow.get_flow_metrics(reference_epoch_ms=trades[-1]["T"])
    hm = m["liquidity_heatmap"]
    assert hm["scope_type"] == "rolling_trades"
    assert hm["scope_size"] == 2000
    assert len(hm["clusters"]) == 1  # faixa estreita => 1 cluster
    c = hm["clusters"][0]
    assert c["trades_count"] == 1500
    assert c["total_volume"] == pytest.approx(1500 * 0.01)
    assert c["center"] == pytest.approx(65000.0, abs=60.0)
    assert dt < INGEST_BUDGET_S, (
        f"ingestão 1500 levou {dt:.1f}s (budget {INGEST_BUDGET_S}s; "
        f"O(N³) histórico ~80s)")
