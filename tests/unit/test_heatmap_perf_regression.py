# tests/unit/test_heatmap_perf_regression.py
"""
H5 — trava de regressão generosa (não frágil) contra retorno do O(N²)/O(N³).

Não usa threshold de microssegundos: margens de 2-10x sobre o pós-fix.
- single N=1000 vs N=4000: razão < 8 (O(N²) dava ~13x; O(N log N) dá ~4x).
- single N=4000 absoluto < 2s (pós-fix ~17ms; pré-fix ~480ms).
- loop add_trade N=1000 step 100ms < 8s (pré-fix ~15s; pós-fix ~2.6s).

Se np.mean O(k) voltar ao laço, a razão e/ou o absoluto estouram.
"""

import math
import random
import time

from market_analysis.liquidity_heatmap import LiquidityHeatmap


def _price(i: int) -> float:
    r = random.Random(i).random()
    return 65000.0 + 30 * math.sin(i / 50.0) + (r - 0.5) * 2


def _single_ms(n: int) -> float:
    hm = LiquidityHeatmap(window_size=max(n, 4000),
                          cluster_threshold_pct=0.003,
                          min_trades_per_cluster=5,
                          update_interval_ms=10 ** 12)
    for i in range(n):
        hm.price_levels.append(float(_price(i)))
        hm.volume_levels.append(0.01)
        hm.side_levels.append("buy" if i % 2 == 0 else "sell")
        hm.timestamp_levels.append(1_700_000_000_000 + i * 100)
    t = time.perf_counter()
    hm._update_clusters()
    return (time.perf_counter() - t) * 1000


def test_single_scales_subquadratically():
    t1000 = _single_ms(1000)
    t4000 = _single_ms(4000)
    assert t4000 < 2000.0, f"N=4000 single {t4000:.1f}ms excede budget generoso"
    ratio = t4000 / max(t1000, 1e-6)
    assert ratio < 8.0, f"escala {ratio:.1f}x (1000:{t1000:.1f}ms -> 4000:{t4000:.1f}ms) sugere O(N2) de volta"


def test_loop_1000_under_generous_budget():
    hm = LiquidityHeatmap(window_size=2000, cluster_threshold_pct=0.003,
                          min_trades_per_cluster=5, update_interval_ms=100)
    t0 = 1_700_000_000_000
    t = time.perf_counter()
    for i in range(1000):
        hm.add_trade(float(_price(i)), 0.1,
                     "buy" if i % 2 == 0 else "sell", t0 + i * 100)
    dt = time.perf_counter() - t
    assert dt < 8.0, f"loop N=1000 levou {dt:.1f}s (pre-fix ~15s)"
    assert len(hm.clusters) >= 1
