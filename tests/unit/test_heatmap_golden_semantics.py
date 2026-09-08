# tests/unit/test_heatmap_golden_semantics.py
"""
H1 — GOLDEN SEMANTICS ANTES DO FIX (Liquidity Heatmap).

Congela comportamento atual de _update_clusters em datasets pequenos/médios.
Captura SEMÂNTICA (não método interno): center/low/high/volumes/imbalance/
trades_count/ordenação + first/last/age determinísticos via _now_ms mockado.

Critério H3: após trocar np.mean incremental por sum/count, todos estes
testes devem continuar verdes (tolerância float pequena, sem exigir
igualdade byte-a-byte).
"""

import pytest

import market_analysis.liquidity_heatmap as hm_mod
from market_analysis.liquidity_heatmap import LiquidityHeatmap

NOW_MS = 1_700_001_000_000
T0 = 1_700_000_000_000


@pytest.fixture(autouse=True)
def _fixed_now(monkeypatch):
    monkeypatch.setattr(hm_mod, "_now_ms", lambda: NOW_MS)


def _mk(window_size=50, pct=0.01, min_trades=2, interval=10 ** 12):
    return LiquidityHeatmap(window_size=window_size,
                            cluster_threshold_pct=pct,
                            min_trades_per_cluster=min_trades,
                            update_interval_ms=interval)


def _feed_direct(hm, prices, volumes=None, sides=None, t0=T0, step=1000):
    n = len(prices)
    volumes = volumes if volumes is not None else [0.1] * n
    sides = sides if sides is not None else ["buy"] * n
    for i, (p, v, s) in enumerate(zip(prices, volumes, sides)):
        hm.price_levels.append(float(p))
        hm.volume_levels.append(float(v))
        hm.side_levels.append(s)
        hm.timestamp_levels.append(int(t0 + i * step))
    hm._update_clusters()
    return hm


def _by_center(clusters):
    return sorted(clusters, key=lambda c: c["center"])


def test_01_all_equal_prices():
    hm = _mk()
    _feed_direct(hm, [65000.0] * 5, volumes=[0.1] * 5,
                 sides=["buy"] * 5)
    assert len(hm.clusters) == 1
    c = hm.clusters[0]
    assert c["center"] == pytest.approx(65000.0)
    assert c["low"] == pytest.approx(65000.0)
    assert c["high"] == pytest.approx(65000.0)
    assert c["width"] == pytest.approx(0.0)
    assert c["total_volume"] == pytest.approx(0.5)
    assert c["buy_volume"] == pytest.approx(0.5)
    assert c["sell_volume"] == pytest.approx(0.0)
    assert c["imbalance"] == pytest.approx(0.5)
    assert c["trades_count"] == 5
    assert c["first_seen_ms"] == T0
    assert c["last_seen_ms"] == T0 + 4 * 1000
    assert c["age_ms"] == NOW_MS - (T0 + 4 * 1000)


def test_02_single_cluster_within_threshold():
    prices = [100.0, 100.5, 101.0, 100.2, 100.8]
    hm = _mk(pct=0.05)
    _feed_direct(hm, prices)
    assert len(hm.clusters) == 1
    c = hm.clusters[0]
    assert c["center"] == pytest.approx(sum(prices) / len(prices))
    assert c["low"] == pytest.approx(100.0)
    assert c["high"] == pytest.approx(101.0)
    assert c["width"] == pytest.approx(1.0)
    assert c["total_volume"] == pytest.approx(0.5)
    assert c["trades_count"] == 5


def test_03_exactly_at_threshold_stays_together():
    # pct=0.01: threshold no centro 100.0 => 1.0; 101.0 dist 1.0 => mesmo cluster
    hm = _mk(pct=0.01, min_trades=2)
    _feed_direct(hm, [100.0, 100.0, 101.0])
    assert len(hm.clusters) == 1
    c = hm.clusters[0]
    assert c["center"] == pytest.approx((100.0 + 100.0 + 101.0) / 3)
    assert c["trades_count"] == 3
    assert c["low"] == pytest.approx(100.0)
    assert c["high"] == pytest.approx(101.0)


def test_04_just_outside_threshold_splits():
    # 101.02 dist 1.02 > 1.0 => novo cluster (2+2 trades p/ finalizar ambos)
    hm = _mk(pct=0.01, min_trades=2)
    _feed_direct(hm, [100.0, 100.0, 101.02, 101.02])
    assert len(hm.clusters) == 2
    ordered = _by_center(hm.clusters)
    assert ordered[0]["center"] == pytest.approx(100.0)
    assert ordered[1]["center"] == pytest.approx(101.02)
    assert ordered[0]["trades_count"] == 2
    assert ordered[1]["trades_count"] == 2


def test_05_multiple_clusters_and_ordering_by_volume():
    # 100-cluster total 0.3 vs 200-cluster total 0.6 => 200 primeiro (volume desc)
    hm = _mk(pct=0.01, min_trades=2)
    _feed_direct(hm, [100.0] * 3 + [200.0] * 3,
                 volumes=[0.1] * 3 + [0.2] * 3)
    assert len(hm.clusters) == 2
    # ordenação: (total_volume, last_seen) desc
    assert hm.clusters[0]["center"] == pytest.approx(200.0)
    assert hm.clusters[1]["center"] == pytest.approx(100.0)
    assert hm.clusters[0]["total_volume"] == pytest.approx(0.6)
    assert hm.clusters[1]["total_volume"] == pytest.approx(0.3)


def test_06_mixed_buy_sell():
    hm = _mk(pct=0.05, min_trades=2)
    _feed_direct(hm, [100.0] * 4, volumes=[0.5, 0.5, 0.3, 0.3],
                 sides=["buy", "buy", "sell", "sell"])
    assert len(hm.clusters) == 1
    c = hm.clusters[0]
    assert c["total_volume"] == pytest.approx(1.6)
    assert c["buy_volume"] == pytest.approx(1.0)
    assert c["sell_volume"] == pytest.approx(0.6)
    assert c["imbalance"] == pytest.approx(0.4)
    assert c["imbalance_ratio"] == pytest.approx(0.4 / 1.6)
    assert c["trades_count"] == 4


def test_07_zero_volume_ignored_by_add_trade():
    hm = _mk()
    hm.add_trade(100.0, 0.0, "buy", T0)  # rejeitado: v>0 exigido
    assert len(hm.price_levels) == 0
    hm.add_trade(100.0, -1.0, "buy", T0)
    assert len(hm.price_levels) == 0
    assert hm.clusters == []


def test_08_out_of_order_timestamps():
    hm = _mk(pct=0.05, min_trades=2)
    prices = [100.0, 101.0, 100.5]
    # timestamps fora de ordem; após sort por preço, first/last vêm de min/max
    hm.price_levels.extend([100.0, 101.0, 100.5])
    hm.volume_levels.extend([0.1, 0.1, 0.1])
    hm.side_levels.extend(["buy", "buy", "buy"])
    hm.timestamp_levels.extend([T0 + 3000, T0 + 1000, T0 + 2000])
    hm._update_clusters()
    assert len(hm.clusters) == 1
    c = hm.clusters[0]
    assert c["first_seen_ms"] == T0 + 1000
    assert c["last_seen_ms"] == T0 + 3000
    assert c["first_seen_ms"] <= c["last_seen_ms"]
    assert c["age_ms"] == NOW_MS - (T0 + 3000)


def test_09_repeated_prices_sorted_into_clusters():
    # entrada intercalada; sort por preço deve agrupar 50s e 100s
    hm = _mk(pct=0.01, min_trades=2)
    _feed_direct(hm, [100.0, 50.0, 100.0, 50.0, 100.0, 50.0])
    assert len(hm.clusters) == 2
    ordered = _by_center(hm.clusters)
    assert ordered[0]["center"] == pytest.approx(50.0)
    assert ordered[1]["center"] == pytest.approx(100.0)
    assert ordered[0]["trades_count"] == 3
    assert ordered[1]["trades_count"] == 3


def test_10_eviction_beyond_window_size():
    hm = _mk(window_size=5, pct=0.05, min_trades=2)
    # 7 trades preço 100 (6 primeiros) + 200 no fim; janela 5 => últimos 5
    prices = [100.0] * 6 + [200.0]
    # pct 0.05: threshold em 100 => 5.0; 200 dist 100 => split, mas 200 sozinho
    # (1 trade < min 2) é descartado => só cluster 100 com últimos 4x100
    _feed_direct(hm, prices)
    assert len(hm.price_levels) == 5  # deque maxlen evictou os 2 mais antigos
    assert list(hm.price_levels) == pytest.approx([100.0] * 4 + [200.0])
    # 200 isolado não finaliza (1 < 2); sobra 1 cluster de 4x100
    assert len(hm.clusters) == 1
    assert hm.clusters[0]["center"] == pytest.approx(100.0)
    assert hm.clusters[0]["trades_count"] == 4
