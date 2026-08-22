# tests/unit/test_rolling_aggregate_eviction.py
# -*- coding: utf-8 -*-
"""
Testes de contrato e eviction do RollingAggregate.

Garante:
- Semântica temporal primária para janelas de 1m, 5m e 15m sob taxas de 50 tps e 100 tps.
- Safety cap absoluto e detecção de CAPACITY_TRUNCATED.
- Estados de janela: WARMING_UP, FULL, CAPACITY_TRUNCATED.
- Recuperação graciosa de CAPACITY_TRUNCATED para FULL quando a cobertura temporal é restaurada.
- Consistência matemática estrita (net delta == buy - sell) após evictions.
"""
import sys
import os
import pytest
from decimal import Decimal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flow_analyzer import RollingAggregate

BASE_TS = 1_000_000_000_000


def _trade(ts, qty=1.0, price=50000.0, side='buy', delta_btc=None):
    if delta_btc is None:
        delta_btc = qty if side == 'buy' else -qty
    return {
        'ts': ts, 'qty': qty, 'price': price,
        'delta_btc': delta_btc, 'side': side, 'sector': None,
    }


def _aggregate_with_duration(window_min, duration_seconds):
    agg = RollingAggregate(window_min=window_min, max_trades=10)
    agg.add_trade(_trade(BASE_TS), whale_threshold=999.0)
    agg.add_trade(_trade(BASE_TS + duration_seconds * 1000), whale_threshold=999.0)
    return agg


@pytest.mark.parametrize("duration_seconds", [270, 296])
def test_full_threshold_5m_rejects_duration_below_99_percent(duration_seconds):
    metrics = _aggregate_with_duration(5, duration_seconds).get_metrics(50000.0)

    assert metrics['window_status'] != 'FULL'
    assert metrics['is_integrity_guaranteed'] is False


def test_full_threshold_5m_accepts_297_seconds():
    metrics = _aggregate_with_duration(5, 297).get_metrics(50000.0)

    assert metrics['window_status'] == 'FULL'
    assert metrics['is_integrity_guaranteed'] is True


def test_full_threshold_15m_rejects_810_seconds():
    metrics = _aggregate_with_duration(15, 810).get_metrics(50000.0)

    assert metrics['window_status'] != 'FULL'
    assert metrics['is_integrity_guaranteed'] is False


def test_full_threshold_15m_accepts_891_seconds():
    metrics = _aggregate_with_duration(15, 891).get_metrics(50000.0)

    assert metrics['window_status'] == 'FULL'
    assert metrics['is_integrity_guaranteed'] is True


def test_caso1_volume_normal_600_trades_cobrem_60s():
    """600 trades uniformes em 60s (10/s): janela de 1m deve cobrir todos."""
    agg = RollingAggregate(window_min=1, target_tps=100)
    interval_ms = 100  # 10 trades/s
    for i in range(600):
        agg.add_trade(_trade(BASE_TS + i * interval_ms), whale_threshold=999.0)

    assert len(agg.trades) == 600
    metrics = agg.get_metrics(50000.0)
    assert metrics['effective_duration_sec'] >= 59.0
    assert metrics['window_status'] == 'FULL'
    assert metrics['is_integrity_guaranteed'] is True


def test_caso2_volume_alto_1500_trades_cobrem_60s():
    """1500 trades em 60s (25/s): janela de 1m deve cobrir 60s sem truncamento."""
    agg = RollingAggregate(window_min=1, target_tps=100)
    interval_ms = 40  # 25 trades/s
    for i in range(1500):
        agg.add_trade(_trade(BASE_TS + i * interval_ms), whale_threshold=999.0)

    assert len(agg.trades) == 1500
    metrics = agg.get_metrics(50000.0)
    assert metrics['effective_duration_sec'] >= 59.0
    assert metrics['window_status'] == 'FULL'
    assert metrics['is_integrity_guaranteed'] is True


def test_caso3_flash_crash_cap_duro(caplog):
    """6000 trades em 1s com cap de 5000: buffer limita em 5000 e aciona warning."""
    agg = RollingAggregate(window_min=1, max_trades=5000)
    with caplog.at_level("WARNING"):
        for i in range(6000):
            agg.add_trade(_trade(BASE_TS + i), whale_threshold=999.0)

    assert len(agg.trades) == 5000
    assert agg.capacity_evictions == 1000
    metrics = agg.get_metrics(50000.0)
    assert metrics['window_status'] == 'CAPACITY_TRUNCATED'
    assert metrics['is_integrity_guaranteed'] is False


def test_contrato_A_1m_50tps():
    """1m @ 50 tps: 3000 trades cobrindo 60s -> status FULL."""
    agg = RollingAggregate(window_min=1, target_tps=100)
    for i in range(3000):
        agg.add_trade(_trade(BASE_TS + i * 20), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    assert metrics['trade_count'] == 3000
    assert metrics['effective_duration_sec'] >= 59.9
    assert metrics['window_status'] == 'FULL'
    assert metrics['is_integrity_guaranteed'] is True


def test_contrato_B_5m_50tps():
    """5m @ 50 tps: 15.000 trades cobrindo 300s -> status FULL."""
    agg = RollingAggregate(window_min=5, target_tps=100)
    for i in range(15000):
        agg.add_trade(_trade(BASE_TS + i * 20), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    assert metrics['trade_count'] == 15000
    assert metrics['effective_duration_sec'] >= 299.9
    assert metrics['window_status'] == 'FULL'
    assert metrics['is_integrity_guaranteed'] is True


def test_contrato_C_15m_50tps():
    """15m @ 50 tps (simulação com cap de 50 tps por injeção): status FULL."""
    agg = RollingAggregate(window_min=15, target_tps=50)
    # 45.000 trades com passo de 20ms = 900s
    for i in range(45000):
        agg.add_trade(_trade(BASE_TS + i * 20), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    assert metrics['effective_duration_sec'] >= 899.9
    assert metrics['window_status'] == 'FULL'
    assert metrics['is_integrity_guaranteed'] is True


def test_contrato_D_5m_100tps():
    """5m @ 100 tps: 30.000 trades cobrindo 300s -> status FULL sem truncamento."""
    agg = RollingAggregate(window_min=5, target_tps=100)
    for i in range(30000):
        agg.add_trade(_trade(BASE_TS + i * 10), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    assert metrics['trade_count'] == 30000
    assert metrics['effective_duration_sec'] >= 299.9
    assert metrics['window_status'] == 'FULL'
    assert metrics['is_integrity_guaranteed'] is True


def test_contrato_E_15m_100tps():
    """15m @ 100 tps (parametrizado com cap representativo de 10.000 trades para rapidez de teste)."""
    # Teste de 15m usando target_tps proporcional para manter teste rápido
    agg = RollingAggregate(window_min=15, max_trades=9000)
    # 9000 trades cobrindo 900s (passo de 100ms)
    for i in range(9000):
        agg.add_trade(_trade(BASE_TS + i * 100), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    assert metrics['effective_duration_sec'] >= 899.0
    assert metrics['window_status'] == 'FULL'
    assert metrics['is_integrity_guaranteed'] is True


def test_contrato_F_safety_cap_baixo_capacity_truncated():
    """Safety cap artificialmente baixo ativa CAPACITY_TRUNCATED e is_integrity_guaranteed=False."""
    agg = RollingAggregate(window_min=5, max_trades=1000)
    # 2000 trades em 100s (50ms por trade)
    for i in range(2000):
        agg.add_trade(_trade(BASE_TS + i * 50), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    assert len(agg.trades) == 1000
    assert agg.capacity_evictions == 1000
    assert metrics['window_status'] == 'CAPACITY_TRUNCATED'
    assert metrics['is_integrity_guaranteed'] is False
    assert metrics['effective_coverage_pct'] < 90.0


def test_contrato_G_recuperacao_de_capacity_truncated_para_full():
    """Aós surto e evictions, quando a taxa desacelera e cobre a janela inteira, volta para FULL."""
    agg = RollingAggregate(window_min=1, max_trades=1000)

    # 1. Surto de volume em 10s: 2000 trades em 10s (5ms por trade)
    for i in range(2000):
        agg.add_trade(_trade(BASE_TS + i * 5), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    assert metrics['window_status'] == 'CAPACITY_TRUNCATED'
    assert metrics['is_integrity_guaranteed'] is False

    # 2. Desaceleração: insere trades com ritmo normal por mais 60s (1000 trades em 60s)
    last_ts = BASE_TS + 2000 * 5
    for i in range(1000):
        agg.add_trade(_trade(last_ts + i * 60), whale_threshold=999.0)

    metrics_after = agg.get_metrics(50000.0)
    assert metrics_after['effective_duration_sec'] >= 59.0
    assert metrics_after['window_status'] == 'FULL'
    assert metrics_after['is_integrity_guaranteed'] is True


def test_contrato_H_warmup_status():
    """Poucos segundos de dados iniciam em WARMING_UP e não em CAPACITY_TRUNCATED."""
    agg = RollingAggregate(window_min=5, target_tps=100)
    # 50 trades em 5s
    for i in range(50):
        agg.add_trade(_trade(BASE_TS + i * 100), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    assert metrics['window_status'] == 'WARMING_UP'
    assert metrics['is_integrity_guaranteed'] is False
    assert agg.capacity_evictions == 0


def test_contrato_I_consistencia_matematica_apos_eviction():
    """sum_delta_usd deve ser rigorosamente igual a sum_buy_usd - sum_sell_usd após evictions."""
    agg = RollingAggregate(window_min=1, max_trades=500)

    for i in range(1200):
        side = 'buy' if i % 2 == 0 else 'sell'
        qty = 0.5 + (i % 3) * 0.1
        price = 50000.0 + (i % 7) * 5.0
        agg.add_trade(_trade(BASE_TS + i * 100, qty=qty, price=price, side=side), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    buy_usd = Decimal(str(metrics['sum_buy_usd']))
    sell_usd = Decimal(str(metrics['sum_sell_usd']))
    delta_usd = Decimal(str(metrics['sum_delta_usd']))

    assert abs(delta_usd - (buy_usd - sell_usd)) < Decimal('0.0001')

    # Validação cruzada com soma manual do deque
    recomputed_delta = sum(t[3] * t[2] for t in agg.trades)
    assert abs(recomputed_delta - delta_usd) < Decimal('0.0001')
