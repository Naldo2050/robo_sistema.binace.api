# tests/unit/test_rolling_aggregate_eviction.py
# -*- coding: utf-8 -*-
"""
Fix 4 - Eviction hibrida do RollingAggregate.

Antes: _evict_if_needed removia por contagem (max_trades=600 para 1m),
segurando ~15s de dados em mercado >10 trades/s e reportando net_flow_1m
como se fosse 60s (~4x menor que o real).

Depois: criterio primario TEMPORAL (janela real 60s) + cap duro de
contagem (5000) so como seguranca contra flash crash.

Casos:
  1. Volume normal (10 trades/s por 60s)  -> janela cobre ~60s reais
  2. Volume alto (25 trades/s por 60s)    -> janela cobre ~60s reais
  3. Flash crash (6000 trades em 1s)      -> buffer capado em 5000 + warning
  4. Consistencia: net_flow == buy - sell (tolerancia 0.01%)
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


def test_caso1_volume_normal_600_trades_cobrem_60s():
    """600 trades uniformes em 60s (10/s): janela de 1m deve cobrir todos."""
    agg = RollingAggregate(window_min=1, max_trades=600)
    interval_ms = 100  # 10 trades/s
    for i in range(600):
        agg.add_trade(_trade(BASE_TS + i * interval_ms), whale_threshold=999.0)

    assert len(agg.trades) == 600, (
        "Eviction prematura: 600 trades em 60s deveriam permanecer inteiros"
    )

    metrics = agg.get_metrics(50000.0)
    span_s = (metrics['last_update'] - agg.trades[0][0]) / 1000.0
    assert span_s >= 59.0, f"Janela cobriu apenas {span_s:.1f}s reais (esperado ~60s)"
    assert metrics['sum_buy_btc'] == 600.0


def test_caso2_volume_alto_1500_trades_cobrem_60s():
    """1500 trades em 60s (25/s): NÃO pode segurar só ~15s (~375 trades)."""
    agg = RollingAggregate(window_min=1, max_trades=600)
    interval_ms = 40  # 25 trades/s
    for i in range(1500):
        agg.add_trade(_trade(BASE_TS + i * interval_ms), whale_threshold=999.0)

    assert len(agg.trades) == 1500, (
        f"Janela segurou {len(agg.trades)}/1500 trades - eviction por contagem "
        "ainda truncando dados (antes do fix: ~375)"
    )

    metrics = agg.get_metrics(50000.0)
    span_s = (metrics['last_update'] - agg.trades[0][0]) / 1000.0
    assert span_s >= 59.0, f"Janela cobriu apenas {span_s:.1f}s reais (esperado ~60s)"
    assert metrics['sum_buy_btc'] == 1500.0


def test_caso3_flash_crash_cap_duro_5000(caplog):
    """6000 trades em 1s: buffer não passa de 5000 e warning é logado."""
    agg = RollingAggregate(window_min=1, max_trades=600)
    with caplog.at_level("WARNING"):
        for i in range(6000):
            agg.add_trade(_trade(BASE_TS + i), whale_threshold=999.0)

    assert len(agg.trades) == 5000, f"Cap duro não respeitado: {len(agg.trades)}"
    assert agg.capacity_evictions == 1000
    assert any("hard cap" in r.message for r in caplog.records), (
        "Warning de hard cap não foi logado"
    )


def test_caso4_consistencia_net_flow_igual_buy_menos_sell():
    """net_flow (sum_delta_usd) deve ser igual a buy_usd - sell_usd."""
    agg = RollingAggregate(window_min=1, max_trades=600)

    # Mix de buys/sells com preços variados em 90s (alguns saem da janela)
    for i in range(900):
        side = 'buy' if i % 3 == 0 else 'sell'
        qty = 1.0 + (i % 5) * 0.25
        price = 50000.0 + (i % 10) * 10.0
        delta = qty if side == 'buy' else -qty
        agg.add_trade(_trade(BASE_TS + i * 100, qty=qty, price=price,
                             side=side, delta_btc=delta), whale_threshold=999.0)

    metrics = agg.get_metrics(50000.0)
    net_flow = metrics['sum_delta_usd']
    buy_minus_sell = metrics['sum_buy_usd'] - metrics['sum_sell_usd']

    total_volume = metrics['sum_buy_usd'] + metrics['sum_sell_usd']
    tolerance = total_volume * 0.0001  # 0.01%
    assert abs(net_flow - buy_minus_sell) <= max(tolerance, 1e-9), (
        f"Consistência quebrada: net_flow={net_flow} vs buy-sell={buy_minus_sell}"
    )

    # Verificação independente: recomputar somas direto do deque
    recomputed_delta = sum(t[3] * t[2] for t in agg.trades)
    assert abs(float(recomputed_delta) - net_flow) <= max(tolerance, 1e-9)
