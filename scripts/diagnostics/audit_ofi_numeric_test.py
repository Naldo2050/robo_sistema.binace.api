# -*- coding: utf-8 -*-
"""
Auditoria OFI - teste numerico (sem corrigir nada).
Sequencia conhecida de trades (o modulo institutional trabalha com trades
por agressao, NAO book updates - verificado: process_trade recebe Trade).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from institutional.order_flow_imbalance import OrderFlowImbalanceAnalyzer
from institutional.base import Trade, Side

# Sequencia: 3 compras agressivas + 2 vendas agressivas, janela de 1s
# t=0.0 buy 1.0 | t=0.2 buy 1.5 | t=0.4 buy 0.5 | t=0.6 sell 0.5 | t=1.1 sell 0.5 (fecha janela)
trades = [
    Trade(timestamp=0.0,  price=50000.0, quantity=1.0, side=Side.BUY),
    Trade(timestamp=0.2,  price=50001.0, quantity=1.5, side=Side.BUY),
    Trade(timestamp=0.4,  price=50002.0, quantity=0.5, side=Side.BUY),
    Trade(timestamp=0.6,  price=50003.0, quantity=0.5, side=Side.SELL),
    Trade(timestamp=1.1,  price=50004.0, quantity=0.5, side=Side.SELL),
]

analyzer = OrderFlowImbalanceAnalyzer(window_seconds=1.0, alert_cooldown_seconds=0)

print("=" * 64)
print("TESTE NUMERICO OFI - institutional/order_flow_imbalance.py")
print("=" * 64)
completed = []
for t in trades:
    bar = analyzer.process_trade(t)
    if bar is not None:
        completed.append(bar)

print(f"Barras fechadas: {len(completed)}")
if completed:
    b = completed[0]
    # Janela 1: buy=3.0, sell=0.5 -> ratio esperado 3.0/0.5 = 6.0
    print(f"  buy_vol={b.buy_volume}  sell_vol={b.sell_volume}")
    print(f"  imbalance_ratio={b.imbalance_ratio}  (esperado: 6.0)")
    print(f"  dominant_side={b.dominant_side}      (esperado: Side.BUY)")
    print(f"  price_change_pct={b.price_change_pct:.2f}%")

ratio_ok = completed and abs(completed[0].imbalance_ratio - 6.0) < 1e-9
side_ok = completed and completed[0].dominant_side == Side.BUY

# get_current_imbalance no estado atual (apos a barra, janela nova com 0.5 sell)
cur = analyzer.get_current_imbalance()
print(f"\nEstado atual (janela nova): ratio={cur['ratio']:.3f} dominant={cur['dominant_side']}")

# Alerta esperado: ratio 6.0 >= EXTREME (5.0)
alerts = analyzer.alerts
print(f"\nAlertas gerados: {len(alerts)}")
if alerts:
    a = alerts[0]
    print(f"  severity={a.severity} ratio={a.ratio} side={a.side}  (esperado: extreme, 6.0, BUY)")

print("=" * 64)
if ratio_ok and side_ok:
    print("RESULTADO: PASS - formula e sinal corretos (buy/sell ratio, buy dominante)")
else:
    print(f"RESULTADO: FAIL - ratio_ok={ratio_ok} side_ok={side_ok}")
print("=" * 64)

# Caso extremo: venda pura (sell_vol > 0, buy_vol = 0) -> ratio deve ser 0
a2 = OrderFlowImbalanceAnalyzer(window_seconds=1.0, alert_cooldown_seconds=0)
b2 = a2.process_trade(Trade(timestamp=0.0, price=50000.0, quantity=2.0, side=Side.SELL))
b2 = a2.process_trade(Trade(timestamp=1.1, price=50001.0, quantity=1.0, side=Side.SELL)) or b2
print(f"\n[edge] Venda pura: ratio={b2.imbalance_ratio} (esperado 0.0) dominant={b2.dominant_side} (esperado SELL)")

# Caso extremo: compra pura -> ratio = inf (tratado)
a3 = OrderFlowImbalanceAnalyzer(window_seconds=1.0, alert_cooldown_seconds=0)
b3 = a3.process_trade(Trade(timestamp=0.0, price=50000.0, quantity=1.0, side=Side.BUY))
b3 = a3.process_trade(Trade(timestamp=1.1, price=50001.0, quantity=1.0, side=Side.BUY)) or b3
print(f"[edge] Compra pura: ratio={b3.imbalance_ratio} (esperado inf) dominant={b3.dominant_side} (esperado BUY)")

# BUG: contaminação de janela — trade que FECHA a janela é acumulado na
# janela seguinte ANTES dela ser inicializada; quando o próximo trade chega,
# a janela re-inicializa com o timestamp dele, mas o volume do trade que
# fechou a janela anterior fica dentro dela (volume de fora do período).
a5 = OrderFlowImbalanceAnalyzer(window_seconds=1.0, alert_cooldown_seconds=0)
a5.process_trade(Trade(timestamp=0.0, price=50000.0, quantity=1.0, side=Side.BUY))
bar1 = a5.process_trade(Trade(timestamp=1.1, price=50001.0, quantity=0.5, side=Side.SELL))  # fecha janela1
# t=2.0: re-inicializa janela (start=2.0); o sell 0.5 de t=1.1 está preso nela
bar2 = a5.process_trade(Trade(timestamp=2.0, price=50002.0, quantity=0.7, side=Side.BUY))  # fecha janela2
cur2 = a5.get_current_imbalance()
print(f"\n[BUG] Janela2 (start t=2.0) deveria ter SO o buy 0.7 de t=2.0,")
print(f"      mas contem sell 0.5 de t=1.1 (fora do periodo):")
print(f"      sell_vol={cur2['sell_volume']} (esperado 0.0) | buy_vol={cur2['buy_volume']} (esperado 0.7)")
if bar2:
    print(f"      bar2 selado: buy={bar2.buy_volume} sell={bar2.sell_volume} (esperado buy=0.7 sell=0.0)")

# Janela vazia (nenhum trade): get_current_imbalance sem dados
a4 = OrderFlowImbalanceAnalyzer(window_seconds=1.0)
print(f"[edge] Janela vazia: ratio={a4.get_current_imbalance()['ratio']} (esperado 0.0, sem crash)")
