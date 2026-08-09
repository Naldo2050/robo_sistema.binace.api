# -*- coding: utf-8 -*-
"""
Auditoria CVD - teste numerico (sem corrigir nada).
Sequencia conhecida, resultado esperado +2.0 BTC.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from monitoring.time_manager import TimeManager
from flow_analyzer.core import FlowAnalyzer

trades = [
    {"p": "50000", "q": "1.0", "T": 1000, "m": False},  # compra agressiva
    {"p": "50000", "q": "1.0", "T": 1001, "m": False},  # compra agressiva
    {"p": "50000", "q": "1.0", "T": 1002, "m": False},  # compra agressiva
    {"p": "50000", "q": "0.5", "T": 1003, "m": True},   # venda agressiva
    {"p": "50000", "q": "0.5", "T": 1004, "m": True},   # venda agressiva
]

# TimeManager real (mesmo do docstring do FlowAnalyzer)
tm = TimeManager()
analyzer = FlowAnalyzer(tm)

invalid = []
for t in trades:
    analyzer.process_trade(t)

cvd_obtido = float(analyzer.cvd)
cvd_esperado = 2.0

print("=" * 60)
print("TESTE NUMERICO CVD - flow_analyzer/core.py (caminho LIVE)")
print("=" * 60)
print(f"CVD obtido  : {cvd_obtido}")
print(f"CVD esperado: +{cvd_esperado}")
status = "PASS" if abs(cvd_obtido - cvd_esperado) < 1e-9 else "FAIL"
print(f"Resultado   : {status}")
if status == "FAIL":
    if abs(cvd_obtido - (-2.0)) < 1e-9:
        print("> Valor foi -2.0 -> INVERSAO DE SINAL (bug critico)")
    else:
        print(f"> Outro valor ({cvd_obtido}) -> erro de calculo")
print("=" * 60)

stats = analyzer.get_stats()
print(f"get_stats()['cvd'] = {stats['cvd']}")
print(f"invalid_trades     = {stats['invalid_trades']}")
print(f"total_processed    = {stats['total_trades_processed']}")
print(f"out_of_order_count = {stats['out_of_order_count']}")
