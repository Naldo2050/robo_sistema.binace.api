# -*- coding: utf-8 -*-
"""
Auditoria: deteccao de absorcao - caminhos ativos em producao

Caminho A (evento): window_processor -> pipeline.detect_signals ->
  create_absorption_event (data_handler.py:1012) -> _compute_absorption_scalar
Caminho B (metrica): flow_analyzer/core.py -> AbsorptionClassifier.classify
  (core.py:1024) -> tipo_absorcao -> AbsorptionAnalyzer.analyze ->
  absorption_analysis.current_absorption.index (consumido em
  market_orchestrator.py:940-950 como trigger da IA, threshold 0.6)

Cenario positivo (absorcao): grande volume de VENDA agressiva (m=True)
  sem queda de preco proporcional -> compra passiva absorvendo.
Cenario negativo: mesmo volume, mas preco caiu proporcionalmente.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flow_analyzer.absorption import AbsorptionClassifier, AbsorptionAnalyzer
from flow_analyzer.core import FlowAnalyzer
from data_processing.data_handler import create_absorption_event, _compute_absorption_scalar

# ============================================================
# CAMINHO A: _compute_absorption_scalar (nucleo do evento)
# ============================================================
print("--- CAMINHO A: _compute_absorption_scalar ---")

# A1 POSITIVO: 10 vendas agressivas, preco praticamente parado
# open=50000, high=50005, low=49998, close=50003 | buy=0, sell=10.0 BTC
d, abs_c, abs_v, idx = _compute_absorption_scalar(50000, 50005, 49998, 50003, 0.5, 0.0, 10.0)
print(f"[A1] vendas 10BTC, preco estavel -> delta={d:.2f} abs_compra={abs_c} abs_venda={abs_v} idx={idx:.3f}")
assert d < -0.5 and abs_c and not abs_v, "A1: deveria detectar agressao vendedora absorvida"

# A2 NEGATIVO: 10 vendas agressivas, preco CAIU (close abaixo do -0.2%)
d2, abs_c2, abs_v2, idx2 = _compute_absorption_scalar(50000, 50000, 49880, 49880, 0.5, 0.0, 10.0)
print(f"[A2] vendas 10BTC, preco caiu     -> delta={d2:.2f} abs_compra={abs_c2} abs_venda={abs_v2} idx={idx2:.3f}")
assert not abs_c2 and not abs_v2, "A2: movimento proporcional nao deveria ser absorcao"

# A3 POSITIVO (lado contrario): 10 compras agressivas, preco estavel
# open=50000, high=50005, low=49997, close=49999 | buy=10.0, sell=0
d3, abs_c3, abs_v3, idx3 = _compute_absorption_scalar(50000, 50005, 49997, 49999, 0.5, 10.0, 0.0)
print(f"[A3] compras 10BTC, preco estavel -> delta={d3:+.2f} abs_compra={abs_c3} abs_venda={abs_v3} idx={idx3:.3f}")
assert d3 > 0.5 and abs_v3 and not abs_c3, "A3: deveria detectar agressao compradora absorvida"

# ============================================================
# CAMINHO A (integrado): create_absorption_event com trades reais
# ============================================================
print("--- CAMINHO A (evento): create_absorption_event ---")

def trades_venda_sem_movimento():
    # 10 vendas agressivas (m=True); preco sobe levemente (50000->50009, +0.018%),
    # fechando na metade superior -> compra passiva absorvendo a venda
    return [{"p": 50000.0 + i, "q": 1.0, "T": 1_700_000_000_000 + i * 100, "m": True}
            for i in range(10)]

ev_pos = create_absorption_event(trades_venda_sem_movimento(), "BTCUSDT", delta_threshold=0.5)
print(f"[A4] evento absorcao: is_signal={ev_pos.get('is_signal')} resultado='{ev_pos.get('resultado_da_batalha')}' delta={ev_pos.get('delta')}")
assert ev_pos.get("is_signal"), "A4: venda forte sem movimento de preco deveria ser sinal"
assert ev_pos.get("resultado_da_batalha") == "Absorção de Venda", f"A4 rotulo inesperado: {ev_pos.get('resultado_da_batalha')}"
assert ev_pos.get("absorption_side") == "buy", f"A4 side inesperado: {ev_pos.get('absorption_side')}"

def trades_venda_com_movimento():
    # Mesmo volume de venda, mas preco despenca (close ~= low)
    base = 1_700_000_000_000
    return [{"p": 50000.0 - (i * 50), "q": 1.0, "T": base + i * 100, "m": True}
            for i in range(10)]

ev_neg = create_absorption_event(trades_venda_com_movimento(), "BTCUSDT", delta_threshold=0.5)
print(f"[A5] evento nao-absorcao: is_signal={ev_neg.get('is_signal')} resultado='{ev_neg.get('resultado_da_batalha')}'")
assert not ev_neg.get("is_signal"), "A5: queda proporcional nao deveria ser sinal"

# A6: compras agressivas absorvidas (lado contrario de A4)
def trades_compra_sem_movimento():
    # 10 compras agressivas (m=False); preco estavel -> vendedores absorvendo
    return [{"p": 50000.0 - i, "q": 1.0, "T": 1_700_000_000_000 + i * 100, "m": False}
            for i in range(10)]

ev_compra = create_absorption_event(trades_compra_sem_movimento(), "BTCUSDT", delta_threshold=0.5)
print(f"[A6] evento compra absorvida: is_signal={ev_compra.get('is_signal')} "
      f"resultado='{ev_compra.get('resultado_da_batalha')}' side={ev_compra.get('absorption_side')} "
      f"delta={ev_compra.get('delta')}")
assert ev_compra.get("is_signal"), "A6: compra forte sem movimento deveria ser sinal"
assert ev_compra.get("resultado_da_batalha") == "Absorção de Compra", f"A6 rotulo inesperado: {ev_compra.get('resultado_da_batalha')}"
assert ev_compra.get("absorption_side") == "sell", f"A6 side inesperado: {ev_compra.get('absorption_side')}"
assert ev_compra.get("resultado_da_batalha") != ev_pos.get("resultado_da_batalha"), \
    "A4 e A6 devem retornar rotulos DIFERENTES (assimetria corrigida)"

# ============================================================
# CAMINHO B: AbsorptionClassifier (fluxo_continuo do core)
# ============================================================
print("--- CAMINHO B: AbsorptionClassifier.classify ---")
cls = AbsorptionClassifier()
B1 = cls.classify(delta_btc=-10.0, open_p=50000, high_p=50005, low_p=49998, close_p=50003)
B2 = cls.classify(delta_btc=-10.0, open_p=50000, high_p=50000, low_p=49880, close_p=49880)
B3 = cls.classify(delta_btc=+10.0, open_p=50000, high_p=50005, low_p=49997, close_p=49999)
print(f"[B1] vendas 10BTC, preco estavel -> '{B1}' (esperado: Absorcao de Venda)")
print(f"[B2] vendas 10BTC, preco caiu    -> '{B2}' (esperado: Neutra)")
print(f"[B3] compras 10BTC, preco estavel -> '{B3}' (esperado: Absorcao de Compra)")
assert B1 == "Absorção de Venda", f"B1 inesperado: {B1}"
assert B2 == "Neutra", f"B2 inesperado: {B2}"
assert B3 == "Absorção de Compra", f"B3 inesperado: {B3}"

# ============================================================
# CAMINHO B (metrica): AbsorptionAnalyzer.analyze -> trigger da IA
# ============================================================
an = AbsorptionAnalyzer()
r = an.analyze(
    delta_usd=-800_000, total_volume_usd=1_000_000, flow_imbalance=-0.8,
    buy_pct=10.0, sell_pct=90.0, absorption_label="Absorção de Venda",
    window_min=1,
)
print(f"[B4] analyze: index={r.index} classification={r.classification} label='{r.label}'")
assert r.index == 0.64 and r.classification == "MODERATE_ABSORPTION", f"B4 inesperado: {r}"
print("     -> index 0.64 > 0.6 = dispararia ANALYSIS_TRIGGER no orchestrator (market_orchestrator.py:950)")

# Rótulo final do evento A4: verificar consistencia com o sinal do delta
print("=" * 62)
print("CONSISTENCIA DE ROTULO entre os 2 caminhos ativos (mesmo cenario):")
print(f"  Caminho A (evento, A4): delta=-10, preco estavel -> '{ev_pos.get('resultado_da_batalha')}'")
print(f"  Caminho B (metrica, B1): delta=-10, preco estavel -> '{B1}'")
print("=" * 62)
assert ev_pos.get("resultado_da_batalha") == B1, "A4 e B1 devem ter o MESMO rotulo (convencao unificada)"
