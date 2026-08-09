# -*- coding: utf-8 -*-
"""
Auditoria: duplo disparo de absorcao no pipeline real

Reproduz o pipeline do window_processor para o MESMO conjunto de trades:
  Caminho A: create_absorption_event (data_handler) -> resultado_da_batalha
  Caminho B: FlowAnalyzer (flow_analyzer/core.py) -> absorption_analysis
             (o evento real anexa fluxo_continuo do analyzer: data_handler.py:1328-1329)

E verifica qual rotulo o ai_payload_optimizer envia a IA (common/ai_payload_optimizer.py:709-715).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flow_analyzer.core import FlowAnalyzer
from data_processing.data_handler import create_absorption_event

def trades_venda_absorvida():
    # 10 vendas agressivas (m=True), preco estavel: open=50000, close=50009 (sobe +0.018%)
    _t0 = int(time.time() * 1000) - 5_000
    return [{"p": 50000.0 + i, "q": 1.0, "T": _t0 + i * 100, "m": True}
            for i in range(10)]

trades = trades_venda_absorvida()
base_ms = trades[0]["T"]

# ============================================================
# CAMINHO A: evento via data_handler (como o window_processor faz)
# ============================================================
ev_a = create_absorption_event(trades, "BTCUSDT", delta_threshold=0.5)
label_a = ev_a.get("resultado_da_batalha")
print(f"[CAMINHO A] evento 'Absorcao': resultado_da_batalha = '{label_a}' | is_signal={ev_a.get('is_signal')}")

# ============================================================
# CAMINHO B: FlowAnalyzer (fluxo_continuo que o evento real anexa)
# ============================================================
analyzer = FlowAnalyzer()
for t in trades:
    analyzer.process_trade({
        "p": str(t["p"]), "q": str(t["q"]), "T": t["T"], "m": t["m"],
    })

now_ms = base_ms + len(trades) * 100
snap = analyzer._create_snapshot(now_ms)
time_index = {"timestamp_utc": str(now_ms), "epoch_ms": now_ms}
metrics = analyzer._compute_accumulated_metrics(snap, time_index)
of = analyzer._compute_order_flow(snap, now_ms, time.perf_counter())
if of:
    metrics.update(of)
absorb = analyzer._compute_absorption_analysis(metrics) or {}
label_b = (absorb.get("current_absorption") or {}).get("label", "(nenhum)")
print(f"[CAMINHO B] fluxo_continuo.absorption_analysis.current_absorption.label = '{label_b}'")

# ============================================================
# EVENTO FINAL REAL: o window_processor anexa o fluxo_continuo ao evento
# (data_handler.py:1328-1329) -> o MESMO evento carrega os 2 rotulos
# ============================================================
ev_final = dict(ev_a)
ev_final["fluxo_continuo"] = metrics  # como create_absorption_event faz
print()
print("EVENTO FINAL ENVIADO AO PIPELINE (mesma janela, mesmo trade set):")
print(f"  resultado_da_batalha (caminho A): '{ev_final['resultado_da_batalha']}'")
print(f"  fluxo_continuo.absorption_analysis.label (caminho B): '{label_b}'")
rotulos_iguais = label_a == label_b
print(f"  ROTULOS CONSISTENTES? {'SIM' if rotulos_iguais else 'NAO - CONTRADICAO NO MESMO EVENTO'}")

# ============================================================
# O que a IA recebe: ai_payload_optimizer.py:709-715
# prioridade: tipo_absorcao -> absorb.label (caminho B) -> resultado_da_batalha (A)
# ============================================================
try:
    from common.ai_payload_optimizer import optimize_for_ai
    optimized = optimize_for_ai(dict(ev_final))
    payload_ai = optimized.get("payload", optimized) if isinstance(optimized, dict) else {}
    def _find(d, key, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == key:
                    return v
                r = _find(v, key, f"{path}.{k}")
                if r is not None:
                    return r
        return None
    abs_lbl = _find(payload_ai, "abs_lbl") or _find(payload_ai, "absorb")
    print(f"  Payload otimizado da IA -> abs_lbl: '{abs_lbl}'")
except Exception as e:
    print(f"  (optimize_for_ai nao executado: {e})")

print("=" * 62)
if not rotulos_iguais:
    print("CONFIRMADO: duplo rotulo CONTRADITORIO no mesmo evento")
else:
    print("Rotulos coincidem neste cenario")
print("=" * 62)
