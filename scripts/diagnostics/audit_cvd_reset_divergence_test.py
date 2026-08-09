# -*- coding: utf-8 -*-
"""
Auditoria: reset periodico do CVD (4h) causa sinal de divergencia falso?

Cadeia real: FlowAnalyzer.cvd (reseta) -> get_flow_metrics()['cvd']
  -> fluxo_continuo.cvd no evento -> _build_cvd_divergence fallback
  (payload_builder_compact.py).

Fix aplicado (2 camadas):
  Camada 1: supressao de cvd_div enquanto CVD ainda aquece pos-reset
            (CVD_DIV_WARMUP_SECONDS, default 300s).
  Camada 2: comparar CVD com a VARIACAO DE PRECO DO MESMO PERIODO
            (price_at_reset capturado no reset), nao com o trend_1h inteiro.
  Camada 2.3: periodo pos-reset < CVD_DIV_MIN_PERIOD_SECONDS (10s) tambem suprime.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flow_analyzer.core import FlowAnalyzer
from market_orchestrator.ai.payload_builder_compact import _build_cvd_divergence

analyzer = FlowAnalyzer()

# ---- Fase 1: 50 trades de compra agressiva, preco subindo (consistente) ----
_t0 = int(time.time() * 1000) - 5_000
for i in range(50):
    analyzer.process_trade({
        "p": str(50000 + i), "q": "1.0", "T": _t0 + i, "m": False,
    })

cvd_antes = float(analyzer.cvd)
print(f"[1] CVD acumulado pre-reset      : {cvd_antes:+.2f} BTC")
assert abs(cvd_antes - 50.0) < 1e-9, "pre-reset deveria ser +50"

# Simula o evento pre-reset (fluxo real: last_reset_ms = init, price_at_reset = None)
evento_antes = {
    "epoch_ms": int(time.time() * 1000),
    "fluxo_continuo": {
        "cvd": cvd_antes,
        "last_reset_ms": analyzer.last_reset_ms,
        "price_at_reset": analyzer._price_at_reset,
    },
    "multi_tf": {"1h": {"tendencia": "Alta"}},
}
div_antes = _build_cvd_divergence(evento_antes)
print(f"[2] cvd_div PRE-reset            : {div_antes or '(vazio = sem divergencia)'}")
assert div_antes == {}, "pre-reset nao deveria gerar divergencia (sem referencia de preco)"

# ---- Fase 2: reset forcado (simula passagem de 4h / CVD_RESET_INTERVAL_HOURS) ----
analyzer._reset_metrics()
cvd_pos = float(analyzer.cvd)
print(f"[3] CVD apos reset               : {cvd_pos:+.2f} BTC")
assert abs(cvd_pos) < 1e-9, "pos-reset deveria ser 0"
assert analyzer._price_at_reset == 50049.0, "price_at_reset deveria capturar o preco no reset"

# ---- Fase 3: trades pos-reset (mercado continua em alta consistente) ----
# 2 vendas agressivas = -1.0 BTC, acima do threshold abs(cvd)>0.5 do fallback
analyzer.process_trade({"p": "50050", "q": "1.0", "T": _t0 + 1001, "m": True})   # venda agressiva 1.0
analyzer.process_trade({"p": "50051", "q": "0.5", "T": _t0 + 1002, "m": True})   # venda agressiva 0.5
cvd_pos2 = float(analyzer.cvd)
print(f"[4] CVD pos-reset apos 2 trades  : {cvd_pos2:+.2f} BTC")

# Evento pos-reset REAL: CVD acabou de resetar (elapsed < warmup) -> Camada 1 suprime
evento_pos = {
    "epoch_ms": int(time.time() * 1000) + 5,
    "fluxo_continuo": {
        "cvd": cvd_pos2,
        "last_reset_ms": analyzer.last_reset_ms,
        "price_at_reset": analyzer._price_at_reset,
    },
    "multi_tf": {"1h": {"tendencia": "Alta"}},
    "preco_fechamento": 50051,
}
div_pos = _build_cvd_divergence(evento_pos)
print(f"[5] cvd_div POS-reset (warm-up)  : {div_pos or '(vazio = suprimido)'}")
assert div_pos == {}, "Camada 1 deveria suprimir divergencia logo apos o reset"

# ---- Fase 4: Camada 2 - comparacao no MESMO periodo (reset ha > warmup) ----
# Reset "antigo" (400s atras), preco subiu de 50050 para 50051 desde o reset,
# CVD caiu para -1.5 no mesmo periodo -> divergencia REAL (bearish_div legítima)
t_antigo = int(time.time() * 1000) - 400_000
fluxo_mesmo_periodo = {
    "cvd": -1.5,
    "last_reset_ms": t_antigo,
    "price_at_reset": 50050.0,
}
div_legitimo = _build_cvd_divergence({
    "epoch_ms": t_antigo + 400_000,
    "fluxo_continuo": fluxo_mesmo_periodo,
    "preco_fechamento": 50051,
})
print(f"[6] cvd_div mesmo-periodo (legit) : {div_legitimo or '(vazio)'}")
assert div_legitimo.get("type") == "bearish_div", "preco subiu + CVD caiu no mesmo periodo = divergencia real"

# ---- Fase 5: Camada 2.3 - periodo pos-reset < 10s tambem suprime ----
t_recente = int(time.time() * 1000) - 5_000
div_recente = _build_cvd_divergence({
    "epoch_ms": int(time.time() * 1000),
    "fluxo_continuo": {
        "cvd": -1.5,
        "last_reset_ms": t_recente,
        "price_at_reset": 50050.0,
    },
    "preco_fechamento": 50051,
})
print(f"[7] cvd_div pos-reset < 10s      : {div_recente or '(vazio = suprimido)'}")
assert div_recente == {}, "Camada 2.3 deveria suprimir com periodo pos-reset < 10s"

print("=" * 62)
print("RESULTADO: PASS - falso sinal apos reset eliminado (Camadas 1, 2 e 2.3)")
print("=" * 62)
