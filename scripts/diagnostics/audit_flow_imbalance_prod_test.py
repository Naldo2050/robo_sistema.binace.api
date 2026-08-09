# -*- coding: utf-8 -*-
"""
Auditoria: flow_imbalance REAL em producao (flow_analyzer/core.py)

Fonte: _compute_detailed_window -> order_flow["flow_imbalance"]
Consumido em payload_builder_compact.py:1099-1109 (Fonte 2 "order_flow").

Testes:
  A) Atribuicao de trades na transicao de janela (sem volume fantasma)
  B) Trade no limite exato da janela (ts == cutoff) nao conta 2x
  C) Comportamento pos-reset (_reset_metrics compartilhado com o CVD)
  D) Consumo no _build_ofi: existe comparacao com periodo longo (bug cvd_div)?
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flow_analyzer.core import FlowAnalyzer
from market_orchestrator.ai.payload_builder_compact import _build_ofi

analyzer = FlowAnalyzer()

# Isolar a logica de atribuicao de janela do guard de amostra minima (testes A/B):
analyzer._flow_imbalance_min_trades = 1

# ============================================================
# A) Transicao de janela (menor janela: 1 min = 60_000 ms)
#    Trades: t=59000 buy 1.0 | t=60000 sell 0.5 | t=61000 buy 2.0
# ============================================================
def snapshot_com(now_ms, trades):
    return {"flow_trades": trades, "_last_price": 100.0}

trades = [
    {"ts": 59000, "price": 100.0, "qty": 1.0, "delta_btc": 1.0,  "delta_usd": 100.0, "side": "buy",  "sector": "large"},
    {"ts": 60000, "price": 100.0, "qty": 0.5, "delta_btc": -0.5, "delta_usd": -50.0, "side": "sell", "sector": "large"},
    {"ts": 61000, "price": 100.0, "qty": 2.0, "delta_btc": 2.0,  "delta_usd": 200.0, "side": "buy",  "sector": "large"},
]

# now=60_000: janela [0, 60_000] -> deve conter SO 59000 e 60000
of1 = {}
analyzer._compute_detailed_window(of1, snapshot_com(60_000, trades), 1, 0, 60_000, 0.5, time.time())
fi1 = of1.get("flow_imbalance")
exp1 = (100.0 - 50.0) / 150.0   # buy 100 USD vs sell 50 USD -> 0.3333
print(f"[A1] now=60s   fi={fi1} (esperado {round(exp1,4)})")
assert fi1 is not None and abs(fi1 - round(exp1, 4)) < 1e-6, f"janela [0,60s] errada: {fi1}"

# now=61_000: janela [1_000, 61_000] (60s) -> 61000 ENTROU; 59000 AINDA DENTRO
of2 = {}
analyzer._compute_detailed_window(of2, snapshot_com(61_000, trades), 1, 1_000, 61_000, 0.5, time.time())
fi2 = of2.get("flow_imbalance")
exp2 = (300.0 - 50.0) / 350.0   # buy 300 USD (59000+61000) vs sell 50 USD -> 0.7143
print(f"[A2] now=61s   fi={fi2} (esperado {round(exp2,4)})")
assert fi2 is not None and abs(fi2 - round(exp2, 4)) < 1e-6, f"janela [1s,61s] errada: {fi2}"

# ============================================================
# B) Trade no limite (ts == cutoff) nao conta 2x nem some
# ============================================================
# now=120_000: janela [60_000, 120_000] -> 59000 EXPIRADO; 60000 (limite) INCLUSO
of3 = {}
analyzer._compute_detailed_window(of3, snapshot_com(120_000, trades), 1, 60_000, 120_000, 0.5, time.time())
fi3 = of3.get("flow_imbalance")
exp3 = (200.0 - 50.0) / 250.0   # 61000 buy + 60000 sell (limite INCLUSO) -> 0.6
print(f"[B1] now=120s  fi={fi3} (esperado {round(exp3,4)})")
assert fi3 is not None and abs(fi3 - round(exp3, 4)) < 1e-6, f"limite de janela errado: {fi3}"
# 59000 nao pode ter "vazado" para a janela [60s,120s]:
assert abs(fi3 - 0.6) < 1e-6, "59000 vazou para a janela seguinte (volume fantasma)"

# now=120_001: cutoff=60_001 -> 60000 (ts == cutoff anterior) EXPIROU exatamente no limite
of3b = {}
analyzer._compute_detailed_window(of3b, snapshot_com(120_001, trades), 1, 60_001, 120_001, 0.5, time.time())
fi3b = of3b.get("flow_imbalance")
print(f"[B2] now=120s+1 fi={fi3b} (esperado 1.0, so 61000)")

# ============================================================
# E) Guard de amostra minima (FLOW_IMBALANCE_MIN_TRADES=5)
# ============================================================
analyzer._flow_imbalance_min_trades = 5  # default real

def fi_da_janela(trades_j):
    ofx = {}
    analyzer._compute_detailed_window(ofx, snapshot_com(300_000, trades_j), 1, 0, 300_000, 0.5, time.time())
    return ofx.get("flow_imbalance")

t1 = {"ts": 210000, "price": 100.0, "qty": 1.0, "delta_btc": 1.0,  "delta_usd": 100.0, "side": "buy",  "sector": "large"}
t2 = {"ts": 210001, "price": 100.0, "qty": 1.0, "delta_btc": 1.0,  "delta_usd": 100.0, "side": "buy",  "sector": "large"}
t3 = {"ts": 210002, "price": 100.0, "qty": 1.0, "delta_btc": -1.0, "delta_usd": -100.0, "side": "sell", "sector": "large"}
t4 = {"ts": 210003, "price": 100.0, "qty": 1.0, "delta_btc": 1.0,  "delta_usd": 100.0, "side": "buy",  "sector": "large"}
t5 = {"ts": 210004, "price": 100.0, "qty": 1.0, "delta_btc": 1.0,  "delta_usd": 100.0, "side": "buy",  "sector": "large"}
t6 = {"ts": 210005, "price": 100.0, "qty": 1.0, "delta_btc": -1.0, "delta_usd": -100.0, "side": "sell", "sector": "large"}

fi_2 = fi_da_janela([t1, t2])
fi_4 = fi_da_janela([t1, t2, t3, t4])
fi_5 = fi_da_janela([t1, t2, t3, t4, t5])
fi_6 = fi_da_janela([t1, t2, t3, t4, t5, t6])   # 4 buy - 2 sell -> (400-200)/600 = 0.3333
print(f"[E1] 2 trades  -> fi={fi_2} (esperado: AUSENTE)")
print(f"[E2] 4 trades  -> fi={fi_4} (esperado: AUSENTE)")
print(f"[E3] 5 trades  -> fi={fi_5} (esperado: presente, 0.6)")
print(f"[E4] 6 trades  -> fi={fi_6} (esperado: presente, 0.3333)")
assert fi_2 is None and fi_4 is None, "janela com < min trades nao deve emitir flow_imbalance"
assert fi_5 is not None and abs(fi_5 - 0.6) < 1e-6, f"5 trades deveria emitir 0.6: {fi_5}"
assert fi_6 is not None and abs(fi_6 - 0.3333) < 1e-6, f"6 trades deveria emitir 0.3333: {fi_6}"

# ============================================================
# C) Pos-reset: flow_imbalance apos _reset_metrics (compartilhado com CVD)
# ============================================================
_t0 = int(time.time() * 1000) - 5_000
for i in range(50):
    analyzer.process_trade({"p": str(50000 + i), "q": "1.0", "T": _t0 + i, "m": False})

analyzer._reset_metrics()  # zera flow_trades, cvd, etc.
snap_pos = analyzer._create_snapshot(analyzer.last_reset_ms + 5)
of4 = {}
analyzer._compute_detailed_window(
    of4, snap_pos, 1,
    snap_pos['last_reset_ms'] + 5 - 60_000, snap_pos['last_reset_ms'] + 5,
    0.0, time.time()
)
print(f"[C1] pos-reset flow_trades={len(snap_pos['flow_trades'])} order_flow={of4}")
assert "flow_imbalance" not in of4, "sem volume, flow_imbalance nao deveria ser setado"

# Depois de 2 trades pos-reset, imbalance e calculado com amostra minuscula:
analyzer.process_trade({"p": "50050", "q": "1.0", "T": snap_pos['last_reset_ms'] + 100, "m": True})
analyzer.process_trade({"p": "50051", "q": "0.5", "T": snap_pos['last_reset_ms'] + 200, "m": True})
snap2 = analyzer._create_snapshot(snap_pos['last_reset_ms'] + 300)
of5 = {}
analyzer._compute_detailed_window(
    of5, snap2, 1,
    snap2['last_reset_ms'] - 60_000, snap2['last_reset_ms'],
    0.0, time.time()
)
fi5 = of5.get("flow_imbalance")
print(f"[C2] pos-reset+2trades fi={fi5} (esperado: AUSENTE - guard amostra minima)")
assert fi5 is None, f"2 trades pos-reset nao deveria emitir flow_imbalance: {fi5}"

# ============================================================
# D) Consumo no _build_ofi: compara com periodo longo? (bug cvd_div?)
# ============================================================
evento = {
    "fluxo_continuo": {"order_flow": {"flow_imbalance": -1.0}},
    "multi_tf": {"1h": {"tendencia": "Alta"}},  # tendencia oposta ao imbalance
}
ofi_payload = _build_ofi(evento)
print(f"[D1] _build_ofi(-1.0, 1h='Alta') -> {ofi_payload}")
assert ofi_payload.get("dir") == "SELL", "direcao deveria ser SELL (threshold fixo, sem trend_1h)"

# Sem order_flow: cai para microstructure / vazio, sem valor 0.0 enganoso
ofi_vazio = _build_ofi({"fluxo_continuo": {"order_flow": {}}})
print(f"[D2] _build_ofi(sem order_flow)   -> {ofi_vazio or '(vazio)'}")

print("=" * 62)
print("RESULTADO: TODOS OS TESTES PASSARAM")
print("=" * 62)
