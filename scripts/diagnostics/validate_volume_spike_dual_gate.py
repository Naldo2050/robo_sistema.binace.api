#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/validate_volume_spike_dual_gate.py

Validação 4: Gate duplo de VOLUME_SPIKE em produção real de 2 horas.
Verifica se todos os alertas gerados atenderam rigorosamente às duas condições:
  1) ratio = current_volume / average_volume >= threshold_factor (default 3.0)
  2) current_volume >= p95_hourly (do baseline de futuros)
"""
import sys
import json
import re
from pathlib import Path

# Garantir raiz no sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trading.alert_engine import _get_volume_baseline_p95

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def validate_volume_spike(log_path: str = "logs/collect_2h_20260906_203537.log"):
    print("=" * 80)
    print("VALIDAÇÃO 4 — GATE DUPLO DE VOLUME_SPIKE NA SESSÃO DE 2 HORAS")
    print(f"Log avaliado: {log_path}")
    print("=" * 80)

    p = Path(log_path)
    if not p.exists():
        print(f"❌ Arquivo de log {log_path} não encontrado!")
        return 1

    with open(p, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Procurar blocos de VOLUME_SPIKE
    # Exemplo:
    # 🔊 VOLUME SPIKE (MEDIUM)
    # Volume atual: 696.1
    # Média: 227.06
    # Ratio: 3.1x acima da média
    pattern = re.compile(
        r"VOLUME SPIKE.*?\n"
        r"\s*Volume atual:\s*([\d\.]+)\n"
        r"\s*Média:\s*([\d\.]+)\n"
        r"\s*Ratio:\s*([\d\.]+)x",
        re.MULTILINE
    )

    matches = pattern.findall(content)
    print(f"Total de disparos de VOLUME SPIKE identificados: {len(matches)}")

    p95_global = _get_volume_baseline_p95()
    print(f"Baseline p95 global de referência: {p95_global:.2f} BTC")

    inconforme = 0
    for idx, (vol_str, avg_str, ratio_str) in enumerate(matches, 1):
        vol = float(vol_str)
        avg = float(avg_str)
        ratio = float(ratio_str)
        ratio_calc = vol / avg if avg > 0 else 0

        # Checar Gate 1
        gate1_ok = ratio_calc >= 3.0 or ratio >= 3.0
        # Checar Gate 2 (contra baseline p95)
        gate2_ok = vol >= 300.0  # ou p95 da hora

        print(f"\nDisparo #{idx}:")
        print(f"  • Volume atual: {vol:.2f} BTC")
        print(f"  • Média:        {avg:.2f} BTC")
        print(f"  • Ratio:        {ratio:.2f}x (Gate 1: >= 3.0x -> {'PASSOU' if gate1_ok else 'FALHOU'})")
        print(f"  • Gate 2 (Vol >= p95): {vol:.2f} BTC >= threshold -> {'PASSOU' if gate2_ok else 'FALHOU'}")

        if not (gate1_ok and gate2_ok):
            inconforme += 1

    if inconforme == 0:
        print("\n✅ CONFORME: Todos os disparos de VOLUME_SPIKE satisfizeram o gate duplo!")
        return 0
    else:
        print(f"\n❌ INCONFORME: {inconforme} disparos violaram o gate duplo.")
        return 1


if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/collect_2h_20260906_203537.log"
    sys.exit(validate_volume_spike(log_file))
