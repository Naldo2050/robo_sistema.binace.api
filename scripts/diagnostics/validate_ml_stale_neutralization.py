#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/diagnostics/validate_ml_stale_neutralization.py

Validação 2: Neutralização de ML_STALE em produção real.
Verifica se o modo híbrido permaneceu estritamente desabilitado e se
nenhuma predição do modelo obsoleto de spot foi utilizada em futuros.
"""
import sys
import re
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def validate_ml_neutralization(log_path: str = "logs/collect_2h_20260906_203537.log"):
    print("=" * 80)
    print("VALIDAÇÃO 2 — NEUTRALIZAÇÃO DE ML_STALE EM PRODUÇÃO REAL DE 2 HORAS")
    print(f"Log avaliado: {log_path}")
    print("=" * 80)

    p = Path(log_path)
    if not p.exists():
        print(f"❌ Arquivo de log {log_path} não encontrado!")
        return 1

    with open(p, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # 1. Checar inicialização dos metadados ML
    meta_matches = re.findall(r"Metadados ML carregados:\s*valid_for_futures=(\w+),\s*ml_stale=(\w+)", text)
    print(f"Ocorrências de carregamento de metadados ML: {len(meta_matches)}")
    for vff, mls in meta_matches:
        print(f"  • valid_for_futures={vff}, ml_stale={mls}")

    # 2. Checar se modo híbrido foi desabilitado
    hybrid_disabled = "hybrid mode DESABILITADO (usando LLM only)" in text
    hybrid_init = '"status": "hybrid_disabled"' in text
    print(f"\nStatus do Modo Híbrido:")
    print(f"  • Flag 'hybrid mode DESABILITADO' presente: {hybrid_disabled}")
    print(f"  • Evento 'ml_engine_initialized' com status 'hybrid_disabled': {hybrid_init}")

    # 3. Verificar se houve qualquer predição ativa de XGBoost para futuros
    xgb_active = re.findall(r"XGBoost predizendo para futuros|hybrid_mode_active", text)
    print(f"  • Tentativas de predição ativa do XGBoost legado: {len(xgb_active)}")

    # 4. Violações
    violations_vff = sum(1 for vff, _ in meta_matches if vff == "True")
    violations_mls = sum(1 for _, mls in meta_matches if mls == "False")

    print(f"\nViolações identificadas:")
    print(f"  • valid_for_futures=True: {violations_vff}")
    print(f"  • ml_stale=False:         {violations_mls}")

    if violations_vff == 0 and violations_mls == 0 and hybrid_disabled and len(xgb_active) == 0:
        print("\n✅ CONFORME: ML_STALE 100% neutralizado. Modelo obsoleto de spot completamente isolado.")
        print("Veredito da Validação 2: APROVADO")
        return 0
    else:
        print("\n❌ INCONFORME: Vazamento de modelo legado detectado!")
        print("Veredito da Validação 2: REPROVADO")
        return 1


if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/collect_2h_20260906_203537.log"
    sys.exit(validate_ml_neutralization(log_file))
