#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI de Diagnóstico e Governança de Reconciliação de Eventos Fallback (Gate G4-D B3-B2-C2-A).

Suporta modo de auditoria offline (--dry-run, padrão) e interface de governança
para aplicação controlada (--apply, --confirm-apply, --reason).

Nesta fase (C2-A), mutações financeiras permanecem ESTRITAMENTE DESABILITADAS.
Qualquer execução autorizada de --apply valida a aquisição exclusiva do lock de manutenção,
porém encerra com bloqueio explícito APPLY_NOT_ENABLED_C2A, garantindo zero mutação.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Adiciona o diretório raiz ao PYTHONPATH para permitir imports do projeto
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime_lock import (
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
)
from events.fallback_reconciler import (
    FallbackReconciler,
    DryRunReport,
    EventClassification,
)

# Flags terminantemente proibidas de mutação sem contrato
FORBIDDEN_FLAGS = ["--force", "--delete", "--repair", "-f"]


def parse_args() -> argparse.Namespace:
    raw_args = sys.argv[1:]

    # 1. Validação antecipada contra flags expressamente proibidas
    for arg in raw_args:
        if arg in FORBIDDEN_FLAGS or arg.startswith("--repair") or arg.startswith("--force"):
            print(f"ERRO DE SEGURANÇA: O parâmetro '{arg}' é expressamente PROIBIDO.", file=sys.stderr)
            print("Operações de reparo cego ou force são vedadas por contrato de segurança.", file=sys.stderr)
            sys.exit(2)

    parser = argparse.ArgumentParser(
        description="Auditoria e reconciliação controlada de eventos fallback (Gate G4-D B3-B2-C2-A).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Executa auditoria em modo somente leitura sem realizar modificações em nenhum arquivo (Padrão).",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Solicita execução de reconciliação com escrita nos destinos operacionais (Requer --confirm-apply e --reason).",
    )

    parser.add_argument(
        "--confirm-apply",
        action="store_true",
        default=False,
        help="Confirmação explícita obrigatória para permitir reconciliação com escrita.",
    )

    parser.add_argument(
        "--reason",
        type=str,
        default="",
        help="Justificativa auditável da reconciliação (obrigatória com pelo menos 15 caracteres ao usar --apply).",
    )

    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Diretório de destino para snapshots forenses pré-reconciliação (padrão: backups/reconcile_snapshots).",
    )

    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Caminho para arquivo específico de fallback a analisar (ex: fallback_events/eventos_20260307.json).",
    )

    parser.add_argument(
        "--fallback-dir",
        type=Path,
        default=Path("fallback_events"),
        help="Diretório contendo arquivos de fallback (padrão: fallback_events).",
    )

    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=Path("dados/trading_bot.db"),
        help="Caminho para o banco SQLite operacional (padrão: dados/trading_bot.db).",
    )

    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=Path("dados/eventos_fluxo.jsonl"),
        help="Caminho para o JSONL operacional ativo (padrão: dados/eventos_fluxo.jsonl).",
    )

    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("dados/archive"),
        help="Diretório de segmentos arquivados G4-C (padrão: dados/archive).",
    )

    parser.add_argument(
        "--lock-path",
        type=Path,
        default=None,
        help="Caminho para o lock operacional (padrão: dados/.operational_runtime.lock).",
    )

    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Exibe o relatório em formato JSON estruturado.",
    )

    args = parser.parse_args()

    # 2. Regra de exclusão mútua entre --dry-run e --apply
    has_explicit_dry_run = "--dry-run" in raw_args
    if has_explicit_dry_run and args.apply:
        print("ERRO DE GOVERNANÇA: --dry-run e --apply são mutuamente exclusivos.", file=sys.stderr)
        sys.exit(2)

    # 3. Regra de confirmação para --apply
    if args.apply:
        if not args.confirm_apply:
            print(
                "ERRO DE GOVERNANÇA: --apply exige confirmação explícita via --confirm-apply. Uso isolado de --apply é PROIBIDO.",
                file=sys.stderr,
            )
            sys.exit(2)

        clean_reason = (args.reason or "").strip()
        if len(clean_reason) < 15:
            print(
                "ERRO DE GOVERNANÇA: --apply exige justificativa auditável (--reason) com pelo menos 15 caracteres significativos.",
                file=sys.stderr,
            )
            sys.exit(2)

    # 3b. Rejeição de confirmação sem apply
    if args.confirm_apply and not args.apply:
        print("ERRO DE GOVERNANÇA: --confirm-apply requer explicitamente a flag --apply.", file=sys.stderr)
        sys.exit(2)

    # 4. Modo padrão se nenhuma das duas for passada explicitamente
    if not args.apply:
        args.dry_run = True

    return args



def main() -> int:
    args = parse_args()

    # Resolução de caminhos canônicos
    sqlite_path = args.sqlite_path.expanduser().resolve()
    jsonl_path = args.jsonl_path.expanduser().resolve()
    archive_dir = args.archive_dir.expanduser().resolve()
    fallback_dir = args.fallback_dir.expanduser().resolve()
    data_dir = sqlite_path.parent
    lock_path = (args.lock_path or (data_dir / ".operational_runtime.lock")).expanduser().resolve()

    # =========================================================================
    # FLUXO DE GOVERNANÇA: --apply (Fase C2-A Bloqueio Controlado)
    # =========================================================================
    if args.apply:
        op_lock = OperationalRuntimeLock(lock_path=lock_path, holder_name="fallback_reconciler_cli")
        try:
            with op_lock.maintenance_session(
                allowed_states=[
                    RuntimeState.CLEAN,
                    RuntimeState.DIRTY,
                    RuntimeState.UNKNOWN,
                    "RUNNING_RESIDUAL",
                ]
            ) as maint_info:
                # Exclusividade física no kernel validada com sucesso (zero processo concorrente ativo).
                # Preserva o estado original do lock intacto.
                # Bloqueio explícito da Fase C2-A:
                msg = (
                    "APPLY_NOT_ENABLED_C2A: A infraestrutura de governança e lock foi validada com sucesso, "
                    "mas mutações financeiras continuam estritamente desabilitadas na Fase C2-A. "
                    "Nenhuma escrita realizada nos destinos operacionais."
                )
                if args.json_output:
                    print(
                        json.dumps(
                            {
                                "status": "APPLY_NOT_ENABLED_C2A",
                                "message": msg,
                                "mutations_performed": False,
                                "lock_holder": op_lock.holder_name,
                                "persisted_lock_status": str(maint_info.get("raw_status")),
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                else:
                    print(f"BLOQUEIO DE SEGURANÇA: {msg}", file=sys.stderr)
                return 1

        except RuntimeLockError as lock_err:
            err_msg = str(lock_err)
            if "Conflito de exclusividade" in err_msg or "retido por processo ativo" in err_msg:
                if args.json_output:
                    print(
                        json.dumps(
                            {
                                "status": "CONCURRENCY_LOCK_CONFLICT",
                                "error": err_msg,
                                "mutations_performed": False,
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                else:
                    print(f"ERRO DE EXCLUSIVIDADE: {err_msg}", file=sys.stderr)
                return 3
            else:
                if args.json_output:
                    print(
                        json.dumps(
                            {
                                "status": "LOCK_ERROR",
                                "error": err_msg,
                                "mutations_performed": False,
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                else:
                    print(f"ERRO DE LOCK: {err_msg}", file=sys.stderr)
                return 1

    # =========================================================================
    # FLUXO PADRÃO: --dry-run (SOMENTE LEITURA)
    # =========================================================================
    fallback_files: List[Path] = []
    if args.file:
        f_p = args.file.expanduser().resolve()
        if not f_p.exists():
            print(f"ERRO: Arquivo especificado não encontrado: {f_p}", file=sys.stderr)
            return 1
        fallback_files.append(f_p)
    else:
        if fallback_dir.exists():
            for ext in ("*.jsonl", "*.json"):
                fallback_files.extend(sorted(fallback_dir.glob(ext)))
        else:
            print(f"AVISO: Diretório de fallback não encontrado: {fallback_dir}", file=sys.stderr)

    if not fallback_files:
        if args.json_output:
            print(json.dumps({"mode": "DRY_RUN_READ_ONLY", "files_analyzed": [], "total_records_read": 0, "message": "Nenhum arquivo de fallback encontrado."}, indent=2))
        else:
            print("Nenhum arquivo de fallback encontrado para análise.")
        return 0

    reconciler = FallbackReconciler(
        sqlite_path=sqlite_path,
        jsonl_path=jsonl_path,
        archive_dir=archive_dir,
        repo_root=PROJECT_ROOT,
    )

    report = reconciler.run_dry_run(fallback_files)

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    # Exibição formatada em texto amigável (pt-BR)
    print("=" * 70)
    print("RELATÓRIO DE AUDITORIA DE FALLBACK — MODO DRY-RUN (READ-ONLY)")
    print("=" * 70)
    print(f"Arquivos analisados ({len(report.files_analyzed)}):")
    for f in report.files_analyzed:
        is_fix = " [FIXTURE BLOQUEADA]" if f in report.fixture_files_detected else ""
        print(f"  - {f}{is_fix}")

    print(f"\nTotal de eventos lidos: {report.total_records_read}")
    print("\nClassificação dos Eventos:")
    for status, count in report.counts.items():
        print(f"  - {status:28s}: {count:5d}")

    print("\nOperações Hipotéticas (Nenhuma executada nesta etapa):")
    for op, count in report.hypothetical_operations.items():
        print(f"  - {op:28s}: {count:5d}")

    if report.fixture_files_detected:
        print("\nFixtures de Teste Detectadas e Preservadas:")
        for fix in report.fixture_files_detected:
            print(f"  - {fix} (Preservada contra alteração; importação bloqueada)")

    if report.errors:
        print("\nAvisos / Erros Encontrados:")
        for err in report.errors:
            print(f"  ! {err}")

    print("=" * 70)
    print("STATUS FINAL: AUDITORIA CONCLUÍDA EM MODO READ-ONLY COM SUCESSO.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
