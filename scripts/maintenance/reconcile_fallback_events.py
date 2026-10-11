#!/usr/bin/env python3
"""
CLI de Diagnóstico de Eventos Fallback (Gate G4-D - Parte A).

Opera exclusivamente em modo READ-ONLY (--dry-run).
Qualquer opção ou tentativa de escrita (--apply, --force, --delete, --repair)
é sumariamente rejeitada por contrato de segurança.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao PYTHONPATH para permitir imports do projeto
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from events.fallback_reconciler import FallbackReconciler, DryRunReport, EventClassification

FORBIDDEN_FLAGS = ["--apply", "--force", "--delete", "--repair", "-f"]


def parse_args():
    # Validação antecipada e rigorosa contra opções de escrita
    for arg in sys.argv[1:]:
        if arg in FORBIDDEN_FLAGS or arg.startswith("--apply") or arg.startswith("--repair"):
            print(f"ERRO DE SEGURANÇA: O parâmetro '{arg}' é expressamente PROIBIDO.", file=sys.stderr)
            print("Nesta fase (G4-D Parte A), apenas o modo offline '--dry-run' é suportado.", file=sys.stderr)
            print("Nenhuma escrita em SQLite, JSONL ou fallback é autorizada.", file=sys.stderr)
            sys.exit(2)

    parser = argparse.ArgumentParser(
        description="Auditoria e reconciliação offline de eventos fallback (SOMENTE --dry-run).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Executa auditoria em modo somente leitura sem realizar modificações em nenhum arquivo (Padrão).",
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
        "--json-output",
        action="store_true",
        help="Exibe o relatório em formato JSON estruturado.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Confirmação de modo
    if not args.dry_run:
        print("ERRO: Apenas modo --dry-run é permitido.", file=sys.stderr)
        sys.exit(2)

    # Identificação dos arquivos de fallback
    fallback_files = []
    if args.file:
        if not args.file.exists():
            print(f"ERRO: Arquivo especificado não encontrado: {args.file}", file=sys.stderr)
            sys.exit(1)
        fallback_files.append(args.file)
    else:
        fb_dir = args.fallback_dir
        if fb_dir.exists():
            # Buscar arquivos no padrão eventos_*.json e eventos_*.jsonl
            for ext in ("*.jsonl", "*.json"):
                fallback_files.extend(sorted(fb_dir.glob(ext)))
        else:
            print(f"AVISO: Diretório de fallback não encontrado: {fb_dir}", file=sys.stderr)

    if not fallback_files:
        print("Nenhum arquivo de fallback encontrado para análise.")
        sys.exit(0)

    reconciler = FallbackReconciler(
        sqlite_path=args.sqlite_path,
        jsonl_path=args.jsonl_path,
        archive_dir=args.archive_dir,
        repo_root=PROJECT_ROOT,
    )

    report = reconciler.run_dry_run(fallback_files)

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return

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


if __name__ == "__main__":
    main()
