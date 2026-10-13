#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Inicialização e Bootstrap Controlado do Lock Operacional (Gate G4-D B3-B2-A).

Opera por padrão em modo estritamente READ-ONLY (Diagnóstico).
Para aplicar a inicialização do ambiente e registrar o estado inicial CLEAN,
exige autorização explícita via parâmetro '--confirm-bootstrap'.

Requisitos de Segurança:
1. Operação offline: robô desligado e nenhum writer concorrente ativo.
2. Exclusividade garantida por lock físico no kernel do SO.
3. Validação prévia de integridade read-only (SQLite, JSONL e Archive).
4. Criação do manifesto permanente anti-bypass dados/.runtime_initialized.manifest.
5. Persistência controlada de estado CLEAN no lock operacional com fsync.
6. Execução estritamente única (bloqueia bootstrap repetido).
7. Nenhuma escrita em SQLite, JSONL ou importação de arquivos de fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from pathlib import Path

# Adiciona o diretório raiz ao PYTHONPATH para permitir imports canônicos
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.runtime_lock import (
    OperationalRuntimeLock,
    RuntimeLockError,
    RuntimeState,
    MANIFEST_FILENAME,
    read_runtime_manifest,
    bootstrap_runtime_environment,
)
from events.fallback_reconciler import (
    verify_sqlite_integrity,
    verify_jsonl_integrity,
    verify_archive_integrity,
)

logger = logging.getLogger("scripts.maintenance.bootstrap_runtime_lock")

FORBIDDEN_FLAGS = ["--apply", "--force", "--delete", "--repair", "-f"]


def parse_args() -> argparse.Namespace:
    """Valida argumentos de linha de comando rejeitando opções de alteração operacional proibidas."""
    for arg in sys.argv[1:]:
        if arg in FORBIDDEN_FLAGS or arg.startswith("--apply") or arg.startswith("--repair"):
            print(f"ERRO DE SEGURANÇA: O parâmetro '{arg}' é expressamente PROIBIDO.", file=sys.stderr)
            print("O script de bootstrap opera somente em diagnóstico ou com '--confirm-bootstrap'.", file=sys.stderr)
            print("Nenhuma escrita em SQLite, JSONL ou fallback é autorizada.", file=sys.stderr)
            sys.exit(2)

    parser = argparse.ArgumentParser(
        description="Bootstrap controlado e diagnóstico do runtime lock operacional (Gate G4-D B3-B2-A).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--confirm-bootstrap",
        action="store_true",
        default=False,
        help="Autorização explícita para criar o manifesto e registrar o estado inicial CLEAN.",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dados"),
        help="Diretório de dados operacionais (padrão: dados).",
    )

    parser.add_argument(
        "--lock-path",
        type=Path,
        default=None,
        help="Caminho do lock operacional (padrão: <data-dir>/.operational_runtime.lock).",
    )

    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=None,
        help="Caminho do banco SQLite (padrão: <data-dir>/trading_bot.db).",
    )

    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=None,
        help="Caminho do JSONL operacional (padrão: <data-dir>/eventos_fluxo.jsonl).",
    )

    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Diretório de arquivos de segmento arquivados (padrão: <data-dir>/archive).",
    )

    parser.add_argument(
        "--json-output",
        action="store_true",
        default=False,
        help="Emite o resultado em formato JSON estruturado.",
    )

    return parser.parse_args()


def run_diagnostic(
    data_dir: Path,
    lock_path: Path,
    sqlite_path: Path,
    jsonl_path: Path,
    archive_dir: Path,
) -> dict:
    """Executa diagnóstico puramente read-only do ambiente."""
    manifest_path = data_dir / MANIFEST_FILENAME
    manifest_exists = manifest_path.exists() and manifest_path.stat().st_size > 0
    manifest_data = read_runtime_manifest(manifest_path) if manifest_exists else None

    lock_exists = lock_path.exists() and lock_path.stat().st_size > 0
    lock_state = OperationalRuntimeLock.read_persisted_state(lock_path) if lock_exists else {
        "status": "MISSING",
        "raw_status": None,
        "reason": "lock_file_not_found",
    }

    # Verificações read-only de integridade
    sq_ok, sq_msg = verify_sqlite_integrity(sqlite_path)
    jl_ok, jl_msg = verify_jsonl_integrity(jsonl_path)
    arch_ok, arch_msg = verify_archive_integrity(archive_dir)

    all_integrity_ok = sq_ok and jl_ok and arch_ok
    already_initialized = manifest_exists and (lock_state.get("status") == RuntimeState.CLEAN)

    ready_for_bootstrap = all_integrity_ok and not already_initialized

    return {
        "mode": "DIAGNOSTIC_READ_ONLY",
        "data_dir": str(data_dir),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_exists,
        "manifest_uuid": manifest_data.get("environment_uuid") if manifest_data else None,
        "lock_path": str(lock_path),
        "lock_exists": lock_exists,
        "lock_status": str(lock_state.get("status")),
        "already_initialized": already_initialized,
        "integrity": {
            "sqlite_ok": sq_ok,
            "sqlite_details": sq_msg,
            "jsonl_ok": jl_ok,
            "jsonl_details": jl_msg,
            "archive_ok": arch_ok,
            "archive_details": arch_msg,
        },
        "ready_for_bootstrap": ready_for_bootstrap,
    }


def main() -> int:
    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    lock_path = (args.lock_path or (data_dir / ".operational_runtime.lock")).expanduser().resolve()
    sqlite_path = (args.sqlite_path or (data_dir / "trading_bot.db")).expanduser().resolve()
    jsonl_path = (args.jsonl_path or (data_dir / "eventos_fluxo.jsonl")).expanduser().resolve()
    archive_dir = (args.archive_dir or (data_dir / "archive")).expanduser().resolve()

    if not args.confirm_bootstrap:
        # Modo Padrão: Somente Diagnóstico
        diag = run_diagnostic(data_dir, lock_path, sqlite_path, jsonl_path, archive_dir)
        if args.json_output:
            print(json.dumps(diag, indent=2))
        else:
            print("============================================================")
            print("DIAGNÓSTICO DE RUNTIME LOCK E INICIALIZAÇÃO (READ-ONLY)")
            print("============================================================")
            print(f"Diretório de dados: {diag['data_dir']}")
            print(f"Manifesto existe:   {diag['manifest_exists']} (UUID: {diag['manifest_uuid']})")
            print(f"Lock existe:        {diag['lock_exists']} (Status: {diag['lock_status']})")
            print(f"Já inicializado:    {diag['already_initialized']}")
            print("Integridade:")
            print(f"  - SQLite:  {'OK' if diag['integrity']['sqlite_ok'] else 'FALHA'} ({diag['integrity']['sqlite_details']})")
            print(f"  - JSONL:   {'OK' if diag['integrity']['jsonl_ok'] else 'FALHA'} ({diag['integrity']['jsonl_details']})")
            print(f"  - Archive: {'OK' if diag['integrity']['archive_ok'] else 'FALHA'} ({diag['integrity']['archive_details']})")
            print("------------------------------------------------------------")
            if diag['already_initialized']:
                print("AVISO: Ambiente já inicializado com sucesso. Bootstrap adicional desnecessário.")
            elif diag['ready_for_bootstrap']:
                print("PRONTO: Ambiente íntegro. Execute com '--confirm-bootstrap' para autorizar inicialização.")
            else:
                print("IMPEDIMENTO: Verificações de integridade falharam ou existem bloqueios.")
            print("============================================================")
        return 0

    # Modo Execução Única: --confirm-bootstrap fornecido
    try:
        result = bootstrap_runtime_environment(
            data_dir=data_dir,
            lock_path=lock_path,
            sqlite_path=sqlite_path,
            jsonl_path=jsonl_path,
            archive_dir=archive_dir,
            holder_name="bootstrap_runtime_lock",
        )
        if args.json_output:
            print(json.dumps(result, indent=2))
        else:
            print("============================================================")
            print("BOOTSTRAP DE RUNTIME CONCLUÍDO COM SUCESSO (ESTADO: CLEAN)")
            print("============================================================")
            print(f"UUID do Ambiente: {result['environment_uuid']}")
            print(f"Manifesto gravado: {result['manifest_path']}")
            print(f"Lock persistido:  {result['lock_path']}")
            print(f"Status Inicial:   {result['status']}")
            print("============================================================")
        return 0
    except RuntimeLockError as rle:
        err_payload = {"success": False, "error": str(rle)}
        if args.json_output:
            print(json.dumps(err_payload, indent=2), file=sys.stderr)
        else:
            print(f"ERRO DE BOOTSTRAP: {rle}", file=sys.stderr)
        return 2
    except Exception as exc:
        err_payload = {"success": False, "error": f"Erro inesperado: {exc}"}
        if args.json_output:
            print(json.dumps(err_payload, indent=2), file=sys.stderr)
        else:
            print(f"ERRO CRÍTICO INESPERADO NO BOOTSTRAP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
