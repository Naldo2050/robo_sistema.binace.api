#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Worker Isolado para Testes de Crash e Interrupção Abrupta de Subprocessos (Gate G4-D B3-B2-C2-B2.3).

Executado exclusivamente como subprocesso filho pelo processo pai de testes (pytest).
Opera estritamente dentro de um sandbox temporário hermético em tmp_path.

Protocolo de Barreira:
1. Intercepta o ponto de injeção exato correspondente ao cenário pretendido (C0 a C12 e variantes).
2. Emite marcador inequívoco: CRASH_BARRIER:REACHED:<scenario>:<execution_id>:<stage>
3. Executa flush imediato do canal IPC (sys.stdout.flush()).
4. Permanece bloqueado indefinidamente aguardando o encerramento forçado (proc.kill) pelo pai.
5. Não invoca os._exit() espontaneamente no fluxo principal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from pathlib import Path

# Garante que o projeto está no PYTHONPATH
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Forçar stdout sem buffer e encoding UTF-8
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from common.runtime_lock import OperationalRuntimeLock
import events.fallback_reconciler as fallback_reconciler
from events.fallback_reconciler import execute_reconciliation_transaction_hermetic


def signal_barrier_and_wait(scenario: str, execution_id: str, stage: str) -> None:
    """
    Emite o marcador inequívoco de barreira e bloqueia o processo até que o pai o encerre com proc.kill().
    """
    marker = f"CRASH_BARRIER:REACHED:{scenario}:{execution_id}:{stage}\n"
    sys.stdout.write(marker)
    sys.stdout.flush()

    # Bloqueia indefinidamente sem consumo de CPU até a interrupção pelo processo pai
    barrier_event = threading.Event()
    barrier_event.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker de crash determinístico para reconciliação.")
    parser.add_argument("--scenario", required=True, help="Cenário de injeção (ex: C0..C12, C5_PARTIAL, C7_PARTIAL_LINE, etc.)")
    parser.add_argument("--execution-id", required=True, help="Identificador exclusivo da execução do teste")
    parser.add_argument("--data-dir", required=True, type=Path, help="Diretório de dados no sandbox")
    parser.add_argument("--sqlite-path", required=True, type=Path, help="Caminho do banco SQLite")
    parser.add_argument("--jsonl-path", required=True, type=Path, help="Caminho do arquivo JSONL")
    parser.add_argument("--archive-dir", required=True, type=Path, help="Caminho do diretório archive")
    parser.add_argument("--fallback-file", required=True, type=Path, help="Caminho do arquivo fallback de origem")
    parser.add_argument("--manifest-path", required=True, type=Path, help="Caminho do manifesto operacional")
    parser.add_argument("--ledger-path", required=True, type=Path, help="Caminho do livro-razão de auditoria")
    parser.add_argument("--backup-dir", required=True, type=Path, help="Caminho base de backups")
    parser.add_argument("--env-uuid", required=True, type=str, help="UUID do ambiente")
    parser.add_argument("--lock-path", required=True, type=Path, help="Caminho do lock operacional")
    parser.add_argument("--operator-reason", default="Justificativa operacional hermética de teste C2-B2.3", help="Razão da transação")

    args = parser.parse_args()
    scenario = args.scenario
    execution_id = args.execution_id

    # =========================================================================
    # INSTRUMENTAÇÃO POR CENÁRIO (C0 a C12)
    # =========================================================================

    if scenario == "C0":
        # Interrupção antes de LOCK
        signal_barrier_and_wait("C0", execution_id, "BEFORE_LOCK")
        # Se por algum motivo continuar:
        return

    elif scenario == "C1":
        # Interrupção após LOCK, antes do PREFLIGHT
        orig_acq = OperationalRuntimeLock.acquire_maintenance

        def wrapped_acq(self, *a, **kw):
            res = orig_acq(self, *a, **kw)
            signal_barrier_and_wait("C1", execution_id, "AFTER_LOCK_BEFORE_PREFLIGHT")
            return res

        OperationalRuntimeLock.acquire_maintenance = wrapped_acq

    elif scenario == "C2":
        # Interrupção após PREFLIGHT, antes do SNAPSHOT
        orig_snap = fallback_reconciler.create_reconciliation_snapshot

        def wrapped_snap_c2(*a, **kw):
            signal_barrier_and_wait("C2", execution_id, "AFTER_PREFLIGHT_BEFORE_SNAPSHOT")
            return orig_snap(*a, **kw)

        fallback_reconciler.create_reconciliation_snapshot = wrapped_snap_c2

    elif scenario == "C3":
        # Interrupção após SNAPSHOT, antes de PREPARED
        orig_snap = fallback_reconciler.create_reconciliation_snapshot

        def wrapped_snap_c3(*a, **kw):
            res = orig_snap(*a, **kw)
            signal_barrier_and_wait("C3", execution_id, "AFTER_SNAPSHOT_BEFORE_PREPARED")
            return res

        fallback_reconciler.create_reconciliation_snapshot = wrapped_snap_c3

    elif scenario == "C4":
        # Interrupção após PREPARED confirmado, antes de SQLite
        orig_audit = fallback_reconciler.record_reconciliation_audit

        def wrapped_audit_c4(p, entry):
            res = orig_audit(p, entry)
            if entry.get("reconciliation_phase") == "PREPARED":
                signal_barrier_and_wait("C4", execution_id, "AFTER_PREPARED_BEFORE_SQLITE")
            return res

        fallback_reconciler.record_reconciliation_audit = wrapped_audit_c4

    elif scenario in ("C5", "C5_PARTIAL"):
        # Interrupção durante escrita SQLite (1º evento gravado e commitado; interrompe antes do 2º)
        sq_count = 0
        orig_insert_sq = fallback_reconciler.insert_event_to_sqlite

        def wrapped_insert_sq(p, ev):
            nonlocal sq_count
            res = orig_insert_sq(p, ev)
            sq_count += 1
            if sq_count == 1:
                signal_barrier_and_wait(scenario, execution_id, "DURING_SQLITE_PARTIAL_COMMITTED")
            return res

        fallback_reconciler.insert_event_to_sqlite = wrapped_insert_sq

    elif scenario == "C6":
        # Interrupção após SQLite completo, antes de JSONL
        orig_prep_jl = fallback_reconciler.validate_and_prepare_jsonl_target

        def wrapped_prep_jl(*a, **kw):
            signal_barrier_and_wait("C6", execution_id, "AFTER_SQLITE_BEFORE_JSONL")
            return orig_prep_jl(*a, **kw)

        fallback_reconciler.validate_and_prepare_jsonl_target = wrapped_prep_jl

    elif scenario in ("C7", "C7_PARTIAL_BATCH"):
        # Interrupção durante append JSONL (1º evento appendado, interrompe antes do 2º)
        jl_count = 0
        orig_append_jl = fallback_reconciler.append_event_to_jsonl

        def wrapped_append_jl_batch(p, ev, prefix_separator=""):
            nonlocal jl_count
            res = orig_append_jl(p, ev, prefix_separator=prefix_separator)
            jl_count += 1
            if jl_count == 1:
                signal_barrier_and_wait(scenario, execution_id, "DURING_JSONL_PARTIAL_BATCH")
            return res

        fallback_reconciler.append_event_to_jsonl = wrapped_append_jl_batch

    elif scenario == "C7_PARTIAL_LINE":
        # Interrupção com linha parcial de JSONL sem newline
        def wrapped_append_jl_partial(p, ev, prefix_separator=""):
            p_path = Path(p)
            with open(p_path, "ab") as f:
                f.write(b'{"event_id": "EVT-TRUNCATED", "epoch_ms": 1700')
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            signal_barrier_and_wait("C7_PARTIAL_LINE", execution_id, "DURING_JSONL_TRUNCATED_LINE")

        fallback_reconciler.append_event_to_jsonl = wrapped_append_jl_partial

    elif scenario == "C7_COMPLETE_NO_FSYNC":
        # Linha completa no arquivo sem fsync
        def wrapped_append_jl_nofsync(p, ev, prefix_separator=""):
            p_path = Path(p)
            with open(p_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")
                f.flush()
            signal_barrier_and_wait("C7_COMPLETE_NO_FSYNC", execution_id, "DURING_JSONL_NO_FSYNC")

        fallback_reconciler.append_event_to_jsonl = wrapped_append_jl_nofsync

    elif scenario == "C8":
        # Interrupção após APPLY completo, antes da VERIFICAÇÃO
        orig_verif = fallback_reconciler.verify_reconciliation_targets

        def wrapped_verif_c8(*a, **kw):
            signal_barrier_and_wait("C8", execution_id, "AFTER_APPLY_BEFORE_VERIFICATION")
            return orig_verif(*a, **kw)

        fallback_reconciler.verify_reconciliation_targets = wrapped_verif_c8

    elif scenario == "C9":
        # Interrupção após VERIFICAÇÃO, antes de MANIFESTO
        orig_verif = fallback_reconciler.verify_reconciliation_targets

        def wrapped_verif_c9(*a, **kw):
            res = orig_verif(*a, **kw)
            signal_barrier_and_wait("C9", execution_id, "AFTER_VERIFICATION_BEFORE_MANIFEST")
            return res

        fallback_reconciler.verify_reconciliation_targets = wrapped_verif_c9

    elif scenario == "C10":
        # Interrupção após MANIFESTO, antes de COMMITTED
        import common.runtime_lock as runtime_lock
        orig_update_man = runtime_lock.update_runtime_manifest_reconciliation_state

        def wrapped_update_man_c10(*a, **kw):
            res = orig_update_man(*a, **kw)
            signal_barrier_and_wait("C10", execution_id, "AFTER_MANIFEST_BEFORE_COMMITTED")
            return res

        runtime_lock.update_runtime_manifest_reconciliation_state = wrapped_update_man_c10

    elif scenario in ("C11", "C11_NO_BYTES"):
        # Interrupção antes de gravar qualquer byte de COMMITTED
        orig_audit = fallback_reconciler.record_reconciliation_audit

        def wrapped_audit_c11_nobytes(p, entry):
            if entry.get("reconciliation_phase") == "COMMITTED":
                signal_barrier_and_wait(scenario, execution_id, "BEFORE_COMMITTED_WRITE")
            return orig_audit(p, entry)

        fallback_reconciler.record_reconciliation_audit = wrapped_audit_c11_nobytes

    elif scenario == "C11_PARTIAL_BYTES":
        # Interrupção durante gravação de COMMITTED (linha truncada sem newline)
        orig_audit = fallback_reconciler.record_reconciliation_audit

        def wrapped_audit_c11_partial(p, entry):
            if entry.get("reconciliation_phase") == "COMMITTED":
                with open(p, "ab") as f:
                    f.write(b'{"audit_id": "' + entry["audit_id"].encode() + b'", "reconciliation_phase": "COMM')
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                signal_barrier_and_wait("C11_PARTIAL_BYTES", execution_id, "DURING_COMMITTED_PARTIAL_BYTES")
            return orig_audit(p, entry)

        fallback_reconciler.record_reconciliation_audit = wrapped_audit_c11_partial

    elif scenario == "C11_COMPLETE_NO_FSYNC":
        # Interrupção com COMMITTED completo no arquivo sem confirmação de fsync
        orig_audit = fallback_reconciler.record_reconciliation_audit

        def wrapped_audit_c11_nofsync(p, entry):
            if entry.get("reconciliation_phase") == "COMMITTED":
                with open(p, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                signal_barrier_and_wait("C11_COMPLETE_NO_FSYNC", execution_id, "COMMITTED_LINE_NO_FSYNC")
            return orig_audit(p, entry)

        fallback_reconciler.record_reconciliation_audit = wrapped_audit_c11_nofsync

    elif scenario == "C12":
        # Interrupção após COMMITTED confirmado com fsync, antes de liberar maintenance lock
        orig_audit = fallback_reconciler.record_reconciliation_audit

        def wrapped_audit_c12(p, entry):
            res = orig_audit(p, entry)
            if entry.get("reconciliation_phase") == "COMMITTED":
                signal_barrier_and_wait("C12", execution_id, "AFTER_COMMITTED_BEFORE_RELEASE")
            return res

        fallback_reconciler.record_reconciliation_audit = wrapped_audit_c12

    # Execução do orquestrador
    op_lock = OperationalRuntimeLock(lock_path=args.lock_path, holder_name=f"worker_{scenario}")
    res = execute_reconciliation_transaction_hermetic(
        data_dir=args.data_dir,
        sqlite_path=args.sqlite_path,
        jsonl_path=args.jsonl_path,
        archive_dir=args.archive_dir,
        source_fallback_file=args.fallback_file,
        manifest_path=args.manifest_path,
        ledger_path=args.ledger_path,
        backup_base_dir=args.backup_dir,
        environment_uuid=args.env_uuid,
        operator_reason=args.operator_reason,
        operational_lock=op_lock,
    )
    # Se completou sem crash (não esperado em cenários com injeção efetiva):
    print(f"WORKER_FINISHED:{res.status.value}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
