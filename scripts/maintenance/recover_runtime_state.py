#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ferramenta Offline de Diagnóstico de Recovery do Runtime Operacional (Gate G4-D B3-B2-B1).

Opera ESTRITAMENTE em modo READ-ONLY / DRY-RUN.
Diagnostica estados anômalos de runtime (DIRTY, UNKNOWN, RUNNING residual),
integridade física do banco relacional SQLite (schema events), JSONL principal,
segmentos arquivados G4-C e consistência cruzada com eventos de fallback.

NÃO executa transição de estado para CLEAN.
NÃO realiza escrita, reparo ou checkpoint WAL em nenhum arquivo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Garante consistência de codificação UTF-8 em stdout/stderr no Windows
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
    verify_environment_anti_bypass,
    validate_recovery_audit_ledger,
    update_runtime_manifest_recovery_state,
    verify_manifest_recovery_consistency,
    verify_manifest_reconciliation_consistency,
)
from events.fallback_reconciler import (
    compute_canonical_payload_hash,
    KNOWN_TEST_FIXTURE_NAMES,
    validate_reconciliation_audit_ledger,
)

logger = logging.getLogger("scripts.maintenance.recover_runtime_state")

# Flags estritamente proibidas que tentam mutação sem governança (fail-closed)
FORBIDDEN_FLAGS = [
    "--clear-dirty-state",
    "--force",
    "--repair",
    "--delete",
    "-f",
]


def parse_args() -> argparse.Namespace:
    """Valida argumentos de linha de comando rejeitando qualquer opção de mutação não governada."""
    for arg in sys.argv[1:]:
        if (
            arg in FORBIDDEN_FLAGS
            or arg.startswith("--clear")
            or arg.startswith("--repair")
            or arg.startswith("--force")
            or arg.startswith("--delete")
        ):
            print(
                f"ERRO DE SEGURANCA: O parametro '{arg}' e expressamente PROIBIDO.",
                file=sys.stderr,
            )
            print("Operações forçadas, destrutivas ou de reparo direto são vetadas.", file=sys.stderr)
            print("A recuperação segura exige exclusivamente '--confirm-recovery' acompanhada de '--reason'.", file=sys.stderr)
            sys.exit(2)

    parser = argparse.ArgumentParser(
        description="Diagnóstico e recuperação controlada do runtime operacional (Gate G4-D B3-B2-B3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Executa auditoria estritamente read-only sem modificar arquivos (Padrão).",
    )

    parser.add_argument(
        "--confirm-recovery",
        action="store_true",
        default=False,
        help="Autorização explícita para executar a recuperação autorizada e a transição para CLEAN.",
    )

    parser.add_argument(
        "--reason",
        type=str,
        default="",
        help="Justificativa operacional obrigatória para o recovery (mínimo de 15 caracteres significativos).",
    )

    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("backups/recovery_snapshots"),
        help="Diretório base onde o snapshot forense obrigatório será gerado (padrão: backups/recovery_snapshots).",
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
        "--fallback-dir",
        type=Path,
        default=Path("fallback_events"),
        help="Diretório contendo arquivos de fallback (padrão: fallback_events).",
    )

    parser.add_argument(
        "--json-output",
        action="store_true",
        default=False,
        help="Emite o relatório em formato JSON estruturado.",
    )

    args = parser.parse_args()

    if args.confirm_recovery:
        clean_reason = (args.reason or "").strip()
        if len(clean_reason) < 15:
            print(
                "ERRO DE SEGURANCA: O parametro '--reason' e OBRIGATORIO quando '--confirm-recovery' e utilizado "
                "e deve conter no minimo 15 caracteres significativos.",
                file=sys.stderr,
            )
            sys.exit(2)

    return args


def is_git_tracked_fixture(file_path: Path, repo_root: Path) -> bool:
    """Verifica se o arquivo de fallback é uma fixture de teste rastreada no Git."""
    if file_path.name in KNOWN_TEST_FIXTURE_NAMES:
        return True
    try:
        rel_path = file_path.relative_to(repo_root)
    except ValueError:
        rel_path = file_path

    try:
        res = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(rel_path)],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return res.returncode == 0 and bool(res.stdout.strip())
    except Exception:
        return False


def diagnose_runtime_recovery(
    data_dir: Path,
    lock_path: Optional[Path] = None,
    fallback_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    is_maintenance_session: bool = False,
) -> Dict[str, Any]:
    """
    Executa a auditoria diagnóstica exaustiva e somente leitura do runtime e dados operacionais.
    """
    d_dir = Path(data_dir).expanduser().resolve()
    l_path = Path(lock_path).expanduser().resolve() if lock_path else (d_dir / ".operational_runtime.lock")
    fb_dir = Path(fallback_dir).expanduser().resolve() if fallback_dir else (PROJECT_ROOT / "fallback_events")
    r_root = Path(repo_root).expanduser().resolve() if repo_root else PROJECT_ROOT

    manifest_path = d_dir / MANIFEST_FILENAME
    sqlite_path = d_dir / "trading_bot.db"
    jsonl_path = d_dir / "eventos_fluxo.jsonl"
    archive_dir = d_dir / "archive"

    conflicts: List[str] = []
    insufficient_evidences: List[str] = []
    recommendations: List[str] = []

    # =========================================================================
    # 1. DIAGNÓSTICO DO RUNTIME LOCK E ANTI-BYPASS
    # =========================================================================
    anti_bypass_ok, anti_bypass_reason = verify_environment_anti_bypass(d_dir, l_path)
    manifest_data = read_runtime_manifest(manifest_path) if manifest_path.exists() else None

    lock_exists = l_path.exists() and l_path.stat().st_size > 0
    lock_info = OperationalRuntimeLock.read_persisted_state(l_path) if lock_exists else {
        "status": RuntimeState.UNKNOWN,
        "raw_status": None,
        "reason": "lock_file_missing_or_empty",
    }

    # Verificação física de trava ativa no kernel
    is_kernel_locked = False
    if l_path.exists() and not is_maintenance_session:
        system = platform.system()
        try:
            test_f = open(l_path, "r+b")
            try:
                if system == "Windows":
                    import msvcrt
                    test_f.seek(0)
                    try:
                        msvcrt.locking(test_f.fileno(), msvcrt.LK_NBLCK, 1)
                        msvcrt.locking(test_f.fileno(), msvcrt.LK_UNLCK, 1)
                        is_kernel_locked = False
                    except OSError:
                        is_kernel_locked = True
                else:
                    import fcntl
                    try:
                        fcntl.flock(test_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(test_f.fileno(), fcntl.LOCK_UN)
                        is_kernel_locked = False
                    except OSError:
                        is_kernel_locked = True
            finally:
                test_f.close()
        except Exception:
            is_kernel_locked = True

    # Classificação formal do estado de recuperação
    persisted_status = lock_info.get("status")
    recovery_classification = "UNKNOWN"
    if not lock_exists and not manifest_data and not sqlite_path.exists() and not jsonl_path.exists():
        recovery_classification = "FRESH_INSTALL_UNINITIALIZED"
    elif not lock_exists and (sqlite_path.exists() or jsonl_path.exists() or manifest_data):
        recovery_classification = "LOCK_MISSING_ANTI_BYPASS_VIOLATION"
    elif lock_exists and not manifest_data:
        recovery_classification = "MANIFEST_MISSING_ANTI_BYPASS_VIOLATION"
    elif is_kernel_locked:
        recovery_classification = "ACTIVE_PROCESS_RUNNING_LOCKED"
    elif persisted_status == RuntimeState.CLEAN:
        recovery_classification = "CLEAN_OPERATIONAL"
    elif persisted_status == RuntimeState.DIRTY:
        recovery_classification = "DIRTY_INCOMPLETE_SHUTDOWN"
    elif persisted_status == RuntimeState.UNKNOWN:
        if "crash_detected" in str(lock_info.get("reason", "")):
            recovery_classification = "CRASH_RESIDUAL_RUNNING"
        elif "invalid_or_truncated_json" in str(lock_info.get("reason", "")):
            recovery_classification = "LOCK_METADATA_CORRUPTED"
        else:
            recovery_classification = "UNKNOWN_ANOMALOUS_STATE"

    lock_diagnostic = {
        "lock_path": str(l_path),
        "lock_exists": lock_exists,
        "persisted_status": str(persisted_status),
        "raw_status": lock_info.get("raw_status"),
        "reason": lock_info.get("reason"),
        "metadata": lock_info.get("metadata"),
        "is_kernel_locked": is_kernel_locked,
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists() and manifest_path.stat().st_size > 0,
        "environment_uuid": manifest_data.get("environment_uuid") if manifest_data else None,
        "anti_bypass_ok": anti_bypass_ok,
        "anti_bypass_reason": anti_bypass_reason,
        "recovery_classification": recovery_classification,
    }

    # =========================================================================
    # 2. DIAGNÓSTICO DO SQLITE (SCHEMA REAL 'events' READ-ONLY)
    # =========================================================================
    sqlite_catalog_by_id: Dict[str, Dict[str, Any]] = {}
    sqlite_catalog_by_corr: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}

    sq_exists = sqlite_path.exists()
    sq_size = sqlite_path.stat().st_size if sq_exists else 0
    sq_integrity_ok = False
    sq_integrity_msg = "Database file does not exist"
    sq_foreign_keys_ok = False
    sq_foreign_keys_msg = ""
    sq_schema_valid = False
    sq_columns: List[str] = []
    sq_events_count = 0
    sq_min_timestamp_ms: Optional[int] = None
    sq_max_timestamp_ms: Optional[int] = None

    wal_path = d_dir / "trading_bot.db-wal"
    shm_path = d_dir / "trading_bot.db-shm"
    wal_exists = wal_path.exists()
    shm_exists = shm_path.exists()
    wal_size = wal_path.stat().st_size if wal_exists else 0
    shm_size = shm_path.stat().st_size if shm_exists else 0

    if sq_exists:
        uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=10.0)
            cur = conn.cursor()

            # 1. PRAGMA integrity_check
            cur.execute("PRAGMA integrity_check")
            row = cur.fetchone()
            if row and row[0] == "ok":
                sq_integrity_ok = True
                sq_integrity_msg = "ok"
            else:
                sq_integrity_ok = False
                sq_integrity_msg = f"Failed integrity_check: {row}"

            # 2. PRAGMA foreign_key_check
            cur.execute("PRAGMA foreign_key_check")
            fk_rows = cur.fetchall()
            if not fk_rows:
                sq_foreign_keys_ok = True
                sq_foreign_keys_msg = "ok (zero violations)"
            else:
                sq_foreign_keys_ok = False
                sq_foreign_keys_msg = f"Violations found: {fk_rows}"

            # 3. PRAGMA table_info(events)
            cur.execute("PRAGMA table_info(events)")
            table_info = cur.fetchall()
            if table_info:
                sq_columns = [col[1] for col in table_info]
                expected_required = {"timestamp_ms", "event_type", "payload"}
                sq_schema_valid = expected_required.issubset(set(sq_columns))

                # 4. Leitura e catalogação de eventos
                cur.execute("""
                    SELECT id, timestamp_ms, event_type, COALESCE(symbol, ''), payload
                    FROM events
                """)
                for row_id, ts_ms, ev_type, sym, payload_raw in cur.fetchall():
                    sq_events_count += 1
                    try:
                        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
                        if not isinstance(payload, dict):
                            payload = {}
                    except Exception:
                        payload = {}

                    eid = payload.get("event_id") or payload.get("id")
                    eid_str = str(eid) if eid is not None and str(eid).strip() else None
                    ts_int = int(ts_ms) if ts_ms is not None and str(ts_ms).isdigit() else 0
                    ev_type_str = str(ev_type or "")
                    sym_str = str(sym or "")
                    canon_hash = compute_canonical_payload_hash(payload) if payload else ""

                    if sq_min_timestamp_ms is None or ts_int < sq_min_timestamp_ms:
                        sq_min_timestamp_ms = ts_int
                    if sq_max_timestamp_ms is None or ts_int > sq_max_timestamp_ms:
                        sq_max_timestamp_ms = ts_int

                    entry = {
                        "row_id": row_id,
                        "timestamp_ms": ts_int,
                        "event_type": ev_type_str,
                        "symbol": sym_str,
                        "canonical_hash": canon_hash,
                        "event_id": eid_str,
                    }
                    if eid_str:
                        sqlite_catalog_by_id[eid_str] = entry

                    corr_key = (ts_int, ev_type_str, sym_str)
                    sqlite_catalog_by_corr.setdefault(corr_key, []).append(entry)

        except Exception as e_sq:
            sq_integrity_ok = False
            sq_integrity_msg = f"Erro ao acessar SQLite: {e_sq}"
        finally:
            if conn:
                conn.close()

    sqlite_diagnostic = {
        "sqlite_path": str(sqlite_path),
        "exists": sq_exists,
        "size_bytes": sq_size,
        "integrity_check_ok": sq_integrity_ok,
        "integrity_check_details": sq_integrity_msg,
        "foreign_key_check_ok": sq_foreign_keys_ok,
        "foreign_key_check_details": sq_foreign_keys_msg,
        "schema_valid": sq_schema_valid,
        "columns_found": sq_columns,
        "events_count": sq_events_count,
        "min_timestamp_ms": sq_min_timestamp_ms,
        "max_timestamp_ms": sq_max_timestamp_ms,
        "wal_exists": wal_exists,
        "wal_size_bytes": wal_size,
        "shm_exists": shm_exists,
        "shm_size_bytes": shm_size,
    }

    # =========================================================================
    # 3. DIAGNÓSTICO DO JSONL ATIVO E ARCHIVE G4-C
    # =========================================================================
    jsonl_catalog_by_id: Dict[str, Dict[str, Any]] = {}
    jsonl_catalog_by_corr: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}

    jl_exists = jsonl_path.exists()
    jl_size = jsonl_path.stat().st_size if jl_exists else 0
    jl_lines_count = 0
    jl_corrupt_lines = 0
    jl_last_line_clean = True
    jl_corrupt_details: List[str] = []

    if jl_exists and jl_size > 0:
        try:
            with open(jsonl_path, "rb") as bf:
                content = bf.read()
            if content:
                # Checa terminação com quebra de linha
                if not content.endswith(b"\n") and not content.endswith(b"\r\n"):
                    jl_last_line_clean = False
                    jl_corrupt_details.append("Arquivo não termina com newline")

                lines = content.splitlines()
                for line_idx, raw_line in enumerate(lines, start=1):
                    s = raw_line.strip()
                    if not s:
                        continue
                    jl_lines_count += 1
                    try:
                        data = json.loads(s.decode("utf-8", errors="strict"))
                        if not isinstance(data, dict):
                            jl_corrupt_lines += 1
                            jl_corrupt_details.append(f"Linha {line_idx} não é um objeto JSON dict")
                            continue

                        eid = data.get("event_id") or data.get("id")
                        eid_str = str(eid) if eid is not None and str(eid).strip() else None
                        ep = data.get("epoch_ms") or data.get("timestamp_ms")
                        ep_int = int(ep) if ep is not None and str(ep).isdigit() else 0
                        ev_type = str(data.get("tipo_evento") or data.get("event_type") or data.get("type") or "")
                        sym = str(data.get("symbol") or data.get("ativo") or "")
                        canon_hash = compute_canonical_payload_hash(data)

                        entry = {
                            "source_file": str(jsonl_path),
                            "line_num": line_idx,
                            "epoch_ms": ep_int,
                            "tipo_evento": ev_type,
                            "symbol": sym,
                            "canonical_hash": canon_hash,
                            "is_archive": False,
                            "event_id": eid_str,
                        }
                        if eid_str:
                            if eid_str in jsonl_catalog_by_id:
                                jl_corrupt_details.append(f"event_id duplicado no JSONL: '{eid_str}' (linha {line_idx})")
                            jsonl_catalog_by_id[eid_str] = entry

                        corr_key = (ep_int, ev_type, sym)
                        jsonl_catalog_by_corr.setdefault(corr_key, []).append(entry)

                    except Exception as json_err:
                        jl_corrupt_lines += 1
                        jl_corrupt_details.append(f"Linha {line_idx} corrompida: {json_err}")
                        if line_idx == len(lines):
                            jl_last_line_clean = False
        except Exception as read_err:
            jl_corrupt_lines += 1
            jl_corrupt_details.append(f"Erro ao ler JSONL: {read_err}")

    jsonl_diagnostic = {
        "jsonl_path": str(jsonl_path),
        "exists": jl_exists,
        "size_bytes": jl_size,
        "total_lines_read": jl_lines_count,
        "corrupt_lines_count": jl_corrupt_lines,
        "last_line_clean": jl_last_line_clean,
        "corrupt_details": jl_corrupt_details[:10],
    }

    # Diagnóstico dos segmentos arquivados (Archive G4-C)
    arch_exists = archive_dir.exists()
    arch_segments: List[str] = []
    arch_tmp_files: List[str] = []
    arch_corrupt_lines = 0
    arch_corrupt_details: List[str] = []
    arch_total_events = 0

    if arch_exists:
        arch_tmp_files = [f.name for f in archive_dir.glob("*.tmp")]
        seg_files = sorted(archive_dir.glob("*.jsonl"))
        for seg in seg_files:
            if seg.is_symlink():
                conflicts.append(f"Origem operacional insegura: link simbólico detectado em archive ({seg.name}).")
                continue
            arch_segments.append(seg.name)
            try:
                with open(seg, "r", encoding="utf-8", errors="strict") as sf:
                    for s_line_idx, line in enumerate(sf, start=1):
                        s = line.strip()
                        if not s:
                            continue
                        arch_total_events += 1
                        try:
                            data = json.loads(s)
                            if not isinstance(data, dict):
                                arch_corrupt_lines += 1
                                arch_corrupt_details.append(f"Segmento {seg.name}:{s_line_idx} não é dict")
                                continue
                            eid = data.get("event_id") or data.get("id")
                            eid_str = str(eid) if eid is not None and str(eid).strip() else None
                            ep = data.get("epoch_ms") or data.get("timestamp_ms")
                            ep_int = int(ep) if ep is not None and str(ep).isdigit() else 0
                            ev_type = str(data.get("tipo_evento") or data.get("event_type") or data.get("type") or "")
                            sym = str(data.get("symbol") or data.get("ativo") or "")
                            canon_hash = compute_canonical_payload_hash(data)

                            entry = {
                                "source_file": str(seg),
                                "line_num": s_line_idx,
                                "epoch_ms": ep_int,
                                "tipo_evento": ev_type,
                                "symbol": sym,
                                "canonical_hash": canon_hash,
                                "is_archive": True,
                                "event_id": eid_str,
                            }
                            if eid_str:
                                jsonl_catalog_by_id[eid_str] = entry
                            corr_key = (ep_int, ev_type, sym)
                            jsonl_catalog_by_corr.setdefault(corr_key, []).append(entry)
                        except Exception as j_err:
                            arch_corrupt_lines += 1
                            arch_corrupt_details.append(f"Segmento {seg.name}:{s_line_idx} inválido: {j_err}")
            except Exception as seg_read_err:
                arch_corrupt_lines += 1
                arch_corrupt_details.append(f"Falha de leitura em {seg.name}: {seg_read_err}")

    archive_diagnostic = {
        "archive_dir": str(archive_dir),
        "exists": arch_exists,
        "segments_count": len(arch_segments),
        "segments": arch_segments[:10],
        "tmp_files_detected": arch_tmp_files,
        "has_residual_tmp": len(arch_tmp_files) > 0,
        "archived_events_count": arch_total_events,
        "corrupt_lines_count": arch_corrupt_lines,
        "corrupt_details": arch_corrupt_details[:10],
    }

    # =========================================================================
    # 4. DIAGNÓSTICO DO DIRETÓRIO FALLBACK
    # =========================================================================
    fb_exists = fb_dir.exists()
    fb_files: List[Dict[str, Any]] = []
    fb_total_records = 0
    fb_valid_events: List[Dict[str, Any]] = []
    fb_fixtures_count = 0
    fb_invalid_records = 0
    fb_missing_id_count = 0

    if fb_exists:
        for fpath in sorted(fb_dir.glob("*")):
            if fpath.is_symlink():
                conflicts.append(f"Origem operacional insegura: link simbólico detectado em fallback_events ({fpath.name}).")
                continue
            if fpath.is_file() and fpath.suffix.lower() in (".json", ".jsonl"):
                is_fix = is_git_tracked_fixture(fpath, r_root)
                if is_fix:
                    fb_fixtures_count += 1

                file_recs = 0
                file_invalids = 0
                file_missing_id = 0

                try:
                    if fpath.suffix.lower() == ".json":
                        with open(fpath, "r", encoding="utf-8", errors="replace") as jf:
                            content = json.load(jf)
                            records_raw = content if isinstance(content, list) else [content]
                    else:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as jlf:
                            records_raw = [json.loads(line) for line in jlf if line.strip()]

                    for idx, rec in enumerate(records_raw, start=1):
                        file_recs += 1
                        fb_total_records += 1
                        if not isinstance(rec, dict):
                            file_invalids += 1
                            if not is_fix:
                                fb_invalid_records += 1
                            continue

                        eid = rec.get("event_id") or rec.get("id")
                        eid_str = str(eid) if eid is not None and str(eid).strip() else None
                        if not eid_str:
                            file_missing_id += 1
                            if not is_fix:
                                fb_missing_id_count += 1
                            continue

                        canon_hash = compute_canonical_payload_hash(rec)
                        ep = rec.get("epoch_ms") or rec.get("timestamp_ms")
                        ep_int = int(ep) if ep is not None and str(ep).isdigit() else 0
                        ev_type = str(rec.get("tipo_evento") or rec.get("event_type") or rec.get("type") or "")
                        sym = str(rec.get("symbol") or rec.get("ativo") or "")

                        fb_valid_events.append({
                            "source_file": fpath.name,
                            "line_or_index": idx,
                            "event_id": eid_str,
                            "epoch_ms": ep_int,
                            "tipo_evento": ev_type,
                            "symbol": sym,
                            "canonical_hash": canon_hash,
                            "is_fixture": is_fix,
                        })

                except Exception as read_fb_err:
                    file_invalids += 1
                    if not is_fix:
                        fb_invalid_records += 1

                fb_files.append({
                    "filename": fpath.name,
                    "is_fixture": is_fix,
                    "records_count": file_recs,
                    "invalid_count": file_invalids,
                    "missing_id_count": file_missing_id,
                })

    fallback_diagnostic = {
        "fallback_dir": str(fb_dir),
        "exists": fb_exists,
        "files_count": len(fb_files),
        "files": fb_files,
        "fixtures_count": fb_fixtures_count,
        "total_records_read": fb_total_records,
        "valid_events_count": len(fb_valid_events),
        "invalid_records_count": fb_invalid_records,
        "missing_id_count": fb_missing_id_count,
    }

    # =========================================================================
    # 5. CONSISTÊNCIA CRUZADA ENTRE DESTINOS
    # =========================================================================
    all_known_ids = set(sqlite_catalog_by_id.keys()) | set(jsonl_catalog_by_id.keys())
    in_both_count = 0
    sqlite_only_count = 0
    jsonl_only_count = 0
    payload_hash_collisions: List[Dict[str, Any]] = []

    for eid in all_known_ids:
        in_sq = eid in sqlite_catalog_by_id
        in_jl = eid in jsonl_catalog_by_id

        if in_sq and in_jl:
            in_both_count += 1
            sq_hash = sqlite_catalog_by_id[eid]["canonical_hash"]
            jl_hash = jsonl_catalog_by_id[eid]["canonical_hash"]
            if sq_hash and jl_hash and sq_hash != jl_hash:
                payload_hash_collisions.append({
                    "event_id": eid,
                    "sqlite_hash": sq_hash,
                    "jsonl_hash": jl_hash,
                })
        elif in_sq and not in_jl:
            sqlite_only_count += 1
        elif in_jl and not in_sq:
            jsonl_only_count += 1

    # Análise de eventos de fallback potencialmente pendentes
    fallback_pending_count = 0
    fallback_already_present_count = 0
    fallback_pending_samples: List[Dict[str, Any]] = []

    for fbev in fb_valid_events:
        if fbev["is_fixture"]:
            continue
        eid = fbev["event_id"]
        in_sq = eid in sqlite_catalog_by_id
        in_jl = eid in jsonl_catalog_by_id

        if not in_sq and not in_jl:
            fallback_pending_count += 1
            if len(fallback_pending_samples) < 10:
                fallback_pending_samples.append(fbev)
        else:
            fallback_already_present_count += 1

    consistency_diagnostic = {
        "unique_events_in_sqlite": len(sqlite_catalog_by_id),
        "unique_events_in_jsonl": len(jsonl_catalog_by_id),
        "events_in_both_sqlite_and_jsonl": in_both_count,
        "events_sqlite_only": sqlite_only_count,
        "events_jsonl_only": jsonl_only_count,
        "payload_hash_collisions_count": len(payload_hash_collisions),
        "payload_hash_collisions": payload_hash_collisions[:5],
        "fallback_pending_unreconciled_count": fallback_pending_count,
        "fallback_already_present_count": fallback_already_present_count,
        "fallback_pending_samples": fallback_pending_samples,
    }

    # =========================================================================
    # 6. CONFLITOS E AUDITORIA DE RECOVERY (Gate G4-D B3-B2-B3-H1)
    # =========================================================================
    audit_file = d_dir / ".recovery_audit.jsonl"
    m_path = d_dir / MANIFEST_FILENAME
    manifest_data = read_runtime_manifest(m_path) if m_path.exists() else None
    env_uuid = manifest_data.get("environment_uuid") if manifest_data else None

    consistency = verify_manifest_recovery_consistency(
        manifest_data=manifest_data,
        ledger_path=audit_file,
        expected_env_uuid=env_uuid,
        lock_info=lock_info,
    )

    audit_val = validate_recovery_audit_ledger(audit_file, expected_environment_uuid=env_uuid)
    audit_exists = audit_val["exists"]
    audit_records_count = audit_val["records_count"]
    last_audit_record = audit_val["last_record"]
    has_uncommitted_prepared = audit_val["has_pending_transaction"]

    if not consistency["is_valid"]:
        conflicts.append(
            f"Inconsistência de histórico de recovery ({consistency['classification']}): {consistency['error']}"
        )

    rec_audit_file = d_dir / ".reconciliation_audit.jsonl"
    rec_consistency = verify_manifest_reconciliation_consistency(
        manifest_data=manifest_data,
        ledger_path=rec_audit_file,
        expected_env_uuid=env_uuid,
        lock_info=lock_info,
    )
    if not rec_consistency["is_valid"]:
        conflicts.append(
            f"Inconsistência de histórico de reconciliação ({rec_consistency['classification']}): {rec_consistency['error']}"
        )

    audit_diagnostic = {
        "audit_file": str(audit_file),
        "exists": audit_exists,
        "is_valid": audit_val["is_valid"],
        "error": audit_val["error"],
        "total_records": audit_records_count,
        "last_record": last_audit_record,
        "has_uncommitted_prepared": has_uncommitted_prepared,
        "orphaned_prepared": audit_val["orphaned_prepared"],
        "history_classification": consistency["classification"],
        "manifest_version": consistency["manifest_version"],
        "manifest_recovery_counter": consistency["manifest_counter"],
    }

    # =========================================================================
    # 7. CONFLITOS, EVIDÊNCIAS INSUFICIENTES E RECOMENDAÇÕES
    # =========================================================================
    # Avaliação de conflitos
    if is_kernel_locked:
        conflicts.append("Recurso físico de lock retido no SO por processo ativo concorrente.")
    if not anti_bypass_ok:
        conflicts.append(f"Violação Anti-Bypass: {anti_bypass_reason}")
    if len(payload_hash_collisions) > 0:
        conflicts.append(f"Detectadas {len(payload_hash_collisions)} colisões de hash entre SQLite e JSONL.")
    if jl_corrupt_lines > 0:
        conflicts.append(f"Arquivo JSONL ativo contém {jl_corrupt_lines} linhas corrompidas ou truncadas.")

    # Verificação de symlinks operacionais (Gate G4-D B3-B2-B3 Hardening)
    for cp in [l_path, m_path, jsonl_path, sqlite_path, wal_path, shm_path]:
        if cp is not None and cp.exists() and cp.is_symlink():
            conflicts.append(f"Origem operacional insegura: link simbólico detectado em '{cp.name}'.")
    if fb_exists:
        for fpath in fb_dir.glob("*"):
            if fpath.is_symlink():
                conflicts.append(f"Origem operacional insegura: link simbólico detectado em fallback_events ({fpath.name}).")
    if arch_exists:
        for fpath in archive_dir.glob("*.jsonl"):
            if fpath.is_symlink():
                conflicts.append(f"Origem operacional insegura: link simbólico detectado em archive ({fpath.name}).")
    if archive_diagnostic["has_residual_tmp"]:
        conflicts.append(f"Diretório de archive contém arquivos temporários residuais .tmp.")
    if archive_diagnostic["corrupt_lines_count"] > 0:
        conflicts.append(f"Diretório de archive contém {archive_diagnostic['corrupt_lines_count']} linhas corrompidas.")
    if not sq_integrity_ok and sq_exists:
        conflicts.append(f"Falha estrutural PRAGMA integrity_check no SQLite: {sq_integrity_msg}")
    if has_uncommitted_prepared:
        conflicts.append(f"Transação de recovery incompleta/interrompida detectada (PREPARED sem COMMITTED).")
    if fallback_pending_count > 0:
        conflicts.append(f"Existem {fallback_pending_count} eventos pendentes não reconciliados no fallback.")
    if fallback_diagnostic["missing_id_count"] > 0:
        conflicts.append(f"Fallback operacional contém {fallback_diagnostic['missing_id_count']} eventos sem identificador.")
    if fallback_diagnostic["invalid_records_count"] > 0:
        conflicts.append(f"Fallback operacional contém {fallback_diagnostic['invalid_records_count']} registros inválidos.")

    # Avaliação de evidências
    if not lock_exists and sq_exists:
        insufficient_evidences.append("Arquivo de lock operacional inexistente apesar da existência de dados.")
    if lock_exists and persisted_status == RuntimeState.UNKNOWN and not is_kernel_locked:
        insufficient_evidences.append("Metadados indicam crash anterior sem rastreabilidade de encerramento gracioso.")

    # Recomendações
    recommendations.append("AVISO: Nenhuma autorização automática de transição para CLEAN é emitida.")

    if recovery_classification == "CLEAN_OPERATIONAL":
        recommendations.append("Ambiente já em estado CLEAN operacional. Nenhum recovery necessário.")
    elif recovery_classification == "DIRTY_INCOMPLETE_SHUTDOWN":
        recommendations.append(
            "Lock retido com status DIRTY devido a shutdown incompleto com persistência ativa. "
            "Exige auditoria de logs e execução formal de recovery com autorização e justificativa."
        )
    elif recovery_classification == "CRASH_RESIDUAL_RUNNING":
        recommendations.append(
            "Crash abrupto detectado (status RUNNING sem kernel lock). "
            "Exige execução de recovery formal com snapshot de backup antes de autorizar boot."
        )
    elif recovery_classification in ("LOCK_MISSING_ANTI_BYPASS_VIOLATION", "MANIFEST_MISSING_ANTI_BYPASS_VIOLATION"):
        recommendations.append(
            "Violação crítica de Anti-Bypass. Não tente recriar arquivos manualmente. "
            "Execute investigação forense para certificar se arquivos foram apagados acidentalmente."
        )

    if fallback_pending_count > 0:
        recommendations.append(
            f"Existem {fallback_pending_count} eventos potencialmente não reconciliados no fallback. "
            "Execute 'reconcile_fallback_events.py' em dry-run antes de religar o robô."
        )

    return {
        "mode": "DRY_RUN_READ_ONLY",
        "timestamp_utc": lock_diagnostic.get("metadata", {}).get("timestamp_utc") if lock_diagnostic.get("metadata") else None,
        "recovery_classification": recovery_classification,
        "lock": lock_diagnostic,
        "sqlite": sqlite_diagnostic,
        "jsonl": jsonl_diagnostic,
        "archive": archive_diagnostic,
        "fallback": fallback_diagnostic,
        "consistency": consistency_diagnostic,
        "audit": audit_diagnostic,
        "conflicts": conflicts,
        "insufficient_evidences": insufficient_evidences,
        "recommendations": recommendations,
        "ready_for_operator_review": len(conflicts) == 0 or recovery_classification in ("DIRTY_INCOMPLETE_SHUTDOWN", "CRASH_RESIDUAL_RUNNING"),
    }


def create_recovery_snapshot(
    data_dir: Path,
    lock_path: Path,
    backup_base_dir: Path,
    env_uuid: str,
    operator_reason: str,
    fallback_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    lock_file_obj: Optional[Any] = None,
) -> Tuple[Path, str]:
    """
    Cria snapshot forense completo e atômico pré-recovery (Gate G4-D B3-B2-B3).
    Aplica verificação preventiva de espaço livre (mínimo 3x tamanho estimado).
    Retorna (snapshot_dir, snapshot_manifest_sha256).
    """
    backup_base_dir = Path(backup_base_dir).expanduser().resolve()
    backup_base_dir.mkdir(parents=True, exist_ok=True)
    d_dir = Path(data_dir).expanduser().resolve()
    fb_dir = Path(fallback_dir).expanduser().resolve() if fallback_dir else (PROJECT_ROOT / "fallback_events")
    r_root = Path(repo_root).expanduser().resolve() if repo_root else PROJECT_ROOT

    # 1. Coleta de arquivos e cálculo de tamanho estimado
    sqlite_path = d_dir / "trading_bot.db"
    wal_path = d_dir / "trading_bot.db-wal"
    shm_path = d_dir / "trading_bot.db-shm"
    jsonl_path = d_dir / "eventos_fluxo.jsonl"
    manifest_path = d_dir / MANIFEST_FILENAME
    archive_dir = d_dir / "archive"

    # Validação de segurança: rejeitar symlinks operacionais (Gate G4-D B3-B2-B3 Hardening)
    for check_p in [lock_path, manifest_path, jsonl_path, sqlite_path, wal_path, shm_path]:
        if check_p.is_symlink():
            raise RuntimeError(
                f"Origem operacional insegura: link simbólico detectado em '{check_p.name}'. "
                "Operação abortada por segurança contra symlinks/path traversal."
            )

    files_to_copy: List[Tuple[Path, str]] = []
    if lock_path.exists():
        files_to_copy.append((lock_path, ".operational_runtime.lock.orig"))
    if manifest_path.exists():
        files_to_copy.append((manifest_path, ".runtime_initialized.manifest.orig"))
    if jsonl_path.exists():
        files_to_copy.append((jsonl_path, "eventos_fluxo.jsonl"))

    # Arquivos WAL e SHM brutos
    if wal_path.exists():
        files_to_copy.append((wal_path, "trading_bot.db-wal.raw"))
    if shm_path.exists():
        files_to_copy.append((shm_path, "trading_bot.db-shm.raw"))

    # Segmentos de archive
    if archive_dir.exists():
        if archive_dir.is_symlink():
            raise RuntimeError("Origem operacional insegura: diretório archive é um link simbólico.")
        for seg in archive_dir.glob("*.jsonl"):
            if seg.is_symlink():
                raise RuntimeError(
                    f"Origem operacional insegura: link simbólico detectado em archive ({seg.name}). "
                    "Operação abortada por segurança."
                )
            files_to_copy.append((seg, f"archive/{seg.name}"))

    # Fallback operacionais (não fixtures)
    if fb_dir.exists():
        if fb_dir.is_symlink():
            raise RuntimeError("Origem operacional insegura: diretório fallback_events é um link simbólico.")
        for fbp in fb_dir.glob("*"):
            if fbp.is_symlink():
                raise RuntimeError(
                    f"Origem operacional insegura: link simbólico detectado em fallback_events ({fbp.name}). "
                    "Operação abortada por segurança contra symlinks/path traversal."
                )
            if fbp.is_file() and not is_git_tracked_fixture(fbp, r_root):
                files_to_copy.append((fbp, f"fallback_events/{fbp.name}"))

    estimated_bytes = sum(f.stat().st_size for f, _ in files_to_copy if f.exists())
    if sqlite_path.exists():
        estimated_bytes += sqlite_path.stat().st_size

    # Checagem preventiva de espaço em disco (mínimo 3x margem)
    try:
        total_d, used_d, free_d = shutil.disk_usage(backup_base_dir)
        required_free = max(10 * 1024 * 1024, estimated_bytes * 3)
        if free_d < required_free:
            raise RuntimeError(
                f"Espaço insuficiente em disco para snapshot forense: livre={free_d} bytes, "
                f"necessário com margem={required_free} bytes (ENOSPC_PREVENTED)."
            )
    except Exception as disk_err:
        if "ENOSPC_PREVENTED" in str(disk_err):
            raise
        logger.warning("Não foi possível verificar quota de disco: %s", disk_err)

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    uuid_short = str(uuid.uuid4())[:8]
    snapshot_name = f"recovery_snapshot_{ts_str}_{uuid_short}"
    snapshot_dir = backup_base_dir / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    manifest_data: Dict[str, Any] = {
        "snapshot_version": "1.0",
        "snapshot_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env_uuid,
        "operator_reason": operator_reason,
        "host": platform.node(),
        "pid": os.getpid(),
        "files": {},
    }

    try:
        # Cópia consistente do SQLite via API oficial sqlite3.connect.backup()
        if sqlite_path.exists() and sqlite_path.stat().st_size > 0:
            dst_db = snapshot_dir / "trading_bot.db"
            src_conn = sqlite3.connect(f"file:{sqlite_path.resolve().as_posix()}?mode=ro", uri=True)
            dst_conn = sqlite3.connect(str(dst_db))
            src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()

            # fsync explícito do SQLite copiado
            with open(dst_db, "r+b") as df_db:
                df_db.flush()
                os.fsync(df_db.fileno())

            # Hash do SQLite no backup
            with open(dst_db, "rb") as df:
                db_bytes = df.read()
                manifest_data["files"]["trading_bot.db"] = {
                    "size_bytes": len(db_bytes),
                    "sha256": hashlib.sha256(db_bytes).hexdigest(),
                    "backup_method": "sqlite3_online_backup_api",
                }

        # Cópia dos demais arquivos
        for src_f, rel_dst in files_to_copy:
            dst_p = snapshot_dir / rel_dst
            dst_p.parent.mkdir(parents=True, exist_ok=True)

            if lock_file_obj is not None and src_f.resolve() == lock_path.resolve():
                curr_pos = lock_file_obj.tell()
                lock_file_obj.seek(0)
                raw = lock_file_obj.read()
                lock_file_obj.seek(curr_pos)
                with open(dst_p, "wb") as df:
                    df.write(raw)
                    df.flush()
                    os.fsync(df.fileno())
                src_sha = hashlib.sha256(raw).hexdigest()
            else:
                shutil.copy2(src_f, dst_p)
                with open(dst_p, "r+b") as df:
                    df.flush()
                    os.fsync(df.fileno())
                raw = dst_p.read_bytes()
                src_sha = hashlib.sha256(src_f.read_bytes()).hexdigest()

            # Validação do hash
            dst_sha = hashlib.sha256(raw).hexdigest()
            if dst_sha != src_sha:
                raise RuntimeError(
                    f"Falha de integridade ao copiar {src_f.name}: hash divergente no snapshot."
                )
            manifest_data["files"][rel_dst] = {
                "size_bytes": len(raw),
                "sha256": dst_sha,
                "backup_method": "raw_copy",
            }

        # Gravação do manifesto do snapshot com fsync
        manifest_file = snapshot_dir / "snapshot_manifest.json"
        manifest_raw = json.dumps(manifest_data, indent=2).encode("utf-8")
        with open(manifest_file, "wb") as mf:
            mf.write(manifest_raw)
            mf.flush()
            os.fsync(mf.fileno())

        # Sincronização durável de diretório quando suportada pela plataforma (ex: POSIX O_DIRECTORY)
        if hasattr(os, "O_DIRECTORY"):
            try:
                dir_fd = os.open(str(snapshot_dir), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass

        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        return snapshot_dir, manifest_sha256

    except Exception as exc:
        # Tratamento de falha parcial: isolar snapshot incompleto
        incomplete_dir = backup_base_dir / f"{snapshot_name}.INCOMPLETE_FAILED_{ts_str}"
        try:
            if snapshot_dir.exists():
                snapshot_dir.rename(incomplete_dir)
        except Exception:
            pass
        raise RuntimeError(f"Falha crítica na criação do snapshot forense: {exc}") from exc


def record_recovery_audit(audit_file: Path, entry: Dict[str, Any]) -> None:
    """Registra evento no livro-razão append-only de auditoria com fsync."""
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    raw_line = json.dumps(entry) + "\n"
    with open(audit_file, "a", encoding="utf-8") as af:
        af.write(raw_line)
        af.flush()
        os.fsync(af.fileno())


def execute_authorized_recovery(
    data_dir: Path,
    lock_path: Path,
    reason: str,
    backup_dir: Optional[Path] = None,
    fallback_dir: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executa a recuperação autorizada para CLEAN com todas as garantias de segurança (Gate G4-D B3-B2-B3).
    """
    d_dir = Path(data_dir).expanduser().resolve()
    l_path = Path(lock_path).expanduser().resolve() if lock_path else (d_dir / ".operational_runtime.lock")
    b_dir = Path(backup_dir).expanduser().resolve() if backup_dir else (d_dir.parent / "backups" / "recovery_snapshots")
    fb_dir = Path(fallback_dir).expanduser().resolve() if fallback_dir else (PROJECT_ROOT / "fallback_events")
    r_root = Path(repo_root).expanduser().resolve() if repo_root else PROJECT_ROOT

    # Validação do motivo
    clean_reason = (reason or "").strip()
    if len(clean_reason) < 15:
        raise ValueError("A justificativa de recuperação deve conter pelo menos 15 caracteres significativos.")

    # 1. Instancia lock operacional para manutenção
    op_lock = OperationalRuntimeLock(lock_path=l_path, holder_name="recovery_tool")

    # 2. Sessão de manutenção: adquire exclusividade no kernel sem transicionar para RUNNING
    with op_lock.maintenance_session(
        allowed_states=[
            RuntimeState.DIRTY,
            RuntimeState.UNKNOWN,
            RuntimeState.RUNNING,
            RuntimeState.CLEAN,
            "RUNNING_RESIDUAL",
        ]
    ) as maint_info:
        raw_status = maint_info.get("raw_status")
        detected_status = maint_info.get("detected_status")

        # Idempotência: se já estiver em CLEAN, não efetua mutações
        if raw_status == RuntimeState.CLEAN.value:
            return {
                "status": "IDEMPOTENT_ALREADY_CLEAN",
                "message": "Ambiente já se encontra no estado CLEAN operacional. Nenhuma recuperação necessária.",
                "lock_path": str(l_path),
                "persisted_status": "CLEAN",
                "mutations_performed": False,
            }

        # 3. Diagnóstico completo de integridade e consistência sob exclusividade
        report = diagnose_runtime_recovery(
            data_dir=d_dir,
            lock_path=l_path,
            fallback_dir=fb_dir,
            repo_root=r_root,
            is_maintenance_session=True,
        )

        # 4. Avaliação de impedimentos críticos
        if not report["lock"]["anti_bypass_ok"]:
            raise RuntimeError(f"Recuperação bloqueada: Violação Anti-Bypass ({report['lock']['anti_bypass_reason']}).")

        if report["sqlite"]["exists"] and not report["sqlite"]["integrity_check_ok"]:
            raise RuntimeError(f"Recuperação bloqueada: Falha de integridade SQLite ({report['sqlite']['integrity_check_details']}).")

        if report["jsonl"]["corrupt_lines_count"] > 0:
            raise RuntimeError(f"Recuperação bloqueada: JSONL ativo contém {report['jsonl']['corrupt_lines_count']} linhas corrompidas.")

        if report["archive"]["has_residual_tmp"]:
            raise RuntimeError("Recuperação bloqueada: Diretório archive contém arquivos temporários residuais .tmp.")

        if report["archive"]["corrupt_lines_count"] > 0:
            raise RuntimeError(f"Recuperação bloqueada: Diretório archive contém {report['archive']['corrupt_lines_count']} linhas corrompidas.")

        if report["consistency"]["payload_hash_collisions_count"] > 0:
            raise RuntimeError(f"Recuperação bloqueada: Detectadas {report['consistency']['payload_hash_collisions_count']} colisões de hash de payload.")

        if report["consistency"]["fallback_pending_unreconciled_count"] > 0:
            raise RuntimeError(
                f"Recuperação bloqueada: Existem {report['consistency']['fallback_pending_unreconciled_count']} "
                "eventos potencialmente pendentes no fallback. Execute a reconciliação antes do recovery."
            )

        if report["fallback"]["missing_id_count"] > 0 or report["fallback"]["invalid_records_count"] > 0:
            raise RuntimeError(
                f"Recuperação bloqueada: Fallback operacional contém {report['fallback']['missing_id_count']} eventos sem identificador "
                f"e {report['fallback']['invalid_records_count']} registros inválidos."
            )

        if len(report["conflicts"]) > 0:
            raise RuntimeError(f"Recuperação bloqueada por conflitos detectados: {report['conflicts']}")

        env_uuid = report["lock"].get("environment_uuid") or "UNKNOWN_UUID"
        audit_file = d_dir / ".recovery_audit.jsonl"
        audit_id = str(uuid.uuid4())

        # 5. Criação do Snapshot Forense com verificação de hashes
        snapshot_dir, manifest_sha256 = create_recovery_snapshot(
            data_dir=d_dir,
            lock_path=l_path,
            backup_base_dir=b_dir,
            env_uuid=env_uuid,
            operator_reason=clean_reason,
            fallback_dir=fb_dir,
            repo_root=r_root,
            lock_file_obj=op_lock._file_obj,
        )

        # 6. Livro-Razão de Auditoria FASE 1: PREPARED
        audit_entry_prepared = {
            "audit_id": audit_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment_uuid": env_uuid,
            "operator_reason": clean_reason,
            "previous_runtime_state": str(raw_status),
            "previous_lock_metadata": maint_info.get("metadata"),
            "snapshot_directory": str(snapshot_dir),
            "snapshot_manifest_sha256": manifest_sha256,
            "sqlite_integrity": report["sqlite"]["integrity_check_details"],
            "jsonl_integrity": f"clean_{report['jsonl']['total_lines_read']}_lines",
            "hash_collisions_count": report["consistency"]["payload_hash_collisions_count"],
            "recovery_phase": "PREPARED",
            "final_action": "PREPARED_FOR_TRANSITION",
            "final_state": "PENDING_CLEAN",
        }
        record_recovery_audit(audit_file, audit_entry_prepared)

        # 7. Transição Física do Lock para CLEAN utilizando o descritor já travado
        op_lock.transition_maintenance_to_clean(
            reason=f"Recovery autorizada: audit_id={audit_id} - {clean_reason}"
        )

        # 8. Atualização Atômica do Manifesto Operacional com fsync (Gate G4-D B3-B2-B3-H1)
        update_runtime_manifest_recovery_state(
            manifest_path=d_dir / MANIFEST_FILENAME,
            audit_id=audit_id,
            audit_timestamp_utc=audit_entry_prepared["timestamp_utc"],
            expected_env_uuid=env_uuid,
            ledger_path=audit_file,
            lock_authorization=op_lock,
        )

        # 9. Livro-Razão de Auditoria FASE 2: COMMITTED
        audit_entry_committed = {
            "audit_id": audit_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment_uuid": env_uuid,
            "operator_reason": clean_reason,
            "previous_runtime_state": str(raw_status),
            "snapshot_directory": str(snapshot_dir),
            "snapshot_manifest_sha256": manifest_sha256,
            "recovery_phase": "COMMITTED",
            "final_action": "TRANSITION_TO_CLEAN_SUCCESS",
            "final_state": "CLEAN",
        }
        record_recovery_audit(audit_file, audit_entry_committed)

        return {
            "status": "RECOVERY_SUCCESS",
            "audit_id": audit_id,
            "previous_state": str(raw_status),
            "new_state": "CLEAN",
            "operator_reason": clean_reason,
            "snapshot_directory": str(snapshot_dir),
            "snapshot_manifest_sha256": manifest_sha256,
            "mutations_performed": True,
            "lock_path": str(l_path),
        }


def main() -> int:
    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    lock_path = (args.lock_path or (data_dir / ".operational_runtime.lock")).expanduser().resolve()
    fallback_dir = args.fallback_dir.expanduser().resolve()
    backup_dir = args.backup_dir.expanduser().resolve()

    if args.confirm_recovery:
        try:
            rec_result = execute_authorized_recovery(
                data_dir=data_dir,
                lock_path=lock_path,
                reason=args.reason,
                backup_dir=backup_dir,
                fallback_dir=fallback_dir,
            )
            if args.json_output:
                print(json.dumps(rec_result, indent=2))
            else:
                print("================================================================================")
                print("RELATÓRIO OFICIAL DE EXECUÇÃO DE RECOVERY AUTORIZADO (GATE G4-D B3-B2-B3)")
                print("================================================================================")
                print(f"Status da Operação:        {rec_result['status']}")
                print(f"Audit ID:                  {rec_result.get('audit_id', 'N/A')}")
                print(f"Estado Anterior:           {rec_result.get('previous_state', 'N/A')}")
                print(f"Novo Estado Persistido:    {rec_result.get('new_state', rec_result.get('persisted_status', 'N/A'))}")
                print(f"Justificativa:             {rec_result.get('operator_reason', 'N/A')}")
                print(f"Snapshot Forense:          {rec_result.get('snapshot_directory', 'N/A')}")
                print(f"SHA256 do Manifesto:       {rec_result.get('snapshot_manifest_sha256', 'N/A')}")
                print(f"Mutações Realizadas:       {rec_result.get('mutations_performed', False)}")
                print("================================================================================")
            return 0
        except Exception as e_rec:
            if args.json_output:
                print(json.dumps({"status": "RECOVERY_FAILED", "error": str(e_rec)}, indent=2))
            else:
                print(f"ERRO CRÍTICO NA EXECUÇÃO DO RECOVERY: {e_rec}", file=sys.stderr)
            return 1

    report = diagnose_runtime_recovery(
        data_dir=data_dir,
        lock_path=lock_path,
        fallback_dir=fallback_dir,
    )

    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        print("================================================================================")
        print("RELATÓRIO DIAGNÓSTICO OFICIAL DE RECOVERY DE RUNTIME (MODO READ-ONLY / DRY-RUN)")
        print("================================================================================")
        print(f"Diretório de Dados:        {report['lock']['lock_path']}")
        print(f"Classificação de Recovery: {report['recovery_classification']}")
        print(f"Status Persistido no Lock: {report['lock']['persisted_status']} (Raw: {report['lock']['raw_status']})")
        print(f"Trava Física no Kernel:    {'ATIVA (PROCESSO VIVO)' if report['lock']['is_kernel_locked'] else 'LIVRE'}")
        print(f"Manifesto UUID:            {report['lock']['environment_uuid'] or 'NÃO ENCONTRADO'}")
        print(f"Anti-Bypass Status:        {'APROVADO' if report['lock']['anti_bypass_ok'] else 'FALHA'}")
        print("--------------------------------------------------------------------------------")
        print("INTEGRIDADE DOS DESTINOS:")
        print(f"  - SQLite ({report['sqlite']['sqlite_path']}):")
        print(f"      Integridade:   {'OK' if report['sqlite']['integrity_check_ok'] else 'FALHA'} ({report['sqlite']['integrity_check_details']})")
        print(f"      Foreign Keys:  {'OK' if report['sqlite']['foreign_key_check_ok'] else 'FALHA'}")
        print(f"      Schema events: {'OK' if report['sqlite']['schema_valid'] else 'INVÁLIDO'}")
        print(f"      Total Eventos: {report['sqlite']['events_count']}")
        print(f"      WAL Ativo:     {report['sqlite']['wal_exists']} ({report['sqlite']['wal_size_bytes']} bytes)")
        print(f"  - JSONL Ativo ({report['jsonl']['jsonl_path']}):")
        print(f"      Linhas Lidas:  {report['jsonl']['total_lines_read']}")
        print(f"      Linhas Ruuns:  {report['jsonl']['corrupt_lines_count']}")
        print(f"      Última Linha:  {'OK' if report['jsonl']['last_line_clean'] else 'TRUNCADA / SEM NEWLINE'}")
        print(f"  - Archive G4-C ({report['archive']['archive_dir']}):")
        print(f"      Segmentos:     {report['archive']['segments_count']} arquivos")
        print(f"      Arquivos .tmp: {len(report['archive']['tmp_files_detected'])}")
        print(f"  - Fallback ({report['fallback']['fallback_dir']}):")
        print(f"      Arquivos:      {report['fallback']['files_count']} (Fixtures Git: {report['fallback']['fixtures_count']})")
        print(f"      Total Records: {report['fallback']['total_records_read']}")
        print("--------------------------------------------------------------------------------")
        print("CONSISTÊNCIA CRUZADA:")
        print(f"  - Presentes em Ambos (SQLite e JSONL): {report['consistency']['events_in_both_sqlite_and_jsonl']}")
        print(f"  - Presentes Somente no SQLite:         {report['consistency']['events_sqlite_only']}")
        print(f"  - Presentes Somente no JSONL/Archive:  {report['consistency']['events_jsonl_only']}")
        print(f"  - Colisões de Hash de Payload:         {report['consistency']['payload_hash_collisions_count']}")
        print(f"  - Eventos Potencialmente Pendentes:    {report['consistency']['fallback_pending_unreconciled_count']}")
        print("--------------------------------------------------------------------------------")
        if report["conflicts"]:
            print("CONFLITOS DETECTADOS:")
            for c in report["conflicts"]:
                print(f"  [!] {c}")
        else:
            print("CONFLITOS DETECTADOS: Nenhum.")
        print("--------------------------------------------------------------------------------")
        print("RECOMENDAÇÕES PARA O OPERADOR:")
        for r in report["recommendations"]:
            print(f"  * {r}")
        print("================================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
