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
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
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
    RuntimeState,
    MANIFEST_FILENAME,
    read_runtime_manifest,
    verify_environment_anti_bypass,
)
from events.fallback_reconciler import (
    compute_canonical_payload_hash,
    KNOWN_TEST_FIXTURE_NAMES,
)

logger = logging.getLogger("scripts.maintenance.recover_runtime_state")

# Flags estritamente proibidas nesta fase de diagnóstico read-only (fail-closed)
FORBIDDEN_FLAGS = [
    "--confirm-recovery",
    "--clear-dirty-state",
    "--force",
    "--repair",
    "--delete",
    "-f",
]


def parse_args() -> argparse.Namespace:
    """Valida argumentos de linha de comando rejeitando qualquer opção de mutação."""
    for arg in sys.argv[1:]:
        if (
            arg in FORBIDDEN_FLAGS
            or arg.startswith("--confirm")
            or arg.startswith("--clear")
            or arg.startswith("--repair")
            or arg.startswith("--force")
            or arg.startswith("--delete")
        ):
            print(
                f"ERRO DE SEGURANCA: O parametro '{arg}' e expressamente PROIBIDO nesta etapa (G4-D B3-B2-B1).",
                file=sys.stderr,
            )
            print("A ferramenta de recovery opera exclusivamente em modo READ-ONLY (--dry-run).", file=sys.stderr)
            print("Nenhuma transicao de estado, reparo ou modificacao em disco e autorizada.", file=sys.stderr)
            sys.exit(2)

    parser = argparse.ArgumentParser(
        description="Diagnóstico offline read-only do estado de recovery do runtime operacional (Gate G4-D B3-B2-B1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Executa auditoria estritamente read-only sem modificar arquivos (Padrão e único modo).",
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

    return parser.parse_args()


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
    if l_path.exists():
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
                            fb_invalid_records += 1
                            continue

                        eid = rec.get("event_id") or rec.get("id")
                        eid_str = str(eid) if eid is not None and str(eid).strip() else None
                        if not eid_str:
                            file_missing_id += 1
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
    # 6. CONFLITOS, EVIDÊNCIAS INSUFICIENTES E RECOMENDAÇÕES
    # =========================================================================
    conflicts: List[str] = []
    insufficient_evidences: List[str] = []
    recommendations: List[str] = []

    # Avaliação de conflitos
    if is_kernel_locked:
        conflicts.append("Recurso físico de lock retido no SO por processo ativo concorrente.")
    if not anti_bypass_ok:
        conflicts.append(f"Violação Anti-Bypass: {anti_bypass_reason}")
    if len(payload_hash_collisions) > 0:
        conflicts.append(f"Detectadas {len(payload_hash_collisions)} colisões de hash entre SQLite e JSONL.")
    if jl_corrupt_lines > 0:
        conflicts.append(f"Arquivo JSONL ativo contém {jl_corrupt_lines} linhas corrompidas ou truncadas.")
    if archive_diagnostic["has_residual_tmp"]:
        conflicts.append(f"Diretório de archive contém arquivos temporários residuais .tmp.")
    if not sq_integrity_ok and sq_exists:
        conflicts.append(f"Falha estrutural PRAGMA integrity_check no SQLite: {sq_integrity_msg}")

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
        "conflicts": conflicts,
        "insufficient_evidences": insufficient_evidences,
        "recommendations": recommendations,
        "ready_for_operator_review": len(conflicts) == 0 or recovery_classification in ("DIRTY_INCOMPLETE_SHUTDOWN", "CRASH_RESIDUAL_RUNNING"),
    }


def main() -> int:
    args = parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    lock_path = (args.lock_path or (data_dir / ".operational_runtime.lock")).expanduser().resolve()
    fallback_dir = args.fallback_dir.expanduser().resolve()

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
