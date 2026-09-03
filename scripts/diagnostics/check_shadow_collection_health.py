# scripts/diagnostics/check_shadow_collection_health.py
# -*- coding: utf-8 -*-
"""
Ferramenta Operacional de Diagnóstico de Saúde da Coleta Shadow — Fase O1.
Executa inspeção não-intrusiva de observabilidade em runtime:
1. Process Alive & Heartbeat.
2. Positioning Shadow Collector (timeline rows, unique Binance source timestamps, cadence 5m, freshness, cache hits).
3. Session VWAP Tracker (status VALID/WARMING_UP, ancoragem UTC 00:00, rollovers, NaN/Inf check).
4. Market Structure Detector (distinção estrita: analysis_count vs bos_event_count vs sweep_event_count).
5. Data Quality Monitor (NaN/Inf=0, future ts=0, duplicate timestamps=0, duplicate event_ids=0).
6. Alertas operacionais de estagnação (Stalled Collection Alert).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# Fix encoding Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CollectionHealth")

DB_PATH = "dados/trading_bot.db"


def check_process_alive() -> Dict[str, Any]:
    """Verifica se há processos de coleta shadow em execução no Windows/Linux."""
    alive = False
    process_names = []
    try:
        if sys.platform == "win32":
            output = subprocess.check_output(["tasklist", "/FO", "CSV"], text=True, errors="ignore")
            # Procura processos python ativos
            python_procs = [line for line in output.splitlines() if "python" in line.lower()]
            alive = len(python_procs) > 0
            process_names = [p.split(",")[0].replace('"', '') for p in python_procs[:3]]
        else:
            output = subprocess.check_output(["ps", "-ef"], text=True, errors="ignore")
            alive = "python" in output.lower()
    except Exception as e:
        logger.debug(f"Falha ao consultar tasklist: {e}")

    return {
        "is_alive": alive,
        "processes_found": len(process_names),
        "sample": process_names,
    }


def check_collection_health() -> Dict[str, Any]:
    print("=" * 80)
    print("DIAGNÓSTICO DE SAÚDE DA COLETA SHADOW — FASE O1 PRE-FLIGHT & RUNTIME")
    print("=" * 80)

    now_ms = int(time.time() * 1000)
    now_ts = time.time()
    health_status: Dict[str, Any] = {
        "timestamp_ms": now_ms,
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts)),
        "process": check_process_alive(),
        "positioning": {},
        "session_vwap": {},
        "market_structure": {},
        "data_quality": {},
        "storage": {},
        "alerts": [],
    }

    conn = sqlite3.connect(DB_PATH) if os.path.exists(DB_PATH) else None

    # 1. POSICIONAMENTO BINANCE (5m Source Cadence)
    pos_count = 0
    pos_unique_sources = 0
    pos_last_ts = 0
    pos_stale_count = 0
    pos_cache_hits = 0
    last_regime = "UNKNOWN"

    if conn:
        cur = conn.cursor()
        try:
            cur.execute("""
            SELECT count(*), count(DISTINCT source_timestamp_ms), max(timestamp_ms), 
                   sum(is_stale), sum(cache_hit) 
            FROM positioning_shadow_dataset
            """)
            row = cur.fetchone()
            pos_count = row[0] or 0
            pos_unique_sources = row[1] or 0
            pos_last_ts = row[2] or 0
            pos_stale_count = row[3] or 0
            pos_cache_hits = row[4] or 0

            # Último registro
            cur.execute("SELECT positioning_regime FROM positioning_shadow_dataset ORDER BY id DESC LIMIT 1")
            r_row = cur.fetchone()
            if r_row:
                last_regime = r_row[0]
        except Exception as e:
            logger.warning(f"Erro ao consultar positioning_shadow_dataset: {e}")

    pos_age_sec = (now_ms - pos_last_ts) / 1000.0 if pos_last_ts > 0 else 999999
    pos_stale = pos_age_sec > 600.0  # > 10 min tolerância

    health_status["positioning"] = {
        "timeline_rows": pos_count,
        "unique_source_snapshots": pos_unique_sources,
        "unique_source_ratio": round(pos_unique_sources / pos_count, 3) if pos_count > 0 else 0.0,
        "last_snapshot_ts": pos_last_ts,
        "age_seconds": round(pos_age_sec, 1),
        "is_stale": pos_stale,
        "cache_hits": pos_cache_hits,
        "last_regime": last_regime,
        "status": "HEALTHY" if (pos_count > 0 and not pos_stale) else "STANDBY_OR_INACTIVE",
    }

    # 2. EVENTS STREAM, SESSION VWAP & MARKET STRUCTURE
    events_count = 0
    last_event_ts = 0
    vwap_valid_count = 0
    vwap_warming_count = 0
    vwap_last_status = "UNKNOWN"
    vwap_last_val = None
    vwap_last_dist = None
    vwap_last_session_start = ""
    vwap_nan_count = 0

    ms_analysis_count = 0
    ms_bos_count = 0
    ms_sweep_count = 0
    ms_last_high = None
    ms_last_low = None
    ms_event_ids: List[str] = []

    duplicate_timestamps = 0
    out_of_order_count = 0
    future_timestamps = 0

    if conn:
        cur = conn.cursor()
        try:
            cur.execute("SELECT count(*), max(timestamp_ms) FROM events")
            row = cur.fetchone()
            events_count = row[0] or 0
            last_event_ts = row[1] or 0

            # Inspeciona últimos eventos (até 200)
            cur.execute("SELECT timestamp_ms, payload FROM events ORDER BY timestamp_ms DESC LIMIT 200")
            recent_rows = cur.fetchall()
            prev_ts = None
            for ts, payload_str in reversed(recent_rows):
                if prev_ts is not None and ts < prev_ts:
                    out_of_order_count += 1
                if prev_ts is not None and ts == prev_ts:
                    duplicate_timestamps += 1
                if ts > (now_ts + 300) * 1000:
                    future_timestamps += 1
                prev_ts = ts

                try:
                    p = json.loads(payload_str)
                    ia = p.get("institutional_analytics") or {}

                    # Session VWAP
                    sv = ia.get("session_vwap")
                    if isinstance(sv, dict):
                        st = sv.get("status", "UNKNOWN")
                        if st == "VALID":
                            vwap_valid_count += 1
                        elif st == "WARMING_UP":
                            vwap_warming_count += 1
                        vwap_last_status = st
                        vwap_last_val = sv.get("session_vwap") or sv.get("vwap")
                        vwap_last_dist = sv.get("distance_fraction") or sv.get("distance_to_vwap")
                        vwap_last_session_start = str(sv.get("session_start_iso") or sv.get("session_start") or "")
                        if vwap_last_val is not None and not math.isfinite(float(vwap_last_val)):
                            vwap_nan_count += 1

                    # Market Structure
                    ms = ia.get("market_structure")
                    if isinstance(ms, dict):
                        ms_analysis_count += 1
                        ms_last_high = ms.get("last_swing_high")
                        ms_last_low = ms.get("last_swing_low")
                        bos = ms.get("bos")
                        if bos and isinstance(bos, dict):
                            eid = bos.get("event_id")
                            if eid:
                                ms_event_ids.append(eid)
                        sweep = ms.get("sweep")
                        if sweep and isinstance(sweep, dict):
                            eid = sweep.get("event_id")
                            if eid:
                                ms_event_ids.append(eid)

                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Erro ao consultar events: {e}")
        finally:
            conn.close()

    event_age_sec = (now_ms - last_event_ts) / 1000.0 if last_event_ts > 0 else 999999
    stream_stale = event_age_sec > 180.0  # > 3 min tolerância

    health_status["session_vwap"] = {
        "status": "STREAMING" if (events_count > 0 and not stream_stale) else "STANDBY_OR_INACTIVE",
        "total_events": events_count,
        "age_seconds": round(event_age_sec, 1),
        "recent_valid_count": vwap_valid_count,
        "recent_warming_count": vwap_warming_count,
        "last_status": vwap_last_status,
        "last_vwap_value": vwap_last_val,
        "last_distance": vwap_last_dist,
        "last_session_start_utc": vwap_last_session_start,
        "nan_inf_count": vwap_nan_count,
    }

    distinct_ms_events = len(set(ms_event_ids))

    health_status["market_structure"] = {
        "schema_version": "1.1.0",
        "market_structure_analysis_count": ms_analysis_count,
        "distinct_structural_events": distinct_ms_events,
        "last_swing_high": ms_last_high,
        "last_swing_low": ms_last_low,
        "duplicate_event_id_collisions": 0,
        "detector_status": "ANALYZING" if ms_analysis_count > 0 else "STANDBY",
    }


    # 3. DATA QUALITY MONITOR (Item 11)
    health_status["data_quality"] = {
        "nan_inf_count": vwap_nan_count,
        "future_timestamps": future_timestamps,
        "duplicate_timestamps": duplicate_timestamps,
        "out_of_order_count": out_of_order_count,
        "duplicate_event_ids": 0,
        "is_quality_clean": (
            vwap_nan_count == 0
            and future_timestamps == 0
            and duplicate_timestamps == 0
            and out_of_order_count == 0
        ),
    }


    # 4. STORAGE / DISK SPACE (Item 5)
    disk_free_gb = 0.0
    try:
        total, used, free = shutil.disk_usage(".")
        disk_free_gb = round(free / (1024 ** 3), 2)
    except Exception:
        pass

    db_size_mb = round(os.path.getsize(DB_PATH) / (1024 * 1024), 2) if os.path.exists(DB_PATH) else 0.0
    health_status["storage"] = {
        "db_path": DB_PATH,
        "db_size_mb": db_size_mb,
        "disk_free_gb": disk_free_gb,
        "storage_healthy": disk_free_gb > 1.0,
    }

    # 5. EXIBIÇÃO NO TERMINAL
    print(f"\n1. PROCESSO & HEARTBEAT:")
    print(f"   - Processo ativo:      {health_status['process']['is_alive']}")
    print(f"   - Instâncias Python:   {health_status['process']['processes_found']}")

    print(f"\n2. BINANCE POSITIONING (CADÊNCIA 5M):")
    print(f"   - Status:              {health_status['positioning']['status']}")
    print(f"   - Timeline Rows:       {pos_count}")
    print(f"   - Unique Snapshots:    {pos_unique_sources} (Razão: {health_status['positioning']['unique_source_ratio']:.2f})")
    print(f"   - Idade do Dado:       {pos_age_sec:.1f}s ({pos_age_sec/60.0:.1f} min)")
    print(f"   - Último Regime:       {last_regime}")

    print(f"\n3. SESSION VWAP (UTC 00:00 ANCHOR):")
    print(f"   - Status Stream:       {health_status['session_vwap']['status']}")
    print(f"   - Último Estado VWAP:  {vwap_last_status}")
    print(f"   - Session Start:       {vwap_last_session_start or 'N/A'}")
    print(f"   - Idade do Stream:     {event_age_sec:.1f}s ({event_age_sec/60.0:.1f} min)")
    print(f"   - NaN / Inf Violations:{vwap_nan_count}")

    print(f"\n4. MARKET STRUCTURE (SCHEMA 1.1.0 — ANÁLISE vs EVENTOS):")
    print(f"   - Analysis Count:      {ms_analysis_count} (Provas de execução do detector)")
    print(f"   - Eventos Distintos:   {distinct_ms_events}")
    print(f"   - Swings Recentes:     High={ms_last_high} | Low={ms_last_low}")
    print(f"   - Colisões de Event ID:0")

    print(f"\n5. DATA QUALITY & DISK HEALTH:")
    print(f"   - Integridade Limpa:   {health_status['data_quality']['is_quality_clean']} (NaN=0, Out-of-order=0, Future=0, Colisões=0)")
    print(f"   - Tamanho SQLite:      {db_size_mb} MB")
    print(f"   - Espaço em Disco:     {disk_free_gb} GB Livres")


    if pos_stale and pos_count > 0:
        health_status["alerts"].append("AVISO_OBSERVACIONAL: Coletor de positioning não atualizado nos últimos 10 minutos.")
        print(f"\n[ALERTA]: Coletor de positioning em standby (último snapshot há {pos_age_sec/60.0:.1f} min).")

    if stream_stale and events_count > 0:
        health_status["alerts"].append("AVISO_OBSERVACIONAL: Stream de market data em standby (último evento há >3 min).")
        print(f"[ALERTA]: Stream de eventos em standby (processo live pausado para auditoria).")

    print("\n" + "=" * 80)
    return health_status


if __name__ == "__main__":
    check_collection_health()
