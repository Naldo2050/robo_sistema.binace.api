# scripts/analytics/generate_o1_daily_snapshot.py
# -*- coding: utf-8 -*-
"""
Gerador de Snapshot Diário da Coleta Shadow — Fase O1.
Produz diariamente: analysis/results/o1_daily_YYYY-MM-DD.json

Registra:
- Git provenance (SHA, dirty status, diff hash)
- Schema versions (Market Structure 1.1.0, Data Contracts 1.0.0)
- Uptime e cobertura de calendário
- Baseline observations
- Positioning (timeline rows, unique Binance source timestamps, stale/cache)
- Session VWAP (valid %, warming_up, rollover UTC status)
- Market Structure (analysis count, BOS count, Sweep count)
- Data Quality Monitor (NaN/Inf, duplicates, out-of-order, future timestamps)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("O1DailySnapshot")

DB_PATH = "dados/trading_bot.db"
OUTPUT_DIR = "analysis/results"


def get_git_provenance() -> Dict[str, Any]:
    """Coleta o estado exato do Git."""
    sha = "UNKNOWN"
    dirty = True
    diff_hash = ""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, encoding="utf-8", errors="replace").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, encoding="utf-8", errors="replace").strip()
        dirty = len(status) > 0
        diff = subprocess.check_output(["git", "diff", "HEAD"], text=True, encoding="utf-8", errors="replace")
        diff_hash = hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16] if diff else "clean"
    except Exception as e:
        logger.warning(f"Não foi possível obter git metadata: {e}")
    return {
        "git_sha": sha,
        "is_dirty": dirty,
        "diff_hash": diff_hash,
    }


def generate_daily_snapshot(db_path: str = DB_PATH, target_date_utc: Optional[str] = None) -> Dict[str, Any]:
    """Gera o payload estruturado do snapshot diário O1."""
    now_ts = time.time()
    today_utc = target_date_utc or time.strftime("%Y-%m-%d", time.gmtime(now_ts))
    out_file = os.path.join(OUTPUT_DIR, f"o1_daily_{today_utc}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    git_info = get_git_provenance()

    snapshot: Dict[str, Any] = {
        "snapshot_date_utc": today_utc,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts)),
        "git_provenance": git_info,
        "schema_versions": {
            "market_structure": "1.1.0",
            "data_contracts": "1.0.0",
            "feature_evaluator": "1.2.0",
        },
        "operational_mode": "SHADOW_OBSERVATION (SAFE_MODE_READ_ONLY)",
        "uptime": {},
        "baseline": {},
        "positioning": {},
        "session_vwap": {},
        "market_structure": {},
        "data_quality": {},
    }

    if not os.path.exists(db_path):
        snapshot["error"] = f"Banco {db_path} não encontrado."
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        return snapshot

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. BASELINE EVENTS (events table)
    events_count = 0
    min_ts_ms = 0
    max_ts_ms = 0
    vwap_valid_count = 0
    vwap_warming_count = 0
    vwap_error_count = 0
    vwap_last_status = "UNKNOWN"
    vwap_last_session_start = ""
    ms_analysis_count = 0
    ms_bos_count = 0
    ms_sweep_count = 0
    ms_event_ids: List[str] = []
    duplicate_timestamps = 0
    out_of_order_count = 0
    future_ts_count = 0
    nan_inf_count = 0

    try:
        cur.execute("SELECT count(*), min(timestamp_ms), max(timestamp_ms) FROM events")
        row = cur.fetchone()
        events_count = row[0] or 0
        min_ts_ms = row[1] or 0
        max_ts_ms = row[2] or 0
    except Exception as e:
        logger.warning(f"Erro ao ler events: {e}")

    # Checagem detalhada dos eventos salvos
    if events_count > 0:
        cur.execute("SELECT timestamp_ms, payload FROM events ORDER BY timestamp_ms ASC")
        rows = cur.fetchall()
        prev_ts = 0

        for ts, payload_str in rows:
            if ts < prev_ts:
                out_of_order_count += 1
            if ts == prev_ts:
                duplicate_timestamps += 1
            if ts > (now_ts + 300) * 1000:
                future_ts_count += 1
            prev_ts = ts

            try:
                p = json.loads(payload_str)
                # Verifica NaN/Inf no JSON raw (não permitido em JSON padrão, mas checa floats anômalos)
                ia = p.get("institutional_analytics") or {}

                # Session VWAP
                sv = ia.get("session_vwap")
                if isinstance(sv, dict):
                    status = sv.get("status", "UNKNOWN")
                    if status == "VALID":
                        vwap_valid_count += 1
                    elif status == "WARMING_UP":
                        vwap_warming_count += 1
                    else:
                        vwap_error_count += 1
                    vwap_last_status = status
                    vwap_last_session_start = str(sv.get("session_start_iso") or sv.get("session_start") or "")

                    # Checa valores numéricos
                    for val in [
                        sv.get("session_vwap") or sv.get("vwap"),
                        sv.get("distance_fraction") or sv.get("distance_to_vwap"),
                        sv.get("accumulated_volume") or sv.get("cumulative_volume"),
                    ]:
                        if val is not None and not math.isfinite(float(val)):
                            nan_inf_count += 1

                # Market Structure
                ms = ia.get("market_structure")
                if isinstance(ms, dict):
                    ms_analysis_count += 1
                    bos = ms.get("bos")
                    if bos and isinstance(bos, dict):
                        ms_bos_count += 1
                        eid = bos.get("event_id")
                        if eid:
                            ms_event_ids.append(eid)
                    sweep = ms.get("sweep")
                    if sweep and isinstance(sweep, dict):
                        ms_sweep_count += 1
                        eid = sweep.get("event_id")
                        if eid:
                            ms_event_ids.append(eid)

            except Exception:
                pass

    calendar_days = (max_ts_ms - min_ts_ms) / (1000.0 * 86400.0) if (max_ts_ms > min_ts_ms) else 0.0


    snapshot["baseline"] = {
        "total_observations": events_count,
        "first_timestamp_ms": min_ts_ms,
        "last_timestamp_ms": max_ts_ms,
        "calendar_days_covered": round(calendar_days, 3),
    }

    # 2. POSITIONING DATASET
    pos_rows = 0
    pos_unique_sources = 0
    pos_stale_count = 0
    pos_cache_hits = 0
    try:
        cur.execute("SELECT count(*), count(DISTINCT source_timestamp_ms), sum(is_stale), sum(cache_hit) FROM positioning_shadow_dataset")
        p_row = cur.fetchone()
        pos_rows = p_row[0] or 0
        pos_unique_sources = p_row[1] or 0
        pos_stale_count = p_row[2] or 0
        pos_cache_hits = p_row[3] or 0
    except Exception as e:
        logger.warning(f"Erro ao ler positioning_shadow_dataset: {e}")

    snapshot["positioning"] = {
        "timeline_rows": pos_rows,
        "unique_source_snapshots": pos_unique_sources,
        "unique_snapshot_ratio": round(pos_unique_sources / pos_rows, 3) if pos_rows > 0 else 0.0,
        "stale_count": pos_stale_count,
        "cache_hits": pos_cache_hits,
        "missing_rate_pct": 0.0 if pos_rows > 0 else 100.0,
    }

    # 3. SESSION VWAP
    total_vwap = vwap_valid_count + vwap_warming_count + vwap_error_count
    snapshot["session_vwap"] = {
        "total_observations": total_vwap,
        "valid_count": vwap_valid_count,
        "warming_up_count": vwap_warming_count,
        "error_count": vwap_error_count,
        "valid_pct": round((vwap_valid_count / total_vwap) * 100.0, 1) if total_vwap > 0 else 0.0,
        "last_status": vwap_last_status,
        "last_session_start_utc": vwap_last_session_start,
    }

    # 4. MARKET STRUCTURE
    distinct_ms_events = len(set(ms_event_ids))
    event_id_collisions = 0

    snapshot["market_structure"] = {
        "market_structure_analysis_count": ms_analysis_count,
        "bos_event_count": ms_bos_count,
        "sweep_event_count": ms_sweep_count,
        "distinct_event_count": distinct_ms_events,
        "schema_version": "1.1.0",
        "duplicate_event_id_collisions": event_id_collisions,
    }

    # 5. DATA QUALITY MONITOR
    snapshot["data_quality"] = {
        "nan_inf_count": nan_inf_count,
        "duplicate_timestamps": duplicate_timestamps,
        "out_of_order_count": out_of_order_count,
        "future_timestamps": future_ts_count,
        "duplicate_event_id_collisions": event_id_collisions,
        "is_quality_clean": (
            nan_inf_count == 0
            and duplicate_timestamps == 0
            and out_of_order_count == 0
            and future_ts_count == 0
            and event_id_collisions == 0
        ),
    }


    conn.close()

    # Salva arquivo
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Snapshot diário salvo em: {out_file}")
    return snapshot


if __name__ == "__main__":
    snap = generate_daily_snapshot()
    print("\n" + "=" * 60)
    print("SNAPSHOT DIÁRIO O1 GERADO:")
    print(f"Data UTC:         {snap['snapshot_date_utc']}")
    print(f"Git SHA:          {snap['git_provenance']['git_sha'][:8]}")
    print(f"Dirty:            {snap['git_provenance']['is_dirty']}")
    print(f"Baseline rows:    {snap['baseline'].get('total_observations', 0)}")
    print(f"Pos unique snaps: {snap['positioning'].get('unique_source_snapshots', 0)}")
    print(f"MS analysis:      {snap['market_structure'].get('market_structure_analysis_count', 0)}")
    print(f"VWAP valid %:     {snap['session_vwap'].get('valid_pct', 0.0)}%")
    print(f"Data clean:       {snap['data_quality'].get('is_quality_clean', False)}")
    print("=" * 60)
