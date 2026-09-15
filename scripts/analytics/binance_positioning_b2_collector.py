# scripts/analytics/binance_positioning_b2_collector.py
# -*- coding: utf-8 -*-
"""
B2 — Coleta prospectiva 15m (research, sem produção).

Cada ciclo busca sobreposição (últimas 5h = 60 barras 5m) e persiste SÓ
barras novas. Gap maior que a sobreposição dispara backfill da janela
faltante (B2.11); gap além da retenção (~30d) vira unrecoverable_gap
(registrado, nunca fabricado). Novas barras: LIVE_OBSERVED + first_seen.
Re-fetch idêntico não duplica; valor alterado vira revisão preservada.

NÃO inicia daemon sozinho: rode manualmente/agendador externo.

Uso:
  python scripts/analytics/binance_positioning_b2_collector.py [--symbol BTCUSDT] [--store DIR]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import aiohttp

from binance_positioning_b2_seed import (
    DEFAULT_STORE,
    ENDPOINTS,
    PERIOD_MS,
    build_snapshots,
    fetch_window,
    health_of,
)
from binance_positioning_store import B2Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("B2Collector")

OVERLAP_BARS = 60  # 5h de sobreposição por ciclo
MAX_RECOVERABLE_MS = 30 * 24 * 3600 * 1000


def plan_recovery(last_known_ms: int | None, now_ms: int) -> dict:
    """Regra pura de recuperação (testável, sem I/O).

    overlap: gap <= 5h -> busca com sobreposição;
    backfill: gap <= 30d -> janela faltante desde (last - overlap);
    unrecoverable: gap > 30d -> registra, nunca fabrica.
    """
    if last_known_ms is None:
        return {"mode": "overlap",
                "start_ms": now_ms - OVERLAP_BARS * PERIOD_MS}
    gap_ms = now_ms - last_known_ms
    if gap_ms > MAX_RECOVERABLE_MS:
        return {"mode": "unrecoverable", "start_ms": None, "gap_ms": gap_ms}
    if gap_ms <= OVERLAP_BARS * PERIOD_MS:
        return {"mode": "overlap",
                "start_ms": last_known_ms - OVERLAP_BARS * PERIOD_MS,
                "gap_ms": gap_ms}
    return {"mode": "backfill",
            "start_ms": last_known_ms - OVERLAP_BARS * PERIOD_MS,
            "gap_ms": gap_ms}


async def collect_cycle(symbol: str, store_dir: str) -> dict:
    store = B2Store(store_dir)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    report: dict = {"symbol": symbol, "endpoints": {}, "unrecoverable_gaps": []}
    connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), connector=connector,
            headers={"User-Agent": "MarketBot-B2Collector/1.0"}) as session:
        for ep, path in ENDPOINTS.items():
            known = store.raw_series(symbol, ep)
            known_ts = {r["source_timestamp"] for r in known}
            plan = plan_recovery(max(known_ts) if known_ts else None, now_ms)
            if plan["mode"] == "unrecoverable":
                report["unrecoverable_gaps"].append(
                    {"endpoint": ep, "last_known": max(known_ts) if known_ts else None,
                     "reason": "gap_beyond_retention"})
                report["endpoints"][ep] = {"new": 0, "status": "unrecoverable_gap"}
                continue
            start_ms = plan["start_ms"]
            rows, stats = await fetch_window(session, path, symbol, start_ms, now_ms)
            new, dups, revs = 0, 0, 0
            for r in rows:
                if not isinstance(r, dict) or r.get("timestamp") is None:
                    continue
                res = store.append_raw(symbol, ep, int(r["timestamp"]), r,
                                       retrieved_at=retrieved_at,
                                       mode="LIVE_OBSERVED")
                if res["stored"]:
                    new += 1
                    if res["revision"] > 0:
                        revs += 1
                elif res["duplicate"]:
                    dups += 1
            report["endpoints"][ep] = {"new": new, "duplicates_suppressed": dups,
                                       "revisions": revs, **stats}
    snap_stats = build_snapshots(store, symbol, "LIVE_OBSERVED", retrieved_at)
    report["normalized_5m"] = snap_stats
    store.write_metadata(symbol, {"mode": "collector", "last_cycle": retrieved_at})
    store.write_health(symbol, {**health_of(store, symbol),
                                "last_successful_collection": retrieved_at})
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--store", default=DEFAULT_STORE)
    args = ap.parse_args()
    print(json.dumps(asyncio.run(collect_cycle(args.symbol, args.store)),
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
