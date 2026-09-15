# scripts/analytics/cftc_r4_outcome_updater.py
# -*- coding: utf-8 -*-
"""
R4 — Matura outcomes PENDING (nunca edita observations).

Para cada observação sem outcome: busca candles diários e, se existir candle
com open >= entry + 28d, anexa outcome em outcomes.jsonl. Sem candle elegível,
permanece PENDING. Nunca preenche antecipadamente.

Uso:
  python scripts/analytics/cftc_r4_outcome_updater.py [--store DIR]
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

from scripts.analytics.cftc_r4_prospective_collector import (  # noqa: E402
    R4Store,
    compute_outcome,
    fetch_klines_daily,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CftcR4Outcomes")

DEFAULT_STORE = Path("dados/research/cftc/r4_prospective")


def update_once(store_dir: str = str(DEFAULT_STORE)) -> dict:
    store = R4Store(Path(store_dir))
    done_ids = set()
    for o in store.outcomes():
        if isinstance(o.get("observation_id"), str):
            done_ids.add(o["observation_id"])
    klines = asyncio.run(fetch_klines_daily(
        "BTCUSDT", int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)))
    summary = {"matured": [], "pending": [], "invalid": []}
    for obs in store.observations():
        oid = obs.get("observation_id")
        if not oid or oid in done_ids:
            continue
        if not obs.get("entry_timestamp"):
            summary["invalid"].append(oid)
            continue
        out = compute_outcome(klines, obs["entry_timestamp"])
        if out is None:
            summary["pending"].append(oid)
            continue
        entry_px = float(obs["entry_price"])
        fwd = out["exit_price"] / entry_px - 1.0
        store.append_outcome({
            "observation_id": oid,
            "report_as_of_date": obs.get("report_as_of_date"),
            "entry_timestamp": obs["entry_timestamp"],
            "entry_price": entry_px,
            "exit_timestamp": out["exit_timestamp"],
            "exit_price": out["exit_price"],
            "forward_return_28d": fwd,
            "matured_at": datetime.now(timezone.utc).isoformat()})
        summary["matured"].append(oid)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    args = ap.parse_args()
    print(json.dumps(update_once(args.store), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
