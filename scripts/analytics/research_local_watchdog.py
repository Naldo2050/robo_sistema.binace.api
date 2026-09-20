# scripts/analytics/research_local_watchdog.py
# -*- coding: utf-8 -*-
"""
Watchdog da coleta LOCAL temporária (Windows). Não altera thresholds OCI.

Critério (positioning, source_timestamp mais recente):
  OK   : idade <= 30 min
  WARN : idade > 30 min   (exit 1)
  CRIT : idade > 2 horas  (exit 2)
  UNKNOWN (sem dados): exit 3

Comando único:
  python scripts/analytics/research_local_watchdog.py [--symbol BTCUSDT] [--store DIR]

Research-only. Sem produção, sem rede (lê o store local).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from binance_positioning_store import B2Store  # noqa: E402

WARN_MIN = 30.0
CRIT_MIN = 120.0
EXIT_OK, EXIT_WARN, EXIT_CRIT, EXIT_UNKNOWN = 0, 1, 2, 3


def check(symbol: str = "BTCUSDT",
          store_dir: str = "dados/research/binance_positioning",
          now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    store = B2Store(store_dir)
    latest = None
    for ep in ("global", "top_account", "top_position", "oi"):
        rec = store.latest_known(symbol, ep)
        if rec and (latest is None or rec["source_timestamp"] > latest):
            latest = rec["source_timestamp"]
    if latest is None:
        return {"level": "UNKNOWN", "exit": EXIT_UNKNOWN,
                "latest_source_timestamp": None, "age_min": None}
    age_min = (now.timestamp() * 1000 - latest) / 60000.0
    latest_iso = datetime.fromtimestamp(latest / 1000, tz=timezone.utc).isoformat()
    if age_min > CRIT_MIN:
        level, code = "CRIT", EXIT_CRIT
    elif age_min > WARN_MIN:
        level, code = "WARN", EXIT_WARN
    else:
        level, code = "OK", EXIT_OK
    return {"level": level, "exit": code,
            "latest_source_timestamp": latest,
            "latest_source_iso": latest_iso,
            "age_min": round(age_min, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--store", default="dados/research/binance_positioning")
    args = ap.parse_args(argv)
    res = check(args.symbol, args.store)
    print(json.dumps(res, ensure_ascii=False))
    return res["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
