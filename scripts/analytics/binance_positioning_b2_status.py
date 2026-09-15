# scripts/analytics/binance_positioning_b2_status.py
# -*- coding: utf-8 -*-
"""B2 — Estado do dataset (contagens; sem inferência).

Uso:
  python scripts/analytics/binance_positioning_b2_status.py [--symbol BTCUSDT] [--store DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from binance_positioning_b2_seed import DEFAULT_STORE, health_of  # noqa: E402
from binance_positioning_store import B2Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--store", default=DEFAULT_STORE)
    args = ap.parse_args()
    store = B2Store(args.store)
    h = health_of(store, args.symbol)
    meta_path = f"{args.store}/metadata/{args.symbol}.json"
    try:
        with open(meta_path, encoding="utf-8") as fh:
            h["metadata"] = json.load(fh)
    except (OSError, ValueError):
        h["metadata"] = None
    print(json.dumps(h, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
