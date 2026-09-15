# scripts/analytics/cftc_r4_status.py
# -*- coding: utf-8 -*-
"""
R4 — Relatório de estado (contagens apenas; sem significância enquanto N pequeno).
Próximo checkpoint sempre derivado de N matured (4/13/26/52).

Uso:
  python scripts/analytics/cftc_r4_status.py [--store DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

from scripts.analytics.cftc_r4_prospective_collector import R4Store  # noqa: E402

DEFAULT_STORE = Path("dados/research/cftc/r4_prospective")
CHECKPOINTS = (4, 13, 26, 52)


def status(store_dir: str = str(DEFAULT_STORE)) -> dict:
    store = R4Store(Path(store_dir))
    obs = store.observations()
    outs = {o.get("observation_id"): o for o in store.outcomes()}
    revs = store.revisions()
    matured = [oid for oid in (o.get("observation_id") for o in obs) if oid in outs]
    pending = [o.get("observation_id") for o in obs
               if o.get("observation_id") not in outs and o.get("entry_timestamp")]
    invalid = [o.get("observation_id") for o in obs if not o.get("entry_timestamp")]
    firsts = sorted(o.get("first_seen_at", "") for o in obs if o.get("first_seen_at"))
    next_mat = None
    pend_entries = sorted(o.get("entry_timestamp", "") for o in obs
                          if o.get("observation_id") in pending)
    if pend_entries:
        next_mat = pend_entries[0][:10] + " +28d"
    bfile = Path(store_dir) / "binance.jsonl"
    n_bin = sum(1 for _ in open(bfile, encoding="utf-8")) if bfile.exists() else 0
    nxt = next((c for c in CHECKPOINTS if len(matured) < c), None)
    return {
        "n_observations": len(obs),
        "n_matured": len(matured),
        "n_pending": len(pending),
        "n_invalid": len(invalid),
        "last_first_seen": firsts[-1] if firsts else None,
        "next_pending_maturity": next_mat,
        "revisions_observed": len(revs),
        "binance_rows": n_bin,
        "binance_overlap_rate": (n_bin / len(obs)) if obs else None,
        "next_checkpoint": (f"N={nxt} matured" if nxt
                            else "N>=52: avaliação confirmatória razoável"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    args = ap.parse_args()
    print(json.dumps(status(args.store), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
