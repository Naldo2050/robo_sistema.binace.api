# scripts/analytics/binance_positioning_store.py
# -*- coding: utf-8 -*-
"""
B2 — Store append-only do dataset prospectivo Binance positioning (research).

Layout (tudo sob dados/research/binance_positioning/, gitignored):
  raw/{symbol}/{endpoint}.jsonl        # resposta oficial máxima, 1 linha/barra
  raw/{symbol}/funding.jsonl           # trilha separada (fundingTime)
  normalized/{symbol}.jsonl            # snapshots canônicos 5m append-only
  metadata/{symbol}.json               # provenance + contadores
  health/{symbol}.json                 # métricas B2.9/B2.14

Regras:
  - dedup por (symbol, endpoint, source_timestamp); re-fetch idêntico não duplica;
  - re-fetch com valor alterado -> nova revisão preservada (nunca sobrescreve);
  - backfill: collection_mode=HISTORICAL_API_BACKFILL, first_seen_at=null,
    usable_for_strict_pit=false;
  - live: collection_mode=LIVE_OBSERVED, first_seen_at=primeiro retrieved real;
  - JSONL append-only; restart-safe via reload de índices.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
ENDPOINTS = ("global", "top_account", "top_position", "oi", "funding")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canon_hash(obj: dict) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:16]


def _num(raw, field: str, problems: list):
    if raw is None or isinstance(raw, bool):
        problems.append(f"missing:{field}")
        return None
    try:
        f = float(str(raw).strip().replace(",", ""))
    except (ValueError, TypeError):
        problems.append(f"unparsable:{field}")
        return None
    if not math.isfinite(f):
        problems.append(f"nonfinite:{field}")
        return None
    if f < 0:
        problems.append(f"negative:{field}")
        return None
    return f


def long_share(ratio):
    if ratio is None:
        return None
    return ratio / (1.0 + ratio)


class B2Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for sub in ("raw", "normalized", "metadata", "health"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        # índice em memória: (symbol, endpoint, ts) -> listaha revisões
        self._raw_index: dict = {}
        self._norm_ids: set = set()
        self._reload()

    # -- paths -----------------------------------------------------------
    def _raw_path(self, symbol: str, endpoint: str) -> Path:
        d = self.root / "raw" / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{endpoint}.jsonl"

    def _norm_path(self, symbol: str) -> Path:
        return self.root / "normalized" / f"{symbol}.jsonl"

    # -- reload (restart-safe) -------------------------------------------
    def _reload(self) -> None:
        for ep_file in (self.root / "raw").rglob("*.jsonl"):
            try:
                symbol = ep_file.parent.name
            except Exception:  # noqa: BLE001
                continue
            endpoint = ep_file.stem
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    continue
                ts = rec.get("source_timestamp")
                if ts is None:
                    continue
                self._raw_index.setdefault(
                    (symbol, endpoint, int(ts)), []).append(rec)
        norm_dir = self.root / "normalized"
        if norm_dir.exists():
            for nf in norm_dir.glob("*.jsonl"):
                for line in nf.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    oid = rec.get("snapshot_id")
                    if oid:
                        self._norm_ids.add(oid)

    # -- raw ---------------------------------------------------------------
    def append_raw(self, symbol: str, endpoint: str, source_timestamp: int,
                   values: dict, retrieved_at: str | None = None,
                   mode: str = "LIVE_OBSERVED",
                   first_seen_at: str | None = None) -> dict:
        """Retorna {stored: bool, revision: int, duplicate: bool}."""
        retrieved_at = retrieved_at or _utcnow_iso()
        if mode == "LIVE_OBSERVED" and not first_seen_at:
            first_seen_at = retrieved_at
        if mode == "HISTORICAL_API_BACKFILL":
            first_seen_at = None
        key = (symbol, endpoint, int(source_timestamp))
        body_hash = _canon_hash(values)
        for rev in self._raw_index.get(key, []):
            if rev.get("body_hash") == body_hash:
                return {"stored": False, "revision": rev.get("revision", 0),
                        "duplicate": True}
        revision = len(self._raw_index.get(key, []))
        rec = {"schema_version": SCHEMA_VERSION, "symbol": symbol,
               "endpoint": endpoint, "source_timestamp": int(source_timestamp),
               "retrieved_at": retrieved_at, "first_seen_at": first_seen_at,
               "collection_mode": mode,
               "usable_for_research": True,
               "usable_for_strict_pit": mode == "LIVE_OBSERVED",
               "revision": revision, "body_hash": body_hash, "values": values}
        with open(self._raw_path(symbol, endpoint), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._raw_index.setdefault(key, []).append(rec)
        return {"stored": True, "revision": revision, "duplicate": False}

    def raw_series(self, symbol: str, endpoint: str) -> list:
        """Todas as revisões ordenadas por (ts, revision)."""
        out = [r for (s, e, _), revs in self._raw_index.items()
               if s == symbol and e == endpoint for r in revs]
        out.sort(key=lambda r: (r["source_timestamp"], r.get("revision", 0)))
        return out

    def latest_known(self, symbol: str, endpoint: str) -> dict | None:
        """Última revisão do maior timestamp (nunca valor futuro)."""
        series = self.raw_series(symbol, endpoint)
        if not series:
            return None
        top_ts = series[-1]["source_timestamp"]
        cands = [r for r in series if r["source_timestamp"] == top_ts]
        return max(cands, key=lambda r: r.get("revision", 0))

    # -- normalized ----------------------------------------------------------
    def append_normalized(self, snap: dict) -> bool:
        oid = snap.get("snapshot_id")
        if not oid or oid in self._norm_ids:
            return False
        with open(self._norm_path(snap["symbol"]), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
        self._norm_ids.add(oid)
        return True

    def normalized_rows(self, symbol: str) -> list:
        p = self._norm_path(symbol)
        if not p.exists():
            return []
        return [json.loads(line) for line in
                p.read_text(encoding="utf-8").splitlines() if line.strip()]

    # -- metadata / health -----------------------------------------------------
    def write_metadata(self, symbol: str, meta: dict) -> None:
        p = self.root / "metadata" / f"{symbol}.json"
        meta = {"schema_version": SCHEMA_VERSION, **meta,
                "updated_at": _utcnow_iso()}
        p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_health(self, symbol: str, health: dict) -> None:
        p = self.root / "health" / f"{symbol}.json"
        health = {**health, "updated_at": _utcnow_iso()}
        p.write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
