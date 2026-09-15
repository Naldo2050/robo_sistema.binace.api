# scripts/analytics/binance_positioning_b2_seed.py
# -*- coding: utf-8 -*-
"""
B2 — Seed/backfill 30d Binance positioning (research, sem produção).

Busca toda a janela disponível dos 4 endpoints (period=5m) + funding full,
valida paginação integral, alinha por timestamp (backward-asof, nunca
forward) e constrói snapshots canônicos 5m. Backfill carrega
collection_mode=HISTORICAL_API_BACKFILL, first_seen_at=null,
usable_for_strict_pit=false.

Uso:
  python scripts/analytics/binance_positioning_b2_seed.py [--symbol BTCUSDT] [--store DIR]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import aiohttp

from binance_positioning_store import B2Store, long_share

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("B2Seed")

BASE = "https://fapi.binance.com"
PERIOD = "5m"
PERIOD_MS = 5 * 60 * 1000
PAGE_LIMIT = 500
LOOKBACK_MS = 31 * 24 * 3600 * 1000
TOLERANCE_MS = 10 * 60 * 1000  # skew máximo entre endpoints (explícito)
DEFAULT_STORE = "dados/research/binance_positioning"

ENDPOINTS = {
    "global": "/futures/data/globalLongShortAccountRatio",
    "top_account": "/futures/data/topLongShortAccountRatio",
    "top_position": "/futures/data/topLongShortPositionRatio",
    "oi": "/futures/data/openInterestHist",
}


async def _get(session, path, params):
    timeout = aiohttp.ClientTimeout(total=15)
    async with session.get(BASE + path, params=params, timeout=timeout) as resp:
        if resp.status != 200:
            return None, f"http_{resp.status}"
        try:
            data = await resp.json()
        except Exception:  # noqa: BLE001
            return None, "bad_json"
        if isinstance(data, list):
            return data, None
        return None, "schema_error"


async def fetch_window(session, path, symbol, start_ms, end_ms):
    """Paginação BACKWARD integral (500/request).

    Motivo (verificado 2026-09-15): startTime além da retenção de ~30d
    retorna HTTP 400 code -1130. Anda endTime para trás a partir do agora;
    para quando o lote mais antigo cruza start_ms, esvazia ou dá erro
    persistente. Retorna (rows, stats).
    """
    rows, errors, pages = [], 0, 0
    cursor = end_ms
    while True:
        batch, err = await _get(session, path, {"symbol": symbol, "period": PERIOD,
                                                "limit": PAGE_LIMIT,
                                                "endTime": cursor})
        pages += 1
        if err:
            errors += 1
            if errors > 5:
                break
            await asyncio.sleep(1.0)
            continue
        if not batch:
            break
        rows.extend(batch)
        ts = sorted(int(r.get("timestamp", 0)) for r in batch if isinstance(r, dict))
        if not ts or ts[0] <= start_ms:
            break
        cursor = ts[0] - 1
        await asyncio.sleep(0.3)
    return rows, {"pages": pages, "http_errors": errors}


async def fetch_funding_full(session, symbol):
    """Backward pagination (endpoint devolve o recente primeiro)."""
    rows, errors = [], 0
    end = None
    while True:
        params = {"symbol": symbol, "limit": 1000}
        if end is not None:
            params["endTime"] = end
        timeout = aiohttp.ClientTimeout(total=15)
        async with session.get(BASE + "/fapi/v1/fundingRate", params=params,
                               timeout=timeout) as resp:
            if resp.status != 200:
                errors += 1
                if errors > 5:
                    break
                await asyncio.sleep(1.0)
                continue
            batch = await resp.json()
        if not batch:
            break
        rows.extend(batch)
        oldest = min(int(r["fundingTime"]) for r in batch)
        if len(batch) < 1000:
            # pode haver mais para trás: continua até esvaziar
            end = oldest - 1
            if oldest <= 1262304000000:  # 2010, segurança
                break
            await asyncio.sleep(0.3)
            continue
        end = oldest - 1
        await asyncio.sleep(0.3)
    return rows, {"http_errors": errors}


def _clean_num(v, field, problems):
    if v is None or isinstance(v, bool):
        problems.append(f"missing:{field}")
        return None
    try:
        f = float(str(v).strip().replace(",", ""))
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


def build_snapshots(store: B2Store, symbol: str, mode: str,
                    retrieved_at: str) -> dict:
    """Alinhamento backward-asof por timestamp (nunca forward).

    Grade = união dos timestamps 5m observados. Para cada T, cada endpoint
    contribui com a barra de maior ts <= T dentro de TOLERANCE_MS.
    Ausente -> null + missing_fields (linha NUNCA descartada por isso).
    """
    series = {}
    for ep in ("global", "top_account", "top_position", "oi"):
        bars = {}
        for r in store.raw_series(symbol, ep):
            v = r["values"]
            bars.setdefault(r["source_timestamp"], []).append((r.get("revision", 0), v))
        # última revisão por ts
        series[ep] = {ts: max(vs)[1] for ts, vs in bars.items()}
    grid = sorted({ts for ep in series.values() for ts in ep})
    by_ep_ts = {ep: sorted(d) for ep, d in series.items()}
    stats = {"snapshots": 0, "complete": 0, "partial": 0, "revisions": 0}
    existing_by_ts: dict = {}
    for oid in store._norm_ids:
        # snapshot_id = f"{symbol}_{ts}_r{rev}"
        try:
            _, ts_part, _ = oid.rsplit("_", 2)
            existing_by_ts.setdefault(int(ts_part), []).append(oid)
        except (ValueError, TypeError):
            continue
    existing_rows = {r.get("source_timestamp"): r
                     for r in store.normalized_rows(symbol)}
    for ts in grid:
        comp, missing, ages, prov = {}, [], {}, {}
        for ep, stamps in by_ep_ts.items():
            import bisect

            i = bisect.bisect_right(stamps, ts) - 1
            if i >= 0 and ts - stamps[i] <= TOLERANCE_MS:
                comp[ep] = series[ep][stamps[i]]
                ages[ep] = ts - stamps[i]
                prov[ep] = {"source_timestamp": stamps[i]}
            else:
                comp[ep] = None
                missing.append(ep)
        snap = assemble_snapshot(symbol, ts, comp, missing, ages, prov, mode,
                                 retrieved_at)
        # revisionamento idempotente: só anexa se o conteúdo mudou.
        # (re-execução sem mudança NÃO duplica o histórico.)
        # collection_mode/first_seen/retrieved NÃO entram no hash: revisão
        # significa mudança no dado de mercado observado, não no modo.
        _SKIP = ("snapshot_id", "revision", "retrieved_at", "first_seen_at",
                 "collection_mode")
        core = {k: v for k, v in snap.items() if k not in _SKIP}
        core_hash = json.dumps(core, sort_keys=True, ensure_ascii=False, default=str)
        prev = existing_rows.get(ts)
        rev = len(existing_by_ts.get(ts, []))
        if prev is not None:
            prev_core = {k: v for k, v in prev.items() if k not in _SKIP}
            if json.dumps(prev_core, sort_keys=True, ensure_ascii=False,
                          default=str) == core_hash:
                continue  # idêntico: nada a anexar
        snap["snapshot_id"] = f"{symbol}_{ts}_r{rev}"
        snap["revision"] = rev
        if rev > 0:
            stats["revisions"] += 1
            snap.setdefault("quality", {}).setdefault("revision_flags", []).append(
                "component_revised")
        if store.append_normalized(snap):
            stats["snapshots"] += 1
            if not missing:
                stats["complete"] += 1
            else:
                stats["partial"] += 1
    return stats


def assemble_snapshot(symbol, ts, comp, missing, ages, prov, mode, retrieved_at):
    q_missing = list(missing)
    g = comp.get("global") or {}
    ta = comp.get("top_account") or {}
    tp = comp.get("top_position") or {}
    oi = comp.get("oi") or {}
    ga = _clean_num(g.get("longShortRatio"), "global_account_ratio", q_missing) \
        if "global" not in missing else None
    if "global" in missing:
        q_missing.append("missing:global_account_ratio")
    ta_r = _clean_num(ta.get("longShortRatio"), "top_account_ratio", q_missing) \
        if "top_account" not in missing else None
    if "top_account" in missing:
        q_missing.append("missing:top_account_ratio")
    tp_r = _clean_num(tp.get("longShortRatio"), "top_position_ratio", q_missing) \
        if "top_position" not in missing else None
    if "top_position" in missing:
        q_missing.append("missing:top_position_ratio")
    oi_v = _clean_num(oi.get("sumOpenInterest"), "open_interest", q_missing) \
        if "oi" not in missing else None
    if "oi" in missing:
        q_missing.append("missing:open_interest")
    oi_usd = _clean_num((oi or {}).get("sumOpenInterestValue"), "open_interest_value",
                        q_missing) if "oi" not in missing else None
    gls, gss = None, None
    if ga is not None:
        gls, gss = long_share(ga), 1 - long_share(ga)
    row = {
        "schema_version": 1, "symbol": symbol, "source_timestamp": ts,
        "retrieved_at": retrieved_at,
        "first_seen_at": None if mode == "HISTORICAL_API_BACKFILL" else retrieved_at,
        "collection_mode": mode,
        "global_account_ratio": ga, "global_long_share": gls, "global_short_share": gss,
        "top_account_ratio": ta_r,
        "top_account_long_share": long_share(ta_r),
        "top_account_short_share": (1 - long_share(ta_r)) if ta_r is not None else None,
        "top_position_ratio": tp_r,
        "top_position_long_share": long_share(tp_r),
        "top_position_short_share": (1 - long_share(tp_r)) if tp_r is not None else None,
        "divergence_account_pp": (long_share(ta_r) - long_share(ga)) * 100
        if (ta_r is not None and ga is not None) else None,
        "divergence_position_pp": (long_share(tp_r) - long_share(ga)) * 100
        if (tp_r is not None and ga is not None) else None,
        "open_interest": oi_v, "open_interest_value": oi_usd,
        "quality": {"status": "complete" if not q_missing else "partial",
                    "missing_fields": sorted(set(q_missing)),
                    "source_ages": ages, "revision_flags": []},
        "provenance": {"endpoints": ["globalLongShortAccountRatio",
                                     "topLongShortAccountRatio",
                                     "topLongShortPositionRatio",
                                     "openInterestHist"],
                       "period": PERIOD,
                       "component_timestamps": {k: v["source_timestamp"] for k, v in prov.items()}},
    }
    return row


def seed_symbol(symbol: str, store_dir: str) -> dict:
    store = B2Store(store_dir)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - LOOKBACK_MS
    report: dict = {"symbol": symbol, "endpoints": {}, "funding": {}}
    retrieved_at = datetime.now(timezone.utc).isoformat()

    async def _run():
        connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30), connector=connector,
                headers={"User-Agent": "MarketBot-B2Seed/1.0"}) as session:
            for ep, path in ENDPOINTS.items():
                rows, stats = await fetch_window(session, path, symbol, start_ms, now_ms)
                stored, dups, revs = 0, 0, 0
                seen_ts = set()
                for r in rows:
                    if not isinstance(r, dict) or r.get("timestamp") is None:
                        continue
                    ts = int(r["timestamp"])
                    if ts in seen_ts:
                        dups += 1
                    seen_ts.add(ts)
                    res = store.append_raw(symbol, ep, ts, r,
                                           retrieved_at=retrieved_at,
                                           mode="HISTORICAL_API_BACKFILL")
                    if res["stored"]:
                        stored += 1
                        if res["revision"] > 0:
                            revs += 1
                    elif res["duplicate"]:
                        dups += 1
                # gaps na grade 5m
                ticks = sorted(seen_ts)
                gaps = sum(1 for a, b in zip(ticks, ticks[1:]) if b - a > PERIOD_MS)
                actual = {"first": min(ticks) if ticks else None,
                          "last": max(ticks) if ticks else None}
                report["endpoints"][ep] = {
                    "requested_start": start_ms, "requested_end": now_ms,
                    "actual_first": actual["first"], "actual_last": actual["last"],
                    "rows": len(rows), "stored": stored, "duplicates": dups,
                    "revisions": revs, "gaps": gaps, **stats}
            frows, fstats = await fetch_funding_full(session, symbol)
            fstored = 0
            for r in frows:
                res = store.append_raw(symbol, "funding", int(r["fundingTime"]), r,
                                       retrieved_at=retrieved_at,
                                       mode="HISTORICAL_API_BACKFILL")
                if res["stored"]:
                    fstored += 1
            fts = sorted(int(r["fundingTime"]) for r in frows) if frows else []
            report["funding"] = {"first": fts[0] if fts else None,
                                 "last": fts[-1] if fts else None,
                                 "rows": len(frows), "stored": fstored, **fstats}

    asyncio.run(_run())
    snap_stats = build_snapshots(store, symbol, "HISTORICAL_API_BACKFILL", retrieved_at)
    report["normalized_5m"] = snap_stats
    # metadata + health
    store.write_metadata(symbol, {"mode": "seed", "period": PERIOD,
                                  "tolerance_ms": TOLERANCE_MS,
                                  "report": {k: v for k, v in report.items()
                                             if k != "normalized_5m"}})
    store.write_health(symbol, health_of(store, symbol))
    return report


def health_of(store: B2Store, symbol: str) -> dict:
    cov = {}
    for ep in ("global", "top_account", "top_position", "oi", "funding"):
        series = store.raw_series(symbol, ep)
        ts = [r["source_timestamp"] for r in series]
        live = sum(1 for r in series if r.get("first_seen_at"))
        cov[ep] = {"rows": len(series),
                   "first": min(ts) if ts else None,
                   "last": max(ts) if ts else None,
                   "live_first_seen_coverage": (live / len(series)) if series else None}
    norm = store.normalized_rows(symbol)
    complete = sum(1 for r in norm if r.get("quality", {}).get("status") == "complete")
    revs = sum(1 for r in norm if r.get("revision", 0) > 0)
    return {"symbol": symbol, "coverage": cov,
            "normalized_rows": len(norm),
            "complete_snapshot_rate": (complete / len(norm)) if norm else None,
            "partial_snapshot_rate": ((len(norm) - complete) / len(norm)) if norm else None,
            "revision_count": revs, "fetch_error_count": 0, "schema_error_count": 0,
            "lag_seconds": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--store", default=DEFAULT_STORE)
    args = ap.parse_args()
    report = seed_symbol(args.symbol, args.store)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
