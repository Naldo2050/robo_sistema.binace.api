# scripts/analytics/cftc_r4_prospective_collector.py
# -*- coding: utf-8 -*-
"""
R4 — Coletor prospectivo TRUE OUT-OF-SAMPLE (H3 primária, H4 diagnóstica).

Regras congeladas no manifesto (config/cftc_r4_manifest.json):
  - somente relatórios com asof >= freeze_date (pré-freeze rejeitado);
  - first_seen_at real obrigatório; calendar fallback PROIBIDO;
  - entry = primeiro candle diário com open_time ESTRITO após first_seen;
  - outcome PENDING até exit >= entry + 28d;
  - revisão nunca reescreve a observação original;
  - append-only (duplicata rejeitada); restart-safe via reload;
  - sem threshold, sem sinal, sem BUY/SELL.

Uso:
  python scripts/analytics/cftc_r4_prospective_collector.py [--store DIR] [--no-binance]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

import aiohttp

from fetchers.cftc_cot_fetcher import CftcCotFetcher, _parse_asof, _safe_int

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CftcR4")

MANIFEST_PATH = Path("config/cftc_r4_manifest.json")
DEFAULT_STORE = Path("dados/research/cftc/r4_prospective")
H3_CODE = "133742"
H3_LONG, H3_SHORT, H3_OI = ("lev_money_positions_long",
                            "lev_money_positions_short", "open_interest_all")
H4_WINDOW = 52
MATURITY_DAYS = 28


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def observation_id(code: str, asof: str, revision: int) -> str:
    return f"{code}_{asof}_r{revision}"


def _num_strict(raw, field: str, problems: list):
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


def h3_net_share(raw: dict, problems: list):
    long = _num_strict(raw.get(H3_LONG), H3_LONG, problems)
    short = _num_strict(raw.get(H3_SHORT), H3_SHORT, problems)
    oi = _num_strict(raw.get(H3_OI), H3_OI, problems)
    if long is None or short is None or oi is None or oi == 0:
        if oi == 0:
            problems.append("oi_zero")
        return None
    return (long - short) / oi


def build_observation(raw: dict, first_seen_at: str, retrieved_at: str,
                      revision: int, freeze_date: str,
                      trailing_net_shares: list) -> dict:
    """Observação pura (sem I/O). Retorna dict com status ok|invalid|pre_freeze."""
    problems: list = []
    if not first_seen_at:
        return {"status": "invalid", "reason": "first_seen_required"}
    asof = _parse_asof(raw.get("report_date_as_yyyy_mm_dd"))
    if asof is None:
        return {"status": "invalid", "reason": "bad_asof"}
    if asof.isoformat() < freeze_date:
        return {"status": "pre_freeze", "report_as_of_date": asof.isoformat()}
    feat = h3_net_share(raw, problems)
    h4 = None
    h4_flag = None
    if feat is not None:
        window = [v for v in trailing_net_shares if v is not None][- (H4_WINDOW - 1):] + [feat]
        if len(window) < H4_WINDOW:
            h4_flag = "insufficient_history"
        else:
            below = sum(1 for v in window[:-1] if v < window[-1])
            h4 = below / (H4_WINDOW - 1)
    else:
        h4_flag = "h3_unavailable"
    if feat is None:
        return {"status": "invalid", "report_as_of_date": asof.isoformat(),
                "reason": "h3_unavailable", "problems": problems}
    return {
        "status": "ok",
        "observation_id": observation_id(H3_CODE, asof.isoformat(), revision),
        "contract_code": H3_CODE,
        "report_as_of_date": asof.isoformat(),
        "revision": revision,
        "first_seen_at": first_seen_at,
        "retrieved_at": retrieved_at,
        "feature_available_at": first_seen_at,
        "h3_net_share_oi": feat,
        "h4_net_share_pct52": h4,
        "h4_flag": h4_flag,
        "open_interest": _safe_int(raw.get(H3_OI)),
        "leveraged_long": _safe_int(raw.get(H3_LONG)),
        "leveraged_short": _safe_int(raw.get(H3_SHORT)),
        "leveraged_spreading": _safe_int(raw.get("lev_money_positions_spread")),
        "raw_hash": None,  # preenchido pelo chamador (content_hash do fetcher)
        "quality": {"problems": problems},
    }


def compute_entry(candles: list, first_seen_at: str) -> dict | None:
    """candles: [(open_ms, close)]. Entry = primeiro open ESTRITO após first_seen."""
    fs = datetime.fromisoformat(str(first_seen_at).replace("Z", "+00:00"))
    if fs.tzinfo is None:
        return None  # naive rejeitado: ambíguo
    fs_ms = int(fs.timestamp() * 1000)
    for open_ms, close in sorted(candles):
        if open_ms > fs_ms and isinstance(close, (int, float)) and math.isfinite(close):
            return {"entry_timestamp": datetime.fromtimestamp(
                open_ms / 1000, tz=timezone.utc).isoformat(), "entry_price": float(close)}
    return None


def compute_outcome(candles: list, entry_timestamp: str):
    """Exit = primeiro candle com open >= entry + 28d. None => PENDING."""
    entry = datetime.fromisoformat(str(entry_timestamp).replace("Z", "+00:00"))
    need_ms = int((entry + timedelta(days=MATURITY_DAYS)).timestamp() * 1000)
    for open_ms, close in sorted(candles):
        if open_ms >= need_ms and isinstance(close, (int, float)) and math.isfinite(close):
            entry_px = None
            return {"exit_timestamp": datetime.fromtimestamp(
                open_ms / 1000, tz=timezone.utc).isoformat(),
                "exit_price": float(close)}
    return None


class R4Store:
    """Append-only JSONL: observations / outcomes / revisions / binance."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ids = set()
        for line in self._read("observations.jsonl"):
            if isinstance(line.get("observation_id"), str):
                self._ids.add(line["observation_id"])

    def _path(self, name: str) -> Path:
        return self.root / name

    def _read(self, name: str) -> list:
        p = self._path(name)
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def append_observation(self, obs: dict) -> bool:
        """False = duplicata rejeitada (nunca sobrescreve)."""
        oid = obs.get("observation_id")
        if not oid or oid in self._ids:
            return False
        with open(self._path("observations.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obs, ensure_ascii=False) + "\n")
        self._ids.add(oid)
        return True

    def append_outcome(self, outcome: dict) -> None:
        with open(self._path("outcomes.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(outcome, ensure_ascii=False) + "\n")

    def append_revision(self, rev: dict) -> None:
        with open(self._path("revisions.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rev, ensure_ascii=False) + "\n")

    def append_binance(self, row: dict) -> None:
        with open(self._path("binance.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def observations(self) -> list:
        return self._read("observations.jsonl")

    def outcomes(self) -> list:
        return self._read("outcomes.jsonl")

    def revisions(self) -> list:
        return self._read("revisions.jsonl")


async def fetch_klines_daily(symbol: str, start_ms: int) -> list:
    url = "https://fapi.binance.com/fapi/v1/klines"
    out = []
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = start_ms
        while True:
            async with session.get(url, params={"symbol": symbol, "interval": "1d",
                                                "startTime": start, "limit": 1000}) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"klines {symbol} HTTP {resp.status}")
                batch = await resp.json()
            if not batch:
                break
            out.extend([(k[0], float(k[4])) for k in batch])
            if len(batch) < 1000:
                break
            start = batch[-1][0] + 1
            await asyncio.sleep(0.3)
    return out


def collect_once(store_dir: str = str(DEFAULT_STORE), with_binance: bool = True) -> dict:
    manifest = load_manifest()
    freeze = manifest["freeze_date"]
    store = R4Store(Path(store_dir))
    fetcher = CftcCotFetcher()
    summary: dict = {"asof": None, "observation": None, "revision": None,
                     "binance": None, "skipped": None}
    row, err = asyncio.run(fetcher.fetch_latest(H3_CODE))
    now_iso = datetime.now(timezone.utc).isoformat()
    if row is None:
        summary["skipped"] = f"fetch_error:{err}"
        return summary
    record, ing_err = fetcher.ingest_row(H3_CODE, row, now_iso)
    if ing_err or record is None:
        summary["skipped"] = f"invalid:{ing_err}"
        return summary
    asof = record.report_as_of_date
    summary["asof"] = asof
    if asof < freeze:
        summary["skipped"] = "pre_freeze"
        return summary
    # trailing H3 para H4: R1 features + observações R4 já registradas
    trailing: list = []
    try:
        import pandas as pd

        hist = pd.read_parquet("dados/research/cftc/normalized/133742_features.parquet")
        h = hist[hist["report_as_of_date"] < asof].sort_values("report_as_of_date")
        trailing = [float(v) for v in h["leveraged_net_share_oi"].tolist()
                    if v is not None and math.isfinite(v)]
    except Exception as e:  # noqa: BLE001 - H4 vira insufficient_history
        logger.warning("R4 trailing indisponível: %s", e)
    for obs in store.observations():
        if obs.get("report_as_of_date", "") < asof and obs.get("h3_net_share_oi") is not None:
            trailing.append(float(obs["h3_net_share_oi"]))
    obs = build_observation(record.raw, record.first_seen_at, record.retrieved_at,
                            record.revision, freeze, trailing)
    if obs.get("status") != "ok":
        summary["skipped"] = obs.get("status") + ":" + str(obs.get("reason", ""))
        return summary
    obs["raw_hash"] = record.content_hash
    # entry imediato (candle já existente em geral)
    klines = asyncio.run(fetch_klines_daily(
        "BTCUSDT", int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)))
    entry = compute_entry(klines, obs["first_seen_at"])
    if entry is None:
        summary["skipped"] = "no_entry_candle_yet"
        return summary
    obs.update(entry)
    if not store.append_observation(obs):
        # asof já registrado: verifica revisão sem reescrever
        known = [o for o in store.observations()
                 if o.get("report_as_of_date") == asof]
        known_hashes = {o.get("raw_hash") for o in known}
        if record.content_hash not in known_hashes:
            store.append_revision({
                "observation_id": observation_id(H3_CODE, asof, record.revision),
                "report_as_of_date": asof, "revision": record.revision,
                "revision_first_seen": record.first_seen_at,
                "revised_feature": h3_net_share_only(record.raw),
                "original_preserved": True})
            summary["revision"] = "recorded"
        else:
            summary["skipped"] = "duplicate_observation"
        return summary
    summary["observation"] = obs["observation_id"]
    if with_binance:
        summary["binance"] = capture_binance(store, obs)
    return summary


def h3_net_share_only(raw: dict):
    problems: list = []
    return h3_net_share(raw, problems)


def capture_binance(store: R4Store, obs: dict) -> dict:
    """Shadow auxiliar. Estrito: só campos com timestamp <= feature_available_at
    valem para overlap; o resto é contexto com flag (nunca backfill)."""
    from fetchers.binance_positioning_fetcher import BinancePositioningFetcher

    fa = obs["feature_available_at"]
    row: dict = {"observation_id": obs["observation_id"],
                 "captured_at": datetime.now(timezone.utc).isoformat(),
                 "strict_fields": {}, "aux": {}, "quality": []}

    async def _snap():
        fetcher = BinancePositioningFetcher()
        return await fetcher.fetch_positioning("BTCUSDT", force_refresh=True)

    try:
        snap = asyncio.run(_snap())
        d = snap.to_dict()
        # snapshot ao vivo é posterior ao first_seen -> aux, não overlap
        row["aux"]["positioning_snapshot"] = d
        row["quality"].append("aux_contemporaneous_not_point_in_time")
    except Exception as e:  # noqa: BLE001
        row["quality"].append(f"positioning_unavailable:{str(e)[:80]}")
    # funding estrito: último fundingTime <= first_seen
    try:
        fund = asyncio.run(_funding_until(fa))
        if fund is None:
            row["strict_fields"]["funding"] = None
            row["quality"].append("funding_absent_strict")
        else:
            row["strict_fields"]["funding"] = fund
    except Exception as e:  # noqa: BLE001
        row["strict_fields"]["funding"] = None
        row["quality"].append(f"funding_error:{str(e)[:80]}")
    row["strict_fields"]["oi"] = None
    row["quality"].append("oi_history_insufficient")
    store.append_binance(row)
    return {"strict": list(row["strict_fields"]), "quality": row["quality"]}


async def _funding_until(first_seen_at: str):
    fa_ms = int(datetime.fromisoformat(
        str(first_seen_at).replace("Z", "+00:00")).timestamp() * 1000)
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        end = fa_ms - 1
        last = None
        while True:
            async with session.get(
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                    params={"symbol": "BTCUSDT", "limit": 1000,
                            "endTime": end}) as resp:
                if resp.status != 200:
                    return None
                batch = await resp.json()
            if not batch:
                return last
            cands = [(r["fundingTime"], float(r["fundingRate"])) for r in batch
                     if r["fundingTime"] < fa_ms]
            if cands:
                return {"fundingTime": max(c[0] for c in cands),
                        "fundingRate": dict(cands)[max(c[0] for c in cands)]}
            oldest = min(r["fundingTime"] for r in batch)
            if len(batch) < 1000:
                return last
            end = oldest - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    ap.add_argument("--no-binance", action="store_true")
    args = ap.parse_args()
    summary = collect_once(args.store, with_binance=not args.no_binance)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
