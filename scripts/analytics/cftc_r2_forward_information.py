# scripts/analytics/cftc_r2_forward_information.py
# -*- coding: utf-8 -*-
"""
R2 — Estudo de informação futura CFTC (PESQUISA, não backtest de estratégia).

Pergunta: features CFTC do dataset R1 têm informação estatística sobre
retornos FUTUROS de BTC/ETH?

Regras anti-look-ahead (asserts em código + testes):
  - COT nunca alinhado ao retorno a partir da terça asof;
  - entry = primeiro fechamento diário ESTRITAMENTE após available_at;
  - feature_timestamp <= assumed_available_at < entry < exit;
  - percentis rolling só com passado+presente (dataset R1);
  - sem shift(-1), sem backfill, sem merge_asof forward (bisect manual).

Produção intocada. Flag CFTC permanece False (nem lida aqui).
Sem preço: sem estudo — sem Binance positioning, sem estratégia, sem sizing.
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import logging
import math
import os
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

import aiohttp
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CftcR2")

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = timezone.utc

R2_SCHEMA_VERSION = 1
HORIZONS = (1, 3, 7, 14, 28)
CONTRACT_TO_SPOT = {"133741": "BTCUSDT", "133742": "BTCUSDT",
                    "146021": "ETHUSDT", "146022": "ETHUSDT"}
CATEGORIES = ("dealer", "asset_manager", "leveraged", "other", "nonreportable")
FEATURES = ("net_share_oi", "net_change_1w", "net_change_4w",
            "net_share_change_1w", "net_share_change_4w",
            "net_share_pct26", "net_share_pct52", "net_share_pct156")
SUBPERIODS = (("2018-2020", "2018-01-01", "2020-12-31"),
              ("2021-2023", "2021-01-01", "2023-12-31"),
              ("2024-2026", "2024-01-01", "2026-12-31"))
MIN_N_QUINTILE = 50
MIN_N_BUCKET = 10
BOOTSTRAP_B = 500
BOOTSTRAP_SEED = 20260915


# ---------- feriados federais US (regra observada) ---------------------------

def _nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    shift = (d.weekday() - weekday) % 7
    return d - timedelta(days=shift)


def us_federal_holidays(year: int) -> set:
    days = {date(year, 1, 1), _nth_weekday(year, 1, 0, 3),
            _nth_weekday(year, 2, 0, 3), _last_weekday(year, 5, 0),
            date(year, 6, 19), date(year, 7, 4), _nth_weekday(year, 9, 0, 1),
            _nth_weekday(year, 10, 0, 2), date(year, 11, 11),
            _nth_weekday(year, 11, 3, 4), date(year, 12, 25)}
    observed = set()
    for d in days:
        if d.weekday() == 5:
            observed.add(d - timedelta(days=1))
        elif d.weekday() == 6:
            observed.add(d + timedelta(days=1))
        else:
            observed.add(d)
    return observed


def _et1530_utc(day: date) -> datetime:
    return datetime.combine(day, time(15, 30), tzinfo=_ET).astimezone(timezone.utc)


def availability_scenarios(asof: date, holidays: set) -> dict:
    """A/B/C. A estimado quando sexta é feriado ou asof não é terça."""
    days_to_friday = (4 - asof.weekday()) % 7
    friday = asof + timedelta(days=days_to_friday)
    a_at = _et1530_utc(friday)
    a_estimated = asof.weekday() != 1 or friday in holidays
    monday = friday + timedelta(days=3)
    tuesday = friday + timedelta(days=4)
    return {
        "A": (a_at, a_estimated),
        "B": (_et1530_utc(monday), False),
        "C": (_et1530_utc(tuesday), False),
    }


# ---------- preços (Binance USD-M Futures, diário) ----------------------------

async def fetch_klines(symbol: str, start_ms: int) -> list:
    url = "https://fapi.binance.com/fapi/v1/klines"
    out, end = [], None
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = start_ms
        while True:
            params = {"symbol": symbol, "interval": "1d", "startTime": start,
                      "limit": 1000}
            if end:
                params["endTime"] = end
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"klines {symbol} HTTP {resp.status}")
                batch = await resp.json()
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 1000:
                break
            start = batch[-1][0] + 1
            await asyncio.sleep(0.3)
    return out


def load_prices(price_dir: Path, symbol: str, do_fetch: bool) -> pd.DataFrame:
    price_dir.mkdir(parents=True, exist_ok=True)
    path = price_dir / f"{symbol}_1d.json"
    if do_fetch:
        start_ms = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        klines = asyncio.run(fetch_klines(symbol, start_ms))
        payload = {"provenance": {
            "source": "binance_usdm_futures_klines_1d",
            "symbol": symbol, "interval": "1d",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "n_candles": len(klines)}, "klines": klines}
        path.write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"date": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).date().isoformat(),
             "close": float(k[4])} for k in payload["klines"]]
    df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    gaps = [df["date"][i] for i in range(1, len(df))
            if (date.fromisoformat(df["date"][i]) - date.fromisoformat(df["date"][i - 1])).days != 1]
    payload["provenance"]["coverage"] = {
        "first": df["date"].iloc[0] if len(df) else None,
        "last": df["date"].iloc[-1] if len(df) else None,
        "n_days": len(df), "n_gaps": len(gaps), "gaps": gaps[:20]}
    path.write_text(json.dumps({**payload, "coverage_computed": True}), encoding="utf-8")
    return df


# ---------- alinhamento feature -> target (sem leakage) -----------------------

def build_rows(feats: pd.DataFrame, prices: pd.DataFrame,
               horizon: int, holidays: set) -> pd.DataFrame:
    """Uma linha por (asof, availability). Asserts anti-leakage embutidos."""
    price_dates = list(prices["date"])
    close_of = dict(zip(prices["date"], prices["close"]))
    rows = []
    for _, f in feats.iterrows():
        asof = date.fromisoformat(f["report_as_of_date"])
        scen = availability_scenarios(asof, holidays)
        for assumption, (avail_at, estimated) in scen.items():
            assert avail_at.tzinfo is not None
            # entry: primeiro fechamento ESTRITAMENTE após available_at
            avail_date = avail_at.date().isoformat()
            i = bisect.bisect_right(price_dates, avail_date)
            if i >= len(price_dates):
                continue
            entry_date = price_dates[i]
            exit_date = (date.fromisoformat(entry_date) + timedelta(days=horizon)).isoformat()
            if exit_date not in close_of:
                continue
            entry_ts = datetime.combine(date.fromisoformat(entry_date),
                                        time(0, 0), tzinfo=timezone.utc)
            exit_ts = datetime.combine(date.fromisoformat(exit_date),
                                       time(0, 0), tzinfo=timezone.utc)
            assert entry_ts > avail_at, "entry deve ser após availability"
            assert exit_ts > entry_ts
            assert asof.isoformat() <= avail_at.date().isoformat(), \
                "feature (asof) deve preceder availability"
            ret = close_of[exit_date] / close_of[entry_date] - 1.0
            if not math.isfinite(ret):
                continue
            rows.append({"asof": f["report_as_of_date"], "availability": assumption,
                         "assumed_available_at": avail_at.isoformat(),
                         "availability_estimated": estimated,
                         "entry_date": entry_date, "exit_date": exit_date,
                         "horizon": horizon, "forward_return": ret,
                         "feature_row": f.to_dict()})
    return pd.DataFrame(rows)


# ---------- estatística --------------------------------------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (nan-aware)."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    m = mask.sum()
    if m == 0:
        return q
    order = np.argsort(p[mask])
    ranked = p[mask][order]
    qv = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    q[mask] = np.minimum(qv[np.argsort(order)], 1.0)
    return q


def ic_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    out = {"n": n, "pearson": None, "pearson_p": None,
           "spearman": None, "spearman_p": None}
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return out
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    out.update({"pearson": float(pr.statistic), "pearson_p": float(pr.pvalue),
                "spearman": float(sr.statistic), "spearman_p": float(sr.pvalue)})
    return out


def quintile_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    out = {"n": len(x), "q": [], "q5_q1": None}
    if len(x) < MIN_N_QUINTILE:
        return out
    try:
        qs = pd.qcut(x, 5, labels=False, duplicates="drop")
    except (ValueError, TypeError):
        return out
    if len(np.unique(qs)) < 5:
        return out
    means = []
    for q in range(5):
        yy = y[qs == q]
        out["q"].append({"n": len(yy), "mean": float(np.mean(yy)),
                         "median": float(np.median(yy)),
                         "std": float(np.std(yy, ddof=1)) if len(yy) > 1 else None,
                         "pos_rate": float(np.mean(yy > 0))})
        means.append(np.mean(yy))
    out["q5_q1"] = float(means[4] - means[0])
    return out


def extreme_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Buckets <=10 / 10-25 / 25-75 / 75-90 / >=90 sobre a feature."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    bounds = [10, 25, 75, 90]
    out = {"n": len(x), "buckets": []}
    if len(x) < MIN_N_BUCKET * 2:
        return out
    qs = np.percentile(x, bounds)
    edges = [("le10", -np.inf, qs[0]), ("p10_25", qs[0], qs[1]),
             ("p25_75", qs[1], qs[2]), ("p75_90", qs[2], qs[3]),
             ("ge90", qs[3], np.inf)]
    for name, lo, hi in edges:
        if name == "le10":
            yy = y[x <= hi]
        elif name == "ge90":
            yy = y[x >= lo]
        else:
            yy = y[(x > lo) & (x <= hi)]
        out["buckets"].append({"bucket": name, "n": len(yy),
                               "mean": float(np.mean(yy)) if len(yy) else None,
                               "median": float(np.median(yy)) if len(yy) else None})
    return out


def block_bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_fn,
                       block: int, b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED):
    """Bootstrap em blocos temporais (sem shuffle). Retorna IC 95%."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3 * block:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    vals = []
    for _ in range(b):
        idx = np.concatenate([np.arange(s, min(s + block, n))
                              for s in rng.integers(0, n, n_blocks)])[:n]
        try:
            v = stat_fn(x[idx], y[idx])
        except (ValueError, TypeError, ZeroDivisionError):
            continue
        if v is not None and math.isfinite(v):
            vals.append(v)
    if len(vals) < 100:
        return None
    return {"lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)), "b": len(vals)}


def _pearson_only(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(stats.pearsonr(x, y).statistic)


# ---------- classificação ------------------------------------------------------

def classify(ev: dict) -> str:
    """Triage de pesquisa (não é sinal). Regras documentadas, sem threshold mágico
    de trading: exigem robustez entre availability, subperíodos e multiple testing."""
    n = ev.get("n", 0)
    if n < MIN_N_QUINTILE:
        return "INSUFFICIENT_SAMPLE"
    signs = [s for s in ev.get("subperiod_signs", {}).values() if s != 0]
    avail_signs = [s for s in ev.get("availability_signs", {}).values() if s != 0]
    stable_sub = len(signs) >= 2 and len(set(signs)) == 1
    stable_avail = len(avail_signs) >= 2 and len(set(avail_signs)) == 1
    ci = ev.get("ic_bootstrap_ci") or {}
    ci_excludes_zero = (ci and (ci["hi"] < 0 or ci["lo"] > 0))
    q_ok = (ev.get("q_value") is not None and ev["q_value"] < 0.10)
    multi_h = ev.get("multi_horizon_same_sign", False)
    if stable_sub and stable_avail and ci_excludes_zero and q_ok and multi_h and n >= 100:
        return "PROMISING_RESEARCH_SIGNAL"
    if stable_sub and stable_avail:
        return "WEAK_STABLE"
    if signs and len(set(signs)) > 1:
        return "WEAK_UNSTABLE"
    return "NO_EVIDENCE"


def analyze_combo(feat_df: pd.DataFrame, prices: pd.DataFrame, holidays: set,
                  contract: str, category: str, feature: str, horizon: int) -> dict:
    # mapeia nomes R2.4 -> colunas R1
    if feature == "net_share_oi":
        col = f"{category}_net_share_oi"
    elif feature.startswith("percentile_"):
        col = f"{category}_net_share_pct{feature.split('_')[1]}"
    else:  # net_change_1w/4w, net_share_change_1w/4w
        col = f"{category}_{feature}"
    rows = build_rows(feat_df, prices, horizon, holidays)
    ev = {"contract": contract, "category": category, "feature": feature,
          "horizon": horizon, "n": 0}
    if rows.empty:
        return ev

    def _fval(r) -> float:
        v = r["feature_row"].get(col)
        return v if isinstance(v, (int, float)) else np.nan

    by_avail = {}
    for assumption in ("A", "B", "C"):
        sub = rows[rows["availability"] == assumption]
        xs = np.array([_fval(r) for _, r in sub.iterrows()], dtype=float)
        ys = sub["forward_return"].to_numpy(dtype=float)
        by_avail[assumption] = (xs, ys)
    # usa A para IC principal (B/C para robustez de sinal)
    xa, ya = by_avail["A"]
    st = ic_stats(xa, ya)
    ev.update(st)
    ev["availability_signs"] = {
        a: int(np.sign(np.nanmean(
            (v[0] - np.nanmean(v[0])) * (v[1] - np.nanmean(v[1])))))
        if len(v[0]) >= 10 else 0 for a, v in by_avail.items()}
    ev["quintiles"] = quintile_stats(xa, ya)
    ev["extremes"] = extreme_stats(xa, ya)
    # subperíodos (cenário A)
    ev["subperiod_signs"] = {}
    for name, start, end in SUBPERIODS:
        m = rows[(rows["availability"] == "A") & (rows["asof"] >= start)
                 & (rows["asof"] <= end)]
        if len(m) < 30:
            continue
        xs = np.array([_fval(r) for _, r in m.iterrows()], dtype=float)
        s = ic_stats(xs, m["forward_return"].to_numpy(dtype=float))
        if s["pearson"] is not None:
            ev["subperiod_signs"][name] = int(np.sign(s["pearson"]))
    block = 4 if horizon <= 3 else 8
    ev["ic_bootstrap_ci"] = block_bootstrap_ci(xa, ya, _pearson_only, block)
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dados/research/cftc")
    ap.add_argument("--out", default="dados/research/cftc/r2")
    ap.add_argument("--summary-out",
                    default="analysis/results/cftc_r2_information_summary.json")
    ap.add_argument("--no-fetch-prices", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    price_dir = Path(args.dataset) / "prices"
    norm_dir = Path(args.dataset) / "normalized"

    holidays = set()
    for yr in range(2017, 2028):
        holidays |= {d.isoformat() for d in us_federal_holidays(yr)}

    price_frames = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        price_frames[symbol] = load_prices(price_dir, symbol,
                                           do_fetch=not args.no_fetch_prices)

    combos = []
    feat_cache = {}
    for code in ("133741", "133742", "146021", "146022"):
        pq = norm_dir / f"{code}_features.parquet"
        if not pq.exists():
            logger.warning("sem features para %s, pulando", code)
            continue
        feat_cache[code] = pd.read_parquet(pq)

    for code, feat_df in feat_cache.items():
        prices = price_frames[CONTRACT_TO_SPOT[code]]
        for category in CATEGORIES:
            for feature in FEATURES:
                for horizon in HORIZONS:
                    ev = analyze_combo(feat_df, prices, holidays, code,
                                       category, feature, horizon)
                    combos.append(ev)

    # FDR sobre Pearson p-values de todos os combos
    pvals = np.array([c.get("pearson_p", np.nan) for c in combos])
    qvals = bh_fdr(pvals)
    for c, q in zip(combos, qvals):
        c["q_value"] = None if (q is None or not math.isfinite(q)) else float(q)

    # estabilidade multi-horizonte (mesmo sinal em >=3 horizontes, cenário A)
    from collections import defaultdict
    by_key = defaultdict(list)
    for c in combos:
        if c.get("pearson") is not None:
            by_key[(c["contract"], c["category"], c["feature"])].append(
                int(np.sign(c["pearson"])))
    for c in combos:
        signs = by_key[(c["contract"], c["category"], c["feature"])]
        c["multi_horizon_same_sign"] = len(signs) >= 3 and len(set(signs)) == 1

    for c in combos:
        c["classification"] = classify(c)

    full = pd.DataFrame([{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
                          for k, v in c.items()} for c in combos])
    full_path = out_dir / "cftc_r2_full_results.parquet"
    full.to_parquet(full_path, index=False)

    # resumo: top por robustez (não por |correlação|)
    rank = {"PROMISING_RESEARCH_SIGNAL": 0, "WEAK_STABLE": 1,
            "WEAK_UNSTABLE": 2, "NO_EVIDENCE": 3, "INSUFFICIENT_SAMPLE": 4}
    top = sorted(combos, key=lambda c: (rank.get(c["classification"], 9),
                                        -(abs(c["pearson"]) if c.get("pearson") is not None else -1)))[:25]
    counts: dict = {}
    for c in combos:
        counts[c["classification"]] = counts.get(c["classification"], 0) + 1
    summary = {
        "schema_version": R2_SCHEMA_VERSION,
        "research_history": True, "point_in_time": False,
        "note": ("availability A=Friday-15:30ET(estimado se feriado/asof-atípico), "
                 "B=segunda, C=terça; conclusões não dependem só de A; "
                 "bootstrap em blocos temporais; FDR BH; sem cruzamento com preço além de targets"),
        "n_combos": len(combos),
        "class_counts": counts,
        "top_by_robustness": [
            {k: c.get(k) for k in ("contract", "category", "feature", "horizon",
                                   "n", "pearson", "spearman", "q_value",
                                   "classification")}
            for c in top],
    }
    spath = Path(args.summary_out)
    spath.parent.mkdir(parents=True, exist_ok=True)
    spath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("combos=%d classes=%s", len(combos), counts)
    print(json.dumps({"n_combos": len(combos), "class_counts": counts,
                      "full_results": str(full_path), "summary": str(spath)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
