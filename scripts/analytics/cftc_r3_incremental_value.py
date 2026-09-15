# scripts/analytics/cftc_r3_incremental_value.py
# -*- coding: utf-8 -*-
"""
R3 — Valor incremental CFTC sobre o baseline (PESQUISA, PSEUDO_OUT_OF_SAMPLE).

Hipóteses CONGELADAS do R2 (não trocar categoria/feature/horizon/sinal):
  H1: 133741 other/net_share_oi h28 +
  H2: 133741 nonreportable/net_share_change_1w h3 -
  H3: 133742 leveraged/net_share_oi h28 +
  H4: 133742 leveraged/net_share_pct52 h28 +

Split: dev asof<=2023-12-31, teste asof>=2024-01-01. Como o R2 selecionou
H1-H4 usando 2024-2026, o holdout é PSEUDO_OUT_OF_SAMPLE (documentado).

Baseline price (trailing, < entry): ret_1d/7d/28d, vol realizada 28d,
distância EMA50. Derivativos: funding (histórico total) e OI-change
(cobertura recente ~500d). Binance positioning: SEM histórico (tabela
inexistente) -> excluído e documentado (R3-B parcial por construção).

Modelos OLS/HAC + logística secundária. Sem XGBoost, sem busca.
Nenhum resultado autoriza produção.
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import aiohttp
import numpy as np
import pandas as pd

import cftc_r2_forward_information as r2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CftcR3")

R3_SCHEMA_VERSION = 1
DEV_END = "2023-12-31"
HYPOTHESES = {
    "H1": {"contract": "133741", "category": "other", "feature": "net_share_oi",
           "horizon": 28, "sign": +1, "r1col": "other_net_share_oi"},
    "H2": {"contract": "133741", "category": "nonreportable", "feature": "net_share_change_1w",
           "horizon": 3, "sign": -1, "r1col": "nonreportable_net_change_1w"},
    "H3": {"contract": "133742", "category": "leveraged", "feature": "net_share_oi",
           "horizon": 28, "sign": +1, "r1col": "leveraged_net_share_oi"},
    "H4": {"contract": "133742", "category": "leveraged", "feature": "net_share_pct52",
           "horizon": 28, "sign": +1, "r1col": "leveraged_net_share_pct52"},
}
BASELINE_COLS = ["ret_1d", "ret_7d", "ret_28d", "realized_vol_28d", "ema50_dist"]
DERIV_COLS = ["funding_last", "funding_mean_7d"]


# ---------- funding + OI history (Binance público) ----------------------------

async def _paged(url: str, params: dict) -> list:
    out = []
    start = params.pop("startTime", None)
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            p = dict(params)
            if start is not None:
                p["startTime"] = start
            async with session.get(url, params=p) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"{url} HTTP {resp.status}")
                batch = await resp.json()
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 1000:
                break
            key = "fundingTime" if "fundingTime" in batch[0] else "timestamp"
            start = batch[-1][key] + 1
            await asyncio.sleep(0.3)
    return out


async def _paged_backward(url: str, params: dict, time_key: str,
                          cutoff_ms: int) -> list:
    """Paginação regressiva (para endpoints que retornam só o recente).

    fundingRate com startTime=0 devolve as 500 mais recentes; anda endTime
    para trás até esvaziar ou cruzar cutoff_ms.
    """
    out = []
    end = None
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            p = dict(params)
            if end is not None:
                p["endTime"] = end
            async with session.get(url, params=p) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"{url} HTTP {resp.status}")
                batch = await resp.json()
            if not batch:
                break
            out.extend(batch)
            oldest = min(r[time_key] for r in batch)
            if oldest <= cutoff_ms or len(batch) < 2:
                break
            end = oldest - 1
            await asyncio.sleep(0.3)
    return out


def load_funding(price_dir: Path, symbol: str, do_fetch: bool):
    path = price_dir / f"{symbol}_funding.json"
    if do_fetch:
        cutoff = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        rows = asyncio.run(_paged_backward(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": 1000}, "fundingTime", cutoff))
        payload = {"provenance": {
            "source": "binance_usdm_fundingRate", "symbol": symbol,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": len(rows)}, "rows": rows}
        path.write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recs = sorted(((r["fundingTime"], float(r["fundingRate"])) for r in payload["rows"]))
    return recs, payload["provenance"]


def load_oi_daily(price_dir: Path, symbol: str, do_fetch: bool):
    """openInterestHist 1d.

    COBERTURA LIMITADA (verificado 2026-09-15): o endpoint devolve ~31 dias
    independente de limit=500. Insuficiente para modelagem histórica ->
    EXCLUÍDO dos modelos (documentado); mantido só o registro bruto.
    """
    path = price_dir / f"{symbol}_oi_1d.json"
    if do_fetch:
        rows = asyncio.run(_paged(
            "https://fapi.binance.com/futures/data/openInterestHist",
            {"symbol": symbol, "period": "1d", "limit": 500}))
        payload = {"provenance": {
            "source": "binance_usdm_openInterestHist_1d", "symbol": symbol,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": len(rows),
            "note": ("endpoint retorna somente ~31 dias recentes: cobertura "
                     "insuficiente para modelos; excluído, sem preenchimento")},
            "rows": rows}
        path.write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recs = sorted(((r["timestamp"], float(r["sumOpenInterest"])) for r in payload["rows"]))
    return recs, payload["provenance"]


# ---------- baseline trailing (tudo < entry) -----------------------------------

def attach_baseline(rows: pd.DataFrame, prices: pd.DataFrame,
                    funding: list, oi_hist: list) -> pd.DataFrame:
    closes = dict(zip(prices["date"], prices["close"]))
    pdates = sorted(closes)
    # EMA50 trailing completa (passado apenas por construção incremental)
    ema = {}
    k = 2.0 / (50 + 1)
    e = None
    for d in pdates:
        e = closes[d] if e is None else closes[d] * k + e * (1 - k)
        ema[d] = e
    logrets = {}
    for i in range(1, len(pdates)):
        logrets[pdates[i]] = math.log(closes[pdates[i]] / closes[pdates[i - 1]])
    f_times = [t for t, _ in funding]
    o_times = [t for t, _ in oi_hist]
    out = []
    for _, r in rows.iterrows():
        entry = r["entry_date"]
        i = bisect.bisect_left(pdates, entry)
        if i < 29:  # histórico insuficiente p/ vol 28d
            continue
        hist = pdates[:i]  # datas < entry (estrito)
        c1 = closes[hist[-1]]
        b = {"ret_1d": c1 / closes[hist[-2]] - 1.0,
             "ret_7d": c1 / closes[hist[-8]] - 1.0 if len(hist) >= 8 else None,
             "ret_28d": c1 / closes[hist[-29]] - 1.0 if len(hist) >= 29 else None,
             "realized_vol_28d": float(np.std([logrets[d] for d in hist[-28:]], ddof=1)),
             "ema50_dist": c1 / ema[hist[-1]] - 1.0}
        entry_ms = int(datetime.combine(date.fromisoformat(entry),
                                        datetime.min.time(),
                                        tzinfo=timezone.utc).timestamp() * 1000)
        j = bisect.bisect_left(f_times, entry_ms)  # fundingTime < entry
        if j == 0:
            b["funding_last"], b["funding_mean_7d"] = None, None
        else:
            vals = [v for _, v in funding[max(0, j - 21):j]]
            b["funding_last"] = funding[j - 1][1]
            b["funding_mean_7d"] = float(np.mean(vals)) if vals else None
        k2 = bisect.bisect_left(o_times, entry_ms)
        if k2 == 0:
            b["oi_level"], b["oi_change_7d"] = None, None
        else:
            b["oi_level"] = oi_hist[k2 - 1][1]
            b["oi_change_7d"] = (oi_hist[k2 - 1][1] / oi_hist[k2 - 8][1] - 1.0
                                 if k2 >= 8 else None)
        for key in ("ret_1d", "ret_7d", "ret_28d", "realized_vol_28d",
                    "ema50_dist", "funding_last", "funding_mean_7d",
                    "oi_level", "oi_change_7d"):
            assert b[key] is None or math.isfinite(b[key]), key
        out.append({**r.to_dict(), **b})
    df = pd.DataFrame(out)
    # freshness máxima: baseline diário exige entry-1d existente (por construção)
    return df


# ---------- OLS/HAC -------------------------------------------------------------

def ols_fit(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    sse = float(resid @ resid)
    return beta, resid, sse, n, p


def hac_se(X: np.ndarray, resid: np.ndarray, bandwidth: int | None = None):
    """Newey-West (Bartlett). Retorna SE por coeficiente + p bicaudal normal."""
    from scipy.stats import norm

    n, p = X.shape
    if bandwidth is None:
        bandwidth = int(math.floor(4 * (n / 100) ** (2 / 9)))
    e = resid.reshape(-1, 1)
    Xe = X * e
    S = Xe.T @ Xe
    for lag in range(1, bandwidth + 1):
        w = 1 - lag / (bandwidth + 1)
        G = Xe[lag:].T @ Xe[:-lag]
        S = S + w * (G + G.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0))
    from scipy.stats import norm as _norm  # noqa
    return se


def fit_models(df: pd.DataFrame, cftc_col: str, deriv_cols: list):
    """M0/M1/B/D em dev. Retorna dict com coefs, HAC, AIC/BIC, OOS handled fora."""
    res = {}
    base = [c for c in BASELINE_COLS]
    cols_m0 = ["const"] + base
    cols_m1 = cols_m0 + ["cftc"]
    cols_b = cols_m0 + [c for c in deriv_cols]
    cols_d = cols_b + ["cftc"]
    for tag, cols in (("M0", cols_m0), ("M1", cols_m1), ("B", cols_b), ("D", cols_d)):
        use = [c for c in cols if c != "const" and c != "cftc"]
        frame_cols = ([c for c in base if c in cols]
                      + [c for c in deriv_cols if c in cols]
                      + (["cftc"] if "cftc" in cols else []))
        sub = df[["target"] + frame_cols].dropna()
        if len(sub) < 30 or sub.shape[1] - 1 == 0:
            res[tag] = {"n": len(sub), "error": "insufficient"}
            continue
        X = np.column_stack([np.ones(len(sub))] +
                            [sub[c].to_numpy(float) for c in frame_cols])
        y = sub["target"].to_numpy(float)
        try:
            beta, resid, sse, n, p = ols_fit(X, y)
        except np.linalg.LinAlgError:
            res[tag] = {"n": len(sub), "error": "singular"}
            continue
        se = hac_se(X, resid)
        from scipy.stats import norm

        z = beta / np.where(se > 0, se, np.nan)
        pvals = 2 * norm.sf(np.abs(z))
        k = len(beta)
        aic = n * math.log(sse / n) + 2 * k if sse > 0 else None
        bic = n * math.log(sse / n) + k * math.log(n) if sse > 0 else None
        res[tag] = {"n": n, "frame_cols": frame_cols, "beta": beta.tolist(),
                    "se": se.tolist(), "p": pvals.tolist(), "sse": sse,
                    "aic": aic, "bic": bic,
                    "cftc_beta": float(beta[-1]) if "cftc" in cols else None,
                    "cftc_se": float(se[-1]) if "cftc" in cols else None,
                    "cftc_p": float(pvals[-1]) if "cftc" in cols else None}
    return res


def oos_eval(fit: dict, df_test: pd.DataFrame, cftc_col: str, deriv_cols: list):
    """R² OOS vs média do teste + direction accuracy, por modelo."""
    out = {}
    y = df_test["target"].to_numpy(float)
    ybar = float(np.mean(y))
    sst = float(((y - ybar) ** 2).sum())
    base = [c for c in BASELINE_COLS]
    spec = {"M0": base, "M1": base + ["cftc"],
            "B": base + deriv_cols, "D": base + deriv_cols + ["cftc"]}
    for tag, cols in spec.items():
        f = fit.get(tag)
        if not f or "beta" not in f:
            out[tag] = {"error": f.get("error", "no-fit") if f else "no-fit"}
            continue
        frame_cols = f["frame_cols"]
        sub = df_test[["target"] + frame_cols].dropna()
        if len(sub) == 0:
            out[tag] = {"error": "no-test-rows"}
            continue
        X = np.column_stack([np.ones(len(sub))] +
                            [sub[c].to_numpy(float) for c in frame_cols])
        pred = X @ np.array(f["beta"])
        yt = sub["target"].to_numpy(float)
        sse = float(((yt - pred) ** 2).sum())
        sst_t = float(((yt - float(np.mean(yt))) ** 2).sum())
        out[tag] = {"n": len(sub),
                    "r2_oos": 1 - sse / sst_t if sst_t > 0 else None,
                    "dir_acc": float(np.mean(np.sign(pred) == np.sign(yt)))}
    return out


def partial_corr(df: pd.DataFrame, target: str, feat: str, controls: list):
    """Correlação parcial via resíduos OLS (sem intercepto extra além de const)."""
    cols = ["target", feat] + controls
    sub = df[cols].dropna()
    if len(sub) < 10:
        return None, len(sub)
    y = sub["target"].to_numpy(float)
    x = sub[feat].to_numpy(float)
    C = np.column_stack([np.ones(len(sub))] +
                        [sub[c].to_numpy(float) for c in controls])
    by, *_ = np.linalg.lstsq(C, y, rcond=None)
    bx, *_ = np.linalg.lstsq(C, x, rcond=None)
    ry, rx = y - C @ by, x - C @ bx
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None, len(sub)
    return float(np.corrcoef(rx, ry)[0, 1]), len(sub)


def vif_table(df: pd.DataFrame, cols: list) -> dict:
    out = {}
    sub = df[cols].dropna()
    if len(sub) < 10:
        return {c: None for c in cols}
    for c in cols:
        others = [o for o in cols if o != c]
        X = np.column_stack([np.ones(len(sub))] +
                            [sub[o].to_numpy(float) for o in others])
        y = sub[c].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss_res = float(((y - X @ beta) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        out[c] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    return out


def _jsonable(v):
    """Conversão estrita para JSON (sem default=str silencioso)."""
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dados/research/cftc")
    ap.add_argument("--out", default="dados/research/cftc/r3")
    ap.add_argument("--summary-out",
                    default="analysis/results/cftc_r3_incremental_summary.json")
    ap.add_argument("--no-fetch-prices", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    price_dir = Path(args.dataset) / "prices"
    norm_dir = Path(args.dataset) / "normalized"
    holidays = set()
    for yr in range(2017, 2028):
        holidays |= {d.isoformat() for d in r2.us_federal_holidays(yr)}
    prices = {s: r2.load_prices(price_dir, s, do_fetch=not args.no_fetch_prices)
              for s in ("BTCUSDT", "ETHUSDT")}
    funding = {}
    oi_hist = {}
    for sym in ("BTCUSDT",):
        funding[sym], fprov = load_funding(price_dir, sym, do_fetch=not args.no_fetch_prices)
        oi_hist[sym], oprov = load_oi_daily(price_dir, sym, do_fetch=not args.no_fetch_prices)
        logger.info("%s funding=%d oi_days=%d", sym, len(funding[sym]), len(oi_hist[sym]))

    results = {"schema_version": R3_SCHEMA_VERSION,
               "oos_class": "PSEUDO_OUT_OF_SAMPLE",
               "oos_note": ("R2 selecionou H1-H4 usando 2024-2026; holdout 2024+ "
                            "NÃO é untouched. Nenhum parâmetro recalibrado no holdout."),
               "hypotheses": {}}
    p_confirm = []
    dev_betas = {}
    for h, spec in HYPOTHESES.items():
        code = spec["contract"]
        symbol = r2.CONTRACT_TO_SPOT[code]
        feat_df = pd.read_parquet(norm_dir / f"{code}_features.parquet")
        rows = r2.build_rows(feat_df, prices[symbol], spec["horizon"], holidays)
        rows = rows[rows["availability"] == "A"]
        full = attach_baseline(rows, prices[symbol], funding[symbol], oi_hist[symbol])
        full["cftc"] = full["feature_row"].apply(
            lambda fr: fr.get(spec["r1col"]) if isinstance(fr, dict) else None)
        full["target"] = full["forward_return"]
        dev = full[full["asof"] <= DEV_END].reset_index(drop=True)
        test = full[full["asof"] > DEV_END].reset_index(drop=True)
        deriv_cols = [c for c in DERIV_COLS]
        fit = fit_models(dev, spec["r1col"], deriv_cols)
        oos = oos_eval(fit, test, spec["r1col"], deriv_cols)
        controls = BASELINE_COLS + deriv_cols
        pc, pc_n = partial_corr(pd.concat([dev, test]), "target", "cftc",
                                [c for c in controls])
        vif = vif_table(pd.concat([dev, test]),
                        ["cftc"] + [c for c in controls
                                    if c in ("funding_last", "ret_28d", "realized_vol_28d")])
        # A/B/C robustez: direção do coef M1 por availability
        avail_sign = {}
        for av in ("A", "B", "C"):
            sub = r2.build_rows(feat_df, prices[symbol], spec["horizon"], holidays)
            sub = sub[sub["availability"] == av]
            sub = attach_baseline(sub, prices[symbol], funding[symbol], oi_hist[symbol])
            sub["cftc"] = sub["feature_row"].apply(
                lambda fr: fr.get(spec["r1col"]) if isinstance(fr, dict) else None)
            sub["target"] = sub["forward_return"]
            d = sub[sub["asof"] > DEV_END]
            cols = ["cftc"] + BASELINE_COLS
            s2 = d[["target"] + cols].dropna()
            if len(s2) >= 30:
                X = np.column_stack([np.ones(len(s2))] +
                                    [s2[c].to_numpy(float) for c in cols[1:]] + [s2["cftc"].to_numpy(float)])
                beta, *_ = np.linalg.lstsq(X, s2["target"].to_numpy(float), rcond=None)
                avail_sign[av] = float(np.sign(beta[-1])) if beta[-1] != 0 else 0.0
        # regimes no teste (regras pré-fixadas, N>=30)
        regimes = {}
        t = test.dropna(subset=["target", "cftc"] + BASELINE_COLS).reset_index(drop=True)
        if len(t) >= 60:
            bull = t["ret_28d"] > 0
            hiv = t["realized_vol_28d"] > t["realized_vol_28d"].median()
            for name, mask in (("bull", bull), ("bear", ~bull),
                               ("highvol", hiv), ("lowvol", ~hiv),
                               ("pos_funding", t["funding_last"] > 0),
                               ("neg_funding", t["funding_last"] <= 0)):
                tt = t[mask]
                if len(tt) >= 30 and tt["cftc"].std() > 0:
                    regimes[name] = {"n": len(tt),
                                     "ic": float(np.corrcoef(tt["cftc"], tt["target"])[0, 1])}
        # outliers no teste (delta_R2 D-vs-B)
        def _d_b(df_):
            a = oos_eval(fit, df_, spec["r1col"], deriv_cols)
            r2d = (a.get("D", {}) or {}).get("r2_oos")
            r2b = (a.get("B", {}) or {}).get("r2_oos")
            return (r2d - r2b) if (r2d is not None and r2b is not None) else None

        # NB: fit é dev-only; outliers variam só o teste
        out_variants = {"full": None, "winsorized": None, "ex_top1pct": None}
        yt = test["target"].dropna()
        lo, hi = float(yt.quantile(0.01)), float(yt.quantile(0.99))
        tw = test.copy()
        tw["target"] = test["target"].clip(lo, hi)
        out_variants["winsorized"] = _d_b(tw)
        thr = float(test["target"].abs().quantile(0.99))
        out_variants["ex_top1pct"] = _d_b(test[test["target"].abs() <= thr])
        loo = {}
        for yr in sorted({a[:4] for a in test["asof"].tolist()}):
            loo[yr] = _d_b(test[~test["asof"].str.startswith(yr)])
        out_variants["leave_one_year_out"] = loo
        out_variants["full"] = _d_b(test)
        p_hac = (fit.get("M1", {}) or {}).get("cftc_p")
        p_confirm.append(p_hac if p_hac is not None else np.nan)
        dev_betas[h] = (fit.get("M1", {}) or {}).get("cftc_beta")
        results["hypotheses"][h] = {
            "spec": spec, "n_dev": len(dev), "n_test": len(test),
            "fit_M1_cftc_beta": (fit.get("M1", {}) or {}).get("cftc_beta"),
            "fit_M1_cftc_p_hac": p_hac,
            "oos": oos, "partial_corr": pc, "partial_corr_n": pc_n, "vif": vif,
            "availability_sign": avail_sign, "regimes": regimes,
            "outliers_delta_D_minus_B": out_variants,
        }
        logger.info("%s dev=%d test=%d M1beta=%s p=%s oosD-B=%s", h, len(dev),
                    len(test), fit.get("M1", {}).get("cftc_beta"), p_hac,
                    out_variants["full"])

    # Holm sobre 4 p-values confirmatórios (HAC, dev M1)
    from scipy.stats import norm as _norm  # noqa (documenta dependência)

    p = np.array(p_confirm, dtype=float)
    order = np.argsort(np.where(np.isnan(p), 1.0, p))
    holm = {}
    m = 4
    ranked = [(["H1", "H2", "H3", "H4"][i]) for i in order]
    for k, h in enumerate(ranked):
        pv = p[order[k]]
        holm[h] = None if np.isnan(pv) else float(min(1.0, pv * (m - k)))
    results["holm_4_hypotheses"] = holm

    # H3+H4 / H1+H3 complementaridade + interação pré-especificada.
    # Pares podem cruzar contratos (mesmo horizonto, mesmo alvo BTC):
    # merge por asof, sem somar posições.
    results["complementarity"] = {}
    for tag, ha, hb in (("H1_H3", "H1", "H3"), ("H3_H4", "H3", "H4")):
        sa, sb = HYPOTHESES[ha], HYPOTHESES[hb]
        if sa["horizon"] != sb["horizon"]:
            continue
        symbol = r2.CONTRACT_TO_SPOT[sa["contract"]]
        frames = {}
        for hh, ss in ((ha, sa), (hb, sb)):
            feat_df = pd.read_parquet(norm_dir / f"{ss['contract']}_features.parquet")
            rows = r2.build_rows(feat_df, prices[symbol], ss["horizon"], holidays)
            rows = rows[rows["availability"] == "A"]
            full = attach_baseline(rows, prices[symbol], funding[symbol], oi_hist[symbol])
            full["fx_h"] = full["feature_row"].apply(lambda fr: fr.get(ss["r1col"]))
            full["target"] = full["forward_return"]
            frames[hh] = full[["asof", "entry_date", "target", "fx_h"] + BASELINE_COLS + DERIV_COLS]
        m = frames[ha].merge(frames[hb], on=["asof", "entry_date"],
                             suffixes=("_a", "_b"))
        entry = {"n_merged": len(m)}
        if len(m) >= 80:
            m = m.rename(columns={"fx_h_a": "fa", "fx_h_b": "fb",
                                  "target_a": "target"})
            # baseline idêntico nos dois ramos (mesmo entry) — usa _a.
            for c in BASELINE_COLS + DERIV_COLS:
                assert (m[f"{c}_a"].fillna(-9999) == m[f"{c}_b"].fillna(-9999)).all(), c
                m[c] = m[f"{c}_a"]
            m["fx"] = m["fa"] * m["fb"]
            base_cols = [c for c in BASELINE_COLS] + [c for c in DERIV_COLS]
            cols = base_cols + ["fa", "fb", "fx"]
            dev = m[m["asof"] <= DEV_END].reset_index(drop=True)
            test = m[m["asof"] > DEV_END].reset_index(drop=True)
            entry.update({"n_dev": len(dev), "n_test": len(test)})
            sub = dev[["target"] + cols].dropna()
            if len(sub) >= 50:
                X = np.column_stack([np.ones(len(sub))] +
                                    [sub[c].to_numpy(float) for c in cols])
                y = sub["target"].to_numpy(float)
                beta, resid, *_ = ols_fit(X, y)
                se = hac_se(X, resid)
                z = beta / np.where(se > 0, se, np.nan)
                from scipy.stats import norm

                entry.update({"beta_fa": float(beta[-3]),
                              "p_fa": float(2 * norm.sf(abs(z[-3]))),
                              "beta_fb": float(beta[-2]),
                              "p_fb": float(2 * norm.sf(abs(z[-2]))),
                              "beta_fx": float(beta[-1]),
                              "p_fx": float(2 * norm.sf(abs(z[-1])))})
                st = test[["target"] + cols].dropna()
                if len(st) > 0:
                    Xt = np.column_stack([np.ones(len(st))] +
                                         [st[c].to_numpy(float) for c in cols])
                    pred = Xt @ beta
                    yt = st["target"].to_numpy(float)
                    sst = ((yt - yt.mean()) ** 2).sum()
                    entry["r2_oos_joint"] = float(1 - ((yt - pred) ** 2).sum() / sst) if sst > 0 else None
        results["complementarity"][tag] = entry

    # decisão por hipótese (R3.13/R3.14 + Holm R3.12)
    holm_all = results["holm_4_hypotheses"]
    for h, ev in results["hypotheses"].items():
        spec = HYPOTHESES[h]
        ev["classification"], ev["incremental_evidence"] = decide(
            ev, spec["sign"], dev_betas.get(h), holm_all.get(h))

    spath = Path("analysis/results/cftc_r3_incremental_summary.json")
    spath.parent.mkdir(parents=True, exist_ok=True)
    spath.write_text(json.dumps(_jsonable(results), ensure_ascii=False, indent=2),
                     encoding="utf-8")
    full_path = Path("dados/research/cftc/r3/cftc_r3_full.parquet")
    logger.info("summary=%s", spath)
    print(json.dumps(_jsonable(
        {h: {"class": v["classification"], "evidence": v["incremental_evidence"]}
         for h, v in results["hypotheses"].items()}),
        ensure_ascii=False, indent=2))


def decide(ev: dict, spec_sign: int, dev_beta: float | None,
           holm_p: float | None) -> tuple:
    """R3.13/R3.14 + R3.12. ROBUST exige ainda p confirmatório Holm < 0.10
    (mesmo padrão q<0.10 do R2; sem isso, incremento OOS com targets h28
    sobrepostos não sustenta promoção)."""
    oos = ev.get("oos", {})
    n_test = ev.get("n_test", 0)
    if n_test < 50:
        return "INSUFFICIENT_OVERLAP", {"reason": f"n_test={n_test}"}
    # Ablação: A=M0 (price), B=price+deriv, C=M1 (price+CFTC), D=price+deriv+CFTC.
    r2d = (oos.get("D", {}) or {}).get("r2_oos")
    r2b = (oos.get("B", {}) or {}).get("r2_oos")
    r2c = (oos.get("M1", {}) or {}).get("r2_oos")
    r2a = (oos.get("M0", {}) or {}).get("r2_oos")
    signs = list((ev.get("availability_sign") or {}).values())
    dir_ok = (len(signs) >= 2 and all(s == spec_sign for s in signs)
              and (dev_beta is None or dev_beta == 0 or np.sign(dev_beta) == spec_sign))
    inc_cb = (r2c - r2a) if (r2c is not None and r2a is not None) else None
    inc_db = (r2d - r2b) if (r2d is not None and r2b is not None) else None
    outv = ev.get("outliers_delta_D_minus_B", {}) or {}
    loo = outv.get("leave_one_year_out", {}) or {}
    loo_vals = [v for v in loo.values() if v is not None]
    loo_ok = (len(loo_vals) >= 2 and all(v > 0 for v in loo_vals)
              if (inc_db is not None and inc_db > 0) else
              (len(loo_vals) >= 2 and all(v <= 0 for v in loo_vals))
              if inc_db is not None else False)
    win_ok = True
    for key in ("winsorized", "ex_top1pct"):
        v = outv.get(key)
        if v is None or (inc_db is not None and (v > 0) != (inc_db > 0)):
            win_ok = False
    vif = (ev.get("vif") or {}).get("cftc")
    pc = ev.get("partial_corr")
    redundant = ((vif is not None and vif >= 10)
                 or (pc is not None and abs(pc) < 0.02 and inc_db is not None
                     and inc_db > 0))
    evidence = {"dir_ok": dir_ok, "inc_C_minus_A": inc_cb, "inc_D_minus_B": inc_db,
                "loo_ok": loo_ok, "winsor_ok": win_ok, "vif_cftc": vif,
                "partial_corr": pc, "n_test": n_test, "holm_p": holm_p}
    if redundant and dir_ok:
        return "REDUNDANT", evidence
    if not dir_ok:
        return "REJECTED", evidence
    confirmatory = holm_p is not None and holm_p < 0.10
    if (confirmatory and inc_db is not None and inc_db > 0
            and inc_cb is not None and inc_cb > 0
            and loo_ok and win_ok and not redundant):
        return "ROBUST_INCREMENTAL", evidence
    if ((inc_db is not None and inc_db > 0) or (inc_cb is not None and inc_cb > 0)):
        return "WEAK_INCREMENTAL", evidence
    return "REJECTED", evidence


if __name__ == "__main__":
    main()
