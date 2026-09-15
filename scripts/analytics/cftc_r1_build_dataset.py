# scripts/analytics/cftc_r1_build_dataset.py
# -*- coding: utf-8 -*-
"""
R1 — Dataset histórico de PESQUISA CFTC/CME COT (TFF Futures Only).

RESEARCH_HISTORY: usa report_as_of_date como índice. NÃO promete
disponibilidade point-in-time (sem first_seen histórico). Nunca reportar
como backtest livre de look-ahead.

Escopo: 4 contratos CME (standard e micro SEPARADOS, nunca agregados).
Produção intocada: não importa orquestrador, ML, risk ou payload.
Flag ENABLE_CFTC_COT_CONTEXT permanece False (nem lida aqui).

Layout (dados/ é gitignored — dataset grande NÃO é commitado):
  dados/research/cftc/raw/{code}.json
  dados/research/cftc/normalized/{code}.parquet
  dados/research/cftc/normalized/{code}_features.parquet
  dados/research/cftc/metadata/build_metadata.json
Saídas versionadas (commitadas):
  analysis/results/cftc_r1_dataset_quality.json
  docs/audit/CFTC_R1_HISTORICAL_DATASET_2026.md (gerado via --report-md)

Uso:
  python scripts/analytics/cftc_r1_build_dataset.py [--out dados/research/cftc]
      [--contracts 133741,133742,146021,146022] [--no-fetch]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, ".")

from fetchers.cftc_cot_fetcher import (
    DATASET_ID,
    SYMBOL_TO_CONTRACT,
    CftcCotFetcher,
    _parse_asof,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CftcR1")

R1_SCHEMA_VERSION = 1
REPORT_FAMILY = "TFF"
REPORT_SCOPE = "futures_only"

CODES = ["133741", "133742", "146021", "146022"]

# Categorias oficiais TFF -> colunas Socrata (long, short, spreading|None).
CATEGORIES = (
    ("dealer", "dealer_positions_long_all", "dealer_positions_short_all",
     "dealer_positions_spread_all"),
    ("asset_manager", "asset_mgr_positions_long", "asset_mgr_positions_short",
     "asset_mgr_positions_spread"),
    ("leveraged", "lev_money_positions_long", "lev_money_positions_short",
     "lev_money_positions_spread"),
    ("other", "other_rept_positions_long", "other_rept_positions_short",
     "other_rept_positions_spread"),
    ("nonreportable", "nonrept_positions_long_all", "nonrept_positions_short_all",
     None),
)

# Campos de auditoria preservados do raw oficial.
AUDIT_FIELDS = (
    "id", "yyyy_report_week_ww", "contract_market_name", "cftc_market_code",
    "cftc_region_code", "cftc_commodity_code", "commodity_name",
    "contract_units", "cftc_subgroup_code", "commodity", "commodity_subgroup_name",
    "commodity_group_name", "futonly_or_combined",
    "tot_rept_positions_long_all", "tot_rept_positions_short",
    "change_in_open_interest_all",
)

PERCENTILE_WINDOWS = (26, 52, 156)


def _num(raw: object, flags: list, field: str):
    """Parse defensivo: string numérica -> int; inválido -> (None, flag)."""
    if raw is None:
        flags.append(f"missing:{field}")
        return None
    if isinstance(raw, bool):
        flags.append(f"bool:{field}")
        return None
    try:
        s = str(raw).strip().replace(",", "")
        if not s:
            flags.append(f"empty:{field}")
            return None
        f = float(s)
        if not math.isfinite(f):
            flags.append(f"nonfinite:{field}")
            return None
        if f < 0:
            flags.append(f"negative:{field}")
            return None
        if f != int(f):
            flags.append(f"noninteger:{field}")
            return None
        return int(f)
    except (ValueError, TypeError):
        flags.append(f"unparsable:{field}")
        return None


def normalize_row(code: str, row: dict) -> dict:
    """Uma linha Socrata -> registro normalizado (sem agregar, sem corrigir)."""
    flags: list = []
    asof = _parse_asof(row.get("report_date_as_yyyy_mm_dd"))
    if asof is None:
        flags.append("invalid:report_as_of_date")
    elif asof.weekday() != 1:
        # CFTC desloca o asof para segunda em semanas de feriado US
        # (ex. 2018-12-24, 2023-07-03). Informativo, não erro.
        flags.append("non_tuesday_asof")
    if str(row.get("cftc_contract_market_code", "")).strip() != code:
        flags.append("contract_code_mismatch")
    if row.get("futonly_or_combined") not in (None, "FutOnly"):
        flags.append("unexpected_scope")
    oi = _num(row.get("open_interest_all"), flags, "open_interest_all")
    if oi == 0:
        flags.append("oi_zero")
    positions = {}
    for cat, f_long, f_short, f_spread in CATEGORIES:
        long = _num(row.get(f_long), flags, f_long)
        short = _num(row.get(f_short), flags, f_short)
        spread = _num(row.get(f_spread), flags, f_spread) if f_spread else None
        positions[cat] = {"long": long, "short": short, "spreading": spread}
    # Identidades oficiais TFF (FutOnly), verificadas contra a linha real:
    #   sum(reportable long)+sum(reportable spread) == tot_rept_long (idem short)
    #   tot_rept + nonrept == open_interest  (ambos os lados)
    _REPT = ("dealer", "asset_manager", "leveraged", "other")
    if oi is not None:
        for side, f_tot, f_nonrep in (
                ("long", "tot_rept_positions_long_all", "nonrept_positions_long_all"),
                ("short", "tot_rept_positions_short", "nonrept_positions_short_all")):
            rlongs = [positions[c]["long" if side == "long" else "short"]
                      for c in _REPT]
            rspreads = [positions[c]["spreading"] for c in _REPT]
            tot = _num(row.get(f_tot), [], f_tot)
            nonrep = positions["nonreportable"]["short" if side == "short" else "long"]
            if (all(v is not None for v in rlongs + rspreads + [tot, nonrep])):
                if sum(rlongs) + sum(rspreads) != tot:
                    flags.append(f"sum_mismatch_tot_rept_{side}")
                if tot + nonrep != oi:
                    flags.append(f"sum_mismatch_oi_{side}")
    audit = {k: row.get(k) for k in AUDIT_FIELDS}
    return {
        "schema_version": R1_SCHEMA_VERSION,
        "contract_code": code,
        "report_as_of_date": asof.isoformat() if asof else None,
        "market_and_exchange_names": row.get("market_and_exchange_names"),
        "open_interest": oi,
        "positions": positions,
        "audit": audit,
        "quality_flags": sorted(set(flags)),
    }


def _sequence_breaks(asofs: list) -> set:
    """As-ofs após os quais há semana(s) faltante(s) real(is).

    Intervalos curtos (6/8 dias, asof deslocado por feriado) NÃO quebram a
    sequência: ambas as linhas são relatórios semanais consecutivos reais.
    """
    breaks = set()
    for prev, cur in zip(asofs, asofs[1:]):
        days = (date.fromisoformat(cur) - date.fromisoformat(prev)).days
        if days > 7 and max(0, round(days / 7) - 1) > 0:
            breaks.add(prev)
    return breaks


def validate_series(code: str, records: list) -> dict:
    """Auditoria da série de um contrato. Nada é corrigido silenciosamente."""
    by_asofs: dict = {}
    dup_asofs: list = []
    for r in records:
        a = r["report_as_of_date"]
        if a is None:
            continue
        if a in by_asofs:
            dup_asofs.append(a)
            continue  # primeira ocorrência vence; duplicata registrada
        by_asofs[a] = r
    asofs = sorted(by_asofs)
    breaks = _sequence_breaks(asofs)
    gaps: list = []
    short_intervals: list = []
    for prev, cur in zip(asofs, asofs[1:]):
        d0 = date.fromisoformat(prev)
        d1 = date.fromisoformat(cur)
        days = (d1 - d0).days
        if prev in breaks:
            gaps.append({"from": prev, "to": cur,
                         "missing_weeks": max(0, round(days / 7) - 1)})
        elif days != 7:
            short_intervals.append({"from": prev, "to": cur, "days": days})
    names = sorted({r["market_and_exchange_names"] for r in by_asofs.values()})
    flag_counts: dict = {}
    for r in by_asofs.values():
        for fl in r["quality_flags"]:
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
    return {
        "contract_code": code,
        "n_raw_rows": len(records),
        "n_weeks": len(asofs),
        "first": asofs[0] if asofs else None,
        "last": asofs[-1] if asofs else None,
        "gaps": gaps,
        "n_gaps": len(gaps),
        "duplicate_asofs": sorted(set(dup_asofs)),
        "n_duplicates": len(dup_asofs),
        "short_intervals": short_intervals,
        "market_names": names,
        "market_name_changes": len(names) > 1,
        "quality_flag_counts": flag_counts,
        "asofs": asofs,
    }


def derive_features(records: list) -> list:
    """Features por linha ordenada. Regras fail-closed (R1.5)."""
    rows = sorted(
        (r for r in records if r["report_as_of_date"] is not None),
        key=lambda r: r["report_as_of_date"])
    # Consecutividade = adjacência na sequência semanal oficial, sem semana
    # faltante entre elas (intervalos curtos de feriado NÃO quebram).
    break_after = _sequence_breaks([r["report_as_of_date"] for r in rows])
    # distância real entre semanas consecutivas (sem forward-fill).
    gaps_after: dict = {}
    for i in range(1, len(rows)):
        gaps_after[rows[i]["report_as_of_date"]] = (
            rows[i - 1]["report_as_of_date"] in break_after)
    out = []
    for i, r in enumerate(rows):
        asof = r["report_as_of_date"]
        oi = r["open_interest"]
        feat = {"schema_version": R1_SCHEMA_VERSION,
                "contract_code": r["contract_code"],
                "report_as_of_date": asof,
                "open_interest": oi,
                "gap_before": gaps_after.get(asof, False)}
        for cat, _, _, _ in CATEGORIES:
            p = r["positions"][cat]
            long, short = p["long"], p["short"]
            net = (long - short) if (long is not None and short is not None) else None
            feat[f"{cat}_net"] = net
            if oi and oi > 0:
                feat[f"{cat}_share_long_oi"] = (long / oi) if long is not None else None
                feat[f"{cat}_share_short_oi"] = (short / oi) if short is not None else None
                feat[f"{cat}_net_share_oi"] = (net / oi) if net is not None else None
            else:
                feat[f"{cat}_share_long_oi"] = None
                feat[f"{cat}_share_short_oi"] = None
                feat[f"{cat}_net_share_oi"] = None
        # Mudanças 1w/4w: somente semanas consecutivas reais.
        for cat, _, _, _ in CATEGORIES:
            for lag, tag in ((1, "1w"), (4, "4w")):
                key_net, key_share = f"{cat}_net_change_{tag}", f"{cat}_net_share_change_{tag}"
                feat[key_net], feat[key_share] = None, None
                if i >= lag and not feat["gap_before"]:
                    window_ok = True
                    for k in range(i - lag + 1, i + 1):
                        if rows[k]["report_as_of_date"] is None:
                            window_ok = False
                    # consecutividade da janela inteira (sem semana faltante)
                    for k in range(i - lag + 1, i + 1):
                        if rows[k - 1]["report_as_of_date"] in break_after:
                            window_ok = False
                    if window_ok:
                        prev = out[i - lag]
                        cur_net = feat[f"{cat}_net"]
                        prev_net = prev.get(f"{cat}_net")
                        if cur_net is not None and prev_net is not None:
                            feat[key_net] = cur_net - prev_net
                        cur_s = feat[f"{cat}_net_share_oi"]
                        prev_s = prev.get(f"{cat}_net_share_oi")
                        if cur_s is not None and prev_s is not None:
                            feat[key_share] = cur_s - prev_s
        # Percentis: janela completa, sem null, sem gap.
        for cat, _, _, _ in CATEGORIES:
            series_key = f"{cat}_net_share_oi"
            for w in PERCENTILE_WINDOWS:
                key = f"{cat}_net_share_pct{w}"
                feat[key] = None
                if i + 1 >= w:
                    vals = []
                    consecutive = True
                    for k, j in enumerate(range(i - w + 1, i + 1)):
                        v = out[j].get(series_key) if j < i else feat.get(series_key)
                        if v is None:
                            vals = []
                            break
                        vals.append(v)
                        if k > 0 and rows[j - 1]["report_as_of_date"] in break_after:
                            consecutive = False
                    if vals and consecutive and len(vals) == w:
                        below = sum(1 for v in vals[:-1] if v < vals[-1])
                        feat[key] = below / (w - 1) if w > 1 else None
        out.append(feat)
    return out


def pearson(xs: list, ys: list):
    """Pearson pairwise-complete, sem dependências. null se indefinido."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None
             and isinstance(x, (int, float)) and isinstance(y, (int, float))]
    n = len(pairs)
    if n < 3:
        return None, n
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx == 0 or syy == 0:
        return None, n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    return sxy / math.sqrt(sxx * syy), n


def compare_pair(std_feats: list, micro_feats: list) -> dict:
    """Standard vs micro: mesma métrica, semanas em comum. Sem agregação."""
    by_asof_std = {f["report_as_of_date"]: f for f in std_feats}
    by_asof_micro = {f["report_as_of_date"]: f for f in micro_feats}
    common = sorted(set(by_asof_std) & set(by_asof_micro))
    res = {"common_weeks": len(common),
           "std_weeks": len(std_feats), "micro_weeks": len(micro_feats)}
    for cat, _, _, _ in CATEGORIES:
        xs = [by_asof_std[a].get(f"{cat}_net_share_oi") for a in common]
        ys = [by_asof_micro[a].get(f"{cat}_net_share_oi") for a in common]
        corr, n = pearson(xs, ys)
        res[f"{cat}_net_share_corr"] = corr
        res[f"{cat}_net_share_corr_n"] = n
        xw = [by_asof_std[a].get(f"{cat}_net_change_1w") for a in common]
        yw = [by_asof_micro[a].get(f"{cat}_net_change_1w") for a in common]
        corrw, nw = pearson(xw, yw)
        res[f"{cat}_wow_corr"] = corrw
        res[f"{cat}_wow_corr_n"] = nw
    for tag, feats in (("std", std_feats), ("micro", micro_feats)):
        if feats:
            ois = [f["open_interest"] for f in feats if f["open_interest"]]
            res[f"{tag}_oi_first"] = ois[0] if ois else None
            res[f"{tag}_oi_last"] = ois[-1] if ois else None
            res[f"{tag}_oi_growth_x"] = (
                ois[-1] / ois[0] if ois and ois[0] else None)
    return res


async def fetch_all(codes: list) -> dict:
    fetcher = CftcCotFetcher()
    out = {}
    for code in codes:
        rows, err = await fetcher.fetch_contract_history(code, limit_total=10000)
        logger.info("fetch %s: %d rows err=%s", code, len(rows), err)
        out[code] = {"rows": rows, "fetch_error": err}
    return out


def build(out_dir: Path, codes: list, do_fetch: bool) -> dict:
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    norm_dir = out_dir / "normalized"
    meta_dir = out_dir / "metadata"
    for d in (raw_dir, norm_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    fetched: dict = {}
    if do_fetch:
        fetched = asyncio.run(fetch_all(codes))
        for code in codes:
            payload = {
                "provenance": {
                    "source": "CFTC",
                    "dataset_id": DATASET_ID,
                    "report_family": REPORT_FAMILY,
                    "report_scope": REPORT_SCOPE,
                    "contract_code": code,
                    "retrieved_at": retrieved_at,
                    "research_history": True,
                    "point_in_time": False,
                    "first_seen_at": None,
                    "note": ("research_history: availability time unproven; "
                             "do not report as look-ahead-free backtest"),
                },
                "fetch_error": fetched[code]["fetch_error"],
                "rows": fetched[code]["rows"],
            }
            (raw_dir / f"{code}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    quality = {"contracts": {}, "comparisons": {}}
    metadata = {
        "schema_version": R1_SCHEMA_VERSION,
        "provenance": {
            "source": "CFTC",
            "dataset_id": DATASET_ID,
            "report_family": REPORT_FAMILY,
            "report_scope": REPORT_SCOPE,
            "retrieved_at": retrieved_at,
            "research_history": True,
            "point_in_time": False,
        },
        "contracts": {},
    }
    try:
        import pandas as pd  # noqa: F401
        has_pd = True
    except ImportError:
        has_pd = False
    for code in codes:
        raw_path = raw_dir / f"{code}.json"
        if not raw_path.exists():
            quality["contracts"][code] = {"error": "raw_missing"}
            continue
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        rows = payload.get("rows", [])
        records = [normalize_row(code, r) for r in rows if isinstance(r, dict)]
        val = validate_series(code, records)
        quality["contracts"][code] = val
        # normalizado: uma linha por (code, asof); duplicata: primeira vence.
        seen = set()
        norm_rows = []
        for r in sorted(records, key=lambda x: (x["report_as_of_date"] or "")):
            key = (code, r["report_as_of_date"])
            if r["report_as_of_date"] is None or key in seen:
                continue
            seen.add(key)
            norm_rows.append(r)
        feats = derive_features(norm_rows)
        if has_pd:
            import pandas as pd

            pd.DataFrame(norm_rows).to_parquet(norm_dir / f"{code}.parquet", index=False)
            pd.DataFrame(feats).to_parquet(norm_dir / f"{code}_features.parquet", index=False)
        else:
            (norm_dir / f"{code}.json").write_text(
                json.dumps(norm_rows, ensure_ascii=False), encoding="utf-8")
            (norm_dir / f"{code}_features.json").write_text(
                json.dumps(feats, ensure_ascii=False), encoding="utf-8")
        metadata["contracts"][code] = {
            "n_raw_rows": val["n_raw_rows"], "n_weeks": val["n_weeks"],
            "first": val["first"], "last": val["last"],
            "market_names": val["market_names"],
        }
        # guarda feats em memória para R1.6
        quality["contracts"][code]["_feats"] = feats
    feats_by_code = {c: quality["contracts"][c].pop("_feats")
                     for c in codes if "_feats" in quality["contracts"].get(c, {})}
    if "133741" in feats_by_code and "133742" in feats_by_code:
        quality["comparisons"]["BTC_standard_vs_micro"] = compare_pair(
            feats_by_code["133741"], feats_by_code["133742"])
    if "146021" in feats_by_code and "146022" in feats_by_code:
        quality["comparisons"]["ETH_standard_vs_micro"] = compare_pair(
            feats_by_code["146021"], feats_by_code["146022"])
    (meta_dir / "build_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # quality pública (sem asofs completos? com resumo + comparações).
    public = {"contracts": {}, "comparisons": quality["comparisons"]}
    for code, v in quality["contracts"].items():
        public["contracts"][code] = {k: val for k, val in v.items() if k != "asofs"}
        public["contracts"][code]["asof_list_url"] = None
    return {"metadata": metadata, "quality": quality, "public_quality": public}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dados/research/cftc")
    ap.add_argument("--contracts", default=",".join(CODES))
    ap.add_argument("--no-fetch", action="store_true",
                    help="reconstrói tudo a partir do raw já baixado")
    ap.add_argument("--quality-out",
                    default="analysis/results/cftc_r1_dataset_quality.json")
    args = ap.parse_args()
    codes = [c.strip() for c in args.contracts.split(",") if c.strip()]
    unknown = [c for c in codes if c not in CODES]
    if unknown:
        raise SystemExit(f"contract codes fora do mapa validado: {unknown}")
    res = build(Path(args.out), codes, do_fetch=not args.no_fetch)
    qpath = Path(args.quality_out)
    qpath.parent.mkdir(parents=True, exist_ok=True)
    qpath.write_text(json.dumps(res["public_quality"], ensure_ascii=False, indent=2),
                     encoding="utf-8")
    for code in codes:
        v = res["quality"]["contracts"].get(code, {})
        logger.info("%s: weeks=%s first=%s last=%s gaps=%s dups=%s flags=%s",
                    code, v.get("n_weeks"), v.get("first"), v.get("last"),
                    v.get("n_gaps"), v.get("n_duplicates"),
                    v.get("quality_flag_counts"))
    logger.info("comparisons: %s",
                json.dumps(res["quality"]["comparisons"], ensure_ascii=False)[:500])
    print(json.dumps(res["public_quality"], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
