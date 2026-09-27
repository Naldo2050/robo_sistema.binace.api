# scripts/analytics/effort_response_distribution.py
# -*- coding: utf-8 -*-
"""
P1-E — Distribuição Empírica Esforço vs Resposta (Somente Descritivo).

Gera relatório estatístico descritivo offline para o dataset shadow de
esforço e resposta de preço contemporâneos.

RESTRIÇÕES ESTRITAS:
- NÃO calcula thresholds operacionais.
- NÃO emite sugestões BUY/SELL ou sinais.
- NÃO otimiza outcomes futuros nem realiza feature selection.
- NÃO treina modelos nem calcula probabilidades ou confidence.
- Regime é tratado estritamente como contexto heurístico observacional
  (HEURISTIC_CONTEXT), sem ground truth.
- Outcomes não resolvidos (PENDING/INSUFFICIENT_DATA) permanecem missing
  e NUNCA são imputados como retorno zero.
- Reporta contagens brutas (raw count), overlaps e gaps sem assumir independência
  estatística das janelas.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# 1 semana tem no máximo 10.080 minutos teóricos (7 dias * 24 horas * 60 minutos)
MAX_THEORETICAL_MINUTES_PER_WEEK: int = 10080

RAW_METRICS: Tuple[str, ...] = (
    "total_aggressive_notional_usd",
    "net_aggressive_notional_usd",
    "buy_notional_usd",
    "sell_notional_usd",
    "buy_share",
    "sell_share",
    "price_displacement_usd",
    "price_displacement_bps",
    "range_usd",
    "range_bps",
    "close_from_high_usd",
    "close_from_high_bps",
    "close_from_low_usd",
    "close_from_low_bps",
    "close_vs_vwap_bps",
    "close_vs_poc_bps",
)


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Calcula percentil p (0..100) via interpolação linear padrão."""
    if not values:
        return None
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 1:
        return sorted_v[0]
    idx = (p / 100.0) * (n - 1)
    low_idx = int(math.floor(idx))
    high_idx = int(math.ceil(idx))
    if low_idx == high_idx:
        return sorted_v[low_idx]
    weight = idx - low_idx
    return sorted_v[low_idx] * (1.0 - weight) + sorted_v[high_idx] * weight


def compute_metric_stats(values_with_nones: Sequence[Any]) -> Dict[str, Any]:
    """Calcula estatísticas descritivas básicas para uma série com possíveis Nones."""
    valid_nums: List[float] = []
    missing_count = 0
    total = len(values_with_nones)

    for v in values_with_nones:
        if v is None:
            missing_count += 1
        elif isinstance(v, (int, float)) and math.isfinite(v):
            valid_nums.append(float(v))
        else:
            missing_count += 1

    valid_count = len(valid_nums)
    missing_pct = (missing_count / total * 100.0) if total > 0 else 0.0

    if valid_count == 0:
        return {
            "count": 0,
            "total_records": total,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 2),
            "mean": None,
            "median": None,
            "std": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    mean_val = sum(valid_nums) / valid_count
    if valid_count > 1:
        var = sum((x - mean_val) ** 2 for x in valid_nums) / (valid_count - 1)
        std_val = math.sqrt(var)
    else:
        std_val = 0.0

    return {
        "count": valid_count,
        "total_records": total,
        "missing_count": missing_count,
        "missing_pct": round(missing_pct, 2),
        "mean": mean_val,
        "median": _percentile(valid_nums, 50.0),
        "std": std_val,
        "p10": _percentile(valid_nums, 10.0),
        "p25": _percentile(valid_nums, 25.0),
        "p50": _percentile(valid_nums, 50.0),
        "p75": _percentile(valid_nums, 75.0),
        "p90": _percentile(valid_nums, 90.0),
        "p95": _percentile(valid_nums, 95.0),
        "p99": _percentile(valid_nums, 99.0),
    }


def compute_lag1_autocorrelation(values_with_nones: Sequence[Any]) -> Optional[float]:
    """Calcula autocorrelação de Pearson lag-1 sem dependências pesadas."""
    valid_pairs: List[Tuple[float, float]] = []
    for i in range(len(values_with_nones) - 1):
        v1 = values_with_nones[i]
        v2 = values_with_nones[i + 1]
        if (
            v1 is not None and v2 is not None
            and isinstance(v1, (int, float)) and math.isfinite(v1)
            and isinstance(v2, (int, float)) and math.isfinite(v2)
        ):
            valid_pairs.append((float(v1), float(v2)))

    if len(valid_pairs) < 3:
        return None

    x = [p[0] for p in valid_pairs]
    y = [p[1] for p in valid_pairs]
    n = len(valid_pairs)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    var_y = sum((y[i] - mean_y) ** 2 for i in range(n))

    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return 0.0
    return cov / denom


def analyze_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Realiza análise descritiva multivariada e estratificada."""
    total_records = len(records)
    if total_records == 0:
        return {
            "summary": {
                "total_records": 0,
                "status": "EMPTY_DATASET",
                "max_theoretical_minutes_per_week": MAX_THEORETICAL_MINUTES_PER_WEEK,
            },
            "temporal_sample_metrics": {},
            "raw_metrics_distribution": {},
            "stratifications": {},
        }

    # Distribuição geral das métricas raw
    raw_dist: Dict[str, Any] = {}
    autocorrs: Dict[str, Any] = {}
    for metric in RAW_METRICS:
        vals = [r.get("features_at_t", {}).get(metric) for r in records]
        raw_dist[metric] = compute_metric_stats(vals)
        autocorrs[metric] = compute_lag1_autocorrelation(vals)

    # Métricas de amostra temporal e overlap
    sorted_recs = sorted(
        records,
        key=lambda r: int(r.get("provenance", {}).get("window_open_ms", 0))
    )
    gaps_count = 0
    overlaps_count = 0
    consecutive_perfect_count = 0
    window_durations: List[float] = []

    for i in range(len(sorted_recs)):
        cur_open = int(sorted_recs[i].get("provenance", {}).get("window_open_ms", 0))
        cur_close = int(sorted_recs[i].get("provenance", {}).get("window_close_ms", 0))
        dur = sorted_recs[i].get("features_at_t", {}).get("window_duration_ms")
        if dur is not None and isinstance(dur, (int, float)) and math.isfinite(dur):
            window_durations.append(float(dur))

        if i > 0:
            prev_close = int(sorted_recs[i - 1].get("provenance", {}).get("window_close_ms", 0))
            if prev_close > 0 and cur_open > 0:
                diff = cur_open - prev_close
                if diff > 0:
                    gaps_count += 1
                elif diff < 0:
                    overlaps_count += 1
                else:
                    consecutive_perfect_count += 1

    first_ts = int(sorted_recs[0].get("provenance", {}).get("window_open_ms", 0))
    last_ts = int(sorted_recs[-1].get("provenance", {}).get("window_close_ms", 0))
    span_days = max(1e-4, (last_ts - first_ts) / (86_400_000.0))
    records_per_day = total_records / span_days

    temporal_sample_metrics = {
        "total_records": total_records,
        "span_days": round(span_days, 2),
        "records_per_day": round(records_per_day, 2),
        "gaps_count": gaps_count,
        "overlaps_count": overlaps_count,
        "consecutive_perfect_count": consecutive_perfect_count,
        "window_duration_ms": compute_metric_stats(window_durations),
        "autocorrelation_lag1": autocorrs,
        "max_theoretical_minutes_per_week": MAX_THEORETICAL_MINUTES_PER_WEEK,
    }

    # Estratificações descritivas
    # 1. Por session_time_bucket
    session_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        bucket = r.get("context_at_t", {}).get("session_time_bucket") or "UNKNOWN_BUCKET"
        session_groups.setdefault(bucket, []).append(r)

    by_session: Dict[str, Any] = {}
    for bucket, grp_records in session_groups.items():
        by_session[bucket] = {
            "record_count": len(grp_records),
            "price_displacement_bps": compute_metric_stats(
                [x.get("features_at_t", {}).get("price_displacement_bps") for x in grp_records]
            ),
            "range_bps": compute_metric_stats(
                [x.get("features_at_t", {}).get("range_bps") for x in grp_records]
            ),
            "total_notional": compute_metric_stats(
                [x.get("features_at_t", {}).get("total_aggressive_notional_usd") for x in grp_records]
            ),
        }

    # 2. Por fim de semana (is_weekend)
    weekend_groups: Dict[str, List[Dict[str, Any]]] = {"weekend": [], "weekday": [], "unknown": []}
    for r in records:
        is_wk = r.get("context_at_t", {}).get("is_weekend")
        if is_wk is True:
            weekend_groups["weekend"].append(r)
        elif is_wk is False:
            weekend_groups["weekday"].append(r)
        else:
            weekend_groups["unknown"].append(r)

    by_weekend: Dict[str, Any] = {}
    for k, grp_records in weekend_groups.items():
        if grp_records:
            by_weekend[k] = {
                "record_count": len(grp_records),
                "range_bps": compute_metric_stats(
                    [x.get("features_at_t", {}).get("range_bps") for x in grp_records]
                ),
                "total_notional": compute_metric_stats(
                    [x.get("features_at_t", {}).get("total_aggressive_notional_usd") for x in grp_records]
                ),
            }

    # 3. Por Regime Heurístico Contemporâneo (apenas descritivo)
    regime_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        reg = r.get("context_at_t", {}).get("regime_current_at_t") or "UNKNOWN"
        regime_groups.setdefault(reg, []).append(r)

    by_regime: Dict[str, Any] = {}
    for reg, grp_records in regime_groups.items():
        by_regime[reg] = {
            "note": "HEURISTIC_CONTEXT",
            "record_count": len(grp_records),
            "price_displacement_bps": compute_metric_stats(
                [x.get("features_at_t", {}).get("price_displacement_bps") for x in grp_records]
            ),
            "range_bps": compute_metric_stats(
                [x.get("features_at_t", {}).get("range_bps") for x in grp_records]
            ),
            "buy_share": compute_metric_stats(
                [x.get("features_at_t", {}).get("buy_share") for x in grp_records]
            ),
        }

    # 4. Estratificação quantílica offline por volume total
    tot_notionals = [
        float(r.get("features_at_t", {}).get("total_aggressive_notional_usd", 0.0))
        for r in records
        if r.get("features_at_t", {}).get("total_aggressive_notional_usd") is not None
    ]
    volume_quantiles: Dict[str, Any] = {}
    if len(tot_notionals) >= 4:
        q25 = _percentile(tot_notionals, 25.0) or 0.0
        q50 = _percentile(tot_notionals, 50.0) or 0.0
        q75 = _percentile(tot_notionals, 75.0) or 0.0

        q_groups: Dict[str, List[Dict[str, Any]]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        for r in records:
            v = r.get("features_at_t", {}).get("total_aggressive_notional_usd")
            if v is None:
                continue
            if v <= q25:
                q_groups["Q1"].append(r)
            elif v <= q50:
                q_groups["Q2"].append(r)
            elif v <= q75:
                q_groups["Q3"].append(r)
            else:
                q_groups["Q4"].append(r)

        for q_name, q_recs in q_groups.items():
            volume_quantiles[q_name] = {
                "record_count": len(q_recs),
                "range_bps": compute_metric_stats(
                    [x.get("features_at_t", {}).get("range_bps") for x in q_recs]
                ),
                "displacement_bps": compute_metric_stats(
                    [x.get("features_at_t", {}).get("price_displacement_bps") for x in q_recs]
                ),
            }

    # 5. Estratificação descritiva por horizonte de outcomes (exclusivamente para status RESOLVED)
    outcomes_summary: Dict[str, Any] = {}
    for h in ("1m", "5m", "15m"):
        h_returns: List[Any] = []
        h_mfes: List[Any] = []
        h_maes: List[Any] = []
        resolved_count = 0
        pending_count = 0
        insufficient_data_count = 0

        for r in records:
            h_data = r.get("outcomes_future", {}).get("horizons", {}).get(h, {})
            st = h_data.get("status")
            if st == "RESOLVED":
                resolved_count += 1
                h_returns.append(h_data.get("return_bps"))
                h_mfes.append(h_data.get("max_excursion_up_bps", h_data.get("mfe_bps")))
                h_maes.append(h_data.get("max_excursion_down_bps", h_data.get("mae_bps")))
            elif st == "INSUFFICIENT_DATA":
                insufficient_data_count += 1
                h_returns.append(None)
                h_mfes.append(None)
                h_maes.append(None)
            else:
                # PENDING: permanece explicitamente None (missing)
                pending_count += 1
                h_returns.append(None)
                h_mfes.append(None)
                h_maes.append(None)

        outcomes_summary[h] = {
            "resolved_count": resolved_count,
            "pending_count": pending_count,
            "insufficient_data_count": insufficient_data_count,
            "return_bps": compute_metric_stats(h_returns),
            "max_excursion_up_bps": compute_metric_stats(h_mfes),
            "max_excursion_down_bps": compute_metric_stats(h_maes),
        }

    return {
        "summary": {
            "total_records": total_records,
            "contract_version": "1.0.0",
            "schema_version": "1.0.0",
            "status": "VALID_OBSERVATIONAL_DISTRIBUTION",
            "max_theoretical_minutes_per_week": MAX_THEORETICAL_MINUTES_PER_WEEK,
        },
        "temporal_sample_metrics": temporal_sample_metrics,
        "raw_metrics_distribution": raw_dist,
        "stratifications": {
            "by_session_time_bucket": by_session,
            "by_weekend": by_weekend,
            "by_regime_heuristic": by_regime,
            "by_volume_quantiles": volume_quantiles,
            "by_outcomes_horizon": outcomes_summary,
        },
    }


def load_records_from_jsonl(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Carrega lista de registros a partir de um arquivo JSONL."""
    path = Path(filepath)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") != "OUTCOME_UPDATE":
                    records.append(data)
            except Exception:
                continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P1-E: Relatório Estatístico Descritivo de Esforço e Resposta de Preço"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dados/datasets/shadow_effort_response.jsonl",
        help="Caminho do arquivo JSONL do dataset shadow",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho para salvar o relatório JSON gerado (opcional)",
    )
    args = parser.parse_args()

    records = load_records_from_jsonl(args.input)
    print(f"[P1-E Distribution] Registros carregados: {len(records)} de '{args.input}'")

    report = analyze_records(records)

    report_str = json.dumps(report, indent=2, allow_nan=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_str, encoding="utf-8")
        print(f"[P1-E Distribution] Relatório salvo em: {args.output}")
    else:
        print(report_str)


if __name__ == "__main__":
    main()
