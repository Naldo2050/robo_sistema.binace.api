# -*- coding: utf-8 -*-
"""
Utilitário para reduzir o payload enviado à LLM sem mutar o original.
Aplica allowlist, poda listas grandes e remove blocos de debug/observability.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, Optional, Tuple


ALLOWED_TOP_LEVEL = {
    "symbol",
    "timestamp",
    "timestamp_ms",
    "epoch_ms",
    "signal_metadata",
    "price_context",
    "flow_context",
    "orderbook_context",
    "technical_indicators",
    "cross_asset_context",
    "macro_context",
    "quant_model",
    "regime_analysis",
    "historical_stats",
    "multi_tf",
    "tf",  # alias compactado para multi_tf
}

FORBIDDEN_KEYS = {
    "raw_event",
    "observability",
    "historical_vp",
    "data_quality",
    "debug",
}

logger = logging.getLogger(__name__)


SECTION_BUDGETS_BASE = {
    "price_context": 800,
    "flow_context": 900,
    "orderbook_context": 1400,
    "macro_context": 900,
    "liquidity_heatmap": 1800,
}


def _json_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _limit_list(value: Any, max_len: int) -> Any:
    if isinstance(value, list) and len(value) > max_len:
        return value[:max_len]
    return value


def _normalize_cluster_times(cluster: Dict[str, Any], epoch_ms: Optional[int]) -> Dict[str, Any]:
    """
    Deriva age_ms e cluster_duration_ms a partir de timestamps de primeiro/último visto.
    Remove campos redundantes de timestamp.
    """
    if not isinstance(cluster, dict) or epoch_ms is None:
        return cluster

    first_ts = cluster.get("first_seen_ms")
    last_ts = (
        cluster.get("last_seen_ms")
        or cluster.get("recent_ts_ms")
        or cluster.get("recent_timestamp")
    )

    derived = dict(cluster)
    derived_flag = False

    try:
        if last_ts is not None:
            age = max(0, int(epoch_ms) - int(last_ts))
            derived["age_ms"] = age
            derived_flag = True
        if last_ts is not None and first_ts is not None:
            duration = max(0, int(last_ts) - int(first_ts))
            derived["cluster_duration_ms"] = duration
            derived_flag = True
    except Exception:
        pass

    for redundant in ["first_seen_ms", "last_seen_ms", "recent_ts_ms", "recent_timestamp"]:
        derived.pop(redundant, None)

    if derived_flag:
        logger.debug("cluster_time_derived=true")

    return derived


def _trim_known_lists(block: Dict[str, Any], epoch_ms: Optional[int]) -> Dict[str, Any]:
    if not isinstance(block, dict):
        return block  # type: ignore[return-value]

    trimmed = {}
    for key, val in block.items():
        if key == "clusters":
            limited = _limit_list(val, 3)
            if isinstance(limited, list):
                trimmed[key] = [
                    _normalize_cluster_times(item, epoch_ms) if isinstance(item, dict) else item
                    for item in limited
                ]
            else:
                trimmed[key] = limited
        elif key in {"hvn_nodes", "lvn_nodes"}:
            trimmed[key] = _limit_list(val, 5)
        elif isinstance(val, dict):
            trimmed[key] = _trim_known_lists(val, epoch_ms)
        elif isinstance(val, list):
            trimmed[key] = [
                _trim_known_lists(item, epoch_ms) if isinstance(item, dict) else item
                for item in _limit_list(val, 10)
            ]
        else:
            trimmed[key] = val

    return trimmed


def _round_numeric_fields(data: Dict[str, Any], keys: Tuple[str, ...], ndigits: int = 4) -> Dict[str, Any]:
    rounded = dict(data)
    for k in keys:
        if k in rounded and isinstance(rounded[k], (int, float)):
            rounded[k] = round(float(rounded[k]), ndigits)
    return rounded


def _reduce_liquidity_heatmap(hm: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(hm, dict):
        return hm  # type: ignore[return-value]

    reduced = dict(hm)
    clusters = reduced.get("clusters")
    if isinstance(clusters, list):
        # Reduz o número de clusters gradualmente
        if len(clusters) > 2:
            clusters = clusters[:2]
        elif len(clusters) > 1:
            clusters = clusters[:1]

        trimmed_clusters = []
        for cl in clusters:
            if isinstance(cl, dict):
                rounded = _round_numeric_fields(
                    cl,
                    ("price", "liquidity", "volume", "ask_price", "bid_price", "notional"),
                    ndigits=4,
                )
                # Remove campos secundários
                for key in list(rounded.keys()):
                    if key.endswith("_std") or key.endswith("variance"):
                        rounded.pop(key, None)
                trimmed_clusters.append(rounded)
            else:
                trimmed_clusters.append(cl)
        reduced["clusters"] = trimmed_clusters

    # Remove metadados volumosos
    for k in ["heatmap_meta", "stats"]:
        reduced.pop(k, None)

    return reduced


def _reduce_orderbook_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        return ctx  # type: ignore[return-value]

    reduced = dict(ctx)
    depth = reduced.get("depth_metrics")
    if isinstance(depth, dict):
        depth_copy = dict(depth)
        for key, val in depth.items():
            if isinstance(val, list):
                if len(val) > 3:
                    depth_copy[key] = val[:3]
                elif len(val) > 1:
                    depth_copy[key] = val[:1]
        reduced["depth_metrics"] = depth_copy

    # Remove métricas deriváveis/menos críticas
    for key in ["market_impact_score", "walls_detected"]:
        reduced.pop(key, None)

    return reduced


def _reduce_macro_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        return ctx  # type: ignore[return-value]
    reduced = dict(ctx)
    reduced.pop("correlations", None)
    reduced.pop("multi_timeframe_trends", None)
    return reduced


def _reduce_flow_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        return ctx  # type: ignore[return-value]
    reduced = dict(ctx)
    # Remove detalhes menos críticos se necessário
    for key in ["whale_activity", "absorption_type"]:
        if key in reduced:
            reduced.pop(key, None)
            break
    return reduced


def _reduce_price_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ctx, dict):
        return ctx  # type: ignore[return-value]
    reduced = _round_numeric_fields(ctx, ("current_price",))
    if "price_action" in reduced:
        reduced = dict(reduced)
        reduced.pop("price_action", None)
    return reduced


def _enforce_section_budget(section: Dict[str, Any], budget_bytes: int, reducer_fn) -> Dict[str, Any]:
    """
    Aplica reduções iterativas em um bloco até ficar dentro do orçamento ou esgotar reduções.
    """
    if not isinstance(section, dict):
        return section  # type: ignore[return-value]

    current = copy.deepcopy(section)
    while _json_bytes(current) > budget_bytes:
        reduced = reducer_fn(current)
        if reduced == current:
            break
        current = reduced
    return current


def _normalize_signal_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": meta.get("type"),
        "battle_result": meta.get("battle_result"),
        "severity": meta.get("severity"),
        "window_id": meta.get("window_id"),
        "description": meta.get("description"),
    }


def _normalize_timestamps(payload: Dict[str, Any]) -> Tuple[int | None, Dict[str, Any]]:
    epoch_ms = payload.get("epoch_ms") or payload.get("timestamp_ms") or payload.get("timestamp")
    try:
        epoch_ms = int(epoch_ms) if epoch_ms is not None else None
    except Exception:
        epoch_ms = None

    cleaned = dict(payload)
    cleaned.pop("timestamp_ms", None)
    cleaned.pop("timestamp", None)
    cleaned["epoch_ms"] = epoch_ms
    return epoch_ms, cleaned


def compress_payload(payload: Dict[str, Any], max_bytes: int = 6144) -> Dict[str, Any]:
    """
    Gera uma versão reduzida do payload para envio à LLM.
    - Não muta o original.
    - Aplica allowlist e poda listas conhecidas.
    - Remove blocos de observabilidade/debug.
    - Normaliza timestamps para epoch_ms.
    - Tenta manter-se abaixo de max_bytes (best-effort).
    """
    base = copy.deepcopy(payload) if isinstance(payload, dict) else {}

    # Remove chaves proibidas antecipadamente
    for k in list(base.keys()):
        if k in FORBIDDEN_KEYS:
            base.pop(k, None)

    # Prepara orçamentos escalados conforme max_bytes
    scale = max_bytes / 6144 if max_bytes else 1
    section_budgets = {
        k: max(256, int(v * scale))
        for k, v in SECTION_BUDGETS_BASE.items()
    }

    compressed: Dict[str, Any] = {}
    for key in ALLOWED_TOP_LEVEL:
        if key not in base:
            continue
        val = base[key]
        if key == "signal_metadata" and isinstance(val, dict):
            compressed[key] = _normalize_signal_metadata(val)
        elif isinstance(val, dict):
            compressed[key] = _trim_known_lists(val, base.get("epoch_ms"))
        else:
            compressed[key] = val

    epoch_ms, compressed = _normalize_timestamps(compressed)
    compressed["_v"] = 2

    # Enforce budgets por seção
    if "price_context" in compressed:
        compressed["price_context"] = _enforce_section_budget(
            compressed["price_context"],
            section_budgets.get("price_context", max_bytes),
            _reduce_price_context,
        )

    if "flow_context" in compressed and isinstance(compressed["flow_context"], dict):
        flow_ctx = dict(compressed["flow_context"])
        if "liquidity_heatmap" in flow_ctx:
            flow_ctx["liquidity_heatmap"] = _enforce_section_budget(
                flow_ctx["liquidity_heatmap"],
                section_budgets.get("liquidity_heatmap", max_bytes),
                _reduce_liquidity_heatmap,
            )
        compressed["flow_context"] = _enforce_section_budget(
            flow_ctx,
            section_budgets.get("flow_context", max_bytes),
            _reduce_flow_context,
        )

    if "orderbook_context" in compressed:
        compressed["orderbook_context"] = _enforce_section_budget(
            compressed["orderbook_context"],
            section_budgets.get("orderbook_context", max_bytes),
            _reduce_orderbook_context,
        )

    if "macro_context" in compressed:
        compressed["macro_context"] = _enforce_section_budget(
            compressed["macro_context"],
            section_budgets.get("macro_context", max_bytes),
            _reduce_macro_context,
        )

    # Estratégia de poda adicional se tamanho exceder
    def _size_ok(d: Dict[str, Any]) -> bool:
        try:
            return _json_bytes(d) <= max_bytes
        except Exception:
            return False

    if not _size_ok(compressed):
        optional_fields = [
            "cross_asset_context",
            "macro_context",
            "technical_indicators",
            "historical_stats",
        ]
        for opt in optional_fields:
            if opt in compressed and not _size_ok(compressed):
                compressed.pop(opt, None)

    # Move multi_tf → tf (alias compactado, sem duplicação)
    # Economiza bytes e padroniza o nome usado pelo consumer
    if "multi_tf" in compressed and isinstance(compressed["multi_tf"], dict) and compressed["multi_tf"]:
        compressed["tf"] = compressed.pop("multi_tf")

    return compressed
