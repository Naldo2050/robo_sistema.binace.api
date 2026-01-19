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
}

FORBIDDEN_KEYS = {
    "raw_event",
    "observability",
    "historical_vp",
    "data_quality",
    "debug",
}

logger = logging.getLogger(__name__)


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


def _normalize_signal_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": meta.get("type"),
        "battle_result": meta.get("battle_result"),
        "severity": meta.get("severity"),
        "window_id": meta.get("window_id"),
        "description": meta.get("description"),
    }


def _normalize_timestamps(payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
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

    # Estratégia de poda adicional se tamanho exceder
    def _size_ok(d: Dict[str, Any]) -> bool:
        try:
            return len(json.dumps(d, ensure_ascii=False).encode("utf-8")) <= max_bytes
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

    return compressed
