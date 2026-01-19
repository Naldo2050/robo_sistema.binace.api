# -*- coding: utf-8 -*-
"""
Agregador leve de m\u00e9tricas do payload (p50/p95, caches, guardrail).

L\u00ea o arquivo JSONL usado pelo guardrail e pelos loggers de payload, ignorando linhas
quebradas, e devolve um resumo num dicion\u00e1rio.
"""

from __future__ import annotations

import json
import logging
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_METRICS_PATH_LOGGED = False


def append_metric_line(obj: Dict[str, object], metrics_path: str = "logs/payload_metrics.jsonl") -> None:
    """Anexa uma linha de métrica em JSONL, logando o caminho absoluto na primeira escrita."""
    global _METRICS_PATH_LOGGED
    try:
        p = Path(metrics_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not _METRICS_PATH_LOGGED:
            logging.info("PAYLOAD_METRICS_FILE path=%s cwd=%s", str(p), str(Path.cwd()))
            _METRICS_PATH_LOGGED = True
        with p.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        logging.debug("Falha ao persistir métricas no JSONL", exc_info=True)


def _safe_loads(line: str) -> Optional[Dict]:
    try:
        return json.loads(line)
    except Exception:
        return None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    idx = int(round((pct / 100) * (len(values) - 1)))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def summarize_metrics(path: str, last_n: int = 2000) -> Dict[str, object]:
    """
    Resume m\u00e9tricas do payload a partir de um JSONL.

    Ignora linhas inv\u00e1lidas e \u00e9 tolerante a arquivos ausentes.
    """
    metrics_path = Path(path)
    if not metrics_path.exists():
        return {}

    lines = deque(maxlen=last_n)
    try:
        with metrics_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                lines.append(line.rstrip("\n"))
    except Exception:
        return {}

    parsed = [_safe_loads(l) for l in lines]
    records = [p for p in parsed if isinstance(p, dict)]
    total = len(records)
    if total == 0:
        return {}

    byte_values: List[float] = []
    guardrail_blocked = 0
    aborts = 0
    fallbacks = 0
    cache_hits: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hit": 0, "total": 0})

    for rec in records:
        val = rec.get("payload_bytes")
        if val is None:
            val = rec.get("bytes_after")
        if isinstance(val, (int, float)):
            byte_values.append(float(val))

        if rec.get("leak_blocked") is True:
            guardrail_blocked += 1
        if rec.get("error") == "no_safe_candidate":
            aborts += 1
        if rec.get("fallback_v1") is True:
            fallbacks += 1

        if "cache_hit" in rec and "section" in rec:
            sec = str(rec.get("section"))
            cache_hits[sec]["total"] += 1
            if rec.get("cache_hit") is True:
                cache_hits[sec]["hit"] += 1

    summary: Dict[str, object] = {
        "count": total,
        "bytes_p50": _percentile(byte_values, 50),
        "bytes_p90": _percentile(byte_values, 90),
        "bytes_p95": _percentile(byte_values, 95),
        "bytes_max": max(byte_values) if byte_values else None,
        "guardrail_block_rate": guardrail_blocked / total,
        "abort_rate": aborts / total,
        "fallback_rate": fallbacks / total,
    }

    cache_summary: Dict[str, float] = {}
    for sec, data in cache_hits.items():
        if data["total"] > 0:
            cache_summary[sec] = data["hit"] / data["total"]
    if cache_summary:
        summary["cache_hit_rate"] = cache_summary

    return summary
