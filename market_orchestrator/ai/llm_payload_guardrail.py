# -*- coding: utf-8 -*-
"""
Guardrail para impedir envio de payload completo/sensível à LLM.
Mantém dependências leves para ser usado em cenários de teste focado.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from market_orchestrator.ai.payload_compressor import compress_payload
from market_orchestrator.ai.payload_metrics_aggregator import append_metric_line


FORBIDDEN_KEYS = {"raw_event", "ANALYSIS_TRIGGER", "contextual_snapshot", "historical_vp", "observability"}


def _log_guardrail(leak_blocked: bool, bytes_before: int, bytes_after: int, root: str, error: Optional[str] = None) -> None:
    metrics = {
        "payload_bytes": bytes_before,
        "leak_blocked": leak_blocked,
        "bytes_after": bytes_after,
        "payload_root_name": root,
    }
    if error:
        metrics["error"] = error

    line = json.dumps(metrics, ensure_ascii=False)
    logging.info(line)
    append_metric_line(metrics)


def ensure_safe_llm_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Guardrail HARD: evita enviar payloads com campos sensíveis para a LLM.
    Retorna um payload seguro ou None (para abortar).
    """
    if not isinstance(payload, dict):
        _log_guardrail(True, 0, 0, "invalid_payload", "payload_not_dict")
        return None

    bytes_before = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    leak_keys = FORBIDDEN_KEYS.intersection(payload.keys())
    if not leak_keys:
        _log_guardrail(False, bytes_before, bytes_before, "event")
        return payload

    candidates = [
        (payload.get("AI_ANALYSIS") or {}).get("ai_payload"),
        payload.get("ai_payload"),
        (payload.get("enriched_snapshot") or {}).get("ai_payload"),
    ]

    safe_candidate: Optional[Dict[str, Any]] = None
    for cand in candidates:
        if isinstance(cand, dict):
            safe_candidate = cand
            break

    if safe_candidate is None:
        logging.error("FULL_PAYLOAD_LEAK_ABORTED forbidden_keys=%s", list(leak_keys), exc_info=True)
        _log_guardrail(True, bytes_before, 0, "event", "no_safe_candidate")
        return None

    # Se o payload já é v2 (já foi comprimido por build_ai_input),
    # NÃO comprimir novamente
    if safe_candidate.get("_v") == 2:
        compressed = safe_candidate
        logging.debug("GUARDRAIL_SKIP_COMPRESSION payload already v2")
    else:
        try:
            compressed = compress_payload(safe_candidate, max_bytes=6144)
        except Exception:
            compressed = safe_candidate

    bytes_after = len(json.dumps(compressed, ensure_ascii=False).encode("utf-8"))
    logging.warning(
        "FULL_PAYLOAD_LEAK_BLOCKED root=event bytes_before=%s bytes_after=%s",
        bytes_before,
        bytes_after,
    )
    _log_guardrail(True, bytes_before, bytes_after, "event")

    sanitized_event: Dict[str, Any] = {
        "ai_payload": compressed,
    }
    for k in ("tipo_evento", "descricao", "symbol", "ativo"):
        if k in payload:
            sanitized_event[k] = payload[k]

    return sanitized_event
