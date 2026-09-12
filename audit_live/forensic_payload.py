# audit_live/forensic_payload.py
# -*- coding: utf-8 -*-
"""Construção local do payload que *seria* enviado à IA, sem transmitir.

Fluxo: event -> builder -> compressor -> guardrail.
NUNCA chama chat.completions.create / OpenAI / Groq / Qwen.
Salva em llm_payloads.jsonl com llm_transmitted=false + estágios em payload_stages.jsonl.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def build_and_capture_future_payload(event_data: Dict[str, Any]) -> Dict[str, Any] | None:
    """Constrói payload futuro. Retorna payload final ou None. Nunca levanta."""
    if os.getenv("FORENSIC_CAPTURE", "0") != "1":
        return None
    try:
        from audit_live.forensic_context import CTX
        from audit_live.forensic_writer import WRITER
        from audit_live.hooks import on_payload_stage, on_quality

        CTX.ensure_init(os.getenv("FORENSIC_DIR", ""))
        event_id = event_data.get("event_id")
        epoch_ms = event_data.get("epoch_ms")
        seq = event_data.get("sequence_id")

        # PRE_BUILDER (deepcopy defensivo; se falhar, registra e aborta sem derrubar)
        try:
            pre = copy.deepcopy(event_data)
        except Exception as e_copy:
            logger.warning("FORENSIC pre_builder deepcopy falhou: %s", e_copy, exc_info=True)
            return None
        on_payload_stage("PRE_BUILDER", event_id, epoch_ms, {"note": "pre_builder_ref", "keys": sorted([str(k) for k in event_data.keys()])[:50]})

        # BUILDER (produção, sem modificação)
        try:
            from market_orchestrator.ai.payload_builder_compact import build_compact_payload

            built = build_compact_payload(event_data)
        except Exception as e_build:
            logger.warning("FORENSIC builder falhou: %s", e_build, exc_info=True)
            WRITER.append_jsonl(
                CTX.path("audit.log"),
                {"ts_ms": int(time.time() * 1000), "event": "forensic_builder_failed",
                 "capture_run_id": CTX.capture_run_id, "event_id": event_id, "error": str(e_build)[:500]},
            )
            return None
        on_payload_stage("POST_BUILDER", event_id, epoch_ms, built)

        # COMPRESSOR (produção)
        try:
            from market_orchestrator.ai.payload_compressor import compress_payload

            compressed = compress_payload(built)
        except Exception as e_comp:
            logger.warning("FORENSIC compressor falhou, usa built: %s", e_comp, exc_info=True)
            compressed = built
        on_payload_stage("POST_COMPRESSOR", event_id, epoch_ms, compressed)

        # GUARDRAIL (produção)
        try:
            from market_orchestrator.ai.llm_payload_guardrail import ensure_safe_llm_payload

            final = ensure_safe_llm_payload(compressed)
        except Exception as e_guard:
            logger.warning("FORENSIC guardrail falhou: %s", e_guard, exc_info=True)
            final = compressed
        if final is None:
            WRITER.append_jsonl(
                CTX.path("audit.log"),
                {"ts_ms": int(time.time() * 1000), "event": "forensic_guardrail_aborted",
                 "capture_run_id": CTX.capture_run_id, "event_id": event_id},
            )
            return None
        on_payload_stage("POST_GUARDRAIL", event_id, epoch_ms, final)

        # Payload final integral (seção 14)
        try:
            size = len(json.dumps(final, ensure_ascii=False, default=str).encode("utf-8"))
        except Exception as e_size:
            logger.warning("FORENSIC final size falhou: %s", e_size, exc_info=True)
            size = -1
        rec = {
            "capture_run_id": CTX.capture_run_id,
            "session_id": CTX.session_id,
            "event_id": event_id,
            "sequence_id": seq,
            "symbol": event_data.get("symbol"),
            "window": event_data.get("janela_numero", event_data.get("window")),
            "epoch_ms": epoch_ms,
            "payload": final,
            "payload_size_bytes": size,
            "llm_transmitted": False,
            "observed_at_ms": int(time.time() * 1000),
        }
        ok = WRITER.append_jsonl(CTX.path("llm_payloads.jsonl"), rec)
        if ok:
            CTX.inc("payloads_generated")
        else:
            logger.warning("FORENSIC llm_payloads append falhou (contabilizado)")

        # Quality audit-only (não modifica score da produção — seção 13)
        try:
            on_quality({
                "event_id": event_id,
                "epoch_ms": epoch_ms,
                "data_quality_score": event_data.get("data_quality_score"),
                "completeness_pct": event_data.get("completeness_pct"),
                "reliability_score": event_data.get("reliability_score"),
                "note": "audit-only mirror, production score untouched",
            })
        except Exception as e_q:
            logger.warning("FORENSIC quality mirror falhou: %s", e_q, exc_info=True)

        # Garantia explícita: NENHUMA chamada LLM aqui (sem import de analyzer/client)
        return final
    except Exception as e:
        try:
            logger.warning("FORENSIC build_and_capture falhou: %s", e, exc_info=True)
        except Exception:
            pass
        return None
