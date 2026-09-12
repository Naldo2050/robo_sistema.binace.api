# audit_live/hooks.py
# -*- coding: utf-8 -*-
"""Hooks observacionais. Cada função NUNCA levanta exceção.

Ativos somente se FORENSIC_CAPTURE=1. Caso contrário retornam imediatamente
(overhead = 1 getenv). Erros internos são contabilizados no WRITER
(audit_writer_errors/dropped) + logging.warning com exc_info — nunca `pass`.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("FORENSIC_CAPTURE", "0") == "1"


def _lazy_ctx():
    from audit_live.forensic_context import CTX

    return CTX


def _lazy_writer():
    from audit_live.forensic_writer import WRITER

    return WRITER


def on_raw_message(message_str: str, raw_obj: Any = None) -> None:
    """Chamar imediatamente após json.loads, ANTES de normalização/clamp/drop."""
    if not _enabled():
        return
    try:
        ctx = _lazy_ctx()
        writer = _lazy_writer()
        base_dir = ctx.ensure_init(os.getenv("FORENSIC_DIR", ""))
        received_at_ms = int(time.time() * 1000)
        try:
            mono_ns = time.monotonic_ns()
        except Exception:
            mono_ns = None
        # raw_obj já decodificado pelo caller; se None, tenta decodificar sem validar
        trade = None
        try:
            trade = raw_obj if isinstance(raw_obj, dict) else json.loads(message_str)
        except Exception as e_dec:
            # Mensagem inválida também é evidência (não descartar silenciosamente)
            writer.append_jsonl(
                ctx.path("audit.log"),
                {"ts_ms": received_at_ms, "event": "raw_decode_failed",
                 "capture_run_id": ctx.capture_run_id, "session_id": ctx.session_id,
                 "error": str(e_dec)[:300]},
            )
            ctx.inc("raw_messages_received")
            return
        envelope = trade.get("data", trade) if isinstance(trade, dict) else {}
        if not isinstance(envelope, dict):
            envelope = {}
        # Preservar EXATAMENTE os campos Binance quando presentes; ausentes -> None (NÃO zero)
        rec: Dict[str, Any] = {
            "capture_run_id": ctx.capture_run_id,
            "session_id": ctx.session_id,
            "received_at_ms": received_at_ms,
            "monotonic_ns": mono_ns,
            "e": envelope.get("e"),
            "E": envelope.get("E"),
            "s": envelope.get("s"),
            "a": envelope.get("a"),
            "p": envelope.get("p"),
            "q": envelope.get("q"),
            "f": envelope.get("f"),
            "l": envelope.get("l"),
            "T": envelope.get("T"),
            "m": envelope.get("m"),
            "M": envelope.get("M"),
        }
        ctx.inc("raw_messages_received")
        # Contabilizar aggTrade (e==aggTrade ou possui 'a'); demais mensagens contam como raw mas não agg
        try:
            is_agg = (envelope.get("e") == "aggTrade") or ("a" in envelope)
        except Exception as e_flag:
            logger.warning("FORENSIC flag agg falhou: %s", e_flag, exc_info=True)
            is_agg = False
        if is_agg:
            ctx.inc("aggtrades_received")
            try:
                flags = ctx.observe_aggtrade_id(envelope.get("a"), envelope.get("T"))
                rec["forensic_flags"] = flags
            except Exception as e_obs:
                logger.warning("FORENSIC observe_aggtrade_id falhou: %s", e_obs, exc_info=True)
                rec["forensic_flags"] = {"observer_error": str(e_obs)[:200]}
        ok = writer.append_jsonl(ctx.path("raw_aggtrades.jsonl"), rec)
        if not ok:
            logger.warning("FORENSIC raw_aggtrades append falhou (contabilizado no writer)")
    except Exception as e:
        # Última barreira: nunca derrubar o bot, mas nunca silenciar
        try:
            logger.warning("FORENSIC on_raw_message falhou: %s", e, exc_info=True)
        except Exception:
            pass


def on_normalized(norm: Optional[Dict[str, Any]], drop_reason: Optional[str] = None) -> None:
    """Chamar após construção de `norm` OU no ponto de descarte (norm=None + reason)."""
    if not _enabled():
        return
    try:
        ctx = _lazy_ctx()
        writer = _lazy_writer()
        ctx.ensure_init(os.getenv("FORENSIC_DIR", ""))
        if norm is None:
            ctx.inc("invalid_trades")
            writer.append_jsonl(
                ctx.path("normalized_trades.jsonl"),
                {"capture_run_id": ctx.capture_run_id, "session_id": ctx.session_id,
                 "dropped": True, "drop_reason": drop_reason or "unknown",
                 "observed_at_ms": int(time.time() * 1000)},
            )
            return
        ctx.inc("normalized_trades")
        # Clone raso observacional (não muta o original)
        try:
            rec = dict(norm)
        except Exception as e_clone:
            logger.warning("FORENSIC clone norm falhou: %s", e_clone, exc_info=True)
            rec = {"clone_error": str(e_clone)[:200]}
        rec["capture_run_id"] = ctx.capture_run_id
        rec["session_id"] = ctx.session_id
        rec["observed_at_ms"] = int(time.time() * 1000)
        ok = writer.append_jsonl(ctx.path("normalized_trades.jsonl"), rec)
        if not ok:
            logger.warning("FORENSIC normalized append falhou (contabilizado)")
    except Exception as e:
        try:
            logger.warning("FORENSIC on_normalized falhou: %s", e, exc_info=True)
        except Exception:
            pass


def on_orderbook_record(record: Dict[str, Any]) -> None:
    """record deve conter record_type: REST_SNAPSHOT | WS_DEPTH_UPDATE | RESYNC | RECONNECT."""
    if not _enabled():
        return
    try:
        ctx = _lazy_ctx()
        writer = _lazy_writer()
        ctx.ensure_init(os.getenv("FORENSIC_DIR", ""))
        ctx.inc("orderbook_updates")
        if record.get("record_type") in ("RESYNC", "RECONNECT"):
            ctx.inc("orderbook_resync_count")
        rec = dict(record)
        rec.setdefault("capture_run_id", ctx.capture_run_id)
        rec.setdefault("session_id", ctx.session_id)
        rec.setdefault("observed_at_ms", int(time.time() * 1000))
        ok = writer.append_jsonl(ctx.path("raw_orderbook.jsonl"), rec)
        if not ok:
            logger.warning("FORENSIC orderbook append falhou (contabilizado)")
    except Exception as e:
        try:
            logger.warning("FORENSIC on_orderbook_record falhou: %s", e, exc_info=True)
        except Exception:
            pass


def on_window(record: Dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        ctx = _lazy_ctx()
        writer = _lazy_writer()
        ctx.ensure_init(os.getenv("FORENSIC_DIR", ""))
        ctx.inc("windows_generated")
        rec = dict(record)
        rec.setdefault("capture_run_id", ctx.capture_run_id)
        rec.setdefault("session_id", ctx.session_id)
        ok = writer.append_jsonl(ctx.path("windows.jsonl"), rec)
        if not ok:
            logger.warning("FORENSIC windows append falhou (contabilizado)")
        # Dump periódico de manifest_base (a cada 10 janelas) para não perder contadores em crash
        try:
            if ctx.snapshot_counters().get("windows_generated", 0) % 10 == 0:
                base = ctx.build_manifest(writer_snapshot=writer.snapshot_counters())
                import json as _json

                with open(ctx.path("manifest_base.json"), "w", encoding="utf-8") as _f:
                    _json.dump(base, _f, ensure_ascii=False, indent=2)
        except Exception as _fe_m:
            logger.warning("FORENSIC manifest_base dump falhou: %s", _fe_m, exc_info=True)
    except Exception as e:
        try:
            logger.warning("FORENSIC on_window falhou: %s", e, exc_info=True)
        except Exception:
            pass


def on_trigger(record: Dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        ctx = _lazy_ctx()
        writer = _lazy_writer()
        ctx.ensure_init(os.getenv("FORENSIC_DIR", ""))
        ctx.inc("analysis_triggers_generated")
        # Deepcopy defensivo para não reter referência mutável
        try:
            rec = copy.deepcopy(record)
        except Exception as e_copy:
            logger.warning("FORENSIC deepcopy trigger falhou: %s", e_copy, exc_info=True)
            rec = {"trigger_copy_error": str(e_copy)[:300]}
        ok = writer.append_jsonl(ctx.path("analysis_triggers.jsonl"), rec)
        if not ok:
            logger.warning("FORENSIC trigger append falhou (contabilizado)")
    except Exception as e:
        try:
            logger.warning("FORENSIC on_trigger falhou: %s", e, exc_info=True)
        except Exception:
            pass


def on_indicator_inputs(record: Dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        ctx = _lazy_ctx()
        writer = _lazy_writer()
        ctx.ensure_init(os.getenv("FORENSIC_DIR", ""))
        rec = dict(record)
        rec.setdefault("capture_run_id", ctx.capture_run_id)
        ok = writer.append_jsonl(ctx.path("indicator_inputs.jsonl"), rec)
        if not ok:
            logger.warning("FORENSIC indicator_inputs append falhou (contabilizado)")
    except Exception as e:
        try:
            logger.warning("FORENSIC on_indicator_inputs falhou: %s", e, exc_info=True)
        except Exception:
            pass


def on_external_api(record: Dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        ctx = _lazy_ctx()
        writer = _lazy_writer()
        ctx.ensure_init(os.getenv("FORENSIC_DIR", ""))
        if not record.get("success", True):
            ctx.inc("external_api_errors")
        rec = dict(record)
        rec.setdefault("capture_run_id", ctx.capture_run_id)
        ok = writer.append_jsonl(ctx.path("external_apis.jsonl"), rec)
        if not ok:
            logger.warning("FORENSIC external_apis append falhou (contabilizado)")
    except Exception as e:
        try:
            logger.warning("FORENSIC on_external_api falhou: %s", e, exc_info=True)
        except Exception:
            pass


def on_quality(record: Dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        ctx = _lazy_ctx()
        writer = _lazy_writer()
        ctx.ensure_init(os.getenv("FORENSIC_DIR", ""))
        rec = dict(record)
        rec.setdefault("capture_run_id", ctx.capture_run_id)
        ok = writer.append_jsonl(ctx.path("quality_events.jsonl"), rec)
        if not ok:
            logger.warning("FORENSIC quality append falhou (contabilizado)")
    except Exception as e:
        try:
            logger.warning("FORENSIC on_quality falhou: %s", e, exc_info=True)
        except Exception:
            pass


def on_payload_stage(stage: str, event_id: Any, epoch_ms: Any, payload_obj: Any) -> None:
    """Captura PRE/POST builder/compressor/guardrail. stage em {PRE_BUILDER,POST_BUILDER,...}."""
    if not _enabled():
        return
    try:
        ctx = _lazy_ctx()
        writer = _lazy_writer()
        ctx.ensure_init(os.getenv("FORENSIC_DIR", ""))
        try:
            size = len(json.dumps(payload_obj, ensure_ascii=False, default=str).encode("utf-8"))
        except Exception as e_size:
            logger.warning("FORENSIC payload size falhou: %s", e_size, exc_info=True)
            size = -1
        rec = {
            "capture_run_id": ctx.capture_run_id,
            "session_id": ctx.session_id,
            "stage": stage,
            "event_id": event_id,
            "epoch_ms": epoch_ms,
            "payload_size_bytes": size,
            "llm_transmitted": False,
            "observed_at_ms": int(time.time() * 1000),
            "payload": payload_obj,
        }
        ok = writer.append_jsonl(ctx.path("payload_stages.jsonl"), rec)
        if not ok:
            logger.warning("FORENSIC payload_stage append falhou (contabilizado)")
    except Exception as e:
        try:
            logger.warning("FORENSIC on_payload_stage falhou: %s", e, exc_info=True)
        except Exception:
            pass
