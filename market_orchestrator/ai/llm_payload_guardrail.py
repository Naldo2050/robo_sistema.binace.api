# -*- coding: utf-8 -*-
"""
Guardrail para impedir envio de payload completo/sensível à LLM.

CORREÇÕES APLICADAS:
    BUG1: candidates buscava ai_payload em lugares errados
    BUG2: historical_vp estava em FORBIDDEN_KEYS indevidamente
    BUG3: payload retornado dentro de wrapper desnecessário
    BUG4: todo payload normal era bloqueado (35.7% block rate)
    BUG5: detecção de compressão prévia era frágil
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from market_orchestrator.ai.payload_compressor import compress_payload
from market_orchestrator.ai.payload_metrics_aggregator import append_metric_line


# ── Chaves que NUNCA devem ser enviadas ao LLM ──────────────────────
# Critério: dados RAW duplicados que já foram processados
# e estão presentes de forma comprimida no ai_payload
#
# REMOVIDOS da lista original:
#   - "historical_vp"   → contém POC/VAH/VAL usados pelo compressor VP
#   - "ANALYSIS_TRIGGER"→ esse campo não existe no payload real
#
# MANTIDOS:
#   - "raw_event"          → cópia bruta do evento (~30KB) já processada
#   - "contextual_snapshot"→ duplicata de enriched_snapshot (~15KB)
#   - "observability"      → métricas internas de performance
#   - "enriched_snapshot"  → duplicata processada pelo pipeline
FORBIDDEN_KEYS = {
    "raw_event",
    "contextual_snapshot",
    "observability",
    "enriched_snapshot",
}

# ── Chaves que identificam payload JÁ comprimido ────────────────────
# Usado para evitar dupla compressão
_COMPRESSED_MARKERS = {
    "_v",           # versão do compressor (v2)
    "epoch_ms",     # payload builder sempre inclui
    "price",        # compressor v1 usa "price"
    "ob",           # compressor v1 usa "ob" (orderbook)
    "flow",         # compressor v1 usa "flow"
}

# ── Tamanho máximo permitido para envio ao LLM ──────────────────────
_MAX_BYTES_LLM = 6144  # 6KB

# ── Tamanho a partir do qual o guardrail age ─────────────────────────
# Payloads menores que isso provavelmente já estão comprimidos
_GUARDRAIL_THRESHOLD_BYTES = 8192  # 8KB


def _is_already_compressed(payload: Dict[str, Any]) -> bool:
    """
    Verifica se o payload já foi comprimido pelo builder ou compressor.

    CORREÇÃO BUG5:
        Antes verificava apenas "price" e "quant" que podem não existir.
        Agora verifica múltiplos marcadores para detecção robusta.

    Returns:
        True se payload já está comprimido e não precisa de novo compress
    """
    if not isinstance(payload, dict):
        return False

    # Versão explícita do compressor
    if payload.get("_v") == 2:
        return True

    # Payload do builder: tem epoch_ms como int + symbol
    has_epoch = isinstance(payload.get("epoch_ms"), (int, float))
    has_symbol = isinstance(payload.get("symbol"), str)
    if has_epoch and has_symbol:
        # Verifica se tem seções comprimidas típicas
        compressed_sections = sum(
            1 for key in ("price", "ob", "flow", "tf", "vp", "quant")
            if key in payload
        )
        if compressed_sections >= 3:
            return True

    # Payload compacto do compressor v1
    has_price = "price" in payload
    has_ob = "ob" in payload
    has_flow = "flow" in payload
    if has_price and has_ob and has_flow:
        return True

    return False


def _log_guardrail(
    leak_blocked: bool,
    bytes_before: int,
    bytes_after: int,
    root: str,
    error: Optional[str] = None,
) -> None:
    """Loga métricas do guardrail para monitoramento."""
    metrics = {
        "payload_bytes": bytes_before,
        "leak_blocked": leak_blocked,
        "bytes_after": bytes_after,
        "payload_root_name": root,
    }
    if error:
        metrics["error"] = error

    line = json.dumps(metrics, ensure_ascii=False)
    logging.debug(line)
    append_metric_line(metrics)


def _extract_safe_candidate(
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extrai o payload seguro de dentro do evento completo.

    CORREÇÃO BUG1:
        Antes buscava em lugares que nunca existiam:
            - payload["AI_ANALYSIS"]["ai_payload"]  → nunca existe
            - payload["enriched_snapshot"]["ai_payload"] → nunca existe

        Agora busca na ordem correta de prioridade:
            1. payload["ai_payload"]     → builder já processou ✅
            2. payload direto            → se já é um payload comprimido
            3. payload["signal"]         → fallback alternativo

    Returns:
        Dict com payload seguro ou None se não encontrado
    """
    # ── Prioridade 1: ai_payload já construído pelo builder ──────────
    ai_payload = payload.get("ai_payload")
    if isinstance(ai_payload, dict) and ai_payload:
        logging.debug(
            "GUARDRAIL_CANDIDATE_FOUND source=ai_payload keys=%s",
            list(ai_payload.keys())[:5]
        )
        return ai_payload

    # ── Prioridade 2: payload direto já comprimido ───────────────────
    # Caso onde o payload já é o dado comprimido (sem wrapper)
    if _is_already_compressed(payload):
        logging.debug("GUARDRAIL_CANDIDATE_FOUND source=payload_direct")
        return payload

    # ── Prioridade 3: dentro de "signal" ────────────────────────────
    signal_data = payload.get("signal")
    if isinstance(signal_data, dict):
        inner_ai = signal_data.get("ai_payload")
        if isinstance(inner_ai, dict) and inner_ai:
            logging.debug("GUARDRAIL_CANDIDATE_FOUND source=signal.ai_payload")
            return inner_ai

    # ── Prioridade 4: construir payload mínimo dos dados disponíveis ─
    # Tenta montar um payload mínimo funcional antes de abortar
    symbol = payload.get("symbol") or payload.get("ativo") or "BTCUSDT"
    price = (
        payload.get("preco_fechamento")
        or payload.get("price")
        or (payload.get("ohlc") or {}).get("close")
    )
    epoch = payload.get("epoch_ms") or payload.get("timestamp_ms")

    if symbol and price and epoch:
        minimal = {
            "symbol": symbol,
            "epoch_ms": int(epoch) if isinstance(epoch, (int, float)) else epoch,
            "price": {"c": price},
            "trigger": payload.get("tipo_evento", "UNKNOWN"),
        }
        logging.warning(
            "GUARDRAIL_MINIMAL_PAYLOAD symbol=%s price=%s epoch=%s",
            symbol, price, epoch
        )
        return minimal

    return None


def ensure_safe_llm_payload(
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Guardrail HARD: evita enviar payloads com campos sensíveis para a LLM.

    FLUXO CORRETO:
        1. Verifica se payload tem chaves proibidas (raw_event, etc.)
        2. Se SIM → extrai ai_payload já processado pelo builder
        3. Se ai_payload já está comprimido → usa direto
        4. Se não comprimido → comprime antes de enviar
        5. Retorna payload LIMPO e comprimido para o LLM

    CORREÇÃO BUG3:
        Antes retornava: {"ai_payload": compressed, "tipo_evento": ...}
        O LLM recebia o payload DENTRO de um wrapper desnecessário.
        Agora retorna o payload DIRETO sem wrapper extra.

    CORREÇÃO BUG4:
        Antes bloqueava 35.7% dos payloads normais porque
        qualquer signal tem raw_event e contextual_snapshot.
        Agora o guardrail age de forma proporcional:
        - Payload pequeno (<8KB) e sem forbidden_keys → passa direto
        - Payload grande com forbidden_keys → extrai e comprime

    Args:
        payload: Dicionário com dados do mercado para análise

    Returns:
        Dict com payload seguro para o LLM
        None se não for possível construir payload seguro (aborta análise)
    """
    # ── Validação básica ─────────────────────────────────────────────
    if not isinstance(payload, dict):
        _log_guardrail(True, 0, 0, "invalid_payload", "payload_not_dict")
        return None

    bytes_before = len(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    )

    # ── Verifica chaves proibidas ────────────────────────────────────
    leak_keys = FORBIDDEN_KEYS.intersection(payload.keys())

    # ── CASO 1: Payload limpo e pequeno → passa direto ───────────────
    if not leak_keys and bytes_before <= _MAX_BYTES_LLM:
        logging.debug(
            "GUARDRAIL_PASS_THROUGH bytes=%s keys_count=%s",
            bytes_before,
            len(payload)
        )
        _log_guardrail(False, bytes_before, bytes_before, "clean_payload")
        return payload

    # ── CASO 2: Payload limpo mas grande → comprime se necessário ────
    if not leak_keys and bytes_before > _MAX_BYTES_LLM:
        if _is_already_compressed(payload):
            # Já comprimido mas ainda grande → aceita assim mesmo
            logging.info(
                "GUARDRAIL_LARGE_BUT_COMPRESSED bytes=%s",
                bytes_before
            )
            _log_guardrail(False, bytes_before, bytes_before, "compressed_payload")
            return payload
        else:
            # Comprime payload limpo que está grande
            try:
                compressed = compress_payload(payload, max_bytes=_MAX_BYTES_LLM)
                bytes_after = len(
                    json.dumps(compressed, ensure_ascii=False).encode("utf-8")
                )
                logging.info(
                    "GUARDRAIL_COMPRESSED_CLEAN bytes=%s→%s",
                    bytes_before, bytes_after
                )
                _log_guardrail(False, bytes_before, bytes_after, "clean_compressed")
                return compressed
            except Exception as e:
                logging.warning(
                    "GUARDRAIL_COMPRESS_FAILED error=%s returning_original",
                    e
                )
                return payload

    # ── CASO 3: Payload com chaves proibidas → extrai parte segura ───
    logging.debug(
        "GUARDRAIL_FORBIDDEN_KEYS_FOUND keys=%s bytes=%s",
        list(leak_keys),
        bytes_before
    )

    # Extrai candidato seguro
    safe_candidate = _extract_safe_candidate(payload)

    # Se não encontrou candidato → aborta
    if safe_candidate is None:
        logging.error(
            "GUARDRAIL_ABORT forbidden_keys=%s bytes=%s "
            "reason=no_safe_candidate_found",
            list(leak_keys),
            bytes_before
        )
        _log_guardrail(
            True, bytes_before, 0, "event", "no_safe_candidate"
        )
        return None

    # ── Comprime se necessário ───────────────────────────────────────
    if _is_already_compressed(safe_candidate):
        # Já comprimido → usa direto
        compressed = safe_candidate
        logging.debug(
            "GUARDRAIL_SKIP_COMPRESSION already_compressed=True"
        )
    else:
        # Precisa comprimir antes de enviar
        try:
            compressed = compress_payload(
                safe_candidate, max_bytes=_MAX_BYTES_LLM
            )
        except Exception as e:
            logging.warning(
                "GUARDRAIL_COMPRESS_ERROR error=%s using_candidate_direct",
                e
            )
            compressed = safe_candidate

    bytes_after = len(
        json.dumps(compressed, ensure_ascii=False).encode("utf-8")
    )

    # ── Log do bloqueio ──────────────────────────────────────────────
    logging.debug(
        "FULL_PAYLOAD_LEAK_BLOCKED root=event "
        "bytes_before=%s bytes_after=%s "
        "forbidden_keys=%s",
        bytes_before,
        bytes_after,
        list(leak_keys),
    )
    _log_guardrail(True, bytes_before, bytes_after, "event")

    # ── CORREÇÃO BUG3: retorna payload DIRETO sem wrapper ────────────
    # Antes: {"ai_payload": compressed, "tipo_evento": ...}
    # Agora:  compressed direto (o LLM recebe os dados sem wrapper)
    #
    # Campos de contexto são injetados DENTRO do compressed
    # apenas se ele não os tiver (para não sobrescrever dados reais)
    for k in ("tipo_evento", "descricao", "symbol", "ativo"):
        if k in payload and k not in compressed:
            compressed[k] = payload[k]

    return compressed
