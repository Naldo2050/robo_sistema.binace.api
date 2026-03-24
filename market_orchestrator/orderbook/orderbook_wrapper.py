# orderbook/orderbook_wrapper.py
# -*- coding: utf-8 -*-

"""
Wrapper para a lógica de análise de orderbook do EnhancedMarketBot.

Equivalente aos métodos:
- _run_orderbook_analyze
- _fetch_orderbook_with_retry
- _refresh_orderbook_async
- _orderbook_fallback

do arquivo market_orchestrator.py ORIGINAL, apenas extraídos para
funções que recebem `bot` como primeiro argumento.
"""

import time
import threading
import asyncio
import logging
import copy  # PATCH W-03: deepcopy para evitar mutação acidental
from typing import Any, Dict, Optional
from collections import deque
from concurrent.futures import TimeoutError as FutureTimeoutError

import config

# ── Fallback counter (rolling window) ──────────────────────
# Registra timestamps de cada fallback para contagem por hora/dia.
_fallback_timestamps: deque = deque(maxlen=1000)
_fallback_lock = threading.Lock()
from orderbook_core.structured_logging import StructuredLogger
from orderbook_core.tracing_utils import TracerWrapper
from orderbook_core.protocols import BotProtocol  # Adicionado import do protocolo

# PATCH 6.3: Import EventFactory para emergency/error
from orderbook_core.event_factory import build_emergency_orderbook_event, build_invalid_orderbook_event


# ============================================================
# 1. EXECUTA OrderBookAnalyzer.analyze() NO LOOP ASSÍNCRONO
# ============================================================

def run_orderbook_analyze(bot: BotProtocol, close_ms: int) -> Optional[Dict[str, Any]]:
    """
    Equivalente a EnhancedMarketBot._run_orderbook_analyze.

    Executa OrderBookAnalyzer.analyze() no loop asyncio dedicado,
    com timeouts interno e externo.
    """

    # Se o bot está em processo de shutdown, não agendar novas corotinas
    if bot.should_stop or getattr(bot, "is_cleaning_up", False):
        return None

    now_ms = int(time.time() * 1000)
    if close_ms <= 0 or close_ms > now_ms + 60_000:
        logging.error(f"❌ close_ms inválido: {close_ms}")
        return None

    loop = getattr(bot, "_async_loop", None)
    if loop is None:
        logging.error("❌ Loop assíncrono não inicializado")
        return None

    # Verifica se loop está rodando
    if loop.is_closed():
        logging.error("❌ Loop assíncrono já foi fechado")
        return None

    # Verifica se thread do loop está viva
    if (
        not getattr(bot, "_async_loop_thread", None)
        or not bot._async_loop_thread.is_alive()
    ):
        logging.error("❌ Thread do loop assíncrono não está rodando")
        return None

    tracer = TracerWrapper(
        service_name="orderbook_service",
        component="wrapper",
        symbol=getattr(bot, "symbol", "UNKNOWN"),
    )

    try:
        with tracer.start_span(
            "run_orderbook_analyze",
            {
                "close_ms": close_ms,
                "window_count": getattr(bot, "window_count", None),
            },
        ):
            # Wrapper assíncrono com timeout interno via asyncio.wait_for
            async def _wrapped_analyze() -> Optional[Dict[str, Any]]:
                try:
                    inner_coro = bot.orderbook_analyzer.analyze(
                        current_snapshot=None,
                        event_epoch_ms=close_ms,
                        window_id=f"W{bot.window_count:04d}",
                    )

                    # Timeout interno da coroutine dentro do loop
                    coro_timeout = float(
                        getattr(config, "ORDERBOOK_CORO_TIMEOUT_SEC", 4.0)
                    )
                    return await asyncio.wait_for(
                        inner_coro,
                        timeout=coro_timeout,
                    )
                except asyncio.TimeoutError:
                    logging.warning(
                        "⏱️ Timeout interno na coroutine de orderbook "
                        "(asyncio.wait_for); resultado será None"
                    )
                    return None

            # Envia a coroutine wrapper para o loop assíncrono dedicado (em outra thread)
            future = asyncio.run_coroutine_threadsafe(
                _wrapped_analyze(),
                loop,
            )

            try:
                # Timeout externo ao aguardar o Future (seguro contra loop travado)
                outer_timeout = float(
                    getattr(config, "ORDERBOOK_FUTURE_TIMEOUT_SEC", 5.0)
                )
                return future.result(timeout=outer_timeout)
            except FutureTimeoutError:
                # Se nem o wrapper respondeu, cancelamos o future explicitamente
                logging.error(
                    "⏱️ Timeout ao aguardar resultado do orderbook "
                    "(async loop) - cancelando Future"
                )
                future.cancel()
                return None

    except Exception as e:
        logging.error(
            f"❌ Erro ao buscar orderbook (async loop): {e}",
            exc_info=True,
        )
        return None


# ============================================================
# 2. FETCH COM RETRY + FALLBACK (PATCH W-02A: salvar sucesso sob lock)
# ============================================================

def fetch_orderbook_with_retry(bot: BotProtocol, close_ms: int) -> Dict[str, Any]:
    """
    Equivalente a EnhancedMarketBot._fetch_orderbook_with_retry.
    """

    try:
        ob_event = run_orderbook_analyze(bot, close_ms)
        if ob_event and ob_event.get("is_valid", False):
            ob_data = ob_event.get("orderbook_data", {}) or {}
            bid_depth = float(ob_data.get("bid_depth_usd", 0.0))
            ask_depth = float(ob_data.get("ask_depth_usd", 0.0))

            min_depth = float(
                getattr(config, "ORDERBOOK_MIN_DEPTH_USD", 500.0)
            )
            if bid_depth >= min_depth or ask_depth >= min_depth:
                # PATCH W-02A: Salvar sucesso sob lock
                with bot._orderbook_refresh_lock:
                    # PATCH W-03: deepcopy para isolamento total
                    bot.last_valid_orderbook = copy.deepcopy(ob_event)
                    bot.last_valid_orderbook_time = time.time()
                    bot.orderbook_fetch_failures = 0
                logging.debug(f"✅ Orderbook OK - Janela #{bot.window_count}")

                # Logger estruturado
                slog = StructuredLogger("orderbook_wrapper", getattr(bot, "symbol", "UNKNOWN"))
                try:
                    slog.info(
                        "orderbook_ok",
                        window_id=bot.window_count,
                        bid_depth_usd=bid_depth,
                        ask_depth_usd=ask_depth,
                        min_depth_usd=min_depth,
                    )
                except Exception:
                    pass

                return ob_event
            else:
                logging.warning(
                    "⚠️ Orderbook com liquidez baixa (best-effort)"
                )
    except Exception as e:
        logging.error(
            f"❌ Erro ao buscar orderbook (best-effort): {e}"
        )

    # fallback + async background refresh
    fallback = orderbook_fallback(bot)
    refresh_orderbook_async(bot, close_ms)
    return fallback


# ============================================================
# 3. CARREGAMENTO EM SEGUNDO PLANO (PATCH 1 - THREAD SAFETY)
# ============================================================

def refresh_orderbook_async(bot: BotProtocol, close_ms: int) -> None:
    if not getattr(bot, "_orderbook_background_refresh", False):
        return

    now = time.time()

    # Tudo dentro do lock para evitar race conditions
    with bot._orderbook_refresh_lock:
        if now - bot._last_async_ob_refresh < bot._orderbook_bg_min_interval:
            return

        thr = getattr(bot, "_orderbook_refresh_thread", None)
        if thr and thr.is_alive():
            return

        bot._last_async_ob_refresh = now

        def _worker() -> None:
            try:
                evt = run_orderbook_analyze(bot, close_ms)
                if evt and evt.get("is_valid", False):
                    # Protege escrita do estado compartilhado
                    with bot._orderbook_refresh_lock:
                        # PATCH W-03: deepcopy para isolamento total
                        bot.last_valid_orderbook = copy.deepcopy(evt)
                        bot.last_valid_orderbook_time = time.time()
                        bot.orderbook_fetch_failures = 0
                    logging.info("♻️ Orderbook cache atualizado em background")
            except Exception as e:
                logging.debug(f"Falha na atualização assíncrona do orderbook: {e}")

        bot._orderbook_refresh_thread = threading.Thread(
            target=_worker,
            daemon=True,
            name=f"OB_Refresh_{getattr(bot, 'window_count', 0):04d}",
        )
        bot._orderbook_refresh_thread.start()


# ============================================================
# 4. FALLBACK DO ORDERBOOK (PATCH W-01, W-02B, W-03, 19)
# ============================================================

def orderbook_fallback(bot: BotProtocol) -> Dict[str, Any]:
    """
    Equivalente a EnhancedMarketBot._orderbook_fallback.
    """

    slog = StructuredLogger("orderbook_wrapper", getattr(bot, "symbol", "UNKNOWN"))

    fallback_max_age = getattr(config, "ORDERBOOK_FALLBACK_MAX_AGE", 600)

    # PATCH W-02B: Capturar estado sob lock
    with bot._orderbook_refresh_lock:
        bot.orderbook_fetch_failures += 1
        failures = bot.orderbook_fetch_failures
        last_evt = bot.last_valid_orderbook
        last_time = bot.last_valid_orderbook_time

    # PATCH W-01: Obter schema_version corretamente
    schema_version = "2.1.0"
    if last_evt and isinstance(last_evt, dict):
        schema_version = last_evt.get("schema_version", "2.1.0")

    # usar cache se não estiver muito velho
    if last_evt and (time.time() - last_time < fallback_max_age):
        age = time.time() - last_time

        # ── Registrar fallback no rolling counter ──
        now = time.time()
        with _fallback_lock:
            _fallback_timestamps.append(now)
            # Contar fallbacks na última hora e nas últimas 24h
            cutoff_1h = now - 3600
            cutoff_24h = now - 86400
            count_1h = sum(1 for t in _fallback_timestamps if t >= cutoff_1h)
            count_24h = sum(1 for t in _fallback_timestamps if t >= cutoff_24h)

        logging.warning(
            f"Usando orderbook em cache (age={age:.0f}s) "
            f"após {failures} falhas | fallbacks: {count_1h}/1h, {count_24h}/24h"
        )

        # PATCH W-03: deepcopy para evitar mutação acidental
        ob_event = copy.deepcopy(last_evt)
        ob_event["data_quality"] = {
            "is_valid": True,
            "data_source": "cache",
            "age_seconds": age,
            "fallbacks_1h": count_1h,
            "fallbacks_24h": count_24h,
        }

        try:
            slog.warning(
                "orderbook_fallback_cache",
                age_seconds=age,
                failures=failures,
                fallbacks_1h=count_1h,
                fallbacks_24h=count_24h,
            )
        except Exception:
            pass

        return ob_event

    # modo emergência
    if getattr(bot, "orderbook_emergency_mode", True):
        logging.warning(
            f"🚨 MODO EMERGÊNCIA: Orderbook indisponível "
            f"(falhas: {failures})"
        )

        symbol = getattr(bot, "symbol", None) or getattr(bot, "market_symbol", None) or "UNKNOWN"
        
        # PATCH 6.3.2: Usar EventFactory para emergency
        thresholds = {
            "ORDERBOOK_CRITICAL_IMBALANCE": getattr(config, "ORDERBOOK_CRITICAL_IMBALANCE", 0.95),
            "ORDERBOOK_MIN_DOMINANT_USD": getattr(config, "ORDERBOOK_MIN_DOMINANT_USD", 2_000_000.0),
            "ORDERBOOK_MIN_RATIO_DOM": getattr(config, "ORDERBOOK_MIN_RATIO_DOM", 20.0),
        }

        engine_version = None
        try:
            if last_evt and isinstance(last_evt, dict):
                engine_version = last_evt.get("engine_version")
        except Exception:
            engine_version = None

        try:
            slog.error(
                "orderbook_emergency_mode",
                failures=failures,
            )
        except Exception:
            pass

        return build_emergency_orderbook_event(
            symbol=symbol,
            schema_version=schema_version,
            engine_version=engine_version,
            ts_ms=int(time.time() * 1000),
            failures=failures,
            bid_depth_usd=1000.0,
            ask_depth_usd=1000.0,
            top_n=getattr(bot, "orderbook_top_n", 20),
            ob_limit=getattr(bot, "orderbook_limit", 100),
            thresholds=thresholds,
            health_stats={
                "emergency_uses": failures,
            },
        )

    # indisponível total → retorna inválido
    logging.error(
        f"❌ Orderbook totalmente indisponível "
        f"(falhas consecutivas: {failures})"
    )

    symbol = getattr(bot, "symbol", None) or getattr(bot, "market_symbol", None) or "UNKNOWN"
    
    # PATCH 6.3.3: Usar EventFactory para indisponibilidade total
    thresholds = {
        "ORDERBOOK_CRITICAL_IMBALANCE": getattr(config, "ORDERBOOK_CRITICAL_IMBALANCE", 0.95),
        "ORDERBOOK_MIN_DOMINANT_USD": getattr(config, "ORDERBOOK_MIN_DOMINANT_USD", 2_000_000.0),
        "ORDERBOOK_MIN_RATIO_DOM": getattr(config, "ORDERBOOK_MIN_RATIO_DOM", 20.0),
    }

    engine_version = None
    try:
        if last_evt and isinstance(last_evt, dict):
            engine_version = last_evt.get("engine_version")
    except Exception:
        engine_version = None

    try:
        slog.error(
            "orderbook_fallback_error",
            failures=failures,
        )
    except Exception:
        pass

    return build_invalid_orderbook_event(
        symbol=symbol,
        schema_version=schema_version,
        engine_version=engine_version,
        ts_ms=int(time.time() * 1000),
        error_msg="Failed after max retries",
        severity="ERROR",
        top_n=getattr(bot, "orderbook_top_n", 20),
        ob_limit=getattr(bot, "orderbook_limit", 100),
        thresholds=thresholds,
        health_stats={
            "emergency_uses": 0,
        },
    )
