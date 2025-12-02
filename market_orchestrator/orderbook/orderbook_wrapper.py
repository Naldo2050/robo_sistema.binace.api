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
from typing import Any, Dict, Optional
from concurrent.futures import TimeoutError as FutureTimeoutError

import config


# ============================================================
# 1. EXECUTA OrderBookAnalyzer.analyze() NO LOOP ASSÍNCRONO
# ============================================================

def run_orderbook_analyze(bot, close_ms: int) -> Optional[Dict[str, Any]]:
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

    try:
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
                logging.error(
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
# 2. FETCH COM RETRY + FALLBACK (ORIGINAL)
# ============================================================

def fetch_orderbook_with_retry(bot, close_ms: int) -> Dict[str, Any]:
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
                bot.last_valid_orderbook = ob_event
                bot.last_valid_orderbook_time = time.time()
                bot.orderbook_fetch_failures = 0
                logging.debug(f"✅ Orderbook OK - Janela #{bot.window_count}")
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
# 3. CARREGAMENTO EM SEGUNDO PLANO (ORIGINAL)
# ============================================================

def refresh_orderbook_async(bot, close_ms: int) -> None:
    if not getattr(bot, "_orderbook_background_refresh", False):
        return

    now = time.time()

    if now - bot._last_async_ob_refresh < bot._orderbook_bg_min_interval:
        return

    with bot._orderbook_refresh_lock:
        if (
            getattr(bot, "_orderbook_refresh_thread", None)
            and bot._orderbook_refresh_thread.is_alive()
        ):
            return

        bot._last_async_ob_refresh = now

        def _worker() -> None:
            try:
                evt = run_orderbook_analyze(bot, close_ms)
                if evt and evt.get("is_valid", False):
                    bot.last_valid_orderbook = evt
                    bot.last_valid_orderbook_time = time.time()
                    bot.orderbook_fetch_failures = 0
                    logging.info("♻️ Orderbook cache atualizado em background")
            except Exception as e:
                logging.debug(
                    f"Falha na atualização assíncrona do orderbook: {e}"
                )

        bot._orderbook_refresh_thread = threading.Thread(
            target=_worker,
            daemon=True,
        )
        bot._orderbook_refresh_thread.start()


# ============================================================
# 4. FALLBACK DO ORDERBOOK (ORIGINAL)
# ============================================================

def orderbook_fallback(bot) -> Dict[str, Any]:
    """
    Equivalente a EnhancedMarketBot._orderbook_fallback.
    """

    bot.orderbook_fetch_failures += 1

    fallback_max_age = getattr(
        config, "ORDERBOOK_FALLBACK_MAX_AGE", 600
    )

    # usar cache se não estiver muito velho
    if bot.last_valid_orderbook and (
        time.time() - bot.last_valid_orderbook_time < fallback_max_age
    ):
        age = time.time() - bot.last_valid_orderbook_time

        logging.warning(
            f"⚠️ Usando orderbook em cache (age={age:.0f}s) "
            f"após {bot.orderbook_fetch_failures} falhas"
        )

        ob_event = bot.last_valid_orderbook.copy()
        ob_event["data_quality"] = {
            "is_valid": True,
            "data_source": "cache",
            "age_seconds": age,
        }
        return ob_event

    # modo emergência
    if getattr(bot, "orderbook_emergency_mode", True):
        logging.warning(
            f"🚨 MODO EMERGÊNCIA: Orderbook indisponível "
            f"(falhas: {bot.orderbook_fetch_failures})"
        )

        return {
            "is_valid": True,
            "emergency_mode": True,
            "orderbook_data": {
                "bid_depth_usd": 1000.0,
                "ask_depth_usd": 1000.0,
                "imbalance": 0.0,
                "mid": 0.0,
                "spread": 0.0,
            },
            "spread_metrics": {
                "bid_depth_usd": 1000.0,
                "ask_depth_usd": 1000.0,
            },
            "data_quality": {
                "is_valid": True,
                "data_source": "emergency",
                "error": (
                    f"Emergency mode after "
                    f"{bot.orderbook_fetch_failures} failures"
                ),
            },
        }

    # indisponível total → retorna inválido
    logging.error(
        f"❌ Orderbook totalmente indisponível "
        f"(falhas consecutivas: {bot.orderbook_fetch_failures})"
    )

    return {
        "is_valid": False,
        "should_skip": False,
        "orderbook_data": {
            "bid_depth_usd": 0.0,
            "ask_depth_usd": 0.0,
            "imbalance": 0.0,
            "mid": 0.0,
            "spread": 0.0,
        },
        "spread_metrics": {
            "bid_depth_usd": 0.0,
            "ask_depth_usd": 0.0,
        },
        "data_quality": {
            "is_valid": False,
            "data_source": "error",
            "error": "Failed after max retries",
        },
    }