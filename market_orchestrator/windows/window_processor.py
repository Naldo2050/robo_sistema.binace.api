# windows/window_processor.py
# -*- coding: utf-8 -*-

"""
Processamento de janela do EnhancedMarketBot.

CÓPIA FIEL do método EnhancedMarketBot._process_window do arquivo
market_orchestrator.py original, apenas movido para este módulo e
trocando `self` por `bot`.

Nenhuma lógica foi alterada.
"""

import logging
import time
from typing import Any, Dict, List

import numpy as np
import config

from ..orderbook.orderbook_wrapper import fetch_orderbook_with_retry
from data_pipeline import DataPipeline
from data_handler import create_absorption_event, create_exhaustion_event


def process_window(bot) -> None:
    """
    Corpo original de EnhancedMarketBot._process_window,
    recebendo `bot` como primeiro argumento.
    """

    if not bot.window_data or bot.should_stop:
        bot.window_data = []
        return

    # PATCH 2: warmup thread-safe com lock
    with bot._warmup_lock:
        if bot.warming_up:
            bot.warmup_windows_remaining -= 1

            logging.info(
                f"⏳ AQUECIMENTO: Janela processada "
                f"({bot.warmup_windows_required - bot.warmup_windows_remaining}/"
                f"{bot.warmup_windows_required})"
            )

            if bot.warmup_windows_remaining <= 0:
                bot.warming_up = False
                logging.info("✅ AQUECIMENTO CONCLUÍDO - Sistema pronto!")

            bot.window_data = []
            return

    # ----------------------------
    # Normalização dos trades
    # ----------------------------
    valid_window_data: List[Dict[str, Any]] = []
    for trade in bot.window_data:
        if "q" in trade and "p" in trade and "T" in trade:
            try:
                trade["q"] = float(trade["q"])
                trade["p"] = float(trade["p"])
                trade["T"] = int(trade["T"])
                valid_window_data.append(trade)
            except (ValueError, TypeError):
                continue

    if len(valid_window_data) < bot.min_trades_for_pipeline:
        logging.warning(
            f"⏳ Janela com apenas {len(valid_window_data)} trades "
            f"(mín: {bot.min_trades_for_pipeline}). Aguardando mais dados..."
        )

        if len(bot.trades_buffer) >= bot.min_trades_for_pipeline:
            logging.info(
                f"🔄 Recuperando {bot.min_trades_for_pipeline} trades "
                f"do buffer de emergência..."
            )
            valid_window_data = list(bot.trades_buffer)[
                -bot.min_trades_for_pipeline :
            ]
        else:
            bot.window_data = []
            return

    total_volume = sum(
        float(trade.get("q", 0)) for trade in valid_window_data
    )
    if total_volume == 0:
        bot.window_data = []
        return

    bot.window_count += 1

    # ----------------------------
    # Cálculo de volumes buy/sell
    # ----------------------------
    try:
        quantities = np.array(
            [t.get("q", 0) for t in valid_window_data], dtype=float
        )
        is_sell = np.array(
            [t.get("m", False) for t in valid_window_data], dtype=bool
        )

        total_sell_volume = float(quantities[is_sell].sum())
        total_buy_volume = float(quantities[~is_sell].sum())
    except Exception as e:
        logging.error(
            f"Erro ao calcular volumes com NumPy, usando fallback: {e}"
        )
        total_buy_volume = 0.0
        total_sell_volume = 0.0
        for trade in valid_window_data:
            if trade.get("m"):
                total_sell_volume += float(trade.get("q", 0))
            else:
                total_buy_volume += float(trade.get("q", 0))

    try:
        # Heartbeat principal
        try:
            bot.health_monitor.heartbeat("main")
        except Exception:
            pass

        # PATCH 3: cálculo robusto de dynamic_delta_threshold (NaN/Inf-safe)
        dynamic_delta_threshold = 0.0
        if len(bot.delta_history) > 10:
            try:
                mean_delta = float(np.mean(bot.delta_history))
                std_delta = float(np.std(bot.delta_history))

                # Valida valores
                if np.isnan(mean_delta) or np.isinf(mean_delta):
                    mean_delta = 0.0
                if np.isnan(std_delta) or np.isinf(std_delta):
                    std_delta = 0.0

                dynamic_delta_threshold = abs(
                    mean_delta + bot.delta_std_dev_factor * std_delta
                )

                # Valida resultado
                if np.isnan(dynamic_delta_threshold) or np.isinf(dynamic_delta_threshold):
                    dynamic_delta_threshold = 0.0
                    logging.warning("⚠️ Threshold calculado inválido, usando 0.0")

            except Exception as e:
                logging.error(f"Erro ao calcular threshold: {e}")
                dynamic_delta_threshold = 0.0

        # ----------------------------
        # Macro contexto + VP histórico
        # ----------------------------
        macro_context = bot.context_collector.get_context()
        historical_profile = macro_context.get("historical_vp", {})

        vp_daily = historical_profile.get("daily", {})
        val = vp_daily.get("val", 0)
        vah = vp_daily.get("vah", 0)
        poc = vp_daily.get("poc", 0)

        if val == 0 or vah == 0 or poc == 0:
            if bot.last_valid_vp and (
                time.time() - bot.last_valid_vp_time < 3600
            ):
                age = time.time() - bot.last_valid_vp_time
                logging.warning(
                    f"⚠️ Value Area zerada, usando cache (age={age:.0f}s)"
                )
                historical_profile = bot.last_valid_vp.copy()
            else:
                logging.warning(
                    "⚠️ Value Area indisponível e sem cache válido"
                )
        else:
            bot.last_valid_vp = historical_profile.copy()
            bot.last_valid_vp_time = time.time()

        # Timestamp de fechamento da janela
        close_ms = bot.window_end_ms

        # Atualiza níveis de VP
        bot.levels.update_from_vp(historical_profile)

        # ----------------------------
        # Criação do DataPipeline
        # ----------------------------
        try:
            pipeline = DataPipeline(
                valid_window_data,
                bot.symbol,
                time_manager=bot.time_manager,
            )
        except ValueError as ve:
            logging.error(
                f"❌ Erro ao criar pipeline (janela #{bot.window_count}): {ve}"
            )
            bot.window_data = []
            return

        # Métricas de fluxo (FlowAnalyzer)
        flow_metrics = bot.flow_analyzer.get_flow_metrics(
            reference_epoch_ms=close_ms
        )

        # Orderbook (via wrapper externo)
        ob_event = fetch_orderbook_with_retry(bot, close_ms)

        # Enriquecimento
        enriched = pipeline.enrich()

        # Contexto adicional no pipeline
        pipeline.add_context(
            flow_metrics=flow_metrics,
            historical_vp=historical_profile,
            orderbook_data=ob_event,
            multi_tf=macro_context.get("mtf_trends", {}),
            derivatives=macro_context.get("derivatives", {}),
            market_context=macro_context.get("market_context", {}),
            market_environment=macro_context.get("market_environment", {}),
        )

        # ----------------------------
        # Detecção de sinais
        # ----------------------------
        signals = pipeline.detect_signals(
            absorption_detector=lambda data, sym: create_absorption_event(
                data,
                sym,
                delta_threshold=dynamic_delta_threshold,
                tz_output=bot.ny_tz,
                flow_metrics=flow_metrics,
                historical_profile=historical_profile,
                time_manager=bot.time_manager,
                event_epoch_ms=close_ms,
            ),
            exhaustion_detector=lambda data, sym: create_exhaustion_event(
                data,
                sym,
                history_volumes=list(bot.volume_history),
                volume_factor=config.VOL_FACTOR_EXH,
                tz_output=bot.ny_tz,
                flow_metrics=flow_metrics,
                historical_profile=historical_profile,
                time_manager=bot.time_manager,
                event_epoch_ms=close_ms,
            ),
            orderbook_data=ob_event,
        )

        # Encaminha para o processador de sinais original
        bot._process_signals(
            signals,
            pipeline,
            flow_metrics,
            historical_profile,
            macro_context,
            ob_event,
            enriched,
            close_ms,
            total_buy_volume,
            total_sell_volume,
            valid_window_data,
        )

    except Exception as e:
        logging.error(
            f"Erro no processamento da janela #{bot.window_count}: {e}",
            exc_info=True,
        )
    finally:
        try:
            if "pipeline" in locals() and hasattr(pipeline, "close"):
                pipeline.close()
        except Exception as e:
            logging.debug(f"Falha ao fechar pipeline: {e}")
        bot.window_data = []