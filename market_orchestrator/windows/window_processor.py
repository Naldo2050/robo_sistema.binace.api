# windows/window_processor.py
# -*- coding: utf-8 -*-

"""
Processamento de janela do EnhancedMarketBot.

CÓPIA FIEL do método EnhancedMarketBot._process_window do arquivo
market_orchestrator.py original, apenas movido para este módulo e
trocando `self` por `bot`.

Nenhuma lógica foi alterada.

═══════════════════════════════════════════════════════════════════════════════
ONDE JANELAS SÃO CRIADAS:
────────────────────────────────────────────────────────────────────────────────
As janelas são CRIADAS no arquivo principal: market_orchestrator/market_orchestrator.py

1. Definição da janela (linha ~744-750):
   - `window_end_ms` é inicializado com `_next_boundary_ms(T)` quando chega o primeiro trade
   - Quando `T >= window_end_ms`, a janela é FECHADA e uma nova é calculada

2. O método `_next_boundary_ms(T)` calcula o próximo limite de janela baseado no
   timestamp do trade e no tamanho da janela (`window_ms`)

3. O método `_process_window()` é CHAMADO quando a janela é fechada
   - Este método (em window_processor.py) processa todos os trades acumulados

4. As variáveis de estado das janelas estão em market_orchestrator.py:
   - `self.window_end_ms` (linha 274): timestamp de fechamento da janela
   - `self.window_data` (linha 274): lista de trades na janela atual
   - `self.window_count` (linha 275): contador de janelas processadas

5. O WindowProcessor (esta classe) é apenas um WRAPPER assíncrono que gerencia
   o processamento de janelas, mas a CRIAÇÃO real acontece no método
   `_on_message` do EnhancedMarketBot.
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import config

from ..orderbook.orderbook_wrapper import fetch_orderbook_with_retry
from data_pipeline import DataPipeline
from data_handler import create_absorption_event, create_exhaustion_event
from orderbook_core.structured_logging import StructuredLogger
from orderbook_core.tracing_utils import TracerWrapper


class WindowProcessor:
    """
    Processador de janelas para o EnhancedMarketBot.
    
    Gerencia o processamento de janelas de tempo e mantém um ticker
    interno para processamento periódico.
    """
    
    def __init__(
        self,
        symbol: str,
        windows_minutes: List[int],
        event_bus: Any,
        time_manager: Any,
        logger: Optional[logging.Logger] = None
    ):
        self.symbol = symbol
        self.windows_minutes = windows_minutes
        self.event_bus = event_bus
        self.time_manager = time_manager
        self.logger = logger or logging.getLogger(__name__)
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ticker_task: Optional[asyncio.Task] = None
        self._queue: "queue.Queue[Optional[Tuple[Any, List[Dict[str, Any]], int]]]" = queue.Queue(
            maxsize=32
        )
        self._worker_stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        
        self.window_count = 0
        self._shutdown_event = asyncio.Event()
        
    async def start(self) -> None:
        """Inicia o WindowProcessor criando tarefas internas."""
        if self._running:
            self.logger.warning("WindowProcessor já está rodando")
            return
            
        self._running = True
        self._shutdown_event.clear()
        self._worker_stop.clear()

        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"window_processor_{self.symbol}",
                daemon=True,
            )
            self._worker_thread.start()
        
        # Cria tarefa para processamento de janelas
        self._task = asyncio.create_task(self._run())
        
        # Cria ticker para processamento periódico (se necessário)
        self._ticker_task = asyncio.create_task(self._ticker())
        
        self.logger.info(f"✅ WindowProcessor iniciado para {self.symbol} com janelas={self.windows_minutes}")
        
    async def stop(self) -> None:
        """Para o WindowProcessor e cancela todas as tarefas."""
        if not self._running:
            return
            
        self.logger.info("🛑 Parando WindowProcessor...")
        
        self._running = False
        self._shutdown_event.set()
        self._worker_stop.set()

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=15.0)
            if self._worker_thread.is_alive():
                self.logger.warning("Timeout ao aguardar worker de janelas finalizar")
        self._worker_thread = None
        
        # Cancela tarefas
        tasks = [self._task, self._ticker_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("✅ WindowProcessor parado")
        
    async def _run(self) -> None:
        """Loop principal do processador de janelas."""
        try:
            while self._running:
                await asyncio.sleep(1.0)  # Check every second
                # Aqui seria implementado o processamento periódico se necessário
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Erro no loop do WindowProcessor: {e}")
            
    async def _ticker(self) -> None:
        """Ticker para processamento periódico."""
        try:
            while self._running:
                await asyncio.sleep(60.0)  # Check every minute
                # Aqui seria implementado o processamento periódico se necessário
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Erro no ticker do WindowProcessor: {e}")

    def submit_window(
        self,
        bot: Any,
        window_data: List[Dict[str, Any]],
        close_ms: Optional[int],
    ) -> bool:
        """
        Enfileira uma janela para processamento em background.
        O snapshot evita que o caminho de ingestao fique bloqueado.
        """
        if not window_data or close_ms is None:
            return False

        snapshot = [dict(trade) for trade in window_data if isinstance(trade, dict)]

        try:
            self._queue.put_nowait((bot, snapshot, int(close_ms)))
        except queue.Full:
            self.logger.warning(
                "Fila de janelas cheia (%d pendentes). Processando janela atual de forma síncrona.",
                self._queue.qsize(),
            )
            return False

        backlog = self._queue.qsize()
        if backlog >= 3:
            self.logger.warning("Backlog de janelas pendentes: %d", backlog)
        return True

    def _worker_loop(self) -> None:
        """Worker dedicado para tirar processamento pesado do caminho do WebSocket."""
        while not self._worker_stop.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                continue

            bot, window_data, close_ms = item
            try:
                process_window_snapshot(bot, window_data, close_ms)
                self.window_count += 1
            except Exception as e:
                self.logger.error(f"Erro no worker de janelas: {e}", exc_info=True)
            finally:
                self._queue.task_done()


def process_window(bot) -> None:
    """
    Corpo original de EnhancedMarketBot._process_window,
    recebendo `bot` como primeiro argumento.
    Agora com logging estruturado e tracing por janela.
    """

    if not bot.window_data or bot.should_stop:
        bot.window_data = []
        return

    window_data = [dict(trade) for trade in bot.window_data if isinstance(trade, dict)]
    close_ms = bot.window_end_ms

    try:
        process_window_snapshot(bot, window_data, close_ms)
    finally:
        bot.window_data = []


def process_window_snapshot(
    bot,
    window_data: List[Dict[str, Any]],
    close_ms: Optional[int],
) -> None:
    """
    Processa uma janela a partir de um snapshot imutável.
    Usado pelo worker em background e pelo caminho síncrono de fallback.
    """
    if not window_data or bot.should_stop or close_ms is None:
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
            return

    # ----------------------------
    # Normalização dos trades
    # ----------------------------
    valid_window_data: List[Dict[str, Any]] = []
    for trade in window_data:
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
        return

    total_volume = sum(
        float(trade.get("q", 0)) for trade in valid_window_data
    )
    if total_volume == 0:
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

    # ============================
    # Structured Logger + Tracer
    # ============================
    slog = getattr(bot, "slog", None)
    if slog is None:
        slog = StructuredLogger("enhanced_market_bot", getattr(bot, "symbol", "UNKNOWN"))

    tracer = getattr(bot, "tracer", None)
    if tracer is None:
        tracer = TracerWrapper(
            service_name="enhanced_market_bot",
            component="window_processor",
            symbol=getattr(bot, "symbol", "UNKNOWN"),
        )

    with tracer.start_span(
        "process_window",
        {
            "window_count": bot.window_count,
            "trade_count": len(valid_window_data),
            "total_volume": float(total_volume),
        },
    ):
        try:
            # Heartbeat principal
            try:
                bot.health_monitor.heartbeat("main")
            except Exception:
                pass

            # PATCH 3: cálculo robusto de dynamic_delta_threshold (NaN/Inf-safe)
            dynamic_delta_threshold = 2.0 # Default warmup (BTC)
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

                    # PATCH: Impor floor mínimo para não silenciar sinais (ex: BTC delta < 0.5 é insignificante)
                    dynamic_delta_threshold = max(0.5, dynamic_delta_threshold)

                    # Valida resultado
                    if np.isnan(dynamic_delta_threshold) or np.isinf(dynamic_delta_threshold):
                        dynamic_delta_threshold = 1.0  # Safe default if math fails
                        logging.warning("⚠️ Threshold calculado inválido, usando 1.0")

                except Exception as e:
                    logging.error(f"Erro ao calcular threshold: {e}")
                    dynamic_delta_threshold = 1.0

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

            try:
                from ml.model_inference import predict_up_probability
            except ImportError:
                predict_up_probability = None

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

            # ----------------------------
            # IA Quantitativa: features finais + probabilidade de alta
            # ----------------------------
            final_features = None
            ml_prob_up = None

            try:
                final_features = pipeline.get_final_features()
            except Exception as e:
                logging.error(
                    f"Erro ao gerar features finais para ML: {e}",
                    exc_info=True,
                )

            _warmup_meta: dict = {}

            if final_features is not None and predict_up_probability is not None:
                try:
                    if not hasattr(bot, 'feature_calc'):
                        from ml.feature_calculator import LiveFeatureCalculator
                        bot.feature_calc = LiveFeatureCalculator()

                    if valid_window_data:
                        # Bug 1 fix: amostrar ~60 trades uniformemente em vez de todos os ~2030.
                        # Alimentar todos os ticks faz RSI calcular em tick-level → valores extremos
                        # (ex: RSI=98 com variação de price <0.01%). Amostragem simula candles de ~1s.
                        trade_count = len(valid_window_data)
                        if trade_count > 60:
                            step = max(1, trade_count // 60)
                            sampled_trades = valid_window_data[::step]
                        else:
                            sampled_trades = valid_window_data

                        for trade in sampled_trades:
                            bot.feature_calc.update(
                                price=float(trade.get("p", 0.0)),
                                volume=float(trade.get("q", 0.0))
                            )

                    computed_features = bot.feature_calc.compute(
                        multi_tf=macro_context.get("mtf_trends")
                    )

                    # Bug 2 fix: separar metadados '_' antes de passar ao XGBoost.
                    # Antes: {**final_features, **computed_features} passava _warmup_ready etc. ao modelo.
                    _warmup_meta = {k: v for k, v in computed_features.items() if k.startswith('_')}
                    model_features_only = {k: v for k, v in computed_features.items() if not k.startswith('_')}

                    if isinstance(final_features, dict):
                        enriched_features = {**final_features, **model_features_only}
                    else:
                        enriched_features = model_features_only

                    # Injetar multi_tf para RSI fallback via FEATURE_MAP
                    mtf = macro_context.get("mtf_trends")
                    if mtf and isinstance(enriched_features, dict):
                        enriched_features.setdefault("multi_tf", mtf)

                    ml_prob_up = predict_up_probability(enriched_features)
                except Exception as e:
                    logging.error(
                        f"Erro ao executar predição do modelo de ML: {e}",
                        exc_info=True,
                    )

            if ml_prob_up is not None:
                try:
                    quant_ctx = macro_context.setdefault("quant_model", {})
                    quant_ctx["prob_up"] = float(ml_prob_up)

                    # Bug 3 fix: propagar metadados de warmup ao quant_ctx para que
                    # hybrid_decision saiba se a predição é confiável (warmup_insufficient).
                    if _warmup_meta:
                        quant_ctx["_warmup_ready"] = _warmup_meta.get("_warmup_ready", True)
                        quant_ctx["_ml_usable"] = _warmup_meta.get("_ml_usable", True)
                        quant_ctx["_features_real_count"] = _warmup_meta.get("_features_real_count", 9)
                        quant_ctx["_features_default_list"] = _warmup_meta.get("_features_default_list", [])

                    logging.info(
                        f"[QUANT] prob_up={ml_prob_up:.3f} para janela {bot.window_count}"
                    )
                except Exception as e:
                    logging.error(
                        f"Erro ao atualizar macro_context com probabilidade quantitativa: {e}",
                        exc_info=True,
                    )

            # Garantir que Timeframe esteja nos sinais — nunca injetar dados neutros falsos
            if signals:
                _mtf_real = macro_context.get("mtf_trends", {})
                for s in signals:
                    if not s.get("multi_tf") and not s.get("tf"):
                        if _mtf_real:
                            s["multi_tf"] = _mtf_real
                        # Se mtf_trends vazio, deixar sem multi_tf — melhor que dados falsos

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

            # ----------------------------
            # Persistência de features para ML / backtesting
            # ----------------------------
            try:
                feature_store = getattr(bot, "feature_store", None)
                # Salva enriched_features se disponíveis (incluem as sintéticas do LiveFeatureCalculator),
                # caso contrário usa as final_features do pipeline.
                features_to_save = locals().get('enriched_features', final_features)
                
                if feature_store is not None and features_to_save is not None:
                    window_id = f"{bot.symbol}_{close_ms}"
                    feature_store.save_features(window_id, features_to_save)
            except Exception as e:
                logging.error(
                    f"Erro ao salvar features no FeatureStore: {e}",
                    exc_info=True,
                )

            # ----------------------------
            # Log estruturado de sucesso da janela
            # ----------------------------
            try:
                ob_evt = ob_event if isinstance(ob_event, dict) else {}
                dq = ob_evt.get("data_quality") or {}
                slog.info(
                    "window_processed",
                    window_count=bot.window_count,
                    trade_count=len(valid_window_data),
                    total_volume=float(total_volume),
                    total_buy_volume=float(total_buy_volume),
                    total_sell_volume=float(total_sell_volume),
                    dynamic_delta_threshold=float(dynamic_delta_threshold),
                    orderbook_source=dq.get("data_source"),
                    orderbook_valid=ob_evt.get("is_valid"),
                    signals_count=len(signals) if signals else 0,
                )
            except Exception:
                pass

        except Exception as e:
            logging.error(
                f"Erro no processamento da janela #{bot.window_count}: {e}",
                exc_info=True,
            )
            try:
                slog.error(
                    "window_process_error",
                    window_count=bot.window_count,
                    error=str(e),
                )
            except Exception:
                pass
        finally:
            try:
                if "pipeline" in locals() and hasattr(pipeline, "close"):
                    pipeline.close()
            except Exception as e:
                logging.debug(f"Falha ao fechar pipeline: {e}")
