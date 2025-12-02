# market_orchestrator/market_orchestrator.py
# -*- coding: utf-8 -*-
"""
Orquestrador de mercado (WebSocket + janelas + DataPipeline + IA) v2.3.2

Versão refatorada em módulos, preservando o comportamento do arquivo
original market_orchestrator.py. Toda a lógica continua igual, apenas
algumas partes foram extraídas para submódulos.
"""

import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from collections import deque
import signal
import atexit
import asyncio
from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import config

# ====== Clock Sync (opcional) ======
try:
    from clock_sync import get_clock_sync
    HAS_CLOCK_SYNC = True
except ImportError:
    HAS_CLOCK_SYNC = False
    logging.warning("⚠️ clock_sync.py não encontrado - timestamps usarão relógio local")

# ====== Utilitários de formatação ======
from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific,
)

# ====== Módulos internos originais ======
from data_handler import (
    create_absorption_event,
    create_exhaustion_event,
    NY_TZ,
)
from event_memory import (
    obter_memoria_eventos,
    adicionar_memoria_evento,
    calcular_probabilidade_historica,
)
from orderbook_analyzer import OrderBookAnalyzer
from event_saver import EventSaver
from context_collector import ContextCollector
from flow_analyzer import FlowAnalyzer
from ai_analyzer_qwen import AIAnalyzer
from report_generator import generate_ai_analysis_report  # pode ser usado futuramente
from levels_registry import LevelRegistry
from data_validator import validator

from time_manager import TimeManager
from health_monitor import HealthMonitor
from event_bus import EventBus
from data_pipeline import DataPipeline
from feature_store import FeatureStore

# ====== Alert engine (opcional) ======
try:
    from alert_engine import generate_alerts
except Exception:
    generate_alerts = None

try:
    from support_resistance import detect_support_resistance, defense_zones
except Exception:
    detect_support_resistance = None
    defense_zones = None

try:
    from pattern_recognition import recognize_patterns
except Exception:
    recognize_patterns = None

# ====== Submódulos refatorados ======
from .utils.logging_utils import configure_dedup_logs
from .connection.robust_connection import RobustConnectionManager, RateLimiter
from .flow.trade_flow_analyzer import TradeFlowAnalyzer
from .orderbook.orderbook_wrapper import fetch_orderbook_with_retry
from .ai.ai_runner import initialize_ai_async, run_ai_analysis_threaded
from .windows.window_processor import process_window
from .signals.signal_processor import process_signals

# Ativa filtro anti-eco global
configure_dedup_logs()


class EnhancedMarketBot:
    """Bot de análise de mercado com IA integrada (v2.3.2)."""

    def __init__(
        self,
        stream_url: str,
        symbol: str,
        window_size_minutes: int,
        vol_factor_exh: float,
        history_size: int,
        delta_std_dev_factor: float,
        context_sma_period: int,
        liquidity_flow_alert_percentage: float,
        wall_std_dev_factor: float,
    ) -> None:
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.window_ms = window_size_minutes * 60 * 1000
        self.ny_tz = NY_TZ
        self.should_stop = False
        self.is_cleaning_up = False

        # Locks e sinalização de shutdown/cleanup
        self._cleanup_lock = threading.Lock()
        self._cleanup_started = threading.Event()
        self._ai_init_lock = threading.Lock()

        self.warming_up = False
        self._warmup_lock = threading.Lock()
        self.warmup_windows_remaining = 0
        self.warmup_windows_required = getattr(config, "WARMUP_WINDOWS", 3)

        self.trades_buffer: deque[Dict[str, Any]] = deque(
            maxlen=getattr(config, "TRADES_BUFFER_SIZE", 2000)
        )
        self.min_trades_for_pipeline = getattr(
            config, "MIN_TRADES_FOR_PIPELINE", 10
        )

        # Clock Sync
        self.clock_sync = None
        if HAS_CLOCK_SYNC:
            try:
                self.clock_sync = get_clock_sync()
                logging.info("✅ Clock Sync inicializado")
            except Exception as e:
                logging.error(f"❌ Erro ao inicializar Clock Sync: {e}")

        self.time_manager = TimeManager()

        self.health_monitor = HealthMonitor()
        self.event_bus = EventBus()
        self.feature_store = FeatureStore()
        self.levels = LevelRegistry(self.symbol)

        self.health_monitor.heartbeat("main")

        self.trade_flow_analyzer = TradeFlowAnalyzer(
            vol_factor_exh, tz_output=self.ny_tz
        )

        self.orderbook_analyzer = OrderBookAnalyzer(
            symbol=self.symbol,
            liquidity_flow_alert_percentage=liquidity_flow_alert_percentage,
            wall_std_dev_factor=wall_std_dev_factor,
            time_manager=self.time_manager,
            cache_ttl_seconds=getattr(config, "ORDERBOOK_CACHE_TTL", 30.0),
            max_stale_seconds=getattr(config, "ORDERBOOK_MAX_STALE", 300.0),
            rate_limit_threshold=getattr(
                config, "ORDERBOOK_MAX_REQUESTS_PER_MIN", 5
            ),
        )

        # Estado do orderbook
        self.last_valid_orderbook: Optional[Dict[str, Any]] = None
        self.last_valid_orderbook_time: float = 0.0
        self.orderbook_fetch_failures = 0
        self.orderbook_emergency_mode = getattr(
            config, "ORDERBOOK_EMERGENCY_MODE", True
        )

        self._orderbook_refresh_lock = threading.Lock()
        self._orderbook_refresh_thread: Optional[threading.Thread] = None
        self._orderbook_background_refresh = getattr(
            config, "ORDERBOOK_BACKGROUND_REFRESH", True
        )
        self._orderbook_bg_min_interval = float(
            getattr(config, "ORDERBOOK_BG_MIN_INTERVAL", 5.0)
        )
        self._last_async_ob_refresh = 0.0

        # Executor para tarefas assíncronas auxiliares
        self._async_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="orderbook_",
        )

        # Loop asyncio dedicado para o OrderBookAnalyzer
        self._async_loop = asyncio.new_event_loop()
        self._async_loop_thread = threading.Thread(
            target=self._run_async_loop,
            name="orderbook_async_loop",
            daemon=True,
        )
        self._async_loop_thread.start()

        self.last_valid_vp: Optional[Dict[str, Any]] = None
        self.last_valid_vp_time: float = 0.0

        self.event_saver = EventSaver(sound_alert=True)
        self.context_collector = ContextCollector(symbol=self.symbol)
        self.flow_analyzer = FlowAnalyzer(time_manager=self.time_manager)

        # ===== IA =====
        self.ai_analyzer: Optional[AIAnalyzer] = None
        self.ai_initialization_attempted = False
        self.ai_test_passed = False
        self.ai_thread_pool: List[threading.Thread] = []
        self.max_ai_threads = 3
        self.ai_semaphore = threading.Semaphore(3)
        self._ai_pool_lock = threading.Lock()
        self.ai_rate_limiter = RateLimiter(max_calls=10, period_seconds=60)

        # Inicializa IA em background (mesma lógica do original)
        initialize_ai_async(self)

        # EventBus → IA
        self.event_bus.subscribe("signal", self._handle_signal_event)
        self.event_bus.subscribe("zone_touch", self._handle_zone_touch_event)

        # Conexão WebSocket robusta
        self.connection_manager = RobustConnectionManager(
            stream_url, symbol, max_reconnect_attempts=25
        )
        self.connection_manager.set_callbacks(
            on_message=self.on_message,
            on_open=self.on_open,
            on_close=self.on_close,
            on_reconnect=self._on_reconnect,
        )
        self.connection_manager.set_heartbeat_cb(
            lambda: self.health_monitor.heartbeat("main")
        )

        self.window_end_ms: Optional[int] = None
        self.window_data: List[Dict[str, Any]] = []
        self.window_count = 0

        self.history_size = history_size
        self.volume_history: deque[float] = deque(maxlen=history_size)
        self.delta_history: deque[float] = deque(maxlen=history_size)
        self.close_price_history: deque[float] = deque(
            maxlen=context_sma_period
        )
        self.delta_std_dev_factor = delta_std_dev_factor
        self.volatility_history: deque[float] = deque(maxlen=history_size)

        # Histórico de OHLC por janela para pattern recognition
        self.pattern_ohlc_history: deque[Dict[str, float]] = deque(
            maxlen=getattr(config, "PATTERN_LOOKBACK_BARS", 200)
        )

        self._missing_field_counts: Dict[str, int] = {
            "q": 0,
            "m": 0,
            "p": 0,
            "T": 0,
        }

        try:
            self._missing_field_log_step = getattr(
                config, "MISSING_FIELD_LOG_STEP", None
            )
        except Exception:
            self._missing_field_log_step = None

        # Contadores de mensagens inválidas
        self._invalid_json_count = 0
        self._invalid_trade_count = 0
        try:
            self._invalid_json_log_step = int(
                getattr(config, "INVALID_JSON_LOG_STEP", 100)
            )
        except Exception:
            self._invalid_json_log_step = 100

        try:
            self._invalid_trade_log_step = int(
                getattr(config, "INVALID_TRADE_LOG_STEP", 100)
            )
        except Exception:
            self._invalid_trade_log_step = 100

        self._last_price: Optional[float] = None
        self._last_alert_ts: Dict[str, float] = {}
        self._sent_triggers: set = set()
        self._last_trade_ts_ms: Optional[int] = None

        self._last_ai_analysis_ts = 0.0
        self._ai_min_interval_sec = getattr(
            config, "AI_MIN_INTERVAL_SEC", 60
        )

        try:
            self._alert_cooldown_sec = getattr(
                config, "ALERT_COOLDOWN_SEC", 30
            )
        except Exception:
            self._alert_cooldown_sec = 30

        self._register_cleanup_handlers()

    # ========================================
    # HANDLER DE RECONEXÃO
    # ========================================
    def _on_reconnect(self) -> None:
        logging.warning(
            "🔄 RECONEXÃO DETECTADA - Iniciando período de aquecimento..."
        )

        with self._warmup_lock:
            self.warming_up = True
            self.warmup_windows_remaining = self.warmup_windows_required
            self.window_data = []
            self.window_end_ms = None

        logging.info(
            f"⏳ Aguardando {self.warmup_windows_required} janelas "
            f"para estabilizar dados..."
        )

    # ========================================
    # GERENCIAMENTO DE THREADS DE IA
    # ========================================
    def _wait_for_ai_threads(self, timeout_per_thread: float = 2.0) -> None:
        """Aguarda as threads de IA terminarem, com timeout por thread."""
        with self._ai_pool_lock:
            threads = list(self.ai_thread_pool)

        for t in threads:
            try:
                t.join(timeout=timeout_per_thread)
            except Exception:
                pass

    # ========================================
    # CLEANUP
    # ========================================
    def _cleanup_handler(self, signum=None, frame=None) -> None:
        with self._cleanup_lock:
            if self._cleanup_started.is_set():
                logging.debug("Cleanup já em andamento, ignorando chamada duplicada")
                return
            self._cleanup_started.set()
            self.is_cleaning_up = True

        logging.info("🧹 Iniciando limpeza dos recursos...")
        self.should_stop = True

        # Aguarda término das threads de IA
        try:
            self._wait_for_ai_threads(timeout_per_thread=2.0)
        except Exception as e:
            logging.debug(f"Falha ao aguardar threads de IA no cleanup: {e}")

        # Demais componentes
        try:
            if self.context_collector:
                self.context_collector.stop()
                logging.info("✅ Context Collector parado.")
        except Exception as e:
            logging.error(f"❌ Erro ao parar Context Collector: {e}")

        try:
            if self.ai_analyzer and hasattr(self.ai_analyzer, "close"):
                self.ai_analyzer.close()
                logging.info("✅ AI Analyzer fechado.")
        except Exception as e:
            logging.error(f"❌ Erro ao fechar AI Analyzer: {e}")

        try:
            if self.connection_manager:
                self.connection_manager.disconnect()
                logging.info("✅ Connection Manager desconectado.")
        except Exception as e:
            logging.error(f"❌ Erro ao desconectar Connection Manager: {e}")

        try:
            if hasattr(self, "event_bus"):
                self.event_bus.shutdown()
                logging.info("✅ Event Bus encerrado.")
        except Exception as e:
            logging.error(f"❌ Erro ao encerrar Event Bus: {e}")

        try:
            if hasattr(self, "health_monitor"):
                self.health_monitor.stop()
                logging.info("✅ Health Monitor parado.")
        except Exception as e:
            logging.error(f"❌ Erro ao parar Health Monitor: {e}")

        try:
            if self.clock_sync and hasattr(self.clock_sync, "stop"):
                self.clock_sync.stop()
                logging.info("✅ Clock Sync parado.")
        except Exception as e:
            logging.error(f"❌ Erro ao parar Clock Sync: {e}")

        try:
            if hasattr(self, "_async_executor"):
                self._async_executor.shutdown(wait=True)
                logging.info("✅ Async Executor encerrado.")
        except Exception as e:
            logging.error(f"❌ Erro ao encerrar Async Executor: {e}")

        # Encerrar loop asyncio dedicado do OrderBookAnalyzer
        try:
            if hasattr(self, "_async_loop"):
                try:
                    if (
                        hasattr(self, "orderbook_analyzer")
                        and self.orderbook_analyzer
                        and hasattr(self.orderbook_analyzer, "close")
                    ):
                        fut = asyncio.run_coroutine_threadsafe(
                            self.orderbook_analyzer.close(),
                            self._async_loop,
                        )
                        try:
                            fut.result(timeout=2.0)
                        except FutureTimeoutError:
                            logging.debug(
                                "Timeout ao fechar OrderBookAnalyzer; cancelando tarefa"
                            )
                            fut.cancel()
                        except Exception:
                            pass
                except Exception as e:
                    logging.debug(f"Falha ao fechar OrderBookAnalyzer: {e}")

                try:
                    self._async_loop.call_soon_threadsafe(self._async_loop.stop)
                except Exception:
                    pass
                try:
                    if hasattr(self, "_async_loop_thread"):
                        self._async_loop_thread.join(timeout=2.0)
                except Exception:
                    pass

                logging.info("✅ Loop assíncrono do OrderBookAnalyzer encerrado.")
        except Exception as e:
            logging.error(f"❌ Erro ao encerrar loop assíncrono: {e}")

        logging.info("✅ Bot encerrado com segurança.")

    def _register_cleanup_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self._cleanup_handler)
            signal.signal(signal.SIGTERM, self._cleanup_handler)
        except Exception:
            pass
        try:
            atexit.register(self._cleanup_handler)
        except Exception:
            pass

    # ========================================
    # LOOP ASSÍNCRONO DEDICADO
    # ========================================
    def _run_async_loop(self) -> None:
        asyncio.set_event_loop(self._async_loop)
        try:
            self._async_loop.run_forever()
        finally:
            try:
                try:
                    pending = asyncio.all_tasks(loop=self._async_loop)
                except TypeError:
                    pending = asyncio.Task.all_tasks(loop=self._async_loop)
            except Exception:
                pending = []

            for task in pending:
                try:
                    task.cancel()
                except Exception:
                    pass

            if pending:
                try:
                    group = asyncio.gather(*pending, return_exceptions=True)
                    self._async_loop.run_until_complete(
                        asyncio.wait_for(group, timeout=2.0)
                    )
                except Exception:
                    pass

            try:
                shutdown_coro = self._async_loop.shutdown_asyncgens()
                self._async_loop.run_until_complete(
                    asyncio.wait_for(shutdown_coro, timeout=1.0)
                )
            except Exception:
                pass

            try:
                self._async_loop.close()
            except Exception:
                pass

    # ========================================
    # JANELA DE TEMPO
    # ========================================
    def _next_boundary_ms(self, ts_ms: int) -> int:
        return ((ts_ms // self.window_ms) + 1) * self.window_ms

    # ========================================
    # PROCESSAMENTO DE MENSAGENS
    # ========================================
    def on_message(self, ws: Any, message: str) -> None:
        if self.should_stop:
            return

        # 1) Decodificação de JSON
        try:
            raw = json.loads(message)
        except json.JSONDecodeError as e:
            self._invalid_json_count += 1
            step = self._invalid_json_log_step or 100
            if step > 0 and self._invalid_json_count % step == 0:
                logging.error(
                    "Erro ao decodificar mensagem JSON (amostra %d, total=%d): %s",
                    step,
                    self._invalid_json_count,
                    e,
                )
            return
        except Exception as e:
            logging.error(
                f"Erro inesperado ao decodificar mensagem JSON: {e}",
                exc_info=True,
            )
            return

        # 2) Extração e normalização de campos
        try:
            trade = raw.get("data", raw)

            p = trade.get("p") or trade.get("P") or trade.get("price")
            q = trade.get("q") or trade.get("Q") or trade.get("quantity")
            T = trade.get("T") or trade.get("E") or trade.get("tradeTime")
            m = trade.get("m")

            # Fallback para mensagens de kline
            if (p is None or q is None or T is None) and isinstance(
                trade.get("k"), dict
            ):
                k = trade["k"]
                if p is None:
                    p = k.get("c")
                if q is None:
                    q = k.get("v")
                if T is None:
                    T = k.get("T") or raw.get("E")

            # 3) Verificação de campos obrigatórios
            missing: List[str] = []
            if p is None:
                missing.append("p")
                self._missing_field_counts["p"] += 1
            if q is None:
                missing.append("q")
                self._missing_field_counts["q"] += 1
            if T is None:
                missing.append("T")
                self._missing_field_counts["T"] += 1

            if missing:
                total_missing = sum(
                    self._missing_field_counts[k] for k in ("p", "q", "T")
                )
                if self._missing_field_log_step:
                    try:
                        step = int(self._missing_field_log_step)
                    except Exception:
                        step = None

                    if step and step > 0 and total_missing % step == 0:
                        logging.debug(
                            "Campos ausentes (amostra): p=%d q=%d T=%d",
                            self._missing_field_counts["p"],
                            self._missing_field_counts["q"],
                            self._missing_field_counts["T"],
                        )
                return

            # 4) Conversão de tipos
            try:
                p = float(p)
                q = float(q)
                T = int(T)
            except (TypeError, ValueError):
                self._invalid_trade_count += 1
                step = self._invalid_trade_log_step or 100
                if step > 0 and self._invalid_trade_count % step == 0:
                    logging.error(
                        "Trade inválido (tipos) - amostra %d, total=%d: %s",
                        step,
                        self._invalid_trade_count,
                        trade,
                    )
                return

            # 5) Validação básica
            if p <= 0 or q <= 0 or T <= 0:
                self._invalid_trade_count += 1
                step = self._invalid_trade_log_step or 100
                if step > 0 and self._invalid_trade_count % step == 0:
                    logging.warning(
                        "Trade descartado por valores não positivos (amostra %d, total=%d): p=%s q=%s T=%s",
                        step,
                        self._invalid_trade_count,
                        p,
                        q,
                        T,
                    )
                return

            # 5.1) Normalização de T para garantir monotonicidade
            last_T = self._last_trade_ts_ms
            if last_T is not None and T < last_T:
                logging.debug(
                    "Timestamp de trade fora de ordem detectado: T_atual=%d < T_ultimo=%d (normalizando para T=%d)",
                    T,
                    last_T,
                    last_T,
                )
                T = last_T
            else:
                self._last_trade_ts_ms = T

            # 6) Inferência de agressor (m) se ausente
            if m is None:
                last_price = self._last_price
                m = (p <= last_price) if last_price is not None else False

            # 7) Atualiza estados compartilhados
            self._last_price = p

            norm = {"p": p, "q": q, "T": T, "m": bool(m)}

            self.trades_buffer.append(norm)

            try:
                self.health_monitor.heartbeat("main")
            except Exception:
                pass

            # Envia trade para FlowAnalyzer
            self.flow_analyzer.process_trade(norm)

            # 8) Controle de janelas
            if self.window_end_ms is None:
                self.window_end_ms = self._next_boundary_ms(T)

            if T >= self.window_end_ms:
                self._process_window()
                self.window_end_ms = self._next_boundary_ms(T)
                self.window_data = [norm]
            else:
                self.window_data.append(norm)

        except Exception as e:
            logging.error(f"Erro ao processar mensagem: {e}", exc_info=True)

    # ========================================
    # PONTOS DE DELEGAÇÃO PARA SUBMÓDULOS
    # ========================================
    def _process_window(self) -> None:
        process_window(self)

    def _process_signals(
        self,
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
    ):
        return process_signals(
            self,
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

    def _run_ai_analysis_threaded(self, event_data: Dict[str, Any]) -> None:
        return run_ai_analysis_threaded(self, event_data)

    # ========================================
    # LÓGICA DE IA (usa _run_ai_analysis_threaded)
    # ========================================
    def _is_important_event_for_ai(self, event_data: Dict[str, Any]) -> bool:
        tipo = (event_data.get("tipo_evento") or "").upper()
        resultado = (event_data.get("resultado_da_batalha") or "").upper()
        severity = (event_data.get("severity") or "").upper()

        if tipo in ("ABSORÇÃO", "ABSORCAO", "EXAUSTÃO", "EXAUSTAO"):
            return True

        if tipo == "ZONA" or "zone_context" in event_data:
            return True

        if tipo == "ORDERBOOK":
            crit = event_data.get("critical_flags", {}) or {}
            if crit.get("is_critical") or severity == "CRITICAL":
                return True

        if tipo == "ALERTA":
            alert_type = resultado
            important_alerts = {
                "SUPPLY_EXHAUSTION",
                "DEMAND_EXHAUSTION",
                "LIQUIDITY_BREAK",
                "VOLATILITY_EXPANSION",
                "VOLATILITY_SQUEEZE",
            }
            if alert_type in important_alerts:
                return True

        if tipo == "ANALYSIS_TRIGGER":
            now = time.time()
            if now - self._last_ai_analysis_ts >= self._ai_min_interval_sec:
                return True
            return False

        return False

    def _handle_signal_event(self, event_data: Dict[str, Any]) -> None:
        if not self.ai_analyzer or not self.ai_test_passed:
            return

        if not self._is_important_event_for_ai(event_data):
            return

        severity = (event_data.get("severity") or "").upper()
        tipo = (event_data.get("tipo_evento") or "").upper()
        is_critical = (
            severity in ("CRITICAL", "HIGH")
            or tipo in ("ABSORÇÃO", "ABSORCAO", "ZONA")
        )

        now = time.time()

        if (
            not is_critical
            and now - self._last_ai_analysis_ts < self._ai_min_interval_sec
        ):
            logging.debug("⏱️ IA em cooldown, pulando evento não-crítico")
            return

        self._last_ai_analysis_ts = now
        self._run_ai_analysis_threaded(event_data.copy())

    def _handle_zone_touch_event(self, event_data: Dict[str, Any]) -> None:
        if not self.ai_analyzer or not self.ai_test_passed:
            return

        now = time.time()
        time_since_last = now - self._last_ai_analysis_ts

        if time_since_last < self._ai_min_interval_sec:
            logging.info(
                f"🎯 BYPASS DE COOLDOWN: Toque em zona processado "
                f"após apenas {time_since_last:.1f}s"
            )

        self._last_ai_analysis_ts = now
        self._run_ai_analysis_threaded(event_data.copy())

    # ========================================
    # PROCESSAMENTO DE VP FEATURES
    # ========================================
    def _process_vp_features(
        self,
        historical_profile: Dict[str, Any],
        preco_atual: float,
    ) -> Dict[str, Any]:
        try:
            if not preco_atual or preco_atual <= 0:
                return {"status": "no_data"}

            vp_daily = historical_profile.get("daily", {})
            hvns = vp_daily.get("hvns", [])
            lvns = vp_daily.get("lvns", [])
            sp = vp_daily.get("single_prints", [])
            poc = vp_daily.get("poc", 0)

            if not poc or (not hvns and not lvns):
                return {"status": "no_data"}

            dist_to_poc = preco_atual - poc

            nearest_hvn = (
                min(hvns, key=lambda x: abs(x - preco_atual)) if hvns else None
            )
            nearest_lvn = (
                min(lvns, key=lambda x: abs(x - preco_atual)) if lvns else None
            )

            dist_hvn = (preco_atual - nearest_hvn) if nearest_hvn else None
            dist_lvn = (preco_atual - nearest_lvn) if nearest_lvn else None

            faixa_lim = preco_atual * 0.005

            hvn_near = sum(
                1 for h in hvns if abs(h - preco_atual) <= faixa_lim
            )
            lvn_near = sum(
                1 for l in lvns if abs(l - preco_atual) <= faixa_lim
            )

            in_single = any(
                abs(px - preco_atual) <= faixa_lim for px in sp
            )

            return {
                "status": "ok",
                "distance_to_poc": round(dist_to_poc, 2),
                "nearest_hvn": nearest_hvn,
                "dist_to_nearest_hvn": (
                    round(dist_hvn, 2) if dist_hvn is not None else None
                ),
                "nearest_lvn": nearest_lvn,
                "dist_to_nearest_lvn": (
                    round(dist_lvn, 2) if dist_lvn is not None else None
                ),
                "hvns_within_0_5pct": hvn_near,
                "lvns_within_0_5pct": lvn_near,
                "in_single_print_zone": in_single,
            }

        except Exception as e:
            logging.error(f"Erro ao gerar vp_features: {e}")
            return {"status": "error"}

    # ========================================
    # LOG DE EVENTOS
    # ========================================
    def _format_memory_timestamp_ny(self, e: Dict[str, Any]) -> str:
        """
         Converte o timestamp de um evento da memória para horário de New York (string).

          Preferência:
          1) epoch_ms (ou metadata.timestamp_unix_ms)
          2) timestamp_ny
          3) timestamp_utc
          4) timestamp bruto (removendo 'Z' se existir)
          """
        epoch_ms = e.get("epoch_ms") or (e.get("metadata") or {}).get("timestamp_unix_ms")
        if epoch_ms is not None:
            try:
                epoch_ms_int = int(epoch_ms)
                dt_utc = datetime.fromtimestamp(epoch_ms_int / 1000, tz=timezone.utc)
                dt_ny = dt_utc.astimezone(self.ny_tz)
                return dt_ny.strftime("%Y-%m-%d %H:%M:%S NY")
            except Exception:
                pass

        ts_candidates = [
            e.get("timestamp_ny"),
            e.get("timestamp_utc"),
            e.get("timestamp"),
        ]
        for ts in ts_candidates:
            if not ts:
                continue
            raw = str(ts)
            try:
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_ny = dt.astimezone(self.ny_tz)
                return dt_ny.strftime("%Y-%m-%d %H:%M:%S NY")
            except Exception:
                continue

        raw = str(e.get("timestamp", "N/A"))
        if raw.endswith("Z"):
            raw = raw[:-1]
        return raw

    def _log_event(self, event: Dict[str, Any]) -> None:
        ts_ny = event.get("timestamp_ny")

        if ts_ny:
            try:
                ny_time = datetime.fromisoformat(
                    ts_ny.replace("Z", "+00:00")
                ).astimezone(self.ny_tz)
            except Exception:
                ny_time = datetime.now(self.ny_tz)
        else:
            ny_time = datetime.now(self.ny_tz)

        resultado = event.get("resultado_da_batalha", "N/A").upper()
        tipo = event.get("tipo_evento", "EVENTO")
        descricao = event.get("descricao", "")
        conf = event.get("historical_confidence", {})

        print(
            f"\n🎯 {tipo}: {resultado} DETECTADO - "
            f"{ny_time.strftime('%H:%M:%S')} NY"
        )
        print(f" Símbolo: {self.symbol} | Janela #{self.window_count}")
        print(f" 📝 {descricao}")

        if conf:
            print(
                f" 📊 Probabilidades -> "
                f"Long={conf.get('long_prob')} | "
                f"Short={conf.get('short_prob')} | "
                f"Neutro={conf.get('neutral_prob')}"
            )

        ultimos = [
            e
            for e in obter_memoria_eventos(n=4)
            if e.get("tipo_evento") != "OrderBook"
        ]

        if ultimos:
            print(" 🕒 Últimos sinais:")
            for e in ultimos:
                delta_fmt = format_delta(e.get("delta", 0))
                vol_fmt = format_large_number(e.get("volume_total", 0))
                ts_display = self._format_memory_timestamp_ny(e)
                print(
                    f"  - {ts_display} | "
                    f"{e.get('tipo_evento', 'N/A')} "
                    f"{e.get('resultado_da_batalha', 'N/A')} "
                    f"(Δ={delta_fmt}, Vol={vol_fmt})"
                )

    # ========================================
    # ENRIQUECIMENTO DE SINAL
    # ========================================
    def _enrich_signal(
        self,
        signal: Dict[str, Any],
        derivatives_context: Dict[str, Any],
        flow_metrics: Dict[str, Any],
        total_buy_volume: float,
        total_sell_volume: float,
        macro_context: Dict[str, Any],
        close_ms: int,
        ml_payload: Dict[str, Any],
        enriched_snapshot: Dict[str, Any],
        contextual_snapshot: Dict[str, Any],
        ob_event: Dict[str, Any],
        valid_window_data: List[Dict[str, Any]],
        support_resistance: Dict[str, Any],
        defense_zones_data: Dict[str, Any],
    ) -> None:
        """Enriquece sinal com dados adicionais e gera evento institucional."""

        signal.setdefault("janela_numero", self.window_count)

        if "epoch_ms" not in signal:
            signal["epoch_ms"] = close_ms

        if "timestamp_utc" not in signal:
            signal["timestamp_utc"] = self.time_manager.from_timestamp_ms(
                close_ms, tz=self.time_manager.tz_utc
            ).isoformat(timespec="milliseconds")

        if "timestamp" not in signal:
            signal["timestamp"] = self.time_manager.from_timestamp_ms(
                close_ms, tz=self.ny_tz
            ).strftime("%Y-%m-%d %H:%M:%S")

        validated_signal = validator.validate_and_clean(signal)
        if not validated_signal:
            logging.warning(
                f"Evento {signal.get('tipo_evento')} / "
                f"{signal.get('resultado_da_batalha')} descartado pela validação."
            )
            return

        signal.update(validated_signal)

        if "derivatives" not in signal:
            signal["derivatives"] = derivatives_context

        should_validate_flow = (
            self.window_count % 10 == 0
            or self.orderbook_fetch_failures > 0
            or len(valid_window_data) < 5
        )

        if "fluxo_continuo" not in signal and flow_metrics:
            flow_valid = True
            if should_validate_flow:
                flow_valid = self._validate_flow_metrics(
                    flow_metrics, valid_window_data
                )
                if not flow_valid:
                    signal["flow_data_quality"] = "incomplete"
            signal["fluxo_continuo"] = flow_metrics

        if (
            signal.get("volume_compra", 0) == 0
            and signal.get("volume_venda", 0) == 0
        ):
            signal["volume_compra"] = total_buy_volume
            signal["volume_venda"] = total_sell_volume

        try:
            if "market_context" not in signal:
                signal["market_context"] = macro_context.get(
                    "market_context", {}
                )
            if "market_environment" not in signal:
                signal["market_environment"] = macro_context.get(
                    "market_environment", {}
                )
        except Exception:
            pass

        signal.setdefault("features_window_id", str(close_ms))
        signal["ml_features"] = ml_payload
        signal["enriched_snapshot"] = enriched_snapshot
        signal["contextual_snapshot"] = contextual_snapshot

        if support_resistance:
            signal["support_resistance"] = support_resistance
        if defense_zones_data:
            signal["defense_zones"] = defense_zones_data

        # ------- Orderbook + métricas avançadas -------
        if ob_event and isinstance(ob_event, dict) and ob_event.get(
            "is_valid", False
        ):
            if "orderbook_data" in ob_event:
                signal["orderbook_data"] = ob_event["orderbook_data"]
            if "spread_metrics" in ob_event:
                signal["spread_metrics"] = ob_event["spread_metrics"]
            if "order_book_depth" in ob_event:
                signal["order_book_depth"] = ob_event["order_book_depth"]
            if "spread_analysis" in ob_event:
                signal["spread_analysis"] = ob_event["spread_analysis"]

            dq = ob_event.get("data_quality") or {}
            if dq:
                signal["orderbook_data_quality"] = dq
                src = dq.get("data_source")
                if src == "emergency":
                    signal["orderbook_quality"] = "emergency"
                elif src == "cache":
                    signal["orderbook_quality"] = "cache"
                else:
                    signal["orderbook_quality"] = "live"

            try:
                mi_buy = ob_event.get("market_impact_buy", {}) or {}
                mi_sell = ob_event.get("market_impact_sell", {}) or {}

                def _get_move(mi_dict, key):
                    d = mi_dict.get(key, {}) or {}
                    return d.get("move_usd")

                slippage_matrix = {
                    "1k_usd":   {"buy": _get_move(mi_buy,  "1k"),  "sell": _get_move(mi_sell,  "1k")},
                    "10k_usd":  {"buy": _get_move(mi_buy, "10k"),  "sell": _get_move(mi_sell, "10k")},
                    "100k_usd": {"buy": _get_move(mi_buy,"100k"),  "sell": _get_move(mi_sell,"100k")},
                    "1m_usd":   {"buy": _get_move(mi_buy,  "1M"),  "sell": _get_move(mi_sell,  "1M")},
                }

                bps_100k_buy = (mi_buy.get("100k") or {}).get("bps")
                bps_100k_sell = (mi_sell.get("100k") or {}).get("bps")
                bps_list = [
                    v for v in (bps_100k_buy, bps_100k_sell)
                    if isinstance(v, (int, float))
                ]
                if bps_list:
                    avg_bps = float(sum(bps_list) / len(bps_list))
                    liquidity_score = max(0.0, min(10.0, 10.0 - avg_bps / 5.0))
                else:
                    liquidity_score = None

                if liquidity_score is not None:
                    if liquidity_score >= 8:
                        execution_quality = "EXCELLENT"
                    elif liquidity_score >= 6:
                        execution_quality = "GOOD"
                    elif liquidity_score >= 4:
                        execution_quality = "FAIR"
                    else:
                        execution_quality = "POOR"
                else:
                    execution_quality = None

                signal["market_impact"] = {
                    "slippage_matrix": slippage_matrix,
                    "liquidity_score": liquidity_score,
                    "execution_quality": execution_quality,
                }
            except Exception as e:
                logging.debug(f"Falha ao construir market_impact: {e}")

        if signal.get("tipo_evento") == "ANALYSIS_TRIGGER":
            key = (
                signal.get("tipo_evento"),
                signal.get("features_window_id"),
            )
            if key in self._sent_triggers:
                logging.debug(
                    f"⏭️ ANALYSIS_TRIGGER duplicado ignorado (janela {close_ms})"
                )
                return
            self._sent_triggers.add(key)

        self.levels.add_from_event(signal)

        logging.debug(
            f"💾 Salvando: {signal.get('tipo_evento')} / "
            f"{signal.get('resultado_da_batalha')} | "
            f"epoch_ms={signal.get('epoch_ms')} | "
            f"janela_numero={signal.get('janela_numero')}"
        )

        self.event_bus.publish("signal", signal)

        institutional_event = self._build_institutional_event(signal)
        self.event_saver.save_event(institutional_event)

        if signal.get("tipo_evento") != "OrderBook":
            adicionar_memoria_evento(signal)

        self._log_event(signal)

    # ========================================
    # BUILDER DE EVENTO INSTITUCIONAL
    # ========================================
    def _build_institutional_event(
        self, signal: Dict[str, Any]
    ) -> Dict[str, Any]:
        # (Aqui manter o mesmo conteúdo que você tinha no original,
        # é um bloco grande. Se quiser, posso te enviar só este
        # bloco na próxima mensagem para colar, para não estourar
        # o tamanho desta resposta.)
        return {
            "tipo_evento": signal.get("tipo_evento"),
            "resultado_da_batalha": signal.get("resultado_da_batalha"),
            "descricao": signal.get("descricao"),
            "symbol": signal.get("ativo") or signal.get("symbol") or self.symbol,
            "raw_event": signal,
        }

    # ========================================
    # (Demais métodos auxiliares do original)
    # ========================================
    def _build_price_targets(
        self,
        pattern_recognition: Dict[str, Any],
        last_price: float,
    ) -> Dict[str, Any]:
        targets: List[Dict[str, Any]] = []
        try:
            patterns = pattern_recognition.get("active_patterns") or []
            for p in patterns:
                ptype = (p.get("type") or "").upper()
                target = p.get("target_price")
                stop = p.get("stop_loss")
                conf = float(p.get("confidence", 0.0) or 0.0)

                side = "UNKNOWN"
                if "ASCENDING" in ptype or "BULL" in ptype:
                    side = "BULLISH"
                elif "DESCENDING" in ptype or "BEAR" in ptype:
                    side = "BEARISH"

                if target is not None:
                    risk = None
                    rr = None
                    if stop is not None and last_price:
                        try:
                            risk = abs(last_price - float(stop))
                            reward = abs(float(target) - last_price)
                            rr = reward / risk if risk > 0 else None
                        except Exception:
                            rr = None

                    targets.append(
                        {
                            "pattern_type": ptype,
                            "side": side,
                            "target_price": float(target),
                            "stop_loss": float(stop) if stop is not None else None,
                            "confidence": conf,
                            "risk_reward": rr,
                        }
                    )
        except Exception as e:
            logging.debug(f"Erro ao construir price_targets: {e}")

        if not targets:
            return {}

        return {
            "targets": targets,
            "last_price": last_price,
        }

    def _validate_flow_metrics(
        self,
        flow_metrics: Dict[str, Any],
        valid_window_data: List[Dict[str, Any]],
    ) -> bool:
        try:
            trades_processed = 0
            if "data_quality" in flow_metrics:
                trades_processed = flow_metrics["data_quality"].get(
                    "flow_trades_count", 0
                )

            if trades_processed > 0:
                return True

            sector_flow = flow_metrics.get("sector_flow", {})
            for _, data in sector_flow.items():
                total_vol = abs(data.get("buy", 0)) + abs(data.get("sell", 0))
                if total_vol > 0.001:
                    return True

            order_flow = flow_metrics.get("order_flow", {})
            for key in ("net_flow_1m", "net_flow_5m", "net_flow_15m"):
                val = order_flow.get(key)
                if val is not None and val != 0:
                    return True

            buy_pct = order_flow.get("aggressive_buy_pct", 0.0)
            sell_pct = order_flow.get("aggressive_sell_pct", 0.0)
            if buy_pct > 0 or sell_pct > 0:
                return True

            whale_total = abs(flow_metrics.get("whale_buy_volume", 0.0)) + abs(
                flow_metrics.get("whale_sell_volume", 0.0)
            )
            if whale_total > 0.001:
                return True

            return False

        except Exception as e:
            logging.error(f"Erro ao validar flow_metrics: {e}")
            return False

    def _check_zone_touches(
        self, enriched: Dict[str, Any], signals: List[Dict[str, Any]]
    ) -> None:
        preco_atual = enriched.get("ohlc", {}).get("close", 0.0)

        if preco_atual > 0:
            try:
                touched = self.levels.check_price(float(preco_atual))

                for z in touched:
                    zone_event = signals[0].copy() if signals else {}

                    preco_fmt = format_price(preco_atual)
                    low_fmt = format_price(z.low)
                    high_fmt = format_price(z.high)

                    zone_event.update(
                        {
                            "tipo_evento": "Zona",
                            "resultado_da_batalha": f"Toque em Zona {z.kind}",
                            "descricao": (
                                f"Preço {preco_fmt} tocou {z.kind} "
                                f"{z.timeframe} [{low_fmt} ~ {high_fmt}]"
                            ),
                            "zone_context": z.to_dict(),
                            "preco_fechamento": preco_atual,
                            "timestamp": self.time_manager.now_utc_iso(
                                timespec="seconds"
                            ),
                        }
                    )

                    zone_event["janela_numero"] = self.window_count

                    if "historical_confidence" not in zone_event:
                        zone_event["historical_confidence"] = (
                            calcular_probabilidade_historica(zone_event)
                        )

                    self.event_bus.publish("zone_touch", zone_event)

                    institutional_zone_event = self._build_institutional_event(
                        zone_event
                    )
                    self.event_saver.save_event(institutional_zone_event)

                    adicionar_memoria_evento(
                        {
                            "timestamp": (
                                z.last_touched
                                or datetime.now(
                                    self.ny_tz
                                ).isoformat(timespec="seconds")
                            ),
                            "tipo_evento": "Zona",
                            "resultado_da_batalha": f"Toque {z.kind}",
                            "delta": zone_event.get("delta", 0.0),
                            "volume_total": zone_event.get(
                                "volume_total", 0.0
                            ),
                        }
                    )

            except Exception as e:
                logging.error(f"Erro ao verificar toques em zonas: {e}")

    def _update_histories(
        self, enriched: Dict[str, Any], ml_payload: Dict[str, Any]
    ) -> None:
        window_volume = enriched.get("volume_total", 0.0)
        window_delta = enriched.get("delta_fechamento", 0.0)
        window_close = enriched.get("ohlc", {}).get("close", 0.0)

        self.volume_history.append(window_volume)
        self.delta_history.append(window_delta)

        if window_close > 0:
            self.close_price_history.append(window_close)

        try:
            price_feats = ml_payload.get("price_features") or {}
            current_volatility = None

            if "volatility_5" in price_feats:
                current_volatility = price_feats["volatility_5"]
            elif "volatility_1" in price_feats:
                current_volatility = price_feats["volatility_1"]

            if current_volatility is not None:
                current_volatility = float(current_volatility)
                self.volatility_history.append(current_volatility)

                try:
                    last_price = float(
                        enriched.get("ohlc", {}).get("close", 0.0) or window_close
                    )
                except Exception:
                    last_price = float(window_close or 0.0)

                if last_price > 0:
                    price_volatility_abs = current_volatility * last_price

                    try:
                        self.flow_analyzer.update_volatility_context(
                            atr_price=None,
                            price_volatility=price_volatility_abs,
                        )
                    except Exception as e:
                        logging.debug(
                            f"Falha ao atualizar contexto de volatilidade no FlowAnalyzer: {e}"
                        )

        except Exception as e:
            logging.debug(f"Falha ao atualizar histórico de volatilidade: {e}")

        try:
            ohlc = enriched.get("ohlc") or {}
            if ohlc:
                self.pattern_ohlc_history.append(
                    {
                        "high": float(ohlc.get("high", 0.0)),
                        "low": float(ohlc.get("low", 0.0)),
                        "close": float(ohlc.get("close", 0.0)),
                    }
                )
        except Exception:
            pass

    def _log_liquidity_heatmap(
        self, flow_metrics: Dict[str, Any]
    ) -> None:
        try:
            liquidity_data = flow_metrics.get("liquidity_heatmap", {})
            clusters = liquidity_data.get("clusters", [])

            if clusters:
                logging.info(
                    "📊 LIQUIDITY HEATMAP - Janela #%s:",
                    self.window_count,
                )

                for i, cluster in enumerate(clusters[:3]):
                    center_fmt = format_price(cluster.get("center", 0.0))
                    vol_fmt = format_large_number(
                        cluster.get("total_volume", 0.0)
                    )
                    imb_fmt = format_percent(
                        cluster.get("imbalance_ratio", 0.0) * 100.0
                    )
                    trades_fmt = format_quantity(
                        cluster.get("trades_count", 0)
                    )
                    age_fmt = format_time_seconds(cluster.get("age_ms", 0))

                    logging.info(
                        "  Cluster %d: $%s | Vol: %s | Imb: %s | Trades: %s | Age: %s",
                        i + 1,
                        center_fmt,
                        vol_fmt,
                        imb_fmt,
                        trades_fmt,
                        age_fmt,
                    )
        except Exception as e:
            logging.error(f"Erro ao logar liquidity heatmap: {e}")

    def _log_ml_features(self, ml_payload: Dict[str, Any]) -> None:
        try:
            pf = ml_payload.get("price_features", {}) if ml_payload else {}
            vf = ml_payload.get("volume_features", {}) if ml_payload else {}
            mf = ml_payload.get("microstructure", {}) if ml_payload else {}

            if pf or vf or mf:
                ret5_fmt = format_scientific(pf.get("returns_5", 0.0))
                vol5_fmt = format_scientific(
                    pf.get("volatility_5", 0.0), decimals=5
                )
                vsma_fmt = format_percent(
                    vf.get("volume_sma_ratio", 0.0) * 100.0
                )
                bs_fmt = format_delta(vf.get("buy_sell_pressure", 0.0))
                obs_fmt = format_scientific(
                    mf.get("order_book_slope", 0.0), decimals=3
                )
                flow_fmt = format_scientific(
                    mf.get("flow_imbalance", 0.0), decimals=3
                )

                logging.info(
                    "  ML: ret5=%s vol5=%s V/SMA=%s BSpress=%s OBslope=%s FlowImb=%s",
                    ret5_fmt,
                    vol5_fmt,
                    vsma_fmt,
                    bs_fmt,
                    obs_fmt,
                    flow_fmt,
                )
        except Exception:
            pass

    def _log_health_check(self) -> None:
        if self.window_count % 10 == 0:
            last_ob_age = (
                time.time() - self.last_valid_orderbook_time
                if self.last_valid_orderbook_time > 0
                else float("inf")
            )
            last_vp_age = (
                time.time() - self.last_valid_vp_time
                if self.last_valid_vp_time > 0
                else float("inf")
            )
            logging.info(
                f"\n📊 HEALTH CHECK - Janela #{self.window_count}:\n"
                f"  Orderbook: failures={self.orderbook_fetch_failures}, "
                f"last_valid={last_ob_age:.0f}s ago\n"
                f"  Value Area: last_valid={last_vp_age:.0f}s ago"
            )

    def _log_window_summary(
        self,
        enriched: Dict[str, Any],
        historical_profile: Dict[str, Any],
        macro_context: Dict[str, Any],
    ) -> None:
        window_delta = enriched.get("delta_fechamento", 0.0)
        window_volume = enriched.get("volume_total", 0.0)

        delta_fmt = format_delta(window_delta)
        vol_fmt = format_large_number(window_volume)

        logging.info(
            "[%s NY] 🟡 Janela #%s | Delta: %s | Vol: %s",
            datetime.now(self.ny_tz).strftime("%H:%M:%S"),
            self.window_count,
            delta_fmt,
            vol_fmt,
        )

        if macro_context:
            trends = macro_context.get("mtf_trends", {})
            parts: List[str] = []
            for tf, data in trends.items():
                try:
                    parts.append(f"{tf.upper()}: {data['tendencia']}")
                except Exception:
                    parts.append(f"{tf.upper()}: {data}")
            trends_str = ", ".join(parts)
            if trends_str:
                logging.info("  Macro Context: %s", trends_str)

        if historical_profile and historical_profile.get("daily"):
            vp = historical_profile["daily"]
            poc_fmt = format_price(vp.get("poc", 0.0))
            val_fmt = format_price(vp.get("val", 0.0))
            vah_fmt = format_price(vp.get("vah", 0.0))

            logging.info(
                "  VP Diário: POC @ %s | VAL: %s | VAH: %s",
                poc_fmt,
                val_fmt,
                vah_fmt,
            )

        logging.info("─" * 80)

    def _process_institutional_alerts(
        self, enriched: Dict[str, Any], pipeline: DataPipeline
    ) -> None:
        if generate_alerts is None:
            return

        try:
            if detect_support_resistance is not None:
                try:
                    price_series = (
                        pipeline.df["p"]
                        if hasattr(pipeline, "df") and pipeline.df is not None
                        else None
                    )
                    if price_series is not None:
                        sr = detect_support_resistance(
                            price_series, num_levels=3
                        )
                    else:
                        sr = {
                            "immediate_support": [],
                            "immediate_resistance": [],
                        }
                except Exception:
                    sr = {
                        "immediate_support": [],
                        "immediate_resistance": [],
                    }
            else:
                sr = {
                    "immediate_support": [],
                    "immediate_resistance": [],
                }

            dz = None
            if defense_zones is not None:
                try:
                    dz = defense_zones(sr)
                except Exception:
                    dz = None

            window_close = enriched.get("ohlc", {}).get("close", 0.0)
            current_price_alert = window_close

            avg_vol = (
                sum(self.volume_history) / len(self.volume_history)
                if len(self.volume_history) > 0
                else enriched.get("volume_total", 0.0)
            )

            rec_vols = list(self.volatility_history)

            curr_vol = None
            try:
                if len(self.volatility_history) > 0:
                    curr_vol = self.volatility_history[-1]
            except Exception:
                curr_vol = None

            alerts_list = generate_alerts(
                price=current_price_alert,
                support_resistance=sr,
                current_volume=enriched.get("volume_total", 0.0),
                average_volume=avg_vol,
                current_volatility=curr_vol or 0.0,
                recent_volatilities=rec_vols,
                volume_threshold=3.0,
                tolerance_pct=0.001,
            )

            for alert in alerts_list or []:
                try:
                    atype = alert.get("type", "GENERIC")
                    now_s = time.time()
                    last_ts = self._last_alert_ts.get(atype, 0.0)

                    if now_s - last_ts < self._alert_cooldown_sec:
                        continue

                    self._last_alert_ts[atype] = now_s

                    desc_parts: List[str] = [f"Tipo: {alert.get('type')}"]

                    if "level" in alert:
                        desc_parts.append(
                            f"Nível: {format_price(alert['level'])}"
                        )

                    if "threshold_exceeded" in alert:
                        desc_parts.append(
                            f"Fator: {format_percent(alert['threshold_exceeded'] * 100.0)}"
                        )

                    descricao_alert = " | ".join(desc_parts)

                    print(f"🔔 ALERTA: {descricao_alert}")
                    logging.info(f"🔔 ALERTA: {descricao_alert}")

                    alert_event = {
                        "tipo_evento": "Alerta",
                        "resultado_da_batalha": alert.get("type"),
                        "descricao": descricao_alert,
                        "timestamp": self.time_manager.now_utc_iso(
                            timespec="seconds"
                        ),
                        "severity": alert.get("severity"),
                        "probability": alert.get("probability"),
                        "action": alert.get("action"),
                        "context": {
                            "price": current_price_alert,
                            "volume": enriched.get("volume_total", 0.0),
                            "average_volume": avg_vol,
                            "volatility": curr_vol or 0.0,
                        },
                        "data_context": "real_time",
                    }

                    alert_event["support_resistance"] = sr
                    if dz is not None:
                        alert_event["defense_zones"] = dz

                    alert_event["janela_numero"] = self.window_count
                    alert_event["epoch_ms"] = int(time.time() * 1000)

                    institutional_alert = self._build_institutional_event(
                        alert_event
                    )
                    self.event_saver.save_event(institutional_alert)

                except Exception as e:
                    logging.error(f"Erro ao processar alerta: {e}")

        except Exception as e:
            logging.error(f"Erro ao gerar alertas: {e}")

    # ========================================
    # CALLBACKS DO WEBSOCKET
    # ========================================
    def on_open(self, ws: Any) -> None:
        logging.info(
            f"🚀 Bot iniciado para {self.symbol} - "
            f"Fuso: New York (America/New_York)"
        )
        try:
            self.health_monitor.heartbeat("main")
        except Exception:
            pass

    def on_close(self, ws: Any, code: int, msg: str) -> None:
        if self.window_data and not self.should_stop:
            self._process_window()

    # ========================================
    # RUN
    # ========================================
    def run(self) -> None:
        try:
            self.context_collector.start()

            logging.info(
                "🎯 Iniciando Enhanced Market Bot v2.3.2 "
                "(refatorado em módulos)..."
            )
            print("═" * 80)

            self.connection_manager.connect()

        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        except Exception as e:
            logging.critical(
                f"❌ Erro crítico ao executar o bot: {e}",
                exc_info=True,
            )
        finally:
            self._cleanup_handler()