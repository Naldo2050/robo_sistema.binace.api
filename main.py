# main.py
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
import socket
import random
import websocket
from collections import deque
import signal
import sys
import atexit
import config
from data_handler import create_absorption_event, create_exhaustion_event, format_timestamp, NY_TZ
from orderbook_analyzer import OrderBookAnalyzer
from event_saver import EventSaver
# ESCOLHA QUAL IA USAR:
# Opção 1: Análise Híbrida Avançada (MELHOR ANÁLISE - RECOMENDADO)
from ai_analyzer_hybrid import AIAnalyzer
# Opção 2: Firefox + Gemini (recomendado para IA externa, mas com problemas de login)
# from ai_analyzer_firefox import AIAnalyzer
# Opção 3: Análise básica sem IA (100% estável, mas análise simples)
# from ai_analyzer_disabled import AIAnalyzer
# Opção 4: Chrome original (problemático com Google)
# from ai_analyzer import AIAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RobustConnectionManager:
    def __init__(self, stream_url, symbol, max_reconnect_attempts=10, initial_delay=1, max_delay=60, backoff_factor=1.5):
        self.stream_url = stream_url
        self.symbol = symbol
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.current_delay = initial_delay
        self.reconnect_count = 0
        self.is_connected = False
        self.last_message_time = None
        self.connection_start_time = None
        self.heartbeat_thread = None
        self.should_stop = False
        self.on_message_callback = None
        self.on_open_callback = None
        self.on_close_callback = None
        self.on_error_callback = None
        self.total_messages_received = 0
        self.total_reconnects = 0

    def set_callbacks(self, on_message=None, on_open=None, on_close=None, on_error=None):
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error

    def _test_connection(self):
        try:
            parsed = urlparse(self.stream_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "wss" else 80)
            with socket.create_connection((host, port), timeout=5):
                return True
        except Exception as e:
            logging.error(f"Erro ao testar conexão: {e}")
            return False

    def _on_message(self, ws, message):
        try:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1
            if self.current_delay > self.initial_delay:
                self.current_delay = max(self.initial_delay, self.current_delay * 0.9)
            if self.on_message_callback:
                self.on_message_callback(ws, message)
        except Exception as e:
            logging.error(f"Erro no processamento da mensagem: {e}")

    def _on_open(self, ws):
        self.is_connected = True
        self.reconnect_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now(timezone.utc)
        self.last_message_time = self.connection_start_time
        logging.info(f"✅ Conexão estabelecida com {self.symbol}")
        self._start_heartbeat()
        if self.on_open_callback:
            self.on_open_callback(ws)

    def _on_close(self, ws, close_status_code, close_msg):
        self.is_connected = False
        logging.warning(f"🔌 Conexão fechada - Código: {close_status_code}, Msg: {close_msg}")
        self._stop_heartbeat()
        if self.on_close_callback:
            self.on_close_callback(ws, close_status_code, close_msg)

    def _on_error(self, ws, error):
        logging.error(f"❌ Erro WebSocket: {error}")
        if self.on_error_callback:
            self.on_error_callback(ws, error)

    def _start_heartbeat(self):
        self.should_stop = False
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()

    def _stop_heartbeat(self):
        self.should_stop = True
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)

    def _heartbeat_monitor(self):
        while not self.should_stop and self.is_connected:
            time.sleep(30)
            if self.last_message_time:
                gap = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                if gap > 120:
                    logging.warning(f"⚠️ Sem mensagens há {gap:.0f}s. Forçando reconexão.")
                    self.is_connected = False
                    break

    def _calculate_next_delay(self):
        delay = min(self.current_delay * self.backoff_factor, self.max_delay)
        jitter = delay * 0.2 * (random.random() - 0.5)
        self.current_delay = max(self.initial_delay, delay + jitter)
        return self.current_delay

    def connect(self):
        while self.reconnect_count < self.max_reconnect_attempts and not self.should_stop:
            try:
                if not self._test_connection():
                    raise ConnectionError("Falha no teste de conectividade")
                logging.info(f"🔄 Tentativa {self.reconnect_count + 1}/{self.max_reconnect_attempts}")
                ws = websocket.WebSocketApp(
                    self.stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)
                if self.should_stop:
                    break
            except KeyboardInterrupt:
                logging.info("⏹️ Interrompido pelo usuário")
                self.should_stop = True
                break
            except Exception as e:
                self.reconnect_count += 1
                self.total_reconnects += 1
                logging.error(f"❌ Erro na conexão ({self.reconnect_count}/{self.max_reconnect_attempts}): {e}")
                if self.reconnect_count < self.max_reconnect_attempts:
                    delay = self._calculate_next_delay()
                    logging.info(f"⏳ Reconectando em {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logging.error("💀 Máximo de tentativas atingido. Encerrando.")
                    break
        self._stop_heartbeat()

    def disconnect(self):
        logging.info("🛑 Iniciando desconexão...")
        self.should_stop = True


class TradeFlowAnalyzer:
    def __init__(self, vol_factor_exh, tz_output: ZoneInfo):
        self.vol_factor_exh = vol_factor_exh
        self.tz_output = tz_output

    def analyze_window(self, window_data, symbol, history_volumes, dynamic_delta_threshold):
        if not window_data or len(window_data) < 2:
            empty_event = {"is_signal": False, "delta": 0, "volume_total": 0, "preco_fechamento": 0}
            return empty_event, empty_event
        
        absorption_event = create_absorption_event(
            window_data, symbol, delta_threshold=dynamic_delta_threshold, tz_output=self.tz_output
        )

        exhaustion_event = create_exhaustion_event(
            window_data,
            symbol,
            history_volumes=history_volumes,
            volume_factor=self.vol_factor_exh,
            tz_output=self.tz_output
        )
        return absorption_event, exhaustion_event


class EnhancedMarketBot:
    def __init__(self, stream_url, symbol, window_size_minutes, vol_factor_exh, history_size,
                 delta_std_dev_factor, context_sma_period, liquidity_flow_alert_percentage,
                 wall_std_dev_factor):
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.window_ms = window_size_minutes * 60 * 1000
        self.interval_str = f"{window_size_minutes}m"
        self.ny_tz = NY_TZ
        self.should_stop = False
        self.is_cleaning_up = False # <<<<<<< NOVA VARIÁVEL DE CONTROLE
        
        # Analisadores
        self.trade_flow_analyzer = TradeFlowAnalyzer(vol_factor_exh, tz_output=self.ny_tz)
        self.orderbook_analyzer = OrderBookAnalyzer(
            symbol=self.symbol,
            liquidity_flow_alert_percentage=liquidity_flow_alert_percentage,
            wall_std_dev_factor=wall_std_dev_factor
        )
        self.event_saver = EventSaver(sound_alert=True)

        # IA Analyzer com gerenciamento melhorado e robusto
        self.ai_analyzer = None
        self.ai_initialization_attempted = False
        self.ai_test_passed = False

        # Controle de threads da IA
        self.ai_thread_pool = []
        self.max_ai_threads = 3

        # Inicializa IA em thread separada para não bloquear o bot principal
        self._initialize_ai_async()

        # Conexão
        self.connection_manager = RobustConnectionManager(stream_url, symbol, max_reconnect_attempts=15)
        self.connection_manager.set_callbacks(on_message=self.on_message, on_open=self.on_open, on_close=self.on_close)

        # Estado da janela e histórico
        self.window_end_ms = None
        self.window_data = []
        self.window_count = 0
        self.history_size = history_size
        self.volume_history = deque(maxlen=history_size)
        self.delta_history = deque(maxlen=history_size)
        self.close_price_history = deque(maxlen=context_sma_period)

        # Parâmetros dinâmicos
        self.delta_std_dev_factor = delta_std_dev_factor

        # Registra handlers para limpeza adequada
        self._register_cleanup_handlers()

    def _initialize_ai_async(self):
        """Inicializa a IA de forma assíncrona para não bloquear o bot principal."""
        def ai_init_worker():
            try:
                if self.ai_initialization_attempted:
                    return

                self.ai_initialization_attempted = True
                logging.info("🧠 Inicializando AI Analyzer Híbrida...")

                self.ai_analyzer = AIAnalyzer(headless=True, user_data_dir="./hybrid_data")
                logging.info("✅ AI Analyzer Híbrida inicializada com sucesso")

                if hasattr(self.ai_analyzer, 'enabled') and not self.ai_analyzer.enabled:
                    self.ai_test_passed = True
                    logging.info("✅ Sistema de análise básica ativado (sem IA externa)")
                else:
                    logging.info("🧪 Testando sistema de análise híbrida...")
                    test_event = { "tipo_evento": "Teste de Conexão", "ativo": self.symbol, "descricao": "Teste inicial de conectividade com a IA", "delta": 100, "volume_total": 5000, "preco_fechamento": 45250.50, "volume_compra": 2800, "volume_venda": 2200, "indice_absorcao": 1.8 }
                    result = self.ai_analyzer.analyze_event(test_event)
                    if result and len(result.strip()) > 50:
                        logging.info("✅ Teste de análise híbrida bem-sucedido!")
                        self.ai_test_passed = True
                        print("\n" + "═"*30 + " TESTE DA ANÁLISE HÍBRIDA " + "═"*30)
                        print("🧪 Resultado do teste:")
                        print("─" * 85)
                        print(result[:400] + "..." if len(result) > 400 else result)
                        print("═" * 85 + "\n")
                    else:
                        logging.warning(f"⚠️ Teste de análise híbrida com problema: {result}")
                        self.ai_test_passed = False
            except Exception as e:
                logging.error(f"❌ Erro ao inicializar AI Analyzer Híbrida: {e}")
                self.ai_analyzer = None
                self.ai_test_passed = False
        ai_init_thread = threading.Thread(target=ai_init_worker, daemon=True)
        ai_init_thread.start()

    def _register_cleanup_handlers(self):
        """Registra handlers para limpeza adequada dos recursos."""
        def cleanup_handler(signum=None, frame=None):
            # <<<<<<< MUDANÇA PRINCIPAL AQUI >>>>>>>>>
            if self.is_cleaning_up:
                return
            self.is_cleaning_up = True
            
            logging.info("🧹 Iniciando limpeza dos recursos...")
            self.should_stop = True
            self._cleanup_ai_threads()
            if self.ai_analyzer:
                try:
                    self.ai_analyzer.close()
                    logging.info("✅ AI Analyzer fechado com sucesso")
                except Exception as e:
                    logging.error(f"❌ Erro ao fechar AI Analyzer: {e}")
            if self.connection_manager:
                self.connection_manager.disconnect()
            
            # sys.exit(0) foi removido daqui para evitar a chamada duplicada

        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        atexit.register(cleanup_handler)

    def _cleanup_ai_threads(self):
        """Limpa threads da IA pendentes."""
        if self.ai_thread_pool:
            logging.info("🧵 Aguardando threads da IA terminarem...")
            for thread in self.ai_thread_pool:
                if thread.is_alive():
                    thread.join(timeout=5)
            self.ai_thread_pool.clear()

    def _next_boundary_ms(self, ts_ms: int) -> int:
        return ((ts_ms // self.window_ms) + 1) * self.window_ms

    def on_message(self, ws, message):
        if self.should_stop:
            return
        try:
            trade = json.loads(message)
            if "T" not in trade:
                return
            ts = int(trade["T"])
            if self.window_end_ms is None:
                self.window_end_ms = self._next_boundary_ms(ts)
            if ts >= self.window_end_ms:
                self._process_window()
                self.window_end_ms = self._next_boundary_ms(ts)
                self.window_data = [trade]
            else:
                self.window_data.append(trade)
        except Exception as e:
            logging.error(f"Erro ao processar mensagem: {e}")

    def _run_ai_analysis_threaded(self, event_data):
        if not self.ai_analyzer or self.should_stop or not self.ai_test_passed:
            if not self.ai_test_passed and self.ai_analyzer:
                logging.warning("⚠️ IA não passou no teste inicial, análise ignorada")
            return
        def ai_worker():
            try:
                self.ai_thread_pool = [t for t in self.ai_thread_pool if t.is_alive()]
                logging.info(f"🧠 Iniciando análise híbrida para evento: {event_data.get('tipo_evento', 'N/A')}")
                analysis_result = self.ai_analyzer.analyze_event(event_data)
                if analysis_result and not self.should_stop and len(analysis_result.strip()) > 10:
                    print("\n" + "═"*20 + " ANÁLISE PROFISSIONAL HÍBRIDA " + "═"*20)
                    print(f"📊 Evento: {event_data.get('tipo_evento', 'N/A')}")
                    print(f"💹 Ativo: {event_data.get('ativo', 'N/A')}")
                    print(f"🕐 Horário: {event_data.get('candle_close_time', 'N/A')}")
                    print(f"📈 Delta: {event_data.get('delta', 0):.2f}")
                    print(f"📊 Volume: {event_data.get('volume_total', 0):,.0f}")
                    print("─" * 75)
                    print(analysis_result)
                    print("═" * 75 + "\n")
                else:
                    logging.warning("⚠️ Análise retornou resultado vazio ou bot foi interrompido")
            except Exception as e:
                logging.error(f"❌ Erro na análise híbrida: {e}")
            finally:
                current_thread = threading.current_thread()
                if current_thread in self.ai_thread_pool:
                    self.ai_thread_pool.remove(current_thread)
        active_threads = [t for t in self.ai_thread_pool if t.is_alive()]
        if len(active_threads) < self.max_ai_threads:
            ai_thread = threading.Thread(target=ai_worker, daemon=True)
            self.ai_thread_pool.append(ai_thread)
            ai_thread.start()
        else:
            logging.warning("⚠️ Máximo de threads de análise atingido, análise ignorada")

    def _process_window(self):
        if not self.window_data or self.should_stop or sum(float(trade.get('q', 0)) for trade in self.window_data) == 0:
            self.window_data = []
            return
        self.window_count += 1
        try:
            dynamic_delta_threshold = 0
            if len(self.delta_history) > 10:
                mean_delta = np.mean(self.delta_history)
                std_delta = np.std(self.delta_history)
                dynamic_delta_threshold = abs(mean_delta + self.delta_std_dev_factor * std_delta)
            sma_context = np.mean(self.close_price_history) if len(self.close_price_history) >= config.CONTEXT_SMA_PERIOD else None
            absorption_event, exhaustion_event = self.trade_flow_analyzer.analyze_window(
                self.window_data, self.symbol, list(self.volume_history), dynamic_delta_threshold
            )
            orderbook_event = self.orderbook_analyzer.analyze_order_book()
            base_candle_metrics = absorption_event.copy()
            open_ms = self.window_end_ms - self.window_ms
            close_ms = self.window_end_ms
            triggered_events = []
            for event in [absorption_event, exhaustion_event, orderbook_event]:
                if event and event.get("is_signal"):
                    triggered_events.append(event)
            if triggered_events:
                for event in triggered_events:
                    final_event_data = event.copy()
                    final_event_data.update(base_candle_metrics)
                    final_event_data.update({
                        "candle_id_ms": close_ms, "candle_open_time_ms": open_ms, "candle_close_time_ms": close_ms,
                        "candle_open_time": format_timestamp(open_ms, tz=self.ny_tz), "candle_close_time": format_timestamp(close_ms, tz=self.ny_tz)
                    })
                    if sma_context and final_event_data.get("preco_fechamento"):
                        pos = "ACIMA" if final_event_data["preco_fechamento"] > sma_context else "ABAIXO"
                        final_event_data["contexto_sma"] = f"Fechou {pos} da SMA({len(self.close_price_history)}) de {sma_context:.2f}"
                    self.event_saver.save_event(final_event_data)
                    if self.ai_analyzer and self.ai_test_passed and not self.should_stop:
                        self._run_ai_analysis_threaded(final_event_data.copy())
                    elif self.ai_analyzer and not self.ai_test_passed:
                        logging.info(f"ℹ️ Análise híbrida indisponível (teste falhou) para evento: {final_event_data.get('tipo_evento')}")
                    else:
                        logging.info(f"ℹ️ Análise híbrida desabilitada para evento: {final_event_data.get('tipo_evento')}")
            else:
                no_event_report = {
                    "is_signal": False, "tipo_evento": "Nenhum Evento Detectado", "ativo": self.symbol,
                    "candle_id_ms": close_ms, "candle_open_time_ms": open_ms, "candle_close_time_ms": close_ms,
                    "candle_open_time": format_timestamp(open_ms, tz=self.ny_tz), "candle_close_time": format_timestamp(close_ms, tz=self.ny_tz),
                }
                no_event_report.update(base_candle_metrics)
                self.event_saver.save_event(no_event_report)
            self.volume_history.append(base_candle_metrics.get("volume_total", 0))
            self.delta_history.append(base_candle_metrics.get("delta", 0))
            self.close_price_history.append(base_candle_metrics.get("preco_fechamento", 0))
            ai_status = "✅" if (self.ai_analyzer and self.ai_test_passed) else "❌" if self.ai_analyzer else "⚠️"
            ai_type = "🧠" if hasattr(self.ai_analyzer, 'use_advanced_analysis') and self.ai_analyzer.use_advanced_analysis else "🤖"
            print(f"[{datetime.now(self.ny_tz).strftime('%H:%M:%S')} NY] 🟡 Janela #{self.window_count} processada. Delta: {base_candle_metrics.get('delta', 0.0):.2f} | Análise: {ai_type}{ai_status}")
            delta_min = base_candle_metrics.get("delta_minimo", 0)
            delta_max = base_candle_metrics.get("delta_maximo", 0)
            reversao = base_candle_metrics.get("reversao_desde_minimo") if base_candle_metrics.get('delta', 0) > 0 else base_candle_metrics.get("reversao_desde_maximo")
            print(f"   Evolução Delta: Min({delta_min:,.0f}) Max({delta_max:,.0f}) Reversão({reversao:,.0f})")
            poc_price = base_candle_metrics.get("poc_price", 0)
            poc_percentage = base_candle_metrics.get("poc_percentage", 0)
            print(f"   Volume Profile (POC): {poc_percentage:.1f}% do volume @ ${poc_price:,.2f}")
            dwell_price = base_candle_metrics.get("dwell_price", 0)
            dwell_seconds = base_candle_metrics.get("dwell_seconds", 0)
            dwell_location = base_candle_metrics.get("dwell_location", "N/A")
            print(f"   Dwell Time: {dwell_seconds:.1f}s em {dwell_location} @ ${dwell_price:,.2f}")
            trades_per_sec = base_candle_metrics.get("trades_per_second", 0)
            avg_trade_size = base_candle_metrics.get("avg_trade_size", 0)
            print(f"   Trade Speed: {trades_per_sec:.2f} trades/s (vol. médio {avg_trade_size:.4f})")
            if dynamic_delta_threshold > 0:
                print(f"   Limiar Dinâmico de Absorção: {dynamic_delta_threshold:.2f}")
            if sma_context:
                print(f"   Contexto (SMA {len(self.close_price_history)}): {sma_context:.2f}")
            print("─" * 80)
        except Exception as e:
            logging.error(f"Erro no processamento da janela #{self.window_count}: {e}")
        finally:
            self.window_data = []

    def on_open(self, ws):
        logging.info(f"🚀 Bot iniciado para {self.symbol} - Fuso: New York (America/New_York)")
        if self.ai_analyzer:
            if hasattr(self.ai_analyzer, 'use_advanced_analysis') and self.ai_analyzer.use_advanced_analysis:
                logging.info("🧠 Análise: HÍBRIDA AVANÇADA ✅")
            elif hasattr(self.ai_analyzer, 'enabled') and not self.ai_analyzer.enabled:
                logging.info("🤖 Análise: BÁSICA (sem IA externa) ✅")
            elif self.ai_test_passed:
                ai_type = "Firefox" if "firefox" in str(type(self.ai_analyzer)).lower() else "Híbrida"
                logging.info(f"🧠 Análise ({ai_type}): ATIVADA ✅")
            else:
                logging.warning("🤖 Análise: TESTANDO... ⏳")
        else:
            logging.warning("⚠️ Análise: DESATIVADA (erro na inicialização)")

    def on_close(self, ws, code, msg):
        if self.window_data and not self.should_stop:
            self._process_window()

    def run(self):
        """Executa o bot principal."""
        try:
            ai_mode = "Híbrida Avançada" if hasattr(AIAnalyzer(headless=True), 'use_advanced_analysis') else "Básica"
            logging.info("🎯 Iniciando Enhanced Market Bot...")
            logging.info(f"📈 Símbolo: {self.symbol}")
            logging.info(f"⏰ Janela: {self.window_size_minutes} minutos")
            logging.info(f"🧠 Análise Mode: {ai_mode}")
            print("=" * 80)
            self.connection_manager.connect()
        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        except Exception as e:
            logging.error(f"❌ Erro durante execução: {e}")
        finally:
            logging.info("🛑 Finalizando bot...")
            self.should_stop = True
            # A chamada de limpeza será gerenciada pelo atexit
            # para garantir que ocorra mesmo em caso de erro.
        
if __name__ == "__main__":
    try:
        bot = EnhancedMarketBot(
            stream_url=config.STREAM_URL,
            symbol=config.SYMBOL,
            window_size_minutes=config.WINDOW_SIZE_MINUTES,
            vol_factor_exh=config.VOL_FACTOR_EXH,
            history_size=config.HISTORY_SIZE,
            delta_std_dev_factor=config.DELTA_STD_DEV_FACTOR,
            context_sma_period=config.CONTEXT_SMA_PERIOD,
            liquidity_flow_alert_percentage=config.LIQUIDITY_FLOW_ALERT_PERCENTAGE,
            wall_std_dev_factor=config.WALL_STD_DEV_FACTOR
        )
        # O atexit.register(cleanup_handler) já garante a limpeza.
        # A chamada explícita no finally é removida para evitar redundância.
        bot.run()
    except Exception as e:
        logging.error(f"❌ Erro crítico: {e}")
        sys.exit(1)