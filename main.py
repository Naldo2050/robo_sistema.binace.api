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

# Importa o arquivo de configurações central
import config

from data_handler import create_absorption_event, create_exhaustion_event, format_timestamp, NY_TZ
from orderbook_analyzer import OrderBookAnalyzer
from event_saver import EventSaver
from context_collector import ContextCollector
from flow_analyzer import FlowAnalyzer
from ai_analyzer_hybrid import AIAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===============================
# GESTOR DE CONEXÃO WEBSOCKET
# ===============================
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
                ws = websocket.WebSocketApp(self.stream_url,
                                            on_message=self._on_message,
                                            on_error=self._on_error,
                                            on_close=self._on_close,
                                            on_open=self._on_open)
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

# ===============================
# ANALISADOR DE TRADE FLOW
# ===============================
class TradeFlowAnalyzer:
    def __init__(self, vol_factor_exh, tz_output: ZoneInfo):
        self.vol_factor_exh = vol_factor_exh
        self.tz_output = tz_output

    def analyze_window(self, window_data, symbol, history_volumes, dynamic_delta_threshold, historical_profile=None):
        if not window_data or len(window_data) < 2:
            return ({"is_signal": False, "delta": 0, "volume_total": 0, "preco_fechamento": 0},
                    {"is_signal": False, "delta": 0, "volume_total": 0, "preco_fechamento": 0})
        
        absorption_event = create_absorption_event(window_data, symbol,
                                                 delta_threshold=dynamic_delta_threshold,
                                                 tz_output=self.tz_output,
                                                 historical_profile=historical_profile)
        
        exhaustion_event = create_exhaustion_event(window_data, symbol,
                                                   history_volumes=history_volumes,
                                                   volume_factor=self.vol_factor_exh,
                                                   tz_output=self.tz_output,
                                                   historical_profile=historical_profile)
        return absorption_event, exhaustion_event

# ===============================
# BOT PRINCIPAL
# ===============================
class EnhancedMarketBot:
    def __init__(self, stream_url, symbol, window_size_minutes, vol_factor_exh, history_size,
                 delta_std_dev_factor, context_sma_period, liquidity_flow_alert_percentage,
                 wall_std_dev_factor):
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.window_ms = window_size_minutes * 60 * 1000
        self.ny_tz = NY_TZ
        self.should_stop = False
        self.is_cleaning_up = False
        
        self.trade_flow_analyzer = TradeFlowAnalyzer(vol_factor_exh, tz_output=self.ny_tz)
        self.orderbook_analyzer = OrderBookAnalyzer(symbol=self.symbol,
                                                    liquidity_flow_alert_percentage=liquidity_flow_alert_percentage,
                                                    wall_std_dev_factor=wall_std_dev_factor)
        self.event_saver = EventSaver(sound_alert=True)
        self.context_collector = ContextCollector(symbol=self.symbol)
        self.flow_analyzer = FlowAnalyzer()
        
        self.ai_analyzer = None
        self.ai_initialization_attempted = False
        self.ai_test_passed = False
        self.ai_thread_pool = []
        self.max_ai_threads = 3
        self._initialize_ai_async()
        
        self.connection_manager = RobustConnectionManager(stream_url, symbol, max_reconnect_attempts=15)
        self.connection_manager.set_callbacks(on_message=self.on_message, on_open=self.on_open, on_close=self.on_close)
        
        self.window_end_ms = None
        self.window_data = []
        self.window_count = 0
        self.history_size = history_size
        self.volume_history = deque(maxlen=history_size)
        self.delta_history = deque(maxlen=history_size)
        self.close_price_history = deque(maxlen=context_sma_period)
        self.delta_std_dev_factor = delta_std_dev_factor
        
        self._register_cleanup_handlers()

    def _initialize_ai_async(self):
        def ai_init_worker():
            try:
                if self.ai_initialization_attempted:
                    return
                self.ai_initialization_attempted = True
                
                print("\n" + "="*30 + " INICIALIZANDO IA " + "="*30)
                logging.info("🧠 Tentando inicializar AI Analyzer Híbrida...")
                
                self.ai_analyzer = AIAnalyzer(headless=True)
                
                logging.info("✅ Módulo da IA carregado. Realizando teste de análise...")
                test_event = {
                    "tipo_evento": "Teste de Conexão",
                    "ativo": self.symbol,
                    "descricao": "Teste inicial do sistema de análise para garantir operacionalidade.",
                    "delta": 150.5,
                    "volume_total": 50000,
                    "preco_fechamento": 69000.0,
                }
                analysis = self.ai_analyzer.analyze_event(test_event)
                
                if analysis and len(analysis.strip()) > 50:
                    self.ai_test_passed = True
                    logging.info("✅ Teste de análise da IA bem-sucedido!")
                    print("\n" + "═"*25 + " RESULTADO DO TESTE DA IA " + "═"*25)
                    print(analysis)
                    print("═"*75 + "\n")
                else:
                    self.ai_test_passed = False
                    logging.warning(f"⚠️ Teste da IA retornou resultado inesperado ou vazio.")
                    print(f"Resultado recebido: {analysis}")
                    print("="*75 + "\n")

            except Exception as e:
                self.ai_analyzer = None
                self.ai_test_passed = False
                print("="*30 + " ERRO NA IA " + "="*30)
                logging.error(f"❌ Falha crítica ao inicializar a IA: {e}", exc_info=True)
                print("="*75 + "\n")
        
        threading.Thread(target=ai_init_worker, daemon=True).start()

    def _cleanup_handler(self, signum=None, frame=None):
        if self.is_cleaning_up:
            return
        self.is_cleaning_up = True
        logging.info("🧹 Iniciando limpeza dos recursos...")
        self.should_stop = True
        if self.context_collector:
            self.context_collector.stop()
        if self.ai_analyzer:
            try:
                self.ai_analyzer.close()
                logging.info("✅ AI Analyzer fechado.")
            except Exception as e:
                logging.error(f"❌ Erro ao fechar AI Analyzer: {e}")
        if self.connection_manager:
            self.connection_manager.disconnect()
        logging.info("✅ Bot encerrado com segurança.")

    def _register_cleanup_handlers(self):
        signal.signal(signal.SIGINT, self._cleanup_handler)
        signal.signal(signal.SIGTERM, self._cleanup_handler)
        atexit.register(self._cleanup_handler)

    def _next_boundary_ms(self, ts_ms: int) -> int:
        return ((ts_ms // self.window_ms) + 1) * self.window_ms

    def on_message(self, ws, message):
        if self.should_stop:
            return
        try:
            trade = json.loads(message)
            self.flow_analyzer.process_trade(trade)
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
        if not self.ai_analyzer or not self.ai_test_passed or self.should_stop:
            if not self.ai_test_passed and self.ai_analyzer:
                logging.warning("⚠️ Análise da IA ignorada: sistema não passou no teste inicial.")
            return

        def ai_worker():
            try:
                logging.info(f"🧠 IA iniciando análise para evento: {event_data.get('resultado_da_batalha', 'N/A')}")
                analysis_result = self.ai_analyzer.analyze_event(event_data)
                if analysis_result and not self.should_stop:
                    print("\n" + "═"*25 + " ANÁLISE PROFISSIONAL DA IA " + "═"*25)
                    print(analysis_result)
                    print("═"*75 + "\n")
            except Exception as e:
                logging.error(f"❌ Erro na thread de análise da IA: {e}")
        
        threading.Thread(target=ai_worker, daemon=True).start()

    def _process_vp_features(self, historical_profile, preco_atual: float):
        """Cria resumo de volume profile em features de alta utilidade para IA."""
        try:
            vp_daily = historical_profile.get("daily", {})
            hvns = vp_daily.get("hvns", [])
            lvns = vp_daily.get("lvns", [])
            sp = vp_daily.get("single_prints", [])
            poc = vp_daily.get("poc", 0)

            # Distâncias
            dist_to_poc = preco_atual - poc if poc else 0
            nearest_hvn = min(hvns, key=lambda x: abs(x - preco_atual)) if hvns else None
            nearest_lvn = min(lvns, key=lambda x: abs(x - preco_atual)) if lvns else None
            dist_hvn = (preco_atual - nearest_hvn) if nearest_hvn else None
            dist_lvn = (preco_atual - nearest_lvn) if nearest_lvn else None

            # Contagem em faixa ±0.5%
            faixa_lim = preco_atual * 0.005
            hvn_near = sum(1 for h in hvns if abs(h - preco_atual) <= faixa_lim)
            lvn_near = sum(1 for l in lvns if abs(l - preco_atual) <= faixa_lim)

            # Single print zone
            in_single = any(abs(p - preco_atual) <= faixa_lim for p in sp)

            return {
                "distance_to_poc": dist_to_poc,
                "nearest_hvn": nearest_hvn,
                "dist_to_nearest_hvn": dist_hvn,
                "nearest_lvn": nearest_lvn,
                "dist_to_nearest_lvn": dist_lvn,
                "hvns_within_0_5pct": hvn_near,
                "lvns_within_0_5pct": lvn_near,
                "in_single_print_zone": in_single
            }
        except Exception as e:
            logging.error(f"Erro ao gerar vp_features: {e}")
            return {}

    def _process_window(self):
        if not self.window_data or self.should_stop or sum(float(trade.get('q', 0)) for trade in self.window_data) == 0:
            self.window_data = []
            return
        
        self.window_count += 1
        try:
            # --- 1. PREPARAÇÃO DOS DADOS ---
            dynamic_delta_threshold = 0
            if len(self.delta_history) > 10:
                mean_delta = np.mean(self.delta_history)
                std_delta = np.std(self.delta_history)
                dynamic_delta_threshold = abs(mean_delta + self.delta_std_dev_factor * std_delta)
            
            macro_context = self.context_collector.get_context()
            historical_profile = macro_context.get("historical_vp", {})
            open_ms, close_ms = self.window_end_ms - self.window_ms, self.window_end_ms

            # --- 2. ANÁLISE DA JANELA ATUAL ---
            absorption_event, exhaustion_event = self.trade_flow_analyzer.analyze_window(
                self.window_data, self.symbol, list(self.volume_history), 
                dynamic_delta_threshold, historical_profile=historical_profile
            )
            orderbook_event = self.orderbook_analyzer.analyze_order_book()
            
            # --- 3. CONSTRUÇÃO DO EVENTO FINAL ---
            base_candle_metrics = absorption_event.copy()
            flow_metrics = self.flow_analyzer.get_flow_metrics()
            if flow_metrics:
                base_candle_metrics["fluxo_continuo"] = flow_metrics

            all_events = [absorption_event, exhaustion_event, orderbook_event]
            master_event = next((event for event in all_events if event and event.get("is_signal")), absorption_event)
            
            final_event_data = master_event.copy()
            final_event_data.update(base_candle_metrics)
            
            if orderbook_event:
                final_event_data.update({
                    "imbalance": orderbook_event.get("imbalance"),
                    "volume_ratio": orderbook_event.get("volume_ratio"),
                    "alertas_liquidez": orderbook_event.get("alertas_liquidez")
                })
            
            preco_atual = base_candle_metrics.get("preco_fechamento", 0)
            vp_features = self._process_vp_features(historical_profile, preco_atual)
            
            final_event_data.update({
                "candle_id_ms": close_ms,
                "candle_open_time_ms": open_ms,
                "candle_close_time_ms": close_ms,
                "candle_open_time": format_timestamp(open_ms, tz=self.ny_tz),
                "candle_close_time": format_timestamp(close_ms, tz=self.ny_tz),
                "contexto_macro": {k: v for k, v in macro_context.items() if k != 'historical_vp'},
                "vp_features": vp_features
            })

            # --- 4. SALVAR E EXECUTAR IA ---
            self.event_saver.save_event(final_event_data)
            self._run_ai_analysis_threaded(final_event_data.copy())
            
            self.volume_history.append(base_candle_metrics.get("volume_total", 0))
            self.delta_history.append(base_candle_metrics.get("delta", 0))
            self.close_price_history.append(base_candle_metrics.get("preco_fechamento", 0))
            
            # --- 5. LOG ---
            print(f"[{datetime.now(self.ny_tz).strftime('%H:%M:%S')} NY] 🟡 Janela #{self.window_count} | Delta: {base_candle_metrics.get('delta', 0):,.2f} | Vol: {base_candle_metrics.get('volume_total', 0):,.2f}")
            if macro_context:
                trends_str = ", ".join([f"{tf.upper()}: {data['tendencia']}" for tf, data in macro_context.get('mtf_trends', {}).items()])
                print(f"   Macro Context: {trends_str}")
                
                if historical_profile and historical_profile.get('daily'):
                    vp = historical_profile['daily']
                    print(f"   VP Diário: POC @ {vp.get('poc', 0):,.2f} | VAL: {vp.get('val', 0):,.2f} | VAH: {vp.get('vah', 0):,.2f}")
                
                derivatives = macro_context.get('derivatives', {})
                if derivatives and self.symbol in derivatives:
                    d = derivatives[self.symbol]
                    funding = d.get('funding_rate_percent', 0)
                    oi = d.get('open_interest', 0)
                    ls_ratio = d.get('long_short_ratio', 0)
                    long_liq = d.get('longs_usd', 0)
                    short_liq = d.get('shorts_usd', 0)
                    liq_interval = int(config.CONTEXT_UPDATE_INTERVAL_SECONDS / 60)
                    print(f"   Derivativos: F:{funding:.4f}%|OI:{oi:,.0f}|L/S:{ls_ratio} | Liq ({liq_interval}m): Longs ${long_liq:,.0f} / Shorts ${short_liq:,.0f}")
            print("─" * 80)
        except Exception as e:
            logging.error(f"Erro no processamento da janela #{self.window_count}: {e}", exc_info=True)
        finally:
            self.window_data = []

    def on_open(self, ws):
        logging.info(f"🚀 Bot iniciado para {self.symbol} - Fuso: New York (America/New_York)")

    def on_close(self, ws, code, msg):
        if self.window_data and not self.should_stop:
            self._process_window()

    def run(self):
        try:
            self.context_collector.start()
            logging.info("🎯 Iniciando Enhanced Market Bot...")
            print("=" * 80)
            self.connection_manager.connect()
        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        finally:
            self._cleanup_handler()

if __name__ == "__main__":
    try:
        bot = EnhancedMarketBot(
            stream_url=config.STREAM_URL, symbol=config.SYMBOL,
            window_size_minutes=config.WINDOW_SIZE_MINUTES, vol_factor_exh=config.VOL_FACTOR_EXH,
            history_size=config.HISTORY_SIZE, delta_std_dev_factor=config.DELTA_STD_DEV_FACTOR,
            context_sma_period=config.CONTEXT_SMA_PERIOD, liquidity_flow_alert_percentage=config.LIQUIDITY_FLOW_ALERT_PERCENTAGE,
            wall_std_dev_factor=config.WALL_STD_DEV_FACTOR
        )
        bot.run()
    except Exception as e:
        logging.critical(f"❌ Erro crítico na inicialização do bot: {e}", exc_info=True)
        sys.exit(1)