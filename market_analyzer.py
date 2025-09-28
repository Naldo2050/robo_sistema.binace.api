# market_analyzer.py
import websocket
import json
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
import ssl
import socket
import random
from zoneinfo import ZoneInfo
from collections import deque
import pandas as pd

# Importações dos módulos do projeto
import config
from data_handler import create_absorption_event, create_exhaustion_event
from orderbook_analyzer import OrderBookAnalyzer
from event_saver import EventSaver

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RobustConnectionManager:
    def __init__(self, stream_url, symbol, max_reconnect_attempts=10, 
                 initial_delay=1, max_delay=60, backoff_factor=1.5):
        self.stream_url = stream_url
        self.symbol = symbol
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
        # Estado da conexão
        self.current_delay = initial_delay
        self.reconnect_count = 0
        self.is_connected = False
        self.last_message_time = None
        self.connection_start_time = None
        
        # Threads de monitoramento
        self.heartbeat_thread = None
        self.connection_monitor_thread = None
        self.should_stop = False
        
        # Callbacks personalizados
        self.on_message_callback = None
        self.on_open_callback = None
        self.on_close_callback = None
        self.on_error_callback = None
        
        # Estatísticas de conexão
        self.total_messages_received = 0
        self.total_reconnects = 0
        self.connection_uptime = timedelta(0)
        
    def set_callbacks(self, on_message=None, on_open=None, on_close=None, on_error=None):
        """Define callbacks personalizados."""
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error
    
    def _test_connection(self):
        """Testa se a URL é acessível antes de conectar."""
        try:
            parsed_url = urlparse(self.stream_url)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'wss' else 80)
            
            with socket.create_connection((host, port), timeout=5) as sock:
                if parsed_url.scheme == 'wss':
                    context = ssl.create_default_context()
                    with context.wrap_socket(sock, server_hostname=host) as ssock:
                        return True
                return True
        except Exception as e:
            logging.error(f"Erro ao testar conexão: {e}")
            return False
    
    def _on_message(self, ws, message):
        """Handler interno para mensagens."""
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
        """Handler interno para abertura de conexão."""
        self.is_connected = True
        self.reconnect_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now(timezone.utc)
        self.last_message_time = self.connection_start_time
        
        logging.info(f"✅ Conexão estabelecida com {self.symbol}")
        self._start_monitoring_threads()
        
        if self.on_open_callback:
            self.on_open_callback(ws)
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handler interno para fechamento de conexão."""
        self.is_connected = False
        
        if self.connection_start_time:
            session_duration = datetime.now(timezone.utc) - self.connection_start_time
            self.connection_uptime += session_duration
        
        logging.warning(f"🔌 Conexão fechada - Código: {close_status_code}, Msg: {close_msg}")
        self._stop_monitoring_threads()
        
        if self.on_close_callback:
            self.on_close_callback(ws, close_status_code, close_msg)
    
    def _on_error(self, ws, error):
        """Handler interno para erros."""
        logging.error(f"❌ Erro WebSocket: {error}")
        if self.on_error_callback:
            self.on_error_callback(ws, error)
    
    def _start_monitoring_threads(self):
        self.should_stop = False
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()
    
    def _stop_monitoring_threads(self):
        self.should_stop = True
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)
    
    def _heartbeat_monitor(self):
        """Monitora se mensagens estão sendo recebidas regularmente."""
        while not self.should_stop and self.is_connected:
            time.sleep(30)
            if self.last_message_time:
                time_since_last = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                if time_since_last > 120:
                    logging.warning(f"⚠️ Sem mensagens há {time_since_last:.0f}s. Forçando reconexão.")
                    self.is_connected = False
                    break
    
    def _calculate_next_delay(self):
        """Calcula próximo delay com backoff exponencial + jitter."""
        delay = min(self.current_delay * self.backoff_factor, self.max_delay)
        jitter = delay * 0.2 * (random.random() - 0.5)
        self.current_delay = max(self.initial_delay, delay + jitter)
        return self.current_delay
    
    def connect(self):
        """Estabelece conexão com sistema de reconexão robusto."""
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
        self._stop_monitoring_threads()
    
    def disconnect(self):
        logging.info("🛑 Iniciando desconexão...")
        self.should_stop = True
    
    def get_connection_stats(self):
        """Retorna estatísticas da conexão."""
        # ... (Método pode ser simplificado ou mantido conforme necessidade) ...
        return {"is_connected": self.is_connected, "total_reconnects": self.total_reconnects}

# ===============================
# CLASSES DE ANÁLISE
# ===============================
class MarketAnalyzer:
    """Invoca as análises de fluxo de trades do data_handler."""
    def __init__(self, k_absor, vol_factor_exh, history_size):
        self.k_absor = k_absor
        self.vol_factor_exh = vol_factor_exh
        self.history_df = pd.DataFrame()
        self.history_size = history_size
        logging.info("MarketAnalyzer inicializado.")

    def analyze_window(self, window_data, symbol):
        if not window_data or len(window_data) < 2:
            return None, None
        
        logging.info(f"Analisando janela de trades com {len(window_data)} trades...")
        
        # CORREÇÃO APLICADA: k=self.k_absor mudado para delta_threshold=self.k_absor
        absorption_event = create_absorption_event(
            window_data,
            symbol,
            delta_threshold=self.k_absor
        )
        
        # --- CORREÇÃO APLICADA AQUI ---
        # Passa uma lista de volumes para a função, em vez do DataFrame completo.
        history_volumes = self.history_df['VolumeTotal'].tolist() if not self.history_df.empty else []
        exhaustion_event = create_exhaustion_event(window_data, symbol, history_volumes, volume_factor=self.vol_factor_exh)
        
        # Atualizar histórico para a próxima análise de exaustão
        current_volume = exhaustion_event.get("volume_total", 0)
        if current_volume > 0:
            new_row = pd.DataFrame([{"VolumeTotal": current_volume}])
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)
            if len(self.history_df) > self.history_size:
                self.history_df = self.history_df.iloc[-self.history_size:]
        
        return absorption_event, exhaustion_event

# ===============================
# BOT PRINCIPAL
# ===============================
class EnhancedMarketBot:
    def __init__(self, stream_url, symbol, window_size_minutes=5, k_absor=100, vol_factor_exh=2.0, history_size=20):
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.ny_tz = ZoneInfo("America/New_York")

        # Inicializar analisadores e saver
        self.market_analyzer = MarketAnalyzer(k_absor, vol_factor_exh, history_size)
        self.orderbook_analyzer = OrderBookAnalyzer(symbol=self.symbol)
        self.event_saver = EventSaver(sound_alert=True)

        # Gerenciador de conexão
        self.connection_manager = RobustConnectionManager(stream_url, symbol, max_reconnect_attempts=15)
        self.connection_manager.set_callbacks(
            on_message=self.on_message,
            on_open=self.on_open
        )

        # Estado da janela
        self.window_data = []
        self.window_end_time = None
        self.window_count = 0

    def _update_window_end_time(self):
        """Calcula o próximo timestamp de fechamento de candle em NY."""
        now_ny = datetime.now(self.ny_tz)
        minutes_into_hour = now_ny.minute
        next_window_minute = ((minutes_into_hour // self.window_size_minutes) + 1) * self.window_size_minutes
        
        end_time_ny = now_ny.replace(second=0, microsecond=0)
        if next_window_minute >= 60:
            end_time_ny += timedelta(hours=1)
            end_time_ny = end_time_ny.replace(minute=(next_window_minute % 60))
        else:
            end_time_ny = end_time_ny.replace(minute=next_window_minute)
            
        self.window_end_time = end_time_ny
        logging.info(f"🕐 Nova janela definida. Próximo fechamento às {self.window_end_time.strftime('%H:%M:%S')} NY")

    def on_message(self, ws, message):
        try:
            trade = json.loads(message)
            if "T" not in trade: return
            
            trade_time = datetime.fromtimestamp(trade["T"] / 1000, tz=self.ny_tz)

            if self.window_end_time is None:
                self._update_window_end_time()

            if trade_time >= self.window_end_time:
                self._process_window()
                self._update_window_end_time()
                self.window_data = [trade]
            else:
                self.window_data.append(trade)
                
        except Exception as e:
            logging.error(f"Erro ao processar mensagem: {e}")

    def _process_window(self):
        """Processa a janela de trades e o livro de ofertas, e salva os eventos."""
        if not self.window_data: return
        self.window_count += 1
        
        try:
            # 1. Análise do Fluxo de Trades
            absorption_event, exhaustion_event = self.market_analyzer.analyze_window(self.window_data, self.symbol)
            
            for event in [absorption_event, exhaustion_event]:
                if event and event["resultado_da_batalha"] not in ["Sem Absorção", "Sem Exaustão", "Erro"]:
                    self.event_saver.save_event(event)
                    self._log_event(event)
            
            # 2. Análise do Livro de Ofertas
            orderbook_event = self.orderbook_analyzer.analyze_order_book()
            if orderbook_event and orderbook_event["resultado_da_batalha"] not in ["Neutro", "Erro"]:
                self.event_saver.save_event(orderbook_event)
                self._log_event(orderbook_event)

            # Log de conclusão da janela
            ny_time = datetime.now(self.ny_tz)
            print(f"[{ny_time.strftime('%H:%M:%S')} NY] 🟡 Janela #{self.window_count} processada. {len(self.window_data)} trades.")
            print("─" * 60)

        except Exception as e:
            logging.error(f"Erro no processamento da janela #{self.window_count}: {e}")
        finally:
            self.window_data = []

    def _log_event(self, event):
        """Loga um evento detectado de forma padronizada."""
        ny_time = datetime.now(self.ny_tz)
        resultado = event.get('resultado_da_batalha', 'N/A').upper()
        tipo = event.get('tipo_evento', 'EVENTO')
        descricao = event.get('descricao', '')
        
        print(f"\n🎯 {tipo}: {resultado} DETECTADO - {ny_time.strftime('%H:%M:%S')} NY")
        print(f"   Símbolo: {self.symbol} | Janela #{self.window_count}")
        print(f"   📝 {descricao}")

    def on_open(self, ws):
        ny_time = datetime.now(self.ny_tz)
        logging.info(f"🚀 Bot iniciado - {ny_time.strftime('%H:%M:%S')} NY")
        self.window_end_time = None

    def run(self):
        try:
            logging.info(f"🤖 Iniciando bot para {self.symbol}...")
            self.connection_manager.connect()
        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        finally:
            self.connection_manager.disconnect()
            logging.info("Bot encerrado.")

# ===============================
# EXECUÇÃO
# ===============================
if __name__ == "__main__":
    bot = EnhancedMarketBot(
        stream_url=config.STREAM_URL,
        symbol=config.SYMBOL,
        window_size_minutes=5,
        k_absor=100,
        vol_factor_exh=2.0,
        history_size=20
    )
    bot.run()
