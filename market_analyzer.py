# market_analyzer.py v2.0.0 - INTEGRAÇÃO COMPLETA

"""
Market Analyzer com integração COMPLETA de todos os componentes.

🔹 CORREÇÕES v2.0.0:
  ✅ Integra FlowAnalyzer (CVD, whale, flow_imbalance, tick_rule_sum)
  ✅ Integra ML Features (momentum, volume_sma_ratio, etc.)
  ✅ Integra OrderBookAnalyzer corrigido
  ✅ Integra IntegrationValidator (rejeita dados ruins)
  ✅ Timestamps consistentes via TimeManager
  ✅ Event completo com TODAS as features
  ✅ Logs detalhados de qualidade
"""

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
from typing import Dict, Any, Optional, List

# Importações dos módulos do projeto
import config

# 🆕 IMPORTS DOS COMPONENTES CORRIGIDOS
from orderbook_analyzer import OrderBookAnalyzer
from flow_analyzer import FlowAnalyzer
from ml_features import generate_ml_features
from time_manager import TimeManager
from integration_validator import IntegrationValidator  # Criar este!
from event_saver import EventSaver

# Imports antigos (mantidos para compatibilidade)
from data_handler import create_absorption_event, create_exhaustion_event

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ===============================
# CONNECTION MANAGER (MANTIDO)
# ===============================

class RobustConnectionManager:
    """Gerenciador robusto de conexão WebSocket."""
    # ... (Código mantido igual ao original) ...
    
    def __init__(self, stream_url, symbol, max_reconnect_attempts=10, 
                 initial_delay=1, max_delay=60, backoff_factor=1.5):
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
            parsed_url = urlparse(self.stream_url)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == 'wss' else 80)
            
            with socket.create_connection((host, port), timeout=5) as sock:
                if parsed_url.scheme == 'wss':
                    context = ssl.create_default_context()
                    with context.wrap_socket(sock, server_hostname=host):
                        return True
                return True
        except Exception as e:
            logging.error(f"Erro ao testar conexão: {e}")
            return False
    
    def _on_message(self, ws, message):
        try:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1
            
            if self.on_message_callback:
                self.on_message_callback(ws, message)
        except Exception as e:
            logging.error(f"Erro no processamento da mensagem: {e}")
    
    def _on_open(self, ws):
        self.is_connected = True
        self.reconnect_count = 0
        self.connection_start_time = datetime.now(timezone.utc)
        self.last_message_time = self.connection_start_time
        
        logging.info(f"✅ Conexão estabelecida com {self.symbol}")
        self._start_monitoring_threads()
        
        if self.on_open_callback:
            self.on_open_callback(ws)
    
    def _on_close(self, ws, close_status_code, close_msg):
        self.is_connected = False
        logging.warning(f"🔌 Conexão fechada - Código: {close_status_code}")
        self._stop_monitoring_threads()
        
        if self.on_close_callback:
            self.on_close_callback(ws, close_status_code, close_msg)
    
    def _on_error(self, ws, error):
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
        while not self.should_stop and self.is_connected:
            time.sleep(30)
            if self.last_message_time:
                time_since_last = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                if time_since_last > 120:
                    logging.warning(f"⚠️ Sem mensagens há {time_since_last:.0f}s. Forçando reconexão.")
                    self.is_connected = False
                    break
    
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
                logging.error(f"❌ Erro ({self.reconnect_count}/{self.max_reconnect_attempts}): {e}")
                if self.reconnect_count < self.max_reconnect_attempts:
                    time.sleep(self.initial_delay * (self.backoff_factor ** self.reconnect_count))
        
        self._stop_monitoring_threads()
    
    def disconnect(self):
        logging.info("🛑 Desconectando...")
        self.should_stop = True


# ===============================
# 🆕 MARKET ANALYZER COMPLETO
# ===============================

class EnhancedMarketAnalyzer:
    """
    Analisador de mercado COMPLETO integrando TODOS os componentes.
    
    🔹 v2.0.0:
      - FlowAnalyzer (CVD, whale, flow_imbalance, tick_rule_sum)
      - ML Features (momentum, volume_sma_ratio, order_book_slope)
      - OrderBookAnalyzer (validado)
      - IntegrationValidator (rejeita dados ruins)
      - TimeManager (timestamps consistentes)
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # 🆕 TimeManager compartilhado
        self.time_manager = TimeManager()
        
        # 🆕 Componentes principais
        self.flow_analyzer = FlowAnalyzer(time_manager=self.time_manager)
        self.orderbook_analyzer = OrderBookAnalyzer(
            symbol=symbol,
            time_manager=self.time_manager,
            cache_ttl_seconds=1.0,
            max_stale_seconds=30.0,
        )
        
        # 🆕 Validador de integração
        self.validator = IntegrationValidator()
        
        # Estatísticas
        self.total_windows_processed = 0
        self.valid_events_count = 0
        self.invalid_events_count = 0
        
        logging.info(f"✅ EnhancedMarketAnalyzer inicializado para {symbol}")
    
    def process_trades(self, trades: List[Dict[str, Any]]):
        """
        Processa lista de trades e atualiza FlowAnalyzer.
        
        Args:
            trades: Lista de trades com 'p', 'q', 'T', 'm'
        """
        for trade in trades:
            try:
                self.flow_analyzer.process_trade(trade)
            except Exception as e:
                logging.debug(f"Erro ao processar trade: {e}")
    
    def analyze_window(
        self, 
        window_data: List[Dict[str, Any]], 
        window_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analisa janela COMPLETA de dados.
        
        Args:
            window_data: Lista de trades da janela
            window_id: ID da janela (para logs)
        
        Returns:
            Dict com evento completo OU None se inválido
        """
        self.total_windows_processed += 1
        
        if not window_data or len(window_data) < 2:
            logging.warning(f"⚠️ Janela {window_id} vazia ou muito pequena")
            return None
        
        try:
            # Timestamp de referência (último trade da janela)
            last_trade_ts = int(window_data[-1].get('T', 0))
            
            if last_trade_ts == 0:
                last_trade_ts = self.time_manager.now_ms()
            
            # ========================================
            # 1. PROCESSA TRADES (Flow Analyzer)
            # ========================================
            self.process_trades(window_data)
            
            # ========================================
            # 2. OBTÉM MÉTRICAS DE FLOW
            # ========================================
            flow_metrics = self.flow_analyzer.get_flow_metrics(
                reference_epoch_ms=last_trade_ts
            )
            
            # ========================================
            # 3. OBTÉM ORDERBOOK
            # ========================================
            orderbook_event = self.orderbook_analyzer.analyze(
                event_epoch_ms=last_trade_ts,
                window_id=window_id,
            )
            
            # Valida orderbook
            if not orderbook_event.get('is_valid', False):
                logging.error(
                    f"❌ Orderbook inválido na janela {window_id}: "
                    f"{orderbook_event.get('erro', 'unknown')}"
                )
                self.invalid_events_count += 1
                return None
            
            orderbook_data = orderbook_event.get('orderbook_data', {})
            
            # ========================================
            # 4. GERA ML FEATURES
            # ========================================
            df_window = pd.DataFrame(window_data)
            
            ml_features = generate_ml_features(
                df=df_window,
                orderbook_data=orderbook_data,
                flow_metrics=flow_metrics,
                lookback_windows=[1, 5, 15],
                volume_ma_window=20,
            )
            
            # ========================================
            # 5. MONTA EVENTO INTEGRADO
            # ========================================
            event = {
                "schema_version": "2.0.0",
                "tipo_evento": "MarketAnalysis",
                "ativo": self.symbol,
                "window_id": window_id,
                
                # Time index
                "time_index": self.time_manager.build_time_index(
                    last_trade_ts,
                    include_local=True,
                    timespec="milliseconds"
                ),
                
                # Flow metrics
                "cvd": flow_metrics.get('cvd', 0),
                "whale_buy_volume": flow_metrics.get('whale_buy_volume', 0),
                "whale_sell_volume": flow_metrics.get('whale_sell_volume', 0),
                "whale_delta": flow_metrics.get('whale_delta', 0),
                
                # Order flow
                "order_flow": flow_metrics.get('order_flow', {}),
                "tipo_absorcao": flow_metrics.get('tipo_absorcao', 'Neutra'),
                
                # Participant analysis
                "participant_analysis": flow_metrics.get('participant_analysis', {}),
                
                # Bursts
                "bursts": flow_metrics.get('bursts', {}),
                
                # Sector flow
                "sector_flow": flow_metrics.get('sector_flow', {}),
                
                # Orderbook
                "orderbook_data": orderbook_data,
                "orderbook_event": orderbook_event,
                
                # ML Features
                "ml_features": ml_features,
                
                # Liquidity heatmap
                "liquidity_heatmap": flow_metrics.get('liquidity_heatmap', {}),
                
                # Stats
                "trades_count": len(window_data),
                "window_duration_ms": window_data[-1]['T'] - window_data[0]['T'] if len(window_data) > 1 else 0,
            }
            
            # ========================================
            # 6. 🆕 VALIDA EVENTO INTEGRADO
            # ========================================
            validation = self.validator.validate_event(event)
            
            event["validation"] = validation
            event["is_valid"] = validation["is_valid"]
            event["should_skip"] = validation["should_skip"]
            
            # ========================================
            # 7. DECIDE SE PROCESSA
            # ========================================
            if validation["should_skip"]:
                self.invalid_events_count += 1
                
                logging.error(
                    f"❌ EVENTO INVÁLIDO (janela {window_id}): "
                    f"{validation['validation_summary']}"
                )
                
                for issue in validation["critical_issues"]:
                    logging.error(f"   🔴 {issue}")
                
                for issue in validation["issues"]:
                    logging.warning(f"   ⚠️ {issue}")
                
                return None
            
            # ========================================
            # 8. EVENTO VÁLIDO
            # ========================================
            self.valid_events_count += 1
            
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    logging.warning(f"⚡ {warning}")
            
            logging.info(
                f"✅ Janela {window_id} processada: "
                f"{len(window_data)} trades, "
                f"delta={event.get('order_flow', {}).get('net_flow_1m', 0):.2f}, "
                f"cvd={event.get('cvd', 0):.2f}"
            )
            
            return event
            
        except Exception as e:
            logging.error(f"❌ Erro ao processar janela {window_id}: {e}", exc_info=True)
            self.invalid_events_count += 1
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do analisador."""
        return {
            "total_windows": self.total_windows_processed,
            "valid_events": self.valid_events_count,
            "invalid_events": self.invalid_events_count,
            "valid_rate_pct": round(
                100 * self.valid_events_count / max(1, self.total_windows_processed),
                2
            ),
            "flow_analyzer_stats": self.flow_analyzer.get_stats(),
            "orderbook_analyzer_stats": self.orderbook_analyzer.get_stats(),
            "validator_stats": self.validator.get_stats(),
        }


# ===============================
# 🆕 BOT PRINCIPAL CORRIGIDO
# ===============================

class EnhancedMarketBot:
    """
    Bot principal com integração COMPLETA.
    
    🔹 v2.0.0:
      - Usa EnhancedMarketAnalyzer
      - Valida eventos antes de salvar
      - Logs detalhados de qualidade
    """
    
    def __init__(
        self, 
        stream_url: str, 
        symbol: str, 
        window_size_minutes: int = 5
    ):
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.ny_tz = ZoneInfo("America/New_York")

        # 🆕 Analisador completo
        self.market_analyzer = EnhancedMarketAnalyzer(symbol=symbol)
        
        # Event saver
        self.event_saver = EventSaver(sound_alert=True)

        # Gerenciador de conexão
        self.connection_manager = RobustConnectionManager(
            stream_url, 
            symbol, 
            max_reconnect_attempts=15
        )
        self.connection_manager.set_callbacks(
            on_message=self.on_message,
            on_open=self.on_open
        )

        # Estado da janela
        self.window_data = []
        self.window_end_time = None
        self.window_count = 0

    def _update_window_end_time(self):
        """Calcula próximo timestamp de fechamento de janela."""
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
        logging.info(f"🕐 Próximo fechamento: {self.window_end_time.strftime('%H:%M:%S')} NY")

    def on_message(self, ws, message):
        try:
            trade = json.loads(message)
            if "T" not in trade:
                return
            
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
        """Processa janela usando analisador completo."""
        if not self.window_data:
            return
        
        self.window_count += 1
        window_id = f"W{self.window_count:04d}"
        
        try:
            # 🆕 ANÁLISE COMPLETA
            event = self.market_analyzer.analyze_window(
                window_data=self.window_data,
                window_id=window_id
            )
            
            # ========================================
            # SALVA EVENTO SE VÁLIDO
            # ========================================
            if event is None:
                logging.warning(f"⚠️ Janela {window_id} retornou None (inválida)")
                return
            
            if event.get('should_skip', False):
                logging.warning(
                    f"⚠️ Janela {window_id} marcada para skip: "
                    f"{event.get('validation', {}).get('validation_summary')}"
                )
                return
            
            # 🆕 Salva evento completo
            self.event_saver.save_event(event)
            self._log_event(event, window_id)
            
            # Log de conclusão
            ny_time = datetime.now(self.ny_tz)
            print(f"[{ny_time.strftime('%H:%M:%S')} NY] ✅ Janela {window_id} salva")
            print("─" * 80)

        except Exception as e:
            logging.error(f"❌ Erro ao processar janela {window_id}: {e}", exc_info=True)
        finally:
            self.window_data = []

    def _log_event(self, event: Dict[str, Any], window_id: str):
        """Loga evento de forma padronizada."""
        ny_time = datetime.now(self.ny_tz)
        
        # Extrai métricas principais
        of = event.get('order_flow', {})
        ob = event.get('orderbook_data', {})
        
        delta = of.get('net_flow_1m', 0)
        flow_imb = of.get('flow_imbalance', 0)
        tick_rule = of.get('tick_rule_sum', 0)
        cvd = event.get('cvd', 0)
        
        bid_depth = ob.get('bid_depth_usd', 0)
        ask_depth = ob.get('ask_depth_usd', 0)
        ob_imb = ob.get('imbalance', 0)
        
        print(f"\n🎯 EVENTO COMPLETO - {ny_time.strftime('%H:%M:%S')} NY")
        print(f"   Janela: {window_id} | Símbolo: {self.symbol}")
        print(f"   📊 Flow: delta=${delta:,.2f}, imbalance={flow_imb:+.3f}, tick_rule={tick_rule:+.1f}")
        print(f"   💰 CVD: {cvd:.2f} BTC")
        print(f"   📚 Book: bid=${bid_depth:,.0f}, ask=${ask_depth:,.0f}, imb={ob_imb:+.3f}")
        print(f"   🎲 Absorção: {event.get('tipo_absorcao', 'N/A')}")

    def on_open(self, ws):
        ny_time = datetime.now(self.ny_tz)
        logging.info(f"🚀 Bot v2.0.0 iniciado - {ny_time.strftime('%H:%M:%S')} NY")
        self.window_end_time = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do bot."""
        return {
            "window_count": self.window_count,
            "analyzer_stats": self.market_analyzer.get_stats(),
        }

    def run(self):
        try:
            logging.info(f"🤖 Iniciando bot v2.0.0 para {self.symbol}...")
            self.connection_manager.connect()
        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        finally:
            # 🆕 Mostra stats ao encerrar
            stats = self.get_stats()
            logging.info(f"\n📊 ESTATÍSTICAS FINAIS:")
            logging.info(f"   Janelas processadas: {stats['window_count']}")
            logging.info(f"   Eventos válidos: {stats['analyzer_stats']['valid_events']}")
            logging.info(f"   Taxa de válidos: {stats['analyzer_stats']['valid_rate_pct']}%")
            
            self.connection_manager.disconnect()
            logging.info("Bot encerrado.")


# ===============================
# EXECUÇÃO
# ===============================

if __name__ == "__main__":
    bot = EnhancedMarketBot(
        stream_url=config.STREAM_URL,
        symbol=config.SYMBOL,
        window_size_minutes=5
    )
    bot.run()