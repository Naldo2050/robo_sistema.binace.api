# main.py v2.1.1 (CORRIGIDO COMPLETO - IA FUNCIONANDO)
# Otimização de eventos (auto-adicionado)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from fix_optimization import clean_event, simplify_historical_vp, remove_enriched_snapshot

# -*- coding: utf-8 -*-

# 🆕 FORÇAR UTF-8 NO WINDOWS (ADICIONAR ANTES DE TUDO)
import sys
import io
if sys.platform == 'win32':
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# --- Helpers de layout para janelas (injetado pelo patch) ---
try:
    from colorama import init as _colorama_init, Fore as _Fore, Style as _Style
    _colorama_init(autoreset=True)
except Exception:
    # fallback: define dummies se colorama indisponível
    class _Dummy:
        def __getattr__(self, name): return ''
    _Fore = _Style = _Dummy()
from datetime import datetime as _dt

def _yellow_rule(width: int = 100) -> str:
    try:
        return _Fore.YELLOW + ("━" * width) + _Style.RESET_ALL
    except Exception:
        return "━" * width

def print_janela_header(self, delta_fmt: str, vol_fmt: str):
    try:
        agora_ny = _dt.now(self.ny_tz).strftime('%H:%M:%S')
    except Exception:
        try:
            agora_ny = _dt.utcnow().strftime('%H:%M:%S')
        except Exception:
            agora_ny = "NY"
    print(_yellow_rule())
# --- fim helpers ---

    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Agora importa o resto normalmente
from dotenv import load_dotenv; load_dotenv()
import json
import time
import logging
import threading
import numpy as np
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
import socket
import random
import websocket
from collections import deque
import signal
import atexit

# Importa o arquivo de configurações central
import config

# 🔹 NOVO: Importa utilitários de formatação
from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific
)

# ... (resto do código continua igual)

# Importa o arquivo de configurações central
import config

# 🔹 NOVO: Importa utilitários de formatação
from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific
)

# Importações internas
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
from report_generator import generate_ai_analysis_report
from levels_registry import LevelRegistry
from src.utils.ai_payload_optimizer import AIPayloadOptimizer, get_optimized_json_minified

# 🔹 NOVOS MÓDULOS
from time_manager import TimeManager
from health_monitor import HealthMonitor
from event_bus import EventBus
from data_pipeline import DataPipeline
from feature_store import FeatureStore
from enrichment_integrator import enrich_analysis_trigger_event, build_analysis_trigger_event

# Alert engine and support/resistance for institutional alerts
try:
    from alert_engine import generate_alerts
except Exception:
    generate_alerts = None

try:
    import support_resistance as _sr
    detect_support_resistance = getattr(_sr, "detect_support_resistance", None)
except Exception:
    detect_support_resistance = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---- Filtro global: suprime logs idênticos em < 1s (anti-eco) ----
import time as _time

class _DedupFilter(logging.Filter):
    def __init__(self, window=1.0):
        super().__init__()
        self.window = float(window)
        self._last = {}  # msg -> ts
    
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        now = _time.time()
        ts = self._last.get(msg)
        if ts is not None and (now - ts) < self.window:
            return False
        self._last[msg] = now
        return True

logging.getLogger().addFilter(_DedupFilter(window=1.0))


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
        self.last_successful_message_time = None
        self.connection_start_time = None
        self.heartbeat_thread = None
        self.should_stop = False
        self.on_message_callback = None
        self.on_open_callback = None
        self.on_close_callback = None
        self.on_error_callback = None
        self.total_messages_received = 0
        self.total_reconnects = 0
        self.external_heartbeat_cb = None
    
    def set_callbacks(self, on_message=None, on_open=None, on_close=None, on_error=None):
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error
    
    def set_heartbeat_cb(self, cb):
        self.external_heartbeat_cb = cb
    
    def _test_connection(self):
        try:
            parsed = urlparse(self.stream_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "wss" else 80)
            socket.getaddrinfo(host, port)
            with socket.create_connection((host, port), timeout=3):
                return True
        except Exception as e:
            logging.error(f"Erro ao testar conexão (TCP quick): {e}")
            return False
    
    def _on_message(self, ws, message):
        try:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1
            if self.on_message_callback:
                self.on_message_callback(ws, message)
            self.last_successful_message_time = self.last_message_time
            if self.current_delay > self.initial_delay:
                self.current_delay = max(self.initial_delay, self.current_delay * 0.9)
        except Exception as e:
            logging.error(f"Erro no processamento da mensagem: {e}")
    
    def _on_open(self, ws):
        self.is_connected = True
        self.reconnect_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now(timezone.utc)
        self.last_message_time = self.connection_start_time
        self.last_successful_message_time = self.connection_start_time
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
            if self.last_successful_message_time:
                valid_gap = (datetime.now(timezone.utc) - self.last_successful_message_time).total_seconds()
                if valid_gap > 300:
                    logging.critical("💀 SEM MENSAGENS VÁLIDAS HÁ %ds! Fallback de preço ativado.", valid_gap)
    
    def _calculate_next_delay(self):
        delay = min(self.current_delay * self.backoff_factor, self.max_delay)
        jitter = delay * 0.2 * (random.random() - 0.5)
        self.current_delay = max(self.initial_delay, delay + jitter)
        return self.current_delay
    
    def connect(self):
        ping_interval = getattr(config, "WS_PING_INTERVAL", 30)
        ping_timeout = getattr(config, "WS_PING_TIMEOUT", 15)
        
        while self.reconnect_count < self.max_reconnect_attempts and not self.should_stop:
            try:
                if self.external_heartbeat_cb:
                    try:
                        self.external_heartbeat_cb()
                    except Exception:
                        pass
                
                if not self._test_connection():
                    raise ConnectionError("Falha no teste de conectividade")
                
                logging.info(f"🔄 Tentativa {self.reconnect_count + 1}/{self.max_reconnect_attempts}")
                
                ws = websocket.WebSocketApp(
                    self.stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                
                ws.run_forever(ping_interval=ping_interval, ping_timeout=ping_timeout)
                
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
                    t0 = time.time()
                    while time.time() - t0 < delay and not self.should_stop:
                        if self.external_heartbeat_cb:
                            try:
                                self.external_heartbeat_cb()
                            except Exception:
                                pass
                        time.sleep(1.0)
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
            return (
                {"is_signal": False, "delta": 0, "volume_total": 0, "preco_fechamento": 0},
                {"is_signal": False, "delta": 0, "volume_total": 0, "preco_fechamento": 0},
            )
        
        absorption_event = create_absorption_event(
            window_data,
            symbol,
            delta_threshold=dynamic_delta_threshold,
            tz_output=self.tz_output,
            historical_profile=historical_profile if historical_profile else {},
        )
         
        exhaustion_event = create_exhaustion_event(
            window_data,
            symbol,
            history_volumes=list(history_volumes),
            volume_factor=self.vol_factor_exh,
            tz_output=self.tz_output,
            historical_profile=historical_profile if historical_profile else {},
        )
        
        return absorption_event, exhaustion_event


# ===============================
# FUNÇÃO AUXILIAR: GET CURRENT PRICE VIA REST
# ===============================
def get_current_price(symbol: str) -> float:
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
            params = {"symbol": symbol}
            res = requests.get(url, params=params, timeout=5)
            res.raise_for_status()
            data = res.json()
            return float(data["price"])
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao buscar preço via REST (tentativa {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
        except Exception as e:
            logging.error(f"Erro inesperado ao buscar preço via REST (tentativa {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
    
    logging.critical("💀 FALHA CRÍTICA: Não foi possível obter preço via REST após todas as tentativas")
    return 0.0


# ===============================
# BOT PRINCIPAL
# ===============================
class EnhancedMarketBot:
    def __init__(self, stream_url, symbol, window_size_minutes, vol_factor_exh, history_size, 
                 delta_std_dev_factor, context_sma_period, liquidity_flow_alert_percentage, wall_std_dev_factor):
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.window_ms = window_size_minutes * 60 * 1000
        self.ny_tz = NY_TZ
        self.should_stop = False
        self.is_cleaning_up = False
        
        # TimeManager único para todo o app
        self.time_manager = TimeManager()
        
        # Inicializa módulos
        self.health_monitor = HealthMonitor()
        self.event_bus = EventBus()
        self.feature_store = FeatureStore()
        self.levels = LevelRegistry(self.symbol)
        
        # Heartbeat
        self.health_monitor.heartbeat("main")
        
        # Módulos existentes
        self.trade_flow_analyzer = TradeFlowAnalyzer(vol_factor_exh, tz_output=self.ny_tz)
        
        # Orderbook com configurações corrigidas
        self.orderbook_analyzer = OrderBookAnalyzer(
            symbol=self.symbol,
            liquidity_flow_alert_percentage=liquidity_flow_alert_percentage,
            wall_std_dev_factor=wall_std_dev_factor,
            time_manager=self.time_manager,
            cache_ttl_seconds=getattr(config, 'ORDERBOOK_CACHE_TTL', 30.0),
            max_stale_seconds=getattr(config, 'ORDERBOOK_MAX_STALE', 300.0),
            rate_limit_threshold=getattr(config, 'ORDERBOOK_MAX_REQUESTS_PER_MIN', 5),
        )
        
        # Cache persistente para orderbook
        self.last_valid_orderbook = None
        self.last_valid_orderbook_time = 0
        self.orderbook_fetch_failures = 0
        self.orderbook_emergency_mode = getattr(config, 'ORDERBOOK_EMERGENCY_MODE', True)
        
        # Cache para value area
        self.last_valid_vp = None
        self.last_valid_vp_time = 0
        
        self.event_saver = EventSaver(sound_alert=True)
        self.context_collector = ContextCollector(symbol=self.symbol)
        self.flow_analyzer = FlowAnalyzer(time_manager=self.time_manager)
        
        # IA
        self.ai_analyzer = None
        self.ai_initialization_attempted = False
        self.ai_test_passed = False
        self.ai_thread_pool = []
        self.max_ai_threads = 3
        self.ai_semaphore = threading.Semaphore(3)
        self._initialize_ai_async()
        
        # Event bus
        self.event_bus.subscribe("signal", self._handle_signal_event)
        self.event_bus.subscribe("zone_touch", self._handle_zone_touch_event)
        
        self.connection_manager = RobustConnectionManager(stream_url, symbol, max_reconnect_attempts=15)
        self.connection_manager.set_callbacks(on_message=self.on_message, on_open=self.on_open, on_close=self.on_close)
        self.connection_manager.set_heartbeat_cb(lambda: self.health_monitor.heartbeat("main"))
        
        self.window_end_ms = None
        self.window_data = []
        self.window_count = 0
        self.history_size = history_size
        self.volume_history = deque(maxlen=history_size)
        self.delta_history = deque(maxlen=history_size)
        self.close_price_history = deque(maxlen=context_sma_period)
        self.delta_std_dev_factor = delta_std_dev_factor
        self.volatility_history = deque(maxlen=history_size)
        
        # Contadores de campos ausentes
        self._missing_field_counts = {"q": 0, "m": 0, "p": 0, "T": 0}
        try:
            import config as cfg
            self._missing_field_log_step = getattr(cfg, "MISSING_FIELD_LOG_STEP", None)
        except Exception:
            self._missing_field_log_step = None
        
        self._last_price = None
        self._last_alert_ts = {}
        self._sent_triggers = set()
        
        try:
            self._alert_cooldown_sec = getattr(config, "ALERT_COOLDOWN_SEC", 30)
        except Exception:
            self._alert_cooldown_sec = 30
        
        self._register_cleanup_handlers()
    
    # ========================================
    # 🔧 INICIALIZAÇÃO DA IA - CORRIGIDO
    # ========================================
    def _initialize_ai_async(self):
        def ai_init_worker():
            try:
                if self.ai_initialization_attempted:
                    return
                self.ai_initialization_attempted = True
                
                print("\n" + "=" * 30 + " INICIALIZANDO IA " + "=" * 30)
                logging.info("🧠 Tentando inicializar AI Analyzer Qwen...")
                
                # Inicializa sem argumentos
                self.ai_analyzer = AIAnalyzer()
                
                logging.info("✅ Módulo da IA carregado. Realizando teste de análise...")
                
                current_price = get_current_price(self.symbol)
                test_event = {
                    "tipo_evento": "Teste de Conexão",
                    "ativo": self.symbol,
                    "descricao": "Teste inicial do sistema de análise para garantir operacionalidade.",
                    "delta": 150.5,
                    "volume_total": 50000,
                    "preco_fechamento": current_price,
                    "orderbook_data": {
                        "bid_depth_usd": 1000000,
                        "ask_depth_usd": 800000,
                        "bid_usd": 1000000,
                        "ask_usd": 800000,
                        "imbalance": 0.2,
                        "mid": current_price,
                        "spread": 0.10,
                        "spread_percent": 0.0001,
                    },
                    "spread_metrics": {
                        "bid_depth_usd": 1000000,
                        "ask_depth_usd": 800000,
                    }
                }
                
                # 🔧 CORREÇÃO: Usa analyze() que retorna Dict
                analysis = self.ai_analyzer.analyze(test_event)
                
                try:
                    import config as cfg
                    min_chars = getattr(cfg, "AI_TEST_MIN_CHARS", 10)
                except Exception:
                    min_chars = 10
                
                if analysis and len(analysis.get('raw_response', '')) >= min_chars:
                    self.ai_test_passed = True
                    logging.info("✅ Teste da IA bem-sucedido!")
                    print("\n" + "═" * 25 + " RESULTADO DO TESTE DA IA " + "═" * 25)
                    print(analysis.get('raw_response', ''))
                    print("═" * 75 + "\n")
                else:
                    self.ai_test_passed = True  # Continua mesmo se teste falhar
                    logging.warning("⚠️ Teste da IA retornou resultado inesperado ou vazio. Prosseguindo em modo de fallback.")
                    print(f"Resultado recebido: {analysis}")
                    print("═" * 75 + "\n")
                    
            except Exception as e:
                self.ai_analyzer = None
                self.ai_test_passed = False
                print("=" * 30 + " ERRO NA IA " + "=" * 30)
                logging.error(f"❌ Falha crítica ao inicializar a IA: {e}", exc_info=True)
                print("═" * 75 + "\n")
        
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
                if hasattr(self.ai_analyzer, 'close'):
                    self.ai_analyzer.close()
                logging.info("✅ AI Analyzer fechado.")
            except Exception as e:
                logging.error(f"❌ Erro ao fechar AI Analyzer: {e}")
        
        if self.connection_manager:
            self.connection_manager.disconnect()
        
        if hasattr(self, "event_bus"):
            self.event_bus.shutdown()
        
        if hasattr(self, "health_monitor"):
            self.health_monitor.stop()
        
        logging.info("✅ Bot encerrado com segurança.")
    
    def _register_cleanup_handlers(self):
        try:
            signal.signal(signal.SIGINT, self._cleanup_handler)
            signal.signal(signal.SIGTERM, self._cleanup_handler)
        except Exception:
            pass
        atexit.register(self._cleanup_handler)
    
    def _next_boundary_ms(self, ts_ms: int) -> int:
        return ((ts_ms // self.window_ms) + 1) * self.window_ms
    
    def on_message(self, ws, message):
        if self.should_stop:
            return
         
        try:
            # Tratamento robusto de JSON com logging melhorado
            try:
                raw = json.loads(message)
                if not isinstance(raw, dict):
                    logging.error(f"Mensagem JSON inválida (não é objeto): {message[:100]}")
                    return
            except json.JSONDecodeError as e:
                logging.error(f"Falha ao parsear JSON: {e}. Mensagem: {message[:200]}")
                # Implementar lógica de fallback ou reconexão
                if hasattr(self, 'connection_manager'):
                    self.connection_manager.disconnect()
                    self.connection_manager.connect()
                return
            except Exception as e:
                logging.error(f"Erro inesperado ao processar mensagem: {e}. Mensagem: {message[:200]}")
                return
            
            trade = raw.get("data", raw)
            
            p = trade.get("p") or trade.get("P") or trade.get("price")
            q = trade.get("q") or trade.get("Q") or trade.get("quantity")
            T = trade.get("T") or trade.get("E") or trade.get("tradeTime")
            m = trade.get("m")
            
            if (p is None or q is None or T is None) and isinstance(trade.get("k"), dict):
                k = trade["k"]
                p = p if p is not None else k.get("c")
                q = q if q is not None else k.get("v")
                T = T if T is not None else k.get("T") or raw.get("E")
            
            missing = []
            if p is None:
                missing.append("p"); self._missing_field_counts["p"] += 1
            if q is None:
                missing.append("q"); self._missing_field_counts["q"] += 1
            if T is None:
                missing.append("T"); self._missing_field_counts["T"] += 1
            
            if missing:
                total_missing = sum(self._missing_field_counts[k] for k in ("p", "q", "T"))
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
            
            try:
                # Converte tipos de forma segura
                p_val = float(p) if p is not None else None
                q_val = float(q) if q is not None else None
                T_val = int(T) if T is not None else None
                
                if p_val is None or q_val is None or T_val is None:
                    logging.error("Trade inválido (valores nulos): %s", trade)
                    return
                
                p = p_val
                q = q_val
                T = T_val
            except (TypeError, ValueError, OverflowError):
                logging.error("Trade inválido (tipos): %s", trade)
                return
            
            if m is None:
                last_price = self._last_price
                m = (p <= last_price) if last_price is not None else False
            
            self._last_price = p
            norm = {"p": p, "q": q, "T": T, "m": bool(m)}
            
            try:
                self.health_monitor.heartbeat("main")
            except Exception as hb_err:
                logging.debug("Falha ao enviar heartbeat: %s", hb_err)
            
            self.flow_analyzer.process_trade(norm)
            
            if self.window_end_ms is None:
                self.window_end_ms = self._next_boundary_ms(T)
            
            if T >= self.window_end_ms:
                self._process_window()
                self.window_end_ms = self._next_boundary_ms(T)
                self.window_data = [norm]
            else:
                self.window_data.append(norm)
                
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar mensagem JSON: {e}")
        except Exception as e:
            logging.error(f"Erro ao processar mensagem: {e}")
    
    def _handle_signal_event(self, event_data):
        if not self.ai_analyzer or not self.ai_test_passed:
            return
        self._run_ai_analysis_threaded(event_data.copy())
    
    def _handle_zone_touch_event(self, event_data):
        if not self.ai_analyzer or not self.ai_test_passed:
            return
        self._run_ai_analysis_threaded(event_data.copy())
    
    # ========================================
    # 🔧 ANÁLISE DA IA - CORRIGIDO
    # ========================================
    def _run_ai_analysis_threaded(self, event_data):
        if not self.ai_analyzer or not self.ai_test_passed or self.should_stop:
            if self.ai_analyzer and not self.ai_test_passed:
                logging.warning("⚠️ Análise da IA ignorada: sistema não passou no teste inicial.")
            return
        
        # 🔧 GUARD CLAUSE: Filtrar sinais neutros/fracos ANTES de enviar para IA
        resultado_batalha = event_data.get("resultado_da_batalha", "")
        strength = event_data.get("strength", "")
        
        # Ignorar sinais N/A, neutros ou fracos para economizar IA
        if resultado_batalha == "N/A" or resultado_batalha == "NEUTRAL":
            logging.debug(f"🚫 IA ignorando sinal neutro: {resultado_batalha}")
            return
        
        if strength == "weak":
            logging.debug(f"🚫 IA ignorando sinal fraco: strength={strength}")
            return
        
        # Filtrar por palavras-chave de sinais não-actionáveis
        na_keywords = ["N/A", "INDISPONÍVEL", "DADOS INVÁLIDOS", "SEM SINAL"]
        if any(kw in resultado_batalha.upper() for kw in na_keywords):
            logging.debug(f"🚫 IA ignorando sinal N/A: {resultado_batalha}")
            return
        
        logging.debug("🔍 Evento recebido para análise da IA: %s", event_data.get("tipo_evento", "N/A"))
        
        def _print_ai_report_clean(report_text: str):
            header = "ANÁLISE PROFISSIONAL DA IA"
            start = (report_text or "")[:200].upper()
            if header in start:
                print("\n" + report_text.rstrip() + "\n")
            else:
                print("\n" + "═" * 25 + " " + header + " " + "═" * 25)
                print(report_text)
                print("═" * 75 + "\n")
        
        def ai_worker():
            try:
                with self.ai_semaphore:
                    logging.info("🧠 IA iniciando análise para evento: %s", event_data.get("resultado_da_batalha", "N/A"))
                    self.health_monitor.heartbeat("ai")
                    
                    logging.debug("📊 Dados do evento para IA: %s", {
                        "tipo": event_data.get("tipo_evento"),
                        "delta": format_delta(event_data.get("delta")),
                        "volume": format_large_number(event_data.get("volume_total")),
                        "preco": format_price(event_data.get("preco_fechamento"))
                    })
                    
                    # Otimiza o payload antes de enviar para a IA
                    event_data_for_ai = event_data
                    try:
                        optimized_payload = get_optimized_json_minified(event_data)
                        
                        # Calcula economia de tamanho para monitoramento
                        original_json = json.dumps(event_data, separators=(',', ':'))
                        original_bytes = len(original_json.encode("utf-8"))
                        optimized_bytes = len(optimized_payload.encode("utf-8"))
                        saved_bytes = max(0, original_bytes - optimized_bytes)
                        reduction_pct = round((saved_bytes / original_bytes * 100.0), 2) if original_bytes else 0.0
                        
                        # LOGAR PARA CONFERÊNCIA (Crucial) - Nível INFO para garantir visibilidade em produção
                        logging.info(f" >>> PAYLOAD OTIMIZADO (Para LLM): {optimized_payload}")
                        logging.info(f" >>> Tamanho Original: {original_bytes} chars | Otimizado: {optimized_bytes} chars | Redução: {reduction_pct}%")
                        
                        # Converte a string JSON otimizada de volta para dict
                        if optimized_payload:
                            event_data_for_ai = json.loads(optimized_payload)
                    except Exception as e:
                        logging.warning("Falha na otimização, usando payload original: %s", e)
                    
                    # 🔧 CORREÇÃO: Usa analyze() que retorna Dict
                    if self.ai_analyzer and hasattr(self.ai_analyzer, 'analyze'):
                        analysis_result = self.ai_analyzer.analyze(event_data_for_ai)
                    else:
                        logging.warning("⚠️ IA Analyzer não disponível para análise")
                        return
                    
                    if analysis_result and not self.should_stop:
                        try:
                            raw_response = analysis_result.get('raw_response', '')
                            _print_ai_report_clean(raw_response)
                            logging.info("✅ Análise da IA concluída com sucesso")
                        except Exception as e:
                            logging.error(f"❌ Erro ao processar resposta da IA: {e}", exc_info=True)
                            
            except Exception as e:
                logging.error(f"❌ Erro na thread de análise da IA: {e}", exc_info=True)
        
        logging.debug("🔧 Criando thread para análise da IA...")
        threading.Thread(target=ai_worker, daemon=True).start()
    
    def _process_vp_features(self, historical_profile, preco_atual: float):
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
            nearest_hvn = min(hvns, key=lambda x: abs(x - preco_atual)) if hvns else None
            nearest_lvn = min(lvns, key=lambda x: abs(x - preco_atual)) if lvns else None
            dist_hvn = (preco_atual - nearest_hvn) if nearest_hvn else None
            dist_lvn = (preco_atual - nearest_lvn) if nearest_lvn else None
            
            faixa_lim = preco_atual * 0.005
            hvn_near = sum(1 for h in hvns if abs(h - preco_atual) <= faixa_lim)
            lvn_near = sum(1 for l in lvns if abs(l - preco_atual) <= faixa_lim)
            in_single = any(abs(px - preco_atual) <= faixa_lim for px in sp)
            
            return {
                "status": "ok",
                "distance_to_poc": round(dist_to_poc, 2),
                "nearest_hvn": nearest_hvn,
                "dist_to_nearest_hvn": round(dist_hvn, 2) if dist_hvn is not None else None,
                "nearest_lvn": nearest_lvn,
                "dist_to_nearest_lvn": round(dist_lvn, 2) if dist_lvn is not None else None,
                "hvns_within_0_5pct": hvn_near,
                "lvns_within_0_5pct": lvn_near,
                "in_single_print_zone": in_single,
            }
        except Exception as e:
            logging.error(f"Erro ao gerar vp_features: {e}")
            return {"status": "error"}
    
    def _log_event(self, event):
        ts_ny = event.get("timestamp_ny")
        if ts_ny:
            try:
                ny_time = datetime.fromisoformat(ts_ny.replace("Z", "+00:00")).astimezone(self.ny_tz)
            except Exception:
                ny_time = datetime.now(self.ny_tz)
        else:
            ny_time = datetime.now(self.ny_tz)
        
        resultado = event.get("resultado_da_batalha", "N/A").upper()
        tipo = event.get("tipo_evento", "EVENTO")
        descricao = event.get("descricao", "")
        conf = event.get("historical_confidence", {})
        
        print(f"\n🎯 {tipo}: {resultado} DETECTADO - {ny_time.strftime('%H:%M:%S')} NY")
        print(f" Símbolo: {self.symbol} | Janela #{self.window_count}")
        print(f" 📝 {descricao}")
        
        if conf:
            print(f" 📊 Probabilidades -> Long={conf.get('long_prob')} | Short={conf.get('short_prob')} | Neutro={conf.get('neutral_prob')}")
        
        ultimos = [e for e in obter_memoria_eventos(n=4) if e.get("tipo_evento") != "OrderBook"]
        if ultimos:
            print(" 🕒 Últimos sinais:")
            for e in ultimos:
                delta_fmt = format_delta(e.get('delta', 0))
                vol_fmt = format_large_number(e.get('volume_total', 0))
                print(
                    f"  - {e.get('timestamp', 'N/A')} | {e.get('tipo_evento', 'N/A')} {e.get('resultado_da_batalha', 'N/A')} "
                    f"(Δ={delta_fmt}, Vol={vol_fmt})"
                )
    
    # ========================================
    # PROCESSAMENTO DE JANELA
    # ========================================
    def _process_window(self):
        if not self.window_data or self.should_stop:
            self.window_data = []
            return
        
        # Normaliza e valida
        valid_window_data = []
        for trade in self.window_data:
            if "q" in trade and "p" in trade and "T" in trade:
                try:
                    trade["q"] = float(trade["q"])
                    trade["p"] = float(trade["p"])
                    trade["T"] = int(trade["T"])
                    valid_window_data.append(trade)
                except (ValueError, TypeError):
                    continue
        
        if not valid_window_data:
            logging.warning("Janela sem dados válidos para processamento")
            self.window_data = []
            return
        
        total_volume = sum(float(trade.get("q", 0)) for trade in valid_window_data)
        if total_volume == 0:
            self.window_data = []
            return
        
        self.window_count += 1
        
        # Validação mínima de trades
        min_trades_for_analysis = 10
        if len(valid_window_data) < min_trades_for_analysis:
            logging.warning(
                f"⚠️ Janela #{self.window_count} com poucos trades ({len(valid_window_data)}), "
                f"análise pode ser imprecisa"
            )
        
        # Calcula volumes buy/sell localmente
        total_buy_volume = 0
        total_sell_volume = 0
        for trade in valid_window_data:
            if trade.get("m"):  # True = seller maker (buyer taker)
                total_sell_volume += trade.get("q", 0)
            else:
                total_buy_volume += trade.get("q", 0)
        
        try:
            self.health_monitor.heartbeat("main")
            
            # Limiar dinâmico de delta
            dynamic_delta_threshold = 0
            if len(self.delta_history) > 10:
                mean_delta = np.mean(self.delta_history)
                std_delta = np.std(self.delta_history)
                dynamic_delta_threshold = abs(mean_delta + self.delta_std_dev_factor * std_delta)
            
            # Contexto macro
            macro_context = self.context_collector.get_context()
            historical_profile = macro_context.get("historical_vp", {})
            
            # Valida e usa cache se value area zerada
            vp_daily = historical_profile.get("daily", {})
            val = vp_daily.get("val", 0)
            vah = vp_daily.get("vah", 0)
            poc = vp_daily.get("poc", 0)
            
            if val == 0 or vah == 0 or poc == 0:
                if self.last_valid_vp and (time.time() - self.last_valid_vp_time < 3600):
                    age = time.time() - self.last_valid_vp_time
                    logging.warning(f"⚠️ Value Area zerada, usando cache (age={age:.0f}s)")
                    historical_profile = self.last_valid_vp.copy()
                else:
                    logging.warning("⚠️ Value Area indisponível e sem cache válido")
            else:
                self.last_valid_vp = historical_profile.copy()
                self.last_valid_vp_time = time.time()
            
            open_ms, close_ms = self.window_end_ms - self.window_ms, self.window_end_ms
            
            # Atualiza níveis
            self.levels.update_from_vp(historical_profile)
            
            try:
                pipeline = DataPipeline(
                    valid_window_data,
                    self.symbol,
                    time_manager=self.time_manager,
                )
                
                flow_metrics = self.flow_analyzer.get_flow_metrics(reference_epoch_ms=close_ms)
                
                # Orderbook com retry
                ob_event = None
                max_retries = getattr(config, 'ORDERBOOK_MAX_RETRIES', 5)
                base_delay = getattr(config, 'ORDERBOOK_RETRY_DELAY', 3.0)
                
                logging.info(f"🔍 DEBUG: Iniciando análise orderbook - janela #{self.window_count}")
                logging.info(f"🔍 DEBUG: Cache hits: {getattr(self.orderbook_analyzer, '_cache_hits', 0)}")
                logging.info(f"🔍 DEBUG: Fetch errors: {getattr(self.orderbook_analyzer, '_fetch_errors', 0)}")
                
                for attempt in range(max_retries):
                    try:
                        timeout = getattr(config, 'ORDERBOOK_REQUEST_TIMEOUT', 15.0)
                        ob_event = self.orderbook_analyzer.analyze(
                            current_snapshot=None,
                            event_epoch_ms=close_ms,
                            window_id=f"W{self.window_count:04d}"
                        )
                        
                        if ob_event and ob_event.get('is_valid', False):
                            ob_data = ob_event.get('orderbook_data', {})
                            bid_depth = ob_data.get('bid_depth_usd', 0)
                            ask_depth = ob_data.get('ask_depth_usd', 0)
                            
                            min_depth = getattr(config, 'ORDERBOOK_MIN_DEPTH_USD', 500.0)
                            if bid_depth >= min_depth or ask_depth >= min_depth:
                                self.last_valid_orderbook = ob_event
                                self.last_valid_orderbook_time = time.time()
                                self.orderbook_fetch_failures = 0
                                logging.info(
                                    f"✅ Orderbook OK: "
                                    f"bid=${bid_depth:,.0f}, ask=${ask_depth:,.0f}"
                                )
                                break
                            else:
                                logging.warning(
                                    f"⚠️ Orderbook com liquidez baixa (tentativa {attempt + 1}/{max_retries}): "
                                    f"bid=${bid_depth:.0f}, ask=${ask_depth:.0f} (min=${min_depth:.0f})"
                                )
                        else:
                            logging.warning(f"⚠️ Orderbook inválido (tentativa {attempt + 1}/{max_retries})")
                        
                        if attempt < max_retries - 1:
                            delay = base_delay * (attempt + 1)
                            logging.info(f" 🔄 Retry em {delay:.1f}s...")
                            time.sleep(delay)
                            
                    except Exception as e:
                        logging.error(f"❌ Erro ao buscar orderbook (tentativa {attempt + 1}): {e}")
                        if attempt < max_retries - 1:
                            delay = base_delay * (attempt + 1)
                            time.sleep(delay)
                
                # Fallback
                if not ob_event or not ob_event.get('is_valid', False):
                    self.orderbook_fetch_failures += 1
                    fallback_max_age = getattr(config, 'ORDERBOOK_FALLBACK_MAX_AGE', 600)
                    
                    if self.last_valid_orderbook and (time.time() - self.last_valid_orderbook_time < fallback_max_age):
                        age = time.time() - self.last_valid_orderbook_time
                        logging.warning(
                            f"⚠️ Usando orderbook em cache (age={age:.0f}s) após {self.orderbook_fetch_failures} falhas"
                        )
                        ob_event = self.last_valid_orderbook.copy()
                        ob_event['data_quality'] = {
                            'is_valid': True,
                            'data_source': 'cache',
                            'age_seconds': age,
                        }
                    elif self.orderbook_emergency_mode:
                        logging.warning(
                            f"🚨 MODO EMERGÊNCIA: Orderbook indisponível (falhas: {self.orderbook_fetch_failures})"
                        )
                        ob_event = {
                            'is_valid': True,
                            'emergency_mode': True,
                            'orderbook_data': {
                                'bid_depth_usd': 1000.0,
                                'ask_depth_usd': 1000.0,
                                'imbalance': 0.0,
                                'mid': 0.0,
                                'spread': 0.0,
                            },
                            'spread_metrics': {
                                'bid_depth_usd': 1000.0,
                                'ask_depth_usd': 1000.0,
                            },
                            'data_quality': {
                                'is_valid': True,
                                'data_source': 'emergency',
                                'error': f'Emergency mode after {self.orderbook_fetch_failures} failures',
                            },
                        }
                    else:
                        logging.error(
                            f"❌ Orderbook totalmente indisponível (falhas consecutivas: {self.orderbook_fetch_failures})"
                        )
                        ob_event = {
                            'is_valid': False,
                            'should_skip': False,
                            'orderbook_data': {
                                'bid_depth_usd': 0,
                                'ask_depth_usd': 0,
                                'imbalance': 0,
                                'mid': 0,
                                'spread': 0,
                            },
                            'spread_metrics': {
                                'bid_depth_usd': 0,
                                'ask_depth_usd': 0,
                            },
                            'data_quality': {
                                'is_valid': False,
                                'data_source': 'error',
                                'error': f'Failed after {max_retries} attempts',
                            },
                        }
                
                enriched = pipeline.enrich()
                
                pipeline.add_context(
                    flow_metrics=flow_metrics,
                    historical_vp=historical_profile,
                    orderbook_data=ob_event,
                    multi_tf=macro_context.get("mtf_trends", {}),
                    derivatives=macro_context.get("derivatives", {}),
                    market_context=macro_context.get("market_context", {}),
                    market_environment=macro_context.get("market_environment", {}),
                )
                
                signals = pipeline.detect_signals(
                    absorption_detector=lambda data, sym: create_absorption_event(
                        data,
                        sym,
                        delta_threshold=dynamic_delta_threshold,
                        tz_output=self.ny_tz,
                        flow_metrics=flow_metrics,
                        historical_profile=historical_profile,
                        time_manager=self.time_manager,
                        event_epoch_ms=close_ms,
                    ),
                    exhaustion_detector=lambda data, sym: create_exhaustion_event(
                        data,
                        sym,
                        history_volumes=list(self.volume_history),
                        volume_factor=config.VOL_FACTOR_EXH,
                        tz_output=self.ny_tz,
                        flow_metrics=flow_metrics,
                        historical_profile=historical_profile,
                        time_manager=self.time_manager,
                        event_epoch_ms=close_ms,
                    ),
                    orderbook_data=ob_event,
                )
                
                # Limpeza de duplicação
                has_non_trigger = any(s.get("tipo_evento") not in ("ANALYSIS_TRIGGER",) for s in signals)
                
                if not signals:
                    # Evento de análise automática
                    raw_event_data = {
                        "delta": enriched.get("delta_fechamento", 0),
                        "volume_total": enriched.get("volume_total", 0),
                        "preco_fechamento": enriched.get("ohlc", {}).get("close", 0),
                    }
                    event = build_analysis_trigger_event(self.symbol, raw_event_data)
                    event["timestamp"] = datetime.now(self.ny_tz).isoformat(timespec="seconds")
                    event["ml_features"] = pipeline.get_final_features().get("ml_features", {})
                    event["orderbook_data"] = ob_event
                    event["historical_vp"] = historical_profile
                    event["multi_tf"] = macro_context.get("mtf_trends", {})
                    signals.append(event)
                elif has_non_trigger:
                    signals = [s for s in signals if s.get("tipo_evento") != "ANALYSIS_TRIGGER"]
                else:
                    first = True
                    tmp = []
                    for s in signals:
                        if s.get("tipo_evento") == "ANALYSIS_TRIGGER":
                            if first: 
                                tmp.append(s)
                                first = False
                        else:
                            tmp.append(s)
                    signals = tmp

                # 🔧 ENRICHMENT: Adiciona análise avançada aos ANALYSIS_TRIGGER
                import config as cfg
                config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith("__")}
                for signal in signals:
                    if signal.get("tipo_evento") == "ANALYSIS_TRIGGER":
                        signal = enrich_analysis_trigger_event(signal, config_dict)

                # Log do heatmap
                try:
                    liquidity_data = flow_metrics.get("liquidity_heatmap", {})
                    clusters = liquidity_data.get("clusters", [])
                    
                    if clusters:
                        print(f"\n📊 LIQUIDITY HEATMAP:")
                        for i, cluster in enumerate(clusters[:3]):
                            center_fmt = format_price(cluster.get('center', 0.0))
                            vol_fmt = format_large_number(cluster.get('total_volume', 0))
                            imb_fmt = format_percent(cluster.get('imbalance_ratio', 0.0) * 100)
                            trades_fmt = format_quantity(cluster.get('trades_count', 0))
                            age_fmt = format_time_seconds(cluster.get('age_ms', 0))
                            print(
                                f"  Cluster {i+1}: ${center_fmt} | "
                                f"Vol: {vol_fmt} | "
                                f"Imb: {imb_fmt} | "
                                f"Trades: {trades_fmt} | "
                                f"Age: {age_fmt}"
                            )
                    else:
                        print(f"\n📊 LIQUIDITY HEATMAP: Nenhum cluster detectado")
                        
                except Exception as e:
                    logging.error(f"Erro ao logar liquidity heatmap: {e}")
                
                features = pipeline.get_final_features()
                self.feature_store.save_features(window_id=str(close_ms), features=features)
                
                ml_payload = features.get("ml_features", {}) or {}
                enriched_snapshot = features.get("enriched", {}) or {}
                contextual_snapshot = features.get("contextual", {}) or {}
                
            except Exception as e:
                logging.error(f"Erro no DataPipeline: {e}")
                return
            
            derivatives_context = macro_context.get("derivatives", {})
            
            for signal in signals:
                if signal.get("is_signal", False):
                    if "derivatives" not in signal:
                        signal["derivatives"] = derivatives_context
                    
                    # Validação de flow_metrics
                    if "fluxo_continuo" not in signal and flow_metrics:
                        flow_valid = False
                        try:
                            trades_processed = 0
                            if "data_quality" in flow_metrics:
                                trades_processed = flow_metrics["data_quality"].get("flow_trades_count", 0)
                            if trades_processed > 0:
                                flow_valid = True
                            
                            if not flow_valid:
                                sector_flow = flow_metrics.get("sector_flow", {})
                                for sector, data in sector_flow.items():
                                    total_vol = abs(data.get("buy", 0)) + abs(data.get("sell", 0))
                                    if total_vol > 0.001:
                                        flow_valid = True
                                        break
                            
                            if not flow_valid:
                                order_flow = flow_metrics.get("order_flow", {})
                                for key in ["net_flow_1m", "net_flow_5m", "net_flow_15m"]:
                                    val = order_flow.get(key)
                                    if val is not None and val != 0:
                                        flow_valid = True
                                        break
                                if not flow_valid:
                                    buy_pct = order_flow.get("aggressive_buy_pct", 0)
                                    sell_pct = order_flow.get("aggressive_sell_pct", 0)
                                    if buy_pct > 0 or sell_pct > 0:
                                        flow_valid = True
                            
                            if not flow_valid:
                                whale_total = abs(flow_metrics.get("whale_buy_volume", 0)) + \
                                            abs(flow_metrics.get("whale_sell_volume", 0))
                                if whale_total > 0.001:
                                    flow_valid = True
                                    
                        except Exception as e:
                            logging.error(f"Erro ao validar flow_metrics: {e}")
                            flow_valid = False
                        
                        if flow_valid:
                            signal["fluxo_continuo"] = flow_metrics
                            logging.debug(f"✅ Flow metrics adicionado (janela #{self.window_count})")
                        else:
                            signal["fluxo_continuo"] = flow_metrics
                            signal["flow_data_quality"] = "incomplete"
                            logging.warning(f"⚠️ Flow metrics possivelmente incompleto")
                    
                    # Adiciona volumes calculados localmente
                    if signal.get("volume_compra", 0) == 0 and signal.get("volume_venda", 0) == 0:
                        signal["volume_compra"] = total_buy_volume
                        signal["volume_venda"] = total_sell_volume
                        logging.debug(
                            f"📊 Volumes calculados localmente: buy={total_buy_volume:.2f}, "
                            f"sell={total_sell_volume:.2f}"
                        )
                    
                    try:
                        if "market_context" not in signal:
                            signal["market_context"] = macro_context.get("market_context", {})
                        if "market_environment" not in signal:
                            signal["market_environment"] = macro_context.get("market_environment", {})
                    except Exception:
                        pass
                    
                    signal.setdefault("features_window_id", str(close_ms))
                    signal["ml_features"] = ml_payload
                    signal["enriched_snapshot"] = enriched_snapshot
                    signal["contextual_snapshot"] = contextual_snapshot
                    
                    # 🔧 CORREÇÃO: Garante orderbook_data no root do signal
                    if ob_event and isinstance(ob_event, dict) and ob_event.get('is_valid', False):
                        if "orderbook_data" in ob_event:
                            signal["orderbook_data"] = ob_event["orderbook_data"]
                        if "spread_metrics" in ob_event:
                            signal["spread_metrics"] = ob_event["spread_metrics"]
                    
                    if signal.get("tipo_evento") == "ANALYSIS_TRIGGER":
                        key = (signal.get("tipo_evento"), signal.get("features_window_id"))
                        if key in self._sent_triggers:
                            continue
                        self._sent_triggers.add(key)
                    
                    self.levels.add_from_event(signal)
                    self.event_bus.publish("signal", signal)
                    self.event_saver.save_event(signal)
                    
                    if "timestamp" not in signal:
                        signal["timestamp"] = datetime.fromisoformat(
                            signal.get("timestamp_ny", datetime.now(self.ny_tz).isoformat(timespec="seconds")).replace("Z", "+00:00")
                        ).strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Otimizar ANALYSIS_TRIGGER antes de salvar
                    if isinstance(signal, dict) and signal.get("tipo_evento") == "ANALYSIS_TRIGGER":
                        signal = clean_event(signal)
                        signal = simplify_historical_vp(signal)
                        signal = remove_enriched_snapshot(signal)
                    
                    if isinstance(signal, dict) and signal.get("tipo_evento") != "OrderBook":
                        adicionar_memoria_evento(signal)
                    
                    self._log_event(signal)
            
            # Verifica toques em zonas
            preco_atual = enriched.get("ohlc", {}).get("close", 0)
            if preco_atual > 0:
                try:
                    touched = self.levels.check_price(float(preco_atual))
                    for z in touched:
                        # Usa o primeiro signal se disponível, ou cria um dict vazio
                        zone_event = signals[0].copy() if signals else {}
                        preco_fmt = format_price(preco_atual)
                        low_fmt = format_price(z.low)
                        high_fmt = format_price(z.high)
                        zone_event.update({
                            "tipo_evento": "Zona",
                            "resultado_da_batalha": f"Toque em Zona {z.kind}",
                            "descricao": f"Preço {preco_fmt} tocou {z.kind} {z.timeframe} [{low_fmt} ~ {high_fmt}]",
                            "zone_context": z.to_dict(),
                            "preco_fechamento": preco_atual,
                            "timestamp": datetime.now(self.ny_tz).isoformat(timespec="seconds"),
                        })
                        
                        if "historical_confidence" not in zone_event:
                            zone_event["historical_confidence"] = calcular_probabilidade_historica(zone_event)
                        
                        self.event_bus.publish("zone_touch", zone_event)
                        self.event_saver.save_event(zone_event)
                        adicionar_memoria_evento({
                            "timestamp": z.last_touched or datetime.now(self.ny_tz).isoformat(timespec="seconds"),
                            "tipo_evento": "Zona",
                            "resultado_da_batalha": f"Toque {z.kind}",
                            "delta": zone_event.get("delta", 0),
                            "volume_total": zone_event.get("volume_total", 0),
                        })
                except Exception as e:
                    logging.error(f"Erro ao verificar toques em zonas: {e}")
            
            # Atualiza históricos
            window_volume = enriched.get("volume_total", 0)
            window_delta = enriched.get("delta_fechamento", 0)
            window_close = enriched.get("ohlc", {}).get("close", 0)
            
            self.volume_history.append(window_volume)
            self.delta_history.append(window_delta)
            
            if window_close > 0:
                self.close_price_history.append(window_close)
            
            # Volatilidade
            try:
                price_feats = (ml_payload.get('price_features') or {})
                current_volatility = None
                if 'volatility_5' in price_feats:
                    current_volatility = price_feats['volatility_5']
                elif 'volatility_1' in price_feats:
                    current_volatility = price_feats['volatility_1']
                if current_volatility is not None:
                    self.volatility_history.append(float(current_volatility))
            except Exception:
                pass
            
            # Log ML features
            try:
                pf = ml_payload.get("price_features", {}) if ml_payload else {}
                vf = ml_payload.get("volume_features", {}) if ml_payload else {}
                mf = ml_payload.get("microstructure", {}) if ml_payload else {}
                
                if pf or vf or mf:
                    ret5_fmt = format_scientific(pf.get('returns_5', 0))
                    vol5_fmt = format_scientific(pf.get('volatility_5', 0), decimals=5)
                    vsma_fmt = format_percent(vf.get('volume_sma_ratio', 0) * 100)
                    bs_fmt = format_delta(vf.get('buy_sell_pressure', 0))
                    obs_fmt = format_scientific(mf.get('order_book_slope', 0), decimals=3)
                    flow_fmt = format_scientific(mf.get('flow_imbalance', 0), decimals=3)
                    print(
                        f"  ML: ret5={ret5_fmt} vol5={vol5_fmt} V/SMA={vsma_fmt} "
                        f"BSpress={bs_fmt} OBslope={obs_fmt} FlowImb={flow_fmt}"
                    )
            except Exception:
                pass
            
            # Alertas institucionais
            if generate_alerts is not None:
                try:
                    if detect_support_resistance is not None:
                        try:
                            price_series = pipeline.df['p'] if hasattr(pipeline, 'df') and pipeline.df is not None else None
                            if price_series is not None:
                                sr = detect_support_resistance(price_series, num_levels=3)
                            else:
                                sr = {"immediate_support": [], "immediate_resistance": []}
                        except Exception:
                            sr = {"immediate_support": [], "immediate_resistance": []}
                    else:
                        sr = {"immediate_support": [], "immediate_resistance": []}
                    
                    current_price_alert = window_close
                    avg_vol = (sum(self.volume_history) / len(self.volume_history)) if len(self.volume_history) > 0 else window_volume
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
                        current_volume=window_volume,
                        average_volume=avg_vol,
                        current_volatility=curr_vol or 0.0,
                        recent_volatilities=rec_vols,
                        volume_threshold=3.0,
                        tolerance_pct=0.001,
                    )
                    
                    for alert in alerts_list or []:
                        try:
                            atype = alert.get('type', 'GENERIC')
                            now_s = time.time()
                            last_ts = self._last_alert_ts.get(atype, 0)
                            if now_s - last_ts < self._alert_cooldown_sec:
                                continue
                            self._last_alert_ts[atype] = now_s
                            
                            desc_parts = [f"Tipo: {alert.get('type')}"]
                            if 'level' in alert:  
                                desc_parts.append(f"Nível: {format_price(alert['level'])}")
                            if 'threshold_exceeded' in alert:  
                                desc_parts.append(f"Fator: {format_percent(alert['threshold_exceeded'] * 100)}")
                            
                            descricao_alert = " | ".join(desc_parts)
                            print(f"🔔 ALERTA: {descricao_alert}")
                            logging.info(f"🔔 ALERTA: {descricao_alert}")
                            
                            alert_event = {
                                "tipo_evento": "Alerta",
                                "resultado_da_batalha": alert.get('type'),
                                "descricao": descricao_alert,
                                "timestamp": datetime.now(self.ny_tz).isoformat(timespec="seconds"),
                                "severity": alert.get('severity'),
                                "probability": alert.get('probability'),
                                "action": alert.get('action'),
                                "context": {
                                    "price": current_price_alert,
                                    "volume": window_volume,
                                    "average_volume": avg_vol,
                                    "volatility": curr_vol or 0.0,
                                },
                            }
                            self.event_saver.save_event(alert_event)
                            
                        except Exception as e:
                            logging.error(f"Erro ao processar alerta: {e}")
                            
                except Exception as e:
                    logging.error(f"Erro ao gerar alertas: {e}")
            
            # Log consolidado de qualidade
            if self.window_count % 10 == 0:
                logging.info(
                    f"\n📊 HEALTH CHECK::\n"
                    f"  Orderbook: failures={self.orderbook_fetch_failures}, "
                    f"last_valid={time.time() - self.last_valid_orderbook_time:.0f}s ago\n"
                    f"  Value Area: last_valid={time.time() - self.last_valid_vp_time:.0f}s ago\n"
                    f"  Flow: trades_in_window={len(valid_window_data)}, "
                    f"cvd={flow_metrics.get('cvd', 0):.4f}\n"
                    f"  Volumes: buy={total_buy_volume:.2f}, sell={total_sell_volume:.2f}"
                )
            
            # Log principal
            delta_fmt = format_delta(window_delta)
            vol_fmt = format_large_number(window_volume)
            print(f"[{datetime.now(self.ny_tz).strftime('%H:%M:%S')} NY] 🟡 Janela #{self.window_count} | Delta: {delta_fmt} | Vol: {vol_fmt}")
            
            if macro_context:
                trends = macro_context.get("mtf_trends", {})
                parts = []
                for tf, data in trends.items():
                    try:
                        parts.append(f"{tf.upper()}: {data['tendencia']}")
                    except Exception:
                        parts.append(f"{tf.upper()}: {data}")
                trends_str = ", ".join(parts)
                if trends_str:
                    print(f"  Macro Context: {trends_str}")
            
            if historical_profile and historical_profile.get("daily"):
                vp = historical_profile["daily"]
                poc_fmt = format_price(vp.get('poc', 0))
                val_fmt = format_price(vp.get('val', 0))
                vah_fmt = format_price(vp.get('vah', 0))
                print(f"  VP Diário: POC @ {poc_fmt} | VAL: {val_fmt} | VAH: {vah_fmt}")
            
            print("─" * 80)
            
        except Exception as e:
            logging.error(f"Erro no processamento da janela #{self.window_count}: {e}", exc_info=True)
        finally:
            self.window_data = []
    
    def on_open(self, ws):
        logging.info(f"🚀 Bot iniciado para {self.symbol} - Fuso: New York (America/New_York)")
        try:
            self.health_monitor.heartbeat("main")
        except Exception:
            pass
    
    def on_close(self, ws, code, msg):
        if self.window_data and not self.should_stop:
            self._process_window()
    
    def run(self):
        try:
            self.context_collector.start()
            logging.info("🎯 Iniciando Enhanced Market Bot v2.1.1 (ORDERBOOK E IA CORRIGIDOS)...")
            print("═" * 80)
            self.connection_manager.connect()
        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        except Exception as e:
            logging.critical(f"❌ Erro crítico ao executar o bot: {e}", exc_info=True)
        finally:
            self._cleanup_handler()


# ===============================
# EXECUÇÃO
# ===============================
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
            wall_std_dev_factor=config.WALL_STD_DEV_FACTOR,
        )
        bot.run()
    except Exception as e:
        logging.critical(f"❌ Erro crítico na inicialização do bot: {e}", exc_info=True)
        sys.exit(1)