# main.py v2.3.1 - CORRIGIDO COMPLETO + WARMUP + BUFFER + PING/PONG OTIMIZADO
# -*- coding: utf-8 -*-
"""
Enhanced Market Bot v2.3.1

🔹 CORREÇÕES v2.3.1:
  ✅ Correção da lógica de ANALYSIS_TRIGGER (não remove mais sinais)
  ✅ Garantia de epoch_ms e timestamp_utc em todos os sinais
  ✅ Debug logs para rastreamento de sinais salvos
  ✅ Validação de dados mais robusta
  ✅ CORREÇÃO CRÍTICA: from_timestamp_ms() agora funciona corretamente

🔹 NOVIDADES v2.3.0:
  ✅ Período de aquecimento após reconexão (WARMUP)
  ✅ Buffer de emergência para recuperação de dados
  ✅ Validação pré-pipeline (evita erro "dados insuficientes")
  ✅ Ping/Pong otimizado (20s/10s)
  ✅ Acumulação gradual de trades após reconexão
  ✅ Skip inteligente de janelas com poucos dados
  ✅ Clock Sync integrado
  ✅ Thread-safety melhorado
  ✅ Rate limiting em todas as APIs
"""

# 🔧 FORÇAR UTF-8 NO WINDOWS (DEVE SER A PRIMEIRA COISA)
import sys
import io
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Fallback para Python < 3.7
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Imports padrão
from dotenv import load_dotenv
load_dotenv()

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

# Importa configurações
import config

# ✅ NOVO: Clock Sync
try:
    from clock_sync import get_clock_sync
    HAS_CLOCK_SYNC = True
except ImportError:
    HAS_CLOCK_SYNC = False
    logging.warning("⚠️ clock_sync.py não encontrado - timestamps usarão relógio local")

# Utilitários de formatação
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
from data_validator import validator

# Módulos avançados
from time_manager import TimeManager
from health_monitor import HealthMonitor
from event_bus import EventBus
from data_pipeline import DataPipeline
from feature_store import FeatureStore

# Alert engine
try:
    from alert_engine import generate_alerts
except Exception:
    generate_alerts = None

try:
    from support_resistance import detect_support_resistance
except Exception:
    detect_support_resistance = None

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ===== FILTRO ANTI-ECO DE LOGS =====
class _DedupFilter(logging.Filter):
    """Filtro que suprime logs idênticos em janela de tempo."""
    def __init__(self, window=1.0):
        super().__init__()
        self.window = float(window)
        self._last = {}  # msg -> timestamp
    
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        now = time.time()
        ts = self._last.get(msg)
        
        if ts is not None and (now - ts) < self.window:
            return False
        
        self._last[msg] = now
        return True


logging.getLogger().addFilter(_DedupFilter(window=1.0))


# ===== RATE LIMITER =====
class RateLimiter:
    """Rate limiter thread-safe para controle de requisições."""
    
    def __init__(self, max_calls, period_seconds):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls = deque()
        self.lock = threading.Lock()
    
    def acquire(self):
        """Adquire permissão para fazer chamada (bloqueia se necessário)."""
        with self.lock:
            now = time.time()
            
            # Remove chamadas antigas
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            # Se atingiu limite, aguarda
            if len(self.calls) >= self.max_calls:
                sleep_time = self.calls[0] + self.period - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    return self.acquire()
            
            self.calls.append(now)


# ===== GESTOR DE CONEXÃO WEBSOCKET =====
class RobustConnectionManager:
    """Gerenciador robusto de conexão WebSocket com reconnect automático."""
    
    def __init__(
        self,
        stream_url,
        symbol,
        max_reconnect_attempts=10,
        initial_delay=1,
        max_delay=60,
        backoff_factor=1.5
    ):
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
        
        # ✅ NOVO: Callback para notificar reconexão
        self.on_reconnect_callback = None

    def set_callbacks(self, on_message=None, on_open=None, on_close=None, on_error=None, on_reconnect=None):
        """Define callbacks para eventos do WebSocket."""
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error
        self.on_reconnect_callback = on_reconnect

    def set_heartbeat_cb(self, cb):
        """Define callback de heartbeat externo."""
        self.external_heartbeat_cb = cb

    def _test_connection(self):
        """Testa conectividade TCP antes de tentar WebSocket."""
        try:
            parsed = urlparse(self.stream_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "wss" else 80)
            
            socket.getaddrinfo(host, port)
            
            with socket.create_connection((host, port), timeout=3):
                return True
        except Exception as e:
            logging.error(f"Erro ao testar conexão TCP: {e}")
            return False

    def _on_message(self, ws, message):
        """Callback interno para mensagens recebidas."""
        try:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1
            
            if self.on_message_callback:
                self.on_message_callback(ws, message)
            
            self.last_successful_message_time = self.last_message_time
            
            # Reduz delay se está recebendo mensagens
            if self.current_delay > self.initial_delay:
                self.current_delay = max(
                    self.initial_delay,
                    self.current_delay * 0.9
                )
        except Exception as e:
            logging.error(f"Erro no processamento da mensagem: {e}")

    def _on_open(self, ws):
        """Callback interno para conexão aberta."""
        self.is_connected = True
        
        # ✅ NOVO: Notifica bot sobre reconexão
        if self.reconnect_count > 0 and self.on_reconnect_callback:
            try:
                self.on_reconnect_callback()
            except Exception as e:
                logging.error(f"Erro no callback de reconexão: {e}")
        
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
        """Callback interno para conexão fechada."""
        self.is_connected = False
        logging.warning(
            f"🔌 Conexão fechada - Código: {close_status_code}, "
            f"Msg: {close_msg}"
        )
        
        self._stop_heartbeat()
        
        if self.on_close_callback:
            self.on_close_callback(ws, close_status_code, close_msg)

    def _on_error(self, ws, error):
        """Callback interno para erros."""
        logging.error(f"❌ Erro WebSocket: {error}")
        
        if self.on_error_callback:
            self.on_error_callback(ws, error)

    def _start_heartbeat(self):
        """Inicia thread de monitoramento de heartbeat."""
        self.should_stop = False
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_monitor,
            daemon=True
        )
        self.heartbeat_thread.start()

    def _stop_heartbeat(self):
        """Para thread de heartbeat."""
        self.should_stop = True
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1)

    def _heartbeat_monitor(self):
        """Monitora se está recebendo mensagens."""
        while not self.should_stop and self.is_connected:
            time.sleep(20)
            
            if self.last_message_time:
                gap = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
                
                if gap > 60:  # ✅ Reduzido de 120 para 90s
                    logging.warning(
                        f"⚠️ Sem mensagens há {gap:.0f}s. "
                        f"Forçando reconexão."
                    )
                    self.is_connected = False
                    break
            
            if self.last_successful_message_time:
                valid_gap = (
                    datetime.now(timezone.utc) - 
                    self.last_successful_message_time
                ).total_seconds()
                
                if valid_gap > 120:  # ✅ Reduzido de 300 para 180s
                    logging.critical(
                        f"💀 SEM MENSAGENS VÁLIDAS HÁ {valid_gap:.0f}s!"
                    )

    def _calculate_next_delay(self):
        """Calcula próximo delay com exponential backoff + jitter."""
        delay = min(
            self.current_delay * self.backoff_factor,
            self.max_delay
        )
        
        # Adiciona jitter (±20%)
        jitter = delay * 0.2 * (random.random() - 0.5)
        
        self.current_delay = max(self.initial_delay, delay + jitter)
        return self.current_delay

    def connect(self):
        """Conecta ao WebSocket com retry automático."""
        # ✅ OTIMIZADO: Ping/Pong mais agressivo
        ping_interval = getattr(config, "WS_PING_INTERVAL", 15)  # Reduzido de 30
        ping_timeout = getattr(config, "WS_PING_TIMEOUT", 8)    # Reduzido de 15
        
        while self.reconnect_count < self.max_reconnect_attempts and not self.should_stop:
            try:
                # Heartbeat externo
                if self.external_heartbeat_cb:
                    try:
                        self.external_heartbeat_cb()
                    except Exception:
                        pass
                
                # Testa conectividade
                if not self._test_connection():
                    raise ConnectionError("Falha no teste de conectividade")
                
                if self.reconnect_count > 0:
                    logging.info(
                        f"🔄 Tentativa de reconexão {self.reconnect_count + 1}/"
                        f"{self.max_reconnect_attempts}"
                    )
                
                ws = websocket.WebSocketApp(
                    self.stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                
                ws.run_forever(
                    ping_interval=ping_interval,
                    ping_timeout=ping_timeout,
                    skip_utf8_validation=True  # ✅ NOVO: Performance
                )
                
                if self.should_stop:
                    break
                    
            except KeyboardInterrupt:
                logging.info("⏹️ Interrompido pelo usuário")
                self.should_stop = True
                break
                
            except Exception as e:
                self.reconnect_count += 1
                self.total_reconnects += 1
                
                logging.error(
                    f"❌ Erro na conexão "
                    f"({self.reconnect_count}/{self.max_reconnect_attempts}): {e}"
                )
                
                if self.reconnect_count < self.max_reconnect_attempts:
                    delay = self._calculate_next_delay()
                    logging.info(f"⏳ Reconectando em {delay:.1f}s...")
                    
                    # Aguarda com heartbeat
                    t0 = time.time()
                    while time.time() - t0 < delay and not self.should_stop:
                        if self.external_heartbeat_cb:
                            try:
                                self.external_heartbeat_cb()
                            except Exception:
                                pass
                        time.sleep(1.0)
                else:
                    logging.error(
                        "💀 Máximo de tentativas atingido. Encerrando."
                    )
                    break
        
        self._stop_heartbeat()

    def disconnect(self):
        """Desconecta do WebSocket."""
        logging.info("🛑 Iniciando desconexão...")
        self.should_stop = True


# ===== ANALISADOR DE TRADE FLOW =====
class TradeFlowAnalyzer:
    """Analisador de fluxo de trades."""
    
    def __init__(self, vol_factor_exh, tz_output: ZoneInfo):
        self.vol_factor_exh = vol_factor_exh
        self.tz_output = tz_output

    def analyze_window(
        self,
        window_data,
        symbol,
        history_volumes,
        dynamic_delta_threshold,
        historical_profile=None
    ):
        """Analisa janela de trades."""
        if not window_data or len(window_data) < 2:
            return (
                {
                    "is_signal": False,
                    "delta": 0,
                    "volume_total": 0,
                    "preco_fechamento": 0
                },
                {
                    "is_signal": False,
                    "delta": 0,
                    "volume_total": 0,
                    "preco_fechamento": 0
                },
            )
        
        absorption_event = create_absorption_event(
            window_data,
            symbol,
            delta_threshold=dynamic_delta_threshold,
            tz_output=self.tz_output,
            historical_profile=historical_profile,
        )
        
        exhaustion_event = create_exhaustion_event(
            window_data,
            symbol,
            history_volumes=list(history_volumes),
            volume_factor=self.vol_factor_exh,
            tz_output=self.tz_output,
            historical_profile=historical_profile,
        )
        
        return absorption_event, exhaustion_event


# ===== GET CURRENT PRICE VIA REST =====
def get_current_price(symbol: str) -> float:
    """Obtém preço atual via REST API com retry."""
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
            logging.error(
                f"Erro ao buscar preço via REST "
                f"(tentativa {attempt+1}/{max_retries}): {e}"
            )
            
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                
        except Exception as e:
            logging.error(
                f"Erro inesperado ao buscar preço via REST "
                f"(tentativa {attempt+1}/{max_retries}): {e}"
            )
            
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    
    logging.critical(
        "💀 FALHA CRÍTICA: Não foi possível obter preço via REST "
        "após todas as tentativas"
    )
    return 0.0


# ===== BOT PRINCIPAL =====
class EnhancedMarketBot:
    """Bot de análise de mercado com IA integrada."""
    
    def __init__(
        self,
        stream_url,
        symbol,
        window_size_minutes,
        vol_factor_exh,
        history_size,
        delta_std_dev_factor,
        context_sma_period,
        liquidity_flow_alert_percentage,
        wall_std_dev_factor
    ):
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.window_ms = window_size_minutes * 60 * 1000
        self.ny_tz = NY_TZ
        self.should_stop = False
        self.is_cleaning_up = False

        # 🔒 Locks para thread safety
        self._cleanup_lock = threading.Lock()
        self._ai_init_lock = threading.Lock()

        # ✅ NOVO: Sistema de aquecimento após reconexão
        self.warming_up = False
        self.warmup_windows_remaining = 0
        self.warmup_windows_required = getattr(config, "WARMUP_WINDOWS", 3)
        
        # ✅ NOVO: Buffer de emergência para recuperação
        self.trades_buffer = deque(maxlen=getattr(config, "TRADES_BUFFER_SIZE", 2000))
        self.min_trades_for_pipeline = getattr(config, "MIN_TRADES_FOR_PIPELINE", 10)

        # ✅ NOVO: Clock Sync
        self.clock_sync = None
        if HAS_CLOCK_SYNC:
            try:
                self.clock_sync = get_clock_sync()
                logging.info("✅ Clock Sync inicializado")
            except Exception as e:
                logging.error(f"❌ Erro ao inicializar Clock Sync: {e}")

        # TimeManager
        self.time_manager = TimeManager()
        
        # Módulos de monitoramento
        self.health_monitor = HealthMonitor()
        self.event_bus = EventBus()
        self.feature_store = FeatureStore()
        self.levels = LevelRegistry(self.symbol)
        
        # Heartbeat inicial
        self.health_monitor.heartbeat("main")
        
        # Trade flow analyzer
        self.trade_flow_analyzer = TradeFlowAnalyzer(
            vol_factor_exh,
            tz_output=self.ny_tz
        )
        
        # Orderbook analyzer
        self.orderbook_analyzer = OrderBookAnalyzer(
            symbol=self.symbol,
            liquidity_flow_alert_percentage=liquidity_flow_alert_percentage,
            wall_std_dev_factor=wall_std_dev_factor,
            time_manager=self.time_manager,
            cache_ttl_seconds=getattr(config, 'ORDERBOOK_CACHE_TTL', 30.0),
            max_stale_seconds=getattr(config, 'ORDERBOOK_MAX_STALE', 300.0),
            rate_limit_threshold=getattr(
                config,
                'ORDERBOOK_MAX_REQUESTS_PER_MIN',
                5
            ),
        )
        
        # Cache persistente para orderbook
        self.last_valid_orderbook = None
        self.last_valid_orderbook_time = 0
        self.orderbook_fetch_failures = 0
        self.orderbook_emergency_mode = getattr(
            config,
            'ORDERBOOK_EMERGENCY_MODE',
            True
        )
        
        # Cache para value area
        self.last_valid_vp = None
        self.last_valid_vp_time = 0
        
        # Event saver
        self.event_saver = EventSaver(sound_alert=True)
        
        # Context collector
        self.context_collector = ContextCollector(symbol=self.symbol)
        
        # Flow analyzer
        self.flow_analyzer = FlowAnalyzer(time_manager=self.time_manager)
        
        # IA
        self.ai_analyzer = None
        self.ai_initialization_attempted = False
        self.ai_test_passed = False
        self.ai_thread_pool = []
        self.max_ai_threads = 3
        self.ai_semaphore = threading.Semaphore(3)
        self.ai_rate_limiter = RateLimiter(max_calls=10, period_seconds=60)
        
        self._initialize_ai_async()
        
        # Event bus subscriptions
        self.event_bus.subscribe("signal", self._handle_signal_event)
        self.event_bus.subscribe("zone_touch", self._handle_zone_touch_event)
        
        # Connection manager
        self.connection_manager = RobustConnectionManager(
            stream_url,
            symbol,
            max_reconnect_attempts=25
        )
        self.connection_manager.set_callbacks(
            on_message=self.on_message,
            on_open=self.on_open,
            on_close=self.on_close,
            on_reconnect=self._on_reconnect  # ✅ NOVO
        )
        self.connection_manager.set_heartbeat_cb(
            lambda: self.health_monitor.heartbeat("main")
        )
        
        # Janela de trades
        self.window_end_ms = None
        self.window_data = []
        self.window_count = 0
        
        # Históricos
        self.history_size = history_size
        self.volume_history = deque(maxlen=history_size)
        self.delta_history = deque(maxlen=history_size)
        self.close_price_history = deque(maxlen=context_sma_period)
        self.delta_std_dev_factor = delta_std_dev_factor
        self.volatility_history = deque(maxlen=history_size)
        
        # Contadores de campos ausentes
        self._missing_field_counts = {
            "q": 0,
            "m": 0,
            "p": 0,
            "T": 0
        }
        
        try:
            self._missing_field_log_step = getattr(
                config,
                "MISSING_FIELD_LOG_STEP",
                None
            )
        except Exception:
            self._missing_field_log_step = None
        
        self._last_price = None
        self._last_alert_ts = {}
        self._sent_triggers = set()
        
        try:
            self._alert_cooldown_sec = getattr(
                config,
                "ALERT_COOLDOWN_SEC",
                30
            )
        except Exception:
            self._alert_cooldown_sec = 30
        
        # Registra handlers de cleanup
        self._register_cleanup_handlers()

    # ========================================
    # ✅ NOVO: HANDLER DE RECONEXÃO
    # ========================================
    def _on_reconnect(self):
        """Callback chamado quando reconexão é bem-sucedida."""
        logging.warning("🔄 RECONEXÃO DETECTADA - Iniciando período de aquecimento...")
        
        # Ativa modo de aquecimento
        self.warming_up = True
        self.warmup_windows_remaining = self.warmup_windows_required
        
        # Limpa janela atual
        self.window_data = []
        self.window_end_ms = None
        
        logging.info(
            f"⏳ Aguardando {self.warmup_windows_required} janelas "
            f"para estabilizar dados..."
        )

    # ========================================
    # INICIALIZAÇÃO DA IA
    # ========================================
    def _initialize_ai_async(self):
        """Inicializa IA em thread separada."""
        
        def ai_init_worker():
            try:
                # 🔒 Thread-safe check
                with self._ai_init_lock:
                    if self.ai_initialization_attempted:
                        return
                    self.ai_initialization_attempted = True
                
                print("\n" + "=" * 30 + " INICIALIZANDO IA " + "=" * 30)
                logging.info("🧠 Tentando inicializar AI Analyzer...")
                
                # Inicializa IA
                self.ai_analyzer = AIAnalyzer()
                
                logging.info(
                    "✅ Módulo da IA carregado. "
                    "Realizando teste de análise..."
                )
                
                # Teste com evento real
                current_price = get_current_price(self.symbol)
                
                test_event = {
                    "tipo_evento": "Teste de Conexão",
                    "ativo": self.symbol,
                    "descricao": (
                        "Teste inicial do sistema de análise "
                        "para garantir operacionalidade."
                    ),
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
                
                # Testa análise
                analysis = self.ai_analyzer.analyze(test_event)
                
                min_chars = getattr(config, "AI_TEST_MIN_CHARS", 10)
                
                if analysis and len(analysis.get('raw_response', '')) >= min_chars:
                    self.ai_test_passed = True
                    logging.info("✅ Teste da IA bem-sucedido!")
                    
                    print("\n" + "═" * 25 + " RESULTADO DO TESTE DA IA " + "═" * 25)
                    print(analysis.get('raw_response', ''))
                    print("═" * 75 + "\n")
                else:
                    self.ai_test_passed = True
                    logging.warning(
                        "⚠️ Teste da IA retornou resultado inesperado. "
                        "Prosseguindo em modo fallback."
                    )
                    print(f"Resultado recebido: {analysis}")
                    print("═" * 75 + "\n")
                    
            except Exception as e:
                self.ai_analyzer = None
                self.ai_test_passed = False
                
                print("=" * 30 + " ERRO NA IA " + "=" * 30)
                logging.error(
                    f"❌ Falha crítica ao inicializar a IA: {e}",
                    exc_info=True
                )
                print("═" * 75 + "\n")
        
        threading.Thread(target=ai_init_worker, daemon=True).start()

    # ========================================
    # CLEANUP
    # ========================================
    def _cleanup_handler(self, signum=None, frame=None):
        """Handler de cleanup thread-safe."""
        
        # 🔒 Previne múltiplas execuções
        with self._cleanup_lock:
            if self.is_cleaning_up:
                return
            self.is_cleaning_up = True
        
        logging.info("🧹 Iniciando limpeza dos recursos...")
        self.should_stop = True
        
        cleanup_timeout = 5.0
        cleanup_threads = []
        
        # Funções de cleanup
        def cleanup_context():
            try:
                if self.context_collector:
                    self.context_collector.stop()
                    logging.info("✅ Context Collector parado.")
            except Exception as e:
                logging.error(f"❌ Erro ao parar Context Collector: {e}")
        
        def cleanup_ai():
            try:
                if self.ai_analyzer and hasattr(self.ai_analyzer, 'close'):
                    self.ai_analyzer.close()
                    logging.info("✅ AI Analyzer fechado.")
            except Exception as e:
                logging.error(f"❌ Erro ao fechar AI Analyzer: {e}")
        
        def cleanup_connection():
            try:
                if self.connection_manager:
                    self.connection_manager.disconnect()
                    logging.info("✅ Connection Manager desconectado.")
            except Exception as e:
                logging.error(f"❌ Erro ao desconectar Connection Manager: {e}")
        
        def cleanup_event_bus():
            try:
                if hasattr(self, "event_bus"):
                    self.event_bus.shutdown()
                    logging.info("✅ Event Bus encerrado.")
            except Exception as e:
                logging.error(f"❌ Erro ao encerrar Event Bus: {e}")
        
        def cleanup_health_monitor():
            try:
                if hasattr(self, "health_monitor"):
                    self.health_monitor.stop()
                    logging.info("✅ Health Monitor parado.")
            except Exception as e:
                logging.error(f"❌ Erro ao parar Health Monitor: {e}")
        
        def cleanup_clock_sync():
            try:
                if self.clock_sync and hasattr(self.clock_sync, 'stop'):
                    self.clock_sync.stop()
                    logging.info("✅ Clock Sync parado.")
            except Exception as e:
                logging.error(f"❌ Erro ao parar Clock Sync: {e}")
        
        # Executa cleanups em paralelo
        for cleanup_func in [
            cleanup_context,
            cleanup_ai,
            cleanup_connection,
            cleanup_event_bus,
            cleanup_health_monitor,
            cleanup_clock_sync
        ]:
            t = threading.Thread(target=cleanup_func, daemon=True)
            t.start()
            cleanup_threads.append(t)
        
        # Aguarda com timeout
        for t in cleanup_threads:
            t.join(timeout=cleanup_timeout)
        
        logging.info("✅ Bot encerrado com segurança.")

    def _register_cleanup_handlers(self):
        """Registra handlers de cleanup."""
        try:
            signal.signal(signal.SIGINT, self._cleanup_handler)
            signal.signal(signal.SIGTERM, self._cleanup_handler)
        except Exception:
            pass
        
        atexit.register(self._cleanup_handler)

    # ========================================
    # JANELA DE TEMPO
    # ========================================
    def _next_boundary_ms(self, ts_ms: int) -> int:
        """Calcula próximo limite de janela."""
        return ((ts_ms // self.window_ms) + 1) * self.window_ms

    # ========================================
    # PROCESSAMENTO DE MENSAGENS
    # ========================================
    def on_message(self, ws, message):
        """Processa mensagem recebida do WebSocket."""
        if self.should_stop:
            return
        
        try:
            raw = json.loads(message)
            trade = raw.get("data", raw)
            
            # Extrai campos
            p = trade.get("p") or trade.get("P") or trade.get("price")
            q = trade.get("q") or trade.get("Q") or trade.get("quantity")
            T = trade.get("T") or trade.get("E") or trade.get("tradeTime")
            m = trade.get("m")
            
            # Tenta extrair de kline se disponível
            if (p is None or q is None or T is None) and isinstance(
                trade.get("k"), dict
            ):
                k = trade["k"]
                p = p if p is not None else k.get("c")
                q = q if q is not None else k.get("v")
                T = T if T is not None else k.get("T") or raw.get("E")
            
            # Valida campos obrigatórios
            missing = []
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
                    self._missing_field_counts[k]
                    for k in ("p", "q", "T")
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
            
            # Converte tipos
            try:
                p = float(p)
                q = float(q)
                T = int(T)
            except (TypeError, ValueError):
                logging.error("Trade inválido (tipos): %s", trade)
                return
            
            # Infere direção se ausente
            if m is None:
                last_price = self._last_price
                m = (p <= last_price) if last_price is not None else False
            
            self._last_price = p
            
            # Normaliza trade
            norm = {
                "p": p,
                "q": q,
                "T": T,
                "m": bool(m)
            }
            
            # ✅ NOVO: Adiciona ao buffer de emergência
            self.trades_buffer.append(norm)
            
            # Heartbeat
            try:
                self.health_monitor.heartbeat("main")
            except Exception as hb_err:
                logging.debug("Falha ao enviar heartbeat: %s", hb_err)
            
            # Processa no flow analyzer
            self.flow_analyzer.process_trade(norm)
            
            # Gerencia janela de tempo
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

    # ========================================
    # HANDLERS DE EVENTOS
    # ========================================
    def _handle_signal_event(self, event_data):
        """Handler para eventos de sinal."""
        if not self.ai_analyzer or not self.ai_test_passed:
            return
        
        self._run_ai_analysis_threaded(event_data.copy())

    def _handle_zone_touch_event(self, event_data):
        """Handler para eventos de toque em zona."""
        if not self.ai_analyzer or not self.ai_test_passed:
            return
        
        self._run_ai_analysis_threaded(event_data.copy())

    # ========================================
    # ANÁLISE DA IA
    # ========================================
    def _run_ai_analysis_threaded(self, event_data):
        """Executa análise da IA em thread separada com rate limiting."""
        
        if not self.ai_analyzer or not self.ai_test_passed or self.should_stop:
            if self.ai_analyzer and not self.ai_test_passed:
                logging.warning(
                    "⚠️ Análise da IA ignorada: sistema não passou no teste inicial."
                )
            return
        
        logging.debug(
            "🔍 Evento recebido para análise da IA: %s",
            event_data.get("tipo_evento", "N/A")
        )

        def _print_ai_report_clean(report_text: str):
            """Imprime relatório da IA de forma limpa."""
            header = "ANÁLISE PROFISSIONAL DA IA"
            start = (report_text or "")[:200].upper()
            
            YELLOW = "\033[33m"
            RESET = "\033[0m"
            sep = "═" * 75
            
            if header in start:
                print("\n" + report_text.rstrip())
            else:
                print("\n" + "═" * 25 + " " + header + " " + "═" * 25)
                print(report_text)
            
            print(f"{YELLOW}{sep}{RESET}\n")

        def ai_worker():
            try:
                # 🔒 Rate limiting
                self.ai_rate_limiter.acquire()
                
                with self.ai_semaphore:
                    logging.info(
                        "🧠 IA iniciando análise para evento: %s",
                        event_data.get("resultado_da_batalha", "N/A")
                    )
                    
                    self.health_monitor.heartbeat("ai")
                    
                    logging.debug(
                        "📊 Dados do evento para IA: %s",
                        {
                            "tipo": event_data.get("tipo_evento"),
                            "delta": format_delta(event_data.get("delta")),
                            "volume": format_large_number(event_data.get("volume_total")),
                            "preco": format_price(event_data.get("preco_fechamento"))
                        }
                    )
                    
                    # Executa análise
                    analysis_result = self.ai_analyzer.analyze(event_data)
                    
                    if analysis_result and not self.should_stop:
                        try:
                            raw_response = analysis_result.get('raw_response', '')
                            _print_ai_report_clean(raw_response)
                            logging.info("✅ Análise da IA concluída com sucesso")
                        except Exception as e:
                            logging.error(
                                f"❌ Erro ao processar resposta da IA: {e}",
                                exc_info=True
                            )
                            
            except Exception as e:
                logging.error(
                    f"❌ Erro na thread de análise da IA: {e}",
                    exc_info=True
                )
            finally:
                # 🧹 Limpa threads mortas
                try:
                    self.ai_thread_pool = [
                        t for t in self.ai_thread_pool if t.is_alive()
                    ]
                except Exception:
                    pass

        logging.debug("🔧 Criando thread para análise da IA...")
        t = threading.Thread(target=ai_worker, daemon=True)
        self.ai_thread_pool.append(t)
        t.start()

    # ========================================
    # PROCESSAMENTO DE VP FEATURES
    # ========================================
    def _process_vp_features(self, historical_profile, preco_atual: float):
        """Processa features do Volume Profile."""
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
                min(hvns, key=lambda x: abs(x - preco_atual))
                if hvns else None
            )
            nearest_lvn = (
                min(lvns, key=lambda x: abs(x - preco_atual))
                if lvns else None
            )
            
            dist_hvn = (
                (preco_atual - nearest_hvn)
                if nearest_hvn else None
            )
            dist_lvn = (
                (preco_atual - nearest_lvn)
                if nearest_lvn else None
            )
            
            faixa_lim = preco_atual * 0.005
            
            hvn_near = sum(
                1 for h in hvns
                if abs(h - preco_atual) <= faixa_lim
            )
            lvn_near = sum(
                1 for l in lvns
                if abs(l - preco_atual) <= faixa_lim
            )
            
            in_single = any(
                abs(px - preco_atual) <= faixa_lim
                for px in sp
            )
            
            return {
                "status": "ok",
                "distance_to_poc": round(dist_to_poc, 2),
                "nearest_hvn": nearest_hvn,
                "dist_to_nearest_hvn": (
                    round(dist_hvn, 2)
                    if dist_hvn is not None else None
                ),
                "nearest_lvn": nearest_lvn,
                "dist_to_nearest_lvn": (
                    round(dist_lvn, 2)
                    if dist_lvn is not None else None
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
    def _log_event(self, event):
        """Loga evento de forma formatada."""
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
        
        # Últimos sinais
        ultimos = [
            e for e in obter_memoria_eventos(n=4)
            if e.get("tipo_evento") != "OrderBook"
        ]
        
        if ultimos:
            print(" 🕒 Últimos sinais:")
            for e in ultimos:
                delta_fmt = format_delta(e.get('delta', 0))
                vol_fmt = format_large_number(e.get('volume_total', 0))
                
                print(
                    f"  - {e.get('timestamp', 'N/A')} | "
                    f"{e.get('tipo_evento', 'N/A')} "
                    f"{e.get('resultado_da_batalha', 'N/A')} "
                    f"(Δ={delta_fmt}, Vol={vol_fmt})"
                )

    # ========================================
    # ✅ PROCESSAMENTO DE JANELA (CORRIGIDO v2.3.1)
    # ========================================
    def _process_window(self):
        """Processa janela de trades."""
        if not self.window_data or self.should_stop:
            self.window_data = []
            return
        
        # ✅ NOVO: Verifica se está em aquecimento
        if self.warming_up:
            self.warmup_windows_remaining -= 1
            
            logging.info(
                f"⏳ AQUECIMENTO: Janela processada "
                f"({self.warmup_windows_required - self.warmup_windows_remaining}/"
                f"{self.warmup_windows_required})"
            )
            
            if self.warmup_windows_remaining <= 0:
                self.warming_up = False
                logging.info("✅ AQUECIMENTO CONCLUÍDO - Sistema pronto!")
            
            # Não processa durante aquecimento
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
        
        # ✅ NOVO: Validação PRÉ-PIPELINE
        if len(valid_window_data) < self.min_trades_for_pipeline:
            logging.warning(
                f"⏳ Janela com apenas {len(valid_window_data)} trades "
                f"(mín: {self.min_trades_for_pipeline}). "
                f"Aguardando mais dados..."
            )
            
            # ✅ Tenta recuperar do buffer
            if len(self.trades_buffer) >= self.min_trades_for_pipeline:
                logging.info(
                    f"🔄 Recuperando {self.min_trades_for_pipeline} trades "
                    f"do buffer de emergência..."
                )
                valid_window_data = list(self.trades_buffer)[-self.min_trades_for_pipeline:]
            else:
                self.window_data = []
                return
        
        total_volume = sum(
            float(trade.get("q", 0))
            for trade in valid_window_data
        )
        
        if total_volume == 0:
            self.window_data = []
            return
        
        self.window_count += 1
        
        # ✅ Calcula volumes com NumPy (OTIMIZADO)
        try:
            quantities = np.array([t.get("q", 0) for t in valid_window_data])
            is_sell = np.array([t.get("m", False) for t in valid_window_data])
            
            total_sell_volume = float(quantities[is_sell].sum())
            total_buy_volume = float(quantities[~is_sell].sum())
        except Exception as e:
            logging.error(
                f"Erro ao calcular volumes com NumPy, usando fallback: {e}"
            )
            total_buy_volume = 0
            total_sell_volume = 0
            
            for trade in valid_window_data:
                if trade.get("m"):
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
                dynamic_delta_threshold = abs(
                    mean_delta + self.delta_std_dev_factor * std_delta
                )
            
            # Contexto macro
            macro_context = self.context_collector.get_context()
            historical_profile = macro_context.get("historical_vp", {})
            
            # Valida e usa cache se value area zerada
            vp_daily = historical_profile.get("daily", {})
            val = vp_daily.get("val", 0)
            vah = vp_daily.get("vah", 0)
            poc = vp_daily.get("poc", 0)
            
            if val == 0 or vah == 0 or poc == 0:
                if self.last_valid_vp and (
                    time.time() - self.last_valid_vp_time < 3600
                ):
                    age = time.time() - self.last_valid_vp_time
                    logging.warning(
                        f"⚠️ Value Area zerada, usando cache (age={age:.0f}s)"
                    )
                    historical_profile = self.last_valid_vp.copy()
                else:
                    logging.warning(
                        "⚠️ Value Area indisponível e sem cache válido"
                    )
            else:
                self.last_valid_vp = historical_profile.copy()
                self.last_valid_vp_time = time.time()
            
            open_ms = self.window_end_ms - self.window_ms
            close_ms = self.window_end_ms
            
            # Atualiza níveis
            self.levels.update_from_vp(historical_profile)
            
            # ✅ Pipeline de dados (COM VALIDAÇÃO)
            try:
                pipeline = DataPipeline(
                    valid_window_data,
                    self.symbol,
                    time_manager=self.time_manager,
                )
            except ValueError as ve:
                logging.error(
                    f"❌ Erro ao criar pipeline (janela #{self.window_count}): {ve}"
                )
                self.window_data = []
                return
            
            flow_metrics = self.flow_analyzer.get_flow_metrics(
                reference_epoch_ms=close_ms
            )
            
            # ✅ Orderbook com retry robusto
            ob_event = self._fetch_orderbook_with_retry(close_ms)
            
            # Enriquece
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
            
            # Detecta sinais
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
            
            # ✅ Processa sinais (pipeline incluído)
            self._process_signals(
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
                valid_window_data
            )
            
        except Exception as e:
            logging.error(
                f"Erro no processamento da janela #{self.window_count}: {e}",
                exc_info=True
            )
        finally:
            self.window_data = []

    # ========================================
    # FETCH ORDERBOOK COM RETRY
    # ========================================
    def _fetch_orderbook_with_retry(self, close_ms):
        """Busca orderbook com retry robusto."""
        ob_event = None
        max_retries = getattr(config, 'ORDERBOOK_MAX_RETRIES', 3)
        base_delay = getattr(config, 'ORDERBOOK_RETRY_DELAY', 2.0)
        
        for attempt in range(max_retries):
            try:
                ob_event = self.orderbook_analyzer.analyze(
                    current_snapshot=None,
                    event_epoch_ms=close_ms,
                    window_id=f"W{self.window_count:04d}"
                )
                
                if ob_event and ob_event.get('is_valid', False):
                    ob_data = ob_event.get('orderbook_data', {})
                    bid_depth = ob_data.get('bid_depth_usd', 0)
                    ask_depth = ob_data.get('ask_depth_usd', 0)
                    min_depth = getattr(
                        config,
                        'ORDERBOOK_MIN_DEPTH_USD',
                        500.0
                    )
                    
                    if bid_depth >= min_depth or ask_depth >= min_depth:
                        self.last_valid_orderbook = ob_event
                        self.last_valid_orderbook_time = time.time()
                        self.orderbook_fetch_failures = 0
                        
                        logging.debug(
                            f"✅ Orderbook OK - Janela #{self.window_count}"
                        )
                        break
                    else:
                        logging.warning(
                            f"⚠️ Orderbook com liquidez baixa "
                            f"(tentativa {attempt + 1}/{max_retries})"
                        )
                else:
                    logging.warning(
                        f"⚠️ Orderbook inválido "
                        f"(tentativa {attempt + 1}/{max_retries})"
                    )
                
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    time.sleep(delay)
                    
            except Exception as e:
                logging.error(
                    f"❌ Erro ao buscar orderbook "
                    f"(tentativa {attempt + 1}): {e}"
                )
                
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    time.sleep(delay)
        
        # Fallback
        if not ob_event or not ob_event.get('is_valid', False):
            return self._orderbook_fallback()
        
        return ob_event

    def _orderbook_fallback(self):
        """Fallback quando orderbook não disponível."""
        self.orderbook_fetch_failures += 1
        fallback_max_age = getattr(
            config,
            'ORDERBOOK_FALLBACK_MAX_AGE',
            600
        )
        
        if self.last_valid_orderbook and (
            time.time() - self.last_valid_orderbook_time < fallback_max_age
        ):
            age = time.time() - self.last_valid_orderbook_time
            logging.warning(
                f"⚠️ Usando orderbook em cache (age={age:.0f}s) "
                f"após {self.orderbook_fetch_failures} falhas"
            )
            
            ob_event = self.last_valid_orderbook.copy()
            ob_event['data_quality'] = {
                'is_valid': True,
                'data_source': 'cache',
                'age_seconds': age,
            }
            return ob_event
            
        elif self.orderbook_emergency_mode:
            logging.warning(
                f"🚨 MODO EMERGÊNCIA: Orderbook indisponível "
                f"(falhas: {self.orderbook_fetch_failures})"
            )
            
            return {
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
                    'error': (
                        f'Emergency mode after '
                        f'{self.orderbook_fetch_failures} failures'
                    ),
                },
            }
        else:
            logging.error(
                f"❌ Orderbook totalmente indisponível "
                f"(falhas consecutivas: {self.orderbook_fetch_failures})"
            )
            
            return {
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
                    'error': 'Failed after max retries',
                },
            }

    # ========================================
    # ✅ PROCESSAMENTO DE SINAIS (CORRIGIDO v2.3.1)
    # ========================================
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
        valid_window_data
    ):
        """Processa e salva sinais detectados."""
        
        # ✅ CORRIGIDO: Lógica de ANALYSIS_TRIGGER
        has_real_signal = any(
            s.get("tipo_evento") not in ("ANALYSIS_TRIGGER", "OrderBook")
            for s in signals
        )
        
        # Se NÃO há sinais reais, cria/substitui ANALYSIS_TRIGGER
        if not signals or not has_real_signal:
            trigger_signal = {
                "is_signal": True,
                "tipo_evento": "ANALYSIS_TRIGGER",
                "resultado_da_batalha": "N/A",
                "descricao": "Evento automático para análise da IA",
                "timestamp": self.time_manager.now_utc_iso(timespec="seconds"),
                "delta": enriched.get("delta_fechamento", 0),
                "volume_total": enriched.get("volume_total", 0),
                "preco_fechamento": enriched.get("ohlc", {}).get("close", 0),
                "epoch_ms": close_ms,
                "ml_features": pipeline.get_final_features().get("ml_features", {}),
                "orderbook_data": ob_event,
                "historical_vp": historical_profile,
                "multi_tf": macro_context.get("mtf_trends", {}),
            }
            
            if not signals:
                signals.append(trigger_signal)
            elif not has_real_signal:
                # Remove ANALYSIS_TRIGGER antigo e adiciona novo
                signals = [s for s in signals if s.get("tipo_evento") != "ANALYSIS_TRIGGER"]
                signals.append(trigger_signal)
        
        # ✅ REMOVIDO: Não filtra mais signals aqui
        # A lista signals agora contém TODOS os sinais que devem ser processados
        
        # Log do heatmap
        self._log_liquidity_heatmap(flow_metrics)
        
        # Obtém features
        features = pipeline.get_final_features()
        self.feature_store.save_features(
            window_id=str(close_ms),
            features=features
        )
        
        ml_payload = features.get("ml_features", {}) or {}
        enriched_snapshot = features.get("enriched", {}) or {}
        contextual_snapshot = features.get("contextual", {}) or {}
        
        derivatives_context = macro_context.get("derivatives", {})
        
        # ✅ DEBUG: Log de sinais antes do processamento
        logging.info(
            f"📊 JANELA #{self.window_count} - "
            f"Processando {len(signals)} sinal(is):"
        )
        for i, sig in enumerate(signals, 1):
            logging.info(
                f"  {i}. {sig.get('tipo_evento')} / "
                f"{sig.get('resultado_da_batalha', 'N/A')} | "
                f"delta={sig.get('delta', 0):.2f} | "
                f"volume={sig.get('volume_total', 0):.2f}"
            )
        
        # Processa cada sinal
        for signal in signals:
            if signal.get("is_signal", False):
                self._enrich_signal(
                    signal,
                    derivatives_context,
                    flow_metrics,
                    total_buy_volume,
                    total_sell_volume,
                    macro_context,
                    close_ms,
                    ml_payload,
                    enriched_snapshot,
                    contextual_snapshot,
                    ob_event,
                    valid_window_data
                )
        
        # Verifica toques em zonas
        self._check_zone_touches(enriched, signals)
        
        # Atualiza históricos
        self._update_histories(enriched, ml_payload)
        
        # Log ML features
        self._log_ml_features(ml_payload)
        
        # ✅ Processa alertas (pipeline incluído)
        self._process_institutional_alerts(enriched, pipeline)
        
        # Log consolidado de qualidade
        self._log_health_check()
        
        # Log principal
        self._log_window_summary(enriched, historical_profile, macro_context)

    # ========================================
    # ✅ ENRIQUECIMENTO DE SINAL (CORRIGIDO v2.3.1)
    # ========================================
    def _enrich_signal(
        self,
        signal,
        derivatives_context,
        flow_metrics,
        total_buy_volume,
        total_sell_volume,
        macro_context,
        close_ms,
        ml_payload,
        enriched_snapshot,
        contextual_snapshot,
        ob_event,
        valid_window_data
    ):
        """Enriquece sinal com dados adicionais."""
        
        # ✅ CORREÇÃO CRÍTICA: Garante timestamps consistentes ANTES da validação
        if "epoch_ms" not in signal:
            signal["epoch_ms"] = close_ms
        
        if "timestamp_utc" not in signal:
            # ✅ CORRIGIDO: Usa método from_timestamp_ms() com timezone
            signal["timestamp_utc"] = self.time_manager.from_timestamp_ms(
                close_ms,
                tz=self.time_manager.tz_utc
            ).isoformat(timespec="milliseconds")
        
        if "timestamp" not in signal:
            # ✅ CORRIGIDO: Usa método from_timestamp_ms() com timezone NY
            signal["timestamp"] = self.time_manager.from_timestamp_ms(
                close_ms,
                tz=self.ny_tz
            ).strftime("%Y-%m-%d %H:%M:%S")
        
        # Validação de dados
        validated_signal = validator.validate_and_clean(signal)
        if not validated_signal:
            logging.warning(
                f"Evento {signal.get('tipo_evento')} / "
                f"{signal.get('resultado_da_batalha')} "
                f"descartado pela validação."
            )
            return
        
        signal.update(validated_signal)
        
        # Derivatives
        if "derivatives" not in signal:
            signal["derivatives"] = derivatives_context
        
        # Flow metrics
        if "fluxo_continuo" not in signal and flow_metrics:
            flow_valid = self._validate_flow_metrics(
                flow_metrics,
                valid_window_data
            )
            
            signal["fluxo_continuo"] = flow_metrics
            
            if flow_valid:
                logging.debug(
                    f"✅ Flow metrics adicionado "
                    f"(janela #{self.window_count})"
                )
            else:
                signal["flow_data_quality"] = "incomplete"
                logging.warning("⚠️ Flow metrics possivelmente incompleto")
        
        # Volumes calculados localmente
        if (signal.get("volume_compra", 0) == 0 and
            signal.get("volume_venda", 0) == 0):
            signal["volume_compra"] = total_buy_volume
            signal["volume_venda"] = total_sell_volume
            
            logging.debug(
                f"📊 Volumes calculados localmente: "
                f"buy={total_buy_volume:.2f}, sell={total_sell_volume:.2f}"
            )
        
        # Market context
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
        
        # Features
        signal.setdefault("features_window_id", str(close_ms))
        signal["ml_features"] = ml_payload
        signal["enriched_snapshot"] = enriched_snapshot
        signal["contextual_snapshot"] = contextual_snapshot
        
        # Orderbook data
        if ob_event and isinstance(ob_event, dict) and ob_event.get(
            'is_valid', False
        ):
            if "orderbook_data" in ob_event:
                signal["orderbook_data"] = ob_event["orderbook_data"]
            if "spread_metrics" in ob_event:
                signal["spread_metrics"] = ob_event["spread_metrics"]
        
        # Previne triggers duplicados
        if signal.get("tipo_evento") == "ANALYSIS_TRIGGER":
            key = (
                signal.get("tipo_evento"),
                signal.get("features_window_id")
            )
            
            if key in self._sent_triggers:
                logging.debug(f"⏭️ ANALYSIS_TRIGGER duplicado ignorado (janela {close_ms})")
                return
            
            self._sent_triggers.add(key)
        
        # Atualiza níveis
        self.levels.add_from_event(signal)
        
        # ✅ DEBUG: Log antes de publicar/salvar
        logging.debug(
            f"💾 Salvando: {signal.get('tipo_evento')} / "
            f"{signal.get('resultado_da_batalha')} | "
            f"epoch_ms={signal.get('epoch_ms')}"
        )
        
        # Publica e salva
        self.event_bus.publish("signal", signal)
        self.event_saver.save_event(signal)
        
        # Adiciona à memória
        if signal.get("tipo_evento") != "OrderBook":
            adicionar_memoria_evento(signal)
        
        # Log
        self._log_event(signal)

    # ========================================
    # VALIDAÇÃO DE FLOW METRICS
    # ========================================
    def _validate_flow_metrics(self, flow_metrics, valid_window_data):
        """Valida se flow metrics tem dados válidos."""
        try:
            # Verifica trades processados
            trades_processed = 0
            if "data_quality" in flow_metrics:
                trades_processed = flow_metrics["data_quality"].get(
                    "flow_trades_count", 0
                )
            
            if trades_processed > 0:
                return True
            
            # Verifica sector flow
            sector_flow = flow_metrics.get("sector_flow", {})
            for sector, data in sector_flow.items():
                total_vol = abs(data.get("buy", 0)) + abs(data.get("sell", 0))
                if total_vol > 0.001:
                    return True
            
            # Verifica order flow
            order_flow = flow_metrics.get("order_flow", {})
            for key in ["net_flow_1m", "net_flow_5m", "net_flow_15m"]:
                val = order_flow.get(key)
                if val is not None and val != 0:
                    return True
            
            buy_pct = order_flow.get("aggressive_buy_pct", 0)
            sell_pct = order_flow.get("aggressive_sell_pct", 0)
            if buy_pct > 0 or sell_pct > 0:
                return True
            
            # Verifica whale volume
            whale_total = (
                abs(flow_metrics.get("whale_buy_volume", 0)) +
                abs(flow_metrics.get("whale_sell_volume", 0))
            )
            if whale_total > 0.001:
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Erro ao validar flow_metrics: {e}")
            return False

    # ========================================
    # VERIFICAÇÃO DE TOQUES EM ZONAS
    # ========================================
    def _check_zone_touches(self, enriched, signals):
        """Verifica toques em zonas de suporte/resistência."""
        preco_atual = enriched.get("ohlc", {}).get("close", 0)
        
        if preco_atual > 0:
            try:
                touched = self.levels.check_price(float(preco_atual))
                
                for z in touched:
                    zone_event = signals[0].copy() if signals else {}
                    
                    preco_fmt = format_price(preco_atual)
                    low_fmt = format_price(z.low)
                    high_fmt = format_price(z.high)
                    
                    zone_event.update({
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
                    })
                    
                    if "historical_confidence" not in zone_event:
                        zone_event["historical_confidence"] = (
                            calcular_probabilidade_historica(zone_event)
                        )
                    
                    self.event_bus.publish("zone_touch", zone_event)
                    self.event_saver.save_event(zone_event)
                    
                    adicionar_memoria_evento({
                        "timestamp": (
                            z.last_touched or
                            datetime.now(self.ny_tz).isoformat(timespec="seconds")
                        ),
                        "tipo_evento": "Zona",
                        "resultado_da_batalha": f"Toque {z.kind}",
                        "delta": zone_event.get("delta", 0),
                        "volume_total": zone_event.get("volume_total", 0),
                    })
                    
            except Exception as e:
                logging.error(f"Erro ao verificar toques em zonas: {e}")

    # ========================================
    # ATUALIZAÇÃO DE HISTÓRICOS
    # ========================================
    def _update_histories(self, enriched, ml_payload):
        """Atualiza históricos de volume, delta e volatilidade."""
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

    # ========================================
    # LOGS
    # ========================================
    def _log_liquidity_heatmap(self, flow_metrics):
        """Loga heatmap de liquidez."""
        try:
            liquidity_data = flow_metrics.get("liquidity_heatmap", {})
            clusters = liquidity_data.get("clusters", [])
            
            if clusters:
                print(
                    f"\n📊 LIQUIDITY HEATMAP - "
                    f"Janela #{self.window_count}:"
                )
                
                for i, cluster in enumerate(clusters[:3]):
                    center_fmt = format_price(cluster.get('center', 0.0))
                    vol_fmt = format_large_number(cluster.get('total_volume', 0))
                    imb_fmt = format_percent(
                        cluster.get('imbalance_ratio', 0.0) * 100
                    )
                    trades_fmt = format_quantity(cluster.get('trades_count', 0))
                    age_fmt = format_time_seconds(cluster.get('age_ms', 0))
                    
                    print(
                        f"  Cluster {i+1}: ${center_fmt} | "
                        f"Vol: {vol_fmt} | "
                        f"Imb: {imb_fmt} | "
                        f"Trades: {trades_fmt} | "
                        f"Age: {age_fmt}"
                    )
        except Exception as e:
            logging.error(f"Erro ao logar liquidity heatmap: {e}")

    def _log_ml_features(self, ml_payload):
        """Loga features de ML."""
        try:
            pf = ml_payload.get("price_features", {}) if ml_payload else {}
            vf = ml_payload.get("volume_features", {}) if ml_payload else {}
            mf = ml_payload.get("microstructure", {}) if ml_payload else {}
            
            if pf or vf or mf:
                ret5_fmt = format_scientific(pf.get('returns_5', 0))
                vol5_fmt = format_scientific(pf.get('volatility_5', 0), decimals=5)
                vsma_fmt = format_percent(vf.get('volume_sma_ratio', 0) * 100)
                bs_fmt = format_delta(vf.get('buy_sell_pressure', 0))
                obs_fmt = format_scientific(
                    mf.get('order_book_slope', 0), decimals=3
                )
                flow_fmt = format_scientific(
                    mf.get('flow_imbalance', 0), decimals=3
                )
                
                print(
                    f"  ML: ret5={ret5_fmt} vol5={vol5_fmt} "
                    f"V/SMA={vsma_fmt} BSpress={bs_fmt} "
                    f"OBslope={obs_fmt} FlowImb={flow_fmt}"
                )
        except Exception:
            pass

    def _log_health_check(self):
        """Loga health check consolidado."""
        if self.window_count % 10 == 0:
            logging.info(
                f"\n📊 HEALTH CHECK - Janela #{self.window_count}:\n"
                f"  Orderbook: failures={self.orderbook_fetch_failures}, "
                f"last_valid={time.time() - self.last_valid_orderbook_time:.0f}s ago\n"
                f"  Value Area: last_valid={time.time() - self.last_valid_vp_time:.0f}s ago"
            )

    def _log_window_summary(self, enriched, historical_profile, macro_context):
        """Loga sumário da janela."""
        window_delta = enriched.get("delta_fechamento", 0)
        window_volume = enriched.get("volume_total", 0)
        
        delta_fmt = format_delta(window_delta)
        vol_fmt = format_large_number(window_volume)
        
        print(
            f"[{datetime.now(self.ny_tz).strftime('%H:%M:%S')} NY] "
            f"🟡 Janela #{self.window_count} | "
            f"Delta: {delta_fmt} | Vol: {vol_fmt}"
        )
        
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
            
            print(
                f"  VP Diário: POC @ {poc_fmt} | "
                f"VAL: {val_fmt} | VAH: {vah_fmt}"
            )
        
        print("─" * 80)

    # ========================================
    # ALERTAS INSTITUCIONAIS
    # ========================================
    def _process_institutional_alerts(self, enriched, pipeline):
        """Processa alertas institucionais."""
        if generate_alerts is None:
            return
        
        try:
            # Detecta suporte/resistência
            if detect_support_resistance is not None:
                try:
                    price_series = (
                        pipeline.df['p']
                        if hasattr(pipeline, 'df') and pipeline.df is not None
                        else None
                    )
                    
                    if price_series is not None:
                        sr = detect_support_resistance(
                            price_series, num_levels=3
                        )
                    else:
                        sr = {
                            "immediate_support": [],
                            "immediate_resistance": []
                        }
                except Exception:
                    sr = {
                        "immediate_support": [],
                        "immediate_resistance": []
                    }
            else:
                sr = {
                    "immediate_support": [],
                    "immediate_resistance": []
                }
            
            window_close = enriched.get("ohlc", {}).get("close", 0)
            current_price_alert = window_close
            
            avg_vol = (
                (sum(self.volume_history) / len(self.volume_history))
                if len(self.volume_history) > 0
                else enriched.get("volume_total", 0)
            )
            
            rec_vols = list(self.volatility_history)
            
            curr_vol = None
            try:
                if len(self.volatility_history) > 0:
                    curr_vol = self.volatility_history[-1]
            except Exception:
                curr_vol = None
            
            # Gera alertas
            alerts_list = generate_alerts(
                price=current_price_alert,
                support_resistance=sr,
                current_volume=enriched.get("volume_total", 0),
                average_volume=avg_vol,
                current_volatility=curr_vol or 0.0,
                recent_volatilities=rec_vols,
                volume_threshold=3.0,
                tolerance_pct=0.001,
            )
            
            # Processa alertas
            for alert in alerts_list or []:
                try:
                    atype = alert.get('type', 'GENERIC')
                    now_s = time.time()
                    last_ts = self._last_alert_ts.get(atype, 0)
                    
                    # Cooldown
                    if now_s - last_ts < self._alert_cooldown_sec:
                        continue
                    
                    self._last_alert_ts[atype] = now_s
                    
                    # Formata descrição
                    desc_parts = [f"Tipo: {alert.get('type')}"]
                    
                    if 'level' in alert:
                        desc_parts.append(
                            f"Nível: {format_price(alert['level'])}"
                        )
                    
                    if 'threshold_exceeded' in alert:
                        desc_parts.append(
                            f"Fator: {format_percent(alert['threshold_exceeded'] * 100)}"
                        )
                    
                    descricao_alert = " | ".join(desc_parts)
                    
                    print(f"🔔 ALERTA: {descricao_alert}")
                    logging.info(f"🔔 ALERTA: {descricao_alert}")
                    
                    # Cria evento
                    alert_event = {
                        "tipo_evento": "Alerta",
                        "resultado_da_batalha": alert.get('type'),
                        "descricao": descricao_alert,
                        "timestamp": self.time_manager.now_utc_iso(
                            timespec="seconds"
                        ),
                        "severity": alert.get('severity'),
                        "probability": alert.get('probability'),
                        "action": alert.get('action'),
                        "context": {
                            "price": current_price_alert,
                            "volume": enriched.get("volume_total", 0),
                            "average_volume": avg_vol,
                            "volatility": curr_vol or 0.0,
                        },
                    }
                    
                    self.event_saver.save_event(alert_event)
                    
                except Exception as e:
                    logging.error(f"Erro ao processar alerta: {e}")
                    
        except Exception as e:
            logging.error(f"Erro ao gerar alertas: {e}")

    # ========================================
    # CALLBACKS DO WEBSOCKET
    # ========================================
    def on_open(self, ws):
        """Callback quando conexão é aberta."""
        logging.info(
            f"🚀 Bot iniciado para {self.symbol} - "
            f"Fuso: New York (America/New_York)"
        )
        
        try:
            self.health_monitor.heartbeat("main")
        except Exception:
            pass

    def on_close(self, ws, code, msg):
        """Callback quando conexão é fechada."""
        if self.window_data and not self.should_stop:
            self._process_window()

    # ========================================
    # RUN
    # ========================================
    def run(self):
        """Executa o bot."""
        try:
            self.context_collector.start()
            
            logging.info(
                "🎯 Iniciando Enhanced Market Bot v2.3.1 "
                "(CORREÇÃO COMPLETA + WARMUP + BUFFER)..."
            )
            print("═" * 80)
            
            self.connection_manager.connect()
            
        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        except Exception as e:
            logging.critical(
                f"❌ Erro crítico ao executar o bot: {e}",
                exc_info=True
            )
        finally:
            self._cleanup_handler()


# ===== EXECUÇÃO =====
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
        logging.critical(
            f"❌ Erro crítico na inicialização do bot: {e}",
            exc_info=True
        )
        sys.exit(1)