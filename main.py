#MAIN.PY
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
import sys
import atexit

# Importa o arquivo de configurações central
import config

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
from report_generator import ReportGenerator
from levels_registry import LevelRegistry

# 🔹 NOVOS MÓDULOS
from time_manager import TimeManager
from health_monitor import HealthMonitor
from event_bus import EventBus
from data_pipeline import DataPipeline
from feature_store import FeatureStore

# Alert engine and support/resistance for institutional alerts
try:
    from alert_engine import generate_alerts
except Exception:
    generate_alerts = None
try:
    from support_resistance import detect_support_resistance
except Exception:
    detect_support_resistance = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ===============================
# GESTOR DE CONEXÃO WEBSOCKET (COM MITIGAÇÃO DE FALHAS)
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
        self.external_heartbeat_cb = None  # <- novo

    def set_callbacks(self, on_message=None, on_open=None, on_close=None, on_error=None):
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error

    def set_heartbeat_cb(self, cb):
        """Permite injetar um heartbeat externo (ex.: HealthMonitor.heartbeat('main'))."""
        self.external_heartbeat_cb = cb

    def _test_connection(self):
        """
        Teste rápido de conectividade:
        - Resolve DNS
        - Abre conexão TCP (sem handshake TLS)
        Evita travar em redes com interceptação TLS/latência alta.
        """
        try:
            parsed = urlparse(self.stream_url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "wss" else 80)

            # DNS resolve
            socket.getaddrinfo(host, port)

            # TCP connect rápido
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
        ping_interval = getattr(config, "WS_PING_INTERVAL", 25)
        ping_timeout = getattr(config, "WS_PING_TIMEOUT", 10)

        while self.reconnect_count < self.max_reconnect_attempts and not self.should_stop:
            try:
                # Mantém heartbeat enquanto tenta
                if self.external_heartbeat_cb:
                    try:
                        self.external_heartbeat_cb()
                    except Exception:
                        pass

                # Teste leve (TCP) — não trava no TLS
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
                # Deixa o run_forever fazer o handshake WebSocket/TLS
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
                    # Mantém heartbeat durante backoff
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
# ANALISADOR DE TRADE FLOW (compatibilidade)
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
            historical_profile=historical_profile,
        )

        exhaustion_event = create_exhaustion_event(
            window_data,
            symbol,
            history_volumes=history_volumes,
            volume_factor=self.vol_factor_exh,
            tz_output=self.tz_output,
            historical_profile=historical_profile,
        )
        return absorption_event, exhaustion_event


# ===============================
# FUNÇÃO AUXILIAR: GET CURRENT PRICE VIA REST (COM MITIGAÇÃO)
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
    def __init__(self, stream_url, symbol, window_size_minutes, vol_factor_exh, history_size, delta_std_dev_factor, context_sma_period, liquidity_flow_alert_percentage, wall_std_dev_factor):
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.window_ms = window_size_minutes * 60 * 1000
        self.ny_tz = NY_TZ
        self.should_stop = False
        self.is_cleaning_up = False

        # 🔹 TimeManager único para todo o app
        self.time_manager = TimeManager()

        # 🔹 INICIALIZA MÓDULOS
        self.health_monitor = HealthMonitor()
        self.event_bus = EventBus()
        self.feature_store = FeatureStore()

        # 🔹 HEARTBEAT
        self.health_monitor.heartbeat("main")

        # Módulos existentes (com TimeManager injetado)
        self.trade_flow_analyzer = TradeFlowAnalyzer(vol_factor_exh, tz_output=self.ny_tz)
        self.orderbook_analyzer = OrderBookAnalyzer(
            symbol=self.symbol,
            liquidity_flow_alert_percentage=liquidity_flow_alert_percentage,
            wall_std_dev_factor=wall_std_dev_factor,
            time_manager=self.time_manager,
        )
        self.event_saver = EventSaver(sound_alert=True)
        self.context_collector = ContextCollector(symbol=self.symbol)
        self.flow_analyzer = FlowAnalyzer(time_manager=self.time_manager)

        self.ai_analyzer = None
        self.ai_initialization_attempted = False
        self.ai_test_passed = False
        self.ai_thread_pool = []
        self.max_ai_threads = 3
        self.ai_semaphore = threading.Semaphore(3)
        self._initialize_ai_async()

        # 🔹 EVENT BUS
        self.event_bus.subscribe("signal", self._handle_signal_event)
        self.event_bus.subscribe("zone_touch", self._handle_zone_touch_event)

        self.connection_manager = RobustConnectionManager(stream_url, symbol, max_reconnect_attempts=15)
        self.connection_manager.set_callbacks(on_message=self.on_message, on_open=self.on_open, on_close=self.on_close)
        # Heartbeat durante reconexões
        self.connection_manager.set_heartbeat_cb(lambda: self.health_monitor.heartbeat("main"))

        self.window_end_ms = None
        self.window_data = []
        self.window_count = 0
        self.history_size = history_size
        self.volume_history = deque(maxlen=history_size)
        self.delta_history = deque(maxlen=history_size)
        self.close_price_history = deque(maxlen=context_sma_period)
        self.delta_std_dev_factor = delta_std_dev_factor

        # Histórico de volatilidade (usado em alertas)
        self.volatility_history = deque(maxlen=history_size)

        self.reporter = ReportGenerator(output_dir="./reports", mode="csv")
        self.levels = LevelRegistry(self.symbol)

        # 🔹 CONTADORES DE CAMPOS AUSENTES
        self._missing_field_counts = {"q": 0, "m": 0, "p": 0, "T": 0}
        # log_step define a cada quantos eventos um aviso de campos ausentes é logado.
        # None ou 0 desliga o logging de amostragem. Valor menor = mais frequente.
        try:
            import config as cfg  # leitura dinâmica de config
            self._missing_field_log_step = getattr(cfg, "MISSING_FIELD_LOG_STEP", None)
        except Exception:
            self._missing_field_log_step = None

        # 🔹 PARA INFERÊNCIA DO 'm'
        self._last_price = None

        # 🔹 Cooldown simples para alertas
        self._last_alert_ts = {}
        try:
            self._alert_cooldown_sec = getattr(config, "ALERT_COOLDOWN_SEC", 30)
        except Exception:
            self._alert_cooldown_sec = 30

        self._register_cleanup_handlers()

    def _initialize_ai_async(self):
        def ai_init_worker():
            try:
                if self.ai_initialization_attempted:
                    return
                self.ai_initialization_attempted = True

                print("\n" + "=" * 30 + " INICIALIZANDO IA " + "=" * 30)
                logging.info("🧠 Tentando inicializar AI Analyzer Qwen...")

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
                }
                analysis = self.ai_analyzer.analyze_event(test_event)

                # Define limite mínimo para considerar o teste da IA como bem-sucedido.
                # Lê de config se disponível, com fallback para 10 caracteres.
                try:
                    import config as cfg
                    min_chars = getattr(cfg, "AI_TEST_MIN_CHARS", 10)
                except Exception:
                    min_chars = 10
                if analysis and len(analysis.strip()) >= min_chars:
                    self.ai_test_passed = True
                    logging.info("✅ Teste da IA bem-sucedido!")
                    print("\n" + "═" * 25 + " RESULTADO DO TESTE DA IA " + "═" * 25)
                    print(analysis)
                    print("═" * 75 + "\n")
                else:
                    # mesmo se o teste retornar vazio, continuamos com modo simplificado para não bloquear análises
                    self.ai_test_passed = True
                    logging.warning("⚠️ Teste da IA retornou resultado inesperado ou vazio. Prosseguindo em modo de fallback.")
                    print(f"Resultado recebido: {analysis}")
                    print("=" * 75 + "\n")

            except Exception as e:
                self.ai_analyzer = None
                self.ai_test_passed = False
                print("=" * 30 + " ERRO NA IA " + "=" * 30)
                logging.error(f"❌ Falha crítica ao inicializar a IA: {e}", exc_info=True)
                print("=" * 75 + "\n")

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
            raw = json.loads(message)

            # Suporta combined stream ({"stream": "...", "data": {...}}) ou payload direto
            trade = raw.get("data", raw)

            # 🔹 Aceita chaves alternativas / maiúsculas
            p = trade.get("p") or trade.get("P") or trade.get("price")
            q = trade.get("q") or trade.get("Q") or trade.get("quantity")
            T = trade.get("T") or trade.get("E") or trade.get("tradeTime")
            m = trade.get("m")  # True => agressor vendedor; False => agressor comprador

            # 🔹 Se vier KLINE, sintetiza p/q/T a partir de k.c/k.v/k.T
            if (p is None or q is None or T is None) and isinstance(trade.get("k"), dict):
                k = trade["k"]
                p = p if p is not None else k.get("c")
                q = q if q is not None else k.get("v")
                T = T if T is not None else k.get("T") or raw.get("E")

            # 🔹 Verifica campos obrigatórios
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
                # Só loga se log_step estiver definido e total_missing é múltiplo do step
                total_missing = sum(self._missing_field_counts[k] for k in ("p", "q", "T"))
                if self._missing_field_log_step:
                    try:
                        step = int(self._missing_field_log_step)
                    except Exception:
                        step = None
                    if step and step > 0 and total_missing % step == 0:
                        # Usa nível DEBUG para reduzir ruído
                        logging.debug(
                            "Campos ausentes (amostra): p=%d q=%d T=%d",
                            self._missing_field_counts["p"],
                            self._missing_field_counts["q"],
                            self._missing_field_counts["T"],
                        )
                # Sempre ignora trade incompleto
                return

            # 🔹 Coerção de tipos
            try:
                p = float(p)
                q = float(q)
                T = int(T)
            except (TypeError, ValueError):
                logging.error("Trade inválido (tipos): %s", trade)
                return

            # 🔹 Inferência de 'm' via tick-direction quando ausente
            if m is None:
                last_price = self._last_price
                if last_price is not None:
                    m = p <= last_price  # preço subiu → m=False (buyer taker)
                else:
                    m = False
            self._last_price = p

            norm = {"p": p, "q": q, "T": T, "m": bool(m)}

            # 🔹 Heartbeat do módulo principal a cada mensagem válida
            try:
                self.health_monitor.heartbeat("main")
            except Exception as hb_err:
                logging.debug("Falha ao enviar heartbeat: %s", hb_err)

            # 🔹 Fluxo normal
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

    # 🔹 HANDLERS PARA EVENT BUS
    def _handle_signal_event(self, event_data):
        """Processa eventos de sinal (Absorção, Exaustão, OrderBook)"""
        if not self.ai_analyzer or not self.ai_test_passed:
            return
        self._run_ai_analysis_threaded(event_data.copy())

    def _handle_zone_touch_event(self, event_data):
        """Processa eventos de toque em zona"""
        if not self.ai_analyzer or not self.ai_test_passed:
            return
        self._run_ai_analysis_threaded(event_data.copy())

    def _run_ai_analysis_threaded(self, event_data):
        if not self.ai_analyzer or not self.ai_test_passed or self.should_stop:
            if self.ai_analyzer and not self.ai_test_passed:
                logging.warning("⚠️ Análise da IA ignorada: sistema não passou no teste inicial.")
            return

        def ai_worker():
            try:
                with self.ai_semaphore:
                    logging.info("🧠 IA iniciando análise para evento: %s", event_data.get("resultado_da_batalha", "N/A"))
                    analysis_result = self.ai_analyzer.analyze_event(event_data)
                    if analysis_result and not self.should_stop:
                        print("\n" + "═" * 25 + " ANÁLISE PROFISSIONAL DA IA " + "═" * 25)
                        print(analysis_result)
                        print("═" * 75 + "\n")
                        self.reporter.save_report(event_data, analysis_result)
                        self.health_monitor.heartbeat("ai")
            except Exception as e:
                logging.error(f"❌ Erro na thread de análise da IA: {e}")

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
        print(f"   Símbolo: {self.symbol} | Janela #{self.window_count}")
        print(f"   📝 {descricao}")
        if conf:
            print(
                f"   📊 Probabilidades -> Long={conf.get('long_prob')} | Short={conf.get('short_prob')} | Neutro={conf.get('neutral_prob')}"
            )

        ultimos = [e for e in obter_memoria_eventos(n=4) if e.get("tipo_evento") != "OrderBook"]
        if ultimos:
            print("   🕒 Últimos sinais:")
            for e in ultimos:
                print(
                    f"      - {e.get('timestamp', 'N/A')} | {e.get('tipo_evento', 'N/A')} {e.get('resultado_da_batalha', 'N/A')} "
                    f"(Δ={e.get('delta', 0):,.2f}, Vol={e.get('volume_total', 0):,.2f})"
                )

    def _process_window(self):
        if not self.window_data or self.should_stop:
            self.window_data = []
            return

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
        try:
            self.health_monitor.heartbeat("main")

            dynamic_delta_threshold = 0
            if len(self.delta_history) > 10:
                mean_delta = np.mean(self.delta_history)
                std_delta = np.std(self.delta_history)
                dynamic_delta_threshold = abs(mean_delta + self.delta_std_dev_factor * std_delta)

            macro_context = self.context_collector.get_context()
            historical_profile = macro_context.get("historical_vp", {})
            open_ms, close_ms = self.window_end_ms - self.window_ms, self.window_end_ms

            self.levels.update_from_vp(historical_profile)

            try:
                pipeline = DataPipeline(
                    valid_window_data,
                    self.symbol,
                    time_manager=self.time_manager,
                )

                flow_metrics = self.flow_analyzer.get_flow_metrics(reference_epoch_ms=close_ms)
                ob_event = self.orderbook_analyzer.analyze_order_book(event_epoch_ms=close_ms, window_id=str(close_ms))

                enriched = pipeline.enrich()
                # Inclui contexto de mercado e ambiente no pipeline
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

                # ---- LOG DO HEATMAP: COM DECIMAIS E SEGUNDOS (ajuste pedido) ----
                try:
                    liquidity_data = flow_metrics.get("liquidity_heatmap", {})
                    clusters = liquidity_data.get("clusters", [])
                    if clusters:
                        print(f"\n📊 LIQUIDITY HEATMAP - Janela #{self.window_count}:")
                        for i, cluster in enumerate(clusters[:3]):
                            age_ms = cluster.get("age_ms", 0)
                            try:
                                age_s = float(age_ms) / 1000.0
                            except Exception:
                                age_s = 0.0
                            print(
                                f"   Cluster {i+1}: ${cluster.get('center', 0.0):,.2f} | "
                                f"Vol: {cluster.get('total_volume', 0):,.0f} | "
                                f"Imb: {cluster.get('imbalance_ratio', 0.0):.2f} | "
                                f"Trades: {cluster.get('trades_count', 0)} | "
                                f"Age: {age_s:.1f}s"
                            )
                    else:
                        print(f"\n📊 LIQUIDITY HEATMAP - Janela #{self.window_count}: Nenhum cluster detectado")
                except Exception as e:
                    logging.error(f"Erro ao logar liquidity heatmap: {e}")

                features = pipeline.get_final_features()
                self.feature_store.save_features(window_id=str(close_ms), features=features)

                # ▼▼ NEW: extratos que vamos injetar nos eventos
                ml_payload = features.get("ml_features", {}) or {}
                enriched_snapshot = features.get("enriched", {}) or {}
                contextual_snapshot = features.get("contextual", {}) or {}

            except Exception as e:
                logging.error(f"Erro no DataPipeline: {e}")
                return

            derivatives_context = macro_context.get("derivatives", {})  # ✅ novo: cache local
            for signal in signals:
                if signal.get("is_signal", False):
                    # ✅ anexa derivativos ao evento para consumo pela IA
                    if "derivatives" not in signal:
                        signal["derivatives"] = derivatives_context

                    if "fluxo_continuo" not in signal and flow_metrics:
                        signal["fluxo_continuo"] = flow_metrics

                    # ✅ adiciona contexto de mercado e ambiente ao sinal
                    try:
                        if "market_context" not in signal:
                            signal["market_context"] = macro_context.get("market_context", {})
                        if "market_environment" not in signal:
                            signal["market_environment"] = macro_context.get("market_environment", {})
                    except Exception:
                        pass

                    # ▼▼ NEW: anexa ML features e snapshots ao evento para a IA usar
                    signal.setdefault("features_window_id", str(close_ms))
                    signal["ml_features"] = ml_payload
                    signal["enriched_snapshot"] = enriched_snapshot
                    signal["contextual_snapshot"] = contextual_snapshot

                    self.levels.add_from_event(signal)
                    self.event_bus.publish("signal", signal)
                    self.event_saver.save_event(signal)

                    if "timestamp" not in signal:
                        signal["timestamp"] = datetime.fromisoformat(
                            signal.get("timestamp_ny", datetime.now(self.ny_tz).isoformat(timespec="seconds")).replace("Z", "+00:00")
                        ).strftime("%Y-%m-%d %H:%M:%S")

                    if signal.get("tipo_evento") != "OrderBook":
                        adicionar_memoria_evento(signal)

                    self._log_event(signal)

            # Verifica toques em zonas
            preco_atual = enriched.get("ohlc", {}).get("close", 0)
            if preco_atual > 0:
                try:
                    touched = self.levels.check_price(float(preco_atual))
                    for z in touched:
                        zone_event = signals[0].copy() if signals else {}
                        zone_event.update(
                            {
                                "tipo_evento": "Zona",
                                "resultado_da_batalha": f"Toque em Zona {z.kind}",
                                "descricao": f"Preço {preco_atual} tocou {z.kind} {z.timeframe} [{z.low} ~ {z.high}]",
                                "zone_context": z.to_dict(),
                                "preco_fechamento": preco_atual,
                                "timestamp": datetime.now(self.ny_tz).isoformat(timespec="seconds"),
                            }
                        )
                        if "historical_confidence" not in zone_event:
                            zone_event["historical_confidence"] = calcular_probabilidade_historica(zone_event)
                        self.event_bus.publish("zone_touch", zone_event)
                        self.event_saver.save_event(zone_event)
                        adicionar_memoria_evento(
                            {
                                "timestamp": z.last_touched or datetime.now(self.ny_tz).isoformat(timespec="seconds"),
                                "tipo_evento": "Zona",
                                "resultado_da_batalha": f"Toque {z.kind}",
                                "delta": zone_event.get("delta", 0),
                                "volume_total": zone_event.get("volume_total", 0),
                            }
                        )
                except Exception as e:
                    logging.error(f"Erro ao verificar toques em zonas: {e}")

            window_volume = enriched.get("volume_total", 0)
            window_delta = enriched.get("delta_fechamento", 0)
            window_close = enriched.get("ohlc", {}).get("close", 0)

            self.volume_history.append(window_volume)
            self.delta_history.append(window_delta)
            if window_close > 0:
                self.close_price_history.append(window_close)

            # Atualiza histórico de volatilidade (usa volatilidade de 5 barras se disponível)
            try:
                price_feats = (ml_payload.get('price_features') or {})
                current_volatility = None
                if 'volatility_5' in price_feats:
                    current_volatility = price_feats['volatility_5']
                elif 'volatility_1' in price_feats:
                    current_volatility = price_feats['volatility_1']
                else:
                    for k, v in price_feats.items():
                        if k.startswith('volatility_'):
                            current_volatility = v
                            break
                if current_volatility is not None:
                    self.volatility_history.append(float(current_volatility))
            except Exception:
                pass

            # (Opcional) Log curto de ML features por janela
            try:
                pf = ml_payload.get("price_features", {}) if ml_payload else {}
                vf = ml_payload.get("volume_features", {}) if ml_payload else {}
                mf = ml_payload.get("microstructure", {}) if ml_payload else {}
                if pf or vf or mf:
                    print(
                        f"   ML: ret5={pf.get('returns_5', 0):+.4f} "
                        f"vol5={pf.get('volatility_5', 0):.5f} "
                        f"V/SMA={vf.get('volume_sma_ratio', 0):.2f} "
                        f"BSpress={vf.get('buy_sell_pressure', 0):+.2f} "
                        f"OBslope={mf.get('order_book_slope', 0):+.3f} "
                        f"FlowImb={mf.get('flow_imbalance', 0):+.3f}"
                    )
            except Exception:
                pass

            # Gera alertas institucionais
            if generate_alerts is not None:
                try:
                    # Obtenha níveis de suporte/resistência se módulo estiver disponível
                    if detect_support_resistance is not None:
                        try:
                            price_series = pipeline.df['p'] if hasattr(pipeline, 'df') else None
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
                            # Cooldown por tipo de alerta
                            atype = alert.get('type', 'GENERIC')
                            now_s = time.time()
                            last_ts = self._last_alert_ts.get(atype, 0)
                            if now_s - last_ts < self._alert_cooldown_sec:
                                continue
                            self._last_alert_ts[atype] = now_s

                            # Constrói uma descrição amigável do alerta
                            desc_parts = [f"Tipo: {alert.get('type')}"]
                            if 'level' in alert:
                                desc_parts.append(f"Nível: {alert['level']}")
                            if 'threshold_exceeded' in alert:
                                desc_parts.append(f"Fator: {alert['threshold_exceeded']}")
                            if 'level' not in alert and 'threshold_exceeded' not in alert:
                                for k, v in alert.items():
                                    if k not in ('type', 'severity', 'probability', 'action'):
                                        desc_parts.append(f"{k}: {v}")
                            descricao_alert = " | ".join(desc_parts)
                            print(f"🔔 ALERTA: {descricao_alert}")
                            logging.info(f"🔔 ALERTA: {descricao_alert}")
                            # Cria evento de alerta e salva
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

            print(
                f"[{datetime.now(self.ny_tz).strftime('%H:%M:%S')} NY] 🟡 Janela #{self.window_count} | Delta: {window_delta:,.2f} | Vol: {window_volume:,.2f}"
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
                    print(f"   Macro Context: {trends_str}")
            if historical_profile and historical_profile.get("daily"):
                vp = historical_profile["daily"]
                print(
                    f"   VP Diário: POC @ {vp.get('poc', 0):,.2f} | VAL: {vp.get('val', 0):,.2f} | VAH: {vp.get('vah', 0):,.2f}"
                )

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
            logging.info("🎯 Iniciando Enhanced Market Bot...")
            print("=" * 80)
            self.connection_manager.connect()
        except KeyboardInterrupt:
            logging.info("⏹️ Bot interrompido pelo usuário.")
        except Exception as e:
            logging.critical(f"❌ Erro crítico ao executar o bot: {e}", exc_info=True)
        finally:
            self._cleanup_handler()


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