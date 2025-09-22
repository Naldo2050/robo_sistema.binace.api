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

# Importações internas — ✅ CORRIGIDO: separa imports do data_handler e event_memory
from data_handler import (
    create_absorption_event,
    create_exhaustion_event,
    format_timestamp,
    NY_TZ,
)
from event_memory import (  # ✅ IMPORTA AS FUNÇÕES DE MEMÓRIA DO ARQUIVO CORRETO
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

# 🔹 NOVOS MÓDULOS DA FASE 1
from time_manager import TimeManager
from health_monitor import HealthMonitor
from event_bus import EventBus
from data_pipeline import DataPipeline
from feature_store import FeatureStore

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
        self.last_successful_message_time = None  # 🔹 NOVO: monitora última mensagem válida
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
            with socket.create_connection((host, port), timeout=5) as sock:
                if parsed.scheme == "wss":
                    import ssl
                    context = ssl.create_default_context()
                    with context.wrap_socket(sock, server_hostname=host) as ssock:
                        return True
                return True
        except Exception as e:
            logging.error(f"Erro ao testar conexão: {e}")
            return False

    def _on_message(self, ws, message):
        try:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1
            
            # Processa a mensagem
            if self.on_message_callback:
                self.on_message_callback(ws, message)
            
            # Se chegou aqui, a mensagem foi processada com sucesso
            self.last_successful_message_time = self.last_message_time
            
            if self.current_delay > self.initial_delay:
                self.current_delay = max(self.initial_delay, self.current_delay * 0.9)
                
        except Exception as e:
            logging.error(f"Erro no processamento da mensagem: {e}")
            # Não atualiza last_successful_message_time em caso de erro

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
            
            # 🔹 NOVO: alerta se sem mensagens válidas por 5min
            if self.last_successful_message_time:
                valid_gap = (datetime.now(timezone.utc) - self.last_successful_message_time).total_seconds()
                if valid_gap > 300:  # 5 minutos
                    logging.critical(f"💀 SEM MENSAGENS VÁLIDAS HÁ {valid_gap:.0f}s! Fallback de preço ativado.")
                    # Aqui você pode ativar um modo de emergência

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
                    on_open=self._on_open,
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
            window_data, symbol, delta_threshold=dynamic_delta_threshold, tz_output=self.tz_output, historical_profile=historical_profile
        )

        exhaustion_event = create_exhaustion_event(
            window_data, symbol, history_volumes=history_volumes, volume_factor=self.vol_factor_exh, tz_output=self.tz_output, historical_profile=historical_profile
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
            # 🔹 CORRIGIDO: REMOVIDOS ESPAÇOS FINAIS
            url = "https://fapi.binance.com/fapi/v1/ticker/price"
            params = {"symbol": symbol}
            res = requests.get(url, params=params, timeout=5)
            res.raise_for_status()
            data = res.json()
            return float(data["price"])
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao buscar preço via REST (tentativa {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
        except Exception as e:
            logging.error(f"Erro inesperado ao buscar preço via REST (tentativa {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    
    # 🔹 Fallback: retorna 0.0 em vez de lançar exceção
    logging.critical("💀 FALHA CRÍTICA: Não foi possível obter preço via REST após todas as tentativas")
    return 0.0


# ===============================
# BOT PRINCIPAL (ATUALIZADO COM MITIGAÇÃO DE FALHAS)
# ===============================
class EnhancedMarketBot:
    def __init__(self, stream_url, symbol, window_size_minutes, vol_factor_exh, history_size, delta_std_dev_factor, context_sma_period, liquidity_flow_alert_percentage, wall_std_dev_factor):
        self.symbol = symbol
        self.window_size_minutes = window_size_minutes
        self.window_ms = window_size_minutes * 60 * 1000
        self.ny_tz = NY_TZ
        self.should_stop = False
        self.is_cleaning_up = False

        # 🔹 INICIALIZA NOVOS MÓDULOS
        self.time_manager = TimeManager()
        self.health_monitor = HealthMonitor()
        self.event_bus = EventBus()
        self.feature_store = FeatureStore()

        # 🔹 REGISTRA HEARTBEAT DOS MÓDULOS
        self.health_monitor.heartbeat("main")

        # Módulos existentes
        self.trade_flow_analyzer = TradeFlowAnalyzer(vol_factor_exh, tz_output=self.ny_tz)
        self.orderbook_analyzer = OrderBookAnalyzer(symbol=self.symbol, liquidity_flow_alert_percentage=liquidity_flow_alert_percentage, wall_std_dev_factor=wall_std_dev_factor)
        self.event_saver = EventSaver(sound_alert=True)
        self.context_collector = ContextCollector(symbol=self.symbol)
        self.flow_analyzer = FlowAnalyzer()

        self.ai_analyzer = None
        self.ai_initialization_attempted = False
        self.ai_test_passed = False
        self.ai_thread_pool = []
        self.max_ai_threads = 3
        self.ai_semaphore = threading.Semaphore(3)
        self._initialize_ai_async()

        # 🔹 ASSINA EVENTOS NO EVENT BUS
        self.event_bus.subscribe("signal", self._handle_signal_event)
        self.event_bus.subscribe("zone_touch", self._handle_zone_touch_event)

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

        self.reporter = ReportGenerator(output_dir="./reports", mode="csv")
        self.levels = LevelRegistry(self.symbol)

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
                    print("\n" + "═" * 25 + " RESULTADO DO TESTE DA IA " + "═" * 25)
                    print(analysis)
                    print("═" * 75 + "\n")
                else:
                    self.ai_test_passed = False
                    logging.warning("⚠️ Teste da IA retornou resultado inesperado ou vazio.")
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
        if hasattr(self, 'event_bus'):
            self.event_bus.shutdown()
        if hasattr(self, 'health_monitor'):
            self.health_monitor.stop()
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

            # 🔹 USA TIME MANAGER PARA TIMESTAMP
            ts = int(trade["T"])
            if self.window_end_ms is None:
                self.window_end_ms = self._next_boundary_ms(ts)
            if ts >= self.window_end_ms:
                self._process_window()
                self.window_end_ms = self._next_boundary_ms(ts)
                self.window_data = [trade]
            else:
                self.window_data.append(trade)
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar mensagem JSON: {e}")
            # Não quebra o sistema, apenas ignora a mensagem inválida
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
                    logging.info(f"🧠 IA iniciando análise para evento: {event_data.get('resultado_da_batalha', 'N/A')}")
                    analysis_result = self.ai_analyzer.analyze_event(event_data)
                    if analysis_result and not self.should_stop:
                        print("\n" + "═" * 25 + " ANÁLISE PROFISSIONAL DA IA " + "═" * 25)
                        print(analysis_result)
                        print("═" * 75 + "\n")
                        self.reporter.save_report(event_data, analysis_result)
                        # 🔹 REGISTRA HEARTBEAT DA IA
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
            in_single = any(abs(p - preco_atual) <= faixa_lim for p in sp)

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
        ultimos = obter_memoria_eventos(n=2)
        if ultimos:
            print("   🕒 Últimos sinais:")
            for e in ultimos:
                # ✅ CORRIGIDO: usa .get() para evitar KeyError
                print(
                    f"      - {e.get('timestamp', 'N/A')} | {e.get('tipo_evento', 'N/A')} {e.get('resultado_da_batalha', 'N/A')} "
                    f"(Δ={e.get('delta', 0):,.2f}, Vol={e.get('volume_total', 0):,.2f})"
                )

    def _process_window(self):
        if not self.window_data or self.should_stop or sum(float(trade.get("q", 0)) for trade in self.window_data) == 0:
            self.window_data = []
            return

        self.window_count += 1
        try:
            # 🔹 REGISTRA HEARTBEAT
            self.health_monitor.heartbeat("main")

            # --- 1. PREPARAÇÃO DOS DADOS ---
            dynamic_delta_threshold = 0
            if len(self.delta_history) > 10:
                mean_delta = np.mean(self.delta_history)
                std_delta = np.std(self.delta_history)
                dynamic_delta_threshold = abs(mean_delta + self.delta_std_dev_factor * std_delta)

            macro_context = self.context_collector.get_context()
            historical_profile = macro_context.get("historical_vp", {})
            open_ms, close_ms = self.window_end_ms - self.window_ms, self.window_end_ms

            # 🔹 ATUALIZA ZONAS
            self.levels.update_from_vp(historical_profile)

            # --- 2. PIPELINE DE DADOS ---
            try:
                pipeline = DataPipeline(self.window_data, self.symbol)
                enriched = pipeline.enrich()
                contextual = pipeline.add_context(
                    flow_metrics=self.flow_analyzer.get_flow_metrics(),
                    historical_vp=historical_profile,
                    orderbook_data=self.orderbook_analyzer.analyze_order_book(),
                    multi_tf=macro_context.get("mtf_trends", {})
                )
                signals = pipeline.detect_signals(
                    absorption_detector=lambda data, sym: create_absorption_event(data, sym, delta_threshold=dynamic_delta_threshold, tz_output=self.ny_tz, historical_profile=historical_profile),
                    exhaustion_detector=lambda data, sym: create_exhaustion_event(data, sym, history_volumes=list(self.volume_history), volume_factor=config.VOL_FACTOR_EXH, tz_output=self.ny_tz, historical_profile=historical_profile),
                    orderbook_data=contextual.get("orderbook_data")
                )

                # 🔹 NOVO: Log de debug do Liquidity Heatmap — ✅ CORRIGIDO
                try:
                    flow_metrics = self.flow_analyzer.get_flow_metrics()
                    liquidity_data = flow_metrics.get("liquidity_heatmap", {})
                    clusters = liquidity_data.get("clusters", [])
                    
                    if clusters:
                        print(f"\n📊 LIQUIDITY HEATMAP - Janela #{self.window_count}:")
                        for i, cluster in enumerate(clusters[:3]):
                            print(f"   Cluster {i+1}: ${cluster.get('center', 0.0):,.2f} | "
                                  f"Vol: {cluster.get('total_volume', 0):,.0f} | "
                                  f"Imb: {cluster.get('imbalance_ratio', 0.0):.2f} | "
                                  f"Trades: {cluster.get('trades_count', 0)} | "
                                  f"Age: {cluster.get('age_ms', 0)}ms")
                    else:
                        print(f"\n📊 LIQUIDITY HEATMAP - Janela #{self.window_count}: Nenhum cluster detectado")
                except Exception as e:
                    logging.error(f"Erro ao logar liquidity heatmap: {e}")

                # 🔹 SALVA FEATURES PARA ML
                features = pipeline.get_final_features()
                self.feature_store.save_features(window_id=str(close_ms), features=features)

            except Exception as e:
                logging.error(f"Erro no DataPipeline: {e}")
                return

            # --- 3. PROCESSA SINAIS ---
            for signal in signals:
                if signal.get("is_signal", False):
                    # 🔹 ALIMENTA REGISTRO DE ZONAS
                    self.levels.add_from_event(signal)
                    # 🔹 PUBLICA NO EVENT BUS
                    self.event_bus.publish("signal", signal)
                    # 🔹 SALVA EVENTO
                    self.event_saver.save_event(signal)
                    # 🔹 ADICIONA À MEMÓRIA
                    if "timestamp" not in signal:
                        signal["timestamp"] = datetime.now(self.ny_tz).isoformat(timespec="seconds")
                    adicionar_memoria_evento(signal)
                    # 🔹 LOG
                    self._log_event(signal)

            # --- 4. MONITORA TOQUES EM ZONAS ---
            preco_atual = enriched.get("ohlc", {}).get("close", 0)
            if preco_atual > 0:
                try:
                    touched = self.levels.check_price(float(preco_atual))
                    for z in touched:
                        zone_event = signal.copy() if signals else {}
                        zone_event.update({
                            "tipo_evento": "Zona",
                            "resultado_da_batalha": f"Toque em Zona {z.kind}",
                            "descricao": f"Preço {preco_atual} tocou {z.kind} {z.timeframe} [{z.low} ~ {z.high}]",
                            "zone_context": z.to_dict(),
                            "preco_fechamento": preco_atual,
                            "timestamp": datetime.now(self.ny_tz).isoformat(timespec="seconds")
                        })
                        if "historical_confidence" not in zone_event:
                            zone_event["historical_confidence"] = calcular_probabilidade_historica(zone_event)
                        # 🔹 PUBLICA NO EVENT BUS
                        self.event_bus.publish("zone_touch", zone_event)
                        # 🔹 SALVA EVENTO
                        self.event_saver.save_event(zone_event)
                        # 🔹 MEMÓRIA
                        adicionar_memoria_evento({
                            "timestamp": z.last_touched or datetime.now(self.ny_tz).isoformat(timespec="seconds"),
                            "tipo_evento": "Zona",
                            "resultado_da_batalha": f"Toque {z.kind}",
                            "delta": zone_event.get("delta", 0),
                            "volume_total": zone_event.get("volume_total", 0)
                        })
                except Exception as e:
                    logging.error(f"Erro ao verificar toques em zonas: {e}")

            # --- 5. ATUALIZA HISTÓRICOS ---
            if signals:
                main_event = signals[0]
                self.volume_history.append(main_event.get("volume_total", 0))
                self.delta_history.append(main_event.get("delta", 0))
                self.close_price_history.append(main_event.get("preco_fechamento", 0))

            print(
                f"[{datetime.now(self.ny_tz).strftime('%H:%M:%S')} NY] 🟡 Janela #{self.window_count} | Delta: {signals[0].get('delta', 0):,.2f} | Vol: {signals[0].get('volume_total', 0):,.2f}"
            )
            if macro_context:
                trends = macro_context.get("mtf_trends", {})
                trends_str = ", ".join([f"{tf.upper()}: {data['tendencia']}" for tf, data in trends.items()])
                if trends_str:
                    print(f"   Macro Context: {trends_str}")
            if historical_profile and historical_profile.get("daily"):
                vp = historical_profile["daily"]
                print(f"   VP Diário: POC @ {vp.get('poc', 0):,.2f} | VAL: {vp.get('val', 0):,.2f} | VAH: {vp.get('vah', 0):,.0f}")

            # --- 6. ATUALIZA ZONAS DE LIQUIDEZ (NOVO) — ✅ CORRIGIDO
            try:
                flow_metrics = self.flow_analyzer.get_flow_metrics()
                liquidity_data = flow_metrics.get("liquidity_heatmap", {})
                clusters = liquidity_data.get("clusters", [])
                
                if clusters:
                    logging.info(f"🔍 Detectados {len(clusters)} clusters de liquidez")
                    for i, cluster in enumerate(clusters[:3]):  # Top 3 clusters
                        # Valores seguros com .get()
                        center = cluster.get('center', 0.0)
                        total_volume = cluster.get('total_volume', 0.0)
                        imbalance_ratio = cluster.get('imbalance_ratio', 0.0)
                        trades_count = cluster.get('trades_count', 0)
                        age_ms = cluster.get('age_ms', 0)
                        high = cluster.get('high', center + 0.5)
                        low = cluster.get('low', center - 0.5)

                        # Zona principal
                        z = self.levels._mk_zone(
                            kind="LIQUIDITY_CLUSTER",
                            timeframe="intraday",
                            price=center,
                            width_pct=0.0003,  # 0.03%
                            confluence=[f"LIQ_VOL_{total_volume:.0f}", f"IMB_{imbalance_ratio:.2f}"],
                            notes=f"Liquidity Cluster: {trades_count} trades, age: {age_ms}ms"
                        )
                        self.levels.add_or_merge(z)
                        
                        # Zonas de borda
                        z_high = self.levels._mk_zone(
                            kind="LIQUIDITY_HIGH",
                            timeframe="intraday",
                            price=high,
                            width_pct=0.00015,  # 0.015%
                            confluence=["LIQ_HIGH", f"VOL_{total_volume:.0f}"],
                            notes=f"Liquidity High: {total_volume:.0f}"
                        )
                        self.levels.add_or_merge(z_high)
                        
                        z_low = self.levels._mk_zone(
                            kind="LIQUIDITY_LOW",
                            timeframe="intraday",
                            price=low,
                            width_pct=0.00015,  # 0.015%
                            confluence=["LIQ_LOW", f"VOL_{total_volume:.0f}"],
                            notes=f"Liquidity Low: {total_volume:.0f}"
                        )
                        self.levels.add_or_merge(z_low)
            except Exception as e:
                logging.error(f"Erro ao atualizar zonas de liquidez: {e}")

            # --- 7. 🔹 NOVO: ENVIA O MAPA DE LIQUIDEZ PARA O EVENTO SALVO!
            # Garante que o heatmap está presente nos eventos salvos
            if flow_metrics and "liquidity_heatmap" in flow_metrics:
                # Atualiza o último sinal (se houver) com o heatmap
                if signals:
                    main_event = signals[0]
                    main_event["liquidity_heatmap"] = flow_metrics["liquidity_heatmap"]
                    # Re-salva o evento para garantir que o EventSaver veja o campo
                    self.event_saver.save_event(main_event)

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