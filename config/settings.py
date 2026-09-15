import os
from config.env_policy import maybe_load_dotenv

# PF-D: carregamento .env centralizado e desabilitável. LOAD_DOTENV=0 ou
# OBSERVATION_MODE=1 => NÃO carrega (observation nunca carrega). Default
# True = compatibilidade com desenvolvimento normal.
maybe_load_dotenv()

# ===== 🆕 CONFIGURAÇÕES DE CIRCUIT BREAKER (ORDERBOOK) =====
# CONFIGURAÇÃO OTIMIZADA PARA MAIOR RESILIÊNCIA
ORDERBOOK_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5  # Abri após 5 falhas consecutivas
ORDERBOOK_CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2  # Fechar após 2 sucessos em half-open
ORDERBOOK_CIRCUIT_BREAKER_TIMEOUT_SECONDS = 30.0  # Tempo em OPEN antes de half-open
ORDERBOOK_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = 3  # Máximo de tentativas em half-open

# NOVAS CONFIGURAÇÕES PARA MAIOR ROBUSTEZ
ORDERBOOK_CIRCUIT_BREAKER_ENABLE_FALLBACK = True  # Habilita fallback para REST quando WS falhar
ORDERBOOK_CIRCUIT_BREAKER_MAX_RETRY_ATTEMPTS = 3  # Máximo de tentativas de retry
ORDERBOOK_CIRCUIT_BREAKER_BASE_RETRY_DELAY = 1.0  # Delay base para retry
ORDERBOOK_CIRCUIT_BREAKER_MAX_RETRY_DELAY = 10.0  # Delay máximo para retry

# CONFIGURAÇÕES DE FALLBACK REST API
ORDERBOOK_REST_FALLBACK_ENABLED = True  # Habilita fallback REST quando WebSocket falhar
ORDERBOOK_REST_REQUEST_TIMEOUT = 15.0  # Timeout para requisições REST
ORDERBOOK_REST_MAX_RETRIES = 5  # Máximo de retries para REST
ORDERBOOK_REST_RETRY_BACKOFF = 2.0  # Fator de backoff exponencial
ORDERBOOK_REST_JITTER_RANGE = 0.25  # Percentual de jitter (0.25 = 25%)
ORDERBOOK_SYNC_TIMEOUT_SEC = 1.5   # Timeout para fetch síncrono no fechamento da janela (opção i)

# CONFIGURAÇÕES DE WEBSOCKET ORDERBOOK (HABILITADO)
ORDERBOOK_WS_ENABLED = True  # ✅ Habilitado para WebSocket
ORDERBOOK_WS_ENDPOINT = "wss://fstream.binance.com/market/ws/{symbol}@depth20@100ms"  # futures; NÃO usar stream.binance.com
ORDERBOOK_WS_RECONNECT_ATTEMPTS = 15  # ✅ Mais tentativas

ORDERBOOK_WS_RECONNECT_DELAY = 2.0  # ✅ Delay inicial menor
ORDERBOOK_WS_MAX_RECONNECT_DELAY = 60.0  # ✅ Delay máximo
ORDERBOOK_WS_BACKOFF_FACTOR = 1.5  # ✅ Backoff exponencial
ORDERBOOK_WS_PING_INTERVAL = 20.0  # ✅ Ping a cada 20s (Binance requer)
ORDERBOOK_WS_PING_TIMEOUT = 10.0  # ✅ Timeout para pong
ORDERBOOK_WS_CONNECT_TIMEOUT = 30.0  # ✅ Timeout de conexão
ORDERBOOK_WS_MESSAGE_TIMEOUT = 5.0  # ✅ Timeout para mensagens
ORDERBOOK_WS_RATE_LIMIT_PER_MIN = 1200  # ✅ Rate limit Binance (1200/min)
ORDERBOOK_WS_MAX_RETRIES = 3  # ✅ Máximo de retries por operação
ORDERBOOK_WS_HEARTBEAT_TIMEOUT = 60.0  # ✅ Timeout sem heartbeat

# ===== CONFIGURAÇÕES DE HEALTH MONITOR =====
HEALTH_CHECK_TIMEOUT = 90      # Warning após 90s de silêncio
HEALTH_CHECK_CRITICAL = 180    # Critical após 180s de silêncio  
HEALTH_CHECK_INTERVAL = 30     # Verificação a cada 30s

# ===== CONFIGURAÇÕES DO HEALTH CHECK DO CONTAINER (pipeline_health) =====
# Intervalo esperado de cada estágio (segundos). O /health falha (503) quando
# um estágio fica sem bater por > intervalo * PIPELINE_HEALTH_THRESHOLD_MULTIPLIER.
PIPELINE_HEALTH_STAGE_INTERVALS = {
    "ws": 120,                    # mensagens do WebSocket (critical 240s)
    "trade_ingestion": 30,        # trades aceitos no buffer
    "trade_buffer": 30,           # batches processados
    "orderbook": 60,              # snapshots analisados
    "window_processor": 300,      # janelas (WINDOW_SIZE_MINUTES)
    "ai": 600,                    # análises de IA
    "event_saver": 600,           # eventos salvos
}
# Multiplicador: threshold crítico = intervalo esperado * multiplicador
PIPELINE_HEALTH_THRESHOLD_MULTIPLIER = 2.0
# Silêncio do WebSocket para estado "degraded" (conexão viva sem mensagens)
PIPELINE_HEALTH_DEGRADED_WS_SILENCE_SECONDS = 90

# ===== ALERTAS DE TRANSIÇÃO DE HEALTH (pipeline_health) =====
# Máximo de 1 alerta por transição de estado a cada X segundos
ALERT_STATE_COOLDOWN_SECONDS = 600
# Lembrete para estado degradado prolongado (unhealthy OU degraded): re-dispara
# um alerta do tipo "health_state_reminder" a cada X segundos enquanto o estado
# permanecer fora de healthy (cobre o gap de transição única).
ALERT_UNHEALTHY_REMINDER_SECONDS = 1800
# Canal opcional: webhook HTTP (formato Slack-compatível). Se vazio, o alerta
# é apenas logado (structured log) e exposto via métrica.
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")

# ===== CONFIGURAÇÕES OCI (DESABILITADO) =====
OCI_COMPARTMENT_ID = None      # None = OCI desabilitado

# ===== CONFIGURAÇÕES DE ORDERBOOK ANALYZER =====
ORDER_BOOK_DEPTH_LEVELS = [1, 5, 10, 25]
SPREAD_TIGHT_THRESHOLD_BPS = 0.2
SPREAD_AVG_WINDOWS_MIN = [60, 1440]
ORDERBOOK_CRITICAL_IMBALANCE = 0.95
ORDERBOOK_MIN_DOMINANT_USD = 2_000_000.0
ORDERBOOK_MIN_RATIO_DOM = 20.0
ORDERBOOK_REQUEST_TIMEOUT = 10.0
ORDERBOOK_RETRY_DELAY = 2.0
ORDERBOOK_MAX_RETRIES = 3
ORDERBOOK_MAX_REQUESTS_PER_MIN = 10
ORDERBOOK_CACHE_TTL = 15.0
ORDERBOOK_MAX_STALE = 60.0
ORDERBOOK_MIN_DEPTH_USD = 1_000.0
ORDERBOOK_ALLOW_PARTIAL = False
ORDERBOOK_USE_FALLBACK = True
ORDERBOOK_FALLBACK_MAX_AGE = 120
ORDERBOOK_EMERGENCY_MODE = False

# ===== CONFIGURAÇÕES DE CONTEXT COLLECTOR =====
CONTEXT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
CONTEXT_EMA_PERIOD = 21
CONTEXT_ATR_PERIOD = 14
CONTEXT_UPDATE_INTERVAL_SECONDS = 300  # 5 min (era 60s — reduzia chamadas yFinance)
INTERMARKET_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DERIVATIVES_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
VP_NUM_DAYS_HISTORY = 30
VP_VALUE_AREA_PERCENT = 0.7
LIQUIDATION_MAP_DEPTH = 500.0
EXTERNAL_MARKETS = {
    "SP500": "^GSPC",
    "DXY": "DX-Y.NYB",     # ✅ Correto (era ^DXY)
    "NASDAQ": "^IXIC",
    "TNX": "^TNX",
    "GOLD": "GC=F",        # ✅ Correto (era XAUUSD)
    "WTI": "CL=F",         # ✅ Correto (era CL)
}
ENABLE_ONCHAIN = False
ONCHAIN_PROVIDERS = []
STABLECOIN_FLOW_TRACKING = False
ENABLE_ALPHAVANTAGE = True
CORRELATION_LOOKBACK = 50
VOLATILITY_PERCENTILES = (0.01, 0.03)  # crypto: <1%=LOW, 1-3%=NORMAL, >3%=HIGH
ADX_PERIOD = 14
RSI_PERIODS = {"short": 14, "long": 21}
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# ===== CONFIGURAÇÕES DE HISTORICAL PROFILER =====
VP_ADVANCED = True

# ===== PARÂMETROS DE TRADING =====
SYMBOL = "BTCUSDT"
MARKET_TYPE = "binance_futures_perp"
STREAM_URL = f"wss://fstream.binance.com/market/ws/{SYMBOL.lower()}@aggTrade"  # aggTrade Binance USD-M Futures. NÃO usar stream.binance.com (Spot) — ver docs/audit/AUDITORIA_JANELAS_EXTRAIDAS_R4b_2026-09-03.md
WINDOW_SIZE_MINUTES = 1
VOL_FACTOR_EXH = 2.5
HISTORY_SIZE = 100
DELTA_STD_DEV_FACTOR = 2.0
CONTEXT_SMA_PERIOD = 20
LIQUIDITY_FLOW_ALERT_PERCENTAGE = 0.15
WALL_STD_DEV_FACTOR = 3.0

# ===== CONFIGURACOES DO TRADE BUFFER E AGGREGATES =====
TRADES_BUFFER_SIZE = 10000          # Aumentado de 5000 para evitar overflow
TRADES_BUFFER_BACKPRESSURE = 0.8    # Aumentado de 0.6 para 0.8 (alerta mais tarde)
TRADES_BUFFER_BATCH_SIZE = 500      # Aumentado de 200 para drenar mais rapido
TRADES_BUFFER_PROCESSING_INTERVAL_MS = 5
TRADES_BUFFER_MAX_PROCESSING_MS = 500.0
ROLLING_AGGREGATE_TARGET_TPS = 100
ROLLING_AGGREGATE_ABSOLUTE_MAX_TRADES = 150_000

# EventSaver json outputs
# SQLite é a fonte de verdade (v5); snapshot JSON desativado por default.
EVENT_SAVER_WRITE_JSON = False
EVENT_SAVER_WRITE_JSONL = True
EVENT_SAVER_MAX_JSON_EVENTS = 1000
EVENT_SAVER_MAX_JSON_MB = 50

# ===== CREDENCIAIS BINANCE =====
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# ===== API KEYS ADICIONAIS =====
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===== ENRICHMENT / ADAPTIVE THRESHOLDS =====
ENABLE_DATA_ENRICHMENT = True

ABSORPTION_THRESHOLD_BASE = 0.15   # base para threshold de absorção
FLOW_THRESHOLD_BASE = 0.10        # base para threshold de fluxo
MIN_VOL_FACTOR = 0.5              # limite inferior de ajuste
MAX_VOL_FACTOR = 2.0              # limite superior de ajuste

# ===== LOGGING CONFIGURATION =====
LOG_LEVEL = "INFO"  # Default logging level

# ===== OTIMIZAÇÕES DE LATÊNCIA =====
# Cache TTL para dados macro (aumentado para reduzir chamadas API)
MACRO_CACHE_TTL = 300  # 5 minutos (era 60s)

# Frequência de cálculo de correlações (reduzido para menor overhead)
CORRELATION_CALC_INTERVAL = 5  # Calcular a cada 5 janelas (era 1)

# Número de workers para processamento paralelo
PARALLEL_WORKERS = 4  # Workers para processamento paralelo de trades

# --- UPDATE INTERVALS (Seconds) ---
# Binance/Crypto (Websockets mantêm realtime, isso é para snapshots/fallback)
CRYPTO_POLLING_INTERVAL = 60   # 1 minuto (Backup do Websocket)
CRYPTO_INTERVAL = 60            # 1 minuto (para consistência)

# Macro Data (Twelve Data/Alpha Vantage)
# 15 minutos = 96 requisições/dia (Muito seguro para conta Free de 800/dia)
CROSS_ASSET_INTERVAL = 900

# Economic Data (FRED)
# 4 horas = Dados econômicos mudam raramente
ECONOMIC_DATA_INTERVAL = 14400

# AI Analysis
# Não basear em tempo, mas em eventos. Se precisar de tempo:
AI_ANALYSIS_INTERVAL = 300     # 5 minutos (Evita spammar a Groq)

# Otimização de IA - pula análise quando volume baixo em regime lateral
AI_SKIP_VOLUME_THRESHOLD = 100_000  # USD - threshold para pular IA em sideways

# ===== ML / HYBRID DECISION =====
HYBRID_ENABLED = False          # Desativado até dataset ter 500+ amostras balanceadas
# NOTA: O estado de ML obsoleto é controlado na fonte via ml/models/model_metadata_latest.json
HYBRID_MODE = "llm_primary"
HYBRID_MODEL_WEIGHT = 0.6
HYBRID_LLM_WEIGHT = 0.4

# ===== CONFIGURAÇÕES DE IA / PIPELINE =====
AI_MIN_INTERVAL_SEC = 300      # Intervalo mínimo entre análises da IA (5 min, era 60s)
AI_TEST_MIN_CHARS = 10         # Mínimo de chars para teste da IA ser considerado ok
MIN_TRADES_FOR_PIPELINE = 10   # Mínimo de trades por janela para processar pipeline
WARMUP_WINDOWS = 3             # Janelas de aquecimento após reconexão

# ===== CVD DIVERGENCE (correção pós-reset) =====
CVD_DIV_WARMUP_SECONDS = 300      # Supressão da divergência CVD após reset do FlowAnalyzer (Camada 1)
CVD_DIV_MIN_PERIOD_SECONDS = 10   # Período mínimo pós-reset p/ comparar preço vs CVD (Camada 2.3)

# ===== FLOW IMBALANCE (guard de amostra mínima) =====
FLOW_IMBALANCE_MIN_TRADES = 5     # Abaixo disso, flow_imbalance não é emitido (ruído estatístico)

# ===== CFTC COT CONTEXT (P6 — context-only, default OFF) =====
# Fonte semanal oficial CFTC/CME (TFF futures-only), separada do positioning
# Binance. Fora do hot path (CftcCotUpdater em background). Ativar só após
# P4/P5 aprovados em produção.
ENABLE_CFTC_COT_CONTEXT = os.getenv("ENABLE_CFTC_COT_CONTEXT", "0") == "1"

# ===== CONFIGURAÇÕES DO PAYLOAD TRIPWIRE =====
PAYLOAD_TRIPWIRE_GUARDRAIL_MAX = 0.30   # Max 30% de bloqueios (era sem config, default implícito)
PAYLOAD_TRIPWIRE_ABORT_MAX = 0.02       # Max 2% de aborts
PAYLOAD_TRIPWIRE_FALLBACK_MAX = 0.03    # Max 3% de fallbacks
PAYLOAD_TRIPWIRE_BYTES_P95_MAX = 90000  # Max 90KB no P95
PAYLOAD_TRIPWIRE_CACHE_HIT_MIN = 0.10   # Min 10% de cache hits
