import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

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

# CONFIGURAÇÕES DE WEBSOCKET ORDERBOOK (HABILITADO)
ORDERBOOK_WS_ENABLED = True  # ✅ Habilitado para WebSocket
ORDERBOOK_WS_ENDPOINT = "wss://stream.binance.com:9443/ws/{symbol}@depth"  # ✅ Endpoint correto
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
CONTEXT_UPDATE_INTERVAL_SECONDS = 60
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
VOLATILITY_PERCENTILES = (0.35, 0.65)
ADX_PERIOD = 14
RSI_PERIODS = {"short": 14, "long": 21}
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# ===== CONFIGURAÇÕES DE HISTORICAL PROFILER =====
VP_ADVANCED = True

# ===== PARÂMETROS DE TRADING =====
SYMBOL = "BTCUSDT"
STREAM_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@trade"
WINDOW_SIZE_MINUTES = 2
VOL_FACTOR_EXH = 2.5
HISTORY_SIZE = 100
DELTA_STD_DEV_FACTOR = 2.0
CONTEXT_SMA_PERIOD = 20
LIQUIDITY_FLOW_ALERT_PERCENTAGE = 0.15
WALL_STD_DEV_FACTOR = 3.0

# ===== CONFIGURACOES DO TRADE BUFFER =====
TRADES_BUFFER_SIZE = 5000
TRADES_BUFFER_BACKPRESSURE = 0.6
TRADES_BUFFER_BATCH_SIZE = 200
TRADES_BUFFER_PROCESSING_INTERVAL_MS = 5
TRADES_BUFFER_MAX_PROCESSING_MS = 500.0

# EventSaver json outputs
EVENT_SAVER_WRITE_JSON = True
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
