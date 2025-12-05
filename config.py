# config.py - v3.0.0 - PYDANTIC REFACTOR
# ==============================================================================
# CONFIGURAÇÕES DO BOT COM VALIDAÇÃO DE TIPOS (PYDANTIC)
# ==============================================================================

import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

class Settings(BaseSettings):
    """
    Classe de configuração centralizada com validação automática de tipos.
    Lê valores do arquivo .env ou usa os defaults definidos aqui.
    """
    
    # === Ambiente e Identificação ===
    environment: str = Field("dev", env="ENVIRONMENT")
    
    # === Credenciais de IA ===
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")
    dashscope_api_key: Optional[str] = Field(None, env="DASHSCOPE_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # === Conexão e Ativo ===
    symbol: str = Field("BTCUSDT", env="SYMBOL")
    
    # === Janela de Análise ===
    window_size_minutes: int = Field(1, ge=1, env="WINDOW_SIZE_MINUTES")

    # === Parâmetros de Fluxo (Data Handler) ===
    vol_factor_exh: float = Field(2.0, ge=0, env="VOL_FACTOR_EXH")
    history_size: int = Field(50, ge=1, env="HISTORY_SIZE")
    delta_std_dev_factor: float = Field(2.5, ge=0, env="DELTA_STD_DEV_FACTOR")
    context_sma_period: int = Field(10, ge=1, env="CONTEXT_SMA_PERIOD")

    # === Orderbook Params ===
    liquidity_flow_alert_percentage: float = Field(0.4, env="LIQUIDITY_FLOW_ALERT_PERCENTAGE")
    min_ob_abs_change_usd: float = Field(750_000.0, env="MIN_OB_ABS_CHANGE_USD")
    wall_std_dev_factor: float = Field(3.0, env="WALL_STD_DEV_FACTOR")
    orderbook_critical_imbalance: float = Field(0.95, env="ORDERBOOK_CRITICAL_IMBALANCE")
    orderbook_min_dominant_usd: float = Field(2_000_000.0, env="ORDERBOOK_MIN_DOMINANT_USD")
    orderbook_min_ratio_dom: float = Field(20.0, env="ORDERBOOK_MIN_RATIO_DOM")
    
    # Orderbook Performance & Resilience
    orderbook_request_timeout: float = Field(10.0, env="ORDERBOOK_REQUEST_TIMEOUT")
    orderbook_retry_delay: float = Field(2.0, env="ORDERBOOK_RETRY_DELAY")
    orderbook_max_retries: int = Field(3, env="ORDERBOOK_MAX_RETRIES")
    orderbook_max_requests_per_min: int = Field(10, env="ORDERBOOK_MAX_REQUESTS_PER_MIN")
    orderbook_rate_limit_buffer: float = Field(1.0, env="ORDERBOOK_RATE_LIMIT_BUFFER")
    orderbook_cache_ttl: float = Field(15.0, env="ORDERBOOK_CACHE_TTL")
    orderbook_max_stale: float = Field(60.0, env="ORDERBOOK_MAX_STALE")
    orderbook_min_depth_usd: float = Field(5_000.0, env="ORDERBOOK_MIN_DEPTH_USD")
    orderbook_allow_partial: bool = Field(False, env="ORDERBOOK_ALLOW_PARTIAL")
    orderbook_min_levels: int = Field(10, env="ORDERBOOK_MIN_LEVELS")
    orderbook_use_fallback: bool = Field(True, env="ORDERBOOK_USE_FALLBACK")
    orderbook_fallback_max_age: int = Field(300, env="ORDERBOOK_FALLBACK_MAX_AGE")
    orderbook_emergency_mode: bool = Field(True, env="ORDERBOOK_EMERGENCY_MODE")
    orderbook_max_age_ms: int = Field(30000, env="ORDERBOOK_MAX_AGE_MS")
    orderbook_require_timestamp: bool = Field(True, env="ORDERBOOK_REQUIRE_TIMESTAMP")
    orderbook_validate_sequence: bool = Field(True, env="ORDERBOOK_VALIDATE_SEQUENCE")

    # === WebSocket ===
    ws_ping_interval: int = Field(20, ge=5, env="WS_PING_INTERVAL")
    ws_ping_timeout: int = Field(10, ge=1, env="WS_PING_TIMEOUT")
    ws_reconnect_delay: float = Field(3.0, env="WS_RECONNECT_DELAY")
    ws_max_reconnect_attempts: int = Field(15, env="WS_MAX_RECONNECT_ATTEMPTS")
    ws_initial_delay: float = Field(1.0, env="WS_INITIAL_DELAY")
    ws_max_delay: float = Field(30.0, env="WS_MAX_DELAY")
    ws_backoff_factor: float = Field(1.5, env="WS_BACKOFF_FACTOR")

    # === Fluxo Contínuo ===
    cvd_reset_interval_hours: int = Field(4, env="CVD_RESET_INTERVAL_HOURS")
    whale_trade_threshold: float = Field(5.0, env="WHALE_TRADE_THRESHOLD")
    order_size_buckets: Dict[str, Tuple[float, float]] = Field(
        default={
            "retail": (0, 0.5),
            "mid": (0.5, 5.0),
            "whale": (5.0, 9999.0)
        }
    )
    burst_volume_threshold: float = Field(5.0, env="BURST_VOLUME_THRESHOLD")
    flow_prefer_window_metrics: bool = Field(True, env="FLOW_PREFER_WINDOW_METRICS")
    flow_log_window_period: bool = Field(True, env="FLOW_LOG_WINDOW_PERIOD")
    flow_validate_consistency: bool = Field(True, env="FLOW_VALIDATE_CONSISTENCY")

    # === Context Collector ===
    context_timeframes: List[str] = Field(["15m", "1h", "4h"], env="CONTEXT_TIMEFRAMES")
    context_ema_period: int = Field(21, env="CONTEXT_EMA_PERIOD")
    context_atr_period: int = Field(14, env="CONTEXT_ATR_PERIOD")
    intermarket_symbols: List[str] = Field(["ETHUSDT"], env="INTERMARKET_SYMBOLS")
    external_markets: Dict[str, str] = Field(
        default={
            "GOLD": "GC=F",
            "SP500": "^GSPC",
            "NASDAQ": "^IXIC",
            "US10Y": "^TNX",
            "OIL": "CL=F"
        }
    )
    derivatives_symbols: List[str] = Field(["BTCUSDT"], env="DERIVATIVES_SYMBOLS") # Default dinâmico tratado no alias se necessário
    liquidation_map_depth: int = Field(100, env="LIQUIDATION_MAP_DEPTH")
    vp_num_days_history: int = Field(1, env="VP_NUM_DAYS_HISTORY")
    vp_value_area_percent: float = Field(0.70, env="VP_VALUE_AREA_PERCENT")
    vp_advanced: bool = Field(True, env="VP_ADVANCED")
    context_update_interval_seconds: int = Field(60, env="CONTEXT_UPDATE_INTERVAL_SECONDS")

    # === Machine Learning ===
    ml_lookback_windows: List[int] = Field([5, 15, 60], env="ML_LOOKBACK_WINDOWS")
    ml_min_sample_size: int = Field(100, env="ML_MIN_SAMPLE_SIZE")
    ml_update_interval: int = Field(600, env="ML_UPDATE_INTERVAL")

    # === Sentimento / On-Chain ===
    enable_onchain: bool = Field(True, env="ENABLE_ONCHAIN")
    onchain_providers: List[str] = Field(["glassnode", "cryptoquant"], env="ONCHAIN_PROVIDERS")
    stablecoin_flow_tracking: bool = Field(True, env="STABLECOIN_FLOW_TRACKING")

    # === Qualidade de Sinal ===
    min_signal_volume_btc: float = Field(1.0, env="MIN_SIGNAL_VOLUME_BTC")
    min_signal_tps: float = Field(2.0, env="MIN_SIGNAL_TPS")
    min_abs_delta_btc: float = Field(0.5, env="MIN_ABS_DELTA_BTC")
    min_reversal_ratio: float = Field(0.2, env="MIN_REVERSAL_RATIO")
    index_atr_floor_pct: float = Field(0.001, env="INDEX_ATR_FLOOR_PCT")

    # === Validação e Segurança ===
    max_trade_volume_btc: float = Field(100.0, env="MAX_TRADE_VOLUME_BTC")
    min_trade_volume_btc: float = Field(0.01, env="MIN_TRADE_VOLUME_BTC")
    max_price_deviation_pct: float = Field(0.01, env="MAX_PRICE_DEVIATION_PCT")
    health_check_interval: int = Field(30, env="HEALTH_CHECK_INTERVAL")
    fallback_delta_threshold: float = Field(1.0, env="FALLBACK_DELTA_THRESHOLD")
    fallback_volume_threshold: float = Field(5.0, env="FALLBACK_VOLUME_THRESHOLD")
    max_missing_fields_ratio: float = Field(0.1, env="MAX_MISSING_FIELDS_RATIO")
    trade_validation_window: int = Field(60, env="TRADE_VALIDATION_WINDOW")
    
    # Logs
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_to_file: bool = Field(True, env="LOG_TO_FILE")
    log_file_max_size: int = Field(10485760, env="LOG_FILE_MAX_SIZE") # 10MB
    log_file_backup_count: int = Field(5, env="LOG_FILE_BACKUP_COUNT")
    missing_field_log_step: Optional[int] = Field(None, env="MISSING_FIELD_LOG_STEP")
    invalid_json_log_step: int = Field(100, env="INVALID_JSON_LOG_STEP")
    invalid_trade_log_step: int = Field(100, env="INVALID_TRADE_LOG_STEP")

    # === Performance Pipeline ===
    ai_test_min_chars: int = Field(10, env="AI_TEST_MIN_CHARS")
    max_pipeline_cache_size: int = Field(100, env="MAX_PIPELINE_CACHE_SIZE")
    pipeline_timeout_seconds: int = Field(10, env="PIPELINE_TIMEOUT_SECONDS")
    max_concurrent_analyses: int = Field(5, env="MAX_CONCURRENT_ANALYSES")
    ai_min_interval_sec: float = Field(60.0, env="AI_MIN_INTERVAL_SEC")

    # === Contexto de Mercado / Regime ===
    correlation_lookback: int = Field(50, env="CORRELATION_LOOKBACK")
    volatility_percentiles: Tuple[float, float] = Field((0.35, 0.65), env="VOLATILITY_PERCENTILES")
    adx_period: int = Field(14, env="ADX_PERIOD")
    rsi_periods: Dict[str, int] = Field(default={"short": 14, "long": 21})
    macd_fast_period: int = Field(12, env="MACD_FAST_PERIOD")
    macd_slow_period: int = Field(26, env="MACD_SLOW_PERIOD")
    macd_signal_period: int = Field(9, env="MACD_SIGNAL_PERIOD")
    
    # === Suporte e Resistência ===
    sr_volume_weight: float = Field(0.6, env="SR_VOLUME_WEIGHT")
    sr_orderbook_weight: float = Field(0.4, env="SR_ORDERBOOK_WEIGHT")
    pivot_timeframes: List[str] = Field(["daily", "weekly", "monthly"], env="PIVOT_TIMEFRAMES")

    # === Profundidade e Spread ===
    order_book_depth_levels: List[int] = Field([1, 5, 10, 25], env="ORDER_BOOK_DEPTH_LEVELS")
    spread_tight_threshold_bps: float = Field(0.2, env="SPREAD_TIGHT_THRESHOLD_BPS")
    spread_avg_windows_min: List[int] = Field([60, 1440], env="SPREAD_AVG_WINDOWS_MIN")

    # === Fluxo de Ordens ===
    net_flow_windows_min: List[int] = Field([1, 5, 15], env="NET_FLOW_WINDOWS_MIN")
    aggressive_order_size_threshold: float = Field(0.0, env="AGGRESSIVE_ORDER_SIZE_THRESHOLD")

    # === Absorção ===
    absorcao_delta_eps: float = Field(1.0, env="ABSORCAO_DELTA_EPS")
    # Nota: Lógica complexa de 'ABSORCAO_GUARD_MODE' movida para alias com base no Environment
    
    # === Whales & Patterns ===
    whale_detection_window_min: int = Field(60, env="WHALE_DETECTION_WINDOW_MIN")
    iceberg_threshold_count: int = Field(3, env="ICEBERG_THRESHOLD_COUNT")
    pattern_lookback_bars: int = Field(200, env="PATTERN_LOOKBACK_BARS")
    fibonacci_levels: List[float] = Field([0.236, 0.382, 0.5, 0.618, 0.786], env="FIBONACCI_LEVELS")

    # === Market Impact ===
    slippage_buckets_usd: List[int] = Field([1_000, 10_000, 100_000, 1_000_000], env="SLIPPAGE_BUCKETS_USD")
    liquidity_weight_depth: float = Field(0.4, env="LIQUIDITY_WEIGHT_DEPTH")
    liquidity_weight_spread: float = Field(0.3, env="LIQUIDITY_WEIGHT_SPREAD")
    liquidity_weight_volume: float = Field(0.3, env="LIQUIDITY_WEIGHT_VOLUME")

    # === Detecção de Regime ===
    regime_change_threshold: float = Field(0.15, env="REGIME_CHANGE_THRESHOLD")
    regime_expected_duration_hrs: Tuple[int, int] = Field((2, 4), env="REGIME_EXPECTED_DURATION_HRS")

    # === Alertas ===
    alert_support_prob_threshold: float = Field(0.75, env="ALERT_SUPPORT_PROB_THRESHOLD")
    alert_volume_spike_threshold: float = Field(3.0, env="ALERT_VOLUME_SPIKE_THRESHOLD")
    price_target_horizons_min: List[int] = Field([5, 15, 60], env="PRICE_TARGET_HORIZONS_MIN")
    price_target_confidence_levels: Dict[str, float] = Field(default={"low": 0.5, "high": 0.7})
    alert_cooldown_sec: int = Field(30, env="ALERT_COOLDOWN_SEC")
    dedup_filter_window: float = Field(1.0, env="DEDUP_FILTER_WINDOW")

    # === Liquidity Heatmap ===
    lhm_window_size: int = Field(2000, env="LHM_WINDOW_SIZE")
    lhm_cluster_threshold_pct: float = Field(0.003, env="LHM_CLUSTER_THRESHOLD_PCT")
    lhm_min_trades_per_cluster: int = Field(5, env="LHM_MIN_TRADES_PER_CLUSTER")
    lhm_update_interval_ms: int = Field(100, env="LHM_UPDATE_INTERVAL_MS")

    # === Burst Detection ===
    burst_window_ms: int = Field(200, env="BURST_WINDOW_MS")
    burst_cooldown_ms: int = Field(200, env="BURST_COOLDOWN_MS")

    # === Qualidade de Dados v2.1 ===
    validate_data_consistency: bool = Field(True, env="VALIDATE_DATA_CONSISTENCY")
    log_discrepancies: bool = Field(True, env="LOG_DISCREPANCIES")
    auto_correct_discrepancies: bool = Field(True, env="AUTO_CORRECT_DISCREPANCIES")
    volume_tolerance_btc: float = Field(0.001, env="VOLUME_TOLERANCE_BTC")
    delta_tolerance_btc: float = Field(0.01, env="DELTA_TOLERANCE_BTC")
    price_tolerance_pct: float = Field(0.001, env="PRICE_TOLERANCE_PCT")
    enable_quality_stats: bool = Field(True, env="ENABLE_QUALITY_STATS")
    log_stats_interval_sec: int = Field(300, env="LOG_STATS_INTERVAL_SEC")

    # === Reconexão e Warmup ===
    warmup_windows: int = Field(3, ge=1, env="WARMUP_WINDOWS")
    trades_buffer_size: int = Field(2000, ge=1, env="TRADES_BUFFER_SIZE")
    min_trades_for_pipeline: int = Field(10, ge=5, env="MIN_TRADES_FOR_PIPELINE")
    health_check_timeout: int = Field(90, env="HEALTH_CHECK_TIMEOUT")
    health_check_critical: int = Field(180, env="HEALTH_CHECK_CRITICAL")
    rest_timeout: int = Field(5, env="REST_TIMEOUT")
    rest_max_retries: int = Field(3, env="REST_MAX_RETRIES")
    rest_retry_delay: float = Field(1.0, env="REST_RETRY_DELAY")
    pipeline_min_absolute_trades: int = Field(3, env="PIPELINE_MIN_ABSOLUTE_TRADES")
    pipeline_allow_limited_data: bool = Field(True, env="PIPELINE_ALLOW_LIMITED_DATA")
    cleanup_timeout: float = Field(5.0, env="CLEANUP_TIMEOUT")
    feature_store_max_size: int = Field(1000, env="FEATURE_STORE_MAX_SIZE")
    use_numpy_vectorization: bool = Field(True, env="USE_NUMPY_VECTORIZATION")
    max_worker_threads: int = Field(5, env="MAX_WORKER_THREADS")

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

# ==============================================================================
# INSTANCIAÇÃO E VALIDAÇÃO AUTOMÁTICA
# ==============================================================================
# Se faltar variável obrigatória, o programa quebra aqui com mensagem clara
settings = Settings()

# ==============================================================================
# ALIASES DE COMPATIBILIDADE (API PÚBLICA LEGADA)
# ==============================================================================
# Mantém compatibilidade com módulos que fazem "import config" e usam variáveis globais.

# -- Ativo e Conexão --
SYMBOL = settings.symbol
STREAM_URL = f"wss://fstream.binance.com/ws/{settings.symbol.lower()}@aggTrade"

# -- IA --
GROQ_API_KEY = settings.groq_api_key
GROQ_MODEL = settings.groq_model
DASHSCOPE_API_KEY = settings.dashscope_api_key
OPENAI_API_KEY = settings.openai_api_key

AI_KEYS = {
    "groq": GROQ_API_KEY,
    "dashscope": DASHSCOPE_API_KEY,
}

# -- Trading Params --
WINDOW_SIZE_MINUTES = settings.window_size_minutes
VOL_FACTOR_EXH = settings.vol_factor_exh
HISTORY_SIZE = settings.history_size
DELTA_STD_DEV_FACTOR = settings.delta_std_dev_factor
CONTEXT_SMA_PERIOD = settings.context_sma_period

# -- Orderbook --
LIQUIDITY_FLOW_ALERT_PERCENTAGE = settings.liquidity_flow_alert_percentage
MIN_OB_ABS_CHANGE_USD = settings.min_ob_abs_change_usd
WALL_STD_DEV_FACTOR = settings.wall_std_dev_factor
ORDERBOOK_CRITICAL_IMBALANCE = settings.orderbook_critical_imbalance
ORDERBOOK_MIN_DOMINANT_USD = settings.orderbook_min_dominant_usd
ORDERBOOK_MIN_RATIO_DOM = settings.orderbook_min_ratio_dom
ORDERBOOK_REQUEST_TIMEOUT = settings.orderbook_request_timeout
ORDERBOOK_RETRY_DELAY = settings.orderbook_retry_delay
ORDERBOOK_MAX_RETRIES = settings.orderbook_max_retries
ORDERBOOK_MAX_REQUESTS_PER_MIN = settings.orderbook_max_requests_per_min
ORDERBOOK_RATE_LIMIT_BUFFER = settings.orderbook_rate_limit_buffer
ORDERBOOK_CACHE_TTL = settings.orderbook_cache_ttl
ORDERBOOK_MAX_STALE = settings.orderbook_max_stale
ORDERBOOK_MIN_DEPTH_USD = settings.orderbook_min_depth_usd
ORDERBOOK_ALLOW_PARTIAL = settings.orderbook_allow_partial
ORDERBOOK_MIN_LEVELS = settings.orderbook_min_levels
ORDERBOOK_USE_FALLBACK = settings.orderbook_use_fallback
ORDERBOOK_FALLBACK_MAX_AGE = settings.orderbook_fallback_max_age
ORDERBOOK_EMERGENCY_MODE = settings.orderbook_emergency_mode
ORDERBOOK_MAX_AGE_MS = settings.orderbook_max_age_ms
ORDERBOOK_REQUIRE_TIMESTAMP = settings.orderbook_require_timestamp
ORDERBOOK_VALIDATE_SEQUENCE = settings.orderbook_validate_sequence

# -- WebSocket --
WS_PING_INTERVAL = settings.ws_ping_interval
WS_PING_TIMEOUT = settings.ws_ping_timeout
WS_RECONNECT_DELAY = settings.ws_reconnect_delay
WS_MAX_RECONNECT_ATTEMPTS = settings.ws_max_reconnect_attempts
WS_INITIAL_DELAY = settings.ws_initial_delay
WS_MAX_DELAY = settings.ws_max_delay
WS_BACKOFF_FACTOR = settings.ws_backoff_factor

# -- Flow / Continuous --
CVD_RESET_INTERVAL_HOURS = settings.cvd_reset_interval_hours
WHALE_TRADE_THRESHOLD = settings.whale_trade_threshold
ORDER_SIZE_BUCKETS = settings.order_size_buckets
BURST_VOLUME_THRESHOLD = settings.burst_volume_threshold
FLOW_PREFER_WINDOW_METRICS = settings.flow_prefer_window_metrics
FLOW_LOG_WINDOW_PERIOD = settings.flow_log_window_period
FLOW_VALIDATE_CONSISTENCY = settings.flow_validate_consistency

# -- Context Collector --
CONTEXT_TIMEFRAMES = settings.context_timeframes
CONTEXT_EMA_PERIOD = settings.context_ema_period
CONTEXT_ATR_PERIOD = settings.context_atr_period
INTERMARKET_SYMBOLS = settings.intermarket_symbols
EXTERNAL_MARKETS = settings.external_markets
DERIVATIVES_SYMBOLS = settings.derivatives_symbols
if not DERIVATIVES_SYMBOLS:
    DERIVATIVES_SYMBOLS = [SYMBOL]

LIQUIDATION_MAP_DEPTH = settings.liquidation_map_depth
VP_NUM_DAYS_HISTORY = settings.vp_num_days_history
VP_VALUE_AREA_PERCENT = settings.vp_value_area_percent
VP_ADVANCED = settings.vp_advanced
CONTEXT_UPDATE_INTERVAL_SECONDS = settings.context_update_interval_seconds

# -- ML --
ML_LOOKBACK_WINDOWS = settings.ml_lookback_windows
ML_MIN_SAMPLE_SIZE = settings.ml_min_sample_size
ML_UPDATE_INTERVAL = settings.ml_update_interval

# -- OnChain --
ENABLE_ONCHAIN = settings.enable_onchain
ONCHAIN_PROVIDERS = settings.onchain_providers
STABLECOIN_FLOW_TRACKING = settings.stablecoin_flow_tracking

# -- Signal Quality --
MIN_SIGNAL_VOLUME_BTC = settings.min_signal_volume_btc
MIN_SIGNAL_TPS = settings.min_signal_tps
MIN_ABS_DELTA_BTC = settings.min_abs_delta_btc
MIN_REVERSAL_RATIO = settings.min_reversal_ratio
INDEX_ATR_FLOOR_PCT = settings.index_atr_floor_pct

# -- Validation --
MAX_TRADE_VOLUME_BTC = settings.max_trade_volume_btc
MIN_TRADE_VOLUME_BTC = settings.min_trade_volume_btc
MAX_PRICE_DEVIATION_PCT = settings.max_price_deviation_pct
HEALTH_CHECK_INTERVAL = settings.health_check_interval
FALLBACK_DELTA_THRESHOLD = settings.fallback_delta_threshold
FALLBACK_VOLUME_THRESHOLD = settings.fallback_volume_threshold
MAX_MISSING_FIELDS_RATIO = settings.max_missing_fields_ratio
TRADE_VALIDATION_WINDOW = settings.trade_validation_window
LOG_LEVEL = settings.log_level
LOG_TO_FILE = settings.log_to_file
LOG_FILE_MAX_SIZE = settings.log_file_max_size
LOG_FILE_BACKUP_COUNT = settings.log_file_backup_count
MISSING_FIELD_LOG_STEP = settings.missing_field_log_step
INVALID_JSON_LOG_STEP = settings.invalid_json_log_step
INVALID_TRADE_LOG_STEP = settings.invalid_trade_log_step

# -- Performance --
AI_TEST_MIN_CHARS = settings.ai_test_min_chars
MAX_PIPELINE_CACHE_SIZE = settings.max_pipeline_cache_size
PIPELINE_TIMEOUT_SECONDS = settings.pipeline_timeout_seconds
MAX_CONCURRENT_ANALYSES = settings.max_concurrent_analyses
AI_MIN_INTERVAL_SEC = settings.ai_min_interval_sec

# -- Market Context --
CORRELATION_LOOKBACK = settings.correlation_lookback
VOLATILITY_PERCENTILES = settings.volatility_percentiles
ADX_PERIOD = settings.adx_period
RSI_PERIODS = settings.rsi_periods
MACD_FAST_PERIOD = settings.macd_fast_period
MACD_SLOW_PERIOD = settings.macd_slow_period
MACD_SIGNAL_PERIOD = settings.macd_signal_period

# -- S/R --
SR_VOLUME_WEIGHT = settings.sr_volume_weight
SR_ORDERBOOK_WEIGHT = settings.sr_orderbook_weight
PIVOT_TIMEFRAMES = settings.pivot_timeframes

# -- Depth/Spread --
ORDER_BOOK_DEPTH_LEVELS = settings.order_book_depth_levels
SPREAD_TIGHT_THRESHOLD_BPS = settings.spread_tight_threshold_bps
SPREAD_AVG_WINDOWS_MIN = settings.spread_avg_windows_min

# -- Net Flow --
NET_FLOW_WINDOWS_MIN = settings.net_flow_windows_min
AGGRESSIVE_ORDER_SIZE_THRESHOLD = settings.aggressive_order_size_threshold

# -- Absorção (Lógica Condicional) --
ABSORCAO_DELTA_EPS = settings.absorcao_delta_eps
ABSORCAO_GUARD_MODE = "raise" if settings.environment.lower() == "dev" else "warn"

# -- Whale/Patterns --
WHALE_DETECTION_WINDOW_MIN = settings.whale_detection_window_min
ICEBERG_THRESHOLD_COUNT = settings.iceberg_threshold_count
PATTERN_LOOKBACK_BARS = settings.pattern_lookback_bars
FIBONACCI_LEVELS = settings.fibonacci_levels

# -- Impact --
SLIPPAGE_BUCKETS_USD = settings.slippage_buckets_usd
LIQUIDITY_WEIGHT_DEPTH = settings.liquidity_weight_depth
LIQUIDITY_WEIGHT_SPREAD = settings.liquidity_weight_spread
LIQUIDITY_WEIGHT_VOLUME = settings.liquidity_weight_volume

# -- Regime --
REGIME_CHANGE_THRESHOLD = settings.regime_change_threshold
REGIME_EXPECTED_DURATION_HRS = settings.regime_expected_duration_hrs

# -- Alerts --
ALERT_SUPPORT_PROB_THRESHOLD = settings.alert_support_prob_threshold
ALERT_VOLUME_SPIKE_THRESHOLD = settings.alert_volume_spike_threshold
PRICE_TARGET_HORIZONS_MIN = settings.price_target_horizons_min
PRICE_TARGET_CONFIDENCE_LEVELS = settings.price_target_confidence_levels
ALERT_COOLDOWN_SEC = settings.alert_cooldown_sec
DEDUP_FILTER_WINDOW = settings.dedup_filter_window

# -- LHM --
LHM_WINDOW_SIZE = settings.lhm_window_size
LHM_CLUSTER_THRESHOLD_PCT = settings.lhm_cluster_threshold_pct
LHM_MIN_TRADES_PER_CLUSTER = settings.lhm_min_trades_per_cluster
LHM_UPDATE_INTERVAL_MS = settings.lhm_update_interval_ms

# -- Burst --
BURST_WINDOW_MS = settings.burst_window_ms
BURST_COOLDOWN_MS = settings.burst_cooldown_ms

# -- Data Quality --
VALIDATE_DATA_CONSISTENCY = settings.validate_data_consistency
LOG_DISCREPANCIES = settings.log_discrepancies
AUTO_CORRECT_DISCREPANCIES = settings.auto_correct_discrepancies
VOLUME_TOLERANCE_BTC = settings.volume_tolerance_btc
DELTA_TOLERANCE_BTC = settings.delta_tolerance_btc
PRICE_TOLERANCE_PCT = settings.price_tolerance_pct
ENABLE_QUALITY_STATS = settings.enable_quality_stats
LOG_STATS_INTERVAL_SEC = settings.log_stats_interval_sec

# -- Reconnect/Warmup --
WARMUP_WINDOWS = settings.warmup_windows
TRADES_BUFFER_SIZE = settings.trades_buffer_size
MIN_TRADES_FOR_PIPELINE = settings.min_trades_for_pipeline
HEALTH_CHECK_TIMEOUT = settings.health_check_timeout
HEALTH_CHECK_CRITICAL = settings.health_check_critical
REST_TIMEOUT = settings.rest_timeout
REST_MAX_RETRIES = settings.rest_max_retries
REST_RETRY_DELAY = settings.rest_retry_delay
PIPELINE_MIN_ABSOLUTE_TRADES = settings.pipeline_min_absolute_trades
PIPELINE_ALLOW_LIMITED_DATA = settings.pipeline_allow_limited_data
CLEANUP_TIMEOUT = settings.cleanup_timeout
FEATURE_STORE_MAX_SIZE = settings.feature_store_max_size
USE_NUMPY_VECTORIZATION = settings.use_numpy_vectorization
MAX_WORKER_THREADS = settings.max_worker_threads

# ==============================================================================
# FUNÇÃO DE VALIDAÇÃO LEGADA (MANTIDA MAS INATIVA)
# ==============================================================================
def validate_config():
    """
    Função legada. A validação real agora é feita na inicialização da classe Settings.
    Mantida apenas para evitar que chamadas antigas quebrem.
    """
    print("✅ Configuração validada automaticamente via Pydantic Settings.")
    return True