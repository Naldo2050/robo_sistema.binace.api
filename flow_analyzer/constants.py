# flow_analyzer/constants.py
"""
Constantes e configurações do FlowAnalyzer.

Este módulo centraliza todas as constantes, magic numbers e valores
de configuração para facilitar manutenção e testes.
"""

from decimal import Decimal
from typing import Dict, Tuple

# ==============================================================================
# VERSÃO
# ==============================================================================
VERSION = "2.4.0"

# ==============================================================================
# CONFIGURAÇÕES PADRÃO (fallback se config.py não disponível)
# ==============================================================================
DEFAULT_NET_FLOW_WINDOWS_MIN = [1, 5, 15]
DEFAULT_ABSORCAO_DELTA_EPS = 1.0
DEFAULT_ABSORCAO_GUARD_MODE = "warn"
DEFAULT_FLOW_TRADES_MAXLEN = 100_000
DEFAULT_FLOW_LOG_PERF = False
DEFAULT_FLOW_LOG_DETAILED = False
DEFAULT_FLOW_TIME_BUDGET_MS = 500.0
DEFAULT_FLOW_CACHE_ENABLED = True
DEFAULT_WHALE_TRADE_THRESHOLD = 5.0
DEFAULT_CVD_RESET_INTERVAL_HOURS = 4

# ==============================================================================
# TIMESTAMPS E SINCRONIZAÇÃO
# ==============================================================================
TIMESTAMP_JITTER_TOLERANCE_MS = 2000
LATE_TRADE_THRESHOLD_MS = 1000  # Trade atrasado > 1s
MAX_LATE_TRADE_MS = 5000  # Trade muito atrasado > 5s

# ==============================================================================
# BURST DETECTION
# ==============================================================================
DEFAULT_BURST_WINDOW_MS = 200
DEFAULT_BURST_COOLDOWN_MS = 200
BURST_END_THRESHOLD_RATIO = 0.5  # Burst termina quando volume < 50% do threshold

# ==============================================================================
# LIQUIDITY HEATMAP
# ==============================================================================
DEFAULT_LHM_WINDOW_SIZE = 2000
DEFAULT_LHM_CLUSTER_THRESHOLD_PCT = 0.003
DEFAULT_LHM_MIN_TRADES_PER_CLUSTER = 5
DEFAULT_LHM_UPDATE_INTERVAL_MS = 100

# ==============================================================================
# ABSORÇÃO
# ==============================================================================
DEFAULT_ABSORCAO_ATR_MULTIPLIER = 0.5
DEFAULT_ABSORCAO_VOL_MULTIPLIER = 1.0
DEFAULT_ABSORCAO_MIN_PCT_TOLERANCE = 0.001
DEFAULT_ABSORCAO_MAX_PCT_TOLERANCE = 0.01
DEFAULT_ABSORCAO_FALLBACK_PCT_TOLERANCE = 0.002

# Thresholds para classificação de absorção
ABSORPTION_INTENSITY_THRESHOLD = 0.15
ABSORPTION_IMBALANCE_THRESHOLD = 0.15

# ==============================================================================
# PRECISÃO NUMÉRICA
# ==============================================================================
DECIMAL_PRECISION_BTC = 8
DECIMAL_PRECISION_USD = 2
DECIMAL_CENT = Decimal('0.01')
DECIMAL_ZERO = Decimal('0')
DECIMAL_TOLERANCE_BTC = Decimal('1e-6')
UI_TOLERANCE_USD = Decimal('0.02')

# ==============================================================================
# PERFORMANCE E OBSERVABILIDADE
# ==============================================================================
PERF_MONITOR_WINDOW_SIZE = 1000
LAZY_LOG_INTERVAL_MS = 1000
HIGH_LATENCY_P99_MS = 100.0
MEMORY_USAGE_WARNING_RATIO = 0.9

# ==============================================================================
# PARTICIPANT ANALYSIS
# ==============================================================================
DEFAULT_ORDER_SIZE_BUCKETS: Dict[str, Tuple[float, float]] = {
    "retail": (0.0, 0.5),
    "mid": (0.5, 2.0),
    "whale": (2.0, 9999.0),
}

# Pesos para composite score
PARTICIPANT_IMBALANCE_WEIGHT = 0.4
PARTICIPANT_PARTICIPATION_WEIGHT = 0.4
PARTICIPANT_FREQUENCY_WEIGHT = 0.2
MAX_TRADES_PER_SECOND = 10.0  # Para normalização de frequência

# ==============================================================================
# CIRCUIT BREAKER
# ==============================================================================
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RECOVERY_TIME_MS = 30_000

# ==============================================================================
# ERROR TRACKING
# ==============================================================================
MAX_ERROR_KEYS = 100  # Limite de tipos de erro rastreados

# ==============================================================================
# ROLLING AGGREGATES
# ==============================================================================
MAX_TRADES_PER_MINUTE_ESTIMATE = 600  # ~10 trades/segundo
MAX_AGGREGATE_TRADES = 10_000  # Limite absoluto por janela