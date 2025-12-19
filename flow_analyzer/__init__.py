# flow_analyzer/__init__.py
"""
FlowAnalyzer - Módulo de análise de fluxo institucional.

Inclui módulos de:
- Core: Processamento de trades
- Metrics: Performance e observabilidade
- Profiling: Memory e lock profiling
- Serialization: JSON seguro para Decimal
- Logging: Logging estruturado
"""

from .constants import VERSION

# Core
from .core import FlowAnalyzer

# Errors
from .errors import (
    FlowAnalyzerError,
    TradeValidationError,
    ConfigurationError,
    TimeBudgetExceededError,
    InvariantViolationError,
    CircuitBreakerOpenError,
    DataQualityError,
    AbsorptionGuardError,
)

# Protocols
from .protocols import (
    IFlowAnalyzer,
    ITimeProvider,
    IClockSync,
    ITradeProcessor,
    IMetricsCollector,
    ILiquidityHeatmap,
    IRollingAggregate,
    ICircuitBreaker,
)

# Aggregates
from .aggregates import RollingAggregate

# Metrics
from .metrics import (
    PerformanceMonitor,
    CircuitBreaker,
    CircuitState,
    HealthChecker,
    HealthStatus,
)

# Absorption
from .absorption import (
    AbsorptionClassifier,
    AbsorptionAnalyzer,
    AbsorptionAnalysis,
    AbsorptionConfig,
    classify_absorption,
    classify_absorption_simple,
)

# Validation
from .validation import (
    TradeSchema,
    validate_ohlc,
    fix_ohlc,
    guard_absorcao,
    FlowAnalyzerConfigValidator,
)

# Utils
from .utils import (
    LazyLog,
    lazy_log,
    to_decimal,
    decimal_round,
    quantize_usd,
    ui_safe_round_usd,
    ui_safe_round_btc,
    BoundedErrorCounter,
    get_current_time_ms,
    elapsed_ms,
    clamp,
    safe_divide,
)

# Serialization (NOVO)
from .serialization import (
    DecimalEncoder,
    MetricsSerializer,
    dumps,
    loads,
)

# Profiling (NOVO)
from .profiling import (
    MemoryProfiler,
    MemorySnapshot,
    LockProfiler,
    LockStats,
    PerformanceProfiler,
    run_benchmark,
)

# Logging (NOVO)
from .logging_config import (
    JSONFormatter,
    FlowAnalyzerLogger,
    StructuredLogger,
    setup_logging,
)

# Prometheus (NOVO - opcional)
try:
    from .prometheus_metrics import (
        PrometheusMetrics,
        MetricsCollector,
        HAS_PROMETHEUS,
    )
except ImportError:
    HAS_PROMETHEUS = False
    PrometheusMetrics = None
    MetricsCollector = None

# Constantes públicas
from .constants import (
    DEFAULT_NET_FLOW_WINDOWS_MIN,
    DEFAULT_WHALE_TRADE_THRESHOLD,
    DEFAULT_ABSORCAO_DELTA_EPS,
    DECIMAL_PRECISION_BTC,
    DECIMAL_PRECISION_USD,
)


__version__ = VERSION
__all__ = [
    # Version
    "__version__",
    "VERSION",
    
    # Core
    "FlowAnalyzer",
    
    # Errors
    "FlowAnalyzerError",
    "TradeValidationError",
    "ConfigurationError",
    "TimeBudgetExceededError",
    "InvariantViolationError",
    "CircuitBreakerOpenError",
    "DataQualityError",
    "AbsorptionGuardError",
    
    # Protocols
    "IFlowAnalyzer",
    "ITimeProvider",
    "IClockSync",
    "ITradeProcessor",
    "IMetricsCollector",
    "ILiquidityHeatmap",
    "IRollingAggregate",
    "ICircuitBreaker",
    
    # Aggregates
    "RollingAggregate",
    
    # Metrics
    "PerformanceMonitor",
    "CircuitBreaker",
    "CircuitState",
    "HealthChecker",
    "HealthStatus",
    
    # Absorption
    "AbsorptionClassifier",
    "AbsorptionAnalyzer",
    "AbsorptionAnalysis",
    "AbsorptionConfig",
    "classify_absorption",
    "classify_absorption_simple",
    
    # Validation
    "TradeSchema",
    "validate_ohlc",
    "fix_ohlc",
    "guard_absorcao",
    "FlowAnalyzerConfigValidator",
    
    # Utils
    "LazyLog",
    "lazy_log",
    "to_decimal",
    "decimal_round",
    "quantize_usd",
    "ui_safe_round_usd",
    "ui_safe_round_btc",
    "BoundedErrorCounter",
    "get_current_time_ms",
    "elapsed_ms",
    "clamp",
    "safe_divide",
    
    # Serialization
    "DecimalEncoder",
    "MetricsSerializer",
    "dumps",
    "loads",
    
    # Profiling
    "MemoryProfiler",
    "MemorySnapshot",
    "LockProfiler",
    "LockStats",
    "PerformanceProfiler",
    "run_benchmark",
    
    # Logging
    "JSONFormatter",
    "FlowAnalyzerLogger",
    "StructuredLogger",
    "setup_logging",
    
    # Prometheus (opcional)
    "PrometheusMetrics",
    "MetricsCollector",
    "HAS_PROMETHEUS",
    
    # Constants
    "DEFAULT_NET_FLOW_WINDOWS_MIN",
    "DEFAULT_WHALE_TRADE_THRESHOLD",
    "DEFAULT_ABSORCAO_DELTA_EPS",
    "DECIMAL_PRECISION_BTC",
    "DECIMAL_PRECISION_USD",
]