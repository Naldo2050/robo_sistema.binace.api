# flow_analyzer.py (na raiz - wrapper de compatibilidade)
"""
Wrapper de compatibilidade para flow_analyzer.

O código foi modularizado em flow_analyzer/

Uso recomendado:
    from flow_analyzer import FlowAnalyzer

Este arquivo mantém compatibilidade com imports existentes.
"""

import warnings as _warnings

# Aviso apenas uma vez por sessão
_warnings.filterwarnings('once', category=DeprecationWarning, module=__name__)

_warnings.warn(
    "Importar de 'flow_analyzer.py' está deprecated. "
    "O módulo foi movido para 'flow_analyzer/' (pasta). "
    "Os imports continuam funcionando normalmente.",
    DeprecationWarning,
    stacklevel=2
)

# Re-exporta tudo do novo módulo
from flow_analyzer.core import FlowAnalyzer
from flow_analyzer.errors import FlowAnalyzerError
from flow_analyzer.aggregates import RollingAggregate
from flow_analyzer.metrics import PerformanceMonitor, CircuitBreaker
from flow_analyzer.absorption import (
    AbsorptionClassifier,
    classify_absorption,
    classify_absorption_simple,
)
from flow_analyzer.validation import (
    TradeSchema,
    validate_ohlc,
    guard_absorcao,
)
from flow_analyzer.utils import (
    LazyLog,
    lazy_log,
    to_decimal,
    decimal_round,
)

# Mantém __all__ para compatibilidade
__all__ = [
    "FlowAnalyzer",
    "FlowAnalyzerError",
    "RollingAggregate",
    "PerformanceMonitor",
    "CircuitBreaker",
    "AbsorptionClassifier",
    "classify_absorption",
    "classify_absorption_simple",
    "TradeSchema",
    "validate_ohlc",
    "guard_absorcao",
    "LazyLog",
    "lazy_log",
    "to_decimal",
    "decimal_round",
]