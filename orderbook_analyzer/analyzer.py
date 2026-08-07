# orderbook_analyzer/analyzer.py — SHIM DE COMPATIBILIDADE
import warnings
warnings.warn(
    "orderbook_analyzer.analyzer está deprecated; "
    "use orderbook_analyzer.legacy_simplified.SimplifiedOrderBookAnalyzer",
    DeprecationWarning,
    stacklevel=2
)
from orderbook_analyzer.legacy_simplified import (
    OrderBookAnalyzer,
    SimplifiedOrderBookAnalyzer,
    OrderBookConfig,
)
__all__ = ["OrderBookAnalyzer", "SimplifiedOrderBookAnalyzer", "OrderBookConfig"]
