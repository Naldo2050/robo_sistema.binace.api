# market_analysis/__init__.py
"""
Pacote de análise de mercado.

Contém: correlações cross-asset, volume profile dinâmico/histórico,
liquidez, padrões, níveis, regime detector.

Nota: cross_asset_correlations não é re-exportado aqui para evitar
circular import. Acesse via market_analysis.cross_asset_correlations.
"""

from .dynamic_volume_profile import DynamicVolumeProfile  # noqa: F401
from .historical_profiler import HistoricalVolumeProfiler  # noqa: F401
from .levels_registry import LevelRegistry  # noqa: F401
from .liquidity_heatmap import LiquidityHeatmap  # noqa: F401
from .pattern_recognition import detect_candlestick_patterns, recognize_patterns  # noqa: F401
from .regime_detector import EnhancedRegimeDetector, RegimeAnalysis  # noqa: F401

__all__ = [
    "DynamicVolumeProfile",
    "EnhancedRegimeDetector",
    "HistoricalVolumeProfiler",
    "LevelRegistry",
    "LiquidityHeatmap",
    "RegimeAnalysis",
    "detect_candlestick_patterns",
    "recognize_patterns",
]
