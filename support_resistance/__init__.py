"""
Pacote institucional de suporte e resistência.

Sistema modular para análise quantitativa de níveis de suporte e resistência
com métricas institucionais, validação estatística e monitoramento em tempo real.
"""

from .config import InstitutionalConfig, MonitorConfig, PivotConfig, SRConfig, VolumeProfileConfig  # noqa: F401
from .constants import (  # noqa: F401
    CONSTANTS,
    ConfidenceLevel,
    LevelType,
    MarketBias,
    QualityRating,
    ReactionType,
    SRAnalysisError,
)
from .core import AdvancedSupportResistance  # noqa: F401
from .defense_zones import DefenseZoneDetector  # noqa: F401
from .monitor import HealthCheckResult, InstitutionalMarketMonitor  # noqa: F401
from .pivot_points import InstitutionalPivotPoints  # noqa: F401
from .reference_prices import ReferencePrices  # noqa: F401
from .sr_strength import SRStrengthScorer  # noqa: F401
from .system import InstitutionalSupportResistanceSystem  # noqa: F401
from .utils import StatisticalUtils  # noqa: F401
from .validation import validate_dataframe, validate_series  # noqa: F401
from .volume_profile import VolumeProfileAnalyzer  # noqa: F401


def daily_pivot(df):
    """Calculate daily pivot points from OHLC DataFrame."""
    if df.empty:
        return {}
    if len(df) < 2:
        # Guard: dados insuficientes — iloc[-2] não existe (ex: primeira execução do dia)
        raise ValueError(
            "daily_pivot requer pelo menos 2 velas: a vela anterior completa "
            "(iloc[-2]) não está disponível com apenas %d vela(s)" % len(df)
        )
    # iloc[-2]: período anterior completo (não o atual em andamento)
    # Bug corrigido: iloc[-1] usava a vela do período atual (OHLC parcial)
    last = df.iloc[-2]
    high = last['high']
    low = last['low']
    close = last['close']
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return {
        'pivot': float(pivot),
        'r1': float(r1), 's1': float(s1),
        'r2': float(r2), 's2': float(s2),
        'r3': float(r3), 's3': float(s3),
        # OHLC do período anterior completo (iloc[-2]): usados diretamente
        # pelo institutional enricher para evitar drift de arredondamento na
        # reconstrução h=2p-s1, l=2p-r1 a partir do pivot já arredondado.
        'high': float(high), 'low': float(low), 'close': float(close),
    }


def weekly_pivot(df):
    """Calculate weekly pivot points from OHLC DataFrame."""
    return daily_pivot(df)


def monthly_pivot(df):
    """Calculate monthly pivot points from OHLC DataFrame."""
    return daily_pivot(df)


__version__ = "2.0.0"
__all__ = [
    "AdvancedSupportResistance",
    "CONSTANTS",
    "ConfidenceLevel",
    "DefenseZoneDetector",
    "HealthCheckResult",
    "InstitutionalConfig",
    "InstitutionalMarketMonitor",
    "MonitorConfig",
    "InstitutionalPivotPoints",
    "InstitutionalSupportResistanceSystem",
    "PivotConfig",
    "LevelType",
    "MarketBias",
    "QualityRating",
    "ReactionType",
    "ReferencePrices",
    "SRAnalysisError",
    "SRConfig",
    "SRStrengthScorer",
    "StatisticalUtils",
    "VolumeProfileAnalyzer",
    "VolumeProfileConfig",
    "daily_pivot",
    "validate_dataframe",
    "validate_series",
    "monthly_pivot",
    "weekly_pivot",
]
