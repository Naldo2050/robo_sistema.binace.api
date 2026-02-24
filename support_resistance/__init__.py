"""
SISTEMA INSTITUCIONAL DE SUPORTE E RESISTÊNCIA
==============================================

Sistema modular para análise quantitativa de níveis de suporte e resistência
com métricas institucionais, validação estatística e monitoramento em tempo real.
"""

from .constants import *
from .config import *
from .validation import *
from .utils import StatisticalUtils, StructuredLogger, timer
from .pivot_points import InstitutionalPivotPoints
from .volume_profile import VolumeProfileAnalyzer
from .core import AdvancedSupportResistance
from .monitor import InstitutionalMarketMonitor, HealthCheckResult
from .system import InstitutionalSupportResistanceSystem
from .reference_prices import ReferencePrices

def daily_pivot(df):
    """Calculate daily pivot points from OHLC DataFrame"""
    if df.empty:
        return {}
    last = df.iloc[-1]
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
        'r3': float(r3), 's3': float(s3)
    }

def weekly_pivot(df):
    """Calculate weekly pivot points from OHLC DataFrame"""
    return daily_pivot(df)  # Same calculation for simplicity

def monthly_pivot(df):
    """Calculate monthly pivot points from OHLC DataFrame"""
    return daily_pivot(df)  # Same calculation for simplicity

__version__ = "2.0.0"
__all__ = [
    'InstitutionalSupportResistanceSystem',
    'InstitutionalConfig',
    'SRConfig',
    'VolumeProfileConfig',
    'MonitorConfig',
    'PivotConfig',
    'AdvancedSupportResistance',
    'InstitutionalMarketMonitor',
    'HealthCheckResult',
    'VolumeProfileAnalyzer',
    'InstitutionalPivotPoints',
    'ReferencePrices',
    'StatisticalUtils',
    'StructuredLogger',
    'timer',
    'LevelType',
    'ReactionType',
    'ConfidenceLevel',
    'MarketBias',
    'QualityRating',
    'SRAnalysisError',
    'InsufficientDataError',
    'InvalidConfigurationError',
    'CalculationError',
    'CONSTANTS',
    'validate_dataframe',
    'validate_series',
    'validate_positive',
    'validate_range',
    'SafeJSONEncoder',
    'daily_pivot',
    'weekly_pivot',
    'monthly_pivot'
]