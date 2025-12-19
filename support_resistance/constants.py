"""
Constantes, Enums e Tipos do Sistema de Suporte/Resistência
"""

import sys
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

# Compatibilidade Python 3.8+
if sys.version_info >= (3, 10):
    from typing import ParamSpec, TypeVar
else:
    from typing_extensions import ParamSpec, TypeVar

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

P = ParamSpec('P')
T = TypeVar('T')

# =============================
#  TYPES E ENUMS
# =============================

class LevelType(str, Enum):
    """Tipos de níveis de preço"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    
    def __str__(self) -> str:
        return self.value


class ReactionType(str, Enum):
    """Tipos de reação em teste de nível"""
    STRONG_DEFENSE = "STRONG_DEFENSE"
    DEFENSE = "DEFENSE"
    WEAK_DEFENSE = "WEAK_DEFENSE"
    NEUTRAL = "NEUTRAL"
    BREAKOUT = "BREAKOUT"


class ConfidenceLevel(str, Enum):
    """Níveis de confiança"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class MarketBias(str, Enum):
    """Viés do mercado"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    BALANCED = "balanced"
    NEUTRAL = "neutral"


class QualityRating(str, Enum):
    """Rating de qualidade"""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    POOR = "POOR"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# =============================
#  TYPED DICTS (apenas para anotação)
# =============================

class ConfidenceIntervalResult(TypedDict, total=False):
    """Resultado de cálculo de intervalo de confiança"""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    ci_width: float
    ci_width_pct: float
    stability_score: float
    sample_size: int
    bootstrap: Dict[str, float]


class ClusterQualityResult(TypedDict):
    """Resultado de avaliação de qualidade de cluster"""
    score: float
    quality: str
    dispersion_score: float
    concentration_score: float
    outlier_score: float
    outlier_count: int


class SRLevelResult(TypedDict, total=False):
    """Estrutura de um nível de suporte/resistência"""
    price: float
    mean: float
    std: float
    touches: int
    cluster_size: int
    volume_strength: float
    recency_score: float
    stability_score: float
    reaction_score: float
    composite_score: float
    confidence_interval: Dict[str, Any]
    cluster_quality: Dict[str, Any]
    type: str
    origin: str
    audit: Dict[str, Any]
    features: Dict[str, float]
    distance: float
    distance_percent: float


class SRAnalysisResult(TypedDict):
    """Resultado completo da análise de S/R"""
    support_levels: List[Dict[str, Any]]
    resistance_levels: List[Dict[str, Any]]
    defense_zones: Dict[str, Any]
    quality_report: Dict[str, Any]
    current_price: float
    lookback_period: int
    timestamp: str


# =============================
#  EXCEÇÕES ESPECÍFICAS
# =============================

class SRAnalysisError(Exception):
    """Erro base para análise de S/R"""
    pass


class InsufficientDataError(SRAnalysisError):
    """Dados insuficientes para análise"""
    pass


class InvalidConfigurationError(SRAnalysisError):
    """Configuração inválida"""
    pass


class CalculationError(SRAnalysisError):
    """Erro durante cálculo"""
    pass


# =============================
#  CONSTANTES
# =============================

from dataclasses import dataclass

@dataclass(frozen=True)
class AnalysisConstants:
    """Constantes centralizadas para análise - imutável"""
    
    # Limiares de reversão
    MIN_REVERSAL_PERCENT: float = 0.005  # 0.5%
    SIGNIFICANT_REVERSAL_PERCENT: float = 0.01  # 1%
    
    # Limiares de cluster
    MIN_CLUSTER_CV: float = 0.05  # Coeficiente de variação mínimo
    OUTLIER_IQR_MULTIPLIER: float = 1.5
    
    # Limiares de Z-score
    ZSCORE_LOW: float = -3.0
    ZSCORE_HIGH: float = 3.0
    
    # Limiares de volume
    EXTREME_VOLUME_RATIO: float = 3.0
    HIGH_VOLUME_RATIO: float = 2.0
    ELEVATED_VOLUME_RATIO: float = 1.5
    LOW_VOLUME_RATIO: float = 0.5
    
    # Limiares de qualidade
    QUALITY_EXCELLENT: float = 8.0
    QUALITY_GOOD: float = 6.0
    QUALITY_MODERATE: float = 4.0
    QUALITY_WEAK: float = 2.0
    
    # Janelas temporais
    DEFAULT_SMOOTHING_SPAN: int = 5
    DEFAULT_PEAK_DISTANCE: int = 5
    DEFAULT_FUTURE_WINDOW: int = 5
    MIN_BOOTSTRAP_SAMPLES: int = 2


CONSTANTS = AnalysisConstants()