# data_pipeline/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

try:
    import config as external_config
except ImportError:
    external_config = None


@dataclass
class PipelineConfig:
    # Validação
    min_trades_pipeline: int = 10
    min_absolute_trades: int = 3
    allow_limited_data: bool = True
    max_price_variance_pct: float = 10.0

    # Adaptação
    enable_adaptive_thresholds: bool = True
    adaptive_learning_rate: float = 0.1
    adaptive_confidence: float = 0.7

    # Cache
    cache_ttl_seconds: int = 3600
    cache_max_items: int = 1000
    cache_allow_expired: bool = True

    # Performance
    enable_vectorized_validation: bool = True
    validation_chunk_size: int = 10000

    # Precisão por símbolo
    price_scales: Dict[str, int] = field(default_factory=lambda: {
        'BTCUSDT': 10,
        'ETHUSDT': 100,
        'BNBUSDT': 100,
        'SOLUSDT': 1000,
        'XRPUSDT': 10000,
        'DOGEUSDT': 100000,
        'ADAUSDT': 10000,
        'DEFAULT': 10
    })

    @classmethod
    def from_config_file(cls) -> "PipelineConfig":
        if external_config is None:
            return cls()

        return cls(
            min_trades_pipeline=getattr(external_config, 'MIN_TRADES_FOR_PIPELINE', 10),
            min_absolute_trades=getattr(external_config, 'PIPELINE_MIN_ABSOLUTE_TRADES', 3),
            allow_limited_data=getattr(external_config, 'PIPELINE_ALLOW_LIMITED_DATA', True),
            enable_adaptive_thresholds=getattr(external_config, 'PIPELINE_ADAPTIVE_THRESHOLDS', True),
            adaptive_learning_rate=getattr(external_config, 'PIPELINE_ADAPTIVE_LEARNING_RATE', 0.1),
            enable_vectorized_validation=getattr(external_config, 'PIPELINE_VECTORIZED_VALIDATION', True),
            cache_allow_expired=getattr(external_config, 'PIPELINE_CACHE_ALLOW_EXPIRED', True),
        )

    def get_price_scale(self, symbol: str) -> int:
        return self.price_scales.get(symbol, self.price_scales['DEFAULT'])

    def get_price_precision(self, symbol: str) -> int:
        scale = self.get_price_scale(symbol)
        return len(str(scale)) - 1