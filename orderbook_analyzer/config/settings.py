# orderbook_analyzer/config/settings.py
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class OrderBookConfig:
    """Configuration for order book analysis."""
    symbol: str
    depth_levels: int = 10
    update_interval_ms: int = 100
    imbalance_threshold: float = 0.7
    volume_threshold: float = 1000.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.depth_levels <= 0:
            raise ValueError("depth_levels must be positive")
        if self.update_interval_ms <= 0:
            raise ValueError("update_interval_ms must be positive")
        if not 0 <= self.imbalance_threshold <= 1:
            raise ValueError("imbalance_threshold must be between 0 and 1")


# Default configurations for different symbols
DEFAULT_CONFIGS = {
    'BTCUSDT': OrderBookConfig(
        symbol='BTCUSDT',
        depth_levels=20,
        update_interval_ms=50,
        imbalance_threshold=0.8,
        volume_threshold=5000.0
    ),
    'ETHUSDT': OrderBookConfig(
        symbol='ETHUSDT',
        depth_levels=15,
        update_interval_ms=75,
        imbalance_threshold=0.75,
        volume_threshold=3000.0
    )
}


def get_config(symbol: str, **overrides) -> OrderBookConfig:
    """Get configuration for a symbol with optional overrides."""
    base_config = DEFAULT_CONFIGS.get(symbol, OrderBookConfig(symbol=symbol))
    
    # Apply overrides
    config_dict = base_config.__dict__.copy()
    config_dict.update(overrides)
    
    return OrderBookConfig(**config_dict)