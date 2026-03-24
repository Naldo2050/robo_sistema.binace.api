# institutional/base.py
"""
Tipos base e exceções para o arsenal institucional.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class InstitutionalError(Exception):
    """Erro base do módulo institucional."""
    pass


class InsufficientDataError(InstitutionalError):
    """Dados insuficientes para cálculo."""
    pass


class ExchangeConnectionError(InstitutionalError):
    """Erro de conexão com exchange."""
    pass


class InvalidParameterError(InstitutionalError):
    """Parâmetro inválido."""
    pass


class Side(str, Enum):
    """Lado da operação."""
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class MarketRegime(str, Enum):
    """Regime de mercado."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class SignalStrength(str, Enum):
    """Força do sinal."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


@dataclass
class Trade:
    """Representação de um trade."""
    timestamp: float
    price: float
    quantity: float
    side: Side
    trade_id: str = ""
    exchange: str = "binance"
    value_usd: float = 0.0

    def __post_init__(self):
        if self.value_usd == 0.0:
            self.value_usd = self.price * self.quantity


@dataclass
class OrderBookLevel:
    """Nível do orderbook."""
    price: float
    quantity: float
    side: Side
    exchange: str = "binance"


@dataclass
class OrderBookSnapshot:
    """Snapshot do orderbook."""
    timestamp: float
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    exchange: str = "binance"
    symbol: str = "BTCUSDT"

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return 0.0

    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0

    @property
    def total_bid_depth(self) -> float:
        return sum(level.quantity for level in self.bids)

    @property
    def total_ask_depth(self) -> float:
        return sum(level.quantity for level in self.asks)

    @property
    def bid_ask_ratio(self) -> float:
        ask_depth = self.total_ask_depth
        if ask_depth > 0:
            return self.total_bid_depth / ask_depth
        return 0.0


@dataclass
class Signal:
    """Sinal de trading institucional."""
    timestamp: float
    signal_type: str
    direction: Side
    strength: SignalStrength
    price: float
    confidence: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "signal_type": self.signal_type,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "price": self.price,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
            "description": self.description,
        }


@dataclass
class AnalysisResult:
    """Resultado de uma análise institucional."""
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    signals: list[Signal] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source,
            "signals": [s.to_dict() for s in self.signals],
            "metrics": self.metrics,
            "metadata": self.metadata,
            "regime": self.regime.value,
            "confidence": self.confidence,
        }
