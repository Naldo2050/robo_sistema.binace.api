"""
Whale Detector — Detecção de trades de baleias.

Monitora trades em tempo real e identifica
operações de grande volume (whales).

Substituto gratuito dos métodos #34 (Dark Pool) e #35 (Whale Alerts).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InvalidParameterError,
    Side,
    Signal,
    SignalStrength,
    Trade,
)


@dataclass
class WhaleEvent:
    """Evento de baleia detectado."""
    timestamp: float
    price: float
    quantity: float
    value_usd: float
    side: Side
    category: str  # "whale", "mega_whale", "institutional"
    percentile: float  # Percentil em relação ao volume médio
    impact_score: float  # 0-1, impacto estimado


@dataclass
class WhaleActivity:
    """Resumo da atividade de baleias em janela."""
    total_whale_volume: float
    total_whale_count: int
    buy_whale_volume: float
    sell_whale_volume: float
    buy_whale_count: int
    sell_whale_count: int
    net_whale_flow: float  # buy - sell
    largest_trade: float
    avg_whale_size: float


class WhaleDetector:
    """
    Detector de trades de baleias.

    Monitora volume de cada trade e categoriza:
    - Retail: abaixo do threshold
    - Whale: acima do threshold (top 1% do volume)
    - Mega Whale: acima de 5x o threshold
    - Institutional: acima de 10x o threshold

    Adaptativo: ajusta thresholds baseado no volume recente.
    """

    def __init__(
        self,
        whale_threshold_usd: float = 100_000.0,
        mega_whale_multiplier: float = 5.0,
        institutional_multiplier: float = 10.0,
        adaptive: bool = True,
        adaptive_percentile: float = 99.0,
        volume_history_size: int = 10000,
        max_events: int = 1000,
    ):
        if whale_threshold_usd <= 0:
            raise InvalidParameterError("whale_threshold_usd must be > 0")

        self.base_threshold = whale_threshold_usd
        self.mega_multiplier = mega_whale_multiplier
        self.institutional_multiplier = institutional_multiplier
        self.adaptive = adaptive
        self.adaptive_percentile = adaptive_percentile

        self._volume_history: deque[float] = deque(maxlen=volume_history_size)
        self._events: deque[WhaleEvent] = deque(maxlen=max_events)
        self._total_trades: int = 0
        self._total_volume: float = 0.0
        self._adaptive_threshold: float = whale_threshold_usd

    @property
    def threshold(self) -> float:
        if self.adaptive and self._adaptive_threshold > 0:
            return self._adaptive_threshold
        return self.base_threshold

    @property
    def events(self) -> list[WhaleEvent]:
        return list(self._events)

    @property
    def total_trades(self) -> int:
        return self._total_trades

    def _update_adaptive_threshold(self) -> None:
        """Atualiza threshold adaptativo baseado no histórico."""
        if len(self._volume_history) < 100:
            return

        sorted_vols = sorted(self._volume_history)
        idx = int(len(sorted_vols) * self.adaptive_percentile / 100)
        idx = min(idx, len(sorted_vols) - 1)

        new_threshold = sorted_vols[idx]
        # Não deixar cair abaixo de metade do base
        self._adaptive_threshold = max(new_threshold, self.base_threshold * 0.5)

    def process_trade(self, trade: Trade) -> Optional[WhaleEvent]:
        """
        Processa trade e retorna WhaleEvent se for baleia.
        """
        self._total_trades += 1
        self._total_volume += trade.value_usd
        self._volume_history.append(trade.value_usd)

        # Atualizar threshold periodicamente
        if self.adaptive and self._total_trades % 1000 == 0:
            self._update_adaptive_threshold()

        # Verificar se é whale
        threshold = self.threshold
        if trade.value_usd < threshold:
            return None

        # Categorizar
        if trade.value_usd >= threshold * self.institutional_multiplier:
            category = "institutional"
        elif trade.value_usd >= threshold * self.mega_multiplier:
            category = "mega_whale"
        else:
            category = "whale"

        # Calcular percentil
        if self._volume_history:
            sorted_vols = sorted(self._volume_history)
            count_below = sum(1 for v in sorted_vols if v <= trade.value_usd)
            percentile = (count_below / len(sorted_vols)) * 100
        else:
            percentile = 99.0

        # Impact score
        avg_vol = self._total_volume / max(self._total_trades, 1)
        impact = min(trade.value_usd / max(avg_vol * 10, 1), 1.0)

        event = WhaleEvent(
            timestamp=trade.timestamp,
            price=trade.price,
            quantity=trade.quantity,
            value_usd=trade.value_usd,
            side=trade.side,
            category=category,
            percentile=percentile,
            impact_score=impact,
        )

        self._events.append(event)
        return event

    def get_whale_activity(
        self,
        window_seconds: float = 300.0,
    ) -> WhaleActivity:
        """Resumo da atividade de baleias na janela."""
        now = time.time()
        recent = [
            e for e in self._events
            if now - e.timestamp <= window_seconds
        ]

        buy_events = [e for e in recent if e.side == Side.BUY]
        sell_events = [e for e in recent if e.side == Side.SELL]

        buy_vol = sum(e.value_usd for e in buy_events)
        sell_vol = sum(e.value_usd for e in sell_events)
        total_vol = buy_vol + sell_vol

        return WhaleActivity(
            total_whale_volume=total_vol,
            total_whale_count=len(recent),
            buy_whale_volume=buy_vol,
            sell_whale_volume=sell_vol,
            buy_whale_count=len(buy_events),
            sell_whale_count=len(sell_events),
            net_whale_flow=buy_vol - sell_vol,
            largest_trade=max((e.value_usd for e in recent), default=0),
            avg_whale_size=total_vol / max(len(recent), 1),
        )

    def get_whale_pressure(self, window_seconds: float = 300.0) -> dict:
        """
        Calcula pressão de baleias (buy vs sell).

        Retorna score de -1.0 (pressão vendedora) a +1.0 (pressão compradora).
        """
        activity = self.get_whale_activity(window_seconds)

        total = activity.buy_whale_volume + activity.sell_whale_volume
        if total == 0:
            return {
                "pressure_score": 0.0,
                "direction": "neutral",
                "confidence": 0.0,
            }

        score = (activity.buy_whale_volume - activity.sell_whale_volume) / total

        if score > 0.3:
            direction = "bullish"
        elif score < -0.3:
            direction = "bearish"
        else:
            direction = "neutral"

        return {
            "pressure_score": score,
            "direction": direction,
            "confidence": abs(score),
            "buy_volume": activity.buy_whale_volume,
            "sell_volume": activity.sell_whale_volume,
            "whale_count": activity.total_whale_count,
        }

    def analyze(self) -> AnalysisResult:
        """Análise completa de atividade de baleias."""
        result = AnalysisResult(
            source="whale_detector",
            timestamp=time.time(),
        )

        activity = self.get_whale_activity(300)
        pressure = self.get_whale_pressure(300)

        result.metrics = {
            "total_trades_analyzed": self._total_trades,
            "whale_events_5min": activity.total_whale_count,
            "whale_volume_5min": activity.total_whale_volume,
            "buy_whale_volume": activity.buy_whale_volume,
            "sell_whale_volume": activity.sell_whale_volume,
            "net_whale_flow": activity.net_whale_flow,
            "whale_pressure": pressure["pressure_score"],
            "largest_trade": activity.largest_trade,
            "current_threshold": self.threshold,
        }

        # Sinal de pressão de baleias
        if abs(pressure["pressure_score"]) > 0.3 and activity.total_whale_count >= 2:
            direction = Side.BUY if pressure["pressure_score"] > 0 else Side.SELL

            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="whale_pressure",
                    direction=direction,
                    strength=(
                        SignalStrength.STRONG if abs(pressure["pressure_score"]) > 0.6
                        else SignalStrength.MODERATE
                    ),
                    price=self._events[-1].price if self._events else 0,
                    confidence=abs(pressure["pressure_score"]),
                    source="whale_detector",
                    description=(
                        f"Whale {pressure['direction']} pressure: "
                        f"score={pressure['pressure_score']:.2f}, "
                        f"{activity.total_whale_count} whale trades"
                    ),
                    metadata={
                        "buy_volume": activity.buy_whale_volume,
                        "sell_volume": activity.sell_whale_volume,
                        "whale_count": activity.total_whale_count,
                    },
                )
            )

        # Sinal de trade individual muito grande
        recent_events = [
            e for e in self._events
            if time.time() - e.timestamp < 60
        ]
        for event in recent_events:
            if event.category in ("mega_whale", "institutional"):
                result.signals.append(
                    Signal(
                        timestamp=event.timestamp,
                        signal_type=f"whale_{event.category}",
                        direction=event.side,
                        strength=SignalStrength.STRONG,
                        price=event.price,
                        confidence=event.impact_score,
                        source="whale_detector",
                        description=(
                            f"{event.category.upper()} {event.side.value}: "
                            f"${event.value_usd:,.0f} "
                            f"({event.quantity:.4f} @ {event.price:,.2f})"
                        ),
                    )
                )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta detector."""
        self._volume_history.clear()
        self._events.clear()
        self._total_trades = 0
        self._total_volume = 0.0
        self._adaptive_threshold = self.base_threshold
