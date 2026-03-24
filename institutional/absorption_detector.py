# institutional/absorption_detector.py
"""
Absorption Analysis (Análise de Absorção)

Identifica quando um grande player absorve toda a pressão
compradora/vendedora sem deixar o preço se mover significativamente.

Preço para + volume explode = alguém grande está acumulando.

Método #7 do Arsenal Institucional.
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
class AbsorptionEvent:
    """Evento de absorção detectado."""
    timestamp: float
    price_level: float
    absorbed_volume: float
    absorption_side: Side  # Quem está absorvendo
    aggressor_side: Side   # Quem está sendo absorvido
    price_range: float     # Range de preço durante absorção
    duration_seconds: float
    trade_count: int
    intensity: float       # 0-1, quão forte a absorção
    description: str = ""


@dataclass
class AbsorptionWindow:
    """Janela de análise de absorção."""
    start_time: float
    trades: list[Trade] = field(default_factory=list)
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    price_high: float = 0.0
    price_low: float = float("inf")
    price_start: float = 0.0
    price_end: float = 0.0

    @property
    def price_range(self) -> float:
        if self.price_low == float("inf"):
            return 0.0
        return self.price_high - self.price_low

    @property
    def price_range_pct(self) -> float:
        mid = (self.price_high + self.price_low) / 2
        if mid > 0:
            return (self.price_range / mid) * 100
        return 0.0

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def volume_delta(self) -> float:
        return self.buy_volume - self.sell_volume


class AbsorptionDetector:
    """
    Detector de absorção institucional.

    Identifica situações onde:
    1. Volume é muito alto (acima da média)
    2. Preço se move muito pouco (abaixo do normal)
    3. Um lado domina agressivamente mas preço não responde

    Isso indica que um participante grande está absorvendo
    toda a pressão do lado oposto.
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        volume_multiplier: float = 2.0,
        price_threshold_pct: float = 0.05,
        min_trades_in_window: int = 10,
        history_size: int = 100,
    ):
        if window_seconds <= 0:
            raise InvalidParameterError("window_seconds must be > 0")

        self.window_seconds = window_seconds
        self.volume_multiplier = volume_multiplier
        self.price_threshold_pct = price_threshold_pct
        self.min_trades = min_trades_in_window
        self.history_size = history_size

        # Janelas históricas para baseline
        self._volume_history: deque[float] = deque(maxlen=history_size)
        self._range_history: deque[float] = deque(maxlen=history_size)

        self._current_window: Optional[AbsorptionWindow] = None
        self._events: list[AbsorptionEvent] = []
        self._total_trades: int = 0

    @property
    def events(self) -> list[AbsorptionEvent]:
        return self._events.copy()

    @property
    def total_trades_processed(self) -> int:
        return self._total_trades

    def _avg_volume(self) -> float:
        if not self._volume_history:
            return 0.0
        return sum(self._volume_history) / len(self._volume_history)

    def _avg_range(self) -> float:
        if not self._range_history:
            return 0.0
        return sum(self._range_history) / len(self._range_history)

    def process_trade(self, trade: Trade) -> Optional[AbsorptionEvent]:
        """Processa trade, retorna AbsorptionEvent se absorção detectada."""
        self._total_trades += 1

        # Iniciar janela
        if self._current_window is None:
            self._current_window = AbsorptionWindow(
                start_time=trade.timestamp,
                price_start=trade.price,
                price_high=trade.price,
                price_low=trade.price,
            )

        window = self._current_window

        # Adicionar trade à janela
        window.trades.append(trade)
        window.price_end = trade.price
        window.price_high = max(window.price_high, trade.price)
        window.price_low = min(window.price_low, trade.price)

        if trade.side == Side.BUY:
            window.buy_volume += trade.quantity
        elif trade.side == Side.SELL:
            window.sell_volume += trade.quantity

        # Verificar se janela fechou
        elapsed = trade.timestamp - window.start_time
        if elapsed >= self.window_seconds:
            return self._analyze_window()

        return None

    def _analyze_window(self) -> Optional[AbsorptionEvent]:
        """Analisa janela fechada e detecta absorção."""
        window = self._current_window
        if window is None:
            return None

        # Registrar histórico
        self._volume_history.append(window.total_volume)
        self._range_history.append(window.price_range_pct)

        event = None

        # Precisamos de baseline para comparar
        if len(self._volume_history) >= 5 and len(window.trades) >= self.min_trades:
            avg_vol = self._avg_volume()
            avg_range = self._avg_range()

            if avg_vol > 0 and avg_range > 0:
                volume_ratio = window.total_volume / avg_vol
                range_ratio = window.price_range_pct / avg_range if avg_range > 0 else 0

                # ABSORÇÃO: volume alto + range baixo
                is_high_volume = volume_ratio >= self.volume_multiplier
                is_low_range = (
                    window.price_range_pct < self.price_threshold_pct
                    or range_ratio < 0.5
                )

                if is_high_volume and is_low_range:
                    # Determinar quem absorve
                    if window.buy_volume > window.sell_volume * 1.5:
                        # Muita compra agressiva mas preço não sobe
                        # = vendedor absorvendo
                        absorption_side = Side.SELL
                        aggressor_side = Side.BUY
                    elif window.sell_volume > window.buy_volume * 1.5:
                        # Muita venda mas preço não cai
                        # = comprador absorvendo
                        absorption_side = Side.BUY
                        aggressor_side = Side.SELL
                    else:
                        # Volume alto mas equilibrado — absorção mútua
                        absorption_side = Side.UNKNOWN
                        aggressor_side = Side.UNKNOWN

                    intensity = min(
                        (volume_ratio / self.volume_multiplier) * 0.5
                        + (1.0 - min(range_ratio, 1.0)) * 0.5,
                        1.0,
                    )

                    duration = window.trades[-1].timestamp - window.start_time

                    event = AbsorptionEvent(
                        timestamp=window.start_time,
                        price_level=(window.price_high + window.price_low) / 2,
                        absorbed_volume=window.total_volume,
                        absorption_side=absorption_side,
                        aggressor_side=aggressor_side,
                        price_range=window.price_range,
                        duration_seconds=duration,
                        trade_count=len(window.trades),
                        intensity=intensity,
                        description=(
                            f"Absorption at {(window.price_high + window.price_low)/2:.2f}: "
                            f"vol={window.total_volume:.4f} "
                            f"({volume_ratio:.1f}x avg), "
                            f"range={window.price_range_pct:.4f}% "
                            f"({range_ratio:.2f}x avg), "
                            f"absorber={absorption_side.value}"
                        ),
                    )

                    self._events.append(event)

        # Reset janela
        self._current_window = AbsorptionWindow(
            start_time=window.trades[-1].timestamp if window.trades else time.time(),
            price_start=window.price_end,
            price_high=window.price_end,
            price_low=window.price_end,
        )

        return event

    def get_recent_events(
        self,
        max_age_seconds: float = 300.0,
    ) -> list[AbsorptionEvent]:
        """Retorna eventos recentes."""
        now = time.time()
        return [
            e for e in self._events
            if now - e.timestamp <= max_age_seconds
        ]

    def get_absorption_zones(
        self,
        price_tolerance_pct: float = 0.1,
        min_events: int = 2,
    ) -> list[dict]:
        """
        Agrupa eventos de absorção em zonas de preço.

        Múltiplas absorções no mesmo nível = zona forte.
        """
        if not self._events:
            return []

        sorted_events = sorted(self._events, key=lambda e: e.price_level)
        zones: list[dict] = []
        current_zone: list[AbsorptionEvent] = [sorted_events[0]]

        for i in range(1, len(sorted_events)):
            event = sorted_events[i]
            zone_price = sum(e.price_level for e in current_zone) / len(current_zone)
            price_diff_pct = abs(event.price_level - zone_price) / zone_price * 100

            if price_diff_pct <= price_tolerance_pct:
                current_zone.append(event)
            else:
                if len(current_zone) >= min_events:
                    zones.append(self._zone_from_events(current_zone))
                current_zone = [event]

        if len(current_zone) >= min_events:
            zones.append(self._zone_from_events(current_zone))

        return sorted(zones, key=lambda z: z["total_volume"], reverse=True)

    def _zone_from_events(self, events: list[AbsorptionEvent]) -> dict:
        """Cria zona a partir de lista de eventos."""
        prices = [e.price_level for e in events]
        return {
            "price_center": sum(prices) / len(prices),
            "price_low": min(prices),
            "price_high": max(prices),
            "event_count": len(events),
            "total_volume": sum(e.absorbed_volume for e in events),
            "avg_intensity": sum(e.intensity for e in events) / len(events),
            "dominant_absorber": max(
                set(e.absorption_side for e in events),
                key=lambda s: sum(1 for e in events if e.absorption_side == s),
            ).value,
            "first_seen": min(e.timestamp for e in events),
            "last_seen": max(e.timestamp for e in events),
        }

    def analyze(self) -> AnalysisResult:
        """Análise completa."""
        result = AnalysisResult(
            source="absorption_detector",
            timestamp=time.time(),
        )

        recent = self.get_recent_events(300)
        zones = self.get_absorption_zones()

        result.metrics = {
            "total_events": len(self._events),
            "recent_events_5min": len(recent),
            "absorption_zones": len(zones),
            "avg_intensity": (
                sum(e.intensity for e in recent) / len(recent)
                if recent else 0.0
            ),
        }

        # Sinais de absorção recente
        for event in recent[-3:]:  # últimos 3 eventos
            if event.absorption_side == Side.BUY:
                direction = Side.BUY  # Comprador forte absorvendo
                desc = "Buyer absorbing sell pressure — accumulation"
            elif event.absorption_side == Side.SELL:
                direction = Side.SELL  # Vendedor forte absorvendo
                desc = "Seller absorbing buy pressure — distribution"
            else:
                direction = Side.UNKNOWN
                desc = "Mutual absorption — consolidation"

            strength = (
                SignalStrength.STRONG if event.intensity > 0.7
                else SignalStrength.MODERATE if event.intensity > 0.4
                else SignalStrength.WEAK
            )

            result.signals.append(
                Signal(
                    timestamp=event.timestamp,
                    signal_type="absorption_detected",
                    direction=direction,
                    strength=strength,
                    price=event.price_level,
                    confidence=event.intensity,
                    source="absorption_detector",
                    description=desc,
                    metadata={
                        "volume": event.absorbed_volume,
                        "price_range": event.price_range,
                        "duration": event.duration_seconds,
                        "trades": event.trade_count,
                    },
                )
            )

        # Sinais de zonas fortes
        for zone in zones[:2]:  # top 2 zonas
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="absorption_zone",
                    direction=(
                        Side.BUY if zone["dominant_absorber"] == "buy"
                        else Side.SELL if zone["dominant_absorber"] == "sell"
                        else Side.UNKNOWN
                    ),
                    strength=SignalStrength.STRONG,
                    price=zone["price_center"],
                    confidence=zone["avg_intensity"],
                    source="absorption_detector",
                    description=(
                        f"Absorption zone at {zone['price_center']:.2f}: "
                        f"{zone['event_count']} events, "
                        f"vol={zone['total_volume']:.4f}"
                    ),
                    metadata=zone,
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta estado."""
        self._volume_history.clear()
        self._range_history.clear()
        self._current_window = None
        self._events.clear()
        self._total_trades = 0
