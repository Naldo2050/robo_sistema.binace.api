# institutional/order_flow_imbalance.py
"""
Order Flow Imbalance (OFI)

Calcula desequilíbrio entre volume comprador e vendedor
em cada barra de preço. Quando um lado domina 300%+,
sinaliza movimento iminente.

Método #5 do Arsenal Institucional.
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
class OFIBar:
    """Barra de Order Flow Imbalance."""
    timestamp: float
    buy_volume: float
    sell_volume: float
    buy_count: int
    sell_count: int
    imbalance_ratio: float  # buy/sell ratio
    price_open: float
    price_close: float
    price_change_pct: float
    dominant_side: Side

    @property
    def net_volume(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume


@dataclass
class OFIAlert:
    """Alerta de desequilíbrio."""
    timestamp: float
    ratio: float
    side: Side
    volume: float
    price: float
    severity: str  # "extreme", "strong", "moderate"


class OrderFlowImbalanceAnalyzer:
    """
    Analisador de desequilíbrio de fluxo de ordens.

    Monitora ratio compra/venda em tempo real.
    Ratio >= 3.0 indica desequilíbrio forte.
    Ratio >= 5.0 indica desequilíbrio extremo.
    """

    # Thresholds de desequilíbrio
    MODERATE_THRESHOLD = 2.0
    STRONG_THRESHOLD = 3.0
    EXTREME_THRESHOLD = 5.0

    def __init__(
        self,
        window_seconds: float = 30.0,
        max_history: int = 500,
        alert_cooldown_seconds: float = 60.0,
    ):
        if window_seconds <= 0:
            raise InvalidParameterError(
                f"window_seconds must be > 0, got {window_seconds}"
            )

        self.window_seconds = window_seconds
        self.max_history = max_history
        self.alert_cooldown = alert_cooldown_seconds

        self.bars: deque[OFIBar] = deque(maxlen=max_history)
        self.alerts: list[OFIAlert] = []

        # Buffer da janela atual
        self._window_start: float = 0.0
        self._window_initialized: bool = False
        self._buy_vol: float = 0.0
        self._sell_vol: float = 0.0
        self._buy_count: int = 0
        self._sell_count: int = 0
        self._price_open: float = 0.0
        self._price_close: float = 0.0
        self._last_alert_time: float = 0.0
        self._total_trades: int = 0

    @property
    def total_trades_processed(self) -> int:
        return self._total_trades

    def process_trade(self, trade: Trade) -> Optional[OFIBar]:
        """Processa trade, retorna OFIBar se janela fechou."""
        self._total_trades += 1

        completed_bar = None

        if not self._window_initialized:
            self._window_start = trade.timestamp
            self._price_open = trade.price
            self._window_initialized = True
        else:
            # Verificar se janela fechou ANTES de adicionar o trade
            elapsed = trade.timestamp - self._window_start
            if elapsed >= self.window_seconds:
                completed_bar = self._close_window(trade.timestamp)

        # Acumular
        if trade.side == Side.BUY:
            self._buy_vol += trade.quantity
            self._buy_count += 1
        elif trade.side == Side.SELL:
            self._sell_vol += trade.quantity
            self._sell_count += 1

        self._price_close = trade.price

        return completed_bar

    def _close_window(self, timestamp: float) -> OFIBar:
        """Fecha janela e calcula métricas."""
        sell_vol = max(self._sell_vol, 1e-10)
        buy_vol = max(self._buy_vol, 1e-10)

        imbalance_ratio = self._buy_vol / sell_vol if self._sell_vol > 0 else (
            float("inf") if self._buy_vol > 0 else 1.0
        )

        if self._buy_vol > self._sell_vol:
            dominant = Side.BUY
        elif self._sell_vol > self._buy_vol:
            dominant = Side.SELL
        else:
            dominant = Side.UNKNOWN

        price_change = 0.0
        if self._price_open > 0:
            price_change = (
                (self._price_close - self._price_open) / self._price_open
            ) * 100

        bar = OFIBar(
            timestamp=self._window_start,
            buy_volume=self._buy_vol,
            sell_volume=self._sell_vol,
            buy_count=self._buy_count,
            sell_count=self._sell_count,
            imbalance_ratio=imbalance_ratio,
            price_open=self._price_open,
            price_close=self._price_close,
            price_change_pct=price_change,
            dominant_side=dominant,
        )

        self.bars.append(bar)

        # Verificar se gera alerta
        self._check_alert(bar)

        # Reset
        self._window_start = timestamp
        self._window_initialized = False
        self._buy_vol = 0.0
        self._sell_vol = 0.0
        self._buy_count = 0
        self._sell_count = 0
        self._price_open = self._price_close
        # _price_close mantém o último valor

        return bar

    def _check_alert(self, bar: OFIBar) -> None:
        """Verifica se barra gera alerta."""
        ratio = bar.imbalance_ratio
        inverse_ratio = 1.0 / ratio if ratio > 0 else float("inf")

        effective_ratio = max(ratio, inverse_ratio)
        if effective_ratio == float("inf"):
            effective_ratio = 100.0

        if effective_ratio < self.MODERATE_THRESHOLD:
            return

        # Cooldown
        if (bar.timestamp - self._last_alert_time) < self.alert_cooldown:
            return

        if effective_ratio >= self.EXTREME_THRESHOLD:
            severity = "extreme"
        elif effective_ratio >= self.STRONG_THRESHOLD:
            severity = "strong"
        else:
            severity = "moderate"

        alert = OFIAlert(
            timestamp=bar.timestamp,
            ratio=effective_ratio,
            side=bar.dominant_side,
            volume=bar.total_volume,
            price=bar.price_close,
            severity=severity,
        )

        self.alerts.append(alert)
        self._last_alert_time = bar.timestamp

    def get_current_imbalance(self) -> dict:
        """Imbalance da janela atual (não fechada)."""
        sell_vol = max(self._sell_vol, 1e-10)
        ratio = self._buy_vol / sell_vol

        if self._buy_vol > self._sell_vol:
            dominant = Side.BUY
        elif self._sell_vol > self._buy_vol:
            dominant = Side.SELL
        else:
            dominant = Side.UNKNOWN

        return {
            "buy_volume": self._buy_vol,
            "sell_volume": self._sell_vol,
            "ratio": ratio,
            "dominant_side": dominant.value,
            "buy_count": self._buy_count,
            "sell_count": self._sell_count,
        }

    def get_trend_strength(self, lookback: int = 5) -> dict:
        """
        Calcula força da tendência baseado em barras recentes.

        Retorna score de -1.0 (venda extrema) a +1.0 (compra extrema).
        """
        if len(self.bars) < lookback:
            return {"score": 0.0, "direction": "neutral", "confidence": 0.0}

        recent = list(self.bars)[-lookback:]

        buy_dominant_count = sum(
            1 for b in recent if b.dominant_side == Side.BUY
        )
        sell_dominant_count = sum(
            1 for b in recent if b.dominant_side == Side.SELL
        )

        avg_ratio = sum(b.imbalance_ratio for b in recent) / len(recent)

        if buy_dominant_count > sell_dominant_count:
            score = buy_dominant_count / lookback
            direction = "bullish"
        elif sell_dominant_count > buy_dominant_count:
            score = -(sell_dominant_count / lookback)
            direction = "bearish"
        else:
            score = 0.0
            direction = "neutral"

        return {
            "score": score,
            "direction": direction,
            "avg_ratio": avg_ratio,
            "buy_dominant_bars": buy_dominant_count,
            "sell_dominant_bars": sell_dominant_count,
            "confidence": abs(score),
        }

    def analyze(self) -> AnalysisResult:
        """Análise completa."""
        result = AnalysisResult(
            source="ofi_analyzer",
            timestamp=time.time(),
        )

        if len(self.bars) < 2:
            result.confidence = 0.0
            return result

        trend = self.get_trend_strength()
        current = self.get_current_imbalance()
        last_bar = self.bars[-1]

        result.metrics = {
            "current_ratio": current["ratio"],
            "current_dominant": current["dominant_side"],
            "trend_score": trend["score"],
            "trend_direction": trend["direction"],
            "avg_ratio": trend["avg_ratio"],
            "recent_alerts": len([
                a for a in self.alerts
                if time.time() - a.timestamp < 300
            ]),
            "bar_count": len(self.bars),
        }

        # Sinais baseados em tendência
        if abs(trend["score"]) > 0.6:
            direction = Side.BUY if trend["score"] > 0 else Side.SELL
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="ofi_trend",
                    direction=direction,
                    strength=SignalStrength.STRONG if abs(trend["score"]) > 0.8 else SignalStrength.MODERATE,
                    price=last_bar.price_close,
                    confidence=abs(trend["score"]),
                    source="ofi_analyzer",
                    description=(
                        f"OFI trend {trend['direction']}: "
                        f"score={trend['score']:.2f}"
                    ),
                )
            )

        # Sinais de alertas recentes
        recent_alerts = [
            a for a in self.alerts
            if time.time() - a.timestamp < 120
        ]
        for alert in recent_alerts:
            result.signals.append(
                Signal(
                    timestamp=alert.timestamp,
                    signal_type=f"ofi_imbalance_{alert.severity}",
                    direction=alert.side,
                    strength=(
                        SignalStrength.STRONG if alert.severity == "extreme"
                        else SignalStrength.MODERATE
                    ),
                    price=alert.price,
                    confidence=min(alert.ratio / 10.0, 1.0),
                    source="ofi_analyzer",
                    description=(
                        f"{alert.severity.upper()} imbalance: "
                        f"{alert.ratio:.1f}x {alert.side.value}"
                    ),
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta estado."""
        self.bars.clear()
        self.alerts.clear()
        self._window_start = 0.0
        self._window_initialized = False
        self._buy_vol = 0.0
        self._sell_vol = 0.0
        self._buy_count = 0
        self._sell_count = 0
        self._price_open = 0.0
        self._price_close = 0.0
        self._last_alert_time = 0.0
        self._total_trades = 0
