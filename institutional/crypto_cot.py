"""
Crypto COT (Commitment of Traders) Equivalent.

Combina Funding Rate + Open Interest + Long/Short Ratio
para replicar o conceito do relatório COT em crypto.

Substituto gratuito do método #44.
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
)


@dataclass
class COTDataPoint:
    """Ponto de dados do Crypto COT."""
    timestamp: float
    funding_rate: float
    open_interest: float
    long_short_ratio: float  # >1 = mais longs, <1 = mais shorts
    top_trader_ls_ratio: float  # Ratio dos top traders
    price: float


@dataclass
class COTSignal:
    """Sinal derivado do COT crypto."""
    signal_type: str
    direction: str
    strength: float
    description: str


class CryptoCOT:
    """
    Equivalente do COT Report para mercados crypto.

    Analisa:
    1. Funding Rate: quem paga quem (longs vs shorts)
    2. Open Interest: dinheiro total posicionado
    3. Long/Short Ratio: proporção de posições
    4. Top Traders Ratio: posicionamento dos maiores

    Sinais:
    - Funding extremo + OI alto = possível squeeze
    - Top traders divergem do varejo = smart money signal
    - OI crescendo com preço = dinheiro novo na tendência
    - OI caindo com preço estável = desalavancagem
    """

    # Thresholds de funding rate
    FUNDING_EXTREME_HIGH = 0.01   # 1% por 8h
    FUNDING_HIGH = 0.005          # 0.5%
    FUNDING_EXTREME_LOW = -0.01
    FUNDING_LOW = -0.005

    def __init__(
        self,
        max_history: int = 500,
        squeeze_threshold: float = 0.01,
        oi_change_threshold_pct: float = 10.0,
    ):
        self.max_history = max_history
        self.squeeze_threshold = squeeze_threshold
        self.oi_change_threshold = oi_change_threshold_pct

        self._data: deque[COTDataPoint] = deque(maxlen=max_history)

    @property
    def data_points(self) -> int:
        return len(self._data)

    @property
    def latest(self) -> Optional[COTDataPoint]:
        return self._data[-1] if self._data else None

    def add_data(
        self,
        timestamp: float,
        funding_rate: float,
        open_interest: float,
        long_short_ratio: float,
        top_trader_ls_ratio: float = 1.0,
        price: float = 0.0,
    ) -> None:
        """Adiciona ponto de dados."""
        self._data.append(COTDataPoint(
            timestamp=timestamp,
            funding_rate=funding_rate,
            open_interest=open_interest,
            long_short_ratio=long_short_ratio,
            top_trader_ls_ratio=top_trader_ls_ratio,
            price=price,
        ))

    def _analyze_funding(self) -> list[COTSignal]:
        """Analisa funding rate."""
        signals = []
        if not self._data:
            return signals

        fr = self._data[-1].funding_rate

        if fr >= self.FUNDING_EXTREME_HIGH:
            signals.append(COTSignal(
                signal_type="funding_extreme_high",
                direction="bearish",
                strength=min(abs(fr) / 0.02, 1.0),
                description=(
                    f"EXTREME positive funding ({fr*100:.4f}%): "
                    f"longs paying heavily. Short squeeze risk low, "
                    f"long squeeze risk HIGH."
                ),
            ))
        elif fr >= self.FUNDING_HIGH:
            signals.append(COTSignal(
                signal_type="funding_high",
                direction="bearish",
                strength=min(abs(fr) / 0.01, 1.0),
                description=(
                    f"High positive funding ({fr*100:.4f}%): "
                    f"market crowded long."
                ),
            ))
        elif fr <= self.FUNDING_EXTREME_LOW:
            signals.append(COTSignal(
                signal_type="funding_extreme_low",
                direction="bullish",
                strength=min(abs(fr) / 0.02, 1.0),
                description=(
                    f"EXTREME negative funding ({fr*100:.4f}%): "
                    f"shorts paying heavily. Long squeeze risk low, "
                    f"short squeeze risk HIGH."
                ),
            ))
        elif fr <= self.FUNDING_LOW:
            signals.append(COTSignal(
                signal_type="funding_low",
                direction="bullish",
                strength=min(abs(fr) / 0.01, 1.0),
                description=(
                    f"Low negative funding ({fr*100:.4f}%): "
                    f"market crowded short."
                ),
            ))

        return signals

    def _analyze_open_interest(self) -> list[COTSignal]:
        """Analisa Open Interest."""
        signals = []
        if len(self._data) < 5:
            return signals

        recent = list(self._data)[-5:]
        oi_start = recent[0].open_interest
        oi_end = recent[-1].open_interest
        price_start = recent[0].price
        price_end = recent[-1].price

        if oi_start <= 0:
            return signals

        oi_change_pct = ((oi_end - oi_start) / oi_start) * 100
        price_change_pct = (
            ((price_end - price_start) / price_start) * 100
            if price_start > 0 else 0
        )

        # OI subindo + preço subindo = dinheiro novo na alta
        if oi_change_pct > self.oi_change_threshold and price_change_pct > 0:
            signals.append(COTSignal(
                signal_type="oi_confirming_uptrend",
                direction="bullish",
                strength=min(oi_change_pct / 20, 1.0),
                description=(
                    f"OI +{oi_change_pct:.1f}% with price +{price_change_pct:.1f}%: "
                    f"new money entering uptrend."
                ),
            ))

        # OI subindo + preço caindo = dinheiro novo na baixa
        elif oi_change_pct > self.oi_change_threshold and price_change_pct < 0:
            signals.append(COTSignal(
                signal_type="oi_confirming_downtrend",
                direction="bearish",
                strength=min(oi_change_pct / 20, 1.0),
                description=(
                    f"OI +{oi_change_pct:.1f}% with price {price_change_pct:.1f}%: "
                    f"new money entering downtrend."
                ),
            ))

        # OI caindo = desalavancagem
        elif oi_change_pct < -self.oi_change_threshold:
            signals.append(COTSignal(
                signal_type="oi_deleveraging",
                direction="neutral",
                strength=min(abs(oi_change_pct) / 20, 1.0),
                description=(
                    f"OI {oi_change_pct:.1f}%: deleveraging in progress."
                ),
            ))

        return signals

    def _analyze_positioning(self) -> list[COTSignal]:
        """Analisa posicionamento long/short e divergência retail vs smart money."""
        signals = []
        if not self._data:
            return signals

        latest = self._data[-1]
        ls = latest.long_short_ratio
        top_ls = latest.top_trader_ls_ratio

        # Divergência entre top traders e varejo
        if top_ls > 0 and ls > 0:
            divergence = top_ls / ls

            if divergence > 1.3:
                # Top traders mais long que varejo
                signals.append(COTSignal(
                    signal_type="smart_money_long",
                    direction="bullish",
                    strength=min((divergence - 1) / 0.5, 1.0),
                    description=(
                        f"Smart money more long than retail: "
                        f"top={top_ls:.2f}, retail={ls:.2f}"
                    ),
                ))
            elif divergence < 0.7:
                # Top traders menos long que varejo
                signals.append(COTSignal(
                    signal_type="smart_money_short",
                    direction="bearish",
                    strength=min((1 - divergence) / 0.5, 1.0),
                    description=(
                        f"Smart money more short than retail: "
                        f"top={top_ls:.2f}, retail={ls:.2f}"
                    ),
                ))

        # Crowding extremo
        if ls > 2.0:
            signals.append(COTSignal(
                signal_type="crowded_long",
                direction="bearish",
                strength=min((ls - 1) / 3, 1.0),
                description=f"Market crowded long: L/S ratio={ls:.2f}",
            ))
        elif ls < 0.5:
            signals.append(COTSignal(
                signal_type="crowded_short",
                direction="bullish",
                strength=min((1 / max(ls, 0.01) - 1) / 3, 1.0),
                description=f"Market crowded short: L/S ratio={ls:.2f}",
            ))

        return signals

    def _detect_squeeze_conditions(self) -> list[COTSignal]:
        """Detecta condições para squeeze."""
        signals = []
        if not self._data:
            return signals

        latest = self._data[-1]
        fr = latest.funding_rate
        ls = latest.long_short_ratio

        # Short squeeze: funding negativo + muitos shorts
        if fr < -self.squeeze_threshold and ls < 0.8:
            signals.append(COTSignal(
                signal_type="short_squeeze_risk",
                direction="bullish",
                strength=min(abs(fr) / 0.02 + (1 - ls), 1.0),
                description=(
                    f"SHORT SQUEEZE conditions: "
                    f"funding={fr*100:.3f}%, L/S={ls:.2f}"
                ),
            ))

        # Long squeeze: funding positivo + muitos longs
        if fr > self.squeeze_threshold and ls > 1.5:
            signals.append(COTSignal(
                signal_type="long_squeeze_risk",
                direction="bearish",
                strength=min(fr / 0.02 + (ls - 1), 1.0),
                description=(
                    f"LONG SQUEEZE conditions: "
                    f"funding={fr*100:.3f}%, L/S={ls:.2f}"
                ),
            ))

        return signals

    def analyze(self) -> AnalysisResult:
        """Análise completa do Crypto COT."""
        result = AnalysisResult(
            source="crypto_cot",
            timestamp=time.time(),
        )

        if not self._data:
            result.confidence = 0.0
            return result

        latest = self._data[-1]

        result.metrics = {
            "funding_rate": latest.funding_rate,
            "funding_rate_pct": latest.funding_rate * 100,
            "open_interest": latest.open_interest,
            "long_short_ratio": latest.long_short_ratio,
            "top_trader_ls_ratio": latest.top_trader_ls_ratio,
            "price": latest.price,
            "data_points": len(self._data),
        }

        # Coletar todos os sinais
        all_signals: list[COTSignal] = []
        all_signals.extend(self._analyze_funding())
        all_signals.extend(self._analyze_open_interest())
        all_signals.extend(self._analyze_positioning())
        all_signals.extend(self._detect_squeeze_conditions())

        for cot_signal in all_signals:
            direction = (
                Side.BUY if cot_signal.direction == "bullish"
                else Side.SELL if cot_signal.direction == "bearish"
                else Side.UNKNOWN
            )

            strength = (
                SignalStrength.STRONG if cot_signal.strength > 0.7
                else SignalStrength.MODERATE if cot_signal.strength > 0.4
                else SignalStrength.WEAK
            )

            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type=f"cot_{cot_signal.signal_type}",
                    direction=direction,
                    strength=strength,
                    price=latest.price,
                    confidence=cot_signal.strength,
                    source="crypto_cot",
                    description=cot_signal.description,
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta dados."""
        self._data.clear()
