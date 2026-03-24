"""
Fourier Transform / Análise Espectral.

Decompõe o preço em ciclos de diferentes frequências.
Identifica ciclos dominantes e periodicidades ocultas.

Método #26 do Arsenal Institucional.
"""
from __future__ import annotations

import math
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
class CycleComponent:
    """Componente cíclico detectado."""
    period: float       # Período em barras
    amplitude: float    # Amplitude relativa
    phase: float        # Fase atual (radianos)
    power: float        # Potência espectral
    rank: int           # Ranking por importância


@dataclass
class FourierResult:
    """Resultado da análise de Fourier."""
    dominant_cycles: list[CycleComponent]
    spectral_entropy: float  # Concentração espectral
    cycle_strength: float    # Quão cíclico é o mercado
    next_peak_estimate: Optional[float]
    next_trough_estimate: Optional[float]


class FourierCycleAnalyzer:
    """
    Analisador de ciclos via Transformada de Fourier (DFT).

    Implementação sem numpy — DFT manual para independência.

    Usos:
    1. Identificar ciclos dominantes no preço
    2. Estimar próximos topos/fundos cíclicos
    3. Medir quão cíclico vs aleatório é o mercado
    """

    def __init__(
        self,
        min_period: int = 5,
        max_period: int = 200,
        top_cycles: int = 5,
        window_size: int = 256,
        max_history: int = 2000,
    ):
        if min_period < 2:
            raise InvalidParameterError("min_period must be >= 2")
        if window_size < 32:
            raise InvalidParameterError("window_size must be >= 32")

        self.min_period = min_period
        self.max_period = min(max_period, window_size // 2)
        self.top_cycles = top_cycles
        self.window_size = window_size

        self._prices: deque[float] = deque(maxlen=max_history)
        self._last_result: Optional[FourierResult] = None

    @property
    def data_points(self) -> int:
        return len(self._prices)

    @property
    def last_result(self) -> Optional[FourierResult]:
        return self._last_result

    def add_price(self, price: float) -> None:
        """Adiciona preço."""
        self._prices.append(price)

    def add_prices(self, prices: list[float]) -> None:
        """Adiciona múltiplos preços."""
        for p in prices:
            self._prices.append(p)

    def _detrend(self, data: list[float]) -> list[float]:
        """Remove tendência linear dos dados."""
        n = len(data)
        if n < 2:
            return data

        # Regressão linear simples
        x_mean = (n - 1) / 2.0
        y_mean = sum(data) / n

        num = sum((i - x_mean) * (data[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))

        if abs(den) < 1e-12:
            return [d - y_mean for d in data]

        slope = num / den
        intercept = y_mean - slope * x_mean

        return [data[i] - (slope * i + intercept) for i in range(n)]

    def _dft(self, data: list[float]) -> list[tuple[float, float]]:
        """
        Discrete Fourier Transform manual.
        Retorna lista de (amplitude, phase) para cada frequência.
        """
        n = len(data)
        result = []

        for k in range(n // 2 + 1):
            real = 0.0
            imag = 0.0

            for t in range(n):
                angle = 2 * math.pi * k * t / n
                real += data[t] * math.cos(angle)
                imag -= data[t] * math.sin(angle)

            amplitude = math.sqrt(real ** 2 + imag ** 2) / n
            phase = math.atan2(imag, real)

            result.append((amplitude, phase))

        return result

    def calculate(self) -> Optional[FourierResult]:
        """
        Calcula análise de Fourier.
        Retorna None se dados insuficientes.
        """
        if len(self._prices) < self.window_size:
            return None

        # Pegar janela mais recente
        window = list(self._prices)[-self.window_size:]

        # Detrend
        detrended = self._detrend(window)

        # DFT
        spectrum = self._dft(detrended)

        # Encontrar ciclos dominantes
        cycles: list[CycleComponent] = []

        for k in range(1, len(spectrum)):
            period = self.window_size / k

            if period < self.min_period or period > self.max_period:
                continue

            amplitude, phase = spectrum[k]
            power = amplitude ** 2

            cycles.append(CycleComponent(
                period=period,
                amplitude=amplitude,
                phase=phase,
                power=power,
                rank=0,
            ))

        # Ordenar por potência
        cycles.sort(key=lambda c: c.power, reverse=True)

        # Atribuir ranks
        for i, cycle in enumerate(cycles):
            cycle.rank = i + 1

        top = cycles[:self.top_cycles]

        # Entropia espectral (concentração)
        total_power = sum(c.power for c in cycles)
        if total_power > 0:
            probs = [c.power / total_power for c in cycles]
            spectral_entropy = -sum(
                p * math.log2(p) for p in probs if p > 0
            )
            max_entropy = math.log2(len(cycles)) if cycles else 1
            norm_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0
        else:
            norm_entropy = 1.0

        cycle_strength = 1.0 - norm_entropy

        # Estimar próximo pico/vale do ciclo dominante
        next_peak = None
        next_trough = None

        if top:
            dominant = top[0]
            current_phase = dominant.phase
            # Fase do pico = 0 ou 2π
            # Fase do vale = π
            bars_to_peak = ((-current_phase) % (2 * math.pi)) / (2 * math.pi) * dominant.period
            bars_to_trough = ((math.pi - current_phase) % (2 * math.pi)) / (2 * math.pi) * dominant.period

            next_peak = bars_to_peak
            next_trough = bars_to_trough

        result = FourierResult(
            dominant_cycles=top,
            spectral_entropy=norm_entropy,
            cycle_strength=cycle_strength,
            next_peak_estimate=next_peak,
            next_trough_estimate=next_trough,
        )

        self._last_result = result
        return result

    def analyze(self) -> AnalysisResult:
        """Análise completa."""
        result = AnalysisResult(
            source="fourier_cycles",
            timestamp=time.time(),
        )

        fourier = self.calculate()
        if fourier is None:
            result.confidence = 0.0
            return result

        result.metrics = {
            "spectral_entropy": fourier.spectral_entropy,
            "cycle_strength": fourier.cycle_strength,
            "dominant_cycles_count": len(fourier.dominant_cycles),
            "data_points": len(self._prices),
        }

        if fourier.next_peak_estimate is not None:
            result.metrics["bars_to_peak"] = fourier.next_peak_estimate
        if fourier.next_trough_estimate is not None:
            result.metrics["bars_to_trough"] = fourier.next_trough_estimate

        for i, cycle in enumerate(fourier.dominant_cycles[:3]):
            result.metrics[f"cycle_{i+1}_period"] = cycle.period
            result.metrics[f"cycle_{i+1}_amplitude"] = cycle.amplitude
            result.metrics[f"cycle_{i+1}_power"] = cycle.power

        # Sinal se mercado é fortemente cíclico
        if fourier.cycle_strength > 0.5:
            desc_parts = [
                f"Market shows strong cyclical behavior "
                f"(strength={fourier.cycle_strength:.2f}). "
            ]
            if fourier.dominant_cycles:
                desc_parts.append(
                    f"Dominant cycle: {fourier.dominant_cycles[0].period:.0f} bars. "
                )
            if fourier.next_peak_estimate is not None:
                desc_parts.append(
                    f"Est. peak in ~{fourier.next_peak_estimate:.0f} bars, "
                    f"trough in ~{fourier.next_trough_estimate:.0f} bars."
                )

            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="fourier_strong_cycle",
                    direction=Side.UNKNOWN,
                    strength=(
                        SignalStrength.STRONG if fourier.cycle_strength > 0.7
                        else SignalStrength.MODERATE
                    ),
                    price=self._prices[-1] if self._prices else 0,
                    confidence=fourier.cycle_strength,
                    source="fourier_cycles",
                    description="".join(desc_parts),
                    metadata={
                        "dominant_period": fourier.dominant_cycles[0].period if fourier.dominant_cycles else 0,
                        "next_peak_bars": fourier.next_peak_estimate,
                        "next_trough_bars": fourier.next_trough_estimate,
                    },
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals),
            default=fourier.cycle_strength * 0.5,
        )

        return result

    def reset(self) -> None:
        self._prices.clear()
        self._last_result = None
