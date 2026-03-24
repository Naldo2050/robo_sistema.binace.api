"""
Confluence Engine — Motor de Confluência.

Combina sinais de TODOS os analisadores institucionais
e gera um score final de confluência.

Quando 3+ camadas concordam = sinal de alta confiança.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from institutional.base import (
    AnalysisResult,
    Side,
    Signal,
    SignalStrength,
)


@dataclass
class ConfluenceScore:
    """Score de confluência."""
    direction: Side
    total_score: float  # -1.0 a +1.0
    buy_signals: int
    sell_signals: int
    neutral_signals: int
    total_signals: int
    sources_agreeing: int
    total_sources: int
    confidence: float
    strength: SignalStrength
    summary: str
    signal_breakdown: dict[str, list[dict]] = field(default_factory=dict)


class ConfluenceEngine:
    """
    Motor de confluência multi-camada.

    Recebe AnalysisResults de múltiplos analisadores e calcula:
    1. Score direcional ponderado
    2. Número de fontes concordantes
    3. Nível de confluência geral

    Pesos por categoria:
    - Order Flow (CVD, Footprint, OFI, Absorption): peso 3
    - Estrutura (SMC, S/R): peso 2
    - Estatístico (Kalman, Hurst, GARCH): peso 1.5
    - Sentimento (COT, Whale, Entropy): peso 1
    - Probabilístico (Monte Carlo, Fourier): peso 0.5

    Confluência forte = 3+ camadas concordam.
    """

    # Pesos por fonte
    SOURCE_WEIGHTS: dict[str, float] = {
        # Order Flow — maior peso (informação mais direta)
        "cvd_analyzer": 3.0,
        "footprint_analyzer": 3.0,
        "ofi_analyzer": 3.0,
        "absorption_detector": 3.0,
        "iceberg_detector": 2.5,
        # Estrutura
        "smart_money_analyzer": 2.0,
        "vwap_twap_analyzer": 2.0,
        # Estatístico
        "kalman_filter": 1.5,
        "hurst_calculator": 1.5,
        "garch_model": 1.5,
        "mean_reversion": 1.5,
        # Sentimento / Posicionamento
        "crypto_cot": 1.0,
        "whale_detector": 1.0,
        "entropy_analyzer": 1.0,
        # Probabilístico
        "monte_carlo": 0.5,
        "fourier_cycles": 0.5,
        "market_regime_hmm": 1.0,
    }

    DEFAULT_WEIGHT = 1.0

    def __init__(
        self,
        min_sources_for_signal: int = 2,
        strong_confluence_threshold: float = 0.6,
        custom_weights: Optional[dict[str, float]] = None,
    ):
        self.min_sources = min_sources_for_signal
        self.strong_threshold = strong_confluence_threshold

        self._weights = dict(self.SOURCE_WEIGHTS)
        if custom_weights:
            self._weights.update(custom_weights)

        self._latest_results: dict[str, AnalysisResult] = {}

    def add_result(self, result: AnalysisResult) -> None:
        """Adiciona resultado de um analisador."""
        self._latest_results[result.source] = result

    def add_results(self, results: list[AnalysisResult]) -> None:
        """Adiciona múltiplos resultados."""
        for r in results:
            self.add_result(r)

    def calculate_confluence(self) -> ConfluenceScore:
        """
        Calcula score de confluência combinando todos os resultados.
        """
        if not self._latest_results:
            return ConfluenceScore(
                direction=Side.UNKNOWN,
                total_score=0.0,
                buy_signals=0,
                sell_signals=0,
                neutral_signals=0,
                total_signals=0,
                sources_agreeing=0,
                total_sources=0,
                confidence=0.0,
                strength=SignalStrength.NEUTRAL,
                summary="No data available",
            )

        weighted_score = 0.0
        total_weight = 0.0
        buy_count = 0
        sell_count = 0
        neutral_count = 0
        source_directions: dict[str, Side] = {}
        breakdown: dict[str, list[dict]] = defaultdict(list)

        for source, result in self._latest_results.items():
            weight = self._weights.get(source, self.DEFAULT_WEIGHT)

            for signal in result.signals:
                signal_weight = weight * signal.confidence

                if signal.direction == Side.BUY:
                    weighted_score += signal_weight
                    buy_count += 1
                    source_directions[source] = Side.BUY
                elif signal.direction == Side.SELL:
                    weighted_score -= signal_weight
                    sell_count += 1
                    source_directions[source] = Side.SELL
                else:
                    neutral_count += 1
                    if source not in source_directions:
                        source_directions[source] = Side.UNKNOWN

                total_weight += signal_weight

                breakdown[source].append({
                    "type": signal.signal_type,
                    "direction": signal.direction.value,
                    "confidence": signal.confidence,
                    "strength": signal.strength.value,
                    "description": signal.description,
                })

        # Normalizar score
        if total_weight > 0:
            normalized_score = weighted_score / total_weight
        else:
            normalized_score = 0.0

        # Clamp entre -1 e 1
        normalized_score = max(-1.0, min(1.0, normalized_score))

        # Direção final
        if normalized_score > 0.1:
            direction = Side.BUY
        elif normalized_score < -0.1:
            direction = Side.SELL
        else:
            direction = Side.UNKNOWN

        # Contar fontes concordantes
        if direction == Side.BUY:
            agreeing = sum(1 for d in source_directions.values() if d == Side.BUY)
        elif direction == Side.SELL:
            agreeing = sum(1 for d in source_directions.values() if d == Side.SELL)
        else:
            agreeing = 0

        total_sources = len(source_directions)
        total_signals = buy_count + sell_count + neutral_count

        # Confiança
        source_agreement = agreeing / max(total_sources, 1)
        score_strength = abs(normalized_score)
        confidence = (source_agreement * 0.6 + score_strength * 0.4)

        # Strength
        if confidence > 0.7 and agreeing >= 3:
            strength = SignalStrength.STRONG
        elif confidence > 0.4 and agreeing >= 2:
            strength = SignalStrength.MODERATE
        elif confidence > 0.2:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NEUTRAL

        # Summary
        dir_str = direction.value if direction != Side.UNKNOWN else "neutral"
        summary = (
            f"Confluence {dir_str.upper()}: "
            f"score={normalized_score:+.2f}, "
            f"{agreeing}/{total_sources} sources agree, "
            f"{buy_count} buy / {sell_count} sell / {neutral_count} neutral signals"
        )

        return ConfluenceScore(
            direction=direction,
            total_score=normalized_score,
            buy_signals=buy_count,
            sell_signals=sell_count,
            neutral_signals=neutral_count,
            total_signals=total_signals,
            sources_agreeing=agreeing,
            total_sources=total_sources,
            confidence=confidence,
            strength=strength,
            summary=summary,
            signal_breakdown=dict(breakdown),
        )

    def analyze(self) -> AnalysisResult:
        """Análise completa com confluência."""
        result = AnalysisResult(
            source="confluence_engine",
            timestamp=time.time(),
        )

        confluence = self.calculate_confluence()

        result.metrics = {
            "confluence_score": confluence.total_score,
            "buy_signals": confluence.buy_signals,
            "sell_signals": confluence.sell_signals,
            "neutral_signals": confluence.neutral_signals,
            "total_signals": confluence.total_signals,
            "sources_agreeing": confluence.sources_agreeing,
            "total_sources": confluence.total_sources,
            "confidence": confluence.confidence,
        }

        if confluence.total_signals >= self.min_sources:
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="confluence",
                    direction=confluence.direction,
                    strength=confluence.strength,
                    price=0,  # Preço vem dos sub-analisadores
                    confidence=confluence.confidence,
                    source="confluence_engine",
                    description=confluence.summary,
                    metadata={
                        "score": confluence.total_score,
                        "agreeing_sources": confluence.sources_agreeing,
                        "breakdown": confluence.signal_breakdown,
                    },
                )
            )

        result.confidence = confluence.confidence
        return result

    def get_layer_summary(self) -> dict[str, dict]:
        """Resumo por camada (flow, structure, statistical, sentiment)."""
        layers = {
            "order_flow": ["cvd_analyzer", "footprint_analyzer", "ofi_analyzer", "absorption_detector", "iceberg_detector"],
            "structure": ["smart_money_analyzer", "vwap_twap_analyzer"],
            "statistical": ["kalman_filter", "hurst_calculator", "garch_model", "mean_reversion", "market_regime_hmm"],
            "sentiment": ["crypto_cot", "whale_detector", "entropy_analyzer"],
            "probabilistic": ["monte_carlo", "fourier_cycles"],
        }

        summary = {}
        for layer_name, sources in layers.items():
            layer_signals = []
            for source in sources:
                if source in self._latest_results:
                    layer_signals.extend(self._latest_results[source].signals)

            buy = sum(1 for s in layer_signals if s.direction == Side.BUY)
            sell = sum(1 for s in layer_signals if s.direction == Side.SELL)

            if buy > sell:
                direction = "bullish"
            elif sell > buy:
                direction = "bearish"
            else:
                direction = "neutral"

            summary[layer_name] = {
                "direction": direction,
                "buy_signals": buy,
                "sell_signals": sell,
                "total_signals": len(layer_signals),
                "active_sources": sum(1 for s in sources if s in self._latest_results),
            }

        return summary

    def reset(self) -> None:
        """Reseta todos os resultados."""
        self._latest_results.clear()
