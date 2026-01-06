"""
Integração do resultado da análise de regime no payload do AI.
"""
import sys
from pathlib import Path
from typing import Dict, Any

# Adiciona o diretório raiz do projeto ao PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.regime_detector import RegimeAnalysis


class AIPayloadIntegrator:
    """
    Integra os resultados da análise de regime no payload do AI.
    """

    @staticmethod
    def integrate_regime_analysis_into_payload(
        base_payload: Dict[str, Any],
        regime_analysis: RegimeAnalysis
    ) -> Dict[str, Any]:
        """
        Adiciona a análise de regime ao payload do AI.

        Args:
            base_payload: Payload base do AI.
            regime_analysis: Resultado da análise de regime.

        Returns:
            Payload atualizado com a análise de regime.
        """
        # Adiciona a análise de regime ao payload
        base_payload["regime_analysis"] = {
            "market_regime": regime_analysis.market_regime.value,
            "correlation_regime": regime_analysis.correlation_regime.value,
            "volatility_regime": regime_analysis.volatility_regime.value,
            "regime_confidence": regime_analysis.regime_confidence,
            "regime_stability": regime_analysis.regime_stability,
            "risk_score": regime_analysis.risk_score,
            "fear_greed_proxy": regime_analysis.fear_greed_proxy,
            "regime_change_warning": regime_analysis.regime_change_warning,
            "divergence_alert": regime_analysis.divergence_alert,
            "primary_driver": regime_analysis.primary_driver,
            "signals_summary": regime_analysis.signals_summary
        }

        return base_payload


# Exemplo de uso
def example_usage():
    """
    Exemplo de como integrar a análise de regime no payload do AI.
    """
    # Payload base do AI (exemplo)
    base_payload = {
        "timestamp": "2026-01-05T00:00:00Z",
        "market_data": {
            "btc_price": 50000.0,
            "eth_price": 3000.0
        },
        "technical_indicators": {
            "rsi": 55.0,
            "macd": 120.0
        }
    }

    # Simula uma análise de regime (em um cenário real, isso viria do EnhancedRegimeDetector)
    from src.analysis.regime_detector import MarketRegime, CorrelationRegime, VolatilityRegime, RegimeAnalysis

    regime_analysis = RegimeAnalysis(
        market_regime=MarketRegime.RISK_ON,
        correlation_regime=CorrelationRegime.MACRO_CORRELATED,
        volatility_regime=VolatilityRegime.LOW_VOL,
        regime_confidence=0.85,
        regime_stability=0.9,
        risk_score=0.7,
        fear_greed_proxy=0.6,
        regime_change_warning=False,
        divergence_alert=False,
        primary_driver="MIXED_SIGNALS",
        signals_summary={
            "vix": "12.5",
            "btc_dominance": "45.0%",
            "usdt_dominance": "5.0%",
            "spy_correlation": "0.70",
            "dxy_momentum": "0.5"
        }
    )

    # Integra a análise de regime no payload
    integrator = AIPayloadIntegrator()
    updated_payload = integrator.integrate_regime_analysis_into_payload(
        base_payload, regime_analysis
    )

    # Exibe o resultado
    print("Payload do AI com Análise de Regime:")
    print("=" * 80)
    import json
    print(json.dumps(updated_payload, indent=2))


if __name__ == "__main__":
    example_usage()
