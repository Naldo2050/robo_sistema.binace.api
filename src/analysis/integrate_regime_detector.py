"""
Exemplo de integração do EnhancedRegimeDetector com o MacroDataProvider.
"""
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Adiciona o diretório raiz do projeto ao PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.macro_data_provider import MacroDataProvider
from src.analysis.regime_detector import EnhancedRegimeDetector


class RegimeAnalysisIntegrator:
    """
    Integra o detector de regime com o provedor de dados macro.
    """

    def __init__(self):
        self.macro_provider = MacroDataProvider()
        self.regime_detector = EnhancedRegimeDetector()

    async def analyze_current_regime(self) -> Dict[str, Any]:
        """
        Coleta dados macro e analisa o regime atual.
        """
        # Coleta dados macro
        macro_data = await self.macro_provider.get_all_macro_data()
        
        # Dados de exemplo para cross_asset_features e current_price_data
        # (Em um cenário real, esses dados viriam de outras fontes)
        cross_asset_features = {
            "correlation_spy": 0.7,  # Exemplo: correlação com SPY
            "btc_dxy_corr_30d": -0.3,  # Exemplo: correlação com DXY
            "dxy_momentum": 0.5  # Exemplo: momentum do DXY
        }
        
        current_price_data = {
            "btc_price": 50000.0,  # Exemplo: preço atual do BTC
            "eth_price": 3000.0   # Exemplo: preço atual do ETH
        }
        
        # Analisa o regime
        regime_analysis = self.regime_detector.detect_regime(
            macro_data=macro_data,
            cross_asset_features=cross_asset_features,
            current_price_data=current_price_data
        )
        
        # Formata o resultado para inclusão no payload do AI
        result = {
            "macro_data": macro_data,
            "regime_analysis": {
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
        }
        
        return result


async def main():
    """
    Exemplo de uso do integrador.
    """
    integrator = RegimeAnalysisIntegrator()
    result = await integrator.analyze_current_regime()
    
    print("Resultado da Análise de Regime:")
    print("=" * 80)
    print(f"Market Regime: {result['regime_analysis']['market_regime']}")
    print(f"Correlation Regime: {result['regime_analysis']['correlation_regime']}")
    print(f"Volatility Regime: {result['regime_analysis']['volatility_regime']}")
    print(f"Risk Score: {result['regime_analysis']['risk_score']:.2f}")
    print(f"Fear & Greed Proxy: {result['regime_analysis']['fear_greed_proxy']:.2f}")
    print(f"Regime Change Warning: {result['regime_analysis']['regime_change_warning']}")
    print(f"Divergence Alert: {result['regime_analysis']['divergence_alert']}")
    print(f"Primary Driver: {result['regime_analysis']['primary_driver']}")
    print("=" * 80)
    print("Signals Summary:")
    for key, value in result['regime_analysis']['signals_summary'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
