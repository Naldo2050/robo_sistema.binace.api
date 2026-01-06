"""
Integração das RegimeBasedRules no fluxo de decisão de trading.
"""
from typing import Dict, Any, Optional
from src.rules.regime_rules import RegimeBasedRules, TradeRecommendation
import logging

logger = logging.getLogger(__name__)


class RegimeIntegration:
    """
    Classe para integrar as regras de regime no fluxo de decisão de trading.
    """
    
    def __init__(self):
        self.regime_rules = RegimeBasedRules()
    
    def process_signal_with_regime(
        self,
        signal: Dict[str, Any],
        ai_payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Processa um sinal de trading considerando o regime de mercado.
        
        Args:
            signal: Dados do sinal de trading.
            ai_payload: Payload da IA contendo regime_analysis.
            
        Returns:
            Sinal ajustado ou None se não deve operar.
        """
        regime_analysis = ai_payload.get("regime_analysis", {})
        
        # 1. Verificar se deve operar
        should_trade, reason = self.regime_rules.should_trade(
            regime_analysis=regime_analysis,
            signal_direction=signal.get("direction", "long"),
            signal_confidence=signal.get("confidence", 0.0)
        )
        
        if not should_trade:
            logger.info(f"Trade bloqueado pelo regime: {reason}")
            return None
        
        # 2. Calcular position size ajustado
        adjusted_size = self.regime_rules.calculate_position_size(
            regime_analysis=regime_analysis,
            base_size=signal.get("suggested_size", 1.0),
            account_balance=signal.get("account_balance", 10000.0)
        )
        
        # 3. Calcular stops ajustados
        stop_loss = self.regime_rules.calculate_stop_loss(
            regime_analysis=regime_analysis,
            entry_price=signal.get("entry_price", 0.0),
            direction=signal.get("direction", "long"),
            atr=signal.get("atr", 1.0)
        )
        
        take_profit = self.regime_rules.calculate_take_profit(
            regime_analysis=regime_analysis,
            entry_price=signal.get("entry_price", 0.0),
            direction=signal.get("direction", "long"),
            atr=signal.get("atr", 1.0)
        )
        
        # 4. Log do regime
        logger.info(self.regime_rules.format_regime_summary(regime_analysis))
        
        # 5. Retornar sinal ajustado
        adjusted_signal = signal.copy()
        adjusted_signal["adjusted_size"] = adjusted_size
        adjusted_signal["stop_loss"] = stop_loss
        adjusted_signal["take_profit"] = take_profit
        adjusted_signal["regime_adjustment"] = self.regime_rules.get_regime_adjustment(regime_analysis)
        
        return adjusted_signal
    
    def get_trade_recommendation(
        self,
        regime_analysis: Dict[str, Any],
        flow_signal: str,
        technical_signal: str,
        confidence: float
    ) -> TradeRecommendation:
        """
        Obtém recomendação de trading baseada no regime.
        """
        return self.regime_rules.get_trade_recommendation(
            regime_analysis=regime_analysis,
            flow_signal=flow_signal,
            technical_signal=technical_signal,
            confidence=confidence
        )