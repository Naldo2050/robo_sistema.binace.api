"""
Regras de trading baseadas no regime de mercado.
Ajusta comportamento, sizing e filtros baseado no contexto macro.
"""
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradeRecommendation(Enum):
    STRONG_LONG = "STRONG_LONG"
    LONG = "LONG"
    WEAK_LONG = "WEAK_LONG"
    NEUTRAL = "NEUTRAL"
    WEAK_SHORT = "WEAK_SHORT"
    SHORT = "SHORT"
    STRONG_SHORT = "STRONG_SHORT"
    NO_TRADE = "NO_TRADE"


@dataclass
class RegimeAdjustment:
    """Ajustes baseados no regime"""
    # Multiplicadores de posição (1.0 = normal)
    position_size_multiplier: float
    
    # Ajustes de stop/target
    stop_loss_multiplier: float  # >1 = stop mais largo
    take_profit_multiplier: float
    
    # Filtros
    min_confidence_required: float
    allowed_directions: list  # ["long", "short", "both", "none"]
    
    # Comportamento
    aggressive_entries: bool
    scale_in_allowed: bool
    scale_out_required: bool
    
    # Alertas
    extra_confirmation_needed: bool
    regime_warning: Optional[str]


class RegimeBasedRules:
    """
    Aplica regras de trading baseadas no regime detectado.
    
    Filosofia:
    - RISK_ON + LOW_VOL: Mais agressivo, posições maiores
    - RISK_OFF + HIGH_VOL: Conservador, posições menores
    - TRANSITION: Cauteloso, aguardar confirmação
    - CRYPTO_NATIVE: Seguir fluxo crypto, ignorar macro
    """
    
    def __init__(self):
        self.base_position_size = 1.0
        self.base_stop_pct = 0.02  # 2%
        self.base_target_pct = 0.04  # 4%
        
    def get_regime_adjustment(
        self,
        regime_analysis: Dict[str, Any]
    ) -> RegimeAdjustment:
        """
        Retorna ajustes baseados no regime atual.
        
        Args:
            regime_analysis: Output do EnhancedRegimeDetector
        """
        market_regime = regime_analysis.get("market_regime", "UNCERTAIN")
        vol_regime = regime_analysis.get("volatility_regime", "NORMAL_VOL")
        corr_regime = regime_analysis.get("correlation_regime", "MACRO_CORRELATED")
        risk_score = regime_analysis.get("risk_score", 0)
        regime_change = regime_analysis.get("regime_change_warning", False)
        
        # ========== MATRIZ DE DECISÃO ==========
        
        # 1. RISK_ON + LOW_VOL = Condições ideais
        if market_regime == "RISK_ON" and vol_regime == "LOW_VOL":
            return RegimeAdjustment(
                position_size_multiplier=1.25,  # 25% maior
                stop_loss_multiplier=0.8,       # Stop mais tight
                take_profit_multiplier=1.2,     # Target maior
                min_confidence_required=0.55,   # Threshold menor
                allowed_directions=["long", "short"],
                aggressive_entries=True,
                scale_in_allowed=True,
                scale_out_required=False,
                extra_confirmation_needed=False,
                regime_warning=None
            )
        
        # 2. RISK_ON + NORMAL_VOL = Bom momento
        elif market_regime == "RISK_ON" and vol_regime == "NORMAL_VOL":
            return RegimeAdjustment(
                position_size_multiplier=1.0,
                stop_loss_multiplier=1.0,
                take_profit_multiplier=1.0,
                min_confidence_required=0.60,
                allowed_directions=["long", "short"],
                aggressive_entries=False,
                scale_in_allowed=True,
                scale_out_required=False,
                extra_confirmation_needed=False,
                regime_warning=None
            )
        
        # 3. RISK_OFF + qualquer VOL = Cuidado
        elif market_regime == "RISK_OFF":
            # Ajuste extra baseado na volatilidade
            if vol_regime in ["HIGH_VOL", "EXTREME_VOL"]:
                return RegimeAdjustment(
                    position_size_multiplier=0.5,  # Metade do tamanho
                    stop_loss_multiplier=1.5,      # Stop mais largo
                    take_profit_multiplier=0.8,    # Target menor (take profit rápido)
                    min_confidence_required=0.75,  # Alta confiança necessária
                    allowed_directions=["short"],  # Apenas shorts
                    aggressive_entries=False,
                    scale_in_allowed=False,
                    scale_out_required=True,       # Obrigatório sair parcial
                    extra_confirmation_needed=True,
                    regime_warning="⚠️ RISK_OFF + HIGH_VOL: Operar com extrema cautela"
                )
            else:
                return RegimeAdjustment(
                    position_size_multiplier=0.75,
                    stop_loss_multiplier=1.2,
                    take_profit_multiplier=0.9,
                    min_confidence_required=0.65,
                    allowed_directions=["long", "short"],
                    aggressive_entries=False,
                    scale_in_allowed=False,
                    scale_out_required=True,
                    extra_confirmation_needed=True,
                    regime_warning="⚠️ RISK_OFF: Reduzir exposição"
                )
        
        # 4. TRANSITION = Aguardar
        elif market_regime == "TRANSITION":
            return RegimeAdjustment(
                position_size_multiplier=0.5,
                stop_loss_multiplier=1.3,
                take_profit_multiplier=0.8,
                min_confidence_required=0.70,
                allowed_directions=["long", "short"],
                aggressive_entries=False,
                scale_in_allowed=False,
                scale_out_required=True,
                extra_confirmation_needed=True,
                regime_warning="⏳ TRANSITION: Regime mudando, aguardar confirmação"
            )
        
        # 5. EXTREME_VOL = Não operar
        elif vol_regime == "EXTREME_VOL":
            return RegimeAdjustment(
                position_size_multiplier=0.0,  # NÃO OPERAR
                stop_loss_multiplier=2.0,
                take_profit_multiplier=0.5,
                min_confidence_required=0.90,
                allowed_directions=["none"],
                aggressive_entries=False,
                scale_in_allowed=False,
                scale_out_required=True,
                extra_confirmation_needed=True,
                regime_warning="🚫 EXTREME_VOL: VIX muito alto, não operar"
            )
        
        # 6. CRYPTO_NATIVE = Seguir fluxo interno
        elif corr_regime == "CRYPTO_NATIVE":
            return RegimeAdjustment(
                position_size_multiplier=1.0,
                stop_loss_multiplier=1.0,
                take_profit_multiplier=1.0,
                min_confidence_required=0.60,
                allowed_directions=["long", "short"],
                aggressive_entries=False,
                scale_in_allowed=True,
                scale_out_required=False,
                extra_confirmation_needed=False,
                regime_warning="🔗 CRYPTO_NATIVE: BTC descolado do macro, seguir fluxo"
            )
        
        # 7. Default = Conservador
        else:
            return RegimeAdjustment(
                position_size_multiplier=0.75,
                stop_loss_multiplier=1.2,
                take_profit_multiplier=1.0,
                min_confidence_required=0.65,
                allowed_directions=["long", "short"],
                aggressive_entries=False,
                scale_in_allowed=False,
                scale_out_required=False,
                extra_confirmation_needed=False,
                regime_warning=None
            )
    
    def should_trade(
        self,
        regime_analysis: Dict[str, Any],
        signal_direction: str,
        signal_confidence: float
    ) -> Tuple[bool, str]:
        """
        Verifica se deve operar baseado no regime.
        
        Returns:
            (should_trade, reason)
        """
        adjustment = self.get_regime_adjustment(regime_analysis)
        
        # 1. Checar se direção é permitida
        if signal_direction not in adjustment.allowed_directions:
            if "none" in adjustment.allowed_directions:
                return False, f"Regime não permite operações: {adjustment.regime_warning}"
            return False, f"Direção {signal_direction} não permitida no regime atual"
        
        # 2. Checar confiança mínima
        if signal_confidence < adjustment.min_confidence_required:
            return False, f"Confiança {signal_confidence:.2f} < {adjustment.min_confidence_required:.2f} requerida"
        
        # 3. Checar regime change warning
        if regime_analysis.get("regime_change_warning", False):
            if adjustment.extra_confirmation_needed:
                return False, "Regime em transição, aguardando confirmação"
        
        # 4. Checar position size multiplier = 0
        if adjustment.position_size_multiplier == 0:
            return False, "Position size = 0, regime não permite operação"
        
        return True, "OK"
    
    def calculate_position_size(
        self,
        regime_analysis: Dict[str, Any],
        base_size: float,
        account_balance: float
    ) -> float:
        """Calcula tamanho da posição ajustado pelo regime"""
        adjustment = self.get_regime_adjustment(regime_analysis)
        adjusted_size = base_size * adjustment.position_size_multiplier
        
        # Cap máximo de 2x o base size
        max_size = base_size * 2.0
        adjusted_size = min(adjusted_size, max_size)
        
        # Cap pelo balance
        max_by_balance = account_balance * 0.1  # Max 10% do balance
        adjusted_size = min(adjusted_size, max_by_balance)
        
        return adjusted_size
    
    def calculate_stop_loss(
        self,
        regime_analysis: Dict[str, Any],
        entry_price: float,
        direction: str,
        atr: float
    ) -> float:
        """Calcula stop loss ajustado pelo regime"""
        adjustment = self.get_regime_adjustment(regime_analysis)
        
        # Base stop = 1.5 ATR
        base_stop_distance = atr * 1.5
        adjusted_distance = base_stop_distance * adjustment.stop_loss_multiplier
        
        if direction == "long":
            return entry_price - adjusted_distance
        else:
            return entry_price + adjusted_distance
    
    def calculate_take_profit(
        self,
        regime_analysis: Dict[str, Any],
        entry_price: float,
        direction: str,
        atr: float
    ) -> float:
        """Calcula take profit ajustado pelo regime"""
        adjustment = self.get_regime_adjustment(regime_analysis)
        
        # Base target = 3 ATR (R:R de 1:2)
        base_target_distance = atr * 3.0
        adjusted_distance = base_target_distance * adjustment.take_profit_multiplier
        
        if direction == "long":
            return entry_price + adjusted_distance
        else:
            return entry_price - adjusted_distance
    
    def get_trade_recommendation(
        self,
        regime_analysis: Dict[str, Any],
        flow_signal: str,  # "bullish", "bearish", "neutral"
        technical_signal: str,  # "bullish", "bearish", "neutral"
        confidence: float
    ) -> TradeRecommendation:
        """
        Combina regime + sinais para recomendação final.
        """
        market_regime = regime_analysis.get("market_regime", "UNCERTAIN")
        risk_score = regime_analysis.get("risk_score", 0)
        
        # Não operar em condições extremas
        if regime_analysis.get("volatility_regime") == "EXTREME_VOL":
            return TradeRecommendation.NO_TRADE
        
        # Calcular direção base
        flow_score = {"bullish": 1, "neutral": 0, "bearish": -1}.get(flow_signal, 0)
        tech_score = {"bullish": 1, "neutral": 0, "bearish": -1}.get(technical_signal, 0)
        
        combined_score = (flow_score * 0.6) + (tech_score * 0.4)  # Flow tem mais peso
        
        # Ajustar pelo regime
        if market_regime == "RISK_ON":
            combined_score *= 1.2  # Amplifica sinal
        elif market_regime == "RISK_OFF":
            combined_score *= 0.8  # Atenua sinal
        elif market_regime == "TRANSITION":
            combined_score *= 0.5  # Muito conservador
        
        # Mapear para recomendação
        if combined_score > 0.8 and confidence > 0.7:
            return TradeRecommendation.STRONG_LONG
        elif combined_score > 0.5:
            return TradeRecommendation.LONG
        elif combined_score > 0.2:
            return TradeRecommendation.WEAK_LONG
        elif combined_score < -0.8 and confidence > 0.7:
            return TradeRecommendation.STRONG_SHORT
        elif combined_score < -0.5:
            return TradeRecommendation.SHORT
        elif combined_score < -0.2:
            return TradeRecommendation.WEAK_SHORT
        else:
            return TradeRecommendation.NEUTRAL
    
    def format_regime_summary(
        self,
        regime_analysis: Dict[str, Any]
    ) -> str:
        """Formata resumo do regime para logs/display"""
        adjustment = self.get_regime_adjustment(regime_analysis)
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    REGIME ANALYSIS                           ║
╠══════════════════════════════════════════════════════════════╣
║ Market Regime:      {regime_analysis.get('market_regime', 'N/A'):>20} ║
║ Volatility Regime:  {regime_analysis.get('volatility_regime', 'N/A'):>20} ║
║ Correlation Regime: {regime_analysis.get('correlation_regime', 'N/A'):>20} ║
║ Risk Score:         {regime_analysis.get('risk_score', 0):>20.2f} ║
║ Fear/Greed Proxy:   {regime_analysis.get('fear_greed_proxy', 0):>20.2f} ║
╠══════════════════════════════════════════════════════════════╣
║                    TRADING ADJUSTMENTS                       ║
╠══════════════════════════════════════════════════════════════╣
║ Position Size:      {adjustment.position_size_multiplier:>19.0%} ║
║ Stop Multiplier:    {adjustment.stop_loss_multiplier:>19.1f}x ║
║ Target Multiplier:  {adjustment.take_profit_multiplier:>19.1f}x ║
║ Min Confidence:     {adjustment.min_confidence_required:>19.0%} ║
║ Allowed Directions: {', '.join(adjustment.allowed_directions):>20} ║
║ Scale-In Allowed:   {str(adjustment.scale_in_allowed):>20} ║
╠══════════════════════════════════════════════════════════════╣
"""
        if adjustment.regime_warning:
            summary += f"║ ⚠️ WARNING: {adjustment.regime_warning:<47} ║\n"
        
        summary += "╚══════════════════════════════════════════════════════════════╝"
        
        return summary