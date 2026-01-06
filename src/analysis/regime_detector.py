"""
Detector de Regime de Mercado baseado em múltiplas fontes.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    RISK_ON = "RISK_ON"           # Apetite por risco alto
    RISK_OFF = "RISK_OFF"         # Flight to safety
    TRANSITION = "TRANSITION"     # Mudando de regime
    UNCERTAIN = "UNCERTAIN"       # Sinais mistos


class CorrelationRegime(Enum):
    MACRO_CORRELATED = "MACRO_CORRELATED"     # BTC seguindo macro
    CRYPTO_NATIVE = "CRYPTO_NATIVE"           # BTC descolado
    INVERSE_MACRO = "INVERSE_MACRO"           # BTC inverso ao macro


class VolatilityRegime(Enum):
    LOW_VOL = "LOW_VOL"           # VIX < 15
    NORMAL_VOL = "NORMAL_VOL"    # VIX 15-25
    HIGH_VOL = "HIGH_VOL"        # VIX 25-35
    EXTREME_VOL = "EXTREME_VOL"  # VIX > 35


@dataclass
class RegimeAnalysis:
    """Resultado completo da análise de regime"""
    market_regime: MarketRegime
    correlation_regime: CorrelationRegime
    volatility_regime: VolatilityRegime
    
    # Scores de confiança (0-1)
    regime_confidence: float
    regime_stability: float  # Quanto tempo no mesmo regime
    
    # Indicadores específicos
    risk_score: float  # -1 (risk off) a +1 (risk on)
    fear_greed_proxy: float  # Baseado em VIX e dominance
    
    # Alertas
    regime_change_warning: bool
    divergence_alert: bool  # BTC divergindo do esperado
    
    # Metadata
    primary_driver: str  # O que está dirigindo o regime
    signals_summary: Dict[str, str]


class EnhancedRegimeDetector:
    """
    Detector de regime usando múltiplas fontes:
    - VIX (medo/volatilidade)
    - Treasury Yields (risk appetite)
    - Dominance (flight to safety crypto)
    - Correlações (macro alignment)
    """
    
    def __init__(self):
        self.regime_history = []
        self.max_history = 100
        
    def detect_regime(
        self,
        macro_data: Dict[str, Any],
        cross_asset_features: Dict[str, Any],
        current_price_data: Dict[str, Any]
    ) -> RegimeAnalysis:
        """
        Analisa múltiplas fontes para determinar regime atual.
        
        Args:
            macro_data: Output do MacroDataProvider
            cross_asset_features: ml_features.cross_asset
            current_price_data: Dados de preço atual
        """
        
        # 1. Análise de Volatilidade (VIX)
        vol_regime = self._analyze_volatility(macro_data)
        
        # 2. Análise de Risk Appetite (Yields, Gold)
        risk_score = self._calculate_risk_score(macro_data)
        
        # 3. Análise de Correlação
        corr_regime = self._analyze_correlation_regime(cross_asset_features)
        
        # 4. Análise de Dominance (Crypto-specific)
        crypto_fear = self._analyze_dominance(macro_data)
        
        # 5. Determinar Market Regime
        market_regime = self._determine_market_regime(
            vol_regime, risk_score, crypto_fear
        )
        
        # 6. Detectar mudanças de regime
        regime_change = self._detect_regime_change(market_regime)
        
        # 7. Detectar divergências
        divergence = self._detect_divergence(
            market_regime, corr_regime, current_price_data
        )
        
        return RegimeAnalysis(
            market_regime=market_regime,
            correlation_regime=corr_regime,
            volatility_regime=vol_regime,
            regime_confidence=self._calculate_confidence(),
            regime_stability=self._calculate_stability(),
            risk_score=risk_score,
            fear_greed_proxy=self._calculate_fear_greed(vol_regime, crypto_fear),
            regime_change_warning=regime_change,
            divergence_alert=divergence,
            primary_driver=self._identify_driver(macro_data),
            signals_summary=self._summarize_signals(macro_data, cross_asset_features)
        )
    
    def _analyze_volatility(self, macro_data: Dict) -> VolatilityRegime:
        """Classifica regime de volatilidade baseado no VIX"""
        vix = macro_data.get("vix")
        if vix is None:
            return VolatilityRegime.NORMAL_VOL
        
        if vix < 15:
            return VolatilityRegime.LOW_VOL
        elif vix < 25:
            return VolatilityRegime.NORMAL_VOL
        elif vix < 35:
            return VolatilityRegime.HIGH_VOL
        else:
            return VolatilityRegime.EXTREME_VOL
    
    def _calculate_risk_score(self, macro_data: Dict) -> float:
        """
        Calcula score de -1 (risk off) a +1 (risk on)
        
        Risk ON indicators:
        - VIX baixo
        - Yields subindo (economia forte)
        - Gold caindo
        - BTC dominance caindo (altseason)
        
        Risk OFF indicators:
        - VIX alto
        - Yields caindo (flight to safety)
        - Gold subindo
        - USDT dominance subindo
        """
        score = 0.0
        weights_used = 0
        
        # VIX component (-1 se alto, +1 se baixo)
        vix = macro_data.get("vix")
        if vix is not None:
            if vix < 15:
                score += 1.0
            elif vix < 20:
                score += 0.5
            elif vix < 25:
                score += 0.0
            elif vix < 35:
                score -= 0.5
            else:
                score -= 1.0
            weights_used += 1
        
        # Yield curve (10y - 2y)
        y10 = macro_data.get("treasury_10y")
        y2 = macro_data.get("treasury_2y")
        if y10 is not None and y2 is not None:
            spread = y10 - y2
            if spread > 0.5:
                score += 0.5  # Curva normal = risk on
            elif spread < 0:
                score -= 0.5  # Curva invertida = risk off
            weights_used += 1
        
        # USDT Dominance (flight to safety)
        usdt_dom = macro_data.get("usdt_dominance")
        if usdt_dom is not None:
            if usdt_dom > 8:
                score -= 0.5  # Alto USDT.D = medo
            elif usdt_dom < 5:
                score += 0.5  # Baixo USDT.D = risk on
            weights_used += 1
        
        return score / max(weights_used, 1)
    
    def _analyze_correlation_regime(
        self, 
        cross_asset: Dict
    ) -> CorrelationRegime:
        """Determina se BTC está seguindo macro ou não"""
        
        spy_corr = cross_asset.get("correlation_spy", 0)
        dxy_corr = cross_asset.get("btc_dxy_corr_30d", 0)
        
        # Se correlação com SPY forte (>0.5 ou <-0.5)
        if abs(spy_corr) > 0.5:
            if spy_corr > 0:
                return CorrelationRegime.MACRO_CORRELATED
            else:
                return CorrelationRegime.INVERSE_MACRO
        
        # Se correlação fraca com tudo
        if abs(spy_corr) < 0.2 and abs(dxy_corr) < 0.2:
            return CorrelationRegime.CRYPTO_NATIVE
        
        return CorrelationRegime.MACRO_CORRELATED
    
    def _analyze_dominance(self, macro_data: Dict) -> float:
        """Analisa dominance para detectar medo/ganância crypto"""
        btc_dom = macro_data.get("btc_dominance", 50)
        usdt_dom = macro_data.get("usdt_dominance", 5)
        
        # BTC.D alto + USDT.D alto = medo extremo
        fear = 0.0
        if btc_dom > 55:
            fear += 0.3
        if usdt_dom > 7:
            fear += 0.4
        
        # ETH.D alto = risk on (altseason)
        eth_dom = macro_data.get("eth_dominance", 15)
        if eth_dom > 18:
            fear -= 0.3
        
        return max(-1, min(1, fear))  # Clamp -1 to 1
    
    def _determine_market_regime(
        self,
        vol_regime: VolatilityRegime,
        risk_score: float,
        crypto_fear: float
    ) -> MarketRegime:
        """Combina indicadores para regime final"""
        
        # Volatilidade extrema = sempre RISK_OFF
        if vol_regime == VolatilityRegime.EXTREME_VOL:
            return MarketRegime.RISK_OFF
        
        # Score consolidado
        total_score = risk_score - crypto_fear
        
        if total_score > 0.3:
            return MarketRegime.RISK_ON
        elif total_score < -0.3:
            return MarketRegime.RISK_OFF
        else:
            return MarketRegime.TRANSITION
    
    def _detect_regime_change(self, current: MarketRegime) -> bool:
        """Detecta se regime está mudando"""
        if len(self.regime_history) < 3:
            self.regime_history.append(current)
            return False
        
        # Manter histórico limitado
        self.regime_history.append(current)
        if len(self.regime_history) > self.max_history:
            self.regime_history.pop(0)
        
        # Se últimos 3 diferentes do anterior
        recent = self.regime_history[-3:]
        if len(set(recent)) > 1:
            return True
        
        return False
    
    def _detect_divergence(
        self,
        regime: MarketRegime,
        corr_regime: CorrelationRegime,
        price_data: Dict
    ) -> bool:
        """Detecta se BTC está divergindo do esperado"""
        # TODO: Implementar lógica de divergência
        return False
    
    def _calculate_confidence(self) -> float:
        """Calcula confiança no regime detectado"""
        if len(self.regime_history) < 5:
            return 0.5
        
        # Quanto mais consistente, maior confiança
        recent = self.regime_history[-5:]
        most_common = max(set(recent), key=recent.count)
        return recent.count(most_common) / 5
    
    def _calculate_stability(self) -> float:
        """Calcula estabilidade do regime"""
        if len(self.regime_history) < 2:
            return 0.0
        
        # Contar quantos iguais consecutivos
        current = self.regime_history[-1]
        count = 0
        for r in reversed(self.regime_history):
            if r == current:
                count += 1
            else:
                break
        
        return min(count / 10, 1.0)  # Normaliza para 0-1
    
    def _calculate_fear_greed(
        self,
        vol_regime: VolatilityRegime,
        crypto_fear: float
    ) -> float:
        """Proxy para Fear & Greed Index (-1 fear, +1 greed)"""
        vol_score = {
            VolatilityRegime.LOW_VOL: 0.8,
            VolatilityRegime.NORMAL_VOL: 0.3,
            VolatilityRegime.HIGH_VOL: -0.3,
            VolatilityRegime.EXTREME_VOL: -0.8
        }.get(vol_regime, 0)
        
        return (vol_score - crypto_fear) / 2
    
    def _identify_driver(self, macro_data: Dict) -> str:
        """Identifica principal driver do regime"""
        vix = macro_data.get("vix")
        if vix and vix > 30:
            return "VIX_ELEVATED"
        
        usdt_dom = macro_data.get("usdt_dominance")
        if usdt_dom and usdt_dom > 8:
            return "CRYPTO_FEAR"
        
        return "MIXED_SIGNALS"
    
    def _summarize_signals(
        self,
        macro_data: Dict,
        cross_asset: Dict
    ) -> Dict[str, str]:
        """Resume todos os sinais para debug"""
        return {
            "vix": f"{macro_data.get('vix', 'N/A')}",
            "btc_dominance": f"{macro_data.get('btc_dominance', 'N/A'):.1f}%",
            "usdt_dominance": f"{macro_data.get('usdt_dominance', 'N/A')}%",
            "spy_correlation": f"{cross_asset.get('correlation_spy', 'N/A'):.2f}",
            "dxy_momentum": f"{cross_asset.get('dxy_momentum', 'N/A')}"
        }