"""
Modelo de Dados Ideal para Análise de Mercado
Data: 03/01/2026
Objetivo: Propor estrutura unificada com todos os blocos desejados
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class AnalysisRegime(str, Enum):
    """Enum para regimes de mercado"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"


class AlertLevel(str, Enum):
    """Enum para níveis de alerta"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VolumeProfile(BaseModel):
    """Dados do perfil de volume"""
    price_levels: Dict[str, float] = Field(description="Preço -> Volume")
    volume_at_price: Dict[str, float] = Field(description="Volume por preço")
    total_volume: float = Field(description="Volume total")
    poc_price: float = Field(description="Point of Control")


class SupportResistance(BaseModel):
    """Dados de suporte e resistência"""
    support_levels: List[float] = Field(description="Níveis de suporte")
    resistance_levels: List[float] = Field(description="Níveis de resistência")
    dynamic_support: List[float] = Field(description="Suporte dinâmico")
    dynamic_resistance: List[float] = Field(description="Resistência dinâmica")
    confidence_scores: Dict[str, float] = Field(description="Score de confiança")


class TechnicalIndicators(BaseModel):
    """Indicadores técnicos"""
    rsi: float = Field(description="RSI")
    macd: Dict[str, float] = Field(description="MACD values")
    bollinger_bands: Dict[str, float] = Field(description="Bandas de Bollinger")
    moving_averages: Dict[str, float] = Field(description="Médias móveis")
    stochastic: Dict[str, float] = Field(description="Estocástico")


class VolatilityMetrics(BaseModel):
    """Métricas de volatilidade"""
    current_volatility: float = Field(description="Volatilidade atual")
    historical_volatility: float = Field(description="Volatilidade histórica")
    implied_volatility: float = Field(description="Volatilidade implícita")
    volatility_percentile: float = Field(description="Percentil de volatilidade")


class MarketAnalysisData(BaseModel):
    """
    Modelo principal unificado para dados de análise de mercado
    Contém todos os 22 blocos solicitados na auditoria
    """
    
    # Blocos de metadados e contexto
    metadata: Dict[str, Union[str, int, float]] = Field(
        default_factory=dict,
        description="Metadados da análise (timestamp, versão, etc.)"
    )
    
    data_source: Dict[str, str] = Field(
        default_factory=dict,
        description="Informações sobre fonte de dados"
    )
    
    market_context: Dict[str, Union[str, float, bool]] = Field(
        default_factory=dict,
        description="Contexto geral do mercado"
    )
    
    # Blocos de dados de preço
    price_data: Dict[str, Union[float, List[float]]] = Field(
        default_factory=dict,
        description="Dados de preço (atual, histórico, etc.)"
    )
    
    support_resistance: Optional[SupportResistance] = Field(
        default=None,
        description="Níveis de suporte e resistência"
    )
    
    defense_zones: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Zonas de defesa/preço"
    )
    
    # Blocos de volume
    volume_profile: Optional[VolumeProfile] = Field(
        default=None,
        description="Perfil de volume"
    )
    
    volume_nodes: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Nodes de volume"
    )
    
    # Blocos de order book
    order_book_depth: Dict[str, Union[int, float, List]] = Field(
        default_factory=dict,
        description="Profundidade do order book"
    )
    
    spread_analysis: Dict[str, float] = Field(
        default_factory=dict,
        description="Análise de spread"
    )
    
    order_flow: Dict[str, Union[int, float]] = Field(
        default_factory=dict,
        description="Fluxo de ordens"
    )
    
    # Blocos de análise de participantes
    participant_analysis: Dict[str, Union[str, float, int]] = Field(
        default_factory=dict,
        description="Análise de participantes"
    )
    
    whale_activity: Dict[str, Union[bool, float, int]] = Field(
        default_factory=dict,
        description="Atividade de grandes investidores"
    )
    
    # Blocos técnicos
    technical_indicators: Optional[TechnicalIndicators] = Field(
        default=None,
        description="Indicadores técnicos"
    )
    
    volatility_metrics: Optional[VolatilityMetrics] = Field(
        default=None,
        description="Métricas de volatilidade"
    )
    
    pattern_recognition: Dict[str, Union[str, float, bool]] = Field(
        default_factory=dict,
        description="Reconhecimento de padrões"
    )
    
    absorption_analysis: Dict[str, Union[float, bool]] = Field(
        default_factory=dict,
        description="Análise de absorção"
    )
    
    market_impact: Dict[str, float] = Field(
        default_factory=dict,
        description="Impacto no mercado"
    )
    
    # Blocos de ML
    ml_features: Dict[str, Union[float, int, bool]] = Field(
        default_factory=dict,
        description="Features para machine learning"
    )
    
    # Blocos de alertas e targets
    alerts: List[Dict[str, Union[str, AlertLevel, datetime]]] = Field(
        default_factory=list,
        description="Lista de alertas"
    )
    
    price_targets: Dict[str, float] = Field(
        default_factory=dict,
        description="Alvos de preço"
    )
    
    # Blocos de regime
    regime_analysis: Dict[str, Union[AnalysisRegime, float, str]] = Field(
        default_factory=dict,
        description="Análise de regime de mercado"
    )


# Função helper para criar instância com valores padrão
def create_market_analysis_template() -> MarketAnalysisData:
    """
    Cria uma instância template do modelo com valores padrão
    para facilitar a implementação
    """
    return MarketAnalysisData(
        metadata={
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "analysis_type": "comprehensive"
        },
        data_source={
            "provider": "internal",
            "quality": "high"
        },
        market_context={
            "market_phase": "active",
            "volatility_level": "medium"
        }
    )


# Exemplo de uso
if __name__ == "__main__":
    # Criar instância template
    analysis = create_market_analysis_template()
    
    # Preencher alguns campos
    analysis.price_data = {
        "current_price": 45000.0,
        "24h_change": 2.5,
        "volume_24h": 1000000.0
    }
    
    analysis.support_resistance = SupportResistance(
        support_levels=[44000.0, 43000.0],
        resistance_levels=[46000.0, 47000.0],
        dynamic_support=[44500.0],
        dynamic_resistance=[45500.0],
        confidence_scores={"support": 0.85, "resistance": 0.90}
    )
    
    analysis.technical_indicators = TechnicalIndicators(
        rsi=65.5,
        macd={"macd": 0.5, "signal": 0.3, "histogram": 0.2},
        bollinger_bands={"upper": 47000, "middle": 45000, "lower": 43000},
        moving_averages={"sma_20": 44800, "ema_12": 45200},
        stochastic={"k": 70, "d": 68}
    )
    
    # Gerar JSON para validação
    json_output = analysis.model_dump_json(indent=2)
    print("Estrutura JSON gerada:")
    print(json_output)