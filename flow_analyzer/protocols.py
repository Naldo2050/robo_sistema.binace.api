# flow_analyzer/protocols.py
"""
Interfaces e Protocols do FlowAnalyzer.

Define contratos para:
- IFlowAnalyzer: Interface principal
- ITimeProvider: Abstração de tempo
- ITradeProcessor: Processamento de trades
- IMetricsCollector: Coleta de métricas
"""

from typing import Protocol, Dict, Any, Optional, List, Tuple, runtime_checkable
from decimal import Decimal


@runtime_checkable
class ITimeProvider(Protocol):
    """Interface para provedores de tempo."""
    
    def now_ms(self) -> int:
        """Retorna timestamp atual em milliseconds (epoch UTC)."""
        ...
    
    def format_timestamp(self, ts_ms: int) -> str:
        """Formata timestamp para string ISO."""
        ...
    
    def build_time_index(
        self, 
        ts_ms: int, 
        include_local: bool = False,
        timespec: str = "milliseconds"
    ) -> Dict[str, Any]:
        """Constrói índice temporal completo."""
        ...


@runtime_checkable
class IClockSync(Protocol):
    """Interface para sincronização de clock."""
    
    def get_server_time_ms(self) -> int:
        """Retorna tempo sincronizado do servidor."""
        ...
    
    def get_offset_ms(self) -> float:
        """Retorna offset entre cliente e servidor."""
        ...


@runtime_checkable
class ITradeProcessor(Protocol):
    """Interface para processamento de trades."""
    
    def process_trade(self, trade: Dict[str, Any]) -> None:
        """
        Processa um trade individual.
        
        Args:
            trade: Dict com campos 'q' (quantity), 'T' (timestamp), 
                   'p' (price), 'm' (is_buyer_maker)
        """
        ...
    
    def process_batch(self, trades: List[Dict[str, Any]]) -> int:
        """
        Processa batch de trades.
        
        Returns:
            Número de trades processados com sucesso.
        """
        ...


@runtime_checkable
class IMetricsCollector(Protocol):
    """Interface para coleta de métricas."""
    
    def record_processing_time(self, duration_ms: float) -> None:
        """Registra tempo de processamento."""
        ...
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Incrementa contador."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas coletadas."""
        ...


@runtime_checkable
class IFlowAnalyzer(Protocol):
    """Interface principal do FlowAnalyzer."""
    
    # === Processamento ===
    def process_trade(self, trade: Dict[str, Any]) -> None:
        """Processa trade individual."""
        ...
    
    # === Métricas ===
    def get_flow_metrics(
        self, 
        reference_epoch_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """Retorna métricas de fluxo completas."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do analyzer."""
        ...
    
    # === Saúde ===
    def health_check(self) -> Dict[str, Any]:
        """Verificação de saúde do sistema."""
        ...
    
    # === Configuração ===
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Atualiza configuração dinamicamente."""
        ...
    
    # === Volatilidade ===
    def update_volatility_context(
        self,
        atr_price: Optional[float] = None,
        price_volatility: Optional[float] = None
    ) -> None:
        """Atualiza contexto de volatilidade."""
        ...


@runtime_checkable
class ILiquidityHeatmap(Protocol):
    """Interface para heatmap de liquidez."""
    
    def add_trade(
        self, 
        price: float, 
        volume: float, 
        side: str, 
        timestamp_ms: int
    ) -> None:
        """Adiciona trade ao heatmap."""
        ...
    
    def get_clusters(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Retorna top N clusters de liquidez."""
        ...
    
    def get_support_resistance(self) -> Tuple[List[float], List[float]]:
        """Retorna níveis de suporte e resistência."""
        ...


@runtime_checkable
class IRollingAggregate(Protocol):
    """Interface para agregação rolling."""
    
    def add_trade(self, trade: Dict[str, Any], whale_threshold: float) -> None:
        """Adiciona trade à janela."""
        ...
    
    def prune(self, cutoff_ms: int) -> None:
        """Remove trades antigos."""
        ...
    
    def get_metrics(self, last_price: float) -> Dict[str, Any]:
        """Retorna métricas da janela."""
        ...
    
    def reset(self) -> None:
        """Reseta estado."""
        ...


class ICircuitBreaker(Protocol):
    """Interface para circuit breaker."""
    
    def record_success(self) -> None:
        """Registra operação bem-sucedida."""
        ...
    
    def record_failure(self) -> None:
        """Registra falha."""
        ...
    
    def can_execute(self) -> bool:
        """Verifica se pode executar operação."""
        ...
    
    @property
    def state(self) -> str:
        """Estado atual: CLOSED, OPEN, HALF_OPEN."""
        ...


# ==============================================================================
# TYPE ALIASES
# ==============================================================================

TradeDict = Dict[str, Any]
MetricsDict = Dict[str, Any]
ConfigDict = Dict[str, Any]
OHLCTuple = Tuple[float, float, float, float]  # (open, high, low, close)
SectorFlowDict = Dict[str, Dict[str, Decimal]]