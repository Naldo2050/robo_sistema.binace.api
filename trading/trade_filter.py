# utils/trade_filter.py

import time
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configuração global
MAX_TRADE_AGE_SECONDS = 30  # Máximo de 30 segundos de atraso
STALE_TRADE_WARNING_THRESHOLD = 10  # Avisar se > 10 segundos

class TradeFilter:
    """
    Filtro para trades atrasados com métricas
    """
    
    def __init__(self, max_age_seconds: int = 30):
        self.max_age_seconds = max_age_seconds
        self.stats = {
            'total_received': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'max_delay_seen': 0
        }
    
    def get_current_time_ms(self) -> int:
        """Retorna timestamp atual em milissegundos"""
        return int(time.time() * 1000)
    
    def calculate_trade_age_ms(self, trade_timestamp_ms: int) -> int:
        """Calcula idade do trade em milissegundos"""
        current_time_ms = self.get_current_time_ms()
        return current_time_ms - trade_timestamp_ms
    
    def is_trade_valid(self, trade: Dict[str, Any]) -> bool:
        """
        Verifica se trade é válido (não está obsoleto)
        
        Args:
            trade: Dicionário com dados do trade (deve ter 'T' ou 'timestamp')
            
        Returns:
            True se trade é válido, False caso contrário
        """
        self.stats['total_received'] += 1
        
        # Extrair timestamp do trade
        trade_timestamp_ms = trade.get('T') or trade.get('timestamp')
        
        if trade_timestamp_ms is None:
            logger.warning("Trade sem timestamp, descartando")
            self.stats['total_rejected'] += 1
            return False
        
        # Se timestamp em segundos, converter para ms
        if trade_timestamp_ms < 1e12:
            trade_timestamp_ms = int(trade_timestamp_ms * 1000)
        
        # Calcular idade
        age_ms = self.calculate_trade_age_ms(trade_timestamp_ms)
        age_seconds = age_ms / 1000
        
        # Atualizar estatísticas
        if age_ms > self.stats['max_delay_seen']:
            self.stats['max_delay_seen'] = age_ms
        
        # Verificar se muito antigo
        if age_seconds > self.max_age_seconds:
            # Log apenas a cada 100 trades rejeitados para não poluir
            if self.stats['total_rejected'] % 100 == 0:
                logger.warning(
                    f"Trade descartado por atraso: {age_seconds:.1f}s "
                    f"(max permitido: {self.max_age_seconds}s) | "
                    f"Total rejeitados: {self.stats['total_rejected']}"
                )
            self.stats['total_rejected'] += 1
            return False
        
        # Trade válido
        self.stats['total_accepted'] += 1
        
        # Avisar se está perto do limite
        if age_seconds > STALE_TRADE_WARNING_THRESHOLD:
            logger.debug(f"Trade com atraso significativo: {age_seconds:.1f}s")
        
        return True
    
    def filter_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtra lista de trades removendo obsoletos
        
        Args:
            trades: Lista de trades
            
        Returns:
            Lista filtrada apenas com trades válidos
        """
        valid_trades = []
        
        for trade in trades:
            if self.is_trade_valid(trade):
                valid_trades.append(trade)
        
        # Log de estatísticas periodicamente
        if self.stats['total_received'] % 1000 == 0:
            acceptance_rate = (
                self.stats['total_accepted'] / self.stats['total_received'] * 100
                if self.stats['total_received'] > 0 else 0
            )
            logger.info(
                f"📊 TradeFilter Stats: "
                f"Recebidos={self.stats['total_received']}, "
                f"Aceitos={self.stats['total_accepted']}, "
                f"Rejeitados={self.stats['total_rejected']}, "
                f"Taxa={acceptance_rate:.1f}%, "
                f"MaxDelay={self.stats['max_delay_seen']/1000:.1f}s"
            )
        
        return valid_trades
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do filtro"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reseta estatísticas"""
        self.stats = {
            'total_received': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'max_delay_seen': 0
        }


# Instância global do filtro
trade_filter = TradeFilter(max_age_seconds=MAX_TRADE_AGE_SECONDS)