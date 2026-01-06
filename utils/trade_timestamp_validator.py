# utils/trade_timestamp_validator.py
"""
Validador de timestamp de trades - MODO MONITORAMENTO APENAS.

Este módulo foi modificado para NÃO DESCARTAR trades.
Todos os trades são aceitos, mas a latência é registrada para diagnóstico.

Deve ser chamado IMEDIATAMENTE após receber o trade do WebSocket.
"""

import time
import logging
from threading import Lock
from collections import deque
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class TradeLatencyMonitor:
    """
    Monitora latência de trades SEM descartar.
    Todos os trades são aceitos, mas latência é registrada para diagnóstico.
    """
    
    def __init__(self, warning_threshold_ms: int = 5000, summary_interval_sec: int = 300):
        """
        Args:
            warning_threshold_ms: Threshold para considerar latência alta (default: 5s)
            summary_interval_sec: Intervalo entre resumos em segundos (default: 5min)
        """
        self.warning_threshold_ms = warning_threshold_ms
        self.summary_interval_sec = summary_interval_sec
        
        self.stats = {
            'total_processed': 0,
            'high_latency_count': 0,
            'max_latency_ms': 0,
            'future_trades': 0,
            'recent_latencies': deque(maxlen=1000)  # Últimas 1000 latências
        }
        self._lock = Lock()
        self._last_summary_time = time.time()
    
    def record_trade(self, trade: dict) -> tuple[bool, str]:
        """
        Registra trade e sua latência. NÃO descarta nenhum trade.
        
        Args:
            trade: Dados do trade
            
        Returns:
            Tuple (accepted, status): Sempre retorna (True, "ok")
        """
        with self._lock:
            self.stats['total_processed'] += 1
        
        # Timestamp atual em ms
        now_ms = int(time.time() * 1000)
        
        # Extrair timestamp do trade
        trade_time_ms = self._extract_timestamp(trade)
        
        if trade_time_ms is None:
            return True, "ok_no_timestamp"
        
        # Calcular latência
        latency_ms = now_ms - trade_time_ms
        
        # Registrar estatísticas
        with self._lock:
            self.stats['recent_latencies'].append(latency_ms)
            
            if latency_ms > self.stats['max_latency_ms']:
                self.stats['max_latency_ms'] = latency_ms
            
            if latency_ms > self.warning_threshold_ms:
                self.stats['high_latency_count'] += 1
            
            # Trade futuro (pode indicar problema de sync)
            if latency_ms < 0:
                self.stats['future_trades'] += 1
        
        # Emitir resumo periódico
        self._maybe_emit_summary()
        
        return True, "ok"
    
    def _extract_timestamp(self, trade: dict) -> Optional[int]:
        """Extrai timestamp do trade."""
        ts = trade.get('T') or trade.get('timestamp') or trade.get('time')
        
        if ts is None:
            return None
        
        # Converter para ms se necessário
        if ts < 1e12:  # Timestamp em segundos
            ts = int(ts * 1000)
        else:  # Timestamp em ms
            ts = int(ts)
        
        return ts
    
    def _maybe_emit_summary(self) -> None:
        """Emite resumo de latência periodicamente"""
        now = time.time()
        
        if now - self._last_summary_time >= self.summary_interval_sec:
            self._emit_summary()
            self._last_summary_time = now
    
    def _emit_summary(self) -> None:
        """Emite resumo das estatísticas de latência"""
        with self._lock:
            if not self.stats['recent_latencies']:
                return
            
            latencies = list(self.stats['recent_latencies'])
            total = self.stats['total_processed']
            high_latency = self.stats['high_latency_count']
        
        if not latencies:
            return
        
        avg_latency = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(latencies) // 2]
        p95 = sorted_latencies[int(len(latencies) * 0.95)]
        
        high_latency_pct = (high_latency / total * 100) if total > 0 else 0
        
        logger.info(
            f"📊 LATÊNCIA DE TRADES (últimos {self.summary_interval_sec//60}min): "
            f"Total={total:,} | "
            f"Avg={avg_latency:.0f}ms | "
            f"P50={p50:.0f}ms | "
            f"P95={p95:.0f}ms | "
            f"Max={self.stats['max_latency_ms']:.0f}ms | "
            f"Alta latência={high_latency_pct:.1f}%"
        )
        
        # Alerta se latência estiver muito alta
        if high_latency_pct > 20:
            logger.warning(
                f"⚠️ Alta taxa de latência detectada: {high_latency_pct:.1f}% dos trades "
                f"com latência > {self.warning_threshold_ms}ms. "
                f"Considere otimizar o processamento."
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas atuais"""
        with self._lock:
            return {
                'total_processed': self.stats['total_processed'],
                'high_latency_count': self.stats['high_latency_count'],
                'max_latency_ms': self.stats['max_latency_ms'],
                'future_trades': self.stats['future_trades'],
                'valid_rate_pct': 100.0,  # Sempre 100% pois não descartamos nada
                'recent_count': len(self.stats['recent_latencies']),
            }
    
    def reset_stats(self):
        """Reseta estatísticas"""
        with self._lock:
            self.stats = {
                'total_processed': 0,
                'high_latency_count': 0,
                'max_latency_ms': 0,
                'future_trades': 0,
                'recent_latencies': deque(maxlen=1000),
            }


# Instância global para uso simplificado
latency_monitor = TradeLatencyMonitor(warning_threshold_ms=5000)


def record_trade_latency(trade: dict) -> tuple[bool, str]:
    """
    Função helper para registrar latência de trade.
    
    Args:
        trade: Dict com dados do trade
        
    Returns:
        Tuple (accepted, status): Sempre (True, "ok")
    """
    return latency_monitor.record_trade(trade)


def get_latency_stats() -> dict:
    """Retorna estatísticas do monitor."""
    return latency_monitor.get_stats()
