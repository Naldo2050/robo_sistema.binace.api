#!/usr/bin/env python3
"""
Trade Validator - Sistema de validação e monitoramento de trades.

Responsabilidades:
- Validar estrutura de trades
- Monitorar latência de trades SEM descartar
- Registrar métricas de latência para diagnóstico
"""

import time
import logging
from collections import deque
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configurações padrão
DEFAULT_MAX_AGE_SECONDS = 30  # 30 segundos máximo de atraso


def is_trade_valid(trade: Dict[str, Any], max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS) -> bool:
    """
    Verifica se um trade é válido. 
    NOTA: Trades atrasados são ACEITOS - apenas registramos a latência para diagnóstico.
    
    Args:
        trade: Dicionário com dados do trade
        max_age_seconds: Idade máxima para registro (apenas informativo)

    Returns:
        True sempre - nenhum trade é descartado
    """
    try:
        trade_timestamp = trade.get('timestamp', 0)

        # Verificar se timestamp existe e é válido
        if not trade_timestamp or trade_timestamp <= 0:
            logger.warning(f"Trade com timestamp inválido: {trade_timestamp}")
            return True  # Ainda aceitamos o trade

        trade_age = time.time() - trade_timestamp

        # Registrar se está atrasado, mas NÃO descartar
        if trade_age > max_age_seconds:
            logger.info(f"📊 Trade com {trade_age:.2f}s de latência (aceito)")

        return True  # Sempre aceitar

    except Exception as e:
        logger.error(f"Erro ao processar trade: {e}")
        return True  # Aceitar em caso de erro


def filter_stale_trades(trades: List[Dict[str, Any]], max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS) -> List[Dict[str, Any]]:
    """
    Filtra lista de trades.
    NOTA: TODOS os trades são aceitos - apenas registramos estatísticas.

    Args:
        trades: Lista de trades para processar
        max_age_seconds: Idade máxima para registro (apenas informativo)

    Returns:
        Lista contendo TODOS os trades de entrada
    """
    if not trades:
        return []

    # Calcular estatísticas sem descartar
    late_count = 0
    for trade in trades:
        trade_timestamp = trade.get('timestamp', 0)
        if trade_timestamp and trade_timestamp > 0:
            trade_age = time.time() - trade_timestamp
            if trade_age > max_age_seconds:
                late_count += 1

    if late_count > 0:
        logger.info(f"📊 trades_processados={len(trades)} | trades_atrasados={late_count} | TODOS ACEITOS")

    return trades  # Retornar todos os trades


def validate_trade_structure(trade: Dict[str, Any]) -> bool:
    """
    Valida se a estrutura básica do trade está correta.

    Args:
        trade: Dicionário com dados do trade

    Returns:
        True se a estrutura é válida
    """
    required_fields = ['timestamp', 'price', 'quantity']

    for field in required_fields:
        if field not in trade:
            logger.warning(f"Trade faltando campo obrigatório '{field}': {trade}")
            return False

        # Verificar se valores são numéricos quando necessário
        if field in ['price', 'quantity', 'timestamp']:
            value = trade[field]
            if not isinstance(value, (int, float)) or value <= 0:
                logger.warning(f"Trade com valor inválido para '{field}': {value}")
                return False

    return True


def filter_invalid_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filtra trades com estrutura inválida.

    Args:
        trades: Lista de trades para validar

    Returns:
        Lista de trades com estrutura válida
    """
    if not trades:
        return []

    valid_trades = []
    invalid_count = 0

    for trade in trades:
        if validate_trade_structure(trade):
            valid_trades.append(trade)
        else:
            invalid_count += 1

    if invalid_count > 0:
        logger.warning(f"Removidos {invalid_count} trades com estrutura inválida")

    return valid_trades


def validate_and_filter_trades(trades: List[Dict[str, Any]],
                               max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS) -> List[Dict[str, Any]]:
    """
    Valida estrutura de trades.
    NOTA: TODOS os trades são aceitos - apenas registramos para diagnóstico.

    Args:
        trades: Lista de trades para processar
        max_age_seconds: Idade máxima para registro (apenas informativo)

    Returns:
        Lista de TODOS os trades de entrada (nenhum descartado)
    """
    if not trades:
        return []

    # Validar estrutura (apenas log, sem descarte)
    invalid_count = 0
    for trade in trades:
        if not validate_trade_structure(trade):
            invalid_count += 1

    # Calcular estatísticas de latência
    late_count = 0
    for trade in trades:
        trade_timestamp = trade.get('timestamp', 0)
        if trade_timestamp and trade_timestamp > 0:
            trade_age = time.time() - trade_timestamp
            if trade_age > max_age_seconds:
                late_count += 1

    if invalid_count > 0 or late_count > 0:
        logger.info(f"📊 trades_processados={len(trades)} | estrutura_inválida={invalid_count} | atraso>{max_age_seconds}s={late_count} | TODOS ACEITOS")

    return trades  # Retornar todos os trades


def get_trade_age_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula estatísticas de idade dos trades.

    Args:
        trades: Lista de trades

    Returns:
        Dicionário com estatísticas de idade
    """
    if not trades:
        return {"count": 0, "avg_age": 0, "max_age": 0, "min_age": 0}

    current_time = time.time()
    ages = []

    for trade in trades:
        timestamp = trade.get('timestamp', 0)
        if timestamp > 0:
            age = current_time - timestamp
            ages.append(age)

    if not ages:
        return {"count": len(trades), "avg_age": 0, "max_age": 0, "min_age": 0}

    return {
        "count": len(trades),
        "avg_age": sum(ages) / len(ages),
        "max_age": max(ages),
        "min_age": min(ages)
    }


class TradeLatencyMonitor:
    """
    Monitora latência de trades SEM descartar.
    Todos os trades são aceitos, mas latência é registrada para diagnóstico.
    """
    
    def __init__(self, warning_threshold_ms: int = 5000):
        self.warning_threshold_ms = warning_threshold_ms
        self.stats = {
            'total_processed': 0,
            'high_latency_count': 0,
            'max_latency_ms': 0,
            'recent_latencies': deque(maxlen=1000)  # Últimas 1000 latências
        }
        self._last_summary_time = time.time()
        self._summary_interval = 300  # Resumo a cada 5 minutos
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Registra trade e sua latência. NÃO descarta nenhum trade.
        
        Args:
            trade: Dados do trade
        """
        self.stats['total_processed'] += 1
        
        # Calcular latência
        now_ms = int(time.time() * 1000)
        trade_time_ms = trade.get('T') or trade.get('timestamp', now_ms)
        
        # Converter se necessário
        if trade_time_ms < 1e12:
            trade_time_ms = int(trade_time_ms * 1000)
        
        latency_ms = now_ms - trade_time_ms
        
        # Registrar estatísticas
        self.stats['recent_latencies'].append(latency_ms)
        
        if latency_ms > self.stats['max_latency_ms']:
            self.stats['max_latency_ms'] = latency_ms
        
        if latency_ms > self.warning_threshold_ms:
            self.stats['high_latency_count'] += 1
        
        # Emitir resumo periódico (não a cada trade!)
        self._maybe_emit_summary()
    
    def _maybe_emit_summary(self) -> None:
        """Emite resumo de latência periodicamente"""
        now = time.time()
        
        if now - self._last_summary_time >= self._summary_interval:
            self._emit_summary()
            self._last_summary_time = now
    
    def _emit_summary(self) -> None:
        """Emite resumo das estatísticas de latência"""
        if not self.stats['recent_latencies']:
            return
        
        latencies = list(self.stats['recent_latencies'])
        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        
        high_latency_pct = (
            self.stats['high_latency_count'] / self.stats['total_processed'] * 100
            if self.stats['total_processed'] > 0 else 0
        )
        
        logger.info(
            f"📊 LATÊNCIA DE TRADES (últimos 5min): "
            f"Total={self.stats['total_processed']:,} | "
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
        return self.stats.copy()


# Instância global
latency_monitor = TradeLatencyMonitor(warning_threshold_ms=5000)


# Teste da funcionalidade
if __name__ == "__main__":
    # Teste básico
    current_time = time.time()

    # Trades de teste
    test_trades = [
        {"timestamp": current_time - 10, "price": 50000, "quantity": 0.1},  # Válido
        {"timestamp": current_time - 60, "price": 50100, "quantity": 0.05}, # Atrasado
        {"timestamp": current_time + 10, "price": 49900, "quantity": 0.2},  # Futuro (aceito)
        {"price": 50000, "quantity": 0.1},  # Sem timestamp (inválido)
    ]

    print("Testando Trade Validator...")
    print(f"Trades de entrada: {len(test_trades)}")

    # Filtrar
    valid_trades = validate_and_filter_trades(test_trades, max_age_seconds=30)
    print(f"Trades válidos: {len(valid_trades)}")

    # Estatísticas
    stats = get_trade_age_stats(valid_trades)
    print(f"Estatísticas: {stats}")

    print("✅ Teste concluído!")