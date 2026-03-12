#!/usr/bin/env python3
"""
Demonstração simples das melhorias do Circuit Breaker.
"""

import asyncio
import logging
import time

from orderbook_core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from orderbook_fallback import OrderBookFallback, FallbackConfig

# Configuração simples de logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


async def demo_circuit_breaker():
    """Demonstra as melhorias do Circuit Breaker."""
    
    print("=" * 60)
    print("DEMONSTRAÇÃO: CIRCUIT BREAKER MELHORADO")
    print("=" * 60)
    
    # 1. Configuração otimizada
    print("\n1. CONFIGURAÇÃO OTIMIZADA:")
    cb_config = CircuitBreakerConfig(
        failure_threshold=5,        # Abrir após 5 falhas
        success_threshold=2,        # Fechar após 2 sucessos
        timeout_seconds=10.0,       # 10 segundos para teste
        half_open_max_calls=3,      # Máximo 3 tentativas
        fallback_enabled=True,      # Habilita fallback
    )
    
    cb = CircuitBreaker(name="demo_orderbook", config=cb_config)
    print(f"   - Failure Threshold: {cb_config.failure_threshold}")
    print(f"   - Success Threshold: {cb_config.success_threshold}")
    print(f"   - Timeout: {cb_config.timeout_seconds}s")
    print(f"   - Estado inicial: {cb.state().value}")
    
    # 2. Simula falhas para abrir
    print("\n2. SIMULANDO FALHAS:")
    for i in range(cb_config.failure_threshold):
        cb.record_failure()
        print(f"   Falha {i+1}: Estado = {cb.state().value}")
    
    print(f"   Circuit Breaker ABERTO: {cb.state().value}")
    
    # 3. Testa fallback robusto
    print("\n3. FALLBACK ROBUSTO:")
    fallback_config = FallbackConfig(
        enabled=True,
        max_retries=3,
        backoff_factor=2.0,
        jitter_range=0.25,
    )
    
    fallback = OrderBookFallback(config=fallback_config)
    print(f"   - Fallback habilitado: {fallback_config.enabled}")
    print(f"   - Endpoints disponíveis: {len(fallback.endpoints)}")
    print(f"   - Health Score inicial: {fallback._health_score}")
    
    # 4. Testa retry com jitter
    print("\n4. RETRY COM JITTER:")
    for attempt in range(4):
        delay = fallback._calculate_retry_delay(attempt)
        print(f"   Attempt {attempt + 1}: {delay:.2f}s")
    
    # 5. Simula recuperação
    print("\n5. SIMULANDO RECUPERAÇÃO:")
    print("   Aguardando timeout...")
    await asyncio.sleep(cb_config.timeout_seconds + 1)
    
    # Verifica se mudou para half-open
    allow_request = cb.allow_request()
    print(f"   Estado após timeout: {cb.state().value}")
    print(f"   Allow Request: {allow_request}")
    
    if allow_request:
        print("   Simulando sucessos para fechar...")
        cb.record_success()
        print(f"   Sucesso 1: {cb.state().value}")
        
        cb.record_success()
        print(f"   Sucesso 2: {cb.state().value}")
    
    print(f"   Estado final: {cb.state().value}")
    
    # 6. Estatísticas finais
    print("\n6. ESTATÍSTICAS FINAIS:")
    cb_stats = cb.snapshot()
    fallback_stats = fallback.get_fallback_stats()
    
    print(f"   Circuit Breaker:")
    print(f"     - Estado: {cb_stats['state']}")
    print(f"     - Failure Count: {cb_stats['failure_count']}")
    print(f"     - Success Count: {cb_stats['success_count']}")
    
    print(f"   Fallback:")
    print(f"     - Healthy: {fallback_stats['healthy']}")
    print(f"     - Health Score: {fallback_stats['health_score']}")
    print(f"     - Consecutive Failures: {fallback_stats['consecutive_failures']}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRAÇÃO CONCLUÍDA")
    print("=" * 60)
    
    print("\nRESUMO DAS MELHORIAS:")
    print("✓ Circuit Breaker otimizado (threshold: 5)")
    print("✓ Fallback robusto multi-endpoint")
    print("✓ Retry com jitter anti-thundering herd")
    print("✓ Half-open state para recuperação gradual")
    print("✓ Health monitoring do fallback")
    print("\nO sistema agora é muito mais resiliente!")


if __name__ == "__main__":
    asyncio.run(demo_circuit_breaker())