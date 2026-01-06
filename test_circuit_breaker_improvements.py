#!/usr/bin/env python3
"""
Teste das melhorias do Circuit Breaker para OrderBook.

Este script demonstra:
1. Circuit Breaker otimizado com thresholds ajustados
2. Fallback robusto para REST API
3. Retry com jitter para evitar thundering herd
4. Half-open state para reconexão gradual
"""

import asyncio
import logging
import time
from typing import Dict, Any

from orderbook_analyzer import OrderBookAnalyzer
from orderbook_core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from orderbook_fallback import OrderBookFallback, FallbackConfig

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)

logger = logging.getLogger(__name__)


async def test_circuit_breaker_improvements():
    """Testa as melhorias do Circuit Breaker."""
    
    print("\n" + "=" * 80)
    print("🧪 TESTE: CIRCUIT BREAKER MELHORADO + FALLBACK ROBUSTO")
    print("=" * 80 + "\n")
    
    # Teste 1: Circuit Breaker otimizado
    print("🔌 Teste 1: Circuit Breaker Otimizado")
    print("-" * 50)
    
    # Configuração otimizada
    cb_config = CircuitBreakerConfig(
        failure_threshold=5,        # Abrir após 5 falhas
        success_threshold=2,        # Fechar após 2 sucessos em half-open
        timeout_seconds=30.0,       # 30 segundos em OPEN antes de half-open
        half_open_max_calls=3,      # Máximo 3 tentativas em half-open
        fallback_enabled=True,      # Habilita fallback
        max_retry_attempts=3,       # Máximo 3 retries
        base_retry_delay=1.0,       # Delay base 1s
        max_retry_delay=10.0,       # Delay máximo 10s
    )
    
    cb = CircuitBreaker(name="test_orderbook", config=cb_config)
    
    print(f"✅ Configuração do Circuit Breaker:")
    print(f"   - Failure Threshold: {cb_config.failure_threshold}")
    print(f"   - Success Threshold: {cb_config.success_threshold}")
    print(f"   - Timeout: {cb_config.timeout_seconds}s")
    print(f"   - Half-open Max Calls: {cb_config.half_open_max_calls}")
    print(f"   - Fallback Enabled: {cb_config.fallback_enabled}")
    
    # Simula falhas
    print(f"\n📉 Simulando falhas para abrir o circuit...")
    for i in range(cb_config.failure_threshold):
        cb.record_failure()
        state = cb.state()
        print(f"   Falha {i+1}: Estado = {state.value}")
    
    print(f"✅ Circuit Breaker aberto: {cb.state().value}")
    
    # Testa se permite request (deve ser False)
    allow_request = cb.allow_request()
    print(f"🔒 Allow Request (deve ser False): {allow_request}")
    
    # Teste 2: Fallback robusto
    print(f"\n🔄 Teste 2: Fallback Robusto")
    print("-" * 50)
    
    fallback_config = FallbackConfig(
        enabled=True,
        request_timeout=15.0,
        max_retries=5,
        backoff_factor=2.0,
        jitter_range=0.25,
    )
    
    fallback = OrderBookFallback(config=fallback_config)
    
    print(f"✅ Configuração do Fallback:")
    print(f"   - Enabled: {fallback_config.enabled}")
    print(f"   - Request Timeout: {fallback_config.request_timeout}s")
    print(f"   - Max Retries: {fallback_config.max_retries}")
    print(f"   - Backoff Factor: {fallback_config.backoff_factor}")
    print(f"   - Jitter Range: {fallback_config.jitter_range * 100}%")
    print(f"   - Health Score: {fallback._health_score}")
    
    # Testa múltiplos endpoints
    print(f"\n🌐 Testando múltiplos endpoints...")
    for i, endpoint in enumerate(fallback.endpoints, 1):
        print(f"   {i}. {endpoint}")
    
    # Teste 3: OrderBookAnalyzer com melhorias
    print(f"\n🤖 Teste 3: OrderBookAnalyzer com Melhorias")
    print("-" * 50)
    
    try:
        async with OrderBookAnalyzer(symbol="BTCUSDT") as oba:
            # Estatísticas iniciais
            stats = oba.get_stats()
            cb_stats = stats.get('circuit_breaker', {})
            fallback_stats = stats.get('fallback', {})
            
            print(f"✅ OrderBookAnalyzer inicializado:")
            print(f"   - Symbol: {oba.symbol}")
            print(f"   - Circuit Breaker State: {cb_stats.get('state', 'unknown')}")
            print(f"   - Fallback Healthy: {fallback_stats.get('healthy', False)}")
            print(f"   - Fallback Health Score: {fallback_stats.get('health_score', 0)}")
            
            # Testa fetch normal
            print(f"\n📡 Testando fetch normal...")
            start_time = time.time()
            
            try:
                # Timeout de 30 segundos para evitar travamento
                result = await asyncio.wait_for(oba.analyze(), timeout=30.0)
                fetch_time = time.time() - start_time
                
                if result.get('is_valid'):
                    print(f"✅ Fetch bem-sucedido em {fetch_time:.2f}s")
                    print(f"   - Data Source: {result.get('data_quality', {}).get('data_source')}")
                    print(f"   - Bid Depth: ${result.get('orderbook_data', {}).get('bid_depth_usd', 0):,.2f}")
                    print(f"   - Ask Depth: ${result.get('orderbook_data', {}).get('ask_depth_usd', 0):,.2f}")
                else:
                    print(f"⚠️ Fetch retornou dados inválidos")
                    print(f"   - Error: {result.get('erro', 'Unknown')}")
                    
            except asyncio.TimeoutError:
                print(f"⏱️ Timeout no fetch (mais de 30s)")
            except Exception as e:
                print(f"💥 Erro no fetch: {e}")
            
            # Estatísticas finais
            final_stats = oba.get_stats()
            final_cb_stats = final_stats.get('circuit_breaker', {})
            final_fallback_stats = final_stats.get('fallback', {})
            
            print(f"\n📊 Estatísticas Finais:")
            print(f"   - Total Fetches: {final_stats.get('total_fetches', 0)}")
            print(f"   - Fetch Errors: {final_stats.get('fetch_errors', 0)}")
            print(f"   - Error Rate: {final_stats.get('error_rate_pct', 0):.1f}%")
            print(f"   - Cache Hit Rate: {final_stats.get('cache_hit_rate_pct', 0):.1f}%")
            print(f"   - Circuit Breaker State: {final_cb_stats.get('state', 'unknown')}")
            print(f"   - Circuit Breaker Failure Count: {final_cb_stats.get('failure_count', 0)}")
            print(f"   - Fallback Health Score: {final_fallback_stats.get('health_score', 0):.2f}")
            print(f"   - Fallback Consecutive Failures: {final_fallback_stats.get('consecutive_failures', 0)}")
            
    except Exception as e:
        print(f"💥 Erro no OrderBookAnalyzer: {e}")
    
    # Teste 4: Demonstração do retry com jitter
    print(f"\n🔄 Teste 4: Retry com Jitter")
    print("-" * 50)
    
    print("Calculando delays de retry com jitter:")
    for attempt in range(5):
        delay = fallback._calculate_retry_delay(attempt)
        print(f"   Attempt {attempt + 1}: {delay:.2f}s")
    
    print(f"\n" + "=" * 80)
    print("✅ TESTES CONCLUÍDOS - CIRCUIT BREAKER MELHORADO")
    print("=" * 80 + "\n")
    
    # Resumo das melhorias
    print("📋 RESUMO DAS MELHORIAS IMPLEMENTADAS:")
    print("1. ✅ Circuit Breaker otimizado (threshold: 5, timeout: 30s)")
    print("2. ✅ Fallback robusto com múltiplos endpoints")
    print("3. ✅ Retry exponencial com jitter")
    print("4. ✅ Half-open state configurado")
    print("5. ✅ Health monitoring do fallback")
    print("6. ✅ Rate limiting inteligente")
    print("7. ✅ Métricas e monitoring aprimorados")
    print("\n🚀 O sistema agora é muito mais resiliente a falhas!")


async def test_circuit_breaker_states():
    """Testa transição de estados do Circuit Breaker."""
    
    print("\n" + "🔄 Teste: Transição de Estados do Circuit Breaker")
    print("-" * 60)
    
    cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=10.0,
        half_open_max_calls=2,
    )
    
    cb = CircuitBreaker(name="state_test", config=cb_config)
    
    # Estado inicial: CLOSED
    print(f"1. Estado inicial: {cb.state().value}")
    print(f"   Allow Request: {cb.allow_request()}")
    
    # Simula falhas para abrir
    print(f"\n2. Simulando falhas para abrir...")
    for i in range(3):
        cb.record_failure()
        print(f"   Falha {i+1}: {cb.state().value}")
    
    # Testa se bloqueia
    print(f"\n3. Testando bloqueio (deve ser False): {cb.allow_request()}")
    
    # Simula passagem de tempo para half-open
    print(f"\n4. Aguardando timeout para half-open...")
    await asyncio.sleep(11)  # Timeout + 1 segundo
    
    # Verifica se mudou para half-open
    allow_request = cb.allow_request()
    print(f"   Estado após timeout: {cb.state().value}")
    print(f"   Allow Request (deve ser True): {allow_request}")
    
    if allow_request:
        # Simula sucesso para fechar
        print(f"\n5. Simulando sucessos para fechar...")
        cb.record_success()
        print(f"   Sucesso 1: {cb.state().value}")
        
        cb.record_success()
        print(f"   Sucesso 2: {cb.state().value}")
    
    print(f"✅ Estado final: {cb.state().value}")


if __name__ == "__main__":
    async def main():
        await test_circuit_breaker_improvements()
        await test_circuit_breaker_states()
    
    asyncio.run(main())