#!/usr/bin/env python3
"""
Test script to demonstrate circuit breaker integration in orderbook_analyzer.py
"""
import asyncio
import logging
from orderbook_analyzer import OrderBookAnalyzer

async def test_circuit_breaker_integration():
    """Test circuit breaker functionality integration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
    )
    
    print("\n" + "=" * 80)
    print("CIRCUIT BREAKER INTEGRATION TEST")
    print("=" * 80 + "\n")
    
    # Test with a normal symbol first
    print("Testing normal operation...")
    async with OrderBookAnalyzer(symbol="BTCUSDT") as oba:
        # Initial successful request
        evt = await oba.analyze()
        print(f"Initial request successful: {evt.get('is_valid')}")
        
        # Check circuit breaker state
        cb_state = oba._circuit_breaker.state()
        print(f"Circuit Breaker State: {cb_state.value}")
        
        # Get circuit breaker snapshot
        cb_snapshot = oba._circuit_breaker.snapshot()
        print(f"Circuit Breaker Snapshot:")
        for key, value in cb_snapshot.items():
            print(f"  - {key}: {value}")
        
        # Test stats include circuit breaker
        stats = oba.get_stats()
        print(f"Stats include circuit breaker: {'circuit_breaker' in stats}")
        if 'circuit_breaker' in stats:
            print(f"  - CB State: {stats['circuit_breaker']['state']}")
    
    print("\n" + "=" * 80)
    print("CIRCUIT BREAKER INTEGRATION TEST COMPLETED")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_circuit_breaker_integration())