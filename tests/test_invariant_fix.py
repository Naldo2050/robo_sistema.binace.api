
import logging
import sys
import os
sys.path.append(os.getcwd()) # Fix ModuleNotFoundError
import pandas as pd
from data_pipeline.metrics.processor import MetricsProcessor
from data_pipeline.pipeline import DataPipeline, PipelineConfig

# Mock classes needed for minimal setup
class MockLogger:
    def performance_info(self, *args, **kwargs): pass
    def runtime_info(self, *args, **kwargs): pass
    def runtime_warning(self, *args, **kwargs): pass
    def runtime_error(self, *args, **kwargs): print(f"ERROR: {args}")
    def validation_info(self, *args, **kwargs): pass
    def adaptive_info(self, *args, **kwargs): pass

def test_invariant_fix():
    print("Testing Invariant Violation Fix...")
    
    # Setup test data
    # 3 trades: 2 buys (taker), 1 sell (taker)
    # Taker Buy: m=False
    # Taker Sell: m=True
    trades = [
        {"a": 1, "p": "50000.0", "q": "1.0001", "f": 1, "l": 1, "T": 1000, "m": False, "M": True}, # Buy 1.0001
        {"a": 2, "p": "50001.0", "q": "0.5002", "f": 2, "l": 2, "T": 1001, "m": False, "M": True}, # Buy 0.5002
        {"a": 3, "p": "49999.0", "q": "0.8003", "f": 3, "l": 3, "T": 1002, "m": True, "M": True},  # Sell 0.8003
    ]
    # Total Vol: 2.3006
    # Buy Vol: 1.5003
    # Sell Vol: 0.8003
    # Delta: 0.7000 (1.5003 - 0.8003)
    
    config = PipelineConfig()
    config.min_trades_pipeline = 1
    config.min_absolute_trades = 1
    config.enable_adaptive_thresholds = False # Simplify
    
    pipeline = DataPipeline(
        raw_trades=trades,
        symbol="BTCUSDT",
        config=config,
        shared_adaptive=False
    )
    
    # 1. Enriched (triggers MetricsProcessor)
    enriched = pipeline.enrich()
    
    print("\n--- MetricsProcessor Output ---")
    print(f"Total: {enriched.get('volume_total')}")
    print(f"Buy: {enriched.get('volume_compra')}")
    print(f"Sell: {enriched.get('volume_venda')}")
    
    assert enriched.get('volume_compra') == 1.5003
    assert enriched.get('volume_venda') == 0.8003
    assert enriched.get('volume_total') == 2.3006
    
    # 2. Signals (triggers analysis_trigger creation)
    # This also calls _validate_invariants internally
    print("\n--- Signal Generation & Invariant Check ---")
    pipeline.add_context() # Required before signals
    signals = pipeline.detect_signals()
    
    trigger = next(s for s in signals if s['tipo_evento'] == 'ANALYSIS_TRIGGER')
    
    print(f"Trigger Buy: {trigger.get('volume_compra')}")
    print(f"Trigger Sell: {trigger.get('volume_venda')}")
    print(f"Trigger Total: {trigger.get('volume_total')}")
    print(f"Trigger Delta: {trigger.get('delta')}") # delta_fechamento
    
    assert trigger.get('volume_compra') == 1.5003
    assert trigger.get('volume_venda') == 0.8003
    assert trigger.get('delta') == 0.7000 # Should be 0.7000 if rounding fix worked
    
    # Invariant Check Logic:
    # Buy - Sell = 0.7000
    # Delta (stored) = 0.7000
    # 0.7000 - 0.7000 = 0.0 -> OK
    
    # Check if standard logger would have printed warning (we can't capture easily without configuring root logger)
    # But since we asserted the values are correct, the invariant check:
    # if abs(vol_sum - vol_total) > TOLERANCE_BTC:
    # 1.5 + 0.8 - 2.3 = 0.0 -> Should pass
    
    print("\n✅ Test Passed: Volumes matched and propagated correctly.")

if __name__ == "__main__":
    test_invariant_fix()
