import sys
from unittest.mock import MagicMock
import os
from collections import deque
import logging
# Prevent clock_sync from floating
sys.modules['clock_sync'] = MagicMock()

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import flow_analyzer as fa_module
fa_module.HAS_CLOCK_SYNC = False
# No need to mock get_clock_sync if module is mocked, but just in case
fa_module.get_clock_sync = lambda: None
from flow_analyzer.core import FlowAnalyzer
import config

# Setup basic logging to stderr to avoid buffering issues
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

def run_verification():
    print("--- Starting Verification ---")
    
    # Mock configs
    config.NET_FLOW_WINDOWS_MIN = [1]
    config.FLOW_TRADES_MAXLEN = 100
    
    analyzer = FlowAnalyzer()
    analyzer.net_flow_windows_min = [1]
    analyzer.flow_trades_maxlen = 100
    analyzer.flow_trades = deque(maxlen=100)
    
    print("\nTest 1: Out of Order Detection")
    # Trade 1: ts=1000
    analyzer.process_trade({'ts': 1000, 'qty': 1.0, 'price': 100.0, 'side': 'buy', 'q': 1.0, 'T': 1000, 'p': 100.0, 'm': False})
    print(f"Trade 1 (1000) -> Max: {analyzer._max_ts_seen}, OOO: {analyzer._out_of_order_seen}")
    
    # Trade 2: ts=3000
    analyzer.process_trade({'ts': 3000, 'qty': 1.0, 'price': 100.0, 'side': 'buy', 'q': 1.0, 'T': 3000, 'p': 100.0, 'm': False})
    print(f"Trade 2 (3000) -> Max: {analyzer._max_ts_seen}, OOO: {analyzer._out_of_order_seen}")
    
    # Trade 3: ts=2000
    analyzer.process_trade({'ts': 2000, 'qty': 1.0, 'price': 100.0, 'side': 'buy', 'q': 1.0, 'T': 2000, 'p': 100.0, 'm': False})
    print(f"Trade 3 (2000) -> Max: {analyzer._max_ts_seen}, OOO: {analyzer._out_of_order_seen}")
    
    if analyzer._out_of_order_seen:
        print("PASS: Out of order detected.")
    else:
        print("FAIL: Out of order NOT detected.")
        sys.exit(1)

    print("\nTest 2: Pruning Robustness")
    # Reset for test 2
    analyzer = FlowAnalyzer()
    analyzer.net_flow_windows_min = [1] # 60000 ms
    analyzer.flow_trades = deque(maxlen=100)
    
    # Add trades with specific timestamps
    # T1: 10000
    analyzer.process_trade({'q': 1.0, 'T': 10000, 'p': 100, 'm': False})
    # T2: 75000
    analyzer.process_trade({'q': 1.0, 'T': 75000, 'p': 100, 'm': False})
    # T3: 65000 (Late)
    analyzer.process_trade({'q': 1.0, 'T': 65000, 'p': 100, 'm': False})
    
    print(f"Trades count: {len(analyzer.flow_trades)}")
    print(f"Timestamps: {[t['ts'] for t in analyzer.flow_trades]}")
    print(f"OOO Seen: {analyzer._out_of_order_seen}")
    
    # Now: 80000. Cutoff: 20000.
    # T1 (10000) < 20000 -> Should remove
    # T2 (75000) > 20000 -> Keep
    # T3 (65000) > 20000 -> Keep
    
    print("Pruning at now=80000...")
    analyzer._prune_flow_history(now_ms=80000)
    
    print(f"Trades count after: {len(analyzer.flow_trades)}")
    print(f"Timestamps after: {[t['ts'] for t in analyzer.flow_trades]}")
    print(f"OOO Seen after: {analyzer._out_of_order_seen}")
    
    if len(analyzer.flow_trades) == 2:
        ts_list = sorted([t['ts'] for t in analyzer.flow_trades])
        if ts_list == [65000, 75000]:
            print("PASS: Pruning Correct.")
        else:
            print(f"FAIL: Wrong timestamps: {ts_list}")
            sys.exit(1)
    else:
        print(f"FAIL: Wrong count: {len(analyzer.flow_trades)}")
        sys.exit(1)

if __name__ == "__main__":
    run_verification()
