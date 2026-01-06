# tests/verify_prune_logic_only.py
"""
Script de verificacao manual para logica de pruning.

Executa testes basicos sem dependencia de pytest.
Util para debugging rapido.

NOTA: Evita caracteres Unicode para compatibilidade com Windows.
"""

import sys
import os
from collections import deque
import logging
from unittest.mock import MagicMock, patch

# Setup path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock clock_sync antes de importar flow_analyzer
sys.modules['clock_sync'] = MagicMock()

# Setup basic logging (ASCII only)
logging.basicConfig(
    stream=sys.stderr, 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def process_trade_mocked(analyzer, trade, reference_ts):
    """Processa trade com timestamp mockado."""
    with patch.object(analyzer, '_get_synced_timestamp_ms', return_value=reference_ts):
        analyzer.process_trade(trade)


def run_verification():
    """Executa verificacoes de pruning."""
    print("=" * 60)
    print("VERIFICACAO DE LOGICA DE PRUNING")
    print("=" * 60)
    
    # Import apos mock
    from flow_analyzer import FlowAnalyzer
    
    all_passed = True
    
    # --- Test 1: Out of Order Detection ---
    print("\n[TEST 1] Deteccao de Out of Order")
    print("-" * 40)
    
    analyzer = FlowAnalyzer()
    analyzer.net_flow_windows_min = [1]  # 60000 ms
    analyzer.flow_trades = deque(maxlen=100)
    analyzer._max_ts_seen = 0
    analyzer._out_of_order_seen = False
    
    ref_ts = 100000
    
    # Trade 1: ts=50000
    process_trade_mocked(analyzer, {
        'q': 1.0, 'T': 50000, 'p': 100.0, 'm': False
    }, ref_ts)
    print("Trade 1 (ts=50000): max=%d, ooo=%s" % (analyzer._max_ts_seen, analyzer._out_of_order_seen))
    
    # Trade 2: ts=80000
    process_trade_mocked(analyzer, {
        'q': 1.0, 'T': 80000, 'p': 100.0, 'm': False
    }, ref_ts)
    print("Trade 2 (ts=80000): max=%d, ooo=%s" % (analyzer._max_ts_seen, analyzer._out_of_order_seen))
    
    # Trade 3: ts=60000 (LATE)
    process_trade_mocked(analyzer, {
        'q': 1.0, 'T': 60000, 'p': 100.0, 'm': False
    }, ref_ts)
    print("Trade 3 (ts=60000): max=%d, ooo=%s" % (analyzer._max_ts_seen, analyzer._out_of_order_seen))
    
    if analyzer._out_of_order_seen:
        print("[PASS] Out of order detectado corretamente.")
    else:
        print("[FAIL] Out of order NAO foi detectado!")
        all_passed = False
    
    # --- Test 2: Pruning Robustness ---
    print("\n[TEST 2] Robustez do Pruning")
    print("-" * 40)
    
    analyzer2 = FlowAnalyzer()
    analyzer2.net_flow_windows_min = [1]  # 60000 ms
    analyzer2.flow_trades = deque(maxlen=100)
    
    ref_ts2 = 80000
    
    # Add trades
    # T1: 10000 (antigo)
    process_trade_mocked(analyzer2, {'q': 1.0, 'T': 10000, 'p': 100, 'm': False}, ref_ts2)
    # T2: 75000 (recente)
    process_trade_mocked(analyzer2, {'q': 1.0, 'T': 75000, 'p': 100, 'm': False}, ref_ts2)
    # T3: 65000 (LATE)
    process_trade_mocked(analyzer2, {'q': 1.0, 'T': 65000, 'p': 100, 'm': False}, ref_ts2)
    
    print("Antes do prune: %d trades" % len(analyzer2.flow_trades))
    print("Timestamps: %s" % [t['ts'] for t in analyzer2.flow_trades])
    print("OOO flag: %s" % analyzer2._out_of_order_seen)
    
    # now=80000, cutoff=20000
    # T1(10000) < 20000 -> REMOVE
    # T2(75000) >= 20000 -> KEEP
    # T3(65000) >= 20000 -> KEEP
    
    print("\nExecutando prune com now=80000...")
    analyzer2._prune_flow_history(now_ms=80000)
    
    print("Apos prune: %d trades" % len(analyzer2.flow_trades))
    print("Timestamps: %s" % sorted([t['ts'] for t in analyzer2.flow_trades]))
    print("OOO flag: %s" % analyzer2._out_of_order_seen)
    
    if len(analyzer2.flow_trades) == 2:
        ts_list = sorted([t['ts'] for t in analyzer2.flow_trades])
        if ts_list == [65000, 75000]:
            print("[PASS] Pruning correto.")
        else:
            print("[FAIL] Timestamps incorretos: %s" % ts_list)
            all_passed = False
    else:
        print("[FAIL] Quantidade incorreta: %d" % len(analyzer2.flow_trades))
        all_passed = False
    
    # --- Test 3: CVD apos operacoes ---
    print("\n[TEST 3] CVD apos operacoes")
    print("-" * 40)
    
    analyzer3 = FlowAnalyzer()
    ref_ts3 = 100000
    
    # Trades de compra e venda
    process_trade_mocked(analyzer3, {'q': 2.0, 'T': 50000, 'p': 100, 'm': False}, ref_ts3)  # +2.0
    process_trade_mocked(analyzer3, {'q': 1.0, 'T': 60000, 'p': 100, 'm': True}, ref_ts3)   # -1.0
    process_trade_mocked(analyzer3, {'q': 0.5, 'T': 70000, 'p': 100, 'm': False}, ref_ts3)  # +0.5
    
    stats = analyzer3.get_stats()
    cvd = float(stats['cvd'])
    expected_cvd = 2.0 - 1.0 + 0.5
    
    print("CVD calculado: %.2f" % cvd)
    print("CVD esperado: %.2f" % expected_cvd)
    
    if abs(cvd - expected_cvd) < 0.001:
        print("[PASS] CVD correto.")
    else:
        print("[FAIL] CVD incorreto!")
        all_passed = False
    
    # --- Test 4: Health Check ---
    print("\n[TEST 4] Health Check")
    print("-" * 40)
    
    health = analyzer3.health_check()
    print("Status: %s" % health['status'])
    print("Issues: %s" % health.get('issues', []))
    
    if 'status' in health:
        print("[PASS] Health check funciona.")
    else:
        print("[FAIL] Health check falhou!")
        all_passed = False
    
    # --- Test 5: Whale Detection ---
    print("\n[TEST 5] Whale Detection")
    print("-" * 40)
    
    analyzer5 = FlowAnalyzer()
    analyzer5.whale_threshold = 5.0  # Decimal
    ref_ts5 = 100000
    
    # Trade baleia (10 BTC)
    process_trade_mocked(analyzer5, {'q': 10.0, 'T': 50000, 'p': 100, 'm': False}, ref_ts5)
    # Trade varejo (0.1 BTC)
    process_trade_mocked(analyzer5, {'q': 0.1, 'T': 60000, 'p': 100, 'm': True}, ref_ts5)
    
    stats5 = analyzer5.get_stats()
    whale_delta = float(stats5['whale_delta'])
    
    print("Whale delta: %.2f" % whale_delta)
    print("Esperado: 10.0 (apenas trade de 10 BTC conta como whale)")
    
    if abs(whale_delta - 10.0) < 0.001:
        print("[PASS] Whale detection correto.")
    else:
        print("[FAIL] Whale detection incorreto!")
        all_passed = False
    
    # --- Resultado Final ---
    print("\n" + "=" * 60)
    if all_passed:
        print("TODOS OS TESTES PASSARAM!")
    else:
        print("ALGUNS TESTES FALHARAM!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)