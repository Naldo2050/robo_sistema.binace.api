import asyncio
import time
import logging
import sys
import os

# Ajusta o path para encontrar orderbook_analyzer
sys.path.append(os.getcwd())

from unittest.mock import MagicMock, patch
from orderbook_analyzer import OrderBookAnalyzer
from orderbook_core.orderbook import OrderBookSnapshot

async def debug_keyerror():
    logging.basicConfig(level=logging.INFO)
    
    # Mock TimeManager
    mock_tm = MagicMock()
    mock_tm.now_ms.return_value = int(time.time() * 1000)
    mock_tm.build_time_index.return_value = {"timestamp_ny": "...", "timestamp_utc": "..."}
    
    # Prevenir sync de tempo no __init__
    with patch('orderbook_analyzer.OrderBookAnalyzer._fetch_time_offset', return_value=0):
        analyzer = OrderBookAnalyzer(symbol="BTCUSDT")
        analyzer.time_manager = mock_tm
        # Mockar slog para evitar falhas se não inicializado
        analyzer.slog = MagicMock()
        analyzer.tracer = MagicMock()
        analyzer.tracer.start_span.return_value.__enter__.return_value = MagicMock()
    
    # Snapshot que causou erro nos testes
    snapshot = OrderBookSnapshot(
        symbol="BTCUSDT",
        last_update_id=123,
        bids=[(50000.0, 1.5), (49900.0, 2.0)],
        asks=[(50100.0, 1.2), (50200.0, 3.0)],
        timestamp=time.time()
    )
    
    print("\n--- TESTANDO _validate_snapshot ---")
    try:
        res = analyzer._validate_snapshot(snapshot)
        print(f"Resultado _validate_snapshot: {res[0]}")
    except Exception as e:
        import traceback
        traceback.print_exc()

    print("\n--- TESTANDO analyze_orderbook (wrapper) ---")
    try:
        res = analyzer.analyze_orderbook(snapshot)
        print(f"Resultado analyze_orderbook: {res.get('is_valid')}")
    except Exception as e:
        import traceback
        traceback.print_exc()

    print("\n--- TESTANDO analyze (direto) ---")
    try:
        res = await analyzer.analyze(snapshot.__dict__ if hasattr(snapshot, '__dict__') else snapshot)
        print(f"Resultado analyze: {res.get('is_valid')}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_keyerror())
