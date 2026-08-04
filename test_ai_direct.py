import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from market_orchestrator.ai.ai_runner import AIRunner

def test():
    os.environ["AI_ENABLED"] = "true"
    os.environ["AI_PROVIDER"] = "groq"
    
    # Initialize the AI runner
    ai_runner = AIRunner.create()
    analyzer = ai_runner.create_analyzer()
    
    # Build a minimal event_data dict based on what the bot might pass
    event_data = {
        "symbol": "BTCUSDT",
        "price": 76538.3,
        "delta": 64.96,
        "volume": 123.64,
        "signal_type": "Exaustão de Compra",
        "window": 6,
        # Add some required fields that the AI might expect
        "tipo_evento": "ANALYSIS_TRIGGER",
        "resultado_da_batalha": "COMPRA",
        "volume_total": 123.64,
        "preco_fechamento": 76538.3,
    }
    
    try:
        # The analyze method is synchronous
        result = analyzer.analyze(event_data)
        print("RESULTADO:", result)
    except Exception as e:
        print(f"ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()