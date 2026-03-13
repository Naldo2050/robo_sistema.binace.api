
import sys
import os
import json
import traceback
import numpy as np
from collections import deque

# Setup path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Imports reais
from ml.feature_calculator import LiveFeatureCalculator as MLFeatureGenerator
from ml.inference_engine import MLInferenceEngine as MLEngine
from ml.hybrid_decision import HybridDecisionMaker
from build_compact_payload import build_compact_payload as build_compact

# Cores para terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def run_check(name, check_fn):
    try:
        result, details = check_fn()
        status = f"{GREEN}✅ PASS{RESET}" if result else f"{RED}❌ FAIL{RESET}"
        print(f"  {status} {name}")
        if not result and details:
            for line in details.split('\n'):
                print(f"         {YELLOW}{line}{RESET}")
        return result
    except Exception as e:
        print(f"  {RED}💥 ERROR{RESET} {name}: {e}")
        return False

def check_ml_features():
    gen = MLFeatureGenerator()
    prices = [67076, 67090, 67100, 67114, 67101, 67120, 67156, 67100, 67080, 67095,
              67110, 67088, 67105, 67130, 67115]
    for p in prices:
        gen.update(price=p, volume=5.0)
    features = gen.compute()
    expected = ['price_close', 'return_1', 'return_5', 'return_10', 'bb_upper', 'bb_lower', 'bb_width', 'rsi', 'volume_ratio']
    missing = [f for f in expected if f not in features]
    if missing:
        return False, f"Features faltando após 15 janelas: {missing}"
    return True, ""

def check_model_not_frozen():
    engine = MLEngine()
    f_n = {'price_close': 67100, 'return_1': 0.0, 'return_5': 0.0, 'return_10': 0.0, 
           'bb_upper': 67300, 'bb_lower': 66900, 'bb_width': 0.006, 'rsi': 50.0, 'volume_ratio': 1.0}
    f_b = {'price_close': 65000, 'return_1': -0.03, 'return_5': -0.08, 'return_10': -0.12, 
           'bb_upper': 67300, 'bb_lower': 66900, 'bb_width': 0.006, 'rsi': 15.0, 'volume_ratio': 5.0}
    p_n = engine.predict(f_n).get('prob_up', 0.5)
    p_b = engine.predict(f_b).get('prob_up', 0.5)
    diff = abs(p_n - p_b)
    if diff < 0.01:
        return False, f"Modelo congelado! N={p_n:.4f}, B={p_b:.4f}, Diff={diff:.6f}"
    return True, f"OK: Diff={diff:.4f}"

def check_hybrid_conflict():
    maker = HybridDecisionMaker()
    ml = {'status': 'ok', 'prob_up': 0.94, 'confidence': 0.94}
    ai = {'action': 'sell', 'confidence': 0.89, 'sentiment': 'bearish', 'rationale': 'test'}
    decision = maker.fuse_decisions(ml, ai)
    if decision.confidence >= 0.89:
        return False, f"Conflito aceito sem penalidade! Conf={decision.confidence:.2f}"
    return True, f"Conflito tratado: Action={decision.action}, Conf={decision.confidence:.2f}"

def check_payload_size():
    event = {'symbol': 'BTCUSDT', 'preco_fechamento': 67114.3, 'janela_numero': 1, 'tipo_evento': 'TEST'}
    compact = build_compact(event)
    size = len(json.dumps(compact))
    if size > 2000:
        return False, f"Payload muito grande: {size} bytes"
    return True, f"Payload OK: {size} bytes"

def check_price_targets():
    from data_enricher import DataEnricher
    enricher = DataEnricher({"SYMBOL": "BTCUSDT"})
    event = {
        "symbol": "BTCUSDT",
        "preco_fechamento": 67000.0,
        "raw_event": {"raw_event": {"preco_fechamento": 67000.0}}
    }
    enricher.enrich_event_with_advanced_analysis(event)
    
    # Busca targets
    raw = event.get("raw_event", {})
    inner = raw.get("raw_event", {})
    advanced = inner.get("advanced_analysis", raw.get("advanced_analysis", {}))
    targets = advanced.get("price_targets", [])
    
    if not targets:
        return False, "Price targets vazios!"
    return True, f"Targets gerados: {len(targets)} (ex: {targets[0]['source']})"

def main():
    print(f"\n{BOLD}{'='*70}")
    print(f"🔍 DIAGNÓSTICO DO SISTEMA - Verificação de Problemas Conhecidos")
    print(f"{'='*70}{RESET}\n")
    checks = [
        ("P0: ML Features Completas", check_ml_features),
        ("P0: Modelo XGBoost Não Congelado", check_model_not_frozen),
        ("P1: Conflitos Híbridos Tratados", check_hybrid_conflict),
        ("P2: Payload Size", check_payload_size),
        ("P3: Price Targets Fallback", check_price_targets),
    ]
    passed = 0
    for name, fn in checks:
        if run_check(name, fn): passed += 1
    print(f"\n📊 RESULTADO: {passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1

if __name__ == '__main__':
    sys.exit(main())
