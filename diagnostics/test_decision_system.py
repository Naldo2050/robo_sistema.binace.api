
import json
import sys
import os

# Adiciona diretório para importar HybridDecisionMaker
sys.path.append(os.getcwd())
from ml.hybrid_decision import HybridDecisionMaker

class DecisionSystemTester:
    def __init__(self):
        self.maker = HybridDecisionMaker()
        self.test_cases = [
            {
                'name': 'Forte Alta (ML + LLM)',
                'ml_prob': 0.85,
                'ml_conf': 0.70,
                'ai_action': 'buy',
                'ai_conf': 0.80,
                'expected': 'buy'
            },
            {
                'name': 'Divergência Crítica (ML Buy vs LLM Sell)',
                'ml_prob': 0.90,
                'ml_conf': 0.80,
                'ai_action': 'sell',
                'ai_conf': 0.85,
                'expected': 'wait' # Devido à penalidade de conflito V5
            },
            {
                'name': 'Forte Baixa (ML + LLM)',
                'ml_prob': 0.20,
                'ml_conf': 0.80,
                'ai_action': 'sell',
                'ai_conf': 0.90,
                'expected': 'sell'
            }
        ]
    
    def run_tests(self):
        """Executa todos os testes"""
        print("\n" + "="*50)
        print("🎯 TESTE DE SISTEMA DE DECISÃO (V5 HYBRID)")
        print("="*50)
        
        passed = 0
        
        for test in self.test_cases:
            ml_pred = {"status": "ok", "prob_up": test['ml_prob'], "confidence": test['ml_conf']}
            ai_res = {"action": test['ai_action'], "confidence": test['ai_conf'], "sentiment": "neutral", "rationale": "Teste"}
            
            result = self.maker.fuse_decisions(ml_pred, ai_res)
            
            status = "✅" if result.action == test['expected'] else "❌"
            if result.action == test['expected']: passed += 1
            
            print(f"\n{status} {test['name']}:")
            print(f"  ML Prob: {test['ml_prob']:.2f} | IA Action: {test['ai_action']}")
            print(f"  → Decisão: {result.action} (conf: {result.confidence:.2f})")
            print(f"  Esperado: {test['expected']}")
            
        print("\n" + "="*50)
        print(f"📊 RESULTADO: {passed}/{len(self.test_cases)} testes passados")
        return passed == len(self.test_cases)

if __name__ == "__main__":
    tester = DecisionSystemTester()
    tester.run_tests()
