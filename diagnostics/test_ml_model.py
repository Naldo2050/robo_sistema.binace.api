
import json
import os
import numpy as np
import xgboost as xgb
import shutil
from datetime import datetime

class MLModelTester:
    def __init__(self):
        self.model_path = "ml/models/xgb_model_latest.json"
        
    def test_model_integrity(self):
        """Verifica integridade do modelo"""
        print("\n" + "="*50)
        print("🧠 TESTE DE INTEGRIDADE DO MODELO ML")
        print("="*50)
        
        issues = []
        
        # 1. Verificar se arquivo existe
        if not os.path.exists(self.model_path):
            issues.append(f"❌ Arquivo do modelo não encontrado: {self.model_path}")
            print(issues[-1])
            return False
        
        try:
            # Tenta carregar como Booster do XGBoost
            model = xgb.Booster()
            model.load_model(self.model_path)
            
            # Features esperadas pelo robô
            expected = ['price_close', 'return_1', 'return_5', 'return_10', 'bb_upper', 
                       'bb_lower', 'bb_width', 'rsi', 'volume_ratio']
            
            # Criar dados de teste (dummy) para verificar predições
            dummy_features = np.random.randn(10, len(expected))
            dmatrix = xgb.DMatrix(dummy_features, feature_names=expected)
            predictions = model.predict(dmatrix)
            
            print(f"✅ Modelo carregado com sucesso.")
            print(f"✅ Predições geradas: {len(predictions)} amostras")
            print(f"   Média: {np.mean(predictions):.4f} | Std: {np.std(predictions):.4f}")
            
            # Verificar se predictions variam (frozen model check)
            if np.std(predictions) < 1e-6:
                issues.append("⚠️ Predições idênticas detectadas - Modelo pode estar degenerado ou congelado.")
            
        except Exception as e:
            issues.append(f"❌ Erro ao testar modelo: {e}")
        
        # Report
        if issues:
            print("\n⚠️ PROBLEMAS IDENTIFICADOS:")
            for issue in issues:
                print(f"  {issue}")
            return False
        else:
            print("\n✅ MODELO VALIDADO COM SUCESSO")
            return True

if __name__ == "__main__":
    tester = MLModelTester()
    tester.test_model_integrity()
