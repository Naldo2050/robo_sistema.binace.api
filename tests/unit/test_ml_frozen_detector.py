
import pytest
import numpy as np
from ml.inference_engine import MLInferenceEngine as MLEngine
from ml.feature_calculator import LiveFeatureCalculator as MLFeatureGenerator

class TestMLFrozenDetector:
    FROZEN_PROB = 0.943161129951477

    def test_detect_frozen_from_log_evidence(self):
        engine = MLEngine()
        # Testar com features que DEVEM produzir resultado diferente
        test_cases = [
            {'price_close': 67100.0, 'return_1': 0.0, 'return_5': 0.0, 'return_10': 0.0,
             'bb_upper': 67300, 'bb_lower': 66900, 'bb_width': 0.006, 'rsi': 50.0, 'volume_ratio': 1.0},
            {'price_close': 65000.0, 'return_1': -0.03, 'return_5': -0.08, 'return_10': -0.12,
             'bb_upper': 67300, 'bb_lower': 66900, 'bb_width': 0.006, 'rsi': 15.0, 'volume_ratio': 5.0},
            {'price_close': 70000.0, 'return_1': 0.03, 'return_5': 0.08, 'return_10': 0.12,
             'bb_upper': 70500, 'bb_lower': 69500, 'bb_width': 0.014, 'rsi': 85.0, 'volume_ratio': 3.0},
        ]
        
        frozen_count = 0
        probs = []
        for features in test_cases:
            pred = engine.predict(features)
            prob = pred.get('prob_up', 0.5)
            probs.append(prob)
            if abs(prob - self.FROZEN_PROB) < 1e-10:
                frozen_count += 1
        
        # O modelo não deve retornar sempre o mesmo valor se os inputs são variados
        assert frozen_count < 2, f"Modelo parece congelado em {self.FROZEN_PROB}. Probs: {probs}"

    def test_feature_names_match_model(self):
        engine = MLEngine()
        gen = MLFeatureGenerator()
        
        model_features = engine.EXPECTED_FEATURES
        gen_output = gen.compute()
        gen_features = set(gen_output.keys())
        model_features_set = set(model_features)
        
        missing_in_gen = model_features_set - gen_features
        assert not missing_in_gen, f"Feature Mismatch! Modelo espera mas gerador não produz: {missing_in_gen}"

    def test_default_detection_in_features(self):
        gen = MLFeatureGenerator()
        prices = [67076, 67090, 67100, 67114, 67101, 67120, 67156, 67100, 67080, 67095]
        
        for p in prices:
            gen.update(price=p, volume=5.0)
        
        last_features = gen.compute()
        expected = ['price_close', 'return_1', 'return_5', 'return_10', 'bb_upper', 'bb_lower', 'bb_width', 'rsi', 'volume_ratio']
        for f in expected:
            assert f in last_features, f"Feature {f} faltando!"
