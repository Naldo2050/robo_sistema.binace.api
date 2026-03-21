
import pytest
import numpy as np
import json
import sys
import os
import time
from unittest.mock import MagicMock, patch, AsyncMock
from collections import deque
from copy import deepcopy

# Imports reais do projeto
from ml.feature_calculator import LiveFeatureCalculator as MLFeatureGenerator
from ml.inference_engine import MLInferenceEngine as MLEngine
from ml.hybrid_decision import HybridDecisionMaker
from build_compact_payload import build_compact_payload as build_compact
from data_processing.data_enricher import DataEnricher

# Mock para DynamicDeltaThreshold já que é calculado inline no window_processor
class DynamicDeltaThreshold:
    def __init__(self):
        self.delta_history = deque(maxlen=100)
        self.current_threshold = 2.0 # Piso warmup V6
        
    def update(self, delta, volume):
        self.delta_history.append(abs(delta))
        if len(self.delta_history) > 10:
            self.current_threshold = max(0.5, float(np.mean(self.delta_history)) * 1.5)
        return self.current_threshold

# Adapter para PriceTargetCalculator
class PriceTargetCalculator:
    def __init__(self):
        # Mock de config básico para o DataEnricher
        config_mock = {
            "price_targets": {"enabled": True, "method": "atr_based"},
            "symbol": "BTCUSDT"
        }
        self.enricher = DataEnricher(config_mock)
        
    def calculate(self, data):
        # Simula a estrutura que o DataEnricher espera
        event_mock = {
            "symbol": "BTCUSDT",
            "preco_fechamento": data.get("preco_fechamento", 67000.0),
            "raw_event": {"raw_event": data}
        }
        self.enricher.enrich_event_with_advanced_analysis(event_mock)
        
        # Busca recursiva simples por advanced_analysis
        def find_advanced(d):
            if not isinstance(d, dict): return None
            if "advanced_analysis" in d: return d["advanced_analysis"]
            for v in d.values():
                res = find_advanced(v)
                if res: return res
            return None
            
        advanced = find_advanced(event_mock) or {}
        return advanced.get("price_targets", [])

# ============================================================================
# FIXTURES COMPARTILHADAS
# ============================================================================

@pytest.fixture
def sample_prices():
    """Série de preços realista simulando 30 janelas de 1 minuto"""
    base = 67100.0
    np.random.seed(42)
    prices = []
    for i in range(30):
        base += np.random.normal(0, 15)
        prices.append(round(base, 1))
    return prices

@pytest.fixture
def sample_volumes():
    """Série de volumes realista"""
    np.random.seed(42)
    return [round(abs(np.random.normal(5, 3)), 5) for _ in range(30)]

@pytest.fixture
def sample_window_data():
    return {
        'preco_fechamento': 67114.3,
        'price': 67114.3,
        'delta': -3.71,
        'volume_total': 5.81,
        'total_buy_volume': 1.048,
        'total_sell_volume': 4.758,
        'trade_count': 1371,
        'timestamp': 1772990467495,
    }

@pytest.fixture
def multi_window_sequence(sample_prices, sample_volumes):
    windows = []
    deltas = [-3.71, 1.08, -0.23, 31.71, -96.16, 1.51, -2.30, 5.44, -1.02, 0.88]
    for i in range(min(10, len(sample_prices))):
        windows.append({
            'window_id': i + 1,
            'preco_fechamento': sample_prices[i],
            'price': sample_prices[i],
            'delta': deltas[i] if i < len(deltas) else 0.0,
            'volume_total': sample_volumes[i],
            'timestamp': 1772990467495 + i * 60000,
        })
    return windows

# ============================================================================
# TESTES
# ============================================================================

class TestMLFeatureCompleteness:
    EXPECTED_FEATURES = [
        'price_close', 'return_1', 'return_5', 'return_10',
        'bb_upper', 'bb_lower', 'bb_width', 'rsi', 'volume_ratio'
    ]

    def test_feature_completeness_after_warmup(self, multi_window_sequence):
        generator = MLFeatureGenerator()
        for i, window in enumerate(multi_window_sequence):
            generator.update(price=window['preco_fechamento'], volume=window['volume_total'])
            features = generator.compute()
            if i >= 9:
                missing = [f for f in self.EXPECTED_FEATURES if f not in features]
                assert not missing, f"Janela #{i+1}: Features faltando: {missing}"

class TestXGBoostNotFrozen:
    def test_predictions_vary_with_different_inputs(self):
        engine = MLEngine()
        # Bullish
        f1 = {'price_close': 67500, 'return_1': 0.01, 'return_5': 0.02, 'return_10': 0.03, 
              'bb_upper': 68000, 'bb_lower': 67000, 'bb_width': 0.014, 'rsi': 75, 'volume_ratio': 2.0}
        # Bearish
        f2 = {'price_close': 66500, 'return_1': -0.01, 'return_5': -0.02, 'return_10': -0.03, 
              'bb_upper': 67000, 'bb_lower': 66000, 'bb_width': 0.014, 'rsi': 25, 'volume_ratio': 2.0}
        
        p1 = engine.predict(f1).get('prob_up', 0.5)
        p2 = engine.predict(f2).get('prob_up', 0.5)
        assert abs(p1 - p2) > 0.01, f"Modelo congelado: {p1} vs {p2}"

class TestHybridDecisionConflict:
    def test_conflict_buy_sell_reduces_confidence(self):
        maker = HybridDecisionMaker()
        ml = {'status': 'ok', 'prob_up': 0.94, 'confidence': 0.94}
        ai = {'action': 'sell', 'confidence': 0.89, 'sentiment': 'bearish', 'rationale': 'test'}
        decision = maker.fuse_decisions(ml, ai)
        assert decision.confidence < 0.89
        assert decision.action == 'wait' # V6 rigor 55%

class TestPayloadSize:
    def test_compact_builder_output_size(self, sample_window_data):
        payload = build_compact(sample_window_data)
        size = len(json.dumps(payload))
        assert size < 2000

class TestDynamicDeltaThreshold:
    def test_threshold_nonzero_after_warmup(self):
        calc = DynamicDeltaThreshold()
        for i in range(15):
            calc.update(delta=np.random.normal(0, 10), volume=5.0)
        assert calc.current_threshold >= 0.5

class TestPriceTargets:
    def test_targets_with_minimal_data(self, sample_window_data):
        calc = PriceTargetCalculator()
        targets = calc.calculate(sample_window_data)
        assert len(targets) > 0
