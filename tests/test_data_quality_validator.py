#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste básico para o método detect_anomalies da classe DataQualityValidator
"""

import pytest
from data_quality_validator import DataQualityValidator


def test_detect_anomalies_basic():
    """Testa se o método detect_anomalies é inicializado corretamente"""
    validator = DataQualityValidator()
    assert hasattr(validator, 'detect_anomalies')
    assert callable(getattr(validator, 'detect_anomalies'))


def test_detect_anomalies_no_anomalies():
    """Testa detecção de anomalias em dados normais"""
    validator = DataQualityValidator()
    
    current_data = {
        "volume_total": 10,
        "spread": 0.1,
        "close": 50000,
        "open": 49990,
        "high": 50010,
        "low": 49980,
        "flow_imbalance": 0.2,
        "bid_depth_usd": 100000,
        "ask_depth_usd": 100000,
        "trades_per_second": 100,
        "realized_vol": 0.02
    }
    
    result = validator.detect_anomalies(current_data)
    
    assert result["anomalies_detected"] is False
    assert result["count"] == 0
    assert result["max_severity"] == "NONE"
    assert result["risk_elevated"] is False
    assert "No anomalies detected" in result["summary"]


def test_detect_anomalies_with_anomalies():
    """Testa detecção de anomalias em dados com valores extremos"""
    validator = DataQualityValidator()
    
    current_data = {
        "volume_total": 200,  # Spike de volume (>100 BTC)
        "spread": 10,         # Spread anormal (> $5)
        "close": 50000,
        "open": 49000,
        "high": 51000,
        "low": 48000,
        "flow_imbalance": 0.8,  # Desequilíbrio extremo (>0.7)
        "bid_depth_usd": 10000,
        "ask_depth_usd": 40000,
        "trades_per_second": 300,  # Intensidade alta (>200)
        "realized_vol": 0.1
    }
    
    result = validator.detect_anomalies(current_data)
    
    assert result["anomalies_detected"] is True
    assert result["count"] > 0
    assert result["max_severity"] in ["CRITICAL", "HIGH"]
    assert result["risk_elevated"] is True
    assert len(result["types_found"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])