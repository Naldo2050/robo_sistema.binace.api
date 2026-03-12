"""Testes para o script de diagnóstico"""

import json
from pathlib import Path
from diagnose_optimization import (
    OptimizationDiagnostic,
    DiagnosticConfig,
    EventAnalysis,
)


def test_detect_non_optimized_event():
    """Testa detecção de evento não otimizado"""
    
    # Evento com campos proibidos
    event = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "epoch_ms": 1768951080000,
        "observability": {},  # Campo proibido
        "enriched_snapshot": {},  # Duplicação
        "contextual_snapshot": {},
    }
    
    config = DiagnosticConfig()
    diagnostic = OptimizationDiagnostic(config)
    
    analysis = diagnostic._analyze_event(event)
    
    assert analysis is not None
    assert not analysis.is_optimized
    assert "observability" in analysis.has_forbidden_fields
    assert "enriched_snapshot + contextual_snapshot" in analysis.has_duplications


def test_detect_optimized_event():
    """Testa detecção de evento otimizado"""
    
    # Evento otimizado
    event = {
        "tipo_evento": "AI_ANALYSIS",
        "symbol": "BTCUSDT",
        "epoch_ms": 1768951080000,
        "_v": 2,  # Schema v2
        "price_context": {},
        "flow_context": {},
    }
    
    config = DiagnosticConfig()
    diagnostic = OptimizationDiagnostic(config)
    
    analysis = diagnostic._analyze_event(event)
    
    assert analysis is not None
    assert analysis.is_optimized
    assert analysis.schema_version == 2
    assert len(analysis.has_forbidden_fields) == 0


if __name__ == "__main__":
    test_detect_non_optimized_event()
    test_detect_optimized_event()
    print("✅ Todos os testes passaram!")