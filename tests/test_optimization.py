"""
Testes de Validação da Otimização

Garante que:
1. Redução de tamanho acontece (>60%)
2. Qualidade mantida (campos críticos presentes)
3. Nenhum erro de estrutura
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from optimize_ai_payload import build_optimized_ai_payload, optimize_event_for_ai


FIXTURE_PATH = Path("tests") / "fixtures" / "sample_analysis_trigger.json"


def test_optimization_reduces_size():
    """Testa se otimização reduz tamanho em >60%."""

    original_event = load_sample_event()
    optimized = optimize_event_for_ai(original_event)

    original_size = len(json.dumps(original_event, ensure_ascii=False))
    optimized_size = len(json.dumps(optimized, ensure_ascii=False))
    reduction_pct = (1 - optimized_size / original_size) * 100

    assert reduction_pct >= 60, f"Redução insuficiente: {reduction_pct:.1f}% (esperado >=60%)"


def test_critical_fields_preserved():
    """Testa se campos críticos foram preservados."""

    optimized = optimize_event_for_ai(load_sample_event())

    critical_fields = [
        "symbol",
        "current_price",
        "flow_imbalance",
        "bid_depth_usd",
        "ask_depth_usd",
        "ohlc",
    ]
    for field in critical_fields:
        assert has_field(optimized, field), f"Campo crítico ausente: {field}"


def test_unnecessary_fields_removed():
    """Testa se campos desnecessários foram removidos."""

    optimized = optimize_event_for_ai(load_sample_event())

    unnecessary_fields = [
        "observability",
        "processing_times_ms",
        "ui_sum_ok",
        "invariants_ok",
        "timestamp_ny",
        "timestamp_sp",
    ]

    for field in unnecessary_fields:
        assert not has_field(optimized, field), f"Campo desnecessário presente: {field}"


def test_ai_payload_structure_and_size():
    """Testa estrutura do payload para IA e limite de tamanho."""

    ai_payload = build_optimized_ai_payload(load_sample_event())

    assert "price_context" in ai_payload
    assert "flow_context" in ai_payload
    assert "orderbook_context" in ai_payload
    assert "macro_context" in ai_payload

    payload_size = len(json.dumps(ai_payload, ensure_ascii=False))
    assert payload_size < 3000, f"Payload muito grande: {payload_size} chars"


def load_sample_event() -> dict[str, Any]:
    if not FIXTURE_PATH.exists():
        pytest.fail(f"Fixture não encontrada: {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def has_field(obj: Any, field: str) -> bool:
    if isinstance(obj, dict):
        if field in obj:
            return True
        return any(has_field(value, field) for value in obj.values())
    if isinstance(obj, list):
        return any(has_field(item, field) for item in obj)
    return False

