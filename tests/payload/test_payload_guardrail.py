import json
from pathlib import Path

import pytest

from market_orchestrator.ai.llm_payload_guardrail import ensure_safe_llm_payload

pytestmark = pytest.mark.payload


def test_guardrail_allows_safe_payload():
    payload = {"ai_payload": {"symbol": "BTCUSDT", "epoch_ms": 1, "price_context": {"current_price": 1}}}
    safe = ensure_safe_llm_payload(payload)
    assert safe == payload


def test_guardrail_blocks_forbidden_and_uses_ai_analysis_payload():
    candidate = {"symbol": "BTCUSDT", "epoch_ms": 1, "price_context": {"current_price": 1}}
    payload = {
        "ANALYSIS_TRIGGER": {"some": "thing"},
        "AI_ANALYSIS": {"ai_payload": candidate},
    }
    safe = ensure_safe_llm_payload(payload)
    assert safe is not None
    assert "ai_payload" in safe
    assert safe["ai_payload"].get("_v") == 2
    assert "ANALYSIS_TRIGGER" not in safe["ai_payload"]


def test_guardrail_aborts_without_candidate(tmp_path):
    payload = {"raw_event": {"a": 1}}
    safe = ensure_safe_llm_payload(payload)
    assert safe is None
