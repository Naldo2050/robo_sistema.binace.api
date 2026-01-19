import json
import logging

import pytest

from ai_analyzer_qwen import _evaluate_payload_tripwires, _log_payload_tripwires

pytestmark = pytest.mark.payload


def test_tripwire_triggers_warning(caplog, monkeypatch):
    summary = {
        "fallback_rate": 0.2,
        "abort_rate": 0.0,
        "guardrail_block_rate": 0.0,
        "bytes_p95": 7000,
        "cache_hit_rate": {"macro_context": 0.4},
    }
    tripwires = {
        "fallback_rate_max": 0.005,
        "abort_rate_max": 0.0001,
        "guardrail_block_rate_max": 0.01,
        "bytes_p95_max": 6144,
        "cache_hit_rate_min": {"macro_context": 0.6},
    }

    # Força thresholds customizados para este teste
    monkeypatch.setattr(
        "ai_analyzer_qwen.get_llm_payload_config",
        lambda: {"tripwires": tripwires},
    )

    with caplog.at_level(logging.WARNING):
        _log_payload_tripwires(summary)

    # Deve haver warn com PAYLOAD_TRIPWIRE_TRIGGERED
    assert any("PAYLOAD_TRIPWIRE_TRIGGERED" in record.message for record in caplog.records)

    violations = _evaluate_payload_tripwires(summary, tripwires)
    assert "fallback_rate" in violations
    assert "bytes_p95" in violations
    assert "cache_hit_rate.macro_context" in violations
