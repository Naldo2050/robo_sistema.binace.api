import json

import pytest

from market_orchestrator.ai.payload_metrics_aggregator import summarize_metrics

pytestmark = pytest.mark.payload


def test_summarize_metrics_basic(tmp_path):
    path = tmp_path / "payload_metrics.jsonl"
    lines = [
        {"payload_bytes": 100, "leak_blocked": True},
        {"payload_bytes": 200},
        {"bytes_after": 50},
        {"payload_bytes": 150, "error": "no_safe_candidate"},
        {"payload_bytes": 120, "fallback_v1": True},
        {"cache_hit": True, "section": "macro_context"},
        {"cache_hit": False, "section": "macro_context"},
    ]
    path.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n", encoding="utf-8")

    summary = summarize_metrics(str(path), last_n=20)
    assert summary["count"] == len(lines)
    # valores de bytes: 100,200,50,150,120 -> mediana 120, max 200
    assert summary["bytes_p50"] == 120
    assert summary["bytes_max"] == 200
    assert summary["guardrail_block_rate"] == pytest.approx(1 / len(lines))
    assert summary["abort_rate"] == pytest.approx(1 / len(lines))
    assert summary["fallback_rate"] == pytest.approx(1 / len(lines))
    cache_rates = summary.get("cache_hit_rate", {})
    assert cache_rates.get("macro_context") == pytest.approx(0.5)
