import json

from ai_analyzer_qwen import AIAnalyzer, SYSTEM_PROMPT


def test_system_prompt_requires_ptbr_and_no_think():
    p = SYSTEM_PROMPT.lower()
    assert "portugues do brasil" in p
    assert "<think>" in p


def test_sanitize_llm_text_strips_think_block_and_leaves_json():
    raw = '<think>internal reasoning</think>{"sentiment":"neutral","confidence":0.5,"action":"wait","rationale":"ok"}'
    cleaned = AIAnalyzer._sanitize_llm_text(raw)
    data = json.loads(cleaned)
    assert data["rationale"] == "ok"

