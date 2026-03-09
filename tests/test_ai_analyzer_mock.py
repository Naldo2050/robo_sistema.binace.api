# tests/test_ai_analyzer_mock.py
from __future__ import annotations

import ai_analyzer_qwen as mod


def test_ai_analyzer_mock_mode(monkeypatch):
    """
    Garante que, forçando _initialize_api a modo mock, analyze()
    retorna fallback estruturado e mode='mock', sem chamar provedores externos.
    """

    # Monkeypatch de _initialize_api para não bater em Groq/OpenAI/DashScope
    def fake_init(self):
        self.mode = None      # modo mock
        self.enabled = True
        self.client = None
        self.client_async = None

    monkeypatch.setattr(mod.AIAnalyzer, "_initialize_api", fake_init)

    analyzer = mod.AIAnalyzer(health_monitor=None)

    event = {
        "tipo_evento": "Teste",
        "ativo": "BTCUSDT",
        "delta": 1.0,
        "volume_total": 100.0,
        "preco_fechamento": 50000.0,
        "resultado_da_batalha": "NEUTRAL",
    }

    result = analyzer.analyze(event)

    assert isinstance(result, dict)
    assert result["success"] is False
    assert result["is_fallback"] is True
    assert result["structured"]["_is_fallback"] is True
    assert result["mode"] == "mock"
    assert "raw_response" in result
    assert len(result["raw_response"]) > 0
