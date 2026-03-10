from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import ai_analyzer_qwen as ai_mod
from build_compact_payload import build_compact_payload
from event_bus import EventBus
from health_monitor import HealthMonitor
from market_orchestrator.connection.robust_connection import RobustConnectionManager
from ml.hybrid_decision import fuse_decisions


SAMPLE_EVENT = {
    "tipo_evento": "Absorção",
    "ativo": "BTCUSDT",
    "symbol": "BTCUSDT",
    "delta": 12.5,
    "volume_total": 1000.0,
    "preco_fechamento": 95000.0,
    "resultado_da_batalha": "NEUTRAL",
}


def _make_analyzer(monkeypatch: pytest.MonkeyPatch) -> ai_mod.AIAnalyzer:
    def fake_init(self):
        self.mode = "groq"
        self.enabled = True
        self.client = object()
        self.client_async = None

    monkeypatch.setattr(ai_mod.AIAnalyzer, "_initialize_api", fake_init)
    analyzer = ai_mod.AIAnalyzer(health_monitor=None)
    monkeypatch.setattr(analyzer, "_should_test_connection", lambda: False)
    return analyzer


def test_groq_json_validate_failed_generates_structured_fallback(monkeypatch):
    analyzer = _make_analyzer(monkeypatch)
    monkeypatch.setattr(
        analyzer,
        "_call_openai_compatible",
        lambda prompt, max_retries=3: ("", "json_validate_failed"),
    )

    result = analyzer.analyze(dict(SAMPLE_EVENT))

    assert result["success"] is False
    assert result["is_fallback"] is True
    assert result["fallback_reason"] == "json_validate_failed"
    assert result["structured"]["_fallback_reason"] == "json_validate_failed"
    assert "**Interpretação (mock):**" not in result["raw_response"]
    assert json.loads(result["raw_response"])["_is_fallback"] is True


def test_invalid_parser_output_generates_structured_fallback(monkeypatch):
    analyzer = _make_analyzer(monkeypatch)
    monkeypatch.setattr(
        analyzer,
        "_call_openai_compatible",
        lambda prompt, max_retries=3: ("not-json-response", None),
    )

    result = analyzer.analyze(dict(SAMPLE_EVENT))

    assert result["success"] is False
    assert result["is_fallback"] is True
    assert result["fallback_reason"] == "json_parse_error"
    assert result["structured"]["rationale"] == "llm_error_json_parse_error"
    assert json.loads(result["raw_response"])["action"] == "wait"


def test_groq_uses_standard_prompt_instead_of_compressed(monkeypatch):
    analyzer = _make_analyzer(monkeypatch)
    prompt = analyzer._get_system_prompt()

    assert "Responda SOMENTE com JSON valido." in prompt
    assert "texto curto PT-BR" in prompt
    assert "_cached=secoes omitidas" not in prompt


def test_groq_payload_summary_is_reduced(monkeypatch):
    analyzer = _make_analyzer(monkeypatch)
    payload = {
        "symbol": "BTCUSDT",
        "trigger": "ANALYSIS_TRIGGER",
        "price": {"c": 100.0, "vwap": 101.0, "shape": "P", "auction": "balanced", "extra": 1},
        "regime": {"trend": "DOWN", "structure": "RANGE", "session": "NY", "sentiment": "BEARISH", "extra": 1},
        "vp": {"daily": {"poc": 99.0, "vah": 102.0, "val": 98.0, "extra": 1}},
        "ob": {"bid": 1000, "ask": 900, "imb": 0.1, "top5_imb": 0.08, "walls": 4},
        "flow": {"net_1m": 10, "imb": 0.2, "agg_buy": 55, "absorption": "SELL_ABS", "trend": "down", "foo": "bar"},
        "tf": {"1m": {"t": "DOWN", "ema": 100, "rsi": 40, "reg": "RANGE", "macd": [0, 0]}},
        "quant": {"prob_up": 0.54, "conf": 0.08, "noise": 123},
        "cross": {"eth_7d": 0.9},
        "deriv": {"btc_oi": 1},
        "whale": {"score": 1},
    }

    reduced = analyzer._build_groq_payload_summary(payload)

    assert "cross" not in reduced
    assert "deriv" not in reduced
    assert "whale" not in reduced
    assert reduced["price"] == {"c": 100.0, "vwap": 101.0, "shape": "P", "auction": "balanced"}
    assert "extra" not in reduced["price"]


def test_groq_openai_call_uses_response_format_when_supported(monkeypatch):
    analyzer = _make_analyzer(monkeypatch)
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Resp",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Msg",
                                    (),
                                    {"content": '{"sentiment":"neutral","confidence":0.2,"action":"wait","rationale":"ok","entry_zone":null,"invalidation_zone":null,"region_type":null}'},
                                )()
                            },
                        )()
                    ]
                },
            )()

    analyzer.client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": _FakeCompletions()})()},
    )()

    content, error = analyzer._call_openai_compatible("{}", max_retries=1)

    assert error is None
    assert content
    assert captured["response_format"] == {"type": "json_object"}


def test_groq_openai_call_falls_back_when_json_mode_is_rejected(monkeypatch):
    analyzer = _make_analyzer(monkeypatch)
    calls = []

    class _FakeCompletions:
        def create(self, **kwargs):
            calls.append(dict(kwargs))
            if len(calls) == 1:
                raise RuntimeError("Error code: 400 - {'error': {'code': 'json_validate_failed'}}")
            return type(
                "Resp",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "finish_reason": "stop",
                                "message": type(
                                    "Msg",
                                    (),
                                    {"content": '{"sentiment":"neutral","confidence":0.2,"action":"wait","rationale":"ok","entry_zone":null,"invalidation_zone":null,"region_type":null}'},
                                )(),
                            },
                        )()
                    ]
                },
            )()

    analyzer.client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": _FakeCompletions()})()},
    )()

    content, error = analyzer._call_openai_compatible("{}", max_retries=1)

    assert error is None
    assert content
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]


def test_try_parse_json_dict_extracts_json_from_markdown_or_text(monkeypatch):
    analyzer = _make_analyzer(monkeypatch)
    raw = (
        "Okay, let me think first.\n\n"
        "```json\n"
        '{"sentiment":"neutral","confidence":0.2,"action":"wait","rationale":"ok","entry_zone":null,"invalidation_zone":null,"region_type":null}\n'
        "```"
    )

    parsed = analyzer._try_parse_json_dict(raw)

    assert parsed is not None
    assert parsed["action"] == "wait"


def test_groq_uses_higher_max_tokens_budget(monkeypatch):
    analyzer = _make_analyzer(monkeypatch)

    params = analyzer._get_model_params()

    assert params["max_tokens"] >= 400


def test_ai_analyze_ok_emitted_only_for_valid_json(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    analyzer = _make_analyzer(monkeypatch)
    valid_json = (
        '{"sentiment":"neutral","confidence":0.42,"action":"wait",'
        '"rationale":"sem setup claro","entry_zone":null,'
        '"invalidation_zone":null,"region_type":null}'
    )
    monkeypatch.setattr(
        analyzer,
        "_call_openai_compatible",
        lambda prompt, max_retries=3: (valid_json, None),
    )

    valid_result = analyzer.analyze(dict(SAMPLE_EVENT))
    assert valid_result["success"] is True
    assert any('"event": "ai_analyze_ok"' in rec.getMessage() for rec in caplog.records)

    caplog.clear()
    monkeypatch.setattr(
        analyzer,
        "_call_openai_compatible",
        lambda prompt, max_retries=3: ("", "json_validate_failed"),
    )
    invalid_result = analyzer.analyze(dict(SAMPLE_EVENT))

    assert invalid_result["success"] is False
    assert not any('"event": "ai_analyze_ok"' in rec.getMessage() for rec in caplog.records)
    assert any(
        '"event": "ai_provider_error"' in rec.getMessage()
        or '"event": "ai_response_invalid"' in rec.getMessage()
        for rec in caplog.records
    )


def test_hybrid_decision_source_is_not_llm_when_llm_fails():
    ml_prediction = {"status": "ok", "prob_up": 0.82, "confidence": 0.91}
    ai_result = {
        "sentiment": "neutral",
        "confidence": 0.0,
        "action": "wait",
        "rationale": "llm_error_json_validate_failed",
        "entry_zone": None,
        "invalidation_zone": None,
        "region_type": None,
        "_is_fallback": True,
        "_is_valid": False,
        "_fallback_reason": "json_validate_failed",
    }

    decision = fuse_decisions(ml_prediction, ai_result)

    assert decision.source != "llm"
    assert decision.source == "model"
    assert decision.llm_is_fallback is True
    assert decision.llm_fallback_reason == "json_validate_failed"


def test_build_compact_payload_handles_missing_tf_and_vp_is_optional(caplog):
    caplog.set_level(logging.INFO)
    event = {
        "tipo_evento": "Absorção",
        "symbol": "BTCUSDT",
        "preco_fechamento": 96000.0,
        "market_environment": {
            "trend_direction": "Alta",
            "market_structure": "Range",
        },
        "fluxo_continuo": {
            "cvd": 1.2,
            "order_flow": {
                "net_flow_1m": 10.0,
                "flow_imbalance": 0.2,
                "aggressive_buy_pct": 55.0,
                "buy_sell_ratio": {"buy_sell_ratio": 1.2},
            },
        },
        "orderbook_data": {
            "bid_depth_usd": 100000.0,
            "ask_depth_usd": 90000.0,
            "imbalance": 0.1,
            "depth_metrics": {"depth_imbalance": 0.08},
        },
        "raw_event": {
            "historical_vp": {
                "daily": {"poc": 95500.0, "vah": 96500.0, "val": 94500.0}
            }
        },
    }

    payload = build_compact_payload(event)

    assert "tf" in payload
    assert payload["tf"]
    assert "1m" in payload["tf"]
    assert payload["tf"]["1m"]["t"] == "UP"
    assert not any("BUILD_COMPACT: DADOS FALTANDO ['VP']" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_shutdown_is_idempotent(monkeypatch, caplog):
    caplog.set_level(logging.INFO)

    bus = EventBus(max_queue_size=10, deduplication_window=1)
    bus.shutdown()
    bus.shutdown()

    monitor = HealthMonitor(check_interval_seconds=1)
    monitor.stop()
    monitor.stop()

    analyzer = _make_analyzer(monkeypatch)
    analyzer.close()
    analyzer.close()

    manager = RobustConnectionManager(
        stream_url="wss://example.invalid/ws",
        symbol="BTCUSDT",
    )
    try:
        await manager.disconnect()
    except Exception as e:
        logger.error(f"Erro em operação async: {e}")
        raise
    try:
        await manager.disconnect()
    except Exception as e:
        logger.error(f"Erro em operação async: {e}")
        raise

    messages = [rec.getMessage() for rec in caplog.records]
    assert sum("EventBus desligado" in msg for msg in messages) == 1
    assert sum("HealthMonitor parado." in msg for msg in messages) == 1
    assert sum("Disconnecting GroqCloud..." in msg for msg in messages) == 1
    assert sum("🛑 Desconectando..." in msg for msg in messages) == 1


def test_oci_warning_message_has_correct_path_and_no_pleasee():
    source = Path("infrastructure/oci/monitoring.py")
    file_text = source.read_text(encoding="utf-8")

    assert "~/.oci/config" in file_text
    assert "pleasee" not in file_text.lower()
    assert "Verifique Instance Principal" in file_text
    assert "Could not find config file" in file_text
    assert "motivo=%s" in file_text