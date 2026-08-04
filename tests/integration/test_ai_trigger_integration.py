"""
Teste de integração do gatilho de IA do EnhancedMarketBot.

Cadeia REAL exercitada:
    event_bus.publish("signal", event)
      -> _handle_signal_event (subscriber registrado no __init__)
      -> _run_ai_analysis_threaded
      -> run_ai_analysis_threaded            (ai_runner.py:255)
      -> AIRunner.build_payload              (ai_runner.py:602, build_compact_payload real)
      -> ai_analyzer.analyze                 (ai_runner.py:659)
      -> event_saver.save_event(AI_ANALYSIS) (ai_runner.py:749-761)

O que é substituído (permitido, é a "IA mockada" e a persistência):
  - ai_analyzer (o modelo LLM real) -> FakeAnalyzer
  - event_saver (persistência real) -> RecorderSaver

O que NÃO é mockado:
  - EnhancedMarketBot.__init__ (inclui os subscribes do fix 6ed63cc)
  - _handle_signal_event / _run_ai_analysis_threaded (métodos reais do bot)
  - run_ai_analysis_threaded / build_compact_payload (funções reais)

Se o fix 6ed63cc (event_bus.subscribe("signal"/"zone_touch")) for revertido,
estes testes falham: sem subscriber o evento é descartado e AI_ANALYSIS nunca
é salvo.
"""
import time

import pytest

from market_orchestrator.market_orchestrator import EnhancedMarketBot


class FakeAnalyzer:
    """Substituto do AIAnalyzer real: registra a chamada e devolve resultado válido."""

    def __init__(self):
        self.calls = []

    def analyze(self, event_data):
        self.calls.append(event_data)
        return {
            "success": True,
            "status": "ok",
            "is_fallback": False,
            "structured": {
                "action": "BUY",
                "confidence": 0.85,
                "reasoning": "teste de integração",
                "direction": "COMPRA",
            },
        }

    def close(self):
        pass

    async def aclose(self):
        pass


class RecorderSaver:
    """Substituto do EventSaver: captura eventos salvos em memória."""

    def __init__(self):
        self.saved = []

    def save_event(self, event):
        self.saved.append(event)


def _fast_ai_init(bot):
    """Inicialização de IA rápida para o teste (sem API/ML reais)."""
    bot.ai_initialization_attempted = True
    bot.ai_analyzer = FakeAnalyzer()
    bot.ai_test_passed = True
    bot.ml_engine = None
    bot.feature_calc = None


def _build_event():
    """Evento 'signal' importante (ABSORÇÃO) com dados mínimos p/ payload real."""
    return {
        "symbol": "BTCUSDT",
        "tipo_evento": "ABSORÇÃO",
        "resultado_da_batalha": "COMPRA",
        "severity": "HIGH",
        "delta": 1.42,
        "volume_total": 520000.0,
        "volume": 520000.0,
        "avg_volume": 400000.0,
        "preco_fechamento": 66355.5,
        "preco_atual": 66355.5,
        "epoch_ms": int(time.time() * 1000),
        "janela_numero": 7,
        "window_id": "W-7",
        "window_count": 7,
        "enriched_snapshot": {
            "ohlc": {"close": 66355.5, "open": 66370.3, "high": 66370.3, "low": 66355.5}
        },
        "contextual_snapshot": {"ohlc": {"close": 66355.5}},
        "orderbook_data": {"imbalance": 0.35, "bids": [[66350.0, 1.0]], "asks": [[66360.0, 1.0]]},
        "fluxo_continuo": {
            "volume_ratio": 1.3,
            "net_1m": -9596,
            "microstructure": {"tick_rule_sum": 0.2},
        },
        "historical_vp": {"poc": 66350.0},
        "institutional_analytics": {"whale_activity": {"score": -41}},
        "regime_analysis": {"volatility_regime": "TRENDING"},
        "features_window_id": "F-7",
    }


@pytest.fixture
def bot(monkeypatch, tmp_path):
    """Bot REAL (inclui os subscribes do fix 6ed63cc), com IA/persistência mockadas."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "market_orchestrator.ai.ai_runner.initialize_ai_async",
        _fast_ai_init,
    )
    b = EnhancedMarketBot(
        stream_url="wss://test",
        symbol="BTCUSDT",
        window_size_minutes=1,
        vol_factor_exh=2.0,
        history_size=1000,
        delta_std_dev_factor=2.0,
        context_sma_period=50,
        liquidity_flow_alert_percentage=30.0,
        wall_std_dev_factor=2.0,
    )
    saver = RecorderSaver()
    b.event_saver = saver
    b._last_ai_analysis_ts = 0.0
    yield b, saver, b.ai_analyzer


def _subscribed_handler_names(bus, topic):
    handlers = getattr(bus, "_handlers", {}) or {}
    return {getattr(fn, "__name__", str(fn)) for fn in handlers.get(topic, [])}


def test_bot_subscribes_ai_handlers_on_event_bus(bot):
    """Fix 6ed63cc: __init__ deve registrar _handle_signal_event/_handle_zone_touch_event."""
    b, _, _ = bot
    names = _subscribed_handler_names(b.event_bus, "signal")
    assert "_handle_signal_event" in names, (
        "subscriber 'signal' ausente — reverta o fix 6ed63cc em "
        "market_orchestrator.py:204-205"
    )
    zone_names = _subscribed_handler_names(b.event_bus, "zone_touch")
    assert "_handle_zone_touch_event" in zone_names


def test_signal_event_flows_to_ai_analysis_and_saves_event(bot):
    """
    Cadeia real: publish("signal") -> _handle_signal_event ->
    run_ai_analysis_threaded -> build_compact_payload -> analyze ->
    save_event(AI_ANALYSIS). Falha se o subscribe for revertido.
    """
    b, saver, analyzer = bot

    b.event_bus.publish("signal", _build_event())

    deadline = time.time() + 15.0
    while time.time() < deadline and not saver.saved:
        time.sleep(0.05)

    assert saver.saved, (
        "nenhum evento AI_ANALYSIS salvo — o fluxo de IA não foi disparado; "
        "confira os subscribes 'signal'/'zone_touch' no __init__ (fix 6ed63cc)"
    )

    ai_event = saver.saved[0]

    # 1) analyze foi chamada de verdade (ai_runner.py:659)
    assert len(analyzer.calls) == 1
    analyzed = analyzer.calls[0]
    assert analyzed.get("tipo_evento") == "ABSORÇÃO"
    # 2) payload real foi anexado antes da chamada (ai_runner.py:602/611)
    assert analyzed.get("ai_payload"), "ai_payload ausente no evento analisado"
    assert not analyzed["ai_payload"].get("_emergency")

    # 3) AI_ANALYSIS salvo com resultado e payload (ai_runner.py:749-761)
    assert ai_event["tipo_evento"] == "AI_ANALYSIS"
    assert ai_event["symbol"] == "BTCUSDT"
    assert ai_event["ai_result"]["action"] == "BUY"
    assert ai_event["ai_payload"] and ai_event["ai_payload"].get("symbol") == "BTCUSDT"
    assert ai_event["timestamp_ms"] and ai_event["anchor_price"] == 66355.5
