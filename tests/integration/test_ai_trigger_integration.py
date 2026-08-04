# -*- coding: utf-8 -*-
"""
Teste de integracao do gatilho de IA do EnhancedMarketBot.

Cadeia REAL exercitada:
    event_bus.publish("signal", event)
      -> _handle_signal_event (subscriber registrado no __init__)
      -> _run_ai_analysis_threaded
      -> run_ai_analysis_threaded            (ai_runner.py:255)
      -> AIRunner.build_payload              (ai_runner.py:602, build_compact_payload real)
      -> ai_analyzer.analyze                 (ai_runner.py:659)
      -> event_saver.save_event(AI_ANALYSIS) (ai_runner.py:749-761)

O que e substituido (permitido, e a "IA mockada" e a persistencia):
  - ai_analyzer (o modelo LLM real) -> FakeAnalyzer
  - event_saver (persistencia real) -> RecorderSaver

O que NAO e mockado:
  - EnhancedMarketBot.__init__ (inclui os subscribes do fix 6ed63cc)
  - _handle_signal_event / _run_ai_analysis_threaded (metodos reais do bot)
  - run_ai_analysis_threaded / build_compact_payload (funcoes reais)

Mensagens de assert em ASCII puro (sem acentos) para nao depender do
encoding do terminal em CI.

Tempo: nao ha sleep artificial - o poll retorna imediatamente quando o
evento AI_ANALYSIS e salvo (a cadeia completa leva < 1s; o custo de ~40s
da suite vem do --cov=. global do pytest.ini, comum a todos os testes).

Threads: a fixture faz teardown via bot._cleanup_handler(), que aguarda
as threads de IA, encerra o EventBus (thread da fila), o ThreadPoolExecutor
e o loop asyncio do OrderBookAnalyzer - sem threads orfas entre testes.
HealthMonitor e ClockSync sao substituidos por stubs (nao sao parte do
fluxo sob teste): o real faz HTTP de sync e mantem threads com sleeps
longos, o que tornaria o teste lento e nao deterministico.
"""
import time

import pytest

from market_orchestrator.market_orchestrator import EnhancedMarketBot


class FakeAnalyzer:
    """Substituto do AIAnalyzer real: registra a chamada e devolve resultado valido."""

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
                "reasoning": "teste de integracao",
                "direction": "COMPRA",
            },
        }

    def close(self):
        pass

    async def aclose(self):
        pass


class RecorderSaver:
    """Substituto do EventSaver: captura eventos salvos em memoria."""

    def __init__(self):
        self.saved = []

    def save_event(self, event):
        self.saved.append(event)


def _fast_ai_init(bot):
    """Inicializacao de IA rapida para o teste (sem API/ML reais)."""
    bot.ai_initialization_attempted = True
    bot.ai_analyzer = FakeAnalyzer()
    bot.ai_test_passed = True
    bot.ml_engine = None
    bot.feature_calc = None


def _build_event():
    """Evento 'signal' importante (ABSORCAO) com dados minimos p/ payload real."""
    return {
        "symbol": "BTCUSDT",
        "tipo_evento": "ABSORCAO",
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
    """Bot REAL (inclui os subscribes do fix 6ed63cc), com IA/persistencia mockadas."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "market_orchestrator.ai.ai_runner.initialize_ai_async",
        _fast_ai_init,
    )

    class _StubHealthMonitor:
        """Sem thread de monitoramento (o real cria _monitor_loop).

        Substituido de proposito, NAO por atalho: o HealthMonitor real faz
        heartbeat/OCI e mantem uma thread com sleep de ate 30s - irrelevante
        para o fluxo de IA sob teste e fonte de timing nao deterministico.
        """

        def __init__(self, *args, **kwargs):
            pass

        def heartbeat(self, module):
            pass

        def stop(self):
            pass

        def get_stats(self):
            return {}

    class _NoopClockSync:
        """Sem thread de sync (o real cria _sync_loop e faz HTTP).

        Substituido de proposito, NAO por atalho: o ClockSync real faz
        sincronizacao de relogio via HTTP (wait_for_sync de ate 5s no
        EventSaver eager) - uma dependencia externa que nao faz parte do
        fluxo de IA sob teste. Restaurar o componente real reintroduz
        chamadas de rede e flakiness em CI.
        """

        def get_server_time_ms(self):
            return int(time.time() * 1000)

        def wait_for_sync(self, timeout=5.0):
            return True

        def get_offset_seconds(self):
            return 0.0

        def stop(self):
            pass

    # Sem threads orfas: HealthMonitor real e ClockSync real (criado eager pelo
    # EventSaver e pelo FlowAnalyzer) sao substituidos por stubs (ver docstrings
    # acima: isolamento de dependencias externas HTTP/timing, nao performance).
    monkeypatch.setattr(
        "market_orchestrator.market_orchestrator.HealthMonitor",
        _StubHealthMonitor,
    )
    monkeypatch.setattr("flow_analyzer.core.get_clock_sync", lambda: _NoopClockSync())
    monkeypatch.setattr("events.event_saver.get_clock_sync", lambda: _NoopClockSync())

    # _ai_throttler e um singleton global em ai_runner.py: outros testes o usam
    # e deixam estado (cooldown/calls_this_hour) que faria should_call_ai
    # retornar False aqui - interferencia entre testes. Stub: ele e camada de
    # controle de custo, nao parte do fluxo evento->analyze->AI_ANALYSIS testado.
    monkeypatch.setattr(
        "market_orchestrator.ai.ai_runner._ai_throttler",
        type("_AlwaysAllowThrottler", (), {"should_call_ai": lambda *a, **k: True,
                                           "get_status": lambda *a, **k: {}})(),
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
    real_saver = b.event_saver  # o EventSaver real iniciou threads flush/cleanup
    saver = RecorderSaver()
    b.event_saver = saver
    b._last_ai_analysis_ts = 0.0
    yield b, saver, b.ai_analyzer
    # Teardown: threads de IA, EventBus, executor, loop asyncio + HealthMonitor
    # (via _cleanup_handler) e threads flush/cleanup do EventSaver real (stop()).
    try:
        b._cleanup_handler()
        real_saver.stop()
    except Exception as e:  # pragma: no cover - melhor esforco
        print(f"[teardown] cleanup falhou (nao-critico): {e}")


def _subscribed_handler_names(bus, topic):
    handlers = getattr(bus, "_handlers", {}) or {}
    return {getattr(fn, "__name__", str(fn)) for fn in handlers.get(topic, [])}


def test_bot_subscribes_ai_handlers_on_event_bus(bot):
    """Fix 6ed63cc: __init__ deve registrar _handle_signal_event/_handle_zone_touch_event."""
    b, _, _ = bot
    names = _subscribed_handler_names(b.event_bus, "signal")
    assert "_handle_signal_event" in names, (
        "subscriber 'signal' ausente - reverta o fix 6ed63cc em "
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

    # Polling curto (sem sleep fixo): retorna assim que AI_ANALYSIS e salvo.
    deadline = time.time() + 10.0
    while time.time() < deadline and not saver.saved:
        time.sleep(0.05)

    assert saver.saved, (
        "nenhum evento AI_ANALYSIS salvo - o fluxo de IA nao foi disparado; "
        "confira os subscribes 'signal'/'zone_touch' no __init__ (fix 6ed63cc)"
    )

    ai_event = saver.saved[0]

    # 1) analyze foi chamada de verdade (ai_runner.py:659)
    assert len(analyzer.calls) == 1
    analyzed = analyzer.calls[0]
    assert analyzed.get("tipo_evento") == "ABSORCAO"
    # 2) payload real foi anexado antes da chamada (ai_runner.py:602/611)
    assert analyzed.get("ai_payload"), "ai_payload ausente no evento analisado"
    assert not analyzed["ai_payload"].get("_emergency")

    # 3) AI_ANALYSIS salvo com resultado e payload (ai_runner.py:749-761)
    assert ai_event["tipo_evento"] == "AI_ANALYSIS"
    assert ai_event["symbol"] == "BTCUSDT"
    assert ai_event["ai_result"]["action"] == "BUY"
    assert ai_event["ai_payload"] and ai_event["ai_payload"].get("symbol") == "BTCUSDT"
    assert ai_event["timestamp_ms"] and ai_event["anchor_price"] == 66355.5

    # 4) nenhuma thread de IA pendente apos a analise (ai_runner.py:788-815)
    deadline = time.time() + 5.0
    while time.time() < deadline and b.ai_thread_pool:
        time.sleep(0.05)
    assert b.ai_thread_pool == [], "threads de IA nao foram encerradas"
