# -*- coding: utf-8 -*-
"""
Testes determinísticos para a mitigação causal cooperativa do WindowProcessor (Fase 5.8B - Achado #8).
"""
import asyncio
import concurrent.futures
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from events.event_bus import EventBus
from market_orchestrator.windows import window_processor as wp
from market_orchestrator.signals import signal_processor as sp


class DummyBot:
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.window_count = 0
        self.should_stop = False
        self.warming_up = False
        self.warmup_windows_remaining = 0
        self.warmup_windows_required = 3
        self._warmup_lock = threading.Lock()
        self.min_trades_for_pipeline = 1
        self.delta_std_dev_factor = 2.0
        self.context_collector = MagicMock(get_context=lambda: {})
        self.event_bus = EventBus()
        self.feature_store = MagicMock()
        self.levels = MagicMock()
        self.time_manager = MagicMock()
        self.ny_tz = None
        self.time_manager.from_timestamp_ms.return_value = MagicMock(
            isoformat=lambda **kwargs: "2026-10-04T00:00:00Z"
        )
        self.orderbook_fetch_failures = 0
        self._orderbook_refresh_lock = threading.Lock()
        self._sent_triggers = set()
        self.event_saver = MagicMock()
        self.flow_analyzer = MagicMock()
        self.delta_history = []
        self.volume_history = []
        self.volatility_history = []
        self._history_lock = threading.Lock()
        self.pattern_ohlc_history = []
        self.candles_history = []
        self.last_valid_vp = None
        self.last_valid_vp_time = 0.0

    def _build_institutional_event(self, signal):
        return {"signal": signal}

    def _enrich_signal(self, *args, **kwargs):
        from market_orchestrator.market_orchestrator import EnhancedMarketBot
        return EnhancedMarketBot._enrich_signal(self, *args, **kwargs)

    def _process_signals(self, *args, **kwargs):
        from market_orchestrator.market_orchestrator import EnhancedMarketBot
        return EnhancedMarketBot._process_signals(self, *args, **kwargs)

    def _update_histories(self, *args, **kwargs):
        pass

    def _check_zone_touches(self, *args, **kwargs):
        pass

    def _log_ml_features(self, *args, **kwargs):
        pass

    def _process_institutional_alerts(self, *args, **kwargs):
        pass

    def _log_health_check(self, *args, **kwargs):
        pass

    def _log_window_summary(self, *args, **kwargs):
        pass

    def _log_liquidity_heatmap(self, *args, **kwargs):
        pass


def _create_processor(symbol="BTCUSDT"):
    return wp.WindowProcessor(
        symbol=symbol,
        windows_minutes=[1],
        event_bus=MagicMock(),
        time_manager=MagicMock(),
    )


# 1. timeout seta cancel_event
def test_timeout_sets_cancel_event():
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    captured_event = None
    orig_submit = concurrent.futures.ThreadPoolExecutor.submit

    def _mock_submit(self, fn, *args, **kwargs):
        nonlocal captured_event
        captured_event = kwargs.get("cancel_event")
        future = orig_submit(self, fn, *args, **kwargs)
        future.result = MagicMock(side_effect=concurrent.futures.TimeoutError("45s timeout"))
        return future

    with patch.object(concurrent.futures.ThreadPoolExecutor, "submit", _mock_submit):
        processor._queue.put_nowait((bot, window_data, 1000))
        processor._worker_stop.set()
        processor._worker_loop()

    assert captured_event is not None
    assert captured_event.is_set() is True


# 2. cada janela recebe token próprio
def test_each_window_gets_own_cancel_event():
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    captured_events = []
    orig_submit = concurrent.futures.ThreadPoolExecutor.submit

    def _mock_submit(self, fn, *args, **kwargs):
        captured_events.append(kwargs.get("cancel_event"))
        future = orig_submit(self, fn, *args, **kwargs)
        return future

    def _dummy_task(*args, **kwargs):
        pass

    with patch.object(concurrent.futures.ThreadPoolExecutor, "submit", _mock_submit),          patch.object(wp, "process_window_snapshot", _dummy_task):
        processor._queue.put_nowait((bot, window_data, 1000))
        processor._queue.put_nowait((bot, window_data, 2000))
        processor._worker_stop.set()
        processor._worker_loop()

    assert len(captured_events) == 2
    assert captured_events[0] is not captured_events[1]
    assert captured_events[0] is not None
    assert captured_events[1] is not None


# 3. stop seta token ativo
@pytest.mark.asyncio
async def test_stop_sets_active_cancel_event():
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    started_ev = threading.Event()
    active_token = None

    def _slow_task(b, wd, close_ms, cancel_event=None):
        nonlocal active_token
        active_token = cancel_event
        started_ev.set()
        if cancel_event is not None:
            cancel_event.wait(timeout=5.0)

    with patch.object(wp, "process_window_snapshot", _slow_task):
        await processor.start()
        processor.submit_window(bot, window_data, 1000)
        assert await asyncio.to_thread(started_ev.wait, 2.0) is True

        # Chama stop enquanto a tarefa está rodando
        stop_task = asyncio.create_task(processor.stop())
        await stop_task

        # O token ativo deve ter sido sinalizado
        assert active_token is not None
        assert active_token.is_set() is True


# 4. cancelamento antes do commit impede finalize/commit escolhido
def test_cancellation_before_commit_skips_finalize():
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}, {"p": 101.0, "q": 2.0, "T": 1001}, {"p": 102.0, "q": 3.0, "T": 1002}]
    cancel_event = threading.Event()
    cancel_event.set()  # Já cancelado

    finalize_called = False
    from core.state_manager import StateManager
    orig_finalize = StateManager.finalize_window

    def _spy_finalize(self):
        nonlocal finalize_called
        finalize_called = True
        return orig_finalize(self)

    with patch.object(StateManager, "finalize_window", _spy_finalize):
        wp.process_window_snapshot(bot, window_data, 1000, cancel_event=cancel_event)

    assert finalize_called is False


# 5. cancelamento antes de signal publish impede EventBus.publish("signal")
def test_cancellation_before_signal_publish_prevents_eventbus():
    bot = DummyBot()
    cancel_event = threading.Event()
    cancel_event.set()

    signal = {"tipo_evento": "ANALYSIS_TRIGGER", "is_signal": True, "epoch_ms": 1000}

    publish_called = False

    def _spy_publish(topic, data):
        nonlocal publish_called
        if topic == "signal":
            publish_called = True

    bot.event_bus.publish = _spy_publish

    # Invoca _enrich_signal com cancel_event setado
    bot._enrich_signal(
        signal,
        derivatives_context={},
        flow_metrics={},
        total_buy_volume=1.0,
        total_sell_volume=1.0,
        macro_context={},
        close_ms=1000,
        ml_payload={},
        enriched_snapshot={},
        contextual_snapshot={},
        ob_event={},
        valid_window_data=[],
        support_resistance={},
        defense_zones_data={},
        cancel_event=cancel_event,
    )

    assert publish_called is False


# 6. ShadowPaperRuntime não recebe novo sinal da janela cancelada
def test_shadow_runtime_does_not_receive_signal_from_cancelled_window():
    bot = DummyBot()
    shadow_runtime = MagicMock()
    bot.event_bus.subscribe("signal", shadow_runtime.on_signal)

    cancel_event = threading.Event()
    cancel_event.set()

    signal = {"tipo_evento": "BUY_PRESSURE", "is_signal": True, "epoch_ms": 1000}

    bot._enrich_signal(
        signal,
        derivatives_context={},
        flow_metrics={},
        total_buy_volume=1.0,
        total_sell_volume=1.0,
        macro_context={},
        close_ms=1000,
        ml_payload={},
        enriched_snapshot={},
        contextual_snapshot={},
        ob_event={},
        valid_window_data=[],
        support_resistance={},
        defense_zones_data={},
        cancel_event=cancel_event,
    )

    shadow_runtime.on_signal.assert_not_called()


# 7. histories não recebem atualização posterior conforme contrato
def test_histories_not_updated_when_cancelled():
    bot = DummyBot()
    cancel_event = threading.Event()
    cancel_event.set()

    histories_called = False

    def _spy_update_histories(*args, **kwargs):
        nonlocal histories_called
        histories_called = True

    bot._update_histories = _spy_update_histories

    pipeline_mock = MagicMock()
    pipeline_mock.get_final_features.return_value = {"feat": 1}

    sp.process_signals(
        bot=bot,
        signals=[{"tipo_evento": "ANALYSIS_TRIGGER", "is_signal": True}],
        pipeline=pipeline_mock,
        flow_metrics={},
        historical_profile={},
        macro_context={},
        ob_event={},
        enriched={},
        close_ms=1000,
        total_buy_volume=1.0,
        total_sell_volume=1.0,
        valid_window_data=[],
        cancel_event=cancel_event,
    )

    assert histories_called is False


# 8. persistence posterior ao boundary não ocorre conforme contrato
def test_persistence_not_executed_when_cancelled():
    bot = DummyBot()
    cancel_event = threading.Event()
    cancel_event.set()

    pipeline_mock = MagicMock()
    pipeline_mock.get_final_features.return_value = {"feat": 1}

    sp.process_signals(
        bot=bot,
        signals=[],
        pipeline=pipeline_mock,
        flow_metrics={},
        historical_profile={},
        macro_context={},
        ob_event={},
        enriched={},
        close_ms=1000,
        total_buy_volume=1.0,
        total_sell_volume=1.0,
        valid_window_data=[],
        cancel_event=cancel_event,
    )

    bot.feature_store.save_features.assert_not_called()


# 9. próxima janela não herda cancelamento
def test_next_window_does_not_inherit_cancellation():
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    events_received = []

    def _capture_task(b, wd, close_ms, cancel_event=None):
        events_received.append(cancel_event)

    orig_submit = concurrent.futures.ThreadPoolExecutor.submit
    submit_count = 0

    def _mock_submit(self, fn, *args, **kwargs):
        nonlocal submit_count
        submit_count += 1
        future = orig_submit(self, fn, *args, **kwargs)
        if submit_count == 1:
            future.result = MagicMock(side_effect=concurrent.futures.TimeoutError("Timeout window 1"))
        return future

    with patch.object(concurrent.futures.ThreadPoolExecutor, "submit", _mock_submit),          patch.object(wp, "process_window_snapshot", _capture_task):
        processor._queue.put_nowait((bot, window_data, 1000))  # Timeout
        processor._queue.put_nowait((bot, window_data, 2000))  # Sucesso
        processor._worker_stop.set()
        processor._worker_loop()

    assert len(events_received) == 2
    # Janela 1 sofreu timeout e seu token foi setado
    assert events_received[0].is_set() is True
    # Janela 2 é independente e seu token NÃO foi setado
    assert events_received[1].is_set() is False


# 10. janela N cancelada não impede N+1
def test_cancelled_window_n_does_not_prevent_n_plus_1():
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    win1_started = threading.Event()
    win1_release = threading.Event()
    win2_success = threading.Event()

    def _task(b, wd, close_ms, cancel_event=None):
        if close_ms == 1000:
            win1_started.set()
            win1_release.wait(timeout=5.0)
        elif close_ms == 2000:
            win2_success.set()

    orig_submit = concurrent.futures.ThreadPoolExecutor.submit
    count = 0

    def _mock_submit(self, fn, *args, **kwargs):
        nonlocal count
        count += 1
        future = orig_submit(self, fn, *args, **kwargs)
        if count == 1:
            def _fast_timeout(timeout=None):
                win1_started.wait(timeout=1.0)
                raise concurrent.futures.TimeoutError("Timeout W1")
            future.result = _fast_timeout
        return future

    with patch.object(concurrent.futures.ThreadPoolExecutor, "submit", _mock_submit),          patch.object(wp, "process_window_snapshot", _task):
        processor._queue.put_nowait((bot, window_data, 1000))
        processor._queue.put_nowait((bot, window_data, 2000))
        processor._worker_stop.set()
        processor._worker_loop()

    win1_release.set()

    assert win2_success.is_set() is True
    # Window count incrementou para a janela 2 bem sucedida
    assert processor.window_count == 1


# 11. side effect anterior ao cancelamento NÃO é revertido
def test_side_effect_before_cancellation_not_reverted():
    bot = DummyBot()
    window_data = [
        {"p": 100.0, "q": 1.0, "T": 1000},
        {"p": 101.0, "q": 2.0, "T": 1001},
        {"p": 102.0, "q": 3.0, "T": 1002},
    ]
    cancel_event = threading.Event()

    # Interromper antes do pipeline (guard 1): window_count já incrementou em L662
    def _cancel_before_pipeline(*args, **kwargs):
        cancel_event.set()
        return None

    # Forçar cancelamento logo após incrementar window_count
    with patch("market_orchestrator.windows.window_processor.fetch_orderbook_with_retry", side_effect=_cancel_before_pipeline):
        wp.process_window_snapshot(bot, window_data, 1000, cancel_event=cancel_event)

    # O incremento de window_count anterior ao cancelamento permanece válido
    assert bot.window_count == 1
    # Mas a publicação de eventos e finalização foram barradas
    bot.event_saver.save_event.assert_not_called()


# 12. worker preso antes da próxima guarda pode continuar vivo, mas o loop continua após timeout
def test_stuck_worker_continues_living_while_loop_advances():
    processor = _create_processor()
    bot = DummyBot()
    window_data = [{"p": 100.0, "q": 1.0, "T": 1000}]

    stuck_worker_started = threading.Event()
    worker_released = threading.Event()

    def _stuck_task(b, wd, close_ms, cancel_event=None):
        stuck_worker_started.set()
        worker_released.wait(timeout=5.0)

    orig_submit = concurrent.futures.ThreadPoolExecutor.submit

    def _timeout_submit(self, fn, *args, **kwargs):
        future = orig_submit(self, fn, *args, **kwargs)
        future.result = MagicMock(side_effect=concurrent.futures.TimeoutError("Timeout"))
        return future

    t0 = time.perf_counter()
    with patch.object(concurrent.futures.ThreadPoolExecutor, "submit", _timeout_submit),          patch.object(wp, "process_window_snapshot", _stuck_task):
        processor._queue.put_nowait((bot, window_data, 1000))
        processor._worker_stop.set()
        processor._worker_loop()

    elapsed = time.perf_counter() - t0
    worker_released.set()

    # O loop avançou rapidamente (< 1.0s) em vez de esperar os 5.0s do worker preso
    assert elapsed < 1.0
    assert processor._queue.unfinished_tasks == 0


# 13. comportamento sem cancel_event permanece retrocompatível
def test_backward_compatibility_without_cancel_event():
    bot = DummyBot()
    window_data = [
        {"p": 100.0, "q": 1.0, "T": 1000},
        {"p": 101.0, "q": 2.0, "T": 1001},
        {"p": 102.0, "q": 3.0, "T": 1002},
    ]

    # Chamada sem passar cancel_event (API original)
    with patch("market_orchestrator.windows.window_processor.fetch_orderbook_with_retry", return_value={"is_valid": True}),          patch.object(bot, "_process_signals") as mock_ps:
        wp.process_window_snapshot(bot, window_data, 1000)
        assert mock_ps.called is True


# ==============================================================================
# NOVOS TESTES DA FASE 5.8B.1
# ==============================================================================

def test_internal_type_error_in_enrich_signal_is_called_exactly_once():
    """14. TypeError interno no corpo de _enrich_signal NÃO sofre retry e é chamado exatamente 1 vez."""
    from market_orchestrator.signals.signal_processor import process_signals

    bot = MagicMock()
    bot.feature_store = MagicMock()
    bot.symbol = "BTCUSDT"

    call_count = 0
    def _faulty_enrich(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise TypeError("Simulated internal TypeError inside _enrich_signal body")

    bot._enrich_signal = _faulty_enrich

    pipeline = MagicMock()
    pipeline.get_final_features.return_value = {
        "ml_features": {},
        "enriched": {"close": 50000.0},
    }

    signals = [{"tipo_evento": "BUY", "action": "BUY", "is_signal": True}]

    with pytest.raises(TypeError, match="Simulated internal TypeError inside _enrich_signal body"):
        process_signals(
            bot=bot,
            signals=signals,
            pipeline=pipeline,
            flow_metrics={},
            historical_profile={},
            macro_context={},
            ob_event={},
            enriched={"close": 50000.0},
            close_ms=1000,
            total_buy_volume=10.0,
            total_sell_volume=5.0,
            valid_window_data=[{"T": 1000}],
            cancel_event=threading.Event(),
        )

    # Prova que foi chamado exatamente UMA vez (nenhum fallback retry)
    assert call_count == 1


def test_internal_type_error_in_process_signals_is_called_exactly_once():
    """15. TypeError interno no corpo de _process_signals NÃO sofre retry na chamada de window_processor."""
    bot = DummyBot()
    call_count = 0

    def _faulty_process_signals(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise TypeError("Simulated internal TypeError in _process_signals")

    bot._process_signals = _faulty_process_signals

    trades = [
        {"p": 50000.0, "q": 1.0, "T": 1000, "is_buyer_maker": False},
        {"p": 50000.0, "q": 2.0, "T": 1001, "is_buyer_maker": True},
        {"p": 50001.0, "q": 1.5, "T": 1002, "is_buyer_maker": False},
    ]

    with patch("market_orchestrator.windows.window_processor.fetch_orderbook_with_retry", return_value={"is_valid": True}):
        wp.process_window_snapshot(bot, trades, 1000, cancel_event=threading.Event())

    assert call_count == 1


def test_cancellation_before_publish_does_not_pollute_sent_triggers(monkeypatch):
    """16. Cancelamento antes do publish NÃO adiciona chave em _sent_triggers, permitindo envio futuro."""
    from market_orchestrator.market_orchestrator import EnhancedMarketBot
    import market_orchestrator.market_orchestrator as mo

    # Mock validator para aceitar o sinal sem rejeição por schema
    monkeypatch.setattr(mo.validator, "validate_and_clean", lambda sig: sig)
    monkeypatch.setattr(mo, "adicionar_memoria_evento", lambda *a, **k: None)

    bot = EnhancedMarketBot.__new__(EnhancedMarketBot)
    bot._sent_triggers = set()
    bot.levels = MagicMock()
    bot.event_bus = MagicMock()
    bot.event_saver = MagicMock()
    bot.time_manager = MagicMock()
    bot.ny_tz = None
    bot.window_count = 1
    bot.symbol = "BTCUSDT"
    bot.orderbook_fetch_failures = 0
    bot.institutional_analytics = None
    bot.volume_history = []
    bot.volatility_history = []
    bot._ai_pool_lock = threading.Lock()

    import datetime
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    bot.time_manager.from_timestamp_ms.return_value = now_dt

    signal = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "features_window_id": "WINDOW_100",
        "epoch_ms": 1700000000000,
        "timestamp_utc": "2026-10-04T00:00:00Z",
    }
    trigger_key = ("ANALYSIS_TRIGGER", "WINDOW_100")

    # 1. Execução cancelada
    cancelled_event = threading.Event()
    cancelled_event.set()

    bot._enrich_signal(
        signal=signal.copy(),
        derivatives_context={},
        flow_metrics={},
        total_buy_volume=10.0,
        total_sell_volume=5.0,
        macro_context={},
        close_ms=1700000000000,
        ml_payload={},
        enriched_snapshot={},
        contextual_snapshot={},
        ob_event={},
        valid_window_data=[],
        support_resistance={},
        defense_zones_data={},
        cancel_event=cancelled_event,
    )

    # Chave NÃO deve estar em _sent_triggers e publish NÃO deve ter sido chamado
    assert trigger_key not in bot._sent_triggers
    assert bot.event_bus.publish.call_count == 0

    # 2. Execução não-cancelada posterior com a mesma chave
    clean_event = threading.Event()

    bot._enrich_signal(
        signal=signal.copy(),
        derivatives_context={},
        flow_metrics={},
        total_buy_volume=10.0,
        total_sell_volume=5.0,
        macro_context={},
        close_ms=1700000000000,
        ml_payload={},
        enriched_snapshot={},
        contextual_snapshot={},
        ob_event={},
        valid_window_data=[],
        support_resistance={},
        defense_zones_data={},
        cancel_event=clean_event,
    )

    # Agora sim a chave foi registrada e o sinal publicado
    assert trigger_key in bot._sent_triggers
    assert bot.event_bus.publish.call_count == 1


def test_cancelled_window_does_not_trigger_audit_hook_or_unbound_error():
    """17. Janela cancelada salta o bloco on_window sem erro de referência não inicializada (_fwin)."""
    bot = DummyBot()
    trades = [
        {"p": 50000.0, "q": 1.0, "T": 1000, "is_buyer_maker": False},
        {"p": 50000.0, "q": 2.0, "T": 1001, "is_buyer_maker": True},
        {"p": 50001.0, "q": 1.5, "T": 1002, "is_buyer_maker": False},
    ]

    cancel_event = threading.Event()
    cancel_event.set()

    with patch("audit_live.hooks.on_window", create=True) as mock_fwin:
        wp.process_window_snapshot(bot, trades, 1000, cancel_event=cancel_event)
        # O mock não deve ser chamado
        mock_fwin.assert_not_called()


def test_direct_submit_passes_cancel_event_without_signature_introspection():
    """18. WindowProcessor submete diretamente cancel_event sem usar inspect.signature."""
    import inspect
    processor = wp.WindowProcessor(
        symbol="BTCUSDT",
        windows_minutes=[1],
        event_bus=MagicMock(),
        time_manager=MagicMock(),
    )
    bot = DummyBot()
    trades = [{"p": 100.0, "q": 1.0, "T": 1000}]

    submitted_cancel_event = None
    done = threading.Event()

    def _mock_process(b, wd, close_ms, cancel_event=None):
        nonlocal submitted_cancel_event
        submitted_cancel_event = cancel_event
        done.set()

    with patch.object(wp, "process_window_snapshot", _mock_process),          patch.object(inspect, "signature", side_effect=AssertionError("inspect.signature must not be called!")):
        processor._queue.put_nowait((bot, trades, 1000))
        processor._worker_stop.set()
        processor._worker_loop()

    assert done.is_set()
    assert isinstance(submitted_cancel_event, threading.Event)
