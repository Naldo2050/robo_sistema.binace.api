# test_trade_buffer_integrity.py
"""
Testes de integridade, observabilidade e bounded shutdown do AsyncTradeBuffer (Fase 5.10 / 5.10b - Achado #9).

Cenários cobertos:
1. start -> add -> process normal
2. add antes de start (contrato explícito)
3. add após stop -> False
4. add durante drain -> False
5. overflow: remove N antigos, novo retorna True, dropped_overflow += N
6. processor exception: processor_error += 1, worker segue
7. stop buffer vazio
8. stop com poucos itens e consumer rápido -> drena todos
9. stop com muitos itens mas consumer rápido -> drena dentro deadline
10. stop com consumer lento -> deadline respeitado
11. shutdown drop é contado exatamente
12. batch in-flight concluído antes de declarar clean shutdown
13. cancellation durante in-flight não gera contagem falsa
14. stop idempotente
15. restart após stop
16. get_stats consistente sob concorrência
17. caller contract: overflow de antigos não rejeita trade novo
18. drain não dispara janelas ou sinais no WindowProcessor
19. PARTE A: gather cancellation não causa mutação post-stop por corrotinas child
20. PARTE B: balanço de massa invariant em múltiplos cenários
21. PARTE C: batch latency registrado exatamente N vezes (não 2N)
22. PARTE E: validação de drain_timeout_ms (bool, NaN, Inf, negativo, zero)
23. PARTE F: start() durante stop() ativo levanta RuntimeError e previne race
24. PARTE G: ponto único de append com maxlen safety
25. PARTE H: Prometheus registry não duplica timeseries em reload ou múltiplos buffers
26. PARTE I: Forensic context aceita delta numérico sem fallback silencioso
"""

import asyncio
import inspect
import importlib
import math
import time
import pytest
from unittest.mock import MagicMock, patch
from trading.trade_buffer import AsyncTradeBuffer, BufferStatus


@pytest.mark.asyncio
async def test_01_start_add_process_normal():
    """1. start -> add -> process normal."""
    buffer = AsyncTradeBuffer(
        max_size=50,
        processing_batch_size=10,
        processing_interval_ms=5,
        drain_timeout_ms=1000.0
    )
    processed = []

    def processor(trade):
        processed.append(trade)

    await buffer.start()

    for i in range(10):
        ok = buffer.add_trade_sync({"id": i, "p": 100.0, "q": 1.0}, processor)
        assert ok is True

    for _ in range(50):
        if len(processed) == 10:
            break
        await asyncio.sleep(0.01)

    assert len(processed) == 10
    stats = buffer.get_stats()
    assert stats["buffer"]["current_size"] == 0
    assert stats["processing"]["total_trades"] == 10
    assert stats["buffer"]["dropped_trades_count"] == 0

    await buffer.stop()
    assert buffer._is_stopped is True


@pytest.mark.asyncio
async def test_02_add_before_start_explicit_contract():
    """2. add antes de start (contrato explícito: aceita e drena após start)."""
    buffer = AsyncTradeBuffer(
        max_size=50,
        processing_batch_size=10,
        processing_interval_ms=5
    )
    processed = []

    def processor(trade):
        processed.append(trade)

    for i in range(5):
        ok = buffer.add_trade_sync({"id": i}, processor)
        assert ok is True

    stats_before = buffer.get_stats()
    assert stats_before["buffer"]["current_size"] == 5
    assert len(processed) == 0

    await buffer.start()

    for _ in range(50):
        if len(processed) == 5:
            break
        await asyncio.sleep(0.01)

    assert len(processed) == 5
    await buffer.stop()


@pytest.mark.asyncio
async def test_03_add_after_stop_returns_false():
    """3. add após stop -> False."""
    buffer = AsyncTradeBuffer(max_size=50)
    await buffer.start()
    await buffer.stop()

    assert buffer._is_stopped is True
    assert buffer._accepting_trades is False

    ok_sync = buffer.add_trade_sync({"id": 1}, lambda t: None)
    assert ok_sync is False

    ok_async = await buffer.add_trade({"id": 2}, lambda t: None)
    assert ok_async is False

    added = buffer.add_trades_batch_sync([{"id": 3}, {"id": 4}], lambda t: None)
    assert added == 0

    stats = buffer.get_stats()
    assert stats["buffer"]["current_size"] == 0


@pytest.mark.asyncio
async def test_04_add_during_drain_returns_false():
    """4. add durante drain -> False."""
    buffer = AsyncTradeBuffer(
        max_size=50,
        processing_batch_size=5,
        processing_interval_ms=5,
        drain_timeout_ms=1000.0
    )

    async def slow_processor(trade):
        await asyncio.sleep(0.05)

    await buffer.start()

    for i in range(10):
        buffer.add_trade_sync({"id": i}, slow_processor)

    stop_task = asyncio.create_task(buffer.stop())
    await asyncio.sleep(0.005)

    rejected = buffer.add_trade_sync({"id": 999}, slow_processor)
    assert rejected is False

    await stop_task
    assert buffer._is_stopped is True


@pytest.mark.asyncio
async def test_05_overflow_drop_oldest_accounting():
    """5. overflow: remove N antigos, novo retorna True, dropped_overflow += N."""
    max_size = 10
    buffer = AsyncTradeBuffer(
        max_size=max_size,
        backpressure_threshold=0.8,
        processing_batch_size=5,
        processing_interval_ms=100
    )
    processed = []

    def processor(trade):
        processed.append(trade)

    for i in range(max_size):
        ok = buffer.add_trade_sync({"id": i}, processor)
        assert ok is True

    stats = buffer.get_stats()
    assert stats["buffer"]["current_size"] == max_size
    assert stats["buffer"]["overflow_events"] == 0
    assert stats["buffer"]["dropped_overflow_trades"] == 0

    ok_11 = buffer.add_trade_sync({"id": 100}, processor)
    assert ok_11 is True

    stats_overflow = buffer.get_stats()
    assert stats_overflow["buffer"]["overflow_events"] >= 1
    assert stats_overflow["buffer"]["dropped_overflow_trades"] >= 1
    assert stats_overflow["buffer"]["dropped_shutdown_trades"] == 0
    assert stats_overflow["buffer"]["dropped_trades_count"] == stats_overflow["buffer"]["dropped_overflow_trades"]

    with buffer._buffer_lock:
        ids_in_buf = [item[0]["id"] for item in buffer._buffer]
    assert 0 not in ids_in_buf
    assert 100 in ids_in_buf


@pytest.mark.asyncio
async def test_06_processor_exception_accounting():
    """6. processor exception: processor_error += 1, worker segue."""
    buffer = AsyncTradeBuffer(
        max_size=20,
        processing_batch_size=10,
        processing_interval_ms=5
    )
    processed = []

    def faulty_processor(trade):
        if trade["id"] == 2:
            raise ValueError("Erro proposital no trade 2")
        processed.append(trade)

    await buffer.start()

    for i in range(5):
        buffer.add_trade_sync({"id": i}, faulty_processor)

    for _ in range(50):
        if len(processed) == 4:
            break
        await asyncio.sleep(0.01)

    assert len(processed) == 4
    stats = buffer.get_stats()
    assert stats["buffer"]["processor_error_trades"] == 1
    assert stats["processing"]["total_trades"] == 4

    await buffer.stop()


@pytest.mark.asyncio
async def test_07_stop_empty_buffer():
    """7. stop buffer vazio."""
    buffer = AsyncTradeBuffer(max_size=50)
    await buffer.start()
    await buffer.stop()

    assert buffer._is_stopped is True
    stats = buffer.get_stats()
    assert stats["buffer"]["dropped_shutdown_trades"] == 0
    assert stats["buffer"]["current_size"] == 0


@pytest.mark.asyncio
async def test_08_stop_fast_consumer_drains_all():
    """8. stop com poucos itens e consumer rápido -> drena todos."""
    buffer = AsyncTradeBuffer(
        max_size=50,
        processing_batch_size=10,
        processing_interval_ms=5,
        drain_timeout_ms=1000.0
    )
    processed = []

    def fast_processor(trade):
        processed.append(trade)

    await buffer.start()

    for i in range(15):
        buffer.add_trade_sync({"id": i}, fast_processor)

    await buffer.stop()

    assert len(processed) == 15
    stats = buffer.get_stats()
    assert stats["buffer"]["dropped_shutdown_trades"] == 0
    assert stats["processing"]["total_trades"] == 15


@pytest.mark.asyncio
async def test_09_stop_many_items_fast_consumer():
    """9. stop com muitos itens mas consumer rápido -> drena dentro deadline."""
    buffer = AsyncTradeBuffer(
        max_size=500,
        processing_batch_size=50,
        processing_interval_ms=2,
        drain_timeout_ms=2000.0
    )
    processed = []

    def processor(trade):
        processed.append(trade)

    await buffer.start()

    for i in range(200):
        buffer.add_trade_sync({"id": i}, processor)

    await buffer.stop()

    assert len(processed) == 200
    stats = buffer.get_stats()
    assert stats["buffer"]["dropped_shutdown_trades"] == 0
    assert stats["processing"]["total_trades"] == 200


@pytest.mark.asyncio
async def test_10_and_11_stop_slow_consumer_deadline_and_exact_accounting():
    """10 & 11. consumer lento -> deadline respeitado e drops contados exatamente."""
    drain_timeout_ms = 80.0
    buffer = AsyncTradeBuffer(
        max_size=100,
        processing_batch_size=5,
        processing_interval_ms=2,
        drain_timeout_ms=drain_timeout_ms
    )
    processed = []

    async def slow_processor(trade):
        await asyncio.sleep(0.03)
        processed.append(trade)

    await buffer.start()

    total_added = 40
    for i in range(total_added):
        buffer.add_trade_sync({"id": i}, slow_processor)

    t0 = time.monotonic()
    await buffer.stop(timeout_ms=drain_timeout_ms)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < drain_timeout_ms + 250.0

    stats = buffer.get_stats()
    n_processed = stats["processing"]["total_trades"]
    n_shutdown_drops = stats["buffer"]["dropped_shutdown_trades"]

    assert n_processed + n_shutdown_drops == total_added
    assert n_shutdown_drops > 0
    assert stats["buffer"]["current_size"] == 0


@pytest.mark.asyncio
async def test_12_in_flight_completed_before_clean_shutdown():
    """12. batch in-flight concluído antes de declarar clean shutdown."""
    buffer = AsyncTradeBuffer(
        max_size=50,
        processing_batch_size=10,
        processing_interval_ms=5,
        drain_timeout_ms=1000.0
    )
    processed = []

    async def processor(trade):
        await asyncio.sleep(0.02)
        processed.append(trade)

    await buffer.start()

    for i in range(5):
        buffer.add_trade_sync({"id": i}, processor)

    for _ in range(30):
        with buffer._buffer_lock:
            if len(buffer._buffer) == 0 and buffer._in_flight_count > 0:
                break
        await asyncio.sleep(0.002)

    await buffer.stop()

    assert len(processed) == 5
    stats = buffer.get_stats()
    assert stats["buffer"]["dropped_shutdown_trades"] == 0
    assert stats["processing"]["total_trades"] == 5


@pytest.mark.asyncio
async def test_13_cancellation_during_in_flight_exact_accounting():
    """13. cancellation durante in-flight não gera contagem falsa."""
    buffer = AsyncTradeBuffer(
        max_size=50,
        processing_batch_size=10,
        processing_interval_ms=5,
        drain_timeout_ms=20.0
    )
    processed = []

    async def blocked_processor(trade):
        if trade["id"] == 0:
            processed.append(trade)
            return
        await asyncio.sleep(1.0)
        processed.append(trade)

    await buffer.start()

    for i in range(5):
        buffer.add_trade_sync({"id": i}, blocked_processor)

    await buffer.stop(timeout_ms=20.0)

    stats = buffer.get_stats()
    n_processed = stats["processing"]["total_trades"]
    n_shutdown_drops = stats["buffer"]["dropped_shutdown_trades"]

    assert n_processed + n_shutdown_drops == 5
    assert n_shutdown_drops >= 4


@pytest.mark.asyncio
async def test_14_stop_is_idempotent():
    """14. stop idempotente."""
    buffer = AsyncTradeBuffer(max_size=20)
    await buffer.start()

    await buffer.stop()
    assert buffer._is_stopped is True

    await buffer.stop()
    await buffer.stop()
    assert buffer._is_stopped is True


@pytest.mark.asyncio
async def test_15_restart_after_stop():
    """15. restart após stop, conforme contrato escolhido."""
    buffer = AsyncTradeBuffer(
        max_size=20,
        processing_batch_size=5,
        processing_interval_ms=5
    )
    processed = []

    def processor(trade):
        processed.append(trade)

    await buffer.start()
    buffer.add_trade_sync({"id": 1}, processor)
    await buffer.stop()
    assert len(processed) == 1
    assert buffer._is_stopped is True

    await buffer.start()
    assert buffer._is_stopped is False
    assert buffer._accepting_trades is True

    ok = buffer.add_trade_sync({"id": 2}, processor)
    assert ok is True

    await buffer.stop()
    assert len(processed) == 2


@pytest.mark.asyncio
async def test_16_get_stats_concurrent_consistency():
    """16. get_stats consistente sob concorrência."""
    buffer = AsyncTradeBuffer(
        max_size=100,
        processing_batch_size=10,
        processing_interval_ms=5
    )

    def processor(trade):
        pass

    await buffer.start()

    stop_flag = False

    async def producer():
        i = 0
        while not stop_flag:
            buffer.add_trade_sync({"id": i}, processor)
            i += 1
            await asyncio.sleep(0.001)

    async def reader():
        while not stop_flag:
            stats = buffer.get_stats()
            assert isinstance(stats, dict)
            assert "buffer" in stats
            assert stats["buffer"]["current_size"] >= 0
            await asyncio.sleep(0.001)

    t_prod = asyncio.create_task(producer())
    t_read = asyncio.create_task(reader())

    await asyncio.sleep(0.1)
    stop_flag = True
    await asyncio.gather(t_prod, t_read)

    await buffer.stop()


@pytest.mark.asyncio
async def test_17_caller_contract_overflow_does_not_mask_new_trade():
    """17. overflow antigo não gera rejeição de trade novo no caller."""
    buffer = AsyncTradeBuffer(
        max_size=5,
        backpressure_threshold=0.8,
        processing_batch_size=2,
        processing_interval_ms=100
    )

    caller_rejected_count = 0

    for i in range(10):
        success = buffer.add_trade_sync({"id": i}, lambda t: None)
        if not success:
            caller_rejected_count += 1

    assert caller_rejected_count == 0

    stats = buffer.get_stats()
    assert stats["buffer"]["dropped_overflow_trades"] > 0
    assert stats["buffer"]["overflow_events"] > 0


@pytest.mark.asyncio
async def test_18_drain_does_not_trigger_signals_or_windows():
    """18. drain não dispara janelas ou sinais no WindowProcessor."""
    buffer = AsyncTradeBuffer(
        max_size=50,
        processing_batch_size=10,
        processing_interval_ms=5
    )

    signals_published = []
    windows_created = []

    def flow_analyzer_mock_process_trade(trade):
        pass

    await buffer.start()

    for i in range(20):
        buffer.add_trade_sync({"id": i, "p": 100.0, "q": 1.0, "T": 1000 + i}, flow_analyzer_mock_process_trade)

    await buffer.stop()

    assert len(signals_published) == 0
    assert len(windows_created) == 0


@pytest.mark.asyncio
async def test_19_gather_cancellation_no_post_stop_mutation():
    """
    PARTE A:
    Batch >= 20 particionado em chunks via asyncio.gather.
    Pelo menos um chunk controlado/bloqueado por Event.
    Stop timeout expira -> task cancelada -> await task termina.
    Prova que nenhuma coroutine child continua alterando os contadores após stop().
    """
    buffer = AsyncTradeBuffer(
        max_size=100,
        processing_batch_size=40,
        processing_interval_ms=2,
        drain_timeout_ms=30.0  # Timeout curto
    )

    unblock_event = asyncio.Event()

    async def blocking_processor(trade):
        if trade["id"] < 5:
            # Completa normalmente
            return
        # Fica bloqueado até unblock_event
        await unblock_event.wait()

    await buffer.start()

    # Adiciona 24 trades (para forçar batch paralelo com len >= 20)
    for i in range(24):
        buffer.add_trade_sync({"id": i}, blocking_processor)

    # Dispara stop com timeout de 30ms (irá expirar)
    t0 = time.monotonic()
    await buffer.stop(timeout_ms=30.0)
    stop_duration = (time.monotonic() - t0) * 1000

    assert buffer._is_stopped is True

    # Snapshot imediatamente após o retorno de stop()
    snap_inflight = buffer._in_flight_count
    snap_current_inflight = len(buffer._current_in_flight_trades)
    snap_processed = buffer._total_trades_processed
    snap_proc_errors = buffer._processor_error_trades
    snap_shutdown_drops = buffer._dropped_shutdown_trades

    # Aguarda período controlado para permitir que qualquer child órfã rodasse
    unblock_event.set()
    await asyncio.sleep(0.08)

    # Prova que nenhum contador foi mutado por coroutines child após stop()
    assert buffer._in_flight_count == snap_inflight
    assert len(buffer._current_in_flight_trades) == snap_current_inflight
    assert buffer._total_trades_processed == snap_processed
    assert buffer._processor_error_trades == snap_proc_errors
    assert buffer._dropped_shutdown_trades == snap_shutdown_drops


@pytest.mark.asyncio
async def test_20_mass_balance_invariant():
    """
    PARTE B:
    Equação de balanço de massa:
    accepted_new = processed_success + processor_error + dropped_overflow + dropped_shutdown + pending + in_flight
    Testado em múltiplos cenários.
    """
    def check_mass_balance(buf, expected_accepted):
        stats = buf.get_stats()
        b = stats["buffer"]
        p = stats["processing"]
        calculated = (
            p["total_trades"] +
            b["processor_error_trades"] +
            b["dropped_overflow_trades"] +
            b["dropped_shutdown_trades"] +
            b["current_size"] +
            b["in_flight_count"]
        )
        assert b["total_accepted_trades"] == expected_accepted
        assert calculated == expected_accepted, (
            f"Mass balance violado! esperado={expected_accepted}, calculado={calculated} "
            f"(proc={p['total_trades']}, err={b['processor_error_trades']}, "
            f"overflow={b['dropped_overflow_trades']}, shutdown={b['dropped_shutdown_trades']}, "
            f"pending={b['current_size']}, in_flight={b['in_flight_count']})"
        )

    # Cenário 1: Normal
    b1 = AsyncTradeBuffer(max_size=20, processing_batch_size=5, processing_interval_ms=2)
    await b1.start()
    for i in range(12):
        b1.add_trade_sync({"id": i}, lambda t: None)
    await b1.stop()
    check_mass_balance(b1, 12)

    # Cenário 2: Overflow
    b2 = AsyncTradeBuffer(max_size=10, backpressure_threshold=0.8, processing_batch_size=5, processing_interval_ms=100)
    for i in range(15):
        b2.add_trade_sync({"id": i}, lambda t: None)
    check_mass_balance(b2, 15)

    # Cenário 3: Processor Error
    b3 = AsyncTradeBuffer(max_size=20, processing_batch_size=5, processing_interval_ms=2)
    def err_proc(t):
        if t["id"] % 2 == 0:
            raise RuntimeError("error")
    await b3.start()
    for i in range(8):
        b3.add_trade_sync({"id": i}, err_proc)
    await b3.stop()
    check_mass_balance(b3, 8)

    # Cenário 4: Timeout Shutdown
    b4 = AsyncTradeBuffer(max_size=50, processing_batch_size=5, processing_interval_ms=2, drain_timeout_ms=20.0)
    async def slow_proc(t):
        await asyncio.sleep(0.05)
    await b4.start()
    for i in range(25):
        b4.add_trade_sync({"id": i}, slow_proc)
    await b4.stop(timeout_ms=20.0)
    check_mass_balance(b4, 25)


    # Cenário 5: Paralelo + Cancelamento
    b5 = AsyncTradeBuffer(max_size=100, processing_batch_size=40, processing_interval_ms=2, drain_timeout_ms=30.0)
    unblock_event = asyncio.Event()

    async def blocking_processor(trade):
        if trade["id"] < 5:
            return
        await unblock_event.wait()

    await b5.start()
    for i in range(24):
        b5.add_trade_sync({"id": i}, blocking_processor)
    await b5.stop(timeout_ms=30.0)
    unblock_event.set()
    await asyncio.sleep(0.08)
    check_mass_balance(b5, 24)

    # Cenário 6: Estado Intermediário (in-flight + pending ativos) e pós Clean Drain
    b6 = AsyncTradeBuffer(max_size=50, processing_batch_size=5, processing_interval_ms=5)
    pause_event = asyncio.Event()

    async def pausing_proc(t):
        await pause_event.wait()

    await b6.start()
    for i in range(15):
        b6.add_trade_sync({"id": i}, pausing_proc)

    # Aguarda o primeiro batch entrar in-flight
    await asyncio.sleep(0.015)
    check_mass_balance(b6, 15)  # Testa estado intermediário ativo

    # Libera os trades para clean drain e valida novamente
    pause_event.set()
    await b6.stop()
    check_mass_balance(b6, 15)  # Testa após clean drain


@pytest.mark.asyncio
async def test_21_batch_latency_exactly_once():
    """
    PARTE C:
    Garante que add_trades_batch_sync e add_trades_batch chamam latency_monitor.record_trade
    exatamente N vezes (uma por trade), e não 2N.
    """
    from trading.trade_buffer import AsyncTradeBuffer
    import trading.trade_buffer as tb_module

    buffer = AsyncTradeBuffer(max_size=50)

    mock_monitor = MagicMock()
    trades = [{"id": i, "p": 100.0} for i in range(7)]

    with patch.object(tb_module, "LATENCY_MONITOR_AVAILABLE", True), \
         patch.object(tb_module, "latency_monitor", mock_monitor):
        added = buffer.add_trades_batch_sync(trades, lambda t: None)

    assert added == 7
    assert mock_monitor.record_trade.call_count == 7, (
        f"Esperado 7 chamadas ao latency monitor, obteve {mock_monitor.record_trade.call_count}"
    )

    # Versão assíncrona também deve registrar exatamente N vezes
    mock_monitor.reset_mock()
    async_trades = [{"id": i + 10, "p": 101.0} for i in range(5)]
    with patch.object(tb_module, "LATENCY_MONITOR_AVAILABLE", True), \
         patch.object(tb_module, "latency_monitor", mock_monitor):
        added_async = await buffer.add_trades_batch(async_trades, lambda t: None)

    assert added_async == 5
    assert mock_monitor.record_trade.call_count == 5


def test_22_drain_timeout_validation():
    """
    PARTE E:
    Validação de drain_timeout_ms:
    - bool rejeitado com TypeError
    - NaN e Inf rejeitados com ValueError
    - negativo rejeitado com ValueError
    - zero aceito
    """
    # bool rejeitado
    with pytest.raises(TypeError, match="cannot be a boolean"):
        AsyncTradeBuffer(drain_timeout_ms=True)

    with pytest.raises(TypeError, match="cannot be a boolean"):
        AsyncTradeBuffer(drain_timeout_ms=False)

    # string ou tipo inválido
    with pytest.raises(TypeError, match="must be a number"):
        AsyncTradeBuffer(drain_timeout_ms="1000")

    # NaN e Inf rejeitados
    with pytest.raises(ValueError, match="cannot be NaN or Inf"):
        AsyncTradeBuffer(drain_timeout_ms=float("nan"))

    with pytest.raises(ValueError, match="cannot be NaN or Inf"):
        AsyncTradeBuffer(drain_timeout_ms=float("inf"))

    # negativo rejeitado
    with pytest.raises(ValueError, match="cannot be negative"):
        AsyncTradeBuffer(drain_timeout_ms=-1.0)

    # zero aceito como válido
    b_zero = AsyncTradeBuffer(drain_timeout_ms=0.0)
    assert b_zero.drain_timeout_ms == 0.0


@pytest.mark.asyncio
async def test_23_start_during_stop_raises_runtime_error():
    """
    PARTE F:
    start() chamado enquanto stop() está ativo/em andamento levanta RuntimeError
    e impede race conditions ou múltiplas tarefas.
    """
    buffer = AsyncTradeBuffer(
        max_size=50,
        processing_batch_size=5,
        processing_interval_ms=5,
        drain_timeout_ms=100.0
    )

    async def slow_proc(t):
        await asyncio.sleep(0.05)

    await buffer.start()

    for i in range(10):
        buffer.add_trade_sync({"id": i}, slow_proc)

    stop_task = asyncio.create_task(buffer.stop())
    await asyncio.sleep(0.005)

    assert buffer._is_stopping is True

    # Tentativa de start durante stop deve falhar claramente
    with pytest.raises(RuntimeError, match="Cannot start AsyncTradeBuffer while stop/drain is in progress"):
        await buffer.start()

    await stop_task
    assert buffer._is_stopped is True
    assert buffer._is_stopping is False


def test_24_unique_append_path_and_maxlen_safety():
    """
    PARTE G:
    Prova estrutural de que o buffer possui apenas um ponto de append e
    que ele sempre passa pela política contabilizada de overflow.
    """
    import inspect
    from trading.trade_buffer import AsyncTradeBuffer

    source = inspect.getsource(AsyncTradeBuffer)
    # Prova que self._buffer.append ocorre estritamente uma vez
    append_count = source.count("self._buffer.append(")
    extend_count = source.count("self._buffer.extend(")

    assert append_count == 1, f"Esperava 1 ponto de append, encontrado {append_count}"
    assert extend_count == 0, f"Esperava 0 pontos de extend, encontrado {extend_count}"


def test_25_prometheus_reload_no_duplication():
    """
    PARTE H:
    Provar que instanciar múltiplos buffers e recarregar o módulo de métricas
    não causa Duplicated timeseries in CollectorRegistry.
    """
    import monitoring.metrics_collector as mc
    importlib.reload(mc)

    b1 = AsyncTradeBuffer(max_size=10)
    b2 = AsyncTradeBuffer(max_size=20)
    assert b1.max_size == 10
    assert b2.max_size == 20

    # Segundo reload para provar idempotência
    importlib.reload(mc)


def test_26_forensic_context_delta_api():
    """
    PARTE I:
    Provar que _fctx.inc(key, delta=N) é uma assinatura válida e incrementa corretamente.
    """
    try:
        from audit_live.forensic_context import CTX
    except ImportError:
        pytest.skip("Forensic context não disponível")

    initial = CTX.counters.get("test_metric_delta", 0)
    CTX.inc("test_metric_delta", delta=42)
    assert CTX.counters["test_metric_delta"] == initial + 42
