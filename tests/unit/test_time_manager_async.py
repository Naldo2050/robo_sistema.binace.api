"""
Testes para TimeManager async patches (Etapa 3).

Cobre: now() non-blocking, sync_async em thread, periodic_sync,
       _validate_offset sem recursão, needs_sync property.
"""

import asyncio
import time
import logging
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from monitoring.time_manager import TimeManager


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def tm():
    """
    TimeManager com sync inicial mockado (evita rede nos testes).
    """
    with patch.object(TimeManager, "_initialize_sync"):
        instance = object.__new__(TimeManager)
        # Simula __init__ mínimo sem singleton guard
        instance._initialized = True
        instance.sync_interval_seconds = 1800
        instance.max_acceptable_offset_ms = 600
        instance.max_init_attempts = 3
        instance.num_sync_samples = 5
        instance.server_time_offset_ms = 0
        instance.last_sync_mono = time.monotonic()
        instance.sync_attempts = 0
        instance.sync_failures = 0
        instance.last_successful_sync_ms = int(time.time() * 1000)
        instance.last_offset_ms = 0
        instance.best_rtt_ms = 50
        instance.last_rtt_ms = 50
        instance.time_sync_status = "ok"
        instance.auto_corrections = 0
        instance._correction_attempts = 0
        instance._last_offset_history = []
        instance._lock = __import__("threading").Lock()
        instance._sync_needed = False
        instance._sync_lock = asyncio.Lock()

        # Constantes de classe que podem ser necessárias
        if not hasattr(instance, "CRITICAL_OFFSET_MS"):
            instance.CRITICAL_OFFSET_MS = 60000
        if not hasattr(instance, "WARNING_OFFSET_MS"):
            instance.WARNING_OFFSET_MS = 30000
        if not hasattr(instance, "MAX_CORRECTION_ATTEMPTS"):
            instance.MAX_CORRECTION_ATTEMPTS = 3

        yield instance


# ──────────────────────────────────────────────
# Testes: now() non-blocking
# ──────────────────────────────────────────────

class TestNowNonBlocking:

    def test_now_returns_ms(self, tm):
        """now() retorna timestamp em ms."""
        result = tm.now()
        assert isinstance(result, int)
        # Deve estar próximo de time.time() * 1000
        expected = int(time.time() * 1000)
        assert abs(result - expected) < 2000  # tolerância 2s

    def test_now_applies_offset(self, tm):
        """now() aplica o offset do servidor."""
        tm.server_time_offset_ms = 5000
        result = tm.now()
        expected = int(time.time() * 1000) + 5000
        assert abs(result - expected) < 100

    def test_now_does_not_call_sync(self, tm):
        """now() NÃO chama _sync_with_binance diretamente."""
        tm.last_sync_mono = 0  # forçar _should_sync() == True
        with patch.object(tm, "_sync_with_binance") as mock_sync:
            tm.now()
            mock_sync.assert_not_called()

    def test_now_sets_sync_needed_flag(self, tm):
        """now() seta _sync_needed quando _should_sync() retorna True."""
        tm.last_sync_mono = 0  # forçar _should_sync() == True
        assert tm._sync_needed is False
        tm.now()
        assert tm._sync_needed is True

    def test_now_does_not_set_flag_when_sync_not_needed(self, tm):
        """now() NÃO seta _sync_needed se sync não é necessário."""
        tm.last_sync_mono = time.monotonic()  # sync recente
        tm.now()
        assert tm._sync_needed is False

    def test_now_error_fallback(self, tm):
        """now() retorna timestamp local em caso de erro."""
        with patch.object(tm, "_should_sync", side_effect=Exception("boom")):
            result = tm.now()
            expected = int(time.time() * 1000)
            assert abs(result - expected) < 100


# ──────────────────────────────────────────────
# Testes: needs_sync property
# ──────────────────────────────────────────────

class TestNeedsSync:

    def test_needs_sync_false_by_default(self, tm):
        """needs_sync é False por padrão."""
        assert tm.needs_sync is False

    def test_needs_sync_true_after_now_triggers(self, tm):
        """needs_sync é True após now() detectar necessidade."""
        tm.last_sync_mono = 0
        tm.now()
        assert tm.needs_sync is True


# ──────────────────────────────────────────────
# Testes: sync_async
# ──────────────────────────────────────────────

class TestSyncAsync:

    @pytest.mark.asyncio
    async def test_sync_async_calls_safe_in_executor(self, tm):
        """sync_async executa _sync_with_binance_safe em thread."""
        with patch.object(tm, "_sync_with_binance_safe") as mock_safe:
            await tm.sync_async()
            mock_safe.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_async_clears_sync_needed(self, tm):
        """sync_async limpa _sync_needed após execução."""
        tm._sync_needed = True
        with patch.object(tm, "_sync_with_binance_safe"):
            await tm.sync_async()
        assert tm._sync_needed is False

    @pytest.mark.asyncio
    async def test_sync_async_dedup_concurrent(self, tm):
        """Chamadas concorrentes de sync_async são serializadas pelo lock."""
        call_count = {"n": 0}
        original_safe = tm._sync_with_binance_safe

        def counting_safe():
            call_count["n"] += 1
            time.sleep(0.05)  # simula latência de rede

        with patch.object(tm, "_sync_with_binance_safe", side_effect=counting_safe):
            # 3 chamadas concorrentes — devem ser serializadas
            await asyncio.gather(
                tm.sync_async(),
                tm.sync_async(),
                tm.sync_async(),
            )
        # Todas executam (serializadas pelo lock), mas não explodem
        assert call_count["n"] == 3


# ──────────────────────────────────────────────
# Testes: _sync_with_binance_safe
# ──────────────────────────────────────────────

class TestSyncWithBinanceSafe:

    def test_safe_calls_sync(self, tm):
        """_sync_with_binance_safe chama _sync_with_binance."""
        with patch.object(tm, "_sync_with_binance") as mock_sync:
            tm._sync_with_binance_safe()
            mock_sync.assert_called_once()

    def test_safe_handles_exception(self, tm):
        """_sync_with_binance_safe não propaga exceções."""
        with patch.object(
            tm, "_sync_with_binance", side_effect=Exception("network error")
        ):
            # Não deve levantar exceção
            tm._sync_with_binance_safe()
            assert tm.time_sync_status == "degraded"

    def test_safe_sets_failed_when_no_previous_sync(self, tm):
        """Se nunca sincronizou, status vai para 'failed'."""
        tm.last_successful_sync_ms = None
        with patch.object(
            tm, "_sync_with_binance", side_effect=Exception("fail")
        ):
            tm._sync_with_binance_safe()
            assert tm.time_sync_status == "failed"


# ──────────────────────────────────────────────
# Testes: periodic_sync
# ──────────────────────────────────────────────

class TestPeriodicSync:

    @pytest.mark.asyncio
    async def test_periodic_sync_calls_sync_async(self, tm):
        """periodic_sync chama sync_async periodicamente."""
        call_count = {"n": 0}

        async def mock_sync_async():
            call_count["n"] += 1

        with patch.object(tm, "sync_async", side_effect=mock_sync_async):
            task = asyncio.create_task(tm.periodic_sync(interval=0.1))
            await asyncio.sleep(0.35)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert call_count["n"] >= 2  # ~3 chamadas em 0.35s com interval=0.1

    @pytest.mark.asyncio
    async def test_periodic_sync_survives_error(self, tm):
        """periodic_sync continua rodando mesmo com erro."""
        call_count = {"n": 0}

        async def failing_sync():
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise Exception("transient error")

        with patch.object(tm, "sync_async", side_effect=failing_sync):
            task = asyncio.create_task(tm.periodic_sync(interval=0.1))
            await asyncio.sleep(0.45)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert call_count["n"] >= 3  # Continua após erros

    @pytest.mark.asyncio
    async def test_periodic_sync_cancellable(self, tm):
        """periodic_sync pode ser cancelada limpo."""
        with patch.object(tm, "sync_async", return_value=None):
            task = asyncio.create_task(tm.periodic_sync(interval=0.1))
            await asyncio.sleep(0.15)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


# ──────────────────────────────────────────────
# Testes: _validate_offset sem recursão
# ──────────────────────────────────────────────

class TestValidateOffsetNoRecursion:

    def test_offset_ok(self, tm):
        """Offset dentro do limite: status ok, sem sync_needed."""
        tm.server_time_offset_ms = 100
        tm._validate_offset()
        assert tm._sync_needed is False
        assert tm._correction_attempts == 0

    def test_offset_degraded_under_10s(self, tm):
        """Offset < 10s: degraded mas sem re-sync."""
        tm.server_time_offset_ms = 5000
        tm._validate_offset()
        assert tm.time_sync_status == "degraded"
        assert tm._sync_needed is False

    def test_offset_high_10s_to_60s(self, tm):
        """Offset 10s-60s: degraded + sync_needed."""
        tm.server_time_offset_ms = 30000
        tm._validate_offset()
        assert tm.time_sync_status == "degraded"
        assert tm._sync_needed is True

    def test_offset_critical(self, tm):
        """Offset > 60s: critical log + sync_needed."""
        tm.server_time_offset_ms = 90000
        with patch.object(tm, "_try_system_ntp_sync", return_value=False):
            tm._validate_offset()
        assert tm.time_sync_status == "degraded"
        assert tm._sync_needed is True

    def test_offset_critical_ntp_success(self, tm):
        """Offset > 60s com NTP sucesso: sync_needed setado."""
        tm.server_time_offset_ms = 90000
        with patch.object(tm, "_try_system_ntp_sync", return_value=True):
            tm._validate_offset()
        assert tm._sync_needed is True

    def test_no_recursive_sync_call(self, tm):
        """_validate_offset NUNCA chama _sync_with_binance."""
        tm.server_time_offset_ms = 90000
        with patch.object(tm, "_try_system_ntp_sync", return_value=True):
            with patch.object(tm, "_sync_with_binance") as mock_sync:
                tm._validate_offset()
                mock_sync.assert_not_called()

    def test_offset_history_appended(self, tm):
        """Cada chamada adiciona ao histórico."""
        tm.server_time_offset_ms = 100
        tm._validate_offset()
        assert len(tm._last_offset_history) == 1
        assert tm._last_offset_history[0] == 100

    def test_offset_history_max_10(self, tm):
        """Histórico mantém no máximo 10 entradas."""
        for i in range(15):
            tm.server_time_offset_ms = i * 10
            tm._validate_offset()
        assert len(tm._last_offset_history) == 10


# ──────────────────────────────────────────────
# Testes: E2E — now() + sync_async
# ──────────────────────────────────────────────

class TestE2EFlow:

    @pytest.mark.asyncio
    async def test_now_flags_then_sync_clears(self, tm):
        """
        Fluxo completo:
        1. now() detecta que precisa sync → seta flag
        2. sync_async() executa → limpa flag
        """
        tm.last_sync_mono = 0  # forçar _should_sync() == True

        # Step 1: now() seta flag
        result = tm.now()
        assert isinstance(result, int)
        assert tm._sync_needed is True

        # Step 2: sync_async limpa flag
        with patch.object(tm, "_sync_with_binance_safe"):
            await tm.sync_async()
        assert tm._sync_needed is False

    @pytest.mark.asyncio
    async def test_now_does_not_block_event_loop(self, tm):
        """now() completa em < 1ms (não bloqueia)."""
        tm.last_sync_mono = 0  # forçar check de sync

        start = time.monotonic()
        for _ in range(1000):
            tm.now()
        elapsed = time.monotonic() - start

        # 1000 chamadas em < 0.1s (ou seja, ~0.1ms cada)
        assert elapsed < 0.1, f"now() muito lento: {elapsed:.3f}s para 1000 chamadas"

    @pytest.mark.asyncio
    async def test_periodic_picks_up_sync_needed(self, tm):
        """
        periodic_sync pega o flag _sync_needed e executa sync.
        """
        synced = {"done": False}

        async def mock_sync():
            synced["done"] = True
            tm._sync_needed = False

        tm._sync_needed = True

        with patch.object(tm, "sync_async", side_effect=mock_sync):
            task = asyncio.create_task(tm.periodic_sync(interval=0.05))
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert synced["done"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
