"""
Testes para SmartAIThrottler v3.
Cobre: rate limit, budget, cooldown, retry parsing, singleton.
"""

import time
import pytest

from common.ai_throttler import (
    SmartAIThrottler,
    get_throttler,
    reset_throttler,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_singleton():
    """Reseta singleton antes e depois de cada teste."""
    reset_throttler()
    yield
    reset_throttler()


@pytest.fixture
def throttler() -> SmartAIThrottler:
    """Throttler com config de teste (intervalos curtos)."""
    return SmartAIThrottler(
        min_interval=10.0,
        hard_min_interval=2.0,
        daily_token_budget=10_000,
        tokens_per_call_estimate=500,
        max_calls_per_hour=5,
        base_cooldown_429=5.0,
        max_cooldown_429=60.0,
    )


# ──────────────────────────────────────────────
# Testes: Intervalo minimo
# ──────────────────────────────────────────────

class TestMinInterval:

    def test_first_call_always_allowed(self, throttler):
        """Primeira chamada deve ser permitida."""
        assert throttler.should_call_ai(event_type="ANALYSIS_TRIGGER") is True

    def test_second_call_within_hard_min_blocked(self, throttler):
        """Chamada dentro do hard_min deve ser bloqueada."""
        throttler.record_call()
        # Imediatamente apos
        assert throttler.should_call_ai(event_type="ANALYSIS_TRIGGER") is False

    def test_second_call_after_hard_min_but_within_soft(self, throttler):
        """Apos hard_min mas antes de soft_min: bloqueada se nao significativa."""
        throttler.record_call()
        # Simular passagem de tempo (> hard_min, < soft_min)
        throttler._last_call_ts = time.time() - 5  # 5s atras (> 2s hard)
        result = throttler.should_call_ai(
            event_type="ANALYSIS_TRIGGER",
            delta=1.0,  # pequeno, nao significativo
            volume=5.0,
            avg_volume=10.0,
        )
        assert result is False

    def test_significant_delta_bypasses_soft_min(self, throttler):
        """Delta grande deve bypassar soft_min."""
        throttler.record_call()
        throttler._last_call_ts = time.time() - 5  # 5s atras
        result = throttler.should_call_ai(
            event_type="ANALYSIS_TRIGGER",
            delta=7.0,  # > 5.0 threshold
            volume=10.0,
            avg_volume=10.0,
        )
        assert result is True

    def test_significant_volume_bypasses_soft_min(self, throttler):
        """Volume anormal (2x media) deve bypassar soft_min."""
        throttler.record_call()
        throttler._last_call_ts = time.time() - 5
        result = throttler.should_call_ai(
            event_type="ANALYSIS_TRIGGER",
            delta=1.0,
            volume=25.0,  # 2.5x avg
            avg_volume=10.0,
        )
        assert result is True

    def test_call_after_soft_min_allowed(self, throttler):
        """Apos soft_min, chamada normal deve ser permitida."""
        throttler.record_call()
        throttler._last_call_ts = time.time() - 15  # > 10s soft_min
        assert throttler.should_call_ai(event_type="ANALYSIS_TRIGGER") is True


# ──────────────────────────────────────────────
# Testes: Eventos prioritarios
# ──────────────────────────────────────────────

class TestPriorityEvents:

    def test_exhaustion_bypasses_soft_interval(self, throttler):
        """Exaustao deve bypassar soft interval."""
        throttler.record_call()
        throttler._last_call_ts = time.time() - 5  # < soft_min
        result = throttler.should_call_ai(event_type="Exaustão")
        assert result is True

    def test_absorption_bypasses_soft_interval(self, throttler):
        """Absorcao deve bypassar soft interval."""
        throttler.record_call()
        throttler._last_call_ts = time.time() - 5
        result = throttler.should_call_ai(event_type="Absorção")
        assert result is True

    def test_priority_event_still_respects_hard_min(self, throttler):
        """Evento prioritario deve respeitar hard_min."""
        throttler.record_call()
        # Imediatamente apos (< hard_min)
        result = throttler.should_call_ai(event_type="Exaustão")
        assert result is False


# ──────────────────────────────────────────────
# Testes: Rate Limit (429)
# ──────────────────────────────────────────────

class TestRateLimit:

    def test_429_blocks_subsequent_calls(self, throttler):
        """Apos 429, chamadas devem ser bloqueadas."""
        throttler.record_rate_limit(retry_after_seconds=60)
        result = throttler.should_call_ai(event_type="ANALYSIS_TRIGGER")
        assert result is False

    def test_429_blocks_even_priority_events(self, throttler):
        """Mesmo eventos prioritarios sao bloqueados durante rate limit."""
        throttler.record_rate_limit(retry_after_seconds=60)
        result = throttler.should_call_ai(event_type="Exaustão")
        assert result is False

    def test_429_cooldown_expires(self, throttler):
        """Apos cooldown expirar, chamadas voltam a ser permitidas."""
        throttler.record_rate_limit(retry_after_seconds=5)
        # Simular passagem do cooldown
        throttler._rate_limit.retry_after_ts = time.time() - 1
        result = throttler.should_call_ai(event_type="ANALYSIS_TRIGGER")
        assert result is True

    def test_429_consecutive_increases_cooldown(self, throttler):
        """429 consecutivos devem aumentar cooldown exponencialmente."""
        throttler.record_rate_limit()  # 5s base
        first_retry = throttler._rate_limit.retry_after_ts

        # Simular segundo 429
        throttler._rate_limit.is_limited = False  # reset para testar
        throttler.record_rate_limit()  # deve ser 10s
        second_retry = throttler._rate_limit.retry_after_ts

        # Segundo cooldown deve ser maior
        assert second_retry > first_retry

    def test_429_max_cooldown_capped(self, throttler):
        """Cooldown nao deve exceder max_cooldown_429."""
        for _ in range(20):  # Muitos 429 consecutivos
            throttler._rate_limit.is_limited = False
            throttler.record_rate_limit()

        now = time.time()
        max_expected = now + throttler.max_cooldown_429 + 1
        assert throttler._rate_limit.retry_after_ts <= max_expected

    def test_success_resets_consecutive_429s(self, throttler):
        """Sucesso deve resetar contador de 429s consecutivos."""
        throttler.record_rate_limit()
        throttler.record_rate_limit()
        assert throttler._rate_limit.consecutive_429s == 2

        throttler.record_success()
        assert throttler._rate_limit.consecutive_429s == 0


# ──────────────────────────────────────────────
# Testes: Budget diario
# ──────────────────────────────────────────────

class TestDailyBudget:

    def test_budget_blocks_when_exhausted(self, throttler):
        """Chamadas devem ser bloqueadas quando budget esgota."""
        # Consumir quase todo o budget
        throttler._tokens_used_today = 9_800  # budget=10_000, per_call=500
        result = throttler.should_call_ai(event_type="ANALYSIS_TRIGGER")
        assert result is False

    def test_budget_allows_when_available(self, throttler):
        """Chamadas permitidas quando ha budget."""
        throttler._tokens_used_today = 5_000
        # Precisa ter passado o soft_min
        throttler._last_call_ts = time.time() - 15
        result = throttler.should_call_ai(event_type="ANALYSIS_TRIGGER")
        assert result is True

    def test_record_call_decrements_budget(self, throttler):
        """record_call deve contabilizar tokens."""
        initial = throttler._tokens_used_today
        throttler.record_call(tokens_used=1234)
        assert throttler._tokens_used_today == initial + 1234

    def test_record_call_uses_estimate_when_zero(self, throttler):
        """Se tokens_used=0, usar estimativa."""
        throttler.record_call(tokens_used=0)
        assert throttler._tokens_used_today == throttler.tokens_per_call_estimate

    def test_daily_reset(self, throttler):
        """Budget deve resetar apos 24h."""
        throttler._tokens_used_today = 9_000
        throttler._day_start_ts = time.time() - 86_401  # > 24h
        # Trigger reset via _evaluate
        throttler.should_call_ai(event_type="ANALYSIS_TRIGGER")
        assert throttler._tokens_used_today == 0


# ──────────────────────────────────────────────
# Testes: Limite por hora
# ──────────────────────────────────────────────

class TestHourlyLimit:

    def test_hourly_limit_blocks(self, throttler):
        """Exceder max_calls_per_hour deve bloquear."""
        throttler._calls_this_hour = 5  # max=5
        throttler._last_call_ts = time.time() - 15
        result = throttler.should_call_ai(event_type="ANALYSIS_TRIGGER")
        assert result is False

    def test_hourly_resets_after_3600s(self, throttler):
        """Contador horario deve resetar apos 1h."""
        throttler._calls_this_hour = 5
        throttler._hour_start_ts = time.time() - 3601
        throttler._last_call_ts = time.time() - 15
        result = throttler.should_call_ai(event_type="ANALYSIS_TRIGGER")
        assert result is True
        assert throttler._calls_this_hour == 0


# ──────────────────────────────────────────────
# Testes: Manutencao (anti-starvation)
# ──────────────────────────────────────────────

class TestMaintenance:

    def test_maintenance_call_after_5_skips(self, throttler):
        """Apos 5 skips consecutivos, permitir chamada de manutencao."""
        throttler._consecutive_skips = 5
        throttler._last_call_ts = time.time() - 15  # > soft_min
        result = throttler.should_call_ai(
            event_type="ANALYSIS_TRIGGER",
            delta=0.1,  # pequeno, nao significativo
        )
        assert result is True

    def test_record_call_resets_skips(self, throttler):
        """record_call deve resetar contador de skips."""
        throttler._consecutive_skips = 10
        throttler.record_call()
        assert throttler._consecutive_skips == 0


# ──────────────────────────────────────────────
# Testes: Parse Retry-After
# ──────────────────────────────────────────────

class TestParseRetryAfter:

    def test_parse_minutes_and_seconds(self):
        """Deve parsear '1m52.32s' corretamente."""
        msg = (
            "Rate limit reached... "
            "Please try again in 1m52.32s."
        )
        result = SmartAIThrottler.parse_retry_after(msg)
        assert abs(result - 112.32) < 0.01

    def test_parse_large_minutes(self):
        """Deve parsear '28m9.12s' corretamente."""
        msg = "try again in 28m9.12s. Need more tokens?"
        result = SmartAIThrottler.parse_retry_after(msg)
        assert abs(result - 1689.12) < 0.01

    def test_parse_seconds_only(self):
        """Deve parsear '52.32s' sem minutos."""
        msg = "try again in 52.32s"
        result = SmartAIThrottler.parse_retry_after(msg)
        assert abs(result - 52.32) < 0.01

    def test_parse_no_match_returns_default(self):
        """Sem match, deve retornar 120s default."""
        msg = "Some random error without retry info"
        result = SmartAIThrottler.parse_retry_after(msg)
        assert result == 120.0


# ──────────────────────────────────────────────
# Testes: Singleton
# ──────────────────────────────────────────────

class TestSingleton:

    def test_get_throttler_returns_same_instance(self):
        """get_throttler deve retornar mesma instancia."""
        t1 = get_throttler()
        t2 = get_throttler()
        assert t1 is t2

    def test_reset_throttler_clears_instance(self):
        """reset_throttler deve limpar singleton."""
        t1 = get_throttler()
        reset_throttler()
        t2 = get_throttler()
        assert t1 is not t2

    def test_get_throttler_passes_kwargs_only_first_time(self):
        """kwargs so sao usados na primeira chamada."""
        t1 = get_throttler(min_interval=999)
        assert t1.min_interval == 999
        t2 = get_throttler(min_interval=1)  # ignorado
        assert t2.min_interval == 999


# ──────────────────────────────────────────────
# Testes: Compatibilidade v2 (dict input)
# ──────────────────────────────────────────────

class TestV2Compatibility:

    def test_accepts_dict_payload(self, throttler):
        """Deve aceitar dict como em v2."""
        payload = {
            "tipo_evento": "ANALYSIS_TRIGGER",
            "delta": 1.0,
            "volume": 5.0,
        }
        # Nao deve dar erro
        result = throttler.should_call_ai(payload)
        assert isinstance(result, bool)

    def test_dict_with_priority_event(self, throttler):
        """Dict com evento prioritario deve bypassar soft interval."""
        throttler.record_call()
        throttler._last_call_ts = time.time() - 5  # < soft_min, > hard_min
        payload = {"tipo_evento": "Exaustão"}
        result = throttler.should_call_ai(payload)
        assert result is True

    def test_none_payload_uses_defaults(self, throttler):
        """None como payload deve usar defaults."""
        result = throttler.should_call_ai(None)
        assert isinstance(result, bool)


# ──────────────────────────────────────────────
# Testes: get_status
# ──────────────────────────────────────────────

class TestGetStatus:

    def test_status_has_all_keys(self, throttler):
        """Status deve conter todas as chaves esperadas."""
        status = throttler.get_status()
        expected_keys = {
            "tokens_used",
            "tokens_remaining",
            "calls_this_hour",
            "max_calls_per_hour",
            "is_rate_limited",
            "cooldown_remaining_s",
            "consecutive_429s",
            "total_429s_today",
            "consecutive_skips",
            "seconds_since_last_call",
        }
        assert expected_keys.issubset(status.keys())

    def test_status_reflects_rate_limit(self, throttler):
        """Status deve refletir estado de rate limit."""
        throttler.record_rate_limit(30)
        status = throttler.get_status()
        assert status["is_rate_limited"] is True
        assert status["cooldown_remaining_s"] > 0
        assert status["consecutive_429s"] == 1


# ──────────────────────────────────────────────
# Testes: Cenario end-to-end (simula sequencia real)
# ──────────────────────────────────────────────

class TestE2EScenario:

    def test_realistic_sequence(self, throttler):
        """
        Simula 10 janelas como nos logs reais.
        Espera: ~2-3 chamadas aprovadas, nao 10.
        """
        approved = 0
        rejected = 0

        for window in range(1, 11):
            throttler._last_call_ts = max(
                0, throttler._last_call_ts
            )
            # Simular passagem de ~60s entre janelas
            if throttler._last_call_ts > 0:
                throttler._last_call_ts -= 60

            result = throttler.should_call_ai(
                event_type="ANALYSIS_TRIGGER",
                delta=(-1) ** window * 2.0,  # alternar +/- 2
                volume=12.0,
                avg_volume=10.0,
                window_count=window,
            )

            if result:
                approved += 1
                throttler.record_call(tokens_used=2000)
            else:
                rejected += 1

        # Com soft_min=10s e 60s entre janelas, mas budget/hourly limita
        assert approved <= 6, f"Muitas chamadas aprovadas: {approved}/10"
        assert rejected >= 4, f"Poucas rejeitadas: {rejected}/10"

    def test_429_then_recovery(self, throttler):
        """
        Simula: chamada OK -> 429 -> cooldown -> recuperacao.
        """
        # 1. Primeira chamada OK
        assert throttler.should_call_ai(event_type="ANALYSIS_TRIGGER")
        throttler.record_call(2000)
        throttler.record_success()

        # 2. Segunda chamada -> 429
        throttler._last_call_ts = time.time() - 15  # > soft_min
        assert throttler.should_call_ai(event_type="ANALYSIS_TRIGGER")
        throttler.record_rate_limit(retry_after_seconds=10)

        # 3. Durante cooldown -> bloqueado
        assert throttler.should_call_ai(event_type="ANALYSIS_TRIGGER") is False

        # 4. Apos cooldown -> permitido
        throttler._rate_limit.retry_after_ts = time.time() - 1
        throttler._last_call_ts = time.time() - 15
        assert throttler.should_call_ai(event_type="ANALYSIS_TRIGGER") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
