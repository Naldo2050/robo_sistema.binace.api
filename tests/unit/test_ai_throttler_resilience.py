# tests/unit/test_ai_throttler_resilience.py
"""
Testes de resili?ncia e contrato de cadencia para SmartAIThrottler (#11).
Valida a separa??o entre:
1. Tentativa fisica de request externo (record_attempt, _last_attempt_ts)
2. An?lise conclu?da com sucesso (record_call, record_success, _last_call_ts)
3. Consumo real de tokens (_tokens_used_today)
"""

import time
from unittest.mock import MagicMock
import pytest

from common.ai_throttler import (
    SmartAIThrottler,
    get_throttler,
    init_throttler,
    reset_throttler_for_tests,
)
from market_orchestrator.ai import analyzer_qwen as mod


@pytest.fixture(autouse=True)
def clean_throttler_singleton():
    """Garante singleton limpo antes e depois de cada teste."""
    reset_throttler_for_tests()
    yield
    reset_throttler_for_tests()


@pytest.fixture
def no_sleep(monkeypatch):
    """Evita delays reais nos loops de retry."""
    monkeypatch.setattr(time, "sleep", lambda s: None)


def _make_mock_response(content: str = '{"action": "neutral", "confidence": 0.5}', total_tokens: int = 150):
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.usage = MagicMock()
    response.usage.total_tokens = total_tokens
    return response


def _make_analyzer(monkeypatch, mock_client=None) -> mod.AIAnalyzer:
    client = mock_client or MagicMock()

    def fake_init(self):
        self.mode = "groq"
        self.enabled = True
        self.client = client
        self.client_async = None
        self.model_name = "llama-3.1-8b-instant"

    monkeypatch.setattr(mod.AIAnalyzer, "_initialize_api", fake_init)
    monkeypatch.setattr(mod.AIAnalyzer, "_start_heartbeat", lambda self: None)

    analyzer = mod.AIAnalyzer(health_monitor=None)
    analyzer.mode = "groq"
    analyzer.enabled = True
    analyzer.client = client
    return analyzer


# ?????????????????????????????????????????????????????????????
# 1. test_record_attempt_only_updates_attempt_timestamp
# ?????????????????????????????????????????????????????????????

def test_record_attempt_only_updates_attempt_timestamp():
    """
    Verifica que record_attempt() atualiza apenas o timestamp da tentativa fisica
    e contador observacional, mantendo chamadas logicas, tokens e skips inalterados.
    """
    throttler = SmartAIThrottler(
        min_interval=60.0,
        hard_min_interval=20.0,
        daily_token_budget=50_000,
        max_calls_per_hour=6,
    )
    throttler._consecutive_skips = 3

    assert throttler._last_attempt_ts == 0.0
    assert throttler._last_call_ts == 0.0
    assert throttler._calls_this_hour == 0
    assert throttler._tokens_used_today == 0
    assert throttler._total_attempts == 0

    before_ts = time.time()
    throttler.record_attempt()
    after_ts = time.time()

    assert before_ts <= throttler._last_attempt_ts <= after_ts
    assert throttler._total_attempts == 1
    # Invariantes intocados:
    assert throttler._last_call_ts == 0.0
    assert throttler._calls_this_hour == 0
    assert throttler._tokens_used_today == 0
    assert throttler._consecutive_skips == 3
    assert throttler._rate_limit.is_limited is False


# ?????????????????????????????????????????????????????????????
# 2. test_timeout_attempt_enforces_hard_min_interval
# ?????????????????????????????????????????????????????????????

def test_timeout_attempt_enforces_hard_min_interval(monkeypatch, no_sleep):
    """
    Apos tentativa com TimeoutError:
    - nova decisao imediatamente depois deve ser bloqueada com reason='hard_min_interval'.
    """
    throttler = init_throttler(min_interval=60.0, hard_min_interval=30.0)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = TimeoutError("Request timed out")

    analyzer = _make_analyzer(monkeypatch, mock_client)

    text, error_reason = analyzer._call_openai_compatible("prompt de teste", max_retries=1)

    assert text == ""
    assert error_reason == "timeout"
    assert throttler._total_attempts == 1
    assert throttler._last_attempt_ts > 0

    # Decis?o imediatamente ap?s a falha de timeout
    should, reason = throttler._evaluate("ANALYSIS_TRIGGER", delta=0.0, volume=0.0, avg_volume=10.0, window_count=99)
    assert should is False
    assert "hard_min_interval" in reason

    # Em t + 31s (> hard_min 30s, mas < soft_min 60s):
    # - evento normal e bloqueado por soft_min_interval
    # - evento prioritario e aprovado (hard_min je expirou)
    time_between_hard_and_soft = throttler._last_attempt_ts + 31.0
    monkeypatch.setattr(time, "time", lambda: time_between_hard_and_soft)
    should_norm, reason_norm = throttler._evaluate("ANALYSIS_TRIGGER", delta=0.0, volume=0.0, avg_volume=10.0, window_count=99)
    assert should_norm is False
    assert "soft_min_interval" in reason_norm

    should_prio, reason_prio = throttler._evaluate("whale_detected", delta=0.0, volume=0.0, avg_volume=10.0, window_count=99)
    assert should_prio is True
    assert "priority_event" in reason_prio

    # Em t + 65s (> soft_min 60s): evento normal passa livremente
    time_after_soft = throttler._last_attempt_ts + 65.0
    monkeypatch.setattr(time, "time", lambda: time_after_soft)
    should_later, reason_later = throttler._evaluate("ANALYSIS_TRIGGER", delta=0.0, volume=0.0, avg_volume=10.0, window_count=99)
    assert should_later is True
    assert reason_later == "normal"


# ?????????????????????????????????????????????????????????????
# 3. test_5xx_attempt_enforces_hard_min_interval
# ?????????????????????????????????????????????????????????????

def test_5xx_attempt_enforces_hard_min_interval(monkeypatch, no_sleep):
    """
    Mesmo contrato para erro 5xx: cadencia fisica e imposta ap?s falha de servidor.
    """
    throttler = init_throttler(min_interval=60.0, hard_min_interval=25.0)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("500 Internal Server Error")

    analyzer = _make_analyzer(monkeypatch, mock_client)

    text, error_reason = analyzer._call_openai_compatible("prompt de teste", max_retries=1)

    assert text == ""
    assert throttler._total_attempts == 1
    assert throttler._last_attempt_ts > 0

    should, reason = throttler._evaluate("ANALYSIS_TRIGGER", delta=0.0, volume=0.0, avg_volume=10.0, window_count=99)
    assert should is False
    assert "hard_min_interval" in reason


# ?????????????????????????????????????????????????????????????
# 4. test_connection_error_attempt_enforces_hard_min_interval
# ?????????????????????????????????????????????????????????????

def test_connection_error_attempt_enforces_hard_min_interval(monkeypatch, no_sleep):
    """
    Mesmo contrato para erro de conex?o de rede.
    """
    throttler = init_throttler(min_interval=60.0, hard_min_interval=25.0)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionResetError("Connection reset by peer")

    analyzer = _make_analyzer(monkeypatch, mock_client)

    text, error_reason = analyzer._call_openai_compatible("prompt de teste", max_retries=1)

    assert text == ""
    assert throttler._total_attempts == 1
    assert throttler._last_attempt_ts > 0

    should, reason = throttler._evaluate("ANALYSIS_TRIGGER", delta=0.0, volume=0.0, avg_volume=10.0, window_count=99)
    assert should is False
    assert "hard_min_interval" in reason


# ?????????????????????????????????????????????????????????????
# 5. test_failed_attempt_does_not_consume_tokens
# ?????????????????????????????????????????????????????????????

def test_failed_attempt_does_not_consume_tokens(monkeypatch, no_sleep):
    """
    Falhas fisicas de provider NAO devem deduzir estimativas nem valores ficticios de tokens.
    """
    throttler = init_throttler(
        daily_token_budget=50_000,
        tokens_per_call_estimate=2500,
        hard_min_interval=10.0,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("503 Service Unavailable")

    analyzer = _make_analyzer(monkeypatch, mock_client)

    analyzer._call_openai_compatible("prompt", max_retries=2)

    assert throttler._total_attempts == 2
    assert throttler._tokens_used_today == 0
    assert throttler.get_status()["tokens_used"] == 0
    assert throttler.get_status()["tokens_remaining"] == 50_000


# ?????????????????????????????????????????????????????????????
# 6. test_retries_then_success_count_one_logical_call
# ?????????????????????????????????????????????????????????????

def test_retries_then_success_count_one_logical_call(monkeypatch, no_sleep):
    """
    Simula: tentativa 1 falha, tentativa 2 falha, tentativa 3 sucesso.
    Verifica:
    - record_attempt = 3
    - _calls_this_hour = +1
    - tokens debitados uma unica vez com usage real
    - record_call executado uma unica vez
    """
    throttler = init_throttler(daily_token_budget=50_000)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        Exception("502 Bad Gateway"),
        TimeoutError("timed out"),
        _make_mock_response(total_tokens=420),
    ]

    analyzer = _make_analyzer(monkeypatch, mock_client)

    text, error_reason = analyzer._call_openai_compatible("prompt", max_retries=3)

    assert error_reason is None
    assert throttler._total_attempts == 3
    assert throttler._calls_this_hour == 1
    assert throttler._tokens_used_today == 420


# ?????????????????????????????????????????????????????????????
# 7. test_all_retries_failed_count_zero_successful_calls
# ?????????????????????????????????????????????????????????????

def test_all_retries_failed_count_zero_successful_calls(monkeypatch, no_sleep):
    """
    Simula falha em todos os retries.
    Verifica:
    - multiplos attempts (3)
    - _calls_this_hour = 0
    - tokens = 0
    - cadencia registrada
    """
    throttler = init_throttler(min_interval=60.0, hard_min_interval=30.0)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        Exception("500 Internal Error"),
        Exception("502 Bad Gateway"),
        Exception("503 Service Unavailable"),
    ]

    analyzer = _make_analyzer(monkeypatch, mock_client)

    text, error_reason = analyzer._call_openai_compatible("prompt", max_retries=3)

    assert text == ""
    assert error_reason is not None
    assert throttler._total_attempts == 3
    assert throttler._calls_this_hour == 0
    assert throttler._tokens_used_today == 0
    assert throttler._last_attempt_ts > 0

    should, reason = throttler._evaluate("ANALYSIS_TRIGGER", 0.0, 0.0, 10.0, 99)
    assert should is False
    assert "hard_min_interval" in reason


# ?????????????????????????????????????????????????????????????
# 8. test_guardrail_rejection_does_not_record_attempt
# ?????????????????????????????????????????????????????????????

def test_guardrail_rejection_does_not_record_attempt(monkeypatch):
    """
    Rejeicao de payload local por guardrail NAO deve registrar tentativa fisica
    nem alterar cadencia nem quota.
    """
    throttler = init_throttler(min_interval=60.0, hard_min_interval=30.0)

    mock_client = MagicMock()
    analyzer = _make_analyzer(monkeypatch, mock_client)

    # For?ar guardrail a rejeitar o payload retornando None
    monkeypatch.setattr("market_orchestrator.ai.analyzer_qwen._ensure_safe_llm_payload", lambda e: None)

    unsafe_event = {"tipo_evento": "Absorcao", "raw_payload": "leak_attempt"}
    result = analyzer.analyze(unsafe_event)

    assert result["is_fallback"] is True
    assert result["structured"]["_fallback_reason"] == "unsafe_payload"
    mock_client.chat.completions.create.assert_not_called()
    assert throttler._total_attempts == 0
    assert throttler._last_attempt_ts == 0.0
    assert throttler._calls_this_hour == 0


# ?????????????????????????????????????????????????????????????
# 9. test_429_records_attempt_and_preserves_rate_limit_cooldown
# ?????????????????????????????????????????????????????????????

def test_429_records_attempt_and_preserves_rate_limit_cooldown(monkeypatch, no_sleep):
    """
    Erro 429:
    - registra tentativa fisica;
    - calls_this_hour NAO e inflado;
    - token budget NAO e debitado;
    - cooldown exponencial continua sob responsabilidade de record_rate_limit.
    """
    throttler = init_throttler(min_interval=60.0, hard_min_interval=30.0, base_cooldown_429=120.0)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception(
        "Rate limit reached. Please try again in 45s. [429]"
    )

    analyzer = _make_analyzer(monkeypatch, mock_client)

    text, error_reason = analyzer._call_openai_compatible("prompt", max_retries=3)

    assert error_reason == "rate_limited"
    assert throttler._total_attempts == 1
    assert throttler._calls_this_hour == 0
    assert throttler._tokens_used_today == 0
    assert throttler._rate_limit.is_limited is True
    assert throttler._rate_limit.consecutive_429s == 1

    # Nova consulta bloqueada especificamente por rate limit cooldown
    should, reason = throttler._evaluate("ANALYSIS_TRIGGER", 0.0, 0.0, 10.0, 99)
    assert should is False
    assert "rate_limit_cooldown" in reason


# ?????????????????????????????????????????????????????????????
# 10. test_success_preserves_existing_throttler_contract
# ?????????????????????????????????????????????????????????????

def test_success_preserves_existing_throttler_contract(monkeypatch):
    """
    Provar que o comportamento historico de record_call continua intacto:
    - sucesso contabiliza 1 chamada logica e debita tokens;
    - chamada direta de record_call sem record_attempt continua respeitando hard_min_interval.
    """
    throttler = init_throttler(
        min_interval=60.0,
        hard_min_interval=20.0,
        daily_token_budget=50_000,
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_response(total_tokens=350)

    analyzer = _make_analyzer(monkeypatch, mock_client)

    text, error_reason = analyzer._call_openai_compatible("prompt")

    assert error_reason is None
    assert throttler._total_attempts == 1
    assert throttler._calls_this_hour == 1
    assert throttler._tokens_used_today == 350
    assert throttler._consecutive_skips == 0
    assert throttler._last_call_ts > 0
    assert throttler._last_attempt_ts > 0

    # Teste de compatibilidade retroativa: throttler onde apenas record_call() e chamado
    legacy_throttler = SmartAIThrottler(hard_min_interval=15.0)
    legacy_throttler.record_call(tokens_used=100)
    assert legacy_throttler._last_attempt_ts == 0.0
    assert legacy_throttler._last_call_ts > 0.0
    # Deve ser bloqueado usando fallback _last_call_ts
    should_legacy, reason_legacy = legacy_throttler._evaluate("ANALYSIS_TRIGGER", 0.0, 0.0, 10.0, 99)
    assert should_legacy is False
    assert "hard_min_interval" in reason_legacy


# ?????????????????????????????????????????????????????????????
# 11. test_record_attempt_exception_propagates_and_aborts_network_request
# ?????????????????????????????????????????????????????????????

def test_record_attempt_exception_propagates_and_aborts_network_request(monkeypatch):
    """
    Contrato A (fail-closed): se record_attempt() lancar erro inesperado:
    - O erro NAO e ignorado silenciosamente;
    - O erro propaga para fora de _call_openai_compatible;
    - client.chat.completions.create NUNCA e executado.
    """
    throttler = init_throttler(min_interval=60.0, hard_min_interval=20.0)

    # Simular falha interna inesperada no throttler
    def broken_record_attempt():
        raise RuntimeError("Falha interna inesperada no throttler")

    monkeypatch.setattr(throttler, "record_attempt", broken_record_attempt)

    mock_client = MagicMock()
    analyzer = _make_analyzer(monkeypatch, mock_client)

    # 1. Provar que _call_openai_compatible propaga a excecao diretamente
    with pytest.raises(RuntimeError, match="Falha interna inesperada no throttler"):
        analyzer._call_openai_compatible("prompt de teste")

    # 2. Provar que o request de rede NUNCA foi enviado
    mock_client.chat.completions.create.assert_not_called()

    # 3. Provar que em alto nivel (analyze), o robo lida de forma segura retornando fallback
    safe_event = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "symbol": "BTCUSDT",
        "preco_fechamento": 90000.0,
    }
    result = analyzer.analyze(safe_event)
    assert result["is_fallback"] is True
    # O request de rede continua NUNCA tendo sido chamado
    mock_client.chat.completions.create.assert_not_called()
