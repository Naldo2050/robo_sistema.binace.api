"""
Smart AI Throttler v3 — com rate limit awareness e budget diário.

Changelog v3:
- Adiciona RateLimitState com cooldown exponencial
- Budget diário de tokens (evita esgotar 100K em 50min)
- Parse do header Retry-After da Groq
- Singleton via get_throttler()
- Compatível com interface v2 (should_call_ai aceita dict OU kwargs)
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Eventos que SEMPRE passam pelo throttler (bypass soft interval)
ALWAYS_PROCESS = frozenset({
    "Exaustão",
    "Absorção",
    "whale_detected",
    "regime_change",
    "LARGE_TRADE",
})


@dataclass
class RateLimitState:
    """Estado de rate limit por provider/modelo."""
    is_limited: bool = False
    retry_after_ts: float = 0.0        # timestamp UTC quando pode tentar
    consecutive_429s: int = 0
    last_429_time: float = 0.0
    total_429s_today: int = 0


@dataclass
class SmartAIThrottler:
    """
    Controla chamadas IA com:
    - Intervalo mínimo entre chamadas (soft/hard)
    - Detecção de mudanças significativas
    - Budget diário de tokens
    - Cooldown inteligente após 429
    - Fallback de modelos
    """

    # --- Intervalos ---
    min_interval: float = 180.0          # soft min (s) — pode ser bypassed
    hard_min_interval: float = 60.0      # hard min (s) — nunca bypassed
    significant_imb_change: float = 0.5  # threshold de mudança de imbalance

    # --- Budget (ajustado para Groq free tier) ---
    daily_token_budget: int = 50_000     # Conservador para free tier
    tokens_per_call_estimate: int = 2_500
    max_calls_per_hour: int = 6          # Era 10 — reduzido para evitar 429

    # --- Cooldown 429 ---
    base_cooldown_429: float = 120.0     # 2 min base
    max_cooldown_429: float = 1800.0     # 30 min max

    # --- Estado interno (não passar no construtor) ---
    _last_call_ts: float = field(default=0.0, repr=False, init=False)
    _last_imbalance: Optional[float] = field(default=None, repr=False, init=False)
    _last_bsr: Optional[float] = field(default=None, repr=False, init=False)
    _calls_this_hour: int = field(default=0, repr=False, init=False)
    _hour_start_ts: float = field(default=0.0, repr=False, init=False)
    _tokens_used_today: int = field(default=0, repr=False, init=False)
    _day_start_ts: float = field(default=0.0, repr=False, init=False)
    _consecutive_skips: int = field(default=0, repr=False, init=False)
    _rate_limit: RateLimitState = field(
        default_factory=RateLimitState, repr=False, init=False
    )

    def __post_init__(self):
        now = time.time()
        self._hour_start_ts = now
        self._day_start_ts = now

    # ──────────────────────────────────────────────
    # API principal
    # ──────────────────────────────────────────────

    def should_call_ai(
        self,
        payload_or_event: Union[dict, None] = None,
        *,
        event_type: str = "ANALYSIS_TRIGGER",
        delta: float = 0.0,
        volume: float = 0.0,
        avg_volume: float = 10.0,
        window_count: int = 99,
    ) -> bool:
        """
        Decide se deve chamar a IA.
        Compatível com v2 (recebe dict) e v3 (recebe kwargs).

        Returns:
            True se deve chamar, False se deve pular.
        """
        # Compatibilidade v2: extrair dados do dict
        if isinstance(payload_or_event, dict):
            event_type = payload_or_event.get(
                "tipo_evento",
                payload_or_event.get("trigger", "ANALYSIS_TRIGGER"),
            )
            delta = payload_or_event.get("delta", 0.0)
            volume = payload_or_event.get("volume", 0.0)
            avg_volume = payload_or_event.get("avg_volume", 10.0)

        should, reason = self._evaluate(
            event_type, delta, volume, avg_volume, window_count
        )

        if not should:
            self._consecutive_skips += 1
            logger.info(
                "AI SKIP: %s | event=%s | delta=%+.2f | skips=%d",
                reason, event_type, delta, self._consecutive_skips,
            )
        else:
            logger.info("AI APPROVED: %s", reason)

        return should

    def _evaluate(
        self,
        event_type: str,
        delta: float,
        volume: float,
        avg_volume: float,
        window_count: int,
    ) -> tuple[bool, str]:
        """Lógica de decisão. Retorna (should_call, reason)."""
        now = time.time()

        # 1) Rate limit ativo? → bloquear
        if self._is_rate_limited(now):
            remaining = self._rate_limit.retry_after_ts - now
            return False, (
                f"rate_limit_cooldown "
                f"({remaining:.0f}s left, "
                f"429s={self._rate_limit.consecutive_429s})"
            )

        # 2) Budget diário esgotado?
        self._maybe_reset_daily(now)
        remaining_budget = self.daily_token_budget - self._tokens_used_today
        if remaining_budget < self.tokens_per_call_estimate:
            return False, (
                f"daily_budget_exhausted "
                f"({self._tokens_used_today}/{self.daily_token_budget})"
            )

        # 3) Limite por hora?
        self._maybe_reset_hourly(now)
        if self._calls_this_hour >= self.max_calls_per_hour:
            return False, (
                f"hourly_limit "
                f"({self._calls_this_hour}/{self.max_calls_per_hour})"
            )

        # 4) Hard min interval (NUNCA bypassed, exceto primeiros eventos)
        elapsed = now - self._last_call_ts
        if self._last_call_ts > 0 and elapsed < self.hard_min_interval:
            return False, (
                f"hard_min_interval "
                f"({elapsed:.0f}s < {self.hard_min_interval:.0f}s)"
            )

        # 5) Evento prioritário? → bypass soft interval
        if event_type in ALWAYS_PROCESS:
            return True, f"priority_event ({event_type})"

        # 6) Soft min interval
        if self._last_call_ts > 0 and elapsed < self.min_interval:
            is_sig = self._is_significant(delta, volume, avg_volume)
            if is_sig:
                return True, f"significant_change (delta={delta:+.2f}, elapsed={elapsed:.0f}s)"
            return False, (
                f"soft_min_interval "
                f"({elapsed:.0f}s < {self.min_interval:.0f}s)"
            )

        # 7) A cada 5 skips consecutivos, permitir chamada de manutenção
        if self._consecutive_skips >= 5:
            return True, f"maintenance_call (after {self._consecutive_skips} skips)"

        # 8) Passou tudo → permitir
        return True, "normal"

    def _is_significant(
        self, delta: float, volume: float, avg_volume: float
    ) -> bool:
        """Detecta mudanças significativas que justificam bypass."""
        # Delta grande
        if abs(delta) >= 5.0:
            return True
        # Volume anormal (2x média)
        if avg_volume > 0 and volume / avg_volume >= 2.0:
            return True
        return False

    # ──────────────────────────────────────────────
    # Registro de chamadas e erros
    # ──────────────────────────────────────────────

    def record_call(self, tokens_used: int = 0):
        """Registrar chamada realizada com sucesso."""
        now = time.time()
        self._last_call_ts = now
        self._calls_this_hour += 1
        self._consecutive_skips = 0

        actual = tokens_used if tokens_used > 0 else self.tokens_per_call_estimate
        self._tokens_used_today += actual

        remaining = self.daily_token_budget - self._tokens_used_today
        calls_left = max(0, remaining // self.tokens_per_call_estimate)

        logger.info(
            "AI Budget: %s/%s tokens | ~%d calls left | hour=%d/%d",
            f"{self._tokens_used_today:,}",
            f"{self.daily_token_budget:,}",
            calls_left,
            self._calls_this_hour,
            self.max_calls_per_hour,
        )

    def record_success(self):
        """Registrar resposta bem-sucedida (reseta contador 429)."""
        self._rate_limit.consecutive_429s = 0

    def record_rate_limit(self, retry_after_seconds: float = 0):
        """
        Registrar erro 429.
        Ativa cooldown exponencial. NÃO deve fazer retry.
        """
        rl = self._rate_limit
        rl.is_limited = True
        rl.consecutive_429s += 1
        rl.total_429s_today += 1
        rl.last_429_time = time.time()

        # Cooldown exponencial: 2min, 4min, 8min, 16min, max 30min
        if retry_after_seconds > 0:
            cooldown = max(retry_after_seconds, self.base_cooldown_429)
        else:
            cooldown = self.base_cooldown_429 * (2 ** (rl.consecutive_429s - 1))

        cooldown = min(cooldown, self.max_cooldown_429)
        rl.retry_after_ts = time.time() + cooldown

        logger.warning(
            "Rate limit #%d (total hoje: %d) | cooldown=%.0fs (%.1f min)",
            rl.consecutive_429s,
            rl.total_429s_today,
            cooldown,
            cooldown / 60,
        )

    # ──────────────────────────────────────────────
    # Utilitários
    # ──────────────────────────────────────────────

    def _is_rate_limited(self, now: float) -> bool:
        """Verifica se está em cooldown de rate limit."""
        if not self._rate_limit.is_limited:
            return False
        if now >= self._rate_limit.retry_after_ts:
            # Cooldown expirou
            self._rate_limit.is_limited = False
            logger.info("Rate limit cooldown expirou")
            return False
        return True

    def _maybe_reset_hourly(self, now: float):
        if now - self._hour_start_ts > 3600:
            self._calls_this_hour = 0
            self._hour_start_ts = now

    def _maybe_reset_daily(self, now: float):
        if now - self._day_start_ts > 86400:
            old_used = self._tokens_used_today
            self._tokens_used_today = 0
            self._rate_limit = RateLimitState()
            self._day_start_ts = now
            logger.info("AI Budget resetado (era %s tokens)", f"{old_used:,}")

    @staticmethod
    def parse_retry_after(error_message: str) -> float:
        """Extrai tempo de espera da mensagem de erro Groq."""
        # Pattern: "try again in 1m52.32s"
        match = re.search(r"try again in (\d+)m([\d.]+)s", error_message)
        if match:
            minutes = int(match.group(1))
            seconds = float(match.group(2))
            return minutes * 60 + seconds

        # Pattern: "try again in 52.32s"
        match = re.search(r"try again in ([\d.]+)s", error_message)
        if match:
            return float(match.group(1))

        return 120.0  # fallback: 2 min

    def get_status(self) -> dict:
        """Status completo para logging/debug."""
        now = time.time()
        return {
            "tokens_used": self._tokens_used_today,
            "tokens_remaining": self.daily_token_budget - self._tokens_used_today,
            "calls_this_hour": self._calls_this_hour,
            "max_calls_per_hour": self.max_calls_per_hour,
            "is_rate_limited": self._rate_limit.is_limited,
            "cooldown_remaining_s": max(
                0, self._rate_limit.retry_after_ts - now
            ),
            "consecutive_429s": self._rate_limit.consecutive_429s,
            "total_429s_today": self._rate_limit.total_429s_today,
            "consecutive_skips": self._consecutive_skips,
            "seconds_since_last_call": now - self._last_call_ts
                if self._last_call_ts > 0 else None,
        }


# ──────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────

_throttler_instance: Optional[SmartAIThrottler] = None


def get_throttler(**kwargs) -> SmartAIThrottler:
    """Retorna instância singleton do throttler."""
    global _throttler_instance
    if _throttler_instance is None:
        _throttler_instance = SmartAIThrottler(**kwargs)
        logger.info(
            "AI Throttler inicializado: interval=%.0fs, budget=%s, max/hour=%d",
            _throttler_instance.min_interval,
            f"{_throttler_instance.daily_token_budget:,}",
            _throttler_instance.max_calls_per_hour,
        )
    return _throttler_instance


def reset_throttler():
    """Reset singleton (para testes)."""
    global _throttler_instance
    _throttler_instance = None
