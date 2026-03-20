"""
common/ai_throttler.py — Smart AI Call Throttler v2.0

Controla a frequência de chamadas à IA para redução de custos.

Regras:
  - ANALYSIS_TRIGGER: mínimo 180s entre chamadas
  - Hard minimum: 60s absoluto (mesmo para eventos reais)
  - Eventos reais (Absorção, Exaustão, etc.): bypass do soft min
  - Mudança significativa: bypass do soft min (threshold ajustado)

Economia estimada: 50-67% das chamadas de ANALYSIS_TRIGGER.

Changelog v2.0 (2026-03-13):
  - hard_min_interval = 60s (mínimo absoluto)
  - significant_imb_change = 0.5 (era 0.25 — muito sensível)
  - BSR threshold exige mudança >50%
  - Logging detalhado com estatísticas
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SmartAIThrottler:
    """
    Controla frequência de chamadas à IA.

    Uso:
        throttler = SmartAIThrottler(min_interval=180)

        if throttler.should_call_ai(compact_payload):
            result = await ai.analyze(compact_payload)
        else:
            # Pular chamada — economia de tokens
            pass
    """

    # Eventos que recebem tratamento prioritário (bypass soft min)
    ALWAYS_PROCESS: set[str] = {
        "Absorção", "Exaustão", "Breakout", "Whale Activity",
        "Divergência", "Reversão", "Volume Spike", "Momentum Shift",
        "ABS", "EXH", "BRK", "WHL", "DIV", "REV", "VSPK", "MOM",
    }

    def __init__(
        self,
        min_interval: int = 180,
        hard_min_interval: int = 60,
        significant_imb_change: float = 0.5,
        significant_bsr_change: float = 0.5,
    ):
        """
        Args:
            min_interval: Intervalo mínimo entre chamadas (soft, pode ser
                         bypassado para eventos importantes ou mudanças
                         significativas). Padrão: 180s (3 min).
            hard_min_interval: Intervalo mínimo ABSOLUTO. Nenhuma chamada
                              será feita antes deste tempo. Padrão: 60s.
            significant_imb_change: Mudança mínima no flow imbalance para
                                   considerar "significativa". Padrão: 0.5.
            significant_bsr_change: Mudança mínima % no BSR para considerar
                                   "significativa". Padrão: 0.5 (50%).
        """
        self.min_interval = min_interval
        self.hard_min_interval = hard_min_interval
        self.significant_imb_change = significant_imb_change
        self.significant_bsr_change = significant_bsr_change

        self._last_call_ts: float = 0.0
        self._last_flow_state: dict = {}
        self._calls_saved: int = 0
        self._calls_total: int = 0
        self._calls_made: int = 0

    def should_call_ai(self, compact_payload: dict) -> bool:
        """
        Decide se a IA deve ser chamada para este payload.

        Returns:
            True: Chamar a IA (gastar tokens)
            False: Pular chamada (economizar tokens)
        """
        self._calls_total += 1
        now = time.time()
        elapsed = now - self._last_call_ts if self._last_call_ts > 0 else float("inf")

        trigger = compact_payload.get("t", compact_payload.get("trigger", ""))

        # ===== HARD MINIMUM: Nunca chamar antes de 60s =====
        if elapsed < self.hard_min_interval:
            self._calls_saved += 1
            logger.debug(
                f"AI_THROTTLE: BLOCKED (hard_min) | "
                f"trigger={trigger} | "
                f"elapsed={elapsed:.0f}s < {self.hard_min_interval}s | "
                f"saved={self._calls_saved}/{self._calls_total} "
                f"({self._saving_pct}%)"
            )
            return False

        # ===== EVENTOS IMPORTANTES: Bypass do soft min =====
        if trigger in self.ALWAYS_PROCESS:
            self._accept_call(now, compact_payload, trigger, "important_event")
            return True

        # ===== ANALYSIS_TRIGGER: Verificar soft min =====
        if elapsed < self.min_interval:
            # Dentro do soft min: só chamar se mudança significativa
            if self._is_significant_change(compact_payload):
                self._accept_call(
                    now, compact_payload, trigger, "significant_change"
                )
                return True

            self._calls_saved += 1
            logger.debug(
                f"AI_THROTTLE: SKIPPED | "
                f"trigger={trigger} | "
                f"elapsed={elapsed:.0f}s < {self.min_interval}s | "
                f"no significant change | "
                f"saved={self._calls_saved}/{self._calls_total} "
                f"({self._saving_pct}%)"
            )
            return False

        # ===== INTERVALO OK (> 180s): Chamar =====
        self._accept_call(now, compact_payload, trigger, "interval_ok")
        return True

    def _accept_call(
        self,
        now: float,
        payload: dict,
        trigger: str,
        reason: str,
    ) -> None:
        """Registra que a chamada foi aceita."""
        elapsed = now - self._last_call_ts if self._last_call_ts > 0 else 0
        self._last_call_ts = now
        self._last_flow_state = self._extract_flow_state(payload)
        self._calls_made += 1

        logger.debug(
            f"AI_THROTTLE: ACCEPTED | "
            f"trigger={trigger} | reason={reason} | "
            f"elapsed={elapsed:.0f}s | "
            f"calls={self._calls_made} | "
            f"saved={self._calls_saved}/{self._calls_total} "
            f"({self._saving_pct}%)"
        )

    def _is_significant_change(self, payload: dict) -> bool:
        """
        Detecta mudanças significativas que justificam chamada antecipada.

        Critérios (v2 — thresholds mais altos):
        1. Mudança no imbalance > 0.5 (era 0.25)
        2. BSR cruzou 1.0 E mudou mais de 50%
        """
        flow = payload.get("f", payload.get("flow", {}))
        prev = self._last_flow_state

        if not prev:
            return True

        # 1. Mudança grande no imbalance
        curr_imb = flow.get("imb", 0)
        prev_imb = prev.get("imb", 0)
        imb_diff = abs(curr_imb - prev_imb)
        if imb_diff > self.significant_imb_change:
            logger.debug(
                f"AI_THROTTLE: Significant IMB change: "
                f"{prev_imb:.2f} → {curr_imb:.2f} (diff={imb_diff:.2f})"
            )
            return True

        # 2. BSR cruzou 1.0 (sell→buy dominant ou vice-versa) E mudança > 50%
        curr_bsr = flow.get("bsr", 0.5)
        prev_bsr = prev.get("bsr", 0.5)

        if (curr_bsr > 1.0) != (prev_bsr > 1.0):
            bsr_change = abs(curr_bsr - prev_bsr) / max(abs(prev_bsr), 0.01)
            if bsr_change > self.significant_bsr_change:
                logger.debug(
                    f"AI_THROTTLE: Significant BSR change: "
                    f"{prev_bsr:.2f} → {curr_bsr:.2f} (change={bsr_change:.0%})"
                )
                return True

        return False

    def _extract_flow_state(self, payload: dict) -> dict:
        """Extrai estado de fluxo para comparação futura."""
        flow = payload.get("f", payload.get("flow", {}))
        return {
            "imb": flow.get("imb", 0),
            "bsr": flow.get("bsr", 0.5),
        }

    @property
    def _saving_pct(self) -> int:
        """Percentual de chamadas economizadas."""
        if self._calls_total == 0:
            return 0
        return int(self._calls_saved * 100 / self._calls_total)

    @property
    def stats(self) -> dict:
        """Estatísticas do throttler."""
        return {
            "total_events": self._calls_total,
            "calls_made": self._calls_made,
            "calls_saved": self._calls_saved,
            "saving_rate": f"{self._saving_pct}%",
            "min_interval": self.min_interval,
            "hard_min_interval": self.hard_min_interval,
            "imb_threshold": self.significant_imb_change,
        }

    def reset_stats(self) -> None:
        """Reseta estatísticas (útil para testes)."""
        self._calls_saved = 0
        self._calls_total = 0
        self._calls_made = 0


# ============================================================
# INSTÂNCIA GLOBAL (singleton pattern)
# ============================================================

_global_throttler: Optional[SmartAIThrottler] = None


def get_throttler(
    min_interval: int = 180,
    hard_min_interval: int = 60,
    significant_imb_change: float = 0.5,
) -> SmartAIThrottler:
    """
    Obtém instância global do throttler (singleton).

    Uso:
        from common.ai_throttler import get_throttler
        throttler = get_throttler()
        if throttler.should_call_ai(payload):
            ...
    """
    global _global_throttler
    if _global_throttler is None:
        _global_throttler = SmartAIThrottler(
            min_interval=min_interval,
            hard_min_interval=hard_min_interval,
            significant_imb_change=significant_imb_change,
        )
    return _global_throttler
