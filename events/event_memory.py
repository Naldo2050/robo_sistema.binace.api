# event_memory.py
"""
Memória de eventos com probabilidade histórica REAL.

v2.0: Usa OutcomeTracker para calcular probabilidades baseadas em outcomes
reais dos sinais passados (não mais hardcoded).
"""

from collections import deque
from typing import Any, List, Dict, Optional
import logging

logger = logging.getLogger("EventMemory")

# Memória global de eventos (em memória — para produção, use Redis, SQLite, etc.)
_event_memory: deque = deque(maxlen=1000)  # Mantém os últimos 1000 eventos

# Outcome tracker para probabilidades reais
try:
    from outcome_tracker import OutcomeTracker
    _outcome_tracker: Optional[Any] = OutcomeTracker()
    _TRACKER_OK = True
except ImportError:
    _outcome_tracker = None
    _TRACKER_OK = False
    logger.warning("OutcomeTracker indisponível, usando probabilidades base")


def obter_memoria_eventos(n: int = 5) -> List[Dict]:
    """
    Retorna os últimos 'n' eventos armazenados.
    """
    return list(_event_memory)[-n:]


def adicionar_memoria_evento(evento: Dict):
    """
    Adiciona um evento ao histórico de memória.
    Também registra no OutcomeTracker para tracking de resultados.
    """
    if isinstance(evento, dict) and "timestamp" in evento:
        _event_memory.append(evento)

        # Registrar sinal para tracking de outcome
        if _TRACKER_OK and _outcome_tracker:
            tipo = evento.get("tipo_evento", "")
            if tipo in ("Absorção", "Exaustão", "Absorção"):
                _outcome_tracker.register_signal(evento)
    else:
        logger.warning("Tentativa de adicionar evento inválido à memória.")


def avaliar_outcomes_pendentes(preco_atual: float, epoch_ms: int):
    """
    Avalia outcomes pendentes com o preço atual.
    Chamado a cada janela de 5 minutos.
    """
    if _TRACKER_OK and _outcome_tracker:
        _outcome_tracker.evaluate_pending_outcomes(preco_atual, epoch_ms)


def calcular_probabilidade_historica(evento: Dict) -> Dict:
    """
    Calcula probabilidades históricas REAIS baseadas em outcomes passados.
    Usa OutcomeTracker (SQLite) quando disponível, senão fallback conservador.
    """
    # Tentar probabilidade real do OutcomeTracker
    if _TRACKER_OK and _outcome_tracker:
        confidence = _outcome_tracker.get_confidence_for_event(evento)
        if confidence.get("has_data"):
            # Usar janela de 15m como referência principal
            w15 = confidence.get("windows", {}).get("15m", {})
            if w15:
                prob_up = w15.get("prob_up", 0.33)
                prob_down = w15.get("prob_down", 0.33)
                prob_flat = 1.0 - prob_up - prob_down
                return {
                    "long_prob": round(prob_up, 4),
                    "short_prob": round(prob_down, 4),
                    "neutral_prob": round(max(0, prob_flat), 4),
                    "samples": w15.get("samples", 0),
                    "win_rate": w15.get("win_rate", 0),
                    "avg_return_pct": w15.get("avg_return_pct", 0),
                    "is_real_data": True,
                    "source": "outcome_tracker_15m",
                }

    # Fallback conservador (sem dados históricos suficientes)
    tipo = evento.get("tipo_evento", "")
    resultado = evento.get("resultado_da_batalha", "")

    # Probabilidades base conservadoras (não hardcoded otimistas)
    base = {"long_prob": 0.33, "short_prob": 0.33, "neutral_prob": 0.34,
            "is_real_data": False, "source": "fallback_base"}

    if "Compra" in resultado:
        base.update({"long_prob": 0.45, "short_prob": 0.25, "neutral_prob": 0.30})
    elif "Venda" in resultado:
        base.update({"long_prob": 0.25, "short_prob": 0.45, "neutral_prob": 0.30})
    elif "Exaustão" in tipo or "Exaust" in tipo:
        base.update({"long_prob": 0.35, "short_prob": 0.35, "neutral_prob": 0.30})

    return base