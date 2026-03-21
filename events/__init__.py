# events/__init__.py
"""
Pacote de eventos do sistema.

Contém: EventBus, EventSaver, EventMemory, EventSimilarity, EventStatsModel.
"""

from .event_bus import EventBus  # noqa: F401
from .event_memory import (  # noqa: F401
    adicionar_memoria_evento,
    avaliar_outcomes_pendentes,
    calcular_probabilidade_historica,
    obter_memoria_eventos,
)
from .event_saver import EventSaver, get_event_saver  # noqa: F401
from .event_similarity import EventSimilaritySearch  # noqa: F401

__all__ = [
    "EventBus",
    "EventSaver",
    "EventSimilaritySearch",
    "adicionar_memoria_evento",
    "avaliar_outcomes_pendentes",
    "calcular_probabilidade_historica",
    "get_event_saver",
    "obter_memoria_eventos",
]
