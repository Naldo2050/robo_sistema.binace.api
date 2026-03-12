# events/__init__.py
"""
Pacote de eventos do sistema.

Contém: EventBus, EventSaver, EventMemory, EventSimilarity, EventStatsModel.
"""

from .event_bus import EventBus
from .event_saver import EventSaver, get_event_saver
from .event_memory import (
    obter_memoria_eventos,
    adicionar_memoria_evento,
    avaliar_outcomes_pendentes,
    calcular_probabilidade_historica,
)

__all__ = [
    "EventBus",
    "EventSaver",
    "get_event_saver",
    "obter_memoria_eventos",
    "adicionar_memoria_evento",
    "avaliar_outcomes_pendentes",
    "calcular_probabilidade_historica",
]
