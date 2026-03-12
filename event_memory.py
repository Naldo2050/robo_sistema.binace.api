# event_memory.py — proxy de compatibilidade
# Modulo movido para events/event_memory.py
from events.event_memory import *  # noqa: F401,F403
from events.event_memory import (  # noqa: F401
    obter_memoria_eventos,
    adicionar_memoria_evento,
    avaliar_outcomes_pendentes,
    calcular_probabilidade_historica,
)
