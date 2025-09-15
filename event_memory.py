# event_memory.py
from collections import deque
from typing import List, Dict
import logging

# Memória global de eventos (em memória — para produção, use Redis, SQLite, etc.)
_event_memory = deque(maxlen=1000)  # Mantém os últimos 1000 eventos

def obter_memoria_eventos(n: int = 5) -> List[Dict]:
    """
    Retorna os últimos 'n' eventos armazenados.
    """
    return list(_event_memory)[-n:]

def adicionar_memoria_evento(evento: Dict):
    """
    Adiciona um evento ao histórico de memória.
    """
    if isinstance(evento, dict) and "timestamp" in evento:
        _event_memory.append(evento)
    else:
        logging.warning("Tentativa de adicionar evento inválido à memória.")

def calcular_probabilidade_historica(evento: Dict) -> Dict[str, float]:
    """
    Calcula probabilidades históricas baseadas em eventos anteriores.
    Simples placeholder — ajuste conforme sua lógica de ML/estatística.
    """
    tipo = evento.get("tipo_evento", "")
    resultado = evento.get("resultado_da_batalha", "")

    if "Absorção de Compra" in resultado:
        return {"long_prob": 0.65, "short_prob": 0.20, "neutral_prob": 0.15}
    elif "Absorção de Venda" in resultado:
        return {"long_prob": 0.20, "short_prob": 0.65, "neutral_prob": 0.15}
    elif "Exaustão" in tipo:
        return {"long_prob": 0.40, "short_prob": 0.40, "neutral_prob": 0.20}
    elif "Zona" in tipo:
        return {"long_prob": 0.50, "short_prob": 0.30, "neutral_prob": 0.20}
    else:
        return {"long_prob": 0.33, "short_prob": 0.33, "neutral_prob": 0.34}