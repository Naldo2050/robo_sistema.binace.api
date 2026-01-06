# utils/logging_utils.py
# -*- coding: utf-8 -*-

"""
Utilitários de logging (filtro anti-eco).
Extraído do arquivo original market_orchestrator.py sem alterações de lógica.
"""

import logging
import time
from typing import Dict


class _DedupFilter(logging.Filter):
    """
    Filtro que suprime logs idênticos dentro de uma janela de tempo.

    Este código foi movido literalmente do market_orchestrator.py.
    Nenhuma modificação de lógica foi feita.
    """

    def __init__(self, window: float = 1.0) -> None:
        super().__init__()
        self.window = float(window)
        self._last: Dict[str, float] = {}  # msg -> timestamp

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        now = time.time()
        ts = self._last.get(msg)

        if ts is not None and (now - ts) < self.window:
            return False

        self._last[msg] = now
        return True


def configure_dedup_logs(window: float = 1.0) -> None:
    """
    Aplica globalmente o filtro anti-eco no logger raiz.

    Essa função simplesmente encapsula o trecho original:
    logging.getLogger().addFilter(_DedupFilter(window=1.0))
    """

    logging.getLogger().addFilter(_DedupFilter(window=window))
