# structured_logging.py
from __future__ import annotations

import logging
import json
from typing import Any, Dict


class StructuredLogger:
    """
    Logger estruturado simples, gera uma linha JSON por evento.

    Uso:
        slog = StructuredLogger("orderbook", "BTCUSDT")
        slog.info("orderbook_event", bid_depth_usd=..., ask_depth_usd=...)
    """

    def __init__(self, name: str, symbol: str):
        self.logger = logging.getLogger(name)
        self.symbol = symbol

    def _build_payload(self, message: str, level: str, **extra: Any) -> str:
        payload: Dict[str, Any] = {
            "symbol": self.symbol,
            "event": message,
            "level": level,
            **extra,
        }
        # json.dumps garante uma linha por log, fácil de parsear em ELK/Splunk
        return json.dumps(payload, ensure_ascii=False)

    def info(self, message: str, **extra: Any) -> None:
        self.logger.info(self._build_payload(message, "INFO", **extra))

    def warning(self, message: str, **extra: Any) -> None:
        self.logger.warning(self._build_payload(message, "WARNING", **extra))

    def error(self, message: str, **extra: Any) -> None:
        self.logger.error(self._build_payload(message, "ERROR", **extra))