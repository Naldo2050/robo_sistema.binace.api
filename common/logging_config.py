# common/logging_config.py
"""
Configuracao centralizada de logging para todo o sistema.

Uso:
    from common.logging_config import setup_logging, get_logger

    # No main.py (uma vez):
    setup_logging(level="INFO", mode="production", log_file="logs/bot.log")

    # Em qualquer modulo:
    logger = get_logger(__name__)
    logger.info("Trade processed", extra={"symbol": "BTCUSDT"})
"""

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formatter que produz logs em JSON (para ELK/Datadog/Loki)."""

    _SKIP_FIELDS = {
        "name", "msg", "args", "created", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs",
        "pathname", "process", "processName", "relativeCreated",
        "stack_info", "exc_info", "exc_text", "thread", "threadName",
        "message", "asctime", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "loc": f"{record.filename}:{record.lineno}",
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Campos extras do caller
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in self._SKIP_FIELDS
        }
        if extra:
            # Garante serializavel
            safe_extra = {}
            for k, v in extra.items():
                try:
                    json.dumps(v)
                    safe_extra[k] = v
                except (TypeError, ValueError):
                    safe_extra[k] = str(v)
            if safe_extra:
                log_data["extra"] = safe_extra

        return json.dumps(log_data, default=str, ensure_ascii=False)


_DEV_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DEV_DATE_FORMAT = "%H:%M:%S"

_initialized = False


def setup_logging(
    level: str = "INFO",
    mode: str = "development",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """
    Configura logging global. Chamar UMA VEZ no main.py.

    Args:
        level: Nivel de log (DEBUG, INFO, WARNING, ERROR)
        mode: "development" (texto legivel) ou "production" (JSON)
        log_file: Caminho para arquivo de log (opcional, rotativo)
        max_bytes: Tamanho maximo do arquivo de log (default 10MB)
        backup_count: Numero de backups do log rotativo
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    if mode == "production":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_DEV_FORMAT, _DEV_DATE_FORMAT))

    root.addHandler(handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    # Reduz ruido de libs externas
    for noisy in ("urllib3", "asyncio", "websockets", "aiohttp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Retorna logger com nome do modulo. Uso: get_logger(__name__)."""
    return logging.getLogger(name)
