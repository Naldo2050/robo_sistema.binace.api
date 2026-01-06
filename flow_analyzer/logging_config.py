# flow_analyzer/logging_config.py
"""
Configuração de logging estruturado para FlowAnalyzer.

Suporta:
- Logging JSON para ELK/Datadog
- Logging texto para desenvolvimento
- Rotação de logs
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """
    Formatter que produz logs em JSON.
    
    Campos incluídos:
    - timestamp: ISO8601
    - level: Nome do nível
    - logger: Nome do logger
    - message: Mensagem
    - extra: Campos adicionais
    
    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.setFormatter(JSONFormatter())
        >>> logger.addHandler(handler)
        >>> logger.info("Trade processed", extra={"cvd": 1.5})
        {"timestamp": "2024-01-01T12:00:00", "level": "INFO", ...}
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
        self._skip_fields = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'message', 'asctime',
        }
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Localização
        log_data['location'] = {
            'file': record.filename,
            'line': record.lineno,
            'function': record.funcName,
        }
        
        # Exceção
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Campos extras
        if self.include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in self._skip_fields:
                    try:
                        # Tenta serializar
                        json.dumps(value)
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)
            
            if extra:
                log_data['extra'] = extra
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class FlowAnalyzerLogger:
    """
    Logger configurado para FlowAnalyzer.
    
    Modos:
    - development: Texto colorido para terminal
    - production: JSON para sistemas de log
    """
    
    def __init__(
        self,
        name: str = "flow_analyzer",
        level: int = logging.INFO,
        mode: str = "development",
        log_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        if mode == "production":
            self._setup_production(log_file, max_bytes, backup_count)
        else:
            self._setup_development()
    
    def _setup_development(self) -> None:
        """Setup para desenvolvimento (texto)."""
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def _setup_production(
        self,
        log_file: Optional[str],
        max_bytes: int,
        backup_count: int,
    ) -> None:
        """Setup para produção (JSON)."""
        # Console JSON
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)
        
        # Arquivo com rotação
        if log_file:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Retorna o logger configurado."""
        return self.logger


class StructuredLogger:
    """
    Wrapper para logging estruturado.
    
    Facilita adicionar contexto a logs.
    
    Example:
        >>> log = StructuredLogger("flow_analyzer")
        >>> log.info("trade_processed", cvd=1.5, whale=True)
    """
    
    def __init__(self, name: str = "flow_analyzer"):
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """Adiciona contexto permanente."""
        new_logger = StructuredLogger(self._logger.name)
        new_logger._context = {**self._context, **kwargs}
        return new_logger
    
    def _log(self, level: int, event: str, **kwargs) -> None:
        """Log interno com contexto."""
        extra = {**self._context, **kwargs, 'event': event}
        self._logger.log(level, event, extra=extra)
    
    def debug(self, event: str, **kwargs) -> None:
        self._log(logging.DEBUG, event, **kwargs)
    
    def info(self, event: str, **kwargs) -> None:
        self._log(logging.INFO, event, **kwargs)
    
    def warning(self, event: str, **kwargs) -> None:
        self._log(logging.WARNING, event, **kwargs)
    
    def error(self, event: str, **kwargs) -> None:
        self._log(logging.ERROR, event, **kwargs)
    
    def critical(self, event: str, **kwargs) -> None:
        self._log(logging.CRITICAL, event, **kwargs)


def setup_logging(
    mode: str = "development",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configura logging para FlowAnalyzer.
    
    Args:
        mode: "development" ou "production"
        level: Nível de log
        log_file: Arquivo de log (opcional)
        
    Returns:
        Logger configurado
    """
    logger_config = FlowAnalyzerLogger(
        mode=mode,
        level=level,
        log_file=log_file,
    )
    return logger_config.get_logger()