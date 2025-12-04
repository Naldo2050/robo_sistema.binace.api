# data_pipeline/logging_utils.py
from __future__ import annotations

import logging
from typing import Any, Dict


class PipelineLogger:
    """
    Sistema de logging com separação de responsabilidades.

    Permite controle granular de níveis de log:
    - pipeline.validation -> DEBUG para desenvolvimento, detalhes de validação
    - pipeline.runtime -> INFO para produção, operações principais
    - pipeline.performance -> Métricas de performance e otimizações
    - pipeline.adaptive -> Sistema adaptativo de thresholds
    - pipeline.ml -> Machine Learning features e predições

    Exemplo de uso:
        logger = PipelineLogger("BTCUSDT")
        logger.validation_debug("Validando trades", count=100)
        logger.runtime_info("Pipeline iniciado")
        logger.performance_info("Cache hit", rate=95.5)
    """

    def __init__(self, symbol: str = "UNKNOWN") -> None:
        self.symbol = symbol

        # Loggers especializados
        self.validation = logging.getLogger(f'pipeline.validation.{symbol}')
        self.runtime = logging.getLogger(f'pipeline.runtime.{symbol}')
        self.performance = logging.getLogger(f'pipeline.performance.{symbol}')
        self.adaptive = logging.getLogger(f'pipeline.adaptive.{symbol}')
        self.ml = logging.getLogger(f'pipeline.ml.{symbol}')

        # Contexto compartilhado para enriquecer logs
        self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs: Any) -> None:
        """
        Define contexto adicional que será adicionado a todos os logs.

        Exemplo:
            logger.set_context(session_id="abc123", batch=5)
            logger.runtime_info("Processando")  # Inclui session_id e batch
        """
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Limpa o contexto compartilhado."""
        self._context.clear()

    def _format_message(self, msg: str) -> str:
        """Formata mensagem incluindo contexto."""
        if self._context:
            ctx_str = " | ".join(f"{k}={v}" for k, v in self._context.items())
            return f"{msg} | {ctx_str}"
        return msg

    # Métodos de conveniência para validação
    def validation_debug(self, msg: str, **kwargs: Any) -> None:
        """Log de debug de validação (detalhes técnicos)."""
        self.set_context(**kwargs)
        self.validation.debug(self._format_message(msg))
        self.clear_context()

    def validation_info(self, msg: str, **kwargs: Any) -> None:
        """Log de info de validação (confirmações)."""
        self.set_context(**kwargs)
        self.validation.info(self._format_message(msg))
        self.clear_context()

    def validation_warning(self, msg: str, **kwargs: Any) -> None:
        """Log de warning de validação (dados suspeitos)."""
        self.set_context(**kwargs)
        self.validation.warning(self._format_message(msg))
        self.clear_context()

    def validation_error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log de erro de validação."""
        self.set_context(**kwargs)
        self.validation.error(self._format_message(msg), exc_info=exc_info)
        self.clear_context()

    # Métodos de conveniência para runtime
    def runtime_debug(self, msg: str, **kwargs: Any) -> None:
        """Log de debug de runtime."""
        self.set_context(**kwargs)
        self.runtime.debug(self._format_message(msg))
        self.clear_context()

    def runtime_info(self, msg: str, **kwargs: Any) -> None:
        """Log de info de runtime (operações normais)."""
        self.set_context(**kwargs)
        self.runtime.info(self._format_message(msg))
        self.clear_context()

    def runtime_warning(self, msg: str, **kwargs: Any) -> None:
        """Log de warning de runtime (situações anormais)."""
        self.set_context(**kwargs)
        self.runtime.warning(self._format_message(msg))
        self.clear_context()

    def runtime_error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log de erro de runtime (falhas críticas)."""
        self.set_context(**kwargs)
        self.runtime.error(self._format_message(msg), exc_info=exc_info)
        self.clear_context()

    # Métodos de conveniência para performance
    def performance_debug(self, msg: str, **kwargs: Any) -> None:
        """Log de debug de performance."""
        self.set_context(**kwargs)
        self.performance.debug(self._format_message(msg))
        self.clear_context()

    def performance_info(self, msg: str, **kwargs: Any) -> None:
        """Log de info de performance (métricas)."""
        self.set_context(**kwargs)
        self.performance.info(self._format_message(msg))
        self.clear_context()

    def performance_warning(self, msg: str, **kwargs: Any) -> None:
        """Log de warning de performance (lentidão)."""
        self.set_context(**kwargs)
        self.performance.warning(self._format_message(msg))
        self.clear_context()

    # Métodos de conveniência para adaptativo
    def adaptive_debug(self, msg: str, **kwargs: Any) -> None:
        """Log de debug do sistema adaptativo."""
        self.set_context(**kwargs)
        self.adaptive.debug(self._format_message(msg))
        self.clear_context()

    def adaptive_info(self, msg: str, **kwargs: Any) -> None:
        """Log de info do sistema adaptativo (ajustes)."""
        self.set_context(**kwargs)
        self.adaptive.info(self._format_message(msg))
        self.clear_context()

    def adaptive_warning(self, msg: str, **kwargs: Any) -> None:
        """Log de warning do sistema adaptativo."""
        self.set_context(**kwargs)
        self.adaptive.warning(self._format_message(msg))
        self.clear_context()

    # Métodos de conveniência para ML
    def ml_debug(self, msg: str, **kwargs: Any) -> None:
        """Log de debug de ML features."""
        self.set_context(**kwargs)
        self.ml.debug(self._format_message(msg))
        self.clear_context()

    def ml_info(self, msg: str, **kwargs: Any) -> None:
        """Log de info de ML features (geração)."""
        self.set_context(**kwargs)
        self.ml.info(self._format_message(msg))
        self.clear_context()

    def ml_warning(self, msg: str, **kwargs: Any) -> None:
        """Log de warning de ML (features ausentes)."""
        self.set_context(**kwargs)
        self.ml.warning(self._format_message(msg))
        self.clear_context()

    def ml_error(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log de erro de ML."""
        self.set_context(**kwargs)
        self.ml.error(self._format_message(msg), exc_info=exc_info)
        self.clear_context()


def setup_pipeline_logging(
    validation_level: int = logging.DEBUG,
    runtime_level: int = logging.INFO,
    performance_level: int = logging.INFO,
    adaptive_level: int = logging.INFO,
    ml_level: int = logging.INFO
) -> None:
    """
    Configura níveis de log para cada componente do pipeline.

    Args:
        validation_level: Nível para validação (padrão: DEBUG)
        runtime_level: Nível para runtime (padrão: INFO)
        performance_level: Nível para performance (padrão: INFO)
        adaptive_level: Nível para sistema adaptativo (padrão: INFO)
        ml_level: Nível para ML features (padrão: INFO)

    Exemplo de uso em produção:
        setup_pipeline_logging(
            validation_level=logging.INFO,      # Menos verbose
            runtime_level=logging.INFO,
            performance_level=logging.WARNING,  # Só alertas
            adaptive_level=logging.INFO,
            ml_level=logging.WARNING
        )

    Exemplo de uso em desenvolvimento:
        setup_pipeline_logging(
            validation_level=logging.DEBUG,     # Tudo
            runtime_level=logging.DEBUG,
            performance_level=logging.DEBUG,
            adaptive_level=logging.DEBUG,
            ml_level=logging.DEBUG
        )
    """
    logging.getLogger('pipeline.validation').setLevel(validation_level)
    logging.getLogger('pipeline.runtime').setLevel(runtime_level)
    logging.getLogger('pipeline.performance').setLevel(performance_level)
    logging.getLogger('pipeline.adaptive').setLevel(adaptive_level)
    logging.getLogger('pipeline.ml').setLevel(ml_level)