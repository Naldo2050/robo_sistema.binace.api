# ai_runner/__init__.py
"""
Pacote de execução de IA/LLM.

Contém: AIRunner, QwenClient, AIModelConfig, exceptions.
"""

from .ai_runner import AIModelConfig, AIRunner, QwenClient  # noqa: F401
from .exceptions import AIAnalysisError, ModelTimeoutError, RateLimitError  # noqa: F401

__all__ = [
    "AIAnalysisError",
    "AIModelConfig",
    "AIRunner",
    "ModelTimeoutError",
    "QwenClient",
    "RateLimitError",
]
