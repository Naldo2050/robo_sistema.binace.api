# ai_runner/exceptions.py
# Mantido para compatibilidade - exceções movidas para common/exceptions.py
from common.exceptions import (  # noqa: F401
    AIAnalysisError,
    AIRateLimitError as RateLimitError,
    ModelTimeoutError,
    AIRunnerError,
)
