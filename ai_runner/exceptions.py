# ai_runner/exceptions.py


class AIAnalysisError(Exception):
    """Raised when AI analysis fails."""
    pass


class RateLimitError(AIAnalysisError):
    """Raised when rate limit is exceeded."""
    pass


class ModelTimeoutError(AIAnalysisError):
    """Raised when model operation times out."""
    pass


class AIRunnerError(Exception):
    """Base exception for AI runner operations."""
    pass