# common/exceptions.py
"""
Hierarquia unificada de excecoes do sistema.

Todas as excecoes do projeto derivam de BotBaseError, permitindo
captura generica com `except BotBaseError` quando necessario.

Os modulos existentes (ai_runner, orderbook_core, risk_management, etc.)
mantem suas proprias excecoes por compatibilidade. Este modulo fornece
a raiz da hierarquia e excecoes compartilhadas.

Hierarquia:
    BotBaseError
    ├── ConfigurationError
    ├── DataQualityError
    ├── ExternalServiceError
    │   ├── APIConnectionError
    │   ├── APITimeoutError
    │   └── RateLimitError
    └── PipelineError
"""


class BotBaseError(Exception):
    """Raiz de todas as excecoes do sistema."""
    pass


class ConfigurationError(BotBaseError):
    """Configuracao invalida ou ausente."""
    pass


class DataQualityError(BotBaseError):
    """Dados invalidos, corrompidos ou insuficientes."""
    pass


class ExternalServiceError(BotBaseError):
    """Falha em servico externo (Binance, Groq, FRED, yFinance)."""
    pass


class APIConnectionError(ExternalServiceError):
    """Falha de conexao com API externa."""
    pass


class APITimeoutError(ExternalServiceError):
    """Timeout em chamada de API externa."""
    pass


class RateLimitError(ExternalServiceError):
    """Rate limit excedido em API externa."""
    pass


class PipelineError(BotBaseError):
    """Falha no pipeline de processamento de dados."""
    pass
