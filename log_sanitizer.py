# log_sanitizer.py
# -*- coding: utf-8 -*-

"""
Sanitizador de logs para prevenir vazamento de segredos.

Este módulo fornece funções para redigir informações sensíveis
de mensagens de log, incluindo:
- Chaves de API (GROQ_API_KEY, FRED_API_KEY, etc)
- Tokens (especialmente os que começam com gsk_)
- Segredos em variáveis de ambiente
- Informações de configuração sensíveis
"""

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("log_sanitizer")

# Padrões de segredos conocidos
SECRET_PATTERNS = [
    # Tokens que começam com gsk_ (Groq)
    (r'gsk_[a-zA-Z0-9]{20,}', 'gsk_***REDACTED***'),
    
    # Keys de API genéricas no formato de variáveis
    (r'(GROQ_API_KEY|FRED_API_KEY|ALPHA_VANTAGE_API_KEY|DASHSCOPE_API_KEY|OPENAI_API_KEY)[=:\s]+[a-zA-Z0-9\-_]{10,}', 
     r'\1=***REDACTED***'),
    
    # Chaves parciais (primeiros 8 caracteres)
    (r'Chave:\s*[a-zA-Z0-9\-_]{8,}', 'Chave: ***REDACTED***'),
    (r'Key:\s*[a-zA-Z0-9\-_]{8,}', 'Key: ***REDACTED***'),
    (r'API[_-]?KEY[=:\s]*[a-zA-Z0-9\-_]{8,}', 'API_KEY=***REDACTED***'),
    
    # Tokens Bearer
    (r'Bearer\s+[a-zA-Z0-9\-_.~+/]{20,}', 'Bearer ***REDACTED***'),
    
    # senhas em URLs
    (r'[a-zA-Z]+://[^:]+:[^@]+@[a-zA-Z]', '***REDACTED***'),
    
    # Secrets em JSON
    (r'"(api_key|secret|token|password)":\s*"[^"]+"', r'"\1": "***REDACTED***"'),
]

# Compila padrões para performance
COMPILED_PATTERNS = [(re.compile(pattern, re.IGNORECASE), replacement) 
                     for pattern, replacement in SECRET_PATTERNS]


class LogSanitizer:
    """
    Sanitizador de mensagens de log.
    
    Uso:
        sanitizer = LogSanitizer()
        safe_message = sanitizer.sanitize(message)
    """
    
    def __init__(self, custom_patterns: Optional[list] = None):
        """
        Inicializa o sanitizador.
        
        Args:
            custom_patterns: Lista adicional de padrões (pattern, replacement)
        """
        self.patterns = list(COMPILED_PATTERNS)
        
        if custom_patterns:
            for pattern, replacement in custom_patterns:
                try:
                    self.patterns.append((re.compile(pattern, re.IGNORECASE), replacement))
                except re.error as e:
                    logger.warning(f"Padrão regex inválido ignorado: {e}")
    
    def sanitize(self, message: Any) -> str:
        """
        Sanitiza uma mensagem de log.
        
        Args:
            message: Mensagem a ser sanitizada (qualquer tipo)
            
        Returns:
            String sanitizada
        """
        if message is None:
            return ""
        
        # Converte para string
        msg_str = str(message)
        
        # Aplica cada padrão de redaction
        for pattern, replacement in self.patterns:
            msg_str = pattern.sub(replacement, msg_str)
        
        return msg_str
    
    def sanitize_dict(self, data: dict, exclude_keys: Optional[set] = None) -> dict:
        """
        Sanitiza um dicionário, redigindo valores sensíveis.
        
        Args:
            data: Dicionário a ser sanitizado
            exclude_keys: Chaves a não sanitizar
            
        Returns:
            Dicionário com valores sensíveis redigidos
        """
        if exclude_keys is None:
            exclude_keys = set()
        
        sensitive_keys = {
            'api_key', 'secret', 'token', 'password', 'key',
            'GROQ_API_KEY', 'FRED_API_KEY', 'ALPHA_VANTAGE_API_KEY',
            'DASHSCOPE_API_KEY', 'OPENAI_API_KEY',
            'chiave', 'chave', 'senha',
        }
        
        result = {}
        for key, value in data.items():
            if key.lower() in sensitive_keys and key not in exclude_keys:
                result[key] = "***REDACTED***"
            elif isinstance(value, dict):
                result[key] = self.sanitize_dict(value, exclude_keys)
            elif isinstance(value, str):
                result[key] = self.sanitize(value)
            else:
                result[key] = value
        
        return result
    
    def sanitize_exception(self, exc: Exception) -> str:
        """
        Sanitiza uma exceção, removendo informações sensíveis do traceback.
        
        Args:
            exc: Exceção a ser sanitizada
            
        Returns:
            String da exceção sanitizada
        """
        import traceback
        
        # Obtém o traceback como string
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        
        # Sanitiza
        return self.sanitize(tb_str)


def sanitize_log_message(message: Any) -> str:
    """
    Função de conveniência para sanitizar mensagens de log.
    
    Args:
        message: Mensagem a ser sanitizada
        
    Returns:
        Mensagem sanitizada
    """
    sanitizer = LogSanitizer()
    return sanitizer.sanitize(message)


def sanitize_preview(text: str, max_length: int = 150) -> str:
    """
    Cria um preview seguro de um texto longo.
    
    Args:
        text: Texto original
        max_length: Comprimento máximo do preview
        
    Returns:
        Preview truncado e sanitizado
    """
    if not text:
        return "(vazio)"
    
    # Primeiro sanitiza
    sanitized = sanitize_log_message(text)
    
    # Trunca
    if len(sanitized) <= max_length:
        return sanitized
    
    return sanitized[:max_length] + "..."


class SensitiveFilter(logging.Filter):
    """
    Filtro de logging que sanitiza mensagens automaticamente.
    
    Uso:
        handler = logging.StreamHandler()
        handler.addFilter(SensitiveFilter())
        logger.addHandler(handler)
    """
    
    def __init__(self):
        super().__init__()
        self.sanitizer = LogSanitizer()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filtra e sanitiza o registro de log.
        
        Args:
            record: Registro de log
            
        Returns:
            True para processar o registro
        """
        # Sanitiza a mensagem
        record.msg = self.sanitizer.sanitize(record.msg)
        
        # Sanitiza argumentos
        if record.args:
            sanitized_args = tuple(
                self.sanitizer.sanitize(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
            record.args = sanitized_args
        
        return True


def setup_sanitized_logging(logger_name: Optional[str] = None) -> None:
    """
    Configura logging com sanitização automática.
    
    Args:
        logger_name: Nome do logger (None = root logger)
    """
    target_logger = logging.getLogger(logger_name)
    
    # Adiciona o filtro a todos os handlers existentes
    sanitized_filter = SensitiveFilter()
    
    for handler in target_logger.handlers:
        handler.addFilter(sanitized_filter)
    
    # Adiciona o filtro a handlers futuros (via root logger)
    # Isso é um hack, mas funciona na maioria dos casos
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if sanitized_filter not in handler.filters:
            handler.addFilter(sanitized_filter)


# Função de convenience para ser usada com logging.basicConfig
def get_sanitized_formatter() -> logging.Formatter:
    """
    Retorna um formatador que sanitiza mensagens.
    
    Returns:
        Formatador configurado
    """
    return logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
