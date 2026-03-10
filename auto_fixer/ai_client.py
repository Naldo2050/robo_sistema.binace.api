"""
Cliente de IA para o Auto-Fixer.
API SEPARADA da usada para análise de mercado.
"""

import os
import json
import logging
from typing import Optional, Any
from pathlib import Path
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class AutoFixerAIClient:
    """
    Cliente de IA dedicado ao auto-fixer.
    
    Configuração via:
    - Variável de ambiente: AUTOFIXER_AI_API_KEY, AUTOFIXER_AI_BASE_URL
    - Arquivo: auto_fixer/config.json
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "groq",  # "groq", "openai", "local"
    ):
        config = self._load_ai_config()
        
        self.provider = provider or config.get("provider", "groq")
        self.api_key = (
            api_key 
            or os.getenv("AUTOFIXER_AI_API_KEY") 
            or config.get("api_key", "")
        )
        self.base_url = (
            base_url 
            or os.getenv("AUTOFIXER_AI_BASE_URL") 
            or config.get("base_url", self._default_url())
        )
        self.model = (
            model 
            or config.get("model", self._default_model())
        )
        
        # Carregar configurações adicionais
        full_config = self._load_config()
        self.scanner_config = full_config.get("scanner", {})
        self.chunker_config = full_config.get("chunker", {})
        self.analyzers_config = full_config.get("analyzers", {})
        self.patcher_config = full_config.get("patcher", {})
        
        self._client = None
        self._init_client()
    
    def _default_url(self) -> str:
        urls = {
            "groq": "https://api.groq.com/openai/v1",
            "openai": "https://api.openai.com/v1",
            "local": "http://localhost:11434/v1",
        }
        return urls.get(self.provider, urls["groq"])
    
    def _default_model(self) -> str:
        models = {
            "groq": "llama-3.1-70b-versatile",
            "openai": "gpt-4o-mini",
            "local": "codellama",
        }
        return models.get(self.provider, "llama-3.1-70b-versatile")
    
    def _load_config(self) -> dict:
        config_path = Path("auto_fixer/config.json")
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_ai_config(self) -> dict:
        """Carrega configuração da seção AI."""
        config = self._load_config()
        return config.get("ai", {})
    
    def _init_client(self):
        """Inicializa o cliente HTTP."""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            logger.info(
                f"AI Client inicializado: {self.provider} / {self.model}"
            )
        except ImportError:
            logger.warning(
                "openai não instalado. pip install openai"
            )
    
    def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> str:
        """Envia mensagem para a IA e retorna resposta."""
        if not self._client:
            logger.error("Cliente de IA não inicializado")
            return ""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content or ""
            
            # Log de uso
            usage = response.usage
            if usage:
                logger.debug(
                    f"Tokens: {usage.prompt_tokens} in, "
                    f"{usage.completion_tokens} out"
                )
            
            return content
            
        except Exception as e:
            logger.error(f"Erro na API de IA: {e}")
            return ""
    
    def is_available(self) -> bool:
        """Verifica se a API está disponível."""
        try:
            response = self.chat(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return len(response) > 0
        except Exception:
            return False
