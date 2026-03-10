"""
Embeddings Generator - Fase 5.
Gera embeddings dos chunks usando API externa.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Configuração da API de embeddings (separada da API de análise)
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_API_URL = "https://api.openai.com/v1/embeddings"

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    HAS_AIOHTTP = False


class EmbeddingsGenerator:
    """Gera embeddings para chunks de código."""
    
    def __init__(
        self,
        api_key: str | None = None,
        api_url: str = DEFAULT_API_URL,
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 100,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.batch_size = batch_size
    
    async def generate_embeddings(
        self, 
        texts: list[str]
    ) -> list[list[float]]:
        """Gera embeddings para uma lista de textos."""
        if not HAS_AIOHTTP:
            logger.error("aiohttp não instalado")
            return []
        
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._call_api(batch)
            embeddings.extend(batch_embeddings)
            
            logger.info(
                f"Embedding batch {i//self.batch_size + 1}: "
                f"{len(batch)} textos"
            )
        
        return embeddings
    
    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Chama a API de embeddings."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "input": texts,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, 
                    json=payload, 
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [
                            item["embedding"] 
                            for item in data.get("data", [])
                        ]
                    else:
                        logger.error(
                            f"API error: {response.status} - "
                            f"{await response.text()}"
                        )
                        return []
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings: {e}")
            return []
    
    def generate_sync(
        self, 
        texts: list[str]
    ) -> list[list[float]]:
        """Versão síncrona (para uso em scripts)."""
        import asyncio
        return asyncio.run(self.generate_embeddings(texts))


class LocalEmbeddingsGenerator:
    """Gerador de embeddings local (sem API externa)."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo local."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Modelo local carregado: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers não instalado. "
                "Use: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
    
    def generate(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings usando modelo local."""
        if not self.model:
            logger.error("Modelo não disponível")
            return []
        
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings: {e}")
            return []


def get_embeddings_generator(
    use_local: bool = False,
    **kwargs
) -> EmbeddingsGenerator | LocalEmbeddingsGenerator:
    """Factory function para obter o gerador de embeddings."""
    if use_local:
        return LocalEmbeddingsGenerator(**kwargs)
    return EmbeddingsGenerator(**kwargs)


if __name__ == "__main__":
    # Teste rápido
    logging.basicConfig(level=logging.INFO)
    
    test_texts = [
        "def hello_world(): print('Hello, world!')",
        "class MyClass: pass",
        "import asyncio",
    ]
    
    # Teste com modelo local
    local_gen = LocalEmbeddingsGenerator()
    embeddings = local_gen.generate(test_texts)
    
    if embeddings:
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")