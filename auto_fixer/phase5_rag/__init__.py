"""
Phase 5 RAG — Retrieval Augmented Generation para código.
Imports são lazy para evitar crash quando chromadb não está instalado
ou tem conflito de versão.
"""

# NÃO importar nada aqui — usar import direto quando necessário
# from .vector_store import CodeVectorStore  ← REMOVIDO
# from .context_retriever import ContextRetriever  ← REMOVIDO

from .embeddings import (
    EmbeddingsGenerator,
    LocalEmbeddingsGenerator,
    get_embeddings_generator,
)

def get_vector_store(*args, **kwargs):
    from .vector_store import CodeVectorStore
    return CodeVectorStore(*args, **kwargs)

def get_context_retriever(*args, **kwargs):
    from .context_retriever import ContextRetriever
    return ContextRetriever(*args, **kwargs)

def __getattr__(name):
    """Loader lazy para CodeVectorStore e ContextRetriever."""
    if name == "CodeVectorStore":
        from .vector_store import CodeVectorStore
        return CodeVectorStore
    elif name == "ContextRetriever":
        from .context_retriever import ContextRetriever
        return ContextRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EmbeddingsGenerator",
    "LocalEmbeddingsGenerator", 
    "get_embeddings_generator",
    "get_vector_store",
    "get_context_retriever",
]
