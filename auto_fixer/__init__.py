"""
Auto-Fixer System.
Sistema de correção automática de código com IA.
"""

__version__ = "1.0.0"

# Imports seguros (não dependem de pacotes opcionais)
from .phase1_scanner import CodebaseScanner
from .phase2_extractor import ASTExtractor
from .phase3_chunker import ChunkEngine
from .phase4_index import CodeIndex
from .phase6_analyzers import (
    BaseAnalyzer, AsyncAnalyzer, APIAnalyzer, 
    WebSocketAnalyzer, ImportAnalyzer
)
from .phase7_patcher import (
    PatchGenerator, Patch, PatchValidator,
    PatchApplier, RollbackManager
)
from .phase8_reporter import ReportGenerator

# Imports opcionais (dependem de chromadb, que pode não estar instalado)
# Usar import lazy para evitar crash com pydantic-core no Python 3.12
def get_vector_store(*args, **kwargs):
    """Factory para CodeVectorStore (import lazy)."""
    from .phase5_rag import CodeVectorStore
    return CodeVectorStore(*args, **kwargs)

def get_context_retriever(*args, **kwargs):
    """Factory para ContextRetriever (import lazy)."""
    try:
        from .phase5_rag import ContextRetriever
        return ContextRetriever(*args, **kwargs)
    except ImportError:
        return None

# Para backward compatibility - acesso direto aos módulos (pode causar crash)
# Recomendado: usar get_vector_store() e get_context_retriever()
_Phase5LazyLoader = None

def __getattr__(name):
    """Loader lazy para CodeVectorStore e ContextRetriever."""
    if name == "CodeVectorStore":
        from .phase5_rag import CodeVectorStore
        return CodeVectorStore
    elif name == "ContextRetriever":
        from .phase5_rag import ContextRetriever
        return ContextRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Phase 1
    "CodebaseScanner",
    # Phase 2
    "ASTExtractor",
    # Phase 3
    "ChunkEngine",
    # Phase 4
    "CodeIndex",
    # Phase 5 (lazy - usar get_vector_store() ou get_context_retriever())
    "get_vector_store",
    "get_context_retriever",
    # Phase 6
    "BaseAnalyzer",
    "AsyncAnalyzer",
    "APIAnalyzer",
    "WebSocketAnalyzer",
    "ImportAnalyzer",
    # Phase 7
    "PatchGenerator",
    "Patch",
    "PatchValidator",
    "PatchApplier",
    "RollbackManager",
    # Phase 8
    "ReportGenerator",
]
