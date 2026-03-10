"""
Context Retriever - Fase 5.
Recupera contexto relevante para análise de bugs.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ContextRetriever:
    """Recupera contexto relevante para análise de bugs."""
    
    def __init__(self, vector_store):
        self.store = vector_store
    
    def get_context_for_error(
        self, 
        error_message: str,
        error_file: Optional[str] = None,
        max_chunks: int = 5,
        max_tokens: int = 8000,
    ) -> str:
        """
        Dado um erro, recupera o contexto relevante.
        Usado para enviar à IA junto com o pedido de correção.
        """
        # Buscar chunks relevantes
        chunks = self.store.search(
            query=error_message,
            top_k=max_chunks * 2,  # Busca mais para filtrar
            filter_file=error_file,
        )
        
        # Montar contexto respeitando limite de tokens
        context_parts = []
        total_tokens = 0
        
        for chunk in chunks:
            token_est = chunk["metadata"].get("token_estimate", 0)
            if total_tokens + token_est > max_tokens:
                break
            
            header = (
                f"\n--- {chunk['chunk_id']} "
                f"(linhas {chunk['metadata']['line_start']}-"
                f"{chunk['metadata']['line_end']}) ---\n"
            )
            context_parts.append(header + chunk["content"])
            total_tokens += token_est
        
        return "\n".join(context_parts)
    
    def get_related_files(
        self, file_path: str, top_k: int = 5
    ) -> list[str]:
        """Encontra arquivos relacionados a um dado arquivo."""
        # Buscar usando o conteúdo do arquivo como query
        chunks = self.store.search(
            query=f"import from {file_path}",
            top_k=top_k * 3,
        )
        
        related = set()
        for chunk in chunks:
            fp = chunk["metadata"]["file_path"]
            if fp != file_path:
                related.add(fp)
        
        return list(related)[:top_k]
