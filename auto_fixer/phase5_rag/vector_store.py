"""
Vector Store - Fase 5.
Armazena e busca embeddings de código.
NOTA: Usa chromadb local (pip install chromadb)
Se preferir FAISS: pip install faiss-cpu
"""

import json
import logging
from pathlib import Path
from typing import Optional, Any, cast

logger = logging.getLogger(__name__)

try:
    import chromadb  # type: ignore[import-untyped]
    from chromadb.config import Settings  # type: ignore[import-untyped]
    HAS_CHROMA = True
except (ImportError, Exception) as _chroma_err:
    chromadb = None  # type: ignore[assignment]
    HAS_CHROMA = False
    logger.warning(f"chromadb não disponível: {_chroma_err}")


class CodeVectorStore:
    """Store vetorial para chunks de código."""
    
    def __init__(
        self,
        collection_name: str = "codebase",
        persist_dir: str = "auto_fixer/output/vectordb"
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_CHROMA:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.collection = None
    
    def add_chunks(self, chunks: list[dict]):
        """Adiciona chunks ao vector store."""
        if not self.collection:
            logger.error("ChromaDB não disponível")
            return
        
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            content = chunk.get("content", "")
            
            if not content.strip():
                continue
            
            ids.append(chunk_id)
            documents.append(content)
            metadatas.append({
                "file_path": chunk["file_path"],
                "chunk_index": chunk["chunk_index"],
                "line_start": chunk["line_start"],
                "line_end": chunk["line_end"],
                "token_estimate": chunk.get("token_estimate", 0),
                "classes": json.dumps(
                    chunk.get("contains_classes", [])
                ),
                "functions": json.dumps(
                    chunk.get("contains_functions", [])
                ),
            })
        
        # ChromaDB aceita batch de até 5461
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )
        
        logger.info(f"Adicionados {len(ids)} chunks ao vector store")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_file: Optional[str] = None
    ) -> list[dict]:
        """Busca chunks relevantes para uma query."""
        if not self.collection:
            return []
        
        # Build where filter with proper typing
        where: Optional[dict[str, Any]] = None
        if filter_file:
            where = {"file_path": filter_file}
        
        # Query with type ignore for chromadb Where type
        results: dict[str, Any] = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,  # type: ignore[arg-type]
        )
        
        matches: list[dict[str, Any]] = []
        
        # Safely extract results with proper None checking
        ids_list = results.get("ids")
        if not ids_list or not ids_list[0]:
            return matches
        
        ids = cast(list[str], ids_list[0])
        docs = cast(list[list[str]], results.get("documents")) or []
        metas = cast(list[list[dict]], results.get("metadatas")) or []
        dists = cast(list[list[float]], results.get("distances"))
        
        for i, chunk_id in enumerate(ids):
            matches.append({
                "chunk_id": chunk_id,
                "content": docs[0][i] if docs and docs[0] else "",
                "metadata": metas[0][i] if metas and metas[0] else {},
                "distance": dists[0][i] if dists and dists[0] else None,
            })
        
        return matches
    
    def get_stats(self) -> dict:
        """Estatísticas do store."""
        if not self.collection:
            return {"status": "unavailable"}
        return {
            "total_chunks": self.collection.count(),
            "persist_dir": str(self.persist_dir),
        }