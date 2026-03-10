"""
Chunk Engine - Fase 3.
Divide arquivos grandes em pedaços processáveis.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_LINES = 400
APPROX_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    chunk_index: int
    total_chunks: int
    line_start: int
    line_end: int
    content: str
    token_estimate: int
    context_imports: str  # Imports do arquivo original
    contains_classes: list[str] = field(default_factory=list)
    contains_functions: list[str] = field(default_factory=list)


@dataclass
class ChunkResult:
    total_files_chunked: int
    total_chunks: int
    total_tokens_estimate: int
    chunks: list[dict] = field(default_factory=list)


class ChunkEngine:
    """Divide arquivos grandes em chunks inteligentes."""
    
    def __init__( 
        self, 
        max_lines: int = DEFAULT_MAX_LINES,
        output_dir: str = "auto_fixer/output"
    ):
        self.max_lines = max_lines
        self.output_dir = Path(output_dir)
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
    
    def process_all(self) -> ChunkResult:
        """Processa todos os arquivos usando o structure_map."""
        structure_path = self.output_dir / "structure_map.json"
        scan_path = self.output_dir / "scan_result.json"
        
        with open(structure_path, encoding="utf-8") as f:
            structure = json.load(f)
        with open(scan_path, encoding="utf-8") as f:
            scan = json.load(f)
        
        # Mapa de linhas por arquivo
        line_map = {
            fi["relative_path"]: fi["line_count"] 
            for fi in scan["files"]
        }
        path_map = {
            fi["relative_path"]: fi["path"] 
            for fi in scan["files"]
        }
        
        all_chunks: list[Chunk] = []
        files_chunked = 0
        
        for file_struct in structure["structures"]:
            rel_path = file_struct["relative_path"]
            line_count = line_map.get(rel_path, 0)
            
            if line_count <= self.max_lines:
                # Arquivo pequeno: um único chunk
                full_path = path_map.get(rel_path)
                if full_path and Path(full_path).exists():
                    content = Path(full_path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                    chunk = Chunk(
                        chunk_id=f"{rel_path}::0",
                        file_path=rel_path,
                        chunk_index=0,
                        total_chunks=1,
                        line_start=1,
                        line_end=line_count,
                        content=content,
                        token_estimate=len(content) // APPROX_CHARS_PER_TOKEN,
                        context_imports="",
                    )
                    all_chunks.append(chunk)
            else:
                # Arquivo grande: dividir
                full_path = path_map.get(rel_path)
                if full_path and Path(full_path).exists():
                    chunks = self._chunk_file(
                        full_path, rel_path, file_struct
                    )
                    all_chunks.extend(chunks)
                    files_chunked += 1
        
        result = ChunkResult(
            total_files_chunked=files_chunked,
            total_chunks=len(all_chunks),
            total_tokens_estimate=sum(c.token_estimate for c in all_chunks),
            chunks=[asdict(c) for c in all_chunks],
        )
        
        # Salvar índice de chunks (sem conteúdo completo)
        index = {
            "total_chunks": result.total_chunks,
            "total_tokens": result.total_tokens_estimate,
            "files_chunked": result.total_files_chunked,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "file_path": c.file_path,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "token_estimate": c.token_estimate,
                    "contains_classes": c.contains_classes,
                    "contains_functions": c.contains_functions,
                }
                for c in all_chunks
            ]
        }
        
        with open(self.output_dir / "chunk_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        
        # Salvar cada chunk como arquivo separado
        for chunk in all_chunks:
            # Tratar separadores de path do Windows e Linux
            safe_name = chunk.chunk_id
            safe_name = safe_name.replace(os.sep, "__")
            safe_name = chunk.chunk_id.replace(os.sep, "__").replace("/", "__").replace("\\", "__").replace("::", "_").replace(".", "_")
            safe_name = safe_name.replace("\\", "__")
            safe_name = safe_name.replace("::", "_")
            safe_name = safe_name.replace(".", "_")
            chunk_file = self.chunks_dir / f"{safe_name}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(asdict(chunk), f, indent=2, ensure_ascii=False)
        
        logger.info(
            f"Chunking completo: {result.total_chunks} chunks, "
            f"~{result.total_tokens_estimate:,} tokens"
        )
        return result
    
    def _chunk_file(
        self, filepath: str, rel_path: str, structure: dict
    ) -> list[Chunk]:
        """Divide um arquivo grande em chunks inteligentes."""
        lines = Path(filepath).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines(keepends=True)
        
        # Extrair imports (primeiras N linhas até primeira classe/função)
        import_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) or \
               stripped == "" or stripped.startswith("#"):
                import_lines.append(line)
            else:
                break
        imports_context = "".join(import_lines)
        
        # Identificar limites naturais (classes e funções top-level)
        boundaries = self._find_boundaries(structure)
        
        # Criar chunks respeitando limites
        chunks = []
        current_start = 0
        chunk_index = 0
        
        while current_start < len(lines):
            # Encontrar o melhor ponto de corte
            ideal_end = current_start + self.max_lines
            
            if ideal_end >= len(lines):
                actual_end = len(lines)
            else:
                # Procurar o limite natural mais próximo
                actual_end = self._find_nearest_boundary(
                    boundaries, ideal_end, current_start
                )
            
            chunk_content = "".join(lines[current_start:actual_end])
            
            # Detectar o que está neste chunk
            classes_in_chunk = []
            functions_in_chunk = []
            for cls in structure.get("classes", []):
                if cls["line_start"] >= current_start + 1 and \
                   cls["line_start"] <= actual_end:
                    classes_in_chunk.append(cls["name"])
            for func in structure.get("functions", []):
                if func["line_start"] >= current_start + 1 and \
                   func["line_start"] <= actual_end:
                    functions_in_chunk.append(func["name"])
            
            chunk = Chunk(
                chunk_id=f"{rel_path}::{chunk_index}",
                file_path=rel_path,
                chunk_index=chunk_index,
                total_chunks=0,  # Atualizado depois
                line_start=current_start + 1,
                line_end=actual_end,
                content=chunk_content,
                token_estimate=len(chunk_content) // APPROX_CHARS_PER_TOKEN,
                context_imports=imports_context,
                contains_classes=classes_in_chunk,
                contains_functions=functions_in_chunk,
            )
            chunks.append(chunk)
            
            current_start = actual_end
            chunk_index += 1
        
        # Atualizar total_chunks
        for c in chunks:
            c.total_chunks = len(chunks)
        
        return chunks
    
    def _find_boundaries(self, structure: dict) -> list[int]:
        """Encontra limites naturais (início de classes/funções)."""
        boundaries = []
        for cls in structure.get("classes", []):
            boundaries.append(cls["line_start"])
        for func in structure.get("functions", []):
            boundaries.append(func["line_start"])
        return sorted(set(boundaries))
    
    def _find_nearest_boundary(
        self, boundaries: list[int], target: int, minimum: int
    ) -> int:
        """Encontra o limite natural mais próximo do target."""
        best = target
        best_distance = float("inf")
        
        for b in boundaries:
            if b > minimum and abs(b - target) < best_distance:
                best = b
                best_distance = abs(b - target)
        
        # Se não encontrou limite próximo, usa o target
        if best_distance > 50:
            return target
        return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ChunkEngine()
    result = engine.process_all()
    print(f"\nChunking:")
    print(f"  Arquivos divididos: {result.total_files_chunked}")
    print(f"  Total chunks: {result.total_chunks}")
    print(f"  Tokens estimados: {result.total_tokens_estimate:,}")