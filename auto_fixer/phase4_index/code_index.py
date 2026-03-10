"""
Code Index - Fase 4.
Índice unificado e buscável da codebase.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexEntry:
    name: str
    entry_type: str  # "class", "function", "module"
    file_path: str
    chunk_id: Optional[str]
    line_start: int
    line_end: int
    metadata: dict = field(default_factory=dict)


class CodeIndex:
    """Índice principal da codebase."""
    
    def __init__(self, output_dir: str = "auto_fixer/output"):
        self.output_dir = Path(output_dir)
        
        # Índices
        self.function_index: dict[str, list[IndexEntry]] = {}
        self.class_index: dict[str, list[IndexEntry]] = {}
        self.module_index: dict[str, list[str]] = {}  # módulo -> [importado por]
        self.file_index: dict[str, dict] = {}
        self.dependency_graph: dict[str, list[str]] = {}
        
    def build(self):
        """Constrói o índice completo."""
        structure = self._load_json("structure_map.json")
        chunk_index = self._load_json("chunk_index.json")
        scan = self._load_json("scan_result.json")
        
        # Construir mapa de chunks
        chunk_map: dict[str, list[dict]] = {}
        if chunk_index:
            for chunk in chunk_index.get("chunks", []):
                fp = chunk["file_path"]
                if fp not in chunk_map:
                    chunk_map[fp] = []
                chunk_map[fp].append(chunk)
        
        # Indexar cada arquivo
        for file_struct in structure.get("structures", []):
            rel_path = file_struct["relative_path"]
            
            # Indexar classes
            for cls in file_struct.get("classes", []):
                chunk_id = self._find_chunk(
                    chunk_map, rel_path, cls["line_start"]
                )
                entry = IndexEntry(
                    name=cls["name"],
                    entry_type="class",
                    file_path=rel_path,
                    chunk_id=chunk_id,
                    line_start=cls["line_start"],
                    line_end=cls["line_end"],
                    metadata={
                        "methods": [m["name"] for m in cls.get("methods", [])],
                        "bases": cls.get("bases", []),
                    },
                )
                if cls["name"] not in self.class_index:
                    self.class_index[cls["name"]] = []
                self.class_index[cls["name"]].append(entry)
            
            # Indexar funções top-level
            for func in file_struct.get("functions", []):
                chunk_id = self._find_chunk(
                    chunk_map, rel_path, func["line_start"]
                )
                entry = IndexEntry(
                    name=func["name"],
                    entry_type="function",
                    file_path=rel_path,
                    chunk_id=chunk_id,
                    line_start=func["line_start"],
                    line_end=func["line_end"],
                    metadata={
                        "is_async": func.get("is_async", False),
                        "args": func.get("args", []),
                        "complexity": func.get("complexity_estimate", 0),
                    },
                )
                if func["name"] not in self.function_index:
                    self.function_index[func["name"]] = []
                self.function_index[func["name"]].append(entry)
            
            # Indexar imports (dependências reversas)
            for imp in file_struct.get("imports", []):
                module = imp["module"]
                if module not in self.module_index:
                    self.module_index[module] = []
                self.module_index[module].append(rel_path)
        
        # Salvar
        self._save_index()
        
        logger.info(
            f"Índice construído: "
            f"{len(self.class_index)} classes, "
            f"{sum(len(v) for v in self.function_index.values())} funções, "
            f"{len(self.module_index)} módulos"
        )
    
    def search_function(self, name: str) -> list[dict]:
        """Busca função por nome."""
        entries = self.function_index.get(name, [])
        return [asdict(e) for e in entries]
    
    def search_class(self, name: str) -> list[dict]:
        """Busca classe por nome."""
        entries = self.class_index.get(name, [])
        return [asdict(e) for e in entries]
    
    def who_imports(self, module: str) -> list[str]:
        """Quem importa este módulo."""
        return self.module_index.get(module, [])
    
    def get_file_dependencies(self, file_path: str) -> list[str]:
        """Dependências de um arquivo."""
        return self.dependency_graph.get(file_path, [])
    
    def _find_chunk(
        self, chunk_map: dict, file_path: str, line: int
    ) -> Optional[str]:
        """Encontra o chunk que contém uma linha."""
        chunks = chunk_map.get(file_path, [])
        for chunk in chunks:
            if chunk["line_start"] <= line <= chunk["line_end"]:
                return chunk["chunk_id"]
        return None
    
    def _load_json(self, filename: str) -> dict:
        """Carrega um JSON do output."""
        path = self.output_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Salva o índice completo."""
        index = {
            "classes": {
                name: [asdict(e) for e in entries]
                for name, entries in self.class_index.items()
            },
            "functions": {
                name: [asdict(e) for e in entries]
                for name, entries in self.function_index.items()
            },
            "module_imports": self.module_index,
            "stats": {
                "unique_classes": len(self.class_index),
                "unique_functions": len(self.function_index),
                "unique_modules": len(self.module_index),
            }
        }
        
        with open(self.output_dir / "code_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    idx = CodeIndex()
    idx.build()
