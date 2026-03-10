"""
Scanner da codebase - Fase 1 do sistema de correção automática.
Mapeia todos os arquivos Python do projeto.
"""

import os
import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

IGNORE_DIRS = {
    "__pycache__", ".git", "node_modules", "venv", 
    ".venv", "backups", ".mypy_cache", ".pytest_cache",
    "features",  # dados particionados, não código
}

IGNORE_FILES = {
    "*.pyc", "*.pyo", "*.bak"
}


@dataclass
class FileInfo:
    """Informações de um arquivo escaneado."""
    path: str
    relative_path: str
    size_bytes: int
    line_count: int
    modified_at: str
    md5_hash: str
    is_test: bool = False
    is_large: bool = False  # > 1000 linhas
    category: str = "unknown"


@dataclass 
class ScanResult:
    """Resultado completo do scan."""
    project_root: str
    scan_timestamp: str
    total_files: int
    total_lines: int
    total_size_bytes: int
    files: list[FileInfo] = field(default_factory=list)
    large_files: list[str] = field(default_factory=list)
    categories: dict[str, int] = field(default_factory=dict)


class CodebaseScanner:
    """Scanner principal da codebase."""
    
    def __init__(self, root_path: str, output_dir: str = "auto_fixer/output"):
        self.root_path = Path(root_path).resolve()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def scan(self) -> ScanResult:
        """Executa o scan completo."""
        logger.info(f"Iniciando scan em: {self.root_path}")
        
        files: list[FileInfo] = []
        
        for py_file in self._find_python_files():
            try:
                info = self._analyze_file(py_file)
                files.append(info)
            except Exception as e:
                logger.warning(f"Erro ao analisar {py_file}: {e}")
        
        result = ScanResult(
            project_root=str(self.root_path),
            scan_timestamp=datetime.now().isoformat(),
            total_files=len(files),
            total_lines=sum(f.line_count for f in files),
            total_size_bytes=sum(f.size_bytes for f in files),
            files=files,
            large_files=[f.relative_path for f in files if f.is_large],
            categories=self._categorize_files(files),
        )
        
        self._save_result(result)
        logger.info(
            f"Scan completo: {result.total_files} arquivos, "
            f"{result.total_lines} linhas"
        )
        return result
    
    def _find_python_files(self):
        """Encontra todos os arquivos .py."""
        for root, dirs, filenames in os.walk(self.root_path):
            # Remove diretórios ignorados
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for fname in filenames:
                if fname.endswith(".py") and not fname.endswith(".pyc"):
                    yield Path(root) / fname
    
    def _analyze_file(self, filepath: Path) -> FileInfo:
        """Analisa um único arquivo."""
        content = filepath.read_bytes()
        text = content.decode("utf-8", errors="replace")
        lines = text.splitlines()
        relative = str(filepath.relative_to(self.root_path))
        
        return FileInfo(
            path=str(filepath),
            relative_path=relative,
            size_bytes=len(content),
            line_count=len(lines),
            modified_at=datetime.fromtimestamp(
                filepath.stat().st_mtime
            ).isoformat(),
            md5_hash=hashlib.md5(content).hexdigest(),
            is_test=self._is_test_file(relative),
            is_large=len(lines) > 1000,
            category=self._detect_category(relative),
        )
    
    def _is_test_file(self, path: str) -> bool:
        """Verifica se é arquivo de teste."""
        return (
            path.startswith("tests/") 
            or "/test_" in path 
            or path.startswith("test_")
        )
    
    def _detect_category(self, path: str) -> str:
        """Detecta a categoria do arquivo."""
        categories = {
            "ai_runner/": "ai",
            "flow_analyzer/": "flow",
            "market_orchestrator/": "orchestrator",
            "support_resistance/": "sr",
            "orderbook_core/": "orderbook",
            "orderbook_analyzer/": "orderbook",
            "risk_management/": "risk",
            "data_pipeline/": "data",
            "ml/": "ml",
            "tests/": "test",
            "scripts/": "script",
            "tools/": "tool",
            "diagnostics/": "diagnostic",
            "src/": "source",
        }
        for prefix, cat in categories.items():
            if path.startswith(prefix):
                return cat
        return "root"
    
    def _categorize_files(self, files: list[FileInfo]) -> dict[str, int]:
        """Conta arquivos por categoria."""
        cats: dict[str, int] = {}
        for f in files:
            cats[f.category] = cats.get(f.category, 0) + 1
        return cats
    
    def _save_result(self, result: ScanResult):
        """Salva resultado em JSON."""
        output_file = self.output_dir / "scan_result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        logger.info(f"Resultado salvo em: {output_file}")


# Uso direto
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scanner = CodebaseScanner(".")
    result = scanner.scan()
    print(f"\nResumo:")
    print(f"  Arquivos: {result.total_files}")
    print(f"  Linhas:   {result.total_lines:,}")
    print(f"  Tamanho:  {result.total_size_bytes / 1024 / 1024:.1f} MB")
    print(f"\nArquivos grandes (>1000 linhas):")
    for f in result.large_files:
        print(f"  - {f}")
    print(f"\nCategorias:")
    for cat, count in sorted(result.categories.items()):
        print(f"  {cat}: {count}")
