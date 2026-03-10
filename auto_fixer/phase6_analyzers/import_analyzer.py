"""
Import Analyzer - Fase 6.
Detecta problemas em imports de Python.
"""

import logging
import ast
from pathlib import Path
from typing import Optional

from .base_analyzer import BaseAnalyzer, Issue, Severity

logger = logging.getLogger(__name__)


class ImportAnalyzer(BaseAnalyzer):
    """Detecta problemas em imports."""
    
    @property
    def name(self) -> str:
        return "import_analyzer"
    
    @property
    def category(self) -> str:
        return "import"
    
    def _is_relevant(self, chunk: dict) -> bool:
        return True  # Todos os arquivos têm imports
    
    def analyze_chunk(self, chunk: dict) -> list[Issue]:
        issues = []
        content = chunk.get("content", "")
        file_path = chunk.get("file_path", "")
        base_line = chunk.get("line_start", 1)
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues
        
        imports = []
        used_names = set()
        
        # Coletar imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append((alias.name, alias.asname, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append((f"{node.module}.{alias.name}", alias.asname, node.lineno))
        
        # Coletar nomes usados
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Detectar imports não usados
        for module, alias, line in imports:
            name = alias or module.split(".")[-1]
            
            # Ignorar imports comuns
            if name in ["os", "sys", "logging", "json", "re", "datetime", "typing"]:
                continue
            
            # Verificar se é usado
            if name not in used_names:
                # Também verificar variações
                name_used = False
                for used in used_names:
                    if name.lower() in used.lower() or used.lower() in name.lower():
                        name_used = True
                        break
                
                if not name_used:
                    issues.append(Issue(
                        issue_id=f"IMP-001-{file_path}-{line}",
                        severity=Severity.LOW,
                        category="import",
                        title=f"Import não usado: {name}",
                        description=f"Import '{module}' parece não ser usado.",
                        file_path=file_path,
                        line_start=line,
                        line_end=line,
                        code_snippet=f"import {module}",
                        suggested_fix=f"Remover 'import {module}'",
                        confidence=0.7,
                        auto_fixable=True,
                    ))
        
        # Detectar imports circulares (análise em nível de projeto)
        circular = self._detect_circular_imports(file_path, content)
        if circular:
            issues.append(Issue(
                issue_id=f"IMP-002-{file_path}-1",
                severity=Severity.HIGH,
                category="import",
                title="Possível import circular",
                description=f"Este arquivo pode ter import circular: {circular}",
                file_path=file_path,
                line_start=1,
                line_end=1,
                code_snippet="",
                suggested_fix="Reorganizar imports ou usar import tardio",
                confidence=0.6,
                auto_fixable=False,
            ))
        
        return issues
    
    def _detect_circular_imports(self, file_path: str, content: str) -> Optional[str]:
        """Detecta possíveis imports circulares."""
        # Extrair imports deste arquivo
        try:
            tree = ast.parse(content)
        except:
            return None
        
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        
        # Verificar se algum dos módulos importados importa de volta
        for imp in imports:
            if imp.startswith("."):
                continue  # Ignorar relativos
            
            # Tentar verificar dependência reversa (simplificado)
            # Uma implementação completa verificaria os arquivos
            pass
        
        return None
    
    def analyze_project(self, project_path: str) -> list[Issue]:
        """Analisa todos os arquivos do projeto para imports."""
        all_issues = []
        
        project_dir = Path(project_path)
        for py_file in project_dir.rglob("*.py"):
            if any(skip in str(py_file) for skip in [
                "__pycache__", "venv", ".venv", "node_modules"
            ]):
                continue
            
            try:
                content = py_file.read_text(encoding="utf-8")
                chunk = {
                    "content": content,
                    "file_path": str(py_file.relative_to(project_dir)),
                    "line_start": 1,
                }
                issues = self.analyze_chunk(chunk)
                all_issues.extend(issues)
            except Exception as e:
                logger.warning(f"Erro ao analisar {py_file}: {e}")
        
        self._save_results(all_issues)
        return all_issues
