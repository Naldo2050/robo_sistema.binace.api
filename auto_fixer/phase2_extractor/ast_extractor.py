"""
Extrator de estrutura AST - Fase 2.
Analisa a estrutura interna de cada arquivo Python.
"""

import ast
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    name: str
    line_start: int
    line_end: int
    args: list[str]
    has_return_type: bool
    has_docstring: bool
    is_async: bool
    decorators: list[str]
    complexity_estimate: int = 0  # Número de if/for/while/try


@dataclass
class ClassInfo:
    name: str
    line_start: int
    line_end: int
    bases: list[str]
    methods: list[FunctionInfo]
    has_docstring: bool
    decorators: list[str]


@dataclass
class ImportInfo:
    module: str
    names: list[str]  # from X import Y,Z -> [Y, Z]
    is_from_import: bool
    line: int


@dataclass
class FileStructure:
    relative_path: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    global_vars: list[str] = field(default_factory=list)
    has_main_guard: bool = False  # if __name__ == "__main__"
    parse_error: Optional[str] = None
    
    # Problemas detectados
    issues: list[dict] = field(default_factory=list)


class ASTExtractor:
    """Extrai estrutura AST de arquivos Python."""
    
    def __init__(self, output_dir: str = "auto_fixer/output"):
        self.output_dir = Path(output_dir)
        
    def extract_all(self, scan_result_path: str | None = None) -> dict:
        """Extrai estrutura de todos os arquivos do scan."""
        if scan_result_path is None:
            scan_result_path = str(self.output_dir / "scan_result.json")
            
        with open(scan_result_path, "r", encoding="utf-8") as f:
            scan = json.load(f)
        
        structures: list[FileStructure] = []
        dependency_graph: dict[str, list[str]] = {}
        
        for file_info in scan["files"]:
            path = file_info["path"]
            rel_path = file_info["relative_path"]
            
            try:
                structure = self._extract_file(path, rel_path)
                structures.append(structure)
                
                # Construir grafo de dependências
                deps = []
                for imp in structure.imports:
                    deps.append(imp.module)
                dependency_graph[rel_path] = deps
                
            except Exception as e:
                logger.warning(f"Erro ao extrair {rel_path}: {e}")
                structures.append(FileStructure(
                    relative_path=rel_path,
                    parse_error=str(e)
                ))
        
        result = {
            "total_files": len(structures),
            "total_classes": sum(len(s.classes) for s in structures),
            "total_functions": sum(len(s.functions) for s in structures),
            "files_with_errors": sum(
                1 for s in structures if s.parse_error
            ),
            "structures": [asdict(s) for s in structures],
            "dependency_graph": dependency_graph,
            "issues_summary": self._summarize_issues(structures),
        }
        
        output_file = self.output_dir / "structure_map.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(
            f"Extração completa: {result['total_classes']} classes, "
            f"{result['total_functions']} funções"
        )
        return result
    
    def _extract_file(self, filepath: str, rel_path: str) -> FileStructure:
        """Extrai estrutura de um arquivo."""
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
        
        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError as e:
            return FileStructure(
                relative_path=rel_path,
                parse_error=f"SyntaxError: {e.msg} (linha {e.lineno})"
            )
        
        structure = FileStructure(relative_path=rel_path)
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                structure.classes.append(self._extract_class(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                structure.functions.append(self._extract_function(node))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    structure.imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        is_from_import=False,
                        line=node.lineno,
                    ))
            elif isinstance(node, ast.ImportFrom):
                structure.imports.append(ImportInfo(
                    module=node.module or "",
                    names=[a.name for a in node.names],
                    is_from_import=True,
                    line=node.lineno,
                ))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        structure.global_vars.append(target.id)
            elif isinstance(node, ast.If):
                # Detectar if __name__ == "__main__"
                if self._is_main_guard(node):
                    structure.has_main_guard = True
        
        # Detectar problemas
        structure.issues = self._detect_issues(structure, source)
        
        return structure
    
    def _extract_function(self, node) -> FunctionInfo:
        """Extrai info de uma função."""
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        
        # Contar complexidade (if/for/while/try)
        complexity = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, 
                                   ast.Try, ast.ExceptHandler)):
                complexity += 1
        
        return FunctionInfo(
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            args=args,
            has_return_type=node.returns is not None,
            has_docstring=self._has_docstring(node),
            is_async=is_async,
            decorators=[self._decorator_name(d) for d in node.decorator_list],
            complexity_estimate=complexity,
        )
    
    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extrai info de uma classe."""
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item))
        
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.dump(base))
        
        return ClassInfo(
            name=node.name,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            bases=bases,
            methods=methods,
            has_docstring=self._has_docstring(node),
            decorators=[self._decorator_name(d) for d in node.decorator_list],
        )
    
    def _has_docstring(self, node) -> bool:
        """Verifica se tem docstring."""
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, (ast.Str, ast.Constant)):
                return True
        return False
    
    def _decorator_name(self, node) -> str:
        """Extrai nome do decorator."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{ast.dump(node)}"
        elif isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        return "unknown"
    
    def _is_main_guard(self, node: ast.If) -> bool:
        """Detecta if __name__ == '__main__'."""
        try:
            test = node.test
            if isinstance(test, ast.Compare):
                if isinstance(test.left, ast.Name):
                    if test.left.id == "__name__":
                        return True
        except Exception:
            pass
        return False
    
    def _detect_issues(
        self, structure: FileStructure, source: str
    ) -> list[dict]:
        """Detecta problemas básicos."""
        issues = []
        
        # Funções sem type hints
        all_funcs = list(structure.functions)
        for cls in structure.classes:
            all_funcs.extend(cls.methods)
        
        for func in all_funcs:
            if not func.has_return_type and func.name != "__init__":
                issues.append({
                    "type": "missing_return_type",
                    "severity": "LOW",
                    "function": func.name,
                    "line": func.line_start,
                })
            
            if not func.has_docstring and not func.name.startswith("_"):
                issues.append({
                    "type": "missing_docstring",
                    "severity": "LOW",
                    "function": func.name,
                    "line": func.line_start,
                })
            
            # Funções muito complexas
            if func.complexity_estimate > 15:
                issues.append({
                    "type": "high_complexity",
                    "severity": "MEDIUM",
                    "function": func.name,
                    "line": func.line_start,
                    "complexity": func.complexity_estimate,
                })
            
            # Funções muito longas
            func_length = func.line_end - func.line_start
            if func_length > 100:
                issues.append({
                    "type": "long_function",
                    "severity": "MEDIUM",
                    "function": func.name,
                    "line": func.line_start,
                    "length": func_length,
                })
        
        return issues
    
    def _summarize_issues(
        self, structures: list[FileStructure]
    ) -> dict:
        """Resume todos os problemas encontrados."""
        summary: dict[str, int] = {}
        for s in structures:
            for issue in s.issues:
                key = issue["type"]
                summary[key] = summary.get(key, 0) + 1
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = ASTExtractor()
    result = extractor.extract_all()
    print(f"\nResumo da extração:")
    print(f"  Classes:  {result['total_classes']}")
    print(f"  Funções:  {result['total_functions']}")
    print(f"  Erros:    {result['files_with_errors']}")
    print(f"\nProblemas encontrados:")
    for issue_type, count in result["issues_summary"].items():
        print(f"  {issue_type}: {count}")
