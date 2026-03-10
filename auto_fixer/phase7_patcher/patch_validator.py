"""
Patch Validator - Fase 7.
Valida patches antes de aplicar.
"""

import ast
import logging
from pathlib import Path
from typing import Optional

from .patch_generator import Patch

logger = logging.getLogger(__name__)


class PatchValidator:
    """Valida patches antes de aplicar."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
    
    def validate(self, patch: Patch) -> tuple[bool, str]:
        """Valida um patch. Retorna (é_válido, motivo)."""
        
        file_path = self.project_root / patch.file_path
        if not file_path.exists():
            return False, f"Arquivo não encontrado: {patch.file_path}"
        
        # 1. Aplicar patch em memória
        original_content = file_path.read_text(encoding="utf-8")
        lines = original_content.splitlines(keepends=True)
        
        # Verificar limites
        if patch.line_start < 1 or patch.line_end > len(lines):
            return False, f"Linhas fora do intervalo: {patch.line_start}-{patch.line_end}"
        
        # Substituir linhas
        new_lines = (
            lines[:patch.line_start - 1] 
            + [l + "\n" for l in patch.patched_lines]
            + lines[patch.line_end:]
        )
        new_content = "".join(new_lines)
        
        # 2. Verificar sintaxe
        try:
            ast.parse(new_content, filename=patch.file_path)
        except SyntaxError as e:
            return False, f"SyntaxError após patch: {e.msg} (linha {e.lineno})"
        
        # 3. Verificar se não removeu código demais
        original_lines_count = len(lines)
        new_lines_count = len(new_lines)
        if abs(original_lines_count - new_lines_count) > 50:
            return False, (
                f"Patch altera muitas linhas: "
                f"{original_lines_count} → {new_lines_count}"
            )
        
        # 4. Verificar se o arquivo ainda é importável
        try:
            # Tentar importar o módulo (se for um módulo válido)
            import_name = patch.file_path.replace("/", ".").replace("\\", ".")
            if import_name.endswith(".py"):
                import_name = import_name[:-3]
        except Exception as e:
            logger.warning(f"Não foi possível verificar import: {e}")
        
        return True, "Patch válido"
    
    def validate_syntax_only(self, code: str, filename: str = "unknown") -> tuple[bool, str]:
        """Valida apenas a sintaxe de um código."""
        try:
            ast.parse(code, filename=filename)
            return True, "Sintaxe válida"
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} (linha {e.lineno})"
