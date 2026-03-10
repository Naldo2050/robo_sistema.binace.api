"""
Test Runner - Executa testes após aplicar patches.
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PatchTestRunner:
    """Executa testes relacionados a um arquivo modificado."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
    
    def run_tests_for_file(
        self, file_path: str, timeout: int = 120
    ) -> tuple[bool, str]:
        """
        Executa testes relacionados ao arquivo.
        Retorna (passou, output).
        """
        test_files = self._find_related_tests(file_path)
        
        if not test_files:
            logger.warning(
                f"Nenhum teste encontrado para {file_path}"
            )
            return True, "Nenhum teste encontrado (aceito por padrão)"
        
        # Executar pytest apenas nos testes relacionados
        cmd = [
            "python", "-m", "pytest",
            *test_files,
            "-x",         # Parar no primeiro erro
            "--tb=short", # Traceback curto
            "-q",         # Quiet mode
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            passed = result.returncode == 0
            output = result.stdout + result.stderr
            
            if passed:
                logger.info(
                    f"✅ Testes passaram para {file_path}"
                )
            else:
                logger.warning(
                    f"❌ Testes falharam para {file_path}:\n{output[-500:]}"
                )
            
            return passed, output
            
        except subprocess.TimeoutExpired:
            return False, "Timeout nos testes"
        except Exception as e:
            return False, f"Erro ao executar testes: {e}"
    
    def _find_related_tests(self, file_path: str) -> list[str]:
        """Encontra arquivos de teste relacionados."""
        tests = []
        
        # Extrair nome do módulo
        stem = Path(file_path).stem  # "orderbook_analyzer" de "orderbook_analyzer.py"
        
        # Padrões de busca
        patterns = [
            f"tests/test_{stem}.py",
            f"tests/test_{stem}_*.py",
            f"tests/**/test_{stem}.py",
        ]
        
        for pattern in patterns:
            for match in self.project_root.glob(pattern):
                tests.append(str(match.relative_to(self.project_root)))
        
        return list(set(tests))
    
    def run_full_suite(self, timeout: int = 600) -> tuple[bool, str]:
        """Executa suite completa de testes."""
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "--tb=short",
            "-q",
            "--timeout=30",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout na suite completa"
        except Exception as e:
            return False, f"Erro na suite completa: {e}"
    
    def run_with_rollback(
        self,
        file_path: str,
        original_content: str,
        patched_content: str,
        timeout: int = 120,
    ) -> tuple[bool, str, bool]:
        """
        Executa testes com suporte a rollback automático.
        
        Returns:
            (testes_passaram, output, rollback_feito)
        """
        # Aplicar patch (escrever conteúdo modificado)
        full_path = self.project_root / file_path
        patched_content_to_apply = patched_content
        
        # Primeiro, salvar backup do original
        backup_content = full_path.read_text(encoding="utf-8")
        
        # Aplicar patch
        try:
            full_path.write_text(patched_content_to_apply, encoding="utf-8")
        except Exception as e:
            return False, f"Erro ao aplicar patch: {e}", False
        
        # Executar testes
        passed, output = self.run_tests_for_file(file_path, timeout)
        
        if not passed:
            # Rollback - restaurar conteúdo original
            logger.warning(f"Testes falharam. Fazendo rollback de {file_path}")
            full_path.write_text(backup_content, encoding="utf-8")
            return passed, output, True
        
        # Testes passaram - manter patch
        return passed, output, False
