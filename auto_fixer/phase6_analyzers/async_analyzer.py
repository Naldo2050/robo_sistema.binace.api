"""
Async Analyzer - Fase 6.
Detecta problemas em código assíncrono.
"""

import logging
from typing import Optional

from .base_analyzer import BaseAnalyzer, Issue, Severity

logger = logging.getLogger(__name__)


class AsyncAnalyzer(BaseAnalyzer):
    """Detecta problemas em código assíncrono."""
    
    @property
    def name(self) -> str:
        return "async_analyzer"
    
    @property
    def category(self) -> str:
        return "async"
    
    def _is_relevant(self, chunk: dict) -> bool:
        content = chunk.get("content", "")
        return any(kw in content for kw in [
            "async ", "await ", "asyncio", "aiohttp",
            "websockets", "create_task"
        ])
    
    def analyze_chunk(self, chunk: dict) -> list[Issue]:
        issues = []
        content = chunk.get("content", "")
        lines = content.splitlines()
        file_path = chunk.get("file_path", "")
        base_line = chunk.get("line_start", 1)
        
        for i, line in enumerate(lines):
            line_num = base_line + i
            stripped = line.strip()
            
            # Padrão: await sem try/except
            if "await " in stripped and not self._in_try_block(lines, i):
                # Verificar se é crítico (chamada de rede)
                if any(kw in stripped for kw in [
                    "fetch", "request", "connect", "send", "recv",
                    "get(", "post(", "put("
                ]):
                    issues.append(Issue(
                        issue_id=f"ASYNC-001-{file_path}-{line_num}",
                        severity=Severity.HIGH,
                        category="async",
                        title="await de operação de rede sem try/except",
                        description=(
                            f"A linha {line_num} faz await de uma operação "
                            f"de rede sem tratamento de exceção. Isso pode "
                            f"causar crash em caso de timeout ou erro de rede."
                        ),
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix=(
                            "Envolver em try/except com tratamento de "
                            "asyncio.TimeoutError e ConnectionError"
                        ),
                        confidence=0.85,
                        auto_fixable=True,
                    ))
            
            # Padrão: asyncio.sleep(0) - possível busy wait
            if "asyncio.sleep(0)" in stripped:
                issues.append(Issue(
                    issue_id=f"ASYNC-002-{file_path}-{line_num}",
                    severity=Severity.LOW,
                    category="async",
                    title="asyncio.sleep(0) - possível busy wait",
                    description="Considere usar asyncio.sleep(0.01) mínimo.",
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    code_snippet=stripped,
                    confidence=0.6,
                    auto_fixable=True,
                ))
            
            # Padrão: time.sleep em código async
            if "time.sleep(" in stripped:
                # Verificar se está em função async
                func_ctx = self._find_enclosing_function(lines, i)
                if func_ctx and "async" in func_ctx:
                    issues.append(Issue(
                        issue_id=f"ASYNC-003-{file_path}-{line_num}",
                        severity=Severity.CRITICAL,
                        category="async",
                        title="time.sleep() em função async - BLOQUEIA EVENT LOOP",
                        description=(
                            "Usar time.sleep() dentro de função async bloqueia "
                            "todo o event loop. Use await asyncio.sleep()."
                        ),
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix="Substituir time.sleep(X) por await asyncio.sleep(X)",
                        confidence=0.95,
                        auto_fixable=True,
                    ))
            
            # Padrão: create_task sem referência armazenada
            if "create_task(" in stripped and "=" not in stripped:
                issues.append(Issue(
                    issue_id=f"ASYNC-004-{file_path}-{line_num}",
                    severity=Severity.MEDIUM,
                    category="async",
                    title="create_task sem referência - possível task perdida",
                    description=(
                        "create_task() retorna uma Task que deve ser "
                        "armazenada para evitar WarnUnreachable."
                    ),
                    file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                    code_snippet=stripped,
                    suggested_fix="Armazenar o retorno: task = asyncio.create_task(...)",
                    confidence=0.7,
                    auto_fixable=False,
                ))
        
        return issues
    
    def _in_try_block(self, lines: list[str], idx: int) -> bool:
        """Verifica se a linha está dentro de um try."""
        current_indent = len(lines[idx]) - len(lines[idx].lstrip())
        for j in range(idx - 1, max(0, idx - 20), -1):
            line = lines[j].strip()
            line_indent = len(lines[j]) - len(lines[j].lstrip())
            if line.startswith("try:") and line_indent < current_indent:
                return True
        return False
    
    def _find_enclosing_function(
        self, lines: list[str], idx: int
    ) -> Optional[str]:
        """Encontra a definição da função que contém a linha."""
        current_indent = len(lines[idx]) - len(lines[idx].lstrip())
        for j in range(idx - 1, -1, -1):
            line = lines[j].strip()
            line_indent = len(lines[j]) - len(lines[j].lstrip())
            if line_indent < current_indent and (
                line.startswith("def ") or line.startswith("async def ")
            ):
                return line
        return None
