"""
API Analyzer - Fase 6.
Detecta problemas em chamadas de API.
"""

import re
import logging
from typing import Optional

from .base_analyzer import BaseAnalyzer, Issue, Severity

logger = logging.getLogger(__name__)


class APIAnalyzer(BaseAnalyzer):
    """Detecta problemas em chamadas de API."""
    
    # Padrões de chamadas de API sem timeout
    API_PATTERNS = [
        r"requests\.(get|post|put|delete|patch)\(",
        r"aiohttp\.(ClientSession|request)\(",
        r"httpx\.(get|post|put|delete|patch)\(",
        r"urllib\.request\.",
        r"fetch\(",
        r"axios\.",
    ]
    
    @property
    def name(self) -> str:
        return "api_analyzer"
    
    @property
    def category(self) -> str:
        return "api"
    
    def _is_relevant(self, chunk: dict) -> bool:
        content = chunk.get("content", "")
        return any(kw in content for kw in [
            "requests.", "aiohttp", "httpx", "urllib",
            "api", "endpoint", "fetch", "axios", "timeout"
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
            
            # Verificar chamada de API sem timeout
            for pattern in self.API_PATTERNS:
                if re.search(pattern, stripped):
                    # Verificar se tem timeout
                    if not self._has_timeout(stripped):
                        issues.append(Issue(
                            issue_id=f"API-001-{file_path}-{line_num}",
                            severity=Severity.HIGH,
                            category="api",
                            title="Chamada de API sem timeout definido",
                            description=(
                                f"Chamada de API na linha {line_num} sem timeout. "
                                "Isso pode causar deadlocks em caso de API lenta."
                            ),
                            file_path=file_path,
                            line_start=line_num,
                            line_end=line_num,
                            code_snippet=stripped,
                            suggested_fix="Adicionar timeout=30 (ou outro valor adequado)",
                            confidence=0.8,
                            auto_fixable=True,
                        ))
            
            # Verificar rate limit ignorado
            if any(kw in stripped for kw in ["429", "rate_limit", "RateLimit"]):
                if "except" not in stripped.lower() and "retry" not in stripped.lower():
                    issues.append(Issue(
                        issue_id=f"API-002-{file_path}-{line_num}",
                        severity=Severity.MEDIUM,
                        category="api",
                        title="Tratamento de rate limit ausente",
                        description=(
                            "Código verifica rate limit (429) mas não tem "
                            "tratamento de retry com backoff."
                        ),
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix="Implementar retry com exponential backoff",
                        confidence=0.7,
                        auto_fixable=False,
                    ))
            
            # Verificar retry sem backoff exponencial
            if "retry" in stripped.lower() and "sleep" in stripped.lower():
                context = "\n".join(lines[max(0,i-10):min(len(lines),i+10)])
                if not re.search(r'(exponential|backoff|\*\s*2|\*\*|\*\s*=)', context, re.IGNORECASE):
                    issues.append(Issue(
                        issue_id=f"API-003-{file_path}-{line_num}",
                        severity=Severity.MEDIUM,
                        category="api",
                        title="Retry sem backoff exponencial",
                        description="Retry com delay fixo pode causar rate limit. Use backoff exponencial.",
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix="Usar: await asyncio.sleep(2 ** attempt)",
                        confidence=0.7,
                        auto_fixable=False,
                    ))
            
            # Verificar retry sem delay
            if "retry" in stripped.lower() and "sleep" not in stripped.lower():
                if "for " in stripped or "while " in stripped:
                    issues.append(Issue(
                        issue_id=f"API-003b-{file_path}-{line_num}",
                        severity=Severity.LOW,
                        category="api",
                        title="Retry sem delay entre tentativas",
                        description="Retry imediato pode falhar em APIs com rate limit.",
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix="Adicionar delay entre tentativas",
                        confidence=0.6,
                        auto_fixable=True,
                    ))
            
            # Verificar credenciais expostas
            if self._has_exposed_credentials(stripped):
                issues.append(Issue(
                    issue_id=f"API-004-{file_path}-{line_num}",
                    severity=Severity.CRITICAL,
                    category="security",
                    title="Credenciais expostas no código",
                    description=(
                        "Chave de API ou token encontrado no código. "
                        "Use variáveis de ambiente!"
                    ),
                    file_path=file_path,
                    line_start=line_num,
                    line_end=line_num,
                    code_snippet=stripped[:80] + "..." if len(stripped) > 80 else stripped,
                    suggested_fix="Usar os.environ.get('API_KEY') ou env vars",
                    confidence=0.9,
                    auto_fixable=False,
                ))
            
            # ── 5. Chamadas síncronas dentro de função async ──
            if re.search(r'requests\.(get|post|put|delete)', stripped):
                # Verificar se está dentro de função async
                async_func = self._find_enclosing_async(lines, i)
                if async_func:
                    issues.append(Issue(
                        issue_id=f"API-005-{file_path}-{line_num}",
                        severity=Severity.CRITICAL,
                        category="api",
                        title="requests síncronos em função async",
                        description=(
                            "Usar requests (síncrono) dentro de função async "
                            "bloqueia o event loop. Use aiohttp ou httpx."
                        ),
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix="Substituir por aiohttp ou httpx async",
                        confidence=0.95,
                        auto_fixable=False,
                    ))
            
            # ── 6. Chamada Binance sem tratamento de exceção ──
            if self._is_binance_call(stripped) and "await" in stripped:
                if not self._has_exception_handler(lines, i, "BinanceAPIException"):
                    issues.append(Issue(
                        issue_id=f"API-006-{file_path}-{line_num}",
                        severity=Severity.HIGH,
                        category="api",
                        title="Chamada Binance sem tratamento de exceção",
                        description=(
                            "Chamada ao client da Binance sem capturar "
                            "BinanceAPIException. Pode causar crash."
                        ),
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix="Envolver em try/except BinanceAPIException",
                        confidence=0.75,
                        auto_fixable=False,
                    ))
        
        return issues
    
    def _has_timeout(self, line: str) -> bool:
        """Verifica se a linha tem timeout definido."""
        timeout_patterns = [
            r"timeout\s*=",
            r"timeout\s*:",
            r"\.timeout\(",
            r"read_timeout\s*=",
            r"connect_timeout\s*=",
        ]
        return any(re.search(p, line) for p in timeout_patterns)
    
    def _has_exposed_credentials(self, line: str) -> bool:
        """Detecta credenciais expostas."""
        # Padrões de credenciais em texto claro
        patterns = [
            r'api[_-]?key["\']?\s*[:=]\s*["\'][A-Za-z0-9_-]{20,}',
            r'token["\']?\s*[:=]\s*["\'][A-Za-z0-9_-]{20,}',
            r'secret["\']?\s*[:=]\s*["\'][A-Za-z0-9_-]{20,}',
            r'password["\']?\s*[:=]\s*["\'][^"\']+',
            r'["\'][A-Za-z0-9/+=]{40,}["\']',  # Base64 longo
        ]
        line_lower = line.lower()
        if any(kw in line_lower for kw in ["apikey", "api_key", "token", "password", "secret"]):
            return any(re.search(p, line, re.IGNORECASE) for p in patterns)
        return False
    
    def _find_enclosing_async(self, lines: list[str], idx: int) -> Optional[str]:
        """Encontra se está dentro de função async."""
        current_indent = len(lines[idx]) - len(lines[idx].lstrip())
        for j in range(idx - 1, -1, -1):
            line = lines[j].strip()
            line_indent = len(lines[j]) - len(lines[j].lstrip())
            if line_indent < current_indent and line.startswith("async def "):
                return line
        return None
    
    def _is_binance_call(self, line: str) -> bool:
        """Verifica se a linha contém chamada Binance."""
        binance_patterns = [
            r'client\.',
            r'binance',
            r'\.exchange',
            r'ccxt',
        ]
        return any(re.search(p, line, re.IGNORECASE) for p in binance_patterns)
    
    def _has_exception_handler(
        self, lines: list[str], idx: int, exception_type: str
    ) -> bool:
        """Verifica se a chamada está protegida por except."""
        current_indent = len(lines[idx]) - len(lines[idx].lstrip())
        for j in range(idx - 1, max(0, idx - 30), -1):
            line = lines[j].strip()
            line_indent = len(lines[j]) - len(lines[j].lstrip())
            if line.startswith("try:") and line_indent < current_indent:
                # Verificar se tem except adequado
                for k in range(j + 1, min(len(lines), j + 20)):
                    if "except" in lines[k]:
                        if exception_type in lines[k] or "Exception" in lines[k]:
                            return True
                return False
        return False
