"""
WebSocket Analyzer - Fase 6.
Detecta problemas em conexões WebSocket.
"""

import logging
import re
from typing import Optional

from .base_analyzer import BaseAnalyzer, Issue, Severity

logger = logging.getLogger(__name__)


class WebSocketAnalyzer(BaseAnalyzer):
    """Detecta problemas em código WebSocket."""
    
    @property
    def name(self) -> str:
        return "websocket_analyzer"
    
    @property
    def category(self) -> str:
        return "websocket"
    
    def _is_relevant(self, chunk: dict) -> bool:
        content = chunk.get("content", "")
        return any(kw in content for kw in [
            "websocket", "websockets", "ws://", "wss://",
            "connect", "send", "recv", "on_message"
        ])
    
    def analyze_chunk(self, chunk: dict) -> list[Issue]:
        issues = []
        content = chunk.get("content", "")
        lines = content.splitlines()
        file_path = chunk.get("file_path", "")
        base_line = chunk.get("line_start", 1)
        
        has_reconnect = False
        has_heartbeat = False
        has_on_close = False
        
        # Primeira passada: detectar features presentes
        for line in lines:
            stripped = line.strip().lower()
            if "reconnect" in stripped or "retry" in stripped:
                has_reconnect = True
            if "heartbeat" in stripped or "ping" in stripped or "pong" in stripped:
                has_heartbeat = True
            if "on_close" in stripped or "on_disconnect" in stripped:
                has_on_close = True
        
        for i, line in enumerate(lines):
            line_num = base_line + i
            stripped = line.strip()
            
            # Verificar reconnect falho
            if "while True" in stripped or "while 1" in stripped:
                # Procurar por reconnect logic nas próximas linhas
                context = "\n".join(lines[i:i+20])
                if "connect" in context.lower() and not has_reconnect:
                    issues.append(Issue(
                        issue_id=f"WS-001-{file_path}-{line_num}",
                        severity=Severity.HIGH,
                        category="websocket",
                        title="Loop de reconexão sem estratégia de reconexão",
                        description=(
                            "Loop de conexão sem estratégia de reconexão "
                            "com backoff exponencial."
                        ),
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix="Implementar reconexão com backoff",
                        confidence=0.7,
                        auto_fixable=False,
                    ))
            
            # Verificar heartbeat ausente
            if "connect" in stripped.lower() or "websocket" in stripped.lower():
                if not has_heartbeat:
                    # Verificar se é função de conexão
                    context_before = "\n".join(lines[max(0, i-5):i])
                    if "def " in context_before:
                        issues.append(Issue(
                            issue_id=f"WS-002-{file_path}-{line_num}",
                            severity=Severity.MEDIUM,
                            category="websocket",
                            title="Heartbeat/ping ausente",
                            description=(
                                "Conexão WebSocket sem mecanismo de heartbeat "
                                "para detectar desconexões."
                            ),
                            file_path=file_path,
                            line_start=line_num,
                            line_end=line_num,
                            code_snippet=stripped,
                            suggested_fix="Adicionar ping/pong periódico",
                            confidence=0.6,
                            auto_fixable=False,
                        ))
            
            # Verificar subscription não restaurada
            if "on_close" in stripped.lower() or "on_disconnect" in stripped.lower():
                has_on_close = True
            
            if "subscribe" in stripped.lower() or "subscription" in stripped.lower():
                if not has_on_close:
                    issues.append(Issue(
                        issue_id=f"WS-003-{file_path}-{line_num}",
                        severity=Severity.MEDIUM,
                        category="websocket",
                        title="Subscriptions não restauradas após reconexão",
                        description=(
                            "Código faz subscribe mas não restaura "
                            "após reconexão."
                        ),
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix="Salvar subscriptions e restaurar no on_open",
                        confidence=0.6,
                        auto_fixable=False,
                    ))
            
            # Verificar erro não tratado em on_message
            if "on_message" in stripped and "def " in stripped:
                func_lines = self._extract_function(lines, i)
                if func_lines and "except" not in "\n".join(func_lines):
                    issues.append(Issue(
                        issue_id=f"WS-004-{file_path}-{line_num}",
                        severity=Severity.HIGH,
                        category="websocket",
                        title="on_message sem tratamento de erros",
                        description=(
                            "Handler de mensagem sem try/except pode "
                            "encerrar a conexão em caso de erro."
                        ),
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        code_snippet=stripped,
                        suggested_fix="Envolver em try/except com log",
                        confidence=0.8,
                        auto_fixable=True,
                    ))
            
            # 5. Erro silencioso (except: pass)
            if stripped.startswith("except"):
                # Verificar se próximo linhas são 'pass'
                next_lines = lines[i+1:i+3] if i+1 < len(lines) else []
                if any(l.strip() == "pass" for l in next_lines):
                    # Verificar se é contexto WebSocket
                    context = "\n".join(lines[max(0,i-10):i+3])
                    if any(kw in context.lower() for kw in ["websocket", "ws", "on_error", "on_close"]):
                        issues.append(Issue(
                            issue_id=f"WS-005-{file_path}-{line_num}",
                            severity=Severity.HIGH,
                            category="websocket",
                            title="Erro silencioso em WebSocket (except: pass)",
                            description=(
                                "Erros em WebSocket estão sendo engolidos com pass. "
                                "Isso esconde problemas de conexão."
                            ),
                            file_path=file_path,
                            line_start=line_num,
                            line_end=line_num + 1,
                            code_snippet=f"{stripped}\n    pass",
                            suggested_fix="Adicionar logging do erro",
                            confidence=0.90,
                            auto_fixable=True,
                        ))
            
            # 6. Buffer de mensagens sem limite
            if ".append(" in stripped or ".extend(" in stripped:
                # Verificar se parece buffer de mensagens WS
                context = "\n".join(lines[max(0,i-5):min(len(lines),i+3)])
                if any(kw in context.lower() for kw in ["message", "event", "trade", "tick", "queue", "buffer"]):
                    # Verificar se tem limite
                    if not re.search(r"(maxlen|max_size|MAX_|limit)", context):
                        issues.append(Issue(
                            issue_id=f"WS-006-{file_path}-{line_num}",
                            severity=Severity.MEDIUM,
                            category="websocket",
                            title="Buffer de mensagens sem limite",
                            description=(
                                "Lista/buffer crescendo sem limite. "
                                "Em produção pode causar OutOfMemory."
                            ),
                            file_path=file_path,
                            line_start=line_num,
                            line_end=line_num,
                            code_snippet=stripped,
                            suggested_fix="Usar collections.deque(maxlen=N) ou implementar pruning",
                            confidence=0.65,
                            auto_fixable=False,
                        ))
        
        return issues
    
    def _extract_function(self, lines: list[str], start_idx: int) -> list[str]:
        """Extrai o conteúdo de uma função."""
        if start_idx >= len(lines):
            return []
        
        func_lines = []
        base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
        
        for i in range(start_idx + 1, min(start_idx + 30, len(lines))):
            line = lines[i]
            if not line.strip():
                func_lines.append(line)
                continue
            
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent and line.strip():
                break
            
            func_lines.append(line)
        
        return func_lines
