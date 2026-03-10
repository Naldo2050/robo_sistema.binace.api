"""
Base Analyzer - Fase 6.
Classe base para todos os analisadores de código.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class Issue:
    """Problema detectado no código."""
    issue_id: str
    severity: Severity
    category: str  # "async", "api", "websocket", "import", "security"
    title: str
    description: str
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str
    suggested_fix: Optional[str] = None
    confidence: float = 0.0  # 0-1
    auto_fixable: bool = False
    related_files: list[str] = field(default_factory=list)


class BaseAnalyzer(ABC):
    """Classe base para analisadores."""
    
    def __init__(
        self,
        ai_client=None,  # Cliente da API de IA separada
        output_dir: str = "auto_fixer/output",
    ):
        self.ai_client = ai_client
        self.output_dir = Path(output_dir)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nome do analisador."""
        ...
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Categoria de problemas que detecta."""
        ...
    
    @abstractmethod
    def analyze_chunk(self, chunk: dict) -> list[Issue]:
        """Analisa um chunk de código."""
        ...
    
    def analyze_all(self, chunks: list[dict]) -> list[Issue]:
        """Analisa todos os chunks relevantes."""
        all_issues = []
        
        for chunk in chunks:
            if self._is_relevant(chunk):
                try:
                    issues = self.analyze_chunk(chunk)
                    all_issues.extend(issues)
                except Exception as e:
                    logger.error(
                        f"Erro no {self.name} para {chunk.get('chunk_id')}: {e}"
                    )
        
        # Salvar resultados
        self._save_results(all_issues)
        
        return all_issues
    
    def _is_relevant(self, chunk: dict) -> bool:
        """Verifica se o chunk é relevante para este analisador."""
        return True  # Override nas subclasses
    
    def _save_results(self, issues: list[Issue]):
        """Salva resultados da análise."""
        results_dir = self.output_dir / "analysis_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        output = {
            "analyzer": self.name,
            "category": self.category,
            "total_issues": len(issues),
            "by_severity": {
                s.value: sum(1 for i in issues if i.severity == s)
                for s in Severity
            },
            "issues": [asdict(i) for i in issues],
        }
        
        with open(results_dir / f"{self.name}_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def _ask_ai(self, prompt: str, code: str) -> str:
        """Envia para a API de IA separada."""
        if not self.ai_client:
            return ""
        
        try:
            # Adaptar ao seu cliente de IA
            response = self.ai_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um auditor de código Python especializado. "
                            "Analise o código e identifique problemas. "
                            "Responda em JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n```python\n{code}\n```",
                    }
                ],
                temperature=0.1,
            )
            return response
        except Exception as e:
            logger.error(f"Erro na API de IA: {e}")
            return ""
