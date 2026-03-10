"""
Patch Generator - Fase 7.
Gera patches para corrigir problemas detectados.
"""

import ast
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Patch:
    patch_id: str
    issue_id: str
    file_path: str
    original_lines: list[str]
    patched_lines: list[str]
    line_start: int
    line_end: int
    description: str
    confidence: float
    created_at: str = ""
    validated: bool = False
    applied: bool = False
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_unified_diff(self) -> str:
        """Gera diff no formato unified."""
        lines = []
        lines.append(f"--- a/{self.file_path}")
        lines.append(f"+++ b/{self.file_path}")
        lines.append(
            f"@@ -{self.line_start},{len(self.original_lines)} "
            f"+{self.line_start},{len(self.patched_lines)} @@"
        )
        for line in self.original_lines:
            lines.append(f"-{line}")
        for line in self.patched_lines:
            lines.append(f"+{line}")
        return "\n".join(lines)


class PatchGenerator:
    """Gera patches para issues detectados."""
    
    # Regras de auto-fix que não precisam de IA
    STATIC_FIXES = {
        "ASYNC-003": {
            "pattern": "time.sleep(",
            "replacement": "await asyncio.sleep(",
            "add_import": "import asyncio",
        },
    }
    
    def __init__(
        self,
        project_root: str = ".",
        ai_client=None,
        output_dir: str = "auto_fixer/output",
    ):
        self.project_root = Path(project_root)
        self.ai_client = ai_client
        self.output_dir = Path(output_dir)
        self.patches_dir = self.output_dir / "patches"
        self.patches_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_patch(self, issue: dict) -> Optional[Patch]:
        """Gera patch para um issue."""
        issue_id = issue["issue_id"]
        
        # Tentar fix estático primeiro
        for prefix, fix_rule in self.STATIC_FIXES.items():
            if issue_id.startswith(prefix):
                return self._apply_static_fix(issue, fix_rule)
        
        # Se tem IA, pedir fix inteligente
        if self.ai_client and issue.get("auto_fixable"):
            return self._generate_ai_fix(issue)
        
        return None
    
    def _apply_static_fix(self, issue: dict, rule: dict) -> Optional[Patch]:
        """Aplica uma correção estática (sem IA)."""
        file_path = self.project_root / issue["file_path"]
        if not file_path.exists():
            return None
        
        lines = file_path.read_text(encoding="utf-8").splitlines(keepends=True)
        line_idx = issue["line_start"] - 1
        
        if line_idx >= len(lines):
            return None
        
        original_line = lines[line_idx]
        patched_line = original_line.replace(
            rule["pattern"], rule["replacement"]
        )
        
        patch = Patch(
            patch_id=f"PATCH-{issue['issue_id']}",
            issue_id=issue["issue_id"],
            file_path=issue["file_path"],
            original_lines=[original_line.rstrip()],
            patched_lines=[patched_line.rstrip()],
            line_start=issue["line_start"],
            line_end=issue["line_start"],
            description=f"Auto-fix: {rule['pattern']} → {rule['replacement']}",
            confidence=0.95,
        )
        
        # Salvar patch
        self._save_patch(patch)
        
        return patch
    
    def _generate_ai_fix(self, issue: dict) -> Optional[Patch]:
        """Gera fix usando IA."""
        if not self.ai_client:
            return None
        
        file_path = self.project_root / issue["file_path"]
        if not file_path.exists():
            return None
        
        # Ler contexto ao redor do problema
        lines = file_path.read_text(encoding="utf-8").splitlines()
        start = max(0, issue["line_start"] - 10)
        end = min(len(lines), issue["line_end"] + 10)
        context = "\n".join(
            f"{i+1}: {lines[i]}" for i in range(start, end)
        )
        
        prompt = f"""Corrija o seguinte problema no código Python:

PROBLEMA: {issue['title']}
DESCRIÇÃO: {issue['description']}
SEVERIDADE: {issue['severity']}

CÓDIGO (linhas {start+1}-{end}):
```
{context}
```

Responda APENAS com o JSON:
{{
"fixed_lines": ["linha corrigida 1", "linha corrigida 2"],
"line_start": {issue['line_start']},
"line_end": {issue['line_end']},
"explanation": "explicação da correção"
}}"""

        try:
            response = self.ai_client.chat(
                messages=[
                    {"role": "system", "content": "Você é um corretor de código Python. Responda apenas JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
            )
            
            fix_data = json.loads(response)
            
            original = lines[fix_data["line_start"]-1:fix_data["line_end"]]
            
            patch = Patch(
                patch_id=f"PATCH-AI-{issue['issue_id']}",
                issue_id=issue["issue_id"],
                file_path=issue["file_path"],
                original_lines=original,
                patched_lines=fix_data["fixed_lines"],
                line_start=fix_data["line_start"],
                line_end=fix_data["line_end"],
                description=fix_data.get("explanation", "AI fix"),
                confidence=0.7,
            )
            
            self._save_patch(patch)
            return patch
            
        except Exception as e:
            logger.error(f"Erro ao gerar fix com IA: {e}")
            return None
    
    def _save_patch(self, patch: Patch):
        """Salva o patch em disco."""
        patch_file = self.patches_dir / f"{patch.patch_id}.json"
        with open(patch_file, "w", encoding="utf-8") as f:
            json.dump(asdict(patch), f, indent=2, ensure_ascii=False)
        logger.info(f"Patch salvo: {patch_file}")
