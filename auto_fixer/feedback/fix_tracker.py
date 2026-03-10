"""
Fix Tracker - Registra histórico de patches para aprendizado.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class FixTracker:
    """Rastreia patches aplicados e resultados."""
    
    def __init__(self, db_path: str = "auto_fixer/output/fix_history.jsonl"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def record_fix(
        self,
        patch_id: str,
        issue_id: str,
        file_path: str,
        severity: str,
        category: str,
        applied: bool,
        tests_passed: Optional[bool] = None,
        reverted: bool = False,
        notes: str = "",
    ):
        """Registra um fix aplicado."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "patch_id": patch_id,
            "issue_id": issue_id,
            "file_path": file_path,
            "severity": severity,
            "category": category,
            "applied": applied,
            "tests_passed": tests_passed,
            "reverted": reverted,
            "notes": notes,
        }
        
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def get_success_rate(self, category: Optional[str] = None) -> dict:
        """Calcula taxa de sucesso de fixes."""
        records = self._load_all()
        
        if category:
            records = [r for r in records if r["category"] == category]
        
        total = len(records)
        if total == 0:
            return {"total": 0, "success_rate": 0}
        
        applied = sum(1 for r in records if r["applied"])
        tests_ok = sum(
            1 for r in records 
            if r["tests_passed"] is True
        )
        reverted = sum(1 for r in records if r["reverted"])
        
        return {
            "total": total,
            "applied": applied,
            "tests_passed": tests_ok,
            "reverted": reverted,
            "success_rate": tests_ok / max(applied, 1),
        }
    
    def get_problematic_files(self, top_n: int = 10) -> list[dict]:
        """Arquivos com mais issues recorrentes."""
        records = self._load_all()
        
        file_counts: dict[str, int] = {}
        for r in records:
            fp = r["file_path"]
            file_counts[fp] = file_counts.get(fp, 0) + 1
        
        sorted_files = sorted(
            file_counts.items(), key=lambda x: x[1], reverse=True
        )
        
        return [
            {"file": f, "issues": c} 
            for f, c in sorted_files[:top_n]
        ]
    
    def get_category_stats(self) -> dict:
        """Estatísticas por categoria."""
        records = self._load_all()
        
        stats: dict[str, dict] = {}
        for r in records:
            cat = r.get("category", "unknown")
            if cat not in stats:
                stats[cat] = {
                    "total": 0,
                    "applied": 0,
                    "tests_passed": 0,
                    "reverted": 0,
                }
            stats[cat]["total"] += 1
            if r.get("applied"):
                stats[cat]["applied"] += 1
            if r.get("tests_passed"):
                stats[cat]["tests_passed"] += 1
            if r.get("reverted"):
                stats[cat]["reverted"] += 1
        
        # Calcular taxas de sucesso
        for cat in stats:
            applied = stats[cat]["applied"]
            if applied > 0:
                stats[cat]["success_rate"] = stats[cat]["tests_passed"] / applied
            else:
                stats[cat]["success_rate"] = 0
        
        return stats
    
    def _load_all(self) -> list[dict]:
        if not self.db_path.exists():
            return []
        
        records = []
        with open(self.db_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
