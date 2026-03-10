"""
Patch Applier - Fase 7.
Aplica patches com backup.
"""

import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .patch_generator import Patch

logger = logging.getLogger(__name__)


class PatchApplier:
    """Aplica patches com backup."""
    
    def __init__(
        self,
        project_root: str = ".",
        backup_dir: str = "auto_fixer/output/backups",
    ):
        self.project_root = Path(project_root)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def apply(self, patch: Patch) -> tuple[bool, str]:
        """Aplica patch com backup."""
        file_path = self.project_root / patch.file_path
        
        # 1. Backup
        backup_name = (
            f"{patch.file_path.replace('/', '__')}"
            f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        )
        backup_path = self.backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backup criado: {backup_path}")
        
        # 2. Aplicar
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines(keepends=True)
            
            new_lines = (
                lines[:patch.line_start - 1]
                + [l + "\n" for l in patch.patched_lines]
                + lines[patch.line_end:]
            )
            
            file_path.write_text("".join(new_lines), encoding="utf-8")
            
            patch.applied = True
            logger.info(f"Patch aplicado: {patch.patch_id}")
            return True, "Patch aplicado com sucesso"
            
        except Exception as e:
            # Rollback
            shutil.copy2(backup_path, file_path)
            logger.error(f"Erro ao aplicar patch, rollback feito: {e}")
            return False, f"Erro: {e}. Rollback realizado."
    
    def apply_batch(self, patches: list[Patch]) -> dict:
        """Aplica múltiplos patches."""
        results = {
            "applied": [],
            "failed": [],
            "total": len(patches),
        }
        
        for patch in patches:
            success, message = self.apply(patch)
            if success:
                results["applied"].append(patch.patch_id)
            else:
                results["failed"].append({
                    "patch_id": patch.patch_id,
                    "error": message,
                })
        
        return results


class RollbackManager:
    """Gerencia rollbacks de patches."""
    
    def __init__(
        self,
        project_root: str = ".",
        backup_dir: str = "auto_fixer/output/backups",
    ):
        self.project_root = Path(project_root)
        self.backup_dir = Path(backup_dir)
    
    def rollback(self, file_path: str) -> bool:
        """Restaura último backup de um arquivo."""
        safe_name = file_path.replace("/", "__")
        backups = sorted(
            self.backup_dir.glob(f"{safe_name}.*.bak"),
            reverse=True,
        )
        
        if not backups:
            logger.error(f"Nenhum backup encontrado para {file_path}")
            return False
        
        latest = backups[0]
        target = self.project_root / file_path
        shutil.copy2(latest, target)
        logger.info(f"Rollback: {latest} → {target}")
        return True
    
    def list_backups(self, file_path: str) -> list[str]:
        """Lista todos os backups de um arquivo."""
        safe_name = file_path.replace("/", "__")
        backups = sorted(
            self.backup_dir.glob(f"{safe_name}.*.bak"),
            reverse=True,
        )
        return [str(b) for b in backups]
    
    def cleanup_old_backups(self, keep_count: int = 5):
        """Remove backups antigos, mantendo apenas os N mais recentes por arquivo."""
        # Agrupar backups por arquivo
        backup_groups = {}
        for backup in self.backup_dir.glob("*.bak"):
            name_parts = backup.stem.split(".")
            if len(name_parts) >= 3:
                file_name = name_parts[0].replace("__", "/") + ".py"
                if file_name not in backup_groups:
                    backup_groups[file_name] = []
                backup_groups[file_name].append(backup)
        
        # Manter apenas os N mais recentes
        for file_name, backups in backup_groups.items():
            if len(backups) > keep_count:
                for old_backup in backups[keep_count:]:
                    old_backup.unlink()
                    logger.info(f"Backup antigo removido: {old_backup}")
