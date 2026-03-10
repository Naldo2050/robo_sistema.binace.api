# auto_fixer/fix_high_issues.py
"""
Fix especializado para os 120 HIGH issues.
Corrige:
  1. Chamadas de API sem timeout (46)
  2. await sem try/except (45)
  3. except: pass em WebSocket (18)
  4. Binance sem BinanceAPIException (6)
  5. on_message sem tratamento (5)

Uso:
  python -m auto_fixer.fix_high_issues              # Dry-run
  python -m auto_fixer.fix_high_issues --apply      # Aplicar
  python -m auto_fixer.fix_high_issues --rollback   # Desfazer
"""

import re
import ast
import json
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fix_high")

BACKUP_DIR = Path("auto_fixer/output/backups_high")
FIX_LOG = Path("auto_fixer/output/fix_high_log.jsonl")

# Arquivos para ignorar (legado, backups, problemáticos)
IGNORE_FILES = {
    "legacy/",
    ".backup_",
    ".bak",
    "main.patched.backup",
    "ai_runner\\",
    "ai_runner/",
    "auto_fixer/phase6_analyzers/api_analyzer.py",
    "phase6_analyzers/api_analyzer.py",
}


@dataclass
class FixResult:
    file_path: str
    fix_type: str
    line_number: int
    status: str  # "applied", "skipped", "error"
    detail: str
    backup_path: Optional[str] = None


class HighIssueFixer:
    """Corrige os HIGH issues de forma inteligente."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results: list[FixResult] = []
        self.files_modified: set[str] = set()
        
    def fix_all(self, dry_run: bool = True) -> list[FixResult]:
        """Corrige todos os HIGH issues."""
        logger.info(f"{'DRY-RUN' if dry_run else 'APLICANDO'} fixes HIGH...")
        
        # Carregar issues
        issues = self._load_high_issues()
        logger.info(f"  {len(issues)} HIGH issues encontrados")
        
        # Agrupar por arquivo para processar uma vez só
        by_file: dict[str, list[dict]] = {}
        for issue in issues:
            fp = issue["file_path"]
            
            # Normalizar caminho (Windows -> Unix)
            fp_normalized = fp.replace("\\", "/")
            
            # Ignorar arquivos de backup/legado
            if any(ign in fp_normalized for ign in IGNORE_FILES):
                continue
                
            if fp not in by_file:
                by_file[fp] = []
            by_file[fp].append(issue)
        
        logger.info(f"  {len(by_file)} arquivos para processar")
        
        # Processar cada arquivo
        for file_path, file_issues in by_file.items():
            self._fix_file(file_path, file_issues, dry_run)
        
        # Salvar log
        if not dry_run:
            self._save_log()
        
        return self.results
    
    def _fix_file(
        self, 
        file_path: str, 
        issues: list[dict], 
        dry_run: bool
    ):
        """Corrige todos os issues de um arquivo."""
        full_path = self.project_root / file_path
        
        if not full_path.exists():
            for issue in issues:
                self.results.append(FixResult(
                    file_path=file_path,
                    fix_type=issue.get("title", "unknown"),
                    line_number=issue["line_start"],
                    status="skipped",
                    detail="Arquivo não encontrado",
                ))
            return
        
        try:
            original_content = full_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Erro ao ler {file_path}: {e}")
            return
        
        content = original_content
        lines = content.splitlines()
        
        # Ordenar issues por linha (de baixo para cima para não invalidar índices)
        issues_sorted = sorted(issues, key=lambda i: i["line_start"], reverse=True)
        
        changes_made = 0
        
        for issue in issues_sorted:
            line_num = issue["line_start"]
            title = issue.get("title", "")
            
            # Aplicar fix baseado no tipo
            if "timeout" in title.lower():
                new_lines, changed = self._fix_timeout(lines, line_num)
            elif "try/except" in title.lower() or "await" in title.lower():
                new_lines, changed = self._fix_await_try_except(lines, line_num)
            elif "except" in title.lower() and "pass" in title.lower():
                new_lines, changed = self._fix_except_pass(lines, line_num)
            elif "binance" in title.lower():
                new_lines, changed = self._fix_binance_exception(lines, line_num)
            elif "on_message" in title.lower():
                new_lines, changed = self._fix_on_message(lines, line_num)
            else:
                changed = False
                new_lines = lines
            
            if changed:
                lines = new_lines
                changes_made += 1
                
                self.results.append(FixResult(
                    file_path=file_path,
                    fix_type=title,
                    line_number=line_num,
                    status="would_apply" if dry_run else "applied",
                    detail="Fix aplicado",
                ))
            else:
                self.results.append(FixResult(
                    file_path=file_path,
                    fix_type=title,
                    line_number=line_num,
                    status="skipped",
                    detail="Padrão não encontrado ou já corrigido",
                ))
        
        # Se houve mudanças e não é dry-run, salvar
        if changes_made > 0 and not dry_run:
            new_content = "\n".join(lines)
            
            # Validar sintaxe antes de salvar
            try:
                ast.parse(new_content)
            except SyntaxError as e:
                logger.error(f"Syntax error após fix em {file_path}: {e}")
                # Marcar como erro
                for r in self.results:
                    if r.file_path == file_path and r.status == "applied":
                        r.status = "error"
                        r.detail = f"SyntaxError: {e}"
                return
            
            # Backup
            backup_path = self._backup_file(full_path)
            
            # Salvar
            full_path.write_text(new_content, encoding="utf-8")
            self.files_modified.add(file_path)
            
            logger.info(f"  {file_path}: {changes_made} fixes aplicados")
    
    def _fix_timeout(
        self, lines: list[str], line_num: int
    ) -> tuple[list[str], bool]:
        """Adiciona timeout em chamadas de API."""
        idx = line_num - 1
        if idx >= len(lines):
            return lines, False
        
        line = lines[idx]
        
        # Já tem timeout?
        if "timeout" in line.lower():
            return lines, False
        
        # Padrões de chamada HTTP
        patterns = [
            (r'aiohttp\.ClientSession\(\)', 
             'aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))'),
            (r'ClientSession\(\)', 
             'ClientSession(timeout=aiohttp.ClientTimeout(total=30))'),
            (r'session\.get\(([^)]+)\)', 
             r'session.get(\1, timeout=30)'),
            (r'session\.post\(([^)]+)\)', 
             r'session.post(\1, timeout=30)'),
            (r'requests\.get\(([^)]+)\)', 
             r'requests.get(\1, timeout=10)'),
            (r'requests\.post\(([^)]+)\)', 
             r'requests.post(\1, timeout=10)'),
        ]
        
        new_line = line
        for pattern, replacement in patterns:
            if re.search(pattern, line):
                new_line = re.sub(pattern, replacement, line)
                break
        
        if new_line != line:
            lines = lines.copy()
            lines[idx] = new_line
            return lines, True
        
        return lines, False
    
    def _fix_await_try_except(
        self, lines: list[str], line_num: int
    ) -> tuple[list[str], bool]:
        """Envolve await em try/except."""
        idx = line_num - 1
        if idx >= len(lines):
            return lines, False
        
        line = lines[idx]
        
        # Já está em try block?
        if self._is_in_try_block(lines, idx):
            return lines, False
        
        # Deve ter await
        if "await " not in line:
            return lines, False
        
        # Calcular indentação
        indent = len(line) - len(line.lstrip())
        spaces = " " * indent
        inner_spaces = " " * (indent + 4)
        
        # Criar try/except
        new_lines = lines.copy()
        
        # Inserir try antes
        new_lines[idx] = f"{spaces}try:\n{inner_spaces}{line.strip()}"
        
        # Inserir except depois
        except_block = (
            f"\n{spaces}except Exception as e:\n"
            f"{inner_spaces}logger.error(f\"Erro em operação async: {{e}}\")\n"
            f"{inner_spaces}raise"
        )
        
        # Encontrar fim da instrução (pode ser multi-linha)
        end_idx = idx
        open_parens = line.count("(") - line.count(")")
        while open_parens > 0 and end_idx < len(new_lines) - 1:
            end_idx += 1
            open_parens += new_lines[end_idx].count("(")
            open_parens -= new_lines[end_idx].count(")")
        
        # Adicionar except após o fim da instrução
        if end_idx < len(new_lines):
            new_lines[end_idx] = new_lines[end_idx] + except_block
        
        return new_lines, True
    
    def _fix_except_pass(
        self, lines: list[str], line_num: int
    ) -> tuple[list[str], bool]:
        """Substitui except:pass por logging."""
        idx = line_num - 1
        if idx >= len(lines) - 1:
            return lines, False
        
        current = lines[idx].strip()
        next_line = lines[idx + 1].strip() if idx + 1 < len(lines) else ""
        
        # Verificar padrão
        if not (current.startswith("except") and next_line == "pass"):
            return lines, False
        
        indent = len(lines[idx]) - len(lines[idx].lstrip())
        spaces = " " * indent
        inner_spaces = " " * (indent + 4)
        
        new_lines = lines.copy()
        
        # Melhorar o except
        if current == "except:":
            new_lines[idx] = f"{spaces}except Exception as e:"
        elif current == "except Exception:":
            new_lines[idx] = f"{spaces}except Exception as e:"
        
        # Substituir pass por logging
        new_lines[idx + 1] = (
            f'{inner_spaces}logger.warning(f"Erro ignorado: {{e}}")'
        )
        
        return new_lines, True
    
    def _fix_binance_exception(
        self, lines: list[str], line_num: int
    ) -> tuple[list[str], bool]:
        """Adiciona tratamento de BinanceAPIException."""
        idx = line_num - 1
        if idx >= len(lines):
            return lines, False
        
        line = lines[idx]
        
        # Já está em try?
        if self._is_in_try_block(lines, idx):
            return lines, False
        
        # Deve ter chamada client.
        if "client." not in line and "await" not in line:
            return lines, False
        
        indent = len(line) - len(line.lstrip())
        spaces = " " * indent
        inner_spaces = " " * (indent + 4)
        
        new_lines = lines.copy()
        
        # Criar try/except específico para Binance
        new_lines[idx] = f"{spaces}try:\n{inner_spaces}{line.strip()}"
        
        except_block = (
            f"\n{spaces}except BinanceAPIException as e:\n"
            f"{inner_spaces}logger.error(f\"Binance API error: {{e.message"
            f"{inner_spaces}}}\")\nraise\n"
            f"{spaces}except Exception as e:\n"
            f"{inner_spaces}logger.error(f\"Erro inesperado: {{e}}\")\n"
            f"{inner_spaces}raise"
        )
        
        # Encontrar fim da instrução
        end_idx = idx
        open_parens = line.count("(") - line.count(")")
        while open_parens > 0 and end_idx < len(new_lines) - 1:
            end_idx += 1
            open_parens += new_lines[end_idx].count("(")
            open_parens -= new_lines[end_idx].count(")")
        
        if end_idx < len(new_lines):
            new_lines[end_idx] = new_lines[end_idx] + except_block
        
        return new_lines, True
    
    def _fix_on_message(
        self, lines: list[str], line_num: int
    ) -> tuple[list[str], bool]:
        """Adiciona tratamento de erros em on_message."""
        idx = line_num - 1
        if idx >= len(lines):
            return lines, False
        
        line = lines[idx]
        
        # Procurar def on_message ou async def on_message
        if "def on_message" not in line and "def _on_message" not in line:
            return lines, False
        
        # Verificar se já tem try no corpo
        # Procurar próximas linhas
        func_indent = len(line) - len(line.lstrip())
        body_indent = func_indent + 4
        
        for i in range(idx + 1, min(idx + 5, len(lines))):
            body_line = lines[i]
            if body_line.strip() and not body_line.strip().startswith("#"):
                current_indent = len(body_line) - len(body_line.lstrip())
                if current_indent == body_indent:
                    if body_line.strip().startswith("try:"):
                        return lines, False  # Já tem try
                break
        
        # Não implementar aqui pois requer reestruturação maior
        # Marcar como necessitando revisão manual
        return lines, False
    
    def _is_in_try_block(self, lines: list[str], idx: int) -> bool:
        """Verifica se a linha está dentro de um try block."""
        current_indent = len(lines[idx]) - len(lines[idx].lstrip())
        
        for i in range(idx - 1, max(0, idx - 30), -1):
            line = lines[i]
            if not line.strip():
                continue
            line_indent = len(line) - len(line.lstrip())
            
            if line_indent < current_indent:
                if line.strip().startswith("try:"):
                    return True
                elif line.strip().startswith(("def ", "async def ", "class ")):
                    return False
        
        return False
    
    def _backup_file(self, filepath: Path) -> str:
        """Faz backup do arquivo."""
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = str(filepath).replace("/", "__").replace("\\", "__")
        backup_path = BACKUP_DIR / f"{safe_name}.{timestamp}.bak"
        
        shutil.copy2(filepath, backup_path)
        return str(backup_path)
    
    def _load_high_issues(self) -> list[dict]:
        """Carrega issues HIGH dos resultados."""
        results_dir = Path("auto_fixer/output/analysis_results")
        issues = []
        
        for f in results_dir.glob("*_results.json"):
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
                for issue in data.get("issues", []):
                    if issue.get("severity") == "HIGH":
                        issues.append(issue)
        
        return issues
    
    def _save_log(self):
        """Salva log dos fixes."""
        FIX_LOG.parent.mkdir(parents=True, exist_ok=True)
        
        with open(FIX_LOG, "a", encoding="utf-8") as f:
            for result in self.results:
                record = asdict(result)
                record["timestamp"] = datetime.now().isoformat()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def rollback(self):
        """Desfaz os últimos fixes."""
        if not BACKUP_DIR.exists():
            logger.error("Nenhum backup encontrado")
            return
        
        backups = sorted(BACKUP_DIR.glob("*.bak"), reverse=True)
        
        if not backups:
            logger.error("Nenhum backup encontrado")
            return
        
        # Restaurar últimos 50 backups
        restored = 0
        for backup in backups[:50]:
            # Extrair nome original
            name = backup.stem.rsplit(".", 1)[0]  # Remove timestamp
            original = name.replace("__", "/")
            
            if Path(original).exists():
                shutil.copy2(backup, original)
                restored += 1
                logger.info(f"  Restaurado: {original}")
        
        logger.info(f"{restored} arquivos restaurados")
    
    def get_summary(self) -> dict:
        """Resumo dos resultados."""
        from collections import Counter
        
        status_count = Counter(r.status for r in self.results)
        type_count = Counter(r.fix_type for r in self.results)
        
        return {
            "total": len(self.results),
            "by_status": dict(status_count),
            "by_type": dict(type_count),
            "files_modified": len(self.files_modified),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Fix especializado para HIGH issues"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Aplicar fixes (sem isso é dry-run)"
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Desfazer últimos fixes"
    )
    
    args = parser.parse_args()
    
    fixer = HighIssueFixer()
    
    if args.rollback:
        fixer.rollback()
        return
    
    mode = "APLICANDO" if args.apply else "DRY-RUN"
    
    print("=" * 60)
    print(f"FIX HIGH ISSUES - {mode}")
    print("=" * 60)
    
    results = fixer.fix_all(dry_run=not args.apply)
    summary = fixer.get_summary()
    
    print()
    print(f"Resultado:")
    print(f"   Total processado: {summary['total']}")
    for status, count in summary["by_status"].items():
        icon = {
            "applied": "OK",
            "would_apply": "OK",
            "skipped": "SKIP",
            "error": "ERRO",
        }.get(status, "?")
        print(f"   {icon} {status}: {count}")
    
    if summary["files_modified"]:
        print(f"\n   Arquivos modificados: {summary['files_modified']}")
    
    if not args.apply:
        print()
        print("Para aplicar: python -m auto_fixer.fix_high_issues --apply")
        print("Para desfazer: python -m auto_fixer.fix_high_issues --rollback")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
