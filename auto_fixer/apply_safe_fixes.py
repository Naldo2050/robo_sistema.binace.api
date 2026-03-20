"""
Aplica fixes seguros automaticamente.
Apenas issues com:
  - auto_fixable = True
  - confidence >= 0.90
  - Backup automático antes de cada fix
  - Validação de sintaxe após cada fix

Uso:
  python -m auto_fixer.apply_safe_fixes              # Dry-run (mostra o que faria)
  python -m auto_fixer.apply_safe_fixes --apply       # Aplica de verdade
  python -m auto_fixer.apply_safe_fixes --rollback    # Desfaz último batch
"""

import sys
import os
import ast
import json
import shutil
import argparse
import logging

# Configurar UTF-8 para suportar emojis no Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for _stream in (sys.stdout, sys.stderr):
        _reconf = getattr(_stream, "reconfigure", None)
        if _reconf and not _stream.closed:
            try:
                _reconf(encoding="utf-8", errors="replace")
            except Exception:
                pass
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BACKUP_DIR = Path("auto_fixer/output/backups")
RESULTS_DIR = Path("auto_fixer/output/analysis_results")
FIX_LOG = Path("auto_fixer/output/fix_log.jsonl")

# ── Regras de fix estático (sem IA) ──
STATIC_FIXES = {
    "ASYNC-003": {
        # time.sleep() em função async
        "find": "time.sleep(",
        "replace": "await asyncio.sleep(",
        "add_import": "import asyncio",
        "description": "time.sleep → await asyncio.sleep",
    },
    "API-001": {
        # requests sem timeout
        "pattern_type": "regex_append",
        "find_pattern": r"requests\.(get|post|put|delete|patch)\([^)]*\)",
        "append_if_missing": ", timeout=10",
        "check_missing": "timeout",
        "description": "Adicionado timeout=10 em requests",
    },
    "WS-002": {
        # except: pass → except com logging
        "pattern_type": "multiline",
        "find_lines": ["except:", "    pass"],
        "replace_lines": [
            "except Exception as e:",
            "    logger.error(f'Erro: {e}')",
        ],
        "description": "except:pass → except com logging",
    },
}

MIN_CONFIDENCE = 0.90


def load_fixable_issues() -> list[dict]:
    """Carrega issues que podem ser corrigidos automaticamente."""
    issues = []
    
    for f in RESULTS_DIR.glob("*_results.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            for issue in data.get("issues", []):
                if issue.get("auto_fixable") and \
                   issue.get("confidence", 0) >= MIN_CONFIDENCE:
                    issues.append(issue)
    
    # Ordenar por severidade
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    issues.sort(key=lambda i: sev_order.get(i["severity"], 4))
    
    return issues


def backup_file(filepath: Path) -> Path:
    """Faz backup do arquivo."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    safe_name = str(filepath).replace(os.sep, "__").replace("/", "__")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{safe_name}.{timestamp}.bak"
    
    shutil.copy2(filepath, backup_path)
    return backup_path


def validate_syntax(filepath: Path) -> tuple[bool, str]:
    """Valida sintaxe Python do arquivo."""
    try:
        content = filepath.read_text(encoding="utf-8")
        ast.parse(content, filename=str(filepath))
        return True, "OK"
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (linha {e.lineno})"


def apply_fix_time_sleep(filepath: Path, line_num: int) -> bool:
    """Fix: time.sleep → await asyncio.sleep em função async."""
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    idx = line_num - 1
    if idx >= len(lines):
        return False
    
    line = lines[idx]
    if "time.sleep(" not in line:
        return False
    
    lines[idx] = line.replace("time.sleep(", "await asyncio.sleep(")
    
    # Verificar se precisa adicionar import asyncio
    has_asyncio_import = any(
        "import asyncio" in l for l in lines[:50]
    )
    if not has_asyncio_import:
        # Adicionar após os outros imports
        insert_at = 0
        for i, l in enumerate(lines):
            if l.strip().startswith(("import ", "from ")):
                insert_at = i + 1
            elif l.strip() and not l.strip().startswith("#") and insert_at > 0:
                break
        lines.insert(insert_at, "import asyncio")
    
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return True


def apply_fix_timeout(filepath: Path, line_num: int) -> bool:
    """Fix: Adicionar timeout em requests."""
    import re
    
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    idx = line_num - 1
    if idx >= len(lines):
        return False
    
    line = lines[idx]
    if "timeout" in line:
        return False  # Já tem timeout
    
    # Encontrar o fechamento do parêntese
    if line.rstrip().endswith(")"):
        # Simples: adicionar antes do último )
        lines[idx] = line.rstrip()[:-1] + ", timeout=10)"
    elif line.rstrip().endswith("),"):
        lines[idx] = line.rstrip()[:-2] + ", timeout=10),"
    else:
        # Chamada multi-linha, mais complexo
        return False
    
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return True


def apply_fix_except_pass(filepath: Path, line_num: int) -> bool:
    """Fix: except:pass → except com logging."""
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    idx = line_num - 1
    if idx >= len(lines) - 1:
        return False
    
    current = lines[idx].strip()
    next_line = lines[idx + 1].strip()
    
    if current not in ("except:", "except Exception:") or next_line != "pass":
        return False
    
    indent = len(lines[idx]) - len(lines[idx].lstrip())
    spaces = " " * indent
    inner_spaces = " " * (indent + 4)
    
    lines[idx] = f"{spaces}except Exception as _fix_err:"
    lines[idx + 1] = (
        f'{inner_spaces}import logging as _log; '
        f'_log.getLogger(__name__).error(f"Erro: {{_fix_err}}")'
    )
    
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return True


def apply_fix(issue: dict, dry_run: bool = True) -> dict:
    """
    Aplica um fix para um issue.
    Retorna resultado.
    """
    issue_id = issue["issue_id"]
    filepath = Path(issue["file_path"])
    line_num = issue["line_start"]
    
    result = {
        "issue_id": issue_id,
        "file": str(filepath),
        "line": line_num,
        "severity": issue["severity"],
        "title": issue["title"],
        "status": "skipped",
        "detail": "",
    }
    
    if not filepath.exists():
        result["status"] = "file_not_found"
        return result
    
    if dry_run:
        result["status"] = "would_fix"
        result["detail"] = issue.get("suggested_fix", "")
        return result
    
    # Backup
    backup_path = backup_file(filepath)
    result["backup"] = str(backup_path)
    
    # Aplicar fix baseado no tipo
    applied = False
    
    if issue_id.startswith("ASYNC-003"):
        applied = apply_fix_time_sleep(filepath, line_num)
    elif issue_id.startswith("API-001"):
        applied = apply_fix_timeout(filepath, line_num)
    elif issue_id.startswith("WS-002"):
        applied = apply_fix_except_pass(filepath, line_num)
    
    if not applied:
        result["status"] = "fix_not_applicable"
        return result
    
    # Validar sintaxe
    valid, msg = validate_syntax(filepath)
    if not valid:
        # Rollback
        shutil.copy2(backup_path, filepath)
        result["status"] = "rollback_syntax_error"
        result["detail"] = msg
        return result
    
    result["status"] = "applied"
    result["detail"] = "Fix aplicado e validado"
    return result


def log_result(result: dict):
    """Registra resultado no log."""
    FIX_LOG.parent.mkdir(parents=True, exist_ok=True)
    result["timestamp"] = datetime.now().isoformat()
    with open(FIX_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def rollback_last_batch():
    """Desfaz o último batch de fixes."""
    if not FIX_LOG.exists():
        print("❌ Nenhum log de fixes encontrado")
        return
    
    records = []
    with open(FIX_LOG, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    # Encontrar o último batch (mesmo timestamp prefix)
    applied = [r for r in records if r["status"] == "applied"]
    if not applied:
        print("❌ Nenhum fix aplicado encontrado")
        return
    
    # Pegar últimos 24h
    recent = applied[-50:]  # Últimos 50
    
    rolled_back = 0
    for record in reversed(recent):
        backup = record.get("backup")
        filepath = record.get("file")
        
        if backup and filepath and Path(backup).exists():
            shutil.copy2(backup, filepath)
            rolled_back += 1
            print(f"  ↩️ Restaurado: {filepath}")
    
    print(f"\n✅ {rolled_back} arquivos restaurados")


def main():
    parser = argparse.ArgumentParser(description="Aplicar fixes seguros")
    parser.add_argument(
        "--apply", action="store_true",
        help="Aplicar fixes de verdade (sem isso é dry-run)"
    )
    parser.add_argument(
        "--rollback", action="store_true",
        help="Desfazer último batch de fixes"
    )
    parser.add_argument(
        "--severity", "-s",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        help="Aplicar apenas para esta severidade"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=20,
        help="Máximo de fixes por execução"
    )
    
    args = parser.parse_args()
    
    if args.rollback:
        rollback_last_batch()
        return
    
    issues = load_fixable_issues()
    
    if args.severity:
        issues = [i for i in issues if i["severity"] == args.severity]
    
    issues = issues[:args.limit]
    
    mode = "APLICANDO" if args.apply else "DRY-RUN (simulação)"
    
    print("=" * 60)
    print(f"🔧 AUTO-FIXER — {mode}")
    print(f"   Issues fixáveis: {len(issues)}")
    print(f"   Confiança mínima: {MIN_CONFIDENCE:.0%}")
    print("=" * 60)
    
    stats = {"applied": 0, "skipped": 0, "error": 0, "would_fix": 0}
    
    for issue in issues:
        result = apply_fix(issue, dry_run=not args.apply)
        
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        
        icon = {
            "applied": "✅",
            "would_fix": "🔍",
            "rollback_syntax_error": "❌",
            "fix_not_applicable": "⏭️",
            "file_not_found": "❓",
            "skipped": "⏭️",
        }.get(status, "❓")
        
        print(
            f"  {icon} [{result['severity']}] "
            f"{result['title'][:50]}  → {status}"
        )
        
        if args.apply:
            log_result(result)
    
    print(f"\n{'=' * 60}")
    print(f"📊 Resultado:")
    for status, count in stats.items():
        if count > 0:
            print(f"   {status}: {count}")
    
    if not args.apply:
        print(f"\n💡 Para aplicar de verdade:")
        print(f"   python -m auto_fixer.apply_safe_fixes --apply")
        print(f"\n💡 Para desfazer:")
        print(f"   python -m auto_fixer.apply_safe_fixes --rollback")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
