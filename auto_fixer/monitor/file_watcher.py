"""
File Watcher — Monitor de arquivos em tempo real.
Detecta mudanças, analisa automaticamente e alerta sobre problemas.

Uso:
  python -m auto_fixer.monitor.file_watcher
  python -m auto_fixer.monitor.file_watcher --watch-dir .
  python -m auto_fixer.monitor.file_watcher --auto-fix
"""

import sys
import os

# Fix para Windows (UTF-8)
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import os
import sys
import ast
import time
import json
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import deque

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("file_watcher")

# Diretórios ignorados
IGNORE_DIRS = {
    "__pycache__", ".git", "node_modules", "venv",
    ".venv", "backups", ".mypy_cache", ".pytest_cache",
    "features", "auto_fixer/output", "logs",
}

# Extensões monitoradas
WATCH_EXTENSIONS = {".py"}


@dataclass
class FileChange:
    """Registro de mudança em arquivo."""
    timestamp: str
    file_path: str
    change_type: str  # "modified", "created", "deleted"
    old_hash: str
    new_hash: str
    size_bytes: int
    syntax_valid: bool
    issues_found: int = 0
    issues_critical: int = 0
    auto_fixed: bool = False


@dataclass
class WatchState:
    """Estado do monitoramento."""
    file_hashes: dict[str, str] = field(default_factory=dict)
    file_sizes: dict[str, int] = field(default_factory=dict)
    file_mtimes: dict[str, float] = field(default_factory=dict)
    last_scan: str = ""
    changes_history: list[dict] = field(default_factory=list)


class FileWatcher:
    """Monitor de arquivos em tempo real."""

    def __init__(
        self,
        watch_dir: str = ".",
        state_file: str = "auto_fixer/output/watch_state.json",
        check_interval: int = 5,
        auto_fix: bool = False,
        auto_scan: bool = True,
    ):
        self.watch_dir = Path(watch_dir).resolve()
        self.state_file = Path(state_file)
        self.check_interval = check_interval
        self.auto_fix = auto_fix
        self.auto_scan = auto_scan

        self.state = self._load_state()
        self.recent_changes: deque[FileChange] = deque(maxlen=100)
        self.running = True

        # Callbacks
        self.on_change_callbacks: list = []
        self.on_error_callbacks: list = []

    def start(self):
        """Inicia o monitoramento."""
        logger.info(f"🔍 File Watcher iniciado")
        logger.info(f"   Diretório: {self.watch_dir}")
        logger.info(f"   Intervalo: {self.check_interval}s")
        logger.info(f"   Auto-fix: {'✅' if self.auto_fix else '❌'}")
        logger.info(f"   Auto-scan: {'✅' if self.auto_scan else '❌'}")

        # Scan inicial
        if not self.state.file_hashes:
            logger.info("📊 Primeiro scan — indexando arquivos...")
            self._full_scan()
            self._save_state()
            logger.info(
                f"   {len(self.state.file_hashes)} arquivos indexados"
            )

        # Loop principal
        try:
            while self.running:
                changes = self._check_changes()

                if changes:
                    self._handle_changes(changes)

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\n⏹️ File Watcher parado pelo usuário")
            self._save_state()

    def stop(self):
        """Para o monitoramento."""
        self.running = False

    def _full_scan(self):
        """Scan completo de todos os arquivos."""
        for py_file in self._find_python_files():
            try:
                rel_path = str(py_file.relative_to(self.watch_dir))
                content = py_file.read_bytes()
                file_hash = hashlib.md5(content).hexdigest()
                file_size = len(content)
                file_mtime = py_file.stat().st_mtime

                self.state.file_hashes[rel_path] = file_hash
                self.state.file_sizes[rel_path] = file_size
                self.state.file_mtimes[rel_path] = file_mtime
            except Exception as e:
                logger.warning(f"Erro ao indexar {py_file}: {e}")

        self.state.last_scan = datetime.now().isoformat()

    def _check_changes(self) -> list[FileChange]:
        """Verifica mudanças desde o último check."""
        changes: list[FileChange] = []
        current_files: set[str] = set()

        for py_file in self._find_python_files():
            try:
                rel_path = str(py_file.relative_to(self.watch_dir))
                current_files.add(rel_path)

                # Verificar mtime primeiro (mais rápido que hash)
                current_mtime = py_file.stat().st_mtime
                stored_mtime = self.state.file_mtimes.get(rel_path, 0)

                if current_mtime <= stored_mtime:
                    continue  # Não mudou

                # mtime mudou — verificar hash
                content = py_file.read_bytes()
                new_hash = hashlib.md5(content).hexdigest()
                old_hash = self.state.file_hashes.get(rel_path, "")

                if new_hash == old_hash:
                    # mtime mudou mas conteúdo não (ex: touch)
                    self.state.file_mtimes[rel_path] = current_mtime
                    continue

                # Arquivo realmente mudou
                change_type = "modified" if old_hash else "created"

                # Validar sintaxe
                text = content.decode("utf-8", errors="replace")
                syntax_valid = self._check_syntax(text, rel_path)

                change = FileChange(
                    timestamp=datetime.now().isoformat(),
                    file_path=rel_path,
                    change_type=change_type,
                    old_hash=old_hash,
                    new_hash=new_hash,
                    size_bytes=len(content),
                    syntax_valid=syntax_valid,
                )

                changes.append(change)

                # Atualizar estado
                self.state.file_hashes[rel_path] = new_hash
                self.state.file_sizes[rel_path] = len(content)
                self.state.file_mtimes[rel_path] = current_mtime

            except Exception as e:
                logger.warning(f"Erro ao verificar {py_file}: {e}")

        # Verificar arquivos deletados
        stored_files = set(self.state.file_hashes.keys())
        deleted_files = stored_files - current_files

        for deleted in deleted_files:
            changes.append(FileChange(
                timestamp=datetime.now().isoformat(),
                file_path=deleted,
                change_type="deleted",
                old_hash=self.state.file_hashes.get(deleted, ""),
                new_hash="",
                size_bytes=0,
                syntax_valid=True,
            ))
            # Remover do estado
            self.state.file_hashes.pop(deleted, None)
            self.state.file_sizes.pop(deleted, None)
            self.state.file_mtimes.pop(deleted, None)

        return changes

    def _handle_changes(self, changes: list[FileChange]):
        """Processa mudanças detectadas."""
        for change in changes:
            self.recent_changes.append(change)

            # Ícones por tipo
            icons = {
                "modified": "📝",
                "created": "🆕",
                "deleted": "🗑️",
            }
            icon = icons.get(change.change_type, "❓")

            # Status da sintaxe
            syntax_icon = "✅" if change.syntax_valid else "❌ SYNTAX ERROR"

            logger.info(
                f"{icon} {change.change_type.upper()}: "
                f"{change.file_path} {syntax_icon}"
            )

            # Se tem erro de sintaxe, alertar
            if not change.syntax_valid:
                logger.error(
                    f"⚠️ ERRO DE SINTAXE em {change.file_path}!"
                )
                self._alert_syntax_error(change)

            # Se auto-scan ativo, analisar o arquivo
            if self.auto_scan and change.change_type != "deleted":
                issues = self._quick_analyze(change.file_path)
                change.issues_found = len(issues)
                change.issues_critical = sum(
                    1 for i in issues
                    if i.get("severity") == "CRITICAL"
                )

                if issues:
                    logger.info(
                        f"   → {len(issues)} issues encontrados"
                        f" ({change.issues_critical} critical)"
                    )

            # Callbacks
            for callback in self.on_change_callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error(f"Erro no callback: {e}")

        # Salvar estado
        self.state.changes_history.extend(
            [asdict(c) for c in changes]
        )
        # Manter últimas 500 mudanças
        self.state.changes_history = self.state.changes_history[-500:]
        self._save_state()

    def _quick_analyze(self, file_path: str) -> list[dict]:
        """Análise rápida de um arquivo (sem scan completo)."""
        issues: list[dict] = []
        full_path = self.watch_dir / file_path

        if not full_path.exists():
            return issues

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            for i, line in enumerate(lines, 1):
                stripped = line.strip()

                # time.sleep em async
                if "time.sleep(" in stripped:
                    # Verificar se está em função async
                    for j in range(max(0, i - 30), i):
                        if j < len(lines) and lines[j].strip().startswith("async def"):
                            issues.append({
                                "severity": "CRITICAL",
                                "title": "time.sleep() em async",
                                "file": file_path,
                                "line": i,
                            })
                            break

                # requests síncronos em async
                if "requests.get(" in stripped or "requests.post(" in stripped:
                    for j in range(max(0, i - 30), i):
                        if j < len(lines) and lines[j].strip().startswith("async def"):
                            issues.append({
                                "severity": "CRITICAL",
                                "title": "requests síncronos em async",
                                "file": file_path,
                                "line": i,
                            })
                            break

                # except: pass
                if stripped == "except:" or stripped == "except Exception:":
                    if i < len(lines) and lines[i].strip() == "pass":
                        issues.append({
                            "severity": "HIGH",
                            "title": "except: pass (erro silencioso)",
                            "file": file_path,
                            "line": i,
                        })

        except Exception as e:
            logger.error(f"Erro na análise rápida de {file_path}: {e}")

        return issues

    def _check_syntax(self, source: str, filename: str) -> bool:
        """Verifica sintaxe Python."""
        try:
            ast.parse(source, filename=filename)
            return True
        except SyntaxError:
            return False

    def _alert_syntax_error(self, change: FileChange):
        """Gera alerta de erro de sintaxe."""
        alert_dir = Path("auto_fixer/output/alerts")
        alert_dir.mkdir(parents=True, exist_ok=True)

        alert = {
            "type": "SYNTAX_ERROR",
            "timestamp": change.timestamp,
            "file": change.file_path,
            "severity": "CRITICAL",
        }

        alert_file = alert_dir / "active_alerts.jsonl"
        with open(alert_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")

    def _find_python_files(self):
        """Encontra arquivos Python monitoráveis."""
        for root, dirs, files in os.walk(self.watch_dir):
            # Filtrar diretórios ignorados
            dirs[:] = [
                d for d in dirs
                if d not in IGNORE_DIRS
                and not d.startswith(".")
            ]

            rel_root = os.path.relpath(root, self.watch_dir)
            # Verificar se algum componente do path deve ser ignorado
            skip = False
            for part in Path(rel_root).parts:
                if part in IGNORE_DIRS:
                    skip = True
                    break
            if skip:
                continue

            for fname in files:
                if Path(fname).suffix in WATCH_EXTENSIONS:
                    yield Path(root) / fname

    def _load_state(self) -> WatchState:
        """Carrega estado salvo."""
        if self.state_file.exists():
            try:
                with open(self.state_file, encoding="utf-8") as f:
                    data = json.load(f)
                return WatchState(**data)
            except Exception:
                pass
        return WatchState()

    def _save_state(self):
        """Salva estado atual."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, indent=2, ensure_ascii=False)

    def get_recent_changes(self, limit: int = 20) -> list[dict]:
        """Retorna mudanças recentes."""
        return [asdict(c) for c in list(self.recent_changes)[-limit:]]

    def get_stats(self) -> dict:
        """Estatísticas do monitoramento."""
        return {
            "files_monitored": len(self.state.file_hashes),
            "total_changes_recorded": len(self.state.changes_history),
            "last_scan": self.state.last_scan,
            "recent_changes": len(self.recent_changes),
        }


def main():
    parser = argparse.ArgumentParser(
        description="File Watcher — Monitor de arquivos em tempo real"
    )
    parser.add_argument(
        "--watch-dir", default=".",
        help="Diretório para monitorar"
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Intervalo de checagem em segundos"
    )
    parser.add_argument(
        "--auto-fix", action="store_true",
        help="Aplicar fixes automaticamente"
    )
    parser.add_argument(
        "--no-scan", action="store_true",
        help="Desabilitar análise automática"
    )

    args = parser.parse_args()

    watcher = FileWatcher(
        watch_dir=args.watch_dir,
        check_interval=args.interval,
        auto_fix=args.auto_fix,
        auto_scan=not args.no_scan,
    )

    print("=" * 50)
    print("🔍 FILE WATCHER — Monitor de Arquivos")
    print("=" * 50)
    print(f"  Diretório: {watcher.watch_dir}")
    print(f"  Intervalo: {args.interval}s")
    print(f"  Auto-fix:  {'Sim' if args.auto_fix else 'Não'}")
    print(f"  Análise:   {'Sim' if not args.no_scan else 'Não'}")
    print()
    print("  Pressione Ctrl+C para parar")
    print("=" * 50)
    print()

    watcher.start()


if __name__ == "__main__":
    main()
