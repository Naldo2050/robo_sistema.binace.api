"""
Health Monitor — Monitora saúde geral do sistema.
Integra File Watcher + Log Watcher + Auto-Fixer.
"""

import time
import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from auto_fixer.monitor.file_watcher import FileWatcher
from auto_fixer.monitor.log_watcher import LogWatcher

logger = logging.getLogger("health_monitor")


@dataclass
class SystemHealth:
    """Estado de saúde do sistema."""
    timestamp: str
    status: str  # "healthy", "degraded", "critical"
    files_monitored: int
    recent_file_changes: int
    syntax_errors: int
    log_alerts: int
    critical_issues: int
    auto_fixes_applied: int
    uptime_seconds: float


class HealthMonitor:
    """Monitor central de saúde."""

    def __init__(
        self,
        watch_dir: str = ".",
        check_interval: int = 5,
        auto_fix: bool = False,
    ):
        self.watch_dir = watch_dir
        self.start_time = time.time()
        self.auto_fix = auto_fix

        # Sub-monitores
        self.file_watcher = FileWatcher(
            watch_dir=watch_dir,
            check_interval=check_interval,
            auto_fix=auto_fix,
        )
        self.log_watcher = LogWatcher(
            check_interval=check_interval,
        )

        # Estado
        self.health_history: list[dict] = []
        self.running = True

    def start(self):
        """Inicia todos os monitores."""
        print("=" * 60)
        print("🏥 HEALTH MONITOR — Sistema de Monitoramento")
        print("=" * 60)
        print(f"  Diretório:  {self.watch_dir}")
        print(f"  Auto-fix:   {'Sim' if self.auto_fix else 'Não'}")
        print(f"  Timestamp:  {datetime.now().isoformat()}")
        print()
        print("  Monitores ativos:")
        print("    📁 File Watcher — Mudanças em arquivos")
        print("    📋 Log Watcher  — Erros nos logs")
        print("    🔧 Auto-Fixer   — Análise de código")
        print()
        print("  Pressione Ctrl+C para parar")
        print("=" * 60)
        print()

        # Iniciar file watcher em thread separada
        file_thread = threading.Thread(
            target=self.file_watcher.start,
            daemon=True,
            name="file_watcher"
        )
        file_thread.start()

        # Iniciar log watcher em thread separada
        log_thread = threading.Thread(
            target=self.log_watcher.start,
            daemon=True,
            name="log_watcher"
        )
        log_thread.start()

        # Loop principal — coleta saúde periodicamente
        try:
            while self.running:
                health = self._collect_health()
                self._report_health(health)
                time.sleep(30)  # Report a cada 30s
        except KeyboardInterrupt:
            logger.info("\n⏹️ Health Monitor parado")
            self.file_watcher.stop()
            self.log_watcher.running = False

    def _collect_health(self) -> SystemHealth:
        """Coleta estado de saúde."""
        file_stats = self.file_watcher.get_stats()
        recent_changes = self.file_watcher.get_recent_changes(20)
        log_summary = self.log_watcher.get_summary()

        # Contar problemas
        syntax_errors = sum(
            1 for c in recent_changes if not c.get("syntax_valid", True)
        )
        critical_from_logs = sum(
            1 for a in log_summary.get("recent", [])
            if a.get("level") == "CRITICAL"
        )

        # Determinar status
        if syntax_errors > 0 or critical_from_logs > 0:
            status = "critical"
        elif log_summary.get("total_alerts", 0) > 10:
            status = "degraded"
        else:
            status = "healthy"

        health = SystemHealth(
            timestamp=datetime.now().isoformat(),
            status=status,
            files_monitored=file_stats.get("files_monitored", 0),
            recent_file_changes=len(recent_changes),
            syntax_errors=syntax_errors,
            log_alerts=log_summary.get("total_alerts", 0),
            critical_issues=critical_from_logs + syntax_errors,
            auto_fixes_applied=0,
            uptime_seconds=time.time() - self.start_time,
        )

        self.health_history.append(asdict(health))
        # Manter últimas 1000
        self.health_history = self.health_history[-1000:]

        # Salvar
        self._save_health(health)

        return health

    def _report_health(self, health: SystemHealth):
        """Reporta estado de saúde."""
        status_icons = {
            "healthy": "💚",
            "degraded": "🟡",
            "critical": "🔴",
        }
        icon = status_icons.get(health.status, "❓")

        uptime_min = health.uptime_seconds / 60

        logger.info(
            f"{icon} Status: {health.status.upper()} | "
            f"Arquivos: {health.files_monitored} | "
            f"Mudanças: {health.recent_file_changes} | "
            f"Alertas: {health.log_alerts} | "
            f"Uptime: {uptime_min:.0f}min"
        )

    def _save_health(self, health: SystemHealth):
        """Salva estado de saúde."""
        output_dir = Path("auto_fixer/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(
            output_dir / "system_health.json", "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(health), f, indent=2, ensure_ascii=False)

    def get_health(self) -> dict:
        """Retorna saúde atual (para API)."""
        if self.health_history:
            return self.health_history[-1]
        return {"status": "unknown"}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Health Monitor — Sistema de Monitoramento"
    )
    parser.add_argument("--watch-dir", default=".")
    parser.add_argument("--auto-fix", action="store_true")
    parser.add_argument("--interval", type=int, default=5)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    monitor = HealthMonitor(
        watch_dir=args.watch_dir,
        auto_fix=args.auto_fix,
        check_interval=args.interval,
    )
    monitor.start()


if __name__ == "__main__":
    main()
