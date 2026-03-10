"""
Log Watcher — Monitora logs do bot em tempo real.
Detecta erros, crashes e anomalias.
"""

import re
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import deque, Counter
from typing import Optional

logger = logging.getLogger("log_watcher")


@dataclass
class LogAlert:
    """Alerta gerado a partir dos logs."""
    timestamp: str
    level: str  # ERROR, CRITICAL, WARNING
    source_file: str
    line_number: int
    message: str
    category: str  # "crash", "api_error", "timeout", "memory", "connection"
    count: int = 1  # Quantas vezes repetiu


# Padrões de erro conhecidos
ERROR_PATTERNS = {
    "crash": [
        r"Traceback \(most recent call last\)",
        r"Fatal error",
        r"SystemExit",
        r"Process died",
    ],
    "api_error": [
        r"BinanceAPIException",
        r"APIError",
        r"rate limit",
        r"429\s",
        r"Too Many Requests",
        r"API key",
    ],
    "timeout": [
        r"TimeoutError",
        r"asyncio\.TimeoutError",
        r"ReadTimeout",
        r"ConnectTimeout",
        r"Connection timed out",
    ],
    "connection": [
        r"ConnectionError",
        r"ConnectionRefusedError",
        r"WebSocket.*closed",
        r"WebSocket.*error",
        r"Disconnected",
        r"reconnect",
    ],
    "memory": [
        r"MemoryError",
        r"OOM",
        r"out of memory",
        r"memory usage",
    ],
    "data_error": [
        r"KeyError",
        r"ValueError",
        r"TypeError",
        r"IndexError",
        r"JSONDecodeError",
    ],
}


class LogWatcher:
    """Monitora arquivos de log em tempo real."""

    def __init__(
        self,
        log_dirs: Optional[list[str]] = None,
        check_interval: int = 3,
        alert_threshold: int = 5,  # Alertar após N erros do mesmo tipo
    ):
        if log_dirs is None:
            log_dirs = ["logs/", "dados/", "."]

        self.log_dirs = [Path(d) for d in log_dirs]
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold

        # Estado
        self.file_positions: dict[str, int] = {}  # arquivo → posição lida
        self.recent_alerts: deque[LogAlert] = deque(maxlen=200)
        self.error_counter: Counter = Counter()
        self.running = True

        # Padrões compilados
        self.compiled_patterns: dict[str, list[re.Pattern]] = {}
        for category, patterns in ERROR_PATTERNS.items():
            self.compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def start(self):
        """Inicia o monitoramento de logs."""
        logger.info("📋 Log Watcher iniciado")

        log_files = self._find_log_files()
        logger.info(f"   Monitorando {len(log_files)} arquivos de log")

        # Posicionar no final dos arquivos
        for log_file in log_files:
            try:
                self.file_positions[str(log_file)] = log_file.stat().st_size
            except Exception:
                self.file_positions[str(log_file)] = 0

        try:
            while self.running:
                self._check_logs()
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("\n⏹️ Log Watcher parado")

    def _check_logs(self):
        """Verifica novos conteúdos nos logs."""
        for log_file in self._find_log_files():
            str_path = str(log_file)

            try:
                current_size = log_file.stat().st_size
                last_position = self.file_positions.get(str_path, 0)

                if current_size <= last_position:
                    if current_size < last_position:
                        # Log foi rotacionado
                        last_position = 0
                    else:
                        continue

                # Ler novas linhas
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(last_position)
                    new_content = f.read()

                self.file_positions[str_path] = current_size

                # Analisar novas linhas
                if new_content.strip():
                    self._analyze_content(new_content, str_path)

            except Exception as e:
                logger.warning(f"Erro ao ler {log_file}: {e}")

    def _analyze_content(self, content: str, source: str):
        """Analisa conteúdo novo do log."""
        lines = content.splitlines()

        for i, line in enumerate(lines):
            for category, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(line):
                        alert = LogAlert(
                            timestamp=datetime.now().isoformat(),
                            level=self._detect_level(line),
                            source_file=source,
                            line_number=i,
                            message=line.strip()[:200],
                            category=category,
                        )
                        self.recent_alerts.append(alert)
                        self.error_counter[category] += 1

                        # Log do alerta
                        level_icon = {
                            "CRITICAL": "🔴",
                            "ERROR": "🟠",
                            "WARNING": "🟡",
                        }.get(alert.level, "ℹ️")

                        logger.warning(
                            f"{level_icon} [{category}] {alert.message[:80]}"
                        )

                        # Verificar threshold
                        if self.error_counter[category] >= self.alert_threshold:
                            self._trigger_alert(category)
                            self.error_counter[category] = 0

                        break  # Um match por linha basta

    def _detect_level(self, line: str) -> str:
        """Detecta o nível do log."""
        upper = line.upper()
        if "CRITICAL" in upper or "FATAL" in upper:
            return "CRITICAL"
        elif "ERROR" in upper:
            return "ERROR"
        elif "WARNING" in upper or "WARN" in upper:
            return "WARNING"
        return "ERROR"

    def _trigger_alert(self, category: str):
        """Dispara alerta quando threshold é atingido."""
        logger.critical(
            f"⚠️ ALERTA: {self.alert_threshold}+ erros de "
            f"'{category}' detectados!"
        )

        # Salvar alerta
        alert_dir = Path("auto_fixer/output/alerts")
        alert_dir.mkdir(parents=True, exist_ok=True)

        alert = {
            "type": "LOG_THRESHOLD",
            "category": category,
            "count": self.alert_threshold,
            "timestamp": datetime.now().isoformat(),
            "recent_messages": [
                asdict(a) for a in list(self.recent_alerts)[-5:]
                if a.category == category
            ],
        }

        with open(
            alert_dir / "active_alerts.jsonl", "a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")

    def _find_log_files(self) -> list[Path]:
        """Encontra arquivos de log."""
        log_files = []
        extensions = {".log", ".jsonl"}

        for log_dir in self.log_dirs:
            if not log_dir.exists():
                continue
            for f in log_dir.iterdir():
                if f.suffix in extensions and f.is_file():
                    log_files.append(f)

        return log_files

    def get_summary(self) -> dict:
        """Resumo dos alertas."""
        return {
            "total_alerts": len(self.recent_alerts),
            "by_category": dict(self.error_counter),
            "recent": [
                asdict(a) for a in list(self.recent_alerts)[-10:]
            ],
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    watcher = LogWatcher()
    watcher.start()
