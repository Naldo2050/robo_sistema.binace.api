# -*- coding: utf-8 -*-
"""
Monitor de Erros 24h — Coletor Automatico para Trading Bot.

Captura ERROR/WARNING/CRITICAL/Exception em tempo real,
salva JSON estruturado + TXT legivel para analise posterior.

Uso:
    # Modo 1: Pipe (stdin)
    python main.py 2>&1 | python scripts/error_monitor.py

    # Modo 2: Arquivo de log
    python scripts/error_monitor.py --file logs/bot_output.log

    # Modo 3: Integrado (roda bot + monitor)
    python scripts/error_monitor.py --run-bot
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Classificacao de erros
# ---------------------------------------------------------------------------

_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("DATA_QUALITY_ALERT", re.compile(r"DATA_QUALITY", re.IGNORECASE)),
    ("ML_SKIP",            re.compile(r"ML_SKIP|ML_EXTREME|ml_warning|warmup", re.IGNORECASE)),
    ("FRED_FALLBACK",      re.compile(r"FRED|fallback", re.IGNORECASE)),
    ("OCI_DESATIVADO",     re.compile(r"OCI", re.IGNORECASE)),
    ("INVARIANTE_VIOLADA", re.compile(r"INVARIANTE", re.IGNORECASE)),
    ("PAYLOAD_TRIPWIRE",   re.compile(r"PAYLOAD_TRIPWIRE|TRIPWIRE", re.IGNORECASE)),
    ("TIMEOUT",            re.compile(r"Timeout|timed?\s*out", re.IGNORECASE)),
    ("CONNECTION",         re.compile(r"Connection|WebSocket|reconnect", re.IGNORECASE)),
    ("EXCEPTION",          re.compile(r"Exception|Traceback|Error(?::|$)", re.IGNORECASE)),
]

_SEVERITY_RE = [
    ("CRITICAL", re.compile(r"\bCRITICAL\b")),
    ("CRITICAL", re.compile(r"Traceback \(most recent call last\)")),
    ("ERROR",    re.compile(r"\bERROR\b")),
    ("WARNING",  re.compile(r"\bWARNING\b")),
]

# Tenta extrair timestamp no formato: 2026-03-21 14:05:13
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
# Tenta extrair numero de janela: janela N, window N, W=N
_WINDOW_RE = re.compile(r"(?:janela|window|W=)\s*(\d+)", re.IGNORECASE)


def classify_severity(line: str) -> str | None:
    for sev, pat in _SEVERITY_RE:
        if pat.search(line):
            return sev
    if "Exception" in line:
        return "CRITICAL"
    return None


def classify_type(line: str) -> str:
    for name, pat in _TYPE_PATTERNS:
        if pat.search(line):
            return name
    return "OTHER"


def extract_timestamp(line: str) -> str:
    m = _TS_RE.search(line)
    if m:
        return m.group(1)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def extract_window(line: str) -> int | None:
    m = _WINDOW_RE.search(line)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Estado do monitor
# ---------------------------------------------------------------------------

class ErrorMonitor:
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session_start = datetime.now(timezone.utc)
        self.date_str = self.session_start.strftime("%Y-%m-%d")

        # Contadores
        self.total_errors = 0
        self.total_warnings = 0
        self.total_critical = 0
        self.total_lines = 0

        # Cronologico
        self.events: list[dict] = []

        # Agrupado por tipo
        self.by_type: dict[str, dict] = {}

        # Multiline traceback buffer
        self._tb_buffer: list[str] = []
        self._in_traceback = False

        # Lock para thread-safety
        self._lock = threading.Lock()

        # Controle de flush
        self._last_flush = time.time()
        self._last_summary = time.time()
        self._dirty = False

    # -- Processamento de linha ------------------------------------------

    def process_line(self, line: str) -> None:
        line = line.rstrip("\n\r")
        if not line:
            return

        self.total_lines += 1

        # Traceback multiline
        if "Traceback (most recent call last)" in line:
            self._in_traceback = True
            self._tb_buffer = [line]
            return

        if self._in_traceback:
            self._tb_buffer.append(line)
            # Fim do traceback: linha que nao comeca com espaco e contem ":"
            if not line.startswith(" ") and not line.startswith("\t") and ":" in line:
                self._in_traceback = False
                full = "\n".join(self._tb_buffer)
                self._record_event(full, "CRITICAL", "EXCEPTION")
                self._tb_buffer = []
            return

        severity = classify_severity(line)
        if severity is None:
            return

        etype = classify_type(line)
        self._record_event(line, severity, etype)

    def _record_event(self, line: str, severity: str, etype: str) -> None:
        ts = extract_timestamp(line)
        window = extract_window(line)

        with self._lock:
            # Contadores
            if severity == "ERROR":
                self.total_errors += 1
            elif severity == "WARNING":
                self.total_warnings += 1
            elif severity == "CRITICAL":
                self.total_critical += 1

            count = self.total_errors + self.total_warnings + self.total_critical

            event = {
                "timestamp": ts,
                "severity": severity,
                "type": etype,
                "message": _clean_message(line),
                "raw_line": line[:500],
                "window": window,
                "count_so_far": count,
            }
            self.events.append(event)

            # Agrupado
            if etype not in self.by_type:
                self.by_type[etype] = {
                    "count": 0,
                    "severity": severity,
                    "first_seen": ts,
                    "last_seen": ts,
                    "sample_message": _clean_message(line),
                }
            grp = self.by_type[etype]
            grp["count"] += 1
            grp["last_seen"] = ts
            # Escalar severidade
            if _sev_rank(severity) > _sev_rank(grp["severity"]):
                grp["severity"] = severity

            self._dirty = True

    # -- Flush periodico -------------------------------------------------

    def maybe_flush(self) -> None:
        now = time.time()
        if self._dirty and (now - self._last_flush) >= 60:
            self.flush()
        if (now - self._last_summary) >= 600:
            self.print_summary()
            self._last_summary = now

    def flush(self) -> None:
        with self._lock:
            self._write_json()
            self._write_txt()
            self._dirty = False
            self._last_flush = time.time()

    # -- JSON output -----------------------------------------------------

    def _write_json(self) -> None:
        duration_h = (datetime.now(timezone.utc) - self.session_start).total_seconds() / 3600
        data = {
            "session_start": self.session_start.isoformat(),
            "session_duration_hours": round(duration_h, 2),
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "total_critical": self.total_critical,
            "total_lines_processed": self.total_lines,
            "summary_by_type": dict(
                sorted(self.by_type.items(), key=lambda x: x[1]["count"], reverse=True)
            ),
            "errors_chronological": self.events,
            "unique_errors": self._unique_errors(),
        }
        path = self.output_dir / f"error_monitor_{self.date_str}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _unique_errors(self) -> list[dict]:
        result = []
        for etype, grp in sorted(self.by_type.items(), key=lambda x: x[1]["count"], reverse=True):
            result.append({
                "type": etype,
                "first_occurrence": grp["first_seen"],
                "last_occurrence": grp["last_seen"],
                "total_occurrences": grp["count"],
                "sample_message": grp["sample_message"],
                "severity": grp["severity"],
            })
        return result

    # -- TXT output ------------------------------------------------------

    def _write_txt(self) -> None:
        duration_h = (datetime.now(timezone.utc) - self.session_start).total_seconds() / 3600
        path = self.output_dir / f"error_monitor_{self.date_str}.txt"
        lines = []
        lines.append("=" * 50)
        lines.append(f"MONITOR DE ERROS -- Sessao {self.date_str}")
        lines.append("=" * 50)
        lines.append(
            f"Inicio: {self.session_start.strftime('%H:%M:%S')} | "
            f"Duracao: {duration_h:.1f}h"
        )
        lines.append(
            f"Errors: {self.total_errors} | "
            f"Warnings: {self.total_warnings} | "
            f"Critical: {self.total_critical}"
        )
        lines.append(f"Linhas processadas: {self.total_lines}")
        lines.append("")

        # Resumo por tipo
        lines.append("RESUMO POR TIPO:")
        for etype, grp in sorted(self.by_type.items(), key=lambda x: x[1]["count"], reverse=True):
            icon = _severity_icon(grp["severity"])
            lines.append(f"  {icon} {etype:<25} x{grp['count']:<4} ({grp['severity']})")
        lines.append("")

        # Erros unicos
        lines.append("ERROS UNICOS (para correcao):")
        for i, u in enumerate(self._unique_errors(), 1):
            lines.append(f"  {i}. {u['type']} -- {u['total_occurrences']} vezes")
            lines.append(f"     \"{u['sample_message'][:100]}\"")
            lines.append(f"     Primeira: {u['first_occurrence']} | Ultima: {u['last_occurrence']}")
            lines.append("")

        # Cronologico
        lines.append("CRONOLOGICO (todos):")
        for ev in self.events:
            ts_short = ev["timestamp"].split(" ")[-1] if " " in ev["timestamp"] else ev["timestamp"]
            lines.append(
                f"  [{ts_short}] {ev['severity']:<8} {ev['type']:<25} "
                f"{ev['message'][:80]}"
            )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # -- Console summary -------------------------------------------------

    def print_summary(self) -> None:
        duration_h = (datetime.now(timezone.utc) - self.session_start).total_seconds() / 3600
        print(flush=True)
        print("=" * 50, flush=True)
        print(
            f"[MONITOR] {datetime.now().strftime('%H:%M:%S')} | "
            f"Duracao: {duration_h:.1f}h | "
            f"Linhas: {self.total_lines}",
            flush=True,
        )
        print(
            f"  Errors: {self.total_errors} | "
            f"Warnings: {self.total_warnings} | "
            f"Critical: {self.total_critical}",
            flush=True,
        )
        if self.by_type:
            top = sorted(self.by_type.items(), key=lambda x: x[1]["count"], reverse=True)[:5]
            for etype, grp in top:
                print(f"  {_severity_icon(grp['severity'])} {etype}: {grp['count']}", flush=True)
        print("=" * 50, flush=True)
        print(flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_message(line: str) -> str:
    """Remove prefixos de timestamp e nivel de log."""
    # Remove: "2026-03-21 14:05:13 [ERROR] module: "
    cleaned = re.sub(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[[\w]+\]\s*[\w.]*:\s*", "", line)
    # Remove emojis comuns
    cleaned = re.sub(r"[^\x00-\x7F]+\s*", "", cleaned)
    return cleaned.strip()[:200]


def _sev_rank(sev: str) -> int:
    return {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}.get(sev, -1)


def _severity_icon(sev: str) -> str:
    return {"CRITICAL": "[!!!]", "ERROR": "[ERR]", "WARNING": "[WRN]", "INFO": "[INF]"}.get(sev, "[---]")


# ---------------------------------------------------------------------------
# Modos de execucao
# ---------------------------------------------------------------------------

def run_pipe(monitor: ErrorMonitor) -> None:
    """Modo 1: Leitura de stdin (pipe)."""
    print(f"[MONITOR] Lendo stdin (pipe)... Ctrl+C para parar.", flush=True)
    try:
        for line in sys.stdin:
            monitor.process_line(line)
            monitor.maybe_flush()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.flush()
        monitor.print_summary()
        print(f"[MONITOR] Finalizado. Arquivos salvos em {monitor.output_dir}/", flush=True)


def run_file(monitor: ErrorMonitor, filepath: str) -> None:
    """Modo 2: Leitura de arquivo (tail -f)."""
    path = Path(filepath)
    if not path.exists():
        print(f"[MONITOR] Aguardando arquivo {filepath}...", flush=True)
        while not path.exists():
            time.sleep(1)

    print(f"[MONITOR] Monitorando {filepath}... Ctrl+C para parar.", flush=True)

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            # Ir para o final se o arquivo ja existe
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    monitor.process_line(line)
                    monitor.maybe_flush()
                else:
                    time.sleep(0.5)
                    monitor.maybe_flush()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.flush()
        monitor.print_summary()
        print(f"[MONITOR] Finalizado. Arquivos salvos em {monitor.output_dir}/", flush=True)


def run_bot(monitor: ErrorMonitor) -> None:
    """Modo 3: Roda o bot como subprocess e monitora."""
    print("[MONITOR] Iniciando bot (python main.py) + monitor...", flush=True)

    proc = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Tambem salva log completo do bot
    bot_log_path = monitor.output_dir / f"bot_output_{monitor.date_str}.log"

    try:
        with open(bot_log_path, "w", encoding="utf-8") as bot_log:
            for line in proc.stdout:
                # Escreve no console
                sys.stdout.write(line)
                sys.stdout.flush()
                # Salva log completo
                bot_log.write(line)
                bot_log.flush()
                # Processa no monitor
                monitor.process_line(line)
                monitor.maybe_flush()
    except KeyboardInterrupt:
        print("\n[MONITOR] Ctrl+C detectado. Encerrando bot...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        monitor.flush()
        monitor.print_summary()
        print(f"[MONITOR] Finalizado. Arquivos salvos em {monitor.output_dir}/", flush=True)
        print(f"[MONITOR] Log completo do bot: {bot_log_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor de Erros 24h para Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  python main.py 2>&1 | python scripts/error_monitor.py\n"
            "  python scripts/error_monitor.py --file logs/bot_output.log\n"
            "  python scripts/error_monitor.py --run-bot\n"
        ),
    )
    parser.add_argument("--file", "-f", help="Arquivo de log para monitorar (modo tail -f)")
    parser.add_argument("--run-bot", action="store_true", help="Rodar bot + monitor integrado")
    parser.add_argument("--output-dir", "-o", default="logs", help="Diretorio de saida (default: logs)")
    args = parser.parse_args()

    monitor = ErrorMonitor(output_dir=args.output_dir)

    print(f"[MONITOR] Sessao iniciada: {monitor.session_start.isoformat()}", flush=True)
    print(f"[MONITOR] Saida: {monitor.output_dir}/error_monitor_{monitor.date_str}.[json|txt]", flush=True)

    if args.run_bot:
        run_bot(monitor)
    elif args.file:
        run_file(monitor, args.file)
    else:
        # Default: pipe (stdin)
        if sys.stdin.isatty():
            print("[MONITOR] Nenhum pipe detectado. Use --file ou --run-bot.", flush=True)
            parser.print_help()
            sys.exit(1)
        run_pipe(monitor)


if __name__ == "__main__":
    main()
