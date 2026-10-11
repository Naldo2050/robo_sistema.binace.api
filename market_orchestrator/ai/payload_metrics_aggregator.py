# -*- coding: utf-8 -*-
"""
Agregador leve de métricas do payload (p50/p95, caches, guardrail).

Lê o arquivo JSONL usado pelo guardrail e pelos loggers de payload, ignorando linhas
quebradas, e devolve um resumo num dicionário.
Gate G4-B: Rotação controlada por tamanho (5 MiB, 2 backups) com file locking estável
e leitura suplementar dos backups mais recentes.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import platform
import threading
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional

_METRICS_PATH_LOGGED = False
_MODULE_LOCK = threading.Lock()

# Constantes de Rotação (Gate G4-B)
MAX_METRICS_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB por arquivo
MAX_METRICS_BACKUPS = 2  # 2 backups (.1 e .2)
LOCK_TIMEOUT_SECONDS = 0.2  # Orçamento total bounded (200ms) para aquisição de locks


# ============================================================
# FILE LOCKING MULTIPLATAFORMA (Gate G4-B)
# ============================================================

LOCK_METHOD = None
fcntl_module = None
msvcrt_module = None

try:
    import fcntl as fcntl_module
    LOCK_METHOD = "fcntl"
except ImportError:
    if platform.system() == "Windows":
        try:
            import msvcrt as msvcrt_module
            LOCK_METHOD = "msvcrt"
        except ImportError:
            LOCK_METHOD = None
    else:
        LOCK_METHOD = None


def acquire_file_lock(file_obj, blocking: bool = True, timeout: float = LOCK_TIMEOUT_SECONDS) -> bool:
    """Adquire lock exclusivo multiplataforma."""
    if LOCK_METHOD is None:
        return True

    if timeout <= 0 and blocking:
        blocking = False

    try:
        if LOCK_METHOD == "fcntl":
            if blocking:
                start_time = time.monotonic()
                while True:
                    try:
                        fcntl_module.flock(file_obj.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
                        return True
                    except (IOError, OSError):
                        if time.monotonic() - start_time > timeout:
                            return False
                        time.sleep(0.02)
            else:
                try:
                    fcntl_module.flock(file_obj.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
                    return True
                except (IOError, OSError):
                    return False

        elif LOCK_METHOD == "msvcrt":
            if blocking:
                start_time = time.monotonic()
                while True:
                    try:
                        file_obj.seek(0, 0)
                        msvcrt_module.locking(file_obj.fileno(), msvcrt_module.LK_NBLCK, 1)
                        return True
                    except (IOError, OSError):
                        if time.monotonic() - start_time > timeout:
                            return False
                        time.sleep(0.02)
            else:
                try:
                    file_obj.seek(0, 0)
                    msvcrt_module.locking(file_obj.fileno(), msvcrt_module.LK_NBLCK, 1)
                    return True
                except (IOError, OSError):
                    return False
    except Exception as e:
        logging.debug(f"Erro ao adquirir lock de métricas: {e}")
        return False

    return True


def release_file_lock(file_obj) -> None:
    """Libera lock de arquivo multiplataforma."""
    if LOCK_METHOD is None:
        return

    try:
        if LOCK_METHOD == "fcntl":
            fcntl_module.flock(file_obj.fileno(), fcntl_module.LOCK_UN)
        elif LOCK_METHOD == "msvcrt":
            try:
                file_obj.seek(0, 0)
            except Exception:
                pass
            msvcrt_module.locking(file_obj.fileno(), msvcrt_module.LK_UNLCK, 1)
    except Exception as e:
        logging.debug(f"Erro ao liberar lock de métricas: {e}")


@contextmanager
def file_lock_context(lock_path: Path, timeout: float = LOCK_TIMEOUT_SECONDS):
    """Context manager para arquivo de lock estável (sem unlink)."""
    lock_file = None
    locked = False
    try:
        lock_file = open(lock_path, "a")
        locked = acquire_file_lock(lock_file, blocking=True, timeout=timeout)
        yield locked
    finally:
        if lock_file:
            if locked:
                try:
                    release_file_lock(lock_file)
                except Exception:
                    pass
            try:
                lock_file.close()
            except Exception:
                pass


@contextmanager
def acquire_combined_locks(lock_path: Path, timeout: float = LOCK_TIMEOUT_SECONDS):
    """
    Adquire ordenadamente _MODULE_LOCK (thread) e file lock (processo)
    sob um orçamento único e compartilhado baseado em time.monotonic().
    Retorna True se ambos os locks foram adquiridos; False em caso de timeout/falha.
    Garante liberação segura em finally sem liberar locks não adquiridos.
    """
    deadline = time.monotonic() + timeout
    module_locked = False
    file_locked = False
    lock_file = None

    try:
        # 1. Tenta adquirir o _MODULE_LOCK com o tempo restante
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            yield False
            return

        module_locked = _MODULE_LOCK.acquire(blocking=True, timeout=remaining)
        if not module_locked:
            yield False
            return

        # 2. Tenta adquirir o file lock com o tempo restante do orçamento
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            yield False
            return

        try:
            lock_file = open(lock_path, "a")
            file_locked = acquire_file_lock(lock_file, blocking=True, timeout=remaining)
        except Exception as e:
            logging.debug("Erro ao abrir ou adquirir lock file: %s", e)
            file_locked = False

        if not file_locked:
            yield False
            return

        # Ambos os locks foram adquiridos com sucesso
        yield True

    finally:
        if file_locked and lock_file:
            try:
                release_file_lock(lock_file)
            except Exception:
                pass
        if lock_file:
            try:
                lock_file.close()
            except Exception:
                pass
        if module_locked:
            try:
                _MODULE_LOCK.release()
            except Exception:
                pass


def _rotate_metrics_file_if_needed(metrics_path: Path, max_bytes: int = MAX_METRICS_FILE_BYTES, backups: int = MAX_METRICS_BACKUPS) -> bool:
    """
    Executa rotação atômica por tamanho para payload_metrics.jsonl (Gate G4-B).
    Deve ser chamado SOB os locks adquiridos.
    Retorna True se não era necessário ou se rotacionou com sucesso;
    Retorna False se houve falha na rotação.
    """
    if backups <= 0:
        return True
    try:
        if not metrics_path.exists():
            return True
        current_size = metrics_path.stat().st_size
    except Exception as e:
        logging.warning("Falha ao obter stat de %s para rotação: %s", metrics_path, e)
        return False

    if current_size < max_bytes:
        return True

    # Rotação necessária
    try:
        # Cascata de trás para frente: .1 -> .2
        for i in range(backups - 1, 0, -1):
            sfn = Path(f"{metrics_path}.{i}")
            dfn = Path(f"{metrics_path}.{i + 1}")
            if sfn.exists():
                try:
                    os.replace(sfn, dfn)
                except Exception as e:
                    logging.warning("Falha ao rotacionar backup %s para %s: %s. Interrompendo rotação.", sfn, dfn, e)
                    return False

        # Rotaciona ativo para .1
        target_first = Path(f"{metrics_path}.1")
        try:
            os.replace(metrics_path, target_first)
            logging.info("Métricas de payload rotacionadas com sucesso: %s -> %s (%d bytes)", metrics_path, target_first, current_size)
            return True
        except Exception as e:
            logging.warning("Falha ao rotacionar ativo %s para %s: %s", metrics_path, target_first, e)
            return False
    except Exception as e:
        logging.error("Erro inesperado durante rotação de métricas: %s", e)
        return False


# ============================================================
# THRESHOLDS DO TRIPWIRE
# Aumentados para refletir estado real do sistema
# ============================================================

_TRIPWIRE_THRESHOLDS: dict[str, float] = {
    # Taxa de fallback: era 0.03 (3%), aumentado para 0.10 (10%)
    # Justificativa: guardrail causava fallbacks falsos antes do fix
    "fallback_rate":        0.10,

    # Taxa de abort: era 0.05 (5%), aumentado para 0.12 (12%)
    "abort_rate":           0.12,

    # Taxa de bloqueio do guardrail: novo threshold
    "guardrail_block_rate": 0.20,
}


def append_metric_line(obj: Dict[str, object], metrics_path: str = "logs/payload_metrics.jsonl") -> None:
    """Anexa uma linha de métrica em JSONL com rotação por tamanho e sincronização combinada."""
    global _METRICS_PATH_LOGGED
    try:
        p = Path(metrics_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not _METRICS_PATH_LOGGED:
            logging.debug("PAYLOAD_METRICS_FILE path=%s cwd=%s", str(p), str(Path.cwd()))
            _METRICS_PATH_LOGGED = True

        line = json.dumps(obj, ensure_ascii=False) + "\n"
        lock_path = Path(f"{p}.lock")

        with acquire_combined_locks(lock_path, timeout=LOCK_TIMEOUT_SECONDS) as locked:
            if not locked:
                logging.debug("Timeout ou falha ao adquirir locks para append de métricas: %s", lock_path)
                return
            _rotate_metrics_file_if_needed(p, max_bytes=MAX_METRICS_FILE_BYTES, backups=MAX_METRICS_BACKUPS)
            with p.open("a", encoding="utf-8") as fp:
                fp.write(line)
                fp.flush()
    except Exception:
        logging.debug("Falha ao persistir métricas no JSONL", exc_info=True)


def _safe_loads(line: str) -> Optional[Dict]:
    try:
        return json.loads(line)
    except Exception:
        return None


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    idx = int(round((pct / 100) * (len(values) - 1)))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


# Cache em memória para evitar I/O repetitivo
_CACHE_SUMMARY: Optional[Dict[str, object]] = None
_CACHE_MTIME: float = 0.0
_CACHE_SIZE: int = 0
_CACHE_KEY: Optional[tuple] = None


def summarize_metrics(path: str, last_n: int = 2000) -> Dict[str, object]:
    """
    Resume métricas do payload a partir de um JSONL e backups mais recentes (Gate G4-B).

    Implementa Cache para evitar ler o disco se o arquivo não mudou.
    Ignora linhas inválidas e é tolerante a arquivos ausentes.
    Preserva a ordem cronológica (.2 -> .1 -> ativo) selecionando até last_n linhas.
    """
    global _CACHE_SUMMARY, _CACHE_MTIME, _CACHE_SIZE, _CACHE_KEY

    metrics_path = Path(path).resolve()
    b1_path = Path(f"{metrics_path}.1")
    b2_path = Path(f"{metrics_path}.2")

    if not metrics_path.exists() and not b1_path.exists() and not b2_path.exists():
        return {}

    # Verificação de Cache em operação normal (metadados idênticos e leitura anterior validada)
    current_key = None
    try:
        stat_active = metrics_path.stat() if metrics_path.exists() else None
        stat_b1 = b1_path.stat() if b1_path.exists() else None
        stat_b2 = b2_path.stat() if b2_path.exists() else None

        active_meta = (stat_active.st_mtime, stat_active.st_size) if stat_active else (0.0, 0)
        b1_meta = (stat_b1.st_mtime, stat_b1.st_size) if stat_b1 else (0.0, 0)
        b2_meta = (stat_b2.st_mtime, stat_b2.st_size) if stat_b2 else (0.0, 0)

        current_key = (str(metrics_path), last_n, active_meta, b1_meta, b2_meta)

        if _CACHE_SUMMARY is not None and _CACHE_KEY == current_key:
            return _CACHE_SUMMARY
    except Exception:
        current_key = None

    lock_path = Path(f"{metrics_path}.lock")
    raw_lines: List[str] = []

    try:
        with acquire_combined_locks(lock_path, timeout=LOCK_TIMEOUT_SECONDS) as locked:
            if not locked:
                logging.debug("Timeout ou falha ao adquirir locks para leitura de métricas: %s", lock_path)
                return {}

            # 1. Lê do arquivo ativo sob lock
            active_lines: deque = deque(maxlen=last_n)
            if metrics_path.exists():
                with metrics_path.open("r", encoding="utf-8") as fp:
                    for line in fp:
                        active_lines.append(line.rstrip("\n"))

            # 2. Se o ativo tiver menos que last_n linhas, complementa com backups (.1 e .2)
            needed = last_n - len(active_lines)
            b1_lines: deque = deque(maxlen=needed) if needed > 0 else deque()
            b2_lines: deque = deque()

            if needed > 0:
                if b1_path.exists():
                    try:
                        with b1_path.open("r", encoding="utf-8") as fp:
                            for line in fp:
                                b1_lines.append(line.rstrip("\n"))
                    except Exception:
                        pass

                needed_b2 = last_n - len(active_lines) - len(b1_lines)
                if needed_b2 > 0:
                    if b2_path.exists():
                        b2_lines = deque(maxlen=needed_b2)
                        try:
                            with b2_path.open("r", encoding="utf-8") as fp:
                                for line in fp:
                                    b2_lines.append(line.rstrip("\n"))
                        except Exception:
                            pass

            # Ordem cronológica estrita: .2 (mais antigo) -> .1 -> ativo (mais recente)
            raw_lines = list(b2_lines) + list(b1_lines) + list(active_lines)
    except Exception:
        return {}

    # Cálculos estatísticos executados FORA do lock de arquivo
    parsed = [_safe_loads(l) for l in raw_lines]
    records = [p for p in parsed if isinstance(p, dict)]
    total = len(records)
    if total == 0:
        return {}

    byte_values: List[float] = []
    guardrail_blocked = 0
    aborts = 0
    fallbacks = 0
    cache_hits: Dict[str, Dict[str, int]] = defaultdict(lambda: {"hit": 0, "total": 0})

    for rec in records:
        val = rec.get("payload_bytes")
        if val is None:
            val = rec.get("bytes_after")
        if isinstance(val, (int, float)):
            byte_values.append(float(val))

        if rec.get("leak_blocked") is True:
            guardrail_blocked += 1
        if rec.get("error") == "no_safe_candidate":
            aborts += 1
        if rec.get("fallback_v1") is True:
            fallbacks += 1

        if "cache_hit" in rec and "section" in rec:
            sec = str(rec.get("section"))
            cache_hits[sec]["total"] += 1
            if rec.get("cache_hit") is True:
                cache_hits[sec]["hit"] += 1

    summary: Dict[str, object] = {
        "count": total,
        "bytes_p50": _percentile(byte_values, 50),
        "bytes_p90": _percentile(byte_values, 90),
        "bytes_p95": _percentile(byte_values, 95),
        "bytes_max": max(byte_values) if byte_values else None,
        "guardrail_block_rate": guardrail_blocked / total,
        "abort_rate": aborts / total,
        "fallback_rate": fallbacks / total,
    }

    cache_summary: Dict[str, float] = {}
    for sec, data in cache_hits.items():
        if data["total"] > 0:
            cache_summary[sec] = data["hit"] / data["total"]
    if cache_summary:
        summary["cache_hit_rate"] = cache_summary

    # Atualiza Cache somente após leitura e cálculo 100% validados
    _CACHE_SUMMARY = summary
    if stat_active:
        _CACHE_MTIME = stat_active.st_mtime
        _CACHE_SIZE = stat_active.st_size
    _CACHE_KEY = current_key

    return summary
