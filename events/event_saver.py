# -*- coding: utf-8 -*-
# event_saver.py - v5.0.0 - SQLITE MIGRATION COMPLETE
"""
EventSaver v5.0.0 - Sistema de salvamento de eventos de trading

🔹 MUDANÇAS v5.0.0 (Etapa 3.2):
  ✅ Remoção de arquivos JSON/JSONL legados.
  ✅ Persistência exclusiva via SQLite (EventStore).
  ✅ Log visual mantido para debug.

🔹 HISTÓRICO:
  ✅ v4.6.0: Dual Write (SQLite + Files)
  ✅ v4.5.5: Lock em _janelas_processadas + timezone mais robusto
"""

import asyncio
import json
import os
import errno
from pathlib import Path
import platform
import logging
import queue
import threading
import time
import atexit
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Union, Tuple
import re
import copy
import math

# ===== VALIDADOR FINANCEIRO E QUARENTENA FORENSE (Gate H1-B2-Q3) =====
try:
    from common.financial_validator import (
        validate_financial_event_semantics,
        FinancialValidationResult,
    )
    HAS_FINANCIAL_VALIDATOR = True
except ImportError:
    HAS_FINANCIAL_VALIDATOR = False
    validate_financial_event_semantics = None
    FinancialValidationResult = None

try:
    from common.identity_contract import (
        canonicalize_structure,
        normalize_last_touched,
    )
    HAS_IDENTITY_CONTRACT = True
except ImportError:
    HAS_IDENTITY_CONTRACT = False
    canonicalize_structure = lambda x: x
    normalize_last_touched = lambda x: x

try:
    from common.financial_quarantine import FinancialQuarantine
    HAS_FINANCIAL_QUARANTINE = True
except ImportError:
    HAS_FINANCIAL_QUARANTINE = False
    FinancialQuarantine = None

try:
    from database.event_store import IntegrityCollisionError
except ImportError:
    class IntegrityCollisionError(Exception):
        pass

# ===== LINUX ATOMIC RENAMEAT2 SUPPORT (Gate G4-C Hardened) =====
_linux_renameat2_func = None
_linux_renameat2_checked = False

def _get_linux_renameat2():
    """Retorna função libc.renameat2 se disponível no Linux; None caso contrário."""
    global _linux_renameat2_func, _linux_renameat2_checked
    if _linux_renameat2_checked:
        return _linux_renameat2_func
    _linux_renameat2_checked = True
    if platform.system() != "Linux":
        return None
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if hasattr(libc, "renameat2"):
            func = libc.renameat2
            func.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            func.restype = ctypes.c_int
            _linux_renameat2_func = func
            return _linux_renameat2_func
    except Exception:
        pass
    return None


# ===== CONFIG (opcional) =====
try:
    import config
except Exception:
    config = None

# ===== IMPORT DO NOVO BACKEND SQLITE =====
try:
    from database.event_store import EventStore, BatchSaveResult
    HAS_SQLITE_STORE = True
except ImportError:
    EventStore = None
    BatchSaveResult = None
    HAS_SQLITE_STORE = False
    logging.warning("⚠️ Módulo database.event_store não encontrado. Escrita no SQLite desabilitada.")


def _is_valid_batch_save_result(res: Any, expected_count: int) -> bool:
    """Valida consistência interna e matemática de um BatchSaveResult (Gate G2)."""
    if BatchSaveResult is None or not isinstance(res, BatchSaveResult):
        return False
    if not (
        isinstance(res.total_received, int)
        and isinstance(res.persisted, int)
        and isinstance(res.rejected_pre_commit, int)
        and isinstance(res.commit_status, str)
    ):
        return False
    if res.total_received < 0 or res.persisted < 0 or res.rejected_pre_commit < 0:
        return False
    if res.total_received != expected_count:
        return False
    if res.persisted > res.total_received or res.rejected_pre_commit > res.total_received:
        return False
    if res.commit_status not in ("COMMITTED", "FAILED", "UNKNOWN"):
        return False
    if res.commit_status == "COMMITTED":
        if res.persisted + res.rejected_pre_commit != res.total_received:
            return False
    elif res.commit_status == "FAILED":
        if res.persisted != 0:
            return False
    elif res.commit_status == "UNKNOWN":
        if res.persisted > res.total_received:
            return False
    return True

# ETAPA 6: NaN/±Inf -> None antes de serializar (JSON RFC 8259)
try:
    from common.json_safe import sanitize_json_safe
    HAS_JSON_SAFE = True
except ImportError:
    HAS_JSON_SAFE = False
    sanitize_json_safe = None

# ===== FILE LOCKING MULTIPLATAFORMA =====
LOCK_METHOD = None
fcntl_module = None
msvcrt_module = None

try:
    import fcntl as fcntl_module
    LOCK_METHOD = 'fcntl'
except ImportError:
    if platform.system() == 'Windows':
        try:
            import msvcrt as msvcrt_module
            LOCK_METHOD = 'msvcrt'
        except ImportError:
            LOCK_METHOD = None
    else:
        LOCK_METHOD = None

# ===== CLOCK SYNC =====
try:
    from monitoring.clock_sync import get_clock_sync
    HAS_CLOCK_SYNC = True
    _clock_sync_instance = None
except ImportError:
    HAS_CLOCK_SYNC = False
    _clock_sync_instance = None

# ===== CONFIGURACOES =====
MAX_BUFFER_SIZE = 10000  # Limite maximo de eventos no buffer
BACKPRESSURE_WAKE_THRESHOLD = 5000  # Acorda worker antecipadamente se backlog crescer (Gate G1)
MAX_JSON_FILE_SIZE = 50 * 1024 * 1024  # 50MB maximo por snapshot
MAX_JSONL_BYTES = 5000  # Limite duro por linha (JSONL)
MAX_VISUAL_LOG_SIZE = 10 * 1024 * 1024  # 10MB maximo por log visual (Gate G4-A)
VISUAL_LOG_MAX_BACKUPS = 2  # Maximo de backups do log visual (Gate G4-A)
CLEANUP_INTERVAL = 300  # Limpa cache a cada 5 min
SEEN_BLOCK_TTL = 3600  # Mantem seen_in_block por 1h

# ===== REGEX PRÉ-COMPILADOS =====
NUMBER_PATTERN = re.compile(r'^[\d\.\-]+$')
PAREN_PATTERN = re.compile(r'\((.*?)\)')

# ===== TRATAMENTO ROBUSTO DE TIMEZONES =====
TIMEZONE_AVAILABLE = False
UTC_TZ = None
NY_TZ = None
SP_TZ = None

try:
    from zoneinfo import ZoneInfo
    UTC_TZ = ZoneInfo("UTC")
    NY_TZ = ZoneInfo("America/New_York")
    SP_TZ = ZoneInfo("America/Sao_Paulo")
    TIMEZONE_AVAILABLE = True
except ImportError:
    try:
        import pytz
        UTC_TZ = pytz.UTC
        NY_TZ = pytz.timezone("America/New_York")
        SP_TZ = pytz.timezone("America/Sao_Paulo")
        TIMEZONE_AVAILABLE = True
    except ImportError:
        logging.warning("Nenhuma biblioteca de timezone disponível. Usando offsets dinâmicos.")
        UTC_TZ = timezone.utc

        class DynamicNYTZ:
            """Timezone de NY com detecção de DST."""
            def __init__(self):
                self.utc = timezone.utc

            def _is_dst(self, dt: datetime) -> bool:
                """Verifica se está em horário de verão (DST)."""
                year = dt.year

                # Encontra segundo domingo de março
                march = datetime(year, 3, 1)
                days_to_sunday = (6 - march.weekday()) % 7
                first_sunday = march + timedelta(days=days_to_sunday)
                second_sunday = first_sunday + timedelta(days=7)
                dst_start = second_sunday.replace(hour=2, minute=0, second=0, microsecond=0, tzinfo=self.utc)

                # Encontra primeiro domingo de novembro
                november = datetime(year, 11, 1)
                days_to_sunday = (6 - november.weekday()) % 7
                first_sunday = november + timedelta(days=days_to_sunday)
                dst_end = first_sunday.replace(hour=2, minute=0, second=0, microsecond=0, tzinfo=self.utc)

                # Compara
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=self.utc)

                return dst_start <= dt < dst_end

            def get_offset(self, dt: datetime) -> timezone:
                """Retorna timezone com offset correto."""
                if self._is_dst(dt):
                    return timezone(timedelta(hours=-4))  # EDT
                else:
                    return timezone(timedelta(hours=-5))  # EST

        NY_TZ = DynamicNYTZ()
        SP_TZ = timezone(timedelta(hours=-3))

# ===== TRATAMENTO DE NUMPY =====
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ===== IMPORT SEGURO DO TIME_MANAGER =====
try:
    from monitoring.time_manager import TimeManager
except ImportError:
    class TimeManager:
        """
        Stub simples de TimeManager para ambientes sem time_manager.py.
        Usa sempre UTC como base.
        """
        def now_iso(self, tz=None):
            tz = tz or timezone.utc
            return datetime.now(tz).isoformat()

        def build_time_index(
            self,
            epoch_ms: int,
            include_local: bool = True,
            timespec: str = "milliseconds"
        ) -> Dict[str, Any]:
            try:
                epoch_ms_int = int(epoch_ms)
            except Exception:
                epoch_ms_int = int(time.time() * 1000)
            dt_utc = datetime.fromtimestamp(epoch_ms_int / 1000, tz=timezone.utc)
            ts_utc = dt_utc.isoformat(timespec=timespec)
            idx = {
                "epoch_ms": epoch_ms_int,
                "timestamp_utc": ts_utc,
            }
            if include_local:
                dt_local = dt_utc.astimezone()
                idx["timestamp_local"] = dt_local.isoformat(timespec=timespec)
            return idx

# ===== IMPORT SEGURO DE FORMAT_UTILS =====
try:
    from common.format_utils import (
        format_price,
        format_quantity,
        format_percent,
        format_large_number,
        format_delta,
        format_time_seconds,
        format_scientific
    )
    HAS_FORMAT_UTILS = True
except ImportError:
    HAS_FORMAT_UTILS = False

    def format_price(value):
        if value is None:
            return "null"
        try:
            if abs(value) >= 1000:
                return f"${value:,.2f}"
            else:
                return f"${value:.4f}"
        except Exception:
            return str(value)

    def format_quantity(value):
        if value is None:
            return "null"
        try:
            if abs(value) >= 1_000_000:
                return f"{value/1_000_000:.2f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.2f}K"
            else:
                return f"{value:.3f}"
        except Exception:
            return str(value)

    def format_percent(value):
        if value is None:
            return "null"
        try:
            return f"{value:.2f}%"
        except Exception:
            return str(value)

    def format_large_number(value):
        if value is None:
            return "null"
        try:
            if abs(value) >= 1_000_000_000:
                return f"{value/1_000_000_000:.2f}B"
            elif abs(value) >= 1_000_000:
                return f"{value/1_000_000:.2f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:.2f}K"
            else:
                return f"{value:,.2f}"
        except Exception:
            return str(value)

    def format_delta(value):
        if value is None:
            return "null"
        try:
            sign = "+" if value > 0 else ""
            return f"{sign}{value:.3f}"
        except Exception:
            return str(value)

    def format_time_seconds(value):
        if value is None:
            return "null"
        try:
            return f"{value:.1f}s"
        except Exception:
            return str(value)

    def format_scientific(value):
        if value is None:
            return "null"
        try:
            if abs(value) < 0.0001:
                return f"{value:.2e}"
            else:
                return f"{value:.6f}"
        except Exception:
            return str(value)

# ===== DIRETÓRIO DE DADOS =====
DATA_DIR = Path("dados")


# ===== FUNÇÕES HELPER DE FILE LOCKING (Mantidas para log visual) =====
def acquire_file_lock(file_obj, blocking=True, timeout=5.0):
    """
    Adquire lock de arquivo multiplataforma.
    """
    if LOCK_METHOD is None:
        return True  # Sem locking disponível

    try:
        if LOCK_METHOD == 'fcntl':
            if blocking:
                start_time = time.time()
                while True:
                    try:
                        fcntl_module.flock(file_obj.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
                        return True
                    except IOError:
                        if time.time() - start_time > timeout:
                            return False
                        time.sleep(0.1)
            else:
                fcntl_module.flock(file_obj.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
                return True

        elif LOCK_METHOD == 'msvcrt':
            if blocking:
                start_time = time.time()
                while True:
                    try:
                        msvcrt_module.locking(file_obj.fileno(), msvcrt_module.LK_NBLCK, 1)
                        return True
                    except IOError:
                        if time.time() - start_time > timeout:
                            return False
                        time.sleep(0.1)
            else:
                msvcrt_module.locking(file_obj.fileno(), msvcrt_module.LK_NBLCK, 1)
                return True

    except Exception as e:
        logging.debug(f"Erro ao adquirir lock: {e}")
        return False

    return True


def release_file_lock(file_obj):
    """Libera lock de arquivo multiplataforma."""
    if LOCK_METHOD is None:
        return

    try:
        if LOCK_METHOD == 'fcntl':
            fcntl_module.flock(file_obj.fileno(), fcntl_module.LOCK_UN)
        elif LOCK_METHOD == 'msvcrt':
            msvcrt_module.locking(file_obj.fileno(), msvcrt_module.LK_UNLCK, 1)
    except Exception as e:
        logging.debug(f"Erro ao liberar lock: {e}")


class EventSaver:
    """
    Classe responsável por salvar e formatar eventos de trading.

    ✅ v5.0.0: Migração completa para SQLite (JSON/JSONL removidos)
    ✅ v4.6.0: Dual Write (SQLite + Files)
    ✅ v4.5.5: Lock em _janelas_processadas + timezone mais robusto
    """

    # Fila limitada do worker de áudio (sem crescimento ilimitado).
    SOUND_QUEUE_MAXSIZE = 8

    @staticmethod
    def _resolve_sound_alert(sound_alert: Optional[bool]) -> bool:
        """Default OFF (server/headless/produção); opt-in via SOUND_ALERT=1."""
        if sound_alert is not None:
            return bool(sound_alert)
        return os.getenv("SOUND_ALERT", "0").strip().lower() in (
            "1", "true", "yes", "on",
        )

    def __init__(
        self,
        sound_alert: Optional[bool] = None,
        health_monitor=None,
        quarantine: Optional[Any] = None,
        quarantine_dir: Optional[Union[str, Path]] = None,
        enforce_financial_validation: bool = True,
    ):
        self._quarantine = quarantine
        self._quarantine_dir = Path(quarantine_dir) if quarantine_dir else None
        self.enforce_financial_validation = enforce_financial_validation
        DATA_DIR.mkdir(exist_ok=True)
        self.sound_alert = self._resolve_sound_alert(sound_alert)
        # Worker único de áudio (lazy): nunca bloqueia o hot path.
        self._sound_queue: "queue.Queue[bool]" = queue.Queue(
            maxsize=self.SOUND_QUEUE_MAXSIZE
        )
        self._sound_thread: Optional[threading.Thread] = None
        self._sound_stop = threading.Event()
        self._sound_lock = threading.Lock()
        self._sound_dropped = 0
        # HealthMonitor opcional: heartbeat de progresso "event_saver"
        # (apenas se injetado; None em testes/uso isolado)
        self.health_monitor = health_monitor
        self.visual_log_file = DATA_DIR / "eventos_visuais.log"
        json_file_name = "eventos-fluxo.json"
        jsonl_file_name = "eventos_fluxo.jsonl"
        if config is not None:
            json_file_name = getattr(
                config, "EVENT_SAVER_JSON_FILE", json_file_name
            )
            jsonl_file_name = getattr(
                config, "EVENT_SAVER_JSONL_FILE", jsonl_file_name
            )
        self.snapshot_file = DATA_DIR / json_file_name
        self.history_file = DATA_DIR / jsonl_file_name
        self.write_json = False  # default: JSON snapshot desativado (SQLite é a fonte de verdade)
        self.write_jsonl = True
        self.max_json_events = 1000
        self.max_json_file_size = MAX_JSON_FILE_SIZE
        self.max_jsonl_bytes = MAX_JSONL_BYTES
        if config is not None:
            self.write_json = bool(
                getattr(config, "EVENT_SAVER_WRITE_JSON", False)
            )
            self.write_jsonl = bool(
                getattr(config, "EVENT_SAVER_WRITE_JSONL", True)
            )
            try:
                self.max_jsonl_bytes = int(
                    getattr(config, "EVENT_SAVER_MAX_JSONL_BYTES", MAX_JSONL_BYTES)
                )
            except Exception:
                self.max_jsonl_bytes = MAX_JSONL_BYTES
            try:
                self.max_json_events = int(
                    getattr(config, "EVENT_SAVER_MAX_JSON_EVENTS", 1000)
                )
            except Exception:
                self.max_json_events = 1000
            try:
                max_mb = float(
                    getattr(config, "EVENT_SAVER_MAX_JSON_MB", 50)
                )
                self.max_json_file_size = int(max_mb * 1024 * 1024)
            except Exception:
                self.max_json_file_size = MAX_JSON_FILE_SIZE

        # Configurações do Log Visual (Gate G4-A)
        self.write_visual_log = True
        self.max_visual_log_bytes = MAX_VISUAL_LOG_SIZE
        self.visual_log_backups = VISUAL_LOG_MAX_BACKUPS

        if config is not None:
            # 1. WRITE_VISUAL_LOG
            raw_write_visual = getattr(config, "EVENT_SAVER_WRITE_VISUAL_LOG", True)
            if isinstance(raw_write_visual, str):
                self.write_visual_log = raw_write_visual.strip().lower() in ("true", "1", "yes", "on")
            else:
                self.write_visual_log = bool(raw_write_visual)

            # 2. MAX_VISUAL_LOG_MB
            default_visual_mb = 10.0
            max_allowed_visual_mb = 1024.0  # Limite máximo seguro (1 GB)
            try:
                raw_max_mb = getattr(config, "EVENT_SAVER_MAX_VISUAL_LOG_MB", default_visual_mb)
                val_mb = float(raw_max_mb)
                import math
                if math.isnan(val_mb) or math.isinf(val_mb) or val_mb <= 0:
                    self.max_visual_log_bytes = int(default_visual_mb * 1024 * 1024)
                elif val_mb > max_allowed_visual_mb:
                    self.max_visual_log_bytes = int(max_allowed_visual_mb * 1024 * 1024)
                else:
                    self.max_visual_log_bytes = int(val_mb * 1024 * 1024)
            except Exception:
                self.max_visual_log_bytes = int(default_visual_mb * 1024 * 1024)

            # 3. VISUAL_LOG_BACKUPS
            default_backups = 2
            max_allowed_backups = 10
            try:
                raw_backups = getattr(config, "EVENT_SAVER_VISUAL_LOG_BACKUPS", default_backups)
                val_backups = int(raw_backups)
                if val_backups <= 0:
                    self.visual_log_backups = default_backups
                elif val_backups > max_allowed_backups:
                    self.visual_log_backups = max_allowed_backups
                else:
                    self.visual_log_backups = val_backups
            except Exception:
                self.visual_log_backups = default_backups

        # Configurações de Rotação e Arquivamento Não Destrutivo do JSONL (Gate G4-C)
        self.auto_archive_jsonl = True
        default_jsonl_mb = 20.0
        max_allowed_jsonl_mb = 1024.0  # Limite máximo seguro (1 GB)
        min_allowed_jsonl_mb = 0.001   # Permite testes com limites pequenos (~1 KB)
        self.max_jsonl_file_bytes = int(default_jsonl_mb * 1024 * 1024)
        self.archive_dir = DATA_DIR / "archive"

        if config is not None:
            # 1. EVENT_SAVER_JSONL_AUTO_ARCHIVE
            raw_auto_archive = getattr(config, "EVENT_SAVER_JSONL_AUTO_ARCHIVE", True)
            if isinstance(raw_auto_archive, str):
                self.auto_archive_jsonl = raw_auto_archive.strip().lower() in ("true", "1", "yes", "on")
            else:
                self.auto_archive_jsonl = bool(raw_auto_archive)

            # 2. EVENT_SAVER_JSONL_MAX_MB
            try:
                raw_jsonl_mb = getattr(config, "EVENT_SAVER_JSONL_MAX_MB", default_jsonl_mb)
                val_jsonl_mb = float(raw_jsonl_mb)
                import math
                if math.isnan(val_jsonl_mb) or math.isinf(val_jsonl_mb) or val_jsonl_mb < min_allowed_jsonl_mb:
                    self.max_jsonl_file_bytes = int(default_jsonl_mb * 1024 * 1024)
                elif val_jsonl_mb > max_allowed_jsonl_mb:
                    self.max_jsonl_file_bytes = int(max_allowed_jsonl_mb * 1024 * 1024)
                else:
                    self.max_jsonl_file_bytes = int(val_jsonl_mb * 1024 * 1024)
            except Exception:
                self.max_jsonl_file_bytes = int(default_jsonl_mb * 1024 * 1024)

            # 3. EVENT_SAVER_JSONL_ARCHIVE_DIR
            try:
                raw_archive_dir = getattr(config, "EVENT_SAVER_JSONL_ARCHIVE_DIR", "dados/archive")
                self.archive_dir = Path(raw_archive_dir)
            except Exception:
                self.archive_dir = DATA_DIR / "archive"

        self.last_window_id = None  # compat legada
        self.time_manager = TimeManager()
        self._window_counter = 0   # pode ser usado por código legado sem janela_numero

        # Logger
        self.logger = logging.getLogger(__name__)

        # ✅ Inicializa Banco de Dados (SQLite)
        self.db = None
        if HAS_SQLITE_STORE:
            try:
                self.db = EventStore()
                self.logger.info("✅ SQLite EventStore inicializado com sucesso")
            except Exception as e:
                self.logger.error(f"❌ Falha ao inicializar SQLite EventStore: {e}")
                self.db = None

        # Controle de janelas processadas
        self._janelas_processadas = set()      # Rastreia janelas que já tiveram separador
        self._last_janela_numero = None        # Última janela com separador escrito
        self._janelas_lock = threading.Lock()  # lock dedicado para _janelas_processadas

        # _seen_in_block com timestamp para limpeza
        self._seen_in_block: Dict[Tuple, float] = {}
        self._last_cleanup_time = time.time()

        # 🆕 Rastreador de timestamps para deduplicação
        self._last_event_timestamp_ms: Optional[int] = None
        self._duplicate_event_count = 0
        self._max_timestamp_gap_ms = 100  # Detecta saltos de tempo fora do esperado

        # Buffer de escrita + thread de flush (Gate G1)
        self._write_buffer: List[Dict] = []
        self._buffer_lock = threading.Lock()
        self._flush_interval = 5
        self._stop_event = threading.Event()
        self._wake_flush_event = threading.Event()
        self._stop_lock = threading.Lock()
        self._shutdown_incomplete = False
        # Gate G3-A: Rastreamento estrito de lifecycle do worker
        self._worker_exited_cleanly = False
        self._worker_exception: Optional[str] = None

        # Contabilidade estrita de eventos (Gate G1)
        self._events_received = 0
        self._events_accepted = 0
        self._events_rejected_overflow = 0
        self._events_rejected_after_stop = 0
        self._events_in_flight = 0
        self._events_finalized = 0
        self._events_failed_terminal = 0

        # Gate G2: Contabilidade por destino independente
        self._sqlite_events_attempted = 0
        self._sqlite_events_persisted = 0
        self._sqlite_events_failed = 0
        self._sqlite_events_unknown = 0
        self._sqlite_events_rejected_pre_commit = 0

        self._jsonl_events_attempted = 0
        self._jsonl_events_written = 0
        self._jsonl_events_failed = 0
        self._jsonl_events_unknown = 0

        self._fallback_events_attempted = 0
        self._fallback_events_written = 0
        self._fallback_events_failed = 0
        self._fallback_events_unknown = 0
        self._enospc_errors = 0

        # Gate G2: Rate-limited warning do worker
        self._rejection_warn_interval = 60.0
        self._last_rejection_warn_time = 0.0
        self._last_warn_overflow = 0
        self._last_warn_after_stop = 0

        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="EventSaverFlushWorker"
        )
        self._flush_thread.start()

        # Thread de cleanup
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        # Stats
        self._buffer_overflow_count = 0
        self._lock_timeout_count = 0

        # ✅ Inicializa Clock Sync
        if HAS_CLOCK_SYNC:
            try:
                global _clock_sync_instance
                if _clock_sync_instance is None:
                    self.logger.info("🕐 Inicializando Clock Sync...")
                    _clock_sync_instance = get_clock_sync()

                    # Aguarda primeira sincronização (até 5s para HTTP completar)
                    synced = _clock_sync_instance.wait_for_sync(timeout=5.0)

                    if synced:
                        offset = _clock_sync_instance.get_offset_seconds()
                        if abs(offset) > 0.5:
                            self.logger.info(
                                f"✅ Clock Sync ativo | "
                                f"Offset: {offset:+.3f}s | "
                                f"Compensação automática habilitada"
                            )
                        else:
                            self.logger.info(
                                f"✅ Clock Sync ativo | "
                                f"Relógio bem sincronizado (offset: {offset:+.3f}s)"
                            )
                    else:
                        self.logger.warning("⚠️ Clock Sync: sincronização inicial pendente (timeout 5s)")

            except Exception as e:
                self.logger.warning(f"⚠️ Clock Sync não disponível: {e}")
                _clock_sync_instance = None
        else:
            self.logger.info("ℹ️ Clock Sync não disponível (usando timestamps locais)")

        # Encerra limpo no exit
        atexit.register(self.stop)

        # Verificação inicial
        self._debug_timezone_check()

        lock_status = "HABILITADO" if LOCK_METHOD else "DESABILITADO"
        lock_type = f"({LOCK_METHOD})" if LOCK_METHOD else ""

        self.logger.info(
            "✅ EventSaver v5.0.0 inicializado | "
            "Buffer max: %d | Cleanup: %ds | File locking: %s %s",
            MAX_BUFFER_SIZE,
            CLEANUP_INTERVAL,
            lock_status,
            lock_type
        )

    def _debug_timezone_check(self):
        """Verifica se os timezones estão funcionando corretamente."""
        try:
            now_utc = datetime.now(UTC_TZ)
            if TIMEZONE_AVAILABLE:
                now_ny = self._convert_timezone(now_utc, NY_TZ)
                now_sp = self._convert_timezone(now_utc, SP_TZ)
                self.logger.info("🕐 Verificação de Timezone:")
                self.logger.info(f"   UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                self.logger.info(f"   NY:  {now_ny.strftime('%Y-%m-%d %H:%M:%S')} (EST/EDT)")
                self.logger.info(f"   SP:  {now_sp.strftime('%Y-%m-%d %H:%M:%S')} (BRT)")

                if hasattr(now_ny, 'utcoffset') and now_ny.utcoffset():
                    utc_offset_ny = now_ny.utcoffset().total_seconds() / 3600
                    self.logger.info(f"   Offset NY vs UTC: {utc_offset_ny:.1f} horas")
                if hasattr(now_sp, 'utcoffset') and now_sp.utcoffset():
                    utc_offset_sp = now_sp.utcoffset().total_seconds() / 3600
                    self.logger.info(f"   Offset SP vs UTC: {utc_offset_sp:.1f} horas")
            else:
                if hasattr(NY_TZ, 'get_offset'):
                    offset_tz = NY_TZ.get_offset(now_utc)
                    dst_status = "EDT (-4h)" if offset_tz.utcoffset(None).total_seconds() == -14400 else "EST (-5h)"
                    self.logger.info(f"   NY usando offset dinâmico: {dst_status}")
                else:
                    self.logger.warning("⚠️ Usando offsets fixos (NY: -5h, SP: -3h)")
        except Exception as e:
            self.logger.error(f"Erro ao verificar timezones: {e}")

    def _convert_timezone(self, dt: datetime, target_tz) -> datetime:
        """
        Converte datetime para timezone alvo de forma segura.
        """
        try:
            # Suporte para DynamicNYTZ
            if hasattr(target_tz, 'get_offset'):
                offset_tz = target_tz.get_offset(dt)
                if dt.tzinfo is not None:
                    return dt.astimezone(offset_tz)
                else:
                    dt_utc = dt.replace(tzinfo=UTC_TZ)
                    return dt_utc.astimezone(offset_tz)

            # Se já tem timezone
            if dt.tzinfo is not None:
                if hasattr(target_tz, 'normalize'):
                    return target_tz.normalize(dt.astimezone(target_tz))
                else:
                    return dt.astimezone(target_tz)
            else:
                if UTC_TZ == timezone.utc:
                    dt_utc = dt.replace(tzinfo=UTC_TZ)
                    return dt_utc.astimezone(target_tz)
                else:
                    if hasattr(UTC_TZ, 'localize'):
                        dt_utc = UTC_TZ.localize(dt)
                    else:
                        dt_utc = dt.replace(tzinfo=UTC_TZ)
                    if hasattr(target_tz, 'normalize'):
                        return target_tz.normalize(dt_utc.astimezone(target_tz))
                    else:
                        return dt_utc.astimezone(target_tz)
        except Exception as e:
            self.logger.error(
                f"❌ FALHA na conversão de timezone: {e}",
                exc_info=True
            )
            # Fallback: garante pelo menos tzinfo=UTC para evitar datetime "naive"
            if isinstance(dt, datetime) and dt.tzinfo is None:
                return dt.replace(tzinfo=UTC_TZ)
            return dt

    @staticmethod
    def _parse_iso8601(ts: str) -> datetime:
        """Parseia timestamp ISO-8601 com suporte a 'Z' (UTC) e offset."""
        if not ts:
            raise ValueError("Timestamp vazio")
        try:
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC_TZ)
            return dt
        except Exception as e:
            try:
                base = ts.split(".")[0]
                if "+" in base:
                    base, offset = base.split("+")
                    ts_clean = f"{base}+{offset}"
                elif "-" in base and base.count("-") > 2:
                    parts = base.split("-")
                    base = "-".join(parts[:3])
                    offset = "-".join(parts[3:])
                    ts_clean = f"{base}-{offset}"
                else:
                    ts_clean = base + "+00:00"
                return datetime.fromisoformat(ts_clean)
            except Exception:
                try:
                    dt_str = ts.replace("Z", "").replace("T", " ").split(".")[0]
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                    return dt.replace(tzinfo=UTC_TZ)
                except Exception:
                    raise ValueError(f"Não foi possível parsear timestamp: {ts}") from e

    def _cleanup_loop(self):
        """Thread que limpa recursos periodicamente."""
        while not self._stop_event.wait(CLEANUP_INTERVAL):
            try:
                self._cleanup_seen_in_block()
            except Exception as e:
                self.logger.error(f"Erro no cleanup: {e}")

    def _cleanup_seen_in_block(self):
        """Limpa entradas antigas de _seen_in_block."""
        try:
            now = time.time()
            cutoff = now - SEEN_BLOCK_TTL

            cleaned = {
                k: v for k, v in self._seen_in_block.items()
                if v > cutoff
            }

            removed = len(self._seen_in_block) - len(cleaned)

            if removed > 0:
                self._seen_in_block = cleaned
                self.logger.debug(
                    f"🧹 Cleanup: {removed} entradas antigas removidas de _seen_in_block "
                    f"({len(cleaned)} restantes)"
                )

            self._last_cleanup_time = now

        except Exception as e:
            self.logger.error(f"Erro ao limpar _seen_in_block: {e}")

        # Limpa set de janelas processadas se ficar muito grande (thread-safe)
        with self._janelas_lock:
            if len(self._janelas_processadas) > 1000:
                self._janelas_processadas.clear()
                self.logger.info("🧹 Set de janelas processadas limpo")

    def stop(self, timeout: float = 5.0):
        """
        Para threads e coordena o encerramento seguro (Gate G1).
        1. Rejeita novas admissões;
        2. Solicita encerramento;
        3. Acorda o worker para permitir drain;
        4. Aguarda término bounded (timeout);
        5. Informa shutdown incompleto se worker permanecer ativo;
        6. NUNCA executa flush concorrente;
        7. NUNCA limpa buffer de worker vivo;
        8. Idempotente: chamadas repetidas não causam deadlock.
        """
        with self._stop_lock:
            if not self._stop_event.is_set():
                self._stop_event.set()
                self._wake_flush_event.set()

            warn_msg = None
            error_msg = None

            # Aguarda o worker de persistência terminar o drain até o timeout bounded
            try:
                if self._flush_thread is not None and self._flush_thread.is_alive():
                    self._flush_thread.join(timeout=timeout)
                    if self._flush_thread.is_alive():
                        with self._buffer_lock:
                            self._shutdown_incomplete = True
                        warn_msg = (
                            f"⚠️ Worker de persistência ainda ativo após timeout ({timeout}s). Shutdown incompleto."
                        )

                # Gate G3-A: Verificação do estado do worker pós-join (ou se já havia encerrado)
                with self._buffer_lock:
                    worker_alive = self._flush_thread.is_alive() if self._flush_thread else False
                    if not worker_alive and not self._worker_exited_cleanly:
                        self._shutdown_incomplete = True
                        error_msg = "❌ Worker de persistência encerrou de forma anômala. Shutdown incompleto."
                    elif not worker_alive and (len(self._write_buffer) > 0 or self._events_in_flight > 0):
                        self._shutdown_incomplete = True
                        error_msg = "⚠️ Eventos pendentes no buffer após término do worker. Shutdown incompleto."
            except Exception as e:
                self.logger.error(f"Erro ao aguardar flush worker no stop: {e}")

            if warn_msg:
                self.logger.warning(warn_msg)
            if error_msg:
                self.logger.error(error_msg)

            # Para thread de cleanup
            try:
                if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
                    self._cleanup_thread.join(timeout=1.0)
            except Exception as e:
                self.logger.error(f"Erro ao parar cleanup thread: {e}")

            # Para worker de áudio (não espera beep em curso além do limite)
            try:
                self._sound_stop.set()
                thread, self._sound_thread = self._sound_thread, None
                if (thread is not None and thread.is_alive()
                        and thread is not threading.current_thread()):
                    thread.join(timeout=1.0)
            except Exception as e:
                self.logger.error(f"Erro ao parar thread de áudio: {e}")

    def _is_nan_or_inf(self, value) -> bool:
        """Verifica se um valor é NaN ou Inf."""
        if value is None:
            return False
        if HAS_NUMPY:
            try:
                return np.isnan(value) or np.isinf(value)
            except (TypeError, ValueError):
                return False
        else:
            try:
                if isinstance(value, float):
                    return value != value or value in (float('inf'), float('-inf'))
                return False
            except Exception:
                return False

    def _clean_numeric_value(
        self,
        value: Any,
        field_type: str = "generic"
    ) -> Optional[Union[int, float]]:
        """Limpa e arredonda um valor numérico."""
        if value is None or value == '':
            return None
        if isinstance(value, str):
            if value.lower() in ['n/a', 'none', 'null', 'nan', 'inf', '-inf']:
                return None
            value = value.replace(',', '')
        try:
            num_value = float(value)
            if self._is_nan_or_inf(num_value):
                return None

            precision_map = {
                "price": 4,
                "quantity": 3,
                "percent": 2,
                "delta": 3,
                "scientific": 8,
                "generic": 4
            }
            precision = precision_map.get(field_type, 4)
            rounded_value = round(num_value, precision)

            if rounded_value == int(rounded_value) and field_type != "scientific":
                return int(rounded_value)
            return rounded_value
        except (ValueError, TypeError, OverflowError):
            return None

    def _clean_event_data(self, event: Dict) -> Dict:
        """Limpa recursivamente todos os campos numéricos do evento."""
        if not isinstance(event, dict):
            return event

        try:
            cleaned = {}

            field_types = {
                'price': {'preco', 'price', 'close', 'open', 'high', 'low', 'poc', 'val', 'vah',
                         'level', 'bid_price', 'ask_price', 'wall_price', 'anchor'},
                'quantity': {'volume', 'quantity', 'size', 'count', 'trades', 'interest'},
                'delta': {'delta', 'flow', 'imbalance', 'net'},
                'percent': {'pct', 'percent', 'ratio', 'prob', 'rate'},
                'scientific': {'volatility', 'returns', 'slope', 'momentum', 'correlation', 'gradient'},
                'timestamp': {'epoch_ms', 'timestamp_ms', 'window_ms', 'time_ms', '_ms', '_ts'}
            }

            def get_field_type(key: str) -> str:
                key_lower = key.lower()
                if any(ts in key_lower for ts in field_types['timestamp']):
                    return 'timestamp'
                for field_type, keywords in field_types.items():
                    if field_type != 'timestamp':
                        if any(kw in key_lower for kw in keywords):
                            return field_type
                return 'generic'

            for key, value in event.items():
                field_type = get_field_type(key)

                if field_type == 'timestamp':
                    cleaned[key] = value
                elif isinstance(value, (int, float)):
                    clean_value = self._clean_numeric_value(value, field_type)
                    if clean_value is not None:
                        cleaned[key] = clean_value
                elif isinstance(value, str) and value.replace('.', '').replace('-', '').replace('+', '').replace(',', '').isdigit():
                    clean_value = self._clean_numeric_value(value, field_type)
                    if clean_value is not None:
                        cleaned[key] = clean_value
                elif isinstance(value, dict):
                    cleaned_dict = self._clean_event_data(value)
                    if cleaned_dict:
                        cleaned[key] = cleaned_dict
                elif isinstance(value, list):
                    cleaned_list = []
                    for item in value:
                        if isinstance(item, dict):
                            clean_item = self._clean_event_data(item)
                            if clean_item:
                                cleaned_list.append(clean_item)
                        elif isinstance(item, (int, float)):
                            clean_value = self._clean_numeric_value(item, field_type)
                            if clean_value is not None:
                                cleaned_list.append(clean_value)
                        elif item is not None:
                            cleaned_list.append(item)

                    if key.lower() in ['hvns', 'lvns', 'historical_volumes'] and len(cleaned_list) > 10:
                        cleaned[key] = cleaned_list
                    elif cleaned_list:
                        cleaned[key] = cleaned_list
                elif value is not None:
                    cleaned[key] = value

            return cleaned

        except Exception as e:
            self.logger.error(f"Erro ao limpar dados: {e}")
            return event

    def _save_to_json(self, event: Dict) -> None:
        """Salva evento em arquivo JSON (snapshot) com lock e retry."""
        if not self.write_json:
            return

        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            lock_file = None
            lock_acquired = False
            lock_file_path = None

            try:
                if self.snapshot_file.exists():
                    file_size = self.snapshot_file.stat().st_size
                    if file_size > self.max_json_file_size:
                        self.logger.warning(
                            "Snapshot JSON grande (%.1fMB), rotacionando.",
                            file_size / 1024 / 1024,
                        )
                        backup = self.snapshot_file.with_suffix(".backup")
                        try:
                            self.snapshot_file.replace(backup)
                        except Exception:
                            self.snapshot_file.rename(backup)
                        events = []
                    else:
                        lock_file_path = self.snapshot_file.with_suffix(".lock")
                        lock_file = open(lock_file_path, "w")
                        lock_acquired = acquire_file_lock(
                            lock_file, blocking=True, timeout=8.0
                        )
                        if not lock_acquired:
                            self._lock_timeout_count += 1
                            raise IOError("Lock timeout")

                        try:
                            with open(
                                self.snapshot_file, "r", encoding="utf-8"
                            ) as f:
                                content = f.read().strip()
                                events = json.loads(content) if content else []
                        except (json.JSONDecodeError, OSError):
                            events = []
                else:
                    events = []

                events.append(event)
                if len(events) > self.max_json_events:
                    events = events[-self.max_json_events :]

                temp_file = self.snapshot_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(
                        events, f, indent=2, ensure_ascii=False, default=str
                    )

                temp_file.replace(self.snapshot_file)

                if lock_file and lock_acquired:
                    release_file_lock(lock_file)
                    lock_file.close()
                    if lock_file_path:
                        try:
                            lock_file_path.unlink(missing_ok=True)
                        except Exception:
                            pass

                return

            except Exception as e:
                self.logger.error(
                    "Erro ao salvar JSON (tentativa %d): %s",
                    attempt + 1,
                    e,
                )

                if lock_file:
                    try:
                        if lock_acquired:
                            release_file_lock(lock_file)
                        lock_file.close()
                        if lock_file_path:
                            lock_file_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    self._save_fallback(event, "json")

    def _optimize_event_for_storage(self, event: Dict) -> Dict:
        """
        Otimiza apenas para armazenamento (JSONL).
        Não muda a estrutura essencial do evento.
        """
        if not isinstance(event, dict):
            return event

        if event.get("tipo_evento") != "ANALYSIS_TRIGGER":
            return event

        event_original = event

        try:
            import copy
            event_copy = copy.deepcopy(event)
        except Exception:
            event_copy = dict(event)

        try:
            from data_processing.fix_optimization import clean_event, simplify_historical_vp, remove_enriched_snapshot
            event_copy = clean_event(event_copy)
            event_copy = simplify_historical_vp(event_copy)
            event_copy = remove_enriched_snapshot(event_copy)
        except Exception:
            self.logger.exception("Falha ao otimizar evento para storage")
            return event_original

        return event_copy

    def _inc_stat(self, field: str, amount: int = 1) -> None:
        """Incrementa contador de forma thread-safe e defensiva a instâncias incompletas (Gate G2)."""
        lock = getattr(self, "_buffer_lock", None)
        if lock is not None:
            with lock:
                setattr(self, field, getattr(self, field, 0) + amount)
        elif hasattr(self, field):
            setattr(self, field, getattr(self, field, 0) + amount)

    def _save_to_jsonl(self, event: Dict) -> str:
        """Salva evento em arquivo JSONL (historico) com retry (Gate G2)."""
        if not self.write_jsonl:
            return "DISABLED"

        self._inc_stat("_jsonl_events_attempted", 1)

        # Rotação não destrutiva por tamanho antes de abrir para escrita (Gate G4-C)
        try:
            self._rotate_jsonl_if_needed()
        except Exception as rot_err:
            self.logger.warning("Erro inesperado em _rotate_jsonl_if_needed: %s", rot_err)

        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                with open(self.history_file, "a", encoding="utf-8") as f:
                    event_to_save = self._optimize_event_for_storage(event)
                    if sanitize_json_safe is not None:
                        event_to_save = sanitize_json_safe(event_to_save)
                    json_line = json.dumps(
                        event_to_save, ensure_ascii=False, default=str, separators=(",", ":")
                    )
                    max_bytes = getattr(self, "max_jsonl_bytes", MAX_JSONL_BYTES)
                    tipo_evento = event.get("tipo_evento") if isinstance(event, dict) else None
                    if (
                        tipo_evento == "ANALYSIS_TRIGGER"
                        and len(json_line.encode("utf-8", errors="replace")) > max_bytes
                    ):
                        event_slim = {
                            "tipo_evento": event.get("tipo_evento") if isinstance(event, dict) else None,
                            "symbol": event.get("symbol") if isinstance(event, dict) else None,
                            "epoch_ms": event.get("epoch_ms") if isinstance(event, dict) else None,
                            "janela_numero": event.get("janela_numero") if isinstance(event, dict) else None,
                            "event_id": event.get("event_id") if isinstance(event, dict) else None,
                            "timestamp_utc": event.get("timestamp_utc") if isinstance(event, dict) else None,
                            "note": "trimmed_by_guardian",
                        }

                        json_line = json.dumps(
                            event_slim, ensure_ascii=False, default=str, separators=(",", ":")
                        )

                        if len(json_line.encode("utf-8", errors="replace")) > max_bytes:
                            json_line = json.dumps(
                                {
                                    "tipo_evento": event_slim.get("tipo_evento"),
                                    "event_id": event_slim.get("event_id"),
                                    "note": "trimmed_by_guardian_hard",
                                },
                                ensure_ascii=False,
                                default=str,
                                separators=(",", ":"),
                            )
                            if len(json_line.encode("utf-8", errors="replace")) > max_bytes:
                                json_line = json.dumps(
                                    {"note": "trimmed_by_guardian_hard"},
                                    ensure_ascii=False,
                                    default=str,
                                    separators=(",", ":"),
                                )
                    f.write(json_line + "\n")
                    f.flush()

                self._inc_stat("_jsonl_events_written", 1)
                return "WRITTEN"
            except Exception as e:
                if isinstance(e, OSError) and (getattr(e, "errno", None) == errno.ENOSPC or "no space left on device" in str(e).lower()):
                    self._enospc_errors += 1
                    self.logger.critical("Falha crítica de armazenamento em disco cheio (ENOSPC) no JSONL: %s", e)
                self.logger.error(
                    "Erro ao salvar JSONL (tentativa %d): %s",
                    attempt + 1,
                    e,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    self._inc_stat("_jsonl_events_failed", 1)
                    self._save_fallback(event, "jsonl")
                    return "FAILED"

        return "UNKNOWN"

    def _save_fallback(self, event: Dict, format_type: str) -> str:
        """Salva evento em fallback_events quando JSON falha (Gate G2 / H1-B2-Q3)."""
        # Validação defensiva: se for financeiramente inválido, quarentena e NUNCA salva em fallback
        if getattr(self, "enforce_financial_validation", True):
            val_res = self._validate_financial_payload(event)
            if val_res is not None and not val_res.is_valid:
                self.logger.warning(
                    f"Evento rejeitado no fallback pelo validador financeiro: {val_res.rejection_reason}"
                )
                if self.quarantine is not None:
                    try:
                        self.quarantine.quarantine_event(
                            copy.deepcopy(event),
                            val_res,
                            source_caller="EventSaver._save_fallback",
                        )
                    except Exception as q_err:
                        self.logger.error(f"Falha ao registrar quarentena no fallback: {q_err}")
                return "FAILED"

        try:
            event = canonicalize_structure(event)
        except Exception:
            pass

        self._inc_stat("_fallback_events_attempted", 1)

        try:
            fallback_dir = (self.history_file.parent / "fallback_events") if hasattr(self, "history_file") and self.history_file else (DATA_DIR / "fallback_events")
            fallback_dir.mkdir(exist_ok=True)

            stamp = datetime.now().strftime("%Y%m%d")
            suffix = "jsonl" if format_type == "jsonl" else "json"
            fallback_file = fallback_dir / f"eventos_{stamp}.{suffix}"

            with open(fallback_file, "a", encoding="utf-8") as f:
                if format_type == "jsonl":
                    event_to_save = self._optimize_event_for_storage(event)
                    if sanitize_json_safe is not None:
                        event_to_save = sanitize_json_safe(event_to_save)
                    json_line = json.dumps(
                        event_to_save, ensure_ascii=False, default=str, separators=(",", ":")
                    )
                    max_bytes = getattr(self, "max_jsonl_bytes", MAX_JSONL_BYTES)
                    tipo_evento = event.get("tipo_evento") if isinstance(event, dict) else None
                    if (
                        tipo_evento == "ANALYSIS_TRIGGER"
                        and len(json_line.encode("utf-8", errors="replace")) > max_bytes
                    ):
                        event_slim = {
                            "tipo_evento": event.get("tipo_evento") if isinstance(event, dict) else None,
                            "symbol": event.get("symbol") if isinstance(event, dict) else None,
                            "epoch_ms": event.get("epoch_ms") if isinstance(event, dict) else None,
                            "janela_numero": event.get("janela_numero") if isinstance(event, dict) else None,
                            "event_id": event.get("event_id") if isinstance(event, dict) else None,
                            "timestamp_utc": event.get("timestamp_utc") if isinstance(event, dict) else None,
                            "note": "trimmed_by_guardian",
                        }

                        json_line = json.dumps(
                            event_slim, ensure_ascii=False, default=str, separators=(",", ":")
                        )

                        if len(json_line.encode("utf-8", errors="replace")) > max_bytes:
                            json_line = json.dumps(
                                {
                                    "tipo_evento": event_slim.get("tipo_evento"),
                                    "event_id": event_slim.get("event_id"),
                                    "note": "trimmed_by_guardian_hard",
                                },
                                ensure_ascii=False,
                                default=str,
                                separators=(",", ":"),
                            )
                            if len(json_line.encode("utf-8", errors="replace")) > max_bytes:
                                json_line = json.dumps(
                                    {"note": "trimmed_by_guardian_hard"},
                                    ensure_ascii=False,
                                    default=str,
                                    separators=(",", ":"),
                                )

                    f.write(json_line + "\n")
                else:
                    event_json = event
                    if sanitize_json_safe is not None:
                        event_json = sanitize_json_safe(event_json)
                    f.write(
                        json.dumps(
                            event_json, ensure_ascii=False, default=str
                        ) + "\n"
                    )
                f.flush()

            self._inc_stat("_fallback_events_written", 1)
            return "WRITTEN"
        except Exception as e:
            if isinstance(e, OSError) and (getattr(e, "errno", None) == errno.ENOSPC or "no space left on device" in str(e).lower()):
                self._enospc_errors += 1
                self.logger.critical(f"Falha crítica de armazenamento em disco cheio (ENOSPC) ao salvar fallback: {e}")
            self.logger.error(f"Falha ao salvar fallback: {e}")
            self._inc_stat("_fallback_events_failed", 1)
            return "FAILED"

    def _flush_loop(self):
        """
        Worker thread única responsável pela persistência periódica e sob demanda (Gate G1).
        Garante que apenas esta thread execute operações de flush/persistência.
        """
        try:
            while not self._stop_event.is_set():
                # Aguarda até o flush interval periódico OU sinal de wake-up antecipado
                self._wake_flush_event.wait(timeout=self._flush_interval)
                self._wake_flush_event.clear()

                self._process_single_flush_batch()
                self._check_and_log_rejection_warnings()

            # Drenagem segura pós-stop: processa o que restou no buffer antes de encerrar
            while self._process_single_flush_batch():
                pass

            # Verificação final pós-drenagem (Gate G2)
            self._check_and_log_rejection_warnings(force=True)

            with self._buffer_lock:
                if (
                    self._stop_event.is_set()
                    and len(self._write_buffer) == 0
                    and self._events_in_flight == 0
                ):
                    self._worker_exited_cleanly = True
        except BaseException as e:
            err_diag = f"{type(e).__name__}: {str(e)[:200]}"
            with self._buffer_lock:
                self._worker_exception = err_diag
            try:
                self.logger.critical(
                    f"Worker de persistência encerrou por exceção fatal: {err_diag}",
                    exc_info=True,
                )
            except Exception:
                pass
            raise

    def _check_and_log_rejection_warnings(self, force: bool = False) -> None:
        """
        Emite aviso rate-limited sobre eventos descartados por backpressure (Gate G2).
        Executa EXCLUSIVAMENTE na thread do worker, fora de _buffer_lock e fora de save_event().
        """
        now = time.monotonic()
        if not force and (now - self._last_rejection_warn_time < self._rejection_warn_interval):
            return

        with self._buffer_lock:
            cur_overflow = self._events_rejected_overflow
            cur_after_stop = self._events_rejected_after_stop

        delta_overflow = cur_overflow - self._last_warn_overflow
        delta_after_stop = cur_after_stop - self._last_warn_after_stop

        if delta_overflow > 0 or delta_after_stop > 0:
            try:
                elapsed = (
                    now - self._last_rejection_warn_time
                    if self._last_rejection_warn_time > 0
                    else self._rejection_warn_interval
                )
                self.logger.warning(
                    "Backpressure warning: %d eventos descartados nos últimos %.1fs "
                    "(overflow=+%d [total=%d], after_stop=+%d [total=%d])",
                    delta_overflow + delta_after_stop,
                    elapsed,
                    delta_overflow,
                    cur_overflow,
                    delta_after_stop,
                    cur_after_stop,
                )
            except Exception:
                pass  # Proteção absoluta: erro no logger não mata worker
            self._last_warn_overflow = cur_overflow
            self._last_warn_after_stop = cur_after_stop
            self._last_rejection_warn_time = now

    def _process_single_flush_batch(self) -> bool:
        """
        Retira um lote do buffer e executa persistência fora de _buffer_lock.
        Retorna True se houve eventos processados, False se buffer estava vazio.
        """
        with self._buffer_lock:
            if not self._write_buffer:
                return False
            batch = self._write_buffer.copy()
            self._write_buffer.clear()
            self._events_in_flight += len(batch)

        try:
            self._flush_buffer(batch)
        except asyncio.CancelledError:
            with self._buffer_lock:
                self._events_in_flight -= len(batch)
                self._events_failed_terminal += len(batch)
            raise
        except Exception as e:
            self.logger.error(f"Erro inesperado durante flush do lote no worker: {e}", exc_info=True)
            with self._buffer_lock:
                self._events_in_flight -= len(batch)
                self._events_failed_terminal += len(batch)
        else:
            with self._buffer_lock:
                self._events_in_flight -= len(batch)
                self._events_finalized += len(batch)

        return True

    def _flush_buffer(self, events: List[Dict]):
        """Escreve eventos em lote no banco de dados e log visual (Gate G2 / H1-B2-Q3)."""

        valid_events = []
        quarantined_ids = set()
        collision_ids = set()

        for event in events:
            raw_copy = copy.deepcopy(event)
            if getattr(self, "enforce_financial_validation", True):
                val_res = self._validate_financial_payload(event)
                if val_res is not None and not val_res.is_valid:
                    self.logger.warning(
                        f"Evento rejeitado pré-persistência no flush do EventSaver: {val_res.rejection_reason} "
                        f"(campo: {val_res.rejected_field})"
                    )
                    ev_id = str(event.get("event_id") or "")
                    if ev_id:
                        quarantined_ids.add(ev_id)
                    if self.quarantine is not None:
                        try:
                            self.quarantine.quarantine_event(
                                raw_copy,
                                val_res,
                                source_caller="EventSaver._flush_buffer",
                            )
                        except Exception as q_err:
                            self.logger.error(f"Falha ao quarentenar no flush: {q_err}")
                    continue
            try:
                valid_events.append(canonicalize_structure(event))
            except Exception:
                valid_events.append(event)

        # 1. Escrita no SQLite (Principal)
        if self.db and valid_events:
            try:
                batch_res = self.db.save_batch(valid_events)
                if _is_valid_batch_save_result(batch_res, len(valid_events)):
                    if getattr(batch_res, "error_type", None) == "IntegrityCollisionError" or "Colisão divergente" in str(getattr(batch_res, "error_message", "")):
                        self.logger.error(f"IntegrityCollision detectado no retorno do save_batch: {batch_res.error_message}")
                        m = re.search(r"ID '([^']+)'", str(batch_res.error_message))
                        if m:
                            collision_ids.add(m.group(1))
                        else:
                            for ve in valid_events:
                                cid = str(ve.get("event_id") or "")
                                if cid:
                                    collision_ids.add(cid)
                        self._inc_stat("_sqlite_events_attempted", len(valid_events))
                        self._inc_stat("_sqlite_events_failed", len(valid_events))
                    else:
                        self._inc_stat("_sqlite_events_attempted", batch_res.total_received)
                        self._inc_stat("_sqlite_events_rejected_pre_commit", batch_res.rejected_pre_commit)
                        if batch_res.commit_status == "COMMITTED":
                            self._inc_stat("_sqlite_events_persisted", batch_res.persisted)
                            self._inc_stat("_sqlite_events_failed", batch_res.rejected_pre_commit)
                        elif batch_res.commit_status == "FAILED":
                            self._inc_stat("_sqlite_events_failed", batch_res.total_received)
                        elif batch_res.commit_status == "UNKNOWN":
                            self._inc_stat("_sqlite_events_persisted", batch_res.persisted)
                            self._inc_stat("_sqlite_events_unknown", batch_res.total_received - batch_res.persisted)
                else:
                    self._inc_stat("_sqlite_events_attempted", len(valid_events))
                    self._inc_stat("_sqlite_events_unknown", len(valid_events))
            except IntegrityCollisionError as coll_err:
                self.logger.error(f"IntegrityCollisionError detectado no flush do EventSaver: {coll_err}")
                cid = getattr(coll_err, "identity_id", None)
                if cid:
                    collision_ids.add(str(cid))
                if self.quarantine is not None:
                    try:
                        col_raw = getattr(coll_err, "incoming_payload", None) or getattr(coll_err, "raw_payload", None)
                        if col_raw is not None and isinstance(col_raw, dict):
                            col_res = FinancialValidationResult(
                                is_valid=False, status="INVALID", category="COLLISION",
                                rejected_field="event_id", rejected_value_repr=str(cid),
                                rejection_reason="INTEGRITY_COLLISION_DIVERGENT_PAYLOAD",
                            )
                            self.quarantine.quarantine_event(
                                col_raw, col_res, source_caller="EventSaver._flush_buffer", category="COLLISION"
                            )
                    except Exception as q_err:
                        self.logger.error(f"Erro ao registrar colisão na quarentena: {q_err}")
                self._inc_stat("_sqlite_events_attempted", len(valid_events))
                self._inc_stat("_sqlite_events_failed", len(valid_events))
            except Exception as e:
                self.logger.error(f"Erro ao salvar batch no SQLite: {e}")
                self._inc_stat("_sqlite_events_attempted", len(valid_events))
                self._inc_stat("_sqlite_events_failed", len(valid_events))
                # Fallback operacional acionado apenas para eventos válidos sob falha do SQLite
                for ve in valid_events:
                    try:
                        self._save_fallback(ve, "jsonl")
                    except Exception as fb_err:
                        self.logger.error(f"Erro ao salvar fallback por falha SQLite: {fb_err}")

        # 2. Escrita no Log Visual (Debug) + JSONs (apenas eventos válidos e não colidentes)
        for event in valid_events:
            ev_id = str(event.get("event_id") or "")
            if ev_id in quarantined_ids or ev_id in collision_ids:
                continue
            try:
                # Adiciona separador ANTES de processar evento, se necessario
                if event.get("_needs_separator"):
                    self._add_visual_separator(event)
                    event.pop("_needs_separator", None)

                cleaned_event = self._clean_event_data(event)
                if cleaned_event:
                    if self.write_json:
                        self._save_to_json(cleaned_event)
                    if self.write_jsonl:
                        self._save_to_jsonl(cleaned_event)

                    log_event = cleaned_event
                    # Injeta ID no log visual para facilitar grep
                    if "event_id" in event:
                        log_event = dict(cleaned_event)
                        log_event["_log_id"] = event["event_id"]
                    self._add_visual_log_entry(log_event)
            except Exception as e:
                self.logger.error(f"Erro ao processar evento no flush visual: {e}")


    @property
    def quarantine(self):
        """Retorna instância de FinancialQuarantine instanciada sob demanda (lazy)."""
        if self._quarantine is None:
            if not HAS_FINANCIAL_QUARANTINE or FinancialQuarantine is None:
                return None
            target_dir = self._quarantine_dir
            if target_dir is None:
                base_dir = self.history_file.parent if hasattr(self, "history_file") and self.history_file else DATA_DIR
                target_dir = base_dir / "quarantine"
            self._quarantine = FinancialQuarantine(quarantine_dir=target_dir)
        return self._quarantine

    def _validate_financial_payload(self, event: Dict[str, Any]) -> Any:
        """
        Valida semanticamente o payload financeiro antes de qualquer escrita física (Gate H1-B2-Q3).
        Aplica regras puras de validação, limites de microestrutura (bottom <= top, val <= poc <= vah),
        proibição de booleanos em campos numéricos, rejeição de NaN/±Inf e preços <= 0.
        """
        if not HAS_FINANCIAL_VALIDATOR or validate_financial_event_semantics is None:
            return None

        val_res = validate_financial_event_semantics(event)

        def _check_microstructure(e: Dict[str, Any], cat: str):
            for bound_key in ("top", "bottom"):
                if bound_key in e:
                    v = e[bound_key]
                    if isinstance(v, bool):
                        return FinancialValidationResult(
                            is_valid=False, status="INVALID", category=cat,
                            rejected_field=bound_key, rejected_value_repr=str(v),
                            rejection_reason="BOOLEAN_NOT_ALLOWED_IN_ZONE_BOUNDS",
                        )
                    if isinstance(v, (int, float)):
                        if math.isnan(v):
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category=cat,
                                rejected_field=bound_key, rejected_value_repr="nan",
                                rejection_reason="NAN_IN_ZONE_BOUNDS",
                            )
                        if math.isinf(v):
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category=cat,
                                rejected_field=bound_key, rejected_value_repr=str(v),
                                rejection_reason="INFINITY_IN_ZONE_BOUNDS",
                            )
                        if v <= 0:
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category=cat,
                                rejected_field=bound_key, rejected_value_repr=str(v),
                                rejection_reason="ZONE_BOUNDS_MUST_BE_POSITIVE",
                            )
            if "top" in e and "bottom" in e:
                try:
                    top_v = float(e["top"])
                    bot_v = float(e["bottom"])
                    if not (math.isnan(top_v) or math.isnan(bot_v) or math.isinf(top_v) or math.isinf(bot_v)):
                        if bot_v > top_v:
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category=cat,
                                rejected_field="bottom", rejected_value_repr=f"bottom={bot_v} > top={top_v}",
                                rejection_reason="INVALID_MICROSTRUCTURE_BOUNDS: bottom > top",
                            )
                except (ValueError, TypeError):
                    pass
            if "poc" in e and "vah" in e:
                try:
                    poc_v = float(e["poc"])
                    vah_v = float(e["vah"])
                    if not (math.isnan(poc_v) or math.isnan(vah_v) or math.isinf(poc_v) or math.isinf(vah_v)):
                        if poc_v > vah_v:
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category=cat,
                                rejected_field="poc", rejected_value_repr=f"poc={poc_v} > vah={vah_v}",
                                rejection_reason="INVALID_MICROSTRUCTURE_BOUNDS: poc > vah",
                            )
                except (ValueError, TypeError):
                    pass
            if "val" in e and "poc" in e:
                try:
                    val_v = float(e["val"])
                    poc_v = float(e["poc"])
                    if not (math.isnan(val_v) or math.isnan(poc_v) or math.isinf(val_v) or math.isinf(poc_v)):
                        if val_v > poc_v:
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category=cat,
                                rejected_field="val", rejected_value_repr=f"val={val_v} > poc={poc_v}",
                                rejection_reason="INVALID_MICROSTRUCTURE_BOUNDS: val > poc",
                            )
                except (ValueError, TypeError):
                    pass
            return None

        if val_res.status == "VALID":
            micro_err = _check_microstructure(event, val_res.category)
            if micro_err:
                return micro_err
            if "last_touched" in event:
                try:
                    normalize_last_touched(event["last_touched"])
                except Exception as lt_err:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category=val_res.category,
                        rejected_field="last_touched",
                        rejected_value_repr=str(event.get("last_touched")),
                        rejection_reason=f"INVALID_LAST_TOUCHED: {lt_err}",
                    )
            return val_res
        elif val_res.status == "INVALID":
            if val_res.rejection_reason == "MANDATORY_FIELD_MISSING":
                micro_err = _check_microstructure(event, val_res.category)
                if micro_err:
                    return micro_err
                for price_key in ("price", "preco", "preco_fechamento", "entry_price", "anchor_price", "level", "poc", "val", "vah", "top", "bottom"):
                    if price_key in event:
                        v = event[price_key]
                        if isinstance(v, bool):
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category=val_res.category,
                                rejected_field=price_key, rejected_value_repr=str(v),
                                rejection_reason="BOOLEAN_NOT_ALLOWED_IN_PRICE",
                            )
                        if isinstance(v, (int, float)):
                            if math.isnan(v):
                                return FinancialValidationResult(
                                    is_valid=False, status="INVALID", category=val_res.category,
                                    rejected_field=price_key, rejected_value_repr="nan",
                                    rejection_reason="NAN_IN_PRICE_FIELD",
                                )
                            if math.isinf(v):
                                return FinancialValidationResult(
                                    is_valid=False, status="INVALID", category=val_res.category,
                                    rejected_field=price_key, rejected_value_repr=str(v),
                                    rejection_reason="INFINITY_IN_PRICE_FIELD",
                                )
                            if v <= 0:
                                return FinancialValidationResult(
                                    is_valid=False, status="INVALID", category=val_res.category,
                                    rejected_field=price_key, rejected_value_repr=str(v),
                                    rejection_reason="PRICE_MUST_BE_POSITIVE",
                                )
                for num_key in ("delta", "volume", "volume_total", "quantity", "quantidade"):
                    if num_key in event:
                        v = event[num_key]
                        if isinstance(v, bool):
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category=val_res.category,
                                rejected_field=num_key, rejected_value_repr=str(v),
                                rejection_reason="BOOLEAN_NOT_ALLOWED_IN_NUMERIC_FIELD",
                            )
                        if isinstance(v, (int, float)):
                            if math.isnan(v):
                                return FinancialValidationResult(
                                    is_valid=False, status="INVALID", category=val_res.category,
                                    rejected_field=num_key, rejected_value_repr="nan",
                                    rejection_reason="NAN_IN_NUMERIC_FIELD",
                                )
                            if math.isinf(v):
                                return FinancialValidationResult(
                                    is_valid=False, status="INVALID", category=val_res.category,
                                    rejected_field=num_key, rejected_value_repr=str(v),
                                    rejection_reason="INFINITY_IN_NUMERIC_FIELD",
                                )
                return FinancialValidationResult(
                    is_valid=True,
                    status="VALID",
                    category=val_res.category,
                )
            return val_res
        else:
            micro_err = _check_microstructure(event, "UNCLASSIFIED_FINANCIAL")
            if micro_err:
                return micro_err
            for price_key in ("price", "preco", "preco_fechamento", "entry_price", "anchor_price", "level", "poc", "val", "vah"):
                if price_key in event:
                    v = event[price_key]
                    if isinstance(v, bool):
                        return FinancialValidationResult(
                            is_valid=False, status="INVALID", category="UNCLASSIFIED_FINANCIAL",
                            rejected_field=price_key, rejected_value_repr=str(v),
                            rejection_reason="BOOLEAN_NOT_ALLOWED_IN_PRICE",
                        )
                    if isinstance(v, (int, float)):
                        if math.isnan(v):
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=price_key, rejected_value_repr="nan",
                                rejection_reason="NAN_IN_PRICE_FIELD",
                            )
                        if math.isinf(v):
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=price_key, rejected_value_repr=str(v),
                                rejection_reason="INFINITY_IN_PRICE_FIELD",
                            )
                        if v <= 0:
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=price_key, rejected_value_repr=str(v),
                                rejection_reason="PRICE_MUST_BE_POSITIVE",
                            )
            for num_key in ("delta", "volume", "volume_total", "quantity", "quantidade"):
                if num_key in event:
                    v = event[num_key]
                    if isinstance(v, bool):
                        return FinancialValidationResult(
                            is_valid=False, status="INVALID", category="UNCLASSIFIED_FINANCIAL",
                            rejected_field=num_key, rejected_value_repr=str(v),
                            rejection_reason="BOOLEAN_NOT_ALLOWED_IN_NUMERIC_FIELD",
                        )
                    if isinstance(v, (int, float)):
                        if math.isnan(v):
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=num_key, rejected_value_repr="nan",
                                rejection_reason="NAN_IN_NUMERIC_FIELD",
                            )
                        if math.isinf(v):
                            return FinancialValidationResult(
                                is_valid=False, status="INVALID", category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=num_key, rejected_value_repr=str(v),
                                rejection_reason="INFINITY_IN_NUMERIC_FIELD",
                            )
            return FinancialValidationResult(is_valid=True, status="VALID", category="UNCLASSIFIED_PERMITTED")

    def save_events(self, events: List[Dict]):
        """API de conveniência para salvar múltiplos eventos em lote."""
        if not isinstance(events, (list, tuple)):
            return
        for e in events:
            self.save_event(e)

    def save_event(self, event: Dict):
        """
        API pública para salvar um evento.
        """
        if not isinstance(event, dict):
            self.logger.error("Evento inválido: não é um dicionário")
            return

        # Captura cópia profunda bruta antes de qualquer mutação ou sanitização (Gate H1-B2-Q3)
        raw_payload_copy = copy.deepcopy(event)

        # Validação financeira semântica na entrada (Gate H1-B2-Q3)
        if getattr(self, "enforce_financial_validation", True):
            val_res = self._validate_financial_payload(event)
            if val_res is not None and not val_res.is_valid:
                self.logger.warning(
                    f"Evento rejeitado pelo validador financeiro no EventSaver: {val_res.rejection_reason} "
                    f"(campo: {val_res.rejected_field}, valor: {val_res.rejected_value_repr})"
                )
                if self.quarantine is not None:
                    try:
                        self.quarantine.quarantine_event(
                            raw_payload_copy,
                            val_res,
                            source_caller="EventSaver.save_event",
                        )
                    except Exception as q_err:
                        self.logger.error(f"Falha ao registrar quarentena no EventSaver: {q_err}")
                return None

        # Canonicalização determinística de -0.0 -> +0.0 (Gate H1-B3-D / H1-B2-Q3)
        try:
            event = canonicalize_structure(event)
        except Exception:
            pass

        # ✅ OTIMIZAÇÃO AUTOMÁTICA
        if event.get("tipo_evento") == "ANALYSIS_TRIGGER":
            try:
                from data_processing.fix_optimization import clean_event, simplify_historical_vp, remove_enriched_snapshot
                event = clean_event(event)
                event = simplify_historical_vp(event)
                event = remove_enriched_snapshot(event)
            except Exception as e:
                self.logger.exception("Falha ao otimizar evento (pré-processamento)")

        # ETAPA 4 — bins internos do volume profile fora da representação persistida.
        # price_bins/volume_per_bin são necessários apenas internamente
        # (historical_profiler -> institutional_analytics -> value_area_volume_pct);
        # persisti-los inflaria o payload em ~100 KB/evento sem uso downstream.
        # strip_profile_bins opera sobre CÓPIA: o evento original (IA/consumidores)
        # permanece com os bins intactos.
        try:
            from data_processing.fix_optimization import strip_profile_bins
            event = strip_profile_bins(event)
        except Exception:
            self.logger.exception("Falha ao sanitizar historical_vp (bins) para persistência")

        if not isinstance(event, dict):
            self.logger.error("Evento inválido: não é um dicionário")
            return

        try:
            # ==== NORMALIZAÇÃO: janela_numero ====
            # Alguns eventos chegam com a janela apenas dentro de "raw_event".
            # Promove para o top-level para permitir:
            # - separador por janela no log visual
            # - cabeçalho "JANELA: X" no eventos_visuais.log
            if "janela_numero" not in event:
                raw_event = event.get("raw_event")
                janela_candidate = None
                if isinstance(raw_event, dict):
                    janela_candidate = raw_event.get("janela_numero")
                    if janela_candidate is None:
                        janela_candidate = raw_event.get("window_count")
                    if janela_candidate is None:
                        janela_candidate = raw_event.get("window_id")
                if janela_candidate is None:
                    janela_candidate = event.get("window_count")
                if janela_candidate is None:
                    janela_candidate = event.get("window_id")

                janela_num = None
                try:
                    if isinstance(janela_candidate, str):
                        m = re.search(r"(\d+)", janela_candidate)
                        if m:
                            janela_num = int(m.group(1))
                    elif isinstance(janela_candidate, (int, float)):
                        janela_num = int(janela_candidate)
                except Exception:
                    janela_num = None

                if janela_num and janela_num > 0:
                    event["janela_numero"] = janela_num

            # ==== TIMESTAMPS / CLOCK SYNC ====
            epoch_ms = event.get("epoch_ms")
            timestamp_str = event.get("timestamp")
            dt_utc: Optional[datetime] = None
            epoch_ms_int: Optional[int] = None

            if HAS_CLOCK_SYNC and _clock_sync_instance and _clock_sync_instance.is_synced():
                if not epoch_ms:
                    epoch_ms = _clock_sync_instance.get_server_time_ms()
                    event["epoch_ms"] = epoch_ms

            # 🆕 DEDUPLICAÇÃO DE TIMESTAMPS
            # Detecta e avisa quando eventos com timestamps duplicados são salvos.
            # Preserva integralmente o payload original (sem mutação artificial de epoch_ms)
            # quando o evento possui event_id explícito ou validação financeira está ativa.
            if epoch_ms is not None:
                try:
                    epoch_ms_int_candidate = int(epoch_ms)
                    if (not getattr(self, "enforce_financial_validation", True)) and ("event_id" not in event) and (self._last_event_timestamp_ms is not None and epoch_ms_int_candidate == self._last_event_timestamp_ms):
                        self._duplicate_event_count += 1
                        self.logger.warning(
                            f"⚠️ Timestamp DUPLICADO detectado: {epoch_ms_int_candidate}ms "
                            f"(ocorrência #{self._duplicate_event_count} nesta sequência) "
                            f"| evento: {event.get('tipo_evento', 'unknown')}"
                        )
                        # Ajustar timestamp para evitar colisão apenas em modo não financeiro sem ID
                        epoch_ms_int = epoch_ms_int_candidate + self._duplicate_event_count
                        event["epoch_ms"] = epoch_ms_int
                        event["_dedup_adjusted"] = True
                        self.logger.info(f"✅ Timestamp ajustado para: {epoch_ms_int}ms")
                    elif epoch_ms_int_candidate - (self._last_event_timestamp_ms or 0) > self._max_timestamp_gap_ms * 10:
                        # Reset contador se houver gap grande
                        self._duplicate_event_count = 0
                        epoch_ms_int = epoch_ms_int_candidate
                        self._last_event_timestamp_ms = epoch_ms_int
                    else:
                        self._duplicate_event_count = 0
                        epoch_ms_int = epoch_ms_int_candidate
                        self._last_event_timestamp_ms = epoch_ms_int
                except Exception as dedup_err:
                    self.logger.debug(f"Erro na deduplicação: {dedup_err}")

            # 🆕 Garante ID único para rastreabilidade
            if "event_id" not in event:
                # Gera ID determinístico baseado em timestamp e tipo se possível, ou aleatório
                import uuid
                event["event_id"] = str(uuid.uuid4())[:8]

            if epoch_ms is not None and isinstance(epoch_ms, (int, float, str)):
                try:
                    if epoch_ms_int is None:
                        epoch_ms_int = int(epoch_ms)
                    if 946684800000 <= epoch_ms_int <= 4102444800000:
                        dt_utc = datetime.fromtimestamp(epoch_ms_int / 1000, tz=UTC_TZ)
                    else:
                        self.logger.error(f"epoch_ms fora de range plausível: {epoch_ms_int}")
                except Exception as e:
                    self.logger.error(f"Erro ao converter epoch_ms: {e}")

            if dt_utc is None and timestamp_str:
                try:
                    dt_parsed = self._parse_iso8601(timestamp_str)
                    dt_utc = dt_parsed.astimezone(UTC_TZ)
                    if epoch_ms_int is None:
                        epoch_ms_int = int(dt_utc.timestamp() * 1000)
                        event.setdefault("epoch_ms", epoch_ms_int)
                except Exception as e:
                    self.logger.error(f"timestamp inválido: {e}")

            if dt_utc is not None:
                now_utc = datetime.now(UTC_TZ)
                time_diff = abs((dt_utc - now_utc).total_seconds())
                event["data_context"] = "historical" if time_diff > 86400 else "real_time"

                time_index = None
                if epoch_ms_int is not None and hasattr(self.time_manager, "build_time_index"):
                    try:
                        time_index = self.time_manager.build_time_index(
                            epoch_ms_int, include_local=True, timespec="milliseconds"
                        )
                    except Exception:
                        time_index = None

                if isinstance(time_index, dict):
                    for k, v in time_index.items():
                        event.setdefault(k, v)
                    if "timestamp" not in event and "timestamp_utc" in time_index:
                        event["timestamp"] = time_index["timestamp_utc"]
                else:
                    event.setdefault("timestamp_utc", dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC"))
                    if "timestamp" not in event:
                        event["timestamp"] = dt_utc.isoformat(timespec="milliseconds")
            else:
                event.setdefault("data_context", "unknown")

            # Enriquece eventos simples
            if event.get("tipo_evento") == "Alerta" and "context" in event:
                context = event["context"]
                enriched = {
                    "price_data": {
                        "current": {"last": context.get("price"), "volume": context.get("volume")}
                    },
                    "volatility_metrics": {"realized_vol_24h": context.get("volatility")},
                    "market_context": {"trading_session": "NY_OVERLAP", "session_phase": "ACTIVE"}
                }
                event.update(enriched)

            # ==== DETECÇÃO DE NOVA JANELA ====
            if "janela_numero" in event:
                janela_num = event["janela_numero"]
                if janela_num != self._last_janela_numero:
                    event["_needs_separator"] = True
                    self._last_janela_numero = janela_num

            # Buffer com limite bounded e contabilidade mínima (Gate G1)
            with self._buffer_lock:
                self._events_received += 1
                if self._stop_event.is_set():
                    self._events_rejected_after_stop += 1
                    return None

                if len(self._write_buffer) >= MAX_BUFFER_SIZE:
                    self._events_rejected_overflow += 1
                    self._buffer_overflow_count = self._events_rejected_overflow
                    self._wake_flush_event.set()
                    return None

                self._write_buffer.append(event)
                self._events_accepted += 1

                # Sinalização antecipada de wake-up para backpressure
                wake_threshold = min(BACKPRESSURE_WAKE_THRESHOLD, max(1, MAX_BUFFER_SIZE // 2))
                if len(self._write_buffer) >= wake_threshold:
                    self._wake_flush_event.set()

                # Heartbeat de progresso: evento enfileirado com sucesso
                if self.health_monitor is not None:
                    try:
                        self.health_monitor.heartbeat("event_saver")
                    except Exception:
                        pass

            # Alerta sonoro (FASE E3-A): nunca bloqueia o hot path.
            if self.sound_alert and event.get("is_signal", False):
                self._alert_sound_async()

        except Exception as e:
            self.logger.error(f"Erro crítico ao processar evento: {e}", exc_info=True)

    def _rotate_visual_log_if_needed(self) -> bool:
        """
        Executa rotação atômica por tamanho do log visual (Gate G4-A).
        Deve ser executado EXCLUSIVAMENTE sob o lock de arquivo do log visual.
        Retorna True se houve rotação com sucesso ou se não era necessária;
        Retorna False se houve falha na rotação.
        """
        if not self.write_visual_log:
            return True

        if self.visual_log_backups <= 0:
            return True

        try:
            if not self.visual_log_file.exists():
                return True
            current_size = self.visual_log_file.stat().st_size
        except Exception as e:
            self.logger.warning(f"Falha ao obter tamanho do log visual para rotação: {e}")
            return False

        if current_size < self.max_visual_log_bytes:
            return True

        # Rotação necessária
        try:
            # Cascata de backups de trás para frente (ex: log.1 -> log.2)
            for i in range(self.visual_log_backups - 1, 0, -1):
                sfn = Path(f"{self.visual_log_file}.{i}")
                dfn = Path(f"{self.visual_log_file}.{i + 1}")
                if sfn.exists():
                    try:
                        os.replace(sfn, dfn)
                    except Exception as e:
                        self.logger.warning(
                            f"Falha ao rotacionar backup {sfn} para {dfn}: {e}. Interrompendo rotação."
                        )
                        return False

            # Rotaciona o arquivo ativo para .1
            target_first = Path(f"{self.visual_log_file}.1")
            try:
                os.replace(self.visual_log_file, target_first)
                self.logger.info(
                    f"Log visual rotacionado com sucesso: {self.visual_log_file} -> {target_first} ({current_size} bytes)"
                )
                return True
            except Exception as e:
                self.logger.warning(
                    f"Falha ao rotacionar arquivo ativo {self.visual_log_file} para {target_first}: {e}"
                )
                return False
        except Exception as e:
            self.logger.error(f"Erro inesperado durante rotação do log visual: {e}")
            return False

    @staticmethod
    def _safe_move_segment_no_overwrite(src: Path, dst: Path) -> bool:
        """
        Move arquivo de src para dst com garantia absoluta anti-sobrescrita
        compatível com Windows e Linux (Gate G4-C Hardened).

        Windows:
          Usa os.rename (MoveFileW), que falha nativamente com FileExistsError
          se dst já existir, sem jamais sobrescrever.

        Linux:
          Usa renameat2 com RENAME_NOREPLACE (atômico de passo único no mesmo FS).
          Se dst já existir, retorna EEXIST (FileExistsError).
          Se renameat2 não estiver disponível no kernel/glibc ou o FS retornar
          EOPNOTSUPP/ENOSYS/EINVAL, a operação é rejeitada (NÃO há fallback para
          os.rename destrutivo, nem cópia com exclusão, nem hardlinks residuais),
          adiando a rotação para preservar a integridade dos eventos e a imutabilidade do histórico.
        """
        if not src.exists():
            return False

        sys_name = platform.system()

        if sys_name == "Windows":
            # No Windows NTFS, os.rename chama MoveFileW, que falha nativamente
            # com FileExistsError se dst existir, sem jamais sobrescrever.
            os.rename(src, dst)
            return True

        elif sys_name == "Linux":
            renameat2_func = _get_linux_renameat2()
            if renameat2_func is None:
                raise NotImplementedError(
                    "renameat2 com RENAME_NOREPLACE não disponível no ambiente Linux. "
                    "Rotação adiada para evitar sobrescrita ou hardlinks residuais."
                )

            AT_FDCWD = -100
            RENAME_NOREPLACE = 1

            src_bytes = os.fsencode(str(src))
            dst_bytes = os.fsencode(str(dst))

            ret = renameat2_func(AT_FDCWD, src_bytes, AT_FDCWD, dst_bytes, RENAME_NOREPLACE)
            if ret != 0:
                import ctypes
                errno_val = ctypes.get_errno()
                if errno_val == errno.EEXIST:
                    raise FileExistsError(
                        errno.EEXIST, f"Destino {dst} já existe (EEXIST via renameat2)", str(dst)
                    )
                elif errno_val in (errno.EACCES, errno.EPERM):
                    raise PermissionError(
                        errno_val, f"Permissão negada ao renomear {src} para {dst}", str(src)
                    )
                elif errno_val == errno.ENOSYS:
                    raise OSError(
                        errno.ENOSYS, "renameat2 syscall não implementada no kernel Linux (ENOSYS)"
                    )
                elif hasattr(errno, "EOPNOTSUPP") and errno_val == errno.EOPNOTSUPP:
                    raise OSError(
                        errno.EOPNOTSUPP, "renameat2(RENAME_NOREPLACE) não suportado pelo filesystem (EOPNOTSUPP)"
                    )
                elif errno_val == errno.EINVAL:
                    raise OSError(
                        errno.EINVAL, "Flag inválida ou filesystem sem suporte a RENAME_NOREPLACE (EINVAL)"
                    )
                elif errno_val == errno.EXDEV:
                    raise OSError(
                        errno.EXDEV, f"Origem e destino em pontos de montagem distintos (EXDEV)"
                    )
                elif errno_val == errno.ENOSPC:
                    raise OSError(
                        errno.ENOSPC, f"Sem espaço no disco ao renomear segmento (ENOSPC)"
                    )
                else:
                    raise OSError(
                        errno_val, f"Erro ao executar renameat2: {os.strerror(errno_val)}"
                    )
            return True
        else:
            raise NotImplementedError(
                f"Sistema operacional '{sys_name}' sem suporte a renomeação garantida anti-sobrescrita. Rotação adiada."
            )

    def _rotate_jsonl_if_needed(self) -> bool:
        """
        Executa rotação não destrutiva por tamanho do JSONL principal (Gate G4-C).
        Move o arquivo ativo para dados/archive/eventos_fluxo_YYYYMMDD_HHMMSS_MONOTONIC_SEQ.jsonl.
        Garante:
          - Zero sobrescrita de segmentos existentes;
          - Prioridade total à persistência de eventos;
          - Se a rotação falhar, continua gravando no arquivo ativo existente;
          - Ignorada durante shutdown para priorizar drenagem rápida e bounded do buffer.
        Retorna True se houve rotação com sucesso ou se não era necessária;
        Retorna False se houve adiamento/falha na rotação.
        """
        if not self.write_jsonl:
            return True

        if not self.auto_archive_jsonl:
            return True

        # Durante shutdown, NUNCA inicia rotação: prioridade absoluta à drenagem (Gate G4-C / Etapa 4)
        if self._stop_event.is_set():
            return False

        try:
            if not self.history_file.exists():
                return True
            current_size = self.history_file.stat().st_size
        except Exception as e:
            self.logger.warning("Falha ao obter tamanho do JSONL para verificação de rotação: %s", e)
            return False

        if current_size < self.max_jsonl_file_bytes:
            return True

        # Limiar de tamanho atingido: rotação necessária
        try:
            archive_dir = self.archive_dir
            archive_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.warning(
                "Falha ao criar diretório de arquivamento '%s': %s. Adiada rotação.",
                self.archive_dir,
                e,
            )
            return False

        stamp_utc = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        monotonic_ms = int((time.monotonic_ns() // 1_000_000) % 1_000_000)

        # Loop anti-colisão com garantia anti-sobrescrita (máximo 10 tentativas)
        rotated = False
        for seq in range(1, 11):
            segment_name = f"eventos_fluxo_{stamp_utc}_{monotonic_ms:06d}_{seq:03d}.jsonl"
            candidate_dst = archive_dir / segment_name

            if candidate_dst.exists():
                continue

            try:
                self._safe_move_segment_no_overwrite(self.history_file, candidate_dst)
                rotated = True
                self.logger.info(
                    "JSONL principal rotacionado com sucesso: %s -> %s (%d bytes)",
                    self.history_file,
                    candidate_dst,
                    current_size,
                )
                break
            except FileExistsError:
                self.logger.debug(
                    "Destino %s já existente durante movimentação atômica; incrementando sequência.",
                    candidate_dst,
                )
                continue
            except (PermissionError, OSError, NotImplementedError) as e:
                # Lock externo no Windows (WinError 32), renameat2 ausente/não suportado (Linux) ou erro de SO:
                # PRESERVAR EVENTOS: adia rotação e continua gravando no arquivo ativo
                self.logger.warning(
                    "Falha ao mover arquivo ativo para %s (%s: %s). Rotação adiada para preservar eventos.",
                    candidate_dst,
                    type(e).__name__,
                    e,
                )
                return False

        if not rotated:
            self.logger.warning(
                "Não foi possível encontrar nome exclusivo para segmento após 10 tentativas. Rotação adiada."
            )
            return False

        return True

    def _add_visual_separator(self, event: Dict):
        """Adiciona separador visual para nova janela de tempo."""
        if not self.write_visual_log:
            return

        janela_num = event.get('janela_numero')
        if not janela_num:
            return

        with self._janelas_lock:
            if janela_num in self._janelas_processadas:
                return
            self._janelas_processadas.add(janela_num)

        # Cabeçalho simples por janela (pedido: identificar no início de cada janela)
        # Mantemos o lock de arquivo já existente para evitar escrita concorrente.
        try:
            window_num = int(janela_num)
        except Exception:
            window_num = None
        if window_num and window_num > 0:
            separator = "\n" + ("-" * 100) + "\n" + f" # Janela {window_num}\n"
            lock_file = None
            lock_acquired = False
            lock_file_path = self.visual_log_file.with_suffix(".lock")
            try:
                lock_file = open(lock_file_path, "a")
                lock_acquired = acquire_file_lock(
                    lock_file, blocking=True, timeout=10.0
                )
                if lock_acquired:
                    self._rotate_visual_log_if_needed()
                    with open(self.visual_log_file, "a", encoding="utf-8") as f:
                        f.write(separator)
                        f.flush()
            except Exception as e:
                self.logger.error(f"Erro ao escrever separador: {e}")
            finally:
                if lock_file:
                    try:
                        if lock_acquired:
                            release_file_lock(lock_file)
                    except Exception:
                        pass
                    try:
                        lock_file.close()
                    except Exception:
                        pass
            return

        try:
            epoch_ms = event.get("epoch_ms") or event.get("window_close_ms")
            dt_utc: Optional[datetime] = None

            if epoch_ms and isinstance(epoch_ms, (int, float, str)):
                try:
                    epoch_ms_int = int(epoch_ms)
                    dt_utc = datetime.fromtimestamp(epoch_ms_int / 1000, tz=UTC_TZ)
                except Exception:
                    pass

            if dt_utc is None:
                dt_utc = datetime.now(UTC_TZ)

            if TIMEZONE_AVAILABLE or hasattr(NY_TZ, 'get_offset'):
                dt_ny = self._convert_timezone(dt_utc, NY_TZ)
                dt_sp = self._convert_timezone(dt_utc, SP_TZ)
                timestamp_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                timestamp_ny = dt_ny.strftime("%Y-%m-%d %H:%M:%S EST/EDT")
                timestamp_sp = dt_sp.strftime("%Y-%m-%d %H:%M:%S BRT")
            else:
                dt_ny = dt_utc.replace(tzinfo=UTC_TZ).astimezone(NY_TZ)
                dt_sp = dt_utc.replace(tzinfo=UTC_TZ).astimezone(SP_TZ)
                timestamp_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                timestamp_ny = dt_ny.strftime("%Y-%m-%d %H:%M:%S") + " (EST)"
                timestamp_sp = dt_sp.strftime("%Y-%m-%d %H:%M:%S") + " (BRT)"

            window_num = janela_num
            separator = f"\n{'='*100}\n"
            separator += f"🗓️  JANELA {window_num}\n"
            separator += f"🕒 UTC: {timestamp_utc}\n"
            separator += f"🗽 New York: {timestamp_ny}\n"
            separator += f"📍 São Paulo: {timestamp_sp}\n"
            separator += f"📊 Contexto: {event.get('data_context', 'real_time')}\n"
            separator += f"{'='*100}\n"

            lock_file = None
            lock_acquired = False

            try:
                lock_file_path = self.visual_log_file.with_suffix('.lock')
                lock_file = open(lock_file_path, 'a')
                lock_acquired = acquire_file_lock(lock_file, blocking=True, timeout=10.0)

                if lock_acquired:
                    self._rotate_visual_log_if_needed()
                    with open(self.visual_log_file, "a", encoding="utf-8") as f:
                        f.write(separator)
                        f.flush()
                else:
                    self.logger.warning("Timeout ao adquirir lock do visual log")
            except Exception as e:
                self.logger.error(f"Erro ao escrever separador: {e}")
            finally:
                if lock_file:
                    try:
                        if lock_acquired:
                            release_file_lock(lock_file)
                    except Exception:
                        pass
                    try:
                        lock_file.close()
                    except Exception:
                        pass

        except Exception as e:
            self.logger.error(f"Erro ao adicionar separador: {e}")

    def _format_value_for_display(self, value: Any, key: str = "") -> str:
        """Formata valor para exibição legível."""
        if value is None:
            return "null"
        if not HAS_FORMAT_UTILS:
            if isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (int, float)):
                return f"{value:,.4f}" if isinstance(value, float) else str(value)
            else:
                return str(value)

        key_lower = key.lower()
        try:
            if isinstance(value, bool):
                return str(value).lower()
            elif isinstance(value, (int, float)):
                if any(x in key_lower for x in ['price', 'preco', 'poc', 'val', 'vah']):
                    return format_price(value)
                elif any(x in key_lower for x in ['volume', 'quantity', 'size']):
                    return format_large_number(value)
                elif 'delta' in key_lower:
                    return format_delta(value)
                elif any(x in key_lower for x in ['pct', 'percent', 'ratio']):
                    return format_percent(value)
                elif any(x in key_lower for x in ['volatility', 'returns', 'slope']):
                    return format_scientific(value)
                else:
                    return str(value)
            else:
                return str(value)
        except Exception:
            return str(value)

    def _add_visual_log_entry(self, event: Dict):
        """Adiciona entrada formatada ao log visual."""
        if not self.write_visual_log:
            return

        try:
            epoch_ms = event.get("epoch_ms")
            timestamp = event.get("timestamp")
            dt_utc: Optional[datetime] = None

            if epoch_ms and isinstance(epoch_ms, (int, float, str)):
                try:
                    epoch_ms_int = int(epoch_ms)
                    dt_utc = datetime.fromtimestamp(epoch_ms_int / 1000, tz=UTC_TZ)
                except Exception:
                    dt_utc = None

            if dt_utc is None and timestamp:
                try:
                    dt = self._parse_iso8601(timestamp)
                    if dt.tzinfo is None:
                        dt_utc = dt.replace(tzinfo=UTC_TZ)
                    else:
                        dt_utc = dt.astimezone(UTC_TZ)
                except Exception:
                    dt_utc = None

            if dt_utc is None:
                dt_utc = datetime.now(UTC_TZ)

            if TIMEZONE_AVAILABLE or hasattr(NY_TZ, 'get_offset'):
                dt_ny = self._convert_timezone(dt_utc, NY_TZ)
                dt_sp = self._convert_timezone(dt_utc, SP_TZ)
                ts_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                ts_ny = dt_ny.strftime("%Y-%m-%d %H:%M:%S EST/EDT")
                ts_sp = dt_sp.strftime("%Y-%m-%d %H:%M:%S BRT")
            else:
                dt_ny = dt_utc.replace(tzinfo=UTC_TZ).astimezone(NY_TZ)
                dt_sp = dt_utc.replace(tzinfo=UTC_TZ).astimezone(SP_TZ)
                ts_utc = dt_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                ts_ny = dt_ny.strftime("%Y-%m-%d %H:%M:%S") + " (EST)"
                ts_sp = dt_sp.strftime("%Y-%m-%d %H:%M:%S") + " (BRT)"

            tipo = event.get("tipo_evento", event.get("type", "GENÉRICO"))
            symbol = event.get("symbol") or event.get("par") or event.get("ativo") or ""
            janela = event.get("janela_numero")
            if janela is None:
                raw_event = event.get("raw_event")
                if isinstance(raw_event, dict):
                    janela = (
                        raw_event.get("janela_numero")
                        or raw_event.get("window_count")
                        or raw_event.get("window_id")
                    )
                if isinstance(janela, str):
                    try:
                        m = re.search(r"(\d+)", janela)
                        janela = int(m.group(1)) if m else None
                    except Exception:
                        janela = None

            header_lines = ["-" * 100]
            title_parts = [f"EVENTO: {tipo}"]
            if symbol:
                title_parts.append(f"SYMBOL: {symbol}")
            if janela is not None:
                title_parts.append(f"JANELA: {janela}")
            header_lines.append(" | ".join(title_parts))
            header_lines.append(f"UTC: {ts_utc}")
            header_lines.append(f"NY:  {ts_ny}")
            header_lines.append(f"SP:  {ts_sp}")
            header_lines.append("-" * 100)

            header = "\n".join(header_lines) + "\n"

            visual_event = self._prepare_visual_event(event)
            try:
                json_block = json.dumps(visual_event, indent=2, ensure_ascii=False, default=str)
            except Exception:
                json_block = str(visual_event)

            json_block = self._optimize_json_display(json_block)
            entry = header + json_block + "\n"
            entry = self._fix_utf8_visual(entry)

            lock_file = None
            lock_acquired = False
            lock_file_path = self.visual_log_file.with_suffix('.lock')
            try:
                lock_file = open(lock_file_path, 'a')
                lock_acquired = acquire_file_lock(lock_file, blocking=True, timeout=10.0)
                if lock_acquired:
                    self._rotate_visual_log_if_needed()
                    with open(self.visual_log_file, "a", encoding="utf-8") as f:
                        f.write(entry)
                        f.flush()
                else:
                    self.logger.warning("Timeout ao adquirir lock do visual log para entrada de evento")
            finally:
                if lock_file:
                    try:
                        if lock_acquired:
                            release_file_lock(lock_file)
                    except Exception:
                        pass
                    try:
                        lock_file.close()
                    except Exception:
                        pass

        except Exception as e:
            self.logger.error(f"Erro ao adicionar entrada no log visual: {e}", exc_info=True)

    @staticmethod
    def _fix_utf8_visual(text: str) -> str:
        """Corrige encoding UTF-8 duplo em strings do log visual."""
        replacements = {
            "AbsorÃ§Ã£o": "Absorção",
            "AcumulaÃ§Ã£o": "Acumulação",
            "ManipulaÃ§Ã£o": "Manipulação",
            "DistribuiÃ§Ã£o": "Distribuição",
            "ConsolidaÃ§Ã£o": "Consolidação",
            "ExaustÃ£o": "Exaustão",
            "AgressÃ£o": "Agressão",
            "anÃ¡lise": "análise",
            "automÃ¡tico": "automático",
            "resistÃªncia": "resistência",
            "mÃ©dia": "média",
            "PreÃ§o": "Preço",
            "Ã­ndice": "índice",
            "Î": "Δ",
        }
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        return text

    def _compress_volume_nodes_for_visual(self, event: Dict) -> Dict:
        """Converte volume_nodes de [[price,vol,str],...] para [{"price":..,"vol":..,"str":..},...]."""
        def _convert_vn_dict(vn: Dict):
            for key in ("hvn_nodes", "lvn_nodes"):
                nodes = vn.get(key)
                if not isinstance(nodes, list) or not nodes:
                    continue
                # Limitar a 10 nós por tipo
                display_nodes = nodes[:5] + nodes[-5:] if len(nodes) > 10 else nodes
                converted = []
                for triplet in display_nodes:
                    if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                        try:
                            converted.append({
                                "price": round(float(triplet[0]), 0),
                                "vol": round(float(triplet[1]), 2),
                                "str": round(float(triplet[2]), 2),
                            })
                        except (ValueError, TypeError):
                            converted.append({"raw": str(triplet)})
                    elif isinstance(triplet, dict):
                        converted.append(triplet)  # Já é JSON
                    else:
                        converted.append({"raw": str(triplet)})
                vn[key] = converted

        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "volume_nodes" and isinstance(v, dict):
                        _convert_vn_dict(v)
                    else:
                        _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)

        try:
            _walk(event)
        except Exception as e:
            self.logger.debug(f"Erro ao converter volume_nodes para visual: {e}")
        return event

    def _optimize_json_display(self, json_str: str) -> str:
        """Otimiza exibição de JSON."""
        def optimize_arrays(match):
            array_content = match.group(1)
            numbers = [x.strip() for x in array_content.split(',')]
            if len(numbers) > 10 and all(NUMBER_PATTERN.match(n) for n in numbers):
                if len(numbers) > 6:
                    optimized = numbers[:3] + ['...'] + numbers[-3:]
                    return '[' + ', '.join(optimized) + ']'
            return match.group(0)
        json_str = re.sub(r'\[([^\[\]]*)\]', optimize_arrays, json_str)
        return json_str

    def _prepare_visual_event(self, event: Dict) -> Dict:
        """Prepara evento para visualização removendo redundâncias."""
        clean = dict(event)
        clean = self._compress_volume_nodes_for_visual(clean)
        fields_to_remove = ['time_ny', 'time_sp', 'time_utc', '_id', '_rev', '_key']
        for field in fields_to_remove:
            clean.pop(field, None)

        if 'contextual_snapshot' in clean:
            contextual = clean.get('contextual_snapshot', {})
            if isinstance(contextual, dict) and 'flow_metrics' in contextual:
                if 'flow_metrics' in clean:
                    clean['contextual_snapshot'] = {
                        k: v for k, v in contextual.items()
                        if k != 'flow_metrics'
                    }

        def remove_empty(obj):
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    cleaned_v = remove_empty(v)
                    # CORREÇÃO: não filtrar 0/0.0 — valores numéricos zero são legítimos
                    # (ex: sector_flow buy=0 significa ausência de compras, não dado vazio)
                    if cleaned_v is not None and cleaned_v != "" and cleaned_v != []:
                        result[k] = cleaned_v
                return result if result else None
            elif isinstance(obj, list):
                result = [remove_empty(item) for item in obj if item is not None]
                return result if result else None
            return obj

        cleaned_event = remove_empty(clean)
        return cleaned_event if cleaned_event else clean

    def _alert_sound_async(self) -> None:
        """Enfileira beep no worker único (não-bloqueante; descarta se cheia).

        Falha de áudio jamais quebra/atrasa o processamento do sinal.
        """
        try:
            self._sound_queue.put_nowait(True)
        except queue.Full:
            self._sound_dropped += 1
            return
        except Exception:
            return
        try:
            with self._sound_lock:
                if self._sound_thread is None or not self._sound_thread.is_alive():
                    self._sound_stop.clear()
                    self._sound_thread = threading.Thread(
                        target=self._sound_worker,
                        name="eventsaver-sound",
                        daemon=True,
                    )
                    self._sound_thread.start()
        except Exception:
            return

    def _sound_worker(self) -> None:
        """Worker único de áudio: um beep por vez; erro nunca propaga."""
        while not self._sound_stop.is_set():
            try:
                self._sound_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._play_sound()
            except Exception as e:
                self.logger.debug(f"Som indisponível (ignorado): {e}")
            finally:
                try:
                    self._sound_queue.task_done()
                except Exception:
                    pass

    def _play_sound(self):
        """Reproduz alerta sonoro multiplataforma (chamado só pelo worker)."""
        try:
            system = platform.system()
            if system == "Windows":
                try:
                    import winsound
                    winsound.Beep(1000, 500)
                except ImportError:
                    print("\n🔔 ALERTA: Sinal detectado! 🔔\n")
            elif system == "Darwin":
                try:
                    import subprocess
                    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], capture_output=True, timeout=2)
                except Exception:
                    print("\n🔔 ALERTA: Sinal detectado! 🔔\n")
            elif system == "Linux":
                try:
                    import subprocess
                    for cmd in [
                        ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                        ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
                        ["speaker-test", "-t", "sine", "-f", "1000", "-l", "1"]
                    ]:
                        try:
                            subprocess.run(cmd, capture_output=True, timeout=1)
                            break
                        except Exception:
                            continue
                except Exception:
                    print("\n🔔 ALERTA: Sinal detectado! 🔔\n")
            else:
                print("\n🔔 ALERTA: Sinal detectado! 🔔\n")
        except Exception as e:
            self.logger.debug(f"Som não disponível: {e}")
            print("\n🔔 ALERTA: Sinal detectado! 🔔\n")

    # ===== PROPRIEDADES THREAD-SAFE DE CONTABILIDADE (GATE G1) =====
    @property
    def events_received(self) -> int:
        with self._buffer_lock:
            return self._events_received

    @property
    def events_accepted(self) -> int:
        with self._buffer_lock:
            return self._events_accepted

    @property
    def events_rejected_overflow(self) -> int:
        with self._buffer_lock:
            return self._events_rejected_overflow

    @property
    def events_rejected_after_stop(self) -> int:
        with self._buffer_lock:
            return self._events_rejected_after_stop

    @property
    def events_waiting(self) -> int:
        with self._buffer_lock:
            return len(self._write_buffer)

    @property
    def events_in_flight(self) -> int:
        with self._buffer_lock:
            return self._events_in_flight

    @property
    def events_finalized(self) -> int:
        with self._buffer_lock:
            return self._events_finalized

    @property
    def events_failed_terminal(self) -> int:
        with self._buffer_lock:
            return self._events_failed_terminal

    @property
    def shutdown_incomplete(self) -> bool:
        with self._buffer_lock:
            return self._shutdown_incomplete

    @property
    def worker_alive(self) -> bool:
        return self._flush_thread.is_alive() if self._flush_thread is not None else False

    @property
    def worker_exited_cleanly(self) -> bool:
        with self._buffer_lock:
            return self._worker_exited_cleanly

    @property
    def worker_exception(self) -> Optional[str]:
        with self._buffer_lock:
            return self._worker_exception

    @property
    def shutdown_status(self) -> str:
        stop_req = self._stop_event.is_set()
        worker_alive = self.worker_alive
        with self._buffer_lock:
            exited_clean = self._worker_exited_cleanly
            waiting = len(self._write_buffer)
            in_flight = self._events_in_flight
            incomplete = self._shutdown_incomplete

        if not stop_req:
            return "RUNNING" if worker_alive else "WORKER_DIED_UNEXPECTEDLY"
        if worker_alive:
            return "INCOMPLETE_STOP" if incomplete else "STOPPING"
        if not exited_clean:
            return "WORKER_DIED_UNEXPECTEDLY"
        if waiting > 0 or in_flight > 0 or incomplete:
            return "INCOMPLETE_STOP"
        return "CLEAN_STOP"

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de operação."""
        stop_req = self._stop_event.is_set()
        worker_alive = self._flush_thread.is_alive() if self._flush_thread is not None else False
        with self._buffer_lock:
            waiting = len(self._write_buffer)
            in_flight = self._events_in_flight
            exited_clean = self._worker_exited_cleanly
            worker_exc = self._worker_exception
            incomplete = self._shutdown_incomplete

            if not stop_req:
                status = "RUNNING" if worker_alive else "WORKER_DIED_UNEXPECTEDLY"
            elif worker_alive:
                status = "INCOMPLETE_STOP" if incomplete else "STOPPING"
            elif not exited_clean:
                status = "WORKER_DIED_UNEXPECTEDLY"
            elif waiting > 0 or in_flight > 0 or incomplete:
                status = "INCOMPLETE_STOP"
            else:
                status = "CLEAN_STOP"

            stats = {
                "buffer_size": waiting,
                "buffer_max": MAX_BUFFER_SIZE,
                "buffer_overflow_count": self._buffer_overflow_count,
                "lock_timeout_count": self._lock_timeout_count,
                "seen_in_block_size": len(self._seen_in_block),
                "seen_block_ttl_sec": SEEN_BLOCK_TTL,
                "last_cleanup_sec_ago": time.time() - self._last_cleanup_time,
                "cleanup_interval_sec": CLEANUP_INTERVAL,
                "lock_method": LOCK_METHOD or "disabled",
                "events_received": self._events_received,
                "events_accepted": self._events_accepted,
                "events_rejected_overflow": self._events_rejected_overflow,
                "events_rejected_after_stop": self._events_rejected_after_stop,
                "events_waiting": waiting,
                "events_in_flight": in_flight,
                "events_finalized": self._events_finalized,
                "events_failed_terminal": self._events_failed_terminal,
                "shutdown_incomplete": incomplete,
                "worker_alive": worker_alive,
                "worker_exited_cleanly": exited_clean,
                "worker_exception": worker_exc,
                "shutdown_status": status,

                # Gate G2: Contabilidade por destino independente
                "sqlite_events_attempted": self._sqlite_events_attempted,
                "sqlite_events_persisted": self._sqlite_events_persisted,
                "sqlite_events_failed": self._sqlite_events_failed,
                "sqlite_events_unknown": self._sqlite_events_unknown,
                "sqlite_events_rejected_pre_commit": self._sqlite_events_rejected_pre_commit,

                "jsonl_events_attempted": self._jsonl_events_attempted,
                "jsonl_events_written": self._jsonl_events_written,
                "jsonl_events_failed": self._jsonl_events_failed,
                "jsonl_events_unknown": self._jsonl_events_unknown,
                "jsonl_max_file_bytes": self.max_jsonl_file_bytes,
                "jsonl_auto_archive": self.auto_archive_jsonl,
                "jsonl_archive_dir": str(self.archive_dir),

                "fallback_events_attempted": self._fallback_events_attempted,
                "fallback_events_written": self._fallback_events_written,
                "fallback_events_failed": self._fallback_events_failed,
                "fallback_events_unknown": self._fallback_events_unknown,
                "enospc_error_count": self._enospc_errors,
            }

        if HAS_CLOCK_SYNC and _clock_sync_instance:
            try:
                clock_stats = _clock_sync_instance.get_stats()
                stats["clock_sync"] = {
                    "enabled": True,
                    "synced": clock_stats.get('is_synced', False),
                    "offset_ms": clock_stats.get('offset_ms', 0),
                    "offset_seconds": clock_stats.get('offset_seconds', 0),
                }
            except Exception:
                stats["clock_sync"] = {"enabled": False}
        else:
            stats["clock_sync"] = {"enabled": False}

        if self.db:
            stats["sqlite"] = self.db.get_stats()

        return stats


    def get_financial_integrity_metrics(self) -> Dict[str, Any]:
        """Retorna telemetria e métricas consolidadas de integridade financeira (Gate G5)."""
        q_inv = 0
        q_col = 0
        q_enospc = 0
        if self.quarantine is not None and hasattr(self.quarantine, "get_quarantine_metrics"):
            qm = self.quarantine.get_quarantine_metrics()
            q_inv = qm.get("quarantined_invalid_count", 0)
            q_col = qm.get("quarantined_collision_count", 0)
            q_enospc = qm.get("enospc_error_count", 0)
        return {
            "fallback_events_attempted": self._fallback_events_attempted,
            "fallback_events_written": self._fallback_events_written,
            "fallback_events_failed": self._fallback_events_failed,
            "enospc_error_count": self._enospc_errors + q_enospc,
            "quarantined_invalid_count": q_inv,
            "quarantined_collision_count": q_col,
        }


# ===== INSTÂNCIA GLOBAL =====
_global_saver = None


def get_event_saver() -> EventSaver:
    """Retorna instância global do EventSaver."""
    global _global_saver
    if _global_saver is None:
        _global_saver = EventSaver()
    return _global_saver


# ===== HELPER PARA DEPENDÊNCIAS =====
def install_dependencies():
    """Helper para instalar dependências opcionais."""
    import subprocess
    import sys
    packages = []
    try:
        import pytz
    except ImportError:
        packages.append("pytz")
    try:
        import numpy
    except ImportError:
        packages.append("numpy")
    if packages:
        print(f"Instalando dependências opcionais: {', '.join(packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✅ Dependências instaladas!")
    else:
        print("✅ Todas as dependências já instaladas!")


# ===== TESTE =====
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*80)
    print("🧪 TESTE DO EVENTSAVER v5.0.0 - SQLITE ONLY")
    print("="*80 + "\n")

    saver = EventSaver(sound_alert=False)

    now_ms = int(time.time() * 1000)
    test_event = {
        "tipo_evento": "TesteSQLiteOnly",
        "is_signal": True,
        "epoch_ms": now_ms,
        "janela_numero": 500,
        "volume_total": 100.0,
        "msg": "Este evento deve ir APENAS para SQLite e Log Visual"
    }
    saver.save_event(test_event)

    print("1. Evento enviado. Aguardando flush...")
    time.sleep(6)

    stats = saver.get_stats()
    print("\n2. Estatísticas:", stats)

    saver.stop()
    print("\n✅ Teste concluído. Verifique 'dados/trading_bot.db' e 'dados/eventos_visuais.log'.")
