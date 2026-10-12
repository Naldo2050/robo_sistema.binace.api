# -*- coding: utf-8 -*-
"""
Módulo de Exclusividade Cooperativa do Runtime Operacional (Gate G4-D B2 Hardened).

Fornece garantia de processo único para o runtime do robô e o reconciliador
de eventos fallback, prevenindo colisões e escritas concorrentes em
dados/trading_bot.db e dados/eventos_fluxo.jsonl.

Hardening do Ciclo de Vida:
- Âncora global de módulo anti-GC para retenção durante shutdown incompleto.
- Proteção de descritor contra fechamento acidental por finalizadores.
- Retenção explícita de trava quando persistirem writers operacionais.
"""

import errno
import json
import logging
import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

logger = logging.getLogger("common.runtime_lock")

# Âncora global de módulo contra desalocação prematura pelo Garbage Collector (GC)
# durante encerramento com threads ativas (Gate G4-D B2/B3-A).
_RETAINED_LOCKS: List["OperationalRuntimeLock"] = []
_RETAINED_FILE_HANDLES: List[Any] = []


class RuntimeState(str, Enum):
    """Estados do ciclo de vida persistidos no lock operacional (Gate G4-D B3-A)."""
    RUNNING = "RUNNING"
    CLEAN = "CLEAN"
    DIRTY = "DIRTY"
    UNKNOWN = "UNKNOWN"


class RuntimeLockError(RuntimeError):
    """Exceção levantada quando a aquisição ou operação no lock falha (fail-closed)."""
    pass


class OperationalRuntimeLock:
    """
    Gerenciador de lock cooperativo multiplataforma baseado em descritores de arquivo do kernel
    com sinalização de estado persistente do ciclo de vida (Gate G4-D B3-A).

    Características de Segurança:
    1. Multiplataforma nativa: fcntl.flock no Linux, msvcrt.locking no Windows.
    2. Exclusividade autoritativa via kernel: o lock físico do SO é a única prova de exclusividade real.
    3. Sinalização de estado persistente: registra RUNNING, CLEAN, DIRTY, UNKNOWN com flush e fsync.
    4. Fail-closed: falhas de escrita ou corrupção de estado abortam imediatamente a operação.
    5. Preservação de arquivo e inodo: nunca remove o arquivo de lock nem usa os.replace para manter
       a identidade exata do descritor aberto no SO.
    6. Detecção de crash: se o estado no disco for RUNNING mas o kernel não tiver trava física ativa,
       classifica rigorosamente como UNKNOWN (crash/falha abrupta anterior).
    7. Retenção defensiva anti-GC: retém o lock no SO e persiste estado DIRTY se houver writers vivos.
    """

    def __init__(
        self,
        lock_path: Union[Path, str] = Path("dados/.operational_runtime.lock"),
        holder_name: str = "operational_runtime",
    ):
        self.lock_path = Path(lock_path).expanduser().resolve()
        self.holder_name = holder_name
        self._file_obj = None
        self._is_locked = False
        self._is_retained = False
        self.previous_persisted_state: Optional[Union[RuntimeState, str]] = None
        self._thread_lock = threading.Lock()
        self._system = platform.system()

    @property
    def is_locked(self) -> bool:
        """Indica se o processo atual detém ativamente o lock."""
        with self._thread_lock:
            return self._is_locked

    @property
    def is_retained(self) -> bool:
        """Indica se o lock foi explicitamente retido devido a shutdown incompleto."""
        with self._thread_lock:
            return self._is_retained

    @property
    def persisted_state(self) -> RuntimeState:
        """Retorna o estado atual persistido no arquivo de lock."""
        res = self.read_persisted_state(self.lock_path)
        return res["status"]

    def _persist_state(self, state: Union[RuntimeState, str], reason: str = "") -> None:
        """
        Grava atômica e confiavelmente o estado e metadados no arquivo sob trava exclusiva.

        Aplica flush() e os.fsync().
        Falha de escrita (ex: ENOSPC, erro de disco) levanta RuntimeLockError (fail-closed).
        """
        if not self._is_locked or self._file_obj is None:
            raise RuntimeLockError(
                "Operação proibida: tentativa de persistir estado operacional sem deter o lock exclusivo."
            )

        state_val = state.value if isinstance(state, RuntimeState) else str(state)
        metadata = {
            "status": state_val,
            "holder": self.holder_name,
            "pid": os.getpid(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
        }
        if reason:
            metadata["reason"] = str(reason)

        f = self._file_obj
        try:
            raw_bytes = json.dumps(metadata, indent=2).encode("utf-8")
            # Prefixo b'#' no byte 0: no Windows, byte 0 é travado pelo msvcrt,
            # permitindo que outros processos leiam os metadados a partir do byte 1
            payload = b"#" + raw_bytes
            f.seek(0)
            f.write(payload)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
        except (OSError, IOError) as write_err:
            logger.critical(
                "Falha crítica de E/S ao persistir estado %s no lock %s: %s",
                state_val,
                self.lock_path,
                write_err,
            )
            # Invalida o payload de dados preservando o byte 0 (lock do SO) para que
            # leitores não interpretem um estado não garantido por fsync como válido
            try:
                f.seek(0)
                f.write(b"#")
                f.truncate(1)
                f.flush()
            except Exception:
                pass
            raise RuntimeLockError(
                f"Falha ao persistir estado '{state_val}' no arquivo de lock: {write_err}"
            ) from write_err

    def update_state(self, state: Union[RuntimeState, str], reason: str = "") -> None:
        """
        Permite ao detentor do lock atualizar explicitamente o estado durante o ciclo de vida.
        Requer exclusividade ativa (falha se não detiver o lock).
        """
        with self._thread_lock:
            self._persist_state(state, reason=reason)

    def acquire(
        self,
        timeout: float = 0.0,
        require_previous_state: Optional[Union[RuntimeState, str]] = None,
    ) -> None:
        """
        Adquire o lock exclusivo no arquivo e registra estado RUNNING com fsync.

        Args:
            timeout: Tempo máximo em segundos para tentar adquirir. Se 0.0, é puramente não-bloqueante.
            require_previous_state: Estado anterior exigido no arquivo sob trava física do SO
                (ex: RuntimeState.CLEAN). Se especificado e o estado prévio em disco diferir,
                aborta imediatamente sem modificar o arquivo de lock e libera o descritor no SO
                (sem criar janela TOCTOU e preservando evidências de crash/inconsistência intactas).

        Raises:
            RuntimeLockError: Se o lock já estiver detido por outro processo, se o estado
                anterior diferir do exigido, ou se falhar ao persistir RUNNING.
        """
        with self._thread_lock:
            if self._is_locked:
                return

            # Garante que o diretório pai existe
            try:
                self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RuntimeLockError(
                    f"Falha ao criar diretório para lock em {self.lock_path}: {e}"
                ) from e

            # Garante que o arquivo existe antes de abrir em r+b
            if not self.lock_path.exists():
                try:
                    open(self.lock_path, "a+b").close()
                except Exception as e:
                    raise RuntimeLockError(
                        f"Não foi possível criar arquivo de lock em {self.lock_path}: {e}"
                    ) from e

            # Abre em modo r+b para permitir seek(0) e truncate() confiáveis sem append forçado
            try:
                f = open(self.lock_path, "r+b")
            except Exception as e:
                raise RuntimeLockError(
                    f"Não foi possível abrir o arquivo de lock em {self.lock_path}: {e}"
                ) from e

            start_time = time.monotonic()
            acquired = False
            last_err = None

            while True:
                try:
                    if self._system == "Windows":
                        import msvcrt
                        # Posiciona no início para travar o byte 0
                        f.seek(0)
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                        acquired = True
                        break
                    else:
                        import fcntl
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                except (BlockingIOError, OSError) as lock_exc:
                    last_err = lock_exc
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        break
                    time.sleep(min(0.05, timeout - elapsed))

            if not acquired:
                holder_info = self._read_existing_metadata_safe(f)
                try:
                    f.close()
                except Exception:
                    pass

                msg = (
                    f"Conflito de exclusividade: não foi possível adquirir o lock em {self.lock_path}. "
                    f"Recurso operacional já em uso por outra instância."
                )
                if holder_info:
                    msg += f" Detentores detectados: {holder_info}"
                if last_err:
                    msg += f" (Erro do SO: {last_err})"

                logger.warning(msg)
                raise RuntimeLockError(msg)

            self._file_obj = f
            self._is_locked = True

            # Gate G4-D B3-B1: Validação atômica de estado anterior sob trava física do SO (sem TOCTOU)
            if require_previous_state is not None:
                existing_meta = self._read_existing_metadata_safe(f)
                prev_status = existing_meta.get("status") if existing_meta else None
                expected_val = (
                    require_previous_state.value
                    if isinstance(require_previous_state, RuntimeState)
                    else str(require_previous_state)
                )
                if prev_status != expected_val:
                    # Rollback seguro: libera a trava física no kernel SEM tocar no arquivo em disco
                    try:
                        if self._system == "Windows":
                            import msvcrt
                            f.seek(0)
                            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                    try:
                        f.close()
                    except Exception:
                        pass
                    self._file_obj = None
                    self._is_locked = False

                    reason_msg = (
                        f"Estado anterior '{prev_status}' diverge do exigido '{expected_val}'"
                        if prev_status is not None
                        else f"Estado anterior ausente ou corrompido (exigia estritamente '{expected_val}')"
                    )
                    logger.warning(
                        "Aquisição de lock em %s rejeitada: %s. Estado persistido em disco preservado inalterado.",
                        self.lock_path,
                        reason_msg,
                    )
                    raise RuntimeLockError(
                        f"Transição proibida: {reason_msg} no lock {self.lock_path}."
                    )

                self.previous_persisted_state = (
                    RuntimeState(prev_status) if prev_status in RuntimeState._value2member_map_ else prev_status
                )
            else:
                self.previous_persisted_state = None

            # Registra estado RUNNING de forma atômica e com fsync (fail-closed)
            try:
                self._persist_state(RuntimeState.RUNNING)
            except Exception as persist_err:
                # Rollback imediato se falhar ao persistir estado (ex: ENOSPC)
                try:
                    if self._system == "Windows":
                        import msvcrt
                        f.seek(0)
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    f.close()
                except Exception:
                    pass
                self._file_obj = None
                self._is_locked = False
                if isinstance(persist_err, RuntimeLockError):
                    raise persist_err
                raise RuntimeLockError(
                    f"Falha ao persistir estado RUNNING no lock: {persist_err}"
                ) from persist_err

            logger.info(
                f"Lock exclusivo de runtime adquirido com sucesso: {self.lock_path} "
                f"(holder={self.holder_name}, status=RUNNING)"
            )

    def retain(self, reason: str = "") -> None:
        """
        Retém explicitamente o lock no kernel do SO durante shutdown incompleto e grava estado DIRTY.

        Registra o objeto e seu descritor de arquivo em âncoras globais de módulo,
        impedindo a coleta de lixo pelo Garbage Collector e marcando o lock
        para que nem __del__ nem chamadas inadvertidas fechem o handle no SO.
        """
        with self._thread_lock:
            if not self._is_locked or self._file_obj is None:
                return

            self._is_retained = True
            if self not in _RETAINED_LOCKS:
                _RETAINED_LOCKS.append(self)
            if self._file_obj not in _RETAINED_FILE_HANDLES:
                _RETAINED_FILE_HANDLES.append(self._file_obj)

            # Persiste estado DIRTY no arquivo de lock
            try:
                self._persist_state(RuntimeState.DIRTY, reason=reason or "writers operacionais ativos")
            except Exception as e_dirty:
                logger.critical("Falha ao gravar estado DIRTY no lock: %s", e_dirty)

            logger.critical(
                "Lock operacional RETIDO no SO por motivo de segurança (shutdown incompleto/DIRTY): "
                "%s (motivo: %s)",
                self.lock_path,
                reason or "writers operacionais ativos",
            )

    def release(self) -> None:
        """
        Libera explicitamente o lock, gravando estado CLEAN antes de destravar o descritor de arquivo.
        Operação estritamente idempotente e segura contra chamadas repetidas.
        O arquivo no disco é preservado para evitar condições de corrida de handle/inodo.
        Recusa liberação caso o lock tenha sido marcado como retido (DIRTY).
        """
        with self._thread_lock:
            if getattr(self, "_is_retained", False):
                logger.warning(
                    "Tentativa de release ignorada: lock %s está explicitamente "
                    "retido devido a shutdown incompleto com writers operacionais (DIRTY).",
                    self.lock_path,
                )
                return

            if not self._is_locked or self._file_obj is None:
                return

            f = self._file_obj

            # Persiste estado CLEAN antes de destravar no SO
            try:
                self._persist_state(RuntimeState.CLEAN)
            except Exception as clean_err:
                logger.error("Erro ao persistir estado CLEAN durante release: %s", clean_err)

            try:
                if self._system == "Windows":
                    import msvcrt
                    f.seek(0)
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError as e:
                        logger.debug(f"Aviso ao destravar msvcrt: {e}")
                else:
                    import fcntl
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except OSError as e:
                        logger.debug(f"Aviso ao destravar fcntl: {e}")
            finally:
                try:
                    f.close()
                except Exception as e:
                    logger.debug(f"Erro ao fechar descritor do lock: {e}")
                self._file_obj = None
                self._is_locked = False

            logger.info(f"Lock exclusivo de runtime liberado com sucesso: {self.lock_path} (status=CLEAN)")

    @classmethod
    def read_persisted_state(cls, lock_path: Union[Path, str]) -> Dict[str, Any]:
        """
        Inspeciona o estado persistido do lock de forma defensiva e não-destrutiva.

        Classificações:
        - CLEAN: Arquivo gravado com CLEAN e destravado.
        - DIRTY: Arquivo gravado com DIRTY (shutdown incompleto).
        - RUNNING: Arquivo gravado com RUNNING E trava física ativamente detida no kernel.
        - UNKNOWN: Arquivo ausente, vazio, JSON truncado/inválido ou status RUNNING sem trava física (crash anterior).
        """
        path = Path(lock_path).expanduser().resolve()
        if not path.exists() or path.stat().st_size == 0:
            return {
                "status": RuntimeState.UNKNOWN,
                "raw_status": None,
                "reason": "missing_or_empty_lock_file",
                "lock_path": str(path),
            }

        try:
            with open(path, "rb") as f:
                f.seek(1)
                content = f.read()
                try:
                    data = json.loads(content.decode("utf-8", errors="strict"))
                except Exception:
                    # Fallback para formato legado sem prefixo de byte 0
                    f.seek(0)
                    content = f.read()
                    data = json.loads(content.decode("utf-8", errors="strict"))
            if not isinstance(data, dict):
                raise ValueError("Payload do lock não é um objeto JSON válido")
        except Exception as json_err:
            return {
                "status": RuntimeState.UNKNOWN,
                "raw_status": None,
                "reason": f"invalid_or_truncated_json: {json_err}",
                "lock_path": str(path),
            }

        raw_status = data.get("status")
        if raw_status == RuntimeState.CLEAN.value:
            return {
                "status": RuntimeState.CLEAN,
                "raw_status": raw_status,
                "metadata": data,
                "lock_path": str(path),
            }
        elif raw_status == RuntimeState.DIRTY.value:
            return {
                "status": RuntimeState.DIRTY,
                "raw_status": raw_status,
                "reason": data.get("reason", "dirty_shutdown"),
                "metadata": data,
                "lock_path": str(path),
            }
        elif raw_status == RuntimeState.RUNNING.value:
            # Verifica se a trava física do SO está realmente ativa
            system = platform.system()
            is_physically_locked = False
            try:
                test_f = open(path, "r+b")
                try:
                    if system == "Windows":
                        import msvcrt
                        test_f.seek(0)
                        try:
                            msvcrt.locking(test_f.fileno(), msvcrt.LK_NBLCK, 1)
                            # Conseguiu a trava -> significa que estava destravado (crash anterior!)
                            msvcrt.locking(test_f.fileno(), msvcrt.LK_UNLCK, 1)
                            is_physically_locked = False
                        except OSError:
                            is_physically_locked = True
                    else:
                        import fcntl
                        try:
                            fcntl.flock(test_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            fcntl.flock(test_f.fileno(), fcntl.LOCK_UN)
                            is_physically_locked = False
                        except OSError:
                            is_physically_locked = True
                finally:
                    test_f.close()
            except Exception:
                is_physically_locked = True

            if is_physically_locked:
                return {
                    "status": RuntimeState.RUNNING,
                    "raw_status": raw_status,
                    "metadata": data,
                    "lock_path": str(path),
                }
            else:
                # O status dizia RUNNING, mas ninguém detém a trava no kernel => Crash detectado!
                return {
                    "status": RuntimeState.UNKNOWN,
                    "raw_status": raw_status,
                    "reason": "crash_detected_running_status_without_kernel_lock",
                    "metadata": data,
                    "lock_path": str(path),
                }
        else:
            return {
                "status": RuntimeState.UNKNOWN,
                "raw_status": raw_status,
                "reason": f"unrecognized_status: {raw_status}",
                "metadata": data,
                "lock_path": str(path),
            }

    def _read_existing_metadata_safe(self, f) -> Optional[Dict[str, Any]]:
        """
        Lê metadados existentes no arquivo aberto para enriquecer mensagens de erro.
        Inicia a leitura a partir do byte 1 para evitar conflito com a trava do byte 0 no Windows.
        Mantém fallback para o byte 0 para compatibilidade com formato legado.
        Não efetua escrita e trata exceções defensivamente.
        """
        try:
            # Tenta a partir do byte 1 (padrão G4-D B3-A: byte 0 reservado para o lock do SO)
            f.seek(1)
            content = f.read(4096)
            if content:
                try:
                    data = json.loads(content.decode("utf-8", errors="strict"))
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass

            # Fallback para offset 0 (formato legado sem byte 0 reservado)
            f.seek(0)
            content = f.read(4096)
            if content:
                data = json.loads(content.decode("utf-8", errors="strict"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def __del__(self):
        """
        Finalizador defensivo.
        NUNCA libera o lock se estiver explicitamente retido (_is_retained == True).

        Documentação de Limitação:
        Durante o encerramento do interpretador Python, o GC e o shutdown do runtime
        podem desalocar objetos locais. As âncoras globais _RETAINED_LOCKS e
        _RETAINED_FILE_HANDLES garantem que o descritor físico permaneça aberto
        enquanto houver threads não-daemon ativas. A liberação definitiva do kernel
        ocorre de forma atômica quando o processo do SO termina.
        """
        try:
            if getattr(self, "_is_retained", False):
                return
            self.release()
        except Exception:
            pass
