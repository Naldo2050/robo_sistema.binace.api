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
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
import uuid

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
        self._thread_lock = threading.RLock()
        self._system = platform.system()
        self._active_maintenance_token: Optional[Any] = None
        self._is_maintenance_session: bool = False

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
                if require_previous_state is not None:
                    expected_val = (
                        require_previous_state.value
                        if isinstance(require_previous_state, RuntimeState)
                        else str(require_previous_state)
                    )
                    raise RuntimeLockError(
                        f"Transição proibida: Arquivo de lock não existe em {self.lock_path} (exigia estritamente estado '{expected_val}')."
                    )
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
                self._is_maintenance_session = False
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
                self._is_maintenance_session = False
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
                self._is_maintenance_session = False

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

    def acquire_maintenance(
        self,
        allowed_states: Optional[List[Union[RuntimeState, str]]] = None,
        timeout: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Adquire lock físico exclusivo no kernel para manutenção/recovery sem transicionar para RUNNING (Gate G4-D B3-B2-B3).

        Garantias:
        1. Exclusividade autoritativa via kernel (msvcrt.locking LK_NBLCK no Windows / fcntl.flock LOCK_EX no Unix).
        2. Não altera o arquivo de lock e NÃO grava RUNNING.
        3. Preserva inodo e arquivo físico (modo r+b sem recriação).
        4. Lê o estado anterior sob exclusividade física (elimina TOCTOU).
        5. Se allowed_states for informado e o estado em disco não for elegível, destrava e levanta RuntimeLockError.
        6. Retorna o dicionário de metadados do estado lido sob exclusividade.
        """
        with self._thread_lock:
            if self._is_locked:
                raise RuntimeLockError("Lock já adquirido por esta instância.")

            if not self.lock_path.exists():
                raise RuntimeLockError(
                    f"Manutenção bloqueada: Arquivo de lock não existe em {self.lock_path}."
                )

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
                    f"Conflito de exclusividade: recurso {self.lock_path} retido por processo ativo. "
                    "Não é possível executar manutenção concorrente."
                )
                if holder_info:
                    msg += f" Detentores detectados: {holder_info}"
                if last_err:
                    msg += f" (Erro do SO: {last_err})"
                logger.warning(msg)
                raise RuntimeLockError(msg)

            self._file_obj = f
            self._is_locked = True
            self._is_maintenance_session = True

            # Lê estado existente sob trava física exclusiva
            existing_meta = self._read_existing_metadata_safe(f) or {}
            raw_status = existing_meta.get("status")

            # Se status dizia RUNNING, mas nós acabamos de travar com sucesso no kernel,
            # então é comprovadamente RUNNING residual (crash de processo anterior)
            detected_status = raw_status
            if raw_status == RuntimeState.RUNNING.value:
                detected_status = "RUNNING_RESIDUAL"
            elif raw_status is None:
                detected_status = RuntimeState.UNKNOWN.value

            if allowed_states is not None:
                allowed_vals = [
                    s.value if isinstance(s, RuntimeState) else str(s)
                    for s in allowed_states
                ]
                if raw_status not in allowed_vals and detected_status not in allowed_vals:
                    # Rollback: destrava e fecha sem alterar nada
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
                    self._is_maintenance_session = False
                    raise RuntimeLockError(
                        f"Estado operacional '{raw_status}' não é elegível para recuperação. "
                        f"Estados permitidos: {allowed_vals}."
                    )

            self.previous_persisted_state = raw_status
            return {
                "raw_status": raw_status,
                "detected_status": detected_status,
                "metadata": existing_meta,
            }

    def check_metadata_integrity_under_lock(self) -> Tuple[bool, Optional[str]]:
        """
        Valida de forma estrita e fail-closed a integridade estrutural dos metadados
        do lock enquanto o descritor de arquivo físico permanece detido sob a trava do kernel (Gate C2-B2.2-H2).

        Distinção essencial:
        - CASO A (Válido): Metadados estruturalmente válidos (JSON parseable, dicionário, contendo chave 'status'
          do tipo string não-vazia com estado operacional conhecido).
        - CASO B (Inválido): Arquivo ausente, vazio, truncado, lixo binário, JSON malformado, não-dicionário,
          chave 'status' ausente, 'status' não-string ou vazio.
        """
        with self._thread_lock:
            if not self._is_locked or self._file_obj is None or getattr(self._file_obj, "closed", True):
                return False, "lock_not_held_or_descriptor_closed"

            f = self._file_obj
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size == 0:
                    return False, "empty_lock_file"

                # Leitura a partir do byte 1 (padrão com byte 0 reservado)
                f.seek(1)
                content = f.read()
                data = None
                json_err_1 = None
                if content:
                    try:
                        data = json.loads(content.decode("utf-8", errors="strict"))
                    except Exception as e:
                        json_err_1 = str(e)

                # Fallback para offset 0 (legado)
                if data is None:
                    f.seek(0)
                    content_0 = f.read()
                    if content_0:
                        try:
                            data = json.loads(content_0.decode("utf-8", errors="strict"))
                        except Exception as e:
                            return False, f"corrupt_or_truncated_json: {json_err_1 or str(e)}"
                    else:
                        return False, "empty_lock_content"

                if not isinstance(data, dict):
                    return False, f"payload_not_dict: expected dict, got {type(data).__name__}"

                if "status" not in data:
                    return False, "missing_status_key"

                raw_st = data["status"]
                if not isinstance(raw_st, str) or not raw_st.strip():
                    return False, f"invalid_status_type_or_empty: {type(raw_st).__name__}"

                valid_statuses = {
                    RuntimeState.CLEAN.value,
                    RuntimeState.DIRTY.value,
                    RuntimeState.RUNNING.value,
                    RuntimeState.UNKNOWN.value,
                }
                if raw_st.strip() not in valid_statuses:
                    return False, f"unrecognized_status_value: '{raw_st}'"

                return True, None

            except Exception as e:
                return False, f"io_error_reading_metadata: {e}"

    def transition_maintenance_to_clean(self, reason: str = "") -> None:
        """
        Transiciona o lock detido em sessão de manutenção para CLEAN utilizando o mesmo descritor
        já travado, aplicando flush e os.fsync.
        """
        with self._thread_lock:
            if not self._is_locked or self._file_obj is None:
                raise RuntimeLockError(
                    "Operação proibida: tentativa de transicionar para CLEAN sem deter o lock de manutenção."
                )
            self._persist_state(RuntimeState.CLEAN, reason=reason)

    @property
    def active_maintenance_token(self) -> Optional["MaintenanceAuthorizationToken"]:
        """Retorna a capacidade de autorização de manutenção ativa associada a esta instância."""
        with self._thread_lock:
            return self._active_maintenance_token

    def create_maintenance_authorization(self, environment_uuid: str) -> "MaintenanceAuthorizationToken":
        """
        Emite uma capacidade formal de autorização para mutações controladas de manutenção (Gate P0).
        Exige lock físico adquirido no kernel, descritor aberto e sessão exclusiva de manutenção.
        """
        with self._thread_lock:
            if not self._is_locked or self._file_obj is None or getattr(self._file_obj, "closed", True):
                raise RuntimeLockError(
                    "Não é possível emitir autorização de manutenção: lock não adquirido ou descritor fechado."
                )
            if not getattr(self, "_is_maintenance_session", False):
                raise RuntimeLockError(
                    "Não é possível emitir autorização de manutenção: lock não foi adquirido em sessão de manutenção exclusiva (acquire_maintenance)."
                )
            token = MaintenanceAuthorizationToken(
                lock_instance=self,
                lock_file_obj=self._file_obj,
                lock_path=self.lock_path,
                environment_uuid=environment_uuid,
            )
            self._active_maintenance_token = token
            return token

    def release_maintenance(self) -> None:
        """
        Libera o lock de manutenção destravando o descritor no SO e fechando o handle.
        Preserva o estado atual do arquivo sem sobrescrever com CLEAN caso não tenha sido transicionado.
        Revoga expressamente qualquer token de autorização de manutenção ativo e encerra a sessão de manutenção.
        """
        with self._thread_lock:
            self._is_maintenance_session = False
            if self._active_maintenance_token is not None:
                self._active_maintenance_token.revoke()
                self._active_maintenance_token = None
            if not self._is_locked or self._file_obj is None or getattr(self._file_obj, "closed", False):
                self._file_obj = None
                self._is_locked = False
                return
            f = self._file_obj
            try:
                if self._system == "Windows":
                    import msvcrt
                    f.seek(0)
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError as e:
                        logger.debug(f"Aviso ao destravar msvcrt em manutenção: {e}")
                else:
                    import fcntl
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except OSError as e:
                        logger.debug(f"Aviso ao destravar fcntl em manutenção: {e}")
            finally:
                try:
                    f.close()
                except Exception as e:
                    logger.debug(f"Erro ao fechar descritor de manutenção: {e}")
                self._file_obj = None
                self._is_locked = False

    @contextmanager
    def maintenance_session(
        self,
        allowed_states: Optional[List[Union[RuntimeState, str]]] = None,
        timeout: float = 0.0,
        environment_uuid: Optional[str] = None,
    ):
        """Context manager que garante a liberação do lock de manutenção em bloco finally."""
        info = self.acquire_maintenance(allowed_states=allowed_states, timeout=timeout)
        token = None
        if environment_uuid:
            token = self.create_maintenance_authorization(environment_uuid=environment_uuid)
            info["authorization"] = token
        try:
            yield info
        finally:
            if token is not None:
                token.revoke()
            self.release_maintenance()


class MaintenanceAuthorizationToken:
    """
    Capacidade de autorização de sessão de manutenção exclusiva (Gate G4-D B3-B2-C2-B2-P0).
    Vinculada à instância ativa de OperationalRuntimeLock, comprovando posse do lock físico
    no kernel para permitir mutações controladas no manifesto operacional.
    """
    def __init__(
        self,
        lock_instance: "OperationalRuntimeLock",
        lock_file_obj: Any,
        lock_path: Path,
        environment_uuid: str,
    ):
        if lock_instance is None or not getattr(lock_instance, "_is_locked", False):
            raise RuntimeLockError("Não é possível emitir MaintenanceAuthorizationToken: lock_instance não está ativo.")
        if lock_file_obj is None or getattr(lock_file_obj, "closed", True):
            raise RuntimeLockError("Não é possível emitir MaintenanceAuthorizationToken: descritor físico fechado.")

        clean_uuid = str(environment_uuid).strip() if environment_uuid else ""
        try:
            uuid.UUID(clean_uuid)
        except Exception as exc:
            raise RuntimeLockError(f"environment_uuid inválido para MaintenanceAuthorizationToken: {exc}") from exc

        self.token_id = str(uuid.uuid4())
        self.lock_instance = lock_instance
        self.lock_file_obj = lock_file_obj
        self.lock_path = Path(lock_path).expanduser().resolve()
        self.environment_uuid = clean_uuid
        self.created_at_utc = datetime.now(timezone.utc).isoformat()
        self._is_active = True

    @property
    def is_active(self) -> bool:
        if not self._is_active:
            return False
        if self.lock_instance is None or not getattr(self.lock_instance, "_is_locked", False):
            return False
        if self.lock_file_obj is None or getattr(self.lock_file_obj, "closed", True):
            return False
        return True

    def revoke(self) -> None:
        """Revoga expressamente a capacidade ao encerrar a sessão de manutenção."""
        self._is_active = False

    def __repr__(self) -> str:
        return (
            f"<MaintenanceAuthorizationToken id={self.token_id} "
            f"env={self.environment_uuid} active={self.is_active} path={self.lock_path}>"
        )


MANIFEST_FILENAME = ".runtime_initialized.manifest"


def write_runtime_manifest(
    manifest_path: Union[Path, str],
    env_uuid: Optional[str] = None,
    holder_name: str = "bootstrap_runtime_lock",
    manifest_version: str = "1.1",
) -> Dict[str, Any]:
    """
    Grava o manifesto permanente de inicialização do ambiente (Gate G4-D B3-B2-A / Gate G4-D B3-B2-B3-H1 / Gate G4-D B3-B2-C2-B1).
    Inicializa em manifest_version="1.1" por padrão ou "1.2" quando solicitado.
    Aplica flush() e os.fsync().
    Não armazena segredos, tokens ou credenciais.
    """
    p = Path(manifest_path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    uid = str(env_uuid or uuid.uuid4())
    payload = {
        "manifest_version": manifest_version,
        "environment_uuid": uid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "holder": holder_name,
        "data_dir_name": p.parent.name,
        "recovery_history_counter": 0,
        "last_recovery_audit_id": None,
        "last_recovery_utc": None,
    }
    if manifest_version == "1.2":
        payload["reconciliation_history_counter"] = 0
        payload["last_reconciliation_audit_id"] = None
        payload["last_reconciliation_utc"] = None

    raw = json.dumps(payload, indent=2).encode("utf-8")
    with open(p, "wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())

    return payload


def validate_manifest_structure(manifest_data: Any) -> Tuple[bool, Optional[str]]:
    """
    Valida estritamente a conformidade estrutural e tipagem do manifesto operacional (Gate G4-D B3-B2-B3-H1 / Gate G4-D B3-B2-C2-B1).
    Suporta versões '1.0', '1.1' e '1.2'.
    Rejeita inteiros booleanos, strings numéricas, valores negativos e formatos corrompidos.
    """
    if not isinstance(manifest_data, dict):
        return False, "Payload do manifesto não é um objeto JSON válido (dict)."

    for bf in ["manifest_version", "environment_uuid", "created_at_utc"]:
        if bf not in manifest_data or not str(manifest_data[bf]).strip():
            return False, f"Manifesto ausente de campo obrigatório básico: '{bf}'."

    # Validação do UUID do ambiente
    try:
        uuid.UUID(str(manifest_data["environment_uuid"]).strip())
    except Exception as exc:
        return False, f"Campo 'environment_uuid' não é um UUID válido: {exc}"

    # Validação do timestamp de criação
    try:
        datetime.fromisoformat(str(manifest_data["created_at_utc"]).strip().replace("Z", "+00:00"))
    except Exception as exc:
        return False, f"Campo 'created_at_utc' não é um timestamp ISO8601 válido: {exc}"

    version = str(manifest_data["manifest_version"]).strip()
    if version == "1.0":
        return True, None

    if version in ("1.1", "1.2"):
        if "recovery_history_counter" not in manifest_data:
            return False, f"Manifesto {version} ausente de campo obrigatório 'recovery_history_counter'."

        counter = manifest_data["recovery_history_counter"]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            return False, f"Campo 'recovery_history_counter' inválido (deve ser inteiro >= 0, rejeitando booleanos e strings): {repr(counter)}"

        if counter == 0:
            last_id = manifest_data.get("last_recovery_audit_id")
            if last_id is not None and str(last_id).strip() != "":
                return False, f"Campo 'last_recovery_audit_id' deve ser None quando contador é 0 (recebido: {last_id})."
            last_utc = manifest_data.get("last_recovery_utc")
            if last_utc is not None and str(last_utc).strip() != "":
                return False, f"Campo 'last_recovery_utc' deve ser None quando contador é 0 (recebido: {last_utc})."
        else:
            last_id = manifest_data.get("last_recovery_audit_id")
            if not last_id or not str(last_id).strip():
                return False, "Campo 'last_recovery_audit_id' é obrigatório quando recovery_history_counter > 0."
            try:
                uuid.UUID(str(last_id).strip())
            except Exception as exc:
                return False, f"Campo 'last_recovery_audit_id' não é um UUID válido: {exc}"

            last_utc = manifest_data.get("last_recovery_utc")
            if not last_utc or not str(last_utc).strip():
                return False, "Campo 'last_recovery_utc' é obrigatório quando recovery_history_counter > 0."
            try:
                datetime.fromisoformat(str(last_utc).strip().replace("Z", "+00:00"))
            except Exception as exc:
                return False, f"Campo 'last_recovery_utc' não é um timestamp ISO8601 válido: {exc}"

        if version == "1.2":
            if "reconciliation_history_counter" not in manifest_data:
                return False, "Manifesto 1.2 ausente de campo obrigatório 'reconciliation_history_counter'."

            rec_counter = manifest_data["reconciliation_history_counter"]
            if isinstance(rec_counter, bool) or not isinstance(rec_counter, int) or rec_counter < 0:
                return False, f"Campo 'reconciliation_history_counter' inválido (deve ser inteiro >= 0, rejeitando booleanos e strings): {repr(rec_counter)}"

            if rec_counter == 0:
                last_rec_id = manifest_data.get("last_reconciliation_audit_id")
                if last_rec_id is not None and str(last_rec_id).strip() != "":
                    return False, f"Campo 'last_reconciliation_audit_id' deve ser None quando contador é 0 (recebido: {last_rec_id})."
                last_rec_utc = manifest_data.get("last_reconciliation_utc")
                if last_rec_utc is not None and str(last_rec_utc).strip() != "":
                    return False, f"Campo 'last_reconciliation_utc' deve ser None quando contador é 0 (recebido: {last_rec_utc})."
            else:
                last_rec_id = manifest_data.get("last_reconciliation_audit_id")
                if not last_rec_id or not str(last_rec_id).strip():
                    return False, "Campo 'last_reconciliation_audit_id' é obrigatório quando reconciliation_history_counter > 0."
                try:
                    uuid.UUID(str(last_rec_id).strip())
                except Exception as exc:
                    return False, f"Campo 'last_reconciliation_audit_id' não é um UUID válido: {exc}"

                last_rec_utc = manifest_data.get("last_reconciliation_utc")
                if not last_rec_utc or not str(last_rec_utc).strip():
                    return False, "Campo 'last_reconciliation_utc' é obrigatório quando reconciliation_history_counter > 0."
                try:
                    dt = datetime.fromisoformat(str(last_rec_utc).strip().replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        return False, "Campo 'last_reconciliation_utc' não é um timestamp ISO8601 válido (deve conter fuso horário UTC 'Z' ou '+00:00')."
                except Exception as exc:
                    return False, f"Campo 'last_reconciliation_utc' não é um timestamp ISO8601 válido: {exc}"

        return True, None

    return False, f"Versão de manifesto desconhecida ou não suportada: '{version}'."


def verify_manifest_recovery_consistency(
    manifest_data: Optional[Dict[str, Any]],
    ledger_path: Union[Path, str],
    expected_env_uuid: Optional[str] = None,
    lock_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Avalia a consistência transacional e integridade entre o manifesto operacional e o livro-razão de auditoria (Gate G4-D B3-B2-B3-H1).
    Compartilhada entre o preflight de startup e o diagnóstico de recovery.
    """
    if manifest_data is None:
        return {
            "classification": "INVALID_MANIFEST",
            "is_valid": False,
            "error": "Manifesto operacional ausente, vazio ou ilegível.",
            "manifest_version": "UNKNOWN",
            "manifest_counter": 0,
            "ledger_completed_count": 0,
            "ledger_has_pending": False,
        }

    is_m_valid, m_err = validate_manifest_structure(manifest_data)
    if not is_m_valid:
        return {
            "classification": "INVALID_MANIFEST",
            "is_valid": False,
            "error": f"Estrutura do manifesto inválida: {m_err}",
            "manifest_version": str(manifest_data.get("manifest_version", "UNKNOWN")),
            "manifest_counter": 0,
            "ledger_completed_count": 0,
            "ledger_has_pending": False,
        }

    env_uuid = str(manifest_data.get("environment_uuid", "")).strip()
    if expected_env_uuid:
        clean_exp = str(expected_env_uuid).strip()
        if env_uuid != clean_exp:
            return {
                "classification": "INVALID_MANIFEST",
                "is_valid": False,
                "error": f"UUID do manifesto divergente do ambiente (manifesto='{env_uuid}', esperado='{clean_exp}').",
                "manifest_version": str(manifest_data.get("manifest_version", "UNKNOWN")),
                "manifest_counter": 0,
                "ledger_completed_count": 0,
                "ledger_has_pending": False,
            }

    version = str(manifest_data.get("manifest_version", "1.0")).strip()
    counter = int(manifest_data.get("recovery_history_counter", 0)) if version in ("1.1", "1.2") else 0

    lp = Path(ledger_path).expanduser().resolve()
    ledger_val = validate_recovery_audit_ledger(lp, expected_environment_uuid=env_uuid)

    if ledger_val["exists"]:
        if not ledger_val["is_valid"]:
            return {
                "classification": "CORRUPTED_LEDGER",
                "is_valid": False,
                "error": f"Livro-razão de auditoria (.recovery_audit.jsonl) corrompido ou inconsistente: {ledger_val['error']}",
                "manifest_version": version,
                "manifest_counter": counter,
                "ledger_completed_count": ledger_val.get("completed_transactions_count", 0),
                "ledger_has_pending": ledger_val.get("has_pending_transaction", False),
            }
        if ledger_val["has_pending_transaction"]:
            orphans_str = ", ".join(ledger_val["orphaned_prepared"])
            return {
                "classification": "PENDING_TRANSACTION",
                "is_valid": False,
                "error": f"Transação de recovery incompleta/interrompida detectada no livro-razão (fase PREPARED sem COMMITTED para audit_id={orphans_str}).",
                "manifest_version": version,
                "manifest_counter": counter,
                "ledger_completed_count": ledger_val.get("completed_transactions_count", 0),
                "ledger_has_pending": True,
            }

    completed_count = ledger_val.get("completed_transactions_count", 0)

    if version in ("1.1", "1.2"):
        if counter > 0:
            if not ledger_val["exists"]:
                return {
                    "classification": "MISSING_LEDGER",
                    "is_valid": False,
                    "error": f"Manifesto registra histórico de {counter} recuperação(ões), mas o livro-razão de auditoria (.recovery_audit.jsonl) está ausente (anti-tamper fail-closed).",
                    "manifest_version": version,
                    "manifest_counter": counter,
                    "ledger_completed_count": 0,
                    "ledger_has_pending": False,
                }
            if completed_count != counter:
                return {
                    "classification": "DIVERGENT_COUNTER",
                    "is_valid": False,
                    "error": f"Contador do manifesto ({counter}) diverge das transações COMMITTED no livro-razão ({completed_count}).",
                    "manifest_version": version,
                    "manifest_counter": counter,
                    "ledger_completed_count": completed_count,
                    "ledger_has_pending": False,
                }
            last_rec = ledger_val.get("last_record") or {}
            last_comm_id = str(last_rec.get("audit_id", "")).strip()
            exp_last_id = str(manifest_data.get("last_recovery_audit_id", "")).strip()
            if last_comm_id != exp_last_id:
                return {
                    "classification": "DIVERGENT_AUDIT_ID",
                    "is_valid": False,
                    "error": f"Último audit_id no manifesto ('{exp_last_id}') diverge do último COMMITTED no livro-razão ('{last_comm_id}').",
                    "manifest_version": version,
                    "manifest_counter": counter,
                    "ledger_completed_count": completed_count,
                    "ledger_has_pending": False,
                }
            return {
                "classification": "CONSISTENT",
                "is_valid": True,
                "error": None,
                "manifest_version": version,
                "manifest_counter": counter,
                "ledger_completed_count": completed_count,
                "ledger_has_pending": False,
            }
        else: # counter == 0
            if ledger_val["exists"] and completed_count > 0:
                return {
                    "classification": "DIVERGENT_COUNTER",
                    "is_valid": False,
                    "error": f"Manifesto {version} registra 0 recuperações mas livro-razão contém {completed_count} transações COMMITTED.",
                    "manifest_version": version,
                    "manifest_counter": 0,
                    "ledger_completed_count": completed_count,
                    "ledger_has_pending": False,
                }
            return {
                "classification": "CONSISTENT",
                "is_valid": True,
                "error": None,
                "manifest_version": version,
                "manifest_counter": 0,
                "ledger_completed_count": completed_count,
                "ledger_has_pending": False,
            }

    elif version == "1.0":
        if ledger_val["exists"]:
            return {
                "classification": "LEGACY_CONSISTENT",
                "is_valid": True,
                "error": None,
                "manifest_version": version,
                "manifest_counter": completed_count,
                "ledger_completed_count": completed_count,
                "ledger_has_pending": False,
            }
        else:
            lock_meta = lock_info.get("metadata", {}) or {} if lock_info else {}
            lock_reason = str(lock_meta.get("reason", "") or (lock_info.get("reason", "") if lock_info else ""))
            lock_holder = str(lock_meta.get("holder", "") if lock_meta else (lock_info.get("holder", "") if lock_info else ""))
            had_recovery = ("Recovery autorizada" in lock_reason) or ("audit_id=" in lock_reason) or (lock_holder == "recovery_tool")
            if had_recovery:
                return {
                    "classification": "INDETERMINATE",
                    "is_valid": False,
                    "error": "Manifesto 1.0 legado com evidência de recovery no lock operacional, mas o livro-razão (.recovery_audit.jsonl) está ausente (histórico indeterminado/anti-tamper).",
                    "manifest_version": version,
                    "manifest_counter": 0,
                    "ledger_completed_count": 0,
                    "ledger_has_pending": False,
                }
            return {
                "classification": "LEGACY_CONSISTENT",
                "is_valid": True,
                "error": None,
                "manifest_version": version,
                "manifest_counter": 0,
                "ledger_completed_count": 0,
                "ledger_has_pending": False,
            }

    return {
        "classification": "INVALID_MANIFEST",
        "is_valid": False,
        "error": f"Versão do manifesto não suportada: '{version}'.",
        "manifest_version": version,
        "manifest_counter": 0,
        "ledger_completed_count": 0,
        "ledger_has_pending": False,
    }


def verify_manifest_reconciliation_consistency(
    manifest_data: Optional[Dict[str, Any]],
    ledger_path: Union[Path, str],
    expected_env_uuid: Optional[str] = None,
    lock_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Avalia a consistência transacional e integridade entre o manifesto operacional e o livro-razão de auditoria de reconciliação (Gate G4-D B3-B2-C2-B1).
    Compartilhada entre o preflight de startup e o diagnóstico de recovery.
    """
    if manifest_data is None:
        return {
            "classification": "INVALID_MANIFEST",
            "is_valid": False,
            "error": "Manifesto operacional ausente, vazio ou ilegível.",
            "manifest_version": "UNKNOWN",
            "manifest_counter": 0,
            "ledger_completed_count": 0,
            "ledger_has_pending": False,
        }

    is_m_valid, m_err = validate_manifest_structure(manifest_data)
    if not is_m_valid:
        return {
            "classification": "INVALID_MANIFEST",
            "is_valid": False,
            "error": f"Estrutura do manifesto inválida: {m_err}",
            "manifest_version": str(manifest_data.get("manifest_version", "UNKNOWN")),
            "manifest_counter": 0,
            "ledger_completed_count": 0,
            "ledger_has_pending": False,
        }

    env_uuid = str(manifest_data.get("environment_uuid", "")).strip()
    if expected_env_uuid:
        clean_exp = str(expected_env_uuid).strip()
        if env_uuid != clean_exp:
            return {
                "classification": "INVALID_MANIFEST",
                "is_valid": False,
                "error": f"UUID do manifesto divergente do ambiente (manifesto='{env_uuid}', esperado='{clean_exp}').",
                "manifest_version": str(manifest_data.get("manifest_version", "UNKNOWN")),
                "manifest_counter": 0,
                "ledger_completed_count": 0,
                "ledger_has_pending": False,
            }

    version = str(manifest_data.get("manifest_version", "1.0")).strip()
    counter = int(manifest_data.get("reconciliation_history_counter", 0)) if version == "1.2" else 0

    from events.fallback_reconciler import validate_reconciliation_audit_ledger
    lp = Path(ledger_path).expanduser().resolve()
    ledger_val = validate_reconciliation_audit_ledger(lp, expected_environment_uuid=env_uuid)

    if ledger_val["exists"]:
        if not ledger_val["is_valid"]:
            if ledger_val.get("has_pending_transaction"):
                orphans_str = ", ".join(ledger_val.get("orphaned_prepared", []))
                return {
                    "classification": "PENDING_TRANSACTION",
                    "is_valid": False,
                    "error": f"Transação de reconciliação incompleta/interrompida detectada no livro-razão de reconciliação (fase PREPARED sem COMMITTED para audit_id={orphans_str}).",
                    "manifest_version": version,
                    "manifest_counter": counter,
                    "ledger_completed_count": ledger_val.get("completed_transactions_count", 0),
                    "ledger_has_pending": True,
                }
            return {
                "classification": "CORRUPTED_LEDGER",
                "is_valid": False,
                "error": f"Livro-razão de reconciliação (.reconciliation_audit.jsonl) corrompido ou inconsistente: {ledger_val['error']}",
                "manifest_version": version,
                "manifest_counter": counter,
                "ledger_completed_count": ledger_val.get("completed_transactions_count", 0),
                "ledger_has_pending": ledger_val.get("has_pending_transaction", False),
            }
        if ledger_val.get("has_pending_transaction"):
            orphans_str = ", ".join(ledger_val.get("orphaned_prepared", []))
            return {
                "classification": "PENDING_TRANSACTION",
                "is_valid": False,
                "error": f"Transação de reconciliação incompleta/interrompida detectada no livro-razão de reconciliação (fase PREPARED sem COMMITTED para audit_id={orphans_str}).",
                "manifest_version": version,
                "manifest_counter": counter,
                "ledger_completed_count": ledger_val.get("completed_transactions_count", 0),
                "ledger_has_pending": True,
            }

    completed_count = ledger_val.get("completed_transactions_count", 0)

    if version == "1.2":
        if counter > 0:
            if not ledger_val["exists"]:
                return {
                    "classification": "MISSING_RECONCILIATION_LEDGER",
                    "is_valid": False,
                    "error": f"Manifesto 1.2 registra histórico de {counter} reconciliação(ões), mas o livro-razão de auditoria (.reconciliation_audit.jsonl) está ausente (anti-tamper fail-closed).",
                    "manifest_version": version,
                    "manifest_counter": counter,
                    "ledger_completed_count": 0,
                    "ledger_has_pending": False,
                }
            if completed_count != counter:
                return {
                    "classification": "DIVERGENT_COUNTER",
                    "is_valid": False,
                    "error": f"Contador de reconciliações do manifesto ({counter}) diverge das transações COMMITTED no livro-razão ({completed_count}).",
                    "manifest_version": version,
                    "manifest_counter": counter,
                    "ledger_completed_count": completed_count,
                    "ledger_has_pending": False,
                }
            last_rec = ledger_val.get("last_record") or {}
            last_comm_id = str(last_rec.get("audit_id", "")).strip()
            exp_last_id = str(manifest_data.get("last_reconciliation_audit_id", "")).strip()
            if last_comm_id != exp_last_id:
                return {
                    "classification": "DIVERGENT_AUDIT_ID",
                    "is_valid": False,
                    "error": f"Último audit_id de reconciliação no manifesto ('{exp_last_id}') diverge do último COMMITTED no livro-razão ('{last_comm_id}').",
                    "manifest_version": version,
                    "manifest_counter": counter,
                    "ledger_completed_count": completed_count,
                    "ledger_has_pending": False,
                }
            return {
                "classification": "CONSISTENT",
                "is_valid": True,
                "error": None,
                "manifest_version": version,
                "manifest_counter": counter,
                "ledger_completed_count": completed_count,
                "ledger_has_pending": False,
            }
        else: # counter == 0
            if ledger_val["exists"] and completed_count > 0:
                return {
                    "classification": "DIVERGENT_COUNTER",
                    "is_valid": False,
                    "error": f"Manifesto 1.2 registra 0 reconciliações mas livro-razão contém {completed_count} transações COMMITTED.",
                    "manifest_version": version,
                    "manifest_counter": 0,
                    "ledger_completed_count": completed_count,
                    "ledger_has_pending": False,
                }
            return {
                "classification": "CONSISTENT",
                "is_valid": True,
                "error": None,
                "manifest_version": version,
                "manifest_counter": 0,
                "ledger_completed_count": completed_count,
                "ledger_has_pending": False,
            }

    elif version in ("1.0", "1.1"):
        if ledger_val["exists"]:
            return {
                "classification": "LEGACY_CONSISTENT",
                "is_valid": True,
                "error": None,
                "manifest_version": version,
                "manifest_counter": completed_count,
                "ledger_completed_count": completed_count,
                "ledger_has_pending": False,
            }
        else:
            lock_meta = lock_info.get("metadata", {}) or {} if lock_info else {}
            lock_reason = str(lock_meta.get("reason", "") or (lock_info.get("reason", "") if lock_info else ""))
            lock_holder = str(lock_meta.get("holder", "") if lock_meta else (lock_info.get("holder", "") if lock_info else ""))
            had_reconcile = ("reconcile" in lock_holder.lower() and "applied" in lock_reason.lower())
            if had_reconcile:
                return {
                    "classification": "INDETERMINATE",
                    "is_valid": False,
                    "error": "Manifesto legado com evidência de reconciliação no lock operacional, mas o livro-razão (.reconciliation_audit.jsonl) está ausente (histórico indeterminado/anti-tamper).",
                    "manifest_version": version,
                    "manifest_counter": 0,
                    "ledger_completed_count": 0,
                    "ledger_has_pending": False,
                }
            return {
                "classification": "LEGACY_CONSISTENT",
                "is_valid": True,
                "error": None,
                "manifest_version": version,
                "manifest_counter": 0,
                "ledger_completed_count": 0,
                "ledger_has_pending": False,
            }

    return {
        "classification": "INVALID_MANIFEST",
        "is_valid": False,
        "error": f"Versão do manifesto não suportada: '{version}'.",
        "manifest_version": version,
        "manifest_counter": 0,
        "ledger_completed_count": 0,
        "ledger_has_pending": False,
    }


def verify_maintenance_authorization(
    lock_authorization: Any,
    manifest_path: Union[Path, str],
    expected_env_uuid: str,
) -> MaintenanceAuthorizationToken:
    """
    Valida de forma estrita e fail-closed a capacidade de autorização de sessão de manutenção (Gate P0).
    Exige comprovação da posse do lock físico antes de autorizar mutações no manifesto operacional.

    Requisitos:
    - Rejeita estritamente None.
    - Rejeita estritamente booleanos (True/False).
    - Rejeita objetos arbitrários (strings, ints, dicts, instâncias genéricas).
    - Rejeita descritores fechados ou inválidos no SO.
    - Rejeita sessões já liberadas ou revogadas.
    - Rejeita autorizações emitidas para outro environment_uuid.
    - Rejeita autorizações cujo lock path aponte para diretório diferente do manifesto.
    - Valida integridade do descritor aberto (fstat) e correspondência com arquivo em disco.

    Limitações Técnicas de Comprovação de Lock de SO (Windows/Linux):
    - No Windows (msvcrt.locking), a trava é imposta por faixa de bytes (byte 0). O SO não
      fornece uma API não-destrutiva de reflexão ("is_fd_locked_by_current_thread(fd)") que
      possa ser consultada sem tentar re-travar. Tentar re-adquirir a trava com msvcrt no
      mesmo descritor ou em descritor duplicado dentro do mesmo processo resulta em erro de
      permissão (EACCES/EDEADLK).
    - No Linux (fcntl.flock), as travas pertencem à open file table description. flock() é
      idempotente para o mesmo descritor, mas também não expõe syscall de reflexão sem
      tentativa de lock concorrente. Tentar uma nova aquisição isolada arriscaria liberar
      ou substituir a trava pré-existente.
    - Consequentemente, a comprovação física no nível de biblioteca ancora-se na retenção
      ininterrupta do descritor de arquivo aberto sob a trava do kernel durante a sessão de
      manutenção, na validação de liveness do fileno via os.fstat(), e na confirmação de que
      o inodo/dispositivo em disco corresponde ao descritor ativo (anti-unlinked/anti-inode-swap).
    """
    if lock_authorization is None:
        raise RuntimeLockError("Autorização de manutenção obrigatória: lock_authorization não pode ser None.")

    if isinstance(lock_authorization, bool):
        raise RuntimeLockError("Objeto de autorização inválido: booleano rejeitado.")

    token: Optional[MaintenanceAuthorizationToken] = None
    if isinstance(lock_authorization, MaintenanceAuthorizationToken):
        token = lock_authorization
        if token.lock_instance is None:
            raise RuntimeLockError("Autorização inválida: token sem instância de lock associada.")
        if not getattr(token, "_is_active", True):
            raise RuntimeLockError("Autorização inválida: sessão de manutenção já foi liberada ou revogada.")
        if not getattr(token.lock_instance, "_is_maintenance_session", False):
            raise RuntimeLockError("Autorização inválida: token não pertence a uma sessão de manutenção exclusiva.")
        if getattr(token.lock_instance, "active_maintenance_token", None) is not token:
            raise RuntimeLockError("Autorização inválida: token não corresponde à capacidade ativa da sessão de manutenção atual.")
    elif isinstance(lock_authorization, OperationalRuntimeLock):
        if not lock_authorization.is_locked or lock_authorization._file_obj is None:
            raise RuntimeLockError("Autorização inválida: OperationalRuntimeLock fornecido não detém lock ativo.")
        if not getattr(lock_authorization, "_is_maintenance_session", False):
            raise RuntimeLockError("Autorização inválida: OperationalRuntimeLock fornecido não está em sessão de manutenção exclusiva (adquirido via acquire_maintenance).")
        token = lock_authorization.active_maintenance_token
        if token is None or not token.is_active:
            token = lock_authorization.create_maintenance_authorization(environment_uuid=expected_env_uuid)
    else:
        raise RuntimeLockError(
            f"Objeto de autorização inválido: esperado MaintenanceAuthorizationToken ou OperationalRuntimeLock ativo em manutenção, "
            f"recebido {type(lock_authorization).__name__}."
        )

    # 1. Validação de revogação explícita da capacidade
    if not getattr(token, "_is_active", True):
        raise RuntimeLockError("Autorização inválida: sessão de manutenção já foi liberada ou revogada.")

    # 2. Validação do descritor de arquivo físico
    f = token.lock_file_obj
    if f is None or getattr(f, "closed", True):
        raise RuntimeLockError("Autorização inválida: descritor físico de lock fechado ou inexistente.")

    # 3. Validação de sessão ativa
    if not token.is_active:
        raise RuntimeLockError("Autorização inválida: sessão de manutenção já foi liberada ou revogada.")

    try:
        fd = f.fileno()
        stat_fd = os.fstat(fd)
    except Exception as fd_exc:
        raise RuntimeLockError(f"Autorização inválida: descritor físico inválido no SO: {fd_exc}") from fd_exc

    # 3. Validação do environment_uuid
    clean_exp_uuid = str(expected_env_uuid).strip()
    if token.environment_uuid != clean_exp_uuid:
        raise RuntimeLockError(
            f"Autorização inválida: token emitido para outro environment_uuid "
            f"('{token.environment_uuid}' != '{clean_exp_uuid}')."
        )

    # 4. Validação de alinhamento entre lock_path e manifest_path
    m_path = Path(manifest_path).expanduser().resolve()
    expected_lock_path = (m_path.parent / ".operational_runtime.lock").resolve()
    if token.lock_path.resolve() != expected_lock_path:
        raise RuntimeLockError(
            f"Autorização inválida: lock path do token ('{token.lock_path.resolve()}') "
            f"não corresponde ao diretório do manifesto ('{expected_lock_path}')."
        )

    # 5. Validação anti-desvinculação e anti-inode swap
    try:
        stat_disk = os.stat(token.lock_path)
        if hasattr(stat_fd, "st_ino") and hasattr(stat_disk, "st_ino"):
            if stat_fd.st_ino != stat_disk.st_ino or stat_fd.st_dev != stat_disk.st_dev:
                raise RuntimeLockError("Autorização inválida: arquivo de lock em disco foi substituído (inode swap detectado).")
    except FileNotFoundError:
        raise RuntimeLockError("Autorização inválida: arquivo de lock foi desvinculado (unlinked) do disco.")

    return token


def update_runtime_manifest_recovery_state(
    manifest_path: Union[Path, str],
    audit_id: str,
    audit_timestamp_utc: str,
    expected_env_uuid: str,
    ledger_path: Union[Path, str],
    lock_authorization: Optional[Any] = None,
    lock_file_obj: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Atualiza atômica e confiavelmente o manifesto operacional com novo estado de recovery (Gate G4-D B3-B2-B3-H1 / Gate P0).
    Aplica validação incondicional de capacidade de autorização, incremento estrito, gravação em arquivo temporário com flush/fsync e os.replace atômico.
    Nunca aplica os.replace ao lock físico.
    Executa toda a mutação retendo o lock de sincronização da sessão de manutenção para proteção estrita anti-TOCTOU.
    """
    auth_source = lock_authorization if lock_authorization is not None else lock_file_obj
    if auth_source is None:
        raise RuntimeLockError("Autorização de manutenção obrigatória: lock_authorization não pode ser None.")

    token = verify_maintenance_authorization(
        lock_authorization=auth_source,
        manifest_path=manifest_path,
        expected_env_uuid=expected_env_uuid,
    )
    lock_instance = token.lock_instance

    with lock_instance._thread_lock:
        # Revalidação estrita DENTRO da região crítica
        token = verify_maintenance_authorization(
            lock_authorization=auth_source,
            manifest_path=manifest_path,
            expected_env_uuid=expected_env_uuid,
        )

        # Validações estritas dos parâmetros
        try:
            uuid.UUID(str(audit_id).strip())
        except Exception as exc:
            raise RuntimeLockError(f"audit_id fornecido não é um UUID válido: {exc}") from exc

        try:
            datetime.fromisoformat(str(audit_timestamp_utc).strip().replace("Z", "+00:00"))
        except Exception as exc:
            raise RuntimeLockError(f"audit_timestamp_utc fornecido não é um timestamp ISO8601 válido: {exc}") from exc

        p = Path(manifest_path).expanduser().resolve()
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeLockError(f"Manifesto operacional ausente ou vazio em {p}. Impossível atualizar histórico.")

        manifest_data = read_runtime_manifest(p)
        if not manifest_data:
            raise RuntimeLockError(f"Falha ao ler manifesto operacional em {p}.")

        is_valid, err_msg = validate_manifest_structure(manifest_data)
        if not is_valid:
            raise RuntimeLockError(f"Manifesto original inválido antes da atualização: {err_msg}")

        m_env_uuid = str(manifest_data.get("environment_uuid", "")).strip()
        if m_env_uuid != str(expected_env_uuid).strip():
            raise RuntimeLockError(
                f"Divergência de UUID ao atualizar manifesto: manifesto='{m_env_uuid}', esperado='{expected_env_uuid}'."
            )

        # Validar integridade e contagem do ledger pré-transição
        lp = Path(ledger_path).expanduser().resolve()
        ledger_val = validate_recovery_audit_ledger(lp, expected_environment_uuid=expected_env_uuid)
        if ledger_val["exists"] and not ledger_val["is_valid"]:
            raise RuntimeLockError(f"Livro-razão de auditoria inválido ao atualizar manifesto: {ledger_val['error']}")

        completed_count = ledger_val.get("completed_transactions_count", 0)
        version = str(manifest_data.get("manifest_version", "1.0")).strip()

        if version == "1.0":
            base_counter = completed_count
        elif version in ("1.1", "1.2"):
            curr_counter = manifest_data["recovery_history_counter"]
            if curr_counter != completed_count:
                raise RuntimeLockError(
                    f"Inconsistência prévia no manifesto {version}: contador={curr_counter} diverge das transações no ledger={completed_count}."
                )
            base_counter = curr_counter
        else:
            raise RuntimeLockError(f"Versão de manifesto não suportada para migração: '{version}'.")

        new_counter = base_counter + 1

        updated_manifest = dict(manifest_data)
        if version == "1.2" or "reconciliation_history_counter" in manifest_data:
            updated_manifest["manifest_version"] = "1.2"
            updated_manifest["reconciliation_history_counter"] = manifest_data.get("reconciliation_history_counter", 0)
            updated_manifest["last_reconciliation_audit_id"] = manifest_data.get("last_reconciliation_audit_id")
            updated_manifest["last_reconciliation_utc"] = manifest_data.get("last_reconciliation_utc")
        else:
            updated_manifest["manifest_version"] = "1.1"

        updated_manifest["recovery_history_counter"] = new_counter
        updated_manifest["last_recovery_audit_id"] = str(audit_id).strip()
        updated_manifest["last_recovery_utc"] = str(audit_timestamp_utc).strip()

        # Gravação atômica em arquivo temporário no mesmo diretório
        tmp_path = p.parent / f"{MANIFEST_FILENAME}.tmp"
        raw = json.dumps(updated_manifest, indent=2).encode("utf-8")
        try:
            with open(tmp_path, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())

            # Revalidação defensiva imediatamente antes de os.replace()
            if not token.is_active or getattr(token.lock_file_obj, "closed", True):
                raise RuntimeLockError(
                    "Tentativa de persistência atômica abortada: sessão de manutenção foi encerrada antes do replace."
                )

            os.replace(tmp_path, p)

            if hasattr(os, "O_DIRECTORY"):
                try:
                    dir_fd = os.open(str(p.parent), os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except Exception:
                    pass
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise RuntimeLockError(f"Falha de persistência atômica ao atualizar manifesto operacional: {exc}") from exc

        return updated_manifest


def update_runtime_manifest_reconciliation_state(
    manifest_path: Union[Path, str],
    audit_id: str,
    audit_timestamp_utc: str,
    expected_env_uuid: str,
    ledger_path: Union[Path, str],
    lock_authorization: Optional[Any] = None,
    lock_file_obj: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Atualiza atômica e confiavelmente o manifesto operacional com novo estado de reconciliação (Gate G4-D B3-B2-C2-B1 / Gate P0).
    Aplica validação prévia incondicional, incremento estrito, gravação em arquivo temporário com flush/fsync e os.replace atômico.
    Preserva integralmente todos os campos de recovery (Gate H1) e migra manifesto para versão 1.2.
    Executa toda a mutação retendo o lock de sincronização da sessão de manutenção para proteção estrita anti-TOCTOU.
    """
    auth_source = lock_authorization if lock_authorization is not None else lock_file_obj
    if auth_source is None:
        raise RuntimeLockError("Autorização de manutenção obrigatória: lock_authorization não pode ser None.")

    token = verify_maintenance_authorization(
        lock_authorization=auth_source,
        manifest_path=manifest_path,
        expected_env_uuid=expected_env_uuid,
    )
    lock_instance = token.lock_instance

    with lock_instance._thread_lock:
        # Revalidação estrita DENTRO da região crítica
        token = verify_maintenance_authorization(
            lock_authorization=auth_source,
            manifest_path=manifest_path,
            expected_env_uuid=expected_env_uuid,
        )

        try:
            uuid.UUID(str(audit_id).strip())
        except Exception as exc:
            raise RuntimeLockError(f"audit_id fornecido não é um UUID válido: {exc}") from exc

        try:
            datetime.fromisoformat(str(audit_timestamp_utc).strip().replace("Z", "+00:00"))
        except Exception as exc:
            raise RuntimeLockError(f"audit_timestamp_utc fornecido não é um timestamp ISO8601 válido: {exc}") from exc

        p = Path(manifest_path).expanduser().resolve()
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeLockError(f"Manifesto operacional ausente ou vazio em {p}. Impossível atualizar histórico.")

        manifest_data = read_runtime_manifest(p)
        if not manifest_data:
            raise RuntimeLockError(f"Falha ao ler manifesto operacional em {p}.")

        is_valid, err_msg = validate_manifest_structure(manifest_data)
        if not is_valid:
            raise RuntimeLockError(f"Manifesto original inválido antes da atualização: {err_msg}")

        m_env_uuid = str(manifest_data.get("environment_uuid", "")).strip()
        if m_env_uuid != str(expected_env_uuid).strip():
            raise RuntimeLockError(
                f"Divergência de UUID ao atualizar manifesto: manifesto='{m_env_uuid}', esperado='{expected_env_uuid}'."
            )

        from events.fallback_reconciler import validate_reconciliation_audit_ledger
        lp = Path(ledger_path).expanduser().resolve()
        ledger_val = validate_reconciliation_audit_ledger(lp, expected_environment_uuid=expected_env_uuid)
        if ledger_val["exists"] and not ledger_val["is_valid"]:
            clean_aid = str(audit_id).strip()
            if ledger_val.get("orphaned_prepared") == [clean_aid] and "PREPARED sem COMMITTED" in str(ledger_val.get("error", "")):
                pass
            else:
                raise RuntimeLockError(f"Livro-razão de auditoria inválido ao atualizar manifesto: {ledger_val['error']}")

        completed_count = ledger_val.get("completed_transactions_count", 0)
        version = str(manifest_data.get("manifest_version", "1.0")).strip()

        if version in ("1.0", "1.1"):
            base_counter = completed_count
        elif version == "1.2":
            curr_counter = manifest_data["reconciliation_history_counter"]
            if curr_counter != completed_count:
                raise RuntimeLockError(
                    f"Inconsistência prévia no manifesto 1.2: contador={curr_counter} diverge das transações no ledger={completed_count}."
                )
            base_counter = curr_counter
        else:
            raise RuntimeLockError(f"Versão de manifesto não suportada para migração: '{version}'.")

        new_counter = base_counter + 1

        updated_manifest = dict(manifest_data)
        updated_manifest["manifest_version"] = "1.2"
        updated_manifest["reconciliation_history_counter"] = new_counter
        updated_manifest["last_reconciliation_audit_id"] = str(audit_id).strip()
        updated_manifest["last_reconciliation_utc"] = str(audit_timestamp_utc).strip()

        # Preservar campos de recovery do Gate H1
        if "recovery_history_counter" not in updated_manifest:
            updated_manifest["recovery_history_counter"] = 0
            updated_manifest["last_recovery_audit_id"] = None
            updated_manifest["last_recovery_utc"] = None

        tmp_path = p.parent / f"{MANIFEST_FILENAME}.tmp"
        raw = json.dumps(updated_manifest, indent=2).encode("utf-8")
        try:
            with open(tmp_path, "wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())

            # Revalidação defensiva imediatamente antes de os.replace()
            if not token.is_active or getattr(token.lock_file_obj, "closed", True):
                raise RuntimeLockError(
                    "Tentativa de persistência atômica abortada: sessão de manutenção foi encerrada antes do replace."
                )

            os.replace(tmp_path, p)

            if hasattr(os, "O_DIRECTORY"):
                try:
                    dir_fd = os.open(str(p.parent), os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except Exception:
                    pass
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise RuntimeLockError(f"Falha de persistência atômica ao atualizar manifesto operacional: {exc}") from exc

        return updated_manifest


def read_runtime_manifest(manifest_path: Union[Path, str]) -> Optional[Dict[str, Any]]:
    """
    Lê defensivamente o manifesto de inicialização.
    Retorna None se inexistente, vazio ou corrompido.
    """
    p = Path(manifest_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        with open(p, "rb") as f:
            data = json.loads(f.read().decode("utf-8", errors="strict"))
        if isinstance(data, dict) and data.get("environment_uuid"):
            return data
    except Exception:
        pass
    return None


def verify_environment_anti_bypass(
    data_dir: Union[Path, str],
    lock_path: Optional[Union[Path, str]] = None,
) -> Tuple[bool, str]:
    """
    Valida a identificação do ambiente contra bypass de estados anômalos (Gate G4-D B3-B2-A).

    Regras:
    1. Lock ausente com dados existentes => Bloqueado.
    2. Manifesto ausente em ambiente com dados ou com lock => Bloqueado.
    3. Manifesto ausente e dados ausentes e lock ausente => Instalação nova não-inicializada => Bloqueado (exige bootstrap).
    4. Manifesto existente com lock ausente => Bloqueado.
    5. Manifesto corrompido ou sem UUID => Bloqueado.
    """
    d_dir = Path(data_dir).expanduser().resolve()
    l_path = Path(lock_path).expanduser().resolve() if lock_path else (d_dir / ".operational_runtime.lock")
    m_path = d_dir / MANIFEST_FILENAME
    sqlite_file = d_dir / "trading_bot.db"
    jsonl_file = d_dir / "eventos_fluxo.jsonl"

    has_sqlite_data = sqlite_file.exists() and sqlite_file.stat().st_size > 0
    has_jsonl_data = jsonl_file.exists() and jsonl_file.stat().st_size > 0
    has_operational_data = has_sqlite_data or has_jsonl_data

    has_manifest_file = m_path.exists() and m_path.stat().st_size > 0
    has_lock_file = l_path.exists() and l_path.stat().st_size > 0

    # Caso 1: Lock ausente com dados existentes
    if not has_lock_file and has_operational_data:
        return (
            False,
            f"Anti-Bypass: Lock operacional ausente ({l_path.name}) em ambiente com dados operacionais existentes. "
            "Possível tentativa de contornar estado DIRTY/UNKNOWN/RUNNING via deleção do arquivo de lock.",
        )

    # Caso 2: Manifesto ausente em ambiente com dados existentes
    if not has_manifest_file and has_operational_data:
        return (
            False,
            f"Anti-Bypass: Manifesto de inicialização ausente ({MANIFEST_FILENAME}) em ambiente com dados operacionais. "
            "Exige execução prévia de bootstrap controlado (scripts/maintenance/bootstrap_runtime_lock.py --confirm-bootstrap).",
        )

    # Caso 3: Manifesto presente mas lock ausente
    if has_manifest_file and not has_lock_file:
        return (
            False,
            f"Anti-Bypass: Lock operacional ausente ({l_path.name}) em ambiente previamente inicializado "
            f"(manifesto {MANIFEST_FILENAME} presente).",
        )

    # Caso 4: Lock presente mas manifesto ausente
    if has_lock_file and not has_manifest_file:
        return (
            False,
            f"Anti-Bypass: Manifesto de inicialização ausente ({MANIFEST_FILENAME}) em ambiente com lock existente. "
            "Possível exclusão acidental ou adulteração do manifesto.",
        )

    # Caso 5: Instalação totalmente nova (nada existe)
    if not has_lock_file and not has_manifest_file and not has_operational_data:
        return (
            False,
            "Ambiente não inicializado (instalação nova). "
            "Exige execução prévia de bootstrap controlado (scripts/maintenance/bootstrap_runtime_lock.py --confirm-bootstrap).",
        )

    # Caso 6: Ambos manifesto e lock existem -> validar integridade do manifesto
    manifest_data = read_runtime_manifest(m_path)
    if not manifest_data:
        return (
            False,
            f"Anti-Bypass: Manifesto de inicialização ({MANIFEST_FILENAME}) corrompido, vazio ou sem UUID válido.",
        )

    # Caso 7: Validação de consistência de UUID entre manifesto e lock
    lock_info = OperationalRuntimeLock.read_persisted_state(l_path)
    lock_uuid = lock_info.get("metadata", {}).get("environment_uuid")
    manifest_uuid = manifest_data.get("environment_uuid")
    if lock_uuid and manifest_uuid and lock_uuid != manifest_uuid:
        return (
            False,
            f"Anti-Bypass: Divergência de identificador de ambiente (lock UUID='{lock_uuid}' != manifesto UUID='{manifest_uuid}').",
        )

    return True, "Anti-Bypass validado com sucesso (manifesto e lock presentes e íntegros)."



def validate_recovery_audit_ledger(
    audit_file: Union[Path, str],
    expected_environment_uuid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Valida de forma estrita e integral o livro-razão append-only de auditoria de recovery (.recovery_audit.jsonl).
    Analisa 100% das linhas do arquivo, garantindo integridade estrutural, transições canônicas (PREPARED -> COMMITTED),
    ausência de PREPARED órfãos, unicidade de audit_id e correspondência de UUID de ambiente (Gate G4-D B3-B2-B3).
    """
    af_path = Path(audit_file).expanduser().resolve()
    if not af_path.exists():
        return {
            "exists": False,
            "is_valid": True,
            "error": None,
            "records_count": 0,
            "transactions_count": 0,
            "completed_transactions_count": 0,
            "orphaned_prepared": [],
            "has_pending_transaction": False,
            "last_record": None,
            "transactions": {},
        }

    try:
        if af_path.stat().st_size == 0:
            return {
                "exists": True,
                "is_valid": False,
                "error": "Livro-razão de auditoria existe mas está vazio (0 bytes).",
                "records_count": 0,
                "transactions_count": 0,
                "completed_transactions_count": 0,
                "orphaned_prepared": [],
                "has_pending_transaction": False,
                "last_record": None,
                "transactions": {},
            }
    except OSError as e:
        return {
            "exists": True,
            "is_valid": False,
            "error": f"Falha de E/S ao acessar livro-razão de auditoria: {e}",
            "records_count": 0,
            "transactions_count": 0,
            "completed_transactions_count": 0,
            "orphaned_prepared": [],
            "has_pending_transaction": False,
            "last_record": None,
            "transactions": {},
        }

    try:
        with open(af_path, "r", encoding="utf-8", errors="strict") as f:
            raw_lines = f.readlines()
    except Exception as e:
        return {
            "exists": True,
            "is_valid": False,
            "error": f"Falha de leitura/decodificação UTF-8 no livro-razão de auditoria: {e}",
            "records_count": 0,
            "transactions_count": 0,
            "completed_transactions_count": 0,
            "orphaned_prepared": [],
            "has_pending_transaction": False,
            "last_record": None,
            "transactions": {},
        }

    if not raw_lines:
        return {
            "exists": True,
            "is_valid": False,
            "error": "Livro-razão de auditoria existe mas não contém linhas.",
            "records_count": 0,
            "transactions_count": 0,
            "completed_transactions_count": 0,
            "orphaned_prepared": [],
            "has_pending_transaction": False,
            "last_record": None,
            "transactions": {},
        }

    records: List[Dict[str, Any]] = []
    transactions: Dict[str, Dict[str, Any]] = {}
    open_transactions: List[str] = []
    completed_transactions: List[str] = []

    for line_idx, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line:
            return {
                "exists": True,
                "is_valid": False,
                "error": f"Linha {line_idx} contém linha em branco ou truncamento estrutural.",
                "records_count": len(records),
                "transactions_count": len(transactions),
                "completed_transactions_count": len(completed_transactions),
                "orphaned_prepared": list(open_transactions),
                "has_pending_transaction": len(open_transactions) > 0,
                "last_record": records[-1] if records else None,
                "transactions": transactions,
            }

        try:
            entry = json.loads(line)
        except Exception as json_err:
            return {
                "exists": True,
                "is_valid": False,
                "error": f"Linha {line_idx} contém JSON inválido ou truncado: {json_err}",
                "records_count": len(records),
                "transactions_count": len(transactions),
                "completed_transactions_count": len(completed_transactions),
                "orphaned_prepared": list(open_transactions),
                "has_pending_transaction": len(open_transactions) > 0,
                "last_record": records[-1] if records else None,
                "transactions": transactions,
            }

        if not isinstance(entry, dict):
            return {
                "exists": True,
                "is_valid": False,
                "error": f"Linha {line_idx} não é um objeto JSON válido (dict).",
                "records_count": len(records),
                "transactions_count": len(transactions),
                "completed_transactions_count": len(completed_transactions),
                "orphaned_prepared": list(open_transactions),
                "has_pending_transaction": len(open_transactions) > 0,
                "last_record": records[-1] if records else None,
                "transactions": transactions,
            }

        required_fields = ["audit_id", "timestamp_utc", "environment_uuid", "operator_reason", "recovery_phase"]
        for rf in required_fields:
            if rf not in entry or not str(entry[rf]).strip():
                return {
                    "exists": True,
                    "is_valid": False,
                    "error": f"Linha {line_idx} ausente de campo obrigatório ou vazio: '{rf}'.",
                    "records_count": len(records),
                    "transactions_count": len(transactions),
                    "completed_transactions_count": len(completed_transactions),
                    "orphaned_prepared": list(open_transactions),
                    "has_pending_transaction": len(open_transactions) > 0,
                    "last_record": records[-1] if records else None,
                    "transactions": transactions,
                }

        audit_id = str(entry["audit_id"]).strip()
        env_uuid = str(entry["environment_uuid"]).strip()
        phase = str(entry["recovery_phase"]).strip()

        if expected_environment_uuid:
            clean_expected_uuid = str(expected_environment_uuid).strip()
            if env_uuid != clean_expected_uuid:
                return {
                    "exists": True,
                    "is_valid": False,
                    "error": f"Linha {line_idx}: UUID divergente do ambiente (registro='{env_uuid}', esperado='{clean_expected_uuid}').",
                    "records_count": len(records),
                    "transactions_count": len(transactions),
                    "completed_transactions_count": len(completed_transactions),
                    "orphaned_prepared": list(open_transactions),
                    "has_pending_transaction": len(open_transactions) > 0,
                    "last_record": records[-1] if records else None,
                    "transactions": transactions,
                }

        if phase not in ("PREPARED", "COMMITTED"):
            return {
                "exists": True,
                "is_valid": False,
                "error": f"Linha {line_idx}: Fase de recovery desconhecida ou inválida: '{phase}'.",
                "records_count": len(records),
                "transactions_count": len(transactions),
                "completed_transactions_count": len(completed_transactions),
                "orphaned_prepared": list(open_transactions),
                "has_pending_transaction": len(open_transactions) > 0,
                "last_record": records[-1] if records else None,
                "transactions": transactions,
            }

        if phase == "PREPARED":
            for pf in ["snapshot_directory", "snapshot_manifest_sha256", "previous_runtime_state"]:
                if pf not in entry or not str(entry[pf]).strip():
                    return {
                        "exists": True,
                        "is_valid": False,
                        "error": f"Linha {line_idx}: Registro PREPARED sem campo obrigatório '{pf}'.",
                        "records_count": len(records),
                        "transactions_count": len(transactions),
                        "completed_transactions_count": len(completed_transactions),
                        "orphaned_prepared": list(open_transactions),
                        "has_pending_transaction": len(open_transactions) > 0,
                        "last_record": records[-1] if records else None,
                        "transactions": transactions,
                    }

            if audit_id in transactions:
                return {
                    "exists": True,
                    "is_valid": False,
                    "error": f"Linha {line_idx}: Transação duplicada ou audit_id reutilizado para PREPARED (audit_id='{audit_id}').",
                    "records_count": len(records),
                    "transactions_count": len(transactions),
                    "completed_transactions_count": len(completed_transactions),
                    "orphaned_prepared": list(open_transactions),
                    "has_pending_transaction": len(open_transactions) > 0,
                    "last_record": records[-1] if records else None,
                    "transactions": transactions,
                }

            transactions[audit_id] = {
                "audit_id": audit_id,
                "prepared_entry": entry,
                "committed_entry": None,
                "state": "PREPARED",
            }
            open_transactions.append(audit_id)

        elif phase == "COMMITTED":
            for pf in ["snapshot_directory", "snapshot_manifest_sha256", "final_state"]:
                if pf not in entry or not str(entry[pf]).strip():
                    return {
                        "exists": True,
                        "is_valid": False,
                        "error": f"Linha {line_idx}: Registro COMMITTED sem campo obrigatório '{pf}'.",
                        "records_count": len(records),
                        "transactions_count": len(transactions),
                        "completed_transactions_count": len(completed_transactions),
                        "orphaned_prepared": list(open_transactions),
                        "has_pending_transaction": len(open_transactions) > 0,
                        "last_record": records[-1] if records else None,
                        "transactions": transactions,
                    }

            if audit_id not in transactions:
                return {
                    "exists": True,
                    "is_valid": False,
                    "error": f"Linha {line_idx}: Inconsistência estrutural: COMMITTED sem fase PREPARED prévia (audit_id='{audit_id}').",
                    "records_count": len(records),
                    "transactions_count": len(transactions),
                    "completed_transactions_count": len(completed_transactions),
                    "orphaned_prepared": list(open_transactions),
                    "has_pending_transaction": len(open_transactions) > 0,
                    "last_record": records[-1] if records else None,
                    "transactions": transactions,
                }

            tx = transactions[audit_id]
            if tx["state"] == "COMMITTED":
                return {
                    "exists": True,
                    "is_valid": False,
                    "error": f"Linha {line_idx}: COMMITTED duplicado para a transação '{audit_id}'.",
                    "records_count": len(records),
                    "transactions_count": len(transactions),
                    "completed_transactions_count": len(completed_transactions),
                    "orphaned_prepared": list(open_transactions),
                    "has_pending_transaction": len(open_transactions) > 0,
                    "last_record": records[-1] if records else None,
                    "transactions": transactions,
                }

            tx["committed_entry"] = entry
            tx["state"] = "COMMITTED"
            if audit_id in open_transactions:
                open_transactions.remove(audit_id)
            completed_transactions.append(audit_id)

        records.append(entry)

    if not records:
        return {
            "exists": True,
            "is_valid": False,
            "error": "Livro-razão de auditoria não contém registros válidos.",
            "records_count": 0,
            "transactions_count": 0,
            "completed_transactions_count": 0,
            "orphaned_prepared": [],
            "has_pending_transaction": False,
            "last_record": None,
            "transactions": {},
        }

    return {
        "exists": True,
        "is_valid": True,
        "error": None,
        "records_count": len(records),
        "transactions_count": len(transactions),
        "completed_transactions_count": len(completed_transactions),
        "orphaned_prepared": list(open_transactions),
        "has_pending_transaction": len(open_transactions) > 0,
        "last_record": records[-1] if records else None,
        "transactions": transactions,
    }



def verify_startup_preflight(
    data_dir: Union[Path, str],
    lock_path: Optional[Union[Path, str]] = None,
    sqlite_path: Optional[Union[Path, str]] = None,
    jsonl_path: Optional[Union[Path, str]] = None,
    archive_dir: Optional[Union[Path, str]] = None,
) -> Tuple[bool, str]:
    """
    Executa a cadeia completa de preflight de segurança e integridade para o startup normal (Gate G4-D B3-B2-A).

    Ordem de Verificação:
    1. Anti-Bypass de ambiente e manifesto.
    2. Inspeção do livro-razão de auditoria de recovery contra corrupção e transações incompletas.
    3. Inspeção read-only do estado do lock: exige estritamente CLEAN.
       Bloqueia DIRTY, UNKNOWN, RUNNING residual, lock vazio ou corrompido.
    4. Integridade estrutural e referencial do SQLite (verify_sqlite_integrity).
    5. Integridade da terminação e última linha do JSONL ativo (verify_jsonl_integrity).
    6. Integridade do diretório de archive e ausência de .tmp (verify_archive_integrity).

    Retorna: (is_ok, reason_message)
    """
    d_dir = Path(data_dir).expanduser().resolve()
    l_path = Path(lock_path).expanduser().resolve() if lock_path else (d_dir / ".operational_runtime.lock")
    s_path = Path(sqlite_path).expanduser().resolve() if sqlite_path else (d_dir / "trading_bot.db")
    j_path = Path(jsonl_path).expanduser().resolve() if jsonl_path else (d_dir / "eventos_fluxo.jsonl")
    a_dir = Path(archive_dir).expanduser().resolve() if archive_dir else (d_dir / "archive")

    # 1. Anti-Bypass
    ab_ok, ab_msg = verify_environment_anti_bypass(d_dir, l_path)
    if not ab_ok:
        return False, ab_msg

    # 2. Inspeção do livro-razão e consistência com manifesto operacional (Gate G4-D B3-B2-B3-H1)
    audit_file = d_dir / ".recovery_audit.jsonl"
    m_path = d_dir / MANIFEST_FILENAME
    manifest_data = read_runtime_manifest(m_path) if m_path.exists() else None
    env_uuid = manifest_data.get("environment_uuid") if manifest_data else None
    lock_info_pre = OperationalRuntimeLock.read_persisted_state(l_path)

    consistency = verify_manifest_recovery_consistency(
        manifest_data=manifest_data,
        ledger_path=audit_file,
        expected_env_uuid=env_uuid,
        lock_info=lock_info_pre,
    )
    if not consistency["is_valid"]:
        return (
            False,
            f"Startup bloqueado fail-closed: {consistency['error']}",
        )

    # 2b. Inspeção do livro-razão de reconciliação e consistência com manifesto operacional (Gate G4-D B3-B2-C2-B1)
    rec_audit_file = d_dir / ".reconciliation_audit.jsonl"
    rec_consistency = verify_manifest_reconciliation_consistency(
        manifest_data=manifest_data,
        ledger_path=rec_audit_file,
        expected_env_uuid=env_uuid,
        lock_info=lock_info_pre,
    )
    if not rec_consistency["is_valid"]:
        return (
            False,
            f"Startup bloqueado fail-closed: {rec_consistency['error']}",
        )

    # 3. Inspeção read-only do estado do lock
    state_info = OperationalRuntimeLock.read_persisted_state(l_path)
    current_status = state_info.get("status")

    if current_status != RuntimeState.CLEAN:
        raw = state_info.get("raw_status")
        reason = state_info.get("reason", "status_invalido")
        return (
            False,
            f"Startup bloqueado: estado operacional anterior é '{current_status}' "
            f"(raw={raw}, motivo={reason}). O startup normal exige estritamente estado anterior CLEAN. "
            "Evidências no disco preservadas intactas sem modificação.",
        )

    # 4. Integridade read-only SQLite, JSONL e Archive
    try:
        from events.fallback_reconciler import (
            verify_sqlite_integrity,
            verify_jsonl_integrity,
            verify_archive_integrity,
        )
    except ImportError as ie:
        return False, f"Falha ao importar verificadores de integridade: {ie}"

    sq_ok, sq_msg = verify_sqlite_integrity(s_path)
    if not sq_ok:
        return False, f"Falha de integridade SQLite: {sq_msg}"

    jl_ok, jl_msg = verify_jsonl_integrity(j_path)
    if not jl_ok:
        return False, f"Falha de integridade JSONL: {jl_msg}"

    arch_ok, arch_msg = verify_archive_integrity(a_dir)
    if not arch_ok:
        return False, f"Falha de integridade Archive: {arch_msg}"

    return True, "Preflight de startup 100% aprovado (estado anterior CLEAN e dados íntegros)."


def bootstrap_runtime_environment(
    data_dir: Union[Path, str],
    lock_path: Optional[Union[Path, str]] = None,
    sqlite_path: Optional[Union[Path, str]] = None,
    jsonl_path: Optional[Union[Path, str]] = None,
    archive_dir: Optional[Union[Path, str]] = None,
    holder_name: str = "bootstrap_runtime_lock",
) -> Dict[str, Any]:
    """
    Executa o procedimento controlado e único de bootstrap do ambiente operacional (Gate G4-D B3-B2-A).

    Contrato de Segurança:
    1. Operação estritamente offline: exige obtenção de lock físico exclusivo do SO.
    2. Execução única: bloqueia se o manifesto já existir e o lock estiver inicializado.
    3. Integridade read-only: dados SQLite, JSONL e Archive existentes devem estar 100% íntegros.
    4. Nenhuma escrita em SQLite, JSONL ou fallback_events.
    5. Grava .runtime_initialized.manifest com UUID, timestamp UTC e sem segredos/credenciais (com fsync).
    6. Persiste lock operacional com status CLEAN (com prefixo b'#' e fsync).
    """
    d_dir = Path(data_dir).expanduser().resolve()
    l_path = Path(lock_path).expanduser().resolve() if lock_path else (d_dir / ".operational_runtime.lock")
    m_path = d_dir / MANIFEST_FILENAME
    s_path = Path(sqlite_path).expanduser().resolve() if sqlite_path else (d_dir / "trading_bot.db")
    j_path = Path(jsonl_path).expanduser().resolve() if jsonl_path else (d_dir / "eventos_fluxo.jsonl")
    a_dir = Path(archive_dir).expanduser().resolve() if archive_dir else (d_dir / "archive")

    # 1. Anti-repetição: verificar se o ambiente já foi inicializado
    if m_path.exists() and m_path.stat().st_size > 0:
        manifest_data = read_runtime_manifest(m_path)
        if manifest_data:
            if l_path.exists() and l_path.stat().st_size > 0:
                lock_state = OperationalRuntimeLock.read_persisted_state(l_path)
                if lock_state.get("status") in (RuntimeState.CLEAN, RuntimeState.RUNNING, RuntimeState.DIRTY):
                    raise RuntimeLockError(
                        "Bootstrap repetido bloqueado: o ambiente já foi previamente inicializado com sucesso "
                        f"(UUID={manifest_data.get('environment_uuid')}, status={lock_state.get('status')})."
                    )

    # 2. Verificação de integridade read-only de dados existentes
    from events.fallback_reconciler import (
        verify_sqlite_integrity,
        verify_jsonl_integrity,
        verify_archive_integrity,
    )

    sq_ok, sq_msg = verify_sqlite_integrity(s_path)
    if not sq_ok:
        raise RuntimeLockError(f"Bootstrap abortado: integridade SQLite comprometida: {sq_msg}")

    jl_ok, jl_msg = verify_jsonl_integrity(j_path)
    if not jl_ok:
        raise RuntimeLockError(f"Bootstrap abortado: integridade JSONL comprometida: {jl_msg}")

    arch_ok, arch_msg = verify_archive_integrity(a_dir)
    if not arch_ok:
        raise RuntimeLockError(f"Bootstrap abortado: integridade Archive comprometida: {arch_msg}")

    # 3. Exclusividade offline via kernel lock
    d_dir.mkdir(parents=True, exist_ok=True)
    if not l_path.exists():
        open(l_path, "a+b").close()

    try:
        f = open(l_path, "r+b")
    except Exception as open_err:
        raise RuntimeLockError(f"Não foi possível abrir arquivo de lock em {l_path}: {open_err}") from open_err

    system = platform.system()
    acquired = False
    try:
        if system == "Windows":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            acquired = True
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
    except (BlockingIOError, OSError) as lock_exc:
        try:
            f.close()
        except Exception:
            pass
        raise RuntimeLockError(
            f"Bootstrap bloqueado por processo concorrente: não foi possível adquirir exclusividade em {l_path} ({lock_exc})."
        ) from lock_exc

    try:
        # 4. Grava o manifesto permanente com fsync
        manifest_meta = write_runtime_manifest(m_path, holder_name=holder_name)

        # 5. Persiste estado CLEAN no lock operacional com fsync
        lock_meta = {
            "status": RuntimeState.CLEAN.value,
            "holder": holder_name,
            "pid": os.getpid(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "host": platform.node(),
            "reason": "bootstrap_initialization",
            "environment_uuid": manifest_meta["environment_uuid"],
        }
        raw_lock_bytes = json.dumps(lock_meta, indent=2).encode("utf-8")
        payload = b"#" + raw_lock_bytes
        f.seek(0)
        f.write(payload)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())

    finally:
        # 6. Libera o lock no SO e fecha o descritor
        try:
            if system == "Windows":
                import msvcrt
                f.seek(0)
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        finally:
            try:
                f.close()
            except Exception:
                pass

    return {
        "success": True,
        "environment_uuid": manifest_meta["environment_uuid"],
        "lock_path": str(l_path),
        "manifest_path": str(m_path),
        "status": RuntimeState.CLEAN.value,
    }
