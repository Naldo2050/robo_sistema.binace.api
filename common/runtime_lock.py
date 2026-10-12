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
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

logger = logging.getLogger("common.runtime_lock")

# Âncora global de módulo contra desalocação prematura pelo Garbage Collector (GC)
# durante encerramento com threads ativas (Gate G4-D B2 Hardened).
_RETAINED_LOCKS: List["OperationalRuntimeLock"] = []
_RETAINED_FILE_HANDLES: List[Any] = []


class RuntimeLockError(RuntimeError):
    """Exceção levantada quando a aquisição do lock operacional falha (fail-closed)."""
    pass


class OperationalRuntimeLock:
    """
    Gerenciador de lock cooperativo multiplataforma baseado em descritores de arquivo do kernel.

    Características de Segurança:
    1. Multiplataforma nativa: fcntl.flock no Linux, msvcrt.locking no Windows.
    2. Zero dependência de texto volátil: o lock é mantido pelo descritor de arquivo aberto.
    3. Liberação automática pelo SO em caso de crash (SIGKILL, segfault, power loss).
    4. Não-bloqueante por padrão com política fail-closed sob colisão.
    5. Arquivo persistente: nunca remove o arquivo de lock no release para evitar corridas de inodo/handle.
    6. Idempotência estrita: liberação dupla é tratada de forma segura sem exceção.
    7. Caminho canônico: resolve caminhos relativos e links simbólicos para o mesmo arquivo físico.
    8. Retenção defensiva: capacidade de reter o lock mesmo após retorno de main() se writers continuarem ativos.
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

    def acquire(self, timeout: float = 0.0) -> None:
        """
        Adquire o lock exclusivo no arquivo.

        Args:
            timeout: Tempo máximo em segundos para tentar adquirir. Se 0.0, é puramente não-bloqueante.

        Raises:
            RuntimeLockError: Se o lock já estiver detido por outro processo ou não puder ser adquirido.
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

            # Abre o arquivo em modo leitura/escrita binária (cria se não existir sem truncar)
            try:
                f = open(self.lock_path, "a+b")
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

            # Grava metadados informativos no arquivo sem invalidar o lock
            try:
                metadata = {
                    "holder": self.holder_name,
                    "pid": os.getpid(),
                    "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                    "host": platform.node(),
                }
                raw_bytes = json.dumps(metadata, indent=2).encode("utf-8")
                f.seek(0)
                f.truncate(0)
                f.write(raw_bytes)
                f.flush()
            except Exception as meta_err:
                logger.debug(f"Aviso ao registrar metadados no arquivo de lock: {meta_err}")

            self._file_obj = f
            self._is_locked = True
            logger.info(
                f"Lock exclusivo de runtime adquirido com sucesso: {self.lock_path} (holder={self.holder_name})"
            )

    def retain(self, reason: str = "") -> None:
        """
        Retém explicitamente o lock no kernel do SO durante shutdown incompleto.

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

            logger.critical(
                "Lock operacional RETIDO no SO por motivo de segurança (shutdown incompleto): "
                "%s (motivo: %s)",
                self.lock_path,
                reason or "writers operacionais ativos",
            )

    def release(self) -> None:
        """
        Libera explicitamente o lock e fecha o descritor de arquivo.
        Operação estritamente idempotente e segura contra chamadas repetidas.
        O arquivo no disco é preservado para evitar condições de corrida de handle/inodo.
        Recusa liberação caso o lock tenha sido marcado como retido.
        """
        with self._thread_lock:
            if getattr(self, "_is_retained", False):
                logger.warning(
                    "Tentativa de release ignorada: lock %s está explicitamente "
                    "retido devido a shutdown incompleto com writers operacionais.",
                    self.lock_path,
                )
                return

            if not self._is_locked or self._file_obj is None:
                return

            f = self._file_obj
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

            logger.info(f"Lock exclusivo de runtime liberado com sucesso: {self.lock_path}")

    def _read_existing_metadata_safe(self, f) -> Optional[Dict[str, Any]]:
        """Lê metadados existentes no arquivo aberto para enriquecer mensagens de erro."""
        try:
            f.seek(0)
            content = f.read(1024)
            if content:
                return json.loads(content.decode("utf-8", errors="ignore"))
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
