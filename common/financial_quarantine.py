# common/financial_quarantine.py
"""
Módulo de Preservação Forense Durável (Quarentena Financeira).

Fase 5.11 - Gate H1-B2-Q1-H:
Componente desacoplado, sem dependências de infraestrutura de persistência
operacional, responsável por persistir com durabilidade física (fsync),
lock exclusivo em offset fixo e confinamento seguro todos os eventos
rejeitados (INVALID), não mapeados (UNCLASSIFIED) ou em conflito (COLLISION).

Garante serialização estruturada e tipada versionada (quarantine_format_version = 2,
encoding = "TYPED_TREE_V1") sem colisões semânticas entre números, Decimals e strings.
Distingue estritamente CONFIRMED, FAILED_BEFORE_WRITE e PERSISTENCE_UNCERTAIN.
"""

from __future__ import annotations

import decimal
import errno
import json
import math
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from common.financial_validator import FinancialValidationResult

QUARANTINE_FORMAT_VERSION: int = 2
PAYLOAD_ENCODING: str = "TYPED_TREE_V1"

# Fases do ciclo de persistência forense (Gate H1-B2-Q1-H2)
_PHASE_BEFORE_WRITE: str = "BEFORE_WRITE"
_PHASE_WRITE_STARTED: str = "WRITE_STARTED"
_PHASE_WRITE_DONE: str = "WRITE_DONE"
_PHASE_FLUSH_DONE: str = "FLUSH_DONE"
_PHASE_FSYNC_DONE: str = "FSYNC_DONE"
_PHASE_CLOSE_DONE: str = "CLOSE_DONE"


@dataclass(frozen=True)
class QuarantineWriteResult:
    """
    Resultado imutável e estruturado da operação de quarentena (Gate H1-B2-Q1-H).

    Status possíveis:
      - CONFIRMED: Escrita e fsync concluídos com sucesso comprovado.
      - FAILED_BEFORE_WRITE: Falha antes de qualquer escrita física no arquivo.
      - PERSISTENCE_UNCERTAIN: Falha durante write/flush/fsync (persistência física incerta).
    """

    status: str  # "CONFIRMED" | "FAILED_BEFORE_WRITE" | "PERSISTENCE_UNCERTAIN"
    success: bool
    file_path: Optional[str]
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Validação defensiva de invariantes lógicas estritas."""
        if self.status == "CONFIRMED":
            if not self.success:
                raise ValueError("Invariante violada: status 'CONFIRMED' exige success=True")
            if self.file_path is None or not str(self.file_path).strip():
                raise ValueError("Invariante violada: status 'CONFIRMED' exige file_path preenchido")
            if self.error_type is not None:
                raise ValueError("Invariante violada: status 'CONFIRMED' exige error_type=None")
            if self.error_message is not None:
                raise ValueError("Invariante violada: status 'CONFIRMED' exige error_message=None")
        elif self.status in ("FAILED_BEFORE_WRITE", "PERSISTENCE_UNCERTAIN"):
            if self.success:
                raise ValueError(f"Invariante violada: status '{self.status}' exige success=False")
            if self.error_type is None or not str(self.error_type).strip():
                raise ValueError(f"Invariante violada: status '{self.status}' exige error_type não vazio")
            if self.error_message is None or not str(self.error_message).strip():
                raise ValueError(f"Invariante violada: status '{self.status}' exige error_message não vazio")
        else:
            raise ValueError(
                f"Invariante violada: status desconhecido '{self.status}'. "
                "Deve ser 'CONFIRMED', 'FAILED_BEFORE_WRITE' ou 'PERSISTENCE_UNCERTAIN'"
            )


def encode_typed_payload(val: Any, seen: Optional[Set[int]] = None) -> Dict[str, Any]:
    """
    Codifica o payload recursivamente em nós estruturados tipados ('TYPED_TREE_V1').

    Distingue inequivocamente:
    - None -> {"type": "null"}
    - bool -> {"type": "bool", "value": bool} (tratado antes de int)
    - int -> {"type": "int", "value": int}
    - str -> {"type": "str", "value": str}
    - float:
        * nan -> {"type": "float", "special": "NaN"}
        * +inf -> {"type": "float", "special": "Infinity"}
        * -inf -> {"type": "float", "special": "-Infinity"}
        * -0.0 -> {"type": "float", "value": 0.0, "is_negative_zero": True}
        * finito -> {"type": "float", "value": float}
    - Decimal:
        * nan -> {"type": "decimal", "special": "NaN"}
        * +inf -> {"type": "decimal", "special": "Infinity"}
        * -inf -> {"type": "decimal", "special": "-Infinity"}
        * finito -> {"type": "decimal", "value": str(Decimal)}
    - dict -> {"type": "dict", "entries": [{"key": <node>, "value": <node>}, ...]}
    - list -> {"type": "list", "items": [<node>, ...]}
    - tuple -> {"type": "tuple", "items": [<node>, ...]}
    - set -> {"type": "set", "items": [<node>, ...]}
    - frozenset -> {"type": "frozenset", "items": [<node>, ...]}

    Rejeita tipos não suportados levantando TypeError sem executar repr() hostil.
    Detecta referências circulares levantando ValueError.
    Nunca muta os objetos de entrada.
    """
    if val is None:
        return {"type": "null"}

    if isinstance(val, bool):
        return {"type": "bool", "value": val}

    if isinstance(val, int):
        return {"type": "int", "value": val}

    if isinstance(val, str):
        return {"type": "str", "value": val}

    if isinstance(val, float):
        if math.isnan(val):
            return {"type": "float", "special": "NaN"}
        if math.isinf(val):
            return {"type": "float", "special": "-Infinity" if val < 0 else "Infinity"}
        if val == 0.0 and math.copysign(1.0, val) < 0.0:
            return {"type": "float", "value": 0.0, "is_negative_zero": True}
        return {"type": "float", "value": val}

    if isinstance(val, decimal.Decimal):
        if val.is_nan():
            return {"type": "decimal", "special": "NaN"}
        if val.is_infinite():
            return {"type": "decimal", "special": "-Infinity" if val < 0 else "Infinity"}
        return {"type": "decimal", "value": str(val)}

    if isinstance(val, dict):
        if seen is None:
            seen = set()
        val_id = id(val)
        if val_id in seen:
            raise ValueError("Referência circular detectada no payload durante serialização tipada")
        seen.add(val_id)
        entries = []
        for k, v in val.items():
            k_node = encode_typed_payload(k, seen)
            v_node = encode_typed_payload(v, seen)
            entries.append({"key": k_node, "value": v_node})
        seen.remove(val_id)
        return {"type": "dict", "entries": entries}

    if isinstance(val, (list, tuple, set, frozenset)):
        if seen is None:
            seen = set()
        val_id = id(val)
        if val_id in seen:
            raise ValueError("Referência circular detectada no container durante serialização tipada")
        seen.add(val_id)
        items = [encode_typed_payload(x, seen) for x in val]
        seen.remove(val_id)
        type_name = (
            "list"
            if isinstance(val, list)
            else ("tuple" if isinstance(val, tuple) else ("set" if isinstance(val, set) else "frozenset"))
        )
        return {"type": type_name, "items": items}

    raise TypeError(
        f"Tipo não suportado para serialização em quarentena: {type(val).__name__}. "
        "Truncamento ou conversão cega para repr() é terminantemente vedada."
    )


def decode_typed_payload(node: Dict[str, Any]) -> Any:
    """
    Decodificador puro de referência que reconstrói os tipos de dados originais
    a partir de um nó tipado 'TYPED_TREE_V1'.
    """
    if not isinstance(node, dict) or "type" not in node:
        raise ValueError("Nó estrutural inválido: esperado dicionário com chave 'type'")

    node_type = node["type"]

    if node_type == "null":
        return None
    if node_type == "bool":
        return bool(node["value"])
    if node_type == "int":
        return int(node["value"])
    if node_type == "str":
        return str(node["value"])
    if node_type == "float":
        special = node.get("special")
        if special == "NaN":
            return float("nan")
        if special == "Infinity":
            return float("inf")
        if special == "-Infinity":
            return float("-inf")
        if node.get("is_negative_zero"):
            return -0.0
        return float(node["value"])
    if node_type == "decimal":
        special = node.get("special")
        if special == "NaN":
            return decimal.Decimal("NaN")
        if special == "Infinity":
            return decimal.Decimal("Infinity")
        if special == "-Infinity":
            return decimal.Decimal("-Infinity")
        return decimal.Decimal(node["value"])
    if node_type == "dict":
        d = {}
        for entry in node.get("entries", []):
            k = decode_typed_payload(entry["key"])
            v = decode_typed_payload(entry["value"])
            d[k] = v
        return d
    if node_type == "list":
        return [decode_typed_payload(item) for item in node.get("items", [])]
    if node_type == "tuple":
        return tuple(decode_typed_payload(item) for item in node.get("items", []))
    if node_type == "set":
        return {decode_typed_payload(item) for item in node.get("items", [])}
    if node_type == "frozenset":
        return frozenset(decode_typed_payload(item) for item in node.get("items", []))

    raise ValueError(f"Tipo estrutural desconhecido no decoder: '{node_type}'")


def _has_symlink(path: Path) -> bool:
    """Verifica defensivamente se algum componente do caminho é um symlink."""
    curr = path
    while True:
        try:
            if curr.is_symlink():
                return True
        except (OSError, ValueError):
            pass
        if curr.parent == curr:
            break
        curr = curr.parent
    return False


class FinancialQuarantine:
    """
    Gerenciador forense de quarentena append-only com garantias de durabilidade.
    """

    def __init__(self, quarantine_dir: Path | str) -> None:
        if not quarantine_dir:
            raise ValueError("quarantine_dir não pode ser vazio")
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantined_invalid_count: int = 0
        self.quarantined_collision_count: int = 0
        self.quarantined_unclassified_count: int = 0
        self.total_quarantined_count: int = 0
        self.enospc_error_count: int = 0

    def quarantine_event(
        self,
        payload: Any,
        validation_result: FinancialValidationResult,
        source_caller: str,
        category: Optional[str] = None,
    ) -> QuarantineWriteResult:
        """
        Grava um evento em quarentena com fsync e lock de arquivo sobre offset fixo.

        Parâmetros:
        - payload: Objeto bruto recebido (não mutado).
        - validation_result: Instância comprovada de FinancialValidationResult.
        - source_caller: Identificação do subsistema chamador.
        - category: Categoria opcional ("INVALID", "UNCLASSIFIED", "COLLISION").
        """
        # 1. Validação estrita de argumentos
        if not isinstance(validation_result, FinancialValidationResult):
            raise TypeError("validation_result deve ser instância de FinancialValidationResult")
        if not isinstance(source_caller, str) or not source_caller.strip():
            raise ValueError("source_caller deve ser string não vazia")

        # 2. Resolução da categoria de destino
        if category is not None:
            cat_clean = category.strip().upper()
        elif validation_result.category and validation_result.category.strip().upper() == "COLLISION":
            cat_clean = "COLLISION"
        else:
            cat_clean = validation_result.status.strip().upper()

        if cat_clean not in ("INVALID", "UNCLASSIFIED", "COLLISION"):
            raise ValueError(
                f"Categoria inválida: '{cat_clean}'. Deve ser INVALID, UNCLASSIFIED ou COLLISION"
            )

        # 3. Serialização tipada e versionada do payload (sem truncamento silencioso)
        try:
            typed_payload = encode_typed_payload(payload)
        except (TypeError, ValueError) as enc_err:
            return QuarantineWriteResult(
                status="FAILED_BEFORE_WRITE",
                success=False,
                file_path=None,
                error_type=type(enc_err).__name__,
                error_message=f"Falha de serialização tipada: {enc_err}",
            )

        subfolder_name = cat_clean.lower()
        now_dt = datetime.now(timezone.utc)
        date_str = now_dt.strftime("%Y%m%d")
        now_iso = now_dt.isoformat()
        q_id = str(uuid.uuid4())

        target_dir = self.quarantine_dir / subfolder_name
        target_file = target_dir / f"{subfolder_name}_events_{date_str}.jsonl"

        record = {
            "quarantine_format_version": QUARANTINE_FORMAT_VERSION,
            "quarantine_id": q_id,
            "quarantine_timestamp_utc": now_iso,
            "category": cat_clean,
            "rejection_reason": validation_result.rejection_reason,
            "rejected_field": validation_result.rejected_field,
            "rejected_value_repr": validation_result.rejected_value_repr,
            "source_caller": source_caller.strip(),
            "original_payload_encoding": PAYLOAD_ENCODING,
            "original_payload": typed_payload,
        }

        try:
            json_line = json.dumps(record, ensure_ascii=False)
            raw_bytes = (json_line + "\n").encode("utf-8")
        except Exception as json_err:
            return QuarantineWriteResult(
                status="FAILED_BEFORE_WRITE",
                success=False,
                file_path=str(target_file),
                error_type=type(json_err).__name__,
                error_message=f"Falha ao codificar JSON do envelope: {json_err}",
            )

        # 4. Validação de segurança de caminhos (symlinks e confinamento)
        if _has_symlink(self.quarantine_dir) or _has_symlink(target_file):
            return QuarantineWriteResult(
                status="FAILED_BEFORE_WRITE",
                success=False,
                file_path=str(target_file),
                error_type="PATH_SECURITY_VIOLATION",
                error_message="Symlink detectado no caminho de quarentena",
            )

        try:
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            q_dir_res = self.quarantine_dir.resolve()
            target_res = target_file.resolve()
            if not target_res.is_relative_to(q_dir_res):
                return QuarantineWriteResult(
                    status="FAILED_BEFORE_WRITE",
                    success=False,
                    file_path=str(target_file),
                    error_type="PATH_SECURITY_VIOLATION",
                    error_message=f"Arquivo de destino fora do quarantine_dir: {target_res} não é relativo a {q_dir_res}",
                )
        except (OSError, PermissionError) as dir_err:
            return QuarantineWriteResult(
                status="FAILED_BEFORE_WRITE",
                success=False,
                file_path=str(target_file),
                error_type=type(dir_err).__name__,
                error_message=f"Falha de I/O na criação do diretório de quarentena: {dir_err}",
            )

        # 5. Abertura segura com Lock em Região Fixa (Offset 0, Comprimento 1) e Verificação de Fronteira
        # Garante a existência do arquivo para abertura estável em modo binário r+b
        try:
            if not target_file.exists():
                target_file.touch()
        except (OSError, PermissionError) as touch_err:
            if getattr(touch_err, "errno", None) == errno.ENOSPC or "No space left on device" in str(touch_err):
                self.enospc_error_count += 1
            return QuarantineWriteResult(
                status="FAILED_BEFORE_WRITE",
                success=False,
                file_path=str(target_file),
                error_type="ENOSPC" if getattr(touch_err, "errno", None) == errno.ENOSPC else type(touch_err).__name__,
                error_message=f"Falha ao inicializar arquivo de quarentena: {touch_err}",
            )

        f = None
        locked = False
        phase = _PHASE_BEFORE_WRITE
        pre_write_error: Optional[Tuple[str, str]] = None
        primary_io_error: Optional[Tuple[str, str]] = None

        try:
            try:
                f = open(target_file, "r+b")
            except (OSError, PermissionError, IOError) as open_err:
                return QuarantineWriteResult(
                    status="FAILED_BEFORE_WRITE",
                    success=False,
                    file_path=str(target_file),
                    error_type=type(open_err).__name__,
                    error_message=f"Falha ao abrir arquivo de quarentena: {open_err}",
                )
            except Exception as open_err:
                return QuarantineWriteResult(
                    status="FAILED_BEFORE_WRITE",
                    success=False,
                    file_path=str(target_file),
                    error_type=type(open_err).__name__,
                    error_message=f"Falha inesperada ao abrir arquivo de quarentena: {open_err}",
                )

            # Lock físico cooperativo em região fixa
            try:
                if platform.system() == "Windows":
                    import msvcrt
                    # Região FIXA: sempre offset 0, comprimento 1 byte
                    f.seek(0, os.SEEK_SET)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
            except (OSError, IOError) as lock_err:
                pre_write_error = (
                    "LOCK_ACQUISITION_FAILED",
                    f"Falha ao adquirir lock exclusivo no arquivo: {lock_err}",
                )
            except Exception as lock_err:
                pre_write_error = (
                    "LOCK_ACQUISITION_FAILED",
                    f"Falha ao adquirir lock exclusivo no arquivo: {lock_err}",
                )

            # 6. Inspeção de fronteira JSONL sob lock
            if pre_write_error is None:
                try:
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    if file_size > 0:
                        f.seek(-1, os.SEEK_END)
                        last_byte = f.read(1)
                        if last_byte != b"\n":
                            pre_write_error = (
                                "PREEXISTING_JSONL_TRUNCATION",
                                "Arquivo de quarentena pré-existente não termina com quebra de linha",
                            )
                except (OSError, IOError) as inspect_err:
                    pre_write_error = (
                        type(inspect_err).__name__,
                        f"Falha durante inspeção de fronteira JSONL: {inspect_err}",
                    )
                except Exception as inspect_err:
                    pre_write_error = (
                        type(inspect_err).__name__,
                        f"Falha durante inspeção de fronteira JSONL: {inspect_err}",
                    )

            # 7. Posicionamento e escrita sob lock com rastreamento granular de fases
            if pre_write_error is None and not f.closed:
                try:
                    f.seek(0, os.SEEK_END)
                except (OSError, IOError) as seek_err:
                    pre_write_error = (
                        type(seek_err).__name__,
                        f"Falha ao posicionar ponteiro no final do arquivo: {seek_err}",
                    )
                except Exception as seek_err:
                    pre_write_error = (
                        type(seek_err).__name__,
                        f"Falha inesperada ao posicionar ponteiro: {seek_err}",
                    )

            if pre_write_error is None and not f.closed:
                phase = _PHASE_WRITE_STARTED
                try:
                    f.write(raw_bytes)
                    phase = _PHASE_WRITE_DONE
                    f.flush()
                    phase = _PHASE_FLUSH_DONE
                    os.fsync(f.fileno())
                    phase = _PHASE_FSYNC_DONE
                except (OSError, IOError) as write_err:
                    if getattr(write_err, "errno", None) == errno.ENOSPC or "No space left on device" in str(write_err):
                        self.enospc_error_count += 1
                        err_name = "ENOSPC"
                    else:
                        err_name = type(write_err).__name__
                    primary_io_error = (
                        err_name,
                        f"Falha durante escrita/flush/fsync (fase {phase}): {write_err}",
                    )
                except Exception as write_err:
                    primary_io_error = (
                        type(write_err).__name__,
                        f"Falha inesperada durante escrita/flush/fsync (fase {phase}): {write_err}",
                    )

        finally:
            unlock_err: Optional[Tuple[str, str]] = None
            if f is not None and locked:
                try:
                    if platform.system() == "Windows":
                        import msvcrt
                        f.seek(0, os.SEEK_SET)
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    locked = False
                except (OSError, IOError) as u_err:
                    unlock_err = (type(u_err).__name__, f"Falha ao liberar lock: {u_err}")
                except Exception as u_err:
                    unlock_err = (type(u_err).__name__, f"Falha inesperada ao liberar lock: {u_err}")

            close_err: Optional[Tuple[str, str]] = None
            if f is not None:
                try:
                    f.close()
                    if (
                        phase == _PHASE_FSYNC_DONE
                        and pre_write_error is None
                        and primary_io_error is None
                        and unlock_err is None
                    ):
                        phase = _PHASE_CLOSE_DONE
                except (OSError, IOError) as c_err:
                    close_err = (type(c_err).__name__, f"Falha ao fechar arquivo: {c_err}")
                except Exception as c_err:
                    close_err = (type(c_err).__name__, f"Falha inesperada ao fechar arquivo: {c_err}")

            # Limpeza defensiva de arquivo vazio residual em caso de falha primária de I/O
            if primary_io_error is not None and target_file.exists():
                try:
                    if target_file.stat().st_size == 0:
                        target_file.unlink()
                except Exception:
                    pass

        # Avaliação determinística do resultado com base na fase atingida e falhas ocorridas:

        # Caso A: Erro primário de E/S ocorrido durante write / flush / fsync
        if primary_io_error is not None:
            err_type, err_msg = primary_io_error
            extras = []
            if unlock_err is not None:
                extras.append(f"unlock_error: {unlock_err[1]}")
            if close_err is not None:
                extras.append(f"close_error: {close_err[1]}")
            if extras:
                err_msg = f"{err_msg} | Adicionais de cleanup: [ {'; '.join(extras)} ]"
            return QuarantineWriteResult(
                status="PERSISTENCE_UNCERTAIN",
                success=False,
                file_path=str(target_file),
                error_type=err_type,
                error_message=err_msg,
            )

        # Caso B: Erro comprovadamente anterior à escrita (lock, truncamento pré-existente, inspeção, seek)
        if pre_write_error is not None:
            err_type, err_msg = pre_write_error
            extras = []
            if unlock_err is not None:
                extras.append(f"unlock_error: {unlock_err[1]}")
            if close_err is not None:
                extras.append(f"close_error: {close_err[1]}")
            if extras:
                err_msg = f"{err_msg} | Adicionais de cleanup: [ {'; '.join(extras)} ]"
            return QuarantineWriteResult(
                status="FAILED_BEFORE_WRITE",
                success=False,
                file_path=str(target_file),
                error_type=err_type,
                error_message=err_msg,
            )

        # Caso C: Falha durante unlock (sem erro primário nem pré-escrita)
        if unlock_err is not None:
            u_type, u_msg = unlock_err
            if close_err is not None:
                u_msg = f"{u_msg} | Adicional: {close_err[1]}"
            if phase in (_PHASE_WRITE_STARTED, _PHASE_WRITE_DONE, _PHASE_FLUSH_DONE, _PHASE_FSYNC_DONE):
                return QuarantineWriteResult(
                    status="PERSISTENCE_UNCERTAIN",
                    success=False,
                    file_path=str(target_file),
                    error_type="UNLOCK_FAILURE",
                    error_message=f"Falha de unlock pós-escrita: {u_msg}",
                )
            else:
                return QuarantineWriteResult(
                    status="FAILED_BEFORE_WRITE",
                    success=False,
                    file_path=str(target_file),
                    error_type="UNLOCK_FAILURE",
                    error_message=f"Falha de unlock pré-escrita: {u_msg}",
                )

        # Caso D: Falha durante close() (sem erro primário, pré-escrita ou unlock)
        if close_err is not None:
            c_type, c_msg = close_err
            if phase in (_PHASE_WRITE_STARTED, _PHASE_WRITE_DONE, _PHASE_FLUSH_DONE, _PHASE_FSYNC_DONE):
                return QuarantineWriteResult(
                    status="PERSISTENCE_UNCERTAIN",
                    success=False,
                    file_path=str(target_file),
                    error_type="CLOSE_FAILURE",
                    error_message=f"Falha durante close pós-escrita/fsync: {c_msg}",
                )
            else:
                return QuarantineWriteResult(
                    status="FAILED_BEFORE_WRITE",
                    success=False,
                    file_path=str(target_file),
                    error_type="CLOSE_FAILURE",
                    error_message=f"Falha durante close pré-escrita: {c_msg}",
                )

        # Caso E: Ciclo concluído integralmente com sucesso
        if (
            phase == _PHASE_CLOSE_DONE
            and primary_io_error is None
            and pre_write_error is None
            and unlock_err is None
            and close_err is None
        ):
            self.total_quarantined_count += 1
            if cat_clean == "INVALID":
                self.quarantined_invalid_count += 1
            elif cat_clean == "COLLISION":
                self.quarantined_collision_count += 1
            elif cat_clean == "UNCLASSIFIED":
                self.quarantined_unclassified_count += 1
            return QuarantineWriteResult(
                status="CONFIRMED",
                success=True,
                file_path=str(target_file),
            )

        # Fallback defensivo: se não atingiu CLOSE_DONE e não houve erro capturado
        return QuarantineWriteResult(
            status="PERSISTENCE_UNCERTAIN" if phase != _PHASE_BEFORE_WRITE else "FAILED_BEFORE_WRITE",
            success=False,
            file_path=str(target_file),
            error_type="INCOMPLETE_IO_LIFECYCLE",
            error_message=f"Ciclo de E/S incompleto: fase final={phase}",
        )

    def get_quarantine_metrics(self) -> Dict[str, int]:
        """Retorna telemetria detalhada de eventos quarentenados e falhas (Gate G5)."""
        return {
            "quarantined_invalid_count": self.quarantined_invalid_count,
            "quarantined_collision_count": self.quarantined_collision_count,
            "quarantined_unclassified_count": self.quarantined_unclassified_count,
            "total_quarantined_count": self.total_quarantined_count,
            "enospc_error_count": self.enospc_error_count,
        }

    def get_financial_integrity_metrics(self) -> Dict[str, int]:
        """Alias para get_quarantine_metrics em conformidade com o contrato de telemetria G5."""
        return self.get_quarantine_metrics()
