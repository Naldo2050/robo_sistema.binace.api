"""
Módulo de Reconciliação e Auditoria de Eventos Fallback (Gate G4-D - Parte A).

Implementa diagnóstico estritamente offline e read-only (SOMENTE dry-run) para
análise de consistência e identidade entre:
1. Banco de dados relacional SQLite (dados/trading_bot.db).
2. Arquivo sequencial JSONL principal (dados/eventos_fluxo.jsonl).
3. Segmentos arquivados não-destrutivos (dados/archive/eventos_fluxo_*.jsonl).
4. Arquivos de fallback (fallback_events/eventos_*.jsonl ou .json).

Garante que nenhuma operação de escrita, recuperação ou alteração de arquivos
seja realizada nesta fase.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    from common.financial_quarantine import FinancialQuarantine
    HAS_FINANCIAL_QUARANTINE = True
except ImportError:
    FinancialQuarantine = None
    HAS_FINANCIAL_QUARANTINE = False

try:
    from common.financial_validator import (
        FinancialValidationResult,
        validate_financial_event_semantics,
    )
    HAS_FINANCIAL_VALIDATOR = True
except ImportError:
    HAS_FINANCIAL_VALIDATOR = False
    FinancialValidationResult = None
    validate_financial_event_semantics = None

try:
    from common.identity_contract import (
        IdentityContractError,
        canonicalize_structure,
        compute_producer_identity,
    )
    HAS_IDENTITY_CONTRACT = True
except ImportError:
    HAS_IDENTITY_CONTRACT = False
    IdentityContractError = Exception
    canonicalize_structure = lambda x: x
    compute_producer_identity = None

logger = logging.getLogger("events.fallback_reconciler")

RECONCILIATION_AUDIT_FILENAME = ".reconciliation_audit.jsonl"

# Campos excluídos do hash canônico de payload.
# Apenas campos transitórios/voláteis gerados pela infraestrutura de persistência
# e transporte são ignorados. Nenhum campo semântico de trading, mercado ou IA é removido.
EXCLUDED_CANONICAL_FIELDS: Set[str] = {
    "_log_id",          # ID de debug injetado no log visual
    "_source_db_id",    # ID de linha injetado por scripts de leitura SQLite
    "id",               # Chave primária autoincrement do SQLite
    "created_at",       # Timestamp do banco de dados (DATETIME DEFAULT CURRENT_TIMESTAMP)
}

# Arquivos rastreados no repositório utilizados como fixtures estáticas de teste
KNOWN_TEST_FIXTURE_NAMES: Set[str] = {
    "eventos_20260307.json",
}


class EventClassification(str, Enum):
    """Classificação rigorosa do estado de reconciliação de cada evento."""
    EVENTO_JA_PRESENTE = "EVENTO_JA_PRESENTE"
    EVENTO_AUSENTE = "EVENTO_AUSENTE"
    EVENTO_SEM_ID = "EVENTO_SEM_ID"
    EVENTO_ID_COLISAO = "EVENTO_ID_COLISAO"
    EVENTO_IDENTIDADE_AMBIGUA = "EVENTO_IDENTIDADE_AMBIGUA"
    EVENTO_INVALIDO = "EVENTO_INVALIDO"
    FIXTURE_BLOQUEADA = "FIXTURE_BLOQUEADA"


@dataclass
class EventReconciliationRecord:
    """Registro individual de análise de um evento no fallback."""
    source_file: str
    line_number: int
    classification: EventClassification
    event_id: Optional[str]
    epoch_ms: Optional[int]
    tipo_evento: Optional[str]
    symbol: Optional[str]
    canonical_hash: Optional[str]
    reason: str
    in_sqlite: bool = False
    in_jsonl: bool = False
    in_archive: bool = False
    divergence_details: Optional[str] = None
    validation_result: Optional[Any] = None


@dataclass
class DryRunReport:
    """Relatório estruturado de diagnóstico gerado em modo somente dry-run."""
    files_analyzed: List[str] = field(default_factory=list)
    total_records_read: int = 0
    counts: Dict[str, int] = field(default_factory=lambda: {c.value: 0 for c in EventClassification})
    records: List[EventReconciliationRecord] = field(default_factory=list)
    fixture_files_detected: List[str] = field(default_factory=list)
    hypothetical_operations: Dict[str, int] = field(default_factory=lambda: {
        "sqlite_inserts_needed": 0,
        "jsonl_appends_needed": 0,
        "no_action_needed": 0,
        "quarantine_needed": 0,
    })
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converte o relatório para dicionário serializável em JSON."""
        return {
            "mode": "DRY_RUN_READ_ONLY",
            "files_analyzed": self.files_analyzed,
            "total_records_read": self.total_records_read,
            "counts": self.counts,
            "hypothetical_operations": self.hypothetical_operations,
            "fixture_files_detected": self.fixture_files_detected,
            "errors": self.errors,
            "records_sample": [
                {
                    "source_file": r.source_file,
                    "line": r.line_number,
                    "classification": r.classification.value,
                    "event_id": r.event_id,
                    "epoch_ms": r.epoch_ms,
                    "tipo_evento": r.tipo_evento,
                    "symbol": r.symbol,
                    "reason": r.reason,
                    "in_sqlite": r.in_sqlite,
                    "in_jsonl": r.in_jsonl,
                    "in_archive": r.in_archive,
                }
                for r in self.records[:50]  # Amostra defensiva para evitar payloads gigantes
            ],
        }


def normalize_json_value(val: Any) -> Any:
    """
    Normaliza recursivamente estruturas de dados para serialização determinística,
    preservando estritamente a fidelidade numérica dos dados originais sem arredondamento.

    Política Numérica de Identidade Forte:
    - Sem tolerância numérica artificial: round(val, 8) removido integralmente.
    - Preserva distinções exatas de tipos e precisão:
        * int (ex: 1) -> serializado como número inteiro nativo (1)
        * float (ex: 1.0) -> serializado como ponto flutuante nativo (1.0)
        * float de alta precisão (ex: 1.000000001) -> preservado sem truncamento
        * str (ex: '50000.123456789') -> preservado como string monetária exata
    - Tratamento explícito de casos especiais de float:
        * math.isnan(val) -> normalizado estavelmente como '__NaN__'
        * math.isinf(val) positivo -> normalizado estavelmente como '__Infinity__'
        * math.isinf(val) negativo -> normalizado estavelmente como '__-Infinity__'
    - Chaves de dicionários são ordenadas lexicograficamente de forma recursiva.
    - Metadados transitórios de infraestrutura (EXCLUDED_CANONICAL_FIELDS) são excluídos.
    """
    if isinstance(val, dict):
        return {k: normalize_json_value(v) for k, v in sorted(val.items()) if k not in EXCLUDED_CANONICAL_FIELDS}
    elif isinstance(val, (list, tuple)):
        return [normalize_json_value(x) for x in val]
    elif isinstance(val, bool):
        # Em Python, bool é subclasse de int. Deve ser tratado antes de int.
        return val
    elif isinstance(val, float):
        if math.isnan(val):
            return "__NaN__"
        if math.isinf(val):
            return "__-Infinity__" if val < 0 else "__Infinity__"
        return val
    elif isinstance(val, (int, str)):
        return val
    return val


def compute_canonical_payload_hash(payload: Dict[str, Any]) -> str:
    """
    Calcula hash SHA256 canônico e estável do payload de um evento.

    Ordena todas as chaves recursivamente, remove metadados voláteis
    de transporte e serializa com separadores compactos padronizados.
    """
    clean_dict = normalize_json_value(payload)
    canonical_json = json.dumps(clean_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def validate_reconciliation_financial_payload(event: Dict[str, Any]) -> Optional[Any]:
    """
    Valida um payload de evento de fallback semanticamente contra regras financeiras puras,
    limites de microestrutura e contratos de identidade institucional (Gate H1-B2-Q4).
    """
    if not isinstance(event, dict):
        if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
            return FinancialValidationResult(
                is_valid=False,
                status="INVALID",
                category="TYPE_ERROR",
                rejection_reason="EVENT_MUST_BE_DICT",
            )
        return None

    # 1. Varredura recursiva de floats não-finitos (NaN, ±Infinity)
    def _check_non_finite(obj: Any, path: str = "") -> Optional[Tuple[str, str, str]]:
        if isinstance(obj, float):
            if math.isnan(obj):
                return (path or "float", "nan", "NAN_IN_FINANCIAL_PAYLOAD")
            if math.isinf(obj):
                return (path or "float", str(obj), "INFINITY_IN_FINANCIAL_PAYLOAD")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else str(k)
                res = _check_non_finite(v, p)
                if res is not None:
                    return res
        elif isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                p = f"{path}[{idx}]"
                res = _check_non_finite(item, p)
                if res is not None:
                    return res
        return None

    non_fin = _check_non_finite(event)
    if non_fin is not None:
        fld, val_repr, reason = non_fin
        if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
            return FinancialValidationResult(
                is_valid=False,
                status="INVALID",
                category="NON_FINITE_FLOAT",
                rejected_field=fld,
                rejected_value_repr=val_repr,
                rejection_reason=reason,
            )

    # 2. Rejeição de booleanos em campos numéricos/monetários
    numeric_monetary_fields = (
        "price", "preco", "volume", "pnl", "size", "amount",
        "top", "bottom", "vah", "poc", "val", "imbalance_ratio",
        "delta", "delta_btc", "delta_fechamento", "anomaly_score",
        "cluster_delta", "score", "weight", "taxa", "funding_rate",
    )
    for num_key in numeric_monetary_fields:
        if num_key in event:
            v = event[num_key]
            if isinstance(v, bool):
                if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category="TYPE_VIOLATION",
                        rejected_field=num_key,
                        rejected_value_repr=str(v),
                        rejection_reason="BOOLEAN_NOT_ALLOWED_IN_NUMERIC_FIELD",
                    )

    # 3. Preços não-positivos (price <= 0)
    for price_key in ("price", "preco"):
        if price_key in event:
            pv = event[price_key]
            if not isinstance(pv, bool) and isinstance(pv, (int, float)):
                if pv <= 0:
                    if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                        return FinancialValidationResult(
                            is_valid=False,
                            status="INVALID",
                            category="INVALID_PRICE",
                            rejected_field=price_key,
                            rejected_value_repr=str(pv),
                            rejection_reason="PRICE_MUST_BE_POSITIVE",
                        )

    # 4. Limites de microestrutura física: Zonas (bottom <= top)
    for b_key in ("top", "bottom"):
        if b_key in event:
            bv = event[b_key]
            if not isinstance(bv, bool) and isinstance(bv, (int, float)):
                if bv <= 0:
                    if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                        return FinancialValidationResult(
                            is_valid=False,
                            status="INVALID",
                            category="ZONE_LIMITS",
                            rejected_field=b_key,
                            rejected_value_repr=str(bv),
                            rejection_reason="ZONE_BOUNDS_MUST_BE_POSITIVE",
                        )
    if "top" in event and "bottom" in event:
        try:
            top_v = float(event["top"])
            bot_v = float(event["bottom"])
            if not (math.isnan(top_v) or math.isnan(bot_v) or math.isinf(top_v) or math.isinf(bot_v)):
                if bot_v > top_v:
                    if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                        return FinancialValidationResult(
                            is_valid=False,
                            status="INVALID",
                            category="ZONE_LIMITS",
                            rejected_field="bottom",
                            rejected_value_repr=f"bottom={bot_v} > top={top_v}",
                            rejection_reason="INVALID_MICROSTRUCTURE_BOUNDS: bottom > top",
                        )
        except (ValueError, TypeError):
            pass

    # 5. Limites de microestrutura física: Market Profile (val <= poc <= vah)
    if "poc" in event and "vah" in event:
        try:
            poc_v = float(event["poc"])
            vah_v = float(event["vah"])
            if not (math.isnan(poc_v) or math.isnan(vah_v) or math.isinf(poc_v) or math.isinf(vah_v)):
                if poc_v > vah_v:
                    if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                        return FinancialValidationResult(
                            is_valid=False,
                            status="INVALID",
                            category="MARKET_PROFILE",
                            rejected_field="poc",
                            rejected_value_repr=f"poc={poc_v} > vah={vah_v}",
                            rejection_reason="INVALID_MICROSTRUCTURE_BOUNDS: poc > vah",
                        )
        except (ValueError, TypeError):
            pass
    if "val" in event and "poc" in event:
        try:
            val_v = float(event["val"])
            poc_v = float(event["poc"])
            if not (math.isnan(val_v) or math.isnan(poc_v) or math.isinf(val_v) or math.isinf(poc_v)):
                if val_v > poc_v:
                    if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                        return FinancialValidationResult(
                            is_valid=False,
                            status="INVALID",
                            category="MARKET_PROFILE",
                            rejected_field="val",
                            rejected_value_repr=f"val={val_v} > poc={poc_v}",
                            rejection_reason="INVALID_MICROSTRUCTURE_BOUNDS: val > poc",
                        )
        except (ValueError, TypeError):
            pass

    # 6. Validação semântica pura contra regras de eventos
    if HAS_FINANCIAL_VALIDATOR and validate_financial_event_semantics is not None:
        try:
            val_res = validate_financial_event_semantics(event)
            if val_res.status == "INVALID":
                return val_res
        except Exception as sem_err:
            if FinancialValidationResult is not None:
                return FinancialValidationResult(
                    is_valid=False,
                    status="INVALID",
                    category="SEMANTIC_ERROR",
                    rejection_reason=str(sem_err),
                )

    # 7. Validação de contrato de identidade institucional (se aplicável)
    tipo = event.get("tipo_evento")
    if tipo in ("ABSORPTION_V1", "EXHAUSTION_V2", "TRIGGER_V1"):
        if HAS_IDENTITY_CONTRACT and compute_producer_identity is not None:
            try:
                compute_producer_identity(event)
            except IdentityContractError as id_err:
                if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category="MALFORMED_IDENTITY",
                        rejection_reason=f"IdentityContractError: {id_err}",
                    )
            except Exception:
                pass
    elif tipo == "ZONE_V2" and any(k in event for k in ("anchor_price", "centro", "kind", "tipo_zona")):
        if HAS_IDENTITY_CONTRACT and compute_producer_identity is not None:
            try:
                compute_producer_identity(event)
            except IdentityContractError as id_err:
                if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                    return FinancialValidationResult(
                        is_valid=False,
                        status="INVALID",
                        category="MALFORMED_IDENTITY",
                        rejection_reason=f"IdentityContractError: {id_err}",
                    )
            except Exception:
                pass

    if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
        return FinancialValidationResult(is_valid=True, status="VALID", category="VALID")
    return None


class FallbackReconciler:
    """
    Reconciliador e auditor de integridade para eventos de fallback.

    Opera estritamente em modo offline e somente leitura (Dry-Run),
    sem permissão de escrita em nenhuma base de dados ou arquivo.
    """

    def __init__(
        self,
        sqlite_path: Path = Path("dados/trading_bot.db"),
        jsonl_path: Path = Path("dados/eventos_fluxo.jsonl"),
        archive_dir: Path = Path("dados/archive"),
        repo_root: Optional[Path] = None,
        quarantine: Optional[Any] = None,
        quarantine_dir: Optional[Union[Path, str]] = None,
    ):
        self.sqlite_path = Path(sqlite_path)
        self.jsonl_path = Path(jsonl_path)
        self.archive_dir = Path(archive_dir)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self._quarantine = quarantine
        self._quarantine_dir = Path(quarantine_dir) if quarantine_dir is not None else None

    @property
    def quarantine(self) -> Optional[Any]:
        """Retorna instância de FinancialQuarantine instanciada sob demanda (lazy)."""
        if self._quarantine is None and self._quarantine_dir is not None:
            if HAS_FINANCIAL_QUARANTINE and FinancialQuarantine is not None:
                self._quarantine = FinancialQuarantine(quarantine_dir=self._quarantine_dir)
        return self._quarantine

    def is_git_tracked_fixture(self, file_path: Path) -> bool:
        """
        Determina se o arquivo de fallback é uma fixture de teste versionada pelo Git.
        """
        file_path_resolved = file_path.resolve()

        # 1. Checagem por nome conhecido
        if file_path.name in KNOWN_TEST_FIXTURE_NAMES:
            return True

        # 2. Checagem ativa via Git
        try:
            rel_path = file_path.relative_to(self.repo_root)
        except ValueError:
            rel_path = file_path

        try:
            res = subprocess.run(
                ["git", "ls-files", "--error-unmatch", str(rel_path)],
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return True
        except Exception:
            pass

        return False

    def load_sqlite_catalog(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[int, str, str], List[Dict[str, Any]]]]:
        """
        Lê e cataloga o banco SQLite de forma estritamente READ-ONLY.

        Retorna:
            by_event_id: {event_id: {id, timestamp_ms, event_type, symbol, canonical_hash, payload}}
            by_correlation: {(timestamp_ms, event_type, symbol): [registros]}
        """
        by_event_id: Dict[str, Dict[str, Any]] = {}
        by_correlation: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}

        if not self.sqlite_path.exists():
            logger.info("Banco SQLite não encontrado em %s (considerando vazio)", self.sqlite_path)
            return by_event_id, by_correlation

        # Abrir conexão SQLite estritamente em modo URI read-only (?mode=ro)
        uri = f"file:{self.sqlite_path.resolve().as_posix()}?mode=ro"
        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=10.0)
            cur = conn.cursor()

            # Verificação estrita do schema da tabela 'events'
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
            if not cur.fetchone():
                logger.warning("Tabela 'events' não existe em %s", self.sqlite_path)
                return by_event_id, by_correlation

            # Confirmação de colunas existentes
            cur.execute("PRAGMA table_info(events)")
            columns = {col[1] for col in cur.fetchall()}
            if not {"timestamp_ms", "event_type", "payload"}.issubset(columns):
                raise ValueError(f"Schema incompatível na tabela 'events': colunas={columns}")

            # Leitura seletiva de eventos
            cur.execute("""
                SELECT id, timestamp_ms, event_type, COALESCE(symbol, ''), payload
                FROM events
            """)

            for row_id, ts_ms, ev_type, sym, payload_raw in cur.fetchall():
                try:
                    payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
                    if not isinstance(payload, dict):
                        payload = {}
                except Exception:
                    payload = {}

                eid = payload.get("event_id") or payload.get("id")
                # Converter para string consistente
                eid_str = str(eid) if eid is not None else None
                ts_int = int(ts_ms) if ts_ms is not None else 0
                ev_type_str = str(ev_type or "")
                sym_str = str(sym or "")
                canon_hash = compute_canonical_payload_hash(payload) if payload else ""

                entry = {
                    "row_id": row_id,
                    "timestamp_ms": ts_int,
                    "event_type": ev_type_str,
                    "symbol": sym_str,
                    "canonical_hash": canon_hash,
                    "event_id": eid_str,
                }

                if eid_str:
                    by_event_id[eid_str] = entry

                corr_key = (ts_int, ev_type_str, sym_str)
                by_correlation.setdefault(corr_key, []).append(entry)

        except Exception as e:
            logger.error("Erro ao ler catálogo do SQLite (%s): %s", self.sqlite_path, e)
            raise
        finally:
            if conn:
                conn.close()

        return by_event_id, by_correlation

    def load_jsonl_catalog(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[int, str, str], List[Dict[str, Any]]]]:
        """
        Lê e cataloga o JSONL principal e todos os segmentos arquivados (G4-C).

        Retorna:
            by_event_id: {event_id: {source_file, line_num, epoch_ms, tipo_evento, symbol, canonical_hash, is_archive}}
            by_correlation: {(epoch_ms, tipo_evento, symbol): [registros]}
        """
        by_event_id: Dict[str, Dict[str, Any]] = {}
        by_correlation: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}

        # 1. Segmentos históricos do archive
        files_to_read: List[Tuple[Path, bool]] = []
        if self.archive_dir.exists():
            for arch_file in sorted(self.archive_dir.glob("eventos_fluxo_*.jsonl")):
                files_to_read.append((arch_file, True))

        # 2. Arquivo ativo
        if self.jsonl_path.exists():
            files_to_read.append((self.jsonl_path, False))

        for file_path, is_arch in files_to_read:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    line_num = 0
                    for line in f:
                        line_num += 1
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            data = json.loads(s)
                            if not isinstance(data, dict):
                                continue
                        except Exception:
                            continue

                        eid = data.get("event_id") or data.get("id")
                        eid_str = str(eid) if eid is not None else None
                        ep = data.get("epoch_ms") or data.get("timestamp_ms")
                        ep_int = int(ep) if ep is not None and str(ep).isdigit() else 0
                        ev_type = str(data.get("tipo_evento") or data.get("event_type") or data.get("type") or "")
                        sym = str(data.get("symbol") or data.get("ativo") or "")
                        canon_hash = compute_canonical_payload_hash(data)

                        entry = {
                            "source_file": str(file_path),
                            "line_num": line_num,
                            "epoch_ms": ep_int,
                            "tipo_evento": ev_type,
                            "symbol": sym,
                            "canonical_hash": canon_hash,
                            "is_archive": is_arch,
                            "event_id": eid_str,
                        }

                        if eid_str:
                            by_event_id[eid_str] = entry

                        corr_key = (ep_int, ev_type, sym)
                        by_correlation.setdefault(corr_key, []).append(entry)
            except Exception as e:
                logger.error("Erro ao ler JSONL %s: %s", file_path, e)

        return by_event_id, by_correlation

    def classify_event(
        self,
        event: Optional[Dict[str, Any]],
        source_file: str,
        line_num: int,
        is_fixture: bool,
        raw_error: Optional[str],
        sqlite_catalog_by_id: Dict[str, Dict[str, Any]],
        sqlite_catalog_by_corr: Dict[Tuple[int, str, str], List[Dict[str, Any]]],
        jsonl_catalog_by_id: Dict[str, Dict[str, Any]],
        jsonl_catalog_by_corr: Dict[Tuple[int, str, str], List[Dict[str, Any]]],
    ) -> EventReconciliationRecord:
        """
        Aplica a máquina de estados rigorosa de classificação para um único evento.
        """
        # Caso 1: Arquivo é fixture de teste versionada
        if is_fixture:
            eid = event.get("event_id") if isinstance(event, dict) else None
            return EventReconciliationRecord(
                source_file=source_file,
                line_number=line_num,
                classification=EventClassification.FIXTURE_BLOQUEADA,
                event_id=str(eid) if eid is not None else None,
                epoch_ms=event.get("epoch_ms") if isinstance(event, dict) else None,
                tipo_evento=event.get("tipo_evento") if isinstance(event, dict) else None,
                symbol=event.get("symbol") or event.get("ativo") if isinstance(event, dict) else None,
                canonical_hash=compute_canonical_payload_hash(event) if isinstance(event, dict) else None,
                reason="Arquivo é fixture de teste versionada pelo Git. Importação bloqueada por segurança.",
            )

        # Caso 2: Linha corrompida / JSON inválido
        if raw_error or event is None:
            return EventReconciliationRecord(
                source_file=source_file,
                line_number=line_num,
                classification=EventClassification.EVENTO_INVALIDO,
                event_id=None,
                epoch_ms=None,
                tipo_evento=None,
                symbol=None,
                canonical_hash=None,
                reason=f"JSON inválido ou linha truncada: {raw_error}",
            )

        # Extração de dados essenciais
        raw_eid = event.get("event_id") or event.get("id")
        event_id = str(raw_eid) if raw_eid is not None and str(raw_eid).strip() else None

        raw_epoch = event.get("epoch_ms") or event.get("timestamp_ms")
        epoch_ms = int(raw_epoch) if raw_epoch is not None and str(raw_epoch).isdigit() else None

        tipo_evento = event.get("tipo_evento") or event.get("event_type") or event.get("type")
        tipo_evento_str = str(tipo_evento) if tipo_evento else None

        symbol = event.get("symbol") or event.get("ativo")
        symbol_str = str(symbol) if symbol else None

        # Caso 3: Ausência de event_id
        if not event_id:
            return EventReconciliationRecord(
                source_file=source_file,
                line_number=line_num,
                classification=EventClassification.EVENTO_SEM_ID,
                event_id=None,
                epoch_ms=epoch_ms,
                tipo_evento=tipo_evento_str,
                symbol=symbol_str,
                canonical_hash=compute_canonical_payload_hash(event),
                reason="Evento não possui identificador (event_id ausente ou vazio).",
            )

        canonical_hash = compute_canonical_payload_hash(event)

        # Validação financeira semântica pura e de microestrutura (Gate H1-B2-Q4)
        val_res = validate_reconciliation_financial_payload(event)
        if val_res is not None and not getattr(val_res, "is_valid", True):
            rej_field = getattr(val_res, "rejected_field", "")
            rej_val = getattr(val_res, "rejected_value_repr", "")
            rej_reason = getattr(val_res, "rejection_reason", "Violação financeira/semântica")
            return EventReconciliationRecord(
                source_file=source_file,
                line_number=line_num,
                classification=EventClassification.EVENTO_INVALIDO,
                event_id=event_id,
                epoch_ms=epoch_ms,
                tipo_evento=tipo_evento_str,
                symbol=symbol_str,
                canonical_hash=canonical_hash,
                reason=f"Violação financeira/semântica: {rej_reason} (campo: {rej_field}, valor: {rej_val})",
                divergence_details=rej_reason,
                validation_result=val_res,
            )

        # Verificação cruzada nos catálogos de destino
        sq_match = sqlite_catalog_by_id.get(event_id)
        jl_match = jsonl_catalog_by_id.get(event_id)

        in_sqlite = False
        in_jsonl = False
        in_archive = False

        if sq_match:
            # Confere se o payload canônico bate ou se colidiu
            if sq_match["canonical_hash"] != canonical_hash:
                return EventReconciliationRecord(
                    source_file=source_file,
                    line_number=line_num,
                    classification=EventClassification.EVENTO_ID_COLISAO,
                    event_id=event_id,
                    epoch_ms=epoch_ms,
                    tipo_evento=tipo_evento_str,
                    symbol=symbol_str,
                    canonical_hash=canonical_hash,
                    reason=f"Colisão de ID: event_id {event_id} já existe no SQLite com payload divergente.",
                    in_sqlite=True,
                    divergence_details=f"SQLite row_id={sq_match['row_id']}, hash={sq_match['canonical_hash']} vs fallback={canonical_hash}",
                )
            in_sqlite = True

        if jl_match:
            if jl_match["canonical_hash"] != canonical_hash:
                return EventReconciliationRecord(
                    source_file=source_file,
                    line_number=line_num,
                    classification=EventClassification.EVENTO_ID_COLISAO,
                    event_id=event_id,
                    epoch_ms=epoch_ms,
                    tipo_evento=tipo_evento_str,
                    symbol=symbol_str,
                    canonical_hash=canonical_hash,
                    reason=f"Colisão de ID: event_id {event_id} já existe no JSONL com payload divergente.",
                    in_jsonl=not jl_match["is_archive"],
                    in_archive=jl_match["is_archive"],
                    divergence_details=f"JSONL {jl_match['source_file']}:{jl_match['line_num']}, hash={jl_match['canonical_hash']} vs fallback={canonical_hash}",
                )
            if jl_match["is_archive"]:
                in_archive = True
            else:
                in_jsonl = True

        # Caso 4: Se o evento não foi achado por ID, verificar se colide por correlação temporal
        if not in_sqlite and not in_jsonl and not in_archive:
            corr_key = (epoch_ms or 0, tipo_evento_str or "", symbol_str or "")
            sq_corrs = sqlite_catalog_by_corr.get(corr_key, [])
            jl_corrs = jsonl_catalog_by_corr.get(corr_key, [])

            # Se existe evento com a mesma tupla exata mas ID diferente
            has_different_id_sqlite = any(c.get("event_id") != event_id for c in sq_corrs)
            has_different_id_jsonl = any(c.get("event_id") != event_id for c in jl_corrs)

            if has_different_id_sqlite or has_different_id_jsonl:
                return EventReconciliationRecord(
                    source_file=source_file,
                    line_number=line_num,
                    classification=EventClassification.EVENTO_IDENTIDADE_AMBIGUA,
                    event_id=event_id,
                    epoch_ms=epoch_ms,
                    tipo_evento=tipo_evento_str,
                    symbol=symbol_str,
                    canonical_hash=canonical_hash,
                    reason="Identidade ambígua: mesma tupla (epoch_ms, tipo, symbol) encontrada nos destinos com IDs diferentes.",
                    divergence_details=f"Ocorrências correlação: SQLite={len(sq_corrs)}, JSONL={len(jl_corrs)}",
                )

        # Caso 5: Totalmente presente em ambos os destinos
        if in_sqlite and (in_jsonl or in_archive):
            return EventReconciliationRecord(
                source_file=source_file,
                line_number=line_num,
                classification=EventClassification.EVENTO_JA_PRESENTE,
                event_id=event_id,
                epoch_ms=epoch_ms,
                tipo_evento=tipo_evento_str,
                symbol=symbol_str,
                canonical_hash=canonical_hash,
                in_sqlite=in_sqlite,
                in_jsonl=in_jsonl,
                in_archive=in_archive,
                reason="Evento já persistido identicamente em ambos os destinos (SQLite e JSONL/Archive).",
            )

        # Caso 6: Ausente em um ou ambos os destinos (elegível para reconciliação)
        missing_dest = []
        if not in_sqlite:
            missing_dest.append("SQLite")
        if not (in_jsonl or in_archive):
            missing_dest.append("JSONL")

        return EventReconciliationRecord(
            source_file=source_file,
            line_number=line_num,
            classification=EventClassification.EVENTO_AUSENTE,
            event_id=event_id,
            epoch_ms=epoch_ms,
            tipo_evento=tipo_evento_str,
            symbol=symbol_str,
            canonical_hash=canonical_hash,
            in_sqlite=in_sqlite,
            in_jsonl=in_jsonl,
            in_archive=in_archive,
            reason=f"Evento elegível ausente em: {', '.join(missing_dest)}.",
        )

    def run_dry_run(self, fallback_files: List[Path]) -> DryRunReport:
        """
        Executa auditoria diagnóstica completa dos arquivos de fallback em modo estritamente READ-ONLY.
        """
        report = DryRunReport()
        report.files_analyzed = [str(f) for f in fallback_files]

        # 1. Carregar catálogos de destino (somente leitura)
        try:
            sq_by_id, sq_by_corr = self.load_sqlite_catalog()
        except Exception as e:
            report.errors.append(f"Falha ao ler SQLite em modo read-only: {e}")
            return report

        try:
            jl_by_id, jl_by_corr = self.load_jsonl_catalog()
        except Exception as e:
            report.errors.append(f"Falha ao ler JSONL em modo read-only: {e}")
            return report

        # 2. Analisar cada arquivo de fallback
        for file_path in fallback_files:
            if not file_path.exists():
                report.errors.append(f"Arquivo não encontrado: {file_path}")
                continue

            is_fixture = self.is_git_tracked_fixture(file_path)
            if is_fixture:
                report.fixture_files_detected.append(str(file_path))

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    line_num = 0
                    for line in f:
                        line_num += 1
                        s = line.strip()
                        if not s:
                            continue

                        report.total_records_read += 1
                        event = None
                        raw_error = None
                        try:
                            event = json.loads(s)
                            if not isinstance(event, dict):
                                raw_error = "Conteúdo não é um objeto JSON válido"
                                event = None
                        except Exception as parse_err:
                            raw_error = str(parse_err)

                        rec = self.classify_event(
                            event=event,
                            source_file=str(file_path),
                            line_num=line_num,
                            is_fixture=is_fixture,
                            raw_error=raw_error,
                            sqlite_catalog_by_id=sq_by_id,
                            sqlite_catalog_by_corr=sq_by_corr,
                            jsonl_catalog_by_id=jl_by_id,
                            jsonl_catalog_by_corr=jl_by_corr,
                        )

                        report.records.append(rec)
                        report.counts[rec.classification.value] += 1

                        # Contabilização de operações hipotéticas
                        if rec.classification == EventClassification.EVENTO_JA_PRESENTE:
                            report.hypothetical_operations["no_action_needed"] += 1
                        elif rec.classification == EventClassification.EVENTO_AUSENTE:
                            if not rec.in_sqlite:
                                report.hypothetical_operations["sqlite_inserts_needed"] += 1
                            if not (rec.in_jsonl or rec.in_archive):
                                report.hypothetical_operations["jsonl_appends_needed"] += 1
                        elif rec.classification in (
                            EventClassification.EVENTO_ID_COLISAO,
                            EventClassification.EVENTO_IDENTIDADE_AMBIGUA,
                            EventClassification.EVENTO_SEM_ID,
                            EventClassification.EVENTO_INVALIDO,
                        ):
                            report.hypothetical_operations["quarantine_needed"] += 1
                        elif rec.classification == EventClassification.FIXTURE_BLOQUEADA:
                            report.hypothetical_operations["no_action_needed"] += 1

            except Exception as read_err:
                report.errors.append(f"Erro ao abrir arquivo {file_path}: {read_err}")

        return report


# ==============================================================================
# MOTOR DE RECONCILIAÇÃO COM ESCRITA (HERMÉTICO — FASE G4-D B1)
# Funções puras e isoladas para testes herméticos em diretórios temporários.
# Proibido o uso de caminhos operacionais como default.
# ==============================================================================

def create_sqlite_backup_consistent(source_sqlite_path: Path, backup_dest_path: Path) -> str:
    """
    Cria cópia transacionalmente consistente do SQLite via SQLite Backup API nativa.

    Lê páginas ativas do banco e do WAL e consolida em um único arquivo de destino.
    Valida integridade com PRAGMA integrity_check e calcula hash SHA256 do arquivo fechado.
    """
    source_sqlite_path = Path(source_sqlite_path)
    backup_dest_path = Path(backup_dest_path)

    if not source_sqlite_path.exists():
        raise FileNotFoundError(f"Banco SQLite de origem não encontrado: {source_sqlite_path}")

    backup_dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Abertura segura da origem em modo leitura e destino em nova conexão
    uri_source = f"file:{source_sqlite_path.resolve().as_posix()}?mode=ro"
    source_conn = sqlite3.connect(uri_source, uri=True, timeout=10.0)
    dest_conn = sqlite3.connect(str(backup_dest_path), timeout=10.0)

    try:
        # 2. Execução da Backup API oficial do SQLite
        source_conn.backup(dest_conn, pages=-1)
    finally:
        dest_conn.close()
        source_conn.close()

    # 3. Validação rigorosa pós-cópia: PRAGMA integrity_check
    check_conn = sqlite3.connect(str(backup_dest_path))
    try:
        cur = check_conn.cursor()
        cur.execute("PRAGMA integrity_check")
        row = cur.fetchone()
        if not row or row[0] != "ok":
            if backup_dest_path.exists():
                backup_dest_path.unlink()
            raise ValueError(f"Integridade do backup SQLite falhou: {row}")
    finally:
        check_conn.close()

    # 4. Cálculo SHA256 do arquivo gerado e fechado
    h = hashlib.sha256()
    with open(backup_dest_path, "rb") as bf:
        while chunk := bf.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_and_prepare_jsonl_target(jsonl_path: Path) -> str:
    """
    Valida o arquivo JSONL de destino e determina se é necessário prefixar um newline.

    Regras:
    - Se o arquivo não existir ou tiver 0 bytes: retorna '' (append direto).
    - Se a última linha terminar em newline e for JSON válido: retorna '' (append direto).
    - Se a última linha NÃO terminar em newline, mas for JSON válido: retorna '\n' (preserva dados e separa).
    - Se a última linha for JSON inválido (truncado): levanta ValueError e aborta sem modificar nada.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        return ""

    with open(jsonl_path, "rb") as f:
        content = f.read()

    if not content:
        return ""

    ends_with_newline = content.endswith(b"\n") or content.endswith(b"\r\n")
    lines = content.splitlines()
    if not lines:
        return ""

    last_line_bytes = lines[-1].strip()
    if not last_line_bytes:
        return ""

    # Validação sintática estrita da última linha
    try:
        last_line_str = last_line_bytes.decode("utf-8", errors="strict")
        json.loads(last_line_str)
    except Exception as err:
        raise ValueError(
            f"Última linha do JSONL ({jsonl_path}) é inválida ou foi corrompida/truncada: {err}. "
            f"Append abortado sem modificação de nenhum byte."
        )

    return "" if ends_with_newline else "\n"


def insert_event_to_sqlite(sqlite_path: Path, event: Dict[str, Any]) -> str:
    """
    Insere um único evento de forma atômica no SQLite dentro de transação BEGIN IMMEDIATE.

    Preserva integralmente o timestamp original.
    Verifica desduplicação pré-insert por timestamp_ms e hash canônico.
    """
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"Banco SQLite não encontrado em {sqlite_path}")

    # Canonicalização estrutural determinística (-0.0 -> +0.0)
    if HAS_IDENTITY_CONTRACT and canonicalize_structure is not None:
        event = canonicalize_structure(event)

    raw_eid = event.get("event_id") or event.get("id")
    if not raw_eid:
        raise ValueError("Evento não possui event_id válido para inserção.")
    event_id = str(raw_eid)

    raw_ts = event.get("epoch_ms") or event.get("timestamp_ms")
    if raw_ts is None:
        raise ValueError(f"Evento {event_id} não possui timestamp_ms ou epoch_ms.")

    try:
        ts_int = int(raw_ts)
        if not (946684800000 <= ts_int <= 4102444800000):
            raise ValueError(f"Timestamp fora do intervalo aceitável: {ts_int}")
    except (ValueError, TypeError) as ts_err:
        raise ValueError(f"Timestamp inválido no evento {event_id}: {ts_err}")

    canonical_hash = compute_canonical_payload_hash(event)
    ev_type = str(event.get("tipo_evento") or event.get("event_type") or event.get("type", "UNKNOWN"))
    sym = str(event.get("symbol") or event.get("ativo", ""))
    wid = str(event.get("window_id", ""))
    is_sig_val = event.get("is_signal")
    is_sig = 1 if (is_sig_val is True or str(is_sig_val).lower() == "true") else 0
    payload_str = json.dumps(event, ensure_ascii=False, default=str, separators=(",", ":"))

    conn = sqlite3.connect(str(sqlite_path), timeout=15.0)
    try:
        # 1. Consulta indexada prévia de desduplicação
        cur = conn.cursor()
        cur.execute(
            "SELECT id, payload FROM events WHERE timestamp_ms = ? AND event_type = ?",
            (ts_int, ev_type),
        )
        for row_id, existing_payload_raw in cur.fetchall():
            try:
                ex_payload = json.loads(existing_payload_raw)
                ex_eid = str(ex_payload.get("event_id") or ex_payload.get("id") or "")
                if ex_eid == event_id:
                    if compute_canonical_payload_hash(ex_payload) == canonical_hash:
                        return "ALREADY_PRESENT"
                    raise ValueError(f"Colisão de ID no SQLite: event_id {event_id} já existe com payload divergente.")
            except json.JSONDecodeError:
                pass

        # 2. Inserção transacional atômica com write lock antecipado
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO events (timestamp_ms, event_type, symbol, window_id, is_signal, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts_int, ev_type, sym, wid, is_sig, payload_str),
        )
        conn.commit()
        return "INSERTED"
    except Exception as commit_err:
        # Tratamento de status UNKNOWN: verifica se o commit gravou antes de propagar falha
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, payload FROM events WHERE timestamp_ms = ? AND event_type = ?",
                (ts_int, ev_type),
            )
            for _, existing_payload_raw in cur.fetchall():
                try:
                    ex_payload = json.loads(existing_payload_raw)
                    if str(ex_payload.get("event_id")) == event_id:
                        if compute_canonical_payload_hash(ex_payload) == canonical_hash:
                            return "INSERTED"
                except Exception:
                    pass
        except Exception:
            pass
        raise commit_err
    finally:
        conn.close()


def append_event_to_jsonl(jsonl_path: Path, event: Dict[str, Any], prefix_separator: str = "") -> None:
    """
    Anexa um único evento ao arquivo JSONL ativo aplicando flush() e os.fsync() mandatórios.
    """
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    json_line = json.dumps(event, ensure_ascii=False, default=str, separators=(",", ":"))
    with open(jsonl_path, "a", encoding="utf-8") as f:
        if prefix_separator:
            f.write(prefix_separator)
        f.write(json_line + "\n")
        f.flush()
        os.fsync(f.fileno())


def record_reconciliation_in_ledger(ledger_path: Path, entry: Dict[str, Any]) -> None:
    """
    Registra checkpoint de reconciliação no ledger de auditoria derivado com fsync.
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(entry, ensure_ascii=False, default=str, separators=(",", ":"))
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def verify_operational_lock_authorization(
    operational_lock: Any,
    target_sqlite_path: Path,
    target_jsonl_path: Path,
) -> None:
    """
    Valida a autorização física e de kernel para escrita em dados operacionais (Gate G4-D B3-B1).

    Exige lock operacional real, ativamente detido pelo processo atual, confinado
    canonicamente no mesmo diretório pai das bases de dados.
    """
    from common.runtime_lock import OperationalRuntimeLock

    if operational_lock is None or not isinstance(operational_lock, OperationalRuntimeLock):
        raise PermissionError(
            "Autorização negada: objeto de lock operacional ausente ou inválido. "
            "Acesso aos dados operacionais exige uma instância real de OperationalRuntimeLock."
        )

    if not operational_lock.is_locked or operational_lock._file_obj is None:
        raise PermissionError(
            "Autorização negada: o lock operacional não está ativamente detido pelo processo atual. "
            "Operação abortada para evitar colisão."
        )

    sqlite_resolved = Path(target_sqlite_path).expanduser().resolve()
    jsonl_resolved = Path(target_jsonl_path).expanduser().resolve()
    lock_resolved = operational_lock.lock_path.expanduser().resolve()

    norm_data_dir = os.path.normcase(str(sqlite_resolved.parent))
    norm_jsonl_dir = os.path.normcase(str(jsonl_resolved.parent))
    norm_lock_dir = os.path.normcase(str(lock_resolved.parent))

    # 1. Confinamento no mesmo diretório canônico
    if norm_data_dir != norm_lock_dir or norm_jsonl_dir != norm_lock_dir:
        raise PermissionError(
            f"Violação de confinamento: os dados operacionais ({norm_data_dir}) "
            f"não residem no mesmo diretório protegido pelo lock ({norm_lock_dir})."
        )

    # 2. Identidade exata do arquivo de lock
    expected_lock_file = os.path.normcase(str(sqlite_resolved.parent / ".operational_runtime.lock"))
    if os.path.normcase(str(lock_resolved)) != expected_lock_file:
        raise PermissionError(
            f"Identidade de lock incorreta: esperado '{expected_lock_file}', "
            f"fornecido '{lock_resolved}'."
        )


def verify_sqlite_integrity(sqlite_path: Path) -> Tuple[bool, str]:
    """
    Executa verificação diagnóstica offline de integridade estrutural e referencial no SQLite (Gate G4-D B3-B1).
    Zero checkpoint WAL, abre em modo read-only URI (?mode=ro).
    """
    p = Path(sqlite_path).expanduser().resolve()
    if not p.exists():
        return True, "Banco SQLite não existe (será criado na primeira operação)"

    uri = f"file:{p.as_posix()}?mode=ro"
    conn = None
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        cur = conn.cursor()

        # PRAGMA integrity_check
        cur.execute("PRAGMA integrity_check")
        row = cur.fetchone()
        if not row or row[0] != "ok":
            return False, f"PRAGMA integrity_check falhou: {row}"

        # PRAGMA foreign_key_check
        cur.execute("PRAGMA foreign_key_check")
        fk_rows = cur.fetchall()
        if fk_rows:
            return False, f"PRAGMA foreign_key_check encontrou violações: {fk_rows}"

        return True, "Integridade SQLite íntegra (integrity_check e FKs OK)"
    except Exception as e:
        return False, f"Erro ao verificar integridade do SQLite: {e}"
    finally:
        if conn:
            conn.close()


def verify_jsonl_integrity(jsonl_path: Path) -> Tuple[bool, str]:
    """
    Valida a terminação e integridade da última linha do arquivo JSONL ativo (Gate G4-D B3-B1).
    """
    p = Path(jsonl_path).expanduser().resolve()
    if not p.exists() or p.stat().st_size == 0:
        return True, "JSONL ativo inexistente ou vazio (pronto para append)"

    try:
        with open(p, "rb") as f:
            content = f.read()
        if not content:
            return True, "JSONL vazio"

        lines = content.splitlines()
        if not lines:
            return True, "JSONL sem linhas"

        last_line_bytes = lines[-1].strip()
        if not last_line_bytes:
            return True, "JSONL termina com linha em branco limpa"

        last_line_str = last_line_bytes.decode("utf-8", errors="strict")
        json.loads(last_line_str)
        return True, "JSONL ativo íntegro (última linha é JSON válido)"
    except Exception as e:
        return False, f"Última linha do JSONL é inválida ou corrompida: {e}"


def verify_archive_integrity(archive_dir: Path) -> Tuple[bool, str]:
    """
    Verifica ausência de arquivos temporários residuais (.tmp) e integridade dos segmentos no archive G4-C.
    Valida legibilidade e estrutura JSON linha a linha de todos os arquivos de segmento (.jsonl).
    """
    p = Path(archive_dir).expanduser().resolve()
    if not p.exists():
        return True, "Diretório de archive não existe ainda (nenhuma rotação pendente)"

    # 1. Verifica se há arquivos residuais .tmp
    tmp_files = list(p.glob("*.tmp"))
    if tmp_files:
        tmp_names = [f.name for f in tmp_files[:5]]
        return False, f"Arquivos temporários residuais detectados no archive (rotação incompleta): {tmp_names}"

    # 2. Varredura e validação de legibilidade e sintaxe de cada segmento .jsonl
    seg_files = sorted(p.glob("*.jsonl"))
    for seg in seg_files:
        if not os.access(seg, os.R_OK):
            return False, f"Segmento arquivado '{seg.name}' inacessível para leitura (permissão negada)"

        try:
            with open(seg, "r", encoding="utf-8", errors="strict") as f:
                for line_idx, line in enumerate(f, start=1):
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                        if not isinstance(obj, dict):
                            return False, f"Segmento arquivado '{seg.name}' possui linha {line_idx} que não é JSON dict"
                    except Exception as json_err:
                        return False, f"Segmento arquivado '{seg.name}' corrompido na linha {line_idx}: {json_err}"
        except Exception as read_err:
            return False, f"Falha de leitura no segmento arquivado '{seg.name}': {read_err}"

    return True, f"Diretório de archive G4-C limpo e íntegro ({len(seg_files)} segmentos validados)"


def verify_reconciliation_preflight(
    source_fallback_file: Path,
    target_sqlite_path: Path,
    target_jsonl_path: Path,
    archive_dir: Path,
    operational_lock: Optional[Any] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Orquestra a cadeia serial de verificações pré-apply (Gate G4-D B3-B1).
    Estritamente read-only. Retorna dicionário estruturado com o veredito de cada gate.
    """
    checks: Dict[str, Tuple[bool, str]] = {}
    errors: List[str] = []

    # 1. Validação de Fixtures versionadas no Git
    reconciler = FallbackReconciler(
        sqlite_path=target_sqlite_path,
        jsonl_path=target_jsonl_path,
        archive_dir=archive_dir,
        repo_root=repo_root,
    )
    if reconciler.is_git_tracked_fixture(source_fallback_file):
        msg = f"Arquivo '{source_fallback_file.name}' é uma fixture versionada no Git. Modificação proibida."
        checks["fixture_check"] = (False, msg)
        errors.append(msg)
    else:
        checks["fixture_check"] = (True, "Arquivo de fallback não é fixture rastreada no Git")

    # 2. Verificação de autorização de Lock Operacional
    try:
        verify_operational_lock_authorization(operational_lock, target_sqlite_path, target_jsonl_path)
        checks["lock_authorization"] = (True, "Lock operacional ativo, autêntico e confinado no mesmo diretório")
    except Exception as lock_err:
        checks["lock_authorization"] = (False, str(lock_err))
        errors.append(str(lock_err))

    # 3. Verificação de Integridade do SQLite
    sq_ok, sq_msg = verify_sqlite_integrity(target_sqlite_path)
    checks["sqlite_integrity"] = (sq_ok, sq_msg)
    if not sq_ok:
        errors.append(sq_msg)

    # 4. Verificação de Integridade do JSONL
    jl_ok, jl_msg = verify_jsonl_integrity(target_jsonl_path)
    checks["jsonl_integrity"] = (jl_ok, jl_msg)
    if not jl_ok:
        errors.append(jl_msg)

    # 5. Verificação de Integridade do Archive
    arch_ok, arch_msg = verify_archive_integrity(archive_dir)
    checks["archive_integrity"] = (arch_ok, arch_msg)
    if not arch_ok:
        errors.append(arch_msg)

    is_ready = len(errors) == 0
    return {
        "ready": is_ready,
        "checks": checks,
        "errors": errors,
    }


def reconcile_events_hermetic(
    source_fallback_file: Path,
    target_sqlite_path: Path,
    target_jsonl_path: Path,
    archive_dir: Path,
    ledger_path: Optional[Path] = None,
    backup_dir: Optional[Path] = None,
    perform_backup: bool = True,
    repo_root: Optional[Path] = None,
    operational_lock: Optional[Any] = None,
    quarantine: Optional[Any] = None,
    quarantine_dir: Optional[Union[Path, str]] = None,
    isolate_poison_pills: bool = False,
    purge_quarantined: bool = False,
) -> Dict[str, Any]:
    """
    Orquestrador hermético do motor de reconciliação com escrita para testes isolados (Gate G4-D B1/B3-B1 / H1-B2-Q4).

    Exige caminhos explícitos do chamador.
    Bloqueia caminhos de produção a menos que comprovada autorização ativa via OperationalRuntimeLock,
    preflight 100% aprovado e backup obrigatório bem-sucedido.
    Bloqueia arquivos de fixture versionados no Git incondicionalmente.
    Suporta quarentena forense lazy e isolamento de poison pills sem retries cegos.
    """
    source_fallback_file = Path(source_fallback_file)
    target_sqlite_path = Path(target_sqlite_path)
    target_jsonl_path = Path(target_jsonl_path)
    archive_dir = Path(archive_dir)

    active_quarantine = quarantine
    if active_quarantine is None and quarantine_dir is not None:
        if HAS_FINANCIAL_QUARANTINE and FinancialQuarantine is not None:
            active_quarantine = FinancialQuarantine(quarantine_dir=Path(quarantine_dir))

    # 1. Determinação canônica e segura da raiz do repositório (independente de cwd)
    try:
        if repo_root is None:
            canonical_repo = Path(__file__).resolve().parent.parent
        else:
            canonical_repo = Path(repo_root).resolve()
    except Exception as root_err:
        raise PermissionError(f"Fail-closed: impossível determinar raiz canônica do repositório: {root_err}")

    # 2. Resolução canônica dos caminhos operacionais protegidos e dos destinos
    try:
        target_sqlite_res = target_sqlite_path.resolve()
        target_jsonl_res = target_jsonl_path.resolve()
        prod_dir_res = (canonical_repo / "dados").resolve()
        prod_sqlite_res = (prod_dir_res / "trading_bot.db").resolve()
        prod_jsonl_res = (prod_dir_res / "eventos_fluxo.jsonl").resolve()

        norm_sqlite = os.path.normcase(str(target_sqlite_res))
        norm_jsonl = os.path.normcase(str(target_jsonl_res))
        norm_prod_dir = os.path.normcase(str(prod_dir_res))
        norm_prod_sqlite = os.path.normcase(str(prod_sqlite_res))
        norm_prod_jsonl = os.path.normcase(str(prod_jsonl_res))

        is_prod_target = (
            norm_sqlite == norm_prod_sqlite
            or norm_jsonl == norm_prod_jsonl
            or os.path.normcase(str(target_sqlite_res.parent)) == norm_prod_dir
            or os.path.normcase(str(target_jsonl_res.parent)) == norm_prod_dir
        )
        if not is_prod_target and target_sqlite_res.exists() and prod_sqlite_res.exists():
            if os.path.samefile(target_sqlite_res, prod_sqlite_res):
                is_prod_target = True
        if not is_prod_target and target_jsonl_res.exists() and prod_jsonl_res.exists():
            if os.path.samefile(target_jsonl_res, prod_jsonl_res):
                is_prod_target = True
    except Exception as id_err:
        raise PermissionError(f"Fail-closed: falha ao determinar identidade física dos recursos: {id_err}")

    # 3. Salvaguarda e autorização sob exclusividade operacional (Gate G4-D B3-B1)
    if is_prod_target:
        # A. Validação estrita de autorização de lock operacional
        verify_operational_lock_authorization(operational_lock, target_sqlite_path, target_jsonl_path)

        # B. Backup obrigatório: rejeita desativação externa e exige backup_dir válido
        if not perform_backup:
            raise PermissionError(
                "Operação proibida: desativação de backup em caminhos operacionais de produção "
                "é expressamente vedada por contrato de segurança."
            )
        if not backup_dir:
            raise ValueError(
                "Operação proibida: backup_dir é estritamente obrigatório para caminhos operacionais de produção."
            )

        # C. Preflight obrigatório antes de qualquer escrita
        preflight = verify_reconciliation_preflight(
            source_fallback_file=source_fallback_file,
            target_sqlite_path=target_sqlite_path,
            target_jsonl_path=target_jsonl_path,
            archive_dir=archive_dir,
            operational_lock=operational_lock,
            repo_root=canonical_repo,
        )
        if not preflight["ready"]:
            err_details = "; ".join(preflight["errors"])
            raise RuntimeError(
                f"Preflight de reconciliação falhou com impedimentos críticos: {err_details}. "
                f"Escrita em produção abortada antes de tocar em qualquer dado."
            )
    elif operational_lock is not None:
        # Em ambiente isolado/teste, se um lock foi fornecido, valida sua consistência
        verify_operational_lock_authorization(operational_lock, target_sqlite_path, target_jsonl_path)

    # 4. Salvaguarda incondicional contra modificação de fixtures de teste Git
    reconciler = FallbackReconciler(
        sqlite_path=target_sqlite_path,
        jsonl_path=target_jsonl_path,
        archive_dir=archive_dir,
        repo_root=canonical_repo,
        quarantine=active_quarantine,
    )
    if reconciler.is_git_tracked_fixture(source_fallback_file):
        raise PermissionError(
            f"Arquivo '{source_fallback_file.name}' é uma fixture de teste versionada pelo Git. "
            f"Modificação ou reconciliação com escrita é ESTRITAMENTE PROIBIDA."
        )

    # 5. Execução e validação do backup consistente (aborto seguro em caso de falha)
    backup_hash = None
    if perform_backup and backup_dir and target_sqlite_path.exists():
        backup_dir_path = Path(backup_dir)
        try:
            backup_dir_path.mkdir(parents=True, exist_ok=True)
            backup_file = backup_dir_path / f"{target_sqlite_path.stem}_backup.db"
            backup_hash = create_sqlite_backup_consistent(target_sqlite_path, backup_file)
        except Exception as bkp_err:
            raise RuntimeError(
                f"Falha crítica na criação do backup consistente em {backup_dir}: {bkp_err}. "
                f"Operação abortada antes de qualquer escrita."
            ) from bkp_err

    # 6. Validação da terminação do arquivo JSONL ativo
    prefix_sep = validate_and_prepare_jsonl_target(target_jsonl_path)

    # 7. Leitura dos catálogos dos destinos
    sq_by_id, sq_by_corr = reconciler.load_sqlite_catalog()
    jl_by_id, jl_by_corr = reconciler.load_jsonl_catalog()

    results = {
        "sqlite_inserted": 0,
        "jsonl_appended": 0,
        "already_present": 0,
        "skipped_quarantine": 0,
        "backup_sha256": backup_hash,
        "reconciled_ids": [],
        "quarantined_ids": [],
        "quarantined_count": 0,
        "unresolved_count": 0,
        "enospc_error_count": 0,
    }

    raw_lines_to_keep: List[str] = []
    has_quarantined_or_poison = False

    # 8. Processamento do arquivo de fallback
    with open(source_fallback_file, "r", encoding="utf-8", errors="replace") as f:
        line_num = 0
        for line in f:
            line_num += 1
            s = line.strip()
            if not s:
                continue

            event = None
            raw_err = None
            try:
                event = json.loads(s)
                if not isinstance(event, dict):
                    raw_err = "Não é dict"
            except Exception as pe:
                raw_err = str(pe)

            raw_event_copy = copy.deepcopy(event) if isinstance(event, dict) else None

            rec = reconciler.classify_event(
                event=event,
                source_file=str(source_fallback_file),
                line_num=line_num,
                is_fixture=False,
                raw_error=raw_err,
                sqlite_catalog_by_id=sq_by_id,
                sqlite_catalog_by_corr=sq_by_corr,
                jsonl_catalog_by_id=jl_by_id,
                jsonl_catalog_by_corr=jl_by_corr,
            )

            if rec.classification == EventClassification.EVENTO_JA_PRESENTE:
                results["already_present"] += 1
            elif rec.classification == EventClassification.EVENTO_AUSENTE:
                # Revalidação de exclusividade contínua antes de operações de escrita
                if operational_lock is not None and not operational_lock.is_locked:
                    raise PermissionError("Lock operacional foi liberado ou perdido durante a reconciliação!")

                eid = rec.event_id
                inserted_sqlite = False
                appended_jsonl = False
                event_to_persist = canonicalize_structure(event) if isinstance(event, dict) else event

                try:
                    if not rec.in_sqlite:
                        res_sq = insert_event_to_sqlite(target_sqlite_path, event_to_persist)
                        if res_sq == "INSERTED":
                            results["sqlite_inserted"] += 1
                            inserted_sqlite = True
                        elif res_sq == "ALREADY_PRESENT":
                            results["already_present"] += 1

                    if not (rec.in_jsonl or rec.in_archive):
                        append_event_to_jsonl(target_jsonl_path, event_to_persist, prefix_separator=prefix_sep)
                        prefix_sep = ""  # Prefixo usado apenas na primeira linha se necessário
                        results["jsonl_appended"] += 1
                        appended_jsonl = True

                    if (inserted_sqlite or appended_jsonl) and ledger_path:
                        record_reconciliation_in_ledger(
                            ledger_path,
                            {
                                "event_id": eid,
                                "canonical_hash": rec.canonical_hash,
                                "source_file": str(source_fallback_file),
                                "inserted_sqlite": inserted_sqlite,
                                "appended_jsonl": appended_jsonl,
                            },
                        )

                    results["reconciled_ids"].append(eid)
                except sqlite3.OperationalError as op_err:
                    # Falha operacional transitória do SQLite (estado incerto):
                    # NÃO quarentenar! Preservar para reconciliação futura!
                    logger.warning("Falha operacional transitória no SQLite ao reconciliar %s: %s", eid, op_err)
                    results["unresolved_count"] += 1
                    raw_lines_to_keep.append(line)

            elif rec.classification == EventClassification.EVENTO_INVALIDO:
                results["skipped_quarantine"] += 1
                has_quarantined_or_poison = True
                if active_quarantine is not None and raw_event_copy is not None:
                    try:
                        q_res = rec.validation_result
                        if q_res is None or not isinstance(q_res, FinancialValidationResult):
                            if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                                q_res = FinancialValidationResult(
                                    is_valid=False,
                                    status="INVALID",
                                    category="INVALID",
                                    rejection_reason=rec.reason,
                                )
                        if q_res is not None:
                            active_quarantine.quarantine_event(
                                raw_event_copy,
                                q_res,
                                source_caller="reconcile_events_hermetic",
                                category="INVALID",
                            )
                            results["quarantined_count"] += 1
                            results["quarantined_ids"].append(rec.event_id or f"line_{line_num}")
                    except Exception as q_err:
                        logger.error("Falha ao quarentenar evento inválido: %s", q_err)

            elif rec.classification == EventClassification.EVENTO_ID_COLISAO:
                results["skipped_quarantine"] += 1
                has_quarantined_or_poison = True
                if active_quarantine is not None and raw_event_copy is not None:
                    try:
                        if HAS_FINANCIAL_VALIDATOR and FinancialValidationResult is not None:
                            col_res = FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category="COLLISION",
                                rejection_reason=rec.reason,
                            )
                            active_quarantine.quarantine_event(
                                raw_event_copy,
                                col_res,
                                source_caller="reconcile_events_hermetic",
                                category="COLLISION",
                            )
                            results["quarantined_count"] += 1
                            results["quarantined_ids"].append(rec.event_id or f"line_{line_num}")
                    except Exception as q_err:
                        logger.error("Falha ao quarentenar colisão de ID: %s", q_err)

            else:
                results["skipped_quarantine"] += 1

    # Isolamento de Poison Pills e Gestão do Arquivo de Fallback
    should_isolate = isolate_poison_pills or purge_quarantined or (active_quarantine is not None and has_quarantined_or_poison)
    if should_isolate and not reconciler.is_git_tracked_fixture(source_fallback_file):
        if len(raw_lines_to_keep) == 0:
            # Todos os eventos foram reconciliados, já estavam presentes ou foram quarentenados.
            # Remove o arquivo para evitar loops infinitos de retry em execuções subsequentes.
            try:
                source_fallback_file.unlink()
            except Exception as unl_err:
                logger.warning("Falha ao remover arquivo de fallback após isolamento completo: %s", unl_err)
        else:
            # Eventos válidos que sofreram falha transitória do SQLite (OperationalError) permanecem preservados!
            tmp_fb = source_fallback_file.with_suffix(".tmp_pending")
            try:
                with open(tmp_fb, "w", encoding="utf-8") as tf:
                    for pl in raw_lines_to_keep:
                        tf.write(pl.rstrip("\r\n") + "\n")
                    tf.flush()
                    os.fsync(tf.fileno())
                tmp_fb.replace(source_fallback_file)
            except Exception as tmp_err:
                if tmp_fb.exists():
                    try:
                        tmp_fb.unlink()
                    except Exception:
                        pass
                raise

    return results


def record_reconciliation_audit(audit_file: Union[Path, str], entry: Dict[str, Any]) -> None:
    """Registra evento no livro-razão append-only de auditoria de reconciliação com fsync."""
    af = Path(audit_file).expanduser().resolve()
    af.parent.mkdir(parents=True, exist_ok=True)
    raw_line = json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n"
    with open(af, "a", encoding="utf-8") as f:
        f.write(raw_line)
        f.flush()
        os.fsync(f.fileno())


def validate_reconciliation_audit_ledger(
    ledger_path: Union[Path, str],
    expected_environment_uuid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Valida a integridade do livro-razão de auditoria de reconciliação (Gate G4-D B3-B2-C2-A).
    Parsing integral do JSONL linha a linha, validação de pares PREPARED -> COMMITTED,
    detecção de PREPARED órfãos, COMMITTED sem PREPARED, duplicatas de audit_id, UUID divergente
    e integridade de formato.
    """
    lp = Path(ledger_path).expanduser().resolve()
    if not lp.exists() or lp.stat().st_size == 0:
        return {
            "exists": lp.exists(),
            "is_valid": True,
            "error": None,
            "records_count": 0,
            "completed_transactions_count": 0,
            "has_pending_transaction": False,
            "orphaned_prepared": [],
            "last_record": None,
        }

    all_records: List[Dict[str, Any]] = []
    open_prepared: Dict[str, Dict[str, Any]] = {}
    completed_transactions: Dict[str, Dict[str, Any]] = {}
    seen_prepared_ids: Set[str] = set()
    seen_committed_ids: Set[str] = set()

    try:
        with open(lp, "rb") as f:
            content = f.read()

        if not (content.endswith(b"\n") or content.endswith(b"\r\n")):
            return {
                "exists": True,
                "is_valid": False,
                "error": "Última linha do livro-razão de reconciliação não termina em newline (possível truncamento durante escrita).",
                "records_count": 0,
                "completed_transactions_count": 0,
                "has_pending_transaction": False,
                "orphaned_prepared": [],
                "last_record": None,
            }

        lines = content.splitlines()
        for idx, line_bytes in enumerate(lines, start=1):
            s = line_bytes.strip()
            if not s:
                continue
            try:
                line_str = s.decode("utf-8", errors="strict")
                entry = json.loads(line_str)
                if not isinstance(entry, dict):
                    return {
                        "exists": True,
                        "is_valid": False,
                        "error": f"Linha {idx} do livro-razão de reconciliação não é um objeto JSON.",
                        "records_count": len(all_records),
                        "completed_transactions_count": len(completed_transactions),
                        "has_pending_transaction": len(open_prepared) > 0,
                        "orphaned_prepared": list(open_prepared.keys()),
                        "last_record": all_records[-1] if all_records else None,
                    }
            except Exception as j_err:
                return {
                    "exists": True,
                    "is_valid": False,
                    "error": f"Linha {idx} do livro-razão de reconciliação contém JSON inválido ou truncado: {j_err}",
                    "records_count": len(all_records),
                    "completed_transactions_count": len(completed_transactions),
                    "has_pending_transaction": len(open_prepared) > 0,
                    "orphaned_prepared": list(open_prepared.keys()),
                    "last_record": all_records[-1] if all_records else None,
                }

            all_records.append(entry)

            # Validação dos campos obrigatórios
            for req in ["audit_id", "reconciliation_phase", "timestamp_utc", "environment_uuid"]:
                if req not in entry or not str(entry[req]).strip():
                    return {
                        "exists": True,
                        "is_valid": False,
                        "error": f"Linha {idx} ausente de campo obrigatório '{req}'.",
                        "records_count": len(all_records),
                        "completed_transactions_count": len(completed_transactions),
                        "has_pending_transaction": len(open_prepared) > 0,
                        "orphaned_prepared": list(open_prepared.keys()),
                        "last_record": all_records[-1] if all_records else None,
                    }

            aid = str(entry["audit_id"]).strip()
            try:
                uuid.UUID(aid)
            except Exception:
                return {
                    "exists": True,
                    "is_valid": False,
                    "error": f"Linha {idx} possui 'audit_id' com formato UUID inválido: {aid}",
                    "records_count": len(all_records),
                    "completed_transactions_count": len(completed_transactions),
                    "has_pending_transaction": len(open_prepared) > 0,
                    "orphaned_prepared": list(open_prepared.keys()),
                    "last_record": all_records[-1] if all_records else None,
                }

            env_u = str(entry["environment_uuid"]).strip()
            if expected_environment_uuid:
                exp_clean = str(expected_environment_uuid).strip()
                if env_u != exp_clean:
                    return {
                        "exists": True,
                        "is_valid": False,
                        "error": f"Linha {idx} possui environment_uuid divergente (esperado='{exp_clean}', registro='{env_u}').",
                        "records_count": len(all_records),
                        "completed_transactions_count": len(completed_transactions),
                        "has_pending_transaction": len(open_prepared) > 0,
                        "orphaned_prepared": list(open_prepared.keys()),
                        "last_record": all_records[-1] if all_records else None,
                    }

            phase = str(entry["reconciliation_phase"]).strip().upper()
            if phase == "PREPARED":
                if aid in seen_prepared_ids:
                    return {
                        "exists": True,
                        "is_valid": False,
                        "error": f"Linha {idx} possui duplicidade de audit_id em fase PREPARED: {aid}",
                        "records_count": len(all_records),
                        "completed_transactions_count": len(completed_transactions),
                        "has_pending_transaction": len(open_prepared) > 0,
                        "orphaned_prepared": list(open_prepared.keys()),
                        "last_record": all_records[-1] if all_records else None,
                    }
                seen_prepared_ids.add(aid)
                open_prepared[aid] = entry

            elif phase == "COMMITTED":
                if aid not in open_prepared:
                    return {
                        "exists": True,
                        "is_valid": False,
                        "error": f"Linha {idx} possui fase COMMITTED sem respectivo PREPARED prévio para audit_id={aid}.",
                        "records_count": len(all_records),
                        "completed_transactions_count": len(completed_transactions),
                        "has_pending_transaction": len(open_prepared) > 0,
                        "orphaned_prepared": list(open_prepared.keys()),
                        "last_record": all_records[-1] if all_records else None,
                    }
                if aid in seen_committed_ids:
                    return {
                        "exists": True,
                        "is_valid": False,
                        "error": f"Linha {idx} possui duplicidade de audit_id em fase COMMITTED: {aid}",
                        "records_count": len(all_records),
                        "completed_transactions_count": len(completed_transactions),
                        "has_pending_transaction": len(open_prepared) > 0,
                        "orphaned_prepared": list(open_prepared.keys()),
                        "last_record": all_records[-1] if all_records else None,
                    }
                seen_committed_ids.add(aid)
                completed_transactions[aid] = entry
                del open_prepared[aid]

            elif phase == "ABORTED":
                if aid in open_prepared:
                    del open_prepared[aid]
            else:
                return {
                    "exists": True,
                    "is_valid": False,
                    "error": f"Linha {idx} possui fase de reconciliação inválida: '{phase}'. Esperado PREPARED ou COMMITTED.",
                    "records_count": len(all_records),
                    "completed_transactions_count": len(completed_transactions),
                    "has_pending_transaction": len(open_prepared) > 0,
                    "orphaned_prepared": list(open_prepared.keys()),
                    "last_record": all_records[-1] if all_records else None,
                }

        if open_prepared:
            orphans_str = ", ".join(open_prepared.keys())
            return {
                "exists": True,
                "is_valid": False,
                "error": f"Transação de reconciliação incompleta/interrompida detectada (PREPARED sem COMMITTED para audit_id={orphans_str}).",
                "records_count": len(all_records),
                "completed_transactions_count": len(completed_transactions),
                "has_pending_transaction": True,
                "orphaned_prepared": list(open_prepared.keys()),
                "last_record": all_records[-1] if all_records else None,
            }

        return {
            "exists": True,
            "is_valid": True,
            "error": None,
            "records_count": len(all_records),
            "completed_transactions_count": len(completed_transactions),
            "has_pending_transaction": False,
            "orphaned_prepared": [],
            "last_record": all_records[-1] if all_records else None,
        }

    except Exception as exc:
        return {
            "exists": True,
            "is_valid": False,
            "error": f"Falha de I/O ao ler livro-razão de reconciliação: {exc}",
            "records_count": len(all_records),
            "completed_transactions_count": len(completed_transactions),
            "has_pending_transaction": False,
            "orphaned_prepared": [],
            "last_record": None,
        }


def create_reconciliation_snapshot(
    data_dir: Union[Path, str],
    lock_path: Union[Path, str],
    backup_base_dir: Union[Path, str],
    env_uuid: str,
    operator_reason: str,
    source_fallback_file: Optional[Union[Path, str]] = None,
    archive_dir: Optional[Union[Path, str]] = None,
    repo_root: Optional[Union[Path, str]] = None,
    lock_file_obj: Optional[Any] = None,
) -> Tuple[Path, str]:
    """
    Cria snapshot forense completo e atômico pré-reconciliação (Gate G4-D B3-B2-C2-A).
    Aplica verificação preventiva de quota de disco (mínimo 3x tamanho estimado).
    Rejeita symlinks e isola snapshots incompletos em caso de falha.
    Retorna (snapshot_dir, snapshot_manifest_sha256).
    """
    from common.runtime_lock import MANIFEST_FILENAME

    backup_base_dir = Path(backup_base_dir).expanduser().resolve()
    backup_base_dir.mkdir(parents=True, exist_ok=True)
    d_dir = Path(data_dir).expanduser().resolve()
    l_path = Path(lock_path).expanduser().resolve()
    a_dir = Path(archive_dir).expanduser().resolve() if archive_dir else (d_dir / "archive")
    r_root = Path(repo_root).expanduser().resolve() if repo_root else Path.cwd()

    sqlite_path = d_dir / "trading_bot.db"
    wal_path = d_dir / "trading_bot.db-wal"
    shm_path = d_dir / "trading_bot.db-shm"
    jsonl_path = d_dir / "eventos_fluxo.jsonl"
    manifest_path = d_dir / MANIFEST_FILENAME

    # 1. Validação de segurança: rejeitar links simbólicos operacionais
    for check_p in [l_path, manifest_path, jsonl_path, sqlite_path, wal_path, shm_path]:
        if check_p.is_symlink():
            raise RuntimeError(
                f"Origem operacional insegura: link simbólico detectado em '{check_p.name}'. "
                "Operação abortada por segurança contra symlinks/path traversal."
            )

    files_to_copy: List[Tuple[Path, str]] = []
    if l_path.exists():
        files_to_copy.append((l_path, ".operational_runtime.lock.orig"))
    if manifest_path.exists():
        files_to_copy.append((manifest_path, ".runtime_initialized.manifest.orig"))
    if jsonl_path.exists():
        files_to_copy.append((jsonl_path, "eventos_fluxo.jsonl"))

    if wal_path.exists():
        files_to_copy.append((wal_path, "trading_bot.db-wal.raw"))
    if shm_path.exists():
        files_to_copy.append((shm_path, "trading_bot.db-shm.raw"))

    if a_dir.exists():
        if a_dir.is_symlink():
            raise RuntimeError("Origem operacional insegura: diretório archive é um link simbólico.")
        for seg in a_dir.glob("*.jsonl"):
            if seg.is_symlink():
                raise RuntimeError(
                    f"Origem operacional insegura: link simbólico detectado em archive ({seg.name})."
                )
            files_to_copy.append((seg, f"archive/{seg.name}"))

    if source_fallback_file:
        sfb = Path(source_fallback_file).expanduser().resolve()
        if sfb.is_symlink():
            raise RuntimeError(
                f"Origem operacional insegura: link simbólico detectado no arquivo de fallback ({sfb.name})."
            )
        if sfb.exists() and sfb.is_file():
            files_to_copy.append((sfb, f"fallback/{sfb.name}"))

    estimated_bytes = sum(f.stat().st_size for f, _ in files_to_copy if f.exists())
    if sqlite_path.exists():
        estimated_bytes += sqlite_path.stat().st_size

    # 2. Checagem preventiva de quota de disco (mínimo 3x)
    try:
        total_d, used_d, free_d = shutil.disk_usage(backup_base_dir)
        required_free = max(10 * 1024 * 1024, estimated_bytes * 3)
        if free_d < required_free:
            raise RuntimeError(
                f"Espaço insuficiente em disco para snapshot forense de reconciliação: livre={free_d} bytes, "
                f"necessário com margem={required_free} bytes (ENOSPC_PREVENTED)."
            )
    except Exception as disk_err:
        if "ENOSPC_PREVENTED" in str(disk_err):
            raise
        logger.warning("Não foi possível verificar quota de disco: %s", disk_err)

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    uuid_short = str(uuid.uuid4())[:8]
    snapshot_name = f"reconcile_snapshot_{ts_str}_{uuid_short}"
    snapshot_dir = backup_base_dir / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    manifest_data: Dict[str, Any] = {
        "snapshot_version": "1.0",
        "snapshot_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_uuid": env_uuid,
        "operator_reason": operator_reason,
        "host": platform.node(),
        "pid": os.getpid(),
        "files": {},
    }

    try:
        # Cópia consistente do SQLite via Backup API
        if sqlite_path.exists() and sqlite_path.stat().st_size > 0:
            dst_db = snapshot_dir / "trading_bot.db"
            src_conn = sqlite3.connect(f"file:{sqlite_path.resolve().as_posix()}?mode=ro", uri=True)
            dst_conn = sqlite3.connect(str(dst_db))
            src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()

            with open(dst_db, "r+b") as df_db:
                df_db.flush()
                os.fsync(df_db.fileno())

            with open(dst_db, "rb") as df:
                db_bytes = df.read()
                manifest_data["files"]["trading_bot.db"] = {
                    "size_bytes": len(db_bytes),
                    "sha256": hashlib.sha256(db_bytes).hexdigest(),
                    "backup_method": "sqlite3_online_backup_api",
                }

        # Cópia dos demais arquivos
        for src_f, rel_dst in files_to_copy:
            dst_p = snapshot_dir / rel_dst
            dst_p.parent.mkdir(parents=True, exist_ok=True)

            if lock_file_obj is not None and src_f.resolve() == l_path.resolve():
                curr_pos = lock_file_obj.tell()
                lock_file_obj.seek(0)
                raw = lock_file_obj.read()
                lock_file_obj.seek(curr_pos)
                with open(dst_p, "wb") as df:
                    df.write(raw)
                    df.flush()
                    os.fsync(df.fileno())
                src_sha = hashlib.sha256(raw).hexdigest()
            else:
                shutil.copy2(src_f, dst_p)
                with open(dst_p, "r+b") as df:
                    df.flush()
                    os.fsync(df.fileno())
                raw = dst_p.read_bytes()
                src_sha = hashlib.sha256(src_f.read_bytes()).hexdigest()

            dst_sha = hashlib.sha256(raw).hexdigest()
            if dst_sha != src_sha:
                raise RuntimeError(
                    f"Falha de integridade ao copiar {src_f.name}: hash divergente no snapshot de reconciliação."
                )
            manifest_data["files"][rel_dst] = {
                "size_bytes": len(raw),
                "sha256": dst_sha,
                "backup_method": "raw_copy",
            }

        # Gravação do manifesto do snapshot com fsync
        manifest_file = snapshot_dir / "snapshot_manifest.json"
        manifest_raw = json.dumps(manifest_data, indent=2).encode("utf-8")
        with open(manifest_file, "wb") as mf:
            mf.write(manifest_raw)
            mf.flush()
            os.fsync(mf.fileno())

        if hasattr(os, "O_DIRECTORY"):
            try:
                dir_fd = os.open(str(snapshot_dir), os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                pass

        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        return snapshot_dir, manifest_sha256

    except Exception as exc:
        incomplete_dir = backup_base_dir / f"{snapshot_name}.INCOMPLETE_FAILED_{ts_str}"
        try:
            if snapshot_dir.exists():
                snapshot_dir.rename(incomplete_dir)
        except Exception:
            pass
        raise RuntimeError(f"Falha crítica na criação do snapshot forense de reconciliação: {exc}") from exc


class TargetVerificationCode(str, Enum):
    """Códigos estruturados de resultado da pós-verificação financeira de destinos."""
    VERIFICATION_SUCCESS = "VERIFICATION_SUCCESS"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    TARGET_UNAVAILABLE = "TARGET_UNAVAILABLE"
    CORRUPTED_TARGET = "CORRUPTED_TARGET"
    SCHEMA_INCOMPATIBLE = "SCHEMA_INCOMPATIBLE"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    EMPTY_EVENT_SET = "EMPTY_EVENT_SET"
    INVALID_EVENT_STRUCTURE = "INVALID_EVENT_STRUCTURE"
    DUPLICATE_IN_EXPECTED_SET = "DUPLICATE_IN_EXPECTED_SET"
    INVALID_DESTINATION_SPEC = "INVALID_DESTINATION_SPEC"
    EMPTY_DESTINATIONS = "EMPTY_DESTINATIONS"


@dataclass
class TargetVerificationResult:
    """Resultado estruturado e auditável da verificação de alvos de reconciliação."""
    approved: bool
    code: TargetVerificationCode
    total_expected: int
    verified_count: int
    missing_count: int
    duplicate_count: int
    divergent_hash_count: int
    conflict_count: int
    targets_checked: List[str] = field(default_factory=list)
    problematic_event_ids: List[str] = field(default_factory=list)
    details_by_event: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "code": self.code.value if isinstance(self.code, Enum) else str(self.code),
            "total_expected": self.total_expected,
            "verified_count": self.verified_count,
            "missing_count": self.missing_count,
            "duplicate_count": self.duplicate_count,
            "divergent_hash_count": self.divergent_hash_count,
            "conflict_count": self.conflict_count,
            "targets_checked": self.targets_checked,
            "problematic_event_ids": self.problematic_event_ids,
            "details_by_event": self.details_by_event,
            "errors": self.errors,
        }


def verify_reconciliation_targets(
    expected_events: Sequence[Dict[str, Any]],
    sqlite_path: Optional[Union[Path, str]] = None,
    jsonl_path: Optional[Union[Path, str]] = None,
    archive_dir: Optional[Union[Path, str]] = None,
    destinations_per_event: Optional[Dict[str, Set[str]]] = None,
    require_all_destinations_default: bool = True,
) -> TargetVerificationResult:
    """
    Motor hermético de pós-verificação financeira de destinos persistidos.

    Valida de forma estritamente READ-ONLY a integridade, presença e unicidade
    física dos eventos de fallback nos destinos finais (SQLite e JSONL/Archive).

    Garantias contratuais:
    - Sem mutações financeiras ou de disco (nenhum insert, update, append ou truncate).
    - Não utiliza caminhos globais implícitos nem acessa dados/ operacional automaticamente.
    - Exaustão de registros no SQLite: rejeita qualquer duplicata sem uso de LIMIT 1.
    - Checagem estrita de hash canônico via compute_canonical_payload_hash.
    - Detecção fail-closed de symlinks indevidos, arquivos ausentes, truncados ou corrompidos.
    - Rejeição estrita de lotes vazios, duplicados no lote de entrada ou com destinos mal configurados.
    """
    targets_checked: List[str] = []

    # 1. Checagem defensiva de symlinks em todos os caminhos fornecidos
    paths_to_audit: List[Tuple[str, Optional[Union[Path, str]]]] = [
        ("sqlite", sqlite_path),
        ("jsonl", jsonl_path),
        ("archive", archive_dir),
    ]
    for target_label, p_raw in paths_to_audit:
        if p_raw is not None:
            p_obj = Path(p_raw)
            if p_obj.is_symlink():
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.SECURITY_VIOLATION,
                    total_expected=len(expected_events) if isinstance(expected_events, (list, tuple)) else 0,
                    verified_count=0,
                    missing_count=len(expected_events) if isinstance(expected_events, (list, tuple)) else 0,
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=targets_checked,
                    problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events if isinstance(e, dict)],
                    errors=[f"Symlink indevido proibido detectado no destino '{target_label}': {p_obj}"],
                )

    if archive_dir is not None:
        p_arch = Path(archive_dir)
        if p_arch.exists() and p_arch.is_dir():
            for sub_file in p_arch.iterdir():
                if sub_file.is_symlink():
                    return TargetVerificationResult(
                        approved=False,
                        code=TargetVerificationCode.SECURITY_VIOLATION,
                        total_expected=len(expected_events) if isinstance(expected_events, (list, tuple)) else 0,
                        verified_count=0,
                        missing_count=len(expected_events) if isinstance(expected_events, (list, tuple)) else 0,
                        duplicate_count=0,
                        divergent_hash_count=0,
                        conflict_count=0,
                        targets_checked=targets_checked,
                        problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events if isinstance(e, dict)],
                        errors=[f"Symlink indevido proibido detectado no diretório archive: {sub_file}"],
                    )

    # 2. Validação estrutural rigorosa do lote esperado (Etapa 2 e 4)
    if not isinstance(expected_events, (list, tuple)):
        return TargetVerificationResult(
            approved=False,
            code=TargetVerificationCode.INVALID_EVENT_STRUCTURE,
            total_expected=0,
            verified_count=0,
            missing_count=0,
            duplicate_count=0,
            divergent_hash_count=0,
            conflict_count=0,
            targets_checked=[],
            problematic_event_ids=[],
            details_by_event={},
            errors=["expected_events deve ser obrigatoriamente uma lista ou tupla de eventos."],
        )

    if len(expected_events) == 0:
        return TargetVerificationResult(
            approved=False,
            code=TargetVerificationCode.EMPTY_EVENT_SET,
            total_expected=0,
            verified_count=0,
            missing_count=0,
            duplicate_count=0,
            divergent_hash_count=0,
            conflict_count=0,
            targets_checked=[],
            problematic_event_ids=[],
            details_by_event={},
            errors=["Lote de eventos esperado está vazio (EMPTY_EVENT_SET). Aprovação por vacuidade recusada."],
        )

    seen_eids: Set[str] = set()

    for idx, ev in enumerate(expected_events):
        if not isinstance(ev, dict):
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.INVALID_EVENT_STRUCTURE,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=[],
                problematic_event_ids=[f"<INVALID_TYPE_{idx}>"],
                details_by_event={},
                errors=[f"Elemento no índice {idx} do lote esperado não é um dicionário: tipo={type(ev).__name__}"],
            )

        raw_eid = ev.get("event_id") or ev.get("id")
        eid = str(raw_eid).strip() if raw_eid is not None and str(raw_eid).strip() else ""

        if not eid:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.INVALID_EVENT_STRUCTURE,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=[],
                problematic_event_ids=[f"<MISSING_ID_{idx}>"],
                details_by_event={},
                errors=[f"Evento no índice {idx} não possui event_id contratual válido."],
            )

        if eid in seen_eids:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.DUPLICATE_IN_EXPECTED_SET,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=1,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=[],
                problematic_event_ids=[eid],
                details_by_event={},
                errors=[f"event_id duplicado detectado no lote esperado de entrada: {eid} (DUPLICATE_IN_EXPECTED_SET)."],
            )

        try:
            compute_canonical_payload_hash(ev)
        except Exception as hash_err:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.INVALID_EVENT_STRUCTURE,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=[],
                problematic_event_ids=[eid],
                details_by_event={},
                errors=[f"Falha ao computar hash canônico para event_id {eid}: {hash_err}"],
            )

        seen_eids.add(eid)

    # 3. Validação estrita do contrato de destinos (Etapa 3)
    VALID_DESTINATIONS: Set[str] = {"sqlite", "jsonl"}
    event_destinations_map: Dict[str, Set[str]] = {}

    if destinations_per_event is not None:
        if not isinstance(destinations_per_event, dict):
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.INVALID_DESTINATION_SPEC,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=[],
                problematic_event_ids=[],
                details_by_event={},
                errors=["destinations_per_event deve ser um dicionário mapeando event_id para Set[str]."],
            )

        if len(destinations_per_event) == 0:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.EMPTY_DESTINATIONS,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=[],
                problematic_event_ids=[],
                details_by_event={},
                errors=["destinations_per_event fornecido está vazio (EMPTY_DESTINATIONS)."],
            )

        map_keys = set(destinations_per_event.keys())
        missing_keys = seen_eids - map_keys
        if missing_keys:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.INVALID_DESTINATION_SPEC,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=[],
                problematic_event_ids=sorted(list(missing_keys)),
                details_by_event={},
                errors=[f"destinations_per_event incompleto: faltam destinos para {sorted(list(missing_keys))}"],
            )

        unknown_keys = map_keys - seen_eids
        if unknown_keys:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.INVALID_DESTINATION_SPEC,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=[],
                problematic_event_ids=sorted(list(unknown_keys)),
                details_by_event={},
                errors=[f"destinations_per_event contém chaves desconhecidas: {sorted(list(unknown_keys))}"],
            )

        for eid, dests in destinations_per_event.items():
            if not isinstance(dests, (set, frozenset)):
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.INVALID_DESTINATION_SPEC,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=[],
                    problematic_event_ids=[eid],
                    details_by_event={},
                    errors=[f"Destino para {eid} deve ser Set[str], obtido: {type(dests).__name__}"],
                )

            if len(dests) == 0:
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.EMPTY_DESTINATIONS,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=[],
                    problematic_event_ids=[eid],
                    details_by_event={},
                    errors=[f"Conjunto de destinos vazio para event_id {eid} (EMPTY_DESTINATIONS)."],
                )

            unsupported = set(dests) - VALID_DESTINATIONS
            if unsupported:
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.INVALID_DESTINATION_SPEC,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=[],
                    problematic_event_ids=[eid],
                    details_by_event={},
                    errors=[f"Destino(s) não suportado(s) para {eid}: {sorted(list(unsupported))}"],
                )

            event_destinations_map[eid] = set(dests)

    else:
        if require_all_destinations_default:
            if sqlite_path is None:
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.TARGET_UNAVAILABLE,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=[],
                    problematic_event_ids=[],
                    details_by_event={},
                    errors=["sqlite_path é obrigatório no modo padrão (require_all_destinations_default=True)."],
                )

            if jsonl_path is None and archive_dir is None:
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.TARGET_UNAVAILABLE,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=[],
                    problematic_event_ids=[],
                    details_by_event={},
                    errors=["jsonl_path ou archive_dir é obrigatório no modo padrão (require_all_destinations_default=True)."],
                )

            for eid in seen_eids:
                event_destinations_map[eid] = {"sqlite", "jsonl"}

        else:
            def_dests: Set[str] = set()
            if sqlite_path is not None:
                def_dests.add("sqlite")
            if jsonl_path is not None or archive_dir is not None:
                def_dests.add("jsonl")

            if len(def_dests) == 0:
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.EMPTY_DESTINATIONS,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=[],
                    problematic_event_ids=[],
                    details_by_event={},
                    errors=["Nenhum caminho de destino físico foi fornecido (EMPTY_DESTINATIONS)."],
                )

            for eid in seen_eids:
                event_destinations_map[eid] = set(def_dests)

    need_sqlite = any("sqlite" in dests for dests in event_destinations_map.values())
    need_jsonl = any("jsonl" in dests for dests in event_destinations_map.values())

    # 4. Verificação e leitura do SQLite (Estritamente Read-Only, Sem LIMIT 1)
    sqlite_records_by_id: Dict[str, List[Dict[str, Any]]] = {}
    sqlite_records_by_corr: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}

    if need_sqlite:
        targets_checked.append("sqlite")
        if sqlite_path is None:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.TARGET_UNAVAILABLE,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=targets_checked,
                problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                errors=["Caminho SQLite não fornecido, porém exigido pelos eventos esperados"],
            )

        p_sq = Path(sqlite_path)
        if not p_sq.exists() or not p_sq.is_file():
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.TARGET_UNAVAILABLE,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=targets_checked,
                problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                errors=[f"Banco SQLite não encontrado em {p_sq}"],
            )

        conn = None
        try:
            try:
                uri = f"file:{p_sq.resolve().as_posix()}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, timeout=10.0)
            except Exception:
                conn = sqlite3.connect(str(p_sq), timeout=10.0)

            cur = conn.cursor()

            # Verificação de integridade física
            cur.execute("PRAGMA quick_check;")
            qc_rows = cur.fetchall()
            if not qc_rows or qc_rows[0][0] != "ok":
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.CORRUPTED_TARGET,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=targets_checked,
                    problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                    errors=[f"Integridade do SQLite comprometida em {p_sq}: {qc_rows}"],
                )

            # Verificação do schema
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='events'")
            if not cur.fetchone():
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.SCHEMA_INCOMPATIBLE,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=targets_checked,
                    problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                    errors=[f"Tabela 'events' não existe em {p_sq}"],
                )

            cur.execute("PRAGMA table_info(events)")
            cols = {row[1] for row in cur.fetchall()}
            if not {"timestamp_ms", "event_type", "payload"}.issubset(cols):
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.SCHEMA_INCOMPATIBLE,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=targets_checked,
                    problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                    errors=[f"Schema incompatível na tabela 'events' de {p_sq}: colunas={cols}"],
                )

            # Varredura exaustiva de todas as linhas (sem LIMIT 1)
            cur.execute("SELECT id, timestamp_ms, event_type, COALESCE(symbol, ''), payload FROM events")
            for row_id, ts_ms, ev_type, sym, payload_raw in cur.fetchall():
                payload_dict = {}
                try:
                    if isinstance(payload_raw, str):
                        payload_dict = json.loads(payload_raw)
                    elif isinstance(payload_raw, dict):
                        payload_dict = payload_raw
                except Exception:
                    payload_dict = {}

                row_eid = str(payload_dict.get("event_id") or payload_dict.get("id") or "").strip()
                row_hash = compute_canonical_payload_hash(payload_dict) if payload_dict else ""
                row_ts = int(ts_ms) if ts_ms is not None else 0
                row_type = str(ev_type or "")
                row_sym = str(sym or "")

                entry = {
                    "row_id": row_id,
                    "timestamp_ms": row_ts,
                    "event_type": row_type,
                    "symbol": row_sym,
                    "canonical_hash": row_hash,
                    "event_id": row_eid,
                    "payload": payload_dict,
                }
                if row_eid:
                    sqlite_records_by_id.setdefault(row_eid, []).append(entry)
                sqlite_records_by_corr.setdefault((row_ts, row_type, row_sym), []).append(entry)

        except sqlite3.DatabaseError as db_err:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.CORRUPTED_TARGET,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=targets_checked,
                problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                errors=[f"Erro de banco SQLite em {p_sq}: {db_err}"],
            )
        except Exception as ex_err:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.TARGET_UNAVAILABLE,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=targets_checked,
                problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                errors=[f"Falha de leitura no SQLite {p_sq}: {ex_err}"],
            )
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    # 5. Verificação e leitura do JSONL e Archive
    jsonl_records_by_id: Dict[str, List[Dict[str, Any]]] = {}
    jsonl_records_by_corr: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}

    if need_jsonl:
        targets_checked.append("jsonl")
        jsonl_files_to_read: List[Tuple[Path, bool]] = []

        if archive_dir is not None:
            p_arch = Path(archive_dir)
            if p_arch.exists() and p_arch.is_dir():
                for f in sorted(p_arch.glob("eventos_fluxo_*.jsonl")):
                    if f.is_file():
                        jsonl_files_to_read.append((f, True))

        if jsonl_path is not None:
            p_jl = Path(jsonl_path)
            if p_jl.exists() and p_jl.is_file():
                jsonl_files_to_read.append((p_jl, False))
            elif p_jl.exists() and not p_jl.is_file():
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.TARGET_UNAVAILABLE,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=targets_checked,
                    problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                    errors=[f"Caminho JSONL não é arquivo regular: {p_jl}"],
                )

        if not jsonl_files_to_read:
            return TargetVerificationResult(
                approved=False,
                code=TargetVerificationCode.TARGET_UNAVAILABLE,
                total_expected=len(expected_events),
                verified_count=0,
                missing_count=len(expected_events),
                duplicate_count=0,
                divergent_hash_count=0,
                conflict_count=0,
                targets_checked=targets_checked,
                problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                errors=["Nenhum arquivo JSONL ou Archive disponível para leitura"],
            )

        for f_path, is_arch in jsonl_files_to_read:
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    line_idx = 0
                    for raw_l in f:
                        line_idx += 1
                        l_str = raw_l.strip()
                        if not l_str:
                            continue
                        try:
                            obj = json.loads(l_str)
                        except Exception as decode_err:
                            return TargetVerificationResult(
                                approved=False,
                                code=TargetVerificationCode.CORRUPTED_TARGET,
                                total_expected=len(expected_events),
                                verified_count=0,
                                missing_count=len(expected_events),
                                duplicate_count=0,
                                divergent_hash_count=0,
                                conflict_count=0,
                                targets_checked=targets_checked,
                                problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                                errors=[f"JSON inválido ou truncado em {f_path.name}:{line_idx}: {decode_err}"],
                            )
                        if not isinstance(obj, dict):
                            return TargetVerificationResult(
                                approved=False,
                                code=TargetVerificationCode.CORRUPTED_TARGET,
                                total_expected=len(expected_events),
                                verified_count=0,
                                missing_count=len(expected_events),
                                duplicate_count=0,
                                divergent_hash_count=0,
                                conflict_count=0,
                                targets_checked=targets_checked,
                                problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                                errors=[f"Linha {line_idx} em {f_path.name} não é objeto JSON válido"],
                            )

                        l_eid = str(obj.get("event_id") or obj.get("id") or "").strip()
                        l_hash = compute_canonical_payload_hash(obj)
                        l_ep = obj.get("epoch_ms") or obj.get("timestamp_ms")
                        l_ep_int = int(l_ep) if l_ep is not None and str(l_ep).isdigit() else 0
                        l_type = str(obj.get("tipo_evento") or obj.get("event_type") or obj.get("type") or "")
                        l_sym = str(obj.get("symbol") or obj.get("ativo") or "")

                        entry = {
                            "file": str(f_path),
                            "line": line_idx,
                            "event_id": l_eid,
                            "canonical_hash": l_hash,
                            "epoch_ms": l_ep_int,
                            "tipo_evento": l_type,
                            "symbol": l_sym,
                            "is_archive": is_arch,
                            "payload": obj,
                        }
                        if l_eid:
                            jsonl_records_by_id.setdefault(l_eid, []).append(entry)
                        jsonl_records_by_corr.setdefault((l_ep_int, l_type, l_sym), []).append(entry)
            except Exception as io_err:
                return TargetVerificationResult(
                    approved=False,
                    code=TargetVerificationCode.TARGET_UNAVAILABLE,
                    total_expected=len(expected_events),
                    verified_count=0,
                    missing_count=len(expected_events),
                    duplicate_count=0,
                    divergent_hash_count=0,
                    conflict_count=0,
                    targets_checked=targets_checked,
                    problematic_event_ids=[str(e.get("event_id") or e.get("id") or "") for e in expected_events],
                    errors=[f"Erro de I/O ao ler {f_path}: {io_err}"],
                )

    # 6. Avaliação rigorosa e independente de cada evento esperado
    verified_count = 0
    missing_count = 0
    duplicate_count = 0
    divergent_hash_count = 0
    conflict_count = 0
    problematic_event_ids: List[str] = []
    details_by_event: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    for idx, ev in enumerate(expected_events):
        raw_eid = ev.get("event_id") or ev.get("id")
        eid = str(raw_eid).strip() if raw_eid is not None and str(raw_eid).strip() else ""

        expected_hash = compute_canonical_payload_hash(ev)
        req_dests = event_destinations_map.get(eid, set())

        ev_valid = True
        ev_errors: List[str] = []
        in_sqlite = False
        in_jsonl = False
        in_archive = False
        sqlite_hash_match = False
        jsonl_hash_match = False

        # Verificação SQLite
        if "sqlite" in req_dests:
            sq_recs = sqlite_records_by_id.get(eid, [])
            if len(sq_recs) == 0:
                ev_valid = False
                missing_count += 1
                ev_errors.append("Ausente no destino SQLite")

                # Checagem diagnóstica por correlação temporal (T12)
                ep_val = ev.get("epoch_ms") or ev.get("timestamp_ms")
                ep_int = int(ep_val) if ep_val is not None and str(ep_val).isdigit() else 0
                t_str = str(ev.get("tipo_evento") or ev.get("event_type") or ev.get("type") or "")
                s_str = str(ev.get("symbol") or ev.get("ativo") or "")
                corr_sq = sqlite_records_by_corr.get((ep_int, t_str, s_str), [])
                if corr_sq:
                    ev_errors.append(
                        f"Coincidência temporal no SQLite com ID divergente {[r['event_id'] for r in corr_sq]}; presença não reconhecida."
                    )
            elif len(sq_recs) > 1:
                ev_valid = False
                duplicate_count += 1
                ev_errors.append(f"Duplicata no SQLite: {len(sq_recs)} linhas para o event_id {eid}")
            else:
                in_sqlite = True
                if sq_recs[0]["canonical_hash"] != expected_hash:
                    ev_valid = False
                    divergent_hash_count += 1
                    conflict_count += 1
                    ev_errors.append(
                        f"Hash divergente no SQLite: esperado={expected_hash}, persistido={sq_recs[0]['canonical_hash']}"
                    )
                else:
                    sqlite_hash_match = True

        # Verificação JSONL e Archive
        if "jsonl" in req_dests:
            jl_recs = jsonl_records_by_id.get(eid, [])
            if len(jl_recs) == 0:
                ev_valid = False
                # Não duplicar missing_count se já foi incrementado pelo SQLite
                if "sqlite" not in req_dests or in_sqlite:
                    missing_count += 1
                ev_errors.append("Ausente no destino JSONL/Archive")

                # Checagem diagnóstica por correlação temporal (T12)
                ep_val = ev.get("epoch_ms") or ev.get("timestamp_ms")
                ep_int = int(ep_val) if ep_val is not None and str(ep_val).isdigit() else 0
                t_str = str(ev.get("tipo_evento") or ev.get("event_type") or ev.get("type") or "")
                s_str = str(ev.get("symbol") or ev.get("ativo") or "")
                corr_jl = jsonl_records_by_corr.get((ep_int, t_str, s_str), [])
                if corr_jl:
                    ev_errors.append(
                        f"Coincidência temporal no JSONL com ID divergente {[r['event_id'] for r in corr_jl]}; presença não reconhecida."
                    )
            elif len(jl_recs) > 1:
                ev_valid = False
                duplicate_count += 1
                ev_errors.append(f"Duplicata no JSONL/Archive: {len(jl_recs)} registros para o event_id {eid}")
            else:
                if jl_recs[0]["is_archive"]:
                    in_archive = True
                else:
                    in_jsonl = True

                if jl_recs[0]["canonical_hash"] != expected_hash:
                    ev_valid = False
                    divergent_hash_count += 1
                    conflict_count += 1
                    ev_errors.append(
                        f"Hash divergente no JSONL/Archive: esperado={expected_hash}, persistido={jl_recs[0]['canonical_hash']}"
                    )
                else:
                    jsonl_hash_match = True

        details_by_event[eid] = {
            "event_id": eid,
            "destinations_required": list(sorted(req_dests)),
            "in_sqlite": in_sqlite,
            "in_jsonl": in_jsonl,
            "in_archive": in_archive,
            "sqlite_hash_match": sqlite_hash_match,
            "jsonl_hash_match": jsonl_hash_match,
            "errors": ev_errors,
        }

        if ev_valid:
            verified_count += 1
        else:
            problematic_event_ids.append(eid)
            errors.extend(ev_errors)

    approved = (
        len(expected_events) > 0
        and len(targets_checked) > 0
        and verified_count == len(expected_events)
        and missing_count == 0
        and duplicate_count == 0
        and divergent_hash_count == 0
        and conflict_count == 0
        and len(errors) == 0
    )

    code = TargetVerificationCode.VERIFICATION_SUCCESS if approved else TargetVerificationCode.VERIFICATION_FAILED

    return TargetVerificationResult(
        approved=approved,
        code=code,
        total_expected=len(expected_events),
        verified_count=verified_count,
        missing_count=missing_count,
        duplicate_count=duplicate_count,
        divergent_hash_count=divergent_hash_count,
        conflict_count=conflict_count,
        targets_checked=targets_checked,
        problematic_event_ids=problematic_event_ids,
        details_by_event=details_by_event,
        errors=errors,
    )


class ReconciliationTransactionStatus(str, Enum):
    """Status estruturado do resultado da transação de reconciliação (Gate C2-B2.2)."""
    SUCCESS = "SUCCESS"
    NO_CHANGES = "NO_CHANGES"
    COMMITTED = "COMMITTED"
    FAILED_BEFORE_PREPARED = "FAILED_BEFORE_PREPARED"
    PREPARED_INCOMPLETE = "PREPARED_INCOMPLETE"
    PERSISTENCE_UNCERTAIN = "PERSISTENCE_UNCERTAIN"


@dataclass
class ReconciliationTransactionResult:
    """
    Resultado auditável e estruturado da transação de reconciliação hermética (Gate C2-B2.2 / H2).

    Semântica dos campos de gravação (prepared_recorded / committed_recorded):
    - True: Persistência durável do registro confirmada no livro-razão append-only.
    - False: Nenhuma tentativa de gravação foi executada (ex: falhas anteriores no preflight ou no-op).
    - None: A tentativa de gravação foi iniciada, mas sua confirmação durável falhou no I/O do disco
            (status PERSISTENCE_UNCERTAIN). Não expressa ausência comprovada; exige perícia fail-closed.
    """
    audit_id: str
    environment_uuid: str
    status: ReconciliationTransactionStatus
    stage_reached: str
    total_events: int = 0
    sqlite_inserted: int = 0
    jsonl_appended: int = 0
    already_present: int = 0
    verification_result: Optional[TargetVerificationResult] = None
    snapshot_dir: Optional[str] = None
    snapshot_manifest_sha256: Optional[str] = None
    prepared_recorded: Optional[bool] = False
    committed_recorded: Optional[bool] = False
    manifest_updated: bool = False
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "environment_uuid": self.environment_uuid,
            "status": self.status.value if isinstance(self.status, Enum) else str(self.status),
            "stage_reached": self.stage_reached,
            "total_events": self.total_events,
            "sqlite_inserted": self.sqlite_inserted,
            "jsonl_appended": self.jsonl_appended,
            "already_present": self.already_present,
            "verification_result": self.verification_result.to_dict() if hasattr(self.verification_result, "to_dict") else None,
            "snapshot_dir": self.snapshot_dir,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "prepared_recorded": self.prepared_recorded,
            "committed_recorded": self.committed_recorded,
            "manifest_updated": self.manifest_updated,
            "errors": self.errors,
            "details": self.details,
        }


def execute_reconciliation_transaction_hermetic(
    data_dir: Union[Path, str],
    sqlite_path: Union[Path, str],
    jsonl_path: Union[Path, str],
    archive_dir: Union[Path, str],
    source_fallback_file: Union[Path, str],
    manifest_path: Union[Path, str],
    ledger_path: Union[Path, str],
    backup_base_dir: Union[Path, str],
    environment_uuid: str,
    operator_reason: str,
    audit_id: Optional[str] = None,
    lock_authorization: Optional[Any] = None,
    operational_lock: Optional[Any] = None,
    repo_root: Optional[Union[Path, str]] = None,
) -> ReconciliationTransactionResult:
    """
    Orquestrador transacional hermético de reconciliação financeira (Gate G4-D B3-B2-C2-B2.2).

    Executa o ciclo completo de reconciliação financeira sob sessão exclusiva de manutenção P0:
    1. Validação estrita de parâmetros e bloqueio fail-closed de links simbólicos.
    2. Comprovação de posse física de lock e autorização formal de manutenção.
    3. Preflight financeiro prévio (integridade SQLite, JSONL, Archive, manifesto, ledger H2 e fixtures Git).
    4. Validação e classificação prévia do lote de fallback (rejeita duplicatas, IDs vazios e colisões).
    5. Snapshot forense completo com verificação preventiva de quota e integridade SHA256.
    6. Registro auditável PREPARED no ledger com flush e fsync.
    7. Aplicação transacional ordenada estritamente (1. SQLite -> 2. JSONL).
    8. Pós-verificação financeira independente e hermética nos destinos finais.
    9. Atualização autorizada do manifesto operacional (versão 1.2 com proteção anti-TOCTOU).
    10. Registro auditável COMMITTED com mesmo audit_id.
    11. Liberação controlada da sessão mantendo integridade dos estados operacionais.
    """
    from common.runtime_lock import (
        MaintenanceAuthorizationToken,
        OperationalRuntimeLock,
        RuntimeLockError,
        RuntimeState,
        read_runtime_manifest,
        update_runtime_manifest_reconciliation_state,
        validate_manifest_structure,
        verify_maintenance_authorization,
        verify_manifest_reconciliation_consistency,
    )

    clean_env_uuid = str(environment_uuid).strip() if environment_uuid else ""
    try:
        uuid.UUID(clean_env_uuid)
    except Exception:
        return ReconciliationTransactionResult(
            audit_id=str(audit_id or ""),
            environment_uuid=clean_env_uuid,
            status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
            stage_reached="PARAM_VALIDATION",
            errors=["environment_uuid fornecido é inválido ou nulo."],
        )

    clean_audit_id = str(audit_id).strip() if audit_id else str(uuid.uuid4())
    try:
        uuid.UUID(clean_audit_id)
    except Exception:
        return ReconciliationTransactionResult(
            audit_id=clean_audit_id,
            environment_uuid=clean_env_uuid,
            status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
            stage_reached="PARAM_VALIDATION",
            errors=["audit_id fornecido não é um UUID válido."],
        )

    clean_reason = str(operator_reason).strip() if operator_reason else ""
    if len(clean_reason) < 10:
        return ReconciliationTransactionResult(
            audit_id=clean_audit_id,
            environment_uuid=clean_env_uuid,
            status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
            stage_reached="PARAM_VALIDATION",
            errors=["operator_reason deve conter justificativa auditável de pelo menos 10 caracteres."],
        )

    # Resolução de caminhos
    p_data = Path(data_dir).expanduser().resolve()
    p_sqlite = Path(sqlite_path).expanduser().resolve()
    p_jsonl = Path(jsonl_path).expanduser().resolve()
    p_archive = Path(archive_dir).expanduser().resolve()
    p_fallback = Path(source_fallback_file).expanduser().resolve()
    p_manifest = Path(manifest_path).expanduser().resolve()
    p_ledger = Path(ledger_path).expanduser().resolve()
    p_backup = Path(backup_base_dir).expanduser().resolve()
    p_repo = Path(repo_root).expanduser().resolve() if repo_root else Path.cwd()

    # Validação de symlinks
    for label, p_chk in [
        ("data_dir", p_data),
        ("sqlite_path", p_sqlite),
        ("jsonl_path", p_jsonl),
        ("archive_dir", p_archive),
        ("source_fallback_file", p_fallback),
        ("manifest_path", p_manifest),
        ("ledger_path", p_ledger),
        ("backup_base_dir", p_backup),
    ]:
        if p_chk.is_symlink():
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="SECURITY_VALIDATION",
                errors=[f"Origem operacional insegura: symlink proibido detectado em '{label}': {p_chk}"],
            )

    # Gerenciamento de lock e autorização P0
    we_acquired_lock = False
    active_token = None
    managed_lock = None

    try:
        if lock_authorization is not None:
            if isinstance(lock_authorization, MaintenanceAuthorizationToken):
                if not lock_authorization.is_active:
                    return ReconciliationTransactionResult(
                        audit_id=clean_audit_id,
                        environment_uuid=clean_env_uuid,
                        status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                        stage_reached="AUTHORIZATION",
                        errors=["Autorização de manutenção fornecida está revogada ou inativa."],
                    )
                active_token = lock_authorization
                managed_lock = lock_authorization.lock_instance
                if not getattr(managed_lock, "_is_maintenance_session", False):
                    return ReconciliationTransactionResult(
                        audit_id=clean_audit_id,
                        environment_uuid=clean_env_uuid,
                        status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                        stage_reached="AUTHORIZATION",
                        errors=["Autorização não foi emitida sob sessão exclusiva de manutenção."],
                    )
            elif isinstance(lock_authorization, OperationalRuntimeLock):
                managed_lock = lock_authorization
                if not getattr(managed_lock, "_is_maintenance_session", False) or not managed_lock.is_locked:
                    return ReconciliationTransactionResult(
                        audit_id=clean_audit_id,
                        environment_uuid=clean_env_uuid,
                        status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                        stage_reached="AUTHORIZATION",
                        errors=["Lock operacional fornecido não está em sessão exclusiva de manutenção ativa."],
                    )
                try:
                    active_token = managed_lock.create_maintenance_authorization(environment_uuid=clean_env_uuid)
                except Exception as auth_err:
                    return ReconciliationTransactionResult(
                        audit_id=clean_audit_id,
                        environment_uuid=clean_env_uuid,
                        status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                        stage_reached="AUTHORIZATION",
                        errors=[f"Falha ao gerar autorização a partir do lock: {auth_err}"],
                    )
            else:
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                    stage_reached="AUTHORIZATION",
                    errors=[f"lock_authorization de tipo inválido: {type(lock_authorization).__name__}"],
                )
        elif operational_lock is not None:
            if not isinstance(operational_lock, OperationalRuntimeLock):
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                    stage_reached="AUTHORIZATION",
                    errors=[f"operational_lock de tipo inválido: {type(operational_lock).__name__}"],
                )
            if operational_lock.is_locked and not getattr(operational_lock, "_is_maintenance_session", False):
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                    stage_reached="AUTHORIZATION",
                    errors=["Lock operacional está retido por processo RUNNING convencional; não autoriza manutenção."],
                )
            if operational_lock.is_locked and getattr(operational_lock, "_is_maintenance_session", False):
                managed_lock = operational_lock
                try:
                    active_token = managed_lock.create_maintenance_authorization(environment_uuid=clean_env_uuid)
                except Exception as auth_err:
                    return ReconciliationTransactionResult(
                        audit_id=clean_audit_id,
                        environment_uuid=clean_env_uuid,
                        status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                        stage_reached="AUTHORIZATION",
                        errors=[f"Falha ao emitir autorização do lock de manutenção: {auth_err}"],
                    )
            else:
                try:
                    operational_lock.acquire_maintenance(
                        allowed_states=[
                            RuntimeState.CLEAN,
                            RuntimeState.DIRTY,
                            RuntimeState.UNKNOWN,
                            "RUNNING_RESIDUAL",
                        ]
                    )
                    we_acquired_lock = True
                    managed_lock = operational_lock
                    active_token = managed_lock.create_maintenance_authorization(environment_uuid=clean_env_uuid)
                except Exception as acq_err:
                    return ReconciliationTransactionResult(
                        audit_id=clean_audit_id,
                        environment_uuid=clean_env_uuid,
                        status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                        stage_reached="LOCK_ACQUISITION",
                        errors=[f"Falha ao adquirir lock exclusivo de manutenção: {acq_err}"],
                    )
        else:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="AUTHORIZATION",
                errors=["Nenhuma autorização de manutenção ou lock foi fornecido."],
            )

        # Validação formal de autorização
        try:
            verify_maintenance_authorization(
                lock_authorization=active_token,
                manifest_path=p_manifest,
                expected_env_uuid=clean_env_uuid,
            )
        except Exception as verif_auth_err:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="AUTHORIZATION",
                errors=[f"Validação da autorização falhou: {verif_auth_err}"],
            )
        # Validação de integridade estrutural dos metadados do lock operacional sob exclusividade física (Gate C2-B2.2-H2 / P1)
        if hasattr(managed_lock, "check_metadata_integrity_under_lock"):
            lock_meta_valid, lock_meta_err = managed_lock.check_metadata_integrity_under_lock()
            if not lock_meta_valid:
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                    stage_reached="PREFLIGHT",
                    errors=[
                        f"CORRUPT_RUNTIME_LOCK_METADATA: Metadados do lock operacional ausentes, corrompidos ou estruturalmente inválidos ({lock_meta_err}). "
                        "Reconciliação abortada fail-closed. Recovery obrigatório antes de prosseguir."
                    ],
                    details={
                        "cause": "CORRUPT_RUNTIME_LOCK_METADATA",
                        "lock_error": lock_meta_err,
                    },
                )

        # Preflight financeiro estrito
        if not p_fallback.exists() or not p_fallback.is_file():
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="PREFLIGHT",
                errors=[f"Arquivo fallback de origem não encontrado em {p_fallback}"],
            )

        reconciler = FallbackReconciler(
            sqlite_path=p_sqlite,
            jsonl_path=p_jsonl,
            archive_dir=p_archive,
            repo_root=p_repo,
        )
        if reconciler.is_git_tracked_fixture(p_fallback):
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="PREFLIGHT",
                errors=[f"Arquivo fallback '{p_fallback.name}' é uma fixture versionada no Git. Modificação proibida."],
            )

        if not p_manifest.exists() or p_manifest.stat().st_size == 0:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="PREFLIGHT",
                errors=[f"Manifesto operacional ausente ou vazio em {p_manifest} (ambiente sem bootstrap)."],
            )

        manifest_data = read_runtime_manifest(p_manifest)
        if not manifest_data:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="PREFLIGHT",
                errors=["Falha ao ler manifesto operacional."],
            )

        is_manifest_valid, manifest_err = validate_manifest_structure(manifest_data)
        if not is_manifest_valid:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="PREFLIGHT",
                errors=[f"Manifesto operacional com estrutura inválida: {manifest_err}"],
            )

        if str(manifest_data.get("environment_uuid", "")).strip() != clean_env_uuid:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="PREFLIGHT",
                errors=[f"Divergência de UUID: manifesto='{manifest_data.get('environment_uuid')}', transação='{clean_env_uuid}'."],
            )

        # Consistência do histórico H2 e ledger
        consistency_h2 = verify_manifest_reconciliation_consistency(
            manifest_data=manifest_data, ledger_path=p_ledger, expected_env_uuid=clean_env_uuid
        )
        if not consistency_h2["is_valid"]:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="PREFLIGHT",
                errors=[f"Inconsistência prévia no histórico H2 de reconciliação: {consistency_h2['error']}"],
            )

        ledger_val = validate_reconciliation_audit_ledger(p_ledger, expected_environment_uuid=clean_env_uuid)
        if ledger_val.get("has_pending_transaction"):
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="PREFLIGHT",
                errors=["Existe transação PREPARED pendente no ledger. Recuperação obrigatória antes de novo apply."],
            )
        if ledger_val.get("exists") and not ledger_val.get("is_valid"):
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="PREFLIGHT",
                errors=[f"Livro-razão de reconciliação inválido ou truncado: {ledger_val['error']}"],
            )

        if p_sqlite.exists():
            sq_ok, sq_msg = verify_sqlite_integrity(p_sqlite)
            if not sq_ok:
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                    stage_reached="PREFLIGHT",
                    errors=[f"Integridade física do SQLite comprometida: {sq_msg}"],
                )

        if p_jsonl.exists():
            jl_ok, jl_msg = verify_jsonl_integrity(p_jsonl)
            if not jl_ok:
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                    stage_reached="PREFLIGHT",
                    errors=[f"Integridade do JSONL ativo comprometida: {jl_msg}"],
                )

        if p_archive.exists():
            arch_ok, arch_msg = verify_archive_integrity(p_archive)
            if not arch_ok:
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                    stage_reached="PREFLIGHT",
                    errors=[f"Integridade do diretório archive comprometida: {arch_msg}"],
                )

        # Leitura e validação dos eventos do fallback
        all_events: List[Dict[str, Any]] = []
        seen_batch_ids: Set[str] = set()

        try:
            with open(p_fallback, "r", encoding="utf-8", errors="replace") as f_fb:
                line_idx = 0
                for line in f_fb:
                    line_idx += 1
                    raw_s = line.strip()
                    if not raw_s:
                        continue
                    try:
                        ev_obj = json.loads(raw_s)
                    except Exception as decode_err:
                        return ReconciliationTransactionResult(
                            audit_id=clean_audit_id,
                            environment_uuid=clean_env_uuid,
                            status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                            stage_reached="EVENT_VALIDATION",
                            errors=[f"JSON inválido no arquivo fallback linha {line_idx}: {decode_err}"],
                        )

                    if not isinstance(ev_obj, dict):
                        return ReconciliationTransactionResult(
                            audit_id=clean_audit_id,
                            environment_uuid=clean_env_uuid,
                            status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                            stage_reached="EVENT_VALIDATION",
                            errors=[f"Linha {line_idx} do arquivo fallback não é um dicionário."],
                        )

                    raw_id = ev_obj.get("event_id") or ev_obj.get("id")
                    clean_id = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() else ""
                    if not clean_id:
                        return ReconciliationTransactionResult(
                            audit_id=clean_audit_id,
                            environment_uuid=clean_env_uuid,
                            status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                            stage_reached="EVENT_VALIDATION",
                            errors=[f"Evento na linha {line_idx} não possui event_id válido."],
                        )

                    if clean_id in seen_batch_ids:
                        return ReconciliationTransactionResult(
                            audit_id=clean_audit_id,
                            environment_uuid=clean_env_uuid,
                            status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                            stage_reached="EVENT_VALIDATION",
                            errors=[f"event_id duplicado no lote do fallback na linha {line_idx}: {clean_id}"],
                        )

                    try:
                        compute_canonical_payload_hash(ev_obj)
                    except Exception as h_err:
                        return ReconciliationTransactionResult(
                            audit_id=clean_audit_id,
                            environment_uuid=clean_env_uuid,
                            status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                            stage_reached="EVENT_VALIDATION",
                            errors=[f"Falha ao calcular hash canônico no evento {clean_id}: {h_err}"],
                        )

                    seen_batch_ids.add(clean_id)
                    all_events.append(ev_obj)

        except Exception as read_fb_err:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="EVENT_VALIDATION",
                errors=[f"Erro ao ler arquivo de fallback: {read_fb_err}"],
            )

        if len(all_events) == 0:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="EVENT_VALIDATION",
                errors=["Lote de eventos de fallback está vazio. Transação rejeitada."],
            )

        # Classificação e destinação prévia
        sq_by_id, sq_by_corr = reconciler.load_sqlite_catalog()
        jl_by_id, jl_by_corr = reconciler.load_jsonl_catalog()

        to_insert_sqlite: List[Dict[str, Any]] = []
        to_append_jsonl: List[Dict[str, Any]] = []
        already_present_count = 0

        for idx, ev in enumerate(all_events, start=1):
            rec = reconciler.classify_event(
                event=ev,
                source_file=str(p_fallback),
                line_num=idx,
                is_fixture=False,
                raw_error=None,
                sqlite_catalog_by_id=sq_by_id,
                sqlite_catalog_by_corr=sq_by_corr,
                jsonl_catalog_by_id=jl_by_id,
                jsonl_catalog_by_corr=jl_by_corr,
            )

            if rec.classification in (
                EventClassification.EVENTO_INVALIDO,
                EventClassification.EVENTO_SEM_ID,
                EventClassification.EVENTO_ID_COLISAO,
                EventClassification.EVENTO_IDENTIDADE_AMBIGUA,
                EventClassification.FIXTURE_BLOQUEADA,
            ):
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                    stage_reached="EVENT_CLASSIFICATION",
                    total_events=len(all_events),
                    errors=[f"Evento {rec.event_id or f'linha {idx}'} rejeitado com classificação impeditiva: {rec.classification.value}. Motivo: {rec.reason}"],
                )

            if rec.classification == EventClassification.EVENTO_JA_PRESENTE:
                already_present_count += 1
            elif rec.classification == EventClassification.EVENTO_AUSENTE:
                if not rec.in_sqlite:
                    to_insert_sqlite.append(ev)
                if not (rec.in_jsonl or rec.in_archive):
                    to_append_jsonl.append(ev)

        # Rota específica de NO-OP: todos os eventos já presentes
        if len(to_insert_sqlite) == 0 and len(to_append_jsonl) == 0:
            verif_res = verify_reconciliation_targets(
                expected_events=all_events,
                sqlite_path=p_sqlite,
                jsonl_path=p_jsonl,
                archive_dir=p_archive,
                require_all_destinations_default=True,
            )
            if not verif_res.approved:
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                    stage_reached="NOOP_VERIFICATION",
                    total_events=len(all_events),
                    already_present=already_present_count,
                    verification_result=verif_res,
                    errors=[f"Pós-verificação de no-op reprovada com código {verif_res.code}: {verif_res.errors}"],
                )

            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.NO_CHANGES,
                stage_reached="NOOP_VERIFIED",
                total_events=len(all_events),
                already_present=len(all_events),
                sqlite_inserted=0,
                jsonl_appended=0,
                verification_result=verif_res,
                snapshot_dir=None,
                snapshot_manifest_sha256=None,
                prepared_recorded=False,
                manifest_updated=False,
                committed_recorded=False,
                errors=[],
                details={"reason": "ALL_EVENTS_ALREADY_PRESENT_VERIFIED"},
            )

        # Criação do snapshot forense
        try:
            snapshot_dir, snapshot_manifest_sha256 = create_reconciliation_snapshot(
                data_dir=p_data,
                lock_path=active_token.lock_path,
                backup_base_dir=p_backup,
                env_uuid=clean_env_uuid,
                operator_reason=clean_reason,
                source_fallback_file=p_fallback,
                archive_dir=p_archive,
                repo_root=p_repo,
                lock_file_obj=getattr(operational_lock, "_file_obj", None),
            )
        except Exception as snap_err:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.FAILED_BEFORE_PREPARED,
                stage_reached="SNAPSHOT",
                total_events=len(all_events),
                already_present=already_present_count,
                errors=[f"Falha na criação do snapshot forense: {snap_err}"],
            )

        # Gravação de PREPARED no ledger
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        prepared_entry = {
            "audit_id": clean_audit_id,
            "environment_uuid": clean_env_uuid,
            "timestamp_utc": timestamp_utc,
            "reconciliation_phase": "PREPARED",
            "action": "PREPARED",
            "operator_reason": clean_reason,
            "source_file": str(p_fallback),
            "snapshot_directory": str(snapshot_dir),
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "events_count": len(all_events),
            "planned_destinations": ["sqlite", "jsonl"],
        }
        try:
            record_reconciliation_audit(p_ledger, prepared_entry)
        except Exception as prep_err:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN,
                stage_reached="PREPARED",
                total_events=len(all_events),
                already_present=already_present_count,
                snapshot_dir=str(snapshot_dir),
                snapshot_manifest_sha256=snapshot_manifest_sha256,
                prepared_recorded=None,
                errors=[f"PREPARED_WRITE_UNCERTAIN: Falha crítica durante tentativa de persistir PREPARED no ledger: {prep_err}"],
                details={
                    "cause": "PREPARED_WRITE_UNCERTAIN",
                    "exception": str(prep_err),
                    "ledger_path": str(p_ledger),
                },
            )

        # Início do APPLY (Ordem obrigatória: 1. SQLite, 2. JSONL)
        sqlite_inserted = 0
        for ev in to_insert_sqlite:
            try:
                res_sq = insert_event_to_sqlite(p_sqlite, ev)
                if res_sq == "INSERTED":
                    sqlite_inserted += 1
                elif res_sq == "ALREADY_PRESENT":
                    already_present_count += 1
            except Exception as sq_err:
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.PREPARED_INCOMPLETE,
                    stage_reached="APPLY_SQLITE",
                    total_events=len(all_events),
                    sqlite_inserted=sqlite_inserted,
                    already_present=already_present_count,
                    snapshot_dir=str(snapshot_dir),
                    snapshot_manifest_sha256=snapshot_manifest_sha256,
                    prepared_recorded=True,
                    errors=[f"Falha durante INSERT no SQLite para evento {ev.get('event_id') or ev.get('id')}: {sq_err}"],
                )

        jsonl_appended = 0
        prefix_sep = validate_and_prepare_jsonl_target(p_jsonl)
        for ev in to_append_jsonl:
            try:
                append_event_to_jsonl(p_jsonl, ev, prefix_separator=prefix_sep)
                prefix_sep = ""
                jsonl_appended += 1
            except Exception as jl_err:
                return ReconciliationTransactionResult(
                    audit_id=clean_audit_id,
                    environment_uuid=clean_env_uuid,
                    status=ReconciliationTransactionStatus.PREPARED_INCOMPLETE,
                    stage_reached="APPLY_JSONL",
                    total_events=len(all_events),
                    sqlite_inserted=sqlite_inserted,
                    jsonl_appended=jsonl_appended,
                    already_present=already_present_count,
                    snapshot_dir=str(snapshot_dir),
                    snapshot_manifest_sha256=snapshot_manifest_sha256,
                    prepared_recorded=True,
                    errors=[f"Falha durante append no JSONL para evento {ev.get('event_id') or ev.get('id')}: {jl_err}"],
                )

        # Pós-verificação financeira independente
        verif_res = verify_reconciliation_targets(
            expected_events=all_events,
            sqlite_path=p_sqlite,
            jsonl_path=p_jsonl,
            archive_dir=p_archive,
            require_all_destinations_default=True,
        )
        if not verif_res.approved:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.PREPARED_INCOMPLETE,
                stage_reached="POST_VERIFICATION",
                total_events=len(all_events),
                sqlite_inserted=sqlite_inserted,
                jsonl_appended=jsonl_appended,
                already_present=already_present_count,
                verification_result=verif_res,
                snapshot_dir=str(snapshot_dir),
                snapshot_manifest_sha256=snapshot_manifest_sha256,
                prepared_recorded=True,
                errors=[f"Pós-verificação financeira rejeitou a reconciliação com código {verif_res.code}: {verif_res.errors}"],
            )

        # Atualização do manifesto operacional (versão 1.2 com proteção P0)
        try:
            update_runtime_manifest_reconciliation_state(
                manifest_path=p_manifest,
                audit_id=clean_audit_id,
                audit_timestamp_utc=timestamp_utc,
                expected_env_uuid=clean_env_uuid,
                ledger_path=p_ledger,
                lock_authorization=active_token,
            )
        except Exception as man_err:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.PREPARED_INCOMPLETE,
                stage_reached="MANIFEST_UPDATE",
                total_events=len(all_events),
                sqlite_inserted=sqlite_inserted,
                jsonl_appended=jsonl_appended,
                already_present=already_present_count,
                verification_result=verif_res,
                snapshot_dir=str(snapshot_dir),
                snapshot_manifest_sha256=snapshot_manifest_sha256,
                prepared_recorded=True,
                manifest_updated=False,
                errors=[f"Falha ao atualizar manifesto operacional: {man_err}"],
            )

        # Gravação de COMMITTED no ledger
        committed_entry = {
            "audit_id": clean_audit_id,
            "environment_uuid": clean_env_uuid,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reconciliation_phase": "COMMITTED",
            "action": "COMMITTED",
            "operator_reason": clean_reason,
            "source_file": str(p_fallback),
            "snapshot_directory": str(snapshot_dir),
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "events_count": len(all_events),
            "sqlite_inserted": sqlite_inserted,
            "jsonl_appended": jsonl_appended,
            "already_present": already_present_count,
        }
        try:
            record_reconciliation_audit(p_ledger, committed_entry)
        except Exception as com_err:
            return ReconciliationTransactionResult(
                audit_id=clean_audit_id,
                environment_uuid=clean_env_uuid,
                status=ReconciliationTransactionStatus.PERSISTENCE_UNCERTAIN,
                stage_reached="COMMITTED",
                total_events=len(all_events),
                sqlite_inserted=sqlite_inserted,
                jsonl_appended=jsonl_appended,
                already_present=already_present_count,
                verification_result=verif_res,
                snapshot_dir=str(snapshot_dir),
                snapshot_manifest_sha256=snapshot_manifest_sha256,
                prepared_recorded=True,
                manifest_updated=True,
                committed_recorded=None,
                errors=[f"Falha crítica ao gravar COMMITTED após manifesto atualizado (PERSISTENCE_UNCERTAIN): {com_err}"],
            )

        return ReconciliationTransactionResult(
            audit_id=clean_audit_id,
            environment_uuid=clean_env_uuid,
            status=ReconciliationTransactionStatus.SUCCESS,
            stage_reached="COMMITTED",
            total_events=len(all_events),
            sqlite_inserted=sqlite_inserted,
            jsonl_appended=jsonl_appended,
            already_present=already_present_count,
            verification_result=verif_res,
            snapshot_dir=str(snapshot_dir),
            snapshot_manifest_sha256=snapshot_manifest_sha256,
            prepared_recorded=True,
            manifest_updated=True,
            committed_recorded=True,
            errors=[],
        )

    finally:
        if we_acquired_lock and managed_lock is not None:
            try:
                managed_lock.release_maintenance()
            except Exception:
                pass
