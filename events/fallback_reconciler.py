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

import hashlib
import json
import logging
import math
import sqlite3
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("events.fallback_reconciler")

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
    ):
        self.sqlite_path = Path(sqlite_path)
        self.jsonl_path = Path(jsonl_path)
        self.archive_dir = Path(archive_dir)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()

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
