# database/event_store.py
import sqlite3
import json
import time
import logging
import asyncio
import errno
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass

# Otimização de eventos (auto-adicionado)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import math

# ETAPA 6: NaN/±Inf -> None antes de serializar (JSON RFC 8259)
from common.json_safe import sanitize_json_safe
from common.financial_validator import (
    validate_financial_event_semantics,
    FinancialValidationResult,
)
from common.identity_contract import (
    canonicalize_structure,
    normalize_last_touched,
    IdentityContractError,
)
from common.financial_quarantine import FinancialQuarantine


class EventStoreError(Exception):
    """Exceção base para erros operacionais do EventStore."""
    pass


class IntegrityCollisionError(EventStoreError):
    """Detectada colisão divergente para o mesmo event_id (no batch ou no SQLite)."""

    def __init__(
        self,
        message: str,
        identity_id: Optional[str] = None,
        incoming_payload: Optional[Any] = None,
        existing_payload: Optional[Any] = None,
    ):
        super().__init__(message)
        self.identity_id = identity_id
        self.incoming_payload = incoming_payload
        self.existing_payload = existing_payload
        self.event_id = identity_id
        self.raw_payload = incoming_payload


class PreexistingDuplicateError(EventStoreError):
    """Detectadas múltiplas ocorrências físicas pré-existentes para o mesmo event_id no SQLite."""
    pass


class CorruptedHistoricalRecordError(EventStoreError):
    """Registro histórico no SQLite contém JSON inválido ou corrompido que impede comparação segura."""
    pass


@dataclass(frozen=True)
class BatchSaveResult:
    """
    Contrato imutável de resultado da operação de persistência em batch (Gate G2 / H1-B1).

    Status explícito:
      - COMMITTED: Transação concluída com confirmação.
      - FAILED: Falha comprovada sem commit bem-sucedido.
      - UNKNOWN: Resultado da transação não pôde ser determinado com segurança.
    """
    total_received: int
    persisted: int
    rejected_pre_commit: int
    commit_status: str  # "COMMITTED" | "FAILED" | "UNKNOWN"
    error_type: str | None = None
    error_message: str | None = None
    duplicates_skipped: int = 0
    rows_inserted: int = 0


def _extract_event_identity(e: Dict[str, Any]) -> Tuple[Optional[str], bool, bool]:
    """
    Extrai determinística e conservadoramente a identidade do evento (Gate H1-B1).
    Retorna: (identity_id, is_valid, is_legacy)
    - identity_id: string com o ID de negócio normalizado, ou None se evento anônimo.
    - is_valid: False se houver ambiguidade, tipo inválido ou conflito interno.
    - is_legacy: True se a identidade provém do campo legado 'id'.
    """
    raw_event_id = e.get("event_id")
    raw_id = e.get("id")

    # 1. Validação de event_id explícito
    if raw_event_id is not None:
        if isinstance(raw_event_id, bool):
            return None, False, False
        if isinstance(raw_event_id, (int, float)):
            event_id_str = str(raw_event_id).strip()
        elif isinstance(raw_event_id, str):
            event_id_str = raw_event_id.strip()
        else:
            return None, False, False

        if not event_id_str:
            return None, False, False

        # Se houver também 'id', verificar conformidade
        if raw_id is not None:
            if isinstance(raw_id, bool):
                return None, False, False
            # Se 'id' for numérico (ex: id relacional autoincrement do SQLite), sem conflito
            if isinstance(raw_id, (int, float)):
                return event_id_str, True, False

            raw_id_str = str(raw_id).strip()
            if raw_id_str and raw_id_str != event_id_str:
                # Se event_id foi gerado pelo EventSaver (uuid4()[:8]), preserva compatibilidade retroativa
                if len(event_id_str) == 8 and all(c in "0123456789abcdefABCDEF" for c in event_id_str):
                    return event_id_str, True, False
                # Dois IDs de negócio fornecidos externamente divergentes -> conflito explícito (fail-closed)
                return None, False, False

        return event_id_str, True, False

    # 2. Caso event_id ausente: avaliar campo legado 'id'
    if raw_id is not None:
        if isinstance(raw_id, bool):
            return None, False, False
        # Se for numérico, NÃO promovemos para identidade de negócio (pode ser ID relacional SQLite)
        if isinstance(raw_id, (int, float)):
            return None, True, False  # Válido, mas tratado como anônimo sem deduplicação por ID

        raw_id_str = str(raw_id).strip()
        if not raw_id_str:
            return None, False, False

        # Se for string puramente numérica (ex: "123"), não promovemos para evitar colisão com ID técnico relacional
        if raw_id_str.isdigit():
            return None, True, False  # Válido, mas anônimo

        # String não-numérica legada (ex: "legacy_order_123") -> aceita como identidade legada
        return raw_id_str, True, True

    # 3. Nenhum identificador fornecido (evento anônimo / testes de integridade legados)
    return None, True, False


class EventStore:
    """
    Gerencia a persistência de eventos de trading usando SQLite.
    Substitui o armazenamento baseado em arquivos JSON/JSONL.

    Características:
    - Modo WAL (Write-Ahead Logging) para alta concorrência.
    - Escrita em batch (executemany) para performance.
    - Índices otimizados para busca por tempo e tipo.
    - Suporte nativo para chamadas assíncronas (via asyncio.to_thread).
    """

    def __init__(
        self,
        db_path: str = "dados/trading_bot.db",
        quarantine: Optional[FinancialQuarantine] = None,
        quarantine_dir: Optional[Any] = None,
        enforce_financial_validation: bool = True,
    ):
        """
        Inicializa o EventStore (Gate H1-B2-Q2).

        Args:
            db_path: Caminho para o arquivo do banco de dados.
            quarantine: Instância de FinancialQuarantine para quarentenar eventos inválidos.
            quarantine_dir: Diretório alternativo para inicializar FinancialQuarantine se não fornecida.
            enforce_financial_validation: Se True, aplica validação semântica fail-closed pré-commit.
        """
        # Usa caminho absoluto para evitar problemas se o processo mudar o cwd
        # (sqlite3.resolve caminhos relativos no momento do connect).
        try:
            self.db_path = Path(db_path).expanduser().resolve()
        except Exception:
            self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)

        self.quarantine = quarantine
        if self.quarantine is None and quarantine_dir is not None:
            self.quarantine = FinancialQuarantine(quarantine_dir=Path(quarantine_dir))
        self.enforce_financial_validation = enforce_financial_validation
        self._quarantined_invalid_count = 0
        self._quarantined_collision_count = 0
        self._enospc_error_count = 0

        # Garante que o diretório existe
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Falha ao criar diretório para DB {self.db_path}: {e}")

        self._init_db()

    def _validate_financial_payload(self, event: Dict[str, Any]) -> FinancialValidationResult:
        """
        Valida o evento semanticamente contra regras financeiras puras (Gate H1-B2-Q2).
        - Para eventos classificados (Absorção, Exaustão, Zona, Trigger, Alerta, AI):
          aplica regras completas de validate_financial_event_semantics.
        - Para eventos não classificados (ex: testes legados sem tipo de sinal):
          garante fail-closed contra valores monetários não finitos (NaN, ±Inf),
          booleanos em campos de preço/volume e valores negativos em campos estritos.
        """
        val_res = validate_financial_event_semantics(event)
        if val_res.status == "VALID":
            # Validação defensiva complementar de limites de microestrutura / Zonas (top/bottom)
            for bound_key in ("top", "bottom"):
                if bound_key in event:
                    v = event[bound_key]
                    if isinstance(v, bool):
                        return FinancialValidationResult(
                            is_valid=False,
                            status="INVALID",
                            category=val_res.category,
                            rejected_field=bound_key,
                            rejected_value_repr=str(v),
                            rejection_reason="BOOLEAN_NOT_ALLOWED_IN_ZONE_BOUNDS",
                        )
                    if isinstance(v, (int, float)):
                        if math.isnan(v):
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category=val_res.category,
                                rejected_field=bound_key,
                                rejected_value_repr="nan",
                                rejection_reason="NAN_IN_ZONE_BOUNDS",
                            )
                        if math.isinf(v):
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category=val_res.category,
                                rejected_field=bound_key,
                                rejected_value_repr=str(v),
                                rejection_reason="INFINITY_IN_ZONE_BOUNDS",
                            )
                        if v <= 0:
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category=val_res.category,
                                rejected_field=bound_key,
                                rejected_value_repr=str(v),
                                rejection_reason="ZONE_BOUNDS_MUST_BE_POSITIVE",
                            )
            # Verificação de inversão de limites em Zonas (bottom <= top)
            if "top" in event and "bottom" in event:
                try:
                    top_v = float(event["top"])
                    bot_v = float(event["bottom"])
                    if not (math.isnan(top_v) or math.isnan(bot_v) or math.isinf(top_v) or math.isinf(bot_v)):
                        if bot_v > top_v:
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category=val_res.category,
                                rejected_field="bottom",
                                rejected_value_repr=f"bottom={bot_v} > top={top_v}",
                                rejection_reason="INVALID_MICROSTRUCTURE_BOUNDS: bottom > top",
                            )
                except (ValueError, TypeError):
                    pass

            # Verificação de inversão de Market Profile (val <= poc <= vah)
            if "poc" in event and "vah" in event:
                try:
                    poc_v = float(event["poc"])
                    vah_v = float(event["vah"])
                    if not (math.isnan(poc_v) or math.isnan(vah_v) or math.isinf(poc_v) or math.isinf(vah_v)):
                        if poc_v > vah_v:
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category=val_res.category,
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
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category=val_res.category,
                                rejected_field="val",
                                rejected_value_repr=f"val={val_v} > poc={poc_v}",
                                rejection_reason="INVALID_MICROSTRUCTURE_BOUNDS: val > poc",
                            )
                except (ValueError, TypeError):
                    pass

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
            # Se a única razão de rejeição for a ausência de campo de preço em eventos legados/parciais de teste,
            # garante validação defensiva dos campos existentes sem barrar eventos válidos que não carregam preço
            if val_res.rejection_reason == "MANDATORY_FIELD_MISSING":
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
            # UNCLASSIFIED: Inspecionar campos de preço/volume para impedir NaN/±Inf silenciosos
            for price_key in ("price", "preco", "preco_fechamento", "entry_price", "anchor_price", "level", "poc", "val", "vah"):
                if price_key in event:
                    v = event[price_key]
                    if isinstance(v, bool):
                        return FinancialValidationResult(
                            is_valid=False,
                            status="INVALID",
                            category="UNCLASSIFIED_FINANCIAL",
                            rejected_field=price_key,
                            rejected_value_repr=str(v),
                            rejection_reason="BOOLEAN_NOT_ALLOWED_IN_PRICE",
                        )
                    if isinstance(v, (int, float)):
                        if math.isnan(v):
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=price_key,
                                rejected_value_repr="nan",
                                rejection_reason="NAN_IN_PRICE_FIELD",
                            )
                        if math.isinf(v):
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=price_key,
                                rejected_value_repr=str(v),
                                rejection_reason="INFINITY_IN_PRICE_FIELD",
                            )
                        if v <= 0:
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=price_key,
                                rejected_value_repr=str(v),
                                rejection_reason="PRICE_MUST_BE_POSITIVE",
                            )
            for num_key in ("delta", "volume", "volume_total", "quantity", "quantidade"):
                if num_key in event:
                    v = event[num_key]
                    if isinstance(v, bool):
                        return FinancialValidationResult(
                            is_valid=False,
                            status="INVALID",
                            category="UNCLASSIFIED_FINANCIAL",
                            rejected_field=num_key,
                            rejected_value_repr=str(v),
                            rejection_reason="BOOLEAN_NOT_ALLOWED_IN_NUMERIC_FIELD",
                        )
                    if isinstance(v, (int, float)):
                        if math.isnan(v):
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=num_key,
                                rejected_value_repr="nan",
                                rejection_reason="NAN_IN_NUMERIC_FIELD",
                            )
                        if math.isinf(v):
                            return FinancialValidationResult(
                                is_valid=False,
                                status="INVALID",
                                category="UNCLASSIFIED_FINANCIAL",
                                rejected_field=num_key,
                                rejected_value_repr=str(v),
                                rejection_reason="INFINITY_IN_NUMERIC_FIELD",
                            )
            return FinancialValidationResult(
                is_valid=True,
                status="VALID",
                category="UNCLASSIFIED_PERMITTED"
            )

    def _get_conn(self) -> sqlite3.Connection:
        """
        Retorna uma conexão configurada para performance.
        check_same_thread=False permite uso em multithreading (com cuidado).
        """
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

        # Otimizações críticas para performance
        conn.execute("PRAGMA journal_mode=WAL;")   # melhora concorrência leitura/escrita
        conn.execute("PRAGMA synchronous=NORMAL;") # reduz fsyncs excessivos (seguro em WAL)
        conn.execute("PRAGMA busy_timeout=5000;")  # espera até 5s se o banco estiver ocupado

        return conn

    def _init_db(self):
        """Cria a tabela de eventos e índices se não existirem."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_ms INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        symbol TEXT,
                        window_id TEXT,
                        is_signal BOOLEAN DEFAULT 0,
                        payload JSON NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp_ms);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_events_window ON events(window_id);"
                )

                # Tabela de outcomes dos sinais (criada aqui para garantir existência atômica)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS signal_outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_epoch_ms INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        battle_result TEXT,
                        entry_price REAL NOT NULL,
                        symbol TEXT DEFAULT 'BTCUSDT',
                        context_json TEXT,
                        outcome_5m_pct REAL,
                        outcome_15m_pct REAL,
                        outcome_30m_pct REAL,
                        outcome_60m_pct REAL,
                        outcome_direction_5m TEXT,
                        outcome_direction_15m TEXT,
                        outcome_direction_30m TEXT,
                        outcome_direction_60m TEXT,
                        evaluated_at INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_outcomes_type ON signal_outcomes(event_type);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_outcomes_battle ON signal_outcomes(battle_result);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_outcomes_epoch ON signal_outcomes(signal_epoch_ms);"
                )

        except Exception as e:
            self.logger.critical(f"Erro fatal ao inicializar banco de dados: {e}")
            raise

    def save_event(self, event: Dict[str, Any]) -> BatchSaveResult:
        """Salva um único evento no banco (Síncrono)."""
        return self.save_batch([event])

    async def save_event_async(self, event: Dict[str, Any]):
        """Salva um único evento no banco (Assíncrono)."""
        return await asyncio.to_thread(self.save_event, event)

    def _determine_failure_status(self, conn: Any, stage: str) -> str:
        """
        Determina determinística e conservadoramente se a falha é FAILED ou UNKNOWN (Gate G2 / H1-B1).

        Regras:
          1. Falha em CONNECTING ou BEGINNING: transação não iniciada -> FAILED comprovado.
          2. Falha em COMMITTING ou COMMITTING_RETRY: commit iniciado mas resultado físico incerto -> UNKNOWN.
          3. Falha em QUERYING, VALIDATING, EXECUTING, EXECUTING_RETRY:
             - Se conn.in_transaction: tenta rollback explícito.
             - Se rollback concluído com sucesso -> FAILED comprovado (transação revertida).
             - Se rollback falhar ou levantar exceção -> UNKNOWN (estado transacional não comprovado).
          4. Qualquer outro estado ou objeto inválido -> UNKNOWN.
        """
        if stage in ("CONNECTING", "BEGINNING", "PRE_COMMIT", "CONNECTING_RETRY"):
            return "FAILED"
        if stage in ("COMMITTING", "COMMITTING_RETRY"):
            return "UNKNOWN"
        if stage in ("QUERYING", "VALIDATING", "EXECUTING", "EXECUTING_RETRY"):
            if conn is not None:
                try:
                    if getattr(conn, "in_transaction", False):
                        conn.rollback()
                    return "FAILED"
                except Exception as rb_err:
                    self.logger.error(f"Falha ao executar rollback no SQLite: {rb_err}")
                    return "UNKNOWN"
            return "UNKNOWN"
        return "UNKNOWN"

    def save_batch(self, events: List[Dict[str, Any]]) -> BatchSaveResult:
        """
        Salva uma lista de eventos com idempotência transacional estrita (Gate H1-B1).

        Garante:
        - Atomicidade absoluta com BEGIN IMMEDIATE na mesma conexão/transação.
        - Deduplicação determinística por identidade de negócio (event_id).
        - Consulta no SQLite por identidade independente de timestamp.
        - Detecção e falha integral (fail-closed) para colisões divergentes (no batch ou no banco).
        - Contabilidade precisa: persisted = rows_inserted + duplicates_skipped.
        - Invariante G2: persisted + rejected_pre_commit == total_received.
        - Reversão segura em rollback (FAILED) ou incerteza em commit (UNKNOWN).
        """
        if not events:
            return BatchSaveResult(
                total_received=0,
                persisted=0,
                rejected_pre_commit=0,
                commit_status="COMMITTED",
                error_type=None,
                error_message=None,
                duplicates_skipped=0,
                rows_inserted=0,
            )

        from events.fallback_reconciler import compute_canonical_payload_hash

        total_received = len(events)
        prepared_items = []
        rejected_pre_commit = 0

        # Etapa A: Validação e preparação integral pré-commit
        for e in events:
            if not isinstance(e, dict):
                rejected_pre_commit += 1
                continue

            # Preserva cópia profunda do payload original bruto antes de qualquer mutação
            raw_payload_copy = copy.deepcopy(e)

            # Validação financeira semântica integrada (Gate H1-B2-Q2)
            if self.enforce_financial_validation:
                val_res = self._validate_financial_payload(e)
                if not val_res.is_valid:
                    rejected_pre_commit += 1
                    self.logger.warning(
                        f"Evento rejeitado pré-commit pelo validador financeiro: {val_res.rejection_reason} "
                        f"(campo: {val_res.rejected_field}, valor: {val_res.rejected_value_repr})"
                    )
                    if self.quarantine is not None:
                        try:
                            self._quarantined_invalid_count += 1
                            self.quarantine.quarantine_event(
                                raw_payload_copy,
                                val_res,
                                source_caller="EventStore.save_batch",
                            )
                        except Exception as q_err:
                            self.logger.error(f"Falha ao registrar quarentena para evento inválido: {q_err}")
                    continue

            # Canonicalização determinística de -0.0 e estruturas (Gate H1-B3-D)
            try:
                canonical_e = canonicalize_structure(e)
            except Exception as can_err:
                rejected_pre_commit += 1
                self.logger.error(f"Erro na canonicalização do evento: {can_err}")
                continue

            identity_id, is_valid_identity, is_legacy = _extract_event_identity(canonical_e)
            if not is_valid_identity:
                rejected_pre_commit += 1
                self.logger.error(
                    f"Evento rejeitado pré-commit devido à identidade inválida ou conflitante: {str(e)[:100]}"
                )
                if self.quarantine is not None:
                    try:
                        self._quarantined_invalid_count += 1
                        id_val_res = FinancialValidationResult(
                            is_valid=False,
                            status="INVALID",
                            category="MALFORMED_IDENTITY",
                            rejected_field="event_id",
                            rejected_value_repr=str(e.get("event_id")),
                            rejection_reason="INVALID_OR_CONFLICTING_IDENTITY",
                        )
                        self.quarantine.quarantine_event(
                            raw_payload_copy,
                            id_val_res,
                            source_caller="EventStore.save_batch",
                        )
                    except Exception as q_err:
                        self.logger.error(f"Falha ao registrar quarentena de identidade inválida: {q_err}")
                continue

            try:
                ts = int(canonical_e.get("epoch_ms") or canonical_e.get("timestamp_ms") or time.time() * 1000)
                etype = str(canonical_e.get("tipo_evento") or canonical_e.get("type", "UNKNOWN"))
                sym = str(canonical_e.get("ativo") or canonical_e.get("symbol", ""))
                wid = str(canonical_e.get("window_id", ""))
                is_sig_val = canonical_e.get("is_signal")
                is_sig = 1 if (is_sig_val is True or str(is_sig_val).lower() == "true") else 0

                sanitized = sanitize_json_safe(canonical_e)
                payload_str = json.dumps(sanitized, default=str)
                # Parse estruturado para hashing idêntico à leitura do SQLite
                payload_for_hash = json.loads(payload_str)
                canonical_hash = compute_canonical_payload_hash(payload_for_hash)

                prepared_items.append(
                    (ts, etype, sym, wid, is_sig, payload_str, identity_id, canonical_hash, is_legacy, raw_payload_copy)
                )
            except Exception as err:
                rejected_pre_commit += 1
                self.logger.error(
                    f"Erro ao preparar evento para salvar: {err} | Evento parcial: {str(e)[:100]}"
                )
                continue

        if not prepared_items:
            return BatchSaveResult(
                total_received=total_received,
                persisted=0,
                rejected_pre_commit=rejected_pre_commit,
                commit_status="FAILED",
                error_type="PreCommitValidationError",
                error_message="Todos os eventos foram rejeitados na preparação pré-commit",
                duplicates_skipped=0,
                rows_inserted=0,
            )

        # Etapa B & C: Deduplicação interna e detecção de colisões divergentes no batch
        seen_in_batch: Dict[str, Tuple[str, Any]] = {}
        unique_batch_events = []
        batch_internal_duplicates_count = 0

        try:
            for item in prepared_items:
                identity_id = item[6]
                canonical_hash = item[7]

                if identity_id is None:
                    # Evento anônimo (sem chave de negócio)
                    unique_batch_events.append(item)
                else:
                    if identity_id in seen_in_batch:
                        prev_hash, prev_item = seen_in_batch[identity_id]
                        if prev_hash != canonical_hash:
                            incoming_raw = item[9] if len(item) > 9 else item
                            prev_raw = prev_item[9] if len(prev_item) > 9 else prev_item
                            if self.quarantine is not None:
                                try:
                                    self._quarantined_collision_count += 1
                                    col_res = FinancialValidationResult(
                                        is_valid=False,
                                        status="INVALID",
                                        category="COLLISION",
                                        rejected_field="event_id",
                                        rejected_value_repr=str(identity_id),
                                        rejection_reason="INTEGRITY_COLLISION_DIVERGENT_BATCH_PAYLOAD",
                                    )
                                    self.quarantine.quarantine_event(
                                        incoming_raw,
                                        col_res,
                                        source_caller="EventStore.save_batch",
                                        category="COLLISION",
                                    )
                                except Exception as q_err:
                                    self.logger.error(f"Falha ao registrar colisão de batch na quarentena: {q_err}")
                            raise IntegrityCollisionError(
                                f"Colisão divergente de payload detectada dentro do mesmo batch para o ID '{identity_id}'",
                                identity_id=identity_id,
                                incoming_payload=incoming_raw,
                                existing_payload=prev_raw,
                            )
                        # Duplicata idêntica interna no mesmo batch
                        batch_internal_duplicates_count += 1
                    else:
                        seen_in_batch[identity_id] = (canonical_hash, item)
                        unique_batch_events.append(item)
        except IntegrityCollisionError as coll_err:
            self.logger.error(f"Falha no batch por colisão interna: {coll_err}")
            return BatchSaveResult(
                total_received=total_received,
                persisted=0,
                rejected_pre_commit=rejected_pre_commit,
                commit_status="FAILED",
                error_type=type(coll_err).__name__,
                error_message=str(coll_err),
                duplicates_skipped=0,
                rows_inserted=0,
            )

        insert_sql = """
            INSERT INTO events (
                timestamp_ms, event_type, symbol, window_id, is_signal, payload
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """

        conn = None
        stage = "CONNECTING"
        try:
            conn = self._get_conn()
            conn.isolation_level = None  # Autocommit desativado no Python para controle explícito de BEGIN/COMMIT

            stage = "BEGINNING"
            conn.execute("BEGIN IMMEDIATE")

            # Etapa D: Consulta de registros existentes no SQLite por identidade
            stage = "QUERYING"
            business_ids = [k for k, (_, item) in seen_in_batch.items() if not item[8]]
            legacy_ids = [k for k, (_, item) in seen_in_batch.items() if item[8]]

            existing_records: Dict[str, List[Tuple[int, str]]] = {}

            # Consulta por event_id de negócio
            for i in range(0, len(business_ids), 500):
                chunk = business_ids[i:i + 500]
                placeholders = ",".join("?" for _ in chunk)
                query = f"SELECT id, payload FROM events WHERE CAST(json_extract(payload, '$.event_id') AS TEXT) IN ({placeholders})"
                cur = conn.execute(query, chunk)
                for row_id, p_str in cur.fetchall():
                    try:
                        p_dict = json.loads(p_str)
                        eid = str(p_dict.get("event_id")).strip() if p_dict.get("event_id") is not None else None
                    except Exception as dec_err:
                        raise CorruptedHistoricalRecordError(
                            f"Payload JSON histórico inválido no SQLite (linha {row_id}): {dec_err}"
                        )
                    if eid:
                        existing_records.setdefault(eid, []).append((row_id, p_str))

            # Consulta por id legado
            for i in range(0, len(legacy_ids), 500):
                chunk = legacy_ids[i:i + 500]
                placeholders = ",".join("?" for _ in chunk)
                query = f"SELECT id, payload FROM events WHERE CAST(json_extract(payload, '$.id') AS TEXT) IN ({placeholders})"
                cur = conn.execute(query, chunk)
                for row_id, p_str in cur.fetchall():
                    try:
                        p_dict = json.loads(p_str)
                        lid = str(p_dict.get("id")).strip() if p_dict.get("id") is not None else None
                    except Exception as dec_err:
                        raise CorruptedHistoricalRecordError(
                            f"Payload JSON histórico inválido no SQLite (linha {row_id}): {dec_err}"
                        )
                    if lid:
                        existing_records.setdefault(lid, []).append((row_id, p_str))

            # Etapa E: Verificação de integridade, duplicatas pré-existentes e colisão divergente
            stage = "VALIDATING"
            already_in_db_identities: Set[str] = set()

            for identity_id, (batch_hash, _) in seen_in_batch.items():
                db_rows = existing_records.get(identity_id, [])
                if len(db_rows) > 1:
                    raise PreexistingDuplicateError(
                        f"Detectadas {len(db_rows)} duplicatas físicas pré-existentes no SQLite para o ID '{identity_id}'"
                    )
                if len(db_rows) == 1:
                    row_id, db_payload_str = db_rows[0]
                    try:
                        db_payload_dict = json.loads(db_payload_str)
                    except Exception as dec_err:
                        raise CorruptedHistoricalRecordError(
                            f"Payload JSON histórico inválido no SQLite para o ID '{identity_id}' (linha {row_id}): {dec_err}"
                        )
                    db_hash = compute_canonical_payload_hash(db_payload_dict)
                    if db_hash != batch_hash:
                        incoming_raw = seen_in_batch[identity_id][1][9] if len(seen_in_batch[identity_id][1]) > 9 else db_payload_dict
                        if self.quarantine is not None:
                            try:
                                self._quarantined_collision_count += 1
                                col_res = FinancialValidationResult(
                                    is_valid=False,
                                    status="INVALID",
                                    category="COLLISION",
                                    rejected_field="event_id",
                                    rejected_value_repr=str(identity_id),
                                    rejection_reason="INTEGRITY_COLLISION_DIVERGENT_HISTORICAL_RECORD",
                                )
                                self.quarantine.quarantine_event(
                                    incoming_raw,
                                    col_res,
                                    source_caller="EventStore.save_batch",
                                    category="COLLISION",
                                )
                            except Exception as q_err:
                                self.logger.error(f"Falha ao registrar colisão histórica na quarentena: {q_err}")
                        raise IntegrityCollisionError(
                            f"Colisão divergente com registro pré-existente no SQLite para o ID '{identity_id}' "
                            f"(banco: {db_hash[:12]} vs recebido: {batch_hash[:12]})",
                            identity_id=identity_id,
                            incoming_payload=incoming_raw,
                            existing_payload=db_payload_dict,
                        )
                    already_in_db_identities.add(identity_id)

            # Separação estrita de novas linhas a inserir
            events_to_insert = []
            db_already_present_count = 0

            for item in unique_batch_events:
                identity_id = item[6]
                if identity_id is not None and identity_id in already_in_db_identities:
                    db_already_present_count += 1
                else:
                    events_to_insert.append(item)

            duplicates_skipped = db_already_present_count + batch_internal_duplicates_count
            rows_inserted = len(events_to_insert)
            persisted = rows_inserted + duplicates_skipped

            # Etapa F: Inserção física atômica (apenas eventos novos)
            stage = "EXECUTING"
            if events_to_insert:
                insert_data = [
                    (x[0], x[1], x[2], x[3], x[4], x[5]) for x in events_to_insert
                ]
                conn.executemany(insert_sql, insert_data)

            # Etapa G: Confirmação da transação
            stage = "COMMITTING"
            conn.commit()
            stage = "COMMITTED"

            return BatchSaveResult(
                total_received=total_received,
                persisted=persisted,
                rejected_pre_commit=rejected_pre_commit,
                commit_status="COMMITTED",
                error_type=None,
                error_message=None,
                duplicates_skipped=duplicates_skipped,
                rows_inserted=rows_inserted,
            )

        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "malformed json" in msg:
                corrupted_err = CorruptedHistoricalRecordError(
                    f"Registro histórico no SQLite contém JSON malformado: {e}"
                )
                self.logger.error(f"Erro no SQLite ao inspecionar histórico JSON: {corrupted_err}")
                status = self._determine_failure_status(conn, stage)
                return BatchSaveResult(
                    total_received=total_received,
                    persisted=0,
                    rejected_pre_commit=rejected_pre_commit,
                    commit_status=status,
                    error_type="CorruptedHistoricalRecordError",
                    error_message=str(corrupted_err),
                    duplicates_skipped=0,
                    rows_inserted=0,
                )

            if "no such table" in msg and "events" in msg:
                try:
                    self.logger.warning(
                        "Tabela 'events' ausente no SQLite; recriando schema..."
                    )
                    self._init_db()
                    return self.save_batch(events)
                except Exception as e2:
                    self.logger.error(f"Erro ao recriar schema e regravar: {e2}")
                    status = self._determine_failure_status(conn, stage)
                    return BatchSaveResult(
                        total_received=total_received,
                        persisted=0,
                        rejected_pre_commit=rejected_pre_commit,
                        commit_status=status,
                        error_type=type(e2).__name__,
                        error_message=str(e2),
                        duplicates_skipped=0,
                        rows_inserted=0,
                    )

            if getattr(e, "errno", None) == errno.ENOSPC or "no space left on device" in str(e).lower() or "disk full" in str(e).lower():
                self._enospc_error_count += 1
            self.logger.error(f"Erro operacional ao gravar batch no SQLite: {e}")
            status = self._determine_failure_status(conn, stage)
            return BatchSaveResult(
                total_received=total_received,
                persisted=0,
                rejected_pre_commit=rejected_pre_commit,
                commit_status=status,
                error_type=type(e).__name__,
                error_message=str(e),
                duplicates_skipped=0,
                rows_inserted=0,
            )
        except Exception as e:
            self.logger.error(f"Erro ao processar batch no SQLite: {e}")
            status = self._determine_failure_status(conn, stage)
            return BatchSaveResult(
                total_received=total_received,
                persisted=0,
                rejected_pre_commit=rejected_pre_commit,
                commit_status=status,
                error_type=type(e).__name__,
                error_message=str(e),
                duplicates_skipped=0,
                rows_inserted=0,
            )
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception as close_err:
                    self.logger.debug(f"Erro ao fechar conexão SQLite no finally: {close_err}")

    async def save_batch_async(self, events: List[Dict[str, Any]]):
        """Salva uma lista de eventos de forma atômica (Assíncrono)."""
        return await asyncio.to_thread(self.save_batch, events)

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Recupera os últimos N eventos ordenados cronologicamente (Síncrono).
        Substitui a leitura do arquivo snapshot JSON.

        Returns:
            Lista de dicionários (os payloads originais).
        """
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT payload FROM events ORDER BY timestamp_ms DESC, id DESC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()
                return [json.loads(row[0]) for row in rows][::-1]
        except Exception as e:
            self.logger.error(f"Erro ao ler eventos recentes: {e}")
            return []

    async def get_recent_events_async(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Recupera os últimos N eventos (Assíncrono)."""
        return await asyncio.to_thread(self.get_recent_events, limit)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas básicas do banco (Síncrono)."""
        try:
            with self._get_conn() as conn:
                count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                last_ts_row = conn.execute("SELECT MAX(timestamp_ms) FROM events").fetchone()
                last_ts = last_ts_row[0] if last_ts_row else None
                db_size = (
                    self.db_path.stat().st_size if self.db_path.exists() else 0
                )

                return {
                    "total_events": count,
                    "last_timestamp": last_ts,
                    "db_size_bytes": db_size,
                    "db_path": str(self.db_path),
                    "quarantined_invalid_count": self._quarantined_invalid_count,
                    "quarantined_collision_count": self._quarantined_collision_count,
                    "enospc_error_count": self._enospc_error_count,
                }
        except Exception as e:
            return {"error": str(e)}

    def get_financial_integrity_metrics(self) -> Dict[str, Any]:
        """Retorna telemetria auditável de integridade financeira do EventStore (Gate G5)."""
        q_metrics = (
            self.quarantine.get_quarantine_metrics()
            if self.quarantine is not None and hasattr(self.quarantine, "get_quarantine_metrics")
            else {}
        )
        return {
            "quarantined_invalid_count": self._quarantined_invalid_count or q_metrics.get("quarantined_invalid_count", 0),
            "quarantined_collision_count": self._quarantined_collision_count or q_metrics.get("quarantined_collision_count", 0),
            "enospc_error_count": self._enospc_error_count or q_metrics.get("enospc_error_count", 0),
        }

    async def get_stats_async(self) -> Dict[str, Any]:
        """Retorna estatísticas básicas do banco (Assíncrono)."""
        return await asyncio.to_thread(self.get_stats)
