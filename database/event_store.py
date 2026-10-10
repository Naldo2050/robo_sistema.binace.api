# database/event_store.py
import sqlite3
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Otimização de eventos (auto-adicionado)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ETAPA 6: NaN/±Inf -> None antes de serializar (JSON RFC 8259)
from common.json_safe import sanitize_json_safe


@dataclass(frozen=True)
class BatchSaveResult:
    """
    Contrato imutável de resultado da operação de persistência em batch (Gate G2).

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

    def __init__(self, db_path: str = "dados/trading_bot.db"):
        """
        Inicializa o EventStore.

        Args:
            db_path: Caminho para o arquivo do banco de dados.
        """
        # Usa caminho absoluto para evitar problemas se o processo mudar o cwd
        # (sqlite3.resolve caminhos relativos no momento do connect).
        try:
            self.db_path = Path(db_path).expanduser().resolve()
        except Exception:
            self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)

        # Garante que o diretório existe
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Falha ao criar diretório para DB {self.db_path}: {e}")

        self._init_db()

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
        Determina determinística e conservadoramente se a falha é FAILED ou UNKNOWN (Gate G2).

        Regras:
          1. Falha em CONNECTING: conexão não aberta -> FAILED comprovado (ausência de transação).
          2. Falha em COMMITTING: commit não pôde ser confirmado -> UNKNOWN.
          3. Falha em EXECUTING: tenta conn.rollback().
             - Se rollback for concluído sem exceção -> FAILED comprovado (transação revertida).
             - Se rollback falhar ou levantar exceção -> UNKNOWN (estado transacional não comprovado).
          4. Qualquer outro estado ou objeto inválido -> UNKNOWN.
        """
        if stage in ("CONNECTING", "CONNECTING_RETRY"):
            return "FAILED"
        if stage in ("COMMITTING", "COMMITTING_RETRY"):
            return "UNKNOWN"
        if stage in ("EXECUTING", "EXECUTING_RETRY"):
            if conn is not None:
                try:
                    conn.rollback()
                    return "FAILED"
                except Exception as rb_err:
                    self.logger.error(f"Falha ao executar rollback no SQLite: {rb_err}")
                    return "UNKNOWN"
            return "UNKNOWN"
        return "UNKNOWN"

    def save_batch(self, events: List[Dict[str, Any]]) -> BatchSaveResult:
        """
        Salva uma lista de eventos de forma atômica (transação única) - Síncrono (Gate G2).

        Args:
            events: Lista de dicionários contendo os dados dos eventos.

        Returns:
            BatchSaveResult com confirmação explícita do status de persistência.
        """
        if not events:
            return BatchSaveResult(
                total_received=0,
                persisted=0,
                rejected_pre_commit=0,
                commit_status="COMMITTED",
                error_type=None,
            )

        total_received = len(events)
        data = []
        rejected_pre_commit = 0
        for e in events:
            try:
                # Extração segura de campos para colunas indexadas
                ts = int(e.get("epoch_ms") or e.get("timestamp_ms") or time.time() * 1000)
                etype = str(e.get("tipo_evento") or e.get("type", "UNKNOWN"))
                sym = str(e.get("ativo") or e.get("symbol", ""))
                wid = str(e.get("window_id", ""))
                is_sig_val = e.get("is_signal")
                is_sig = 1 if (is_sig_val is True or str(is_sig_val).lower() == "true") else 0

                payload = json.dumps(sanitize_json_safe(e), default=str)
                data.append((ts, etype, sym, wid, is_sig, payload))
            except Exception as err:
                rejected_pre_commit += 1
                self.logger.error(
                    f"Erro ao preparar evento para salvar: {err} | Evento parcial: {str(e)[:100]}"
                )
                continue

        if not data:
            return BatchSaveResult(
                total_received=total_received,
                persisted=0,
                rejected_pre_commit=rejected_pre_commit,
                commit_status="FAILED",
                error_type="PreCommitValidationError",
                error_message="Todos os eventos foram rejeitados na preparação pré-commit",
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
            stage = "EXECUTING"
            conn.executemany(insert_sql, data)
            stage = "COMMITTING"
            conn.commit()
            stage = "COMMITTED"
            return BatchSaveResult(
                total_received=total_received,
                persisted=len(data),
                rejected_pre_commit=rejected_pre_commit,
                commit_status="COMMITTED",
                error_type=None,
            )
        except sqlite3.OperationalError as e:
            # Auto-recuperação: se a tabela ainda não existe por qualquer motivo,
            # recria o schema e tenta 1 vez.
            msg = str(e).lower()
            if "no such table" in msg and "events" in msg:
                conn2 = None
                try:
                    self.logger.warning(
                        "Tabela 'events' ausente no SQLite; recriando schema e tentando novamente..."
                    )
                    self._init_db()
                    stage = "CONNECTING_RETRY"
                    conn2 = self._get_conn()
                    stage = "EXECUTING_RETRY"
                    conn2.executemany(insert_sql, data)
                    stage = "COMMITTING_RETRY"
                    conn2.commit()
                    stage = "COMMITTED"
                    return BatchSaveResult(
                        total_received=total_received,
                        persisted=len(data),
                        rejected_pre_commit=rejected_pre_commit,
                        commit_status="COMMITTED",
                        error_type=None,
                    )
                except Exception as e2:
                    self.logger.error(
                        f"Erro ao recriar schema do SQLite e regravar batch ({len(data)} eventos): {e2}"
                    )
                    status = self._determine_failure_status(conn2, stage)
                    return BatchSaveResult(
                        total_received=total_received,
                        persisted=0,
                        rejected_pre_commit=rejected_pre_commit,
                        commit_status=status,
                        error_type=type(e2).__name__,
                        error_message=str(e2),
                    )
                finally:
                    if conn2 is not None:
                        try:
                            conn2.close()
                        except Exception:
                            pass

            self.logger.error(
                f"Erro ao gravar batch no SQLite ({len(data)} eventos): {e}"
            )
            status = self._determine_failure_status(conn, stage)
            return BatchSaveResult(
                total_received=total_received,
                persisted=0,
                rejected_pre_commit=rejected_pre_commit,
                commit_status=status,
                error_type=type(e).__name__,
                error_message=str(e),
            )
        except Exception as e:
            self.logger.error(f"Erro ao gravar batch no SQLite ({len(data)} eventos): {e}")
            status = self._determine_failure_status(conn, stage)
            return BatchSaveResult(
                total_received=total_received,
                persisted=0,
                rejected_pre_commit=rejected_pre_commit,
                commit_status=status,
                error_type=type(e).__name__,
                error_message=str(e),
            )
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

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
                }
        except Exception as e:
            return {"error": str(e)}

    async def get_stats_async(self) -> Dict[str, Any]:
        """Retorna estatísticas básicas do banco (Assíncrono)."""
        return await asyncio.to_thread(self.get_stats)
