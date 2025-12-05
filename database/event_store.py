# database/event_store.py
import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

class EventStore:
    """
    Gerencia a persistência de eventos de trading usando SQLite.
    Substitui o armazenamento baseado em arquivos JSON/JSONL.
    
    Características:
    - Modo WAL (Write-Ahead Logging) para alta concorrência.
    - Escrita em batch (executemany) para performance.
    - Índices otimizados para busca por tempo e tipo.
    """

    def __init__(self, db_path: str = "dados/trading_bot.db"):
        """
        Inicializa o EventStore.
        
        Args:
            db_path: Caminho para o arquivo do banco de dados.
        """
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
        conn.execute("PRAGMA journal_mode=WAL;")  # Melhora concorrência leitura/escrita
        conn.execute("PRAGMA synchronous=NORMAL;") # Reduz fsyncs excessivos (seguro em WAL)
        conn.execute("PRAGMA busy_timeout=5000;") # Espera até 5s se o banco estiver travado
        
        return conn

    def _init_db(self):
        """Cria a tabela de eventos e índices se não existirem."""
        try:
            with self._get_conn() as conn:
                # Tabela híbrida: colunas relacionais para busca + payload JSON completo
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
                
                # Índices para consultas rápidas
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp_ms);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_window ON events(window_id);")
                
        except Exception as e:
            self.logger.critical(f"Erro fatal ao inicializar banco de dados: {e}")
            raise

    def save_event(self, event: Dict[str, Any]):
        """Salva um único evento no banco."""
        self.save_batch([event])

    def save_batch(self, events: List[Dict[str, Any]]):
        """
        Salva uma lista de eventos de forma atômica (transação única).
        
        Args:
            events: Lista de dicionários contendo os dados dos eventos.
        """
        if not events:
            return

        data = []
        for e in events:
            try:
                # Extração segura de campos para colunas indexadas
                ts = int(e.get('epoch_ms') or e.get('timestamp_ms') or time.time() * 1000)
                etype = str(e.get('tipo_evento') or e.get('type', 'UNKNOWN'))
                sym = str(e.get('ativo') or e.get('symbol', ''))
                wid = str(e.get('window_id', ''))
                
                # Normaliza booleano para 0 ou 1
                is_sig_val = e.get('is_signal')
                is_sig = 1 if (is_sig_val is True or str(is_sig_val).lower() == 'true') else 0
                
                # Serializa o evento completo
                payload = json.dumps(e, default=str)
                
                data.append((ts, etype, sym, wid, is_sig, payload))
            except Exception as err:
                self.logger.error(f"Erro ao preparar evento para salvar: {err} | Evento parcial: {str(e)[:100]}")
                continue

        if not data:
            return

        try:
            with self._get_conn() as conn:
                conn.executemany("""
                    INSERT INTO events (timestamp_ms, event_type, symbol, window_id, is_signal, payload)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, data)
                # Commit é automático no bloco with
        except Exception as e:
            self.logger.error(f"Erro ao gravar batch no SQLite ({len(data)} eventos): {e}")

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Recupera os últimos N eventos ordenados cronologicamente.
        Substitui a leitura do arquivo snapshot JSON.
        
        Returns:
            Lista de dicionários (os payloads originais).
        """
        try:
            with self._get_conn() as conn:
                # Busca os mais recentes primeiro (DESC)
                cursor = conn.execute(
                    "SELECT payload FROM events ORDER BY timestamp_ms DESC, id DESC LIMIT ?", 
                    (limit,)
                )
                rows = cursor.fetchall()
                
                # Desserializa e inverte para ordem cronológica (Antigo -> Novo)
                # Isso mantém o comportamento de listas de histórico
                return [json.loads(row[0]) for row in rows][::-1]
                
        except Exception as e:
            self.logger.error(f"Erro ao ler eventos recentes: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas básicas do banco."""
        try:
            with self._get_conn() as conn:
                count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                last_ts = conn.execute("SELECT MAX(timestamp_ms) FROM events").fetchone()[0]
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
                
                return {
                    "total_events": count,
                    "last_timestamp": last_ts,
                    "db_size_bytes": db_size,
                    "db_path": str(self.db_path)
                }
        except Exception as e:
            return {"error": str(e)}