# outcome_tracker.py
"""
Rastreador de outcomes (resultados) dos eventos de trading.

Para cada sinal emitido (Absorção, Exaustão), registra o preço no momento
e depois verifica o preço N minutos depois para calcular:
- Taxa de acerto por tipo de evento
- Retorno médio por tipo de evento
- Probabilidade condicional (ex: "Absorção de Venda em VAL + funding positivo -> 73% alta")

Usa SQLite (event_store) como fonte de dados.
Sem API externa - cálculo 100% local.
"""

import sqlite3
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("OutcomeTracker")

# Janelas de avaliação em minutos
EVAL_WINDOWS_MIN = [5, 15, 30, 60]


class OutcomeTracker:
    """
    Rastreia e calcula outcomes dos sinais emitidos pelo sistema.
    """

    def __init__(self, db_path: str = "dados/trading_bot.db"):
        self.db_path = Path(db_path).expanduser().resolve()
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=3000;")
        return conn

    def _ensure_table(self):
        """Cria tabela de outcomes se não existir."""
        try:
            with self._get_conn() as conn:
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
            logger.error(f"Erro ao criar tabela signal_outcomes: {e}")

    def register_signal(self, event: Dict[str, Any]):
        """
        Registra um novo sinal para tracking de outcome.
        Chamado quando um evento de Absorção/Exaustão é gerado.
        """
        try:
            epoch_ms = event.get("epoch_ms", int(time.time() * 1000))
            event_type = event.get("tipo_evento", "UNKNOWN")
            battle_result = event.get("resultado_da_batalha", "")
            entry_price = event.get("preco_fechamento", 0)
            symbol = event.get("ativo", event.get("symbol", "BTCUSDT"))

            if entry_price <= 0:
                return

            # Contexto compacto para análise posterior
            context = {
                "delta": event.get("delta", 0),
                "volume_total": event.get("volume_total", 0),
                "indice_absorcao": event.get("indice_absorcao", 0),
                "session": event.get("market_context", {}).get("trading_session", ""),
                "trend": event.get("market_environment", {}).get("trend_direction", ""),
                "volatility": event.get("market_environment", {}).get("volatility_regime", ""),
                "whale_score": event.get("institutional_analytics", {}).get(
                    "flow_analysis", {}
                ).get("whale_accumulation", {}).get("score", 0),
            }

            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO signal_outcomes
                    (signal_epoch_ms, event_type, battle_result, entry_price, symbol, context_json)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (epoch_ms, event_type, battle_result, entry_price, symbol,
                     json.dumps(context, default=str)),
                )
        except Exception as e:
            logger.error(f"Erro ao registrar sinal: {e}")

    def evaluate_pending_outcomes(self, current_price: float, current_epoch_ms: int):
        """
        Avalia outcomes pendentes comparando preço atual com preço de entrada.
        Chamado periodicamente (a cada janela de 5 min).
        """
        try:
            with self._get_conn() as conn:
                # Buscar sinais que ainda não foram totalmente avaliados
                cursor = conn.execute(
                    """SELECT id, signal_epoch_ms, entry_price, event_type, battle_result
                    FROM signal_outcomes
                    WHERE outcome_60m_pct IS NULL
                    AND signal_epoch_ms < ?
                    ORDER BY signal_epoch_ms ASC
                    LIMIT 100""",
                    (current_epoch_ms - 300_000,)  # pelo menos 5 min atrás
                )

                for row in cursor.fetchall():
                    row_id, signal_ms, entry_price, event_type, battle_result = row
                    elapsed_min = (current_epoch_ms - signal_ms) / 60_000

                    if entry_price <= 0:
                        continue

                    pct_change = ((current_price - entry_price) / entry_price) * 100
                    direction = "UP" if pct_change > 0.01 else ("DOWN" if pct_change < -0.01 else "FLAT")

                    updates = {}
                    if elapsed_min >= 5 and not self._has_outcome(conn, row_id, "5m"):
                        updates["outcome_5m_pct"] = round(pct_change, 4)
                        updates["outcome_direction_5m"] = direction
                    if elapsed_min >= 15 and not self._has_outcome(conn, row_id, "15m"):
                        updates["outcome_15m_pct"] = round(pct_change, 4)
                        updates["outcome_direction_15m"] = direction
                    if elapsed_min >= 30 and not self._has_outcome(conn, row_id, "30m"):
                        updates["outcome_30m_pct"] = round(pct_change, 4)
                        updates["outcome_direction_30m"] = direction
                    if elapsed_min >= 60 and not self._has_outcome(conn, row_id, "60m"):
                        updates["outcome_60m_pct"] = round(pct_change, 4)
                        updates["outcome_direction_60m"] = direction

                    if updates:
                        updates["evaluated_at"] = current_epoch_ms
                        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
                        conn.execute(
                            f"UPDATE signal_outcomes SET {set_clause} WHERE id = ?",
                            list(updates.values()) + [row_id],
                        )

        except Exception as e:
            logger.error(f"Erro ao avaliar outcomes: {e}")

    def _has_outcome(self, conn, row_id: int, window: str) -> bool:
        cursor = conn.execute(
            f"SELECT outcome_{window}_pct FROM signal_outcomes WHERE id = ?",
            (row_id,),
        )
        row = cursor.fetchone()
        return row is not None and row[0] is not None

    def get_historical_probability(
        self,
        event_type: str = "",
        battle_result: str = "",
        window: str = "15m",
        min_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Calcula probabilidade histórica real baseada em outcomes passados.

        Args:
            event_type: Tipo de evento (ex: "Absorção", "Exaustão")
            battle_result: Resultado da batalha (ex: "Absorção de Venda")
            window: Janela de avaliação ("5m", "15m", "30m", "60m")
            min_samples: Mínimo de amostras para resultado confiável

        Returns:
            Dict com probabilidades e estatísticas
        """
        try:
            direction_col = f"outcome_direction_{window}"
            pct_col = f"outcome_{window}_pct"

            conditions = [f"{pct_col} IS NOT NULL"]
            params = []

            if event_type:
                conditions.append("event_type LIKE ?")
                params.append(f"%{event_type}%")
            if battle_result:
                conditions.append("battle_result LIKE ?")
                params.append(f"%{battle_result}%")

            where = " AND ".join(conditions)

            with self._get_conn() as conn:
                # Total de amostras
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM signal_outcomes WHERE {where}", params
                )
                total = cursor.fetchone()[0]

                if total < min_samples:
                    return {
                        "status": "insufficient_data",
                        "samples": total,
                        "min_required": min_samples,
                        "window": window,
                    }

                # Contar direções
                cursor = conn.execute(
                    f"""SELECT {direction_col}, COUNT(*), AVG({pct_col}),
                    MIN({pct_col}), MAX({pct_col})
                    FROM signal_outcomes
                    WHERE {where}
                    GROUP BY {direction_col}""",
                    params,
                )

                results = {}
                for row in cursor.fetchall():
                    direction, count, avg_pct, min_pct, max_pct = row
                    results[direction] = {
                        "count": count,
                        "pct": round(count / total * 100, 1),
                        "avg_return_pct": round(avg_pct, 4),
                        "min_return_pct": round(min_pct, 4),
                        "max_return_pct": round(max_pct, 4),
                    }

                # Calcular métricas agregadas
                cursor = conn.execute(
                    f"""SELECT AVG({pct_col}),
                    AVG(CASE WHEN {pct_col} > 0 THEN {pct_col} END),
                    AVG(CASE WHEN {pct_col} < 0 THEN {pct_col} END)
                    FROM signal_outcomes WHERE {where}""",
                    params,
                )
                agg = cursor.fetchone()

                up_data = results.get("UP", {})
                down_data = results.get("DOWN", {})

                return {
                    "status": "ok",
                    "window": window,
                    "samples": total,
                    "event_type": event_type,
                    "battle_result": battle_result,
                    "prob_up": round(up_data.get("pct", 0) / 100, 4),
                    "prob_down": round(down_data.get("pct", 0) / 100, 4),
                    "prob_flat": round(results.get("FLAT", {}).get("pct", 0) / 100, 4),
                    "avg_return_pct": round(agg[0] or 0, 4),
                    "avg_win_pct": round(agg[1] or 0, 4),
                    "avg_loss_pct": round(agg[2] or 0, 4),
                    "win_rate": round(up_data.get("pct", 0), 1),
                    "details": results,
                    "is_real_data": True,
                }

        except Exception as e:
            logger.error(f"Erro ao calcular probabilidade histórica: {e}")
            return {"status": "error", "error": str(e)}

    def get_all_probabilities(self, min_samples: int = 10) -> Dict[str, Any]:
        """
        Retorna probabilidades para todas as combinações de event_type/battle_result.
        Útil para injetar como feature de confiança no payload da IA.
        """
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """SELECT DISTINCT event_type, battle_result
                    FROM signal_outcomes
                    WHERE outcome_15m_pct IS NOT NULL
                    GROUP BY event_type, battle_result
                    HAVING COUNT(*) >= ?""",
                    (min_samples,),
                )

                combos = cursor.fetchall()

            results = {}
            for event_type, battle_result in combos:
                key = f"{event_type}|{battle_result}"
                for window in ["5m", "15m", "30m", "60m"]:
                    prob = self.get_historical_probability(
                        event_type=event_type,
                        battle_result=battle_result,
                        window=window,
                        min_samples=min_samples,
                    )
                    if prob.get("status") == "ok":
                        results[f"{key}|{window}"] = prob

            return {
                "status": "ok",
                "combinations": len(results),
                "probabilities": results,
                "is_real_data": True,
            }

        except Exception as e:
            logger.error(f"Erro ao calcular todas as probabilidades: {e}")
            return {"status": "error", "error": str(e)}

    def get_confidence_for_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retorna a confiança estatística para um evento específico.
        Usado para injetar no payload da IA como feature adicional.
        """
        event_type = event.get("tipo_evento", "")
        battle_result = event.get("resultado_da_batalha", "")

        confidence = {}
        for window in ["5m", "15m", "30m"]:
            prob = self.get_historical_probability(
                event_type=event_type,
                battle_result=battle_result,
                window=window,
                min_samples=5,
            )
            if prob.get("status") == "ok":
                confidence[window] = {
                    "prob_up": prob["prob_up"],
                    "prob_down": prob["prob_down"],
                    "win_rate": prob["win_rate"],
                    "avg_return_pct": prob["avg_return_pct"],
                    "samples": prob["samples"],
                }

        return {
            "has_data": bool(confidence),
            "windows": confidence,
            "is_real_data": True,
        }
