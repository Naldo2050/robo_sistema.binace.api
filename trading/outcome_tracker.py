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

# Política boundary-only fail-closed:
# um horizonte só é preenchido quando
#   0 <= current_epoch_ms - (signal_epoch_ms + horizon*60_000) <= OUTCOME_BOUNDARY_TOLERANCE_MS
# Ou seja: o preço recebido precisa ser o fechamento da janela que fechou
# EXATAMENTE no boundary do horizonte. O candle seguinte (+60_000ms) NUNCA
# preenche o horizonte anterior; boundary perdido permanece NULL.
OUTCOME_BOUNDARY_TOLERANCE_MS = 1000

# (horizonte_min, sufixo_da_coluna)
_OUTCOME_HORIZONS = ((5, "5m"), (15, "15m"), (30, "30m"), (60, "60m"))


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

            if epoch_ms % 60_000 != 0:
                logger.warning(
                    "register_signal: signal_epoch_ms=%s nao esta alinhado ao "
                    "boundary de 1m (resto %s ms). Com a politica boundary-only, "
                    "este sinal pode nunca encontrar drift dentro da tolerancia "
                    "e seus outcomes permanecerao NULL. Nenhuma normalizacao "
                    "foi aplicada ao timestamp.",
                    epoch_ms,
                    epoch_ms % 60_000,
                )

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
        Avalia outcomes pendentes sob política boundary-only fail-closed.

        Contrato por horizonte (5m/15m/30m/60m):
            target_epoch_ms = signal_epoch_ms + horizon_min * 60_000
            drift_ms        = current_epoch_ms - target_epoch_ms
            grava SOMENTE se 0 <= drift_ms <= OUTCOME_BOUNDARY_TOLERANCE_MS

        O chamador deve fornecer current_price = fechamento da janela que
        acabou de fechar em current_epoch_ms. Não há interpolação, nearest
        ou busca de preço posterior: boundary perdido permanece NULL
        (consumidores já ignoram NULL em denominadores/probabilidades).

        evaluated_at = timestamp da ÚLTIMA avaliação que efetivamente
        preencheu algum horizonte. NÃO é timestamp específico de um
        outcome individual (5m/15m/etc.) e não deve ser usado como tal.
        """
        try:
            with self._get_conn() as conn:
                # Pré-filtro LARGO (apenas performance):
                #  - limite superior inclusivo: permite idade exatamente 5m;
                #  - limite inferior: um sinal só ainda pode receber ALGUM
                #    stamp enquanto drift do último horizonte (60m) estiver
                #    em [0, tol]; mais velho que isso todos os horizontes
                #    estão permanentemente perdidos e a linha sairia da
                #    varredura sem alterar nenhum resultado possível
                #    (elimina churn de rescan etário). A decisão definitiva
                #    continua sendo o drift por horizonte abaixo.
                cursor = conn.execute(
                    """SELECT id, signal_epoch_ms, entry_price, event_type, battle_result
                    FROM signal_outcomes
                    WHERE outcome_60m_pct IS NULL
                    AND signal_epoch_ms <= ?
                    AND signal_epoch_ms >= ?
                    ORDER BY signal_epoch_ms ASC
                    LIMIT 100""",
                    (
                        current_epoch_ms - 300_000,
                        current_epoch_ms - 3_600_000 - OUTCOME_BOUNDARY_TOLERANCE_MS,
                    ),
                )

                for row in cursor.fetchall():
                    row_id, signal_ms, entry_price, event_type, battle_result = row

                    if entry_price <= 0:
                        continue

                    pct_change = ((current_price - entry_price) / entry_price) * 100
                    direction = "UP" if pct_change > 0.01 else ("DOWN" if pct_change < -0.01 else "FLAT")

                    for horizon_min, window in _OUTCOME_HORIZONS:
                        # Otimização; a garantia real de concorrência está no
                        # UPDATE condicional (outcome_X_pct IS NULL) abaixo.
                        if self._has_outcome(conn, row_id, window):
                            continue

                        target_epoch_ms = signal_ms + horizon_min * 60_000
                        drift_ms = current_epoch_ms - target_epoch_ms
                        if not (0 <= drift_ms <= OUTCOME_BOUNDARY_TOLERANCE_MS):
                            continue

                        result = conn.execute(
                            f"""UPDATE signal_outcomes
                            SET outcome_{window}_pct = ?,
                                outcome_direction_{window} = ?,
                                evaluated_at = ?
                            WHERE id = ?
                              AND outcome_{window}_pct IS NULL""",
                            (
                                round(pct_change, 4),
                                direction,
                                current_epoch_ms,
                                row_id,
                            ),
                        )
                        _ = result.rowcount  # 0 => outro writer venceu; não sobrescrever

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
