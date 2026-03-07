# event_similarity.py
"""
Busca de eventos similares usando numpy (sem dependência externa).

Implementa memória longa contextual: quando um novo sinal é detectado,
busca eventos passados com características semelhantes e retorna
seus outcomes para dar contexto à IA.

Features usadas para comparação:
- delta (normalizado)
- volume_total (normalizado)
- indice_absorcao
- flow_imbalance
- whale_score
- trend_direction (encoded)
- volatility_regime (encoded)
- buy_sell_ratio

Sem API externa - 100% local com numpy.
"""

import sqlite3
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger("EventSimilarity")

# Features a extrair de cada evento para o vetor
FEATURE_KEYS = [
    "delta",
    "volume_total",
    "indice_absorcao",
    "flow_imbalance",
    "whale_score",
    "buy_sell_ratio",
    "trend_encoded",
    "volatility_encoded",
    "session_encoded",
    "absorption_side_encoded",
]

# Encodings categóricos
TREND_MAP = {"UP": 1.0, "DOWN": -1.0, "SIDEWAYS": 0.0, "RANGE_BOUND": 0.0}
VOL_MAP = {"HIGH": 1.0, "LOW": -1.0, "NORMAL": 0.0, "MODERATE": 0.0}
SESSION_MAP = {"US": 1.0, "EUROPE": 0.5, "ASIA": -0.5, "OVERLAP": 0.75}
SIDE_MAP = {"buy": 1.0, "sell": -1.0}


class EventSimilaritySearch:
    """
    Busca eventos historicamente similares ao evento atual.
    Usa vetores de features + distância coseno/euclidiana.
    """

    def __init__(self, db_path: str = "dados/trading_bot.db", max_events: int = 5000):
        self.db_path = Path(db_path).expanduser().resolve()
        self.max_events = max_events
        self._vectors: Optional[np.ndarray] = None
        self._metadata: List[Dict] = []
        self._loaded = False

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _load_vectors(self, force: bool = False):
        """Carrega vetores de eventos do SQLite para numpy."""
        if self._loaded and not force:
            return

        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    """SELECT payload FROM events
                    WHERE event_type IN ('Absorção', 'Exaustão', 'AbsorÃ§Ã£o')
                    AND is_signal = 1
                    ORDER BY timestamp_ms DESC
                    LIMIT ?""",
                    (self.max_events,),
                )
                rows = cursor.fetchall()

            vectors = []
            metadata = []

            for (payload_str,) in rows:
                try:
                    event = json.loads(payload_str)
                    vec = self._extract_feature_vector(event)
                    if vec is not None:
                        vectors.append(vec)
                        metadata.append({
                            "epoch_ms": event.get("epoch_ms", 0),
                            "tipo_evento": event.get("tipo_evento", ""),
                            "resultado_da_batalha": event.get("resultado_da_batalha", ""),
                            "preco_fechamento": event.get("preco_fechamento", 0),
                            "delta": event.get("delta", 0),
                        })
                except Exception:
                    continue

            if vectors:
                self._vectors = np.array(vectors, dtype=np.float32)
                # Normalizar cada feature (z-score)
                means = self._vectors.mean(axis=0)
                stds = self._vectors.std(axis=0)
                stds[stds == 0] = 1  # evitar divisão por zero
                self._vectors = (self._vectors - means) / stds
                self._norm_means = means
                self._norm_stds = stds
                self._metadata = metadata
                self._loaded = True
                logger.info(f"Similarity search: {len(vectors)} eventos carregados")
            else:
                logger.warning("Nenhum evento encontrado para similarity search")
                self._vectors = np.empty((0, len(FEATURE_KEYS)), dtype=np.float32)
                self._metadata = []
                self._loaded = True

        except Exception as e:
            logger.error(f"Erro ao carregar vetores: {e}")

    def _extract_feature_vector(self, event: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extrai vetor de features de um evento."""
        try:
            flow = event.get("fluxo_continuo", {})
            order_flow = flow.get("order_flow", {})
            bsr = order_flow.get("buy_sell_ratio", {})
            inst = event.get("institutional_analytics", {})
            whale = inst.get("flow_analysis", {}).get("whale_accumulation", {})
            mkt_env = event.get("market_environment", {})
            mkt_ctx = event.get("market_context", {})

            vec = np.array([
                float(event.get("delta", 0)),
                float(event.get("volume_total", 0)),
                float(event.get("indice_absorcao", 0)),
                float(order_flow.get("flow_imbalance", 0)),
                float(whale.get("score", 0)),
                float(bsr.get("buy_sell_ratio", 1.0)),
                TREND_MAP.get(mkt_env.get("trend_direction", ""), 0.0),
                VOL_MAP.get(mkt_env.get("volatility_regime", ""), 0.0),
                SESSION_MAP.get(mkt_ctx.get("trading_session", ""), 0.0),
                SIDE_MAP.get(event.get("absorption_side", ""), 0.0),
            ], dtype=np.float32)

            # Validar que não é tudo zero
            if np.all(vec == 0):
                return None

            return vec

        except Exception:
            return None

    def find_similar(
        self,
        current_event: Dict[str, Any],
        top_k: int = 5,
        method: str = "cosine",
    ) -> Dict[str, Any]:
        """
        Encontra os top_k eventos mais similares ao evento atual.

        Args:
            current_event: Evento atual para buscar similares
            top_k: Número de resultados
            method: "cosine" ou "euclidean"

        Returns:
            Dict com eventos similares e seus outcomes
        """
        self._load_vectors()

        if self._vectors is None or len(self._vectors) == 0:
            return {
                "status": "no_data",
                "similar_events": [],
                "summary": "Sem dados históricos para comparação",
            }

        query_vec = self._extract_feature_vector(current_event)
        if query_vec is None:
            return {
                "status": "invalid_event",
                "similar_events": [],
                "summary": "Evento atual não tem features suficientes",
            }

        # Normalizar query com mesmos parâmetros
        query_normalized = (query_vec - self._norm_means) / self._norm_stds

        if method == "cosine":
            distances = self._cosine_distances(query_normalized, self._vectors)
        else:
            distances = self._euclidean_distances(query_normalized, self._vectors)

        # Top K (menores distâncias)
        top_indices = np.argsort(distances)[:top_k]

        similar = []
        for idx in top_indices:
            meta = self._metadata[idx]
            similarity = float(1.0 - distances[idx]) if method == "cosine" else float(1.0 / (1.0 + distances[idx]))
            similar.append({
                "similarity": round(similarity, 4),
                "distance": round(float(distances[idx]), 4),
                "epoch_ms": meta["epoch_ms"],
                "tipo_evento": meta["tipo_evento"],
                "resultado_da_batalha": meta["resultado_da_batalha"],
                "preco_fechamento": meta["preco_fechamento"],
                "delta": meta["delta"],
            })

        # Enriquecer com outcomes se disponível
        similar_with_outcomes = self._enrich_with_outcomes(similar)

        # Resumo estatístico
        summary = self._build_summary(similar_with_outcomes)

        return {
            "status": "ok",
            "query_features": {k: round(float(v), 4) for k, v in zip(FEATURE_KEYS, query_vec)},
            "similar_events": similar_with_outcomes,
            "summary": summary,
            "method": method,
            "total_events_in_db": len(self._vectors),
            "is_real_data": True,
        }

    def _cosine_distances(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Calcula distâncias coseno entre query e todos os vetores."""
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.ones(len(vectors))

        vec_norms = np.linalg.norm(vectors, axis=1)
        vec_norms[vec_norms == 0] = 1

        similarities = np.dot(vectors, query) / (vec_norms * query_norm)
        return 1.0 - similarities  # distância = 1 - similaridade

    def _euclidean_distances(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Calcula distâncias euclidianas."""
        return np.linalg.norm(vectors - query, axis=1)

    def _enrich_with_outcomes(self, similar: List[Dict]) -> List[Dict]:
        """Enriquece eventos similares com outcomes do OutcomeTracker."""
        try:
            with self._get_conn() as conn:
                for event in similar:
                    epoch = event.get("epoch_ms", 0)
                    if epoch == 0:
                        continue

                    cursor = conn.execute(
                        """SELECT outcome_5m_pct, outcome_15m_pct, outcome_30m_pct,
                        outcome_direction_5m, outcome_direction_15m, outcome_direction_30m
                        FROM signal_outcomes
                        WHERE signal_epoch_ms = ?
                        LIMIT 1""",
                        (epoch,),
                    )
                    row = cursor.fetchone()
                    if row:
                        event["outcome"] = {
                            "5m": {"pct": row[0], "direction": row[3]},
                            "15m": {"pct": row[1], "direction": row[4]},
                            "30m": {"pct": row[2], "direction": row[5]},
                        }
        except Exception:
            pass  # tabela pode não existir ainda

        return similar

    def _build_summary(self, events: List[Dict]) -> Dict[str, Any]:
        """Constrói resumo estatístico dos eventos similares."""
        outcomes_up = 0
        outcomes_down = 0
        returns = []

        for ev in events:
            outcome = ev.get("outcome", {})
            o15 = outcome.get("15m", {})
            if o15.get("direction") == "UP":
                outcomes_up += 1
            elif o15.get("direction") == "DOWN":
                outcomes_down += 1
            if o15.get("pct") is not None:
                returns.append(o15["pct"])

        total = outcomes_up + outcomes_down
        return {
            "similar_count": len(events),
            "with_outcomes": total,
            "outcomes_up": outcomes_up,
            "outcomes_down": outcomes_down,
            "historical_win_rate": round(outcomes_up / total * 100, 1) if total > 0 else None,
            "avg_return_pct": round(sum(returns) / len(returns), 4) if returns else None,
            "interpretation": self._interpret(outcomes_up, outcomes_down, returns),
        }

    def _interpret(self, up: int, down: int, returns: List[float]) -> str:
        total = up + down
        if total < 3:
            return "Dados insuficientes para interpretação confiável"

        win_rate = up / total * 100
        avg_ret = sum(returns) / len(returns) if returns else 0

        if win_rate >= 70:
            return f"Cenários similares tiveram {win_rate:.0f}% de alta (retorno médio: {avg_ret:.3f}%)"
        elif win_rate <= 30:
            return f"Cenários similares tiveram {100-win_rate:.0f}% de queda (retorno médio: {avg_ret:.3f}%)"
        else:
            return f"Cenários similares tiveram resultado misto ({win_rate:.0f}% alta, retorno médio: {avg_ret:.3f}%)"
