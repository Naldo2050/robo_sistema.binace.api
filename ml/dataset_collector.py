# ml/dataset_collector.py
# -*- coding: utf-8 -*-

"""
Coletor incremental de dados para retreino de modelo ML.

A cada janela processada, salva features + preço.
Labels são calculados retroativamente quando o horizonte futuro fica disponível.
Flush automático a cada 50 registros com label preenchido.

Uso:
    from ml.dataset_collector import get_dataset_collector
    collector = get_dataset_collector()
    collector.collect_window(features_dict, price_close, timestamp)
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

BUFFER_PATH = Path("ml/datasets/collection_buffer.parquet")

# Thresholds alinhados com generate_dataset.py
DEFAULT_HORIZON = 15
DEFAULT_THRESHOLD = 0.002


class DatasetCollector:
    """Coleta contínua de dados para treino futuro."""

    def __init__(
        self,
        horizon_minutes: int = DEFAULT_HORIZON,
        threshold: float = DEFAULT_THRESHOLD,
        flush_every: int = 50,
    ):
        self.horizon = horizon_minutes
        self.threshold = threshold
        self.flush_every = flush_every
        self._buffer: list[dict] = []

    # ──────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────

    def collect_window(
        self,
        features: dict,
        price_close: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Chamado a cada janela processada.
        Salva features + preço. Label será calculado retroativamente.
        """
        record = {
            "price_close": price_close,
            "timestamp": timestamp or time.time(),
            "label_direction": None,
        }
        # Flatten features de primeiro nível
        for k, v in features.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                record[k] = v

        self._buffer.append(record)

        # Calcular labels retroativamente para registros antigos
        self._backfill_labels()

        # Flush registros com label preenchido
        labeled_count = sum(
            1 for r in self._buffer if r["label_direction"] is not None
        )
        if labeled_count >= self.flush_every:
            self._flush_to_disk()

    def get_stats(self) -> dict:
        """Retorna estatísticas da coleta."""
        if not BUFFER_PATH.exists():
            return {"total": 0, "buffer_pending": len(self._buffer), "ready": False}

        try:
            df = pd.read_parquet(BUFFER_PATH)
        except Exception:
            return {"total": 0, "buffer_pending": len(self._buffer), "ready": False}

        total = len(df)
        dist = df["label_direction"].value_counts().to_dict()
        min_class = (
            min(dist.get(-1, 0), dist.get(0, 0), dist.get(1, 0)) / max(total, 1)
        )

        return {
            "total": total,
            "buffer_pending": len(self._buffer),
            "distribution": dist,
            "min_class_ratio": round(min_class, 3),
            "ready_for_training": total >= 500 and min_class >= 0.15,
        }

    # ──────────────────────────────────────────────
    # Internos
    # ──────────────────────────────────────────────

    def _backfill_labels(self) -> None:
        """Preenche labels de registros que já têm dados futuros."""
        for i, record in enumerate(self._buffer):
            if record["label_direction"] is not None:
                continue

            future_idx = i + self.horizon
            if future_idx >= len(self._buffer):
                break

            current_price = record["price_close"]
            future_price = self._buffer[future_idx]["price_close"]

            if current_price <= 0 or future_price <= 0:
                continue

            ret = (future_price - current_price) / current_price

            if ret > self.threshold:
                record["label_direction"] = 1
            elif ret < -self.threshold:
                record["label_direction"] = -1
            else:
                record["label_direction"] = 0

    def _flush_to_disk(self) -> None:
        """Salva registros com label preenchido no disco."""
        labeled = [r for r in self._buffer if r["label_direction"] is not None]

        if not labeled:
            return

        df_new = pd.DataFrame(labeled)

        try:
            if BUFFER_PATH.exists():
                df_existing = pd.read_parquet(BUFFER_PATH)
                df_combined = pd.concat(
                    [df_existing, df_new], ignore_index=True
                )
                df_combined.drop_duplicates(
                    subset=["timestamp"], keep="last", inplace=True
                )
            else:
                df_combined = df_new

            BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
            df_combined.to_parquet(BUFFER_PATH, index=False)

            # Remover registros já salvos do buffer em memória
            self._buffer = [
                r for r in self._buffer if r["label_direction"] is None
            ]

            total = len(df_combined)
            dist = df_combined["label_direction"].value_counts().to_dict()
            logger.info(
                "Dataset collector: %d amostras salvas | dist=%s",
                total, dist,
            )

            if total >= 500:
                bullish_pct = dist.get(1, 0) / total
                bearish_pct = dist.get(-1, 0) / total
                if bullish_pct >= 0.15 and bearish_pct >= 0.15:
                    logger.info(
                        "Dataset PRONTO para retreino! "
                        "%d amostras, distribuicao balanceada",
                        total,
                    )

        except Exception as e:
            logger.error("Erro ao salvar collection_buffer: %s", e)


# ──────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────

_collector_instance: Optional[DatasetCollector] = None


def get_dataset_collector(**kwargs) -> DatasetCollector:
    """Retorna instância singleton."""
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = DatasetCollector(**kwargs)
        logger.info(
            "DatasetCollector inicializado: horizon=%d, threshold=%.3f",
            _collector_instance.horizon, _collector_instance.threshold,
        )
    return _collector_instance
