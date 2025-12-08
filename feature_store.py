import json
import time
import logging
import datetime
from typing import Any, Dict, Optional
from pathlib import Path

import pandas as pd


class FeatureStore:
    """
    Armazena features por janela em Parquet particionado por dia.
    - Faz flatten/serialização segura para campos aninhados (dict/list).
    - Usa buffer in-memory com flush para Parquet.
    - Particionamento por data (date=YYYY-MM-DD).
    """

    def __init__(
        self,
        base_dir: str = "features",
        filename: str = "features.csv",
        rotate_max_bytes: Optional[int] = 20_000_000,
        rotate_max_rows: Optional[int] = 200_000,
        lock_filename: str = ".features.lock",
        backup_suffix: str = ".bak",
        engine: str = "pyarrow",
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.buffer = []
        self.buffer_size_limit = 100
        self.engine = engine
        logging.info(f"FeatureStore Parquet inicializado em {self.base_dir}")

    @staticmethod
    def _flatten(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (d or {}).items():
            key = f"{parent}{sep}{k}" if parent else k
            if isinstance(v, dict):
                out.update(FeatureStore._flatten(v, key, sep))
            else:
                out[key] = v
        return out

    @staticmethod
    def _to_serializable(v: Any) -> Any:
        # Serializa estruturas complexas; normaliza NaN/Inf
        try:
            if isinstance(v, (dict, list, tuple)):
                return json.dumps(v, ensure_ascii=False)
            if isinstance(v, float):
                if v != v or v in (float("inf"), float("-inf")):
                    return None
            return v
        except Exception:
            return str(v)

    @staticmethod
    def safe_to_numeric(series):
        """Converte série para numérico de forma segura"""
        try:
            # Primeiro tenta converter diretamente
            return pd.to_numeric(series)
        except (ValueError, TypeError):
            # Se falhar, tenta converter strings
            try:
                return pd.to_numeric(series.astype(str).str.replace(',', '.'))
            except:
                # Se ainda falhar, mantém como está
                return series


    def save_features(self, window_id: str, features: Dict[str, Any]) -> None:
        """
        Salva uma linha de features referente à window_id.
        - Faz flatten e serialização.
        - Adiciona window_id e saved_at.
        - Acrescenta ao buffer; flush se limite atingido.
        """
        if not isinstance(features, dict):
            logging.warning("FeatureStore: features não é dict; ignorado.")
            return

        flat = self._flatten(features)
        flat = {k: self._to_serializable(v) for k, v in flat.items()}
        flat["window_id"] = window_id
        flat["saved_at"] = datetime.datetime.utcnow()

        self.buffer.append(flat)
        if len(self.buffer) >= self.buffer_size_limit:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = self.safe_to_numeric(df[col])

        df["saved_at"] = pd.to_datetime(df["saved_at"])
        df["date"] = df["saved_at"].dt.strftime("%Y-%m-%d")

        grouped = df.groupby("date")
        for date_str, group in grouped:
            partition_dir = self.base_dir / f"date={date_str}"
            partition_dir.mkdir(exist_ok=True)
            timestamp_ms = int(time.time() * 1000)
            filepath = partition_dir / f"part_{timestamp_ms}.parquet"
            try:
                group.drop(columns=["date"]).to_parquet(
                    filepath, index=False, compression="snappy", engine=self.engine
                )
                logging.info(f"{len(group)} linhas gravadas em {filepath}")
            except Exception as e:
                logging.error(f"Erro ao gravar Parquet para {date_str}: {e}")
                # Não limpa buffer em caso de erro

        self.buffer.clear()

    def close(self) -> None:
        self._flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
