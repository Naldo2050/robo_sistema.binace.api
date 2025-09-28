import os
import json
import logging
from typing import Any, Dict, Optional

import pandas as pd


class FeatureStore:
    """
    Armazena features por janela em CSV (features/features.csv por padrão).
    - Evita FutureWarning do pandas ao concatenar com DF vazio/todos-NA.
    - Faz flatten/serialização segura para campos aninhados (dict/list).
    - Garante união de colunas entre o arquivo existente e a nova linha.
    """

    def __init__(self, base_dir: str = "features", filename: str = "features.csv"):
        self.base_dir = base_dir
        self.filename = filename
        self.file_path = os.path.join(base_dir, filename)
        os.makedirs(self.base_dir, exist_ok=True)
        logging.info(f"✅ FeatureStore inicializado. Dados salvos em: {self.base_dir}")

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

    def _read_existing(self) -> Optional[pd.DataFrame]:
        if not os.path.exists(self.file_path):
            return None
        try:
            df = pd.read_csv(self.file_path)
            # Normaliza: garante DataFrame válido
            if df is None or df.empty:
                return None
            return df
        except Exception as e:
            logging.warning(f"FeatureStore: falha ao ler arquivo existente ({self.file_path}): {e}")
            return None

    def save_features(self, window_id: str, features: Dict[str, Any]) -> None:
        """
        Salva uma linha de features referente à window_id.
        - Não concatena DataFrames vazios/todos-NA.
        - Une colunas com o arquivo existente para evitar dtype/colunas inconsistentes.
        """
        if not isinstance(features, dict):
            logging.warning("FeatureStore: features não é dict; ignorado.")
            return

        # Flatten + serialização segura
        flat = self._flatten(features)
        flat = {k: self._to_serializable(v) for k, v in flat.items()}
        flat["window_id"] = window_id

        row_df = pd.DataFrame([flat])

        # Se a linha nova é toda NA (após serialização), não escreve
        if row_df.drop(columns=["window_id"], errors="ignore").dropna(how="all").empty:
            logging.debug("FeatureStore: linha de features toda NA; não será gravada.")
            return

        existing_df = self._read_existing()

        try:
            if existing_df is None or existing_df.empty:
                # Escreve do zero sem concat (evita FutureWarning)
                row_df.to_csv(self.file_path, index=False)
                return

            # Unifica colunas entre existente e nova linha
            union_cols = list(dict.fromkeys(list(existing_df.columns) + list(row_df.columns)))
            existing_df = existing_df.reindex(columns=union_cols)
            row_df = row_df.reindex(columns=union_cols)

            # Concat final, sem colunas todos-NA desnecessárias
            final_df = pd.concat([existing_df, row_df], ignore_index=True, sort=False, copy=False)
            # Opcional: dropar colunas 100% NA (mantém as relevantes)
            # final_df = final_df.dropna(axis=1, how="all")

            final_df.to_csv(self.file_path, index=False)
        except Exception as e:
            logging.error(f"FeatureStore: erro ao salvar features: {e}")