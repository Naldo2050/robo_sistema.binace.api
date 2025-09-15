import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime

class FeatureStore:
    def __init__(self, base_dir: str = "./features"):
        """
        Armazena features estruturadas de cada janela para uso futuro em:
        - Machine Learning (treino de modelos preditivos)
        - Backtest avançado (com múltiplas features)
        - Análise estatística de setups
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"✅ FeatureStore inicializado. Dados salvos em: {self.base_dir}")

    def _get_daily_path(self, symbol: str) -> Path:
        """Retorna caminho do arquivo diário no formato Parquet."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{symbol}_{today_str}.parquet"
        return self.base_dir / filename

    def save_features(self, window_id: str, features: dict):
        """
        Salva um conjunto de features de uma janela.
        - window_id: identificador único da janela (ex: timestamp de fechamento)
        - features: dicionário com todas as métricas e contextos da janela
        """
        if not features or not isinstance(features, dict):
            logging.warning("⚠️ Features vazias ou inválidas. Nada salvo.")
            return

        try:
            # Flatten do dicionário para colunas (ex: enriched.delta → coluna "enriched__delta")
            flat_features = self._flatten_dict(features, sep="__")
            flat_features["window_id"] = window_id
            flat_features["saved_at"] = datetime.now().isoformat()

            df = pd.DataFrame([flat_features])

            # Salva ou adiciona ao arquivo diário
            file_path = self._get_daily_path(features.get("symbol", "UNKNOWN"))

            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_parquet(file_path, index=False)
            logging.debug(f"💾 Features salvas para janela {window_id} em {file_path.name}")

        except Exception as e:
            logging.error(f"❌ Erro ao salvar features: {e}")

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '__') -> dict:
        """
        Achata um dicionário aninhado em um único nível.
        Ex: {"a": {"b": 1}} → {"a__b": 1}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple)):
                items.append((new_key, str(v)))  # listas viram string
            else:
                items.append((new_key, v))
        return dict(items)

    def load_daily_features(self, symbol: str, date_str: str = None) -> pd.DataFrame:
        """
        Carrega features de um dia específico.
        - date_str: formato "YYYY-MM-DD". Se None, usa hoje.
        """
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        file_path = self.base_dir / f"{symbol}_{date_str}.parquet"

        if not file_path.exists():
            logging.warning(f"⚠️ Arquivo não encontrado: {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(file_path)
            logging.info(f"📂 Carregado {len(df)} registros de {file_path.name}")
            return df
        except Exception as e:
            logging.error(f"❌ Erro ao carregar features: {e}")
            return pd.DataFrame()