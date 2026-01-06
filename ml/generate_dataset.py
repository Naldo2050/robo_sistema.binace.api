# ml/generate_dataset.py
# -*- coding: utf-8 -*-

"""
Gerador de Dataset para Treinamento de Modelos Quantitativos.

Combina:
- Features históricas do FeatureStore (Parquet)
- Decisões da IA (do banco SQLite)
- Labels futuros (direção do preço)

Como usar:
    python ml/generate_dataset.py [--horizon 15] [--threshold-up 0.002] [--threshold-down 0.002]

Saída:
    ml/datasets/training_dataset.parquet
"""

import argparse
import logging
import sqlite3
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import requests

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DatasetGenerator")


# =============================================================================
# CONFIGURAÇÕES (ajuste aqui os parâmetros de label)
# =============================================================================

class DatasetConfig:
    """Configurações centralizadas para geração de dataset."""
    
    # Horizonte de previsão (em minutos)
    HORIZON_MINUTES = 15
    
    # Thresholds para label (em fração, ex: 0.002 = 0.2%)
    THRESHOLD_UP = 0.002      # +0.2% para label +1
    THRESHOLD_DOWN = 0.002    # -0.2% para label -1
    
    # Diretórios
    FEATURES_DIR = Path("features")
    DB_PATH = Path("dados/trading_bot.db")
    OUTPUT_DIR = Path("ml/datasets")
    OUTPUT_FILE = "training_dataset.parquet"
    
    # Colunas para remover do dataset final
    DROP_COLUMNS = [
        "window_id", "saved_at", "date", "symbol", "ativo",
        "timestamp_utc", "epoch_ms", "future_price", "future_return"
    ]
    
    # API Binance para preços (fallback)
    BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


# =============================================================================
# FUNÇÕES DE LABEL
# =============================================================================

def compute_label_direction(
    current_price: float,
    future_price: float,
    threshold_up: float = DatasetConfig.THRESHOLD_UP,
    threshold_down: float = DatasetConfig.THRESHOLD_DOWN
) -> int:
    """
    Calcula label de direção baseado no retorno futuro.
    
    Args:
        current_price: Preço atual
        future_price: Preço no horizonte futuro
        threshold_up: Threshold para label +1 (ex: 0.002 = 0.2%)
        threshold_down: Threshold para label -1 (ex: 0.002 = 0.2%)
        
    Returns:
        +1: Se subir mais que threshold_up
        -1: Se cair mais que threshold_down
         0: Se ficar no meio (ruído)
    """
    if current_price <= 0 or future_price <= 0:
        return 0
    
    ret = (future_price - current_price) / current_price
    
    if ret > threshold_up:
        return 1
    elif ret < -threshold_down:
        return -1
    else:
        return 0


def compute_label_binary(
    current_price: float,
    future_price: float,
    threshold: float = DatasetConfig.THRESHOLD_UP
) -> int:
    """
    Calcula label binário (compatível com train_model.py existente).
    
    Returns:
        1: Se subir mais que threshold
        0: Caso contrário
    """
    if current_price <= 0 or future_price <= 0:
        return 0
    
    ret = (future_price - current_price) / current_price
    return 1 if ret > threshold else 0


# =============================================================================
# CARREGADORES DE DADOS
# =============================================================================

class DatasetGenerator:
    """Classe principal para geração de dataset."""
    
    def __init__(
        self,
        horizon_minutes: int = DatasetConfig.HORIZON_MINUTES,
        threshold_up: float = DatasetConfig.THRESHOLD_UP,
        threshold_down: float = DatasetConfig.THRESHOLD_DOWN,
        output_dir: Path = DatasetConfig.OUTPUT_DIR
    ):
        self.horizon_minutes = horizon_minutes
        self.threshold_up = threshold_up
        self.threshold_down = threshold_down
        self.output_dir = output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Configuração:")
        logger.info(f"  Horizonte: {horizon_minutes} minutos")
        logger.info(f"  Threshold Up: +{threshold_up*100:.2f}%")
        logger.info(f"  Threshold Down: -{threshold_down*100:.2f}%")
    
    def load_features(self) -> Optional[pd.DataFrame]:
        """Carrega todas as features do FeatureStore."""
        logger.info(f"Carregando features de: {DatasetConfig.FEATURES_DIR}")
        
        parquet_files = list(DatasetConfig.FEATURES_DIR.glob("**/*.parquet"))
        
        if not parquet_files:
            logger.error("Nenhum arquivo Parquet encontrado.")
            return None
        
        logger.info(f"Encontrados {len(parquet_files)} arquivos Parquet.")
        
        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
                logger.info(f"  {file.name}: {len(df)} linhas")
            except Exception as e:
                logger.warning(f"Erro ao ler {file}: {e}")
        
        if not dfs:
            logger.error("Nenhum DataFrame carregado.")
            return None
        
        full_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total features: {len(full_df)} linhas, {len(full_df.columns)} colunas")
        
        return full_df
    
    def load_ai_decisions(self) -> Optional[pd.DataFrame]:
        """Carrega decisões da IA do banco SQLite."""
        if not DatasetConfig.DB_PATH.exists():
            logger.warning(f"Banco de dados não encontrado: {DatasetConfig.DB_PATH}")
            return None
        
        logger.info(f"Carregando decisões da IA de: {DatasetConfig.DB_PATH}")
        
        try:
            conn = sqlite3.connect(str(DatasetConfig.DB_PATH))
            cursor = conn.cursor()
            cursor.execute("SELECT payload FROM events ORDER BY timestamp_ms ASC")
            
            ai_records = []
            for (payload_json,) in cursor:
                try:
                    ev = json.loads(payload_json)
                except json.JSONDecodeError:
                    continue
                
                # Filtra eventos de IA
                if (ev.get("tipo_evento") or "").upper() != "AI_ANALYSIS":
                    continue
                
                ai_result = ev.get("ai_result", {})
                ai_payload = ev.get("ai_payload", {})
                
                ts_ms = ev.get("timestamp_ms") or ev.get("epoch_ms")
                
                if ts_ms is None:
                    continue
                
                record = {
                    "timestamp_ms": int(ts_ms),
                    "ai_action": ai_result.get("action", "unknown"),
                    "ai_sentiment": ai_result.get("sentiment", "neutral"),
                    "ai_confidence": ai_result.get("confidence", 0.0),
                    "ai_entry_zone": ai_result.get("entry_zone"),
                    "ai_invalidation_zone": ai_result.get("invalidation_zone"),
                    "anchor_price": ev.get("anchor_price"),
                }
                
                # Adiciona contexto quantitativo se existir
                quant = ai_payload.get("quant_model", {})
                if quant:
                    record["ml_prob_up"] = quant.get("model_probability_up")
                    record["ml_action_bias"] = quant.get("action_bias")
                    record["ml_confidence"] = quant.get("confidence_score")
                
                ai_records.append(record)
            
            conn.close()
            
            if not ai_records:
                logger.warning("Nenhuma decisão de IA encontrada.")
                return None
            
            df_ai = pd.DataFrame(ai_records)
            logger.info(f"Decisões da IA carregadas: {len(df_ai)} eventos")
            
            return df_ai
            
        except Exception as e:
            logger.error(f"Erro ao acessar banco: {e}")
            return None
    
    def find_price_column(self, df: pd.DataFrame) -> Optional[str]:
        """Encontra coluna de preço no DataFrame."""
        candidates = [
            "enriched.ohlc.close",
            "price_close",
            "close",
            "p",
            "last_price",
            "price"
        ]
        
        for col in candidates:
            if col in df.columns:
                logger.info(f"Coluna de preço identificada: {col}")
                return col
        
        # Busca parcial
        for col in df.columns:
            if any(term in col.lower() for term in ['close', 'price']):
                logger.info(f"Coluna de preço (match parcial): {col}")
                return col
        
        return None
    
    def find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Encontra coluna de timestamp no DataFrame."""
        candidates = ["saved_at", "timestamp", "epoch_ms", "timestamp_utc"]
        
        for col in candidates:
            if col in df.columns:
                return col
        
        return None
    
    def add_future_prices(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Adiciona preço futuro e calcula labels."""
        logger.info(f"Calculando preços futuros (horizonte: {self.horizon_minutes} janelas)...")
        
        # Preço futuro (shift negativo)
        df["future_price"] = df[price_col].shift(-self.horizon_minutes)
        
        # Retorno futuro
        df["future_return"] = (df["future_price"] - df[price_col]) / df[price_col]
        
        # Labels
        df["label_direction"] = df.apply(
            lambda row: compute_label_direction(
                row[price_col] if pd.notna(row[price_col]) else 0,
                row["future_price"] if pd.notna(row["future_price"]) else 0,
                self.threshold_up,
                self.threshold_down
            ),
            axis=1
        )
        
        df["label_binary"] = df.apply(
            lambda row: compute_label_binary(
                row[price_col] if pd.notna(row[price_col]) else 0,
                row["future_price"] if pd.notna(row["future_price"]) else 0,
                self.threshold_up
            ),
            axis=1
        )
        
        # Remove linhas sem futuro (últimas N)
        df = df.dropna(subset=["future_return"]).copy()
        
        logger.info(f"Após cálculo de labels: {len(df)} linhas")
        
        # Estatísticas de labels
        if len(df) > 0:
            label_counts = df["label_direction"].value_counts()
            logger.info(f"Distribuição label_direction:")
            logger.info(f"  +1 (up):   {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
            logger.info(f"   0 (flat): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
            logger.info(f"  -1 (down): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def merge_with_ai_decisions(
        self,
        df_features: pd.DataFrame,
        df_ai: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Combina features com decisões da IA (quando disponível)."""
        if df_ai is None or len(df_ai) == 0:
            logger.warning("Sem decisões de IA para merge. Adicionando colunas vazias.")
            df_features["ai_action"] = None
            df_features["ai_sentiment"] = None
            df_features["ai_confidence"] = None
            return df_features
        
        # Encontra coluna de timestamp nas features
        ts_col = self.find_timestamp_column(df_features)
        
        if ts_col is None:
            logger.warning("Sem coluna de timestamp nas features. Merge não possível.")
            df_features["ai_action"] = None
            df_features["ai_sentiment"] = None
            df_features["ai_confidence"] = None
            return df_features
        
        logger.info(f"Fazendo merge por timestamp (coluna: {ts_col})...")
        
        # Converte timestamps para ms
        df_features = df_features.copy()
        df_features["_ts_ms"] = pd.to_datetime(df_features[ts_col]).astype("int64") // 10**6
        
        # Merge asof (closest match)
        df_features = df_features.sort_values("_ts_ms")
        df_ai = df_ai.sort_values("timestamp_ms")
        
        df_merged = pd.merge_asof(
            df_features,
            df_ai,
            left_on="_ts_ms",
            right_on="timestamp_ms",
            direction="nearest",
            tolerance=300000  # 5 minutos de tolerância
        )
        
        df_merged.drop(columns=["_ts_ms", "timestamp_ms"], errors="ignore", inplace=True)
        
        matched = df_merged["ai_action"].notna().sum()
        logger.info(f"Merge concluído: {matched}/{len(df_merged)} linhas com decisão da IA")
        
        return df_merged
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa dataset removendo colunas desnecessárias."""
        logger.info("Limpando dataset...")
        
        # Remove colunas de drop
        cols_to_drop = [col for col in DatasetConfig.DROP_COLUMNS if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors="ignore")
            logger.info(f"Removidas {len(cols_to_drop)} colunas: {cols_to_drop}")
        
        # Remove colunas vazias ou com única valor
        initial_cols = len(df.columns)
        
        # Remove colunas com todos NaN
        df = df.dropna(axis=1, how="all")
        
        # Remove colunas com variância zero (único valor)
        nunique = df.nunique()
        cols_single = nunique[nunique <= 1].index.tolist()
        if cols_single:
            df = df.drop(columns=cols_single, errors="ignore")
        
        logger.info(f"Colunas após limpeza: {len(df.columns)} (removidas {initial_cols - len(df.columns)})")
        
        return df
    
    def generate(self) -> Optional[Path]:
        """Executa pipeline completo de geração de dataset."""
        logger.info("="*60)
        logger.info("GERAÇÃO DE DATASET PARA TREINAMENTO")
        logger.info("="*60)
        
        # 1. Carrega features
        df_features = self.load_features()
        if df_features is None or len(df_features) == 0:
            logger.error("Falha ao carregar features.")
            return None
        
        # 2. Encontra coluna de preço
        price_col = self.find_price_column(df_features)
        if price_col is None:
            logger.error("Coluna de preço não encontrada.")
            return None
        
        # 3. Ordena por timestamp
        ts_col = self.find_timestamp_column(df_features)
        if ts_col:
            df_features[ts_col] = pd.to_datetime(df_features[ts_col], errors="coerce")
            df_features = df_features.sort_values(ts_col).reset_index(drop=True)
            logger.info(f"Dados ordenados por: {ts_col}")
        
        # 4. Adiciona preços futuros e labels
        df_features = self.add_future_prices(df_features, price_col)
        
        # 5. Carrega e merge com decisões da IA
        df_ai = self.load_ai_decisions()
        df_final = self.merge_with_ai_decisions(df_features, df_ai)
        
        # 6. Limpa dataset
        df_final = self.clean_dataset(df_final)
        
        # 7. Remove duplicatas
        initial_len = len(df_final)
        df_final = df_final.drop_duplicates()
        if len(df_final) < initial_len:
            logger.info(f"Removidas {initial_len - len(df_final)} duplicatas")
        
        # 8. Salva dataset
        output_path = self.output_dir / DatasetConfig.OUTPUT_FILE
        
        try:
            df_final.to_parquet(output_path, index=False, compression="snappy")
            logger.info(f"Dataset salvo: {output_path}")
            logger.info(f"  Linhas: {len(df_final)}")
            logger.info(f"  Colunas: {len(df_final.columns)}")
        except Exception as e:
            logger.error(f"Erro ao salvar Parquet: {e}")
            # Fallback para CSV
            csv_path = output_path.with_suffix(".csv")
            df_final.to_csv(csv_path, index=False)
            logger.info(f"Fallback CSV salvo: {csv_path}")
            output_path = csv_path
        
        # 9. Imprime resumo das colunas
        self._print_column_summary(df_final)
        
        return output_path
    
    def _print_column_summary(self, df: pd.DataFrame):
        """Imprime resumo das colunas do dataset."""
        print("\n" + "="*60)
        print("ESTRUTURA DO DATASET")
        print("="*60)
        
        # Agrupa colunas por tipo
        feature_cols = [c for c in df.columns if c not in [
            "label_direction", "label_binary", "future_return",
            "ai_action", "ai_sentiment", "ai_confidence",
            "ai_entry_zone", "ai_invalidation_zone", "anchor_price",
            "ml_prob_up", "ml_action_bias", "ml_confidence"
        ]]
        
        ai_cols = [c for c in df.columns if c.startswith("ai_")]
        ml_cols = [c for c in df.columns if c.startswith("ml_")]
        label_cols = [c for c in df.columns if c.startswith("label_") or c == "future_return"]
        
        print(f"\nFeatures ({len(feature_cols)} colunas):")
        print(f"  {', '.join(feature_cols[:10])}...")
        
        print(f"\nDecisões da IA ({len(ai_cols)} colunas):")
        for col in ai_cols:
            print(f"  - {col}")
        
        print(f"\nContexto ML ({len(ml_cols)} colunas):")
        for col in ml_cols:
            print(f"  - {col}")
        
        print(f"\nLabels ({len(label_cols)} colunas):")
        for col in label_cols:
            print(f"  - {col}")
        
        print("="*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gera dataset para treinamento de modelos quantitativos."
    )
    parser.add_argument(
        "--horizon", "-n",
        type=int,
        default=DatasetConfig.HORIZON_MINUTES,
        help=f"Horizonte de previsão em janelas/minutos (default: {DatasetConfig.HORIZON_MINUTES})"
    )
    parser.add_argument(
        "--threshold-up", "-u",
        type=float,
        default=DatasetConfig.THRESHOLD_UP,
        help=f"Threshold para label +1 (default: {DatasetConfig.THRESHOLD_UP})"
    )
    parser.add_argument(
        "--threshold-down", "-d",
        type=float,
        default=DatasetConfig.THRESHOLD_DOWN,
        help=f"Threshold para label -1 (default: {DatasetConfig.THRESHOLD_DOWN})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(DatasetConfig.OUTPUT_DIR),
        help=f"Diretório de saída (default: {DatasetConfig.OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(
        horizon_minutes=args.horizon,
        threshold_up=args.threshold_up,
        threshold_down=args.threshold_down,
        output_dir=Path(args.output_dir)
    )
    
    output_path = generator.generate()
    
    if output_path:
        print(f"\n✅ Dataset gerado com sucesso: {output_path}")
        print(f"\nPara ajustar parâmetros, edite:")
        print(f"  - Horizonte: --horizon N (ou DatasetConfig.HORIZON_MINUTES)")
        print(f"  - Thresholds: --threshold-up X --threshold-down Y")
        print(f"\nPróximos passos:")
        print(f"  python ml/train_model.py  # Para treinar modelo XGBoost")
    else:
        print("\n❌ Falha na geração do dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
