# ml/train_model.py
"""
Pipeline de Treinamento de ML Institucional com validação rigorosa.
"""

import logging
import json
import warnings
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
import joblib

# Configurações
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ml/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MLTrainer")


class ModelTrainer:
    """Classe principal para treinamento de modelos."""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        
        # Diretórios
        self.features_dir = Path("features")
        self.models_dir = Path("ml/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_path = self.models_dir / f"xgb_model_{timestamp}.json"
        self.metadata_path = self.models_dir / f"model_metadata_{timestamp}.json"
        self.scaler_path = self.models_dir / f"scaler_{timestamp}.pkl"
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configurações do arquivo YAML."""
        default_config = {
            "model": {
                "lookahead_windows": 15,
                "min_return_threshold": 0.002,
                "test_size": 0.2,
                "validation_size": 0.1,
                "random_state": 42,
            },
            "xgboost": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
                "n_jobs": -1,
                "random_state": 42,
                "scale_pos_weight": 1,
            },
            "features": {
                "required_columns": [],
                "drop_columns": [
                    "window_id", "saved_at", "symbol", 
                    "timestamp", "timestamp_utc", "epoch_ms"
                ],
                "max_null_percentage": 0.3,
                "correlation_threshold": 0.95,
            },
            "sampling": {
                "use_smote": False,
                "smote_ratio": 0.5,
            }
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Merge com default
                for key in default_config:
                    if key in config:
                        default_config[key].update(config[key])
            logger.info(f"Configuração carregada: {config_path}")
        else:
            logger.warning(f"Arquivo de configuração não encontrado. Usando default.")
            
        return default_config
    
    def load_and_validate_data(self) -> Optional[pd.DataFrame]:
        """
        Carrega dados do FeatureStore com validação rigorosa.
        
        Returns:
            DataFrame validado ou None em caso de erro.
        """
        logger.info(f"Carregando dados de: {self.features_dir}")
        
        # Encontra arquivos Parquet
        parquet_files = list(self.features_dir.glob("**/*.parquet"))
        
        if not parquet_files:
            logger.error("Nenhum arquivo Parquet encontrado.")
            return None
        
        logger.info(f"Encontrados {len(parquet_files)} arquivos.")
        
        # Carrega e concatena
        dfs = []
        for file in parquet_files:
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Erro ao ler {file}: {e}")
        
        if not dfs:
            logger.error("Nenhum DataFrame válido carregado.")
            return None
        
        full_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Dados brutos: {len(full_df)} linhas, {len(full_df.columns)} colunas")
        
        # Validação de dados
        validation_result = self._validate_data(full_df)
        if not validation_result["valid"]:
            logger.error(f"Dados inválidos: {validation_result['errors']}")
            return None
        
        # Ordenação temporal
        df_sorted = self._sort_by_timestamp(full_df)
        
        # Remove duplicatas
        initial_len = len(df_sorted)
        df_clean = df_sorted.drop_duplicates()
        if len(df_clean) < initial_len:
            logger.warning(f"Removidas {initial_len - len(df_clean)} duplicatas.")
        
        logger.info(f"Dados limpos: {len(df_clean)} linhas")
        return df_clean
    
    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida qualidade dos dados."""
        errors = []
        warnings_list = []
        
        # Verifica colunas obrigatórias
        required = self.config["features"].get("required_columns", [])
        if required:
            missing = [col for col in required if col not in df.columns]
            if missing:
                errors.append(f"Colunas obrigatórias faltando: {missing}")
        
        # Garante que exista alguma coluna de preço identificável
        price_col = self._find_price_column(df)
        if not price_col:
            errors.append("Nenhuma coluna de preço encontrada (close/price).")
        
        # Verifica valores nulos
        null_percentage = df.isnull().mean()
        high_null_cols = null_percentage[null_percentage > 0.3].index.tolist()
        if high_null_cols:
            warnings_list.append(f"Colunas com >30% nulos: {high_null_cols[:5]}")
        
        # Verifica tamanho mínimo
        if len(df) < 100:
            errors.append(f"Dados insuficientes: {len(df)} linhas (mínimo: 100)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings_list,
            "null_percentage": null_percentage.to_dict(),
            "shape": df.shape
        }
    
    def _sort_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordena DataFrame por timestamp."""
        timestamp_cols = ["saved_at", "timestamp", "epoch_ms"]
        
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    if col == "saved_at":
                        df[col] = pd.to_datetime(df[col])
                    df = df.sort_values(col).reset_index(drop=True)
                    logger.info(f"Dados ordenados por: {col}")
                    return df
                except Exception as e:
                    logger.warning(f"Erro ao ordenar por {col}: {e}")
        
        logger.warning("Nenhuma coluna temporal encontrada. Mantendo ordem original.")
        return df
    
    def prepare_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepara features e targets com feature engineering.
        """
        logger.info("Preparando features e targets...")
        
        # Identifica coluna de preço
        price_col = self._find_price_column(df)
        if not price_col:
            logger.error("Coluna de preço não encontrada.")
            return None, None, None
        
        # Remove nulos no preço
        df = df.dropna(subset=[price_col]).copy()
        
        # Calcula retorno futuro
        lookahead = self.config["model"]["lookahead_windows"]
        threshold = self.config["model"]["min_return_threshold"]
        
        df["future_price"] = df[price_col].shift(-lookahead)
        df["future_return"] = (df["future_price"] - df[price_col]) / df[price_col]
        
        # Remove últimas N linhas sem futuro
        df = df.dropna(subset=["future_return"]).copy()
        
        # Cria target binário
        df["target"] = (df["future_return"] > threshold).astype(int)
        
        # Balanceamento
        pos = df["target"].sum()
        total = len(df)
        logger.info(f"Balanceamento: {pos}/{total} positivos ({pos/total:.1%})")
        
        # Feature engineering
        df = self._create_features(df, price_col)
        
        # Separa features e target
        drop_cols = self.config["features"]["drop_columns"] + [
            "target", "future_return", "future_price"
        ]
        
        # Remove colunas com muitos nulos
        null_threshold = self.config["features"]["max_null_percentage"]
        valid_cols = df.columns[df.isnull().mean() <= null_threshold]
        df = df[valid_cols]
        
        # Seleciona features numéricas
        feature_cols = [
            col for col in df.columns 
            if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        # Remove features altamente correlacionadas
        feature_cols = self._remove_highly_correlated(df[feature_cols])
        
        X = df[feature_cols]
        y = df["target"]
        
        # Preenche nulos restantes
        X = X.fillna(X.median())
        
        logger.info(f"Features: {len(feature_cols)}, Shape: {X.shape}")
        return X, y, feature_cols
    
    def _find_price_column(self, df: pd.DataFrame) -> Optional[str]:
        """Encontra coluna de preço."""
        price_candidates = [
            "enriched.ohlc.close",
            "price_close",
            "close",
            "p",
            "last_price",
            "price"
        ]
        
        for col in price_candidates:
            if col in df.columns:
                logger.info(f"Usando coluna de preço: {col}")
                return col
        
        # Tenta encontrar coluna que contenha 'close' ou 'price'
        for col in df.columns:
            if any(term in col.lower() for term in ['close', 'price', 'p_']):
                logger.info(f"Usando coluna de preço (match parcial): {col}")
                return col
        
        return None
    
    def _create_features(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Cria features técnicas."""
        logger.info("Criando features técnicas...")
        
        # Returns
        df["return_1"] = df[price_col].pct_change(1)
        df["return_5"] = df[price_col].pct_change(5)
        df["return_10"] = df[price_col].pct_change(10)
        
        # Médias móveis
        df["sma_5"] = df[price_col].rolling(5).mean()
        df["sma_10"] = df[price_col].rolling(10).mean()
        df["sma_20"] = df[price_col].rolling(20).mean()
        
        # Bollinger Bands
        df["bb_middle"] = df[price_col].rolling(20).mean()
        bb_std = df[price_col].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        
        # RSI (simplificado)
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # Volume features (se disponível)
        if "volume" in df.columns:
            df["volume_sma"] = df["volume"].rolling(20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        logger.info(f"Features criadas. Total colunas: {len(df.columns)}")
        return df
    
    def _remove_highly_correlated(self, X: pd.DataFrame) -> list:
        """Remove features altamente correlacionadas."""
        threshold = self.config["features"]["correlation_threshold"]
        
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            logger.info(f"Removendo {len(to_drop)} features altamente correlacionadas")
            return [col for col in X.columns if col not in to_drop]
        
        return list(X.columns)
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Treina modelo XGBoost com validação temporal."""
        logger.info("Iniciando treinamento do modelo...")
        
        # Split temporal (manter ordem)
        test_size = self.config["model"]["test_size"]
        val_size = self.config["model"]["validation_size"]
        
        n = len(X)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        X_train, X_val = X.iloc[:train_end], X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_train, y_val = y.iloc[:train_end], y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]
        
        logger.info(f"Treino: {len(X_train)}, Validação: {len(X_val)}, Teste: {len(X_test)}")
        
        # Balanceamento (SMOTE opcional)
        if self.config["sampling"]["use_smote"]:
            if SMOTE is None:
                logger.warning("use_smote=True, mas imbalanced-learn não está instalado. Ignorando SMOTE.")
            else:
                logger.info("Aplicando SMOTE para balanceamento...")
                smote = SMOTE(sampling_strategy=self.config["sampling"]["smote_ratio"])
                X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Configura modelo
        xgb_params = self.config["xgboost"].copy()
        
        # Ajusta scale_pos_weight baseado no desbalanceamento (com proteção)
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        if n_pos == 0:
            pos_weight = 1.0
        else:
            pos_weight = n_neg / max(n_pos, 1)
        xgb_params["scale_pos_weight"] = pos_weight
        
        # Cria modelo
        model = xgb.XGBClassifier(**xgb_params)
        
        # Compatível com versões mais antigas do xgboost (sem early_stopping_rounds)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
        
        # Avaliação
        self._evaluate_model(model, X_test, y_test, X_val, y_val)
        
        return model
    
    def _evaluate_model(self, model, X_test, y_test, X_val, y_val):
        """Avalia modelo com métricas detalhadas."""
        logger.info("\n" + "="*60)
        logger.info("AVALIAÇÃO DO MODELO")
        logger.info("="*60)
        
        # Previsões
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas básicas
        logger.info(f"\nRelatório de Classificação (Teste):")
        logger.info(classification_report(y_test, y_pred))
        
        # Métricas adicionais
        metrics = {
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_proba),
            "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        
        for name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{name}: {value:.3f}")
        
        # Precisão por nível de confiança
        logger.info("\nPrecisão por Nível de Confiança:")
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for t in thresholds:
            mask = y_proba >= t
            if mask.sum() > 0:
                precision_t = precision_score(y_test[mask], y_pred[mask])
                logger.info(f"  Conf >= {t:.1f}: {precision_t:.1%} ({mask.sum()} amostras)")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Features:")
        logger.info(self.feature_importance.head(10).to_string())
        
        self.metrics = metrics
    
    def save_artifacts(self, model, feature_names):
        """Salva todos os artefatos do modelo."""
        logger.info("Salvando artefatos do modelo...")
        
        # 1. Salva modelo
        model.save_model(str(self.model_path))
        
        # 2. Salva metadados
        metadata = {
            "created_at": datetime.now().isoformat(),
            "model_type": "XGBoost",
            "feature_names": list(feature_names),
            "config": self.config,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance.to_dict('records')[:20],
            "training_info": {
                "n_samples": len(feature_names),
                "n_features": len(feature_names),
                "positive_class_ratio": self.feature_importance.get('positive_ratio', 'N/A')
            }
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 3. Cria symlink para modelo mais recente
        latest_path = self.models_dir / "xgb_model_latest.json"
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(self.model_path.name)
            logger.info(f"Symlink criado: {latest_path} -> {self.model_path.name}")
        except Exception as e:
            logger.warning(f"Falha ao criar symlink, copiando arquivo em vez disso: {e}")
            import shutil
            shutil.copy2(self.model_path, latest_path)
            logger.info(f"Arquivo copiado para: {latest_path}")
    
    def run_pipeline(self):
        """Executa pipeline completo de treinamento."""
        logger.info("="*60)
        logger.info("INICIANDO PIPELINE DE TREINAMENTO")
        logger.info("="*60)
        
        try:
            # 1. Carrega dados
            df = self.load_and_validate_data()
            if df is None:
                return False
            
            # 2. Prepara features
            X, y, features = self.prepare_features_targets(df)
            if X is None:
                return False
            
            # 3. Treina modelo
            model = self.train_model(X, y)
            
            # 4. Salva artefatos
            self.save_artifacts(model, features)
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE CONCLUÍDO COM SUCESSO!")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro no pipeline: {e}", exc_info=True)
            return False


def main():
    """Função principal."""
    trainer = ModelTrainer(config_path="config/model_config.yaml")
    
    success = trainer.run_pipeline()
    
    if success:
        logger.info("Modelo treinado e salvo com sucesso!")
        return 0
    else:
        logger.error("Falha no treinamento do modelo.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)