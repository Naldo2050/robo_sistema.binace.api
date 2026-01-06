# ml/train_model.py
"""
Pipeline de Treinamento de ML Institucional com validação rigorosa.
"""

import logging
import json
import warnings
import pickle
import io
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

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
    precision_recall_curve,
    f1_score
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
        self.training_log = io.StringIO()
        
        # Diretórios
        self.features_dir = Path("features")
        self.models_dir = Path("ml/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging adicional
        self._setup_additional_logging()
        
        # Paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_path = self.models_dir / f"xgb_model_{timestamp}.json"
        self.metadata_path = self.models_dir / f"model_metadata_{timestamp}.json"
        self.scaler_path = self.models_dir / f"scaler_{timestamp}.pkl"
        
    def _setup_additional_logging(self):
        """Configura logging adicional para capturar output."""
        handler = logging.StreamHandler(self.training_log)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configurações do arquivo YAML."""
        default_config = {
            "model": {
                "lookahead_windows": 15,
                "min_return_threshold": 0.002,
                "test_size": 0.2,
                "validation_size": 0.15,
                "random_state": 42,
                "early_stopping_rounds": 50,
                "cv_folds": 5,
            },
            "xgboost": {
                "n_estimators": 1000,
                "learning_rate": 0.01,
                "max_depth": 4,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "eval_metric": ["logloss", "error"],
                "n_jobs": -1,
                "random_state": 42,
                "use_label_encoder": False,
                "verbosity": 0,
            },
            "features": {
                "required_columns": [],
                "drop_columns": [
                    "window_id", "saved_at", "symbol", 
                    "timestamp", "timestamp_utc", "epoch_ms"
                ],
                "max_null_percentage": 0.2,
                "correlation_threshold": 0.85,
            },
            "sampling": {
                "use_smote": False,
                "smote_ratio": 0.5,
            },
            "validation": {
                "min_class_balance": 0.1,
                "max_overfitting_threshold": 0.1,
            },
            "data": {
                "chunk_size": 100000,
                "max_file_size_mb": 100,
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
        
        # Carrega em chunks para evitar problemas de memória
        dfs = []
        max_size_mb = self.config["data"]["max_file_size_mb"]
        chunk_size = self.config["data"]["chunk_size"]
        
        for file in parquet_files:
            try:
                file_size_mb = file.stat().st_size / (1024 * 1024)
                
                if file_size_mb > max_size_mb:
                    logger.info(f"Lendo arquivo grande em chunks: {file.name} ({file_size_mb:.1f} MB)")
                    # Ler em chunks
                    reader = pd.read_parquet(file, chunksize=chunk_size)
                    for i, chunk in enumerate(reader):
                        dfs.append(chunk)
                        if i % 10 == 0:
                            logger.info(f"  Processado chunk {i+1}")
                else:
                    df = pd.read_parquet(file)
                    dfs.append(df)
                    logger.info(f"  Carregado: {file.name}")
                    
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
        
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(warning)
        
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
        null_threshold = self.config["features"]["max_null_percentage"]
        high_null_cols = null_percentage[null_percentage > null_threshold].index.tolist()
        if high_null_cols:
            warnings_list.append(f"Colunas com >{null_threshold*100:.0f}% nulos: {high_null_cols[:5]}")
        
        # Verifica tamanho mínimo
        if len(df) < 100:
            errors.append(f"Dados insuficientes: {len(df)} linhas (mínimo: 100)")
        
        # Verifica se há dados temporais
        timestamp_cols = ["saved_at", "timestamp", "epoch_ms"]
        has_timestamp = any(col in df.columns for col in timestamp_cols)
        if not has_timestamp:
            warnings_list.append("Nenhuma coluna temporal encontrada")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings_list,
            "null_percentage": null_percentage.to_dict(),
            "shape": df.shape
        }
    
    def _sort_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordena DataFrame por timestamp com validação robusta."""
        timestamp_cols = ["saved_at", "timestamp", "epoch_ms"]
        
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    if col == "saved_at":
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif col == "epoch_ms":
                        df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')
                    else:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    # Verificar se a conversão foi bem-sucedida
                    null_count = df[col].isnull().sum()
                    if null_count == len(df):
                        logger.warning(f"Coluna {col} não pôde ser convertida para datetime")
                        continue
                    
                    if null_count > 0:
                        logger.warning(f"Coluna {col} tem {null_count} valores nulos após conversão")
                    
                    df = df.sort_values(col).reset_index(drop=True)
                    logger.info(f"Dados ordenados por: {col} (nulos: {null_count})")
                    return df
                except Exception as e:
                    logger.warning(f"Erro ao ordenar por {col}: {e}")
                    continue
        
        logger.warning("Nenhuma coluna temporal válida encontrada. Mantendo ordem original.")
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
        initial_len = len(df)
        df = df.dropna(subset=[price_col]).copy()
        if len(df) < initial_len:
            logger.warning(f"Removidas {initial_len - len(df)} linhas com preço nulo")
        
        # Calcula retorno futuro
        lookahead = self.config["model"]["lookahead_windows"]
        threshold = self.config["model"]["min_return_threshold"]
        
        df["future_price"] = df[price_col].shift(-lookahead)
        df["future_return"] = (df["future_price"] - df[price_col]) / df[price_col]
        
        # Remove últimas N linhas sem futuro
        df = df.dropna(subset=["future_return"]).copy()
        logger.info(f"Dados após cálculo de futuro: {len(df)} linhas")
        
        # Cria target binário
        df["target"] = (df["future_return"] > threshold).astype(int)
        
        # Balanceamento
        pos = df["target"].sum()
        total = len(df)
        pos_ratio = pos / total if total > 0 else 0
        logger.info(f"Balanceamento: {pos}/{total} positivos ({pos_ratio:.1%})")
        
        # Verificar balanceamento mínimo
        min_balance = self.config["validation"]["min_class_balance"]
        if 0 < pos_ratio < min_balance or pos_ratio > (1 - min_balance):
            logger.warning(f"Balanceamento muito desequilibrado: {pos_ratio:.1%}")
        
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
        logger.info(f"Distribuição do target: {y.value_counts().to_dict()}")
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
        """Cria features técnicas com tratamento de erros."""
        logger.info("Criando features técnicas...")
        
        # Garantir que temos dados suficientes
        min_samples = 50
        if len(df) < min_samples:
            logger.warning(f"Dados insuficientes para feature engineering: {len(df)} < {min_samples}")
            return df
        
        original_cols = set(df.columns)
        
        try:
            # Returns com tratamento de divisão por zero
            for window in [1, 5, 10]:
                col_name = f"return_{window}"
                df[col_name] = df[price_col].pct_change(window)
                # Substituir inf/nan
                df[col_name].replace([np.inf, -np.inf], np.nan, inplace=True)
                logger.debug(f"Criada feature: {col_name}")
        except Exception as e:
            logger.warning(f"Erro ao calcular returns: {e}")
        
        try:
            # Médias móveis
            for window in [5, 10, 20]:
                col_name = f"sma_{window}"
                df[col_name] = df[price_col].rolling(window, min_periods=1).mean()
                logger.debug(f"Criada feature: {col_name}")
        except Exception as e:
            logger.warning(f"Erro ao calcular médias móveis: {e}")
        
        try:
            # Bollinger Bands
            window_bb = 20
            df["bb_middle"] = df[price_col].rolling(window_bb, min_periods=1).mean()
            bb_std = df[price_col].rolling(window_bb, min_periods=1).std()
            df["bb_upper"] = df["bb_middle"] + 2 * bb_std
            df["bb_lower"] = df["bb_middle"] - 2 * bb_std
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
            logger.debug("Criadas features: Bollinger Bands")
        except Exception as e:
            logger.warning(f"Erro ao calcular Bollinger Bands: {e}")
        
        try:
            # RSI (simplificado)
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
            df["rsi"].replace([np.inf, -np.inf], 50, inplace=True)  # Valor neutro para inf
            logger.debug("Criada feature: RSI")
        except Exception as e:
            logger.warning(f"Erro ao calcular RSI: {e}")
        
        # Volume features (se disponível)
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        if volume_cols:
            volume_col = volume_cols[0]  # Pega a primeira coluna de volume
            try:
                df["volume_sma"] = df[volume_col].rolling(20, min_periods=1).mean()
                df["volume_ratio"] = df[volume_col] / df["volume_sma"]
                df["volume_ratio"].replace([np.inf, -np.inf], 1, inplace=True)
                logger.debug("Criadas features: volume")
            except Exception as e:
                logger.warning(f"Erro ao calcular features de volume: {e}")
        
        # Log das novas features criadas
        new_cols = set(df.columns) - original_cols
        logger.info(f"Criadas {len(new_cols)} novas features")
        
        return df
    
    def _remove_highly_correlated(self, X: pd.DataFrame) -> list:
        """Remove features altamente correlacionadas."""
        threshold = self.config["features"]["correlation_threshold"]
        
        # Calcula matriz de correlação
        corr_matrix = X.corr().abs()
        
        # Seleciona triângulo superior
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Encontra colunas para remover
        to_drop = []
        for column in upper.columns:
            if any(upper[column] > threshold):
                to_drop.append(column)
        
        if to_drop:
            logger.info(f"Removendo {len(to_drop)} features altamente correlacionadas: {to_drop}")
            return [col for col in X.columns if col not in to_drop]
        
        return list(X.columns)
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Treina modelo XGBoost com validação temporal cruzada."""
        logger.info("Iniciando treinamento do modelo...")

        # Split temporal inicial para validação holdout
        test_size = self.config["model"]["test_size"]
        val_size = self.config["model"]["validation_size"]

        n = len(X)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        X_train_full, X_val = X.iloc[:train_end], X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_train_full, y_val = y.iloc[:train_end], y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]

        logger.info(f"Treino: {len(X_train_full)}, Validação: {len(X_val)}, Teste: {len(X_test)}")

        # Configura modelo
        xgb_params = self.config["xgboost"].copy()

        # Ajusta scale_pos_weight baseado no desbalanceamento
        n_pos = (y_train_full == 1).sum()
        n_neg = (y_train_full == 0).sum()
        if n_pos == 0:
            pos_weight = 1.0
        else:
            pos_weight = n_neg / max(n_pos, 1)
        xgb_params["scale_pos_weight"] = pos_weight

        # ===== TRATAMENTO PARA POUCOS DADOS =====
        n_train_samples = len(X_train_full)
        cv_folds_cfg = self.config["model"]["cv_folds"]

        if n_train_samples < 5:
            logger.warning(
                f"Dados de treino muito pequenos ({n_train_samples} amostras). "
                f"Pulando cross-validation e treinando modelo direto."
            )
        else:
            # Garante que o número de folds não ultrapasse o permitido
            # (TimeSeriesSplit exige n_splits < n_samples)
            max_folds_allowed = max(2, n_train_samples - 1)
            cv_folds = min(cv_folds_cfg, max_folds_allowed)

            if cv_folds != cv_folds_cfg:
                logger.warning(
                    f"cv_folds={cv_folds_cfg} ajustado para {cv_folds} "
                    f"devido ao número reduzido de amostras ({n_train_samples})."
                )

            tscv = TimeSeriesSplit(n_splits=cv_folds)

            cv_scores = []
            best_score = -np.inf
            best_model = None

            logger.info(f"Executando {cv_folds}-fold TimeSeries Cross-Validation")

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full)):
                X_train = X_train_full.iloc[train_idx]
                X_val_cv = X_train_full.iloc[val_idx]
                y_train = y_train_full.iloc[train_idx]
                y_val_cv = y_train_full.iloc[val_idx]

                logger.info(f"Fold {fold+1}: Treino={len(X_train)}, Validação={len(X_val_cv)}")

                # Balanceamento (SMOTE opcional)
                if self.config["sampling"]["use_smote"]:
                    if SMOTE is None:
                        logger.warning("use_smote=True, mas imbalanced-learn não está instalado. Ignorando SMOTE.")
                    else:
                        logger.info("Aplicando SMOTE para balanceamento...")
                        smote = SMOTE(sampling_strategy=self.config["sampling"]["smote_ratio"])
                        X_train, y_train = smote.fit_resample(X_train, y_train)

                # Cria e treina modelo
                model = xgb.XGBClassifier(**xgb_params)

                try:
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val_cv, y_val_cv)],
                        early_stopping_rounds=self.config["model"]["early_stopping_rounds"],
                        verbose=50,
                    )
                except TypeError:
                    # Compatibilidade com versões antigas do xgboost que não aceitam early_stopping_rounds
                    logger.warning(
                        "Versão do XGBoost não suporta 'early_stopping_rounds' na API XGBClassifier.fit. "
                        "Treinando este fold sem early stopping."
                    )
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val_cv, y_val_cv)],
                        verbose=50,
                    )

                # Avaliação no fold
                y_pred = model.predict(X_val_cv)
                score = precision_score(y_val_cv, y_pred)
                cv_scores.append(score)

                logger.info(f"Fold {fold+1}: Precision = {score:.3f}")

                # Mantém o melhor modelo (não é usado depois, mas mantemos para possível extensão futura)
                if score > best_score:
                    best_score = score
                    best_model = model

            # Estatísticas da CV
            logger.info(f"\nCross-Validation Results:")
            logger.info(f"Scores: {cv_scores}")
            logger.info(f"Média: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")

        # Treina modelo final com todos os dados de treino
        logger.info("\nTreinando modelo final com todos os dados de treino...")

        if self.config["sampling"]["use_smote"] and SMOTE is not None:
            smote = SMOTE(sampling_strategy=self.config["sampling"]["smote_ratio"])
            X_train_full_resampled, y_train_full_resampled = smote.fit_resample(X_train_full, y_train_full)
        else:
            X_train_full_resampled, y_train_full_resampled = X_train_full, y_train_full

        final_model = xgb.XGBClassifier(**xgb_params)
        try:
            final_model.fit(
                X_train_full_resampled,
                y_train_full_resampled,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.config["model"]["early_stopping_rounds"],
                verbose=50,
            )
        except TypeError:
            logger.warning(
                "Versão do XGBoost não suporta 'early_stopping_rounds' na API XGBClassifier.fit. "
                "Treinando modelo final sem early stopping."
            )
            final_model.fit(
                X_train_full_resampled,
                y_train_full_resampled,
                eval_set=[(X_val, y_val)],
                verbose=50,
            )

        # Avaliação final
        self._evaluate_model(final_model, X_train_full, X_val, X_test, y_train_full, y_val, y_test)

        return final_model
    
    def _evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        """Avalia modelo com métricas detalhadas e verificação de overfitting."""
        logger.info("\n" + "="*60)
        logger.info("AVALIAÇÃO DO MODELO")
        logger.info("="*60)
        
        # Previsões
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_val = model.predict_proba(X_val)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]
        
        # Métricas básicas
        logger.info(f"\nRelatório de Classificação (Treino):")
        logger.info(classification_report(y_train, y_pred_train))
        
        logger.info(f"\nRelatório de Classificação (Validação):")
        logger.info(classification_report(y_val, y_pred_val))
        
        logger.info(f"\nRelatório de Classificação (Teste):")
        logger.info(classification_report(y_test, y_pred_test))
        
        # Métricas adicionais
        metrics = {
            "Train_Precision": precision_score(y_train, y_pred_train),
            "Train_Recall": recall_score(y_train, y_pred_train),
            "Train_F1": f1_score(y_train, y_pred_train),
            "Train_AUC_ROC": roc_auc_score(y_train, y_proba_train),
            
            "Val_Precision": precision_score(y_val, y_pred_val),
            "Val_Recall": recall_score(y_val, y_pred_val),
            "Val_F1": f1_score(y_val, y_pred_val),
            "Val_AUC_ROC": roc_auc_score(y_val, y_proba_val),
            
            "Test_Precision": precision_score(y_test, y_pred_test),
            "Test_Recall": recall_score(y_test, y_pred_test),
            "Test_F1": f1_score(y_test, y_pred_test),
            "Test_AUC_ROC": roc_auc_score(y_test, y_proba_test),
            
            "Train_Confusion_Matrix": confusion_matrix(y_train, y_pred_train).tolist(),
            "Val_Confusion_Matrix": confusion_matrix(y_val, y_pred_val).tolist(),
            "Test_Confusion_Matrix": confusion_matrix(y_test, y_pred_test).tolist(),
        }
        
        logger.info("\nMétricas Detalhadas:")
        for name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{name}: {value:.3f}")
        
        # Verificação de overfitting
        train_precision = metrics["Train_Precision"]
        val_precision = metrics["Val_Precision"]
        overfitting_gap = train_precision - val_precision
        
        max_overfitting = self.config["validation"]["max_overfitting_threshold"]
        if overfitting_gap > max_overfitting:
            logger.warning(f"⚠️  POSSÍVEL OVERFITTING DETECTADO!")
            logger.warning(f"   Diferença treino-validação: {overfitting_gap:.3f} > {max_overfitting:.3f}")
        else:
            logger.info(f"✓ Diferença treino-validação: {overfitting_gap:.3f} (OK)")
        
        # Precisão por nível de confiança
        logger.info("\nPrecisão por Nível de Confiança (Teste):")
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for t in thresholds:
            mask = y_proba_test >= t
            if mask.sum() > 0:
                precision_t = precision_score(y_test[mask], y_pred_test[mask])
                coverage = mask.sum() / len(y_test)
                logger.info(f"  Conf >= {t:.1f}: {precision_t:.1%} (cobertura: {coverage:.1%}, amostras: {mask.sum()})")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 15 Features:")
        logger.info(self.feature_importance.head(15).to_string(index=False))
        
        logger.info("\nBottom 10 Features:")
        logger.info(self.feature_importance.tail(10).to_string(index=False))
        
        self.metrics = metrics
    
    def save_artifacts(self, model, feature_names):
        """Salva todos os artefatos do modelo."""
        logger.info("Salvando artefatos do modelo...")

        # 1. Salva modelo
        model.save_model(str(self.model_path))
        logger.info(f"Modelo salvo: {self.model_path}")

        # 2. Salva metadados com threshold_used
        threshold = self.config["model"]["min_return_threshold"]
        metadata = {
            "created_at": datetime.now().isoformat(),
            "model_type": "XGBoost",
            "feature_names": list(feature_names),
            "feature_count": len(feature_names),
            "threshold_used": threshold,
            "config": self.config,
            "metrics": self.metrics,
            "feature_importance": self.feature_importance.to_dict('records'),
            "training_info": {
                "n_features": len(feature_names),
                "training_log": self.training_log.getvalue()[:5000],  # Primeiros 5000 chars
            },
            "paths": {
                "model_path": str(self.model_path),
                "metadata_path": str(self.metadata_path),
                "scaler_path": str(self.scaler_path) if hasattr(self, 'scaler_path') else None
            }
        }

        with open(self.metadata_path, 'w', encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadados salvos: {self.metadata_path}")

        # 3. Cria/atualiza arquivos "latest" para modelo e metadados
        latest_model = self.models_dir / "xgb_model_latest.json"
        latest_meta_fixed = self.models_dir / "model_metadata.json"         # usado pelo MLInferenceEngine
        latest_meta_alias = self.models_dir / "model_metadata_latest.json"  # alias

        try:
            # Modelo
            if latest_model.exists() or latest_model.is_symlink():
                latest_model.unlink()
            latest_model.symlink_to(self.model_path.name)
            logger.info(f"Symlink do modelo criado: {latest_model} -> {self.model_path.name}")

            # Metadados (nome fixo e alias)
            for target in (latest_meta_fixed, latest_meta_alias):
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(self.metadata_path.name)
                logger.info(f"Symlink dos metadados criado: {target} -> {self.metadata_path.name}")

        except Exception as e:
            logger.warning(f"Falha ao criar symlink, copiando arquivo em vez disso: {e}")
            import shutil
            shutil.copy2(self.model_path, latest_model)
            shutil.copy2(self.metadata_path, latest_meta_fixed)
            shutil.copy2(self.metadata_path, latest_meta_alias)
            logger.info(f"Arquivos copiados para: {latest_model}, {latest_meta_fixed}, {latest_meta_alias}")

        # 4. Salva feature importance separadamente
        importance_path = self.models_dir / f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance salva: {importance_path}")
    
    def run_pipeline(self):
        """Executa pipeline completo de treinamento."""
        logger.info("="*60)
        logger.info("INICIANDO PIPELINE DE TREINAMENTO")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # 1. Carrega dados
            logger.info("\n[1/4] Carregando e validando dados...")
            df = self.load_and_validate_data()
            if df is None:
                return False
            
            # 2. Prepara features
            logger.info("\n[2/4] Preparando features e targets...")
            X, y, features = self.prepare_features_targets(df)
            if X is None:
                return False
            
            # 3. Treina modelo
            logger.info("\n[3/4] Treinando modelo...")
            model = self.train_model(X, y)
            self.model = model
            
            # 4. Salva artefatos
            logger.info("\n[4/4] Salvando artefatos...")
            self.save_artifacts(model, features)
            
            # Estatísticas finais
            elapsed = datetime.now() - start_time
            logger.info("\n" + "="*60)
            logger.info("PIPELINE CONCLUÍDO COM SUCESSO!")
            logger.info(f"Tempo total: {elapsed}")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Erro no pipeline: {e}", exc_info=True)
            
            # Salva log de erro
            error_log_path = self.models_dir / f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(error_log_path, 'w') as f:
                f.write(f"Error at {datetime.now()}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Traceback:\n")
                import traceback
                traceback.print_exc(file=f)
            
            logger.error(f"Log de erro salvo em: {error_log_path}")
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