# ml/model_inference.py
"""
Módulo de Inferência em Tempo Real para Modelo Quantitativo.

Carrega automaticamente o modelo XGBoost mais recente e seus metadados,
recebe dicionário de features, flatten e reordena conforme treino,
retorna probabilidade de alta (classe 1).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger("ModelInference")

# Cache em memória para evitar recarregar o modelo toda hora
_MODEL: Optional[xgb.XGBClassifier] = None
_FEATURE_NAMES: Optional[List[str]] = None
_METADATA: Optional[Dict[str, Any]] = None
_LAST_LOADED_MODEL_PATH: Optional[Path] = None


def _load_latest_model(models_dir: Path = Path("ml/models")) -> Tuple[Optional[xgb.XGBClassifier], Optional[List[str]], Optional[Dict[str, Any]]]:
    """
    Localiza e carrega o modelo e metadados mais recentes.

    Returns:
        Tupla (modelo, feature_names, metadados) ou (None, None, None) se erro.
    """
    if not models_dir.exists():
        logger.warning(f"Diretório de modelos não encontrado: {models_dir}")
        return None, None, None

    # 1. Localizar modelo mais recente
    latest_model_path = models_dir / "xgb_model_latest.json"
    if not latest_model_path.exists():
        # Fallback: encontrar xgb_model_*.json mais recente por mtime
        model_files = list(models_dir.glob("xgb_model_*.json"))
        if not model_files:
            logger.warning("Nenhum arquivo de modelo encontrado.")
            return None, None, None
        latest_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Usando modelo fallback: {latest_model_path.name}")

    # 2. Localizar metadados mais recentes
    metadata_files = list(models_dir.glob("model_metadata_*.json"))
    if not metadata_files:
        logger.warning("Nenhum arquivo de metadados encontrado.")
        return None, None, None
    latest_metadata_path = max(metadata_files, key=lambda p: p.stat().st_mtime)

    try:
        # Carregar metadados
        with open(latest_metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_names = metadata.get("feature_names", [])
        if not feature_names:
            logger.warning("Lista de feature_names vazia nos metadados.")
            return None, None, None

        # Carregar modelo
        model = xgb.XGBClassifier()
        model.load_model(str(latest_model_path))

        # Atualizar cache global
        global _MODEL, _FEATURE_NAMES, _METADATA, _LAST_LOADED_MODEL_PATH
        _MODEL = model
        _FEATURE_NAMES = feature_names
        _METADATA = metadata
        _LAST_LOADED_MODEL_PATH = latest_model_path

        logger.info(f"Modelo carregado: {latest_model_path.name}")
        logger.info(f"Metadados carregados: {latest_metadata_path.name}")
        logger.info(f"Features: {len(feature_names)}")

        return model, feature_names, metadata

    except Exception as e:
        logger.error(f"Erro ao carregar modelo/metadados: {e}")
        return None, None, None


def get_model() -> Tuple[Optional[xgb.XGBClassifier], Optional[List[str]], Optional[Dict[str, Any]]]:
    """
    Retorna modelo, feature_names e metadados do cache, ou carrega se necessário.

    Returns:
        Tupla (modelo, feature_names, metadados).
    """
    if _MODEL is not None:
        return _MODEL, _FEATURE_NAMES, _METADATA
    else:
        return _load_latest_model()


def _flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten de dicionário aninhado para formato flat (igual ao FeatureStore).
    """
    flat: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key, sep))
        else:
            flat[key] = v
    return flat


def _prepare_feature_row(features: Dict[str, Any], feature_names: List[str]) -> Optional[np.ndarray]:
    """
    Prepara linha de features: flatten, reordena conforme feature_names, converte para numpy.

    Args:
        features: Dicionário de features (igual a pipeline.get_final_features()).
        feature_names: Lista de nomes de features do treino.

    Returns:
        Array numpy com features na ordem correta, ou None se erro.
    """
    try:
        # Flatten do dicionário
        flat = _flatten_dict(features)

        # Criar DataFrame com uma linha
        df = pd.DataFrame([flat])

        # Adicionar colunas faltantes com valor 0.0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0

        # Selecionar apenas feature_names na ordem correta
        df = df[feature_names]

        # Converter para numpy array
        return df.astype(float).to_numpy()

    except Exception as e:
        logger.error(f"Erro ao preparar features: {e}")
        return None


def predict_up_probability(features: Dict[str, Any]) -> Optional[float]:
    """
    Prediz probabilidade de alta (classe 1) para as features fornecidas.

    Args:
        features: Dicionário de features finais (igual a pipeline.get_final_features()).

    Returns:
        Probabilidade float entre 0 e 1, ou None se erro/modelo indisponível.
    """
    try:
        # Obter modelo do cache
        model, feature_names, metadata = get_model()

        if model is None or feature_names is None:
            logger.warning("Modelo ou feature_names não disponíveis.")
            return None

        # Preparar linha de features
        X = _prepare_feature_row(features, feature_names)
        if X is None:
            return None

        # Predição
        proba = model.predict_proba(X)[0, 1]
        return float(proba)

    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return None