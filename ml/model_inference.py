# ml/model_inference.py
"""
Módulo de Inferência em Tempo Real para Modelo Quantitativo.

Atua como uma camada de conveniência que utiliza o MLInferenceEngine.
"""

import logging
from typing import Any, Dict, Optional

from ml.inference_engine import MLInferenceEngine

logger = logging.getLogger("ModelInference")

# Singleton do Engine para evitar recarga de modelo
_ENGINE: Optional[MLInferenceEngine] = None


def get_engine() -> MLInferenceEngine:
    """Retorna instância singleton do MLInferenceEngine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MLInferenceEngine()
    return _ENGINE

def _flatten_dict(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten de dicionário aninhado para formato flat.
    Exportado para uso no MLInferenceEngine.
    """
    flat: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key, sep))
        else:
            flat[key] = v
    return flat

def predict_up_probability(features: Dict[str, Any]) -> Optional[float]:
    """
    Prediz probabilidade de alta (classe 1) para as features fornecidas.
    Utiliza o mapeamento robusto do MLInferenceEngine.
    """
    try:
        engine = get_engine()
        result = engine.predict(features)
        
        if result.get("status") == "ok":
            return result.get("prob_up")
        
        return None

    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        return None
