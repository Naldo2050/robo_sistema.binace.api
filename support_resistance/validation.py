"""
Validação de dados e decorators para o sistema de Suporte/Resistência
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
import inspect
from functools import wraps
from .constants import P, T


# =============================
#  VALIDAÇÃO DE DADOS
# =============================

def validate_dataframe(df: pd.DataFrame, 
                       required_cols: List[str] = None,
                       min_rows: int = 50,
                       allow_nan: bool = False) -> None:
    """
    Valida DataFrame de entrada
    
    Args:
        df: DataFrame a validar
        required_cols: Lista de colunas obrigatórias
        min_rows: Número mínimo de linhas
        allow_nan: Se permite valores NaN
        
    Raises:
        ValueError: Se validação falhar
    """
    if df is None:
        raise ValueError("DataFrame não pode ser None")
    
    if df.empty:
        raise ValueError("DataFrame não pode ser vazio")
    
    required_cols = required_cols or ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando: {missing}")
    
    if not allow_nan:
        for col in required_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                raise ValueError(f"Coluna '{col}' contém {nan_count} valores NaN")
    
    # Verificar valores inválidos para preços
    price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in required_cols]
    for col in price_cols:
        if (df[col] <= 0).any():
            invalid_count = (df[col] <= 0).sum()
            raise ValueError(f"Coluna '{col}' contém {invalid_count} valores <= 0")
    
    # Verificar valores negativos em volume
    if 'volume' in required_cols and (df['volume'] < 0).any():
        raise ValueError("Coluna 'volume' contém valores negativos")
    
    # Só agora checar se há linhas suficientes
    if len(df) < min_rows:
        raise ValueError(f"DataFrame precisa ter pelo menos {min_rows} linhas (tem {len(df)})")


def validate_series(series: pd.Series, 
                   name: str = "series",
                   min_length: int = 20,
                   allow_nan: bool = False,
                   must_be_positive: bool = True) -> None:
    """
    Valida Series de entrada
    
    Args:
        series: Series a validar
        name: Nome para mensagens de erro
        min_length: Comprimento mínimo
        allow_nan: Se permite valores NaN
        must_be_positive: Se valores devem ser positivos
        
    Raises:
        ValueError: Se validação falhar
    """
    if series is None:
        raise ValueError(f"{name} não pode ser None")
    
    # 1) Primeiro checar NaN (para que o erro de NaN apareça antes do erro de tamanho)
    has_nan = series.isna().any()
    if not allow_nan and has_nan:
        nan_count = series.isna().sum()
        raise ValueError(f"{name} contém {nan_count} valores NaN")
    
    if len(series) < min_length:
        raise ValueError(f"DEBUG_FAIL: has_nan={has_nan}, allow_nan={allow_nan}, len={len(series)}, {name} precisa ter pelo menos {min_length}")
    
    # 3) Finalmente, checar se deve ser positivo
    if must_be_positive and (series <= 0).any():
        invalid_count = (series <= 0).sum()
        raise ValueError(f"{name} contém {invalid_count} valores <= 0")


# =============================
#  DECORATORS DE VALIDAÇÃO
# =============================

def validate_positive(*param_names: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator para validar que parâmetros são positivos"""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            for param_name in param_names:
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if value is not None and value <= 0:
                        raise ValueError(f"Parâmetro '{param_name}' deve ser positivo, recebeu: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_range(param_name: str, min_val: float, max_val: float) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator para validar que parâmetro está em um range"""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            if param_name in bound.arguments:
                value = bound.arguments[param_name]
                if value is not None and not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Parâmetro '{param_name}' deve estar entre {min_val} e {max_val}, recebeu: {value}"
                    )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator