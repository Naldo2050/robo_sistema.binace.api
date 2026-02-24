"""
Utilitários do sistema de Suporte/Resistência
"""

import time
import logging
import json
import threading
from contextlib import contextmanager
from functools import wraps
from collections import OrderedDict, deque
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime, date
from decimal import Decimal
import hashlib
import numpy as np
from scipy import stats
import pandas as pd

from .constants import (
    CONSTANTS, ConfidenceIntervalResult, ClusterQualityResult, 
    QualityRating, P, T
)
from .config import SafeJSONEncoder


# =============================
#  UTILITÁRIOS DE PERFORMANCE
# =============================

@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None, 
          metrics_dict: Optional[Dict] = None):
    """
    Context manager para medir tempo de execução
    
    Args:
        name: Nome da operação
        logger: Logger para output (opcional)
        metrics_dict: Dicionário para armazenar métricas (opcional)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    
    if logger:
        logger.debug(f"{name}: {elapsed:.4f}s")
    
    if metrics_dict is not None:
        metrics_dict[f"{name}_time"] = elapsed


# =============================
#  LOGGING ESTRUTURADO
# =============================

class StructuredLogger:
    """Logger com suporte a dados estruturados"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._context: Dict[str, Any] = {}
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager para adicionar contexto temporário ao log"""
        old_context = self._context.copy()
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = old_context
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Formata mensagem com contexto"""
        data = {**self._context, **kwargs}
        if data:
            return f"{message} | {json.dumps(data, default=str)}"
        return message
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        self.logger.error(self._format_message(message, **kwargs), exc_info=exc_info)
    
    def performance(self, operation: str, duration_ms: float, **kwargs):
        """Log específico para métricas de performance"""
        self.info(f"PERF:{operation}", duration_ms=duration_ms, **kwargs)


# =============================
#  UTILITÁRIOS ESTATÍSTICOS
# =============================

class StatisticalUtils:
    """Utilitários estatísticos para análise quantitativa"""
    
    # Cache com thread safety e limite LRU
    _cache: OrderedDict = OrderedDict()
    _cache_enabled: bool = True
    _cache_lock: threading.RLock = threading.RLock()
    _cache_max_size: int = 1000
    
    @classmethod
    def enable_cache(cls, enabled: bool = True, max_size: int = 1000) -> None:
        """Habilita ou desabilita cache com limite de tamanho"""
        with cls._cache_lock:
            cls._cache_enabled = enabled
            cls._cache_max_size = max_size
    
    @classmethod
    def clear_cache(cls) -> None:
        """Limpa o cache de forma thread-safe"""
        with cls._cache_lock:
            cls._cache.clear()
    
    @classmethod
    def _get_from_cache(cls, key: tuple) -> Optional[Any]:
        """Recupera valor do cache de forma thread-safe"""
        with cls._cache_lock:
            return cls._cache.get(key)
    
    @classmethod
    def _set_in_cache(cls, key: tuple, value: Any) -> None:
        """Armazena valor no cache com eviction LRU"""
        with cls._cache_lock:
            # Eviction se exceder tamanho máximo
            while len(cls._cache) >= cls._cache_max_size:
                cls._cache.popitem(last=False)  # Remove o mais antigo
            cls._cache[key] = value
    
    @staticmethod
    def _hash_array(arr: np.ndarray) -> str:
        """Gera hash único para array"""
        return hashlib.md5(arr.tobytes()).hexdigest()
    
    @staticmethod
    def clamp(value: Union[float, np.floating], min_val: Union[float, np.floating] = 0.0, max_val: Union[float, np.floating] = 10.0) -> float:
        """Limita valor entre min e max"""
        if np.isnan(value) or np.isinf(value):
            return float(min_val)
        return float(np.clip(value, min_val, max_val))
    
    @staticmethod
    def safe_divide(numerator: Union[float, np.floating, np.ndarray], denominator: Union[float, np.floating, np.ndarray], default: float = 0.0) -> float:
        """Divisão segura que evita divisão por zero"""
        if isinstance(denominator, np.ndarray):
            if denominator.size == 1:
                denominator = float(denominator.item())
            else:
                # Para arrays, retorna default se qualquer elemento for zero
                if np.any(denominator == 0) or np.any(np.isnan(denominator)) or np.any(np.isinf(denominator)):
                    return default
        elif denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default
        
        result = numerator / denominator
        if isinstance(result, np.ndarray):
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                return default
            result = float(result.item()) if result.size == 1 else float(np.mean(result))
        elif np.isnan(result) or np.isinf(result):
            return default
        return float(result)
    
    @staticmethod
    def normalize_score(value: float, min_val: float, max_val: float, 
                       target_min: float = 0.0, target_max: float = 10.0) -> float:
        """Normaliza score para escala alvo"""
        if max_val == min_val:
            return (target_min + target_max) / 2
        
        normalized = (value - min_val) / (max_val - min_val)
        return StatisticalUtils.clamp(target_min + normalized * (target_max - target_min), target_min, target_max)
    
    @staticmethod
    def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> ConfidenceIntervalResult:
        """
        Calcula intervalo de confiança e métricas de qualidade
        
        Args:
            data: Array de dados
            confidence: Nível de confiança (default 0.95 = 95%)
            
        Returns:
            ConfidenceIntervalResult com todas as métricas
        """
        if len(data) == 0:
            return {
                "mean": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "std": 0.0,
                "ci_width": 0.0,
                "ci_width_pct": 0.0,
                "stability_score": 10.0,
                "sample_size": 0
            }
         
        if len(data) == 1:
            value = float(data[0])
            return {
                "mean": value,
                "ci_lower": value,
                "ci_upper": value,
                "std": 0.0,
                "ci_width": 0.0,
                "ci_width_pct": 0.0,
                "stability_score": 10.0,
                "sample_size": 1
            }
         
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))  # ddof=1 para amostra

        # Se não há variância, CI degenera no ponto
        if np.isclose(std, 0.0):
            return {
                "mean": mean,
                "ci_lower": mean,
                "ci_upper": mean,
                "std": 0.0,
                "ci_width": 0.0,
                "ci_width_pct": 0.0,
                "stability_score": 10.0,
                "sample_size": len(data)
            }

        sem = stats.sem(data)
        if sem is None or np.isnan(sem) or np.isclose(sem, 0.0):
            ci_lower, ci_upper = mean, mean
        else:
            ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
            ci_lower, ci_upper = float(ci[0]), float(ci[1])

        # Calcular largura do CI e percentual
        ci_width = ci_upper - ci_lower
        ci_width_pct = StatisticalUtils.safe_divide(ci_width, abs(mean), 0.0) * 100
        
        # stability_score (0-10): penaliza clusters muito dispersos
        # Se a largura do CI for > 5% do preço, score cai drasticamente
        stability_score = StatisticalUtils.clamp(10.0 - (ci_width_pct * 2.0), 0, 10)
        
        return {
            "mean": mean,
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "std": std,
            "ci_width": float(ci_width),
            "ci_width_pct": float(ci_width_pct),
            "stability_score": float(stability_score),
            "sample_size": len(data)
        }
    
    @staticmethod
    def calculate_zscore(value: float, data: np.ndarray) -> float:
        """Calcula Z-score para um valor em relação a uma distribuição"""
        if len(data) < 2:
            return 0.0
        std = np.std(data)
        if std == 0 or np.isnan(std):
            return 0.0
        return float((value - np.mean(data)) / std)
    
    @classmethod
    def bootstrap_ci(cls, data: np.ndarray, statistic: str = "median", 
                    n: int = 1000, confidence: float = 0.95,
                    use_cache: bool = True) -> Dict:
        """
        Calcula intervalo de confiança bootstrap para estatística robusta
        
        Args:
            data: Array de dados
            statistic: "median" ou "mean"
            n: Número de amostras bootstrap
            confidence: Nível de confiança (0.95 = 95%)
            use_cache: Se deve usar cache
            
        Returns:
            Dict com ci_lower, ci_upper, ci_width, ci_width_pct
        """
        if len(data) < CONSTANTS.MIN_BOOTSTRAP_SAMPLES:
            return {
                "ci_lower": 0.0, 
                "ci_upper": 0.0, 
                "ci_width": 0.0, 
                "ci_width_pct": 0.0,
                "statistic_value": float(data[0]) if len(data) == 1 else 0.0
            }
        
        # Verificar cache
        if use_cache and cls._cache_enabled:
            cache_key = (cls._hash_array(data), statistic, n, confidence)
            cached = cls._get_from_cache(cache_key)
            if cached is not None:
                return cached
        
        if statistic == "median":
            stat_func = np.median
        else:
            stat_func = np.mean
        
        # Gerar amostras bootstrap
        rng = np.random.default_rng(42)  # reprodutível
        bootstrap_samples = rng.choice(data, size=(n, len(data)), replace=True)
        
        # Calcular estatística para cada amostra
        bootstrap_stats = np.apply_along_axis(lambda x: float(stat_func(x)), 1, bootstrap_samples)
        
        # Calcular percentis
        lower_percentile = (1 - confidence) / 2
        upper_percentile = 1 - lower_percentile
        
        ci_lower = float(np.percentile(bootstrap_stats, lower_percentile * 100, out=None))
        ci_upper = float(np.percentile(bootstrap_stats, upper_percentile * 100, out=None))
        
        stat_value = float(stat_func(data))
        ci_width = ci_upper - ci_lower
        
        result = {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": float(ci_width),
            "ci_width_pct": float(cls.safe_divide(ci_width, abs(stat_value), 0.0) * 100),
            "statistic_value": stat_value
        }
        
        # Salvar no cache
        if use_cache and cls._cache_enabled:
            cls._set_in_cache(cache_key, result)
        
        return result
    
    @staticmethod
    def cluster_prices(prices: np.ndarray, eps_percent: float = 0.005,
                      min_cluster_size: int = 3) -> List[np.ndarray]:
        """
        Agrupa preços usando clustering hierárquico para séries temporais financeiras
        
        Args:
            prices: Array de preços
            eps_percent: Tolerância como percentual do range
            min_cluster_size: Tamanho mínimo do cluster
            
        Returns:
            Lista de arrays (clusters), ordenados por tamanho decrescente
        """
        if len(prices) == 0:
            return []
        
        if len(prices) < min_cluster_size:
            return []
        
        price_range = np.max(prices) - np.min(prices)
        if price_range == 0:
            # Todos os preços são iguais
            return [prices] if len(prices) >= min_cluster_size else []
        
        eps = price_range * eps_percent
        
        clusters = []
        sorted_prices = np.sort(prices)
        
        current_cluster = [sorted_prices[0]]
        for price in sorted_prices[1:]:
            if price - current_cluster[-1] <= eps:
                current_cluster.append(price)
            else:
                if len(current_cluster) >= min_cluster_size:
                    clusters.append(np.array(current_cluster))
                current_cluster = [price]
        
        if len(current_cluster) >= min_cluster_size:
            clusters.append(np.array(current_cluster))
        
        clusters.sort(key=lambda x: len(x), reverse=True)
        return clusters
    
    @staticmethod
    def calculate_cluster_quality_score(cluster: np.ndarray, 
                                       full_price_range: float) -> ClusterQualityResult:
        """
        Avalia qualidade do cluster
        
        Args:
            cluster: Array de preços no cluster
            full_price_range: Range total de preços
            
        Returns:
            ClusterQualityResult com score e métricas componentes
        """
        if len(cluster) < 3:
            return {
                "score": 0.0, 
                "quality": QualityRating.INSUFFICIENT_DATA.value,
                "dispersion_score": 0.0,
                "concentration_score": 0.0,
                "outlier_score": 0.0,
                "outlier_count": 0
            }
        
        cluster_range = np.max(cluster) - np.min(cluster)
        
        if full_price_range == 0:
            relative_dispersion = 0.0
        else:
            relative_dispersion = cluster_range / full_price_range
        
        # Menor dispersão = melhor cluster
        dispersion_score = 10 * (1 - min(relative_dispersion * 10, 1))
        
        # Verificar concentração usando coeficiente de variação
        mean_val = np.mean(cluster)
        if mean_val == 0:
            cv = 0
        else:
            cv = np.std(cluster) / abs(mean_val)
        concentration_score = 10 * (1 - min(cv * 5, 1))
        
        # Verificar outliers usando IQR
        q1, q3 = np.percentile(cluster, [25, 75], out=None)
        iqr = q3 - q1
        
        if iqr == 0:
            outliers = 0
        else:
            outliers = np.sum((cluster < q1 - CONSTANTS.OUTLIER_IQR_MULTIPLIER*iqr) | (cluster > q3 + CONSTANTS.OUTLIER_IQR_MULTIPLIER*iqr))
        
        outlier_ratio = outliers / len(cluster)
        outlier_score = 10 * (1 - outlier_ratio)
        
        final_score = (dispersion_score * 0.4 + concentration_score * 0.4 + outlier_score * 0.2)
        
        if final_score >= CONSTANTS.QUALITY_EXCELLENT:
            quality = QualityRating.EXCELLENT.value
        elif final_score >= CONSTANTS.QUALITY_GOOD:
            quality = QualityRating.GOOD.value
        elif final_score >= CONSTANTS.QUALITY_MODERATE:
            quality = QualityRating.MODERATE.value
        else:
            quality = QualityRating.POOR.value
        
        return {
            "score": StatisticalUtils.clamp(final_score, 0, 10),
            "quality": quality,
            "dispersion_score": float(dispersion_score),
            "concentration_score": float(concentration_score),
            "outlier_score": float(outlier_score),
            "outlier_count": int(outliers)
        }
    
    @staticmethod
    def calculate_relative_strength(level: float, prices: np.ndarray, 
                                   volumes: Optional[np.ndarray] = None,
                                   tolerance_pct: float = 0.005) -> Dict:
        """
        Calcula força relativa do nível (RSL)
        
        Args:
            level: Preço do nível
            prices: Array de preços
            volumes: Array de volumes (opcional)
            tolerance_pct: Tolerância como percentual do nível
            
        Returns:
            Dict com score e métricas componentes
        """
        tolerance = level * tolerance_pct
        touches = np.abs(prices - level) <= tolerance
        touch_indices = np.where(touches)[0]
        
        if len(touch_indices) == 0:
            return {
                "score": 0.0,
                "rejection_rate": 0.0,
                "volume_ratio": 1.0,
                "recency": 0.0,
                "touch_count": 0
            }
        
        # Rejeição = preço toca e reverte significativamente
        rejections = 0
        rejection_volumes = []
        
        for idx in touch_indices:
            if idx >= len(prices) - CONSTANTS.DEFAULT_FUTURE_WINDOW:
                continue
            
            future = prices[idx:idx + CONSTANTS.DEFAULT_FUTURE_WINDOW]
            if len(future) < 2:
                continue
                
            reversal = abs(future[-1] - level) / level if level != 0 else 0
            
            if reversal > CONSTANTS.MIN_REVERSAL_PERCENT:
                rejections += 1
                if volumes is not None and idx < len(volumes):
                    rejection_volumes.append(volumes[idx])
        
        # Score baseado em rejeições
        rejection_rate = StatisticalUtils.safe_divide(rejections, len(touch_indices), 0.0)
        
        # Volume nas rejeições vs média
        if rejection_volumes and volumes is not None and len(volumes) > 0:
            avg_vol = np.mean(volumes)
            if avg_vol > 0:
                vol_ratio = np.mean(rejection_volumes) / avg_vol
            else:
                vol_ratio = 1.0
        else:
            vol_ratio = 1.0
        
        # Recência do último toque
        recency = 1 - StatisticalUtils.safe_divide(
            len(prices) - touch_indices[-1], 
            len(prices), 
            0.0
        )
        
        # Score final
        rsl = (rejection_rate * 0.4 + min(vol_ratio, 2) / 2 * 0.3 + recency * 0.3) * 10
        
        return {
            "score": StatisticalUtils.clamp(rsl, 0, 10),
            "rejection_rate": float(rejection_rate),
            "volume_ratio": float(vol_ratio),
            "recency": float(recency),
            "touch_count": len(touch_indices),
            "rejection_count": rejections
        }