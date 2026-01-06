"""
Pivot Points Institucionais com múltiplos timeframes e confluência
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from .utils import StatisticalUtils
from .config import PivotConfig


def _get_month_resample_rule() -> str:
    """Retorna a regra de resample para mês conforme versão do pandas"""
    pandas_version = tuple(int(x) for x in pd.__version__.split('.')[:2])
    return 'ME' if pandas_version >= (2, 0) else 'M'


class InstitutionalPivotPoints:
    """Pivot Points com métricas avançadas e validação estatística"""
    
    @staticmethod
    def calculate_enhanced_pivot_points(high: float, low: float, close: float, 
                                       volume: float = None, 
                                       prev_high: float = None,
                                       prev_low: float = None,
                                       methods: List[str] = None) -> Dict:
        """
        Calcula pivot points com métricas de qualidade e bandas de confiança
        
        Args:
            high: Máxima do período
            low: Mínima do período
            close: Fechamento do período
            volume: Volume (opcional)
            prev_high: Máxima do período anterior (opcional)
            prev_low: Mínima do período anterior (opcional)
            methods: Lista de métodos a calcular (classic, camarilla, woodie, fibonacci)
            
        Returns:
            Dict com pivot points de cada método
        """
        if methods is None:
            methods = ["classic", "camarilla", "woodie", "fibonacci"]
        
        range_size = high - low
        result = {}
        
        # Classic Pivot Points
        if "classic" in methods:
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + range_size
            s2 = pivot - range_size
            r3 = pivot + 2 * range_size
            s3 = pivot - 2 * range_size
            
            pivot_quality = {
                "range_percent": StatisticalUtils.safe_divide(range_size, close, 0.0) * 100,
                "close_to_pivot": StatisticalUtils.safe_divide(abs(close - pivot), pivot, 0.0) * 100,
                "is_balanced": StatisticalUtils.safe_divide(abs(high - pivot), abs(low - pivot), 1.0) if pivot != low else 1.0
            }
            
            result["classic"] = {
                "pivot": float(pivot), 
                "r1": float(r1), "s1": float(s1), 
                "r2": float(r2), "s2": float(s2),
                "r3": float(r3), "s3": float(s3),
                "quality": pivot_quality
            }
        
        # Camarilla Pivot Points (fórmula correta)
        if "camarilla" in methods:
            r1 = close + range_size * 1.1 / 12
            s1 = close - range_size * 1.1 / 12
            r2 = close + range_size * 1.1 / 6
            s2 = close - range_size * 1.1 / 6
            r3 = close + range_size * 1.1 / 4
            s3 = close - range_size * 1.1 / 4
            r4 = close + range_size * 1.1 / 2
            s4 = close - range_size * 1.1 / 2
            
            result["camarilla"] = {
                "r1": float(r1), "s1": float(s1),
                "r2": float(r2), "s2": float(s2),
                "r3": float(r3), "s3": float(s3), 
                "r4": float(r4), "s4": float(s4)
            }
        
        # Woodie Pivot Points
        if "woodie" in methods:
            pivot_woodie = (high + low + 2 * close) / 4
            r1_woodie = 2 * pivot_woodie - low
            s1_woodie = 2 * pivot_woodie - high
            r2_woodie = pivot_woodie + range_size
            s2_woodie = pivot_woodie - range_size
            
            result["woodie"] = {
                "pivot": float(pivot_woodie), 
                "r1": float(r1_woodie), "s1": float(s1_woodie),
                "r2": float(r2_woodie), "s2": float(s2_woodie)
            }
        
        # Fibonacci Pivot Points
        if "fibonacci" in methods:
            pivot_fib = (high + low + close) / 3
            r1_fib = pivot_fib + 0.382 * range_size
            s1_fib = pivot_fib - 0.382 * range_size
            r2_fib = pivot_fib + 0.618 * range_size
            s2_fib = pivot_fib - 0.618 * range_size
            r3_fib = pivot_fib + 1.0 * range_size
            s3_fib = pivot_fib - 1.0 * range_size
            
            result["fibonacci"] = {
                "pivot": float(pivot_fib),
                "r1": float(r1_fib), "s1": float(s1_fib),
                "r2": float(r2_fib), "s2": float(s2_fib),
                "r3": float(r3_fib), "s3": float(s3_fib)
            }
        
        # Metadata
        if prev_high and prev_low:
            trend_strength = StatisticalUtils.safe_divide(
                close - (prev_high + prev_low) / 2, 
                range_size, 
                0.0
            )
        else:
            pivot_ref = (high + low + close) / 3
            trend_strength = StatisticalUtils.safe_divide(close - pivot_ref, range_size, 0.0)
        
        result["metadata"] = {
            "range": float(range_size),
            "trend_strength": float(trend_strength),
            "centrality": StatisticalUtils.safe_divide(
                abs(close - (high + low + close) / 3), 
                range_size, 
                0.0
            )
        }
        
        return result
    
    @staticmethod
    def calculate_multi_timeframe_pivots(df: pd.DataFrame, 
                                        logger: Optional[logging.Logger] = None,
                                        config: Optional[PivotConfig] = None) -> Dict:
        """
        Calcula pivot points para múltiplos timeframes e analisa confluência
        
        Args:
            df: DataFrame com OHLCV
            logger: Logger para mensagens (opcional)
            config: Configuração de pivot points (opcional)
            
        Returns:
            Dict com pivots por timeframe e análise de confluência
        """
        if config is None:
            config = PivotConfig()
        
        if df.empty or len(df) < 20:
            return {}
        
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except pd.errors.OutOfBoundsDatetime as e:
                if logger:
                    logger.warning(f"Data fora do range: {e}")
                return {}
            except Exception as e:
                if logger:
                    logger.warning(f"Não foi possível converter índice para datetime: {e}")
                return {}
        
        pivots = {}
        
        # Detectar regra de resample para mês automaticamente
        month_rule = _get_month_resample_rule()
        
        timeframes = {
            'daily': ('D', 1),
            'weekly': ('W', 5),
            'monthly': (month_rule, 20)
        }
        
        for tf_name, (resample_period, min_bars) in timeframes.items():
            try:
                resampled = df.resample(resample_period).agg({
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                if len(resampled) >= 2:
                    last_period = resampled.iloc[-2]
                    prev_period = resampled.iloc[-3] if len(resampled) >= 3 else None
                    
                    high = float(last_period['high'])
                    low = float(last_period['low'])
                    close = float(last_period['close'])
                    volume = float(last_period['volume'])
                    prev_high = float(prev_period['high']) if prev_period is not None else None
                    prev_low = float(prev_period['low']) if prev_period is not None else None
                    
                    pivots[tf_name] = InstitutionalPivotPoints.calculate_enhanced_pivot_points(
                        high, low, close, volume, prev_high, prev_low,
                        methods=config.methods
                    )
            except KeyError as e:
                if logger:
                    logger.warning(f"Coluna faltando para {tf_name}: {e}")
                continue
            except ValueError as e:
                if logger:
                    logger.warning(f"Valor inválido para {tf_name}: {e}")
                continue
            except Exception as e:
                if logger:
                    logger.error(f"Erro inesperado em {tf_name}: {e}", exc_info=True)
                continue
        
        # Análise de confluência
        confluence_analysis = InstitutionalPivotPoints._analyze_pivot_confluence(
            pivots, 
            config.confluence_tolerance_percent
        )
        pivots['confluence'] = confluence_analysis
        
        return pivots
    
    @staticmethod
    def _analyze_pivot_confluence(pivots: Dict, tolerance_percent: float = 0.5) -> Dict:
        """
        Analisa confluência entre pivot points de diferentes timeframes
        
        Args:
            pivots: Dict com pivots por timeframe
            tolerance_percent: Tolerância para considerar níveis confluentes
            
        Returns:
            Dict com clusters de confluência
        """
        if not pivots:
            return {}
        
        all_levels = []
        level_weights = []
        
        weight_map = {'daily': 1.0, 'weekly': 1.5, 'monthly': 2.0}
        
        for tf_name, tf_pivots in pivots.items():
            if tf_name in ['confluence', 'metadata']:
                continue
            
            weight = weight_map.get(tf_name, 1.0)
            
            # Coletar níveis de todos os métodos
            for method in ['classic', 'woodie', 'fibonacci']:
                if method in tf_pivots:
                    method_data = tf_pivots[method]
                    for level_name in ['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']:
                        level_value = method_data.get(level_name)
                        if level_value is not None:
                            all_levels.append(level_value)
                            level_weights.append(weight)
            
            # Níveis Camarilla
            if 'camarilla' in tf_pivots:
                cam_data = tf_pivots['camarilla']
                for level_name in ['r1', 'r2', 'r3', 'r4', 's1', 's2', 's3', 's4']:
                    level_value = cam_data.get(level_name)
                    if level_value is not None:
                        all_levels.append(level_value)
                        level_weights.append(weight)
        
        if not all_levels:
            return {}
        
        tolerance = (tolerance_percent / 100) * np.mean(all_levels)
        clusters = []
        
        sorted_indices = np.argsort(all_levels)
        sorted_levels = np.array(all_levels)[sorted_indices]
        sorted_weights = np.array(level_weights)[sorted_indices]
        
        i = 0
        while i < len(sorted_levels):
            current_level = sorted_levels[i]
            current_weight = sorted_weights[i]
            cluster_members = [current_level]
            cluster_weights = [current_weight]
            
            j = i + 1
            while j < len(sorted_levels) and abs(sorted_levels[j] - current_level) <= tolerance:
                cluster_members.append(sorted_levels[j])
                cluster_weights.append(sorted_weights[j])
                j += 1
            
            weighted_level = np.average(cluster_members, weights=cluster_weights)
            cluster_score = len(cluster_members) * np.mean(cluster_weights)
            
            clusters.append({
                'price': float(weighted_level),
                'score': float(StatisticalUtils.clamp(cluster_score, 0, 10)),
                'members': len(cluster_members),
                'weighted_members': [float(w) for w in cluster_weights]
            })
            
            i = j
        
        clusters.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'clusters': clusters[:5],
            'strongest_cluster': clusters[0] if clusters else None,
            'total_clusters': len(clusters)
        }