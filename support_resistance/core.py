"""
Núcleo do sistema de detecção avançada de suporte e resistência
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from .constants import (
    LevelType, QualityRating, CONSTANTS, 
    SRLevelResult, SRAnalysisResult
)
from .config import SRConfig
from .utils import StatisticalUtils
from .validation import validate_positive, validate_range


class AdvancedSupportResistance:
    """Detecção avançada de suporte e resistência com métricas institucionais"""
    
    def __init__(self, price_series: pd.Series, volume_series: pd.Series = None,
                 config: Optional[SRConfig] = None):
        """
        Inicializa o detector de S/R
        
        Args:
            price_series: Series de preços (close)
            volume_series: Series de volumes (opcional)
            config: Configuração do detector
        """
        if config is None:
            config = SRConfig()
        
        self.config = config
        self.price_series = price_series.dropna()
        
        # Alinhar índices
        if volume_series is not None:
            self.volume_series = volume_series.dropna()
            common_idx = self.price_series.index.intersection(self.volume_series.index)
            self.price_series = self.price_series.loc[common_idx]
            self.volume_series = self.volume_series.loc[common_idx]
        else:
            self.volume_series = None
        
        self.lookback_period = min(config.lookback_period, len(self.price_series))
        self.utils = StatisticalUtils()
    
    @validate_positive('num_levels')
    @validate_range('num_levels', 1, 50)
    def detect_with_metrics(self, num_levels: int = 5) -> SRAnalysisResult:
        """
        Detecta níveis de suporte e resistência com métricas quantitativas
        
        Args:
            num_levels: Número máximo de níveis a retornar
            
        Returns:
            SRAnalysisResult com support_levels, resistance_levels, defense_zones, quality_report
        """
        if len(self.price_series) < 20:
            return self._empty_result()
        
        recent_prices = self.price_series.iloc[-self.lookback_period:].values
        recent_volumes = (self.volume_series.iloc[-self.lookback_period:].values 
                         if self.volume_series is not None else None)
        
        current_price = recent_prices[-1]
        
        # Encontrar extremos locais
        local_minima, local_maxima = self._find_local_extrema(recent_prices)
        
        # Clusterizar
        support_clusters = self.utils.cluster_prices(
            local_minima, 
            self.config.cluster_eps_percent,
            self.config.min_cluster_size
        )
        resistance_clusters = self.utils.cluster_prices(
            local_maxima, 
            self.config.cluster_eps_percent,
            self.config.min_cluster_size
        )
        
        # Calcular níveis com métricas
        support_levels = self._calculate_cluster_levels(
            support_clusters, recent_prices, recent_volumes, LevelType.SUPPORT.value
        )
        resistance_levels = self._calculate_cluster_levels(
            resistance_clusters, recent_prices, recent_volumes, LevelType.RESISTANCE.value
        )
        
        # Filtrar por lado do preço atual
        support_levels = self._filter_levels_by_side(support_levels, current_price, LevelType.SUPPORT.value)
        resistance_levels = self._filter_levels_by_side(resistance_levels, current_price, LevelType.RESISTANCE.value)
        
        # Rankear e filtrar
        support_levels = self._filter_and_rank_levels(
            support_levels, current_price, num_levels, ascending=False
        )
        resistance_levels = self._filter_and_rank_levels(
            resistance_levels, current_price, num_levels, ascending=True
        )

        # Zonas de defesa
        defense_zones = self._calculate_defense_zones(support_levels, resistance_levels, current_price)
        
        # Relatório de qualidade
        quality_report = self._generate_quality_report(support_levels, resistance_levels, recent_prices)
        
        return {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "defense_zones": defense_zones,
            "quality_report": quality_report,
            "current_price": float(current_price),
            "lookback_period": self.lookback_period,
            "timestamp": datetime.now().isoformat()
        }
    
    def _filter_levels_by_side(self, levels: List[Dict], current_price: float, level_type: str) -> List[Dict]:
        """Filtra níveis pelo lado correto do preço atual"""
        if not levels or current_price <= 0:
            return levels

        tol_abs = current_price * 0.001  # 0.1%
        filtered: List[Dict] = []

        for level in levels:
            p = float(level.get("price", 0.0))

            # Aceita níveis no preço atual
            if abs(p - current_price) <= tol_abs:
                filtered.append(level)
                continue

            if level_type == LevelType.SUPPORT.value:
                if p < (current_price - tol_abs):
                    filtered.append(level)
            else:  # resistance
                if p > (current_price + tol_abs):
                    filtered.append(level)

        return filtered
    
    def _find_local_extrema(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encontra mínimos e máximos locais usando scipy.signal.find_peaks
        
        Args:
            prices: Array de preços
            
        Returns:
            Tuple (minima, maxima) como arrays de preços
        """
        if len(prices) < 10:
            return np.array([]), np.array([])
        
        # Suavização usando EMA
        smoothed = pd.Series(prices).ewm(span=CONSTANTS.DEFAULT_SMOOTHING_SPAN).mean().values
        
        # Calcular volatilidade para prominence
        vol = np.median(np.abs(np.diff(smoothed)))
        current_price = prices[-1]
        prominence = max(vol * self.config.prom_k, current_price * self.config.min_prom_pct)
        
        # Encontrar máximos (resistências)
        peaks, _ = find_peaks(smoothed, prominence=prominence, distance=CONSTANTS.DEFAULT_PEAK_DISTANCE)
        maxima = np.array([prices[i] for i in peaks])
        
        # Encontrar mínimos (suportes)
        troughs, _ = find_peaks(-smoothed, prominence=prominence, distance=CONSTANTS.DEFAULT_PEAK_DISTANCE)
        minima = np.array([prices[i] for i in troughs])
        
        return minima, maxima
    
    def _calculate_cluster_levels(self, clusters: List[np.ndarray],
                                  prices: np.ndarray,
                                  volumes: Optional[np.ndarray],
                                  level_type: str) -> List[Dict[str, Any]]:
        """Calcula níveis representativos a partir de clusters"""
        levels: List[Dict[str, Any]] = []

        if len(prices) < 3:
            return levels

        # Volatilidade robusta
        vol = np.median(np.abs(np.diff(prices)))
        if vol == 0:
            vol = 0.01 * np.mean(prices)

        current_price = prices[-1]
        price_range = np.max(prices) - np.min(prices)

        # Primeiro passo: coletar métricas brutas
        raw_levels = []
        for cluster in clusters:
            if len(cluster) < self.config.min_cluster_size:
                continue

            cluster_price = np.median(cluster)
            cluster_mean = np.mean(cluster)
            cluster_std = np.std(cluster)
            cluster_size = len(cluster)

            tolerance = max(
                cluster_std, 
                self.config.tol_k * vol, 
                current_price * self.config.min_tol_pct
            )
            touches = int(np.sum(np.abs(prices - cluster_mean) <= tolerance))

            # Volume bruto
            volume_strength_raw = 0.0
            if volumes is not None and len(volumes) == len(prices):
                nearby_mask = np.abs(prices - cluster_mean) <= tolerance
                avg_vol = np.mean(volumes)
                if avg_vol > 0:
                    volume_strength_raw = float(np.sum(volumes[nearby_mask]) / avg_vol)
            
            raw_levels.append({
                "price": float(cluster_price),
                "mean": float(cluster_mean),
                "std": float(cluster_std),
                "touches": touches,
                "cluster_size": cluster_size,
                "volume_strength_raw": volume_strength_raw,
                "cluster_data": cluster,
                "tolerance": float(tolerance),
                "vol_proxy": float(vol)
            })

        if not raw_levels:
            return []

        # Cap de volume winsorizado
        all_raw_strengths = [l["volume_strength_raw"] for l in raw_levels]
        vol_cap = np.percentile(all_raw_strengths, self.config.volume_cap_percentile * 100) if all_raw_strengths else 1.0
        
        # Segundo passo: calcular scores finais
        for l in raw_levels:
            cluster_mean = l["mean"]
            cluster = l["cluster_data"]
            tolerance = l["tolerance"]
            touches = l["touches"]
            cluster_size = l["cluster_size"]
            volume_strength_raw = l["volume_strength_raw"]

            # Aplicar cap
            volume_strength = min(volume_strength_raw, vol_cap)

            # Recency
            recency_metadata = self._calculate_recency_score(prices, cluster_mean, tolerance)
            recency_score = recency_metadata["score"]
            
            # Reaction (micro-backtest)
            reaction_metadata = self._calculate_reaction_score_vectorized(
                prices, cluster_mean, tolerance, level_type
            )
            reaction_score = reaction_metadata["score"]

            # Scores individuais (0-10)
            touches_score = StatisticalUtils.clamp(touches / self.config.expected_touches * 10)
            density_score = StatisticalUtils.clamp(cluster_size / self.config.expected_cluster_size * 10)
            volume_score = StatisticalUtils.clamp(np.log1p(volume_strength) * 3.0, 0, 10)
            
            # Bootstrap CI
            boot_ci = self.utils.bootstrap_ci(cluster, statistic="median", n=1000)
            ci = self.utils.calculate_confidence_interval(cluster)
            ci_dict = dict(ci)
            ci_dict["bootstrap"] = boot_ci
            
            stability_score = ci_dict["stability_score"]

            # Qualidade do cluster
            cluster_quality = self.utils.calculate_cluster_quality_score(cluster, price_range)

            # Composite score com pesos configuráveis
            # weights: (touches, density, volume, recency, stability, reaction)
            w = self.config.weights
            composite_score = (
                w[0] * touches_score +
                w[1] * density_score +
                w[2] * volume_score +
                w[3] * recency_score +
                w[4] * stability_score +
                w[5] * reaction_score
            )
            composite_score = StatisticalUtils.clamp(composite_score, 0, 10)

            levels.append({
                "price": l["price"],
                "mean": float(cluster_mean),
                "std": float(l["std"]),
                "touches": touches,
                "cluster_size": cluster_size,
                "volume_strength": float(volume_strength),
                "recency_score": float(recency_score),
                "stability_score": float(stability_score),
                "reaction_score": float(reaction_score),
                "composite_score": float(composite_score),
                "confidence_interval": ci_dict,
                "cluster_quality": cluster_quality,
                "type": level_type,
                "origin": "local_extrema_cluster",
                "audit": {
                    "tolerance_used": float(tolerance),
                    "volatility_proxy": float(l["vol_proxy"]),
                    "last_touch_index": int(recency_metadata["last_touch_index"]),
                    "touch_count_recent": int(recency_metadata["touch_count_recent"]),
                    "touch_window": int(recency_metadata["touch_window"]),
                    "reaction_reversals": reaction_metadata.get("reversals", []),
                    "volume_cap_used": float(vol_cap),
                    "adaptive_threshold": reaction_metadata.get("adaptive_threshold", 0.0)
                },
                "features": {
                    "touches_score": float(touches_score),
                    "density_score": float(density_score),
                    "volume_score": float(volume_score),
                    "recency_score": float(recency_score),
                    "stability_score": float(stability_score),
                    "reaction_score": float(reaction_score)
                }
            })

        return levels
    
    def _calculate_recency_score(self, prices: np.ndarray, level: float, 
                                tolerance: float) -> Dict[str, Any]:
        """Calcula score de recência para um nível"""
        recent_window = min(50, len(prices))
        
        all_touches = np.where(np.abs(prices - level) <= tolerance)[0]
        
        if len(all_touches) == 0:
            return {
                "score": 0.0,
                "last_touch_index": -1,
                "touch_count_recent": 0,
                "touch_window": recent_window
            }
        
        last_touch_idx = int(all_touches[-1])
        position_from_end = len(prices) - 1 - last_touch_idx
        
        recent_touches = int(np.sum(all_touches >= (len(prices) - recent_window)))
        
        max_distance = len(prices) - 1
        recency_score_raw = 1.0 - StatisticalUtils.safe_divide(position_from_end, max_distance, 0.0)
        recency_score = float(StatisticalUtils.clamp(recency_score_raw * 10, 0, 10))
        
        return {
            "score": recency_score,
            "last_touch_index": last_touch_idx,
            "touch_count_recent": recent_touches,
            "touch_window": recent_window
        }

    def _calculate_reaction_score_vectorized(
        self, 
        prices: np.ndarray, 
        level: float, 
        tolerance: float, 
        level_type: str
    ) -> Dict[str, Any]:
        """
        Versão vetorizada do cálculo de reaction score.
        ~5x mais rápido para arrays grandes.
        """
        window = self.config.reaction_window
        touches_mask = np.abs(prices - level) <= tolerance
        touch_indices = np.where(touches_mask)[0]
        
        if len(touch_indices) == 0:
            return {"score": 0.0, "reversals": [], "avg_reversal_pct": 0.0, "adaptive_threshold": 0.0}
        
        # Calcular volatilidade local
        if len(prices) > 20:
            local_volatility = np.percentile(np.abs(np.diff(prices[-20:])), 75) / level * 100
        else:
            local_volatility = 0.5
        
        adaptive_threshold = max(local_volatility, self.config.min_reversal_pct)
        
        # Filtrar índices válidos (com espaço para janela futura)
        valid_mask = touch_indices < len(prices) - 1
        valid_indices = touch_indices[valid_mask]
        
        if len(valid_indices) == 0:
            return {"score": 0.0, "reversals": [], "avg_reversal_pct": 0.0, 
                    "adaptive_threshold": adaptive_threshold}
        
        # Vetorizar cálculo de reversão
        # Criar matriz de preços futuros para todos os toques de uma vez
        max_future = min(window, len(prices) - valid_indices.max() - 1)
        if max_future <= 0:
            return {"score": 0.0, "reversals": [], "avg_reversal_pct": 0.0,
                    "adaptive_threshold": adaptive_threshold}
        
        # Construir índices para slice vetorizado
        future_indices = valid_indices[:, np.newaxis] + np.arange(1, max_future + 1)
        future_indices = np.clip(future_indices, 0, len(prices) - 1)
        
        future_prices = prices[future_indices]
        
        if level_type == LevelType.SUPPORT.value:
            extremes = np.max(future_prices, axis=1)
            reversals = (extremes - level) / level * 100
        else:
            extremes = np.min(future_prices, axis=1)
            reversals = (level - extremes) / level * 100
        
        reversals = np.maximum(reversals, 0)  # Clamp negatives
        
        median_reversal = float(np.median(reversals))
        score = StatisticalUtils.clamp(median_reversal / adaptive_threshold * 10, 0, 10)
        
        return {
            "score": float(score),
            "reversals": reversals.tolist(),
            "avg_reversal_pct": float(np.mean(reversals)),
            "adaptive_threshold": float(adaptive_threshold)
        }
    
    def _filter_and_rank_levels(self, levels: List[Dict], current_price: float,
                               num_levels: int, ascending: bool) -> List[Dict]:
        """Filtra e classifica níveis"""
        if not levels:
            return []
        
        for level in levels:
            level["distance"] = abs(level["price"] - current_price)
            level["distance_percent"] = StatisticalUtils.safe_divide(
                level["distance"], current_price, 0.0
            ) * 100
        
        # Ordenar: maior score primeiro, depois menor distância
        levels.sort(key=lambda x: (
            -x["composite_score"],
            x["distance"],
            x["price"] if ascending else -x["price"]
        ))
        
        # Filtrar níveis muito próximos
        filtered_levels = []
        for level in levels:
            if not filtered_levels:
                filtered_levels.append(level)
            else:
                too_close = any(
                    abs(level["price"] - existing["price"]) / current_price < self.config.merge_tolerance
                    for existing in filtered_levels
                )
                if not too_close:
                    filtered_levels.append(level)
            
            if len(filtered_levels) >= num_levels:
                break
        
        return filtered_levels
    
    def _calculate_defense_zones(self, support_levels: List[Dict], 
                                resistance_levels: List[Dict],
                                current_price: float) -> Dict:
        """Calcula zonas de defesa institucionais"""
        bull_defense, bear_defense, no_mans_land = None, None, None
        
        if len(support_levels) >= 2:
            primary = support_levels[0]
            secondary = support_levels[1]
            bull_defense = {
                "primary": {
                    "price": primary["price"],
                    "width": abs(primary["price"] - secondary["price"]),
                    "strength": primary["composite_score"],
                    "confidence": primary["confidence_interval"]
                },
                "secondary": {
                    "price": secondary["price"],
                    "width": abs(primary["price"] - secondary["price"]),
                    "strength": secondary["composite_score"],
                    "confidence": secondary["confidence_interval"]
                },
                "zone_type": "bull_defense",
                "depth": len(support_levels)
            }
        elif support_levels:
            primary = support_levels[0]
            bull_defense = {
                "primary": {
                    "price": primary["price"],
                    "width": 0.0,
                    "strength": primary["composite_score"],
                    "confidence": primary["confidence_interval"]
                },
                "secondary": None,
                "zone_type": "bull_defense",
                "depth": 1
            }
        
        if len(resistance_levels) >= 2:
            primary = resistance_levels[0]
            secondary = resistance_levels[1]
            bear_defense = {
                "primary": {
                    "price": primary["price"],
                    "width": abs(primary["price"] - secondary["price"]),
                    "strength": primary["composite_score"],
                    "confidence": primary["confidence_interval"]
                },
                "secondary": {
                    "price": secondary["price"],
                    "width": abs(primary["price"] - secondary["price"]),
                    "strength": secondary["composite_score"],
                    "confidence": secondary["confidence_interval"]
                },
                "zone_type": "bear_defense",
                "depth": len(resistance_levels)
            }
        elif resistance_levels:
            primary = resistance_levels[0]
            bear_defense = {
                "primary": {
                    "price": primary["price"],
                    "width": 0.0,
                    "strength": primary["composite_score"],
                    "confidence": primary["confidence_interval"]
                },
                "secondary": None,
                "zone_type": "bear_defense",
                "depth": 1
            }
        
        if support_levels and resistance_levels:
            strongest_support = max(support_levels, key=lambda x: x["composite_score"])
            strongest_resistance = max(resistance_levels, key=lambda x: x["composite_score"])
            
            if strongest_support["price"] < strongest_resistance["price"]:
                width = strongest_resistance["price"] - strongest_support["price"]
                no_mans_land = {
                    "start": strongest_support["price"],
                    "end": strongest_resistance["price"],
                    "width": float(width),
                    "width_percent": float(StatisticalUtils.safe_divide(width, current_price, 0.0) * 100),
                    "contains_price": bool(strongest_support["price"] <= current_price <= strongest_resistance["price"])
                }
        
        return {
            "bull_defense": bull_defense,
            "bear_defense": bear_defense,
            "no_mans_land": no_mans_land,
            "market_context": {
                "is_bullish": len(support_levels) > len(resistance_levels),
                "support_count": len(support_levels),
                "resistance_count": len(resistance_levels),
                "closest_support": min([abs(s["price"] - current_price) for s in support_levels]) if support_levels else None,
                "closest_resistance": min([abs(r["price"] - current_price) for r in resistance_levels]) if resistance_levels else None
            }
        }
    
    def _generate_quality_report(self, support_levels: List[Dict], 
                                resistance_levels: List[Dict],
                                prices: np.ndarray) -> Dict:
        """Gera relatório de qualidade dos níveis"""
        all_levels = support_levels + resistance_levels
        
        if not all_levels:
            return {
                "overall_quality": 0.0,
                "avg_composite_score": 0.0,
                "avg_touches": 0.0,
                "avg_cluster_size": 0.0,
                "avg_stability": 0.0,
                "avg_reaction": 0.0,
                "diversity_score": 0.0,
                "level_count": 0,
                "support_count": len(support_levels),
                "resistance_count": len(resistance_levels),
                "quality_rating": QualityRating.INSUFFICIENT_DATA.value,
                "details": {}
            }
        
        avg_composite_score = np.mean([l["composite_score"] for l in all_levels])
        avg_touches = np.mean([l["touches"] for l in all_levels])
        avg_cluster_size = np.mean([l["cluster_size"] for l in all_levels])
        avg_stability = np.mean([l.get("stability_score", 5.0) for l in all_levels])
        avg_reaction = np.mean([l.get("reaction_score", 5.0) for l in all_levels])
        
        level_prices = [l["price"] for l in all_levels]
        price_range = max(level_prices) - min(level_prices) if level_prices else 0
        mean_price = np.mean(prices)
        diversity_score = StatisticalUtils.safe_divide(price_range, mean_price, 0.0) if mean_price > 0 else 0
        
        overall_quality = float(
            0.35 * avg_composite_score +
            0.20 * min(avg_touches / 10 * 10, 10) +
            0.15 * min(avg_cluster_size / 10 * 10, 10) +
            0.15 * avg_stability +
            0.10 * avg_reaction +
            0.05 * min(diversity_score * 50, 10)
        )
        
        clamped_quality = StatisticalUtils.clamp(overall_quality, 0, 10)
        
        return {
            "overall_quality": float(clamped_quality),
            "avg_composite_score": float(avg_composite_score),
            "avg_touches": float(avg_touches),
            "avg_cluster_size": float(avg_cluster_size),
            "avg_stability": float(avg_stability),
            "avg_reaction": float(avg_reaction),
            "diversity_score": float(diversity_score),
            "level_count": len(all_levels),
            "support_count": len(support_levels),
            "resistance_count": len(resistance_levels),
            "quality_rating": self._get_quality_rating(clamped_quality).value
        }
    
    def _get_quality_rating(self, score: float) -> QualityRating:
        """Converte score em rating qualitativo"""
        if score >= CONSTANTS.QUALITY_EXCELLENT:
            return QualityRating.EXCELLENT
        elif score >= CONSTANTS.QUALITY_GOOD:
            return QualityRating.GOOD
        elif score >= CONSTANTS.QUALITY_MODERATE:
            return QualityRating.MODERATE
        elif score >= CONSTANTS.QUALITY_WEAK:
            return QualityRating.WEAK
        else:
            return QualityRating.POOR
    
    def _empty_result(self) -> SRAnalysisResult:
        """Resultado vazio"""
        return {
            "support_levels": [],
            "resistance_levels": [],
            "defense_zones": {
                "bull_defense": None,
                "bear_defense": None,
                "no_mans_land": None,
                "market_context": {}
            },
            "quality_report": {
                "overall_quality": 0.0,
                "avg_composite_score": 0.0,
                "avg_touches": 0.0,
                "avg_cluster_size": 0.0,
                "avg_stability": 0.0,
                "avg_reaction": 0.0,
                "diversity_score": 0.0,
                "level_count": 0,
                "support_count": 0,
                "resistance_count": 0,
                "quality_rating": QualityRating.INSUFFICIENT_DATA.value
            },
            "current_price": 0.0,
            "lookback_period": self.lookback_period,
            "timestamp": datetime.now().isoformat()
        }