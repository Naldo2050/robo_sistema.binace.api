"""
Análise de Volume Profile Institucional
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional

from .utils import StatisticalUtils
from .config import VolumeProfileConfig
from .constants import MarketBias


class VolumeProfileAnalyzer:
    """Análise de Volume Profile com métricas institucionais"""
    
    def __init__(self, price_data: pd.Series, volume_data: pd.Series, 
                 config: Optional[VolumeProfileConfig] = None):
        """
        Inicializa o analisador de Volume Profile
        
        Args:
            price_data: Series de preços (close)
            volume_data: Series de volumes
            config: Configuração do Volume Profile
        """
        if config is None:
            config = VolumeProfileConfig()
        
        self.config = config
        self.price_data = price_data.dropna()
        
        # Reindex pelo índice limpo do preço
        self.volume_data = volume_data.reindex(self.price_data.index).dropna()
        
        # Garantir alinhamento
        common_idx = self.price_data.index.intersection(self.volume_data.index)
        self.price_data = self.price_data.loc[common_idx]
        self.volume_data = self.volume_data.loc[common_idx]
    
    def calculate_profile(self) -> Dict:
        """
        Calcula volume profile completo com métricas institucionais
        
        Returns:
            Dict com POC, Value Area, HVN/LVN e métricas
        """
        if len(self.price_data) < self.config.min_data_points or len(self.volume_data) < self.config.min_data_points:
            return self._empty_profile()
        
        prices = self.price_data.values
        volumes = self.volume_data.values
        
        min_price, max_price = np.min(prices), np.max(prices)
        
        # Edge case: preço constante
        if np.isclose(min_price, max_price):
            return self._handle_constant_price(prices, volumes)
        
        # Criar bins usando linspace
        edges = np.linspace(min_price, max_price, self.config.bins + 1)
        
        # Usar np.histogram com weights
        volume_per_bin, _ = np.histogram(prices, bins=edges, weights=volumes)
        price_centers = (edges[:-1] + edges[1:]) / 2
        
        total_volume = np.sum(volumes)
        if total_volume == 0:
            return self._empty_profile()
        
        # Verificar e corrigir conservação de volume
        calculated_total = float(np.sum(volume_per_bin))
        if calculated_total <= 0:
            return self._empty_profile()
        
        volume_loss = abs(calculated_total - total_volume) / total_volume
        if volume_loss > 1e-9:
            volume_per_bin = volume_per_bin * (total_volume / calculated_total)
        
        # POC (Point of Control)
        poc_idx = np.argmax(volume_per_bin)
        poc_price = price_centers[poc_idx]
        poc_volume = volume_per_bin[poc_idx]
        
        # Value Area (70% do volume)
        sorted_indices = np.argsort(volume_per_bin)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative_volume += volume_per_bin[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= total_volume * self.config.value_area_percent:
                break
        
        value_area_prices = price_centers[value_area_indices]
        value_area_low = np.min(value_area_prices)
        value_area_high = np.max(value_area_prices)
        
        # HVN/LVN
        volume_mean = np.mean(volume_per_bin)
        volume_std = np.std(volume_per_bin)
        
        hvn_threshold = volume_mean + self.config.hvn_sigma * volume_std
        lvn_threshold = volume_mean - self.config.lvn_sigma * volume_std
        
        hvn_indices = volume_per_bin > hvn_threshold
        lvn_indices = volume_per_bin < lvn_threshold
        
        hvn_prices = price_centers[hvn_indices]
        lvn_prices = price_centers[lvn_indices]
        
        # Força dos HVNs com base no z-score
        hvn_levels = []
        for idx in np.where(hvn_indices)[0]:
            z_score = StatisticalUtils.safe_divide(
                volume_per_bin[idx] - volume_mean, 
                volume_std, 
                0.0
            )
            strength = StatisticalUtils.normalize_score(z_score, -3, 3, 0, 10)
            hvn_levels.append({
                "price": float(price_centers[idx]),
                "strength": float(strength),
                "z": float(z_score),
                "volume": float(volume_per_bin[idx])
            })
        
        # Força do POC
        poc_z_score = StatisticalUtils.safe_divide(poc_volume - volume_mean, volume_std, 0.0)
        poc_strength = StatisticalUtils.normalize_score(poc_z_score, -3, 3, 0, 10)
        
        # Força de VAH/VAL
        top_20_idx = int(len(volume_per_bin) * 0.2)
        top_20_percent_volume = np.sum(np.sort(volume_per_bin)[-max(top_20_idx, 1):])
        volume_concentration = StatisticalUtils.safe_divide(top_20_percent_volume, total_volume, 0.0)
        vah_val_strength = StatisticalUtils.clamp(7.0 + (volume_concentration * 3), 5, 9)
        
        # Posição atual
        current_price = prices[-1]
        poc_deviation = StatisticalUtils.safe_divide(current_price - poc_price, poc_price, 0.0) * 100
        
        # Balance analysis
        profile_balance = self._analyze_profile_balance(price_centers, volume_per_bin, current_price)
        
        # Distribuição de volume
        epsilon = 1e-10
        volume_distribution = {
            "skewness": float(stats.skew(volume_per_bin)),
            "kurtosis": float(stats.kurtosis(volume_per_bin)),
            "entropy": float(stats.entropy(volume_per_bin + epsilon))
        }
        
        return {
            "price_bins": price_centers.tolist(),
            "volume_per_bin": volume_per_bin.tolist(),
            "poc": {
                "price": float(poc_price),
                "volume": float(poc_volume),
                "percent_of_total": float(StatisticalUtils.safe_divide(poc_volume, total_volume, 0.0) * 100),
                "strength": float(poc_strength)
            },
            "value_area": {
                "low": float(value_area_low),
                "high": float(value_area_high),
                "width": float(value_area_high - value_area_low),
                "percent_width": float(StatisticalUtils.safe_divide(
                    value_area_high - value_area_low, poc_price, 0.0
                ) * 100)
            },
            "volume_nodes": {
                "hvn": hvn_prices.tolist(),
                "lvn": lvn_prices.tolist(),
                "hvn_count": len(hvn_prices),
                "lvn_count": len(lvn_prices),
                "hvn_levels": hvn_levels,
                "poc_strength": float(poc_strength),
                "vah_val_strength": float(vah_val_strength)
            },
            "current_position": {
                "price": float(current_price),
                "vs_poc": float(poc_deviation),
                "in_value_area": bool(value_area_low <= current_price <= value_area_high),
                "distance_to_value_area": float(
                    min(abs(current_price - value_area_low), abs(current_price - value_area_high)) 
                    if not (value_area_low <= current_price <= value_area_high) else 0
                )
            },
            "profile_metrics": profile_balance,
            "total_volume": float(total_volume),
            "volume_distribution": volume_distribution
        }
    
    def _handle_constant_price(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """Trata caso de preço constante"""
        total_volume = float(np.sum(volumes))
        if total_volume <= 0:
            return self._empty_profile()

        poc_price = float(prices[0])

        return {
            "price_bins": [poc_price],
            "volume_per_bin": [total_volume],
            "poc": {
                "price": poc_price, 
                "volume": total_volume, 
                "percent_of_total": 100.0,
                "strength": 10.0
            },
            "value_area": {
                "low": poc_price, 
                "high": poc_price, 
                "width": 0.0, 
                "percent_width": 0.0
            },
            "volume_nodes": {
                "hvn": [], "lvn": [], 
                "hvn_count": 0, "lvn_count": 0,
                "hvn_levels": [],
                "poc_strength": 10.0,
                "vah_val_strength": 7.0
            },
            "current_position": {
                "price": float(prices[-1]),
                "vs_poc": 0.0,
                "in_value_area": True,
                "distance_to_value_area": 0.0
            },
            "profile_metrics": {
                "balance": 0.5,
                "bias": MarketBias.BALANCED.value,
                "volume_left": 0.0,
                "volume_right": total_volume,
                "volume_concentration": 1.0,
                "is_concentrated": True
            },
            "total_volume": total_volume,
            "volume_distribution": {"skewness": 0.0, "kurtosis": 0.0, "entropy": 0.0}
        }
    
    def _analyze_profile_balance(self, price_centers: np.ndarray, volume_per_bin: np.ndarray, 
                                current_price: float) -> Dict:
        """Analisa balanceamento do perfil de volume"""
        current_idx = np.searchsorted(price_centers, current_price)
        if current_idx >= len(price_centers):
            current_idx = len(price_centers) - 1
        
        volume_left = np.sum(volume_per_bin[:current_idx])
        volume_right = np.sum(volume_per_bin[current_idx:])
        
        total_volume = volume_left + volume_right
        if total_volume == 0:
            return {"balance": 0.5, "bias": MarketBias.NEUTRAL.value, "volume_left": 0.0, "volume_right": 0.0}
        
        balance_ratio = volume_left / total_volume
        
        if balance_ratio > 0.6:
            bias = MarketBias.BULLISH.value
        elif balance_ratio < 0.4:
            bias = MarketBias.BEARISH.value
        else:
            bias = MarketBias.BALANCED.value
        
        top_20_idx = max(int(len(volume_per_bin) * 0.2), 1)
        top_20_percent_volume = np.sum(np.sort(volume_per_bin)[-top_20_idx:])
        volume_concentration = StatisticalUtils.safe_divide(top_20_percent_volume, total_volume, 0.0)
        
        return {
            "balance": float(balance_ratio),
            "bias": bias,
            "volume_left": float(volume_left),
            "volume_right": float(volume_right),
            "volume_concentration": float(volume_concentration),
            "is_concentrated": bool(volume_concentration > 0.6)
        }
    
    def _empty_profile(self) -> Dict:
        """Retorna profile vazio"""
        return {
            "price_bins": [],
            "volume_per_bin": [],
            "poc": {"price": 0.0, "volume": 0.0, "percent_of_total": 0.0, "strength": 0.0},
            "value_area": {"low": 0.0, "high": 0.0, "width": 0.0, "percent_width": 0.0},
            "volume_nodes": {
                "hvn": [], "lvn": [], 
                "hvn_count": 0, "lvn_count": 0,
                "hvn_levels": [],
                "poc_strength": 0.0,
                "vah_val_strength": 0.0
            },
            "current_position": {
                "price": 0.0, "vs_poc": 0.0, "in_value_area": False, "distance_to_value_area": 0.0
            },
            "profile_metrics": {"balance": 0.5, "bias": MarketBias.NEUTRAL.value},
            "total_volume": 0.0,
            "volume_distribution": {"skewness": 0.0, "kurtosis": 0.0, "entropy": 0.0}
        }