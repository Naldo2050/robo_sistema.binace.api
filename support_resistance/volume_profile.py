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
        
        prices = np.asarray(self.price_data, dtype=np.float64)
        volumes = np.asarray(self.volume_data, dtype=np.float64)
        
        min_price, max_price = np.min(prices), np.max(prices)
        
        # Edge case: preço constante
        if np.isclose(min_price, max_price):
            return self._handle_constant_price(prices, volumes)
        
        # Criar bins usando linspace
        edges = np.linspace(min_price, max_price, self.config.bins + 1)
        
        # Usar np.histogram com weights
        volume_per_bin, _ = np.histogram(prices, bins=edges, weights=volumes)
        price_centers = (edges[:-1] + edges[1:]) / 2
        
        total_volume = float(np.sum(volumes))
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
            volume_at_idx = float(volume_per_bin[idx])
            z_score = StatisticalUtils.safe_divide(
                volume_at_idx - float(volume_mean), 
                float(volume_std), 
                0.0
            )
            strength = StatisticalUtils.normalize_score(z_score, -3, 3, 0, 10)
            hvn_levels.append({
                "price": float(price_centers[idx]),
                "strength": float(strength),
                "z": float(z_score),
                "volume": volume_at_idx
            })
        
        # Força do POC
        poc_z_score = StatisticalUtils.safe_divide(float(poc_volume) - float(volume_mean), float(volume_std), 0.0)
        poc_strength = StatisticalUtils.normalize_score(poc_z_score, -3, 3, 0, 10)
        
        # Força de VAH/VAL
        top_20_idx = int(len(volume_per_bin) * 0.2)
        top_20_percent_volume = float(np.sum(np.sort(volume_per_bin)[-max(top_20_idx, 1):]))
        volume_concentration = StatisticalUtils.safe_divide(top_20_percent_volume, float(total_volume), 0.0)
        vah_val_strength = StatisticalUtils.clamp(7.0 + (volume_concentration * 3), 5, 9)
        
        # Posição atual
        current_price = float(prices[-1])
        poc_deviation = StatisticalUtils.safe_divide(current_price - float(poc_price), float(poc_price), 0.0) * 100
        
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
                "percent_of_total": float(StatisticalUtils.safe_divide(float(poc_volume), float(total_volume), 0.0) * 100),
                "strength": float(poc_strength)
            },
            "value_area": {
                "low": float(value_area_low),
                "high": float(value_area_high),
                "width": float(value_area_high - value_area_low),
                "percent_width": float(StatisticalUtils.safe_divide(
                    float(value_area_high - value_area_low), float(poc_price), 0.0
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
        
        volume_left = float(np.sum(volume_per_bin[:current_idx]))
        volume_right = float(np.sum(volume_per_bin[current_idx:]))
        
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
        top_20_percent_volume = float(np.sum(np.sort(volume_per_bin)[-top_20_idx:]))
        volume_concentration = StatisticalUtils.safe_divide(top_20_percent_volume, float(total_volume), 0.0)
        
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

    def calculate_value_area_volume_pct(self, profile: Optional[Dict] = None) -> Dict:
        """
        Calcula que porcentagem do volume total está dentro da Value Area.
        
        Normal: ~70% (por definição).
        Se > 80%: Volume muito concentrado → breakout iminente.
        Se < 60%: Volume disperso → mercado em tendência.
        
        Args:
            profile: Resultado de calculate_profile(). Se None, calcula internamente.
            
        Returns:
            Dict com porcentagem, interpretação e risco de breakout.
        """
        if profile is None:
            try:
                profile = self.calculate_profile()
            except Exception:
                return {
                    "value_area_volume_pct": 0.0,
                    "interpretation": "error",
                    "breakout_risk": "UNKNOWN",
                }

        val = profile.get("value_area", {}).get("low", 0)
        vah = profile.get("value_area", {}).get("high", 0)
        
        if val == 0 or vah == 0 or val >= vah:
            return {
                "value_area_volume_pct": 0.0,
                "interpretation": "insufficient_data",
                "breakout_risk": "UNKNOWN",
            }

        # Calcular volume dentro da VA vs total
        try:
            if hasattr(self, 'price_data') and hasattr(self, 'volume_data'):
                mask = (self.price_data >= val) & (self.price_data <= vah)
                vol_in_va = float(self.volume_data[mask].sum())
                total_vol = float(self.volume_data.sum())
            else:
                # Fallback: usar dados do profile
                poc_vol_pct = profile.get("poc", {}).get("percent_of_total", 0)
                # Estimativa baseada na configuração (default 70%)
                vol_in_va = 70.0
                total_vol = 100.0
        except Exception:
            vol_in_va = 70.0
            total_vol = 100.0

        if total_vol <= 0:
            va_pct = 0.0
        else:
            va_pct = round((vol_in_va / total_vol) * 100, 1)

        # Classificar
        if va_pct > 85:
            interpretation = "extremely_compressed"
            breakout_risk = "VERY_HIGH"
        elif va_pct > 80:
            interpretation = "compressed"
            breakout_risk = "HIGH"
        elif va_pct > 75:
            interpretation = "slightly_compressed"
            breakout_risk = "MEDIUM"
        elif va_pct > 65:
            interpretation = "normal"
            breakout_risk = "LOW"
        elif va_pct > 55:
            interpretation = "dispersed"
            breakout_risk = "LOW"
        else:
            interpretation = "very_dispersed_trending"
            breakout_risk = "LOW"

        return {
            "value_area_volume_pct": va_pct,
            "interpretation": interpretation,
            "breakout_risk": breakout_risk,
            "volume_in_va": round(vol_in_va, 4),
            "total_volume": round(total_vol, 4),
            "compression_signal": va_pct > 80,
        }

    def detect_no_mans_land(self, profile: Optional[dict] = None, current_price: float = 0) -> dict:
        """
        Detecta No-Man's Land: zonas de preço com BAIXO volume (LVN)
        entre High Volume Nodes onde o preço tende a se mover RÁPIDO.
        
        Se o preço entrar num no-man's land, pode haver um move explosivo
        até o próximo HVN. Essencial para sizing, stops e expectations.
        
        Args:
            profile: Resultado de calculate_profile(). Se None, calcula internamente.
            current_price: Preço atual para calcular proximidade e posição.
            
        Returns:
            Dict com zonas de no-man's land, posição do preço e riscos.
        """
        default = {
            "zones": [],
            "price_in_no_mans_land": False,
            "nearest_no_mans_land": None,
            "total_zones": 0,
            "status": "no_data",
        }

        if profile is None:
            try:
                profile = self.calculate_profile()
            except Exception:
                return default

        volume_nodes = profile.get("volume_nodes", {})
        hvn_list = volume_nodes.get("hvn", [])
        lvn_list = volume_nodes.get("lvn", [])

        if len(hvn_list) < 2:
            return {**default, "status": "insufficient_hvns"}

        # Ordenar HVNs
        hvn_sorted = sorted(hvn_list)

        # Usar current_price para referência
        if current_price <= 0:
            pos = profile.get("current_position", {})
            current_price = pos.get("price", 0)

        no_mans_lands = []

        # Encontrar gaps entre HVNs consecutivos
        for i in range(len(hvn_sorted) - 1):
            zone_low = hvn_sorted[i]
            zone_high = hvn_sorted[i + 1]
            gap_size = zone_high - zone_low

            if zone_low <= 0:
                continue

            gap_pct = (gap_size / zone_low) * 100

            # Gap significativo: > 0.3% do preço
            if gap_pct < 0.3:
                continue

            # Verificar se há LVNs nessa zona (confirma baixo volume)
            lvns_in_gap = [lvn for lvn in lvn_list if zone_low < lvn < zone_high]
            has_lvn_confirmation = len(lvns_in_gap) > 0

            # Classificar risco
            if gap_pct > 1.5:
                risk = "CRITICAL"
            elif gap_pct > 1.0:
                risk = "HIGH"
            elif gap_pct > 0.5:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            # Distância ao preço atual
            if current_price > 0:
                if current_price < zone_low:
                    distance_to_zone = zone_low - current_price
                    direction = "above"
                elif current_price > zone_high:
                    distance_to_zone = current_price - zone_high
                    direction = "below"
                else:
                    distance_to_zone = 0
                    direction = "inside"
                distance_pct = (distance_to_zone / current_price) * 100 if current_price > 0 else 0
            else:
                distance_to_zone = 0
                distance_pct = 0
                direction = "unknown"

            no_mans_lands.append({
                "range_low": round(zone_low, 2),
                "range_high": round(zone_high, 2),
                "gap_size": round(gap_size, 2),
                "gap_size_pct": round(gap_pct, 4),
                "risk": risk,
                "lvn_confirmed": has_lvn_confirmation,
                "lvns_in_zone": len(lvns_in_gap),
                "nearest_hvn_below": round(zone_low, 2),
                "nearest_hvn_above": round(zone_high, 2),
                "distance_from_price": round(distance_to_zone, 2),
                "distance_pct": round(distance_pct, 4),
                "direction": direction,
            })

        # Verificar se preço está em no-man's land
        price_in_nml = any(
            nml["range_low"] <= current_price <= nml["range_high"]
            for nml in no_mans_lands
        ) if current_price > 0 else False

        # Encontrar zona mais próxima
        nearest = None
        if no_mans_lands and current_price > 0:
            # Ordenar por distância
            sorted_by_dist = sorted(no_mans_lands, key=lambda z: z["distance_from_price"])
            nearest = sorted_by_dist[0]

        # Ordenar zones por gap_size (maiores primeiro)
        no_mans_lands.sort(key=lambda z: z["gap_size_pct"], reverse=True)

        return {
            "zones": no_mans_lands[:10],  # Top 10 maiores gaps
            "price_in_no_mans_land": price_in_nml,
            "nearest_no_mans_land": nearest,
            "total_zones": len(no_mans_lands),
            "max_gap_pct": round(no_mans_lands[0]["gap_size_pct"], 4) if no_mans_lands else 0,
            "warning": (
                "PRICE IN NO-MANS LAND - Expect fast move to nearest HVN"
                if price_in_nml
                else "Price at HVN - Normal conditions"
                if not no_mans_lands
                else None
            ),
            "status": "success",
        }

    def score_volume_nodes(
        self,
        profile: Optional[dict] = None,
        current_price: float = 0,
        weekly_nodes: Optional[dict] = None,
        monthly_nodes: Optional[dict] = None,
    ) -> dict:
        """
        Pontua HVN/LVN de 0 a 100 baseado em:
          1. Volume relativo ao POC (0-35 pontos)
          2. Proximidade ao preço atual (0-30 pontos)
          3. Confluência multi-timeframe (0-25 pontos)
          4. Posição relativa à Value Area (0-10 pontos)
        
        Args:
            profile: Resultado de calculate_profile(). Se None, calcula internamente.
            current_price: Preço atual para proximidade.
            weekly_nodes: Dict com hvn/lvn semanais para confluência.
                         Ex: {"hvn": [68000, 69125], "lvn": [66500]}
            monthly_nodes: Dict com hvn/lvn mensais para confluência.
                          Ex: {"hvn": [73139, 77540], "lvn": [75000]}
            
        Returns:
            Dict com HVNs e LVNs pontuados e ordenados por strength.
        """
        default = {
            "scored_hvns": [],
            "scored_lvns": [],
            "strongest_hvn": None,
            "strongest_lvn": None,
            "status": "no_data",
        }

        if profile is None:
            try:
                profile = self.calculate_profile()
            except Exception:
                return default

        volume_nodes = profile.get("volume_nodes", {})
        hvn_list = volume_nodes.get("hvn", [])
        lvn_list = volume_nodes.get("lvn", [])
        hvn_levels = volume_nodes.get("hvn_levels", [])

        poc_data = profile.get("poc", {})
        poc_price = poc_data.get("price", 0)
        poc_volume = poc_data.get("volume", 0)

        va = profile.get("value_area", {})
        val = va.get("low", 0)
        vah = va.get("high", 0)

        if current_price <= 0:
            pos = profile.get("current_position", {})
            current_price = pos.get("price", 0)

        # Preparar dados de confluência multi-TF
        weekly_hvn = set()
        weekly_lvn = set()
        monthly_hvn = set()
        monthly_lvn = set()

        if weekly_nodes and isinstance(weekly_nodes, dict):
            weekly_hvn = set(weekly_nodes.get("hvn", []) or [])
            weekly_lvn = set(weekly_nodes.get("lvn", []) or [])
        if monthly_nodes and isinstance(monthly_nodes, dict):
            monthly_hvn = set(monthly_nodes.get("hvn", []) or [])
            monthly_lvn = set(monthly_nodes.get("lvn", []) or [])

        def _is_near(price1, price2, tolerance_pct=0.3):
            """Verifica se dois preços estão próximos (dentro de tolerance_pct%)."""
            if price1 <= 0 or price2 <= 0:
                return False
            return abs(price1 - price2) / price1 * 100 < tolerance_pct

        def _score_node(node_price, node_type="hvn"):
            """Calcula score de 0-100 para um node."""
            score = 0

            # 1. Volume relativo (0-35 pontos)
            # Usar hvn_levels se disponível para pegar volume do node
            node_volume = 0
            node_z = 0
            for lvl in hvn_levels:
                if isinstance(lvl, dict) and _is_near(lvl.get("price", 0), node_price, 0.05):
                    node_volume = lvl.get("volume", 0)
                    node_z = lvl.get("z", 0)
                    break

            if poc_volume > 0 and node_volume > 0:
                vol_ratio = node_volume / poc_volume
                score += min(35, vol_ratio * 35)
            elif node_z > 0:
                score += min(35, node_z * 10)
            else:
                score += 15  # Score base se não tem dados detalhados

            # 2. Proximidade ao preço atual (0-30 pontos)
            if current_price > 0 and node_price > 0:
                distance_pct = abs(node_price - current_price) / current_price * 100
                # Mais próximo = mais pontos (decay linear)
                proximity_score = max(0, 30 - (distance_pct * 5))
                score += proximity_score

            # 3. Confluência multi-TF (0-25 pontos)
            confluence_sources = []

            # Confluência com weekly
            for w_node in (weekly_hvn if node_type == "hvn" else weekly_lvn):
                if _is_near(node_price, w_node):
                    confluence_sources.append("weekly")
                    score += 12
                    break

            # Confluência com monthly
            for m_node in (monthly_hvn if node_type == "hvn" else monthly_lvn):
                if _is_near(node_price, m_node):
                    confluence_sources.append("monthly")
                    score += 13
                    break

            # 4. Posição relativa à Value Area (0-10 pontos)
            if val > 0 and vah > 0:
                if val <= node_price <= vah:
                    # Dentro da VA = mais relevante para HVN
                    score += 10 if node_type == "hvn" else 5
                else:
                    # Fora da VA = mais relevante para LVN
                    score += 5 if node_type == "hvn" else 10

            return {
                "price": round(node_price, 2),
                "strength": min(round(score), 100),
                "volume_score": round(min(35, (node_volume / poc_volume * 35) if poc_volume > 0 else 15), 1),
                "proximity_score": round(max(0, 30 - (abs(node_price - current_price) / current_price * 100 * 5)) if current_price > 0 else 0, 1),
                "multi_tf_confluence": len(confluence_sources) > 0,
                "confluence_sources": confluence_sources,
                "in_value_area": val <= node_price <= vah if val > 0 and vah > 0 else None,
            }

        # Pontuar todos os HVNs
        scored_hvns = [_score_node(h, "hvn") for h in hvn_list if h > 0]
        scored_hvns.sort(key=lambda x: x["strength"], reverse=True)

        # Pontuar todos os LVNs
        scored_lvns = [_score_node(l, "lvn") for l in lvn_list if l > 0]
        scored_lvns.sort(key=lambda x: x["strength"], reverse=True)

        return {
            "scored_hvns": scored_hvns[:10],  # Top 10
            "scored_lvns": scored_lvns[:10],
            "strongest_hvn": scored_hvns[0] if scored_hvns else None,
            "strongest_lvn": scored_lvns[0] if scored_lvns else None,
            "total_hvns": len(scored_hvns),
            "total_lvns": len(scored_lvns),
            "avg_hvn_strength": round(
                sum(h["strength"] for h in scored_hvns) / len(scored_hvns), 1
            ) if scored_hvns else 0,
            "avg_lvn_strength": round(
                sum(l["strength"] for l in scored_lvns) / len(scored_lvns), 1
            ) if scored_lvns else 0,
            "status": "success",
        }