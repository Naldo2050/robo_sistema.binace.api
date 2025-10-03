# VOLUME PROFILE
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

class DynamicVolumeProfile:
    def __init__(self, symbol: str, base_bins: int = 20):
        """
        Volume Profile Dinâmico — adapta bins e zonas com base em:
        - Volatilidade (ATR)
        - Atividade de whales (volume > threshold)
        - Fluxo de ordens (CVD, delta)
        """
        self.symbol = symbol
        self.base_bins = base_bins
        logging.info(f"✅ Volume Profile Dinâmico inicializado para {symbol} (bins base: {base_bins})")

    def calculate_dynamic_params(self, df: pd.DataFrame, atr: float = 0.0, whale_ratio: float = 0.0, cvd_ratio: float = 0.0) -> Dict[str, Any]:
        """
        Calcula parâmetros dinâmicos para o Volume Profile.
        - df: DataFrame com trades (p, q, m)
        - atr: Average True Range (normalizado pelo preço)
        - whale_ratio: % do volume vindo de whales
        - cvd_ratio: |CVD| / volume_total
        """
        if df.empty or len(df) < 10:
            return {
                "dynamic_bins": self.base_bins,
                "value_area_pct": 0.70,
                "hvn_sensitivity": 1.0,
                "lvn_sensitivity": 1.0
            }
        
        price_range = df['p'].max() - df['p'].min()
        if price_range <= 0:
            price_range = df['p'].iloc[-1] * 0.01  # fallback: 1% do preço
        
        # 🔹 1. Bins dinâmicos: mais bins em alta volatilidade e atividade de whales
        volatility_factor = max(0.5, min(2.0, atr / df['p'].iloc[-1] if df['p'].iloc[-1] > 0 else 0.01))
        whale_factor = max(0.8, min(1.5, 1.0 + whale_ratio))  # +50% bins se whale_ratio > 0.5
        trend_factor = max(0.7, min(1.3, 1.0 + cvd_ratio))    # +30% bins se forte tendência
        
        dynamic_bins = int(self.base_bins * volatility_factor * whale_factor * trend_factor)
        dynamic_bins = max(10, min(50, dynamic_bins))  # limita entre 10 e 50 bins
        
        # 🔹 2. Value Area dinâmica: expande em alta volatilidade
        value_area_pct = max(0.60, min(0.80, 0.70 + (volatility_factor - 1.0) * 0.1))
        
        # 🔹 3. Sensibilidade HVN/LVN: mais sensível em mercados calmos
        hvn_sensitivity = max(0.5, min(2.0, 2.0 - volatility_factor))
        lvn_sensitivity = max(0.5, min(2.0, 2.0 - volatility_factor))
        
        return {
            "dynamic_bins": dynamic_bins,
            "value_area_pct": value_area_pct,
            "hvn_sensitivity": hvn_sensitivity,
            "lvn_sensitivity": lvn_sensitivity,
            "volatility_factor": volatility_factor,
            "whale_factor": whale_factor,
            "trend_factor": trend_factor
        }

    def calculate(self, df: pd.DataFrame, atr: float = 0.0, whale_buy_volume: float = 0.0, 
                  whale_sell_volume: float = 0.0, cvd: float = 0.0) -> Dict[str, Any]:
        """
        Calcula Volume Profile Dinâmico completo.
        Retorna: POC, VAH, VAL, HVNs, LVNs + parâmetros usados
        """
        if df.empty:
            return {
                "poc_price": 0.0, "vah": 0.0, "val": 0.0, "hvns": [], "lvns": [],
                "status": "error", "message": "DataFrame vazio"
            }
        
        # Validação de dados
        df = df.copy()
        df['p'] = pd.to_numeric(df['p'], errors='coerce')
        df['q'] = pd.to_numeric(df['q'], errors='coerce')
        df = df.dropna(subset=['p', 'q'])
        
        if df.empty:
            return {
                "poc_price": 0.0, "vah": 0.0, "val": 0.0, "hvns": [], "lvns": [],
                "status": "error", "message": "Dados inválidos após limpeza"
            }
        
        # Calcula parâmetros dinâmicos
        total_volume = df['q'].sum()
        whale_volume = whale_buy_volume + whale_sell_volume
        whale_ratio = whale_volume / total_volume if total_volume > 0 else 0.0
        cvd_ratio = abs(cvd) / total_volume if total_volume > 0 else 0.0
        
        params = self.calculate_dynamic_params(df, atr, whale_ratio, cvd_ratio)
        
        # Prepara dados para VP
        min_p, max_p = df['p'].min(), df['p'].max()
        if min_p == max_p:
            min_p = max_p - (max_p * 0.001)  # adiciona 0.1% de range artificial
        
        # Cria bins dinâmicos
        dynamic_bins = params["dynamic_bins"]
        bin_edges = np.linspace(min_p, max_p, dynamic_bins + 1)
        df['bin'] = pd.cut(df['p'], bins=bin_edges, include_lowest=True)
        volume_by_bin = df.groupby('bin', observed=False)['q'].sum()
        
        if len(volume_by_bin) == 0:
            return {
                "poc_price": 0.0, "vah": 0.0, "val": 0.0, "hvns": [], "lvns": [],
                "status": "error", "message": "Falha ao agrupar em bins"
            }
        
        # 🔹 POC (Point of Control)
        poc_bin = volume_by_bin.idxmax()
        poc_price = poc_bin.mid if hasattr(poc_bin, 'mid') else (poc_bin.left + poc_bin.right) / 2
        
        # 🔹 Value Area Dinâmica
        total_volume = volume_by_bin.sum()
        target_volume = total_volume * params["value_area_pct"]
        current_volume = volume_by_bin[poc_bin]
        lower_idx = volume_by_bin.index.get_loc(poc_bin)
        upper_idx = lower_idx
        
        # Expande VA para cima e para baixo
        while current_volume < target_volume:
            can_go_up = upper_idx + 1 < len(volume_by_bin)
            can_go_down = lower_idx - 1 >= 0
            
            if not can_go_up and not can_go_down:
                break
            
            vol_up = volume_by_bin.iloc[upper_idx + 1] if can_go_up else 0
            vol_down = volume_by_bin.iloc[lower_idx - 1] if can_go_down else 0
            
            if vol_up >= vol_down and can_go_up:
                upper_idx += 1
                current_volume += vol_up
            elif can_go_down:
                lower_idx -= 1
                current_volume += vol_down
            elif can_go_up:
                upper_idx += 1
                current_volume += vol_up
        
        val_bin = volume_by_bin.index[lower_idx]
        vah_bin = volume_by_bin.index[upper_idx]
        
        val = val_bin.mid if hasattr(val_bin, 'mid') else (val_bin.left + val_bin.right) / 2
        vah = vah_bin.mid if hasattr(vah_bin, 'mid') else (vah_bin.left + vah_bin.right) / 2
        
        # 🔹 HVNs e LVNs dinâmicos
        mean_vol = volume_by_bin.mean()
        std_vol = volume_by_bin.std()
        
        hvn_threshold = mean_vol + (std_vol * params["hvn_sensitivity"])
        lvn_threshold = max(mean_vol * 0.3 * params["lvn_sensitivity"], 1)
        
        hvns = []
        lvns = []
        
        for bin_interval, volume in volume_by_bin.items():
            price = bin_interval.mid if hasattr(bin_interval, 'mid') else (bin_interval.left + bin_interval.right) / 2
            if volume >= hvn_threshold:
                hvns.append(float(price))
            elif volume <= lvn_threshold:
                lvns.append(float(price))
        
        # Ordena HVNs e LVNs
        hvns.sort()
        lvns.sort()
        
        return {
            "poc_price": float(poc_price),
            "vah": float(vah),
            "val": float(val),
            "hvns": hvns,
            "lvns": lvns,
            "params_used": params,
            "total_volume": float(total_volume),
            "num_bins": dynamic_bins,
            "status": "success"
        }

    def get_zones_for_level_registry(self, vp_data: Dict) -> List[Dict]:
        """
        Converte resultado do VPD em formato compatível com LevelRegistry.
        """
        if vp_data.get("status") != "success":
            return []
        
        zones = []
        poc = vp_data["poc_price"]
        vah = vp_data["vah"]
        val = vp_data["val"]
        
        # Adiciona POC, VAH, VAL
        zones.append({"kind": "POC", "price": poc, "width_pct": 0.00035})
        zones.append({"kind": "VAH", "price": vah, "width_pct": 0.00040})
        zones.append({"kind": "VAL", "price": val, "width_pct": 0.00040})
        
        # Adiciona HVNs e LVNs
        for price in vp_data["hvns"]:
            zones.append({"kind": "HVN", "price": price, "width_pct": 0.00030})
        
        for price in vp_data["lvns"]:
            zones.append({"kind": "LVN", "price": price, "width_pct": 0.00025})
        
        return zones