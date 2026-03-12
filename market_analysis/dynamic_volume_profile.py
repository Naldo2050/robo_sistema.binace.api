# VOLUME PROFILE
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, cast
import logging
from pandas import Interval

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
        
        # Inicializa com valor padrão
        poc_price = 0.0
        
        if isinstance(poc_bin, Interval):
            poc_price = poc_bin.mid
        elif isinstance(poc_bin, (int, float, np.number)):
            poc_price = float(poc_bin)
        elif isinstance(poc_bin, str):
            try:
                poc_price = float(poc_bin)
            except:
                pass  # mantém valor padrão
        else:
            # Verifica se tem atributos left e right (somente se não for str)
            try:
                if hasattr(poc_bin, 'left') and hasattr(poc_bin, 'right'):
                    poc_price = (poc_bin.left + poc_bin.right) / 2
            except:
                pass  # mantém valor padrão
        
        # 🔹 Value Area Dinâmica
        total_volume = volume_by_bin.sum()
        target_volume = total_volume * params["value_area_pct"]
        current_volume = volume_by_bin[poc_bin]
        lower_idx = volume_by_bin.index.get_loc(poc_bin)
        upper_idx = lower_idx
        
        # Garante que lower_idx e upper_idx são inteiros
        if not isinstance(lower_idx, int) or not isinstance(upper_idx, int):
            return {
                "poc_price": float(poc_price), "vah": float(poc_price), "val": float(poc_price), 
                "hvns": [], "lvns": [], "params_used": params, 
                "total_volume": float(total_volume), "num_bins": dynamic_bins, "status": "warning"
            }
        
        # Expande VA para cima e para baixo
        while current_volume < target_volume:
            can_go_up = upper_idx + 1 < len(volume_by_bin)
            can_go_down = lower_idx - 1 >= 0
            
            if not can_go_up and not can_go_down:
                break
            
            vol_up = volume_by_bin.iloc[upper_idx + 1] if can_go_up else 0
            vol_down = volume_by_bin.iloc[lower_idx - 1] if can_go_down else 0
            
            # Garante que vol_up e vol_down são valores numéricos e não Series
            vol_up = float(vol_up) if vol_up is not None else 0.0
            vol_down = float(vol_down) if vol_down is not None else 0.0
            
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
        
        if isinstance(val_bin, Interval):
            val = val_bin.mid
        elif hasattr(val_bin, 'left') and hasattr(val_bin, 'right'):
            val = (val_bin.left + val_bin.right) / 2
        else:
            val = float(val_bin)
        
        if isinstance(vah_bin, Interval):
            vah = vah_bin.mid
        elif hasattr(vah_bin, 'left') and hasattr(vah_bin, 'right'):
            vah = (vah_bin.left + vah_bin.right) / 2
        else:
            vah = float(vah_bin)
        
        # 🔹 HVNs e LVNs dinâmicos
        mean_vol = volume_by_bin.mean()
        std_vol = volume_by_bin.std()
        
        hvn_threshold = mean_vol + (std_vol * params["hvn_sensitivity"])
        lvn_threshold = max(mean_vol * 0.3 * params["lvn_sensitivity"], 1)
        
        hvns = []
        lvns = []
        
        for bin_interval, volume in volume_by_bin.items():
            if isinstance(bin_interval, Interval):
                price = bin_interval.mid
            elif isinstance(bin_interval, (int, float, np.number)):
                price = float(bin_interval)
            elif hasattr(bin_interval, 'left') and hasattr(bin_interval, 'right'):
                # Força o tipo para pandas.Interval para eliminar erros de typing
                bi = cast(Interval, bin_interval)
                price = (bi.left + bi.right) / 2
            else:
                price = 0.0
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

    def detect_poor_extremes(self, df, vp_result: Optional[dict] = None) -> dict:
        """
        Detecta Poor High / Poor Low do Market Profile.
        
        Poor High: Topo da sessão com volume significativo.
                   O mercado NÃO rejeitou o nível → vai revisitar.
                   
        Poor Low:  Fundo da sessão com volume significativo.
                   O mercado NÃO rejeitou o nível → vai revisitar.
                   
        Excess High: Topo com volume mínimo (single prints).
                     Rejeição forte → menos provável revisitar.
                     
        Excess Low:  Fundo com volume mínimo (single prints).
                     Rejeição forte → menos provável revisitar.
        
        Args:
            df: DataFrame de trades com colunas 'p' (price) e 'q' (quantity)
            vp_result: Resultado de calculate() (opcional, para contexto)
            
        Returns:
            Dict com análise de poor/excess para high e low
        """
        import numpy as np

        default = {
            "poor_high": {"detected": False, "price": 0, "implication": "insufficient_data", "volume_ratio": 0},
            "poor_low": {"detected": False, "price": 0, "implication": "insufficient_data", "volume_ratio": 0},
            "excess_high": False,
            "excess_low": False,
            "status": "no_data",
        }

        if df is None or df.empty or "p" not in df.columns or "q" not in df.columns:
            return default

        try:
            prices = df["p"].astype(float)
            volumes = df["q"].astype(float)
        except Exception:
            return default

        if len(prices) < 10:
            return default

        session_high = float(prices.max())
        session_low = float(prices.min())
        price_range = session_high - session_low

        if price_range <= 0:
            return default

        # Definir zonas extremas: top 10% e bottom 10% do range
        high_zone_threshold = session_high - (price_range * 0.10)
        low_zone_threshold = session_low + (price_range * 0.10)

        # Volume nas zonas extremas
        vol_at_high_zone = float(volumes[prices >= high_zone_threshold].sum())
        vol_at_low_zone = float(volumes[prices <= low_zone_threshold].sum())
        total_vol = float(volumes.sum())

        if total_vol <= 0:
            return default

        # Volume médio por zona (10 zonas de 10%)
        avg_vol_per_zone = total_vol / 10.0

        # Ratios
        high_vol_ratio = vol_at_high_zone / avg_vol_per_zone if avg_vol_per_zone > 0 else 0
        low_vol_ratio = vol_at_low_zone / avg_vol_per_zone if avg_vol_per_zone > 0 else 0

        # Poor = volume no extremo > 50% da média por zona
        # Excess = volume no extremo < 20% da média por zona
        poor_high_detected = high_vol_ratio > 0.50
        poor_low_detected = low_vol_ratio > 0.50
        excess_high = high_vol_ratio < 0.20
        excess_low = low_vol_ratio < 0.20

        # Análise extra com close position (se OHLC disponível no vp_result)
        poc_price = 0
        if vp_result and isinstance(vp_result, dict):
            poc_price = vp_result.get("poc_price", 0)

        # Implicações
        if poor_high_detected:
            high_impl = "High likely to be revisited - unfinished auction"
        elif excess_high:
            high_impl = "High properly rejected - strong excess"
        else:
            high_impl = "High moderately tested - neutral"

        if poor_low_detected:
            low_impl = "Low likely to be revisited - unfinished auction"
        elif excess_low:
            low_impl = "Low properly rejected - strong excess"
        else:
            low_impl = "Low moderately tested - neutral"

        return {
            "poor_high": {
                "detected": poor_high_detected,
                "price": round(session_high, 2),
                "volume_ratio": round(high_vol_ratio, 4),
                "implication": high_impl,
            },
            "poor_low": {
                "detected": poor_low_detected,
                "price": round(session_low, 2),
                "volume_ratio": round(low_vol_ratio, 4),
                "implication": low_impl,
            },
            "excess_high": excess_high,
            "excess_low": excess_low,
            "session_high": round(session_high, 2),
            "session_low": round(session_low, 2),
            "poc_reference": round(poc_price, 2) if poc_price else None,
            "action_bias": (
                "expect_retest_high" if poor_high_detected and not poor_low_detected
                else "expect_retest_low" if poor_low_detected and not poor_high_detected
                else "expect_retest_both" if poor_high_detected and poor_low_detected
                else "range_complete" if excess_high and excess_low
                else "neutral"
            ),
            "status": "success",
        }

    def classify_profile_shape(self, df, vp_result: Optional[dict] = None) -> dict:
        """
        Classifica a forma do Volume Profile em categorias Market Profile.
        
        Shapes:
            P-shape: Volume concentrado no TOPO.
                     Indica short covering rally. Bearish bias após.
                     
            b-shape: Volume concentrado no FUNDO.
                     Indica long liquidation. Bullish bias após.
                     
            D-shape: Volume distribuído uniformemente (sino).
                     Mercado encontrou fair value. Range continuation.
                     
            B-shape: Bimodal (2 picos de volume separados).
                     Double distribution. Breakout iminente.
                     
            Thin:    Volume fino/disperso em todo o range.
                     Price discovery. Alta volatilidade esperada.
        
        Args:
            df: DataFrame de trades com colunas 'p' (price) e 'q' (quantity)
            vp_result: Resultado de calculate() (opcional)
            
        Returns:
            Dict com shape, implicação e distribuição por terços
        """
        import numpy as np

        default = {
            "shape": "unknown",
            "implication": "insufficient_data",
            "distribution": {"lower_third_pct": 0, "middle_third_pct": 0, "upper_third_pct": 0},
            "status": "no_data",
        }

        if df is None or df.empty or "p" not in df.columns or "q" not in df.columns:
            return default

        try:
            prices = df["p"].astype(float)
            volumes = df["q"].astype(float)
        except Exception:
            return default

        if len(prices) < 20:
            return default

        session_high = float(prices.max())
        session_low = float(prices.min())
        price_range = session_high - session_low

        if price_range <= 0:
            return default

        # Dividir em 3 terços
        lower_bound = session_low + (price_range / 3)
        upper_bound = session_high - (price_range / 3)

        lower_vol = float(volumes[prices <= lower_bound].sum())
        middle_vol = float(volumes[(prices > lower_bound) & (prices <= upper_bound)].sum())
        upper_vol = float(volumes[prices > upper_bound].sum())
        total_vol = lower_vol + middle_vol + upper_vol

        if total_vol <= 0:
            return default

        lower_pct = lower_vol / total_vol
        middle_pct = middle_vol / total_vol
        upper_pct = upper_vol / total_vol

        # Classificação
        if upper_pct > 0.45:
            shape = "P"
            implication = "Short covering rally - bearish bias expected"
            trading_signal = "BEARISH_AFTER"
        elif lower_pct > 0.45:
            shape = "b"
            implication = "Long liquidation - bullish bias expected"
            trading_signal = "BULLISH_AFTER"
        elif middle_pct > 0.50:
            shape = "D"
            implication = "Balanced market - fair value found, range likely"
            trading_signal = "RANGE"
        elif lower_pct > 0.32 and upper_pct > 0.32 and middle_pct < 0.30:
            shape = "B"
            implication = "Double distribution - breakout imminent"
            trading_signal = "BREAKOUT_EXPECTED"
        else:
            shape = "Thin"
            implication = "Price discovery - high volatility expected"
            trading_signal = "VOLATILE"

        # Detectar bimodalidade mais precisamente para B-shape
        if shape != "B":
            try:
                num_bins = max(10, min(30, len(prices) // 50))
                hist, bin_edges = np.histogram(
                    prices, bins=num_bins, weights=volumes
                )
                if len(hist) >= 5:
                    # Normalizar
                    hist_norm = hist / hist.max() if hist.max() > 0 else hist
                    # Procurar por 2 picos separados por vale
                    peaks = []
                    for i in range(1, len(hist_norm) - 1):
                        if hist_norm[i] > hist_norm[i - 1] and hist_norm[i] > hist_norm[i + 1]:
                            if hist_norm[i] > 0.4:
                                peaks.append(i)
                    
                    if len(peaks) >= 2:
                        # Verificar se há vale entre os picos
                        valley_min = min(hist_norm[peaks[0]:peaks[-1] + 1])
                        peak_avg = (hist_norm[peaks[0]] + hist_norm[peaks[-1]]) / 2
                        if valley_min < peak_avg * 0.5:
                            shape = "B"
                            implication = "Double distribution - breakout imminent"
                            trading_signal = "BREAKOUT_EXPECTED"
            except Exception:
                pass

        return {
            "shape": shape,
            "implication": implication,
            "trading_signal": trading_signal,
            "distribution": {
                "lower_third_pct": round(lower_pct * 100, 1),
                "middle_third_pct": round(middle_pct * 100, 1),
                "upper_third_pct": round(upper_pct * 100, 1),
            },
            "dominant_zone": (
                "upper" if upper_pct > lower_pct and upper_pct > middle_pct
                else "lower" if lower_pct > upper_pct and lower_pct > middle_pct
                else "middle"
            ),
            "status": "success",
        }