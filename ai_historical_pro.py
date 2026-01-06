# ai_historical_pro.py - Enhanced Version for AI Training
# Coletor histórico profissional aprimorado para IA com:
# - Microestrutura de mercado avançada
# - Volume Profile adaptativo com OHLC weighted
# - Regime detection e contexto de sessões
# - Features dinâmicas e normalizadas para ML
# - Validação de qualidade de dados
# - Rate limiting inteligente
# - Sistema de cache otimizado

import json
import time
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import requests
import numpy as np
import pandas as pd
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UTC = ZoneInfo("UTC")
NY_TZ = ZoneInfo("America/New_York")
SP_TZ = ZoneInfo("America/Sao_Paulo")
LONDON_TZ = ZoneInfo("Europe/London")

BINANCE_URL = "https://api.binance.com/api/v3/klines"

# Diretórios de saída
FEATURES_DIR = Path("features"); FEATURES_DIR.mkdir(exist_ok=True)
REPORTS_DIR  = Path("reports");  REPORTS_DIR.mkdir(exist_ok=True)
MEMORY_DIR   = Path("memory");   MEMORY_DIR.mkdir(exist_ok=True)
CACHE_DIR    = Path("cache");    CACHE_DIR.mkdir(exist_ok=True)

# -----------------------
# SISTEMA DE CACHE INTELIGENTE
# -----------------------
class SmartCache:
    def __init__(self, ttl_hours: int = 1):
        self.ttl_seconds = ttl_hours * 3600
        self.cache = {}
        self.timestamps = {}
    
    def _get_key(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> str:
        data = f"{symbol}_{interval}_{start_ms}_{end_ms}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> Optional[List]:
        key = self._get_key(symbol, interval, start_ms, end_ms)
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl_seconds:
                logger.debug(f"Cache hit para {symbol} {interval}")
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, symbol: str, interval: str, start_ms: int, end_ms: int, data: List):
        key = self._get_key(symbol, interval, start_ms, end_ms)
        self.cache[key] = data
        self.timestamps[key] = time.time()
    
    def clear_expired(self):
        now = time.time()
        expired = [k for k, ts in self.timestamps.items() if now - ts > self.ttl_seconds]
        for k in expired:
            del self.cache[k]
            del self.timestamps[k]

# -----------------------
# VALIDAÇÃO DE QUALIDADE DOS DADOS
# -----------------------
class DataQualityValidator:
    def __init__(self):
        self.quality_thresholds = {
            "max_zero_volume_pct": 0.05,  # Máximo 5% de períodos com volume zero
            "max_extreme_move_pct": 0.10,  # Máximo 10% movimento de preço
            "min_data_completeness": 0.95,  # Mínimo 95% dados completos
            "max_time_gap_minutes": 5  # Máximo 5 minutos de gap
        }
    
    def validate_klines(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida qualidade dos dados de klines"""
        if df.empty:
            return {"is_valid": False, "score": 0, "issues": ["DataFrame vazio"]}
        
        issues = []
        score = 100
        
        # 1. Verifica volume zero
        zero_vol_pct = (df["volume"] == 0).sum() / len(df)
        if zero_vol_pct > self.quality_thresholds["max_zero_volume_pct"]:
            issues.append(f"Alto percentual de volume zero: {zero_vol_pct:.2%}")
            score -= 20
        
        # 2. Verifica movimentos extremos de preço
        price_changes = df["close"].pct_change().abs()
        extreme_moves_pct = (price_changes > self.quality_thresholds["max_extreme_move_pct"]).sum() / len(df)
        if extreme_moves_pct > 0.01:  # >1% dos dados
            issues.append(f"Muitos movimentos extremos: {extreme_moves_pct:.2%}")
            score -= 15
        
        # 3. Verifica completude dos dados
        completeness = df.notna().all(axis=1).sum() / len(df)
        if completeness < self.quality_thresholds["min_data_completeness"]:
            issues.append(f"Dados incompletos: {completeness:.2%}")
            score -= 25
        
        # 4. Verifica gaps temporais
        if len(df) > 1:
            time_gaps = df["open_time"].diff().dt.total_seconds() / 60
            max_gap = time_gaps.max()
            if max_gap > self.quality_thresholds["max_time_gap_minutes"]:
                issues.append(f"Gap temporal detectado: {max_gap:.1f} minutos")
                score -= 10
        
        # 5. Verifica consistência OHLC
        ohlc_valid = (df["high"] >= df[["open", "close"]].max(axis=1)).all() and \
                    (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        if not ohlc_valid:
            issues.append("Inconsistência em dados OHLC")
            score -= 30
        
        return {
            "is_valid": len(issues) == 0,
            "quality_score": max(0, score),
            "issues": issues,
            "metrics": {
                "zero_volume_pct": zero_vol_pct,
                "extreme_moves_pct": extreme_moves_pct,
                "completeness": completeness,
                "max_time_gap_min": time_gaps.max() if len(df) > 1 else 0
            }
        }

# -----------------------
# RATE LIMITING INTELIGENTE
# -----------------------
class AdaptiveRateLimiter:
    def __init__(self):
        self.request_times = deque(maxlen=100)
        self.current_delay = 0.12
        self.min_delay = 0.05
        self.max_delay = 2.0
        self.error_count = 0
    
    def should_wait(self) -> float:
        """Retorna tempo de espera baseado na performance recente"""
        if self.request_times:
            avg_time = np.mean(list(self.request_times))
            if avg_time > 1.0:  # Resposta lenta
                self.current_delay = min(self.max_delay, self.current_delay * 1.2)
            elif avg_time < 0.3 and self.error_count == 0:  # Resposta rápida
                self.current_delay = max(self.min_delay, self.current_delay * 0.9)
        
        # Penalidade por erros
        if self.error_count > 0:
            self.current_delay = min(self.max_delay, self.current_delay * (1 + self.error_count * 0.5))
        
        return self.current_delay
    
    def record_request(self, duration: float, error: bool = False):
        """Registra tempo de request e ajusta delay"""
        self.request_times.append(duration)
        if error:
            self.error_count = min(5, self.error_count + 1)
        else:
            self.error_count = max(0, self.error_count - 1)

# -----------------------
# DOWNLOAD OTIMIZADO DE KLINES
# -----------------------
def fetch_klines_enhanced(
    symbol: str, 
    interval: str, 
    start_ms: int, 
    end_ms: int, 
    cache: Optional[SmartCache] = None,
    rate_limiter: Optional[AdaptiveRateLimiter] = None
) -> List:
    """Fetch klines com cache e rate limiting inteligente"""
    
    # Verifica cache primeiro
    if cache:
        cached_data = cache.get(symbol, interval, start_ms, end_ms)
        if cached_data:
            return cached_data
    
    out = []
    limit = 1000
    cur = start_ms
    
    while cur < end_ms:
        # Rate limiting
        if rate_limiter:
            delay = rate_limiter.should_wait()
            time.sleep(delay)
        
        start_time = time.time()
        error_occurred = False
        
        try:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cur,
                "endTime": end_ms,
                "limit": limit
            }
            
            r = requests.get(BINANCE_URL, params=params, timeout=25)
            r.raise_for_status()
            kl = r.json()
            
            if not kl:
                break
                
            out.extend(kl)
            last_close = kl[-1][6]  # close_time
            cur = last_close + 1
            
        except Exception as e:
            logger.error(f"Erro ao baixar klines: {e}")
            error_occurred = True
            time.sleep(min(5.0, (time.time() - start_time) * 2))
            cur += 60000  # Pula 1 minuto
        
        finally:
            # Registra performance
            if rate_limiter:
                rate_limiter.record_request(time.time() - start_time, error_occurred)
    
    # Salva no cache
    if cache and out:
        cache.set(symbol, interval, start_ms, end_ms, out)
    
    return out

def klines_to_df(kl):
    """Converte klines para DataFrame com validação"""
    if not kl:
        return pd.DataFrame()
    
    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ]
    
    try:
        df = pd.DataFrame(kl, columns=cols)
        num_cols = ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]
        df[num_cols] = df[num_cols].astype(float)
        df["trades"] = df["trades"].astype(int)
        df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.sort_values("open_time").reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Erro ao converter klines: {e}")
        return pd.DataFrame()

# -----------------------
# VOLUME PROFILE AVANÇADO
# -----------------------
def enhanced_volume_profile(df: pd.DataFrame, bins="adaptive", method="ohlc_weighted") -> Dict[str, Any]:
    """
    Volume Profile avançado com diferentes métodos de cálculo
    """
    if df is None or len(df) == 0:
        return {"poc": None, "vah": None, "val": None, "hvns": [], "lvns": [], "method": method}

    try:
        # Escolhe preço base baseado no método
        if method == "ohlc_weighted":
            # Média ponderada OHLC
            prices = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
        elif method == "typical":
            # Preço típico (HLC/3)
            prices = (df["high"] + df["low"] + df["close"]) / 3.0
        elif method == "vwap_weighted":
            # Ponderado pelo volume
            cumvol = df["volume"].cumsum()
            prices = (df["close"] * df["volume"]).cumsum() / cumvol
        else:
            # Método padrão (close)
            prices = df["close"]

        vols = df["volume"].to_numpy()
        prices = prices.to_numpy()

        if len(prices) == 0 or np.nansum(vols) == 0:
            return {"poc": None, "vah": None, "val": None, "hvns": [], "lvns": [], "method": method}

        lo, hi = float(np.nanmin(prices)), float(np.nanmax(prices))
        if lo == hi:
            return {"poc": lo, "vah": lo, "val": lo, "hvns": [lo], "lvns": [], "method": method}

        # Bins adaptativos baseados na volatilidade
        if bins == "adaptive":
            volatility = df["close"].pct_change().std()
            bins = max(100, min(500, int(200 + volatility * 1000)))
        elif isinstance(bins, str):
            bins = 240  # fallback

        # Calcula histograma
        hist, edges = np.histogram(prices, bins=bins, range=(lo, hi), weights=vols)
        centers = (edges[:-1] + edges[1:]) / 2.0
        total = hist.sum()

        if total == 0:
            return {"poc": None, "vah": None, "val": None, "hvns": [], "lvns": [], "method": method}

        # POC (Point of Control)
        poc_idx = int(hist.argmax())
        poc = float(centers[poc_idx])

        # Value Area (70% do volume)
        sorted_indices = np.argsort(hist)[::-1]
        acc_volume = 0.0
        value_area_indices = set()
        
        for idx in sorted_indices:
            value_area_indices.add(int(idx))
            acc_volume += hist[idx]
            if acc_volume / total >= 0.70:
                break

        vah = float(centers[max(value_area_indices)])  # Value Area High
        val = float(centers[min(value_area_indices)])  # Value Area Low

        # High Volume Nodes (top 15%)
        high_threshold = np.percentile(hist[hist > 0], 85)
        hvn_indices = np.where(hist >= high_threshold)[0]
        hvns = [float(centers[i]) for i in hvn_indices]

        # Low Volume Nodes (bottom 15%, mas não zero)
        nonzero_hist = hist[hist > 0]
        if len(nonzero_hist) > 0:
            low_threshold = np.percentile(nonzero_hist, 15)
            lvn_indices = np.where((hist > 0) & (hist <= low_threshold))[0]
            lvns = [float(centers[i]) for i in lvn_indices]
        else:
            lvns = []

        return {
            "poc": poc,
            "vah": vah,
            "val": val,
            "hvns": hvns,
            "lvns": lvns,
            "method": method,
            "total_volume": float(total),
            "value_area_volume_pct": float(acc_volume / total * 100),
            "bins_used": int(bins)
        }

    except Exception as e:
        logger.error(f"Erro no volume profile: {e}")
        return {"poc": None, "vah": None, "val": None, "hvns": [], "lvns": [], "method": method, "error": str(e)}

# -----------------------
# CONTEXTO DE SESSÕES DE TRADING
# -----------------------
def add_trading_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona contexto de sessões de trading globais"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Horários em diferentes fusos
    df["hour_ny"] = df["open_time"].dt.tz_convert(NY_TZ).dt.hour
    df["hour_london"] = df["open_time"].dt.tz_convert(LONDON_TZ).dt.hour
    df["hour_tokyo"] = df["open_time"].dt.tz_convert("Asia/Tokyo").dt.hour
    
    # Sessões principais
    df["session_asian"] = ((df["hour_tokyo"] >= 9) & (df["hour_tokyo"] <= 15)).astype(int)
    df["session_london"] = ((df["hour_london"] >= 8) & (df["hour_london"] <= 16)).astype(int)
    df["session_ny"] = ((df["hour_ny"] >= 9) & (df["hour_ny"] <= 16)).astype(int)
    
    # Overlaps importantes
    df["overlap_london_ny"] = (df["session_london"] & df["session_ny"]).astype(int)
    df["overlap_asian_london"] = (df["session_asian"] & df["session_london"]).astype(int)
    
    # Momentos críticos
    df["london_open"] = (df["hour_london"] == 8).astype(int)
    df["ny_open"] = (df["hour_ny"] == 9).astype(int)
    df["asian_close"] = (df["hour_tokyo"] == 15).astype(int)
    
    # Fim de semana (baixa liquidez)
    df["weekend"] = df["open_time"].dt.weekday.isin([5, 6]).astype(int)
    
    return df

# -----------------------
# DETECÇÃO DE REGIME DE MERCADO
# -----------------------
def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta diferentes regimes de mercado"""
    if df.empty or len(df) < 20:
        return df
    
    df = df.copy()
    
    # 1. Trending vs Ranging
    # ADR (Average Daily Range) como proxy para volatilidade
    df["adr_20"] = (df["high"] - df["low"]).rolling(20).mean()
    df["price_momentum_20"] = df["close"].pct_change(20)
    
    # Score de tendência
    df["trending_score"] = np.abs(df["price_momentum_20"]) / (df["adr_20"] / df["close"] + 1e-9)
    df["is_trending"] = (df["trending_score"] > df["trending_score"].rolling(50).quantile(0.7)).astype(int)
    
    # 2. Regime de volatilidade
    df["returns"] = df["close"].pct_change()
    df["vol_20"] = df["returns"].rolling(20).std()
    df["vol_60"] = df["returns"].rolling(60).std()
    df["vol_regime"] = (df["vol_20"] > df["vol_60"]).astype(int)  # 1 = alta vol, 0 = baixa vol
    
    # 3. Momentum regime
    df["rsi_14"] = calculate_rsi(df["close"], 14)
    df["momentum_regime"] = np.where(df["rsi_14"] > 70, 1,  # Overbought
                           np.where(df["rsi_14"] < 30, -1, 0))  # Oversold, Neutral
    
    # 4. Volume regime
    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["vol_expansion"] = (df["volume"] > df["vol_ma_20"] * 1.5).astype(int)
    
    return df

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calcula RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -----------------------
# FEATURES DE MICROESTRUTURA
# -----------------------
def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computa features avançadas de microestrutura de mercado"""
    if df.empty:
        return df
    
    df = df.copy()
    eps = 1e-9
    
    # 1. Order Flow Features
    df["order_flow_imbalance"] = (df["taker_buy_base"] / (df["volume"] + eps)) - 0.5
    df["cumulative_delta"] = (df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])).cumsum()
    
    # 2. Volume-Price Features
    # Volume-Price Trend (VPT)
    df["vpt"] = (df["close"].pct_change() * df["volume"]).cumsum()
    
    # Accumulation/Distribution Line
    df["ad_line"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / ((df["high"] - df["low"]) + eps) * df["volume"]
    df["ad_line"] = df["ad_line"].cumsum()
    
    # 3. Price-Volume Efficiency
    df["pv_efficiency"] = df["close"].pct_change().abs() / (df["volume"].rolling(20).mean() + eps)
    
    # 4. Spread Proxies (usando OHLC)
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"]
    df["oc_gap"] = np.abs(df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    
    # 5. Market Impact Features
    df["volume_impact"] = df["close"].pct_change().abs() / np.log(df["volume"] + eps)
    
    # 6. Tick Direction (proxy usando close)
    df["tick_direction"] = np.sign(df["close"].diff())
    df["tick_momentum"] = df["tick_direction"].rolling(10).sum()
    
    # 7. Volume Distribution Features
    df["buy_volume_ratio"] = df["taker_buy_base"] / (df["volume"] + eps)
    df["sell_volume_ratio"] = 1 - df["buy_volume_ratio"]
    df["volume_imbalance"] = df["buy_volume_ratio"] - 0.5
    
    return df

# -----------------------
# FEATURES DINÂMICAS PARA ML
# -----------------------
class DynamicFeatureEngine:
    def __init__(self):
        self.base_periods = [5, 15, 30, 60, 240]  # em minutos
        self.volatility_periods = [10, 20, 50]
    
    def compute_adaptive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computa features que se adaptam à volatilidade atual"""
        if df.empty or len(df) < max(self.base_periods):
            return df
        
        df = df.copy()
        
        # Detecta regime de volatilidade atual
        current_vol = df["close"].pct_change().rolling(20).std().iloc[-1]
        
        # Ajusta períodos baseado na volatilidade
        if current_vol > 0.02:  # Alta volatilidade
            periods = [max(1, p // 2) for p in self.base_periods]
        elif current_vol < 0.005:  # Baixa volatilidade  
            periods = [p * 2 for p in self.base_periods]
        else:
            periods = self.base_periods
        
        # Computa features para cada período
        for p in periods:
            if p < len(df):
                # Momentum features
                df[f"momentum_{p}"] = df["close"].pct_change(p)
                df[f"momentum_vol_{p}"] = df["close"].pct_change(p).rolling(p).std()
                
                # Volume features
                df[f"vol_ratio_{p}"] = df["volume"] / (df["volume"].rolling(p).mean() + 1e-9)
                df[f"vol_trend_{p}"] = df["volume"].rolling(p).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
                
                # Price position features
                df[f"price_position_{p}"] = ((df["close"] - df["low"].rolling(p).min()) / 
                                           (df["high"].rolling(p).max() - df["low"].rolling(p).min() + 1e-9))
                
                # Bollinger-like features
                df[f"bb_position_{p}"] = ((df["close"] - df["close"].rolling(p).mean()) / 
                                        (df["close"].rolling(p).std() + 1e-9))
        
        return df

def create_ml_optimized_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features otimizadas para modelos de Machine Learning"""
    if df.empty:
        return df
    
    df = df.copy()
    eps = 1e-9
    
    # 1. Normalized features (importantes para neural networks)
    scaler_20 = lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + eps)
    
    df["price_norm"] = scaler_20(df["close"])
    df["volume_norm"] = scaler_20(df["volume"])
    df["delta_norm"] = scaler_20(df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"]))
    
    # 2. Cyclical time features (evita vazamento temporal)
    hour = df["open_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    dow = df["open_time"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    
    # 3. Interaction features
    df["vol_price_interaction"] = df["volume_norm"] * df["price_norm"]
    df["momentum_vol_interaction"] = df["close"].pct_change(15) * df["volume_norm"]
    
    # 4. Lag features (importantes para séries temporais)
    for lag in [1, 3, 5, 15]:
        df[f"price_lag_{lag}"] = df["close"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        df[f"delta_lag_{lag}"] = df["delta_norm"].shift(lag)
    
    # 5. Rolling statistics
    for window in [5, 15, 60]:
        df[f"price_mean_{window}"] = df["close"].rolling(window).mean()
        df[f"price_std_{window}"] = df["close"].rolling(window).std()
        df[f"volume_mean_{window}"] = df["volume"].rolling(window).mean()
        df[f"volume_std_{window}"] = df["volume"].rolling(window).std()
    
    return df

# -----------------------
# FEATURES PRINCIPAIS (PIPELINE COMPLETO)
# -----------------------
def compute_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de features para IA"""
    if df.empty:
        return df
    
    logger.info("Computando features básicas...")
    df = compute_basic_features(df)
    
    logger.info("Adicionando contexto de sessões...")
    df = add_trading_sessions(df)
    
    logger.info("Detectando regime de mercado...")
    df = detect_market_regime(df)
    
    logger.info("Computando microestrutura...")
    df = compute_microstructure_features(df)
    
    logger.info("Adicionando features dinâmicas...")
    feature_engine = DynamicFeatureEngine()
    df = feature_engine.compute_adaptive_features(df)
    
    logger.info("Otimizando para ML...")
    df = create_ml_optimized_features(df)
    
    return df

def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features básicas OHLCV"""
    if df.empty:
        return df
    
    df = df.copy()
    eps = 1e-9
    
    # OHLC features
    df["body"] = np.abs(df["close"] - df["open"])
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])
    df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]
    
    # Ratios
    df["upper_wick_ratio"] = df["upper_wick"] / (df["range"] + eps)
    df["lower_wick_ratio"] = df["lower_wick"] / (df["range"] + eps)
    df["body_ratio"] = df["body"] / (df["range"] + eps)
    
    # Delta features
    df["delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])
    df["buy_ratio"] = df["taker_buy_base"] / (df["volume"] + eps)
    df["cvd"] = df["delta"].cumsum()
    
    # Returns
    for period in [1, 5, 15, 60]:
        df[f"ret_{period}m"] = df["close"].pct_change(period)
    
    # VWAP
    df["vwap_session"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    
    # Rolling features
    for window in [5, 15, 60]:
        df[f"vol_sum_{window}m"] = df["volume"].rolling(window).sum()
        df[f"delta_sum_{window}m"] = df["delta"].rolling(window).sum()
        
        # Z-scores
        df[f"vol_z_{window}m"] = (df["volume"] - df["volume"].rolling(window).mean()) / (df["volume"].rolling(window).std() + eps)
        df[f"delta_z_{window}m"] = (df["delta"] - df["delta"].rolling(window).mean()) / (df["delta"].rolling(window).std() + eps)
    
    # Stop runs detection
    roll_hi_20 = df["high"].rolling(20).max()
    roll_lo_20 = df["low"].rolling(20).min()
    df["stop_run_up"] = ((df["high"] > roll_hi_20.shift(1)) & 
                        (df["upper_wick_ratio"] > 0.6) & 
                        (df["body_ratio"] < 0.4)).astype(int)
    df["stop_run_down"] = ((df["low"] < roll_lo_20.shift(1)) & 
                          (df["lower_wick_ratio"] > 0.6) & 
                          (df["body_ratio"] < 0.4)).astype(int)
    
    return df

# -----------------------
# ABSORÇÕES APRIMORADAS
# -----------------------
def detect_enhanced_absorptions(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta absorções com heurísticas mais robustas e adaptativas"""
    if df.empty or len(df) < 60:
        return df
    
    df = df.copy()
    eps = 1e-9
    
    # Thresholds dinâmicos baseados na distribuição
    q_delta_hi = df["delta"].quantile(0.90)  # Top 10%
    q_delta_lo = df["delta"].quantile(0.10)  # Bottom 10%
    q_vol_hi = df["volume"].quantile(0.85)   # Top 15%
    
    # Z-scores robustos com rolling window
    z_delta = (df["delta"] - df["delta"].rolling(60, min_periods=20).mean()) / (df["delta"].rolling(60, min_periods=20).std() + eps)
    z_vol = (df["volume"] - df["volume"].rolling(60, min_periods=20).mean()) / (df["volume"].rolling(60, min_periods=20).std() + eps)
    z_delta = z_delta.fillna(0.0)
    z_vol = z_vol.fillna(0.0)
    
    # Condições para absorção mais refinadas
    ask_absorption_cond = (
        (df["delta"] >= q_delta_hi) &
        (df["volume"] >= q_vol_hi) &
        (df["upper_wick_ratio"] >= 0.5) &  # Menos restritivo
        (df["body_ratio"] <= 0.5) &
        (z_delta > 1.5) &  # Significância estatística
        (z_vol > 1.0)
    )
    
    bid_absorption_cond = (
        (df["delta"] <= q_delta_lo) &
        (df["volume"] >= q_vol_hi) &
        (df["lower_wick_ratio"] >= 0.5) &
        (df["body_ratio"] <= 0.5) &
        (z_delta < -1.5) &
        (z_vol > 1.0)
    )
    
    # Scores mais sofisticados
    ask_score = (z_delta.clip(lower=0) * 2 + 
                z_vol.clip(lower=0) + 
                df["upper_wick_ratio"] * 3 +
                (df["buy_ratio"] - 0.5) * 2)  # Considera proporção de compra
    
    bid_score = ((-z_delta).clip(lower=0) * 2 + 
                z_vol.clip(lower=0) + 
                df["lower_wick_ratio"] * 3 +
                (0.5 - df["buy_ratio"]) * 2)  # Considera proporção de venda
    
    df["ask_absorption"] = ask_absorption_cond.astype(int)
    df["bid_absorption"] = bid_absorption_cond.astype(int)
    df["ask_absorption_score"] = ask_score.where(ask_absorption_cond, 0.0)
    df["bid_absorption_score"] = bid_score.where(bid_absorption_cond, 0.0)
    
    # Absorções combinadas (mais forte)
    df["total_absorption"] = df["ask_absorption"] + df["bid_absorption"]
    df["absorption_strength"] = df["ask_absorption_score"] + df["bid_absorption_score"]
    
    return df

def cluster_defense_zones(df: pd.DataFrame, min_cluster_size=3, max_gap_minutes=5) -> List[Dict[str, Any]]:
    """Identifica clusters de absorção como zonas de defesa"""
    if df.empty:
        return []
    
    zones = []
    
    for side_col, side_name in [("ask_absorption", "ASK_DEFENSE"), ("bid_absorption", "BID_DEFENSE")]:
        absorption_points = df[df[side_col] == 1].copy()
        
        if len(absorption_points) < min_cluster_size:
            continue
        
        # Agrupa absorções próximas no tempo
        absorption_points["time_group"] = (
            absorption_points["open_time"].diff().dt.total_seconds() > (max_gap_minutes * 60)
        ).cumsum()
        
        for group_id, group in absorption_points.groupby("time_group"):
            if len(group) >= min_cluster_size:
                zone = {
                    "type": side_name,
                    "start_time": group["open_time"].iloc[0].isoformat(),
                    "end_time": group["open_time"].iloc[-1].isoformat(),
                    "price_anchor": float(group["close"].median()),
                    "price_low": float(group["low"].min()),
                    "price_high": float(group["high"].max()),
                    "total_volume": float(group["volume"].sum()),
                    "avg_delta": float(group["delta"].mean()),
                    "strength_score": float(group[f"{side_col.split('_')[0]}_absorption_score"].sum()),
                    "touches_count": len(group),
                    "confidence": min(1.0, len(group) / 10.0)  # Máximo 10 toques = 100%
                }
                zones.append(zone)
    
    # Ordena por força
    zones.sort(key=lambda x: x["strength_score"], reverse=True)
    return zones

# -----------------------
# DISTÂNCIAS A NÍVEIS APRIMORADAS
# -----------------------
def attach_enhanced_distances(df: pd.DataFrame, profile: dict, tolerance_bps=15, prefix="d1") -> pd.DataFrame:
    """Calcula distâncias para níveis com mais contexto"""
    if df.empty:
        return df
    
    df = df.copy()
    close_prices = df["close"].to_numpy()
    
    def bps_distance(price_array, reference_level):
        if reference_level is None or reference_level <= 0:
            return pd.Series(np.nan, index=df.index)
        return pd.Series(10000.0 * (price_array - reference_level) / reference_level, index=df.index)
    
    # Níveis principais
    poc = profile.get("poc")
    vah = profile.get("vah") 
    val = profile.get("val")
    hvns = np.array(profile.get("hvns", []), dtype=float)
    lvns = np.array(profile.get("lvns", []), dtype=float)
    
    # Distâncias básicas
    df[f"dist_poc_bps_{prefix}"] = bps_distance(close_prices, poc)
    df[f"dist_vah_bps_{prefix}"] = bps_distance(close_prices, vah)
    df[f"dist_val_bps_{prefix}"] = bps_distance(close_prices, val)
    
    # Níveis mais próximos
    if len(hvns) > 0:
        nearest_hvn_idx = np.argmin(np.abs(hvns.reshape(-1, 1) - close_prices.reshape(1, -1)), axis=0)
        nearest_hvns = hvns[nearest_hvn_idx]
        df[f"dist_nearest_hvn_bps_{prefix}"] = bps_distance(close_prices, nearest_hvns)
        df[f"nearest_hvn_price_{prefix}"] = nearest_hvns
    else:
        df[f"dist_nearest_hvn_bps_{prefix}"] = np.nan
        df[f"nearest_hvn_price_{prefix}"] = np.nan
    
    if len(lvns) > 0:
        nearest_lvn_idx = np.argmin(np.abs(lvns.reshape(-1, 1) - close_prices.reshape(1, -1)), axis=0)
        nearest_lvns = lvns[nearest_lvn_idx]
        df[f"dist_nearest_lvn_bps_{prefix}"] = bps_distance(close_prices, nearest_lvns)
        df[f"nearest_lvn_price_{prefix}"] = nearest_lvns
    else:
        df[f"dist_nearest_lvn_bps_{prefix}"] = np.nan
        df[f"nearest_lvn_price_{prefix}"] = np.nan
    
    # Proximidade com diferentes tolerâncias
    for tol_name, tol_val in [("tight", 5), ("normal", tolerance_bps), ("wide", 25)]:
        # Resistências
        near_vah = (~df[f"dist_vah_bps_{prefix}"].isna()) & (df[f"dist_vah_bps_{prefix}"].abs() <= tol_val)
        near_hvn = (~df[f"dist_nearest_hvn_bps_{prefix}"].isna()) & (df[f"dist_nearest_hvn_bps_{prefix}"].abs() <= tol_val)
        df[f"near_resistance_{tol_name}_{prefix}"] = (near_vah | near_hvn).astype(int)
        
        # Suportes
        near_val = (~df[f"dist_val_bps_{prefix}"].isna()) & (df[f"dist_val_bps_{prefix}"].abs() <= tol_val)
        near_lvn = (~df[f"dist_nearest_lvn_bps_{prefix}"].isna()) & (df[f"dist_nearest_lvn_bps_{prefix}"].abs() <= tol_val)
        df[f"near_support_{tol_name}_{prefix}"] = (near_val | near_lvn).astype(int)
    
    # Posição relativa no value area
    if poc is not None and vah is not None and val is not None:
        value_area_range = vah - val
        if value_area_range > 0:
            df[f"va_position_{prefix}"] = (close_prices - val) / value_area_range
            df[f"above_poc_{prefix}"] = (close_prices > poc).astype(int)
            df[f"below_poc_{prefix}"] = (close_prices < poc).astype(int)
            df[f"in_value_area_{prefix}"] = ((close_prices >= val) & (close_prices <= vah)).astype(int)
        else:
            df[f"va_position_{prefix}"] = 0.5
            df[f"above_poc_{prefix}"] = 0
            df[f"below_poc_{prefix}"] = 0
            df[f"in_value_area_{prefix}"] = 1
    
    return df

# -----------------------
# PIPELINE PRINCIPAL APRIMORADO  
# -----------------------
def build_enhanced_historical(
    symbol="BTCUSDT", 
    bins_profile="adaptive", 
    deep_mode=False, 
    save_raw=False,
    use_cache=True,
    quality_threshold=70
):
    """Pipeline principal aprimorado para coleta de dados históricos"""
    
    # Inicialização
    cache = SmartCache(ttl_hours=2) if use_cache else None
    rate_limiter = AdaptiveRateLimiter()
    validator = DataQualityValidator()
    
    now = datetime.now(UTC)
    start_24h = now - timedelta(hours=24)
    start_7d = now - timedelta(days=7)
    start_30d = now - timedelta(days=30)
    
    logger.info(f"Iniciando coleta histórica para {symbol}")
    logger.info(f"Modo deep: {deep_mode}, Cache: {use_cache}, Threshold qualidade: {quality_threshold}")
    
    # Intervalos (deep mode usa 1m para tudo)
    intervals = {
        "24h": "1m",
        "7d": "1m" if deep_mode else "15m", 
        "30d": "1m" if deep_mode else "1h"
    }
    
    datasets = {}
    profiles = {}
    
    # Coleta dados para cada período
    for period, interval in intervals.items():
        start_time = {"24h": start_24h, "7d": start_7d, "30d": start_30d}[period]
        
        logger.info(f"Baixando dados {period} ({interval}) para {symbol}...")
        
        try:
            klines = fetch_klines_enhanced(
                symbol, interval,
                int(start_time.timestamp() * 1000),
                int(now.timestamp() * 1000),
                cache=cache,
                rate_limiter=rate_limiter
            )
            
            if not klines:
                logger.warning(f"Nenhum dado recebido para {period}")
                continue
            
            df = klines_to_df(klines)
            
            if df.empty:
                logger.warning(f"DataFrame vazio para {period}")
                continue
            
            # Validação de qualidade
            quality = validator.validate_klines(df)
            logger.info(f"Qualidade dos dados {period}: {quality['quality_score']:.1f}%")
            
            if quality['quality_score'] < quality_threshold:
                logger.warning(f"Qualidade baixa para {period}: {quality['issues']}")
                if not deep_mode:  # Em modo normal, continua mesmo com qualidade baixa
                    logger.warning("Continuando com dados de baixa qualidade...")
            
            # Processa features
            logger.info(f"Processando features para {period}...")
            df = compute_comprehensive_features(df)
            df = detect_enhanced_absorptions(df)
            
            datasets[period] = df
            
            # Volume Profile
            logger.info(f"Calculando volume profile {period}...")
            profile = enhanced_volume_profile(df, bins=bins_profile, method="ohlc_weighted")
            profiles[period] = profile
            
        except Exception as e:
            logger.error(f"Erro ao processar {period}: {e}")
            continue
    
    if not datasets:
        raise RuntimeError("Nenhum dataset válido foi coletado")
    
    # Dataset principal (24h ou o mais recente disponível)
    main_period = "24h" if "24h" in datasets else list(datasets.keys())[0]
    main_df = datasets[main_period].copy()
    
    logger.info("Integrando dados multi-timeframe...")
    
    # Adiciona distâncias para todos os perfis disponíveis
    for period, profile in profiles.items():
        if profile and profile.get("poc"):
            prefix_map = {"24h": "d1", "7d": "w1", "30d": "m1"}
            prefix = prefix_map.get(period, period)
            main_df = attach_enhanced_distances(main_df, profile, prefix=prefix)
    
    # Clusters de defesa
    logger.info("Identificando zonas de defesa...")
    defense_zones = cluster_defense_zones(main_df)
    
    # Candidatos de entrada multi-timeframe
    logger.info("Identificando candidatos de entrada...")
    support_cols = [col for col in main_df.columns if col.startswith("near_support_")]
    resistance_cols = [col for col in main_df.columns if col.startswith("near_resistance_")]
    
    if support_cols:
        support_confluence = main_df[support_cols].sum(axis=1)
        main_df["support_confluence_score"] = support_confluence
        main_df["entry_long_candidate"] = (
            (main_df["ask_absorption"] == 1) & 
            (support_confluence >= 2)  # Pelo menos 2 timeframes
        ).astype(int)
    
    if resistance_cols:
        resistance_confluence = main_df[resistance_cols].sum(axis=1)
        main_df["resistance_confluence_score"] = resistance_confluence  
        main_df["entry_short_candidate"] = (
            (main_df["bid_absorption"] == 1) & 
            (resistance_confluence >= 2)
        ).astype(int)
    
    # Timestamps em diferentes fusos
    main_df["ts_utc"] = main_df["open_time"]
    main_df["ts_ny"] = main_df["open_time"].dt.tz_convert(NY_TZ)
    main_df["ts_sp"] = main_df["open_time"].dt.tz_convert(SP_TZ)
    
    # Preparação para saída
    day_tag = now.astimezone(SP_TZ).strftime("%Y-%m-%d")
    
    # Seleciona colunas importantes (evita DataFrame muito grande)
    important_cols = [
        # Timestamps
        "ts_utc", "ts_ny", "ts_sp",
        # OHLCV básico
        "open", "high", "low", "close", "volume", "trades",
        # Order flow
        "taker_buy_base", "delta", "buy_ratio", "cvd",
        # Features básicas
        "body", "range", "upper_wick", "lower_wick", 
        "upper_wick_ratio", "lower_wick_ratio", "body_ratio",
        # VWAP e returns
        "vwap_session", "ret_1m", "ret_5m", "ret_15m", "ret_60m",
        # Sessões
        "session_ny", "session_london", "overlap_london_ny",
        # Regime de mercado
        "is_trending", "vol_regime", "momentum_regime",
        # Microestrutura
        "order_flow_imbalance", "cumulative_delta", "pv_efficiency",
        # Volume features
        "vol_sum_5m", "vol_sum_15m", "vol_sum_60m",
        "vol_z_5m", "vol_z_15m", "vol_z_60m",
        # Absorções
        "ask_absorption", "bid_absorption", 
        "ask_absorption_score", "bid_absorption_score",
        # Candidatos
        "entry_long_candidate", "entry_short_candidate",
        # Features ML
        "price_norm", "volume_norm", "hour_sin", "hour_cos"
    ]
    
    # Adiciona colunas de distância se existirem
    distance_cols = [col for col in main_df.columns if "dist_" in col or "near_" in col]
    important_cols.extend(distance_cols)
    
    # Filtra colunas existentes
    available_cols = [col for col in important_cols if col in main_df.columns]
    output_df = main_df[available_cols].copy()
    
    # Resumo executivo
    summary = {
        "symbol": symbol,
        "generated_at_utc": now.isoformat(),
        "data_quality": {
            period: validator.validate_klines(df)["quality_score"] 
            for period, df in datasets.items()
        },
        "periods_collected": list(datasets.keys()),
        "total_candles": {period: len(df) for period, df in datasets.items()},
        "profiles": profiles,
        "defense_zones_count": len(defense_zones),
        "entry_candidates": {
            "long": int(output_df["entry_long_candidate"].sum()) if "entry_long_candidate" in output_df else 0,
            "short": int(output_df["entry_short_candidate"].sum()) if "entry_short_candidate" in output_df else 0
        }
    }
    
    # Níveis consolidados
    levels_data = {
        "symbol": symbol,
        "created_at": now.isoformat(),
        "timeframes": {
            period: {
                "profile": profile,
                "data_quality": summary["data_quality"].get(period, 0),
                "candles_count": len(datasets[period]) if period in datasets else 0
            }
            for period, profile in profiles.items()
        }
    }
    
    # Paths de saída
    paths = {
        "dataset_parquet": FEATURES_DIR / f"ai_hist_enhanced_{symbol}_{day_tag}.parquet",
        "dataset_csv": REPORTS_DIR / f"ai_hist_enhanced_{symbol}_{day_tag}.csv",
        "summary_json": REPORTS_DIR / f"ai_summary_enhanced_{symbol}_{day_tag}.json",
        "levels_json": MEMORY_DIR / f"ai_levels_enhanced_{symbol}_{day_tag}.json",
        "defense_zones_json": REPORTS_DIR / f"ai_defense_zones_{symbol}_{day_tag}.json"
    }
    
    # Salva arquivos principais
    logger.info("Salvando arquivos de saída...")
    
    # Dataset principal
    output_df.to_parquet(paths["dataset_parquet"], index=False)
    output_df.to_csv(paths["dataset_csv"], index=False, encoding="utf-8")
    
    # Resumo e metadados
    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    with open(paths["levels_json"], "w", encoding="utf-8") as f:
        json.dump(levels_data, f, indent=2, ensure_ascii=False)
    
    with open(paths["defense_zones_json"], "w", encoding="utf-8") as f:
        json.dump({
            "symbol": symbol,
            "created_at": now.isoformat(),
            "zones": defense_zones
        }, f, indent=2, ensure_ascii=False)
    
    # Salva datasets brutos opcionalmente
    if save_raw:
        logger.info("Salvando datasets brutos...")
        for period, df in datasets.items():
            raw_path = FEATURES_DIR / f"ai_raw_{period}_{symbol}_{day_tag}.parquet"
            try:
                df.to_parquet(raw_path, index=False)
                paths[f"raw_{period}"] = raw_path
            except Exception as e:
                logger.warning(f"Erro ao salvar raw {period}: {e}")
    
    # Relatório final
    logger.info("=" * 80)
    logger.info("COLETA HISTÓRICA FINALIZADA")
    logger.info("=" * 80)
    logger.info(f"Símbolo: {symbol}")
    logger.info(f"Períodos coletados: {', '.join(datasets.keys())}")
    logger.info(f"Total de features: {len(output_df.columns)}")
    logger.info(f"Candles finais: {len(output_df)}")
    logger.info(f"Zonas de defesa identificadas: {len(defense_zones)}")
    logger.info(f"Candidatos Long: {summary['entry_candidates']['long']}")
    logger.info(f"Candidatos Short: {summary['entry_candidates']['short']}")
    
    for period, quality in summary["data_quality"].items():
        logger.info(f"Qualidade {period}: {quality:.1f}%")
    
    logger.info("\nArquivos salvos:")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")
    
    logger.info("=" * 80)
    
    return {
        "datasets": output_df,
        "summary": summary,
        "defense_zones": defense_zones,
        "profiles": profiles,
        "file_paths": {str(k): str(v) for k, v in paths.items()}
    }

# -----------------------
# FUNÇÃO MAIN APRIMORADA
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced AI Historical Data Exporter - Versão Profissional para Machine Learning"
    )
    
    parser.add_argument("--symbol", default="BTCUSDT", 
                       help="Símbolo para análise (default: BTCUSDT)")
    parser.add_argument("--bins", default="adaptive", 
                       help="Bins para volume profile: 'adaptive' ou número (default: adaptive)")
    parser.add_argument("--deep", action="store_true", 
                       help="Modo profundo: usa 1m para todos os timeframes (mais lento)")
    parser.add_argument("--save-raw", action="store_true", 
                       help="Salva datasets brutos adicionalmente")
    parser.add_argument("--no-cache", action="store_true", 
                       help="Desabilita cache (força novo download)")
    parser.add_argument("--quality-threshold", type=int, default=70,
                       help="Threshold mínimo de qualidade dos dados (default: 70)")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Nível de logging")
    
    args = parser.parse_args()
    
    # Configura logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Converte bins se for numérico
    bins = args.bins
    if bins.isdigit():
        bins = int(bins)
    
    try:
        result = build_enhanced_historical(
            symbol=args.symbol,
            bins_profile=bins,
            deep_mode=args.deep,
            save_raw=args.save_raw,
            use_cache=not args.no_cache,
            quality_threshold=args.quality_threshold
        )
        
        print("\n✅ PROCESSO FINALIZADO COM SUCESSO!")
        print(f"📊 Dataset principal: {len(result['datasets'])} registros")
        print(f"🎯 Features extraídas: {len(result['datasets'].columns)}")
        print(f"📁 Arquivos salvos em: features/, reports/, memory/")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ ERRO CRÍTICO: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())