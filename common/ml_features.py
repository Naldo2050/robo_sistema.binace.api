"""
ml_features.py v2.0.0 - CORRIGIDO

🔹 CORREÇÕES:
  ✅ tick_rule_sum calculado CORRETAMENTE via diff de preços
  ✅ order_book_slope retorna 0.0 quando dados zerados (não dict vazio)
  ✅ flow_imbalance com logs em falhas
  ✅ volume_sma_ratio com cap em 500% (5x)
  ✅ Validação robusta de dados
  ✅ Flags de qualidade
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio

import numpy as np
import pandas as pd

# Import para correlações cross-asset
try:
    from cross_asset_correlations import get_cross_asset_features
except ImportError as e:
    get_cross_asset_features = None
    logging.warning(f"cross_asset_correlations indisponível: {e}")


# ===============================
# 🆕 Exceção customizada
# ===============================

class MLFeaturesError(Exception):
    """Levantada quando features não podem ser calculadas."""
    pass


# ===============================
# Helpers internos
# ===============================

def _get_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Retorna série de preços: prioridade 'close' > 'p' > 'price' > 'c'.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    for col in ("close", "p", "price", "c"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return s.astype(float)

    return pd.Series(dtype=float)


def _ensure_close_df(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame com coluna 'close'."""
    s = _get_price_series(df)
    if s.empty:
        return pd.DataFrame(columns=["close"])
    return pd.DataFrame({"close": s.values})


def _ensure_volume_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DF com 'q' (volume), 'm' (direção), 'p' (preço se disponível).
    
    🔹 CORREÇÃO v2.0.0:
      - Infere 'm' corretamente via tick rule
      - Valida que tem dados suficientes
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["q", "m", "p"])

    out = pd.DataFrame(index=df.index)
    
    # Volume
    if "q" in df.columns:
        out["q"] = pd.to_numeric(df["q"], errors="coerce")
    else:
        out["q"] = np.nan

    # Preço
    price_col = None
    for col in ("p", "close", "price", "c"):
        if col in df.columns:
            price_col = col
            break
    
    if price_col:
        out["p"] = pd.to_numeric(df[price_col], errors="coerce")
    else:
        out["p"] = np.nan

    # 🆕 Direção agressora 'm' (Binance: True=buyer maker, False=buyer taker)
    if "m" in df.columns:
        m = df["m"]
        if m.dtype == bool:
            out["m"] = m
        else:
            # Normaliza strings/ints para bool
            out["m"] = m.astype(str).str.strip().str.lower().isin(
                ["true", "1", "sell", "seller", "yes"]
            )
    else:
        # 🆕 INFERIR 'm' via tick rule
        # Regra: preço cai/igual → buyer maker (True), preço sobe → buyer taker (False)
        if out["p"].notna().sum() >= 2:
            diff = out["p"].diff()
            # diff <= 0 → buyer maker (True)
            # diff > 0 → buyer taker (False)
            inferred_m = diff.apply(
                lambda x: True if (pd.notna(x) and x <= 0) else False
            )
            inferred_m.iloc[0] = False  # Primeira linha = buyer taker (neutro)
            out["m"] = inferred_m.astype(bool)
        else:
            out["m"] = False

    # Limpa linhas inválidas
    out["q"] = pd.to_numeric(out["q"], errors="coerce")
    out = out.dropna(subset=["q"])
    out = out[out["q"] > 0]

    return out.reset_index(drop=True)


# ===============================
# Price features
# ===============================

def calculate_price_features(
    df_or_series: pd.DataFrame | pd.Series,
    lookback_windows: List[int]
) -> Dict[str, Any]:
    """
    Calcula retornos, volatilidade e momentum.
    
    🔹 SEM MUDANÇAS (código original está OK).
    """
    features: Dict[str, Any] = {}

    # Normaliza para Series
    if isinstance(df_or_series, pd.Series):
        prices = pd.to_numeric(df_or_series, errors="coerce").dropna().astype(float)
    else:
        close_df = _ensure_close_df(df_or_series)
        if close_df.empty or "close" not in close_df:
            return features
        prices = close_df["close"].astype(float)

    if prices.empty:
        return features

    prices = prices.reset_index(drop=True)

    # Cálculo por janela
    computed_returns: List[float] = []
    
    for window in lookback_windows:
        key_r = f"returns_{window}"
        key_v = f"volatility_{window}"

        if len(prices) >= window + 1:
            current_close = float(prices.iloc[-1])
            past_close = float(prices.iloc[-(window + 1)])
            ret = (current_close - past_close) / past_close if past_close != 0 else 0.0
            features[key_r] = float(ret)
            computed_returns.append(ret)

            # Volatilidade realizada
            p_slice = prices.iloc[-(window + 1):].values.astype(float)
            prev = p_slice[:-1]
            nxt = p_slice[1:]
            valid_mask = (prev > 0) & (nxt > 0)
            
            if valid_mask.sum() > 0:
                lr = np.log(nxt[valid_mask] / prev[valid_mask])
                vol = float(np.std(lr, ddof=0))
            else:
                vol = 0.0
            
            features[key_v] = vol
        else:
            features[key_r] = 0.0
            features[key_v] = 0.0

    # Momentum score
    if computed_returns:
        last_window = max(lookback_windows)
        r_long = features.get(f"returns_{last_window}", 0.0)
        mu_ret = float(np.mean(computed_returns))
        sigma_ret = float(np.std(computed_returns, ddof=0))
        momentum = (r_long - mu_ret) / sigma_ret if sigma_ret > 0 else 0.0
        features["momentum_score"] = float(momentum)

    return features


# ===============================
# Volume features
# ===============================

def calculate_volume_features(
    df: pd.DataFrame, 
    volume_ma_window: int
) -> Dict[str, Any]:
    """
    Calcula features de volume.
    
    🔹 CORREÇÃO v2.0.0:
      - volume_sma_ratio com cap em 500%
      - Validação robusta
    """
    features: Dict[str, Any] = {}

    vol_df = _ensure_volume_df(df)
    if vol_df.empty or "q" not in vol_df.columns:
        return features

    vols = vol_df["q"].astype(float)
    current_vol = float(vols.sum())

    # 🆕 SMA de volume com validação
    if len(vols) >= volume_ma_window:
        sma_vol = float(
            vols.rolling(
                window=volume_ma_window, 
                min_periods=volume_ma_window
            ).sum().iloc[-1]
        )
    else:
        # 🆕 Se janela pequena, usa média simples
        sma_vol = float(vols.mean() * volume_ma_window)
    
    sma_vol = sma_vol if sma_vol > 0 else 1e-9
    
    # 🆕 CAP em 500% (5x)
    ratio = current_vol / sma_vol
    ratio = min(ratio, 5.0)  # Cap
    features["volume_sma_ratio"] = float(ratio)

    # Volume momentum
    half = len(vols) // 2
    vol_first = float(vols.iloc[:half].sum()) if half > 0 else 0.0
    vol_second = float(vols.iloc[half:].sum())
    
    if vol_first > 0:
        features["volume_momentum"] = float((vol_second - vol_first) / vol_first)
    else:
        features["volume_momentum"] = float(vol_second)

    # Buy/Sell pressure
    sell_mask = vol_df["m"].astype(bool)
    buy_mask = ~sell_mask
    buy_vol = float(vols[buy_mask].sum())
    sell_vol = float(vols[sell_mask].sum())
    total = buy_vol + sell_vol
    
    features["buy_sell_pressure"] = float((buy_vol - sell_vol) / total) if total > 0 else 0.0

    # Liquidity gradient
    if vol_first > 0:
        features["liquidity_gradient"] = float((vol_second - vol_first) / vol_first)
    else:
        features["liquidity_gradient"] = float(vol_second)

    return features


# ===============================
# 🆕 Microstructure features CORRIGIDO
# ===============================

def calculate_microstructure_features(
    orderbook_data: Dict[str, Any],
    flow_metrics: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,  # 🆕 Para tick_rule_sum
) -> Dict[str, Any]:
    """
    Microestrutura com cálculos CORRETOS.
    
    🔹 CORREÇÕES v2.0.0:
      - tick_rule_sum calculado via diff de preços (não spread!)
      - order_book_slope retorna 0.0 explicitamente quando dados zerados
      - flow_imbalance com logs em falhas
      - Validação robusta
    """
    features: Dict[str, Any] = {}

    # ========================================
    # 1. 🆕 ORDER BOOK SLOPE (CORRIGIDO)
    # ========================================
    try:
        ob_depth = (orderbook_data or {}).get("order_book_depth", {})
        l1 = ob_depth.get("L1", {}) if isinstance(ob_depth, dict) else {}
        l10 = ob_depth.get("L10", {}) if isinstance(ob_depth, dict) else {}

        # 🆕 Valida que tem dados
        if l1 and l10 and isinstance(l1, dict) and isinstance(l10, dict):
            bid1 = float(l1.get("bids", 0.0) or 0.0)
            ask1 = float(l1.get("asks", 0.0) or 0.0)
            bid10 = float(l10.get("bids", 0.0) or 0.0)
            ask10 = float(l10.get("asks", 0.0) or 0.0)
            
            # 🆕 Valida que não são todos zeros
            if (bid1 + ask1 + bid10 + ask10) > 0:
                denom = (bid10 + ask10) if (bid10 + ask10) != 0 else 1e-9
                slope = ((bid1 + ask1) - (bid10 + ask10)) / denom
                features["order_book_slope"] = float(slope)
            else:
                features["order_book_slope"] = 0.0
        else:
            # 🆕 RETORNA 0.0 EXPLICITAMENTE
            features["order_book_slope"] = 0.0
            
    except Exception as e:
        logging.warning(f"⚠️ Erro ao calcular order_book_slope: {e}")
        features["order_book_slope"] = 0.0

    # ========================================
    # 2. 🆕 FLOW IMBALANCE (COM LOGS)
    # ========================================
    try:
        fm = flow_metrics or {}
        
        # Tenta pegar de order_flow primeiro
        if "order_flow" in fm and isinstance(fm["order_flow"], dict):
            of = fm["order_flow"]
            
            # 🆕 Usa flow_imbalance se disponível
            if "flow_imbalance" in of:
                features["flow_imbalance"] = float(of["flow_imbalance"])
            else:
                # Calcula via net_flow
                net_flow = float(of.get("net_flow_1m", 0.0) or 0.0)
                bsr = float(of.get("buy_sell_ratio", 0.0) or 0.0)
                
                # Aproximação: total ≈ |net| / max(ratio - 1, eps) para ratio > 1
                if bsr > 1:
                    total_flow = abs(net_flow) / max(bsr - 1, 1e-9)
                elif bsr > 0:
                    total_flow = abs(net_flow) / max(1 - bsr, 1e-9)
                else:
                    total_flow = abs(net_flow)
                
                features["flow_imbalance"] = float(net_flow / total_flow) if total_flow > 0 else 0.0
        else:
            # Fallback: sector_flow
            sector_flow = fm.get("sector_flow", {}) if isinstance(fm, dict) else {}
            buy_total = float(sum(
                float(v.get("buy", 0.0) or 0.0) 
                for v in sector_flow.values()
            ))
            sell_total = float(sum(
                float(v.get("sell", 0.0) or 0.0) 
                for v in sector_flow.values()
            ))
            net_flow = buy_total - sell_total
            total_flow = buy_total + sell_total
            
            features["flow_imbalance"] = float(net_flow / total_flow) if total_flow > 0 else 0.0
            
    except Exception as e:
        logging.error(f"❌ Erro ao calcular flow_imbalance: {e}")
        features["flow_imbalance"] = 0.0

    # ========================================
    # 3. 🆕 TICK RULE SUM (CORRIGIDO!)
    # ========================================
    try:
        # 🆕 MÉTODO 1: Usa dados de flow_metrics se disponível
        if flow_metrics and "order_flow" in flow_metrics:
            tick_rule = flow_metrics["order_flow"].get("tick_rule_sum")
            
            if tick_rule is not None:
                features["tick_rule_sum"] = float(tick_rule)
            else:
                # Calcula se tem DataFrame
                if df is not None and not df.empty:
                    features["tick_rule_sum"] = _calculate_tick_rule_sum(df)
                else:
                    features["tick_rule_sum"] = 0.0
        else:
            # 🆕 MÉTODO 2: Calcula do DataFrame de trades
            if df is not None and not df.empty:
                features["tick_rule_sum"] = _calculate_tick_rule_sum(df)
            else:
                features["tick_rule_sum"] = 0.0
                
    except Exception as e:
        logging.error(f"❌ Erro ao calcular tick_rule_sum: {e}")
        features["tick_rule_sum"] = 0.0

    # ========================================
    # 4. TRADE INTENSITY (OK)
    # ========================================
    try:
        bursts = (flow_metrics or {}).get("bursts", {}) or {}
        count = int(bursts.get("count", 0) or 0)
        window_ms = int(
            (flow_metrics or {}).get("metadata", {}).get("burst_window_ms", 1000) or 1000
        )
        intensity = count / (window_ms / 1000.0) if window_ms > 0 else 0.0
        features["trade_intensity"] = float(intensity)
    except Exception as e:
        logging.warning(f"⚠️ Erro ao calcular trade_intensity: {e}")
        features["trade_intensity"] = 0.0

    return features


def _calculate_tick_rule_sum(df: pd.DataFrame) -> float:
    """
    🆕 CALCULA TICK RULE SUM CORRETAMENTE.
    
    Tick rule:
      - Uptick (preço sobe): +1
      - Downtick (preço cai): -1
      - Mesmo preço: 0
    
    Retorna soma de todos os ticks.
    """
    try:
        # Pega série de preços
        prices = _get_price_series(df)
        
        if len(prices) < 2:
            return 0.0
        
        # Calcula diffs
        diff = prices.diff()
        
        # Aplica tick rule
        tick_rule = diff.apply(lambda x: 
            +1.0 if (pd.notna(x) and x > 0) else
            (-1.0 if (pd.notna(x) and x < 0) else 0.0)
        )
        
        # Soma total
        tick_rule_sum = float(tick_rule.sum())
        
        return tick_rule_sum
        
    except Exception as e:
        logging.error(f"❌ Erro em _calculate_tick_rule_sum: {e}")
        return 0.0


def calculate_cross_asset_features(
    symbol: str,
    now_utc: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    🆕 Calcula features de correlação cross-asset para BTCUSDT.
    
    Foca especialmente em:
    - BTC x DXY (inversa) - correlação esperada negativa
    - BTC x NDX - correlação com mercado tech
    - BTC x ETH - correlação entre principais cryptos
    
    Args:
        symbol: Símbolo do ativo (ex: BTCUSDT)
        now_utc: Timestamp atual em UTC (opcional)
        
    Returns:
        Dict com features de correlação cross-asset
    """
    features: Dict[str, Any] = {}
    
    try:
        # Só calcula para BTCUSDT conforme especificação
        if symbol != "BTCUSDT":
            logging.debug(f"Cross-asset features apenas para BTCUSDT, ignorando {symbol}")
            return features
        
        # Verifica se módulo está disponível
        if get_cross_asset_features is None:
            logging.warning("Módulo cross_asset_correlations não disponível")
            return features
        
        # Define timestamp atual se não fornecido
        if now_utc is None:
            now_utc = datetime.utcnow()
        
        logging.debug("Calculando features cross-asset para BTCUSDT...")
        
        # Calcula correlações usando a função principal unificada
        try:
            correlations = get_cross_asset_features(now_utc)
            
            if correlations.get("status") != "ok":
                logging.warning(f"Falha ao calcular correlações: {correlations.get('error')}")
                return features
            
            # Extrai e mapeia as features conforme especificação
            # Features BTC x ETH (dados de 1h)
            features["btc_eth_corr_7d"] = correlations.get("btc_eth_corr_7d", float("nan"))
            features["btc_eth_corr_30d"] = correlations.get("btc_eth_corr_30d", float("nan"))
            
            # Features BTC x DXY (dados diários - foco especial)
            features["btc_dxy_corr_30d"] = correlations.get("btc_dxy_corr_30d", float("nan"))
            features["btc_dxy_corr_90d"] = correlations.get("btc_dxy_corr_90d", float("nan"))
            
            # Features BTC x NDX (dados diários - secundário)
            features["btc_ndx_corr_30d"] = correlations.get("btc_ndx_corr_30d", None)
            
            # Features de retornos DXY
            features["dxy_return_5d"] = correlations.get("dxy_return_5d", float("nan"))
            features["dxy_return_20d"] = correlations.get("dxy_return_20d", float("nan"))
            
            # Features derivadas para análise adicional
            dxy_30d = features["btc_dxy_corr_30d"]
            dxy_90d = features["btc_dxy_corr_90d"]
            
            if pd.notna(dxy_30d) and pd.notna(dxy_90d):
                # Estabilidade da correlação BTC x DXY
                features["btc_dxy_correlation_stability"] = float(abs(dxy_30d - dxy_90d))
                
                # Força da correlação inversa (média)
                features["btc_dxy_inverse_strength"] = float(abs((dxy_30d + dxy_90d) / 2))
            else:
                features["btc_dxy_correlation_stability"] = float("nan")
                features["btc_dxy_inverse_strength"] = float("nan")
            
            # Força do USD (DXY)
            dxy_5d = features["dxy_return_5d"]
            dxy_20d = features["dxy_return_20d"]
            
            if pd.notna(dxy_5d) and pd.notna(dxy_20d):
                features["dxy_momentum"] = float(dxy_20d - dxy_5d)  # Aceleração
            else:
                features["dxy_momentum"] = float("nan")
            
            # ========== 🆕 NOVAS MÉTRICAS ENHANCED ==========
            
            # VIX Metrics (Fear Index)
            features["vix_current"] = correlations.get("vix_current", float("nan"))
            features["vix_change_1d"] = correlations.get("vix_change_1d", float("nan"))
            features["btc_vix_corr_30d"] = correlations.get("btc_vix_corr_30d", float("nan"))
            
            # Treasury Yields
            features["us10y_yield"] = correlations.get("us10y_yield", float("nan"))
            features["us10y_change_1d"] = correlations.get("us10y_change_1d", float("nan"))
            features["us2y_yield"] = correlations.get("us2y_yield", float("nan"))
            features["us2y_change_1d"] = correlations.get("us2y_change_1d", float("nan"))
            features["btc_yields_corr_30d"] = correlations.get("btc_yields_corr_30d", float("nan"))
            
            # Crypto Dominance
            features["btc_dominance"] = correlations.get("btc_dominance", float("nan"))
            features["btc_dominance_change_7d"] = correlations.get("btc_dominance_change_7d", 0.0)
            features["eth_dominance"] = correlations.get("eth_dominance", float("nan"))
            features["usdt_dominance"] = correlations.get("usdt_dominance", float("nan"))
            
            # Commodities
            features["gold_price"] = correlations.get("gold_price", float("nan"))
            features["gold_change_1d"] = correlations.get("gold_change_1d", float("nan"))
            features["btc_gold_corr_30d"] = correlations.get("btc_gold_corr_30d", float("nan"))
            features["oil_price"] = correlations.get("oil_price", float("nan"))
            features["oil_change_1d"] = correlations.get("oil_change_1d", float("nan"))
            features["btc_oil_corr_30d"] = correlations.get("btc_oil_corr_30d", float("nan"))
            
            # Regime Detection
            features["macro_regime"] = correlations.get("macro_regime", "UNKNOWN")
            features["correlation_regime"] = correlations.get("correlation_regime", "UNKNOWN")
            
            logging.debug(f"✅ Enhanced cross-asset features calculadas: {len(features)} features")
            
            logging.debug(f"✅ Features cross-asset calculadas: {len(features)} features")
            
        except Exception as e:
            logging.error(f"❌ Erro ao calcular correlações: {e}")
            features["error"] = str(e)
        
    except Exception as e:
        logging.error(f"❌ Erro ao calcular cross-asset features: {e}")
        features["error"] = str(e)
    
    return features


# ===============================
# 🆕 Orquestração CORRIGIDA
# ===============================

# Contador para frequência de correlações (OTIMIZAÇÃO)
_correlation_call_count = 0


def generate_ml_features(
    df: pd.DataFrame,
    orderbook_data: Dict[str, Any],
    flow_metrics: Dict[str, Any],
    lookback_windows: List[int] = [1, 5, 15],
    volume_ma_window: int = 20,
    symbol: str = "BTCUSDT",  # 🆕 Adiciona símbolo para cross-asset features
) -> Dict[str, Any]:
    """
    Gera features consolidadas.
    
    🔹 CORREÇÕES v2.0.0:
      - Passa df para microstructure (para tick_rule_sum)
      - Validação de dados
      - Logs em falhas
      - Flags de qualidade
    
    Returns:
        Dict com:
          - price_features
          - volume_features
          - microstructure
          - data_quality (🆕)
    """
    issues: List[str] = []
    
    # -------- Price --------
    try:
        price_df = _ensure_close_df(df)
        price_feats = calculate_price_features(price_df, lookback_windows)
    except Exception as e:
        logging.error(f"❌ Erro ao calcular price_features: {e}")
        price_feats = {}
        issues.append(f"price_features_error: {e}")

    # -------- Volume --------
    try:
        volume_df = _ensure_volume_df(df)
        volume_feats = calculate_volume_features(volume_df, volume_ma_window)
    except Exception as e:
        logging.error(f"❌ Erro ao calcular volume_features: {e}")
        volume_feats = {}
        issues.append(f"volume_features_error: {e}")

    # -------- Microestrutura --------
    try:
        # 🆕 Passa df para cálculo de tick_rule_sum
        micro_feats = calculate_microstructure_features(
            orderbook_data or {}, 
            flow_metrics or {},
            df=df  # 🆕
        )
    except Exception as e:
        logging.error(f"❌ Erro ao calcular microstructure: {e}")
        micro_feats = {}
        issues.append(f"microstructure_error: {e}")

    # 🆕 Cross-Asset Features (apenas para BTCUSDT)
    try:
        cross_asset_feats = calculate_cross_asset_features(symbol)
    except Exception as e:
        logging.error(f"❌ Erro ao calcular cross_asset_features: {e}")
        cross_asset_feats = {}
        issues.append(f"cross_asset_error: {e}")

    # 🆕 Qualidade de dados
    data_quality = {
        "has_price_features": len(price_feats) > 0,
        "has_volume_features": len(volume_feats) > 0,
        "has_microstructure": len(micro_feats) > 0,
        "has_cross_asset": len(cross_asset_feats) > 0,
        "issues": issues,
        "is_valid": len(issues) == 0,
    }

    return {
        "price_features": price_feats,
        "volume_features": volume_feats,
        "microstructure": micro_feats,
        "cross_asset": cross_asset_feats,  # 🆕 Cross-asset features
        "data_quality": data_quality,
    }


# ===============================
# 🆕 TESTE
# ===============================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    print("\n" + "="*80)
    print("🧪 TESTE DE ML_FEATURES v2.0.0")
    print("="*80 + "\n")
    
    # Dados de teste
    test_df = pd.DataFrame({
        'p': [50000, 50010, 50005, 50020, 50015],  # Preços com upticks/downticks
        'q': [1.0, 2.0, 1.5, 3.0, 2.5],
        'm': [False, True, True, False, True],
    })
    
    test_orderbook = {
        "order_book_depth": {
            "L1": {"bids": 100000, "asks": 95000},
            "L10": {"bids": 500000, "asks": 480000},
        },
        "spread_metrics": {"spread": 5.0},
    }
    
    test_flow = {
        "order_flow": {
            "net_flow_1m": -1500.0,
            "buy_sell_ratio": 0.8,
            "flow_imbalance": -0.11,
            "tick_rule_sum": -2.0,  # 2 downticks a mais que upticks
        },
        "bursts": {"count": 5},
        "metadata": {"burst_window_ms": 1000},
    }
    
    # Gera features
    features = generate_ml_features(
        df=test_df,
        orderbook_data=test_orderbook,
        flow_metrics=test_flow,
    )
    
    print("📊 PRICE FEATURES:")
    for k, v in features.get("price_features", {}).items():
        print(f"  {k}: {v:.6f}")
    
    print("\n📊 VOLUME FEATURES:")
    for k, v in features.get("volume_features", {}).items():
        print(f"  {k}: {v:.6f}")
    
    print("\n📊 MICROSTRUCTURE:")
    micro = features.get("microstructure", {})
    print(f"  order_book_slope: {micro.get('order_book_slope', 0):.6f}")
    print(f"  flow_imbalance: {micro.get('flow_imbalance', 0):.6f}")
    print(f"  tick_rule_sum: {micro.get('tick_rule_sum', 0):.6f}  ← CORRIGIDO!")
    print(f"  trade_intensity: {micro.get('trade_intensity', 0):.6f}")
    
    print("\n📊 DATA QUALITY:")
    dq = features.get("data_quality", {})
    print(f"  is_valid: {dq.get('is_valid')}")
    print(f"  issues: {dq.get('issues')}")
    
    print("\n" + "="*80)
    print("✅ TESTE CONCLUÍDO")
    print("="*80 + "\n")
    
    # Teste de tick_rule_sum isolado
    print("🧪 TESTE ISOLADO: tick_rule_sum")
    tick_sum = _calculate_tick_rule_sum(test_df)
    print(f"  Preços: {test_df['p'].tolist()}")
    print(f"  Diffs: {test_df['p'].diff().tolist()}")
    print(f"  tick_rule_sum: {tick_sum:.2f}")
    print(f"  Esperado: +1 (50010>50000) -1 (50005<50010) +1 (50020>50005) -1 (50015<50020) = 0")
    print()