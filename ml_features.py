"""
ml_features.py

Gera features de ML a partir de dados de preço, volume e microestrutura.
Compatível com janelas de trades (colunas típicas: p, q, m, T) OU
com candles (colunas: close/high/low).

Principais pontos:
- Não depende estritamente de 'close': usa 'p' se 'close' não existir.
- Infere 'm' (direção agressora) pelo tick-rule se estiver ausente.
- Robusto a dataframes vazios/incompletos (retorna dicts vazios quando aplicável).
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


# ===============================
# Helpers internos
# ===============================

def _get_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Retorna uma série de preços para cálculo de features:
    prioridade: 'close' > 'p' > 'price' > 'c'.
    Sempre retorna float e sem NaN (dropna), índice preservado.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    for col in ("close", "p", "price", "c"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return s.astype(float)

    return pd.Series(dtype=float)


def _ensure_close_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna um DataFrame com UMA coluna 'close', derivando de 'p' ou similares
    quando 'close' não existir. Se nada existir, retorna DF vazio.
    """
    s = _get_price_series(df)
    if s.empty:
        return pd.DataFrame(columns=["close"])
    return pd.DataFrame({"close": s.values})


def _ensure_volume_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna DF contendo no mínimo 'q' (volume) e, se possível, 'm' (direção).
    - Se 'm' não existir, tenta inferir via tick-rule usando diffs de preço 'p'.
    - Mantém 'p' se existir (não é obrigatório).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["q", "m"])

    out = pd.DataFrame(index=df.index)
    # volume
    if "q" in df.columns:
        out["q"] = pd.to_numeric(df["q"], errors="coerce")
    else:
        # sem volume não há como computar as features de volume
        out["q"] = np.nan

    # preço (opcional, útil para inferir m)
    price_col = None
    for col in ("p", "close", "price", "c"):
        if col in df.columns:
            price_col = col
            break
    if price_col:
        out["p"] = pd.to_numeric(df[price_col], errors="coerce")
    else:
        out["p"] = np.nan

    # direção agressora: True=sell, False=buy
    if "m" in df.columns:
        m = df["m"]
        # normaliza para boolean com fallback
        if m.dtype == bool:
            out["m"] = m
        else:
            out["m"] = m.astype(str).str.strip().str.lower().isin(["true", "1", "sell", "seller"])
    else:
        # inferir via tick-rule: preço sobe => buyer (False), cai/igual => seller (True)
        # se não houver preço, define False (neutro/comprador)
        if out["p"].notna().sum() >= 2:
            diff = out["p"].diff()
            # primeira linha sem diff: considera buyer (False)
            inferred_m = diff.apply(lambda x: True if (pd.notna(x) and x <= 0) else False)
            inferred_m.iloc[0] = False
            out["m"] = inferred_m.astype(bool)
        else:
            out["m"] = False

    # limpa linhas inválidas (sem volume ou volume <= 0)
    out["q"] = pd.to_numeric(out["q"], errors="coerce")
    out = out.dropna(subset=["q"])
    out = out[out["q"] > 0]

    return out.reset_index(drop=True)


# ===============================
# Price features
# ===============================

def calculate_price_features(df_or_series: pd.DataFrame | pd.Series,
                             lookback_windows: List[int]) -> Dict[str, Any]:
    """
    Calcula retornos e volatilidade realizada (log-returns) para janelas em 'rows'.

    Aceita:
      - DataFrame com 'close' e/ou 'p'
      - Series de preços

    Retorna chaves como: returns_1, volatility_1, ..., momentum_score.
    """
    features: Dict[str, Any] = {}

    # Normaliza para Series de preço
    if isinstance(df_or_series, pd.Series):
        prices = pd.to_numeric(df_or_series, errors="coerce").dropna().astype(float)
    else:
        close_df = _ensure_close_df(df_or_series)
        if close_df.empty or "close" not in close_df:
            return features
        prices = close_df["close"].astype(float)

    if prices.empty:
        return features

    # Garante ordenação temporal se houver índice com T
    try:
        prices = prices.reset_index(drop=True)
    except Exception:
        pass

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

            # Realized volatility (std de log-returns) na subjanela (últimos window+1 pontos)
            p_slice = prices.iloc[-(window + 1):].values.astype(float)
            # evita divisões por zero
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

    # Momentum score: z-score do maior window vs média/DesvPad dos retornos calculados
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

def calculate_volume_features(df: pd.DataFrame, volume_ma_window: int) -> Dict[str, Any]:
    """
    Calcula features de volume a partir de trades:
      - volume_sma_ratio (volume atual / SMA da janela)
      - volume_momentum (segunda metade vs primeira)
      - buy_sell_pressure ((buy - sell) / total)
      - liquidity_gradient (igual ao momentum de volume)
    'm': True=sell, False=buy (como na Binance aggTrades).
    """
    features: Dict[str, Any] = {}

    vol_df = _ensure_volume_df(df)
    if vol_df.empty or "q" not in vol_df.columns:
        return features

    vols = vol_df["q"].astype(float)
    current_vol = float(vols.sum())

    # SMA de volume sobre a própria janela de trades (rolling por count de trades)
    if len(vols) >= volume_ma_window:
        sma_vol = float(
            vols.rolling(window=volume_ma_window, min_periods=volume_ma_window).sum().iloc[-1]
        )
    else:
        sma_vol = float(vols.sum())
    sma_vol = sma_vol if sma_vol > 0 else 1e-9
    features["volume_sma_ratio"] = float(current_vol / sma_vol)

    # Momentum de volume: 2ª metade vs 1ª metade
    half = len(vols) // 2
    vol_first = float(vols.iloc[:half].sum()) if half > 0 else 0.0
    vol_second = float(vols.iloc[half:].sum())
    if vol_first > 0:
        features["volume_momentum"] = float((vol_second - vol_first) / vol_first)
    else:
        features["volume_momentum"] = float(vol_second)

    # Buy/Sell pressure via 'm'
    sell_mask = vol_df["m"].astype(bool)
    buy_mask = ~sell_mask
    buy_vol = float(vols[buy_mask].sum())
    sell_vol = float(vols[sell_mask].sum())
    total = buy_vol + sell_vol
    features["buy_sell_pressure"] = float((buy_vol - sell_vol) / total) if total > 0 else 0.0

    # Liquidity gradient (mesmo conceito do momentum)
    if vol_first > 0:
        features["liquidity_gradient"] = float((vol_second - vol_first) / vol_first)
    else:
        features["liquidity_gradient"] = float(vol_second)

    return features


# ===============================
# Microstructure features
# ===============================

def calculate_microstructure_features(orderbook_data: Dict[str, Any],
                                      flow_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Microestrutura a partir de book/fluxo:
      - order_book_slope
      - flow_imbalance
      - tick_rule_sum (aproximação via spread sign)
      - trade_intensity (contagem de 'bursts' / janela em segundos)
    """
    features: Dict[str, Any] = {}

    # Order book slope: diferença L1 vs L10 normalizada pelo depth de L10
    try:
        ob_depth = (orderbook_data or {}).get("order_book_depth", {})
        l1 = ob_depth.get("L1", {}) if isinstance(ob_depth, dict) else {}
        l10 = ob_depth.get("L10", {}) if isinstance(ob_depth, dict) else {}

        if l1 and l10:
            bid1 = float(l1.get("bids", 0.0) or 0.0)
            ask1 = float(l1.get("asks", 0.0) or 0.0)
            bid10 = float(l10.get("bids", 0.0) or 0.0)
            ask10 = float(l10.get("asks", 0.0) or 0.0)
            denom = (bid10 + ask10) if (bid10 + ask10) != 0 else 1e-9
            slope = ((bid1 + ask1) - (bid10 + ask10)) / denom
            features["order_book_slope"] = float(slope)
    except Exception:
        features["order_book_slope"] = 0.0

    # Flow imbalance
    try:
        fm = flow_metrics or {}
        if "order_flow" in fm and isinstance(fm["order_flow"], dict):
            f = fm["order_flow"]
            net_flow = float(f.get("net_flow_1m", 0.0) or 0.0)
            bsr = float(f.get("buy_sell_ratio", 0.0) or 0.0)
            # aproximação: total ≈ |net| / max(ratio, eps)
            total_flow = abs(net_flow) / max(bsr, 1e-9)
        else:
            sector_flow = fm.get("sector_flow", {}) if isinstance(fm, dict) else {}
            buy_total = float(sum(float(v.get("buy", 0.0) or 0.0) for v in sector_flow.values()))
            sell_total = float(sum(float(v.get("sell", 0.0) or 0.0) for v in sector_flow.values()))
            net_flow = buy_total - sell_total
            total_flow = buy_total + sell_total
        features["flow_imbalance"] = float(net_flow / total_flow) if total_flow > 0 else 0.0
    except Exception:
        features["flow_imbalance"] = 0.0

    # Tick rule sum (placeholder com sinal do spread)
    try:
        spread = (orderbook_data or {}).get("spread_metrics", {}).get("spread", 0.0)
        if spread is None:
            features["tick_rule_sum"] = 0.0
        else:
            features["tick_rule_sum"] = float(np.sign(float(spread)))
    except Exception:
        features["tick_rule_sum"] = 0.0

    # Trade intensity (bursts por segundo)
    try:
        bursts = (flow_metrics or {}).get("bursts", {}) or {}
        count = int(bursts.get("count", 0) or 0)
        window_ms = int((flow_metrics or {}).get("metadata", {}).get("burst_window_ms", 1000) or 1000)
        intensity = count / (window_ms / 1000.0) if window_ms > 0 else 0.0
        features["trade_intensity"] = float(intensity)
    except Exception:
        features["trade_intensity"] = 0.0

    return features


# ===============================
# Orquestração
# ===============================

def generate_ml_features(
    df: pd.DataFrame,
    orderbook_data: Dict[str, Any],
    flow_metrics: Dict[str, Any],
    lookback_windows: List[int] = [1, 5, 15],
    volume_ma_window: int = 20,
) -> Dict[str, Any]:
    """
    Gera features consolidadas:
      - price_features: retornos/volatilidade/momentum
      - volume_features: razões, momentum, pressão, gradiente
      - microstructure: book/fluxo/tick-rule/intensidade
    """
    # -------- Price --------
    try:
        price_df = _ensure_close_df(df)
        price_feats = calculate_price_features(price_df, lookback_windows)
    except Exception:
        price_feats = {}

    # -------- Volume --------
    try:
        volume_df = _ensure_volume_df(df)
        volume_feats = calculate_volume_features(volume_df, volume_ma_window)
    except Exception:
        volume_feats = {}

    # -------- Microestrutura --------
    try:
        micro_feats = calculate_microstructure_features(orderbook_data or {}, flow_metrics or {})
    except Exception:
        micro_feats = {}

    return {
        "price_features": price_feats,
        "volume_features": volume_feats,
        "microstructure": micro_feats,
    }
