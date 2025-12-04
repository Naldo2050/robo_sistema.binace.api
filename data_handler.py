# data_handler.py - REFATORADO v2.0.0 - ELIMINAÇÃO DE DUPLICIDADES
"""
Data Handler com lógica unificada (NumPy como fonte da verdade).

🔹 REFATORAÇÃO v2.0.0:
  ✅ Eliminada duplicidade Pandas vs NumPy/Listas
  ✅ NumPy/Lista = "fonte da verdade" (implementação principal)
  ✅ Pandas = wrapper (chama NumPy internamente quando possível)
  ✅ Princípio DRY: Uma única implementação por conceito
  ✅ Compatibilidade 100% mantida com código existente
  ✅ Performance otimizada para tempo real

🔹 HIERARQUIA DE IMPLEMENTAÇÃO:
  📌 Camada 1 (Core - NumPy/Escalar):
     - _normalize_m_value() → scalar
     - infer_or_fill_m_array() → array
     - _compute_absorption_scalar() → scalar
     - _compute_intra_candle_metrics_array() → array
     - _static_volume_profile_from_arrays() → array
     - _compute_dwell_time_array() → array
     - _compute_trade_speed_array() → array

  📌 Camada 2 (Pandas Wrappers):
     - _normalize_m_column() → chama _normalize_m_value()
     - infer_or_fill_m() → chama infer_or_fill_m_array()
     - calcular_delta() → usa infer_or_fill_m()
     - detectar_absorcao() → chama _compute_absorption_scalar()
     - calcular_metricas_intra_candle() → chama _compute_intra_candle_metrics_array()
     - calcular_volume_profile() → chama _static_volume_profile_from_arrays()
     - calcular_dwell_time() → chama _compute_dwell_time_array()
     - calcular_trade_speed() → chama _compute_trade_speed_array()

  📌 Camada 3 (Eventos - Tempo Real):
     - create_absorption_event() → usa funções Core diretamente
     - create_exhaustion_event() → usa funções Core diretamente
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
import hashlib
from flow_analyzer import FlowAnalyzer, _guard_absorcao  # usado para recalcular rótulo e validação
from typing import Dict, Any, List, Optional, Tuple
import math

import config

# 🔹 TIME MANAGER (injeção recomendada)
from time_manager import TimeManager

# 🔹 VOLUME PROFILE DINÂMICO
from dynamic_volume_profile import DynamicVolumeProfile

NY_TZ = ZoneInfo("America/New_York")

SCHEMA_VERSION = "1.1.0"


# ===============================
# 📌 CAMADA 1: CORE (NumPy/Escalar) - FONTE DA VERDADE
# ===============================

def _normalize_m_value(x: Any) -> Optional[bool]:
    """
    ✅ CORE: Normalização escalar de 'm' (taker side).

    Semântica interna:
      - True  => agressor VENDEDOR  (taker SELL)
      - False => agressor COMPRADOR (taker BUY)

    Compatível com Binance aggTrade:
      - m = True  => BUYER IS MAKER  → taker SELL
      - m = False => BUYER IS TAKER  → taker BUY

    Aceita também rótulos textuais:
      - "SELL"/"ask"/"s"/"seller"/"yes" -> True
      - "BUY"/"bid"/"b"/"buyer"/"no"    -> False

    Returns:
        True / False / None (None = NA)
    """
    if x is None:
        return None

    # Trata NaN/infinito
    if isinstance(x, (float, np.floating)) and (not math.isfinite(x)):
        return None

    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if isinstance(x, (int, np.integer)):
        if x == 1:
            return True
        if x == 0:
            return False
        return None

    if isinstance(x, str):
        t = x.strip().lower()
        if t in {"true", "t", "1", "sell", "ask", "s", "seller", "yes"}:
            return True
        if t in {"false", "f", "0", "buy", "bid", "b", "buyer", "no"}:
            return False
        return None

    return None


def infer_or_fill_m_array(
    prices: np.ndarray,
    raw_m: Optional[List[Any]] = None,
    prev_price: Optional[float] = None,
    prev_m: Optional[bool] = None,
) -> List[bool]:
    """
    ✅ CORE: Inferência de 'm' com tick-rule para arrays.

    - prices: array de preços (ordenados por tempo)
    - raw_m: lista com valores brutos de 'm' (compatível com Binance)
    - prev_price: último preço da janela anterior (para tick-rule no 1º trade)
    - prev_m: último 'm' da janela anterior (para herança correta)

    Semântica de `m` (taker side):
      - True  => agressor VENDEDOR  (taker SELL)
      - False => agressor COMPRADOR (taker BUY)

    Lógica:
      1. Se 'm' fornecido, usa (após normalizar com _normalize_m_value)
      2. Se não, usa tick-rule (subida=BUY, queda=SELL)
      3. Preços iguais herdam 'm' anterior
      4. Default final: False (BUY)

    Returns:
        Lista de bool (sem None)
    """
    n = int(len(prices))
    if n == 0:
        return []

    # Base: m explícito (normalizado via CORE)
    base: List[Optional[bool]] = [None] * n
    if raw_m is not None:
        for i in range(min(n, len(raw_m))):
            base[i] = _normalize_m_value(raw_m[i])  # ✅ USA CORE

    # Tick-rule (subida=BUY=False, queda=SELL=True)
    tick: List[Optional[bool]] = [None] * n
    for i in range(n):
        px = float(prices[i])
        if i == 0:
            if prev_price is None:
                continue
            d = px - float(prev_price)
        else:
            d = px - float(prices[i - 1])

        if d > 0:
            tick[i] = False  # BUY
        elif d < 0:
            tick[i] = True   # SELL
        # d == 0 => None (herança)

    # Combinação base + tick
    out: List[Optional[bool]] = [None] * n
    for i in range(n):
        out[i] = base[i] if base[i] is not None else tick[i]

    # Herança inicial com prev_m
    if prev_m is not None and out[0] is None:
        out[0] = bool(prev_m)

    # Forward-fill
    last_val: Optional[bool] = bool(prev_m) if prev_m is not None else None
    for i in range(n):
        if out[i] is not None:
            last_val = out[i]
            out[i] = last_val
        else:
            out[i] = last_val

    # Default False para NAs remanescentes
    final = [bool(x) if x is not None else False for x in out]
    return final


def _compute_absorption_scalar(
    o: float,
    h: float,
    l: float,
    c: float,
    delta_threshold: float,
    volume_buy_btc: float,
    volume_sell_btc: float,
) -> Tuple[float, bool, bool, float]:
    """
    ✅ CORE: Detecção de absorção para uma única janela/candle.

    Retorna:
      - delta_btc
      - absorcao_compra (bool)  [absorção de venda: agressão vendedora absorvida]
      - absorcao_venda (bool)   [absorção de compra: agressão compradora absorvida]
      - indice_absorcao (float)
    """
    try:
        delta_btc = float(volume_buy_btc - volume_sell_btc)

        candle_range = float(h - l)
        if candle_range <= 0 or not np.isfinite(candle_range):
            candle_range = 0.0001

        close_pos_compra = (c - l) / candle_range
        close_pos_venda = (h - c) / candle_range

        # Mesma lógica de detectar_absorcao (para 1 linha)
        absorcao_compra = (
            (delta_btc < -abs(delta_threshold)) and
            (c >= o * 0.998) and
            (close_pos_compra > 0.5)
        )
        absorcao_venda = (
            (delta_btc > abs(delta_threshold)) and
            (c <= o * 1.002) and
            (close_pos_venda > 0.5)
        )

        min_atr = c * 0.001  # 0.1%
        atr = max(candle_range, min_atr) if np.isfinite(min_atr) else candle_range
        indice_absorcao = (abs(delta_btc) / atr) if atr > 0 else 0.0
        if not np.isfinite(indice_absorcao):
            indice_absorcao = 0.0

        return delta_btc, absorcao_compra, absorcao_venda, indice_absorcao
    except Exception as e:
        logging.error(f"Erro em _compute_absorption_scalar: {e}")
        return 0.0, False, False, 0.0


def _compute_intra_candle_metrics_array(
    qtys: np.ndarray,
    m_flags: np.ndarray,
) -> dict:
    """
    ✅ CORE: Métricas intra-candle usando arrays.

    Args:
        qtys: Quantidades (BTC)
        m_flags: Flags de lado (True=SELL, False=BUY)

    Returns:
        Dict com métricas de reversão
    """
    try:
        if qtys.size == 0 or m_flags.size == 0:
            return {
                "delta_minimo": 0.0,
                "delta_maximo": 0.0,
                "delta_fechamento": 0.0,
                "reversao_desde_minimo": 0.0,
                "reversao_desde_maximo": 0.0,
            }

        trade_delta = np.where(~m_flags, qtys, -qtys)
        delta_cumulativo = np.cumsum(trade_delta)

        if delta_cumulativo.size == 0:
            return {
                "delta_minimo": 0.0,
                "delta_maximo": 0.0,
                "delta_fechamento": 0.0,
                "reversao_desde_minimo": 0.0,
                "reversao_desde_maximo": 0.0,
            }

        delta_min = float(np.nanmin(delta_cumulativo))
        delta_max = float(np.nanmax(delta_cumulativo))
        delta_close = float(delta_cumulativo[-1])

        if not np.isfinite(delta_min):
            delta_min = 0.0
        if not np.isfinite(delta_max):
            delta_max = 0.0
        if not np.isfinite(delta_close):
            delta_close = 0.0

        rev_buy = delta_close - delta_min
        rev_sell = delta_max - delta_close

        if not np.isfinite(rev_buy):
            rev_buy = 0.0
        if not np.isfinite(rev_sell):
            rev_sell = 0.0

        return {
            "delta_minimo": delta_min,
            "delta_maximo": delta_max,
            "delta_fechamento": delta_close,
            "reversao_desde_minimo": rev_buy,
            "reversao_desde_maximo": rev_sell,
        }
    except Exception as e:
        logging.error(f"Erro em _compute_intra_candle_metrics_array: {e}")
        return {
            "delta_minimo": 0.0,
            "delta_maximo": 0.0,
            "delta_fechamento": 0.0,
            "reversao_desde_minimo": 0.0,
            "reversao_desde_maximo": 0.0,
        }


def _static_volume_profile_from_arrays(
    prices: np.ndarray,
    qtys: np.ndarray,
    num_bins: int = 20,
) -> dict:
    """
    ✅ CORE: Volume Profile ESTÁTICO usando apenas NumPy.

    Returns:
        {"poc_price", "poc_volume", "poc_percentage"}
    """
    try:
        if prices.size == 0 or qtys.size == 0:
            return {"poc_price": 0.0, "poc_volume": 0.0, "poc_percentage": 0.0}

        mask = (
            np.isfinite(prices) & np.isfinite(qtys) &
            (prices > 0) & (qtys >= 0)
        )
        prices = prices[mask]
        qtys = qtys[mask]

        if prices.size == 0:
            return {"poc_price": 0.0, "poc_volume": 0.0, "poc_percentage": 0.0}

        min_p = float(np.min(prices))
        max_p = float(np.max(prices))

        total_volume = float(np.sum(qtys))
        if not np.isfinite(total_volume) or total_volume <= 0:
            return {"poc_price": min_p if np.isfinite(min_p) else 0.0,
                    "poc_volume": 0.0,
                    "poc_percentage": 0.0}

        if min_p == max_p or not np.isfinite(min_p) or not np.isfinite(max_p):
            return {
                "poc_price": min_p if np.isfinite(min_p) else 0.0,
                "poc_volume": total_volume,
                "poc_percentage": 100.0,
            }

        num_bins = max(1, int(num_bins))
        bin_width = (max_p - min_p) / num_bins
        if bin_width <= 0:
            return {
                "poc_price": min_p,
                "poc_volume": total_volume,
                "poc_percentage": 100.0,
            }

        volumes = np.zeros(num_bins, dtype=float)
        indices = ((prices - min_p) / bin_width).astype(int)
        indices = np.clip(indices, 0, num_bins - 1)

        for idx, vol in zip(indices, qtys):
            volumes[idx] += float(vol)

        if not np.any(np.isfinite(volumes)):
            return {"poc_price": 0.0, "poc_volume": 0.0, "poc_percentage": 0.0}

        poc_idx = int(np.nanargmax(volumes))
        poc_volume = float(volumes[poc_idx])
        poc_price = float(min_p + (poc_idx + 0.5) * bin_width)
        poc_percentage = (poc_volume / total_volume) * 100.0 if total_volume > 0 else 0.0

        return {
            "poc_price": poc_price,
            "poc_volume": poc_volume,
            "poc_percentage": poc_percentage,
        }
    except Exception as e:
        logging.error(f"Erro em _static_volume_profile_from_arrays: {e}")
        return {"poc_price": 0.0, "poc_volume": 0.0, "poc_percentage": 0.0}


def _compute_dwell_time_array(
    prices: np.ndarray,
    timestamps: np.ndarray,
    num_bins: int = 20,
) -> dict:
    """
    ✅ CORE: Dwell time usando apenas NumPy.

    Returns:
        {"dwell_price", "dwell_seconds", "dwell_location"}
    """
    try:
        if prices.size < 2 or timestamps.size < 2:
            return {"dwell_price": 0.0, "dwell_seconds": 0.0, "dwell_location": "N/A"}

        mask = (
            np.isfinite(prices) & np.isfinite(timestamps) &
            (prices > 0) & (timestamps > 0)
        )
        prices = prices[mask]
        timestamps = timestamps[mask]

        if prices.size < 2:
            return {"dwell_price": 0.0, "dwell_seconds": 0.0, "dwell_location": "N/A"}

        min_p = float(np.min(prices))
        max_p = float(np.max(prices))

        if (min_p == max_p) or not np.isfinite(min_p) or not np.isfinite(max_p):
            dwell_seconds = float((float(np.max(timestamps)) - float(np.min(timestamps))) / 1000.0)
            if not np.isfinite(dwell_seconds):
                dwell_seconds = 0.0
            return {
                "dwell_price": min_p if np.isfinite(min_p) else 0.0,
                "dwell_seconds": dwell_seconds,
                "dwell_location": "Mid",
            }

        num_bins = max(1, int(num_bins))
        bin_width = (max_p - min_p) / num_bins
        if bin_width <= 0:
            dwell_seconds = float((float(np.max(timestamps)) - float(np.min(timestamps))) / 1000.0)
            if not np.isfinite(dwell_seconds):
                dwell_seconds = 0.0
            return {
                "dwell_price": min_p,
                "dwell_seconds": dwell_seconds,
                "dwell_location": "Mid",
            }

        # Agrupa tempo por bin de preço
        bins_t_min = np.full(num_bins, np.inf, dtype=float)
        bins_t_max = np.full(num_bins, -np.inf, dtype=float)

        indices = ((prices - min_p) / bin_width).astype(int)
        indices = np.clip(indices, 0, num_bins - 1)

        for idx, ts in zip(indices, timestamps):
            ts_f = float(ts)
            if ts_f < bins_t_min[idx]:
                bins_t_min[idx] = ts_f
            if ts_f > bins_t_max[idx]:
                bins_t_max[idx] = ts_f

        dwell_times = bins_t_max - bins_t_min
        dwell_times[~np.isfinite(dwell_times)] = 0.0

        if not np.any(dwell_times > 0):
            return {"dwell_price": 0.0, "dwell_seconds": 0.0, "dwell_location": "N/A"}

        dwell_idx = int(np.nanargmax(dwell_times))
        dwell_ms = float(dwell_times[dwell_idx])
        dwell_seconds = max(dwell_ms / 1000.0, 0.0)
        if not np.isfinite(dwell_seconds):
            dwell_seconds = 0.0

        dwell_price = float(min_p + (dwell_idx + 0.5) * bin_width)

        cr = max_p - min_p
        if cr <= 0 or not np.isfinite(cr):
            loc = "Mid"
        elif dwell_price >= max_p - (cr * 0.2):
            loc = "High"
        elif dwell_price <= min_p + (cr * 0.2):
            loc = "Low"
        else:
            loc = "Mid"

        return {
            "dwell_price": dwell_price,
            "dwell_seconds": dwell_seconds,
            "dwell_location": loc,
        }
    except Exception as e:
        logging.error(f"Erro em _compute_dwell_time_array: {e}")
        return {"dwell_price": 0.0, "dwell_seconds": 0.0, "dwell_location": "N/A"}


def _compute_trade_speed_array(
    qtys: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """
    ✅ CORE: Trade speed usando apenas NumPy.

    Returns:
        {"trades_per_second", "avg_trade_size"}
    """
    try:
        if qtys.size < 2 or timestamps.size < 2:
            return {"trades_per_second": 0.0, "avg_trade_size": 0.0}

        mask = (
            np.isfinite(qtys) & np.isfinite(timestamps) &
            (timestamps > 0)
        )
        qtys = qtys[mask]
        timestamps = timestamps[mask]

        if qtys.size < 2:
            return {"trades_per_second": 0.0, "avg_trade_size": 0.0}

        t_min = float(np.min(timestamps))
        t_max = float(np.max(timestamps))
        duration_s = (t_max - t_min) / 1000.0
        num = int(qtys.size)

        if duration_s > 0 and np.isfinite(duration_s):
            tps = num / duration_s
        else:
            tps = 0.0

        sum_q = float(np.sum(qtys))
        avg = sum_q / num if num > 0 and np.isfinite(sum_q) else 0.0

        if not np.isfinite(tps):
            tps = 0.0
        if not np.isfinite(avg):
            avg = 0.0

        return {"trades_per_second": tps, "avg_trade_size": avg}
    except Exception as e:
        logging.error(f"Erro em _compute_trade_speed_array: {e}")
        return {"trades_per_second": 0.0, "avg_trade_size": 0.0}


# ===============================
# 📌 CAMADA 2: PANDAS WRAPPERS (Chamam CORE)
# ===============================

def _normalize_m_column(vals, default=False) -> pd.Series:
    """
    ✅ WRAPPER: Normaliza coluna 'm' usando _normalize_m_value (CORE).

    Semântica interna:
      - m = True  => agressor VENDEDOR  (taker SELL)
      - m = False => agressor COMPRADOR (taker BUY)

    Compatible with Binance aggTrade 'm' field (buyer_is_maker).
    """
    try:
        s = pd.Series(vals)
    except Exception:
        s = pd.Series(vals, dtype="object")

    # ✅ USA CORE (_normalize_m_value)
    out = s.map(_normalize_m_value)
    out = out.astype("boolean")
    
    if default is not None:
        out = out.fillna(bool(default))
    
    return out


def infer_or_fill_m(
    df: pd.DataFrame, 
    prev_price: Optional[float] = None, 
    prev_m: Optional[bool] = None
) -> pd.Series:
    """
    ✅ WRAPPER: Inferência de 'm' usando infer_or_fill_m_array (CORE).

    Combina m parcial com tick-rule, respeitando contexto de janela anterior.

    Semântica de m (taker side):
      - True  => agressor VENDEDOR  (taker SELL)
      - False => agressor COMPRADOR (taker BUY)

    Compatible with Binance aggTrade.
    """
    # Extrai arrays
    if "p" not in df.columns:
        logging.warning("Coluna 'p' ausente, retornando m=False")
        return pd.Series(False, index=df.index, dtype="boolean")
    
    prices = pd.to_numeric(df["p"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    
    raw_m = None
    if "m" in df.columns:
        raw_m = df["m"].tolist()
    
    # ✅ USA CORE (infer_or_fill_m_array)
    m_list = infer_or_fill_m_array(
        prices=prices,
        raw_m=raw_m,
        prev_price=prev_price,
        prev_m=prev_m,
    )
    
    return pd.Series(m_list, index=df.index, dtype="boolean")


def calcular_delta(
    df: pd.DataFrame, 
    *, 
    inplace: bool = False, 
    prev_price: Optional[float] = None, 
    prev_m: Optional[bool] = None
) -> pd.DataFrame:
    """
    ✅ WRAPPER: Delta = VolumeBuyMarket - VolumeSellMarket.

    Usa infer_or_fill_m (WRAPPER → CORE) para normalizar 'm'.

    Se o DF já tiver VolumeBuyMarket/VolumeSellMarket (frame agregado), usa diretamente.
    Caso contrário, deriva de q/m.

    Mantido para uso histórico/offline.
    """
    try:
        out = df if inplace else df.copy()

        if {"VolumeBuyMarket", "VolumeSellMarket"}.issubset(out.columns):
            out["VolumeBuyMarket"] = pd.to_numeric(out["VolumeBuyMarket"], errors="coerce").fillna(0.0)
            out["VolumeSellMarket"] = pd.to_numeric(out["VolumeSellMarket"], errors="coerce").fillna(0.0)
            out["Delta"] = out["VolumeBuyMarket"] - out["VolumeSellMarket"]
            return out

        q = pd.to_numeric(out.get("q", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

        try:
            # ✅ USA WRAPPER → CORE
            m_bool = infer_or_fill_m(out, prev_price=prev_price, prev_m=prev_m).astype(bool).to_numpy()
        except Exception:
            logging.debug("infer_or_fill_m falhou; assumindo m=False.")
            m_bool = np.zeros_like(q, dtype=bool)

        out["VolumeBuyMarket"] = np.where(~m_bool, q, 0.0)
        out["VolumeSellMarket"] = np.where(m_bool, q, 0.0)
        out["Delta"] = out["VolumeBuyMarket"] - out["VolumeSellMarket"]
        return out

    except Exception as e:
        logging.error(f"Erro ao calcular delta: {e}")
        out = df if inplace else df.copy()
        out["VolumeBuyMarket"] = pd.to_numeric(out.get("VolumeBuyMarket", 0.0), errors="coerce").fillna(0.0)
        out["VolumeSellMarket"] = pd.to_numeric(out.get("VolumeSellMarket", 0.0), errors="coerce").fillna(0.0)
        out["Delta"] = out["VolumeBuyMarket"] - out["VolumeSellMarket"]
        return out


def calcular_delta_normalizado(df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
    """Delta normalizado pelo range de preço (uso offline)."""
    try:
        df = df if inplace else df.copy()
        price_range = df.get("High", 0) - df.get("Low", 0)
        min_range = df.get("Close", 0) * 0.0001
        price_range = np.maximum(price_range, min_range)
        df["DeltaNorm"] = (df.get("Delta", 0) / price_range).replace([np.inf, -np.inf], 0).fillna(0)
        return df
    except Exception as e:
        logging.error(f"Erro ao calcular delta normalizado: {e}")
        out = df if inplace else (df.copy() if 'df' in locals() else pd.DataFrame())
        out["DeltaNorm"] = 0
        return out


def detectar_absorcao(
    df: pd.DataFrame, 
    delta_threshold: float = 0.5, 
    *, 
    inplace: bool = False
) -> pd.DataFrame:
    """
    ✅ WRAPPER: Detecta absorção em DataFrame usando _compute_absorption_scalar (CORE).

    Aplica CORE linha por linha para manter compatibilidade com código existente.
    """
    try:
        df = df if inplace else df.copy()
        required_cols = ["Delta", "Close", "Open", "High", "Low"]
        if not all(col in df.columns for col in required_cols):
            missing = ", ".join(c for c in required_cols if c not in df.columns)
            raise ValueError(f"Colunas ausentes para detectar absorção: {missing}")

        # Inicializa colunas
        df["AbsorcaoCompra"] = 0
        df["AbsorcaoVenda"] = 0
        df["IndiceAbsorcao"] = 0.0

        # Calcula buy/sell volumes se não existirem
        if "VolumeBuyMarket" not in df.columns or "VolumeSellMarket" not in df.columns:
            logging.warning("VolumeBuyMarket/VolumeSellMarket ausentes, usando Delta para aproximação")
            df["VolumeBuyMarket"] = df["Delta"].clip(lower=0)
            df["VolumeSellMarket"] = (-df["Delta"]).clip(lower=0)

        # ✅ Aplica CORE linha por linha
        for idx, row in df.iterrows():
            try:
                o = float(row.get("Open", 0))
                h = float(row.get("High", 0))
                l = float(row.get("Low", 0))
                c = float(row.get("Close", 0))
                buy_vol = float(row.get("VolumeBuyMarket", 0))
                sell_vol = float(row.get("VolumeSellMarket", 0))

                delta_btc, abs_compra, abs_venda, idx_abs = _compute_absorption_scalar(
                    o, h, l, c, delta_threshold, buy_vol, sell_vol
                )

                df.at[idx, "AbsorcaoCompra"] = int(abs_compra)
                df.at[idx, "AbsorcaoVenda"] = int(abs_venda)
                df.at[idx, "IndiceAbsorcao"] = idx_abs

            except Exception as e:
                logging.debug(f"Erro ao processar linha {idx}: {e}")
                continue

        return df

    except Exception as e:
        logging.error(f"Erro ao detectar absorção: {e}")
        out = df if inplace else (df.copy() if 'df' in locals() else pd.DataFrame())
        out["AbsorcaoCompra"] = 0
        out["AbsorcaoVenda"] = 0
        out["IndiceAbsorcao"] = 0
        return out


def aplicar_metricas_absorcao(
    df: pd.DataFrame, 
    delta_threshold: float, 
    *, 
    inplace: bool = False, 
    prev_price: Optional[float] = None, 
    prev_m: Optional[bool] = None
) -> pd.DataFrame:
    """
    ✅ WRAPPER: Pipeline de absorção completo (uso offline/histórico).

    Usa wrappers que chamam CORE internamente.
    """
    try:
        if inplace:
            calcular_delta(df, inplace=True, prev_price=prev_price, prev_m=prev_m)
            calcular_delta_normalizado(df, inplace=True)
            detectar_absorcao(df, delta_threshold, inplace=True)
            return df
        else:
            return (
                df.pipe(calcular_delta, inplace=False, prev_price=prev_price, prev_m=prev_m)
                  .pipe(calcular_delta_normalizado, inplace=False)
                  .pipe(detectar_absorcao, delta_threshold=delta_threshold, inplace=False)
            )
    except Exception as e:
        logging.error(f"Erro absorção: {e}")
        df['Delta'] = df.get('Delta', 0.0)
        df['DeltaNorm'] = df.get('DeltaNorm', 0.0)
        df['IndiceAbsorcao'] = df.get('IndiceAbsorcao', 0.0)
        df['AbsorcaoCompra'] = df.get('AbsorcaoCompra', 0)
        df['AbsorcaoVenda'] = df.get('AbsorcaoVenda', 0)
        return df


def calcular_metricas_intra_candle(df: pd.DataFrame) -> dict:
    """
    ✅ WRAPPER: Métricas intra-candle usando _compute_intra_candle_metrics_array (CORE).

    Converte DataFrame para arrays e usa implementação CORE.
    """
    try:
        if df.empty:
            return {
                "delta_minimo": 0.0,
                "delta_maximo": 0.0,
                "delta_fechamento": 0.0,
                "reversao_desde_minimo": 0.0,
                "reversao_desde_maximo": 0.0
            }

        q = pd.to_numeric(df['q'], errors='coerce').fillna(0.0).to_numpy(dtype=float) if 'q' in df.columns else np.array([], dtype=float)

        # ✅ USA WRAPPER → CORE para normalizar 'm'
        m = infer_or_fill_m(df).astype(bool).to_numpy()

        if q.size == 0 or m.size == 0:
            return {
                "delta_minimo": 0.0,
                "delta_maximo": 0.0,
                "delta_fechamento": 0.0,
                "reversao_desde_minimo": 0.0,
                "reversao_desde_maximo": 0.0
            }

        # ✅ USA CORE
        return _compute_intra_candle_metrics_array(q, m)

    except Exception as e:
        logging.error(f"Erro intra-candle: {e}")
        return {
            "delta_minimo": 0.0,
            "delta_maximo": 0.0,
            "delta_fechamento": 0.0,
            "reversao_desde_minimo": 0.0,
            "reversao_desde_maximo": 0.0
        }


def calcular_volume_profile(df: pd.DataFrame, num_bins=20) -> dict:
    """
    ✅ WRAPPER: Volume Profile usando _static_volume_profile_from_arrays (CORE).

    Converte DataFrame para arrays e usa implementação CORE.
    """
    try:
        if df.empty:
            return {"poc_price": 0.0, "poc_volume": 0.0, "poc_percentage": 0.0}

        df_copy = df.copy()
        df_copy['p'] = pd.to_numeric(df_copy['p'], errors='coerce')
        df_copy['q'] = pd.to_numeric(df_copy['q'], errors='coerce')
        df_copy = df_copy.dropna(subset=['p', 'q'])

        if df_copy.empty:
            return {"poc_price": 0.0, "poc_volume": 0.0, "poc_percentage": 0.0}

        prices = df_copy['p'].to_numpy(dtype=float)
        qtys = df_copy['q'].to_numpy(dtype=float)

        # ✅ USA CORE
        return _static_volume_profile_from_arrays(prices, qtys, num_bins=num_bins)

    except Exception as e:
        logging.error(f"Volume profile erro: {e}")
        return {"poc_price": 0.0, "poc_volume": 0.0, "poc_percentage": 0.0}


def calcular_dwell_time(df: pd.DataFrame, num_bins=20) -> dict:
    """
    ✅ WRAPPER: Dwell time usando _compute_dwell_time_array (CORE).

    Converte DataFrame para arrays e usa implementação CORE.
    """
    try:
        if df.empty or len(df) < 2:
            return {"dwell_price": 0.0, "dwell_seconds": 0.0, "dwell_location": "N/A"}

        df_copy = df.copy()
        df_copy['p'] = pd.to_numeric(df_copy['p'], errors='coerce')
        df_copy['T'] = pd.to_numeric(df_copy['T'], errors='coerce')
        df_copy = df_copy.dropna(subset=['p', 'T'])

        if df_copy.empty:
            return {"dwell_price": 0.0, "dwell_seconds": 0.0, "dwell_location": "N/A"}

        prices = df_copy['p'].to_numpy(dtype=float)
        timestamps = df_copy['T'].to_numpy(dtype=float)

        # ✅ USA CORE
        return _compute_dwell_time_array(prices, timestamps, num_bins=num_bins)

    except Exception as e:
        logging.error(f"Dwell erro: {e}")
        return {"dwell_price": 0.0, "dwell_seconds": 0.0, "dwell_location": "N/A"}


def calcular_trade_speed(df: pd.DataFrame) -> dict:
    """
    ✅ WRAPPER: Trade speed usando _compute_trade_speed_array (CORE).

    Converte DataFrame para arrays e usa implementação CORE.
    """
    try:
        if df.empty or len(df) < 2:
            return {"trades_per_second": 0.0, "avg_trade_size": 0.0}

        df_copy = df.copy()
        df_copy['T'] = pd.to_numeric(df_copy['T'], errors='coerce')
        df_copy['q'] = pd.to_numeric(df_copy['q'], errors='coerce')
        df_copy = df_copy.dropna(subset=['T', 'q'])

        if df_copy.empty:
            return {"trades_per_second": 0.0, "avg_trade_size": 0.0}

        qtys = df_copy['q'].to_numpy(dtype=float)
        timestamps = df_copy['T'].to_numpy(dtype=float)

        # ✅ USA CORE
        return _compute_trade_speed_array(qtys, timestamps)

    except Exception as e:
        logging.error(f"TradeSpeed erro: {e}")
        return {"trades_per_second": 0.0, "avg_trade_size": 0.0}


# ===============================
# FUNÇÕES BÁSICAS DE VALIDAÇÃO
# ===============================

def format_timestamp(ts_ms: int, tz=NY_TZ) -> str:
    """Formata timestamp (ms) em string ISO, com validações de range (uso apenas para debug/UI)."""
    try:
        current_year = datetime.now().year
        if ts_ms < 1577836800000:  # < 2020-01-01
            logging.warning(f"Timestamp muito antigo: {ts_ms}, substituindo por atual")
            ts_ms = int(datetime.now().timestamp() * 1000)
        elif ts_ms > (current_year + 5) * 31536000000:
            logging.warning(f"Timestamp futuro detectado: {ts_ms}, usando atual")
            ts_ms = int(datetime.now().timestamp() * 1000)
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(tz)
        return dt.isoformat(timespec="seconds")
    except Exception as e:
        logging.error(f"Erro ao formatar timestamp {ts_ms}: {e}")
        return datetime.now(tz).isoformat(timespec="seconds")


def validate_market_data(data: dict) -> tuple[bool, dict]:
    """Valida dados de mercado individuais — versão corrigida para lidar com strings."""
    try:
        for key in ['p', 'q', 'T']:
            if key in data and isinstance(data[key], str):
                try:
                    if '.' in data[key] or 'e' in data[key].lower():
                        data[key] = float(data[key])
                    else:
                        data[key] = int(data[key])
                except Exception:
                    data[key] = None

        validations = {
            'has_data': len(data) > 0,
            'has_price': 'p' in data and data['p'] is not None and isinstance(data['p'], (int, float)) and data['p'] > 0,
            'has_quantity': 'q' in data and data['q'] is not None and isinstance(data['q'], (int, float)) and data['q'] >= 0,
            'has_timestamp': 'T' in data and data['T'] is not None and isinstance(data['T'], (int, float)) and data['T'] > 1577836800000,
            'price_positive': data.get('p', 0) > 0 if data.get('p') is not None else False,
            'quantity_positive': data.get('q', 0) >= 0 if data.get('q') is not None else False,
            'timestamp_valid': data.get('T', 0) > 1577836800000 if data.get('T') is not None else False
        }
        return all(validations.values()), validations
    except Exception as e:
        logging.error(f"Erro ao validar dados de mercado: {e}")
        return False, {}


def validate_window_data(window_data: list) -> tuple[bool, list]:
    """Valida múltiplos trades (janela de análise)."""
    if not window_data:
        return False, []
    valid_data = []
    for trade in window_data:
        is_valid, _ = validate_market_data(trade)
        if is_valid:
            valid_data.append(trade)
    if len(valid_data) < 2:
        logging.warning(f"Janela com menos de 2 trades válidos: {len(valid_data)} válidos de {len(window_data)} totais")
        return False, []
    return True, valid_data


# ===============================
# HELPERS DE EVENTO
# ===============================

def _mk_event_id(symbol: str, tipo_evento: str, window_open_ms: int, delta_btc: float) -> str:
    """
    Gera um ID único para um evento de trading.

    O ID é baseado em campos críticos que identificam uma janela de evento e
    seus parâmetros principais (tempo de abertura, símbolo, tipo e delta). Usamos
    SHA-256 para reduzir a probabilidade de colisões.
    """
    base = f"{window_open_ms}|{symbol}|{tipo_evento}|{delta_btc:.8f}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _attach_time_index(event: Dict[str, Any], tm: TimeManager, epoch_ms: int) -> Dict[str, Any]:
    idx = tm.build_time_index(epoch_ms, include_local=True, timespec="milliseconds")
    # Alias compatível
    event["timestamp"] = idx["timestamp_utc"]
    event.update(idx)
    return event


# ===============================
# 📌 CAMADA 3: EVENTOS (Usam CORE diretamente)
# ===============================

def create_absorption_event(
    window_data: list,
    symbol: str,
    delta_threshold: float = 0.5,
    tz_output=timezone.utc,
    flow_metrics: dict = None,
    historical_profile: dict = None,
    time_manager: Optional[TimeManager] = None,
    event_epoch_ms: Optional[int] = None,
    data_context: str = "real_time",
    tick_context: Optional[dict] = None,
) -> dict:
    """
    ✅ Cria evento de Absorção usando funções CORE diretamente.

    🚀 OTIMIZADO PARA TEMPO REAL:
      - Não cria DataFrame para a janela de trades
      - Usa listas/NumPy via CORE
      - Usa Pandas apenas para DynamicVolumeProfile

    - Unidades: preço em USDT, q em BTC, T em ms
    - m (taker side, compatível com Binance):
        True  -> agressor VENDEDOR (taker SELL)
        False -> agressor COMPRADOR (taker BUY)
    """
    try:
        is_valid, clean_data = validate_window_data(window_data)
        if not is_valid:
            return {
                "schema_version": SCHEMA_VERSION,
                "is_signal": False,
                "tipo_evento": "Absorção",
                "resultado_da_batalha": "Dados inválidos",
                "descricao": "Poucos dados",
                "ativo": symbol,
            }

        tm = time_manager or TimeManager()

        # Ordena e garante chaves básicas sem usar DataFrame
        clean_data = [t for t in clean_data if t.get("p") and t.get("q") and t.get("T")]
        if not clean_data:
            return {
                "schema_version": SCHEMA_VERSION,
                "is_signal": False,
                "tipo_evento": "Absorção",
                "resultado_da_batalha": "Janela vazia",
                "descricao": "Sem dados válidos",
                "ativo": symbol,
            }

        clean_data.sort(key=lambda x: x["T"])

        prices = np.asarray([float(t["p"]) for t in clean_data], dtype=float)
        qtys = np.asarray([float(t["q"]) for t in clean_data], dtype=float)
        times = np.asarray([int(t["T"]) for t in clean_data], dtype=np.int64)
        raw_ms = [t.get("m") for t in clean_data]

        # Filtro de sanidade adicional
        mask = (
            np.isfinite(prices) & np.isfinite(qtys) & np.isfinite(times) &
            (prices > 0) & (qtys > 0) & (times > 0)
        )
        if not np.all(mask):
            prices = prices[mask]
            qtys = qtys[mask]
            times = times[mask]
            raw_ms = [raw_ms[i] for i, ok in enumerate(mask) if ok]

        if prices.size == 0 or qtys.size == 0 or times.size == 0:
            return {
                "schema_version": SCHEMA_VERSION,
                "is_signal": False,
                "tipo_evento": "Absorção",
                "resultado_da_batalha": "Janela vazia",
                "descricao": "Sem dados válidos após filtro",
                "ativo": symbol,
            }

        # OHLC
        ohlc = {
            "Open": float(prices[0]),
            "High": float(np.max(prices)),
            "Low": float(np.min(prices)),
            "Close": float(prices[-1]),
        }
        if any((not np.isfinite(v) or v <= 0) for v in ohlc.values()):
            return {
                "schema_version": SCHEMA_VERSION,
                "is_signal": False,
                "tipo_evento": "Absorção",
                "resultado_da_batalha": "Preços inválidos",
                "descricao": "OHLC inválido",
                "ativo": symbol,
            }

        # ✅ USA CORE: infer_or_fill_m_array
        ctx = tick_context or {}
        m_list = infer_or_fill_m_array(
            prices,
            raw_m=raw_ms,
            prev_price=ctx.get("prev_price"),
            prev_m=ctx.get("prev_m"),
        )
        m_flags = np.asarray(m_list, dtype=bool)

        # Direcionalidade e notionais
        notional_usdt = prices * qtys
        buy_mask = ~m_flags
        sell_mask = m_flags

        volume_buy_btc = float(qtys[buy_mask].sum()) if buy_mask.any() else 0.0
        volume_sell_btc = float(qtys[sell_mask].sum()) if sell_mask.any() else 0.0
        volume_total_btc = volume_buy_btc + volume_sell_btc

        buy_notional_usdt = float(notional_usdt[buy_mask].sum()) if buy_mask.any() else 0.0
        sell_notional_usdt = float(notional_usdt[sell_mask].sum()) if sell_mask.any() else 0.0
        total_notional_usdt = buy_notional_usdt + sell_notional_usdt

        # ✅ USA CORE: _compute_absorption_scalar
        delta_btc, absorcao_compra, absorcao_venda, indice_absorcao = _compute_absorption_scalar(
            ohlc["Open"],
            ohlc["High"],
            ohlc["Low"],
            ohlc["Close"],
            delta_threshold,
            volume_buy_btc,
            volume_sell_btc,
        )

        # Rótulos (mesma semântica anterior)
        resultado = "Sem Absorção"
        descricao = f"Δ={delta_btc:.2f}, índice={indice_absorcao:.2f}"
        absorption_side = None
        aggression_side = "sell" if delta_btc < 0 else ("buy" if delta_btc > 0 else "flat")

        # CORREÇÃO PRINCIPAL (mantida): inversão de rótulos
        if absorcao_compra:
            # Agressão vendedora absorvida -> evento é "Absorção de Venda"
            resultado = "Absorção de Venda"
            absorption_side = "buy"
            descricao = f"Agressão vendedora absorvida. Δ={delta_btc:.2f}, índice={indice_absorcao:.2f}"
        elif absorcao_venda:
            # Agressão compradora absorvida -> evento é "Absorção de Compra"
            resultado = "Absorção de Compra"
            absorption_side = "sell"
            descricao = f"Agressão compradora absorvida. Δ={delta_btc:.2f}, índice={indice_absorcao:.2f}"

            # ========================
            # Revalidação de rótulo
            # ========================
            # Garante consistência entre delta_btc e o rótulo calculado.
            try:
                eps = float(getattr(config, "ABSORCAO_DELTA_EPS", 1.0))
            except Exception:
                eps = 1.0
            rotulo_recalc = FlowAnalyzer.classificar_absorcao_por_delta(delta_btc, eps)
            if rotulo_recalc != "Neutra" and resultado != rotulo_recalc:
                logging.warning(
                    f"[ABSORCAO_RECALC] Inconsistência de rótulo: "
                    f"delta={delta_btc:.4f}, original='{resultado}', recalculado='{rotulo_recalc}'"
                )
                resultado = rotulo_recalc
                # Ajusta absorption_side de acordo com rótulo recalculado
                absorption_side = "sell" if rotulo_recalc == "Absorção de Compra" else (
                    "buy" if rotulo_recalc == "Absorção de Venda" else None
                )
                descricao = f"Rótulo recalculado. Δ={delta_btc:.2f}"

            # Validação final via guard
            _guard_absorcao(delta_btc, resultado, eps, getattr(config, "ABSORCAO_GUARD_MODE", "warn"))

        # Time index e janela
        window_open_ms = int(times.min())
        window_close_ms = int(times.max())
        event_ms = int(event_epoch_ms) if event_epoch_ms is not None else window_close_ms
        window_duration_ms = int(max(0, window_close_ms - window_open_ms))
        window_id = str(window_close_ms)

        # VP Dinâmico (com fallback estático leve)
        vp_fields: Dict[str, Any] = {}
        try:
            vpd = DynamicVolumeProfile(symbol=symbol)
            cvd = float((flow_metrics or {}).get("cvd", 0.0))
            whale_buy = float((flow_metrics or {}).get("whale_buy_volume", 0.0))
            whale_sell = float((flow_metrics or {}).get("whale_sell_volume", 0.0))
            atr = float(ohlc["High"] - ohlc["Low"]) if prices.size > 0 else 0.0

            # Criamos DataFrame apenas aqui (uso restrito ao VPD)
            df_vp = pd.DataFrame({
                "p": prices,
                "q": qtys,
                "T": times,
            })
            vp_data = vpd.calculate(
                df_vp,
                atr=atr,
                whale_buy_volume=whale_buy,
                whale_sell_volume=whale_sell,
                cvd=cvd,
            )
            if vp_data.get("status") == "success":
                hvns = sorted(set(float(x) for x in vp_data.get("hvns", [])))
                lvns = sorted(set(float(x) for x in vp_data.get("lvns", [])))
                poc = float(vp_data.get("poc_price", 0.0))
                vah = float(vp_data.get("vah", 0.0))
                val = float(vp_data.get("val", 0.0))
                if not (val <= poc <= vah):
                    logging.debug("VP inconsistente (VAL ≤ POC ≤ VAH) — mantendo assim mesmo para diagnóstico.")
                vp_fields.update({
                    "poc_price": poc,
                    "vah": vah,
                    "val": val,
                    "hvns": hvns,
                    "lvns": lvns,
                    "vpd_params": vp_data.get("params_used", {}),
                })
            else:
                # ✅ USA CORE como fallback
                vp_fields.update(_static_volume_profile_from_arrays(prices, qtys))
                logging.warning("VPD falhou, usando volume profile estático (CORE).")
        except Exception as e:
            logging.error(f"Erro ao calcular VPD: {e}")
            # ✅ USA CORE como fallback
            vp_fields.update(_static_volume_profile_from_arrays(prices, qtys))

        # ✅ USA CORE: Métricas adicionais
        intra: Dict[str, Any] = {}
        dwell: Dict[str, Any] = {}
        speed: Dict[str, Any] = {}
        try:
            intra = _compute_intra_candle_metrics_array(qtys, m_flags)
        except Exception as e:
            logging.error(f"Erro ao adicionar métricas intra-candle: {e}")
        try:
            dwell = _compute_dwell_time_array(
                prices,
                times,
                num_bins=getattr(config, "MAX_DWELL_BINS", 20),
            )
        except Exception as e:
            logging.error(f"Erro ao adicionar dwell time: {e}")
        try:
            speed = _compute_trade_speed_array(qtys, times)
        except Exception as e:
            logging.error(f"Erro ao adicionar trade speed: {e}")

        # Evento final
        event: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "is_signal": bool(resultado != "Sem Absorção"),
            "tipo_evento": "Absorção",
            "resultado_da_batalha": resultado,
            "descricao": descricao,
            "ativo": symbol,

            # Janela
            "window_open_ms": window_open_ms,
            "window_close_ms": window_close_ms,
            "window_duration_ms": window_duration_ms,
            "window_id": window_id,

            # Unidades padronizadas
            "volume_total_btc": volume_total_btc,
            "volume_compra_btc": volume_buy_btc,
            "volume_venda_btc": volume_sell_btc,
            "buy_notional_usdt": buy_notional_usdt,
            "sell_notional_usdt": sell_notional_usdt,
            "total_notional_usdt": total_notional_usdt,

            # Compat (legados)
            "volume_total": volume_total_btc,
            "volume_compra": volume_buy_btc,
            "volume_venda": volume_sell_btc,

            "delta": delta_btc,
            "indice_absorcao": indice_absorcao,
            "absorption_side": absorption_side,
            "aggression_side": aggression_side,

            # OHLC
            "preco_abertura": ohlc["Open"],
            "preco_maxima": ohlc["High"],
            "preco_minima": ohlc["Low"],
            "preco_fechamento": ohlc["Close"],
            "ohlc": {
                "open": ohlc["Open"],
                "high": ohlc["High"],
                "low": ohlc["Low"],
                "close": ohlc["Close"],
            },

            # Métricas extras
            **intra,
            **dwell,
            **speed,

            # Contexto
            "layer": "signal",
            "data_context": data_context,
            "source": {"exchange": "binance_futures", "stream": "trades"},
        }

        # VP e fluxo/histórico opcionais
        if vp_fields:
            event.update(vp_fields)
        if flow_metrics:
            event["fluxo_continuo"] = flow_metrics
        if historical_profile:
            event["historical_vp"] = historical_profile

        # Timestamps coerentes
        _attach_time_index(event, tm, event_ms)

        # event_id (novo formato baseado em window_open_ms e delta)
        event["event_id"] = _mk_event_id(
            symbol,
            "Absorção",
            window_open_ms,
            delta_btc,
        )

        # Checagem simples de unidades
        event["units_check_passed"] = (
            abs((volume_buy_btc - volume_sell_btc) - delta_btc) < 1e-8 and
            abs((volume_buy_btc + volume_sell_btc) - volume_total_btc) < 1e-8
        )

        # Contagens
        event["trades_count"] = int(prices.size)
        event["duration_s"] = float(max(0, (window_close_ms - window_open_ms)) / 1000.0)

        # Contexto para próxima janela (opcional)
        try:
            event["tick_context_out"] = {
                "last_price": float(prices[-1]),
                "last_m": bool(m_flags[-1]),
            }
        except Exception:
            pass

        return event

    except Exception as e:
        logging.error(f"Erro absorção: {e}", exc_info=True)
        return {
            "schema_version": SCHEMA_VERSION,
            "is_signal": False,
            "tipo_evento": "Erro",
            "resultado_da_batalha": "Erro",
            "descricao": str(e),
            "ativo": symbol,
        }


def create_exhaustion_event(
    window_data: list,
    symbol: str,
    history_volumes: Optional[List[float]] = None,
    volume_factor: float = 2.0,
    tz_output=timezone.utc,
    flow_metrics: dict = None,
    historical_profile: dict = None,
    time_manager: Optional[TimeManager] = None,
    event_epoch_ms: Optional[int] = None,
    data_context: str = "real_time",
    tick_context: Optional[dict] = None,
) -> dict:
    """
    ✅ Cria evento de Exaustão usando funções CORE diretamente.

    🚀 OTIMIZADO PARA TEMPO REAL:
      - Não cria DataFrame para a janela de trades
      - Usa listas/NumPy via CORE
      - Usa Pandas apenas para DynamicVolumeProfile

    - Compara volume da janela vs média histórica (history_volumes)
    - Unidades, timestamps e metadados alinhados à absorção

    m (taker side, compatível com Binance):
      - True  -> agressor VENDEDOR (taker SELL)
      - False -> agressor COMPRADOR (taker BUY)
    """
    try:
        is_valid, clean_data = validate_window_data(window_data)
        if not is_valid:
            return {
                "schema_version": SCHEMA_VERSION,
                "is_signal": False,
                "tipo_evento": "Exaustão",
                "resultado_da_batalha": "Dados inválidos",
                "descricao": "Poucos dados",
                "ativo": symbol,
            }

        tm = time_manager or TimeManager()

        history_volumes = [v for v in (history_volumes or []) if v > 0 and np.isfinite(v)]

        clean_data = [t for t in clean_data if t.get("p") and t.get("q") and t.get("T")]
        if not clean_data:
            return {
                "schema_version": SCHEMA_VERSION,
                "is_signal": False,
                "tipo_evento": "Exaustão",
                "resultado_da_batalha": "Janela vazia",
                "descricao": "Sem dados válidos",
                "ativo": symbol,
            }

        clean_data.sort(key=lambda x: x["T"])

        prices = np.asarray([float(t["p"]) for t in clean_data], dtype=float)
        qtys = np.asarray([float(t["q"]) for t in clean_data], dtype=float)
        times = np.asarray([int(t["T"]) for t in clean_data], dtype=np.int64)
        raw_ms = [t.get("m") for t in clean_data]

        mask = (
            np.isfinite(prices) & np.isfinite(qtys) & np.isfinite(times) &
            (prices > 0) & (qtys > 0) & (times > 0)
        )
        if not np.all(mask):
            prices = prices[mask]
            qtys = qtys[mask]
            times = times[mask]
            raw_ms = [raw_ms[i] for i, ok in enumerate(mask) if ok]

        if prices.size == 0 or qtys.size == 0 or times.size == 0:
            return {
                "schema_version": SCHEMA_VERSION,
                "is_signal": False,
                "tipo_evento": "Exaustão",
                "resultado_da_batalha": "Janela vazia",
                "descricao": "Sem dados válidos após filtro",
                "ativo": symbol,
            }

        # ✅ USA CORE: infer_or_fill_m_array
        ctx = tick_context or {}
        m_list = infer_or_fill_m_array(
            prices,
            raw_m=raw_ms,
            prev_price=ctx.get("prev_price"),
            prev_m=ctx.get("prev_m"),
        )
        m_flags = np.asarray(m_list, dtype=bool)

        # OHLC
        ohlc = {
            "Open": float(prices[0]),
            "High": float(np.max(prices)),
            "Low": float(np.min(prices)),
            "Close": float(prices[-1]),
        }

        # Direcionalidade e notionais
        notional_usdt = prices * qtys
        buy_mask = ~m_flags
        sell_mask = m_flags

        buy_btc = float(qtys[buy_mask].sum()) if buy_mask.any() else 0.0
        sell_btc = float(qtys[sell_mask].sum()) if sell_mask.any() else 0.0
        volume_total_btc = buy_btc + sell_btc

        buy_notional_usdt = float(notional_usdt[buy_mask].sum()) if buy_mask.any() else 0.0
        sell_notional_usdt = float(notional_usdt[sell_mask].sum()) if sell_mask.any() else 0.0
        total_notional_usdt = buy_notional_usdt + sell_notional_usdt

        avg_volume = float(np.mean(history_volumes)) if len(history_volumes) >= 5 else 0.0
        price_move = float(ohlc["Close"] - ohlc["Open"])

        resultado = "Sem Exaustão"
        descricao = f"Vol {volume_total_btc:.2f} (média {avg_volume:.2f})"
        is_signal = False

        if len(history_volumes) >= 5 and avg_volume > 0 and volume_total_btc > avg_volume * volume_factor:
            if price_move > 0 and buy_btc > sell_btc:
                is_signal = True
                resultado = "Exaustão de Compra"
                descricao = f"Pico de compra {volume_total_btc:.2f} vs média {avg_volume:.2f}"
            elif price_move < 0 and sell_btc > buy_btc:
                is_signal = True
                resultado = "Exaustão de Venda"
                descricao = f"Pico de venda {volume_total_btc:.2f} vs média {avg_volume:.2f}"

        # Time index e janela
        window_open_ms = int(times.min())
        window_close_ms = int(times.max())
        event_ms = int(event_epoch_ms) if event_epoch_ms is not None else window_close_ms
        window_duration_ms = int(max(0, window_close_ms - window_open_ms))
        window_id = str(window_close_ms)

        # VPD Dinâmico (com fallback)
        vp_fields: Dict[str, Any] = {}
        try:
            vpd = DynamicVolumeProfile(symbol=symbol)
            cvd = float((flow_metrics or {}).get("cvd", 0.0))
            whale_buy = float((flow_metrics or {}).get("whale_buy_volume", 0.0))
            whale_sell = float((flow_metrics or {}).get("whale_sell_volume", 0.0))
            atr = float(ohlc["High"] - ohlc["Low"]) if prices.size > 0 else 0.0

            df_vp = pd.DataFrame({
                "p": prices,
                "q": qtys,
                "T": times,
            })
            vp_data = vpd.calculate(
                df_vp,
                atr=atr,
                whale_buy_volume=whale_buy,
                whale_sell_volume=whale_sell,
                cvd=cvd,
            )
            if vp_data.get("status") == "success":
                hvns = sorted(set(float(x) for x in vp_data.get("hvns", [])))
                lvns = sorted(set(float(x) for x in vp_data.get("lvns", [])))
                vp_fields.update({
                    "poc_price": float(vp_data.get("poc_price", 0.0)),
                    "vah": float(vp_data.get("vah", 0.0)),
                    "val": float(vp_data.get("val", 0.0)),
                    "hvns": hvns,
                    "lvns": lvns,
                    "vpd_params": vp_data.get("params_used", {}),
                })
            else:
                # ✅ USA CORE como fallback
                vp_fields.update(_static_volume_profile_from_arrays(prices, qtys))
                logging.warning("VPD falhou, usando volume profile estático (CORE).")
        except Exception as e:
            logging.error(f"Erro ao calcular VPD: {e}")
            # ✅ USA CORE como fallback
            vp_fields.update(_static_volume_profile_from_arrays(prices, qtys))

        # ✅ USA CORE: Métricas adicionais
        intra: Dict[str, Any] = {}
        dwell: Dict[str, Any] = {}
        speed: Dict[str, Any] = {}
        try:
            intra = _compute_intra_candle_metrics_array(qtys, m_flags)
        except Exception as e:
            logging.error(f"Erro ao adicionar métricas intra-candle: {e}")
        try:
            dwell = _compute_dwell_time_array(
                prices,
                times,
                num_bins=getattr(config, "MAX_DWELL_BINS", 20),
            )
        except Exception as e:
            logging.error(f"Erro ao adicionar dwell time: {e}")
        try:
            speed = _compute_trade_speed_array(qtys, times)
        except Exception as e:
            logging.error(f"Erro ao adicionar trade speed: {e}")

        # Evento final
        event: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "is_signal": bool(is_signal),
            "tipo_evento": "Exaustão",
            "resultado_da_batalha": resultado,
            "descricao": descricao,
            "ativo": symbol,

            # Janela
            "window_open_ms": window_open_ms,
            "window_close_ms": window_close_ms,
            "window_duration_ms": window_duration_ms,
            "window_id": window_id,

            # Unidades padronizadas
            "volume_total_btc": volume_total_btc,
            "volume_compra_btc": buy_btc,
            "volume_venda_btc": sell_btc,
            "buy_notional_usdt": buy_notional_usdt,
            "sell_notional_usdt": sell_notional_usdt,
            "total_notional_usdt": total_notional_usdt,

            # Compat (legados)
            "volume_total": volume_total_btc,
            "volume_compra": buy_btc,
            "volume_venda": sell_btc,

            # OHLC
            "preco_abertura": ohlc["Open"],
            "preco_maxima": ohlc["High"],
            "preco_minima": ohlc["Low"],
            "preco_fechamento": ohlc["Close"],
            "ohlc": {
                "open": ohlc["Open"],
                "high": ohlc["High"],
                "low": ohlc["Low"],
                "close": ohlc["Close"],
            },

            # Métricas extras
            **intra,
            **dwell,
            **speed,

            # Contexto
            "layer": "signal",
            "data_context": data_context,
            "source": {"exchange": "binance_futures", "stream": "trades"},
        }

        if vp_fields:
            event.update(vp_fields)
        if flow_metrics:
            event["fluxo_continuo"] = flow_metrics
        if historical_profile:
            event["historical_vp"] = historical_profile

        # Timestamps coerentes
        _attach_time_index(event, tm, event_ms)

        # event_id (novo formato)
        event["event_id"] = _mk_event_id(
            symbol,
            "Exaustão",
            window_open_ms,
            0.0,
        )

        # Contagens
        event["trades_count"] = int(prices.size)
        event["duration_s"] = float(max(0, (window_close_ms - window_open_ms)) / 1000.0)

        # Contexto para próxima janela (opcional)
        try:
            event["tick_context_out"] = {
                "last_price": float(prices[-1]),
                "last_m": bool(m_flags[-1]),
            }
        except Exception:
            pass

        return event

    except Exception as e:
        logging.error(f"Erro exaustão: {e}", exc_info=True)
        return {
            "schema_version": SCHEMA_VERSION,
            "is_signal": False,
            "tipo_evento": "Erro",
            "resultado_da_batalha": "Erro",
            "descricao": str(e),
            "ativo": symbol,
        }


# ===============================
# CONFIGURAÇÕES PADRÃO
# ===============================

def get_default_config():
    return {
        'MIN_SIGNAL_VOLUME_BTC': 1.0,
        'MIN_SIGNAL_TPS': 2.0,
        'MIN_ABS_DELTA_BTC': 0.5,
        'MIN_REVERSAL_RATIO': 0.2,
        'INDEX_ATR_FLOOR_PCT': 0.001,
        'MAX_VOLUME_BINS': 20,
        'MAX_DWELL_BINS': 20
    }

try:
    if not hasattr(config, 'MIN_SIGNAL_VOLUME_BTC'):
        for k, v in get_default_config().items():
            setattr(config, k, v)
except Exception as e:
    logging.warning(f"Config não carregado: {e}")
    class Config:
        MIN_SIGNAL_VOLUME_BTC=1.0
        MIN_SIGNAL_TPS=2.0
        MIN_ABS_DELTA_BTC=0.5
        MIN_REVERSAL_RATIO=0.2
        INDEX_ATR_FLOOR_PCT=0.001
    config = Config()