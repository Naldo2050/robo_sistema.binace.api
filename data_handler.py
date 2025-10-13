
# -*- coding: utf-8 -*-
# data_handler
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
import hashlib
from typing import Dict, Any, List, Optional

import config

# 🔹 TIME MANAGER (injeção recomendada)
from time_manager import TimeManager

# 🔹 VOLUME PROFILE DINÂMICO
from dynamic_volume_profile import DynamicVolumeProfile

NY_TZ = ZoneInfo("America/New_York")

SCHEMA_VERSION = "1.1.0"


# ===============================
# UTILIDADES
# ===============================

def _normalize_m_column(vals, default=False) -> pd.Series:
    """
    Normaliza a coluna 'm' (agressor) para dtype BooleanDtype ('boolean' com suporte a NA).
      - True  => agressor vendedor  (taker sell)  [Semântica Binance: m=True -> buyer is maker]
      - False => agressor comprador (taker buy)

    Aceita: True/False, 1/0, "true"/"false", "SELL"/"BUY", "ask"/"bid", "s"/"b".
    Caso default seja None, mantém NAs (sem fill).
    """
    try:
        s = pd.Series(vals)
    except Exception:
        s = pd.Series(vals, dtype="object")

    def _coerce_one(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, np.integer)):
            if x == 1:
                return True
            if x == 0:
                return False
            return pd.NA
        if isinstance(x, str):
            t = x.strip().lower()
            if t in {"true", "t", "1", "sell", "ask", "s", "seller", "yes"}:
                return True
            if t in {"false", "f", "0", "buy", "bid", "b", "buyer", "no"}:
                return False
            return pd.NA
        return pd.NA

    out = s.map(_coerce_one)
    out = out.astype("boolean")
    if default is not None:
        out = out.fillna(bool(default))
    return out


def _infer_m_tick_rule(df: pd.DataFrame) -> pd.Series:
    """
    Inferência por tick-rule (apenas para linhas com m ausente):
    - preço subiu → agressor comprador (m=False)
    - preço caiu  → agressor vendedor (m=True)
    - preço igual → herda anterior; se primeira linha, assume False (BUY)
    """
    px = pd.to_numeric(df["p"], errors="coerce")
    prev = px.shift(1)
    m_tick = px <= prev  # True se caiu/igual => vendedor
    # Para o primeiro registro, se NA, assume False (BUY)
    if m_tick.isna().any():
        m_tick = m_tick.fillna(False)
    return m_tick.astype("boolean")


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
# CÁLCULOS DE ABSORÇÃO E MÉTRICAS
# ===============================

def calcular_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Delta = VolumeBuyMarket - VolumeSellMarket.

    Se o DF já tiver VolumeBuyMarket/VolumeSellMarket (frame agregado), usa diretamente.
    Caso contrário, deriva de q/m (normalizando 'm').
    """
    try:
        out = df.copy()

        if {"VolumeBuyMarket", "VolumeSellMarket"}.issubset(out.columns):
            out["VolumeBuyMarket"] = pd.to_numeric(out["VolumeBuyMarket"], errors="coerce").fillna(0.0)
            out["VolumeSellMarket"] = pd.to_numeric(out["VolumeSellMarket"], errors="coerce").fillna(0.0)
            out["Delta"] = out["VolumeBuyMarket"] - out["VolumeSellMarket"]
            return out

        q = pd.to_numeric(out.get("q", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)

        if "m" in out.columns:
            m_bool = _normalize_m_column(out["m"], default=None).astype("boolean")
            # Preenche NAs com tick-rule
            if m_bool.isna().any():
                try:
                    m_tick = _infer_m_tick_rule(out)
                    m_bool = m_bool.fillna(m_tick)
                except Exception:
                    m_bool = m_bool.fillna(False)
            m_bool = m_bool.astype(bool).to_numpy()
        else:
            # Sem 'm': tick-rule
            try:
                m_bool = _infer_m_tick_rule(out).astype(bool).to_numpy()
            except Exception:
                logging.debug("Campo 'm' ausente e tick-rule falhou; assumindo m=False.")
                m_bool = np.zeros_like(q, dtype=bool)

        out["VolumeBuyMarket"] = np.where(~m_bool, q, 0.0)
        out["VolumeSellMarket"] = np.where(m_bool, q, 0.0)
        out["Delta"] = out["VolumeBuyMarket"] - out["VolumeSellMarket"]
        return out

    except Exception as e:
        logging.error(f"Erro ao calcular delta: {e}")
        out = df.copy()
        out["VolumeBuyMarket"] = pd.to_numeric(out.get("VolumeBuyMarket", 0.0), errors="coerce").fillna(0.0)
        out["VolumeSellMarket"] = pd.to_numeric(out.get("VolumeSellMarket", 0.0), errors="coerce").fillna(0.0)
        out["Delta"] = out["VolumeBuyMarket"] - out["VolumeSellMarket"]
        return out


def calcular_delta_normalizado(df: pd.DataFrame) -> pd.DataFrame:
    """Delta normalizado pelo range de preço."""
    try:
        df = df.copy()
        price_range = df.get("High", 0) - df.get("Low", 0)
        min_range = df.get("Close", 0) * 0.0001
        price_range = np.maximum(price_range, min_range)
        df["DeltaNorm"] = (df.get("Delta", 0) / price_range).replace([np.inf, -np.inf], 0).fillna(0)
        return df
    except Exception as e:
        logging.error(f"Erro ao calcular delta normalizado: {e}")
        df_copy = df.copy() if 'df' in locals() else pd.DataFrame()
        df_copy["DeltaNorm"] = 0
        return df_copy


def detectar_absorcao(df: pd.DataFrame, delta_threshold: float = 0.5) -> pd.DataFrame:
    """Detecta absorção de compra/venda."""
    try:
        df = df.copy()
        required_cols = ["Delta", "Close", "Open", "High", "Low"]
        if not all(col in df.columns for col in required_cols):
            missing = ", ".join(c for c in required_cols if c not in df.columns)
            raise ValueError(f"Colunas ausentes para detectar absorção: {missing}")

        candle_range = df["High"] - df["Low"]
        candle_range = candle_range.replace(0, 0.0001)

        close_pos_compra = (df["Close"] - df["Low"]) / candle_range
        close_pos_venda = (df["High"] - df["Close"]) / candle_range

        cond_absorcao_compra = (
            (df["Delta"] < -abs(delta_threshold)) &
            (df["Close"] >= df["Open"] * 0.998) &
            (close_pos_compra > 0.5)
        )
        cond_absorcao_venda = (
            (df["Delta"] > abs(delta_threshold)) &
            (df["Close"] <= df["Open"] * 1.002) &
            (close_pos_venda > 0.5)
        )

        df["AbsorcaoCompra"] = cond_absorcao_compra.astype(int)
        df["AbsorcaoVenda"] = cond_absorcao_venda.astype(int)

        min_atr = df["Close"] * 0.001  # 0.1%
        atr = np.maximum(candle_range.rolling(14, min_periods=1).mean(), min_atr)
        df["IndiceAbsorcao"] = (df["Delta"].abs() / atr).replace([np.inf, -np.inf], 0).fillna(0)
        return df
    except Exception as e:
        logging.error(f"Erro ao detectar absorção: {e}")
        df_copy = df.copy() if 'df' in locals() else pd.DataFrame()
        df_copy["AbsorcaoCompra"] = 0
        df_copy["AbsorcaoVenda"] = 0
        df_copy["IndiceAbsorcao"] = 0
        return df_copy


def aplicar_metricas_absorcao(df: pd.DataFrame, delta_threshold: float) -> pd.DataFrame:
    try:
        df = calcular_delta(df)
        df = calcular_delta_normalizado(df)
        df = detectar_absorcao(df, delta_threshold)
    except Exception as e:
        logging.error(f"Erro absorção: {e}")
        df['Delta'] = df.get('Delta', 0.0)
        df['DeltaNorm'] = df.get('DeltaNorm', 0.0)
        df['IndiceAbsorcao'] = df.get('IndiceAbsorcao', 0.0)
        df['AbsorcaoCompra'] = df.get('AbsorcaoCompra', 0)
        df['AbsorcaoVenda'] = df.get('AbsorcaoVenda', 0)
    return df


# ===============================
# MÉTRICAS INTRA-CANDLE
# ===============================

def calcular_metricas_intra_candle(df: pd.DataFrame) -> dict:
    try:
        if df.empty:
            return {k: 0.0 for k in ["delta_minimo","delta_maximo","delta_fechamento",
                                     "reversao_desde_minimo","reversao_desde_maximo"]}
        q = pd.to_numeric(df['q'], errors='coerce').fillna(0.0).to_numpy(dtype=float) if 'q' in df.columns else np.array([], dtype=float)

        # normaliza m, mas sem viés (usa tick-rule para NAs)
        if 'm' in df.columns:
            m_series = _normalize_m_column(df['m'], default=None)
            if m_series.isna().any():
                m_series = m_series.fillna(_infer_m_tick_rule(df))
        else:
            m_series = _infer_m_tick_rule(df)
        m = m_series.astype(bool).to_numpy()

        if q.size == 0 or m.size == 0:
            return {k: 0.0 for k in ["delta_minimo","delta_maximo","delta_fechamento",
                                     "reversao_desde_minimo","reversao_desde_maximo"]}

        trade_delta = np.where(~m, q, -q)
        delta_cumulativo = np.cumsum(trade_delta)

        if delta_cumulativo.size == 0:
            return {k: 0.0 for k in ["delta_minimo","delta_maximo","delta_fechamento",
                                     "reversao_desde_minimo","reversao_desde_maximo"]}

        delta_min = float(np.nanmin(delta_cumulativo))
        delta_max = float(np.nanmax(delta_cumulativo))
        delta_close = float(delta_cumulativo[-1])

        if not np.isfinite(delta_min): delta_min = 0.0
        if not np.isfinite(delta_max): delta_max = 0.0
        if not np.isfinite(delta_close): delta_close = 0.0

        rev_buy = delta_close - delta_min
        rev_sell = delta_max - delta_close

        if not np.isfinite(rev_buy): rev_buy = 0.0
        if not np.isfinite(rev_sell): rev_sell = 0.0

        return {
            "delta_minimo": delta_min,
            "delta_maximo": delta_max,
            "delta_fechamento": delta_close,
            "reversao_desde_minimo": rev_buy,
            "reversao_desde_maximo": rev_sell
        }
    except Exception as e:
        logging.error(f"Erro intra-candle: {e}")
        return {k: 0.0 for k in ["delta_minimo","delta_maximo","delta_fechamento",
                                 "reversao_desde_minimo","reversao_desde_maximo"]}


# ===============================
# VOLUME PROFILE, DWELL, SPEED
# ===============================

def calcular_volume_profile(df: pd.DataFrame, num_bins=20) -> dict:
    """Volume Profile ESTÁTICO (fallback para VPD)."""
    try:
        if df.empty:
            return {"poc_price":0.0,"poc_volume":0.0,"poc_percentage":0.0}
        df_copy = df.copy()
        df_copy['p'] = pd.to_numeric(df_copy['p'], errors='coerce')
        df_copy['q'] = pd.to_numeric(df_copy['q'], errors='coerce')
        df_copy = df_copy.dropna(subset=['p','q'])
        if df_copy.empty:
            return {"poc_price":0.0,"poc_volume":0.0,"poc_percentage":0.0}

        min_p, max_p = df_copy['p'].min(), df_copy['p'].max()
        if min_p == max_p or not np.isfinite(min_p) or not np.isfinite(max_p):
            return {
                "poc_price": float(min_p if np.isfinite(min_p) else 0.0),
                "poc_volume": float(df_copy['q'].sum() if np.isfinite(df_copy['q'].sum()) else 0.0),
                "poc_percentage": 100.0
            }

        price_bins = pd.cut(df_copy['p'], bins=num_bins, include_lowest=True)
        volume_por_bin = df_copy.groupby(price_bins, observed=False)['q'].sum()
        if volume_por_bin.empty:
            return {"poc_price":0.0,"poc_volume":0.0,"poc_percentage":0.0}

        poc_bin = volume_por_bin.idxmax()
        poc_price = float(poc_bin.mid) if hasattr(poc_bin, 'mid') else float(poc_bin)
        poc_volume = float(volume_por_bin.max())
        total_volume = float(df_copy['q'].sum())
        poc_percentage = (poc_volume / total_volume)*100 if total_volume > 0 else 0.0

        return {"poc_price":poc_price,"poc_volume":poc_volume,"poc_percentage":poc_percentage}
    except Exception as e:
        logging.error(f"Volume profile erro: {e}")
        return {"poc_price":0.0,"poc_volume":0.0,"poc_percentage":0.0}


def calcular_dwell_time(df: pd.DataFrame, num_bins=20) -> dict:
    try:
        if df.empty or len(df) < 2:
            return {"dwell_price":0.0,"dwell_seconds":0.0,"dwell_location":"N/A"}
        df_copy = df.copy()
        df_copy['p'] = pd.to_numeric(df_copy['p'], errors='coerce')
        df_copy['T'] = pd.to_numeric(df_copy['T'], errors='coerce')
        df_copy = df_copy.dropna(subset=['p','T'])
        if df_copy.empty:
            return {"dwell_price":0.0,"dwell_seconds":0.0,"dwell_location":"N/A"}

        min_p, max_p = df_copy['p'].min(), df_copy['p'].max()
        if min_p == max_p or not np.isfinite(min_p) or not np.isfinite(max_p):
            dwell_seconds = (df_copy['T'].max()-df_copy['T'].min())/1000.0
            return {
                "dwell_price": float(min_p if np.isfinite(min_p) else 0.0),
                "dwell_seconds": float(dwell_seconds if np.isfinite(dwell_seconds) else 0.0),
                "dwell_location":"Mid"
            }

        price_bins = pd.cut(df_copy['p'], bins=num_bins, include_lowest=True)
        dwell_times = df_copy.groupby(price_bins, observed=False)['T'].apply(lambda x: x.max()-x.min())
        if dwell_times.empty:
            return {"dwell_price":0.0,"dwell_seconds":0.0,"dwell_location":"N/A"}

        dwell_bin = dwell_times.idxmax()
        dwell_price = float(dwell_bin.mid) if hasattr(dwell_bin, 'mid') else float(dwell_bin)
        dwell_seconds = float(dwell_times.max())/1000.0
        if not np.isfinite(dwell_seconds): dwell_seconds = 0.0

        cr = max_p-min_p
        if cr <= 0:
            loc = "Mid"
        elif dwell_price >= max_p-(cr*0.2):
            loc = "High"
        elif dwell_price <= min_p+(cr*0.2):
            loc = "Low"
        else:
            loc = "Mid"

        return {"dwell_price":dwell_price,"dwell_seconds":max(dwell_seconds,0.0),"dwell_location":loc}
    except Exception as e:
        logging.error(f"Dwell erro: {e}")
        return {"dwell_price":0.0,"dwell_seconds":0.0,"dwell_location":"N/A"}


def calcular_trade_speed(df: pd.DataFrame) -> dict:
    try:
        if df.empty or len(df) < 2:
            return {"trades_per_second":0.0,"avg_trade_size":0.0}
        df_copy = df.copy()
        df_copy['T'] = pd.to_numeric(df_copy['T'],errors='coerce')
        df_copy['q'] = pd.to_numeric(df_copy['q'],errors='coerce')
        df_copy = df_copy.dropna(subset=['T','q'])
        if df_copy.empty:
            return {"trades_per_second":0.0,"avg_trade_size":0.0}

        duration_s=(df_copy['T'].max()-df_copy['T'].min())/1000.0
        num=len(df_copy)
        tps=(num/duration_s) if duration_s>0 and np.isfinite(duration_s) else 0.0
        avg=df_copy['q'].sum()/num if num>0 and np.isfinite(df_copy['q'].sum()) else 0.0

        if not np.isfinite(tps): tps = 0.0
        if not np.isfinite(avg): avg = 0.0

        return {"trades_per_second":tps,"avg_trade_size":avg}
    except Exception as e:
        logging.error(f"TradeSpeed erro: {e}")
        return {"trades_per_second":0.0,"avg_trade_size":0.0}


# ===============================
# HELPERS DE EVENTO
# ===============================

def _mk_event_id(symbol: str, tipo_evento: str, window_close_ms: int, resultado: str, delta_btc: float, volume_total_btc: float) -> str:
    base = f"{symbol}|{tipo_evento}|{window_close_ms}|{resultado}|{delta_btc:.8f}|{volume_total_btc:.8f}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _attach_time_index(event: Dict[str, Any], tm: TimeManager, epoch_ms: int) -> Dict[str, Any]:
    idx = tm.build_time_index(epoch_ms, include_local=True, timespec="milliseconds")
    # Alias compatível
    event["timestamp"] = idx["timestamp_utc"]
    event.update(idx)
    return event


# ===============================
# EVENTOS DE ABSORÇÃO
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
) -> dict:
    """
    Cria evento de Absorção a partir de trades (aggTrade).
    - Unidades: preço em USDT, q em BTC, T em ms.
    - m: True (taker SELL), False (taker BUY). Fallback: tick-rule por trade.
    - Timestamps: derivados de event_epoch_ms (ou Tmax da janela).
    """
    try:
        is_valid, clean_data = validate_window_data(window_data)
        if not is_valid:
            return {"is_signal": False, "tipo_evento": "Absorção", "resultado_da_batalha": "Dados inválidos", "descricao": "Poucos dados", "ativo": symbol}

        tm = time_manager or TimeManager()

        df = pd.DataFrame(clean_data).copy()
        df["p"] = pd.to_numeric(df.get("p", 0), errors='coerce').fillna(0.0)
        df["q"] = pd.to_numeric(df.get("q", 0), errors='coerce').fillna(0.0)
        df["T"] = pd.to_numeric(df.get("T", 0), errors='coerce').fillna(0).astype(np.int64)

        df = df.dropna(subset=["p","q","T"])
        df = df[(df["p"] > 0) & (df["q"] > 0) & (df["T"] > 0)]
        df = df.sort_values("T").reset_index(drop=True)

        if df.empty:
            return {"is_signal": False, "tipo_evento": "Absorção", "resultado_da_batalha": "Janela vazia", "descricao": "Sem dados válidos", "ativo": symbol}

        # OHLC
        ohlc = {
            "Open": float(df["p"].iloc[0]),
            "High": float(df["p"].max()),
            "Low": float(df["p"].min()),
            "Close": float(df["p"].iloc[-1])
        }
        if any((not np.isfinite(v) or v <= 0) for v in ohlc.values()):
            return {"is_signal": False, "tipo_evento": "Absorção", "resultado_da_batalha": "Preços inválidos", "descricao": "OHLC inválido", "ativo": symbol}

        # Normaliza m (sem viés)
        if "m" in df.columns:
            m_series = _normalize_m_column(df["m"], default=None)
            if m_series.isna().any():
                m_series = m_series.fillna(_infer_m_tick_rule(df))
        else:
            m_series = _infer_m_tick_rule(df)
        df["m"] = m_series.astype(bool)

        # Direcionalidade e notionais
        df["notional_usdt"] = df["p"] * df["q"]
        buy_mask = ~df["m"]
        sell_mask = df["m"]

        volume_buy_btc = float(df.loc[buy_mask, "q"].sum())
        volume_sell_btc = float(df.loc[sell_mask, "q"].sum())
        volume_total_btc = volume_buy_btc + volume_sell_btc

        buy_notional_usdt = float(df.loc[buy_mask, "notional_usdt"].sum())
        sell_notional_usdt = float(df.loc[sell_mask, "notional_usdt"].sum())
        total_notional_usdt = buy_notional_usdt + sell_notional_usdt

        # Agregado para detecção
        agg_df = pd.DataFrame([{
            "Open": ohlc["Open"], "High": ohlc["High"], "Low": ohlc["Low"], "Close": ohlc["Close"],
            "VolumeBuyMarket": volume_buy_btc,
            "VolumeSellMarket": volume_sell_btc
        }])
        agg_df = aplicar_metricas_absorcao(agg_df, delta_threshold)

        delta_btc = float(agg_df["Delta"].iloc[0]) if len(agg_df) > 0 else (volume_buy_btc - volume_sell_btc)
        absorcao_compra = bool(agg_df["AbsorcaoCompra"].iloc[0]) if len(agg_df) > 0 else False
        absorcao_venda  = bool(agg_df["AbsorcaoVenda"].iloc[0])  if len(agg_df) > 0 else False
        indice_absorcao = float(agg_df["IndiceAbsorcao"].iloc[0]) if len(agg_df) > 0 else 0.0

        # CORREÇÃO PRINCIPAL: Inversão dos rótulos para refletir corretamente a natureza da agressão absorvida
        resultado = "Sem Absorção"
        descricao = f"Δ={delta_btc:.2f}, índice={indice_absorcao:.2f}"
        absorption_side = None
        aggression_side = "sell" if delta_btc < 0 else ("buy" if delta_btc > 0 else "flat")

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

        # Time index e janela
        window_open_ms = int(df["T"].min())
        window_close_ms = int(df["T"].max())
        event_ms = int(event_epoch_ms) if event_epoch_ms is not None else window_close_ms
        window_duration_ms = int(max(0, window_close_ms - window_open_ms))
        window_id = str(window_close_ms)

        # VP Dinâmico (com fallback)
        vp_fields = {}
        try:
            vpd = DynamicVolumeProfile(symbol=symbol)
            cvd = float((flow_metrics or {}).get("cvd", 0.0))
            whale_buy = float((flow_metrics or {}).get("whale_buy_volume", 0.0))
            whale_sell = float((flow_metrics or {}).get("whale_sell_volume", 0.0))
            atr = float(df["p"].max() - df["p"].min()) if len(df) > 0 else 0.0
            vp_data = vpd.calculate(df, atr=atr, whale_buy_volume=whale_buy, whale_sell_volume=whale_sell, cvd=cvd)
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
                    "vpd_params": vp_data.get("params_used", {})
                })
            else:
                vp_fields.update(calcular_volume_profile(df))
                logging.warning("VPD falhou, usando volume profile estático")
        except Exception as e:
            logging.error(f"Erro ao calcular VPD: {e}")
            vp_fields.update(calcular_volume_profile(df))

        # Métricas adicionais
        intra = {}
        dwell = {}
        speed = {}
        try: intra = calcular_metricas_intra_candle(df)
        except Exception as e: logging.error(f"Erro ao adicionar métricas intra-candle: {e}")
        try: dwell = calcular_dwell_time(df)
        except Exception as e: logging.error(f"Erro ao adicionar dwell time: {e}")
        try: speed = calcular_trade_speed(df)
        except Exception as e: logging.error(f"Erro ao adicionar trade speed: {e}")

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
            "ohlc": {"open": ohlc["Open"], "high": ohlc["High"], "low": ohlc["Low"], "close": ohlc["Close"]},

            # Métricas extras
            **intra, **dwell, **speed,

            # Contexto
            "layer": "signal",
            "data_context": data_context,
            "source": {"exchange": "binance_futures", "stream": "trades"},
        }

        # VP e fluxo/histórico opcionais
        if vp_fields: event.update(vp_fields)
        if flow_metrics: event["fluxo_continuo"] = flow_metrics
        if historical_profile: event["historical_vp"] = historical_profile

        # Timestamps coerentes
        _attach_time_index(event, tm, event_ms)

        # event_id
        event["event_id"] = _mk_event_id(symbol, "Absorção", window_close_ms, resultado, delta_btc, volume_total_btc)

        # Checagem simples de unidades
        event["units_check_passed"] = abs((volume_buy_btc - volume_sell_btc) - delta_btc) < 1e-8 and \
                                      abs((volume_buy_btc + volume_sell_btc) - volume_total_btc) < 1e-8

        # Contagens
        event["trades_count"] = int(len(df))
        event["duration_s"] = float(max(0, (window_close_ms - window_open_ms)) / 1000.0)

        return event

    except Exception as e:
        logging.error(f"Erro absorção: {e}", exc_info=True)
        return {"is_signal": False, "tipo_evento": "Erro", "resultado_da_batalha": "Erro", "descricao": str(e), "ativo": symbol}


# ===============================
# EVENTOS DE EXAUSTÃO
# ===============================

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
) -> dict:
    """
    Cria evento de Exaustão.
    - Compara volume da janela vs média histórica (history_volumes).
    - Unidades, timestamps e metadados alinhados à absorção.
    """
    try:
        is_valid, clean_data = validate_window_data(window_data)
        if not is_valid:
            return {"is_signal": False, "tipo_evento": "Exaustão", "resultado_da_batalha": "Dados inválidos", "descricao": "Poucos dados", "ativo": symbol}

        tm = time_manager or TimeManager()

        history_volumes = [v for v in (history_volumes or []) if v > 0 and np.isfinite(v)]

        df = pd.DataFrame(clean_data).copy()
        df["p"] = pd.to_numeric(df.get("p", 0), errors='coerce').fillna(0.0)
        df["q"] = pd.to_numeric(df.get("q", 0), errors='coerce').fillna(0.0)
        df["T"] = pd.to_numeric(df.get("T", 0), errors='coerce').fillna(0).astype(np.int64)
        df = df.dropna(subset=["p","q","T"])
        df = df[(df["p"] > 0) & (df["q"] > 0) & (df["T"] > 0)]
        df = df.sort_values("T").reset_index(drop=True)

        if df.empty:
            return {"is_signal": False, "tipo_evento": "Exaustão", "resultado_da_batalha": "Janela vazia", "descricao": "Sem dados válidos", "ativo": symbol}

        # Normaliza m para computar buy/sell volumes
        if "m" in df.columns:
            m_series = _normalize_m_column(df["m"], default=None)
            if m_series.isna().any():
                m_series = m_series.fillna(_infer_m_tick_rule(df))
        else:
            m_series = _infer_m_tick_rule(df)
        df["m"] = m_series.astype(bool)

        # OHLC
        ohlc = {
            "Open": float(df["p"].iloc[0]),
            "High": float(df["p"].max()),
            "Low": float(df["p"].min()),
            "Close": float(df["p"].iloc[-1])
        }

        # Direcionalidade e notionais
        df["notional_usdt"] = df["p"] * df["q"]
        buy_mask = ~df["m"]
        sell_mask = df["m"]

        buy_btc = float(df.loc[buy_mask, "q"].sum())
        sell_btc = float(df.loc[sell_mask, "q"].sum())
        volume_total_btc = buy_btc + sell_btc

        buy_notional_usdt = float(df.loc[buy_mask, "notional_usdt"].sum())
        sell_notional_usdt = float(df.loc[sell_mask, "notional_usdt"].sum())
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
        window_open_ms = int(df["T"].min())
        window_close_ms = int(df["T"].max())
        event_ms = int(event_epoch_ms) if event_epoch_ms is not None else window_close_ms
        window_duration_ms = int(max(0, window_close_ms - window_open_ms))
        window_id = str(window_close_ms)

        # VPD Dinâmico (com fallback)
        vp_fields = {}
        try:
            vpd = DynamicVolumeProfile(symbol=symbol)
            cvd = float((flow_metrics or {}).get("cvd", 0.0))
            whale_buy = float((flow_metrics or {}).get("whale_buy_volume", 0.0))
            whale_sell = float((flow_metrics or {}).get("whale_sell_volume", 0.0))
            atr = float(df["p"].max() - df["p"].min()) if len(df) > 0 else 0.0
            vp_data = vpd.calculate(df, atr=atr, whale_buy_volume=whale_buy, whale_sell_volume=whale_sell, cvd=cvd)
            if vp_data.get("status") == "success":
                hvns = sorted(set(float(x) for x in vp_data.get("hvns", [])))
                lvns = sorted(set(float(x) for x in vp_data.get("lvns", [])))
                vp_fields.update({
                    "poc_price": float(vp_data.get("poc_price", 0.0)),
                    "vah": float(vp_data.get("vah", 0.0)),
                    "val": float(vp_data.get("val", 0.0)),
                    "hvns": hvns,
                    "lvns": lvns,
                    "vpd_params": vp_data.get("params_used", {})
                })
            else:
                vp_fields.update(calcular_volume_profile(df))
                logging.warning("VPD falhou, usando volume profile estático")
        except Exception as e:
            logging.error(f"Erro ao calcular VPD: {e}")
            vp_fields.update(calcular_volume_profile(df))

        # Métricas adicionais
        intra = {}
        dwell = {}
        speed = {}
        try: intra = calcular_metricas_intra_candle(df)
        except Exception as e: logging.error(f"Erro ao adicionar métricas intra-candle: {e}")
        try: dwell = calcular_dwell_time(df)
        except Exception as e: logging.error(f"Erro ao adicionar dwell time: {e}")
        try: speed = calcular_trade_speed(df)
        except Exception as e: logging.error(f"Erro ao adicionar trade speed: {e}")

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
            "ohlc": {"open": ohlc["Open"], "high": ohlc["High"], "low": ohlc["Low"], "close": ohlc["Close"]},

            # Métricas extras
            **intra, **dwell, **speed,

            # Contexto
            "layer": "signal",
            "data_context": data_context,
            "source": {"exchange": "binance_futures", "stream": "trades"},
        }

        if vp_fields: event.update(vp_fields)
        if flow_metrics: event["fluxo_continuo"] = flow_metrics
        if historical_profile: event["historical_vp"] = historical_profile

        # Timestamps coerentes
        _attach_time_index(event, tm, event_ms)

        # event_id
        event["event_id"] = _mk_event_id(symbol, "Exaustão", window_close_ms, resultado, 0.0, volume_total_btc)

        # Contagens
        event["trades_count"] = int(len(df))
        event["duration_s"] = float(max(0, (window_close_ms - window_open_ms)) / 1000.0)

        return event

    except Exception as e:
        logging.error(f"Erro exaustão: {e}", exc_info=True)
        return {"is_signal": False, "tipo_evento": "Erro", "resultado_da_batalha": "Erro", "descricao": str(e), "ativo": symbol}


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