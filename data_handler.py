import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
import config

NY_TZ = ZoneInfo("America/New_York")

def format_timestamp(ts_ms: int, tz=NY_TZ) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(tz)
    return dt.isoformat(timespec="seconds")

# ===============================
# MÉTRICAS DE ABSORÇÃO
# ===============================
def calcular_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Delta"] = df["VolumeBuyMarket"] - df["VolumeSellMarket"]
    return df

def calcular_delta_normalizado(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_range = df["High"] - df["Low"]
    df["DeltaNorm"] = df["Delta"] / price_range.replace(0, np.nan)
    return df

def detectar_absorcao(df: pd.DataFrame, delta_threshold: float) -> pd.DataFrame:
    df = df.copy()
    cond_absorcao_compra = (df["Delta"] < -abs(delta_threshold)) & (df["Close"] >= df["Open"])
    cond_absorcao_venda = (df["Delta"] > abs(delta_threshold)) & (df["Close"] <= df["Open"])
    df["AbsorcaoCompra"] = cond_absorcao_compra.astype(int)
    df["AbsorcaoVenda"] = cond_absorcao_venda.astype(int)
    price_range = df["High"] - df["Low"]
    atr = price_range.rolling(window=14, min_periods=1).mean()
    df["IndiceAbsorcao"] = (df["Delta"].abs() / atr).replace([np.inf, -np.inf], 0).fillna(0)
    return df

def aplicar_metricas_absorcao(df: pd.DataFrame, delta_threshold: float) -> pd.DataFrame:
    df = calcular_delta(df)
    df = calcular_delta_normalizado(df)
    df = detectar_absorcao(df, delta_threshold=delta_threshold)
    return df

# ===============================
# MÉTRICAS DE EXAUSTÃO
# ===============================
def detectar_exaustao_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["VolumeTotal"] = df["VolumeBuyMarket"] + df["VolumeSellMarket"]
    return df

# ===============================
# MÉTRICAS INTRA-CANDLE
# ===============================
def calcular_metricas_intra_candle(df: pd.DataFrame) -> dict:
    df_copy = df.copy()
    df_copy['trade_delta'] = np.where(df_copy['m'] == False, df_copy['q'], -df_copy['q'])
    df_copy['delta_cumulativo'] = df_copy['trade_delta'].cumsum()
    
    if df_copy.empty:
        return { "delta_minimo": 0, "delta_maximo": 0, "delta_fechamento": 0, "reversao_desde_minimo": 0, "reversao_desde_maximo": 0 }
        
    delta_min = float(df_copy['delta_cumulativo'].min())
    delta_max = float(df_copy['delta_cumulativo'].max())
    delta_close = float(df_copy['delta_cumulativo'].iloc[-1])
    reversao_compradora = delta_close - delta_min
    reversao_vendedora = delta_max - delta_close
    
    return {
        "delta_minimo": delta_min, "delta_maximo": delta_max, "delta_fechamento": delta_close,
        "reversao_desde_minimo": reversao_compradora, "reversao_desde_maximo": reversao_vendedora
    }

# ===============================
# VOLUME PROFILE INTRA-CANDLE
# ===============================
def calcular_volume_profile(df: pd.DataFrame, num_bins=20) -> dict:
    if df.empty or df['q'].sum() == 0:
        return {"poc_price": 0, "poc_volume": 0, "poc_percentage": 0}
    df_copy = df.copy()
    min_price = df_copy['p'].min()
    max_price = df_copy['p'].max()
    if min_price == max_price:
        return {"poc_price": float(min_price), "poc_volume": float(df_copy['q'].sum()), "poc_percentage": 100.0}
    price_bins = pd.cut(df_copy['p'], bins=num_bins)
    volume_por_bin = df_copy.groupby(price_bins, observed=False)['q'].sum()
    poc_bin = volume_por_bin.idxmax()
    poc_price = float(poc_bin.mid)
    poc_volume = float(volume_por_bin.max())
    total_volume = float(df_copy['q'].sum())
    poc_percentage = (poc_volume / total_volume) * 100 if total_volume > 0 else 0
    return { "poc_price": poc_price, "poc_volume": poc_volume, "poc_percentage": poc_percentage }

# ===============================
# DWELL TIME
# ===============================
def calcular_dwell_time(df: pd.DataFrame, num_bins=20) -> dict:
    if df.empty or len(df) < 2:
        return {"dwell_price": 0, "dwell_seconds": 0, "dwell_location": "N/A"}
    df_copy = df.copy()
    min_price = df_copy['p'].min()
    max_price = df_copy['p'].max()
    if min_price == max_price:
        dwell_seconds = (df_copy['T'].max() - df_copy['T'].min()) / 1000
        return {"dwell_price": float(min_price), "dwell_seconds": dwell_seconds, "dwell_location": "Mid"}
    price_bins = pd.cut(df_copy['p'], bins=num_bins)
    dwell_times = df_copy.groupby(price_bins, observed=False)['T'].apply(lambda x: x.max() - x.min())
    dwell_bin = dwell_times.idxmax()
    dwell_price = float(dwell_bin.mid)
    dwell_seconds = float(dwell_times.max()) / 1000
    candle_range = max_price - min_price
    top_20_percent = max_price - (candle_range * 0.2)
    bottom_20_percent = min_price + (candle_range * 0.2)
    location = "Mid"
    if dwell_price >= top_20_percent:
        location = "High"
    elif dwell_price <= bottom_20_percent:
        location = "Low"
    return {"dwell_price": dwell_price, "dwell_seconds": dwell_seconds, "dwell_location": location}

# ===============================
# TRADE SPEED
# ===============================
def calcular_trade_speed(df: pd.DataFrame) -> dict:
    num_trades = len(df)
    if num_trades < 2:
        return {"trades_per_second": 0, "avg_trade_size": 0}
    duration_ms = df['T'].max() - df['T'].min()
    duration_s = duration_ms / 1000.0
    trades_per_second = (num_trades / duration_s) if duration_s > 0 else 0
    total_volume = df['q'].sum()
    avg_trade_size = (total_volume / num_trades) if num_trades > 0 else 0
    return {"trades_per_second": trades_per_second, "avg_trade_size": avg_trade_size}

# ===============================
# EVENTOS (Absorção e Exaustão)
# ===============================
def create_absorption_event(window_data: list, symbol: str, delta_threshold: float,
                            tz_output=NY_TZ, 
                            flow_metrics: dict=None,
                            historical_profile: dict=None) -> dict:
    try:
        df = pd.DataFrame(window_data)
        df["p"] = pd.to_numeric(df.get("p"), errors="coerce")
        df["q"] = pd.to_numeric(df.get("q"), errors="coerce")
        df["T"] = pd.to_numeric(df.get("T"), errors="coerce")
        df = df.dropna(subset=["p", "q", "T"])
        if df.empty:
            raise ValueError("Dados inválidos na janela")

        metricas_intra_candle = calcular_metricas_intra_candle(df)
        volume_profile_metrics = calcular_volume_profile(df)
        dwell_time_metrics = calcular_dwell_time(df)
        trade_speed_metrics = calcular_trade_speed(df)

        # OHLC
        ohlc = { 
            "Open": float(df["p"].iloc[0]), 
            "High": float(df["p"].max()), 
            "Low": float(df["p"].min()), 
            "Close": float(df["p"].iloc[-1]) 
        }
        df["VolumeBuyMarket"] = np.where(df["m"] == False, df["q"], 0.0)
        df["VolumeSellMarket"] = np.where(df["m"] == True, df["q"], 0.0)
        agg_df = pd.DataFrame({
            "Open": [ohlc["Open"]], "High": [ohlc["High"]], "Low": [ohlc["Low"]], "Close": [ohlc["Close"]],
            "VolumeBuyMarket": [float(df["VolumeBuyMarket"].sum())], 
            "VolumeSellMarket": [float(df["VolumeSellMarket"].sum())],
        })
        # Métrica padrão (marca absorção bruta)
        agg_df = aplicar_metricas_absorcao(agg_df, delta_threshold=delta_threshold)
        delta = float(agg_df["Delta"].iloc[0])
        absorcao_compra = int(agg_df["AbsorcaoCompra"].iloc[0])
        absorcao_venda = int(agg_df["AbsorcaoVenda"].iloc[0])
        indice_absorcao_raw = float(agg_df["IndiceAbsorcao"].iloc[0])

        # Índice de absorção robusto (piso de ATR p/ evitar explosões em range mínimo)
        price_range = ohlc["High"] - ohlc["Low"]
        atr_floor = ohlc["Close"] * float(getattr(config, "INDEX_ATR_FLOOR_PCT", 0.001))
        atr_ref = max(price_range, atr_floor)
        indice_absorcao = abs(delta) / (atr_ref if atr_ref > 0 else 1e-9)

        is_signal = False
        resultado = "Sem Absorção"
        descricao = f"Delta {delta:.4f}"
        if absorcao_compra:
            is_signal = True
            resultado = "Absorção de Compra"
            descricao = f"Vendedores agressivos absorvidos. Delta {delta:.4f}, Índice {indice_absorcao:.2f}"
        elif absorcao_venda:
            is_signal = True
            resultado = "Absorção de Venda"
            descricao = f"Compradores agressivos absorvidos. Delta {delta:.4f}, Índice {indice_absorcao:.2f}"

        event = {
            "is_signal": is_signal, "tipo_evento": "Absorção", "resultado_da_batalha": resultado, "descricao": descricao, "ativo": symbol,
            "preco_abertura": ohlc["Open"], "preco_fechamento": ohlc["Close"], "preco_maxima": ohlc["High"], "preco_minima": ohlc["Low"],
            "volume_total": float(agg_df["VolumeBuyMarket"].iloc[0] + agg_df["VolumeSellMarket"].iloc[0]),
            "volume_compra": float(agg_df["VolumeBuyMarket"].iloc[0]), "volume_venda": float(agg_df["VolumeSellMarket"].iloc[0]),
            "delta": delta, "indice_absorcao": float(indice_absorcao),
        }
        event.update(metricas_intra_candle)
        event.update(volume_profile_metrics)
        event.update(dwell_time_metrics)
        event.update(trade_speed_metrics)

        # Filtros de qualidade do sinal (gating)
        reasons = []
        # thresholds do config
        min_vol = float(getattr(config, "MIN_SIGNAL_VOLUME_BTC", 1.0))
        min_tps = float(getattr(config, "MIN_SIGNAL_TPS", 2.0))
        min_abs_delta = float(getattr(config, "MIN_ABS_DELTA_BTC", 0.5))
        min_rev_ratio = float(getattr(config, "MIN_REVERSAL_RATIO", 0.2))
        effective_delta_threshold = max(abs(delta_threshold or 0.0), min_abs_delta)

        tps = float(event.get("trades_per_second", 0))
        vol_total = float(event.get("volume_total", 0.0))

        # reversões
        rev_buy = float(event.get("reversao_desde_minimo", 0.0))
        rev_sell = float(event.get("reversao_desde_maximo", 0.0))

        if vol_total < min_vol:
            reasons.append(f"volume_baixo(<{min_vol})")
        if tps < min_tps:
            reasons.append(f"tps_baixo(<{min_tps})")
        if abs(delta) < effective_delta_threshold:
            reasons.append(f"delta_insuficiente(<{effective_delta_threshold})")

        # reversão mínima relativa ao |delta| quando marcado como absorção
        if absorcao_compra and (rev_buy < (min_rev_ratio * abs(delta))):
            reasons.append(f"reversao_insuficiente_buy(<{min_rev_ratio*abs(delta):.4f})")
        if absorcao_venda and (rev_sell < (min_rev_ratio * abs(delta))):
            reasons.append(f"reversao_insuficiente_sell(<{min_rev_ratio*abs(delta):.4f})")

        passed = (len(reasons) == 0)
        # Só mantém sinal se passou nos filtros
        event["is_signal"] = bool(is_signal and passed)
        event["signal_quality"] = {
            "passed": bool(passed),
            "reasons": reasons,
            "effective_delta_threshold": float(effective_delta_threshold),
            "indice_absorcao_raw": float(indice_absorcao_raw)
        }

        # Integração com fluxo contínuo
        if flow_metrics:
            event["fluxo_continuo"] = flow_metrics

        # Integração com perfil histórico (POC diário/VAH/VAL)
        if historical_profile:
            event["historical_vp"] = historical_profile

        return event
    except Exception as e:
        logging.error(f"Erro ao criar evento de absorção: {e}")
        return {"is_signal": False, "tipo_evento": "Erro", "resultado_da_batalha": "Erro", "descricao": str(e), "ativo": symbol}

def create_exhaustion_event(window_data: list, symbol: str, history_volumes=None,
                            volume_factor: float = 2.0,
                            tz_output=NY_TZ,
                            flow_metrics: dict=None,
                            historical_profile: dict=None) -> dict:
    try:
        history_volumes = history_volumes or []
        df = pd.DataFrame(window_data)
        df["p"] = pd.to_numeric(df.get("p"), errors="coerce")
        df["q"] = pd.to_numeric(df.get("q"), errors="coerce")
        df["T"] = pd.to_numeric(df.get("T"), errors="coerce")
        df = df.dropna(subset=["p", "q", "T"])
        if df.empty:
            raise ValueError("Dados inválidos na janela")

        metricas_intra_candle = calcular_metricas_intra_candle(df)
        volume_profile_metrics = calcular_volume_profile(df)
        dwell_time_metrics = calcular_dwell_time(df)
        trade_speed_metrics = calcular_trade_speed(df)

        # OHLC
        ohlc = { "Open": float(df["p"].iloc[0]), "High": float(df["p"].max()), "Low": float(df["p"].min()), "Close": float(df["p"].iloc[-1]) }
        df["VolumeBuyMarket"] = np.where(df["m"] == False, df["q"], 0.0)
        df["VolumeSellMarket"] = np.where(df["m"] == True, df["q"], 0.0)
        agg_df = pd.DataFrame({
            "Open": [ohlc["Open"]], "High": [ohlc["High"]], "Low": [ohlc["Low"]], "Close": [ohlc["Close"]],
            "VolumeBuyMarket": [float(df["VolumeBuyMarket"].sum())], "VolumeSellMarket": [float(df["VolumeSellMarket"].sum())],
        })
        agg_df = detectar_exaustao_volume(agg_df)
        current_volume = float(agg_df["VolumeBuyMarket"].iloc[0] + agg_df["VolumeSellMarket"].iloc[0])
        avg_volume = (sum(history_volumes) / len(history_volumes)) if history_volumes else 0.0
        resultado = "Sem Exaustão"
        descricao = f"Volume {current_volume:.2f} (média {avg_volume:.2f})"
        is_signal = False
        open_price = float(agg_df["Open"].iloc[0])
        close_price = float(agg_df["Close"].iloc[0])
        price_movement = close_price - open_price
        if history_volumes and current_volume > avg_volume * volume_factor:
            if price_movement > 0:
                is_signal = True
                resultado = "Exaustão de Compra"
                descricao = f"Pico de volume de compra ({current_volume:.2f} vs média {avg_volume:.2f}) pode indicar fim do movimento."
            elif price_movement < 0:
                is_signal = True
                resultado = "Exaustão de Venda"
                descricao = f"Pico de volume de venda ({current_volume:.2f} vs média {avg_volume:.2f}) pode indicar fim do movimento."

        event = {
            "is_signal": is_signal, "tipo_evento": "Exaustão", "resultado_da_batalha": resultado, "descricao": descricao, "ativo": symbol,
            "preco_abertura": open_price, "preco_fechamento": close_price, "preco_maxima": ohlc["High"], "preco_minima": ohlc["Low"],
            "volume_total": current_volume, "volume_compra": float(agg_df["VolumeBuyMarket"].iloc[0]), "volume_venda": float(agg_df["VolumeSellMarket"].iloc[0]),
        }
        event.update(metricas_intra_candle)
        event.update(volume_profile_metrics)
        event.update(dwell_time_metrics)
        event.update(trade_speed_metrics)

        # Integração com fluxo contínuo
        if flow_metrics:
            event["fluxo_continuo"] = flow_metrics

        # Integração com perfil histórico (POC diário/VAH/VAL)
        if historical_profile:
            event["historical_vp"] = historical_profile

        return event
    except Exception as e:
        logging.error(f"Erro ao criar evento de exaustão: {e}")
        return {"is_signal": False, "tipo_evento": "Erro", "resultado_da_batalha": "Erro", "descricao": str(e), "ativo": symbol}