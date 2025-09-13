import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
import config

NY_TZ = ZoneInfo("America/New_York")

# ===============================
# FUNÇÕES BÁSICAS DE VALIDAÇÃO
# ===============================

def format_timestamp(ts_ms: int, tz=NY_TZ) -> str:
    """Formata timestamp (ms) em string ISO, com validações de range"""
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
    """Valida dados de mercado individuais"""
    validations = {
        'has_data': len(data) > 0,
        'has_price': 'p' in data and data['p'] is not None,
        'has_quantity': 'q' in data and data['q'] is not None,
        'has_timestamp': 'T' in data and data['T'] is not None,
        'price_positive': data.get('p', 0) > 0,
        'quantity_positive': data.get('q', 0) >= 0,
        'timestamp_valid': data.get('T', 0) > 1577836800000
    }
    
    return all(validations.values()), validations

def validate_window_data(window_data: list) -> tuple[bool, list]:
    """Valida múltiplos trades (janela de análise)"""
    if not window_data:
        return False, []
    
    valid_data = []
    for trade in window_data:
        is_valid, _ = validate_market_data(trade)
        if is_valid:
            valid_data.append(trade)
    
    # precisa de pelo menos 2 trades
    return len(valid_data) >= 2, valid_data

# ===============================
# CÁLCULOS DE ABSORÇÃO
# ===============================

def calcular_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Delta = Volume de compra - Volume de venda"""
    df = df.copy()
    df['VolumeBuyMarket'] = df.get('VolumeBuyMarket', 0).fillna(0)
    df['VolumeSellMarket'] = df.get('VolumeSellMarket', 0).fillna(0)
    df['Delta'] = df['VolumeBuyMarket'] - df['VolumeSellMarket']
    return df

def calcular_delta_normalizado(df: pd.DataFrame) -> pd.DataFrame:
    """Delta normalizado pelo range de preço"""
    df = df.copy()
    price_range = df.get("High", 0) - df.get("Low", 0)
    min_range = df.get("Close", 0) * 0.0001
    price_range = np.maximum(price_range, min_range)
    df["DeltaNorm"] = (df.get("Delta", 0) / price_range).replace([np.inf, -np.inf], 0).fillna(0)
    return df

def detectar_absorcao(df: pd.DataFrame, delta_threshold: float = 0.5) -> pd.DataFrame:
    """Detecta absorção de compra/venda"""
    df = df.copy()
    
    cond_absorcao_compra = (
        (df["Delta"] < -abs(delta_threshold)) & 
        (df["Close"] >= df["Open"]) &
        (df["Close"] > 0) & (df["Open"] > 0)
    )
    cond_absorcao_venda = (
        (df["Delta"] > abs(delta_threshold)) &
        (df["Close"] <= df["Open"]) &
        (df["Close"] > 0) & (df["Open"] > 0)
    )
    
    df["AbsorcaoCompra"] = cond_absorcao_compra.astype(int)
    df["AbsorcaoVenda"] = cond_absorcao_venda.astype(int)
    
    price_range = df["High"] - df["Low"]
    min_atr = df["Close"] * 0.001  # 0.1% do preço
    atr = np.maximum(price_range.rolling(14, min_periods=1).mean(), min_atr)
    df["IndiceAbsorcao"] = (df["Delta"].abs() / atr).replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def aplicar_metricas_absorcao(df: pd.DataFrame, delta_threshold: float) -> pd.DataFrame:
    try:
        df = calcular_delta(df)
        df = calcular_delta_normalizado(df)
        df = detectar_absorcao(df, delta_threshold)
    except Exception as e:
        logging.error(f"Erro absorção: {e}")
        df['Delta'] = df['DeltaNorm'] = df['IndiceAbsorcao'] = 0.0
        df['AbsorcaoCompra'] = df['AbsorcaoVenda'] = 0
    return df

# ===============================
# MÉTRICAS INTRA-CANDLE
# ===============================

def calcular_metricas_intra_candle(df: pd.DataFrame) -> dict:
    try:
        df_copy = df.copy()
        df_copy['q'] = pd.to_numeric(df_copy['q'], errors='coerce').fillna(0)
        df_copy['m'] = df_copy['m'].astype(bool)

        df_copy['trade_delta'] = np.where(df_copy['m'] == False, df_copy['q'], -df_copy['q'])
        df_copy['delta_cumulativo'] = df_copy['trade_delta'].cumsum()
        
        if df_copy.empty:
            return {k: 0.0 for k in ["delta_minimo","delta_maximo","delta_fechamento",
                                     "reversao_desde_minimo","reversao_desde_maximo"]}
        
        delta_min = float(df_copy['delta_cumulativo'].min())
        delta_max = float(df_copy['delta_cumulativo'].max())
        delta_close = float(df_copy['delta_cumulativo'].iloc[-1])
        
        rev_buy = delta_close - delta_min
        rev_sell = delta_max - delta_close
        
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
    try:
        if df.empty: return {"poc_price":0,"poc_volume":0,"poc_percentage":0}
        df_copy = df.copy()
        df_copy['p'] = pd.to_numeric(df_copy['p'], errors='coerce')
        df_copy['q'] = pd.to_numeric(df_copy['q'], errors='coerce')
        df_copy = df_copy.dropna(subset=['p','q'])
        if df_copy.empty: return {"poc_price":0,"poc_volume":0,"poc_percentage":0}
        
        min_p, max_p = df_copy['p'].min(), df_copy['p'].max()
        if min_p == max_p:
            return {"poc_price":float(min_p),"poc_volume":float(df_copy['q'].sum()),"poc_percentage":100}
        
        price_bins = pd.cut(df_copy['p'], bins=num_bins, include_lowest=True)
        volume_por_bin = df_copy.groupby(price_bins, observed=False)['q'].sum()
        poc_bin = volume_por_bin.idxmax()
        poc_price = float(poc_bin.mid)
        poc_volume = float(volume_por_bin.max())
        poc_percentage = (poc_volume / df_copy['q'].sum())*100
        return {"poc_price":poc_price,"poc_volume":poc_volume,"poc_percentage":poc_percentage}
    except Exception as e:
        logging.error(f"Volume profile erro: {e}")
        return {"poc_price":0,"poc_volume":0,"poc_percentage":0}

def calcular_dwell_time(df: pd.DataFrame, num_bins=20) -> dict:
    try:
        if df.empty or len(df)<2: return {"dwell_price":0,"dwell_seconds":0,"dwell_location":"N/A"}
        df_copy = df.copy()
        df_copy['p'] = pd.to_numeric(df_copy['p'], errors='coerce')
        df_copy['T'] = pd.to_numeric(df_copy['T'], errors='coerce')
        df_copy = df_copy.dropna(subset=['p','T'])
        if df_copy.empty: return {"dwell_price":0,"dwell_seconds":0,"dwell_location":"N/A"}
        
        min_p, max_p = df_copy['p'].min(), df_copy['p'].max()
        if min_p==max_p:
            dwell_seconds = (df_copy['T'].max()-df_copy['T'].min())/1000.0
            return {"dwell_price":float(min_p),"dwell_seconds":dwell_seconds,"dwell_location":"Mid"}
        
        price_bins = pd.cut(df_copy['p'], bins=num_bins, include_lowest=True)
        dwell_times = df_copy.groupby(price_bins, observed=False)['T'].apply(lambda x: x.max()-x.min())
        dwell_bin = dwell_times.idxmax()
        dwell_price = float(dwell_bin.mid)
        dwell_seconds = float(dwell_times.max())/1000.0
        # localização
        cr = max_p-min_p
        if dwell_price >= max_p-(cr*0.2): loc="High"
        elif dwell_price <= min_p+(cr*0.2): loc="Low"
        else: loc="Mid"
        return {"dwell_price":dwell_price,"dwell_seconds":max(dwell_seconds,0),"dwell_location":loc}
    except Exception as e:
        logging.error(f"Dwell erro: {e}")
        return {"dwell_price":0,"dwell_seconds":0,"dwell_location":"N/A"}

def calcular_trade_speed(df: pd.DataFrame) -> dict:
    try:
        if df.empty or len(df)<2: return {"trades_per_second":0,"avg_trade_size":0}
        df_copy = df.copy()
        df_copy['T'] = pd.to_numeric(df_copy['T'],errors='coerce')
        df_copy['q'] = pd.to_numeric(df_copy['q'],errors='coerce')
        df_copy = df_copy.dropna(subset=['T','q'])
        if df_copy.empty: return {"trades_per_second":0,"avg_trade_size":0}
        
        duration_s=(df_copy['T'].max()-df_copy['T'].min())/1000.0
        num=len(df_copy)
        tps=(num/duration_s) if duration_s>0 else 0
        avg=df_copy['q'].sum()/num if num>0 else 0
        return {"trades_per_second":tps,"avg_trade_size":avg}
    except Exception as e:
        logging.error(f"TradeSpeed erro: {e}")
        return {"trades_per_second":0,"avg_trade_size":0}

# ===============================
# EVENTOS DE ABSORÇÃO
# ===============================

def create_absorption_event(window_data: list, symbol: str, delta_threshold: float = 0.5,
                            flow_metrics: dict=None,
                            historical_profile: dict=None) -> dict:
    try:
        # validar dados
        is_valid, clean_data = validate_window_data(window_data)
        if not is_valid:
            return {"is_signal":False,"tipo_evento":"Absorção","resultado_da_batalha":"Dados inválidos","descricao":"Poucos dados","ativo":symbol}
        
        df = pd.DataFrame(clean_data)
        df["p"]=pd.to_numeric(df["p"],errors="coerce")
        df["q"]=pd.to_numeric(df["q"],errors="coerce")
        df["T"]=pd.to_numeric(df["T"],errors="coerce")
        df = df.dropna(subset=["p","q","T"])
        if df.empty: raise ValueError("DataFrame vazio após limpeza")
        
        ohlc={"Open":float(df["p"].iloc[0]),"High":float(df["p"].max()),"Low":float(df["p"].min()),"Close":float(df["p"].iloc[-1])}
        if any(v<=0 for v in ohlc.values()):
            return {"is_signal":False,"tipo_evento":"Absorção","resultado_da_batalha":"Preços inválidos","descricao":"OHLC inválido","ativo":symbol}
        
        df["VolumeBuyMarket"]=np.where(df["m"]==False,df["q"],0.0)
        df["VolumeSellMarket"]=np.where(df["m"]==True,df["q"],0.0)
        
        agg_df=pd.DataFrame([{
            "Open":ohlc["Open"],"High":ohlc["High"],"Low":ohlc["Low"],"Close":ohlc["Close"],
            "VolumeBuyMarket":df["VolumeBuyMarket"].sum(),"VolumeSellMarket":df["VolumeSellMarket"].sum()
        }])
        agg_df=aplicar_metricas_absorcao(agg_df,delta_threshold)
        
        delta=float(agg_df["Delta"].iloc[0])
        absorcao_compra=bool(agg_df["AbsorcaoCompra"].iloc[0])
        absorcao_venda=bool(agg_df["AbsorcaoVenda"].iloc[0])
        indice_absorcao=float(agg_df["IndiceAbsorcao"].iloc[0])
        
        resultado="Sem Absorção"; descricao=f"Delta {delta:.2f}"
        is_signal=False
        if absorcao_compra:
            is_signal=True; resultado="Absorção de Compra"; descricao=f"Agressão vendedora absorvida. Δ={delta:.2f}, índice={indice_absorcao:.2f}"
        elif absorcao_venda:
            is_signal=True; resultado="Absorção de Venda"; descricao=f"Agressão compradora absorvida. Δ={delta:.2f}, índice={indice_absorcao:.2f}"
        
        event={"is_signal":is_signal,"tipo_evento":"Absorção","resultado_da_batalha":resultado,"descricao":descricao,"ativo":symbol,
               "preco_abertura":ohlc["Open"],"preco_maxima":ohlc["High"],"preco_minima":ohlc["Low"],"preco_fechamento":ohlc["Close"],
               "volume_total":float(df["q"].sum()),"volume_compra":float(df["VolumeBuyMarket"].sum()),"volume_venda":float(df["VolumeSellMarket"].sum()),
               "delta":delta,"indice_absorcao":indice_absorcao}
        
        # adds
        event.update(calcular_metricas_intra_candle(df))
        event.update(calcular_volume_profile(df))
        event.update(calcular_dwell_time(df))
        event.update(calcular_trade_speed(df))
        
        if flow_metrics: event["fluxo_continuo"]=flow_metrics
        if historical_profile: event["historical_vp"]=historical_profile
        return event
    except Exception as e:
        logging.error(f"Erro absorção: {e}")
        return {"is_signal":False,"tipo_evento":"Erro","resultado_da_batalha":"Erro","descricao":str(e),"ativo":symbol}

# ===============================
# EVENTOS DE EXAUSTÃO
# ===============================

def create_exhaustion_event(window_data: list, symbol: str, history_volumes=None, volume_factor: float=2.0,
                            flow_metrics: dict=None,
                            historical_profile: dict=None) -> dict:
    try:
        is_valid, clean_data=validate_window_data(window_data)
        if not is_valid:
            return {"is_signal":False,"tipo_evento":"Exaustão","resultado_da_batalha":"Dados inválidos","descricao":"Poucos dados","ativo":symbol}
        
        history_volumes=history_volumes or []
        history_volumes=[v for v in history_volumes if v>0]
        
        df=pd.DataFrame(clean_data)
        df["p"]=pd.to_numeric(df["p"],errors="coerce")
        df["q"]=pd.to_numeric(df["q"],errors="coerce")
        df["T"]=pd.to_numeric(df["T"],errors="coerce")
        df=df.dropna(subset=["p","q","T"])
        if df.empty: raise ValueError("DataFrame vazio após limpeza")
        
        ohlc={"Open":float(df["p"].iloc[0]),"High":float(df["p"].max()),"Low":float(df["p"].min()),"Close":float(df["p"].iloc[-1])}
        
        df["VolumeBuyMarket"]=np.where(df["m"]==False,df["q"],0.0)
        df["VolumeSellMarket"]=np.where(df["m"]==True,df["q"],0.0)
        buy=float(df["VolumeBuyMarket"].sum())
        sell=float(df["VolumeSellMarket"].sum())
        current_volume=buy+sell
        
        avg_volume=np.mean(history_volumes) if history_volumes else 0
        resultado="Sem Exaustão"; descricao=f"Vol {current_volume:.2f} (média {avg_volume:.2f})"; is_signal=False
        price_move=ohlc["Close"]-ohlc["Open"]
        
        if history_volumes and len(history_volumes)>=5 and current_volume>avg_volume*volume_factor:
            if price_move>0: 
                is_signal=True; resultado="Exaustão de Compra"; descricao=f"Pico de compra {current_volume:.2f} vs média {avg_volume:.2f}"
            elif price_move<0:
                is_signal=True; resultado="Exaustão de Venda"; descricao=f"Pico de venda {current_volume:.2f} vs média {avg_volume:.2f}"
        
        event={"is_signal":is_signal,"tipo_evento":"Exaustão","resultado_da_batalha":resultado,"descricao":descricao,
               "ativo":symbol,"preco_abertura":ohlc["Open"],"preco_maxima":ohlc["High"],"preco_minima":ohlc["Low"],"preco_fechamento":ohlc["Close"],
               "volume_total":current_volume,"volume_compra":buy,"volume_venda":sell}
        
        event.update(calcular_metricas_intra_candle(df))
        event.update(calcular_volume_profile(df))
        event.update(calcular_dwell_time(df))
        event.update(calcular_trade_speed(df))
        
        if flow_metrics: event["fluxo_continuo"]=flow_metrics
        if historical_profile: event["historical_vp"]=historical_profile
        return event
    except Exception as e:
        logging.error(f"Erro exaustão: {e}")
        return {"is_signal":False,"tipo_evento":"Erro","resultado_da_batalha":"Erro","descricao":str(e),"ativo":symbol}

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
    if not hasattr(config,'MIN_SIGNAL_VOLUME_BTC'):
        for k,v in get_default_config().items(): setattr(config,k,v)
except Exception as e:
    logging.warning(f"Config não carregado: {e}")
    class Config:
        MIN_SIGNAL_VOLUME_BTC=1.0
        MIN_SIGNAL_TPS=2.0
        MIN_ABS_DELTA_BTC=0.5
        MIN_REVERSAL_RATIO=0.2
        INDEX_ATR_FLOOR_PCT=0.001
    config=Config()