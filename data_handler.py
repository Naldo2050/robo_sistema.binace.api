import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
import config

# 🔹 IMPORTA TIME MANAGER
from time_manager import TimeManager

# 🔹 NOVO: IMPORTA VOLUME PROFILE DINÂMICO
from dynamic_volume_profile import DynamicVolumeProfile

NY_TZ = ZoneInfo("America/New_York")

# Inicializa TimeManager global
time_manager = TimeManager()

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
    """Valida dados de mercado individuais — versão corrigida para lidar com strings"""
    try:
        # Converte campos numéricos se forem string
        for key in ['p', 'q', 'T']:
            if key in data and isinstance(data[key], str):
                try:
                    if '.' in data[key] or 'e' in data[key].lower():
                        data[key] = float(data[key])
                    else:
                        data[key] = int(data[key])
                except ValueError:
                    data[key] = None  # Marca como inválido

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
    """Valida múltiplos trades (janela de análise)"""
    if not window_data:
        return False, []
    
    valid_data = []
    for trade in window_data:
        is_valid, _ = validate_market_data(trade)
        if is_valid:
            valid_data.append(trade)
    
    # precisa de pelo menos 2 trades
    if len(valid_data) < 2:
        logging.warning(f"Janela com menos de 2 trades válidos: {len(valid_data)} válidos de {len(window_data)} totais")
        return False, []
    
    return True, valid_data

# ===============================
# CÁLCULOS DE ABSORÇÃO
# ===============================

def calcular_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Delta = Volume de compra - Volume de venda"""
    try:
        df = df.copy()
        df['VolumeBuyMarket'] = df.get('VolumeBuyMarket', 0).fillna(0)
        df['VolumeSellMarket'] = df.get('VolumeSellMarket', 0).fillna(0)
        df['Delta'] = df['VolumeBuyMarket'] - df['VolumeSellMarket']
        return df
    except Exception as e:
        logging.error(f"Erro ao calcular delta: {e}")
        # Retorna DataFrame com colunas zeradas
        df_copy = df.copy() if 'df' in locals() else pd.DataFrame()
        df_copy['VolumeBuyMarket'] = 0
        df_copy['VolumeSellMarket'] = 0
        df_copy['Delta'] = 0
        return df_copy

def calcular_delta_normalizado(df: pd.DataFrame) -> pd.DataFrame:
    """Delta normalizado pelo range de preço"""
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
    """Detecta absorção de compra/venda"""
    try:
        df = df.copy()
        
        # Validação de dados
        if "Close" not in df.columns or "Open" not in df.columns:
            raise ValueError("Colunas Close ou Open ausentes")
            
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
        df['Delta'] = df['DeltaNorm'] = df['IndiceAbsorcao'] = 0.0
        df['AbsorcaoCompra'] = df['AbsorcaoVenda'] = 0
    return df

# ===============================
# MÉTRICAS INTRA-CANDLE
# ===============================

def calcular_metricas_intra_candle(df: pd.DataFrame) -> dict:
    try:
        # 🔹 OTIMIZADO: usa vetores NumPy (Fase 2)
        if df.empty:
            return {k: 0.0 for k in ["delta_minimo","delta_maximo","delta_fechamento",
                                     "reversao_desde_minimo","reversao_desde_maximo"]}
                                     
        q = df['q'].values if 'q' in df.columns else np.array([])
        m = df['m'].values if 'm' in df.columns else np.array([])
        
        # Validação dos arrays
        if len(q) == 0 or len(m) == 0:
            return {k: 0.0 for k in ["delta_minimo","delta_maximo","delta_fechamento",
                                     "reversao_desde_minimo","reversao_desde_maximo"]}
        
        trade_delta = np.where(m == False, q, -q)
        delta_cumulativo = np.cumsum(trade_delta)
        
        if len(delta_cumulativo) == 0:
            return {k: 0.0 for k in ["delta_minimo","delta_maximo","delta_fechamento",
                                     "reversao_desde_minimo","reversao_desde_maximo"]}
        
        # Validação de valores numéricos
        delta_min = float(np.nanmin(delta_cumulativo)) if len(delta_cumulativo) > 0 else 0.0
        delta_max = float(np.nanmax(delta_cumulativo)) if len(delta_cumulativo) > 0 else 0.0
        delta_close = float(delta_cumulativo[-1]) if len(delta_cumulativo) > 0 else 0.0
        
        # Verificação de valores válidos
        if not np.isfinite(delta_min): delta_min = 0.0
        if not np.isfinite(delta_max): delta_max = 0.0
        if not np.isfinite(delta_close): delta_close = 0.0
        
        rev_buy = delta_close - delta_min
        rev_sell = delta_max - delta_close
        
        # Validação final
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
    """Volume Profile ESTÁTICO (fallback para VPD)"""
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
            return {"poc_price":float(min_p if np.isfinite(min_p) else 0.0),
                    "poc_volume":float(df_copy['q'].sum() if np.isfinite(df_copy['q'].sum()) else 0.0),
                    "poc_percentage":100.0}
        
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
        if df.empty or len(df)<2: 
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
            return {"dwell_price":float(min_p if np.isfinite(min_p) else 0.0),
                    "dwell_seconds":float(dwell_seconds if np.isfinite(dwell_seconds) else 0.0),
                    "dwell_location":"Mid"}
        
        price_bins = pd.cut(df_copy['p'], bins=num_bins, include_lowest=True)
        dwell_times = df_copy.groupby(price_bins, observed=False)['T'].apply(lambda x: x.max()-x.min())
        if dwell_times.empty:
            return {"dwell_price":0.0,"dwell_seconds":0.0,"dwell_location":"N/A"}
            
        dwell_bin = dwell_times.idxmax()
        dwell_price = float(dwell_bin.mid) if hasattr(dwell_bin, 'mid') else float(dwell_bin)
        dwell_seconds = float(dwell_times.max())/1000.0
        if not np.isfinite(dwell_seconds): dwell_seconds = 0.0
        
        # localização
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
        if df.empty or len(df)<2: 
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
# EVENTOS DE ABSORÇÃO
# ===============================

def create_absorption_event(window_data: list, symbol: str, delta_threshold: float = 0.5,
                            tz_output=NY_TZ, flow_metrics: dict=None,
                            historical_profile: dict=None) -> dict:
    try:
        # validar dados
        is_valid, clean_data = validate_window_data(window_data)
        if not is_valid:
            logging.warning("Dados inválidos para criação de evento de absorção")
            return {"is_signal":False,"tipo_evento":"Absorção","resultado_da_batalha":"Dados inválidos","descricao":"Poucos dados","ativo":symbol}
        
        df = pd.DataFrame(clean_data)
        df["p"]=pd.to_numeric(df["p"],errors="coerce")
        df["q"]=pd.to_numeric(df["q"],errors="coerce")
        df["T"]=pd.to_numeric(df["T"],errors="coerce")
        df = df.dropna(subset=["p","q","T"])
        if df.empty: 
            logging.warning("DataFrame vazio após limpeza para evento de absorção")
            raise ValueError("DataFrame vazio após limpeza")
        
        ohlc={"Open":float(df["p"].iloc[0]) if len(df) > 0 else 0.0,
              "High":float(df["p"].max()) if len(df) > 0 else 0.0,
              "Low":float(df["p"].min()) if len(df) > 0 else 0.0,
              "Close":float(df["p"].iloc[-1]) if len(df) > 0 else 0.0}
              
        # Validação OHLC
        if any(not np.isfinite(v) or v <= 0 for v in ohlc.values()):
            logging.warning(f"OHLC inválido: {ohlc}")
            return {"is_signal":False,"tipo_evento":"Absorção","resultado_da_batalha":"Preços inválidos","descricao":"OHLC inválido","ativo":symbol}
        
        df["VolumeBuyMarket"]=np.where(df["m"]==False,df["q"],0.0) if 'm' in df.columns and 'q' in df.columns else 0.0
        df["VolumeSellMarket"]=np.where(df["m"]==True,df["q"],0.0) if 'm' in df.columns and 'q' in df.columns else 0.0
        
        agg_df=pd.DataFrame([{
            "Open":ohlc["Open"],"High":ohlc["High"],"Low":ohlc["Low"],"Close":ohlc["Close"],
            "VolumeBuyMarket":df["VolumeBuyMarket"].sum() if hasattr(df["VolumeBuyMarket"], 'sum') else 0.0,
            "VolumeSellMarket":df["VolumeSellMarket"].sum() if hasattr(df["VolumeSellMarket"], 'sum') else 0.0
        }])
        agg_df=aplicar_metricas_absorcao(agg_df,delta_threshold)
        
        delta=float(agg_df["Delta"].iloc[0]) if len(agg_df) > 0 else 0.0
        absorcao_compra=bool(agg_df["AbsorcaoCompra"].iloc[0]) if len(agg_df) > 0 else False
        absorcao_venda=bool(agg_df["AbsorcaoVenda"].iloc[0]) if len(agg_df) > 0 else False
        indice_absorcao=float(agg_df["IndiceAbsorcao"].iloc[0]) if len(agg_df) > 0 else 0.0
        
        resultado="Sem Absorção"; descricao=f"Delta {delta:.2f}"
        is_signal=False
        if absorcao_compra:
            is_signal=True; resultado="Absorção de Compra"; descricao=f"Agressão vendedora absorvida. Δ={delta:.2f}, índice={indice_absorcao:.2f}"
        elif absorcao_venda:
            is_signal=True; resultado="Absorção de Venda"; descricao=f"Agressão compradora absorvida. Δ={delta:.2f}, índice={indice_absorcao:.2f}"
        
        event={"is_signal":is_signal,"tipo_evento":"Absorção","resultado_da_batalha":resultado,"descricao":descricao,"ativo":symbol,
               "preco_abertura":ohlc["Open"],"preco_maxima":ohlc["High"],"preco_minima":ohlc["Low"],"preco_fechamento":ohlc["Close"],
               "volume_total":float(df["q"].sum()) if 'q' in df.columns else 0.0,
               "volume_compra":float(df["VolumeBuyMarket"].sum()) if hasattr(df["VolumeBuyMarket"], 'sum') else 0.0,
               "volume_venda":float(df["VolumeSellMarket"].sum()) if hasattr(df["VolumeSellMarket"], 'sum') else 0.0,
               "delta":delta,"indice_absorcao":indice_absorcao}
        
        # 🔹 NOVO: Volume Profile Dinâmico (VPD)
        try:
            vpd = DynamicVolumeProfile(symbol=symbol)
            flow_metrics = flow_metrics or {}
            cvd = flow_metrics.get("cvd", 0.0)
            whale_buy = flow_metrics.get("whale_buy_volume", 0.0)
            whale_sell = flow_metrics.get("whale_sell_volume", 0.0)
            atr = (df['p'].max() - df['p'].min()) if len(df) > 0 else 0.0
            
            vp_data = vpd.calculate(df, atr=atr, whale_buy_volume=whale_buy, whale_sell_volume=whale_sell, cvd=cvd)
            
            if vp_data.get("status") == "success":
                event.update({
                    "poc_price": vp_data["poc_price"],
                    "vah": vp_data["vah"],
                    "val": vp_data["val"],
                    "hvns": vp_data["hvns"],
                    "lvns": vp_data["lvns"],
                    "vpd_params": vp_data["params_used"]
                })
            else:
                # 🔹 Fallback: usa volume profile estático
                event.update(calcular_volume_profile(df))
                logging.warning("VPD falhou, usando volume profile estático")
        except Exception as e:
            logging.error(f"Erro ao calcular VPD: {e}")
            # 🔹 Fallback: usa volume profile estático
            event.update(calcular_volume_profile(df))
        
        # adds (mantém as outras métricas)
        try:
            event.update(calcular_metricas_intra_candle(df))
        except Exception as e:
            logging.error(f"Erro ao adicionar métricas intra-candle: {e}")
            
        try:
            event.update(calcular_dwell_time(df))
        except Exception as e:
            logging.error(f"Erro ao adicionar dwell time: {e}")
            
        try:
            event.update(calcular_trade_speed(df))
        except Exception as e:
            logging.error(f"Erro ao adicionar trade speed: {e}")
        
        if flow_metrics: event["fluxo_continuo"]=flow_metrics
        if historical_profile: event["historical_vp"]=historical_profile
        
        # 🔹 USA TIME MANAGER
        event["timestamp"] = time_manager.now_iso(tz=tz_output)
        
        return event
    except Exception as e:
        logging.error(f"Erro absorção: {e}")
        return {"is_signal":False,"tipo_evento":"Erro","resultado_da_batalha":"Erro","descricao":str(e),"ativo":symbol}

# ===============================
# EVENTOS DE EXAUSTÃO
# ===============================

def create_exhaustion_event(window_data: list, symbol: str, history_volumes=None, volume_factor: float=2.0,
                            tz_output=NY_TZ,
                            flow_metrics: dict=None,
                            historical_profile: dict=None) -> dict:
    try:
        is_valid, clean_data=validate_window_data(window_data)
        if not is_valid:
            logging.warning("Dados inválidos para criação de evento de exaustão")
            return {"is_signal":False,"tipo_evento":"Exaustão","resultado_da_batalha":"Dados inválidos","descricao":"Poucos dados","ativo":symbol}
        
        history_volumes=history_volumes or []
        history_volumes=[v for v in history_volumes if v>0 and np.isfinite(v)]
        
        df=pd.DataFrame(clean_data)
        df["p"]=pd.to_numeric(df["p"],errors="coerce")
        df["q"]=pd.to_numeric(df["q"],errors="coerce")
        df["T"]=pd.to_numeric(df["T"],errors="coerce")
        df=df.dropna(subset=["p","q","T"])
        if df.empty: 
            logging.warning("DataFrame vazio após limpeza para evento de exaustão")
            raise ValueError("DataFrame vazio após limpeza")
        
        ohlc={"Open":float(df["p"].iloc[0]) if len(df) > 0 else 0.0,
              "High":float(df["p"].max()) if len(df) > 0 else 0.0,
              "Low":float(df["p"].min()) if len(df) > 0 else 0.0,
              "Close":float(df["p"].iloc[-1]) if len(df) > 0 else 0.0}
        
        df["VolumeBuyMarket"]=np.where(df["m"]==False,df["q"],0.0) if 'm' in df.columns and 'q' in df.columns else 0.0
        df["VolumeSellMarket"]=np.where(df["m"]==True,df["q"],0.0) if 'm' in df.columns and 'q' in df.columns else 0.0
        buy=float(df["VolumeBuyMarket"].sum()) if hasattr(df["VolumeBuyMarket"], 'sum') else 0.0
        sell=float(df["VolumeSellMarket"].sum()) if hasattr(df["VolumeSellMarket"], 'sum') else 0.0
        current_volume=buy+sell
        
        avg_volume=np.mean(history_volumes) if len(history_volumes) >= 5 else 0.0
        resultado="Sem Exaustão"; descricao=f"Vol {current_volume:.2f} (média {avg_volume:.2f})"; is_signal=False
        price_move=ohlc["Close"]-ohlc["Open"] if np.isfinite(ohlc["Close"]) and np.isfinite(ohlc["Open"]) else 0.0
        
        if len(history_volumes) >= 5 and current_volume > avg_volume * volume_factor and np.isfinite(avg_volume) and avg_volume > 0:
            if price_move > 0: 
                is_signal=True; resultado="Exaustão de Compra"; descricao=f"Pico de compra {current_volume:.2f} vs média {avg_volume:.2f}"
            elif price_move < 0:
                is_signal=True; resultado="Exaustão de Venda"; descricao=f"Pico de venda {current_volume:.2f} vs média {avg_volume:.2f}"
        
        event={"is_signal":is_signal,"tipo_evento":"Exaustão","resultado_da_batalha":resultado,"descricao":descricao,
               "ativo":symbol,"preco_abertura":ohlc["Open"],"preco_maxima":ohlc["High"],"preco_minima":ohlc["Low"],"preco_fechamento":ohlc["Close"],
               "volume_total":current_volume,"volume_compra":buy,"volume_venda":sell}
        
        # 🔹 NOVO: Volume Profile Dinâmico (VPD)
        try:
            vpd = DynamicVolumeProfile(symbol=symbol)
            flow_metrics = flow_metrics or {}
            cvd = flow_metrics.get("cvd", 0.0)
            whale_buy = flow_metrics.get("whale_buy_volume", 0.0)
            whale_sell = flow_metrics.get("whale_sell_volume", 0.0)
            atr = (df['p'].max() - df['p'].min()) if len(df) > 0 else 0.0
            
            vp_data = vpd.calculate(df, atr=atr, whale_buy_volume=whale_buy, whale_sell_volume=whale_sell, cvd=cvd)
            
            if vp_data.get("status") == "success":
                event.update({
                    "poc_price": vp_data["poc_price"],
                    "vah": vp_data["vah"],
                    "val": vp_data["val"],
                    "hvns": vp_data["hvns"],
                    "lvns": vp_data["lvns"],
                    "vpd_params": vp_data["params_used"]
                })
            else:
                # 🔹 Fallback: usa volume profile estático
                event.update(calcular_volume_profile(df))
                logging.warning("VPD falhou, usando volume profile estático")
        except Exception as e:
            logging.error(f"Erro ao calcular VPD: {e}")
            # 🔹 Fallback: usa volume profile estático
            event.update(calcular_volume_profile(df))
        
        # adds (mantém as outras métricas)
        try:
            event.update(calcular_metricas_intra_candle(df))
        except Exception as e:
            logging.error(f"Erro ao adicionar métricas intra-candle: {e}")
            
        try:
            event.update(calcular_dwell_time(df))
        except Exception as e:
            logging.error(f"Erro ao adicionar dwell time: {e}")
            
        try:
            event.update(calcular_trade_speed(df))
        except Exception as e:
            logging.error(f"Erro ao adicionar trade speed: {e}")
        
        if flow_metrics: event["fluxo_continuo"]=flow_metrics
        if historical_profile: event["historical_vp"]=historical_profile
        
        # 🔹 USA TIME MANAGER
        event["timestamp"] = time_manager.now_iso(tz=tz_output)
        
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
        # ===============================
    config=Config()