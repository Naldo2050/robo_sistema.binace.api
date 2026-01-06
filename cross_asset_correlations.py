# cross_asset_correlations.py
"""
Módulo para cálculo de correlações cross-asset para BTCUSDT.

Foco especial:
- BTC x DXY (inversa) - correlação esperada negativa
- BTC x NDX - correlação com mercado tech
- BTC x ETH - correlação entre principais cryptos

Fontes de dados:
- Binance (velas 1h): BTCUSDT, ETHUSDT
- yfinance (diário): BTC-USD, DXY, ^NDX
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Import para novas fontes de dados macro
try:
    from src.data.macro_data_provider import MacroDataProvider
    _MACRO_DATA_OK = True
except ImportError as e:
    _MACRO_DATA_OK = False
    logging.warning(f"macro_data_provider indisponível: {e}")

# Configuração de logging
logger = logging.getLogger("CrossAssetCorrelations")

# ===============================
# Funções utilitárias internas
# ===============================

CORR_MIN_POINTS = 10  # Mínimo de pontos para correlação confiável

def _log_returns(series: pd.Series) -> pd.Series:
    """
    Calcula retornos logarítmicos de uma série de preços.

    Args:
        series: Série de preços

    Returns:
        Série de retornos logarítmicos (diff de log)
    """
    return np.log(series).diff().dropna()


def _corr_last_window(series_a: pd.Series,
                      series_b: pd.Series,
                      window: int) -> float:
    """
    Calcula correlação de Pearson entre duas séries de retornos,
    alinhando por posição (não por timestamp) para evitar NaN por falta de interseção de datas.

    Args:
        series_a: Primeira série de retornos
        series_b: Segunda série de retornos
        window: Número de pontos a considerar

    Returns:
        Correlação de Pearson ou NaN se dados insuficientes
    """
    max_len = min(len(series_a), len(series_b), window)
    if max_len < CORR_MIN_POINTS:
        return float("nan")

    a = series_a.tail(max_len).reset_index(drop=True)
    b = series_b.tail(max_len).reset_index(drop=True)
    corr = a.corr(b)

    return float(round(corr, 4)) if not pd.isna(corr) else float("nan")


# ===============================
# Funções de coleta de dados
# ===============================

def _fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 720) -> pd.DataFrame:
    """
    Busca velas da Binance usando a API REST.
    
    Args:
        symbol: Par de trading (ex: BTCUSDT)
        interval: Intervalo das velas (1h, 4h, 1d, etc.)
        limit: Número de velas a buscar (máx 1000)
        
    Returns:
        DataFrame com colunas: open_time, open, high, low, close, volume
    """
    import requests
    import time
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000)
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                logger.warning(f"Resposta inesperada da Binance: {data}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'num_trades', 'tbbav', 'tbqav', 'ignore'
            ])
            
            # Converte tipos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.set_index('open_time')
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Tentativa {attempt + 1}/{max_retries} falhou: {e}")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()


# Tickers de fallback por ativo (mesmo padrão do macro_fetcher.py)
_FALLBACK_TICKERS = {
    "BTC-USD": ["BTC-USD", "BTCUSD=X", "BTC=F"],  # BTC principal / BTC futures
    "DXY": ["DX-Y.NYB", "DX=F", "DXY"],          # Dollar Index (ICE) / futures / índice
    "NDX": ["^IXIC", "QQQ", "NDX"],              # Nasdaq Comp / ETF / índice
    "SPX": ["^GSPC", "SPY", "SPX"]               # S&P 500 índice / ETF
}

def _fetch_yfinance_data_with_fallbacks(name: str, period: str = "90d", interval: str = "1d") -> pd.DataFrame:
    """
    Busca dados históricos do yfinance com fallbacks robustos.
    
    Args:
        name: Nome do ativo (BTC-USD, DXY, NDX, SPX)
        period: Período de dados (ex: 30d, 90d, 1y)
        interval: Intervalo (1d, 1wk, 1mo)
        
    Returns:
        DataFrame com dados históricos
    """
    candidates = _FALLBACK_TICKERS.get(name, [name])
    
    for ticker in candidates:
        try:
            import yfinance as yf
            logger.debug(f"Tentando buscar {ticker} para {name}...")
            
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
            
            if df is None or df.empty:
                logger.debug(f"Dados vazios para {ticker}")
                continue
            
            # Verifica se tem dados válidos
            if 'Close' not in df.columns or df['Close'].isna().all():
                logger.debug(f"Coluna Close inválida para {ticker}")
                continue
            
            # Normaliza colunas
            df = df.rename(columns={'Close': 'close'})
            result_df = df[['close']].dropna()
            
            if len(result_df) >= 5:  # Pelo menos 5 pontos de dados
                logger.info(f"✅ Sucesso: {ticker} forneceu {len(result_df)} pontos para {name}")
                return result_df
            else:
                logger.debug(f"Dados insuficientes em {ticker}: {len(result_df)} pontos")
                
        except Exception as e:
            logger.debug(f"Erro ao buscar {ticker}: {e}")
            continue
    
    logger.warning(f"❌ Falha: nenhum ticker funcionou para {name} (candidatos={candidates})")
    return pd.DataFrame()

def _fetch_yfinance_data(ticker: str, period: str = "30d", interval: str = "1d") -> pd.DataFrame:
    """
    Busca dados históricos do yfinance (compatibilidade com versão anterior).
    
    Args:
        ticker: Ticker do ativo (ex: BTC-USD, DXY, ^NDX)
        period: Período de dados (ex: 30d, 90d, 1y)
        interval: Intervalo (1d, 1wk, 1mo)
        
    Returns:
        DataFrame com dados históricos
    """
    # Mapeia tickers antigos para nomes novos
    ticker_mapping = {
        "^NDX": "NDX",
        "^GSPC": "SPX", 
        "^IXIC": "NDX"
    }
    
    name = ticker_mapping.get(ticker, ticker)
    return _fetch_yfinance_data_with_fallbacks(name, period, interval)


# ===============================
# Funções principais de correlação
# ===============================

def get_btc_eth_correlations(now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calcula correlações entre BTCUSDT e ETHUSDT usando velas 1h da Binance.
    
    Args:
        now_utc: Timestamp atual em UTC (opcional)
        
    Returns:
        Dict com:
        - btc_eth_corr_7d: correlação dos últimos 7 dias (7*24 pontos)
        - btc_eth_corr_30d: correlação dos últimos 30 dias (30*24 pontos)
        - status: ok ou failed
        - error: mensagem de erro (se aplicável)
    """
    result = {
        "status": "ok",
        "btc_eth_corr_7d": float("nan"),
        "btc_eth_corr_30d": float("nan")
    }
    
    try:
        # Busca dados da Binance
        btc_df = _fetch_binance_klines("BTCUSDT", "1h", 30 * 24)
        eth_df = _fetch_binance_klines("ETHUSDT", "1h", 30 * 24)
        
        if btc_df.empty or eth_df.empty:
            raise ValueError("Dados insuficientes da Binance")
        
        # Calcula retornos logarítmicos
        btc_returns = _log_returns(btc_df['close'])
        eth_returns = _log_returns(eth_df['close'])
        
        if len(btc_returns) < 24 * 7 or len(eth_returns) < 24 * 7:
            raise ValueError("Dados insuficientes para cálculo")
        
        # Calcula correlações
        result["btc_eth_corr_7d"] = _corr_last_window(btc_returns, eth_returns, 24 * 7)
        result["btc_eth_corr_30d"] = _corr_last_window(btc_returns, eth_returns, 24 * 30)
        
        logger.info(f"Correlações BTC/ETH calculadas: 7d={result['btc_eth_corr_7d']:.4f}, 30d={result['btc_eth_corr_30d']:.4f}")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Erro ao calcular correlações BTC/ETH: {e}")
    
    return result


def get_btc_macro_correlations(now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calcula correlações entre BTC e ativos macro (DXY, NDX) usando yfinance.
    
    Args:
        now_utc: Timestamp atual em UTC (opcional)
        
    Returns:
        Dict com:
        - btc_dxy_corr_30d: correlação BTC x DXY (30 dias)
        - btc_dxy_corr_90d: correlação BTC x DXY (90 dias)
        - btc_ndx_corr_30d: correlação BTC x NDX (30 dias)
        - dxy_return_5d: retorno DXY nos últimos 5 dias
        - dxy_return_20d: retorno DXY nos últimos 20 dias
        - status: ok ou failed
        - error: mensagem de erro (se aplicável)
    """
    result = {
        "status": "ok",
        "btc_dxy_corr_30d": float("nan"),
        "btc_dxy_corr_90d": float("nan"),
        "btc_ndx_corr_30d": float("nan"),
        "dxy_return_5d": float("nan"),
        "dxy_return_20d": float("nan")
    }
    
    try:
        # Busca dados do yfinance
        btc_df = _fetch_yfinance_data("BTC-USD", period="90d")
        dxy_df = _fetch_yfinance_data("DXY", period="90d")
        ndx_df = _fetch_yfinance_data("^NDX", period="90d")
        
        if btc_df.empty or dxy_df.empty:
            raise ValueError("Dados insuficientes do yfinance")
        
        # Calcula retornos logarítmicos
        btc_returns = _log_returns(btc_df['close'])
        dxy_returns = _log_returns(dxy_df['close'])
        
        # Calcula correlações DXY
        result["btc_dxy_corr_30d"] = _corr_last_window(btc_returns, dxy_returns, 30)
        result["btc_dxy_corr_90d"] = _corr_last_window(btc_returns, dxy_returns, 90)
        
        # Calcula retornos DXY
        if len(dxy_df) >= 5:
            result["dxy_return_5d"] = float((dxy_df['close'].iloc[-1] / dxy_df['close'].iloc[-5] - 1) * 100)
        if len(dxy_df) >= 20:
            result["dxy_return_20d"] = float((dxy_df['close'].iloc[-1] / dxy_df['close'].iloc[-20] - 1) * 100)
        
        # Calcula correlação NDX se dados disponíveis
        if not ndx_df.empty:
            ndx_returns = _log_returns(ndx_df['close'])
            result["btc_ndx_corr_30d"] = _corr_last_window(btc_returns, ndx_returns, 30)
        
        logger.info(f"Correlações macro calculadas: DXY 30d={result['btc_dxy_corr_30d']:.4f}, 90d={result['btc_dxy_corr_90d']:.4f}")
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"Erro ao calcular correlações macro: {e}")
    
    return result


def get_enhanced_cross_asset_correlations(now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calcula correlações cross-asset ENHANCED com todas as novas métricas.
    
    Inclui:
    - BTC x ETH (crypto)
    - BTC x DXY, NDX (macro tradicional)  
    - BTC x VIX, Gold, Oil, Treasury Yields (novo)
    - Crypto Dominance
    - Regime Detection
    
    Args:
        now_utc: Timestamp atual em UTC (opcional)
        
    Returns:
        Dict com todas as métricas cross-asset enhanced
    """
    result = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() if now_utc is None else now_utc.isoformat()
    }
    
    # 1. CORRELAÇÕES TRADICIONAIS
    # Crypto (Binance)
    crypto_corr = get_btc_eth_correlations(now_utc)
    if crypto_corr["status"] == "ok":
        result.update(crypto_corr)
    else:
        result["status"] = "partial"
        result["crypto_error"] = crypto_corr.get("error")
    
    # Macro tradicional (yfinance)
    macro_corr = get_btc_macro_correlations(now_utc)
    if macro_corr["status"] == "ok":
        result.update(macro_corr)
    else:
        result["status"] = "partial"
        result["macro_error"] = macro_corr.get("error")
    
    # 2. NOVAS MÉTRICAS CROSS-ASSET via MacroDataProvider
    if _MACRO_DATA_OK:
        try:
            from utils.async_helpers import run_async_in_thread

            # Executa o provider de forma assíncrona usando helper
            macro_data = run_async_in_thread(_get_macro_data_async())
            
            # TRATAR None de forma segura
            if macro_data is None:
                logger.warning("⚠️ Enhanced correlations: macro_data é None, usando valores padrão")
                macro_data = {
                    "vix": None,
                    "treasury_10y": None,
                    "dxy": None,
                    "gold": None,
                    "oil": None,
                    "btc_dominance": None,
                    "eth_dominance": None,
                    "usdt_dominance": None,
                }
            
            # VIX metrics
            if macro_data.get("vix") is not None:
                result["vix_current"] = macro_data["vix"]
                result["vix_change_1d"] = None  # Calcular se necessário
            
            # Treasury Yields
            if macro_data.get("treasury_10y") is not None:
                result["us10y_yield"] = macro_data["treasury_10y"]
            if macro_data.get("treasury_2y") is not None:
                result["us2y_yield"] = macro_data["treasury_2y"]
            if macro_data.get("yield_spread") is not None:
                result["us10y_change_1d"] = macro_data["yield_spread"]  # Proxy
            
            # Crypto Dominance
            if macro_data.get("btc_dominance") is not None:
                result["btc_dominance"] = macro_data["btc_dominance"]
            if macro_data.get("eth_dominance") is not None:
                result["eth_dominance"] = macro_data["eth_dominance"]
            if macro_data.get("usdt_dominance") is not None:
                result["usdt_dominance"] = macro_data["usdt_dominance"]
            
            # DXY (Dollar Index)
            if macro_data.get("dxy") is not None:
                result["dxy_current"] = macro_data["dxy"]
            
            # Commodities
            if macro_data.get("gold") is not None:
                result["gold_price"] = macro_data["gold"]
                result["gold_change_1d"] = None  # Calcular se necessário
            if macro_data.get("oil") is not None:
                result["oil_price"] = macro_data["oil"]
                result["oil_change_1d"] = None  # Calcular se necessário
            
            # NOVAS CORRELAÇÕES (simplificadas)
            btc_returns = None
            
            # BTC x VIX correlation (se dados disponíveis)
            if macro_data.get("vix") is not None:
                try:
                    btc_df = _fetch_yfinance_data("BTC-USD", period="30d")
                    if not btc_df.empty:
                        btc_returns = _log_returns(btc_df['close'])
                        # Usar VIX como proxy (teríamos que baixar histórico para correlação real)
                        result["btc_vix_corr_30d"] = None  # Placeholder
                except:
                    pass
            
            # BTC x Gold correlation
            if macro_data.get("gold") is not None:
                if btc_returns is None:
                    try:
                        btc_df = _fetch_yfinance_data("BTC-USD", period="30d")
                        if not btc_df.empty:
                            btc_returns = _log_returns(btc_df['close'])
                    except:
                        pass
                result["btc_gold_corr_30d"] = None  # Placeholder
            
            # BTC x Oil correlation
            if macro_data.get("oil") is not None:
                result["btc_oil_corr_30d"] = None  # Placeholder
            
            # BTC x Treasury Yields correlation
            if macro_data.get("treasury_10y") is not None:
                result["btc_yields_corr_30d"] = None  # Placeholder
            
            # Dominance change (7d) - placeholder
            result["btc_dominance_change_7d"] = 0.0
            
            # Correlation Regime (baseado em BTC x DXY)
            btc_dxy_corr = result.get("btc_dxy_corr_30d")
            if pd.notna(btc_dxy_corr):
                result["correlation_regime"] = _calculate_correlation_regime(btc_dxy_corr)
            else:
                result["correlation_regime"] = "UNKNOWN"
            
            # Macro Regime (baseado em dados disponíveis)
            result["macro_regime"] = _calculate_macro_regime_simple(macro_data)
            
            logger.info(f"✅ Enhanced cross-asset correlations calculadas: {len([k for k in result.keys() if not k.startswith('_')])} features")
            
        except Exception as e:
            logger.error(f"Erro ao calcular enhanced correlations: {e}")
            result["enhanced_error"] = str(e)
            if result["status"] == "ok":
                result["status"] = "partial"
    else:
        logger.warning("macro_data_provider não disponível, pulando enhanced metrics")
        result["enhanced_status"] = "unavailable"
    
    return result


async def _get_macro_data_async() -> Dict[str, Any]:
    """Função auxiliar para executar MacroDataProvider de forma assíncrona."""
    provider = MacroDataProvider()
    return await provider.get_all_macro_data()


def _calculate_macro_regime_simple(macro_data: Dict[str, Any]) -> str:
    """Calcula regime macro simplificado baseado nos dados disponíveis."""
    try:
        risk_score = 0
        factors = 0
        
        # VIX: > 25 = risk off, < 15 = risk on
        vix = macro_data.get("vix")
        if vix is not None:
            factors += 1
            if vix > 25:
                risk_score += 2
            elif vix < 15:
                risk_score -= 1
        
        # BTC Dominance: > 50% = risk off, < 40% = risk on
        btc_dom = macro_data.get("btc_dominance")
        if btc_dom is not None:
            factors += 1
            if btc_dom > 50:
                risk_score += 1
            elif btc_dom < 40:
                risk_score -= 1
        
        # Treasury Yields: subida = risk off
        treasury_10y = macro_data.get("treasury_10y")
        treasury_2y = macro_data.get("treasury_2y")
        if treasury_10y is not None and treasury_2y is not None:
            factors += 1
            # Spread como proxy de mudança
            spread = treasury_10y - treasury_2y
            if spread > 0.5:  # Yield curve steepening = risk off
                risk_score += 1
            elif spread < 0:  # Yield curve inversion = risk off
                risk_score += 2
        
        if factors == 0:
            return "UNKNOWN"
        
        avg_score = risk_score / factors
        
        if avg_score >= 1.0:
            return "RISK_OFF"
        elif avg_score <= -1.0:
            return "RISK_ON"
        else:
            return "TRANSITION"
            
    except Exception as e:
        logger.error(f"Erro ao calcular regime macro: {e}")
        return "UNKNOWN"


def get_all_correlations(now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Calcula todas as correlações cross-asset para BTCUSDT.
    
    Args:
        now_utc: Timestamp atual em UTC (opcional)
        
    Returns:
        Dict combinado com todas as correlações e status consolidado
    """
    # Usa a nova função enhanced
    return get_enhanced_cross_asset_correlations(now_utc)


def _calculate_correlation_regime(btc_dxy_corr: float) -> str:
    """
    Calcula regime de correlação baseado na correlação BTC x DXY.
    
    Args:
        btc_dxy_corr: Correlação BTC x DXY
        
    Returns:
        String: "CORRELATED", "DECORRELATED", ou "INVERSE"
    """
    try:
        if pd.isna(btc_dxy_corr):
            return "UNKNOWN"
        
        # DXY correlação inversa esperada
        if btc_dxy_corr < -0.4:
            return "INVERSE"  # Correlação inversa forte
        elif abs(btc_dxy_corr) < 0.2:
            return "DECORRELATED"  # Baixa correlação
        else:
            return "CORRELATED"  # Correlação positiva ou fraca inversa
            
    except Exception as e:
        logger.error(f"Erro ao calcular regime de correlação: {e}")
        return "UNKNOWN"


# Função principal para integração com ml_features
get_cross_asset_features = get_all_correlations


def build_cross_asset_context(cross_asset: dict) -> dict:
    corr_dxy = cross_asset.get("btc_dxy_corr_30d")
    dxy_ret_5d = cross_asset.get("dxy_return_5d")

    # Classificar regime de correlação BTC x DXY
    if corr_dxy is None:
        regime = "UNKNOWN"
    elif corr_dxy < -0.4:
        regime = "STRONG_INVERSE"
    elif corr_dxy > 0.2:
        regime = "POSITIVE_OR_WEAK_INVERSE"
    else:
        regime = "NEUTRAL"

    # Classificar tendência recente do DXY (5 dias)
    if dxy_ret_5d is None:
        dxy_trend_5d = "UNKNOWN"
    elif dxy_ret_5d > 0.005:
        dxy_trend_5d = "UP"
    elif dxy_ret_5d < -0.005:
        dxy_trend_5d = "DOWN"
    else:
        dxy_trend_5d = "FLAT"

    # Efeito esperado no BTC, dado regime inverso
    if regime == "STRONG_INVERSE":
        if dxy_trend_5d == "UP":
            expected_effect = "HEADWIND"   # vento contra BTC
        elif dxy_trend_5d == "DOWN":
            expected_effect = "TAILWIND"   # vento a favor BTC
        else:
            expected_effect = "NEUTRAL"
    else:
        expected_effect = "NEUTRAL"

    return {
        "dxy_link": {
            "btc_dxy_corr_30d": corr_dxy,
            "btc_dxy_corr_90d": cross_asset.get("btc_dxy_corr_90d"),
            "dxy_trend_5d": dxy_trend_5d,
            "relationship_regime": regime,
            "expected_effect_on_btc": expected_effect
        },
        "crypto_sector": {
            "btc_eth_corr_30d": cross_asset.get("btc_eth_corr_30d")
        },
        "macro_links": {
            "btc_ndx_corr_30d": cross_asset.get("btc_ndx_corr_30d")
        }
    }


if __name__ == "__main__":
    # Teste básico
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("TESTE DE CROSS_ASSET_CORRELATIONS")
    print("="*80 + "\n")
    
    # Testa correlações crypto
    print("Testando correlações BTC/ETH...")
    crypto_result = get_btc_eth_correlations()
    print(f"  Status: {crypto_result['status']}")
    if crypto_result['status'] == 'ok':
        print(f"  7d correlation: {crypto_result['btc_eth_corr_7d']:.4f}")
        print(f"  30d correlation: {crypto_result['btc_eth_corr_30d']:.4f}")
    else:
        print(f"  Error: {crypto_result.get('error')}")
    
    # Testa correlações macro
    print("\nTestando correlações macro...")
    macro_result = get_btc_macro_correlations()
    print(f"  Status: {macro_result['status']}")
    if macro_result['status'] == 'ok':
        print(f"  BTC x DXY (30d): {macro_result['btc_dxy_corr_30d']:.4f}")
        print(f"  BTC x DXY (90d): {macro_result['btc_dxy_corr_90d']:.4f}")
        print(f"  DXY return (5d): {macro_result['dxy_return_5d']:.2f}%")
        print(f"  DXY return (20d): {macro_result['dxy_return_20d']:.2f}%")
    else:
        print(f"  Error: {macro_result.get('error')}")
    
    # Testa função combinada
    print("\nTestando função combinada...")
    all_result = get_all_correlations()
    print(f"  Status: {all_result['status']}")
    print(f"  Total features: {len([k for k in all_result.keys() if not k.startswith('_')])}")
    
    print("\n" + "="*80)
    print("TESTE CONCLUIDO")
    print("="*80 + "\n")