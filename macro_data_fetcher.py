# macro_data_fetcher.py
"""
Módulo estendido para busca de dados macro e cross-asset.
Suporte adicional para:
- VIX (fear index)
- Crypto Dominance 
- Treasury Yields
- Commodities (Gold, Oil)
- Regime Detection

Fontes: yfinance, CoinGecko, APIs locais
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger("MacroDataFetcher")

# Tickers de fallback por ativo (expandido)
_FALLBACK_TICKERS = {
    "BTC-USD": ["BTC-USD", "BTCUSD=X", "BTC=F"],
    "DXY": ["DX=F", "DXY", "DX-Y.NYB"],  # Priorizar Yahoo Finance
    "NDX": ["^IXIC", "QQQ", "NDX"],
    "SPX": ["^GSPC", "SPY", "SPX"],
    "VIX": ["^VIX", "VIX", "VIXC"],           # Fear Index
    "US10Y": ["^TNX", "TNX", "US10Y"],       # Treasury 10Y
    "US2Y": ["^TNY", "TNY", "US2Y"],         # Treasury 2Y
    "GOLD": ["GC=F", "XAUUSD=X", "GOLD"],    # Gold futures/spot
    "OIL": ["CL=F", "USO", "OIL"],           # WTI Oil futures/ETF
}

# Mapeamento de crypto dominance (CoinGecko)
_CRYPTO_DOMINANCE_IDS = {
    "bitcoin": "btc",
    "ethereum": "eth", 
    "tether": "usdt"
}

def _fetch_yfinance_data_with_fallbacks(name: str, period: str = "90d", interval: str = "1d") -> pd.DataFrame:
    """
    Busca dados históricos do yfinance com fallbacks robustos (versão estendida).
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


def fetch_crypto_dominance() -> Dict[str, Any]:
    """
    Busca dados de dominância crypto via CoinGecko API (gratuita).
    
    Returns:
        Dict com btc_dominance, eth_dominance, usdt_dominance
    """
    result = {
        "status": "failed",
        "btc_dominance": None,
        "eth_dominance": None,
        "usdt_dominance": None,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        # CoinGecko API - dados globais
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            global_data = data.get("data", {})
            
            result["btc_dominance"] = float(global_data.get("market_cap_percentage", {}).get("btc", 0))
            result["eth_dominance"] = float(global_data.get("market_cap_percentage", {}).get("eth", 0))
            result["usdt_dominance"] = float(global_data.get("market_cap_percentage", {}).get("usdt", 0))
            result["status"] = "ok"
            
            logger.info(f"✅ Dominância crypto: BTC={result['btc_dominance']:.2f}%, ETH={result['eth_dominance']:.2f}%, USDT={result['usdt_dominance']:.2f}%")
        else:
            logger.warning(f"CoinGecko API retornou status {response.status_code}")
            
    except Exception as e:
        logger.error(f"Erro ao buscar dominância crypto: {e}")
    
    return result


def fetch_vix_data(period: str = "30d") -> Dict[str, Any]:
    """
    Busca dados do VIX (Fear Index).
    
    Args:
        period: Período de dados (ex: 30d, 90d)
        
    Returns:
        Dict com vix_current, vix_change_1d, histórico
    """
    result = {
        "status": "failed",
        "vix_current": None,
        "vix_change_1d": None,
        "historical": pd.DataFrame(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        # Busca dados históricos
        vix_df = _fetch_yfinance_data_with_fallbacks("VIX", period=period)
        
        if vix_df.empty:
            raise ValueError("Dados do VIX indisponíveis")
        
        # Dados atuais
        result["vix_current"] = float(vix_df['close'].iloc[-1])
        
        # Variação 1 dia
        if len(vix_df) >= 2:
            prev_close = float(vix_df['close'].iloc[-2])
            current_close = result["vix_current"]
            result["vix_change_1d"] = float((current_close - prev_close) / prev_close * 100)
        
        result["historical"] = vix_df
        result["status"] = "ok"
        
        logger.info(f"✅ VIX: atual={result['vix_current']:.2f}, variação 1d={result['vix_change_1d']:.2f}%")
        
    except Exception as e:
        logger.error(f"Erro ao buscar dados do VIX: {e}")
    
    return result


def fetch_treasury_yields(period: str = "30d") -> Dict[str, Any]:
    """
    Busca dados de Treasury Yields (US 10Y e 2Y).
    
    Args:
        period: Período de dados
        
    Returns:
        Dict com us10y_yield, us10y_change_1d, us2y_yield, us2y_change_1d
    """
    result = {
        "status": "failed",
        "us10y_yield": None,
        "us10y_change_1d": None,
        "us2y_yield": None,
        "us2y_change_1d": None,
        "historical_10y": pd.DataFrame(),
        "historical_2y": pd.DataFrame(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        # Busca dados do US 10Y
        us10y_df = _fetch_yfinance_data_with_fallbacks("US10Y", period=period)
        us2y_df = _fetch_yfinance_data_with_fallbacks("US2Y", period=period)
        
        if not us10y_df.empty:
            result["us10y_yield"] = float(us10y_df['close'].iloc[-1])
            
            if len(us10y_df) >= 2:
                prev_close = float(us10y_df['close'].iloc[-2])
                current_close = result["us10y_yield"]
                result["us10y_change_1d"] = float((current_close - prev_close) / prev_close * 100)
            
            result["historical_10y"] = us10y_df
        
        if not us2y_df.empty:
            result["us2y_yield"] = float(us2y_df['close'].iloc[-1])
            
            if len(us2y_df) >= 2:
                prev_close = float(us2y_df['close'].iloc[-2])
                current_close = result["us2y_yield"]
                result["us2y_change_1d"] = float((current_close - prev_close) / prev_close * 100)
            
            result["historical_2y"] = us2y_df
        
        if result["us10y_yield"] is not None or result["us2y_yield"] is not None:
            result["status"] = "ok"
            logger.info(f"✅ Treasury Yields: 10Y={result['us10y_yield']:.3f}%, 2Y={result['us2y_yield']:.3f}%")
        else:
            raise ValueError("Nenhum dado de Treasury Yield disponível")
            
    except Exception as e:
        logger.error(f"Erro ao buscar Treasury Yields: {e}")
    
    return result


def fetch_commodities_data(period: str = "90d") -> Dict[str, Any]:
    """
    Busca dados de commodities (Gold, Oil).
    
    Args:
        period: Período de dados
        
    Returns:
        Dict com gold_price, gold_change_1d, oil_price, oil_change_1d
    """
    result = {
        "status": "failed",
        "gold_price": None,
        "gold_change_1d": None,
        "oil_price": None,
        "oil_change_1d": None,
        "historical_gold": pd.DataFrame(),
        "historical_oil": pd.DataFrame(),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        # Busca dados do Gold
        gold_df = _fetch_yfinance_data_with_fallbacks("GOLD", period=period)
        oil_df = _fetch_yfinance_data_with_fallbacks("OIL", period=period)
        
        if not gold_df.empty:
            result["gold_price"] = float(gold_df['close'].iloc[-1])
            
            if len(gold_df) >= 2:
                prev_close = float(gold_df['close'].iloc[-2])
                current_close = result["gold_price"]
                result["gold_change_1d"] = float((current_close - prev_close) / prev_close * 100)
            
            result["historical_gold"] = gold_df
        
        if not oil_df.empty:
            result["oil_price"] = float(oil_df['close'].iloc[-1])
            
            if len(oil_df) >= 2:
                prev_close = float(oil_df['close'].iloc[-2])
                current_close = result["oil_price"]
                result["oil_change_1d"] = float((current_close - prev_close) / prev_close * 100)
            
            result["historical_oil"] = oil_df
        
        if result["gold_price"] is not None or result["oil_price"] is not None:
            result["status"] = "ok"
            logger.info(f"✅ Commodities: Gold=${result['gold_price']:.2f}, Oil=${result['oil_price']:.2f}")
        else:
            raise ValueError("Nenhum dado de commodity disponível")
            
    except Exception as e:
        logger.error(f"Erro ao buscar dados de commodities: {e}")
    
    return result


def calculate_macro_regime(
    vix_data: Dict[str, Any],
    dominance_data: Dict[str, Any],
    treasury_data: Dict[str, Any]
) -> str:
    """
    Calcula regime macro baseado em múltiplos indicadores.
    
    Args:
        vix_data: Dados do VIX
        dominance_data: Dados de dominância crypto
        treasury_data: Dados de Treasury Yields
        
    Returns:
        String: "RISK_ON", "RISK_OFF", ou "TRANSITION"
    """
    try:
        risk_score = 0
        factors = 0
        
        # VIX: > 25 = risk off, < 15 = risk on
        vix_current = vix_data.get("vix_current")
        if vix_current is not None:
            factors += 1
            if vix_current > 25:
                risk_score += 2
            elif vix_current > 20:
                risk_score += 1
            elif vix_current < 15:
                risk_score -= 1
            elif vix_current < 12:
                risk_score -= 2
        
        # BTC Dominance: > 50% = risk off, < 40% = risk on
        btc_dom = dominance_data.get("btc_dominance")
        if btc_dom is not None:
            factors += 1
            if btc_dom > 50:
                risk_score += 1
            elif btc_dom < 40:
                risk_score -= 1
        
        # Treasury Yields: subida = risk off
        us10y_change = treasury_data.get("us10y_change_1d")
        if us10y_change is not None:
            factors += 1
            if us10y_change > 0.05:  # Subida significativa
                risk_score += 1
            elif us10y_change < -0.05:  # Queda significativa
                risk_score -= 1
        
        if factors == 0:
            return "UNKNOWN"
        
        # Classifica regime
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


def calculate_correlation_regime(
    btc_dxy_corr: float,
    btc_vix_corr: float = None,
    btc_gold_corr: float = None
) -> str:
    """
    Calcula regime de correlação baseado em correlações BTC.
    
    Args:
        btc_dxy_corr: Correlação BTC x DXY
        btc_vix_corr: Correlação BTC x VIX (opcional)
        btc_gold_corr: Correlação BTC x Gold (opcional)
        
    Returns:
        String: "CORRELATED", "DECORRELATED", ou "INVERSE"
    """
    try:
        correlations = []
        
        # DXY: correlação inversa esperada (-0.3 a -0.7)
        if pd.notna(btc_dxy_corr):
            if btc_dxy_corr < -0.3:
                correlations.append("INVERSE")
            elif abs(btc_dxy_corr) < 0.2:
                correlations.append("DECORRELATED")
            else:
                correlations.append("CORRELATED")
        
        # VIX: correlação inversa esperada
        if pd.notna(btc_vix_corr):
            if btc_vix_corr < -0.2:
                correlations.append("INVERSE")
            elif abs(btc_vix_corr) < 0.2:
                correlations.append("DECORRELATED")
            else:
                correlations.append("CORRELATED")
        
        # Gold: correlação variável mas geralmente positiva
        if pd.notna(btc_gold_corr):
            if btc_gold_corr > 0.2:
                correlations.append("CORRELATED")
            elif abs(btc_gold_corr) < 0.2:
                correlations.append("DECORRELATED")
            else:
                correlations.append("INVERSE")
        
        if not correlations:
            return "UNKNOWN"
        
        # Voto majoritário
        unique_corrs = list(set(correlations))
        if len(unique_corrs) == 1:
            return unique_corrs[0]
        else:
            # Em caso de empate, usa DXY como decisor
            return correlations[0] if correlations else "DECORRELATED"
            
    except Exception as e:
        logger.error(f"Erro ao calcular regime de correlação: {e}")
        return "UNKNOWN"


def fetch_all_macro_data() -> Dict[str, Any]:
    """
    Busca todos os dados macro de uma vez.
    
    Returns:
        Dict com todos os dados macro
    """
    logger.info("🔄 Iniciando busca de todos os dados macro...")
    
    # Busca dados em paralelo (simulado com calls sequenciais)
    results = {}
    
    # Crypto Dominance
    logger.debug("Buscando dominância crypto...")
    results["dominance"] = fetch_crypto_dominance()
    
    # VIX
    logger.debug("Buscando dados do VIX...")
    results["vix"] = fetch_vix_data()
    
    # Treasury Yields
    logger.debug("Buscando Treasury Yields...")
    results["treasury"] = fetch_treasury_yields()
    
    # Commodities
    logger.debug("Buscando dados de commodities...")
    results["commodities"] = fetch_commodities_data()
    
    # Calcula regimes
    results["macro_regime"] = calculate_macro_regime(
        results["vix"], 
        results["dominance"], 
        results["treasury"]
    )
    
    logger.info(f"✅ Dados macro coletados. Regime: {results['macro_regime']}")
    
    return results


if __name__ == "__main__":
    # Teste básico
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("TESTE DE MACRO_DATA_FETCHER")
    print("="*80 + "\n")
    
    # Testa busca de todos os dados
    macro_data = fetch_all_macro_data()
    
    print("📊 RESULTADOS:")
    print(f"  Dominance Status: {macro_data['dominance']['status']}")
    print(f"  VIX Status: {macro_data['vix']['status']}")
    print(f"  Treasury Status: {macro_data['treasury']['status']}")
    print(f"  Commodities Status: {macro_data['commodities']['status']}")
    print(f"  Macro Regime: {macro_data['macro_regime']}")
    
    print("\n" + "="*80)
    print("TESTE CONCLUÍDO")
    print("="*80 + "\n")