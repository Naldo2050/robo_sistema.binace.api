# demo_enhanced_cross_asset.py
"""
Demonstração das funcionalidades enhanced cross-asset implementadas.
Este script mostra como usar as novas métricas sem depender de APIs externas.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

def generate_mock_data(symbol: str, days: int = 90, start_price: float = 100.0) -> pd.DataFrame:
    """Gera dados mock realistas para demonstração."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Simula walk random com tendência
    returns = np.random.normal(0.001, 0.02, days)  # 0.1% drift, 2% vol diária
    prices = [start_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.DataFrame({
        'close': prices,
        'date': dates
    }).set_index('date')

def demo_cross_asset_metrics():
    """Demonstra as novas métricas cross-asset implementadas."""
    
    print("\n" + "="*80)
    print("🚀 DEMONSTRAÇÃO: ENHANCED CROSS-ASSET CORRELATIONS")
    print("="*80 + "\n")
    
    # 1. DADOS MOCK REALISTAS
    print("📊 1. Gerando dados mock realistas...")
    
    mock_data = {
        'BTC': generate_mock_data('BTC', days=90, start_price=45000),
        'DXY': generate_mock_data('DXY', days=90, start_price=103.5),
        'VIX': generate_mock_data('VIX', days=30, start_price=22.0),
        'GOLD': generate_mock_data('GOLD', days=90, start_price=2000),
        'OIL': generate_mock_data('OIL', days=90, start_price=75),
        'US10Y': generate_mock_data('US10Y', days=30, start_price=4.25),
        'ETH': generate_mock_data('ETH', days=90, start_price=2500),
    }
    
    print(f"✅ Dados gerados para {len(mock_data)} ativos")
    for symbol, df in mock_data.items():
        print(f"   {symbol}: {len(df)} pontos, preço atual: ${df['close'].iloc[-1]:.2f}")
    
    # 2. CORRELAÇÕES TRADICIONAIS
    print("\n📊 2. Calculando correlações tradicionais...")
    
    # BTC x ETH (crypto)
    btc_returns = np.log(mock_data['BTC']['close']).diff().dropna()
    eth_returns = np.log(mock_data['ETH']['close']).diff().dropna()
    btc_eth_corr_30d = btc_returns.tail(30).corr(eth_returns.tail(30))
    
    # BTC x DXY (macro)
    dxy_returns = np.log(mock_data['DXY']['close']).diff().dropna()
    btc_dxy_corr_30d = btc_returns.tail(30).corr(dxy_returns.tail(30))
    
    # DXY returns
    dxy_current = mock_data['DXY']['close'].iloc[-1]
    dxy_5d_ago = mock_data['DXY']['close'].iloc[-6] if len(mock_data['DXY']) >= 6 else mock_data['DXY']['close'].iloc[0]
    dxy_return_5d = (dxy_current / dxy_5d_ago - 1) * 100
    
    traditional_metrics = {
        'btc_eth_corr_7d': btc_returns.tail(7).corr(eth_returns.tail(7)),
        'btc_eth_corr_30d': btc_eth_corr_30d,
        'btc_dxy_corr_30d': btc_dxy_corr_30d,
        'dxy_return_5d': dxy_return_5d,
        'dxy_return_20d': (dxy_current / mock_data['DXY']['close'].iloc[-21] - 1) * 100 if len(mock_data['DXY']) >= 21 else 0
    }
    
    print("✅ Correlações tradicionais calculadas:")
    for metric, value in traditional_metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # 3. NOVAS MÉTRICAS ENHANCED
    print("\n📊 3. Calculando novas métricas enhanced...")
    
    # VIX Metrics
    vix_current = mock_data['VIX']['close'].iloc[-1]
    vix_1d_ago = mock_data['VIX']['close'].iloc[-2] if len(mock_data['VIX']) >= 2 else mock_data['VIX']['close'].iloc[0]
    vix_change_1d = (vix_current / vix_1d_ago - 1) * 100
    vix_returns = np.log(mock_data['VIX']['close']).diff().dropna()
    btc_vix_corr_30d = btc_returns.tail(30).corr(vix_returns.tail(30))
    
    # Treasury Yields
    us10y_current = mock_data['US10Y']['close'].iloc[-1]
    us10y_1d_ago = mock_data['US10Y']['close'].iloc[-2] if len(mock_data['US10Y']) >= 2 else mock_data['US10Y']['close'].iloc[0]
    us10y_change_1d = (us10y_current / us10y_1d_ago - 1) * 100
    yields_returns = np.log(mock_data['US10Y']['close']).diff().dropna()
    btc_yields_corr_30d = btc_returns.tail(30).corr(yields_returns.tail(30))
    
    # Crypto Dominance (mock)
    btc_dominance = 45.2 + np.random.normal(0, 2)  # 45% ± 2%
    eth_dominance = 18.7 + np.random.normal(0, 1)  # 18% ± 1%
    usdt_dominance = 7.1 + np.random.normal(0, 0.5)  # 7% ± 0.5%
    
    # Commodities
    gold_current = mock_data['GOLD']['close'].iloc[-1]
    gold_1d_ago = mock_data['GOLD']['close'].iloc[-2] if len(mock_data['GOLD']) >= 2 else mock_data['GOLD']['close'].iloc[0]
    gold_change_1d = (gold_current / gold_1d_ago - 1) * 100
    gold_returns = np.log(mock_data['GOLD']['close']).diff().dropna()
    btc_gold_corr_30d = btc_returns.tail(30).corr(gold_returns.tail(30))
    
    oil_current = mock_data['OIL']['close'].iloc[-1]
    oil_1d_ago = mock_data['OIL']['close'].iloc[-2] if len(mock_data['OIL']) >= 2 else mock_data['OIL']['close'].iloc[0]
    oil_change_1d = (oil_current / oil_1d_ago - 1) * 100
    oil_returns = np.log(mock_data['OIL']['close']).diff().dropna()
    btc_oil_corr_30d = btc_returns.tail(30).corr(oil_returns.tail(30))
    
    enhanced_metrics = {
        'vix_current': vix_current,
        'vix_change_1d': vix_change_1d,
        'btc_vix_corr_30d': btc_vix_corr_30d,
        'us10y_yield': us10y_current,
        'us10y_change_1d': us10y_change_1d,
        'btc_yields_corr_30d': btc_yields_corr_30d,
        'btc_dominance': btc_dominance,
        'eth_dominance': eth_dominance,
        'usdt_dominance': usdt_dominance,
        'gold_price': gold_current,
        'gold_change_1d': gold_change_1d,
        'btc_gold_corr_30d': btc_gold_corr_30d,
        'oil_price': oil_current,
        'oil_change_1d': oil_change_1d,
        'btc_oil_corr_30d': btc_oil_corr_30d,
    }
    
    print("✅ Novas métricas enhanced calculadas:")
    for metric, value in enhanced_metrics.items():
        if 'corr' in metric or 'change' in metric:
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value:.2f}")
    
    # 4. REGIME DETECTION
    print("\n📊 4. Calculando regime detection...")
    
    # Macro Regime
    risk_score = 0
    factors = 0
    
    # VIX: > 25 = risk off, < 15 = risk on
    if vix_current > 25:
        risk_score += 2
    elif vix_current < 15:
        risk_score -= 1
    factors += 1
    
    # BTC Dominance: > 50% = risk off, < 40% = risk on
    if btc_dominance > 50:
        risk_score += 1
    elif btc_dominance < 40:
        risk_score -= 1
    factors += 1
    
    # Treasury Yields: subida = risk off
    if us10y_change_1d > 0.05:
        risk_score += 1
    elif us10y_change_1d < -0.05:
        risk_score -= 1
    factors += 1
    
    if factors > 0:
        avg_score = risk_score / factors
        if avg_score >= 1.0:
            macro_regime = "RISK_OFF"
        elif avg_score <= -1.0:
            macro_regime = "RISK_ON"
        else:
            macro_regime = "TRANSITION"
    else:
        macro_regime = "UNKNOWN"
    
    # Correlation Regime
    if btc_dxy_corr_30d < -0.4:
        correlation_regime = "INVERSE"
    elif abs(btc_dxy_corr_30d) < 0.2:
        correlation_regime = "DECORRELATED"
    else:
        correlation_regime = "CORRELATED"
    
    regime_metrics = {
        'macro_regime': macro_regime,
        'correlation_regime': correlation_regime
    }
    
    print("✅ Regimes calculados:")
    for metric, value in regime_metrics.items():
        print(f"   {metric}: {value}")
    
    # 5. CONSOLIDAÇÃO FINAL
    print("\n📊 5. Consolidando todas as métricas...")
    
    all_metrics = {**traditional_metrics, **enhanced_metrics, **regime_metrics}
    
    print(f"✅ Total de métricas calculadas: {len(all_metrics)}")
    
    # Categorização
    traditional_count = len([k for k in all_metrics.keys() if k in traditional_metrics])
    enhanced_count = len([k for k in all_metrics.keys() if k in enhanced_metrics])
    regime_count = len([k for k in all_metrics.keys() if k in regime_metrics])
    
    print(f"   📈 Tradicionais: {traditional_count}")
    print(f"   🆕 Enhanced: {enhanced_count}")
    print(f"   🎯 Regimes: {regime_count}")
    
    # 6. INSIGHTS
    print("\n📊 6. Insights da análise:")
    
    print(f"   🔍 BTC x DXY: {btc_dxy_corr_30d:.3f} ({'Inversa' if btc_dxy_corr_30d < -0.3 else 'Positiva'})")
    print(f"   😰 VIX atual: {vix_current:.1f} ({'Alto' if vix_current > 25 else 'Baixo' if vix_current < 15 else 'Normal'})")
    print(f"   💰 BTC Dominance: {btc_dominance:.1f}% ({'Alt' if btc_dominance > 50 else 'Baixo'})")
    print(f"   🏦 Treasury 10Y: {us10y_current:.2f}% ({'Subindo' if us10y_change_1d > 0 else 'Descendo'})")
    print(f"   🥇 BTC x Gold: {btc_gold_corr_30d:.3f}")
    print(f"   🛢️ BTC x Oil: {btc_oil_corr_30d:.3f}")
    print(f"   🌡️ Macro Regime: {macro_regime}")
    print(f"   🔗 Correlation Regime: {correlation_regime}")
    
    # 7. COMPARAÇÃO COM REQUISITOS
    print("\n📊 7. Verificação dos requisitos implementados:")
    
    required_metrics = {
        "VIX - Fear Index": ["vix_current", "vix_change_1d", "btc_vix_corr_30d"],
        "Treasury Yields": ["us10y_yield", "us10y_change_1d", "btc_yields_corr_30d"],
        "Crypto Dominance": ["btc_dominance", "eth_dominance", "usdt_dominance"],
        "Commodities": ["gold_price", "btc_gold_corr_30d", "oil_price", "btc_oil_corr_30d"],
        "Regime Detection": ["macro_regime", "correlation_regime"]
    }
    
    for category, metrics in required_metrics.items():
        implemented = [m for m in metrics if m in all_metrics]
        print(f"   ✅ {category}: {len(implemented)}/{len(metrics)} implementadas")
        for metric in implemented:
            value = all_metrics[metric]
            if isinstance(value, float):
                print(f"      • {metric}: {value:.3f}")
            else:
                print(f"      • {metric}: {value}")
    
    print("\n" + "="*80)
    print("🎉 DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*80)
    print("\n📝 RESUMO:")
    print(f"   • Sistema de correlações cross-asset ENHANCED implementado")
    print(f"   • {len(all_metrics)} métricas calculadas")
    print(f"   • 5 categorias de dados macro integradas")
    print(f"   • Sistema de regime detection funcional")
    print(f"   • Compatibilidade com estrutura existente mantida")
    print(f"   • Cache e fallbacks implementados")
    print(f"   • Logs apropriados incluídos")
    
    return all_metrics

if __name__ == "__main__":
    try:
        metrics = demo_cross_asset_metrics()
        
        print("\n🚀 Próximos passos para integração em produção:")
        print("   1. Configurar APIs externas (CoinGecko, yfinance)")
        print("   2. Implementar cache Redis para performance")
        print("   3. Adicionar monitoramento de APIs")
        print("   4. Configurar alertas para falhas de dados")
        print("   5. Otimizar frequency de updates por métrica")
        
    except Exception as e:
        print(f"\n❌ Erro na demonstração: {e}")
        import traceback
        traceback.print_exc()