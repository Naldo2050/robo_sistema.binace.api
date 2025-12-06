# diagnostics/analyze_ai_results.py
# -*- coding: utf-8 -*-

"""
Script de Análise de Performance da IA.

Lê o CSV gerado por `evaluate_ai_performance.py` e gera insights estatísticos
para otimizar a lógica de trading.
"""

import pandas as pd
import numpy as np
import sys
import os

# Caminho do relatório gerado
SCRIPT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPORT_FILE = os.path.join(BASE_DIR, "ai_performance_report.csv")

def load_data():
    if not os.path.exists(REPORT_FILE):
        print(f"❌ Arquivo {REPORT_FILE} não encontrado.")
        print("   Execute primeiro: python diagnostics/evaluate_ai_performance.py")
        sys.exit(1)
    
    df = pd.read_csv(REPORT_FILE)
    print(f"✅ Dados carregados: {len(df)} recomendações.")
    return df

def print_header(title):
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print(f"{'='*60}")

def analyze_general_performance(df):
    print_header("PERFORMANCE GERAL")
    
    # Filtra apenas entradas válidas (buy/sell)
    df_trades = df[df['action'].isin(['buy', 'sell'])]
    
    if df_trades.empty:
        print("⚠️ Nenhum trade (buy/sell) encontrado para análise.")
        return

    horizons = [c for c in df.columns if c.startswith('ret_') and c.endswith('m')]
    
    for h in horizons:
        horizon_name = h.replace('ret_', '').replace('m', ' min')
        
        # Win Rate (> 0.05% para cobrir taxas, por exemplo)
        win_rate = (df_trades[h] > 0.05).mean() * 100
        avg_ret = df_trades[h].mean()
        med_ret = df_trades[h].median()
        
        print(f"\nHorizonte {horizon_name}:")
        print(f"  • Win Rate:      {win_rate:.2f}%")
        print(f"  • Retorno Médio: {avg_ret:.4f}%")
        print(f"  • Mediana:       {med_ret:.4f}%")
        
    # Taxa de Invalidação Global
    inval_rate = df_trades['hit_invalidation'].mean() * 100
    print(f"\n🚫 Taxa de Invalidação (Stop Loss): {inval_rate:.2f}%")

def analyze_correlations(df):
    print_header("ANÁLISE DE CONTEXTO E CORRELAÇÕES")
    
    df_trades = df[df['action'].isin(['buy', 'sell'])].copy()
    
    # Métrica principal para análise: Retorno em 15m (ou o que você preferir)
    target_metric = 'ret_15m'
    if target_metric not in df_trades.columns:
        target_metric = 'ret_60m' # Fallback
    
    print(f"🎯 Métrica alvo para correlação: {target_metric}\n")

    # 1. Por Ação (Buy vs Sell)
    print("--- Por Ação ---")
    print(df_trades.groupby('action')[target_metric].describe()[['count', 'mean', '50%']])

    # 2. Por Confiança da IA
    if 'confidence' in df_trades.columns:
        print("\n--- Por Confiança da IA ---")
        df_trades['conf_bucket'] = pd.cut(
            df_trades['confidence'],
            bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        grp = df_trades.groupby('conf_bucket')[target_metric].agg(
            Count='count',
            Avg_Ret='mean',
            Win_Rate=lambda x: (x > 0.05).mean()
        )
        print(grp)

    # 3. Por Regime de Mercado (se disponível)
    if 'regime' in df_trades.columns:
        print("\n--- Por Regime de Mercado ---")
        print(df_trades.groupby('regime')[target_metric].agg(
            Count='count', Avg_Ret='mean', Win_Rate=lambda x: (x > 0.05).mean()
        ))

    # 4. Por Net Flow (Positivo vs Negativo)
    if 'net_flow' in df_trades.columns:
        print("\n--- Por Fluxo (Alinhamento) ---")
        # Trade Long com Fluxo Positivo vs Negativo
        df_trades['flow_aligned'] = np.where(
            ((df_trades['action'] == 'buy') & (df_trades['net_flow'] > 0)) |
            ((df_trades['action'] == 'sell') & (df_trades['net_flow'] < 0)),
            'Aligned', 'Divergent'
        )
        print(df_trades.groupby('flow_aligned')[target_metric].agg(
            Count='count', Avg_Ret='mean', Win_Rate=lambda x: (x > 0.05).mean()
        ))

def suggest_optimizations(df):
    print_header("🤖 PROPOSTAS DE OTIMIZAÇÃO (FILTROS)")
    
    df_trades = df[df['action'].isin(['buy', 'sell'])].copy()
    target_metric = 'ret_15m' if 'ret_15m' in df_trades.columns else 'ret_60m'
    
    suggestions = []
    
    # Lógica heurística para sugerir filtros
    
    # 1. Filtro de Confiança
    low_conf_wr = df_trades[df_trades['confidence'] < 0.7][target_metric].gt(0.05).mean()
    if low_conf_wr < 0.40:
        suggestions.append(
            f"❌ IGNORAR sinais com confiança < 0.7 (Win Rate atual: {low_conf_wr:.1%})"
        )
    
    # 2. Filtro de Regime
    if 'regime' in df_trades.columns:
        regime_stats = df_trades.groupby('regime')[target_metric].mean()
        for regime, ret in regime_stats.items():
            if ret < -0.1: # Retorno médio muito negativo
                suggestions.append(
                    f"❌ EVITAR trades no regime '{regime}' (Retorno médio: {ret:.2f}%)"
                )
    
    # 3. Filtro de Alinhamento de Fluxo
    if 'net_flow' in df_trades.columns:
        divergent_wr = df_trades[
            ((df_trades['action'] == 'buy') & (df_trades['net_flow'] < 0)) |
            ((df_trades['action'] == 'sell') & (df_trades['net_flow'] > 0))
        ][target_metric].gt(0.05).mean()
        
        if divergent_wr < 0.45:
            suggestions.append(
                f"✅ EXIGIR alinhamento de Net Flow (Ignorar Buy com fluxo negativo / Sell com fluxo positivo)"
            )

    if not suggestions:
        print("✅ O sistema parece bem calibrado! Nenhuma anomalia óbvia detectada nos filtros básicos.")
    else:
        for i, s in enumerate(suggestions, 1):
            print(f"{i}. {s}")

    print("\n💡 DICA: Aplique estes filtros no 'market_orchestrator.py' ou no prompt da IA.")

if __name__ == "__main__":
    data = load_data()
    analyze_general_performance(data)
    analyze_correlations(data)
    suggest_optimizations(data)