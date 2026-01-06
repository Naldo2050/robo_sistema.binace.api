import json
import pandas as pd
import numpy as np
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_events():
    with open('dados/eventos_parsed.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_safe_nested(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d

def check_numerical_consistency(evt, idx):
    issues = []
    
    # Identificar tipo de schema
    if 'raw_event' in evt:
        # Schema Trigger
        data_source = evt['raw_event']
        context = "TRIGGER"
    elif 'ai_payload' in evt:
        # Schema Analysis
        data_source = evt['ai_payload']
        context = "ANALYSIS"
    else:
        return ["CRÍTICO: Formato desconhecido (nem raw_event nem ai_payload)"]

    # 1. Validação de Volumes e Fluxo
    if context == "TRIGGER":
        vol_total = data_source.get('volume_total', 0)
        # Validar ML Features se existirem
        ml = data_source.get('ml_features', {})
        if ml:
            vol_feats = ml.get('volume_features', {})
            # Checagens específicas de features
            pass

    elif context == "ANALYSIS":
        flow = data_source.get('flow_context', {})
        heatmap = data_source.get('liquidity_heatmap', {})
        
        # Validar Flow
        net_flow = flow.get('net_flow', 0)
        agg_buy = flow.get('aggressive_buyers', 0)
        agg_sell = flow.get('aggressive_sellers', 0)
        
        # Se agg_buy/sell forem percentuais (soma ~100)
        if agg_buy + agg_sell > 0:
            if not (99 <= (agg_buy + agg_sell) <= 101):
                issues.append(f"MODERADO: Soma de agressão != 100% ({agg_buy + agg_sell:.2f}%)")
        
        # Validar Heatmap Clusters
        for cluster in heatmap.get('clusters', []):
            buy_vol = cluster.get('buy_volume', 0)
            sell_vol = cluster.get('sell_volume', 0)
            total_vol = cluster.get('total_volume', 0)
            
            if abs((buy_vol + sell_vol) - total_vol) > (total_vol * 0.05): # 5% tolerância
                issues.append(f"LEVE: Cluster Volume mismatch (B+S={buy_vol+sell_vol:.2f} vs T={total_vol:.2f})")
                
            width = cluster.get('width', 0)
            if width <= 0:
                issues.append(f"MODERADO: Cluster com largura inválida ({width})")

    # 3. Preços
    if context == "TRIGGER":
        price = data_source.get('preco_fechamento')
        if price and price <= 0:
            issues.append(f"CRÍTICO: Preço inválido ({price})")
            
    elif context == "ANALYSIS":
        anchor = evt.get('anchor_price')
        if anchor and anchor <= 0:
            issues.append(f"CRÍTICO: Anchor Price inválido ({anchor})")

    return issues, context

def analyze_institutional_context(evt):
    analysis = {
        "scenario": "Indefinido",
        "strength": "Desconhecida",
        "action": "Ficar fora",
        "reason": []
    }
    
    # Tentar extrair flow context de onde estiver
    flow = {}
    if 'ai_payload' in evt:
        flow = evt['ai_payload'].get('flow_context', {})
    elif 'raw_event' in evt and 'ai_payload' in evt['raw_event']:
        flow = evt['raw_event']['ai_payload'].get('flow_context', {})
        
    if flow:
        imbalance = flow.get('flow_imbalance', 0)
        absorption = flow.get('absorption_type', 'None')
        whale = flow.get('whale_activity', {})
        
        # Cenário baseado em absorção e imbalance
        if absorption != "None":
            analysis['scenario'] = f"{absorption}"
        elif imbalance > 0.3:
            analysis['scenario'] = "Agressão Compradora"
        elif imbalance < -0.3:
            analysis['scenario'] = "Agressão Vendedora"
        else:
            analysis['scenario'] = "Equilíbrio"
            
        # Força
        analysis['strength'] = f"Imb: {imbalance:.2f}"
        if whale:
            whale_delta = whale.get('whale_delta', 0)
            analysis['strength'] += f" | WhaleDelta: {whale_delta:.1f}"
            
        # Ação baseada em imbalance simples (lógica placeholder)
        if imbalance > 0.4:
            analysis['action'] = "COMPRA"
        elif imbalance < -0.4:
            analysis['action'] = "VENDA"
            
    return analysis

def main():
    events = load_events()
    print(f"Iniciando auditoria v2 em {len(events)} eventos...\n")
    
    for i, evt in enumerate(events):
        print(f"=== JANELA {i+1} ===")
        
        # 1. Consistência Numérica
        issues, context = check_numerical_consistency(evt, i)
        print(f"  Tipo: {evt.get('tipo_evento', 'N/A')} ({context})")
        
        print(">> Consistência Numérica:")
        if issues:
            for issue in issues:
                print(f"  [X] {issue}")
        else:
            print(f"  [OK] Dados consistentes.")
            
        # 2. Leitura Institucional
        inst = analyze_institutional_context(evt)
        print("\n>> Leitura Institucional:")
        print(f"  Cenário: {inst['scenario']}")
        print(f"  Força: {inst['strength']}")
        print(f"  Ação Sugerida: {inst['action']}")
        
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
