import json
import re
from typing import List, Dict, Any

LOG_FILE = 'dados/eventos_visuais.log'

def load_recent_ai_events(limit: int = 10) -> List[Dict[str, Any]]:
    events = []
    
    # Leitura reversa otimizada seria ideal, mas vamos ler tudo e filtrar para garantir parse correto
    # dado o formato misto do arquivo.
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex para capturar blocos JSON que tenham "tipo_evento": "AI_ANALYSIS"
    # O log tem separadores, podemos tentar splitar por linhas ou achar blocos.
    # Assumindo que cada JSON está em uma linha ou bloco formatado.
    # O dump anterior mostrou JSON pretty-printed.
    
    # Estratégia: Encontrar "tipo_evento": "AI_ANALYSIS" e extrair o objeto JSON envolvente.
    # Como são pretty-printed, precisamos contar chaves balanceadas {}
    
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        try:
            # Pula whitespace
            while pos < len(content) and content[pos].isspace():
                pos += 1
            
            if pos >= len(content):
                break
            
            # Tenta decodificar um objeto JSON a partir de 'pos'
            if content[pos] == '{':
                obj, new_pos = decoder.raw_decode(content, pos)
                pos = new_pos
                
                # Verifica se é o evento que queremos
                if isinstance(obj, dict):
                    if obj.get('tipo_evento') == 'AI_ANALYSIS':
                        events.append(obj)
            else:
                # Se não é início de JSON, avança um caracter (robustez para lixo no log)
                pos += 1
                
        except json.JSONDecodeError:
            pos += 1
            continue

    return events[-limit:]

def audit_event(idx: int, evt: Dict[str, Any]):
    issues = []
    eid = evt.get('event_id', 'N/A')
    payload = evt.get('ai_payload', {})
    
    print(f"\n=== EVENTO {idx+1} | ID: {eid} ===")
    
    # --- 1. PRICE ACTION CHECK ---
    price_ctx = payload.get('price_context', {})
    pa = price_ctx.get('price_action', {})
    
    if not pa:
        issues.append("Price Action block MISSING")
        print("PERDA: Bloco price_action não encontrado.")
    else:
        # Extrair valores
        cp = pa.get('close_position')
        bp = pa.get('candle_body_pct')
        
        # Validar ranges
        if cp is not None:
            if not (0 <= cp <= 1.01): # pequena tolerância float
                issues.append(f"Close Position fora de range (0-1): {cp}")
        
        if bp is not None:
            if bp < 0:
                issues.append(f"Body Pct negativo: {bp}")
                
        # Consistência lógica básica (se tiver OHLC)
        ohlc = price_ctx.get('ohlc', {})
        if ohlc:
            op = ohlc.get('open')
            cl = ohlc.get('close')
            hi = ohlc.get('high')
            lo = ohlc.get('low')
            
            if all(v is not None for v in [op, cl, hi, lo]):
                # Recalcula para conferir
                rng = hi - lo
                if rng > 0:
                    calc_cp = (cl - lo) / rng
                    if cp is not None and abs(calc_cp - cp) > 0.05:
                        issues.append(f"Divergência ClosePosition: Calc={calc_cp:.2f} vs Log={cp:.2f}")
    
    if not issues:
        print("[PRICE ACTION] OK")
    else:
        print(f"[PRICE ACTION] PROBLEMAS: {issues}")

    # --- 2. DEPTH CHECK ---
    ob_ctx = payload.get('orderbook_context', {})
    depth = ob_ctx.get('depth_metrics', {})
    
    issues_depth = []
    if not depth:
        # Pode ser que não tenha sido gerado se orderbook falhou, mas deveria estar vazio {} e não None
        if 'depth_metrics' not in ob_ctx:
             issues_depth.append("Depth Metrics block MISSING")
    else:
        bid5 = depth.get('bid_liquidity_top5', 0)
        ask5 = depth.get('ask_liquidity_top5', 0)
        dimb = depth.get('depth_imbalance', 0)
        
        if bid5 < 0 or ask5 < 0:
             issues_depth.append(f"Liquidez negativa: B={bid5}, A={ask5}")
             
        # Checar consistência do imbalance
        # (Bid - Ask) / (Bid + Ask)
        total = bid5 + ask5
        if total > 0:
            calc_dimb = (bid5 - ask5) / total
            if abs(calc_dimb - dimb) > 0.05:
                 issues_depth.append(f"Divergência Imbalance: Calc={calc_dimb:.2f} vs Log={dimb:.2f}")
        
    if not issues_depth:
        print("[DEPTH] OK")
    else:
        print(f"[DEPTH] PROBLEMAS: {issues_depth}")
        
    # --- 3. ANÁLISE DE MERCADO ---
    print("\n>> ANÁLISE DE MERCADO:")
    market_read = analyze_market_context(evt, pa, depth)
    print(market_read)

def analyze_market_context(evt, pa, depth):
    # Extração de dados
    msg = []
    
    # 1. Price Action
    sentiment_pa = "Neutro"
    
    # Se pa for None, aborta
    if not pa:
        return "Dados de Price Action indisponíveis."
        
    cp = pa.get('close_position', 0.5)
    bp = pa.get('candle_body_pct', 0)
    
    if bp > 0.05: # Tem corpo relevante
        if cp > 0.7:
            sentiment_pa = "Força Compradora (Fechou no Topo)"
        elif cp < 0.3:
            sentiment_pa = "Força Vendedora (Fechou no Fundo)"
        else:
            sentiment_pa = "Indecisão (Corpo no meio)"
    else:
        sentiment_pa = "Doji / Indecisão"
        
    msg.append(f"Candle: {sentiment_pa} (Pos={cp:.2f}, Body={bp:.2f}%)")
    
    # 2. Depth
    sentiment_depth = "Equilibrado"
    dimb = depth.get('depth_imbalance', 0)
    
    if dimb > 0.2:
        sentiment_depth = f"Suporte Passivo Forte (+{dimb:.2f})"
    elif dimb < -0.2:
        sentiment_depth = f"Resistência Passiva Forte ({dimb:.2f})"
        
    msg.append(f"Depth: {sentiment_depth}")
    
    # 3. Fluxo (CVD / Whale)
    # Pegar do payload
    payload = evt.get('ai_payload', {})
    flow = payload.get('flow_context', {})
    net_flow = flow.get('net_flow', 0)
    whale = flow.get('whale_activity', {}).get('whale_delta', 0)
    
    # 4. Síntese
    action = "FICAR FORA"
    reason = "Sinais mistos"
    
    # Cenário de Compra
    if (cp > 0.6) and (whale > 0) and (dimb > -0.1): # Candle up + Whale Buy + Depth não muito contra
        action = "COMPRA"
        reason = "Fluxo Whale + Price Action confirmando"
        
    # Cenário de Venda
    elif (cp < 0.4) and (whale < 0) and (dimb < 0.1): 
        action = "VENDA"
        reason = "Fluxo Whale Vendedor + Price Action confirmando"
        
    msg.append(f"Fluxo: WhaleDelta={whale:.1f}")
    msg.append(f"DECISÃO: {action} ({reason})")
    
    return "\n".join(msg)

def main():
    print("Iniciando auditoria de novas features...")
    events = load_recent_ai_events(10)
    print(f"Carregados {len(events)} eventos AI_ANALYSIS recentes.")
    
    for i, evt in enumerate(events):
        audit_event(i, evt)

if __name__ == "__main__":
    main()
