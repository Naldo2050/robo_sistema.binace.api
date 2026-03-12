# log_formatter.py - VERSÃO CORRIGIDA

from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("log_formatter")


def format_flow_log(event: dict, previous_event: dict = None) -> str:
    """
    Formata log de fluxo com clareza sobre delta vs CVD.
    
    Args:
        event: Evento com métricas de fluxo
        previous_event: CVD da janela anterior (opcional)
    
    Returns:
        String formatada para log
    """
    order_flow = event.get("order_flow", {})
    cvd = event.get("cvd", 0.0)
    window_delta = order_flow.get("net_flow_1m", 0.0)
    
    # 🆕 Timestamp CORRIGIDO - suporta múltiplos formatos
    time_str = "N/A"
    
    # Tentar pegar timestamp de várias fontes
    time_index = event.get("time_index", {})
    if isinstance(time_index, dict):
        # Preferência: local_time
        time_str = time_index.get("local_time") or \
                   time_index.get("timestamp_local") or \
                   time_index.get("timestamp_utc") or \
                   time_index.get("timestamp_ny")
    
    # Fallback para campos diretos
    if not time_str or time_str == "N/A":
        time_str = event.get("timestamp") or \
                   event.get("timestamp_utc") or \
                   event.get("timestamp_local") or \
                   "N/A"
    
    # Se ainda for N/A, tentar formatar epoch_ms
    if not time_str or time_str == "N/A":
        epoch_ms = event.get("epoch_ms") or event.get("event_epoch_ms")
        if epoch_ms and isinstance(epoch_ms, (int, float)):
            try:
                time_str = datetime.fromtimestamp(epoch_ms / 1000).strftime("%H:%M:%S")
            except:
                time_str = "N/A"
    
    # Se time_index tem timestamp_utc, extrair só a hora
    if isinstance(time_str, str) and len(time_str) > 8 and 'T' in time_str:
        # Formato ISO: 2025-01-10T14:10:00.123456Z
        try:
            if 'Z' in time_str:
                time_str = time_str.split('T')[1].split('.')[0]  # Pega só HH:MM:SS
            elif '+' in time_str or '-' in time_str:
                time_str = time_str.split('T')[1].split('+')[0].split('-')[0].split('.')[0]
        except:
            pass
    
    # Linha básica
    log_parts = [
        f"{'='*80}",
        f"🕐 {time_str}",
        f"{'─'*80}",
        f"📊 FLUXO DA JANELA (últimos 60s):",
        f"   Net Flow:        {window_delta:+,.2f} USD",
        f"   Buy Volume:      {order_flow.get('buy_volume', 0):+,.2f} USD",
        f"   Sell Volume:     {order_flow.get('sell_volume', 0):+,.2f} USD",
        f"   Flow Imbalance:  {order_flow.get('flow_imbalance', 0):+.4f}",
        f"",
        f"💰 CVD ACUMULADO (desde início do dia):",
        f"   CVD Total:       {cvd:+,.4f} BTC",
    ]
    
    # Se temos CVD anterior, mostrar incremento
    if previous_event is not None:
        previous_cvd = previous_event.get("cvd", 0.0)
        delta_cvd = cvd - previous_cvd
        log_parts.extend([
            f"   CVD Anterior:    {previous_cvd:+,.4f} BTC",
            f"   Incremento CVD:  {delta_cvd:+,.4f} BTC",
            f"",
            f"✅ Consistência: CVD novo = CVD anterior + incremento",
            f"   {cvd:.4f} = {previous_cvd:.4f} + {delta_cvd:.4f}",
        ])
    
    # Whale metrics
    whale_buy = event.get("whale_buy_volume", 0.0)
    whale_sell = event.get("whale_sell_volume", 0.0)
    whale_delta = event.get("whale_delta", 0.0)
    
    if whale_buy > 0 or whale_sell > 0:
        log_parts.extend([
            f"",
            f"🐋 WHALE FLOW:",
            f"   Buy Volume:   {whale_buy:+,.2f} BTC",
            f"   Sell Volume:  {whale_sell:+,.2f} BTC",
            f"   Delta:        {whale_delta:+,.2f} BTC (buy - sell)",
        ])
    
    # OrderBook
    ob = event.get("orderbook_data", {})
    if ob:
        bid_depth = ob.get("bid_depth_usd", 0.0)
        ask_depth = ob.get("ask_depth_usd", 0.0)
        imbalance = ob.get("imbalance", 0.0)
        
        if bid_depth > 0 or ask_depth > 0:
            log_parts.extend([
                f"",
                f"📚 ORDER BOOK:",
                f"   Bid Depth:    ${bid_depth:+,.0f}",
                f"   Ask Depth:    ${ask_depth:+,.0f}",
                f"   Imbalance:    {imbalance:+.4f}",
            ])
    
    # Absorção
    tipo_absorcao = event.get("tipo_absorcao", "Neutra")
    if tipo_absorcao != "Neutra":
        log_parts.extend([
            f"",
            f"🎲 Absorção: {tipo_absorcao}",
        ])
    
    log_parts.append(f"{'='*80}\n")
    
    return "\n".join(log_parts)


def track_cvd_consistency(events: list) -> dict:
    """
    Analisa consistência do CVD em uma série de eventos.
    
    Args:
        events: Lista de eventos ordenados por tempo
    
    Returns:
        Relatório de consistência
    """
    if not events:
        return {"status": "error", "message": "Nenhum evento fornecido"}
    
    inconsistencies = []
    previous_cvd = None
    
    for i, event in enumerate(events):
        cvd = event.get("cvd", 0.0)
        window_id = event.get("window_id", f"W{i+1}")
        
        if previous_cvd is not None:
            # CVD deve sempre aumentar ou diminuir de forma contínua
            delta_cvd = cvd - previous_cvd
            
            # Não deve ser exatamente zero (indica possível problema)
            if abs(delta_cvd) < 0.0001 and i > 0:
                inconsistencies.append({
                    "window": window_id,
                    "issue": "CVD não mudou entre janelas",
                    "cvd": cvd,
                    "previous_cvd": previous_cvd,
                })
            
            # Saltos muito grandes podem indicar problema
            # (ajustar threshold conforme seu caso)
            if abs(delta_cvd) > 1000.0:  # exemplo: 1000 BTC em uma janela
                inconsistencies.append({
                    "window": window_id,
                    "issue": "Salto anormalmente grande no CVD",
                    "delta_cvd": delta_cvd,
                    "cvd": cvd,
                    "previous_cvd": previous_cvd,
                })
        
        previous_cvd = cvd
    
    return {
        "status": "ok" if not inconsistencies else "warning",
        "total_events": len(events),
        "inconsistencies_found": len(inconsistencies),
        "inconsistencies": inconsistencies,
        "first_cvd": events[0].get("cvd", 0.0) if events else None,
        "last_cvd": events[-1].get("cvd", 0.0) if events else None,
        "total_change": (events[-1].get("cvd", 0.0) - events[0].get("cvd", 0.0)) if len(events) > 1 else 0.0,
    }


# Exemplo de uso no market_analyzer.py
def log_with_context(event: dict, previous_event: dict = None):
    """Loga evento com contexto do anterior para clareza."""
    formatted_log = format_flow_log(event, previous_event)
    print(formatted_log)


if __name__ == "__main__":
    # Teste com dados simulados
    import json
    
    test_events = [
        {
            "window_id": "W1",
            "cvd": 1.561,
            "order_flow": {
                "net_flow_1m": 1650.0,
                "buy_volume": 50000.0,
                "sell_volume": 48350.0,
                "flow_imbalance": 0.033,
            },
            "whale_buy_volume": 0.5,
            "whale_sell_volume": 0.2,
            "whale_delta": 0.3,
            "time_index": {"timestamp_utc": "2025-01-10T18:18:00.123456Z"},
        },
        {
            "window_id": "W2",
            "cvd": 17.108,
            "order_flow": {
                "net_flow_1m": 15400.0,
                "buy_volume": 120000.0,
                "sell_volume": 104600.0,
                "flow_imbalance": 0.128,
            },
            "whale_buy_volume": 5.5,
            "whale_sell_volume": 2.0,
            "whale_delta": 3.5,
            "time_index": {"timestamp_utc": "2025-01-10T18:19:00.123456Z"},
        },
    ]
    
    print("🧪 TESTE DO FORMATADOR DE LOGS\n")
    
    # Log formatado
    for i, event in enumerate(test_events):
        previous = test_events[i-1] if i > 0 else None
        log_with_context(event, previous)
    
    # Análise de consistência
    print("\n📊 ANÁLISE DE CONSISTÊNCIA\n")
    report = track_cvd_consistency(test_events)
    print(json.dumps(report, indent=2))