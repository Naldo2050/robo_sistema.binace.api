# export_signals.py
# -*- coding: utf-8 -*-

"""
Módulo para exportação de sinais resumidos para CSV.

Este módulo contém a definição da dataclass ChartSignal e a função
para exportar sinais para um arquivo CSV formatado.
"""

import os
import csv
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime, timezone


# Mapeamento de símbolos para compatibilidade com MetaTrader 5
SYMBOL_MAP_FOR_MT5 = {
    "BTCUSDT": "Bitcoin",
    # Adicione outros mapeamentos conforme necessário
}


@dataclass
class ChartSignal:
    """Estrutura de dados para sinais de análise de gráficos."""
    timestamp_utc: str  # ISO 8601 format
    symbol: str
    exchange: str
    event_type: str
    side: str  # "buy", "sell" ou "none"
    price: float
    delta: float
    volume: float
    poc: Optional[float]  # float ou vazio
    val: Optional[float]  # float ou vazio
    vah: Optional[float]  # float ou vazio
    regime: str  # "trend_up", "range", "unknown"
    strength: str  # "weak", "medium", "strong"
    context: str  # string curta


def export_signal_to_csv(signal: ChartSignal) -> None:
    """
    Exporta um sinal para arquivo CSV.
    
    Args:
        signal (ChartSignal): O sinal a ser exportado.
    """
    # Cria o diretório se não existir
    csv_dir = "C:\\mt5_signals"
    csv_path = os.path.join(csv_dir, "signals.csv")
    
    try:
        os.makedirs(csv_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"❌ Falha ao criar diretório {csv_dir}: {e}")
        return
    
    # Cabeçalho do CSV
    headers = [
        "timestamp_utc",
        "symbol", 
        "exchange",
        "event_type",
        "side",
        "price",
        "delta",
        "volume",
        "poc",
        "val", 
        "vah",
        "regime",
        "strength",
        "context"
    ]
    
    # Dados do sinal
    row_data = [
        signal.timestamp_utc,
        signal.symbol,
        signal.exchange,
        signal.event_type,
        signal.side,
        signal.price,
        signal.delta,
        signal.volume,
        signal.poc if signal.poc is not None else "",
        signal.val if signal.val is not None else "",
        signal.vah if signal.vah is not None else "",
        signal.regime,
        signal.strength,
        signal.context
    ]
    
    try:
        # Verifica se o arquivo já existe para determinar se precisa escrever o cabeçalho
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Escreve cabeçalho apenas na primeira vez
            if not file_exists or os.path.getsize(csv_path) == 0:
                writer.writerow(headers)
            
            # Escreve os dados do sinal
            writer.writerow(row_data)
            
        logging.debug(f"✅ Sinal exportado para {csv_path}")
        
    except Exception as e:
        logging.error(f"❌ Falha ao exportar sinal para CSV: {e}", exc_info=True)


def determine_side(event_type: str) -> str:
    """
    Determina o lado (side) baseado no tipo do evento.
    
    Args:
        event_type (str): Tipo do evento.
        
    Returns:
        str: "buy", "sell" ou "none".
    """
    event_type_lower = event_type.lower()
    
    if "absorção de venda" in event_type_lower:
        return "buy"
    elif "absorção de compra" in event_type_lower:
        return "sell"
    else:
        return "none"


def calculate_strength(delta: float, volume: float, orderbook_imbalance: Optional[float] = None) -> str:
    """
    Calcula a força do sinal baseado em métricas disponíveis.
    
    Args:
        delta (float): Delta da janela.
        volume (float): Volume da janela.
        orderbook_imbalance (Optional[float]): Desequilíbrio do orderbook.
        
    Returns:
        str: "weak", "medium" ou "strong".
    """
    try:
        # Critérios para força forte
        strong_conditions = 0
        
        # 1. Delta absoluto alto (>= 500 como exemplo)
        if abs(delta) >= 500:
            strong_conditions += 1
        
        # 2. Volume alto (>= 100000 como exemplo)
        if volume >= 100000:
            strong_conditions += 1
        
        # 3. Imbalance forte (>= 0.6 ou <= -0.6)
        if orderbook_imbalance is not None and abs(orderbook_imbalance) >= 0.6:
            strong_conditions += 1
        
        # Lógica de classificação
        if strong_conditions >= 3:
            return "strong"
        elif strong_conditions >= 2:
            return "medium"
        else:
            return "weak"
            
    except Exception as e:
        logging.warning(f"⚠️ Erro ao calcular strength: {e}")
        return "weak"


def extract_volume_profile_data(historical_profile: Dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Extrai dados do volume profile diário (POC, VAL, VAH).
    
    Args:
        historical_profile (Dict[str, Any]): Dados do volume profile.
        
    Returns:
        tuple: (poc, val, vah)
    """
    try:
        daily_vp = historical_profile.get("daily", {})
        
        poc = daily_vp.get("poc")
        val = daily_vp.get("val")
        vah = daily_vp.get("vah")
        
        # Converte para float se não for None
        poc = float(poc) if poc is not None else None
        val = float(val) if val is not None else None
        vah = float(vah) if vah is not None else None
        
        return poc, val, vah
        
    except Exception as e:
        logging.debug(f"Falha ao extrair dados do volume profile: {e}")
        return None, None, None


def extract_regime(market_environment: Dict[str, Any]) -> str:
    """
    Extrai informações de regime do mercado.
    
    Args:
        market_environment (Dict[str, Any]): Dados do ambiente de mercado.
        
    Returns:
        str: "trend_up", "range" ou "unknown".
    """
    try:
        # Tenta extrair regime do market_environment
        trend_direction = market_environment.get("trend_direction", "").lower()
        market_structure = market_environment.get("market_structure", "").lower()
        
        # Lógica simples para determinar regime
        if "up" in trend_direction or "bullish" in trend_direction:
            return "trend_up"
        elif "range" in market_structure or "sideways" in market_structure:
            return "range"
        else:
            return "unknown"
            
    except Exception as e:
        logging.debug(f"Falha ao extrair regime: {e}")
        return "unknown"


def create_chart_signal_from_event(
    event_data: Dict[str, Any],
    symbol: str,
    exchange: str = "BINANCE",
    enriched_snapshot: Optional[Dict[str, Any]] = None,
    historical_profile: Optional[Dict[str, Any]] = None,
    market_environment: Optional[Dict[str, Any]] = None,
    orderbook_data: Optional[Dict[str, Any]] = None
) -> ChartSignal:
    """
    Cria um ChartSignal a partir dos dados do evento.
    
    Args:
        event_data (Dict[str, Any]): Dados do evento original.
        symbol (str): Símbolo do ativo.
        exchange (str): Nome da exchange.
        enriched_snapshot (Optional[Dict[str, Any]]): Dados enriquecidos.
        historical_profile (Optional[Dict[str, Any]]): Volume profile histórico.
        market_environment (Optional[Dict[str, Any]]): Ambiente de mercado.
        orderbook_data (Optional[Dict[str, Any]]): Dados do orderbook.
        
    Returns:
        ChartSignal: Sinal formatado.
    """
    try:
        # Timestamp - novo formato compatível com MetaTrader 5
        timestamp_ms = event_data.get("epoch_ms") or event_data.get("timestamp_ms")
        if timestamp_ms:
            dt_utc = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc)
        else:
            dt_utc = datetime.now(timezone.utc)
        
        # Formata no formato YYYY.MM.DD HH:MM:SS
        timestamp_utc = dt_utc.strftime("%Y.%m.%d %H:%M:%S")
        
        # Dados básicos do evento
        event_type = event_data.get("tipo_evento", "UNKNOWN")
        delta = float(event_data.get("delta", 0))
        volume = float(event_data.get("volume_total", 0))
        
        # Preço atual
        price = float(event_data.get("preco_fechamento", 0))
        if price == 0 and enriched_snapshot:
            price = float(enriched_snapshot.get("ohlc", {}).get("close", 0))
        
        # Volume Profile (POC, VAL, VAH)
        poc, val, vah = (None, None, None)
        if historical_profile:
            poc, val, vah = extract_volume_profile_data(historical_profile)
        
        # Regime de mercado
        regime = "unknown"
        if market_environment:
            regime = extract_regime(market_environment)
        
        # Orderbook imbalance
        imbalance = None
        if orderbook_data:
            imbalance = orderbook_data.get("imbalance")
        
        # Determina side e strength
        side = determine_side(event_type)
        strength = calculate_strength(delta, volume, imbalance)
        
        # Aplica mapeamento de símbolo para compatibilidade com MetaTrader 5
        symbol_for_export = SYMBOL_MAP_FOR_MT5.get(symbol, symbol)
        
        # Contexto curto
        context = f"Delta: {delta:.1f}, Vol: {volume:.0f}"
        if imbalance is not None:
            context += f", Imb: {imbalance:.2f}"
        
        signal = ChartSignal(
            timestamp_utc=timestamp_utc,
            symbol=symbol_for_export,
            exchange=exchange,
            event_type=event_type,
            side=side,
            price=price,
            delta=delta,
            volume=volume,
            poc=poc,
            val=val,
            vah=vah,
            regime=regime,
            strength=strength,
            context=context
        )
        
        return signal
        
    except Exception as e:
        logging.error(f"❌ Erro ao criar ChartSignal: {e}", exc_info=True)
        
        # Retorna sinal padrão em caso de erro
        dt_utc = datetime.now(timezone.utc)
        symbol_for_export = SYMBOL_MAP_FOR_MT5.get(symbol, symbol)
        return ChartSignal(
            timestamp_utc=dt_utc.strftime("%Y.%m.%d %H:%M:%S"),
            symbol=symbol_for_export,
            exchange=exchange,
            event_type="ERROR",
            side="none",
            price=0.0,
            delta=0.0,
            volume=0.0,
            poc=None,
            val=None,
            vah=None,
            regime="unknown",
            strength="weak",
            context=f"Erro: {str(e)}"
        )