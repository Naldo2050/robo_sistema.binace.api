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
from typing import Optional, Dict, Any, Union
from pathlib import Path
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


DEFAULT_MT5_SIGNALS_DIR = "C:\\mt5_signals"
DEFAULT_MT5_SIGNALS_FILENAME = "signals.csv"
ENV_MT5_SIGNALS_CSV_PATH = "MT5_SIGNALS_CSV_PATH"
ENV_MT5_SIGNALS_DIR = "MT5_SIGNALS_DIR"


def resolve_signals_csv_path(target_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve o caminho de destino do arquivo CSV de sinais respeitando a seguinte precedência:
    1. Parâmetro explícito target_path (se fornecido).
    2. Variável de ambiente MT5_SIGNALS_CSV_PATH (se definida e não vazia).
    3. Variável de ambiente MT5_SIGNALS_DIR (se definida e não vazia, combinada com 'signals.csv').
    4. Padrão operacional de produção: 'C:\\mt5_signals\\signals.csv'.

    A resolução ocorre estritamente em tempo de execução (runtime), evitando qualquer
    efeito colateral durante o import de módulos. Rejeita caminhos vazios com ValueError.
    """
    if target_path is not None:
        if isinstance(target_path, str) and not target_path.strip():
            raise ValueError("target_path não pode ser vazio ou conter apenas espaços.")
        resolved = Path(target_path)
        if (resolved.exists() and resolved.is_dir()) or str(target_path).endswith(("\\", "/")):
            return resolved / DEFAULT_MT5_SIGNALS_FILENAME
        return resolved

    env_path = os.environ.get(ENV_MT5_SIGNALS_CSV_PATH, "").strip()
    if env_path:
        return Path(env_path)

    env_dir = os.environ.get(ENV_MT5_SIGNALS_DIR, "").strip()
    if env_dir:
        return Path(env_dir) / DEFAULT_MT5_SIGNALS_FILENAME

    return Path(DEFAULT_MT5_SIGNALS_DIR) / DEFAULT_MT5_SIGNALS_FILENAME


def export_signal_to_csv(
    signal: ChartSignal,
    target_path: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """
    Exporta um sinal para arquivo CSV formatado para integração com MetaTrader 5.

    Args:
        signal (ChartSignal): O sinal a ser exportado.
        target_path (Optional[Union[str, Path]]): Caminho explícito do arquivo CSV ou
            diretório de destino. Se None, resolve via variáveis de ambiente
            (MT5_SIGNALS_CSV_PATH / MT5_SIGNALS_DIR) ou assume o padrão operacional
            'C:\\mt5_signals\\signals.csv'.

    Returns:
        Optional[Path]: Caminho do arquivo gravado, ou None em caso de falha.
    """
    try:
        resolved_path = resolve_signals_csv_path(target_path)
    except Exception as e:
        logging.error(f"❌ Falha ao resolver caminho de exportação CSV: {e}")
        return None

    csv_dir = resolved_path.parent
    try:
        os.makedirs(csv_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"❌ Falha ao criar diretório {csv_dir}: {e}")
        return None

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
        file_exists = os.path.exists(resolved_path)

        with open(resolved_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Escreve cabeçalho apenas na primeira vez
            if not file_exists or os.path.getsize(resolved_path) == 0:
                writer.writerow(headers)

            # Escreve os dados do sinal
            writer.writerow(row_data)

        logging.debug(f"✅ Sinal exportado para {resolved_path}")
        return resolved_path

    except Exception as e:
        logging.error(f"❌ Falha ao exportar sinal para CSV: {e}", exc_info=True)
        return None


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
