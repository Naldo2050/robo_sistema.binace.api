# test_export_signals.py
# -*- coding: utf-8 -*-

"""
Script de teste para o módulo de exportação de sinais.
"""

import os
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Configura logging para o teste
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

import trading.export_signals as export_signals_module
from trading.export_signals import (
    ChartSignal, 
    export_signal_to_csv, 
    create_chart_signal_from_event,
    determine_side,
    calculate_strength,
    SYMBOL_MAP_FOR_MT5,
    ENV_MT5_SIGNALS_CSV_PATH,
    ENV_MT5_SIGNALS_DIR,
)


class _RedirectedPath:
    def __init__(self, target: Path):
        self._target = target

    def join(self, *parts):
        if parts and str(parts[0]).rstrip("\\/").lower() in ("c:\\mt5_signals", "c:/mt5_signals"):
            return str(self._target)
        return os.path.join(*parts)

    def exists(self, path):
        return os.path.exists(path)

    def getsize(self, path):
        return os.path.getsize(path)


class _RedirectedOS:
    def __init__(self, target: Path):
        self.path = _RedirectedPath(target)

    def makedirs(self, path, exist_ok=False):
        return os.makedirs(self.path._target.parent, exist_ok=exist_ok)

    def __getattr__(self, name):
        return getattr(os, name)


@pytest.fixture(autouse=True)
def redirect_csv(tmp_path, monkeypatch):
    target = tmp_path / "mt5_signals" / "signals.csv"
    monkeypatch.setenv(ENV_MT5_SIGNALS_CSV_PATH, str(target))
    monkeypatch.delenv(ENV_MT5_SIGNALS_DIR, raising=False)
    monkeypatch.setattr(export_signals_module, "os", _RedirectedOS(target))
    real_open = open
    def safe_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and "mt5_signals" in str(file).lower():
            if not str(file).startswith(str(tmp_path)):
                file = str(target)
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr("builtins.open", safe_open)
    yield target


def test_chart_signal_creation(redirect_csv):
    """Testa criacao manual de ChartSignal."""
    print("Testando criacao manual de ChartSignal...")
    
    # Formata timestamp no novo formato
    dt_utc = datetime.now(timezone.utc)
    timestamp_utc = dt_utc.strftime("%Y.%m.%d %H:%M:%S")
    
    signal = ChartSignal(
        timestamp_utc=timestamp_utc,
        symbol="BTCUSDT",  # Símbolo original mantido na criação manual
        exchange="BINANCE",
        event_type="Absorção de Venda Detectada",
        side="buy",
        price=45000.0,
        delta=750.5,
        volume=125000.0,
        poc=44800.0,
        val=44500.0,
        vah=45200.0,
        regime="trend_up",
        strength="strong",
        context="Delta: 750.5, Vol: 125000, Imb: 0.75"
    )
    
    assert signal.symbol == "BTCUSDT"

    export_signal_to_csv(signal, target_path=redirect_csv)
    assert redirect_csv.exists()
    content = redirect_csv.read_text(encoding="utf-8")
    assert "BTCUSDT" in content
    assert "timestamp_utc" in content
    print("[OK] Teste de criacao manual concluido!")


def test_determine_side():
    """Testa a funcao determine_side."""
    print("Testando funcao determine_side...")
    
    test_cases = [
        ("Absorção de Venda Detectada", "buy"),
        ("Absorção de Compra Identificada", "sell"),
        ("Normal Market Action", "none"),
        ("ABSORÇÃO DE VENDA", "buy"),
        ("Teste de Absorção de Compra", "sell"),
    ]
    
    for event_type, expected in test_cases:
        result = determine_side(event_type)
        assert result == expected
        print(f"[OK] {event_type} -> {result} (esperado: {expected})")
    
    print("[OK] Teste de determine_side concluido!")


def test_calculate_strength():
    """Testa a funcao calculate_strength."""
    print("Testando funcao calculate_strength...")
    
    test_cases = [
        # (delta, volume, imbalance, expected)
        (1000, 150000, 0.8, "strong"),
        (800, 120000, 0.5, "medium"),
        (200, 50000, 0.3, "weak"),
        (0, 0, None, "weak"),
        (600, 80000, 0.7, "medium"),
    ]
    
    for delta, volume, imbalance, expected in test_cases:
        result = calculate_strength(delta, volume, imbalance)
        assert result == expected
        print(f"[OK] Delta={delta}, Vol={volume}, Imb={imbalance} -> {result} (esperado: {expected})")
    
    print("[OK] Teste de calculate_strength concluido!")


def test_symbol_mapping():
    """Testa o mapeamento de símbolos para MetaTrader 5."""
    print("Testando mapeamento de símbolos...")
    
    # Teste 1: BTCUSDT deve ser mapeado para Bitcoin
    event_data = {
        "tipo_evento": "Absorção de Venda Detectada",
        "epoch_ms": int(datetime.utcnow().timestamp() * 1000),
        "delta": 850.5,
        "volume_total": 135000.0,
        "preco_fechamento": 45200.0,
    }
    
    signal = create_chart_signal_from_event(
        event_data=event_data,
        symbol="BTCUSDT",  # Símbolo original
        exchange="BINANCE"
    )
    
    assert signal.symbol == "Bitcoin"
    print("[OK] BTCUSDT mapeado para Bitcoin")
    
    # Teste 2: Símbolo não mapeado deve ser mantido
    signal2 = create_chart_signal_from_event(
        event_data=event_data,
        symbol="ETHUSDT",  # Símbolo não mapeado
        exchange="BINANCE"
    )
    
    assert signal2.symbol == "ETHUSDT"
    print("[OK] ETHUSDT mantido (não mapeado)")
    
    print("[OK] Teste de mapeamento de símbolos concluido!")


def test_event_to_signal_conversion(redirect_csv):
    """Testa conversao de evento para ChartSignal."""
    print("Testando conversao de evento para ChartSignal...")
    
    # Dados de exemplo simulando um evento real
    event_data = {
        "tipo_evento": "Absorção de Venda Detectada",
        "epoch_ms": int(datetime.utcnow().timestamp() * 1000),
        "delta": 850.5,
        "volume_total": 135000.0,
        "preco_fechamento": 45200.0,
        "resultado_da_batalha": "Vencedores: Vendedores",
    }
    
    enriched_snapshot = {
        "ohlc": {
            "close": 45200.0,
            "open": 44800.0,
            "high": 45350.0,
            "low": 44700.0
        }
    }
    
    historical_profile = {
        "daily": {
            "poc": 45000.0,
            "val": 44800.0,
            "vah": 45300.0
        }
    }
    
    market_environment = {
        "trend_direction": "bullish",
        "market_structure": "trending"
    }
    
    orderbook_data = {
        "imbalance": 0.65
    }
    
    # Cria o sinal
    signal = create_chart_signal_from_event(
        event_data=event_data,
        symbol="BTCUSDT",
        exchange="BINANCE",
        enriched_snapshot=enriched_snapshot,
        historical_profile=historical_profile,
        market_environment=market_environment,
        orderbook_data=orderbook_data
    )
    
    print(f"Sinal criado:")
    print(f"  Timestamp: {signal.timestamp_utc}")
    print(f"  Symbol: {signal.symbol}")
    print(f"  Event Type: {signal.event_type}")
    print(f"  Side: {signal.side}")
    print(f"  Price: {signal.price}")
    print(f"  Delta: {signal.delta}")
    print(f"  Volume: {signal.volume}")
    print(f"  POC: {signal.poc}")
    print(f"  VAL: {signal.val}")
    print(f"  VAH: {signal.vah}")
    print(f"  Regime: {signal.regime}")
    print(f"  Strength: {signal.strength}")
    print(f"  Context: {signal.context}")
    
    assert signal.symbol == "Bitcoin"
    assert validate_timestamp_format(signal.timestamp_utc)

    # Exporta o sinal
    export_signal_to_csv(signal, target_path=redirect_csv)
    assert redirect_csv.exists()
    content = redirect_csv.read_text(encoding="utf-8")
    assert "Bitcoin" in content

    print("[OK] Teste de conversao de evento concluido!")


def validate_timestamp_format(timestamp_str: str) -> bool:
    """Valida se o timestamp está no formato YYYY.MM.DD HH:MM:SS."""
    import re
    pattern = r'^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}$'
    return bool(re.match(pattern, timestamp_str))


def check_csv_output(csv_path=None):
    """Verifica se o arquivo CSV foi criado e mostra seu conteudo."""
    if csv_path is None:
        print("[SKIP] Caminho CSV nao informado; nenhuma escrita operacional sera feita.")
        return

    if os.path.exists(csv_path):
        print(f"[OK] Arquivo CSV encontrado: {csv_path}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print("Conteudo do CSV:")
                print("=" * 80)
                print(content)
                print("=" * 80)
                
                # Valida formato do timestamp
                lines = content.strip().split('\n')
                if len(lines) > 1:  # Tem cabeçalho + pelo menos uma linha de dados
                    data_lines = lines[1:]  # Ignora cabeçalho
                    for line in data_lines:
                        if line.strip():
                            timestamp = line.split(',')[0]
                            if validate_timestamp_format(timestamp):
                                print(f"[OK] Timestamp válido: {timestamp}")
                            else:
                                print(f"[ERRO] Timestamp inválido: {timestamp}")
                 
        except Exception as e:
            print(f"[ERRO] Erro ao ler CSV: {e}")
    else:
        print(f"[ERRO] Arquivo CSV nao encontrado: {csv_path}")


def main():
    """Executa todos os testes usando somente um diretório temporário."""
    print("Iniciando testes do modulo de exportacao de sinais...")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "mt5_signals" / "signals.csv"
        os.environ[ENV_MT5_SIGNALS_CSV_PATH] = str(target)
        os.environ.pop(ENV_MT5_SIGNALS_DIR, None)
        with patch.object(export_signals_module, "os", _RedirectedOS(target)):
            try:
                test_symbol_mapping()
                print()

                test_determine_side()
                print()

                test_calculate_strength()
                print()

                test_event_to_signal_conversion(target)
                print()

                test_chart_signal_creation(target)
                print()

                check_csv_output(target)

                print()
                print("[SUCESSO] Todos os testes concluidos com sucesso!")

            except Exception as e:
                print(f"[ERRO] Erro durante os testes: {e}", exc_info=True)


if __name__ == "__main__":
    main()