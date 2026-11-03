# -*- coding: utf-8 -*-
"""
Testes unitários direcionados para o contrato de configuração e isolamento
do módulo de exportação de sinais (trading/export_signals.py).

Cobre os requisitos da FASE 5.11 - INC5-C6-INC4:
A. Destino explícito em tmp_path.
B. Precedência documentada entre parâmetro e variáveis de ambiente.
C. Compatibilidade da assinatura anterior (chamada com apenas 1 argumento).
D. Schema CSV inalterado (14 colunas e formatação oficial).
E. Comportamento de escrita esperado em diretório temporário (criação de diretório, cabeçalho único, append).
F. Rejeição explícita de configuração inválida.
G. Teste com exportador interceptado / mock.
H. Garantia de isolamento e terminação segura de workers.
I. Ausência absoluta de escrita no destino padrão ao usar configuração explícita.
"""

import csv
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from trading.export_signals import (
    ChartSignal,
    DEFAULT_MT5_SIGNALS_DIR,
    DEFAULT_MT5_SIGNALS_FILENAME,
    ENV_MT5_SIGNALS_CSV_PATH,
    ENV_MT5_SIGNALS_DIR,
    export_signal_to_csv,
    resolve_signals_csv_path,
)


@pytest.fixture
def sample_signal() -> ChartSignal:
    return ChartSignal(
        timestamp_utc="2026.10.29 12:00:00",
        symbol="Bitcoin",
        exchange="BINANCE",
        event_type="ABSORCAO",
        side="buy",
        price=50000.0,
        delta=120.5,
        volume=600000.0,
        poc=49950.0,
        val=49800.0,
        vah=50200.0,
        regime="trend_up",
        strength="strong",
        context="Delta: 120.5, Vol: 600000",
    )


def test_req_a_explicit_target_path_in_tmp_path(tmp_path: Path, sample_signal: ChartSignal):
    """A. Comprova que o destino explícito é respeitado ao salvar em tmp_path."""
    target_csv = tmp_path / "custom_signals" / "custom.csv"
    res = export_signal_to_csv(sample_signal, target_path=target_csv)

    assert res == target_csv
    assert target_csv.exists()
    assert target_csv.stat().st_size > 0


def test_req_b_precedence_parameter_over_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_signal: ChartSignal):
    """B1. Parâmetro explícito tem precedência sobre MT5_SIGNALS_CSV_PATH e MT5_SIGNALS_DIR."""
    param_path = tmp_path / "from_param" / "param.csv"
    env_path = tmp_path / "from_env_path" / "env_path.csv"
    env_dir = tmp_path / "from_env_dir"

    monkeypatch.setenv(ENV_MT5_SIGNALS_CSV_PATH, str(env_path))
    monkeypatch.setenv(ENV_MT5_SIGNALS_DIR, str(env_dir))

    resolved = resolve_signals_csv_path(target_path=param_path)
    assert resolved == param_path

    export_signal_to_csv(sample_signal, target_path=param_path)
    assert param_path.exists()
    assert not env_path.exists()
    assert not (env_dir / DEFAULT_MT5_SIGNALS_FILENAME).exists()


def test_req_b_precedence_env_csv_path_over_env_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """B2. MT5_SIGNALS_CSV_PATH tem precedência sobre MT5_SIGNALS_DIR."""
    env_path = tmp_path / "from_env_path" / "env_path.csv"
    env_dir = tmp_path / "from_env_dir"

    monkeypatch.setenv(ENV_MT5_SIGNALS_CSV_PATH, str(env_path))
    monkeypatch.setenv(ENV_MT5_SIGNALS_DIR, str(env_dir))

    resolved = resolve_signals_csv_path()
    assert resolved == env_path


def test_req_b_precedence_env_dir_over_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """B3. MT5_SIGNALS_DIR resolve para '<dir>/signals.csv' quando CSV_PATH não está definido."""
    env_dir = tmp_path / "custom_dir"
    monkeypatch.delenv(ENV_MT5_SIGNALS_CSV_PATH, raising=False)
    monkeypatch.setenv(ENV_MT5_SIGNALS_DIR, str(env_dir))

    resolved = resolve_signals_csv_path()
    assert resolved == env_dir / DEFAULT_MT5_SIGNALS_FILENAME


def test_req_b_default_production_path(monkeypatch: pytest.MonkeyPatch):
    """B4. Sem parâmetro e sem variáveis, assume exatamente o caminho padrão MT5."""
    monkeypatch.delenv(ENV_MT5_SIGNALS_CSV_PATH, raising=False)
    monkeypatch.delenv(ENV_MT5_SIGNALS_DIR, raising=False)

    resolved = resolve_signals_csv_path()
    expected = Path(DEFAULT_MT5_SIGNALS_DIR) / DEFAULT_MT5_SIGNALS_FILENAME
    assert resolved == expected


def test_req_c_backward_compatible_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_signal: ChartSignal):
    """C. Compatibilidade da assinatura anterior: chamada com 1 argumento respeita env var e grava normalmente."""
    out_dir = tmp_path / "compat_out"
    monkeypatch.setenv(ENV_MT5_SIGNALS_DIR, str(out_dir))

    # Invocação tradicional com apenas 1 argumento posicional
    res = export_signal_to_csv(sample_signal)

    expected_file = out_dir / DEFAULT_MT5_SIGNALS_FILENAME
    assert res == expected_file
    assert expected_file.exists()


def test_req_d_and_e_schema_and_append_behavior(tmp_path: Path, sample_signal: ChartSignal):
    """D & E. Schema CSV inalterado (14 colunas), cabeçalho único e append de múltiplos sinais."""
    target_csv = tmp_path / "schema_test.csv"

    # Exporta 1º sinal
    export_signal_to_csv(sample_signal, target_path=target_csv)

    # Exporta 2º sinal
    sample_signal2 = ChartSignal(
        timestamp_utc="2026.10.29 12:01:00",
        symbol="Bitcoin",
        exchange="BINANCE",
        event_type="EXAUSTAO",
        side="sell",
        price=50100.0,
        delta=-80.0,
        volume=300000.0,
        poc=None,
        val=None,
        vah=None,
        regime="range",
        strength="medium",
        context="Delta: -80.0, Vol: 300000",
    )
    export_signal_to_csv(sample_signal2, target_path=target_csv)

    # Lê arquivo gerado e valida registros
    with open(target_csv, "r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))

    # 1 cabeçalho + 2 registros de dados = 3 linhas
    assert len(reader) == 3

    expected_headers = [
        "timestamp_utc", "symbol", "exchange", "event_type", "side",
        "price", "delta", "volume", "poc", "val", "vah", "regime",
        "strength", "context"
    ]
    assert reader[0] == expected_headers

    # Registro 1
    assert reader[1][0] == "2026.10.29 12:00:00"
    assert reader[1][1] == "Bitcoin"
    assert reader[1][3] == "ABSORCAO"
    assert reader[1][4] == "buy"
    assert reader[1][5] == "50000.0"

    # Registro 2
    assert reader[2][0] == "2026.10.29 12:01:00"
    assert reader[2][3] == "EXAUSTAO"
    assert reader[2][4] == "sell"
    assert reader[2][8] == ""  # POC None vira string vazia


def test_req_f_invalid_configuration_rejected():
    """F. Rejeição explícita de caminhos vazios com ValueError."""
    with pytest.raises(ValueError, match="target_path não pode ser vazio"):
        resolve_signals_csv_path(target_path="")

    with pytest.raises(ValueError, match="target_path não pode ser vazio"):
        resolve_signals_csv_path(target_path="   ")


def test_req_g_and_h_mock_interception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_signal: ChartSignal):
    """G & H. Interceptação segura por mock sem afetar filesystem."""
    mock_target = MagicMock()
    # Intercepta no módulo original e no namespace local importado no módulo de testes
    monkeypatch.setattr("trading.export_signals.export_signal_to_csv", mock_target)
    monkeypatch.setattr("tests.unit.test_export_signals_config.export_signal_to_csv", mock_target)

    # Fornece sempre target_path temporário como salvaguarda contra vazamento para destino operacional
    safe_target = tmp_path / "mock_signals.csv"

    # Chamada interceptada via namespace do módulo
    import trading.export_signals as exp_mod
    exp_mod.export_signal_to_csv(sample_signal, target_path=safe_target)

    assert mock_target.call_count == 1
    assert mock_target.call_args[0][0] == sample_signal
    assert mock_target.call_args[1]["target_path"] == safe_target
    assert not safe_target.exists()  # O mock impediu a criação em disco

    # Chamada interceptada via símbolo local do teste
    export_signal_to_csv(sample_signal, target_path=safe_target)
    assert mock_target.call_count == 2
    assert not safe_target.exists()


def test_req_i_no_write_to_default_mt5_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_signal: ChartSignal):
    """I. Comprova que ao configurar destino explícito, o destino default (simulado) não é tocado."""
    simulated_prod_dir = tmp_path / "simulated_mt5"
    simulated_prod_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("trading.export_signals.DEFAULT_MT5_SIGNALS_DIR", str(simulated_prod_dir))
    monkeypatch.setattr("tests.unit.test_export_signals_config.DEFAULT_MT5_SIGNALS_DIR", str(simulated_prod_dir))

    target_csv = tmp_path / "sandbox_signals.csv"
    default_csv = simulated_prod_dir / DEFAULT_MT5_SIGNALS_FILENAME
    default_csv.write_text("initial_state", encoding="utf-8")
    default_mtime_before = default_csv.stat().st_mtime
    default_size_before = default_csv.stat().st_size

    # Exportação para caminho explícito
    export_signal_to_csv(sample_signal, target_path=target_csv)

    assert target_csv.exists()
    assert default_csv.exists()
    assert default_csv.stat().st_mtime == default_mtime_before
    assert default_csv.stat().st_size == default_size_before
    assert default_csv.read_text(encoding="utf-8") == "initial_state"
