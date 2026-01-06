# tests/test_data_pipeline.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any
from types import SimpleNamespace
import os
import sys

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa DataPipeline do mesmo jeito que o código de produção faz.
try:
    from data_pipeline import DataPipeline
except ImportError:
    # Fallback se, no futuro, você mover para data_pipeline/pipeline.py
    from data_pipeline.pipeline import DataPipeline

# ==========================================
# FIXTURES (Dados de Exemplo)
# ==========================================

@pytest.fixture
def sample_trades() -> List[Dict[str, Any]]:
    """Gera uma lista de trades válida para testes."""
    return [
        {"p": 100.0, "q": 1.0, "T": 1600000000000, "m": False}, # Buy
        {"p": 101.0, "q": 2.0, "T": 1600000001000, "m": True},  # Sell
        {"p": 100.5, "q": 1.5, "T": 1600000002000, "m": False}, # Buy
        {"p": 102.0, "q": 0.5, "T": 1600000003000, "m": True},  # Sell
        {"p": 101.0, "q": 1.0, "T": 1600000004000, "m": False}, # Buy
    ]

@pytest.fixture
def mock_config():
    """Configuração de teste relaxada (stub simples, sem depender de PipelineConfig real)."""
    config = SimpleNamespace()
    # Atributos básicos
    config.min_trades_pipeline = 3
    config.min_absolute_trades = 2
    config.allow_limited_data = True
    config.max_price_variance_pct = 10.0
    # Adaptação
    config.enable_adaptive_thresholds = False
    config.adaptive_learning_rate = 0.1
    config.adaptive_confidence = 0.7
    # Cache
    config.cache_ttl_seconds = 3600
    config.cache_max_items = 1000
    config.cache_allow_expired = True
    # Performance
    config.enable_vectorized_validation = True
    config.validation_chunk_size = 10000
    # Price scales
    config.price_scales = {
        'BTCUSDT': 10,
        'ETHUSDT': 100,
        'BNBUSDT': 100,
        'SOLUSDT': 1000,
        'XRPUSDT': 10000,
        'DOGEUSDT': 100000,
        'ADAUSDT': 10000,
        'DEFAULT': 10
    }
    # Methods
    config.get_price_scale = lambda symbol: config.price_scales.get(symbol, config.price_scales['DEFAULT'])
    config.get_price_precision = lambda symbol: len(str(config.get_price_scale(symbol))) - 1
    return config

# ==========================================
# TESTES DE INICIALIZAÇÃO E CARGA
# ==========================================

def test_pipeline_initialization(sample_trades, mock_config):
    """Testa se o pipeline inicializa e carrega o DataFrame corretamente."""
    pipeline = DataPipeline(
        raw_trades=sample_trades,
        symbol="BTCUSDT",
        config=mock_config,
        shared_adaptive=False
    )

    assert pipeline.df is not None
    assert not pipeline.df.empty
    assert len(pipeline.df) == 5
    # Verifica se colunas essenciais existem e tipos estão corretos
    assert all(col in pipeline.df.columns for col in ["p", "q", "T", "m"])
    assert pipeline.symbol == "BTCUSDT"

def test_pipeline_insufficient_data(mock_config):
    """Testa comportamento com dados insuficientes."""
    few_trades = [{"p": 100.0, "q": 1.0, "T": 1000, "m": False}]
    
    # Configura para NÃO permitir dados limitados, deve levantar erro
    mock_config.allow_limited_data = False
    mock_config.min_trades_pipeline = 5

    with pytest.raises(ValueError, match="Dados insuficientes"):
        DataPipeline(
            raw_trades=few_trades,
            symbol="BTCUSDT",
            config=mock_config
        )

def test_pipeline_empty_data(mock_config):
    """Testa comportamento com lista vazia."""
    with pytest.raises(ValueError):
        DataPipeline(
            raw_trades=[],
            symbol="BTCUSDT",
            config=mock_config
        )

# ==========================================
# TESTES DE ENRICH (MÉTRICAS)
# ==========================================

@patch("data_handler.calcular_metricas_intra_candle")
@patch("data_handler.calcular_volume_profile")
@patch("data_handler.calcular_dwell_time")
@patch("data_handler.calcular_trade_speed")
def test_enrich_metrics(
    mock_speed, mock_dwell, mock_vp, mock_intra,
    sample_trades, mock_config
):
    """
    Testa o método enrich e cálculo de métricas básicas.
    Mockamos as funções do data_handler para isolar o teste do pipeline.
    """
    # Setup dos mocks
    mock_intra.return_value = {"delta_fechamento": 5.0}
    mock_vp.return_value = {"poc_price": 101.0}
    mock_dwell.return_value = {"dwell_seconds": 10}
    mock_speed.return_value = {"trades_per_second": 1.5}

    pipeline = DataPipeline(sample_trades, "BTCUSDT", config=mock_config)
    
    enriched = pipeline.enrich()

    # Verificações OHLC (calculado internamente pelo MetricsProcessor)
    assert enriched["ohlc"]["open"] == 100.0
    assert enriched["ohlc"]["high"] == 102.0
    assert enriched["ohlc"]["low"] == 100.0
    assert enriched["ohlc"]["close"] == 101.0
    
    # Verificações Volume
    # Total q: 1+2+1.5+0.5+1 = 6.0
    assert enriched["volume_total"] == 6.0
    
    # Verifica se mocks foram chamados e integrados
    assert enriched["delta_fechamento"] == 5.0
    assert enriched["poc_price"] == 101.0
    assert enriched["trades_per_second"] == 1.5
    
    # Verifica cache
    assert pipeline.enriched_data is not None

# ==========================================
# TESTES DE CONTEXTO E SINAIS
# ==========================================

def test_add_context(sample_trades, mock_config):
    """Testa a adição de contexto externo."""
    pipeline = DataPipeline(sample_trades, "BTCUSDT", config=mock_config)
    
    # Gera enriched primeiro (necessário)
    with patch("data_handler.calcular_metricas_intra_candle", return_value={}):
        with patch("data_handler.calcular_volume_profile", return_value={}):
            with patch("data_handler.calcular_dwell_time", return_value={}):
                with patch("data_handler.calcular_trade_speed", return_value={}):
                    pipeline.enrich()

    external_flow = {"net_flow_1m": 1000}
    external_ob = {"bid_depth": 50000}

    context = pipeline.add_context(
        flow_metrics=external_flow,
        orderbook_data=external_ob
    )

    assert context["flow_metrics"] == external_flow
    assert context["orderbook_data"] == external_ob
    assert "ohlc" in context  # Deve manter dados enriched

def test_detect_signals(sample_trades, mock_config):
    """Testa o sistema de detecção de sinais com injeção de dependência."""
    pipeline = DataPipeline(sample_trades, "BTCUSDT", config=mock_config)
    
    # Simula estado enriquecido e contextual
    pipeline.enriched_data = {"delta_fechamento": 10.0, "volume_total": 100.0, "ohlc": {"close": 100, "close_time": 123456}}
    pipeline.contextual_data = pipeline.enriched_data.copy()

    # Mocks de detectores
    mock_abs_detector = MagicMock(return_value={
        "is_signal": True, "tipo_evento": "Absorção", "delta": 10.0
    })
    mock_exh_detector = MagicMock(return_value={
        "is_signal": False, "tipo_evento": "Exaustão"
    })

    signals = pipeline.detect_signals(
        absorption_detector=mock_abs_detector,
        exhaustion_detector=mock_exh_detector
    )

    # Deve ter 2 sinais: 1 Absorção (True) + 1 Analysis Trigger (Sempre gerado)
    assert len(signals) == 2
    
    types = [s["tipo_evento"] for s in signals]
    assert "Absorção" in types
    assert "ANALYSIS_TRIGGER" in types
    
    # Verifica se o detector foi chamado corretamente
    mock_abs_detector.assert_called_once()

# ==========================================
# TESTES DE EXTRAÇÃO DE FEATURES (ML)
# ==========================================

def test_extract_features_success(sample_trades, mock_config):
    """Testa extração de features quando generate_ml_features existe."""
    pipeline = DataPipeline(sample_trades, "BTCUSDT", config=mock_config)
    pipeline.contextual_data = {} # Mock context

    # Mock do módulo ml_features importado dentro do pipeline (ou globalmente)
    # Como o import é try/except no topo do arquivo, precisamos patchear onde ele é usado
    with patch("data_pipeline.pipeline.generate_ml_features") as mock_gen:
        mock_gen.return_value = {"feature_1": 0.5, "feature_2": 1.0}
        
        features = pipeline.extract_features()
        
        assert features == {"feature_1": 0.5, "feature_2": 1.0}
        mock_gen.assert_called_once()

def test_extract_features_missing_module(sample_trades, mock_config):
    """Testa comportamento quando ml_features não está disponível."""
    pipeline = DataPipeline(sample_trades, "BTCUSDT", config=mock_config)
    
    # Simula generate_ml_features como None
    with patch("data_pipeline.pipeline.generate_ml_features", None):
        features = pipeline.extract_features()
        assert features == {}

# ==========================================
# TESTES DE FALLBACK E RESILIÊNCIA
# ==========================================

def test_enrich_fallback_on_error(sample_trades, mock_config):
    """Testa se o pipeline sobrevive a erros no cálculo de métricas."""
    pipeline = DataPipeline(sample_trades, "BTCUSDT", config=mock_config)

    # Simula erro no MetricsProcessor
    pipeline._metrics.calculate_ohlc = MagicMock(side_effect=Exception("Erro Crítico no OHLC"))

    # O método enrich deve capturar a exceção, registrar no fallback e retornar dados mínimos
    enriched = pipeline.enrich()

    assert "ohlc" in enriched
    # Verifica se o fallback registry capturou
    stats = pipeline.fallback_registry.get_stats()
    assert stats["total_fallbacks"] > 0
    assert "enrich:complete_failure" in str(stats)

def test_get_final_features_integration(sample_trades, mock_config):
    """Testa o fluxo completo de get_final_features."""
    pipeline = DataPipeline(sample_trades, "BTCUSDT", config=mock_config)
    
    # Mockando componentes internos para evitar dependências reais
    with patch("data_handler.calcular_metricas_intra_candle", return_value={}):
        with patch("data_handler.calcular_volume_profile", return_value={}):
            with patch("data_handler.calcular_dwell_time", return_value={}):
                with patch("data_handler.calcular_trade_speed", return_value={}):
                      final = pipeline.get_final_features()

    assert "schema_version" in final
    assert final["symbol"] == "BTCUSDT"
    assert "enriched" in final
    assert "contextual" in final
    assert "signals" in final
    assert "ml_features" in final