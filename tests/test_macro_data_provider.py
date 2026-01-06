"""
Testes para o MacroDataProvider com foco no Treasury 10Y (TNX).
"""
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock
import pytest

# Adicionar src ao path
sys.path.insert(0, 'src')

from data.macro_data_provider import MacroDataProvider


@pytest.fixture
def macro_provider():
    """Retorna uma instância do MacroDataProvider para testes."""
    # Resetar instância para garantir isolamento entre testes
    MacroDataProvider.reset_instance()
    return MacroDataProvider.get_instance()


def test_treasury_10y_fallback_hierarchy():
    """Testa a hierarquia de fallback para Treasury 10Y."""
    provider = MacroDataProvider.get_instance()
    
    # Verificar que o método existe
    assert hasattr(provider, 'get_treasury_10y')
    assert hasattr(provider, '_fetch_treasury_10y_impl')


@pytest.mark.asyncio
async def test_treasury_10y_twelve_data_success(macro_provider):
    """Testa sucesso na busca do TNX via Twelve Data."""
    # Mock da resposta da Twelve Data
    mock_response = {
        "values": [
            {
                "close": "4.25"
            }
        ]
    }
    
    with patch('aiohttp.ClientSession.get') as mock_get:
        # Configurar mock
        mock_resp = MagicMock()
        mock_resp.status = 200
        
        # Mock para json() como corrotina
        async def mock_json():
            return mock_response
        
        mock_resp.json = mock_json
        mock_get.return_value.__aenter__.return_value = mock_resp
        
        # Configurar chave da API
        with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "test_key"}):
            result = await macro_provider._fetch_treasury_10y_impl()
            
            # Verificar que a chamada foi feita corretamente
            assert result == 4.25
            mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_treasury_10y_twelve_data_fallback_to_yahoo(macro_provider):
    """Testa fallback para Yahoo Finance quando Twelve Data falha."""
    # Mock da resposta do Yahoo Finance
    mock_yahoo_data = MagicMock()
    mock_yahoo_data.empty = False
    mock_yahoo_data.__getitem__.return_value.iloc = [4.30]
    
    with patch('aiohttp.ClientSession.get') as mock_twelve_get, \
         patch('yfinance.Ticker') as mock_yahoo_ticker:
        
        # Configurar Twelve Data para falhar
        mock_twelve_resp = MagicMock()
        mock_twelve_resp.status = 404
        mock_twelve_get.return_value.__aenter__.return_value = mock_twelve_resp
        
        # Configurar Yahoo Finance para retornar sucesso
        mock_yahoo_ticker.return_value = mock_yahoo_data
        mock_yahoo_data.history.return_value = mock_yahoo_data
        
        # Configurar chave da API
        with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "test_key"}):
            result = await macro_provider._fetch_treasury_10y_impl()
            
            # Verificar que o fallback funcionou
            assert result == 4.30
            mock_yahoo_ticker.assert_called_with("^TNX")


@pytest.mark.asyncio
async def test_treasury_10y_cache_fallback(macro_provider):
    """Testa que o último valor cacheado é retornado quando ambas as APIs falham."""
    # Configurar cache com um valor
    macro_provider._set_cache_thread_safe("treasury_10y", 4.50)
    
    with patch('aiohttp.ClientSession.get') as mock_twelve_get, \
         patch('yfinance.Ticker') as mock_yahoo_ticker:
        
        # Configurar ambas as APIs para falhar
        mock_twelve_resp = MagicMock()
        mock_twelve_resp.status = 404
        mock_twelve_get.return_value.__aenter__.return_value = mock_twelve_resp
        
        mock_yahoo_ticker.side_effect = Exception("Yahoo API Error")
        
        # Configurar chave da API
        with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "test_key"}):
            result = await macro_provider._fetch_treasury_10y_impl()
            
            # Verificar que o valor cacheado foi retornado
            assert result == 4.50


@pytest.mark.asyncio
async def test_treasury_10y_no_cache_no_apis(macro_provider):
    """Testa que None é retornado quando não há cache e ambas as APIs falham."""
    with patch('aiohttp.ClientSession.get') as mock_twelve_get, \
         patch('yfinance.Ticker') as mock_yahoo_ticker:
        
        # Configurar ambas as APIs para falhar
        mock_twelve_resp = MagicMock()
        mock_twelve_resp.status = 404
        mock_twelve_get.return_value.__aenter__.return_value = mock_twelve_resp
        
        mock_yahoo_ticker.side_effect = Exception("Yahoo API Error")
        
        # Configurar chave da API
        with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "test_key"}):
            result = await macro_provider._fetch_treasury_10y_impl()
            
            # Verificar que None foi retornado
            assert result is None


def test_treasury_10y_cache_management(macro_provider):
    """Testa o gerenciamento de cache para Treasury 10Y."""
    # Verificar que o TTL está configurado corretamente
    assert "treasury_10y" in macro_provider._ttl_config
    assert macro_provider._ttl_config["treasury_10y"] == 300
    
    # Testar cache
    macro_provider._set_cache_thread_safe("treasury_10y", 4.25)
    cached_value = macro_provider._get_cached_thread_safe("treasury_10y")
    
    assert cached_value == 4.25


@pytest.mark.asyncio
async def test_dxy_yahoo_finance_success(macro_provider):
    """Testa sucesso na busca do DXY via Yahoo Finance (fonte de verdade)."""
    # Mock da resposta do Yahoo Finance
    mock_yahoo_data = MagicMock()
    mock_yahoo_data.empty = False
    mock_yahoo_data.__getitem__.return_value.iloc = [102.50]
    
    with patch('yfinance.Ticker') as mock_yahoo_ticker:
        # Configurar Yahoo Finance para retornar sucesso
        mock_yahoo_ticker.return_value = mock_yahoo_data
        mock_yahoo_data.history.return_value = mock_yahoo_data
        
        result = await macro_provider._fetch_dxy_impl()
        
        # Verificar que o valor foi obtido corretamente
        assert result == 102.50
        mock_yahoo_ticker.assert_called_with("DX-Y.NYB")


@pytest.mark.asyncio
async def test_dxy_cache_fallback(macro_provider):
    """Testa que o último valor cacheado é retornado quando Yahoo Finance falha."""
    # Configurar cache com um valor
    macro_provider._set_cache_thread_safe("dxy", 103.25)
    
    with patch('yfinance.Ticker') as mock_yahoo_ticker:
        # Configurar Yahoo Finance para falhar
        mock_yahoo_ticker.side_effect = Exception("Yahoo API Error")
        
        result = await macro_provider._fetch_dxy_impl()
        
        # Verificar que o valor cacheado foi retornado
        assert result == 103.25


@pytest.mark.asyncio
async def test_dxy_no_cache_no_api(macro_provider):
    """Testa que None é retornado quando não há cache e Yahoo Finance falha."""
    with patch('yfinance.Ticker') as mock_yahoo_ticker:
        # Configurar Yahoo Finance para falhar
        mock_yahoo_ticker.side_effect = Exception("Yahoo API Error")
        
        result = await macro_provider._fetch_dxy_impl()
        
        # Verificar que None foi retornado
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])