from __future__ import annotations

# tests/conftest.py
import sys
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Desabilitar métricas do Prometheus para evitar duplicação em testes
os.environ["ORDERBOOK_METRICS_ENABLED"] = "0"

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock

import pytest

# Limpar registry do Prometheus antes dos testes
try:
    from prometheus_client import CollectorRegistry
    # Usar um registry limpo para os testes
    import prometheus_client
    prometheus_client.registry.REGISTRY = CollectorRegistry()
except ImportError:
    pass  # prometheus_client não disponível


@dataclass
class FakeTimeManager:
    now: int  # epoch_ms fixo

    def now_ms(self) -> int:
        return int(self.now)

    def build_time_index(
        self,
        epoch_ms: int,
        include_local: bool = False,
        timespec: str = "seconds",
    ) -> Dict[str, Any]:
        # suficiente para o analyzer (ny/utc podem ser None)
        return {
            "timestamp_ny": None,
            "timestamp_utc": None,
        }


def make_valid_snapshot(epoch_ms: int) -> Dict[str, Any]:
    # bids em ordem decrescente, asks em ordem crescente
    bids: List[Tuple[float, float]] = [
        (100.0, 5.0),
        (99.5, 5.0),
        (99.0, 5.0),
        (98.5, 5.0),
        (98.0, 5.0),
    ]
    asks: List[Tuple[float, float]] = [
        (100.5, 5.0),
        (101.0, 5.0),
        (101.5, 5.0),
        (102.0, 5.0),
        (102.5, 5.0),
    ]
    return {
        "E": int(epoch_ms),
        "bids": bids,
        "asks": asks,
    }


@pytest.fixture
def fixed_now_ms() -> int:
    # um timestamp fixo para evitar flakiness
    return 1_700_000_000_000


@pytest.fixture
def tm(fixed_now_ms: int) -> FakeTimeManager:
    return FakeTimeManager(now=fixed_now_ms)


@pytest.fixture
def orchestrator_config():
    """Fixture para a configuração do orchestrator (usada pelos testes antigos)."""
    from tests.config_test import OrchestratorConfigTest
    return OrchestratorConfigTest()


# ===== AI MOCKS FIXTURES =====
import pytest
from tests.mock_ai_responses import MockQwenClient, MOCK_AI_ANALYSIS, MOCK_QWEN_RESPONSE

@pytest.fixture
def mock_qwen_client(mocker):
    """Mock para o QwenClient."""
    # Cria um mock que retorna objetos com a interface correta
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = json.dumps({
        'signal': 'STRONG_BUY',
        'confidence': 0.92,
        'reasoning': 'Strong buy pressure with high volume absorption at key support levels.',
        'price_targets': {
            'short_term': 51000,
            'medium_term': 52500,
            'long_term': 54000
        },
        'risk_level': 'MEDIUM',
        'key_observations': [
            'Large bid wall at 49999',
            'Low ask liquidity near current price'
        ],
        'recommended_action': 'Consider accumulating position with tight stop loss'
    })
    
    mock_client.chat.completions.create.return_value = mock_response
    
    # Patch the QwenClient to return our mock
    return mocker.patch('ai_runner.QwenClient', return_value=mock_client)

@pytest.fixture
def mock_ai_response():
    """Retorna uma resposta mockada da IA."""
    return MOCK_AI_ANALYSIS

@pytest.fixture
def mock_qwen_response():
    """Retorna a resposta mockada do Qwen."""
    return MOCK_QWEN_RESPONSE

# Adicione isto no FINAL do arquivo tests/conftest.py
# Substitua o que foi adicionado anteriormente

import sys
from unittest.mock import MagicMock

class MockQwenClient:
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key
        self.model = model
        self.call_count = 0
        self.fail_until = 0
        self.response_content = '{"analysis": "Test analysis", "confidence": 0.85, "signal": "NEUTRAL"}'
        self.timeout_mode = False
        self.timeout_seconds = 35
        self.last_messages = None  # NOVO: armazena últimas mensagens
        self.last_kwargs = None    # NOVO: armazena kwargs
        self.custom_instructions_found = False  # NOVO: flag para instruções customizadas
        
    def set_fail_until(self, n):
        self.fail_until = n
    
    def set_response_content(self, content):
        self.response_content = content
    
    def set_timeout_mode(self, timeout_seconds=35):
        self.timeout_mode = True
        self.timeout_seconds = timeout_seconds
    
    def call(self, messages, **kwargs):
        self.call_count += 1
        self.last_messages = messages  # Armazena mensagens
        self.last_kwargs = kwargs      # Armazena kwargs
        
        # Verifica se há instruções customizadas nas mensagens
        for msg in messages:
            if isinstance(msg, dict) and 'content' in msg:
                content = msg['content']
                if 'custom instruction' in content.lower() or 'focus on liquidity' in content.lower():
                    self.custom_instructions_found = True
        
        # Modo timeout
        if self.timeout_mode:
            import time
            time.sleep(self.timeout_seconds)
            raise Exception(f"Timeout after {self.timeout_seconds} seconds")
        
        # Para o teste de retry failure
        if self.call_count <= self.fail_until:
            raise Exception("AI analysis failed")
        
        # Retorna resposta
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = self.response_content
        mock_response.choices = [mock_choice]
        return mock_response

# Injeta o mock no módulo ai_runner
try:
    if 'ai_runner' in sys.modules:
        sys.modules['ai_runner'].QwenClient = MockQwenClient
except (KeyError, AttributeError):
    # Se o módulo não existe ainda, a injeção será feita quando for importado
    pass

# Fixture para configurar falhas controladas
import pytest

@pytest.fixture
def failing_qwen_client():
    """Fixture para simular falhas controladas"""
    client = MockQwenClient()
    client.set_fail_until(3)  # Falha 3 vezes antes de sucesso
    return client

@pytest.fixture
def qwen_client_low_confidence():
    """Fixture para retornar confiança baixa (0.5)"""
    client = MockQwenClient()
    client.set_response_content('{"analysis": "Test analysis", "confidence": 0.5, "signal": "NEUTRAL"}')
    return client