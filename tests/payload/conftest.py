import os
import pytest


@pytest.fixture(autouse=True)
def disable_external_calls(monkeypatch):
    """
    Torna os testes de payload herméticos: sem chamadas externas de dados.
    """
    os.environ["UNIT_TEST"] = "1"
    os.environ["DISABLE_EXTERNAL_DATA"] = "1"

    # Stub das dependências externas usadas no builder
    from market_orchestrator.ai import ai_payload_builder as builder

    # Desabilita regime detector e macro provider para evitar rede/dados
    monkeypatch.setattr(builder, "MacroDataProvider", None)
    monkeypatch.setattr(builder, "EnhancedRegimeDetector", None)

    yield
