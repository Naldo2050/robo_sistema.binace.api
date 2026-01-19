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

    fake_correlations = {
        "status": "ok",
        "btc_eth_corr_7d": 0.1,
        "btc_eth_corr_30d": 0.2,
        "btc_dxy_corr_30d": -0.3,
        "btc_dxy_corr_90d": -0.4,
        "btc_ndx_corr_30d": 0.15,
        "dxy_return_5d": 0.0,
        "dxy_return_20d": 0.0,
        "dxy_momentum": 0.0,
    }

    monkeypatch.setattr(
        builder,
        "get_cross_asset_features",
        lambda *args, **kwargs: fake_correlations,
    )

    # Desabilita regime detector e macro provider para evitar rede/dados
    monkeypatch.setattr(builder, "MacroDataProvider", None)
    monkeypatch.setattr(builder, "EnhancedRegimeDetector", None)

    yield
