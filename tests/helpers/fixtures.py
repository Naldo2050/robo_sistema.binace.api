# tests/fixtures.py
from dataclasses import dataclass

@dataclass
class TestOrchestratorConfig:
    """Configuração para testes - compatível com EnhancedMarketBot atual"""
    stream_url: str = "wss://test.com"
    symbol: str = "BTCUSDT"
    window_size_minutes: int = 1
    vol_factor_exh: float = 2.0
    history_size: int = 50
    delta_std_dev_factor: float = 2.5
    context_sma_period: int = 10
    liquidity_flow_alert_percentage: float = 0.4
    wall_std_dev_factor: float = 3.0

# Podemos também criar uma fixture que retorna uma instância desta classe
# Mas note: os testes estão usando a classe diretamente, então vamos apenas disponibilizar a classe.