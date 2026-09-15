# market_orchestrator/capabilities.py
"""
Capability Contract — Declaração formal das capacidades de microestrutura e dados da Binance.
Define formalmente quais features são arquiteturalmente suportadas vs. não-suportadas com REST L2 ~60s.

NOTA DE ESCOPO:
Estas constantes representam CAPACIDADES ARQUITETURAIS SUPORTADAS pelo design do pipeline.
Elas NÃO representam saúde, conectividade ou disponibilidade em tempo de execução (runtime health).
Exemplo: CONTINUOUS_TRADES_WS = True declara que a arquitetura suporta e consome aggTrades contínuos
via WebSocket, mas NÃO atesta se o socket está conectado ou healthy no instante pontual da análise.
"""

# Fonte: WebSocket contínuo fapi/v1/aggTrades
CONTINUOUS_TRADES_WS: bool = True

# Fonte: REST L2 fapi/v1/depth (~60s interval)
POINT_IN_TIME_L2_SNAPSHOT: bool = True

# Não implementado / não suportado na arquitetura atual de dados:
CONTINUOUS_L2: bool = False
ORDER_CANCEL_TRACKING: bool = False
QUEUE_REPLENISHMENT_TRACKING: bool = False
ICEBERG_DETECTION_SUPPORTED: bool = False
SPOOFING_DETECTION_SUPPORTED: bool = False
HIDDEN_ORDERS_SUPPORTED: bool = False
