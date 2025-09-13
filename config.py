# ==============================================================================
# CONFIGURAÇÕES GERAIS DO BOT
# ==============================================================================

# -- Ativo e Conexão --
SYMBOL = "BTCUSDT"
STREAM_URL = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@aggTrade"

# -- Janela de Análise (Candle) --
WINDOW_SIZE_MINUTES = 1  # Tamanho da janela de tempo para agrupar trades (em minutos)

# ==============================================================================
# PARÂMETROS DE ANÁLISE DE FLUXO (data_handler.py)
# ==============================================================================

# -- Exaustão --
# Um pico de volume é considerado exaustão se for X vezes maior que a média histórica.
VOL_FACTOR_EXH = 2.0
HISTORY_SIZE = 50  # Número de janelas anteriores para calcular a média de volume/delta.

# -- Absorção --
# O threshold de delta para absorção será calculado dinamicamente.
# Formula: média(delta) + (fator * desvio_padrão(delta))
DELTA_STD_DEV_FACTOR = 2.5

# -- Média Móvel de Contexto (SMA) --
# Usada para dar um contexto simples de tendência (preço acima/abaixo da média).
CONTEXT_SMA_PERIOD = 10

# ==============================================================================
# PARÂMETROS DO LIVRO DE OFERTAS (orderbook_analyzer.py)
# ==============================================================================

# -- Fluxo de Liquidez --
# Alerta se a liquidez no topo do book mudar mais que X% entre as checagens.
LIQUIDITY_FLOW_ALERT_PERCENTAGE = 0.4  # 40%

# -- Paredes de Liquidez (Walls) --
# Uma ordem é considerada uma "parede" se for X desvios padrão maior que a média das ordens.
WALL_STD_DEV_FACTOR = 3.0

# ==============================================================================
# PARÂMETROS DO FLUXO CONTÍNUO (flow_analyzer.py)
# ==============================================================================

# -- CVD & Whale Flow --
CVD_RESET_INTERVAL_HOURS = 24  # Reseta métricas de CVD/Whale a cada X horas

# Sensibilidade de trades "whale" (em BTC) para o cálculo de Whale Flow (não afeta buckets)
WHALE_TRADE_THRESHOLD = 5.0

# -- Buckets de tamanho de ordem (retail/mid/whale) usados na segmentação por players
ORDER_SIZE_BUCKETS = {
    "retail": (0, 0.5),      # até 0.5 BTC
    "mid": (0.5, 2.0),       # de 0.5 a 2 BTC
    "whale": (2.0, 9999.0)   # acima de 2 BTC
}

# -- Bursts (microtempo)
# Volume agregado (BTC) dentro de 200ms para caracterizar uma rajada (burst).
# Ajuste conforme sua sensibilidade: 3.0–8.0 BTC são comuns para BTCUSDT em aggTrades.
BURST_VOLUME_THRESHOLD = 5.0

# ==============================================================================
# PARÂMETROS DO COLETOR DE CONTEXTO (context_collector.py)
# ==============================================================================

# -- Análise Multi-Timeframe (MTF) --
CONTEXT_TIMEFRAMES = ['15m', '1h', '4h']
CONTEXT_EMA_PERIOD = 21
CONTEXT_ATR_PERIOD = 14

# -- Análise Intermarket Cripto --
INTERMARKET_SYMBOLS = ["ETHUSDT"]  # Outros criptoativos para correlação rápida
# O DXY vem de yfinance

# -- Análise Intermarket Global (via yfinance) --
EXTERNAL_MARKETS = {
    "GOLD": "GC=F",      # Ouro (Future)
    "SP500": "^GSPC",    # S&P 500
    "NASDAQ": "^IXIC",   # Nasdaq
    "US10Y": "^TNX",     # Rend. Treasury 10 anos
    "OIL": "CL=F"        # Petróleo WTI
}

# -- Análise de Derivativos --
DERIVATIVES_SYMBOLS = [SYMBOL]  # Usa o mesmo SYMBOL definido acima

# -- Heatmap de Liquidações --
LIQUIDATION_MAP_DEPTH = 100  # Agrupamento em US$ (100 = buckets de $100)

# -- Volume Profile Histórico --
VP_NUM_DAYS_HISTORY = 1
VP_VALUE_AREA_PERCENT = 0.70
VP_ADVANCED = True  # se True, calcula HVN/LVN/multi-timeframes/single prints

# -- Intervalo de atualização --
CONTEXT_UPDATE_INTERVAL_SECONDS = 60 * 5  # 5 min

# ==============================================================================
# PARÂMETROS DE MACHINE LEARNING / ESTATÍSTICA
# ==============================================================================

# Conjunto de janelas para olhar resultados após setups (em minutos)
ML_LOOKBACK_WINDOWS = [5, 15, 60]
# Número mínimo de exemplos necessários
ML_MIN_SAMPLE_SIZE = 100
# De quanto em quanto atualizar estatísticas
ML_UPDATE_INTERVAL = 60 * 10  # 10 minutos

# ==============================================================================
# PARÂMETROS DE SENTIMENTO / ON-CHAIN
# ==============================================================================

ENABLE_ONCHAIN = True
ONCHAIN_PROVIDERS = ["glassnode", "cryptoquant"]  # adaptável depois
STABLECOIN_FLOW_TRACKING = True

# ==============================================================================
# PARÂMETROS DE QUALIDADE DE SINAL (IA e Eventos)
# ==============================================================================

# Gating para não marcar sinal em janelas "magras" ou irrelevantes
MIN_SIGNAL_VOLUME_BTC = 1.0       # Volume mínimo na janela para validar sinal
MIN_SIGNAL_TPS = 2.0              # Trades por segundo mínimo para validar sinal
MIN_ABS_DELTA_BTC = 0.5           # Piso de |delta| para validar absorção (além do delta_threshold dinâmico)
MIN_REVERSAL_RATIO = 0.2          # Reversão mínima relativa ao |delta| para caracterizar absorção (20%)
INDEX_ATR_FLOOR_PCT = 0.001       # Piso de ATR como % do preço para cálculo robusto do índice de absorção