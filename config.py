# ==============================================================================
# CONFIGURAÇÕES GERAIS DO BOT
# ==============================================================================

# -- Ativo e Conexão --
SYMBOL = "BTCUSDT"
STREAM_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@aggTrade"

# -- Janela de Análise (Candle) --
WINDOW_SIZE_MINUTES = 1 # Tamanho da janela de tempo para agrupar trades (em minutos)

# ==============================================================================
# PARÂMETROS DE ANÁLISE DE FLUXO (data_handler.py)
# ==============================================================================

# -- Exaustão --
# Um pico de volume é considerado exaustão se for X vezes maior que a média histórica.
VOL_FACTOR_EXH = 2.0 
HISTORY_SIZE = 50 # Número de janelas anteriores para calcular a média de volume/delta.

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
LIQUIDITY_FLOW_ALERT_PERCENTAGE = 0.4 # 40%

# -- Paredes de Liquidez (Walls) --
# Uma ordem é considerada uma "parede" se for X desvios padrão maior que a média das ordens.
WALL_STD_DEV_FACTOR = 3.0

# ==============================================================================
# PARÂMETROS DO FLUXO CONTÍNUO (flow_analyzer.py)
# ==============================================================================

# -- CVD & Whale Flow --
CVD_RESET_INTERVAL_HOURS = 24 # Reseta as métricas de CVD e Whales a cada X horas.
WHALE_TRADE_THRESHOLD = 20.0 # Trade individual (em BTC) para ser considerado "whale".

# ==============================================================================
# PARÂMETROS DO COLETOR DE CONTEXTO (context_collector.py)
# ==============================================================================

# -- Análise Multi-Timeframe (MTF) --
CONTEXT_TIMEFRAMES = ['15m', '1h', '4h']
CONTEXT_EMA_PERIOD = 21
CONTEXT_ATR_PERIOD = 14

# -- Análise Intermarket --
INTERMARKET_SYMBOLS = ["ETHUSDT"] # Outros criptoativos para monitorar correlação
# O DXY (Índice do Dólar) é pego via yfinance e não precisa ser listado aqui.

# -- Análise de Derivativos --
DERIVATIVES_SYMBOLS = ["BTCUSDT"] # Símbolos para buscar Funding, OI, etc.

# -- Volume Profile Histórico (Diário) --
VP_NUM_DAYS_HISTORY = 1 # Quantos dias de histórico para o VP (1 = VP do dia atual)
VP_VALUE_AREA_PERCENT = 0.70 # Percentual do volume para definir a Área de Valor (70% é padrão)

# -- Intervalo de Atualização --
# O coletor de contexto buscará novos dados a cada X segundos.
CONTEXT_UPDATE_INTERVAL_SECONDS = 60 * 5 # 5 minutos