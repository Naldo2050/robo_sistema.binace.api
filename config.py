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

# Mínimo absoluto de variação (USD) na profundidade do book para disparar alerta de fluxo.
# (Se o analisador não importar esta chave, ele usa fallback interno de 500_000.)
MIN_OB_ABS_CHANGE_USD = 500_000  # aumente para 1_000_000 para reduzir ruído de alertas

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
# CORREÇÃO: Reduzido de 5 minutos para 1 minuto para dados mais recentes.
CONTEXT_UPDATE_INTERVAL_SECONDS = 60 * 1  # 1 min

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

# ==============================================================================
# PARÂMETROS DE VALIDAÇÃO E SEGURANÇA (NOVOS)
# ==============================================================================

# -- Limite de volume para trades considerados válidos
MAX_TRADE_VOLUME_BTC = 100.0      # Volume máximo considerado válido (evita outliers)
MIN_TRADE_VOLUME_BTC = 0.001      # Volume mínimo considerado válido

# -- Limite de preço para trades considerados válidos
MAX_PRICE_DEVIATION_PCT = 0.05    # 5% de desvio máximo em relação ao preço médio recente

# -- Intervalo de tempo entre atualizações de segurança
HEALTH_CHECK_INTERVAL = 30        # Segundos entre verificações de saúde do sistema

# -- Parâmetros de fallback para dados ausentes
FALLBACK_DELTA_THRESHOLD = 1.0    # Threshold de delta para fallback quando dados são inconsistentes
FALLBACK_VOLUME_THRESHOLD = 5.0   # Volume mínimo para fallback

# -- Parâmetros de tolerância para dados incompletos
MAX_MISSING_FIELDS_RATIO = 0.1    # Máximo de 10% de campos ausentes permitidos
TRADE_VALIDATION_WINDOW = 60      # Janela de tempo para validação de trades (em segundos)

# -- Configurações de log e monitoramento
LOG_LEVEL = "INFO"                # Nível de log (DEBUG, INFO, WARNING, ERROR)
LOG_TO_FILE = True                # Se deve logar para arquivo
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5         # Número de arquivos de backup

# -- Configurações de performance
MAX_PIPELINE_CACHE_SIZE = 100     # Tamanho máximo do cache do pipeline
PIPELINE_TIMEOUT_SECONDS = 10     # Timeout para operações do pipeline
MAX_CONCURRENT_ANALYSES = 5       # Número máximo de análises concorrentes
