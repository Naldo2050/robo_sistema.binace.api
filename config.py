# ==============================================================================
# CONFIGURAÇÕES GERAIS DO BOT
# ==============================================================================

# -- Ativo e Conexão --
SYMBOL = "BTCUSDT"
STREAM_URL = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@aggTrade"

# === Credenciais de IA (DashScope/Qwen) ===
# Coloquei aqui a sua chave para o Qwen/DashScope.
# O ai_analyzer_qwen.py já lê primeiro de variável de ambiente e,
# se não houver, usa estas variáveis abaixo.
DASHSCOPE_API_KEY = "sk-a5f340c393d74b34af1fad2c85f236f6"
AI_KEYS = {
    "dashscope": DASHSCOPE_API_KEY,
}
# (Opcional) Se algum dia usar OpenAI, pode definir aqui também:
OPENAI_API_KEY = None

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

# Log de campos ausentes: defina o passo de amostragem ou None para desativar
MISSING_FIELD_LOG_STEP = None  # None = desativa logs de campos ausentes; ex: 100 para logar a cada 100 eventos

# Tamanho mínimo de caracteres para considerar a análise de IA válida durante o teste inicial
AI_TEST_MIN_CHARS = 10

# -- Configurações de performance
MAX_PIPELINE_CACHE_SIZE = 100     # Tamanho máximo do cache do pipeline
PIPELINE_TIMEOUT_SECONDS = 10     # Timeout para operações do pipeline
MAX_CONCURRENT_ANALYSES = 5       # Número máximo de análises concorrentes

# ==============================================================================
# PARÂMETROS PARA CONTEXTO DE MERCADO E DETECÇÃO DE REGIME
# ==============================================================================

# Tamanho da janela (em candles) usada para calcular correlações entre o ativo e
# índices externos (DXY, SP500, GOLD, etc.). Esse valor serve para
# correlacionar variações de preço em uma janela relativamente curta, ajudando a
# identificar influência intermarket.
CORRELATION_LOOKBACK = 50

# Percentis de volatilidade usados para classificar o regime de volatilidade.
# Valores abaixo do primeiro percentil serão considerados regime de baixa
# volatilidade; valores acima do segundo percentil indicam alta volatilidade.
VOLATILITY_PERCENTILES = (0.35, 0.65)

# Parâmetros para o cálculo dos indicadores ADX, RSI e MACD. Essas métricas são
# empregadas por traders institucionais para avaliar força de tendência e
# momentum【935918323600793†L31-L47】【935918323600793†L129-L148】. Ajuste os
# períodos conforme a sensibilidade desejada.
ADX_PERIOD = 14
RSI_PERIODS = {
    "short": 14,
    "long": 21,
}
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

# ==============================================================================
# PARÂMETROS DE SUPORTE/RESISTÊNCIA E PIVOTS
# ==============================================================================

# Fatores de ponderação na avaliação de força de níveis de suporte e resistência.
# A soma dos pesos deve ser 1.0. Ajuste para dar mais ou menos importância ao
# volume negociado (volume profile) ou à profundidade do livro de ofertas.
SR_VOLUME_WEIGHT = 0.6
SR_ORDERBOOK_WEIGHT = 0.4

# Timeframes nos quais os pivots clássicos (Daily, Weekly, Monthly) serão
# calculados. Esses níveis ajudam a determinar pontos de reversão potenciais.
PIVOT_TIMEFRAMES = ["daily", "weekly", "monthly"]

# ==============================================================================
# PARÂMETROS DE PROFUNDIDADE DO BOOK E ANÁLISE DE SPREAD
# ==============================================================================

# Quantidade de níveis de preço a considerar ao calcular a profundidade
# agregada do livro de ofertas. Os níveis especificados serão usados para
# computar liquidez acumulada em L1, L5, L10 e L25.
ORDER_BOOK_DEPTH_LEVELS = [1, 5, 10, 25]

# Threshold em basis points (bps) para definir o que é considerado spread
# "estreito". Um spread abaixo deste valor indica mercado altamente líquido.
SPREAD_TIGHT_THRESHOLD_BPS = 0.2

# Janelas (em minutos) para cálculo da média e volatilidade do spread.
# Tipicamente usa-se 60 minutos (1h) e 1.440 minutos (24h).
SPREAD_AVG_WINDOWS_MIN = [60, 1440]

# ==============================================================================
# PARÂMETROS DE FLUXO DE ORDENS E PARTICIPANTES
# ==============================================================================

# Janelas (em minutos) para cálculo do fluxo líquido. Permitem analisar o
# desequilíbrio de agressões de compra e venda em múltiplos horizontes.
NET_FLOW_WINDOWS_MIN = [1, 5, 15]

# Threshold (em BTC) usado para classificar ordens agressivas de acordo com
# seu tamanho. Ordens de mercado acima deste valor serão tratadas como
# particularmente impactantes; valores baixos considerarão praticamente todas
# as ordens de mercado como agressivas.
AGGRESSIVE_ORDER_SIZE_THRESHOLD = 0.0

# ==============================================================================
# PARÂMETROS DE DETECÇÃO DE WHALES
# ==============================================================================

# Janela (em minutos) para acompanhar ordens grandes ao calcular a atividade
# de whales. Ajuste para considerar diferentes horizontes de acumulação.
WHALE_DETECTION_WINDOW_MIN = 60

# Número mínimo de execuções consecutivas no mesmo preço para considerar um
# padrão de iceberg/ordem escondida. Aumente para reduzir falsos positivos.
ICEBERG_THRESHOLD_COUNT = 3

# ==============================================================================
# PARÂMETROS DE PADRÕES E FIBONACCI
# ==============================================================================

# Número de barras (ticks ou candles) a retroceder para detectar padrões de
# continuação ou reversão (ex.: triângulos, bandeiras, ombro-cabeça-ombro).
PATTERN_LOOKBACK_BARS = 200

# Níveis de retração de Fibonacci usados nas análises de suporte/resistência
# avançadas. Esses valores são percentuais do movimento recente.
FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

# ==============================================================================
# PARÂMETROS DE IMPACTO DE MERCADO E SLIPPAGE
# ==============================================================================

# Tamanhos de ordem (em USD) para calcular a matriz de slippage. Ajuste as
# buckets conforme o perfil de execução desejado.
SLIPPAGE_BUCKETS_USD = [1_000, 10_000, 100_000, 1_000_000]

# Pesos usados no cálculo do score de liquidez (0–10). Definem a influência
# relativa da profundidade do livro, do spread e do volume negociado.
LIQUIDITY_WEIGHT_DEPTH = 0.4
LIQUIDITY_WEIGHT_SPREAD = 0.3
LIQUIDITY_WEIGHT_VOLUME = 0.3

# ==============================================================================
# PARÂMETROS DE DETECÇÃO DE REGIME
# ==============================================================================

# Probabilidade mínima para disparar alerta de mudança de regime de mercado.
REGIME_CHANGE_THRESHOLD = 0.15

# Intervalo estimado (em horas) de duração esperada dos regimes. Usado
# apenas para fins de exibição ou modelagem simples.
REGIME_EXPECTED_DURATION_HRS = (2, 4)

# ==============================================================================
# PARÂMETROS DE ALERTAS E ALVOS DE PREÇO
# ==============================================================================

# Probabilidade mínima para emitir alerta de teste de suporte/resistência.
ALERT_SUPPORT_PROB_THRESHOLD = 0.75

# Multiplicador do volume médio que caracteriza um pico de volume para alertas.
ALERT_VOLUME_SPIKE_THRESHOLD = 3.0

# Horizontes (em minutos) para projeções de alvo de preço. Correspondem aos
# intervalos de curto prazo utilizados na geração de cenários bull/bear.
PRICE_TARGET_HORIZONS_MIN = [5, 15, 60]

# Níveis de confiança padrão para estimativas de alvos de preço. Podem ser
# usados para calibrar modelos de probabilidade de cenários altista/baixista.
PRICE_TARGET_CONFIDENCE_LEVELS = {
    "low": 0.5,
    "high": 0.7,
}
