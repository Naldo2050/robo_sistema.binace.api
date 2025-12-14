# CONFIG.PY - VERSÃO 2.3.1 - GROQ CORRIGIDO
# ==============================================================================
# CONFIGURAÇÕES GERAIS DO BOT
# ==============================================================================

import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env (se existir)
# Exemplo de .env:
# GROQ_API_KEY=...
# DASHSCOPE_API_KEY=...
# OPENAI_API_KEY=...  (opcional)
load_dotenv()

# ✅ OCI Vault Integration
try:
    from infrastructure.oci.vault_helper import get_vault_secret
except ImportError:
    # Fallback caso a pasta infrastructure não esteja no path ou em dev
    def get_vault_secret(x): return x

# OCI Configuration
OCI_COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")

# -- Ativo e Conexão --
SYMBOL = "BTCUSDT"
STREAM_URL = f"wss://fstream.binance.com/ws/{SYMBOL.lower()}@aggTrade"

# === Credenciais de IA (GroqCloud + DashScope/Qwen) ===
# ✅ PRIORIDADE 1: GroqCloud (rápido e eficiente)
GROQ_API_KEY = get_vault_secret(os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"  # ✅ CORRIGIDO: era llama-3.1-70b-versatile (descontinuado)

# FALLBACK: DashScope
DASHSCOPE_API_KEY = get_vault_secret(os.getenv("DASHSCOPE_API_KEY"))

AI_KEYS = {
    "groq": GROQ_API_KEY,           # ✅ Prioridade 1
    "dashscope": DASHSCOPE_API_KEY, # Fallback
}

# (Opcional) OpenAI
OPENAI_API_KEY = get_vault_secret(os.getenv("OPENAI_API_KEY"))

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
# Fórmula: média(delta) + (fator * desvio_padrão(delta))
DELTA_STD_DEV_FACTOR = 2.5

# -- Média Móvel de Contexto (SMA) --
# Usada para dar um contexto simples de tendência (preço acima/abaixo da média).
CONTEXT_SMA_PERIOD = 10

# ==============================================================================
# PARÂMETROS DO LIVRO DE OFERTAS (orderbook_analyzer.py) - CORRIGIDOS v2.1
# ==============================================================================

# -- Fluxo de Liquidez --
# Alerta se a liquidez no topo do book mudar mais que X% entre as checagens.
LIQUIDITY_FLOW_ALERT_PERCENTAGE = 0.4  # 40%

# 🔧 CORRIGIDO: Aumentado para valor mais realista para BTCUSDT
MIN_OB_ABS_CHANGE_USD = 750_000  # ✅ AUMENTADO de 250_000 para 750_000

# -- Paredes de Liquidez (Walls) --
# Uma ordem é considerada uma "parede" se for X desvios padrão maior que a média das ordens.
WALL_STD_DEV_FACTOR = 3.0

# ---- CRÍTICO: limiares para desequilíbrio extremo no livro ----
# Promove alerta "CRITICAL" quando QUALQUER destes cenários ocorrer:
# 1) |imbalance| >= 0.95 e (ratio_dom >= 20x OU lado dominante >= 2M USD)
# 2) ratio_dom >= 50x (independente do imbalance) — proteção para assimetrias extremas
ORDERBOOK_CRITICAL_IMBALANCE = 0.95
ORDERBOOK_MIN_DOMINANT_USD = 2_000_000.0
ORDERBOOK_MIN_RATIO_DOM = 20.0

# ==============================================================================
# 🔧 CORREÇÕES CRÍTICAS PARA ORDERBOOK v2.1
# ==============================================================================

# ===== TIMEOUTS E RETRIES =====
ORDERBOOK_REQUEST_TIMEOUT = 10.0        # ✅ REDUZIDO: 10 segundos (era 15.0)
ORDERBOOK_RETRY_DELAY = 2.0            # ✅ REDUZIDO: 2 segundos (era 3.0)
ORDERBOOK_MAX_RETRIES = 3              # ✅ REDUZIDO: 3 tentativas (era 5)

# ===== RATE LIMITING =====
ORDERBOOK_MAX_REQUESTS_PER_MIN = 10    # ✅ AUMENTADO: 10 req/min (era 5) - Binance suporta bem
ORDERBOOK_RATE_LIMIT_BUFFER = 1.0      # ✅ REDUZIDO: 1 segundo de buffer (era 2.0)

# ===== CACHE - DRASTICAMENTE REDUZIDO =====
ORDERBOOK_CACHE_TTL = 15.0             # ✅ REDUZIDO: 15 segundos (era 30.0)
ORDERBOOK_MAX_STALE = 60.0             # ✅ REDUZIDO: 1 minuto (era 5 MINUTOS!)

# ===== VALIDAÇÃO - MUITO MAIS RIGOROSA =====
ORDERBOOK_MIN_DEPTH_USD = 5_000.0      # ✅ AUMENTADO: $5,000 mínimo (era $500!)
ORDERBOOK_ALLOW_PARTIAL = False        # ✅ DESATIVADO: Exige bid E ask válidos (era True)
ORDERBOOK_MIN_LEVELS = 10              # ✅ AUMENTADO: Mínimo 10 níveis (era 3)

# ===== FALLBACK / EMERGENCY MODE =====
# Mantém fallback ativado, mas com limites rigorosos e modo emergência ligado.
ORDERBOOK_USE_FALLBACK = True          # Mantém, mas com limites rigorosos
ORDERBOOK_FALLBACK_MAX_AGE = 300       # ✅ 5 minutos (coerente com seção de reconexão v2.3.0)
ORDERBOOK_EMERGENCY_MODE = True        # ✅ Ativa modo emergência para falhas leves do orderbook

# ===== 🆕 VALIDAÇÃO DE TIMESTAMP (NOVOS PARÂMETROS) =====
ORDERBOOK_MAX_AGE_MS = 30000           # ✅ NOVO: Rejeita dados com mais de 30 segundos
ORDERBOOK_REQUIRE_TIMESTAMP = True     # ✅ NOVO: Exige timestamp válido sempre
ORDERBOOK_VALIDATE_SEQUENCE = True     # ✅ NOVO: Valida sequência de updates

# ===== WEBSOCKET =====
WS_PING_INTERVAL = 20                  # ✅ REDUZIDO: 20 segundos (era 30)
WS_PING_TIMEOUT = 10                   # Timeout de 10 segundos
WS_RECONNECT_DELAY = 3.0               # Delay de 3 segundos entre reconexões

# ==============================================================================
# PARÂMETROS DO FLUXO CONTÍNUO (flow_analyzer.py) - CORRIGIDOS v2.1
# ==============================================================================

# -- CVD & Whale Flow --
# ✅ CORRIGIDO: Alinhado com flow_analyzer.py v2.3.0
CVD_RESET_INTERVAL_HOURS = 4  # ✅ REDUZIDO: 4 horas (era 24 HORAS!)

# Sensibilidade de trades "whale" (em BTC) para o cálculo de Whale Flow (não afeta buckets)
WHALE_TRADE_THRESHOLD = 5.0

# -- Buckets de tamanho de ordem (retail/mid/whale) usados na segmentação por players
# DEPOIS (unificado com WHALE_TRADE_THRESHOLD = 5.0)
ORDER_SIZE_BUCKETS = {
    "retail": (0, 0.5),      # até 0.5 BTC
    "mid": (0.5, 5.0),       # de 0.5 a 5 BTC
    "whale": (5.0, 9999.0)   # >= 5 BTC
}

# -- Bursts (microtempo)
# Volume agregado (BTC) dentro de 200ms para caracterizar uma rajada (burst).
# Ajuste conforme sua sensibilidade: 3.0–8.0 BTC são comuns para BTCUSDT em aggTrades.
BURST_VOLUME_THRESHOLD = 5.0

# ===== 🆕 CONFIGURAÇÕES PARA SEPARAÇÃO JANELA/ACUMULADO (NOVOS PARÂMETROS) =====
FLOW_PREFER_WINDOW_METRICS = True      # ✅ NOVO: Preferir métricas de janela sobre acumulado
FLOW_LOG_WINDOW_PERIOD = True          # ✅ NOVO: Logar claramente qual período está sendo usado
FLOW_VALIDATE_CONSISTENCY = True       # ✅ NOVO: Validar consistência entre janelas e acumulado

# ==============================================================================
# PARÂMETROS DO COLETOR DE CONTEXTO (context_collector.py)
# ==============================================================================

# -- Análise Multi-Timeframe (MTF) --
CONTEXT_TIMEFRAMES = ['15m', '1h', '4h']
CONTEXT_EMA_PERIOD = 21
CONTEXT_ATR_PERIOD = 14

# -- Análise Intermarket Cripto --
INTERMARKET_SYMBOLS = ["ETHUSDT"]  # Outros criptoativos para correlação rápida

# -- Análise Intermarket Global (via yfinance) --
EXTERNAL_MARKETS = {
    "GOLD": "GC=F",     # Ouro (Future)
    "SP500": "^GSPC",   # S&P 500
    "NASDAQ": "^IXIC",  # Nasdaq
    "US10Y": "^TNX",    # Rend. Treasury 10 anos
    "OIL": "CL=F"       # Petróleo WTI
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
# ✅ CORRETO: 1 minuto para dados recentes
CONTEXT_UPDATE_INTERVAL_SECONDS = 60  # 1 minuto

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
# INTELIGÊNCIA HÍBRIDA (XGBoost + LLM)
# ==============================================================================

# Habilita sistema híbrido de decisão
HYBRID_ENABLED = True

# Modo de operação:
# - "llm_primary": IA decide, modelo como contexto (default - use quando não tiver dados)
# - "model_primary": Modelo decide, IA comenta (use quando modelo for confiável)
# - "ensemble": Ponderação de ambos (use em produção estável)
HYBRID_MODE = "llm_primary"

# Pesos para modo ensemble (devem somar 1.0)
HYBRID_MODEL_WEIGHT = 0.6  # Peso do modelo XGBoost
HYBRID_LLM_WEIGHT = 0.4    # Peso da IA generativa

# Confiança mínima do modelo para ele decidir (em model_primary)
HYBRID_MODEL_MIN_CONFIDENCE = 0.6

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
MIN_SIGNAL_VOLUME_BTC = 1.0    # Volume mínimo na janela para validar sinal
MIN_SIGNAL_TPS = 2.0           # Trades por segundo mínimo para validar sinal
MIN_ABS_DELTA_BTC = 0.5        # Piso de |delta| para validar absorção (além do delta_threshold dinâmico)
MIN_REVERSAL_RATIO = 0.2       # Reversão mínima relativa ao |delta| para caracterizar absorção (20%)
INDEX_ATR_FLOOR_PCT = 0.001    # Piso de ATR como % do preço para cálculo robusto do índice de absorção

# ==============================================================================
# PARÂMETROS DE VALIDAÇÃO E SEGURANÇA - CORRIGIDOS v2.1
# ==============================================================================

# -- Limite de volume para trades considerados válidos --
MAX_TRADE_VOLUME_BTC = 100.0   # Volume máximo considerado válido (evita outliers)
MIN_TRADE_VOLUME_BTC = 0.01    # ✅ AUMENTADO: 0.01 BTC (~$1,100) (era 0.001)

# -- Limite de preço para trades considerados válidos --
MAX_PRICE_DEVIATION_PCT = 0.01  # ✅ REDUZIDO: 1% de desvio máximo (era 5%!)

# -- Intervalo de tempo entre atualizações de segurança --
HEALTH_CHECK_INTERVAL = 30      # Segundos entre verificações de saúde do sistema

# -- Parâmetros de fallback para dados ausentes --
FALLBACK_DELTA_THRESHOLD = 1.0  # Threshold de delta para fallback quando dados são inconsistentes
FALLBACK_VOLUME_THRESHOLD = 5.0 # Volume mínimo para fallback

# -- Parâmetros de tolerância para dados incompletos --
MAX_MISSING_FIELDS_RATIO = 0.1  # Máximo de 10% de campos ausentes permitidos
TRADE_VALIDATION_WINDOW = 60    # Janela de tempo para validação de trades (em segundos)

# -- Configurações de log e monitoramento --
LOG_LEVEL = "INFO"              # Nível de log (DEBUG, INFO, WARNING, ERROR)
LOG_TO_FILE = True              # Se deve logar para arquivo
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5       # Número de arquivos de backup

# Log de campos ausentes: defina o passo de amostragem ou None para desativar
MISSING_FIELD_LOG_STEP = None   # None = desativa logs de campos ausentes

# Tamanho mínimo de caracteres para considerar a análise de IA válida durante o teste inicial
AI_TEST_MIN_CHARS = 10

# -- Configurações de performance --
MAX_PIPELINE_CACHE_SIZE = 100   # Tamanho máximo do cache do pipeline
PIPELINE_TIMEOUT_SECONDS = 10   # Timeout para operações do pipeline
MAX_CONCURRENT_ANALYSES = 5     # Número máximo de análises concorrentes

# ==============================================================================
# PARÂMETROS PARA CONTEXTO DE MERCADO E DETECÇÃO DE REGIME
# ==============================================================================

# Tamanho da janela (em candles) usada para calcular correlações entre o ativo e
# índices externos (DXY, SP500, GOLD, etc.).
CORRELATION_LOOKBACK = 50

# Percentis de volatilidade usados para classificar o regime de volatilidade.
VOLATILITY_PERCENTILES = (0.35, 0.65)

# Parâmetros para o cálculo dos indicadores ADX, RSI e MACD.
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
SR_VOLUME_WEIGHT = 0.6
SR_ORDERBOOK_WEIGHT = 0.4

# Timeframes nos quais os pivots clássicos (Daily, Weekly, Monthly) serão calculados.
PIVOT_TIMEFRAMES = ["daily", "weekly", "monthly"]

# ==============================================================================
# PARÂMETROS DE PROFUNDIDADE DO BOOK E ANÁLISE DE SPREAD
# ==============================================================================

# Quantidade de níveis de preço a considerar ao calcular a profundidade agregada do livro de ofertas.
ORDER_BOOK_DEPTH_LEVELS = [1, 5, 10, 25]

# Threshold em basis points (bps) para definir o que é considerado spread "estreito".
SPREAD_TIGHT_THRESHOLD_BPS = 0.2

# Janelas (em minutos) para cálculo da média e volatilidade do spread.
SPREAD_AVG_WINDOWS_MIN = [60, 1440]

# ==============================================================================
# PARÂMETROS DE FLUXO DE ORDENS E PARTICIPANTES
# ==============================================================================

# Janelas (em minutos) para cálculo do fluxo líquido.
# ✅ CORRETO: Valores adequados
NET_FLOW_WINDOWS_MIN = [1, 5, 15]

# Threshold (em BTC) usado para classificar ordens agressivas de acordo com seu tamanho.
AGGRESSIVE_ORDER_SIZE_THRESHOLD = 0.0

# ==============================================================================
# PARÂMETROS DE DETECÇÃO DE ABSORÇÃO
# ==============================================================================

# Epsilon para classificação de absorção por delta
ABSORCAO_DELTA_EPS = 1.0

# Modo de guarda: "off", "warn", "raise"
# Definir ABSORCAO_GUARD_MODE dinamicamente. Por padrão, em ambiente de
# desenvolvimento (ENVIRONMENT=dev) o modo é "raise" para destacar
# inconsistências; em produção o padrão continua "warn". Isso pode ser
# sobrescrito via variável de ambiente ABSORCAO_GUARD_MODE.
ABSORCAO_GUARD_MODE = os.getenv(
    "ABSORCAO_GUARD_MODE",
    "raise" if os.getenv("ENVIRONMENT", "dev").lower() == "dev" else "warn"
)

# ==============================================================================
# PARÂMETROS DE DETECÇÃO DE WHALES
# ==============================================================================

# Janela (em minutos) para acompanhar ordens grandes ao calcular a atividade de whales.
WHALE_DETECTION_WINDOW_MIN = 60

# Número mínimo de execuções consecutivas no mesmo preço para considerar um padrão de iceberg.
ICEBERG_THRESHOLD_COUNT = 3

# ==============================================================================
# PARÂMETROS DE PADRÕES E FIBONACCI
# ==============================================================================

# Número de barras (ticks ou candles) a retroceder para detectar padrões.
PATTERN_LOOKBACK_BARS = 200

# Níveis de retração de Fibonacci usados nas análises de suporte/resistência avançadas.
FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

# ==============================================================================
# PARÂMETROS DE IMPACTO DE MERCADO E SLIPPAGE
# ==============================================================================

# Tamanhos de ordem (em USD) para calcular a matriz de slippage.
SLIPPAGE_BUCKETS_USD = [1_000, 10_000, 100_000, 1_000_000]

# Pesos usados no cálculo do score de liquidez (0–10).
LIQUIDITY_WEIGHT_DEPTH = 0.4
LIQUIDITY_WEIGHT_SPREAD = 0.3
LIQUIDITY_WEIGHT_VOLUME = 0.3

# ==============================================================================
# PARÂMETROS DE DETECÇÃO DE REGIME
# ==============================================================================

# Probabilidade mínima para disparar alerta de mudança de regime de mercado.
REGIME_CHANGE_THRESHOLD = 0.15

# Intervalo estimado (em horas) de duração esperada dos regimes.
REGIME_EXPECTED_DURATION_HRS = (2, 4)

# ==============================================================================
# PARÂMETROS DE ALERTAS E ALVOS DE PREÇO
# ==============================================================================

# Probabilidade mínima para emitir alerta de teste de suporte/resistência.
ALERT_SUPPORT_PROB_THRESHOLD = 0.75

# Multiplicador do volume médio que caracteriza um pico de volume para alertas.
ALERT_VOLUME_SPIKE_THRESHOLD = 3.0

# Horizontes (em minutos) para projeções de alvo de preço.
PRICE_TARGET_HORIZONS_MIN = [5, 15, 60]

# Níveis de confiança padrão para estimativas de alvos de preço.
PRICE_TARGET_CONFIDENCE_LEVELS = {
    "low": 0.5,
    "high": 0.7,
}

# ==============================================================================
# PARÂMETROS DO LIQUIDITY HEATMAP
# ==============================================================================

# Tamanho da janela de trades para o heatmap
LHM_WINDOW_SIZE = 2000

# Threshold de clustering (% do preço)
LHM_CLUSTER_THRESHOLD_PCT = 0.003

# Mínimo de trades por cluster
LHM_MIN_TRADES_PER_CLUSTER = 5

# Intervalo de atualização em ms
LHM_UPDATE_INTERVAL_MS = 100

# ==============================================================================
# PARÂMETROS DE BURST DETECTION
# ==============================================================================

# Janela de tempo para detectar bursts (ms)
BURST_WINDOW_MS = 200

# Cooldown entre bursts (ms)
BURST_COOLDOWN_MS = 200

# ==============================================================================
# 🔧 CONFIGURAÇÕES DE ALERTA PARA DEBUG
# ==============================================================================

# Cooldown entre alertas do mesmo tipo (em segundos)
ALERT_COOLDOWN_SEC = 30

# Configurações específicas para supressão de logs duplicados
DEDUP_FILTER_WINDOW = 1.0  # segundos para considerar logs duplicados

# ==============================================================================
# 🆕 CONFIGURAÇÕES DE QUALIDADE DE DADOS v2.1
# ==============================================================================

# Validação de consistência de dados
VALIDATE_DATA_CONSISTENCY = True       # ✅ NOVO: Validar consistência entre métricas
LOG_DISCREPANCIES = True               # ✅ NOVO: Logar todas as discrepâncias detectadas
AUTO_CORRECT_DISCREPANCIES = True      # ✅ NOVO: Corrigir automaticamente inconsistências menores

# Limites de tolerância para validação
VOLUME_TOLERANCE_BTC = 0.001           # ✅ NOVO: Tolerância para volumes em BTC
DELTA_TOLERANCE_BTC = 0.01             # ✅ NOVO: Tolerância para deltas em BTC
PRICE_TOLERANCE_PCT = 0.001            # ✅ NOVO: Tolerância para preços (0.1%)

# Contadores e estatísticas
ENABLE_QUALITY_STATS = True            # ✅ NOVO: Habilitar contadores de qualidade
LOG_STATS_INTERVAL_SEC = 300           # ✅ NOVO: Logar estatísticas a cada 5 minutos

# ==============================================================================
# 🆕 CONFIGURAÇÕES DE RECONEXÃO E ESTABILIDADE v2.3.0
# ==============================================================================

# ===== SISTEMA DE AQUECIMENTO (WARMUP) APÓS RECONEXÃO =====
WARMUP_WINDOWS = 3                     # ✅ CRÍTICO: Janelas para aguardar após reconexão
TRADES_BUFFER_SIZE = 2000              # ✅ CRÍTICO: Buffer de emergência para recuperação
MIN_TRADES_FOR_PIPELINE = 10           # ✅ CRÍTICO: Mínimo de trades para criar pipeline

# ===== TIMEOUTS DE RECONEXÃO =====
HEALTH_CHECK_TIMEOUT = 90              # ✅ CRÍTICO: Sem mensagens por 90s → reconecta
HEALTH_CHECK_CRITICAL = 180            # ✅ CRÍTICO: Sem dados válidos por 3min → alerta crítico

# ===== CONFIGURAÇÕES REST API =====
REST_TIMEOUT = 5                       # ✅ Timeout para requisições REST (segundos)
REST_MAX_RETRIES = 3                   # ✅ Tentativas de retry REST
REST_RETRY_DELAY = 1.0                 # ✅ Delay entre retries REST

# ===== WEBSOCKET RECONEXÃO AVANÇADA =====
WS_MAX_RECONNECT_ATTEMPTS = 15         # ✅ Máximo de tentativas de reconexão
WS_INITIAL_DELAY = 1.0                 # ✅ Delay inicial entre reconexões (segundos)
WS_MAX_DELAY = 30.0                    # ✅ Delay máximo (reduzido de 60s)
WS_BACKOFF_FACTOR = 1.5                # ✅ Fator de crescimento do delay

# ===== VALIDAÇÃO DE PIPELINE =====
PIPELINE_MIN_ABSOLUTE_TRADES = 3       # ✅ Mínimo absoluto (erro fatal se menor)
PIPELINE_ALLOW_LIMITED_DATA = True     # ✅ Permite processar com aviso se >= 3 trades

# ===== CORREÇÕES ESPECÍFICAS PARA ORDERBOOK =====
# (já integradas acima em ORDERBOOK_EMERGENCY_MODE e ORDERBOOK_FALLBACK_MAX_AGE)

# ===== CONFIGURAÇÕES DE CLEANUP =====
CLEANUP_TIMEOUT = 5.0                  # ✅ Timeout para cleanup de recursos

# ===== FEATURE STORE =====
FEATURE_STORE_MAX_SIZE = 1000          # ✅ Máximo de janelas armazenadas

# ===== PERFORMANCE =====
USE_NUMPY_VECTORIZATION = True         # ✅ Usa NumPy para otimização
MAX_WORKER_THREADS = 5                 # ✅ Pool de threads

# ==============================================================================
# 🆕 VALIDAÇÃO AUTOMÁTICA DE CONFIGURAÇÃO v2.3.1
# ==============================================================================

def validate_config():
    """
    Valida configurações críticas automaticamente.
    ✅ ATUALIZADO v2.3.1 - Valida Groq
    """
    errors = []
    warnings = []
    
    # Valida credenciais de IA
    if not GROQ_API_KEY and not DASHSCOPE_API_KEY:
        warnings.append(
            "⚠️ Nenhuma chave de IA configurada. Sistema rodará em modo MOCK."
        )
    
    if GROQ_API_KEY and not GROQ_API_KEY.startswith("gsk_"):
        errors.append(
            f"❌ GROQ_API_KEY inválida (deve começar com 'gsk_')"
        )
    
    # Valida WebSocket
    if WS_PING_INTERVAL < 10:
        warnings.append(
            f"⚠️ WS_PING_INTERVAL muito baixo ({WS_PING_INTERVAL}s). "
            f"Recomendado: >= 20s"
        )
    
    if WS_PING_TIMEOUT >= WS_PING_INTERVAL:
        errors.append(
            f"❌ WS_PING_TIMEOUT ({WS_PING_TIMEOUT}) deve ser menor que "
            f"WS_PING_INTERVAL ({WS_PING_INTERVAL})"
        )
    
    # Valida warmup
    if WARMUP_WINDOWS < 1:
        errors.append(
            f"❌ WARMUP_WINDOWS deve ser >= 1 (atual: {WARMUP_WINDOWS})"
        )
    
    if MIN_TRADES_FOR_PIPELINE < 5:
        warnings.append(
            f"⚠️ MIN_TRADES_FOR_PIPELINE muito baixo ({MIN_TRADES_FOR_PIPELINE}). "
            f"Recomendado: >= 10"
        )
    
    # Valida orderbook
    if ORDERBOOK_MAX_RETRIES > 5:
        warnings.append(
            f"⚠️ ORDERBOOK_MAX_RETRIES alto ({ORDERBOOK_MAX_RETRIES}). "
            f"Pode causar lentidão."
        )
    
    # Valida health check
    if HEALTH_CHECK_TIMEOUT < WS_PING_INTERVAL * 3:
        warnings.append(
            f"⚠️ HEALTH_CHECK_TIMEOUT ({HEALTH_CHECK_TIMEOUT}s) deve ser >= "
            f"{WS_PING_INTERVAL * 3}s (3x ping interval)"
        )
    
    # Imprime resultados
    if errors:
        print("\n❌ ERROS DE CONFIGURAÇÃO:")
        for error in errors:
            print(f"  {error}")
        raise ValueError("Configuração inválida! Corrija os erros acima.")
    
    if warnings:
        print("\n⚠️ AVISOS DE CONFIGURAÇÃO:")
        for warning in warnings:
            print(f"  {warning}")
    
    return True


# Auto-validação ao importar
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔧 VALIDANDO CONFIGURAÇÕES v2.3.1 (com Groq)...")
    print("=" * 70)
    
    try:
        validate_config()
        print("\n✅ Todas as configurações estão válidas!")
        
        print("\n" + "=" * 70)
        print("📊 CONFIGURAÇÕES CRÍTICAS:")
        print("=" * 70)
        print(f"  Símbolo: {SYMBOL}")
        print(f"  Janela: {WINDOW_SIZE_MINUTES} min")
        print(f"  IA Groq: {'✅ Configurado' if GROQ_API_KEY else '❌ Não configurado'}")
        print(f"  Modelo Groq: {GROQ_MODEL}")
        print(f"  IA DashScope: {'✅ Fallback' if DASHSCOPE_API_KEY else '❌ Não configurado'}")
        print(f"  WebSocket Ping: {WS_PING_INTERVAL}s / Timeout: {WS_PING_TIMEOUT}s")
        print(f"  Warmup: {WARMUP_WINDOWS} janelas")
        print(f"  Min Trades: {MIN_TRADES_FOR_PIPELINE}")
        print(f"  Buffer: {TRADES_BUFFER_SIZE} trades")
        print(f"  Orderbook Retries: {ORDERBOOK_MAX_RETRIES} x {ORDERBOOK_RETRY_DELAY}s")
        print(f"  Emergency Mode: {ORDERBOOK_EMERGENCY_MODE}")
        print(f"  Health Check: {HEALTH_CHECK_TIMEOUT}s / Crítico: {HEALTH_CHECK_CRITICAL}s")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}\n")
        exit(1)

# ==============================================================================
# FIM DO CONFIG.PY v2.3.1 - GROQ CORRIGIDO
# ==============================================================================