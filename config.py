# config.py
SYMBOL = "BTCUSDT"  # Símbolo do ativo na Binance
STREAM_URL = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@trade"

# ===============================================
# PARÂMETROS DE ANÁLISE (JANELA E HISTÓRICO)
# ===============================================
WINDOW_SIZE_MINUTES = 5      # Tamanho da janela/candle em minutos
HISTORY_SIZE = 50            # Quantas janelas usar no histórico para cálculos dinâmicos

# ===============================================
# PARÂMETROS DE ABSORÇÃO (AGORA DINÂMICOS)
# ===============================================
# O limiar de delta para absorção será: Média(Delta) + (FATOR * DesvioPadrão(Delta))
# Um fator de 2.0 significa que um delta precisa estar 2 desvios padrão acima da média para ser notável.
DELTA_STD_DEV_FACTOR = 2.0

# ===============================================
# PARÂMETROS DE EXAUSTÃO
# ===============================================
VOL_FACTOR_EXH = 2.0         # Múltiplo do volume médio para "exaustão"

# ===============================================
# PARÂMETROS DE CONTEXTO E FLUXO DE LIQUIDEZ
# ===============================================
CONTEXT_SMA_PERIOD = 10      # Período da Média Móvel Simples para contexto de mercado
LIQUIDITY_FLOW_ALERT_PERCENTAGE = 0.5  # Alerta se liquidez no top 10 mudar em 50%

# NOVO PARÂMETRO: Fator de desvio padrão para detectar uma "parede" de liquidez.
# Um valor de 3.0 significa que uma ordem precisa ser 3x maior que o desvio padrão + a média para ser considerada uma parede.
# Aumente para detectar apenas paredes muito grandes, diminua para ser mais sensível.
WALL_STD_DEV_FACTOR = 3.0