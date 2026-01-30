"""
Configuração de Otimização de Payload para IA

Define quais campos devem ser:
- REMOVIDOS (não usados pela IA)
- SIMPLIFICADOS (compactados)
- MANTIDOS (críticos para decisões)
"""

# ==============================================================================
# CAMPOS A REMOVER COMPLETAMENTE
# ==============================================================================

FIELDS_TO_REMOVE: set[str] = {
    # ==========================================
    # 1. OBSERVABILITY (Debugging interno)
    # ==========================================
    # Por que remover: métricas de performance do sistema, não usadas pela IA
    "observability",
    "processing_times_ms",
    "circuit_breaker",
    "memory",
    # ==========================================
    # 2. DATA QUALITY (Flags internos)
    # ==========================================
    "data_quality",
    "total_trades_processed",
    "valid_rate_pct",
    "flow_trades_count",
    "processing_time_ms",
    # ==========================================
    # 3. METADATA DE CONFIGURAÇÃO
    # ==========================================
    "metadata",
    "burst_window_ms",
    "last_reset_ms",
    "config_version",
    # ==========================================
    # 4. FLAGS DE VALIDAÇÃO
    # ==========================================
    "ui_sum_ok",
    "invariants_ok",
    "units_check_passed",
    "is_valid",
    "validation_passed",
    # ==========================================
    # 5. IDs INTERNOS (mantém apenas event_id)
    # ==========================================
    "_log_id",
    "features_window_id",
    "window_id",
    # ==========================================
    # 6. TIMESTAMPS REDUNDANTES
    # ==========================================
    # Manter apenas: timestamp_utc e epoch_ms
    "timestamp_ny",
    "timestamp_sp",
    "timestamp",  # Redundante com timestamp_utc
    "time_index",  # Já temos epoch_ms
}

# ==============================================================================
# CAMPOS DO HISTORICAL_VP A REMOVER
# ==============================================================================

HISTORICAL_VP_FIELDS_TO_REMOVE: set[str] = {
    # - hvns/lvns: arrays com 50-100 valores; IA só usa alguns níveis próximos
    # - single_prints: nunca referenciado pela IA
    # - volume_nodes: strings/estruturas grandes duplicando hvns/lvns
    "hvns",
    "lvns",
    "single_prints",
    "volume_nodes",
}

# ==============================================================================
# CAMPOS A MANTER (CRÍTICOS PARA IA)
# ==============================================================================

# Mapa de campos críticos: chave -> None (qualquer valor) ou lista de subcampos.
CRITICAL_FIELDS: dict[str, object] = {
    # Identificação
    "symbol": None,
    "tipo_evento": None,
    # Preço e OHLC
    "current_price": None,
    "preco_fechamento": None,
    "ohlc": ["open", "high", "low", "close", "vwap"],
    # Volume Profile (essenciais)
    "poc": None,
    "vah": None,
    "val": None,
    # Fluxo de ordens
    "delta": None,
    "cvd": None,
    "flow_imbalance": None,
    "net_flow_1m": None,
    "aggressive_buy_pct": None,
    "aggressive_sell_pct": None,
    # Orderbook
    "bid_depth_usd": None,
    "ask_depth_usd": None,
    "imbalance": None,
    "pressure": None,
    # Contexto macro
    "trading_session": None,
    "volatility_regime": None,
    "trend_direction": None,
    "market_structure": None,
}

# ==============================================================================
# CONFIGURAÇÃO DE ARREDONDAMENTO
# ==============================================================================

ROUNDING_CONFIG: dict[str, dict[str, object]] = {
    # Preços: 2 casas decimais (centavos)
    "price_fields": {
        "decimals": 2,
        "fields": [
            "current_price",
            "preco_fechamento",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "poc",
            "vah",
            "val",
            "poc_price",
            "dwell_price",
        ],
    },
    # Volumes: 3 casas decimais (milésimos)
    "volume_fields": {
        "decimals": 3,
        "fields": [
            "volume_total",
            "volume_compra",
            "volume_venda",
            "delta",
            "cvd",
            "buy_volume_btc",
            "sell_volume_btc",
        ],
    },
    # Percentuais e Índices: 2 casas decimais
    "percentage_fields": {
        "decimals": 2,
        "fields": [
            "flow_imbalance",
            "imbalance",
            "pressure",
            "aggressive_buy_pct",
            "aggressive_sell_pct",
            "rsi_short",
            "rsi_long",
        ],
    },
    # Inteiros (sem decimais)
    "integer_fields": {
        "decimals": 0,
        "fields": [
            "num_trades",
            "trades_count",
            "dwell_seconds",
            "open_time",
            "close_time",
            "epoch_ms",
        ],
    },
    # Valores muito pequenos: 4 casas decimais
    "small_numbers": {
        "decimals": 4,
        "threshold": 0.01,  # Se abs(valor) < 0.01
        "fields": [],
    },
}

# ==============================================================================
# CONFIGURAÇÃO DE FILTROS
# ==============================================================================

FILTER_CONFIG: dict[str, dict[str, object]] = {
    # HVNs/LVNs: manter apenas níveis próximos
    "volume_profile_nearby": {
        "enabled": True,
        "max_distance_percent": 5.0,  # ±5% do preço atual
        "max_levels": 5,  # Máximo 5 níveis de cada tipo
        "sort_by_distance": True,
    },
    # Clusters de Liquidez: limitar quantidade
    "liquidity_clusters": {
        "enabled": True,
        "max_clusters": 3,
        "min_volume_threshold": 0.5,
    },
}

# ==============================================================================
# CONFIGURAÇÃO DE SEGURANÇA
# ==============================================================================

SAFETY_CONFIG: dict[str, object] = {
    # Campos que NÃO PODEM faltar no payload otimizado
    "required_fields": [
        "symbol",
        "current_price",
        "ohlc",
        "flow_imbalance",
        "bid_depth_usd",
        "ask_depth_usd",
    ],
    # Se algum destes faltar, abortar otimização
    "abort_if_missing": [
        "symbol",
        "tipo_evento",
    ],
    # Criar backup antes de modificar
    "create_backup": True,
    # Log de campos removidos (para debug)
    "log_removed_fields": True,
}

