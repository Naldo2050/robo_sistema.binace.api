# AUDITORIA DA ESTRUTURA DOS ARQUIVOS JSON DE ANÁLISE DE MERCADO

**Data:** 03/01/2026  
**Objetivo:** Verificar se os JSONs gerados pelo sistema contêm os blocos de primeiro nível desejados

## 📋 ARQUIVOS PYTHON RESPONSÁVEIS POR GERAR JSONs DE ANÁLISE

### 1. `ai_historical_pro.py`
- **Função:** `build_enhanced_historical()`
- **JSONs gerados:**
  - `summary_json` - Resumo executivo da análise
  - `levels_json` - Dados de níveis de suporte/resistência  
  - `defense_zones_json` - Zonas de defesa identificadas

### 2. `event_saver.py`
- **Classe:** `EventSaver`
- **JSONs gerados:**
  - `eventos-fluxo.json` - Snapshot de eventos
  - `eventos_fluxo.jsonl` - Histórico de eventos

### 3. `levels_registry.py`
- **Classe:** `LevelRegistry`
- **JSONs gerados:**
  - `levels_{symbol}.json` - Registry de níveis

### 4. `ai_analyzer_qwen.py`
- **Classe:** `AIAnalyzer`
- **Função:** `analyze()`
- **JSON estruturado retornado:** Resultado da análise de IA

### 5. `market_orchestrator/ai/ai_payload_builder.py`
- **Função:** `build_ai_input()`
- **JSON gerado:** Payload estruturado para IA

### 6. Outros arquivos relevantes:
- `pattern_recognition.py` - Reconhecimento de padrões
- `ml_features.py` - Features para ML
- `market_impact.py` - Análise de impacto de mercado
- `technical_indicators.py` - Indicadores técnicos

## 📊 MAPEAMENTO DAS CHAVES DE PRIMEIRO NÍVEL

### 1. AI Historical Pro (`summary_json`)
```json
{
  "symbol": "BTCUSDT",
  "generated_at_utc": "2026-01-03T01:15:51.658Z",
  "data_quality": {"24h": 95.5, "7d": 88.2, "30d": 92.1},
  "periods_collected": ["24h", "7d", "30d"],
  "total_candles": {"24h": 1440, "7d": 672, "30d": 720},
  "profiles": {...},
  "defense_zones_count": 12,
  "entry_candidates": {"long": 8, "short": 4}
}
```

**Chaves de primeiro nível:**
- `symbol`
- `generated_at_utc` 
- `data_quality`
- `periods_collected`
- `total_candles`
- `profiles`
- `defense_zones_count`
- `entry_candidates`

### 2. AI Historical Pro (`levels_json`)
```json
{
  "symbol": "BTCUSDT",
  "created_at": "2026-01-03T01:15:51.658Z",
  "timeframes": {
    "24h": {
      "profile": {"poc": 95000, "vah": 96000, "val": 94000},
      "data_quality": 95.5,
      "candles_count": 1440
    }
  }
}
```

**Chaves de primeiro nível:**
- `symbol`
- `created_at`
- `timeframes`

### 3. AI Historical Pro (`defense_zones_json`)
```json
{
  "symbol": "BTCUSDT",
  "created_at": "2026-01-03T01:15:51.658Z",
  "zones": [
    {
      "type": "ASK_DEFENSE",
      "start_time": "2026-01-03T01:10:00.000Z",
      "end_time": "2026-01-03T01:12:00.000Z",
      "price_anchor": 95200.0,
      "strength_score": 15.5
    }
  ]
}
```

**Chaves de primeiro nível:**
- `symbol`
- `created_at`
- `zones`

### 4. Event Saver (JSON de eventos)
```json
[
  {
    "tipo_evento": "Absorção",
    "symbol": "BTCUSDT",
    "preco_fechamento": 95000,
    "delta": -15.5,
    "volume_total": 125.3,
    "timestamp_utc": "2026-01-03T01:15:51.658Z"
  }
]
```

**Chaves de primeiro nível (exemplo de evento):**
- Lista simples de eventos (não há estrutura de blocos padronizada)

### 5. AI Analyzer Qwen (`analyze()`)
```json
{
  "raw_response": "🧠 Análise Institucional...",
  "structured": null,
  "tipo_evento": "Absorção",
  "ativo": "BTCUSDT",
  "timestamp": "2026-01-03T01:15:51.658Z",
  "success": true,
  "mode": "groq",
  "model": "qwen-plus"
}
```

**Chaves de primeiro nível:**
- `raw_response`
- `structured`
- `tipo_evento`
- `ativo`
- `timestamp`
- `success`
- `mode`
- `model`
- `error` (opcional)

### 6. AI Payload Builder (`ai_payload`)
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2026-01-03T01:15:51.658Z",
  "signal_metadata": {...},
  "price_context": {
    "current_price": 95000,
    "ohlc": {...},
    "price_action": {...},
    "volume_profile_daily": {...},
    "volatility": {...}
  },
  "flow_context": {
    "net_flow": -1500,
    "cvd_accumulated": 12500,
    "flow_imbalance": -0.11,
    "whale_activity": {...}
  },
  "orderbook_context": {
    "bid_depth_usd": 1000000,
    "ask_depth_usd": 950000,
    "imbalance": 0.05,
    "depth_metrics": {...}
  },
  "technical_indicators": {
    "rsi": 65.2,
    "macd": {...},
    "adx": 28.5
  },
  "macro_context": {
    "session": "NY_OVERLAP",
    "regime": {...},
    "correlations": {...}
  },
  "ml_features": {...},
  "historical_stats": {...},
  "quant_model": {...},
  "ml_str": "..."
}
```

**Chaves de primeiro nível:**
- `symbol`
- `timestamp`
- `signal_metadata`
- `price_context`
- `flow_context`
- `orderbook_context`
- `technical_indicators`
- `macro_context`
- `ml_features`
- `historical_stats`
- `quant_model`
- `ml_str`

### 7. Pattern Recognition (`recognize_patterns()`)
```json
{
  "active_patterns": [
    {
      "type": "ASCENDING_TRIANGLE",
      "completion": 0.75,
      "target_price": 98000,
      "stop_loss": 92000,
      "confidence": 0.85
    }
  ],
  "fibonacci_levels": {
    "high": 97000,
    "low": 93000,
    "23.6": 93952,
    "38.2": 94524,
    "50.0": 95000,
    "61.8": 95476,
    "78.6": 96138
  }
}
```

**Chaves de primeiro nível:**
- `active_patterns`
- `fibonacci_levels`

### 8. ML Features (`generate_ml_features()`)
```json
{
  "price_features": {
    "returns_1": 0.0025,
    "returns_5": 0.0085,
    "volatility_1": 0.015,
    "momentum_score": 1.25
  },
  "volume_features": {
    "volume_sma_ratio": 1.25,
    "volume_momentum": 0.15,
    "buy_sell_pressure": -0.05
  },
  "microstructure": {
    "order_book_slope": 0.025,
    "flow_imbalance": -0.11,
    "tick_rule_sum": -2,
    "trade_intensity": 5.2
  },
  "data_quality": {
    "has_price_features": true,
    "has_volume_features": true,
    "has_microstructure": true,
    "issues": [],
    "is_valid": true
  }
}
```

**Chaves de primeiro nível:**
- `price_features`
- `volume_features`
- `microstructure`
- `data_quality`

### 9. Market Impact (`compute_market_impact()`)
```json
{
  "buy": {
    "avg_filled_price": 95025,
    "final_price": 95030,
    "filled_base": 1.052,
    "partial_fill": false,
    "impact_usd": 25,
    "slippage_percent": 0.026
  },
  "sell": {
    "avg_filled_price": 94975,
    "final_price": 94970,
    "filled_base": 1.053,
    "partial_fill": false,
    "impact_usd": -25,
    "slippage_percent": 0.026
  },
  "quality_flags": []
}
```

**Chaves de primeiro nível:**
- `buy`
- `sell`
- `quality_flags`

## 🔍 TABELA COMPARATIVA: BLOCOS DESEJADOS vs EXISTENTES

| Bloco Desejado | Existe no Código? | Nome Real no Código | Observações |
|---|---|---|---|
| **metadata** | ✅ Parcial | `signal_metadata` (ai_payload) / campos root | Presente em alguns contextos |
| **data_source** | ❌ Ausente | - | Não implementado |
| **market_context** | ✅ Parcial | `macro_context` (ai_payload) | Contexto macro presente |
| **price_data** | ✅ Sim | `price_context` (ai_payload) | Estrutura bem definida |
| **support_resistance** | ✅ Parcial | `levels_json` / `timeframes` | Níveis分散 em diferentes arquivos |
| **defense_zones** | ✅ Sim | `defense_zones_json.zones` | Implementado especificamente |
| **volume_profile** | ✅ Sim | `profiles` (summary) / `volume_profile_daily` | Presente em múltiplos contextos |
| **volume_nodes** | ✅ Parcial | `hvns`/`lvns` (profiles) | Nodes de volume presentes |
| **order_book_depth** | ✅ Parcial | `orderbook_context` / `order_book_depth` | Dados de profundidade presentes |
| **spread_analysis** | ✅ Parcial | `spread_percent` (orderbook_context) | Análise de spread básica |
| **order_flow** | ✅ Sim | `flow_context` / `order_flow` | Contexto de fluxo bem estruturado |
| **participant_analysis** | ❌ Ausente | `whale_activity` (flow_context) | Apenas atividade whale, não análise completa |
| **whale_activity** | ✅ Sim | `whale_activity` (flow_context) | Presente no contexto de fluxo |
| **technical_indicators** | ✅ Sim | `technical_indicators` (ai_payload) | Indicadores técnicos estruturados |
| **volatility_metrics** | ✅ Parcial | `volatility` (price_context) | Métricas básicas presentes |
| **pattern_recognition** | ✅ Sim | `active_patterns` (pattern_recognition) | Reconhecimento de padrões implementado |
| **absorption_analysis** | ✅ Parcial | `zones` (defense_zones) / tipos de zona | Análise de absorção presente |
| **market_impact** | ✅ Sim | Função `compute_market_impact()` | Implementado como função independente |
| **ml_features** | ✅ Sim | `ml_features` (ai_payload) / `generate_ml_features()` | Features ML bem estruturadas |
| **alerts** | ❌ Ausente | - | Não implementado como bloco |
| **price_targets** | ✅ Parcial | `target_price` (patterns) | Presente apenas em patterns |
| **regime_analysis** | ✅ Parcial | `regime` (macro_context) | Análise de regime básica |

## 📝 OBSERVAÇÕES GERAIS

### ✅ **Blocos Bem Implementados:**
1. **price_data** - Estrutura robusta com OHLC, price action e volume profile
2. **order_flow** - Contexto completo com métricas de fluxo e whale activity
3. **technical_indicators** - Indicadores técnicos padronizados
4. **ml_features** - Features bem organizadas por categoria
5. **market_impact** - Análise específica de impacto

### ⚠️ **Blocos Parcialmente Implementados:**
1. **support_resistance** -分散 em `levels_json` e `timeframes`, sem estrutura unificada
2. **market_context** - Contexto macro presente, mas pode ser expandido
3. **volatility_metrics** - Apenas métricas básicas, faltam análises avançadas
4. **pattern_recognition** - Padrões básicos implementados, pode ser expandido

### ❌ **Blocos Ausentes ou Limitados:**
1. **data_source** - Não há metadata sobre fontes de dados
2. **participant_analysis** - Apenas whale activity, falta análise completa de participantes
3. **alerts** - Não há sistema de alertas estruturado
4. **price_targets** - Limitado a patterns, faltam alvos baseados em outros métodos

### 🔄 **Inconsistências Estruturais:**
1. **Fragmentação** - Dados similares estão分散 em múltiplos arquivos
2. **Nomenclatura** - Nem sempre segue convenções padronizadas
3. **Granularidade** - Diferentes níveis de detalhamento entre módulos
4. **Integração** - Falta unificação entre diferentes tipos de análise

## 🎯 RECOMENDAÇÕES PARA PADRONIZAÇÃO

1. **Criar modelo de dados unificado** com todos os blocos desejados
2. **Implementar blocos ausentes** como data_source, participant_analysis, alerts
3. **Reorganizar estruturas existentes** para maior consistência
4. **Estabelecer nomenclatura padrão** para todos os blocos
5. **Integrar módulos dispersos** em uma estrutura coesa

## 📋 PRÓXIMOS PASSOS

1. ✅ Auditoria concluída
2. ⏳ Propor modelo de dados ideal
3. ⏳ Implementar estrutura padronizada (em fase posterior)

---
**Status:** AUDITORIA COMPLETA - Dados coletados e analisados