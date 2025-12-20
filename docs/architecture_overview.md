# Visão Geral da Arquitetura – Enhanced Market Bot

Este documento descreve a arquitetura de alto nível do sistema:

- Entry point (`main.py`)
- Orquestrador de mercado (`EnhancedMarketBot`)
- Conexão WebSocket (`RobustConnectionManager`)
- Processamento de janelas (`window_processor`)
- Módulo de OrderBook (`OrderBookAnalyzer` + wrapper)
- Fluxo e DataPipeline
- IA (quantitativa + generativa)
- EventBus, EventSaver e HealthMonitor

---

## 1. Entry Point (`main.py`)

Responsabilidades:

- Carregar variáveis de ambiente (`dotenv` + `config.py`).
- Validar parâmetros obrigatórios (`_validate_required_config`).
- Configurar logging principal (nível via `config.LOG_LEVEL`).
- Iniciar servidor Prometheus (se `prometheus_client` disponível).
- Criar instância do `EnhancedMarketBot`:
  - `stream_url`, `symbol`, `window_size_minutes`, etc.
- Executar:
  - `asyncio.run(bot.run())`
- Garantir cleanup:
  - `bot._cleanup_handler()` em bloco `finally`.

---

## 2. Orquestrador Principal – `EnhancedMarketBot`

**Arquivo:** `market_orchestrator/market_orchestrator.py`  
Responsável por:

- Receber todos os trades via WebSocket.
- Manter janelas de tempo (`window_size_minutes`).
- Orquestrar:
  - fluxo de dados (`DataPipeline`, `FlowAnalyzer`),
  - OrderBook (`OrderBookAnalyzer` via wrapper),
  - IA (quantitativa + generativa),
  - alertas institucionais,
  - persistência de eventos (`EventSaver`),
  - publicação de sinais (`EventBus`),
  - monitoramento de saúde (`HealthMonitor`).

### Componentes principais criados no `__init__`:

- `TimeManager`
- `HealthMonitor`
- `EventBus`
- `FeatureStore`
- `LevelRegistry`
- `TradeFlowAnalyzer`
- `OrderBookAnalyzer` (novo design assíncrono)
- `EventSaver`
- `ContextCollector`
- `FlowAnalyzer`
- IA:
  - `AIAnalyzer` (via `initialize_ai_async`)
  - `MLInferenceEngine` (quantitativo)
- Conexão:
  - `RobustConnectionManager` (aiohttp WebSocket)
- Estruturas de janela:
  - `window_end_ms`, `window_data`, `window_count`
  - históricos: `volume_history`, `delta_history`, `close_price_history`, `volatility_history`
  - pattern history: `pattern_ohlc_history`

### Loop principal: `run()`

- Inicia o `ContextCollector`.
- Loga início do bot (versão, símbolo, fuso horário).
- Chama `await self.connection_manager.connect()`.
- Sempre que o WebSocket recebe mensagem, chama `on_message`.

---

## 3. Conexão WebSocket – `RobustConnectionManager`

**Arquivo:** `market_orchestrator/connection/robust_connection.py`

Responsabilidades:

- Conectar via `aiohttp.ClientSession().ws_connect(...)`.
- Manter loop:
  - reconecta automaticamente em caso de erro/close.
  - usa backoff exponencial com jitter:
    - `initial_delay`, `backoff_factor`, `max_delay`.
- Chamar callbacks:
  - `on_message`, `on_open`, `on_close`, `on_error`, `on_reconnect`.
- Expor health stats:
  - `get_stats()` com:
    - `total_messages_received`, `total_reconnects`,
    - `connection_uptime_sec`, `last_message_age_sec`, etc.
- Logs estruturados:
  - `ws_connect_attempt`, `ws_connected`, `ws_reconnect_scheduled`,
  - `ws_max_reconnect_reached`, `ws_disconnect_called`.
- Tracing (OpenTelemetry opcional):
  - span `"ws_connect_loop"`.

---

## 4. Processamento de Mensagens e Janelas

### 4.1. `on_message`

- Recebe `message` JSON bruto.
- Normaliza estrutura para um trade:
  - `p` (preço),
  - `q` (quantidade),
  - `T` (timestamp),
  - `m` (agressor).
- Valida:
  - tipos (`float`, `int`),
  - valores > 0,
  - garante monotonicidade de `T`.
- Atualiza:
  - `trades_buffer` (para emergências),
  - `flow_analyzer.process_trade(...)`.
- Gerencia janelas:
  - `window_end_ms` definido por `_next_boundary_ms(T)`.
  - Quando `T >= window_end_ms`:
    - chama `self._process_window()`
    - abre nova janela.

### 4.2. `_process_window` → `window_processor.process_window(bot)`

**Arquivo:** `market_orchestrator/windows/window_processor.py`

Responsável por:

- Normalizar trades da janela (tipos corretos em `p`, `q`, `T`).
- Garantir número mínimo de trades (`min_trades_for_pipeline`)
  - se insuficiente, tenta completar com `trades_buffer`.
- Calcular:
  - `total_buy_volume`, `total_sell_volume`.
- Atualizar health (`health_monitor.heartbeat("main")`).
- Calcular `dynamic_delta_threshold` com base em histórico.
- Obter contexto:
  - Macro + VP (`context_collector.get_context()`).
  - Atualizar `levels` via `update_from_vp`.
- Criar `DataPipeline(valid_window_data, symbol, time_manager)`.
- Fluxo:
  - `flow_metrics = flow_analyzer.get_flow_metrics(...)`.
- OrderBook:
  - `ob_event = fetch_orderbook_with_retry(bot, close_ms)`.
- Enriquecimento:
  - `enriched = pipeline.enrich()`.
  - `pipeline.add_context(...)` com flow, VP, orderbook, macro.
- Sinais:
  - `signals = pipeline.detect_signals(...)` com:
    - `create_absorption_event`,
    - `create_exhaustion_event`.
- IA Quantitativa:
  - `ml_prediction = ml_engine.predict(...)` (se disponível).
  - injeta `quant_model.prob_up` no `macro_context`.
- Delegar ao bot:
  - `bot._process_signals(...)` (enriquecimento final + IA generativa).
- Persistir features:
  - `feature_store.save_features(window_id, final_features)`.

Logs estruturados:

- `window_processed` (por janela),
- `window_process_error` em caso de falha.

Tracing:

- span `"process_window"`.

---

## 5. Module de OrderBook

### 5.1. `OrderBookAnalyzer` (núcleo)

**Arquivo:** `orderbook_analyzer.py`  
**Suporte:** `orderbook_core/*`

Responsável por:

- Buscar o orderbook na Binance Futures:
  - `_fetch_orderbook()` (aiohttp, sessão reutilizável, timeout configurável).
- Validar snapshot:
  - `_validate_snapshot` (estrutura, timestamp, liquidez mínima, ordenação, spread, etc.).
- Calcular métricas:
  - `_spread_and_depth`, `_imbalance_ratio_pressure`,
  - `_detect_walls`, `_iceberg_reload`,
  - `_simulate_market_impact`, `_detect_anomalies`,
  - `advanced_metrics` (weighted imbalance, liquidity concentration, microstructure).
- Gerar evento padrão de OrderBook:
  - `analyze()` → retorna dict com:
    - `orderbook_data`, `spread_metrics`, `order_book_depth`,
    - `spread_analysis`, `market_impact_*`,
    - `advanced_metrics`, `data_quality`, `health_stats`.

Pontos institucionais:

- Circuit Breaker:
  - `CircuitBreaker` em `_fetch_orderbook`:
    - `allow_request()` bloqueia live fetch com falhas repetidas.
    - fallback para cache/stale/emergency.
- Métricas Prometheus:
  - `OrderBookMetrics` (counters, histogram, gauges).
- Logging estruturado:
  - `orderbook_event`, `orderbook_invalid`.
- Tracing:
  - span `"orderbook_analyze"`.

### 5.2. Wrapper de OrderBook

**Arquivo:** `market_orchestrator/orderbook/orderbook_wrapper.py`

Responsável por:

- Executar `OrderBookAnalyzer.analyze()` em loop async dedicado (thread separada).
- `fetch_orderbook_with_retry(bot, close_ms)`:
  - usa resultado se for válido e tiver liquidez mínima,
  - atualiza `bot.last_valid_orderbook` + cache local,
  - fallback via `orderbook_fallback`.
- `orderbook_fallback(bot)`:
  - usa cache se não estiver muito velho,
  - entra em modo emergência se permitido,
  - ou retorna evento inválido com schema padronizado (via `event_factory`).

Logs estruturados:

- `orderbook_ok`, `orderbook_fallback_cache`,  
  `orderbook_emergency_mode`, `orderbook_fallback_error`.

---

## 6. IA – Quantitativa e Generativa

### 6.1. IA Quantitativa

- `ml.inference_engine.MLInferenceEngine`:
  - prevê probabilidade de alta (`prob_up`) e confiança,
  - extrai `ml_features` quando necessário.

Usada em:

- `ai_runner.initialize_ai_async`:
  - inicializa ML engine e testa com um payload simples.
- `ai_runner.run_ai_analysis_threaded`:
  - executa `ml_engine.predict(event_data)` antes da IA generativa.

### 6.2. IA Generativa (`AIAnalyzer`)

**Arquivo:** `ai_analyzer_qwen.py`

Responsável por:

- Interagir com:
  - GroqCloud (prioridade 1),
  - OpenAI (fallback),
  - DashScope (fallback),
  - Mock (fallback final).
- Construir prompt:
  - a partir de `ai_payload` estruturado (price_context, flow_context, orderbook_context, quant_model, etc.).
- Suporte a Structured Output:
  - JSON Mode + Pydantic (`AITradeAnalysis`).
- Núcleo:
  - `_analyze_internal` → constrói prompt, chama modelo, faz fallback para mock se necessário.
- Exposição:
  - `analyze(event_data)` → retorna:
    - `raw_response` (texto),
    - `structured` (dict ou None),
    - metadados (`tipo_evento`, `ativo`, `mode`, `model`, `success`, etc.).

Integração com:

- `health_monitor` (heartbeat periódico “ai”).
- logging estruturado:
  - `ai_provider_selected`, `ai_ping_ok/failed`,
  - `ai_analyze_ok`, `ai_analyze_error`.

### 6.3. Orquestração de IA (`ai_runner`)

**Arquivo:** `market_orchestrator/ai/ai_runner.py`

- `initialize_ai_async(bot)`:
  - roda em thread,
  - inicializa `AIAnalyzer` + `MLInferenceEngine`,
  - faz teste rápido de inferência.
- `run_ai_analysis_threaded(bot, event_data)`:
  - respeita:
    - `ai_rate_limiter`,
    - `ai_semaphore`,
    - pool de threads `ai_thread_pool`,
  - monta `ai_payload` (via `ai_payload_builder`),
  - chama `ai_analyzer.analyze(event_data)`,
  - salva evento de análise (`AI_ANALYSIS`) no `EventSaver`.

Tracing:

- spans `"ai_init"` e `"ai_analysis"`.

---

## 7. EventBus e HealthMonitor

### 7.1. EventBus (`event_bus.py`)

- Fila interna (`deque`), thread de processamento.
- Normalização numérica e de timestamps.
- Deduplicação de eventos por ID gerado de:
  - `timestamp`, `delta`, `volume`, `price`.
- `subscribe(event_type, handler)` / `publish(event_type, event_data)`.
- Logs estruturados:
  - `event_bus_started`, `event_normalization_error`,
  - `event_bus_process_error`, `event_bus_shutdown`.
- `get_stats()`:
  - tamanho da fila, tipos de eventos, contadores de normalização.

### 7.2. HealthMonitor (`health_monitor.py`)

- Recebe heartbeats por módulo:
  - `heartbeat("main")`, `heartbeat("ai")`, etc.
- Thread de monitoramento:
  - se um módulo fica em silêncio:
    - `warn_silence` → `WARNING`,
    - `critical_silence` → `CRITICAL`.
- Integração opcional com OCI (métricas de sistema).
- Logs estruturados:
  - `health_monitor_started`, `module_silence_warning/critical`,
  - `module_recovered`, `health_monitor_stopped`.
- `get_stats()`:
  - `monitored_modules`,
  - `heartbeats`, `active_critical_alerts`, `active_warning_alerts`.

---

Este overview deve ser suficiente para:

- on-board de novos devs,
- auditorias internas,
- e para você ter um mapa claro do fluxo ponta a ponta do bot.