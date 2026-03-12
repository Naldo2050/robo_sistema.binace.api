# Estrutura Completa do Sistema - Robo Binance API

## Visao Geral do Projeto

Sistema de trading automatizado para Binance com analise de fluxo de ordens, suporte/resistencia, deteccao de regime de mercado e integracao com IA.

---

## Raiz do Projeto (Root)

### Arquivos de Configuracao
| Arquivo | Descricao |
|---------|-----------|
| `.gitignore` | Configuracoes de gitignore |
| `.coveragerc` | Configuracao de coverage de testes |
| `.dockerignore` | Configuracao Docker ignore |
| `mypy.ini` | Configuracao de type checking |
| `pyproject.toml` | Configuracao do projeto Python |
| `pyrightconfig.json` | Configuracao do pyright |
| `pytest.ini` | Configuracao do pytest |
| `docker-compose.yml` | Orquestracao de containers |
| `Dockerfile` | Imagem Docker do projeto |
| `requirements.txt` | Dependencias Python |
| `requirements-dev.txt` | Dependencias de desenvolvimento |

### Arquivos Principais (Raiz)
| Arquivo | Descricao |
|---------|-----------|
| `main.py` | Ponto de entrada principal |
| `config.py` | Configuracoes globais |
| `config.json` | Arquivo de configuracao JSON |

### Modulos de Producao (Raiz)

Modulos que permanecem na raiz por terem muitos importadores, risco de import circular ou carregamento dinamico:

| Arquivo | Descricao | Razao |
|---------|-----------|-------|
| `ai_analyzer_qwen.py` | Analisador IA principal (150KB) | 8 importadores + circular com market_orchestrator/ai/ |
| `orderbook_analyzer.py` | Analisador de orderbook (123KB) | Carregado via importlib por orderbook_analyzer/ |
| `institutional_enricher.py` | Enriquecedor institucional (85KB) | Import dinamico em market_orchestrator |
| `build_compact_payload.py` | Construtor de payload compactado | 4 importadores + circular com market_orchestrator/ai/ |
| `orderbook_fallback.py` | Fallback do orderbook | 3 importadores, acoplado ao orderbook_analyzer |
| `fix_optimization.py` | Correcao de otimizacao (usado em producao) | 3 importadores em testes de producao |
| `diagnose_optimization.py` | Diagnostico de otimizacao | 1 importador em testes |

### Proxies de Compatibilidade (Raiz)

Arquivos pequenos (3-4 linhas) que redirecionam imports para os novos pacotes:

| Proxy | Redireciona para |
|-------|------------------|
| `event_bus.py` | `events/event_bus.py` |
| `event_saver.py` | `events/event_saver.py` |
| `event_memory.py` | `events/event_memory.py` |
| `trade_buffer.py` | `trading/trade_buffer.py` |
| `fred_fetcher.py` | `fetchers/fred_fetcher.py` |
| `cross_asset_correlations.py` | `market_analysis/cross_asset_correlations.py` |
| `dynamic_volume_profile.py` | `market_analysis/dynamic_volume_profile.py` |
| `levels_registry.py` | `market_analysis/levels_registry.py` |
| `data_handler.py` | `data_processing/data_handler.py` |
| `data_enricher.py` | `data_processing/data_enricher.py` |
| `data_validator.py` | `data_processing/data_validator.py` |
| `data_quality_validator.py` | `data_processing/data_quality_validator.py` |
| `time_manager.py` | `monitoring/time_manager.py` |
| `health_monitor.py` | `monitoring/health_monitor.py` |
| `metrics_collector.py` | `monitoring/metrics_collector.py` |
| `format_utils.py` | `common/format_utils.py` |
| `context_collector.py` | `fetchers/context_collector.py` |
| `enrichment_integrator.py` | `data_processing/enrichment_integrator.py` |
| `feature_store.py` | `data_processing/feature_store.py` |
| `export_signals.py` | `trading/export_signals.py` |
| `historical_profiler.py` | `market_analysis/historical_profiler.py` |
| `report_generator.py` | `common/report_generator.py` |
| `optimize_ai_payload.py` | `common/optimize_ai_payload.py` |
| `payload_optimizer_config.py` | `common/payload_optimizer_config.py` |
| `ai_payload_compressor.py` | `common/ai_payload_compressor.py` |
| `ai_response_validator.py` | `common/ai_response_validator.py` |

---

## Pacotes Organizados (NOVO - Reorganizacao 03/2026)

### `events/` - Sistema de Eventos
```
events/
├── __init__.py
├── event_bus.py          # Barramento de eventos
├── event_saver.py        # Persistencia de eventos (JSONL/JSON)
├── event_memory.py       # Memoria de eventos com OutcomeTracker
├── event_similarity.py   # Similaridade entre eventos
└── event_stats_model.py  # Modelo estatistico de eventos
```

---

### `trading/` - Trading e Execucao
```
trading/
├── __init__.py
├── trade_buffer.py       # AsyncTradeBuffer com backpressure
├── trade_validator.py    # Validacao de trades
├── export_signals.py     # Exportador de sinais para CSV/MQL5
├── alert_engine.py       # Motor de alertas
├── alert_manager.py      # Gerenciador de alertas
└── outcome_tracker.py    # Rastreador de resultados
```

---

### `fetchers/` - Coletores de Dados Externos
```
fetchers/
├── __init__.py
├── fred_fetcher.py          # Coletor do FRED API
├── context_collector.py     # Coletor de contexto (VIX, Fear&Greed, macro)
├── macro_data_fetcher.py    # Coletor de dados macroeconomicos
├── macro_fetcher.py         # Fetcher de macro alternativo
├── onchain_fetcher.py       # Coletor de dados on-chain
└── funding_aggregator.py    # Agregador de funding rates
```

---

### `market_analysis/` - Analise de Mercado
```
market_analysis/
├── __init__.py
├── cross_asset_correlations.py  # Correlacoes BTC/ETH/DXY/NDX
├── dynamic_volume_profile.py    # Perfil de volume dinamico
├── levels_registry.py           # Registro de niveis de preco
├── historical_profiler.py       # Profiler historico de volume
├── liquidity_heatmap.py         # Mapa de calor de liquidez
├── market_impact.py             # Analise de impacto de mercado
└── pattern_recognition.py       # Reconhecimento de padroes
```

---

### `data_processing/` - Processamento de Dados
```
data_processing/
├── __init__.py
├── data_handler.py              # Manipulador de dados (eventos, absorcao)
├── data_enricher.py             # Enriquecedor de dados
├── data_validator.py            # Validador de dados
├── data_quality_validator.py    # Validador de qualidade
├── enrichment_integrator.py     # Integrador de enriquecimento
└── feature_store.py             # Store de features (Parquet particionado)
```

---

### `monitoring/` - Monitoramento e Sistema
```
monitoring/
├── __init__.py
├── time_manager.py        # Gerenciador de tempo (sincronizacao Binance)
├── health_monitor.py      # Monitor de saude do sistema
├── metrics_collector.py   # Coletor de metricas (Prometheus)
├── clock_sync.py          # Sincronizacao de relogio
├── websocket_handler.py   # Manipulador WebSocket
└── orderbook_ws_manager.py # Gerenciador WebSocket do orderbook
```

---

### `common/` - Utilitarios Comuns
```
common/
├── __init__.py
├── format_utils.py            # Formatacao de precos, quantidades, percentuais
├── report_generator.py        # Gerador de relatorios
├── optimize_ai_payload.py     # Otimizador de payload IA
├── payload_optimizer_config.py # Configuracao do otimizador
├── ai_payload_compressor.py   # Compressor de payload IA
├── ai_response_validator.py   # Validador de respostas IA
├── technical_indicators.py    # Indicadores tecnicos (EMA, RSI, etc.)
└── ml_features.py             # Features de ML (cross-asset)
```

---

## Modulos Principais (Pre-existentes)

### `ai_runner/` - Executor de IA
```
ai_runner/
├── __init__.py
├── ai_runner.py         # Executor principal de IA
└── exceptions.py        # Excecoes especificas
```

---

### `flow_analyzer/` - Analise de Fluxo de Ordens
```
flow_analyzer/
├── __init__.py
├── absorption.py        # Deteccao de absorcao
├── aggregates.py        # Agregacao de dados (RollingAggregate)
├── constants.py         # Constantes do modulo
├── core.py              # Motor principal (FlowAnalyzer)
├── errors.py            # Tratamento de erros
├── logging_config.py    # Configuracao de logging
├── metrics.py           # Metricas e CircuitBreaker
├── profiling.py         # Memory e lock profiling
├── prometheus_metrics.py # Integracao Prometheus
├── protocols.py         # Definicoes de protocolos
├── serialization.py     # Serializacao (Decimal-safe JSON)
├── utils.py             # Utilitarios
├── validation.py        # Validacao de dados
└── whale_score.py       # Score de whales
```

---

### `market_orchestrator/` - Orquestrador Principal
```
market_orchestrator/
├── __init__.py
├── market_orchestrator.py  # Orquestrador principal (87KB)
├── orchestrator.py         # Orquestrador base (26KB)
├── ai/
│   ├── __init__.py
│   ├── ai_enrichment_context.py   # Contexto de enriquecimento
│   ├── ai_payload_builder.py       # Construtor de payload (50KB)
│   ├── ai_runner.py                # Executor de IA (31KB)
│   ├── llm_payload_guardrail.py   # Guardrails de payload
│   ├── llm_response_validator.py  # Validador de respostas LLM
│   ├── payload_compressor.py      # Compressor v1
│   ├── payload_compressor_v3.py   # Compressor v3 (39KB)
│   ├── payload_metrics_aggregator.py
│   ├── payload_section_cache.py   # Cache de secoes
│   └── raw_event_deduplicator.py  # Deduplicador de eventos
├── analysis/
│   ├── __init__.py
│   └── institutional_analytics.py
├── connection/
│   └── robust_connection.py       # Conexao robusta com reconnect
├── flow/
│   ├── __init__.py
│   ├── risk_manager.py            # Gerenciamento de risco
│   ├── signal_processor.py        # Processador de sinais
│   ├── trade_executor.py          # Execucao de trades
│   └── trade_flow_analyzer.py
├── orderbook/
│   ├── __init__.py
│   └── orderbook_wrapper.py
├── signals/
│   ├── __init__.py
│   └── signal_processor.py
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   └── price_fetcher.py
└── windows/
    ├── __init__.py
    └── window_processor.py        # Processador de janelas
```

---

### `support_resistance/` - Suporte e Resistencia
```
support_resistance/
├── __init__.py
├── config.py              # Configuracoes
├── constants.py           # Constantes
├── core.py                # Motor principal
├── defense_zones.py       # Zonas de defesa
├── monitor.py             # Monitor em tempo real
├── pivot_points.py        # Pontos de pivo
├── reference_prices.py    # Precos de referencia
├── sr_strength.py         # Forca de S/R
├── system.py              # Sistema completo
├── utils.py               # Utilitarios
├── validation.py          # Validacao
└── volume_profile.py      # Perfil de volume
```

---

### `ml/` - Machine Learning
```
ml/
├── feature_calculator.py   # Calculador de features
├── generate_dataset.py     # Geracao de datasets
├── hybrid_decision.py      # Decisao hibrida (ML + IA)
├── inference_engine.py     # Motor de inferencia
├── model_inference.py      # Inferencia XGBoost
├── train_model.py          # Treinamento de modelo
├── datasets/
│   └── training_dataset.parquet
└── models/
    ├── xgb_model_*.json
    ├── model_metadata_latest.json
    └── feature_importance_*.csv
```

---

### `data_pipeline/` - Pipeline de Dados
```
data_pipeline/
├── __init__.py
├── config.py
├── logging_utils.py
├── pipeline.py              # Pipeline principal por janela
├── cache/
│   ├── __init__.py
│   ├── buffer.py
│   └── lru_cache.py
├── fallback/
│   ├── __init__.py
│   └── registry.py
├── metrics/
│   ├── __init__.py
│   ├── data_quality_metrics.py
│   └── processor.py
└── validation/
    ├── __init__.py
    ├── adaptive.py
    └── validator.py
```

---

### `orderbook_core/` - Nucleo do Orderbook
```
orderbook_core/
├── __init__.py
├── circuit_breaker.py
├── constants.py
├── event_factory.py
├── exceptions.py
├── metrics.py
├── orderbook_config.py
├── orderbook.py
├── protocols.py
├── structured_logging.py
└── tracing_utils.py
```

---

### `orderbook_analyzer/` - Analisador de Orderbook (pacote)
```
orderbook_analyzer/
├── __init__.py
├── analyzer.py
├── spread_tracker.py
└── config/
    ├── __init__.py
    └── settings.py
```

---

### `risk_management/` - Gerenciamento de Risco
```
risk_management/
├── __init__.py
├── exceptions.py
└── risk_manager.py
```

---

### `config/` - Configuracoes
```
config/
├── __init__.py
└── model_config.yaml     # Config LLM payload e XGBoost
```

---

### `auto_fixer/` - Sistema de Auto-correcao
```
auto_fixer/
├── __init__.py
├── ai_client.py
├── apply_safe_fixes.py
├── config.json
├── fix_bugs.py
├── fix_high_issues.py
├── runner.py
├── scheduler.py
├── test_runner.py
├── validate_installation.py
├── view_issues.py
├── feedback/
│   └── fix_tracker.py
├── monitor/
│   ├── __init__.py
│   ├── file_watcher.py
│   ├── health_monitor.py
│   └── log_watcher.py
├── output/
│   ├── analysis_results/
│   ├── backups_high/
│   ├── chunks/
│   ├── patches/
│   ├── reports/
│   └── vectordb/
├── phase1_scanner/
├── phase2_extractor/
├── phase3_chunker/
├── phase4_index/
├── phase5_rag/
├── phase6_analyzers/
├── phase7_patcher/
└── phase8_reporter/
```

---

### `src/` - Codigo Fonte (Regime, Macro, Bridges)
```
src/
├── analysis/
│   ├── ai_payload_integrator.py
│   ├── integrate_regime_detector.py
│   ├── regime_detector.py
│   └── regime_integration.py
├── bridges/
│   ├── __init__.py
│   └── async_bridge.py
├── data/
│   ├── indices_futures.csv
│   ├── macro_data.json
│   └── macro_data_provider.py
├── rules/
│   └── regime_rules.py
├── services/
│   ├── __init__.py
│   ├── macro_service.py
│   └── macro_update_service.py
└── utils/
    ├── __init__.py
    ├── ai_payload_optimizer.py
    ├── async_helpers.py
    └── types_fredapi.pyi
```

---

## Diretorios de Suporte

### `tests/` - Suite de Testes (105 arquivos)
```
tests/
├── conftest.py                    # Fixtures globais + Prometheus cleanup
├── fixtures.py
├── fixtures/
│   └── sample_analysis_trigger.json
├── mock_ai_responses.py
├── mock_qwen.py
├── payload/                       # Testes focados de payload
│   ├── conftest.py
│   ├── pytest.ini
│   ├── test_payload_compressor.py
│   ├── test_payload_guardrail.py
│   ├── test_payload_metrics_aggregator.py
│   ├── test_payload_optimizer.py
│   └── test_payload_tripwires.py
├── test_ai_*.py                   # Testes de IA (7 arquivos)
├── test_orderbook_*.py            # Testes de orderbook (9 arquivos)
├── test_flow_*.py                 # Testes de fluxo
├── test_support_resistance_*.py   # Testes de S/R
├── test_enrich_*.py               # Testes de enriquecimento
├── test_data_*.py                 # Testes de dados
├── test_circuit_breaker_*.py      # Testes de circuit breaker
├── test_event_*.py                # Testes de eventos
├── test_trade_*.py                # Testes de trading
├── teste_*.py                     # Testes em portugues (legacy)
└── ... (105 arquivos total)
```

---

### `scripts/` - Scripts de Utilidade
```
scripts/
├── ab_test_prompt_styles.py
├── analyze_ai_usage.py
├── app.py                          # Aplicacao web
├── audit_json_payload_costs.py
├── audit_new_features.py
├── audit_script.py
├── backup_to_oci.py
├── dashboard.py                    # Dashboard (43KB)
├── disaster_recovery.sh
├── enhanced_market_bot.py
├── full_audit.py
├── integration_validator.py
├── log_formatter.py
├── log_sanitizer.py
├── modelo_dados_ideal.py
├── process_csv_data.py
├── prometheus_exporter.py
├── remote_health_check.sh
├── validate_regime_system.py
├── validation_check.py
├── debug/                          # Scripts de debug
│   ├── debug_bot.py
│   ├── debug_env.py
│   ├── debug_keyerror.py
│   ├── debug_payload.py
│   └── debug_validator.py
├── diagnostics/                    # Scripts de diagnostico
│   ├── diagnose_crash.py
│   ├── final_replace.py
│   ├── reproduce_issue.py
│   ├── show_problem_lines.py
│   ├── validar_evento.py
│   ├── verificar_otimizacao.py
│   ├── verify_implementations.py
│   └── verify_patch.py
├── demos/                          # Demonstracoes
│   ├── demo_circuit_breaker.py
│   ├── demo_enhanced_cross_asset.py
│   └── demo_enhanced_cross_asset_simple.py
├── fixes/                          # Scripts de correcao
│   ├── fix_bot_run.py
│   ├── fix_broken_tests.py
│   ├── fix_duplicatas.py
│   ├── fix_playwright.py
│   ├── fix_separador_final.py
│   └── fix_timestamp.py
└── structure/                      # Analise de estrutura
    ├── compare_structure.py
    ├── compare_structure_filtered.py
    ├── create_structure.py
    ├── find_missing_files.py
    ├── generate_updated_structure.py
    └── list_project_files.py
```

---

### `legacy/` - Codigo Legado
```
legacy/
├── ai_analyzer_disabled.py
├── ai_analyzer_qwen_patch2.py
├── ai_historical_pro.py
├── data_pipeline_legacy..py
├── main.patched.py
├── market_analyzer.py
├── market_analyzer_2_3_0.py
├── patch_ai_analyzer.py
└── support_resistance_legacy.py
```

---

### `docs/` - Documentacao
```
docs/
├── architecture.md
├── RUNBOOK.md
├── troubleshooting.md
├── CORRECAO_ENRICH_EVENT_SUMMARY.md
├── CORRECAO_FETCH_INTERMARKET_DATA.md
├── PATCH_SUMMARY.md
├── RELATORIO_ENRICHMENT_CROSS_ASSET.md
├── RELATORIO_FINAL_MACRO_PROVIDER.md
├── RESUMO_EXPORT_SINAIS.md
├── auditoria_estrutura_json.md
├── orderbook_severity_analysis.md
└── relatorio_auditoria_json.md
```

---

### Outros Diretorios

| Diretorio | Descricao |
|-----------|-----------|
| `utils/` | Utilitarios (async_helpers, heartbeat, trade_filter) |
| `database/` | Banco de dados (event_store.py) |
| `infrastructure/` | Docker, Terraform, OCI |
| `tools/` | Ferramentas (inspect_db, ws_test, groq tests) |
| `diagnostics/` | Diagnosticos (performance, replay, ML) |
| `arquivos para diagnostico/` | Diagnostico de janelas |
| `Regras/` | Documentacao de regras (.odt, .docx) |
| `memory/` | Sistema de memoria (levels_BTCUSDT.json) |
| `MQL5/` | Integracao MetaTrader |
| `fallback_events/` | Eventos de fallback |
| `backups/` | Backups de seguranca |

---

## Arquivos de Dados

| Diretorio | Conteudo |
|-----------|----------|
| `dados/` | eventos_fluxo.jsonl, trading_bot.db (SQLite) |
| `logs/` | last_llm_payload.json, payload_metrics.jsonl |
| `features/` | Dados particionados por data (date=YYYY-MM-DD/) |

---

## Arquitetura de Alto Nivel

```
┌─────────────────────────────────────────────────────────────┐
│                     MAIN.PY                                  │
│                  (Ponto de Entrada)                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
 ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
 │   MARKET    │  │    AI RUNNER     │  │   FLOW      │
 │ ORCHESTRATOR│  │   (Analise IA)   │  │  ANALYZER   │
 └─────────────┘  └──────────────────┘  └─────────────┘
         │                │                │
         ▼                ▼                ▼
 ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
 │  EVENTS     │  │   TRADING        │  │  MONITORING │
 │  (eventos)  │  │  (buffer/alerts) │  │  (health)   │
 └─────────────┘  └──────────────────┘  └─────────────┘
         │                │                │
         ▼                ▼                ▼
 ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
 │  DATA       │  │   MARKET         │  │  FETCHERS   │
 │ PROCESSING  │  │  ANALYSIS        │  │  (externo)  │
 └─────────────┘  └──────────────────┘  └─────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
              ┌─────────────────────┐
              │   ORDERBOOK CORE    │
              │   + S/R + ML        │
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   DATABASE/LOGS     │
              │   (Persistencia)    │
              └─────────────────────┘
```

---

## Dependencias Principais

- **Binance**: `binance-connector`, `python-binance`
- **IA/ML**: `openai` (Groq), `xgboost`
- **Dados**: `pandas`, `numpy`, `polars`
- **Async**: `asyncio`, `aiohttp`, `websockets`
- **Database**: `sqlalchemy`, `sqlite3`, `orjson`
- **Monitoring**: `prometheus-client`, `structlog`
- **Testing**: `pytest`, `pytest-asyncio`, `coverage`
- **Macro**: `yfinance`, `fredapi`

---

## Estatisticas do Projeto

- **Arquivos .py na raiz**: ~25 (26 proxies + 7 modulos de producao + config/main)
- **Pacotes organizados**: 7 novos + 12 pre-existentes
- **Total de arquivos Python**: ~250+
- **Testes**: 105 arquivos em tests/
- **Dados de features**: 34+ datas

---

## Historico de Reorganizacao (2026-03-12)

| Etapa | Arquivos | Destino |
|-------|----------|---------|
| Testes da raiz | 37 | `tests/` |
| Debug/diagnostico | 28 | `scripts/debug\|diagnostics\|structure\|demos\|fixes` |
| Relatorios .md | 9 | `docs/` |
| Auditorias | 3 | `scripts/` |
| Disabled/patches IA | 4 | `legacy/` |
| Scripts standalone | 12 | `scripts/` e `legacy/` |
| Eventos | 5 | `events/` (com proxies) |
| Trading | 5 | `trading/` (com proxy) |
| Fetchers | 5 | `fetchers/` (com proxy) |
| Market analysis | 6 | `market_analysis/` (com proxies) |
| Data processing | 4 | `data_processing/` (com proxies) |
| Monitoring | 6 | `monitoring/` (com proxies) |
| Common utils | 3 | `common/` (com proxy) |
| Producao (batch 2) | 10 | `fetchers/`, `data_processing/`, `trading/`, `market_analysis/`, `common/` (com proxies) |

**Total movido: ~140 arquivos. Raiz: 129 -> ~25 (-81%)**

---

*Ultima atualizacao: 2026-03-12 (pos-reorganizacao)*
