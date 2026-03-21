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
| `.env.example` | Template de variaveis de ambiente |

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
| `fix_optimization.py` | `data_processing/fix_optimization.py` |
| `diagnose_optimization.py` | `scripts/diagnostics/diagnose_optimization.py` |
| `orderbook_fallback.py` | `orderbook_core/orderbook_fallback.py` |

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
├── trade_filter.py       # Filtro de trades
├── trade_timestamp_validator.py # Validador de timestamps
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
├── feature_store.py             # Store de features (Parquet particionado)
└── fix_optimization.py          # Limpeza de eventos (clean_event, simplify_historical_vp)
```

---

### `monitoring/` - Monitoramento e Sistema
```
monitoring/
├── __init__.py
├── time_manager.py        # Gerenciador de tempo (sincronizacao Binance)
├── health_monitor.py      # Monitor de saude do sistema
├── metrics_collector.py   # Coletor de metricas (Prometheus)
├── heartbeat_manager.py   # Gerenciador de heartbeats
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
├── ai_throttler.py            # Controlador de taxa de chamadas IA
├── ai_field_legend.py        # Legenda de campos do payload IA
├── technical_indicators.py   # Indicadores tecnicos (EMA, RSI, etc.)
├── ml_features.py             # Features de ML (cross-asset)
├── async_helpers.py           # Utilitarios async
├── exceptions.py              # Hierarquia unificada de excecoes (BotBaseError)
└── logging_config.py         # Logging centralizado (JSON/texto, rotativo)
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
├── absorption.py         # Deteccao de absorcao
├── aggregates.py         # Agregacao de dados (RollingAggregate)
├── constants.py          # Constantes do modulo
├── core.py               # Motor principal (FlowAnalyzer)
├── errors.py             # Tratamento de erros
├── logging_config.py     # Configuracao de logging
├── metrics.py            # Metricas e CircuitBreaker
├── profiling.py          # Memory e lock profiling
├── prometheus_metrics.py # Integracao Prometheus
├── protocols.py          # Definicoes de protocolos
├── serialization.py      # Serializacao (Decimal-safe JSON)
├── utils.py              # Utilitarios
├── validation.py         # Validacao de dados
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
├── tracing_utils.py
└── orderbook_fallback.py  # Fallback REST API com retry e circuit breaker
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

### `tests/` - Suite de Testes (~107 arquivos, organizado)
```
tests/
├── conftest.py                    # Fixtures globais + Prometheus cleanup
├── test_regression.py            # Testes de regressao
├── test_window_state.py          # Testes de estado de janela
├── fixtures/
│   └── sample_analysis_trigger.json
├── unit/                          # 30 testes unitarios (modulo isolado)
│   ├── test_event_bus.py
│   ├── test_flow_analyzer.py
│   ├── test_data_validator.py
│   ├── test_data_quality_validator.py
│   ├── test_cross_asset.py
│   ├── test_defense_zones.py
│   ├── test_circuit_breaker.py
│   ├── test_feature_store.py
│   ├── test_absorption_zone_mapper.py
│   ├── test_ai_response_validator.py
│   ├── test_config_imports.py
│   ├── test_orderbook_analyzer.py
│   ├── test_orderbook_helpers.py
│   ├── test_orderbook_validate_snapshot.py
│   ├── test_passive_aggressive_flow.py
│   ├── test_rolling_aggregate.py
│   ├── test_sr_strength.py
│   ├── test_support_resistance_consolidated.py
│   ├── test_support_resistance_modular.py
│   ├── test_patch_compressor.py
│   ├── test_patch_epoch_ms.py
│   ├── test_patch_guardrail.py
│   ├── test_patch_validator.py
│   ├── test_rate_limiter.py
│   ├── test_simple_correlations.py
│   ├── test_updated_correlations.py
│   ├── test_ml_frozen_detector.py
│   └── test_ai_analyzer_language_and_think_strip.py
├── integration/                   # 50+ testes de integracao (multiplos modulos)
│   ├── test_ai_runner.py
│   ├── test_ai_runner_comprehensive.py
│   ├── test_ai_analyzer_mock.py
│   ├── test_ai_llm_fallback_flow.py
│   ├── test_pipeline_integration.py
│   ├── test_orderbook_core_comprehensive.py
│   ├── test_orderbook_analyzer_comprehensive.py
│   ├── test_orderbook_analyzer_full_coverage.py
│   ├── test_orderbook_analyzer_coverage.py
│   ├── test_orderbook_analyzer_missing.py
│   ├── test_orderbook_wrapper_fallback.py
│   ├── test_orderbook_wrapper_fetch_with_retry.py
│   ├── test_orderbook_analyze_core.py
│   ├── test_orderbook_config_injection.py
│   ├── test_circuit_breaker_improvements.py
│   ├── test_circuit_breaker_integration.py
│   ├── test_cross_asset_integration.py
│   ├── test_enhanced_cross_asset.py
│   ├── test_dynamic_volume_profile_2.py
│   ├── test_data_pipeline.py
│   ├── test_trade_buffer_optimization.py
│   ├── test_trade_flow_analyzer.py
│   ├── test_risk_manager_comprehensive.py
│   ├── test_regime_integration.py
│   ├── test_regime_integration_legacy.py
│   ├── test_window_processor.py
│   ├── test_window_processor_queue.py
│   ├── test_update_histories.py
│   ├── test_out_of_order_pruning.py
│   ├── test_integration_full_flow.py
│   ├── test_enrich_signal.py
│   ├── test_enrich_simple.py
│   ├── test_enrich_correction.py
│   ├── test_enrich_event.py
│   ├── test_macro_data_provider.py
│   ├── test_integrated_macro_provider.py
│   ├── test_macro_singleton_fix.py
│   ├── test_institutional_alerts.py
│   ├── test_fixes_simple.py
│   ├── test_fixes_simple_fixed.py
│   ├── test_patch_2_fallback_controlado.py
│   ├── test_patch_2_simples.py
│   ├── test_patch_compressor_v3.py
│   ├── test_latency_fix_simple.py
│   ├── test_corrections.py
│   ├── test_optimization.py
│   ├── test_fix_optimization_storage.py
│   ├── test_event_saver_jsonl_guardian.py
│   ├── test_new_payload.py
│   └── test_invariant_fix.py
├── e2e/                           # 12 testes end-to-end (sistema completo)
│   ├── test_system_health.py
│   ├── test_performance_benchmarks.py
│   ├── test_websocket.py
│   ├── test_connection.py
│   ├── test_export_signals.py
│   ├── test_orchestrator_initialization.py
│   ├── test_market_orchestrator_comprehensive.py
│   ├── test_run_diagnosis.py
│   ├── test_diagnostic.py
│   ├── test_functions.py
│   ├── backtester.py
│   └── regime_scenario_tester.py
├── helpers/                       # Utilitarios de teste
│   ├── fixtures.py
│   ├── mock_ai_responses.py
│   ├── mock_qwen.py
│   ├── config_test.py
│   ├── fix_broken_tests.py
│   └── fix_qwen_import.py
├── legacy/                        # Testes antigos (pt-BR, verificacoes)
│   ├── teste_rapido.py
│   ├── teste_rapido_corrigido.py
│   ├── teste_separador.py
│   ├── teste_cross_asset_final.py
│   ├── verify_patch_2.py
│   ├── verify_prune_logic_only.py
│   └── verify_day4_implementations.py
└── payload/                       # Testes focados de payload
    ├── conftest.py
    ├── pytest.ini
    ├── test_payload_compressor.py
    ├── test_payload_guardrail.py
    ├── test_payload_tripwires.py
    ├── test_payload_optimizer.py
    ├── test_payload_metrics_aggregator.py
    ├── test_build_compact_v3.py
    └── test_ai_throttler_v2.py
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
├── deploy_oracle.sh
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
├── run_tests_windows.bat
├── run_tests_with_coverage.sh
├── setup_test_environment.sh
├── test_fixes.py
├── test_fixes_simple.py
├── test_fixes_final.py
├── validate_regime_system.py
├── validation_check.py
├── test_payload.ps1
├── test_payload.sh
├── debug/                          # Scripts de debug
│   ├── debug_bot.py
│   ├── debug_env.py
│   ├── debug_keyerror.py
│   ├── debug_payload.py
│   └── debug_validator.py
├── diagnostics/                    # Scripts de diagnostico
│   ├── analyze_ai_results.py
│   ├── auto_fix.py
│   ├── diagnose_crash.py
│   ├── diagnose_optimization.py
│   ├── evaluate_ai_performance.py
│   ├── final_replace.py
│   ├── final_validation.py
│   ├── performance_metrics.py
│   ├── replay_validator.py
│   ├── reproduce_issue.py
│   ├── show_problem_lines.py
│   ├── test_decision_system.py
│   ├── test_integrated.py
│   ├── test_latency.py
│   ├── test_ml_model.py
│   ├── validate_event.py
│   ├── verify_optimization.py
│   ├── verify_implementations.py
│   ├── verify_ml_integration.py
│   └── verify_patch.py
├── demos/                          # Demonstracoes
│   ├── demo_circuit_breaker.py
│   ├── demo_enhanced_cross_asset.py
│   └── demo_enhanced_cross_asset_simple.py
├── fixes/                          # Scripts de correcao
│   ├── fix_bot_run.py
│   ├── fix_broken_tests.py
│   ├── fix_duplicates.py
│   ├── fix_playwright.py
│   ├── fix_separator_final.py
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
├── ESTRUTURA_VISUAL_SISTEMA.md
├── README_OPTIMIZATION.md
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
| `utils/` | Proxy — modulos movidos para common/, monitoring/, trading/ |
| `database/` | Banco de dados (event_store.py) |
| `infrastructure/` | Docker, Terraform, OCI |
| `tools/` | Ferramentas (inspect_db, ws_test, groq tests) |
| `diagnostics/` | Proxy — modulos movidos para scripts/diagnostics/ |
| `diagnostic_files/` | Diagnostico de janelas |
| `.github/workflows/` | CI/CD (lint + unit tests + integration tests) |
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

- **Arquivos .py na raiz**: ~25 (29 proxies + 4 modulos de producao + config/main)
- **Pacotes organizados**: 7 novos + 12 pre-existentes
- **Total de arquivos Python**: ~250+
- **Testes**: ~107 arquivos em tests/ (unit/28, integration/50, e2e/12, helpers/7, legacy/7, payload/5)
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

## Atualizacoes Posteriores (2026-03-20)

| Categoria | Arquivos Adicionados |
|-----------|---------------------|
 | common/ | ai_throttler.py, ai_field_legend.py, async_helpers.py |
 | monitoring/ | heartbeat_manager.py |
 | trading/ | trade_filter.py, trade_timestamp_validator.py |
 | flow_analyzer/ | whale_score.py |
 | tests/ | ~65+ novos arquivos de teste |
 | scripts/ | +15 novos scripts |
 | docs/ | ESTRUTURA_VISUAL_SISTEMA.md, README_OPTIMIZATION.md |
 | tools/ | export_db_to_jsonl.py, test_groq_*.py |
 | infrastructure/ | market-bot.
 
 
 service, terraform/, oci/ |
 | core/ | state_manager.py, window_state.py |

---

*Ultima atualizacao: 2026-03-20 (atualizacao completa: novos arquivos em common/, monitoring/, trading/, tests/ expandidos para ~120 arquivos, scripts/ atualizados, docs/ adicionados, .github/workflows criado, infrastructure/, core/, diagnostic_files/, tools/ documentados)*
