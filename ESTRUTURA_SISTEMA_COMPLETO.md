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
| `config.json` | Arquivo de configuracao JSON |

### Modulos de Producao (Raiz)

Modulos que permanecem na raiz por terem muitos importadores, risco de import circular ou carregamento dinamico:

| Arquivo | Descricao | Razao |
|---------|-----------|-------|
| `ai_analyzer_qwen.py` | **REMOVIDO (2026-08-06)** — movido para `market_orchestrator/ai/analyzer_qwen.py` (commit `d92c02f`) | ~8 importadores |
| `build_compact_payload.py` | **REMOVIDO (2026-08-06)** — movido para `market_orchestrator/ai/payload_builder_compact.py` (commit `106b4b7`) | ~4 importadores |

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

### Shims Deprecated (Raiz)

Arquivos que agora são apenas proxies com `DeprecationWarning`, apontando para o conteúdo real nos pacotes:

| Shim | Redireciona para | Observacao |
|------|------------------|------------|
| `orderbook_analyzer.py` | `orderbook_analyzer/core.py` | Conteudo integral movido (v2.2.0); shim emite DeprecationWarning |
| `institutional_enricher.py` | `institutional/enricher.py` | Conteudo integral movido; shim emite DeprecationWarning |

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

### `institutional/` - Analise Institucional

```
institutional/
├── __init__.py
├── absorption_detector.py
├── base.py
├── confluence_engine.py
├── crypto_cot.py
├── cvd.py
├── enricher.py              # Conteudo de institutional_enricher.py (raiz, migrado)
├── entropy_analyzer.py
├── event_bridge.py
├── footprint.py
├── fourier_cycles.py
├── garch_volatility.py
├── hurst_exponent.py
├── iceberg_detector.py
├── kalman_filter.py
├── market_regime_hmm.py
├── mean_reversion.py
├── monte_carlo.py
├── order_flow_imbalance.py
├── smart_money.py
├── vwap_twap.py
├── whale_detector.py
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
│   ├── analyzer_qwen.py            # Analisador IA principal (ex-ai_analyzer_qwen.py raiz, movido 2026-08-06)
│   ├── llm_payload_guardrail.py   # Guardrails de payload
│   ├── llm_response_validator.py  # Validador de respostas LLM
│   ├── payload_builder_compact.py # Construtor de payload compactado (ex-build_compact_payload.py raiz)
│   ├── payload_compressor.py      # Compressor v1
│   ├── payload_compressor_v3.py   # Compressor v3 (39KB)
│   ├── payload_metrics_aggregator.py
│   ├── payload_section_cache.py   # Cache de secoes
│   ├── raw_event_deduplicator.py  # Deduplicador de eventos
│   ├── payload_sections/
│   │   ├── __init__.py
│   │   ├── flow_summary.py
│   │   ├── institutional_summary.py
│   │   ├── quality_summary.py
│   │   ├── regime_summary.py
│   │   ├── skill_bridge.py
│   │   └── sr_summary.py
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
├── __init__.py            # Re-export direto (zero importlib); SimplifiedOrderBookAnalyzer via __getattr__ lazy
├── core.py                # OrderBookAnalyzer v2.2.0 (conteudo de orderbook_analyzer.py raiz, migrado)
├── legacy_simplified.py   # Implementacao simplificada legada (SimplifiedOrderBookAnalyzer, DEPRECATED)
├── analyzer.py            # Shim de compatibilidade (DeprecationWarning) -> legacy_simplified
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

### `src_old/` - Codigo Fonte (Regime, Macro, Bridges) — ARQUIVADO

> Renomeado de `src/` em 2026-08-06 (commit `e8858c5`) — camada de proxy eliminada, imports corrigidos para importar diretamente dos pacotes reais. Mantido apenas como referencia historica; NAO e importado em producao.
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
│   ├── test_ai_analyzer_language_and_think_strip.py
└── test_simple_correlations.py
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
│   ├── test_invariant_fix.py
├── test_ai_llm_fallback_flow.py
├── test_ai_runner_comprehensive.py
├── test_data_pipeline.py
├── test_enhanced_cross_asset.py
├── test_enrich_signal.py
├── test_latency_fix_simple.py
├── test_macro_data_provider.py
├── test_orderbook_analyzer_coverage.py
├── test_orderbook_analyzer_full_coverage.py
├── test_orderbook_analyzer_missing.py
├── test_orderbook_config_injection.py
├── test_orderbook_core_comprehensive.py
├── test_patch_2_fallback_controlado.py
├── test_risk_manager_comprehensive.py
├── test_trade_buffer_optimization.py
└── test_window_processor.py
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
    ├── test_ai_throttler_v2.py
├── test_build_compact_payload_budget.py
├── test_build_compact_payload_fixes.py
├── test_build_compact_payload_pending_regressions.py
├── test_build_compact_payload_scenarios.py
├── test_build_compact_payload_smoke.py
├── test_build_compact_payload_snapshot.py
├── test_payload_integration_e2e.py
├── test_payload_sections.py
└── test_skill_bridge.py
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
├── diagnostics/                    # Scripts de diagnostico (40)
│   ├── analyze_ai_results.py
│   ├── audit_absorption_duplo_disparo_test.py
│   ├── audit_absorption_prod_test.py
│   ├── audit_cvd_numeric_test.py
│   ├── audit_cvd_reset_divergence_test.py
│   ├── audit_flow_imbalance_prod_test.py
│   ├── audit_live_data_invariants.py   # SEM commit (working tree)
│   ├── audit_market_data.py
│   ├── audit_ofi_numeric_test.py
│   ├── audit_support_resistance_test.py
│   ├── audit_va_poc_outward.py
│   ├── auto_fix.py
│   ├── capture_compact_sr.py           # SEM commit (working tree)
│   ├── data_health_check.py
│   ├── diagnose_crash.py
│   ├── diagnose_optimization.py
│   ├── evaluate_ai_performance.py
│   ├── final_replace.py
│   ├── final_validation.py
│   ├── map_pbc_keys.py                 # SEM commit (working tree)
│   ├── measure_dedup_effect.py
│   ├── measure_out_of_order_trades.py
│   ├── performance_metrics.py
│   ├── replay_etapa6_j1j4.py           # SEM commit (working tree)
│   ├── replay_j4_scorer.py             # SEM commit (working tree)
│   ├── replay_validator.py
│   ├── reproduce_issue.py
│   ├── run_production_observation.py
│   ├── run_shadow_observation.py       # SEM commit (working tree)
│   ├── show_problem_lines.py
│   ├── test_decision_system.py
│   ├── test_integrated.py
│   ├── test_latency.py
│   ├── test_ml_model.py
│   ├── validate_event.py
│   ├── validate_production_run.py
│   ├── verify_implementations.py
│   ├── verify_ml_integration.py
│   ├── verify_optimization.py
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
├── relatorio_auditoria_json.md
├── audit/
│   ├── FASE1_IMPORTS_PROXIES.md
│   ├── FASE2_ASYNC_WEBSOCKET.md
│   ├── FASE3_ERROS_RESILIENCIA.md
│   ├── FASE4_CONFIG_SEGURANCA.md
│   ├── FASE5_AI_ML_PIPELINE.md
│   ├── FASE6_TESTES.md
│   ├── FASE7_8_PERFORMANCE_ESTADO.md
│   ├── FASE9_10_DEPS_DOCKER_DADOS.md
│   ├── RELATORIO_FINAL.md
│   ├── SUMARIO_EXECUTIVO_AUDITORIA_2026-08.md
│   ├── AUDITORIA_PIVOT_POINTS_2026-08-09.md
│   ├── RELATORIO_OBSERVACAO_2026-08-09.md
│   ├── RELATORIO_FLOW_INVARIANTS_2026-08-10.md
│   ├── COMMIT_FLOW_INVARIANTS_2026-08-10.md
│   ├── EVENT_BUS_METRICS_FLAKE_2026-08-10.md
│   ├── ETAPA_5B_HLC_VP_SEMANTICA_2026-08-11.md
│   └── ETAPA_6_MACRO_NAN_FRED_CACHE_2026-08-11.md
```

---

### Outros Diretorios

| Diretorio | Descricao |
|-----------|-----------|
| `utils/` | Proxy apenas — so `__init__.py` (reexporta de common/, monitoring/, trading/); duplicatas removidas |
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
- **Pacotes organizados**: 8 novos + 12 pre-existentes
- **Total de arquivos Python**: ~250+
- **Testes**: ~161 arquivos em tests/ (unit/79, integration/55, e2e/10, payload/17)
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

## Atualizacoes Posteriores (2026-03-23)

| Categoria | Arquivos Adicionados |
|-----------|---------------------|
| fetchers/ | macro_data_provider.py, macro_service.py, macro_update_service.py |
| market_analysis/ | integrate_regime_detector.py, regime_detector.py, regime_integration.py, regime_rules.py |
| tests/unit/ | test_support_resistance_consolidated.py |
| tests/e2e/ | regime_scenario_tester.py |
| tests/integration/ | test_ai_analyzer_mock.py, test_corrections.py, test_dynamic_volume_profile_2.py, test_enrich_correction.py, test_event_saver_jsonl_guardian.py, test_fix_optimization_storage.py, test_fixes_simple_fixed.py, test_institutional_alerts.py, test_integrated_macro_provider.py, test_invariant_fix.py, test_latency_fix_simple.py, test_macro_singleton_fix.py, test_new_payload.py, test_optimization.py, test_orderbook_analyze_core.py, test_orderbook_config_injection.py, test_out_of_order_pruning.py, test_patch_2_fallback_controlado.py, test_patch_2_simples.py, test_patch_compressor_v3.py, test_regime_integration.py, test_regime_integration_legacy.py, test_window_processor_queue.py |
| tests/payload/ | test_ai_throttler_v2.py, test_build_compact_v3.py, test_payload_metrics_aggregator.py |
| scripts/ | error_monitor.py, validate_regime_system.py, validation_check.py, enhanced_market_bot.py |
| scripts/migration/ | commit_etapa2.sh, commit_etapa3.sh, etapa0_baseline.sh, etapa1_dependency_map.py, etapa2_check_exceptions.py, etapa3_check_duplicates.py, etapa4_check_src.py, etapa5_proxy_eliminator.py, etapa6_check_config.py, etapa7_check_contracts.py, etapa8_check_language.py, validate_after_step.sh, validate_all_final.py, validate_fix3_dedup.py, validate_fix4_fake_data.py |
| scripts/diagnostics/ | diagnose_crash.py, final_replace.py, reproduce_issue.py, show_problem_lines.py, validate_event.py, verify_implementations.py, verify_optimization.py, verify_patch.py |
| scripts/structure/ | compare_structure_filtered.py |
| infrastructure/ | market-bot.service, oci/monitoring.py, oci/security_config.md, oci/vault_helper.py, terraform/main.tf |
| tools/ | test_groq_models_http.py, test_groq_models_v2.py, test_groq_official.py, inspect_events_schema.py |
| .github/workflows/ | deploy_oci.yml |
| MQL5/Indicators/ | ChartSignalsFromCSV.mq5 |
| config/ | settings.py |
| ml/ | hybrid_decision.py, inference_engine.py, model_metadata.json, model_metadata_latest.json |
| orderbook_core/ | event_factory.py, structured_logging.py, tracing_utils.py |
| orderbook_analyzer/config/ | settings.py |
| monitoring/ | orderbook_ws_manager.py (placeholder) |

---

## Atualizacoes Posteriores (2026-04-02)

| Categoria | Arquivos Adicionados |
|-----------|---------------------|
| common/ | yfinance_cache.py |
| dados/ | fred_cache.json |
| institutional/ | __init__.py, absorption_detector.py, base.py, confluence_engine.py, crypto_cot.py, cvd.py, enricher.py, entropy_analyzer.py, event_bridge.py, footprint.py, fourier_cycles.py, garch_volatility.py, hurst_exponent.py, iceberg_detector.py, kalman_filter.py, market_regime_hmm.py, mean_reversion.py, monte_carlo.py, order_flow_imbalance.py, smart_money.py, vwap_twap.py, whale_detector.py |
| ml/ | bias_monitor.py, dataset_collector.py |
| tests/unit/ | test_ai_throttler_v3.py, test_data_invariants.py, test_institutional_absorption.py, test_institutional_base.py, test_institutional_confluence.py, test_institutional_cot.py, test_institutional_cvd.py, test_institutional_entropy.py, test_institutional_footprint.py, test_institutional_fourier.py, test_institutional_garch.py, test_institutional_hmm.py, test_institutional_hurst.py, test_institutional_iceberg.py, test_institutional_kalman.py, test_institutional_mean_reversion.py, test_institutional_monte_carlo.py, test_institutional_ofi.py, test_institutional_smart_money.py, test_institutional_vwap.py, test_institutional_whale.py, test_ml_bias_monitor.py, test_time_manager_async.py, test_yfinance_cache.py |

---

## Atualizacoes Posteriores (2026-04-03)

| Categoria | Arquivos Adicionados/Modificados |
|-----------|---------------------|
| raiz | flow_analyzer.py |
| tests/ | test_support_resistance_institutional.py, test_support_resistance_consolidated.py, test_support_resistance_modular.py, test_flow_analyzer.py, test_rolling_aggregate.py, test_out_of_order_pruning.py, verify_patch_2.py, verify_prune_logic_only.py |
| flow_analyzer/ | errors.py, logging_config.py, profiling.py, prometheus_metrics.py, protocols.py, serialization.py, utils.py, validation.py |
| support_resistance/ | config.py, constants.py, system.py |

---

## Atualizacoes Posteriores (2026-04-04)

| Categoria | Arquivos Adicionados |
|-----------|---------------------|
| raiz | flow_analyzer.py |
| tests/ | test_support_resistance_institutional.py, test_support_resistance_consolidated.py, test_support_resistance_modular.py, test_flow_analyzer.py, test_rolling_aggregate.py, test_out_of_order_pruning.py, verify_patch_2.py, verify_prune_logic_only.py |
| flow_analyzer/ | errors.py, logging_config.py, profiling.py, prometheus_metrics.py, protocols.py, serialization.py, utils.py, validation.py |
| support_resistance/ | config.py, constants.py, system.py |

---

*Ultima atualizacao: 2026-04-04 (novos arquivos: flow_analyzer.py completo, testes adicionais, atualizacoes support_resistance)*

---

## Atualizacoes Posteriores (2026-04-06)

| Categoria | Arquivos Adicionados |
|-----------|---------------------|
| .claude/ | settings.json |
| .github/ | AGENTES_CUSTOMIZADOS.md, AGENTES_INSTALACAO.md, AGENTES_REFERENCIA_RAPIDA.md, agents/ |
| Regras/ | AGENTES DE IA.docx |
| raiz | ARQUIVOS_ALTERADOS_DETALHADO.md, MENSAGEM_COMMIT_SUGERIDA.md, VALIDACAO_FINAL_RESUMO.md |
| common/ | ai_payload_types.py, ai_protocols.py |
| market_orchestrator/ | adapters.py, protocols.py |
| tests/payload/ | test_ai_payload_types.py |
| tests/unit/ | test_architecture_regressions.py, test_orchestrator_adapters.py |

---

*Ultima atualizacao: 2026-04-06 (novos arquivos: agentes IA, validacao final, tipos payload, adapters orchestrator)*

---

## Atualizacoes Posteriores (2026-03-24 até 2026-04-07)

✅ **ATUALIZACAO CONFIRMADA VIA GIT LOG - TODOS ARQUIVOS CRIADOS/MODIFICADOS DEPOIS DE 23/03/2026**

| Categoria | Arquivos Adicionados |
|-----------|---------------------|
| **NOVO PACOTE INSTITUCIONAL** | 21 arquivos completos: `institutional/__init__.py`, absorption_detector.py, base.py, confluence_engine.py, crypto_cot.py, cvd.py, enricher.py, entropy_analyzer.py, event_bridge.py, footprint.py, fourier_cycles.py, garch_volatility.py, hurst_exponent.py, iceberg_detector.py, kalman_filter.py, market_regime_hmm.py, mean_reversion.py, monte_carlo.py, order_flow_imbalance.py, smart_money.py, vwap_twap.py, whale_detector.py |
| **market_orchestrator/ai** | payload_sections/ COMPLETO: __init__.py, flow_summary.py, institutional_summary.py, quality_summary.py, regime_summary.py, skill_bridge.py, sr_summary.py |
| **flow_analyzer/** | errors.py, protocols.py, utils.py, validation.py, serialization.py, profiling.py, logging_config.py, prometheus_metrics.py, aggregates.py |
| **support_resistance/** | system.py, constants.py, config.py |
| **common/** | ml_features.py, technical_indicators.py, yfinance_cache.py, ai_throttler.py |
| **fetchers/** | funding_aggregator.py, onchain_fetcher.py, macro_data_provider.py |
| **monitoring/** | clock_sync.py, time_manager.py, health_monitor.py atualizado |
| **ml/** | hybrid_decision.py, inference_engine.py, bias_monitor.py, dataset_collector.py |
| **docs/audit/** | 10 documentos completos de auditoria: FASE1 a FASE10 + RELATORIO_FINAL.md |
| **tests/payload/** | +11 novos arquivos de teste payload |
| **tests/integration/** | +14 arquivos de integracao novos |
| **tests/unit/** | +27 novos testes (institutional completo, time_manager, yfinance_cache, ai_throttler) |
| **tests/raiz** | test_support_resistance_institutional.py, test_support_resistance_consolidated.py, test_flow_analyzer.py, test_rolling_aggregate.py, verify_patch_2.py, verify_prune_logic_only.py, test_out_of_order_pruning.py, test_support_resistance_modular.py |
| **dados/** | fred_cache.json, indices_futures.csv, macro_data.json |
| **config/** | model_config.yaml atualizado, settings.py |

---

## Atualizacoes (2026-08-06) — Fase 1: Consolidacao de modulos

| Categoria | Mudanca |
|-----------|---------|
| raiz | `institutional_enricher.py` e `orderbook_analyzer.py` viram **shims deprecated** (DeprecationWarning) apontando para `institutional/enricher.py` e `orderbook_analyzer/core.py` (conteudo integral, MD5 identico) |
| institutional/ | + `enricher.py` (conteudo migrado, 2202 linhas) |
| orderbook_analyzer/ | + `core.py` (v2.2.0, 2985 linhas), + `legacy_simplified.py` (era analyzer.py, alias `SimplifiedOrderBookAnalyzer`); `analyzer.py` vira shim de compat; `__init__.py` re-export direto (zero importlib; export lazy via `__getattr__`) |
| utils/ | Removidas duplicatas `trade_filter.py`, `heartbeat_manager.py`, `trade_timestamp_validator.py`, `async_helpers.py` (originais em common/, monitoring/, trading/); `__init__.py` proxy mantido |
| src/utils/ | Removido `async_helpers.py` (proxy morto) |
| raiz | Removidos `config.py` (proxy morto, `import config` resolve para `config/`), `ai_analyzer_qwen.py.bak`, artefato `coverage_html/flow_analyzer_py.html` |
| main.py | `from utils import HeartbeatManager` -> `from monitoring.heartbeat_manager import HeartbeatManager` |
| scripts/ | `test_fixes.py` e `test_fixes_simple.py`: `utils.async_helpers` -> `common.async_helpers` |
| market_orchestrator/ | `market_orchestrator.py:110` passa a importar `institutional.enricher` |
| tests/ | `test_market_orchestrator_comprehensive.py` e `test_orderbook_analyzer_comprehensive.py` usam `legacy_simplified.SimplifiedOrderBookAnalyzer` |

---

*✅ Ultima atualizacao REAL: 2026-08-06 | Fase 1 concluida — suites completas: 1557 passed, 0 failed*

---

## Atualizacoes (2026-08-08 a 2026-08-14) — Auditoria de Corrupcao Silenciosa + Fixes

| Categoria | Arquivos Adicionados/Modificados |
|-----------|---------------------------------|
| **common/** | `json_safe.py` (NOVO — sanitizador canonico NaN/±Inf→null, RFC 8259; commit `d448221`), `ai_field_legend.py`, `ai_payload_optimizer.py`, `payload_optimizer_config.py` |
| **database/** | `event_store.py` (sanitizacao RFC 8259 em save_event/save_batch) |
| **events/** | `event_saver.py` (sanitizacao jsonl/fallback + clean_event sem redondo) |
| **fetchers/** | `fred_fetcher.py` (timestamps UTC timezone-aware), `context_collector.py`, `macro_data_provider.py`, `macro_update_service.py` |
| **flow_analyzer/** | `core.py` (flow_imbalance min trades, T_raw OOO, CVD reset, metricas), `constants.py`, `metrics.py`, `aggregates.py` (eviction time-based + cap 5000), `absorption.py`, `validation.py` |
| **support_resistance/** | `volume_profile.py` (value area POC-outward contigua + fail-closed), `defense_zones.py`, `__init__.py` (pivots iloc[-2] periodo anterior) |
| **institutional/** | `enricher.py` (funding_rate_percent is-not-None + x100; dedup fonte canonica) |
| **market_orchestrator/** | `market_orchestrator.py` (T_raw/obs OOO, source tiering), `ai/payload_builder_compact.py` (cvd_4h, guardrails non-finite), `ai/analyzer_qwen.py`, `ai/llm_payload_guardrail.py` (sanitiza entrada), `ai/payload_sections/flow_summary.py`, `ai/payload_sections/quality_summary.py` (latency POOR→0.4, unknown→0.3), `analysis/institutional_analytics.py`, `signals/signal_processor.py` |
| **trading/** | `alert_engine.py` (distingue volatilidade vs squeeze) |
| **data_processing/** | `fix_optimization.py`, `data_handler.py` |
| **market_analysis/** | `historical_profiler.py` (value area de distribuicao real) |
| **config/** | `settings.py` (FLOW_IMBALANCE_MIN_TRADES, TTLs macro, ENABLE_ALPHAVANTAGE) |
| **raiz** | `check_integrity.py` (NOVO, commit `e8858c5`), `AUDIT_REPORT.md` (NOVO), `ai_analyzer_qwen.py`→`market_orchestrator/ai/analyzer_qwen.py`, `build_compact_payload.py`→`market_orchestrator/ai/payload_builder_compact.py`, `src/`→`src_old/` (proxy layer eliminada) |
| **scripts/diagnostics/** | `data_health_check.py`, `measure_out_of_order_trades.py`, `audit_cvd_numeric_test.py`, `audit_cvd_reset_divergence_test.py`, `audit_flow_imbalance_prod_test.py`, `audit_absorption_duplo_disparo_test.py`, `audit_absorption_prod_test.py`, `audit_support_resistance_test.py`, `audit_va_poc_outward.py`, `audit_ofi_numeric_test.py`, `run_production_observation.py`, `validate_production_run.py`, `measure_dedup_effect.py`, `audit_market_data.py` |
| **tests/unit/** | `test_signal_direction_absorption.py`, `test_flow_analyzer_metrics.py`, `test_quality_summary_latency.py`, `test_rolling_aggregate_eviction.py`, `test_buy_sell_ratio_flow_trend.py`, `test_flow_consistency_regression.py`, `test_latency_reliability_regression.py`, `test_quality_liquidity_freshness_regression.py`, `test_volume_profile_etapa4_regression.py`, `test_volume_profile_etapa4b_fail_closed.py`, `test_sr_etapa5_audit.py`, `test_sr_etapa5b_contract.py`, `test_etapa6_forense_contract.py`, `test_audit_market_data.py`, `test_volatility_alert_type.py` |
| **docs/audit/** | `SUMARIO_EXECUTIVO_AUDITORIA_2026-08.md`, `AUDITORIA_PIVOT_POINTS_2026-08-09.md`, `RELATORIO_OBSERVACAO_2026-08-09.md`, `RELATORIO_FLOW_INVARIANTS_2026-08-10.md`, `COMMIT_FLOW_INVARIANTS_2026-08-10.md`, `EVENT_BUS_METRICS_FLAKE_2026-08-10.md`, `ETAPA_5B_HLC_VP_SEMANTICA_2026-08-11.md`, `ETAPA_6_MACRO_NAN_FRED_CACHE_2026-08-11.md` (+ atualizacoes em FASE3/FASE5) |
| **outros** | `dados/eventos-fluxo.json.legacy` (removido 2026-08-09), `logs/issues.log.legacy-*`, `.gitignore` (fred_cache, ooo_report, observation_*) |

**Scripts de diagnostico criados apos 2026-08-11 (em working tree, SEM commit):**
`audit_live_data_invariants.py`, `capture_compact_sr.py`, `map_pbc_keys.py`, `replay_etapa6_j1j4.py`, `replay_j4_scorer.py`, `run_shadow_observation.py` (todos em `scripts/diagnostics/`) e `tests/unit/test_funding_rate_fallback.py`.

**Arquivos modificados pendentes (working tree):** `docs/audit/ETAPA_6_MACRO_NAN_FRED_CACHE_2026-08-11.md`

*Ultima atualizacao: 2026-08-16 (novos arquivos: common/json_safe.py, check_integrity.py, AUDIT_REPORT.md, 14 scripts de diagnostico, 15 testes unitarios, 8 docs de auditoria; HEAD `72951ca`)*

---

## Atualizacoes Posteriores (2026-08-16 ate 2026-09-15) — 85 commits

> **Data base anterior:** 2026-08-16 15:06:59 -0300 (commit `38916cc`)
> **HEAD atual:** `4c8335c` (2026-09-14 `fix(ai): gate unsupported microstructure claims`)
> **Periodo:** 2026-08-22 a 2026-09-14 — 85 commits (fail-closed/no-fabrication, migracao Futures, Golden, O1/R1, forense opt-in)

### NOVO PACOTE `audit_live/` - Captura Forense Opt-in (2026-09-11)
```
audit_live/
├── __init__.py          # Pacote instrumentacao forense observacional (FORENSIC_CAPTURE=1, fire-and-forget)
├── forensic_context.py  # Contexto forense LIVE
├── forensic_payload.py  # Payload forense (commit 0408ed9)
├── forensic_writer.py   # Writer forense + manifest.json (audit_writer_errors)
└── hooks.py             # Hooks nao-intrusivos (nunca alteram logica de negocio)
```

### NOVO PACOTE `tests/golden/` - Harness Deterministico G1-G4 (2026-09-08)
```
tests/golden/
├── __init__.py
├── conftest.py
├── fixtures/
│   ├── gw1_balanced.json
│   ├── gw2_buy_pressure.json
│   ├── gw3_sell_absorption.json
│   ├── gw4_external_unavailable.json
│   ├── gw5_high_activity.json       # GW5 slow (1500 trades)
│   └── gw6_weekend_holiday.json
├── test_gw_math.py         # GW1-4, GW6 (math/enrichment)
├── test_gw_enrichment.py
├── test_gw_ml.py           # TS-1 chaves eligible/unversioned/mismatch
├── test_gw_payload.py      # shape real end-to-end
├── test_gw_system.py
└── test_gw5_slow.py        # G4 slow + budget generoso
```

### NOVO `scripts/analytics/` - Avaliacao de Features/Posicionamento
```
scripts/analytics/
├── feature_value_validator.py       # Validador valor de feature (V1)
├── generate_o1_daily_snapshot.py    # Snapshot diario O1
├── hardened_feature_evaluator.py    # Avaliador hardenado (V1.1/V1.2)
├── positioning_evaluator.py         # Avaliador positioning P1.1
└── positioning_shadow_collector.py  # Coletor shadow positioning
```

### NOVO `analysis/results/` - Resultados de Auditoria
```
analysis/results/
├── o1_daily_2026-09-03.json
├── v1_1_hardening_results.json
├── v1_2_hardening_results.json
└── v1_feature_value_validation.json
```

### `config/` - Novos arquivos
```
config/
├── env_policy.py               # PF-D: bootstrap dotenv centralizado + observation guard (a31bdb9)
├── o1_cohort_manifest.json     # Manifesto coorte O1 (O1_START=2026-09-03T01:35:00Z)
├── r1_cohort_manifest.json     # Manifesto coorte R1 (28d replay, BTCUSDT)
└── volume_baseline_fut.json    # Baseline volume Futures (calib thresh whale 2.0, spike p95/h)
```

### `institutional/` - Novos modulos P1.2/P1.3
| Arquivo | Descricao |
|---------|-----------|
| `session_vwap.py` | VWAP por sessao UTC + historico corrente (bc9ae9f, P1.2) |
| `market_structure.py` | Estrutura de mercado + stress forense (P1.3/P1.3B/C) |

### `fetchers/` - Novos modulos
| Arquivo | Descricao |
|---------|-----------|
| `binance_positioning_fetcher.py` | Fetcher positioning Binance (P1.1, long/short accounts) |
| `onchain_updater.py` | Onchain fora do hot path (snapshot + DI + freshness, FASE B/C/D) |
| `macro_cache_validator.py` | Validador cache macro/FRED |

### `market_analysis/` - Novo modulo
| Arquivo | Descricao |
|---------|-----------|
| `cross_asset_updater.py` | CrossAsset fora do hot path (snapshot + DI, E3-B; shared-session returns F5-C) |

### `common/` - Novos modulos
| Arquivo | Descricao |
|---------|-----------|
| `signal_direction.py` | Canonico infer_signal_side/classify_outcome/get_directional_confidence (LONG/SHORT/NEUTRAL/UNKNOWN) |
| `ai_payload_integrator.py` | Integrador payload IA |
| `async_bridge.py` | Bridge async |
| `twap_validator.py` | Validador TWAP |

### `market_orchestrator/` - Novos modulos
| Arquivo | Descricao |
|---------|-----------|
| `capabilities.py` | Capability Contract (CONTINUOUS_TRADES_WS, POINT_IN_TIME_L2_SNAPSHOT; CONTINUOUS_L2/SPOOFING/ICEBERG=False) |

### `tools/` - Novos utilitarios
| Arquivo | Descricao |
|---------|-----------|
| `audit_manifest.py` | Auditoria de manifestos O1/R1 |
| `audit_shadow.py` | Auditoria shadow observation |

### `dados/audit/` + `dados/` - Datasets de calibracao R1-R4b
| Arquivo | Descricao |
|---------|-----------|
| `dados/audit/compact_J21.json` | Compact J21 |
| `dados/audit/klines_accept_futures.json` | Aceite migracao Futures |
| `dados/audit/klines_fut_s1.json` / `klines_spot_s1.json` | Crosscheck spot vs futures |
| `dados/audit/r4b_tabela_75_janelas.csv` | Tabela 75 janelas R4b |
| `dados/audit/windows_flat.csv` | Janelas flat |
| `dados/o1_cohort_manifest.json` / `dados/r1_cohort_manifest.json` | Manifestos replicados |

### `docs/audit/` - +23 documentos (2026-09)
AUDITORIA_JANELAS_EXTRAIDAS_2026-09-03, R2/R3/R4/R4b_2026-09-03, BINANCE_POSITIONING_DESIGN_2026, INSTITUTIONAL_CAPABILITIES_AUDIT_2026, INSTITUTIONAL_DATA_CONTRACTS_2026, O1_PREFLIGHT_VERIFICATION_REPORT_2026, P0_PIPELINE_INTEGRITY_EXECUTION_2026, P1_1_BINANCE_POSITIONING_EXECUTION_2026, P1_1B_POSITIONING_SHADOW_VALIDATION_2026, P1_2_SESSION_VWAP_EXECUTION_2026, P1_3_MARKET_STRUCTURE_EXECUTION_2026, P1_3B_MARKET_STRUCTURE_FORENSIC_AUDIT_2026, P1_3C_MARKET_STRUCTURE_SEMANTIC_FIX_2026, POST_AUDIT_VALIDATION_2026, TEST_DEBT, V1_1_METHODOLOGY_HARDENING_2026, V1_2_VALIDATOR_FORENSICS_2026, V1_FEATURE_VALUE_VALIDATION_2026 + `docs/freeze_consolidation_2026-09-01.md`

### `scripts/diagnostics/` - +27 scripts commitados (pos 16/08)
accept_futures_migration, audit_capabilities_deep, audit_data_attrition, audit_deep_modules, audit_live_data_invariants (era working-tree, agora commitado), audit_shadow_metrics, audit_sr_recurrence, audit_windows_offline, backup_shadow_dataset, calc_r4b_stats, capture_compact_sr (era working-tree), check_shadow_collection_health, collect_2h.ps1, crosscheck_klines_session, forensic_market_structure_stress, inspect_database_data, map_pbc_keys (era working-tree), measure_critical_path_positioning, measure_market_structure_performance, measure_session_vwap_performance, recheck_r4b, replay_etapa6_j1j4 + replay_j4_scorer (eram working-tree), run_o1_shadow_observation, run_shadow_observation (era working-tree), verify_safe_mode + outputs AUDITORIA_ACEITACAO_FINAL_20260901, AUDITORIA_FORENSE_2026-08-23, AUDITORIA_POS_FIX_1H_20260831

### Migracoes / Fixes estruturais principais (pos 16/08)
| Data | Mudanca |
|------|---------|
| 2026-08-22 | OutcomeTracker import restaurado; rolling aggregates preservam integridade temporal; raw history capacity; flow temporal quality no payload; labels em boundaries exatos |
| 2026-08-30/31 | Win rates direction-aware; health recovery vs falha; funding canonico (x100); payload compact sem prosa redundante |
| 2026-09-01/02 | Telemetria latencia por janela; freeze O1 baseline shadow; preflight O1; O1 ABANDONED_INSUFFICIENT_COVERAGE -> pivot R1 |
| 2026-09-05 | **cc249d7: trades/klines unificados em Binance USD-M Futures (era spot)**; calib thresholds futures + snapshot sincrono book + ml_stale; relatorios R1-R4b |
| 2026-09-07/08 | SEC-1/3/4 (AlphaVantage fail-closed, backup OCI exclui db/jsonl/env, eval->literal_eval); FASE A-D onchain; E3-A/B beep/cross-asset fora hot path; B-P0-1..4 + P1-A/B/C (ausente permanece ausente, UNKNOWN nunca 0.0); F5-C shared-session returns; Golden G1-G4; PF-S1/S2/M2/M3/D (shutdown cooperativo, dotenv centralizado) |
| 2026-09-11/12/14 | Forense opt-in + replay offline; VWAP UTC corrente + pending-vs-gap; onchain coverage vs freshness; flow vs depth desambiguados; market-impact fail-closed sem profundidade; S/R walls snapshot vs estruturais; capabilities gate microstructure |

### Testes novos (pos 16/08) — ~70 arquivos
- `tests/golden/`: 6 (ver acima)
- `tests/unit/`: ~55 — absorption_evidence, aggressive_pct_contract, alphavantage_fail_closed, backup_excludes, binance_positioning_p1_1, crossasset_* (5), dataset_provenance, dominance_change_missing, enrichment_context_no_fabrication, env_policy, event_memory_import_regression, event_similarity_direction_aware, feature_evaluator_diagnostics, flow_trades_capacity_diagnostic, flowdata_optional_widening, funding_rate_fallback (era working-tree), health_recovery, heatmap_golden/perf/scope, macro_session_lifecycle, market_orchestrator_direction_confidence, market_structure_p1_3, microstructure_claims_neutralization, ml_eligibility, ml_imbalance_missing, ml_stale_neutralization, mtf_nonfinite_none, no_eval, onchain_coverage/freshness/latency/window_contract, orderbook_market_impact_contract + insufficient_liquidity, orderbook_sync_snapshot, outcome_tracker_boundary + direction_aware, p01_imbalance_per_window, p04_onchain_provenance, p05_quality_usability, p06_retention_snapshot, ratio_nonfinite_contract, regime_corr_missing, session_vwap_coverage + p1_2, signal_direction, signal_orderbook_schema, sound_alert_nonblocking, stream_parser_aggtrade, volume_spike_dual_gate, windowstate_real_shape
- `tests/payload/`: flow_window_quality_propagation, funding_rate_pipeline_p0, p00_compact_preserved, p02_semantic_contract, positioning_provenance_p1_1b
- `tests/integration/`: test_ml_stale_real_event_pipeline

### Pendentes em working tree (SEM commit em 2026-09-15)
`docs/audit/FASE1_DEPRECATION_MECHANISM_FAILURE_2026-09.md`, `ORDERBOOK_LAG_EMPIRICAL_RECONCILIATION_2026-09.md`, `SIGNOFF_AUDITORIA_FUTURES_2026-09.md`, `query`, ~30 scripts `scripts/diagnostics/` (analyze_orderbook_sync_session, audit_c1_contamination, audit_latency_spikes, audit_val2, benchmark_dump_raw_trades_overhead, check_r4_citations, check_validations_data, decompose_london_ny_trades, detailed_volume_spike_audit, find_recent_files, generate_r4b_markdown, inspect_121201_log, inspect_all_dbs, inspect_latency_context, inspect_run_log, inspect_schema_real, inspect_snapshot_moment_trades, inspect_window97, investigate_asia_event, multi_day_stratified_audit, run_3_validation_scripts, run_all_validations, run_r4b_full, stratified_trade_sampling, test_store_init, val0_integrity, val2_check, validate_memory_and_latency_health, validate_ml_stale_neutralization, validate_signals_orderbook_schema, validate_volume_spike_dual_gate, validate_whale_threshold), `tests/unit/test_dump_raw_trades.py` + 6 modificados (klines_accept_futures.json, TEST_DEBT.md, orderbook_wrapper.py, collect_2h.ps1, test_ml_stale_real_event_pipeline.py, test_orderbook_sync_snapshot.py)

*Ultima atualizacao: 2026-09-15 (HEAD `4c8335c`; 85 commits desde 2026-08-16; novos pacotes: audit_live/, tests/golden/, scripts/analytics/, analysis/results/; novos modulos: capabilities, signal_direction, env_policy, session_vwap, market_structure, binance_positioning_fetcher, onchain_updater, cross_asset_updater; migracao Futures cc249d7)*
