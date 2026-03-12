# 📁 Estrutura Completa do Sistema - Robo Binance API

## Visão Geral do Projeto

Este é um sistema de trading automatizado para Binance com análise de fluxo de ordens, suporte/resistência, detecção de regime de mercado e integração com IA.

---

## 📂 Raiz do Projeto (Root)

### Arquivos de Configuração
| Arquivo | Descrição |
|---------|-----------|
| `.gitignore` | Configurações de gitignore |
| `.coveragerc` | Configuração de coverage de testes |
| `.dockerignore` | Configuração Docker ignore |
| `mypy.ini` | Configuração de type checking |
| `pyproject.toml` | Configuração do projeto Python |
| `pyrightconfig.json` | Configuração do pyright |
| `pytest.ini` | Configuração do pytest |
| `docker-compose.yml` | Orquestração de containers |
| `Dockerfile` | Imagem Docker do projeto |
| `requirements.txt` | Dependências Python |
| `requirements-dev.txt` | Dependências de desenvolvimento |

### Arquivos Principais
| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Ponto de entrada principal |
| `main.patched.py` | Versão com patches aplicados |
| `config.py` | Configurações globais |
| `config.json` | Arquivo de configuração JSON |
| `fix_bot_run.py` | Script de correção do bot |
| `test_connection.py` | Teste de conexão |
| `validation_check.py` | Validação de dados |

---

## 📂 Módulos Principais

### 🤖 [`ai_runner/`](ai_runner/)
Módulo de execução de IA para análise de mercado

```
ai_runner/
├── __init__.py
├── ai_runner.py         # Executor principal de IA
└── exceptions.py        # Exceções específicas
```

---

### 📊 [`flow_analyzer/`](flow_analyzer/)
Sistema de análise de fluxo de ordens (Order Flow)

```
flow_analyzer/
├── __init__.py
├── absorption.py        # Detecção de absorção
├── aggregates.py        # Agregação de dados
├── constants.py         # Constantes do módulo
├── core.py              # Motor principal
├── errors.py            # Tratamento de erros
├── logging_config.py    # Configuração de logging
├── metrics.py           # Métricas do módulo
├── profiling.py         # Ferramentas de profiling
├── prometheus_metrics.py# Integração Prometheus
├── protocols.py         # Definições de protocolos
├── serialization.py     # Serialização de dados
├── utils.py             # Utilitários
├── validation.py        # Validação de dados
└── whale_score.py       # Score de whales
```

---

### 🏛️ [`market_orchestrator/`](market_orchestrator/)
Orquestrador principal do mercado

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
│   ├── llm_payload_guardrail.py   # Guardrails
│   ├── llm_response_validator.py  # Validador de respostas IA
│   ├── payload_compressor.py      # Compressor v1
│   ├── payload_compressor_v3.py    # Compressor v3 (39KB)
│   ├── payload_metrics_aggregator.py
│   ├── payload_section_cache.py    # Cache de seções
│   └── raw_event_deduplicator.py   # Deduplicador
├── analysis/
│   ├── __init__.py
│   └── institutional_analytics.py  # Análise institucional
├── connection/
│   └── robust_connection.py  # Conexão robusta
├── flow/
│   ├── __init__.py
│   ├── risk_manager.py       # Gerenciamento de risco
│   ├── signal_processor.py  # Processador de sinais
│   ├── trade_executor.py     # Execução de trades
│   └── trade_flow_analyzer.py
├── orderbook/
│   ├── __init__.py
│   └── orderbook_wrapper.py  # Wrapper do orderbook
├── signals/
│   ├── __init__.py
│   └── signal_processor.py   # Processador de sinais
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   └── price_fetcher.py
└── windows/
    ├── __init__.py
    └── window_processor.py   # Processador de janelas
```

---

### 📈 [`support_resistance/`](support_resistance/)
Sistema de Suporte e Resistência

```
support_resistance/
├── __init__.py
├── config.py              # Configurações
├── constants.py           # Constantes
├── core.py                # Motor principal
├── defense_zones.py       # Zonas de defesa
├── monitor.py             # Monitor em tempo real
├── pivot_points.py        # Pontos de pivô
├── reference_prices.py    # Preços de referência
├── sr_strength.py         # Força de S/R
├── system.py              # Sistema completo
├── utils.py               # Utilitários
├── validation.py         # Validação
└── volume_profile.py      # Perfil de volume
```

---

### 🧠 [`ml/`](ml/)
Machine Learning e Inferência

```
ml/
├── feature_calculator.py   # Calculador de features
├── generate_dataset.py     # Geração de datasets
├── hybrid_decision.py      # Decisão híbrida
├── inference_engine.py     # Motor de inferência
├── model_inference.py      # Inferência de modelo
├── train_model.py          # Treinamento de modelo
├── datasets/
│   └── training_dataset.parquet  # Dataset de treinamento
└── models/
    ├── error_log_*.txt           # Logs de erros
    ├── feature_importance_*.csv  # Importância de features
    ├── model_metadata*.json      # Metadados dos modelos
    ├── xgb_model_*.json          # Modelos XGBoost
    └── model_metadata_latest.json
```

---

### 🔄 [`data_pipeline/`](data_pipeline/)
Pipeline de processamento de dados

```
data_pipeline/
├── __init__.py
├── config.py
├── logging_utils.py
├── pipeline.py              # Pipeline principal
├── cache/
│   ├── __init__.py
│   ├── buffer.py            # Buffer de cache
│   └── lru_cache.py         # Cache LRU
├── fallback/
│   ├── __init__.py
│   └── registry.py          # Registro de fallbacks (NOVO 03/2026)
├── metrics/
│   ├── __init__.py
│   ├── data_quality_metrics.py  # Métricas de qualidade (NOVO 03/2026)
│   └── processor.py         # Processador de métricas (NOVO 03/2026)
└── validation/
    ├── __init__.py
    ├── adaptive.py
    └── validator.py
```

---

### 📦 [`src/`](src/)
Código fonte principal

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
│   └── macro_data_provider.py  # Provider de dados macro
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

### 📚 [`tests/`](tests/)
Suíte de testes

```
tests/
├── __init__.py
├── backtester.py
├── config_test.py
├── conftest.py
├── fixtures.py
├── fixtures/
│   └── sample_analysis_trigger.json
├── mock_ai_responses.py
├── mock_qwen.py
├── regime_scenario_tester.py
├── test_absorption_zone_mapper.py
├── test_ai_analyzer_language_and_think_strip.py
├── test_ai_analyzer_mock.py
├── test_ai_llm_fallback_flow.py
├── test_ai_response_validator.py
├── test_ai_runner.py
├── test_ai_runner_comprehensive.py
├── test_circuit_breaker.py
├── test_corrections.py
├── test_data_pipeline.py
├── test_data_quality_validator.py
├── test_data_validator.py
├── test_defense_zones.py
├── test_enrich_signal.py
├── test_event_bus.py
├── test_event_saver_jsonl_guardian.py
├── test_fix_optimization_storage.py
├── test_flow_analyzer.py
├── test_institutional_alerts.py
├── test_integration_full_flow.py
├── test_invariant_fix.py
├── test_macro_data_provider.py
├── test_market_orchestrator_comprehensive.py
├── test_ml_frozen_detector.py
├── test_orchestrator_initialization.py
├── test_orderbook_analyze_core.py
├── test_orderbook_analyzer.py
├── test_orderbook_analyzer_coverage.py
├── test_orderbook_analyzer_full_coverage.py
├── test_orderbook_analyzer_missing.py
├── test_orderbook_config_injection.py
├── test_orderbook_core_comprehensive.py
├── test_orderbook_helpers.py
├── test_orderbook_validate_snapshot.py
├── test_orderbook_wrapper_fallback.py
├── test_orderbook_wrapper_fetch_with_retry.py
├── test_out_of_order_pruning.py
├── test_passive_aggressive_flow.py
├── test_patch_2_fallback_controlado.py
├── test_patch_2_simples.py
├── test_performance_benchmarks.py
├── test_rate_limiter.py
├── test_regime_integration.py
├── test_risk_manager_comprehensive.py
├── test_rolling_aggregate.py
├── test_run_diagnosis.py
├── test_sr_strength.py
├── test_support_resistance_consolidated.py
├── test_support_resistance_modular.py
├── test_system_health.py
├── test_trade_flow_analyzer.py
├── test_update_histories.py
├── test_window_processor.py
├── test_window_processor_queue.py
├── verify_day4_implementations.py
├── verify_patch_2.py
├── verify_prune_logic_only.py
├── fix_broken_tests.py
├── fix_qwen_import.py
└── payload/
    ├── conftest.py
    ├── pytest.ini
    ├── test_payload_compressor.py
    ├── test_payload_guardrail.py
    ├── test_payload_metrics_aggregator.py
    ├── test_payload_optimizer.py
    └── test_payload_tripwires.py
```

---

### 🎯 [`orderbook_core/`](orderbook_core/)
Núcleo do analisador de orderbook

```
orderbook_core/
├── __init__.py
├── circuit_breaker.py    # Circuit breaker
├── constants.py
├── event_factory.py      # Fábrica de eventos
├── exceptions.py
├── metrics.py
├── orderbook_config.py
├── orderbook.py          # Orderbook principal
├── protocols.py
├── structured_logging.py
└── tracing_utils.py
```

---

### 📉 [`orderbook_analyzer/`](orderbook_analyzer/)
Analisador de orderbook

```
orderbook_analyzer/
├── __init__.py
├── analyzer.py
├── spread_tracker.py     # Rastreador de spread
└── config/
    ├── __init__.py
    └── settings.py
```

---

### ⚠️ [`risk_management/`](risk_management/)
Gerenciamento de risco

```
risk_management/
├── __init__.py
├── exceptions.py
└── risk_manager.py
```

---

### ⚙️ [`config/`](config/)
Configurações do projeto

```
config/
├── __init__.py
└── model_config.yaml
```

---

### 🔧 [`auto_fixer/`](auto_fixer/)
Sistema automático de correção de código

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
│   ├── __init__.py
│   └── codebase_scanner.py
├── phase2_extractor/
│   ├── __init__.py
│   └── ast_extractor.py
├── phase3_chunker/
│   ├── __init__.py
│   └── chunk_engine.py
├── phase4_index/
│   ├── __init__.py
│   └── code_index.py
├── phase5_rag/
│   ├── __init__.py
│   ├── context_retriever.py
│   ├── embeddings.py
│   └── vector_store.py
├── phase6_analyzers/
│   ├── __init__.py
│   ├── api_analyzer.py
│   ├── async_analyzer.py
│   ├── base_analyzer.py
│   ├── import_analyzer.py
│   └── websocket_analyzer.py
├── phase7_patcher/
│   ├── __init__.py
│   ├── patch_applier.py
│   ├── patch_generator.py
│   └── patch_validator.py
└── phase8_reporter/
    ├── __init__.py
    └── report_generator.py
```

---

## 📂 Diretórios de Suporte

### 📁 [`scripts/`](scripts/)
Scripts de utilidade

```
scripts/
├── ab_test_prompt_styles.py
├── analyze_ai_usage.py
├── audit_json_payload_costs.py
├── backup_to_oci.py
├── disaster_recovery.sh
├── remote_health_check.sh
├── test_fixes.py
├── test_fixes_simple.py
├── test_fixes_final.py
├── test_payload.sh
└── validate_regime_system.py
```

---

### 🔧 [`arquivos para diagnostico/`](arquivos para diagnostico/)
Arquivos para diagnóstico de janelas

```
arquivos para diagnostico/
├── __init__.py
└── diagnostico de janelas geradas/
    ├── __init__.py
    ├── diagnostico_avancado.py     # Diagnóstico avançado (NOVO 03/2026)
    ├── diagnostico_duplicatas.py   # Diagnóstico de duplicatas (NOVO 03/2026)
    ├── diagnostico_janelas.py       # Diagnóstico de janelas (NOVO 03/2026)
    └── fix_duplicatas_completo.py  # Correção de duplicatas (NOVO 03/2026)
```

---

### 🔧 [`tools/`](tools/)
Ferramentas de diagnóstico

```
tools/
├── export_db_to_jsonl.py
├── inspect_db.py
├── inspect_events_schema.py
├── test_groq_models_http.py
├── test_groq_models_v2.py
├── test_groq_official.py
└── ws_test.py
```

---

### 🔍 [`diagnostics/`](diagnostics/)
Ferramentas de diagnóstico

```
diagnostics/
├── analyze_ai_results.py
├── auto_fix.py
├── evaluate_ai_performance.py
├── final_validation.py
├── performance_metrics.py
├── replay_validator.py
├── test_decision_system.py
├── test_integrated.py
├── test_latency.py
├── test_ml_model.py
└── verify_ml_integration.py
```

---

### 🗄️ [`database/`](database/)
Sistema de banco de dados

```
database/
├── __init__.py
└── event_store.py
```

---

### 🏗️ [`infrastructure/`](infrastructure/)
Infraestrutura

```
infrastructure/
├── __init__.py
├── market-bot.service
└── oci/
    ├── __init__.py
    ├── monitoring.py
    ├── security_config.md
    └── vault_helper.py
└── terraform/
    └── main.tf
```

---

### 📄 [`docs/`](docs/)
Documentação

```
docs/
├── architecture.md
├── RUNBOOK.md
└── troubleshooting.md
```

---

### 📜 [`legacy/`](legacy/)
Código legado

```
legacy/
├── data_pipeline_legacy..py
├── market_analyzer_2_3_0.py
└── support_resistance_legacy.py
```

---

### 🗂️ [`Regras/`](Regras/)
Regras e documentação

```
Regras/
├── COMPRIMIR DADOS.API.odt
├── Correção automática.docx
├── metodos institucional.docx     # (NOVO 03/2026)
├── regras para o codigo.odt
├── Teia de monitoramento Mini Dolar (B3).odt
└── Rastreando robos/
    ├── ESTRUTURANDO ARQUIVO JSON.odt
    └── ROBOS X INTEGIGENCIA IA.odt
```

---

### 🤖 [`ai_runner/`](ai_runner/) (alternative location)

```
ai_runner/
├── __init__.py
├── ai_runner.py
└── exceptions.py
```

---

### 🧠 [`utils/`](utils/)
Utilitários adicionais

```
utils/
├── __init__.py
├── async_helpers.py
├── heartbeat_manager.py
├── trade_filter.py
└── trade_timestamp_validator.py
```

---

### 🧮 [`memory/`](memory/)
Sistema de memória

```
memory/
├── __init__.py
└── levels_BTCUSDT.json  # Níveis de preço BTCUSDT (NOVO 03/2026)
```

---

### 📈 [`MQL5/`](MQL5/)
Integração MQL5 (MetaTrader)

```
MQL5/
├── __init__.py
└── Indicators/
    └── ChartSignalsFromCSV.mq5
```

---

### 🔄 [`fallback_events/`](fallback_events/)
Eventos de fallback

```
fallback_events/
└── eventos_20260307.json
```

---

### 💾 [`backups/`](backups/)
Backups de segurança

```
backups/
└── time_manager.py.20260308_144713.bak
```

---

## 📂 Arquivos de Dados

### 📁 [`dados/`](dados/)
```
dados/
├── eventos_fluxo.jsonl     # Eventos de fluxo
├── eventos-fluxo.json     # Eventos de fluxo (JSON)
├── eventos_visuais.log    # Eventos visuais
└── trading_bot.db         # Banco de dados SQLite
```

---

### 📁 [`logs/`](logs/)
Diretório de logs

```
logs/
├── last_llm_payload.json
├── payload_metrics.jsonl
├── payload_metrics.jsonl.zip
└── payload_section_cache.json
```

---

### 📁 [`features/`](features/)
Dados de features por data (date=YYYY-MM-DD/)

```
features/
├── date=2025-12-08/
├── date=2025-12-09/
├── date=2025-12-11/
├── date=2025-12-17/
├── date=2025-12-18/
├── date=2025-12-20/
├── date=2025-12-21/
├── date=2026-01-01/
├── date=2026-01-02/
├── date=2026-01-03/
├── date=2026-01-04/
├── date=2026-01-05/
├── date=2026-01-06/
├── date=2026-01-13/
├── date=2026-01-19/
├── date=2026-01-20/
├── date=2026-01-21/
├── date=2026-01-24/
├── date=2026-01-30/
├── date=2026-02-09/
├── date=2026-02-11/
├── date=2026-02-12/
├── date=2026-02-21/
├── date=2026-02-22/
├── date=2026-02-23/
├── date=2026-02-24/
├── date=2026-02-25/
├── date=2026-03-01/
├── date=2026-03-05/
├── date=2026-03-06/
├── date=2026-03-07/
├── date=2026-03-08/
├── date=2026-03-09/
├── date=2026-03-10/
├── date=2026-03-11/
└── date=2026-03-12/
```

---

## 📂 Arquivos Principais (Raiz)

### Análise de IA
| Arquivo | Descrição |
|---------|-----------|
| `ai_analyzer_qwen.py` | Analisador IA principal (150KB) |
| `ai_analyzer_qwen_patch2.py` | Patch v2 do analisador |
| `ai_analyzer_disabled.py` | Analisador desabilitado |
| `ai_historical_pro.py` | Histórico de IA |
| `ai_payload_compressor.py` | Compressor de payload |
| `ai_response_validator.py` | Validador de respostas IA |
| `context_collector.py` | Coletor de contexto |
| `optimize_ai_payload.py` | Otimizador de payload IA (NOVO 03/2026) |
| `payload_optimizer_config.py` | Configuração do otimizador (NOVO 03/2026) |
| `integration_validator.py` | Validador de integração (NOVO 03/2026) |

### Análise de Mercado
| Arquivo | Descrição |
|---------|-----------|
| `orderbook_analyzer.py` | Analisador de orderbook (123KB) |
| `cross_asset_correlations.py` | Correlações cross-asset |
| `pattern_recognition.py` | Reconhecimento de padrões |
| `liquidity_heatmap.py` | Mapa de calor de liquidez |
| `dynamic_volume_profile.py` | Perfil de volume dinâmico |
| `orderbook_fallback.py` | Fallback do orderbook |
| `orderbook_ws_manager.py` | Gerenciador WebSocket |
| `market_impact.py` | Análise de impacto de mercado (NOVO 03/2026) |
| `funding_aggregator.py` | Agregador de funding rates (NOVO 03/2026) |
| `levels_registry.py` | Registro de níveis de preço (NOVO 03/2026) |

### Dados e Validação
| Arquivo | Descrição |
|---------|-----------|
| `data_handler.py` | Manipulador de dados |
| `data_enricher.py` | Enriquecedor de dados |
| `data_validator.py` | Validador de dados |
| `data_quality_validator.py` | Validador de qualidade |
| `feature_store.py` | Store de features |
| `process_csv_data.py` | Processador de dados CSV (NOVO 03/2026) |
| `build_compact_payload.py` | Construtor de payload compactado (NOVO 03/2026) |

### Trading e Execução
| Arquivo | Descrição |
|---------|-----------|
| `trade_buffer.py` | Buffer de trades |
| `trade_validator.py` | Validador de trades |
| `alert_engine.py` | Motor de alertas |
| `alert_manager.py` | Gerenciador de alertas |
| `metrics_collector.py` | Coletor de métricas |
| `outcome_tracker.py` | Rastreador de resultados (NOVO 03/2026) |

### Integração Externa
| Arquivo | Descrição |
|---------|-----------|
| `macro_data_fetcher.py` | Coletor de dados macroeconômicos |
| `macro_fetcher.py` | Fetcher de macro |
| `fred_fetcher.py` | Coletor do FRED |
| `websocket_handler.py` | Manipulador WebSocket |
| `onchain_fetcher.py` | Coletor de dados on-chain (NOVO 03/2026) |

### Sistema
| Arquivo | Descrição |
|---------|-----------|
| `event_bus.py` | Barramento de eventos |
| `event_saver.py` | Salvador de eventos |
| `time_manager.py` | Gerenciador de tempo |
| `clock_sync.py` | Sincronização de relógio |
| `health_monitor.py` | Monitor de saúde |
| `event_memory.py` | Memória de eventos |
| `event_similarity.py` | Similaridade de eventos |
| `event_stats_model.py` | Modelo de estatísticas |

### Utilitários
| Arquivo | Descrição |
|---------|-----------|
| `format_utils.py` | Utilitários de formatação |
| `technical_indicators.py` | Indicadores técnicos |
| `ml_features.py` | Features de ML |
| `export_signals.py` | Exportador de sinais |
| `report_generator.py` | Gerador de relatórios |
| `historical_profiler.py` | Profiler histórico (NOVO 03/2026) |
| `log_formatter.py` | Formatador de logs (NOVO 03/2026) |
| `log_sanitizer.py` | Sanitizador de logs (NOVO 03/2026) |

### Institucional
| Arquivo | Descrição |
|---------|-----------|
| `institutional_enricher.py` | Enriquecedor institucional (85KB) |
| `enrichment_integrator.py` | Integrador de enriquecimento (NOVO 03/2026) |

### Debug e Desenvolvimento
| Arquivo | Descrição |
|---------|-----------|
| `debug_bot.py` | Debug do bot |
| `debug_env.py` | Debug de ambiente |
| `debug_keyerror.py` | Debug de KeyError |
| `debug_payload.py` | Debug de payload |
| `diagnose_crash.py` | Diagnóstico de crash |
| `fix_optimization.py` | Correção de otimização (NOVO 03/2026) |
| `diagnose_optimization.py` | Diagnóstico de otimização (NOVO 03/2026) |
| `final_replace.py` | Substituição final (NOVO 03/2026) |
| `verify_implementations.py` | Verificador de implementações (NOVO 03/2026) |
| `verify_patch.py` | Verificador de patches (NOVO 03/2026) |
| `verificar_otimizacao.py` | Verificação de otimização (NOVO 03/2026) |
| `validar_evento.py` | Validador de eventos (NOVO 03/2026) |
| `debug_validator.py` | Validador de debug (NOVO 03/2026) |
| `reproduce_issue.py` | Reproduzir problema (NOVO 03/2026) |
| `show_problem_lines.py` | Mostrar linhas problemáticas (NOVO 03/2026) |

### Dashboard e Visualização
| Arquivo | Descrição |
|---------|-----------|
| `dashboard.py` | Dashboard (43KB) |
| `app.py` | Aplicação principal (NOVO 03/2026) |

### Scripts de Estrutura
| Arquivo | Descrição |
|---------|-----------|
| `create_structure.py` | Criador de estrutura (NOVO 03/2026) |
| `generate_updated_structure.py` | Gerador de estrutura atualizada (NOVO 03/2026) |
| `compare_structure.py` | Comparador de estrutura (NOVO 03/2026) |
| `compare_structure_filtered.py` | Comparador filtrado (NOVO 03/2026) |
| `find_missing_files.py` | Localizador de arquivos faltantes (NOVO 03/2026) |
| `list_project_files.py` | Listador de arquivos (NOVO 03/2026) |

### Demonstrações e Testes
| Arquivo | Descrição |
|---------|-----------|
| `demo_circuit_breaker.py` | Demo de circuit breaker (NOVO 03/2026) |
| `demo_enhanced_cross_asset.py` | Demo cross-asset avançado (NOVO 03/2026) |
| `demo_enhanced_cross_asset_simple.py` | Demo cross-asset simples (NOVO 03/2026) |

### Dados de Mercado
| Arquivo | Descrição |
|---------|-----------|
| `dados_mercado.csv` | Dados de mercado (NOVO 03/2026) |
| `reg_test_report.json` | Relatório de testes de regressão (NOVO 03/2026) |
| `relatorio.json` | Relatório geral (NOVO 03/2026) |
| `modelo_dados_ideal.py` | Modelo de dados ideal (NOVO 03/2026) |

### Documentação
| Arquivo | Descrição |
|---------|-----------|
| `README_OPTIMIZATION.md` | Documentação de otimização (NOVO 03/2026) |
| `PATCH_SUMMARY.md` | Resumo de patches (NOVO 03/2026) |
| `orderbook_severity_analysis.md` | Análise de severidade (NOVO 03/2026) |
| `auditoria_estrutura_json.md` | Auditoria de estrutura JSON (NOVO 03/2026) |
| `RELATORIO_ENRICHMENT_CROSS_ASSET.md` | Relatório de enriquecimento (NOVO 03/2026) |
| `RELATORIO_FINAL_MACRO_PROVIDER.md` | Relatório macro provider (NOVO 03/2026) |
| `RESUMO_EXPORT_SINAIS.md` | Resumo de exportação de sinais (NOVO 03/2026) |
| `CORRECAO_ENRICH_EVENT_SUMMARY.md` | Correção de enriquecimento (NOVO 03/2026) |
| `CORRECAO_FETCH_INTERMARKET_DATA.md` | Correção de dados intermarket (NOVO 03/2026) |

### Scripts de Auditoria
| Arquivo | Descrição |
|---------|-----------|
| `audit_new_features.py` | Auditoria de novas features (NOVO 03/2026) |
| `audit_script.py` | Script de auditoria (NOVO 03/2026) |
| `full_audit.py` | Auditoria completa (NOVO 03/2026) |

---

## 📊 Estatísticas do Projeto

- **Total de arquivos Python**: ~250+
- **Total de módulos**: 25+
- **Linhas de código principais**: 100,000+
- **Testes**: 80+ arquivos de teste
- **Configurações**: YAML, JSON, INI, TOML
- **Dados de features**: 34 datas

---

## 🏗️ Arquitetura de Alto Nível

```
┌─────────────────────────────────────────────────────────────┐
│                     MAIN.PY                                  │
│                  (Ponto de Entrada)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
 ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
 │   MARKET    │  │    AI RUNNER     │  │   FLOW      │
 │ ORCHESTRATOR│  │   (Análise IA)   │  │  ANALYZER   │
 └─────────────┘  └──────────────────┘  └─────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
              ┌─────────────────────┐
              │   ORDERBOOK CORE     │
              │   (Order Book)       │
              └─────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
 ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
 │  SUPPORT   │  │   RISK MGMT      │  │   DATA      │
 │  RESISTANCE│  │  (Gerenciamento) │  │  PIPELINE   │
 └─────────────┘  └──────────────────┘  └─────────────┘
         │                │                │
         └────────────────┼────────────────┘
                          ▼
              ┌─────────────────────┐
              │   DATABASE/LOGS     │
              │   (Persistência)    │
              └─────────────────────┘
```

---

## 🔗 Dependencies Principais

- **Binance**: `binance-connector`, `python-binance`
- **IA/ML**: `openai`, `anthropic`, `transformers`, `torch`
- **Dados**: `pandas`, `numpy`, `polars`
- **Async**: `asyncio`, `aiohttp`, `websockets`
- **Database**: `sqlalchemy`, `sqlite3`, `orjson`
- **Monitoring**: `prometheus-client`, `structlog`
- **Testing**: `pytest`, `pytest-asyncio`, `coverage`

---

*Última atualização: 2026-03-12*
