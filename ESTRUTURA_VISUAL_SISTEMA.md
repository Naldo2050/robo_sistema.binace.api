# 🗂️ ESTRUTURA VISUAL DO SISTEMA - ROBO BINANCE

```
robo_sistema.binace.api/
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         📁 CONFIGURAÇÃO E SETUP                             │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── config/                                    # Configurações centralizadas
│   ├── __init__.py
│   └── model_config.yaml
│
├── config.json                                # Configuração principal
├── config.py                                  # Módulo de configuração
├── requirements.txt                           # Dependências Python
├── requirements-dev.txt                       # Dependências de desenvolvimento
├── pyproject.toml                             # Configuração do projeto
├── pytest.ini                                 # Configuração de testes
├── mypy.ini                                   # Configuração MyPy
├── .gitignore                                 # Ignorar arquivos Git
├── .dockerignore                              # Ignorar arquivos Docker
├── Dockerfile                                 # Imagem Docker
└── docker-compose.yml                         # Orquestração Docker
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🚀 PONTO DE ENTRADA                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── main.py                                    # Ponto de entrada principal
├── main.patched.py                            # Versão com patches
├── dashboard.py                               # Dashboard de visualização
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🤖 INTELIGÊNCIA ARTIFICIAL                           │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── ai_runner/                                 # Executor de IA
│   ├── __init__.py
│   ├── ai_runner.py
│   └── exceptions.py
│
├── ai_analyzer_qwen.py                        # Analisador Qwen
├── ai_analyzer_qwen_patch2.py                 # Analisador Qwen (patch 2)
├── ai_analyzer_disabled.py                    # Analisador desativado
├── ai_historical_pro.py                       # Analisador histórico PRO
├── ai_payload_compressor.py                   # Compressor de payload IA
├── optimize_ai_payload.py                     # Otimizador de payload
├── payload_optimizer_config.py                # Configuração do otimizador
│
├─────────────────────────────────────────────────────────────────────────────┐
│                    🎯 ORQUESTRADOR DE MERCADO                               │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── market_orchestrator/                       # Orquestrador principal
│   │
│   ├── ai/                                    # Componentes de IA
│   │   ├── ai_runner.py
│   │   ├── ai_payload_builder.py
│   │   ├── ai_enrichment_context.py
│   │   ├── payload_compressor.py
│   │   ├── llm_payload_guardrail.py
│   │   ├── payload_metrics_aggregator.py
│   │   └── payload_section_cache.py
│   │
│   ├── analysis/                              # Análises institucionais
│   │   └── institutional_analytics.py
│   │
│   ├── connection/                            # Conexões robustas
│   │   └── robust_connection.py
│   │
│   ├── flow/                                  # Fluxo de trades
│   │   ├── trade_flow_analyzer.py
│   │   ├── signal_processor.py
│   │   ├── trade_executor.py
│   │   └── risk_manager.py
│   │
│   ├── orderbook/                             # Wrapper do orderbook
│   │   └── orderbook_wrapper.py
│   │
│   ├── signals/                               # Processamento de sinais
│   │   └── signal_processor.py
│   │
│   ├── utils/                                 # Utilitários
│   │   ├── logging_utils.py
│   │   └── price_fetcher.py
│   │
│   ├── windows/                               # Processamento de janelas
│   │   └── window_processor.py
│   │
│   ├── __init__.py
│   ├── market_orchestrator.py
│   └── orchestrator.py
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         📊 ORDERBOOK                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── orderbook_core/                            # Núcleo do orderbook
│   ├── __init__.py
│   ├── orderbook.py
│   ├── orderbook_config.py
│   ├── circuit_breaker.py
│   ├── event_factory.py
│   ├── metrics.py
│   ├── protocols.py
│   ├── structured_logging.py
│   ├── tracing_utils.py
│   ├── constants.py
│   └── exceptions.py
│
├── orderbook_analyzer/                        # Analisador de orderbook
│   ├── __init__.py
│   ├── analyzer.py
│   ├── spread_tracker.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
│
├── orderbook_analyzer.py                      # Analisador principal (legado)
├── orderbook_fallback.py                      # Fallback do orderbook
├── orderbook_ws_manager.py                    # Gerenciador WebSocket
│
├─────────────────────────────────────────────────────────────────────────────┐
│                    📈 SUPORTE E RESISTÊNCIA                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── support_resistance/                        # Sistema S/R completo
│   ├── __init__.py
│   ├── core.py                                # Núcleo S/R
│   ├── system.py                              # Sistema integrado
│   ├── config.py                              # Configurações
│   ├── constants.py                           # Constantes
│   ├── defense_zones.py                       # Zonas de defesa
│   ├── monitor.py                             # Monitoramento
│   ├── pivot_points.py                        # Pontos de pivô
│   ├── reference_prices.py                    # Preços de referência
│   ├── sr_strength.py                         # Força S/R
│   ├── utils.py                               # Utilitários
│   ├── validation.py                          # Validação
│   └── volume_profile.py                      # Perfil de volume
│
├── levels_registry.py                         # Registro de níveis
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🔥 ANÁLISE DE FLUXO                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── flow_analyzer/                             # Analisador de fluxo
│   ├── __init__.py
│   ├── core.py                                # Núcleo
│   ├── absorption.py                          # Absorção
│   ├── aggregates.py                          # Agregados
│   ├── whale_score.py                         # Score de baleias
│   ├── metrics.py                             # Métricas
│   ├── protocols.py                           # Protocolos
│   ├── validation.py                          # Validação
│   ├── serialization.py                       # Serialização
│   ├── utils.py                               # Utilitários
│   ├── profiling.py                           # Profiling
│   ├── logging_config.py                      # Configuração de logs
│   ├── prometheus_metrics.py                  # Métricas Prometheus
│   ├── errors.py                              # Erros
│   └── constants.py                           # Constantes
│
├── flow_analyzer.py                           # Entry point legado
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🔄 PIPELINE DE DADOS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── data_pipeline/                             # Pipeline de dados
│   │
│   ├── cache/                                 # Cache
│   │   ├── __init__.py
│   │   ├── buffer.py
│   │   └── lru_cache.py
│   │
│   ├── fallback/                              # Fallback
│   │   ├── __init__.py
│   │   └── registry.py
│   │
│   ├── metrics/                               # Métricas
│   │   ├── __init__.py
│   │   ├── data_quality_metrics.py
│   │   └── processor.py
│   │
│   ├── validation/                            # Validação
│   │   ├── __init__.py
│   │   ├── adaptive.py
│   │   └── validator.py
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── logging_utils.py
│   └── pipeline.py
│
├── data_handler.py                            # Manipulador de dados
├── data_enricher.py                           # Enriquecedor de dados
├── data_validator.py                          # Validador de dados
├── data_quality_validator.py                  # Validador de qualidade
├── context_collector.py                       # Coletor de contexto
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         📡 EVENTOS E MENSAGERIA                             │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── event_bus.py                               # Barramento de eventos
├── event_saver.py                             # Salvador de eventos
├── event_memory.py                            # Memória de eventos
├── event_stats_model.py                       # Modelo de estatísticas
├── enrichment_integrator.py                   # Integrador de enriquecimento
├── cross_asset_correlations.py                # Correlações cross-asset
├── websocket_handler.py                       # Handler WebSocket
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🧠 MACHINE LEARNING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── ml/                                        # ML Models
│   │
│   ├── datasets/                              # Datasets
│   │   └── training_dataset.parquet
│   │
│   ├── models/                                # Modelos treinados
│   │   ├── xgb_model_latest.json
│   │   ├── model_metadata_latest.json
│   │   ├── feature_importance_*.csv
│   │   └── error_log_*.txt
│   │
│   ├── generate_dataset.py                    # Gerador de dataset
│   ├── train_model.py                         # Treinamento
│   ├── inference_engine.py                    # Engine de inferência
│   ├── model_inference.py                     # Inferência
│   ├── hybrid_decision.py                     # Decisão híbrida
│   └── training.log                           # Log de treinamento
│
├── ml_features.py                             # Features ML
├── feature_store.py                           # Feature store
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🌐 DADOS MACROECONÔMICOS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── src/                                       # Código fonte principal
│   │
│   ├── analysis/                              # Análise
│   │   ├── regime_detector.py
│   │   ├── regime_integration.py
│   │   ├── integrate_regime_detector.py
│   │   └── ai_payload_integrator.py
│   │
│   ├── bridges/                               # Pontes
│   │   └── async_bridge.py
│   │
│   ├── data/                                  # Dados
│   │   ├── macro_data_provider.py
│   │   ├── indices_futures.csv
│   │   └── macro_data.json
│   │
│   ├── rules/                                 # Regras
│   │   └── regime_rules.py
│   │
│   ├── services/                              # Serviços
│   │   ├── macro_service.py
│   │   └── macro_update_service.py
│   │
│   └── utils/                                 # Utilitários
│       ├── ai_payload_optimizer.py
│       ├── async_helpers.py
│       └── types_fredapi.pyi
│
├── macro_data_fetcher.py                      # Fetcher de dados macro
├── macro_fetcher.py                           # Fetcher alternativo
├── fred_fetcher.py                            # Fetcher FRED
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         ⚠️ ALERTAS E MONITORAMENTO                          │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── alert_engine.py                            # Motor de alertas
├── alert_manager.py                           # Gerenciador de alertas
├── health_monitor.py                          # Monitor de saúde
├── metrics_collector.py                       # Coletor de métricas
├── prometheus_exporter.py                     # Exportador Prometheus
├── trade_validator.py                         # Validador de trades
├── trade_buffer.py                            # Buffer de trades
├── time_manager.py                            # Gerenciador de tempo
├── clock_sync.py                              # Sincronização de clock
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🛡️ GERENCIAMENTO DE RISCO                          │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── risk_management/                           # Gestão de risco
│   ├── __init__.py
│   ├── risk_manager.py
│   └── exceptions.py
│
├── demo_circuit_breaker.py                    # Demo circuit breaker
├── market_impact.py                           # Impacto de mercado
├── dynamic_volume_profile.py                  # Perfil de volume dinâmico
├── liquidity_heatmap.py                       # Mapa de calor de liquidez
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🗄️ BANCO DE DADOS E ARMAZENAMENTO                   │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── database/                                  # Banco de dados
│   ├── __init__.py
│   └── event_store.py
│
├── dados/                                     # Dados da aplicação
│   ├── eventos-fluxo.json
│   ├── eventos_fluxo.jsonl
│   ├── eventos_visuais.log
│   └── trading_bot.db
│
├── memory/                                    # Memória/cache
│   ├── __init__.py
│   └── levels_BTCUSDT.json
│
├── features/                                  # Features (Parquet)
│   ├── date=2025-12-08/
│   ├── date=2025-12-09/
│   ├── date=2025-12-11/
│   ├── date=2025-12-17/
│   ├── date=2025-12-18/
│   ├── date=2025-12-20/
│   ├── date=2025-12-21/
│   ├── date=2026-01-01/
│   ├── date=2026-01-02/
│   ├── date=2026-01-03/
│   ├── date=2026-01-04/
│   ├── date=2026-01-05/
│   ├── date=2026-01-06/
│   ├── date=2026-01-13/
│   ├── date=2026-01-19/
│   ├── date=2026-01-20/
│   ├── date=2026-01-21/
│   ├── date=2026-01-24/
│   ├── date=2026-01-30/
│   ├── date=2026-02-09/
│   ├── date=2026-02-11/
│   ├── date=2026-02-12/
│   ├── date=2026-02-21/
│   ├── date=2026-02-22/
│   ├── date=2026-02-23/
│   ├── date=2026-02-24/
│   ├── date=2026-02-25/
│   └── date=2026-03-01/
│
├── logs/                                      # Logs
│   ├── eventos-fluxo.optimized.jsonl
│   ├── last_llm_payload.json
│   ├── payload_metrics.jsonl
│   └── payload_section_cache.json
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🔧 UTILITÁRIOS E FERRAMENTAS                        │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── utils/                                     # Utilitários
│   ├── __init__.py
│   ├── async_helpers.py
│   ├── heartbeat_manager.py
│   ├── trade_filter.py
│   └── trade_timestamp_validator.py
│
├── tools/                                     # Ferramentas
│   ├── export_db_to_jsonl.py
│   ├── inspect_db.py
│   ├── inspect_events_schema.py
│   └── ws_test.py
│
├── scripts/                                   # Scripts
│   ├── ab_test_prompt_styles.py
│   ├── analyze_ai_usage.py
│   ├── audit_json_payload_costs.py
│   ├── backup_to_oci.py
│   ├── validate_regime_system.py
│   ├── test_fixes.py
│   ├── disaster_recovery.sh
│   └── remote_health_check.sh
│
├── format_utils.py                            # Utilitários de formatação
├── process_csv_data.py                        # Processamento CSV
├── export_signals.py                          # Exportação de sinais
├── report_generator.py                        # Gerador de relatórios
├── pattern_recognition.py                     # Reconhecimento de padrões
├── technical_indicators.py                    # Indicadores técnicos
├── validation_check.py                        # Validação
├── validar_evento.py                          # Validar evento
├── log_formatter.py                           # Formatador de logs
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🧪 TESTES                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── tests/                                     # Testes
│   │
│   ├── fixtures/                              # Fixtures
│   │   └── sample_analysis_trigger.json
│   │
│   ├── payload/                               # Testes de payload
│   │   ├── test_payload_compressor.py
│   │   ├── test_payload_guardrail.py
│   │   ├── test_payload_metrics_aggregator.py
│   │   ├── test_payload_optimizer.py
│   │   └── test_payload_tripwires.py
│   │
│   ├── __init__.py
│   ├── conftest.py                            # Config pytest
│   ├── backtester.py                          # Backtester
│   ├── mock_ai_responses.py                   # Mock IA
│   ├── mock_qwen.py                           # Mock Qwen
│   ├── regime_scenario_tester.py              # Teste de cenários
│   │
│   ├── test_ai_runner.py                      # Testes AI Runner
│   ├── test_ai_runner_comprehensive.py
│   ├── test_market_orchestrator_comprehensive.py
│   ├── test_orderbook_analyzer.py
│   ├── test_orderbook_analyzer_comprehensive.py
│   ├── test_orderbook_core_comprehensive.py
│   ├── test_flow_analyzer.py
│   ├── test_data_pipeline.py
│   ├── test_support_resistance_consolidated.py
│   ├── test_support_resistance_modular.py
│   ├── test_defense_zones.py
│   ├── test_sr_strength.py
│   ├── test_risk_manager_comprehensive.py
│   ├── test_regime_integration.py
│   ├── test_cross_asset_integration.py
│   ├── test_enhanced_cross_asset.py
│   ├── test_macro_data_provider.py
│   ├── test_data_validator.py
│   ├── test_event_bus.py
│   ├── test_trade_flow_analyzer.py
│   ├── test_window_processor.py
│   ├── test_out_of_order_pruning.py
│   ├── test_rolling_aggregate.py
│   ├── test_circuit_breaker.py
│   ├── test_export_signals.py
│   ├── test_feature_store.py
│   ├── test_enrich_signal.py
│   ├── test_institutional_alerts.py
│   ├── test_performance_benchmarks.py
│   ├── test_trade_buffer_optimization.py
│   └── ... (outros 50+ arquivos de teste)
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🔍 DIAGNÓSTICO E DEBUG                              │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── diagnostics/                               # Diagnósticos
│   ├── analyze_ai_results.py
│   ├── evaluate_ai_performance.py
│   ├── final_validation.py
│   ├── performance_metrics.py
│   ├── replay_validator.py
│   └── verify_ml_integration.py
│
├── arquivos para diagnostico/                 # Diagnósticos de janelas
│   └── diagnostico de janelas geradas/
│       ├── diagnostico_avancado.py
│       ├── diagnostico_duplicatas.py
│       ├── diagnostico_janelas.py
│       └── fix_duplicatas_completo.py
│
├── debug_bot.py                               # Debug do bot
├── debug_env.py                               # Debug de ambiente
├── debug_keyerror.py                          # Debug KeyError
├── debug_payload.py                           # Debug payload
├── debug_validator.py                         # Debug validador
├── diagnose_crash.py                          # Diagnóstico de crash
├── diagnose_optimization.py                   # Diagnóstico otimização
├── fix_bot_run.py                             # Fix execução bot
├── fix_broken_tests.py                        # Fix testes quebrados
├── fix_duplicatas.py                          # Fix duplicatas
├── fix_optimization.py                        # Fix otimização
├── fix_timestamp.py                           # Fix timestamp
├── fix_separador_final.py                     # Fix separador
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         📚 DOCUMENTAÇÃO                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── docs/                                      # Documentação
│   ├── architecture.md
│   ├── RUNBOOK.md
│   └── troubleshooting.md
│
├── README.md                                  # Leia-me principal
├── README_OPTIMIZATION.md                     # Otimização
├── ESTRUTURA_SISTEMA_COMPLETO.md              # Estrutura completa
├── auditoria_estrutura_json.md                # Auditoria JSON
├── relatorio_auditoria_json.md                # Relatório auditoria
├── RELATORIO_ENRICHMENT_CROSS_ASSET.md        # Enriquecimento
├── RELATORIO_FINAL_MACRO_PROVIDER.md          # Macro provider
├── RESUMO_EXPORT_SINAIS.md                    # Export sinais
├── PATCH_SUMMARY.md                           # Resumo patches
├── CORRECAO_ENRICH_EVENT_SUMMARY.md           # Correções
├── CORRECAO_FETCH_INTERMARKET_DATA.md
└── orderbook_severity_analysis.md
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🏗️ INFRAESTRUTURA                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── infrastructure/                            # Infraestrutura
│   ├── oci/                                   # Oracle Cloud
│   │   ├── monitoring.py
│   │   ├── vault_helper.py
│   │   └── security_config.md
│   │
│   └── terraform/                             # Terraform
│       └── main.tf
│
├── deploy_oracle.sh                           # Deploy Oracle
├── setup_test_environment.sh                  # Setup ambiente teste
├── run_tests_with_coverage.sh                 # Rodar testes com cobertura
├── run_tests_windows.bat                      # Rodar testes Windows
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         📋 REGRAS E DOCUMENTOS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── Regras/                                    # Regras de negócio
│   ├── Rastreando robos/
│   │   ├── ESTRUTURANDO ARQUIVO JSON.odt
│   │   └── ROBOS X INTEGIGENCIA IA.odt
│   ├── COMPRIMIR DADOS.API.odt
│   ├── metodos institucional.docx
│   ├── regras para o codigo.odt
│   └── Teia de monitoramento Mini Dolar (B3).odt
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🎮 MQL5 (METATRADER)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── MQL5/                                      # MetaTrader 5
│   ├── Indicators/
│   │   └── ChartSignalsFromCSV.mq5
│   └── __init__.py
│
├─────────────────────────────────────────────────────────────────────────────┐
│                         🧪 LEGADO E BACKUPS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│
├── legacy/                                    # Código legado
│   ├── data_pipeline_legacy.py
│   ├── market_analyzer_2_3_0.py
│   └── support_resistance_legacy.py
│
└── [arquivos de backup e versões antigas]

```

---

## 📊 Resumo da Estrutura

| Categoria | Diretórios Principais |
|-----------|----------------------|
| **Core** | `main.py`, `market_orchestrator/`, `orderbook_core/` |
| **Análise** | `flow_analyzer/`, `support_resistance/`, `orderbook_analyzer/` |
| **Dados** | `data_pipeline/`, `src/`, `database/`, `dados/`, `features/` |
| **IA/ML** | `ai_runner/`, `ml/`, `context_collector.py` |
| **Infra** | `infrastructure/`, `config/`, `Dockerfile` |
| **Testes** | `tests/` (241 arquivos) |
| **Docs** | `docs/`, `*.md` |

---

**Total: 60.266 arquivos | 2.823 diretórios | 27.568 arquivos Python**
