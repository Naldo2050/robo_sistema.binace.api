import os

def generate_markdown_structure():
    root_dir = "."
    md_content = []
    
    md_content.append("# 🗂️ ESTRUTURA VISUAL DO SISTEMA - ROBO BINANCE\n")
    md_content.append("```")
    md_content.append("robo_sistema.binace.api/")
    md_content.append("│")
    
    # Categorias com seus diretorios e arquivos
    categories = {
        "📁 CONFIGURAÇÃO E SETUP": {
            "config/": [
                "__init__.py",
                "model_config.yaml"
            ],
            "root": [
                "config.json",
                "config.py",
                "requirements.txt",
                "requirements-dev.txt",
                "pyproject.toml",
                "pytest.ini",
                "mypy.ini",
                "pyrightconfig.json",
                ".coveragerc",
                ".gitignore",
                ".dockerignore",
                "Dockerfile",
                "docker-compose.yml",
                ".env"
            ]
        },
        
        "🚀 PONTO DE ENTRADA": {
            "root": [
                "main.py",
                "main.patched.py",
                "dashboard.py",
                "enhanced_market_bot.py",
                "market_analyzer.py"
            ]
        },
        
        "🤖 INTELIGÊNCIA ARTIFICIAL": {
            "ai_runner/": [
                "__init__.py",
                "ai_runner.py",
                "exceptions.py"
            ],
            "root": [
                "ai_analyzer_qwen.py",
                "ai_analyzer_qwen_patch2.py",
                "ai_analyzer_disabled.py",
                "ai_historical_pro.py",
                "ai_payload_compressor.py",
                "optimize_ai_payload.py",
                "payload_optimizer_config.py",
                "build_compact_payload.py"
            ]
        },
        
        "🎯 ORQUESTRADOR DE MERCADO": {
            "market_orchestrator/": [
                "ai/",
                "analysis/",
                "connection/",
                "flow/",
                "orderbook/",
                "signals/",
                "utils/",
                "windows/",
                "__init__.py",
                "market_orchestrator.py",
                "orchestrator.py"
            ],
            "market_orchestrator/ai/": [
                "__init__.py",
                "ai_enrichment_context.py",
                "ai_payload_builder.py",
                "ai_runner.py",
                "llm_payload_guardrail.py",
                "payload_compressor.py",
                "payload_compressor_v3.py",
                "payload_metrics_aggregator.py",
                "payload_section_cache.py",
                "raw_event_deduplicator.py"
            ],
            "market_orchestrator/analysis/": [
                "__init__.py",
                "institutional_analytics.py"
            ],
            "market_orchestrator/connection/": [
                "__init__.py",
                "robust_connection.py"
            ],
            "market_orchestrator/flow/": [
                "__init__.py",
                "risk_manager.py",
                "signal_processor.py",
                "trade_executor.py",
                "trade_flow_analyzer.py"
            ],
            "market_orchestrator/orderbook/": [
                "__init__.py",
                "orderbook_wrapper.py"
            ],
            "market_orchestrator/signals/": [
                "__init__.py",
                "signal_processor.py"
            ],
            "market_orchestrator/utils/": [
                "__init__.py",
                "logging_utils.py",
                "price_fetcher.py"
            ],
            "market_orchestrator/windows/": [
                "__init__.py",
                "window_processor.py"
            ]
        },
        
        "📊 ORDERBOOK": {
            "orderbook_core/": [
                "__init__.py",
                "orderbook.py",
                "orderbook_config.py",
                "circuit_breaker.py",
                "event_factory.py",
                "metrics.py",
                "protocols.py",
                "structured_logging.py",
                "tracing_utils.py",
                "constants.py",
                "exceptions.py"
            ],
            "orderbook_analyzer/": [
                "__init__.py",
                "analyzer.py",
                "spread_tracker.py",
                "config/"
            ],
            "orderbook_analyzer/config/": [
                "__init__.py",
                "settings.py"
            ],
            "root": [
                "orderbook_analyzer.py",
                "orderbook_fallback.py",
                "orderbook_ws_manager.py"
            ]
        },
        
        "📈 SUPORTE E RESISTÊNCIA": {
            "support_resistance/": [
                "__init__.py",
                "core.py",
                "system.py",
                "config.py",
                "constants.py",
                "defense_zones.py",
                "monitor.py",
                "pivot_points.py",
                "reference_prices.py",
                "sr_strength.py",
                "utils.py",
                "validation.py",
                "volume_profile.py"
            ],
            "root": [
                "levels_registry.py"
            ]
        },
        
        "🔥 ANÁLISE DE FLUXO": {
            "flow_analyzer/": [
                "__init__.py",
                "core.py",
                "absorption.py",
                "aggregates.py",
                "whale_score.py",
                "metrics.py",
                "protocols.py",
                "validation.py",
                "serialization.py",
                "utils.py",
                "profiling.py",
                "logging_config.py",
                "prometheus_metrics.py",
                "errors.py",
                "constants.py"
            ],
            "root": [
                "flow_analyzer.py"
            ]
        },
        
        "🔄 PIPELINE DE DADOS": {
            "data_pipeline/": [
                "cache/",
                "fallback/",
                "metrics/",
                "validation/",
                "__init__.py",
                "config.py",
                "logging_utils.py",
                "pipeline.py"
            ],
            "data_pipeline/cache/": [
                "__init__.py",
                "buffer.py",
                "lru_cache.py"
            ],
            "data_pipeline/fallback/": [
                "__init__.py",
                "registry.py"
            ],
            "data_pipeline/metrics/": [
                "__init__.py",
                "data_quality_metrics.py",
                "processor.py"
            ],
            "data_pipeline/validation/": [
                "__init__.py",
                "adaptive.py",
                "validator.py"
            ],
            "root": [
                "data_handler.py",
                "data_enricher.py",
                "data_validator.py",
                "data_quality_validator.py",
                "context_collector.py"
            ]
        },
        
        "📡 EVENTOS E MENSAGERIA": {
            "root": [
                "event_bus.py",
                "event_saver.py",
                "event_memory.py",
                "event_stats_model.py",
                "enrichment_integrator.py",
                "cross_asset_correlations.py",
                "websocket_handler.py",
                "event_similarity.py"
            ]
        },
        
        "🧠 MACHINE LEARNING": {
            "ml/": [
                "datasets/",
                "models/",
                "generate_dataset.py",
                "train_model.py",
                "inference_engine.py",
                "model_inference.py",
                "hybrid_decision.py",
                "training.log"
            ],
            "ml/datasets/": [
                "training_dataset.parquet"
            ],
            "root": [
                "ml_features.py",
                "feature_store.py"
            ]
        },
        
        "🌐 DADOS MACROECONÔMICOS": {
            "src/": [
                "analysis/",
                "bridges/",
                "data/",
                "rules/",
                "services/",
                "utils/"
            ],
            "src/analysis/": [
                "regime_detector.py",
                "regime_integration.py",
                "integrate_regime_detector.py",
                "ai_payload_integrator.py"
            ],
            "src/bridges/": [
                "__init__.py",
                "async_bridge.py"
            ],
            "src/data/": [
                "macro_data_provider.py",
                "indices_futures.csv",
                "macro_data.json"
            ],
            "src/rules/": [
                "regime_rules.py"
            ],
            "src/services/": [
                "__init__.py",
                "macro_service.py",
                "macro_update_service.py"
            ],
            "src/utils/": [
                "__init__.py",
                "ai_payload_optimizer.py",
                "async_helpers.py",
                "types_fredapi.pyi"
            ],
            "root": [
                "macro_data_fetcher.py",
                "macro_fetcher.py",
                "fred_fetcher.py",
                "onchain_fetcher.py"
            ]
        },
        
        "⚠️ ALERTAS E MONITORAMENTO": {
            "root": [
                "alert_engine.py",
                "alert_manager.py",
                "health_monitor.py",
                "metrics_collector.py",
                "prometheus_exporter.py",
                "trade_validator.py",
                "trade_buffer.py",
                "time_manager.py",
                "clock_sync.py"
            ]
        },
        
        "🛡️ GERENCIAMENTO DE RISCO": {
            "risk_management/": [
                "__init__.py",
                "risk_manager.py",
                "exceptions.py"
            ],
            "root": [
                "demo_circuit_breaker.py",
                "market_impact.py",
                "dynamic_volume_profile.py",
                "liquidity_heatmap.py"
            ]
        },
        
        "🗄️ BANCO DE DADOS E ARMAZENAMENTO": {
            "database/": [
                "__init__.py",
                "event_store.py"
            ],
            "dados/": [
                "eventos-fluxo.json",
                "eventos_fluxo.jsonl",
                "eventos_visuais.log",
                "trading_bot.db"
            ],
            "memory/": [
                "__init__.py",
                "levels_BTCUSDT.json"
            ],
            "features/": [
                "date=*/"  # Representa todos os subdiretorios de datas
            ]
        },
        
        "🔧 UTILITÁRIOS E FERRAMENTAS": {
            "utils/": [
                "__init__.py",
                "async_helpers.py",
                "heartbeat_manager.py",
                "trade_filter.py",
                "trade_timestamp_validator.py"
            ],
            "tools/": [
                "export_db_to_jsonl.py",
                "inspect_db.py",
                "inspect_events_schema.py",
                "ws_test.py"
            ],
            "scripts/": [
                "ab_test_prompt_styles.py",
                "analyze_ai_usage.py",
                "audit_json_payload_costs.py",
                "backup_to_oci.py",
                "validate_regime_system.py",
                "test_fixes.py",
                "test_fixes_final.py",
                "test_fixes_simple.py",
                "test_payload.sh",
                "disaster_recovery.sh",
                "remote_health_check.sh"
            ],
            "root": [
                "format_utils.py",
                "process_csv_data.py",
                "export_signals.py",
                "report_generator.py",
                "pattern_recognition.py",
                "technical_indicators.py",
                "validation_check.py",
                "validar_evento.py",
                "log_formatter.py",
                "funding_aggregator.py",
                "historical_profiler.py"
            ]
        },
        
        "🧪 TESTES": {
            "tests/": [
                "fixtures/",
                "payload/",
                "__init__.py",
                "conftest.py",
                "backtester.py",
                "config_test.py",
                "mock_ai_responses.py",
                "mock_qwen.py",
                "regime_scenario_tester.py",
                "fix_broken_tests.py",
                "fix_qwen_import.py"
            ],
            "tests/fixtures/": [
                "sample_analysis_trigger.json"
            ],
            "tests/payload/": [
                "conftest.py",
                "pytest.ini",
                "test_payload_compressor.py",
                "test_payload_guardrail.py",
                "test_payload_metrics_aggregator.py",
                "test_payload_optimizer.py",
                "test_payload_tripwires.py"
            ],
            "root": [
                "test_circuit_breaker_improvements.py",
                "test_circuit_breaker_integration.py",
                "test_config_imports.py",
                "test_connection.py",
                "test_cross_asset.py",
                "test_cross_asset_integration.py",
                "test_diagnostic.py",
                "test_dynamic_volume_profile_2.py",
                "test_enhanced_cross_asset.py",
                "test_enrich_correction.py",
                "test_enrich_event.py",
                "test_enrich_simple.py",
                "test_export_signals.py",
                "test_feature_store.py",
                "test_fixes_simple.py",
                "test_fixes_simple_fixed.py",
                "test_functions.py",
                "test_integrated_macro_provider.py",
                "test_latency_fix_simple.py",
                "test_macro_singleton_fix.py",
                "test_new_payload.py",
                "test_optimization.py",
                "test_pipeline_integration.py",
                "test_regime_integration.py",
                "test_simple_correlations.py",
                "test_trade_buffer_optimization.py",
                "test_updated_correlations.py",
                "test_websocket.py",
                "teste_cross_asset_final.py",
                "teste_rapido.py",
                "teste_rapido_corrigido.py",
                "teste_separador.py"
            ]
        },
        
        "🔍 DIAGNÓSTICO E DEBUG": {
            "diagnostics/": [
                "analyze_ai_results.py",
                "evaluate_ai_performance.py",
                "final_validation.py",
                "performance_metrics.py",
                "replay_validator.py",
                "verify_ml_integration.py"
            ],
            "arquivos para diagnostico/": [
                "__init__.py",
                "diagnostico de janelas geradas/"
            ],
            "arquivos para diagnostico/diagnostico de janelas geradas/": [
                "__init__.py",
                "diagnostico_avancado.py",
                "diagnostico_duplicatas.py",
                "diagnostico_janelas.py",
                "fix_duplicatas_completo.py"
            ],
            "root": [
                "debug_bot.py",
                "debug_env.py",
                "debug_keyerror.py",
                "debug_payload.py",
                "debug_validator.py",
                "diagnose_crash.py",
                "diagnose_optimization.py",
                "fix_bot_run.py",
                "fix_broken_tests.py",
                "fix_duplicatas.py",
                "fix_optimization.py",
                "fix_playwright.py",
                "fix_separador_final.py",
                "fix_timestamp.py"
            ]
        },
        
        "📚 DOCUMENTAÇÃO": {
            "docs/": [
                "architecture.md",
                "RUNBOOK.md",
                "troubleshooting.md"
            ],
            "root": [
                "README.md",
                "README_OPTIMIZATION.md",
                "ESTRUTURA_SISTEMA_COMPLETO.md",
                "ESTRUTURA_VISUAL_SISTEMA.md",
                "auditoria_estrutura_json.md",
                "relatorio_auditoria_json.md",
                "RELATORIO_ENRICHMENT_CROSS_ASSET.md",
                "RELATORIO_FINAL_MACRO_PROVIDER.md",
                "RESUMO_EXPORT_SINAIS.md",
                "PATCH_SUMMARY.md",
                "CORRECAO_ENRICH_EVENT_SUMMARY.md",
                "CORRECAO_FETCH_INTERMARKET_DATA.md",
                "orderbook_severity_analysis.md"
            ]
        },
        
        "🏗️ INFRAESTRUTURA": {
            "infrastructure/": [
                "oci/",
                "terraform/",
                "__init__.py",
                "market-bot.service"
            ],
            "infrastructure/oci/": [
                "__init__.py",
                "monitoring.py",
                "vault_helper.py",
                "security_config.md"
            ],
            "infrastructure/terraform/": [
                "main.tf"
            ],
            "root": [
                "deploy_oracle.sh",
                "setup_test_environment.sh",
                "run_tests_with_coverage.sh",
                "run_tests_windows.bat"
            ]
        },
        
        "🎮 MQL5 (METATRADER)": {
            "MQL5/": [
                "Indicators/",
                "__init__.py"
            ],
            "MQL5/Indicators/": [
                "ChartSignalsFromCSV.mq5"
            ]
        },
        
        "📋 REGRAS E DOCUMENTOS": {
            "Regras/": [
                "Rastreando robos/",
                "COMPRIMIR DADOS.API.odt",
                "metodos institucional.docx",
                "regras para o codigo.odt",
                "Teia de monitoramento Mini Dolar (B3).odt"
            ],
            "Regras/Rastreando robos/": [
                "ESTRUTURANDO ARQUIVO JSON.odt",
                "ROBOS X INTEGIGENCIA IA.odt"
            ]
        },
        
        "🧪 LEGADO E BACKUPS": {
            "legacy/": [
                "data_pipeline_legacy..py",
                "market_analyzer_2_3_0.py",
                "support_resistance_legacy.py"
            ],
            "root": [
                "0.17",
                "0.5.0"
            ]
        },
        
        "📦 OUTROS": {
            "root": [
                "dados_mercado.csv",
                "example_analysis_trigger_structure.json",
                "regime_test_report.json",
                "relatorio.json",
                "create_structure.py",
                "integration_validator.py",
                "institutional_enricher.py",
                "outcome_tracker.py",
                "reproduce_issue.py",
                "show_problem_lines.py",
                "verificar_otimizacao.py",
                "verify_implementations.py",
                "verify_patch.py",
                "audit_new_features.py",
                "audit_script.py",
                "full_audit.py",
                "modelo_dados_ideal.py"
            ]
        }
    }
    
    # Gerar cada categoria
    for category, structure in categories.items():
        md_content.append("│")
        md_content.append(f"├─────────────────────────────────────────────────────────────────────────────┐")
        md_content.append(f"│                         {category}                             │")
        md_content.append(f"├─────────────────────────────────────────────────────────────────────────────┤")
        md_content.append("│")
        
        # Processar subdiretorios
        for path, items in structure.items():
            if path == "root":
                for item in items:
                    if os.path.exists(item):
                        md_content.append(f"├── {item}")
            else:
                if os.path.exists(path):
                    md_content.append(f"├── {path}")
                    for item in items:
                        item_path = os.path.join(path, item)
                        if os.path.exists(item_path):
                            md_content.append(f"│   ├── {item}")
                        elif item.endswith("/"):
                            # Se é um subdiretorio que pode nao existir ou é padrao
                            md_content.append(f"│   ├── {item}")
    
    md_content.append("│")
    md_content.append("└── [arquivos temporários, backups e versões antigas]")
    md_content.append("```")
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    md_content.append("## 📊 Resumo da Estrutura")
    md_content.append("")
    md_content.append("| Categoria | Diretórios Principais |")
    md_content.append("|-----------|----------------------|")
    md_content.append("| **Core** | `main.py`, `market_orchestrator/`, `orderbook_core/` |")
    md_content.append("| **Análise** | `flow_analyzer/`, `support_resistance/`, `orderbook_analyzer/` |")
    md_content.append("| **Dados** | `data_pipeline/`, `src/`, `database/`, `dados/`, `features/` |")
    md_content.append("| **IA/ML** | `ai_runner/`, `ml/`, `context_collector.py` |")
    md_content.append("| **Infra** | `infrastructure/`, `config/`, `Dockerfile` |")
    md_content.append("| **Testes** | `tests/` |")
    md_content.append("| **Docs** | `docs/`, `*.md` |")
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    
    # Calcular estatisticas
    total_files = 0
    total_dirs = 0
    python_files = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Ignorar diretorios especificos
        if any(excl in dirpath for excl in ['.git', '.mypy_cache', '__pycache__', '.pytest_cache', 'coverage_html', '.agent', '.claude', '-p', '.venv', 'playwright_user_data', 'logs']):
            continue
        
        total_dirs += 1
        for filename in filenames:
            total_files += 1
            if filename.endswith('.py'):
                python_files += 1
    
    md_content.append(f"**Total: {total_files:,} arquivos | {total_dirs:,} diretórios | {python_files:,} arquivos Python**")
    
    return "\n".join(md_content)

def main():
    md_content = generate_markdown_structure()
    with open("ESTRUTURA_VISUAL_SISTEMA.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("ESTRUTURA_VISUAL_SISTEMA.md atualizado com sucesso!")

if __name__ == "__main__":
    main()
