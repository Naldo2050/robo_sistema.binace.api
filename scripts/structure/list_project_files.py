import os

def list_files_by_category():
    root_dir = "."
    
    categories = {
        "Configuração e Setup": [
            ".coveragerc",
            ".dockerignore",
            ".env",
            ".gitignore",
            "config.json",
            "config.py",
            "requirements.txt",
            "requirements-dev.txt",
            "pyproject.toml",
            "pytest.ini",
            "mypy.ini",
            "pyrightconfig.json",
            "Dockerfile",
            "docker-compose.yml",
            "config/",
            "infrastructure/",
        ],
        
        "Ponto de Entrada": [
            "main.py",
            "main.patched.py",
            "dashboard.py",
            "enhanced_market_bot.py",
            "market_analyzer.py"
        ],
        
        "Inteligência Artificial": [
            "ai_runner/",
            "ai_analyzer_qwen.py",
            "ai_analyzer_qwen_patch2.py",
            "ai_analyzer_disabled.py",
            "ai_historical_pro.py",
            "ai_payload_compressor.py",
            "optimize_ai_payload.py",
            "payload_optimizer_config.py",
            "build_compact_payload.py"
        ],
        
        "Orquestrador de Mercado": [
            "market_orchestrator/",
        ],
        
        "Orderbook": [
            "orderbook_core/",
            "orderbook_analyzer/",
            "orderbook_analyzer.py",
            "orderbook_fallback.py",
            "orderbook_ws_manager.py"
        ],
        
        "Suporte e Resistência": [
            "support_resistance/",
            "levels_registry.py"
        ],
        
        "Análise de Fluxo": [
            "flow_analyzer/",
            "flow_analyzer.py"
        ],
        
        "Pipeline de Dados": [
            "data_pipeline/",
            "data_handler.py",
            "data_enricher.py",
            "data_validator.py",
            "data_quality_validator.py",
            "context_collector.py"
        ],
        
        "Eventos e Mensageria": [
            "event_bus.py",
            "event_saver.py",
            "event_memory.py",
            "event_stats_model.py",
            "enrichment_integrator.py",
            "cross_asset_correlations.py",
            "websocket_handler.py",
            "event_similarity.py"
        ],
        
        "Machine Learning": [
            "ml/",
            "ml_features.py",
            "feature_store.py"
        ],
        
        "Dados Macroeconômicos": [
            "src/",
            "macro_data_fetcher.py",
            "macro_fetcher.py",
            "fred_fetcher.py",
            "onchain_fetcher.py"
        ],
        
        "Alertas e Monitoramento": [
            "alert_engine.py",
            "alert_manager.py",
            "health_monitor.py",
            "metrics_collector.py",
            "prometheus_exporter.py",
            "trade_validator.py",
            "trade_buffer.py",
            "time_manager.py",
            "clock_sync.py"
        ],
        
        "Gerenciamento de Risco": [
            "risk_management/",
            "demo_circuit_breaker.py",
            "market_impact.py",
            "dynamic_volume_profile.py",
            "liquidity_heatmap.py"
        ],
        
        "Banco de Dados e Armazenamento": [
            "database/",
            "dados/",
            "memory/",
            "features/"
        ],
        
        "Utilitários e Ferramentas": [
            "utils/",
            "tools/",
            "scripts/",
            "format_utils.py",
            "process_csv_data.py",
            "export_signals.py",
            "report_generator.py",
            "pattern_recognition.py",
            "technical_indicators.py",
            "validation_check.py",
            "validate_event.py",
            "log_formatter.py",
            "funding_aggregator.py",
            "historical_profiler.py"
        ],
        
        "Testes": [
            "tests/",
        ],
        
        "Diagnóstico e Debug": [
            "diagnostics/",
            "diagnostic_files/",
            "debug_bot.py",
            "debug_env.py",
            "debug_keyerror.py",
            "debug_payload.py",
            "debug_validator.py",
            "diagnose_crash.py",
            "diagnose_optimization.py",
            "fix_bot_run.py",
            "fix_broken_tests.py",
            "fix_duplicates.py",
            "fix_optimization.py",
            "fix_playwright.py",
            "fix_separator_final.py",
            "fix_timestamp.py"
        ],
        
        "Documentação": [
            "docs/",
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
        ],
        
        "Infraestrutura": [
            "infrastructure/",
            "deploy_oracle.sh",
            "setup_test_environment.sh",
            "run_tests_with_coverage.sh",
            "run_tests_windows.bat"
        ],
        
        "Legado e Backups": [
            "legacy/",
        ],
        
        "Outros": [
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
            "full_audit.py"
        ]
    }
    
    for category, paths in categories.items():
        print(f"\n=== {category} ===")
        for path in paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    print(f"- {path}/")
                    # List contents of directory
                    try:
                        for item in sorted(os.listdir(path)):
                            item_path = os.path.join(path, item)
                            if os.path.isdir(item_path):
                                print(f"  ├─ {item}/")
                            else:
                                print(f"  ├─ {item}")
                    except:
                        pass
                else:
                    print(f"- {path}")
    
    print(f"\nTotal de categorias: {len(categories)}")

if __name__ == "__main__":
    list_files_by_category()
