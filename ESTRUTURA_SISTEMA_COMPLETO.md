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
| `mypy.ini` | Configuração de type checking |
| `pytest.ini` | Configuração do pytest |
| `pyproject.toml` | Configuração do projeto Python |
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
└── validation.py        # Validação de dados
```

---

### 🏛️ [`market_orchestrator/`](market_orchestrator/)
Orquestrador principal do mercado

```
market_orchestrator/
├── __init__.py
├── market_orchestrator.py  # Orquestrador principal
├── orchestrator.py
├── ai/
│   ├── __init__.py
│   ├── ai_enrichment_context.py   # Contexto de enriquecimento
│   ├── ai_payload_builder.py       # Construtor de payload
│   ├── ai_runner.py                # Executor de IA
│   ├── llm_payload_guardrail.py   # Guardrails
│   ├── payload_compressor.py      # Compressão
│   └── payload_metrics_aggregator.py
├── connection/
│   └── robust_connection.py  # Conexão robusta
├── flow/
│   ├── risk_manager.py       # Gerenciamento de risco
│   ├── signal_processor.py  # Processador de sinais
│   ├── trade_executor.py     # Execução de trades
│   └── trade_flow_analyzer.py
├── orderbook/
│   └── orderbook_wrapper.py  # Wrapper do orderbook
├── signals/
│   └── signal_processor.py   # Processador de sinais
├── utils/
│   ├── logging_utils.py
│   └── price_fetcher.py
└── windows/
    └── window_processor.py   # Processador de janelas
```

---

### 📈 [`support_resistance/`](support_resistance/)
Sistema de Suporte e Resistência

```
support_resistance/
├── __init__.py
├── config.py          # Configurações
├── constants.py       # Constantes
├── core.py            # Motor principal
├── monitor.py         # Monitor em tempo real
├── pivot_points.py    # Pontos de pivô
├── system.py          # Sistema completo
├── utils.py           # Utilitários
├── validation.py     # Validação
└── volume_profile.py  # Perfil de volume
```

---

### 🧠 [`ml/`](ml/)
Machine Learning e Inferência

```
ml/
├── generate_dataset.py     # Geração de datasets
├── hybrid_decision.py     # Decisão híbrida
├── inference_engine.py    # Motor de inferência
├── model_inference.py     # Inferência de modelo
└── train_model.py         # Treinamento de modelo
```

---

### 🔄 [`data_pipeline/`](data_pipeline/)
Pipeline de processamento de dados

```
data_pipeline/
├── __init__.py
├── config.py
├── logging_utils.py
├── pipeline.py           # Pipeline principal
├── cache/
│   ├── __init__.py
│   ├── buffer.py        # Buffer de cache
│   └── lru_cache.py     # Cache LRU
├── fallback/
│   ├── __init__.py
│   └── registry.py       # Registro de fallbacks
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

### 📚 [`tests/`](tests/)
Suíte de testes

```
tests/
├── __init__.py
├── backtester.py
├── conftest.py
├── config_test.py
├── fixtures.py
├── fixtures/
│   └── sample_analysis_trigger.json
├── mock_ai_responses.py
├── mock_qwen.py
├── regime_scenario_tester.py
├── payload/
│   ├── conftest.py
│   ├── pytest.ini
│   ├── test_payload_compressor.py
│   ├── test_payload_guardrail.py
│   ├── test_payload_metrics_aggregator.py
│   ├── test_payload_optimizer.py
│   └── test_payload_tripwires.py
├── test_ai_analyzer_language_and_think_strip.py
├── test_ai_analyzer_mock.py
├── test_ai_runner.py
├── test_ai_runner_comprehensive.py
├── test_circuit_breaker.py
├── test_data_pipeline.py
├── test_data_validator.py
├── test_enrich_signal.py
├── test_event_bus.py
├── test_integration_full_flow.py
├── test_macro_data_provider.py
├── test_market_orchestrator_comprehensive.py
├── test_orderbook_analyzer.py
├── test_orderbook_analyzer_coverage.py
├── test_orderbook_analyzer_full_coverage.py
├── test_orderbook_analyzer_missing.py
├── test_orderbook_analyze_core.py
├── test_orderbook_config_injection.py
├── test_orderbook_core_comprehensive.py
├── test_orderbook_helpers.py
├── test_orderbook_validate_snapshot.py
├── test_orderbook_wrapper_fallback.py
├── test_orderbook_wrapper_fetch_with_retry.py
└── ... (muitos outros arquivos de teste)
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
├── test_fixes_final.py
├── test_fixes_simple.py
├── test_payload.sh
└── validate_regime_system.py
```

---

### 🔧 [`tools/`](tools/)
Ferramentas de diagnóstico

```
tools/
├── export_db_to_jsonl.py
├── inspect_db.py
├── inspect_events_schema.py
└── ws_test.py
```

---

### 🔍 [`diagnostics/`](diagnostics/)
Ferramentas de diagnóstico

```
diagnostics/
├── analyze_ai_results.py
├── evaluate_ai_performance.py
├── final_validation.py
├── performance_metrics.py
├── replay_validator.py
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
├── oci/
│   ├── __init__.py
│   ├── monitoring.py
│   ├── security_config.md
│   └── vault_helper.py
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
├── data_pipeline_legacy.py
├── market_analyzer_2_3_0.py
└── support_resistance_legacy.py
```

---

### 🗂️ [`Regras/`](Regras/)
Regras e documentação

```
Regras/
├── COMPRIMIR DADOS.API.odt
├── regras para o codigo.odt
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

## 📂 Arquivos de Dados

### 📁 [`dados/`](dados/)
```
dados/
└── trading_bot.db  # Banco de dados SQLite
```

---

### 📁 [`logs/`](logs/)
Diretório de logs

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
└── date=2026-02-23/
```

---

## 📂 Arquivos Principais (Raiz)

### Análise de IA
| Arquivo | Descrição |
|---------|-----------|
| `ai_analyzer_qwen.py` | Analisador IA principal (119KB) |
| `ai_analyzer_qwen_patch2.py` | Patch v2 do analisador |
| `ai_analyzer_disabled.py` | Analisador desabilitado |
| `ai_historical_pro.py` | Histórico de IA |
| `ai_payload_compressor.py` | Compressor de payload |
| `context_collector.py` | Coletor de contexto |

### Análise de Mercado
| Arquivo | Descrição |
|---------|-----------|
| `orderbook_analyzer.py` | Analisador de orderbook (123KB) |
| `cross_asset_correlations.py` | Correlações cross-asset |
| `pattern_recognition.py` | Reconhecimento de padrões |
| `liquidity_heatmap.py` | Mapa de calor de liquidez |
| `dynamic_volume_profile.py` | Perfil de volume dinâmico |

### Dados e Validação
| Arquivo | Descrição |
|---------|-----------|
| `data_handler.py` | Manipulador de dados |
| `data_enricher.py` | Enriquecedor de dados |
| `data_validator.py` | Validador de dados |
| `data_quality_validator.py` | Validador de qualidade |
| `feature_store.py` | Store de features |

### Trading e Execução
| Arquivo | Descrição |
|---------|-----------|
| `trade_buffer.py` | Buffer de trades |
| `trade_validator.py` | Validador de trades |
| `alert_engine.py` | Motor de alertas |
| `alert_manager.py` | Gerenciador de alertas |
| `metrics_collector.py` | Coletor de métricas |

### Integração Externa
| Arquivo | Descrição |
|---------|-----------|
| `macro_data_fetcher.py` | Coletor de dados macroeconômicos |
| `macro_fetcher.py` | Fetcher de macro |
| `fred_fetcher.py` | Coletor do FRED |
| `websocket_handler.py` | Manipulador WebSocket |

### Sistema
| Arquivo | Descrição |
|---------|-----------|
| `event_bus.py` | Barramento de eventos |
| `event_saver.py` | Salvador de eventos |
| `time_manager.py` | Gerenciador de tempo |
| `clock_sync.py` | Sincronização de relógio |
| `health_monitor.py` | Monitor de saúde |

### Utilitários
| Arquivo | Descrição |
|---------|-----------|
| `format_utils.py` | Utilitários de formatação |
| `technical_indicators.py` | Indicadores técnicos |
| `ml_features.py` | Features de ML |
| `export_signals.py` | Exportador de sinais |
| `report_generator.py` | Gerador de relatórios |

### Debug e Desenvolvimento
| Arquivo | Descrição |
|---------|-----------|
| `debug_bot.py` | Debug do bot |
| `debug_env.py` | Debug de ambiente |
| `debug_keyerror.py` | Debug de KeyError |
| `debug_payload.py` | Debug de payload |
| `diagnose_crash.py` | Diagnóstico de crash |

---

## 📊 Estatísticas do Projeto

- **Total de arquivos Python**: ~150+
- **Total de módulos**: 20+
- **Linhas de código principais**: 50,000+
- **Testes**: 50+ arquivos de teste
- **Configurações**: YAML, JSON, INI, TOML

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

*Última atualização: 2026-02-23*
