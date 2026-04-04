# Fase 1 — Imports, Proxies e Inicialização

> Data: 2026-03-31 | Branch: audit/2026-03-31

---

## 1.1 Auditoria de Proxies na Raiz

**CRÍTICO: Todos os 29 arquivos de proxy foram REMOVIDOS da raiz.**

O documento `ESTRUTURA_SISTEMA_COMPLETO.md` descreve 29 proxies, mas nenhum existe na raiz.

### Status dos Pacotes Destino

| Proxy (raiz) | Destino | Destino Existe? | Status | Risco |
|---|---|---|---|---|
| `event_bus.py` | `events/event_bus.py` | ✅ Sim | Pacote usa import relativo internamente | BAIXO |
| `event_saver.py` | `events/event_saver.py` | ✅ Sim | OK | BAIXO |
| `event_memory.py` | `events/event_memory.py` | ✅ Sim | OK | BAIXO |
| `trade_buffer.py` | `trading/trade_buffer.py` | ✅ Sim | OK | BAIXO |
| `fred_fetcher.py` | `fetchers/fred_fetcher.py` | ✅ Sim | OK | BAIXO |
| `cross_asset_correlations.py` | `market_analysis/cross_asset_correlations.py` | ✅ Sim | OK | BAIXO |
| `dynamic_volume_profile.py` | `market_analysis/dynamic_volume_profile.py` | ✅ Sim | OK | BAIXO |
| `levels_registry.py` | `market_analysis/levels_registry.py` | ✅ Sim | OK | BAIXO |
| `data_handler.py` | `data_processing/data_handler.py` | ✅ Sim | OK | BAIXO |
| `data_enricher.py` | `data_processing/data_enricher.py` | ✅ Sim | OK | BAIXO |
| `data_validator.py` | `data_processing/data_validator.py` | ✅ Sim | OK | BAIXO |
| `data_quality_validator.py` | `data_processing/data_quality_validator.py` | ✅ Sim | OK | BAIXO |
| `time_manager.py` | `monitoring/time_manager.py` | ✅ Sim | OK | BAIXO |
| `health_monitor.py` | `monitoring/health_monitor.py` | ✅ Sim | OK | BAIXO |
| `metrics_collector.py` | `monitoring/metrics_collector.py` | ✅ Sim | OK | BAIXO |
| `format_utils.py` | `common/format_utils.py` | ✅ Sim | OK | BAIXO |
| `context_collector.py` | `fetchers/context_collector.py` | ✅ Sim | OK | BAIXO |
| `enrichment_integrator.py` | `data_processing/enrichment_integrator.py` | ✅ Sim | OK | BAIXO |
| `feature_store.py` | `data_processing/feature_store.py` | ✅ Sim | OK | BAIXO |
| `export_signals.py` | `trading/export_signals.py` | ✅ Sim | OK | BAIXO |
| `historical_profiler.py` | `market_analysis/historical_profiler.py` | ✅ Sim | OK | BAIXO |
| `report_generator.py` | `common/report_generator.py` | ✅ Sim | OK | BAIXO |
| `optimize_ai_payload.py` | `common/optimize_ai_payload.py` | ✅ Sim | Docstring cita "from optimize_ai_payload" mas não é import real | BAIXO |
| `payload_optimizer_config.py` | `common/payload_optimizer_config.py` | ✅ Sim | OK | BAIXO |
| `ai_payload_compressor.py` | `common/ai_payload_compressor.py` | ✅ Sim | OK | BAIXO |
| `ai_response_validator.py` | `common/ai_response_validator.py` | ✅ Sim | OK | BAIXO |
| `fix_optimization.py` | `data_processing/fix_optimization.py` | ✅ Sim | OK | BAIXO |
| `diagnose_optimization.py` | `scripts/diagnostics/diagnose_optimization.py` | ✅ Sim | OK | BAIXO |
| `orderbook_fallback.py` | `orderbook_core/orderbook_fallback.py` | ✅ Sim | OK | BAIXO |

### Impacto Real

Verificação via AST de todo o codebase: **nenhum arquivo de produção faz import absoluto dos nomes de proxy**. Todos os pacotes usam imports relativos (`from .module import X`). Os proxies foram removidos com sucesso e os importadores já migrados.

**Exceção**: `utils/__init__.py` usa imports absolutos (`from monitoring.heartbeat_manager`, `from trading.trade_filter`) — funciona enquanto o projeto root estiver no `sys.path`.

---

## 1.2 Imports Circulares

### Cadeia Crítica (latente)

```
ai_analyzer_qwen.py
  └─ from market_orchestrator.ai.llm_payload_guardrail import ...
  └─ from market_orchestrator.ai.ai_payload_builder import ...
  └─ from market_orchestrator.ai.payload_metrics_aggregator import ...

market_orchestrator/ai/ai_runner.py
  └─ from ai_analyzer_qwen import AIAnalyzer  ← circula de volta!
```

| Cadeia | Tipo | Ativo? | Risco |
|---|---|---|---|
| `ai_analyzer_qwen` → `market_orchestrator.ai.*` → `ai_runner` → `ai_analyzer_qwen` | Latente | Não (lazy init) | MÉDIO |
| `market_orchestrator.__init__` usa `__getattr__` lazy | Mitigação | Ativo | — |

**Por que não quebra**: `market_orchestrator/__init__.py` usa `__getattr__` lazy (só carrega `EnhancedMarketBot` quando pedido). `market_orchestrator/ai/__init__.py` está vazio (verificado). Portanto, importar `market_orchestrator.ai.ai_payload_builder` não carrega `ai_runner.py` automaticamente.

**Risco residual**: Se alguém importar `market_orchestrator.ai.ai_runner` antes de `ai_analyzer_qwen` estar totalmente carregado, haverá `ImportError` circular.

---

## 1.3 __init__.py dos Pacotes

| Pacote | `__init__.py` | `__all__` | Imports Problemáticos |
|---|---|---|---|
| `events/` | ✅ | ✅ | Nenhum |
| `trading/` | ✅ | ✅ | Nenhum |
| `fetchers/` | ✅ | ✅ | Nenhum |
| `market_analysis/` | ✅ | ✅ | Nota: `cross_asset_correlations` intencionalmente excluído para evitar circular |
| `data_processing/` | ✅ | ✅ (parcial) | `EnrichmentIntegrator` ausente do `__all__` |
| `monitoring/` | ✅ | ✅ | Nenhum |
| `common/` | ✅ | ✅ | Nenhum |
| `orderbook_core/` | ✅ | ✅ | Nenhum |
| `market_orchestrator/` | ✅ | ✅ | Usa `__getattr__` lazy — **correto** |
| `market_orchestrator/ai/` | ✅ (vazio) | ❌ | Sem exports — módulos importados diretamente |
| `ai_runner/` | ✅ | ❌ | Sem `__all__` |
| `flow_analyzer/` | ✅ | ❌ | Sem `__all__` |
| `config/` | ✅ | ❌ | Sem `__all__` |
| `utils/` | ✅ | ✅ | Usa imports absolutos (não relativos) — depende do sys.path |

---

## 1.4 Ordem de Inicialização (main.py)

| # | Módulo | Tipo | Risco |
|---|---|---|---|
| 1 | `sys`, `os`, `io` | stdlib | — |
| 2 | `dotenv.load_dotenv()` | externo, import-time side effect | BAIXO |
| 3 | `config` | via `config.py` → `from config.settings import *` | **WILDCARD IMPORT** — MÉDIO |
| 4 | `market_orchestrator.EnhancedMarketBot` | lazy via `__getattr__` | OK |
| 5 | `utils.HeartbeatManager` | proxy → `monitoring.heartbeat_manager` | OK |
| 6 | `prometheus_client.start_http_server` | dentro do `main()`, lazy | OK |
| 7 | `src.services.macro_update_service` | dentro do `main()`, lazy | OK |

**Problema identificado**: `config.py` usa `from config.settings import *`. Isso:
- Impossibilita análise estática (pylint, mypy não conseguem resolver nomes)
- Cria risco de colisão de nomes se `config/settings.py` for atualizado
- Impede IDEs de navegar para definições

**Variáveis de ambiente**: lidas via `load_dotenv()` no import-time. Valores de API key (GROQ, FRED, Binance) são lidos em runtime quando os clientes são instanciados.
