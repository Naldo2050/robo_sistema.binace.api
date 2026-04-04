# Fase 3 — Tratamento de Erros e Resiliência

> Data: 2026-03-31 | Branch: audit/2026-03-31

---

## 3.1 Exception Handling

### Estatísticas Gerais (código de produção, excluindo legacy/scripts/tests)

| Padrão | Contagem | Risco |
|---|---|---|
| `except:` (bare except) | 43 | MÉDIO–ALTO |
| `except Exception: pass` | 166 | MÉDIO–ALTO |

### Concentração dos Problemas

| Arquivo | `except: pass` | `bare except` | Avaliação |
|---|---|---|---|
| `ai_analyzer_qwen.py` | 20+ | 0 | Alto — módulo crítico de IA com muitos pontos de falha silenciosa |
| `common/format_utils.py` | 0 | 13 | Médio — formatação falha silenciosamente, retorna valor default |
| `data_pipeline/pipeline.py` | ? | 2 | Alto — pipeline de dados crítico |
| `events/event_saver.py` | ? | 2 | Alto — persistência de eventos falha silenciosamente |
| `data_processing/feature_store.py` | ? | 1 | Alto — store de features crítico |
| `market_orchestrator/market_orchestrator.py` | múltiplos | 0 | ALTO — orquestrador principal |

### Exemplos Críticos

```python
# ai_analyzer_qwen.py:256 — falha silenciosa na inicialização de componente
except Exception:
    pass

# events/event_saver.py:204 — persistência falha sem log
except:
    pass  # Evento perdido silenciosamente

# data_pipeline/pipeline.py:474 — pipeline completo swallowed
except:
    pass
```

### Recomendação

Os `except Exception: pass` em `ai_analyzer_qwen.py` são aceitáveis para fallbacks de importação opcional (bibliotecas como `dashscope`, `jinja2`). Porém, os presentes em lógica de negócio (análise, construção de payload) precisam logar o erro mesmo sem re-raise.

---

## 3.2 Circuit Breakers e Fallbacks

### Circuit Breakers Existentes

| Componente | Arquivo | Half-open? | Prometheus? | Jitter? | Avaliação |
|---|---|---|---|---|---|
| OrderBook CB | `orderbook_core/circuit_breaker.py` | ✅ Sim | ❌ Não | — | MÉDIO — sem métricas expostas |
| Flow Analyzer CB | `flow_analyzer/metrics.py` | ✅ Sim | ✅ Sim | — | OK |
| OrderBook REST Fallback | `orderbook_core/orderbook_fallback.py` | — | — | ✅ (via config) | OK |

**Detalhe crítico**: `orderbook_core/circuit_breaker.py` NÃO expõe métricas Prometheus para transições de estado (OPEN/CLOSED/HALF_OPEN). Isso significa que um circuit breaker aberto em produção pode passar despercebido até que alguém note o impacto no trading.

### Configuração de Circuit Breaker (config/settings.py)

```
ORDERBOOK_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
ORDERBOOK_CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2  
ORDERBOOK_CIRCUIT_BREAKER_TIMEOUT_SECONDS = 30.0
ORDERBOOK_REST_RETRY_BACKOFF = 2.0
ORDERBOOK_REST_JITTER_RANGE = 0.25  # 25% jitter ✅
ORDERBOOK_REST_MAX_RETRIES = 5
```

### Compressores de Payload (v1 vs v3)

| Compressor | Arquivo | Usado em |
|---|---|---|
| v1 (`payload_compressor.py`) | `market_orchestrator/ai/` | Importado por `ai_payload_builder.py` |
| v3 (`payload_compressor_v3.py`) | `market_orchestrator/ai/` | Usado em testes; path alternativo |
| `build_compact_payload.py` | raiz | Usado pelo `ai_runner.py` (path principal) |

**Ambos v1 e v3 estão ativos**. O path principal usa `build_compact_payload` (raiz) + v1 via `ai_payload_builder`. V3 é usado em tests.

### Fallback LLM

`ai_runner/ai_runner.py` (legado) tenta `qwen_client → MockQwenClient` como fallback. O path principal (`market_orchestrator/ai/ai_runner.py`) usa throttler + guardrail mas não tem fallback para modelo menor explicitamente documentado.
