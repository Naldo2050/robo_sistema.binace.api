# Fases 7-8 — Performance, Memory Leaks e Estado Compartilhado

> Data: 2026-03-31 | Branch: audit/2026-03-31

---

## 7.1 Estruturas Sem Limite de Crescimento

### Deques sem maxlen

| Arquivo:Linha | Estrutura | Eviction Manual? | Risco |
|---|---|---|---|
| `flow_analyzer/aggregates.py:87` | `self.trades = deque()` | ✅ Sim — `_evict_if_needed()` via `MAX_AGGREGATE_TRADES` | BAIXO |
| `orderbook_analyzer.py:388` | `self._request_times_mono = deque()` | ❓ Verificar se tem trim | MÉDIO |
| `orderbook_analyzer/spread_tracker.py:39` | `self._history: deque = deque()` | ❓ Verificar limite | MÉDIO |

**Nota sobre `flow_analyzer/aggregates.py`**: Embora use `deque()` sem maxlen, o comentário documenta "Deque sem maxlen para controle manual" e existe `_evict_if_needed()` com `MAX_AGGREGATE_TRADES`. Padrão aceitável.

### Arquivos com Mais Inicializações de Listas/Dicts

| Arquivo | self.X = [] / {} | Risco |
|---|---|---|
| `orderbook_analyzer.py` | 5 | Verificar limites em runtime |
| `orderbook_core/orderbook.py` | 4 | Verificar flush/trim |
| `market_analysis/liquidity_heatmap.py` | 3 | Verificar eviction |
| `events/event_bus.py` | 2 | Subscriber list — crescimento limitado |
| `fetchers/context_collector.py` | 2 | Cache com TTL — OK |

### Caches

| Módulo | Cache | TTL | Limite | Status |
|---|---|---|---|---|
| `market_analysis/cross_asset_correlations.py` | Correlações | 5 min | 1 entrada | OK |
| `fetchers/context_collector.py` | Macro/FRED | 300s | — | OK |
| `market_orchestrator/ai/payload_section_cache.py` | Seções do payload | Configurável | — | Verificar |
| `fetchers/fred_fetcher.py` | `dados/fred_cache.json` | — | Arquivo em disco | BAIXO |

---

## 7.2 Hot Paths e Serialização

### Uso de orjson vs json

| Módulo | Usa orjson? | Observação |
|---|---|---|
| `events/event_saver.py` | Verificar | Crítico — persiste a cada evento |
| `flow_analyzer/serialization.py` | ✅ Decimal-safe JSON | Especializado |
| `build_compact_payload.py` | `json` padrão | Hot path — candidato a orjson |
| `ai_analyzer_qwen.py` | `json` padrão | Hot path de parsing |

### Logging em Hot Paths

`ai_analyzer_qwen.py` tem logging extensivo incluindo payloads completos (`logs/last_llm_payload.json` é salvo a cada análise). O arquivo pode crescer indefinidamente sem rotação.

---

## 8. Estado Compartilhado e Singletons

### Mapa de Singletons

| Singleton / Estado | Definido em | Acessado por | Sincronização | Risco |
|---|---|---|---|---|
| `EnhancedMarketBot` | `market_orchestrator/market_orchestrator.py` | `main.py` | — | OK — instância única |
| `ContextCollector` | `fetchers/context_collector.py` | orchestrator, ai_runner | `asyncio.Lock` (verificar) | MÉDIO |
| `EventMemory` | `events/event_memory.py` | orchestrator, event_saver | Verificar lock | MÉDIO |
| `FeatureStore` | `data_processing/feature_store.py` | pipeline, ai_runner | Verificar lock | MÉDIO |
| `LevelRegistry` | `market_analysis/levels_registry.py` | orchestrator, support_resistance | Sem lock aparente | MÉDIO |
| `SmartAIThrottler` | `common/ai_throttler.py` | `market_orchestrator/ai/ai_runner.py` | `asyncio.Lock` | OK |
| Clock sync | `monitoring/clock_sync.py` | time_manager | Singleton pattern | OK |

### Padrão de Init Múltiplo

`market_orchestrator/__init__.py` usa `__getattr__` lazy — carrega `EnhancedMarketBot` apenas na primeira referência. Correto e evita múltiplas instâncias.

### Offset de Clock

Offset de ~6.7s com Binance documentado em memória do projeto. `monitoring/clock_sync.py` gerencia isso com fallback. Não é bloqueante para operação mas pode causar timestamps incorretos em logs/eventos nas primeiras janelas.

---

## Itens para Verificação Manual

| Item | Por que verificar |
|---|---|
| `orderbook_analyzer.py:388` `_request_times_mono` sem maxlen | Rate limiter — pode crescer em alta frequência |
| `events/event_memory.py` — lock em acesso multi-task | Múltiplas tasks async podem escrever simultaneamente |
| `logs/last_llm_payload.json` — rotação | Arquivo sobrescrito a cada análise sem versioning |
| `logs/payload_metrics.jsonl` — tamanho | Cresce indefinidamente sem cleanup |
