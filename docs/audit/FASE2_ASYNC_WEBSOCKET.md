# Fase 2 — Concorrência, Async e WebSocket

> Data: 2026-03-31 | Branch: audit/2026-03-31

---

## 2.1 Auditoria de Async e Concorrência

### Chamadas Bloqueantes em Contexto Async

Varredura de todos os módulos de produção (excluindo scripts/tests/legacy):

**Resultado: NENHUMA chamada `time.sleep()` ou `requests.get/post` encontrada dentro de funções `async def` em código de produção.**

### Padrões Problemáticos Encontrados

| Arquivo:Linha | Padrão | Tipo | Risco |
|---|---|---|---|
| `common/format_utils.py:55,83,115,...` | 13x `except:` (bare except) | Engole erros silenciosamente | MÉDIO |
| `data_pipeline/pipeline.py:474,777` | `except:` (bare except) | Engole erros em pipeline | MÉDIO |
| `data_processing/feature_store.py:70` | `except:` (bare except) | Falha silenciosa em store | MÉDIO |
| `events/event_saver.py:204,217` | `except:` (bare except) | Persistência falha silenciosamente | ALTO |
| `ai_analyzer_qwen.py` | 20+ `except Exception: pass` | Muitos pontos de falha silenciosa | ALTO |

### Estado Compartilhado

| Módulo | Estrutura | Lock? | Risco |
|---|---|---|---|
| `trading/trade_buffer.py` | `AsyncTradeBuffer` | `asyncio.Lock` presente | OK |
| `events/event_memory.py` | `deque` + dict | Verificar lock | MÉDIO |
| `events/event_bus.py` | `dict` de subscribers | `asyncio.Lock` a verificar | MÉDIO |
| `data_processing/feature_store.py` | Parquet append | Sem lock visível | MÉDIO |
| `market_analysis/levels_registry.py` | `dict` de níveis | Singleton sem lock explícito | MÉDIO |

---

## 2.2 WebSocket e Reconexão

### monitoring/websocket_handler.py

| Mecanismo | Status | Detalhes |
|---|---|---|
| Exponential backoff | ✅ Implementado | `initial_delay * (2 ** reconnect_count)` |
| Jitter | ✅ Implementado | `±20%` do delay calculado |
| Mínimo de delay | ✅ 0.5s | `max(0.5, delay + jitter)` |
| Limite de tentativas | ✅ 25 tentativas | `max_reconnect_attempts=25` |
| Ping/pong | ✅ Implementado | `ping_interval=20s`, `ping_timeout=10s` |
| Resposta a PING | ✅ | `await self._ws.pong(msg.data)` |
| Reset de estado | ✅ | `reconnect_count = 0` após sucesso |

### config/settings.py (orderbook WebSocket)

| Parâmetro | Valor | Avaliação |
|---|---|---|
| `ORDERBOOK_WS_RECONNECT_ATTEMPTS` | 15 | OK |
| `ORDERBOOK_WS_BACKOFF_FACTOR` | 1.5 | OK |
| `ORDERBOOK_WS_MAX_RECONNECT_DELAY` | 60s | OK |
| `ORDERBOOK_WS_PING_INTERVAL` | 20s | OK (Binance requer) |
| `ORDERBOOK_WS_HEARTBEAT_TIMEOUT` | 60s | OK |

### Gap Detection e Limpeza de Estado

| Verificação | Status | Risco |
|---|---|---|
| Gap detection de mensagens perdidas | ❓ Não encontrado explicitamente | MÉDIO |
| Limpeza de buffers após desconexão | ❓ A verificar em `robust_connection.py` | MÉDIO |
| Múltiplas instâncias WS concorrentes | `orderbook_ws_manager.py` existente | Verificar singleton | BAIXO |

### Avaliação Geral

WebSocket handler está bem implementado com todos os mecanismos básicos de resiliência. Principais riscos residuais são na detecção de gaps de mensagens e na garantia de limpeza de estado após reconexão.
