# RELATÓRIO FINAL DE AUDITORIA — Robô Binance BTCUSDT

> **Data**: 2026-03-31  
> **Branch**: `audit/2026-03-31`  
> **Auditor**: Claude Sonnet 4.6 (modo read-only)  
> **Escopo**: 10 fases, ~250 arquivos Python, todos os pacotes

---

## 1. RESUMO EXECUTIVO

| Severidade | Contagem |
|---|---|
| 🔴 CRÍTICO | 4 |
| 🟠 ALTO | 12 |
| 🟡 MÉDIO | 18 |
| 🟢 BAIXO | 8 |
| **TOTAL** | **42** |

**Estado geral do sistema**: Funcional em desenvolvimento local. **NÃO apto para deploy Docker sem correções**. Testes em degradação progressiva (131 falhas/erros de 1088 testes).

---

## 2. TOP 10 ISSUES — PRIORIDADE MÁXIMA

| # | ID | Arquivo | Problema | Severidade | Esforço |
|---|---|---|---|---|---|
| 1 | D-01 | `Dockerfile:10` | `max_user_watches=524288` listado como pacote `apt-get` — **build Docker quebra** | 🔴 CRÍTICO | 2min |
| 2 | D-02 | `docker-compose.yml` | Monta `./data:/app/data` mas código usa `dados/` — **dados perdidos em produção** | 🔴 CRÍTICO | 5min |
| 3 | T-01 | `tests/unit/test_simple_correlations.py:38` | `sys.exit(1)` no escopo de módulo — **quebra pytest com INTERNALERROR** | 🔴 CRÍTICO | 2min |
| 4 | T-02 | `tests/payload/conftest.py:28` | `get_cross_asset_features` ausente em `ai_payload_builder` — **73 testes de payload bloqueados** | 🔴 CRÍTICO | 30min |
| 5 | L-01 | `requirements.txt` | `binance-connector` e `python-binance` ausentes — **bot não conecta à Binance sem eles** | 🟠 ALTO | 5min |
| 6 | T-03 | `tests/integration/test_orderbook_*.py` | `make_valid_snapshot` ausente em `tests/conftest.py` — 2 testes bloqueados | 🟠 ALTO | 15min |
| 7 | B-01 | `flow_analyzer/absorption.py` (ou similar) | `AbsorptionZoneMapper.record_event()` não persiste eventos — 3 testes falham com 0 zonas | 🟠 ALTO | 1h |
| 8 | E-01 | `events/event_saver.py:204,217` | `except:` (bare except) — persistência de eventos falha silenciosamente | 🟠 ALTO | 30min |
| 9 | C-01 | `orderbook_core/circuit_breaker.py` | Circuit breaker sem métricas Prometheus — falhas invisíveis em produção | 🟠 ALTO | 2h |
| 10 | D-03 | `Dockerfile` | `mkdir -p data logs features` sem `dados/` — SQLite e JSONL sem diretório em Docker | 🟠 ALTO | 2min |

---

## 3. IMPORTS E PROXIES

### Status

Todos os 29 arquivos proxy foram removidos da raiz. **Impacto: zero** — todos os pacotes migrados usam imports relativos e funcionam independentemente.

| Verificação | Status |
|---|---|
| Proxy files existem | ❌ Todos removidos (29/29) |
| Imports absolutos quebrados | ✅ Nenhum encontrado em produção |
| Pacotes destino OK | ✅ Todos os 29 destinos existem |
| Circular import ai_analyzer_qwen ↔ ai_runner | ⚠️ Latente (não ativa na init) |

### Circular Import Latente

```
ai_analyzer_qwen.py → market_orchestrator.ai.llm_payload_guardrail
                     → market_orchestrator.ai.ai_payload_builder
                     → market_orchestrator.ai.payload_metrics_aggregator
market_orchestrator/ai/ai_runner.py → from ai_analyzer_qwen import AIAnalyzer
```

Não é ativa porque `market_orchestrator/__init__.py` usa lazy `__getattr__`. **Risco**: qualquer mudança que force import eagerly pode quebrar.

### config.py — Wildcard Import

```python
# config.py
from config.settings import *  # noqa: F401,F403
```
Impossibilita análise estática completa. Mypy e pylint não conseguem resolver nomes de configuração.

---

## 4. CONCORRÊNCIA

| Verificação | Status | Risco |
|---|---|---|
| `time.sleep` em async | ✅ Não encontrado | — |
| `requests.get` em async | ✅ Não encontrado | — |
| WebSocket backoff exponencial | ✅ Implementado | — |
| WebSocket jitter | ✅ ±20% | — |
| WebSocket heartbeat/ping-pong | ✅ 20s ping, pong manual | — |
| Limite máximo de reconexões | ✅ 25 tentativas | — |
| `bare except:` em formato utils | ⚠️ 13 instâncias | MÉDIO |
| `except Exception: pass` em ai_analyzer | ⚠️ 20+ instâncias | MÉDIO |

---

## 5. SEGURANÇA

| Verificação | Status | Risco |
|---|---|---|
| API keys em código-fonte | ✅ OK — apenas mocks | — |
| `.env` no `.gitignore` | ✅ OK | — |
| `eval()` em produção | ✅ OK | — |
| `eval()` em testes | ⚠️ `test_patch_validator.py:49` | BAIXO |
| Subprocess shell injection | ✅ OK | — |
| Prompt injection (dados → LLM) | ❓ Guardrail presente mas não auditado o schema | MÉDIO |
| `.dockerignore` sem `dados/`, `memory/` | ⚠️ Dados locais podem entrar na imagem | MÉDIO |

---

## 6. TESTES

| Camada | Total | Passou | Falhou | Erros | % Sucesso |
|---|---|---|---|---|---|
| Unit | 640 | 633 | 7 | 1 (INTERNALERROR) | 98.9% |
| Integration | 375 | 320 | 51 | 4 | 85.3% |
| Payload | 73 | 0 | 0 | 73 (collection error) | **0%** |
| **TOTAL** | **1088** | **953** | **58** | **78** | **87.6%** |

### Causa Raiz das Falhas por Categoria

| Categoria | Impacto | Causa |
|---|---|---|
| `sys.exit(1)` no test_simple_correlations | INTERNALERROR no pytest | Script standalone na pasta de testes |
| `get_cross_asset_features` ausente | 73 payload tests bloqueados | Refatoração sem atualizar conftest |
| `make_valid_snapshot` ausente | 2 integration tests bloqueados | Fixture removida do conftest |
| `AbsorptionZoneMapper` bug | 3 unit tests falham | Bug real no código de produção |
| `LogSanitizer` desatualizado | 3 unit tests falham | Implementação mudou |
| System prompt sem "portugues do brasil" | 1 unit test falha | Teste desatualizado |
| Schema de orderbook mudou | ~20 integration tests falham | API evoluiu sem atualizar testes |
| RiskManager API mudou | 5 integration tests falham | Interface evoluiu |

---

## 7. AI/ML PIPELINE

| Verificação | Status | Fase corrigida |
|---|---|---|
| FEATURE_MAP Bollinger Bands | ✅ Correto (v3) | 2026-03-13 |
| RSI fallback 50.0 | ✅ Implementado | 2026-03-13 |
| System prompt 8B vs 70B | ✅ Corrigido | 2026-03-11 |
| Default model llama-3.1-8b-instant | ✅ Aplicado | 2026-03-11 |
| Macro data VIX real | ✅ Implementado | 2026-03-11 |
| Throttler integrado | ✅ no path principal | 2026-03-13 |
| Guardrail integrado | ✅ no path principal | 2026-03-11 |
| `get_cross_asset_features` em conftest | ❌ Ausente | Pendente |
| Payload com flow/whale/ob ausentes | ⚠️ Warning no compressor v3 | Pendente investigação |

---

## 8. PERFORMANCE

| Issue | Arquivo | Risco |
|---|---|---|
| `deque()` sem maxlen em `orderbook_analyzer.py:388` | `_request_times_mono` | MÉDIO — pode crescer em alta freq |
| `deque()` sem maxlen em `spread_tracker.py:39` | `_history` | MÉDIO |
| `logs/payload_metrics.jsonl` sem rotação | `payload_metrics_aggregator.py` | MÉDIO — cresce indefinidamente |
| `json` padrão em `build_compact_payload.py` | hot path | BAIXO — candidato a orjson |
| `flow_analyzer/aggregates.py` deque com eviction manual | Controlado | OK |

---

## 9. CONFIGURAÇÃO E INFRAESTRUTURA

| Issue | Arquivo | Severidade |
|---|---|---|
| `max_user_watches` no apt-get | `Dockerfile:10` | 🔴 CRÍTICO |
| Volume `./data` vs `dados/` | `docker-compose.yml` | 🔴 CRÍTICO |
| `mkdir dados/` ausente | `Dockerfile:26` | 🟠 ALTO |
| `pytest` e `playwright` em requirements.txt | `requirements.txt` | 🟡 MÉDIO |
| `binance-connector`/`python-binance` ausentes | `requirements.txt` | 🟠 ALTO |
| Config wildcard import | `config.py` | 🟡 MÉDIO |
| Schema migrations SQLite ausentes | `database/` | 🟡 MÉDIO |
| `last_llm_payload.json` sem rotação | `ai_analyzer_qwen.py:3792` | 🟡 MÉDIO |

---

## 10. PLANO DE AÇÃO — Ordenado por Prioridade

### Imediato (< 1h, sem risco)

| Prioridade | Issue | Ação |
|---|---|---|
| 1 | D-01 | Remover `max_user_watches=524288` do `apt-get install` no Dockerfile |
| 2 | D-02 | Corrigir volume `./data` → `./dados` no `docker-compose.yml` |
| 3 | D-03 | Adicionar `mkdir -p dados` no Dockerfile |
| 4 | T-01 | Mover `tests/unit/test_simple_correlations.py` → `scripts/` ou remover `sys.exit(1)` |
| 5 | L-01 | Adicionar `binance-connector` e/ou `python-binance` ao `requirements.txt` |

### Alta Prioridade (1h–1 dia)

| Prioridade | Issue | Ação |
|---|---|---|
| 6 | T-02 | Corrigir `tests/payload/conftest.py:28` — atualizar `get_cross_asset_features` para nome correto |
| 7 | T-03 | Adicionar `make_valid_snapshot` ao `tests/conftest.py` |
| 8 | B-01 | Investigar e corrigir `AbsorptionZoneMapper.record_event()` |
| 9 | E-01 | Trocar `except:` por `except Exception as e: logger.warning(...)` em `event_saver.py` |
| 10 | L-01 | Mover `pytest` e `playwright` para `requirements-dev.txt` |

### Média Prioridade (1 dia–1 semana)

| Prioridade | Issue | Ação |
|---|---|---|
| 11 | C-01 | Adicionar counters Prometheus ao `orderbook_core/circuit_breaker.py` |
| 12 | T-04 | Atualizar `test_ai_analyzer_language_and_think_strip.py` e `TestLogSanitizer` |
| 13 | D-04 | Atualizar `.dockerignore` com `dados/`, `memory/`, `backups/`, `fallback_events/` |
| 14 | M-01 | Adicionar `maxlen` ao `_request_times_mono` em `orderbook_analyzer.py` |
| 15 | M-02 | Adicionar rotação ao `logs/payload_metrics.jsonl` |

### Baixa Prioridade (backlog)

| Prioridade | Issue | Ação |
|---|---|---|
| 16 | C-02 | Substituir `from config.settings import *` por imports explícitos |
| 17 | T-05 | Atualizar ~35 integration tests com schema obsoleto |
| 18 | SEC-01 | Substituir `eval()` por `json.loads()` em `test_patch_validator.py` |
| 19 | PERF-01 | Migrar `build_compact_payload.py` de `json` para `orjson` |
| 20 | DOC-01 | Atualizar `ESTRUTURA_SISTEMA_COMPLETO.md` — proxies não existem mais |

---

## Arquivos de Relatório Detalhado

| Fase | Arquivo |
|---|---|
| 1 — Imports/Proxies | [FASE1_IMPORTS_PROXIES.md](FASE1_IMPORTS_PROXIES.md) |
| 2 — Async/WebSocket | [FASE2_ASYNC_WEBSOCKET.md](FASE2_ASYNC_WEBSOCKET.md) |
| 3 — Erros/Resiliência | [FASE3_ERROS_RESILIENCIA.md](FASE3_ERROS_RESILIENCIA.md) |
| 4 — Config/Segurança | [FASE4_CONFIG_SEGURANCA.md](FASE4_CONFIG_SEGURANCA.md) |
| 5 — AI/ML Pipeline | [FASE5_AI_ML_PIPELINE.md](FASE5_AI_ML_PIPELINE.md) |
| 6 — Testes | [FASE6_TESTES.md](FASE6_TESTES.md) |
| 7-8 — Performance/Estado | [FASE7_8_PERFORMANCE_ESTADO.md](FASE7_8_PERFORMANCE_ESTADO.md) |
| 9-10 — Deps/Docker/Dados | [FASE9_10_DEPS_DOCKER_DADOS.md](FASE9_10_DEPS_DOCKER_DADOS.md) |
