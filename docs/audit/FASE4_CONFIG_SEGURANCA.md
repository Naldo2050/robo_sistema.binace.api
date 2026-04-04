# Fase 4 — Configuração, Ambiente e Segurança

> Data: 2026-03-31 | Branch: audit/2026-03-31

---

## 4.1 Validação de Configuração

### Problema Crítico: Wildcard Import

| Arquivo | Linha | Problema | Severidade |
|---|---|---|---|
| `config.py` | 2 | `from config.settings import *` | MÉDIO |

**Impacto**: Impossibilita análise estática (mypy, pylint não resolvem nomes), IDE não navega para definições, nomes do módulo podem colidir com vars locais, qualquer adição ao `config/settings.py` vaza automaticamente para o namespace global.

### Configurações Duplicadas / Conflitantes

| Variável | config/settings.py | config.json | Status |
|---|---|---|---|
| `config.json` existe na raiz | — | Arquivo presente | Não conflita com `config/settings.py` — propósito diferente (auto_fixer?) |

**Verificar**: `config.json` na raiz não é o `config/settings.py`. Verificar se algum módulo ainda lê `config.json` diretamente. Encontrado: usado por `auto_fixer/`.

### Valores Padrão e Timeouts

| Parâmetro | Valor | Avaliação |
|---|---|---|
| `ORDERBOOK_REQUEST_TIMEOUT` | 10.0s | OK |
| `ORDERBOOK_REST_MAX_RETRIES` | 5 | OK |
| `CONTEXT_UPDATE_INTERVAL_SECONDS` | 300s (5min) | OK |
| `ORDERBOOK_WS_MAX_RECONNECT_DELAY` | 60s | OK |

Nenhum timeout=0 ou retry=infinito encontrado.

---

## 4.2 Segurança

### Avaliação de Secrets

| Verificação | Status | Detalhes |
|---|---|---|
| API keys em código-fonte | ✅ OK | Apenas mock keys (`"mock_key"`) em tests |
| `.env` no `.gitignore` | ✅ OK | `.env` listado no `.gitignore` |
| Secrets em `.env.example` | ✅ OK | Apenas placeholders (`your_*_here`) |
| `eval()` em produção | ✅ OK | `eval()` só em `tests/unit/test_patch_validator.py:49` |
| `exec()` em produção | ✅ OK | Não encontrado |
| `subprocess shell=True` | ✅ OK | Não encontrado em produção |

### Potenciais Falso Positivos

| Arquivo | Linha | Conteúdo | Avaliação |
|---|---|---|---|
| `ai_runner/ai_runner.py` | 175 | `api_key="mock_key"` | Falso positivo — mock para testes |
| `tests/integration/test_ai_runner.py` | 414–642 | Múltiplos padrões | Fixtures de teste com valores fixos |

### eval() em Testes

```python
# tests/unit/test_patch_validator.py:49 — RISCO BAIXO (só em testes)
"entry_zone": eval(zone_input) if zone_input != 'null' else None
```

Embora em testes, `eval()` com input não sanitizado é uma prática ruim. Substituir por `json.loads()`.

---

## 4.3 Caminhos Hardcoded

### Resumo (40 ocorrências em 26 arquivos)

| Padrão | Contagem | Arquivos Críticos |
|---|---|---|
| `dados/` | 22 | `database/event_store.py`, `events/event_saver.py`, `events/event_similarity.py`, `trading/outcome_tracker.py` |
| `logs/` | 5 | `ai_analyzer_qwen.py`, `common/logging_config.py`, `main.py`, `market_orchestrator/ai/` |
| `ml/models/` | 4 | `diagnostics/`, `ml/dataset_collector.py` |
| `ml/datasets/` | 1 | `ml/dataset_collector.py` |
| `features/` | 0 | — |
| `memory/` | 0 | — |

### Problemas Críticos de Docker

| Arquivo:Linha | Problema | Severidade |
|---|---|---|
| `Dockerfile:10` | `max_user_watches=524288` listado como pacote apt-get | **CRÍTICO** — Build Docker QUEBRA |
| `docker-compose.yml` volumes | Monta `./data:/app/data` mas código usa `dados/` | **ALTO** — Dados perdidos em produção Docker |
| `Dockerfile` | `mkdir -p data logs features` — não cria `dados/` | **ALTO** — SQLite e JSONL não terão pasta |
| `Dockerfile HEALTHCHECK` | Verifica `/app/logs/health_status` | MÉDIO — pode não existir |
| `.dockerignore` | Não lista `dados/`, `memory/`, `backups/`, `.env` | MÉDIO — `.env` real não está no .gitignore do Docker |

**Detalhe do bug no Dockerfile**:
```dockerfile
# LINHA QUEBRADA — 'max_user_watches=524288' não é pacote apt
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    max_user_watches=524288 \   ← ERRO: isto não é um pacote!
    python3-dev \
```
Isso causa falha no `docker build` com erro `Unable to locate package max_user_watches=524288`.

**Mismatch de volume**:
```yaml
# docker-compose.yml — monta ./data mas código usa ./dados
volumes:
  - ./data:/app/data    ← código usa 'dados/', não 'data/'
  - ./logs:/app/logs
```
Corrigir para `./dados:/app/dados` ou padronizar os caminhos no código.
