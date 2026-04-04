# Fases 9-10 — Dependências, Docker e Dados

> Data: 2026-03-31 | Branch: audit/2026-03-31

---

## 9.1 Dependências Python

### requirements.txt — Análise

| Pacote | Versão | Observação |
|---|---|---|
| `requests>=2.31.0` | Flexível | OK — aiohttp também presente para async |
| `aiohttp>=3.9.0` | Flexível | OK |
| `numpy>=1.24.0` | Flexível | OK |
| `pandas>=2.0.0` | Flexível | Breaking changes entre major versions possível |
| `xgboost>=2.0.0` | Flexível | OK |
| `openai>=1.0.0` | Flexível | API client para Groq |
| `dashscope>=1.17.0` | Flexível | Alibaba Cloud / Qwen — usado como fallback opcional |
| `tiktoken>=0.5.0` | Opcional (comentado) | "Optional: for accurate token counting" |
| `yfinance>=0.2.43` | Flexível | OK |
| `fredapi>=0.5.0` | Flexível | OK |
| `oci>=2.115.0` | Flexível | Oracle Cloud — pesado para deploy |
| `playwright>=1.40.0` | Flexível | "Helper/Legacy optional" mas em requirements.txt principal |
| `pytest>=7.0.0` | Em requirements.txt! | **PROBLEMA**: pytest deveria estar só em requirements-dev.txt |
| `opentelemetry-api/sdk>=1.27.0` | Flexível | OK para observabilidade |

### Problemas Encontrados

| Problema | Severidade | Detalhes |
|---|---|---|
| `pytest` em `requirements.txt` | MÉDIO | Framework de teste não deve ir para produção/Docker |
| `playwright` em `requirements.txt` | MÉDIO | Browser automation não é necessário em produção |
| `oci>=2.115.0` em produção | BAIXO | Se OCI desabilitado (config `OCI_COMPARTMENT_ID = None`), o pacote é desnecessário em deploy sem Oracle |
| Binance packages ausentes | **ALTO** | `ESTRUTURA_SISTEMA_COMPLETO.md` cita `binance-connector` e `python-binance` mas **nenhum está no `requirements.txt`** |
| `requirements.txt` vs `pyproject.toml` | MÉDIO | `pyproject.toml` tem `[tool.pytest.ini_options]` mas sem `[project.dependencies]` — não é fonte de truth das deps |

### requirements-dev.txt

```
pytest-cov>=4.1.0
mypy>=1.10.0
pytest-timeout>=2.4.0
```

Mínimo necessário. Falta: `pylint`, `bandit`, `radon`, `black`/`ruff` para lint e segurança.

---

## 9.2 Docker e Deploy

### Bugs Críticos no Dockerfile

| Linha | Problema | Severidade |
|---|---|---|
| 10 | `max_user_watches=524288` dentro de `apt-get install` | **CRÍTICO** — Build quebra |
| 26 | `mkdir -p data logs features` — não cria `dados/` | **ALTO** — SQLite/JSONL sem diretório |
| HEALTHCHECK | Verifica `/app/logs/health_status` (não `/app/logs/health_status.txt` ou similar) | MÉDIO |

**Dockerfile linha 10 (reprodução do bug)**:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    max_user_watches=524288 \   # ← BUG: sysctl parameter, não pacote apt!
    python3-dev \
    curl \
    git \
```
Esse parâmetro deveria estar em `/etc/sysctl.conf` ou em `docker-compose.yml` via `sysctls:`, não em `apt-get`.

### docker-compose.yml

| Verificação | Status |
|---|---|
| Health checks configurados | ❌ Sem `healthcheck:` no serviço |
| Volume `dados/` mapeado | ❌ Monta `./data` ao invés de `./dados` |
| Prometheus porta exposta | ❌ Porta 8000 comentada no Dockerfile |
| Secrets via env_file | ✅ `env_file: .env` |
| Restart policy | ✅ `unless-stopped` |
| Log rotation | ✅ `max-size: 10m, max-file: 5` |

### .dockerignore

**Itens que deveriam ser adicionados**:
```
dados/
memory/
backups/
fallback_events/
features/
coverage_html/
.env
*.env
```

---

## 10. Consistência de Dados e Logs

### Fontes de Dados

| Fonte | Formato | Escritores | Leitores | Status |
|---|---|---|---|---|
| `dados/eventos_fluxo.jsonl` | JSONL | `events/event_saver.py` | scripts de análise | OK — JSONL guardian presente |
| `dados/trading_bot.db` | SQLite | `database/event_store.py`, `events/event_saver.py` | múltiplos | Verificar schema migrations |
| `logs/last_llm_payload.json` | JSON | `ai_analyzer_qwen.py:3792` | scripts de diagnóstico | Sem rotação — sobrescreve |
| `logs/payload_metrics.jsonl` | JSONL | `market_orchestrator/ai/payload_metrics_aggregator.py` | `scripts/analyze_ai_usage.py` | Cresce indefinidamente |

### Logging

| Módulo | Formato | Rotação | Observação |
|---|---|---|---|
| `common/logging_config.py` | Texto + JSON opcional | ✅ RotatingFileHandler | OK |
| `main.py` | Texto | ✅ RotatingFileHandler `issues.log` (5MB×3) | OK |
| `ai_analyzer_qwen.py` | `last_llm_payload.json` | ❌ Sem rotação | Sobrescreve cada análise |
| `market_orchestrator/ai/payload_section_cache.json` | JSON | ❌ Sem rotação | Cache persistente |

### Schema Migrations (SQLite)

Não foram encontrados arquivos de migration (Alembic, SQL scripts) para `dados/trading_bot.db`. Mudanças de schema precisam ser feitas manualmente.
