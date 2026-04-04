# Fase 6 — Testes e Cobertura

> Data: 2026-03-31 | Branch: audit/2026-03-31

---

## 6.2 Resultados dos Testes (Smoke Test)

### Resumo por Camada

| Camada | Total | Passou | Falhou | Erros | % Sucesso |
|---|---|---|---|---|---|
| **Unit** | 640 | 633 | 7 | 1 (INTERNALERROR) | 98.9% |
| **Integration** | 375 | 320 | 51 | 4 | 85.3% |
| **Payload** | 73 | 0 | 0 | 73 | 0% |
| **E2E** | — | — | — | — | Não rodado |

---

## Falhas de Unit Tests (7 falhas)

| Teste | Erro | Causa |
|---|---|---|
| `test_absorption_zone_mapper_single_zone` | `'no_events' != 'success'` | `record_event()` não está persistindo eventos — bug lógico |
| `test_absorption_zone_mapper_multiple_events_same_zone` | `0 == 1` (total_zones) | Mesma causa |
| `test_absorption_zone_mapper_multiple_zones` | `0 == 2` (total_zones) | Mesma causa |
| `test_system_prompt_requires_ptbr_and_no_think` | `'portugues do brasil' not in prompt` | System prompt não contém a frase esperada — teste desatualizado |
| `TestLogSanitizer::test_env_var_redaction` | Falha na redação de env vars | `LogSanitizer` não redige vars de ambiente |
| `TestLogSanitizer::test_groq_token_redaction` | Falha na redação de token | Padrão de redação desatualizado |
| `TestLogSanitizer::test_partial_key_redaction` | Falha na redação parcial | Padrão de redação desatualizado |

## INTERNALERROR (1 arquivo quebrado)

| Arquivo | Linha | Problema | Severidade |
|---|---|---|---|
| `tests/unit/test_simple_correlations.py` | 38 | `sys.exit(1)` no escopo de módulo | **ALTO** — quebra o runner pytest inteiro com INTERNALERROR |

**Esse arquivo não é um teste pytest válido** — é um script standalone que foi colocado na pasta de testes. Precisa ser movido para `scripts/` ou ter o `sys.exit(1)` removido.

---

## Erros de Importação em Integration Tests (4 erros)

| Arquivo | Linha | Erro | Causa |
|---|---|---|---|
| `tests/integration/test_orderbook_analyze_core.py:9` | — | `cannot import name 'make_valid_snapshot' from 'tests.conftest'` | Fixture removida do conftest sem atualizar os testes |
| `tests/integration/test_orderbook_config_injection.py:10` | — | Mesmo erro | Mesma causa |
| `tests/integration/test_window_processor.py` | — | 4 errors collecting | A verificar (running em background) |

---

## Falhas de Integration Tests (51 falhas — selecionadas)

| Grupo | Falhas | Causa Provável |
|---|---|---|
| `test_orderbook_analyzer_full_coverage.py` | 2 | Schema/comportamento mudou |
| `test_orderbook_analyzer_missing.py` | 1 | Symbol inválido não tratado |
| `test_orderbook_core_comprehensive.py::test_circuit_breaker_integration` | 1 | CircuitBreaker API mudou |
| `test_patch_2_fallback_controlado.py` | 5 | Fallback Groq→OpenAI lógica mudou |
| `test_risk_manager_comprehensive.py` | 5 | Interface RiskManager mudou |
| `test_trade_buffer_optimization.py::test_async_trade_buffer` | 1 | AsyncTradeBuffer API mudou |

---

## Erros de Payload Tests (73 erros — TODOS)

**Causa única**: `tests/payload/conftest.py:28` tenta `monkeypatch.setattr(builder, "get_cross_asset_features", ...)` mas `market_orchestrator.ai.ai_payload_builder` não tem esse atributo.

Todos os 73 testes de payload falham antes de executar. Corrigir o `conftest.py` para usar o nome correto (ou adicionar o atributo ao módulo) resolve todos.

---

## Módulos Críticos sem Cobertura Adequada

| Módulo | Cobertura Estimada | Testes Existentes | Risco |
|---|---|---|---|
| `trading/trade_buffer.py` | ~14% | 1 integration (falhando) | ALTO |
| `trading/alert_engine.py` | ~5% | Indireto via e2e | MÉDIO |
| `trading/outcome_tracker.py` | 0% | Nenhum | ALTO |
| `market_orchestrator/ai/ai_runner.py` | Indireto | Via integration | MÉDIO |
| `monitoring/websocket_handler.py` | Baixa | Nenhum unit direto | MÉDIO |
| `ml/hybrid_decision.py` | ? | `test_ml_frozen_detector.py` | MÉDIO |

---

## Resumo de Riscos por Tipo de Falha

| Tipo | Contagem | Prioridade de Correção |
|---|---|---|
| `sys.exit(1)` em arquivo de teste | 1 | IMEDIATA — quebra CI |
| Fixture ausente (`make_valid_snapshot`) | 2 | ALTA |
| Atributo ausente em conftest (`get_cross_asset_features`) | 1 (bloqueia 73 testes) | ALTA |
| Testes desatualizados (sistema prompt, LogSanitizer) | 4 | MÉDIA |
| Bug lógico real (`AbsorptionZoneMapper`) | 3 | ALTA — bug em produção |
| Testes obsoletos (schema mudou) | ~35 | MÉDIA |
