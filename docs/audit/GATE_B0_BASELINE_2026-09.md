# Gate B0: Baseline de Testes e Higiene do Repositório

- **Data (UTC):** 2026-09-20
- **Branch:** `main`
- **Ambiente:** Python 3.12.8, pytest 9.0.1 (Windows)

---

## 1. Tabela de Categorização das Pendências (Passo 1)

| # | Arquivo | Categoria | Destino no Gate B0 |
|---|---|---|---|
| 1 | `ESTRUTURA_SISTEMA_COMPLETO.md` | (a) Documentação/Auditoria | Commit `chore(audit)` |
| 2 | `dados/audit/klines_accept_futures.json` | (a) Documentação/Auditoria | Commit `chore(audit)` |
| 3 | `docs/audit/FASE1_DEPRECATION_MECHANISM_FAILURE_2026-09.md` | (a) Documentação/Auditoria | Commit `chore(audit)` |
| 4 | `docs/audit/ORDERBOOK_LAG_EMPIRICAL_RECONCILIATION_2026-09.md` | (a) Documentação/Auditoria | Commit `chore(audit)` |
| 5 | `docs/audit/PRE_PAPER_TRADING_READINESS_2026-09.md` | (a) Documentação/Auditoria | Commit `chore(audit)` |
| 6 | `docs/audit/SIGNOFF_AUDITORIA_FUTURES_2026-09.md` | (a) Documentação/Auditoria | Commit `chore(audit)` |
| 7 | `query` | (c) Artefato Descartável | Adicionado ao `.gitignore` |
| 8 | `logs/run.log.1` | (c) Artefato Descartável | Adicionado ao `.gitignore` |
| 9 | `scripts/diagnostics/collect_2h.ps1` | (b) Scripts/Testes | Commit `chore(audit)` |
| 10 | `tests/integration/test_ml_stale_real_event_pipeline.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 11 | `scripts/analytics/research_backup.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 12 | `scripts/analytics/research_local_watchdog.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 13 | `scripts/diagnostics/analyze_orderbook_sync_session.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 14 | `scripts/diagnostics/audit_c1_contamination.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 15 | `scripts/diagnostics/audit_latency_spikes.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 16 | `scripts/diagnostics/audit_val2.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 17 | `scripts/diagnostics/benchmark_dump_raw_trades_overhead.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 18 | `scripts/diagnostics/check_r4_citations.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 19 | `scripts/diagnostics/check_validations_data.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 20 | `scripts/diagnostics/decompose_london_ny_trades.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 21 | `scripts/diagnostics/detailed_volume_spike_audit.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 22 | `scripts/diagnostics/find_recent_files.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 23 | `scripts/diagnostics/generate_r4b_markdown.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 24 | `scripts/diagnostics/inspect_121201_log.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 25 | `scripts/diagnostics/inspect_all_dbs.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 26 | `scripts/diagnostics/inspect_latency_context.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 27 | `scripts/diagnostics/inspect_run_log.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 28 | `scripts/diagnostics/inspect_schema_real.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 29 | `scripts/diagnostics/inspect_snapshot_moment_trades.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 30 | `scripts/diagnostics/inspect_window97.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 31 | `scripts/diagnostics/investigate_asia_event.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 32 | `scripts/diagnostics/multi_day_stratified_audit.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 33 | `scripts/diagnostics/run_3_validation_scripts.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 34 | `scripts/diagnostics/run_all_validations.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 35 | `scripts/diagnostics/run_r4b_full.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 36 | `scripts/diagnostics/stratified_trade_sampling.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 37 | `scripts/diagnostics/val0_integrity.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 38 | `scripts/diagnostics/val2_check.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 39 | `scripts/diagnostics/validate_memory_and_latency_health.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 40 | `scripts/diagnostics/validate_ml_stale_neutralization.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 41 | `scripts/diagnostics/validate_signals_orderbook_schema.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 42 | `scripts/diagnostics/validate_volume_spike_dual_gate.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 43 | `scripts/diagnostics/validate_whale_threshold.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 44 | `tests/unit/test_dump_raw_trades.py` | (b) Scripts/Testes | Commit `chore(audit)` |
| 45 | `tests/unit/test_ops_data_bridge.py` | (b) Scripts/Testes | Commit `chore(audit)` |

---

## 2. Hashes dos Commits do Gate B0

- **Commit 1 (Passo 2):**  
  `f8eb6eb8ce6da203846767796a62223860d9881c`  
  *Mensagem:* `chore(audit): artefatos de auditoria e diagnósticos 2026-09`
- **Commit 2 (Passo 3):**  
  `ee942f27d8a73da5dacfc96e102e37cee4009c53`  
  *Mensagem:* `test: quarentena de testes de rede com marker network`

---

## 3. Testes em Quarentena de Rede (`@pytest.mark.network`)

1. `tests/e2e/test_websocket.py::test_binance_stream` (conecta diretamente a `wss://stream.binance.com:9443/ws/btcusdt@aggTrade`)
2. `tests/e2e/test_connection.py` (conecta diretamente a `wss://fstream.binance.com/ws/btcusdt@aggTrade`; módulo marcado com `pytestmark = pytest.mark.network`)

Registrado em `pytest.ini`:
```ini
markers =
    network: acessa rede real; excluído por padrão.
```
Adicionado `-m "not network"` em `addopts` do `pytest.ini`.

---

## 4. Resultado da Suíte Completa (Passo 4 Baseline)

Comando executado: `python -m pytest -m "not network" --timeout=120 -q`

- **Passed:** 2700
- **Failed:** 12
- **Skipped:** 4
- **Deselected:** 1 (`tests/e2e/test_websocket.py`)
- **Duração:** 443.47s (07min 23s)
- **Cobertura:** 51.60% (limiar mínimo exigido: 10%)

### Lista de Testes com Falhas Existentes (Nenhum Corrigido)

| # | Arquivo | Teste | Erro Resumido | Classificação |
|---|---|---|---|---|
| 1 | `tests/integration/test_ai_llm_fallback_flow.py` | `test_groq_payload_summary_is_reduced` | `assert "cross" not in reduced` (payload evoluiu incluindo cross-asset, teste legado) | Quebrado (legado) |
| 2 | `tests/integration/test_out_of_order_pruning.py` | `TestOutOfOrderPruning::test_pruning_robustness_with_ooo` | `assert 3 == 2` (tamanho de deque de trades sob nova retenção) | Quebrado (legado) |
| 3 | `tests/integration/test_out_of_order_pruning.py` | `TestOutOfOrderPruning::test_pruning_fast_path_when_ordered` | `assert 5 == 1` | Quebrado (legado) |
| 4 | `tests/integration/test_out_of_order_pruning.py` | `TestEdgeCases::test_single_trade_prune_remove` | `assert 1 == 0` | Quebrado (legado) |
| 5 | `tests/integration/test_out_of_order_pruning.py` | `TestEdgeCases::test_trades_at_cutoff_boundary` | `assert 1 == 0` | Quebrado (legado) |
| 6 | `tests/integration/test_out_of_order_pruning.py` | `TestCVDAfterOOO::test_cvd_correct_after_pruning` | `assert 3 == 2` | Quebrado (legado) |
| 7 | `tests/integration/test_patch_2_fallback_controlado.py` | `TestPatch2FallbackControlado::test_patch_2_groq_fail_com_fallback_openai` | `None != 'openai'` (espera chave OpenAI/Groq não configurada) | Dependente de dado local / env |
| 8 | `tests/integration/test_patch_2_fallback_controlado.py` | `TestPatch2FallbackControlado::test_patch_2_groq_funciona_normal` | `None != 'groq'` | Dependente de dado local / env |
| 9 | `tests/integration/test_patch_2_fallback_controlado.py` | `TestPatch2FallbackControlado::test_patch_2_multiple_fallbacks` | `'dashscope' != 'openai'` | Quebrado / dependente de env |
| 10 | `tests/integration/test_patch_2_fallback_controlado.py` | `TestPatch2FallbackControlado::test_patch_2_provider_nao_groq_vai_para_openai` | `None != 'openai'` | Dependente de dado local / env |
| 11 | `tests/integration/test_window_processor.py` | `test_process_window_happy_path_calls_process_signals_and_feature_store` | `TypeError: FakePipeline.__init__() got unexpected keyword argument 'onchain_updater'` | Quebrado (stub desatualizado) |
| 12 | `tests/unit/test_sound_alert_nonblocking.py` | `test_stop_kills_sound_thread` | `AttributeError: '_NoopClockSync' object has no attribute 'is_synced'` | Quebrado (stub desatualizado) |

*Nota de Flakiness/Timeout:* O teste `tests/unit/test_p06_retention_snapshot.py::test_high_rate_separates_time_from_capacity` processa 12.000 trades com cálculos de desvio padrão em clusters. Sob instrumentação de cobertura (`--cov=.`), requer ~65s (excedendo o timeout padrão de 60s); sem cobertura, roda em ~25s e passa integralmente (classificado como **flaky / dependente de overhead de coverage**).

---

## 5. Status Final do Repositório

Comando: `git status --short`  
Saída: *(vazio após commit deste documento)*
