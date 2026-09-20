# Pre-Paper Trading Readiness

Auditoria somente-leitura do estado REAL do repositório antes da infraestrutura
de testes shadow/paper trading. Nenhum código foi alterado, nenhum teste foi
corrigido, nenhum commit foi feito, nenhuma chamada externa foi realizada.

- Data (UTC): 2026-09-20
- Branch: `main`
- HEAD: `c16ab1a03d7b1ebe17e6be9cba737a74623e04f1`
- Método: leitura de código/imports/chamadas + execução isolada de testes
  herméticos. Nada foi deduzido de documentação.

## 1. Git baseline

```
git status --short   -> 44 linhas (4 modificados + 40 untracked)
git branch --show-current -> main
git rev-parse HEAD -> c16ab1a03d7b1ebe17e6be9cba737a74623e04f1
git log -10 --oneline:
  c16ab1a docs(audit): vetor shadow-collectors explicitamente fora do guard (ACAO-2)
  35aa68f feat(audit): guard backfill-vs-live, enforcement DT-02, DT-04 e resultados ACAO-1
  13ad19b feat(dt02): instrumentacao LAT_CRIT com contexto de buffer + harness de estresse H1
  2221e68 docs(audit): INV-A emissor slim morto/residuo + INV-B hipotese eviccao DT-02
  7024386 docs(audit): DT-02 reclassificado para Critica com evidencia 18-09; formaliza guarda SLA c2df8fa
  c2df8fa fix(orderbook): guarda de SLA para offset >1500ms com fallback cache_bg
  71923a2 research(ops): B3 scheduling and observability for research collectors
  c11ccfb research(binance): B2 lossless prospective positioning dataset with 30d seed
  5b3a96e research(cftc): R4 prospective TRUE-OOS protocol, frozen H3 manifest and append-only collectors
  4a214a1 research(cftc): R3 incremental value pseudo-OOS with ablation and Holm-4
```

Arquivos modificados (tracked, não commitados):

- `ESTRUTURA_SISTEMA_COMPLETO.md` (doc — divergência potencial doc-vs-HEAD, ver §1.1)
- `dados/audit/klines_accept_futures.json` (dado de auditoria)
- `scripts/diagnostics/collect_2h.ps1`
- `tests/integration/test_ml_stale_real_event_pipeline.py`

Arquivos untracked (40): 3 docs em `docs/audit/`
(`FASE1_DEPRECATION_MECHANISM_FAILURE_2026-09.md`,
`ORDERBOOK_LAG_EMPIRICAL_RECONCILIATION_2026-09.md`,
`SIGNOFF_AUDITORIA_FUTURES_2026-09.md`), `logs/run.log.1`, `query`,
2 scripts em `scripts/analytics/`, ~26 scripts em `scripts/diagnostics/`,
`tests/unit/test_dump_raw_trades.py`, `tests/unit/test_ops_data_bridge.py`.

### 1.1 Divergências documentação × repositório

1. `ESTRUTURA_SISTEMA_COMPLETO.md` está modificado no working tree: qualquer
   afirmação dele deve ser revalidada contra o HEAD, não assumida.
2. `scripts/diagnostics/verify_safe_mode.py:83-85` fixa
   `"paper_shadow_mode": True` e `"trade_executor_state": "PASSIVE_OBSERVER_SHADOW"`
   como constantes — é alegação do script, não estado verificado: NÃO existe
   PaperExecutor nem shadow ledger no repositório (ver §5).

## 2. Test baseline

Configuração existente (`pytest.ini` na raiz):

- `testpaths = tests`, `python_files = test_*.py`, `--strict-markers`,
  `-p no:warnings`, `--cov=. --cov-report=term-missing --cov-report=html`,
  `--cov-fail-under=10`, `asyncio_mode = auto`, `timeout = 60`
  (`timeout_method = thread`), `log_cli = true`.
- Markers: `slow, integration, unit, async, performance, db, network, payload`.
- Isolamento por conftest: `tests/conftest.py` fixa `BOT_TEST_MODE=1` (+ fixture
  TimeManager mockado, limpeza de registry Prometheus);
  `tests/payload/conftest.py` fixa `UNIT_TEST=1`/`DISABLE_EXTERNAL_DATA=1` e
  anula `MacroDataProvider`/`EnhancedRegimeDetector` no builder;
  `tests/golden/conftest.py` tem guarda `no_network` (requests/urllib/socket/
  yfinance/aiohttp explodem) + relógio congelado.
- Ambiente: Python 3.12.8, pytest 9.0.1 (global e `.venv`), 244 arquivos
  `test_*.py` (`e2e/`, `golden/`, `integration/`, `payload/`, `unit/`, …).

Execuções realizadas (único desvio registrado: `-o addopts=""` para não
mascarar resultado unitário com `--cov-fail-under=10`; demais ini preservados):

| Suíte | Comando | passed | failed | skipped | xfailed | Duração | Exit |
|---|---|---|---|---|---|---|---|
| Rápida/isolada (5 arquivos: outcome_tracker_direction_aware, outcome_tracker_boundary, signal_direction, event_similarity_direction_aware, backfill_guard) | `python -m pytest <5 arquivos> -o addopts="" -q` | 119 | 0 | 0 | 0 | 18.37s | 0 |
| Herméticas (tests/golden + tests/payload) | `python -m pytest tests/golden tests/payload -o addopts="" -q` | 359 | 0 | 0 | 0 | 17.63s | 0 |
| Completa (244 arquivos) | NÃO EXECUTADA | — | — | — | — | — | — |

Suíte completa BLOQUEADA: `tests/e2e/` (`test_websocket.py`,
`test_connection.py`, `test_market_orchestrator_comprehensive.py`, …) e parte
de `tests/integration/` não têm isolamento comprovado (sem guarda no-network);
risco de acesso a Binance/IA/MT5. Warnings relevantes: nenhum (cauda `-q`
limpa nas duas execuções).

## 3. Production data path

Caminho real confirmado por imports/chamadas (arquivo:linha). Tudo abaixo está
no caminho de produção via `main.py` → `EnhancedMarketBot`.

| # | Estágio | Arquivo | Classe/Função | Quem chama | Entrada → Saída | Em produção? |
|---|---|---|---|---|---|---|
| 1 | Entry/startup | `main.py:197-437` | `main()` | `__main__` | env/config → `bot.run()` | SIM |
| 2 | Guard observação | `main.py:323-328`, `config/env_policy.py:105-135` | `assert_observation_safe` | `main()` | settings/env → aborta ou no-op | SIM |
| 3 | Bot | `market_orchestrator/market_orchestrator.py:246` | `EnhancedMarketBot.__init__` | `main.py:377` | config → buffers/analyzers/connection | SIM |
| 4 | WS trades (Binance Futures público) | `market_orchestrator/connection/robust_connection.py` | `RobustConnectionManager` (`connect`, callbacks `on_message`) | `run():2884` | WS `@trade/@aggTrade` → `on_message` | SIM |
| 5 | Normalização/clamp OOO | `market_orchestrator.py:777-1037` | `on_message` | connection_manager | raw JSON → `norm{p,q,T,T_raw,m,source,trade_id}`; clamp `T<last_T`, original em `T_raw` (:953) | SIM |
| 6 | Buffer + Flow | `trading/trade_buffer.py`, `flow_analyzer.py`, `market_orchestrator/flow/trade_flow_analyzer.py` | `trades_buffer.add_trade_sync`, `flow_analyzer.process_trade` (:985) | `on_message` | trade norm → métricas de fluxo | SIM |
| 7 | Klines (REST fechados) | `market_orchestrator.py:2597-2680` | `_prefetch_ohlc_history` | `initialize():2588` | `GET fapi.binance.com/fapi/v1/klines` 1m → `pattern_ohlc_history` + SessionVWAP bootstrap; candle aberto NUNCA incluído (:2615-2616, :2636-2637) | SIM |
| 8 | Janelas [1,5,15] | `market_orchestrator/windows/window_processor.py`, `market_orchestrator.py:1045-1068` | `WindowProcessor.submit_window` → `process_window_snapshot` | `_process_window` (:1056) | snapshot + `close_ms` → queue/worker; fila cheia descarta (:1059-1064); warmup consome N janelas (:485-502) | SIM |
| 9 | FlowAnalyzer sinais | `signals/signal_processor.py:38` | `process_signals` | `window_processor.py:899` via `bot._process_signals` | janela válida → `signals[]` | SIM |
| 10 | S/R | `support_resistance/*`, `market_orchestrator.py:2373-2381` | `detect_support_resistance` | `_process_institutional_alerts` / enrich | janela → `support_resistance`, `defense_zones_data` | SIM |
| 11 | Institutional | `market_orchestrator/analysis/institutional_analytics.py:31`, `institutional/enricher.py` | `InstitutionalAnalyticsEngine.compute_all` (:1638), `enrich_signal` (:1710) | `_enrich_signal` | signal + janela → `institutional_analytics` | SIM |
| 12 | Regime (gate) | `market_analysis/regime_rules.py`, `market_orchestrator.py:1207-1241` | `RegimeBasedRules.should_trade` | `_handle_signal_event` | `regime_analysis` + lado/confiança → bloqueia ou libera IA | SIM |
| 13 | Enrichment/publicação | `market_orchestrator.py:1431-1775` | `_enrich_signal` → `event_bus.publish("signal")` (:1743) | `process_signals` (:246) | signal validado/enriquecido → bus + `EventSaver.save_event` (:1746) + `avaliar_outcomes_pendentes` (:1755) | SIM |
| 14 | Payload IA (builder REAL) | `market_orchestrator/ai/payload_builder_compact.py:1977` | `build_compact_payload` | `ai_runner.py:634` via `AIRunner.build_payload` | `event_data` → payload flat v2 (`mkt/symbol/epoch_ms/trigger/price/flow/ob/tf/sr/...`) | SIM |
| 15 | Decisão/sinal IA | `market_orchestrator/ai/analyzer_qwen.py`, `ai_runner.py:693,748` | `AIAnalyzer.analyze` → `fuse_decisions` (hybrid) → `validate_llm_response` | `run_ai_analysis_threaded` | payload → `analysis_result{structured,status,success,is_fallback}`; log JSON, sem executor | SIM (análise apenas) |
| 16 | RiskManager | `risk_management/risk_manager.py`, `market_orchestrator/flow/risk_manager.py` (proxy) | `RiskManager.check_trade_request` | NINGUÉM em produção (só `orchestrator.py` legado com `None` e testes) | — | NÃO (existe, desligado) |
| 17 | Executor | `market_orchestrator/flow/trade_executor.py:6` | `TradeExecutor` (stub `is_active=False`) | NINGUÉM em produção | — | NÃO (existe, desligado) |
| 18 | OutcomeTracker | `trading/outcome_tracker.py:42`, `events/event_memory.py:23,50,61` | `register_signal` / `evaluate_pending_outcomes` / `get_historical_probability` | `adicionar_memoria_evento`, `avaliar_outcomes_pendentes` (:1755) | signal/preço → SQLite `signal_outcomes` | SIM (tracking; `get_confidence_for_event` direto DESLIGADO em `:1767 if False`) |
| 19 | Persistência/exportação | `events/event_saver.py`, `audit_live/forensic_payload.py`, `trading/export_signals.py` | `save_event`, `build_and_capture_future_payload` (`llm_transmitted=false`) | `_enrich_signal`, forensic hooks | eventos → JSONL/DB | SIM |

O que existe mas NÃO está no caminho real: `RiskManager`, `TradeExecutor`,
`MarketOrchestrator` (`orchestrator.py` — harness de testes, componentes
`None`), `ai_payload_builder.build_payload_with_cross_asset` (builder real é o
compacto via `AIRunner.create`), `OutcomeTracker.get_confidence_for_event`
direto no sinal.

## 4. Behaviour with AI disabled

Sem `GROQ_API_KEY` (ou analyzer ausente), `ai_runner.py:218-235` fixa
`ai_test_passed=False` (apenas log warning, sem exceção):

- o sistema INICIA normalmente;
- continua coletando mercado (WS → buffer → FlowAnalyzer);
- continua criando janelas (`WindowProcessor`);
- continua gerando payload? NÃO gera `ai_payload` no path normal (o build
  ocorre dentro de `run_ai_analysis_threaded`, após o gate); MAS o hook
  forense (`signal_processor.py:309-318`, gated por `FORENSIC_CAPTURE=1`)
  constrói o payload futuro localmente sem transmitir;
- NÃO tenta chamar IA: `_handle_signal_event` retorna em
  `market_orchestrator.py:1185-1186` e `run_ai_analysis_threaded` retorna em
  `ai_runner.py:283-288` (com warning);
- fallback: nenhum fallback de decisão é acionado — o evento simplesmente não
  é analisado; nenhuma exceção; nenhuma pipeline bloqueada (enriquecimento,
  `EventSaver`, outcome tracking seguem);
- feature flags / kill-switches APROPRIADOS existentes:
  `OBSERVATION_MODE=1` (aborta startup com credenciais de trade/IA ou
  `HYBRID_ENABLED`/`EXECUTION_ENABLED` — `env_policy.py:105-135`),
  `FORENSIC_NO_LLM=1` (suprime análise IA — `ai_runner.py:274-280`),
  throttler de IA (`init_throttler` em `main.py:23-28`, gate em
  `ai_runner.py:673-691`), cooldown `AI_MIN_INTERVAL_SEC` e bypass de
  `SIDEWAYS`+volume baixo;
- chamadas que poderiam atingir provedores LLM (nenhuma executada nesta
  auditoria): `AIAnalyzer.analyze` (`analyzer_qwen.py`, Groq) chamado apenas
  em `ai_runner.py:693`, após todos os gates acima.

## 5. Execution safety

1. **PaperExecutor real?** NÃO. Só existe o stub `TradeExecutor`
   (`market_orchestrator/flow/trade_executor.py:6-55`, `is_active=False`,
   sem chamada de rede, sem modo paper). Nenhuma classe `Paper*` no repo.
2. **dry-run?** NÃO para trading (só flags `--dry-run` de `auto_fixer/` e
   `data_processing/`, sem relação com ordens).
3. **shadow mode?** PARCIAL. Há observação forense (`FORENSIC_CAPTURE=1` +
   `llm_transmitted=false`, `audit_live/forensic_payload.py:21-80`), runners
   `scripts/diagnostics/run_*_observation.py` e guarda `OBSERVATION_MODE`.
   Não há ledger de ordens shadow.
4. **Ordem real por acidente?** Nenhum caminho encontrado: grep repo-wide por
   `/fapi/v1/order`, `/fapi/v2/order`, `/api/v3/order`, `/fapi/v1/batchOrders`,
   `newOrder`, `futures_create*`, `change_leverage` retorna ZERO em código de
   produção (só o próprio scanner os cita:
   `scripts/diagnostics/verify_safe_mode.py:41`). `EXECUTION_ENABLED` não
   existe em `config/settings.py` (default `False` via `getattr`). Ressalva:
   `BINANCE_API_KEY/SECRET` são carregadas do `.env` em modo normal
   (`settings.py:169-170`) — presentes mas sem consumidor de trading.
5. **Separação decisão/execução?** SIM por ausência: a decisão termina em
   log/`EventSaver`; nenhum executor é instanciado ou chamado em produção.
6. **Idempotência/dedup de ordem?** NÃO (nada a dedupe). Existe dedup de
   EVENTOS: `EventBus` (janela 30s, `events/event_bus.py:98-111`),
   `raw_event_deduplicator` (redução de tamanho p/ LLM, `ai_runner.py:618-630`),
   `_sent_triggers` p/ `ANALYSIS_TRIGGER` (`market_orchestrator.py:1722-1732`).
7. **Kill switch?** PARCIAL: `OBSERVATION_MODE` (abort de startup),
   `FORENSIC_NO_LLM=1`, throttler + cooldowns de IA, circuit-breaker de
   clock-sync (`can_place_order`, sem ordens a proteger). Sem halt de trading
   em runtime (não há trading).
8. **Posição duplicada?** PARCIAL: `RiskManager.add_position` rejeita símbolo
   duplicado (`risk_manager.py:199-203`), mas o módulo não é usado em produção.
9. **Reconciliação de posição?** NÃO (só reconciliação de volumes em
   `data_processing/data_validator.py:472-645`, sem relação com posições).
10. **Fees/slippage/funding simulados?** NÃO. `funding_rate_percent` trafega
    como dado de mercado até o payload (`fr`, `test_funding_rate_pipeline_p0`);
    `assess_liquidity_risk` estima `slippage_estimate`
    (`risk_manager.py:473-521`) mas é código morto em produção. Sem modelo de
    custo.

## 6. Outcome/persistence capabilities

`OutcomeTracker` REAL (`trading/outcome_tracker.py:42`, SQLite, "Sem API
externa - cálculo 100% local"), ligado à produção via `events/event_memory.py`
(`register_signal` em `adicionar_memoria_evento`, `evaluate_pending_outcomes`
via `market_orchestrator.py:1755`). Schema `signal_outcomes`: `id`,
`signal_epoch_ms`, `event_type`, `battle_result`, `entry_price`, `symbol`,
`context_json`, `outcome_{5m,15m,30m,60m}_pct`,
`outcome_direction_{5m,15m,30m,60m}`, `evaluated_at`, `created_at`.
Horizontes boundary-only fail-closed (`OUTCOME_BOUNDARY_TOLERANCE_MS=1000`,
`:36`; NULL permanente se boundary perdido).

| Campo | Status | Evidência |
|---|---|---|
| window_id | PARCIAL | `janela_numero`, `features_window_id`, `symbol_closeMs` em logs/eventos; sem coluna dedicada |
| decision_id | NÃO EXISTE | grep repo-wide: zero ocorrências em código de produção |
| timestamp da janela | EXISTE | `close_ms`/`epoch_ms` (= close da janela, `market_orchestrator.py:1452-1453`) |
| timestamp da decisão | PARCIAL | `event_data["ai_payload"]` carrega epoch do evento; sem timestamp próprio da decisão |
| timestamp de entrada | NÃO EXISTE | `entry_price` existe, mas sem `entry_ms` (preço é do close do sinal) |
| LONG/SHORT/NEUTRAL/UNKNOWN | EXISTE | `common/signal_direction.py` (`infer_signal_side`, `classify_outcome`) |
| confidence | PARCIAL | `historical_confidence`/`statistical_confidence`/`directional_win_rate`; sem confiança canônica da decisão |
| entry | EXISTE | `entry_price` (`preco_fechamento` no close) |
| SL / TP | NÃO EXISTE | só dataclasses avulsas em `RiskManager`/`TradeRequest`, sem uso |
| exit / motivo da saída | NÃO EXISTE | horizontes fixos substituem saída; sem razão de saída |
| direction_correct | PARCIAL | `directional_win_rate` agregado + `classify_outcome`; sem coluna por trade |
| trade_win | NÃO EXISTE | sem conceito de trade (só outcome direcional de preço) |
| PnL bruto | NÃO EXISTE | só `outcome_*_pct` (variação %) |
| fees / slippage / funding | NÃO EXISTE | ver §5.10 |
| PnL líquido | NÃO EXISTE | consequência dos anteriores |
| regime | PARCIAL | `context_json` tem `trend`+`volatility`; `regime_analysis` no evento, fora do SQLite |
| sessão | EXISTE | `context_json.session` (`trading_session`) |
| data quality | PARCIAL | `qual/comp`, `orderbook_quality`, `data_source` no evento; fora do SQLite |
| modelo/prompt/config/commit | PARCIAL | `decision_features_hash` no payload (`ai_payload_builder.py:1014-1279`) + `capture_run_id` forense; sem ledger de experimento |

## 7. AI payload readiness

- Builder REAL: `build_compact_payload(event_data)`
  (`market_orchestrator/ai/payload_builder_compact.py:1977`), injetado via
  `AIRunner.create` (`ai_runner.py:83-91`) e chamado em `ai_runner.py:634`.
  (O `ai_payload_builder.py` legado não é o caminho; regime/ML entram via
  seções do compacto e `ml.hybrid_decision.fuse_decisions`.)
- Dados de entrada: trigger, `price`, `regime`, `flow`, `ob`, `tf`, `sr`,
  `w` (whale), `quant` (só se `pu != 0.5`), `ext`, `alerts`, `qual`, `mkt`,
  `symbol`, `epoch_ms`; seções obrigatórias sempre presentes mesmo vazias
  (`{"_": "no_data"}`, `:2078-2086`).
- Sanitização NaN/Inf: `ensure_safe_llm_payload` aplica `sanitize_json_safe`
  (non-finite → `None`, sem inventar número — `llm_payload_guardrail.py:289-293`);
  whitelist de keys (`:68-73`), teto 6KB (`:56`), compressão proporcional.
- Timestamp/freshness: `epoch_ms` do evento com fallback para wall-clock
  (`:2021`, `:2317`) — RESSALVA: fallback mascara ausência (marcar em vez de
  omitir seria melhor); freshness explícita via `qual{lat,liq,ms,src,comp}` +
  `latency/calendar` snapshots; capabilities via `market_orchestrator/capabilities.py`
  (`CONTINUOUS_L2`, `ICEBERG_DETECTION_SUPPORTED`, import `:1388`).
- Ausência vs zero: `qual` só emitida com dado (`:2053`); `quant` omitida se
  `0.5`; stubs `no_data`; feriado/origem degradada propagados (`:2045-2069`).
- Serialização RFC 8259: garantida por sanitização + `allow_nan=False` nos
  testes (`tests/golden/conftest.py:135-137`, `test_p00_compact_preserved`,
  `test_binance_positioning_b2`); sem `allow_nan=False` no `dumps` de produção
  — PARCIAL.
- Debug dump: `audit_live/forensic_payload.py` (`llm_payloads.jsonl` com
  `llm_transmitted=false`, `payload_stages.jsonl` PRE/POST builder/compressor/
  guardrail) + `payload_metrics_aggregator.py` (métricas JSONL).
- Payload sem IA: SIM — `build_payload(event_data)` é função pura chamável
  direto; captura forense prova o fluxo sem `chat.completions.create`.
- Exemplo baseado em fixture existente (`tests/payload/test_build_compact_payload_smoke.py:17-63`
  + `tests/golden/test_gw_payload.py`): evento `{symbol, epoch_ms,
  tipo_evento, preco_fechamento, contextual_snapshot.ohlc, market_context,
  market_environment, external_markets, historical_vp, derivatives{...,
  funding_rate_percent}, ml_features, ...}` → payload
  `{mkt, symbol, epoch_ms, trigger, price{c, fr,...}, regime, flow, ob, tf, sr,
  qual?, ...}` com `_v=2`.

## 8. Temporal/lookahead risks

1. **Candle aberto como fechado (MITIGADO)**: prefetch filtra
   `k[0] <= last_closed_open`
   (`market_orchestrator.py:2615-2616, 2636-2637`).
2. **Sinal fora do boundary de 1m**: `register_signal` apenas avisa e mantém
   (`outcome_tracker.py:109-118`) — outcomes viram NULL permanente
   (fail-closed documentado; perda de dado, não lookahead).
3. **Entry anterior à decisão (GAP p/ paper)**: `entry_price =
   preco_fechamento` do close do sinal (`outcome_tracker.py:103`,
   `market_orchestrator.py:1452-1453`); a decisão IA ocorre DEPOIS. Regra
   "entry estritamente após decisão" (R19) não existe.
4. **Dados posteriores à janela**: `evaluate_pending_outcomes` usa o preço do
   sinal CORRENTE como preço futuro do boundary
   (`market_orchestrator.py:1751-1755` + `outcome_tracker.py:202-205`, drift
   `0..1000ms`) — correto por construção, MAS depende do chamador passar o
   close do boundary; sem teste de contrato desse chamador.
5. **TP/SL com informação futura**: N/A hoje (sem TP/SL); horizontes usam
   `UPDATE ... WHERE outcome_X_pct IS NULL` (`:207-221`, sem sobrescrita;
   `rowcount` ignora corrida — segundo writer perde silenciosamente).
6. **`evaluated_at` compartilhado** (`:207-220`): é o ÚLTIMO preenchimento,
   não por-horizonte (documentado em `:158-160`; fácil de misinterpretar).
7. **Trades perdedores abertos descartados**: N/A (sem posições); janelas
   descartadas por fila cheia são logadas (`:1059-1064`).
8. **Duplicação por retry**: `_sent_triggers` é só memória
   (`:1722-1732`); dedup do `EventBus` é janela de 30s; sem `decision_id`
   persistente — restart reprocessa.
9. **Heurística de agressor**: `m = (p <= last_price)` quando ausente
   (`:958-960`) — misclassificação possível em dado sem flag.
10. **Warmup inconsistente**: `process_window_snapshot` descarta janelas em
    warmup (`window_processor.py:480-502`), mas o `FlowAnalyzer` já consumiu
    os trades na ingestão (`market_orchestrator.py:985`) — estado de fluxo ≠
    janelas processadas no início.
11. **Epoch de emergência**: fallback usa wall-clock (`ai_runner.py:659`);
    payload de emergência carrega `_emergency: true` (`:660-667`) — ok se
    filtrado nas análises.

## 9. Existing test coverage

Coberto (arquivo → requisito):

- OutcomeTracker/boundary: `tests/unit/test_outcome_tracker_boundary.py`,
  `test_outcome_tracker_direction_aware.py` (inclui simetria LONG/SHORT),
  `tests/integration/test_corrections.py`.
- Direction-aware: `test_signal_direction.py`, `test_event_similarity_direction_aware.py`,
  `test_market_orchestrator_direction_confidence.py`.
- Backfill-vs-live: `test_backfill_guard.py` (+ `common/backfill_guard.py`).
- Payload sem IA / hermético: `tests/payload/*` (conftest hermética),
  `tests/golden/*` (guarda no-network, relógio congelado, `assert_json_strict`).
- Mock LLM / offline: `tests/integration/test_ai_analyzer_mock.py`,
  `test_ai_llm_fallback_flow.py`,
  `test_microstructure_claims_neutralization.py::test_offline_no_llm_enforcement`.
- Funding como dado: `test_funding_rate_pipeline_p0.py`,
  `test_funding_rate_fallback.py`, `test_gw_payload.py` (`fr` presente/ausente).
- Slippage analítico: `test_orderbook_analyzer_full_coverage.py`,
  `test_orderbook_market_impact_contract.py` (matriz de impacto, sem modelo de execução).
- Leak de ML (escopo ML, não prod-path):
  `test_feature_evaluator_diagnostics.py::test_future_leakage_detected`.
- Idempotência pontual: `test_ai_llm_fallback_flow.py::test_shutdown_is_idempotent`,
  `test_binance_positioning_b2.py` (re-execução idempotente), shutdown macro
  idempotente.

NÃO coberto (grep em `tests/` retorna zero ou só escopo alheio):

- paper trading, PaperExecutor, dry-run de trading, shadow ledger;
- `decision_id`/`window_id` rastreáveis fim-a-fim; entry/decision timestamps;
- duplicate order / idempotência de ordem; kill switch de trading;
- placebo/random baseline; replay de trades; scorecard; calibração de confiança
  (nota: `test_sr_etapa5b_contract.py` marca scores como *uncalibrated* —
  não-calibração explícita);
- anti-lookahead do caminho de produção (só ML-feature leak);
- métricas agregadas por regime/sessão; ledger de experimento
  (git/config/model/prompt hashes).

## 10. Missing components

Ver tabela R01–R36 abaixo. Faltam construir, no mínimo: contrato canônico de
decisão (`decision_id`, timestamps, lado, confiança, entry/SL/TP), PaperExecutor
com ledger persistente, modelo de custos (fees/slippage/funding), harness de
replay + placebo + scorecard, e quarentena dos testes e2e/integração com rede.

### R01–R36

| ID | Requisito | Status | Evidência | Arquivo | Risco |
|---|---|---|---|---|---|
| R01 | Binance Futures data ingestion | READY | WS `@trade/@aggTrade` + REST `fapi/.../klines` e `futures/data/*` no caminho real | `market_orchestrator.py:497-517,2597-2637`, `robust_connection.py` | Baixo |
| R02 | Trades continuous | READY | `AsyncTradeBuffer` + reconnect + heartbeat `trade_ingestion` | `market_orchestrator.py:296-320,1004-1007` | Baixo |
| R03 | Orderbook usable | PARTIAL | fetch live + emergency/cache + `orderbook_quality`/`data_source`; sem teste de SLA dedicado | `market_orchestrator.py:373-416,1781-1833` | Médio |
| R04 | Klines usable | PARTIAL | prefetch closed-only + bootstrap VWAP; sem teste dedicado do prefetch | `market_orchestrator.py:2597-2680` | Médio |
| R05 | Window construction | READY | `WindowProcessor [1,5,15]`, boundary `((T//W)+1)*W`, descarte sob pressão | `window_processor.py`, `market_orchestrator.py:771-772,1045-1068` | Baixo |
| R06 | Flow analysis | READY | `FlowAnalyzer`+`TradeFlowAnalyzer` no hot path; golden math | `market_orchestrator.py:369-371,427,984-985` | Baixo |
| R07 | S/R | READY | `detect_support_resistance` + defense zones no enrich e no payload; testes `sr_*` | `market_orchestrator.py:2373-2381`, `support_resistance/*` | Baixo |
| R08 | Institutional enrichment | READY | `compute_all` + `enrich_signal` no caminho; bateria `institutional_*` | `market_orchestrator.py:1562-1713` | Baixo |
| R09 | Regime | READY | regime no payload + `RegimeBasedRules.should_trade` como gate; testes regime | `market_orchestrator.py:1207-1241` | Baixo |
| R10 | Data quality/freshness | PARTIAL | `qual/comp`, `quality_summary`, fail-closed; sem SLA unificado testado | `payload_builder_compact.py:2029-2076` | Médio |
| R11 | Payload can be generated without LLM | READY | captura forense + `build_payload` pura; 359 testes herméticos verdes | `forensic_payload.py:21-80`, `ai_runner.py:96-98` | Baixo |
| R12 | LLM can be disabled safely | READY | `ai_test_passed=False` → early return, pipeline segue; teste offline | `ai_runner.py:218-235,283-288` | Baixo |
| R13 | Mock LLM supported | READY | mocks + fallback flow + enforcement offline | `tests/integration/test_ai_analyzer_mock.py`, `test_ai_llm_fallback_flow.py` | Baixo |
| R14 | Canonical decision contract | PARTIAL | `validate_llm_response` + `CompactAIPayload` + whitelist; sem `decision_id`/ledger | `llm_response_validator.py`, `llm_payload_guardrail.py`, `common/ai_payload_types.py` | Alto |
| R15 | RiskManager before execution | MISSING | biblioteca completa porém sem instanciação em produção | `risk_management/risk_manager.py`, `flow/risk_manager.py` (proxy) | Alto (quando houver execução) |
| R16 | PaperExecutor | MISSING | só stub `TradeExecutor(is_active=False)` desligado | `flow/trade_executor.py:6-55` | Alto |
| R17 | No-real-order safety | READY | zero endpoints de ordem no repo (grep); scanner dedicado; `EXECUTION_ENABLED` ausente (default False); guard observation | `verify_safe_mode.py`, `env_policy.py`, `settings.py` | Baixo (ressalva: chaves Binance carregadas sem consumidor) |
| R18 | decision_id/window_id traceability | PARTIAL | `janela_numero`/`features_window_id`/`epoch_ms`; sem `decision_id` | `market_orchestrator.py:1450-1453,1722-1732` | Alto |
| R19 | Entry strictly after decision | MISSING | entry = close do sinal, decisão depois; sem regra | `outcome_tracker.py:100-107` | Alto |
| R20 | TP/SL/horizon resolution | PARTIAL | horizontes 5/15/30/60 boundary-only; sem TP/SL nem motivo de saída | `outcome_tracker.py:144-232` | Alto |
| R21 | Fees | MISSING | sem modelo de fee em qualquer caminho | — | Alto |
| R22 | Slippage | PARTIAL | estimativa analítica em código morto; matriz de impacto no payload; sem modelo de execução | `risk_manager.py:473-521` | Alto |
| R23 | Funding | PARTIAL | funding como dado de mercado (`fr`); sem acrual de custo | `test_funding_rate_pipeline_p0.py` | Médio |
| R24 | OutcomeTracker | READY | ligado via `event_memory`; boundary tests verdes (119 passed inclui) | `event_memory.py:20-61`, `outcome_tracker.py` | Baixo |
| R25 | direction_correct separate from trade_win | PARTIAL | `directional_win_rate`/`classify_outcome` agregados; sem colunas por trade | `outcome_tracker.py:234-304`, `signal_direction.py` | Médio |
| R26 | Persistent experiment ledger | PARTIAL | SQLite `signal_outcomes` + eventos JSONL; sem ledger de experimento com hashes | `outcome_tracker.py:57-92`, `event_saver.py` | Alto |
| R27 | Replay | MISSING | só guarda ANTI-backfill + replay escopo CFTC-research | `test_backfill_guard.py`, `scripts/analytics/cftc_*` | Alto |
| R28 | Placebo/random baseline | MISSING | zero ocorrências em `tests/` | — | Alto |
| R29 | LONG/SHORT symmetry test | READY | testes simétricos LONG/SHORT verdes | `test_outcome_tracker_direction_aware.py:46-134` | Baixo |
| R30 | Anti-lookahead test | PARTIAL | só leak de feature ML; boundary-only por design sem teste de contrato do chamador | `test_feature_evaluator_diagnostics.py:122-152` | Médio |
| R31 | Scorecard | MISSING | zero ocorrências em `tests/` | — | Alto |
| R32 | Confidence calibration | PARTIAL | scores marcados *uncalibrated* explicitamente; bias monitor sem curva de calibração | `test_sr_etapa5b_contract.py:214`, `ai_runner.py:741-759` | Médio |
| R33 | Metrics by regime/session | PARTIAL | sessão/tendência/volatilidade em `context_json`; sem agregação testada | `outcome_tracker.py:120-132` | Médio |
| R34 | Kill switches | PARTIAL | observation/FORENSIC_NO_LLM/throttler; sem kill de ordens nem teste | `env_policy.py`, `ai_runner.py:274-280` | Médio |
| R35 | Idempotency/dedup | PARTIAL | dedup de eventos (bus 30s, triggers em memória); sem idempotência de ordem | `event_bus.py:98-111`, `market_orchestrator.py:1722-1732` | Alto (quando houver execução) |
| R36 | Reproducibility: git/config/model/prompt hashes | PARTIAL | `decision_features_hash` + `capture_run_id`; sem ledger | `ai_payload_builder.py:1014-1279`, `forensic_context.py` | Médio |

### BLOCKERS para rodar shadow/paper SEM IA

- **B1 — Sem executor nem ledger (R16, R26, R15):** construir `PaperExecutor`
  + ledger persistente com `RiskManager` no caminho antes de qualquer simulação.
- **B2 — Sem contrato de decisão (R14, R18, R19):** definir `decision_id`,
  `window_id`, timestamps de janela/decisão/entrada e regra entry-após-decisão.
- **B3 — Sem custos (R21, R22, R23):** modelar fees/slippage/funding antes de
  qualquer PnL, para não validar edge fictício.
- **B4 — Sem harness de validação (R27, R28, R31):** replay + placebo/random +
  scorecard são pré-requisitos para ler qualquer resultado shadow.
- **B5 — Suíte completa não validada:** quarentenar `tests/e2e` e integração
  com rede (isolamento não provado) antes de gatear shadow em CI.

Não-blockers confirmados: R17 (sem caminho de ordem real — risco residual
baixo, limitado a credenciais carregadas sem consumidor) e R12 (IA pode ficar
desligada com segurança; payload pode ser gerado/capturado sem LLM — R11).

*Fim do relatório. Nada foi implementado, corrigido ou commitado.*
