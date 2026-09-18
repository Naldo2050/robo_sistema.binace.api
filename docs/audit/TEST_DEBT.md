# Dívida Técnica de Testes (TEST_DEBT)

Este documento registra os testes atualmente quebrados no repositório, identificando o commit exato de regressão (via `git log -S`) e a causa raiz técnica.

> **Status:** Registrado para resolução futura. Nenhuma correção de lógica executada nesta fase.

---

## 1. Testes de `CryptoCOT` (`tests/unit/test_institutional_cot.py`)

* **Commit que quebrou:** `e8bb9d5` (*freeze(o1): production shadow observation baseline (P0-P1.3C, V1-V1.2, test-the-tester, O1 tools)*)
* **Causa Raiz Geral:** O commit `e8bb9d5` reestruturou a classe `CryptoCOT` (`institutional/crypto_cot.py`) de um modelo de rastreamento com estado em memória (`add_data()`, `reset()`, etc.) para um analisador estatutário/funcional sem estado baseado no contrato `positioning_data` e no enum `PositioningRegime`. Os testes unitários originais não foram migrados para a nova interface.

| # | Teste Unitário | Commit | Causa Específica |
|---|---|---|---|
| 1 | `TestCOTInit::test_default` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'data_points'` (propriedades `data_points` e `latest` removidas do novo design). |
| 2 | `TestCOTAddData::test_add_single` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'` (método de acúmulo temporal `add_data` foi descontinuado). |
| 3 | `TestCOTFundingAnalysis::test_extreme_positive_funding` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 4 | `TestCOTFundingAnalysis::test_extreme_negative_funding` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 5 | `TestCOTFundingAnalysis::test_normal_funding_no_signal` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 6 | `TestCOTOIAnalysis::test_oi_confirming_uptrend` | `e8bb9d5` | `TypeError: CryptoCOT.__init__() got an unexpected keyword argument 'oi_change_threshold_pct'`. |
| 7 | `TestCOTSqueeze::test_short_squeeze_conditions` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 8 | `TestCOTSqueeze::test_long_squeeze_conditions` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 9 | `TestCOTAnalysis::test_analyze_empty` | `e8bb9d5` | `TypeError: CryptoCOT.analyze() missing 1 required positional argument: 'positioning_data'`. |
| 10 | `TestCOTAnalysis::test_analyze_complete` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'`. |
| 11 | `TestCOTReset::test_reset` | `e8bb9d5` | `AttributeError: 'CryptoCOT' object has no attribute 'add_data'` (método `reset` e armazenamento interno foram descontinuados). |

---

## 2. Teste de Estrutura do Snapshot Price (`tests/payload/test_build_compact_payload_snapshot.py`)

* **Commit que quebrou:** `e8bb9d5` (*freeze(o1): production shadow observation baseline (P0-P1.3C, V1-V1.2, test-the-tester, O1 tools)*)
* **Causa Raiz:** O commit `e8bb9d5` incluiu a chave `fr` (*funding rate*) dentro da seção `price` do payload compacto (`market_orchestrator/ai/payload_builder_compact.py`), porém não adicionou `'fr'` ao conjunto de chaves permitidas (`OPTIONAL_PRICE_KEYS`) em `tests/payload/test_build_compact_payload_snapshot.py`.

| # | Teste Unitário | Commit | Causa Específica |
|---|---|---|---|
| 12 | `test_snapshot_price_section_structure` | `e8bb9d5` | `AssertionError: Chaves inesperadas em price: {'fr'}` (chave `fr` adicionada no builder sem atualizar `OPTIONAL_PRICE_KEYS`). |

---

## 3. Calibração de Thresholds de Fluxo e Volume Futures (Rodada 2026-09-05)

* **Motivação:** No mercado de Binance Futures (USD-M `btcusdt@aggTrade`), a distribuição de tamanho de ordens e volume é significativamente mais elevada que em Spot. Manter limiares legados de Spot (whale trade >= 1.0 BTC, spike por multiplicador puro sem piso p95) distorcia métricas e saturava alertas.
* **Base Empírica:** 
  - `dados/audit/aggtrades_sample.json`: amostra de 1h (2026-09-04 14:00–15:00 UTC) com p99 = 2.0 BTC.
  - `dados/audit/klines_fut_28d.json`: 28 dias de klines 1m (40.321 velas) gerando `config/volume_baseline_fut.json` (p95 global = 367.6 BTC/minuto e tabela horária 0..23 UTC).

### Ajustes de Código e Testes Atualizados

| Componente / Arquivo | Ajuste Realizado | Testes Unitários Ajustados | Motivo do Ajuste no Teste |
|---|---|---|---|
| `flow_analyzer/constants.py` | `DEFAULT_WHALE_TRADE_THRESHOLD`: 1.0 → 2.0 BTC.<br>`ORDER_SIZE_BUCKETS`: retail (0, 0.2), mid (0.2, 2.0), whale (2.0, None). | `tests/unit/test_flow_consistency_regression.py` | Fixtures de teste utilizavam 0.99 BTC para retail e 1.5 BTC para whale; atualizado para 0.1999 BTC (retail) e 2.5 BTC / 2.0 BTC (whale). |
| `flow_analyzer/constants.py` | Buckets de tamanho de ordem atualizados. | `tests/unit/test_flow_analyzer_metrics.py` | Trades de teste usavam 1.5 BTC para testar contagem de whale; atualizado para 2.5 BTC (whale) e 0.1 BTC (retail). |
| `trading/alert_engine.py` | Gate duplo para `VOLUME_SPIKE`: `ratio >= threshold` E `current_volume >= p95_hourly[hour_utc]`. | `tests/unit/test_volume_spike_dual_gate.py` (Novo) | Validação do gate duplo: volume abaixo do p95 bloqueia o alerta mesmo com ratio alto; volume acima do p95 dispara normalmente. |
| `orderbook_wrapper.py` & `data_handler.py` | Snapshot síncrono com timeout 1.5s e schema padronizado (`timestamps`, `source`, `snapshot_offset_ms`). | `tests/unit/test_orderbook_sync_snapshot.py`<br>`tests/unit/test_signal_orderbook_schema.py` (Novos) | Validação de fallback síncrono para cache background e paridade de schema entre sinais e triggers. |

---

## 4. Débitos Técnicos da Auditoria de Produção Futures (2026-09-08)

Identificados durante a execução e validação da Coleta Oficial de 2 Horas (Item 8) na janela de overlap Londres/NY (08/09/2026 13:35–15:35 UTC, $N = 204.773$ trades, 120 janelas).

### DT-01 — Observabilidade Contínua de Memória Heap/RSS do Processo Python

* **ID:** `DT-01`
* **Dono:** Equipe de Infraestrutura / Telemetria
* **Prioridade:** Média
* **Gatilho de Revisão:** Antes de habilitar execução ininterrupta 24/7 em produção contínua.
* **Descrição Técnica:** A Validação 6 atestou a ausência de saturação com base no crescimento controlado em disco (`trading_bot.db` em 2.956 KB e `collect_2h.log` em 707 KB) e no health check estático de `MacroUpdateService` (`memory_mb <= 500 MB`). Não há, contudo, amostragem contínua minuto a minuto da memória residente real (RSS) do processo Python persistida no SQLite ou exportada periodicamente no `/metrics`.
* **Impacto:** Impossibilidade de diagnosticar vazamentos lentos e progressivos de memória (memory leaks sutis em deques, closures assíncronas ou caches de enriquecimento) antes que atinjam o threshold de OOM (*Out Of Memory*) do container em sessões estendidas de dias ou semanas.
* **Critério Objetivo de Resolução:** Série temporal de memória RSS (`psutil.Process().memory_info().rss` em Python ou `$proc.WorkingSet64` em PowerShell) registrada minuto a minuto em coleta futura de 2h+, comprovando empiricamente a ausência de crescimento linear sustentado ao longo da sessão, acompanhada da exposição contínua do gauge `process_memory_rss_bytes` no endpoint `/metrics`.

### DT-02 — Otimização de Concorrência e Contenção de Lock em `FlowAnalyzer`

* **ID:** `DT-02`
* **Dono:** Equipe de Engenharia do Core / FlowAnalyzer
* **Prioridade:** Crítica (reclassificada de Alta em 18/09/2026 — ver Adendo abaixo; bloqueante antes de qualquer expansão de escopo: multi-símbolo, novos detectores, aumento de carga)
* **Gatilho de Revisão:** Antes de reativar detectores adicionais de alta frequência ou monitorar múltiplos pares concorrentes no mesmo loop.
* **Descrição Técnica:** O método `process_trade()` em `flow_analyzer/core.py` e o método `_create_snapshot()` (acionado por `get_metrics()` a cada fechamento de janela de 1 minuto) compartilham o mesmo lock reentrante (`self._lock`). Sob rajadas de volume (>100 trades/s) ou durante a cópia e poda do histórico (`flow_trades_copy = [t for t in self.flow_trades if t['ts'] >= cutoff]`), a espera pelo lock provocou 11 alertas de `LATÊNCIA CRÍTICA: process_trade took > 200ms` (picos de 1.6s a 2.1s). Embora o `AsyncTradeBuffer` tenha absorvido 100% dos trades sem perdas, a contenção transitória no caminho quente de ingestão deve ser minimizada.
* **Impacto:** Em regimes extremos de liquidação em cascata (>1.000 trades/s sustentados), o enfileiramento por contenção de lock pode elevar a latência ponta a ponta do pipeline e pressionar desnecessariamente a capacidade do buffer de entrada.
* **Critério Objetivo de Resolução:** Desacoplamento da cópia de histórico de janelas em relação ao lock de ingestão (via *shallow copy* atômica, buffer circular imutável ou *read-copy-update*), validado em nova sessão sob estresse (>300 trades/s sustentados) com redução de $\ge 90\%$ nos eventos de latência crítica (>200ms), mantendo 100% de paridade entre trades recebidos e processados.
* **Adendo 18/09/2026 (observação go-live, N=549.928, 76,4 trades/s médios):** 39 eventos (3,5x para 2,7x de volume — escala aprox. proporcional em contagem), porém cauda máxima de **4471ms** (2,1x o teto de 08/09) com **4 eventos >1500ms**, dos quais os 2 piores (4471ms e 3137ms) ocorreram em janelas de volume **modesto/baixo** e **nenhum** coincidiu com fechamento de janela — fora do padrão caracterizado em 08/09 (rajada + fecho). Mecanismo incompletamente caracterizado e com gatilhos além do throughput: reclassificado para **Crítica/Bloqueante** para expansão de escopo. Operação contínua single-símbolo permanece aprovada (buffer absorveu 100%, 0 gaps >2min, 0 crashes).
* **INV-B/18-09 — Segunda fonte de contenção (tempestade de evicção):** nos instantes dos piores stalls, o log mostra `flow_trades_capacity_truncated` (deque 100k saturado, ~200k evicções acumuladas), `RollingAggregate(*m) capacity limit hit` e flush parquet do feature store nos mesmos segundos (janela #50 levou 7,7s). Top-3 da cauda acoplados a truncamentos em 0–5s (4471ms→0s, 3137ms→5s, 1962ms→2s); 23/39 eventos a ±60s de truncamentos. **Hipótese testável H1:** stalls >1500ms co-ocorrem com rajadas de evicção de buffers limitados (cópia+poda de 100k sob `self._lock`), não com trades/s instantâneos — validar logando tamanho do buffer/evicções em cada `LATÊNCIA CRÍTICA` (predição: ≥70% dos >1500ms a ±10s de truncamento). **H2 (secundária):** flush parquet + fecho pesado de janela disputam a mesma thread (evento de 2501ms às 12:21:30, sem truncamento próximo, precede o shutdown). Direção do fix: evicção incremental no insert, caps maiores, cópia de snapshot fora do lock.
* **AÇÃO 1/18-09 — Teste controlado H1/H2 (instrumentação + 3 runs, 630k trades):** instrumentado o log `LATÊNCIA CRÍTICA` com `buffer_size | evict_5s | parquet_flush` (`flow_analyzer/core.py`; flag `FLUSH_IN_PROGRESS` em `data_processing/feature_store.py`) + harness `scripts/diagnostics/stress_flow_buffer.py` (250 t/s, 14 min, saturação confirmada nos 3 runs: 15 linhas `capacity_truncated`, ~100k evicções). Run A (serializado): 0 stalls >200ms. Run B (+`get_flow_metrics` concorrente a cada 30s, picos de 483ms): 0 stalls. Run C (B + 2% trades OOO): **1 stall de 768,94ms com `buffer_size=100000 | evict_5s=1075 | parquet_flush=False`**, 0 eventos >1500ms. Evidência bruta: `dados/audit/stress_dt02_20260918_125912.{json,log}`, `stress_dt02_conc.json`, `stress_dt02_ooo.{json,log}`.
  * **H1-strict (evicção ⇒ stalls, causa suficiente): REFUTADA em isolamento** — saturação+evicção sem o sistema completo não produz stalls (0 >1500ms em 630k trades). **Nenhum fix especulativo aplicado.**
  * **H1-weak (saturação+evicção como contexto contribuinte): sustentada** — o único stall instrumentado ocorreu com buffer cheio sob evicção intensa, e o top-3 da produção acopla a truncamentos em 0–5s.
  * **H2 (flush parquet como causa primária): REBAIXADA** — `parquet_flush=False` no stall instrumentado; só ~3 flushes/2h na produção.
  * Conclusão: stalls de produção exigem o sistema completo (custos sob lock amplificados por contenção GIL/CPU/I-O de WS, orderbook REST, SQLite WAL, ML). Investigação transferida para **DT-04**. DT-02 permanece **CRÍTICA/bloqueante** para expansão de escopo.
* **AÇÃO 2/18-09 — Guard anti-contaminação:** auditoria de escritores do `trading_bot.db` não encontrou nenhum script vivo que grave `data_context=historical` (final_validation usa test-DB; replay_etapa6 usa tmp). Criado `common/backfill_guard.py` (`is_live_bot_running` por argv[script]==main.py, excluindo self e menções em shell) + `enforce_no_live_bot()` com a mensagem padrão; fiado em `scripts/diagnostics/test_store_init.py` (padrão copiado por backfills). Cobertura: `tests/unit/test_backfill_guard.py` (6 testes, incluindo anti-auto-match) + teste vivo com stub `main.py` (bloqueio com RuntimeError exato; liberação após kill).
* **Vetor NÃO coberto pelo guard (explícito, sem falsa sensação de proteção):** `scripts/analytics/cftc_cot_shadow_collector.py` → tabela `cftc_cot_shadow_dataset` e `scripts/analytics/positioning_shadow_collector.py` → tabela `positioning_shadow_dataset` escrevem no `trading_bot.db` via `sqlite3` direto (sem `EventStore`, sem `backfill_guard`; o positioning roda como daemon de 5 min **concorrente** ao bot por desenho do `run_o1_shadow_observation.py`). Não contaminam `events` (tabelas dedicadas), mas acoplam no mesmo arquivo/WAL. Mitigação correta é estrutural (`trading_bot_historical.db` / DB de research separado — o coletor CFTC já aceita `--db`; o positioning precisa ganhar a flag), mantida como higiene futura. Até lá, este vetor segue **descoberto**.
* **AÇÃO 3/18-09 — Enforcement DT-02 no código:** `SUPPORTED_SYMBOLS = ["BTCUSDT"]` com bloqueio documentado em `config/settings.py` + comentário anti-expansão de detectores no `pipeline.detect_signals` (`window_processor.py`). Referência `DT-02` agora grepável em `settings.py`, `window_processor.py`, `core.py` e `backfill_guard.py`.

### DT-03 — Instrumentação de Profundidade de Fila (Queue Depth) e Backpressure no `AsyncTradeBuffer`

* **ID:** `DT-03`
* **Dono:** Equipe de Engenharia do Core / Ingestion Pipeline
* **Prioridade:** Média
* **Gatilho de Revisão:** Antes de campanhas de stress test de alta volatilidade (ex: CPI, FOMC).
* **Descrição Técnica:** O `AsyncTradeBuffer` possui thresholds reativos de status (`NORMAL`, `WARNING` a 80%, `CRITICAL` a 90%, `OVERFLOW` a 100%), mas não registra metricamente a série temporal contínua da profundidade instantânea da fila (`len(self._buffer)`) nem o pico máximo absoluto atingido durante rajadas de volume (como as de 404 trades/s e 318 trades/s registradas em 08/09).
* **Impacto:** Impossibilidade de mensurar com precisão matemática a margem real de folga do buffer antes do acionamento de descarte por backpressure em eventos de estresse agudo.
* **Critério Objetivo de Resolução:** Implementação do registro de `peak_buffer_size` e `buffer_fill_ratio` dentro das métricas de payload das janelas no SQLite e exposição contínua via gauge Prometheus `trades_buffer_queue_depth`, permitindo auditar o percentual exato de capacidade consumido em cada segundo de operação.

### DT-04 — Profiling de Stalls em Sistema Completo (causa raiz dos >1500ms de 18/09)

* **ID:** `DT-04`
* **Dono:** Equipe de Engenharia do Core / Performance
* **Prioridade:** Alta (abaixo de DT-02-Crítica; alimenta a resolução dela)
* **Gatilho de Revisão:** Antes de aplicar qualquer fix de contenção (evicção incremental, cópia fora do lock) — sem causa confirmada, sem fix especulativo.
* **Descrição Técnica:** A AÇÃO 1 (3 runs controlados, 630k trades, saturação+evicção+OOO+métricas concorrentes) produziu **0 eventos >1500ms** no `FlowAnalyzer` isolado, refutando evicção como causa suficiente. Os stalls de 1,5–4,4s da produção exigem o sistema completo. Candidatos não testados: pausas de GC do Python, stalls de checkpoint WAL do SQLite sob escrita concorrente, drenagem em lote do `AsyncTradeBuffer`, contenção GIL (WS + REST + flush + ML).
* **Critério Objetivo de Resolução:** Próxima sessão ao vivo com `gc.set_debug` estatístico + `py-spy` (ou `faulthandler.dump_traceback_later`) capturando stack do thread de ingestão durante stalls; correlação de stalls com checkpoints WAL (`PRAGMA wal_checkpoint`) e ciclos de GC. Hipótese a confirmar/refutar: amplificação ≥10x dos custos sob lock por contenção sistêmica.

---

## 5. Mudança Pós-Sign-Off Formalizada (2026-09-18)

* **Guarda de SLA de offset do orderbook** (`market_orchestrator/orderbook/orderbook_wrapper.py`, commit `c2df8fa`): snapshots `live_sync` com `snapshot_offset_ms > 1500ms` são descartados em favor do `cache_bg` (com `refresh_orderbook_async`); semântica de offset do fallback harmonizada para valor absoluto positivo + campo `cache_age_ms`. Mudança conservadora (aumenta fallback, nunca aceita dado estagnado como live). Cobertura: `tests/unit/test_orderbook_sync_snapshot.py::test_orderbook_sync_excessive_offset_triggers_fallback`. Validada na observação go-live de 18/09 (seção 4 do sign-off): 0 vazamentos >1500ms, p90 1272ms.
* **Nota V3/18-09:** 3.272 sinais `is_signal=1` sem bloco `orderbook_data` no DB completo datam de **08/09 18:09 UTC a 15/09** (3.271 `Absorção` slim + 1 `ANALYSIS_TRIGGER` variante iceberg de 11/09) — i.e., **fora** das janelas auditadas e **após** o fix do Item 4 (`e696ee2`, 05/09): emissores alternativos não cobertos pelo contrato do Item 4. Investigação dedicada pendente (novo débito proposto: cobertura de schema para o emissor slim de `Absorção` e variantes de detectores).
* **INV-A/18-09 — Emissor slim MORTO, sem Item 9:** (i) `events/event_stats_model.py::create_absorption_event` (legado, sem `orderbook_data`) tem **zero imports** no repo — só listagens de inventário citam o nome do arquivo; (ii) único `pipeline.detect_signals` produtivo (`window_processor.py:727`) sempre passa `orderbook_data=ob_event`; (iii) `TradeFlowAnalyzer.analyze_window` (sem orderbook) é instanciado mas **nunca invocado**; (iv) `orderbook_fallback` nunca retorna `None` (cache_bg/emergency/invalid); (v) **17/17 `Absorção real_time`** no DB (10 das sessões auditadas + 7 de 11–12/09) têm orderbook completo `live_sync` — os 3.271 incompletos são 100% `data_context=historical` (carimbado pelo saver para timestamps 2023, cadência bulk de ~109–436/h), i.e., **resíduo de replays/backfills gravados no DB de produção, não bug vivo**. Nenhum cron/scheduler ativo referencia replay com escrita (runners shadow são live-stream). Ação: higiene — seções de validação futuras devem filtrar `data_context='real_time'`; considerar DB separado para backfills.


