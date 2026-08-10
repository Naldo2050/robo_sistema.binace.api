# RELATÓRIO FINAL — OBSERVAÇÃO EM PRODUÇÃO 2026-08-09

**Contexto:** Rodada de validação (não desenvolvimento) — nenhuma correção aplicada durante a observação. Duração real: ~60 min (restrição de tempo do usuário; 2 runs de ~30 min).

---

## 1. Execução

| | Run 1 | Run 2 |
|---|---|---|
| Início | 2026-08-09 22:37:41Z (19:37 NY) | 2026-08-09 23:11:13Z (20:11 NY) |
| Fim | 2026-08-09 23:07:52Z | 2026-08-09 23:41:19Z |
| Duração | 30,2 min | 30,1 min |
| Trades processados (FlowAnalyzer) | 38.462 | 54.862 |
| Trades inválidos | 0 | 0 |
| CVD final | +8,837 | -6,162 |
| FlowAnalyzer flow_trades_count | 16.282 | 30.769 |
| OOO (FlowAnalyzer) | 0 | 0 |
| OOO (orchestrator, T_raw) | 0 | 0 |
| **timestamp_corrected_total (Prometheus)** | **0** | **0** |
| Reset forçado de CVD | — | 23:26:16Z (15 min de run) |
| Eventos gravados | ids 1-40 | ids 41-84 |

Harness: `scripts/diagnostics/run_production_observation.py` (boot idêntico ao main.py; NÃO altera código de produção). Resumo em `logs/observation_final_summary.json`; amostras de métricas a cada 60s em `logs/observation_stats.jsonl`.

---

## 2. PARTE A — Infraestrutura (SDK + conexão): **APROVADO**

1. **WebSocket real conectado e processando continuamente** ✓ — `ws_connected` no boot; 38.462 + 54.862 trades processados nos 2 runs, sem crash nem exceção de SDK; reconexão 0.
2. **SDK openai 2.26.0 ponta a ponta** ✓ — teste direto pré-run: `OpenAI(api_key=..., base_url=https://api.groq.com/openai/v1)`, model `openai/gpt-oss-120b`, resposta real ('TESTE', finish_reason=stop, usage 87 tokens), **sem erro de 'proxies'**. Na produção: "Groq client configured" e 11 eventos `AI_ANALYSIS` gerados sem erro.
3. **Warnings/errors novos?** Nenhum ERROR. Warnings esperados que se repetiram (não são novos): ClockSync offset degradado (+~1,9s), Twelve Data 404 TNX, coingecko SSL expirado, "ML hybrid DESABILITADO", "Janela com poucos trades", latência de trades (P95 ~1,8s).

---

## 3. PARTE B — Correções em tempo real

| # | Item | Status | Evidência |
|---|---|---|---|
| B.1 | `orchestrator_trades_timestamp_corrected_total` | **PASS** | 0 clamps nos 2 runs (baseline anterior: 11.967 trades / 0 clamps). Sem reconexão → sem clamps. |
| B.2 | `flow_analyzer_ooo_total` / OOO real (T_raw) | **PASS** | 0 OOO nos 2 runs (sem falsos positivos). |
| B.3 | Reset de CVD + warmup 300s + price_at_reset | **PASS (com observação)** | Reset forçado 23:26:16Z: fc.cvd caiu de **-24,30 → +2,30** no evento seguinte; `price_at_reset=64946,00` presente em **todos** os eventos pós-reset; CVD continuou acumulando corretamente. `cvd_div` nunca emitido no período — nos casos reais preço e CVD estiveram alinhados (ex.: ev 68 cvd=-30,8, preço caiu → aligned). **Observação:** `last_reset_ms` AUSENTE no `fluxo_continuo` dos ANALYSIS_TRIGGER (presente em eventos de sinal Absorção/Exaustão) → no caminho de trigger a Camada-1 de supressão (elapsed<300s) é pulada; a Camada-2 (price_at_reset) compensou e não gerou falso sinal. Registrar para prompt separado. |
| B.4 | Absorção: `resultado_da_batalha` vs `fluxo_continuo.absorption_analysis.label` | **PASS (1 evento)** | Ev 7: delta=+2,57, preço estável → evento: "Absorção de Compra" + descrição "Agressão compradora absorvida" (convenção correta ✓); flow_analyzer: "Neutra" (classificador próprio, índice 0,039 < threshold 0,6 — sem contradição de rótulo). Só ocorreu 1 evento real de absorção na janela. **Anomalia:** `events/event_stats_model.py` linhas 267-270 ainda tem as descrições TROCADAS — caminho morto (produção usa `data_handler`), registrar. |
| B.5 | Pivots usam período anterior COMPLETO | **NÃO — ANOMALIA CONFIRMADA** | O caminho de produção do `event.pivot_points` é `enricher._build_pivot_points` → `historical_vp.daily` (VAH/VAL/POC como H/L/C) → `HistoricalVolumeProfiler.update_profiles` (`market_analysis/historical_profiler.py:222`), que calcula o VP daily com klines 1m **de 00:00 UTC de HOJE até agora — o DIA CORRENTE PARCIAL**, não o dia anterior completo. Weekly = janela rolante 7 dias; monthly = janela rolante 30 dias (não períodos de calendário). **Confirmado com recálculo exato:** VP de 2026-08-09 00:00Z→23:22Z reproduz POC=64930, VAH=65294, VAL=64845 — match perfeito com os eventos (e por isso o pivot não bate com nenhuma vela 1d Binance: usa dados de hoje, não de ontem). O fix `iloc[-2]` existe em `support_resistance/pivot_points.py` (`calculate_multi_timeframe_pivots`, usado pelo MultiTFUnifiedFeatureBuilder) — caminho PARALELO que **não** alimenta `event.pivot_points`. **Registrar para prompt separado.** |
| B.6 | VAH/VAL região contígua | **PASS** | Ev 69: top.vah=64887,75 ≥ poc=64879,11 ≥ val=64853,17 ✓. historical_vp daily: VAH=65294 ≥ POC=64930 ≥ VAL=64845 ✓. |

---

## 4. PARTE C — Validação cruzada (trades brutos vs eventos)

Execução manual + `scripts/diagnostics/validate_production_run.py` (criado; sintaxe OK; roda com `python scripts/diagnostics/validate_production_run.py [ids]`).

| Evento | delta gravado | delta recalculado (aggTrades) | diff % | CVD gravado (desde reset) | CVD recalculado | diff % |
|---|---|---|---|---|---|---|
| Ev 7 (Absorção, run1, janela 22:42:01–22:42:59) | **+2,56933** | **+2,81419** (n=~50) | **9,5%** | -93,02837 | **-90,26029** | **2,98%** |
| Ev 58 (Exaustão, run2) | -3,53901 | +0,26847 (fetch truncado) | — | -29,58272 | +3,96991 (fetch truncado) | — |
| Ev 69 (Exaustão, run2, pós-reset) | — | — | — | -36,76758 | — | — |

**Metodologia:** GET `/api/v3/aggTrades` (agregados; 'm' = buyer is maker, mesmo do stream @trade). Agregação mescla trades de mesmo preço → pequenas divergências esperadas vs trades individuais.
**Limitação encontrada:** paginação time-based parou com HTTP 400 no ev 58 (cursor ~23:18Z); paginação by_id retornou volumes irrealistas (57k em 5 min vs ~9k esperados). Evidência sólida apenas no ev 7: delta 9,5% e CVD 2,98% de divergência — aceitável pela metodologia, mas NÃO é possível fechar a auditoria de CVD com precisão < 1% sem os trades individuais do WS persistidos (hoje não são). **Registrar para prompt separado: persistir trades brutos das janelas.**

---

## 5. Anomalias NÃO esperadas

1. **[B.5] VP "histórico" é o dia corrente parcial** — `historical_profiler.py:222-237`: daily = 00:00 UTC→agora; weekly = rolante 7d; monthly = rolante 30d. Pivot diário dos eventos é derivado desse VP corrente parcial (match exato reproduzido). Não cumpre "período anterior completo". O fix iloc[-2] está em caminho paralelo que não alimenta `event.pivot_points`. Durante o dia o VP "drift" conforme dados acumulam; à 00:00 UTC seria degenerado (flag insufficient_data → fallback ATR).
2. **`last_reset_ms` ausente no fluxo_continuo dos ANALYSIS_TRIGGER** — enfraquece a supressão Camada-1 do cvd_div no caminho de trigger (Camada-2 compensou nesta rodada).
3. **`events/event_stats_model.py` (dead code) com descrição trocada** — "Absorção de Compra" emparelhada com "Agressão vendedora absorvida" (linhas 268-270). Produção usa data_handler (correto).
4. **Métricas Prometheus do FlowAnalyzer parcialmente mortas**: `flow_analyzer_cvd`, `flow_analyzer_trades_total`, `flow_analyzer_trades_invalid_total` NUNCA atualizados em `flow_analyzer/core.py` (só `record_ooo` é chamado). `flow_analyzer_cvd` ficou 0.0 durante toda a observação. Não é regressão — registrar.
5. **`dados/eventos-fluxo.json.legacy` deletado** (working tree sujo, `git rm` pendente — estado pré-existente).
6. **Harness (não-produção):** 1ª rodada do stats loop morreu no 1º minuto (asyncio.wait_for sem try) — corrigido; `observation_final_summary.json` do run 1 reportou `timestamp_corrected_total` errado (sample `_created`) — corrigido no run 2 (0,0 ✓).

---

## 6. Respostas às perguntas do relatório

1. Duração real: **60,3 min** (2 runs de ~30 min) — abaixo dos 60-90 min ideais (restrição de tempo); reset forçado manual aos 15 min do run 2 cumpriu o papel do ciclo de 4h; fase inicial do ciclo registrada (boot = início do ciclo).
2. Parte A: **SIM** ✓ com evidência de log (SDK 2.26.0, Groq, WS real).
3. Parte B: 4/5 itens PASS (B.3 com ressalva, B.4 pontual); **B.5 NÃO validado — anomalia confirmada e explicada** (VP do dia corrente parcial).
4. Parte C: 1/3 eventos validado com divergência < 10% (delta) e < 3% (CVD); 2/3 bloqueados por falha de paginação da API/limite de tempo.
5. Anomalias não esperadas: §5 (6 itens, todos registrados para prompts separados).

**Arquivos desta rodada:** `scripts/diagnostics/run_production_observation.py`, `scripts/diagnostics/validate_production_run.py`, `logs/observation_stats.jsonl`, `logs/observation_console*.log`, `logs/observation_final_summary.json`, `dados/eventos_fluxo.jsonl`, `dados/trading_bot.db` (84 eventos).
