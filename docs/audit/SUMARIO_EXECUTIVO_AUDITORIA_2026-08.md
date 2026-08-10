# SUMÁRIO EXECUTIVO — AUDITORIA DE CORREÇÃO MATEMÁTICA 2026-08

> **Período**: 2026-08-06 a 2026-08-09
> **Escopo**: módulos que extraem/calculam dados de mercado para análise da IA (flow_analyzer, market_orchestrator, institutional/enricher, support_resistance, market_analysis, payload)
> **Resultado**: 9 bugs de produção corrigidos (6 de corrupção silenciosa de sinal + 1 de observabilidade), 3 itens validados como corretos com stream real, 1 bloco de dead code mapeado, 27 commits
> **Status**: rodada encerrada — backlog remanescente priorizado em §4

---

## 1. ESCOPO

Auditoria de **correção matemática** dos módulos que extraem ou calculam dados de
mercado que alimentam a análise da IA: sinais de fluxo (CVD, flow imbalance,
absorção, OOO), níveis de preço (pivots clássicos, Value Area, volume profile),
e o payload compacto que a IA recebe (`build_compact_payload`). Inclui a
validação empírica em produção (2 observações ao vivo em 2026-08-09 e
2026-08-10) e o encerramento formal com validação pós-fix.

Não faz parte desta rodada: auditoria de arquitetura/imports (2026-03-31,
`RELATORIO_FINAL.md`), refactors de pacotes (início de 2026-08-06, commits
`772f755`–`e8858c5`).

---

## 2. METODOLOGIA

Para cada item auditado, o mesmo fluxo de 5 etapas:

1. **Confirmação de causa raiz com evidência de código** — leitura do caminho
   real de produção (ex: `market_orchestrator.py:1460` como dead wire), nunca
   por suposição; documentado em `docs/audit/AUDITORIA_PIVOT_POINTS_2026-08-09.md`
   e seções novas de `FASE3_ERROS_RESILIENCIA.md` / `FASE5_AI_ML_PIPELINE.md`.
2. **Teste numérico sintético** — scripts em `scripts/diagnostics/`
   (`audit_cvd_numeric_test.py`, `audit_cvd_reset_divergence_test.py`,
   `audit_absorption_*`, `audit_ofi_numeric_test.py`, `audit_va_poc_outward.py`)
   reproduzem o cenário do bug antes de corrigir.
3. **Correção mínima** — o menor patch que elimina a causa raiz, sem mudar
   contrato de dados da IA (ex: `source` novo no `pivot_points`, preservação de
   `cvd_div` `{det, type}`).
4. **Validação com testes automatizados** — suítes `tests/unit/`,
   `tests/payload/` (265 passed), `tests/integration/` e novos testes de
   regressão (ex: `test_signal_direction_absorption.py`).
5. **Validação empírica com stream real de produção** —
   `scripts/diagnostics/run_production_observation.py` (boot idêntico ao
   main.py) + `validate_production_run.py`: 2 runs de ~30 min em 2026-08-09
   (60,3 min, 93.324 trades, 0 OOO, 0 clamps) e 1 run de 35,1 min em
   2026-08-10 pós-fix pivots (47.431 trades, 15/15 eventos `source=classic`).

---

## 3. TABELA CONSOLIDADA — 13 ITENS AUDITADOS

### 3.1 Bugs corrigidos (9)

| # | Item | Causa raiz | Fix | Commit |
|---|---|---|---|---|
| 1 | **Timestamp OOO mascarado** no `on_message` | `market_orchestrator.py:741` descartava `T_raw` do trade; OOO invisível e contaminação latente | `T_raw` preservado + observabilidade (`_ooo_trades_count`, Prometheus) | `b93c714` |
| 2 | **Falso sinal de divergência CVD pós-reset** | Reset 4h zerava CVD; fallback comparava CVD zerado com trend 1h → `bearish_div` falso `src=inferred` | 3 camadas: warmup 300s, comparação com `price_at_reset`, supressão < 10s (`CVD_DIV_WARMUP_SECONDS`, `CVD_DIV_MIN_PERIOD_SECONDS`) | `9169a13` |
| 3 | **flow_imbalance extremo por amostra insuficiente** | Janelas com poucos trades geravam sinal extremo por ruído estatístico | Supressão quando `< FLOW_IMBALANCE_MIN_TRADES` | `1d49f17` |
| 4 | **Inversão de label de absorção + `signal_direction`** | Caminho B (`absorption.py`) usava convenção oposta; IA recebia label errado; `signal_direction` nunca capturava "Absorção de Venda"→long | Unificação da convenção (lado absorvido) em 4 arquivos + frozenset `BULLISH_RESULTS` | `c3e5062` |
| 5 | **Pivots clássicos usavam vela ATUAL parcial** | `iloc[-1]` = período em andamento (OHLC parcial) | `iloc[-2]` = período anterior completo | `75bd3ec` |
| 6 | **Value Area sem região contígua** | VA calculado por ordenação de nodes, não por contiguidade a partir do POC | Método POC-outward (região contígua) + scripts de auditoria | `b9a644d` + `5d92765` |
| 7 | **Direção do whale e TTL do macro** | Direção derivada errada (não `buy_pct`); macro `all_macro` desalinhado ao bloco 900s | `buy_pct` para direção do whale; TTL alinhado ao bloco 900s com force; `multi_tf`/VP/orderbook preservados no trigger sem sinal | `7a1f7cf` |
| 8 | **Dead wire do pivot classic + fallback VP parcial rotulado de clássico** (duplo bug) | `market_orchestrator.py:1460` lia `contextual_snapshot.pivots` (inexistente); `enricher._build_pivot_points` usava VP intraday parcial de `historical_profiler.py:229` (00:00Z→agora) como se fosse pivot clássico | Wiring `macro_context.get("pivots")` + `signal["pivots"]`; enricher com fonte clássica primária (`source: classic`), fallback VP marcado (`vp_fallback`/`multi_tf_fallback`); OHLC propagado de `daily_pivot` (0% drift); `calculated_at_ms` | `aa97cf1` |
| 9 | **Métricas Prometheus do FlowAnalyzer nunca atualizadas** | Observabilidade: infra de exposição OK (REGISTRY correto, `/metrics` servido), mas dead-wire de ATUALIZAÇÃO — só `record_ooo()` era chamado em `flow_analyzer/core.py`; `set_cvd`/`set_whale_delta`/`set_flow_trades_count`/`record_trade` nunca invocados; 2ª instância de `PrometheusMetrics()` degrada para `_prometheus=None` (try/except do construtor engole `ValueError: Duplicated timeseries`) | Wiring mínimo em `process_trade` (guard `if self._prometheus is not None`): setters de CVD/whale/flow_trades_count + `record_trade` (válido/inválido); teste `test_flow_analyzer_metrics.py` com cleanup de REGISTRY; 39 unit + 4 integração + 24 REGISTRY PASS | `ab0ad8a` |

### 3.2 Itens validados como corretos (3)

| # | Item | Evidência | Commit |
|---|---|---|---|
| 9 | **CVD: sinal e acumulação matematicamente corretos** | `audit_cvd_numeric_test.py` PASS; ev 7 real: delta 9,5% / CVD 2,98% de divergência vs recálculo aggTrades (limite metodológico ~3-10%) | `c41f84c` |
| 10 | **VAH/VAL região contígua em produção** | Observação 2026-08-09 B.6: ev 69 `top.vah ≥ poc ≥ val` e `historical_vp` VAH ≥ POC ≥ VAL ✓ | `b9a644d` (validado ao vivo) |
| 11 | **Zero OOO / zero clamps em produção** | 2 runs (93.324 trades): 0 OOO, 0 clamps (baseline anterior 11.967 trades/0 clamps); medição live 901s + 16 testes | `b93c714` (validado ao vivo) |

### 3.3 Dead code mapeado (1)

| # | Item | Descoberta | Registro |
|---|---|---|---|
| 12 | **13 módulos `institutional/` + caminhos mortos da IA** | `InstitutionalEventBridge` nunca instanciado no live (só em `test_architecture_regressions.py:62`); 13 módulos (garch, hurst, kalman, monte_carlo, fourier, HMM, smart_money, whale, iceberg, footprint, mean_reversion, entropy, confluence) + `order_flow_imbalance.py` e `event_stats_model.py` (descrições trocadas, caminho morto). Auditoria de pivots revelou mais: `build_ai_input` NUNCA chamado em produção, `payload_compressor_v3` só em testes/fallback, `contextual_snapshot.pivots` inexistente, `features/multi_tf_feature_builder.py` NÃO EXISTE, `calculate_multi_timeframe_pivots` só roda em testes/health_check | `003a3e7` + seções FASE5 + auditoria pivots |

**Total**: 9 bugs corrigidos + 3 validados + 1 bloco de dead code mapeado = **13 itens**.

---

## 4. BACKLOG REMANESCENTE (priorizado)

| # | Prioridade | Item | Detalhe | Estado |
|---|---|---|---|---|
| 1 | 🔴 ALTA | **ML: BB `ddof` + NaN fill (bloqueante para `HYBRID_ENABLED=True`)** | `feature_calculator.py:153` usa `np.std()` (ddof=0) vs treino/inferência ddof=1 → bandas divergem 0,2-2%; treino faz `fillna(median)` e inferência manda NaN nativo ao XGBoost. Modelo atual tem 9 amostras (AUC sem significado) — religar exige dataset ≥ 500 + retreino | Documentado `ed85946`; corrigir junto do próximo ciclo de retreino |
| 2 | 🟠 MÉDIA | **`institutional/`: 13 módulos + OFI institucional — integrar ou remover** | Dead code (item 12); se o bridge for plugado ao live, os 13 módulos exigem auditoria matemática prévia; `enricher.py` é o único ativo e fica | Decisão pendente (`003a3e7`) |
| 3 | 🟠 MÉDIA | **`last_reset_ms` ausente em `ANALYSIS_TRIGGER`** | Presente em eventos de sinal (Absorção/Exaustão) mas não no `fluxo_continuo` dos triggers → supressão Camada-1 do `cvd_div` é pulada no caminho de trigger (Camada-2 compensou na observação) | Registrado no `RELATORIO_OBSERVACAO_2026-08-09` §3-B.3/§5.2 |
| 4 | 🟡 BAIXA | **Contaminação histórica potencial em `signal_outcomes`** | Labels de absorção invertidos desde `d181947` (out/2025); banco local `signal_outcomes` VAZIO (0 registros) — nada a migrar localmente; produção externa pode ter dados afetados | Query de diagnóstico pendente (`SELECT event_type, battle_result, COUNT(*) FROM signal_outcomes GROUP BY 1,2 ORDER BY 3 DESC`); decisão de descarte em backlog (FASE5) |

### Backlog: verificar exclusividade de threshold em bucketing de qty

- Observado durante debug do teste de métricas Prometheus
  (`test_flow_analyzer_metrics.py`): qty 0.5 cai no bucket "mid"
  por limite exclusivo (não inclusivo)
- Verificar se essa é a classificação correta esperada
  (ex: threshold de whale/retail) ou se há inconsistência de
  borda semelhante à já corrigida no OFI institucional
  (`_window_initialized`)
- Não bloqueante — apenas confirmar intencionalidade

**Não-bloqueante** (registrado, fora do escopo): `except Exception:` silenciosos
(TECH_DEBT 2026-08-04), persistência de trades brutos por janela (limitação da
Parte C da observação), `r1`/`s1` clássicos sem `round` no ramo classic do
enricher (cosmético, sem impacto funcional).

---

## 5. COMMITS DA RODADA (ordem cronológica, 2026-08-06 a 2026-08-09)

| Hash | Tipo | Descrição |
|---|---|---|
| `772f755` | refactor | move `institutional_enricher` → `institutional/enricher.py` |
| `87564e4` | refactor | consolida `orderbook_analyzer` em pacote, elimina shim importlib |
| `80b03e3` | chore | remove resíduos Fase 1 (config.py proxy morto, .bak, coverage) |
| `106b4b7` | refactor | move `build_compact_payload` → `market_orchestrator.ai.payload_builder_compact` |
| `d92c02f` | refactor | move `ai_analyzer_qwen` → `market_orchestrator.ai.analyzer_qwen` |
| `e8858c5` | refactor | elimina camada proxy `src/` e sys.path hacks |
| `7a1f7cf` | **fix** | paraleliza fetches macro (timeout 8s), TTL `all_macro` alinhado ao bloco 900s com force, preserva multi_tf/VP/orderbook no trigger sem sinal, corrige direção do whale pelo `buy_pct` |
| `bea30fe` | feat | `init_throttler` fonte única de config; preserva `current_price` pós-compressão v2; run.log rotativo + saneamento BOM do issues.log |
| `5328688` | docs(audit) | seção 11 (fixes validados ao vivo), script `data_health_check`, backups legados |
| `b93c714` | **fix** | preservar `T_raw` + observabilidade OOO |
| `7891d0d` | docs(audit) | registrar correção de timestamp OOO (b93c714) |
| `fe81d0c` | chore(diagnostics) | script de medição de trades out-of-order |
| `3dbf42c` | chore(gitignore) | caches de runtime e artefatos de teste |
| `c41f84c` | chore(diagnostics) | teste numérico de validação do CVD (sinal correto, PASS) |
| `9169a13` | **fix** | falso sinal de divergência CVD pós-reset |
| `1d49f17` | **fix** | suprimir flow_imbalance com amostra insuficiente |
| `c3e5062` | **fix** | unificar convenção de rótulo de absorção e corrigir `signal_direction` |
| `4369171` | docs(audit) | registrar inversão de label de absorção e signal_direction (FASE5) |
| `003a3e7` | docs(audit) | registrar 13 módulos institutional/ como dead code |
| `75bd3ec` | **fix** | pivots usam `iloc[-2]` (período anterior completo) em vez de `iloc[-1]` |
| `ed85946` | docs(audit) | registrar defeitos ML como bloqueantes para HYBRID_ENABLED |
| `b9a644d` | **fix** | Value Area método POC-outward (região contígua) |
| `5d92765` | chore(diagnostics) | scripts de auditoria OFI e Value Area (regressão futura) |
| `aa97cf1` | **fix** | dead wire do pivot classic + fallback VP rotulado (duplo bug de pivot_points) |
| `3986822` | chore(housekeeping) | remove backup legado `eventos-fluxo.json.legacy`; ignora `logs/observation_*` |
| `1431cf5` | docs(audit) | validação pós-fix de pivot_points em produção real |
| `ab0ad8a` | **fix** | wiring das métricas Prometheus do FlowAnalyzer (CVD/whale/flow_trades_count/trades_total/invalid) + teste de regressão `test_flow_analyzer_metrics.py` |

**Legenda**: `**fix**` = correção de bug de produção; demais = refactor/chore/docs de suporte à rodada.

---

## 6. FECHAMENTO

- **Validação final em produção (2026-08-10, pós-fix pivots)**: 35,1 min ao vivo,
  47.431 trades, 0 OOO, 0 clamps; **15/15** eventos ANALYSIS_TRIGGER com
  `source="classic"`; pivot **estável** em `65035.38` (1º vs último evento);
  `calculated_at_ms` com intervalos 306s/307s (ciclo de 300s do
  `CONTEXT_UPDATE_INTERVAL_SECONDS`); divergência vs `(H+L+C)/3` da vela 1d
  anterior (API Binance) = **0,0%**; `validate_production_run.py` B.6 **PASS**
  (era WARN pré-fix).
- **Métricas Prometheus do FlowAnalyzer** (`ab0ad8a`): CVD, whale_delta,
  flow_trades_count, trades_total e trades_invalid_total agora são atualizados
  a cada `process_trade` (antes só `record_ooo`); validado por
  `test_flow_analyzer_metrics.py` (3 testes lendo o REGISTRY padrão) + suítes
  unit/integração (39 + 4 + 24 PASS). Nota: 2ª instância de
  `PrometheusMetrics()` no mesmo processo degrada para `_prometheus=None`
  (try/except no construtor) — investigação futura se houver múltiplos bots.
- **git status**: limpo ao final da rodada.

**Documentos da rodada**: `AUDITORIA_PIVOT_POINTS_2026-08-09.md`,
`RELATORIO_OBSERVACAO_2026-08-09.md`, `FASE3_ERROS_RESILIENCIA.md` (seção OOO),
`FASE5_AI_ML_PIPELINE.md` (seções 2026-08-08/09), este sumário.
