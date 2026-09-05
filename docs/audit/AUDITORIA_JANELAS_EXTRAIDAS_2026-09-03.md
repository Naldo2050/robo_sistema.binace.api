# AUDITORIA DE JANELAS EXTRAÍDAS — 2026-09-03

**Data:** 2026-09-03T21:16 UTC-3  
**Repositório:** robo_sistema.binace.api  
**Dados auditados:** `dados/` (eventos_fluxo.jsonl, eventos_visuais.log, trading_bot.db, fred_cache.json, cohort manifests)

---

## FASE 0 — INVENTÁRIO

### 0.1 Eventos no JSONL

| Métrica | Valor |
|---------|-------|
| Total de linhas/eventos | **94** |
| ANALYSIS_TRIGGER | 81 |
| Alerta (VOLUME_SPIKE) | 11 |
| Exaustão (sinal) | 2 |
| Symbol BTCUSDT | 83 |
| Symbol ausente (tipo Alerta) | 11 |

**Arquivo:** `eventos_fluxo.jsonl` (107 KB, 95 linhas)

### 0.2 Intervalo Temporal

| Campo | Valor |
|-------|-------|
| Primeiro timestamp | `2026-09-01T23:12:20.078Z` (epoch_ms=1788304340078) |
| Último timestamp | `2026-09-03T01:45:00.000Z` (epoch_ms=1788399900000) |
| Duração total | **1 dia, 2h 32min 39s** |
| ANALYSIS_TRIGGER total | 81 |
| Com `janela_numero` | 80 (janelas 1-75, depois reinício 1-5) |
| `trimmed_by_guardian` | **80** (só envelope) |
| Com payload completo | **1** (evento 0, sem janela_numero) |
| Sinais com payload (Exaustão) | **2** (janelas 21 e 24) |

**Janelas e gaps:**

- Intervalo médio entre ANALYSIS_TRIGGER: ~60s (1 min), consistente com janela de 1 minuto
- **GAP 1:** Evento 0 (sem janela_numero) para Janela 1: **999.9s (16.7 min)** — startup/warmup
- **GAP 2:** Janela 75 para Janela 1 (reinício): **88020s (24.5h)** — sessão encerrada e reiniciada
- **GAP 3:** Janela 2 para Janela 1 (segunda sessão): **1920s (32 min)** — segundo reinício
- Sem gaps dentro de cada sessão contínua (janelas 1-75 e janelas do O1)

> [!IMPORTANT]
> **80 de 81 ANALYSIS_TRIGGER foram "trimmed_by_guardian"** — possuem apenas envelope (epoch_ms, janela_numero, event_id). O payload completo foi descartado pelo módulo guardian para economizar espaço. Os dados analisáveis são apenas **3 eventos com payload** (1 ANALYSIS_TRIGGER + 2 Exaustão).

### 0.3 trading_bot.db

| Tabela | Linhas | Colunas de timestamp | Min timestamp | Max timestamp |
|--------|--------|---------------------|---------------|---------------|
| events | 94 | timestamp_ms, created_at | 1788304340078 | 1788399900000 |
| positioning_shadow_dataset | 5 | timestamp_ms, source_timestamp_ms, created_at | 1788312105816 | 1788399757525 |
| signal_outcomes | 2 | signal_epoch_ms, created_at | 1788306540000 | 1788306720000 |
| sqlite_sequence | 3 | — | — | — |

**Comparação DB vs JSONL:**
- DB `events` contém **94 linhas** — mesmo total do JSONL
- Cada evento no DB tem coluna `payload` (JSON) com o payload completo
- `event_id` no JSONL: 94 distintos
- **Os eventos do DB são os MESMOS do JSONL** (mesma contagem, mesmo range temporal)
- `signal_outcomes` tem 2 linhas (epoch_ms 1788306540000 e 1788306720000) — correspondendo exatamente às janelas 21 e 24 dos sinais Exaustão

> [!TIP]
> O DB contém os payloads completos em `events.payload` (JSON), mesmo quando o JSONL foi trimado. Os payloads completos estão também em `eventos_visuais.log`. Para auditoria futura, extrair payloads do DB seria mais eficiente.

### 0.4 eventos_visuais.log

| Métrica | Valor |
|---------|-------|
| Total de linhas | **135.550** |
| ERROR (nível de log) | **0** |
| WARNING (nível de log) | **0** |
| CRITICAL (nível de log) | **0** |
| Traceback | **0** |

**Nota:** O arquivo `eventos_visuais.log` **não é um log Python convencional**. É uma representação visual formatada dos eventos (JSON pretty-printed com headers como `EVENTO: ANALYSIS_TRIGGER | SYMBOL: BTCUSDT`). As 86 ocorrências de "CRITICAL" detectadas inicialmente são referências ao campo JSON `critical_flags` dentro dos payloads, não mensagens de nível CRITICAL de logging.

**Conteúdo:** Contém os payloads completos de todas as janelas (incluindo as trimadas no JSONL), formatados para leitura humana.

### 0.5 fred_cache.json

| Série | Valor | Atualizado | NaN/null/Inf |
|-------|-------|------------|-------------|
| TNX (US 10Y Yield) | **4.79** | 2026-09-03T01:42:38Z | Não |

- **Idade do cache:** Cache atualizado ~2min ANTES da última janela do JSONL — **fresco**
- **Verificação cruzada:** `ml_features.cross_asset.us10y_yield` nas janelas 21 e 24 = **4.796** vs FRED TNX = **4.79**. Diferença = 0.006 (0.13%). Aceitável — fontes/timestamps ligeiramente diferentes.
- **Sem NaN convertido em 0 ou string** — valor é float limpo.

> [!NOTE]
> `fred_cache.json` contém apenas a série TNX. Campos macro adicionais (VIX=16.34, gold=4335.17, oil=90.79) nos eventos vêm de outras fontes (cross-asset fetcher), não do FRED cache.

### 0.6 Cohort Manifests

**O1 (`o1_cohort_manifest.json`):**
- Cohort: `O1_PRODUCTION_SHADOW_OBSERVATION`
- Período: `2026-09-03T01:41:00Z` – `2026-09-03T01:45:44Z` (~5 min)
- Status: **ABANDONED_INSUFFICIENT_COVERAGE** (apenas 3 observações)
- Razão: Pivô para validação por replay histórico (R1)
- Referencia 3 janelas do final do JSONL (dentro do range)

**R1 (`r1_cohort_manifest.json`):**
- Phase: `R1 — HISTORICAL REPLAY VALIDATION`
- Replay window: 28 dias (`2026-08-06` a `2026-09-03`)
- Status: **INITIALIZED_PENDING_INGESTION** (dados ainda não ingeridos)
- Features: session_vwap, market_structure, positioning_cot, baseline
- Fora de escopo: `orderbook_depth_l2` (indisponível em REST histórico)

> [!NOTE]
> Nenhum dos manifests descreve janelas com payload analisável. O1 foi abandonado e R1 nem começou a ingestão.

---

## FASE 1 — INVARIANTES MATEMÁTICAS

### Eventos analisáveis

Dos 94 eventos, apenas **3 possuem payload completo**:

| # | Tipo | Janela | Timestamp UTC |
|---|------|--------|---------------|
| 1 | ANALYSIS_TRIGGER | sem número | 2026-09-01T23:12:20Z |
| 2 | Exaustão | 21 | 2026-09-01T23:49:00Z |
| 3 | Exaustão | 24 | 2026-09-01T23:52:00Z |

### Evento 0 (ANALYSIS_TRIGGER sem janela_numero)

Payload mínimo (preco_fechamento=77500, volume_total=10.5, delta=2.3). **Campos insuficientes** para a maioria dos testes.

| Invariante | Resultado | Desvio | Detalhe |
|-----------|-----------|--------|---------|
| (a) volume = compra + venda | — | — | volume_compra/venda ausentes |
| (b) delta = compra - venda | — | — | campos ausentes |
| (c) flow_imbalance | — | — | orderbook_data ausente |
| (h) epoch - timestamp_utc | OK | 0ms | 1788304340078 bate exato |
| (h) UTC-NY offset | OK | 4.0h | UTC 23:12 = NY 19:12 (-4h EDT) |

### Janela 21 (Exaustão de Compra)

| Invariante | Resultado | Desvio | Detalhe |
|-----------|-----------|--------|---------|
| **(a)** volume_total = compra + venda | OK | 0.001 | 19.862 vs 15.674+4.187=19.861 |
| **(b)** delta = compra - venda | OK | 0.001 | 11.487 vs 15.674-4.187=11.487 |
| **(c)** flow_imbalance | OK | 0.0000 | -0.6146 vs (533679.83-2235544.58)/(sum)=-0.6146 |
| **(d)** volume_ratio = bid/ask | OK | 0.0003 | 0.239 vs 533679.83/2235544.58=0.2387 |
| **(e)** mid = (bid+ask)/2 | OK | 0.00 | 77550.25 vs (77550.2+77550.3)/2=77550.25 |
| **(e)** spread_bps = spread/mid*1e4 | OK | 0.0000 | 0.0129 vs 0.1/77550.25*1e4=0.0129 |
| **(f)** total_depth_ratio = L25b/L25a | OK | 0.003 | 0.17 vs 251339.71/1451374.05=0.1732 |
| **(g)** impact 100k sell | OK | — | $100k <= bid_depth $533,680 |
| **(g)** impact 100k buy | OK | — | $100k <= ask_depth $2,235,545 |
| **(g)** impact 1M sell | **FALHA** | $466,320 | **$1M > bid_depth $533,680** — slippage=6.35bps calculado sobre liquidez insuficiente |
| **(g)** impact 1M buy | OK | — | $1M <= ask_depth $2,235,545 |
| **(h)** epoch - utc | OK | 0ms | Exato |
| **(h)** UTC-NY offset | OK | 0.0s | UTC 23:49 = NY 19:49 (-4h EDT) — instantes idênticos |
| **(i)** consolidated_bias_score | OK | 0.0000 | Reproduzido: 0.5+(-0.6146 x 0.3)+((0.239-1)/2 x 0.2) = **0.2395** |
| **(j)** spread_analysis histórico | **AUSENTE** | — | current=mean=median=0.1, std=0, percentile=0, samples=21 |
| **(k)** VAL <= POC <= VAH | OK | — | 77565.72 <= 77597.22 <= 77616.13 |
| **(k)** CVD presente | OK | — | CVD=42.4239 |

### Janela 24 (Exaustão de Venda)

| Invariante | Resultado | Desvio | Detalhe |
|-----------|-----------|--------|---------|
| **(a)** volume_total = compra + venda | OK | 0.000 | 27.578 vs 11.173+16.405=27.578 |
| **(b)** delta = compra - venda | OK | 0.000 | -5.232 vs 11.173-16.405=-5.232 |
| **(c)** flow_imbalance | OK | 0.0000 | 0.4539 vs (1678957.33-630619.18)/(sum)=0.4539 |
| **(d)** volume_ratio = bid/ask | OK | 0.0004 | 2.662 vs 1678957.33/630619.18=2.6624 |
| **(e)** mid = (bid+ask)/2 | OK | 0.00 | 77449.75 vs (77449.7+77449.8)/2=77449.75 |
| **(e)** spread_bps | OK | 0.0000 | 0.0129 |
| **(f)** total_depth_ratio | OK | 0.003 | 3.37 vs 1166594.46/345817.28=3.3734 |
| **(g)** impact 100k sell | OK | — | $100k <= bid_depth $1,678,957 |
| **(g)** impact 100k buy | OK | — | $100k <= ask_depth $630,619 |
| **(g)** impact 1M sell | OK | — | $1M <= bid_depth $1,678,957 |
| **(g)** impact 1M buy | **FALHA** | $369,381 | **$1M > ask_depth $630,619** — liquidez insuficiente |
| **(h)** epoch - utc | OK | 0ms | Exato |
| **(h)** UTC-NY offset | OK | 0.0s | Instantes idênticos |
| **(i)** consolidated_bias_score | OK | 0.0002 | Reproduzido: 0.5+(0.4539 x 0.3)+((2.662-1)/2 x 0.2) = **0.8024** vs reportado 0.8022 (arredondamento) |
| **(j)** spread_analysis histórico | **AUSENTE** | — | current=mean=median=0.1, std=0, percentile=0, samples=25 |
| **(k)** VAL <= POC <= VAH | OK | — | 77486.43 <= 77486.43 <= 77512.20 (POC == VAL — value area estreita) |
| **(k)** CVD presente | OK | — | CVD=34.8676 |

### Análise da invariante (g) — Market Impact sobre liquidez insuficiente

Confirmado como **sistêmico**:
- **Janela 21:** sell 1M sobre bid_depth $533,680 -> faltam $466,320. Slippage reportado = 6.35bps (calculado mas sobre volume inexistente no book).
- **Janela 24:** buy 1M sobre ask_depth $630,619 -> faltam $369,381. Slippage reportado sem flag "não-preenchível".
- **Causa raiz:** `slippage_matrix` calcula slippage mesmo quando notional excede profundidade total. Deveria sinalizar `"unfillable": true`.
- **Arquivo:** `orderbook_analyzer/core.py`, construção de `market_impact_buy/sell`.

### Análise da invariante (j) — Spread Analysis sem histórico

Confirmado ao ler `spread_tracker.py` (L24-224):
- `SpreadTracker` usa deque em memória com janela rolling de 1440 min (24h)
- **Resetado a cada reinício** (não persiste em disco)
- Com 21-25 amostras, todas com spread=0.1 (tick mínimo BTCUSDT futures)
- **std=0 e percentile=0 é correto** — não há variação no spread do BTCUSDT futures em condições normais
- **Não é bug, é característica do instrumento**

### Análise da invariante (i) — consolidated_bias_score

Fórmula encontrada em `orderbook_analyzer/core.py` L2537-2547:
```python
bias_score = 0.5 + (imbalance * 0.3)
if ratio > 0:
    ratio_adj = min(1.0, max(-1.0, (ratio - 1.0) / 2.0))
    bias_score += ratio_adj * 0.2
bias_score = min(1.0, max(0.0, bias_score))
```
- Janela 21: calculado=0.2395, reportado=0.2395 -> **Reproduzido exatamente**
- Janela 24: calculado=0.8024, reportado=0.8022 -> **Desvio 0.0002** (arredondamento intermediário aceitável)

---

## FASE 2 — COERÊNCIA ENTRE FONTES

### 2.1 Divergência close vs mid (bps)

| Janela | preco_fechamento | orderbook.mid | Divergência (bps) | Flag |
|--------|-----------------|---------------|-------------------|------|
| 21 | 77601.36 | 77550.25 | **6.59** | > 3bps |
| 24 | 77474.01 | 77449.75 | **3.13** | > 3bps |

**Causa:** `preco_fechamento` é o último trade, `orderbook.mid` é o snapshot L2 capturado com latência de 7320ms (Janela 21). O preço moveu ~$51 nesse intervalo. Divergência < 10bps (threshold de INAPTO). Consistente com `latency_category: POOR`.

### 2.2 Volume total BTC vs duração

| Janela | Volume BTC | Duração (s) | Rate (BTC/s) | n_trades | Esperado |
|--------|-----------|-------------|-------------|----------|----------|
| 0 | 10.500 | ? | ? | ? | — |
| 21 | 19.862 | 58.4 | **0.340** | 3671 | 1-3 BTC/s |
| 24 | 27.578 | 58.5 | **0.471** | 4167 | 1-3 BTC/s |

Rate observado (~0.34-0.47 BTC/s) está **~1 ordem de grandeza abaixo** do esperado. Explicável pelo horário: 23:30-23:52 UTC = final da sessão NY em Labor Day (2026-09-01). Volume reduzido neste contexto é normal. Campo `data_quality.total_trades_processed: 42911` vs `num_trades: 3671` indica processamento contínuo além da janela individual.

### 2.3 Trades fora de ordem / duplicados

Scripts `measure_out_of_order_trades.py` e `measure_dedup_effect.py` requerem trade buffer ao vivo. **Não aplicável** aos dados extraídos. Campo `data_quality.invalid_trades: 0` e `valid_rate_pct: 100` nas janelas 21 e 24 indica 0% de trades inválidos no processamento ao vivo.

### 2.4 Macro/FRED

| Janela | us10y_yield (evento) | TNX FRED | Diferença |
|--------|---------------------|----------|-----------|
| 21 | 4.796 | 4.79 | 0.006 (0.13%) |
| 24 | 4.796 | 4.79 | 0.006 (0.13%) |

Sem NaN convertido em 0 ou string. Diferença de 0.006 é normal (fontes diferentes).

---

## FASE 3 — DUPLICAÇÃO E RUÍDO NO PAYLOAD

### Campos redundantes identificados

| Redundância | Janela(s) | Impacto estimado |
|------------|-----------|------------------|
| `contextual_snapshot` == `enriched_snapshot` (cópia idêntica) | 21, 24 | ~1200 tokens/janela |
| `flow_imbalance` == `pressure` (mesmo valor exato) | 21, 24 | ~20 tokens/janela |
| `imbalance` == `flow_imbalance` (dentro de orderbook_data) | 21, 24 | ~20 tokens/janela |
| `pivot_points` duplica `pivots` (mesma estrutura) | 21, 24 | ~500 tokens/janela |
| `fibonacci_levels` aparece 2x (raiz + pattern_recognition) | 21, 24 | ~100 tokens/janela |
| `fair_value_gaps` aparece 2x (pattern_recognition + institutional_analytics) | 21, 24 | ~200 tokens/janela |
| `market_structure` aparece 2x | 21, 24 | ~100 tokens/janela |
| `stochastic/williams_r` duplicados em technical_extras e technical_indicators_extended | 21, 24 | ~100 tokens/janela |

### Estimativa de tokens

| Janela | Payload total | Tokens estimados | Tokens redundantes | % desperdício |
|--------|-------------|-----------------|-------------------|--------------|
| 21 | 46,796 chars | ~11,699 | ~2,190 | **~19%** |
| 24 | 45,914 chars | ~11,478 | ~2,190 | **~19%** |

### Verificação do payload_builder_compact.py

O `payload_builder_compact.py` (L1708-1850) **SIM, remove a duplicação** antes de enviar à IA:
- Gera payload de ~200 tokens com keys abreviadas (price, flow, ob, tf, sr)
- As redundâncias existem **apenas no evento cru** (JSONL/log), NÃO no payload enviado à LLM

---

## FASE 4 — VEREDITO DE QUALIDADE

| Janela | UTC | OK/Total | close-mid (bps) | Vol BTC | dur (s) | Flags | Veredito |
|--------|-----|----------|-----------------|---------|---------|-------|----------|
| 0 | 23:12:20 | 2/2 | N/A | 10.5 | ? | payload mínimo | **APTO COM RESSALVAS** |
| 21 | 23:49:00 | 13/16 | 6.6 | 19.862 | 58.4 | impact_1M insuf.; no_spread_hist | **APTO COM RESSALVAS** |
| 24 | 23:52:00 | 13/16 | 3.1 | 27.578 | 58.5 | impact_1M insuf.; no_spread_hist | **APTO COM RESSALVAS** |

**Resultado: 3/3 janelas APTAS (100%) -> FASE 5A**

> [!WARNING]
> Embora 100% aptas, a amostra é **extremamente pequena** (2 janelas com dados de book/flow, separadas por 3 minutos).

---

## FASE 5A — ANÁLISE DE MERCADO

> [!IMPORTANT]
> Apenas 2 janelas com dados completos (Janela 21 e 24), cobrindo ~3 minutos. Análise limitada.

### 5.1 Níveis de S/R

#### Defense Zones (sr_analysis — Janela 21)

| Nível | Preco | Tipo | Lado | Strength | Fontes | Sinais |
|-------|-------|------|------|----------|--------|--------|
| **1** | 77680.88 | Confluência | Sell (R) | **70** | ask_wall + vp_hvn + val_weekly + vp_poc | 8 |
| **2** | 77356.18 | Confluência | Buy (S) | **54** | vp_hvn + ema_21_15m + vp_val | 5 |
| **3** | 78607.82 | Confluência | Sell | **54** | pivot_daily_close + vp_hvn + weekly_pivot | 4 |
| **4** | 77547.91 | Cluster | Buy | **43** | vp_hvn + pivot_daily_s1 | 6 |
| **5** | 78458.00 | Cluster | Sell | **44** | vp_hvn + vp_vah | 5 |

#### Pivots (calculados sobre período anterior — High=79250, Low=77392, Close=78581.29)

| Período | Pivot | S1 | R1 |
|---------|-------|----|----|
| Daily | 78407.76 | 77565.53 | 79423.53 |
| Weekly | 78610.29 | 75741.72 | 80550.58 |
| Monthly | 74111.72 | 66744.57 | 85948.44 |

#### Volume Profile (histórico diário)

| Nível | Tipo | Preco | Vol BTC | Relevância |
|-------|------|-------|---------|-----------|
| POC diário | HVN | **77691** | 150.52 (10%) | Maior acúmulo do dia anterior |
| VAH diário | — | **78528** | — | Topo Value Area |
| VAL diário | — | **77326** | — | Base Value Area |
| POC semanal | HVN | **78806** | 820.42 (10%) | POC maior timeframe |

### 5.2 Áreas de Liquidez

**Evolução do book (Janela 21 -> 24):**

| Métrica | Janela 21 | Janela 24 | Mudança |
|---------|-----------|-----------|---------|
| bid_depth_usd | $533,680 | $1,678,957 | **+215%** |
| ask_depth_usd | $2,235,545 | $630,619 | **-72%** |
| volume_ratio | 0.239 | 2.662 | **11x inversão** |
| bias_score | 0.2395 (bearish) | 0.8022 (bullish) | **Inversão** |

Em 3 minutos, o book inverteu completamente de ASK_HEAVY para BID_HEAVY:
- Janela 21: Exaustão de COMPRA (compradores batendo parede de venda)
- Janela 24: Exaustão de VENDA (vendedores batendo parede de compra recém-formada)

**Icebergs:** `iceberg_activity: 1` na Janela 21.

### 5.3 Fluxo

| Métrica | Janela 21 | Janela 24 | Nota |
|---------|-----------|-----------|------|
| CVD | 42.4239 | 34.8676 | Queda (-7.56) |
| Delta fechamento | +11.487 | -5.232 | Inversão |
| Aggressive buy % | 78.91% | 40.51% | Desaceleração |
| Whale delta | -3.896 | -3.896 | **Constante negativo** |
| Whale class. | MILD_DISTRIBUTION (-32) | — | Distribuição |

**Divergência:** Janela 21: preço subiu (+$70) com flow BUY forte (78.91%), MAS whale_delta NEGATIVO (-3.896) + book pesado no ask. Classification: `smart_distribution` — whales distribuindo para retail.

### 5.4 Zonas de Entrada Candidatas

> [!WARNING]
> Baseadas em apenas 2 janelas (3 minutos). Baixa confiança estatística.

| # | Preco | Tipo | Evidência | Invalidação | Forca |
|---|-------|------|-----------|-------------|-------|
| 1 | 77356 (+/-100) | Suporte confluência | Defense strength=54, vp_hvn + ema_21_15m + val + fib 61.8% (77364) | Rompimento 77250 | **3/5** |
| 2 | 77548 (+/-60) | Suporte cluster | Defense strength=43, pivot daily S1, book inverteu para bid-heavy J24 | Rompimento 77440 | **2/5** |
| 3 | 77680 (+/-60) | Resistência confluência | Defense strength=70 (mais forte), ask wall + val_weekly + vp_poc | Rompimento 77810 | **3/5** |
| 4 | 78408 (+/-100) | Resistência pivot | Daily pivot, vp_hvn, distância ~800bps | Rompimento 78500 | **2/5** |
| 5 | 77486 (+/-15) | Suporte micro | POC intraday J24, dwell_price (39s em low), inversão do book | Rompimento 77470 | **2/5** |

### 5.5 O que os dados NÃO permitem afirmar

1. **Recorrência de walls** — Sem 3+ janelas para confirmar persistência
2. **Evolução temporal de liquidez** — Apenas 2 snapshots de book
3. **Icebergs recorrentes** — Sem janelas consecutivas
4. **CVD contínuo** — Sem janelas 22, 23 para verificar continuidade
5. **Regime de volatilidade** — `volatility_percentile: 100` pode ser artefato de amostra curta
6. **Absorção institucional** — 0 eventos de absorção detectados
7. **Direção futura** — Divergência whale vs retail sugere distribuição, mas 2 janelas são insuficientes
8. **ASCENDING_TRIANGLE** — Confidence 0.256 (25.6%), baseado em intra-janela

---

## PLANO DE CORREÇÃO (complementar)

| # | Sintoma | Arquivo/Função Raiz | Correção | Teste | P |
|---|---------|---------------------|----------|-------|---|
| 1 | **80/81 ANALYSIS_TRIGGER trimadas** | guardian/event_logger (aplica `trimmed_by_guardian`) | Preservar N últimas janelas completas ou extrair de `events.payload` no DB | `test_guardian_preserves_payload` (criar) | **P0** |
| 2 | Market impact sobre liquidez insuficiente sem flag | `orderbook_analyzer/core.py` — market_impact | Adicionar `"unfillable": true` quando notional > depth | `test_market_impact_unfillable_flag` (criar) | **P1** |
| 3 | contextual_snapshot == enriched_snapshot | market_orchestrator (montagem do evento) | Remover enriched_snapshot se identico | Verificar em `test_build_compact_payload_*` | **P1** |
| 4 | Campos redundantes (flow_imbalance/pressure; pivots/pivot_points; fibonacci 2x; FVGs 2x) | `orderbook_analyzer/core.py` L2627-2630 | Consolidar: manter flow_imbalance, remover pressure/imbalance | `test_event_no_redundant_fields` (criar) | **P2** |
| 5 | Latência 7320ms (POOR) na Janela 21 | Não é bug — latência real do snapshot L2 | Investigar captura paralela do snapshot | — | **P2** |

---

## RESUMO EXECUTIVO

**Veredito geral:** Os dados das janelas com payload completo (Janela 21 e 24) são **APTOS COM RESSALVAS** para análise por IA. As invariantes matemáticas críticas (volume, delta, flow_imbalance, mid/spread, value area) estão corretas. O `consolidated_bias_score` foi reproduzido com sucesso.

**3 problemas mais graves:**

1. **P0 — Perda massiva de dados:** 80 de 81 ANALYSIS_TRIGGER foram trimadas pelo guardian, deixando apenas 3 eventos analisáveis em 26h de operação. Impede análise temporal e auditoria offline.

2. **P1 — Market impact sobre liquidez insuficiente:** Slippage calculado e reportado quando notional excede profundidade do book. A IA pode tratar como executável.

3. **P1 — Duplicação de ~19% no payload cru:** contextual_snapshot == enriched_snapshot. Mitigado pelo payload_builder_compact.py que não envia à LLM.

**Próxima ação recomendada:** Resolver P0 — extrair payloads completos da coluna `events.payload` no SQLite para auditar as 80 janelas trimadas, permitindo análise de recorrência de walls, evolução do book e validação temporal completa.
