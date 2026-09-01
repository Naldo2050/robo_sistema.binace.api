# AUDITORIA FORENSE READ-ONLY — Última Execução Real
**Execução auditada:** 2026-08-23T00:21:12Z → 00:32:00Z (~11 min, 12 janelas de 1m)
**Data da auditoria:** 2026-08-23 · **Modo:** somente leitura (SQLite `mode=ro`, zero escrita, zero rede)
**Commit de referência do código:** 7cd78af (inclui flow.q)

---

## 1. RESUMO EXECUTIVO

A execução produziu **dados matematicamente corretos e temporalmente coerentes**. Todas as fórmulas-chave (delta, imbalance, net_flow, índice de absorção, pivots, funding) foram recalculadas independentemente e conferem com o código-fonte. O payload da IA está RFC-8259 limpo, carrega `flow.q` corretamente propagado (commit 7cd78af operando em produção) e os deltas compactos d1/d5/d15 batem 6/6 com as fontes.

**Dois achados relevantes:**
1. **P1 — OutcomeTracker:** o `outcome_5m_pct=+0.1815%` imputa preço 77244.23, que corresponde ao close de **00:29** (+7 min após a entrada), não ao horizonte de +5min (00:27 → +0.116%). `evaluated_at` também é 00:28 (+6 min). Suspeita de preço "live tick" vs candle boundary no labeling.
2. **P2 — Eficiência:** `summary` ocupa ~45% dos bytes do payload e grande parte é prosa que repete números já presentes nas seções estruturadas.

Zero P0. Classificação final: **READY WITH WARNINGS**.

---

## 2. EVENTOS ANALISADOS

| Fonte | Registros | Válidos | Período |
|---|---|---|---|
| dados/eventos_fluxo.jsonl | 16 linhas | 16 (0 inválidas) | 00:21:00Z → 00:32:00Z |
| trading_bot.db `events` | 16 (payloads completos 882–39.850 B) | 16 JSON ok | idem |
| trading_bot.db `signal_outcomes` | 1 | 1 | criado 00:22:06Z |
| logs/payload_metrics.jsonl | 3.092 | 3.092 | histórico multi-run (sem ts/linha) |
| logs/last_llm_payload.json | 1 dump | — | **STALE: 2026-08-04** (19 dias) |
| dados/fred_cache.json | 1 série (TNX) | 1 | updated 00:30:46Z |
| dados/eventos_visuais.log | presente em `dados/` (510.938 B); não auditado numericamente |

Composição dos 16 eventos: 12× ANALYSIS_TRIGGER (aparados pelo guardian), 1× Absorção (completo), 2× AI_ANALYSIS (com ai_payload completo), 1× Alerta.
NaN/±Inf em payloads DB: **0**. Campos null críticos: nenhum nos eventos ricos.

## 3. INVARIANTES MATEMÁTICOS (FASE 3)

**251 verificações: 251 PASS efetivos** (242 PASS estritos + 9 reclassificados como arredondamento de persistência).

| Invariante | Resultado | Evidência |
|---|---|---|
| total == buy+sell | PASS ×13 | diffs ≤ 8.9e-16 BTC |
| delta == buy−sell | PASS c/ tolerância ≤1e-4 | diffs ≤ 5e-5 (delta arredondado 4dp vs volumes 6–7 dígitos) — ex. id11: −1.9471 vs −1.94705 |
| buy_pct+sell_pct == 100 | PASS ×13 | máx desvio 0.0 |
| bsr == buy/sell | PASS ×13 | ex. 5.039/2.935 = 1.7171 ✓ |
| flow_imbalance == (b−s)/(b+s) | PASS ×13 | recálculo independente, tol 5e-4 |
| **net_flow_1m == buy_notional−sell_notional** | **PASS diff=0.00** | id2: 162222.97 ✓ (marcado PARTIAL cov=98.4%) |
| ob.imbalance == (bid−ask)/(bid+ask) | PASS ×13 | ex. id2: −0.5324 ✓ |
| bid/ask ≥ 0; spread ≥ 0; best_ask ≥ best_bid | PASS ×13 | ask sempre bid+0.1 |
| L25 ≥ L1 (acumulado, bids e asks) | PASS ×26 | |
| VP VAL ≤ POC ≤ VAH | PASS ×14 | ⚠ VAH==POC degenerado ids 8–13 (ver Fase 4) |
| pivots r1<r2<r3; s3<s2<s1; s1<piv<r1 | PASS ×39 | semântica clássica confirmada no schema |
| indice_absorcao == \|Δ\|/max(range; 0.1%·c) | PASS exato | 2.104/77.104 = 0.027292 ✓ (data_handler.py:233-235) |
| rótulo/direção absorção coerente c/ detector | PASS | Δ>thr ∧ c≤o·1.002 ∧ close_pos_venda=0.571>0.5 ⇒ "Absorção de Compra", aggression=buy, absorption=sell ✓ |
| funding em percent plausível (\|fr\|≤0.75%) | PASS ×13 | 0.01% = 0.0001 fração; **sem dupla conversão ×100** |
| longs_usd+shorts_usd == open_interest_usd | PASS ×13 | 8.167B exato |
| whale_delta == wbuy−wsell | PASS ×13 | |
| cvd == Σ sector deltas | PASS ×13 | tol 0.01 ✓ |
| net_flow_15m vs 5m | PARTIAL correto | 15m WARMING_UP toda a run (11 min < 15 min); valores nunca tratados como janela cheia |

## 4. DISCORDÂNCIAS CROSS-MODULE (FASE 4)

| # | Comparação | Evidência | Classificação |
|---|---|---|---|
| C1 | preço id2 (77104.29) vs id3 (77104.30) mesma janela | diff 0.01 | **OK** (rounding int) |
| C2 | delta id2 vs id3 | 2.1043 == 2.1043 | **OK** |
| C3 | delta id1 (−0.57) vs id2 (+2.10) | janelas diferentes (abertura vs fechamento) | **EXPLICÁVEL** |
| C4 | flow imb +0.26 (buy) vs orderbook imb −0.53 (sell) | composite.agreement=0 documentado no próprio evento | **EXPLICÁVEL** (agressivo×passivo, by design) |
| C5 | regime_analysis=BREAKOUT vs multi_tf Range/Manipulação vs payload mode=BRK | módulos independentes; payload coerente com regime_analysis | **EXPLICÁVEL** |
| C6 | VP daily muda às 00:28 (poc 77059→77196) sem troca de sessão; VAH==POC por 6 eventos | VP dinâmico intradiário sob tendência forte | **EXPLICÁVEL** (monitorar) |
| C7 | integridade id1 vs id2 (cov 50.5→98.4%) | offset de stamping ~0.7s entre convenções | **EXPLICÁVEL** |
| C8 | funding ctx.fr==derivatives.funding_rate_percent (0.01) nos 2 AI payloads | igualdade exata | **OK** |
| C9 | TNX cache 4.69 vs ml_features.us10y_yield 4.738 | fontes/momentos distintos (Δ≈1%) | **SUSPEITO leve** |

Nenhum caso ERRO. Nenhuma inversão de sinal cross-module na mesma janela. Integridade FULL/trunc consistente entre source e payload.

## 5. DUPLICATAS (FASE 2)

- **Exatas** (hash canônico SHA-256, completo e excluindo voláteis): **0 grupos**.
- **Semânticas** (symbol+tipo+janela+preço+delta+resultado): **0 grupos**.
- **JSONL vs DB**: mesmo event_id com serialização diferente — JSONL é projeção aparada (`trimmed_by_guardian`, 12/16 linhas). Redundância entre sinks BY DESIGN (DB 398 KB vs JSONL 43 KB para os mesmos 16 eventos).
- **Intra-evento**: `enriched_snapshot == contextual_snapshot` só no evento de sinal (dedup do orchestrator); nos TRIGGERs são objetos distintos — candidato a revisão (P2).

## 6. REDUNDÂNCIA DE PAYLOAD (FASE 6-D)

Payloads analisados: id4 = 2.594 B ≈ 648 tok; id12 = 2.699 B ≈ 675 tok (cap 6.144 B folgado).
Redundância estimada: **~350–500 B (~90–125 tok, ≈15–18%) por payload**, quase toda na prosa de `summary.*.note` que repete níveis/values presentes em sr/flow/regime estruturados + campos-fonte (`ofi.src`, `vwap.src`, `mr.src`) e `_compacted`.

## 7. RANKING DE CUSTO/TOKEN POR SEÇÃO

| Seção | id4 | id12 | % médio | ~tok médio |
|---|---|---|---|---|
| summary | 1171 B | 1204 B | **45.2%** | ~296 |
| flow (+q) | 333 B | 354 B | 13.1% | ~86 |
| ctx | 165 B | 173 B | 6.4% | ~42 |
| tf | 145 B | 145 B | 5.5% | ~36 |
| sr | 102 B | 115 B | 4.1% | ~27 |
| price | 112 B | 112 B | 4.3% | ~28 |
| ob | 113 B | 111 B | 4.2% | ~28 |
| regime/vwap/mr/ofi/alerts/liq/qual/w/resto | 451 B | 485 B | 17.2% | ~116 |

## 8. PROBLEMAS P0/P1/P2/P3

### P0 (dado corrompido em decisão): **nenhum**

### P1 (parcial/contraditório apresentado como íntegro)
- **P1-1 · OutcomeTracker — horizonte/preço do outcome_5m**
  - Arquivo/campo: `trading_bot.db.signal_outcomes` (id=1) — `outcome_5m_pct`, `evaluated_at`
  - Evidência: entry=77104.29 @ signal_epoch 00:22:00; `outcome_5m_pct=+0.1815%` imputa **77244.23**, idêntico ao close do evento id=10 (**ts=00:29:00**, janela 00:28→00:29). Close real de +5min (00:27, evento id=8) = 77193.7 ⇒ **+0.116%**. `evaluated_at=1787444880000` = **+6 min** após signal_epoch.
  - Observado: +0.1815% / preço de ~00:29 · Esperado: +0.116% @ 00:27 (ou definição explícita de horizonte)
  - Impacto: contamina labeling para treino/análise de performance (não afeta decisão live)
  - Severidade: **P1**

### P2 (redundância/custo/observabilidade)
- **P2-1** `summary` = 45% do payload; notas repetem números estruturados (~90–125 tok/payload).
- **P2-2** `logs/payload_metrics.jsonl`: sem timestamp por linha (impossível atribuir à execução); p95=9.294 B > cap 6.144; max=96.393 B; 212 linhas acima do cap (histórico).
- **P2-3** `logs/last_llm_payload.json` stale (2026-08-04) — dump não rotativo.
- **P2-4** `enriched_snapshot` ≠ `contextual_snapshot` nos ANALYSIS_TRIGGERs — possível bloco redundante nesses eventos.
- **P2-5** `eventos_fluxo.jsonl` espessa duplicação do DB em projeção aparada (by design, mas sem TTL/rotação).

### P3 (cosmético)
- **P3-1** Out-of-order na persistência: id3 (…519329) gravado após id2 (…520000), Δ=−0.67s (convenções window_close_ms vs boundary).
- **P3-2** `event_id` int em 1 linha (17799293) vs hex-string nos demais; `source` str|dict; `alerts` dict|list; `summary` str|dict; `market_structure` dict|str; `timestamp_utc` str|int(1x).
- **P3-3** VAH==POC degenerado por 6 eventos consecutivos (VP dinâmico sob tendência).
- **P3-4** us10y_yield (4.738) vs FRED cache TNX (4.69) divergem ~1%.

## 9. NOTAS DE PRONTIDÃO (0–100, baseadas em dados reais desta execução)

| Dimensão | Nota | Justificativa |
|---|---|---|
| Integridade matemática | **96** | 251 invariantes OK; únicos desvios = arredondamento ≤5e-5 |
| Integridade temporal | **92** | TZ trios corretos; gaps 60s regulares; 1 out-of-order −0.7s |
| Consistência cross-module | **94** | 0 ERRO; divergências todas explicáveis e documentadas |
| Qualidade flow/orderbook | **97** | fórmulas exatas; depths consistentes; PARTIAL respeitado |
| Qualidade S/R | **95** | ordenações válidas; VAH==POC transitório |
| Qualidade institutional | **93** | funding unidade correta; whales=0 legítimo (threshold) |
| Qualidade macro | **90** | FRED fresh 1.2min; pequena divergência feature×cache |
| Qualidade payload IA | **97** | RFC limpo; q propagado; d* 6/6; fail-closed quality ativo |
| Eficiência de tokens | **82** | 2.6KB/payload ok, mas summary=45% com prosa repetitiva |
| Persistência/OutcomeTracker | **78** | operante pós-9f02e94, porém P1-1 de horizonte/preço |

### Classificação: **READY WITH WARNINGS**
Critério aplicado: zero P0 ✓; um P1 não mitigado (P1-1, impacto em labeling, não em decisão) ⇒ READY bloqueado apenas por esse item.

## 10. CORREÇÕES RECOMENDADAS (NÃO implementadas)

| ID | Arquivo/Campo | Valor observado | Valor esperado | Impacto | Severidade |
|---|---|---|---|---|---|
| R1 | OutcomeTracker (avaliador de outcomes) — fonte de preço e `evaluated_at` | preço de ~00:29 imputado em outcome "5m"; evaluated_at=+6min | preço do candle boundary em signal_epoch+5min (=00:27 → 77193.7) ou documentação explícita do horizonte | labeling distorcido p/ treino/analytics | **P1** |
| R2 | builder summary (`_build_summary_section`/notes) | 45% dos bytes em prosa que repete sr/regime/flow | comprimir notes p/ tokens não-numéricos ou mover detalhe p/ seções estruturadas | −90..125 tok/payload | P2 |
| R3 | `payload_metrics_aggregator.append_metric_line` | linhas sem timestamp; p95 histórico > cap | adicionar `ts_ms`; alertar quando bytes_after>max_bytes | observabilidade | P2 |
| R4 | dump `logs/last_llm_payload.json` | stale de 19 dias | rotação/timestamp no nome | higiene | P2 |
| R5 | ANALYSIS_TRIGGERs — `enriched_snapshot` | objeto extra ≠ contextual | dedup como no path de sinal ou remoção | bytes DB/jsonl | P2 |
| R6 | gerador de `event_id` (fallback int) + normalização de tipos (`source`, `alerts`, `summary`, `market_structure`) | tipos inconsistentes entre eventos | contrato único de tipo | parsing downstream | P3 |
| R7 | persistência de eventos | ordem id≠ts (Δ0.67s) | stampar TRIGGERs no mesmo boundary do close | consulta/ordenação | P3 |
| R8 | ml_features cross_asset `us10y_yield` | 4.738 vs cache 4.69 | alinhar fonte/carimbo de tempo | consistência macro | P3 |

---
*Método: leitura pura (`sqlite3 mode=ro`, json parse com parse_constant estrito, hashes SHA-256 canonizados, recálculo independente das fórmulas contra `data_handler.py`/`core.py`). Nenhum arquivo de dado foi modificado; nenhum INSERT/UPDATE/DELETE; sem rede/LLM.*
