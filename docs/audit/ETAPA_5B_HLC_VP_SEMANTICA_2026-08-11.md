# ETAPA 5B — Auditoria: H/L/C vs Volume Profile (pivot_points.vah/val/poc)

Data: 2026-08-11 · Escopo: documentação/classificação + patch mínimo de legenda
Sem commit (working tree mantido para revisão).

---

## FASE 0 — Baseline

- HEAD `97edd0a` limpo (3 untracked out-of-scope pré-existentes).
- Suite dirigida: **480 passed**.
- Instrumento: `git show` para hash do blob vs working tree (sem diffs falsos).

## FASE 1 — Origem da semântica H/L/C

- Introduzida no commit `aa97cf1` ("fix(pivots)", 2026-08-09): escolha deliberada de
  compatibilidade. `pivot_points.{daily,weekly,monthly}.vah/val/poc` com
  `source=classic` são **H/L/C do período anterior COMPLETO (iloc[-2])**, NÃO Volume
  Profile (os VP reais vivem em `historical_vp`).
- Documentado em `docs/audit/` (ETAPA 5) e `common/ai_field_legend.py:23`.

## FASE 2 — Consumidores (subagente)

- **Nenhum consumidor de produção usa `pivot_points.vah/val/poc`.** MQL5/export usam
  `historical_vp`. `ctx.poc/val/vah` do payload vêm de `historical_vp` (VP real).
- Risco ALTO só no fallback do enricher (linha 458): VP intraday parcial vira H/L/C
  (source=`vp_fallback`); degenerado → estimado via ATR/EMA (source=`multi_tf_fallback`).
  Ambos distinguidos por `source`.

## FASE 3 — O que chega ao LLM (payload compacto real, evento J4)

Capturado com `scripts/diagnostics/capture_compact_sr.py` (dados J4 reais):

```json
"sr":  {"r1": [64772, 52], "r1_dist": 30, "def_bias": "strong_sel"},
"ctx": {"ses": "_", "poc": 64689, "val": 64520, "vah": 65133},
"price": {"c": 64742}
```

- `sr.r1/r2/s1/s2 = [preço, força]` — força (0-100) é **confluência de defense zones**
  (orderbook+VP+pivots), NÃO proximidade. `r*_dist` = |close−center|; `r*_conf` = fontes.
- `ctx.poc/val/vah` = VP diário real. **O AI NÃO vê** `pivot_points`,
  `immediate_support`, `support_strength` nem `resistance_strength`.

## FASE 5 — Prompt

- `_get_system_prompt` (analyzer_qwen.py:1649): groq → SYSTEM_PROMPT(+FIELD_LEGEND);
  compressão → SYSTEM_PROMPT_COMPRESSED (+COMPRESSED_KEY_DICTIONARY); senão SYSTEM_PROMPT.
- **ACHADO**: SYSTEM_PROMPT (linhas 643-646 e 703-709) documentava
  `immediate_resistance/support`, `resistance_strength/support_strength`,
  `defense.sell_zone/buy_zone` — chaves que NÃO existem no payload compacto — e não
  documentava `sr.r1/s1`, `r*_dist`, `def_bias`. Legend stale → risco de o LLM inferir
  campos inexistentes e interpretar errado a força dos níveis.

## FASE 7 — SRStrengthScorer (sr_strength.py, 391 linhas)

- `_collect_candidates`: pivot_data iterado como `{método: {nível: preço}}` → H/L/C
  viram candidatos `pivot_classic_{high,low,close}` (weight 1.3). VP: poc/vah/val/hvns.
- `_merge_nearby_levels`: tolerance 0.15%, primary_source = maior weight, confluences.
- `_count_touches`: usa high/low de candles dentro de 0.15% do nível.
- **ACHADO**: `SRStrengthScorer` NÃO é chamado em produção (só `scripts/diagnostics/`
  e testes). Dívida cosmética pré-existente: docstrings/comentários duplicados em
  `sr_strength.py` (linhas 155-196 etc.) — código íntegro, importa e passa.

## FASE 8 — Avaliação de H/L/C como S/R

- Em produção (`defense_zones._extract_pivot_defense`), `pivot_keys` = pivot/pp/r1-r3/s1-s3
  — **H/L/C de pivots são ignorados como sinais de defesa** (linha 337 `continue`).
- H/L/C só entram como S/R fora da produção (scorer standalone) ou como fallback
  documentado (vp_fallback/multi_tf_fallback no enricher).

## FASE 9 — Replay J4 com o scorer (scripts/diagnostics/replay_j4_scorer.py)

| Nível | primary_source | Confluências | Touches | Força | Type |
|---|---|---|---|---|---|
| 64733.54 | **pivot_classic_low** | [pivot_classic_low, hvn_daily] | 7 | **74** | support |
| 65017.69 | pivot_classic_pivot | [round_number, pivot] | 3 | 64 | resistance |
| 64901.59 | **pivot_classic_close** | [pivot_classic_close] | 5 | **60** | resistance |
| 65474.46 | **pivot_classic_high** | [pivot_classic_high] | 4 | **56** | resistance |

- Sem H/L/C: o mesmo hvn (64737) sozinho vale 56; o low (64730.08) merge com o hvn e
  vira o nível #1 com força 74. **Confluência fabricada pela proximidade do merge** —
  confirma o risco se o scorer algum dia entrar em produção.

## FASE 10 — Contratos (antes do patch) — tests/unit/test_sr_etapa5b_contract.py

10 testes, todos passam antes e depois do patch (compatibilidade garantida):

- A. `pivot_points.daily.vah/val/poc` == H/L/C clássico (source=classic), ≠ historical_vp.
- B. `historical_vp` nunca mutado pelo pipeline.
- C. Payload compacto NÃO contém pivot_points/immediate_support/support_strength.
- D. defense strength é confluence-based (J4: strength 52, source_count 2).
- E. H/L/C de pivots NÃO viram sinais de defesa (pivot_keys exclui high/low/close).
- F. `source` ∈ {classic, vp_fallback, multi_tf_fallback}.
- G. Aliases legados pivot/pp continuam aceitos no detector.
- +3 contratos de legend do prompt (regressão do patch).

## FASE 11 — Classificação

| Problema | Severidade | Impacto produtivo |
|---|---|---|
| `pivot_points.vah/val/poc` = H/L/C com nomes de VP | Média (nomenclatura enganosa) | Nulo (sem consumidor produtivo; fallbacks documentados por `source`) |
| Legend do prompt (SYSTEM_PROMPT/SYSTEM_PROMPT_COMPRESSED/FIELD_LEGEND) não documenta `sr.*` e cita chaves inexistentes | **Alta** | Qualidade de decisão do LLM (campos fantasma, força mal explicada) |
| Scorer standalone pode fabricar confluência H/L/C+hvn (replay J4: 56→74) | Média | Nulo hoje (fora de produção); risco futuro |
| Docstrings duplicadas em sr_strength.py | Baixa (dívida cosmética) | Nulo |

## FASE 12 — Patch mínimo (documentação apenas, sem tocar fórmulas/valores)

1. `market_orchestrator/ai/analyzer_qwen.py` — legend NÍVEIS TÉCNICOS (643-646) e seção
   SUPORTE E RESISTÊNCIA (703-709) reescritos com as chaves reais `sr.r1/r2/s1/s2`,
   `r*_dist`, `r*_conf`, `def_bias`, `ctx.poc/val/vah` e a semântica de força
   (confluência de defesa, NÃO proximidade). Removidas referências a chaves inexistentes.
2. `common/ai_payload_optimizer.py` — COMPRESSED_KEY_DICTIONARY ganhou linhas `sr:` e `ctx:`.
3. `common/ai_field_legend.py` — FIELD_LEGEND ganhou `sr=defense_zones` (semântica de
   força) e nota de que `pivot_points` não é enviado no payload compacto.

Diff total: **3 arquivos, +18/−11**.

## FASE 13 — Validação

- Suite unitária completa: **983 passed** (baseline 480; +503 incluindo novos).
- Contratos novos: 10/10 verdes antes e depois do patch.
- Nenhuma fórmula/valor alterado: strength 52, centros, sources, side, distâncias
  (J1-J4) idênticos aos reproduzidos na ETAPA 5.

## Anexos (scripts de diagnóstico, sem commit)

- `scripts/diagnostics/replay_j4_scorer.py` — replay J4 com/sem H/L/C.
- `scripts/diagnostics/capture_compact_sr.py` — captura do payload compacto real.
- `scripts/diagnostics/map_pbc_keys.py` — mapa de chaves do payload_builder_compact.
