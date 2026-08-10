# AUDITORIA — CAMINHO REAL DE PIVOT_POINTS EM PRODUÇÃO (2026-08-09)

**Escopo:** mapear todos os caminhos de `pivot_points`/pivots no caminho live, confirmar o que a IA recebe, e concluir bug vs. nomenclatura. **Nenhuma correção aplicada.**

---

## PASSO 1 — MAPA DE CAMINHOS (tabela fonte → destino → ativo? → período)

| # | Fonte (cálculo) | Fonte de dados | Destino | Ativo no live? | Período |
|---|---|---|---|---|---|
| 1 | `support_resistance/__init__.py:30` `daily_pivot(df)` — **iloc[-2] fix 75bd3ec** | klines Binance 1d/1w/1M (limit=5) | `context_collector._calculate_pivots` → `context["pivots"]` (macro_context) | **SIM (calculado a cada ciclo ~60s)** | **CORRETO** (anterior completo) |
| 2 | `fetchers/context_collector.py:1052` `_calculate_pivots` (chama daily_pivot/weekly_pivot/monthly_pivot; import em `:59`) | klines 1d/1w/1M | `macro_context["pivots"]` | SIM | CORRETO |
| 3 | `market_orchestrator.py:1460` `_pivot_data = signal.get("contextual_snapshot", {}).get("pivots", {})` | **lê campo que NÃO EXISTE** (`contextual_snapshot.pivots` ausente em 100% dos eventos reais) | `institutional_analytics.compute_all(pivot_data=...)` | SIM, mas **SIEMPRE vazio** → defense zones sem `pivot_classic_*` | **DEAD WIRE — dado correto descartado** |
| 4 | `market_analysis/historical_profiler.py:222` `update_profiles` | klines **1m de 00:00Z hoje → agora** (daily); **rolante 7d** 5m (weekly); **rolante 30d** 15m (monthly) | `macro_context["historical_vp"]` → `event.historical_vp` | SIM (task "profile" no context_collector:1133) | **PARCIAL** (dia corrente) |
| 5 | `institutional/enricher.py:363` `_build_pivot_points` (chamado em `enrich_signal:2017`) | `event.historical_vp.daily/weekly/monthly` (parcial) — H=VAH, L=VAL, C=POC | **`event.pivot_points`** (pivot/r1-r3/s1-s3/vah/val/poc) | SIM | **PARCIAL** — fórmula clássica com input parcial |
| 6 | `payload_builder_compact.py:977` `_build_static_context` | `event.historical_vp.daily` (parcial) | **`ai_payload.ctx.poc/val/vah`** | SIM | PARCIAL — legenda diz "volume_profile_daily" ✓ |
| 7 | `payload_builder_compact.py:919` `_build_sr` | `institutional_analytics.sr_analysis.defense_zones` (fontes `vp_poc`, `vp_vah`, `vp_val`, `sr_level_*_daily` — VP parcial + EMAs) | **`ai_payload.sr.r1/r2/s1/s2`** | SIM | PARCIAL (níveis VP) |
| 8 | `support_resistance/pivot_points.py:147` `calculate_multi_timeframe_pivots` (resample 'D'/'W' + iloc[-2]) | DataFrame (ex.: pattern_ohlc_history) | via `system.py:76 analyze_market` | **NÃO-live** — chamado apenas em testes (`tests/unit/test_support_resistance_*.py`) e health_check | (n/a) |
| 9 | ~~`features/multi_tf_feature_builder.py`~~ | — | — | **NÃO EXISTE** — arquivo citado em auditoria anterior é resultado de grep corrompido; não está no repo (`git ls-files` confirma) | (n/a) |

**Trechos-chave:**
```python
# support_resistance/__init__.py:35-42 — o fix
# Guard: dados insuficientes — iloc[-2] não existe...
last = df.iloc[-2]   # período anterior completo (não o atual em andamento)
```
```python
# fetchers/context_collector.py:1057-1060 — usa o fix com klines 1d
df_d = await self._fetch_klines(session, self.symbol, '1d', limit=5)
if not df_d.empty:
    pivots["daily"] = daily_pivot(df_d)          # ← CORRETO (iloc[-2])
```
```python
# market_orchestrator.py:1460 — o DEAD WIRE
_pivot_data = signal.get("contextual_snapshot", {}).get("pivots", {})   # nunca existe
```
```python
# market_analysis/historical_profiler.py:229-234 — o PARCIAL
start_day = now.replace(hour=0, minute=0, second=0, microsecond=0)      # 00:00Z HOJE
df_daily = self._fetch_historical_data(int(start_day.timestamp()*1000), int(now.timestamp()*1000), interval="1m")
```
```python
# institutional/enricher.py:394-396 — fórmula clássica com input VP parcial
if not degenerate and vp.get("vah") and vp.get("val") and vp.get("poc"):
    h, l, c = float(vp["vah"]), float(vp["val"]), float(vp["poc"])
```

---

## PASSO 2 — O QUE A IA REALMENTE RECEBE

Evento real `AI_ANALYSIS` (id 71, Exaustão, 23:31Z — payload compacto v3 enviado ao Groq):

- **`ai_payload` NÃO contém campo "pivots"/"pivot_points"** (keys: price, regime, flow, ob, tf, sr, w, alerts, ctx, ofi, vwap, liq, summary, ext...).
- `ctx`: `"poc": 64930, "val": 64845, "vah": 65294` ← `historical_vp.daily` (dia corrente PARCIAL, verificado com recálculo independente na observação — match exato).
- `sr.r2 = [65294, 51]` ← defense zone com fontes `vp_vah` + `sr_level_vah_daily` (mesmo VP parcial). Nenhuma zona com fonte `pivot_classic_*`.
- `ext.fib.hi = 65474 / lo = 64851` ← swing de hoje (parcial).
- O `event.pivot_points` (id 57/69): `daily: pivot=65023.0, r1=65201.0, s1=64752.0, vah=65294.0, val=64845.0, poc=64930.0` — derivado do VP parcial; **NÃO é o pivot clássico do dia anterior completo** (ontem: H=65192,54/L=64784,19/C=64962,60). E este campo nem sequer é enviado no payload compacto.

**Resposta do PASSO 2:** a IA recebe **VP intraday do dia corrente parcial** (que muda a cada ciclo de atualização ~60s, sem sentido como nível fixo de referência) sob os nomes `ctx.poc/val/vah` e dentro das defense zones (`sr`). **O pivot clássico do dia anterior completo NUNCA chega à IA** — apesar de ser calculado corretamente (caminho #1/#2), é descartado pelo dead wire (#3).

---

## PASSO 3 — BUG OU SEMÂNTICA DIFERENTE?

**Evidências de intenção:**
1. `enricher._build_pivot_points` docstring (enricher.py:365-369): *"Calcula pivot points clássicos (daily, weekly, monthly). Usa VP histórico quando disponível... Cada TF usa seus próprios dados"* + nomes de campo clássicos (`pivot`, `r1`-`r3`, `s1`-`s3`) + fórmula clássica `P=(H+L+C)/3` → **intenção: pivot clássico**.
2. `historical_profiler` docstring (linha 10-11): *"Calcula Volume Profile histórico (daily/weekly/monthly) a partir de klines da Binance"* — mas `update_profiles` usa **00:00Z hoje → agora** e janelas rolantes → **"histórico" é nome enganoso; na prática é VP intraday**.
3. `common/ai_field_legend.py:22`: `ctx=...poc/val/vah=volume_profile_daily` — a **legenda da IA documenta corretamente** o que ela recebe (volume profile, não pivot clássico).
4. `config/model_config.yaml` e schemas: nenhuma documentação adicional define `pivot_points` do evento como "período anterior" — a única declaração de intenção é a docstring do enricher (#1).

**Conclusão (objetiva):** **CONCLUSÃO A — bug real**, com nuance B:

- **A (bug de input + wiring) para `event.pivot_points` e defesas:** o campo promete pivot clássico (nome, fórmula e docstring) mas é alimentado pelo VP do dia corrente parcial; além disso, o único cálculo clássico CORRETO já existente (`macro_context["pivots"]`, iloc[-2]) é computado e desperdiçado por erro de wiring (`market_orchestrator.py:1460` lê `contextual_snapshot.pivots` inexistente). Dois bugs independentes.
- **B (nomenclatura) para o componente VP:** `historical_profiler` chama de "histórico" o que é VP intraday corrente; e a IA recebe `ctx.poc/val/vah` documentado como "volume_profile_daily" — a semântica de **volume profile** está correta, mas o "daily" parcial muda a cada ciclo, então mesmo para a IA a referência é instável. Renomear/documentar resolve esse lado.
- O commit 75bd3ec (iloc[-2]) **corrigiu o lugar certo** (`daily_pivot`), mas esse lugar é usado apenas por `_calculate_pivots` → `macro_context["pivots"]`, cujo resultado morre no dead wire; o caminho que de fato produz `event.pivot_points` (`enricher` → VP parcial) **não foi tocado pelo fix**.

---

## PASSO 4 — PROPOSTA DE MENOR PATCH (NÃO APLICADO)

Arquiteturas são **compatíveis** (ambas usam REST klines da Binance; `daily_pivot` aceita DataFrame de klines — o `_calculate_pivots` já é o adaptador). Dois patches mínimos:

1. **Ativar o caminho correto (1 linha)** — `market_orchestrator.py:1460`:
   ```python
   _pivot_data = macro_context.get("pivots", {}) or signal.get("contextual_snapshot", {}).get("pivots", {})
   ```
   → defense zones passariam a incluir `pivot_classic_*` reais (dia anterior completo) — impacta `ai_payload.sr` que a IA recebe.

2. **Fazer `event.pivot_points` refletir o período anterior** — no `_build_pivot_points` (enricher.py:363): injetar `signal["pivots"] = macro_context.get("pivots", {})` no `market_orchestrator` e usar como fonte primária (`pivot_data[period]` com H/L/C do iloc[-2]); manter VAH/VAL/POC (VP) como fallback — OU renomear o campo do evento para `volume_profile_intraday` e documentar em `ai_field_legend.py`, se a preferência for preservar o VP como está (componente B).

Recomendação: fazer o **patch 1 + patch 2** (corrigir o campo para o que o nome promete) e, em paralelo, ajustar docstrings do profiler e da legenda para distinguir `volume_profile_intraday` (dinâmico) de `daily_pivot_closed` (clássico fixo).

**Arquivos tocados (proposta):** `market_orchestrator/market_orchestrator.py:1460`, `institutional/enricher.py:363-442`, `common/ai_field_legend.py` — todos pendentes de prompt de autorização.

---

## PASSO 5 — VALIDAÇÃO PÓS-FIX EM PRODUÇÃO (2026-08-10, commit aa97cf1 + 3986822)

**Fix aplicado** (`fix(pivots): corrigir dead wire do pivot classic (duplo bug de pivot_points)`): ativação do caminho clássico (patch 1 + patch 2 da proposta) + `calculated_at_ms` de rastreabilidade.

**Observação real** (`scripts/diagnostics/run_production_observation.py`): 35.1 min conectado ao stream real, 00:41:17Z → 01:16:23Z, 47.431 trades, 0 OOO, 0 clamps, 0 invalid.

### Resultados

| Verificação | Resultado |
|---|---|
| Eventos pós-fix (ANALYSIS_TRIGGER) | 15 |
| `source="classic"` | **15 / 15 (100%)** |
| `source="vp_fallback"` | **0** |
| Estabilidade do pivot (1º ev 112 vs último ev 129) | **SIM** — `65035.38` idêntico nos 15 |
| `calculated_at_ms` presente | SIM — 3 valores únicos (01:01:55Z, 01:07:01Z, 01:12:08Z), intervalos **306s e 307s ≈ ciclo de 300s** do `CONTEXT_UPDATE_INTERVAL_SECONDS` |
| Divergência vs cálculo manual `(H+L+C)/3` (vela 1d de 2026-08-09 via API: H=65474.46, L=64730.08, C=64901.59) | **0.0%** — gravado `65035.38` = manual `65035.38` exato |
| `validate_production_run.py` B.6 | **PASS** (classic=16 dos 20 últimos eventos, 0 fallback) |

### Notas

- Os 3 eventos pós-fix **sem** `pivot_points` (ev 113, 118, 124) são `Alerta`/`AI_ANALYSIS` — tipos que não passam pelo institutional enricher (só `ANALYSIS_TRIGGER` carrega `pivot_points`). Não são fallback.
- `r1`/`s1` gravados são flutuantes (`65340.673333…`): o enricher propaga os níveis clássicos com `round(pivot, 2)` no pivot mas sem arredondar `r1`/`s1` no ramo classic (mesmo padrão pré-fix do VP). Sem impacto funcional; pode ser polido depois.
- Primeiro evento pós-fix (ev 112) já nasce com `source="classic"`: o ciclo de `_calculate_pivots` do 1º boot (01:01:55Z) foi consumido antes do primeiro ANALYSIS_TRIGGER.
- **Conclusão**: duplo bug (dead wire + fallback VP parcial rotulado como clássico) **corrigido e validado em produção real** — pivot agora é o clássico do período anterior completo, fixo durante o dia, com rastreabilidade de cálculo.

### Validação manual do cálculo (API Binance klines 1d, limit=3)

```
vela 2026-08-08: H=65192.54  L=64784.19  C=64962.60   (iloc[-3])
vela 2026-08-09: H=65474.46  L=64730.08  C=64901.59   (iloc[-2] = anterior completa)
vela 2026-08-10: H=65322.58  L=64826.78  C=65012.01   (dia corrente, em andamento)
pivot = (65474.46 + 64730.08 + 64901.59) / 3 = 65035.38 ✓ (gravado em todos os 15 eventos)
```
