# CFTC/CME COT — P5.1 Temporal Hardening + Shadow (2026-09-15)

`ENABLE_CFTC_COT_CONTEXT` permanece `false`. Nenhum sinal direcional. Nenhuma
mudança em execution/risk/sizing/ML.

## 1. Quatro instantes (CFTC e Binance)

- `report_as_of_date` / `source_as_of`: referência da fonte (terça CFTC;
  max source_timestamp Binance). Nunca é disponibilidade.
- `first_seen_at` (CFTC): primeiro instante em que ESTA versão foi observada.
  Revisão tem `first_seen` próprio; re-fetch preserva; restart carrega do
  cache (`test_shadow_first_seen_lifecycle`). Correção P5.1: revisão herda
  **novo** `first_seen`, nunca o da v0 (bug encontrado pelos testes novos).
- `retrieved_at`: quando a resposta foi recebida (CFTC ingest; Binance
  `snapshot.retrieved_at` no término do fetch — P5.1 MEDIUM-2).
- `analyzed_at` (`observed_at` legado no Binance): quando a análise rodou.
  `retrieved_at` nunca é preenchido com `analyzed_at`.

## 2. availability_basis (HIGH-1)

`select_point_in_time` retorna sempre: `record, reason, report_as_of_date,
revision, effective_available_at, availability_basis, first_seen_at,
calendar_estimated_at`. Basis ∈ `FIRST_SEEN | CALENDAR_FALLBACK | NONE`
(`OFFICIAL_PUBLICATION` reservado, nunca emitido — sem `published_at`).
`calendar_estimated_at` nunca é escrito em `first_seen_at`.

## 3. Strict mode (HIGH-2)

Default: `strict_point_in_time=True`, `allow_calendar_fallback=False`.
Sem `first_seen` real → inelegível (`UNAVAILABLE_FOR_POINT_IN_TIME` quando
há registros não-provados). Opt-in explícito só para pesquisa, marcado
`CALENDAR_FALLBACK` + `estimated_availability`. Nenhum caminho produtivo,
shadow ou LLM habilita fallback. Limitação assumida: calendário cobre só
feriados de sexta em 2025–2026; atraso de semana com feriado em outro dia
NÃO é previsto — por isso o fallback nunca é evidência.

## 4. Cache single-writer (MEDIUM-1)

Invariante: **um writer por diretório** (um bot por working dir).
Defesa: lock interprocess stdlib (fcntl/msvcrt, padrão `event_saver`),
não-bloqueante — segundo writer falha fechado (`cache_write_errors++`,
memória intacta). Save faz merge disco+memória (união por
revision+hash); `tmp`+`os.replace` mantido; `.tmp` órfão nunca lido.
Contadores `cache_write_errors`/`cache_lock_contended` expostos para saúde.

## 5. MEDIUM-3 — idades CFTC

`age_reference_seconds` = now − asof. `age_available_seconds` = now −
`first_seen` real, ou `null` sem ele. `quality.estimated_availability`
= `True` quando disponível sem `first_seen`. Payload `cftc` (ainda
desligado) levará só `age` de referência até P6 futuro.

## 6. Histórico: research vs strict

`select_research_history` (ordenado por asof, `point_in_time_guarantee:
False`) é só exploração. Backtest estrito exige `select_point_in_time`
strict + `FIRST_SEEN`; sem `first_seen` histórico, usar margem conservadora
documentada ou aguardar shadow contínuo. `research_history` nunca pode ser
reportado como backtest livre de look-ahead.

## 7. Shadow contínuo

`scripts/analytics/cftc_cot_shadow_collector.py` usa só observação real
(fetch → ingest com `first_seen` preservado → SQLite); nunca chama
`expected_publication_utc`, nunca cai em fallback. `shadow_health(db)`
expõe: `last_successful_fetch, last_first_seen, current_report_as_of,
current_revision, fetch_errors, schema_errors, cache_write_errors (via
cache_metrics opt-in), shadow_observation_count`. Sem Prometheus
 Cardinalidade mínima por desenho. Shadow só busca/versiona/persiste/mede;
payload/LLM/decisão intocados (flag OFF).
