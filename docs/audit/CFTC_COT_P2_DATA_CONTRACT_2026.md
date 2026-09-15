# CFTC/CME COT — P2 Data Contract (2026-09-15)

Independente do `CryptoCOT` Binance. Terminologia oficial TFF. Sem `smart money`/`varejo`/direção implícita.
Base oficial: `docs/audit/CFTC_COT_P1_OFFICIAL_DISCOVERY_2026.md` (TFF `gpe5-46if`, futures_only, codes `133741/133742/146021/146022`).

## 1. Identidade

| Campo | Tipo | Nullable | Regra |
|---|---|---|---|
| `schema_version` | int | não | `1` |
| `source` | str | não | `"cftc_cot"` |
| `source_kind` | str | não | `"cftc_socrata"` |
| `report_family` | str | não | `"TFF"` (fallback documentado `"Legacy"` só com flag explícita) |
| `report_scope` | str | não | `"futures_only"` (canônico) |
| `symbol` | str | não | símbolo interno Binance (`BTCUSDT`/`ETHUSDT`/`MBTUSDT`? ver §8) |
| `market_and_exchange_name` | str\|null | sim | literal oficial; null quando `UNSUPPORTED` |
| `cftc_contract_market_code` | str\|null | sim | `133741/133742/146021/146022`; null quando `UNSUPPORTED` |

## 2. Temporalidade (nunca inventar `published_at`)

| Campo | Tipo | Nullable | Regra |
|---|---|---|---|
| `report_as_of_date` | str(YYYY-MM-DD)\|null | sim | terça de referência (`report_date_as_yyyy_mm_dd`); null se sem cobertura |
| `published_at` | — | — | **AUSENTE por definição.** CFTC não publica o campo. Não serializar nem como null para não sugerir existência |
| `first_seen_at` | str ISO UTC\|null | sim | coleta viva: quando o fetch observou o registro pela 1ª vez; null em replay sem esse dado |
| `retrieved_at` | str ISO UTC | não | quando este snapshot foi montado |
| `analyzed_at` | str ISO UTC | não | quando a interpretação foi calculada |
| `age_reference_seconds` | float\|null | sim | `retrieved_at - report_as_of(UTC midnight)`; null sem `report_as_of_date` |
| `age_available_seconds` | float\|null | sim | `retrieved_at - first_seen_at`; null sem `first_seen_at` |

## 3. Status

`status ∈ {AVAILABLE, PARTIAL, STALE, UNAVAILABLE, UNSUPPORTED, INVALID}`.

| Status | `is_available` | `is_stale` | Significado |
|---|---|---|---|
| `AVAILABLE` | true | false | todas as 5 categorias + OI presentes e dentro da freshness calendar-aware |
| `PARTIAL` | true | false | alguma categoria ausente mas OI + ≥1 categoria presentes; `quality.missing_fields` lista tudo; **números parciais nunca apresentados como completos** |
| `STALE` | true | true | último válido além da freshness (relatório perdido); idade explícita |
| `UNAVAILABLE` | false | false/true | fetch falhou sem cache válido, ou contrato abaixo do limiar de 20 traders (`error_code=below_reportability_threshold`) |
| `UNSUPPORTED` | false | false | símbolo sem mapeamento CME (`error_code=unsupported_symbol`) |
| `INVALID` | false | false | schema/número inválido (`error_code` + `quality.validation_errors`) |

`UNKNOWN` (Binance) **não** é reutilizado aqui — os 6 estados acima são exaustivos para CFTC.

## 4. Posições (terminologia oficial, sem renomear)

```text
positions.dealer          ← dealer_positions_long/short/spread_all
positions.asset_manager   ← asset_mgr_positions_long/short/spread
positions.leveraged       ← lev_money_positions_long/short/spread
positions.other           ← other_rept_positions_long/short/spread
positions.nonreportable   ← nonrept_positions_long/short_all (sem spread)
```

Cada categoria: `{long: int|null, short: int|null, spreading: int|null, net: int|null (=long-short), share_oi_long: float|null, share_oi_short: float|null}`.
Regras: inteiros ≥0 ou null; `net` só quando ambos presentes; `share = pos/open_interest_total` (null se OI ausente/zero); nunca percentuais pré-computados da CFTC como fonte de verdade para `net` (recalcular); `change_in_*`/`pct_of_oi_*`/`traders_*`/`conc_*` vão para `provenance.raw_excerpt` (auditoria), não para decisão.

## 5. Open interest e derivados mínimos

- `open_interest: {total: int|null, change_wow: int|null}` (`change_wow` = total − total da semana anterior **do mesmo contrato**, null sem anterior).
- `derived_metrics` (P3/P7, só com histórico suficiente): `{wow_net_change: {dealer, asset_manager, leveraged, other, nonreportable} (int|null), week_over_week_report: {prev_report_as_of_date|null}}`. **Nenhum percentil/COT Index nesta fase** (requer P7 com min_periods).
- Proibido: agregar standard+micro em contratos; proibido forward-fill; proibido zero em lacuna.

## 6. Qualidade / proveniência / erro

- `quality: {missing_fields: str[], validation_errors: str[], revision: int, is_revision: bool, weeks_missing_in_window: int|null}`.
- `provenance: {dataset_id: "gpe5-46if", query: str, source_row_id: str (ex. 260908133741F), contract_units: str, retrieved_at, cache_hit: bool, content_hash: str (sha256 do raw canônico), revision_of: str|null}`.
- `error_code: null | unsupported_symbol | fetch_error | http_429 | http_5xx | schema_error | below_reportability_threshold | stale_cache | invalid_number`.

## 7. Validação (fail-closed, RFC 8259)

1. `cftc_contract_market_code` deve estar no allowlist `133741/133742/146021/146022`; `market_and_exchange_names` deve ser consistente (comparação normalizada trim+single-space, mas code prevalece).
2. Números Socrata chegam como **string** (`"21083"`, `"-96"`, `"61.8"`): parse `str→int/float`, rejeitar `bool`, `""`, `NaN/Inf`, overflow; `0` preservado; null preservado.
3. `report_date_as_yyyy_mm_dd` deve ser terça (`weekday()==1` em America/New_York); senão `INVALID` + `validation_errors`.
4. Campos desconhecidos: preservar em `provenance.raw_excerpt` (amostra), nunca promover a decisão; teste de schema-drift deve falhar em CI, não em produção.
5. Duplicidade `(report_as_of_date, code, scope)`: primeira ocorrência vence; segunda com conteúdo distinto vira `revision+1` com novo `first_seen_at`; **revisão nunca reescreve** a versão persistida.
6. Histórico insuficiente: qualquer métrica WoW/percentil sem base ⇒ `null` + `quality` explícito.
7. Serialização: `json.dumps(..., allow_nan=False)` deve passar; nenhum `NaN/Inf`.

## 8. Mapeamento symbol → contrato

| `symbol` | code | `market_and_exchange_name` |
|---|---|---|
| `BTCUSDT` | `133741` | `BITCOIN - CHICAGO MERCANTILE EXCHANGE` |
| `MBTUSDT` (se adotado) | `133742` | `MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE` |
| `ETHUSDT` | `146021` | `ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE` |
| `METUSDT` (se adotado) | `146022` | `MICRO ETHER - CHICAGO MERCANTILE EXCHANGE` |

Qualquer outro símbolo ⇒ `UNSUPPORTED` (nunca fallback para BTC/ETH, nunca fuzzy por nome). Micro só é consultado se o símbolo micro existir no roteamento do bot; por padrão o mapa ativo é `{BTCUSDT→133741, ETHUSDT→146021}`.

## 9. Fixtures

- `tests/fixtures/cftc_cot/tff_btc_futonly_2026-09-08.json` — linha real `260908133741F` minimizada (todos os campos §4 + OI + metadados de linha).
- `tests/fixtures/cftc_cot/tff_unsupported_doge.json` — caso `UNSUPPORTED` esperado.
- `tests/fixtures/cftc_cot/tff_btc_revision.json` — mesma `report_as_of_date` com 1 campo alterado (para testes de versionamento P3/P5).
