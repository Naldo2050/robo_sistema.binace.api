# CFTC/CME COT — P1 Official Discovery (2026-09-15)

Fontes exclusivamente oficiais: `cftc.gov`, `publicreporting.cftc.gov` (Socrata), `cmegroup.com` (apenas specs de contrato).
Nenhum blog/site de trading usado como autoridade. Verificação executada via leitura direta de páginas oficiais e queries Socrata live em 2026-09-15.

## 1. Contratos BTC/ETH com cobertura oficial atual

| Symbol interno | `market_and_exchange_names` (oficial) | `cftc_contract_market_code` | Exchange | Status |
|---|---|---|---|---|
| BTC (standard) | `BITCOIN - CHICAGO MERCANTILE EXCHANGE` | `133741` | CME | Ativo (2026-09-08 OI=21083 FutOnly) |
| BTC (micro) | `MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE` | `133742` | CME | Ativo (2026-09-08 OI=35179 FutOnly) |
| ETH (standard) | `ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE` | `146021` | CME | Ativo (2026-09-08 OI=26564 FutOnly) |
| ETH (micro) | `MICRO ETHER - CHICAGO MERCANTILE EXCHANGE` | `146022` | CME | Ativo (2026-09-08 OI=63819 FutOnly) |

Evidência (queries live):
- `https://publicreporting.cftc.gov/resource/6dca-aqww.json?$where=upper(market_and_exchange_names) like '%BITCOIN%'` → rows `133741` + `133742` em `2026-09-08`.
- `.../gpe5-46if.json?$where=... like '%BITCOIN%'` → mesmos dois + `contract_market_name` = `BITCOIN` / `MICRO BITCOIN`.
- `.../6dca-aqww.json?$where=... like '%ETHER%'` → `146021 ETHER CASH SETTLED`, `146022 MICRO ETHER` (+ Coinbase `146LM1/146LM3`, fora de escopo — não-CME).
- Coinbase (`133LM5`, `146LM1`, `146LM3`, Nano etc.) existe mas é **outra exchange** — mapeamento interno cobre **somente CME**.

## 2. Família de relatório

- **TFF (Traders in Financial Futures): SIM.** BTC e ETH aparecem em `gpe5-46if` (TFF Futures Only) e `yw9f-hn96` (TFF Combined). Linha completa `133741/2026-09-08` retornada com `dealer_positions_long_all=6799 ... lev_money_positions_short=13038 ... commodity_subgroup_name=DIGITAL ASSET, commodity_group_name=FINANCIAL INSTRUMENTS, futonly_or_combined=FutOnly`.
- **Legacy: SIM.** Mesmos contratos em `6dca-aqww` (Futures Only) e `jun7-fc8e` (Combined). Linha `133741/2026-09-08` com `noncomm_positions_long_all=17600, comm_positions_long_all=64 ... futonly_or_combined=FutOnly`.
- **Disaggregated: NÃO.** `https://publicreporting.cftc.gov/resource/72hh-3qpy.json?$where=... like '%BITCOIN%'` → `[]`. Consistente com a definição oficial (Disaggregated = agriculture/petroleum/natgas/electricity/metals): https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm (seção Types of Reports, item 3).
- **Supplemental CIT: NÃO** (13 agricultural contracts): mesma fonte, item 2.
- **Decisão P1: família canônica = TFF** (granularidade Dealer/AssetMgr/Leveraged/Other + Nonreportable, adequada a cripto financeiro). Legacy mantido como fallback/validação cruzada. Disaggregated descartado para cripto.

Categorias oficiais TFF (fonte: mesma página, item 4): `Dealer/Intermediary`, `Asset Manager/Institutional`, `Leveraged Funds`, `Other Reportables` (+ `Nonreportable Positions`). Nomes de campo Socrata: `dealer_positions_*`, `asset_mgr_positions_*`, `lev_money_positions_*`, `other_rept_positions_*`, `nonrept_positions_*`, cada um com `long/short/spread` (+ `change_in_*`, `pct_of_oi_*`, `traders_*`).

## 3. Futures Only vs Combined

Ambos existem e **divergem** (prova de que são scopes distintos):
- `133741/2026-09-08`: FutOnly OI=`21083` (`6dca-aqww` e `gpe5-46if`) vs Combined OI=`21498` (`jun7-fc8e` e `yw9f-hn96`).
- Datasets: FutOnly `gpe5-46if` (TFF) / `6dca-aqww` (Legacy); Combined `yw9f-hn96` (TFF) / `jun7-fc8e` (Legacy).
- **Decisão P1: scope canônico = `futures_only`** (posições puras, sem delta-adjust de opções). Combined apenas para referência/auditoria.

## 4. `market_and_exchange_name` exato (copiar literal, sem fuzzy)

- `BITCOIN - CHICAGO MERCANTILE EXCHANGE`
- `MICRO BITCOIN - CHICAGO MERCANTILE EXCHANGE`
- `ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE`
- `MICRO ETHER - CHICAGO MERCANTILE EXCHANGE` (nota: TFF retorna com 2 espaços `MICRO ETHER  - ...` em uma linha — normalizar trim+single-space na comparação, mas mapear por `cftc_contract_market_code`, nunca por nome).

## 5. Contract codes

`133741` (BTC), `133742` (MBT), `146021` (ETH), `146022` (MET). Campo Socrata: `cftc_contract_market_code` (type `text`).

## 6. Standard vs micro: tamanhos

- BTC standard: `(5 Bitcoins)` — campo `contract_units` da própria linha Socrata `133741` (oficial CFTC).
- Micro BTC: `0.10 bitcoin` — CME fact card oficial: https://www.cmegroup.com/content/dam/cmegroup/markets/cryptocurrencies/files/micro-bitcoin-and-ether-options-fact-card.pdf (`futures contract = 0.10 bitcoin`), corroborado por `contract_units=(Bitcoin X $0.10)` na linha `133742`.
- ETH standard: `50 ether` — https://www.cmegroup.com/markets/cryptocurrencies/ether/ether (`Contract Unit. 50 ether`).
- Micro ETH: `0.10 ether` — mesmo fact card CME (`futures contract = 0.10 ether`).

## 7. Agregados ou separados?

**Separados.** Na mesma `report_date` há uma linha por code (ex. `2026-09-08`: `133741` OI 21083 + `133742` OI 35179). Não existe linha consolidada BTC na API. **Proibido somar standard+micro em unidades de contrato** (denominadores diferentes: 5 vs 0.1). P2/P3 devem expor por contrato e, se necessário, converter para notional BTC/ETH antes de qualquer agregação — sem agregação na P3.

## 8-9. Dataset Socrata exato + IDs

| Família | Scope | Dataset ID | Verificação |
|---|---|---|---|
| TFF | Futures Only | `gpe5-46if` | queries acima retornam BTC/ETH |
| TFF | Combined | `yw9f-hn96` | `133741/2026-09-08` OI 21498 |
| Legacy | Futures Only | `6dca-aqww` | `133741/2026-09-08` full row |
| Legacy | Combined | `jun7-fc8e` | `133741` OI 21498 |
| Disaggregated | FutOnly | `72hh-3qpy` | BTC query → `[]` |
| Disaggregated | Combined | `kh3c-gbw2` | não aplicável (sem cripto) |
| TFF_All | ambos | `udgc-27he` | view pai (`modifyingViewUid` de `gpe5-46if`); exige filtro `futonly_or_combined` — **não usar** (risco double-count) |

Base URL: `https://publicreporting.cftc.gov/resource/<ID>.json`. Metadados: `https://publicreporting.cftc.gov/api/views/<ID>`.

## 10. Schema/campos reais (TFF FutOnly `gpe5-46if`)

87 colunas (metadados `api/views/gpe5-46if`: 46.551 linhas, `smallest report_date 2006-06-13`). Campos mínimos confirmados na linha real:
`id, market_and_exchange_names, report_date_as_yyyy_mm_dd (calendar_date), yyyy_report_week_ww, contract_market_name, cftc_contract_market_code, cftc_market_code, cftc_region_code, cftc_commodity_code, commodity_name, open_interest_all, dealer_positions_{long,short,spread}_all, asset_mgr_positions_{long,short,spread}, lev_money_positions_{long,short,spread}, other_rept_positions_{long,short,spread}, tot_rept_positions_long_all, tot_rept_positions_short, nonrept_positions_{long,short}_all, change_in_*, pct_of_oi_*, traders_*, conc_gross/net_le_{4,8}_tdr_{long,short}(_all), contract_units, cftc_subgroup_code, commodity, commodity_subgroup_name, commodity_group_name, futonly_or_combined`.
`id` = `<YYMMDD><code><F|C>` (ex. `260908133741F`) — id de linha, **não** primary key global (FAQ: "There is no primary key in the dataset").

## 11. Representação numérica

Metadados declaram `number`, mas o **JSON entrega strings**: `"open_interest_all":"21083"`, `"change_in_other_rept_long":"-96"`, `"pct_of_oi_lev_money_short":"61.8"`. Contratos devem fazer parse defensivo string→número (rejeitar bool/NaN/Inf/vazio; preservar `0`; nunca `float("nan")`). Datas: `"2026-09-08T00:00:00.000"` (midnight, sem TZ — interpretar como data de referência, não timestamp).

## 12. `report_as_of_date`

Campo `report_date_as_yyyy_mm_dd` = terça-feira de referência (ex. `2026-09-08` = terça). É a **data do snapshot**, não da disponibilidade.

## 13. Publication timestamp

**Não existe** no dataset. Nenhum campo `published_at/release_at` nas 87 colunas nem nos metadados. A disponibilidade deve ser derivada de (prioridade): 1) calendário oficial + 2) `first_seen_at` próprio. Nunca inventar `published_at`.

## 14. Paginação real (verificada)

SoQL funciona: `$select`, `$where` (com `upper(...) like`), `$order`, `$limit` usados com sucesso nas queries acima. `$offset` é o mecanismo padrão Socrata para páginas seguintes (mesma API; semântica documentada no COT PRE User Guide: https://publicreporting.cftc.gov/stories/s/COT-Help/p2fg-u73y/). Estratégia P3: `limit=1000` + `offset` até esvaziar, ordenado por `report_date_as_yyyy_mm_dd ASC`, filtro por `cftc_contract_market_code` (4 valores). Limite máximo exato do servidor não publicado — paginação defensiva resolve independente do teto.

## 15. Limites/restrições oficiais

Sem token: "Currently, we are not providing tokens... as long as you are not overusing the API, you should be able to use the API without a token" (https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm, FAQ 13). Sem quota numérica publicada → P3 usa throttle conservador (≤1 req/2s, backoff em 429/5xx, cache agressivo) e contato `publicreporting@cftc.gov` em caso de bloqueio.

## 16. Revisões/correções

"No, historical data is not updated once published." (mesma página, FAQ 6 Corrected Data). Na prática: tratar `(report_date, contract_code, scope)` como imutável; se o mesmo `report_as_of_date` reaparecer com `id`/conteúdo distinto, versionar como `revision N` com `first_seen_at` novo e **nunca reescrever** a versão já persistida (P3 raw cache append-only).

## 17. Calendário normal

Quarta-feira de manhã CFTC recebe dados → sexta-feira à tarde publica. "Generally, the data in the COT reports is from Tuesday and released Friday." Publicação **sexta 15:30 Eastern** com dados da terça imediatamente anterior (mesma página, FAQ 5).

## 18. Atrasos em feriados

"Federal holidays may delay release by one or two days." Calendário tentativo 2026 com `*Delayed release date` (ex. Jan 05*, Jun 22*, Jul 06*, Nov 16*/30*, Dec 28*): https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm. P3/P5: freshness **calendar-aware** (próxima publicação esperada = sexta 15:30 ET + deslocamento de feriado + grace), nunca TTL fixo cego.

## 19. Timezone oficial

**Eastern Time (US)** — `3:30 p.m. Eastern time` (release schedule). Com DST (EDT UTC-4 / EST UTC-5). Persistir `report_as_of_date` como data + `available_at` derivado em UTC com conversão explícita America/New_York→UTC (testes DST em P5).

## 20. Disponibilidade histórica

- TFF/Disaggregated desde `2006-06-13`; Legacy desde `1986-01-15`; CIT desde `2006-01-03` (FAQ 12).
- BTC `133741`: primeira linha `2018-04-10` (query ASC). Micro BTC `133742`: `2021-05-04`. ETH standard `146021`: `2021-04-06`. Micro ETH `146022`: `2021-12-14`. Contagem `133741` em TFF FutOnly: 440 linhas semanais.
- Limite operacional: ≥20 large traders por mercado, senão o contrato some do relatório da semana (FAQ 1) → `UNAVAILABLE` pontual com `error_code=below_reportability_threshold` (distinto de `UNSUPPORTED`).

## GATE P1

- dataset: `gpe5-46if` (canônico) + `6dca-aqww` (fallback) — DETERMINADO
- report family: TFF — DETERMINADO
- contract codes: `133741/133742/146021/146022` — DETERMINADO
- scope: `futures_only` — DETERMINADO
- campos mínimos: §10 — DETERMINADO

**GATE P1: PASS → autorizada P2.**
