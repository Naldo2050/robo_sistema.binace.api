# CFTC/CME COT — R1 Historical Research Dataset (2026-09-15)

RESEARCH_HISTORY. Sem promessa point-in-time (sem `first_seen` histórico).
Nunca reportar como backtest livre de look-ahead. Sem estudo de retorno,
sem correlação com preço, sem LLM, sem produção (`ENABLE_CFTC_COT_CONTEXT`
permanece `false`).

## 1. Cobertura por contrato (TFF Futures Only, `gpe5-46if`)

| Contrato | Code | First | Last | Semanas | Gaps | Dups | Nomes distintos |
|---|---|---|---|---|---|---|---|
| BTC standard | 133741 | 2018-04-10 | 2026-09-08 | 440 | 0 | 0 | 1 |
| BTC micro | 133742 | 2021-05-04 | 2026-09-08 | 280 | 0 | 0 | 1 |
| ETH standard | 146021 | 2021-04-06 | 2026-09-08 | 284 | 0 | 0 | 1 |
| ETH micro | 146022 | 2021-12-14 | 2026-09-08 | 248 | 0 | 0 | 1 |

Semanas "curtas" (asof de segunda por feriado US, ex. 2018-12-24,
2023-07-03): 5 no BTC standard, 2 em cada outro — informativas
(`non_tuesday_asof` + `short_intervals`), sem quebra de sequência.

## 2. Validação (R1.3)

- Duplicatas: 0 em todos. Revisões históricas: invisíveis na view atual
  (1 linha por asof); versionamento vale daqui para frente (raw cache P3).
- Valores negativos / NaN / Inf: 0 ocorrências no raw oficial.
- OI zero: 0. Categorias ausentes: 0.
- Mudanças de `market_name`: 0 (nomes estáveis; micro ETH tem 2 espaços,
  tratado por code, nunca por nome).
- `contract_code` inconsistente / standard-micro misturados: 0.
- Identidades TFF verificadas em 100% das linhas, ambos os lados:
  `sum(reportable)+sum(spread)==tot_rept` e `tot_rept+nonrept==OI`.
  Duas fórmulas erradas do próprio validador foram encontradas e corrigidas
  durante o build (soma incluía nonreportable; gap contava intervalos de
  feriado) — dados oficiais intactos, sem nenhuma correção silenciosa.

## 3. Schema e layout (R1.4, schema v1)

`dados/research/cftc/` (gitignored — dataset NÃO commitado):
`raw/{code}.json` (linhas oficiais + provenance), `normalized/{code}.parquet`
(1 linha por code+asof, parse defensivo + `quality_flags`),
`normalized/{code}_features.parquet`, `metadata/build_metadata.json`
(provenance R1.2: source CFTC, dataset `gpe5-46if`, TFF/futures_only,
`retrieved_at`, `research_history:true`, `point_in_time:false`, sem
`first_seen` fabricado).

## 4. Features (R1.5)

Por categoria: `net`, `share_long/short/net_oi` (null se OI≤0),
`net_change_1w/4w` e `net_share_change_1w/4w` (só janelas consecutivas
reais, sem forward-fill), percentis 26/52/156 de `net_share_oi`
(min_periods=janela cheia, null se insuficiente/gap/null).
Sem `signal/side/BUY/SELL/confidence` em nenhum campo.

## 5. Standard vs micro (R1.6, descritivo — sem preço)

BTC (280 semanas comuns): `net_share` dealer 0.31, leveraged 0.34,
asset 0.10, other 0.08, nonrep 0.24; WoW ≈ 0 ou negativo em todas.
ETH (248 semanas): asset 0.52 e nonrep 0.53 em nível, mas WoW ≤ 0.26;
other −0.23 em nível. OI cresceu ~12x (BTC std e micro), ~15x (ETH std),
~10x (ETH micro). Leitura: micro **não** é redundante — níveis
moderadamente alinhados em algumas categorias, fluxos semanais
praticamente ortogonais. Nenhuma agregação entre contratos foi feita.

## 6. Limitações

Revisões passadas invisíveis; `first_seen` só prospectivo; feriados deslocam
asof (segundas); percentil 156 exige 3 anos (BTC ok, ETH micro parcial);
amostra semanal pequena para inferência (52 obs/ano).

## 7. Classificação

**DATASET_READY_FOR_RESEARCH** (research_history; strict point-in-time
segue pendente de `first_seen` prospectivo).
