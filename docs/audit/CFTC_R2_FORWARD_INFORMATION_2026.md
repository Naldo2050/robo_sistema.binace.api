# CFTC — R2 Forward Information Study (2026-09-15)

PESQUISA. Não é backtest de estratégia. Sem produção, sem LLM, sem sizing.
`research_history=true`, `point_in_time=false`. Preços: Binance USD-M
Futures diário (`BTCUSDT` desde 2019-09, `ETHUSDT` desde 2020-01 —
limitação registrada; COT anterior sem preço é descartado com flag).
800 combos (4 contratos × 5 categorias × 8 features × 5 horizontes;
A/B/C dentro de cada combo).

## 1. Disponibilidade e leakage

A = sexta 15:30 ET pós-asof (estimated se feriado/asof atípico),
B = segunda seguinte, C = terça seguinte. Entry = primeiro close
estritamente após availability (sábado no caso base); asserts em código:
`entry>available`, `exit>entry`, `asof<=available`, sem `shift(-1)`,
sem backfill, bisect manual (nunca `merge_asof forward`).
Nenhuma conclusão depende só de A (todas as 4 promissoras têm A=B=C
no mesmo sinal).

## 2. Classificação (800 combos)

PROMISING_RESEARCH_SIGNAL 4 · WEAK_STABLE 189 · WEAK_UNSTABLE 445 ·
NO_EVIDENCE 162. FDR-BH sobre todos os p-values; bootstrap em blocos
temporais (seed fixa); quintis só com N≥50; sem score combinado.

## 3. Top por robustez (não por |r|)

1. `133741/other/net_share_oi h28` (n=436, r=0.30/ρ=0.29, q≈0,
   A=B=C=+, 3/3 subperíodos +, boot [0.11,0.45], Q5−Q1=+0.20)
2. `133742/leveraged/net_share_oi h28` (n=276, r=0.24, q=0.005,
   A=B=C=+, 2/2 sub +, boot [0.04,0.45], Q5−Q1=+0.11)
3. `133742/leveraged/net_share_pct52 h28` (n=225, r=0.27, q=0.005,
   A=B=C=+, 2/2 sub +, boot [0.03,0.47], Q5−Q1=+0.12)
4. `133741/nonreportable/net_share_change_1w h3` (n=439, r=−0.15,
   q=0.089, A=B=C=−, 3/3 sub −, boot [−0.24,−0.06]) — única de
   horizonte curto, direção contrária (aumento de net nonreportable
   precede retorno menor em 3d).

## 4. Leituras agregadas

- Level vs Flow: 3/4 promissoras são NÍVEL (`net_share_oi`, percentil);
  flow aparece só no caso contrarian nonreportable h3. Nível > fluxo
  nesta amostra, com a exceção registrada.
- Extremos: h28 mostra gradiente le10<…<ge90 nos 3 casos de nível
  (ex. lev micro: −0.038/−0.035/+0.025/+0.043/+0.101) — extremos
  carregam mais que o miolo, sem virar regra BUY/SELL.
- Standard vs micro: BTC micro/leveraged replica o padrão do standard
  (`other` std r=0.30 vs lev micro r=0.24, mesmo sinal e horizontes);
  ETH sem nenhum PROMISING (melhor: asset `net_share_oi` h28 r=−0.21,
  WEAK_STABLE). Micro adiciona confirmação, não contradição.
- Estabilidade: 445/800 trocam de sinal entre subperíodos
  (WEAK_UNSTABLE) — a maioria das relações é instável; as 4 acima são
  a exceção documentada.
- Robustez A/B/C: B e C replicam o sinal de A nos 4 casos.

## 5. Limitações (bloqueiam produção)

h28 com targets sobrepostos (autocorrelação tratada via blocos, mas
persistente); N semanal pequeno; preço futures só desde 2019/2020;
sem cruzamento com Binance positioning; sem custos/latência de execução.

## 6. Classificação global

**CANDIDATES_FOR_R3** — 4 relações de nível/contrarian para R3
(estudo fora-da-amostra + Binance), nenhuma para produção.
Sem BUY/SELL/STRONG em nenhum campo do JSON.
