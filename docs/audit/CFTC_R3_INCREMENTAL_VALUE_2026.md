# CFTC — R3 Incremental Value, PSEUDO_OUT_OF_SAMPLE (2026-09-15)

PESQUISA. Nenhuma mudança em produção/LLM/ML/execution/risk/sizing.
OOS = PSEUDO (R2 selecionou H1–H4 usando 2024–2026; holdout 2024+ não é
untouched; nenhum parâmetro recalibrado no holdout).

## 1. Desenho

Dev: asof ≤ 2023-12-31. Teste: asof ≥ 2024-01-01 (H1/H2 n≈221/141,
H3/H4 n≈139/137). Baseline price trailing (ret 1/7/28d, vol 28d,
EMA50) + funding (histórico total, 7686 registros). OI-change excluído
dos modelos (endpoint só retorna ~31 dias — documentado, sem preencher).
Binance positioning excluído (sem histórico — tabela inexistente).
Modelos OLS/HAC (Newey-West) + logística secundária; sem XGBoost/busca.

## 2. Vereditos

| H | Classe | Direção | ΔR² C−A | ΔR² D−B | Holm | VIF | Parcial |
|---|---|---|---|---|---|---|---|
| H1 other h28 | REJECTED | dev beta − (esperado +) | −0.011 | +0.025 | 0.82 | 1.15 | +0.10 |
| H2 nonrep h3 | REJECTED | B/C invertem vs A | +0.017 | +0.020 | 0.11 | 1.02 | −0.10 |
| H3 lev micro h28 | WEAK_INCREMENTAL | A=B=C=+ | +0.118 | +0.153 | 0.76 | 1.03 | +0.27 |
| H4 lev pct52 h28 | WEAK_INCREMENTAL | A=B=C=+ | +0.047 | +0.174 | 0.60 | 1.02 | +0.27 |

H1: o sinal marginal positivo do R2 inverte condicional ao baseline —
rejeitada pela regra de direção (R3.14), não por falta de incremento.
H2: promissora no R2, mas o sinal inverte em B/C — rejeitada (R3.10).
H3/H4: direção, incremento (C−A e D−B), LOO-anos, winsor e VIF ok;
**Holm confirmatório falha (0.76/0.60)** — por isso WEAK, nunca ROBUST.
Direction accuracy (teste): H3 D 0.65 vs B 0.54; H4 D 0.60 vs B 0.54
(secundário, sem threshold).

## 3. Ablação / redundância / regimes / complementaridade

C vs A e D vs B acima. CFTC×funding/OI/retornos: VIF ≈ 1.0–1.15
(não redundante; positioning Binance intestável por falta de overlap).
Regimes (teste, N≥30): H3/H4 positivos em bull/bear/high/low-vol/funding+
(0.25–0.59) — sem concentração num regime só; OI-regime INSUFFICIENT_SAMPLE.
Joint H1+H3 e H3+H4 colapsam OOS (−0.03/−0.28) apesar de betas dev
significativos — overfit por colinearidade; H3 absorve H4. Interação
pré-especificada nula.

## 4. Global e limites

**WEAK_INCREMENTAL_VALUE** (2 fracas, 0 robustas). h28 sobreposto limita
inferência; amostra semanal pequena; sem custos de execução; sem cruzar
com Binance positioning (overlap inexistente). Nada autoriza produção.
