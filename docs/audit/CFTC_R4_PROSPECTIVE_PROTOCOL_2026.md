# CFTC — R4 Prospective Protocol (TRUE OOS, 2026-09-15)

Manifesto imutável: `config/cftc_r4_manifest.json` (H3 primária
`133742/leveraged/net_share_oi` h28 +; H4 pct52 só diagnóstica).
Métricas e sucesso congelados no manifesto (Spearman principal;
checkpoints N=4/13/26/52; sem optional stopping; sem threshold em feature).

## Pipeline

`cftc_r4_prospective_collector.py` (observa; primeira rodada 2026-09-15:
último relatório asof 2026-09-08 é pré-freeze → 0 observações, correto) →
`cftc_r4_outcome_updater.py` (matura PENDING após exit ≥ entry+28d) →
`cftc_r4_status.py` (contagens, próximo checkpoint; sem significância).

Store append-only em `dados/research/cftc/r4_prospective/` (gitignored):
`observations.jsonl`, `outcomes.jsonl`, `revisions.jsonl`, `binance.jsonl`.
IDs `code_asof_rN` imutáveis; duplicata rejeitada; restart recarrega.

## Regras temporais

`feature_available_at = first_seen_at` real (calendar proibido, verificado
por teste de ausência no módulo). Entry = primeiro candle diário BTCUSDT
USD-M com open ESTRITO após first_seen (candle ambíguo pulado; naive
rejeitado). Revisão preserva original e registra à parte. Binance paralelo:
snapshot ao vivo como aux com flag (posterior ao first_seen), funding
estrito via histórico (`fundingTime <= first_seen`), OI null+flag
(histórico insuficiente). Sem backfill, sem threshold, sem sinal.
