# Binance Positioning B2 — Lossless Prospective Dataset (2026-09-15)

Research/data infra. Produção intocada. Flag CFTC False. Sem threshold,
sem sinal, sem nomenclatura retail/institucional/smart-money.

## 1. Seed BTCUSDT (execução real)

| Endpoint | First | Last | Rows | Gaps | Dups | Revisões | HTTP err |
|---|---|---|---|---|---|---|---|
| global | 2026-08-15 | 2026-09-15 | 8928 | 0 | 0 | 0 | 0 |
| top_account | 2026-08-15 | 2026-09-15 | 8928 | 0 | 0 | 0 | 0 |
| top_position | 2026-08-15 | 2026-09-15 | 8928 | 0 | 0 | 0 | 0 |
| oi | 2026-08-15 | 2026-09-15 | 8928 | 0 | 0 | 0 | 0 |
| funding | 2019-09-02 | 2026-09-15 | 7686 | 0 | 0 | — | 0 |

19 páginas/endpoint. Achado operacional: `startTime` além de ~30d retorna
400/-1130 — paginação é backward via `endTime` (documentado no script).
Normalized 5m: 8928 snapshots, 100% complete, 0 partial, 0 revisions.
Backfill: `first_seen_at=null`, `usable_for_strict_pit=false`.

## 2. Live validation (1 ciclo)

+1 barra/endpoint (499 duplicatas de sobreposição suprimidas), +1 snapshot
complete LIVE_OBSERVED com `first_seen` real (strict PIT elegível: 1).
Total: 8929 snapshots, complete_rate 1.0.

## 3. Decisões

Raw JSONL append-only por endpoint (não SQLite único; não Parquet reescrito).
Grade canônica = união dos ts; join backward com tolerância explícita
10 min, nunca forward; ausente vira null+missing (linha preservada).
Revisão = mudança no dado (modo excluído do hash — bug encontrado e
corrigido pelos testes). Lock: single-writer por diretório (coletor 15m).
Clock: `retrieved_at` local + `clock_suspect` se retrieved < source.
Derivações 15m/1h a partir do raw 5m (raw nunca descartado).

## 4. Pronto para coleta contínua: YES (manual/agendada; sem daemon auto).
