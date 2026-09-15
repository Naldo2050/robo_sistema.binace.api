# CFTC/CME COT — P5 Point-in-Time (2026-09-15)

## 1. Mecanismo

`institutional/cftc_cot.py:select_point_in_time(records, at)` — puro, sem I/O.
Visibilidade exige disponibilidade comprovada ≤ t:
1) `first_seen_at` real (coleta viva / raw cache), ou
2) calendário oficial (sexta 15:30 ET + feriado, `expected_publication_utc`) + grace 2h — somente se `allow_calendar_fallback=True`.
Sem `first_seen` e sem fallback → `UNAVAILABLE_FOR_POINT_IN_TIME` (nunca `report_as_of_date` como proxy).
Revisões do mesmo asof competem por `first_seen_at`: replay antigo sempre vê a versão original (teste `test_later_revision_does_not_rewrite_history`).

## 2. Cobertura de testes

`tests/unit/test_cftc_cot_point_in_time.py` — 13 casos: terça pré-publicação, sexta pré-15:30, pós-publicação, fim de semana, feriado (julho/2026), atraso de 3 dias, revisão 9 dias depois, first_seen ausente (com/sem fallback), DST out/2026 (EDT→EST: 19:30→20:30 UTC), fronteira UTC/ET minuto a minuto, duplicata, `expected_publication` da semana canônica 2026-09-08 = `2026-09-11T19:30Z`.

## 3. GATE P5

23/23 testes P3+P5 verdes, nenhum look-ahead detectado. **GATE P5: PASS → P6 autorizado.**
