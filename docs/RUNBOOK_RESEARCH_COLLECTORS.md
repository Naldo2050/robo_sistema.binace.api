# Runbook — Research Collectors (B3)

> **RESEARCH-ONLY.** Estes datasets NÃO alimentam trading, LLM, ML,
> execution, risk, sizing ou confluence. Consumidores autorizados: análises
> R1–R4 e pesquisa offline. `ENABLE_CFTC_COT_CONTEXT` permanece `false`.

## 1. O que cada coletor faz (e com que frequência)

| Job | Comando | Frequência | Por quê |
|---|---|---|---|
| `positioning` | `research_collectors_run.py --job positioning` → `binance_positioning_b2_collector.py` | **15 min** | Binance retém ~30d; overlap de 5h/ciclo tolera atrasos de horas; 15m equilibra frescor, rate (~384 req/dia, ~0,1% do limite) e storage |
| `cftc-collect` | `--job cftc-collect` → R4 collector | **diária 06:10** | relatório é semanal, mas coleta diária fixa `first_seen_at` perto da publicação real (crítico p/ TRUE-OOS) e absorve atraso de feriado |
| `cftc-outcomes` | `--job cftc-outcomes` | **diária 06:25** | matura outcomes PENDING (exit ≥ entry+28d) |
| `status` | `--job status` + `research_collectors_check.py` | **diária 06:40** | agrega `b2_status` + `cftc_r4_status` e avalia falha silenciosa |

## 2. Instalação do agendamento

**OCI/Linux (primário, host do bot):**
```bash
sudo cp infrastructure/systemd/research-collectors/*.service infrastructure/systemd/research-collectors/*.timer /etc/systemd/system/
# ajustar WorkingDirectory=/opt/market-bot (checkout do repo) nos .service
sudo systemctl daemon-reload
sudo systemctl enable --now research-positioning.timer research-cftc-collect.timer \
  research-cftc-outcomes.timer research-status.timer
systemctl list-timers | grep research
```

**Windows local (requer PowerShell ADMIN):**
```powershell
$py='C:\...\python.exe'; $repo='C:\...\robo_sistema.binace.api'
$run='scripts/analytics/research_collectors_run.py'
$mk={param($n,$a,$t) New-ScheduledTaskAction -Execute $py -Argument "$run $a" -WorkingDirectory $repo}
Register-ScheduledTask ResearchPositioning15m -Action (&$mk '' '--job positioning --timeout 900' '') `
  -Trigger (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Minutes 15)) -User SYSTEM -Force
# diários 06:10/06:25/06:40: New-ScheduledTaskTrigger -Daily -At '06:10' (idem p/ demais jobs)
```
Status B3: units OCI entregues; Windows local **não instalado** (sem admin em 2026-09-15) — rodar o bloco acima como admin.

## 3. Saúde e alertas

- `python scripts/analytics/research_collectors_check.py` → exit 0/1/2 (ok/warning/critical) + `dados/research/.state/alert.json` (só alerta em transição) + webhook opcional via `RESEARCH_ALERT_WEBHOOK_URL`.
- Limiares: positioning WARNING >2h / CRITICAL >24h sem sucesso; CFTC WARNING >8d / CRITICAL >10d; falhas seguidas ≥3/≥10; `unrecoverable_gap` = CRITICAL imediato.
- Logs: `logs/research_collectors/{job}-YYYYMMDD.log`; estado: `dados/research/.state/{job}.json`.

## 4. Se parou — janela de perda irreversível: 30 dias (Binance ratios/OI)

1. Rode `research_collectors_check.py` e leia `last_success`.
2. Rode o job manualmente via runner (lock/timeout/exit codes cuidam do resto).
3. O coletor faz backfill da janela faltante sozinho (`plan_recovery`: overlap ≤5h, backfill ≤30d).
4. Gap >30d → `unrecoverable_gap` registrado; **nunca fabricar**; rode `binance_positioning_b2_status.py` para confirmar.
5. Downtime/host restart: nada a fazer além de religar o scheduler (estado em JSONL, restart-safe).

## 5. Integridade e retenção

- `dados/` é gitignored (correto — dataset nunca no git); só JSONs pequenos de `analysis/results/` são versionados.
- Backup OCI (`backup_to_oci.py`) **exclui** `.jsonl/.db/.env`: raw/normalized JSONL ficam **fora** do backup; `metadata/*.json`, `health/*.json` e futuros `.parquet` entram. Dado público, sem credenciais.
- Recomendação: compactação mensal JSONL→Parquet (entra no backup; ~1/3 do tamanho). Estimativa: ~15 KB/dia/símbolo JSONL (~5 MB/ano) + funding separado (~2 MB/ano); Parquet ~1/3.
- Validação: `b2_status` (complete/partial/revisions/gaps) + testes `test_research_collectors_ops.py` + `test_binance_positioning_b2.py`.
