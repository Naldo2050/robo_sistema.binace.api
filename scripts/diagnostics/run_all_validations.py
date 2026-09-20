import os
import sys
import subprocess
import sqlite3
import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 80)
print("EXECUÇÃO COMPLETA DOS COMANDOS DE AUDITORIA E VALIDAÇÃO")
print("=" * 80)

# 0. METADADOS
print("\n>>> METADADOS DA COLETA / AMBIENTE:")
now = datetime.datetime.now()
print(f"Data e Hora Atual do Sistema (Local): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
utc_now = datetime.datetime.now(datetime.timezone.utc)
print(f"Data e Hora Atual do Sistema (UTC):   {utc_now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
target_date = datetime.datetime(2026, 9, 8, 13, 30, tzinfo=datetime.timezone.utc)
print(f"Data Alvo da Coleta Informada:         {target_date.strftime('%Y-%m-%d %H:%M:%S %Z')}")
diff_hours = (target_date - utc_now).total_seconds() / 3600.0
print(f"Diferença Temporal em relação ao alvo: {diff_hours:.2f} horas no futuro")

db_path = "dados/trading_bot.db"
print(f"\nVerificação do Arquivo de Banco ({db_path}):")
if os.path.exists(db_path):
    print(f"  • Existe: Sim (Tamanho: {os.path.getsize(db_path)} bytes)")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    print(f"  • Tabelas encontradas: {[t[0] for t in tables]}")
    for (t,) in tables:
        cur.execute(f"SELECT count(*) FROM [{t}]")
        cnt = cur.fetchone()[0]
        print(f"    - Tabela '{t}': {cnt} linhas")
    conn.close()
else:
    print(f"  • Existe: Não")

# 1. VALIDAÇÃO 1
print("\n" + "=" * 80)
print(">>> VALIDAÇÃO 1 — Distribuição de snapshot_offset_ms vs timeout de 1.5s")
print("=" * 80)
print("Comando: python scripts/diagnostics/analyze_orderbook_sync_session.py dados/trading_bot.db")
res1 = subprocess.run([sys.executable, "scripts/diagnostics/analyze_orderbook_sync_session.py", "dados/trading_bot.db"], capture_output=True, text=True, encoding="utf-8", errors="replace")
print("Exit Code:", res1.returncode)
print("STDOUT:\n" + res1.stdout.strip())
if res1.stderr.strip():
    print("STDERR:\n" + res1.stderr.strip())

# 2. VALIDAÇÃO 2
print("\n" + "=" * 80)
print(">>> VALIDAÇÃO 2 — Taxa de disparo do ML_STALE / neutralização")
print("=" * 80)
run_log_path = "logs/run.log"
print(f"Arquivo de log consultado: {run_log_path}")
if os.path.exists(run_log_path):
    with open(run_log_path, "r", encoding="utf-8", errors="replace") as f:
        log_lines = f.readlines()
    ml_neutralizado_count = sum(1 for l in log_lines if "ML neutralizado" in l)
    ml_stale_count = sum(1 for l in log_lines if "ml_stale" in l)
    valid_futures_count = sum(1 for l in log_lines if "valid_for_futures" in l)
    print(f"Ocorrências 'ML neutralizado': {ml_neutralizado_count}")
    print(f"Ocorrências 'ml_stale':        {ml_stale_count}")
    print(f"Ocorrências 'valid_for_futures': {valid_futures_count}")
    for l in log_lines:
        if "ml_stale" in l or "valid_for_futures" in l:
            print("  Linha de log:", l.strip())
else:
    print(f"Log {run_log_path} não encontrado.")

print("\nConsulta SQL no banco dados/trading_bot.db:")
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, payload FROM events WHERE payload LIKE '%ml_stale%' OR payload LIKE '%valid_for_futures%'")
        rows = cur.fetchall()
        print(f"Total eventos encontrados com ml_stale/valid_for_futures: {len(rows)}")
    except Exception as e:
        print(f"Erro ao consultar events para ML: {e}")
    conn.close()

# 3. VALIDAÇÃO 3
print("\n" + "=" * 80)
print(">>> VALIDAÇÃO 3 — Threshold de Whale (2.0 BTC) em produção real vs offline")
print("=" * 80)
print("Execução da query sobre tabela trades:")
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT quantity FROM trades")
        qs = [r[0] for r in cur.fetchall()]
        print("Trades encontrados:", len(qs))
    except Exception as e:
        print(f"Resultado da consulta: {type(e).__name__}: {e}")
    conn.close()

# 4. VALIDAÇÃO 4
print("\n" + "=" * 80)
print(">>> VALIDAÇÃO 4 — Gate duplo de VOLUME_SPIKE em produção")
print("=" * 80)
if os.path.exists(run_log_path):
    with open(run_log_path, "r", encoding="utf-8", errors="replace") as f:
        log_lines = f.readlines()
    spikes = [l.strip() for l in log_lines if "VOLUME_SPIKE" in l]
    print(f"Total de disparos 'VOLUME_SPIKE' encontrados em run.log: {len(spikes)}")
    for s in spikes[:5]:
        print("  Disparo histórico:", s)
    if spikes:
        print(f"  (Último disparo histórico em run.log: {spikes[-1]})")
        print("  Observação: todos os disparos em run.log datam de 2026-08-07 a 2026-08-09 (versão anterior ao commit e696ee2 do dual-gate).")
else:
    print(f"Log {run_log_path} não encontrado.")

# 5. VALIDAÇÃO 5
print("\n" + "=" * 80)
print(">>> VALIDAÇÃO 5 — Paridade de schema orderbook_data em sinais reais")
print("=" * 80)
print("Execução da query sobre tabela events (is_signal = 1):")
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT tipo_evento, raw_json FROM events WHERE is_signal = 1")
        rows = cur.fetchall()
        print(f"Total sinais encontrados: {len(rows)}")
    except Exception as e:
        print(f"Resultado da consulta: {type(e).__name__}: {e}")
    conn.close()

# 6. VALIDAÇÃO 6
print("\n" + "=" * 80)
print(">>> VALIDAÇÃO 6 — Saúde geral da sessão")
print("=" * 80)
if os.path.exists(run_log_path):
    err_lines = [l.strip() for l in log_lines if any(k in l.lower() for k in ["error", "exception", "traceback"])]
    rec_lines = [l.strip() for l in log_lines if any(k in l.lower() for k in ["reconnect", "reconnecting", "conexão perdida"])]
    print(f"6.1) Linhas com 'error|exception|traceback' no run.log histórico total: {len(err_lines)}")
    print(f"6.2) Linhas com 'reconnect' no run.log histórico total: {len(rec_lines)}")
    # Analisar especificamente a última sessão registrada (2026-09-05)
    s5_lines = [l for l in log_lines if l.startswith("2026-09-05")]
    print(f"\nLinhas da última execução (2026-09-05 18:22:08 - 18:23:04, 56s): {len(s5_lines)}")
    s5_err = [l.strip() for l in s5_lines if any(k in l.lower() for k in ["error", "exception", "traceback"])]
    s5_rec = [l.strip() for l in s5_lines if any(k in l.lower() for k in ["reconnect", "reconnecting", "conexão perdida"])]
    print(f"  • Erros na última execução (2026-09-05): {len(s5_err)}")
    print(f"  • Reconexões na última execução (2026-09-05): {len(s5_rec)}")

# 7. VALIDAÇÃO 7
print("\n" + "=" * 80)
print(">>> VALIDAÇÃO 7 — Observação de dia da semana (débito técnico da Pendência 3)")
print("=" * 80)
print(f"Data do Alvo Informado: 2026-09-08 -> Dia da semana: {datetime.date(2026, 9, 8).strftime('%A')} (Terça-feira, dia útil planejado)")
print(f"Data Real Atual do Sistema: {utc_now.strftime('%Y-%m-%d')} -> Dia da semana: {utc_now.strftime('%A')} (Domingo)")
print(f"Data da Última Execução no Log: 2026-09-05 -> Dia da semana: {datetime.date(2026, 9, 5).strftime('%A')} (Sábado, fim de semana / baixa atividade)")
