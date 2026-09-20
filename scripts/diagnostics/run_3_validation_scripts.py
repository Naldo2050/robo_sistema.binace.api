import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cmds = [
    [sys.executable, 'scripts/diagnostics/analyze_orderbook_sync_session.py', '--db', 'dados/trading_bot.db', '--session', '2026-09-08'],
    [sys.executable, 'scripts/diagnostics/validate_whale_threshold.py', '--dump-path', 'dados/trades_collect_2h.jsonl', '--db', 'dados/trading_bot.db'],
    [sys.executable, 'scripts/diagnostics/validate_signals_orderbook_schema.py', '--db', 'dados/trading_bot.db']
]

for cmd in cmds:
    print('=' * 80)
    print('EXECUTANDO:', ' '.join(cmd[1:]))
    print('=' * 80)
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print('Exit code:', res.returncode)
    print(res.stdout.strip())
    if res.stderr.strip():
        print('STDERR:', res.stderr.strip())
    print()
