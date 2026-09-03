# -*- coding: utf-8 -*-
"""
Auditoria Forense e Qualidade de Dados do Shadow Run Sem IA
"""
import sys
import os
import json
import sqlite3
import re
import statistics

def run_audit():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 80)
    print("AUDITORIA COMPLETA DO SHADOW RUN SEM IA")
    print("=" * 80)
    
    # Read log
    log_path = 'logs/shadow_no_ai_20260901.log'
    if not os.path.exists(log_path):
        print(f"ERRO: {log_path} nao existe")
        return
        
    with open(log_path, 'r', encoding='utf-16', errors='replace') as f:
        log_lines = [l.rstrip() for l in f]

    print(f"Total de linhas lidas do log: {len(log_lines)}")

    # 1. LATENCIA
    print("\n" + "=" * 50)
    print("1. AUDITORIA DE LATENCIA (window_latency_breakdown)")
    print("=" * 50)
    lat_lines = [l for l in log_lines if 'event=window_latency_breakdown' in l]
    print(f"Total eventos de latencia: {len(lat_lines)}")
    
    pipe_list = []
    dec_list = []
    deliv_list = []
    
    pat_lat = re.compile(r'window_id=(\S+).*?window_delivery_delay_ms=(\d+).*?pipeline_processing_ms=(\d+).*?decision_delay_ms=(\d+).*?flow_window_status=(\S+).*?orderbook_source=(\S+)')
    
    print(f"{'window_id':25s} | {'pipeline_ms':>12s} | {'decision_ms':>12s} | {'delivery_ms':>12s} | {'flow_status':>15s} | {'ob_source':>10s}")
    print("-" * 100)
    
    for l in lat_lines:
        m = pat_lat.search(l)
        if m:
            wid, deliv, pipe, dec, flow, ob = m.groups()
            pipe_list.append(int(pipe))
            dec_list.append(int(dec))
            deliv_list.append(int(deliv))
            print(f"{wid:25s} | {pipe:>12s} | {dec:>12s} | {deliv:>12s} | {flow:>15s} | {ob:>10s}")

    if pipe_list:
        p50_p = statistics.median(pipe_list)
        p90_p = sorted(pipe_list)[int(len(pipe_list) * 0.9)]
        max_p = max(pipe_list)
        
        p50_d = statistics.median(dec_list)
        p90_d = sorted(dec_list)[int(len(dec_list) * 0.9)]
        max_d = max(dec_list)
        
        print(f"\nEstatisticas pipeline_processing_ms: p50={p50_p:.1f}ms, p90={p90_p:.1f}ms, max={max_p:.1f}ms")
        print(f"Estatisticas decision_delay_ms:      p50={p50_d:.1f}ms, p90={p90_d:.1f}ms, max={max_d:.1f}ms")

    # 2. PERSISTENCIA
    print("\n" + "=" * 50)
    print("2. AUDITORIA DE PERSISTENCIA")
    print("=" * 50)
    conn = sqlite3.connect('dados/trading_bot.db')
    cur = conn.cursor()
    for t in ['events', 'signals', 'signal_outcomes']:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            cnt = cur.fetchone()[0]
            print(f"Tabela {t:15s}: {cnt} registros")
        except Exception as e:
            print(f"Tabela {t:15s}: ERRO - {e}")
            
    cur.execute("SELECT rowid, id, timestamp_ms, event_type, symbol, window_id, is_signal, created_at FROM events ORDER BY rowid DESC LIMIT 3")
    print("\nUltimos 3 eventos salvos no SQLite:")
    for r in cur.fetchall():
        print(r)
    conn.close()

    # 3. ABSORCAO E EXAUSTAO
    print("\n" + "=" * 50)
    print("3. ABSORCAO E EXAUSTAO")
    print("=" * 50)
    abs_lines = [l for l in log_lines if any(k in l.lower() for k in ['absorc', 'absorption', 'exaust', 'exhaustion'])]
    for l in abs_lines:
        print("RAW LOG:", l)

    # 4. FLUXO DIRECIONAL
    print("\n" + "=" * 50)
    print("4. FLUXO DIRECIONAL E INVARIANTES")
    print("=" * 50)
    wp_lines = [l for l in log_lines if 'window_processed' in l]
    flow_invariants_pass = True
    for l in wp_lines:
        idx = l.find('{')
        if idx != -1:
            try:
                d = json.loads(l[idx:])
                w = d.get('window_count', 0)
                tot = d.get('total_volume', 0)
                b = d.get('total_buy_volume', 0)
                s = d.get('total_sell_volume', 0)
                diff = abs((b + s) - tot)
                pass_str = "PASS" if diff < 0.0001 else "FAIL"
                if diff >= 0.0001:
                    flow_invariants_pass = False
                print(f"Janela #{w:2d} | Tot={tot:8.4f} | Buy={b:8.4f} | Sell={s:8.4f} | B+S={b+s:8.4f} | Invariante: {pass_str}")
            except Exception:
                pass
    print(f"Resultado Invariante Buy+Sell=Total: {'PASS (100% Coerente)' if flow_invariants_pass else 'FAIL'}")

    # 5. S/R E VOLUME PROFILE
    print("\n" + "=" * 50)
    print("5. SUPORTE, RESISTENCIA E VOLUME PROFILE")
    print("=" * 50)
    sr_lines = [l for l in log_lines if 'VP Di' in l or 'POC @' in l or 'sr.r1' in l]
    for l in sr_lines[:15]:
        print("RAW LOG:", l)
        
    # Check VAL < POC < VAH
    vp_pat = re.compile(r'POC @ ([\d,\.]+) \| VAL: ([\d,\.]+) \| VAH: ([\d,\.]+)')
    vp_pass = True
    for l in sr_lines:
        m = vp_pat.search(l)
        if m:
            poc = float(m.group(1).replace(',', ''))
            val = float(m.group(2).replace(',', ''))
            vah = float(m.group(3).replace(',', ''))
            check = (val < poc < vah)
            if not check:
                vp_pass = False
            print(f"VP Check: VAL={val:.2f} < POC={poc:.2f} < VAH={vah:.2f} => {'PASS' if check else 'FAIL'}")
    print(f"Resultado Invariante VAL < POC < VAH: {'PASS (100% Invariante Respeitado)' if vp_pass else 'FAIL'}")

    # 6. ORDERBOOK
    print("\n" + "=" * 50)
    print("6. ORDERBOOK")
    print("=" * 50)
    ob_lines = [l for l in log_lines if 'orderbook_event' in l or 'orderbook_source' in l or 'spread_bps' in l]
    print(f"Total eventos de orderbook: {len(ob_lines)}")
    for l in ob_lines[-15:]:
        print("RAW LOG:", l)

    # 7. WHALE / SMART MONEY
    print("\n" + "=" * 50)
    print("7. WHALE E SMART MONEY")
    print("=" * 50)
    whale_lines = [l for l in log_lines if any(k in l.lower() for k in ['whale', 'iceberg', 'smart_money', 'bos', 'fvg'])]
    print(f"Total eventos de whale/smart money: {len(whale_lines)}")
    for l in whale_lines:
        print("RAW LOG:", l)

    # 8. REGIME DE MERCADO
    print("\n" + "=" * 50)
    print("8. REGIME DE MERCADO")
    print("=" * 50)
    regime_lines = [l for l in log_lines if 'Macro Context' in l or 'regime_analysis' in l or 'current_regime' in l]
    print(f"Total linhas de regime: {len(regime_lines)}")
    for l in regime_lines:
        print("RAW LOG:", l)

    # 9. DUPLICACOES
    print("\n" + "=" * 50)
    print("9. DUPLICACOES")
    print("=" * 50)
    sig_lines = [l for l in log_lines if 'event=signal_enrichment_timings' in l]
    wids = [re.search(r'window_id=(\S+)', l).group(1) for l in sig_lines if re.search(r'window_id=(\S+)', l)]
    from collections import Counter
    counts = Counter(wids)
    print(f"Total signal_enrichment_timings: {len(sig_lines)}")
    dup_windows = [k for k, v in counts.items() if v > 1]
    print(f"Janelas com >1 enriquecimento: {dup_windows}")
    for k, v in counts.items():
        print(f"  {k}: {v} enriquecimento(s)")

    trig_lines = [l for l in log_lines if 'ANALYSIS_TRIGGER' in l and 'Janela' not in l]
    trig_wids = [re.search(r'window_id=(\S+)', l).group(1) for l in trig_lines if re.search(r'window_id=(\S+)', l)]
    trig_counts = Counter(trig_wids)
    print(f"ANALYSIS_TRIGGER disparados mais de uma vez por window_id? {[k for k, v in trig_counts.items() if v > 1]}")

    # 10. ERROS
    print("\n" + "=" * 50)
    print("10. AUDITORIA DE ERROS E ANOMALIAS")
    print("=" * 50)
    err_pat = re.compile(r'\b(error|exception|traceback|warning|invalid|corrupted|nan|inf)\b', re.IGNORECASE)
    err_lines = [
        l for l in log_lines 
        if err_pat.search(l) 
        and not any(ign in l for ign in ['DeprecationWarning', '_validation_error.*null', 'INFO', 'level": "INFO', 'level": "WARNING', 'level":"INFO', 'level":"WARNING'])
        and not any(ign in l.lower() for ign in ['infinite', 'infinity'])
    ]
    print(f"Total de linhas com erro/aviso no log: {len(err_lines)}")
    for l in err_lines[:30]:
        print("LOG ERRO/AVISO:", l)

if __name__ == '__main__':
    run_audit()
