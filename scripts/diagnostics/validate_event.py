# -*- coding: utf-8 -*-
import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def validar_ultimo_analysis_trigger():
    path = 'dados/eventos_visuais.log'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocos = content.split('EVENTO: ANALYSIS_TRIGGER')
    if len(blocos) < 2:
        print("Nenhum ANALYSIS_TRIGGER encontrado no log")
        return

    ultimo = blocos[-1]
    inicio_json = ultimo.find('{')
    fim_bloco = ultimo.find('----------------------------------------------------------------------------------------------------', inicio_json)
    if inicio_json == -1 or fim_bloco == -1:
        print("Bloco inválido (não achei JSON/----)")
        return

    json_section = ultimo[inicio_json:fim_bloco].strip()

    # Encontrar fechamento correto do JSON por contagem de chaves
    count = 0
    fim_json = -1
    for i, ch in enumerate(json_section):
        if ch == '{':
            count += 1
        elif ch == '}':
            count -= 1
            if count == 0:
                fim_json = i
                break
    if fim_json == -1:
        print("Não consegui fechar o JSON corretamente")
        return

    json_str = json_section[:fim_json+1]

    # Remover partes inválidas do JSON (como "..." em arrays)
    json_str = json_str.replace(', ...,', ',')
    json_str = json_str.replace('[...,', '[')
    json_str = json_str.replace(', ...]', ']')

    event = json.loads(json_str)

    raw_event = event.get("raw_event", {})
    inner_raw = raw_event.get("raw_event", {})
    aa = inner_raw.get("advanced_analysis", {})

    print("\n===== DEBUG CONTEXTO EXTERNO (raw_event) =====")
    print(f"raw_event keys: {list(raw_event.keys())}")

    multi_tf = raw_event.get("multi_tf")
    historical_vp = raw_event.get("historical_vp")
    ts_utc = raw_event.get("timestamp_utc")

    print(f"multi_tf presente: {bool(multi_tf)}")
    if multi_tf:
        print(f"  multi_tf timeframes: {list(multi_tf.keys())}")
        tf_1d = multi_tf.get("1d", {})
        print(f"  multi_tf['1d']['realized_vol']: {tf_1d.get('realized_vol', 'AUSENTE')}")

    print(f"historical_vp presente: {bool(historical_vp)}")
    if historical_vp:
        daily = historical_vp.get("daily", {})
        print(f"  daily.poc: {daily.get('poc', 'AUSENTE')}")
        print(f"  daily.vah: {daily.get('vah', 'AUSENTE')}")
        print(f"  daily.val: {daily.get('val', 'AUSENTE')}")

    print(f"timestamp_utc em raw_event: {ts_utc}")
    print("=============================================\n")

    print("="*60)
    print("VALIDAÇÃO DO advanced_analysis (último ANALYSIS_TRIGGER)")
    print("="*60)

    timestamp = aa.get("timestamp")
    current_vol = aa.get("adaptive_thresholds", {}).get("current_volatility")
    price_targets = aa.get("price_targets", [])

    print(f"timestamp: {timestamp if timestamp else 'AUSENTE'}")
    print(f"current_volatility: {current_vol if current_vol is not None else 'AUSENTE'}")
    print(f"price_targets count: {len(price_targets)}")

    ok = True

    if not timestamp:
        print("❌ FALHA: timestamp ausente")
        ok = False
    if current_vol is None or current_vol == 0.01:
        print(f"❌ FALHA: current_volatility = {current_vol} (fallback)")
        ok = False
    if len(price_targets) < 3:
        print(f"❌ FALHA: price_targets tem {len(price_targets)} itens (< 3)")
        ok = False

    if ok:
        print("✅ VALIDAÇÃO PASSOU: advanced_analysis está completo e adaptativo")
    else:
        print("❌ VALIDAÇÃO FALHOU: advanced_analysis ainda está incompleto")

    print("="*60)

if __name__ == "__main__":
    validar_ultimo_analysis_trigger()