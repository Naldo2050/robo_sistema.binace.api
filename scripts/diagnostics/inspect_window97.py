import sqlite3
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

conn = sqlite3.connect('dados/trading_bot.db')
cur = conn.cursor()

# 1. Encontrar o evento correspondente à janela 97 (por volta de 15:12:00 UTC / 12:12:00 local)
cur.execute("SELECT id, timestamp_ms, event_type, window_id, payload FROM events")
rows = cur.fetchall()

w97_event = None
all_ob_stats = []

for eid, ts, etype, wid, raw in rows:
    p = json.loads(raw) if isinstance(raw, str) else (raw or {})
    ob = p.get("orderbook_data") or {}
    src = ob.get("source") or ob.get("source_type")
    
    j_num = p.get("janela_numero") or wid
    if str(j_num) in ("97", "W0097", "97.0") or (ts and 1788880320000 <= ts <= 1788880380000):
        w97_event = (eid, ts, etype, wid, p)

    # Analisar depths dos eventos live_sync
    ob_depth = p.get("order_book_depth") or {}
    bid_usd = ob_depth.get("bid_depth_usd") or ob.get("bid_depth_usd")
    ask_usd = ob_depth.get("ask_depth_usd") or ob.get("ask_depth_usd")
    bids = p.get("bids") or ob.get("bids") or []
    asks = p.get("asks") or ob.get("asks") or []

    if src:
        all_ob_stats.append({
            "id": eid, "src": src,
            "bid_usd": bid_usd, "ask_usd": ask_usd,
            "bids_count": len(bids), "asks_count": len(asks)
        })

print(f"Total de eventos com orderbook: {len(all_ob_stats)}")
live_events = [e for e in all_ob_stats if e["src"] == "live_sync"]
cache_events = [e for e in all_ob_stats if e["src"] == "cache_bg"]
print(f"Live sync: {len(live_events)} | Cache bg: {len(cache_events)}")

if live_events:
    bids_usd_live = [e["bid_usd"] for e in live_events if e["bid_usd"] is not None]
    asks_usd_live = [e["ask_usd"] for e in live_events if e["ask_usd"] is not None]
    if bids_usd_live:
        print(f"Bid Depth USD (live_sync): min=${min(bids_usd_live):,.0f} | med=${sum(bids_usd_live)/len(bids_usd_live):,.0f} | max=${max(bids_usd_live):,.0f}")
    if asks_usd_live:
        print(f"Ask Depth USD (live_sync): min=${min(asks_usd_live):,.0f} | med=${sum(asks_usd_live)/len(asks_usd_live):,.0f} | max=${max(asks_usd_live):,.0f}")

print("\n--- Detalhes do Evento da Janela 97 (12:12:01 local / 15:12:01 UTC) ---")
if w97_event:
    eid, ts, etype, wid, p = w97_event
    ob = p.get("orderbook_data") or {}
    print(f"ID: {eid} | Janela: {wid} | TS: {ts}")
    print(f"Source: {ob.get('source')} | Source Type: {ob.get('source_type')}")
    print(f"Snapshot Offset: {ob.get('snapshot_offset_ms')} ms")
    print(f"OrderBook Depth no payload: {p.get('order_book_depth')}")
    print(f"OrderBook Data chaves: {list(ob.keys())}")
else:
    print("Janela 97 não localizada diretamente por chave.")
