import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

trades_at_moment = []
with open('dados/trades_collect_2h.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        t = json.loads(line)
        ts = t['timestamp']
        if 1788880319000 <= ts <= 1788880323000:
            trades_at_moment.append(t)

print(f"Total de trades entre 15:12:00 e 15:12:02 UTC: {len(trades_at_moment)}")
buy_vols = sum(t["quantity"] for t in trades_at_moment if not t.get("is_buyer_maker"))
sell_vols = sum(t["quantity"] for t in trades_at_moment if t.get("is_buyer_maker"))
print(f"Volume de compra agressiva: {buy_vols:.4f} BTC")
print(f"Volume de venda agressiva:  {sell_vols:.4f} BTC")

print("\nPrimeiros 15 trades nesse intervalo:")
for t in trades_at_moment[:15]:
    side = 'SELL' if t.get('is_buyer_maker') else 'BUY'
    print(f"  TS={t['timestamp']} | {side:4s} | P={t['price']} | Q={t['quantity']}")
