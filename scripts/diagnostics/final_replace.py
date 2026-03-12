# Final script to replace the method properly
with open('ai_analyzer_qwen.py', 'rb') as f:
    content = f.read()

# Find the method boundaries
method_start = content.find(b'def _extract_orderbook_data')
if method_start == -1:
    print("Method not found!")
    exit(1)

# Find the next method (_get_system_prompt)
next_method = content.find(b'\n    def _get_system_prompt', method_start)
if next_method == -1:
    print("Next method not found!")
    exit(1)

print(f"Found method at position {method_start}")
print(f"Next method at position {next_method}")

# Extract old method to see exact bytes
old_method = content[method_start:next_method]
print(f"\nOld method length: {len(old_method)}")
print(f"First 100 bytes: {old_method[:100]}")

# The new method with correct 4-space indentation for class method
# Note: the method is inside a class, so it needs 4 spaces indent
new_method = b'''    def _extract_orderbook_data(self, event_data: dict) -> dict:
        \"\"\"
        Extrai dados de orderbook do evento.
        
        Busca em TODOS os formatos possiveis:
        - Formato original: event_data["orderbook_data"] com bid_depth_usd/ask_depth_usd
        - Formato compacto: event_data["ob"] com bid/ask (pos LEAK_BLOCKED)
        - Formato contextual: contextual_snapshot.orderbook_data
        - Formato order_book_depth: event_data["order_book_depth"]
        \"\"\"
        # ============================================================
        # FONTE 1: Formato compacto "ob" (pos build_compact_payload)
        # Prioridade MAXIMA porque apos LEAK_BLOCKED e o unico que sobra
        # ============================================================
        ob_compact = event_data.get("ob")
        if isinstance(ob_compact, dict):
            bid = ob_compact.get("bid", ob_compact.get("bid_depth_usd", 0))
            ask = ob_compact.get("ask", ob_compact.get("ask_depth_usd", 0))
            if bid or ask:  # Pelo menos um > 0
                logging.debug(
                    "Orderbook extracted from compact 'ob': bid=$%.0f, ask=$%.0f",
                    bid, ask,
                )
                return {
                    "bid_depth_usd": float(bid or 0),
                    "ask_depth_usd": float(ask or 0),
                    "imbalance": float(ob_compact.get("imb", ob_compact.get("imbalance", 0))),
                    "depth_imbalance": float(ob_compact.get("top5_imb", ob_compact.get("depth_imbalance", 0))),
                    "spread": float(ob_compact.get("spread", ob_compact.get("spread_bps", 0))),
                    "volume_ratio": float(ob_compact.get("vol_ratio", ob_compact.get("volume_ratio", 0))),
                    "pressure": float(ob_compact.get("pressure", 0)),
                    "_source": "compact_ob",
                }

        # ============================================================
        # FONTE 2: Formato original (pre LEAK_BLOCKED)
        # ============================================================
        candidates = [
            event_data.get("orderbook_data"),
            event_data.get("spread_metrics"),
            (event_data.get("contextual_snapshot") or {}).get("orderbook_data"),
            (event_data.get("contextual") or {}).get("orderbook_data"),
            (event_data.get("enriched_snapshot") or {}).get("orderbook_data"),
            (event_data.get("raw_event") or {}).get("orderbook_data"),
        ]

        for candidate in candidates:
            if isinstance(candidate, dict):
                bid = candidate.get("bid_depth_usd", candidate.get("bid", 0))
                ask = candidate.get("ask_depth_usd", candidate.get("ask", 0))
                if bid or ask:
                    logging.debug(
                        "Orderbook extracted from original format: bid=$%.0f, ask=$%.0f",
                        bid, ask,
                    )
                    return {
                        "bid_depth_usd": float(bid or 0),
                        "ask_depth_usd": float(ask or 0),
                        "imbalance": float(candidate.get("imbalance", 0)),
                        "depth_imbalance": float(
                            candidate.get("depth_metrics", {}).get("depth_imbalance",
                            candidate.get("depth_imbalance", 0))
                        ),
                        "spread": float(candidate.get("spread", 0)),
                        "volume_ratio": float(candidate.get("volume_ratio", 0)),
                        "pressure": float(candidate.get("pressure", 0)),
                        "_source": "original_orderbook_data",
                    }

        # ============================================================
        # FONTE 3: order_book_depth (formato com niveis L1/L5/L10)
        # ============================================================
        obd = event_data.get("order_book_depth")
        if isinstance(obd, dict):
            # Tenta L5 primeiro (mais representativo), depois L1, L10, L25
            for level_key in ("L5", "L1", "L10", "L25"):
                level = obd.get(level_key)
                if isinstance(level, dict):
                    bid = level.get("bids", 0)
                    ask = level.get("asks", 0)
                    if bid or ask:
                        logging.debug(
                            "Orderbook extracted from order_book_depth.%s: bid=$%.0f, ask=$%.0f",
                            level_key, bid, ask,
                        )
                        return {
                            "bid_depth_usd": float(bid),
                            "ask_depth_usd": float(ask),
                            "imbalance": float(level.get("imbalance", 0)),
                            "depth_imbalance": float(level.get("imbalance", 0)),
                            "spread": float(event_data.get("spread_analysis", {}).get("current_spread_bps", 0)),
                            "volume_ratio": float(obd.get("total_depth_ratio", 0)),
                            "pressure": 0.0,
                            "_source": f"order_book_depth_{level_key}",
                        }

        # ============================================================
        # FONTE 4: bid/ask diretos no top-level do evento
        # ============================================================
        bid_direct = event_data.get("bid_depth_usd", event_data.get("bid", 0))
        ask_direct = event_data.get("ask_depth_usd", event_data.get("ask", 0))
        if bid_direct or ask_direct:
            logging.debug(
                "Orderbook extracted from top-level bid/ask: bid=$%.0f, ask=$%.0f",
                float(bid_direct), float(ask_direct),
            )
            total = float(bid_direct or 0) + float(ask_direct or 0)
            imb = ((float(bid_direct or 0) - float(ask_direct or 0)) / total) if total > 0 else 0
            return {
                "bid_depth_usd": float(bid_direct or 0),
                "ask_depth_usd": float(ask_direct or 0),
                "imbalance": round(imb, 4),
                "depth_imbalance": round(imb, 4),
                "spread": 0.0,
                "volume_ratio": 0.0,
                "pressure": 0.0,
                "_source": "top_level_bid_ask",
            }

        # ============================================================
        # NENHUMA FONTE ENCONTRADA
        # ============================================================
        logging.warning(
            "No valid orderbook source found | available_keys=%s",
            [k for k in event_data.keys() if "ob" in k.lower() or "book" in k.lower() or "bid" in k.lower() or "depth" in k.lower()]
        )
        return {
            "bid_depth_usd": 0.0,
            "ask_depth_usd": 0.0,
            "imbalance": 0.0,
            "depth_imbalance": 0.0,
            "spread": 0.0,
            "volume_ratio": 0.0,
            "pressure": 0.0,
            "_source": "none_found",
        }

'''

# Replace
new_content = content[:method_start] + new_method + content[next_method:]
with open('ai_analyzer_qwen.py', 'wb') as f:
    f.write(new_content)

print("\nMethod replaced successfully!")
