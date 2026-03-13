# test_enrich_event.py
import json
import config
from enrichment_integrator import enrich_analysis_trigger_event

# Monte um ANALYSIS_TRIGGER mínimo baseado em um dos seus eventos reais
event = {
    "tipo_evento": "ANALYSIS_TRIGGER",
    "symbol": "BTCUSDT",
    "raw_event": {
        "symbol": "BTCUSDT",
        "preco_fechamento": 90174.1,
        "volume_total": 1.111,
        "pattern_recognition": {
            "fibonacci_levels": {
                "high": 90174.1,
                "low": 90156.2,
                "23.6": 90160.4244,
                "38.2": 90163.0378,
                "50.0": 90165.15,
                "61.8": 90167.2622,
                "78.6": 90170.2694
            }
        },
        "historical_vp": {
            "daily": {
                "poc": 90100.0,
                "vah": 90200.0,
                "val": 90000.0,
                "volume_nodes": {
                    "hvn_nodes": "89999|28.68|8.70; 90054|12.84|3.90; 90067|19.58|5.94"
                }
            }
        },
        "timestamp_utc": "2026-01-03T19:46:00.000Z",
        "liquidity_heatmap": {
            "clusters": [
                {
                    "center": 90174.1235,
                    "total_volume": 1.111
                }
            ]
        },
        "multi_tf": {
            "1d": {
                "realized_vol": 0.0202
            }
        }
    }
}

config_dict = {k: v for k, v in vars(config).items() if not k.startswith("__")}

enriched_event = enrich_analysis_trigger_event(event, config_dict)

aa = enriched_event["raw_event"].get("advanced_analysis")
print("advanced_analysis:")
print(json.dumps(aa, indent=2, ensure_ascii=False))

print("\nPrice targets:")
for t in aa.get("price_targets", []):
    print(f" - {t['source']}: {t['price']} (type={t['type']})")