import json
import subprocess
import sys
from pathlib import Path

from institutional.event_bridge import InstitutionalEventBridge


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(script: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


def test_build_compact_payload_import_does_not_eager_load_payload_sections() -> None:
    data = _run_isolated_python(
        """
import json
import sys
import build_compact_payload

print(json.dumps({
    "payload_sections_loaded": "market_orchestrator.ai.payload_sections" in sys.modules,
    "module_loaded": "build_compact_payload" in sys.modules,
}))
"""
    )

    assert data["module_loaded"] is True
    assert data["payload_sections_loaded"] is False


def test_ai_runner_import_does_not_eager_load_root_ai_modules() -> None:
    data = _run_isolated_python(
        """
import json
import sys
import market_orchestrator.ai.ai_runner

print(json.dumps({
    "ai_runner_loaded": "market_orchestrator.ai.ai_runner" in sys.modules,
    "ai_analyzer_qwen_loaded": "ai_analyzer_qwen" in sys.modules,
    "build_compact_payload_loaded": "build_compact_payload" in sys.modules,
}))
"""
    )

    assert data["ai_runner_loaded"] is True
    assert data["ai_analyzer_qwen_loaded"] is False
    assert data["build_compact_payload_loaded"] is False


def test_institutional_event_bridge_processes_minimal_event() -> None:
    bridge = InstitutionalEventBridge(monte_carlo_simulations=100)

    event = {
        "epoch_ms": 1_775_340_000_000,
        "symbol": "BTCUSDT",
        "ativo": "BTCUSDT",
        "preco_fechamento": 73500.0,
        "ohlc": {
            "open": 73480.0,
            "high": 73520.0,
            "low": 73460.0,
            "close": 73500.0,
        },
        "volume_total": 12.5,
        "volume_total_btc": 12.5,
        "buy_notional_usdt": 250000.0,
        "sell_notional_usdt": 120000.0,
        "volume_compra_btc": 3.4,
        "volume_venda_btc": 1.7,
        "derivatives": {
            "BTCUSDT": {
                "funding_rate_percent": 0.01,
                "open_interest": 1000000,
                "long_short_ratio": 1.1,
            }
        },
        "orderbook_data": {
            "is_valid": True,
            "bid": 73499.5,
            "ask": 73500.5,
            "mid": 73500.0,
            "bid_depth_usd": 800000.0,
            "ask_depth_usd": 780000.0,
        },
    }

    result = bridge.process_event(event)

    assert result["events_processed"] == 1
    assert result["price"] == 73500.0
    assert "confluence" in result
    assert "layers" in result
    assert "modules" in result
    assert isinstance(result["modules"], dict)
