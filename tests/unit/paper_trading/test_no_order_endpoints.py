# tests/unit/paper_trading/test_no_order_endpoints.py
"""
Static audit proof verifying zero live order execution endpoints or trade client imports in paper_trading/.
"""

import os
from pathlib import Path

FORBIDDEN_PATTERNS = [
    "/fapi/v1/order",
    "/fapi/v2/order",
    "/api/v3/order",
    "/fapi/v1/batchOrders",
    "newOrder",
    "futures_create_order",
    "change_leverage",
    "Client(",
    "AsyncClient(",
    "BinanceFuturesClient",
    "send_order",
    "place_order",
]


def test_zero_forbidden_order_endpoints_in_paper_trading():
    """Verify that paper_trading/ contains strictly zero live exchange order endpoints or imports."""
    paper_dir = Path(__file__).resolve().parents[3] / "paper_trading"
    assert paper_dir.exists() and paper_dir.is_dir()

    violations = []

    for root, _, files in os.walk(paper_dir):
        for f in files:
            if f.endswith(".py"):
                file_path = os.path.join(root, f)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                    for pattern in FORBIDDEN_PATTERNS:
                        if pattern in content:
                            violations.append(f"{f}: found forbidden pattern '{pattern}'")

    assert not violations, f"Forbidden order execution endpoints detected in paper_trading: {violations}"
