# event_factory.py
from __future__ import annotations

from typing import Dict, Any, Optional


def _base_orderbook_event(
    *,
    symbol: str,
    schema_version: str,
    engine_version: Optional[str],
    ts_ms: int,
    timestamp_ny: Optional[str],
    timestamp_utc: Optional[str],
    top_n: int,
    ob_limit: int,
    endpoint: str = "fapi/v1/depth",
    exchange: str = "binance_futures",
) -> Dict[str, Any]:
    evt: Dict[str, Any] = {
        "schema_version": schema_version,
        "engine_version": engine_version,
        "tipo_evento": "OrderBook",
        "ativo": symbol,

        "top_n": int(top_n),
        "ob_limit": int(ob_limit),

        "timestamps": {
            "exchange_ms": int(ts_ms),
            "timestamp_ny": timestamp_ny,
            "timestamp_utc": timestamp_utc,
        },

        "source": {
            "exchange": exchange,
            "endpoint": endpoint,
            "symbol": symbol,
        },

        # defaults compatíveis com o “shape” que você já usa
        "alertas_liquidez": [],
        "walls": {"bids": [], "asks": []},
        "order_book_depth": {},
        "spread_analysis": {},
        "depth_metrics": {},
        "advanced_metrics": {},
    }
    return evt


def build_invalid_orderbook_event(
    *,
    symbol: str,
    schema_version: str,
    engine_version: Optional[str],
    ts_ms: int,
    error_msg: str,
    severity: str,
    timestamp_ny: Optional[str] = None,
    timestamp_utc: Optional[str] = None,
    top_n: int = 20,
    ob_limit: int = 100,
    thresholds: Optional[Dict[str, Any]] = None,
    health_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or {}

    evt = _base_orderbook_event(
        symbol=symbol,
        schema_version=schema_version,
        engine_version=engine_version,
        ts_ms=ts_ms,
        timestamp_ny=timestamp_ny,
        timestamp_utc=timestamp_utc,
        top_n=top_n,
        ob_limit=ob_limit,
    )

    evt.update({
        "is_valid": False,
        "should_skip": True,
        "emergency_mode": False,
        "erro": error_msg,

        "descricao": f"❌ Order book indisponível: {error_msg}",
        "resultado_da_batalha": "INDISPONÍVEL",

        "imbalance": 0.0,
        "volume_ratio": 1.0,
        "pressure": 0.0,

        "spread_metrics": {
            "mid": 0.0,
            "spread": 0.0,
            "spread_percent": 0.0,
            "bid_depth_usd": 0.0,
            "ask_depth_usd": 0.0,
        },

        "iceberg_reloaded": False,
        "iceberg_score": 0.0,

        "market_impact_buy": {
            "100k": {"usd": 100000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
            "1M": {"usd": 1000000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
        },
        "market_impact_sell": {
            "100k": {"usd": 100000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
            "1M": {"usd": 1000000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
        },

        "labels": {
            "dominant_label": "INDISPONÍVEL",
            "note": "Order book não pôde ser obtido ou validado.",
        },

        "severity": severity,

        "critical_flags": {
            "is_critical": False,
            "abs_imbalance": 0.0,
            "ratio_dom": 1.0,
            "dominant_usd": 0.0,
            "thresholds": thresholds,
        },

        "orderbook_data": {
            "mid": 0.0,
            "spread": 0.0,
            "spread_percent": 0.0,
            "bid_depth_usd": 0.0,
            "ask_depth_usd": 0.0,
            "imbalance": 0.0,
            "volume_ratio": 1.0,
            "pressure": 0.0,
        },

        "data_quality": {
            "is_valid": False,
            "data_source": "error",
            "age_seconds": 0.0,
            "validation_passed": False,
            "validation_issues": [error_msg],
            "warnings": [],
            "emergency_mode": False,
        },

        "health_stats": health_stats or {},
    })

    # carrega error também no source
    evt["source"]["error"] = error_msg
    evt["alertas_liquidez"] = [f"🚫 ERRO: {error_msg}"]

    return evt


def build_emergency_orderbook_event(
    *,
    symbol: str,
    schema_version: str,
    engine_version: Optional[str],
    ts_ms: int,
    failures: int,
    bid_depth_usd: float = 1000.0,
    ask_depth_usd: float = 1000.0,
    timestamp_ny: Optional[str] = None,
    timestamp_utc: Optional[str] = None,
    top_n: int = 20,
    ob_limit: int = 100,
    thresholds: Optional[Dict[str, Any]] = None,
    health_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or {}
    error_msg = f"Emergency mode after {failures} failures"

    evt = _base_orderbook_event(
        symbol=symbol,
        schema_version=schema_version,
        engine_version=engine_version,
        ts_ms=ts_ms,
        timestamp_ny=timestamp_ny,
        timestamp_utc=timestamp_utc,
        top_n=top_n,
        ob_limit=ob_limit,
    )

    evt.update({
        "is_valid": True,
        "should_skip": False,
        "emergency_mode": True,
        "erro": error_msg,

        "descricao": "Orderbook indisponível (emergency mode)",
        "resultado_da_batalha": "EMERGÊNCIA",

        "imbalance": 0.0,
        "volume_ratio": 1.0,
        "pressure": 0.0,

        "spread_metrics": {
            "mid": 0.0,
            "spread": 0.0,
            "spread_percent": 0.0,
            "bid_depth_usd": float(bid_depth_usd),
            "ask_depth_usd": float(ask_depth_usd),
        },

        "iceberg_reloaded": False,
        "iceberg_score": 0.0,

        "market_impact_buy": {
            "100k": {"usd": 100000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
            "1M": {"usd": 1000000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
        },
        "market_impact_sell": {
            "100k": {"usd": 100000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
            "1M": {"usd": 1000000, "move_usd": 0.0, "bps": 0.0, "levels": 0, "vwap": None},
        },

        "labels": {
            "dominant_label": "EMERGENCY",
            "note": "Orderbook indisponível (modo emergência)",
        },

        "severity": "WARNING",

        "critical_flags": {
            "is_critical": False,
            "abs_imbalance": 0.0,
            "ratio_dom": 1.0,
            "dominant_usd": 0.0,
            "thresholds": thresholds,
        },

        "orderbook_data": {
            "mid": 0.0,
            "spread": 0.0,
            "spread_percent": 0.0,
            "bid_depth_usd": float(bid_depth_usd),
            "ask_depth_usd": float(ask_depth_usd),
            "imbalance": 0.0,
            "volume_ratio": 1.0,
            "pressure": 0.0,
        },

        "data_quality": {
            "is_valid": True,
            "data_source": "emergency",
            "age_seconds": 0.0,
            "validation_passed": False,
            "validation_issues": [error_msg],
            "warnings": ["EMERGENCY_MODE"],
            "emergency_mode": True,
        },

        "health_stats": health_stats or {},
    })

    evt["source"]["error"] = error_msg
    evt["alertas_liquidez"] = ["EMERGENCY_MODE"]

    return evt