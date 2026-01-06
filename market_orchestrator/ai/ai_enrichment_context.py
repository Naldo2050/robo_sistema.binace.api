# market_orchestrator/ai/ai_enrichment_context.py
"""
Gera contextos para o ai_payload a partir de raw_event.advanced_analysis.
"""

from __future__ import annotations

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def build_enriched_ai_context(raw_event: Dict[str, Any]) -> Dict[str, Any]:
    advanced = raw_event.get("advanced_analysis") or {}
    if not advanced:
        return {}

    ctx: Dict[str, Any] = {}

    # 1) Targets Context
    price_targets: List[Dict[str, Any]] = advanced.get("price_targets") or []
    if price_targets:
        sorted_targets = sorted(
            price_targets,
            key=lambda t: (t.get("confidence", 0.0) * t.get("weight", 0.0)),
            reverse=True,
        )
        primary = sorted_targets[0] if sorted_targets else None
        secondary = sorted_targets[1:4] if len(sorted_targets) > 1 else []

        ctx["targets_context"] = {
            "primary_target": primary,
            "secondary_targets": secondary,
            "confluence_score": _calculate_confluence_score(price_targets),
            "total_targets": len(price_targets),
        }

    # 2) Options Context
    opt = advanced.get("options_metrics") or {}
    if opt:
        sentiment = "bearish" if opt.get("put_call_ratio", 1.0) > 1.0 else "bullish"
        ctx["options_context"] = {
            "put_call_ratio": opt.get("put_call_ratio"),
            "iv_rank": opt.get("iv_rank"),
            "iv_percentile": opt.get("iv_percentile"),
            "gamma_exposure": opt.get("gamma_exposure"),
            "max_pain": opt.get("max_pain"),
            "skew": opt.get("skew"),
            "sentiment": sentiment,
        }

    # 3) On-chain Context
    onch = advanced.get("onchain_metrics") or {}
    if onch:
        sentiment = (
            "accumulation" if (onch.get("exchange_netflow", 0.0) < 0.0) else "distribution"
        )
        ctx["onchain_context"] = {
            "exchange_netflow": onch.get("exchange_netflow"),
            "whale_transactions": onch.get("whale_transactions"),
            "sopr": onch.get("sopr"),
            "hash_rate": onch.get("hash_rate"),
            "funding_rates": onch.get("funding_rates"),
            "sentiment": sentiment,
        }

    # 4) Risk / Adaptive thresholds
    at = advanced.get("adaptive_thresholds") or {}
    if at:
        regime = "high_vol" if at.get("current_volatility", 0.0) > 0.01 else "low_vol"
        ctx["risk_context"] = {
            "current_volatility": at.get("current_volatility"),
            "volatility_factor": at.get("volatility_factor"),
            "absorption_threshold": at.get("absorption_threshold"),
            "flow_threshold": at.get("flow_threshold"),
            "market_regime": regime,
        }

    return ctx


def _calculate_confluence_score(price_targets: List[Dict[str, Any]]) -> float:
    if not price_targets:
        return 0.0

    sources = {t.get("source", "") for t in price_targets}
    unique_sources = len(sources)
    avg_conf = sum(t.get("confidence", 0.0) for t in price_targets) / len(price_targets)
    avg_weight = sum(t.get("weight", 0.0) for t in price_targets) / len(price_targets)

    score = (unique_sources * 15.0) + (avg_conf * 40.0) + (avg_weight * 30.0)
    return max(0.0, min(100.0, score))