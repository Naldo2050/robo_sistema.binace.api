# market_orchestrator/ai/ai_enrichment_context.py
"""
Gera contextos para o ai_payload a partir de raw_event.advanced_analysis.
"""

from __future__ import annotations

from typing import Dict, Any, List
import logging

from common.json_safe import is_non_finite_number

logger = logging.getLogger(__name__)


def _present_number(value: Any) -> bool:
    """True se value é número real presente (None/NaN/±Inf = ausente).

    0.0 legítimo continua presente — ausência nunca vira zero conclusivo.
    """
    if value is None or is_non_finite_number(value):
        return False
    return isinstance(value, (int, float))


def build_enriched_ai_context(raw_event: Dict[str, Any]) -> Dict[str, Any]:
    advanced = raw_event.get("advanced_analysis") or {}
    if not advanced:
        return {}

    ctx: Dict[str, Any] = {}

    # 1) Targets Context
    price_targets: List[Dict[str, Any]] = advanced.get("price_targets") or []
    if price_targets:
        def _score_key(t: dict) -> float:
            conf = t.get("confidence")
            weight = t.get("weight")
            conf = conf if _present_number(conf) else 0.0
            weight = weight if _present_number(weight) else 0.0
            return conf * weight

        sorted_targets = sorted(
            price_targets,
            key=_score_key,
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

    # 2) Options Context (ausência -> "unknown", nunca conclusão fabricada)
    opt = advanced.get("options_metrics") or {}
    if opt:
        pcr = opt.get("put_call_ratio")
        if not _present_number(pcr):
            sentiment = "unknown"
        else:
            sentiment = "bearish" if pcr > 1.0 else "bullish"
        ctx["options_context"] = {
            "put_call_ratio": opt.get("put_call_ratio"),
            "iv_rank": opt.get("iv_rank"),
            "iv_percentile": opt.get("iv_percentile"),
            "gamma_exposure": opt.get("gamma_exposure"),
            "max_pain": opt.get("max_pain"),
            "skew": opt.get("skew"),
            "sentiment": sentiment,
        }

    # 3) On-chain Context (ausência -> "unknown"; 0.0 presente -> "neutral")
    onch = advanced.get("onchain_metrics") or {}
    if onch:
        netflow = onch.get("exchange_netflow")
        if not _present_number(netflow):
            sentiment = "unknown"
        elif netflow < 0.0:
            sentiment = "accumulation"
        elif netflow > 0.0:
            sentiment = "distribution"
        else:
            sentiment = "neutral"
        ctx["onchain_context"] = {
            "exchange_netflow": onch.get("exchange_netflow"),
            "whale_transactions": onch.get("whale_transactions"),
            "sopr": onch.get("sopr"),
            "hash_rate": onch.get("hash_rate"),
            "funding_rates": onch.get("funding_rates"),
            "sentiment": sentiment,
        }

    # 4) Risk / Adaptive thresholds (ausência -> "unknown")
    at = advanced.get("adaptive_thresholds") or {}
    if at:
        vol = at.get("current_volatility")
        if not _present_number(vol):
            regime = "unknown"
        else:
            regime = "high_vol" if vol > 0.01 else "low_vol"
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

    def _num(t: dict, key: str) -> float:
        v = t.get(key)
        return v if _present_number(v) else 0.0

    avg_conf = sum(_num(t, "confidence") for t in price_targets) / len(price_targets)
    avg_weight = sum(_num(t, "weight") for t in price_targets) / len(price_targets)

    score = (unique_sources * 15.0) + (avg_conf * 40.0) + (avg_weight * 30.0)
    return max(0.0, min(100.0, score))