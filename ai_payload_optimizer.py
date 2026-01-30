from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Tuple


class AIPayloadOptimizer:
    """Builds a compact, LLM-ready payload from a raw event."""

    def __init__(self, max_orderbook_levels: int = 50) -> None:
        self.max_orderbook_levels = max_orderbook_levels

    @classmethod
    def optimize(cls, event: Dict[str, Any], max_orderbook_levels: int = 50) -> Dict[str, Any]:
        optimizer = cls(max_orderbook_levels=max_orderbook_levels)
        return optimizer.optimize_event(event)

    def optimize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(event, dict):
            return {}

        # Se parecer um payload estruturado (v2/v1 do builder), apenas limpa campos vazios.
        if self._looks_like_structured_payload(event):
            # IMPORTANT: aqui a otimização deve ser "no-op" para não quebrar compatibilidade
            # (ex.: remover seções inteiras quando campos são None).
            return dict(event)

        raw_event = self._unwrap_raw_event(event)
        if raw_event is None:
            return self._strip_nones(dict(event))

        symbol = (
            event.get("symbol")
            or event.get("ativo")
            or raw_event.get("symbol")
            or raw_event.get("ativo")
        )
        window_id = (
            event.get("janela_numero")
            or raw_event.get("janela_numero")
            or raw_event.get("window_id")
            or raw_event.get("features_window_id")
        )
        epoch_ms = (
            event.get("epoch_ms")
            or event.get("timestamp_ms")
            or raw_event.get("epoch_ms")
            or raw_event.get("timestamp_ms")
            or raw_event.get("window_close_ms")
            or raw_event.get("window_id")
        )

        payload: Dict[str, Any] = {
            "_v": 1,
            "symbol": symbol,
            "_w": window_id,
            "ts": epoch_ms or raw_event.get("timestamp_utc") or event.get("timestamp_utc"),
            "price": self._extract_price(raw_event),
            "flow": self._extract_flow(raw_event),
            "ob": self._extract_orderbook(raw_event),
            "tf": self._extract_multi_tf(raw_event),
            "vp": self._extract_historical_vp(raw_event),
            "ctx": self._extract_market_context(raw_event),
            "deriv": self._extract_derivatives(raw_event, symbol),
        }

        return self._strip_nones(payload)

    @classmethod
    def estimate_savings(cls, event: Dict[str, Any], max_orderbook_levels: int = 50) -> Dict[str, Any]:
        if not isinstance(event, dict):
            return {"bytes_before": 0, "bytes_after": 0, "saved_bytes": 0, "saved_pct": 0.0}

        original_json = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        bytes_before = len(original_json.encode("utf-8"))
        optimized = cls.optimize(event, max_orderbook_levels=max_orderbook_levels)
        optimized_json = json.dumps(optimized, ensure_ascii=False, separators=(",", ":"))
        bytes_after = len(optimized_json.encode("utf-8"))
        saved = max(0, bytes_before - bytes_after)
        saved_pct = (saved / bytes_before * 100.0) if bytes_before else 0.0
        reduction_pct = round(saved_pct, 2)
        original_tokens_est = cls._estimate_tokens(original_json)
        optimized_tokens_est = cls._estimate_tokens(optimized_json)
        return {
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "saved_bytes": saved,
            "saved_pct": reduction_pct,
            "reduction_pct": reduction_pct,
            "original_bytes": bytes_before,
            "optimized_bytes": bytes_after,
            "original_tokens_est": original_tokens_est,
            "optimized_tokens_est": optimized_tokens_est,
            "tokens_saved": max(0, original_tokens_est - optimized_tokens_est),
        }

    @staticmethod
    def _looks_like_structured_payload(event: Dict[str, Any]) -> bool:
        # Heurística simples: payload v2/v1 do builder usa estas chaves.
        return any(k in event for k in ("signal_metadata", "price_context", "flow_context", "orderbook_context", "_v"))

    @staticmethod
    def _unwrap_raw_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = event.get("raw_event")
        if isinstance(raw, dict):
            # Alguns formatos carregam raw_event.raw_event
            inner = raw.get("raw_event")
            if isinstance(inner, dict):
                return inner
            return raw

        # Se o prÃ³prio event parecer um raw_event, aceita.
        if any(k in event for k in ("preco_fechamento", "ohlc", "fluxo_continuo", "orderbook_data", "historical_vp")):
            return event
        return None

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        """
        Estimativa rÃ¡pida de tokens (best-effort).
        - Se tiktoken estiver disponÃ­vel, usa cl100k_base (aproximaÃ§Ã£o razoÃ¡vel).
        - Caso contrÃ¡rio, usa heurÃ­stica ~ 4 chars/token.
        """
        if not text:
            return 0
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return int(len(enc.encode(text)))
        except Exception:
            return max(1, int(round(len(text) / 4)))

    @classmethod
    def _extract_price(cls, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        ohlc = cls._extract_ohlc(raw_event)
        close = raw_event.get("preco_fechamento") or raw_event.get("preco_atual") or ohlc.get("close")
        price: Dict[str, Any] = {
            "c": close,
            "o": ohlc.get("open"),
            "h": ohlc.get("high"),
            "l": ohlc.get("low"),
        }
        return price

    @staticmethod
    def _extract_ohlc(raw_event: Dict[str, Any]) -> Dict[str, Any]:
        ohlc = (
            (raw_event.get("ohlc") or {})
            or ((raw_event.get("enriched_snapshot") or {}).get("ohlc") or {})
            or ((raw_event.get("contextual_snapshot") or {}).get("ohlc") or {})
            or (((raw_event.get("enriched_snapshot") or {}).get("enriched_snapshot") or {}).get("ohlc") or {})
        )
        ohlc = dict(ohlc) if isinstance(ohlc, dict) else {}
        return {
            "open": ohlc.get("open"),
            "high": ohlc.get("high"),
            "low": ohlc.get("low"),
            "close": ohlc.get("close"),
        }

    @staticmethod
    def _extract_flow(raw_event: Dict[str, Any]) -> Dict[str, Any]:
        fluxo = raw_event.get("fluxo_continuo") or (raw_event.get("contextual_snapshot") or {}).get("flow_metrics") or {}
        order_flow = fluxo.get("order_flow") or {}
        abs_analysis = fluxo.get("absorption_analysis") or {}
        current_abs = abs_analysis.get("current_absorption") or {}
        heatmap = fluxo.get("liquidity_heatmap") or raw_event.get("liquidity_heatmap") or {}
        clusters = heatmap.get("clusters")
        if isinstance(clusters, list):
            heatmap = dict(heatmap)
            heatmap["clusters"] = clusters[:2]
        return {
            "delta": raw_event.get("delta") or raw_event.get("delta_fechamento"),
            "cvd": fluxo.get("cvd"),
            "net_flow_1m": order_flow.get("net_flow_1m"),
            "net_flow_5m": order_flow.get("net_flow_5m"),
            "net_flow_15m": order_flow.get("net_flow_15m"),
            "flow_imbalance": order_flow.get("flow_imbalance"),
            "aggr_buy_pct": order_flow.get("aggressive_buy_pct"),
            "aggr_sell_pct": order_flow.get("aggressive_sell_pct"),
            "abs_label": fluxo.get("tipo_absorcao") or current_abs.get("label") or raw_event.get("resultado_da_batalha"),
            "abs_idx": current_abs.get("index") or raw_event.get("indice_absorcao"),
            "abs_side": raw_event.get("absorption_side"),
            "liq_hm": heatmap if isinstance(heatmap, dict) else None,
        }

    def _extract_orderbook(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        ob = raw_event.get("orderbook_data") or (raw_event.get("contextual_snapshot") or {}).get("orderbook_data") or {}
        compact = {
            "spr": ob.get("spread") or ob.get("spread_percent"),
            "imb": ob.get("imbalance"),
            "bid": ob.get("bid_depth_usd"),
            "ask": ob.get("ask_depth_usd"),
            "dimb": (ob.get("depth_metrics") or {}).get("depth_imbalance") if isinstance(ob.get("depth_metrics"), dict) else None,
        }
        for side in ("bids", "asks"):
            levels = ob.get(side)
            if isinstance(levels, list):
                if len(levels) > self.max_orderbook_levels:
                    compact[side] = levels[: self.max_orderbook_levels]
                    compact[f"{side}_total_levels"] = len(levels)
                    compact[f"{side}_truncated"] = True
                else:
                    compact[side] = levels
        return compact

    @staticmethod
    def _extract_multi_tf(raw_event: Dict[str, Any]) -> Dict[str, Any]:
        mtf = (
            raw_event.get("multi_tf")
            or (raw_event.get("contextual_snapshot") or {}).get("multi_tf")
            or raw_event.get("multi_timeframe_trends")
            or {}
        )
        if not isinstance(mtf, dict):
            return {}

        def compact_tf(tf_data: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "trend": tf_data.get("tendencia") or tf_data.get("trend"),
                "regime": tf_data.get("regime"),
                "rsi": tf_data.get("rsi_short") or tf_data.get("rsi") or tf_data.get("rsi_14"),
                "adx": tf_data.get("adx") or tf_data.get("adx_14"),
                "macd": tf_data.get("macd") or tf_data.get("macd_line"),
                "macd_signal": tf_data.get("macd_signal") or tf_data.get("signal"),
            }

        compacted: Dict[str, Any] = {}
        for tf_name in ("1d", "4h", "1h", "15m", "5m"):
            tf_data = mtf.get(tf_name)
            if isinstance(tf_data, dict):
                compacted[tf_name] = AIPayloadOptimizer._strip_nones(compact_tf(tf_data))
        return compacted

    @staticmethod
    def _extract_historical_vp(raw_event: Dict[str, Any]) -> Dict[str, Any]:
        vp = (
            raw_event.get("historical_vp")
            or (raw_event.get("contextual_snapshot") or {}).get("historical_vp")
            or {}
        )

        current_price = (
            raw_event.get("preco_fechamento")
            or raw_event.get("preco_atual")
            or (AIPayloadOptimizer._extract_ohlc(raw_event) or {}).get("close")
        )
        try:
            current_price_f = float(current_price) if current_price is not None else None
        except Exception:
            current_price_f = None

        if not isinstance(vp, dict) or not vp:
            return {}

        return compact_historical_vp(
            vp,
            current_price=current_price_f,
            pct_range=0.05,
            max_levels=5,
            timeframes=("daily", "weekly", "monthly"),
        )

    @staticmethod
    def _extract_market_context(raw_event: Dict[str, Any]) -> Dict[str, Any]:
        ctx = raw_event.get("market_context") or (raw_event.get("contextual_snapshot") or {}).get("market_context") or {}
        env = raw_event.get("market_environment") or (raw_event.get("contextual_snapshot") or {}).get("market_environment") or {}
        if not isinstance(ctx, dict):
            ctx = {}
        if not isinstance(env, dict):
            env = {}
        return {
            "sess": ctx.get("trading_session") or ctx.get("session") or ctx.get("session_name"),
            "phase": ctx.get("session_phase"),
            "vol": env.get("volatility_regime"),
            "trend": env.get("trend_direction"),
            "struct": env.get("market_structure"),
            "risk": env.get("risk_sentiment"),
        }

    @staticmethod
    def _extract_derivatives(raw_event: Dict[str, Any], symbol: Optional[str]) -> Dict[str, Any]:
        deriv = raw_event.get("derivatives") or (raw_event.get("contextual_snapshot") or {}).get("derivatives") or {}
        if not isinstance(deriv, dict) or not deriv:
            return {}
        cand: Optional[Dict[str, Any]] = None
        if symbol and isinstance(deriv.get(symbol), dict):
            cand = deriv.get(symbol)
        if cand is None:
            for v in deriv.values():
                if isinstance(v, dict):
                    cand = v
                    break
        if not isinstance(cand, dict):
            return {}
        return {
            "fr": cand.get("funding_rate_percent") or cand.get("funding_rate"),
            "oi": cand.get("open_interest"),
            "lsr": cand.get("long_short_ratio"),
        }

    @staticmethod
    def _strip_nones(data: Any) -> Any:
        if isinstance(data, dict):
            cleaned = {k: AIPayloadOptimizer._strip_nones(v) for k, v in data.items()}
            return {k: v for k, v in cleaned.items() if v is not None and v != {} and v != []}
        if isinstance(data, list):
            return [AIPayloadOptimizer._strip_nones(v) for v in data]
        return data


def optimize_for_ai(event: Dict[str, Any], max_orderbook_levels: int = 50) -> Dict[str, Any]:
    return AIPayloadOptimizer.optimize(event, max_orderbook_levels=max_orderbook_levels)


def get_optimized_json(event: Dict[str, Any], max_orderbook_levels: int = 50) -> str:
    payload = AIPayloadOptimizer.optimize(event, max_orderbook_levels=max_orderbook_levels)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def get_optimized_json_minified(event: Dict[str, Any], max_orderbook_levels: int = 50) -> str:
    payload = AIPayloadOptimizer.optimize(event, max_orderbook_levels=max_orderbook_levels)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def compact_historical_vp(
    historical_vp: Dict[str, Any],
    *,
    current_price: Optional[float],
    pct_range: float = 0.05,
    max_levels: int = 5,
    timeframes: Iterable[str] = ("daily", "weekly", "monthly"),
) -> Dict[str, Any]:
    """
    Compacta o historical_vp para uso em contexto de LLM:
    - Mantém POC/VAH/VAL + status
    - Remove granularidade pesada (single_prints, volume_nodes, etc.) por omissão
    - Filtra HVNs/LVNs para manter apenas os mais próximos do preço atual

    Observação: a compactação é "best-effort" e tolerante a formatos variados.
    """
    if not isinstance(historical_vp, dict) or not historical_vp:
        return {}

    def _as_floats(values: Any) -> list[float]:
        out: list[float] = []
        if not isinstance(values, list):
            return out
        for v in values:
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    def _pick_nearby(levels: Any) -> Optional[list[float]]:
        vals = _as_floats(levels)
        if not vals:
            return None

        # Se não houver preço atual válido, retorna apenas os primeiros N.
        if current_price is None or current_price <= 0:
            trimmed = vals[: max(0, int(max_levels))]
            return trimmed or None

        band = abs(float(current_price)) * float(pct_range)
        nearby = [p for p in vals if abs(p - float(current_price)) <= band]
        if not nearby:
            return None

        nearby_sorted = sorted(nearby, key=lambda p: abs(p - float(current_price)))
        return nearby_sorted[: max(0, int(max_levels))] or None

    compacted: Dict[str, Any] = {}
    for tf in timeframes:
        tf_data = historical_vp.get(tf)
        if not isinstance(tf_data, dict) or not tf_data:
            continue

        out_tf: Dict[str, Any] = {
            "poc": tf_data.get("poc"),
            "vah": tf_data.get("vah"),
            "val": tf_data.get("val"),
            "status": tf_data.get("status"),
        }

        hvns_nearby = _pick_nearby(tf_data.get("hvns"))
        if hvns_nearby:
            out_tf["hvns_nearby"] = hvns_nearby

        lvns_nearby = _pick_nearby(tf_data.get("lvns"))
        if lvns_nearby:
            out_tf["lvns_nearby"] = lvns_nearby

        compacted[tf] = AIPayloadOptimizer._strip_nones(out_tf)

    return AIPayloadOptimizer._strip_nones(compacted)
