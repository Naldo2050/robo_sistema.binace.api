# ai_payload_optimizer.py  v2.0.0
"""
Otimizador de payload para API de IA.

v2.0.0 — Integração com compressão profunda:
  ✅ Interface original preservada (optimize, estimate_savings)
  ✅ Eliminação de duplicações (raw_event aninhado, contextual_snapshot)
  ✅ Compressão de chaves (nomes curtos)
  ✅ Redução de precisão numérica (por tipo de dado)
  ✅ Filtragem de HVNs/LVNs (apenas próximos ao preço)
  ✅ Cache de seções estáticas (onchain, options, VP mensal)
  ✅ Remoção de campos deriváveis/calculáveis
  ✅ System prompt compacto com dicionário integrado
  ✅ Todos os testes existentes continuam passando
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ================================================================
# SECTION CACHE — evita reenviar dados que não mudaram
# ================================================================


class SectionCache:
    """
    Cache de seções que mudam lentamente entre janelas consecutivas.
    Evita reenviar dados idênticos, economizando ~15-20% adicional.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl: Dict[str, int] = {
            "onchain": 300,       # 5 min — dados on-chain mudam devagar
            "options": 300,       # 5 min — métricas de opções
            "vp_monthly": 3600,   # 1 hora — VP mensal quase não muda
            "vp_weekly": 600,     # 10 min — VP semanal
            "deriv": 60,          # 1 min — derivativos mudam rápido
            "macro": 120,         # 2 min — sessão/regime
            "cross": 120,         # 2 min — correlações cross-asset
            "adapt_thresh": 300,  # 5 min — thresholds adaptativos
        }

    def _hash(self, data: Any) -> str:
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_and_update(self, section_name: str, data: Any) -> Tuple[bool, str]:
        """
        Verifica se a seção mudou desde o último envio.

        Returns:
            (changed: bool, hash: str)
        """
        if not data:
            return False, ""

        current_hash = self._hash(data)
        cached = self._cache.get(section_name)

        if cached is not None:
            ttl = self._ttl.get(section_name, 120)
            age = time.time() - cached["ts"]
            if cached["hash"] == current_hash and age < ttl:
                return False, current_hash

        self._cache[section_name] = {"hash": current_hash, "ts": time.time()}
        return True, current_hash

    def get_cached_sections(self) -> List[str]:
        """Retorna nomes das seções atualmente em cache (não reenviadas)."""
        now = time.time()
        active: List[str] = []
        for section, info in self._cache.items():
            ttl = self._ttl.get(section, 120)
            if (now - info["ts"]) < ttl:
                active.append(section)
        return active

    def clear(self) -> None:
        self._cache.clear()


# ================================================================
# PRECISION ROUNDER — reduz casas decimais sem perder significância
# ================================================================


class PrecisionRounder:
    """Arredonda valores numéricos conforme tipo de dado."""

    PRECISION: Dict[str, int] = {
        "price": 2,
        "percent": 2,
        "ratio": 3,
        "volume_btc": 3,
        "volume_usd": 0,
        "correlation": 3,
        "indicator": 2,
        "rate": 4,
        "score": 2,
        "default": 4,
    }

    @classmethod
    def r(cls, value: Any, precision_type: str = "default") -> Any:
        """Arredonda valor numérico. Retorna None se input for None."""
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            return value
        num_value: float = float(value)
        decimals = cls.PRECISION.get(precision_type, cls.PRECISION["default"])
        if decimals == 0:
            return int(round(num_value))
        return round(num_value, decimals)


# Atalho global
_r = PrecisionRounder.r


# ================================================================
# CLASSE PRINCIPAL — AIPayloadOptimizer v2.0
# ================================================================


class AIPayloadOptimizer:
    """
    Builds a compact, LLM-ready payload from a raw event.

    Dois modos de operação:
    - optimize()        : compressão básica (compatibilidade legada)
    - optimize_deep()   : compressão profunda com cache e precisão
    - estimate_savings() : estima economia em bytes/tokens

    A compressão profunda (optimize_deep) é chamada automaticamente
    pelo ai_analyzer_qwen.py quando payload_compression=true.
    """

    MAX_HVNS_NEARBY: int = 5
    MAX_LVNS_NEARBY: int = 3
    MAX_CLUSTERS: int = 5

    def __init__(
        self,
        max_orderbook_levels: int = 50,
        enable_section_cache: bool = False,
    ) -> None:
        self.max_orderbook_levels = max_orderbook_levels
        self._section_cache: Optional[SectionCache] = (
            SectionCache() if enable_section_cache else None
        )
        self._stats: Dict[str, int] = {
            "total_calls": 0,
            "total_chars_original": 0,
            "total_chars_compressed": 0,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Estatísticas acumuladas de compressão."""
        total_orig = self._stats["total_chars_original"]
        total_comp = self._stats["total_chars_compressed"]
        saving_pct = (1 - total_comp / total_orig) * 100 if total_orig > 0 else 0
        return {
            "calls": self._stats["total_calls"],
            "total_original_chars": total_orig,
            "total_compressed_chars": total_comp,
            "average_saving_pct": round(saving_pct, 1),
        }

    # ================================================================
    # INTERFACE PÚBLICA — COMPATIBILIDADE LEGADA
    # ================================================================

    @classmethod
    def optimize(
        cls, event: Dict[str, Any], max_orderbook_levels: int = 50
    ) -> Dict[str, Any]:
        """
        Otimiza payload (interface legada).
        Mantida para compatibilidade com testes e código existente.
        """
        optimizer = cls(max_orderbook_levels=max_orderbook_levels)
        return optimizer.optimize_event(event)

    def optimize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Otimização legada — compatível com testes existentes."""
        if not isinstance(event, dict):
            return {}

        if self._looks_like_structured_payload(event):
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
            "ts": (
                epoch_ms
                or raw_event.get("timestamp_utc")
                or event.get("timestamp_utc")
            ),
            "price": self._extract_price_legacy(raw_event),
            "flow": self._extract_flow_legacy(raw_event, event),
            "ob": self._extract_orderbook_legacy(raw_event),
            "tf": self._extract_multi_tf_legacy(raw_event),
            "vp": self._extract_historical_vp_legacy(raw_event),
            "ctx": self._extract_market_context_legacy(raw_event),
            "deriv": self._extract_derivatives_legacy(raw_event, symbol),
        }

        return self._strip_nones(payload)

    # ================================================================
    # INTERFACE PÚBLICA — COMPRESSÃO PROFUNDA (NOVO)
    # ================================================================

    @classmethod
    def optimize_deep(
        cls,
        event: Dict[str, Any],
        section_cache: Optional[SectionCache] = None,
    ) -> Dict[str, Any]:
        """
        Compressão profunda com eliminação de duplicações,
        precisão controlada, e cache de seções estáticas.

        Redução estimada: ~70% vs JSON original.

        Args:
            event: JSON original completo (com duplicações)
            section_cache: SectionCache compartilhado entre janelas

        Returns:
            JSON comprimido otimizado para LLM
        """
        optimizer = cls(enable_section_cache=section_cache is not None)
        if section_cache is not None:
            optimizer._section_cache = section_cache
        return optimizer._compress_deep(event)

    def _compress_deep(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compressão profunda — núcleo."""
        try:
            inner = self._navigate_to_inner(raw_payload)
            if not inner:
                logger.warning("Payload sem dados internos, usando optimize legado")
                return self.optimize_event(raw_payload)

            compressed: Dict[str, Any] = {}

            # === Metadados únicos ===
            inner_raw = inner.get("raw_event") or {}
            adv = inner_raw.get("advanced_analysis") or {}
            snap = inner.get("contextual_snapshot") or {}
            ohlc = snap.get("ohlc") or {}

            compressed["s"] = self._first_valid(
                raw_payload.get("symbol"),
                inner.get("symbol"),
                adv.get("symbol"),
                default="BTCUSDT",
            )
            compressed["t"] = inner.get("epoch_ms", raw_payload.get("epoch_ms"))
            compressed["w"] = raw_payload.get(
                "janela_numero", inner.get("janela_numero")
            )
            compressed["ev"] = raw_payload.get("tipo_evento", "ANALYSIS_TRIGGER")
            compressed["ctx"] = inner.get("data_context", "real_time")

            current_price = (
                ohlc.get("close")
                or inner_raw.get("preco_fechamento")
                or adv.get("price")
                or inner.get("preco_fechamento")
                or 0
            )

            # === Seções de dados (1 vez cada, sem duplicação) ===
            compressed["price"] = self._build_price_deep(
                ohlc, snap, inner_raw, current_price
            )
            compressed["vol"] = self._build_volume_deep(snap, inner_raw, inner)
            compressed["ob"] = self._build_orderbook_deep(inner, snap)
            compressed["flow"] = self._build_flow_deep(inner, raw_payload)

            # Multi-TF: fonte única (inner > snap), sem preco_atual duplicado
            multi_tf = inner.get("multi_tf") or snap.get("multi_tf") or {}
            compressed["tf"] = self._build_tf_deep(multi_tf)

            # Volume Profile com cache para semanal/mensal
            vp_data = snap.get("historical_vp") or inner.get("historical_vp") or {}
            compressed["vp"] = self._build_vp_deep(vp_data, current_price)

            # Derivativos — com cache
            deriv = inner.get("derivatives") or snap.get("derivatives") or {}
            compressed["deriv"] = self._maybe_cache(
                "deriv",
                lambda: self._build_deriv_deep(deriv),
                deriv,
            )

            # Macro/sessão — com cache
            mkt_ctx = (
                inner.get("market_context")
                or snap.get("market_context")
                or {}
            )
            mkt_env = (
                inner.get("market_environment")
                or snap.get("market_environment")
                or {}
            )
            compressed["macro"] = self._maybe_cache(
                "macro",
                lambda: self._build_macro_deep(mkt_ctx, mkt_env),
                {**mkt_ctx, **mkt_env},
            )

            # Cross-asset — com cache
            ml = inner.get("ml_features") or {}
            cross = ml.get("cross_asset") or {}
            compressed["cross"] = self._maybe_cache(
                "cross",
                lambda: self._build_cross_deep(cross, mkt_env),
                {
                    **cross,
                    **{k: v for k, v in mkt_env.items() if k.startswith("correlation_")},
                },
            )

            # On-chain — com cache
            onchain = adv.get("onchain_metrics") or {}
            if onchain:
                compressed["onchain"] = self._maybe_cache(
                    "onchain",
                    lambda: self._build_onchain_deep(onchain),
                    onchain,
                )

            # Options — com cache
            options = adv.get("options_metrics") or {}
            if options:
                compressed["options"] = self._maybe_cache(
                    "options",
                    lambda: self._build_options_deep(options),
                    options,
                )

            # Adaptive thresholds — com cache
            adapt = adv.get("adaptive_thresholds") or {}
            if adapt:
                compressed["adapt"] = self._maybe_cache(
                    "adapt_thresh",
                    lambda: {
                        "vol": _r(adapt.get("current_volatility"), "ratio"),
                        "vf": _r(adapt.get("volatility_factor"), "ratio"),
                        "abs_th": _r(adapt.get("absorption_threshold"), "ratio"),
                        "flow_th": _r(adapt.get("flow_threshold"), "ratio"),
                    },
                    adapt,
                )

            # ML Features
            if ml:
                compressed["ml"] = self._build_ml_deep(ml)

            # Quant Model (pode vir de ai_payload ou inner)
            quant = raw_payload.get("quant_model") or inner.get("quant_model") or {}
            if quant:
                compressed["quant"] = self._build_quant_deep(quant)

            # Seções em cache (omitidas por não terem mudado)
            if self._section_cache is not None:
                cached = self._section_cache.get_cached_sections()
                if cached:
                    compressed["_cached"] = cached

            # Limpar seções vazias
            compressed = {
                k: v
                for k, v in compressed.items()
                if v is not None and v != {} and v != []
            }

            # Métricas
            self._log_savings(raw_payload, compressed)

            return compressed

        except Exception as e:
            logger.error(f"Erro na compressão profunda: {e}", exc_info=True)
            return self.optimize_event(raw_payload)

    # ================================================================
    # DEEP BUILDERS — Compressão profunda com precisão controlada
    # ================================================================

    def _build_price_deep(
        self,
        ohlc: Dict[str, Any],
        snap: Dict[str, Any],
        inner_raw: Dict[str, Any],
        current_price: Any,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"c": _r(current_price, "price")}

        for key, short in [("open", "o"), ("high", "h"), ("low", "l")]:
            v = ohlc.get(key)
            if v is not None:
                result[short] = _r(v, "price")

        vwap = ohlc.get("vwap")
        if vwap is not None:
            result["vwap"] = _r(vwap, "price")

        poc = snap.get("poc_price")
        if poc is not None:
            result["poc"] = _r(poc, "price")
            pv = snap.get("poc_volume")
            if pv is not None:
                result["poc_v"] = _r(pv, "volume_btc")
            pp = snap.get("poc_percentage")
            if pp is not None:
                result["poc_p"] = _r(pp, "percent")

        dwell = snap.get("dwell_price")
        if dwell is not None:
            result["dw"] = _r(dwell, "price")
            ds = snap.get("dwell_seconds")
            if ds is not None:
                result["dw_s"] = ds
            dl = snap.get("dwell_location")
            if dl is not None:
                result["dw_l"] = dl

        return result

    def _build_volume_deep(
        self,
        snap: Dict[str, Any],
        inner_raw: Dict[str, Any],
        inner: Dict[str, Any],
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        vol_total = snap.get("volume_total") or inner_raw.get("volume_total")
        if vol_total is not None:
            result["tot"] = _r(vol_total, "volume_btc")

        vol_usd = snap.get("volume_total_usdt")
        if vol_usd is not None:
            result["usd"] = _r(vol_usd, "volume_usd")

        buy = snap.get("volume_compra") or inner.get("volume_compra")
        sell = snap.get("volume_venda") or inner.get("volume_venda")
        if buy is not None:
            result["buy"] = _r(buy, "volume_btc")
        if sell is not None:
            result["sell"] = _r(sell, "volume_btc")

        trades = snap.get("num_trades")
        if trades is not None:
            result["n"] = trades

        tps = snap.get("trades_per_second")
        if tps is not None:
            result["tps"] = _r(tps, "percent")

        avg = snap.get("avg_trade_size")
        if avg is not None:
            result["avg"] = _r(avg, "rate")

        # Deltas
        d_min = snap.get("delta_minimo")
        d_max = snap.get("delta_maximo")
        d_close = (
            snap.get("delta_fechamento")
            or inner_raw.get("delta")
            or inner.get("delta")
        )
        if any(v is not None for v in [d_min, d_max, d_close]):
            delta: Dict[str, Any] = {}
            if d_min is not None:
                delta["min"] = _r(d_min, "volume_btc")
            if d_max is not None:
                delta["max"] = _r(d_max, "volume_btc")
            if d_close is not None:
                delta["c"] = _r(d_close, "volume_btc")
            result["d"] = delta

        rev_min = snap.get("reversao_desde_minimo")
        rev_max = snap.get("reversao_desde_maximo")
        if rev_min is not None or rev_max is not None:
            rev: Dict[str, Any] = {}
            if rev_min is not None:
                rev["min"] = _r(rev_min, "volume_btc")
            if rev_max is not None:
                rev["max"] = _r(rev_max, "volume_btc")
            result["rev"] = rev

        return result

    def _build_orderbook_deep(
        self, inner: Dict[str, Any], snap: Dict[str, Any]
    ) -> Dict[str, Any]:
        ob = inner.get("orderbook_data") or snap.get("orderbook_data") or {}
        if not ob:
            return {}

        result: Dict[str, Any] = {}

        mid = ob.get("mid")
        if mid is not None:
            result["mid"] = _r(mid, "price")

        spr = ob.get("spread")
        if spr is not None:
            result["spr"] = _r(spr, "price")

        bid = ob.get("bid_depth_usd")
        if bid is not None:
            result["bid"] = _r(bid, "volume_usd")

        ask = ob.get("ask_depth_usd")
        if ask is not None:
            result["ask"] = _r(ask, "volume_usd")

        imb = ob.get("imbalance")
        if imb is not None:
            result["imb"] = _r(imb, "ratio")

        pressure = ob.get("pressure")
        if pressure is not None:
            result["prs"] = _r(pressure, "ratio")

        # Depth levels compactos
        ob_depth = inner.get("order_book_depth") or {}
        if ob_depth:
            depth: Dict[str, Any] = {}
            for level in ["L1", "L5", "L10", "L25"]:
                lvl = ob_depth.get(level)
                if isinstance(lvl, dict):
                    depth[level] = {
                        "b": _r(lvl.get("bids"), "volume_usd"),
                        "a": _r(lvl.get("asks"), "volume_usd"),
                        "i": _r(lvl.get("imbalance"), "ratio"),
                    }
            if depth:
                result["dep"] = depth

            ratio = ob_depth.get("total_depth_ratio")
            if ratio is not None:
                result["dr"] = _r(ratio, "ratio")

        # Spread BPS
        spread_a = inner.get("spread_analysis") or {}
        bps = spread_a.get("current_spread_bps")
        if bps is not None:
            result["bps"] = _r(bps, "rate")

        # Market impact
        impact = inner.get("market_impact") or {}
        if impact:
            slip = impact.get("slippage_matrix") or {}
            if slip:
                imp: Dict[str, Any] = {}
                for k, v in slip.items():
                    if isinstance(v, dict):
                        short = k.replace("_usd", "")
                        imp[short] = {
                            "b": _r(v.get("buy"), "percent"),
                            "s": _r(v.get("sell"), "percent"),
                        }
                if imp:
                    result["imp"] = imp

            liq = impact.get("liquidity_score")
            if liq is not None:
                result["liq"] = _r(liq, "score")

            exq = impact.get("execution_quality")
            if exq is not None:
                result["exq"] = exq

        # Depth metrics (do ob original)
        dm = ob.get("depth_metrics")
        if isinstance(dm, dict):
            dimb = dm.get("depth_imbalance")
            if dimb is not None:
                result["dimb"] = _r(dimb, "ratio")

        return result

    def _build_flow_deep(
        self, inner: Dict[str, Any], raw_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        flow = (
            inner.get("fluxo_continuo")
            or raw_payload.get("fluxo_continuo")
            or {}
        )
        if not flow:
            return {}

        of = flow.get("order_flow") or {}
        absorb = (
            (flow.get("absorption_analysis") or {}).get("current_absorption")
            or {}
        )
        sectors = flow.get("sector_flow") or {}
        part = flow.get("participant_analysis") or {}

        result: Dict[str, Any] = {}

        # Delta do evento
        delta = (
            inner.get("delta")
            or (inner.get("raw_event") or {}).get("delta")
        )
        if delta is not None:
            result["delta"] = _r(delta, "volume_btc")

        cvd = flow.get("cvd")
        if cvd is not None:
            result["cvd"] = _r(cvd, "volume_btc")

        fi = of.get("flow_imbalance")
        if fi is not None:
            result["imb"] = _r(fi, "ratio")

        agr_b = of.get("aggressive_buy_pct")
        agr_s = of.get("aggressive_sell_pct")
        if agr_b is not None:
            result["ab"] = _r(agr_b, "percent")
        if agr_s is not None:
            result["as"] = _r(agr_s, "percent")

        # Net flows multi-window
        net: Dict[str, Any] = {}
        for window in ["1m", "5m", "15m"]:
            nf = of.get(f"net_flow_{window}")
            if nf is not None:
                net[window] = _r(nf, "volume_usd")
        if net:
            result["net"] = net

        # Absorções multi-window
        abs_dict: Dict[str, str] = {}
        for window in ["1m", "5m", "15m"]:
            a = of.get(f"absorcao_{window}")
            if a is not None:
                abs_dict[window] = a
        if abs_dict:
            result["abs"] = abs_dict

        # Volumes USD
        bv = of.get("buy_volume")
        sv = of.get("sell_volume")
        if bv is not None:
            result["bv"] = _r(bv, "volume_usd")
        if sv is not None:
            result["sv"] = _r(sv, "volume_usd")

        # Volumes BTC
        bb = of.get("buy_volume_btc")
        sb = of.get("sell_volume_btc")
        if bb is not None:
            result["bb"] = _r(bb, "volume_btc")
        if sb is not None:
            result["sb"] = _r(sb, "volume_btc")

        # Absorção detalhada
        abs_label = (
            flow.get("tipo_absorcao")
            or absorb.get("label")
            or inner.get("resultado_da_batalha")
        )
        if abs_label is not None:
            result["abs_lbl"] = abs_label

        if absorb:
            absr: Dict[str, Any] = {}
            idx = absorb.get("index")
            if idx is not None:
                absr["i"] = _r(idx, "ratio")
            cls_ = absorb.get("classification")
            if cls_ is not None:
                absr["c"] = cls_
            bs = absorb.get("buyer_strength")
            if bs is not None:
                absr["bs"] = bs
            se = absorb.get("seller_exhaustion")
            if se is not None:
                absr["se"] = se
            cp = absorb.get("continuation_probability")
            if cp is not None:
                absr["cp"] = _r(cp, "ratio")
            if absr:
                result["absr"] = absr

        # Setores (merge com participant_analysis)
        if sectors or part:
            merged: Dict[str, Any] = {}
            all_keys = set(list(sectors.keys()) + list(part.keys()))
            for sector in all_keys:
                entry: Dict[str, Any] = {}
                s_data = sectors.get(sector) or {}
                p_data = part.get(sector) or {}

                for k, v in s_data.items():
                    if isinstance(v, (int, float)):
                        entry[k] = _r(v, "volume_btc")
                    else:
                        entry[k] = v

                if p_data:
                    vp_ = p_data.get("volume_pct")
                    if vp_ is not None:
                        entry["vp"] = _r(vp_, "percent")
                    d = p_data.get("direction")
                    if d is not None:
                        entry["dir"] = d
                    sent = p_data.get("sentiment")
                    if sent is not None:
                        entry["sent"] = sent
                    sc = p_data.get("composite_score")
                    if sc is not None:
                        entry["sc"] = _r(sc, "score")

                if entry:
                    merged[sector] = entry

            if merged:
                result["sec"] = merged

        # Clusters de liquidez (limitados)
        heatmap = flow.get("liquidity_heatmap") or inner.get("liquidity_heatmap") or {}
        clusters = heatmap.get("clusters") or []
        if clusters:
            result["clust"] = [
                {
                    "c": _r(c.get("center"), "price"),
                    "lo": _r(c.get("low"), "price"),
                    "hi": _r(c.get("high"), "price"),
                    "v": _r(c.get("total_volume"), "volume_btc"),
                    "imb": _r(c.get("imbalance_ratio"), "ratio"),
                    "n": c.get("trades_count"),
                    "dur": c.get("cluster_duration_ms"),
                }
                for c in clusters[: self.MAX_CLUSTERS]
            ]

            supports = heatmap.get("supports") or []
            if supports:
                result["sup"] = [_r(s, "price") for s in supports[:5]]

        return result

    def _build_tf_deep(self, multi_tf: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-TF comprimido. NÃO inclui preco_atual (já em price.c)."""
        if not multi_tf:
            return {}

        result: Dict[str, Any] = {}
        for tf_key, data in multi_tf.items():
            if not isinstance(data, dict):
                continue

            entry: Dict[str, Any] = {}

            MAPPING = [
                ("tendencia", "trend", None),
                ("trend", "trend", None),
                ("regime", "regime", None),
                ("mme_21", "ema", "price"),
                ("atr", "atr", "price"),
                ("rsi_short", "rsi", "indicator"),
                ("rsi", "rsi", "indicator"),
                ("rsi_long", "rsi_l", "indicator"),
                ("macd", "macd", "indicator"),
                ("macd_signal", "macd_s", "indicator"),
                ("adx", "adx", "indicator"),
                ("realized_vol", "rvol", "rate"),
            ]

            seen_keys: set = set()
            for src, dst, prec in MAPPING:
                if dst in seen_keys:
                    continue
                v = data.get(src)
                if v is not None:
                    entry[dst] = _r(v, prec) if prec else v
                    seen_keys.add(dst)

            if entry:
                result[tf_key] = entry

        return result

    def _build_vp_deep(
        self, vp: Dict[str, Any], current_price: float
    ) -> Dict[str, Any]:
        """Volume Profile com cache para semanal/mensal."""
        if not vp:
            return {}

        result: Dict[str, Any] = {}

        # Daily — sempre enviar
        daily = vp.get("daily") or {}
        if daily and daily.get("status") == "success":
            d_entry: Dict[str, Any] = {
                "poc": daily.get("poc"),
                "vah": daily.get("vah"),
                "val": daily.get("val"),
            }
            hvns = daily.get("hvns") or []
            if hvns and current_price:
                sorted_h = sorted(hvns, key=lambda x: abs(x - current_price))
                d_entry["hvn"] = sorted_h[: self.MAX_HVNS_NEARBY]
            result["daily"] = self._strip_nones(d_entry)

        # Weekly — com cache
        weekly = vp.get("weekly") or {}
        if weekly and weekly.get("status") == "success":
            w_core = {
                "poc": weekly.get("poc"),
                "vah": weekly.get("vah"),
                "val": weekly.get("val"),
            }
            changed, _ = (
                self._section_cache.check_and_update("vp_weekly", w_core)
                if self._section_cache
                else (True, "")
            )
            if changed:
                w_entry = dict(w_core)
                hvns = weekly.get("hvns") or []
                if hvns and current_price:
                    sorted_h = sorted(hvns, key=lambda x: abs(x - current_price))
                    w_entry["hvn"] = sorted_h[: self.MAX_HVNS_NEARBY]
                lvns = weekly.get("lvns") or []
                if lvns and current_price:
                    sorted_l = sorted(lvns, key=lambda x: abs(x - current_price))
                    w_entry["lvn"] = sorted_l[: self.MAX_LVNS_NEARBY]
                result["weekly"] = self._strip_nones(w_entry)

        # Monthly — com cache (muda raramente)
        monthly = vp.get("monthly") or {}
        if monthly and monthly.get("status") == "success":
            m_core = {
                "poc": monthly.get("poc"),
                "vah": monthly.get("vah"),
                "val": monthly.get("val"),
            }
            changed, _ = (
                self._section_cache.check_and_update("vp_monthly", m_core)
                if self._section_cache
                else (True, "")
            )
            if changed:
                result["monthly"] = self._strip_nones(m_core)

        return result

    def _build_deriv_deep(self, deriv: Dict[str, Any]) -> Dict[str, Any]:
        if not deriv:
            return {}

        KEY_MAP = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
        result: Dict[str, Any] = {}

        for full_key, data in deriv.items():
            if not isinstance(data, dict):
                continue
            short = KEY_MAP.get(full_key, full_key[:3])
            entry: Dict[str, Any] = {}

            fr = data.get("funding_rate_percent") or data.get("funding_rate")
            if fr is not None:
                entry["fr"] = _r(fr, "rate")
            oi = data.get("open_interest")
            if oi is not None:
                entry["oi"] = _r(oi, "volume_btc")
            oi_usd = data.get("open_interest_usd")
            if oi_usd is not None:
                entry["oi$"] = _r(oi_usd, "volume_usd")
            lsr = data.get("long_short_ratio")
            if lsr is not None:
                entry["lsr"] = _r(lsr, "ratio")

            if entry:
                result[short] = entry

        return result

    def _build_macro_deep(
        self, ctx: Dict[str, Any], env: Dict[str, Any]
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for key, short in [
            ("trading_session", "ses"),
            ("session_phase", "ph"),
            ("market_hours_type", "hrs"),
        ]:
            v = ctx.get(key) or ctx.get(key.split("_")[-1])
            if v is not None:
                result[short] = v

        day = ctx.get("day_of_week")
        if day is not None:
            result["day"] = day

        close = ctx.get("time_to_session_close")
        if close is not None:
            result["cls_in"] = close

        for key, short in [
            ("volatility_regime", "vr"),
            ("trend_direction", "td"),
            ("market_structure", "ms"),
            ("liquidity_environment", "le"),
            ("risk_sentiment", "rs"),
        ]:
            v = env.get(key)
            if v is not None:
                result[short] = v

        return result

    def _build_cross_deep(
        self, cross: Dict[str, Any], env: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not cross and not env:
            return {}

        result: Dict[str, Any] = {}

        # Correlações BTC-ETH
        corr_pairs = [
            ("btc_eth_corr_7d", "btc_eth_corr_30d", "be"),
            ("btc_dxy_corr_30d", "btc_dxy_corr_90d", "bd"),
        ]
        for short_key, long_key, name in corr_pairs:
            s = cross.get(short_key)
            l_ = cross.get(long_key)
            if s is not None or l_ is not None:
                entry: Dict[str, Any] = {}
                if s is not None:
                    entry["s"] = _r(s, "correlation")
                if l_ is not None:
                    entry["l"] = _r(l_, "correlation")
                result[name] = entry

        bn = cross.get("btc_ndx_corr_30d")
        if bn is not None:
            result["bn"] = _r(bn, "correlation")

        # Correlações do environment
        cor: Dict[str, Any] = {}
        for env_key, short in [
            ("correlation_spy", "spy"),
            ("correlation_dxy", "dxy"),
            ("correlation_gold", "gld"),
        ]:
            v = env.get(env_key)
            if v is not None:
                cor[short] = _r(v, "correlation")
        if cor:
            result["cor"] = cor

        # DXY momentum
        dr5 = cross.get("dxy_return_5d")
        dr20 = cross.get("dxy_return_20d")
        if dr5 is not None or dr20 is not None:
            dr: Dict[str, Any] = {}
            if dr5 is not None:
                dr["5"] = _r(dr5, "percent")
            if dr20 is not None:
                dr["20"] = _r(dr20, "percent")
            result["dr"] = dr

        dm = cross.get("dxy_momentum")
        if dm is not None:
            result["dm"] = _r(dm, "indicator")

        regime = cross.get("correlation_regime")
        if regime is not None:
            result["rg"] = regime

        return result

    def _build_onchain_deep(self, oc: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        MAPPING: List[Tuple[str, str, Optional[str]]] = [
            ("exchange_netflow", "nf", "volume_btc"),
            ("whale_transactions", "wh", None),
            ("miner_flows", "mn", "volume_btc"),
            ("exchange_reserves", "res", "volume_usd"),
            ("active_addresses", "aa", None),
            ("hash_rate", "hr", "indicator"),
            ("difficulty", "df", "indicator"),
            ("sopr", "sopr", "ratio"),
        ]

        for src, dst, prec in MAPPING:
            v = oc.get(src)
            if v is not None:
                result[dst] = _r(v, prec) if prec else v

        funding = oc.get("funding_rates") or {}
        if funding:
            result["fr"] = {
                k[:3].lower(): _r(v, "rate") for k, v in funding.items()
            }

        return result

    def _build_options_deep(self, opt: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        MAPPING: List[Tuple[str, str, Optional[str]]] = [
            ("put_call_ratio", "pcr", "ratio"),
            ("implied_volatility", "iv", "ratio"),
            ("iv_percentile", "ivp", "ratio"),
            ("iv_rank", "ivr", "ratio"),
            ("gamma_exposure", "gex", "volume_usd"),
            ("max_pain", "mp", None),
            ("vix", "vix", "indicator"),
            ("skew", "skew", "ratio"),
        ]

        for src, dst, prec in MAPPING:
            v = opt.get(src)
            if v is not None:
                result[dst] = _r(v, prec) if prec else v

        return result

    def _build_ml_deep(self, ml: Dict[str, Any]) -> Dict[str, Any]:
        pf = ml.get("price_features") or {}
        vf = ml.get("volume_features") or {}
        ms = ml.get("microstructure") or {}

        result: Dict[str, Any] = {}

        # Valores muito pequenos — manter notação original
        ret = pf.get("returns_15")
        if ret is not None:
            result["r15"] = ret
        vol = pf.get("volatility_15")
        if vol is not None:
            result["v15"] = vol

        mom = pf.get("momentum_score")
        if mom is not None:
            result["mom"] = _r(mom, "ratio")

        for src, dst in [
            ("volume_sma_ratio", "vsr"),
            ("volume_momentum", "vm"),
            ("buy_sell_pressure", "bsp"),
            ("liquidity_gradient", "lg"),
        ]:
            v = vf.get(src)
            if v is not None:
                result[dst] = _r(v, "ratio")

        fi = ms.get("flow_imbalance")
        if fi is not None:
            result["fi"] = _r(fi, "ratio")

        ts = ms.get("tick_rule_sum")
        if ts is not None:
            result["ts"] = ts

        return result

    def _build_quant_deep(self, quant: Dict[str, Any]) -> Dict[str, Any]:
        """Comprime dados do modelo quantitativo."""
        result: Dict[str, Any] = {}

        pu = quant.get("model_probability_up")
        if pu is not None:
            result["pu"] = _r(pu, "ratio")

        pd_ = quant.get("model_probability_down")
        if pd_ is not None:
            result["pd"] = _r(pd_, "ratio")

        ab = quant.get("action_bias")
        if ab is not None:
            result["ab"] = ab

        cs = quant.get("confidence_score")
        if cs is not None:
            result["cs"] = _r(cs, "ratio")

        sent = quant.get("model_sentiment")
        if sent is not None:
            result["sent"] = sent

        fu = quant.get("features_used")
        tf = quant.get("total_features")
        if fu is not None and tf is not None:
            result["feat"] = f"{fu}/{tf}"

        return result

    # ================================================================
    # NAVEGAÇÃO E UTILIDADES
    # ================================================================

    def _navigate_to_inner(
        self, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Navega pela estrutura aninhada até encontrar os dados reais.
        
        PRIORIDADE:
        1. raw_event (contém TODOS os dados de mercado)
        2. contextual_snapshot no nível atual
        3. ai_payload (fallback — já filtrado, menos dados)
        4. payload como está
        
        Lida com o padrão raw_event.raw_event do sistema.
        """
        if not isinstance(payload, dict):
            return None

        # ====================================================
        # CASO 1: raw_event existe (PRIORIDADE MÁXIMA)
        # Contém todos os dados brutos de mercado
        # ====================================================
        raw = payload.get("raw_event")
        if isinstance(raw, dict):
            # Caso 1a: raw_event tem contextual_snapshot diretamente
            if "contextual_snapshot" in raw:
                return raw

            # Caso 1b: raw_event.raw_event (aninhamento duplo)
            inner_raw = raw.get("raw_event")
            if isinstance(inner_raw, dict):
                # Mesclar: manter tudo do nível intermediário (raw)
                # mas substituir raw_event pelo inner real
                merged = dict(raw)
                merged["raw_event"] = inner_raw
                # Se contextual_snapshot está no nível intermediário, manter
                if "contextual_snapshot" not in merged:
                    # Procurar contextual_snapshot em qualquer nível
                    snap = raw.get("contextual_snapshot") or inner_raw.get("contextual_snapshot")
                    if isinstance(snap, dict):
                        merged["contextual_snapshot"] = snap
                return merged

            # Caso 1c: raw_event simples (sem aninhamento duplo)
            # Verificar se tem dados úteis de mercado
            has_market_data = any(
                k in raw
                for k in (
                    "multi_tf",
                    "orderbook_data",
                    "fluxo_continuo",
                    "flow_metrics",
                    "historical_vp",
                    "derivatives",
                    "market_context",
                    "market_environment",
                    "ml_features",
                )
            )
            if has_market_data:
                return raw

        # ====================================================
        # CASO 2: contextual_snapshot no nível raiz
        # ====================================================
        if "contextual_snapshot" in payload:
            return payload

        # ====================================================
        # CASO 3: ai_payload (fallback — dados já filtrados)
        # Menos dados, mas melhor que nada
        # ====================================================
        ai_payload = payload.get("ai_payload")
        if isinstance(ai_payload, dict):
            # NÃO retornar se for structured payload (já é compacto)
            # O compressor não consegue extrair dados dele
            if not self._looks_like_structured_payload(ai_payload):
                return ai_payload

        # ====================================================
        # CASO 4: o próprio payload tem dados de mercado
        # ====================================================
        has_direct_data = any(
            k in payload
            for k in (
                "preco_fechamento",
                "ohlc",
                "fluxo_continuo",
                "orderbook_data",
                "historical_vp",
                "multi_tf",
            )
        )
        if has_direct_data:
            return payload

        return payload

    @staticmethod
    def _first_valid(*values: Any, default: Any = None) -> Any:
        """Retorna o primeiro valor não-None."""
        for v in values:
            if v is not None:
                return v
        return default

    def _maybe_cache(
        self,
        section: str,
        builder: Callable[[], Dict[str, Any]],
        raw_data: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Verifica cache antes de construir seção.
        Retorna None se dados não mudaram (será listada em _cached).
        """
        if not raw_data:
            return None

        if self._section_cache is not None:
            changed, _ = self._section_cache.check_and_update(section, raw_data)
            if not changed:
                return None

        result = builder()
        return result if result else None

    def _log_savings(
        self, original: Dict[str, Any], compressed: Dict[str, Any]
    ) -> None:
        """Registra métricas de economia."""
        try:
            orig_str = json.dumps(original, default=str)
            comp_str = json.dumps(compressed, default=str)
            orig_chars = len(orig_str)
            comp_chars = len(comp_str)

            self._stats["total_calls"] += 1
            self._stats["total_chars_original"] += orig_chars
            self._stats["total_chars_compressed"] += comp_chars

            saving_pct = (1 - comp_chars / orig_chars) * 100 if orig_chars > 0 else 0
            orig_tokens = int(orig_chars / 3.5)
            comp_tokens = int(comp_chars / 3.5)

            logger.info(
                "PAYLOAD_COMPRESSED chars=%d→%d (%.1f%%) tokens≈%d→%d",
                orig_chars,
                comp_chars,
                saving_pct,
                orig_tokens,
                comp_tokens,
            )
        except Exception as e:
            logger.debug(f"Erro ao calcular métricas: {e}")

    # ================================================================
    # ESTIMATE SAVINGS (interface mantida)
    # ================================================================

    @classmethod
    def estimate_savings(
        cls, event: Dict[str, Any], max_orderbook_levels: int = 50
    ) -> Dict[str, Any]:
        """Estima economia de bytes/tokens com a otimização."""
        if not isinstance(event, dict):
            return {
                "bytes_before": 0,
                "bytes_after": 0,
                "saved_bytes": 0,
                "saved_pct": 0.0,
            }

        original_json = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        bytes_before = len(original_json.encode("utf-8"))

        optimized = cls.optimize(event, max_orderbook_levels=max_orderbook_levels)
        optimized_json = json.dumps(
            optimized, ensure_ascii=False, separators=(",", ":")
        )
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

    @classmethod
    def estimate_savings_deep(
        cls,
        event: Dict[str, Any],
        section_cache: Optional[SectionCache] = None,
    ) -> Dict[str, Any]:
        """Estima economia com compressão profunda."""
        if not isinstance(event, dict):
            return {
                "bytes_before": 0,
                "bytes_after": 0,
                "saved_bytes": 0,
                "saved_pct": 0.0,
            }

        original_json = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        bytes_before = len(original_json.encode("utf-8"))

        optimized = cls.optimize_deep(event, section_cache=section_cache)
        optimized_json = json.dumps(
            optimized, ensure_ascii=False, separators=(",", ":")
        )
        bytes_after = len(optimized_json.encode("utf-8"))

        saved = max(0, bytes_before - bytes_after)
        saved_pct = (saved / bytes_before * 100.0) if bytes_before else 0.0

        original_tokens = cls._estimate_tokens(original_json)
        optimized_tokens = cls._estimate_tokens(optimized_json)

        return {
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
            "saved_bytes": saved,
            "saved_pct": round(saved_pct, 2),
            "original_tokens_est": original_tokens,
            "optimized_tokens_est": optimized_tokens,
            "tokens_saved": max(0, original_tokens - optimized_tokens),
        }

    # ================================================================
    # MÉTODOS LEGADOS (compatibilidade com testes)
    # ================================================================

    @staticmethod
    def _looks_like_structured_payload(event: Dict[str, Any]) -> bool:
        return any(
            k in event
            for k in (
                "signal_metadata",
                "price_context",
                "flow_context",
                "orderbook_context",
                "_v",
            )
        )

    @staticmethod
    def _unwrap_raw_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = event.get("raw_event")
        if isinstance(raw, dict):
            inner = raw.get("raw_event")
            if isinstance(inner, dict):
                return inner
            return raw
        if any(
            k in event
            for k in (
                "preco_fechamento",
                "ohlc",
                "fluxo_continuo",
                "orderbook_data",
                "historical_vp",
            )
        ):
            return event
        return None

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        if not text:
            return 0
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return int(len(enc.encode(text)))
        except Exception:
            return max(1, int(round(len(text) / 4)))

    @classmethod
    def _extract_price_legacy(cls, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        ohlc = cls._extract_ohlc(raw_event)
        close = (
            raw_event.get("preco_fechamento")
            or raw_event.get("preco_atual")
            or ohlc.get("close")
        )
        return {"c": close, "o": ohlc.get("open"), "h": ohlc.get("high"), "l": ohlc.get("low")}

    @staticmethod
    def _extract_ohlc(raw_event: Dict[str, Any]) -> Dict[str, Any]:
        ohlc = (
            (raw_event.get("ohlc") or {})
            or ((raw_event.get("enriched_snapshot") or {}).get("ohlc") or {})
            or ((raw_event.get("contextual_snapshot") or {}).get("ohlc") or {})
            or (
                (
                    (raw_event.get("enriched_snapshot") or {}).get(
                        "enriched_snapshot"
                    )
                    or {}
                ).get("ohlc")
                or {}
            )
        )
        ohlc = dict(ohlc) if isinstance(ohlc, dict) else {}
        return {
            "open": ohlc.get("open"),
            "high": ohlc.get("high"),
            "low": ohlc.get("low"),
            "close": ohlc.get("close"),
        }

    @staticmethod
    def _extract_flow_legacy(
        raw_event: Dict[str, Any], event: Dict[str, Any]
    ) -> Dict[str, Any]:
        fluxo = (
            raw_event.get("fluxo_continuo")
            or event.get("fluxo_continuo")
            or (raw_event.get("contextual_snapshot") or {}).get("flow_metrics")
            or {}
        )
        order_flow = fluxo.get("order_flow") or {}
        abs_analysis = fluxo.get("absorption_analysis") or {}
        current_abs = abs_analysis.get("current_absorption") or {}
        heatmap = (
            fluxo.get("liquidity_heatmap")
            or raw_event.get("liquidity_heatmap")
            or {}
        )
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
            "abs_label": (
                fluxo.get("tipo_absorcao")
                or current_abs.get("label")
                or raw_event.get("resultado_da_batalha")
            ),
            "abs_idx": current_abs.get("index") or raw_event.get("indice_absorcao"),
            "abs_side": raw_event.get("absorption_side"),
            "liq_hm": heatmap if isinstance(heatmap, dict) else None,
        }

    def _extract_orderbook_legacy(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        ob = (
            raw_event.get("orderbook_data")
            or (raw_event.get("contextual_snapshot") or {}).get("orderbook_data")
            or {}
        )
        compact: Dict[str, Any] = {
            "spr": ob.get("spread") or ob.get("spread_percent"),
            "imb": ob.get("imbalance"),
            "bid": ob.get("bid_depth_usd"),
            "ask": ob.get("ask_depth_usd"),
            "dimb": (
                (ob.get("depth_metrics") or {}).get("depth_imbalance")
                if isinstance(ob.get("depth_metrics"), dict)
                else None
            ),
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
    def _extract_multi_tf_legacy(raw_event: Dict[str, Any]) -> Dict[str, Any]:
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
                "rsi": (
                    tf_data.get("rsi_short")
                    or tf_data.get("rsi")
                    or tf_data.get("rsi_14")
                ),
                "adx": tf_data.get("adx") or tf_data.get("adx_14"),
                "macd": tf_data.get("macd") or tf_data.get("macd_line"),
                "macd_signal": tf_data.get("macd_signal") or tf_data.get("signal"),
            }

        compacted: Dict[str, Any] = {}
        for tf_name in ("1d", "4h", "1h", "15m", "5m"):
            tf_data = mtf.get(tf_name)
            if isinstance(tf_data, dict):
                compacted[tf_name] = AIPayloadOptimizer._strip_nones(
                    compact_tf(tf_data)
                )
        return compacted

    @staticmethod
    def _extract_historical_vp_legacy(raw_event: Dict[str, Any]) -> Dict[str, Any]:
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
    def _extract_market_context_legacy(raw_event: Dict[str, Any]) -> Dict[str, Any]:
        ctx = (
            raw_event.get("market_context")
            or (raw_event.get("contextual_snapshot") or {}).get("market_context")
            or {}
        )
        env = (
            raw_event.get("market_environment")
            or (raw_event.get("contextual_snapshot") or {}).get("market_environment")
            or {}
        )
        if not isinstance(ctx, dict):
            ctx = {}
        if not isinstance(env, dict):
            env = {}
        return {
            "sess": (
                ctx.get("trading_session")
                or ctx.get("session")
                or ctx.get("session_name")
            ),
            "phase": ctx.get("session_phase"),
            "vol": env.get("volatility_regime"),
            "trend": env.get("trend_direction"),
            "struct": env.get("market_structure"),
            "risk": env.get("risk_sentiment"),
        }

    @staticmethod
    def _extract_derivatives_legacy(
        raw_event: Dict[str, Any], symbol: Optional[str]
    ) -> Dict[str, Any]:
        deriv = (
            raw_event.get("derivatives")
            or (raw_event.get("contextual_snapshot") or {}).get("derivatives")
            or {}
        )
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
            cleaned = {
                k: AIPayloadOptimizer._strip_nones(v) for k, v in data.items()
            }
            return {k: v for k, v in cleaned.items() if v is not None and v != {} and v != []}
        if isinstance(data, list):
            return [AIPayloadOptimizer._strip_nones(v) for v in data]
        return data


# ================================================================
# FUNÇÕES PÚBLICAS DE CONVENIÊNCIA
# ================================================================


def optimize_for_ai(
    event: Dict[str, Any], max_orderbook_levels: int = 50
) -> Dict[str, Any]:
    """Otimização legada — compatibilidade."""
    return AIPayloadOptimizer.optimize(event, max_orderbook_levels=max_orderbook_levels)


def optimize_deep_for_ai(
    event: Dict[str, Any],
    section_cache: Optional[SectionCache] = None,
) -> Dict[str, Any]:
    """Compressão profunda para uso no ai_analyzer_qwen.py."""
    return AIPayloadOptimizer.optimize_deep(event, section_cache=section_cache)


def get_optimized_json(
    event: Dict[str, Any], max_orderbook_levels: int = 50
) -> str:
    payload = AIPayloadOptimizer.optimize(event, max_orderbook_levels=max_orderbook_levels)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def get_optimized_json_minified(
    event: Dict[str, Any], max_orderbook_levels: int = 50
) -> str:
    payload = AIPayloadOptimizer.optimize(event, max_orderbook_levels=max_orderbook_levels)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def get_deep_optimized_json(
    event: Dict[str, Any],
    section_cache: Optional[SectionCache] = None,
) -> str:
    """JSON comprimido profundamente, minificado."""
    payload = AIPayloadOptimizer.optimize_deep(event, section_cache=section_cache)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ================================================================
# COMPACT VP (mantido para compatibilidade)
# ================================================================


def compact_historical_vp(
    historical_vp: Dict[str, Any],
    *,
    current_price: Optional[float],
    pct_range: float = 0.05,
    max_levels: int = 5,
    timeframes: Iterable[str] = ("daily", "weekly", "monthly"),
) -> Dict[str, Any]:
    """
    Compacta o historical_vp para uso em contexto de LLM.
    Mantido para compatibilidade com código existente.
    """
    if not isinstance(historical_vp, dict) or not historical_vp:
        return {}

    def _as_floats(values: Any) -> List[float]:
        out: List[float] = []
        if not isinstance(values, list):
            return out
        for v in values:
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    def _pick_nearby(levels: Any) -> Optional[List[float]]:
        vals = _as_floats(levels)
        if not vals:
            return None

        if current_price is None or current_price <= 0:
            trimmed = vals[: max(0, int(max_levels))]
            return trimmed or None

        # current_price já verificado como não-None acima
        price_val: float = float(current_price)
        band = abs(price_val) * float(pct_range)
        nearby = [p for p in vals if abs(p - price_val) <= band]
        if not nearby:
            return None

        nearby_sorted = sorted(nearby, key=lambda p: abs(p - price_val))
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


# ================================================================
# SYSTEM PROMPT COMPACTO (para ai_analyzer_qwen.py)
# ================================================================


COMPRESSED_KEY_DICTIONARY = (
    "Chaves comprimidas:\n"
    "s=symbol t=epoch_ms w=janela ev=evento ctx=contexto\n"
    "price: c=close o=open h=high l=low vwap poc poc_v poc_p dw=dwell\n"
    "vol: tot=total usd buy sell n=trades tps avg d={min,max,c} rev\n"
    "ob: mid spr=spread bid ask imb prs=pressure dep=depth dr bps imp liq exq dimb\n"
    "flow: delta cvd imb ab=agr_buy as=agr_sell net abs bv sv bb sb abs_lbl absr sec clust sup\n"
    "tf: trend regime ema=ema21 atr rsi rsi_l macd macd_s adx rvol\n"
    "vp: daily/weekly/monthly poc vah val hvn lvn\n"
    "deriv: fr=funding oi oi$ lsr\n"
    "macro: ses=session ph=phase hrs day cls_in vr td ms le rs\n"
    "cross: be=btc_eth bd=btc_dxy bn cor dr dm rg\n"
    "onchain: nf=netflow wh=whales mn=miner res aa hr df sopr fr\n"
    "options: pcr iv ivp ivr gex mp vix skew\n"
    "ml: r15 v15 mom vsr vm bsp lg fi ts\n"
    "quant: pu=prob_up pd=prob_down ab=action_bias cs=confidence sent feat\n"
    "_cached=secoes estáveis omitidas"
)

SYSTEM_PROMPT_COMPRESSED = (
    "Analista institucional de fluxo e microestrutura cripto. "
    "Horizonte: scalp 5-15min.\n\n"
    + COMPRESSED_KEY_DICTIONARY
    + "\n\n"
    "REGRAS:\n"
    "1. Responda APENAS em portugues BR\n"
    "2. Use quant.ab (action_bias) como base principal\n"
    "3. So contrarie com evidencia MUITO forte no fluxo/orderbook\n"
    "4. Em duvida, action=wait\n"
    "5. _cached=secoes omitidas pois nao mudaram\n"
    "6. Responda SOMENTE JSON:\n"
    '{"sentiment":"bullish|bearish|neutral","confidence":0.0-1.0,'
    '"action":"buy|sell|hold|flat|wait|avoid",'
    '"rationale":"texto curto","entry_zone":"preco|null",'
    '"invalidation_zone":"preco|null","region_type":"tipo|null"}'
)