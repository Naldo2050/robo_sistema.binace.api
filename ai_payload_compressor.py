# ai_payload_compressor.py
"""
Compressor de payload para API de IA — v1.0.0
Elimina duplicações, comprime chaves e reduz precisão numérica.
Redução estimada: ~70% dos tokens sem perda de informação analítica.

Integra-se ao fluxo existente via ai_analyzer_qwen.py._create_prompt()
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SectionCache:
    """
    Cache de seções que mudam lentamente.
    Evita reenviar dados idênticos entre janelas consecutivas.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        # TTL em segundos para cada seção
        self._ttl: Dict[str, int] = {
            "onchain": 300,      # 5 min
            "options": 300,      # 5 min
            "vp_monthly": 3600,  # 1 hora
            "vp_weekly": 600,    # 10 min
            "deriv": 60,         # 1 min
            "macro": 120,        # 2 min
            "cross": 120,        # 2 min
            "adapt_thresh": 300, # 5 min
        }

    def _hash(self, data: Any) -> str:
        """Gera hash compacto dos dados."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_and_update(
        self, section_name: str, data: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Verifica se a seção mudou desde o último envio.

        Returns:
            (changed: bool, hash: str)
        """
        current_hash = self._hash(data)
        cached = self._cache.get(section_name)

        if cached is not None:
            ttl = self._ttl.get(section_name, 120)
            age = time.time() - cached["ts"]

            if cached["hash"] == current_hash and age < ttl:
                return False, current_hash

        self._cache[section_name] = {
            "hash": current_hash,
            "ts": time.time(),
        }
        return True, current_hash

    def get_cache_refs(self) -> Dict[str, str]:
        """Retorna referências de cache ativas para inclusão no payload."""
        refs: Dict[str, str] = {}
        now = time.time()
        for section, info in self._cache.items():
            ttl = self._ttl.get(section, 120)
            if (now - info["ts"]) < ttl:
                refs[section] = info["hash"]
        return refs


class PayloadCompressor:
    """
    Comprime o payload JSON antes de enviar à API de IA.

    Estratégias:
    1. Eliminação de duplicações (raw_event aninhado, contextual_snapshot)
    2. Compressão de chaves (nomes curtos)
    3. Redução de precisão numérica (por tipo de dado)
    4. Filtragem de HVNs/LVNs (apenas próximos ao preço)
    5. Cache de seções estáticas (onchain, options, VP mensal)
    6. Remoção de campos deriváveis/calculáveis
    """

    # Precisão decimal por tipo de dado
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

    MAX_HVNS_NEARBY: int = 5
    MAX_LVNS_NEARBY: int = 3

    def __init__(self, enable_section_cache: bool = True) -> None:
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
        """Retorna estatísticas de compressão acumuladas."""
        total_orig = self._stats["total_chars_original"]
        total_comp = self._stats["total_chars_compressed"]
        saving_pct = (
            (1 - total_comp / total_orig) * 100 if total_orig > 0 else 0
        )
        return {
            "calls": self._stats["total_calls"],
            "total_original_chars": total_orig,
            "total_compressed_chars": total_comp,
            "average_saving_pct": round(saving_pct, 1),
        }

    def compress(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprime o payload eliminando duplicações e otimizando dados.

        Args:
            raw_payload: JSON original com todas as duplicações
                         (pode ser o evento completo ou raw_event)

        Returns:
            JSON comprimido otimizado para a IA
        """
        try:
            # Navegação segura pela estrutura aninhada
            inner = self._navigate_to_inner(raw_payload)
            if not inner:
                logger.warning(
                    "Payload sem dados internos válidos, retornando minimals"
                )
                return self._minimal_payload(raw_payload)

            compressed: Dict[str, Any] = {}

            # === Metadados únicos ===
            compressed["s"] = self._find_first(
                raw_payload.get("symbol"),
                inner.get("symbol"),
                (inner.get("raw_event") or {}).get(
                    "advanced_analysis", {}
                ).get("symbol"),
                default="BTCUSDT",
            )
            compressed["t"] = inner.get(
                "epoch_ms", raw_payload.get("epoch_ms")
            )
            compressed["w"] = raw_payload.get(
                "janela_numero", inner.get("janela_numero")
            )
            compressed["ev"] = raw_payload.get(
                "tipo_evento", "ANALYSIS_TRIGGER"
            )
            compressed["ctx"] = inner.get("data_context", "real_time")

            # === Fonte principal: contextual_snapshot ===
            snap = inner.get("contextual_snapshot") or {}
            ohlc = snap.get("ohlc") or {}
            inner_raw = inner.get("raw_event") or {}
            adv = inner_raw.get("advanced_analysis") or {}

            current_price = (
                ohlc.get("close")
                or inner_raw.get("preco_fechamento")
                or adv.get("price")
                or 0
            )

            # === Seções de dados (1 vez cada) ===
            compressed["price"] = self._build_price(
                ohlc, snap, inner_raw, current_price
            )
            compressed["vol"] = self._build_volume(snap, inner_raw)
            compressed["ob"] = self._build_orderbook(inner, snap)
            compressed["flow"] = self._build_flow(inner)
            compressed["tf"] = self._build_timeframes(
                inner.get("multi_tf") or snap.get("multi_tf") or {}
            )

            # === Seções com cache ===
            vp_data = snap.get("historical_vp") or inner.get(
                "historical_vp"
            ) or {}
            compressed["vp"] = self._build_vp_with_cache(
                vp_data, current_price
            )

            deriv = inner.get("derivatives") or snap.get("derivatives") or {}
            compressed["deriv"] = self._maybe_cache(
                "deriv",
                lambda: self._build_derivatives(deriv),
                deriv,
            )

            mkt_ctx = inner.get("market_context") or snap.get(
                "market_context"
            ) or {}
            mkt_env = inner.get("market_environment") or snap.get(
                "market_environment"
            ) or {}
            compressed["macro"] = self._maybe_cache(
                "macro",
                lambda: self._build_macro(mkt_ctx, mkt_env),
                {**mkt_ctx, **mkt_env},
            )

            ml = inner.get("ml_features") or {}
            cross = ml.get("cross_asset") or {}
            compressed["cross"] = self._maybe_cache(
                "cross",
                lambda: self._build_cross_asset(cross, mkt_env),
                {**cross, **{
                    k: v for k, v in mkt_env.items()
                    if k.startswith("correlation_")
                }},
            )

            onchain = adv.get("onchain_metrics") or {}
            if onchain:
                compressed["onchain"] = self._maybe_cache(
                    "onchain",
                    lambda: self._build_onchain(onchain),
                    onchain,
                )

            options = adv.get("options_metrics") or {}
            if options:
                compressed["options"] = self._maybe_cache(
                    "options",
                    lambda: self._build_options(options),
                    options,
                )

            adapt = adv.get("adaptive_thresholds") or {}
            if adapt:
                compressed["adapt"] = self._maybe_cache(
                    "adapt_thresh",
                    lambda: {
                        "vol": self._r(
                            adapt.get("current_volatility"), "ratio"
                        ),
                        "vf": self._r(
                            adapt.get("volatility_factor"), "ratio"
                        ),
                        "abs_th": self._r(
                            adapt.get("absorption_threshold"), "ratio"
                        ),
                        "flow_th": self._r(
                            adapt.get("flow_threshold"), "ratio"
                        ),
                    },
                    adapt,
                )

            # === ML Features ===
            if ml:
                compressed["ml"] = self._build_ml_features(ml)

            # === Quant Model (se presente no ai_payload) ===
            quant = raw_payload.get("quant_model") or inner.get(
                "quant_model"
            ) or {}
            if quant:
                compressed["quant"] = self._build_quant(quant)

            # === Cache refs ===
            if self._section_cache is not None:
                refs = self._section_cache.get_cache_refs()
                if refs:
                    compressed["_cached"] = list(refs.keys())

            # === Remover seções vazias ===
            compressed = {
                k: v
                for k, v in compressed.items()
                if v is not None and v != {} and v != []
            }

            # === Log de economia ===
            self._log_savings(raw_payload, compressed)

            return compressed

        except Exception as e:
            logger.error(
                f"Erro na compressão do payload: {e}", exc_info=True
            )
            return raw_payload  # fallback seguro

    # ================================================================
    # NAVEGAÇÃO
    # ================================================================

    def _navigate_to_inner(
        self, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Navega pela estrutura aninhada até encontrar os dados reais.
        Lida com o padrão raw_event.raw_event do sistema.
        """
        # Caso 1: payload já é o inner (tem contextual_snapshot)
        if "contextual_snapshot" in payload:
            return payload

        # Caso 2: raw_event no topo
        raw = payload.get("raw_event")
        if isinstance(raw, dict):
            if "contextual_snapshot" in raw:
                return raw
            # Caso 3: raw_event.raw_event (aninhamento duplo)
            inner_raw = raw.get("raw_event")
            if isinstance(inner_raw, dict):
                # Mesclar dados úteis do nível intermediário
                merged = dict(raw)
                merged["raw_event"] = inner_raw
                return merged

        # Caso 4: ai_payload já montado
        ai_payload = payload.get("ai_payload")
        if isinstance(ai_payload, dict):
            return ai_payload

        return payload if isinstance(payload, dict) else None

    def _find_first(self, *values: Any, default: Any = None) -> Any:
        """Retorna o primeiro valor não-None."""
        for v in values:
            if v is not None:
                return v
        return default

    def _minimal_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Payload mínimo quando dados não podem ser extraídos."""
        return {
            "s": payload.get("symbol", "BTCUSDT"),
            "t": payload.get("epoch_ms"),
            "ev": payload.get("tipo_evento", "UNKNOWN"),
            "error": "unable_to_extract_data",
        }

    # ================================================================
    # BUILDERS
    # ================================================================

    def _build_price(
        self,
        ohlc: Dict[str, Any],
        snap: Dict[str, Any],
        inner_raw: Dict[str, Any],
        current_price: Any,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "c": self._r(current_price, "price"),
        }

        # OHLC - só se diferente do close
        o = ohlc.get("open")
        h = ohlc.get("high")
        lo = ohlc.get("low")
        if o is not None:
            result["o"] = self._r(o, "price")
        if h is not None:
            result["h"] = self._r(h, "price")
        if lo is not None:
            result["l"] = self._r(lo, "price")

        vwap = ohlc.get("vwap")
        if vwap is not None:
            result["vwap"] = self._r(vwap, "price")

        # POC da janela
        poc = snap.get("poc_price")
        if poc is not None:
            result["poc"] = self._r(poc, "price")
            poc_vol = snap.get("poc_volume")
            if poc_vol is not None:
                result["poc_v"] = self._r(poc_vol, "volume_btc")
            poc_pct = snap.get("poc_percentage")
            if poc_pct is not None:
                result["poc_p"] = self._r(poc_pct, "percent")

        # Dwell
        dwell = snap.get("dwell_price")
        if dwell is not None:
            result["dw"] = self._r(dwell, "price")
            dwell_s = snap.get("dwell_seconds")
            if dwell_s is not None:
                result["dw_s"] = dwell_s
            dwell_loc = snap.get("dwell_location")
            if dwell_loc is not None:
                result["dw_l"] = dwell_loc

        return result

    def _build_volume(
        self, snap: Dict[str, Any], inner_raw: Dict[str, Any]
    ) -> Dict[str, Any]:
        vol_total = snap.get("volume_total") or inner_raw.get("volume_total")
        vol_buy = snap.get("volume_compra") or inner_raw.get("volume_compra")
        vol_sell = snap.get("volume_venda") or inner_raw.get("volume_venda")

        result: Dict[str, Any] = {}

        if vol_total is not None:
            result["tot"] = self._r(vol_total, "volume_btc")
        vol_usd = snap.get("volume_total_usdt")
        if vol_usd is not None:
            result["usd"] = self._r(vol_usd, "volume_usd")
        if vol_buy is not None:
            result["buy"] = self._r(vol_buy, "volume_btc")
        if vol_sell is not None:
            result["sell"] = self._r(vol_sell, "volume_btc")

        trades = snap.get("num_trades")
        if trades is not None:
            result["n"] = trades

        tps = snap.get("trades_per_second")
        if tps is not None:
            result["tps"] = self._r(tps, "percent")

        avg = snap.get("avg_trade_size")
        if avg is not None:
            result["avg"] = self._r(avg, "rate")

        # Deltas
        d_min = snap.get("delta_minimo")
        d_max = snap.get("delta_maximo")
        d_close = snap.get("delta_fechamento") or inner_raw.get("delta")
        if any(v is not None for v in [d_min, d_max, d_close]):
            delta: Dict[str, Any] = {}
            if d_min is not None:
                delta["min"] = self._r(d_min, "volume_btc")
            if d_max is not None:
                delta["max"] = self._r(d_max, "volume_btc")
            if d_close is not None:
                delta["c"] = self._r(d_close, "volume_btc")
            result["d"] = delta

        # Reversals
        rev_min = snap.get("reversao_desde_minimo")
        rev_max = snap.get("reversao_desde_maximo")
        if rev_min is not None or rev_max is not None:
            rev: Dict[str, Any] = {}
            if rev_min is not None:
                rev["min"] = self._r(rev_min, "volume_btc")
            if rev_max is not None:
                rev["max"] = self._r(rev_max, "volume_btc")
            result["rev"] = rev

        return result

    def _build_orderbook(
        self, inner: Dict[str, Any], snap: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Fonte única de orderbook
        ob = (
            inner.get("orderbook_data")
            or snap.get("orderbook_data")
            or {}
        )
        if not ob:
            return {}

        result: Dict[str, Any] = {
            "mid": self._r(ob.get("mid"), "price"),
            "spr": self._r(ob.get("spread"), "price"),
        }

        bid = ob.get("bid_depth_usd")
        ask = ob.get("ask_depth_usd")
        if bid is not None:
            result["bid"] = self._r(bid, "volume_usd")
        if ask is not None:
            result["ask"] = self._r(ask, "volume_usd")

        imb = ob.get("imbalance")
        if imb is not None:
            result["imb"] = self._r(imb, "ratio")

        pressure = ob.get("pressure")
        if pressure is not None:
            result["prs"] = self._r(pressure, "ratio")

        # Depth levels
        ob_depth = inner.get("order_book_depth") or {}
        if ob_depth:
            depth: Dict[str, Any] = {}
            for level in ["L1", "L5", "L10", "L25"]:
                lvl = ob_depth.get(level)
                if isinstance(lvl, dict):
                    depth[level] = {
                        "b": self._r(lvl.get("bids"), "volume_usd"),
                        "a": self._r(lvl.get("asks"), "volume_usd"),
                        "i": self._r(lvl.get("imbalance"), "ratio"),
                    }
            if depth:
                result["dep"] = depth

            ratio = ob_depth.get("total_depth_ratio")
            if ratio is not None:
                result["dr"] = self._r(ratio, "ratio")

        # Spread analysis
        spread_a = inner.get("spread_analysis") or {}
        spr_bps = spread_a.get("current_spread_bps")
        if spr_bps is not None:
            result["bps"] = self._r(spr_bps, "rate")

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
                            "b": self._r(v.get("buy"), "percent"),
                            "s": self._r(v.get("sell"), "percent"),
                        }
                if imp:
                    result["imp"] = imp

            liq_score = impact.get("liquidity_score")
            if liq_score is not None:
                result["liq"] = self._r(liq_score, "score")

            exec_q = impact.get("execution_quality")
            if exec_q is not None:
                result["exq"] = exec_q

        return result

    def _build_flow(self, inner: Dict[str, Any]) -> Dict[str, Any]:
        flow = inner.get("fluxo_continuo") or {}
        if not flow:
            return {}

        of = flow.get("order_flow") or {}
        absorb = (
            (flow.get("absorption_analysis") or {}).get(
                "current_absorption"
            )
            or {}
        )
        sectors = flow.get("sector_flow") or {}
        part = flow.get("participant_analysis") or {}

        result: Dict[str, Any] = {}

        cvd = flow.get("cvd")
        if cvd is not None:
            result["cvd"] = self._r(cvd, "volume_btc")

        fi = of.get("flow_imbalance")
        if fi is not None:
            result["imb"] = self._r(fi, "ratio")

        agr_b = of.get("aggressive_buy_pct")
        agr_s = of.get("aggressive_sell_pct")
        if agr_b is not None:
            result["ab"] = self._r(agr_b, "percent")
        if agr_s is not None:
            result["as"] = self._r(agr_s, "percent")

        # Net flows multi-window
        net: Dict[str, Any] = {}
        for window in ["1m", "5m", "15m"]:
            nf = of.get(f"net_flow_{window}")
            if nf is not None:
                net[window] = self._r(nf, "volume_usd")
        if net:
            result["net"] = net

        # Absorções
        abs_dict: Dict[str, Any] = {}
        for window in ["1m", "5m", "15m"]:
            a = of.get(f"absorcao_{window}")
            if a is not None:
                abs_dict[window] = a
        if abs_dict:
            result["abs"] = abs_dict

        # Volumes USD/BTC
        buy_usd = of.get("buy_volume")
        sell_usd = of.get("sell_volume")
        if buy_usd is not None:
            result["bv"] = self._r(buy_usd, "volume_usd")
        if sell_usd is not None:
            result["sv"] = self._r(sell_usd, "volume_usd")

        buy_btc = of.get("buy_volume_btc")
        sell_btc = of.get("sell_volume_btc")
        if buy_btc is not None:
            result["bb"] = self._r(buy_btc, "volume_btc")
        if sell_btc is not None:
            result["sb"] = self._r(sell_btc, "volume_btc")

        # Absorção detalhada
        if absorb:
            result["absr"] = {
                "i": self._r(absorb.get("index"), "ratio"),
                "c": absorb.get("classification"),
                "l": absorb.get("label"),
                "bs": absorb.get("buyer_strength"),
                "se": absorb.get("seller_exhaustion"),
                "cp": self._r(
                    absorb.get("continuation_probability"), "ratio"
                ),
            }

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
                        entry[k] = self._r(v, "volume_btc")
                    else:
                        entry[k] = v

                if p_data:
                    vp = p_data.get("volume_pct")
                    if vp is not None:
                        entry["vp"] = self._r(vp, "percent")
                    d = p_data.get("direction")
                    if d is not None:
                        entry["dir"] = d
                    sent = p_data.get("sentiment")
                    if sent is not None:
                        entry["sent"] = sent
                    sc = p_data.get("composite_score")
                    if sc is not None:
                        entry["sc"] = self._r(sc, "score")

                if entry:
                    merged[sector] = entry

            if merged:
                result["sec"] = merged

        # Clusters de liquidez (comprimido)
        heatmap = flow.get("liquidity_heatmap") or {}
        clusters = heatmap.get("clusters") or []
        if clusters:
            result["clust"] = [
                {
                    "c": self._r(c.get("center"), "price"),
                    "lo": self._r(c.get("low"), "price"),
                    "hi": self._r(c.get("high"), "price"),
                    "v": self._r(c.get("total_volume"), "volume_btc"),
                    "imb": self._r(c.get("imbalance_ratio"), "ratio"),
                    "n": c.get("trades_count"),
                    "dur": c.get("cluster_duration_ms"),
                }
                for c in clusters[:5]  # máximo 5 clusters
            ]

            supports = heatmap.get("supports") or []
            if supports:
                result["sup"] = [
                    self._r(s, "price") for s in supports[:5]
                ]

        return result

    def _build_timeframes(self, multi_tf: Dict[str, Any]) -> Dict[str, Any]:
        """Comprime multi-TF removendo preco_atual duplicado."""
        if not multi_tf:
            return {}

        result: Dict[str, Any] = {}
        for tf_key, data in multi_tf.items():
            if not isinstance(data, dict):
                continue
            # NÃO inclui preco_atual (já está em price.c)
            entry: Dict[str, Any] = {}

            trend = data.get("tendencia")
            if trend is not None:
                entry["tr"] = trend

            regime = data.get("regime")
            if regime is not None:
                entry["rg"] = regime

            ema = data.get("mme_21")
            if ema is not None:
                entry["ema"] = self._r(ema, "price")

            atr = data.get("atr")
            if atr is not None:
                entry["atr"] = self._r(atr, "price")

            rsi_s = data.get("rsi_short")
            if rsi_s is not None:
                entry["rs"] = self._r(rsi_s, "indicator")

            rsi_l = data.get("rsi_long")
            if rsi_l is not None:
                entry["rl"] = self._r(rsi_l, "indicator")

            macd = data.get("macd")
            if macd is not None:
                entry["m"] = self._r(macd, "indicator")

            macd_sig = data.get("macd_signal")
            if macd_sig is not None:
                entry["ms"] = self._r(macd_sig, "indicator")

            adx = data.get("adx")
            if adx is not None:
                entry["ax"] = self._r(adx, "indicator")

            rvol = data.get("realized_vol")
            if rvol is not None:
                entry["rv"] = self._r(rvol, "rate")

            if entry:
                result[tf_key] = entry

        return result

    def _build_vp_with_cache(
        self, vp: Dict[str, Any], current_price: float
    ) -> Dict[str, Any]:
        """Volume Profile com cache para semanal/mensal."""
        result: Dict[str, Any] = {}

        # Daily - sempre enviar (muda constantemente)
        daily = vp.get("daily") or {}
        if daily and daily.get("status") == "success":
            d_entry: Dict[str, Any] = {
                "poc": daily.get("poc"),
                "vah": daily.get("vah"),
                "val": daily.get("val"),
            }
            hvns = daily.get("hvns") or []
            if hvns and current_price:
                sorted_h = sorted(
                    hvns, key=lambda x: abs(x - current_price)
                )
                d_entry["hvn"] = sorted_h[: self.MAX_HVNS_NEARBY]
            result["d"] = d_entry

        # Weekly - com cache
        weekly = vp.get("weekly") or {}
        if weekly and weekly.get("status") == "success":
            w_data = {
                "poc": weekly.get("poc"),
                "vah": weekly.get("vah"),
                "val": weekly.get("val"),
            }
            changed, _ = (
                self._section_cache.check_and_update("vp_weekly", w_data)
                if self._section_cache
                else (True, "")
            )
            if changed:
                w_entry = dict(w_data)
                hvns = weekly.get("hvns") or []
                if hvns and current_price:
                    sorted_h = sorted(
                        hvns, key=lambda x: abs(x - current_price)
                    )
                    w_entry["hvn"] = sorted_h[: self.MAX_HVNS_NEARBY]
                lvns = weekly.get("lvns") or []
                if lvns and current_price:
                    sorted_l = sorted(
                        lvns, key=lambda x: abs(x - current_price)
                    )
                    w_entry["lvn"] = sorted_l[: self.MAX_LVNS_NEARBY]
                result["w"] = w_entry

        # Monthly - com cache (muda raramente)
        monthly = vp.get("monthly") or {}
        if monthly and monthly.get("status") == "success":
            m_data = {
                "poc": monthly.get("poc"),
                "vah": monthly.get("vah"),
                "val": monthly.get("val"),
            }
            changed, _ = (
                self._section_cache.check_and_update("vp_monthly", m_data)
                if self._section_cache
                else (True, "")
            )
            if changed:
                result["m"] = m_data

        return result

    def _build_derivatives(self, deriv: Dict[str, Any]) -> Dict[str, Any]:
        if not deriv:
            return {}

        KEY_MAP = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
        result: Dict[str, Any] = {}

        for full_key, data in deriv.items():
            if not isinstance(data, dict):
                continue
            short = KEY_MAP.get(full_key, full_key[:3])
            entry: Dict[str, Any] = {}

            fr = data.get("funding_rate_percent")
            if fr is not None:
                entry["fr"] = self._r(fr, "rate")
            oi = data.get("open_interest")
            if oi is not None:
                entry["oi"] = self._r(oi, "volume_btc")
            oi_usd = data.get("open_interest_usd")
            if oi_usd is not None:
                entry["oi$"] = self._r(oi_usd, "volume_usd")
            lsr = data.get("long_short_ratio")
            if lsr is not None:
                entry["lsr"] = self._r(lsr, "ratio")

            if entry:
                result[short] = entry

        return result

    def _build_macro(
        self, ctx: Dict[str, Any], env: Dict[str, Any]
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        for key, short in [
            ("trading_session", "ses"),
            ("session_phase", "ph"),
            ("market_hours_type", "hrs"),
        ]:
            v = ctx.get(key)
            if v is not None:
                result[short] = v

        day = ctx.get("day_of_week")
        if day is not None:
            result["day"] = day

        close = ctx.get("time_to_session_close")
        if close is not None:
            result["cls"] = close

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

    def _build_cross_asset(
        self, cross: Dict[str, Any], env: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not cross and not env:
            return {}

        result: Dict[str, Any] = {}

        # Correlações BTC
        be_7 = cross.get("btc_eth_corr_7d")
        be_30 = cross.get("btc_eth_corr_30d")
        if be_7 is not None or be_30 is not None:
            be: Dict[str, Any] = {}
            if be_7 is not None:
                be["7"] = self._r(be_7, "correlation")
            if be_30 is not None:
                be["30"] = self._r(be_30, "correlation")
            result["be"] = be

        bd_30 = cross.get("btc_dxy_corr_30d")
        bd_90 = cross.get("btc_dxy_corr_90d")
        if bd_30 is not None or bd_90 is not None:
            bd: Dict[str, Any] = {}
            if bd_30 is not None:
                bd["30"] = self._r(bd_30, "correlation")
            if bd_90 is not None:
                bd["90"] = self._r(bd_90, "correlation")
            result["bd"] = bd

        bn_30 = cross.get("btc_ndx_corr_30d")
        if bn_30 is not None:
            result["bn30"] = self._r(bn_30, "correlation")

        # Correlações de environment
        for env_key, short in [
            ("correlation_spy", "spy"),
            ("correlation_dxy", "dxy"),
            ("correlation_gold", "gld"),
        ]:
            v = env.get(env_key)
            if v is not None:
                if "cor" not in result:
                    result["cor"] = {}
                result["cor"][short] = self._r(v, "correlation")

        # DXY
        dr5 = cross.get("dxy_return_5d")
        dr20 = cross.get("dxy_return_20d")
        if dr5 is not None or dr20 is not None:
            dr: Dict[str, Any] = {}
            if dr5 is not None:
                dr["5"] = self._r(dr5, "percent")
            if dr20 is not None:
                dr["20"] = self._r(dr20, "percent")
            result["dr"] = dr

        dm = cross.get("dxy_momentum")
        if dm is not None:
            result["dm"] = self._r(dm, "indicator")

        regime = cross.get("correlation_regime")
        if regime is not None:
            result["rg"] = regime

        return result

    def _build_onchain(self, oc: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        MAPPING = [
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
                result[dst] = self._r(v, prec) if prec else v

        funding = oc.get("funding_rates") or {}
        if funding:
            result["fr"] = {
                k[:3].lower(): self._r(v, "rate")
                for k, v in funding.items()
            }

        return result

    def _build_options(self, opt: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        MAPPING = [
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
                result[dst] = self._r(v, prec) if prec else v

        return result

    def _build_ml_features(self, ml: Dict[str, Any]) -> Dict[str, Any]:
        pf = ml.get("price_features") or {}
        vf = ml.get("volume_features") or {}
        ms = ml.get("microstructure") or {}

        result: Dict[str, Any] = {}

        # Manter valores científicos como estão (são muito pequenos)
        ret = pf.get("returns_15")
        if ret is not None:
            result["r15"] = ret  # mantém notação científica
        vol = pf.get("volatility_15")
        if vol is not None:
            result["v15"] = vol

        mom = pf.get("momentum_score")
        if mom is not None:
            result["mom"] = self._r(mom, "ratio")

        for src, dst in [
            ("volume_sma_ratio", "vsr"),
            ("volume_momentum", "vm"),
            ("buy_sell_pressure", "bsp"),
            ("liquidity_gradient", "lg"),
        ]:
            v = vf.get(src)
            if v is not None:
                result[dst] = self._r(v, "ratio")

        fi = ms.get("flow_imbalance")
        if fi is not None:
            result["fi"] = self._r(fi, "ratio")

        ts = ms.get("tick_rule_sum")
        if ts is not None:
            result["ts"] = ts

        return result

    def _build_quant(self, quant: Dict[str, Any]) -> Dict[str, Any]:
        """Comprime dados do modelo quantitativo."""
        result: Dict[str, Any] = {}

        pu = quant.get("model_probability_up")
        if pu is not None:
            result["pu"] = self._r(pu, "ratio")

        pd_ = quant.get("model_probability_down")
        if pd_ is not None:
            result["pd"] = self._r(pd_, "ratio")

        ab = quant.get("action_bias")
        if ab is not None:
            result["ab"] = ab

        cs = quant.get("confidence_score")
        if cs is not None:
            result["cs"] = self._r(cs, "ratio")

        fu = quant.get("features_used")
        tf = quant.get("total_features")
        if fu is not None and tf is not None:
            result["feat"] = f"{fu}/{tf}"

        return result

    # ================================================================
    # CACHE HELPER
    # ================================================================

    def _maybe_cache(
        self,
        section: str,
        builder: Any,
        raw_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Verifica cache antes de construir seção.
        Se não mudou, retorna None (será omitida do payload).
        """
        if not raw_data:
            return None

        if self._section_cache is not None:
            changed, _ = self._section_cache.check_and_update(
                section, raw_data
            )
            if not changed:
                return None  # Será listada em _cached

        return builder()

    # ================================================================
    # UTILIDADES
    # ================================================================

    def _r(self, value: Any, precision_type: Optional[str] = "default") -> Any:
        """Arredonda valor numérico conforme tipo de precisão."""
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            return value
        if precision_type is None:
            return value
        decimals = self.PRECISION.get(precision_type, self.PRECISION["default"])
        if decimals == 0:
            return int(round(value))
        return round(value, decimals)

    def _log_savings(
        self, original: Dict[str, Any], compressed: Dict[str, Any]
    ) -> None:
        """Loga a economia de tokens."""
        try:
            orig_str = json.dumps(original, default=str)
            comp_str = json.dumps(compressed, default=str)
            orig_chars = len(orig_str)
            comp_chars = len(comp_str)
            saving_pct = (
                (1 - comp_chars / orig_chars) * 100 if orig_chars > 0 else 0
            )

            self._stats["total_calls"] += 1
            self._stats["total_chars_original"] += orig_chars
            self._stats["total_chars_compressed"] += comp_chars

            # Estimativa de tokens (1 token ≈ 3.5 chars para JSON)
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
            logger.error(f"Erro ao calcular métricas de compressão: {e}")


# ================================================================
# SYSTEM PROMPT COMPACTO + DICIONÁRIO
# ================================================================

COMPRESSED_KEY_DICTIONARY = """Chaves comprimidas do JSON de dados:
s=symbol t=epoch_ms w=janela ev=evento ctx=contexto
price: c=close o=open h=high l=low vwap poc poc_v poc_p dw=dwell dw_s dw_l
vol: tot=total usd buy sell n=trades tps avg d={min,max,c} rev={min,max}
ob: mid spr=spread bid ask imb=imbalance prs=pressure dep=depth(b/a/i) dr=ratio bps imp liq exq
flow: cvd imb ab=agr_buy as=agr_sell net={1m,5m,15m} abs={1m,5m,15m} bv=buy_vol sv=sell_vol bb sb absr={i,c,l,bs,se,cp} sec=sectors clust sup
tf: tr=tendencia rg=regime ema=ema21 atr rs=rsi_short rl=rsi_long m=macd ms=macd_sig ax=adx rv=realized_vol
vp: d=daily w=weekly m=monthly poc vah val hvn lvn
deriv: fr=funding oi oi$=oi_usd lsr
macro: ses=session ph=phase hrs day cls=close_in vr td ms le rs
cross: be=btc_eth bd=btc_dxy bn30 cor={spy,dxy,gld} dr={5d,20d} dm=dxy_mom rg=regime
onchain: nf=netflow wh=whales mn=miner res=reserves aa=active_addr hr=hashrate df=difficulty sopr fr
options: pcr iv ivp ivr gex mp=max_pain vix skew
ml: r15 v15 mom vsr vm bsp lg fi ts
quant: pu=prob_up pd=prob_down ab=action_bias cs=confidence feat
_cached=secoes nao reenviadas pois nao mudaram"""

SYSTEM_PROMPT_COMPRESSED = (
    "Analista institucional de fluxo e microestrutura cripto. "
    "Horizonte: scalp 5-15min.\n\n"
    + COMPRESSED_KEY_DICTIONARY
    + "\n\n"
    "REGRAS:\n"
    "1. Responda APENAS em portugues BR, sem ingles\n"
    "2. Use quant.ab (action_bias) como base. So contrarie com evidencia MUITO forte\n"
    "3. Em duvida, action=wait\n"
    "4. _cached = secoes omitidas pois nao mudaram (considere estáveis)\n"
    "5. Responda SOMENTE JSON:\n"
    '{"sentiment":"bullish|bearish|neutral","confidence":0.0-1.0,'
    '"action":"buy|sell|hold|flat|wait|avoid",'
    '"rationale":"texto curto","entry_zone":"preco|null",'
    '"invalidation_zone":"preco|null","region_type":"tipo|null"}'
)