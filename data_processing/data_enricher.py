# data_enricher.py
"""
Enriquecimento de dados para o sistema:
- Gera price_targets a partir de:
  - Fibonacci (pattern_recognition.fibonacci_levels)
  - HVNs (historical_vp.*.volume_nodes.hvn_nodes)
  - Liquidity clusters (liquidity_heatmap.clusters)
  - NOVO: ATR-based targets (calculado do multi_tf)
  - NOVO: Volume Profile POC/VAH/VAL como targets primários
- Adiciona métricas mock de:
  - options_metrics
  - onchain_metrics
- Calcula adaptive_thresholds pela realized_vol de multi_tf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import asyncio

# Importar métricas do sistema
try:
    from monitoring.metrics_collector import (
        record_enrich_error,
        track_latency,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

    def record_enrich_error(event_type: str, error: Optional[str] = None) -> None:
        pass

    def track_latency(metric_name: str):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

# Importar fetchers reais
try:
    from fetchers.onchain_fetcher import OnchainFetcher
    _ONCHAIN_FETCHER = OnchainFetcher()
    _ONCHAIN_OK = True
except ImportError:
    _ONCHAIN_OK = False
    _ONCHAIN_FETCHER = None
    logger.warning("onchain_fetcher indisponível, usando dados parciais")

try:
    from fetchers.funding_aggregator import FundingAggregator
    _FUNDING_AGG = FundingAggregator()
    _FUNDING_OK = True
except ImportError:
    _FUNDING_OK = False
    _FUNDING_AGG = None
    logger.warning("funding_aggregator indisponível, usando dados parciais")


@dataclass
class PriceTarget:
    level: float
    confidence: float
    source: str
    weight: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": round(self.level, 2),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "weight": round(self.weight, 3),
            "timestamp": self.timestamp.isoformat(),
        }


# FIX 4.4: Campos de onchain que têm dados reais (não dependem de API paga)
_REAL_ONCHAIN_FIELDS = frozenset({
    "difficulty", "active_addresses", "mempool_size",
    "mempool_vsize_mb", "fees_fastest_sat_vb", "fees_half_hour_sat_vb",
    "fees_hour_sat_vb", "total_btc_sent_24h", "trade_volume_btc_24h",
    "difficulty_adjustment", "data_source", "is_real_data", "status",
})


def _filter_real_onchain(onchain: dict) -> dict:
    """Remove campos zerados por falta de API paga (hash_rate, exchange_netflow, etc.)."""
    if not onchain:
        return onchain
    return {k: v for k, v in onchain.items() if k in _REAL_ONCHAIN_FIELDS or v not in (0, None)}


class DataEnricher:
    """
    Enriquecedor de dados baseado APENAS no raw_event atual.

    v2: Melhorias no price_targets:
    - Extrai POC/VAH/VAL do historical_vp como targets primários (SEMPRE disponíveis)
    - Gera targets baseados em ATR quando multi_tf disponível
    - Fallback matemático robusto baseado em volatilidade
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbol = config.get("SYMBOL", "BTCUSDT")

        self.abs_base = config.get("ABSORPTION_THRESHOLD_BASE", 0.15)
        self.flow_base = config.get("FLOW_THRESHOLD_BASE", 0.10)
        self.min_vol_factor = config.get("MIN_VOL_FACTOR", 0.5)
        self.max_vol_factor = config.get("MAX_VOL_FACTOR", 2.0)
        self._cached_historical_vp = None

    def enrich_from_raw_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        inner = raw_event.get("raw_event") if isinstance(raw_event.get("raw_event"), dict) else {}
        assert isinstance(inner, dict)

        price = (
            raw_event.get("preco_fechamento")
            or inner.get("preco_fechamento")
            or (raw_event.get("ohlc") or {}).get("close")
            or (inner.get("ohlc") or {}).get("close")
        ) or None
        volume = raw_event.get("volume_total", inner.get("volume_total", 0.0))
        symbol = raw_event.get("symbol", inner.get("symbol", self.symbol))

        if price is None:
            logger.debug("raw_event sem preco_fechamento nem ohlc.close, não será enriquecido")
            if METRICS_AVAILABLE:
                record_enrich_error("enrich_from_raw_event")
            return {}

        timestamp = (
            raw_event.get("timestamp_utc")
            or raw_event.get("timestamp")
            or inner.get("timestamp_utc")
            or inner.get("timestamp")
        )

        try:
            price_targets = self._build_price_targets(raw_event, current_price=price)
            price_targets_dicts = [pt.to_dict() for pt in price_targets]
        except Exception as e:
            logger.error(f"[DataEnricher] Erro ao gerar price_targets: {e}", exc_info=True)
            if METRICS_AVAILABLE:
                record_enrich_error("price_targets")
            price_targets_dicts = []

        options_metrics = self._build_options_metrics()
        onchain_metrics = _filter_real_onchain(self._build_onchain_metrics())
        adaptive_thresholds = self._build_adaptive_thresholds(raw_event)

        result = {
            "symbol": symbol,
            "timestamp": timestamp,
            "price": price,
            "volume": volume,
            "price_targets": price_targets_dicts,
            "onchain_metrics": onchain_metrics,
            "adaptive_thresholds": adaptive_thresholds,
        }
        # FIX 4.1: Só incluir options_metrics se tiver dados reais
        if options_metrics and options_metrics.get("is_real_data"):
            result["options_metrics"] = options_metrics
        return result

    def enrich_event_with_advanced_analysis(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula advanced_analysis usando dados COMPLETOS do raw_event externo.
        """
        outer_raw = {}
        inner_raw = {}
        try:
            logger.debug("[DEBUG] enrich_event_with_advanced_analysis CHAMADA!")

            outer_raw = event.get("raw_event", {})
            inner_raw = outer_raw.get("raw_event", {})

            price = (
                inner_raw.get("preco_fechamento")
                or outer_raw.get("preco_fechamento")
                or outer_raw.get("ohlc", {}).get("close")
            )
            if price is None:
                price = 45000.0
                logger.warning(f"preco_fechamento ausente, usando valor padrão: {price}")
            
            volume = inner_raw.get("volume_total", outer_raw.get("volume_total", 0))

            multi_tf = outer_raw.get("multi_tf", {})
            timestamp_utc = outer_raw.get("timestamp_utc")

            # Resolver historical_vp de múltiplas fontes
            historical_vp_src = (
                outer_raw.get("historical_vp")
                or outer_raw.get("volume_profile_historical")
                or outer_raw.get("contextual", {}).get("historical_vp")
                or outer_raw.get("enriched", {}).get("contextual", {}).get("historical_vp")
                or getattr(self, '_last_historical_vp', None)
                or getattr(self, '_cached_historical_vp', None)
            )

            if historical_vp_src and historical_vp_src != getattr(self, '_cached_historical_vp', None):
                self._cached_historical_vp = historical_vp_src

            if historical_vp_src:
                self._last_historical_vp = historical_vp_src
                if "historical_vp" not in outer_raw:
                    outer_raw["historical_vp"] = historical_vp_src

            # Construir adaptive_thresholds PRIMEIRO para obter volatilidade real
            adaptive_thresholds = self._build_adaptive_thresholds(outer_raw)
            # Usar volatilidade real do adaptive_thresholds (não hardcoded 0.01)
            current_volatility = adaptive_thresholds.get("current_volatility", 0.01)

            # ── NOVO: Construir price_targets com método melhorado ──
            price_targets = self._build_price_targets_enhanced(
                outer_raw, price, current_volatility, historical_vp_src
            )
            # FIX 4.1: Filtrar options_metrics fake (is_real_data=False)
            _options = inner_raw.get("advanced_analysis", {}).get(
                "options_metrics", self._build_options_metrics()
            )
            # FIX 4.4: Filtrar onchain_metrics — manter apenas campos com dados reais
            _onchain = inner_raw.get("advanced_analysis", {}).get(
                "onchain_metrics", self._build_onchain_metrics()
            )
            _onchain = _filter_real_onchain(_onchain)

            advanced_analysis = {
                "symbol": outer_raw.get("symbol", inner_raw.get("symbol", "BTCUSDT")),
                "price": price,
                "volume": volume,
                "timestamp": timestamp_utc,
                # FIX 3.6: price_targets REMOVIDO — fonte canônica é event["price_targets"] (root)
                "adaptive_thresholds": adaptive_thresholds,
                "onchain_metrics": _onchain,
            }
            # FIX 4.1: Só incluir options_metrics se tiver dados reais
            if _options and _options.get("is_real_data"):
                advanced_analysis["options_metrics"] = _options

            # Salvar no raw_event
            if inner_raw:
                inner_raw["advanced_analysis"] = advanced_analysis
            else:
                outer_raw["advanced_analysis"] = advanced_analysis

            return event

        except Exception as e:
            logger.error(
                "[DataEnricher] ERRO CRÍTICO em enrich_event_with_advanced_analysis: %s",
                e, exc_info=True
            )
            return {}

    # ──────────────────────────────────────────────────────────────────────
    #   NOVO: Price targets melhorado (nunca retorna vazio)
    # ──────────────────────────────────────────────────────────────────────

    def _build_price_targets_enhanced(
        self,
        raw_event: Dict[str, Any],
        current_price: float,
        volatility: float,
        historical_vp: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Constrói price_targets garantindo que NUNCA retorna vazio.
        
        Prioridade:
        1. VP levels (POC/VAH/VAL) - sempre disponíveis após 1ª janela
        2. Fibonacci levels
        3. HVN nodes
        4. Liquidity clusters
        5. ATR-based targets (do multi_tf)
        6. Fallback matemático (volatilidade)
        """
        targets: List[PriceTarget] = []

        # 1. VP Levels (POC/VAH/VAL) - fonte PRIMÁRIA
        if historical_vp:
            targets.extend(self._extract_vp_levels(historical_vp, current_price))

        # 2. Fibonacci
        pr = raw_event.get("pattern_recognition", {})
        fib_levels = pr.get("fibonacci_levels") or {}
        targets.extend(self._extract_fibonacci_targets(fib_levels, current_price))

        # 3. HVNs
        hvp = raw_event.get("historical_vp", {})
        targets.extend(self._extract_hvn_targets(hvp))

        # 4. Liquidity clusters
        lh = raw_event.get("liquidity_heatmap")
        if not lh:
            lh = raw_event.get("fluxo_continuo", {}).get("liquidity_heatmap")
        if lh:
            targets.extend(self._extract_liquidity_cluster_targets(lh))

        # 5. ATR-based targets
        multi_tf = raw_event.get("multi_tf", {})
        targets.extend(self._extract_atr_targets(multi_tf, current_price))

        # Consolidar
        consolidated = self._consolidate_price_targets(targets)

        # Validação: filtrar targets dentro de 0.1% do preço atual (micro-range)
        # e garantir que existam targets acima E abaixo do preço
        min_distance_pct = 0.001  # 0.1% mínimo de distância
        filtered = [
            pt for pt in consolidated
            if abs(pt.level - current_price) / current_price > min_distance_pct
        ]
        has_above = any(pt.level > current_price for pt in filtered)
        has_below = any(pt.level < current_price for pt in filtered)

        # Se faltam targets em uma direção, usar ATR ou volatilidade para gerar
        if filtered and (not has_above or not has_below):
            vol_offset = max(0.005, volatility or 0.01) * current_price
            if not has_above:
                filtered.append(PriceTarget(
                    level=round(current_price + vol_offset, 2),
                    confidence=0.3, source="validated_r1", weight=0.1,
                ))
            if not has_below:
                filtered.append(PriceTarget(
                    level=round(current_price - vol_offset, 2),
                    confidence=0.3, source="validated_s1", weight=0.1,
                ))

        result = [pt.to_dict() for pt in (filtered or consolidated)]

        # 6. Fallback se ainda vazio
        if not result:
            vol_factor = max(0.005, volatility or 0.01)
            result = [
                {"level": round(current_price * (1 + vol_factor), 2), "confidence": 0.3,
                 "source": "fallback_r1", "weight": 0.1},
                {"level": round(current_price * (1 + vol_factor * 2), 2), "confidence": 0.2,
                 "source": "fallback_r2", "weight": 0.05},
                {"level": round(current_price * (1 - vol_factor), 2), "confidence": 0.3,
                 "source": "fallback_s1", "weight": 0.1},
                {"level": round(current_price * (1 - vol_factor * 2), 2), "confidence": 0.2,
                 "source": "fallback_s2", "weight": 0.05},
                {"level": round(current_price, 2), "confidence": 0.4,
                 "source": "fallback_poc", "weight": 0.15},
            ]
            logger.info(f"[PT] Usando fallback matemático (vol={vol_factor:.4f})")
        else:
            logger.info(f"[PT] ✅ {len(result)} price targets gerados")

        return result

    def _extract_vp_levels(
        self, historical_vp: Dict[str, Any], current_price: float
    ) -> List[PriceTarget]:
        """Extrai POC/VAH/VAL de todos os períodos do volume profile."""
        targets: List[PriceTarget] = []
        
        # Peso por timeframe (daily mais importante que weekly/monthly)
        tf_weights = {"daily": 0.35, "weekly": 0.25, "monthly": 0.20}
        
        for period, weight in tf_weights.items():
            vp = historical_vp.get(period, {})
            if not isinstance(vp, dict):
                continue
            
            poc = vp.get("poc")
            vah = vp.get("vah")
            val = vp.get("val")
            
            if isinstance(poc, (int, float)) and poc > 0:
                distance = abs(poc - current_price) / current_price
                conf = max(0.4, min(0.95, 1.0 - distance * 5))
                targets.append(PriceTarget(
                    level=poc, confidence=conf,
                    source=f"poc_{period}", weight=weight
                ))
            
            if isinstance(vah, (int, float)) and vah > 0:
                distance = abs(vah - current_price) / current_price
                conf = max(0.3, min(0.9, 1.0 - distance * 5))
                targets.append(PriceTarget(
                    level=vah, confidence=conf,
                    source=f"vah_{period}", weight=weight * 0.9
                ))
            
            if isinstance(val, (int, float)) and val > 0:
                distance = abs(val - current_price) / current_price
                conf = max(0.3, min(0.9, 1.0 - distance * 5))
                targets.append(PriceTarget(
                    level=val, confidence=conf,
                    source=f"val_{period}", weight=weight * 0.9
                ))
        
        return targets

    def _extract_atr_targets(
        self, multi_tf: Dict[str, Any], current_price: float
    ) -> List[PriceTarget]:
        """Gera targets baseados em ATR/volatilidade do multi_tf."""
        targets: List[PriceTarget] = []
        
        if not multi_tf:
            return targets
        
        # Usar ATR do 1h ou 4h se disponível
        for tf in ["1h", "4h", "1d"]:
            tf_data = multi_tf.get(tf, {})
            atr = tf_data.get("atr") or tf_data.get("realized_vol")
            if not isinstance(atr, (int, float)) or atr <= 0:
                continue
            
            # Para realized_vol (percentual), converter para preço
            if atr < 1:
                atr_price = current_price * atr
            else:
                atr_price = atr
            
            targets.append(PriceTarget(
                level=current_price + atr_price,
                confidence=0.35,
                source=f"atr_{tf}_r1",
                weight=0.15,
            ))
            targets.append(PriceTarget(
                level=current_price - atr_price,
                confidence=0.35,
                source=f"atr_{tf}_s1",
                weight=0.15,
            ))
            targets.append(PriceTarget(
                level=current_price + atr_price * 1.5,
                confidence=0.25,
                source=f"atr_{tf}_r2",
                weight=0.10,
            ))
            targets.append(PriceTarget(
                level=current_price - atr_price * 1.5,
                confidence=0.25,
                source=f"atr_{tf}_s2",
                weight=0.10,
            ))
            break  # Usar apenas o primeiro TF disponível
        
        return targets

    # ──────────────────────────────────────────────────────────────────────
    #   PRICE TARGETS (originais mantidos)
    # ──────────────────────────────────────────────────────────────────────

    def _build_price_targets(
        self, raw_event: Dict[str, Any], current_price: float
    ) -> List[PriceTarget]:
        targets: List[PriceTarget] = []

        pr = raw_event.get("pattern_recognition", {})
        fib_levels = pr.get("fibonacci_levels") or {}
        targets.extend(self._extract_fibonacci_targets(fib_levels, current_price))

        historical_vp = raw_event.get("historical_vp", {})
        targets.extend(self._extract_hvn_targets(historical_vp))

        lh = raw_event.get("liquidity_heatmap")
        if not lh:
            lh = raw_event.get("fluxo_continuo", {}).get("liquidity_heatmap")
        if lh:
            targets.extend(self._extract_liquidity_cluster_targets(lh))

        return self._consolidate_price_targets(targets)

    def _extract_fibonacci_targets(
        self, fib: Dict[str, Any], current_price: float
    ) -> List[PriceTarget]:
        targets: List[PriceTarget] = []
        if not isinstance(fib, dict):
            return targets

        high = fib.get("high")
        low = fib.get("low")

        for key, value in fib.items():
            if key in ("high", "low"):
                continue
            if not isinstance(value, (int, float)):
                continue

            distance_pct = abs(value - current_price) / current_price
            confidence = max(0.2, 1.0 - distance_pct * 4)
            confidence = min(0.95, confidence)

            targets.append(PriceTarget(
                level=value, confidence=confidence,
                source=f"fib_{key.replace('.', '_')}", weight=0.15,
            ))

        if isinstance(high, (int, float)):
            targets.append(PriceTarget(
                level=high, confidence=0.3,
                source="fib_swing_high", weight=0.1,
            ))
        if isinstance(low, (int, float)):
            targets.append(PriceTarget(
                level=low, confidence=0.3,
                source="fib_swing_low", weight=0.1,
            ))

        return targets

    def _extract_hvn_targets(self, historical_vp: Dict[str, Any]) -> List[PriceTarget]:
        targets: List[PriceTarget] = []

        for timeframe, vp in historical_vp.items():
            if not isinstance(vp, dict):
                continue
            vol_nodes = vp.get("volume_nodes", {})
            hvn_str = vol_nodes.get("hvn_nodes")
            if not hvn_str or not isinstance(hvn_str, str):
                continue

            entries = [
                e.strip()
                for e in hvn_str.split(";")
                if e.strip() and not e.strip().startswith("['")
            ]
            parsed = []
            for e in entries:
                try:
                    parts = e.split("|")
                    if len(parts) < 2:
                        continue
                    price = float(parts[0])
                    score = float(parts[1])
                    parsed.append((price, score))
                except Exception:
                    continue

            if not parsed:
                continue

            max_score = max(s for _, s in parsed) or 1.0
            for price, score in parsed:
                vol_ratio = score / max_score
                targets.append(PriceTarget(
                    level=price,
                    confidence=min(0.9, 0.5 + 0.5 * vol_ratio),
                    source=f"hvn_{timeframe}",
                    weight=0.25,
                ))

        return targets

    def _extract_liquidity_cluster_targets(
        self, heatmap: Dict[str, Any]
    ) -> List[PriceTarget]:
        targets: List[PriceTarget] = []
        clusters = heatmap.get("clusters", [])
        if not isinstance(clusters, list) or not clusters:
            return targets

        max_vol = 0.0
        for c in clusters:
            vol = c.get("total_volume")
            if isinstance(vol, (int, float)):
                max_vol = max(max_vol, vol)
        if max_vol <= 0:
            max_vol = 1.0

        for c in clusters:
            center = c.get("center")
            vol = c.get("total_volume")
            if not isinstance(center, (int, float)) or not isinstance(vol, (int, float)):
                continue
            vol_ratio = vol / max_vol
            targets.append(PriceTarget(
                level=center,
                confidence=min(0.9, 0.4 + 0.6 * vol_ratio),
                source="liquidity_cluster",
                weight=0.2,
            ))

        return targets

    # ──────────────────────────────────────────────────────────────────────
    #   CONSOLIDAÇÃO
    # ──────────────────────────────────────────────────────────────────────

    def _consolidate_price_targets(
        self, targets: List[PriceTarget], consolidation_pct: float = 0.005
    ) -> List[PriceTarget]:
        if not targets:
            return []

        targets = sorted(targets, key=lambda t: t.level)
        consolidated: List[PriceTarget] = []
        cluster: List[PriceTarget] = []

        for t in targets:
            if not cluster:
                cluster.append(t)
                continue

            avg_price = sum(x.level for x in cluster) / len(cluster)
            if abs(t.level - avg_price) / avg_price <= consolidation_pct:
                cluster.append(t)
            else:
                consolidated.append(self._merge_cluster(cluster))
                cluster = [t]

        if cluster:
            consolidated.append(self._merge_cluster(cluster))

        consolidated.sort(key=lambda x: x.confidence * x.weight, reverse=True)
        return consolidated[:10]

    def _merge_cluster(self, cluster: List[PriceTarget]) -> PriceTarget:
        if len(cluster) == 1:
            return cluster[0]

        total_weight = sum(c.weight * c.confidence for c in cluster) or 1.0
        merged_price = sum(
            c.level * c.weight * c.confidence for c in cluster
        ) / total_weight
        avg_conf = sum(c.confidence for c in cluster) / len(cluster)
        avg_weight = sum(c.weight for c in cluster) / len(cluster)

        sources = {c.source for c in cluster}
        source_str = "confluence_" + "_".join(sorted(list(sources))[:2])

        return PriceTarget(
            level=merged_price,
            confidence=min(0.99, avg_conf * 1.2),
            source=source_str,
            weight=min(0.5, avg_weight * 1.5),
        )

    # ──────────────────────────────────────────────────────────────────────
    #   OPTIONS (requer API paga)
    # ──────────────────────────────────────────────────────────────────────

    def _build_options_metrics(self) -> Dict[str, Any]:
        return {
            "status": "requires_paid_api",
            "provider_needed": "Deribit/Laevitas/TradingView",
            "put_call_ratio": None,
            "implied_volatility": None,
            "iv_percentile": None,
            "iv_rank": None,
            "gamma_exposure": None,
            "max_pain": None,
            "skew": None,
            "is_real_data": False,
        }

    # ──────────────────────────────────────────────────────────────────────
    #   ON-CHAIN
    # ──────────────────────────────────────────────────────────────────────

    def _build_onchain_metrics(self) -> Dict[str, Any]:
        if _ONCHAIN_OK and _ONCHAIN_FETCHER is not None:
            try:
                try:
                    asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, _ONCHAIN_FETCHER.fetch_all())
                        return future.result(timeout=15)
                except RuntimeError:
                    return asyncio.run(_ONCHAIN_FETCHER.fetch_all())
            except Exception as e:
                logger.warning(f"Falha ao buscar on-chain real: {e}")

        return {
            "hash_rate": 0,
            "difficulty": 0,
            "mempool_size": 0,
            "is_real_data": False,
            "status": "fetcher_unavailable",
        }

    # ──────────────────────────────────────────────────────────────────────
    #   ADAPTIVE THRESHOLDS
    # ──────────────────────────────────────────────────────────────────────

    def _build_adaptive_thresholds(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        mtf = raw_event.get("multi_tf", {}) or {}
        realized_vol = None

        for tf in ["1d", "4h", "1h", "15m"]:
            d = mtf.get(tf) or {}
            v = d.get("realized_vol")
            if isinstance(v, (int, float)) and v > 0:
                realized_vol = v
                break

        if realized_vol is None:
            # CORREÇÃO: fallback 0.03 (3%) é mais realista para BTC
            # Antes: 0.01 causava thresholds desproporcionais e
            # price targets errados no raw_event
            realized_vol = 0.03

        vol_factor = 0.5 / realized_vol
        vol_factor = max(self.min_vol_factor, min(self.max_vol_factor, vol_factor))

        return {
            "current_volatility": realized_vol,
            "volatility_factor": vol_factor,
            "absorption_threshold": self.abs_base * vol_factor,
            "flow_threshold": self.flow_base * vol_factor,
        }