# data_enricher.py
"""
Enriquecimento de dados para o sistema:
- Gera price_targets a partir de:
  - Fibonacci (pattern_recognition.fibonacci_levels)
  - HVNs (historical_vp.*.volume_nodes.hvn_nodes)
  - Liquidity clusters (liquidity_heatmap.clusters)
- Adiciona métricas mock de:
  - options_metrics
  - onchain_metrics
- Calcula adaptive_thresholds pela realized_vol de multi_tf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Importar métricas do sistema
try:
    from metrics_collector import (
        record_enrich_error,
        record_event_processed,
        track_latency
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    def record_enrich_error(): pass
    def record_event_processed(event_type): pass
    def track_latency(operation_type):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()

logger = logging.getLogger(__name__)


@dataclass
class PriceTarget:
    level: float
    confidence: float  # 0-1
    source: str        # 'fib_...', 'hvn_daily', 'liquidity_cluster', etc.
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


class DataEnricher:
    """
    Enriquecedor de dados baseado APENAS no raw_event atual.
    Não depende de DataFrame histórico externo.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbol = config.get("SYMBOL", "BTCUSDT")

        # Parâmetros base para thresholds adaptativos
        self.abs_base = config.get("ABSORPTION_THRESHOLD_BASE", 0.15)
        self.flow_base = config.get("FLOW_THRESHOLD_BASE", 0.10)
        self.min_vol_factor = config.get("MIN_VOL_FACTOR", 0.5)
        self.max_vol_factor = config.get("MAX_VOL_FACTOR", 2.0)

    def enrich_from_raw_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        with track_latency("enrich_event"):
            # Suporta dois formatos:
            # A) preco_fechamento e volume_total no root
            # B) preco_fechamento e volume_total dentro de raw_event["raw_event"]
            inner = raw_event.get("raw_event") if isinstance(raw_event.get("raw_event"), dict) else {}

            price = raw_event.get("preco_fechamento", inner.get("preco_fechamento"))
            volume = raw_event.get("volume_total", inner.get("volume_total", 0.0))
            symbol = raw_event.get("symbol", inner.get("symbol", self.symbol))

            if price is None:
                logger.warning("raw_event sem preco_fechamento (root e inner), não será enriquecido")
                if METRICS_AVAILABLE:
                    record_enrich_error()
                return {}

            # timestamp vem do outer (porque inner não tem)
            timestamp = (
                raw_event.get("timestamp_utc")
                or raw_event.get("timestamp")
                or inner.get("timestamp_utc")
                or inner.get("timestamp")
            )

            # A partir daqui, use raw_event externo completo (ele tem historical_vp, multi_tf, liquidity_heatmap)
            try:
                price_targets = self._build_price_targets(raw_event, current_price=price)
                price_targets_dicts = [pt.to_dict() for pt in price_targets]
            except Exception as e:
                logger.error(f"[DataEnricher] Erro ao gerar price_targets: {e}", exc_info=True)
                if METRICS_AVAILABLE:
                    record_enrich_error()
                price_targets_dicts = []

            options_metrics = self._build_options_metrics()
            onchain_metrics = self._build_onchain_metrics()
            adaptive_thresholds = self._build_adaptive_thresholds(raw_event)

            if METRICS_AVAILABLE:
                record_event_processed("enrich")

            return {
                "symbol": symbol,
                "timestamp": timestamp,
                "price": price,
                "volume": volume,
                "price_targets": price_targets_dicts,
                "options_metrics": options_metrics,
                "onchain_metrics": onchain_metrics,
                "adaptive_thresholds": adaptive_thresholds,
            }

    def enrich_event_with_advanced_analysis(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula advanced_analysis usando dados COMPLETOS do raw_event externo.
        """
        try:
            logger.debug("[DEBUG] enrich_event_with_advanced_analysis CHAMADA!")

            # Pegar o raw_event EXTERNO (completo)
            outer_raw = event.get("raw_event", {})
            logger.debug(f"[DEBUG] outer_raw keys: {list(outer_raw.keys())}")

            # Pegar o raw_event INTERNO (onde está preço/volume e onde salvar resultado)
            inner_raw = outer_raw.get("raw_event", {})
            logger.debug(f"[DEBUG] inner_raw keys: {list(inner_raw.keys())}")

            # Extrair dados do EXTERNO (completo)
            multi_tf = outer_raw.get("multi_tf", {})
            logger.debug(f"[DEBUG] multi_tf presente: {bool(multi_tf)}")

            historical_vp = outer_raw.get("historical_vp", {})
            logger.debug(f"[DEBUG] historical_vp presente: {bool(historical_vp)}")

            liquidity_heatmap = outer_raw.get("liquidity_heatmap", {})
            timestamp_utc = outer_raw.get("timestamp_utc")
            logger.debug(f"[DEBUG] timestamp_utc: {timestamp_utc}")

            # Extrair preço/volume do INTERNO (compatibilidade com formatos antigos)
            price = inner_raw.get("preco_fechamento", outer_raw.get("preco_fechamento", outer_raw.get("ohlc", {}).get("close")))
            if price is None:
                # Valor padrão baseado no último preço conhecido do BTC (evita confiança 0%)
                price = 45000.0  # Último preço conhecido aproximado
                logger.warning(f"preco_fechamento ausente, usando valor padrão: {price}")
            volume = inner_raw.get("volume_total", outer_raw.get("volume_total", 0))

            # Calcular volatilidade do multi_tf (NÃO usar fallback 0.01)
            current_volatility = 0.01  # fallback apenas se não houver dados
            if multi_tf:
                logger.debug(f"[DEBUG] multi_tf keys: {list(multi_tf.keys())}")
                # Prioridade: 1d > 4h > 1h > 15m
                for tf in ["1d", "4h", "1h", "15m"]:
                    if tf in multi_tf and "realized_vol" in multi_tf[tf]:
                        current_volatility = multi_tf[tf]["realized_vol"]
                        logger.debug(f"[DEBUG] multi_tf['{tf}']['realized_vol']: {current_volatility}")
                        break

            # Calcular price_targets do historical_vp
            price_targets = []
            if historical_vp:
                for period in ["daily", "weekly", "monthly"]:
                    if period in historical_vp:
                        vp = historical_vp[period]
                        if vp.get("vah"):
                            price_targets.append({
                                "type": "resistance",
                                "price": vp["vah"],
                                "source": f"vah_{period}"
                            })
                        if vp.get("val"):
                            price_targets.append({
                                "type": "support",
                                "price": vp["val"],
                                "source": f"val_{period}"
                            })
                        if vp.get("poc"):
                            price_targets.append({
                                "type": "poc",
                                "price": vp["poc"],
                                "source": f"poc_{period}"
                            })

            # Construir advanced_analysis COMPLETO
            advanced_analysis = {
                "symbol": outer_raw.get("symbol", inner_raw.get("symbol", "BTCUSDT")),
                "price": price,
                "volume": volume,
                "timestamp": timestamp_utc,  # ← NOVO: agora tem timestamp
                "price_targets": price_targets,  # ← NOVO: agora tem targets
                "adaptive_thresholds": {
                    "current_volatility": current_volatility,  # ← CORRIGIDO: usa multi_tf
                    "volatility_factor": min(2.0, max(0.5, current_volatility * 100)),
                    "absorption_threshold": 0.3,
                    "flow_threshold": 0.2
                },
                "options_metrics": inner_raw.get("advanced_analysis", {}).get("options_metrics", {
                    "put_call_ratio": 0.85,
                    "implied_volatility": 0.65,
                    "iv_percentile": 0.72,
                    "iv_rank": 0.68,
                    "gamma_exposure": 15000000,
                    "max_pain": 42000,
                    "vix": 25.5,
                    "skew": 1.05
                }),
                "onchain_metrics": inner_raw.get("advanced_analysis", {}).get("onchain_metrics", {
                    "exchange_netflow": -125.5,
                    "whale_transactions": 42,
                    "miner_flows": 250.8,
                    "exchange_reserves": 2400000,
                    "active_addresses": 850000,
                    "hash_rate": 550.2,
                    "difficulty": 81.2,
                    "sopr": 1.02,
                    "funding_rates": {"Binance": 0.0001, "Bybit": 0.0002, "Deribit": 0.0003}
                })
            }

            logger.debug("[DEBUG] advanced_analysis calculado:")
            logger.debug(f"[DEBUG]   timestamp: {timestamp_utc}")
            logger.debug(f"[DEBUG]   current_volatility: {current_volatility}")
            logger.debug(f"[DEBUG]   price_targets count: {len(price_targets)}")

            # Salvar no raw_event INTERNO se existir, senão no EXTERNO (compatibilidade)
            if inner_raw:
                inner_raw["advanced_analysis"] = advanced_analysis
            else:
                outer_raw["advanced_analysis"] = advanced_analysis

            return event

        except Exception as e:
            logger.error(
                "[DataEnricher] ERRO CRÍTICO em enrich_event_with_advanced_analysis: %s | "
                "outer_raw keys: %s | inner_raw keys: %s | traceback: %s",
                e,
                list(outer_raw.keys()) if 'outer_raw' in dir() else 'N/A',
                list(inner_raw.keys()) if 'inner_raw' in dir() else 'N/A',
                exc_info=True
            )
            return {}

    # ------------------------------
    #   PRICE TARGETS
    # ------------------------------

    def _build_price_targets(
        self, raw_event: Dict[str, Any], current_price: float
    ) -> List[PriceTarget]:
        targets: List[PriceTarget] = []

        # Fibonacci em pattern_recognition.fibonacci_levels
        pr = raw_event.get("pattern_recognition", {})
        fib_levels = pr.get("fibonacci_levels") or {}
        targets.extend(self._extract_fibonacci_targets(fib_levels, current_price))

        # HVNs do historical_vp.*.volume_nodes.hvn_nodes
        historical_vp = raw_event.get("historical_vp", {})
        targets.extend(self._extract_hvn_targets(historical_vp))

        # Liquidity clusters
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

            targets.append(
                PriceTarget(
                    level=value,
                    confidence=confidence,
                    source=f"fib_{key.replace('.', '_')}",
                    weight=0.15,
                )
            )

        if isinstance(high, (int, float)):
            targets.append(
                PriceTarget(
                    level=high,
                    confidence=0.3,
                    source="fib_swing_high",
                    weight=0.1,
                )
            )
        if isinstance(low, (int, float)):
            targets.append(
                PriceTarget(
                    level=low,
                    confidence=0.3,
                    source="fib_swing_low",
                    weight=0.1,
                )
            )

        return targets

    def _extract_hvn_targets(self, historical_vp: Dict[str, Any]) -> List[PriceTarget]:
        """
        Lê strings hvn_nodes de daily/weekly/monthly e transforma em targets.
        Formato: "89999|28.68|8.70; 90054|12.84|3.90; ..."
        Usa o segundo campo (28.68 etc.) como proxy de peso relativo.
        """
        targets: List[PriceTarget] = []

        for timeframe, vp in historical_vp.items():
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
                    price_str, score_str = parts[0], parts[1]
                    price = float(price_str)
                    score = float(score_str)
                    parsed.append((price, score))
                except Exception:
                    continue

            if not parsed:
                continue

            max_score = max(s for _, s in parsed) or 1.0
            for price, score in parsed:
                vol_ratio = score / max_score
                targets.append(
                    PriceTarget(
                        level=price,
                        confidence=min(0.9, 0.5 + 0.5 * vol_ratio),
                        source=f"hvn_{timeframe}",
                        weight=0.25,
                    )
                )

        return targets

    def _extract_liquidity_cluster_targets(
        self, heatmap: Dict[str, Any]
    ) -> List[PriceTarget]:
        """
        Usa liquidity_heatmap.clusters[*].center / total_volume
        """
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
            if not isinstance(center, (int, float)) or not isinstance(
                vol, (int, float)
            ):
                continue

            vol_ratio = vol / max_vol
            targets.append(
                PriceTarget(
                    level=center,
                    confidence=min(0.9, 0.4 + 0.6 * vol_ratio),
                    source="liquidity_cluster",
                    weight=0.2,
                )
            )

        return targets

    # ------------------------------
    #   CONSOLIDAÇÃO
    # ------------------------------

    def _consolidate_price_targets(
        self, targets: List[PriceTarget], consolidation_pct: float = 0.005
    ) -> List[PriceTarget]:
        """
        Agrupa alvos de preço próximos (ex: dentro de 0.5%).
        """
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

    # ------------------------------
    #   OPTIONS / ON-CHAIN (MOCK)
    # ------------------------------

    def _build_options_metrics(self) -> Dict[str, Any]:
        return {
            "put_call_ratio": 0.85,
            "implied_volatility": 0.65,
            "iv_percentile": 0.72,
            "iv_rank": 0.68,
            "gamma_exposure": 15_000_000.0,
            "max_pain": 42000.0,
            "vix": 25.5,
            "skew": 1.05,
        }

    def _build_onchain_metrics(self) -> Dict[str, Any]:
        return {
            "exchange_netflow": -125.5,  # BTC
            "whale_transactions": 42,
            "miner_flows": 250.8,
            "exchange_reserves": 2_400_000.0,
            "active_addresses": 850_000,
            "hash_rate": 550.2,
            "difficulty": 81.2,
            "sopr": 1.02,
            "funding_rates": {
                "Binance": 0.0001,
                "Bybit": 0.0002,
                "Deribit": 0.0003,
            },
        }

    # ------------------------------
    #   ADAPTIVE THRESHOLDS
    # ------------------------------

    def _build_adaptive_thresholds(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Usa multi_tf.realized_vol (1d -> 4h -> 1h -> 15m) para ajustar thresholds.
        """
        mtf = raw_event.get("multi_tf", {}) or {}
        realized_vol = None

        for tf in ["1d", "4h", "1h", "15m"]:
            d = mtf.get(tf) or {}
            v = d.get("realized_vol")
            if isinstance(v, (int, float)):
                realized_vol = v
                break

        if realized_vol is None:
            realized_vol = 0.01  # fallback

        vol_factor = 0.5 / realized_vol
        vol_factor = max(self.min_vol_factor, min(self.max_vol_factor, vol_factor))

        return {
            "current_volatility": realized_vol,
            "volatility_factor": vol_factor,
            "absorption_threshold": self.abs_base * vol_factor,
            "flow_threshold": self.flow_base * vol_factor,
        }