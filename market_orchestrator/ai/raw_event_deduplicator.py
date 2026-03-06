# market_orchestrator/ai/raw_event_deduplicator.py
"""
Raw Event Deduplicator V3
=========================
Remove duplicacoes do evento ANALYSIS_TRIGGER ANTES do ai_payload ser construido.

Economia estimada: ~7,000 caracteres por evento (~40% do evento bruto).
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class RawEventDeduplicator:
    """Remove campos duplicados/redundantes do evento bruto."""

    def __init__(self, deep_copy: bool = True):
        self.deep_copy = deep_copy
        self._stats: Dict[str, int] = {"fields_removed": 0, "chars_saved_estimate": 0}

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    def deduplicate(self, event: dict) -> dict:
        if not isinstance(event, dict):
            return event
        if self.deep_copy:
            event = copy.deepcopy(event)

        self._stats = {"fields_removed": 0, "chars_saved_estimate": 0}

        self._remove_flow_metrics_duplicate(event)
        self._remove_orderbook_duplicate(event)
        self._remove_volume_duplicates(event)
        self._remove_price_duplicate(event)
        self._remove_order_book_depth(event)
        self._remove_time_index(event)
        self._clean_heatmap_clusters(event)
        self._remove_strongest_copies(event)
        self._clean_contextual_snapshot(event)
        self._remove_raw_event_nested_raw(event)

        if self._stats["fields_removed"] > 0:
            logger.debug(
                "Deduplicacao: %d campos removidos, ~%d chars economizados",
                self._stats["fields_removed"],
                self._stats["chars_saved_estimate"],
            )
        return event

    # -- 1. flow_metrics <-> fluxo_continuo (CRITICO - maior duplicacao) --

    def _remove_flow_metrics_duplicate(self, event: dict) -> None:
        raw = event.get("raw_event")
        if not isinstance(raw, dict):
            return
        if "fluxo_continuo" in event and "flow_metrics" in raw:
            del raw["flow_metrics"]
            self._track("flow_metrics (raw_event)", 3500)

    # -- 2. orderbook_data duplicado --

    def _remove_orderbook_duplicate(self, event: dict) -> None:
        raw = event.get("raw_event")
        if not isinstance(raw, dict):
            return
        if "orderbook_data" in event and "orderbook_data" in raw:
            top_ob = event.get("orderbook_data", {})
            if isinstance(top_ob, dict) and ("depth_metrics" in top_ob or len(top_ob) >= len(raw.get("orderbook_data", {}))):
                del raw["orderbook_data"]
                self._track("orderbook_data (raw_event)", 400)

    # -- 3. volume_compra/venda triplicado --

    def _remove_volume_duplicates(self, event: dict) -> None:
        raw = event.get("raw_event") or {}
        cs = event.get("contextual_snapshot") or {}
        for field in ("volume_compra", "volume_venda"):
            if field in event:
                if isinstance(raw, dict) and field in raw:
                    del raw[field]
                    self._track(f"{field} (raw_event)", 30)
                if isinstance(cs, dict) and field in cs:
                    del cs[field]
                    self._track(f"{field} (contextual_snapshot)", 30)

    # -- 4. preco_fechamento duplicado --

    def _remove_price_duplicate(self, event: dict) -> None:
        raw = event.get("raw_event")
        if isinstance(raw, dict) and "preco_fechamento" in event and "preco_fechamento" in raw:
            del raw["preco_fechamento"]
            self._track("preco_fechamento (raw_event)", 30)

    # -- 5. order_book_depth eh subset de orderbook_data --

    def _remove_order_book_depth(self, event: dict) -> None:
        ob_data = event.get("orderbook_data")
        ob_depth = event.get("order_book_depth")
        if isinstance(ob_data, dict) and isinstance(ob_depth, dict):
            if "total_depth_ratio" in ob_depth:
                ob_data["total_depth_ratio"] = ob_depth["total_depth_ratio"]
            del event["order_book_depth"]
            self._track("order_book_depth", 500)

    # -- 6. time_index redundante --

    def _remove_time_index(self, event: dict) -> None:
        fc = event.get("fluxo_continuo")
        if isinstance(fc, dict) and "time_index" in fc:
            del fc["time_index"]
            self._track("time_index (fluxo_continuo)", 250)

    # -- 7. Limpar clusters do heatmap --

    def _clean_heatmap_clusters(self, event: dict) -> None:
        fc = event.get("fluxo_continuo")
        if not isinstance(fc, dict):
            return
        hm = fc.get("liquidity_heatmap")
        if not isinstance(hm, dict):
            return
        for cluster in hm.get("clusters", []):
            if not isinstance(cluster, dict):
                continue
            removed = 0
            for ts_field in ("recent_timestamp", "recent_ts_ms"):
                if ts_field in cluster:
                    del cluster[ts_field]
                    removed += 1
            for internal in ("bin_threshold_usd", "volume_std", "price_std"):
                if internal in cluster:
                    del cluster[internal]
                    removed += 1
            if removed:
                self._track(f"heatmap_cluster x{removed}", removed * 30)

    # -- 8. strongest_buy/sell sao copias de defense[0] --

    def _remove_strongest_copies(self, event: dict) -> None:
        ia = event.get("institutional_analytics")
        if not isinstance(ia, dict):
            return

        # defense_zones
        sr = ia.get("sr_analysis") or {}
        dz = sr.get("defense_zones") or {}
        if isinstance(dz, dict):
            if dz.get("buy_defense") and "strongest_buy" in dz:
                del dz["strongest_buy"]
                self._track("strongest_buy", 200)
            if dz.get("sell_defense") and "strongest_sell" in dz:
                del dz["strongest_sell"]
                self._track("strongest_sell", 200)

        # volume_node_strength
        pa = ia.get("profile_analysis") or {}
        vns = pa.get("volume_node_strength") or {}
        if isinstance(vns, dict):
            if vns.get("scored_hvns") and "strongest_hvn" in vns:
                del vns["strongest_hvn"]
                self._track("strongest_hvn", 150)
            if vns.get("scored_lvns") and "strongest_lvn" in vns:
                del vns["strongest_lvn"]
                self._track("strongest_lvn", 150)
            # Limitar arrays a 5
            for arr_key in ("scored_hvns", "scored_lvns"):
                arr = vns.get(arr_key)
                if isinstance(arr, list) and len(arr) > 5:
                    orig = len(arr)
                    vns[arr_key] = arr[:5]
                    self._track(f"{arr_key} {orig}->5", (orig - 5) * 100)

        # sr_strength supports/resistances
        sr_str = sr.get("sr_strength") or {}
        if isinstance(sr_str, dict) and "levels" in sr_str:
            for dup_key in ("supports", "resistances", "nearest_support", "nearest_resistance"):
                if dup_key in sr_str:
                    del sr_str[dup_key]
                    self._track(f"sr_strength.{dup_key}", 300)

    # -- 9. Limpar contextual_snapshot --

    def _clean_contextual_snapshot(self, event: dict) -> None:
        cs = event.get("contextual_snapshot")
        if not isinstance(cs, dict):
            return
        duplicate_keys = {
            "symbol", "volume_total", "volume_compra", "volume_venda",
            "flow_metrics", "historical_vp", "orderbook_data",
            "multi_tf", "derivatives", "market_context", "market_environment",
        }
        removed = 0
        for key in list(cs.keys()):
            if key in duplicate_keys:
                del cs[key]
                removed += 1
        if removed:
            self._track(f"contextual_snapshot x{removed}", removed * 200)

    # -- 10. raw_event > raw_event aninhado --

    def _remove_raw_event_nested_raw(self, event: dict) -> None:
        raw = event.get("raw_event")
        if not isinstance(raw, dict):
            return
        inner = raw.get("raw_event")
        if isinstance(inner, dict):
            for key, val in inner.items():
                if key != "raw_event":
                    raw.setdefault(key, val)
            del raw["raw_event"]
            self._track("raw_event.raw_event (nested)", 2000)

    def _track(self, field_name: str, chars_estimate: int) -> None:
        self._stats["fields_removed"] += 1
        self._stats["chars_saved_estimate"] += chars_estimate


def deduplicate_event(event: dict, deep_copy: bool = True) -> dict:
    """Funcao de conveniencia para deduplicar evento."""
    return RawEventDeduplicator(deep_copy=deep_copy).deduplicate(event)
