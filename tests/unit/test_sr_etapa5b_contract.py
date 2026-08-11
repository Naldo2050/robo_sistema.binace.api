"""ETAPA 5B FASE 10 - Contratos semantico-estruturais (antes de qualquer patch).

Documenta o comportamento atual, passando antes e depois:
  A) pivot_points.vah/val/poc (source=classic) = H/L/C do periodo anterior
     (NAO e Volume Profile) - datasets distintos (historical_vp).
  B) historical_vp nunca e alterado pelo pipeline S/R.
  C) O payload compacto do AI NAO contem pivot_points nem
     immediate_support/support_strength/resistance_strength.
  D) defense strength (sr.r1/s1) e confluence-based (proximidade NAO domina).
  E) H/L/C de pivots NAO viram sinais de defesa (pivot_keys exclui high/low/close).
  F) source distingue classic | vp_fallback | multi_tf_fallback.
  G) aliases legados (pivot/pp) continuam funcionando no detector.
"""

import pytest

from institutional.enricher import _build_pivot_points
from market_orchestrator.ai.payload_builder_compact import build_compact_payload
from support_resistance.defense_zones import DefenseZoneDetector

PRICE = 64742.1


def _detector_event(pivot_data=None, vp_data=None, orderbook_data=None):
    return DefenseZoneDetector().detect(
        current_price=PRICE,
        pivot_data=pivot_data,
        vp_data=vp_data,
        orderbook_data=orderbook_data,
    )


def _classic_event():
    return {
        "preco_fechamento": PRICE,
        "pivots": {"daily": {
            "pivot": 65035.38, "r1": 65340.67, "s1": 64596.29,
            "r2": 65779.76, "s2": 64291.0, "r3": 66085.05, "s3": 63851.91,
            "high": 65474.46, "low": 64730.08, "close": 64901.59,
        }},
        "historical_vp": {"daily": {
            "poc": 64689, "vah": 65133, "val": 64520, "status": "success",
        }},
    }


class TestContractPivotPointsVsVP:
    def test_pivot_points_vah_val_poc_are_classic_ohlc(self):
        sr = _build_pivot_points(_classic_event())
        daily = sr["pivot_points"]["daily"]
        assert daily["source"] == "classic"
        assert daily["vah"] == 65474.46
        assert daily["val"] == 64730.08
        assert daily["poc"] == 64901.59
        assert daily["vah"] != 65133

    def test_historical_vp_never_mutated(self):
        event = _classic_event()
        hist_before = dict(event["historical_vp"]["daily"])
        _build_pivot_points(event)
        assert event["historical_vp"]["daily"] == hist_before

    def test_pivot_points_source_enum(self):
        sr = _build_pivot_points(_classic_event())
        assert sr["pivot_points"]["daily"]["source"] in ("classic", "vp_fallback", "multi_tf_fallback")


class TestContractPayloadDoesNotLeakPivots:
    def _payload_keys_of(self, compact):
        def walk(obj, prefix=""):
            keys = set()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    keys.add(f"{prefix}{k}")
                    keys |= walk(v, f"{prefix}{k}.")
            return keys
        return walk(compact)

    def test_compact_payload_excludes_pivot_points_and_immediate(self):
        event = _classic_event()
        payload = build_compact_payload(event)
        flat = " ".join(self._payload_keys_of(payload))
        assert "pivot_points" not in flat
        assert "immediate_support" not in flat
        assert "support_strength" not in flat
        assert "resistance_strength" not in flat


class TestContractDefenseStrengthSemantics:
    def test_defense_strength_is_confluence_based(self):
        res = _detector_event(
            orderbook_data={"bid_depth_usd": 50000, "ask_depth_usd": 100000, "imbalance": -0.25},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64737.0]},
        )
        zone = None
        for z in res["buy_defense"] + res["sell_defense"]:
            if "orderbook_ask_wall" in z["sources"]:
                zone = z
        assert zone is not None
        assert zone["strength"] == 52
        assert zone["source_count"] == 2

    def test_pivot_hlc_never_become_defense_signals(self):
        pivots_classic = {
            "classic": {
                "pivot": 65035.38, "r1": 65340.67, "s1": 64596.29,
                "r2": 65779.76, "s2": 64291.0, "r3": 66085.05, "s3": 63851.91,
                "high": 65474.46, "low": 64730.08, "close": 64901.59,
            }
        }
        res = _detector_event(pivot_data=pivots_classic)
        zones = res["buy_defense"] + res["sell_defense"]
        for z in zones:
            for s in z["sources"]:
                assert "high" not in s and "low" not in s and "close" not in s

    def test_pivot_pp_alias_still_accepted(self):
        for key in ("pivot", "pp"):
            res = _detector_event(pivot_data={"classic": {key: 65035.38}})
            zones = res["buy_defense"] + res["sell_defense"]
            assert any(f"pivot_classic_{key}" in z["sources"] for z in zones)


class TestContractPromptLegend:
    """Legend do prompt documenta as chaves REAIS do payload compacto."""

    def test_no_stale_sr_legend_keys_in_system_prompt(self):
        import importlib
        mod = importlib.import_module("market_orchestrator.ai.analyzer_qwen")
        prompt = mod.SYSTEM_PROMPT
        assert "immediate_resistance" not in prompt
        assert "immediate_support" not in prompt
        assert "resistance_strength" not in prompt
        assert "support_strength" not in prompt
        assert "sr.r1" in prompt
        assert "def_bias" in prompt
        assert "ctx.poc" in prompt

    def test_compressed_dictionary_documents_sr(self):
        import importlib
        mod = importlib.import_module("common.ai_payload_optimizer")
        assert "sr:" in mod.COMPRESSED_KEY_DICTIONARY
        assert "def_bias" in mod.COMPRESSED_KEY_DICTIONARY
        assert "ctx:" in mod.COMPRESSED_KEY_DICTIONARY

    def test_field_legend_documents_sr_and_pivot_points_not_in_compact(self):
        import importlib
        mod = importlib.import_module("common.ai_field_legend")
        assert "sr=defense_zones" in mod.FIELD_LEGEND
        assert "NÃO é enviado no payload compacto" in mod.FIELD_LEGEND
