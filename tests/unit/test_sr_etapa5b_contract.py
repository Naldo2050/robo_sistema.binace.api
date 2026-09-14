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
        # Fix provenance 2026-09: força por evidência (wall ratio + confluência),
        # nunca mais o antigo min(40,|imb|*200)×1.3=52 do imbalance global.
        # Wall ratio 2.0 → sinal 20; + hvn 25 → 22.5×1.6 = 36, ancorado na wall.
        res = _detector_event(
            orderbook_data={
                "bid_depth_usd": 50000, "ask_depth_usd": 100000, "imbalance": -0.25,
                "walls": {"bids": [], "asks": [
                    {"side": "ask", "price": 64780.0, "qty": 2.0, "limit_threshold": 1.0},
                ]},
            },
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64737.0]},
        )
        zone = None
        for z in res["buy_defense"] + res["sell_defense"]:
            if "orderbook_ask_wall" in z["sources"]:
                zone = z
        assert zone is not None
        assert zone["strength"] == 36
        assert zone["source_count"] == 2
        assert zone["center"] == 64780.0
        assert zone["observed"] is True
        assert zone["snapshot_only"] is True

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


def _offline_groq_prompt(model_name="llama-3.3-70b-versatile"):
    """Reconstrói OFFLINE o system prompt final (sem cliente, sem LLM)."""
    import importlib
    mod = importlib.import_module("market_orchestrator.ai.analyzer_qwen")
    inst = mod.AIAnalyzer.__new__(mod.AIAnalyzer)
    inst.mode = "groq"
    inst.model_name = model_name
    inst.config = {}
    inst._compression_enabled = False
    return inst._get_system_prompt()


def _offline_prompt(mode, model_name, compression, config=None):
    """Qualquer modo de _get_system_prompt, offline, sem cliente."""
    import importlib
    mod = importlib.import_module("market_orchestrator.ai.analyzer_qwen")
    inst = mod.AIAnalyzer.__new__(mod.AIAnalyzer)
    inst.mode = mode
    inst.model_name = model_name
    inst.config = config or {}
    inst._compression_enabled = compression
    return inst._get_system_prompt()


GROQ_MODES = {
    "groq-large": ("groq", "llama-3.3-70b-versatile", False, {}),
    "groq-strict": ("groq", "llama-3.1-8b-instant", False, {}),
    "groq-default": ("groq", "test-model-xyz", False, {}),
}


class TestPromptModeEquivalence:
    """Pré-commit prompt unification: mesma semântica em todos os modos."""

    def _prompts(self):
        return {name: _offline_prompt(*args) for name, args in GROQ_MODES.items()} | {
            "nongroq-compressed": _offline_prompt("openai", "x", True, {}),
            "nongroq-default": _offline_prompt("openai", "x", False, {}),
        }

    def test_all_live_modes_define_core_flow_keys(self):
        for name, prompt in self._prompts().items():
            assert "trade_imb" in prompt, f"{name} sem trade_imb"
            assert "depth_imb" in prompt, f"{name} sem depth_imb"

    def test_all_live_modes_define_wall_semantics(self):
        for name, prompt in self._prompts().items():
            assert "OBS_WALL" in prompt, f"{name} sem OBS_WALL"
            assert "PROJECTED_DEPTH" in prompt, f"{name} sem PROJECTED_DEPTH"

    def test_all_live_modes_mark_scores_uncalibrated(self):
        for name, prompt in self._prompts().items():
            low = prompt.lower()
            assert "calibra" in low, f"{name} sem aviso de nao-calibracao"
        for name, prompt in self._prompts().items():
            if name.startswith("groq") or name == "nongroq-default":
                assert "NOT probability" in prompt or "não é probabilidade" in prompt, name

    def test_all_live_modes_define_market_impact(self):
        for name, prompt in self._prompts().items():
            assert "slip" in prompt.lower(), f"{name} sem slippage"
            assert "null" in prompt.lower(), f"{name} sem semantica null/partial"

    def test_all_live_modes_define_quality_onchain_vwap(self):
        for name, prompt in self._prompts().items():
            low = prompt.lower()
            assert "onchain" in low, f"{name} sem onchain"
            assert "svw" in low or "session_vwap" in low or "session vwap" in low, \
                f"{name} sem vwap"
        for name, prompt in self._prompts().items():
            if name.startswith("groq") or name == "nongroq-default":
                assert "qual" in prompt.lower(), f"{name} sem quality"

    def test_legacy_mode_keeps_separate_contract(self):
        """Legacy recebe schema estendido antigo — documentado, sem P02."""
        prompt = _offline_prompt("openai", "x", False, {"ai": {"prompt_style": "legacy"}})
        import importlib
        mod = importlib.import_module("market_orchestrator.ai.analyzer_qwen")
        assert prompt == mod.SYSTEM_PROMPT_LEGACY
        assert "trade_imb" not in prompt  # prova: legacy não lê payload compacto

    def test_no_mode_teaches_stale_sr_rule(self):
        for name, prompt in self._prompts().items():
            low = prompt.lower()
            assert "força > 50" not in low, name
            assert "strength > 50" not in low, name
            assert "confluência de fontes de defesa" not in low, name
            assert "resistências institucionais" not in low, name

    def test_nongroq_default_matches_groq_default_semantics(self):
        """Mesmo payload compacto, mesma legenda canônica (unificação)."""
        groq = _offline_prompt("groq", "test-model-xyz", False, {})
        default = _offline_prompt("openai", "x", False, {})
        import importlib
        legend = importlib.import_module("common.ai_field_legend").FIELD_LEGEND
        assert legend in groq and legend in default


class TestRealPayloadCoverage:
    """Payload real com todas as seções críticas tem definição no prompt."""

    def _payload(self):
        return {
            "symbol": "BTCUSDT", "epoch_ms": 1741400000000, "trigger": "AT",
            "price": {"c": 77399.1},
            "flow": {"d1": "+1", "trade_imb": 0.70},
            "ob": {"b": "1.0M", "a": "1.0M", "depth_imb": -0.60, "depth_t5": -0.55,
                   "slip_s": 500.0},
            "tf": {"t": "UP"},
            "sr": {"s1": [64690, 40], "s1_dist": 12709, "def_bias": "neutral"},
            "quant": {"pu": 0.80, "c": 0.60},
            "qual": {"liq": "NORMAL", "comp": 95},
            "vwap": {"svw": 77300.0, "dist": 0.0013, "side": "above"},
            "onchain": {"st": "ok", "age": 30.0, "active_addr": 400000},
            "w": {"s": -41},
        }

    def test_every_critical_key_defined_in_groq_prompt(self):
        payload = self._payload()
        prompt = _offline_groq_prompt()
        low = prompt.lower()
        key_to_concept = {
            "flow": "trade_imb", "ob": "depth_imb", "sr": "sr.r1",
            "quant": "calibra", "qual": "qual", "vwap": "svw",
            "onchain": "onchain", "w": "whale",
        }
        for section, concept in key_to_concept.items():
            assert section in payload, section
            assert concept.lower() in low, f"{section}->{concept}"

    def test_every_critical_key_defined_in_compressed_prompt(self):
        payload = self._payload()
        prompt = _offline_prompt("openai", "x", True, {})
        low = prompt.lower()
        for section, concept in {"flow": "trade_imb", "ob": "depth_imb",
                                 "sr": "OBS_WALL", "quant": "calibra",
                                 "qual": "qual", "vwap": "svw",
                                 "onchain": "onchain"}.items():
            assert section in payload, section
            assert concept.lower() in low, f"{section}->{concept}"


class TestOfflinePromptSemanticClosure:
    """FASE S/R SEMANTIC CLOSURE: prompt final sem contradições (offline)."""

    def test_no_stale_strength_rule(self):
        prompt = _offline_groq_prompt()
        low = prompt.lower()
        assert "força > 50" not in low
        assert "strength > 50" not in low
        assert "forca > 50" not in low
        assert "confluência de fontes de defesa" not in low
        assert "confluencia de fontes de defesa" not in low
        assert "resistências institucionais" not in low
        assert "resistencias institucionais" not in low
        assert "suportes institucionais" not in low
        assert "defense.sell_zone" not in prompt
        assert "defense.buy_zone" not in prompt

    def test_current_sr_contract_present(self):
        prompt = _offline_groq_prompt()
        low = prompt.lower()
        assert "snapshot" in low
        assert "heuristic" in low or "heurístic" in low
        assert "trade_imb" in prompt
        assert "depth_imb" in prompt
        assert "sr.r1" in prompt
        assert "ctx.poc" in prompt

    def test_no_probability_attached_to_scores(self):
        prompt = _offline_groq_prompt()
        assert "NÃO é probabilidade" in prompt or "NOT probability" in prompt
        assert "probabilidade calibrada" not in prompt
        assert "calibrated probability" not in prompt

    def test_market_impact_text_preserved(self):
        prompt = _offline_groq_prompt()
        assert "VWAP" in prompt
        assert "point-in-time" in prompt
        assert "P1M" in prompt

    def test_no_llm_call_during_prompt_build(self):
        """Prova NO-LLM: cliente que explode se tocado; só strings são usadas."""
        from unittest.mock import MagicMock
        import importlib
        mod = importlib.import_module("market_orchestrator.ai.analyzer_qwen")
        inst = mod.AIAnalyzer.__new__(mod.AIAnalyzer)
        inst.mode = "groq"
        inst.model_name = "llama-3.3-70b-versatile"
        inst.config = {}
        inst._compression_enabled = False
        inst.client = MagicMock()
        inst.client.chat.completions.create.side_effect = AssertionError("LLM must not be called")
        prompt = inst._get_system_prompt()
        assert isinstance(prompt, str) and len(prompt) > 1000
        inst.client.chat.completions.create.assert_not_called()

    def test_compact_legend_is_dead_code_documented(self):
        """FIELD_LEGEND_COMPACT não tem consumidor — documentado, não enviado."""
        import pathlib
        hits = [
            p for p in pathlib.Path("market_orchestrator").rglob("*.py")
            if "FIELD_LEGEND_COMPACT" in p.read_text(encoding="utf-8", errors="ignore")
        ] + [
            p for p in pathlib.Path("common").rglob("*.py")
            if "FIELD_LEGEND_COMPACT" in p.read_text(encoding="utf-8", errors="ignore")
            and p.name != "ai_field_legend.py"
        ]
        assert hits == [], f"FIELD_LEGEND_COMPACT ganhou consumidor: {hits}"


class TestPayloadPromptContract:
    """§18: payload com sinais opostos + evidências + prompt offline coerente."""

    def _event(self):
        return {
            "symbol": "BTCUSDT",
            "tipo_evento": "ANALISE_TECNICA",
            "preco_fechamento": 77399.1,
            "fluxo_continuo": {"order_flow": {"net_flow_1m": 1.0, "flow_imbalance": 0.70}},
            "orderbook_data": {"bid_depth_usd": 1e6, "ask_depth_usd": 1e6,
                               "flow_imbalance": -0.60, "imbalance": -0.60},
            "order_book_depth": {"L5": {"flow_imbalance": -0.55, "imbalance": -0.55}},
            "market_impact": {"slippage_matrix": {"100k_usd": {"buy": None, "sell": 5.0}}},
            "institutional_analytics": {"sr_analysis": {"defense_zones": {
                "buy_defense": [
                    {"center": 77399.0, "strength": 36, "source_count": 1,
                     "sources": ["orderbook_bid_wall"], "liquidity_only": True,
                     "observed": True, "center_origin": "observed_wall_price"},
                    {"center": 64690.0, "strength": 40, "source_count": 2,
                     "sources": ["vp_hvn", "pivot_daily_s1"],
                     "has_structural_confluence": True,
                     "structural_sources": ["vp_hvn", "pivot_daily_s1"]},
                ],
                "sell_defense": [
                    {"center": 77553.9, "strength": 39, "source_count": 1,
                     "sources": ["depth_asymmetry"], "observed": False,
                     "projected": True, "projected_only": True},
                ],
                "defense_asymmetry": {"bias": "neutral"}}}},
        }

    def test_opposite_signals_and_gating_in_payload(self):
        from market_orchestrator.ai.payload_builder_compact import build_compact_payload
        compact = build_compact_payload(self._event())
        # Sinais opostos permitidos, sem absorção forçada.
        assert compact["flow"]["trade_imb"] == 0.70
        assert compact["ob"]["depth_imb"] == -0.60
        # Wall-only e depth-only fora de sr; só confluência estrutural.
        assert compact["sr"] == {"s1": [64690, 40], "s1_dist": 12709, "def_bias": "neutral"} or \
            {k: v for k, v in compact["sr"].items() if k in ("s1", "s1_dist")} == \
            {"s1": [64690, 40], "s1_dist": 12709}
        assert "r1" not in compact["sr"]
        # Market impact parcial: fail-closed (sem número inventado no buy).
        assert compact["ob"].get("slip_b") is None
        assert compact["ob"]["slip_s"] == 500.0

    def test_prompt_over_gated_payload_has_no_stale_claims(self):
        prompt = _offline_groq_prompt()
        low = prompt.lower()
        for forbidden in ["força > 50", "strength > 50",
                          "confluência de fontes de defesa",
                          "resistências institucionais",
                          "probabilidade calibrada"]:
            assert forbidden not in low, forbidden
        assert "snapshot" in low and "trade_imb" in prompt and "depth_imb" in prompt
