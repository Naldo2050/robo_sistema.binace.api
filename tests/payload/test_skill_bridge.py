"""
Testes do skill_bridge — contrato, registry e integração com payload.
"""

import pytest
from market_orchestrator.ai.payload_sections.skill_bridge import (
    BaseAnalyticalSkill,
    SkillRegistry,
    MeanReversionContextSkill,
    AbsorptionStrengthSkill,
    run_skills,
    get_registry,
)


def make_mr_payload() -> dict:
    return {
        "trigger": "AT",
        "regime": {"cs": "BEAR", "cf": 1.0, "v": "H", "mode": "MR"},
        "flow": {
            "d1": "+16K",
            "imb": 0.18,
            "pa": "buy_absorption",
            "abs_buy_str": 6.2,
            "abs_sell_exh": 1.5,
            "abs_cont": 0.28,
        },
        "ext": {
            "hurst": 0.341,
            "reg": {"pos": 0.43, "sl": -2.8, "dev": -0.06},
        },
        "mr": {
            "score": 0.72,
            "sig": "stretched_bearish",
        },
    }


def make_abs_payload() -> dict:
    return {
        "trigger": "ABS",
        "regime": {"mode": "RB"},
        "flow": {
            "d1": "+28K",
            "imb": 0.31,
            "pa": "buy_absorption",
            "abs_buy_str": 7.8,
            "abs_sell_exh": 0.9,
            "abs_cont": 0.45,
        },
        "ext": {"hurst": 0.38},
    }


class TestBaseSkill:

    def test_should_run_all_regimes_all_triggers(self):
        skill = MeanReversionContextSkill()
        payload = {"trigger": "AT", "regime": {"mode": "MR"}}
        assert skill.should_run(payload) is True

    def test_should_not_run_wrong_regime(self):
        skill = MeanReversionContextSkill()
        payload = {"trigger": "AT", "regime": {"mode": "BRK"}}
        assert skill.should_run(payload) is False

    def test_should_not_run_wrong_trigger(self):
        skill = AbsorptionStrengthSkill()
        payload = {"trigger": "DIV", "regime": {"mode": "MR"}}
        assert skill.should_run(payload) is False

    def test_execute_returns_empty_on_exception(self):
        class BrokenSkill(BaseAnalyticalSkill):
            name = "broken"
            version = "1.0"
            cost_class = "low"
            active_regimes: set = {"ALL"}
            active_triggers: set = {"ALL"}

            def _run(self, payload):
                raise RuntimeError("Falha simulada")

        skill = BrokenSkill()
        result = skill.execute({"trigger": "AT"})
        assert result == {}

    def test_execute_does_not_propagate_exception(self):
        class BrokenSkill(BaseAnalyticalSkill):
            name = "broken2"
            version = "1.0"
            cost_class = "low"
            active_regimes: set = {"ALL"}
            active_triggers: set = {"ALL"}

            def _run(self, payload):
                raise ValueError("Erro interno")

        skill = BrokenSkill()
        try:
            result = skill.execute({})
            assert result == {}
        except Exception:
            pytest.fail("execute() não deve propagar exceções")


class TestSkillRegistry:

    def test_register_and_retrieve(self):
        registry = SkillRegistry()
        skill = MeanReversionContextSkill()
        registry.register(skill)
        assert "mean_reversion_context" in registry.registered

    def test_unregister_removes_skill(self):
        registry = SkillRegistry()
        skill = MeanReversionContextSkill()
        registry.register(skill)
        registry.unregister("mean_reversion_context")
        assert "mean_reversion_context" not in registry.registered

    def test_run_all_returns_results_for_eligible_skills(self):
        registry = SkillRegistry()
        registry.register(MeanReversionContextSkill())
        results = registry.run_all(make_mr_payload())
        assert "mean_reversion_context" in results

    def test_run_all_skips_ineligible_regime(self):
        registry = SkillRegistry()
        registry.register(MeanReversionContextSkill())
        payload = {"trigger": "AT", "regime": {"mode": "BRK"}}
        results = registry.run_all(payload)
        assert "mean_reversion_context" not in results

    def test_run_all_respects_cost_limit(self):
        class HighCostSkill(BaseAnalyticalSkill):
            name = "high_cost"
            version = "1.0"
            cost_class = "high"
            active_regimes: set = {"ALL"}
            active_triggers: set = {"ALL"}

            def _run(self, payload):
                return {"result": "expensive"}

        registry = SkillRegistry()
        registry.register(HighCostSkill())

        results_low = registry.run_all(make_mr_payload(), cost_limit="low")
        results_high = registry.run_all(make_mr_payload(), cost_limit="high")

        assert "high_cost" not in results_low
        assert "high_cost" in results_high

    def test_run_all_handles_broken_skill_gracefully(self):
        class BrokenSkill(BaseAnalyticalSkill):
            name = "broken3"
            version = "1.0"
            cost_class = "low"
            active_regimes: set = {"ALL"}
            active_triggers: set = {"ALL"}

            def _run(self, payload):
                raise RuntimeError("Erro")

        registry = SkillRegistry()
        registry.register(BrokenSkill())
        registry.register(MeanReversionContextSkill())

        results = registry.run_all(make_mr_payload())

        assert "broken3" not in results
        assert "mean_reversion_context" in results


class TestMeanReversionContextSkill:

    def test_returns_tendency_mean_reverting_when_hurst_low(self):
        skill = MeanReversionContextSkill()
        result = skill.execute(make_mr_payload())
        assert result["tendency"] == "mean_reverting"
        assert result["strength"] > 0

    def test_returns_channel_position(self):
        skill = MeanReversionContextSkill()
        result = skill.execute(make_mr_payload())
        assert result["channel_pos"] == "mid_channel"

    def test_includes_mr_score_when_available(self):
        skill = MeanReversionContextSkill()
        result = skill.execute(make_mr_payload())
        assert result["mr_score"] == 0.72
        assert result["signal"] == "stretched_bearish"

    def test_returns_empty_when_no_data(self):
        skill = MeanReversionContextSkill()
        result = skill.execute({"trigger": "AT", "regime": {"mode": "MR"}})
        assert result == {}

    def test_trending_hurst(self):
        skill = MeanReversionContextSkill()
        payload = make_mr_payload()
        payload["ext"]["hurst"] = 0.72
        result = skill.execute(payload)
        assert result["tendency"] == "trending"

    def test_near_low_channel_position(self):
        skill = MeanReversionContextSkill()
        payload = make_mr_payload()
        payload["ext"]["reg"]["pos"] = 0.1
        result = skill.execute(payload)
        assert result["channel_pos"] == "near_low"

    def test_near_high_channel_position(self):
        skill = MeanReversionContextSkill()
        payload = make_mr_payload()
        payload["ext"]["reg"]["pos"] = 0.9
        result = skill.execute(payload)
        assert result["channel_pos"] == "near_high"


class TestAbsorptionStrengthSkill:

    def test_strong_buy_absorption(self):
        skill = AbsorptionStrengthSkill()
        result = skill.execute(make_abs_payload())
        assert result["quality"] == "strong_buy"
        assert result["operable"] is True

    def test_weak_absorption_not_operable(self):
        skill = AbsorptionStrengthSkill()
        payload = make_abs_payload()
        payload["flow"]["abs_buy_str"] = 1.5
        payload["flow"]["abs_sell_exh"] = 1.2
        result = skill.execute(payload)
        assert result["operable"] is False

    def test_returns_continuation_probability(self):
        skill = AbsorptionStrengthSkill()
        result = skill.execute(make_abs_payload())
        assert result["cont_prob"] == 0.45

    def test_returns_signal_from_pa(self):
        skill = AbsorptionStrengthSkill()
        result = skill.execute(make_abs_payload())
        assert "buy_absorp" in result["signal"] or "buy_absorpt" in result["signal"]

    def test_returns_empty_when_no_absorption_data(self):
        skill = AbsorptionStrengthSkill()
        result = skill.execute({"trigger": "ABS", "regime": {"mode": "MR"}, "flow": {}})
        assert result == {}

    def test_strong_sell_absorption(self):
        skill = AbsorptionStrengthSkill()
        payload = make_abs_payload()
        payload["flow"]["abs_buy_str"] = 0.8
        payload["flow"]["abs_sell_exh"] = 8.2
        payload["flow"]["pa"] = "sell_absorption"
        result = skill.execute(payload)
        assert result["quality"] == "strong_sell"
        assert result["operable"] is True

    def test_moderate_buy_absorption(self):
        skill = AbsorptionStrengthSkill()
        payload = make_abs_payload()
        payload["flow"]["abs_buy_str"] = 5.5
        payload["flow"]["abs_sell_exh"] = 2.0
        result = skill.execute(payload)
        assert result["quality"] == "moderate_buy"


class TestRunSkillsIntegration:

    def test_run_skills_returns_dict(self):
        result = run_skills(make_mr_payload())
        assert isinstance(result, dict)

    def test_run_skills_includes_eligible_skills(self):
        result = run_skills(make_mr_payload())
        assert "mean_reversion_context" in result

    def test_run_skills_abs_trigger_includes_absorption(self):
        result = run_skills(make_abs_payload())
        assert "absorption_strength" in result

    def test_run_skills_result_is_compact(self):
        result = run_skills(make_mr_payload())
        import json
        size = len(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        assert size < 500, f"Skills result muito grande: {size} bytes"
