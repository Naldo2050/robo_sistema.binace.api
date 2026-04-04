"""
skill_bridge.py — Ponte entre o sistema de payload e o futuro framework de skills.

Este módulo define:
  1. O contrato que uma skill analítica deve seguir
  2. Um registry simples de skills disponíveis
  3. A função que o payload builder usará para ativar skills por contexto

Quando o framework de skills/ for criado, este módulo será o ponto
de integração — sem precisar alterar build_compact_payload.py.

Filosofia:
  - Skills são read-only (só leem, não executam ordens)
  - Skills retornam dicts compactos — mesmo formato dos gaps
  - Skills podem ser ativadas por regime, trigger ou contexto
  - Skills com erro são ignoradas silenciosamente
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ============================================================
# CONTRATO DE SKILL
# ============================================================

@runtime_checkable
class AnalyticalSkill(Protocol):
    """
    Protocolo que toda skill analítica deve implementar.

    Uma skill:
      - recebe o payload compacto já montado
      - retorna um dict compacto com sua análise
      - nunca lança exceção para o chamador
      - nunca modifica o payload recebido
    """

    @property
    def name(self) -> str:
        """Identificador único da skill."""
        ...

    @property
    def version(self) -> str:
        """Versão da skill."""
        ...

    @property
    def cost_class(self) -> str:
        """Custo computacional: 'low' | 'medium' | 'high'"""
        ...

    @property
    def active_regimes(self) -> set[str]:
        """
        Regimes em que a skill deve rodar.
        Conjunto de: 'MR' | 'RB' | 'TRD' | 'BRK' | 'ALL'
        """
        ...

    @property
    def active_triggers(self) -> set[str]:
        """
        Triggers em que a skill deve rodar.
        Conjunto de: 'AT' | 'ABS' | 'BRK' | 'WHL' | 'ALL'
        """
        ...

    def should_run(self, payload: dict) -> bool:
        """
        Decide se a skill deve rodar com base no contexto atual.
        Nunca lança exceção.
        """
        ...

    def execute(self, payload: dict) -> dict:
        """
        Executa a análise e retorna resultado compacto.
        Nunca lança exceção — retorna {} em caso de erro.
        """
        ...


# ============================================================
# BASE CLASS — facilita implementação de skills concretas
# ============================================================

class BaseAnalyticalSkill:
    """
    Classe base com implementação padrão de should_run e proteção de execute.

    Subclasses devem implementar:
      - name, version, cost_class
      - active_regimes, active_triggers
      - _run(payload) -> dict  ← lógica real da skill
    """

    name: str = "base_skill"
    version: str = "1.0"
    cost_class: str = "low"
    active_regimes: set[str] = {"ALL"}
    active_triggers: set[str] = {"ALL"}

    def should_run(self, payload: dict) -> bool:
        """Verifica regime e trigger contra os sets da skill."""
        if "ALL" in self.active_regimes and "ALL" in self.active_triggers:
            return True

        regime = payload.get("regime", {}).get("mode", "")
        trigger = payload.get("trigger", "")

        regime_ok = (
            "ALL" in self.active_regimes
            or regime in self.active_regimes
        )
        trigger_ok = (
            "ALL" in self.active_triggers
            or trigger in self.active_triggers
        )

        return regime_ok and trigger_ok

    def execute(self, payload: dict) -> dict:
        """Wrapper seguro que captura exceções de _run."""
        try:
            return self._run(payload)
        except Exception as exc:
            logger.warning(
                "SKILL_ERROR: skill=%s error=%s",
                self.name,
                str(exc),
            )
            return {}

    def _run(self, payload: dict) -> dict:
        """Implementar na subclasse."""
        return {}


# ============================================================
# REGISTRY DE SKILLS
# ============================================================

class SkillRegistry:
    """
    Registry de skills analíticas disponíveis.

    Uso:
        registry = SkillRegistry()
        registry.register(MySkill())
        results = registry.run_all(payload)
    """

    def __init__(self) -> None:
        self._skills: dict[str, BaseAnalyticalSkill] = {}

    def register(self, skill: BaseAnalyticalSkill) -> None:
        """Registra uma skill pelo nome."""
        self._skills[skill.name] = skill
        logger.debug("SKILL_REGISTERED: %s v%s", skill.name, skill.version)

    def unregister(self, name: str) -> None:
        """Remove uma skill do registry."""
        self._skills.pop(name, None)

    def get(self, name: str) -> BaseAnalyticalSkill | None:
        return self._skills.get(name)

    def run_all(
        self,
        payload: dict,
        cost_limit: str = "medium",
    ) -> dict[str, Any]:
        """
        Roda todas as skills elegíveis para o contexto atual.

        Args:
            payload: payload compacto já montado
            cost_limit: nível máximo de custo permitido
                        'low' → só skills low
                        'medium' → low + medium
                        'high' → todas

        Returns:
            dict com resultados por nome de skill
        """
        _cost_order = {"low": 0, "medium": 1, "high": 2}
        max_cost = _cost_order.get(cost_limit, 1)

        results: dict[str, Any] = {}

        for name, skill in self._skills.items():
            skill_cost = _cost_order.get(skill.cost_class, 0)
            if skill_cost > max_cost:
                continue

            if not skill.should_run(payload):
                continue

            result = skill.execute(payload)
            if result:
                results[name] = result

        return results

    @property
    def registered(self) -> list[str]:
        return list(self._skills.keys())


# ============================================================
# INSTÂNCIA GLOBAL
# ============================================================

_registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    """Retorna o registry global de skills."""
    return _registry


def register_skill(skill: BaseAnalyticalSkill) -> None:
    """Registra uma skill no registry global."""
    _registry.register(skill)


def run_skills(
    payload: dict,
    cost_limit: str = "medium",
) -> dict[str, Any]:
    """
    Roda todas as skills elegíveis e retorna resultados.

    Integração com build_compact_payload:

        skills_results = run_skills(payload)
        if skills_results:
            payload["skills"] = skills_results

    Args:
        payload: payload compacto já montado
        cost_limit: 'low' | 'medium' | 'high'

    Returns:
        dict com resultados por skill — vazio se nenhuma skill rodou
    """
    return _registry.run_all(payload, cost_limit=cost_limit)


# ============================================================
# EXEMPLO DE SKILL CONCRETA
# (mostra o padrão que skills futuras devem seguir)
# ============================================================

class MeanReversionContextSkill(BaseAnalyticalSkill):
    """
    Skill de exemplo: contexto de mean reversion.

    Ativa em regime MR e RB.
    Combina hurst, reg.pos, mr_score (se disponível) em uma leitura.
    """

    name = "mean_reversion_context"
    version = "1.0"
    cost_class = "low"
    active_regimes: set[str] = {"MR", "RB"}
    active_triggers: set[str] = {"ALL"}

    def _run(self, payload: dict) -> dict:
        ext = payload.get("ext", {})
        hurst = ext.get("hurst")
        reg = ext.get("reg", {})
        pos = reg.get("pos") if isinstance(reg, dict) else None
        mr = payload.get("mr", {})
        mr_score = mr.get("score")
        mr_sig = mr.get("sig", "")

        if hurst is None and pos is None:
            return {}

        result: dict[str, Any] = {}

        if hurst is not None:
            h = float(hurst)
            if h < 0.45:
                result["tendency"] = "mean_reverting"
                result["strength"] = round((0.5 - h) * 2, 2)
            elif h > 0.55:
                result["tendency"] = "trending"
                result["strength"] = round((h - 0.5) * 2, 2)
            else:
                result["tendency"] = "random_walk"
                result["strength"] = 0.0

        if pos is not None:
            p = float(pos)
            if p < 0.2:
                result["channel_pos"] = "near_low"
            elif p > 0.8:
                result["channel_pos"] = "near_high"
            else:
                result["channel_pos"] = "mid_channel"

        if mr_score is not None:
            result["mr_score"] = mr_score
        if mr_sig:
            result["signal"] = mr_sig

        return result


class AbsorptionStrengthSkill(BaseAnalyticalSkill):
    """
    Skill de exemplo: força da absorção.

    Ativa em qualquer regime com trigger ABS ou AT.
    Qualifica se a absorção atual é operável.
    """

    name = "absorption_strength"
    version = "1.0"
    cost_class = "low"
    active_regimes: set[str] = {"ALL"}
    active_triggers: set[str] = {"ABS", "AT"}

    def _run(self, payload: dict) -> dict:
        flow = payload.get("flow", {})
        abs_buy = flow.get("abs_buy_str") or 0.0
        abs_sell = flow.get("abs_sell_exh") or 0.0
        abs_cont = flow.get("abs_cont") or 0.0
        pa = str(flow.get("pa", "")).lower()

        if abs_buy == 0 and abs_sell == 0:
            return {}

        quality = "weak"
        if abs_buy > 7.0 and abs_sell < 1.5:
            quality = "strong_buy"
        elif abs_sell > 7.0 and abs_buy < 1.5:
            quality = "strong_sell"
        elif abs_buy > 5.0 and abs_sell < 2.5:
            quality = "moderate_buy"
        elif abs_sell > 5.0 and abs_buy < 2.5:
            quality = "moderate_sell"
        elif abs_buy > 3.0 or abs_sell > 3.0:
            quality = "developing"

        operable = quality in ("strong_buy", "strong_sell", "moderate_buy", "moderate_sell")

        return {
            "quality": quality,
            "operable": operable,
            "cont_prob": round(abs_cont, 2) if abs_cont else 0.0,
            "signal": pa[:10] if pa else "unknown",
        }


# Registrar as skills de exemplo no registry global
register_skill(MeanReversionContextSkill())
register_skill(AbsorptionStrengthSkill())
