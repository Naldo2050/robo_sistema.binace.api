# config/env_policy.py
# -*- coding: utf-8 -*-
"""PF-D: política ÚNICA e centralizada de bootstrap de ambiente.

Contrato de carregamento do `.env`:
  - `LOAD_DOTENV` (default True = compatibilidade dev) controla se o `.env`
    é carregado. Parser booleano robusto:
      True  <- "1", "true", "yes", "on" (case-insensitive, com strip);
      False <- "0", "false", "no", "off";
      qualquer outro valor -> ValueError (fail-closed, erro de config
      explícito; o valor impresso é o da FLAG, nunca credencial).
  - `OBSERVATION_MODE=1` implica `.env` NUNCA carregado, independente de
    `LOAD_DOTENV` (defesa em profundidade).
  - NÃO depende de `PYTHON_DOTENV_DISABLED` (comprovado incompatível com o
    python-dotenv instalado — `load_dotenv()` resolve o `.env` a partir do
    arquivo chamador, não do cwd).

Contrato do `OBSERVATION_MODE=1` (defesa em profundidade):
  - Aborta startup ANTES do bot se qualquer credencial de trading estiver
    visível (env ou config): nunca apaga silenciosamente, nunca imprime
    valores/prefixos — só NOMES na mensagem de erro.
  - IA/Groq/OpenAI configurada -> aborta (não corrige silenciosamente).
  - `HYBRID_ENABLED` / `EXECUTION_ENABLED` truthy -> aborta.
  - Retorna True quando o modo está ativo e o processo está seguro;
    retorna False quando o modo está desligado (no-op total).

Todos os pontos produtivos que carregavam `.env` incondicionalmente devem
usar `maybe_load_dotenv()` (D4: mesma política em todos os pontos).
"""

from __future__ import annotations

import os
from typing import Optional

TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
FALSE_TOKENS = frozenset({"0", "false", "no", "off"})

TRADING_CREDENTIAL_NAMES = (
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "BINANCE_SECRET_KEY",
)
AI_CREDENTIAL_NAMES = (
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
)


def parse_env_bool(name: str, default: bool = True) -> bool:
    """Parser booleano robusto para flags de ambiente.

    Valor inválido -> ValueError (fail-closed). `name` e o valor da FLAG
    aparecem na mensagem; esta função nunca recebe credenciais.
    """
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    token = raw.strip().lower()
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    raise ValueError(
        f"PF-D: env {name}={raw!r} inválido; use um de "
        f"{sorted(TRUE_TOKENS)} (verdadeiro) ou {sorted(FALSE_TOKENS)} (falso)."
    )


def observation_mode() -> bool:
    """True somente com OBSERVATION_MODE explicitamente verdadeiro."""
    return parse_env_bool("OBSERVATION_MODE", default=False)


def should_load_dotenv() -> bool:
    """Observation nunca carrega; senão, obedece LOAD_DOTENV (default True)."""
    if observation_mode():
        return False
    return parse_env_bool("LOAD_DOTENV", default=True)


def maybe_load_dotenv() -> bool:
    """Carregamento guardado e centralizado. Retorna True se carregou."""
    if not should_load_dotenv():
        return False
    from dotenv import load_dotenv

    load_dotenv()
    return True


def _present_names(names, settings=None) -> list:
    """NOMES presentes (env ou attrs de config). Nunca valores."""
    found = [n for n in names if os.getenv(n)]
    if settings is not None:
        for n in names:
            try:
                if getattr(settings, n, None):
                    found.append(n)
            except Exception:
                pass
    return sorted(set(found))


def assert_observation_safe(settings=None) -> bool:
    """Impõe o contrato do OBSERVATION_MODE. Ver docstring do módulo.

    Raises:
        RuntimeError: modo ativo e processo inseguro (mensagem com NOMES).
        ValueError: flag booleana inválida (via parse_env_bool).
    """
    if not observation_mode():
        return False
    trading = _present_names(TRADING_CREDENTIAL_NAMES, settings)
    if trading:
        raise RuntimeError(
            "Observation mode refuses to start with trading credentials "
            f"present: {','.join(trading)}"
        )
    ai = _present_names(AI_CREDENTIAL_NAMES, settings)
    if ai:
        raise RuntimeError(
            "Observation mode refuses to start with AI credentials "
            f"configured: {','.join(ai)}"
        )
    if settings is not None:
        if bool(getattr(settings, "HYBRID_ENABLED", False)):
            raise RuntimeError(
                "Observation mode refuses to start with HYBRID_ENABLED on"
            )
        if bool(getattr(settings, "EXECUTION_ENABLED", False)):
            raise RuntimeError(
                "Observation mode refuses to start with EXECUTION_ENABLED on"
            )
    return True


def observation_safety_report(settings=None) -> dict:
    """Relatório só-booleanos (para logs/preflight). Nunca valores."""
    try:
        active = observation_mode()
    except ValueError:
        active = None
    return {
        "observation_mode": active,
        "load_dotenv": None if active is None else should_load_dotenv(),
        "trading_credential_present": bool(
            _present_names(TRADING_CREDENTIAL_NAMES, settings)),
        "ai_credential_present": bool(
            _present_names(AI_CREDENTIAL_NAMES, settings)),
    }
