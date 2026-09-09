# tests/unit/test_env_policy.py
"""PF-D5: política de bootstrap de ambiente — testes herméticos.

- Usa SOMENTE credenciais falsas (`FAKE_*_FOR_TESTS`), nunca reais.
- Casos com `.env`/imports rodam em subprocesso com env controlado e
  cwd=tmp (o `.env` real do repo nunca é lido nem impresso).
- Nenhum teste imprime valores — só booleanos e NOMES.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from config.env_policy import (
    AI_CREDENTIAL_NAMES,
    TRADING_CREDENTIAL_NAMES,
    assert_observation_safe,
    observation_mode,
    parse_env_bool,
    should_load_dotenv,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

FAKE_ENV_FILE = (
    "BINANCE_API_KEY=FAKE_KEY_FOR_TESTS_NOT_REAL\n"
    "BINANCE_API_SECRET=FAKE_SECRET_FOR_TESTS_NOT_REAL\n"
)

CONTROLLED_NAMES = (
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "BINANCE_SECRET_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "LOAD_DOTENV",
    "OBSERVATION_MODE",
    "PYTHON_DOTENV_DISABLED",
)


def _clean_env(extra=None):
    env = dict(os.environ)
    for n in CONTROLLED_NAMES:
        env.pop(n, None)
    if extra:
        env.update(extra)
    return env


def _run_child(code, cwd, extra_env):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd,
        env=_clean_env(extra_env),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _write_fake_dotenv(path):
    with open(os.path.join(path, ".env"), "w", encoding="utf-8") as f:
        f.write(FAKE_ENV_FILE)


# H) matriz booleana + inválido (fail-closed documentado).
@pytest.mark.parametrize("raw,expected", [
    ("0", False), ("false", False), ("FALSE", False), ("off", False),
    ("OFF", False), ("no", False), ("No", False),
    ("1", True), ("true", True), ("TRUE", True), ("on", True),
    ("ON", True), ("yes", True), ("Yes", True),
    ("  true  ", True), (" 0 ", False),
])
def test_H_parse_env_bool_matrix(monkeypatch, raw, expected):
    monkeypatch.setenv("LOAD_DOTENV", raw)
    assert parse_env_bool("LOAD_DOTENV") is expected


def test_H_invalid_bool_fails_closed(monkeypatch):
    monkeypatch.setenv("LOAD_DOTENV", "maybe")
    with pytest.raises(ValueError):
        parse_env_bool("LOAD_DOTENV")
    monkeypatch.setenv("OBSERVATION_MODE", "")
    with pytest.raises(ValueError):
        observation_mode()


def test_H_defaults(monkeypatch):
    for n in ("LOAD_DOTENV", "OBSERVATION_MODE"):
        monkeypatch.delenv(n, raising=False)
    assert should_load_dotenv() is True  # compat dev
    assert observation_mode() is False


# A) LOAD_DOTENV=0 + .env (fake) presente -> nada reaparece após imports.
def test_A_no_dotenv_no_reintroduce(tmp_path):
    _write_fake_dotenv(str(tmp_path))
    proc = _run_child(
        """
        import sys
        sys.path.insert(0, %r)
        import os
        import config.settings as s
        import main
        names = %r
        env_now = {n: (n in os.environ) for n in names}
        cfg = {n: bool(getattr(s, n, None)) for n in
               ('BINANCE_API_KEY', 'BINANCE_API_SECRET')}
        assert not any(env_now.values()), env_now
        assert not any(cfg.values()), cfg
        print('A_OK')
        """
        % (REPO_ROOT, list(CONTROLLED_NAMES[:5])),
        cwd=str(tmp_path),
        extra_env={"LOAD_DOTENV": "0"},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "A_OK" in proc.stdout


# B) LOAD_DOTENV=1 -> comportamento dev compatível (lê env normalmente).
def test_B_load_enabled_dev_compat(tmp_path):
    _write_fake_dotenv(str(tmp_path))
    proc = _run_child(
        """
        import sys
        sys.path.insert(0, %r)
        from config.env_policy import maybe_load_dotenv, should_load_dotenv
        import config.settings as s
        assert should_load_dotenv() is True
        assert maybe_load_dotenv() is True
        # env explícito continua valendo (dotenv não sobrescreve):
        assert s.BINANCE_API_KEY == 'FAKE_PRESET_VIA_ENV'
        print('B_OK')
        """
        % REPO_ROOT,
        cwd=str(tmp_path),
        extra_env={
            "LOAD_DOTENV": "1",
            "BINANCE_API_KEY": "FAKE_PRESET_VIA_ENV",
        },
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "B_OK" in proc.stdout


def _obs_child(extra_env, body):
    script = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "import config.settings as s\n"
        "from config.env_policy import assert_observation_safe\n"
        "%s\n"
    ) % (REPO_ROOT, textwrap.dedent(body))
    return _run_child(
        script,
        cwd=None,  # herda cwd do pytest; sem .env fake: só env conta
        extra_env=extra_env,
    )


# C) observation sem credenciais -> passa.
def test_C_observation_clean_passes():
    proc = _obs_child(
        {"OBSERVATION_MODE": "1", "LOAD_DOTENV": "0"},
        "assert assert_observation_safe(s) is True\nprint('C_OK')",
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "C_OK" in proc.stdout


# D/E) observation + credencial fake -> aborta (só NOMES no erro).
@pytest.mark.parametrize("name", ["BINANCE_API_KEY", "BINANCE_API_SECRET"])
def test_DE_observation_trading_credential_aborts(name):
    proc = _obs_child(
        {"OBSERVATION_MODE": "1", "LOAD_DOTENV": "0",
         name: "FAKE_FOR_TESTS"},
        "try:\n"
        "    assert_observation_safe(s)\n"
        "    raise SystemExit('NO_ABORT')\n"
        "except RuntimeError as e:\n"
        "    assert 'FAKE' not in str(e), 'valor vazou no erro!'\n"
        "    print('ABORT_OK')\n",
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ABORT_OK" in proc.stdout


# F) observation + GROQ/OPENAI fake -> aborta (não corrige silenciosamente).
@pytest.mark.parametrize("name", ["GROQ_API_KEY", "OPENAI_API_KEY"])
def test_F_observation_ai_credential_aborts(name):
    proc = _obs_child(
        {"OBSERVATION_MODE": "1", "LOAD_DOTENV": "0",
         name: "FAKE_FOR_TESTS"},
        "try:\n"
        "    assert_observation_safe(s)\n"
        "    raise SystemExit('NO_ABORT')\n"
        "except RuntimeError as e:\n"
        "    assert 'FAKE' not in str(e), 'valor vazou no erro!'\n"
        "    print('ABORT_OK')\n",
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ABORT_OK" in proc.stdout


# G) observation + hybrid/trading habilitado -> aborta.
@pytest.mark.parametrize("attr", ["HYBRID_ENABLED", "EXECUTION_ENABLED"])
def test_G_observation_hybrid_trading_aborts(attr):
    proc = _obs_child(
        {"OBSERVATION_MODE": "1", "LOAD_DOTENV": "0"},
        "s.%s = True\n"
        "try:\n"
        "    assert_observation_safe(s)\n"
        "    raise SystemExit('NO_ABORT')\n"
        "except RuntimeError:\n"
        "    print('ABORT_OK')\n" % attr,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ABORT_OK" in proc.stdout


def test_guard_noop_when_mode_off(monkeypatch):
    monkeypatch.delenv("OBSERVATION_MODE", raising=False)
    assert assert_observation_safe(None) is False
