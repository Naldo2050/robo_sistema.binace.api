from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

from scripts.app import app

client = TestClient(app)


def _sentinel(record):
    def fake_run(*args, **kwargs):
        record.append((args, kwargs))
        cmd = args[0] if args else kwargs.get("args")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="sentinel-stdout",
            stderr="",
        )

    return fake_run


@pytest.mark.parametrize("enabled", [None, "false", "1", "TRUE"])
def test_flag_missing_or_not_true_returns_403(monkeypatch, enabled):
    if enabled is None:
        monkeypatch.delenv("AUTO_FIXER_ENABLED", raising=False)
    else:
        monkeypatch.setenv("AUTO_FIXER_ENABLED", enabled)
    monkeypatch.setenv("AUTO_FIXER_TOKEN", "expected-token")
    calls = []
    monkeypatch.setattr(subprocess, "run", _sentinel(calls))

    response = client.post(
        "/auto-fixer/run",
        headers={"X-Auto-Fixer-Token": "expected-token"},
    )

    assert response.status_code == 403
    assert calls == []


@pytest.mark.parametrize("server_token", [None, ""])
def test_enabled_with_missing_or_empty_server_token_returns_503(monkeypatch, server_token):
    monkeypatch.setenv("AUTO_FIXER_ENABLED", "true")
    if server_token is None:
        monkeypatch.delenv("AUTO_FIXER_TOKEN", raising=False)
    else:
        monkeypatch.setenv("AUTO_FIXER_TOKEN", server_token)
    calls = []
    monkeypatch.setattr(subprocess, "run", _sentinel(calls))

    response = client.post(
        "/auto-fixer/run",
        headers={"X-Auto-Fixer-Token": "any-token"},
    )

    assert response.status_code == 503
    assert calls == []


@pytest.mark.parametrize("header_value", [None, "", "wrong-token"])
def test_client_token_missing_or_wrong_returns_401(monkeypatch, header_value):
    monkeypatch.setenv("AUTO_FIXER_ENABLED", "true")
    monkeypatch.setenv("AUTO_FIXER_TOKEN", "expected-token")
    calls = []
    monkeypatch.setattr(subprocess, "run", _sentinel(calls))

    headers = {} if header_value is None else {"X-Auto-Fixer-Token": header_value}
    response = client.post("/auto-fixer/run", headers=headers)

    assert response.status_code == 401
    assert calls == []


def test_correct_token_reaches_subprocess_with_expected_args(monkeypatch):
    monkeypatch.setenv("AUTO_FIXER_ENABLED", "true")
    monkeypatch.setenv("AUTO_FIXER_TOKEN", "expected-token")
    calls = []
    monkeypatch.setattr(subprocess, "run", _sentinel(calls))

    response = client.post(
        "/auto-fixer/run?phase=2",
        headers={"X-Auto-Fixer-Token": "expected-token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "stdout": "sentinel-stdout",
        "stderr": "",
    }
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == ["python", "-m", "auto_fixer.runner", "--phase", "2"]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["timeout"] == 300
