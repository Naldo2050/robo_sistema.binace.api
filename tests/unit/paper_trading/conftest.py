# tests/unit/paper_trading/conftest.py
"""
Harness fixtures for hermetic paper trading tests.

Enforces zero-network policy and provides deterministic clock utilities.
"""

from __future__ import annotations

import socket
import pytest


class FakeClock:
    """Deterministic simulated clock."""

    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        self._now_ms = int(start_ms)

    def now_ms(self) -> int:
        return self._now_ms

    def advance(self, ms: int) -> None:
        self._now_ms += int(ms)


def _network_forbidden(*args, **kwargs):
    raise AssertionError("Network access is strictly forbidden in hermetic tests.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Guards against any real network access."""
    import urllib.request

    try:
        import requests
        monkeypatch.setattr(requests, "get", _network_forbidden)
        monkeypatch.setattr(requests, "post", _network_forbidden)
        monkeypatch.setattr(requests, "request", _network_forbidden)
    except ImportError:
        pass

    monkeypatch.setattr(urllib.request, "urlopen", _network_forbidden)
    monkeypatch.setattr(socket, "create_connection", _network_forbidden)

    try:
        import aiohttp
        monkeypatch.setattr(aiohttp.ClientSession, "_request", _network_forbidden)
    except ImportError:
        pass

    yield


@pytest.fixture
def fake_clock():
    return FakeClock()
