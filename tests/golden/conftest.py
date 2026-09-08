# tests/golden/conftest.py — harness determinístico Golden Windows V1.
#
# Regras:
# - relógio congelado (FakeClock/FakeMonotonic injetados; nunca wall real);
# - estado global resetado por teste (StateManager, adaptive, caches, ctx);
# - guarda sem-rede: qualquer HTTP/rede explode (prova "nenhum retry/sleep
#   de rede no hot path" junto com o contador de sleep escopado abaixo);
# - time.sleep NÃO é bloqueado globalmente (não quebrar cleanup de libs);
#   record_project_sleeps() prova zero sleeps de código do projeto no caminho.
# - sem Groq/LLM, sem ordens.

import inspect
import json
import math
import socket
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FakeClock:
    """Clock congelado p/ FlowAnalyzer/DataPipeline/TimeManager."""

    def __init__(self, start_ms=1_757_000_000_000):
        self._now = int(start_ms)

    def now_ms(self):
        return self._now

    def advance(self, ms):
        self._now += int(ms)

    def build_time_index(self, ts_ms, include_local=False, timespec="milliseconds"):
        return {"timestamp_utc": str(ts_ms), "epoch_ms": int(ts_ms)}

    def format_timestamp(self, ts_ms):
        return str(ts_ms)

    def from_timestamp_ms(self, ts_ms, tz=None):
        return None


class FakeMonotonic:
    def __init__(self, start=1000.0):
        self._t = float(start)

    def __call__(self):
        return self._t

    def advance(self, s):
        self._t += float(s)


def load_fixture(name):
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


def gen_trades(spec):
    """Gera trades determinísticos {p,q,T,m} (m=True => sell)."""
    tcfg = spec["trades"]
    t0 = int(tcfg["t0_ms"])
    step = int(tcfg.get("step_ms", 500))
    out = []
    i = 0

    def _push(side, qty, price):
        nonlocal i
        out.append({"p": float(price), "q": float(qty),
                    "T": t0 + i * step, "m": side == "sell"})
        i += 1

    mode = tcfg.get("mode", "interleaved")
    if mode == "sine":  # GW5: faixa estreita determinística, sem RNG
        base = float(tcfg["price_base"])
        amp = float(tcfg.get("amp", 30.0))
        period = float(tcfg.get("period", 50.0))
        n = int(tcfg["n"])
        qty = float(tcfg.get("qty", 0.01))
        for k in range(n):
            price = base + amp * math.sin(k / period)
            _push("sell" if k % 2 else "buy", qty, price)
        return out
    nb = int(tcfg.get("n_buy", 0))
    ns = int(tcfg.get("n_sell", 0))
    qb = float(tcfg.get("qty_buy", 0.01))
    qs = float(tcfg.get("qty_sell", 0.01))
    flat = float(tcfg.get("price_flat", spec.get("price_base", 65000.0)))
    buys = ["buy"] * nb
    sells = ["sell"] * ns
    if mode == "blocks":
        seq = buys + sells
    else:  # interleaved determinístico
        seq = []
        ib = is_ = 0
        while ib < nb or is_ < ns:
            if ib < nb:
                seq.append("buy")
                ib += 1
            if is_ < ns:
                seq.append("sell")
                is_ += 1
    for side in seq:
        _push(side, qb if side == "buy" else qs, flat)
    return out


def expected_volumes(spec):
    tcfg = spec["trades"]
    if tcfg.get("mode") == "sine":
        n = int(tcfg["n"])
        qty = float(tcfg.get("qty", 0.01))
        nb = (n + 1) // 2
        ns = n // 2
        return nb * qty, ns * qty
    buy = int(tcfg.get("n_buy", 0)) * float(tcfg.get("qty_buy", 0.01))
    sell = int(tcfg.get("n_sell", 0)) * float(tcfg.get("qty_sell", 0.01))
    return buy, sell


def assert_no_nonfinite(obj, path="$"):
    if isinstance(obj, float):
        assert not (obj != obj or obj in (float("inf"), float("-inf"))), path
    elif isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_nonfinite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for j, v in enumerate(obj):
            assert_no_nonfinite(v, f"{path}[{j}]")


def assert_json_strict(payload):
    import json as _json
    _json.dumps(payload, allow_nan=False)


def assert_contract_version(values, expected, where=""):
    """expected_contract_version da fixture vs produzido; mismatch = migração."""
    got = None
    if isinstance(values, dict):
        got = values.get("correlation_contract_version",
                         values.get("cross_asset_contract_version"))
    assert got == expected, (
        f"contract migration necessária {where}: esperado "
        f"expected_contract_version={expected}, produzido={got}"
    )


class ProjectSleepRecorder:
    """Conta time.sleep chamados a partir de código do projeto (não libs)."""

    def __init__(self, real_sleep):
        self.real_sleep = real_sleep
        self.calls = []

    def __call__(self, seconds):
        for frame in inspect.getouterframes(inspect.currentframe(), 2):
            fname = frame.filename.replace("\\", "/")
            if fname.startswith(str(REPO_ROOT).replace("\\", "/")) and "/tests/" not in fname:
                self.calls.append((fname, frame.lineno, seconds))
                break


@pytest.fixture
def frozen_state(monkeypatch):
    """Reseta estado global mutável entre cenários."""
    from core.state_manager import StateManager
    from data_pipeline.pipeline import DataPipeline
    import market_analysis.cross_asset_correlations as ca

    StateManager._instance = None
    DataPipeline._shared_adaptive_thresholds.clear()
    ca._CORR_CACHE.clear()
    import market_orchestrator.ai.payload_builder_compact as bcp
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0
    yield
    StateManager._instance = None
    DataPipeline._shared_adaptive_thresholds.clear()
    ca._CORR_CACHE.clear()
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0


def _boom(*args, **kwargs):
    raise AssertionError("rede proibida no Golden (HTTP/border externa)")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Guarda forte: qualquer HTTP/rede explode. time.sleep segue permitido."""
    import requests
    import urllib.request

    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "request", _boom)
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    try:
        import yfinance as yf
        monkeypatch.setattr(yf.Ticker, "history", _boom)
    except ImportError:
        pass
    try:
        import aiohttp
        monkeypatch.setattr(aiohttp.ClientSession, "_request", _boom)
    except ImportError:
        pass
    yield


@pytest.fixture
def project_sleeps(monkeypatch):
    """Prova zero sleeps de código do projeto no caminho (libs intactas)."""
    import time as _time

    rec = ProjectSleepRecorder(_time.sleep)
    monkeypatch.setattr(_time, "sleep", rec)
    return rec
