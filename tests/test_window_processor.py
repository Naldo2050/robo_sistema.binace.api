# tests/test_window_processor.py
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from market_orchestrator.windows import window_processor as wp


# ==== Stubs / Fakes ====

@dataclass
class ContextCollectorStub:
    def get_context(self) -> Dict[str, Any]:
        # perfil mínimo para não disparar paths de erro
        return {
            "historical_vp": {
                "daily": {
                    "val": 1.0,
                    "vah": 2.0,
                    "poc": 1.5,
                }
            },
            "mtf_trends": {},
            "derivatives": {},
            "market_context": {},
            "market_environment": {},
        }


@dataclass
class LevelsStub:
    updated_with: Optional[Dict[str, Any]] = None

    def update_from_vp(self, vp: Dict[str, Any]) -> None:
        self.updated_with = vp


@dataclass
class FlowAnalyzerStub:
    def get_flow_metrics(self, reference_epoch_ms: int) -> Dict[str, Any]:
        return {
            "order_flow": {},
            "sector_flow": {},
            "data_quality": {"flow_trades_count": 10},
        }


@dataclass
class HealthMonitorStub:
    count: int = 0

    def heartbeat(self, name: str) -> None:
        self.count += 1


@dataclass
class FeatureStoreStub:
    saved: List[Dict[str, Any]] = field(default_factory=list)

    def save_features(self, window_id: str, features: Dict[str, Any]) -> None:
        self.saved.append({"window_id": window_id, "features": features})

    def close(self) -> None:
        pass


@dataclass
class FakePipeline:
    """
    Fake de DataPipeline para isolar o teste de window_processor.
    Implementa apenas o necessário.
    """
    data: List[Dict[str, Any]]
    symbol: str
    time_manager: Any
    enriched: Dict[str, Any] = field(default_factory=dict)
    context_added: Dict[str, Any] = field(default_factory=dict)
    closed: bool = False

    def __init__(self, data, symbol, time_manager=None):
        self.data = data
        self.symbol = symbol
        self.time_manager = time_manager
        self.enriched = {}
        self.context_added = {}
        self.closed = False

    def enrich(self) -> Dict[str, Any]:
        total_vol = float(sum(t.get("q", 0.0) for t in self.data))
        close = float(self.data[-1]["p"]) if self.data else 0.0
        self.enriched = {
            "volume_total": total_vol,
            "ohlc": {"close": close},
        }
        return self.enriched

    def add_context(self, **kwargs) -> None:
        self.context_added.update(kwargs)

    def detect_signals(self, absorption_detector, exhaustion_detector, orderbook_data):
        # Não chamamos de fato os detectores nos testes
        self.last_orderbook_data = orderbook_data
        self.signals = [
            {
                "tipo_evento": "TEST_SIGNAL",
                "resultado_da_batalha": "TEST",
                "descricao": "",
            }
        ]
        return self.signals

    def get_final_features(self) -> Dict[str, Any]:
        return {"dummy_feature": 1.0}

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeBot:
    symbol: str = "BTCUSDT"
    window_size_minutes: int = 1
    window_ms: int = 60_000
    should_stop: bool = False

    # estado de janelas
    window_data: List[Dict[str, Any]] = field(default_factory=list)
    window_end_ms: int = 0
    window_count: int = 0

    # warmup
    warming_up: bool = False
    warmup_windows_remaining: int = 0
    warmup_windows_required: int = 3
    _warmup_lock: threading.Lock = field(default_factory=threading.Lock)

    # buffers / históricos
    trades_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    min_trades_for_pipeline: int = 3
    delta_history: deque = field(default_factory=lambda: deque(maxlen=100))
    delta_std_dev_factor: float = 2.0
    volume_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # componentes stub
    context_collector: Any = field(default_factory=ContextCollectorStub)
    levels: Any = field(default_factory=LevelsStub)
    flow_analyzer: Any = field(default_factory=FlowAnalyzerStub)
    health_monitor: Any = field(default_factory=HealthMonitorStub)
    feature_store: Any = field(default_factory=FeatureStoreStub)
    time_manager: Any = None  # será injetado pelo fixture tm

    # orderbook / vp
    last_valid_vp: Optional[Dict[str, Any]] = None
    last_valid_vp_time: float = 0.0

    # IA/sinais
    last_process_signals_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.time_manager is None:
            # será sobrescrito no teste com fixture tm
            class _DummyTM:
                def now_ms(self): return 0
            self.time_manager = _DummyTM()

    def _process_signals(
        self,
        signals,
        pipeline,
        flow_metrics,
        historical_profile,
        macro_context,
        ob_event,
        enriched,
        close_ms,
        total_buy_volume,
        total_sell_volume,
        valid_window_data,
    ):
        # Guarda os parâmetros para inspeção no teste
        self.last_process_signals_args = {
            "signals": signals,
            "pipeline": pipeline,
            "flow_metrics": flow_metrics,
            "historical_profile": historical_profile,
            "macro_context": macro_context,
            "ob_event": ob_event,
            "enriched": enriched,
            "close_ms": close_ms,
            "total_buy_volume": total_buy_volume,
            "total_sell_volume": total_sell_volume,
            "valid_window_data": valid_window_data,
        }


# ==== Testes ====


@pytest.mark.asyncio
async def test_process_window_warmup_does_not_advance_window(monkeypatch, tm):
    """
    Quando bot.warming_up=True, process_window deve apenas
    decrementar warmup_windows_remaining e não criar pipeline
    nem avançar window_count.
    """
    bot = FakeBot()
    bot.time_manager = tm
    bot.window_data = [{"p": 100.0, "q": 1.0, "T": tm.now_ms()}]
    bot.warming_up = True
    bot.warmup_windows_remaining = bot.warmup_windows_required

    # Se DataPipeline for chamado, queremos saber (falha)
    def _boom(*args, **kwargs):
        raise AssertionError("DataPipeline não deve ser chamado em warmup")

    monkeypatch.setattr(wp, "DataPipeline", _boom)

    # Também não deve chamar orderbook
    def _fetch_boom(*args, **kwargs):
        raise AssertionError("fetch_orderbook_with_retry não deve ser chamado em warmup")

    monkeypatch.setattr(wp, "fetch_orderbook_with_retry", _fetch_boom)

    wp.process_window(bot)

    assert bot.window_count == 0
    assert bot.warming_up is True  # ainda em aquecimento (faltam janelas)
    assert bot.window_data == []


def test_process_window_insufficient_trades_no_buffer(monkeypatch, tm):
    """
    Janela com poucos trades e buffer insuficiente deve apenas limpar window_data
    e não avançar window_count.
    """
    bot = FakeBot()
    bot.time_manager = tm
    bot.min_trades_for_pipeline = 3
    bot.window_data = [{"p": 100.0, "q": 1.0, "T": tm.now_ms()}]  # 1 trade
    bot.trades_buffer.clear()  # buffer vazio

    # DataPipeline não deve ser chamado
    def _boom(*args, **kwargs):
        raise AssertionError("DataPipeline não deve ser chamado com dados insuficientes")

    monkeypatch.setattr(wp, "DataPipeline", _boom)

    wp.process_window(bot)

    assert bot.window_count == 0
    assert bot.window_data == []


def test_process_window_uses_trades_buffer_when_insufficient(monkeypatch, tm):
    """
    Quando a janela tem poucos trades mas o trades_buffer tem dados suficientes,
    process_window deve usar o buffer e ainda assim processar a janela.
    """
    bot = FakeBot()
    bot.time_manager = tm
    bot.min_trades_for_pipeline = 3

    # Apenas 1 trade na janela
    bot.window_data = [{"p": 100.0, "q": 1.0, "T": tm.now_ms()}]

    # Buffer com trades suficientes
    bot.trades_buffer.clear()
    for i in range(3):
        bot.trades_buffer.append({"p": 100.0 + i, "q": 1.0 + i, "T": tm.now_ms() + i})

    # Patch de DataPipeline -> FakePipeline
    monkeypatch.setattr(wp, "DataPipeline", FakePipeline)

    # Patch de fetch_orderbook_with_retry para contar chamadas
    calls = {"count": 0}

    def _fake_fetch(bot_arg, close_ms):
        calls["count"] += 1
        return {
            "is_valid": True,
            "orderbook_data": {"bid_depth_usd": 1000.0, "ask_depth_usd": 1000.0},
            "data_quality": {"data_source": "test"},
        }

    monkeypatch.setattr(wp, "fetch_orderbook_with_retry", _fake_fetch)

    # Stubs de detectors para não depender de data_handler real
    monkeypatch.setattr(
        wp,
        "create_absorption_event",
        lambda *a, **k: {"tipo_evento": "ABS", "resultado_da_batalha": "ABS_TEST"},
    )
    monkeypatch.setattr(
        wp,
        "create_exhaustion_event",
        lambda *a, **k: {"tipo_evento": "EXH", "resultado_da_batalha": "EXH_TEST"},
    )

    wp.process_window(bot)

    assert bot.window_count == 1
    assert calls["count"] == 1  # orderbook foi buscado
    # Deve ter salvo features pelo FakePipeline
    assert len(bot.feature_store.saved) == 1
    # Data usada no pipeline deve vir do buffer (3 trades)
    pipeline_instance: FakePipeline = bot.last_process_signals_args["pipeline"]
    assert len(pipeline_instance.data) == 3


def test_process_window_happy_path_calls_process_signals_and_feature_store(monkeypatch, tm):
    """
    Caminho feliz: janela com trades suficientes, DataPipeline fake,
    verifica se _process_signals e feature_store.save_features foram chamados.
    """
    bot = FakeBot()
    bot.time_manager = tm
    bot.min_trades_for_pipeline = 3

    # 3 trades válidos na janela
    now = tm.now_ms()
    bot.window_data = [
        {"p": 100.0, "q": 1.0, "T": now},
        {"p": 101.0, "q": 2.0, "T": now + 1},
        {"p": 102.0, "q": 3.0, "T": now + 2},
    ]
    bot.window_end_ms = now + 3

    # Patch de DataPipeline -> FakePipeline
    monkeypatch.setattr(wp, "DataPipeline", FakePipeline)

    # Orderbook fake
    def _fake_fetch(bot_arg, close_ms):
        return {
            "is_valid": True,
            "orderbook_data": {"bid_depth_usd": 2000.0, "ask_depth_usd": 1500.0},
            "data_quality": {"data_source": "live"},
        }

    monkeypatch.setattr(wp, "fetch_orderbook_with_retry", _fake_fetch)

    # Stubs de detectores
    monkeypatch.setattr(
        wp,
        "create_absorption_event",
        lambda *a, **k: {"tipo_evento": "ABS", "resultado_da_batalha": "ABS_TEST"},
    )
    monkeypatch.setattr(
        wp,
        "create_exhaustion_event",
        lambda *a, **k: {"tipo_evento": "EXH", "resultado_da_batalha": "EXH_TEST"},
    )

    wp.process_window(bot)

    # Janela foi processada
    assert bot.window_count == 1
    # _process_signals foi chamado e registrou os args
    assert bot.last_process_signals_args is not None
    # feature_store recebeu 1 save_features
    assert len(bot.feature_store.saved) == 1
    saved = bot.feature_store.saved[0]
    assert saved["window_id"].startswith(bot.symbol)
    assert "dummy_feature" in saved["features"]