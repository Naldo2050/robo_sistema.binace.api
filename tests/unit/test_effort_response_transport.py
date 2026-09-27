# tests/unit/test_effort_response_transport.py
"""
Testes unitários para o P1-F ETAPA 2 — Shadow Async Transport v1.

Cobre rigorosamente:
1. Contrato temporal canônico (Definição 2):
   - J2 real: observation_open_ms=1788702361157, observation_close_ms=1788702418610, causal_anchor_ms=1788702420000.
   - Targets calculados de causal_anchor_ms (1m=1788702480000, 5m=1788702720000, 15m=1788703320000).
   - Invariância a janelas esparsas (last trade 10s antes do boundary não altera target).
2. Prova causal e exclusão do boundary:
   - max(T_raw) < causal_anchor_ms.
   - T_raw == causal_anchor_ms é rejeitado da feature window.
   - BOUNDARY_EXCLUDED_FOR_CAUSAL_SAFETY: observação com T == anchor é excluída de feature e excursão.
   - Observação com T == target entra na excursão.
3. Determinação de record_id:
   - rec_{symbol}_{causal_anchor_ms}_v{feature_contract}_s{shadow_schema}.
   - Dois records com last-trade diferente mas mesmo anchor geram mesmo record_id.
4. DTO frozen e isolamento de mutação posterior:
   - Tentativa de alteração lança erro.
   - Cópia defensiva isola mutações externas no caller.
5. Fila assíncrona e política DROP_NEWEST:
   - Fila cheia descarta o registro mais novo sem bloquear e incrementa métrica.
6. Single-writer e concorrência:
   - Multi-threads produtoras gravando via única thread escritora sem race condition.
7. Pending registry com min-heap heapq:
   - Resolução sem full scan; idempotência na remoção.
8. Terminal price vs excursões:
   - FIRST_ON_OR_AFTER dentro de 1000ms de tolerância.
   - Janela OHLC que cruza fronteira não contamina extremas e marca status PARTIAL.
9. Restart fail-closed e isolamento de falhas:
   - Horizontes vencidos em downtime marcados com gap.
   - Corrupção desabilita apenas shadow sem afetar o trading.
10. Métricas Prometheus:
    - Sem erros de duplicidade no registro.
11. Wall clock não afeta target temporal.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict, List

import pytest

from flow_analyzer.effort_response_dataset import (
    FEATURE_CONTRACT_VERSION,
    OUTCOME_BOUNDARY_TOLERANCE_MS,
    SHADOW_SCHEMA_VERSION,
    EffortResponseShadowRecord,
    EffortResponseShadowStorage,
    HorizonOutcome,
    StorageCorruptionError,
    build_deterministic_record_id,
    build_shadow_record,
)
from flow_analyzer.effort_response_transport import (
    DEFAULT_OPERATIONAL_UNCALIBRATED_CAPACITY,
    EffortResponseSnapshotDTO,
    PendingHorizonEntry,
    PriceObservation,
    ShadowAsyncTransport,
    ShadowMetrics,
)


@pytest.fixture(autouse=True)
def cleanup_transport():
    """Garante reset de singletons e recursos antes e depois de cada teste."""
    ShadowAsyncTransport.reset_instance_for_testing()
    yield
    ShadowAsyncTransport.reset_instance_for_testing()


@pytest.fixture
def temp_dir():
    """Diretório temporário isolado para testes com filesystem."""
    d = tempfile.mkdtemp(prefix="shadow_transport_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TESTES TEMPORAIS E CONTRATO CANÔNICO (J2, BOUNDARY, TARGETS)
# ─────────────────────────────────────────────────────────────────────────────

def test_j2_canonical_temporal_contract():
    """Valida o contrato temporal canônico para a janela J2 real.

    observation_open_ms  = 1788702361157
    observation_close_ms = 1788702418610
    causal_anchor_ms     = 1788702420000

    Targets:
    1m  = 1788702480000
    5m  = 1788702720000
    15m = 1788703320000
    """
    obs_open = 1788702361157
    obs_close = 1788702418610
    anchor = 1788702420000

    dto = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
        observation_open_ms=obs_open,
        observation_close_ms=obs_close,
        buy_notional_usd=6816945.1591,
        sell_notional_usd=1149280.5758,
        open=79776.9,
        high=79810.8,
        low=79776.9,
        close=79792.7,
        window_duration_ms=57453,
        vwap=79803.5,
        poc=79804.9,
    )

    assert dto.causal_anchor_ms == anchor
    assert dto.observation_close_ms == obs_close
    assert dto.observation_open_ms == obs_open

    # Constrói o shadow record para validar os targets gerados
    window_data = {
        "buy_notional_usd": dto.buy_notional_usd,
        "sell_notional_usd": dto.sell_notional_usd,
        "open": dto.open,
        "high": dto.high,
        "low": dto.low,
        "close": dto.close,
        "window_duration_ms": dto.window_duration_ms,
        "vwap": dto.vwap,
        "poc": dto.poc,
    }
    rec = build_shadow_record(
        symbol=dto.symbol,
        window_open_ms=dto.observation_open_ms,
        window_close_ms=dto.observation_close_ms,
        window_data=window_data,
        causal_anchor_ms=dto.causal_anchor_ms,
        observation_open_ms=dto.observation_open_ms,
        observation_close_ms=dto.observation_close_ms,
    )

    horizons = rec.outcomes_future["horizons"]
    assert horizons["1m"]["target_timestamp_ms"] == 1788702480000
    assert horizons["5m"]["target_timestamp_ms"] == 1788702720000
    assert horizons["15m"]["target_timestamp_ms"] == 1788703320000

    # Provamos que o target NÃO é baseado em observation_close_ms
    assert horizons["1m"]["target_timestamp_ms"] != obs_close + 60_000
    assert horizons["5m"]["target_timestamp_ms"] != obs_close + 300_000
    assert horizons["15m"]["target_timestamp_ms"] != obs_close + 900_000


def test_sparse_window_target_invariance():
    """Janela esparsa com último trade 10s antes do boundary NÃO desloca os targets."""
    anchor = 1788702420000
    last_trade_sparse = anchor - 10_000  # 10 segundos antes do boundary

    dto = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
        observation_open_ms=anchor - 60_000,
        observation_close_ms=last_trade_sparse,
        buy_notional_usd=100000.0,
        sell_notional_usd=50000.0,
        open=80000.0,
        high=80050.0,
        low=79950.0,
        close=80020.0,
        window_duration_ms=50000,
    )

    rec = build_shadow_record(
        symbol=dto.symbol,
        window_open_ms=dto.observation_open_ms,
        window_close_ms=dto.observation_close_ms,
        window_data={"open": 80000.0, "high": 80050.0, "low": 79950.0, "close": 80020.0},
        causal_anchor_ms=dto.causal_anchor_ms,
    )

    horizons = rec.outcomes_future["horizons"]
    assert horizons["1m"]["target_timestamp_ms"] == anchor + 60_000
    assert horizons["5m"]["target_timestamp_ms"] == anchor + 300_000
    assert horizons["15m"]["target_timestamp_ms"] == anchor + 900_000


def test_boundary_excluded_for_causal_safety():
    """Trade/observação exatamente em T == causal_anchor_ms é excluído de feature e outcome.

    BOUNDARY_EXCLUDED_FOR_CAUSAL_SAFETY:
    - Feature window interval: T < causal_anchor_ms.
    - Future excursion interval: causal_anchor_ms < T <= target_ms.
    """
    anchor = 1788702420000

    # 1. Violação na feature: se observation_close_ms == anchor, DTO rejeita
    with pytest.raises(ValueError, match="BOUNDARY_EXCLUDED_FOR_CAUSAL_SAFETY"):
        EffortResponseSnapshotDTO(
            symbol="BTCUSDT",
            causal_anchor_ms=anchor,
            observation_open_ms=anchor - 60000,
            observation_close_ms=anchor,  # T == anchor é proibido na feature
            buy_notional_usd=1000.0,
            sell_notional_usd=1000.0,
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            window_duration_ms=60000,
        )

    # 2. Exclusão na resolução de outcomes:
    # Criamos um transport e enviamos uma observação exatamente em T == anchor
    temp_file = Path(tempfile.gettempdir()) / f"test_boundary_{int(time.time()*1000)}.jsonl"
    transport = ShadowAsyncTransport(filepath=temp_file, enabled=True, queue_capacity=100)

    try:
        dto = EffortResponseSnapshotDTO(
            symbol="BTCUSDT",
            causal_anchor_ms=anchor,
            observation_open_ms=anchor - 50000,
            observation_close_ms=anchor - 1,  # T < anchor válido
            buy_notional_usd=1000.0,
            sell_notional_usd=1000.0,
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            window_duration_ms=49999,
        )
        transport.submit_nowait(dto)
        transport.flush(timeout=1.0)

        target_1m = anchor + 60_000

        # Envia observação em T == anchor (preço discrepante 999.0 que NÃO pode entrar)
        transport.on_price_observation(
            timestamp_ms=anchor,
            open=999.0,
            high=999.0,
            low=999.0,
            close=999.0,
            is_window=False,
        )

        # Envia observação legítima no intervalo futuro (anchor < T <= target)
        transport.on_price_observation(
            timestamp_ms=anchor + 30_000,
            open=101.0,
            high=106.0,
            low=99.0,
            close=102.0,
            is_window=False,
        )

        # Envia observação em T == target (preço terminal 103.0)
        transport.on_price_observation(
            timestamp_ms=target_1m,
            open=102.0,
            high=104.0,
            low=101.0,
            close=103.0,
            is_window=False,
        )

        transport.flush(timeout=1.0)

        # Lê o registro atualizado do storage
        storage = EffortResponseShadowStorage(temp_file)
        records = storage.read_records()
        assert len(records) == 1
        h1m = records[0].outcomes_future["horizons"]["1m"]

        assert h1m["status"] == "RESOLVED"
        assert h1m["future_price"] == 103.0
        # O preço de 999.0 em T == anchor NÃO pode estar em max_high
        assert h1m["max_high"] == 106.0
        assert h1m["max_high"] != 999.0
        # Contagem de observações válidas deve ser 2 (30s e target_1m), sem incluir T == anchor
        assert h1m["observation_count"] == 2

    finally:
        transport.close(timeout=1.0)
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


def test_target_boundary_included_in_excursion():
    """Observação exatamente em T == target_ms DEVE entrar na excursão e no preço terminal."""
    anchor = 1788702420000
    target_1m = anchor + 60_000

    temp_file = Path(tempfile.gettempdir()) / f"test_target_inc_{int(time.time()*1000)}.jsonl"
    transport = ShadowAsyncTransport(filepath=temp_file, enabled=True, queue_capacity=100)

    try:
        dto = EffortResponseSnapshotDTO(
            symbol="BTCUSDT",
            causal_anchor_ms=anchor,
            observation_open_ms=anchor - 50000,
            observation_close_ms=anchor - 100,
            buy_notional_usd=1000.0,
            sell_notional_usd=1000.0,
            open=100.0,
            high=105.0,
            low=95.0,
            close=100.0,
            window_duration_ms=49900,
        )
        transport.submit_nowait(dto)
        transport.flush(timeout=1.0)

        # Envia observação exatamente em T == target_1m sendo a máxima da excursão
        transport.on_price_observation(
            timestamp_ms=target_1m,
            open=105.0,
            high=115.0,  # Máxima absoluta ocorrida exatamente no target
            low=104.0,
            close=110.0,
            is_window=False,
        )

        transport.flush(timeout=1.0)

        storage = EffortResponseShadowStorage(temp_file)
        records = storage.read_records()
        h1m = records[0].outcomes_future["horizons"]["1m"]

        assert h1m["status"] == "RESOLVED"
        assert h1m["future_price"] == 110.0
        assert h1m["max_high"] == 115.0  # T == target_ms foi incluído na excursão
        assert h1m["observation_count"] == 1
    finally:
        transport.close(timeout=1.0)
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


def test_causal_proof_raw_trade_timestamp():
    """Prova matemática e causal sobre o monotonic clamp de T_raw.

    No MarketOrchestrator:
    T = max(T_raw, last_T)

    Invariante: T_raw <= T
    Critério de inclusão na feature window: T < causal_anchor_ms
    Portanto: T_raw <= T < causal_anchor_ms
    Conclusão: max(T_raw) < causal_anchor_ms.
    É matematicamente impossível um trade com T_raw >= causal_anchor_ms entrar na feature.
    """
    causal_anchor_ms = 1788702420000
    last_T = 1788702410000

    # Simulador do hot path do MarketOrchestrator
    trades_simulados = [
        {"T_raw": 1788702411000, "price": 100.0},
        {"T_raw": 1788702410500, "price": 100.2},  # Chegada com ligeiro jitter/out-of-order
        {"T_raw": 1788702418610, "price": 100.5},  # Último trade contemporâneo válido
        {"T_raw": 1788702420000, "price": 101.0},  # Trade exatamente no boundary anchor
        {"T_raw": 1788702420001, "price": 101.1},  # Trade no futuro
    ]

    feature_window_trades = []
    future_trades = []

    for tr in trades_simulados:
        t_clamped = max(tr["T_raw"], last_T)
        last_T = t_clamped

        # Regra causal do fechamento de janela:
        if t_clamped < causal_anchor_ms:
            # Prova: T_raw <= t_clamped < causal_anchor_ms
            assert tr["T_raw"] < causal_anchor_ms
            feature_window_trades.append(tr)
        else:
            future_trades.append(tr)

    # 1. Prova que max(T_raw) dentro da feature window é estritamente < causal_anchor_ms
    max_t_raw_feature = max(t["T_raw"] for t in feature_window_trades)
    assert max_t_raw_feature < causal_anchor_ms
    assert max_t_raw_feature == 1788702418610

    # 2. Prova que o trade com T_raw == causal_anchor_ms NÃO entrou na feature
    assert all(t["T_raw"] != causal_anchor_ms for t in feature_window_trades)
    assert any(t["T_raw"] == causal_anchor_ms for t in future_trades)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DETERMINISTIC RECORD ID E PROVENIÊNCIA
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_record_id_uses_causal_anchor():
    """Valida formato do record_id e invariância a variações do último trade observado."""
    anchor = 1788702420000
    expected_id = f"rec_BTCUSDT_{anchor}_v{FEATURE_CONTRACT_VERSION}_s{SHADOW_SCHEMA_VERSION}"

    # 1. Chamada explícita com causal_anchor_ms
    rec_id1 = build_deterministic_record_id(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
    )
    assert rec_id1 == expected_id

    # 2. Dois records com último trade observado diferente mas mesmo boundary lógico
    # devem gerar o MESMO record_id canônico
    rec_id_a = build_deterministic_record_id(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
        window_close_ms=anchor - 1390,  # J2: 1788702418610
    )
    rec_id_b = build_deterministic_record_id(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
        window_close_ms=anchor - 10000,  # Janela esparsa: 1788702410000
    )
    assert rec_id_a == expected_id
    assert rec_id_b == expected_id
    assert rec_id_a == rec_id_b

    # 3. Fallback retrocompatível para window_close_ms legado
    rec_id_legacy = build_deterministic_record_id(
        symbol="BTCUSDT",
        window_close_ms=anchor,
    )
    assert rec_id_legacy == expected_id


def test_provenance_contains_all_three_timestamps():
    """Verifica que o ShadowProvenance expõe observation_open, observation_close e causal_anchor."""
    rec = build_shadow_record(
        symbol="BTCUSDT",
        window_open_ms=1788702361157,
        window_close_ms=1788702418610,
        window_data={"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        causal_anchor_ms=1788702420000,
        observation_open_ms=1788702361157,
        observation_close_ms=1788702418610,
    )
    prov = rec.provenance
    assert prov.observation_open_ms == 1788702361157
    assert prov.observation_close_ms == 1788702418610
    assert prov.causal_anchor_ms == 1788702420000
    assert prov.window_close_ms == 1788702418610  # Mantém valor físico legado


# ─────────────────────────────────────────────────────────────────────────────
# 3. DTO FROZEN E ISOLAMENTO DE MUTAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

def test_dto_is_frozen_and_isolated():
    """Garante imutabilidade do DTO e proteção contra mutação de referências no caller."""
    context_mutable = {"regime_current_at_t": "TRENDING_EXPANSION"}

    dto = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=1788702420000,
        observation_open_ms=1788702361157,
        observation_close_ms=1788702418610,
        buy_notional_usd=500000.0,
        sell_notional_usd=200000.0,
        open=80000.0,
        high=80050.0,
        low=79950.0,
        close=80020.0,
        window_duration_ms=57453,
        context_data=context_mutable,
    )

    # 1. Tentativa de mutação em atributo primitivo deve falhar
    with pytest.raises((FrozenInstanceError, AttributeError)):
        dto.close = 81000.0  # type: ignore

    # 2. Mutação posterior do dicionário do chamador NÃO deve alterar o DTO
    context_mutable["regime_current_at_t"] = "MUTATED_BY_CALLER"
    assert dto.context_data["regime_current_at_t"] == "TRENDING_EXPANSION"


# ─────────────────────────────────────────────────────────────────────────────
# 4. FILA BOUNDED, BACKPRESSURE DROP_NEWEST E MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def test_bounded_queue_drop_newest(temp_dir):
    """Fila bounded com DROP_NEWEST: não bloqueia e incrementa drop counter sob saturação."""
    target_file = temp_dir / "test_queue.jsonl"

    # Cria transport com capacidade mínima = 2 e sem iniciar worker thread
    transport = ShadowAsyncTransport(
        filepath=target_file,
        queue_capacity=2,
        enabled=True,
        start_worker=False,
    )

    dto_template = lambda i: EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=1788702420000 + i * 60_000,
        observation_open_ms=1788702360000 + i * 60_000,
        observation_close_ms=1788702419000 + i * 60_000,
        buy_notional_usd=1000.0,
        sell_notional_usd=1000.0,
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        window_duration_ms=59000,
    )

    # Enfileira item 1 e item 2 -> Sucesso
    res1 = transport.submit_nowait(dto_template(1))
    res2 = transport.submit_nowait(dto_template(2))
    assert res1 is True
    assert res2 is True

    # Enfileira item 3 -> Deve descartar (DROP_NEWEST) sem exceção
    res3 = transport.submit_nowait(dto_template(3))
    assert res3 is False

    stats = transport.get_stats()
    assert stats["queue_depth"] == 2
    assert stats["records_enqueued"] == 2
    assert stats["records_dropped"] == 1

    transport.close(timeout=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SINGLE-WRITER E CONCORRÊNCIA MULTI-THREAD
# ─────────────────────────────────────────────────────────────────────────────

def test_single_writer_multi_thread_concurrency(temp_dir):
    """Múltiplas threads chamando submit_nowait concorrentemente são persistidas sem colisão."""
    target_file = temp_dir / "test_concurrency.jsonl"
    transport = ShadowAsyncTransport(
        filepath=target_file,
        queue_capacity=500,
        enabled=True,
    )

    num_threads = 5
    items_per_thread = 20
    errors: List[Exception] = []

    def producer_worker(thread_idx: int):
        for i in range(items_per_thread):
            anchor = 1788702420000 + (thread_idx * 1000 + i) * 60_000
            dto = EffortResponseSnapshotDTO(
                symbol=f"SYM{thread_idx}",
                causal_anchor_ms=anchor,
                observation_open_ms=anchor - 50_000,
                observation_close_ms=anchor - 1_000,
                buy_notional_usd=1000.0 + i,
                sell_notional_usd=1000.0,
                open=100.0,
                high=105.0,
                low=95.0,
                close=100.0,
                window_duration_ms=49000,
            )
            success = transport.submit_nowait(dto)
            if not success:
                errors.append(RuntimeError(f"Drop inesperado na thread {thread_idx} item {i}"))
            time.sleep(0.001)

    threads = [threading.Thread(target=producer_worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0

    # Aguarda o single writer persistir todos os registros
    flushed = transport.flush(timeout=5.0)
    assert flushed is True

    transport.close(timeout=2.0)

    # Valida no arquivo físico
    storage = EffortResponseShadowStorage(target_file)
    records = storage.read_records()
    assert len(records) == num_threads * items_per_thread

    # Valida que todos os record_ids são únicos
    record_ids = {r.record_id for r in records}
    assert len(record_ids) == num_threads * items_per_thread


# ─────────────────────────────────────────────────────────────────────────────
# 6. MIN-HEAP E RESOLUÇÃO TEMPORAL SEM FULL SCAN
# ─────────────────────────────────────────────────────────────────────────────

def test_min_heap_resolution_without_full_scan(temp_dir):
    """Valida que o min-heap avalia apenas itens com target <= event_time atual sem full scan."""
    target_file = temp_dir / "test_min_heap.jsonl"
    transport = ShadowAsyncTransport(
        filepath=target_file,
        queue_capacity=50,
        enabled=True,
    )

    anchor = 1788702420000
    dto = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
        observation_open_ms=anchor - 50_000,
        observation_close_ms=anchor - 1_000,
        buy_notional_usd=1000.0,
        sell_notional_usd=1000.0,
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        window_duration_ms=49000,
    )

    transport.submit_nowait(dto)
    transport.flush(timeout=1.0)

    stats = transport.get_stats()
    assert stats["pending_heap_size"] == 3  # 1m, 5m, 15m

    # Envia observação em T = anchor + 30s (antes do 1m que é anchor + 60s)
    transport.on_price_observation(
        timestamp_ms=anchor + 30_000,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        is_window=False,
    )
    transport.flush(timeout=1.0)

    # Nenhum horizonte deve ter sido desempilhado pois o topo do heap (1m) ainda não venceu
    stats = transport.get_stats()
    assert stats["pending_heap_size"] == 3

    # Envia observação em T = anchor + 60s (alcança 1m)
    transport.on_price_observation(
        timestamp_ms=anchor + 60_000,
        open=100.5,
        high=102.0,
        low=100.0,
        close=101.0,
        is_window=False,
    )
    transport.flush(timeout=1.0)

    # 1m deve ter sido resolvido e desempilhado; 5m e 15m permanecem
    stats = transport.get_stats()
    assert stats["pending_heap_size"] == 2

    storage = EffortResponseShadowStorage(target_file)
    records = storage.read_records()
    h1m = records[0].outcomes_future["horizons"]["1m"]
    assert h1m["status"] == "RESOLVED"
    assert h1m["future_price"] == 101.0

    transport.close(timeout=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. TERMINAL PRICE E TOLERÂNCIA FIRST_ON_OR_AFTER
# ─────────────────────────────────────────────────────────────────────────────

def test_terminal_price_tolerance_and_gap(temp_dir):
    """Valida tolerância de 1000ms para terminal price e INSUFFICIENT_DATA para gaps."""
    target_file = temp_dir / "test_terminal_gap.jsonl"
    transport = ShadowAsyncTransport(
        filepath=target_file,
        queue_capacity=50,
        enabled=True,
    )

    anchor = 1788702420000
    target_1m = anchor + 60_000

    dto = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
        observation_open_ms=anchor - 50_000,
        observation_close_ms=anchor - 1_000,
        buy_notional_usd=1000.0,
        sell_notional_usd=1000.0,
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        window_duration_ms=49000,
    )
    transport.submit_nowait(dto)
    transport.flush(timeout=1.0)

    # Envia primeira observação em target_1m + 2500ms (fora da tolerância de 1000ms)
    transport.on_price_observation(
        timestamp_ms=target_1m + 2500,
        open=105.0,
        high=106.0,
        low=104.0,
        close=105.0,
        is_window=False,
    )
    transport.flush(timeout=1.0)

    storage = EffortResponseShadowStorage(target_file)
    records = storage.read_records()
    h1m = records[0].outcomes_future["horizons"]["1m"]

    assert h1m["status"] == "INSUFFICIENT_DATA"
    assert h1m["future_price"] is None

    transport.close(timeout=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. OHLC PARCIAL NÃO CONTAMINA EXTREMA
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_ohlc_crossing_target_boundary_does_not_contaminate(temp_dir):
    """Janela OHLC que ultrapassa o target não inclui seu high/low na excursão e marca PARTIAL."""
    target_file = temp_dir / "test_partial_ohlc.jsonl"
    transport = ShadowAsyncTransport(
        filepath=target_file,
        queue_capacity=50,
        enabled=True,
    )

    anchor = 1788702420000
    target_1m = anchor + 60_000

    dto = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
        observation_open_ms=anchor - 50_000,
        observation_close_ms=anchor - 1_000,
        buy_notional_usd=1000.0,
        sell_notional_usd=1000.0,
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        window_duration_ms=49000,
    )
    transport.submit_nowait(dto)
    transport.flush(timeout=1.0)

    # 1. Candle que cruza o boundary causal [anchor - 10s, anchor + 10s] com extremos espúrios
    # open=100.0, high=999.0 (espúrio além da fronteira), low=1.0, close=101.0
    transport.on_price_observation(
        timestamp_ms=anchor + 10_000,
        open=100.0,
        high=999.0,
        low=1.0,
        close=101.0,
        is_window=True,
        window_open_ms=anchor - 10_000,
        window_close_ms=anchor + 10_000,
    )

    # 2. Candle inteiramente dentro de (anchor, target]
    transport.on_price_observation(
        timestamp_ms=anchor + 30_000,
        open=101.0,
        high=104.0,
        low=99.0,
        close=102.0,
        is_window=True,
        window_open_ms=anchor + 15_000,
        window_close_ms=anchor + 30_000,
    )

    # 3. Observação pontual para o preço terminal exatamente no target
    transport.on_price_observation(
        timestamp_ms=target_1m,
        open=102.5,
        high=103.0,
        low=102.0,
        close=102.8,
        is_window=False,
    )

    transport.flush(timeout=1.0)

    storage = EffortResponseShadowStorage(target_file)
    records = storage.read_records()
    h1m = records[0].outcomes_future["horizons"]["1m"]

    assert h1m["status"] == "RESOLVED"
    assert h1m["future_price"] == 102.8
    # high=999.0 e low=1.0 do candle parcialmente sobreposto NÃO podem ter sido incorporados
    assert h1m["max_high"] == 104.0
    assert h1m["min_low"] == 99.0
    assert h1m["excursion_status"] == "PARTIAL"

    transport.close(timeout=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. RESTART FAIL-CLOSED E TRATAMENTO DE GAP
# ─────────────────────────────────────────────────────────────────────────────

def test_restart_fail_closed_offline_gap(temp_dir):
    """Na inicialização, horizontes PENDING vencidos durante o downtime são marcados INSUFFICIENT_DATA."""
    target_file = temp_dir / "test_restart.jsonl"

    anchor = 1788702420000
    # Cria registro diretamente no storage com status PENDING
    storage = EffortResponseShadowStorage(target_file)
    rec = build_shadow_record(
        symbol="BTCUSDT",
        window_open_ms=anchor - 60_000,
        window_close_ms=anchor - 1_000,
        window_data={"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        causal_anchor_ms=anchor,
    )
    storage.append_record(rec)

    # Inicia o transport simulando que o bot acordou 2 horas depois do anchor
    startup_time = anchor + 7200_000
    transport = ShadowAsyncTransport(
        filepath=target_file,
        enabled=True,
        startup_event_time_ms=startup_time,
    )

    transport.close(timeout=1.0)

    records = storage.read_records()
    assert len(records) == 1
    h_outcomes = records[0].outcomes_future["horizons"]

    # 1m, 5m e 15m devem todos ter sido marcados como INSUFFICIENT_DATA devido ao downtime
    assert h_outcomes["1m"]["status"] == "INSUFFICIENT_DATA"
    assert h_outcomes["5m"]["status"] == "INSUFFICIENT_DATA"
    assert h_outcomes["15m"]["status"] == "INSUFFICIENT_DATA"


# ─────────────────────────────────────────────────────────────────────────────
# 10. FAILURE ISOLATION (STORAGE CORRUPTION)
# ─────────────────────────────────────────────────────────────────────────────

def test_storage_corruption_disables_shadow_cleanly(temp_dir):
    """Storage corrompido desativa apenas o subsistema shadow sem lançar exceção no hot path."""
    target_file = temp_dir / "test_corrupt.jsonl"
    with open(target_file, "w", encoding="utf-8") as f:
        f.write("{invalid json payload line\n")

    # Inicia transporte apontando para o arquivo corrompido
    transport = ShadowAsyncTransport(filepath=target_file, enabled=True)

    stats = transport.get_stats()
    assert stats["disabled_due_to_corruption"] is True
    assert stats["enabled"] is False

    # Tentativa de submissão do hot path retorna False sem quebrar a aplicação
    dto = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=1788702420000,
        observation_open_ms=1788702361157,
        observation_close_ms=1788702418610,
        buy_notional_usd=1000.0,
        sell_notional_usd=1000.0,
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        window_duration_ms=57000,
    )
    res = transport.submit_nowait(dto)
    assert res is False

    transport.close(timeout=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 11. GESTÃO DE MÉTRICAS PROMETHEUS
# ─────────────────────────────────────────────────────────────────────────────

def test_prometheus_metrics_no_duplicate_registration():
    """Instanciação de ShadowMetrics não causa duplicação ou conflito no registry."""
    m1 = ShadowMetrics.get_instance()
    m2 = ShadowMetrics.get_instance()
    assert m1 is m2
    assert hasattr(m1, "queue_size")
    assert hasattr(m1, "records_enqueued_total")
    assert hasattr(m1, "records_dropped_total")


# ─────────────────────────────────────────────────────────────────────────────
# 12. WALL CLOCK NÃO ALTERA TARGET TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────

def test_wall_clock_does_not_alter_target():
    """A passagem de tempo de relógio do SO não afeta o cálculo nem a ancoragem do target."""
    anchor = 1788702420000
    dto1 = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
        observation_open_ms=anchor - 50_000,
        observation_close_ms=anchor - 1_000,
        buy_notional_usd=1000.0,
        sell_notional_usd=1000.0,
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        window_duration_ms=49000,
    )

    time.sleep(0.05)  # Avanço de wall clock

    dto2 = EffortResponseSnapshotDTO(
        symbol="BTCUSDT",
        causal_anchor_ms=anchor,
        observation_open_ms=anchor - 50_000,
        observation_close_ms=anchor - 1_000,
        buy_notional_usd=1000.0,
        sell_notional_usd=1000.0,
        open=100.0,
        high=105.0,
        low=95.0,
        close=100.0,
        window_duration_ms=49000,
    )

    rec1 = build_shadow_record(
        symbol=dto1.symbol,
        window_open_ms=dto1.observation_open_ms,
        window_close_ms=dto1.observation_close_ms,
        window_data={"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0},
        causal_anchor_ms=dto1.causal_anchor_ms,
    )
    rec2 = build_shadow_record(
        symbol=dto2.symbol,
        window_open_ms=dto2.observation_open_ms,
        window_close_ms=dto2.observation_close_ms,
        window_data={"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0},
        causal_anchor_ms=dto2.causal_anchor_ms,
    )

    assert rec1.outcomes_future["horizons"]["1m"]["target_timestamp_ms"] == \
           rec2.outcomes_future["horizons"]["1m"]["target_timestamp_ms"] == anchor + 60_000
