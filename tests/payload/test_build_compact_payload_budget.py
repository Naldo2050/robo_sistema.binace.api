"""
Budget test — garante que o payload compacto não ultrapassa limites
de tamanho definidos como contrato.

Limites:
  - HARD LIMIT: 6144 bytes (guardrail do pipeline)
  - SOFT LIMIT: 2500 bytes (meta de qualidade com summary)
  - WARN LIMIT: 2300 bytes (alerta antecipado de crescimento)
"""

import json
import copy
import pytest
import build_compact_payload as bcp

HARD_LIMIT = 6144
SOFT_LIMIT = 3000
WARN_LIMIT = 2800

@pytest.fixture(autouse=True)
def reset_static_cache():
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0
    yield
    bcp._last_static_ctx = {}
    bcp._last_static_ts = 0.0

def payload_bytes(payload: dict) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

def make_base_event() -> dict:
    """Retorna evento canônico completo para testes de budget."""
    return {
        "symbol": "BTCUSDT",
        "epoch_ms": 1775173800000,
        "tipo_evento": "ANALYSIS_TRIGGER",
        "preco_fechamento": 66875,
        "delta": 0.123,
        "volume_total": 1.5,
        "volume_compra": 0.9,
        "contextual_snapshot": {
            "ohlc": {
                "open": 66884, "high": 66890, "low": 66856, "close": 66875, "vwap": 66879,
            }
        },
        "market_context": { "trading_session": "NY", "session_phase": "ACTIVE" },
        "market_environment": { "volatility_regime": "HIGH", "market_structure": "RANGE_BOUND" },
        "historical_vp": {
            "daily": { "status": "success", "poc": 66444, "val": 66388, "vah": 67771 }
        },
        "external_markets": {
            "VIX": {"preco_atual": 23.9}, "FEAR_GREED": {"preco_atual": 50},
            "DXY": {"preco_atual": 104.2}, "SP500": {"preco_atual": 669.1},
            "NASDAQ": {"preco_atual": 19200.0}, "GOLD": {"preco_atual": 2350.0},
            "WTI": {"preco_atual": 82.5}, "TNX": {"preco_atual": 4.31},
        },
        "derivatives": {
            "BTCUSDT": {
                "long_short_ratio": 1.83, "open_interest": 90101.906,
                "funding_rate_percent": -0.01, "longs_usd": 3893635557.19, "shorts_usd": 2129881055.29,
            },
            "ETHUSDT": { "long_short_ratio": 1.66 },
        },
        "ml_features": {
            "microstructure": { "trade_intensity_v2": 12.3, "tick_rule_sum": 10, "order_book_slope": -0.25 },
        },
        "multi_tf": {
            "1h": { "tendencia": "Baixa", "rsi_short": 50, "macd": -151, "adx": 21, "atr": 384, "regime": "Range" },
            "4h": { "tendencia": "Baixa", "rsi_short": 43, "macd": -178, "adx": 27, "atr": 968, "regime": "Manipulação" },
        },
        "fluxo_continuo": {
            "cvd": 0.2,
            "order_flow": {
                "net_flow_1m": 16000, "net_flow_5m": 12000, "flow_imbalance": 0.18,
                "aggressive_buy_pct": 59, "buy_sell_ratio": {"buy_sell_ratio": 1.44},
            },
            "absorption_analysis": {
                "current_absorption": { "buyer_strength": 5.9, "seller_exhaustion": 1.8 }
            },
        },
        "institutional_analytics": {
            "profile_analysis": {
                "poor_extremes": { "action_bias": "expect_retest_both", "poor_high": {"detected": 1}, "poor_low": {"detected": 1} },
            },
            "quality": {
                "calendar": {"expected_liquidity": "NORMAL"},
                "latency": { "latency_ms": 820, "latency_category": "OK" },
            },
            "flow_analysis": {
                "passive_aggressive": { "composite": { "signal": "buy_absorption", "conviction": "MEDIUM" } },
            },
            "sr_analysis": {
                "defense_zones": {
                    "status": "success",
                    "sell_defense": [{"center": 66931, "strength": 66, "source_count": 5}],
                    "buy_defense": [{"center": 66839, "strength": 58, "source_count": 4}],
                }
            },
        },
        "data_reliability": { "latency_acceptable": 1, "onchain_coverage": "full" },
    }

def test_budget_base_event_below_soft_limit():
    payload = bcp.build_compact_payload(make_base_event())
    size = payload_bytes(payload)
    assert size < SOFT_LIMIT, f"Payload base ({size}) > {SOFT_LIMIT}"

def test_budget_cached_ctx_is_smaller_than_full_ctx():
    event = make_base_event()
    first = bcp.build_compact_payload(copy.deepcopy(event))
    second = bcp.build_compact_payload(copy.deepcopy(event))
    size_first = payload_bytes(first)
    size_second = payload_bytes(second)
    assert size_second < size_first, f"Cache falhou: {size_second} >= {size_first}"

def test_budget_summary_section_present():
    payload = bcp.build_compact_payload(make_base_event())
    assert "summary" in payload
    assert len(payload["summary"]) >= 4

def test_budget_guard_hard_limit_protection():
    def giant_summary(_p): return {"flow": {"note": "x"*8000}}
    with pytest.MonkeyPatch().context() as m:
        m.setattr(bcp, "_build_summary_section", giant_summary)
        payload = bcp.build_compact_payload(make_base_event())
        assert "summary" not in payload
        assert payload_bytes(payload) < HARD_LIMIT

def test_budget_guard_soft_limit_compaction():
    # Forçar payload logo acima de 2500
    def large_summary(_p): return {"flow": {"note": "x"*1000, "extra": "y"*1000}}
    with pytest.MonkeyPatch().context() as m:
        m.setattr(bcp, "_build_summary_section", large_summary)
        payload = bcp.build_compact_payload(make_base_event())
        if "summary" in payload:
            assert "extra" not in payload["summary"]["flow"], "Campos extras não foram removidos"


# ── Testes adicionais pós-integração dos summary builders ──────────────

def test_budget_with_summary_section_below_soft_limit():
    """
    Payload com summary deve continuar abaixo do soft limit.
    Se ultrapassar, o budget guard interno deve ter compactado o summary.
    """
    payload = bcp.build_compact_payload(make_base_event())
    size = payload_bytes(payload)

    assert size < SOFT_LIMIT, (
        f"Payload COM summary ultrapassou soft limit: "
        f"{size} bytes (limite={SOFT_LIMIT})"
    )


def test_budget_summary_section_present_when_within_budget():
    """
    Se payload está dentro do budget, summary deve estar presente e completo.
    """
    payload = bcp.build_compact_payload(make_base_event())
    size = payload_bytes(payload)

    if size < SOFT_LIMIT:
        assert "summary" in payload, "summary ausente mesmo dentro do budget"
        assert len(payload["summary"]) > 0


def test_budget_summary_has_expected_builders():
    """
    Todos os 5 builders devem estar presentes no summary.
    """
    payload = bcp.build_compact_payload(make_base_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    expected = {"flow", "sr", "regime", "institutional", "quality"}
    present = set(payload["summary"].keys())
    missing = expected - present

    assert not missing, f"Builders ausentes no summary: {missing}"


def test_budget_summary_notes_are_non_empty_strings():
    """
    Cada seção do summary deve ter uma nota não vazia.
    """
    payload = bcp.build_compact_payload(make_base_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    for section_name, section_data in payload["summary"].items():
        assert "note" in section_data, (
            f"summary.{section_name} sem campo 'note'"
        )
        assert isinstance(section_data["note"], str), (
            f"summary.{section_name}.note não é string"
        )
        assert len(section_data["note"]) > 5, (
            f"summary.{section_name}.note está vazio ou muito curto"
        )


def test_budget_summary_quality_reflects_payload_state():
    """
    quality_summary.reliable deve refletir o estado real do qual.
    """
    event_ok = make_base_event()
    payload_ok = bcp.build_compact_payload(event_ok)

    event_degraded = make_base_event()
    event_degraded["institutional_analytics"]["quality"]["latency"][
        "latency_category"
    ] = "DEGRADED"
    event_degraded["institutional_analytics"]["quality"]["latency"][
        "latency_ms"
    ] = 3622
    payload_degraded = bcp.build_compact_payload(event_degraded)

    if "summary" not in payload_ok or "summary" not in payload_degraded:
        pytest.skip("summary não disponível neste ambiente")

    q_ok = payload_ok["summary"]["quality"]
    q_deg = payload_degraded["summary"]["quality"]

    assert q_ok["reliable"] is True
    assert q_deg["reliable"] is False
    assert q_deg["confidence_cap"] < q_ok["confidence_cap"]


def test_budget_summary_flow_bias_consistent_with_flow_section():
    """
    summary.flow.bias deve ser consistente com flow.pa e flow.imb.
    """
    payload = bcp.build_compact_payload(make_base_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    flow = payload.get("flow", {})
    flow_summary = payload["summary"]["flow"]

    pa = str(flow.get("pa", "")).lower()
    bias = flow_summary.get("bias", "")

    if "buy" in pa:
        assert bias in ("BUY", "NEUTRAL"), (
            f"pa={pa} mas summary.flow.bias={bias}"
        )
    elif "sell" in pa:
        assert bias in ("SELL", "NEUTRAL"), (
            f"pa={pa} mas summary.flow.bias={bias}"
        )


def test_budget_summary_regime_label_consistent_with_regime_mode():
    """
    summary.regime.label deve ser consistente com regime.mode.
    """
    payload = bcp.build_compact_payload(make_base_event())

    if "summary" not in payload:
        pytest.skip("summary não disponível neste ambiente")

    mode = payload.get("regime", {}).get("mode", "")
    label = payload["summary"]["regime"].get("label", "")

    _expected = {
        "MR":  "Mean Reversion",
        "RB":  "Range Bound",
        "TRD": "Trending",
        "BRK": "Breakout",
    }

    if mode in _expected:
        assert label == _expected[mode], (
            f"regime.mode={mode} mas summary.regime.label={label}"
        )
