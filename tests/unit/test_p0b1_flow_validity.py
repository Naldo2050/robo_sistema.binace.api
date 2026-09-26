# tests/unit/test_p0b1_flow_validity.py — P0-B1: integridade do produtor + metadata temporal.
#
# Contrato (sem threshold novo; usa exclusivamente flow_window_integrity):
# - VALID   <=> integrity[tf].is_temporal_coverage_valid is True.
# - PARTIAL <=> aritmética válida + cobertura não-FULL (WARMING_UP,
#               CAPACITY_TRUNCATED, TRUNCATED, ...) ou integridade ausente
#               (reason=NO_INTEGRITY_INFO; nunca VALID assumido).
# - INVALID <=> aritmética inválida (non-finite, denominador ausente/
#               não-finito/não-positivo, |imb|>1 além do epsilon 1e-9
#               preexistente): valor null (chave numérica omitida, como antes),
#               reason=INVARIANT_VIOLATION. Nunca clamp silencioso para ±1.0.
# - Janela sem chave net_flow_Xm => ausente em valores E em validity.
# - Valores PARTIAL permanecem numéricos (observabilidade), mas metadata deixa
#   claro que NÃO são confirmação temporal completa (consumers: P0-B2).
# - Aplica-se às duas implementações (metrics ativa em produção via
#   core.py:1306; aggregates em paridade via testes). Núcleo canônico único:
#   flow_analyzer/metrics.compute_window_imbalances (aggregates importa de
#   metrics; metrics nunca importa aggregates — sem ciclo).

import json
import math

import pytest

from flow_analyzer.aggregates import (
    calculate_buy_sell_ratios as aggregates_calc,
)
from flow_analyzer.metrics import calculate_buy_sell_ratios as metrics_calc

CALCS = [metrics_calc, aggregates_calc]


def _full_integrity():
    return {
        "1m": {"status": "FULL", "effective_coverage_pct": 100.0,
               "is_temporal_coverage_valid": True},
        "5m": {"status": "FULL", "effective_coverage_pct": 100.0,
               "is_temporal_coverage_valid": True},
        "15m": {"status": "FULL", "effective_coverage_pct": 100.0,
                "is_temporal_coverage_valid": True},
    }


def _j2_integrity():
    # Cenário J2: 1m 95.8% / 5m 31.1% / 15m 10.4%, todos valid=false.
    return {
        "1m": {"status": "WARMING_UP", "effective_coverage_pct": 95.8,
               "is_temporal_coverage_valid": False},
        "5m": {"status": "WARMING_UP", "effective_coverage_pct": 31.1,
               "is_temporal_coverage_valid": False},
        "15m": {"status": "WARMING_UP", "effective_coverage_pct": 10.4,
                "is_temporal_coverage_valid": False},
    }


def _no_sentinels(obj):
    text = json.dumps(obj, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    return text


# ── A. FULL ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_a_full_valid(calc):
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 2.0, "net_flow_5m": 5.0, "net_flow_15m": -3.0,
                "total_volume_1m": 5.0, "total_volume_5m": 25.0,
                "total_volume_15m": 15.0,
                "flow_window_integrity": _full_integrity()})
    assert out["ratios"]["imbalance_1m"] == 0.4
    assert out["ratios"]["imbalance_5m"] == 0.2
    assert out["ratios"]["imbalance_15m"] == -0.2
    for tf in ("1m", "5m", "15m"):
        assert out["imbalance_validity"][tf] == {"validity": "VALID", "reason": None}
    _no_sentinels(out)


# ── B. WARMING_UP ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_b_warming_up_partial_keeps_numbers(calc):
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 2.0, "net_flow_5m": 5.0, "net_flow_15m": -3.0,
                "total_volume_1m": 5.0, "total_volume_5m": 25.0,
                "total_volume_15m": 15.0,
                "flow_window_integrity": _j2_integrity()})
    # Valores parciais preservados para observabilidade...
    assert out["ratios"]["imbalance_1m"] == 0.4
    assert out["ratios"]["imbalance_5m"] == 0.2
    assert out["ratios"]["imbalance_15m"] == -0.2
    # ...mas nenhum é VALID.
    assert out["imbalance_validity"]["1m"] == {"validity": "PARTIAL", "reason": "WARMING_UP"}
    assert out["imbalance_validity"]["5m"] == {"validity": "PARTIAL", "reason": "WARMING_UP"}
    assert out["imbalance_validity"]["15m"] == {"validity": "PARTIAL", "reason": "WARMING_UP"}
    _no_sentinels(out)


# ── C. CAPACITY_TRUNCATED ────────────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_c_capacity_truncated_is_partial_never_valid(calc):
    integrity = _full_integrity()
    integrity["15m"] = {"status": "CAPACITY_TRUNCATED",
                        "effective_coverage_pct": 60.0,
                        "is_temporal_coverage_valid": False}
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 2.0, "net_flow_5m": 5.0, "net_flow_15m": -3.0,
                "total_volume_1m": 5.0, "total_volume_5m": 25.0,
                "total_volume_15m": 15.0,
                "flow_window_integrity": integrity})
    assert out["ratios"]["imbalance_15m"] == -0.2
    assert out["imbalance_validity"]["15m"] == {
        "validity": "PARTIAL", "reason": "CAPACITY_TRUNCATED"}
    assert out["imbalance_validity"]["1m"]["validity"] == "VALID"
    _no_sentinels(out)


# ── D. invariant violation (J2 1.056) ────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_d_imbalance_1056_is_invalid_null_never_clamped(calc):
    # Reprodução J2: numerador/denominador incompatíveis geram 8.416/7.97=1.056.
    # Com P01 o produtor não cruza janelas, mas a guarda fail-closed vale para
    # QUALQUER origem (path antigo, dado externo, corrupção): nunca publica
    # 1.056, nunca clamp para 1.0.
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 2.0, "net_flow_5m": 8.416, "net_flow_15m": -3.0,
                "total_volume_1m": 5.0, "total_volume_5m": 7.97,
                "total_volume_15m": 15.0,
                "flow_window_integrity": _j2_integrity()})
    assert "imbalance_5m" not in out["ratios"]
    assert out["imbalance_validity"]["5m"] == {
        "validity": "INVALID", "reason": "INVARIANT_VIOLATION"}
    # Janelas válidas vizinhas intactas:
    assert out["ratios"]["imbalance_1m"] == 0.4
    assert out["imbalance_validity"]["1m"]["validity"] == "PARTIAL"
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
def test_d_negative_violation_is_invalid(calc):
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_5m": -8.416, "total_volume_5m": 7.97,
                "flow_window_integrity": _full_integrity()})
    assert "imbalance_5m" not in out["ratios"]
    assert out["imbalance_validity"]["5m"] == {
        "validity": "INVALID", "reason": "INVARIANT_VIOLATION"}
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
def test_d_fp_epsilon_still_snaps(calc):
    # Tolerância 1e-9 preexistente reutilizada: 1+5e-10 vira 1.0 (VALID se FULL).
    out = calc({"buy_volume_btc": 1.0, "sell_volume_btc": 1.0,
                "net_flow_1m": 1.0 + 5e-10, "total_volume_1m": 1.0,
                "flow_window_integrity": _full_integrity()})
    assert out["ratios"]["imbalance_1m"] == 1.0
    assert out["imbalance_validity"]["1m"]["validity"] == "VALID"


# ── E. NaN/Inf ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_e_nonfinite_net_is_invalid(calc, bad):
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_5m": bad, "total_volume_5m": 25.0,
                "flow_window_integrity": _full_integrity()})
    assert "imbalance_5m" not in out["ratios"]
    assert out["imbalance_validity"]["5m"] == {
        "validity": "INVALID", "reason": "INVARIANT_VIOLATION"}
    _no_sentinels(out)


@pytest.mark.parametrize("calc", CALCS)
def test_e_missing_denominator_is_invalid(calc):
    # P01: sem total_5m não há fallback (nulo, nunca 1.056 por cruzamento),
    # e a ausência do denominador é INVALID explícita, não silenciosa.
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 2.0, "net_flow_5m": 5.0,
                "total_volume": 5.0,
                "flow_window_integrity": _full_integrity()})
    assert out["ratios"]["imbalance_1m"] == 0.4
    assert "imbalance_5m" not in out["ratios"]
    assert out["imbalance_validity"]["5m"] == {
        "validity": "INVALID", "reason": "INVARIANT_VIOLATION"}
    _no_sentinels(out)


# ── F. triple-window warm-up ─────────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_f_same_trades_triple_window_marks_5m_15m_partial(calc):
    # Mesmos trades => mesmos net/total nas 3 janelas (colapso de warm-up):
    # valores idênticos, mas 5m/15m explicitamente PARTIAL (não confirmação).
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 8.416, "net_flow_5m": 8.416, "net_flow_15m": 8.416,
                "total_volume_1m": 10.0, "total_volume_5m": 10.0,
                "total_volume_15m": 10.0,
                "flow_window_integrity": _j2_integrity()})
    assert out["ratios"]["imbalance_1m"] == out["ratios"]["imbalance_5m"] == \
        out["ratios"]["imbalance_15m"] == round(8.416 / 10.0, 4)
    assert out["imbalance_validity"]["1m"]["validity"] == "PARTIAL"
    assert out["imbalance_validity"]["5m"]["validity"] == "PARTIAL"
    assert out["imbalance_validity"]["15m"]["validity"] == "PARTIAL"
    assert "VALID" not in [v["validity"] for v in out["imbalance_validity"].values()]
    _no_sentinels(out)


# ── G. regressão bit-equivalente ─────────────────────────────────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_g_full_values_bit_equivalent_without_integrity(calc):
    # Sem integridade (chamadores legados): numerics idênticos aos de antes de
    # P0-B1; validade fail-safe PARTIAL/NO_INTEGRITY_INFO (nunca VALID).
    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 2.0, "net_flow_5m": 5.0, "net_flow_15m": -3.0,
                "total_volume_1m": 5.0, "total_volume_5m": 25.0,
                "total_volume_15m": 15.0})
    assert out["ratios"]["imbalance_1m"] == 0.4
    assert out["ratios"]["imbalance_5m"] == 0.2
    assert out["ratios"]["imbalance_15m"] == -0.2
    assert out["buy_sell_ratio"] == 1.5
    for tf in ("1m", "5m", "15m"):
        assert out["imbalance_validity"][tf] == {
            "validity": "PARTIAL", "reason": "NO_INTEGRITY_INFO"}
    _no_sentinels(out)


# ── J2: P01 prova que o código atual não gera 1.056 por cruzamento ───────────

@pytest.mark.parametrize("calc", CALCS)
def test_j2_same_population_never_exceeds_one(calc):
    # Com P01 (denominador da própria janela), a mesma população jamais gera
    # |imb|>1: J2 1.056 veio de versão/path anterior (net_15m/total_1m), não do
    # código atual. Prova por grade: net/total da mesma janela sempre bounded.
    for buy, sell in [(100.0, 0.0), (0.0, 100.0), (60.0, 40.0), (1.0, 999.0)]:
        total = buy + sell
        net = buy - sell
        out = calc({"buy_volume_btc": buy, "sell_volume_btc": sell,
                    "net_flow_1m": net, "net_flow_5m": net, "net_flow_15m": net,
                    "total_volume_1m": total, "total_volume_5m": total,
                    "total_volume_15m": total,
                    "flow_window_integrity": _j2_integrity()})
        for key in ("imbalance_1m", "imbalance_5m", "imbalance_15m"):
            assert -1.0 <= out["ratios"][key] <= 1.0
            assert math.isfinite(out["ratios"][key])
        _no_sentinels(out)


# ── Serialização + payload compact (leitura, sem interpretar) ────────────────

@pytest.mark.parametrize("calc", CALCS)
def test_json_safe_and_compact_passthrough(calc):
    from common.json_safe import json_dumps_rfc8259, sanitize_json_safe

    out = calc({"buy_volume_btc": 3.0, "sell_volume_btc": 2.0,
                "net_flow_1m": 2.0, "net_flow_5m": 8.416, "net_flow_15m": -3.0,
                "total_volume_1m": 5.0, "total_volume_5m": 7.97,
                "total_volume_15m": 15.0,
                "flow_window_integrity": _j2_integrity()})
    clean = sanitize_json_safe(out)
    json_dumps_rfc8259(clean)  # lança se houver NaN/Inf residual

    # Payload compact recebe o evento com a nova metadata sem precisar
    # interpretá-la ainda (P0-B2): só prova que não quebra a construção.
    from market_orchestrator.ai.payload_builder_compact import build_compact_payload
    event = {"symbol": "BTCUSDT", "tipo_evento": "ANALYSIS_TRIGGER",
             "preco_fechamento": 65000.0, "epoch_ms": 1700000000000,
             "ml_features": {}, "multi_tf": {},
             "fluxo_continuo": {"order_flow": {
                 "net_flow_1m": 2.0, "net_flow_5m": 8.416,
                 "buy_sell_ratio": out, "flow_imbalance": 0.4},
                 "flow_window_integrity": _j2_integrity()}}
    payload = build_compact_payload(event)
    assert isinstance(payload, dict)
    json_dumps_rfc8259(sanitize_json_safe(payload))
