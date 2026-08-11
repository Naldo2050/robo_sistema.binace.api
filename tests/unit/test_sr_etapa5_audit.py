"""ETAPA 5 — Auditoria matemática e semântica de Support/Resistance.

Reproduções e contratos documentados antes de qualquer patch:
  - J4 (caso real): price=64742.1 → zona center=64771.92 com
    sources=[vp_hvn, orderbook_ask_wall], side ré-emitido aqui.
  - side = voto majoritário por sinal; empate decide por posição.
  - sources canônicos: uma observação econômica = uma entrada
    (rotas duplas sr_level_* + direta são a MESMA fonte).
  - pivots clássicos: fórmulas (H+L+C)/3, 2P-L, 2P-H etc.
  - pivot_points.vah/val/poc source=classic são H/L/C do período
    anterior, NÃO VP — datasets distintos (historical_vp).
"""

import math

import pandas as pd
import pytest

from institutional.enricher import _build_pivot_points
from support_resistance import daily_pivot
from support_resistance.defense_zones import DefenseZoneDetector

PRICE = 64742.1


def _run(current_price=PRICE, orderbook_data=None, vp_data=None,
         sr_levels=None, absorption_events=None, pivot_data=None,
         ema_values=None):
    return DefenseZoneDetector().detect(
        current_price=current_price,
        orderbook_data=orderbook_data,
        vp_data=vp_data,
        sr_levels=sr_levels,
        absorption_events=absorption_events,
        pivot_data=pivot_data,
        ema_values=ema_values,
    )


def _all_zones(result):
    return result["buy_defense"] + result["sell_defense"]


def _zone_with(result, source):
    for z in _all_zones(result):
        if source in z["sources"]:
            return z
    return None


# ---------------------------------------------------------------------------
# FASE 5/7/9 — Reprodução exata da J4 (documenta fórmulas, passa antes e depois)
# ---------------------------------------------------------------------------

class TestJ4Reproduction:
    def test_j4_zone_arithmetic_exact(self):
        res = _run(
            orderbook_data={"bid_depth_usd": 50000, "ask_depth_usd": 100000, "imbalance": -0.25},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64737.0]},
        )
        zone = _zone_with(res, "orderbook_ask_wall")
        assert zone is not None
        assert zone["center"] == 64771.92
        assert zone["range_low"] == 64688.44
        assert zone["range_high"] == 64855.40
        assert zone["strength"] == 52
        assert zone["source_count"] == 2
        assert zone["signals_in_zone"] == 2
        assert zone["type"] == "cluster"
        assert zone["distance_from_price"] == 29.82
        assert set(zone["sources"]) == {"orderbook_ask_wall", "vp_hvn"}

    def test_j4_zone_above_price_is_not_buy_defense(self):
        res = _run(
            orderbook_data={"bid_depth_usd": 50000, "ask_depth_usd": 100000, "imbalance": -0.25},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64737.0]},
        )
        zone = _zone_with(res, "orderbook_ask_wall")
        assert zone is not None
        assert zone["side"] == "sell", (
            "zona centrada ACIMA do preço com ask wall não pode ser buy_defense"
        )
        assert zone in res["sell_defense"]

    def test_ask_wall_alone_is_sell(self):
        res = _run(
            orderbook_data={"bid_depth_usd": 50000, "ask_depth_usd": 100000, "imbalance": -0.3},
        )
        zone = _zone_with(res, "orderbook_ask_wall")
        assert zone is not None
        assert zone["side"] == "sell"

    def test_bid_wall_below_price_is_buy(self):
        res = _run(
            orderbook_data={"bid_depth_usd": 100000, "ask_depth_usd": 50000, "imbalance": 0.25},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64677.0]},
        )
        zone = _zone_with(res, "orderbook_bid_wall")
        assert zone is not None
        assert zone["side"] == "buy"
        assert zone in res["buy_defense"]


# ---------------------------------------------------------------------------
# FASE 13 — Pivot side classification (bug: case-sensitive + keys auxiliares)
# ---------------------------------------------------------------------------

class TestPivotSide:
    def test_pivot_s_level_below_price_is_buy(self):
        res = _run(
            pivot_data={"daily": {"s1": 64690.0, "r1": 64695.0}},
        )
        zone = _zone_with(res, "pivot_daily_s1")
        assert zone is not None
        assert zone["side"] == "buy"

    def test_no_phantom_pivot_ohlc_keys(self):
        res = _run(
            pivot_data={"daily": {"pivot": 64700.0, "high": 64700.0, "low": 64698.0, "close": 64699.0}},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64700.0]},
        )
        for z in _all_zones(res):
            for src in z["sources"]:
                assert not src.startswith("pivot_daily_high"), f"phantom source {src}"
                assert not src.startswith("pivot_daily_low"), f"phantom source {src}"
                assert not src.startswith("pivot_daily_close"), f"phantom source {src}"


# ---------------------------------------------------------------------------
# FASE 3/4 — Deduplicação / aliases (rotas duplas = mesma observação)
# ---------------------------------------------------------------------------

class TestCanonicalSources:
    def test_same_pivot_two_routes_not_independent(self):
        res = _run(
            pivot_data={"daily": {"s1": 64696.0, "pivot": 64700.0, "r1": 65000.0}},
            sr_levels=[{"price": 64696.0, "primary_source": "pivot_daily_s1", "strength": 45}],
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64699.0]},
        )
        zone = _zone_with(res, "vp_hvn")
        assert zone is not None
        assert "pivot_daily_s1" in zone["sources"]
        assert "sr_level_pivot_daily_s1" not in zone["sources"]
        assert zone["source_count"] == 3
        assert zone["signals_in_zone"] == 3

    def test_independent_sources_keep_confluence(self):
        res = _run(
            pivot_data={"daily": {"s1": 64696.0, "pivot": 64700.0, "r1": 65000.0}},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64699.0]},
        )
        zone = _zone_with(res, "vp_hvn")
        assert zone is not None
        assert zone["source_count"] == 3
        assert zone["signals_in_zone"] == 3

    def test_ema_source_name_not_duplicated(self):
        res = _run(
            ema_values={"ema_21_1h": 64698.0},
            sr_levels=[{"price": 64698.0, "primary_source": "ema_21_1h", "strength": 45}],
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64699.0]},
        )
        zone = _zone_with(res, "vp_hvn")
        assert zone is not None
        assert "ema_21_1h" in zone["sources"]
        assert "ema_ema_21_1h" not in zone["sources"]
        assert "sr_level_ema_21_1h" not in zone["sources"]
        assert zone["source_count"] == 2
        assert zone["signals_in_zone"] == 2

    def test_adjacent_hvn_bins_are_independent_observations(self):
        # Bins de $1 do historical_profiler são NODES independentes (um entry
        # por bin com volume acima do threshold). Não colapsam por proximidade
        # — mas também não inflam source_count (mesma fonte canônica).
        res = _run(
            orderbook_data={"bid_depth_usd": 50000, "ask_depth_usd": 100000, "imbalance": -0.25},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64804.0, 64805.0, 64806.0]},
        )
        zone = _zone_with(res, "orderbook_ask_wall")
        assert zone is not None
        assert zone["source_count"] == 2
        assert zone["signals_in_zone"] == 4
        assert zone["side"] == "sell"

    def test_distinct_hvn_nodes_not_collapsed_by_proximity(self):
        # Caso da auditoria: 64804 e 64890 estão a 86 USD (0.13%) — DENTRO da
        # tolerância de zona (0.15%) — mas são nodes distintos do produtor.
        # O dedup NÃO pode colapsá-los (só o clustering os agrupa se for o caso).
        d = DefenseZoneDetector()
        signals = [
            {"price": 64804.0, "source": "vp_hvn", "strength": 25, "side": "sell"},
            {"price": 64890.0, "source": "vp_hvn", "strength": 25, "side": "sell"},
        ]
        out = d._dedupe_signals(signals, PRICE)
        assert len(out) == 2

    def test_identity_dedup_collapses_same_tick_aliases_only(self):
        # Rotas duplas com preço no MESMO tick (round 2) = mesma observação.
        d = DefenseZoneDetector()
        signals = [
            {"price": 64696.0, "source": "pivot_daily_s1", "strength": 20, "side": "buy"},
            {"price": 64696.0, "source": "sr_level_pivot_daily_s1", "strength": 30, "side": "buy"},
        ]
        out = d._dedupe_signals(signals, PRICE)
        assert len(out) == 1
        assert out[0]["source"] == "pivot_daily_s1"
        assert out[0]["strength"] == 30

    def test_identity_dedup_keeps_same_tick_different_source(self):
        # Fontes DIFERENTES no mesmo preço são observações diferentes
        # (confluência real) — não colapsam.
        d = DefenseZoneDetector()
        signals = [
            {"price": 64696.0, "source": "pivot_daily_s1", "strength": 20, "side": "buy"},
            {"price": 64696.0, "source": "ema_21_1d", "strength": 30, "side": "buy"},
        ]
        out = d._dedupe_signals(signals, PRICE)
        assert len(out) == 2

    def test_canonical_source_not_double_counted_even_if_price_diverges(self):
        # O merge do scorer pode deslocar o preço (média de grupo), mas a
        # fonte canônica ainda conta 1 na confluência da zona.
        res = _run(
            vp_data={"poc": 64689.0, "vah": 0, "val": 0, "hvns": [64690.0]},
            sr_levels=[{"price": 64702.48, "primary_source": "poc_daily", "strength": 45}],
        )
        zone = _zone_with(res, "vp_poc")
        assert zone is not None
        assert zone["source_count"] == 2
        assert zone["signals_in_zone"] == 3
        assert set(zone["sources"]) == {"vp_hvn", "vp_poc"}

    def test_ema_legacy_tf_source_keeps_prefix(self):
        # Formato legado ("1d") preserva o prefixo ema_ (contrato HEAD).
        res = _run(
            ema_values={"1d": 64698.0},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64699.0]},
        )
        zone = _zone_with(res, "ema_1d")
        assert zone is not None
        assert zone["source_count"] == 2


# ---------------------------------------------------------------------------
# FASE 15 — Tie-break de side (empate decidido por posição em relação ao preço)
# ---------------------------------------------------------------------------

class TestTieBreakSide:
    def test_tie_center_below_price_is_buy(self):
        res = _run(
            pivot_data={"daily": {"s1": 64690.0, "r1": 64700.0}},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": []},
        )
        zone = _zone_with(res, "pivot_daily_s1")
        assert zone is not None
        assert zone["source_count"] == 2
        assert zone["center"] == 64695.0
        assert zone["center"] < PRICE
        assert zone["side"] == "buy"

    def test_tie_center_above_price_is_sell(self):
        res = _run(
            pivot_data={"daily": {"s1": 64750.0, "r1": 64760.0}},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": []},
        )
        zone = _zone_with(res, "pivot_daily_s1")
        assert zone is not None
        assert zone["center"] == 64755.0
        assert zone["center"] > PRICE
        assert zone["side"] == "sell"

    def test_tie_center_equal_price_is_sell(self):
        # Contrato: buy apenas se center < price (espelha a classificação
        # final em detect(): zona sem side com center >= price → sell_defense).
        res = _run(
            pivot_data={"daily": {"s1": PRICE, "r1": PRICE}},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": []},
        )
        zone = _zone_with(res, "pivot_daily_s1")
        assert zone is not None
        assert zone["center"] == PRICE
        assert zone["side"] == "sell"


# ---------------------------------------------------------------------------
# FASE 11 — immediate_support/resistance (fórmula dominada por proximidade)
# ---------------------------------------------------------------------------

class TestImmediateLevels:
    def test_support_strength_proximity_formula(self):
        event = {
            "preco_fechamento": PRICE,
            "pivots": {"daily": {
                "pivot": 65035.38, "r1": 65340.67, "s1": 64596.29,
                "r2": 65779.76, "s2": 64291.0, "r3": 66085.05, "s3": 63851.91,
                "high": 65474.46, "low": 64730.08, "close": 64901.59,
            }},
            "historical_vp": {"daily": {"poc": 64689, "vah": 65133, "val": 64520, "status": "success"}},
        }
        sr = _build_pivot_points(event)
        assert 64730.08 in sr["immediate_support"]
        idx = sr["immediate_support"].index(64730.08)
        dist_pct = abs(64730.08 - PRICE) / PRICE * 100
        expected = min(100.0, round(1.0 * 100 * (1 - dist_pct / 10), 1))
        assert sr["support_strength"][idx] == pytest.approx(expected, abs=0.15)
        assert expected > 99.0, "nível a ~0.02% do preço satura em ~99.8 (proximidade)"

    def test_pivot_points_vah_val_poc_are_classic_ohlc(self):
        event = {
            "preco_fechamento": PRICE,
            "pivots": {"daily": {
                "pivot": 65035.38, "r1": 65340.67, "s1": 64596.29,
                "r2": 65779.76, "s2": 64291.0, "r3": 66085.05, "s3": 63851.91,
                "high": 65474.46, "low": 64730.08, "close": 64901.59,
            }},
            "historical_vp": {"daily": {"poc": 64689, "vah": 65133, "val": 64520, "status": "success"}},
        }
        sr = _build_pivot_points(event)
        daily = sr["pivot_points"]["daily"]
        assert daily["source"] == "classic"
        assert daily["vah"] == 65474.46
        assert daily["val"] == 64730.08
        assert daily["poc"] == 64901.59
        assert event["historical_vp"]["daily"]["vah"] == 65133
        assert daily["vah"] != event["historical_vp"]["daily"]["vah"]


# ---------------------------------------------------------------------------
# FASE 11b — Contaminação do payload: zona no lado errado do preço
# ---------------------------------------------------------------------------

class TestImmediateLevelsDefenseSide:
    def _event(self, dz):
        return {
            "preco_fechamento": PRICE,
            "institutional_analytics": {"sr_analysis": {"defense_zones": dz}},
            "pivots": {"daily": {
                "pivot": 65035.38, "r1": 65340.67, "s1": 64596.29,
                "r2": 65779.76, "s2": 64291.0, "r3": 66085.05, "s3": 63851.91,
                "high": 65474.46, "low": 64730.08, "close": 64901.59,
            }},
            "historical_vp": {"daily": {"poc": 64689, "vah": 65133, "val": 64520, "status": "success"}},
        }

    def test_buy_zone_above_price_not_in_immediate_support(self):
        # 2 fontes buy-naturais ACIMA do preço (pivot s1 + pivot) -> zona buy
        # com center acima do preço. NÃO pode virar immediate_support.
        dz = DefenseZoneDetector().detect(
            current_price=PRICE,
            pivot_data={"daily": {"s1": 64900.0, "pivot": 64950.0, "r1": 65000.0}},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": []},
        )
        buy_above = [z for z in dz["buy_defense"] if z["center"] > PRICE]
        assert buy_above, "cénario deve gerar zona buy acima do preço"
        sr = _build_pivot_points(self._event(dz))
        sup = sr.get("immediate_support", [])
        assert all(p <= PRICE for p in sup), f"suporte acima do preço: {sup}"
        assert round(buy_above[0]["center"], 2) not in sup

    def test_sell_zone_below_price_not_in_immediate_resistance(self):
        # 2 fontes sell-naturais ABAIXO do preço (r2 + r3) -> zona sell com
        # center abaixo do preço. NÃO pode virar immediate_resistance.
        dz = DefenseZoneDetector().detect(
            current_price=PRICE,
            pivot_data={"daily": {"r2": 64600.0, "r3": 64650.0}},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": []},
        )
        sell_below = [z for z in dz["sell_defense"] if z["center"] < PRICE]
        assert sell_below, "cénario deve gerar zona sell abaixo do preço"
        sr = _build_pivot_points(self._event(dz))
        res = sr.get("immediate_resistance", [])
        assert all(p >= PRICE for p in res), f"resistência abaixo do preço: {res}"
        assert round(sell_below[0]["center"], 2) not in res

    def test_buy_zone_below_price_in_immediate_support(self):
        # Guarda não bloqueia o caso legítimo: buy zone ABAIXO do preço.
        dz = DefenseZoneDetector().detect(
            current_price=PRICE,
            pivot_data={"daily": {"s1": 64690.0, "pivot": 64700.0, "r1": 64720.0}},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": []},
        )
        buy_below = [z for z in dz["buy_defense"] if z["center"] < PRICE]
        assert buy_below, "cénario deve gerar zona buy abaixo do preço"
        sr = _build_pivot_points(self._event(dz))
        assert round(buy_below[0]["center"], 2) in sr.get("immediate_support", [])

    def test_sell_zone_above_price_in_immediate_resistance(self):
        # Guarda não bloqueia o caso legítimo: sell zone ACIMA do preço.
        dz = DefenseZoneDetector().detect(
            current_price=PRICE,
            pivot_data={"daily": {"r1": 64760.0, "r2": 64770.0}},
            vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": []},
        )
        sell_above = [z for z in dz["sell_defense"] if z["center"] > PRICE]
        assert sell_above, "cénario deve gerar zona sell acima do preço"
        sr = _build_pivot_points(self._event(dz))
        assert round(sell_above[0]["center"], 2) in sr.get("immediate_resistance", [])

    def test_buy_zone_center_equal_price_in_immediate_support(self):
        # Borda inclusiva: center == preço é suporte imediato (<=), consistente
        # com o filtro h/pivot/l do enricher (level > price → resistance).
        sr = _build_pivot_points(self._event({
            "buy_defense": [{"center": PRICE, "strength": 50}],
            "sell_defense": [],
        }))
        assert PRICE in sr.get("immediate_support", [])

    def test_sell_zone_center_equal_price_not_in_immediate_resistance(self):
        # Borda: sell zone == preço NÃO entra em resistance (estritamente >).
        # Determínistico e espelha h/pivot/l (== → support).
        sr = _build_pivot_points(self._event({
            "buy_defense": [],
            "sell_defense": [{"center": PRICE, "strength": 50}],
        }))
        assert PRICE not in sr.get("immediate_resistance", [])


# ---------------------------------------------------------------------------
# FASE 13 — Pivot points matemática (classic)
# ---------------------------------------------------------------------------

class TestPivotClassicMath:
    def test_daily_pivot_formulas(self):
        df = pd.DataFrame({
            "high": [65474.46, 65322.58],
            "low": [64730.08, 64826.78],
            "close": [64901.59, 65012.01],
        })
        p = daily_pivot(df)
        expected = {
            "pivot": 65035.37667,
            "r1": 65340.67333,
            "s1": 64596.29333,
            "r2": 65779.75667,
            "s2": 64290.99667,
            "r3": 66085.05333,
            "s3": 63851.91333,
        }
        for key, val in expected.items():
            assert p[key] == pytest.approx(val, abs=0.01), f"{key} divergente"