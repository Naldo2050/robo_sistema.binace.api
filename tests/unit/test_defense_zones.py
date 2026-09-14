import pytest
from support_resistance.defense_zones import DefenseZoneDetector


def test_defense_zone_detector_initialization():
    """Testa a inicialização do detector de zonas de defesa."""
    detector = DefenseZoneDetector()
    assert isinstance(detector, DefenseZoneDetector)


def test_defense_zone_detector_with_minimal_data():
    """Testa a detecção de zonas de defesa com dados mínimos."""
    detector = DefenseZoneDetector()
    
    result = detector.detect(
        current_price=64892,
        orderbook_data={
            "bid_depth_usd": 1000000,
            "ask_depth_usd": 500000,
            "imbalance": 0.1,
            "depth_metrics": {"depth_imbalance": 0.15},
            "clusters": [
                {"center": 64850, "total_volume": 10, "imbalance_ratio": 0.2},
                {"center": 64950, "total_volume": 8, "imbalance_ratio": -0.15}
            ]
        },
        vp_data={
            "poc": 64880,
            "vah": 64920,
            "val": 64850,
            "hvns": [64850, 64900]
        },
        sr_levels=[
            {"price": 64850, "strength": 85, "type": "support", "primary_source": "swing"},
            {"price": 64920, "strength": 90, "type": "resistance", "primary_source": "volume"}
        ],
        absorption_events=[
            {"price": 64850, "type": "buy", "strength": 0.8},
            {"price": 64920, "type": "sell", "strength": 0.9}
        ],
        pivot_data={
            "standard": {"S1": 64840, "PP": 64880, "R1": 64920},
            "fibonacci": {"S1": 64830, "PP": 64880, "R1": 64930}
        },
        ema_values={
            "1d": 64870,
            "4h": 64885,
            "1h": 64890
        }
    )
    
    assert result["status"] == "success"
    assert result["total_zones"] > 0
    assert "buy_defense" in result
    assert "sell_defense" in result
    assert isinstance(result["defense_asymmetry"], dict)
    assert "ratio" in result["defense_asymmetry"]


def test_defense_zone_detector_with_empty_data():
    """Testa a detecção de zonas de defesa com dados vazios."""
    detector = DefenseZoneDetector()
    
    result = detector.detect(
        current_price=64892,
        orderbook_data=None,
        vp_data=None,
        sr_levels=None,
        absorption_events=None,
        pivot_data=None,
        ema_values=None
    )
    
    assert result["status"] == "no_data"
    assert result["total_zones"] == 0
    assert len(result["buy_defense"]) == 0
    assert len(result["sell_defense"]) == 0


def test_defense_zone_detector_with_invalid_price():
    """Testa a detecção de zonas de defesa com preço inválido."""
    detector = DefenseZoneDetector()
    
    result = detector.detect(
        current_price=0,
        orderbook_data={},
        vp_data={},
        sr_levels=[],
        absorption_events=[],
        pivot_data={},
        ema_values={}
    )
    
    assert result["status"] == "no_data"
    assert result["total_zones"] == 0


def test_defense_zone_detector_custom_parameters():
    """Testa a detecção de zonas de defesa com parâmetros customizados."""
    detector = DefenseZoneDetector(
        zone_width_pct=0.2,
        min_sources_for_zone=3,
        max_zones_per_side=3
    )
    
    result = detector.detect(
        current_price=64892,
        orderbook_data={
            "bid_depth_usd": 1000000,
            "ask_depth_usd": 500000,
            "imbalance": 0.1,
            "depth_metrics": {"depth_imbalance": 0.15},
            "clusters": [
                {"center": 64850, "total_volume": 10, "imbalance_ratio": 0.2},
                {"center": 64950, "total_volume": 8, "imbalance_ratio": -0.15}
            ]
        },
        vp_data={
            "poc": 64880,
            "vah": 64920,
            "val": 64850,
            "hvns": [64850, 64900]
        },
        sr_levels=[
            {"price": 64850, "strength": 85, "type": "support", "primary_source": "swing"},
            {"price": 64920, "strength": 90, "type": "resistance", "primary_source": "volume"}
        ],
        absorption_events=[
            {"price": 64850, "type": "buy", "strength": 0.8},
            {"price": 64920, "type": "sell", "strength": 0.9}
        ],
        pivot_data={
            "standard": {"S1": 64840, "PP": 64880, "R1": 64920},
            "fibonacci": {"S1": 64830, "PP": 64880, "R1": 64930}
        },
        ema_values={
            "1d": 64870,
            "4h": 64885,
            "1h": 64890
        }
    )
    
    assert result["status"] == "success"
    assert "buy_defense" in result
    assert "sell_defense" in result
    assert len(result["buy_defense"]) <= 3
    assert len(result["sell_defense"]) <= 3


# ---------------------------------------------------------------------------
# FIX provenance 2026-09 — contrato de wall observada.
# ---------------------------------------------------------------------------

# Snapshot forense real (E=1789158300976): walls detectadas por _detect_walls
# no top-50 com quantil-90% × 3.0.
FORENSIC_BID_WALL = {"side": "bid", "price": 77399.0, "qty": 20.809, "limit_threshold": 7.5033}
FORENSIC_ASK_WALL = {"side": "ask", "price": 77399.1, "qty": 1.334, "limit_threshold": 0.3906}
FORENSIC_PRICE = 77399.1


def _detect_ob_walls(current_price=FORENSIC_PRICE, walls=None):
    ob = {
        "bid_depth_usd": 3360755.38,
        "ask_depth_usd": 226553.4,
        "imbalance": 0.8737,
        "walls": walls if walls is not None else {
            "bids": [dict(FORENSIC_BID_WALL)],
            "asks": [dict(FORENSIC_ASK_WALL)],
        },
    }
    return DefenseZoneDetector().detect(current_price=current_price, orderbook_data=ob)


def test_real_wall_buy_center_is_wall_price():
    """REAL WALL TEST (BUY): center = 77399.0, nunca current_price*0.999."""
    res = _detect_ob_walls()
    assert res["status"] == "success"
    assert len(res["buy_defense"]) == 1
    zone = res["buy_defense"][0]
    assert zone["center"] == 77399.0
    assert zone["center"] != round(FORENSIC_PRICE * 0.999, 2)
    assert zone["sources"] == ["orderbook_bid_wall"]
    assert zone["source_count"] == 1
    assert zone["signals_in_zone"] == 1
    assert zone["type"] == "single"


def test_real_wall_sell_center_is_wall_price():
    """REAL WALL TEST (SELL): center = 77399.1, nunca current_price*1.001."""
    res = _detect_ob_walls()
    assert len(res["sell_defense"]) == 1
    zone = res["sell_defense"][0]
    assert zone["center"] == 77399.1
    assert zone["center"] != round(FORENSIC_PRICE * 1.001, 2)
    assert zone["sources"] == ["orderbook_ask_wall"]


def test_real_wall_strength_is_evidence_based_and_capped():
    """Força por excesso qty/threshold; single-wall nunca atinge 52."""
    res = _detect_ob_walls()
    buy = res["buy_defense"][0]
    sell = res["sell_defense"][0]
    # bid ratio 20.809/7.5033 ≈ 2.77 → 27.73×1.3 ≈ 36
    assert buy["strength"] == 36
    # ask ratio 1.334/0.3906 ≈ 3.42 → cap 30×1.3 = 39
    assert sell["strength"] == 39
    assert buy["strength"] < 52
    assert sell["strength"] < 52


def test_no_wall_with_high_imbalance_produces_no_wall_zone():
    """NO-WALL TEST: imbalance alto + walls vazias → sem orderbook_*_wall."""
    res = _detect_ob_walls(walls={"bids": [], "asks": []})
    assert res["status"] == "no_data"
    assert res["total_zones"] == 0


def test_no_walls_key_produces_no_wall_zone():
    """orderbook_data legado (sem chave walls) também não projeta nível."""
    res = DefenseZoneDetector().detect(
        current_price=FORENSIC_PRICE,
        orderbook_data={
            "bid_depth_usd": 3360755.38,
            "ask_depth_usd": 226553.4,
            "imbalance": 0.8737,
        },
    )
    all_zones = res["buy_defense"] + res["sell_defense"]
    assert all(
        "orderbook_bid_wall" not in z["sources"]
        and "orderbook_ask_wall" not in z["sources"]
        for z in all_zones
    )


def test_provenance_invariant_for_observed_walls():
    """Toda zona orderbook_*_wall é rastreável à wall observada (SNAPSHOT_ONLY)."""
    res = _detect_ob_walls()
    zones = [z for z in res["buy_defense"] + res["sell_defense"]
             if any(s.startswith("orderbook_") and s.endswith("_wall") for s in z["sources"])]
    assert len(zones) == 2
    for zone in zones:
        assert zone["observed"] is True
        assert zone["projected"] is False
        assert zone["basis"] == "rest_l2_snapshot"
        assert zone["snapshot_scope"] == "top50"
        assert zone["snapshot_only"] is True
        assert zone["persistence_confirmed"] is False
        assert zone["center_origin"] == "observed_wall_price"
        assert zone["center"] in zone["wall_prices"]
        assert zone["wall_qty_btc"] > 0
        assert zone["wall_notional_usd"] > 0


def test_opposite_side_walls_never_merge():
    """Bid wall e ask wall no toque formam zonas buy/sell separadas."""
    res = _detect_ob_walls()
    assert len(res["buy_defense"]) == 1
    assert len(res["sell_defense"]) == 1
    assert res["buy_defense"][0]["center"] == 77399.0
    assert res["sell_defense"][0]["center"] == 77399.1


def test_same_side_walls_anchor_on_observed_mean():
    """Multi-wall mesmo lado: center = média de preços observados."""
    res = _detect_ob_walls(walls={
        "bids": [
            {"side": "bid", "price": 77399.0, "qty": 20.809, "limit_threshold": 7.5033},
            {"side": "bid", "price": 77395.0, "qty": 10.0, "limit_threshold": 7.5033},
        ],
        "asks": [],
    })
    assert len(res["buy_defense"]) == 1
    zone = res["buy_defense"][0]
    assert zone["center"] == 77397.0
    assert zone["observed"] is True
    assert zone["center_origin"] == "observed_wall_price"
    assert zone["wall_prices"] == [77395.0, 77399.0]


def test_wall_ratio_preserved_on_signal_and_zone():
    """wall_ratio = qty/threshold auditável viaja do sinal até a zona."""
    res = _detect_ob_walls()
    buy = res["buy_defense"][0]
    sell = res["sell_defense"][0]
    assert buy["wall_ratio"] == round(20.809 / 7.5033, 4)
    assert sell["wall_ratio"] == round(1.334 / 0.3906, 4)
    assert buy["wall_qty_btc"] == 20.809
    assert buy["wall_notional_usd"] == round(77399.0 * 20.809, 2)


def test_wall_only_flags_without_structural_source():
    """Wall-only: liquidity_only=True, sem confluência estrutural."""
    from support_resistance.defense_zones import is_wall_only_zone, has_structural_confluence
    res = _detect_ob_walls()
    for zone in res["buy_defense"] + res["sell_defense"]:
        assert zone["liquidity_only"] is True
        assert zone["has_structural_confluence"] is False
        assert zone["structural_sources"] == []
        assert is_wall_only_zone(zone) is True
        assert has_structural_confluence(zone) is False


def test_multi_wall_same_side_is_still_liquidity_only():
    """2-5 walls BID sem fonte estrutural: cluster mas continua liquidity_only."""
    from support_resistance.defense_zones import is_wall_only_zone
    res = _detect_ob_walls(walls={
        "bids": [
            {"side": "bid", "price": 77399.0, "qty": 20.809, "limit_threshold": 7.5033},
            {"side": "bid", "price": 77395.0, "qty": 10.0, "limit_threshold": 7.5033},
        ],
        "asks": [],
    })
    zone = res["buy_defense"][0]
    assert zone["signals_in_zone"] == 2
    assert zone["liquidity_only"] is True
    assert zone["has_structural_confluence"] is False
    assert is_wall_only_zone(zone) is True


def test_wall_plus_vp_has_structural_confluence():
    """Wall + VP: confluência estrutural preservada, center na wall."""
    from support_resistance.defense_zones import is_wall_only_zone, has_structural_confluence
    res = DefenseZoneDetector().detect(
        current_price=64742.1,
        orderbook_data={
            "bid_depth_usd": 1000000, "ask_depth_usd": 500000, "imbalance": 0.25,
            "walls": {"bids": [{"side": "bid", "price": 64705.0, "qty": 2.0,
                                "limit_threshold": 1.0}], "asks": []},
        },
        vp_data={"poc": 0, "vah": 0, "val": 0, "hvns": [64700.0]},
    )
    zones = [z for z in res["buy_defense"] if "orderbook_bid_wall" in z["sources"]]
    assert len(zones) == 1
    zone = zones[0]
    assert zone["center"] == 64705.0
    assert zone["has_structural_confluence"] is True
    assert zone["liquidity_only"] is False
    assert "vp_hvn" in zone["structural_sources"]
    assert is_wall_only_zone(zone) is False
    assert has_structural_confluence(zone) is True


def test_wall_only_helpers_cover_legacy_stored_zones():
    """Zonas antigas (só sources, sem flags) também são reconhecidas."""
    from support_resistance.defense_zones import is_wall_only_zone, has_structural_confluence
    assert is_wall_only_zone({"sources": ["orderbook_bid_wall"]}) is True
    assert is_wall_only_zone({"sources": ["orderbook_bid_wall", "orderbook_ask_wall"]}) is True
    assert is_wall_only_zone({"sources": ["orderbook_bid_wall", "vp_hvn"]}) is False
    assert is_wall_only_zone({"sources": ["vp_hvn"]}) is False
    assert is_wall_only_zone({"center": 1.0}) is False
    assert has_structural_confluence({"sources": ["orderbook_bid_wall", "vp_hvn"]}) is True
    assert has_structural_confluence({"sources": ["orderbook_bid_wall"]}) is False


# ---------------------------------------------------------------------------
# FIX 2026-09 (semantic closure) — depth_asymmetry é PROJECTED_HEURISTIC.
# ---------------------------------------------------------------------------

DEPTH_OB = {
    "bid_depth_usd": 1000000,
    "ask_depth_usd": 1000000,
    "imbalance": 0.0,
    "depth_metrics": {"depth_imbalance": -0.5},
}
DEPTH_PRICE = round(77399.1 * 1.002, 2)  # 77553.9


def _detect_depth(current_price=77399.1, ob=None, **kwargs):
    return DefenseZoneDetector().detect(
        current_price=current_price,
        orderbook_data=dict(DEPTH_OB) if ob is None else ob,
        **kwargs,
    )


def test_depth_only_signal_carries_projected_provenance():
    """Sinal depth: observed=False, projected=True, basis=depth_asymmetry."""
    det = DefenseZoneDetector()
    signals = det._extract_orderbook_defense(dict(DEPTH_OB), 77399.1)
    assert len(signals) == 1
    sig = signals[0]
    assert sig["source"] == "depth_asymmetry"
    assert sig["price"] == 77399.1 * 1.002
    assert sig["observed"] is False
    assert sig["projected"] is True
    assert sig["basis"] == "depth_asymmetry"
    assert sig["snapshot_only"] is True
    assert sig["persistence_confirmed"] is False


def test_depth_only_zone_is_projected_only():
    """depth-only: zona projected_only, sem tick observado."""
    from support_resistance.defense_zones import is_non_structural_zone
    res = _detect_depth()
    zones = res["sell_defense"]
    assert len(zones) == 1
    zone = zones[0]
    assert zone["center"] == DEPTH_PRICE
    assert zone["sources"] == ["depth_asymmetry"]
    assert zone["observed"] is False
    assert zone["projected"] is True
    assert zone["projected_only"] is True
    assert zone["has_structural_confluence"] is False
    assert zone["structural_sources"] == []
    assert is_non_structural_zone(zone) is True


def test_depth_only_not_promoted_to_immediate_or_sr():
    """DEPTH-ONLY TEST: sem immediate, sem s1/r1; evidência fica no full."""
    from institutional.enricher import _build_pivot_points
    from market_orchestrator.ai.payload_builder_compact import _build_sr
    res = _detect_depth()
    assert res["status"] == "success"
    ev = {"preco_fechamento": 77399.1,
          "institutional_analytics": {"sr_analysis": {"defense_zones": res}}}
    sr = _build_pivot_points(ev)
    assert sr.get("immediate_support", []) == []
    assert sr.get("immediate_resistance", []) == []
    compact = _build_sr(ev)
    assert "s1" not in compact
    assert "r1" not in compact


def test_depth_plus_pivot_confluence_promoted_with_provenance():
    """DEPTH+PIVOT: confluência permitida; provenance revela projected+structural."""
    from market_orchestrator.ai.payload_builder_compact import _build_sr
    res = _detect_depth(pivot_data={"daily": {"r1": 77560.0}})
    confl = [z for z in res["sell_defense"] if z.get("has_structural_confluence")]
    assert len(confl) == 1
    zone = confl[0]
    assert set(zone["sources"]) == {"depth_asymmetry", "pivot_daily_r1"}
    assert zone["has_projected_component"] is True
    assert "pivot_daily_r1" in zone["structural_sources"]
    assert zone.get("projected_only", False) is False
    # center é média (sem âncora observada) — documentado, não finge tick.
    assert zone["center"] == round((DEPTH_PRICE + 77560.0) / 2, 2)
    ev = {"preco_fechamento": 77399.1,
          "institutional_analytics": {"sr_analysis": {"defense_zones": res}}}
    compact = _build_sr(ev)
    assert "r1" in compact
    assert compact.get("r1_src") == "PROJECTED_DEPTH"


def test_depth_plus_wall_is_not_structural_confluence():
    """DEPTH+WALL: mesmo snapshot L2 → sem confluência, sem promoção."""
    from institutional.enricher import _build_pivot_points
    from market_orchestrator.ai.payload_builder_compact import _build_sr
    res = DefenseZoneDetector().detect(
        current_price=77399.1,
        orderbook_data={
            "bid_depth_usd": 1000000, "ask_depth_usd": 1000000, "imbalance": 0.0,
            "depth_metrics": {"depth_imbalance": -0.5},
            "walls": {"bids": [{"side": "bid", "price": 77399.0, "qty": 20.809,
                                "limit_threshold": 7.5033}], "asks": []},
        },
    )
    zones = [z for z in res["buy_defense"] + res["sell_defense"]
             if "depth_asymmetry" in z["sources"] or "orderbook_bid_wall" in z["sources"]]
    assert len(zones) == 2  # wall buy single + depth sell single, sem fusão
    for z in zones:
        assert z.get("has_structural_confluence", False) is False
    ev = {"preco_fechamento": 77399.1,
          "institutional_analytics": {"sr_analysis": {"defense_zones": res}}}
    sr = _build_pivot_points(ev)
    assert sr.get("immediate_support", []) == []
    assert sr.get("immediate_resistance", []) == []
    assert "s1" not in _build_sr(ev) and "r1" not in _build_sr(ev)


def test_is_non_structural_zone_legacy_coverage():
    """Helper unificado cobre flags novas e zones legadas só-com-sources."""
    from support_resistance.defense_zones import is_non_structural_zone
    assert is_non_structural_zone({"liquidity_only": True}) is True
    assert is_non_structural_zone({"projected_only": True}) is True
    assert is_non_structural_zone({"has_structural_confluence": True}) is False
    assert is_non_structural_zone({"sources": ["depth_asymmetry"]}) is True
    assert is_non_structural_zone(
        {"sources": ["orderbook_bid_wall", "depth_asymmetry"]}) is True
    assert is_non_structural_zone(
        {"sources": ["depth_asymmetry", "pivot_daily_r1"]}) is False
    assert is_non_structural_zone({"sources": ["vp_hvn"]}) is False
    assert is_non_structural_zone({"center": 1.0}) is False
    assert is_non_structural_zone(None) is False


def test_invalid_walls_are_ignored_never_projected():
    """Walls malformadas (preço/qty inválidos) não geram sinal nem projeção."""
    res = _detect_ob_walls(walls={
        "bids": [{"side": "bid", "price": 0, "qty": 5.0, "limit_threshold": 1.0},
                 {"side": "bid", "price": 77399.0, "qty": 0, "limit_threshold": 1.0},
                 "not-a-dict"],
        "asks": [],
    })
    assert res["status"] == "no_data"


def test_enrich_propagates_observed_walls_into_orderbook_data():
    """Pré-check §1: walls do evento analyzer chegam a signal/orderbook_data."""
    from market_orchestrator.market_orchestrator import EnhancedMarketBot
    signal: dict = {}
    ob_event = {
        "is_valid": True,
        "orderbook_data": {"mid": 77399.05, "bid_depth_usd": 3360755.38,
                           "ask_depth_usd": 226553.4, "imbalance": 0.8737},
        "walls": {"bids": [dict(FORENSIC_BID_WALL)], "asks": [dict(FORENSIC_ASK_WALL)]},
    }
    EnhancedMarketBot._enrich_orderbook_metrics(signal, ob_event)
    walls = signal["orderbook_data"]["walls"]
    assert walls["bids"][0]["price"] == 77399.0
    assert walls["bids"][0]["qty"] == 20.809
    assert walls["asks"][0]["price"] == 77399.1
    # Detector consome o formato propagado sem projeção.
    res = DefenseZoneDetector().detect(
        current_price=FORENSIC_PRICE, orderbook_data=signal["orderbook_data"])
    assert res["buy_defense"][0]["center"] == 77399.0
    assert res["sell_defense"][0]["center"] == 77399.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])