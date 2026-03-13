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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])