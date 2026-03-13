import pytest
from flow_analyzer.absorption import AbsorptionZoneMapper


def test_absorption_zone_mapper_initialization():
    """Testa a inicialização do AbsorptionZoneMapper."""
    mapper = AbsorptionZoneMapper()
    assert isinstance(mapper, AbsorptionZoneMapper)
    
    # Teste com parâmetros customizados
    mapper_custom = AbsorptionZoneMapper(
        zone_tolerance_pct=0.2,
        max_history_hours=12,
        min_index_threshold=0.15
    )
    assert isinstance(mapper_custom, AbsorptionZoneMapper)


def test_absorption_zone_mapper_record_event():
    """Testa o registro de eventos de absorção."""
    mapper = AbsorptionZoneMapper()
    
    # Registro de evento básico
    mapper.record_event(
        price=64800,
        classification="Absorção de Compra",
        index=0.65,
        timestamp_ms=1771888200000,
        buyer_strength=8.5,
        seller_exhaustion=6.2,
        volume_usd=150000
    )
    
    # Verificar que o evento foi registrado
    summary = mapper.get_summary()
    assert summary["status"] == "ok"
    assert summary["total_events"] == 1
    assert summary["buy_absorptions"] == 1
    assert summary["sell_absorptions"] == 0


def test_absorption_zone_mapper_get_zones_empty():
    """Testa a obtenção de zonas com histórico vazio."""
    mapper = AbsorptionZoneMapper()
    zones = mapper.get_zones(current_price=64892)
    
    assert zones["status"] == "no_events"
    assert len(zones["zones"]) == 0
    assert zones["total_zones"] == 0
    assert zones["total_events"] == 0


def test_absorption_zone_mapper_single_zone():
    """Testa a obtenção de zonas com um único evento."""
    mapper = AbsorptionZoneMapper()
    mapper.record_event(
        price=64800,
        classification="Absorção de Compra",
        index=0.65,
        timestamp_ms=1771888200000,
        buyer_strength=8.5,
        seller_exhaustion=6.2,
        volume_usd=150000
    )
    
    zones = mapper.get_zones(current_price=64892)
    assert zones["status"] == "success"
    assert len(zones["zones"]) == 1
    assert zones["total_zones"] == 1
    assert zones["total_events"] == 1
    assert zones["buy_zone_count"] == 1
    assert zones["sell_zone_count"] == 0
    
    # Verificar detalhes da zona
    zone = zones["zones"][0]
    assert zone["center"] == 64800.0
    assert zone["event_count"] == 1
    assert zone["buy_events"] == 1
    assert zone["sell_events"] == 0
    assert zone["dominant_side"] == "buy_defense"
    assert zone["total_strength"] == 0.65
    assert zone["avg_strength"] == 0.65
    assert zone["max_strength"] == 0.65
    assert zone["total_volume_usd"] == 150000.0
    assert zone["last_event_ms"] == 1771888200000
    assert zone["last_classification"] == "Absorção de Compra"
    assert zone["distance_from_price"] == 92.0
    assert zone["direction"] == "below"


def test_absorption_zone_mapper_multiple_events_same_zone():
    """Testa a agregação de eventos na mesma zona."""
    mapper = AbsorptionZoneMapper(zone_tolerance_pct=0.2)
    
    # Registra eventos próximos (muitos compras)
    mapper.record_event(price=64800, classification="Absorção de Compra", index=0.65, timestamp_ms=1771888200000)
    mapper.record_event(price=64810, classification="Absorção de Compra", index=0.70, timestamp_ms=1771888260000)
    mapper.record_event(price=64805, classification="Absorção de Compra", index=0.68, timestamp_ms=1771888320000)
    
    zones = mapper.get_zones(current_price=64892)
    assert zones["total_zones"] == 1
    assert zones["total_events"] == 3
    assert zones["buy_zone_count"] == 1
    
    zone = zones["zones"][0]
    assert zone["event_count"] == 3
    assert zone["buy_events"] == 3
    assert zone["sell_events"] == 0
    assert zone["dominant_side"] == "buy_defense"
    assert zone["total_strength"] > 0.65  # Deve ser soma dos índices
    assert zone["avg_strength"] > 0.65
    assert zone["max_strength"] == 0.70


def test_absorption_zone_mapper_multiple_zones():
    """Testa a criação de múltiplas zonas de absorção."""
    mapper = AbsorptionZoneMapper(zone_tolerance_pct=0.15)
    
    # Zona de compra
    mapper.record_event(price=64800, classification="Absorção de Compra", index=0.65, timestamp_ms=1771888200000)
    mapper.record_event(price=64810, classification="Absorção de Compra", index=0.70, timestamp_ms=1771888260000)
    
    # Zona de venda (distante o suficiente)
    mapper.record_event(price=64950, classification="Absorção de Venda", index=0.75, timestamp_ms=1771888320000)
    mapper.record_event(price=64940, classification="Absorção de Venda", index=0.68, timestamp_ms=1771888380000)
    
    zones = mapper.get_zones(current_price=64892)
    assert zones["total_zones"] == 2
    assert zones["total_events"] == 4
    assert zones["buy_zone_count"] == 1
    assert zones["sell_zone_count"] == 1
    
    # Verificar zona mais forte
    strongest_zone = zones["strongest_zone"]
    assert strongest_zone["total_strength"] > 1.3  # Soma de índices
    assert strongest_zone["dominant_side"] in ["buy_defense", "sell_defense"]


def test_absorption_zone_mapper_reset():
    """Testa a limpeza do histórico de eventos."""
    mapper = AbsorptionZoneMapper()
    mapper.record_event(price=64800, classification="Absorção de Compra", index=0.65)
    
    # Verificar que há eventos
    summary = mapper.get_summary()
    assert summary["total_events"] == 1
    
    # Limpar histórico
    mapper.reset()
    
    # Verificar que está vazio
    summary_after_reset = mapper.get_summary()
    assert summary_after_reset["status"] == "empty"
    assert summary_after_reset["total_events"] == 0
    
    zones_after_reset = mapper.get_zones(current_price=64892)
    assert zones_after_reset["status"] == "no_events"
    assert len(zones_after_reset["zones"]) == 0


def test_absorption_zone_mapper_get_summary():
    """Testa o resumo de absorções."""
    mapper = AbsorptionZoneMapper()
    
    # Sem eventos
    summary = mapper.get_summary()
    assert summary["status"] == "empty"
    assert summary["total_events"] == 0
    
    # Com eventos mistos
    mapper.record_event(price=64800, classification="Absorção de Compra", index=0.65)
    mapper.record_event(price=64810, classification="Absorção de Compra", index=0.70)
    mapper.record_event(price=64950, classification="Absorção de Venda", index=0.75)
    
    summary = mapper.get_summary()
    assert summary["status"] == "ok"
    assert summary["total_events"] == 3
    assert summary["buy_absorptions"] == 2
    assert summary["sell_absorptions"] == 1
    assert summary["avg_index"] > 0.65
    assert summary["dominant_side"] == "buy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])