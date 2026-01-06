import pytest
import time
import threading
from datetime import datetime
import os
import sys

# Garante que a raiz do projeto esteja no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from event_bus import EventBus

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def bus():
    """Fixture que cria uma instância limpa do EventBus para cada teste."""
    # Cria o barramento com janela de deduplicação curta para testes
    event_bus = EventBus(max_queue_size=100, deduplication_window=1)
    
    # Aguarda a thread iniciar
    time.sleep(0.1)
    
    yield event_bus
    
    # Teardown: encerra o barramento
    event_bus.shutdown()
    time.sleep(0.1)

# ==========================================
# TESTES
# ==========================================

def test_subscribe_and_publish_single(bus):
    """Testa publicação para um único assinante."""
    received_events = []
    
    def callback(event):
        received_events.append(event)
    
    bus.subscribe("topic_A", callback)
    
    # Publica evento com timestamp para garantir ID único
    payload = {"id": 1, "msg": "teste", "timestamp": time.time()}
    bus.publish("topic_A", payload)
    
    # Aguarda processamento (thread)
    time.sleep(0.2)
    
    assert len(received_events) == 1
    assert received_events[0]["msg"] == "teste"

def test_multiple_subscribers(bus):
    """Testa múltiplos assinantes no mesmo tópico."""
    sub1_events = []
    sub2_events = []
    
    bus.subscribe("topic_multi", lambda e: sub1_events.append(e))
    bus.subscribe("topic_multi", lambda e: sub2_events.append(e))
    
    bus.publish("topic_multi", {"data": "shared", "timestamp": time.time()})
    
    time.sleep(0.2)
    
    assert len(sub1_events) == 1
    assert len(sub2_events) == 1
    assert sub1_events[0]["data"] == "shared"
    assert sub2_events[0]["data"] == "shared"

def test_different_topics(bus):
    """Testa isolamento entre tópicos diferentes."""
    topic_a_events = []
    topic_b_events = []
    
    bus.subscribe("topic_A", lambda e: topic_a_events.append(e))
    bus.subscribe("topic_B", lambda e: topic_b_events.append(e))
    
    # Publica em A
    bus.publish("topic_A", {"source": "A", "timestamp": time.time()})
    
    time.sleep(0.2)
    
    assert len(topic_a_events) == 1
    assert len(topic_b_events) == 0
    assert topic_a_events[0]["source"] == "A"

def test_publish_no_subscribers(bus):
    """Testa publicação em tópico sem assinantes (não deve quebrar)."""
    try:
        bus.publish("topic_void", {"msg": "hello", "timestamp": time.time()})
        time.sleep(0.1)
    except Exception as e:
        pytest.fail(f"Publicar sem assinantes causou exceção: {e}")

def test_callback_data_integrity_and_normalization(bus):
    """
    Garante que o callback recebe os dados e verifica a normalização padrão.
    O EventBus normaliza 'price' para 4 casas e 'volume' para 8 casas.
    """
    received = []
    bus.subscribe("topic_data", lambda e: received.append(e))
    
    data = {
        "price": 123.456789,       # Esperado: 123.4568 (4 casas)
        "volume": 1.123456789,     # Esperado: 1.12345679 (8 casas)
        "timestamp": time.time()
    }
    
    bus.publish("topic_data", data)
    time.sleep(0.2)
    
    assert len(received) == 1
    event = received[0]
    
    # Verifica normalização numérica
    assert event["price"] == 123.4568
    assert event["volume"] == 1.12345679

def test_shutdown_behavior(bus):
    """Testa comportamento após shutdown."""
    received = []
    bus.subscribe("topic_shutdown", lambda e: received.append(e))
    
    # 1. Evento antes do shutdown
    bus.publish("topic_shutdown", {"seq": 1, "timestamp": time.time()})
    time.sleep(0.2)
    assert len(received) == 1
    
    # 2. Shutdown
    bus.shutdown()
    
    # 3. Evento após shutdown (thread parada, não deve processar)
    bus.publish("topic_shutdown", {"seq": 2, "timestamp": time.time() + 1})
    time.sleep(0.2)
    
    # Ainda deve ter apenas 1 evento processado
    assert len(received) == 1

def test_deduplication(bus):
    """
    Testa se eventos duplicados são ignorados.
    O EventBus gera ID baseado em: timestamp, delta, volume, price.
    """
    received = []
    bus.subscribe("topic_dedup", lambda e: received.append(e))
    
    # Mesmo timestamp e dados = mesmo ID gerado -> duplicata
    fixed_ts = 1700000000000
    data = {
        "price": 50000,
        "volume": 1.5,
        "timestamp": fixed_ts
    }
    
    # Publica duas vezes o mesmo evento
    bus.publish("topic_dedup", data)
    bus.publish("topic_dedup", data)
    
    time.sleep(0.2)
    
    # Deve receber apenas 1
    assert len(received) == 1

def test_normalization_bypass(bus):
    """Testa publicação com normalize=False (preserva dados brutos)."""
    received = []
    bus.subscribe("topic_raw", lambda e: received.append(e))
    
    raw_data = {
        "price": 123.456789,
        "timestamp": time.time()
    }
    
    # Publica sem normalizar
    bus.publish("topic_raw", raw_data, normalize=False)
    time.sleep(0.2)
    
    # Deve manter precisão original
    assert received[0]["price"] == 123.456789