# event_bus.py - Sistema de eventos com normalização numérica

import time
import threading
from collections import deque
from typing import Dict, Any, Callable, Union, List
import logging
import hashlib
import re

# 🔹 IMPORTA UTILITÁRIOS DE FORMATAÇÃO
from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific,
    format_epoch_ms,
    format_ratio,
    format_integer,
    auto_format
)


class EventBus:
    def __init__(self, max_queue_size=1000, deduplication_window=30):
        """
        max_queue_size: Tamanho máximo da fila de eventos
        deduplication_window: Tempo em segundos para deduplicação (30s padrão)
        """
        self._handlers = {}
        self._queue = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()
        self._processing = False
        self._thread = None
        self._stop = False
        self._dedup_cache = {}
        self._dedup_window = deduplication_window
        self._logger = logging.getLogger("EventBus")
        
        # Iniciar thread de processamento
        self.start()

    def _parse_numeric_string(self, value: Any) -> Union[float, int, Any]:
        """
        Converte string numérica formatada para número puro.
        Remove vírgulas, %, K/M/B, etc.
        """
        if value is None or value == '':
            return None
            
        if isinstance(value, (int, float)):
            return value
            
        if not isinstance(value, str):
            return value
            
        try:
            # Remove espaços
            cleaned = value.strip()
            
            # Se for string vazia ou N/A
            if not cleaned or cleaned.lower() in ['n/a', 'none', 'null', '-']:
                return None
            
            # Remove símbolo de moeda
            cleaned = cleaned.replace('$', '').replace('R$', '')
            
            # Detecta e processa notação K/M/B
            multiplier = 1
            if cleaned.upper().endswith('K'):
                multiplier = 1_000
                cleaned = cleaned[:-1]
            elif cleaned.upper().endswith('M'):
                multiplier = 1_000_000
                cleaned = cleaned[:-1]
            elif cleaned.upper().endswith('B'):
                multiplier = 1_000_000_000
                cleaned = cleaned[:-1]
            
            # Remove %
            is_percent = cleaned.endswith('%')
            if is_percent:
                cleaned = cleaned[:-1]
            
            # Remove vírgulas (separador de milhar)
            cleaned = cleaned.replace(',', '')
            
            # Converte para número
            num = float(cleaned)
            
            # Aplica multiplicador
            num *= multiplier
            
            # Se era percentual, pode precisar dividir por 100
            # (depende do contexto, mas aqui mantemos como está)
            
            # Retorna int se for número inteiro
            if num == int(num):
                return int(num)
            return num
            
        except (ValueError, TypeError):
            # Se falhar, retorna valor original
            return value

    def _normalize_value(self, key: str, value: Any) -> Union[float, int, None]:
        """
        Normaliza um valor baseado no tipo de campo.
        Aplica regras do format_utils com for_json=True.
        """
        # Primeiro tenta converter string para número
        value = self._parse_numeric_string(value)
        
        if value is None:
            return None
            
        try:
            # Usa auto_format para determinar o tipo e formatar
            normalized = auto_format(key, value, for_json=True)
            return normalized
        except:
            return value

    def _normalize_event_data(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza todos os valores numéricos do evento.
        Aplica precisão correta para cada tipo de campo.
        """
        normalized = {}
        
        # Campos que são listas de preços
        price_list_fields = {'hvns', 'lvns', 'single_prints', 'levels', 'prices'}
        
        for key, value in event.items():
            if value is None:
                normalized[key] = None
                continue
                
            # Normaliza listas
            if isinstance(value, list):
                if key in price_list_fields or any(x in key.lower() for x in ['price', 'level']):
                    # Lista de preços: normaliza cada item
                    normalized_list = []
                    for item in value:
                        norm_item = self._normalize_value('price', item)
                        if norm_item is not None:
                            normalized_list.append(norm_item)
                    normalized[key] = normalized_list
                else:
                    # Lista genérica: processa recursivamente se contiver dicts
                    normalized_list = []
                    for item in value:
                        if isinstance(item, dict):
                            normalized_list.append(self._normalize_event_data(item))
                        else:
                            normalized_list.append(item)
                    normalized[key] = normalized_list
                    
            # Normaliza dicionários aninhados
            elif isinstance(value, dict):
                normalized[key] = self._normalize_event_data(value)
                
            # Normaliza valores individuais
            else:
                normalized[key] = self._normalize_value(key, value)
                
        return normalized

    def _generate_event_id(self, event: Dict) -> str:
        """
        Gera ID único para deduplicação.
        Usa valores normalizados para garantir consistência.
        """
        # Normaliza valores antes de gerar o hash
        norm_timestamp = self._normalize_value('timestamp', event.get('timestamp', ''))
        norm_delta = self._normalize_value('delta', event.get('delta', ''))
        norm_volume = self._normalize_value('volume_total', event.get('volume_total', ''))
        norm_price = self._normalize_value('preco_fechamento', event.get('preco_fechamento', ''))
        
        # Formata valores com precisão consistente
        timestamp_str = str(norm_timestamp) if norm_timestamp else ''
        delta_str = f"{norm_delta:.2f}" if norm_delta is not None else ''
        volume_str = str(int(norm_volume)) if norm_volume is not None else ''
        price_str = f"{norm_price:.4f}" if norm_price is not None else ''
        
        # Gera chave única
        key = f"{timestamp_str}|{delta_str}|{volume_str}|{price_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def _is_duplicate(self, event: Dict) -> bool:
        """Verifica se evento é duplicado usando valores normalizados"""
        event_id = self._generate_event_id(event)
        current_time = time.time()
        
        # Limpar cache antigo
        expired_keys = [k for k, t in self._dedup_cache.items() if current_time - t > self._dedup_window]
        for k in expired_keys:
            del self._dedup_cache[k]
        
        # Verificar duplicado
        if event_id in self._dedup_cache:
            return True
            
        # Registrar novo evento
        self._dedup_cache[event_id] = current_time
        return False

    def subscribe(self, event_type: str, handler: Callable):
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

    def publish(self, event_type: str, event_data: Dict, normalize: bool = True):
        """
        Publica um evento no barramento.
        
        Args:
            event_type: Tipo do evento
            event_data: Dados do evento
            normalize: Se True, normaliza valores numéricos antes de publicar
        """
        with self._lock:
            # Normaliza dados se solicitado
            if normalize:
                try:
                    event_data = self._normalize_event_data(event_data)
                except Exception as e:
                    self._logger.warning(f"Erro ao normalizar evento: {e}")
            
            # Ignorar eventos duplicados
            if self._is_duplicate(event_data):
                self._logger.debug(f"Evento duplicado ignorado: {event_type}")
                return
                
            # Adicionar à fila
            self._queue.append((event_type, event_data))
            
            # Log de debug
            self._logger.debug(f"Evento publicado: {event_type}")

    def _process_queue(self):
        while not self._stop:
            try:
                if self._queue:
                    with self._lock:
                        event_type, event_data = self._queue.popleft()
                    
                    # Processar evento
                    self._dispatch(event_type, event_data)
                else:
                    time.sleep(0.01)  # Pequena pausa para reduzir uso de CPU
            except Exception as e:
                self._logger.error(f"Erro no processamento de eventos: {e}")

    def _dispatch(self, event_type: str, event_data: Dict):
        """Envia evento para todos os handlers registrados"""
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    self._logger.error(f"Erro no handler para {event_type}: {e}")
        else:
            self._logger.debug(f"Nenhum handler para {event_type}")

    def start(self):
        if not self._thread or not self._thread.is_alive():
            self._stop = False
            self._thread = threading.Thread(target=self._process_queue, daemon=True)
            self._thread.start()
            self._logger.info("EventBus iniciado")

    def shutdown(self):
        self._stop = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._logger.info("EventBus desligado")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do EventBus"""
        with self._lock:
            stats = {
                "queue_size": len(self._queue),
                "handlers_count": sum(len(h) for h in self._handlers.values()),
                "event_types": list(self._handlers.keys()),
                "dedup_cache_size": len(self._dedup_cache),
                "is_running": self._thread.is_alive() if self._thread else False
            }
        return stats


# Exemplo de uso com normalização
if __name__ == "__main__":
    # Teste de normalização
    bus = EventBus()
    
    # Evento com valores formatados (strings)
    test_event = {
        "timestamp": "1,759,761,480,000.00",  # epoch com vírgulas
        "preco_fechamento": "123,456.789",     # preço com vírgulas
        "delta": "+1,234.56",                  # delta com sinal e vírgulas
        "volume_total": "1.5M",                # volume com notação M
        "imbalance_ratio": "60.46%",          # percentual
        "buy_sell_ratio": "0.41",              # ratio (sem %)
        "num_trades": "1343.0",                # inteiro com .0
        "duration_s": "13.67",                 # segundos
        "hvns": ["123,172.00", "124,500.50", "125,000"],  # lista de preços
        "nested": {
            "poc": "$126,789.12",              # preço com $
            "volatility_5": "0.00045"          # volatilidade
        }
    }
    
    # Normaliza
    normalized = bus._normalize_event_data(test_event)
    
    print("Original:")
    print(test_event)
    print("\nNormalizado:")
    print(normalized)
    
    # Testa deduplicação
    event_id_1 = bus._generate_event_id(test_event)
    
    # Mesmo evento mas com formatação diferente
    test_event_2 = {
        "timestamp": "1759761480000",         # sem vírgulas
        "preco_fechamento": "123456.789",      # sem vírgulas
        "delta": "1234.56",                    # sem sinal
        "volume_total": "1500000",             # expandido
    }
    
    event_id_2 = bus._generate_event_id(test_event_2)
    
    print(f"\nHash evento 1: {event_id_1}")
    print(f"Hash evento 2: {event_id_2}")
    print(f"São iguais após normalização? {event_id_1 == event_id_2}")