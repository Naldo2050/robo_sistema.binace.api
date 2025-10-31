# event_bus.py v2.1.2 - Sistema de eventos com normalização numérica e timestamp ISO 8601

import time
import threading
from collections import deque
from typing import Dict, Any, Callable, Union, List, Optional
from datetime import datetime, timezone
import logging
import hashlib
import re
from decimal import Decimal

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
    """
    Sistema de eventos com normalização numérica inteligente.
    
    🔹 CORREÇÕES v2.1.2:
      ✅ Validação inteligente (não tenta converter BEARISH/BULLISH/NEUTRAL)
      ✅ Suporte completo para timestamps ISO 8601
      ✅ Preserva precisão máxima para campos críticos (timestamps, volumes BTC)
      ✅ Não converte volumes BTC para int (preserva 8 casas decimais)
      ✅ Timestamps nunca são normalizados (precisão de milissegundos)
      ✅ Deduplicação usa precisão correta para cada tipo de campo
      ✅ Logs reduzidos (apenas DEBUG para strings não-timestamp)
    """

    # 🆕 Campos que NÃO devem ser normalizados (precisão crítica)
    CRITICAL_PRECISION_FIELDS = {
        'timestamp', 'epoch_ms', 'timestamp_utc', 'timestamp_ny', 'timestamp_sp',
        'first_seen_ms', 'last_seen_ms', 'recent_timestamp', 'recent_ts_ms',
        'age_ms', 'duration_ms', 'last_update_ms', 'cluster_duration_ms',
        'T', 'E', 'time', 'created_at',  # Campos de timestamp da Binance
    }
    
    # 🆕 Campos que precisam de 8 casas decimais (volumes BTC)
    BTC_PRECISION_FIELDS = {
        'volume', 'qty', 'q', 'delta', 'delta_btc', 'cvd',
        'buy_volume_btc', 'sell_volume_btc', 'total_volume_btc',
        'whale_buy_volume', 'whale_sell_volume', 'whale_delta',
        'volume_compra', 'volume_venda', 'volume_total',
    }
    
    # 🆕 Campos que são preços (2-4 casas decimais dependendo do par)
    PRICE_FIELDS = {
        'price', 'p', 'preco', 'preco_abertura', 'preco_fechamento',
        'preco_maximo', 'preco_minimo', 'poc', 'vah', 'val',
        'close', 'open', 'high', 'low', 'center',
    }

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
        
        # 🆕 Contadores de validação
        self._normalization_warnings = 0
        self._precision_loss_warnings = 0
        self._total_events_normalized = 0
        
        # Iniciar thread de processamento
        self.start()

    def _normalize_timestamp(self, timestamp: Any) -> Optional[int]:
        """
        Normaliza timestamp de diferentes formatos para int (ms).
        
        🆕 v2.1.2 - Validação inteligente (ignora strings não-timestamp)
        
        Aceita:
        - int/float (ms ou s)
        - datetime object
        - string ISO 8601 (com ou sem timezone, com ou sem 'Z')
        - None
        
        Retorna int (ms) ou None
        """
        if timestamp is None:
            return None
            
        # Já é int/float
        if isinstance(timestamp, (int, float)):
            ts = int(timestamp)
            # Se está em segundos (< ano 3000), converte para ms
            if ts < 32503680000:  # 01/01/3000 em segundos
                return ts * 1000
            return ts
            
        # É datetime
        if isinstance(timestamp, datetime):
            return int(timestamp.timestamp() * 1000)
            
        # É string
        if isinstance(timestamp, str):
            # 🆕 VALIDAÇÃO: Ignora strings que claramente NÃO são timestamps
            cleaned = timestamp.strip().upper()
            
            # Lista de valores conhecidos que NÃO são timestamps
            non_timestamp_values = {
                'BEARISH', 'BULLISH', 'NEUTRAL',
                'BUY', 'SELL', 'HOLD',
                'LONG', 'SHORT',
                'UP', 'DOWN', 'SIDEWAYS',
                'POSITIVE', 'NEGATIVE',
                'HIGH', 'MEDIUM', 'LOW',
                'TRUE', 'FALSE',
                'N/A', 'NONE', 'NULL', 'UNKNOWN',
            }
            
            if cleaned in non_timestamp_values:
                return None  # ✅ Retorna None silenciosamente (sem warning)
            
            # 🆕 Validação adicional: timestamp deve ter formato mínimo
            # Aceita formatos como: "2025-10-31", "1730394063", "1730394063000"
            if len(cleaned) < 10:  # Timestamp mínimo tem 10 chars
                return None
            
            try:
                # Remove 'Z' final e garante formato compatível
                if cleaned.endswith('Z'):
                    cleaned = cleaned[:-1] + '+00:00'
                
                # Tenta parsing direto (Python 3.7+)
                try:
                    # Tenta com timezone
                    dt = datetime.fromisoformat(cleaned)
                except ValueError:
                    # Tenta sem timezone (assume UTC)
                    dt = datetime.fromisoformat(cleaned.replace('+00:00', '')).replace(tzinfo=timezone.utc)
                
                return int(dt.timestamp() * 1000)
                
            except (ValueError, AttributeError):
                # Se falhar, tenta converter diretamente para int (epoch)
                try:
                    ts = int(float(cleaned))
                    if ts < 32503680000:  # Converter segundos para ms
                        return ts * 1000
                    return ts
                except (ValueError, TypeError):
                    # ✅ Apenas loga se parecer um timestamp malformado
                    if any(char in cleaned for char in ['-', ':', 'T', '/']):
                        self._logger.debug(f"⚠️ Timestamp malformado '{timestamp}'")
                    return None
                    
        return None

    def _parse_numeric_string(self, value: Any) -> Union[float, int, Any]:
        """
        Converte string numérica formatada para número puro.
        Remove vírgulas, %, K/M/B, etc.
        
        🆕 CORREÇÃO: Preserva precisão máxima, nunca converte para int
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
            
            # Remove sinal de + no início (mas preserva -)
            if cleaned.startswith('+'):
                cleaned = cleaned[1:]
            
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
            
            # Converte para número usando float para máxima precisão
            num = float(cleaned)
            
            # Aplica multiplicador
            num *= multiplier
            
            # 🆕 CORREÇÃO: NUNCA converte para int automaticamente
            # Preserva precisão máxima como float
            return num
            
        except (ValueError, TypeError):
            # Se falhar, retorna valor original
            return value

    def _get_field_precision(self, key: str) -> Optional[int]:
        """
        🆕 Determina a precisão necessária para um campo.
        
        Returns:
            Número de casas decimais ou None para não normalizar
        """
        key_lower = key.lower()
        
        # Timestamps: NUNCA normalizar (preservar milissegundos)
        if key in self.CRITICAL_PRECISION_FIELDS or 'timestamp' in key_lower or 'epoch' in key_lower:
            return None  # Não normalizar
        
        # Volumes BTC: 8 casas decimais
        if key in self.BTC_PRECISION_FIELDS or 'volume' in key_lower or 'qty' in key_lower or 'delta' in key_lower:
            return 8
        
        # Preços: 4 casas decimais (suficiente para BTC/USDT)
        if key in self.PRICE_FIELDS or 'price' in key_lower or 'preco' in key_lower:
            return 4
        
        # Ratios/Percentuais: 4 casas decimais
        if 'ratio' in key_lower or 'pct' in key_lower or 'percent' in key_lower or 'imbalance' in key_lower:
            return 4
        
        # Contadores: 0 casas decimais (int)
        if 'count' in key_lower or 'num_' in key_lower or key.endswith('_n'):
            return 0
        
        # Padrão: 8 casas decimais
        return 8

    def _normalize_value(self, key: str, value: Any, validate: bool = True) -> Union[float, int, None, str]:
        """
        Normaliza um valor baseado no tipo de campo.
        
        🆕 CORREÇÕES v2.1.2:
          - Validação inteligente (não tenta converter strings conhecidas)
          - Preserva strings de viés/tendência (BEARISH, BULLISH, etc)
          - Suporta timestamps ISO 8601
        
        Args:
            key: Nome do campo
            value: Valor a normalizar
            validate: Se True, valida perda de precisão
        """
        # ✅ Se for None, retorna None
        if value is None:
            return None
        
        # ✅ NOVO: Preserva strings conhecidas que não são números
        if isinstance(value, str):
            cleaned = value.strip().upper()
            
            # Lista expandida de valores string a preservar
            preserve_strings = {
                # Viés/Direção
                'BEARISH', 'BULLISH', 'NEUTRAL',
                'BUY', 'SELL', 'HOLD',
                'LONG', 'SHORT',
                'UP', 'DOWN', 'SIDEWAYS',
                'POSITIVE', 'NEGATIVE',
                # Níveis
                'HIGH', 'MEDIUM', 'LOW',
                'STRONG', 'WEAK', 'MODERATE',
                # Booleanos/Status
                'TRUE', 'FALSE', 'YES', 'NO',
                'ACTIVE', 'INACTIVE', 'PENDING',
                'OK', 'ERROR', 'WARNING',
                # Valores vazios
                'N/A', 'NONE', 'NULL', 'UNKNOWN', '-',
            }
            
            if cleaned in preserve_strings:
                return value  # ✅ Retorna string original sem normalizar
        
        # 🆕 Tenta normalizar timestamp se for campo de tempo
        if key in self.CRITICAL_PRECISION_FIELDS or 'timestamp' in key.lower() or 'time' in key.lower():
            normalized_ts = self._normalize_timestamp(value)
            if normalized_ts is not None:
                return normalized_ts
            # Se retornou None, continua tentando outras normalizações
        
        # Tenta converter string numérica para número
        parsed = self._parse_numeric_string(value)
        
        if parsed is None:
            return None
        
        # Se não for numérico, retorna original (preserva strings)
        if not isinstance(parsed, (int, float)):
            return parsed
        
        try:
            # Determina precisão necessária
            precision = self._get_field_precision(key)
            
            # Se não deve normalizar (timestamps, etc)
            if precision is None:
                # 🆕 Garante que é int se for timestamp (epoch em ms)
                if isinstance(parsed, float) and parsed == int(parsed):
                    return int(parsed)
                return parsed
            
            # Normaliza com precisão correta
            if precision == 0:
                # Contador/inteiro
                normalized = int(round(parsed))
            else:
                # Float com precisão específica
                normalized = round(parsed, precision)
            
            # 🆕 VALIDAÇÃO: Detecta perda de precisão significativa
            if validate and isinstance(parsed, float):
                original = parsed
                diff = abs(original - normalized)
                
                # Define tolerância baseada na precisão
                if precision >= 8:
                    tolerance = 1e-8
                elif precision >= 4:
                    tolerance = 1e-4
                elif precision >= 2:
                    tolerance = 1e-2
                else:
                    tolerance = 1.0
                
                if diff > tolerance:
                    self._precision_loss_warnings += 1
                    self._logger.debug(
                        f"⚠️ PERDA DE PRECISÃO: {key}={original} → {normalized} "
                        f"(diff={diff:.10f})"
                    )
            
            return normalized
            
        except Exception as e:
            self._logger.debug(f"Erro ao normalizar {key}={value}: {e}")
            return parsed

    def _normalize_event_data(self, event: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
        """
        Normaliza todos os valores numéricos do evento.
        
        🆕 CORREÇÕES:
          - Preserva precisão crítica
          - Valida alterações
          - Loga warnings em casos suspeitos
        
        Args:
            event: Evento a normalizar
            validate: Se True, valida alterações
        """
        normalized = {}
        
        # Campos que são listas de preços
        price_list_fields = {'hvns', 'lvns', 'single_prints', 'levels', 'prices', 'supports', 'resistances'}
        
        for key, value in event.items():
            if value is None:
                normalized[key] = None
                continue
            
            # Normaliza listas
            if isinstance(value, list):
                if key in price_list_fields or any(x in key.lower() for x in ['price', 'level']):
                    # Lista de preços: normaliza cada item com precisão de preço
                    normalized_list = []
                    for item in value:
                        norm_item = self._normalize_value('price', item, validate=validate)
                        if norm_item is not None:
                            normalized_list.append(norm_item)
                    normalized[key] = normalized_list
                else:
                    # Lista genérica: processa recursivamente se contiver dicts
                    normalized_list = []
                    for item in value:
                        if isinstance(item, dict):
                            normalized_list.append(self._normalize_event_data(item, validate=validate))
                        else:
                            # Tenta normalizar baseado na chave da lista
                            norm_item = self._normalize_value(key, item, validate=validate)
                            normalized_list.append(norm_item)
                    normalized[key] = normalized_list
                    
            # Normaliza dicionários aninhados
            elif isinstance(value, dict):
                normalized[key] = self._normalize_event_data(value, validate=validate)
                
            # Normaliza valores individuais
            else:
                normalized[key] = self._normalize_value(key, value, validate=validate)
        
        if validate:
            self._total_events_normalized += 1
        
        return normalized

    def _generate_event_id(self, event: Dict) -> str:
        """
        Gera ID único para deduplicação.
        
        🆕 CORREÇÃO v2.1.2: Usa normalização de timestamp robusta
        """
        # Extrai campos chave
        timestamp = event.get('timestamp') or event.get('time') or event.get('created_at')
        delta = event.get('delta')
        volume = event.get('volume_total') or event.get('volume')
        price = event.get('preco_fechamento') or event.get('price')
        
        # Normaliza valores (SEM validação para performance)
        norm_timestamp = self._normalize_timestamp(timestamp)
        norm_delta = self._normalize_value('delta', delta, validate=False)
        norm_volume = self._normalize_value('volume_total', volume, validate=False)
        norm_price = self._normalize_value('preco_fechamento', price, validate=False)
        
        # 🆕 CORREÇÃO: Usa precisão correta para cada campo
        timestamp_str = str(norm_timestamp) if norm_timestamp else ''
        delta_str = f"{norm_delta:.8f}" if norm_delta is not None else ''
        volume_str = f"{norm_volume:.8f}" if norm_volume is not None else ''
        price_str = f"{norm_price:.4f}" if norm_price is not None else ''
        
        # Gera chave única
        key = f"{timestamp_str}|{delta_str}|{volume_str}|{price_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def _is_duplicate(self, event: Dict) -> bool:
        """Verifica se evento é duplicado usando valores normalizados"""
        try:
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
            
        except Exception as e:
            self._logger.debug(f"⚠️ Erro ao verificar duplicação: {e}")
            return False  # Em caso de erro, permite o evento

    def subscribe(self, event_type: str, handler: Callable):
        """Registra handler para um tipo de evento."""
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
            self._logger.debug(f"Handler registrado para {event_type}")

    def publish(self, event_type: str, event_data: Dict, normalize: bool = True, validate: bool = True):
        """
        Publica um evento no barramento.
        
        🆕 CORREÇÕES:
          - Opção de desabilitar normalização
          - Opção de desabilitar validação (para performance)
        
        Args:
            event_type: Tipo do evento
            event_data: Dados do evento
            normalize: Se True, normaliza valores numéricos antes de publicar
            validate: Se True, valida alterações causadas por normalização
        """
        with self._lock:
            # Normaliza dados se solicitado
            if normalize:
                try:
                    event_data = self._normalize_event_data(event_data, validate=validate)
                except Exception as e:
                    self._normalization_warnings += 1
                    self._logger.debug(f"⚠️ Erro ao normalizar evento {event_type}: {e}")
            
            # Ignorar eventos duplicados
            if self._is_duplicate(event_data):
                self._logger.debug(f"Evento duplicado ignorado: {event_type}")
                return
                
            # Adicionar à fila
            self._queue.append((event_type, event_data))
            
            # Log de debug
            self._logger.debug(f"📢 Evento publicado: {event_type}")

    def _process_queue(self):
        """Thread de processamento de eventos."""
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
                self._logger.error(f"❌ Erro no processamento de eventos: {e}", exc_info=True)

    def _dispatch(self, event_type: str, event_data: Dict):
        """Envia evento para todos os handlers registrados"""
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    self._logger.error(f"❌ Erro no handler para {event_type}: {e}", exc_info=True)
        else:
            self._logger.debug(f"⚠️ Nenhum handler para {event_type}")

    def start(self):
        """Inicia thread de processamento."""
        if not self._thread or not self._thread.is_alive():
            self._stop = False
            self._thread = threading.Thread(target=self._process_queue, daemon=True)
            self._thread.start()
            self._logger.info("✅ EventBus v2.1.2 iniciado (validação inteligente)")

    def shutdown(self):
        """Para thread de processamento."""
        self._stop = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._logger.info("🔄 EventBus desligado")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do EventBus"""
        with self._lock:
            stats = {
                "queue_size": len(self._queue),
                "handlers_count": sum(len(h) for h in self._handlers.values()),
                "event_types": list(self._handlers.keys()),
                "dedup_cache_size": len(self._dedup_cache),
                "is_running": self._thread.is_alive() if self._thread else False,
                "total_events_normalized": self._total_events_normalized,
                "normalization_warnings": self._normalization_warnings,
                "precision_loss_warnings": self._precision_loss_warnings,
            }
        return stats