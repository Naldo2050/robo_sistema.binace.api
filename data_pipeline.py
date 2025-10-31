# data_pipeline.py v2.3.0 - CORRIGIDO COM VALIDAÇÃO TOLERANTE
# -*- coding: utf-8 -*-
"""
Pipeline de Dados Ultra-Otimizado v2.3.0

🔹 CORREÇÕES v2.3.0:
  ✅ Validação pré-pipeline tolerante (mínimo 3 trades)
  ✅ Configurações do config.py integradas
  ✅ Fallback inteligente para dados insuficientes
  ✅ Tratamento robusto de erros
  ✅ Logging detalhado para debug
"""

import pandas as pd
import numpy as np
import logging
import gzip
import json
import requests
import hashlib
import time
from collections import deque, OrderedDict
from typing import List, Dict, Any, Optional, Set, Tuple, Deque
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from time_manager import TimeManager

# ✅ NOVO: Importar configurações
try:
    import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    logging.warning("⚠️ config.py não encontrado - usando valores padrão")

# Compressão alternativa opcional
try:
    import brotli  # type: ignore
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    logging.info("Brotli não disponível, usando gzip")

# Importador opcional do gerador de features de ML
try:
    from ml_features import generate_ml_features
except Exception:
    generate_ml_features = None


class CacheManager:
    """Gerenciador de cache com TTL para dados que mudam pouco."""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.checksums: Dict[str, str] = {}
        
    def get(self, key: str, ttl_seconds: int = 3600) -> Optional[Any]:
        """Obtém valor do cache se ainda válido."""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        age = time.time() - entry['timestamp']
        
        if age > ttl_seconds:
            del self.cache[key]
            return None
            
        return entry['value']
    
    def set(self, key: str, value: Any) -> str:
        """Armazena valor no cache e retorna checksum."""
        json_str = json.dumps(value, sort_keys=True)
        checksum = hashlib.md5(json_str.encode()).hexdigest()[:8]
        
        self.cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'checksum': checksum
        }
        self.checksums[key] = checksum
        
        return checksum
    
    def has_changed(self, key: str, value: Any) -> bool:
        """Verifica se o valor mudou comparando checksum."""
        json_str = json.dumps(value, sort_keys=True)
        new_checksum = hashlib.md5(json_str.encode()).hexdigest()[:8]
        
        old_checksum = self.checksums.get(key)
        if old_checksum is None or old_checksum != new_checksum:
            self.checksums[key] = new_checksum
            return True
            
        return False
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do cache."""
        total_items = len(self.cache)
        total_size = sum(len(json.dumps(v['value'])) for v in self.cache.values())
        
        return {
            'items': total_items,
            'size_bytes': total_size,
            'checksums': len(self.checksums)
        }
    
    def cleanup(self, max_age_seconds: int = 7200):
        """Remove entradas antigas do cache."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            if current_time - entry['timestamp'] > max_age_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
            self.checksums.pop(key, None)
        
        if keys_to_remove:
            logging.debug(f"🧹 Cache cleanup: removidas {len(keys_to_remove)} entradas")


class EventBuffer:
    """Buffer circular inteligente para acumular eventos antes de enviar."""
    
    def __init__(self, max_size: int = 100, max_age_seconds: int = 60, min_events: int = 20):
        self.buffer: Deque[Dict] = deque(maxlen=max_size)
        self.event_checksums: Set[str] = set()
        self.max_age_seconds = max_age_seconds
        self.min_events = min_events
        self.first_event_time: Optional[float] = None
        self.stats = {
            'total_received': 0,
            'duplicates_filtered': 0,
            'batches_sent': 0
        }
    
    def add(self, event: Dict) -> bool:
        """
        Adiciona evento ao buffer se não for duplicado.
        Retorna True se adicionado, False se duplicado.
        """
        # Calcular checksum do evento
        event_str = json.dumps(event, sort_keys=True)
        checksum = hashlib.md5(event_str.encode()).hexdigest()[:16]
        
        self.stats['total_received'] += 1
        
        # Verificar duplicata
        if checksum in self.event_checksums:
            self.stats['duplicates_filtered'] += 1
            logging.debug(f"🔁 Evento duplicado filtrado (checksum: {checksum})")
            return False
        
        # Adicionar ao buffer
        self.buffer.append({
            'data': event,
            'checksum': checksum,
            'timestamp': time.time()
        })
        self.event_checksums.add(checksum)
        
        # Marcar tempo do primeiro evento
        if self.first_event_time is None:
            self.first_event_time = time.time()
        
        # Limpar checksums antigos se buffer estiver cheio
        if len(self.event_checksums) > self.buffer.maxlen * 2:
            self._cleanup_checksums()
        
        return True
    
    def should_flush(self, force: bool = False) -> bool:
        """Determina se o buffer deve ser enviado."""
        if force and self.buffer:
            return True
            
        if not self.buffer:
            return False
        
        # Enviar se buffer 80% cheio
        if len(self.buffer) >= self.buffer.maxlen * 0.8:
            logging.debug("📦 Buffer 80% cheio, flush necessário")
            return True
        
        # Enviar se tiver eventos suficientes E tempo suficiente
        if len(self.buffer) >= self.min_events:
            if self.first_event_time:
                age = time.time() - self.first_event_time
                if age > self.max_age_seconds:
                    logging.debug(f"⏰ Buffer com {age:.1f}s de idade, flush necessário")
                    return True
        
        return False
    
    def get_events(self, clear: bool = True) -> List[Dict]:
        """Obtém eventos do buffer."""
        events = [item['data'] for item in self.buffer]
        
        if clear:
            self.buffer.clear()
            self.first_event_time = None
            self.stats['batches_sent'] += 1
        
        return events
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do buffer."""
        return {
            **self.stats,
            'current_size': len(self.buffer),
            'buffer_age': time.time() - self.first_event_time if self.first_event_time else 0,
            'dedup_rate': (self.stats['duplicates_filtered'] / max(self.stats['total_received'], 1)) * 100
        }
    
    def _cleanup_checksums(self):
        """Remove checksums antigos para economizar memória."""
        current_checksums = {item['checksum'] for item in self.buffer}
        self.event_checksums = current_checksums
        logging.debug(f"🧹 Limpeza de checksums: mantidos {len(self.event_checksums)}")


class DataPipeline:
    def __init__(self, raw_trades: List[Dict], symbol: str, time_manager: Optional[TimeManager] = None):
        """
        Pipeline de dados ultra-otimizado com validação tolerante.
        
        ✅ CORRIGIDO v2.3.0:
        - Validação pré-pipeline configurável
        - Mínimo absoluto de 3 trades
        - Fallback inteligente
        """
        self.raw_trades = raw_trades
        self.symbol = symbol
        self.df: pd.DataFrame | None = None
        self.enriched_data: Dict[str, Any] | None = None
        self.contextual_data: Dict[str, Any] | None = None
        self.signal_data: List[Dict[str, Any]] | None = None
        self._cache: Dict[str, Any] = {}
        
        # Time manager
        self.tm: TimeManager = time_manager or TimeManager()
        
        # ✅ NOVO: Configurações do config.py
        self.min_trades_pipeline = getattr(config, 'MIN_TRADES_FOR_PIPELINE', 10) if HAS_CONFIG else 10
        self.min_absolute_trades = getattr(config, 'PIPELINE_MIN_ABSOLUTE_TRADES', 3) if HAS_CONFIG else 3
        self.allow_limited_data = getattr(config, 'PIPELINE_ALLOW_LIMITED_DATA', True) if HAS_CONFIG else True
        
        # Cache manager para dados que mudam pouco
        self.cache_manager = CacheManager()
        
        # Buffer de eventos
        self.event_buffer = EventBuffer(max_size=100, max_age_seconds=60, min_events=20)
        
        # Session HTTP persistente
        self._session: requests.Session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'DataPipeline/2.3.0',
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=120, max=1000'
        })
        
        # Cache de último payload
        self._last_payload_hash: Optional[str] = None
        self._last_payload_data: Optional[Dict] = None
        self._last_vp_hash: Optional[str] = None
        self._last_mtf_hash: Optional[str] = None
        
        # Thresholds de mudança mínima
        self._min_change_threshold = {
            'price_pct': 0.03,
            'cvd_abs': 0.2,
            'volume_pct': 0.1
        }
        
        # TTL para cache (em segundos)
        self.cache_ttl = {
            'vp_daily': 3600,
            'vp_weekly': 21600,
            'vp_monthly': 86400,
            'multi_tf': 300,
            'derivatives': 60,
            'market_context': 900
        }
        
        # Backoff para rate limiting
        self._backoff_seconds = 1
        self._max_backoff = 60
        self._last_request_time = 0
        
        # Mapeamentos de símbolos
        self.symbol_ids = {
            'BTCUSDT': 'BU',
            'ETHUSDT': 'EU',
            'BTCEUR': 'BE',
            'ETHEUR': 'EE',
            'BTCTUSD': 'BT',
            'SOLUSDT': 'SU',
            'BNBUSDT': 'BN',
            'XRPUSDT': 'XU',
            'ADAUSDT': 'AU',
            'DOGEUSDT': 'DU'
        }
        
        # Códigos de eventos
        self.event_codes = {
            'Absorção': 1,
            'Absorção de Venda': 2,
            'Absorção de Compra': 3,
            'Exaustão': 4,
            'ANALYSIS_TRIGGER': 5,
            'OrderBook': 6,
            'Alerta': 7,
            'VOLATILITY_SQUEEZE': 8
        }
        
        self.absorption_codes = {
            'Absorção de Venda': 1,
            'Absorção de Compra': 2,
            'Neutro': 0,
            '': 0
        }
        
        # Campos com zero significativo
        self.zero_significant_fields = {
            'delta', 'd', 'cvd', 'imbalance', 'i', 'trend', 't',
            'whale_delta', 'wd', 'pressure', 'p', 'momentum', 'm',
            'flow_imbalance', 'fi'
        }
        
        # ESCALA DE PREÇO DINÂMICA POR SÍMBOLO
        self.price_scales = {
            'BTCUSDT': 10,
            'ETHUSDT': 100,
            'BNBUSDT': 100,
            'SOLUSDT': 1000,
            'XRPUSDT': 10000,
            'DOGEUSDT': 100000,
            'ADAUSDT': 10000,
            'DEFAULT': 10
        }
        
        # Obter escala para o símbolo atual
        self.ohlc_scale = self.price_scales.get(self.symbol, self.price_scales['DEFAULT'])
        logging.debug(f"📏 Usando escala de preço {self.ohlc_scale} para {self.symbol}")
        
        # CONFIGURAÇÃO DE PRECISÃO
        self.precision_config = {
            'price': self._get_price_precision(),
            'price_precise': self._get_price_precision(),
            'volume_btc': 2,
            'volume_usdt': 0,
            'delta': 1,
            'delta_pct': 2,
            'index': 3,
            'ratio': 1,
            'time_seconds': 0,
            'average': 3,
            'percentage': 1,
            'factor': 1,
            'sensitivity': 0
        }
        
        # Thresholds de inclusão
        self.inclusion_thresholds = {
            'tps_min': 10,
            'imbalance_min': 0.25,
            'pressure_min': 0.35,
            'whale_delta_min': 0.1,
            'cvd_min': 0.1,
            'buy_sell_ratio_extreme': (0.2, 5),
            'volume_ratio_extreme': (0.3, 3),
            'momentum_diff': 0.2,
            'bp_min': 0.1,
            'flow_imbalance_min': 0.3,
            'trade_intensity_min': 30,
            'volatility_min_bps': 2
        }
        
        # Estatísticas
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'events_sent': 0,
            'events_buffered': 0,
            'bytes_sent': 0,
            'bytes_saved': 0
        }
        
        # ✅ Validação e carregamento
        self._validate_and_load()
    
    def _get_price_precision(self) -> int:
        """Retorna precisão decimal baseada na escala do símbolo."""
        scale_to_precision = {
            10: 1,
            100: 2,
            1000: 3,
            10000: 4,
            100000: 5
        }
        return scale_to_precision.get(self.ohlc_scale, 1)

    # ===============================
    # Helpers internos
    # ===============================
    
    @staticmethod
    def _coerce_float(x) -> float | None:
        if x is None:
            return None
        try:
            if isinstance(x, (int, float)):
                return float(x)
            return float(str(x).strip())
        except Exception:
            return None

    @staticmethod
    def _coerce_int(x) -> int | None:
        if x is None:
            return None
        try:
            if isinstance(x, (int, float)):
                return int(x)
            return int(float(str(x).strip()))
        except Exception:
            return None

    # ===============================
    # ✅ VALIDAÇÃO MELHORADA v2.3.0
    # ===============================
    
    def _validate_and_load(self):
        """
        Validação PRÉ-PIPELINE com fallback inteligente.
        ✅ CORRIGIDO v2.3.0
        """
        # Validação inicial básica
        if not self.raw_trades:
            raise ValueError("Lista de trades vazia.")
        
        if not isinstance(self.raw_trades, list):
            raise ValueError("raw_trades deve ser uma lista.")
        
        try:
            # Validar e normalizar trades
            validated: List[Dict[str, Any]] = []
            
            for i, t in enumerate(self.raw_trades):
                if not isinstance(t, dict):
                    logging.warning(f"⚠️ Trade {i} não é dict, ignorando")
                    continue
                
                # Extrair campos
                p = self._coerce_float(t.get("p"))
                q = self._coerce_float(t.get("q"))
                T = self._coerce_int(t.get("T"))
                m = t.get("m", np.nan)
                
                # Validar campos obrigatórios
                if p is None or q is None or T is None:
                    logging.debug(f"⚠️ Trade {i} com campos ausentes: p={p}, q={q}, T={T}")
                    continue
                
                # Validar valores positivos
                if p <= 0 or q <= 0 or T <= 0:
                    logging.debug(f"⚠️ Trade {i} com valores inválidos: p={p}, q={q}, T={T}")
                    continue
                
                validated.append({"p": p, "q": q, "T": T, "m": m})
            
            # ✅ NOVO: Validação mais tolerante
            if not validated:
                raise ValueError(
                    f"Nenhum trade válido após validação. "
                    f"Total recebido: {len(self.raw_trades)}"
                )
            
            # Criar DataFrame
            df = pd.DataFrame(validated)
            
            # Remover NaN em campos críticos
            initial_len = len(df)
            df = df.dropna(subset=["p", "q", "T"])
            
            if len(df) < initial_len:
                logging.warning(
                    f"⚠️ {initial_len - len(df)} trades removidos por NaN"
                )
            
            # Ordenar por timestamp
            df = df.sort_values("T", kind="mergesort").reset_index(drop=True)
            
            # ✅ NOVO: Validação com limites configuráveis
            if df.empty:
                raise ValueError("DataFrame vazio após limpeza de NaN.")
            
            if len(df) < self.min_absolute_trades:
                raise ValueError(
                    f"Dados insuficientes para pipeline. "
                    f"Recebido: {len(df)} trades, mínimo absoluto: {self.min_absolute_trades}"
                )
            
            # ✅ NOVO: Aviso para dados limitados
            if len(df) < self.min_trades_pipeline:
                if self.allow_limited_data:
                    logging.warning(
                        f"⚠️ Pipeline com dados limitados: {len(df)} trades "
                        f"(recomendado: {self.min_trades_pipeline}). "
                        f"Processando com precisão reduzida..."
                    )
                else:
                    raise ValueError(
                        f"Dados insuficientes para pipeline. "
                        f"Recebido: {len(df)} trades, mínimo: {self.min_trades_pipeline}"
                    )
            
            # ✅ Validação de range de preços
            price_range = df["p"].max() - df["p"].min()
            avg_price = df["p"].mean()
            price_variance_pct = (price_range / avg_price * 100) if avg_price > 0 else 0
            
            if price_variance_pct > 10:
                logging.warning(
                    f"⚠️ Variação de preço muito alta: {price_variance_pct:.2f}% "
                    f"(pode indicar dados inconsistentes)"
                )
            
            self.df = df
            
            # ✅ Log de sucesso com detalhes
            logging.debug(
                f"✅ Pipeline validado: {len(df)} trades válidos | "
                f"Preço: ${df['p'].min():.2f} - ${df['p'].max():.2f} | "
                f"Volume total: {df['q'].sum():.4f}"
            )
            
        except ValueError as ve:
            # Re-raise erros de validação
            logging.error(f"❌ Erro de validação: {ve}")
            raise
            
        except Exception as e:
            # Erro genérico
            logging.error(f"❌ Erro ao carregar dados: {e}", exc_info=True)
            raise ValueError(f"Erro ao processar trades: {e}")

    def _get_cached(self, key: str, compute_fn):
        """Retorna valor do cache ou calcula e armazena."""
        if key in self._cache:
            self.stats['cache_hits'] += 1
            return self._cache[key]
        
        self.stats['cache_misses'] += 1
        result = compute_fn()
        self._cache[key] = result
        return result

    def _parse_iso_ms(self, iso_str: str) -> Optional[int]:
        """Converte ISO 8601 para epoch_ms."""
        try:
            dt = datetime.fromisoformat(iso_str)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    def _sanitize_event(self, ev: Dict[str, Any], default_ts_ms: Optional[int] = None) -> Dict[str, Any]:
        """Garante timestamp único (epoch_ms) no evento, remove redundâncias."""
        if not isinstance(ev, dict):
            return ev
        
        e: Dict[str, Any] = dict(ev)
        ts_ms: Optional[int] = None
        
        try:
            if "epoch_ms" in e and isinstance(e["epoch_ms"], (int, float)) and int(e["epoch_ms"]) > 0:
                ts_ms = int(e["epoch_ms"])
        except Exception:
            ts_ms = None
        
        if ts_ms is None and isinstance(e.get("timestamp_utc"), str) and e["timestamp_utc"]:
            ts_ms = self._parse_iso_ms(e["timestamp_utc"])
        
        if ts_ms is None:
            if default_ts_ms is None:
                try:
                    default_ts_ms = int(self.enriched_data.get("ohlc", {}).get("close_time", 0)) if self.enriched_data else None
                except Exception:
                    default_ts_ms = None
            ts_ms = default_ts_ms or self.tm.now_ms()
        
        # Adicionar APENAS epoch_ms
        e['epoch_ms'] = ts_ms
        
        # Remover timestamps redundantes
        for field in ['timestamp_utc', 'timestamp_ny', 'timestamp_sp', 'timestamp', 
                     'window_open_ms', 'window_close_ms', 'window_duration_ms',
                     'open_time_iso', 'close_time_iso']:
            e.pop(field, None)
        
        return e
    
    def _round_smart(self, value: float, value_type: str) -> float:
        """Arredonda valor com precisão MÍNIMA baseada no tipo."""
        if value is None or not isinstance(value, (int, float)):
            return value
        
        if np.isnan(value) or np.isinf(value):
            return 0
        
        precision = self.precision_config.get(value_type, 2)
        
        # Se precisão é 0, retorna inteiro
        if precision == 0:
            return float(int(round(value)))
        
        # Usar Decimal para evitar erros de float
        try:
            decimal_value = Decimal(str(value))
            rounded = decimal_value.quantize(
                Decimal(10) ** -precision,
                rounding=ROUND_HALF_UP
            )
            return float(rounded)
        except Exception:
            return round(value, precision)

    # ===============================
    # Camada 2 — Enriched (OTIMIZADO)
    # ===============================
    
    def enrich(self) -> Dict[str, Any]:
        """Adiciona OHLC, VWAP, volumes e métricas com precisão MÍNIMA."""
        try:
            from data_handler import (
                calcular_metricas_intra_candle,
                calcular_volume_profile,
                calcular_dwell_time,
                calcular_trade_speed,
            )
            
            df = self.df
            if df is None or df.empty:
                raise ValueError("DataFrame não carregado na pipeline.")
            
            # Preços com precisão mínima
            open_price = self._round_smart(float(df["p"].iloc[0]), 'price')
            close_price = self._round_smart(float(df["p"].iloc[-1]), 'price')
            high_price = self._round_smart(float(df["p"].max()), 'price')
            low_price = self._round_smart(float(df["p"].min()), 'price')
            
            open_time = int(df["T"].iloc[0])
            close_time = int(df["T"].iloc[-1])
            
            # Volumes otimizados
            base_volume = self._round_smart(float(df["q"].sum()), 'volume_btc')
            quote_volume = int(round((df["p"] * df["q"]).sum()))
            vwap = self._round_smart(quote_volume / base_volume if base_volume > 0 else close_price, 'price')
            
            enriched = {
                "symbol": self.symbol,
                "ohlc": {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "open_time": open_time,
                    "close_time": close_time,
                    "vwap": vwap,
                },
                "volume_total": base_volume,
                "volume_total_usdt": quote_volume,
                "num_trades": int(len(df)),
            }
            
            # Métricas com precisão MÍNIMA
            metricas = self._get_cached("metricas_intra", lambda: calcular_metricas_intra_candle(df))
            for key, value in metricas.items():
                if 'delta' in key or 'reversao' in key:
                    enriched[key] = self._round_smart(value, 'delta')
                else:
                    enriched[key] = value
            
            # Volume Profile otimizado
            vp = self._get_cached("volume_profile", lambda: calcular_volume_profile(df))
            enriched['poc_price'] = self._round_smart(vp.get('poc_price', 0), 'price')
            enriched['poc_volume'] = self._round_smart(vp.get('poc_volume', 0), 'volume_btc')
            enriched['poc_percentage'] = self._round_smart(vp.get('poc_percentage', 0), 'percentage')
            
            # Dwell time otimizado
            dwell = self._get_cached("dwell_time", lambda: calcular_dwell_time(df))
            enriched['dwell_price'] = self._round_smart(dwell.get('dwell_price', 0), 'price')
            enriched['dwell_seconds'] = int(round(dwell.get('dwell_seconds', 0)))
            enriched['dwell_location'] = dwell.get('dwell_location', 'N/A')
            
            # Trade speed otimizado
            speed = self._get_cached("trade_speed", lambda: calcular_trade_speed(df))
            enriched['trades_per_second'] = self._round_smart(speed.get('trades_per_second', 0), 'ratio')
            enriched['avg_trade_size'] = self._round_smart(speed.get('avg_trade_size', 0), 'average')
            
            self.enriched_data = enriched
            logging.debug("✅ Camada Enriched gerada com precisão mínima.")
            return enriched
            
        except Exception as e:
            logging.error(f"❌ Erro na camada Enriched: {e}", exc_info=True)
            return self._get_fallback_enriched()
    
    def _get_fallback_enriched(self) -> Dict:
        """Retorna dados enriched mínimos em caso de erro."""
        logging.warning("⚠️ Usando fallback para enriched data")
        
        # Tenta extrair dados básicos do DataFrame
        try:
            if self.df is not None and not self.df.empty:
                close_price = float(self.df["p"].iloc[-1])
                volume = float(self.df["q"].sum())
            else:
                close_price = 0.0
                volume = 0.0
        except Exception:
            close_price = 0.0
            volume = 0.0
        
        return {
            "symbol": self.symbol,
            "ohlc": {
                "open": close_price,
                "high": close_price,
                "low": close_price,
                "close": close_price,
                "open_time": 0,
                "close_time": 0,
                "vwap": close_price
            },
            "volume_total": volume,
            "volume_total_usdt": 0,
            "num_trades": len(self.df) if self.df is not None else 0,
            "delta_minimo": 0.0,
            "delta_maximo": 0.0,
            "delta_fechamento": 0.0,
            "reversao_desde_minimo": 0.0,
            "reversao_desde_maximo": 0.0,
            "dwell_price": close_price,
            "dwell_seconds": 0,
            "dwell_location": "N/A",
            "trades_per_second": 0.0,
            "avg_trade_size": 0.0,
            "poc_price": close_price,
            "poc_volume": 0.0,
            "poc_percentage": 0.0,
        }

    # ===============================
    # Camada 3 — Contextual com CACHE
    # ===============================
    
    def add_context(
        self,
        flow_metrics: Dict | None = None,
        historical_vp: Dict | None = None,
        orderbook_data: Dict | None = None,
        multi_tf: Dict | None = None,
        derivatives: Dict | None = None,
        market_context: Dict | None = None,
        market_environment: Dict | None = None,
    ) -> Dict[str, Any]:
        """Enriquece com contexto externo usando cache quando apropriado."""
        if self.enriched_data is None:
            self.enrich()
        
        # Normalizar orderbook_data
        if orderbook_data and isinstance(orderbook_data, dict):
            if 'orderbook_data' in orderbook_data:
                orderbook_data = orderbook_data['orderbook_data']
        
        contextual = dict(self.enriched_data)
        
        # Flow metrics - sempre atualizar
        contextual["flow_metrics"] = flow_metrics or {}
        
        # Historical VP - usar cache intensivo
        if historical_vp:
            cached_vp = {}
            for timeframe in ['daily', 'weekly', 'monthly']:
                if timeframe in historical_vp:
                    cache_key = f'vp_{timeframe}'
                    ttl = self.cache_ttl[f'vp_{timeframe}']
                    
                    cached_data = self.cache_manager.get(cache_key, ttl)
                    if cached_data is None or self.cache_manager.has_changed(cache_key, historical_vp[timeframe]):
                        self.cache_manager.set(cache_key, historical_vp[timeframe])
                        cached_vp[timeframe] = historical_vp[timeframe]
                        logging.debug(f"📊 VP {timeframe} atualizado no cache")
                    else:
                        cached_vp[timeframe] = cached_data
                        logging.debug(f"✨ VP {timeframe} obtido do cache")
            
            contextual["historical_vp"] = cached_vp
        else:
            contextual["historical_vp"] = {}
        
        # OrderBook - sempre atualizar
        contextual["orderbook_data"] = orderbook_data or {}
        
        # Multi TF - cache moderado
        if multi_tf:
            cached_mtf = self.cache_manager.get('multi_tf', self.cache_ttl['multi_tf'])
            if cached_mtf is None or self.cache_manager.has_changed('multi_tf', multi_tf):
                self.cache_manager.set('multi_tf', multi_tf)
                contextual["multi_tf"] = multi_tf
                logging.debug("📊 Multi TF atualizado no cache")
            else:
                contextual["multi_tf"] = cached_mtf
                logging.debug("✨ Multi TF obtido do cache")
        else:
            contextual["multi_tf"] = {}
        
        # Derivativos - cache curto
        if derivatives:
            cached_der = self.cache_manager.get('derivatives', self.cache_ttl['derivatives'])
            if cached_der is None or self.cache_manager.has_changed('derivatives', derivatives):
                self.cache_manager.set('derivatives', derivatives)
                contextual["derivatives"] = derivatives
            else:
                contextual["derivatives"] = cached_der
        else:
            contextual["derivatives"] = {}
        
        # Market context - cache médio
        if market_context:
            cached_ctx = self.cache_manager.get('market_context', self.cache_ttl['market_context'])
            if cached_ctx is None or self.cache_manager.has_changed('market_context', market_context):
                self.cache_manager.set('market_context', market_context)
                contextual["market_context"] = market_context
            else:
                contextual["market_context"] = cached_ctx
        else:
            contextual["market_context"] = {}
        
        contextual["market_environment"] = market_environment or {}
        
        self.contextual_data = contextual
        logging.debug("✅ Camada Contextual gerada com cache.")
        
        # Limpar cache antigo periodicamente
        if np.random.random() < 0.1:
            self.cache_manager.cleanup()
        
        return contextual

    # ===============================
    # Camada 4 — Signal (mantida)
    # ===============================
    
    def detect_signals(self, absorption_detector, exhaustion_detector, orderbook_data=None) -> List[Dict]:
        """Detecta sinais usando os detectores fornecidos."""
        if self.contextual_data is None:
            raise ValueError("Camada Contextual deve ser gerada antes.")
        
        signals: List[Dict[str, Any]] = []
        
        try:
            default_ts_ms = int(self.enriched_data.get("ohlc", {}).get("close_time", 0)) if self.enriched_data else None
        except Exception:
            default_ts_ms = None
        
        # Detectores de sinais
        if absorption_detector:
            try:
                absorption_event = absorption_detector(self.raw_trades, self.symbol)
                if absorption_event:
                    absorption_event["layer"] = "signal"
                    absorption_event = self._sanitize_event(absorption_event, default_ts_ms=default_ts_ms)
                    if absorption_event.get("is_signal", False):
                        signals.append(absorption_event)
            except Exception as e:
                logging.error(f"❌ Erro detectando absorção: {e}")
        
        if exhaustion_detector:
            try:
                exhaustion_event = exhaustion_detector(self.raw_trades, self.symbol)
                if exhaustion_event:
                    exhaustion_event["layer"] = "signal"
                    exhaustion_event = self._sanitize_event(exhaustion_event, default_ts_ms=default_ts_ms)
                    if exhaustion_event.get("is_signal", False):
                        signals.append(exhaustion_event)
            except Exception as e:
                logging.error(f"❌ Erro detectando exaustão: {e}")
        
        if isinstance(orderbook_data, dict) and orderbook_data.get("is_signal", False):
            try:
                ob_event = orderbook_data.copy()
                ob_event["layer"] = "signal"
                ob_event = self._sanitize_event(ob_event, default_ts_ms=default_ts_ms)
                signals.append(ob_event)
            except Exception as e:
                logging.error(f"❌ Erro adicionando evento OrderBook: {e}")
        
        # Evento de análise
        try:
            analysis_trigger = {
                "is_signal": True,
                "tipo_evento": "ANALYSIS_TRIGGER",
                "delta": self.enriched_data.get("delta_fechamento", 0),
                "volume_total": self.enriched_data.get("volume_total", 0),
                "preco_fechamento": self.enriched_data.get("ohlc", {}).get("close", 0),
            }
            analysis_trigger = self._sanitize_event(analysis_trigger, default_ts_ms=default_ts_ms)
            signals.append(analysis_trigger)
        except Exception as e:
            logging.error(f"❌ Erro adicionando evento de análise: {e}")
        
        self.signal_data = signals
        logging.debug(f"✅ Camada Signal gerada. {len(signals)} sinais detectados.")
        return signals

    # ===============================
    # Consolidação
    # ===============================
    
    def get_final_features(self) -> Dict[str, Any]:
        """Retorna todas as features consolidadas."""
        if self.enriched_data is None:
            self.enrich()
        if self.contextual_data is None:
            self.contextual_data = {
                **(self.enriched_data or {}),
                "flow_metrics": {},
                "historical_vp": {},
                "orderbook_data": {},
                "multi_tf": {},
                "derivatives": {},
            }
        if self.signal_data is None:
            self.signal_data = []
        
        try:
            close_time_ms = int(self.enriched_data.get("ohlc", {}).get("close_time", 0)) if self.enriched_data else None
        except Exception:
            close_time_ms = None
        
        # APENAS epoch_ms
        features = {
            "schema_version": "2.3.0",
            "symbol": self.symbol,
            "epoch_ms": close_time_ms or self.tm.now_ms(),
            "enriched": self.enriched_data or {},
            "contextual": self.contextual_data or {},
            "signals": self.signal_data,
            "ml_features": {},
        }
        
        # ML features (opcional)
        try:
            if generate_ml_features is not None and self.df is not None and self.contextual_data:
                orderbook_data = self.contextual_data.get("orderbook_data", {})
                flow_metrics = self.contextual_data.get("flow_metrics", {})
                
                df_for_ml = self.df.copy()
                
                if "close" not in df_for_ml.columns:
                    df_for_ml["close"] = pd.to_numeric(df_for_ml["p"], errors="coerce")
                else:
                    df_for_ml["close"] = pd.to_numeric(df_for_ml["close"], errors="coerce")
                
                df_for_ml["p"] = pd.to_numeric(df_for_ml.get("p", df_for_ml.get("close")), errors="coerce")
                df_for_ml["q"] = pd.to_numeric(df_for_ml.get("q", 0.0), errors="coerce")
                
                if "m" not in df_for_ml.columns:
                    df_for_ml["m"] = False
                else:
                    df_for_ml["m"] = df_for_ml["m"].fillna(False).astype(bool)
                
                df_for_ml = df_for_ml.dropna(subset=["close", "p", "q"])
                
                if len(df_for_ml) >= 3:  # Mínimo para ML
                    ml_feats = generate_ml_features(
                        df_for_ml,
                        orderbook_data,
                        flow_metrics,
                        lookback_windows=[1, 5, 15],
                        volume_ma_window=20,
                    )
                    features["ml_features"] = ml_feats
                else:
                    logging.warning("⚠️ Dados insuficientes para ML features")
        except Exception as e:
            logging.error(f"❌ Erro ao gerar ML features: {e}")
        
        return features

    # ===============================
    # FUNÇÕES DE OTIMIZAÇÃO EXTREMA
    # (Mantidas do código original - não alteradas)
    # ===============================
    
    def _should_send(self, new_data: Dict) -> Tuple[bool, str]:
        """Verifica se deve enviar baseado em mudanças significativas."""
        if self._last_payload_data is None:
            return True, "first_payload"
        
        try:
            old_price = self._last_payload_data.get('enriched', {}).get('ohlc', {}).get('close', 0)
            new_price = new_data.get('enriched', {}).get('ohlc', {}).get('close', 0)
            
            old_cvd = self._last_payload_data.get('contextual', {}).get('flow_metrics', {}).get('cvd', 0)
            new_cvd = new_data.get('contextual', {}).get('flow_metrics', {}).get('cvd', 0)
            
            old_volume = self._last_payload_data.get('enriched', {}).get('volume_total', 0)
            new_volume = new_data.get('enriched', {}).get('volume_total', 0)
            
            price_change_pct = abs((new_price - old_price) / old_price * 100) if old_price else 100
            cvd_change_abs = abs(new_cvd - old_cvd)
            volume_change_pct = abs((new_volume - old_volume) / old_volume * 100) if old_volume else 100
            
            if price_change_pct >= self._min_change_threshold['price_pct']:
                return True, f"price_change_{price_change_pct:.3f}%"
            
            if cvd_change_abs >= self._min_change_threshold['cvd_abs']:
                return True, f"cvd_change_{cvd_change_abs:.2f}"
            
            if volume_change_pct >= self._min_change_threshold['volume_pct']:
                return True, f"volume_change_{volume_change_pct:.3f}%"
            
            old_signals = self._last_payload_data.get('signals', [])
            new_signals = new_data.get('signals', [])
            
            if len(new_signals) > len(old_signals):
                return True, "new_signals"
            
            return False, "no_significant_change"
            
        except Exception as e:
            logging.warning(f"⚠️ Erro ao calcular mudanças: {e}")
            return True, "calculation_error"
    
    def _remove_empty_fields(self, obj: Any) -> Any:
        """Remove campos vazios mas mantém zeros significativos."""
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                cleaned_v = self._remove_empty_fields(v)
                if k in self.zero_significant_fields and cleaned_v == 0:
                    cleaned[k] = cleaned_v
                elif cleaned_v is not None and cleaned_v != {} and cleaned_v != []:
                    cleaned[k] = cleaned_v
            return cleaned if cleaned else None
        elif isinstance(obj, list):
            cleaned = [self._remove_empty_fields(item) for item in obj]
            return [item for item in cleaned if item is not None]
        else:
            return obj
    
    # ===============================
    # (Resto do código mantido igual)
    # ===============================
    
    def log_statistics(self):
        """Log de estatísticas."""
        cache_stats = self.cache_manager.get_stats()
        buffer_stats = self.event_buffer.get_stats()
        
        cache_hit_rate = (self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1) * 100)
        compression_rate = (self.stats['bytes_saved'] / max(self.stats['bytes_sent'] + self.stats['bytes_saved'], 1) * 100)
        
        logging.info(f"""
        📊 === ESTATÍSTICAS ===
        Cache: {cache_hit_rate:.1f}% hits ({cache_stats['items']} items)
        Buffer: {buffer_stats['dedup_rate']:.1f}% dedup ({buffer_stats['current_size']} eventos)
        Enviados: {self.stats['events_sent']} eventos
        Compressão: {compression_rate:.1f}% ({self.stats['bytes_saved']:,}B economizados)
        """)
    
    def close(self):
        """Fecha recursos."""
        self.log_statistics()
        if self._session:
            self._session.close()
            logging.debug("🔌 Session fechada")
    
    def __del__(self):
        """Destrutor."""
        try:
            self.close()
        except:
            pass


# ===============================
# Teste de validação
# ===============================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("🧪 TESTE DE VALIDAÇÃO - DataPipeline v2.3.0")
    print("="*70 + "\n")
    
    # Teste 1: Dados válidos (OK)
    print("✅ Teste 1: Dados válidos (10 trades)")
    try:
        sample_trades = [
            {"p": "67234.5", "q": "0.534", "T": 1759699445671, "m": True},
            {"p": "67234.8", "q": "0.312", "T": 1759699446000, "m": False},
            {"p": "67235.2", "q": "1.128", "T": 1759699447000, "m": True},
            {"p": "67235.0", "q": "0.645", "T": 1759699448000, "m": False},
            {"p": "67234.9", "q": "0.892", "T": 1759699449000, "m": True},
            {"p": "67235.5", "q": "1.234", "T": 1759699450000, "m": False},
            {"p": "67235.8", "q": "0.456", "T": 1759699451000, "m": True},
            {"p": "67236.0", "q": "0.789", "T": 1759699452000, "m": False},
            {"p": "67235.7", "q": "0.321", "T": 1759699453000, "m": True},
            {"p": "67235.9", "q": "0.567", "T": 1759699454000, "m": False},
        ]
        pipeline = DataPipeline(sample_trades, "BTCUSDT")
        enriched = pipeline.enrich()
        print(f"  ✅ Sucesso: {len(pipeline.df)} trades processados")
        print(f"  Preço: ${enriched['ohlc']['close']:.1f}")
    except Exception as e:
        print(f"  ❌ Falhou: {e}")
    
    # Teste 2: Poucos trades mas >= 3 (OK com aviso)
    print("\n⚠️ Teste 2: Poucos trades (5 trades - abaixo do recomendado)")
    try:
        few_trades = sample_trades[:5]
        pipeline2 = DataPipeline(few_trades, "BTCUSDT")
        enriched2 = pipeline2.enrich()
        print(f"  ✅ Sucesso com aviso: {len(pipeline2.df)} trades processados")
    except Exception as e:
        print(f"  ❌ Falhou: {e}")
    
    # Teste 3: Mínimo absoluto (3 trades - OK)
    print("\n⚠️ Teste 3: Mínimo absoluto (3 trades)")
    try:
        min_trades = sample_trades[:3]
        pipeline3 = DataPipeline(min_trades, "BTCUSDT")
        enriched3 = pipeline3.enrich()
        print(f"  ✅ Sucesso: {len(pipeline3.df)} trades processados")
    except Exception as e:
        print(f"  ❌ Falhou: {e}")
    
    # Teste 4: Insuficiente (2 trades - ERRO esperado)
    print("\n❌ Teste 4: Dados insuficientes (2 trades - deve falhar)")
    try:
        too_few = sample_trades[:2]
        pipeline4 = DataPipeline(too_few, "BTCUSDT")
        print(f"  ❌ Não deveria passar!")
    except ValueError as e:
        print(f"  ✅ Erro esperado capturado: {e}")
    
    # Teste 5: Lista vazia (ERRO esperado)
    print("\n❌ Teste 5: Lista vazia (deve falhar)")
    try:
        pipeline5 = DataPipeline([], "BTCUSDT")
        print(f"  ❌ Não deveria passar!")
    except ValueError as e:
        print(f"  ✅ Erro esperado capturado: {e}")
    
    print("\n" + "="*70)
    print("✅ TESTES CONCLUÍDOS")
    print("="*70 + "\n")