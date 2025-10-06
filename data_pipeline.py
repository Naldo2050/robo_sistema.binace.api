# data_pipeline.py - VERSÃO FINAL ULTRA-OTIMIZADA COM MÁXIMA REDUÇÃO
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
        self.min_events = min_events  # Aumentado para economizar requisições
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
        Pipeline de dados ultra-otimizado com precisão mínima necessária.
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
        
        # Cache manager para dados que mudam pouco
        self.cache_manager = CacheManager()
        
        # Buffer de eventos (min_events aumentado para economizar requisições)
        self.event_buffer = EventBuffer(max_size=100, max_age_seconds=60, min_events=20)
        
        # Session HTTP persistente
        self._session: requests.Session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'DataPipeline/5.0',
            'Connection': 'keep-alive',
            'Keep-Alive': 'timeout=120, max=1000'
        })
        
        # Cache de último payload
        self._last_payload_hash: Optional[str] = None
        self._last_payload_data: Optional[Dict] = None
        self._last_vp_hash: Optional[str] = None
        self._last_mtf_hash: Optional[str] = None
        
        # Thresholds de mudança mínima (mais restritivos)
        self._min_change_threshold = {
            'price_pct': 0.03,    # 0.03% de mudança mínima no preço
            'cvd_abs': 0.2,       # 0.2 de mudança absoluta no CVD
            'volume_pct': 0.1     # 0.1% de mudança no volume
        }
        
        # TTL para cache (em segundos)
        self.cache_ttl = {
            'vp_daily': 3600,      # 1 hora
            'vp_weekly': 21600,    # 6 horas
            'vp_monthly': 86400,   # 24 horas
            'multi_tf': 300,       # 5 minutos
            'derivatives': 60,     # 1 minuto
            'market_context': 900  # 15 minutos
        }
        
        # Backoff para rate limiting
        self._backoff_seconds = 1
        self._max_backoff = 60
        self._last_request_time = 0
        
        # Mapeamentos de símbolos (2 caracteres apenas)
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
        
        # ESCALA DE PREÇO DINÂMICA POR SÍMBOLO (OTIMIZADA)
        self.price_scales = {
            'BTCUSDT': 10,         # Tick 0.1 → escala 10
            'ETHUSDT': 100,        # Tick 0.01 → escala 100
            'BNBUSDT': 100,        # Tick 0.01 → escala 100
            'SOLUSDT': 1000,       # Tick 0.001 → escala 1000
            'XRPUSDT': 10000,      # Tick 0.0001 → escala 10000
            'DOGEUSDT': 100000,    # Tick 0.00001 → escala 100000
            'ADAUSDT': 10000,      # Tick 0.0001 → escala 10000
            'DEFAULT': 10
        }
        
        # Obter escala para o símbolo atual
        self.ohlc_scale = self.price_scales.get(self.symbol, self.price_scales['DEFAULT'])
        logging.debug(f"📏 Usando escala de preço {self.ohlc_scale} para {self.symbol}")
        
        # CONFIGURAÇÃO DE PRECISÃO ULTRA-OTIMIZADA
        self.precision_config = {
            # Preços - mínimo necessário baseado no tick
            'price': self._get_price_precision(),
            'price_precise': self._get_price_precision(),  # Sem precisão extra
            
            # Volumes - REDUZIDO
            'volume_btc': 2,       # 4.11 (2 casas suficiente para BTC)
            'volume_usdt': 0,      # Inteiro sempre
            
            # Deltas - REDUZIDO
            'delta': 1,            # 1.2 (1 casa suficiente)
            'delta_pct': 2,        # 0.12 (2 casas para %)
            
            # Índices e ratios - REDUZIDO
            'index': 3,            # 0.096 (3 casas)
            'ratio': 1,            # 2.6 (1 casa)
            
            # Tempos - REDUZIDO
            'time_seconds': 0,     # 54 (inteiro)
            
            # Médias - REDUZIDO
            'average': 3,          # 0.033 (3 casas)
            'percentage': 1,       # 15.2 (1 casa)
            
            # Fatores - REDUZIDO
            'factor': 1,           # 1.3 (1 casa)
            'sensitivity': 0       # 2 (inteiro)
        }
        
        # Thresholds de inclusão (mais restritivos)
        self.inclusion_thresholds = {
            'tps_min': 10,                # TPS mínimo para incluir
            'imbalance_min': 0.25,        # Imbalance mínimo
            'pressure_min': 0.35,          # Pressure mínimo
            'whale_delta_min': 0.1,        # Whale delta mínimo
            'cvd_min': 0.1,                # CVD mínimo
            'buy_sell_ratio_extreme': (0.2, 5),  # Só se < 0.2 ou > 5
            'volume_ratio_extreme': (0.3, 3),    # Só se < 0.3 ou > 3
            'momentum_diff': 0.2,          # |momentum - 1| mínimo
            'bp_min': 0.1,                 # Buy/sell pressure mínimo
            'flow_imbalance_min': 0.3,     # Flow imbalance mínimo
            'trade_intensity_min': 30,     # Trade intensity mínimo
            'volatility_min_bps': 2        # Volatilidade mínima em bps
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
        
        self._validate_and_load()
    
    def _get_price_precision(self) -> int:
        """Retorna precisão decimal baseada na escala do símbolo."""
        scale_to_precision = {
            10: 1,      # 0.1
            100: 2,     # 0.01
            1000: 3,    # 0.001
            10000: 4,   # 0.0001
            100000: 5   # 0.00001
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

    def _validate_and_load(self):
        """Valida, normaliza e carrega trades crus em DataFrame."""
        if not self.raw_trades or len(self.raw_trades) < 2:
            raise ValueError("Dados insuficientes para pipeline.")
        
        try:
            validated: List[Dict[str, Any]] = []
            for t in self.raw_trades:
                if not isinstance(t, dict):
                    continue
                
                p = self._coerce_float(t.get("p"))
                q = self._coerce_float(t.get("q"))
                T = self._coerce_int(t.get("T"))
                m = t.get("m", np.nan)
                
                if p is None or q is None or T is None:
                    continue
                if p <= 0 or q <= 0 or T <= 0:
                    continue
                
                validated.append({"p": p, "q": q, "T": T, "m": m})
            
            if not validated:
                raise ValueError("Nenhum trade válido após validação.")
            
            df = pd.DataFrame(validated)
            df = df.dropna(subset=["p", "q", "T"])
            df = df.sort_values("T", kind="mergesort").reset_index(drop=True)
            
            if df.empty:
                raise ValueError("DataFrame vazio após limpeza.")
            if len(df) < 2:
                raise ValueError("Menos de 2 trades válidos após limpeza.")
            
            self.df = df
            
        except Exception as e:
            logging.error(f"Erro ao carregar dados crus: {e}")
            raise

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
        
        # Remover TODOS timestamps redundantes
        for field in ['timestamp_utc', 'timestamp_ny', 'timestamp_sp', 'timestamp', 
                     'window_open_ms', 'window_close_ms', 'window_duration_ms',
                     'open_time_iso', 'close_time_iso']:
            e.pop(field, None)
        
        return e
    
    def _round_smart(self, value: float, value_type: str) -> float:
        """
        Arredonda valor com precisão MÍNIMA baseada no tipo.
        """
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
    # Camada 2 — Enriched (ULTRA-OTIMIZADO)
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
                raise ValueError("DF não carregado na pipeline.")
            
            # Preços com precisão mínima
            open_price = self._round_smart(float(df["p"].iloc[0]), 'price')
            close_price = self._round_smart(float(df["p"].iloc[-1]), 'price')
            high_price = self._round_smart(float(df["p"].max()), 'price')
            low_price = self._round_smart(float(df["p"].min()), 'price')
            
            open_time = int(df["T"].iloc[0])
            close_time = int(df["T"].iloc[-1])
            
            # Volumes otimizados
            base_volume = self._round_smart(float(df["q"].sum()), 'volume_btc')
            quote_volume = int(round((df["p"] * df["q"]).sum()))  # Sempre inteiro
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
                "volume_total_usdt": quote_volume,  # Inteiro
                "num_trades": int(len(df)),
            }
            
            # Métricas com precisão MÍNIMA
            metricas = self._get_cached("metricas_intra", lambda: calcular_metricas_intra_candle(df))
            for key, value in metricas.items():
                if 'delta' in key or 'reversao' in key:
                    enriched[key] = self._round_smart(value, 'delta')  # 1 casa decimal
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
            enriched['dwell_seconds'] = int(round(dwell.get('dwell_seconds', 0)))  # Inteiro
            enriched['dwell_location'] = dwell.get('dwell_location', 'N/A')
            
            # Trade speed otimizado
            speed = self._get_cached("trade_speed", lambda: calcular_trade_speed(df))
            enriched['trades_per_second'] = self._round_smart(speed.get('trades_per_second', 0), 'ratio')
            enriched['avg_trade_size'] = self._round_smart(speed.get('avg_trade_size', 0), 'average')
            
            self.enriched_data = enriched
            logging.debug("✅ Camada Enriched gerada com precisão mínima.")
            return enriched
            
        except Exception as e:
            logging.error(f"Erro na camada Enriched: {e}")
            return self._get_fallback_enriched()
    
    def _get_fallback_enriched(self) -> Dict:
        """Retorna dados enriched mínimos em caso de erro."""
        return {
            "symbol": self.symbol,
            "ohlc": {
                "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0,
                "open_time": 0, "close_time": 0, "vwap": 0.0
            },
            "volume_total": 0.0,
            "volume_total_usdt": 0,
            "num_trades": 0,
            "delta_minimo": 0.0,
            "delta_maximo": 0.0,
            "delta_fechamento": 0.0,
            "reversao_desde_minimo": 0.0,
            "reversao_desde_maximo": 0.0,
            "dwell_price": 0.0,
            "dwell_seconds": 0,
            "dwell_location": "N/A",
            "trades_per_second": 0.0,
            "avg_trade_size": 0.0,
            "poc_price": 0.0,
            "poc_volume": 0.0,
            "poc_percentage": 0.0,
        }

    # ===============================
    # Camada 3 — Contextual com CACHE (mantida)
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
    # Camada 4 — Signal (mantida mas otimizada)
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
                logging.error(f"Erro detectando absorção: {e}")
        
        if exhaustion_detector:
            try:
                exhaustion_event = exhaustion_detector(self.raw_trades, self.symbol)
                if exhaustion_event:
                    exhaustion_event["layer"] = "signal"
                    exhaustion_event = self._sanitize_event(exhaustion_event, default_ts_ms=default_ts_ms)
                    if exhaustion_event.get("is_signal", False):
                        signals.append(exhaustion_event)
            except Exception as e:
                logging.error(f"Erro detectando exaustão: {e}")
        
        if isinstance(orderbook_data, dict) and orderbook_data.get("is_signal", False):
            try:
                ob_event = orderbook_data.copy()
                ob_event["layer"] = "signal"
                ob_event = self._sanitize_event(ob_event, default_ts_ms=default_ts_ms)
                signals.append(ob_event)
            except Exception as e:
                logging.error(f"Erro adicionando evento OrderBook: {e}")
        
        # Evento de análise (simplificado)
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
            logging.error(f"Erro adicionando evento de análise: {e}")
        
        self.signal_data = signals
        logging.debug(f"✅ Camada Signal gerada. {len(signals)} sinais detectados.")
        return signals

    # ===============================
    # Consolidação (otimizada)
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
            "schema_version": "1.1.0",
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
                
                ml_feats = generate_ml_features(
                    df_for_ml,
                    orderbook_data,
                    flow_metrics,
                    lookback_windows=[1, 5, 15],
                    volume_ma_window=20,
                )
                features["ml_features"] = ml_feats
        except Exception as e:
            logging.error(f"Erro ao gerar ML features: {e}")
        
        return features

    # ===============================
    # FUNÇÕES DE OTIMIZAÇÃO EXTREMA
    # ===============================
    
    def _should_send(self, new_data: Dict) -> Tuple[bool, str]:
        """Verifica se deve enviar baseado em mudanças significativas (MAIS RESTRITIVO)."""
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
            logging.warning(f"Erro ao calcular mudanças: {e}")
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
    
    def _compress_ohlc(self, ohlc: Dict) -> List[int]:
        """OHLC como array de inteiros com escala dinâmica."""
        if not ohlc:
            return []
        
        return [
            int(round(ohlc.get('open', 0) * self.ohlc_scale)),
            int(round(ohlc.get('high', 0) * self.ohlc_scale)),
            int(round(ohlc.get('low', 0) * self.ohlc_scale)),
            int(round(ohlc.get('close', 0) * self.ohlc_scale))
        ]
    
    def compress_for_api(self, data: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
        """
        Compressão EXTREMA com thresholds rigorosos.
        """
        symbol_id = self.symbol_ids.get(self.symbol, self.symbol[:2])
        
        compressed = {
            'v': '5.0',  # Versão com máxima otimização
            'sym': symbol_id,
            'ts': int(data.get('epoch_ms', self.tm.now_ms()) / 1000),  # segundos
            's': self.ohlc_scale  # escala de preço
        }
        
        # Enriched data (ULTRA-COMPACTO)
        if 'enriched' in data and data['enriched']:
            e = data['enriched']
            compressed['e'] = {
                'o': self._compress_ohlc(e.get('ohlc', {})),  # Array de inteiros
                'v': self._round_smart(e.get('volume_total', 0), 'volume_btc'),  # 2 casas
                'vu': e.get('volume_total_usdt', 0),  # Já é inteiro
                'd': self._round_smart(e.get('delta_fechamento', 0), 'delta'),  # 1 casa
            }
            
            # POC apenas se existir (inteiro escalado)
            if e.get('poc_price', 0) > 0:
                compressed['e']['poc'] = int(round(e['poc_price'] * self.ohlc_scale))
            
            # VWAP (inteiro escalado)
            if 'ohlc' in e and e['ohlc'].get('vwap', 0) > 0:
                compressed['e']['vw'] = int(round(e['ohlc']['vwap'] * self.ohlc_scale))
            
            # Dwell price (inteiro escalado)
            if e.get('dwell_price', 0) > 0:
                compressed['e']['dp'] = int(round(e['dwell_price'] * self.ohlc_scale))
            
            # Trade speed apenas se > threshold
            tps = e.get('trades_per_second', 0)
            if tps > self.inclusion_thresholds['tps_min']:
                compressed['e']['tps'] = int(round(tps))  # Inteiro
        
        # Contextual com thresholds rigorosos
        if 'contextual' in data and data['contextual']:
            ctx = data['contextual']
            compressed['c'] = {}
            
            # Flow metrics (com thresholds)
            if 'flow_metrics' in ctx and ctx['flow_metrics']:
                fm = ctx['flow_metrics']
                flow_data = {}
                
                # CVD apenas se significativo
                cvd = fm.get('cvd', 0)
                if abs(cvd) > self.inclusion_thresholds['cvd_min']:
                    flow_data['c'] = self._round_smart(cvd, 'delta')  # 1 casa
                
                # Whale delta apenas se significativo
                wd = fm.get('whale_delta', 0)
                if abs(wd) > self.inclusion_thresholds['whale_delta_min']:
                    flow_data['w'] = self._round_smart(wd, 'delta')  # 1 casa
                
                # Tipo de absorção
                ta = self.absorption_codes.get(fm.get('tipo_absorcao', ''), 0)
                if ta != 0:
                    flow_data['t'] = ta
                
                # Order flow apenas se extremo
                if 'order_flow' in fm:
                    of = fm['order_flow']
                    
                    # Buy/sell ratio apenas se extremo
                    ratio = of.get('buy_sell_ratio', 1)
                    min_r, max_r = self.inclusion_thresholds['buy_sell_ratio_extreme']
                    if ratio < min_r or ratio > max_r:
                        flow_data['r'] = self._round_smart(ratio, 'ratio')  # 1 casa
                
                if flow_data:
                    compressed['c']['f'] = flow_data
            
            # OrderBook apenas se desequilibrado
            if 'orderbook_data' in ctx and ctx['orderbook_data']:
                ob = ctx['orderbook_data']
                ob_data = {}
                
                # Imbalance com threshold
                imb = ob.get('imbalance', 0)
                if abs(imb) > self.inclusion_thresholds['imbalance_min']:
                    ob_data['i'] = self._round_smart(imb, 'ratio')  # 1 casa
                
                # Pressure com threshold
                p = ob.get('pressure', 0)
                if abs(p) > self.inclusion_thresholds['pressure_min']:
                    ob_data['p'] = self._round_smart(p, 'ratio')  # 1 casa
                
                # Volume ratio apenas se extremo
                vr = ob.get('volume_ratio', 1)
                min_vr, max_vr = self.inclusion_thresholds['volume_ratio_extreme']
                if vr < min_vr or vr > max_vr:
                    ob_data['vr'] = self._round_smart(vr, 'ratio')  # 1 casa
                
                # Spread em bps inteiros
                if 'spread_metrics' in ob:
                    spread_pct = ob['spread_metrics'].get('spread_percent', 0)
                    if spread_pct > 0:
                        ob_data['s'] = int(spread_pct * 10000)  # bps inteiros
                
                if ob_data:
                    compressed['c']['ob'] = ob_data
            
            # VP - usar hash sempre que possível
            if use_cache and 'historical_vp' in ctx and ctx['historical_vp']:
                vp = ctx['historical_vp']
                vp_str = json.dumps(vp, sort_keys=True)
                vp_hash = hashlib.md5(vp_str.encode()).hexdigest()[:6]  # Apenas 6 chars
                
                if self._last_vp_hash and self._last_vp_hash == vp_hash:
                    compressed['c']['vh'] = vp_hash  # Apenas hash
                else:
                    compressed['c']['vp'] = self._ultra_compress_vp(vp)
                    self._last_vp_hash = vp_hash
            
            # MTF - usar hash sempre que possível
            if use_cache and 'multi_tf' in ctx and ctx['multi_tf']:
                mtf = ctx['multi_tf']
                mtf_str = json.dumps(mtf, sort_keys=True)
                mtf_hash = hashlib.md5(mtf_str.encode()).hexdigest()[:6]  # Apenas 6 chars
                
                if self._last_mtf_hash and self._last_mtf_hash == mtf_hash:
                    compressed['c']['mh'] = mtf_hash  # Apenas hash
                else:
                    compressed['c']['mt'] = self._compress_mtf(mtf)
                    self._last_mtf_hash = mtf_hash
            
            # Derivativos (funding rate em bps inteiros)
            if 'derivatives' in ctx and ctx['derivatives']:
                der_compressed = self._compress_derivatives_optimized(ctx['derivatives'])
                if der_compressed:
                    compressed['c']['d'] = der_compressed
        
        # Sinais ultra-compactos
        if 'signals' in data and data['signals']:
            signals_compressed = self._compress_signals_optimized(data['signals'])
            if signals_compressed:
                compressed['sg'] = signals_compressed
        
        # ML Features - MÍNIMO NECESSÁRIO
        if 'ml_features' in data and data['ml_features']:
            ml_compressed = self._compress_ml_features_extreme(data['ml_features'])
            if ml_compressed:
                compressed['ml'] = ml_compressed
        
        # Remover campos vazios
        compressed = self._remove_empty_fields(compressed)
        
        return compressed
    
    def _ultra_compress_vp(self, vp: Dict) -> List[int]:
        """VP como array flat de inteiros [d_poc, d_val, d_vah, w_poc, ...]"""
        values = []
        for tf in ['daily', 'weekly', 'monthly']:
            if tf in vp and vp[tf] and vp[tf].get('status') == 'success':
                data = vp[tf]
                values.extend([
                    int(data.get('poc', 0)),
                    int(data.get('val', 0)),
                    int(data.get('vah', 0))
                ])
            else:
                values.extend([0, 0, 0])
        return values if any(values) else None
    
    def _compress_mtf(self, mtf: Dict) -> List[int]:
        """MTF como array de tendências [-1, 0, 1]"""
        trends = []
        for tf in ['15m', '1h', '4h']:
            if tf in mtf and mtf[tf]:
                trend = mtf[tf].get('tendencia', '').lower()
                trends.append(1 if 'alta' in trend else -1 if 'baixa' in trend else 0)
            else:
                trends.append(0)
        return trends if any(t != 0 for t in trends) else None
    
    def _compress_derivatives_optimized(self, derivatives: Dict) -> Dict:
        """Comprime derivativos com funding rate em bps inteiros."""
        compressed = {}
        for symbol, data in derivatives.items():
            if data and isinstance(data, dict):
                key = self.symbol_ids.get(symbol, symbol[:2])
                der_data = {}
                
                # Funding rate em bps inteiros
                fr = data.get('funding_rate_percent', 0)
                if fr != 0:
                    der_data['f'] = int(fr * 10000)  # bps inteiros
                
                # Open interest apenas se significativo
                oi = data.get('open_interest', 0)
                if oi > 1000:  # Threshold mínimo
                    der_data['o'] = int(oi / 1000)  # Em milhares
                
                # Long/short ratio apenas se extremo
                ls = data.get('long_short_ratio', 1)
                if ls < 0.5 or ls > 2:
                    der_data['l'] = int(ls * 10)  # 1 casa como inteiro
                
                if der_data:
                    compressed[key] = der_data
        
        return compressed if compressed else None
    
    def _compress_signals_optimized(self, signals: List[Dict]) -> List[Dict]:
        """Sinais ultra-compactos com mínimo de dados."""
        compressed = []
        for sig in signals:
            if not sig.get('is_signal'):
                continue
            
            c_sig = [self.event_codes.get(sig.get('tipo_evento', ''), 0)]
            
            # Adicionar dados mínimos por tipo
            if sig.get('tipo_evento') in ['Absorção', 'Absorção de Venda', 'Absorção de Compra']:
                # Índice apenas se muito significativo
                idx = sig.get('indice_absorcao', 0)
                if abs(idx) > 0.01:
                    c_sig.append(int(idx * 1000))  # 3 casas como inteiro
            
            elif sig.get('tipo_evento') == 'ANALYSIS_TRIGGER':
                # Apenas preço escalado
                c_sig.append(int(round(sig.get('preco_fechamento', 0) * self.ohlc_scale)))
            
            compressed.append(c_sig)
        
        return compressed if compressed else None
    
    def _compress_ml_features_extreme(self, ml: Dict) -> Dict:
        """ML features com máxima compressão e thresholds rigorosos."""
        compressed = {}
        
        # Price features - MÍNIMO
        if 'price_features' in ml:
            pf = ml['price_features']
            
            # Momentum apenas se muito diferente de neutro
            mom = pf.get('momentum_score', 1)
            if abs(mom - 1) > self.inclusion_thresholds['momentum_diff']:
                compressed['m'] = int((mom - 1) * 10)  # Delta de 1, 1 casa como inteiro
            
            # Volatilidades apenas se significativas (bps inteiros)
            for period in ['1', '5', '15']:
                vol_key = f'volatility_{period}'
                if vol_key in pf:
                    vol_bps = int(pf[vol_key] * 10000)
                    if vol_bps >= self.inclusion_thresholds['volatility_min_bps']:
                        compressed[f'v{period}'] = vol_bps
        
        # Volume features - MÍNIMO
        if 'volume_features' in ml:
            vf = ml['volume_features']
            
            # Buy/sell pressure apenas se extremo
            bp = vf.get('buy_sell_pressure', 0)
            if abs(bp) > self.inclusion_thresholds['bp_min']:
                compressed['b'] = int(bp * 10)  # 1 casa como inteiro
            
            # Volume ratio apenas se muito anormal
            vr = vf.get('volume_sma_ratio', 1)
            if vr > 3 or vr < 0.3:
                compressed['vr'] = int(vr * 10)  # 1 casa como inteiro
        
        # Microstructure apenas se extremo
        if 'microstructure' in ml:
            ms = ml['microstructure']
            
            # Flow imbalance apenas se muito significativo
            fi = ms.get('flow_imbalance', 0)
            if abs(fi) > self.inclusion_thresholds['flow_imbalance_min']:
                compressed['fi'] = int(fi * 10)  # 1 casa como inteiro
            
            # Trade intensity apenas se muito alta
            ti = ms.get('trade_intensity', 0)
            if ti > self.inclusion_thresholds['trade_intensity_min']:
                compressed['ti'] = int(ti)
        
        return compressed if compressed else None
    
    def create_batch_payload(self, events: List[Dict]) -> Dict:
        """Cria payload de batch ultra-compacto."""
        if not events:
            return {}
        
        symbol_id = self.symbol_ids.get(self.symbol, self.symbol[:2])
        
        # Dados comuns mínimos
        common = {
            's': symbol_id,
            't': int(time.time()),
            'sc': self.ohlc_scale,
            'n': len(events)
        }
        
        # Extrair VP/MTF comum apenas se todos eventos têm o mesmo
        first_event = events[0] if events else {}
        if 'contextual' in first_event:
            ctx = first_event['contextual']
            
            # VP comum (apenas se todos eventos têm o mesmo)
            if 'historical_vp' in ctx:
                vp_compressed = self._ultra_compress_vp(ctx['historical_vp'])
                if vp_compressed and any(vp_compressed):
                    common['v'] = vp_compressed
            
            # MTF comum
            if 'multi_tf' in ctx:
                mtf_compressed = self._compress_mtf(ctx['multi_tf'])
                if mtf_compressed:
                    common['m'] = mtf_compressed
        
        # Eventos individuais mínimos
        individual_events = []
        for event in events:
            compressed = self.compress_for_api(event, use_cache=True)
            
            # Remover dados já em common
            compressed.pop('sym', None)
            compressed.pop('s', None)
            
            if 'c' in compressed:
                compressed['c'].pop('vp', None)
                compressed['c'].pop('vh', None)
                compressed['c'].pop('mt', None)
                compressed['c'].pop('mh', None)
                
                if not compressed['c']:
                    compressed.pop('c', None)
            
            individual_events.append(compressed)
        
        return {
            'c': common,
            'e': individual_events
        }
    
    def _apply_backoff(self):
        """Aplica backoff exponencial."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._backoff_seconds:
            sleep_time = self._backoff_seconds - time_since_last
            logging.info(f"⏱️ Backoff: {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _handle_response_error(self, response: requests.Response):
        """Trata erros de resposta."""
        if response.status_code == 429:
            self._backoff_seconds = min(self._backoff_seconds * 2, self._max_backoff)
            logging.warning(f"⚠️ Rate limit. Backoff: {self._backoff_seconds}s")
        elif response.status_code >= 500:
            self._backoff_seconds = min(self._backoff_seconds * 1.5, self._max_backoff)
            logging.warning(f"⚠️ Erro servidor. Backoff: {self._backoff_seconds}s")
        elif response.status_code == 200:
            self._backoff_seconds = max(1, self._backoff_seconds * 0.9)
    
    def send_optimized_batch(self, events: List[Dict], api_url: str, api_key: str = None, use_buffer: bool = True) -> Optional[requests.Response]:
        """Envia batch ultra-otimizado com máxima economia."""
        try:
            # Buffer de eventos
            if use_buffer:
                added_count = 0
                for event in events:
                    if self.event_buffer.add(event):
                        added_count += 1
                        self.stats['events_buffered'] += 1
                
                logging.debug(f"📥 {added_count}/{len(events)} eventos no buffer")
                
                # Verificar se deve enviar
                if not self.event_buffer.should_flush(force=False):
                    buffer_stats = self.event_buffer.get_stats()
                    logging.debug(f"⏳ Buffer: {buffer_stats['current_size']} eventos, "
                                f"{buffer_stats['buffer_age']:.1f}s")
                    return None
                
                events_to_send = self.event_buffer.get_events(clear=True)
                logging.info(f"📤 Flush: {len(events_to_send)} eventos únicos")
            else:
                events_to_send = events
            
            if not events_to_send:
                return None
            
            # Verificar mudanças
            if events_to_send:
                should_send, reason = self._should_send(events_to_send[0])
                if not should_send:
                    logging.debug(f"📊 Pulado: {reason}")
                    # Re-adicionar ao buffer se não enviado
                    if use_buffer:
                        for event in events_to_send:
                            self.event_buffer.add(event)
                    return None
                logging.debug(f"📊 Enviando: {reason}")
            
            # Backoff
            self._apply_backoff()
            
            # Criar payload
            payload = self.create_batch_payload(events_to_send)
            
            # Comprimir
            json_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            
            compressed = None
            encoding = 'gzip'
            
            if BROTLI_AVAILABLE:
                try:
                    compressed = brotli.compress(json_bytes, quality=11)
                    encoding = 'br'
                except Exception:
                    pass
            
            if compressed is None:
                compressed = gzip.compress(json_bytes, compresslevel=9)
                encoding = 'gzip'
            
            # Headers
            headers = {
                'Content-Type': 'application/json',
                'Content-Encoding': encoding,
                'X-Format-Version': '5.0'
            }
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            # Estatísticas
            original_size = sum(len(json.dumps(e).encode()) for e in events_to_send) if events_to_send else 1
            compressed_size = len(compressed)
            savings = original_size - compressed_size
            savings_pct = (savings / original_size * 100) if original_size > 0 else 0
            
            self.stats['bytes_sent'] += compressed_size
            self.stats['bytes_saved'] += savings
            self.stats['events_sent'] += len(events_to_send)
            
            logging.info(f"📤 Batch: {len(events_to_send)} eventos")
            logging.info(f"📊 {compressed_size:,}B ({savings_pct:.1f}% economia)")
            
            # Enviar
            response = self._session.post(api_url, data=compressed, headers=headers, timeout=30)
            
            self._handle_response_error(response)
            
            if response.status_code == 200:
                logging.info("✅ Sucesso")
                if events_to_send:
                    self._last_payload_data = events_to_send[0]
            else:
                logging.error(f"❌ HTTP {response.status_code}")
            
            return response
            
        except requests.exceptions.Timeout:
            logging.error("⏱️ Timeout")
            self._backoff_seconds = min(self._backoff_seconds * 2, self._max_backoff)
            return None
        except Exception as e:
            logging.error(f"❌ Erro: {e}")
            return None
    
    def run_optimized(self, absorption_detector, exhaustion_detector, orderbook_data, api_url: str, api_key: str = None):
        """Pipeline completo ultra-otimizado."""
        try:
            # Gerar dados
            self.enrich()
            self.add_context(orderbook_data=orderbook_data)
            signals = self.detect_signals(absorption_detector, exhaustion_detector, orderbook_data)
            
            # Evento consolidado
            consolidated_event = {
                'enriched': self.enriched_data,
                'contextual': self.contextual_data,
                'signals': signals,
                'ml_features': self.get_final_features().get('ml_features', {})
            }
            
            # Enviar
            response = self.send_optimized_batch([consolidated_event], api_url, api_key, use_buffer=True)
            
            # Estatísticas periódicas
            if np.random.random() < 0.05:
                self.log_statistics()
            
            return response
            
        except Exception as e:
            logging.error(f"❌ Erro: {e}")
            return None
    
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
    
    def flush_buffer(self, api_url: str, api_key: str = None) -> Optional[requests.Response]:
        """Força flush do buffer."""
        if self.event_buffer.buffer:
            logging.info(f"🚀 Flush forçado: {len(self.event_buffer.buffer)} eventos")
            events = self.event_buffer.get_events(clear=True)
            return self.send_optimized_batch(events, api_url, api_key, use_buffer=False)
        return None
    
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
    
    # Métodos de compatibilidade
    def batch_events(self, events: List[Dict], max_size: int = 10) -> List[List[Dict]]:
        """Compatibilidade."""
        batches = []
        current_batch = []
        for event in events:
            current_batch.append(event)
            if len(current_batch) >= max_size:
                batches.append(current_batch)
                current_batch = []
        if current_batch:
            batches.append(current_batch)
        return batches
    
    def send_to_api(self, batch: List[Dict], api_url: str, api_key: str = None):
        """Compatibilidade."""
        return self.send_optimized_batch(batch, api_url, api_key, use_buffer=False)
    
    def run_and_send(self, absorption_detector, exhaustion_detector, orderbook_data, api_url: str, api_key: str = None):
        """Compatibilidade."""
        return self.run_optimized(absorption_detector, exhaustion_detector, orderbook_data, api_url, api_key)


# ===============================
# Exemplo de uso
# ===============================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Dados de exemplo para BTCUSDT com valores realistas
    sample_trades = [
        {"p": "122726.9", "q": "1.534", "T": 1759699445671, "m": True},
        {"p": "122726.8", "q": "0.812", "T": 1759699446000, "m": False},
        {"p": "122726.9", "q": "2.128", "T": 1759699447000, "m": True},
    ]
    
    # Criar pipeline
    pipeline = DataPipeline(sample_trades, "BTCUSDT")
    
    try:
        # Simular múltiplas janelas
        for i in range(5):
            logging.info(f"\n=== Janela {i+1} ===")
            
            # Simular pequena variação (menos de 0.03%)
            for trade in sample_trades:
                trade["p"] = str(float(trade["p"]) * (1 + np.random.uniform(-0.0001, 0.0001)))
                trade["q"] = str(float(trade["q"]) * (1 + np.random.uniform(-0.05, 0.05)))
            
            pipeline = DataPipeline(sample_trades, "BTCUSDT")
            pipeline.run_optimized(
                absorption_detector=None,
                exhaustion_detector=None,
                orderbook_data={'imbalance': 0.3, 'pressure': 0.4, 'volume_ratio': 0.2},
                api_url="https://api.example.com/analyze",
                api_key="your_api_key"
            )
            
            time.sleep(0.1)
        
        # Flush final
        pipeline.flush_buffer("https://api.example.com/analyze", "your_api_key")
        
    finally:
        pipeline.close()