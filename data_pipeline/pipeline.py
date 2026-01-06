# data_pipeline/pipeline.py
from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import pandas as pd

from .config import PipelineConfig
from .logging_utils import PipelineLogger, setup_pipeline_logging
from .validation.validator import TradeValidator
from .validation.adaptive import AdaptiveThresholds
from .cache.lru_cache import LRUCache
from .metrics.processor import MetricsProcessor
from .fallback.registry import FallbackRegistry
from enrichment_integrator import enrich_analysis_trigger_event, build_analysis_trigger_event
import config as global_config
from data_enricher import DataEnricher

if TYPE_CHECKING:
    from time_manager import TimeManager

try:
    from ml_features import generate_ml_features
except ImportError:
    generate_ml_features = None

try:
    from datetime import datetime, timezone
    from cross_asset_correlations import get_cross_asset_features
except ImportError:
    get_cross_asset_features = None


class DataPipeline:
    """
    Pipeline de dados completo v3.2.1.

    Pipeline em 4 camadas:
    1. **Validação**: Limpa e valida dados brutos
    2. **Enriched**: Calcula métricas básicas (OHLC, volumes, etc)
    3. **Contextual**: Adiciona contexto externo (orderbook, flow, etc)
    4. **Signal**: Detecta sinais de trading

    Características:
    - Validação vetorizada (10-18x mais rápida)
    - Sistema adaptativo de thresholds
    - Cache inteligente com TTL
    - Logging granular (5 níveis)
    - Fallback automático
    - ML features encapsulado
    - Rastreabilidade completa
    """

    # thresholds compartilhados por símbolo (opcional)
    _shared_adaptive_thresholds: Dict[str, AdaptiveThresholds] = {}

    def __init__(
        self,
        raw_trades: List[Dict[str, Any]],
        symbol: str,
        time_manager: Optional["TimeManager"] = None,
        config: Optional[PipelineConfig] = None,
        shared_adaptive: bool = True,
        validator: Optional[TradeValidator] = None,
        metrics_processor: Optional[MetricsProcessor] = None,
    ) -> None:
        """
        Inicializa pipeline.

        Args:
            raw_trades: Lista de trades brutos
            symbol: Símbolo do ativo (ex: "BTCUSDT")
            time_manager: Gerenciador de tempo (opcional)
            config: Configurações customizadas (opcional)
            shared_adaptive: Se True, usa thresholds adaptativos compartilhados
            validator: Validador customizado (opcional, para DI)
            metrics_processor: Processador de métricas customizado (opcional, para DI)
        """
        self.symbol = symbol
        self.config = config or PipelineConfig.from_config_file()
        self.time_manager = time_manager

        # Logger especializado
        self.logger = PipelineLogger(symbol)

        # Fallback registry
        self.fallback_registry = FallbackRegistry()

        # Sistema adaptativo
        if self.config.enable_adaptive_thresholds:
            if shared_adaptive:
                # 1 instância de AdaptiveThresholds por símbolo
                adaptive = DataPipeline._shared_adaptive_thresholds.get(symbol)
                if adaptive is None:
                    adaptive = AdaptiveThresholds(
                        initial_min_trades=self.config.min_trades_pipeline,
                        absolute_min_trades=self.config.min_absolute_trades,
                        learning_rate=self.config.adaptive_learning_rate,
                        confidence_threshold=self.config.adaptive_confidence,
                    )
                    DataPipeline._shared_adaptive_thresholds[symbol] = adaptive
                self.adaptive = adaptive
            else:
                # Instância independente por pipeline
                self.adaptive = AdaptiveThresholds(
                    initial_min_trades=self.config.min_trades_pipeline,
                    absolute_min_trades=self.config.min_absolute_trades,
                    learning_rate=self.config.adaptive_learning_rate,
                    confidence_threshold=self.config.adaptive_confidence,
                )
        else:
            self.adaptive = None

        # Cache
        self._cache = LRUCache(
            max_items=self.config.cache_max_items,
            ttl_seconds=self.config.cache_ttl_seconds
        )

        # Injeção de dependências (permite mocks em teste)
        self._validator = validator or TradeValidator(
            enable_vectorized=self.config.enable_vectorized_validation,
            logger=self.logger
        )
        self._metrics = metrics_processor or MetricsProcessor(self.config, symbol, self.logger)

        # Inicializa DataEnricher com config global (config.py)
        try:
            config_dict = {
                k: getattr(global_config, k)
                for k in dir(global_config)
                if not k.startswith("__")
            }
        except Exception:
            config_dict = {}

        self._data_enricher = DataEnricher(config_dict)

        # Dados
        self.df: Optional[pd.DataFrame] = None
        self.enriched_data: Optional[Dict[str, Any]] = None
        self.contextual_data: Optional[Dict[str, Any]] = None
        self.signal_data: Optional[List[Dict[str, Any]]] = None

        # Stats
        self._load_stats: Optional[Dict[str, Any]] = None
        self._creation_time = time.time()

        # Carregar dados
        self._load_trades(raw_trades)

    def _load_trades(self, raw_trades: List[Dict[str, Any]]) -> None:
        """
        Carrega e valida trades com sistema adaptativo.

        Args:
            raw_trades: Lista de trades brutos
        """
        try:
            current_threshold = self.config.min_trades_pipeline

            # Sistema adaptativo
            if self.adaptive:
                self.adaptive.record_observation(len(raw_trades))
                new_threshold, reason = self.adaptive.adjust(
                    self.config.allow_limited_data
                )
                current_threshold = new_threshold

                if reason.startswith('adjusted'):
                    self.logger.adaptive_info(
                        f"🧠 Threshold adaptado",
                        new_threshold=new_threshold,
                        reason=reason
                    )

            # Validar trades
            self.df, validation_stats = self._validator.validate_batch(
                raw_trades,
                min_trades=self.config.min_absolute_trades,
                max_price_variance_pct=self.config.max_price_variance_pct
            )

            self._load_stats = validation_stats

            # Avisar se dados limitados
            if len(self.df) < current_threshold:
                if self.config.allow_limited_data:
                    self.logger.validation_warning(
                        f"⚠️ Dados limitados",
                        trades=len(self.df),
                        recommended=current_threshold,
                        time_ms=validation_stats['validation_time_ms']
                    )
                else:
                    raise ValueError(
                        f"Dados insuficientes: {len(self.df)} < {current_threshold}"
                    )
            else:
                self.logger.validation_info(
                    f"✅ Pipeline carregado",
                    trades=len(self.df),
                    method=validation_stats['method'],
                    time_ms=validation_stats['validation_time_ms'],
                    rate=f"{validation_stats.get('trades_per_ms', 0):.0f}/ms"
                )

        except Exception as e:
            self.logger.runtime_error(
                f"❌ Erro ao carregar trades: {e}",
                exc_info=True
            )
            raise

    def enrich(self) -> Dict[str, Any]:
        """
        Gera camada Enriched com métricas básicas.

        Calcula:
        - OHLC (Open, High, Low, Close, VWAP)
        - Volumes (base e quote)
        - Métricas intra-candle
        - Volume profile
        - Dwell time
        - Trade speed

        Returns:
            Dicionário com dados enriquecidos
        """
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame não carregado")

        cache_key = f"enriched_{self.symbol}_{len(self.df)}_{int(self.df['T'].iloc[-1])}"

        # Verificar cache
        cached = self._cache.get(cache_key, allow_expired=True)
        if cached and not self._cache.is_expired(cache_key):
            self.logger.performance_info("✨ Enriched cache hit")
            self.enriched_data = cached
            return cached

        try:
            from data_handler import (
                calcular_metricas_intra_candle,
                calcular_volume_profile,
                calcular_dwell_time,
                calcular_trade_speed,
            )

            # Métricas básicas
            ohlc = self._metrics.calculate_ohlc(self.df)
            volume_metrics = self._metrics.calculate_volume_metrics(self.df)

            enriched: Dict[str, Any] = {
                "symbol": self.symbol,
                "ohlc": ohlc,
                **volume_metrics,
            }

            # Métricas avançadas com fallback individual
            try:
                metricas = calcular_metricas_intra_candle(self.df)
                for key, value in metricas.items():
                    if isinstance(value, (int, float)):
                        # Use 4 decimals for volume/delta related metrics to avoid invariant violations
                        # against Buy/Sell volume sums (which are 4 decimals)
                        if "delta" in key or "volume" in key:
                            enriched[key] = self._metrics.round_value(value, 4)
                        else:
                            enriched[key] = self._metrics.round_value(value, 2)
                    else:
                        enriched[key] = value
            except Exception as e:
                fallback_info = self.fallback_registry.register(
                    'metricas_intra_candle',
                    'calculation_error',
                    e
                )
                enriched.update(fallback_info)
                self.logger.runtime_warning(
                    f"⚠️ Fallback: métricas intra-candle",
                    error=str(e)[:50]
                )

            # Volume Profile
            try:
                vp = calcular_volume_profile(self.df)
                enriched['poc_price'] = self._metrics.round_value(vp.get('poc_price', 0))
                enriched['poc_volume'] = self._metrics.round_value(vp.get('poc_volume', 0), 2)
                enriched['poc_percentage'] = self._metrics.round_value(vp.get('poc_percentage', 0), 1)
            except Exception as e:
                fallback_info = self.fallback_registry.register(
                    'volume_profile',
                    'calculation_error',
                    e
                )
                enriched.update(fallback_info)
                self.logger.runtime_warning(
                    f"⚠️ Fallback: volume profile",
                    error=str(e)[:50]
                )

            # Dwell Time
            try:
                dwell = calcular_dwell_time(self.df)
                enriched['dwell_price'] = self._metrics.round_value(dwell.get('dwell_price', 0))
                enriched['dwell_seconds'] = int(round(dwell.get('dwell_seconds', 0)))
                enriched['dwell_location'] = dwell.get('dwell_location', 'N/A')
            except Exception as e:
                fallback_info = self.fallback_registry.register(
                    'dwell_time',
                    'calculation_error',
                    e
                )
                enriched.update(fallback_info)
                self.logger.runtime_warning(
                    f"⚠️ Fallback: dwell time",
                    error=str(e)[:50]
                )

            # Trade Speed
            try:
                speed = calcular_trade_speed(self.df)
                enriched['trades_per_second'] = self._metrics.round_value(speed.get('trades_per_second', 0), 2)
                enriched['avg_trade_size'] = self._metrics.round_value(speed.get('avg_trade_size', 0), 3)
            except Exception as e:
                fallback_info = self.fallback_registry.register(
                    'trade_speed',
                    'calculation_error',
                    e
                )
                enriched.update(fallback_info)
                self.logger.runtime_warning(
                    f"⚠️ Fallback: trade speed",
                    error=str(e)[:50]
                )

            # NOVO: advanced_analysis completo (com price_targets, etc.)
            try:
                if getattr(global_config, "ENABLE_DATA_ENRICHMENT", True):
                    # Usa o snapshot enriquecido completo como raw_event para o DataEnricher
                    advanced = self._data_enricher.enrich_from_raw_event(enriched)
                    enriched["advanced_analysis"] = advanced
            except Exception as e:
                self.logger.runtime_warning(
                    "⚠️ Fallback: advanced_analysis enrichment",
                    error=str(e)[:80]
                )

            # Armazenar no cache
            self._cache.set(cache_key, enriched, force_fresh=True)
            self.enriched_data = enriched

            # Validação de invariantes
            if self.enriched_data:
                self._validate_invariants(self.enriched_data, context="enrich")

            self.logger.runtime_info("✅ Camada Enriched gerada")
            return enriched

        except Exception as e:
            fallback_info = self.fallback_registry.register(
                'enrich',
                'complete_failure',
                e
            )
            self.logger.runtime_error(
                f"❌ Fallback completo: enrich",
                exc_info=True
            )
            result = self._get_minimal_enriched()
            result.update(fallback_info)
            return result

    def _get_minimal_enriched(self) -> Dict[str, Any]:
        """Retorna dados enriched mínimos em caso de erro total."""
        if self.df is None or self.df.empty:
            close_price = 0.0
            volume = 0.0
        else:
            close_price = float(self.df["p"].iloc[-1])
            volume = float(self.df["q"].sum())

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
        }

    def add_context(
        self,
        flow_metrics: Optional[Dict[str, Any]] = None,
        historical_vp: Optional[Dict[str, Any]] = None,
        orderbook_data: Optional[Dict[str, Any]] = None,
        multi_tf: Optional[Dict[str, Any]] = None,
        derivatives: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        market_environment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Adiciona contexto externo aos dados enriquecidos.

        Args:
            flow_metrics: Métricas de fluxo
            historical_vp: Volume profile histórico
            orderbook_data: Dados do orderbook
            multi_tf: Dados multi-timeframe
            derivatives: Dados de derivativos
            market_context: Contexto geral do mercado
            market_environment: Ambiente de mercado

        Returns:
            Dicionário com dados contextuais
        """
        if self.enriched_data is None:
            self.enrich()

        # Normalizar orderbook se necessário
        if orderbook_data and 'orderbook_data' in orderbook_data:
            orderbook_data = orderbook_data['orderbook_data']

        contextual: Dict[str, Any] = {
            **self.enriched_data,
            "flow_metrics": flow_metrics or {},
            "historical_vp": historical_vp or {},
            "orderbook_data": orderbook_data or {},
            "multi_tf": multi_tf or {},
            "derivatives": derivatives or {},
            "market_context": market_context or {},
            "market_environment": market_environment or {},
        }

        # Enriquecimento institucional no snapshot COMPLETO (outer raw_event)
        try:
            if getattr(global_config, "ENABLE_DATA_ENRICHMENT", True):
                advanced = self._data_enricher.enrich_from_raw_event(contextual)
                contextual["advanced_analysis"] = advanced

                # Compatibilidade: também copia para o raw_event interno (se existir)
                if isinstance(contextual.get("raw_event"), dict):
                    contextual["raw_event"]["advanced_analysis"] = advanced
        except Exception as e:
            self.logger.runtime_warning(
                "⚠️ Fallback: advanced_analysis (contextual)",
                error=str(e)[:80]
            )

        self.contextual_data = contextual
        self.logger.runtime_info("✅ Camada Contextual gerada")

        return contextual

    def detect_signals(
        self,
        absorption_detector: Optional[Callable] = None,
        exhaustion_detector: Optional[Callable] = None,
        orderbook_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detecta sinais de trading usando detectores fornecidos.

        Args:
            absorption_detector: Função para detectar absorção
            exhaustion_detector: Função para detectar exaustão
            orderbook_data: Dados do orderbook com possíveis sinais

        Returns:
            Lista de sinais detectados
        """
        if self.contextual_data is None:
            raise ValueError("Camada Contextual deve ser gerada antes")

        signals: List[Dict[str, Any]] = []

        # Timestamp padrão
        try:
            default_ts_ms = int(
                self.enriched_data.get("ohlc", {}).get("close_time", 0)
            )
        except:
            default_ts_ms = int(time.time() * 1000)

        # Detectar absorção
        if absorption_detector and callable(absorption_detector):
            try:
                absorption_event = absorption_detector(
                    self.df.to_dict('records'),
                    self.symbol
                )
                if absorption_event and absorption_event.get("is_signal"):
                    absorption_event["epoch_ms"] = absorption_event.get(
                        "epoch_ms",
                        default_ts_ms
                    )
                    self._validate_invariants(absorption_event, context="signal")
                    signals.append(absorption_event)
            except Exception as e:
                self.logger.runtime_error(
                    f"❌ Erro detectando absorção: {e}"
                )

        # Detectar exaustão
        if exhaustion_detector and callable(exhaustion_detector):
            try:
                exhaustion_event = exhaustion_detector(
                    self.df.to_dict('records'),
                    self.symbol
                )
                if exhaustion_event and exhaustion_event.get("is_signal"):
                    exhaustion_event["epoch_ms"] = exhaustion_event.get(
                        "epoch_ms",
                        default_ts_ms
                    )
                    self._validate_invariants(exhaustion_event, context="signal")
                    signals.append(exhaustion_event)
            except Exception as e:
                self.logger.runtime_error(
                    f"❌ Erro detectando exaustão: {e}"
                )

        # OrderBook signal
        if orderbook_data and orderbook_data.get("is_signal"):
            try:
                ob_event = orderbook_data.copy()
                ob_event["epoch_ms"] = ob_event.get("epoch_ms", default_ts_ms)
                self._validate_invariants(ob_event, context="signal")
                signals.append(ob_event)
            except Exception as e:
                self.logger.runtime_error(
                    f"❌ Erro OrderBook: {e}"
                )

        # Evento de análise (sempre gerado)
        try:
            # Dados para o raw_event - INCLUIR DADOS CONTEXTUAIS COMPLETOS
            raw_event_data = {
                # Dados básicos do enriched
                "delta": self.enriched_data.get("delta_fechamento", 0),
                "volume_total": self.enriched_data.get("volume_total", 0),
                "volume_compra": self.enriched_data.get("volume_compra", 0),
                "volume_venda": self.enriched_data.get("volume_venda", 0),
                "preco_fechamento": self.enriched_data.get("ohlc", {}).get("close", 0),
                "advanced_analysis": (
                    self.contextual_data.get("advanced_analysis", {})
                    if self.contextual_data
                    else self.enriched_data.get("advanced_analysis", {})
                ),
                # Dados contextuais necessários para enrich_event_with_advanced_analysis
                "multi_tf": self.contextual_data.get("multi_tf", {}) if self.contextual_data else {},
                "historical_vp": self.contextual_data.get("historical_vp", {}) if self.contextual_data else {},
                "liquidity_heatmap": self.contextual_data.get("liquidity_heatmap", {}) if self.contextual_data else {},
                "flow_metrics": self.contextual_data.get("flow_metrics", {}) if self.contextual_data else {},
                "orderbook_data": self.contextual_data.get("orderbook_data", {}) if self.contextual_data else {},
                "timestamp_utc": self.enriched_data.get("ohlc", {}).get("close_time"),
            }
            analysis_trigger = build_analysis_trigger_event(self.symbol, raw_event_data)
            analysis_trigger["epoch_ms"] = default_ts_ms
            self._validate_invariants(analysis_trigger, context="signal")

            try:
                self._attach_advanced_analysis_to_full_raw_event(analysis_trigger)
            except Exception as e:
                self.logger.runtime_warning(
                    "⚠️ Fallback: attach advanced_analysis (full raw_event)",
                    error=str(e)[:80]
                )

            signals.append(analysis_trigger)
        except Exception as e:
            self.logger.runtime_error(f"❌ Erro análise: {e}")

        self.signal_data = signals
        self.logger.runtime_info(
            f"✅ Camada Signal gerada",
            signals=len(signals)
        )

        return signals

    def _attach_advanced_analysis_to_full_raw_event(self, event: dict) -> None:
        """
        event = o evento ANALYSIS_TRIGGER completo.
        Usa enrich_event_with_advanced_analysis para calcular usando raw_event EXTERNO.
        """
        if not hasattr(self, "_data_enricher") or self._data_enricher is None:
            return

        # Usar a nova função que calcula usando raw_event EXTERNO
        # CORREÇÃO: Capturar o retorno da função e atualizar o evento
        try:
            self.logger.runtime_info("🔧 Chamando enrich_event_with_advanced_analysis...")
            updated_event = self._data_enricher.enrich_event_with_advanced_analysis(event)
            if updated_event and updated_event != event:
                # Atualizar o evento com os dados modificados
                event.update(updated_event)
                
                # Garantir que o raw_event também seja atualizado se necessário
                if "raw_event" in updated_event:
                    event["raw_event"] = updated_event["raw_event"]
                    
                # Log de diagnóstico para verificar se advanced_analysis foi adicionado
                raw_event = event.get("raw_event", {})
                if "advanced_analysis" in raw_event:
                    advanced = raw_event["advanced_analysis"]
                    self.logger.runtime_info(
                        f"✅ advanced_analysis adicionado com sucesso - "
                        f"keys={list(advanced.keys()) if isinstance(advanced, dict) else 'N/A'}"
                    )
                else:
                    self.logger.runtime_warning(
                        "⚠️ advanced_analysis NÃO foi adicionado ao raw_event"
                    )
            else:
                self.logger.runtime_warning(
                    "⚠️ enrich_event_with_advanced_analysis não retornou dados válidos"
                )
        except Exception as e:
            self.logger.runtime_error(
                f"❌ Erro ao chamar enrich_event_with_advanced_analysis: {e}"
            )

    # ============================
    # VALIDATE INVARIANTS
    # ============================

    def _validate_invariants(self, data: Dict[str, Any], context: str = "enrich") -> None:
        """
        Valida invariantes.

        Contexto 'enrich': Valida OHLC básico.
        Contexto 'signal': Valida consistência de volumes de compra/venda e delta.
        """
        try:
            TOLERANCE_BTC = 1e-2  # Aumentar tolerância para 0.01 BTC para evitar erros de centavos

            if context == "enrich":
                # Validação OHLC
                ohlc = data.get("ohlc", {})
                if ohlc:
                    h = float(ohlc.get("high", 0))
                    l = float(ohlc.get("low", 0))
                    c = float(ohlc.get("close", 0))
                    o = float(ohlc.get("open", 0))

                    if l > h:
                        logging.warning(f"⚠️ INVARIANTE VIOLADA (OHLC): Low ({l}) > High ({h}) em {self.symbol}")

                    if not (l <= c <= h) or not (l <= o <= h):
                        logging.warning(f"⚠️ INVARIANTE VIOLADA (OHLC): Open/Close fora dos limites High/Low")

            elif context == "signal":
                # Validação de Volumes de Sinal (onde temos a quebra Buy/Sell)
                vol_buy = float(data.get("volume_compra", 0.0))
                vol_sell = float(data.get("volume_venda", 0.0))
                vol_total = float(data.get("volume_total", 0.0))
                delta = float(data.get("delta", 0.0))

                # 1. Soma dos volumes
                vol_sum = vol_buy + vol_sell
                # Nota: volume_total às vezes vem direto do candle original e buy/sell são inferidos
                # Uma divergência grande indica perda de dados na inferência de agressão 'm'
                if abs(vol_sum - vol_total) > TOLERANCE_BTC:
                    # Apenas loga se a diferença for relevante (> 0.0001 BTC por exemplo)
                    if abs(vol_sum - vol_total) > 0.0001:
                        logging.warning(
                            f"⚠️ INVARIANTE VIOLADA (Vol Sum): "
                            f"Buy({vol_buy:.4f}) + Sell({vol_sell:.4f}) != Total({vol_total:.4f}) "
                            f"[Diff: {vol_sum - vol_total:.6f}] em {self.symbol}"
                        )

                # 2. Consistência do Delta
                delta_calc = vol_buy - vol_sell
                if abs(delta_calc - delta) > TOLERANCE_BTC:
                    logging.warning(
                        f"⚠️ INVARIANTE VIOLADA (Delta): "
                        f"Buy - Sell ({delta_calc:.4f}) != Delta Armazenado ({delta:.4f}) "
                        f"em {self.symbol}"
                    )
        except Exception as e:
            logging.debug(f"Erro na validação de invariantes Pipeline ({context}): {e}", exc_info=True)

    def extract_features(self) -> Dict[str, Any]:
        """
        🤖 Extrai features de ML de forma encapsulada.

        Returns:
            Dicionário com features ML ou vazio se não disponível
        """
        if not generate_ml_features:
            self.logger.ml_warning("⚠️ generate_ml_features não disponível")
            return {}

        if self.df is None or len(self.df) < 3:
            self.logger.ml_warning(
                "⚠️ Dados insuficientes para ML",
                trades=len(self.df) if self.df is not None else 0
            )
            return {}

        try:
            df_ml = self.df.copy()
            df_ml["close"] = df_ml["p"]

            orderbook_data = (
                self.contextual_data.get("orderbook_data", {})
                if self.contextual_data else {}
            )
            flow_metrics = (
                self.contextual_data.get("flow_metrics", {})
                if self.contextual_data else {}
            )

            ml_features = generate_ml_features(
                df_ml,
                orderbook_data,
                flow_metrics,
                lookback_windows=[1, 5, 15],
                volume_ma_window=20,
                symbol=self.symbol,  # 🆕 Para cross-asset features
            )

            # 🆕 NOVO: Adicionar cross-asset features para BTCUSDT
            if self.symbol == "BTCUSDT" and get_cross_asset_features is not None:
                try:
                    now_utc = datetime.now(timezone.utc)
                    cross_asset_features = get_cross_asset_features(now_utc)
                    
                    if cross_asset_features.get("status") == "ok":
                        # Adicionar ao ml_features
                        if "cross_asset" not in ml_features:
                            ml_features["cross_asset"] = {}
                        
                        # Mapear as features conforme especificação
                        ml_features["cross_asset"].update({
                            "btc_eth_corr_7d": cross_asset_features.get("btc_eth_corr_7d"),
                            "btc_eth_corr_30d": cross_asset_features.get("btc_eth_corr_30d"),
                            "btc_dxy_corr_30d": cross_asset_features.get("btc_dxy_corr_30d"),
                            "btc_dxy_corr_90d": cross_asset_features.get("btc_dxy_corr_90d"),
                            "btc_ndx_corr_30d": cross_asset_features.get("btc_ndx_corr_30d"),
                            "dxy_return_5d": cross_asset_features.get("dxy_return_5d"),
                            "dxy_return_20d": cross_asset_features.get("dxy_return_20d"),
                        })
                        
                        self.logger.ml_info(
                            "✅ Cross-asset features adicionadas ao ANALYSIS_TRIGGER",
                            cross_asset_count=len(ml_features["cross_asset"])
                        )
                    else:
                        self.logger.ml_warning(
                            f"⚠️ Cross-asset features falharam: {cross_asset_features.get('error')}"
                        )
                        
                except Exception as e:
                    self.logger.ml_warning(
                        f"⚠️ Erro ao calcular cross-asset features: {e}"
                    )

            self.logger.ml_info(
                "✅ ML features geradas",
                feature_count=len(ml_features)
            )

            return ml_features

        except Exception as e:
            fallback_info = self.fallback_registry.register(
                'ml_features',
                'extraction_error',
                e
            )
            self.logger.ml_warning(
                f"⚠️ Erro extraindo ML features",
                error=str(e)[:50]
            )
            return fallback_info

    def get_final_features(self) -> Dict[str, Any]:
        """
        Retorna todas as features consolidadas.

        Returns:
            Dicionário com todas as camadas e ML features
        """
        if self.enriched_data is None:
            self.enrich()

        if self.contextual_data is None:
            self.add_context()

        if self.signal_data is None:
            self.signal_data = []

        # Timestamp
        try:
            close_time_ms = int(
                self.enriched_data.get("ohlc", {}).get("close_time", 0)
            )
        except:
            close_time_ms = int(time.time() * 1000)

        features: Dict[str, Any] = {
            "schema_version": "3.2.1",
            "symbol": self.symbol,
            "epoch_ms": close_time_ms,
            "enriched": self.enriched_data,
            "contextual": self.contextual_data,
            "signals": self.signal_data,
            "ml_features": self.extract_features(),
        }

        # Adicionar metadados de fallback se houver
        fallback_stats = self.fallback_registry.get_stats()
        if fallback_stats['total_fallbacks'] > 0:
            features['_fallback_stats'] = fallback_stats

        return features

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas completas do pipeline.

        Returns:
            Dicionário com todas as métricas
        """
        uptime = time.time() - self._creation_time

        stats: Dict[str, Any] = {
            'symbol': self.symbol,
            'trades': len(self.df) if self.df is not None else 0,
            'cache': self._cache.stats(),
            'validation': self._validator.get_stats(),
            'uptime_seconds': round(uptime, 2),
        }

        if self._load_stats:
            stats['load'] = self._load_stats

        if self.adaptive:
            stats['adaptive'] = self.adaptive.get_stats()

        # Fallback stats
        fallback_stats = self.fallback_registry.get_stats()
        if fallback_stats['total_fallbacks'] > 0:
            stats['fallbacks'] = fallback_stats

        return stats

    def close(self) -> None:
        """Fecha recursos do pipeline."""
        self._cache.clear()
        self.logger.runtime_info("🔌 Pipeline fechado")

    @classmethod
    def reset_adaptive_thresholds(cls) -> None:
        """Reseta thresholds adaptativos compartilhados."""
        for adaptive in cls._shared_adaptive_thresholds.values():
            adaptive.reset()
