"""
Institutional Analytics Engine — Coordenador dos 23 módulos de análise.

Centraliza todas as chamadas aos módulos implementados nos Dias 1-5.
Chamado UMA VEZ por janela de processamento.
Retorna resultado unificado para enriquecimento de sinais e payload IA.

Uso:
    engine = InstitutionalAnalyticsEngine(symbol="BTCUSDT")
    result = engine.compute_all(
        current_price=64892,
        flow_metrics={...},
        vp_data={...},
        orderbook_data={...},
        candles_df=df,
        macro_context={...},
        derivatives_data={...},
    )
"""

import logging
import time
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class InstitutionalAnalyticsEngine:
    """
    Motor de análise institucional.
    
    Mantém estado dos trackers que precisam de histórico:
      - SpreadTracker (spread percentile)
      - AbsorptionZoneMapper (zonas de absorção)
      - WhaleAccumulationCalculator (score de acumulação)
      - ReferencePrices (closes anteriores)
    
    Os outros módulos são stateless e chamados diretamente.
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self._initialized = False
        self._init_errors: List[str] = []

        # ═══════════════════════════════════════
        # Módulos STATEFUL (mantêm histórico)
        # ═══════════════════════════════════════
        
        # #20 — Spread Percentile Tracker
        try:
            from orderbook_analyzer.spread_tracker import SpreadTracker
            self.spread_tracker = SpreadTracker(window_minutes=1440)
        except Exception as e:
            self.spread_tracker = None
            self._init_errors.append(f"SpreadTracker: {e}")

        # #17 — Absorption Zone Mapper
        try:
            from flow_analyzer.absorption import AbsorptionZoneMapper
            self.absorption_mapper = AbsorptionZoneMapper(
                zone_tolerance_pct=0.15,
                max_history_hours=24,
            )
        except Exception as e:
            self.absorption_mapper = None
            self._init_errors.append(f"AbsorptionZoneMapper: {e}")

        # #16 — Whale Accumulation Calculator
        try:
            from flow_analyzer.whale_score import WhaleAccumulationCalculator
            self.whale_calculator = WhaleAccumulationCalculator(history_window=30)
        except Exception as e:
            self.whale_calculator = None
            self._init_errors.append(f"WhaleAccumulationCalculator: {e}")

        # #2 — Reference Prices
        try:
            from support_resistance.reference_prices import ReferencePrices
            self.reference_prices = ReferencePrices()
        except Exception as e:
            self.reference_prices = None
            self._init_errors.append(f"ReferencePrices: {e}")

        # ═══════════════════════════════════════
        # Módulos STATELESS (importar para uso)
        # ═══════════════════════════════════════
        self._modules = {}

        # #4 — S/R Strength
        try:
            from support_resistance.sr_strength import SRStrengthScorer
            self._modules["sr_strength"] = SRStrengthScorer()
        except Exception as e:
            self._init_errors.append(f"SRStrengthScorer: {e}")

        # #5 — Defense Zones
        try:
            from support_resistance.defense_zones import DefenseZoneDetector
            self._modules["defense_zones"] = DefenseZoneDetector()
        except Exception as e:
            self._init_errors.append(f"DefenseZoneDetector: {e}")

        # Log de inicialização
        if self._init_errors:
            for err in self._init_errors:
                logger.warning(f"InstitutionalAnalytics init warning: {err}")
        
        self._initialized = True
        logger.info(
            f"InstitutionalAnalyticsEngine initialized for {symbol} "
            f"({len(self._init_errors)} warnings)"
        )

    def compute_all(
        self,
        current_price: float,
        flow_metrics: Optional[Dict[str, Any]] = None,
        vp_data: Optional[Dict[str, Any]] = None,
        orderbook_data: Optional[Dict[str, Any]] = None,
        candles_df: Optional[pd.DataFrame] = None,
        trades_df: Optional[pd.DataFrame] = None,
        macro_context: Optional[Dict[str, Any]] = None,
        derivatives_data: Optional[Dict[str, Any]] = None,
        onchain_data: Optional[Dict[str, Any]] = None,
        absorption_data: Optional[Dict[str, Any]] = None,
        pivot_data: Optional[Dict[str, Any]] = None,
        ema_values: Optional[Dict[str, Any]] = None,
        weekly_vp: Optional[Dict[str, Any]] = None,
        monthly_vp: Optional[Dict[str, Any]] = None,
        window_close_ms: Optional[int] = None,
        time_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Calcula TODOS os indicadores institucionais de uma vez.
        
        Retorna dict organizado por seção. Cada seção pode falhar
        independentemente sem afetar as outras.
        """
        result = {
            "status": "ok",
            "computed_at_ms": int(time.time() * 1000),
            "errors": [],
        }

        # ═══════════════════════════════════════
        # SEÇÃO 1: INDICADORES TÉCNICOS EXTRAS
        # (#8 StochRSI, #9 Williams%R, #3 TWAP)
        # ═══════════════════════════════════════
        result["technical_extras"] = self._compute_technical_extras(
            candles_df, current_price, flow_metrics
        )

        # ═══════════════════════════════════════
        # SEÇÃO 2: VOLUME PROFILE AVANÇADO
        # (#7 Poor H/L, #11 Shape, #12 VA%, 
        #  #6 No-Man's Land, #13 HVN/LVN Strength)
        # ═══════════════════════════════════════
        result["profile_analysis"] = self._compute_profile_analysis(
            trades_df, vp_data, current_price, weekly_vp, monthly_vp
        )

        # ═══════════════════════════════════════
        # SEÇÃO 3: FLOW AVANÇADO
        # (#14 Passive/Aggressive, #15 Buy/Sell Ratio,
        #  #16 Whale Score, #17 Absorption Zones)
        # ═══════════════════════════════════════
        result["flow_analysis"] = self._compute_flow_analysis(
            flow_metrics, orderbook_data, absorption_data,
            derivatives_data, onchain_data
        )

        # ═══════════════════════════════════════
        # SEÇÃO 4: S/R AVANÇADO
        # (#4 S/R Strength, #5 Defense Zones,
        #  #2 Reference Prices)
        # ═══════════════════════════════════════
        result["sr_analysis"] = self._compute_sr_analysis(
            current_price, vp_data, pivot_data, ema_values,
            candles_df, weekly_vp, monthly_vp, orderbook_data,
            absorption_data
        )

        # ═══════════════════════════════════════
        # SEÇÃO 5: QUALIDADE E INFRAESTRUTURA
        # (#18 Completeness, #19 Anomaly, 
        #  #20 Spread %, #21 Latency, #22 Calendar)
        # ═══════════════════════════════════════
        result["quality"] = self._compute_quality(
            current_price, orderbook_data, flow_metrics,
            window_close_ms, time_manager
        )

        # ═══════════════════════════════════════
        # SEÇÃO 6: CANDLESTICK PATTERNS
        # (#23 — já integrado via recognize_patterns)
        # ═══════════════════════════════════════
        result["candlestick_patterns"] = self._compute_candlestick(candles_df)

        return result

    # ═══════════════════════════════════════════════════════════
    # SEÇÃO 1: INDICADORES TÉCNICOS EXTRAS
    # ═══════════════════════════════════════════════════════════
    def _compute_technical_extras(
        self, candles_df, current_price, flow_metrics
    ) -> dict:
        extras = {}

        try:
            if candles_df is not None and len(candles_df) >= 14:
                from technical_indicators import stochastic_rsi, williams_r, twap_vwap_analysis

                # Detectar colunas
                close_col = None
                high_col = None
                low_col = None
                vol_col = None
                for c in candles_df.columns:
                    cl = c.lower()
                    if cl in ("close", "c"):
                        close_col = c
                    elif cl in ("high", "h"):
                        high_col = c
                    elif cl in ("low", "l"):
                        low_col = c
                    elif cl in ("volume", "v", "q"):
                        vol_col = c

                if close_col:
                    closes = candles_df[close_col].astype(float)

                    # #8 — Stochastic RSI
                    try:
                        k, d = stochastic_rsi(closes)
                        k_val = float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0
                        d_val = float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50.0
                        extras["stoch_rsi"] = {
                            "k": round(k_val, 2),
                            "d": round(d_val, 2),
                            "overbought": k_val > 80,
                            "oversold": k_val < 20,
                            "crossover": (
                                "bullish" if len(k) >= 2 and k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]
                                else "bearish" if len(k) >= 2 and k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2]
                                else "none"
                            ),
                        }
                    except Exception as e:
                        extras["stoch_rsi"] = {"error": str(e)}

                    # #9 — Williams %R
                    if high_col and low_col:
                        try:
                            highs = candles_df[high_col].astype(float)
                            lows = candles_df[low_col].astype(float)
                            wr = williams_r(highs, lows, closes)
                            wr_val = float(wr.iloc[-1]) if not pd.isna(wr.iloc[-1]) else -50.0
                            extras["williams_r"] = {
                                "value": round(wr_val, 2),
                                "overbought": wr_val > -20,
                                "oversold": wr_val < -80,
                                "zone": (
                                    "overbought" if wr_val > -20
                                    else "oversold" if wr_val < -80
                                    else "neutral"
                                ),
                            }
                        except Exception as e:
                            extras["williams_r"] = {"error": str(e)}

                    # #3 — TWAP vs VWAP
                    if vol_col:
                        try:
                            volumes = candles_df[vol_col].astype(float)
                            vwap_ref = None
                            if flow_metrics and isinstance(flow_metrics, dict):
                                of = flow_metrics.get("order_flow", {})
                                if isinstance(of, dict):
                                    vwap_ref = of.get("vwap")
                            
                            twap_result = twap_vwap_analysis(closes, volumes, vwap_ref)
                            extras["twap_analysis"] = twap_result
                        except Exception as e:
                            extras["twap_analysis"] = {"error": str(e)}

        except Exception as e:
            extras["_error"] = str(e)

        return extras

    # ═══════════════════════════════════════════════════════════
    # SEÇÃO 2: VOLUME PROFILE AVANÇADO
    # ═══════════════════════════════════════════════════════════
    def _compute_profile_analysis(
        self, trades_df, vp_data, current_price, weekly_vp, monthly_vp
    ) -> dict:
        profile = {}

        # #7 — Poor High/Low e #11 — Profile Shape
        if trades_df is not None and len(trades_df) > 20:
            try:
                from dynamic_volume_profile import DynamicVolumeProfile
                dvp = DynamicVolumeProfile(self.symbol)

                # #7
                try:
                    profile["poor_extremes"] = dvp.detect_poor_extremes(trades_df, vp_data)
                except Exception as e:
                    profile["poor_extremes"] = {"status": "error", "error": str(e)}

                # #11
                try:
                    profile["profile_shape"] = dvp.classify_profile_shape(trades_df, vp_data)
                except Exception as e:
                    profile["profile_shape"] = {"status": "error", "error": str(e)}

            except Exception as e:
                profile["_dvp_error"] = str(e)

        # #6, #12, #13 — Precisam do VolumeProfileAnalyzer
        if vp_data and isinstance(vp_data, dict):
            try:
                from support_resistance.volume_profile import VolumeProfileAnalyzer

                # Construir profile dict no formato esperado
                vp_profile = {
                    "poc": {"price": vp_data.get("poc", 0), "volume": 0, "percent_of_total": 0},
                    "value_area": {
                        "low": vp_data.get("val", 0),
                        "high": vp_data.get("vah", 0),
                    },
                    "volume_nodes": {
                        "hvn": vp_data.get("hvns", []),
                        "lvn": vp_data.get("lvns", []),
                        "hvn_levels": vp_data.get("hvn_levels", []),
                    },
                    "current_position": {"price": current_price},
                }

                # #6 — No-Man's Land
                try:
                    # Criar instância com dados mínimos se possível
                    # VolumeProfileAnalyzer precisa de price_data e volume_data
                    # Mas podemos usar o método com profile direto
                    # Instanciar com dados dummy e usar profile pré-calculado
                    dummy_prices = pd.Series([current_price])
                    dummy_vols = pd.Series([1.0])
                    vpa = VolumeProfileAnalyzer(dummy_prices, dummy_vols)
                    profile["no_mans_land"] = vpa.detect_no_mans_land(
                        vp_profile, current_price
                    )
                except Exception as e:
                    profile["no_mans_land"] = {"status": "error", "error": str(e)}

                # #12 — Value Area Volume %
                try:
                    if not hasattr(self, '_vpa_instance'):
                        dummy_prices = pd.Series([current_price])
                        dummy_vols = pd.Series([1.0])
                        vpa = VolumeProfileAnalyzer(dummy_prices, dummy_vols)
                    profile["va_volume_pct"] = vpa.calculate_value_area_volume_pct(vp_profile)
                except Exception as e:
                    profile["va_volume_pct"] = {"status": "error", "error": str(e)}

                # #13 — HVN/LVN Strength
                try:
                    weekly_nodes = None
                    monthly_nodes = None
                    if weekly_vp and isinstance(weekly_vp, dict):
                        weekly_nodes = {
                            "hvn": weekly_vp.get("hvns", []),
                            "lvn": weekly_vp.get("lvns", []),
                        }
                    if monthly_vp and isinstance(monthly_vp, dict):
                        monthly_nodes = {
                            "hvn": monthly_vp.get("hvns", []),
                            "lvn": monthly_vp.get("lvns", []),
                        }
                    profile["volume_node_strength"] = vpa.score_volume_nodes(
                        vp_profile, current_price, weekly_nodes, monthly_nodes
                    )
                except Exception as e:
                    profile["volume_node_strength"] = {"status": "error", "error": str(e)}

            except Exception as e:
                profile["_vpa_error"] = str(e)

        return profile

    # ═══════════════════════════════════════════════════════════
    # SEÇÃO 3: FLOW AVANÇADO
    # ═══════════════════════════════════════════════════════════
    def _compute_flow_analysis(
        self, flow_metrics, orderbook_data, absorption_data,
        derivatives_data, onchain_data
    ) -> dict:
        flow = {}

        if not flow_metrics or not isinstance(flow_metrics, dict):
            return flow

        order_flow = flow_metrics.get("order_flow", {})
        sector_flow = flow_metrics.get("sector_flow", {})

        # #14 — Passive/Aggressive Flow
        try:
            from flow_analyzer.aggregates import analyze_passive_aggressive_flow
            flow_input = {
                "aggressive_buy_pct": order_flow.get("aggressive_buy_pct", 50),
                "aggressive_sell_pct": order_flow.get("aggressive_sell_pct", 50),
                "buy_volume_btc": order_flow.get("buy_volume_btc", 0),
                "sell_volume_btc": order_flow.get("sell_volume_btc", 0),
                "flow_imbalance": order_flow.get("flow_imbalance", 0),
            }
            flow["passive_aggressive"] = analyze_passive_aggressive_flow(
                flow_input, orderbook_data
            )
        except Exception as e:
            flow["passive_aggressive"] = {"status": "error", "error": str(e)}

        # #15 — Buy/Sell Ratio
        try:
            from flow_analyzer.aggregates import calculate_buy_sell_ratios
            ratio_input = {
                "buy_volume_btc": order_flow.get("buy_volume_btc", 0),
                "sell_volume_btc": order_flow.get("sell_volume_btc", 0),
                "net_flow_1m": order_flow.get("net_flow_1m", 0),
                "net_flow_5m": order_flow.get("net_flow_5m", 0),
                "net_flow_15m": order_flow.get("net_flow_15m", 0),
                "total_volume": order_flow.get("total_volume", 0),
                "sector_flow": sector_flow,
            }
            flow["buy_sell_ratio"] = calculate_buy_sell_ratios(ratio_input)
        except Exception as e:
            flow["buy_sell_ratio"] = {"status": "error", "error": str(e)}

        # #16 — Whale Accumulation Score
        if self.whale_calculator:
            try:
                cvd = flow_metrics.get("cvd", 0)
                flow["whale_accumulation"] = self.whale_calculator.calculate(
                    sector_flow=sector_flow,
                    orderbook_data=orderbook_data,
                    absorption_data=absorption_data,
                    derivatives_data=derivatives_data,
                    onchain_data=onchain_data,
                    cvd=cvd,
                )
            except Exception as e:
                flow["whale_accumulation"] = {"status": "error", "error": str(e)}

        # #17 — Absorption Zones
        if self.absorption_mapper:
            try:
                # Registrar evento de absorção atual (se houver)
                if absorption_data and isinstance(absorption_data, dict):
                    abs_inner = absorption_data.get("current_absorption", absorption_data)
                    if isinstance(abs_inner, dict):
                        classification = abs_inner.get("label", abs_inner.get("classification", ""))
                        index = abs_inner.get("index", 0)
                        if index > 0.1 and "Neutra" not in str(classification):
                            # Precisamos do preço — extrair do flow_metrics
                            price = 0
                            if order_flow:
                                price = order_flow.get("last_price", 0)
                            if price <= 0 and flow_metrics:
                                price = flow_metrics.get("last_price", 0)
                            
                            if price > 0:
                                self.absorption_mapper.record_event(
                                    price=price,
                                    classification=str(classification),
                                    index=float(index),
                                    buyer_strength=float(abs_inner.get("buyer_strength", 0)),
                                    seller_exhaustion=float(abs_inner.get("seller_exhaustion", 0)),
                                    volume_usd=float(abs_inner.get("total_volume_usd", 0)),
                                )

                flow["absorption_zones"] = self.absorption_mapper.get_zones(
                    current_price=order_flow.get("last_price", 0) if order_flow else 0
                )
            except Exception as e:
                flow["absorption_zones"] = {"status": "error", "error": str(e)}

        return flow

    # ═══════════════════════════════════════════════════════════
    # SEÇÃO 4: S/R AVANÇADO
    # ═══════════════════════════════════════════════════════════
    def _compute_sr_analysis(
        self, current_price, vp_data, pivot_data, ema_values,
        candles_df, weekly_vp, monthly_vp, orderbook_data,
        absorption_data
    ) -> dict:
        sr = {}

        # #2 — Reference Prices
        if self.reference_prices:
            try:
                sr["reference_prices"] = self.reference_prices.get_context(
                    self.symbol, current_price
                )
            except Exception as e:
                sr["reference_prices"] = {"status": "error", "error": str(e)}

        # #4 — S/R Strength
        sr_scorer = self._modules.get("sr_strength")
        if sr_scorer and current_price > 0:
            try:
                sr["sr_strength"] = sr_scorer.score_levels(
                    current_price=current_price,
                    vp_data=vp_data,
                    pivot_data=pivot_data,
                    ema_values=ema_values,
                    recent_candles=candles_df,
                    weekly_vp=weekly_vp,
                    monthly_vp=monthly_vp,
                )
            except Exception as e:
                sr["sr_strength"] = {"status": "error", "error": str(e)}

        # #5 — Defense Zones
        defense_detector = self._modules.get("defense_zones")
        if defense_detector and current_price > 0:
            try:
                # Preparar absorption events para defense zones
                abs_events = []
                if self.absorption_mapper:
                    zones = self.absorption_mapper.get_zones(current_price)
                    if zones.get("zones"):
                        for z in zones["zones"]:
                            abs_events.append({
                                "price": z["center"],
                                "type": f"Absorção de {'Compra' if z['dominant_side'] == 'buy_defense' else 'Venda'}",
                                "strength": z["avg_strength"],
                            })

                # Usar sr_levels do scoring se disponível
                sr_levels = []
                if "sr_strength" in sr and sr["sr_strength"].get("levels"):
                    sr_levels = sr["sr_strength"]["levels"]

                sr["defense_zones"] = defense_detector.detect(
                    current_price=current_price,
                    orderbook_data=orderbook_data,
                    vp_data=vp_data,
                    sr_levels=sr_levels,
                    absorption_events=abs_events,
                    pivot_data=pivot_data,
                    ema_values=ema_values,
                )
            except Exception as e:
                sr["defense_zones"] = {"status": "error", "error": str(e)}

        return sr

    # ═══════════════════════════════════════════════════════════
    # SEÇÃO 5: QUALIDADE E INFRAESTRUTURA
    # ═══════════════════════════════════════════════════════════
    def _compute_quality(
        self, current_price, orderbook_data, flow_metrics,
        window_close_ms, time_manager
    ) -> dict:
        quality = {}

        # #22 — Holiday Calendar
        if time_manager and hasattr(time_manager, 'get_market_calendar_context'):
            try:
                quality["calendar"] = time_manager.get_market_calendar_context()
            except Exception as e:
                quality["calendar"] = {"error": str(e)}

        # #21 — Latency Tracking
        if time_manager and window_close_ms and hasattr(time_manager, 'track_data_latency'):
            try:
                quality["latency"] = time_manager.track_data_latency(window_close_ms)
            except Exception as e:
                quality["latency"] = {"error": str(e)}

        # #20 — Spread Percentile
        if self.spread_tracker and orderbook_data:
            try:
                spread = orderbook_data.get("spread", 0)
                if spread and spread > 0:
                    self.spread_tracker.update(spread)
                    quality["spread_percentile"] = self.spread_tracker.get_metrics(spread)
            except Exception as e:
                quality["spread_percentile"] = {"error": str(e)}

        # #19 — Anomaly Detection
        try:
            from data_quality_validator import DataQualityValidator
            dqv = DataQualityValidator()
            
            anomaly_data = {}
            if flow_metrics and isinstance(flow_metrics, dict):
                of = flow_metrics.get("order_flow", {})
                anomaly_data.update({
                    "volume_total": of.get("total_volume_btc", 0),
                    "flow_imbalance": of.get("flow_imbalance", 0),
                    "trades_per_second": flow_metrics.get("trades_per_second", 0),
                })
            if orderbook_data and isinstance(orderbook_data, dict):
                anomaly_data.update({
                    "spread": orderbook_data.get("spread", 0),
                    "bid_depth_usd": orderbook_data.get("bid_depth_usd", 0),
                    "ask_depth_usd": orderbook_data.get("ask_depth_usd", 0),
                })
            anomaly_data["close"] = current_price

            quality["anomalies"] = dqv.detect_anomalies(anomaly_data)
        except Exception as e:
            quality["anomalies"] = {"error": str(e)}

        return quality

    # ═══════════════════════════════════════════════════════════
    # SEÇÃO 6: CANDLESTICK PATTERNS
    # ═══════════════════════════════════════════════════════════
    def _compute_candlestick(self, candles_df) -> dict:
        if candles_df is None or len(candles_df) < 3:
            return {"patterns_detected": 0, "patterns": []}

        try:
            from pattern_recognition import detect_candlestick_patterns
            return detect_candlestick_patterns(candles_df)
        except Exception as e:
            return {"patterns_detected": 0, "patterns": [], "error": str(e)}

    # ═══════════════════════════════════════════════════════════
    # MÉTODOS AUXILIARES
    # ═══════════════════════════════════════════════════════════
    async def update_reference_prices(self, client) -> bool:
        """Atualiza preços de referência via API Binance."""
        if self.reference_prices:
            try:
                return await self.reference_prices.update(self.symbol, client)
            except Exception as e:
                logger.warning(f"Failed to update reference prices: {e}")
        return False

    def update_reference_prices_from_candles(self, candles: dict) -> None:
        """Atualiza preços de referência de candles já disponíveis."""
        if self.reference_prices:
            try:
                self.reference_prices.update_from_candles(self.symbol, candles)
            except Exception as e:
                logger.warning(f"Failed to update reference prices from candles: {e}")

    def record_absorption_event(
        self, price: float, classification: str, index: float, **kwargs
    ) -> None:
        """Registra evento de absorção no mapper."""
        if self.absorption_mapper and price > 0 and index > 0.05:
            try:
                self.absorption_mapper.record_event(
                    price=price,
                    classification=classification,
                    index=index,
                    **kwargs,
                )
            except Exception as e:
                logger.debug(f"Failed to record absorption: {e}")

    def update_spread(self, spread: float) -> None:
        """Atualiza spread no tracker."""
        if self.spread_tracker and spread > 0:
            try:
                self.spread_tracker.update(spread)
            except Exception:
                pass

    def get_status(self) -> dict:
        """Retorna status do engine para health check."""
        return {
            "initialized": self._initialized,
            "init_errors": self._init_errors,
            "spread_tracker": self.spread_tracker is not None,
            "absorption_mapper": self.absorption_mapper is not None,
            "whale_calculator": self.whale_calculator is not None,
            "reference_prices": self.reference_prices is not None,
            "sr_strength": "sr_strength" in self._modules,
            "defense_zones": "defense_zones" in self._modules,
        }
