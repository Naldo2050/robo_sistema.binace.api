"""
Bridge entre eventos do sistema e módulos institucionais.
Extrai dados do evento e alimenta cada analisador.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from institutional.base import (
    AnalysisResult,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    Trade,
)
from institutional.cvd import CVDAnalyzer
from institutional.footprint import FootprintAnalyzer
from institutional.order_flow_imbalance import OrderFlowImbalanceAnalyzer
from institutional.absorption_detector import AbsorptionDetector
from institutional.iceberg_detector import IcebergDetector
from institutional.vwap_twap import VWAPTWAPAnalyzer
from institutional.kalman_filter import KalmanTrendFilter
from institutional.hurst_exponent import HurstCalculator
from institutional.smart_money import SmartMoneyAnalyzer, Candle
from institutional.garch_volatility import GARCHModel
from institutional.market_regime_hmm import MarketRegimeHMM
from institutional.whale_detector import WhaleDetector
from institutional.crypto_cot import CryptoCOT
from institutional.entropy_analyzer import EntropyAnalyzer
from institutional.fourier_cycles import FourierCycleAnalyzer
from institutional.mean_reversion import MeanReversionAnalyzer
from institutional.monte_carlo import MonteCarloSimulator
from institutional.confluence_engine import ConfluenceEngine


class InstitutionalEventBridge:
    """
    Conecta eventos do sistema existente aos módulos institucionais.

    Uso:
        bridge = InstitutionalEventBridge()

        # Para cada evento do sistema:
        result = bridge.process_event(event_dict)

        # result contém análise de confluência de TODOS os módulos
    """

    def __init__(
        self,
        whale_threshold_usd: float = 100_000.0,
        kalman_process_noise: float = 0.01,
        kalman_measurement_noise: float = 5.0,
        hurst_min_samples: int = 50,
        garch_alpha: float = 0.10,
        garch_beta: float = 0.85,
        monte_carlo_simulations: int = 1000,
    ):
        # Inicializar todos os analisadores
        self.cvd = CVDAnalyzer(bar_interval_seconds=60)
        self.footprint = FootprintAnalyzer(tick_size=1.0, bar_interval_seconds=60)
        self.ofi = OrderFlowImbalanceAnalyzer(window_seconds=30)
        self.absorption = AbsorptionDetector(window_seconds=30)
        self.iceberg = IcebergDetector()
        self.vwap_twap = VWAPTWAPAnalyzer()
        self.kalman = KalmanTrendFilter(
            process_noise=kalman_process_noise,
            measurement_noise=kalman_measurement_noise,
        )
        self.hurst = HurstCalculator(min_samples=hurst_min_samples)
        self.smart_money = SmartMoneyAnalyzer(swing_lookback=5)
        self.garch = GARCHModel(alpha=garch_alpha, beta=garch_beta)
        self.hmm = MarketRegimeHMM(return_window=20)
        self.whale = WhaleDetector(whale_threshold_usd=whale_threshold_usd)
        self.cot = CryptoCOT()
        self.entropy = EntropyAnalyzer(n_bins=20, window_size=100)
        self.fourier = FourierCycleAnalyzer(window_size=128)
        self.mean_rev = MeanReversionAnalyzer(lookback=20)
        self.monte_carlo = MonteCarloSimulator(
            n_simulations=monte_carlo_simulations,
        )
        self.confluence = ConfluenceEngine()

        self._events_processed: int = 0

    @property
    def events_processed(self) -> int:
        return self._events_processed

    def process_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """
        Processa um evento completo do sistema e retorna análise institucional.

        Args:
            event: Dicionário do evento (como o JSON mostrado)

        Returns:
            Dicionário com análise de confluência e métricas de cada módulo
        """
        self._events_processed += 1
        timestamp = event.get("epoch_ms", time.time() * 1000) / 1000.0
        results: list[AnalysisResult] = []

        # --- 1. Extrair preço ---
        price = self._extract_price(event)
        if price <= 0:
            return {"error": "no_price", "events_processed": self._events_processed}

        # --- 2. Atualizar módulos baseados em preço ---
        self.kalman.update(timestamp, price)
        self.hurst.add_price(price)
        self.garch.update(timestamp, price)
        self.hmm.update(timestamp, price)
        self.entropy.add_price(timestamp, price)
        self.fourier.add_price(price)
        self.mean_rev.add_price(timestamp, price)
        self.monte_carlo.add_price(price)

        # --- 3. Atualizar VWAP/TWAP com candle ---
        ohlc = event.get("ohlc", {})
        if ohlc:
            self.vwap_twap.add_candle(
                timestamp,
                high=ohlc.get("high", price),
                low=ohlc.get("low", price),
                close=ohlc.get("close", price),
                volume=event.get("volume_total_btc", event.get("volume_total", 0)),
            )

        # --- 4. Atualizar Smart Money com candle ---
        if ohlc:
            candle = Candle(
                timestamp=timestamp,
                open=ohlc.get("open", price),
                high=ohlc.get("high", price),
                low=ohlc.get("low", price),
                close=ohlc.get("close", price),
                volume=event.get("volume_total_btc", 0),
            )
            self.smart_money.add_candle(candle)

        # --- 5. Atualizar Crypto COT ---
        derivatives = event.get("derivatives", {}).get("BTCUSDT", {})
        if derivatives:
            self.cot.add_data(
                timestamp=timestamp,
                funding_rate=derivatives.get("funding_rate_percent", 0) / 100,
                open_interest=derivatives.get("open_interest", 0),
                long_short_ratio=derivatives.get("long_short_ratio", 1.0),
                top_trader_ls_ratio=1.0,  # Não disponível no evento
                price=price,
            )

        # --- 6. Atualizar Whale Detector com trade resumo ---
        buy_notional = event.get("buy_notional_usdt", 0)
        sell_notional = event.get("sell_notional_usdt", 0)

        if buy_notional > 0:
            self.whale.process_trade(Trade(
                timestamp=timestamp,
                price=price,
                quantity=event.get("volume_compra_btc", event.get("volume_compra", 0)),
                side=Side.BUY,
                value_usd=buy_notional,
            ))
        if sell_notional > 0:
            self.whale.process_trade(Trade(
                timestamp=timestamp,
                price=price,
                quantity=event.get("volume_venda_btc", event.get("volume_venda", 0)),
                side=Side.SELL,
                value_usd=sell_notional,
            ))

        # --- 7. Atualizar Orderbook para Iceberg ---
        ob_data = event.get("orderbook_data", {})
        if ob_data.get("is_valid"):
            bid_price = ob_data.get("bid", ob_data.get("mid", price) - 0.05)
            ask_price = ob_data.get("ask", ob_data.get("mid", price) + 0.05)

            snapshot = OrderBookSnapshot(
                timestamp=timestamp,
                bids=[OrderBookLevel(bid_price, ob_data.get("bid_depth_usd", 0) / max(bid_price, 1), Side.BUY)],
                asks=[OrderBookLevel(ask_price, ob_data.get("ask_depth_usd", 0) / max(ask_price, 1), Side.SELL)],
            )
            self.iceberg.process_snapshot(snapshot)

        # --- 8. Coletar análises de todos os módulos ---
        results.append(self.kalman.analyze(price))
        results.append(self.hurst.analyze())
        results.append(self.garch.analyze())
        results.append(self.hmm.analyze())
        results.append(self.smart_money.analyze())
        results.append(self.vwap_twap.analyze(price))
        results.append(self.whale.analyze())
        results.append(self.cot.analyze())
        results.append(self.entropy.analyze())
        results.append(self.fourier.analyze())
        results.append(self.mean_rev.analyze())
        results.append(self.monte_carlo.analyze())
        results.append(self.iceberg.analyze())

        # --- 9. Confluência ---
        self.confluence.reset()
        self.confluence.add_results(results)
        confluence_result = self.confluence.analyze()
        confluence_score = self.confluence.calculate_confluence()
        layer_summary = self.confluence.get_layer_summary()

        # --- 10. Montar resposta ---
        return {
            "timestamp": timestamp,
            "price": price,
            "events_processed": self._events_processed,
            "confluence": {
                "direction": confluence_score.direction.value,
                "score": confluence_score.total_score,
                "strength": confluence_score.strength.value,
                "confidence": confluence_score.confidence,
                "buy_signals": confluence_score.buy_signals,
                "sell_signals": confluence_score.sell_signals,
                "sources_agreeing": confluence_score.sources_agreeing,
                "total_sources": confluence_score.total_sources,
                "summary": confluence_score.summary,
            },
            "layers": layer_summary,
            "modules": {
                r.source: {
                    "regime": r.regime.value,
                    "confidence": r.confidence,
                    "signals_count": len(r.signals),
                    "metrics": r.metrics,
                }
                for r in results
                if r.source
            },
            "all_signals": [
                s.to_dict()
                for r in results
                for s in r.signals
            ],
        }

    def _extract_price(self, event: dict[str, Any]) -> float:
        """Extrai preço do evento."""
        # Tentar múltiplos campos
        price = event.get("preco_fechamento", 0)
        if price <= 0:
            ohlc = event.get("ohlc", {})
            price = ohlc.get("close", 0)
        if price <= 0:
            price = event.get("contextual_snapshot", {}).get("ohlc", {}).get("close", 0)
        if price <= 0:
            price = event.get("tick_context_out", {}).get("last_price", 0)
        return float(price)

    def get_status(self) -> dict:
        """Status de todos os módulos."""
        return {
            "events_processed": self._events_processed,
            "kalman_initialized": self.kalman.is_initialized,
            "hurst_samples": self.hurst.sample_count,
            "garch_initialized": self.garch.is_initialized,
            "hmm_history": len(self.hmm.history),
            "smart_money_candles": self.smart_money.candle_count,
            "vwap_data_points": self.vwap_twap.vwap.data_points,
            "whale_events": len(self.whale.events),
            "cot_data_points": self.cot.data_points,
            "entropy_current": self.entropy.current_entropy,
            "fourier_data_points": self.fourier.data_points,
            "mean_rev_data_points": self.mean_rev.data_points,
            "monte_carlo_data_points": self.monte_carlo.data_points,
            "iceberg_candidates": len(self.iceberg.candidates),
        }

    def reset(self) -> None:
        """Reseta todos os módulos."""
        self.cvd.reset()
        self.footprint.reset()
        self.ofi.reset()
        self.absorption.reset()
        self.iceberg.reset()
        self.vwap_twap.reset()
        self.kalman.reset()
        self.hurst.reset()
        self.smart_money.reset()
        self.garch.reset()
        self.hmm.reset()
        self.whale.reset()
        self.cot.reset()
        self.entropy.reset()
        self.fourier.reset()
        self.mean_rev.reset()
        self.monte_carlo.reset()
        self.confluence.reset()
        self._events_processed = 0
