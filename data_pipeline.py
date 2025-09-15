import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

class DataPipeline:
    def __init__(self, raw_trades: List[Dict], symbol: str):
        """
        Pipeline de dados em camadas para transformar trades crus em eventos prontos para IA.
        Camadas:
          1. Raw → dados crus da Binance
          2. Enriched → métricas básicas (delta, volume, speed)
          3. Contextual → adiciona fluxo, VP, MTF, ordem book
          4. Signal → detecta Absorção/Exaustão/OrderBook/Zonas
        """
        self.raw_trades = raw_trades
        self.symbol = symbol
        self.df = None
        self.enriched_data = None
        self.contextual_data = None
        self.signal_data = None
        self._cache = {}  # 🔹 Cache para cálculos caros (Fase 2)
        self._validate_and_load()

    def _validate_and_load(self):
        """Valida e carrega trades crus em DataFrame."""
        if not self.raw_trades or len(self.raw_trades) < 2:
            raise ValueError("Dados insuficientes para pipeline.")
        try:
            self.df = pd.DataFrame(self.raw_trades)
            self.df["p"] = pd.to_numeric(self.df["p"], errors="coerce")
            self.df["q"] = pd.to_numeric(self.df["q"], errors="coerce")
            self.df["T"] = pd.to_numeric(self.df["T"], errors="coerce")
            self.df = self.df.dropna(subset=["p", "q", "T"])
            if self.df.empty:
                raise ValueError("DataFrame vazio após limpeza.")
        except Exception as e:
            logging.error(f"Erro ao carregar dados crus: {e}")
            raise

    def _get_cached(self, key: str, compute_fn):
        """Retorna valor do cache ou calcula e armazena."""
        if key in self._cache:
            return self._cache[key]
        result = compute_fn()
        self._cache[key] = result
        return result

    def enrich(self) -> Dict[str, Any]:
        """Camada 2: Adiciona métricas básicas intra-candle."""
        from data_handler import (
            calcular_metricas_intra_candle,
            calcular_volume_profile,
            calcular_dwell_time,
            calcular_trade_speed
        )

        try:
            enriched = {
                "symbol": self.symbol,
                "ohlc": {
                    "open": float(self.df["p"].iloc[0]),
                    "high": float(self.df["p"].max()),
                    "low": float(self.df["p"].min()),
                    "close": float(self.df["p"].iloc[-1]),
                },
                "volume_total": float(self.df["q"].sum()),
                "num_trades": len(self.df),
            }

            # 🔹 OTIMIZADO: usa cache para evitar recálculo
            enriched.update(self._get_cached("metricas_intra", lambda: calcular_metricas_intra_candle(self.df)))
            enriched.update(self._get_cached("volume_profile", lambda: calcular_volume_profile(self.df)))
            enriched.update(self._get_cached("dwell_time", lambda: calcular_dwell_time(self.df)))
            enriched.update(self._get_cached("trade_speed", lambda: calcular_trade_speed(self.df)))

            self.enriched_data = enriched
            logging.debug("✅ Camada Enriched gerada.")
            return enriched
        except Exception as e:
            logging.error(f"Erro na camada Enriched: {e}")
            raise

    def add_context(self, flow_metrics: Dict = None, historical_vp: Dict = None, orderbook_data: Dict = None, multi_tf: Dict = None) -> Dict[str, Any]:
        """Camada 3: Enriquece com contexto externo."""
        if self.enriched_data is None:
            self.enrich()

        contextual = self.enriched_data.copy()
        contextual.update({
            "flow_metrics": flow_metrics or {},
            "historical_vp": historical_vp or {},
            "orderbook_data": orderbook_data or {},  # ✅ CORRIGIDO: parâmetro agora existe
            "multi_tf": multi_tf or {},
        })

        self.contextual_data = contextual
        logging.debug("✅ Camada Contextual gerada.")
        return contextual

    def detect_signals(self, absorption_detector, exhaustion_detector, orderbook_data=None) -> List[Dict]:
        """Camada 4: Detecta sinais usando os detectores fornecidos."""
        if self.contextual_data is None:
            raise ValueError("Camada Contextual deve ser gerada antes.")

        signals = []

        # Absorção
        try:
            absorption_event = absorption_detector(self.raw_trades, self.symbol)
            if absorption_event.get("is_signal", False):
                absorption_event["layer"] = "signal"
                signals.append(absorption_event)
        except Exception as e:
            logging.error(f"Erro detectando absorção: {e}")

        # Exaustão
        try:
            exhaustion_event = exhaustion_detector(self.raw_trades, self.symbol)
            if exhaustion_event.get("is_signal", False):
                exhaustion_event["layer"] = "signal"
                signals.append(exhaustion_event)
        except Exception as e:
            logging.error(f"Erro detectando exaustão: {e}")

        # OrderBook (opcional)
        if orderbook_data and orderbook_data.get("is_signal", False):
            try:
                ob_event = orderbook_data  # já é um evento
                ob_event["layer"] = "signal"
                signals.append(ob_event)
            except Exception as e:
                logging.error(f"Erro adicionando evento OrderBook: {e}")

        self.signal_data = signals
        logging.debug(f"✅ Camada Signal gerada. {len(signals)} sinais detectados.")
        return signals

    def get_final_features(self) -> Dict[str, Any]:
        """Retorna todas as features consolidadas (útil para Feature Store ou IA)."""
        if self.signal_data is None:
            raise ValueError("Camada Signal deve ser gerada antes.")

        features = {
            "symbol": self.symbol,
            "timestamp": self.contextual_data.get("ohlc", {}).get("close_time_iso", ""),
            "enriched": self.enriched_data,
            "contextual": self.contextual_data,
            "signals": self.signal_data,
        }
        return features