# data_pipeline.py
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from time_manager import TimeManager

# Importador opcional do gerador de features de ML
try:
    from ml_features import generate_ml_features
except Exception:
    generate_ml_features = None


class DataPipeline:
    def __init__(self, raw_trades: List[Dict], symbol: str, time_manager: Optional[TimeManager] = None):
        """
        Pipeline de dados em camadas para transformar trades crus em eventos prontos para IA.
        Camadas:
          1. Raw → dados crus da Binance
          2. Enriched → métricas básicas (OHLC, VWAP, volumes, speed, dwell, delta intra)
          3. Contextual → adiciona fluxo, VP histórico, MTF, order book, derivativos, macro/ambiente
          4. Signal → detecta Absorção/Exaustão/OrderBook/Zonas
        """
        self.raw_trades = raw_trades
        self.symbol = symbol
        self.df: pd.DataFrame | None = None
        self.enriched_data: Dict[str, Any] | None = None
        self.contextual_data: Dict[str, Any] | None = None
        self.signal_data: List[Dict[str, Any]] | None = None
        self._cache: Dict[str, Any] = {}  # Cache para cálculos caros

        # Time manager (para timestamps consistentes)
        self.tm: TimeManager = time_manager or TimeManager()

        self._validate_and_load()

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
        """Valida, normaliza e carrega trades crus em DataFrame, ordenado por tempo."""
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

                # m: mantém como vier; se ausente, deixa NaN (tick-rule pode inferir depois)
                m = t.get("m", np.nan)

                if p is None or q is None or T is None:
                    continue
                if p <= 0 or q <= 0 or T <= 0:
                    continue

                validated.append({"p": p, "q": q, "T": T, "m": m})

            if not validated:
                raise ValueError("Nenhum trade válido após validação.")

            df = pd.DataFrame(validated)
            # remove nulos residuais e ordena por tempo
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
            return self._cache[key]
        result = compute_fn()
        self._cache[key] = result
        return result

    def _parse_iso_ms(self, iso_str: str) -> Optional[int]:
        """Tenta converter ISO 8601 para epoch_ms; retorna None em caso de falha."""
        try:
            # fromisoformat aceita 'YYYY-MM-DDTHH:MM:SS+00:00'
            dt = datetime.fromisoformat(iso_str)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    def _sanitize_event(self, ev: Dict[str, Any], default_ts_ms: Optional[int] = None) -> Dict[str, Any]:
        """
        Garante timestamps consistentes no evento:
        - Resolve epoch_ms a partir do próprio evento, do timestamp_utc ou usa default_ts_ms.
        - Injeta epoch_ms, timestamp_utc, timestamp_ny e timestamp_sp (overwrite=True).
        """
        if not isinstance(ev, dict):
            return ev

        e: Dict[str, Any] = dict(ev)  # cópia rasa

        ts_ms: Optional[int] = None
        # 1) Se o próprio evento já trouxe epoch_ms válido, respeite
        try:
            if "epoch_ms" in e and isinstance(e["epoch_ms"], (int, float)) and int(e["epoch_ms"]) > 0:
                ts_ms = int(e["epoch_ms"])
        except Exception:
            ts_ms = None

        # 2) Se há timestamp_utc em ISO, derivar
        if ts_ms is None and isinstance(e.get("timestamp_utc"), str) and e["timestamp_utc"]:
            ts_ms = self._parse_iso_ms(e["timestamp_utc"])

        # 3) Fallback: use default_ts_ms (ex.: close_time da janela)
        if ts_ms is None:
            if default_ts_ms is None:
                try:
                    default_ts_ms = int(self.enriched_data.get("ohlc", {}).get("close_time", 0)) if self.enriched_data else None
                except Exception:
                    default_ts_ms = None
            ts_ms = default_ts_ms or self.tm.now_ms()

        # Injeta timestamps padronizados e únicos
        self.tm.attach_timestamps(e, ts_ms=ts_ms, include_local=True, overwrite=True, timespec="seconds")

        return e

    # ===============================
    # Camada 2 — Enriched
    # ===============================

    def enrich(self) -> Dict[str, Any]:
        """Adiciona OHLC, VWAP, volumes (base e quote) e métricas intra-candle."""
        try:
            # Importação local para evitar dependências circulares
            from data_handler import (
                calcular_metricas_intra_candle,
                calcular_volume_profile,
                calcular_dwell_time,
                calcular_trade_speed,
            )

            df = self.df
            # garantias
            if df is None or df.empty:
                raise ValueError("DF não carregado na pipeline.")

            open_price = float(df["p"].iloc[0])
            close_price = float(df["p"].iloc[-1])
            high_price = float(df["p"].max())
            low_price = float(df["p"].min())

            open_time = int(df["T"].iloc[0])
            close_time = int(df["T"].iloc[-1])

            # ISO 8601 com timezone correto via TimeManager
            open_iso = self.tm.format_timestamp(open_time, tz=self.tm.tz_utc, timespec="seconds")
            close_iso = self.tm.format_timestamp(close_time, tz=self.tm.tz_utc, timespec="seconds")

            base_volume = float(df["q"].sum())               # ex.: BTC
            quote_volume = float((df["p"] * df["q"]).sum())  # ex.: USDT
            vwap = float(quote_volume / base_volume) if base_volume > 0 else close_price

            enriched = {
                "symbol": self.symbol,
                "ohlc": {
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "open_time": open_time,
                    "close_time": close_time,
                    "open_time_iso": open_iso,
                    "close_time_iso": close_iso,
                    "vwap": vwap,
                },
                "volume_total": base_volume,             # volume na moeda base (ex.: BTC)
                "volume_total_usdt": quote_volume,       # volume na moeda de cotação (ex.: USDT)
                "num_trades": int(len(df)),
            }

            # 🔹 Cache de métricas caras
            enriched.update(self._get_cached("metricas_intra", lambda: calcular_metricas_intra_candle(df)))
            enriched.update(self._get_cached("volume_profile", lambda: calcular_volume_profile(df)))
            enriched.update(self._get_cached("dwell_time", lambda: calcular_dwell_time(df)))
            enriched.update(self._get_cached("trade_speed", lambda: calcular_trade_speed(df)))

            self.enriched_data = enriched
            logging.debug("✅ Camada Enriched gerada.")
            return enriched

        except Exception as e:
            logging.error(f"Erro na camada Enriched: {e}")
            # Fallback mínimo
            return {
                "symbol": self.symbol,
                "ohlc": {
                    "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0,
                    "open_time": 0, "close_time": 0,
                    "open_time_iso": "", "close_time_iso": "", "vwap": 0.0
                },
                "volume_total": 0.0,
                "volume_total_usdt": 0.0,
                "num_trades": 0,
                "delta_minimo": 0.0,
                "delta_maximo": 0.0,
                "delta_fechamento": 0.0,
                "reversao_desde_minimo": 0.0,
                "reversao_desde_maximo": 0.0,
                "dwell_price": 0.0,
                "dwell_seconds": 0.0,
                "dwell_location": "N/A",
                "trades_per_second": 0.0,
                "avg_trade_size": 0.0,
                "poc_price": 0.0,
                "poc_volume": 0.0,
                "poc_percentage": 0.0,
            }

    # ===============================
    # Camada 3 — Contextual
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
        """
        Enriquece com contexto externo (fluxo contínuo, VP histórico, MTF, order book, derivativos,
        contexto de mercado e ambiente). Adiciona todas as fontes ao dicionário contextual.
        """
        if self.enriched_data is None:
            self.enrich()

        contextual = dict(self.enriched_data)  # cópia rasa
        contextual.update({
            "flow_metrics": flow_metrics or {},
            "historical_vp": historical_vp or {},
            "orderbook_data": orderbook_data or {},  # compatível com main.py
            "multi_tf": multi_tf or {},
            "derivatives": derivatives or {},
            "market_context": market_context or {},
            "market_environment": market_environment or {},
        })

        self.contextual_data = contextual
        logging.debug("✅ Camada Contextual gerada.")
        return contextual

    # ===============================
    # Camada 4 — Signal
    # ===============================

    def detect_signals(self, absorption_detector, exhaustion_detector, orderbook_data=None) -> List[Dict]:
        """
        Detecta sinais usando os detectores fornecidos (funções/lambdas).
        Espera-se que o chamador injete quaisquer parâmetros extras via closures (vide main.py).
        """
        if self.contextual_data is None:
            raise ValueError("Camada Contextual deve ser gerada antes.")

        signals: List[Dict[str, Any]] = []

        # Timestamp de referência para esta janela (final do cluster de trades)
        try:
            default_ts_ms = int(self.enriched_data.get("ohlc", {}).get("close_time", 0)) if self.enriched_data else None
        except Exception:
            default_ts_ms = None

        # Absorção
        if absorption_detector:
            try:
                absorption_event = absorption_detector(self.raw_trades, self.symbol)
                if absorption_event:
                    absorption_event["layer"] = "signal"
                    # Sanitize timestamps do evento
                    absorption_event = self._sanitize_event(absorption_event, default_ts_ms=default_ts_ms)
                    if absorption_event.get("is_signal", False):
                        signals.append(absorption_event)
            except Exception as e:
                logging.error(f"Erro detectando absorção: {e}")

        # Exaustão
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

        # OrderBook (opcional, já vem como evento pronto do analisador)
        if isinstance(orderbook_data, dict) and orderbook_data.get("is_signal", False):
            try:
                ob_event = orderbook_data.copy()
                ob_event["layer"] = "signal"
                ob_event = self._sanitize_event(ob_event, default_ts_ms=default_ts_ms)
                signals.append(ob_event)
            except Exception as e:
                logging.error(f"Erro adicionando evento OrderBook: {e}")

        self.signal_data = signals
        logging.debug(f"✅ Camada Signal gerada. {len(signals)} sinais detectados.")
        return signals

    # ===============================
    # Consolidação
    # ===============================

    def get_final_features(self) -> Dict[str, Any]:
        """Retorna todas as features consolidadas (útil para Feature Store / IA)."""
        if self.enriched_data is None:
            self.enrich()
        if self.contextual_data is None:
            # cria contexto mínimo
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

        # Tempo de referência = close_time da janela
        try:
            close_time_ms = int(self.enriched_data.get("ohlc", {}).get("close_time", 0)) if self.enriched_data else None
        except Exception:
            close_time_ms = None

        time_index = self.tm.build_time_index(close_time_ms, include_local=True, timespec="seconds")

        features = {
            "schema_version": "1.1.0",
            "symbol": self.symbol,
            "timestamp": time_index.get("timestamp_utc", ""),  # compat legada
            "time_index": time_index,                           # novo bloco padronizado
            "enriched": self.enriched_data or {},
            "contextual": self.contextual_data or {},
            "signals": self.signal_data,
            "ml_features": {},  # será preenchido abaixo
        }

        # ✅ Geração robusta de ML features com DF de ticks (corrige falta de 'close')
        try:
            if generate_ml_features is not None and self.df is not None and self.contextual_data:
                orderbook_data = self.contextual_data.get("orderbook_data", {})
                flow_metrics = self.contextual_data.get("flow_metrics", {})

                # Monta DF amigável ao gerador de ML
                df_for_ml = self.df.copy()

                # Garantir colunas esperadas
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
