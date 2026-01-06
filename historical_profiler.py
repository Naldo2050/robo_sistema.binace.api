import requests
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta

from config import VP_ADVANCED


class HistoricalVolumeProfiler:
    """
    Calcula Volume Profile histórico (daily/weekly/monthly) a partir de klines da Binance.

    Para cada perfil retorna:
        {
          "poc": float,
          "vah": float,
          "val": float,
          "hvns": [preços...],
          "lvns": [preços...],
          "single_prints": [preços...],
          "volume_nodes": {
            "hvn_nodes": [[price, volume, strength], ...],
            "lvn_nodes": [[price, volume, strength], ...]
          },
          "status": "success" | "no_data" | ...
        }
    """

    def __init__(self, symbol: str, num_days: int, value_area_percent: float):
        self.symbol = symbol
        self.num_days = num_days
        self.value_area_percent = value_area_percent
        self.api_url = "https://api.binance.com/api/v3/klines"
        self.profile = {}

    def _fetch_historical_data(self, start_time_ms: int, end_time_ms: int, interval: str = "1m") -> pd.DataFrame:
        """Busca klines no intervalo [start_time_ms, end_time_ms] para gerar o volume profile."""
        try:
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": 1000,
            }
            response = requests.get(self.api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(
                data,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "qav",
                    "num_trades",
                    "tbbav",
                    "tbqav",
                    "ignore",
                ],
            )
            df["p"] = pd.to_numeric(df["close"], errors="coerce")
            df["q"] = pd.to_numeric(df["volume"], errors="coerce")
            return df[["p", "q"]].dropna()
        except Exception as e:
            logging.error(f"Erro ao buscar dados históricos para {self.symbol}: {e}")
            return pd.DataFrame()

    def _calculate_profile(self, df: pd.DataFrame, label: str = "daily") -> dict:
        """Calcula POC, VAH, VAL, HVNs, LVNs, single prints e volume_nodes para um DataFrame de preços/volumes."""
        empty_result = {
            "poc": 0.0,
            "vah": 0.0,
            "val": 0.0,
            "hvns": [],
            "lvns": [],
            "single_prints": [],
            "volume_nodes": {"hvn_nodes": [], "lvn_nodes": []},
            "status": "no_data",
        }

        if df is None or df.empty or df["q"].sum() == 0:
            return empty_result

        try:
            # Agrupa por preço arredondado em $1
            price_volume = df.groupby(df["p"].round(0))["q"].sum()
            if price_volume.empty:
                res = empty_result.copy()
                res["status"] = "no_volume_data"
                return res

            total_volume = float(price_volume.sum())
            poc_price = float(price_volume.idxmax())

            # Cálculo da Value Area
            value_area_volume_target = total_volume * float(self.value_area_percent)

            poc_index = price_volume.index.get_loc(poc_price)
            current_volume_in_va = float(price_volume.iloc[poc_index])
            upper_index, lower_index = poc_index + 1, poc_index - 1

            while current_volume_in_va < value_area_volume_target:
                can_expand_up = upper_index < len(price_volume)
                can_expand_down = lower_index >= 0
                if not can_expand_up and not can_expand_down:
                    break

                vol_upper = float(price_volume.iloc[upper_index]) if can_expand_up else 0.0
                vol_lower = float(price_volume.iloc[lower_index]) if can_expand_down else 0.0

                if vol_upper > vol_lower:
                    current_volume_in_va += vol_upper
                    upper_index += 1
                else:
                    current_volume_in_va += vol_lower
                    lower_index -= 1

            vah_price = float(price_volume.index[min(upper_index - 1, len(price_volume) - 1)])
            val_price = float(price_volume.index[max(lower_index + 1, 0)])

            # ---------- HVNs / LVNs + volume_nodes ----------
            hvn_prices: list[float] = []
            lvn_prices: list[float] = []
            single_prints: list[float] = []
            hvn_nodes: list[list[float]] = []  # [price, volume, strength]
            lvn_nodes: list[list[float]] = []

            if VP_ADVANCED:
                mean_vol = float(price_volume.mean())
                std_vol = float(price_volume.std())
                high_threshold = mean_vol + std_vol
                low_threshold = max(mean_vol * 0.3, 1.0)

                max_vol = float(price_volume.max()) if price_volume.max() > 0 else 0.0

                # Classifica HVNs/LVNs e monta nodes compactos [price, volume, strength]
                for price, vol in price_volume.items():
                    price_f = float(price)
                    vol_f = float(vol)

                    if vol_f >= high_threshold:
                        hvn_prices.append(price_f)
                        strength = (vol_f / max_vol * 10.0) if max_vol > 0 else 0.0
                        hvn_nodes.append([price_f, vol_f, round(strength, 2)])
                    elif vol_f <= low_threshold:
                        lvn_prices.append(price_f)
                        if max_vol > 0:
                            strength = (1.0 - (vol_f / max_vol)) * 10.0
                        else:
                            strength = 0.0
                        lvn_nodes.append([price_f, vol_f, round(strength, 2)])

                # Single prints: vales muito fundos entre dois volumes maiores
                vols = price_volume.values
                idxs = price_volume.index
                for i in range(1, len(vols) - 1):
                    if vols[i] < vols[i - 1] * 0.3 and vols[i] < vols[i + 1] * 0.3:
                        single_prints.append(float(idxs[i]))

            result = {
                "poc": float(poc_price),
                "vah": float(vah_price),
                "val": float(val_price),
                "hvns": hvn_prices,
                "lvns": lvn_prices,
                "single_prints": single_prints,
                "volume_nodes": {
                    "hvn_nodes": hvn_nodes,
                    "lvn_nodes": lvn_nodes,
                },
                "status": "success",
            }
            return result

        except Exception as e:
            logging.error(f"Falha ao calcular perfil {label} para {self.symbol}: {e}", exc_info=True)
            err = empty_result.copy()
            err["status"] = "calculation_error"
            return err

    def update_profiles(self) -> dict:
        """Atualiza perfis daily / weekly / monthly e retorna o dicionário completo."""
        try:
            now = datetime.now(timezone.utc)
            profiles: dict = {}

            # DAILY
            start_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            df_daily = self._fetch_historical_data(
                int(start_day.timestamp() * 1000),
                int(now.timestamp() * 1000),
                interval="1m",
            )
            profiles["daily"] = self._calculate_profile(df_daily, "daily")

            if VP_ADVANCED:
                # WEEKLY
                start_week = now - timedelta(days=7)
                df_week = self._fetch_historical_data(
                    int(start_week.timestamp() * 1000),
                    int(now.timestamp() * 1000),
                    interval="5m",
                )
                profiles["weekly"] = self._calculate_profile(df_week, "weekly")

                # MONTHLY
                start_month = now - timedelta(days=30)
                df_month = self._fetch_historical_data(
                    int(start_month.timestamp() * 1000),
                    int(now.timestamp() * 1000),
                    interval="15m",
                )
                profiles["monthly"] = self._calculate_profile(df_month, "monthly")

            self.profile = profiles
            logging.info("✅ Volume Profile Histórico atualizado para %s.", self.symbol)
            return profiles

        except Exception as e:
            logging.error(f"Erro crítico ao atualizar Volume Profile para {self.symbol}: {e}", exc_info=True)
            return {
                "daily": {
                    "poc": 0.0,
                    "vah": 0.0,
                    "val": 0.0,
                    "hvns": [],
                    "lvns": [],
                    "single_prints": [],
                    "volume_nodes": {"hvn_nodes": [], "lvn_nodes": []},
                    "status": "critical_error",
                }
            }