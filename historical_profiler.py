import requests
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from config import VP_ADVANCED

class HistoricalVolumeProfiler:
    def __init__(self, symbol, num_days, value_area_percent):
        self.symbol = symbol
        self.num_days = num_days
        self.value_area_percent = value_area_percent
        self.api_url = "https://api.binance.com/api/v3/klines"
        self.profile = {}

    def _fetch_historical_data(self, start_time_ms, end_time_ms, interval="1m"):
        """Busca klines de 1 minuto (ou maior para períodos longos) para gerar volume profile."""
        try:
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": 1000
            }
            response = requests.get(self.api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','tbbav','tbqav','ignore'])
            df['p'] = pd.to_numeric(df['close'])
            df['q'] = pd.to_numeric(df['volume'])
            return df[['p', 'q']]
        except Exception as e:
            logging.error(f"Erro ao buscar dados históricos: {e}")
            return pd.DataFrame()

    def _calculate_profile(self, df, label="daily"):
        if df.empty or df['q'].sum() == 0:
            return {'poc': 0, 'vah': 0, 'val': 0, 'hvns': [], 'lvns': [], 'single_prints': [], 'status': 'no_data'}

        try:
            price_volume = df.groupby(df['p'].round(0))['q'].sum()  # arredonda preço em $1
            if price_volume.empty:
                return {'poc': 0, 'vah': 0, 'val': 0, 'hvns': [], 'lvns': [], 'single_prints': [], 'status': 'no_volume_data'}

            total_volume = price_volume.sum()
            poc_price = price_volume.idxmax()
            value_area_volume_target = total_volume * self.value_area_percent

            poc_index = price_volume.index.get_loc(poc_price)
            current_volume_in_va = price_volume.iloc[poc_index]
            upper_index, lower_index = poc_index+1, poc_index-1

            while current_volume_in_va < value_area_volume_target:
                can_expand_up = upper_index < len(price_volume)
                can_expand_down = lower_index >= 0
                if not can_expand_up and not can_expand_down:
                    break
                vol_upper = price_volume.iloc[upper_index] if can_expand_up else 0
                vol_lower = price_volume.iloc[lower_index] if can_expand_down else 0
                if vol_upper > vol_lower:
                    current_volume_in_va += vol_upper
                    upper_index += 1
                else:
                    current_volume_in_va += vol_lower
                    lower_index -= 1

            vah_price = price_volume.index[min(upper_index-1, len(price_volume)-1)]
            val_price = price_volume.index[max(lower_index+1, 0)]

            hvns, lvns, single_prints = [], [], []
            if VP_ADVANCED:
                mean_vol = price_volume.mean()
                std_vol = price_volume.std()
                high_threshold = mean_vol + std_vol
                low_threshold = max(mean_vol*0.3, 1)

                for price, vol in price_volume.items():
                    if vol >= high_threshold:
                        hvns.append(float(price))
                    elif vol <= low_threshold:
                        lvns.append(float(price))

                vols = price_volume.values
                idxs = price_volume.index
                for i in range(1, len(vols)-1):
                    if vols[i] < vols[i-1]*0.3 and vols[i] < vols[i+1]*0.3:
                        single_prints.append(float(idxs[i]))

            return {
                'poc': float(poc_price), 'vah': float(vah_price), 'val': float(val_price),
                'hvns': hvns, 'lvns': lvns, 'single_prints': single_prints, 'status': 'success'
            }
        except Exception as e:
            logging.error(f"Falha ao calcular perfil {label}: {e}", exc_info=True)
            return {'poc': 0, 'vah': 0, 'val': 0, 'hvns': [], 'lvns': [], 'single_prints': [], 'status': 'calculation_error'}

    def update_profiles(self):
        try:
            now = datetime.now(timezone.utc)
            profiles = {}
            # DAILY
            start_day = now.replace(hour=0,minute=0,second=0,microsecond=0)
            df_daily = self._fetch_historical_data(int(start_day.timestamp()*1000), int(now.timestamp()*1000))
            profiles["daily"] = self._calculate_profile(df_daily, "daily")

            if VP_ADVANCED:
                # WEEKLY
                start_week = now - timedelta(days=7)
                df_week = self._fetch_historical_data(int(start_week.timestamp()*1000), int(now.timestamp()*1000), interval="5m")
                profiles["weekly"] = self._calculate_profile(df_week, "weekly")

                # MONTHLY
                start_month = now - timedelta(days=30)
                df_month = self._fetch_historical_data(int(start_month.timestamp()*1000), int(now.timestamp()*1000), interval="15m")
                profiles["monthly"] = self._calculate_profile(df_month, "monthly")

            self.profile = profiles
            logging.info("✅ Volume Profile Histórico atualizado.")
            return profiles
        except Exception as e:
            logging.error(f"Erro crítico atualizar VP: {e}", exc_info=True)
            return {"daily": {'poc': 0, 'vah': 0, 'val': 0, 'hvns': [], 'lvns': [], 'single_prints': [], 'status':'critical_error'}}