# historical_profiler.py
import requests
import pandas as pd
import logging
from datetime import datetime, timezone

class HistoricalVolumeProfiler:
    def __init__(self, symbol, num_days, value_area_percent):
        self.symbol = symbol
        self.num_days = num_days
        self.value_area_percent = value_area_percent
        self.api_url = "https://api.binance.com/api/v3/klines" # MUDANÇA: Usar klines é mais leve
        self.profile = {}

    def _fetch_historical_data(self, start_time_ms, end_time_ms):
        """Busca dados de klines de 1 minuto para o perfil de volume. É mais leve que aggTrades."""
        try:
            params = {
                "symbol": self.symbol,
                "interval": "1m",
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": 1000 # Binance limita a 1000 klines por chamada
            }
            response = requests.get(self.api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            
            # Converte as colunas relevantes para numérico
            # 'close' será nosso nível de preço 'p', e 'volume' será nossa quantidade 'q'
            df['p'] = pd.to_numeric(df['close'])
            df['q'] = pd.to_numeric(df['volume'])
            return df[['p', 'q']]
        except Exception as e:
            logging.error(f"Erro ao buscar klines históricos para VP: {e}")
            return pd.DataFrame()

    def _calculate_profile(self, df):
        if df.empty or df['q'].sum() == 0:
            return {'poc': 0, 'vah': 0, 'val': 0, 'status': 'no_data'}

        try:
            # Arredondar o preço ajuda a agrupar o volume em níveis discretos
            price_precision = 0 
            price_volume = df.groupby(df['p'].round(price_precision))['q'].sum()
            
            if price_volume.empty:
                return {'poc': 0, 'vah': 0, 'val': 0, 'status': 'no_volume_data'}

            total_volume = price_volume.sum()
            poc_price = price_volume.idxmax()
            value_area_volume_target = total_volume * self.value_area_percent
            
            poc_index = price_volume.index.get_loc(poc_price)
            current_volume_in_va = price_volume.iloc[poc_index]
            upper_index, lower_index = poc_index + 1, poc_index - 1
            
            while current_volume_in_va < value_area_volume_target:
                can_expand_up = upper_index < len(price_volume)
                can_expand_down = lower_index >= 0
                if not can_expand_up and not can_expand_down: break

                vol_upper = price_volume.iloc[upper_index] if can_expand_up else 0
                vol_lower = price_volume.iloc[lower_index] if can_expand_down else 0

                if vol_upper > vol_lower:
                    current_volume_in_va += vol_upper; upper_index += 1
                else:
                    current_volume_in_va += vol_lower; lower_index -= 1
            
            vah_price = price_volume.index[min(upper_index - 1, len(price_volume) - 1)]
            val_price = price_volume.index[max(lower_index + 1, 0)]

            return {'poc': float(poc_price), 'vah': float(vah_price), 'val': float(val_price), 'status': 'success'}
        except Exception as e:
            logging.error(f"Falha ao calcular o perfil de volume: {e}", exc_info=True)
            return {'poc': 0, 'vah': 0, 'val': 0, 'status': f'calculation_error'}

    def update_profiles(self):
        try:
            logging.info("Atualizando Volume Profile Histórico...")
            now = datetime.now(timezone.utc)
            start_of_day_utc = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_of_day_ms = int(start_of_day_utc.timestamp() * 1000)
            
            df_daily = self._fetch_historical_data(start_of_day_ms, int(now.timestamp() * 1000))
            daily_profile = self._calculate_profile(df_daily)
            
            self.profile = {"daily": daily_profile}
            logging.info(f"Volume Profile Histórico atualizado.")
            return self.profile
        except Exception as e:
            logging.error(f"Falha crítica ao atualizar perfis de volume: {e}", exc_info=True)
            return {"daily": {'poc': 0, 'vah': 0, 'val': 0, 'status': f'critical_error'}}