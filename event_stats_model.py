import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

import config

DATA_DIR = Path("dados")
HISTORY_FILE = DATA_DIR / "eventos_fluxo.jsonl"

class EventStatsModel:
    def __init__(self):
        self.events = pd.DataFrame()
        self.lookbacks = config.ML_LOOKBACK_WINDOWS
        self.min_sample = config.ML_MIN_SAMPLE_SIZE
        self.update_interval = config.ML_UPDATE_INTERVAL
        logging.info("📊 EventStatsModel iniciado.")

    def load_events(self):
        """Carrega eventos do arquivo histórico JSONL para um DataFrame pandas"""
        try:
            if not HISTORY_FILE.exists():
                logging.warning("Nenhum arquivo de histórico encontrado.")
                return
            self.events = pd.read_json(HISTORY_FILE, lines=True)
            logging.info(f"{len(self.events)} eventos carregados para análise estatística.")
        except Exception as e:
            logging.error(f"Erro ao carregar histórico de eventos: {e}")

    def _filter_setups(self, setup_filter: dict):
        """Filtra eventos que atendem ao setup especificado (dict de chaves/valores)."""
        df = self.events
        for key, val in setup_filter.items():
            if key not in df.columns:
                return pd.DataFrame()
            df = df[df[key] == val]
        return df

    def _calculate_outcomes(self, filtered_df: pd.DataFrame):
        """Calcula probabilidades de resultado após cada setup filtrado."""
        results = {}
        if filtered_df.empty or len(filtered_df) < self.min_sample:
            return {"status": "amostra_insuficiente", "sample_size": len(filtered_df)}

        # Garante ordem temporal
        filtered_df = filtered_df.sort_values("candle_close_time_ms")
        closes = filtered_df["preco_fechamento"].values
        times = filtered_df["candle_close_time_ms"].values

        df_all = self.events.sort_values("candle_close_time_ms")
        outcomes = {"sample_size": len(filtered_df)}

        for lb in self.lookbacks:
            moves = []
            for close_price, close_time in zip(closes, times):
                future = df_all[df_all["candle_close_time_ms"] > close_time]
                future = future[future["candle_close_time_ms"] <= close_time + lb*60*1000]
                if future.empty:
                    continue
                pct_move = (future["preco_fechamento"].iloc[-1] - close_price) / close_price
                moves.append(pct_move)
            if moves:
                up_prob = sum(1 for m in moves if m > 0) / len(moves)
                down_prob = sum(1 for m in moves if m < 0) / len(moves)
                avg_move = sum(moves)/len(moves)
                outcomes[f"forward_{lb}m"] = {
                    "prob_up": round(up_prob*100,2),
                    "prob_down": round(down_prob*100,2),
                    "avg_pct_move": round(avg_move*100,3),
                    "sample_size": len(moves)
                }
            else:
                outcomes[f"forward_{lb}m"] = {"status": "sem_dados"}
        return outcomes

    def calculate_probabilities(self, setup_filter: dict):
        """
        Calcula probabilidade condicional de movimentos dado um setup.
        Exemplo setup_filter: {"tipo_evento":"Absorção","spoofing_detected":True}
        """
        if self.events.empty:
            self.load_events()
        filtered_df = self._filter_setups(setup_filter)
        return self._calculate_outcomes(filtered_df)

# ===============================
# Exemplo de uso manual
# ===============================
if __name__ == "__main__":
    model = EventStatsModel()
    model.load_events()
    setup = {"tipo_evento": "Absorção", "spoofing_detected": True}
    result = model.calculate_probabilities(setup)
    print("Resultado Estatístico:", json.dumps(result, indent=2, ensure_ascii=False))