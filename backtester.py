import os
import pandas as pd
from datetime import datetime, timedelta

# 🔹 IMPORTA TIME MANAGER (opcional, mas útil para consistência)
from time_manager import TimeManager

REPORTS_DIR = "./reports"

class Backtester:
    def __init__(self, reports_dir=REPORTS_DIR):
        self.reports_dir = reports_dir
        # 🔹 Inicializa TimeManager (opcional, para consistência futura)
        self.time_manager = TimeManager()

    def load_reports(self, symbol: str = None, start_date: str = None, end_date: str = None):
        """Carrega relatórios CSV dentro do intervalo escolhido"""
        all_files = []
        for f in os.listdir(self.reports_dir):
            if not f.endswith(".csv"):
                continue
            if symbol and not f.startswith(symbol):
                continue
            all_files.append(os.path.join(self.reports_dir, f))

        dfs = []
        for file in all_files:
            df = pd.read_csv(file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            dfs.append(df)

        if not dfs:
            print("Nenhum relatório encontrado.")
            return pd.DataFrame()

        full_df = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)

        if start_date:
            full_df = full_df[full_df["timestamp"] >= pd.to_datetime(start_date)]
        if end_date:
            full_df = full_df[full_df["timestamp"] <= pd.to_datetime(end_date)]

        return full_df

    def evaluate_signals(self, df: pd.DataFrame, price_data: pd.DataFrame, horizon_minutes=15):
        """
        Avalia os sinais comparando com preços futuros
        - df: dataframe de sinais gerados pela IA
        - price_data: dataframe OHLC ou preços futuros com timestamps
        - horizon_minutes: horizonte de tempo p/ medir resultado
        """

        results = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            ts_dt = pd.to_datetime(ts)

            # pega preço no momento do sinal
            if "close" in price_data.columns:
                # 🔹 Busca o último preço ANTES ou NO momento do sinal
                mask = price_data["timestamp"] <= ts_dt
                if mask.any():
                    price_now = price_data.loc[mask, "close"].iloc[-1]
                else:
                    continue
            else:
                continue

            # horizonte no futuro
            horizon_dt = ts_dt + timedelta(minutes=horizon_minutes)
            df_future = price_data[(price_data["timestamp"] > ts_dt) & (price_data["timestamp"] <= horizon_dt)]
            if df_future.empty:
                continue

            price_future = df_future["close"].iloc[-1]

            # classificação
            analysis_text = row.get("ai_analysis", "").lower()
            if "compra" in analysis_text or "long" in analysis_text:
                signal_dir = "long"
            elif "venda" in analysis_text or "short" in analysis_text:
                signal_dir = "short"
            else:
                signal_dir = "neutral"

            if signal_dir == "long":
                pnl = price_future - price_now
            elif signal_dir == "short":
                pnl = price_now - price_future
            else:
                pnl = 0

            results.append({
                "timestamp": ts_dt,
                "ativo": row["ativo"],
                "tipo_evento": row["tipo_evento"],
                "direcao": signal_dir,
                "preco_entrada": price_now,
                "preco_saida": price_future,
                "pnl": pnl
            })

        return pd.DataFrame(results)

    def report_stats(self, trades_df: pd.DataFrame):
        """Resumo estatístico do backtest"""
        if trades_df.empty:
            print("Nenhum trade gerado.")
            return

        win_rate = (trades_df["pnl"] > 0).mean() * 100
        avg_pnl = trades_df["pnl"].mean()
        total_pnl = trades_df["pnl"].sum()

        print("\n===== 📊 RESULTADOS BACKTEST =====")
        print(f"Total sinais: {len(trades_df)}")
        print(f"WinRate: {win_rate:.2f}%")
        print(f"Média PnL: {avg_pnl:.2f}")
        print(f"PnL total: {total_pnl:.2f}")

        print("\nPor tipo de evento:")
        print(trades_df.groupby("tipo_evento")["pnl"].describe())