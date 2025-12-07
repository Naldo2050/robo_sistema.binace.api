# backtester.py - v3.0.1 (SQLite + Institutional Metrics)
import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging

# Configuração
DB_PATH = Path("dados/trading_bot.db")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class Backtester:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        if not self.db_path.exists():
            raise FileNotFoundError(f"Banco não encontrado: {db_path}")

    # ------------------------------------------------------------------ #
    # CARREGAMENTO DE SINAIS
    # ------------------------------------------------------------------ #
    def load_signals(self, limit: int = 5000) -> pd.DataFrame:
        """
        Carrega potenciais sinais de trading do banco SQLite.

        Não depende de is_signal = 1 (no banco atual esse campo está 0).
        Em vez disso, usa resultado_da_batalha para inferir BUY/SELL.
        """
        logging.info("Carregando eventos do banco de dados...")

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

        # Carrega todos os eventos (ou um subconjunto) e filtra em Python
        query = """
            SELECT timestamp_ms, event_type, symbol, is_signal, payload
            FROM events
            ORDER BY timestamp_ms ASC
            LIMIT ?
        """

        try:
            df = pd.read_sql_query(query, conn, params=(limit,))
        except Exception as e:
            logging.error(f"Erro ao carregar eventos: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

        if df.empty:
            logging.warning("Consulta de eventos retornou DataFrame vazio.")
            return pd.DataFrame()

        signals = []

        for _, row in df.iterrows():
            try:
                data = json.loads(row["payload"])
            except Exception:
                continue

            # 1) Direção do sinal:
            #    usa resultado_da_batalha; se não existir, usa ai_payload.flow_context.absorption_type
            resultado = data.get("resultado_da_batalha")
            if not resultado:
                ai_payload = data.get("ai_payload") or {}
                if isinstance(ai_payload, dict):
                    flow_ctx = ai_payload.get("flow_context") or {}
                    if isinstance(flow_ctx, dict):
                        resultado = flow_ctx.get("absorption_type")

            resultado = str(resultado or "")
            side = "NEUTRAL"

            # Absorção/Exaustão de Venda -> BUY (bullish)
            if "Venda" in resultado or "Short" in resultado:
                side = "BUY"
            # Absorção/Exaustão de Compra -> SELL (bearish)
            elif "Compra" in resultado or "Long" in resultado:
                side = "SELL"

            if side == "NEUTRAL":
                # Ignora eventos onde não conseguimos inferir direção
                continue

            # Preço de entrada
            price_raw = (
                data.get("preco_fechamento")
                or data.get("price")
                or data.get("anchor_price")
            )
            try:
                price = float(price_raw) if price_raw not in (None, "") else np.nan
            except Exception:
                price = np.nan

            if not np.isfinite(price) or price <= 0:
                continue  # sem preço não dá para backtestar

            # Delta e volume (opcionais)
            delta_raw = data.get("delta") or 0
            volume_raw = data.get("volume_total") or 0
            try:
                delta = float(delta_raw)
            except Exception:
                delta = 0.0
            try:
                volume = float(volume_raw)
            except Exception:
                volume = 0.0

            signals.append(
                {
                    "timestamp": pd.to_datetime(row["timestamp_ms"], unit="ms"),
                    "type": data.get("tipo_evento"),
                    "result": resultado,
                    "price": price,
                    "delta": delta,
                    "volume": volume,
                    "side": side,
                }
            )

        if not signals:
            logging.warning("Nenhum sinal válido encontrado após processamento.")
            return pd.DataFrame()

        signals_df = pd.DataFrame(signals).sort_values("timestamp").reset_index(
            drop=True
        )
        logging.info(f"{len(signals_df)} sinais válidos carregados para backtest.")
        return signals_df

    # ------------------------------------------------------------------ #
    # ESTRATÉGIA SIMPLES
    # ------------------------------------------------------------------ #
    def run_simple_strategy(
        self, tp_pct: float = 0.005, sl_pct: float = 0.002, hold_time_minutes: int = 15
    ) -> pd.DataFrame | None:
        """
        Simula uma estratégia simples:
        - Entra no sinal (Absorção de Venda -> BUY, Absorção de Compra -> SELL)
        - Sai no TP (Take Profit), SL (Stop Loss) ou Tempo Limite.
        - OBS: usa apenas os preços de outros sinais como proxy do futuro.
        """
        df = self.load_signals()
        if df.empty:
            logging.warning("Nenhum sinal encontrado para backtest.")
            return None

        if len(df) < 5:
            logging.warning("Quantidade de sinais muito pequena para backtest.")
            return None

        logging.info(f"🔬 Iniciando Backtest em {len(df)} sinais...")
        logging.info(
            "   Parâmetros: TP=%.2f%%, SL=%.2f%%, Tempo=%dmin",
            tp_pct * 100,
            sl_pct * 100,
            hold_time_minutes,
        )

        results = []

        # Ignora os últimos N eventos para garantir alguma janela futura
        max_lookahead = 20
        min_future = 3  # pelo menos 3 pontos futuros
        upper_limit = max(len(df) - max_lookahead, len(df) - min_future)

        for i in range(0, upper_limit):
            row = df.iloc[i]
            entry_price = row["price"]
            entry_time = row["timestamp"]
            side = row["side"]

            # Janela futura de busca (ex: próximos 20 eventos)
            future_window = df.iloc[i + 1 : i + 1 + max_lookahead]

            exit_price = entry_price
            reason = "Hold"  # se acabar o tempo sem bater TP/SL

            for _, future_row in future_window.iterrows():
                curr_price = future_row["price"]
                time_diff = (
                    future_row["timestamp"] - entry_time
                ).total_seconds() / 60.0

                # 1. Sai por tempo
                if time_diff > hold_time_minutes:
                    exit_price = curr_price
                    reason = "Time"
                    break

                # 2. Verifica TP/SL
                if side == "BUY":
                    if curr_price >= entry_price * (1 + tp_pct):
                        exit_price = entry_price * (1 + tp_pct)
                        reason = "TP"
                        break
                    if curr_price <= entry_price * (1 - sl_pct):
                        exit_price = entry_price * (1 - sl_pct)
                        reason = "SL"
                        break
                else:  # SELL
                    if curr_price <= entry_price * (1 - tp_pct):
                        exit_price = entry_price * (1 - tp_pct)
                        reason = "TP"
                        break
                    if curr_price >= entry_price * (1 + sl_pct):
                        exit_price = entry_price * (1 + sl_pct)
                        reason = "SL"
                        break

            # 3. Calcula PnL em %
            if side == "BUY":
                pnl_pct = (exit_price - entry_price) / entry_price
            else:  # SELL
                pnl_pct = (entry_price - exit_price) / entry_price

            # Desconta taxas (0,04% taker x 2 = 0,08%)
            pnl_pct -= 0.0008

            results.append(
                {
                    "entry_time": entry_time,
                    "side": side,
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl_pct": pnl_pct,
                    "reason": reason,
                }
            )

        trades_df = pd.DataFrame(results)
        if trades_df.empty:
            logging.warning("Nenhum trade gerado na simulação.")
            return None

        return trades_df

    # ------------------------------------------------------------------ #
    # MÉTRICAS INSTITUCIONAIS
    # ------------------------------------------------------------------ #
    def calculate_metrics(self, trades_df: pd.DataFrame | None) -> None:
        """Calcula e imprime métricas institucionais básicas."""
        if trades_df is None or trades_df.empty:
            logging.warning("Sem trades para calcular métricas.")
            return

        total_trades = len(trades_df)
        wins = trades_df[trades_df["pnl_pct"] > 0]
        losses = trades_df[trades_df["pnl_pct"] <= 0]

        win_rate = len(wins) / total_trades
        avg_win = wins["pnl_pct"].mean() if not wins.empty else 0.0
        avg_loss = losses["pnl_pct"].mean() if not losses.empty else 0.0

        # Curva de equity
        trades_df = trades_df.copy()
        trades_df["cum_return"] = (1 + trades_df["pnl_pct"]).cumprod()
        total_return = trades_df["cum_return"].iloc[-1] - 1

        # Sharpe (simplificado)
        mean_return = trades_df["pnl_pct"].mean()
        std_return = trades_df["pnl_pct"].std()
        annualization_factor = np.sqrt(2520)  # ~10 trades/dia * 252 dias
        sharpe = (
            annualization_factor * (mean_return / std_return)
            if std_return > 0
            else 0.0
        )

        # Max Drawdown
        cum_ret = trades_df["cum_return"]
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()

        # Profit Factor
        if losses.empty or losses["pnl_pct"].sum() == 0:
            profit_factor = float("inf")
        else:
            profit_factor = wins["pnl_pct"].sum() / abs(losses["pnl_pct"].sum())

        print("\n" + "=" * 50)
        print("📊 RELATÓRIO DE BACKTEST (INSTITUCIONAL)")
        print("=" * 50)
        print(f"Total Trades:      {total_trades}")
        print(f"Win Rate:          {win_rate:.2%}")
        print(f"Retorno Total:     {total_return:.2%}")
        print(f"Média Lucro:       {avg_win:.2%}")
        print(f"Média Prejuízo:    {avg_loss:.2%}")
        if np.isfinite(profit_factor):
            print(f"Profit Factor:     {profit_factor:.2f}")
        else:
            print("Profit Factor:     Inf")
        print("-" * 50)
        print(f"Sharpe Ratio:      {sharpe:.2f} (est.)")
        print(f"Max Drawdown:      {max_dd:.2%}")
        print("=" * 50 + "\n")

        print("Motivo de Saída:")
        print(trades_df["reason"].value_counts())


if __name__ == "__main__":
    bt = Backtester()
    try:
        print(">> Iniciando backtest...")
        # Roda estratégia: TP 0.5%, SL 0.2%, 15min max hold
        results = bt.run_simple_strategy(tp_pct=0.005, sl_pct=0.002, hold_time_minutes=15)
        if results is None or results.empty:
            print("Nenhum trade foi gerado no backtest (possivelmente faltam sinais válidos).")
        else:
            bt.calculate_metrics(results)
    except Exception as e:
        print(f"Erro no backtest: {e}")