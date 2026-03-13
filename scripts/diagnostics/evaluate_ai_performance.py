# diagnostics/evaluate_ai_performance.py
# -*- coding: utf-8 -*-

"""
Script de Avaliação de Performance da IA (Post-Mortem).

Lê eventos 'AI_ANALYSIS' do banco SQLite e cruza com dados de mercado
para calcular a rentabilidade e acerto das recomendações.

Como usar:
1. Certifique-se de ter dados de preço (candles/trades) cobrindo o período dos eventos.
2. Implemente a função `get_future_prices` abaixo com sua fonte real de dados.
3. Execute: python diagnostics/evaluate_ai_performance.py
"""

import sqlite3
import json
import logging
import sys
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# Import das métricas avançadas
from performance_metrics import (
    calculate_profit_factor,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_all_metrics,
)

# Configuração de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AIEvaluator")

# Caminhos
DB_PATH = "dados/trading_bot.db"
OUTPUT_CSV = "ai_performance_report.csv"

# Horizontes de tempo para avaliação (em minutos)
HORIZONS = [1, 5, 15, 60]

# ============================================================================
# 1. FONTE DE DADOS DE PREÇO (STUB - IMPLEMENTE AQUI)
# ============================================================================

def get_future_prices(symbol: str, start_ts_ms: int, duration_minutes: int = 60) -> pd.DataFrame:
    """
    Recupera preços futuros a partir de um timestamp via API Binance.

    Args:
        symbol: Par de trading (ex: BTCUSDT).
        start_ts_ms: Timestamp inicial (momento da recomendação).
        duration_minutes: Quanto tempo no futuro buscar.

    Returns:
        pd.DataFrame com colunas ['timestamp_ms', 'price', 'high', 'low'].
        Deve estar ordenado por timestamp_ms.
    """
    # Primeiro, tenta carregar de arquivo CSV local se existir
    historical_file = f"../dados/history_{symbol}.csv"
    if os.path.exists(historical_file):
        try:
            df = pd.read_csv(historical_file)
            end_ts_ms = start_ts_ms + (duration_minutes * 60 * 1000)
            mask = (df['timestamp_ms'] >= start_ts_ms) & (df['timestamp_ms'] <= end_ts_ms)
            df_filtered = df.loc[mask].copy()
            if 'close' in df_filtered.columns:
                df_filtered.rename(columns={'close': 'price'}, inplace=True)
            return df_filtered.sort_values('timestamp_ms')
        except Exception as e:
            logger.warning(f"Erro ao ler CSV local: {e}, tentando API...")

    # Fallback: Busca via API Binance
    try:
        base_url = "https://api.binance.com/api/v3/klines"
        interval = "1m"  # 1 minuto para precisão
        start_time = start_ts_ms
        end_time = start_ts_ms + (duration_minutes * 60 * 1000)

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000  # Máximo por request
        }

        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if not data:
            logger.warning(f"Nenhum dado retornado da API para {symbol}")
            return pd.DataFrame()

        # Converte para DataFrame
        # Formato klines: [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(data, columns=[
            'timestamp_ms', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Converte tipos
        df['timestamp_ms'] = df['timestamp_ms'].astype(int)
        df['price'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        # Filtra apenas colunas necessárias
        df = df[['timestamp_ms', 'price', 'high', 'low']].copy()

        # Ordena por timestamp
        df = df.sort_values('timestamp_ms').reset_index(drop=True)

        logger.info(f"Buscados {len(df)} candles via API para {symbol}")
        return df

    except requests.RequestException as e:
        logger.error(f"Erro na requisição API Binance: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Erro ao processar dados da API: {e}")
        return pd.DataFrame()


# ============================================================================
# 2. MOTOR DE AVALIAÇÃO
# ============================================================================

class AIPerformanceEvaluator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def load_ai_events(self) -> List[Dict[str, Any]]:
        """Carrega eventos de análise da IA do banco de dados."""
        if not os.path.exists(self.db_path):
            logger.error(f"Banco de dados não encontrado: {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Lê apenas o payload, como no replay_validator
            cursor.execute("SELECT payload FROM events ORDER BY timestamp_ms ASC")

            results: List[Dict[str, Any]] = []

            for (payload_json,) in cursor:
                try:
                    ev = json.loads(payload_json)
                except json.JSONDecodeError:
                    continue

                # Filtra apenas eventos de IA
                if (ev.get("tipo_evento") or "").upper() != "AI_ANALYSIS":
                    continue

                # Extrai timestamp e símbolo a partir do payload
                ts_ms = (
                    ev.get("timestamp_ms")
                    or ev.get("epoch_ms")
                    or ev.get("metadata", {}).get("timestamp_unix_ms")
                )
                sym = ev.get("symbol") or ev.get("ativo") or "BTCUSDT"

                if ts_ms is None:
                    continue

                results.append(
                    {
                        "timestamp_ms": int(ts_ms),
                        "symbol": sym,
                        "data": ev,
                    }
                )

            conn.close()
            logger.info(f"Carregados {len(results)} eventos de IA.")
            return results

        except Exception as e:
            logger.error(f"Erro ao acessar banco de dados: {e}", exc_info=True)
            return []

    def evaluate_signal(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Avalia um único sinal da IA contra o mercado futuro.
        """
        ts_ms = event["timestamp_ms"]
        symbol = event["symbol"]
        data = event["data"]
        
        # Extrai resultado da IA
        ai_result = data.get("ai_result", {})
        if not ai_result:
            return None
            
        action = ai_result.get("action", "unknown").lower()
        sentiment = ai_result.get("sentiment", "neutral")
        confidence = ai_result.get("confidence", 0.0)
        
        # Filtro de confiança: abaixo de 0.7, não avaliamos o sinal (equivalente a "wait")
        if confidence < 0.7:
            return None
                
        # Extrai contexto (opcional, para análise de correlação)
        ai_payload = data.get("ai_payload", {})
        flow_ctx = ai_payload.get("flow_context", {})
        macro_ctx = ai_payload.get("macro_context", {})
        
        # Busca preços futuros (máximo horizonte)
        max_horizon = max(HORIZONS)
        df_prices = get_future_prices(symbol, ts_ms, max_horizon)
        
        if df_prices.empty:
            return None
            
        # Preço de entrada: tenta usar anchor_price vindo do evento; se não existir, usa o primeiro preço futuro
        entry_price = data.get("anchor_price")
        if entry_price is None:
            entry_price = df_prices.iloc[0]['price']
        
        metrics = {
            "timestamp_iso": datetime.fromtimestamp(ts_ms/1000).isoformat(),
            "symbol": symbol,
            "action": action,
            "sentiment": sentiment,
            "confidence": confidence,
            "entry_price": entry_price,

            # Contexto Extra
            "regime": macro_ctx.get("regime", {}).get("structure", "N/A"),
            "net_flow": flow_ctx.get("net_flow", 0),
            "whale_delta": flow_ctx.get("whale_activity", {}).get("whale_delta", 0),

            # Novos campos de contexto de mercado
            "anchor_price": data.get("anchor_price"),
            "anchor_window_id": data.get("anchor_window_id"),
            "trend_1h": macro_ctx.get("trend_1h"),
            "trend_4h": macro_ctx.get("trend_4h"),
            "regime_4h": macro_ctx.get("regime_4h"),
            "flow_imbalance": flow_ctx.get("flow_imbalance"),
            "net_flow_1m": flow_ctx.get("net_flow_1m"),
            "absorption_type": ai_payload.get("absorption_type")
        }
        
        # Direção do trade
        direction = 0
        if action == "buy" or (action == "hold" and sentiment == "bullish"):
            direction = 1
        elif action == "sell" or (action == "hold" and sentiment == "bearish"):
            direction = -1
        
        # Avaliação por horizonte
        for minutes in HORIZONS:
            # Filtra dados até esse horizonte
            horizon_ms = ts_ms + (minutes * 60 * 1000)
            df_h = df_prices[df_prices['timestamp_ms'] <= horizon_ms]
            
            if df_h.empty:
                metrics[f"ret_{minutes}m"] = None
                continue
                
            last_price = df_h.iloc[-1]['price']
            
            # Retorno simples
            raw_return = (last_price - entry_price) / entry_price
            trade_return = raw_return * direction
            
            metrics[f"ret_{minutes}m"] = round(trade_return * 100, 4) # Em %
            
            # Máximo Favorável (MFE) e Adverso (MAE)
            if direction == 1: # Long
                max_price = df_h['price'].max() # Ou 'high' se tiver
                min_price = df_h['price'].min() # Ou 'low' se tiver
                mfe = (max_price - entry_price) / entry_price
                mae = (min_price - entry_price) / entry_price
            elif direction == -1: # Short
                max_price = df_h['price'].max()
                min_price = df_h['price'].min()
                mfe = (entry_price - min_price) / entry_price # Lucro na queda
                mae = (entry_price - max_price) / entry_price # Prejuízo na alta
            else:
                mfe = 0
                mae = 0
                
            metrics[f"mfe_{minutes}m"] = round(mfe * 100, 4)
            metrics[f"mae_{minutes}m"] = round(mae * 100, 4)

        # Verifica Invalidação (Stop Loss sugerido pela IA)
        invalidation_zone = ai_result.get("invalidation_zone")
        hit_invalidation = False
        
        if invalidation_zone:
            # Tenta parsear número da string (ex: "abaixo de 95000")
            try:
                # Extração simplista de números
                import re
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(invalidation_zone).replace(",", ""))
                if nums:
                    stop_price = float(nums[0])
                    
                    # Verifica se tocou no stop em qualquer momento do horizonte máximo
                    if direction == 1: # Long, stop abaixo
                        min_reached = df_prices['price'].min()
                        if min_reached <= stop_price:
                            hit_invalidation = True
                    elif direction == -1: # Short, stop acima
                        max_reached = df_prices['price'].max()
                        if max_reached >= stop_price:
                            hit_invalidation = True
            except:
                pass # Falha ao parsear stop
                
        metrics["hit_invalidation"] = hit_invalidation
        
        return metrics

    def run(self):
        logger.info("Iniciando avaliação de performance da IA...")
        
        events = self.load_ai_events()
        results = []
        
        for i, event in enumerate(events):
            res = self.evaluate_signal(event)
            if res:
                results.append(res)
            
            if (i+1) % 50 == 0:
                logger.info(f"Processados {i+1}/{len(events)} eventos...")

        if not results:
            logger.warning("Nenhum resultado gerado (falta de dados de preço ou eventos?).")
            return

        # Cria DataFrame e Salva CSV
        df_res = pd.DataFrame(results)
        
        # Ordena colunas
        cols = [
            "timestamp_iso", "symbol", "action", "sentiment", "confidence",
            "regime", "hit_invalidation"
        ]
        # Adiciona colunas dinâmicas de retorno
        for m in HORIZONS:
            cols.extend([f"ret_{m}m", f"mfe_{m}m", f"mae_{m}m"])
            
        # Mantém colunas extras no final
        existing_cols = [c for c in cols if c in df_res.columns]
        extra_cols = [c for c in df_res.columns if c not in cols]
        df_res = df_res[existing_cols + extra_cols]
        
        df_res.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Relatório salvo em: {OUTPUT_CSV}")
        
        self.print_summary(df_res)

    def print_summary(self, df: pd.DataFrame):
        """Imprime um resumo estatístico no console."""
        print("\n" + "="*70)
        print("📊 RESUMO DE PERFORMANCE DA IA")
        print("="*70)
        
        print(f"Total de recomendações avaliadas: {len(df)}")
        
        # Filtra apenas ações direcionais (ignora 'wait', 'hold' neutro)
        df_active = df[df['action'].isin(['buy', 'sell'])]
        
        if df_active.empty:
            print("Nenhum trade ativo (buy/sell) encontrado.")
            return

        print(f"\nTrades Ativos (Buy/Sell): {len(df_active)}")
        
        # Tabela de métricas por horizonte
        print("\n" + "-"*70)
        print("MÉTRICAS POR HORIZONTE")
        print("-"*70)
        print(f"{'Horizonte':<12} {'Trades':>7} {'Win Rate':>10} {'Avg Ret':>10} {'PF':>8} {'Max DD':>10} {'Sharpe':>8}")
        print("-"*70)
        
        for m in HORIZONS:
            col = f"ret_{m}m"
            if col not in df_active.columns: continue
            
            returns = df_active[col].dropna().values
            
            if len(returns) == 0:
                continue
            
            # Métricas básicas
            avg_ret = np.mean(returns)
            win_rate = (returns > 0).mean() * 100
            
            # Métricas avançadas
            pf = calculate_profit_factor(returns)
            mdd, _, _ = calculate_max_drawdown(returns)
            sharpe = calculate_sharpe_ratio(returns)
            
            # Formata valores
            pf_str = f"{pf:.2f}" if pf is not None else "N/A"
            mdd_str = f"{mdd:.2f}%" if mdd is not None else "N/A"
            sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
            
            print(f"{m:>4} min     {len(returns):>7} {win_rate:>9.1f}% {avg_ret:>9.3f}% {pf_str:>8} {mdd_str:>10} {sharpe_str:>8}")
        
        print("-"*70)
        
        # Tabela por bucket de confiança (horizonte 15min)
        target_col = 'ret_15m' if 'ret_15m' in df_active.columns else 'ret_60m'
        if target_col in df_active.columns and 'confidence' in df_active.columns:
            print("\n" + "-"*70)
            print(f"MÉTRICAS POR BUCKET DE CONFIANÇA ({target_col})")
            print("-"*70)
            print(f"{'Confiança':<15} {'Trades':>7} {'Win Rate':>10} {'Avg Ret':>10} {'PF':>8} {'Sharpe':>8}")
            print("-"*70)
            
            # Define buckets
            buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
            
            for low, high in buckets:
                mask = (df_active['confidence'] >= low) & (df_active['confidence'] < high)
                returns = df_active.loc[mask, target_col].dropna().values
                
                if len(returns) < 2:
                    continue
                
                avg_ret = np.mean(returns)
                win_rate = (returns > 0).mean() * 100
                pf = calculate_profit_factor(returns)
                sharpe = calculate_sharpe_ratio(returns)
                
                pf_str = f"{pf:.2f}" if pf is not None else "N/A"
                sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
                
                print(f"{low:.1f} - {high:.1f}      {len(returns):>7} {win_rate:>9.1f}% {avg_ret:>9.3f}% {pf_str:>8} {sharpe_str:>8}")
            
            print("-"*70)

        # Invalidação
        inval_rate = df_active['hit_invalidation'].mean() * 100
        print(f"\n🚫 Taxa de Invalidação (Stop Loss atingido): {inval_rate:.1f}%")
        
        # Distribuição
        print("\n📈 Distribuição de Resultados (15min):")
        if 'ret_15m' in df_active.columns:
            ret_15 = df_active['ret_15m'].dropna()
            big_wins = (ret_15 > 0.5).sum()
            big_loss = (ret_15 < -0.5).sum()
            print(f"   Big Wins (>0.5%):  {big_wins}")
            print(f"   Big Loss (<-0.5%): {big_loss}")
        
        print("="*70)


if __name__ == "__main__":
    # Garante que o diretório de saída existe
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    evaluator = AIPerformanceEvaluator(DB_PATH)
    evaluator.run()