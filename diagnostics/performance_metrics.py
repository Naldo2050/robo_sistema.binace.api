# diagnostics/performance_metrics.py
# -*- coding: utf-8 -*-

"""
Módulo de Métricas Avançadas de Performance.

Funções reutilizáveis para calcular:
- Profit Factor
- Max Drawdown
- Sharpe Ratio

Todas as funções esperam retornos em percentual (ex: 0.5 = 0.5%, não 50%).
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
import logging

logger = logging.getLogger("PerformanceMetrics")

# Risk-free rate configurável (pode ser sobrescrito via config)
DEFAULT_RISK_FREE_RATE = 0.0


def calculate_profit_factor(returns: Union[pd.Series, np.ndarray, List[float]]) -> Optional[float]:
    """
    Calcula o Profit Factor.
    
    PF = (soma dos ganhos) / (soma das perdas em valor absoluto)
    
    Args:
        returns: Array/Series de retornos (em %)
        
    Returns:
        float: Profit Factor (>1 = lucrativo, <1 = prejuízo)
        None: Se não houver dados suficientes
        
    Regras especiais:
        - Se não houver perdas → retorna 999.0 + warning
        - Se não houver ganhos → retorna 0.0
        - Se não houver trades → retorna None
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]  # Remove NaN
    
    if len(returns) == 0:
        return None
    
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    
    total_gains = gains.sum()
    total_losses = np.abs(losses.sum())
    
    if total_losses == 0:
        if total_gains > 0:
            logger.warning("Profit Factor: Sem perdas registradas. Retornando 999.0")
            return 999.0
        return None  # Sem ganhos nem perdas
    
    if total_gains == 0:
        return 0.0
    
    return round(total_gains / total_losses, 2)


def calculate_max_drawdown(returns: Union[pd.Series, np.ndarray, List[float]]) -> Tuple[float, int, int]:
    """
    Calcula o Maximum Drawdown (MDD).
    
    MDD = maior queda percentual peak-to-trough da curva de equity.
    
    Args:
        returns: Array/Series de retornos (em %)
        
    Returns:
        Tuple[float, int, int]: 
            - max_drawdown: MDD em % (valor negativo)
            - peak_idx: Índice do pico antes do drawdown
            - trough_idx: Índice do vale (fundo do drawdown)
            
    Exemplo:
        Se equity vai 100 → 110 → 90 → 105, o MDD é -18.2% (110→90)
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]  # Remove NaN
    
    if len(returns) == 0:
        return (0.0, 0, 0)
    
    # Calcula curva de equity cumulativa (começando em 100)
    equity_curve = 100 * np.cumprod(1 + returns / 100)
    
    # Calcula running maximum (picos)
    running_max = np.maximum.accumulate(equity_curve)
    
    # Calcula drawdowns em cada ponto
    drawdowns = (equity_curve - running_max) / running_max * 100
    
    # Encontra o máximo drawdown (mais negativo)
    max_dd_idx = np.argmin(drawdowns)
    max_dd = drawdowns[max_dd_idx]
    
    # Encontra o pico correspondente
    peak_idx = np.argmax(equity_curve[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
    
    return (round(max_dd, 2), int(peak_idx), int(max_dd_idx))


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray, List[float]],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    annualization_factor: float = 1.0
) -> Optional[float]:
    """
    Calcula o Sharpe Ratio.
    
    Sharpe = (mean(returns) - rf) / std(returns)
    
    Args:
        returns: Array/Series de retornos (em %)
        risk_free_rate: Taxa livre de risco (default: 0.0)
        annualization_factor: Fator para anualização (1.0 = sem anualização)
        
    Returns:
        float: Sharpe Ratio
        None: Se não houver dados suficientes ou std=0
        
    Nota:
        Para anualização, use:
        - annualization_factor = sqrt(252) para retornos diários
        - annualization_factor = sqrt(12) para retornos mensais
        - annualization_factor = 1.0 para retornos por trade (sem anualização)
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]  # Remove NaN
    
    if len(returns) < 2:
        return None
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)  # ddof=1 para sample std
    
    if std_return == 0:
        logger.warning("Sharpe Ratio: std=0, retornando None")
        return None
    
    excess_return = mean_return - risk_free_rate
    sharpe = (excess_return / std_return) * annualization_factor
    
    return round(sharpe, 3)


def calculate_all_metrics(
    returns: Union[pd.Series, np.ndarray, List[float]],
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> dict:
    """
    Calcula todas as métricas de performance.
    
    Args:
        returns: Array/Series de retornos (em %)
        risk_free_rate: Taxa livre de risco
        
    Returns:
        dict com todas as métricas
    """
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return {
            "count": 0,
            "win_rate": None,
            "avg_return": None,
            "profit_factor": None,
            "max_drawdown": None,
            "sharpe_ratio": None,
        }
    
    # Métricas básicas
    win_rate = (returns > 0).mean() * 100
    avg_return = returns.mean()
    
    # Métricas avançadas
    pf = calculate_profit_factor(returns)
    mdd, _, _ = calculate_max_drawdown(returns)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    
    return {
        "count": len(returns),
        "win_rate": round(win_rate, 2),
        "avg_return": round(avg_return, 4),
        "profit_factor": pf,
        "max_drawdown": mdd,
        "sharpe_ratio": sharpe,
    }


def format_metrics_table(metrics_by_horizon: dict, title: str = "Performance Metrics") -> str:
    """
    Formata métricas em tabela markdown.
    
    Args:
        metrics_by_horizon: Dict[horizonte, dict_metricas]
        
    Returns:
        str: Tabela formatada em markdown
    """
    lines = [
        f"\n### {title}\n",
        "| Horizonte | Trades | Win Rate | Avg Ret | Profit Factor | Max DD | Sharpe |",
        "|-----------|--------|----------|---------|---------------|--------|--------|"
    ]
    
    for horizon, metrics in metrics_by_horizon.items():
        count = metrics.get("count", 0)
        wr = f"{metrics.get('win_rate', 0):.1f}%" if metrics.get('win_rate') else "N/A"
        avg = f"{metrics.get('avg_return', 0):.3f}%" if metrics.get('avg_return') else "N/A"
        pf = f"{metrics.get('profit_factor', 0):.2f}" if metrics.get('profit_factor') else "N/A"
        mdd = f"{metrics.get('max_drawdown', 0):.2f}%" if metrics.get('max_drawdown') else "N/A"
        sharpe = f"{metrics.get('sharpe_ratio', 0):.2f}" if metrics.get('sharpe_ratio') else "N/A"
        
        lines.append(f"| {horizon} | {count} | {wr} | {avg} | {pf} | {mdd} | {sharpe} |")
    
    return "\n".join(lines)
