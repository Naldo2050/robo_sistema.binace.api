# flow_analyzer/aggregates.py
"""
Agregação rolling do FlowAnalyzer.

Implementa agregação incremental O(1) com:
- Soma incremental (add)
- Subtração no prune (remove)
- OHLC lazy (recompute quando necessário)
- Sector e whale tracking
"""

import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple

from .constants import (
    DECIMAL_ZERO,
    MAX_AGGREGATE_TRADES,
)
from .utils import lazy_log


@dataclass
class RollingAggregate:
    """
    Agregação rolling correta por janela (soma incremental + prune com subtração).
    
    Características:
    - O(1) amortizado para add e prune
    - OHLC lazy: recomputa high/low apenas quando necessário
    - Tracking separado de whales e sectors
    - Limite de capacidade com eviction
    
    Args:
        window_min: Tamanho da janela em minutos
        max_trades: Limite máximo de trades (eviction se exceder)
    
    Example:
        >>> agg = RollingAggregate(window_min=1, max_trades=1000)
        >>> agg.add_trade({'ts': 1234, 'qty': 0.5, 'price': 50000, ...}, whale_threshold=5.0)
        >>> agg.prune(cutoff_ms=1000)
        >>> metrics = agg.get_metrics(last_price=50100)
    """
    
    window_min: int
    max_trades: int = MAX_AGGREGATE_TRADES
    
    # Estado interno (inicializado em __post_init__)
    window_ms: int = field(init=False)
    trades: deque = field(init=False)
    
    # Somas incrementais
    sum_delta_btc: Decimal = field(init=False)
    sum_delta_usd: Decimal = field(init=False)
    sum_buy_btc: Decimal = field(init=False)
    sum_sell_btc: Decimal = field(init=False)
    sum_buy_usd: Decimal = field(init=False)
    sum_sell_usd: Decimal = field(init=False)
    
    # Whale tracking
    whale_buy: Decimal = field(init=False)
    whale_sell: Decimal = field(init=False)
    
    # Sector tracking
    sector_agg: Dict = field(init=False)
    
    # OHLC lazy
    _open: Optional[float] = field(init=False)
    _close: Optional[float] = field(init=False)
    _high: Optional[float] = field(init=False)
    _low: Optional[float] = field(init=False)
    _dirty_hilo: bool = field(init=False)
    
    # Métricas
    last_update: int = field(init=False)
    capacity_evictions: int = field(init=False)
    
    def __post_init__(self):
        self.window_ms = int(self.window_min * 60 * 1000)
        self.reset()
    
    def reset(self) -> None:
        """Reseta todo o estado do aggregate."""
        # Deque sem maxlen para controle manual
        self.trades = deque()
        
        # Somas
        self.sum_delta_btc = DECIMAL_ZERO
        self.sum_delta_usd = DECIMAL_ZERO
        self.sum_buy_btc = DECIMAL_ZERO
        self.sum_sell_btc = DECIMAL_ZERO
        self.sum_buy_usd = DECIMAL_ZERO
        self.sum_sell_usd = DECIMAL_ZERO
        
        # Whales
        self.whale_buy = DECIMAL_ZERO
        self.whale_sell = DECIMAL_ZERO
        
        # Sectors
        self.sector_agg = defaultdict(lambda: {
            'buy_btc': DECIMAL_ZERO,
            'sell_btc': DECIMAL_ZERO,
            'buy_usd': DECIMAL_ZERO,
            'sell_usd': DECIMAL_ZERO
        })
        
        # OHLC
        self._open = None
        self._close = None
        self._high = None
        self._low = None
        self._dirty_hilo = False
        
        # Métricas
        self.last_update = 0
        self.capacity_evictions = 0
    
    def _evict_if_needed(self) -> None:
        """Evict manual para respeitar max_trades mantendo somas consistentes."""
        while len(self.trades) > self.max_trades:
            self.capacity_evictions += 1
            self._remove_left()
    
    def _remove_left(self) -> None:
        """Remove trade mais antigo e atualiza todas as somas."""
        if not self.trades:
            return
        
        ts, qty, price, delta_btc, side, sector, is_whale = self.trades.popleft()
        
        # Subtrai somas
        self.sum_delta_btc -= delta_btc
        self.sum_delta_usd -= (delta_btc * price)
        
        if side == 'buy':
            self.sum_buy_btc -= qty
            self.sum_buy_usd -= qty * price
        else:
            self.sum_sell_btc -= qty
            self.sum_sell_usd -= qty * price
        
        # Whale
        if is_whale:
            if side == 'buy':
                self.whale_buy -= qty
            else:
                self.whale_sell -= qty
        
        # Sector
        if sector:
            if side == 'buy':
                self.sector_agg[sector]['buy_btc'] -= qty
                self.sector_agg[sector]['buy_usd'] -= qty * price
            else:
                self.sector_agg[sector]['sell_btc'] -= qty
                self.sector_agg[sector]['sell_usd'] -= qty * price
        
        # OHLC: marca dirty se removemos high ou low
        p = float(price)
        if self._high is not None and abs(p - self._high) < 1e-12:
            self._dirty_hilo = True
        if self._low is not None and abs(p - self._low) < 1e-12:
            self._dirty_hilo = True
        
        # Atualiza open/close
        if not self.trades:
            self._open = self._close = self._high = self._low = None
            self._dirty_hilo = False
        else:
            self._open = float(self.trades[0][2])
            self._close = float(self.trades[-1][2])
    
    def prune(self, cutoff_ms: int) -> int:
        """
        Remove trades antigos (ts < cutoff_ms) subtraindo das somas.
        
        Args:
            cutoff_ms: Timestamp de corte
            
        Returns:
            Número de trades removidos
        """
        removed = 0
        while self.trades and self.trades[0][0] < cutoff_ms:
            self._remove_left()
            removed += 1
        return removed
    
    def add_trade(self, trade: Dict[str, Any], whale_threshold: float) -> bool:
        """
        Adiciona trade na janela (incremental) + atualiza OHLC.
        
        Args:
            trade: Dict com ts, qty, price, delta_btc, side, sector
            whale_threshold: Threshold para classificar como whale
            
        Returns:
            True se adicionado, False se rejeitado (out-of-order)
        """
        ts = int(trade['ts'])
        
        # Proteção: não aceite out-of-order dentro do aggregate
        if self.last_update and ts < self.last_update:
            return False
        
        qty = Decimal(str(trade['qty']))
        price = Decimal(str(trade['price']))
        delta_btc = Decimal(str(trade['delta_btc']))
        side = trade['side']
        sector = trade.get('sector')
        
        is_whale = float(qty) >= whale_threshold
        
        # Append (ts, qty, price, delta_btc, side, sector, is_whale)
        self.trades.append((ts, qty, price, delta_btc, side, sector, is_whale))
        self.last_update = ts
        
        # Atualiza somas
        self.sum_delta_btc += delta_btc
        self.sum_delta_usd += (delta_btc * price)
        
        if side == 'buy':
            self.sum_buy_btc += qty
            self.sum_buy_usd += qty * price
        else:
            self.sum_sell_btc += qty
            self.sum_sell_usd += qty * price
        
        # Whale
        if is_whale:
            if side == 'buy':
                self.whale_buy += qty
            else:
                self.whale_sell += qty
        
        # Sector
        if sector:
            if side == 'buy':
                self.sector_agg[sector]['buy_btc'] += qty
                self.sector_agg[sector]['buy_usd'] += qty * price
            else:
                self.sector_agg[sector]['sell_btc'] += qty
                self.sector_agg[sector]['sell_usd'] += qty * price
        
        # OHLC
        p = float(price)
        if self._open is None:
            self._open = p
        self._close = p
        
        if self._high is None or p > self._high:
            self._high = p
        if self._low is None or p < self._low:
            self._low = p
        
        # Eviction se necessário
        self._evict_if_needed()
        
        return True
    
    def _recompute_hilo_if_dirty(self) -> None:
        """Recomputa high/low se marcado como dirty."""
        if not self.trades:
            return
        
        if self._dirty_hilo:
            prices = [float(x[2]) for x in self.trades]
            self._high = max(prices)
            self._low = min(prices)
            self._dirty_hilo = False
    
    def get_ohlc(self, last_price: float = 0.0) -> Tuple[float, float, float, float]:
        """
        Retorna OHLC da janela.
        
        Args:
            last_price: Preço para usar se janela vazia
            
        Returns:
            Tuple (open, high, low, close)
        """
        if not self.trades:
            if last_price > 0:
                return (last_price, last_price, last_price, last_price)
            return (0.0, 0.0, 0.0, 0.0)
        
        self._recompute_hilo_if_dirty()
        return (self._open, self._high, self._low, self._close)
    
    def get_metrics(self, last_price: float = 0.0) -> Dict[str, Any]:
        """
        Retorna métricas rolling completas.
        
        Args:
            last_price: Último preço conhecido
            
        Returns:
            Dict com todas as métricas da janela
        """
        ohlc = self.get_ohlc(last_price)
        
        return {
            'sum_delta_btc': float(self.sum_delta_btc),
            'sum_delta_usd': float(self.sum_delta_usd),
            'sum_buy_btc': float(self.sum_buy_btc),
            'sum_sell_btc': float(self.sum_sell_btc),
            'sum_buy_usd': float(self.sum_buy_usd),
            'sum_sell_usd': float(self.sum_sell_usd),
            'whale_buy': float(self.whale_buy),
            'whale_sell': float(self.whale_sell),
            'whale_delta': float(self.whale_buy - self.whale_sell),
            'capacity_evictions': self.capacity_evictions,
            'ohlc': ohlc,
            'last_update': self.last_update,
            'trade_count': len(self.trades),
            'sector_agg': {
                k: {
                    'buy_btc': float(v['buy_btc']),
                    'sell_btc': float(v['sell_btc']),
                    'buy_usd': float(v['buy_usd']),
                    'sell_usd': float(v['sell_usd']),
                    'delta_btc': float(v['buy_btc'] - v['sell_btc']),
                }
                for k, v in self.sector_agg.items()
                if any(v[x] != DECIMAL_ZERO for x in ['buy_btc', 'sell_btc'])
            }
        }
    
    def __len__(self) -> int:
        return len(self.trades)
    
    def __repr__(self) -> str:
        return (
            f"RollingAggregate(window_min={self.window_min}, "
            f"trades={len(self.trades)}, "
            f"delta_btc={float(self.sum_delta_btc):.4f})"
        )