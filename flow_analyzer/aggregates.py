# flow_analyzer/aggregates.py
"""
Agregação rolling do FlowAnalyzer.

Implementa agregação incremental O(1) com:
- Soma incremental (add)
- Subtração no prune (remove)
- OHLC lazy (recompute quando necessário)
- Sector e whale tracking
"""

from collections import deque, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
import logging
from typing import Dict, Any, Optional, Tuple

from .constants import (
    DECIMAL_ZERO,
    MAX_AGGREGATE_TRADES,
)

logger = logging.getLogger(__name__)

# Cap duro de contagem (segurança contra flash crash com volume anormal).
# O critério PRIMÁRIO de eviction é temporal (janela real); este cap só
# protege a memória em cenários de volume absurdo (ex: >5000 trades/60s).
MAX_TRADES_HARD_CAP = 5000


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
        self._last_cap_log = 0
    
    def _evict_if_needed(self) -> None:
        """
        Eviction HÍBRIDA:
        - Critério primário: temporal (janela real, ex: 60000ms para 1m).
          Garante que a janela cobre ~60s reais de dados mesmo em mercado
          acelerado (>10 trades/s), em vez de truncar por contagem (~15s).
        - Critério de segurança: cap duro de contagem (MAX_TRADES_HARD_CAP),
          só para não estourar memória em flash crash com volume anormal.
        Mantém somas consistentes (toda remoção passa por _remove_left).
        """
        # Critério primário: evictar por tempo (janela real)
        if self.trades and self.last_update:
            cutoff_ms = self.last_update - self.window_ms
            while self.trades and self.trades[0][0] < cutoff_ms:
                self.capacity_evictions += 1
                self._remove_left()

        # Critério de segurança: cap duro de contagem
        count_cap = max(self.max_trades, MAX_TRADES_HARD_CAP)
        while len(self.trades) > count_cap:
            self.capacity_evictions += 1
            self._remove_left()
            # logar 1x por segundo para não spam
            if not hasattr(self, '_last_cap_log') or \
               self.last_update - self._last_cap_log > 1000:
                logger.warning(
                    "RollingAggregate(%dm): hard cap hit (%d trades), "
                    "evicting oldest. Market volume abnormally high.",
                    self.window_min, count_cap
                )
                self._last_cap_log = self.last_update
    
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
    
    def get_ohlc(self, last_price: float = 0.0) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
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


# ==============================================================================
# BUY/SELL RATIO CALCULATOR
# ==============================================================================

def calculate_buy_sell_ratios(flow_data: dict) -> dict:
    """
    Calcula Buy/Sell Ratios em múltiplas janelas temporais.
    
    Ratio > 1.0 = mais compra que venda (bullish pressure)
    Ratio < 1.0 = mais venda que compra (bearish pressure)
    Ratio = 1.0 = equilibrado
    
    Também detecta tendência do ratio (aceleração/desaceleração).
    
    Args:
        flow_data: Dict com dados de fluxo. Espera chaves como:
            - buy_volume ou buy_volume_btc
            - sell_volume ou sell_volume_btc
            - Opcionalmente: net_flow_1m, net_flow_5m, net_flow_15m
            - Opcionalmente: sector_flow com retail/mid/whale
            
    Returns:
        Dict com ratios por janela e análise de tendência.
    """
    # Extrair volumes de compra/venda
    buy_vol = (
        flow_data.get("buy_volume_btc")
        or flow_data.get("buy_volume")
        or 0
    )
    sell_vol = (
        flow_data.get("sell_volume_btc")
        or flow_data.get("sell_volume")
        or 0
    )

    # Ratio principal
    if sell_vol > 0:
        main_ratio = round(buy_vol / sell_vol, 4)
    else:
        main_ratio = 1.0 if buy_vol == 0 else 99.0

    # Extrair flows de múltiplas janelas
    net_flow_1m = flow_data.get("net_flow_1m", 0)
    net_flow_5m = flow_data.get("net_flow_5m", 0)
    net_flow_15m = flow_data.get("net_flow_15m", 0)
    total_volume = flow_data.get("total_volume", 0) or flow_data.get("total_volume_btc", 0)

    # Calcular ratios por janela usando net_flow
    # net_flow > 0 = mais compra, net_flow < 0 = mais venda
    ratios = {
        "current": main_ratio,
    }

    # Imbalance por janela (normalizado)
    if total_volume and total_volume > 0:
        ratios["imbalance_1m"] = round(net_flow_1m / total_volume, 4) if net_flow_1m else 0
        ratios["imbalance_5m"] = round(net_flow_5m / total_volume, 4) if net_flow_5m else 0
        ratios["imbalance_15m"] = round(net_flow_15m / total_volume, 4) if net_flow_15m else 0

    # Sector ratios (se disponível)
    sector_flow = flow_data.get("sector_flow", {})
    sector_ratios = {}
    for sector_name, sector_data in sector_flow.items():
        if isinstance(sector_data, dict):
            s_buy = sector_data.get("buy", 0)
            s_sell = sector_data.get("sell", 0)
            if s_sell > 0:
                # Cap ratio em 10.0 para evitar valores extremos
                sector_ratios[sector_name] = round(min(s_buy / s_sell, 10.0), 4)
            elif s_buy > 0:
                sector_ratios[sector_name] = 10.0  # buy-only: cap máximo
            else:
                sector_ratios[sector_name] = 1.0

    # Detecção de tendência do fluxo (imbalance normalizado por janela)
    imbalance_1m = ratios.get("imbalance_1m", 0)
    imbalance_5m = ratios.get("imbalance_5m", 0)

    if not imbalance_1m and not imbalance_5m:
        trend = "insufficient_data"
    else:
        recent = abs(imbalance_1m)
        medium = abs(imbalance_5m)
        THRESHOLD = 0.05
        if recent > medium + THRESHOLD:
            flow_trend = "accelerating"
        elif recent < medium - THRESHOLD:
            flow_trend = "decelerating"
        else:
            flow_trend = "stable"
        direction = "selling" if imbalance_1m < 0 else "buying"
        trend = f"{flow_trend}_{direction}"

    # Classificação do pressure
    if main_ratio > 2.0:
        pressure = "STRONG_BUY"
    elif main_ratio > 1.3:
        pressure = "MODERATE_BUY"
    elif main_ratio > 1.05:
        pressure = "SLIGHT_BUY"
    elif main_ratio > 0.95:
        pressure = "NEUTRAL"
    elif main_ratio > 0.7:
        pressure = "SLIGHT_SELL"
    elif main_ratio > 0.5:
        pressure = "MODERATE_SELL"
    else:
        pressure = "STRONG_SELL"

    return {
        "buy_sell_ratio": main_ratio,
        "ratios": ratios,
        "sector_ratios": sector_ratios,
        "pressure": pressure,
        "flow_trend": trend,
        "buy_volume": round(buy_vol, 4),
        "sell_volume": round(sell_vol, 4),
    }


def analyze_passive_aggressive_flow(
    flow_data: dict,
    orderbook_data: Optional[dict] = None,
) -> dict:
    """
    Analisa relação entre fluxo agressivo (taker) e passivo (maker).
    
    Agressivo = Taker (market orders que removem liquidez do book)
    Passivo   = Maker (limit orders que adicionam liquidez ao book)
    
    Cenários chave:
      - Agressivos compram + Passivos compram = Tendência forte de alta
      - Agressivos compram + Passivos vendem  = Absorção (possível reversão)
      - Agressivos vendem  + Passivos vendem  = Tendência forte de baixa
      - Agressivos vendem  + Passivos compram = Absorção (possível reversão)
    
    Args:
        flow_data: Dict com dados de fluxo de ordens.
            Espera: {
                "aggressive_buy_pct": float,   # % de volume agressivo comprador
                "aggressive_sell_pct": float,   # % de volume agressivo vendedor
                "buy_volume_btc": float,
                "sell_volume_btc": float,
                "flow_imbalance": float,        # -1 a +1
                "net_flow_1m": float,
            }
        orderbook_data: Dados do order book para inferir fluxo passivo.
            Espera: {
                "bid_depth_usd": float,
                "ask_depth_usd": float,
                "imbalance": float,  # > 0 = mais bids (passivo comprador)
            }
    
    Returns:
        Dict com análise agressivo/passivo e sinal composto.
    """
    default = {
        "aggressive": {"dominance": "unknown", "buy_pct": 0, "sell_pct": 0, "net": 0},
        "passive": {"dominance": "unknown", "inference": "no_data"},
        "composite": {"agreement": None, "signal": "insufficient_data"},
        "status": "no_data",
    }

    if not flow_data or not isinstance(flow_data, dict):
        return default

    # --- Fluxo Agressivo (direto dos dados de trades) ---
    agg_buy_pct = flow_data.get("aggressive_buy_pct", 50)
    agg_sell_pct = flow_data.get("aggressive_sell_pct", 50)
    buy_vol = flow_data.get("buy_volume_btc", 0) or flow_data.get("buy_volume", 0)
    sell_vol = flow_data.get("sell_volume_btc", 0) or flow_data.get("sell_volume", 0)
    flow_imb = flow_data.get("flow_imbalance", 0)

    agg_net = agg_buy_pct - agg_sell_pct
    agg_dominance = "buyers" if agg_net > 2 else "sellers" if agg_net < -2 else "balanced"

    aggressive = {
        "buy_pct": round(agg_buy_pct, 2),
        "sell_pct": round(agg_sell_pct, 2),
        "net_pct": round(agg_net, 2),
        "dominance": agg_dominance,
        "buy_volume": round(buy_vol, 4),
        "sell_volume": round(sell_vol, 4),
    }

    # --- Fluxo Passivo (inferido do order book) ---
    passive = {
        "dominance": "unknown",
        "inference": "no_orderbook_data",
        "bid_depth": 0,
        "ask_depth": 0,
    }

    if orderbook_data and isinstance(orderbook_data, dict):
        bid_depth = orderbook_data.get("bid_depth_usd", 0)
        ask_depth = orderbook_data.get("ask_depth_usd", 0)
        ob_imbalance = orderbook_data.get("imbalance", 0)

        # Bid depth > ask depth = mais limit buys (passivo comprador)
        total_depth = bid_depth + ask_depth
        if total_depth > 0:
            passive_ratio = bid_depth / total_depth
            # 🔄 BUG #9: Se ob_imbalance veio zerado mas temos depth, calculamos na hora
            if abs(ob_imbalance) < 1e-6:
                ob_imbalance = (bid_depth - ask_depth) / total_depth
        else:
            passive_ratio = 0.5
            ob_imbalance = 0.0

        passive_dominance = (
            "buyers" if passive_ratio > 0.55
            else "sellers" if passive_ratio < 0.45
            else "balanced"
        )

        passive = {
            "dominance": passive_dominance,
            "inference": "from_orderbook_depth",
            "bid_depth": round(bid_depth, 2),
            "ask_depth": round(ask_depth, 2),
            "bid_ratio": round(passive_ratio, 4),
            "ob_imbalance": round(ob_imbalance, 4),
        }

    # --- Análise Composta ---
    agg_buying = agg_dominance == "buyers"
    agg_selling = agg_dominance == "sellers"
    passive_buying = passive["dominance"] == "buyers"
    passive_selling = passive["dominance"] == "sellers"

    if passive["dominance"] == "unknown":
        agreement = None
        signal = "passive_unknown"
        interpretation = "Cannot determine passive flow - orderbook data needed"
    elif agg_buying and passive_buying:
        agreement = True
        signal = "strong_bullish"
        interpretation = "Both aggressive and passive buyers active - strong upward trend"
    elif agg_selling and passive_selling:
        agreement = True
        signal = "strong_bearish"
        interpretation = "Both aggressive and passive sellers active - strong downward trend"
    elif agg_buying and passive_selling:
        agreement = False
        signal = "buy_absorption"
        interpretation = "Aggressive buyers hitting passive sell walls - potential reversal or breakout"
    elif agg_selling and passive_buying:
        agreement = False
        signal = "sell_absorption"
        interpretation = "Aggressive sellers hitting passive buy walls - potential reversal or breakdown"
    elif agg_dominance == "balanced" and passive["dominance"] == "balanced":
        agreement = True
        signal = "neutral_balanced"
        interpretation = "Both sides balanced - range/consolidation expected"
    else:
        agreement = None
        signal = "mixed"
        interpretation = "Mixed signals between aggressive and passive flow"

    composite = {
        "agreement": agreement,
        "signal": signal,
        "interpretation": interpretation,
        "conviction": (
            "HIGH" if agreement is True and (agg_buying or agg_selling)
            else "MEDIUM" if agreement is False
            else "LOW"
        ),
    }

    return {
        "aggressive": aggressive,
        "passive": passive,
        "composite": composite,
        "status": "success",
    }