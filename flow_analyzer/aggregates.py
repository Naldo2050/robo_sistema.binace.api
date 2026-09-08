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
    DEFAULT_ROLLING_AGGREGATE_TARGET_TPS,
    DEFAULT_ROLLING_AGGREGATE_ABSOLUTE_MAX_TRADES,
)

logger = logging.getLogger(__name__)


@dataclass
class RollingAggregate:
    """
    Agregação rolling correta por janela (soma incremental + prune com subtração).
    
    Características:
    - O(1) amortizado para add e prune
    - OHLC lazy: recomputa high/low apenas quando necessário
    - Tracking separado de whales e sectors
    - Limite de capacidade derivado de TARGET_TPS com Safety Cap absoluto
    - Telemetria e estados explícitos: WARMING_UP, FULL, CAPACITY_TRUNCATED
    """
    
    window_min: int
    max_trades: Optional[int] = None
    target_tps: int = DEFAULT_ROLLING_AGGREGATE_TARGET_TPS
    absolute_max_trades: int = DEFAULT_ROLLING_AGGREGATE_ABSOLUTE_MAX_TRADES
    
    # Estado interno (inicializado em __post_init__)
    window_ms: int = field(init=False)
    capacity_limit: int = field(init=False)
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
    _first_trade_ts: int = field(init=False)
    _recent_capacity_eviction_ts: int = field(init=False)
    _last_cap_log: int = field(init=False)
    
    def __post_init__(self):
        if self.window_min <= 0:
            raise ValueError("window_min must be greater than 0")
        if self.target_tps <= 0:
            raise ValueError("target_tps must be greater than 0")
        if self.absolute_max_trades <= 0:
            raise ValueError("absolute_max_trades must be greater than 0")
        if self.max_trades is not None and self.max_trades <= 0:
            raise ValueError("max_trades must be greater than 0")

        self.window_ms = int(self.window_min * 60 * 1000)
        if self.max_trades is None:
            calculated_cap = int(self.window_min * 60 * self.target_tps)
            self.max_trades = min(calculated_cap, self.absolute_max_trades)
        else:
            self.max_trades = min(int(self.max_trades), self.absolute_max_trades)
        self.capacity_limit = self.max_trades
        self.reset()
    
    def reset(self) -> None:
        """Reseta todo o estado do aggregate."""
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
        
        # Métricas de tempo e capacidade
        self.last_update = 0
        self.capacity_evictions = 0
        self._first_trade_ts = 0
        self._recent_capacity_eviction_ts = 0
        self._last_cap_log = 0
    
    def _evict_if_needed(self) -> None:
        """
        Eviction com semântica temporal primária:
        - Primário: evictar trades com ts < (last_update - window_ms)
        - Segurança: evictar trades se len(trades) > capacity_limit
        """
        # Critério primário: evictar por tempo (janela real)
        if self.trades and self.last_update:
            cutoff_ms = self.last_update - self.window_ms
            while self.trades and self.trades[0][0] < cutoff_ms:
                self._remove_left()

        # Critério de segurança: cap por capacidade
        hit_cap = False
        while len(self.trades) > self.capacity_limit:
            self.capacity_evictions += 1
            self._recent_capacity_eviction_ts = self.last_update
            self._remove_left()
            hit_cap = True

        if hit_cap:
            if not hasattr(self, '_last_cap_log'):
                self._last_cap_log = 0
            if self.last_update - self._last_cap_log >= 30000:
                oldest_ts = self.trades[0][0] if self.trades else 0
                newest_ts = self.last_update
                duration_sec = (newest_ts - oldest_ts) / 1000.0 if (self.trades and newest_ts > oldest_ts) else 0.0
                trade_count = len(self.trades)
                coverage_pct = round(min(100.0, (duration_sec / (self.window_min * 60)) * 100.0), 1)
                status_str, _ = self.get_window_integrity_status()
                logger.warning(
                    f"RollingAggregate({self.window_min}m): capacity limit hit ({self.capacity_limit} trades), evicting oldest. | "
                    f"event=rolling_aggregate_capacity_truncated | window_min={self.window_min} | trade_count={trade_count} | "
                    f"capacity_limit={self.capacity_limit} | effective_duration_sec={duration_sec:.2f}s | "
                    f"effective_coverage_pct={coverage_pct:.1f}% | capacity_evictions_total={self.capacity_evictions} | "
                    f"window_status={status_str}"
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
        if not self._first_trade_ts:
            self._first_trade_ts = ts
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
    
    def get_window_integrity_status(self) -> Tuple[str, bool]:
        """
        Retorna (window_status, is_integrity_guaranteed).

        Estados:
        - WARMING_UP: a instância ainda não acumulou a duração completa da janela.
        - FULL: a janela cobre >= 99% do tempo nominal sem truncamento ativo.
        - CAPACITY_TRUNCATED: ocorreu eviction por capacidade que encurtou a cobertura temporal.
        """
        if not self.trades or not self.last_update:
            return "WARMING_UP", False

        oldest_ts = self.trades[0][0]
        newest_ts = self.last_update
        effective_duration_ms = newest_ts - oldest_ts
        coverage_ratio = effective_duration_ms / self.window_ms

        # Se a janela atual cobre >= 99% do tempo nominal, está FULL (mesmo com evictions passadas)
        if coverage_ratio >= 0.99:
            return "FULL", True

        # Se atingiu a capacidade máxima OU se sofreu eviction recente sem cobrir 99% da duração
        if len(self.trades) >= self.capacity_limit or (self._recent_capacity_eviction_ts > 0 and self._recent_capacity_eviction_ts >= oldest_ts):
            return "CAPACITY_TRUNCATED", False

        # Se a idade da instância desde o 1º trade for menor que a janela nominal -> WARMING_UP
        observed_age_ms = (newest_ts - self._first_trade_ts) if self._first_trade_ts > 0 else effective_duration_ms
        if observed_age_ms < self.window_ms:
            return "WARMING_UP", False

        return "WARMING_UP", False

    def get_metrics(self, last_price: float = 0.0) -> Dict[str, Any]:
        """
        Retorna métricas rolling completas.
        
        Args:
            last_price: Último preço conhecido
            
        Returns:
            Dict com todas as métricas da janela
        """
        ohlc = self.get_ohlc(last_price)

        oldest_ts = self.trades[0][0] if self.trades else 0
        newest_ts = self.last_update if self.trades else 0
        effective_duration_sec = round((newest_ts - oldest_ts) / 1000.0, 2) if (self.trades and newest_ts > oldest_ts) else 0.0
        effective_coverage_pct = round(min(100.0, (effective_duration_sec / (self.window_min * 60)) * 100.0), 1)
        window_status, is_integrity_guaranteed = self.get_window_integrity_status()
        
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
            },
            'window_status': window_status,
            'effective_duration_sec': effective_duration_sec,
            'effective_coverage_pct': effective_coverage_pct,
            'is_integrity_guaranteed': is_integrity_guaranteed,
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

    Contrato serializável (B-P0-1, idêntico a flow_analyzer.metrics):
    ratio None + ratio_state quando não computável; nunca NaN/Inf/99.
    """
    from .metrics import _finite_volume, _present_volume

    # Extrair volumes de compra/venda (finito ou None; sem coagir p/ 0)
    buy_vol = _present_volume(flow_data, "buy_volume_btc", "buy_volume")
    sell_vol = _present_volume(flow_data, "sell_volume_btc", "sell_volume")

    # Ratio principal (serializável)
    if (buy_vol is None or sell_vol is None
            or buy_vol < 0 or sell_vol < 0):
        main_ratio, ratio_state = None, "invalid"
    elif buy_vol == 0 and sell_vol == 0:
        main_ratio, ratio_state = None, "no_volume"
    elif buy_vol > 0 and sell_vol > 0:
        main_ratio, ratio_state = round(buy_vol / sell_vol, 4), "two_sided"
    elif sell_vol > 0:
        # buy == 0: limite matemático 0.0 (válido)
        main_ratio, ratio_state = 0.0, "sell_only"
    else:
        # buy > 0, sell == 0: infinito não serializa; direção nos volumes
        main_ratio, ratio_state = None, "buy_only"

    # Extrair flows de múltiplas janelas (None = ausente; sem coagir p/ 0)
    net_flow_1m = _finite_volume(flow_data.get("net_flow_1m"))
    net_flow_5m = _finite_volume(flow_data.get("net_flow_5m"))
    net_flow_15m = _finite_volume(flow_data.get("net_flow_15m"))
    total_volume = flow_data.get("total_volume", 0) or flow_data.get("total_volume_btc", 0)

    # Calcular ratios por janela usando net_flow
    # net_flow > 0 = mais compra, net_flow < 0 = mais venda
    ratios = {
        "current": main_ratio,
    }

    # Imbalance por janela (normalizado); chave omitida se net ausente
    if total_volume and total_volume > 0:
        for key, net_flow in (("imbalance_1m", net_flow_1m),
                              ("imbalance_5m", net_flow_5m),
                              ("imbalance_15m", net_flow_15m)):
            if net_flow is not None:
                ratios[key] = round(net_flow / total_volume, 4)

    # Sector ratios (se disponível; mesma regra: só com ambos finitos)
    sector_flow = flow_data.get("sector_flow", {})
    sector_ratios = {}
    for sector_name, sector_data in sector_flow.items():
        if isinstance(sector_data, dict):
            s_buy = _present_volume(sector_data, "buy")
            s_sell = _present_volume(sector_data, "sell")
            if s_buy is None or s_sell is None or (s_buy == 0 and s_sell == 0):
                sector_ratios[sector_name] = None
            elif s_sell > 0:
                # Cap ratio em 10.0 para evitar valores extremos
                sector_ratios[sector_name] = round(min(s_buy / s_sell, 10.0), 4)
            elif s_buy > 0:
                sector_ratios[sector_name] = None  # buy-only: direção nos volumes
            else:
                sector_ratios[sector_name] = 0.0  # sell-only: limite válido

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

    # Classificação do pressure (ratio ausente => pressure ausente)
    if main_ratio is None:
        pressure = None
    elif main_ratio > 2.0:
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
        "ratio_state": ratio_state,
        "ratios": ratios,
        "sector_ratios": sector_ratios,
        "pressure": pressure,
        "flow_trend": trend,
        "buy_volume": round(buy_vol, 4) if buy_vol is not None else None,
        "sell_volume": round(sell_vol, 4) if sell_vol is not None else None,
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

    # --- Fluxo Agressivo (P1-B: disponibilidade explícita, nunca default 50) ---
    # status presente e != observed => unknown/insufficient_data (shape default).
    # status ausente (legado): classifica se pcts finitos (compat), senão unknown.
    import math as _math

    def _finite_pct(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        fv = float(value)
        return fv if _math.isfinite(fv) else None

    agg_status = flow_data.get("aggressive_status")
    if agg_status is not None and agg_status != "observed":
        return dict(default)
    agg_buy_pct = _finite_pct(flow_data.get("aggressive_buy_pct"))
    agg_sell_pct = _finite_pct(flow_data.get("aggressive_sell_pct"))
    if agg_buy_pct is None or agg_sell_pct is None:
        return dict(default)
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