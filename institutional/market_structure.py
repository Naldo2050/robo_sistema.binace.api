# institutional/market_structure.py
# -*- coding: utf-8 -*-
"""
Market Structure Detector: Break of Structure (BOS) & Liquidity Sweep.
Fase P1.3C (Correção Semântica & Provenance Canônica).

Implementa:
1. Detecção rigorosa de Swing High / Swing Low com janela de confirmação canônica (left_bars=2, right_bars=2).
2. Garantia anti-lookahead: swings só são ativados no timestamp em que ficam confirmados (T + right_bars).
3. Break of Structure (BOS) bullish e bearish com confirmação de fechamento de candle.
4. Liquidity Sweep (buy-side / sell-side / both) caracterizado por excursão e reclaim/rejeição.
5. Exclusão mútua estrita entre BOS e Sweep para o mesmo candle e mesmo nível.
6. Timeframe canônico explícito ("1m") e proveniência temporal completa com event_id determinístico.
7. Schema versioning para isolamento de dados históricos pré-fix.

RESTRIÇÃO ARQUITETURAL:
Módulo estritamente CONTEXT-ONLY. Não gera sinais direcionais de trade, não altera position sizing nem risco.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("MarketStructure")

# Versão canônica do schema de estrutura de mercado
MARKET_STRUCTURE_SCHEMA_VERSION = "1.1.0"


class StructurePointType(str, Enum):
    HIGH = "high"
    LOW = "low"


class BOSType(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class SweepType(str, Enum):
    BUY_SIDE = "buy_side"    # Varredura de swing high (liquidez de compra / stops de shorts)
    SELL_SIDE = "sell_side"  # Varredura de swing low (liquidez de venda / stops de longs)
    BOTH = "both"            # Varredura simultânea de topo e fundo no mesmo candle largo


@dataclass
class SwingLevel:
    """Ponto de inflexão estrutural confirmado."""
    timestamp_ms: int
    price: float
    point_type: StructurePointType
    candle_index: int
    confirmed_at_ms: int
    confirmed_index: int
    is_broken: bool = False
    is_swept: bool = False
    schema_version: str = MARKET_STRUCTURE_SCHEMA_VERSION

    @property
    def event_id(self) -> str:
        return f"SWING:{self.point_type.value}:{round(self.price, 2)}:{self.timestamp_ms}:{self.confirmed_at_ms}:{self.schema_version}"


@dataclass
class BOSEvent:
    """Evento de Break of Structure confirmado por fechamento."""
    type: BOSType
    level: float
    break_price: float
    swing_timestamp_ms: int
    confirmed_at_ms: int
    candle_index: int
    timeframe: str = "1m"
    strength_pct: float = 0.0
    symbol: str = "BTCUSDT"
    schema_version: str = MARKET_STRUCTURE_SCHEMA_VERSION

    @property
    def event_id(self) -> str:
        return f"{self.symbol}:{self.timeframe}:BOS_{self.type.value.upper()}:{round(self.level, 2)}:{self.swing_timestamp_ms}:{self.confirmed_at_ms}:{self.schema_version}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["event_id"] = self.event_id
        return d


@dataclass
class LiquiditySweepEvent:
    """Evento de Liquidity Sweep (excursão + rejeição/reclaim)."""
    type: SweepType
    level: float
    wick_price: float
    close_price: float
    excursion_fraction: float  # abs(wick - level) / level
    swing_timestamp_ms: int
    confirmed_at_ms: int
    candle_index: int
    timeframe: str = "1m"
    symbol: str = "BTCUSDT"
    schema_version: str = MARKET_STRUCTURE_SCHEMA_VERSION

    @property
    def event_id(self) -> str:
        return f"{self.symbol}:{self.timeframe}:SWEEP_{self.type.value.upper()}:{round(self.level, 2)}:{self.swing_timestamp_ms}:{self.confirmed_at_ms}:{self.schema_version}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["event_id"] = self.event_id
        return d


@dataclass
class MarketStructureResult:
    """Resultado consolidado da análise de estrutura de mercado."""
    active_bos: Optional[BOSEvent] = None
    active_sweep: Optional[LiquiditySweepEvent] = None
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    last_swing_high_ts: Optional[int] = None
    last_swing_low_ts: Optional[int] = None
    confirmed_swings_count: int = 0
    timeframe: str = "1m"
    left_bars: int = 2
    right_bars: int = 2
    schema_version: str = MARKET_STRUCTURE_SCHEMA_VERSION
    status: str = "VALID"
    observed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bos": self.active_bos.to_dict() if self.active_bos else None,
            "sweep": self.active_sweep.to_dict() if self.active_sweep else None,
            "last_swing_high": self.last_swing_high,
            "last_swing_low": self.last_swing_low,
            "last_swing_high_ts": self.last_swing_high_ts,
            "last_swing_low_ts": self.last_swing_low_ts,
            "confirmed_swings_count": self.confirmed_swings_count,
            "timeframe": self.timeframe,
            "left_bars": self.left_bars,
            "right_bars": self.right_bars,
            "schema_version": self.schema_version,
            "status": self.status,
            "observed_at": self.observed_at,
        }


class MarketStructureDetector:
    """
    Detector determinístico de Swings, BOS e Liquidity Sweeps.
    Implementação pura, sem estado global ou chamadas de rede.
    """

    def __init__(
        self,
        left_bars: int = 2,
        right_bars: int = 2,
        timeframe: str = "1m",
        max_active_swings: int = 20,
        symbol: str = "BTCUSDT",
    ):
        if left_bars < 1 or right_bars < 1:
            raise ValueError("left_bars e right_bars devem ser >= 1")

        self.left_bars = left_bars
        self.right_bars = right_bars
        self.timeframe = timeframe
        self.max_active_swings = max_active_swings
        self.symbol = symbol

    def analyze_candles(
        self,
        candles: List[Dict[str, Any]],
    ) -> MarketStructureResult:
        """
        Analisa uma lista cronológica de candles fechados e detecta a estrutura no último candle.
        
        Garantias Anti-Lookahead:
        - Para cada candle T, apenas swings cuja confirmação (center + right_bars) <= T
          são visíveis e ativos.
        """
        n = len(candles)
        min_required = self.left_bars + self.right_bars + 1
        if n < min_required:
            return MarketStructureResult(
                status="INSUFFICIENT_DATA",
                timeframe=self.timeframe,
                left_bars=self.left_bars,
                right_bars=self.right_bars,
            )

        # 1. Extração defensiva de dados com validação de monotonicidade
        ts_list: List[int] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []

        last_ts = -1
        is_monotonic = True

        for c in candles:
            if isinstance(c, dict):
                ts = int(c.get("open_time") or c.get("timestamp") or c.get("t", 0))
                h = float(c.get("high") or c.get("h", 0))
                l = float(c.get("low") or c.get("l", 0))
                close = float(c.get("close") or c.get("c", 0))
            elif isinstance(c, (list, tuple)) and len(c) >= 5:
                ts = int(c[0])
                h = float(c[2])
                l = float(c[3])
                close = float(c[4])
            else:
                continue

            # Validação contra NaN/Inf/Zero
            if not (math.isfinite(h) and math.isfinite(l) and math.isfinite(close)):
                continue
            if h <= 0 or l <= 0 or close <= 0 or h < l:
                continue

            if ts > 0:
                if ts <= last_ts:
                    is_monotonic = False
                last_ts = ts

            ts_list.append(ts)
            highs.append(h)
            lows.append(l)
            closes.append(close)

        valid_n = len(closes)
        if valid_n < min_required:
            return MarketStructureResult(
                status="INSUFFICIENT_DATA",
                timeframe=self.timeframe,
                left_bars=self.left_bars,
                right_bars=self.right_bars,
            )

        # 2. Rastreamento cronológico de Swings, BOS e Sweeps (Zero Lookahead)
        confirmed_swings: List[SwingLevel] = []
        last_bos: Optional[BOSEvent] = None
        last_sweep: Optional[LiquiditySweepEvent] = None

        for curr_idx in range(valid_n):
            curr_ts = ts_list[curr_idx]
            curr_h = highs[curr_idx]
            curr_l = lows[curr_idx]
            curr_c = closes[curr_idx]

            # 2.1. Verificar se um novo swing ficou confirmado NESTE candle curr_idx
            # O candle central de um swing é (curr_idx - right_bars)
            center_idx = curr_idx - self.right_bars
            if center_idx >= self.left_bars:
                center_h = highs[center_idx]
                center_l = lows[center_idx]
                center_ts = ts_list[center_idx]

                # Confirmação de Swing High
                is_swing_high = True
                has_lower_neighbor = False
                for offset in range(-self.left_bars, self.right_bars + 1):
                    if offset != 0:
                        neighbor_h = highs[center_idx + offset]
                        if neighbor_h > center_h:
                            is_swing_high = False
                            break
                        if neighbor_h < center_h:
                            has_lower_neighbor = True

                if is_swing_high and has_lower_neighbor:
                    confirmed_swings.append(
                        SwingLevel(
                            timestamp_ms=center_ts,
                            price=center_h,
                            point_type=StructurePointType.HIGH,
                            candle_index=center_idx,
                            confirmed_at_ms=curr_ts,
                            confirmed_index=curr_idx,
                            schema_version=MARKET_STRUCTURE_SCHEMA_VERSION,
                        )
                    )

                # Confirmação de Swing Low
                is_swing_low = True
                has_higher_neighbor = False
                for offset in range(-self.left_bars, self.right_bars + 1):
                    if offset != 0:
                        neighbor_l = lows[center_idx + offset]
                        if neighbor_l < center_l:
                            is_swing_low = False
                            break
                        if neighbor_l > center_l:
                            has_higher_neighbor = True

                if is_swing_low and has_higher_neighbor:
                    confirmed_swings.append(
                        SwingLevel(
                            timestamp_ms=center_ts,
                            price=center_l,
                            point_type=StructurePointType.LOW,
                            candle_index=center_idx,
                            confirmed_at_ms=curr_ts,
                            confirmed_index=curr_idx,
                            schema_version=MARKET_STRUCTURE_SCHEMA_VERSION,
                        )
                    )

            # 2.2. Avaliar interações do candle curr_idx com swings confirmados ANTERIORMENTE
            eligible_swings = [
                s for s in confirmed_swings
                if s.confirmed_index < curr_idx and not s.is_broken
            ]

            curr_buy_sweep: Optional[LiquiditySweepEvent] = None
            curr_sell_sweep: Optional[LiquiditySweepEvent] = None

            # Verificar interações com Swing Highs
            for sw in [s for s in eligible_swings if s.point_type == StructurePointType.HIGH]:
                if curr_c > sw.price:
                    # 1. BOS Bullish: fechamento confirmado acima do Swing High
                    sw.is_broken = True
                    strength = round((curr_c - sw.price) / sw.price, 4)
                    last_bos = BOSEvent(
                        type=BOSType.BULLISH,
                        level=round(sw.price, 2),
                        break_price=round(curr_c, 2),
                        swing_timestamp_ms=sw.timestamp_ms,
                        confirmed_at_ms=curr_ts,
                        candle_index=curr_idx,
                        timeframe=self.timeframe,
                        strength_pct=strength,
                        symbol=self.symbol,
                        schema_version=MARKET_STRUCTURE_SCHEMA_VERSION,
                    )
                elif curr_h > sw.price and curr_c <= sw.price:
                    # 2. Buy-Side Liquidity Sweep: wick violou o topo, mas candle fechou abaixo
                    sw.is_swept = True
                    excursion = round((curr_h - sw.price) / sw.price, 4)
                    curr_buy_sweep = LiquiditySweepEvent(
                        type=SweepType.BUY_SIDE,
                        level=round(sw.price, 2),
                        wick_price=round(curr_h, 2),
                        close_price=round(curr_c, 2),
                        excursion_fraction=excursion,
                        swing_timestamp_ms=sw.timestamp_ms,
                        confirmed_at_ms=curr_ts,
                        candle_index=curr_idx,
                        timeframe=self.timeframe,
                        symbol=self.symbol,
                        schema_version=MARKET_STRUCTURE_SCHEMA_VERSION,
                    )

            # Verificar interações com Swing Lows
            for sw in [s for s in eligible_swings if s.point_type == StructurePointType.LOW]:
                if curr_c < sw.price:
                    # 1. BOS Bearish: fechamento confirmado abaixo do Swing Low
                    sw.is_broken = True
                    strength = round((sw.price - curr_c) / sw.price, 4)
                    last_bos = BOSEvent(
                        type=BOSType.BEARISH,
                        level=round(sw.price, 2),
                        break_price=round(curr_c, 2),
                        swing_timestamp_ms=sw.timestamp_ms,
                        confirmed_at_ms=curr_ts,
                        candle_index=curr_idx,
                        timeframe=self.timeframe,
                        strength_pct=strength,
                        symbol=self.symbol,
                        schema_version=MARKET_STRUCTURE_SCHEMA_VERSION,
                    )
                elif curr_l < sw.price and curr_c >= sw.price:
                    # 2. Sell-Side Liquidity Sweep: wick violou o fundo, mas candle fechou acima
                    sw.is_swept = True
                    excursion = round((sw.price - curr_l) / sw.price, 4)
                    curr_sell_sweep = LiquiditySweepEvent(
                        type=SweepType.SELL_SIDE,
                        level=round(sw.price, 2),
                        wick_price=round(curr_l, 2),
                        close_price=round(curr_c, 2),
                        excursion_fraction=excursion,
                        swing_timestamp_ms=sw.timestamp_ms,
                        confirmed_at_ms=curr_ts,
                        candle_index=curr_idx,
                        timeframe=self.timeframe,
                        symbol=self.symbol,
                        schema_version=MARKET_STRUCTURE_SCHEMA_VERSION,
                    )

            # Resolução Não-Viesada de Double Sweep (Candle Largo)
            if curr_buy_sweep and curr_sell_sweep:
                # Double Sweep: o mesmo candle varreu tanto o topo quanto o fundo e fechou dentro do range
                last_sweep = LiquiditySweepEvent(
                    type=SweepType.BOTH,
                    level=curr_buy_sweep.level,
                    wick_price=curr_buy_sweep.wick_price,
                    close_price=curr_c,
                    excursion_fraction=max(curr_buy_sweep.excursion_fraction, curr_sell_sweep.excursion_fraction),
                    swing_timestamp_ms=curr_buy_sweep.swing_timestamp_ms,
                    confirmed_at_ms=curr_ts,
                    candle_index=curr_idx,
                    timeframe=self.timeframe,
                    symbol=self.symbol,
                    schema_version=MARKET_STRUCTURE_SCHEMA_VERSION,
                )
            elif curr_buy_sweep:
                last_sweep = curr_buy_sweep
            elif curr_sell_sweep:
                last_sweep = curr_sell_sweep

        # 3. Extrair os últimos swings de referência
        high_swings = [s for s in confirmed_swings if s.point_type == StructurePointType.HIGH]
        low_swings = [s for s in confirmed_swings if s.point_type == StructurePointType.LOW]

        last_sh = high_swings[-1].price if high_swings else None
        last_sh_ts = high_swings[-1].timestamp_ms if high_swings else None
        last_sl = low_swings[-1].price if low_swings else None
        last_sl_ts = low_swings[-1].timestamp_ms if low_swings else None

        # Filtrar relevância temporal: se BOS ou Sweep ocorreu há mais de 20 barras, não é o foco imediato
        active_bos = last_bos if (last_bos and (valid_n - 1 - last_bos.candle_index) <= 20) else None
        active_sweep = last_sweep if (last_sweep and (valid_n - 1 - last_sweep.candle_index) <= 20) else None

        return MarketStructureResult(
            active_bos=active_bos,
            active_sweep=active_sweep,
            last_swing_high=round(last_sh, 2) if last_sh is not None else None,
            last_swing_low=round(last_sl, 2) if last_sl is not None else None,
            last_swing_high_ts=last_sh_ts,
            last_swing_low_ts=last_sl_ts,
            confirmed_swings_count=len(confirmed_swings),
            timeframe=self.timeframe,
            left_bars=self.left_bars,
            right_bars=self.right_bars,
            schema_version=MARKET_STRUCTURE_SCHEMA_VERSION,
            status="VALID",
        )
