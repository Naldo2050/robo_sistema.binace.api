# institutional/smart_money.py
"""
Smart Money Concepts (SMC) / ICT

Detecta:
- Fair Value Gaps (FVG) — desequilíbrios de preço
- Break of Structure (BOS) — mudanças de estrutura
- Order Blocks — zonas de entrada institucional
- Liquidity Sweeps — caça aos stops

Método #48 do Arsenal Institucional.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InvalidParameterError,
    Side,
    Signal,
    SignalStrength,
)


@dataclass
class Candle:
    """Representação de uma vela."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        return self.high - self.low

    @property
    def body_ratio(self) -> float:
        """Proporção do corpo em relação ao range."""
        if self.range_size > 0:
            return self.body_size / self.range_size
        return 0.0


@dataclass
class FairValueGap:
    """Fair Value Gap detectado."""
    timestamp: float
    gap_type: str  # "bullish" or "bearish"
    top: float
    bottom: float
    candle_index: int
    filled: bool = False
    fill_percentage: float = 0.0

    @property
    def size(self) -> float:
        return self.top - self.bottom

    @property
    def midpoint(self) -> float:
        return (self.top + self.bottom) / 2.0


@dataclass
class OrderBlock:
    """Order Block detectado."""
    timestamp: float
    block_type: str  # "bullish" or "bearish"
    high: float
    low: float
    volume: float
    candle_index: int
    tested: bool = False
    broken: bool = False

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass
class StructurePoint:
    """Ponto de estrutura (swing high/low)."""
    timestamp: float
    price: float
    point_type: str  # "high" or "low"
    candle_index: int


@dataclass
class LiquiditySweep:
    """Varredura de liquidez detectada."""
    timestamp: float
    sweep_type: str  # "buy_side" or "sell_side"
    level_swept: float
    wick_price: float
    close_price: float
    candle_index: int


class SmartMoneyAnalyzer:
    """
    Analisador de Smart Money Concepts.

    Implementa conceitos ICT/SMC para detectar
    atividade institucional via estrutura de preço.
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        fvg_min_size_pct: float = 0.01,
        order_block_body_ratio: float = 0.6,
        max_candles: int = 500,
    ):
        if swing_lookback < 2:
            raise InvalidParameterError("swing_lookback must be >= 2")

        self.swing_lookback = swing_lookback
        self.fvg_min_size_pct = fvg_min_size_pct
        self.ob_body_ratio = order_block_body_ratio
        self.max_candles = max_candles

        self.candles: deque[Candle] = deque(maxlen=max_candles)
        self.fvgs: list[FairValueGap] = []
        self.order_blocks: list[OrderBlock] = []
        self.structure_points: list[StructurePoint] = []
        self.liquidity_sweeps: list[LiquiditySweep] = []

    @property
    def candle_count(self) -> int:
        return len(self.candles)

    def add_candle(self, candle: Candle) -> dict:
        """
        Adiciona vela e detecta padrões SMC.
        Retorna dicionário com detecções.
        """
        self.candles.append(candle)
        idx = len(self.candles) - 1

        detections = {
            "fvgs": [],
            "order_blocks": [],
            "structure_breaks": [],
            "liquidity_sweeps": [],
        }

        if len(self.candles) >= 3:
            fvgs = self._detect_fvg(idx)
            detections["fvgs"] = fvgs
            self.fvgs.extend(fvgs)

        if len(self.candles) >= self.swing_lookback * 2 + 1:
            sp = self._detect_structure_point(idx)
            if sp:
                self.structure_points.append(sp)
                # Verificar BOS
                bos = self._check_break_of_structure(sp)
                if bos:
                    detections["structure_breaks"].append(bos)

        if len(self.candles) >= 3:
            ob = self._detect_order_block(idx)
            if ob:
                self.order_blocks.append(ob)
                detections["order_blocks"].append(ob)

        sweep = self._detect_liquidity_sweep(idx)
        if sweep:
            self.liquidity_sweeps.append(sweep)
            detections["liquidity_sweeps"].append(sweep)

        # Atualizar FVGs (verificar fills)
        self._update_fvg_fills(candle)

        return detections

    def _detect_fvg(self, idx: int) -> list[FairValueGap]:
        """Detecta Fair Value Gaps."""
        candles = list(self.candles)
        if idx < 2:
            return []

        fvgs = []
        c0 = candles[idx - 2]  # Vela anterior
        c2 = candles[idx]      # Vela atual

        # Bullish FVG: low da vela atual > high da vela [idx-2]
        if c2.low > c0.high:
            gap_size_pct = (c2.low - c0.high) / c0.high * 100 if c0.high > 0 else 0
            if gap_size_pct >= self.fvg_min_size_pct:
                fvgs.append(
                    FairValueGap(
                        timestamp=c2.timestamp,
                        gap_type="bullish",
                        top=c2.low,
                        bottom=c0.high,
                        candle_index=idx,
                    )
                )

        # Bearish FVG: high da vela atual < low da vela [idx-2]
        if c2.high < c0.low:
            gap_size_pct = (c0.low - c2.high) / c2.high * 100 if c2.high > 0 else 0
            if gap_size_pct >= self.fvg_min_size_pct:
                fvgs.append(
                    FairValueGap(
                        timestamp=c2.timestamp,
                        gap_type="bearish",
                        top=c0.low,
                        bottom=c2.high,
                        candle_index=idx,
                    )
                )

        return fvgs

    def _detect_structure_point(self, idx: int) -> Optional[StructurePoint]:
        """Detecta swing highs e lows."""
        candles = list(self.candles)
        lookback = self.swing_lookback
        center_idx = idx - lookback

        if center_idx < lookback or center_idx >= len(candles):
            return None

        center = candles[center_idx]

        # Swing High: center.high é o maior entre os vizinhos
        is_swing_high = all(
            center.high >= candles[center_idx + i].high
            for i in range(-lookback, lookback + 1)
            if i != 0 and 0 <= center_idx + i < len(candles)
        )

        # Swing Low: center.low é o menor entre os vizinhos
        is_swing_low = all(
            center.low <= candles[center_idx + i].low
            for i in range(-lookback, lookback + 1)
            if i != 0 and 0 <= center_idx + i < len(candles)
        )

        if is_swing_high:
            return StructurePoint(
                timestamp=center.timestamp,
                price=center.high,
                point_type="high",
                candle_index=center_idx,
            )
        elif is_swing_low:
            return StructurePoint(
                timestamp=center.timestamp,
                price=center.low,
                point_type="low",
                candle_index=center_idx,
            )

        return None

    def _check_break_of_structure(
        self,
        new_point: StructurePoint,
    ) -> Optional[dict]:
        """
        Verifica se houve Break of Structure (BOS).

        Bullish BOS: novo higher high após uma série de lower highs
        Bearish BOS: novo lower low após uma série de higher lows
        """
        same_type = [
            sp for sp in self.structure_points[:-1]
            if sp.point_type == new_point.point_type
        ]

        if len(same_type) < 2:
            return None

        prev = same_type[-1]

        if new_point.point_type == "high":
            if new_point.price > prev.price:
                return {
                    "type": "bullish_bos",
                    "broken_level": prev.price,
                    "new_level": new_point.price,
                    "timestamp": new_point.timestamp,
                }
        else:  # low
            if new_point.price < prev.price:
                return {
                    "type": "bearish_bos",
                    "broken_level": prev.price,
                    "new_level": new_point.price,
                    "timestamp": new_point.timestamp,
                }

        return None

    def _detect_order_block(self, idx: int) -> Optional[OrderBlock]:
        """
        Detecta Order Blocks.

        Bullish OB: última vela bearish antes de um impulso bullish forte
        Bearish OB: última vela bullish antes de um impulso bearish forte
        """
        candles = list(self.candles)
        if idx < 2:
            return None

        current = candles[idx]
        prev = candles[idx - 1]
        prev2 = candles[idx - 2]

        # Bullish Order Block
        if (
            prev.is_bearish
            and current.is_bullish
            and current.body_size > prev.body_size * 1.5
            and current.body_ratio >= self.ob_body_ratio
        ):
            return OrderBlock(
                timestamp=prev.timestamp,
                block_type="bullish",
                high=prev.high,
                low=prev.low,
                volume=prev.volume,
                candle_index=idx - 1,
            )

        # Bearish Order Block
        if (
            prev.is_bullish
            and current.is_bearish
            and current.body_size > prev.body_size * 1.5
            and current.body_ratio >= self.ob_body_ratio
        ):
            return OrderBlock(
                timestamp=prev.timestamp,
                block_type="bearish",
                high=prev.high,
                low=prev.low,
                volume=prev.volume,
                candle_index=idx - 1,
            )

        return None

    def _detect_liquidity_sweep(self, idx: int) -> Optional[LiquiditySweep]:
        """
        Detecta Liquidity Sweeps.

        Quando preço rompe um swing high/low brevemente e volta
        (sweep do stop-loss pool).
        """
        candles = list(self.candles)
        if idx < 1:
            return None

        current = candles[idx]

        # Verificar sweep de highs recentes
        recent_highs = [
            sp for sp in self.structure_points
            if sp.point_type == "high"
            and sp.candle_index < idx
        ]

        for high_point in recent_highs[-3:]:
            # Wick acima do high mas fechou abaixo
            if current.high > high_point.price and current.close < high_point.price:
                return LiquiditySweep(
                    timestamp=current.timestamp,
                    sweep_type="buy_side",  # Varreu stops dos shorts
                    level_swept=high_point.price,
                    wick_price=current.high,
                    close_price=current.close,
                    candle_index=idx,
                )

        # Verificar sweep de lows recentes
        recent_lows = [
            sp for sp in self.structure_points
            if sp.point_type == "low"
            and sp.candle_index < idx
        ]

        for low_point in recent_lows[-3:]:
            if current.low < low_point.price and current.close > low_point.price:
                return LiquiditySweep(
                    timestamp=current.timestamp,
                    sweep_type="sell_side",  # Varreu stops dos longs
                    level_swept=low_point.price,
                    wick_price=current.low,
                    close_price=current.close,
                    candle_index=idx,
                )

        return None

    def _update_fvg_fills(self, candle: Candle) -> None:
        """Atualiza status de preenchimento dos FVGs."""
        for fvg in self.fvgs:
            if fvg.filled:
                continue

            if fvg.gap_type == "bullish":
                # Bullish FVG preenchido se preço cai para dentro do gap
                if candle.low <= fvg.top:
                    overlap = min(fvg.top, candle.high) - max(fvg.bottom, candle.low)
                    if overlap > 0:
                        fvg.fill_percentage = min(
                            overlap / fvg.size * 100, 100.0
                        )
                        if fvg.fill_percentage >= 50:
                            fvg.filled = True
            else:
                # Bearish FVG preenchido se preço sobe para dentro do gap
                if candle.high >= fvg.bottom:
                    overlap = min(fvg.top, candle.high) - max(fvg.bottom, candle.low)
                    if overlap > 0:
                        fvg.fill_percentage = min(
                            overlap / fvg.size * 100, 100.0
                        )
                        if fvg.fill_percentage >= 50:
                            fvg.filled = True

    def get_active_fvgs(self, max_age_candles: int = 50) -> list[FairValueGap]:
        """Retorna FVGs não preenchidos recentes."""
        min_idx = max(0, len(self.candles) - max_age_candles)
        return [
            fvg for fvg in self.fvgs
            if not fvg.filled and fvg.candle_index >= min_idx
        ]

    def get_active_order_blocks(
        self,
        max_age_candles: int = 100,
    ) -> list[OrderBlock]:
        """Retorna Order Blocks não quebrados."""
        min_idx = max(0, len(self.candles) - max_age_candles)
        return [
            ob for ob in self.order_blocks
            if not ob.broken and ob.candle_index >= min_idx
        ]

    def get_market_structure(self) -> dict:
        """Retorna estrutura de mercado atual."""
        if len(self.structure_points) < 2:
            return {"trend": "unknown", "points": []}

        highs = [sp for sp in self.structure_points if sp.point_type == "high"]
        lows = [sp for sp in self.structure_points if sp.point_type == "low"]

        trend = "unknown"
        if len(highs) >= 2 and len(lows) >= 2:
            hh = highs[-1].price > highs[-2].price  # Higher High
            hl = lows[-1].price > lows[-2].price     # Higher Low

            lh = highs[-1].price < highs[-2].price   # Lower High
            ll = lows[-1].price < lows[-2].price      # Lower Low

            if hh and hl:
                trend = "bullish"
            elif lh and ll:
                trend = "bearish"
            elif hh and ll:
                trend = "expanding"  # Range expandindo
            elif lh and hl:
                trend = "contracting"  # Range contraindo
            else:
                trend = "mixed"

        return {
            "trend": trend,
            "last_high": highs[-1].price if highs else None,
            "last_low": lows[-1].price if lows else None,
            "structure_points": len(self.structure_points),
        }

    def analyze(self) -> AnalysisResult:
        """Análise completa SMC."""
        result = AnalysisResult(
            source="smart_money_analyzer",
            timestamp=time.time(),
        )

        if len(self.candles) < 10:
            result.confidence = 0.0
            return result

        structure = self.get_market_structure()
        active_fvgs = self.get_active_fvgs()
        active_obs = self.get_active_order_blocks()
        current_price = self.candles[-1].close

        result.metrics = {
            "market_structure": structure["trend"],
            "active_fvgs": len(active_fvgs),
            "active_order_blocks": len(active_obs),
            "structure_points": len(self.structure_points),
            "liquidity_sweeps": len(self.liquidity_sweeps),
            "candle_count": len(self.candles),
        }

        # Sinais de FVGs próximos ao preço
        for fvg in active_fvgs:
            distance_pct = abs(current_price - fvg.midpoint) / current_price * 100

            if distance_pct < 1.0:  # FVG dentro de 1% do preço
                if fvg.gap_type == "bullish":
                    direction = Side.BUY
                    desc = (
                        f"Bullish FVG nearby: {fvg.bottom:.2f}-{fvg.top:.2f} "
                        f"(potential support)"
                    )
                else:
                    direction = Side.SELL
                    desc = (
                        f"Bearish FVG nearby: {fvg.bottom:.2f}-{fvg.top:.2f} "
                        f"(potential resistance)"
                    )

                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="smc_fvg",
                        direction=direction,
                        strength=SignalStrength.MODERATE,
                        price=fvg.midpoint,
                        confidence=max(0.3, 1.0 - distance_pct),
                        source="smart_money_analyzer",
                        description=desc,
                    )
                )

        # Sinais de Order Blocks
        for ob in active_obs:
            distance_pct = abs(current_price - ob.midpoint) / current_price * 100

            if distance_pct < 0.5:  # OB muito próximo
                direction = Side.BUY if ob.block_type == "bullish" else Side.SELL
                result.signals.append(
                    Signal(
                        timestamp=time.time(),
                        signal_type="smc_order_block",
                        direction=direction,
                        strength=SignalStrength.STRONG,
                        price=ob.midpoint,
                        confidence=max(0.5, 1.0 - distance_pct * 2),
                        source="smart_money_analyzer",
                        description=(
                            f"{ob.block_type.title()} Order Block at "
                            f"{ob.low:.2f}-{ob.high:.2f}"
                        ),
                    )
                )

        # Sinais de Liquidity Sweep recente
        recent_sweeps = self.liquidity_sweeps[-3:]
        for sweep in recent_sweeps:
            if sweep.sweep_type == "buy_side":
                direction = Side.SELL  # Sweep bullish = possível reversão bearish
                desc = f"Buy-side liquidity swept at {sweep.level_swept:.2f}"
            else:
                direction = Side.BUY  # Sweep bearish = possível reversão bullish
                desc = f"Sell-side liquidity swept at {sweep.level_swept:.2f}"

            result.signals.append(
                Signal(
                    timestamp=sweep.timestamp,
                    signal_type="smc_liquidity_sweep",
                    direction=direction,
                    strength=SignalStrength.STRONG,
                    price=sweep.close_price,
                    confidence=0.7,
                    source="smart_money_analyzer",
                    description=desc,
                )
            )

        # Sinal de estrutura
        if structure["trend"] in ("bullish", "bearish"):
            direction = Side.BUY if structure["trend"] == "bullish" else Side.SELL
            result.signals.append(
                Signal(
                    timestamp=time.time(),
                    signal_type="smc_structure",
                    direction=direction,
                    strength=SignalStrength.MODERATE,
                    price=current_price,
                    confidence=0.6,
                    source="smart_money_analyzer",
                    description=f"Market structure: {structure['trend']}",
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta estado."""
        self.candles.clear()
        self.fvgs.clear()
        self.order_blocks.clear()
        self.structure_points.clear()
        self.liquidity_sweeps.clear()
