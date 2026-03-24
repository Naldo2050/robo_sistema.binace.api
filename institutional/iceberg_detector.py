# institutional/iceberg_detector.py
"""
Iceberg Order Detection (Detecção de Ordens Iceberg)

Detecta ordens grandes "escondidas" que se renovam automaticamente.
Quando um institucional quer comprar sem mover o preço,
coloca ordens pequenas que se recarregam repetidamente.

Método #6 do Arsenal Institucional.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from institutional.base import (
    AnalysisResult,
    InvalidParameterError,
    OrderBookSnapshot,
    Side,
    Signal,
    SignalStrength,
)


@dataclass
class IcebergCandidate:
    """Candidato a ordem iceberg."""
    price_level: float
    side: Side
    first_seen: float
    last_seen: float
    refill_count: int
    avg_quantity: float
    total_consumed: float
    confidence: float
    is_confirmed: bool = False


@dataclass
class OrderLevelHistory:
    """Histórico de um nível de preço no orderbook."""
    price: float
    side: Side
    observations: list[dict] = field(default_factory=list)
    # Cada observation: {"timestamp": float, "quantity": float, "was_consumed": bool}


class IcebergDetector:
    """
    Detector de ordens iceberg no orderbook.

    Monitora níveis de preço no orderbook e detecta padrões de:
    1. Ordem aparece em um preço
    2. É consumida (parcial ou totalmente)
    3. Reaparece com quantidade similar no mesmo preço

    Quando isso acontece 3+ vezes = provável iceberg.
    """

    def __init__(
        self,
        min_refills: int = 3,
        quantity_tolerance_pct: float = 30.0,
        max_tracked_levels: int = 100,
        price_bucket_size: float = 1.0,
        expiry_seconds: float = 300.0,
    ):
        if min_refills < 2:
            raise InvalidParameterError("min_refills must be >= 2")

        self.min_refills = min_refills
        self.quantity_tolerance = quantity_tolerance_pct / 100.0
        self.max_tracked_levels = max_tracked_levels
        self.price_bucket_size = price_bucket_size
        self.expiry_seconds = expiry_seconds

        # Tracking por nível de preço
        self._bid_levels: dict[float, OrderLevelHistory] = {}
        self._ask_levels: dict[float, OrderLevelHistory] = {}

        # Icebergs detectados
        self._candidates: list[IcebergCandidate] = []
        self._confirmed: list[IcebergCandidate] = []

        self._last_snapshot: Optional[OrderBookSnapshot] = None
        self._snapshots_processed: int = 0

    @property
    def confirmed_icebergs(self) -> list[IcebergCandidate]:
        return self._confirmed.copy()

    @property
    def candidates(self) -> list[IcebergCandidate]:
        return self._candidates.copy()

    def _bucket_price(self, price: float) -> float:
        """Agrupa preço em buckets."""
        return round(price / self.price_bucket_size) * self.price_bucket_size

    def process_snapshot(
        self,
        snapshot: OrderBookSnapshot,
    ) -> list[IcebergCandidate]:
        """
        Processa um snapshot do orderbook e detecta padrões iceberg.

        Retorna novos icebergs detectados neste snapshot.
        """
        self._snapshots_processed += 1
        new_icebergs: list[IcebergCandidate] = []

        if self._last_snapshot is not None:
            new_icebergs.extend(
                self._compare_levels(
                    self._last_snapshot, snapshot, Side.BUY
                )
            )
            new_icebergs.extend(
                self._compare_levels(
                    self._last_snapshot, snapshot, Side.SELL
                )
            )

        self._last_snapshot = snapshot
        self._cleanup_expired()

        return new_icebergs

    def _compare_levels(
        self,
        prev: OrderBookSnapshot,
        curr: OrderBookSnapshot,
        side: Side,
    ) -> list[IcebergCandidate]:
        """Compara níveis entre dois snapshots."""
        prev_levels = prev.bids if side == Side.BUY else prev.asks
        curr_levels = curr.bids if side == Side.BUY else curr.asks
        tracked = self._bid_levels if side == Side.BUY else self._ask_levels

        # Mapear níveis atuais
        prev_map: dict[float, float] = {}
        for level in prev_levels:
            bucket = self._bucket_price(level.price)
            prev_map[bucket] = prev_map.get(bucket, 0) + level.quantity

        curr_map: dict[float, float] = {}
        for level in curr_levels:
            bucket = self._bucket_price(level.price)
            curr_map[bucket] = curr_map.get(bucket, 0) + level.quantity

        new_icebergs: list[IcebergCandidate] = []
        now = curr.timestamp

        for price_bucket, prev_qty in prev_map.items():
            curr_qty = curr_map.get(price_bucket, 0)

            # Caso 1: Nível foi consumido significativamente
            if curr_qty < prev_qty * 0.3 and prev_qty > 0:
                if price_bucket in tracked:
                    tracked[price_bucket].observations.append({
                        "timestamp": now,
                        "quantity": prev_qty,
                        "was_consumed": True,
                    })
                else:
                    tracked[price_bucket] = OrderLevelHistory(
                        price=price_bucket,
                        side=side,
                        observations=[{
                            "timestamp": now,
                            "quantity": prev_qty,
                            "was_consumed": True,
                        }],
                    )

            # Caso 2: Nível reapareceu (refill)
            elif price_bucket in tracked and curr_qty > prev_qty * 0.7:
                history = tracked[price_bucket]
                consumed_obs = [
                    o for o in history.observations if o["was_consumed"]
                ]

                if consumed_obs:
                    last_consumed = consumed_obs[-1]
                    # Verificar se quantidade é similar (iceberg refill)
                    qty_diff = abs(curr_qty - last_consumed["quantity"])
                    qty_ratio = qty_diff / last_consumed["quantity"] if last_consumed["quantity"] > 0 else 1.0

                    if qty_ratio <= self.quantity_tolerance:
                        history.observations.append({
                            "timestamp": now,
                            "quantity": curr_qty,
                            "was_consumed": False,
                        })

                        refill_count = sum(
                            1 for o in history.observations if o["was_consumed"]
                        )

                        if refill_count >= self.min_refills:
                            quantities = [
                                o["quantity"] for o in history.observations
                            ]
                            avg_qty = sum(quantities) / len(quantities)
                            total_consumed = sum(
                                o["quantity"]
                                for o in history.observations
                                if o["was_consumed"]
                            )

                            confidence = min(refill_count / (self.min_refills * 2), 1.0)

                            candidate = IcebergCandidate(
                                price_level=price_bucket,
                                side=side,
                                first_seen=history.observations[0]["timestamp"],
                                last_seen=now,
                                refill_count=refill_count,
                                avg_quantity=avg_qty,
                                total_consumed=total_consumed,
                                confidence=confidence,
                                is_confirmed=refill_count >= self.min_refills + 1,
                            )

                            # Atualizar ou adicionar
                            existing = self._find_candidate(price_bucket, side)
                            if existing:
                                existing.refill_count = refill_count
                                existing.last_seen = now
                                existing.total_consumed = total_consumed
                                existing.confidence = confidence
                                existing.is_confirmed = candidate.is_confirmed
                            else:
                                self._candidates.append(candidate)
                                if candidate.is_confirmed:
                                    self._confirmed.append(candidate)
                                new_icebergs.append(candidate)

        # Limpar levels antigos
        if len(tracked) > self.max_tracked_levels:
            # Manter só os mais recentes
            sorted_levels = sorted(
                tracked.items(),
                key=lambda x: x[1].observations[-1]["timestamp"] if x[1].observations else 0,
                reverse=True,
            )
            new_tracked = dict(sorted_levels[:self.max_tracked_levels])
            tracked.clear()
            tracked.update(new_tracked)

        return new_icebergs

    def _find_candidate(
        self,
        price: float,
        side: Side,
    ) -> Optional[IcebergCandidate]:
        """Encontra candidato existente."""
        for c in self._candidates:
            if abs(c.price_level - price) < self.price_bucket_size and c.side == side:
                return c
        return None

    def _cleanup_expired(self) -> None:
        """Remove candidatos expirados."""
        now = time.time()
        self._candidates = [
            c for c in self._candidates
            if now - c.last_seen < self.expiry_seconds
        ]
        self._confirmed = [
            c for c in self._confirmed
            if now - c.last_seen < self.expiry_seconds * 2
        ]

    def analyze(self) -> AnalysisResult:
        """Análise completa."""
        result = AnalysisResult(
            source="iceberg_detector",
            timestamp=time.time(),
        )

        result.metrics = {
            "snapshots_processed": self._snapshots_processed,
            "active_candidates": len(self._candidates),
            "confirmed_icebergs": len(self._confirmed),
            "tracked_bid_levels": len(self._bid_levels),
            "tracked_ask_levels": len(self._ask_levels),
        }

        for iceberg in self._confirmed:
            direction = iceberg.side
            strength = (
                SignalStrength.STRONG if iceberg.confidence > 0.7
                else SignalStrength.MODERATE
            )

            result.signals.append(
                Signal(
                    timestamp=iceberg.last_seen,
                    signal_type="iceberg_detected",
                    direction=direction,
                    strength=strength,
                    price=iceberg.price_level,
                    confidence=iceberg.confidence,
                    source="iceberg_detector",
                    description=(
                        f"Iceberg {direction.value} at {iceberg.price_level:.2f}: "
                        f"{iceberg.refill_count} refills, "
                        f"avg qty={iceberg.avg_quantity:.4f}, "
                        f"total consumed={iceberg.total_consumed:.4f}"
                    ),
                    metadata={
                        "refill_count": iceberg.refill_count,
                        "avg_quantity": iceberg.avg_quantity,
                        "total_consumed": iceberg.total_consumed,
                        "first_seen": iceberg.first_seen,
                    },
                )
            )

        result.confidence = max(
            (s.confidence for s in result.signals), default=0.0
        )

        return result

    def reset(self) -> None:
        """Reseta estado."""
        self._bid_levels.clear()
        self._ask_levels.clear()
        self._candidates.clear()
        self._confirmed.clear()
        self._last_snapshot = None
        self._snapshots_processed = 0
