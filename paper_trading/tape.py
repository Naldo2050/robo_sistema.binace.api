# paper_trading/tape.py
"""
Tape reader and synthetic tick generator for hermetic replay.

TapeReader supports the raw trade dump format used in live logging.
SyntheticTape provides fully deterministic pseudo-random walks for unit/property tests.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Union


@dataclass(frozen=True)
class TapeTick:
    """Standard market trade tick representation."""

    T: int
    p: float
    q: float
    m: bool
    trade_id: Union[int, str] = 0


class TapeReader:
    """
    Reads historical raw trade dumps from JSON or JSONL files.

    Accommodates Binance live dump fields:
    - Timestamp: 'T', 'timestamp', 'time', 'E'
    - Price: 'p', 'price'
    - Quantity: 'q', 'quantity', 'qty'
    - Maker side: 'm', 'is_buyer_maker'
    - Trade ID: 'trade_id', 't', 'id'
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tape dump file not found: {file_path}")

    def __iter__(self) -> Iterator[TapeTick]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                tick = self._parse_dict(data)
                if tick is not None:
                    yield tick

    @staticmethod
    def _parse_dict(data: dict) -> Optional[TapeTick]:
        # Resolve timestamp
        ts = data.get("T") or data.get("timestamp") or data.get("time") or data.get("E")
        if ts is None:
            return None

        # Resolve price
        p = data.get("p") or data.get("price")
        if p is None:
            return None

        # Resolve quantity
        q = data.get("q") or data.get("quantity") or data.get("qty") or 1.0

        # Resolve maker side
        m = data.get("m") if "m" in data else data.get("is_buyer_maker", False)

        # Resolve trade_id
        t_id = data.get("trade_id") or data.get("t") or data.get("id") or 0

        try:
            return TapeTick(
                T=int(ts),
                p=float(p),
                q=float(q),
                m=bool(m),
                trade_id=t_id,
            )
        except (ValueError, TypeError):
            return None


class SyntheticTape:
    """
    Generates a deterministic stream of price ticks using a pseudo-random walk.
    """

    def __init__(
        self,
        seed: int,
        n_ticks: int = 1000,
        start_price: float = 50_000.0,
        vol_bps: float = 2.0,
        start_ts_ms: int = 1_700_000_000_000,
        interval_ms: int = 100,
    ) -> None:
        self.seed = seed
        self.n_ticks = n_ticks
        self.start_price = start_price
        self.vol_bps = vol_bps
        self.start_ts_ms = start_ts_ms
        self.interval_ms = interval_ms

    def __iter__(self) -> Iterator[TapeTick]:
        rng = random.Random(self.seed)
        price = self.start_price
        ts = self.start_ts_ms

        for i in range(self.n_ticks):
            # Log-normal or gaussian fractional drift
            change_fraction = rng.gauss(0.0, self.vol_bps * 1e-4)
            price = max(0.01, round(price * (1.0 + change_fraction), 4))
            qty = round(rng.uniform(0.001, 1.5), 4)
            maker = rng.choice([True, False])
            t_id = i + 1

            yield TapeTick(
                T=ts,
                p=price,
                q=qty,
                m=maker,
                trade_id=t_id,
            )
            ts += self.interval_ms
