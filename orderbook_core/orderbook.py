# orderbook_core/orderbook.py
import json
import math
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import time

from .exceptions import OrderBookError, InvalidUpdateError

MAX_ORDERBOOK_DEPTH = 1000


@dataclass
class Order:
    """Represents a single order in the order book."""
    price: float
    quantity: float
    order_id: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def __iter__(self):
        yield self.price
        yield self.quantity

    def __getitem__(self, index):
        return (self.price, self.quantity)[index]


@dataclass
class OrderBookUpdate:
    """Represents an update to the order book."""
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]  # [(price, quantity), ...]
    symbol: str = ""
    update_id: int = 0
    timestamp: Any = None  # float or datetime
    is_snapshot: bool = False
    sequence: Optional[int] = None  # compatibility alias for update_id

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        elif self.timestamp == 0.0:
            self.timestamp = datetime.now()
        if self.sequence is not None and self.update_id == 0:
            self.update_id = self.sequence

    def validate(self):
        """Validate bids/asks; raises InvalidUpdateError on bad data."""
        for price, volume in self.bids:
            p, v = float(price), float(volume)
            if not math.isfinite(p) or p <= 0:
                raise InvalidUpdateError(f"Invalid bid price: {price}")
            if not math.isfinite(v) or v <= 0:
                raise InvalidUpdateError(f"Invalid bid volume: {volume}")
        for price, volume in self.asks:
            p, v = float(price), float(volume)
            if not math.isfinite(p) or p <= 0:
                raise InvalidUpdateError(f"Invalid ask price: {price}")
            if not math.isfinite(v) or v <= 0:
                raise InvalidUpdateError(f"Invalid ask volume: {volume}")
        return True


class OrderBookSnapshot:
    """Represents a complete order book snapshot."""

    def __init__(
        self,
        symbol: str,
        bids: List = None,
        asks: List = None,
        timestamp: Any = None,
        last_update_id: int = 0,
        exchange: str = "binance",
        sequence: Optional[int] = None,
        spread: Optional[float] = None,
        mid_price: Optional[float] = None,
    ):
        self.symbol = symbol
        self.bids = list(bids) if bids is not None else []
        self.asks = list(asks) if asks is not None else []
        self.timestamp = timestamp if timestamp is not None else datetime.now()
        self.exchange = exchange
        self.spread = spread
        self.mid_price = mid_price

        # Resolve last_update_id vs sequence
        if sequence is not None and last_update_id == 0:
            self.last_update_id = sequence
            self.sequence = sequence
        else:
            self.last_update_id = last_update_id
            self.sequence = sequence if sequence is not None else last_update_id

        # Compute spread/mid_price if not provided
        if self.spread is None and self.bids and self.asks:
            best_bid = float(self.bids[0][0])
            best_ask = float(self.asks[0][0])
            self.spread = best_ask - best_bid
        if self.mid_price is None and self.bids and self.asks:
            best_bid = float(self.bids[0][0])
            best_ask = float(self.asks[0][0])
            self.mid_price = (best_bid + best_ask) / 2.0

    # --- Convenience properties ---

    @property
    def best_bid(self) -> Optional[float]:
        return float(self.bids[0][0]) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return float(self.asks[0][0]) if self.asks else None

    # --- Methods expected by tests ---

    def validate(self) -> bool:
        """Return True if snapshot is internally consistent."""
        if not self.bids or not self.asks:
            return False
        # bids descending
        bid_prices = [float(b[0]) for b in self.bids]
        if not all(bid_prices[i] >= bid_prices[i + 1] for i in range(len(bid_prices) - 1)):
            return False
        # asks ascending
        ask_prices = [float(a[0]) for a in self.asks]
        if not all(ask_prices[i] <= ask_prices[i + 1] for i in range(len(ask_prices) - 1)):
            return False
        # positive prices and non-negative volumes
        for price, volume in self.bids:
            if float(price) <= 0 or float(volume) < 0:
                return False
        for price, volume in self.asks:
            if float(price) <= 0 or float(volume) < 0:
                return False
        return True

    def is_too_old(self, max_age_ms: float = 5000) -> bool:
        """Return True if snapshot is older than max_age_ms milliseconds."""
        ts = self.timestamp
        if isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts)
        age_ms = (datetime.now() - ts).total_seconds() * 1000
        return age_ms > max_age_ms

    def to_json(self) -> str:
        """Serialize snapshot to JSON string."""
        ts = self.timestamp
        if isinstance(ts, datetime):
            ts_str = ts.isoformat()
        else:
            ts_str = datetime.fromtimestamp(ts).isoformat()
        return json.dumps({
            'symbol': self.symbol,
            'sequence': self.sequence,
            'lastUpdateId': self.last_update_id,
            'bids': [[float(p), float(q)] for p, q in self.bids],
            'asks': [[float(p), float(q)] for p, q in self.asks],
            'timestamp': ts_str,
            'spread': self.spread,
            'mid_price': self.mid_price,
            'exchange': self.exchange,
        })

    @classmethod
    def from_json(cls, json_data) -> 'OrderBookSnapshot':
        """Create snapshot from JSON string or dict."""
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        ts_raw = data.get('timestamp', None)
        if isinstance(ts_raw, str):
            ts = datetime.fromisoformat(ts_raw)
        elif isinstance(ts_raw, (int, float)):
            ts = datetime.fromtimestamp(ts_raw)
        else:
            ts = datetime.now()

        seq = data.get('sequence', data.get('lastUpdateId', 0))
        bids_raw = data.get('bids', [])
        asks_raw = data.get('asks', [])
        bids = [(float(p), float(q)) for p, q in bids_raw]
        asks = [(float(p), float(q)) for p, q in asks_raw]

        return cls(
            symbol=data.get('symbol', 'BTCUSDT'),
            last_update_id=data.get('lastUpdateId', 0),
            sequence=seq,
            bids=bids,
            asks=asks,
            timestamp=ts,
            exchange=data.get('exchange', 'binance'),
            spread=data.get('spread'),
            mid_price=data.get('mid_price'),
        )

    def get_imbalance(self, levels: int = 10, depth: Optional[int] = None) -> Optional[float]:
        if depth is not None:
            levels = depth
        if not self.bids or not self.asks:
            return None
        bid_volume = sum(float(qty) for _, qty in self.bids[:levels])
        ask_volume = sum(float(qty) for _, qty in self.asks[:levels])
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        return (bid_volume - ask_volume) / total_volume

    def get_spread(self) -> Optional[float]:
        return self.spread

    def get_mid_price(self) -> Optional[float]:
        return self.mid_price

    def get_total_bid_volume(self, levels: int = 10) -> float:
        return sum(float(qty) for _, qty in self.bids[:levels])

    def get_total_ask_volume(self, levels: int = 10) -> float:
        return sum(float(qty) for _, qty in self.asks[:levels])

    def get_orderbook_snapshot(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'lastUpdateId': self.last_update_id,
            'bids': [[float(p), float(q)] for p, q in self.bids],
            'asks': [[float(p), float(q)] for p, q in self.asks],
            'timestamp': self.timestamp,
            'exchange': self.exchange,
        }


class OrderBook:
    """Order book implementation for managing bids and asks."""

    def __init__(self, symbol: str = "BTCUSDT", max_depth: int = MAX_ORDERBOOK_DEPTH):
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive")

        self.symbol = symbol
        self.max_depth = max_depth
        self.bids: List[Order] = []
        self.asks: List[Order] = []
        self.last_update_id: int = 0
        self.last_update_time: datetime = datetime.now()
        self._bids_dict: Dict[float, float] = {}  # price -> quantity
        self._asks_dict: Dict[float, float] = {}  # price -> quantity
        self._lock = threading.RLock()
        self.circuit_breaker = None

        # Statistics
        self._update_count: int = 0
        self._total_volume_processed: float = 0.0
        self._max_depth_reached: int = 0
        self._error_count: int = 0
        self._last_update_duration: float = 0.0
        self._start_time: datetime = datetime.now()

    @property
    def last_sequence(self) -> int:
        """Compatibility alias for last_update_id."""
        return self.last_update_id

    @last_sequence.setter
    def last_sequence(self, value: int):
        self.last_update_id = value

    def _validate_update(self, update: 'OrderBookUpdate') -> None:
        """Raise InvalidUpdateError if update contains bad data."""
        update.validate()

    def update(self, update_data) -> bool:
        """Update order book. Returns False for stale sequences, raises on invalid data."""
        start = time.time()
        try:
            if self.circuit_breaker:
                result = self.circuit_breaker.execute(self._update_internal, update_data)
            else:
                result = self._update_internal(update_data)
            return result
        except (InvalidUpdateError, OrderBookError):
            self._error_count += 1
            raise
        except Exception as e:
            self._error_count += 1
            raise InvalidUpdateError(f"Failed to update order book: {e}") from e
        finally:
            self._last_update_duration = time.time() - start

    def _update_internal(self, update_data) -> bool:
        if update_data is None:
            raise InvalidUpdateError("Update data cannot be None")

        # Convert to OrderBookUpdate if needed
        if isinstance(update_data, OrderBookUpdate):
            update = update_data
        elif isinstance(update_data, dict):
            if 'bids' in update_data and 'asks' in update_data:
                bids = [(float(p), float(q)) for p, q in update_data['bids']]
                asks = [(float(p), float(q)) for p, q in update_data['asks']]
                seq = update_data.get('lastUpdateId', update_data.get('sequence', self.last_update_id + 1))
                update = OrderBookUpdate(
                    bids=bids, asks=asks,
                    sequence=seq,
                    timestamp=update_data.get('timestamp', datetime.now()),
                )
            elif 'b' in update_data and 'a' in update_data:
                return self._apply_websocket_delta(update_data)
            else:
                raise InvalidUpdateError("Invalid update format")
        else:
            raise InvalidUpdateError(f"Unsupported update type: {type(update_data)}")

        # Validate data
        self._validate_update(update)

        with self._lock:
            # Check sequence — prefer .sequence if it was explicitly set
            seq = update.sequence if update.sequence is not None else update.update_id
            if seq <= self.last_update_id:
                return False

            # Snapshot-replace the book
            new_bids: List[Order] = []
            new_bids_dict: Dict[float, float] = {}
            for price, qty in update.bids:
                p, q = float(price), float(qty)
                order = Order(price=p, quantity=q)
                new_bids.append(order)
                new_bids_dict[p] = q

            new_asks: List[Order] = []
            new_asks_dict: Dict[float, float] = {}
            for price, qty in update.asks:
                p, q = float(price), float(qty)
                order = Order(price=p, quantity=q)
                new_asks.append(order)
                new_asks_dict[p] = q

            # Sort
            new_bids.sort(key=lambda x: x.price, reverse=True)
            new_asks.sort(key=lambda x: x.price)

            # Enforce max_depth
            new_bids = new_bids[:self.max_depth]
            new_asks = new_asks[:self.max_depth]

            self.bids = new_bids
            self.asks = new_asks
            self._bids_dict = new_bids_dict
            self._asks_dict = new_asks_dict

            self.last_update_id = seq
            ts = update.timestamp
            self.last_update_time = ts if isinstance(ts, datetime) else datetime.fromtimestamp(ts)

            # Update stats
            self._update_count += 1
            vol = sum(float(q) for _, q in update.bids) + sum(float(q) for _, q in update.asks)
            self._total_volume_processed += vol
            depth = max(len(self.bids), len(self.asks))
            if depth > self._max_depth_reached:
                self._max_depth_reached = depth

        return True

    def _apply_websocket_delta(self, ws_data: Dict[str, Any]) -> bool:
        """Apply a Binance WebSocket delta update."""
        update_id = ws_data.get('U', ws_data.get('u', 0))
        if update_id <= self.last_update_id:
            return False

        with self._lock:
            for bid_data in ws_data.get('b', []):
                price, quantity = float(bid_data[0]), float(bid_data[1])
                if quantity == 0:
                    self._bids_dict.pop(price, None)
                    self.bids = [o for o in self.bids if o.price != price]
                else:
                    self._bids_dict[price] = quantity
                    existing = next((o for o in self.bids if o.price == price), None)
                    if existing:
                        existing.quantity = quantity
                    else:
                        self.bids.append(Order(price=price, quantity=quantity))

            for ask_data in ws_data.get('a', []):
                price, quantity = float(ask_data[0]), float(ask_data[1])
                if quantity == 0:
                    self._asks_dict.pop(price, None)
                    self.asks = [o for o in self.asks if o.price != price]
                else:
                    self._asks_dict[price] = quantity
                    existing = next((o for o in self.asks if o.price == price), None)
                    if existing:
                        existing.quantity = quantity
                    else:
                        self.asks.append(Order(price=price, quantity=quantity))

            self.bids.sort(key=lambda x: x.price, reverse=True)
            self.asks.sort(key=lambda x: x.price)
            self.bids = self.bids[:self.max_depth]
            self.asks = self.asks[:self.max_depth]
            self.last_update_id = update_id
            self.last_update_time = datetime.now()
            self._update_count += 1

        return True

    # --- Basic accessors ---

    def get_best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    def get_spread(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_mid_price(self) -> Optional[float]:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return None

    def get_spread_percentage(self) -> Optional[float]:
        spread = self.get_spread()
        mid_price = self.get_mid_price()
        if spread is not None and mid_price and mid_price > 0:
            return (spread / mid_price) * 100.0
        return None

    # --- Volume ---

    def get_imbalance(self, levels: int = 10, depth: Optional[int] = None) -> float:
        if depth is not None:
            levels = depth
        bid_volume = sum(o.quantity for o in self.bids[:levels])
        ask_volume = sum(o.quantity for o in self.asks[:levels])
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        return (bid_volume - ask_volume) / total_volume

    def get_total_bid_volume(self, levels: int = 10) -> float:
        return sum(o.quantity for o in self.bids[:levels])

    def get_total_ask_volume(self, levels: int = 10) -> float:
        return sum(o.quantity for o in self.asks[:levels])

    def get_total_volume(self, side: str = 'all') -> float:
        """Get total volume for a side ('bid', 'ask', or 'all')."""
        s = side.lower()
        if s == 'bid':
            return sum(o.quantity for o in self.bids)
        elif s == 'ask':
            return sum(o.quantity for o in self.asks)
        else:
            return sum(o.quantity for o in self.bids) + sum(o.quantity for o in self.asks)

    def get_volume_at_price(self, price: float, side: str = 'bid') -> float:
        """Get volume at a specific price level for a given side."""
        s = side.lower()
        if s == 'bid':
            return self._bids_dict.get(float(price), 0.0)
        elif s == 'ask':
            return self._asks_dict.get(float(price), 0.0)
        else:
            raise KeyError(f"Invalid side: {side}. Must be 'bid' or 'ask'.")

    # --- Depth / levels ---

    def get_depth(self, side: str, levels: int = 10) -> List[Tuple[float, float]]:
        if side.lower() == 'bid':
            return [(o.price, o.quantity) for o in self.bids[:levels]]
        elif side.lower() == 'ask':
            return [(o.price, o.quantity) for o in self.asks[:levels]]
        else:
            raise ValueError("Side must be 'bid' or 'ask'")

    def get_price_levels(self, side: str, depth: int = 10) -> Any:
        """Return price/volume levels for a side.

        Returns a list of {'price': x, 'volume': y} dicts for 'bid' or 'ask',
        or a dict {'bids': [...], 'asks': [...]} for 'all'.
        """
        s = side.lower()
        if s == 'bid':
            return [{'price': o.price, 'volume': o.quantity} for o in self.bids[:depth]]
        elif s == 'ask':
            return [{'price': o.price, 'volume': o.quantity} for o in self.asks[:depth]]
        else:  # 'all'
            return {
                'bids': [{'price': o.price, 'volume': o.quantity} for o in self.bids[:depth]],
                'asks': [{'price': o.price, 'volume': o.quantity} for o in self.asks[:depth]],
            }

    def get_weighted_average_price(self, side: str, levels: int = 10, depth: Optional[int] = None) -> Optional[float]:
        """Volume-weighted average price for a side."""
        if depth is not None:
            levels = depth
        if levels == 0:
            return None
        orders = self.bids[:levels] if side.lower() == 'bid' else self.asks[:levels]
        total_volume = sum(o.quantity for o in orders)
        if total_volume == 0:
            return None
        return sum(o.price * o.quantity for o in orders) / total_volume

    # --- Market impact ---

    def get_market_impact(self, side: str, quantity: float = 0.0, size: Optional[float] = None) -> Dict[str, Any]:
        """Calculate market impact. Returns impact_price, slippage, filled_size, etc."""
        if size is not None:
            quantity = size

        if quantity <= 0:
            ref_price = self.get_best_ask() if side.lower() == 'buy' else self.get_best_bid()
            return {
                'impact_price': None,
                'slippage': 0.0,
                'filled_size': 0,
                'remaining_size': quantity,
                'estimated_cost': 0.0,
                'levels_used': 0,
                'quantity_requested': quantity,
                'quantity_filled': 0,
                'total_cost': 0.0,
                'average_price': None,
            }

        s = side.lower()
        if s not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")

        orders = self.asks if s == 'buy' else self.bids
        if not orders:
            return {
                'impact_price': None,
                'slippage': 0.0,
                'filled_size': 0,
                'remaining_size': quantity,
                'estimated_cost': 0.0,
                'levels_used': 0,
                'quantity_requested': quantity,
                'quantity_filled': 0,
                'total_cost': 0.0,
                'average_price': None,
            }

        best_price = orders[0].price
        remaining = quantity
        total_cost = 0.0
        levels_used = 0

        for order in orders:
            if remaining <= 0:
                break
            fill = min(order.quantity, remaining)
            total_cost += order.price * fill
            remaining -= fill
            levels_used += 1

        filled = quantity - remaining
        avg_price = total_cost / filled if filled > 0 else None
        slippage = (avg_price - best_price) if avg_price is not None and s == 'buy' else \
                   (best_price - avg_price) if avg_price is not None else 0.0

        return {
            'impact_price': avg_price,
            'slippage': max(0.0, slippage) if slippage is not None else 0.0,
            'filled_size': filled,
            'remaining_size': remaining,
            'estimated_cost': total_cost,
            'levels_used': levels_used,
            'quantity_requested': quantity,
            'quantity_filled': filled,
            'total_cost': total_cost,
            'average_price': avg_price,
        }

    # --- Order simulation ---

    def simulate_order(self, order_type: str = 'market', side: str = 'buy',
                       size: float = 0.0, price: Optional[float] = None,
                       quantity: float = 0.0) -> Dict[str, Any]:
        """Simulate order execution. Modifies the book for market orders."""
        qty = size if size else quantity

        if qty <= 0:
            return {'success': False, 'error': 'Size must be positive',
                    'order_type': order_type, 'side': side,
                    'filled_size': 0, 'remaining_size': qty, 'slippage': 0.0}

        s = side.lower()
        ot = order_type.lower()

        if ot not in ('market', 'limit'):
            return {'success': False, 'error': f'Invalid order type: {order_type}',
                    'order_type': order_type, 'side': side,
                    'filled_size': 0, 'remaining_size': qty, 'slippage': 0.0}

        if ot == 'market':
            return self._execute_market_order(side=s, size=qty)
        else:  # limit
            return self._execute_limit_order(side=s, size=qty, limit_price=price)

    def _execute_market_order(self, side: str, size: float) -> Dict[str, Any]:
        """Execute a market order, modifying the book."""
        orders = self.asks if side == 'buy' else self.bids
        order_dict = self._asks_dict if side == 'buy' else self._bids_dict

        if not orders:
            return {'success': False, 'error': 'No liquidity', 'order_type': 'market',
                    'side': side, 'filled_size': 0, 'remaining_size': size, 'slippage': 0.0}

        best_price = orders[0].price
        remaining = size
        total_cost = 0.0
        levels_used = 0
        to_remove = []

        for order in list(orders):
            if remaining <= 0:
                break
            fill = min(order.quantity, remaining)
            total_cost += order.price * fill
            remaining -= fill
            levels_used += 1
            order.quantity -= fill
            order_dict[order.price] = order.quantity
            if order.quantity <= 0:
                to_remove.append(order.price)

        # Remove depleted levels from book
        for p in to_remove:
            order_dict.pop(p, None)
        if side == 'buy':
            self.asks = [o for o in self.asks if o.quantity > 0]
        else:
            self.bids = [o for o in self.bids if o.quantity > 0]

        filled = size - remaining
        avg_price = total_cost / filled if filled > 0 else None
        slippage = max(0.0, avg_price - best_price) if avg_price and side == 'buy' else \
                   max(0.0, best_price - avg_price) if avg_price else 0.0

        return {
            'success': True,
            'order_type': 'market',
            'side': side,
            'filled_size': filled,
            'remaining_size': remaining,
            'estimated_cost': total_cost,
            'impact_price': avg_price,
            'slippage': slippage,
            'levels_used': levels_used,
        }

    def _execute_limit_order(self, side: str, size: float, limit_price: Optional[float]) -> Dict[str, Any]:
        """Execute a limit order."""
        if limit_price is None:
            return {'success': False, 'error': 'Limit price required',
                    'order_type': 'limit', 'side': side, 'filled_size': 0,
                    'remaining_size': size, 'slippage': 0.0}

        if side == 'buy':
            best_ask = self.get_best_ask()
            if best_ask is None or limit_price < best_ask:
                return {'success': False, 'error': 'Limit price below best ask',
                        'order_type': 'limit', 'side': side, 'filled_size': 0,
                        'remaining_size': size, 'slippage': 0.0}
            # Can fill immediately
            result = self._execute_market_order(side='buy', size=size)
            result['order_type'] = 'limit'
            return result
        else:
            best_bid = self.get_best_bid()
            if best_bid is None or limit_price > best_bid:
                return {'success': False, 'error': 'Limit price above best bid',
                        'order_type': 'limit', 'side': side, 'filled_size': 0,
                        'remaining_size': size, 'slippage': 0.0}
            result = self._execute_market_order(side='sell', size=size)
            result['order_type'] = 'limit'
            return result

    # --- Liquidity profile ---

    def get_liquidity_profile(self, price_range_percent: float = 0.02) -> Dict[str, Any]:
        """Get liquidity distribution within a price range around mid price."""
        mid = self.get_mid_price()
        if mid is None:
            return {}

        low = mid * (1 - price_range_percent)
        high = mid * (1 + price_range_percent)

        bid_liq = sum(o.quantity * o.price for o in self.bids if o.price >= low)
        ask_liq = sum(o.quantity * o.price for o in self.asks if o.price <= high)
        total = bid_liq + ask_liq

        return {
            'bid_liquidity': bid_liq,
            'ask_liquidity': ask_liq,
            'total_liquidity': total,
            'liquidity_ratio': bid_liq / ask_liq if ask_liq > 0 else 0.0,
            'price_range': {'min': low, 'max': high, 'center': mid},
        }

    # --- Statistics ---

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive order book statistics."""
        now = datetime.now()
        age = (now - self._start_time).total_seconds()
        return {
            'symbol': self.symbol,
            'update_count': self._update_count,
            'total_volume_processed': self._total_volume_processed,
            'max_depth_reached': self._max_depth_reached,
            'current_depth': max(len(self.bids), len(self.asks)),
            'bid_levels': len(self.bids),
            'ask_levels': len(self.asks),
            'best_bid': self.get_best_bid(),
            'best_ask': self.get_best_ask(),
            'spread': self.get_spread(),
            'mid_price': self.get_mid_price(),
            'imbalance': self.get_imbalance(),
            'last_update_id': self.last_update_id,
            'age_seconds': age,
            'updates_per_second': self._update_count / age if age > 0 else 0.0,
            'last_update_duration': self._last_update_duration,
            'error_count': self._error_count,
        }

    # --- Snapshot ---

    def create_snapshot(self) -> 'OrderBookSnapshot':
        """Create an OrderBookSnapshot from current state."""
        spread = self.get_spread()
        mid = self.get_mid_price()
        return OrderBookSnapshot(
            symbol=self.symbol,
            last_update_id=self.last_update_id,
            sequence=self.last_update_id,
            bids=[(o.price, o.quantity) for o in self.bids],
            asks=[(o.price, o.quantity) for o in self.asks],
            timestamp=self.last_update_time,
            spread=spread,
            mid_price=mid,
        )

    def get_orderbook_snapshot(self) -> Dict[str, Any]:
        """Get complete order book snapshot as dict."""
        return {
            'symbol': self.symbol,
            'lastUpdateId': self.last_update_id,
            'bids': [[o.price, o.quantity] for o in self.bids],
            'asks': [[o.price, o.quantity] for o in self.asks],
            'timestamp': self.last_update_time,
        }

    # --- Reset / clear ---

    def clear(self) -> None:
        """Clear the order book."""
        with self._lock:
            self.bids = []
            self.asks = []
            self._bids_dict = {}
            self._asks_dict = {}
            self.last_update_id = 0
            self.last_update_time = datetime.now()

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    # --- Validity ---

    def is_valid(self) -> bool:
        if self.bids:
            if any(self.bids[i].price < self.bids[i-1].price for i in range(1, len(self.bids))):
                return False
        if self.asks:
            if any(self.asks[i].price > self.asks[i-1].price for i in range(1, len(self.asks))):
                return False
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None and best_ask <= best_bid:
            return False
        for order in self.bids + self.asks:
            if order.price <= 0 or order.quantity < 0:
                return False
        return True

    # --- Helper methods (used by other tests) ---

    def _to_float_list(self, price_quantity_pairs):
        result = []
        for price_str, qty_str in price_quantity_pairs:
            try:
                price = float(price_str)
                qty = float(qty_str)
                if price > 0 and qty > 0:
                    result.append((price, qty))
            except (ValueError, TypeError):
                continue
        return result

    def _sum_depth_usd(self, levels: List[Tuple[float, float]]) -> float:
        return sum(price * qty for price, qty in levels)

    def __len__(self) -> int:
        return len(self.bids) + len(self.asks)

    def __str__(self) -> str:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        spread = self.get_spread()
        return (f"OrderBook({self.symbol}): "
                f"Bid={best_bid}, Ask={best_ask}, "
                f"Spread={spread}, Orders={len(self)}")

    def __repr__(self) -> str:
        return self.__str__()
