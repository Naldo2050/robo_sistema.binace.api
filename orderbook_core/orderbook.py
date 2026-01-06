# orderbook_core/orderbook.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

from .exceptions import OrderBookError, InvalidUpdateError


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


@dataclass
class OrderBookUpdate:
    """Represents an update to the order book."""
    symbol: str
    update_id: int
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]  # [(price, quantity), ...]
    timestamp: float
    is_snapshot: bool = False

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class OrderBookSnapshot:
    """Represents a complete order book snapshot."""
    symbol: str
    last_update_id: int
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]  # [(price, quantity), ...]
    timestamp: float
    exchange: str = "binance"

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def best_bid(self) -> Optional[float]:
        """Get the best bid price."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Get the best ask price."""
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        """Calculate the bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate the mid price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    def get_imbalance(self, levels: int = 10) -> Optional[float]:
        """Calculate order book imbalance."""
        if not self.bids or not self.asks:
            return None

        bid_volume = sum(qty for _, qty in self.bids[:levels])
        ask_volume = sum(qty for _, qty in self.asks[:levels])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        return self.spread

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        return self.mid_price

    def get_total_bid_volume(self, levels: int = 10) -> float:
        """Get total bid volume."""
        return sum(qty for _, qty in self.bids[:levels])

    def get_total_ask_volume(self, levels: int = 10) -> float:
        """Get total ask volume."""
        return sum(qty for _, qty in self.asks[:levels])

    @classmethod
    def from_json(cls, json_data: dict) -> 'OrderBookSnapshot':
        """Cria snapshot a partir de JSON (necessário para testes)"""
        return cls(
            symbol=json_data.get('symbol', 'BTCUSDT'),
            last_update_id=json_data.get('lastUpdateId', 0),
            bids=[(float(p), float(q)) for p, q in json_data.get('bids', [])],
            asks=[(float(p), float(q)) for p, q in json_data.get('asks', [])],
            timestamp=json_data.get('timestamp', time.time()),
            exchange=json_data.get('exchange', 'binance')
        )

    def get_orderbook_snapshot(self) -> Dict[str, Any]:
        """Get order book snapshot as dictionary."""
        return {
            'symbol': self.symbol,
            'lastUpdateId': self.last_update_id,
            'bids': [[price, quantity] for price, quantity in self.bids],
            'asks': [[price, quantity] for price, quantity in self.asks],
            'timestamp': self.timestamp,
            'exchange': self.exchange
        }


class OrderBook:
    """Order book implementation for managing bids and asks."""

    def __init__(self, symbol: str, max_depth: int = 400):
        self.symbol = symbol
        self.max_depth = max_depth
        self.bids: List[Order] = []
        self.asks: List[Order] = []
        self.last_update_id: int = 0
        self.last_update_time: float = 0.0
        self._bids_dict: Dict[float, float] = {}  # price -> quantity
        self._asks_dict: Dict[float, float] = {}  # price -> quantity

    def update(self, update_data: Dict[str, Any]) -> bool:
        """Update order book with new data."""
        try:
            # Handle different data formats
            if 'bids' in update_data and 'asks' in update_data:
                # Snapshot format
                self._update_from_snapshot(update_data)
            elif 'b' in update_data and 'a' in update_data:
                # Binance WebSocket format
                self._update_from_websocket(update_data)
            else:
                raise InvalidUpdateError("Invalid update format")

            self.last_update_time = time.time()
            return True

        except Exception as e:
            raise InvalidUpdateError(f"Failed to update order book: {e}")

    def _update_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Update from snapshot data."""
        self.last_update_id = snapshot.get('lastUpdateId', 0)
        
        # Update bids
        self.bids = []
        self._bids_dict = {}
        for bid_data in snapshot.get('bids', []):
            if len(bid_data) >= 2:
                price, quantity = float(bid_data[0]), float(bid_data[1])
                if price > 0 and quantity > 0:
                    order = Order(price=price, quantity=quantity)
                    self.bids.append(order)
                    self._bids_dict[price] = quantity

        # Update asks
        self.asks = []
        self._asks_dict = {}
        for ask_data in snapshot.get('asks', []):
            if len(ask_data) >= 2:
                price, quantity = float(ask_data[0]), float(ask_data[1])
                if price > 0 and quantity > 0:
                    order = Order(price=price, quantity=quantity)
                    self.asks.append(order)
                    self._asks_dict[price] = quantity

        # Sort bids descending, asks ascending
        self.bids.sort(key=lambda x: x.price, reverse=True)
        self.asks.sort(key=lambda x: x.price)

    def _update_from_websocket(self, ws_data: Dict[str, Any]) -> None:
        """Update from WebSocket data."""
        update_id = ws_data.get('U', 0)
        if update_id <= self.last_update_id:
            return  # Skip out-of-order update

        # Update bids
        for bid_data in ws_data.get('b', []):
            price, quantity = float(bid_data[0]), float(bid_data[1])
            if quantity == 0:
                # Remove order
                if price in self._bids_dict:
                    del self._bids_dict[price]
                self.bids = [order for order in self.bids if order.price != price]
            else:
                # Add/update order
                self._bids_dict[price] = quantity
                # Update or add to bids list
                existing_order = next((order for order in self.bids if order.price == price), None)
                if existing_order:
                    existing_order.quantity = quantity
                else:
                    self.bids.append(Order(price=price, quantity=quantity))

        # Update asks
        for ask_data in ws_data.get('a', []):
            price, quantity = float(ask_data[0]), float(ask_data[1])
            if quantity == 0:
                # Remove order
                if price in self._asks_dict:
                    del self._asks_dict[price]
                self.asks = [order for order in self.asks if order.price != price]
            else:
                # Add/update order
                self._asks_dict[price] = quantity
                # Update or add to asks list
                existing_order = next((order for order in self.asks if order.price == price), None)
                if existing_order:
                    existing_order.quantity = quantity
                else:
                    self.asks.append(Order(price=price, quantity=quantity))

        # Sort bids descending, asks ascending
        self.bids.sort(key=lambda x: x.price, reverse=True)
        self.asks.sort(key=lambda x: x.price)

        self.last_update_id = update_id

    def get_best_bid(self) -> Optional[float]:
        """Get the best bid price."""
        return self.bids[0].price if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        """Get the best ask price."""
        return self.asks[0].price if self.asks else None

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return None

    def get_imbalance(self, levels: int = 10) -> float:
        """Calculate order book imbalance."""
        bid_volume = sum(order.quantity for order in self.bids[:levels])
        ask_volume = sum(order.quantity for order in self.asks[:levels])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total_volume

    def get_total_bid_volume(self, levels: int = 10) -> float:
        """Get total bid volume."""
        return sum(order.quantity for order in self.bids[:levels])

    def get_total_ask_volume(self, levels: int = 10) -> float:
        """Get total ask volume."""
        return sum(order.quantity for order in self.asks[:levels])

    def get_depth(self, side: str, levels: int = 10) -> List[Tuple[float, float]]:
        """Get depth for a specific side."""
        if side.lower() == 'bid':
            return [(order.price, order.quantity) for order in self.bids[:levels]]
        elif side.lower() == 'ask':
            return [(order.price, order.quantity) for order in self.asks[:levels]]
        else:
            raise ValueError("Side must be 'bid' or 'ask'")

    def get_orderbook_snapshot(self) -> Dict[str, Any]:
        """Get complete order book snapshot."""
        return {
            'symbol': self.symbol,
            'lastUpdateId': self.last_update_id,
            'bids': [[order.price, order.quantity] for order in self.bids],
            'asks': [[order.price, order.quantity] for order in self.asks],
            'timestamp': self.last_update_time
        }

    def get_spread_percentage(self) -> Optional[float]:
        """Get spread as percentage of mid price."""
        spread = self.get_spread()
        mid_price = self.get_mid_price()
        
        if spread is not None and mid_price is not None and mid_price > 0:
            return (spread / mid_price) * 100.0
        return None

    def get_market_impact(self, side: str, quantity: float) -> Dict[str, Any]:
        """Calculate market impact for a given quantity."""
        if side.lower() not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")

        remaining_quantity = quantity
        total_cost = 0.0
        levels_used = 0
        average_price = 0.0

        if side.lower() == 'buy':
            # Buy against asks (lowest price first)
            for order in self.asks:
                if remaining_quantity <= 0:
                    break
                
                if order.quantity <= remaining_quantity:
                    # Fill entire order
                    total_cost += order.price * order.quantity
                    remaining_quantity -= order.quantity
                    levels_used += 1
                else:
                    # Partial fill
                    total_cost += order.price * remaining_quantity
                    remaining_quantity = 0
                    levels_used += 1
        else:
            # Sell against bids (highest price first)
            for order in self.bids:
                if remaining_quantity <= 0:
                    break
                
                if order.quantity <= remaining_quantity:
                    # Fill entire order
                    total_cost += order.price * order.quantity
                    remaining_quantity -= order.quantity
                    levels_used += 1
                else:
                    # Partial fill
                    total_cost += order.price * remaining_quantity
                    remaining_quantity = 0
                    levels_used += 1

        if quantity > 0:
            average_price = total_cost / quantity
        else:
            average_price = 0.0

        return {
            'quantity_requested': quantity,
            'quantity_filled': quantity - remaining_quantity,
            'total_cost': total_cost,
            'average_price': average_price,
            'levels_used': levels_used,
            'slippage': 0.0  # Would need reference price to calculate
        }

    def is_valid(self) -> bool:
        """Check if order book is in a valid state."""
        # Check if bids and asks are sorted correctly
        if self.bids:
            if any(self.bids[i].price < self.bids[i-1].price for i in range(1, len(self.bids))):
                return False
        
        if self.asks:
            if any(self.asks[i].price > self.asks[i-1].price for i in range(1, len(self.asks))):
                return False

        # Check if best ask > best bid
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            if best_ask <= best_bid:
                return False

        # Check for negative prices or quantities
        for order in self.bids + self.asks:
            if order.price <= 0 or order.quantity < 0:
                return False

        return True

    def clear(self) -> None:
        """Clear the order book."""
        self.bids.clear()
        self.asks.clear()
        self._bids_dict.clear()
        self._asks_dict.clear()
        self.last_update_id = 0
        self.last_update_time = 0.0

    def _to_float_list(self, price_quantity_pairs):
        """Helper method for tests - converts string pairs to float tuples"""
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
        """Calcula soma em USD - usado pelos testes"""
        return sum(price * qty for price, qty in levels)

    def __len__(self) -> int:
        """Return total number of orders."""
        return len(self.bids) + len(self.asks)

    def __str__(self) -> str:
        """String representation of order book."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        spread = self.get_spread()
        
        return (f"OrderBook({self.symbol}): "
                f"Bid={best_bid}, Ask={best_ask}, "
                f"Spread={spread}, Orders={len(self)}")

    def __repr__(self) -> str:
        return self.__str__()