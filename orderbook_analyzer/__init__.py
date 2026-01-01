# orderbook_analyzer/__init__.py
# Import OrderBookAnalyzer from the root-level orderbook_analyzer.py module
# Use importlib to avoid circular imports
import importlib.util
import sys
import os

# Load the module from the root level
spec = importlib.util.spec_from_file_location("orderbook_analyzer_module", 
                                               os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                            "orderbook_analyzer.py"))
orderbook_analyzer_module = importlib.util.module_from_spec(spec)
sys.modules["orderbook_analyzer_module"] = orderbook_analyzer_module
spec.loader.exec_module(orderbook_analyzer_module)

# Now import the class and functions
OrderBookAnalyzer = orderbook_analyzer_module.OrderBookAnalyzer
_to_float_list = orderbook_analyzer_module._to_float_list
_sum_depth_usd = orderbook_analyzer_module._sum_depth_usd
_simulate_market_impact = orderbook_analyzer_module._simulate_market_impact

__all__ = ['OrderBookAnalyzer', '_to_float_list', '_sum_depth_usd', '_simulate_market_impact']