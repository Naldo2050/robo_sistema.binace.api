import warnings
warnings.warn(
    "orderbook_analyzer.py na raiz está deprecated; "
    "importe de orderbook_analyzer.core",
    DeprecationWarning,
    stacklevel=2
)
from orderbook_analyzer.core import *
