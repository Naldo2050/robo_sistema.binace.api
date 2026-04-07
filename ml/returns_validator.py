# -*- coding: utf-8 -*-
"""
ReturnsValidator para FIX #1 - Returns ML Zerados

Corrige o bug onde returns_1/5/10 são sempre 0.0 durante warmup
quando len(prices) < window_size.

Solução: Usar min(window_size, len(prices)) para adaptive window
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any


class ReturnsValidator:
    """Valida e corrige cálculos de returns com adaptive window"""

    RETURN_WINDOWS = [1, 5, 10]
    FALLBACK_THRESHOLD = 0.0001

    @staticmethod
    def validate_return(current_price: float, prices: np.ndarray, window: int) -> Tuple[float, bool, str]:
        """
        Valida e calcula return com adaptive window.

        Args:
            current_price: Preço atual
            prices: Array de preços históricos
            window: Tamanho da janela (1, 5, ou 10)

        Returns:
            (return_value, is_valid, reason)
        """
        try:
            prices_array = np.asarray(prices, dtype=float)
            n = len(prices_array)

            if n < 1:
                return 0.0, True, "EMPTY"

            if n < 2 and window == 1:
                return 0.0, True, "WARMUP"

            # Janela adaptiva: min(window, len(prices)-1)
            adaptive_window = min(window, n - 1) if n > 1 else 1

            if adaptive_window < 1:
                return 0.0, True, "WARMUP"

            price_idx = -adaptive_window
            if price_idx < -n:
                return 0.0, True, "WARMUP"

            past_price = float(prices_array[price_idx])

            if not np.isfinite(current_price) or not np.isfinite(past_price):
                return 0.0, False, "INVALID_PRICE"
            if past_price <= 0 or current_price <= 0:
                return 0.0, False, "ZERO_PRICE"

            ret = float(current_price / past_price - 1.0)

            if not np.isfinite(ret):
                return 0.0, False, "INVALID_RETURN"

            if abs(ret) > 0.5:
                logging.warning(f"Return extremo: {ret:.4f}")
                ret = np.clip(ret, -0.5, 0.5)

            return ret, True, "VALID"

        except Exception as e:
            logging.error(f"Erro em validate_return: {e}")
            return 0.0, False, f"ERROR: {str(e)}"

    @staticmethod
    def validate_all_returns(current_price: float, prices: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Valida todos os returns (1, 5, 10)"""
        result = {}
        prices_array = np.asarray(prices, dtype=float)
        n = len(prices_array)

        for window in ReturnsValidator.RETURN_WINDOWS:
            ret_val, is_valid, reason = ReturnsValidator.validate_return(current_price, prices_array, window)
            adaptive_window = min(window, max(1, n - 1)) if n > 1 else 1
            
            result[f"return_{window}"] = {
                "value": ret_val,
                "is_valid": is_valid,
                "reason": reason,
                "adaptive_window": adaptive_window,
                "requested_window": window,
                "is_warmup": reason == "WARMUP"
            }

        return result

