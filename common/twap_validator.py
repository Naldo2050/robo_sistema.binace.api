# -*- coding: utf-8 -*-
"""
TWAPValidator para FIX #2 - TWAP Impossível

Corrige o bug onde TWAP (média dos closes) viola invariante low ≤ TWAP ≤ high

Solução: Validar bounds e usar VWAP como fallback se violado
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional


logger = logging.getLogger(__name__)


class TWAPValidator:
    """Valida e corrige cálculos de TWAP com bounds checking"""

    BOUNDS_TOLERANCE = 0.01  # Tolerância de 1% para erros numéricos
    MIN_DATA_POINTS = 2

    @staticmethod
    def calculate_twap(close_prices: np.ndarray) -> float:
        """
        Calcula TWAP (Time-Weighted Average Price).
        
        Args:
            close_prices: Array de preços de fechamento
            
        Returns:
            Valor TWAP (float)
        """
        try:
            prices_array = np.asarray(close_prices, dtype=float)
            
            if len(prices_array) < 1:
                return 0.0
            
            # Filtrar NaN e infinitos
            clean_prices = prices_array[np.isfinite(prices_array)]
            if len(clean_prices) == 0:
                return 0.0
            
            twap = float(np.mean(clean_prices))
            
            if not np.isfinite(twap):
                return 0.0
            
            return twap
            
        except Exception as e:
            logger.error(f"Erro ao calcular TWAP: {e}")
            return 0.0

    @staticmethod
    def calculate_vwap(close_prices: np.ndarray, volumes: np.ndarray) -> float:
        """
        Calcula VWAP (Volume-Weighted Average Price).
        
        VWAP = sum(close * volume) / sum(volume)
        
        Args:
            close_prices: Array de preços de fechamento
            volumes: Array de volumes correspondentes
            
        Returns:
            Valor VWAP (float)
        """
        try:
            prices_array = np.asarray(close_prices, dtype=float)
            volumes_array = np.asarray(volumes, dtype=float)
            
            if len(prices_array) == 0 or len(volumes_array) == 0:
                return 0.0
            
            if len(prices_array) != len(volumes_array):
                logger.warning("Mismatched lengths: prices vs volumes in VWAP")
                min_len = min(len(prices_array), len(volumes_array))
                prices_array = prices_array[:min_len]
                volumes_array = volumes_array[:min_len]
            
            # Filtrar NaN
            valid_mask = np.isfinite(prices_array) & np.isfinite(volumes_array) & (volumes_array > 0)
            prices_clean = prices_array[valid_mask]
            volumes_clean = volumes_array[valid_mask]
            
            if len(prices_clean) == 0 or np.sum(volumes_clean) == 0:
                return TWAPValidator.calculate_twap(prices_array)  # Fallback to TWAP
            
            vwap = float(np.sum(prices_clean * volumes_clean) / np.sum(volumes_clean))
            
            if not np.isfinite(vwap):
                return TWAPValidator.calculate_twap(prices_array)
            
            return vwap
            
        except Exception as e:
            logger.error(f"Erro ao calcular VWAP: {e}")
            return TWAPValidator.calculate_twap(close_prices)

    @staticmethod
    def validate_twap_bounds(
        twap_value: float,
        low: float,
        high: float,
        symbol: str = "UNKNOWN"
    ) -> Tuple[float, bool, str]:
        """
        Valida que low ≤ TWAP ≤ high.
        
        Args:
            twap_value: Valor TWAP calculado
            low: Baixa do período
            high: Alta do período
            symbol: Símbolo para logging
            
        Returns:
            (twap_corrected, is_valid, reason)
        """
        try:
            if not (np.isfinite(twap_value) and np.isfinite(low) and np.isfinite(high)):
                return twap_value, False, "INVALID_VALUES"
            
            if low <= 0 or high <= 0:
                return twap_value, False, "ZERO_BOUNDS"
            
            if low > high:
                return twap_value, False, "INVERTED_BOUNDS"
            
            # Tolerância para erros numéricos
            lower_bound = low * (1.0 - TWAPValidator.BOUNDS_TOLERANCE)
            upper_bound = high * (1.0 + TWAPValidator.BOUNDS_TOLERANCE)
            
            # Checar se TWAP viola bounds
            if twap_value < lower_bound:
                logger.warning(
                    f"[FIX #2] TWAP_BELOW_LOW em {symbol}: "
                    f"twap={twap_value:.2f}, low={low:.2f}, "
                    f"lower_bound={lower_bound:.2f}"
                )
                return twap_value, False, "BELOW_LOW"
            
            if twap_value > upper_bound:
                logger.warning(
                    f"[FIX #2] TWAP_ABOVE_HIGH em {symbol}: "
                    f"twap={twap_value:.2f}, high={high:.2f}, "
                    f"upper_bound={upper_bound:.2f}"
                )
                return twap_value, False, "ABOVE_HIGH"
            
            return twap_value, True, "VALID"
            
        except Exception as e:
            logger.error(f"Erro em validate_twap_bounds: {e}")
            return twap_value, False, f"ERROR: {str(e)}"

    @staticmethod
    def validate_twap_with_fallback(
        close_prices: np.ndarray,
        volumes: np.ndarray,
        low: float,
        high: float,
        symbol: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        Calcula TWAP, valida bounds, e usa VWAP como fallback se violado.
        
        Returns:
            {
                "twap": float,
                "vwap_fallback": float,
                "is_valid": bool,
                "reason": str,
                "used_fallback": bool,
                "final_value": float (TWAP se valid, else VWAP)
            }
        """
        try:
            prices_array = np.asarray(close_prices, dtype=float)
            volumes_array = np.asarray(volumes, dtype=float)
            
            # Calcular TWAP
            twap_val = TWAPValidator.calculate_twap(prices_array)
            
            # Validar bounds
            twap_corrected, is_valid, reason = TWAPValidator.validate_twap_bounds(
                twap_val, low, high, symbol
            )
            
            # Se válido, usar TWAP
            if is_valid:
                return {
                    "twap": twap_val,
                    "vwap_fallback": None,
                    "is_valid": True,
                    "reason": reason,
                    "used_fallback": False,
                    "final_value": twap_val,
                }
            
            # Se inválido, calcular VWAP como fallback
            logger.info(f"[FIX #2] TWAP inválido ({reason}), usando VWAP como fallback para {symbol}")
            vwap_val = TWAPValidator.calculate_vwap(prices_array, volumes_array)
            
            return {
                "twap": twap_val,
                "vwap_fallback": vwap_val,
                "is_valid": False,
                "reason": reason,
                "used_fallback": True,
                "final_value": vwap_val,
            }
            
        except Exception as e:
            logger.error(f"Erro geral em validate_twap_with_fallback: {e}")
            return {
                "twap": 0.0,
                "vwap_fallback": None,
                "is_valid": False,
                "reason": f"ERROR: {str(e)}",
                "used_fallback": False,
                "final_value": 0.0,
            }
