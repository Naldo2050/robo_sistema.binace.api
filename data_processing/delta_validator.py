# delta_validator.py
import logging
from typing import Dict, Any, Tuple

class DeltaValidator:
    DELTA_TOLERANCE = 1e-2
    VOLUME_TOLERANCE = 1e-4
    
    @staticmethod
    def validate_delta_invariant(buy_volume: float, sell_volume: float, stored_delta: float, symbol: str = 'UNKNOWN') -> Tuple[float, bool, str]:
        delta_calc = float(buy_volume - sell_volume)
        if abs(delta_calc - stored_delta) <= DeltaValidator.DELTA_TOLERANCE:
            return delta_calc, True, 'VALID'
        if abs(delta_calc) < DeltaValidator.VOLUME_TOLERANCE and abs(stored_delta) > DeltaValidator.VOLUME_TOLERANCE:
            logging.warning(f'Delta MISMATCH em {symbol}')
            return delta_calc, False, 'MISMATCH'
        elif abs(delta_calc) > DeltaValidator.VOLUME_TOLERANCE and abs(stored_delta) < DeltaValidator.VOLUME_TOLERANCE:
            logging.error(f'DELTA ZERO VIOLATION em {symbol}')
            return delta_calc, False, 'ZERO_BUG'
        return delta_calc, False, 'DRIFT'
    
    @staticmethod
    def build_delta_result(buy_volume: float, sell_volume: float) -> Dict[str, Any]:
        delta = float(buy_volume - sell_volume)
        is_verified = not (abs(delta) < DeltaValidator.VOLUME_TOLERANCE and (buy_volume + sell_volume) > DeltaValidator.VOLUME_TOLERANCE)
        return {'delta': delta, 'delta_verified': is_verified, 'delta_source': 'calculated' if is_verified else 'fallback'}
