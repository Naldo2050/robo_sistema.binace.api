# flow_analyzer/validation.py
"""
Validação de dados do FlowAnalyzer.

Inclui:
- TradeSchema: Validação de trades
- Validação OHLC
- Guard de absorção
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Tuple, Optional, Dict, Any, Set

from .utils import lazy_log
from .errors import TradeValidationError, AbsorptionGuardError


# ==============================================================================
# TRADE SCHEMA
# ==============================================================================

@dataclass
class TradeSchema:
    """
    Schema de validação para trades.
    
    Campos obrigatórios:
    - q: quantity (volume)
    - T: timestamp (epoch ms)
    - p: price
    
    Campos opcionais:
    - m: is_buyer_maker (bool)
    
    Example:
        >>> valid, reason = TradeSchema.validate({"q": 1.5, "T": 1234567890, "p": 50000})
        >>> valid
        True
    """
    
    required_fields: Set[str] = frozenset({'q', 'T', 'p'})
    optional_fields: Set[str] = frozenset({'m'})
    
    field_types: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.field_types is None:
            self.field_types = {
                'q': (int, float, Decimal, str),
                'T': (int,),
                'p': (int, float, Decimal, str),
                'm': (bool, int, float, str, type(None))
            }
    
    @classmethod
    def validate(cls, trade: Any) -> Tuple[bool, str]:
        """
        Valida trade rapidamente sem levantar exceções.
        
        Args:
            trade: Dict com dados do trade
            
        Returns:
            Tuple (is_valid, reason)
            - is_valid: True se válido
            - reason: "ok" ou código de erro
        """
        if not isinstance(trade, dict):
            return False, "not_dict"
        
        # Campos obrigatórios
        for field in cls.required_fields:
            if field not in trade:
                return False, f"missing_{field}"
        
        # Tipos
        schema = cls()
        for field, value in trade.items():
            if field in schema.field_types:
                expected_types = schema.field_types[field]
                if not isinstance(value, expected_types):
                    return False, f"invalid_type_{field}"
        
        return True, "ok"
    
    @classmethod
    def validate_and_extract(
        cls, 
        trade: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Valida e extrai dados processados do trade.
        
        Returns:
            Tuple (is_valid, reason, processed_data)
            - processed_data: Dict com qty, ts, price, is_buyer_maker, conversion
        """
        valid, reason = cls.validate(trade)
        if not valid:
            return False, reason, None
        
        try:
            qty = float(trade['q'])
            ts = int(trade['T'])
            price = float(trade['p'])
            
            # Validação de valores
            if qty <= 0:
                return False, "invalid_qty_value", None
            if price <= 0:
                return False, "invalid_price_value", None
            if ts <= 0:
                return False, "invalid_ts_value", None
            
            # Processamento de is_buyer_maker
            is_buyer_maker_raw = trade.get('m')
            is_buyer_maker = False
            conversion = False
            
            if isinstance(is_buyer_maker_raw, bool):
                is_buyer_maker = is_buyer_maker_raw
            elif isinstance(is_buyer_maker_raw, (int, float)):
                is_buyer_maker = bool(int(is_buyer_maker_raw))
            elif isinstance(is_buyer_maker_raw, str):
                is_buyer_maker = is_buyer_maker_raw.strip().lower() in {
                    "true", "t", "1", "yes"
                }
                conversion = True
            elif is_buyer_maker_raw is None:
                is_buyer_maker = False
            else:
                return False, "invalid_maker_type", None
            
            processed = {
                'qty': qty,
                'ts': ts,
                'price': price,
                'is_buyer_maker': is_buyer_maker,
                'conversion': conversion
            }
            
            return True, "ok", processed
            
        except (ValueError, TypeError) as e:
            return False, f"parse_error_{type(e).__name__}", None


# ==============================================================================
# OHLC VALIDATION
# ==============================================================================

def validate_ohlc(
    open_p: float, 
    high_p: float, 
    low_p: float, 
    close_p: float
) -> bool:
    """
    Validação completa de OHLC.
    
    Verifica:
    1. Todos são números válidos
    2. Open e Close são positivos
    3. High >= Low
    4. High >= max(Open, Close)
    5. Low <= min(Open, Close)
    
    Args:
        open_p: Preço de abertura
        high_p: Preço máximo
        low_p: Preço mínimo
        close_p: Preço de fechamento
        
    Returns:
        True se OHLC é válido
        
    Examples:
        >>> validate_ohlc(100, 110, 95, 105)
        True
        >>> validate_ohlc(100, 90, 95, 105)  # High < Low
        False
    """
    # Tipo check
    if not all(isinstance(x, (int, float)) for x in [open_p, high_p, low_p, close_p]):
        return False
    
    # Valores válidos
    if any(x <= 0 for x in [open_p, close_p]):
        return False
    
    # Relações OHLC
    if high_p < low_p:
        return False
    
    if high_p < max(open_p, close_p):
        return False
    
    if low_p > min(open_p, close_p):
        return False
    
    return True


def fix_ohlc(
    open_p: float, 
    high_p: float, 
    low_p: float, 
    close_p: float,
    fallback_price: Optional[float] = None
) -> Tuple[float, float, float, float]:
    """
    Corrige OHLC inválido.
    
    Args:
        open_p, high_p, low_p, close_p: Valores OHLC
        fallback_price: Preço para usar se tudo inválido
        
    Returns:
        Tuple OHLC corrigido
    """
    # Se já válido, retorna
    if validate_ohlc(open_p, high_p, low_p, close_p):
        return (open_p, high_p, low_p, close_p)
    
    # Tenta corrigir
    try:
        # Garante valores positivos
        prices = [p for p in [open_p, high_p, low_p, close_p] 
                  if isinstance(p, (int, float)) and p > 0]
        
        if not prices:
            # Fallback total
            fb = fallback_price if fallback_price and fallback_price > 0 else 1.0
            return (fb, fb, fb, fb)
        
        # Recalcula
        new_open = open_p if open_p > 0 else prices[0]
        new_close = close_p if close_p > 0 else prices[-1]
        new_high = max(prices)
        new_low = min(prices)
        
        return (new_open, new_high, new_low, new_close)
        
    except Exception:
        fb = fallback_price if fallback_price and fallback_price > 0 else 1.0
        return (fb, fb, fb, fb)


# ==============================================================================
# ABSORPTION GUARD
# ==============================================================================

def guard_absorcao(
    delta: float, 
    rotulo: str, 
    eps: float, 
    mode: str = "warn"
) -> bool:
    """
    Validação de consistência para absorção.
    
    Verifica se o rótulo de absorção é consistente com o delta:
    - Delta negativo → deveria ser "Absorção de Compra"
    - Delta positivo → deveria ser "Absorção de Venda"
    
    Args:
        delta: Delta BTC
        rotulo: Rótulo de absorção
        eps: Epsilon para considerar neutro
        mode: "off", "warn", ou "raise"
        
    Returns:
        True se consistente, False se há mismatch
        
    Raises:
        AbsorptionGuardError: Se mode="raise" e há mismatch
    """
    try:
        mode = (mode or "warn").strip().lower()
    except Exception:
        mode = "warn"
    
    if mode == "off":
        return True
    
    rotulo = (rotulo or "").strip()
    
    # Só valida se é absorção
    if "Absorção" not in rotulo:
        return True
    
    # Detecta mismatch
    # Delta negativo significa mais vendas → absorção de COMPRA (compradores absorveram)
    # Delta positivo significa mais compras → absorção de VENDA (vendedores absorveram)
    mismatch = False
    
    if delta < -eps and "Compra" not in rotulo:
        mismatch = True
    elif delta > eps and "Venda" not in rotulo:
        mismatch = True
    
    if mismatch:
        msg = (
            f"[ABSORCAO_GUARD] delta={delta:.4f} eps={eps} "
            f"rotulo='{rotulo}' (modo={mode})"
        )
        
        if mode == "raise":
            raise AbsorptionGuardError(delta, rotulo, eps)
        
        if lazy_log.should_log("absorcao_guard"):
            logging.warning(msg)
        
        return False
    
    return True


# ==============================================================================
# CONFIG VALIDATION (usando dataclass simples, sem Pydantic)
# ==============================================================================

@dataclass
class FlowAnalyzerConfigValidator:
    """
    Validador de configuração do FlowAnalyzer.
    
    Valida todos os parâmetros de configuração e retorna
    valores corrigidos quando possível.
    """
    
    @staticmethod
    def validate_whale_threshold(value: Any) -> float:
        """Valida whale threshold (deve ser > 0)."""
        try:
            val = float(value)
            if val <= 0:
                raise ValueError("whale_threshold must be > 0")
            return val
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid whale_threshold: {e}")
    
    @staticmethod
    def validate_absorcao_eps(value: Any) -> float:
        """Valida epsilon de absorção (deve ser >= 0)."""
        try:
            val = float(value)
            if val < 0:
                raise ValueError("absorcao_eps must be >= 0")
            return val
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid absorcao_eps: {e}")
    
    @staticmethod
    def validate_guard_mode(value: Any) -> str:
        """Valida modo do guard (off, warn, raise)."""
        try:
            mode = str(value).lower().strip()
            if mode not in ('off', 'warn', 'raise'):
                raise ValueError("absorcao_guard_mode must be 'off', 'warn', or 'raise'")
            return mode
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid absorcao_guard_mode: {e}")
    
    @staticmethod
    def validate_windows(value: Any) -> list:
        """Valida janelas de tempo (lista de inteiros positivos)."""
        try:
            if not isinstance(value, (list, tuple)):
                raise ValueError("windows must be a list")
            
            windows = [int(w) for w in value]
            
            if not windows:
                raise ValueError("windows cannot be empty")
            
            if any(w <= 0 for w in windows):
                raise ValueError("all windows must be > 0")
            
            return sorted(set(windows))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid windows: {e}")
    
    @classmethod
    def validate_all(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida todas as configurações.
        
        Args:
            config: Dict com configurações
            
        Returns:
            Dict com configurações validadas
            
        Raises:
            ValueError: Se alguma configuração é inválida
        """
        validated = {}
        errors = []
        
        validators = {
            'whale_threshold': cls.validate_whale_threshold,
            'absorcao_eps': cls.validate_absorcao_eps,
            'absorcao_guard_mode': cls.validate_guard_mode,
            'net_flow_windows_min': cls.validate_windows,
        }
        
        for key, value in config.items():
            if key in validators:
                try:
                    validated[key] = validators[key](value)
                except ValueError as e:
                    errors.append(str(e))
            else:
                # Passa sem validação
                validated[key] = value
        
        if errors:
            raise ValueError("; ".join(errors))
        
        return validated