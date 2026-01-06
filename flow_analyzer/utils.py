# flow_analyzer/utils.py
"""
Funções utilitárias do FlowAnalyzer.

Inclui:
- Conversões numéricas seguras
- Arredondamento com Decimal
- Validações de UI invariance
- Helpers diversos
"""

import logging
import time
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Tuple, Any, Optional

from .constants import (
    DECIMAL_ZERO,
    DECIMAL_CENT,
    DECIMAL_PRECISION_BTC,
    DECIMAL_PRECISION_USD,
    UI_TOLERANCE_USD,
    LAZY_LOG_INTERVAL_MS,
)


# ==============================================================================
# LAZY LOGGING
# ==============================================================================

class LazyLog:
    """
    Helper para logging lazy com rate limiting.
    
    Evita spam de logs repetitivos limitando a frequência por chave.
    
    Example:
        >>> lazy_log = LazyLog(interval_ms=1000)
        >>> if lazy_log.should_log("my_warning"):
        ...     logging.warning("Algo aconteceu")
    """
    
    def __init__(self, interval_ms: int = LAZY_LOG_INTERVAL_MS):
        self.last_log: dict = {}
        self.interval_ms = interval_ms
    
    def should_log(self, key: str) -> bool:
        """
        Verifica se deve logar para a chave dada.
        
        Args:
            key: Identificador único do tipo de log
            
        Returns:
            True se passou tempo suficiente desde o último log
        """
        now = time.monotonic() * 1000
        if key not in self.last_log or (now - self.last_log[key]) > self.interval_ms:
            self.last_log[key] = now
            return True
        return False
    
    def reset(self, key: Optional[str] = None) -> None:
        """Reseta rate limiting para uma ou todas as chaves."""
        if key is None:
            self.last_log.clear()
        elif key in self.last_log:
            del self.last_log[key]


# Instância global para uso conveniente
lazy_log = LazyLog()


# ==============================================================================
# CONVERSÕES NUMÉRICAS
# ==============================================================================

def to_decimal(value: Any) -> Decimal:
    """
    Converte valor para Decimal de forma segura e eficiente.
    
    Args:
        value: Valor a converter (int, float, str, Decimal ou None)
        
    Returns:
        Decimal correspondente, ou DECIMAL_ZERO se conversão falhar
        
    Examples:
        >>> to_decimal(1.5)
        Decimal('1.5')
        >>> to_decimal("3.14")
        Decimal('3.14')
        >>> to_decimal(None)
        Decimal('0')
        >>> to_decimal(Decimal('2.5'))
        Decimal('2.5')
    """
    if value is None:
        return DECIMAL_ZERO
    
    if isinstance(value, Decimal):
        return value
    
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            if lazy_log.should_log("decimal_conversion_number"):
                logging.warning(f"⚠️ Falha ao converter número {value} para Decimal")
            return DECIMAL_ZERO
    
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as e:
        if lazy_log.should_log("decimal_conversion"):
            logging.warning(f"⚠️ Conversão falhou para {value}: {e}")
        return DECIMAL_ZERO


def decimal_round(value: float, decimals: int = DECIMAL_PRECISION_BTC) -> float:
    """
    Arredonda usando Decimal para evitar erros de ponto flutuante.
    
    Args:
        value: Valor a arredondar
        decimals: Número de casas decimais
        
    Returns:
        Valor arredondado como float
        
    Examples:
        >>> decimal_round(1.23456789, 4)
        1.2346
        >>> decimal_round(0.1 + 0.2, 1)
        0.3
    """
    try:
        d = to_decimal(value)
        quantize_str = '0.' + '0' * decimals
        rounded = d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
        return float(rounded)
    except (InvalidOperation, ValueError):
        return round(value, decimals)


def quantize_usd(d: Decimal) -> Decimal:
    """
    Quantiza Decimal para 2 casas (centavos USD).
    
    Args:
        d: Valor Decimal
        
    Returns:
        Valor quantizado para centavos
    """
    try:
        return d.quantize(DECIMAL_CENT, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return Decimal('0.00')


# ==============================================================================
# UI INVARIANCE
# ==============================================================================

def ui_safe_round_usd(
    buy_dec: Decimal, 
    sell_dec: Decimal, 
    tol: Decimal = UI_TOLERANCE_USD
) -> Tuple[float, float, float, bool, float, float]:
    """
    Garante invariância de UI: total == buy + sell (arredondado).
    
    Para exibição em UI, é crucial que os valores arredondados
    mantenham consistência matemática.
    
    Args:
        buy_dec: Volume de compra em Decimal
        sell_dec: Volume de venda em Decimal
        tol: Tolerância para discrepância
        
    Returns:
        Tuple contendo:
        - buy_rounded: Volume compra arredondado (float)
        - sell_rounded: Volume venda arredondado (float)
        - total_rounded: Total arredondado (float)
        - is_ok: Se invariância está OK (bool)
        - tolerance: Tolerância usada (float)
        - gap: Discrepância encontrada (float)
    """
    try:
        total_dec = buy_dec + sell_dec
    except (TypeError, InvalidOperation):
        buy_dec = DECIMAL_ZERO
        sell_dec = DECIMAL_ZERO
        total_dec = DECIMAL_ZERO
    
    buy_r = quantize_usd(buy_dec)
    sell_r = quantize_usd(sell_dec)
    total_r = quantize_usd(total_dec)
    
    sum_components = buy_r + sell_r
    gap_original = abs(total_r - sum_components)
    
    if gap_original > tol:
        if lazy_log.should_log("ui_invariant_usd"):
            logging.warning(
                f"[UI-INVARIANT-USD] |total - (buy+sell)|={gap_original} > {tol}"
            )
        # Ajuste para garantir invariância
        total_r = sum_components
    
    is_ok = gap_original <= tol
    
    return (
        float(buy_r), 
        float(sell_r), 
        float(total_r), 
        bool(is_ok), 
        float(tol), 
        float(gap_original)
    )


def ui_safe_round_btc(
    buy_dec: Decimal, 
    sell_dec: Decimal,
    decimals: int = DECIMAL_PRECISION_BTC
) -> Tuple[float, float, float, float]:
    """
    Garante invariância de UI para BTC.
    
    Args:
        buy_dec: Volume de compra em Decimal
        sell_dec: Volume de venda em Decimal
        decimals: Precisão decimal
        
    Returns:
        Tuple contendo:
        - buy_rounded: Volume compra arredondado
        - sell_rounded: Volume venda arredondado
        - total_rounded: Total (sempre soma dos componentes)
        - diff: Diferença entre cálculos
    """
    try:
        total_dec = buy_dec + sell_dec
    except (TypeError, InvalidOperation):
        buy_dec = DECIMAL_ZERO
        sell_dec = DECIMAL_ZERO
        total_dec = DECIMAL_ZERO
    
    buy_r = decimal_round(float(buy_dec), decimals=decimals)
    sell_r = decimal_round(float(sell_dec), decimals=decimals)
    total_r_calc = decimal_round(float(total_dec), decimals=decimals)
    total_r_sum = decimal_round(buy_r + sell_r, decimals=decimals)
    
    diff = abs(total_r_calc - total_r_sum)
    
    if diff > 10 ** (-decimals):
        if lazy_log.should_log("ui_invariant_btc"):
            logging.debug(
                f"[UI-INVARIANT-BTC] Ajustando total de {total_r_calc:.{decimals}f} "
                f"→ {total_r_sum:.{decimals}f} (diff={diff:.{decimals}f})"
            )
    
    # Sempre usar soma dos componentes para consistência
    return buy_r, sell_r, total_r_sum, diff


# ==============================================================================
# BOUNDED CONTAINERS
# ==============================================================================

class BoundedErrorCounter:
    """
    Contador de erros com limite de chaves para evitar memory leak.
    
    Mantém apenas as N chaves mais recentes, removendo as mais antigas
    quando o limite é atingido.
    
    Example:
        >>> counter = BoundedErrorCounter(max_keys=3)
        >>> counter.increment("error_a")
        >>> counter.increment("error_b")
        >>> counter.increment("error_c")
        >>> counter.increment("error_d")  # Remove error_a
        >>> "error_a" in counter.get_all()
        False
    """
    
    def __init__(self, max_keys: int = 100):
        from collections import OrderedDict
        self._counts: OrderedDict = OrderedDict()
        self._max_keys = max_keys
    
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Incrementa contador para a chave.
        
        Returns:
            Novo valor do contador
        """
        if key in self._counts:
            self._counts.move_to_end(key)
            self._counts[key] += amount
        else:
            # Remove mais antigo se necessário
            while len(self._counts) >= self._max_keys:
                self._counts.popitem(last=False)
            self._counts[key] = amount
        
        return self._counts[key]
    
    def get(self, key: str) -> int:
        """Retorna contador para a chave (0 se não existe)."""
        return self._counts.get(key, 0)
    
    def get_all(self) -> dict:
        """Retorna cópia de todos os contadores."""
        return dict(self._counts)
    
    def reset(self) -> None:
        """Reseta todos os contadores."""
        self._counts.clear()
    
    def __len__(self) -> int:
        return len(self._counts)


# ==============================================================================
# TIME HELPERS
# ==============================================================================

def get_current_time_ms() -> int:
    """Retorna timestamp atual em milliseconds."""
    return int(time.time() * 1000)


def elapsed_ms(start_perf_counter: float) -> float:
    """
    Calcula tempo decorrido em milliseconds.
    
    Args:
        start_perf_counter: Valor de time.perf_counter() no início
        
    Returns:
        Tempo decorrido em ms
    """
    return (time.perf_counter() - start_perf_counter) * 1000


# ==============================================================================
# MISC HELPERS
# ==============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Limita valor entre min e max."""
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divisão segura que retorna default se denominador é zero."""
    if denominator == 0:
        return default
    return numerator / denominator