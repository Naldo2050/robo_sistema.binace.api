# format_utils.py - Utilitários centralizados de formatação (v3)
from typing import Union, Optional, List
import math

def format_price(value: Union[float, int, None], decimals: Optional[int] = None, for_json: bool = False) -> str:
    """
    Formata preço com decimais inteligentes (2-8 casas)
    - Valores > 1000: 2 decimais
    - Valores > 100: 3 decimais  
    - Valores > 1: 4 decimais
    - Valores < 1: até 8 decimais significativos
    
    Args:
        value: Valor a formatar
        decimals: Força número específico de decimais
        for_json: Se True, não usa separador de milhar
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0.00"
    
    try:
        value = float(value)
        if value == 0:
            return "0.00"
            
        if decimals is not None:
            if for_json:
                return f"{value:.{decimals}f}"
            else:
                return f"{value:,.{decimals}f}"
            
        # Lógica inteligente baseada no tamanho do valor
        if abs(value) >= 1000:
            dec = 2
        elif abs(value) >= 100:
            dec = 3
        elif abs(value) >= 1:
            dec = 4
        else:
            # Para valores < 1, usa até 8 decimais mas remove zeros à direita
            if for_json:
                formatted = f"{value:.8f}".rstrip('0').rstrip('.')
            else:
                formatted = f"{value:.8f}".rstrip('0').rstrip('.')
            if '.' not in formatted:
                formatted += ".00"
            elif len(formatted.split('.')[1]) < 2:
                formatted += "0"
            return formatted
        
        if for_json:
            return f"{value:.{dec}f}"
        else:
            return f"{value:,.{dec}f}"
    except:
        return "0.00"

def format_quantity(value: Union[float, int, None], for_json: bool = False) -> str:
    """
    Formata quantidade removendo decimais desnecessários.
    Inteiros são mostrados sem .0
    
    Args:
        value: Valor a formatar
        for_json: Se True, não usa separador de milhar
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0"
    
    try:
        value = float(value)
        if value == int(value):
            if for_json:
                return str(int(value))
            else:
                return f"{int(value):,}"
        else:
            # Máximo 2 decimais para quantidades fracionárias
            if for_json:
                return f"{value:.2f}".rstrip('0').rstrip('.')
            else:
                return f"{value:,.2f}".rstrip('0').rstrip('.')
    except:
        return "0"

def format_percent(value: Union[float, int, None], decimals: int = 2, as_fraction: bool = False) -> str:
    """
    Formata percentual com 2 casas decimais por padrão.
    
    Args:
        value: Valor a formatar
        decimals: Número de casas decimais (padrão 2)
        as_fraction: Se True, retorna como fração (0.60) ao invés de percentual (60.00%)
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0.00%" if not as_fraction else "0.00"
    
    try:
        value = float(value)
        
        if as_fraction:
            # Retorna como fração: 0.60 ao invés de 60%
            if abs(value) > 1:
                # Assume que já está em percentual, converte para fração
                return f"{value/100:.{decimals}f}"
            else:
                return f"{value:.{decimals}f}"
        else:
            # Retorna como percentual: 60.00%
            # Se valor < 1, assume que é fração e multiplica por 100
            if abs(value) <= 1.0 and value != 0:
                return f"{value*100:.{decimals}f}%"
            else:
                return f"{value:.{decimals}f}%"
    except:
        return "0.00%" if not as_fraction else "0.00"

def format_large_number(value: Union[float, int, None], force_decimals: int = 2, for_json: bool = False) -> str:
    """
    Formata números grandes com notação K/M/B.
    
    Args:
        value: Valor a formatar
        force_decimals: Número de decimais
        for_json: Se True, retorna número puro sem notação K/M/B
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0"
    
    try:
        value = float(value)
        
        if for_json:
            # Para JSON, retorna número puro sem formatação
            if value == int(value):
                return str(int(value))
            else:
                return f"{value:.{force_decimals}f}"
        
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        
        if abs_value >= 1_000_000_000:
            return f"{sign}{abs_value/1_000_000_000:.{force_decimals}f}B"
        elif abs_value >= 1_000_000:
            return f"{sign}{abs_value/1_000_000:.{force_decimals}f}M"
        elif abs_value >= 10_000:
            return f"{sign}{abs_value/1_000:.{force_decimals}f}K"
        else:
            return format_quantity(value)
    except:
        return "0"

def format_delta(value: Union[float, int, None], for_json: bool = False) -> str:
    """
    Formata delta com sinal + ou -.
    
    Args:
        value: Valor a formatar
        for_json: Se True, não usa formatação especial
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0.00" if for_json else "+0.00"
    
    try:
        value = float(value)
        
        if for_json:
            return f"{value:.2f}"
        
        sign = "+" if value >= 0 else ""
        # Para deltas grandes, usa notação compacta
        if abs(value) >= 10_000:
            return f"{sign}{format_large_number(value)}"
        else:
            return f"{sign}{value:,.2f}"
    except:
        return "0.00" if for_json else "+0.00"

def format_time_seconds(value: Union[float, int, None], show_unit: bool = True) -> str:
    """
    Formata tempo em segundos.
    
    Args:
        value: Valor em segundos (ou millisegundos se > 1000)
        show_unit: Se True, adiciona 's' ao final
    """
    if value is None:
        return "0s" if show_unit else "0"
    
    try:
        value = float(value)
        
        # Se valor > 1000, assume millisegundos
        if value > 1000:
            value = value / 1000.0
        
        if value >= 60:
            minutes = int(value / 60)
            secs = value % 60
            return f"{minutes}m {secs:.1f}s" if show_unit else f"{minutes}:{secs:04.1f}"
        else:
            return f"{value:.1f}s" if show_unit else f"{value:.1f}"
    except:
        return "0s" if show_unit else "0"

def format_scientific(value: Union[float, int, None], decimals: int = 4) -> str:
    """
    Formata números muito pequenos em notação científica quando necessário
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0.0000"
    
    try:
        value = float(value)
        if value == 0:
            return "0.0000"
        
        # Se muito pequeno, usa notação científica
        if 0 < abs(value) < 0.0001:
            return f"{value:.{decimals}e}"
        else:
            return f"{value:.{decimals}f}"
    except:
        return "0.0000"

def format_epoch_ms(value: Union[float, int, None]) -> str:
    """
    Formata epoch em millisegundos (sem separadores, sem decimais)
    """
    if value is None:
        return "0"
    
    try:
        # Remove decimais e retorna como string pura
        return str(int(value))
    except:
        return "0"

def format_ratio(value: Union[float, int, None], decimals: int = 4) -> str:
    """
    Formata ratios como frações (0.41) sem símbolo %
    Sempre retorna valor entre -1 e 1 (ou próximo)
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "0.0000"
    
    try:
        value = float(value)
        
        # Se valor > 1, assume que veio como percentual e converte
        if abs(value) > 1:
            value = value / 100.0
            
        return f"{value:.{decimals}f}"
    except:
        return "0.0000"

def format_integer(value: Union[float, int, None]) -> str:
    """
    Formata inteiros sem casas decimais
    """
    if value is None:
        return "0"
    
    try:
        return str(int(value))
    except:
        return "0"

def normalize_ratio_value(value: Union[str, float, int]) -> float:
    """
    Normaliza valores de ratio/imbalance/gradient/pressure para fração (0-1 ou -1 a 1).
    
    Args:
        value: Valor a normalizar (pode ser "60.46%", 60.46, ou 0.6046)
        
    Returns:
        Valor normalizado como fração
    """
    if value is None:
        return 0.0
        
    try:
        # Se for string
        if isinstance(value, str):
            # Remove espaços e vírgulas
            cleaned = value.strip().replace(',', '')
            
            # Se termina com %, remove e divide por 100
            if cleaned.endswith('%'):
                cleaned = cleaned[:-1]
                return round(float(cleaned) / 100.0, 4)
            else:
                # Converte direto
                num = float(cleaned)
                # Se > 1, assume percentual e converte
                if abs(num) > 1:
                    return round(num / 100.0, 4)
                return round(num, 4)
        
        # Se for número
        else:
            num = float(value)
            # Se > 1, assume percentual e converte
            if abs(num) > 1:
                return round(num / 100.0, 4)
            return round(num, 4)
            
    except:
        return 0.0

def normalize_percent_value(value: Union[str, float, int]) -> float:
    """
    Normaliza valores percentuais para escala 0-100.
    
    Args:
        value: Valor a normalizar (pode ser "60.46%", 0.6046, ou 60.46)
        
    Returns:
        Valor normalizado como percentual 0-100
    """
    if value is None:
        return 0.0
        
    try:
        # Se for string
        if isinstance(value, str):
            # Remove espaços e vírgulas
            cleaned = value.strip().replace(',', '')
            
            # Se termina com %, remove e retorna direto
            if cleaned.endswith('%'):
                cleaned = cleaned[:-1]
                return round(float(cleaned), 2)
            else:
                # Converte direto
                num = float(cleaned)
                # Se <= 1, assume fração e multiplica por 100
                if abs(num) <= 1.0:
                    return round(num * 100.0, 2)
                return round(num, 2)
        
        # Se for número
        else:
            num = float(value)
            # Se <= 1, assume fração e multiplica por 100
            if abs(num) <= 1.0:
                return round(num * 100.0, 2)
            return round(num, 2)
            
    except:
        return 0.0

def auto_format(key: str, value: Union[float, int, None], for_json: bool = False) -> Union[str, float, int]:
    """
    Formata automaticamente baseado no nome do campo.
    
    Args:
        key: Nome do campo
        value: Valor a formatar
        for_json: Se True, retorna valor numérico ao invés de string formatada
    
    Returns:
        Valor formatado (string para display, número para JSON)
    """
    if value is None:
        return None if for_json else "N/A"
    
    key_lower = key.lower()
    
    # Epoch/timestamps
    if 'epoch' in key_lower or key_lower.endswith('_ms'):
        return int(value) if for_json else format_epoch_ms(value)
    
    # Inteiros
    if any(x in key_lower for x in ['count', 'num_', 'day_of_week', 'trades_count']):
        return int(value) if for_json else format_integer(value)
    
    # Tempo/duração
    if any(x in key_lower for x in ['duration', 'time_to_', '_seconds', '_s']):
        return float(value) if for_json else format_time_seconds(value)
    
    # 🔹 CORREÇÃO: Ratios/imbalances/gradients/pressure (frações 0-1)
    if any(x in key_lower for x in ['ratio', 'imbalance', 'gradient', 'pressure']) and \
       not any(x in key_lower for x in ['pct', 'percent']):
        if for_json:
            return normalize_ratio_value(value)
        else:
            return format_ratio(normalize_ratio_value(value))
    
    # 🔹 CORREÇÃO: Percentuais (0-100)
    if any(x in key_lower for x in ['pct', 'percent', 'prob']):
        if for_json:
            return normalize_percent_value(value)
        else:
            return format_percent(normalize_percent_value(value), as_fraction=False)
    
    # Preços
    if any(x in key_lower for x in ['price', 'preco', 'poc', 'val', 'vah', 'level']):
        if for_json:
            return float(format_price(value, for_json=True))
        else:
            return format_price(value)
    
    # Volumes/quantidades
    if any(x in key_lower for x in ['volume', 'quantity', 'size']):
        if for_json:
            return float(value) if value != int(value) else int(value)
        else:
            return format_quantity(value)
    
    # Deltas
    if 'delta' in key_lower or 'flow' in key_lower:
        if for_json:
            return round(float(value), 2)
        else:
            return format_delta(value)
    
    # Volatilidade/científicos
    if any(x in key_lower for x in ['volatility', 'returns', 'slope', 'momentum']):
        if for_json:
            return round(float(value), 6)
        else:
            return format_scientific(value)
    
    # Genérico
    if for_json:
        if isinstance(value, float) and value == int(value):
            return int(value)
        else:
            return round(float(value), 4)
    else:
        return str(value)