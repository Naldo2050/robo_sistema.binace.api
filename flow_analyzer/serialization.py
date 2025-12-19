# flow_analyzer/serialization.py
"""
Serialização segura para FlowAnalyzer.

Inclui:
- DecimalEncoder para JSON
- Serialização de métricas
- Compressão opcional
"""

import json
import gzip
import io
from decimal import Decimal
from datetime import datetime
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict, is_dataclass


class DecimalEncoder(json.JSONEncoder):
    """
    Encoder JSON que suporta Decimal e outros tipos especiais.
    
    Tipos suportados:
    - Decimal -> float
    - datetime -> ISO string
    - dataclass -> dict
    - bytes -> base64 string
    - set -> list
    
    Example:
        >>> import json
        >>> from decimal import Decimal
        >>> data = {'value': Decimal('1.234567890')}
        >>> json.dumps(data, cls=DecimalEncoder)
        '{"value": 1.23456789}'
    """
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            # Preserva precisão razoável (8 casas para BTC)
            return float(round(obj, 8))
        
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        
        if isinstance(obj, bytes):
            import base64
            return base64.b64encode(obj).decode('ascii')
        
        if isinstance(obj, set):
            return list(obj)
        
        if isinstance(obj, frozenset):
            return list(obj)
        
        # Fallback para representação string
        try:
            return str(obj)
        except Exception:
            return super().default(obj)


class MetricsSerializer:
    """
    Serializador otimizado para métricas do FlowAnalyzer.
    
    Features:
    - Compressão gzip opcional
    - Precisão configurável para floats
    - Remoção de campos nulos
    """
    
    def __init__(
        self,
        compress: bool = False,
        float_precision: int = 8,
        remove_nulls: bool = True,
    ):
        self.compress = compress
        self.float_precision = float_precision
        self.remove_nulls = remove_nulls
    
    def serialize(self, data: Dict[str, Any]) -> Union[str, bytes]:
        """
        Serializa dados para JSON (opcionalmente comprimido).
        
        Args:
            data: Dicionário de métricas
            
        Returns:
            JSON string ou bytes comprimidos
        """
        # Limpa dados
        if self.remove_nulls:
            data = self._remove_nulls(data)
        
        # Serializa
        json_str = json.dumps(
            data,
            cls=DecimalEncoder,
            separators=(',', ':'),  # Compacto
            ensure_ascii=False,
        )
        
        if not self.compress:
            return json_str
        
        # Comprime
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb') as f:
            f.write(json_str.encode('utf-8'))
        return buf.getvalue()
    
    def deserialize(self, data: Union[str, bytes]) -> Dict[str, Any]:
        """
        Deserializa JSON (com suporte a gzip).
        
        Args:
            data: JSON string ou bytes comprimidos
            
        Returns:
            Dicionário de métricas
        """
        if isinstance(data, bytes):
            # Tenta descomprimir
            try:
                buf = io.BytesIO(data)
                with gzip.GzipFile(fileobj=buf, mode='rb') as f:
                    data = f.read().decode('utf-8')
            except gzip.BadGzipFile:
                data = data.decode('utf-8')
        
        return json.loads(data)
    
    def _remove_nulls(self, data: Any) -> Any:
        """Remove valores None recursivamente."""
        if isinstance(data, dict):
            return {
                k: self._remove_nulls(v)
                for k, v in data.items()
                if v is not None
            }
        if isinstance(data, list):
            return [self._remove_nulls(v) for v in data if v is not None]
        return data


def dumps(data: Dict[str, Any], **kwargs) -> str:
    """
    Conveniência: serializa métricas para JSON.
    
    Example:
        >>> from flow_analyzer.serialization import dumps
        >>> metrics = analyzer.get_flow_metrics()
        >>> json_str = dumps(metrics)
    """
    return json.dumps(data, cls=DecimalEncoder, **kwargs)


def loads(data: str) -> Dict[str, Any]:
    """
    Conveniência: deserializa JSON para dict.
    """
    return json.loads(data)