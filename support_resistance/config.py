"""
Classes de configuração do sistema de Suporte/Resistência
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np


# =============================
#  SERIALIZAÇÃO SEGURA
# =============================

class SafeJSONEncoder(json.JSONEncoder):
    """Encoder JSON que lida com tipos Python/NumPy comuns"""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        if hasattr(obj, 'value') and hasattr(obj, '__class__') and issubclass(obj.__class__, Enum):
            return obj.value
        return super().default(obj)


# =============================
#  CONFIGURAÇÕES
# =============================

@dataclass
class SRConfig:
    """Configuração para detecção de suporte e resistência"""
    lookback_period: int = 100
    merge_tolerance: float = 0.01
    cluster_eps_percent: float = 0.005
    min_cluster_size: int = 3
    expected_touches: int = 6
    expected_cluster_size: int = 8
    
    # Pesos para composite score (devem somar 1.0)
    # touches, density, volume, recency, stability, reaction
    weights: tuple = (0.25, 0.12, 0.23, 0.12, 0.14, 0.14)
    
    volume_cap_percentile: float = 0.95
    tol_k: float = 1.5
    min_tol_pct: float = 0.001
    prom_k: float = 1.0
    min_prom_pct: float = 0.002
    
    # Reaction score
    reaction_window: int = 10
    min_reversal_pct: float = 0.1  # Mínimo 0.1% para considerar reversão

    def __post_init__(self):
        """Valida configurações de pesos e parâmetros"""
        if len(self.weights) != 6:
            raise ValueError("SRConfig.weights deve ter exatamente 6 valores (touches, density, volume, recency, stability, reaction).")
        
        weight_sum = sum(self.weights)
        if not (0.999 <= weight_sum <= 1.001):
            raise ValueError(f"SRConfig.weights deve somar 1.0 (atual: {weight_sum}).")


@dataclass
class VolumeProfileConfig:
    """Configuração para análise de Volume Profile"""
    bins: int = 50
    value_area_percent: float = 0.70
    hvn_sigma: float = 1.0
    lvn_sigma: float = 1.0
    min_data_points: int = 20


@dataclass
class MonitorConfig:
    """Configuração para monitoramento em tempo real"""
    tolerance_percent: float = 0.5
    lookback_ticks: int = 100
    max_test_history: int = 1000  # NOVO: limite de histórico de testes
    strong_delta_threshold: float = 0.6
    moderate_delta_threshold: float = 0.3
    trend_strong_threshold: float = 2.0
    trend_weak_threshold: float = 0.5
    high_volatility_threshold: float = 1.0


@dataclass
class PivotConfig:
    """Configuração para Pivot Points"""
    methods: List[str] = field(default_factory=lambda: ["classic", "camarilla", "woodie", "fibonacci"])
    confluence_tolerance_percent: float = 0.5


@dataclass
class InstitutionalConfig:
    """Configuração centralizada do sistema"""
    sr: SRConfig = field(default_factory=SRConfig)
    volume_profile: VolumeProfileConfig = field(default_factory=VolumeProfileConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    pivot: PivotConfig = field(default_factory=PivotConfig)
    
    # Global
    min_data_points: int = 50
    confidence_level: float = 0.95
    enable_cache: bool = True
    enable_performance_logging: bool = False
    
    @classmethod
    def from_json(cls, path: str) -> "InstitutionalConfig":
        """Carrega config de arquivo JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            sr=SRConfig(**data.get('sr', {})),
            volume_profile=VolumeProfileConfig(**data.get('volume_profile', {})),
            monitor=MonitorConfig(**data.get('monitor', {})),
            pivot=PivotConfig(**data.get('pivot', {})),
            min_data_points=data.get('min_data_points', 50),
            confidence_level=data.get('confidence_level', 0.95),
            enable_cache=data.get('enable_cache', True),
            enable_performance_logging=data.get('enable_performance_logging', False)
        )
    
    def to_json(self, path: str, indent: int = 2) -> None:
        """Salva config em arquivo JSON"""
        data = {
            'sr': self._dataclass_to_dict(self.sr),
            'volume_profile': self._dataclass_to_dict(self.volume_profile),
            'monitor': self._dataclass_to_dict(self.monitor),
            'pivot': self._dataclass_to_dict(self.pivot),
            'min_data_points': self.min_data_points,
            'confidence_level': self.confidence_level,
            'enable_cache': self.enable_cache,
            'enable_performance_logging': self.enable_performance_logging,
            '_metadata': {
                'version': '2.0.0',
                'saved_at': datetime.now().isoformat()
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, cls=SafeJSONEncoder)
    
    @staticmethod
    def _dataclass_to_dict(obj) -> Dict:
        """Converte dataclass para dict de forma segura"""
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        return result