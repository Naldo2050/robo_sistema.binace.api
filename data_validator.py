# data_validator.py

import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import hashlib
import logging
import numpy as np

class DataValidator:
    """Validador completo de dados do sistema"""
    
    def __init__(self, min_absorption_index=0.02, max_orderbook_change=0.3):
        """
        min_absorption_index: Mínimo para considerar absorção válida (2%)
        max_orderbook_change: Máximo de mudança percentual no orderbook (30%)
        """
        self.seen_events = set()
        self.last_orderbook = {}
        self.last_whale_data = {}
        self.logger = logging.getLogger("DataValidator")
        
        # Configurações
        self.min_absorption_index = min_absorption_index
        self.max_orderbook_change = max_orderbook_change
        self.deduplication_window = 30  # segundos
        self.current_year = datetime.now().year

    def validate_and_clean(self, data: Dict) -> Optional[Dict]:
        """Valida e limpa dados antes do processamento"""
        if not data:
            return None
            
        # 1. Corrigir data (2025 → ano atual)
        data = self._fix_year(data)
        
        # 2. Validar estrutura básica
        if not self._validate_structure(data):
            return None
            
        # 3. Remover duplicatas
        event_id = self._generate_event_id(data)
        if event_id in self.seen_events:
            self.logger.debug(f"Evento duplicado ignorado: {event_id}")
            return None
        self.seen_events.add(event_id)
        
        # 4. Limpar cache antigo
        current_time = time.time()
        self._cleanup_old_cache(current_time)
        
        # 5. Validar dados específicos
        if not self._validate_data(data):
            return None
            
        # 6. Corrigir dados inconsistentes
        data = self._correct_inconsistencies(data)
        
        # 7. Normalizar valores
        data = self._normalize_values(data)
        
        return data

    def _fix_year(self, data: Dict) -> Dict:
        """Corrige anos futuros para o ano atual"""
        for key, value in data.items():
            if isinstance(value, str) and "2025" in value:
                data[key] = value.replace("2025", str(self.current_year))
                
            # Recursão para dicionários aninhados
            if isinstance(value, dict):
                data[key] = self._fix_year(value)
                
            # Processar listas
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        data[key][i] = self._fix_year(item)
                        
        return data

    def _generate_event_id(self, data: Dict) -> str:
        """Gera ID único para deduplicação"""
        # Campos mais relevantes para identificação única
        key = f"{data.get('timestamp', '')}|{data.get('delta', '')}|{data.get('volume_total', '')}|{data.get('preco_fechamento', '')}"
        return hashlib.md5(key.encode()).hexdigest()

    def _validate_structure(self, data: Dict) -> bool:
        """Valida estrutura básica dos dados"""
        required_fields = ['timestamp', 'delta', 'volume_total', 'preco_fechamento']
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Campo ausente: {field}")
                return False
                
        # Verificar se é um evento real ou apenas um trigger
        if data.get('is_signal', False) and data.get('tipo_evento', '') == 'ANALYSIS_TRIGGER':
            self.logger.debug("Evento de trigger encontrado - validação relaxada")
            return True
            
        # Validação mais rigorosa para eventos reais
        required_real_fields = ['ml_features', 'orderbook_data', 'historical_vp']
        for field in required_real_fields:
            if field not in data:
                self.logger.warning(f"Campo opcional ausente: {field}")
                
        return True

    def _validate_data(self, data: Dict) -> bool:
        """Valida a integridade dos dados"""
        # 1. Validação de absorção
        if 'indice_absorcao' in data:
            if abs(data['indice_absorcao']) < self.min_absorption_index:
                self.logger.warning(f"Índice de absorção muito baixo: {data['indice_absorcao']:.4%}")
                return False
        
        # 2. Validação de order book
        if 'orderbook_data' in data:
            if not self._validate_orderbook(data['orderbook_data']):
                return False
                
        # 3. Validação de volumes
        if 'fluxo_continuo' in data and 'sector_flow' in data['fluxo_continuo']:
            if not self._validate_volumes(data['fluxo_continuo']['sector_flow']):
                return False
                
        # 4. Validação de participação
        if 'participant_analysis' in data:
            if not self._validate_participant_analysis(data['participant_analysis']):
                return False
                
        # 5. Validação de tempo
        if 'age_ms' in data:
            if data['age_ms'] < 0:
                self.logger.warning(f"age_ms negativo: {data['age_ms']}")
                return False
                
        return True

    def _validate_orderbook(self, orderbook: Dict) -> bool:
        """Valida dados do orderbook"""
        if 'bid_depth_usd' not in orderbook or 'ask_depth_usd' not in orderbook:
            return True  # Não requeridos para validação básica
            
        # Calcular razão
        bid = orderbook['bid_depth_usd']
        ask = orderbook['ask_depth_usd']
        
        if bid <= 0 or ask <= 0:
            return False
            
        # Verificar mudanças bruscas
        if self.last_orderbook:
            last_bid = self.last_orderbook.get('bid_depth_usd', 0)
            last_ask = self.last_orderbook.get('ask_depth_usd', 0)
            
            if last_bid > 0 and last_ask > 0:
                bid_change = abs(bid - last_bid) / last_bid
                ask_change = abs(ask - last_ask) / last_ask
                
                if bid_change > self.max_orderbook_change or ask_change > self.max_orderbook_change:
                    self.logger.error(f"Mudança brusca no orderbook: {bid_change:.2%}/{ask_change:.2%}")
                    return False
                    
        # Atualizar último orderbook
        self.last_orderbook = orderbook.copy()
        return True

    def _validate_volumes(self, sector_flow: Dict) -> bool:
        """Valida consistência dos volumes"""
        # Verificar se whale volume está consistente
        whale = sector_flow.get('whale', {})
        if whale:
            buy = whale.get('buy', 0)
            sell = whale.get('sell', 0)
            delta = whale.get('delta', 0)
            
            # Valores devem seguir a relação: delta = buy - sell
            if abs(delta - (buy - sell)) > 0.01:
                self.logger.warning(f"Inconsistência no whale volume: delta={delta}, buy={buy}, sell={sell}")
                return False
                
        # Verificar se os volumes são positivos
        for role in ['retail', 'mid', 'whale']:
            role_data = sector_flow.get(role, {})
            if role_data.get('buy', 0) < 0 or role_data.get('sell', 0) < 0:
                self.logger.warning(f"Volume negativo para {role}")
                return False
                
        return True

    def _validate_participant_analysis(self, analysis: Dict) -> bool:
        """Valida análise de participantes"""
        total = 0
        for role in ['retail', 'mid', 'whale']:
            pct = analysis.get(role, {}).get('volume_pct', 0)
            total += pct
            
        # Deve somar 100% com tolerância
        if abs(total - 100) > 0.5:
            self.logger.warning(f"Percentuais não somam 100%: {total:.2f}%")
            return False
            
        return True

    def _correct_inconsistencies(self, data: Dict) -> Dict:
        """Corrige inconsistências conhecidas"""
        # Corrigir whale volume
        if 'fluxo_continuo' in data and 'sector_flow' in data['fluxo_continuo']:
            data = self._reconcile_whale_volume(data)
            
        # Corrigir timestamps
        if 'timestamp' in data and not data['timestamp'].endswith('Z'):
            data['timestamp'] = f"{data['timestamp']}Z"
            
        return data

    def _reconcile_whale_volume(self, data: Dict) -> Dict:
        """Reconcilia whale volume entre diferentes seções"""
        sector_flow = data['fluxo_continuo'].get('sector_flow', {})
        whale = sector_flow.get('whale', {})
        
        if whale:
            # Atualizar campos do fluxo_contínuo com dados do sector_flow
            data['fluxo_continuo']['whale_buy_volume'] = whale.get('buy', 0)
            data['fluxo_continuo']['whale_sell_volume'] = whale.get('sell', 0)
            data['fluxo_continuo']['whale_delta'] = whale.get('delta', 0)
            
        return data

    def _normalize_values(self, data: Dict) -> Dict:
        """Normaliza valores para precisão adequada"""
        # Limitar precisão decimal
        decimal_precision = 4
        for key in ['delta', 'volume_total', 'preco_fechamento', 'indice_absorcao']:
            if key in data and isinstance(data[key], (int, float)):
                data[key] = round(data[key], decimal_precision)
                
        return data

    def _cleanup_old_cache(self, current_time: float):
        """Limpa cache de eventos antigos"""
        to_remove = []
        for event_id, timestamp in self.seen_events.items():
            if current_time - timestamp > self.deduplication_window:
                to_remove.append(event_id)
                
        for event_id in to_remove:
            if event_id in self.seen_events:
                del self.seen_events[event_id]

# Inicialização global
validator = DataValidator(min_absorption_index=0.02, max_orderbook_change=0.3)