# data_validator.py - VERSÃO CORRIGIDA v2.0

import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import hashlib
import logging
import numpy as np

class DataValidator:
    """Validador completo de dados do sistema com CORREÇÃO AUTOMÁTICA de whale delta"""
    
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
        
        # Contadores de correções
        self.corrections_count = {
            'whale_delta': 0,
            'year': 0,
            'timestamp': 0,
            'volumes': 0
        }

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
            
        # 6. Corrigir dados inconsistentes (INCLUINDO WHALE DELTA)
        data = self._correct_inconsistencies(data)
        
        # 7. Normalizar valores
        data = self._normalize_values(data)
        
        # 8. Log de estatísticas de correção
        self._log_correction_stats()
        
        return data

    def _fix_year(self, data: Dict) -> Dict:
        """Corrige anos futuros para o ano atual"""
        corrections_made = False
        
        for key, value in data.items():
            if isinstance(value, str) and "2025" in value:
                data[key] = value.replace("2025", str(self.current_year))
                corrections_made = True
                
            # Recursão para dicionários aninhados
            if isinstance(value, dict):
                data[key] = self._fix_year(value)
                
            # Processar listas
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        data[key][i] = self._fix_year(item)
        
        if corrections_made:
            self.corrections_count['year'] += 1
                        
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
                
        # 3. Validação de volumes (COM CORREÇÃO AUTOMÁTICA)
        if 'fluxo_continuo' in data and 'sector_flow' in data['fluxo_continuo']:
            if not self._validate_and_fix_volumes(data['fluxo_continuo']['sector_flow']):
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

    def _validate_and_fix_volumes(self, sector_flow: Dict) -> bool:
        """Valida e CORRIGE consistência dos volumes"""
        # Verificar e CORRIGIR whale volume
        whale = sector_flow.get('whale', {})
        if whale:
            buy = whale.get('buy', 0)
            sell = whale.get('sell', 0)
            delta = whale.get('delta', 0)
            
            # FÓRMULA CORRETA: delta = buy - sell
            correct_delta = buy - sell
            
            # Detectar inconsistência
            if abs(delta - correct_delta) > 0.01:
                self.logger.warning(
                    f"🔧 CORRIGINDO whale delta: ERRADO={delta:.3f} → CORRETO={correct_delta:.3f} "
                    f"(buy={buy:.3f}, sell={sell:.3f})"
                )
                # CORRIGIR O VALOR
                whale['delta'] = correct_delta
                sector_flow['whale'] = whale
                self.corrections_count['whale_delta'] += 1
                
        # Verificar se os volumes são positivos
        for role in ['retail', 'mid', 'whale']:
            role_data = sector_flow.get(role, {})
            if role_data.get('buy', 0) < 0 or role_data.get('sell', 0) < 0:
                self.logger.warning(f"Volume negativo para {role} - corrigindo para zero")
                if role_data.get('buy', 0) < 0:
                    role_data['buy'] = 0
                if role_data.get('sell', 0) < 0:
                    role_data['sell'] = 0
                sector_flow[role] = role_data
                self.corrections_count['volumes'] += 1
                
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
            # Não falha, apenas avisa
            
        return True

    def _correct_inconsistencies(self, data: Dict) -> Dict:
        """Corrige inconsistências conhecidas"""
        # CORRIGIR WHALE VOLUME EM TODOS OS LUGARES
        if 'fluxo_continuo' in data:
            data = self._reconcile_whale_volume(data)
            
        # Corrigir whale delta no nível raiz se existir
        if 'whale_delta' in data and 'whale_buy_volume' in data and 'whale_sell_volume' in data:
            correct_delta = data['whale_buy_volume'] - data['whale_sell_volume']
            if abs(data['whale_delta'] - correct_delta) > 0.01:
                self.logger.warning(
                    f"🔧 CORRIGINDO whale_delta raiz: {data['whale_delta']:.3f} → {correct_delta:.3f}"
                )
                data['whale_delta'] = correct_delta
                self.corrections_count['whale_delta'] += 1
            
        # Corrigir timestamps
        if 'timestamp' in data and not data['timestamp'].endswith('Z'):
            data['timestamp'] = f"{data['timestamp']}Z"
            self.corrections_count['timestamp'] += 1
            
        return data

    def _reconcile_whale_volume(self, data: Dict) -> Dict:
        """Reconcilia e CORRIGE whale volume entre diferentes seções"""
        fluxo = data.get('fluxo_continuo', {})
        sector_flow = fluxo.get('sector_flow', {})
        whale = sector_flow.get('whale', {})
        
        if whale:
            buy = whale.get('buy', 0)
            sell = whale.get('sell', 0)
            
            # CALCULAR DELTA CORRETO
            correct_delta = buy - sell
            
            # Verificar e corrigir delta no sector_flow
            if 'delta' in whale:
                old_delta = whale['delta']
                if abs(old_delta - correct_delta) > 0.01:
                    self.logger.info(
                        f"✅ Whale delta CORRIGIDO em sector_flow: "
                        f"{old_delta:.3f} → {correct_delta:.3f} (buy={buy:.3f}, sell={sell:.3f})"
                    )
                    whale['delta'] = correct_delta
                    self.corrections_count['whale_delta'] += 1
            
            # Atualizar/corrigir campos do fluxo_contínuo
            if 'whale_buy_volume' in fluxo:
                fluxo['whale_buy_volume'] = buy
            if 'whale_sell_volume' in fluxo:
                fluxo['whale_sell_volume'] = sell
            if 'whale_delta' in fluxo:
                old_fluxo_delta = fluxo['whale_delta']
                if abs(old_fluxo_delta - correct_delta) > 0.01:
                    self.logger.info(
                        f"✅ Whale delta CORRIGIDO em fluxo_continuo: "
                        f"{old_fluxo_delta:.3f} → {correct_delta:.3f}"
                    )
                    fluxo['whale_delta'] = correct_delta
                    self.corrections_count['whale_delta'] += 1
            
        # Também verificar no nível raiz
        if 'whale_buy_volume' in data and 'whale_sell_volume' in data:
            root_buy = data['whale_buy_volume']
            root_sell = data['whale_sell_volume']
            root_correct_delta = root_buy - root_sell
            
            if 'whale_delta' in data:
                if abs(data['whale_delta'] - root_correct_delta) > 0.01:
                    self.logger.info(
                        f"✅ Whale delta CORRIGIDO no nível raiz: "
                        f"{data['whale_delta']:.3f} → {root_correct_delta:.3f}"
                    )
                    data['whale_delta'] = root_correct_delta
                    self.corrections_count['whale_delta'] += 1
            
        return data

    def _normalize_values(self, data: Dict) -> Dict:
        """Normaliza valores para precisão adequada"""
        # Limitar precisão decimal
        decimal_precision = 4
        for key in ['delta', 'volume_total', 'preco_fechamento', 'indice_absorcao', 'whale_delta']:
            if key in data and isinstance(data[key], (int, float)):
                data[key] = round(data[key], decimal_precision)
                
        # Normalizar em fluxo_continuo também
        if 'fluxo_continuo' in data:
            for key in ['whale_delta', 'whale_buy_volume', 'whale_sell_volume']:
                if key in data['fluxo_continuo'] and isinstance(data['fluxo_continuo'][key], (int, float)):
                    data['fluxo_continuo'][key] = round(data['fluxo_continuo'][key], decimal_precision)
                
        return data

    def _cleanup_old_cache(self, current_time: float):
        """Limpa cache de eventos antigos"""
        # Como seen_events agora é um set, não podemos iterar com timestamp
        # Vamos limpar periodicamente todo o cache se ficar muito grande
        if len(self.seen_events) > 1000:
            self.logger.info("Limpando cache de eventos (>1000 entradas)")
            self.seen_events.clear()

    def _log_correction_stats(self):
        """Loga estatísticas de correções a cada 100 eventos"""
        total_corrections = sum(self.corrections_count.values())
        if total_corrections > 0 and total_corrections % 100 == 0:
            self.logger.info(
                f"📊 Estatísticas de Correções (total={total_corrections}): "
                f"whale_delta={self.corrections_count['whale_delta']}, "
                f"year={self.corrections_count['year']}, "
                f"timestamp={self.corrections_count['timestamp']}, "
                f"volumes={self.corrections_count['volumes']}"
            )

    def get_correction_stats(self) -> Dict[str, int]:
        """Retorna estatísticas de correções realizadas"""
        return self.corrections_count.copy()

# Inicialização global
validator = DataValidator(min_absorption_index=0.02, max_orderbook_change=0.3)

# Função auxiliar para teste
def test_whale_delta_correction():
    """Testa a correção do whale delta"""
    test_data = {
        'timestamp': '2024-01-10T10:21:59Z',
        'delta': 100,
        'volume_total': 1000,
        'preco_fechamento': 50000,
        'whale_buy_volume': 59.902,
        'whale_sell_volume': 50.491,
        'whale_delta': -1.82,  # ERRADO! Deveria ser +9.411
        'fluxo_continuo': {
            'whale_delta': -1.82,  # ERRADO também aqui
            'whale_buy_volume': 59.902,
            'whale_sell_volume': 50.491,
            'sector_flow': {
                'whale': {
                    'buy': 59.902,
                    'sell': 50.491,
                    'delta': -1.82  # ERRADO aqui também
                }
            }
        }
    }
    
    print("🧪 Testando correção de whale delta...")
    print(f"Antes: whale_delta = {test_data['whale_delta']}")
    
    corrected = validator.validate_and_clean(test_data)
    
    if corrected:
        print(f"Depois: whale_delta = {corrected['whale_delta']}")
        print(f"Correções realizadas: {validator.get_correction_stats()}")
    else:
        print("❌ Validação falhou")

if __name__ == "__main__":
    # Executar teste ao rodar o arquivo diretamente
    test_whale_delta_correction()