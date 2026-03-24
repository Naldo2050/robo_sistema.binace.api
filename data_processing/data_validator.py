# -*- coding: utf-8 -*-
# data_validator.py v2.3.3 - SUPER-VALIDATOR + QUALITY METRICS

import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import hashlib
import logging
import numpy as np

# Importar métricas de qualidade (opcional, sem quebrar se não existir)
try:
    from data_pipeline.metrics.data_quality_metrics import get_quality_metrics
    _QUALITY_METRICS_AVAILABLE = True
except ImportError:
    _QUALITY_METRICS_AVAILABLE = False

class DataValidator:
    """
    Validador e limpador de dados completo com precisão máxima.
    
    🔹 CORREÇÕES v2.3.2:
      ✅ Mesmo comportamento de v2.3.1, com correção extra:
         - Check "Whale buy volume excede volume total de compra" só é aplicado
           quando os volumes claramente estão no MESMO horizonte (sem fluxo_continuo).
         - Evita comparar whale acumulado (desde o reset) com volume_compra da janela.
    
    🔹 CORREÇÕES v2.3.1:
      ✅ Precisão de 8 casas decimais para volumes BTC
      ✅ Validação rigorosa de timestamps (positivos, em range válido, first <= last)
      ✅ Correção automática de age_ms negativo
      ✅ Tolerâncias adequadas para cada tipo de campo
      ✅ Validação de consistência whale_delta em TODOS os níveis
      ✅ Logs detalhados de correções
      ✅ Contadores separados por tipo de correção
    """
    
    # 🆕 Constantes de precisão
    BTC_PRECISION = 8       # Volumes BTC
    PRICE_PRECISION = 4     # Preços USDT
    RATIO_PRECISION = 6     # Ratios e percentuais
    
    # 🆕 Tolerâncias por tipo
    BTC_TOLERANCE = 2e-4    # 0.0002 BTC (~$14 @ $71K, cobre float rounding)
    USD_TOLERANCE = 0.01    # $0.01
    
    # 🆕 Limites de timestamp válido
    MIN_VALID_TIMESTAMP_MS = 1609459200000  # 2021-01-01 00:00:00 UTC
    MAX_VALID_TIMESTAMP_MS = 2147483647000  # 2038-01-19 03:14:07 UTC (limite do Unix timestamp de 32 bits)
    
    def __init__(self, min_absorption_index=0.02, max_orderbook_change=0.3):
        self.seen_events = set()
        self.last_orderbook = {}
        self.logger = logging.getLogger("DataValidator")
        
        self.min_absorption_index = min_absorption_index
        self.max_orderbook_change = max_orderbook_change
        self.current_year = datetime.now().year
        
        # 🆕 Contadores mais detalhados
        self.corrections_count = {
            'recalculated_delta': 0,
            'recalculated_whale_delta': 0,
            'recalculated_poc_percentage': 0,
            'reconciled_total_volume': 0,
            'reconciled_whale_volume': 0,
            'fixed_utf8_encoding': 0,
            'sanitized_session_time': 0,
            'year': 0,
            'timestamp': 0,
            'volumes': 0,
            'volume_consistency_failed': 0,
            'participant_direction_mismatch': 0,
            'temporal_inconsistency': 0,
            'timestamp_validation_failed': 0,
            'age_ms_corrected': 0,
            'first_last_seen_corrected': 0,
            'precision_corrections': 0,
        }
        self.last_event_timestamp_ms = 0

    def validate_and_clean(self, data: Dict) -> Optional[Dict]:
        """Pipeline completo de validação e limpeza de dados."""
        start_time = time.perf_counter()
        corrections_applied: List[str] = []
        
        if not data:
            self._record_quality_metric("discarded", [], start_time)
            return None
            
        # 1. Correções primárias (antes da validação)
        data = self._fix_year(data)
        data = self._fix_utf8_encoding(data)
        
        # 2. Validar e corrigir timestamps ANTES de outras validações
        _ts_result = self._validate_and_fix_timestamps(data)
        if _ts_result is None:
            self._record_quality_metric("discarded", ["timestamp_validation_failed"], start_time)
            return None
        data = _ts_result
        
        # 3. Validar estrutura e remover duplicatas
        if not self._validate_structure(data):
            self._record_quality_metric("discarded", ["structure_invalid"], start_time)
            return None
        event_id = self._generate_event_id(data)
        if event_id in self.seen_events:
            self.logger.debug(f"Evento duplicado ignorado: {event_id}")
            self._record_quality_metric("discarded", ["duplicate"], start_time)
            return None
        self.seen_events.add(event_id)
        
        # 4. Limpeza de cache
        self._cleanup_old_cache()
        
        # 5. Correções de inconsistências lógicas
        corrections_before = self._get_correction_snapshot()
        data = self._correct_all_inconsistencies(data)
        corrections_applied = self._get_corrections_since(corrections_before)
        
        # 6. Validação de dados pós-correção
        if not self._validate_data_integrity(data):
            self._record_quality_metric("discarded", ["integrity_failed"], start_time)
            return None
            
        # 7. Normalização final COM PRECISÃO CORRETA
        data = self._normalize_values(data)
        
        # 8. Log de estatísticas
        self._log_correction_stats()
        
        # 9. Registrar métrica de qualidade
        status = "corrected" if corrections_applied else "valid"
        self._record_quality_metric(status, corrections_applied, start_time)
        
        return data
    
    def _record_quality_metric(
        self,
        status: str,
        correction_types: List[str],
        start_time: float
    ) -> None:
        """Registra métrica de qualidade de dados."""
        if not _QUALITY_METRICS_AVAILABLE:
            return
        
        try:
            latency_ms = (time.perf_counter() - start_time) * 1000
            get_quality_metrics().record_event(
                status=status,
                correction_types=correction_types,
                latency_ms=latency_ms
            )
        except Exception:
            pass  # Não falhar validação por erro de métricas
    
    def _get_correction_snapshot(self) -> Dict[str, int]:
        """Retorna snapshot dos contadores de correção."""
        return self.corrections_count.copy()
    
    def _get_corrections_since(self, snapshot: Dict[str, int]) -> List[str]:
        """Retorna tipos de correção aplicados desde o snapshot."""
        corrections = []
        for key, current_val in self.corrections_count.items():
            if current_val > snapshot.get(key, 0):
                corrections.append(key)
        return corrections

    # ========================================================================
    # VALIDAÇÃO E CORREÇÃO DE TIMESTAMPS
    # ========================================================================
    
    def _validate_and_fix_timestamps(self, data: Dict) -> Optional[Dict]:
        """
        Valida e corrige todos os timestamps no evento.
        
        Validações:
        - Timestamps são positivos
        - Timestamps estão em range válido (2021-2038)
        - first_seen_ms <= last_seen_ms
        - age_ms >= 0
        
        IMPROVEMENTS v2.3.4:
        - Lógica de validação mais inteligente
        - Evita correções desnecessárias em timestamps válidos
        """
        try:
            # Valida timestamp principal
            timestamp_fields = ['epoch_ms', 'timestamp_utc', 'timestamp']
            main_timestamp = None
            
            for field in timestamp_fields:
                if field in data:
                    if field == 'epoch_ms':
                        main_timestamp = data[field]
                    elif field in ['timestamp_utc', 'timestamp']:
                        try:
                            ts_str = data[field]
                            # Normaliza timezone antes de validar
                            if '+00:00' in ts_str:
                                ts_str = ts_str.replace('+00:00', 'Z')
                                data[field] = ts_str  # Atualiza no data também
                            
                            # Só parse se não for um timestamp muito distante
                            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                            main_timestamp = int(dt.timestamp() * 1000)
                        except Exception:
                            pass
                    
                    if main_timestamp is not None:
                        break
            
            # Valida timestamp principal
            if main_timestamp is not None:
                if not self._is_valid_timestamp(main_timestamp):
                    self.logger.error(
                        f"❌ Timestamp principal inválido: {main_timestamp}"
                    )
                    self.corrections_count['timestamp_validation_failed'] += 1
                    return None
            
            # Valida e corrige first_seen_ms e last_seen_ms
            data = self._fix_first_last_seen(data, main_timestamp)
            
            # Valida e corrige age_ms
            data = self._fix_age_ms(data, main_timestamp)
            
            # Valida timestamps em nested structures
            if 'fluxo_continuo' in data:
                if 'liquidity_heatmap' in data['fluxo_continuo']:
                    heatmap = data['fluxo_continuo']['liquidity_heatmap']
                    if 'clusters' in heatmap:
                        for cluster in heatmap['clusters']:
                            cluster = self._fix_first_last_seen(cluster, main_timestamp)
                            cluster = self._fix_age_ms(cluster, main_timestamp)
            
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao validar timestamps: {e}", exc_info=True)
            return None
    
    def _is_valid_timestamp(self, ts_ms: int) -> bool:
        """Verifica se timestamp está em range válido."""
        try:
            ts = int(ts_ms)
            if ts <= 0:
                return False
            if ts < self.MIN_VALID_TIMESTAMP_MS or ts > self.MAX_VALID_TIMESTAMP_MS:
                return False
            return True
        except Exception:
            return False
    
    def _fix_first_last_seen(
        self, 
        data: Dict, 
        reference_ts_ms: Optional[int]
    ) -> Dict:
        """Corrige first_seen_ms e last_seen_ms garantindo first <= last."""
        try:
            first_seen = data.get('first_seen_ms')
            last_seen = data.get('last_seen_ms')
            
            if first_seen is not None and last_seen is not None:
                first = int(first_seen)
                last = int(last_seen)
                
                if first <= 0 or last <= 0:
                    self.logger.warning(
                        f"⚠️ first_seen ou last_seen não-positivo: "
                        f"first={first}, last={last}"
                    )
                    if reference_ts_ms and reference_ts_ms > 0:
                        data['first_seen_ms'] = reference_ts_ms
                        data['last_seen_ms'] = reference_ts_ms
                        self.corrections_count['first_last_seen_corrected'] += 1
                    return data
                
                if first > last:
                    self.logger.warning(
                        f"⚠️ TIMESTAMP INVERTIDO: "
                        f"first_seen ({first}) > last_seen ({last}). "
                        f"Invertendo valores."
                    )
                    data['first_seen_ms'] = last
                    data['last_seen_ms'] = first
                    self.corrections_count['first_last_seen_corrected'] += 1
            
            elif reference_ts_ms and reference_ts_ms > 0:
                if first_seen is None and last_seen is not None:
                    data['first_seen_ms'] = min(int(last_seen), reference_ts_ms)
                    self.corrections_count['first_last_seen_corrected'] += 1
                elif last_seen is None and first_seen is not None:
                    data['last_seen_ms'] = max(int(first_seen), reference_ts_ms)
                    self.corrections_count['first_last_seen_corrected'] += 1
            
            return data
            
        except Exception as e:
            self.logger.error(f"Erro ao corrigir first/last seen: {e}")
            return data
    
    def _fix_age_ms(self, data: Dict, reference_ts_ms: Optional[int]) -> Dict:
        """Corrige age_ms garantindo que seja >= 0."""
        try:
            age_ms = data.get('age_ms')
            
            if age_ms is not None:
                age = int(age_ms)
                
                if age < 0:
                    self.logger.warning(
                        f"⚠️ age_ms negativo: {age}. "
                        f"Tentando recalcular..."
                    )
                    
                    last_seen = data.get('last_seen_ms')
                    if last_seen and reference_ts_ms:
                        recalculated_age = reference_ts_ms - int(last_seen)
                        if recalculated_age >= 0:
                            data['age_ms'] = recalculated_age
                            self.corrections_count['age_ms_corrected'] += 1
                            self.logger.info(
                                f"✅ age_ms corrigido: {age} → {recalculated_age}"
                            )
                        else:
                            data['age_ms'] = 0
                            self.corrections_count['age_ms_corrected'] += 1
                    else:
                        data['age_ms'] = 0
                        self.corrections_count['age_ms_corrected'] += 1
            
            return data
            
        except Exception as e:
            self.logger.error(f"Erro ao corrigir age_ms: {e}")
            return data

    # ========================================================================
    # CORREÇÕES DE INCONSISTÊNCIAS (COM PRECISÃO CORRETA)
    # ========================================================================

    def _fix_utf8_encoding(self, data: Any) -> Any:
        """Corrige recursivamente problemas de encoding UTF-8 em strings."""
        if isinstance(data, dict):
            corrected_dict = {}
            for k, v in data.items():
                corrected_dict[k] = self._fix_utf8_encoding(v)
            return corrected_dict
        elif isinstance(data, list):
            return [self._fix_utf8_encoding(item) for item in data]
        elif isinstance(data, str):
            replacements = {
                "AbsorÃ§Ã£o": "Absorção", "Absorï¿½ï¿½o": "Absorção",
                "AcumulaÃ§Ã£o": "Acumulação", "Acumulaï¿½ï¿½o": "Acumulação",
                "ManipulaÃ§Ã£o": "Manipulação", "Manipulaï¿½ï¿½o": "Manipulação",
                "DistribuiÃ§Ã£o": "Distribuição", "Distribuiï¿½ï¿½o": "Distribuição",
                "ConsolidaÃ§Ã£o": "Consolidação", "Consolidaï¿½ï¿½o": "Consolidação",
            }
            original_value = data
            for wrong, right in replacements.items():
                data = data.replace(wrong, right)
            
            if original_value != data:
                self.corrections_count['fixed_utf8_encoding'] += 1
            return data
        return data

    def _correct_all_inconsistencies(self, data: Dict) -> Dict:
        """Orquestra todas as funções de correção de dados."""
        data = self._reconcile_total_volume(data)
        data = self._reconcile_whale_volume(data)
        data = self._recalculate_deltas(data)
        data = self._recalculate_whale_delta(data)
        data = self._recalculate_poc_percentage(data)
        data = self._sanitize_session_time(data)
        
        if 'timestamp' in data and data['timestamp']:
            ts = data['timestamp']
            # Só adiciona 'Z' se não tiver timezone explícito
            if not (ts.endswith('Z') or '+' in ts[-6:] or '-' in ts[-6:]):
                data['timestamp'] += 'Z'
                self.corrections_count['timestamp'] += 1
            # Converte timezone +00:00 para Z para padronizar
            elif '+00:00' in ts:
                data['timestamp'] = ts.replace('+00:00', 'Z')
                if data['timestamp'] != ts:
                    self.corrections_count['timestamp'] += 1
            
        return data

    def _recalculate_deltas(self, data: Dict) -> Dict:
        """Recalcula 'delta' e 'delta_fechamento' com base nos volumes de compra/venda."""
        if 'volume_compra' in data and 'volume_venda' in data:
            buy = float(data['volume_compra'])
            sell = float(data['volume_venda'])
            correct_delta = buy - sell
            
            if abs(float(data.get('delta', 0)) - correct_delta) > self.BTC_TOLERANCE:
                self.corrections_count['recalculated_delta'] += 1
                data['delta'] = round(correct_delta, self.BTC_PRECISION)

        if 'enriched_snapshot' in data and 'delta_fechamento' in data['enriched_snapshot']:
            buy = float(data.get('volume_compra', 0))
            sell = float(data.get('volume_venda', 0))
            correct_delta = buy - sell
            
            if abs(float(data['enriched_snapshot'].get('delta_fechamento', 0)) - correct_delta) > self.BTC_TOLERANCE:
                self.corrections_count['recalculated_delta'] += 1
                data['enriched_snapshot']['delta_fechamento'] = round(correct_delta, self.BTC_PRECISION)

        return data

    def _recalculate_whale_delta(self, data: Dict) -> Dict:
        """Recalcula o whale_delta em TODOS os lugares onde aparece."""
        whale_buy = 0
        whale_sell = 0
        
        if 'whale_buy_volume' in data and 'whale_sell_volume' in data:
            whale_buy = float(data['whale_buy_volume'])
            whale_sell = float(data['whale_sell_volume'])
        elif 'fluxo_continuo' in data:
            fluxo = data['fluxo_continuo']
            if 'whale_buy_volume' in fluxo and 'whale_sell_volume' in fluxo:
                whale_buy = float(fluxo['whale_buy_volume'])
                whale_sell = float(fluxo['whale_sell_volume'])
            elif 'sector_flow' in fluxo and 'whale' in fluxo['sector_flow']:
                whale_sector = fluxo['sector_flow']['whale']
                whale_buy = float(whale_sector.get('buy', 0))
                whale_sell = float(whale_sector.get('sell', 0))
        
        correct_whale_delta = whale_buy - whale_sell
        
        if 'whale_delta' in data:
            current_delta = float(data['whale_delta'])
            if abs(current_delta - correct_whale_delta) > self.BTC_TOLERANCE:
                self.corrections_count['recalculated_whale_delta'] += 1
                data['whale_delta'] = round(correct_whale_delta, self.BTC_PRECISION)
                self.logger.debug(
                    f"Corrigido whale_delta na raiz: "
                    f"{current_delta:.8f} -> {correct_whale_delta:.8f}"
                )
        
        if 'fluxo_continuo' in data:
            fluxo = data['fluxo_continuo']
            if 'whale_delta' in fluxo:
                current_delta = float(fluxo['whale_delta'])
                if abs(current_delta - correct_whale_delta) > self.BTC_TOLERANCE:
                    self.corrections_count['recalculated_whale_delta'] += 1
                    fluxo['whale_delta'] = round(correct_whale_delta, self.BTC_PRECISION)
                    self.logger.debug(
                        f"Corrigido whale_delta em fluxo_continuo: "
                        f"{current_delta:.8f} -> {correct_whale_delta:.8f}"
                    )
            
            if 'sector_flow' in fluxo and 'whale' in fluxo['sector_flow']:
                whale_sector = fluxo['sector_flow']['whale']
                if 'delta' in whale_sector:
                    current_delta = float(whale_sector['delta'])
                    if abs(current_delta - correct_whale_delta) > self.BTC_TOLERANCE:
                        self.corrections_count['recalculated_whale_delta'] += 1
                        whale_sector['delta'] = round(correct_whale_delta, self.BTC_PRECISION)
                        self.logger.debug(
                            f"Corrigido whale_delta em sector_flow: "
                            f"{current_delta:.8f} -> {correct_whale_delta:.8f}"
                        )
        
        return data

    def _reconcile_total_volume(self, data: Dict) -> Dict:
        """
        Padroniza o 'volume_total' usando enriched_snapshot como fonte da verdade.
        """
        if 'enriched_snapshot' not in data or 'volume_total' not in data['enriched_snapshot']:
            return data

        authoritative_volume = float(data['enriched_snapshot']['volume_total'])
        
        if abs(float(data.get('volume_total', 0)) - authoritative_volume) > self.BTC_TOLERANCE:
            self.corrections_count['reconciled_total_volume'] += 1
            data['volume_total'] = round(authoritative_volume, self.BTC_PRECISION)

        if 'fluxo_continuo' in data and 'order_flow' in data['fluxo_continuo']:
            flow = data['fluxo_continuo']['order_flow']
            if abs(float(flow.get('total_volume_btc', 0)) - authoritative_volume) > self.BTC_TOLERANCE:
                self.corrections_count['reconciled_total_volume'] += 1
                flow['total_volume_btc'] = round(authoritative_volume, self.BTC_PRECISION)
        
        return data

    def _recalculate_poc_percentage(self, data: Dict) -> Dict:
        """Recalcula 'poc_percentage' com base no volume total corrigido."""
        if 'enriched_snapshot' not in data:
            return data
        
        snapshot = data['enriched_snapshot']
        poc_vol = float(snapshot.get('poc_volume', 0))
        total_vol = float(snapshot.get('volume_total', 0))

        if poc_vol > 0 and total_vol > 0:
            correct_pct = (poc_vol / total_vol) * 100
            if abs(float(snapshot.get('poc_percentage', 0)) - correct_pct) > 0.1:
                self.corrections_count['recalculated_poc_percentage'] += 1
                snapshot['poc_percentage'] = round(correct_pct, 2)
        
        return data

    def _sanitize_session_time(self, data: Dict) -> Dict:
        """Verifica e corrige 'time_to_session_close' se for irrealista."""
        if 'market_context' in data and 'time_to_session_close' in data['market_context']:
            session_time = data['market_context']['time_to_session_close']
            if session_time is not None and int(session_time) > 1440:
                self.corrections_count['sanitized_session_time'] += 1
                data['market_context']['time_to_session_close'] = None
        return data

    def _fix_year(self, data: Dict) -> Dict:
        """
        Corrige anos futuros para o ano atual APENAS em campos que não são timestamps.
        
        Evita corrigir timestamps que são válidos (ex: 2025 em timestamps é normal).
        """
        corrections_made = False
        timestamp_fields = {'timestamp', 'timestamp_utc', 'timestamp_ny', 'timestamp_sp', 
                           'open_time', 'close_time', 'recent_timestamp', 
                           'last_seen_ms', 'first_seen_ms'}
        
        for key, value in data.items():
            # Pula campos de timestamp - eles podem ter anos futuros legitimately
            if key in timestamp_fields or key.endswith('_time') or key.endswith('_timestamp'):
                continue
                
            if isinstance(value, str) and "2025" in value:
                # Verifica se não é um timestamp válido
                if not any(tz_indicator in value for tz_indicator in ['T', 'Z', '+', '-']):
                    data[key] = value.replace("2025", str(self.current_year))
                    corrections_made = True
            elif isinstance(value, dict):
                data[key] = self._fix_year(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        data[key][i] = self._fix_year(item)
                    elif isinstance(item, str) and "2025" in item:
                        # Também verifica se não é um timestamp
                        if not any(tz_indicator in item for tz_indicator in ['T', 'Z', '+', '-']):
                            data[key][i] = item.replace("2025", str(self.current_year))
                            corrections_made = True
        
        if corrections_made:
            self.corrections_count['year'] += 1
        return data

    def _generate_event_id(self, data: Dict) -> str:
        """Gera ID único para deduplicação."""
        timestamp_str = str(data.get('epoch_ms', ''))
        delta_str = f"{float(data.get('delta', 0)):.8f}"
        volume_str = f"{float(data.get('volume_total', 0)):.8f}"
        price_str = f"{float(data.get('preco_fechamento', 0)):.4f}"
        
        key = f"{timestamp_str}|{delta_str}|{volume_str}|{price_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def _validate_structure(self, data: Dict) -> bool:
        """Valida a estrutura mínima e comum a todos os eventos."""
        if 'timestamp_utc' not in data and 'epoch_ms' not in data and 'timestamp' not in data:
            self.logger.error("Estrutura inválida: evento não possui nenhum campo de timestamp.")
            return False
        return True

    def _validate_data_integrity(self, data: Dict) -> bool:
        """Valida a integridade dos dados após correções."""
        if 'indice_absorcao' in data and abs(data['indice_absorcao']) < self.min_absorption_index:
            # Eventos cujo tipo_evento é Absorção são válidos independente do índice
            tipo = data.get('tipo_evento', '')
            if 'Absor' in str(tipo):
                self.logger.debug(
                    f"Índice de absorção baixo ({data['indice_absorcao']:.4%}) "
                    f"ignorado para evento do tipo '{tipo}'"
                )
            else:
                self.logger.warning(f"Índice de absorção muito baixo: {data['indice_absorcao']:.4%}")
                return False
        
        if 'orderbook_data' in data and not self._validate_orderbook(data['orderbook_data']):
            return False
        
        if 'fluxo_continuo' in data and 'sector_flow' in data['fluxo_continuo']:
            if not self._validate_and_fix_volumes(data['fluxo_continuo']['sector_flow']):
                return False
        
        if not self._validate_volume_consistency(data):
            self.corrections_count['volume_consistency_failed'] += 1
        
        if 'fluxo_continuo' in data and 'participant_analysis' in data['fluxo_continuo']:
            if not self._validate_participant_analysis(data['fluxo_continuo']):
                return False
        
        if not self._validate_temporal_consistency(data):
            self.corrections_count['temporal_inconsistency'] += 1
            return False
        
        return True
    
    def _reconcile_whale_volume(self, data: Dict) -> Dict:
        """
        Reconcilia e CORRIGE whale volume entre sector_flow e campos whale_*.
        """
        fluxo = data.get('fluxo_continuo', {})
        sector_flow = fluxo.get('sector_flow', {})
        whale_sector = sector_flow.get('whale', {})
        
        if whale_sector:
            buy_from_sector = float(whale_sector.get('buy', 0))
            sell_from_sector = float(whale_sector.get('sell', 0))
            
            if 'whale_buy_volume' in fluxo:
                if abs(float(fluxo['whale_buy_volume']) - buy_from_sector) > self.BTC_TOLERANCE:
                    fluxo['whale_buy_volume'] = round(buy_from_sector, self.BTC_PRECISION)
                    self.corrections_count['reconciled_whale_volume'] += 1
            elif buy_from_sector > 0:
                fluxo['whale_buy_volume'] = round(buy_from_sector, self.BTC_PRECISION)
                self.corrections_count['reconciled_whale_volume'] += 1
            
            if 'whale_sell_volume' in fluxo:
                if abs(float(fluxo['whale_sell_volume']) - sell_from_sector) > self.BTC_TOLERANCE:
                    fluxo['whale_sell_volume'] = round(sell_from_sector, self.BTC_PRECISION)
                    self.corrections_count['reconciled_whale_volume'] += 1
            elif sell_from_sector > 0:
                fluxo['whale_sell_volume'] = round(sell_from_sector, self.BTC_PRECISION)
                self.corrections_count['reconciled_whale_volume'] += 1
            
            if 'whale_buy_volume' in data:
                if abs(float(data['whale_buy_volume']) - buy_from_sector) > self.BTC_TOLERANCE:
                    data['whale_buy_volume'] = round(buy_from_sector, self.BTC_PRECISION)
                    self.corrections_count['reconciled_whale_volume'] += 1
            elif buy_from_sector > 0:
                data['whale_buy_volume'] = round(buy_from_sector, self.BTC_PRECISION)
                
            if 'whale_sell_volume' in data:
                if abs(float(data['whale_sell_volume']) - sell_from_sector) > self.BTC_TOLERANCE:
                    data['whale_sell_volume'] = round(sell_from_sector, self.BTC_PRECISION)
                    self.corrections_count['reconciled_whale_volume'] += 1
            elif sell_from_sector > 0:
                data['whale_sell_volume'] = round(sell_from_sector, self.BTC_PRECISION)
        
        return data

    def _normalize_values(self, data: Dict) -> Dict:
        """
        Normaliza valores para precisão adequada recursivamente.
        """
        btc_fields = [
            'delta', 'volume_total', 'whale_delta',
            'whale_buy_volume', 'whale_sell_volume', 
            'volume_compra', 'volume_venda', 'cvd'
        ]
        
        price_fields = ['preco_fechamento', 'preco_abertura', 'preco_maximo', 'preco_minimo']
        ratio_fields = ['indice_absorcao']
        
        for key in btc_fields:
            if key in data and isinstance(data[key], (int, float)):
                original = data[key]
                data[key] = round(data[key], self.BTC_PRECISION)
                if abs(original - data[key]) > self.BTC_TOLERANCE:
                    self.corrections_count['precision_corrections'] += 1
        
        for key in price_fields:
            if key in data and isinstance(data[key], (int, float)):
                data[key] = round(data[key], self.PRICE_PRECISION)
        
        for key in ratio_fields:
            if key in data and isinstance(data[key], (int, float)):
                data[key] = round(data[key], self.RATIO_PRECISION)
        
        if 'fluxo_continuo' in data:
            for key in btc_fields:
                if key in data['fluxo_continuo'] and isinstance(data['fluxo_continuo'][key], (int, float)):
                    data['fluxo_continuo'][key] = round(data['fluxo_continuo'][key], self.BTC_PRECISION)
            
            if 'sector_flow' in data['fluxo_continuo']:
                for sector in data['fluxo_continuo']['sector_flow'].values():
                    if isinstance(sector, dict):
                        for key in ['buy', 'sell', 'delta']:
                            if key in sector and isinstance(sector[key], (int, float)):
                                sector[key] = round(sector[key], self.BTC_PRECISION)
        
        if 'enriched_snapshot' in data:
            for key in data['enriched_snapshot']:
                if isinstance(data['enriched_snapshot'][key], (int, float)):
                    data['enriched_snapshot'][key] = round(
                        data['enriched_snapshot'][key], 
                        self.PRICE_PRECISION
                    )
        
        return data

    def _cleanup_old_cache(self):
        """Limpa cache de eventos se ficar muito grande."""
        if len(self.seen_events) > 1000:
            self.logger.info("Limpando cache de eventos (>1000 entradas)")
            self.seen_events.clear()

    def _log_correction_stats(self):
        """Loga estatísticas de correções periodicamente."""
        total_corrections = sum(self.corrections_count.values())
        if total_corrections > 0 and total_corrections % 10 == 0:
            self.logger.info(
                f"📊 Estatísticas de Correções (total={total_corrections}): "
                f"{self.corrections_count}"
            )

    # --- MÉTODOS DE VALIDAÇÃO COMPLETOS ---
    
    def _validate_temporal_consistency(self, data: Dict) -> bool:
        """
        Valida consistência temporal dos timestamps.

        v2.4.0 - Fix: kline boundary timestamps (múltiplos de 60s) não são
        usados como referência temporal. Eles são sintéticos (close time do
        candle) e causavam falsos positivos quando eventos reais chegavam
        com timestamps anteriores ao boundary.
        Tolerância aumentada de 2s → 5s para acomodar clock offset Binance
        e jitter de rede.
        """
        TEMPORAL_TOLERANCE_MS = 5000  # 5s — rede + clock offset residual
        current_timestamp_ms = data.get('epoch_ms')
        if current_timestamp_ms and self.last_event_timestamp_ms > 0:
            time_diff = self.last_event_timestamp_ms - current_timestamp_ms

            if time_diff > TEMPORAL_TOLERANCE_MS:
                self.logger.error(
                    f"❌ Inconsistência temporal SIGNIFICATIVA: "
                    f"diff={time_diff}ms (atual={current_timestamp_ms}, "
                    f"último={self.last_event_timestamp_ms})"
                )
                return False

            elif time_diff > 0:
                self.logger.debug(
                    f"⚠️ Evento fora de ordem (tolerado): "
                    f"diff={time_diff}ms (tolerância={TEMPORAL_TOLERANCE_MS}ms)"
                )
                return True

        if current_timestamp_ms:
            self._update_reference_timestamp(current_timestamp_ms)

        return True

    def _update_reference_timestamp(self, ts_ms: int) -> None:
        """
        Atualiza referência temporal apenas com timestamps reais.

        Kline boundaries (múltiplos de 60000ms) são timestamps sintéticos
        do close do candle — usá-los como referência faz eventos reais
        subsequentes parecerem "no passado", gerando descartes falsos.
        """
        is_kline_boundary = (ts_ms % 60000 == 0)

        if not is_kline_boundary:
            if ts_ms > self.last_event_timestamp_ms:
                self.last_event_timestamp_ms = ts_ms
        else:
            # Boundary: registra apenas se posterior e sem retroagir
            if ts_ms > self.last_event_timestamp_ms:
                self.logger.debug(
                    f"Boundary timestamp {ts_ms} ignorado como referência temporal"
                )

    def _validate_volume_consistency(self, event: Dict) -> bool:
        """
        Valida que os volumes sejam consistentes entre si.
        
        v2.3.2:
          - Check "whale > total" na raiz só é aplicado se NÃO houver fluxo_continuo.
            (evita comparar whale acumulado com volume_compra da janela)
        """
        # Check simples de volume_total = volume_compra + volume_venda
        if 'volume_compra' in event and 'volume_venda' in event and 'volume_total' in event:
            buy = float(event['volume_compra'])
            sell = float(event['volume_venda'])
            total = float(event['volume_total'])
            expected_total = buy + sell
            
            if abs(total - expected_total) > self.BTC_TOLERANCE:
                self.logger.warning(
                    f"Volume total inconsistente: "
                    f"{total:.8f} != {buy:.8f} + {sell:.8f}"
                )
                # Auto-corrige
                event['volume_total'] = round(expected_total, self.BTC_PRECISION)
                return False
        
        # ---- CHECK DE WHALE NA RAIZ (SÓ QUANDO NÃO HÁ FLUXO_CONTINUO) ----
        # Em eventos em tempo real, whale_buy_volume costuma ser ACUMULADO (desde o reset),
        # enquanto volume_compra é da JANELA. Comparar os dois gera falsos positivos.
        if 'fluxo_continuo' not in event:
            if 'whale_buy_volume' in event and 'volume_compra' in event:
                if float(event['whale_buy_volume']) > float(event['volume_compra']) + self.BTC_TOLERANCE:
                    self.logger.warning("Whale buy volume excede volume total de compra")
                    return False
            
            if 'whale_sell_volume' in event and 'volume_venda' in event:
                if float(event['whale_sell_volume']) > float(event['volume_venda']) + self.BTC_TOLERANCE:
                    self.logger.warning("Whale sell volume excede volume total de venda")
                    return False
        
        # (Opcional) Futuro: aqui poderíamos validar também whale_*_window vs buy/sell_volume_btc
        # dentro de event['fluxo_continuo']['order_flow'], se necessário.
        
        return True

    def _validate_orderbook(self, orderbook: Dict) -> bool:
        """Valida mudanças no orderbook."""
        if not orderbook:
            return True
        
        for side in ['bid', 'ask']:
            if side in orderbook:
                for level_data in orderbook[side].values():
                    if isinstance(level_data, dict) and 'change' in level_data:
                        change = abs(float(level_data['change']))
                        if change > self.max_orderbook_change:
                            self.logger.warning(
                                f"Mudança muito grande no orderbook {side}: {change}"
                            )
                            return False
        
        self.last_orderbook = orderbook.copy()
        return True

    def _validate_and_fix_volumes(self, sector_flow: Dict) -> bool:
        """
        Valida e corrige volumes por setor.
        """
        for sector_name, sector_data in sector_flow.items():
            if not isinstance(sector_data, dict):
                continue
            
            buy = float(sector_data.get('buy', 0))
            sell = float(sector_data.get('sell', 0))
            delta = float(sector_data.get('delta', 0))
            
            expected_delta = buy - sell
            if abs(delta - expected_delta) > self.BTC_TOLERANCE:
                self.logger.debug(
                    f"Corrigindo delta do setor {sector_name}: "
                    f"{delta:.8f} -> {expected_delta:.8f}"
                )
                sector_data['delta'] = round(expected_delta, self.BTC_PRECISION)
                self.corrections_count['volumes'] += 1
            
            if buy < 0 or sell < 0:
                self.logger.warning(f"Volumes negativos no setor {sector_name}")
                return False
        
        return True

    def _validate_participant_analysis(self, fluxo_continuo: Dict) -> bool:
        """Valida análise de participantes."""
        participant_analysis = fluxo_continuo.get('participant_analysis', {})
        
        if not participant_analysis:
            return True
        
        if 'whale_delta' in fluxo_continuo:
            whale_delta = float(fluxo_continuo['whale_delta'])
            whale_direction = participant_analysis.get('whale_direction', '')
            
            if whale_delta > self.BTC_TOLERANCE and whale_direction == 'vendendo':
                self.logger.warning(
                    f"Inconsistência: whale_delta positivo ({whale_delta:.8f}) "
                    f"mas direção é 'vendendo'"
                )
                self.corrections_count['participant_direction_mismatch'] += 1
                participant_analysis['whale_direction'] = 'comprando'
            elif whale_delta < -self.BTC_TOLERANCE and whale_direction == 'comprando':
                self.logger.warning(
                    f"Inconsistência: whale_delta negativo ({whale_delta:.8f}) "
                    f"mas direção é 'comprando'"
                )
                self.corrections_count['participant_direction_mismatch'] += 1
                participant_analysis['whale_direction'] = 'vendendo'
        
        return True

    def get_correction_stats(self) -> Dict[str, int]:
        """Retorna estatísticas de correções realizadas."""
        return self.corrections_count.copy()

    def reset_stats(self):
        """Reseta contadores de estatísticas."""
        for key in self.corrections_count:
            self.corrections_count[key] = 0


# Inicialização global
validator = DataValidator(min_absorption_index=0.02, max_orderbook_change=0.3)

# Função auxiliar para teste
def test_whale_delta_correction():
    """Testa a correção do whale delta"""
    test_data = {
        'epoch_ms': 1704884519000,
        'timestamp': '2024-01-10T10:21:59Z',
        'delta': 100,
        'volume_total': 1000,
        'volume_compra': 600,
        'volume_venda': 400,
        'preco_fechamento': 50000,
        'whale_buy_volume': 59.902,
        'whale_sell_volume': 50.491,
        'whale_delta': -1.82,
        'fluxo_continuo': {
            'whale_delta': -1.82,
            'whale_buy_volume': 59.902,
            'whale_sell_volume': 50.491,
            'sector_flow': {
                'whale': {
                    'buy': 59.902,
                    'sell': 50.491,
                    'delta': -1.82
                }
            },
            'participant_analysis': {
                'whale_direction': 'vendendo'
            },
            'liquidity_heatmap': {
                'clusters': [{
                    'first_seen_ms': 1704884519999,
                    'last_seen_ms': 1704884519000,
                    'age_ms': -500
                }]
            }
        }
    }
    
    print("="*80)
    print("🧪 TESTE DE CORREÇÃO v2.3.2")
    print("="*80)
    print(f"\n📋 ANTES:")
    print(f"   whale_delta: {test_data['whale_delta']}")
    print(f"   whale_direction: {test_data['fluxo_continuo']['participant_analysis']['whale_direction']}")
    print(f"   first_seen: {test_data['fluxo_continuo']['liquidity_heatmap']['clusters'][0]['first_seen_ms']}")
    print(f"   last_seen: {test_data['fluxo_continuo']['liquidity_heatmap']['clusters'][0]['last_seen_ms']}")
    print(f"   age_ms: {test_data['fluxo_continuo']['liquidity_heatmap']['clusters'][0]['age_ms']}")
    
    logging.basicConfig(level=logging.INFO)
    
    corrected = validator.validate_and_clean(test_data)
    
    if corrected:
        print(f"\n✅ DEPOIS:")
        print(f"   whale_delta: {corrected['whale_delta']:.8f}")
        print(f"   whale_delta (fluxo): {corrected['fluxo_continuo']['whale_delta']:.8f}")
        print(f"   whale_delta (sector): {corrected['fluxo_continuo']['sector_flow']['whale']['delta']:.8f}")
        print(f"   whale_direction: {corrected['fluxo_continuo']['participant_analysis']['whale_direction']}")
        
        cluster = corrected['fluxo_continuo']['liquidity_heatmap']['clusters'][0]
        print(f"   first_seen: {cluster['first_seen_ms']}")
        print(f"   last_seen: {cluster['last_seen_ms']}")
        print(f"   age_ms: {cluster['age_ms']}")
        
        print(f"\n📊 CORREÇÕES REALIZADAS:")
        for key, count in validator.get_correction_stats().items():
            if count > 0:
                print(f"   {key}: {count}")
    else:
        print("❌ Validação falhou")
    
    print("="*80)

if __name__ == "__main__":
    test_whale_delta_correction()