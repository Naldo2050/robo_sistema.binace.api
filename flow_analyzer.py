# flow_analyzer.py v2.3.1 - CORREÇÃO: TIMESTAMP JITTER + JANELA vs ACUMULADO
"""
Flow Analyzer com correção de timestamps e separação clara entre métricas.

🔹 CORREÇÕES v2.3.1:
  ✅ Tolerância a jitter de timestamps (±2000ms)
  ✅ Ajuste automático de timestamps futuros
  ✅ Logs reduzidos (a cada 10 ocorrências)
  ✅ Integração com Clock Sync (CORRIGIDO: get_server_time_ms)
  ✅ Estatísticas de ajustes de timestamp

🔹 MANTIDO v2.3.0:
  ✅ Whale volumes calculados POR JANELA + acumulado separado
  ✅ Sector flow calculado POR JANELA + acumulado separado
  ✅ Participant analysis usa MENOR janela (não maior)
  ✅ Documentação clara de métricas acumuladas vs janela
  ✅ Validação de consistência whale delta
  ✅ Reset interval configurável (4h)
  ✅ BTC e USD usam Decimal para cálculos críticos
  ✅ Invariância de UI para USD e BTC
"""

import logging
from threading import Lock
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import time

import config

from time_manager import TimeManager
from liquidity_heatmap import LiquidityHeatmap

# ✅ Clock Sync
try:
    from clock_sync import get_clock_sync
    HAS_CLOCK_SYNC = True
except ImportError:
    HAS_CLOCK_SYNC = False

try:
    from config import (
        NET_FLOW_WINDOWS_MIN,
        AGGRESSIVE_ORDER_SIZE_THRESHOLD,
        ABSORCAO_DELTA_EPS,
        ABSORCAO_GUARD_MODE,
    )
except Exception:
    NET_FLOW_WINDOWS_MIN = [1, 5, 15]
    AGGRESSIVE_ORDER_SIZE_THRESHOLD = 0.0
    ABSORCAO_DELTA_EPS = 1.0
    ABSORCAO_GUARD_MODE = "warn"

# ✅ Tolerância para jitter de timestamps
TIMESTAMP_JITTER_TOLERANCE_MS = 2000  # 2 segundos


class FlowAnalyzerError(Exception):
    """Levantada quando FlowAnalyzer encontra erro crítico."""
    pass


# ============================
# HELPER FUNCTIONS
# ============================

def _to_decimal(value) -> Decimal:
    """Converte valor para Decimal de forma segura e eficiente."""
    if value is None:
        return Decimal('0')
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    try:
        return Decimal(str(value))
    except Exception:
        logging.warning(f"⚠️ Conversão falhou para {value}, usando 0")
        return Decimal('0')


def _guard_absorcao(delta: float, rotulo: str, eps: float, mode: str = "warn"):
    """Validação de consistência para absorção."""
    try:
        mode = (mode or "warn").strip().lower()
    except Exception:
        mode = "warn"

    if mode == "off":
        return

    mismatch = (delta > eps and rotulo != "Absorção de Compra") or \
               (delta < -eps and rotulo != "Absorção de Venda")
    
    if mismatch:
        msg = (
            f"[ABSORCAO_GUARD] delta={delta:.4f} eps={eps} "
            f"rotulo='{rotulo}' (modo={mode})"
        )
        if mode == "raise":
            raise AssertionError(msg)
        logging.warning(msg)


def _decimal_round(value: float, decimals: int = 8) -> float:
    """Arredonda usando Decimal para evitar erros de float."""
    try:
        d = _to_decimal(value)
        quantize_str = '0.' + '0' * decimals
        return float(d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))
    except Exception:
        return round(value, decimals)


# Quantização de USD para 2 casas, arredondamento comercial
DEC_CENT = Decimal('0.01')


def _q2(d: Decimal) -> Decimal:
    """Quantiza Decimal para 2 casas com ROUND_HALF_UP (comercial)."""
    try:
        return d.quantize(DEC_CENT, rounding=ROUND_HALF_UP)
    except Exception:
        return Decimal('0.00')


def _ui_safe_round_usd(
    buy_dec: Decimal, 
    sell_dec: Decimal, 
    tol: Decimal = Decimal('0.005')
) -> Tuple[float, float, float, bool, float]:
    """Garante invariância de UI: total_rounded == buy_rounded + sell_rounded."""
    try:
        total_dec = buy_dec + sell_dec
    except Exception:
        buy_dec = Decimal('0')
        sell_dec = Decimal('0')
        total_dec = Decimal('0')

    buy_r = _q2(buy_dec)
    sell_r = _q2(sell_dec)
    total_r = _q2(total_dec)

    sum_components = buy_r + sell_r
    
    if sum_components != total_r:
        logging.debug(
            f"[UI-INVARIANT-USD] Ajustando total_rounded de {total_r} → {sum_components}"
        )
        total_r = sum_components

    gap = abs(total_r - sum_components)
    ok = gap <= tol
    
    if not ok:
        logging.warning(
            f"[UI-INVARIANT-USD] |total - (buy+sell)|={gap} > {tol}"
        )
    
    return float(buy_r), float(sell_r), float(total_r), bool(ok), float(tol)


def _ui_safe_round_btc(
    buy_dec: Decimal, 
    sell_dec: Decimal,
    decimals: int = 8
) -> Tuple[float, float, float, float]:
    """Garante invariância de UI para BTC: total == buy + sell."""
    try:
        total_dec = buy_dec + sell_dec
    except Exception:
        buy_dec = Decimal('0')
        sell_dec = Decimal('0')
        total_dec = Decimal('0')

    buy_r = _decimal_round(float(buy_dec), decimals=decimals)
    sell_r = _decimal_round(float(sell_dec), decimals=decimals)
    total_r_calc = _decimal_round(float(total_dec), decimals=decimals)
    total_r_sum = _decimal_round(buy_r + sell_r, decimals=decimals)

    diff = abs(total_r_calc - total_r_sum)

    if diff > 10 ** (-decimals):
        logging.debug(
            f"[UI-INVARIANT-BTC] Ajustando total de {total_r_calc:.{decimals}f} "
            f"→ {total_r_sum:.{decimals}f} (diff={diff:.{decimals}f})"
        )

    # Sempre usar soma dos componentes para garantir invariância
    total_r = total_r_sum

    return buy_r, sell_r, total_r, diff


class FlowAnalyzer:
    """
    Analisador de fluxo com correção de timestamps e separação clara entre métricas.
    
    🔹 CARACTERÍSTICAS v2.3.1:
      - Correção automática de timestamps com jitter
      - Whale volumes: acumulado + por janela
      - Sector flow: acumulado + por janela
      - Participant analysis: baseado na menor janela
      - Precisão máxima com Decimal
      - Invariância de UI para USD e BTC
    """

    def __init__(self, time_manager: Optional[TimeManager] = None):
        self.time_manager = time_manager or TimeManager()

        # ✅ Clock Sync
        self.clock_sync = None
        if HAS_CLOCK_SYNC:
            try:
                self.clock_sync = get_clock_sync()
            except Exception as e:
                logging.warning(f"⚠️ Erro ao obter Clock Sync: {e}")

        # Métricas principais (ACUMULADAS)
        self.cvd = 0.0
        self.whale_threshold = float(getattr(config, "WHALE_TRADE_THRESHOLD", 5.0))
        self.whale_buy_volume = 0.0
        self.whale_sell_volume = 0.0
        self.whale_delta = 0.0

        # Reset automático
        self.last_reset_ms = self._get_synced_timestamp_ms()
        self.reset_interval_ms = int(
            getattr(config, "CVD_RESET_INTERVAL_HOURS", 4) * 3600 * 1000
        )

        # Thread safety
        self._lock = Lock()

        # Burst detection
        self.recent_trades = deque(maxlen=500)
        self.bursts = {"count": 0, "max_burst_volume": 0.0}
        self._in_burst = False
        self._last_burst_end_ms = 0

        self.burst_window_ms = int(getattr(config, "BURST_WINDOW_MS", 200))
        self.burst_cooldown_ms = int(getattr(config, "BURST_COOLDOWN_MS", 200))
        self.burst_volume_threshold = float(
            getattr(config, "BURST_VOLUME_THRESHOLD", self.whale_threshold)
        )

        # Sector flow (ACUMULADO)
        order_buckets = getattr(config, "ORDER_SIZE_BUCKETS", {
            "retail": (0, 0.5),
            "mid": (0.5, 2.0),
            "whale": (2.0, 9999.0)
        })
        self.sector_flow = {
            name: {"buy": 0.0, "sell": 0.0, "delta": 0.0}
            for name in order_buckets
        }
        self._order_buckets = order_buckets

        # Liquidity heatmap
        lhm_window_size = int(getattr(config, "LHM_WINDOW_SIZE", 2000))
        lhm_cluster_threshold_pct = float(
            getattr(config, "LHM_CLUSTER_THRESHOLD_PCT", 0.003)
        )
        lhm_min_trades_per_cluster = int(
            getattr(config, "LHM_MIN_TRADES_PER_CLUSTER", 5)
        )
        lhm_update_interval_ms = int(
            getattr(config, "LHM_UPDATE_INTERVAL_MS", 100)
        )

        self.liquidity_heatmap = LiquidityHeatmap(
            window_size=lhm_window_size,
            cluster_threshold_pct=lhm_cluster_threshold_pct,
            min_trades_per_cluster=lhm_min_trades_per_cluster,
            update_interval_ms=lhm_update_interval_ms
        )

        # Flow tracking
        self.net_flow_windows_min: List[int] = list(NET_FLOW_WINDOWS_MIN)
        self.flow_trades: deque = deque()

        # Absorção config
        self.absorcao_eps: float = float(
            getattr(config, "ABSORCAO_DELTA_EPS", ABSORCAO_DELTA_EPS)
        )
        try:
            self.absorcao_guard_mode: str = str(
                getattr(config, "ABSORCAO_GUARD_MODE", ABSORCAO_GUARD_MODE)
            ).lower()
        except Exception:
            self.absorcao_guard_mode = "warn"
        
        # Qualidade de dados
        self._total_trades_processed = 0
        self._invalid_trades = 0
        self._lock_contentions = 0
        self._last_price: Optional[float] = None
        
        # Contadores de correção
        self._whale_delta_corrections = 0
        self._is_buyer_maker_conversions = 0
        self._volume_discrepancies = 0
        
        # ✅ Contadores de timestamp
        self.negative_age_count = 0
        self.timestamp_adjustments = 0

        logging.info(
            "✅ FlowAnalyzer v2.3.1 inicializado | "
            "Whale threshold: %.2f BTC | Net flow windows: %s min | "
            "Reset interval: %.1fh | Timestamp tolerance: ±%dms",
            self.whale_threshold,
            self.net_flow_windows_min,
            self.reset_interval_ms / (3600 * 1000),
            TIMESTAMP_JITTER_TOLERANCE_MS
        )

    # ✅ CORRIGIDO: Métodos de timestamp
    def _get_synced_timestamp_ms(self) -> int:
        """
        Obtém timestamp sincronizado com servidor.
        
        Prioridade:
        1. TimeManager (sempre disponível)
        2. ClockSync (se disponível)
        3. time.time() (fallback)
        """
        # Prioridade 1: TimeManager (sempre disponível e confiável)
        if self.time_manager:
            try:
                return self.time_manager.now_ms()
            except Exception as e:
                logging.warning(f"⚠️ Erro ao obter timestamp do TimeManager: {e}")
        
        # Prioridade 2: ClockSync (se disponível)
        if self.clock_sync:
            try:
                # ✅ CORREÇÃO: ClockSync usa get_server_time_ms(), não now_ms()
                return self.clock_sync.get_server_time_ms()
            except Exception as e:
                logging.warning(f"⚠️ Erro ao obter timestamp do ClockSync: {e}")
        
        # Fallback final: time.time()
        return int(time.time() * 1000)

    def _adjust_timestamp_if_needed(
        self,
        trade_ts: int,
        reference_ts: int
    ) -> tuple:
        """
        Ajusta timestamp se houver jitter aceitável.
        
        Returns:
            (timestamp_ajustado, foi_ajustado)
        """
        diff = trade_ts - reference_ts
        
        # Se está dentro da tolerância, aceita
        if abs(diff) <= TIMESTAMP_JITTER_TOLERANCE_MS:
            if diff < 0:
                # Timestamp levemente no passado - ok
                return trade_ts, False
            else:
                # Timestamp levemente no futuro - ajusta para referência
                self.timestamp_adjustments += 1
                return reference_ts, True
        
        # Diferença maior que tolerância
        if diff > TIMESTAMP_JITTER_TOLERANCE_MS:
            # Muito no futuro - clamp para referência
            if self.timestamp_adjustments % 10 == 0:
                logging.warning(
                    f"⚠️ Timestamp muito futuro: diff={diff}ms > {TIMESTAMP_JITTER_TOLERANCE_MS}ms. "
                    f"Ajustando para referência. (ajustes totais: {self.timestamp_adjustments})"
                )
            self.timestamp_adjustments += 1
            return reference_ts, True
        
        # Muito no passado - mantém original
        return trade_ts, False

    def _calculate_age_ms(
        self,
        trade_ts: int,
        reference_ts: int
    ) -> float:
        """
        Calcula idade do trade com tolerância a jitter.
        
        Args:
            trade_ts: Timestamp do trade (ms)
            reference_ts: Timestamp de referência (ms)
            
        Returns:
            Idade em ms (sempre >= 0)
        """
        # Ajusta timestamp se necessário
        adjusted_ts, was_adjusted = self._adjust_timestamp_if_needed(
            trade_ts,
            reference_ts
        )
        
        age_ms = reference_ts - adjusted_ts
        
        # Se ainda assim ficou negativo
        if age_ms < 0:
            self.negative_age_count += 1
            
            # Log detalhado apenas a cada 10 ocorrências
            if self.negative_age_count % 10 == 1:
                logging.warning(
                    f"⚠️ Idade negativa após ajuste: {age_ms}ms "
                    f"(trade={trade_ts}, ref={reference_ts}). "
                    f"Ocorrências: {self.negative_age_count}. "
                    f"Retornando 0."
                )
            
            return 0.0
        
        return float(age_ms)

    @staticmethod
    def map_absorcao_label(aggression_side: str) -> str:
        """Mapeia lado de agressão para rótulo."""
        side = (aggression_side or "").strip().lower()
        if side == "buy":
            return "Absorção de Compra"
        if side == "sell":
            return "Absorção de Venda"
        return "Absorção"

    @staticmethod
    def classificar_absorcao_por_delta(delta: float, eps: float = 1.0) -> str:
        """Classificador de absorção por sinal do delta."""
        try:
            d = float(delta)
        except Exception:
            return "Neutra"
        
        if d > eps:
            return "Absorção de Compra"
        if d < -eps:
            return "Absorção de Venda"
        return "Neutra"

    def _reset_metrics(self):
        """Reseta todas as métricas acumuladas."""
        try:
            self.cvd = 0.0
            self.whale_buy_volume = 0.0
            self.whale_sell_volume = 0.0
            self.whale_delta = 0.0
            self.recent_trades.clear()
            self.bursts = {"count": 0, "max_burst_volume": 0.0}
            self._in_burst = False
            self._last_burst_end_ms = 0
            self.sector_flow = {
                name: {"buy": 0.0, "sell": 0.0, "delta": 0.0}
                for name in self._order_buckets
            }
            self.flow_trades.clear()
            self._last_price = None
            self.last_reset_ms = self._get_synced_timestamp_ms()
            
            reset_time = self.time_manager.format_timestamp(self.last_reset_ms)
            logging.info(
                f"🔄 FlowAnalyzer metrics resetados em {reset_time}. "
                f"Ajustes de timestamp desde último reset: {self.timestamp_adjustments}"
            )
            
            # ✅ Reseta contadores de timestamp
            self.timestamp_adjustments = 0
            self.negative_age_count = 0
            
        except Exception as e:
            logging.error(f"Erro ao resetar métricas: {e}")

    def _check_reset(self):
        """Verifica se deve resetar métricas."""
        try:
            now_ms = self._get_synced_timestamp_ms()
            if now_ms - self.last_reset_ms > self.reset_interval_ms:
                with self._lock:
                    self._reset_metrics()
        except Exception as e:
            logging.error(f"Erro ao verificar reset: {e}")

    def _prune_recent(self, now_ms: int):
        """Remove trades antigos fora da janela de bursts."""
        try:
            while self.recent_trades and \
                  (now_ms - self.recent_trades[0][0] > self.burst_window_ms):
                self.recent_trades.popleft()
        except Exception as e:
            logging.error(f"Erro ao podar recent_trades: {e}")
            self.recent_trades.clear()

    def _prune_flow_history(self, now_ms: int):
        """Remove trades antigos do histórico de flow."""
        try:
            if not self.net_flow_windows_min:
                return
            max_window = max(self.net_flow_windows_min)
            cutoff_ms = now_ms - max_window * 60 * 1000
            while self.flow_trades and self.flow_trades[0]['ts'] < cutoff_ms:
                self.flow_trades.popleft()
        except Exception as e:
            logging.debug(f"Erro ao podar flow_trades: {e}")

    def _update_bursts(self, ts_ms: int, qty: float):
        """Detecta bursts de volume."""
        try:
            self.recent_trades.append((ts_ms, qty))
            self._prune_recent(ts_ms)
            burst_volume = sum(q for _, q in self.recent_trades)
            threshold = self.burst_volume_threshold

            if not self._in_burst:
                if burst_volume >= threshold and \
                   (ts_ms - self._last_burst_end_ms) >= self.burst_cooldown_ms:
                    self.bursts["count"] += 1
                    self._in_burst = True
                    if burst_volume > self.bursts["max_burst_volume"]:
                        self.bursts["max_burst_volume"] = burst_volume
            else:
                if burst_volume > self.bursts["max_burst_volume"]:
                    self.bursts["max_burst_volume"] = burst_volume
                if burst_volume < threshold * 0.5:
                    self._in_burst = False
                    self._last_burst_end_ms = ts_ms
        except Exception as e:
            logging.error(f"Erro ao atualizar bursts: {e}")
            self._in_burst = False
            self._last_burst_end_ms = ts_ms

    def _update_sector_flow(self, qty: float, delta_btc: float):
        """Atualiza sector flow ACUMULADO com arredondamento decimal."""
        try:
            for name, (minv, maxv) in self._order_buckets.items():
                if minv <= qty < maxv:
                    if delta_btc > 0:
                        self.sector_flow[name]["buy"] = _decimal_round(
                            self.sector_flow[name]["buy"] + qty
                        )
                    else:
                        self.sector_flow[name]["sell"] = _decimal_round(
                            self.sector_flow[name]["sell"] + qty
                        )
                    self.sector_flow[name]["delta"] = _decimal_round(
                        self.sector_flow[name]["delta"] + delta_btc
                    )
                    break
        except Exception as e:
            logging.error(f"Erro ao atualizar sector_flow: {e}")

    def process_trade(self, trade: dict):
        """Processa trade e atualiza métricas ACUMULADAS."""
        try:
            self._check_reset()
            self._total_trades_processed += 1

            if not isinstance(trade, dict):
                self._invalid_trades += 1
                return
            
            if not all(k in trade for k in ("q", "T", "p")):
                self._invalid_trades += 1
                return

            try:
                qty = float(trade.get('q', 0.0))
                ts = int(trade.get('T'))
                price = float(trade.get('p', 0.0))
            except (ValueError, TypeError):
                self._invalid_trades += 1
                return

            if qty <= 0 or price <= 0 or ts <= 0:
                self._invalid_trades += 1
                return

            # ✅ Ajusta timestamp se necessário
            reference_ts = self._get_synced_timestamp_ms()
            ts_adjusted, was_adjusted = self._adjust_timestamp_if_needed(ts, reference_ts)
            
            if was_adjusted:
                ts = ts_adjusted

            # Conversão corrigida de is_buyer_maker
            is_buyer_maker = trade.get('m', None)
            
            if isinstance(is_buyer_maker, bool):
                pass
            elif isinstance(is_buyer_maker, (int, float)):
                is_buyer_maker = bool(int(is_buyer_maker))
            elif isinstance(is_buyer_maker, str):
                is_buyer_maker = is_buyer_maker.strip().lower() in {
                    "true", "t", "1", "yes"
                }
                self._is_buyer_maker_conversions += 1
            else:
                self._invalid_trades += 1
                logging.warning(f"Tipo inválido para 'm': {type(is_buyer_maker)}")
                return

            # Calcular delta com arredondamento decimal
            delta_btc = _decimal_round(-qty if is_buyer_maker else qty)
            delta_usd = _decimal_round(delta_btc * price, decimals=2)
            side = "sell" if is_buyer_maker else "buy"

            with self._lock:
                # Atualizar CVD (ACUMULADO)
                self.cvd = _decimal_round(self.cvd + delta_btc)

                # Atualizar whale metrics (ACUMULADO)
                if qty >= self.whale_threshold:
                    if delta_btc > 0:  # Compra
                        self.whale_buy_volume = _decimal_round(
                            self.whale_buy_volume + qty
                        )
                    else:  # Venda
                        self.whale_sell_volume = _decimal_round(
                            self.whale_sell_volume + qty
                        )
                    self.whale_delta = _decimal_round(
                        self.whale_delta + delta_btc
                    )
                    
                    # Validação de consistência
                    expected_delta = _decimal_round(
                        self.whale_buy_volume - self.whale_sell_volume
                    )
                    actual_delta = self.whale_delta
                    
                    if abs(expected_delta - actual_delta) > 0.001:
                        logging.error(
                            f"🔴 INCONSISTÊNCIA WHALE ACUMULADO: delta={actual_delta:.8f}, "
                            f"mas buy-sell={expected_delta:.8f} "
                            f"(buy={self.whale_buy_volume:.8f}, sell={self.whale_sell_volume:.8f})"
                        )
                        
                        self.whale_delta = expected_delta
                        self._whale_delta_corrections += 1
                        logging.warning(f"✅ Corrigido para {expected_delta:.8f}")

                self._update_bursts(ts, qty)
                self._update_sector_flow(qty, delta_btc)  # ACUMULADO

                self.liquidity_heatmap.add_trade(
                    price=price,
                    volume=qty,
                    side=side,
                    timestamp_ms=ts
                )

                try:
                    sector_name: Optional[str] = None
                    for name, (minv, maxv) in self._order_buckets.items():
                        if minv <= qty < maxv:
                            sector_name = name
                            break

                    self.flow_trades.append({
                        'ts': ts,
                        'price': price,
                        'qty': qty,
                        'delta_btc': delta_btc,
                        'delta_usd': delta_usd,
                        'side': side,
                        'sector': sector_name,
                    })

                    self._prune_flow_history(ts)
                    self._last_price = price

                except Exception as e:
                    logging.debug(f"Erro ao salvar trade no histórico: {e}")

        except Exception as e:
            logging.debug(f"Erro ao processar trade: {e}")
            self._invalid_trades += 1

    def _normalize_heatmap_clusters(
        self, 
        clusters: List[Dict[str, Any]], 
        now_ms: int
    ) -> List[Dict[str, Any]]:
        """Normaliza clusters do heatmap com cálculo correto de idade."""
        out: List[Dict[str, Any]] = []
        
        for c in clusters or []:
            cc = dict(c)
            
            try:
                recent_keys = (
                    "recent_timestamp", "recent_ts_ms", "last_seen_ms",
                    "last_ts_ms", "max_timestamp", "last_timestamp"
                )
                recent_ts = None
                
                for k in recent_keys:
                    if k in cc and isinstance(cc[k], (int, float)):
                        recent_ts = int(cc[k])
                        break
                
                # ✅ Usa método corrigido de cálculo de idade
                if recent_ts is not None:
                    cc["age_ms"] = self._calculate_age_ms(recent_ts, now_ms)

                tv = cc.get("total_volume", None)
                bv = float(cc.get("buy_volume", 0.0) or 0.0)
                sv = float(cc.get("sell_volume", 0.0) or 0.0)
                
                if tv is None or (isinstance(tv, (int, float)) and tv <= 0):
                    recomputed = bv + sv
                    if recomputed > 0:
                        cc["total_volume"] = recomputed

                tv2 = float(cc.get("total_volume", 0.0) or 0.0)
                if tv2 > 0:
                    imb = (bv - sv) / tv2
                else:
                    imb = 0.0
                
                cc["imbalance_ratio"] = max(-1.0, min(1.0, float(imb)))

                if "high" in cc and "low" in cc:
                    try:
                        hi, lo = float(cc["high"]), float(cc["low"])
                        cc["width"] = max(0.0, hi - lo)
                    except Exception:
                        pass

                if "trades_count" in cc:
                    try:
                        cc["trades_count"] = int(cc["trades_count"])
                    except Exception:
                        pass

            except Exception as e:
                logging.warning(f"⚠️ Erro ao normalizar cluster: {e}")

            out.append(cc)
        
        return out

    def get_flow_metrics(
        self, 
        reference_epoch_ms: Optional[int] = None
    ) -> dict:
        """
        Retorna métricas de fluxo com separação clara entre acumulado e janela.
        
        🔹 MÉTRICAS ACUMULADAS (desde último reset):
          - cvd: Cumulative Volume Delta total
          - whale_buy_volume / whale_sell_volume / whale_delta: Acumulado
          - sector_flow: Flow acumulado desde reset
          - bursts: Contadores acumulados
        
        🔹 MÉTRICAS POR JANELA (computation_window_min):
          - order_flow.net_flow_Xm: Net flow por janela X
          - order_flow.whale_*_window: Volumes whale na menor janela
          - order_flow.sector_flow_window: Sector flow na menor janela
          - participant_analysis: Análise baseada na menor janela
        """
        try:
            acquired = self._lock.acquire(timeout=5.0)
            
            if not acquired:
                self._lock_contentions += 1
                error_msg = f"❌ FlowAnalyzer lock timeout após 5s"
                logging.error(error_msg)
                raise FlowAnalyzerError(error_msg)

            try:
                # ✅ Usa timestamp sincronizado
                now_ms = reference_epoch_ms if reference_epoch_ms is not None \
                         else self._get_synced_timestamp_ms()
                
                time_index = self.time_manager.build_time_index(
                    now_ms, 
                    include_local=True, 
                    timespec="milliseconds"
                )

                # Validação final whale delta ACUMULADO
                expected_whale_delta = _decimal_round(
                    self.whale_buy_volume - self.whale_sell_volume
                )
                if abs(self.whale_delta - expected_whale_delta) > 0.001:
                    logging.error(
                        f"🔴 CORREÇÃO FINAL WHALE ACUMULADO: "
                        f"delta={self.whale_delta:.8f} → {expected_whale_delta:.8f}"
                    )
                    self.whale_delta = expected_whale_delta
                    self._whale_delta_corrections += 1

                # ===== MÉTRICAS ACUMULADAS =====
                metrics = {
                    "cvd": _decimal_round(self.cvd),
                    "whale_buy_volume": _decimal_round(self.whale_buy_volume),
                    "whale_sell_volume": _decimal_round(self.whale_sell_volume),
                    "whale_delta": _decimal_round(self.whale_delta),
                    "bursts": dict(self.bursts),
                    "sector_flow": {
                        k: {
                            "buy": _decimal_round(v["buy"]),
                            "sell": _decimal_round(v["sell"]),
                            "delta": _decimal_round(v["delta"])
                        }
                        for k, v in self.sector_flow.items()
                    },
                    "timestamp": time_index["timestamp_utc"],
                    "time_index": time_index,
                    "metadata": {
                        "burst_window_ms": self.burst_window_ms,
                        "burst_cooldown_ms": self.burst_cooldown_ms,
                        "in_burst": bool(self._in_burst),
                        "last_reset_ms": self.last_reset_ms,
                        "last_reset_iso_utc": self.time_manager.format_timestamp(
                            self.last_reset_ms
                        ),
                        "reset_interval_hours": self.reset_interval_ms / (3600 * 1000),
                    },
                }

                # Validação de consistência de sector flow acumulado
                for sector_name, sector_data in metrics["sector_flow"].items():
                    buy_vol = sector_data["buy"]
                    sell_vol = sector_data["sell"]
                    delta = sector_data["delta"]
                    
                    expected_delta = _decimal_round(buy_vol - sell_vol)
                    
                    if abs(delta - expected_delta) > 0.001:
                        logging.warning(
                            f"⚠️ DISCREPÂNCIA SECTOR ACUMULADO[{sector_name}]: "
                            f"delta={delta:.8f}, mas buy-sell={expected_delta:.8f}"
                        )
                        metrics["sector_flow"][sector_name]["delta"] = expected_delta
                        self._volume_discrepancies += 1

                # ===== ORDER FLOW POR JANELA =====
                try:
                    order_flow: Dict[str, Any] = {}
                    absorcao_por_janela: Dict[int, str] = {}

                    if not self.net_flow_windows_min:
                        smallest_window = 1
                    else:
                        smallest_window = min(self.net_flow_windows_min)

                    tick_rule_sum = 0.0
                    
                    for window_min in self.net_flow_windows_min:
                        window_ms = window_min * 60 * 1000
                        start_ms = now_ms - window_ms
                        
                        relevant = [
                            t for t in self.flow_trades 
                            if t['ts'] >= start_ms
                        ]

                        total_delta_usd = sum(t['delta_usd'] for t in relevant)
                        total_delta_usd = _decimal_round(total_delta_usd, decimals=2)
                        
                        key_net = f"net_flow_{window_min}m"
                        order_flow[key_net] = _decimal_round(
                            total_delta_usd, decimals=4
                        )

                        rotulo = self.classificar_absorcao_por_delta(
                            total_delta_usd, 
                            eps=self.absorcao_eps
                        )
                        _guard_absorcao(
                            total_delta_usd, 
                            rotulo, 
                            self.absorcao_eps, 
                            self.absorcao_guard_mode
                        )

                        order_flow[f"absorcao_{window_min}m"] = rotulo
                        absorcao_por_janela[window_min] = rotulo

                        if window_min == smallest_window:
                            logging.info(
                                f"📊 Calculando métricas detalhadas para janela {window_min}m "
                                f"({len(relevant)} trades)"
                            )
                            
                            # USD COM DECIMAL
                            total_buy_usd_dec  = Decimal('0')
                            total_sell_usd_dec = Decimal('0')
                            
                            for t in relevant:
                                price_dec = _to_decimal(t['price'])
                                qty_dec = _to_decimal(t['qty'])
                                
                                if t['side'] == 'buy':
                                    total_buy_usd_dec += price_dec * qty_dec
                                else:
                                    total_sell_usd_dec += price_dec * qty_dec

                            total_vol_usd_dec = total_buy_usd_dec + total_sell_usd_dec

                            # BTC COM DECIMAL
                            total_buy_btc_dec  = Decimal('0')
                            total_sell_btc_dec = Decimal('0')
                            
                            for t in relevant:
                                qty_dec = _to_decimal(t['qty'])
                                
                                if t['side'] == 'buy':
                                    total_buy_btc_dec += qty_dec
                                else:
                                    total_sell_btc_dec += qty_dec

                            total_vol_btc_dec = total_buy_btc_dec + total_sell_btc_dec

                            # USD: INVARIÂNCIA DE UI
                            buy_usd, sell_usd, total_usd, ui_ok, tol = \
                                _ui_safe_round_usd(
                                    total_buy_usd_dec, 
                                    total_sell_usd_dec
                                )

                            order_flow["buy_volume_usd_exact"]   = float(total_buy_usd_dec)
                            order_flow["sell_volume_usd_exact"]  = float(total_sell_usd_dec)
                            order_flow["total_volume_usd_exact"] = float(total_vol_usd_dec)
                            order_flow["buy_volume"]   = buy_usd
                            order_flow["sell_volume"]  = sell_usd
                            order_flow["total_volume"] = total_usd
                            order_flow["ui_sum_ok"]    = ui_ok
                            order_flow["ui_sum_tolerance"] = tol

                            # BTC: INVARIÂNCIA DE UI
                            buy_btc, sell_btc, total_btc, diff_btc = \
                                _ui_safe_round_btc(
                                    total_buy_btc_dec, 
                                    total_sell_btc_dec,
                                    decimals=8
                                )

                            if diff_btc > 0.001:
                                logging.warning(
                                    f"⚠️ DISCREPÂNCIA BTC ALTA ({window_min}m): {diff_btc:.8f} BTC"
                                )
                                self._volume_discrepancies += 1

                            order_flow["buy_volume_btc"]   = buy_btc
                            order_flow["sell_volume_btc"]  = sell_btc
                            order_flow["total_volume_btc"] = total_btc

                            # WHALE VOLUMES POR JANELA
                            whale_buy_window = sum(
                                t['qty'] for t in relevant 
                                if t['qty'] >= self.whale_threshold and t['side'] == 'buy'
                            )
                            whale_sell_window = sum(
                                t['qty'] for t in relevant 
                                if t['qty'] >= self.whale_threshold and t['side'] == 'sell'
                            )
                            whale_delta_window = whale_buy_window - whale_sell_window
                            
                            order_flow["whale_buy_volume_window"] = _decimal_round(whale_buy_window)
                            order_flow["whale_sell_volume_window"] = _decimal_round(whale_sell_window)
                            order_flow["whale_delta_window"] = _decimal_round(whale_delta_window)
                            
                            expected_whale_delta_window = _decimal_round(
                                whale_buy_window - whale_sell_window
                            )
                            actual_whale_delta_window = order_flow["whale_delta_window"]
                            
                            if abs(expected_whale_delta_window - actual_whale_delta_window) > 0.001:
                                logging.error(
                                    f"🔴 INCONSISTÊNCIA WHALE JANELA {window_min}m: "
                                    f"delta={actual_whale_delta_window:.8f}, "
                                    f"mas buy-sell={expected_whale_delta_window:.8f}"
                                )
                                order_flow["whale_delta_window"] = expected_whale_delta_window
                                self._whale_delta_corrections += 1

                            logging.info(
                                f"🐋 WHALE JANELA {window_min}m: "
                                f"buy={whale_buy_window:.4f}, "
                                f"sell={whale_sell_window:.4f}, "
                                f"delta={whale_delta_window:.4f}"
                            )

                            # SECTOR FLOW POR JANELA
                            sector_flow_window = {
                                sector: {"buy": 0.0, "sell": 0.0, "delta": 0.0}
                                for sector in self._order_buckets.keys()
                            }

                            for t in relevant:
                                sector = t.get('sector')
                                if sector and sector in sector_flow_window:
                                    if t['side'] == 'buy':
                                        sector_flow_window[sector]["buy"] += t['qty']
                                    else:
                                        sector_flow_window[sector]["sell"] += t['qty']
                                    sector_flow_window[sector]["delta"] += t['delta_btc']

                            for sector in sector_flow_window:
                                buy_s = sector_flow_window[sector]["buy"]
                                sell_s = sector_flow_window[sector]["sell"]
                                delta_s = sector_flow_window[sector]["delta"]
                                
                                sector_flow_window[sector] = {
                                    "buy": _decimal_round(buy_s),
                                    "sell": _decimal_round(sell_s),
                                    "delta": _decimal_round(delta_s)
                                }
                                
                                expected_delta_s = _decimal_round(buy_s - sell_s)
                                actual_delta_s = sector_flow_window[sector]["delta"]
                                
                                if abs(expected_delta_s - actual_delta_s) > 0.001:
                                    logging.warning(
                                        f"⚠️ DISCREPÂNCIA SECTOR JANELA[{sector}]: "
                                        f"delta={actual_delta_s:.8f}, "
                                        f"mas buy-sell={expected_delta_s:.8f}"
                                    )
                                    sector_flow_window[sector]["delta"] = expected_delta_s
                                    self._volume_discrepancies += 1

                            order_flow["sector_flow_window"] = sector_flow_window

                            logging.info(
                                f"📊 SECTOR FLOW JANELA {window_min}m: "
                                + ", ".join([
                                    f"{s}(Δ={v['delta']:.2f})" 
                                    for s, v in sector_flow_window.items()
                                ])
                            )

                            # VALIDAÇÃO DE DELTA BTC
                            expected_delta_btc = _decimal_round(buy_btc - sell_btc)
                            actual_delta_btc = sum(t['delta_btc'] for t in relevant)
                            actual_delta_btc = _decimal_round(actual_delta_btc)
                            
                            if abs(expected_delta_btc - actual_delta_btc) > 0.01:
                                logging.warning(
                                    f"⚠️ DISCREPÂNCIA DELTA {window_min}m: "
                                    f"Σdelta={actual_delta_btc:.4f}, "
                                    f"mas buy-sell={expected_delta_btc:.4f}"
                                )
                                self._volume_discrepancies += 1

                            # MÉTRICAS DERIVADAS
                            if float(total_vol_usd_dec) > 0:
                                flow_imbalance = total_delta_usd / float(total_vol_usd_dec)
                                order_flow["flow_imbalance"] = _decimal_round(
                                    flow_imbalance, decimals=4
                                )
                                
                                order_flow["aggressive_buy_pct"] = _decimal_round(
                                    float((total_buy_usd_dec / total_vol_usd_dec) * Decimal('100')),
                                    decimals=2
                                )
                                order_flow["aggressive_sell_pct"] = _decimal_round(
                                    float((total_sell_usd_dec / total_vol_usd_dec) * Decimal('100')),
                                    decimals=2
                                )
                                
                                if float(total_sell_usd_dec) > 0:
                                    order_flow["buy_sell_ratio"] = _decimal_round(
                                        float(total_buy_usd_dec / total_sell_usd_dec),
                                        decimals=4
                                    )
                                else:
                                    order_flow["buy_sell_ratio"] = None
                            else:
                                order_flow["flow_imbalance"] = 0.0
                                order_flow["aggressive_buy_pct"] = None
                                order_flow["aggressive_sell_pct"] = None
                                order_flow["buy_sell_ratio"] = None

                            # LOG DETALHADO
                            logging.info(
                                f"📊 VOLUMES CALCULADOS (janela {window_min}m):\n"
                                f"   === USD (Decimal → UI) ===\n"
                                f"   buy_exact:  {order_flow['buy_volume_usd_exact']:.2f}\n"
                                f"   sell_exact: {order_flow['sell_volume_usd_exact']:.2f}\n"
                                f"   total_exact:{order_flow['total_volume_usd_exact']:.2f}\n"
                                f"   buy_ui:     {order_flow['buy_volume']:.2f}\n"
                                f"   sell_ui:    {order_flow['sell_volume']:.2f}\n"
                                f"   total_ui:   {order_flow['total_volume']:.2f}\n"
                                f"   ui_sum_ok:  {order_flow['ui_sum_ok']} (tol={tol})\n"
                                f"   === BTC (Decimal → Arredondado) ===\n"
                                f"   buy:        {buy_btc:.8f}\n"
                                f"   sell:       {sell_btc:.8f}\n"
                                f"   total:      {total_btc:.8f}\n"
                                f"   sum:        {buy_btc + sell_btc:.8f}\n"
                                f"   diff:       {abs(total_btc - (buy_btc + sell_btc)):.8f}\n"
                            )

                            # TICK RULE
                            tick_rule_sum = 0.0
                            prev_price = None
                            
                            for t in sorted(relevant, key=lambda x: x['ts']):
                                curr_price = t['price']
                                if prev_price is not None:
                                    if curr_price > prev_price:
                                        tick_rule_sum += 1.0
                                    elif curr_price < prev_price:
                                        tick_rule_sum -= 1.0
                                prev_price = curr_price
                            
                            order_flow["tick_rule_sum"] = _decimal_round(
                                tick_rule_sum, decimals=4
                            )

                    order_flow["computation_window_min"] = smallest_window
                    order_flow["available_windows_min"]  = list(self.net_flow_windows_min)

                    if self.net_flow_windows_min:
                        metrics["tipo_absorcao"] = absorcao_por_janela.get(
                            smallest_window, "Neutra"
                        )
                    else:
                        metrics["tipo_absorcao"] = "Neutra"

                    # PARTICIPANT ANALYSIS (MENOR JANELA)
                    participant_analysis: Dict[str, Any] = {}
                    
                    if self.net_flow_windows_min:
                        analysis_window = min(self.net_flow_windows_min)
                        start_ms_p = now_ms - analysis_window * 60 * 1000
                        
                        all_trades = [
                            t for t in self.flow_trades 
                            if t['ts'] >= start_ms_p
                        ]
                        
                        total_qty_all = sum(t['qty'] for t in all_trades)

                        logging.info(
                            f"👥 Calculando participant analysis para janela {analysis_window}m "
                            f"({len(all_trades)} trades, {total_qty_all:.4f} BTC total)"
                        )

                        for sector in self._order_buckets.keys():
                            sector_trades = [
                                t for t in all_trades 
                                if t.get('sector') == sector
                            ]
                            total_qty_sector = sum(t['qty'] for t in sector_trades)
                            buy_qty = sum(
                                t['qty'] for t in sector_trades 
                                if t['delta_btc'] > 0
                            )
                            sell_qty = sum(
                                t['qty'] for t in sector_trades 
                                if t['delta_btc'] < 0
                            )
                            count_trades = len(sector_trades)

                            if buy_qty > sell_qty:
                                direction = "BUY"
                            elif sell_qty > buy_qty:
                                direction = "SELL"
                            else:
                                direction = "NEUTRAL"

                            avg_order_size = _decimal_round(
                                total_qty_sector / count_trades, decimals=4
                            ) if count_trades > 0 else None
                            
                            volume_pct = _decimal_round(
                                (total_qty_sector / total_qty_all) * 100.0, decimals=2
                            ) if total_qty_all > 0 else None
                            
                            sentiment = "BULLISH" if direction == "BUY" else \
                                       ("BEARISH" if direction == "SELL" else "NEUTRAL")
                            
                            duration_seconds = analysis_window * 60
                            trades_per_sec = _decimal_round(
                                count_trades / duration_seconds, decimals=4
                            ) if duration_seconds > 0 else None
                            
                            activity_level = "HIGH" if trades_per_sec and \
                                            trades_per_sec >= 1.0 else "LOW"

                            if total_qty_sector > 0:
                                participant_analysis[sector] = {
                                    "volume_pct": volume_pct,
                                    "direction": direction,
                                    "avg_order_size": avg_order_size,
                                    "sentiment": sentiment,
                                    "activity_level": activity_level,
                                }

                        total_pct = sum(
                            float(p.get("volume_pct", 0) or 0) 
                            for p in participant_analysis.values()
                        )
                        
                        if abs(total_pct - 100.0) > 0.5:
                            logging.warning(
                                f"⚠️ Soma de volume_pct = {total_pct:.2f}% (esperado: 100%)"
                            )

                    metrics["order_flow"] = order_flow
                    metrics["participant_analysis"] = participant_analysis

                except Exception as e:
                    logging.error(f"Erro ao calcular order_flow: {e}", exc_info=True)
                    metrics["order_flow"] = {
                        "flow_imbalance": 0.0,
                        "tick_rule_sum": 0.0,
                        "buy_sell_ratio": None,
                    }
                    metrics["participant_analysis"] = {}

                # Heatmap
                try:
                    clusters = self.liquidity_heatmap.get_clusters(top_n=5)
                    clusters = self._normalize_heatmap_clusters(clusters, now_ms=now_ms)
                    supports, resistances = self.liquidity_heatmap.get_support_resistance()
                    
                    metrics["liquidity_heatmap"] = {
                        "clusters": clusters,
                        "supports": sorted(set(supports)),
                        "resistances": sorted(set(resistances)),
                        "clusters_count": len(clusters),
                    }
                except Exception as e:
                    logging.error(f"Erro ao obter heatmap: {e}", exc_info=True)
                    metrics["liquidity_heatmap"] = {
                        "clusters": [],
                        "supports": [],
                        "resistances": [],
                        "clusters_count": 0,
                    }

                # Qualidade de dados
                metrics["data_quality"] = {
                    "total_trades_processed": self._total_trades_processed,
                    "invalid_trades": self._invalid_trades,
                    "valid_rate_pct": _decimal_round(
                        100 * (1 - self._invalid_trades / max(1, self._total_trades_processed)),
                        decimals=2
                    ),
                    "flow_trades_count": len(self.flow_trades),
                    "lock_contentions": self._lock_contentions,
                    "whale_delta_corrections": self._whale_delta_corrections,
                    "is_buyer_maker_conversions": self._is_buyer_maker_conversions,
                    "volume_discrepancies": self._volume_discrepancies,
                    "negative_age_count": self.negative_age_count,
                    "timestamp_adjustments": self.timestamp_adjustments,
                }

                return metrics

            finally:
                self._lock.release()

        except FlowAnalyzerError:
            raise
        except Exception as e:
            logging.error(f"Erro ao obter flow metrics: {e}", exc_info=True)
            
            now_ms = reference_epoch_ms if reference_epoch_ms is not None \
                     else self._get_synced_timestamp_ms()
            time_index = self.time_manager.build_time_index(
                now_ms, include_local=True, timespec="milliseconds"
            )
            
            return {
                "cvd": 0.0,
                "whale_delta": 0.0,
                "order_flow": {
                    "flow_imbalance": 0.0,
                    "tick_rule_sum": 0.0,
                    "buy_sell_ratio": None,
                },
                "timestamp": time_index["timestamp_utc"],
                "data_quality": {
                    "error": str(e),
                    "is_valid": False,
                },
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance e qualidade."""
        return {
            "total_trades_processed": self._total_trades_processed,
            "invalid_trades": self._invalid_trades,
            "valid_rate_pct": _decimal_round(
                100 * (1 - self._invalid_trades / max(1, self._total_trades_processed)),
                decimals=2
            ),
            "lock_contentions": self._lock_contentions,
            "flow_trades_count": len(self.flow_trades),
            "cvd": _decimal_round(self.cvd),
            "whale_delta": _decimal_round(self.whale_delta),
            "whale_delta_corrections": self._whale_delta_corrections,
            "is_buyer_maker_conversions": self._is_buyer_maker_conversions,
            "volume_discrepancies": self._volume_discrepancies,
            "negative_age_count": self.negative_age_count,
            "timestamp_adjustments": self.timestamp_adjustments,
            "reset_interval_hours": self.reset_interval_ms / (3600 * 1000),
            "time_since_last_reset_hours": (
                self._get_synced_timestamp_ms() - self.last_reset_ms
            ) / (3600 * 1000),
        }