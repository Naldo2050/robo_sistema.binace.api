# market_analyzer.py v2.3.0 — SUPER-CORRECTED
"""
Market Analyzer com integração COMPLETA, validação robusta e precisão máxima.

📌 CORREÇÕES v2.3.0:
  ✅ Tolerância correta (1e-8 BTC em vez de 0.001 BTC)
  ✅ Validação de timestamps em clusters do heatmap
  ✅ Correção automática de age_ms negativo
  ✅ Validação de first_seen <= last_seen
  ✅ Logs informativos sem poluir console
  ✅ Integração com data_validator para validação final
  ✅ Precisão máxima em TODOS os cálculos
  ✅ Contadores de correções detalhados

📌 Componentes integrados:
  • RobustConnectionManager  — WebSocket robusto
  • EnhancedMarketAnalyzer   — análise e validação completa
  • EnhancedMarketBot        — orquestração de janelas
  • Validação de timestamps  — previne erros críticos
  • Data validator           — validação final de eventos

Autor: Sistema de Trading Institucional
Data: 2025-01-11
Versão: 2.3.0
"""

from __future__ import annotations

import json
import logging
import random
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

# Config e componentes do projeto
import config
from orderbook_analyzer import OrderBookAnalyzer
from flow_analyzer import FlowAnalyzer
from ml_features import generate_ml_features
from time_manager import TimeManager
from event_saver import EventSaver
from health_monitor import HealthMonitor
from log_formatter import format_flow_log, track_cvd_consistency

# 🆕 Importa validador com tratamento de erro
try:
    from data_validator import DataValidator
    DATA_VALIDATOR_AVAILABLE = True
except ImportError:
    DATA_VALIDATOR_AVAILABLE = False
    logging.warning("⚠️ DataValidator não disponível. Continuando sem validação final.")

# 🆕 Importa IntegrationValidator com tratamento de erro
try:
    from integration_validator import IntegrationValidator
    INTEGRATION_VALIDATOR_AVAILABLE = True
except ImportError:
    INTEGRATION_VALIDATOR_AVAILABLE = False
    logging.warning("⚠️ IntegrationValidator não disponível. Continuando sem validação integrada.")

try:
    import websocket  # type: ignore
except Exception as e:
    raise RuntimeError("O pacote 'websocket-client' é obrigatório. pip install websocket-client") from e

SCHEMA_VERSION = "2.3.0"

logger = logging.getLogger("market_analyzer")


# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def _decimal_round(value: float, decimals: int = 8) -> float:
    """
    Arredonda usando Decimal para evitar erros de float.
    
    🆕 CORREÇÃO: Trata None e valores inválidos
    
    Args:
        value: Valor a arredondar
        decimals: Número de casas decimais
        
    Returns:
        Valor arredondado com precisão decimal
    """
    if value is None:
        return 0.0
    
    try:
        # Converte para Decimal usando string para evitar erros de float
        d = Decimal(str(value))
        quantize_str = '0.' + '0' * decimals
        return float(d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))
    except (ValueError, TypeError, Exception) as e:
        logger.warning(f"⚠️ Erro ao arredondar {value}: {e}. Retornando 0.0")
        return 0.0


def _validate_volume_consistency(
    volume_total: float,
    volume_compra: float,
    volume_venda: float,
    window_id: str = "UNKNOWN"
) -> Tuple[bool, float]:
    """
    Valida consistência de volumes e retorna flag + volume corrigido.
    
    🆕 CORREÇÃO: Tolerância correta (1e-8 BTC em vez de 0.001 BTC)
    
    Args:
        volume_total: Volume total reportado
        volume_compra: Volume de compra
        volume_venda: Volume de venda
        window_id: ID da janela (para log)
        
    Returns:
        Tupla (is_consistent: bool, corrected_total: float)
    """
    try:
        vt = _decimal_round(volume_total, decimals=8)
        vc = _decimal_round(volume_compra, decimals=8)
        vv = _decimal_round(volume_venda, decimals=8)
        
        expected_total = _decimal_round(vc + vv, decimals=8)
        discrepancy = abs(vt - expected_total)
        
        # 🆕 CORREÇÃO: Tolerância adequada (1e-8 BTC = 0.00000001 BTC)
        tolerance = 1e-8
        
        if discrepancy > tolerance:
            # 🆕 CORREÇÃO: Log apenas se discrepância for significativa (> 0.0001 BTC)
            if discrepancy > 0.0001:
                logger.error(
                    f"🔴 DISCREPÂNCIA SIGNIFICATIVA DE VOLUME (janela {window_id}):\n"
                    f"   volume_compra: {vc:.8f} BTC\n"
                    f"   volume_venda: {vv:.8f} BTC\n"
                    f"   Soma calculada: {expected_total:.8f} BTC\n"
                    f"   Total reportado: {vt:.8f} BTC\n"
                    f"   DIFERENÇA: {discrepancy:.8f} BTC (${discrepancy * 100000:.2f} @ $100k/BTC)\n"
                    f"   ---\n"
                    f"   ✅ Usando soma calculada (mais confiável)"
                )
            else:
                # Log de debug para discrepâncias pequenas
                logger.debug(
                    f"⚠️ Pequena discrepância de volume (janela {window_id}): "
                    f"{discrepancy:.8f} BTC. Corrigindo silenciosamente."
                )
            
            return False, expected_total
        
        return True, vt
        
    except Exception as e:
        logger.error(f"Erro ao validar volumes: {e}")
        # Fallback: usa soma calculada
        return False, _decimal_round(volume_compra + volume_venda, decimals=8)


def _validate_and_fix_cluster_timestamps(
    cluster: Dict[str, Any],
    reference_ts_ms: int,
    window_id: str = "UNKNOWN"
) -> Dict[str, Any]:
    """
    🆕 Valida e corrige timestamps em clusters do heatmap.
    
    Garante:
    - first_seen_ms <= last_seen_ms
    - age_ms >= 0
    - Timestamps positivos
    
    Args:
        cluster: Cluster a validar
        reference_ts_ms: Timestamp de referência
        window_id: ID da janela (para log)
        
    Returns:
        Cluster corrigido
    """
    try:
        first_seen = cluster.get('first_seen_ms')
        last_seen = cluster.get('last_seen_ms')
        age_ms = cluster.get('age_ms')
        
        # Valida first_seen e last_seen
        if first_seen is not None and last_seen is not None:
            first = int(first_seen)
            last = int(last_seen)
            
            # Validar que são positivos
            if first <= 0 or last <= 0:
                logger.warning(
                    f"⚠️ Cluster com timestamps inválidos (janela {window_id}): "
                    f"first={first}, last={last}. Usando reference."
                )
                cluster['first_seen_ms'] = reference_ts_ms
                cluster['last_seen_ms'] = reference_ts_ms
            
            # Validar que first <= last
            elif first > last:
                logger.warning(
                    f"⚠️ Cluster com timestamps invertidos (janela {window_id}): "
                    f"first_seen ({first}) > last_seen ({last}). Invertendo."
                )
                cluster['first_seen_ms'] = last
                cluster['last_seen_ms'] = first
        
        # Valida age_ms
        if age_ms is not None:
            age = int(age_ms)
            
            if age < 0:
                # Tenta recalcular
                last_seen = cluster.get('last_seen_ms')
                if last_seen and reference_ts_ms:
                    recalculated_age = reference_ts_ms - int(last_seen)
                    if recalculated_age >= 0:
                        cluster['age_ms'] = recalculated_age
                        logger.debug(
                            f"✅ age_ms corrigido em cluster (janela {window_id}): "
                            f"{age} → {recalculated_age}"
                        )
                    else:
                        cluster['age_ms'] = 0
                        logger.warning(
                            f"⚠️ age_ms negativo irrecuperável (janela {window_id}). "
                            f"Zerado."
                        )
                else:
                    cluster['age_ms'] = 0
        
        return cluster
        
    except Exception as e:
        logger.error(f"Erro ao validar timestamps de cluster: {e}")
        return cluster


# ========================================================================
# ROBUST CONNECTION MANAGER (COM RECONEXÃO MELHORADA)
# ========================================================================

class RobustConnectionManager:
    """
    Gerenciador robusto de conexão WebSocket com reconexão automática.

    Recursos:
      - Backoff exponencial com jitter
      - Detecção de stale connection (sem mensagens) e FECHAMENTO do socket
      - Heartbeat/uptime/contadores
      - Callbacks de integração com o bot
    """

    def __init__(
        self,
        stream_url: str,
        symbol: str,
        max_reconnect_attempts: int = 15,
        initial_delay: float = 1.0,
        max_delay: float = 120.0,
        backoff_factor: float = 2.0,
        heartbeat_interval: int = 30,
        heartbeat_timeout: int = 120,
    ) -> None:
        self.stream_url = stream_url
        self.symbol = symbol
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout

        self.current_delay = initial_delay
        self.reconnect_count = 0
        self.is_connected = False
        self.last_message_time: Optional[datetime] = None
        self.connection_start_time: Optional[datetime] = None

        self.heartbeat_thread: Optional[threading.Thread] = None
        self.should_stop = False

        self.on_message_callback = None
        self.on_open_callback = None
        self.on_close_callback = None
        self.on_error_callback = None

        self.total_messages_received = 0
        self.total_reconnects = 0
        self.total_errors = 0

        self.ws: Optional[websocket.WebSocketApp] = None

        logger.info(
            "🔌 ConnectionManager v2.3.0 inicializado: %s | max_reconnects=%d | backoff %.1f..%.1fs",
            symbol, max_reconnect_attempts, initial_delay, max_delay
        )

    def set_callbacks(self, on_message=None, on_open=None, on_close=None, on_error=None) -> None:
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_close_callback = on_close
        self.on_error_callback = on_error

    # ---------------- internal ----------------

    def _test_connection(self) -> bool:
        """Testa host/porta antes de abrir o WebSocket."""
        try:
            parsed_url = urlparse(self.stream_url)
            host = parsed_url.hostname
            port = parsed_url.port or (443 if parsed_url.scheme == "wss" else 80)
            if not host:
                return False

            logger.debug("🔍 Testando conectividade: %s:%s", host, port)

            with socket.create_connection((host, port), timeout=5) as sock:
                if parsed_url.scheme == "wss":
                    context = ssl.create_default_context()
                    with context.wrap_socket(sock, server_hostname=host):
                        return True
                return True

        except socket.timeout:
            logger.error("❌ Timeout ao testar conexão")
            return False
        except OSError as e:
            logger.error("❌ Erro de socket: %s", e)
            return False
        except Exception as e:
            logger.error("❌ Erro ao testar conexão: %s", e)
            return False

    def _on_message(self, ws, message):
        """Handler interno para delegar ao callback e atualizar heartbeat."""
        try:
            self.last_message_time = datetime.now(timezone.utc)
            self.total_messages_received += 1
            if self.on_message_callback:
                self.on_message_callback(ws, message)
        except Exception as e:
            logger.error("❌ Erro no processamento da mensagem: %s", e, exc_info=True)

    def _on_open(self, ws):
        self.is_connected = True
        self.reconnect_count = 0
        self.current_delay = self.initial_delay
        self.connection_start_time = datetime.now(timezone.utc)
        self.last_message_time = self.connection_start_time

        logger.info("✅ Conexão estabelecida com %s (tentativa=%d)", self.symbol, self.total_reconnects + 1)
        self._start_monitoring_threads()
        if self.on_open_callback:
            self.on_open_callback(ws)

    def _on_close(self, ws, close_status_code, close_msg):
        self.is_connected = False
        logger.warning("🔌 Conexão fechada: code=%s, msg=%s", close_status_code, close_msg)
        self._stop_monitoring_threads()
        if self.on_close_callback:
            self.on_close_callback(ws, close_status_code, close_msg)

    def _on_error(self, ws, error):
        self.total_errors += 1
        logger.error("❌ Erro WebSocket: %s", error)
        if self.on_error_callback:
            self.on_error_callback(ws, error)

    def _start_monitoring_threads(self) -> None:
        self.should_stop = False
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_monitor, daemon=True)
        self.heartbeat_thread.start()

    def _stop_monitoring_threads(self) -> None:
        self.should_stop = True
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=1.0)

    def _heartbeat_monitor(self) -> None:
        while not self.should_stop and self.is_connected:
            time.sleep(self.heartbeat_interval)
            if not self.last_message_time:
                continue
            time_since_last = (datetime.now(timezone.utc) - self.last_message_time).total_seconds()
            if time_since_last > self.heartbeat_timeout:
                logger.warning("⚠️ Stale connection: %.0fs sem mensagens — fechando socket p/ reconectar", time_since_last)
                try:
                    if self.ws:
                        self.ws.close()
                except Exception:
                    pass
                self.is_connected = False
                break

    # ---------------- public ----------------

    def connect(self) -> None:
        """Tenta conectar com retry/backoff até should_stop ou atingir o limite."""
        while self.reconnect_count < self.max_reconnect_attempts and not self.should_stop:
            try:
                if not self._test_connection():
                    raise ConnectionError("Falha no teste de conectividade")

                logger.info(
                    "🔄 Tentativa %d/%d | delay atual: %.1fs",
                    self.reconnect_count + 1, self.max_reconnect_attempts, self.current_delay
                )

                self.ws = websocket.WebSocketApp(
                    self.stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )

                self.ws.run_forever(ping_interval=self.heartbeat_interval, ping_timeout=10)

                if self.should_stop:
                    break

            except KeyboardInterrupt:
                logger.info("⏹️ Interrompido pelo usuário")
                self.should_stop = True
                break

            except Exception as e:
                self.reconnect_count += 1
                self.total_reconnects += 1
                logger.error("❌ Erro na conexão (%d/%d): %s", self.reconnect_count, self.max_reconnect_attempts, e)

                if self.reconnect_count < self.max_reconnect_attempts and not self.should_stop:
                    jitter = random.uniform(0, 0.1 * self.current_delay)
                    sleep_time = min(self.current_delay + jitter, self.max_delay)
                    logger.info("⏳ Aguardando %.1fs antes de reconectar...", sleep_time)
                    time.sleep(sleep_time)
                    self.current_delay = min(self.current_delay * self.backoff_factor, self.max_delay)

        if self.reconnect_count >= self.max_reconnect_attempts:
            logger.critical("💀 Falha após %d tentativas. Verifique a rede/stream.", self.max_reconnect_attempts)

        self._stop_monitoring_threads()

    def disconnect(self) -> None:
        """Sinaliza parada e fecha o socket para destravar o run_forever."""
        logger.info("🛑 Desconectando...")
        self.should_stop = True
        try:
            if self.ws:
                self.ws.close()
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        uptime = None
        if self.connection_start_time:
            uptime = (datetime.now(timezone.utc) - self.connection_start_time).total_seconds()
        return {
            "is_connected": self.is_connected,
            "total_messages": self.total_messages_received,
            "total_reconnects": self.total_reconnects,
            "total_errors": self.total_errors,
            "uptime_seconds": uptime,
            "reconnect_count": self.reconnect_count,
            "current_delay": self.current_delay,
        }


# ========================================================================
# ENHANCED MARKET ANALYZER
# ========================================================================

@dataclass
class AnalyzerStats:
    total_windows: int = 0
    valid_events: int = 0
    invalid_events: int = 0
    volume_corrections: int = 0
    timestamp_corrections: int = 0  # 🆕


class EnhancedMarketAnalyzer:
    """
    Analisador de mercado COMPLETO com validação robusta v2.3.0.
    
    🆕 CORREÇÕES v2.3.0:
      - Validação de timestamps em clusters do heatmap
      - Correção automática de age_ms negativo
      - Validação de first_seen <= last_seen
      - Tolerância correta (1e-8 BTC)
      - Integração com data_validator
      - Logs informativos sem poluir console
    """

    def __init__(
        self,
        symbol: str,
        time_manager: Optional[TimeManager] = None,
        flow_analyzer: Optional[FlowAnalyzer] = None,
        orderbook_analyzer: Optional[OrderBookAnalyzer] = None,
        validator: Optional[Any] = None,  # 🆕 Tipo flexível
        data_validator: Optional[Any] = None,  # 🆕 Tipo flexível
    ) -> None:
        self.symbol = symbol
        self.time_manager = time_manager or TimeManager()
        self.flow_analyzer = flow_analyzer or FlowAnalyzer(time_manager=self.time_manager)
        self.orderbook_analyzer = orderbook_analyzer or OrderBookAnalyzer(
            symbol=symbol,
            time_manager=self.time_manager,
            cache_ttl_seconds=1.0,
            max_stale_seconds=30.0,
            rate_limit_threshold=10,
        )
        
        # 🆕 Inicialização condicional dos validadores
        if INTEGRATION_VALIDATOR_AVAILABLE:
            self.validator = validator or IntegrationValidator()
        else:
            self.validator = None
            
        if DATA_VALIDATOR_AVAILABLE:
            self.data_validator = data_validator or DataValidator()
        else:
            self.data_validator = None

        self.stats = AnalyzerStats()
        self.last_event: Optional[Dict[str, Any]] = None

        logger.info("=" * 72)
        logger.info("✅ EnhancedMarketAnalyzer v%s inicializado", SCHEMA_VERSION)
        logger.info("   Symbol:           %s", symbol)
        logger.info("   Schema Version:   %s", SCHEMA_VERSION)
        logger.info("   Components:       FlowAnalyzer, OrderBook, ML")
        if self.validator:
            logger.info("   Integration Validator: ATIVO")
        if self.data_validator:
            logger.info("   Data Validator:   ATIVO")
        logger.info("   Features:         Volume Validation, Timestamp Validation, Precision Control")
        logger.info("=" * 72)

    # ---------------- helpers ----------------

    @staticmethod
    def _sanitize_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normaliza e filtra trades inválidos; aceita payload bruto ou {'data': {...}}."""
        clean: List[Dict[str, Any]] = []
        for t in trades or []:
            try:
                if "T" not in t and "data" in t and isinstance(t["data"], dict):
                    t = t["data"]

                T = int(t.get("T", 0))
                p = float(t.get("p", 0) or 0)
                q = float(t.get("q", 0) or 0)
                if T <= 0 or p <= 0 or q <= 0:
                    continue

                m = t.get("m", None)
                if isinstance(m, str):
                    m = m.strip().lower() in {"true", "t", "1", "sell", "ask", "s"}

                clean.append({"T": T, "p": p, "q": q, "m": m})
            except Exception:
                continue
        return clean

    def process_trades(self, trades: List[Dict[str, Any]]) -> None:
        for trade in self._sanitize_trades(trades):
            try:
                self.flow_analyzer.process_trade(trade)
            except Exception as e:
                logger.debug("Erro ao processar trade no FlowAnalyzer: %s", e)

    # ---------------- API principal ----------------

    def analyze_window(self, window_data: List[Dict[str, Any]], window_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Analisa uma janela de trades e retorna um evento integrado.
        
        🆕 v2.3.0: Validação completa de timestamps e volumes
        """
        self.stats.total_windows += 1

        clean_window = self._sanitize_trades(window_data)
        if len(clean_window) < 2:
            logger.warning("⚠️ Janela %s inválida: %d trades válidos", window_id, len(clean_window))
            self.stats.invalid_events += 1
            return None

        try:
            last_trade_ts = clean_window[-1]["T"]

            # Atualiza estado do fluxo
            self.process_trades(clean_window)

            # Métricas de fluxo
            flow_metrics = self.flow_analyzer.get_flow_metrics(reference_epoch_ms=last_trade_ts)

            # Orderbook validado
            orderbook_event = self.orderbook_analyzer.analyze(event_epoch_ms=last_trade_ts, window_id=window_id)
            if not orderbook_event.get("is_valid", False):
                logger.error("❌ Orderbook inválido (janela %s): %s", window_id, orderbook_event.get("erro", "unknown"))
                self.stats.invalid_events += 1
                return None
            orderbook_data = orderbook_event.get("orderbook_data", {})

            # ML features
            df_window = pd.DataFrame(clean_window)
            ml_features = generate_ml_features(
                df=df_window,
                orderbook_data=orderbook_data,
                flow_metrics=flow_metrics,
                lookback_windows=[1, 5, 15],
                volume_ma_window=20,
            )

            # ============================================================
            # 🆕 MAPEAMENTO E VALIDAÇÃO DE VOLUMES (v2.3.0)
            # ============================================================
            order_flow_data = flow_metrics.get("order_flow", {})
            
            # Extrair volumes em BTC com precisão de 8 casas
            volume_compra_btc = _decimal_round(order_flow_data.get("buy_volume_btc", 0.0), decimals=8)
            volume_venda_btc = _decimal_round(order_flow_data.get("sell_volume_btc", 0.0), decimals=8)
            
            # Calcular total com arredondamento decimal
            volume_total_calculated = _decimal_round(volume_compra_btc + volume_venda_btc, decimals=8)
            
            # Validar consistência com tolerância correta
            reported_total = _decimal_round(order_flow_data.get("total_volume_btc", 0.0), decimals=8)
            
            if reported_total > 0:
                is_consistent, volume_total_btc = _validate_volume_consistency(
                    volume_total=reported_total,
                    volume_compra=volume_compra_btc,
                    volume_venda=volume_venda_btc,
                    window_id=window_id or "UNKNOWN"
                )
                
                if not is_consistent:
                    self.stats.volume_corrections += 1
            else:
                volume_total_btc = volume_total_calculated
            
            # Extrair volumes em USD (precisão de 2 casas para display)
            volume_compra_usd = _decimal_round(order_flow_data.get("buy_volume", 0.0), decimals=2)
            volume_venda_usd = _decimal_round(order_flow_data.get("sell_volume", 0.0), decimals=2)

            # ============================================================
            # 🆕 VALIDAÇÃO DE TIMESTAMPS EM CLUSTERS (v2.3.0)
            # ============================================================
            liquidity_heatmap = flow_metrics.get("liquidity_heatmap", {})
            if "clusters" in liquidity_heatmap:
                corrected_clusters = []
                for cluster in liquidity_heatmap["clusters"]:
                    corrected_cluster = _validate_and_fix_cluster_timestamps(
                        cluster=cluster,
                        reference_ts_ms=last_trade_ts,
                        window_id=window_id or "UNKNOWN"
                    )
                    corrected_clusters.append(corrected_cluster)
                    
                    # Incrementa contador se houve correção
                    if corrected_cluster != cluster:
                        self.stats.timestamp_corrections += 1
                
                # Substitui clusters por versão corrigida
                liquidity_heatmap["clusters"] = corrected_clusters
                flow_metrics["liquidity_heatmap"] = liquidity_heatmap

            # ============================================================
            # CRIAÇÃO DO EVENTO
            # ============================================================
            window_duration_ms = clean_window[-1]["T"] - clean_window[0]["T"]
            event: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "tipo_evento": "MarketAnalysis",
                "ativo": self.symbol,
                "window_id": window_id,
                
                # tempo padronizado
                "time_index": self.time_manager.build_time_index(
                    last_trade_ts, 
                    include_local=True, 
                    timespec="milliseconds"
                ),
                
                # fluxo principal
                "cvd": flow_metrics.get("cvd", 0.0),
                "whale_buy_volume": flow_metrics.get("whale_buy_volume", 0.0),
                "whale_sell_volume": flow_metrics.get("whale_sell_volume", 0.0),
                "whale_delta": flow_metrics.get("whale_delta", 0.0),
                "order_flow": flow_metrics.get("order_flow", {}),
                "tipo_absorcao": flow_metrics.get("tipo_absorcao", "Neutra"),
                "participant_analysis": flow_metrics.get("participant_analysis", {}),
                "bursts": flow_metrics.get("bursts", {}),
                "sector_flow": flow_metrics.get("sector_flow", {}),
                
                # volumes mapeados (validados)
                "volume_compra": volume_compra_btc,
                "volume_venda": volume_venda_btc,
                "volume_total": volume_total_btc,
                "volume_compra_usd": volume_compra_usd,
                "volume_venda_usd": volume_venda_usd,
                
                # orderbook
                "orderbook_data": orderbook_data,
                "orderbook_event": orderbook_event,
                
                # ML
                "ml_features": ml_features,
                
                # liquidity heatmap (com timestamps corrigidos)
                "liquidity_heatmap": liquidity_heatmap,
                
                # stats
                "trades_count": len(clean_window),
                "window_duration_ms": int(window_duration_ms) if window_duration_ms > 0 else 0,
            }

            # ============================================================
            # 🆕 VALIDAÇÃO FINAL COM DATA VALIDATOR (v2.3.0)
            # ============================================================
            if self.data_validator:
                try:
                    validated_event = self.data_validator.validate_and_clean(event)
                    
                    if validated_event is None:
                        logger.error("❌ Evento rejeitado pelo DataValidator (janela %s)", window_id)
                        self.stats.invalid_events += 1
                        return None
                    
                    # Usa evento validado
                    event = validated_event
                    
                except Exception as e:
                    logger.error(f"❌ Erro no DataValidator: {e}. Usando evento sem validação final.")
            else:
                logger.debug("⏭️ DataValidator não disponível - pulando validação final")

            # Validação integrada (IntegrationValidator)
            if self.validator:
                validation = self.validator.validate_event(event)
                event["validation"] = validation
                event["is_valid"] = bool(validation.get("is_valid", False))
                event["should_skip"] = bool(validation.get("should_skip", False))

                if event["should_skip"]:
                    self.stats.invalid_events += 1
                    logger.error("❌ EVENTO INVÁLIDO (janela %s): %s", window_id, validation.get("validation_summary"))
                    for issue in validation.get("critical_issues", []):
                        logger.error("   🔴 %s", issue)
                    for issue in validation.get("issues", []):
                        logger.warning("   ⚠️ %s", issue)
                    return None
            else:
                # Se não tiver validator, assume que o evento é válido
                event["is_valid"] = True
                event["should_skip"] = False
                event["validation"] = {
                    "is_valid": True,
                    "should_skip": False,
                    "validation_summary": "Sem validação - aceito por padrão",
                    "issues": [],
                    "critical_issues": [],
                    "warnings": []
                }

            self.stats.valid_events += 1
            self.last_event = event

            if self.validator:
                for warning in event.get("validation", {}).get("warnings", []):
                    logger.debug("⚡ %s", warning)  # 🆕 Debug em vez de warning para não poluir

            of = event.get("order_flow", {})
            logger.info(
                "✅ Janela %s válida: %d trades, delta=%.2f, cvd=%.2f",
                window_id, len(clean_window), of.get("net_flow_1m", 0.0), event.get("cvd", 0.0)
            )
            return event

        except Exception as e:
            logger.exception("❌ Erro ao processar janela %s: %s", window_id, e)
            self.stats.invalid_events += 1
            return None

    # ---------------- métricas/diagnóstico ----------------

    def get_stats(self) -> Dict[str, Any]:
        valid_rate = 100.0 * self.stats.valid_events / max(1, self.stats.total_windows)
        return {
            "total_windows": self.stats.total_windows,
            "valid_events": self.stats.valid_events,
            "invalid_events": self.stats.invalid_events,
            "volume_corrections": self.stats.volume_corrections,
            "timestamp_corrections": self.stats.timestamp_corrections,  # 🆕
            "valid_rate_pct": round(valid_rate, 2),
            "flow_analyzer_stats": self.flow_analyzer.get_stats(),
            "orderbook_analyzer_stats": self.orderbook_analyzer.get_stats(),
            "validator_stats": self.validator.get_stats() if self.validator else {"status": "não disponível"},
            "data_validator_stats": self.data_validator.get_correction_stats() if self.data_validator else {"status": "não disponível"},  # 🆕
        }

    def diagnose(self) -> Dict[str, Any]:
        stats = self.get_stats()
        logger.info("🔍 DIAGNÓSTICO DO MARKET ANALYZER v%s", SCHEMA_VERSION)
        logger.info("-" * 72)
        logger.info("📊 Janelas: total=%d | válidas=%d (%.2f%%) | inválidas=%d",
                    stats["total_windows"], stats["valid_events"], stats["valid_rate_pct"], stats["invalid_events"])
        logger.info("🔧 Correções de volume: %d", stats["volume_corrections"])
        logger.info("⏰ Correções de timestamp: %d", stats["timestamp_corrections"])
        logger.info("🌊 Flow Analyzer: %s", stats["flow_analyzer_stats"])
        logger.info("📚 OrderBook Analyzer: %s", stats["orderbook_analyzer_stats"])
        if self.validator:
            logger.info("✅ Integration Validator: %s", stats["validator_stats"])
        if self.data_validator:
            logger.info("🛡️ Data Validator: %s", stats["data_validator_stats"])
        logger.info("-" * 72)
        return stats


# ========================================================================
# ENHANCED MARKET BOT (orquestra)
# ========================================================================

class EnhancedMarketBot:
    """
    Bot principal v2.3.0 com validação completa.
    """

    def __init__(
        self,
        stream_url: str,
        symbol: str,
        window_size_minutes: int = 5,
        time_manager: Optional[TimeManager] = None,
    ) -> None:
        self.symbol = symbol
        self.window_size_minutes = max(1, int(window_size_minutes))
        self.ny_tz = ZoneInfo("America/New_York")

        self.time_manager = time_manager or TimeManager()

        # componentes principais
        self.market_analyzer = EnhancedMarketAnalyzer(symbol=symbol, time_manager=self.time_manager)
        self.event_saver = EventSaver(sound_alert=True)
        self.health_monitor = HealthMonitor()

        # conexão
        self.connection_manager = RobustConnectionManager(
            stream_url=stream_url,
            symbol=symbol,
            max_reconnect_attempts=15,
            initial_delay=1.0,
            max_delay=120.0,
            backoff_factor=2.0,
        )
        self.connection_manager.set_callbacks(
            on_message=self.on_message,
            on_open=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
        )

        # estado da janela
        self.window_data: List[Dict[str, Any]] = []
        self.window_end_time: Optional[datetime] = None
        self.window_count = 0
        self.previous_event = None

        logger.info("🎯 Enhanced Market Bot v%s | %s | janela=%d min (NY)", SCHEMA_VERSION, symbol, self.window_size_minutes)

    # ---------------- janelas ----------------

    def _update_window_end_time(self) -> None:
        now_ny = datetime.now(self.ny_tz)
        minutes_into_hour = now_ny.minute
        next_window_minute = (minutes_into_hour // self.window_size_minutes + 1) * self.window_size_minutes
        end_time_ny = now_ny.replace(second=0, microsecond=0)
        if next_window_minute >= 60:
            end_time_ny += timedelta(hours=1)
            end_time_ny = end_time_ny.replace(minute=(next_window_minute % 60))
        else:
            end_time_ny = end_time_ny.replace(minute=next_window_minute)

        self.window_end_time = end_time_ny
        logger.info("🕐 Próximo fechamento: %s NY", self.window_end_time.strftime("%H:%M:%S"))

    # ---------------- callbacks websocket ----------------

    def on_message(self, ws, message) -> None:
        try:
            self.health_monitor.heartbeat("market_analyzer")

            if isinstance(message, (bytes, bytearray)):
                message = message.decode("utf-8", errors="ignore")

            raw = json.loads(message)
            trade = raw.get("data", raw)

            if "T" not in trade:
                return

            trade_time = datetime.fromtimestamp(int(trade["T"]) / 1000, tz=self.ny_tz)

            if self.window_end_time is None:
                self._update_window_end_time()

            if trade_time >= self.window_end_time:
                self._process_window()
                self._update_window_end_time()
                self.window_data = [trade]
            else:
                self.window_data.append(trade)

        except json.JSONDecodeError as e:
            logger.error("❌ Erro JSON: %s", e)
        except Exception as e:
            logger.error("❌ Erro on_message: %s", e, exc_info=True)

    def _process_window(self) -> None:
        if not self.window_data:
            logger.warning("⚠️ Janela vazia — pulando")
            return

        self.window_count += 1
        window_id = f"W{self.window_count:04d}"

        try:
            event = self.market_analyzer.analyze_window(window_data=self.window_data, window_id=window_id)

            if event is None:
                logger.warning("⚠️ Janela %s retornou None (inválida)", window_id)
                return

            if event.get("should_skip", False):
                logger.warning("⚠️ Janela %s marcada para skip: %s",
                               window_id, event.get("validation", {}).get("validation_summary"))
                return

            try:
                self.event_saver.save_event(event)
            except Exception as e:
                logger.error("❌ Falha ao salvar evento: %s", e)

            self._log_event(event, window_id)

            ny_time = datetime.now(self.ny_tz)
            print(f"[{ny_time.strftime('%H:%M:%S')} NY] ✅ Janela {window_id} salva")
            print("─" * 80)

        except Exception as e:
            logger.error("❌ Erro ao processar janela %s: %s", window_id, e, exc_info=True)
        finally:
            self.window_data = []

    def _log_event(self, event: Dict[str, Any], window_id: str) -> None:
        """Log formatado com contexto de CVD."""
        try:
            formatted_log = format_flow_log(event, self.previous_event)
            print(formatted_log)
            self.previous_event = event
            
        except Exception as e:
            logger.warning(f"⚠️ Erro ao formatar log: {e}")
            # Fallback simples
            print(f"\n⚠️ Evento {window_id} processado (erro no log)")

    def on_open(self, ws) -> None:
        ny_time = datetime.now(self.ny_tz)
        logger.info("🚀 Bot v%s iniciado para %s — %s NY", SCHEMA_VERSION, self.symbol, ny_time.strftime("%H:%M:%S"))
        self.window_end_time = None

    def on_close(self, ws, close_status_code, close_msg) -> None:
        logger.warning("🔌 Conexão fechada: %s - %s", close_status_code, close_msg)

    def on_error(self, ws, error) -> None:
        logger.error("❌ Erro WebSocket: %s", error)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "window_count": self.window_count,
            "analyzer_stats": self.market_analyzer.get_stats(),
            "connection_stats": self.connection_manager.get_stats(),
            "health_stats": self.health_monitor.get_stats(),
        }

    def diagnose(self) -> None:
        logger.info("\n" + "=" * 72)
        logger.info("🔍 DIAGNÓSTICO COMPLETO DO BOT v%s", SCHEMA_VERSION)
        logger.info("=" * 72)
        stats = self.get_stats()
        logger.info("📊 Bot: janelas processadas: %d", stats["window_count"])
        self.market_analyzer.diagnose()
        logger.info("🔌 Connection: %s", stats["connection_stats"])
        logger.info("🏥 Health: %s", stats["health_stats"])
        logger.info("=" * 72)

    def run(self) -> None:
        try:
            logger.info("🤖 Iniciando bot v%s para %s...", SCHEMA_VERSION, self.symbol)
            self.connection_manager.connect()
        except KeyboardInterrupt:
            logger.info("⏹️ Bot interrompido pelo usuário")
        finally:
            self.diagnose()
            stats = self.get_stats()
            logger.info(
                "\n📊 Estatísticas finais: "
                "janelas=%d | eventos válidos=%d | taxa=%.2f%% | "
                "correções volume=%d | correções timestamp=%d",
                stats["window_count"],
                stats["analyzer_stats"]["valid_events"],
                stats["analyzer_stats"]["valid_rate_pct"],
                stats["analyzer_stats"]["volume_corrections"],
                stats["analyzer_stats"]["timestamp_corrections"]
            )
            self.connection_manager.disconnect()
            self.health_monitor.stop()
            logger.info("✅ Encerrado com segurança.")


# ========================================================================
# EXECUÇÃO DIRETA
# ========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    tm = TimeManager(
        sync_interval_minutes=30,
        max_init_attempts=3,
        max_acceptable_offset_ms=600,  # 🆕 Ajustado
    )

    bot = EnhancedMarketBot(
        stream_url=config.STREAM_URL,
        symbol=config.SYMBOL,
        window_size_minutes=getattr(config, "WINDOW_SIZE_MINUTES", 5),
        time_manager=tm,
    )

    bot.run()