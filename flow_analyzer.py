# flow_analyzer.py
import time
import logging
from threading import Lock
from config import CVD_RESET_INTERVAL_HOURS, WHALE_TRADE_THRESHOLD

class FlowAnalyzer:
    def __init__(self):
        # Métricas de CVD
        self.cvd = 0.0
        
        # Métricas de Whale Flow
        self.whale_threshold = WHALE_TRADE_THRESHOLD
        self.whale_buy_volume = 0.0
        self.whale_sell_volume = 0.0
        self.whale_delta = 0.0

        # Controle de Reset
        self.last_reset_time = time.time()
        self.reset_interval_seconds = CVD_RESET_INTERVAL_HOURS * 3600
        self._lock = Lock()
        logging.info("✅ Analisador de Fluxo Contínuo (CVD & Whale Flow) inicializado.")

    def _reset_metrics(self):
        """Reseta todas as métricas acumuladas."""
        self.cvd = 0.0
        self.whale_buy_volume = 0.0
        self.whale_sell_volume = 0.0
        self.whale_delta = 0.0
        self.last_reset_time = time.time()
        logging.info("🔄 Métricas de Fluxo Contínuo (CVD, Whale Flow) foram resetadas.")

    def _check_reset(self):
        """Verifica se as métricas devem ser resetadas com base no tempo."""
        if time.time() - self.last_reset_time > self.reset_interval_seconds:
            with self._lock:
                self._reset_metrics()

    def process_trade(self, trade: dict):
        """Processa um único trade para atualizar o CVD e o Whale Flow."""
        self._check_reset()
        try:
            qty = float(trade.get('q', 0))
            is_buyer_maker = trade.get('m', False)

            trade_delta = -qty if is_buyer_maker else qty
            
            with self._lock:
                # Atualiza CVD com todos os trades
                self.cvd += trade_delta
                
                # Verifica e atualiza Whale Flow apenas para trades grandes
                if qty >= self.whale_threshold:
                    if trade_delta > 0: # Agressão de compra
                        self.whale_buy_volume += qty
                    else: # Agressão de venda
                        self.whale_sell_volume += abs(trade_delta)
                    
                    self.whale_delta += trade_delta

        except (ValueError, TypeError) as e:
            logging.warning(f"Trade inválido recebido para análise de fluxo: {trade} - Erro: {e}")

    def get_flow_metrics(self) -> dict:
        """Retorna as métricas de fluxo atuais de forma segura."""
        with self._lock:
            return {
                "cvd": self.cvd,
                "whale_buy_volume": self.whale_buy_volume,
                "whale_sell_volume": self.whale_sell_volume,
                "whale_delta": self.whale_delta
            }