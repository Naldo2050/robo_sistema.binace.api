# orderbook_analyzer.py
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
import numpy as np

NY_TZ = ZoneInfo("America/New_York")

class OrderBookAnalyzer:
    def __init__(self, symbol="BTCUSDT", limit=100, liquidity_flow_alert_percentage=0.5, wall_std_dev_factor=3.0):
        self.symbol = symbol
        self.limit = limit # Aumentado para ter uma amostra estatística melhor
        self.api_url = "https://api.binance.com/api/v3/depth"
        self.liquidity_flow_alert_percentage = liquidity_flow_alert_percentage
        self.wall_std_dev_factor = wall_std_dev_factor
        
        # Estado para rastrear fluxo de liquidez
        self.prev_top_bids_volume = None
        self.prev_top_asks_volume = None

    def fetch_order_book(self):
        params = {"symbol": self.symbol, "limit": self.limit}
        response = requests.get(self.api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        bids = [(float(price), float(qty)) for price, qty in data['bids']]
        asks = [(float(price), float(qty)) for price, qty in data['asks']]
        return bids, asks

    def calculate_metrics(self, bids, asks):
        # Volumes totais (para imbalance)
        buy_volume_total = sum(qty for _, qty in bids)
        sell_volume_total = sum(qty for _, qty in asks)
        total_volume = buy_volume_total + sell_volume_total
        imbalance = (buy_volume_total - sell_volume_total) / total_volume if total_volume > 0 else 0

        # Volumes no topo do livro (para ratio)
        top_buy_volume = sum(qty for _, qty in bids[:10])
        top_sell_volume = sum(qty for _, qty in asks[:10])
        volume_ratio = top_buy_volume / top_sell_volume if top_sell_volume > 0 else float('inf')

        # Pressão ponderada pelo preço
        top_bids_pressure = sum(price * qty for price, qty in bids[:10])
        top_asks_pressure = sum(price * qty for price, qty in asks[:10])
        total_pressure = top_bids_pressure + top_asks_pressure
        pressure = (top_bids_pressure - top_asks_pressure) / total_pressure if total_pressure > 0 else 0

        return imbalance, volume_ratio, pressure, top_buy_volume, top_sell_volume

    def analyze_liquidity_flow(self, current_bids_volume, current_asks_volume):
        """Analisa a mudança na liquidez no topo do livro de ofertas."""
        alerts = []
        if self.prev_top_bids_volume and current_bids_volume > 0:
            change_percent = (current_bids_volume - self.prev_top_bids_volume) / self.prev_top_bids_volume
            if change_percent > self.liquidity_flow_alert_percentage:
                alerts.append(f"FLUXO: Aumento de {change_percent:.1%} na liquidez de compra (Bids).")
            elif change_percent < -self.liquidity_flow_alert_percentage:
                alerts.append(f"FLUXO: Retirada de {abs(change_percent):.1%} da liquidez de compra (Bids).")
        if self.prev_top_asks_volume and current_asks_volume > 0:
            change_percent = (current_asks_volume - self.prev_top_asks_volume) / self.prev_top_asks_volume
            if change_percent > self.liquidity_flow_alert_percentage:
                alerts.append(f"FLUXO: Aumento de {change_percent:.1%} na liquidez de venda (Asks).")
            elif change_percent < -self.liquidity_flow_alert_percentage:
                alerts.append(f"FLUXO: Retirada de {abs(change_percent):.1%} da liquidez de venda (Asks).")
        self.prev_top_bids_volume = current_bids_volume
        self.prev_top_asks_volume = current_asks_volume
        return alerts

    def analyze_liquidity_walls(self, bids, asks):
        """Analisa o livro de ofertas para encontrar 'paredes' de liquidez (ordens outliers)."""
        alerts = []
        quantities = [qty for _, qty in bids] + [qty for _, qty in asks]
        if len(quantities) < 10: return alerts # Amostra muito pequena

        mean_qty = np.mean(quantities)
        std_qty = np.std(quantities)
        wall_threshold = mean_qty + (self.wall_std_dev_factor * std_qty)

        # Verifica paredes de compra nos 20 primeiros níveis de preço
        for price, qty in bids[:20]:
            if qty > wall_threshold:
                alerts.append(f"PAREDE DE COMPRA: {qty:,.2f} @ ${price:,.2f} (limite: {wall_threshold:,.2f})")
        
        # Verifica paredes de venda nos 20 primeiros níveis de preço
        for price, qty in asks[:20]:
            if qty > wall_threshold:
                alerts.append(f"PAREDE DE VENDA: {qty:,.2f} @ ${price:,.2f} (limite: {wall_threshold:,.2f})")
        
        return alerts

    def analyze_order_book(self):
        try:
            bids, asks = self.fetch_order_book()
            if not bids or not asks:
                raise ValueError("Livro de ofertas vazio ou inválido.")

            imbalance, volume_ratio, pressure, top_buy_vol, top_sell_vol = self.calculate_metrics(bids, asks)
            flow_alerts = self.analyze_liquidity_flow(top_buy_vol, top_sell_vol)
            wall_alerts = self.analyze_liquidity_walls(bids, asks) # <<<<<<< NOVA CHAMADA

            all_alerts = flow_alerts + wall_alerts
            is_signal = False
            
            # Um alerta de liquidez (fluxo ou parede) agora também é considerado um sinal
            if all_alerts:
                is_signal = True

            if imbalance > 0.3 and volume_ratio > 1.2:
                result = "Demanda Forte"
                description = f"Pressão de compra detectada. Imbalance: {imbalance:.2%}, Ratio: {volume_ratio:.2f}"
                is_signal = True
            elif imbalance < -0.3 and volume_ratio < 0.8:
                result = "Oferta Forte"
                description = f"Pressão de venda detectada. Imbalance: {imbalance:.2%}, Ratio: {volume_ratio:.2f}"
                is_signal = True
            else:
                result = "Equilíbrio"
                description = "Livro de ofertas equilibrado."

            timestamp = datetime.now(timezone.utc).astimezone(NY_TZ).isoformat(timespec="seconds")
            event = {
                "is_signal": is_signal, "timestamp": timestamp, "tipo_evento": "OrderBook",
                "resultado_da_batalha": result, "descricao": description, "ativo": self.symbol,
                "imbalance": round(imbalance, 4),
                "volume_ratio": round(volume_ratio, 4) if volume_ratio != float('inf') else 'inf',
                "pressure": round(pressure, 4),
            }
            if all_alerts:
                event["alertas_liquidez"] = all_alerts
                if result == "Equilíbrio": # Se não havia sinal, mas há alertas
                    event["resultado_da_batalha"] = "Alerta de Liquidez"
                    event["descricao"] = " | ".join(all_alerts)
            
            return event
        except Exception as e:
            logging.error(f"Erro ao analisar o livro de ofertas: {e}")
            timestamp = datetime.now(timezone.utc).astimezone(NY_TZ).isoformat(timespec="seconds")
            return {
                "is_signal": False, "timestamp": timestamp, "tipo_evento": "OrderBook",
                "resultado_da_batalha": "Erro", "descricao": f"Falha na análise: {e}", "ativo": self.symbol,
            }