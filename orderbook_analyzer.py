import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging
import numpy as np
import time

NY_TZ = ZoneInfo("America/New_York")

class OrderBookAnalyzer:
    def __init__(self, symbol="BTCUSDT", limit=100, liquidity_flow_alert_percentage=0.5, wall_std_dev_factor=3.0):
        self.symbol = symbol
        self.limit = limit
        self.api_url = "https://api.binance.com/api/v3/depth"
        self.liquidity_flow_alert_percentage = liquidity_flow_alert_percentage
        self.wall_std_dev_factor = wall_std_dev_factor
        
        # Estado para fluxo de liquidez
        self.prev_top_bids_volume = None
        self.prev_top_asks_volume = None

        # <<< ALTERAÇÃO ETAPA 5: Estado para detecção de Icebergs
        self.iceberg_tracker = {} # Formato: {'price_side': {'hits': int, 'last_qty': float, 'timestamp': float}}
        self.ICEBERG_HIT_THRESHOLD = 3 # Alerta após 3 recargas/consumos parciais
        self.ICEBERG_TTL_SECONDS = 60 * 10 # Um candidato a iceberg expira após 10 minutos sem atividade

    def fetch_order_book(self):
        params = {"symbol": self.symbol, "limit": self.limit}
        response = requests.get(self.api_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        bids = [(float(price), float(qty)) for price, qty in data['bids']]
        asks = [(float(price), float(qty)) for price, qty in data['asks']]
        return bids, asks

    def calculate_metrics(self, bids, asks):
        buy_volume_total = sum(qty for _, qty in bids)
        sell_volume_total = sum(qty for _, qty in asks)
        total_volume = buy_volume_total + sell_volume_total
        imbalance = (buy_volume_total - sell_volume_total) / total_volume if total_volume > 0 else 0
        top_buy_volume = sum(qty for _, qty in bids[:10])
        top_sell_volume = sum(qty for _, qty in asks[:10])
        volume_ratio = top_buy_volume / top_sell_volume if top_sell_volume > 0 else float('inf')
        top_bids_pressure = sum(price * qty for price, qty in bids[:10])
        top_asks_pressure = sum(price * qty for price, qty in asks[:10])
        total_pressure = top_bids_pressure + top_asks_pressure
        pressure = (top_bids_pressure - top_asks_pressure) / total_pressure if total_pressure > 0 else 0
        return imbalance, volume_ratio, pressure, top_buy_volume, top_sell_volume

    def analyze_liquidity_flow(self, current_bids_volume, current_asks_volume):
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
        alerts = []
        quantities = [qty for _, qty in bids] + [qty for _, qty in asks]
        if len(quantities) < 10: return alerts, 0
        mean_qty = np.mean(quantities); std_qty = np.std(quantities)
        wall_threshold = mean_qty + (self.wall_std_dev_factor * std_qty)
        for price, qty in bids[:20]:
            if qty > wall_threshold:
                alerts.append(f"PAREDE DE COMPRA: {qty:,.2f} @ ${price:,.2f} (limite: {wall_threshold:,.2f})")
        for price, qty in asks[:20]:
            if qty > wall_threshold:
                alerts.append(f"PAREDE DE VENDA: {qty:,.2f} @ ${price:,.2f} (limite: {wall_threshold:,.2f})")
        return alerts, wall_threshold

    # <<< ALTERAÇÃO ETAPA 5: Nova função para detectar icebergs
    def _detect_iceberg_orders(self, bids, asks, wall_threshold):
        alerts = []
        now = time.time()
        
        # Transforma o livro de ofertas atual em um dicionário para busca rápida
        current_book_map = {f"{price}_{side}": qty for side, book in [("bid", bids), ("ask", asks)] for price, qty in book}

        # 1. Atualiza e verifica os icebergs rastreados
        for key, tracker in list(self.iceberg_tracker.items()):
            price, side = key.split("_")
            
            # Se a ordem não existe mais ou expirou, remove do rastreador
            if key not in current_book_map or (now - tracker['timestamp']) > self.ICEBERG_TTL_SECONDS:
                del self.iceberg_tracker[key]
                continue

            current_qty = current_book_map[key]
            last_qty = tracker['last_qty']

            # Lógica de detecção: o volume foi consumido (diminuiu), mas depois recarregado (voltou a ser grande)
            if current_qty < last_qty:
                # O volume foi consumido. Apenas atualizamos o valor.
                tracker['last_qty'] = current_qty
                tracker['timestamp'] = now
            elif current_qty > last_qty and last_qty < tracker['initial_qty']:
                # O volume foi recarregado após um consumo! Isso é um "hit".
                tracker['hits'] += 1
                tracker['last_qty'] = current_qty
                tracker['initial_qty'] = max(tracker['initial_qty'], current_qty) # Atualiza o tamanho inicial
                tracker['timestamp'] = now
                
                if tracker['hits'] >= self.ICEBERG_HIT_THRESHOLD:
                    side_str = "COMPRA" if side == "bid" else "VENDA"
                    alerts.append(f"ICEBERG DE {side_str}?: Atividade de recarga em ${float(price):,.2f} (detectada {tracker['hits']}x)")
                    # Reseta os hits para não poluir o log com o mesmo alerta
                    tracker['hits'] = 0

        # 2. Adiciona novas paredes como candidatas a icebergs
        all_walls = [(p, q, "bid") for p, q in bids[:20] if q > wall_threshold] + \
                    [(p, q, "ask") for p, q in asks[:20] if q > wall_threshold]
        
        for price, qty, side in all_walls:
            key = f"{price}_{side}"
            if key not in self.iceberg_tracker:
                self.iceberg_tracker[key] = {'hits': 0, 'last_qty': qty, 'initial_qty': qty, 'timestamp': now}
                
        return alerts

    def analyze_order_book(self):
        try:
            bids, asks = self.fetch_order_book()
            if not bids or not asks:
                raise ValueError("Livro de ofertas vazio ou inválido.")

            imbalance, volume_ratio, pressure, top_buy_vol, top_sell_vol = self.calculate_metrics(bids, asks)
            flow_alerts = self.analyze_liquidity_flow(top_buy_vol, top_sell_vol)
            wall_alerts, wall_threshold = self.analyze_liquidity_walls(bids, asks)
            
            # <<< ALTERAÇÃO ETAPA 5: Chamada da nova função
            iceberg_alerts = self._detect_iceberg_orders(bids, asks, wall_threshold)

            all_alerts = flow_alerts + wall_alerts + iceberg_alerts
            is_signal = bool(all_alerts) # Qualquer alerta de liquidez agora é um sinal

            if imbalance > 0.3 and volume_ratio > 1.2:
                result, description = "Demanda Forte", f"Pressão de compra. Imbalance: {imbalance:.2%}, Ratio: {volume_ratio:.2f}"
                is_signal = True
            elif imbalance < -0.3 and volume_ratio < 0.8:
                result, description = "Oferta Forte", f"Pressão de venda. Imbalance: {imbalance:.2%}, Ratio: {volume_ratio:.2f}"
                is_signal = True
            else:
                result, description = "Equilíbrio", "Livro de ofertas equilibrado."

            if is_signal and result == "Equilíbrio":
                result = "Alerta de Liquidez"
                description = " | ".join(all_alerts)

            event = {"is_signal": is_signal, "timestamp": datetime.now(timezone.utc).astimezone(NY_TZ).isoformat(timespec="seconds"), 
                     "tipo_evento": "OrderBook", "resultado_da_batalha": result, "descricao": description, "ativo": self.symbol,
                     "imbalance": round(imbalance, 4), "volume_ratio": round(volume_ratio, 4) if volume_ratio != float('inf') else 'inf',
                     "pressure": round(pressure, 4)}
            if all_alerts:
                event["alertas_liquidez"] = all_alerts
            
            return event
        except Exception as e:
            logging.error(f"Erro ao analisar o livro de ofertas: {e}")
            return {"is_signal": False, "timestamp": datetime.now(timezone.utc).astimezone(NY_TZ).isoformat(timespec="seconds"), 
                    "tipo_evento": "OrderBook", "resultado_da_batalha": "Erro", "descricao": f"Falha na análise: {e}", "ativo": self.symbol}