# report_generator.py

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def generate_ai_analysis_report(event_data, ml_features, orderbook_data, historical_vp, multi_tf):
    """
    Gera um relatório estruturado de análise institucional com base nos dados de evento e contexto.
    """
    # --- Extração segura de dados ---
    event_type = event_data.get("tipo_evento", "")
    absorption_side = event_data.get("absorption_side", "")
    aggression_side = event_data.get("aggression_side", "")
    delta = event_data.get("delta", 0)
    volume_total = event_data.get("volume_total", 0)
    price_close = event_data.get("preco_fechamento", 0)
    
    # Order flow
    aggressive_buy_pct = event_data.get("fluxo_continuo", {}).get("order_flow", {}).get("aggressive_buy_pct", 0)
    aggressive_sell_pct = event_data.get("fluxo_continuo", {}).get("order_flow", {}).get("aggressive_sell_pct", 0)
    buy_sell_ratio = event_data.get("fluxo_continuo", {}).get("order_flow", {}).get("buy_sell_ratio", 0)
    volume_compra = event_data.get("volume_compra", 0)
    volume_venda = event_data.get("volume_venda", 0)

    # ML Features
    flow_imbalance = ml_features.get("microstructure", {}).get("flow_imbalance", 0)
    order_book_slope = ml_features.get("microstructure", {}).get("order_book_slope", 0)
    liquidity_gradient = ml_features.get("volume_features", {}).get("liquidity_gradient", 0)
    volume_sma_ratio = ml_features.get("volume_features", {}).get("volume_sma_ratio", 0)
    momentum_score = ml_features.get("price_features", {}).get("momentum_score", 0)
    tick_rule_sum = ml_features.get("microstructure", {}).get("tick_rule_sum", 0)

    # Volume Profile Diário
    vp_daily = historical_vp.get("daily", {})
    poc_daily = vp_daily.get("poc", 0)
    val_daily = vp_daily.get("val", 0)
    vah_daily = vp_daily.get("vah", 0)

    # HVNs próximos (filtrar acima do preço atual)
    hvns = vp_daily.get("hvns", [])
    hvns_acima = [hvn for hvn in hvns if hvn > price_close]
    hvns_acima.sort()
    proximos_hvns = hvns_acima[:6]  # até 6 níveis

    # --- Interpretação ---
    interpretation_lines = []

    # 1) Order Flow
    interpretation_lines.append(f"- **Order Flow:** Δ = {delta:+.2f} (positivo)" if delta >= 0 else f"- **Order Flow:** Δ = {delta:+.2f} (negativo)")
    interpretation_lines.append(f"  Buy volume ({volume_compra:.2f}) {'>' if volume_compra > volume_venda else '<'} Sell volume ({volume_venda:.2f}).")
    interpretation_lines.append(f"  Razão Buy/Sell = {buy_sell_ratio:.2f}, corroborando pressão {'compradora' if buy_sell_ratio > 1 else 'vendedora'}.")

    # Correção crítica: NÃO multiplicar por 100
    if aggressive_buy_pct > 0 or aggressive_sell_pct > 0:
        interpretation_lines.append(
            f"  Fluxo agressivo mostra {aggressive_buy_pct:.2f}% buy vs {aggressive_sell_pct:.2f}% sell."
        )

    # 2) Liquidez
    bid_depth = orderbook_data.get("spread_metrics", {}).get("bid_depth_usd", 0)
    ask_depth = orderbook_data.get("spread_metrics", {}).get("ask_depth_usd", 0)
    imbalance_ob = orderbook_data.get("imbalance", 0)
    interpretation_lines.append(
        f"- **Liquidez:** Profundidade do livro: bids = ${bid_depth:,.0f}, asks = ${ask_depth:,.0f}. "
        f"Imbalance = {imbalance_ob:+.2f} → {'oferta mais profunda' if imbalance_ob < 0 else 'demanda mais profunda'}."
    )

    # 3) Zona
    dentro_value_area = val_daily <= price_close <= vah_daily
    zona_status = f"Preço atual (${price_close:,.2f}) {'está' if dentro_value_area else 'não está'} dentro da Value Area diária (${val_daily:,.0f} – ${vah_daily:,.0f})."
    interpretation_lines.append(f"- **Zona:** {zona_status}")
    if proximos_hvns:
        hvns_str = ", ".join([f"${h:.0f}" for h in proximos_hvns])
        interpretation_lines.append(f"  Próximos HVNs acima: {hvns_str}.")
    else:
        interpretation_lines.append("  Nenhum HVN estrutural próximo acima.")

    # 4) Microestrutura / ML
    ml_lines = []
    ml_lines.append(f"  - flow_imbalance = {flow_imbalance:+.3f} → {'desvio positivo' if flow_imbalance > 0 else 'pressão vendedora'} no fluxo.")
    ml_lines.append(f"  - order_book_slope = {order_book_slope:+.3f} → {'maior densidade ofertada' if order_book_slope < 0 else 'maior densidade demandada'}.")
    ml_lines.append(f"  - liquidity_gradient = {liquidity_gradient:+.2f}")
    ml_lines.append(f"  - volume_sma_ratio = {volume_sma_ratio:.2f} → volume {'elevado' if volume_sma_ratio > 100 else 'normal/baixo'} vs média.")
    ml_lines.append(f"  - momentum_score = {momentum_score:.5f} → viés de {'alta' if momentum_score > 1 else 'baixa'}.")
    ml_lines.append(f"  - tick_rule_sum = {tick_rule_sum:.3f} → {'predomínio de trades com tick up' if tick_rule_sum > 0 else 'predomínio de trades com tick down'}.")

    interpretation_lines.append("- **Microestrutura/ML:**")
    interpretation_lines.extend([f"  {line}" for line in ml_lines])

    # --- Força dominante ---
    if delta > 0:
        if absorption_side == "sell":
            dominant_force = "**Compradores agressivos**, com vendedores absorvendo sob resistência."
        else:
            dominant_force = "**Compradores agressivos**."
    else:
        if absorption_side == "buy":
            dominant_force = "**Vendedores agressivos**, com compradores absorvendo suporte."
        else:
            dominant_force = "**Vendedores agressivos**."

    # --- Expectativa ---
    curto_prazo = "Teste das zonas de HVN acima." if delta > 0 else "Pressão vendedora testando suportes."
    medio_prazo = "Sustentação acima do POC diário favorece viés de alta." if price_close > poc_daily else "Romper POC com volume será necessário para reversão."

    # --- Montagem final do relatório ---
    report = f"""
═════════════════════════ ANÁLISE PROFISSIONAL DA IA ═════════════════════════
1) **Interpretação (order flow, liquidez, zona, microestrutura/ML):**
{'\n'.join(interpretation_lines)}

2) **Força dominante:**
{dominant_force}

3) **Expectativa (curto/médio prazo):**
- **Curto prazo:** {curto_prazo}
- **Médio prazo:** {medio_prazo}

═══════════════════════════════════════════════════════════════════════════
"""
    return report.strip()