# report_generator.py - Gerador de relatórios com formatação consistente

import logging
from datetime import datetime

# 🔹 Importa utilitários de formatação
from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific
)

logger = logging.getLogger(__name__)


def generate_ai_analysis_report(event_data, ml_features, orderbook_data, historical_vp, multi_tf):
    """
    Gera um relatório estruturado de análise institucional com base nos dados de evento e contexto.
    Aplica formatação consistente para todos os valores numéricos.
    """
    try:
        # --- Extração segura de dados ---
        event_type = event_data.get("tipo_evento", "")
        absorption_side = event_data.get("absorption_side", "")
        aggression_side = event_data.get("aggression_side", "")
        delta = event_data.get("delta", 0)
        volume_total = event_data.get("volume_total", 0)
        price_close = event_data.get("preco_fechamento", 0)
        
        # Order flow
        flow_cont = event_data.get("fluxo_continuo", {})
        order_flow = flow_cont.get("order_flow", {})
        aggressive_buy_pct = order_flow.get("aggressive_buy_pct", 0)
        aggressive_sell_pct = order_flow.get("aggressive_sell_pct", 0)
        buy_sell_ratio = order_flow.get("buy_sell_ratio", 0)
        volume_compra = event_data.get("volume_compra", 0)
        volume_venda = event_data.get("volume_venda", 0)

        # ML Features
        microstructure = ml_features.get("microstructure", {})
        volume_features = ml_features.get("volume_features", {})
        price_features = ml_features.get("price_features", {})
        
        flow_imbalance = microstructure.get("flow_imbalance", 0)
        order_book_slope = microstructure.get("order_book_slope", 0)
        liquidity_gradient = volume_features.get("liquidity_gradient", 0)
        volume_sma_ratio = volume_features.get("volume_sma_ratio", 0)
        momentum_score = price_features.get("momentum_score", 0)
        tick_rule_sum = microstructure.get("tick_rule_sum", 0)

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

        # --- Interpretação com formatação ---
        interpretation_lines = []

        # 1) Order Flow - COM FORMATAÇÃO CORRIGIDA
        delta_fmt = format_delta(delta)
        vol_compra_fmt = format_large_number(volume_compra)
        vol_venda_fmt = format_large_number(volume_venda)
        
        interpretation_lines.append(
            f"- **Order Flow:** Δ = {delta_fmt} ({'positivo' if delta >= 0 else 'negativo'})"
        )
        interpretation_lines.append(
            f"  Buy volume ({vol_compra_fmt}) "
            f"{'>' if volume_compra > volume_venda else '<'} "
            f"Sell volume ({vol_venda_fmt})."
        )
        
        # 🔹 CORREÇÃO: Buy/Sell Ratio formatado adequadamente
        bs_ratio_fmt = f"{buy_sell_ratio:.2f}" if buy_sell_ratio < 10 else format_large_number(buy_sell_ratio)
        interpretation_lines.append(
            f"  Razão Buy/Sell = {bs_ratio_fmt}, corroborando pressão "
            f"{'compradora' if buy_sell_ratio > 1 else 'vendedora'}."
        )

        # 🔹 CORREÇÃO CRÍTICA: Percentuais já estão em escala 0-100 (não multiplicar)
        if aggressive_buy_pct > 0 or aggressive_sell_pct > 0:
            buy_pct_fmt = format_percent(aggressive_buy_pct)
            sell_pct_fmt = format_percent(aggressive_sell_pct)
            interpretation_lines.append(
                f"  Fluxo agressivo mostra {buy_pct_fmt} buy vs {sell_pct_fmt} sell."
            )

        # 2) Liquidez - COM FORMATAÇÃO CORRIGIDA
        spread_metrics = orderbook_data.get("spread_metrics", {})
        bid_depth = spread_metrics.get("bid_depth_usd", 0)
        ask_depth = spread_metrics.get("ask_depth_usd", 0)
        imbalance_ob = orderbook_data.get("imbalance", 0)
        
        bid_depth_fmt = format_large_number(bid_depth)
        ask_depth_fmt = format_large_number(ask_depth)
        imbalance_fmt = format_delta(imbalance_ob)
        
        interpretation_lines.append(
            f"- **Liquidez:** Profundidade do livro: bids = ${bid_depth_fmt}, asks = ${ask_depth_fmt}. "
            f"Imbalance = {imbalance_fmt} → "
            f"{'oferta mais profunda' if imbalance_ob < 0 else 'demanda mais profunda'}."
        )

        # 3) Zona - COM FORMATAÇÃO CORRIGIDA
        dentro_value_area = val_daily <= price_close <= vah_daily
        
        price_fmt = format_price(price_close)
        val_fmt = format_price(val_daily)
        vah_fmt = format_price(vah_daily)
        
        zona_status = (
            f"Preço atual (${price_fmt}) "
            f"{'está' if dentro_value_area else 'não está'} "
            f"dentro da Value Area diária (${val_fmt} – ${vah_fmt})."
        )
        interpretation_lines.append(f"- **Zona:** {zona_status}")
        
        if proximos_hvns:
            hvns_str = ", ".join([f"${format_price(h)}" for h in proximos_hvns])
            interpretation_lines.append(f"  Próximos HVNs acima: {hvns_str}.")
        else:
            interpretation_lines.append("  Nenhum HVN estrutural próximo acima.")

        # 4) Microestrutura / ML - COM FORMATAÇÃO CORRIGIDA
        ml_lines = []
        
        # Flow imbalance
        flow_imb_fmt = format_scientific(flow_imbalance, decimals=3)
        ml_lines.append(
            f"  - flow_imbalance = {flow_imb_fmt} → "
            f"{'desvio positivo' if flow_imbalance > 0 else 'pressão vendedora'} no fluxo."
        )
        
        # Order book slope
        ob_slope_fmt = format_scientific(order_book_slope, decimals=3)
        ml_lines.append(
            f"  - order_book_slope = {ob_slope_fmt} → "
            f"{'maior densidade ofertada' if order_book_slope < 0 else 'maior densidade demandada'}."
        )
        
        # Liquidity gradient
        liq_grad_fmt = format_scientific(liquidity_gradient, decimals=2)
        ml_lines.append(f"  - liquidity_gradient = {liq_grad_fmt}")
        
        # Volume SMA ratio - CORREÇÃO: converter para percentual se necessário
        if volume_sma_ratio <= 10:  # Provavelmente está em escala 0-10 (ex: 1.5 = 150%)
            vol_sma_display = format_percent(volume_sma_ratio * 100)
        else:  # Já está em percentual (ex: 150)
            vol_sma_display = format_percent(volume_sma_ratio)
        
        ml_lines.append(
            f"  - volume_sma_ratio = {vol_sma_display} → "
            f"volume {'elevado' if volume_sma_ratio > 1.5 else 'normal/baixo'} vs média."
        )
        
        # Momentum score
        momentum_fmt = format_scientific(momentum_score, decimals=5)
        ml_lines.append(
            f"  - momentum_score = {momentum_fmt} → "
            f"viés de {'alta' if momentum_score > 1 else 'baixa'}."
        )
        
        # Tick rule sum
        tick_fmt = format_scientific(tick_rule_sum, decimals=3)
        ml_lines.append(
            f"  - tick_rule_sum = {tick_fmt} → "
            f"{'predomínio de trades com tick up' if tick_rule_sum > 0 else 'predomínio de trades com tick down'}."
        )

        interpretation_lines.append("- **Microestrutura/ML:**")
        interpretation_lines.extend(ml_lines)

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
        curto_prazo = (
            "Teste das zonas de HVN acima." if delta > 0 
            else "Pressão vendedora testando suportes."
        )
        
        poc_fmt = format_price(poc_daily)
        medio_prazo = (
            f"Sustentação acima do POC diário (${poc_fmt}) favorece viés de alta." 
            if price_close > poc_daily 
            else f"Romper POC (${poc_fmt}) com volume será necessário para reversão."
        )

        # --- Contexto Multi-Timeframe (se disponível) ---
        mtf_section = ""
        if multi_tf:
            mtf_lines = ["- **Contexto Multi-Timeframe:**"]
            for tf, data in multi_tf.items():
                if isinstance(data, dict):
                    tendencia = data.get('tendencia', 'N/A')
                    mtf_lines.append(f"  - {tf.upper()}: {tendencia}")
                else:
                    mtf_lines.append(f"  - {tf.upper()}: {data}")
            
            if len(mtf_lines) > 1:  # Só adiciona se houver dados
                mtf_section = "\n".join(mtf_lines) + "\n"

        # --- Montagem final do relatório ---
        report = f"""
═════════════════════════ ANÁLISE PROFISSIONAL DA IA ═════════════════════════

1) **Interpretação (order flow, liquidez, zona, microestrutura/ML):**
{chr(10).join(interpretation_lines)}

2) **Força dominante:**
{dominant_force}

3) **Expectativa (curto/médio prazo):**
- **Curto prazo:** {curto_prazo}
- **Médio prazo:** {medio_prazo}

{mtf_section}
═══════════════════════════════════════════════════════════════════════════
"""
        return report.strip()
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório de análise: {e}", exc_info=True)
        return f"""
═════════════════════════ ANÁLISE PROFISSIONAL DA IA ═════════════════════════

⚠️ Erro ao gerar relatório completo: {str(e)}

Dados parciais disponíveis:
- Tipo de evento: {event_data.get('tipo_evento', 'N/A')}
- Preço: ${format_price(event_data.get('preco_fechamento', 0))}
- Delta: {format_delta(event_data.get('delta', 0))}
- Volume: {format_large_number(event_data.get('volume_total', 0))}

═══════════════════════════════════════════════════════════════════════════
"""


def generate_simple_event_summary(event_data):
    """
    Gera um sumário simples e formatado de um evento.
    Útil para logs e notificações rápidas.
    
    Args:
        event_data: Dicionário com dados do evento
        
    Returns:
        String formatada com sumário do evento
    """
    try:
        tipo = event_data.get('tipo_evento', 'EVENTO')
        resultado = event_data.get('resultado_da_batalha', 'N/A')
        timestamp = event_data.get('timestamp', 'N/A')
        
        # Formatação dos valores principais
        price_fmt = format_price(event_data.get('preco_fechamento', 0))
        delta_fmt = format_delta(event_data.get('delta', 0))
        volume_fmt = format_large_number(event_data.get('volume_total', 0))
        
        summary = f"""
{'='*60}
🎯 {tipo}: {resultado}
⏰ {timestamp}
💰 Preço: ${price_fmt}
📊 Delta: {delta_fmt}
📦 Volume: {volume_fmt}
{'='*60}
"""
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Erro ao gerar sumário do evento: {e}")
        return f"Erro ao gerar sumário: {str(e)}"


def generate_market_context_report(market_context, historical_vp):
    """
    Gera relatório do contexto de mercado com formatação adequada.
    
    Args:
        market_context: Dicionário com contexto de mercado
        historical_vp: Dicionário com dados de Volume Profile
        
    Returns:
        String formatada com relatório de contexto
    """
    try:
        lines = []
        lines.append("="*80)
        lines.append("📊 CONTEXTO DE MERCADO")
        lines.append("="*80)
        
        # Volume Profile
        if historical_vp and historical_vp.get("daily"):
            vp = historical_vp["daily"]
            lines.append("\n📈 Volume Profile Diário:")
            lines.append(f"   POC: ${format_price(vp.get('poc', 0))}")
            lines.append(f"   VAL: ${format_price(vp.get('val', 0))}")
            lines.append(f"   VAH: ${format_price(vp.get('vah', 0))}")
            
            hvns = vp.get('hvns', [])
            if hvns:
                lines.append(f"   HVNs: {len(hvns)} níveis identificados")
                # Mostra os 3 primeiros HVNs
                for i, hvn in enumerate(hvns[:3]):
                    lines.append(f"      - ${format_price(hvn)}")
                if len(hvns) > 3:
                    lines.append(f"      ... e mais {len(hvns) - 3} níveis")
            
            lvns = vp.get('lvns', [])
            if lvns:
                lines.append(f"   LVNs: {len(lvns)} níveis identificados")
        
        # Market Environment
        if market_context and market_context.get('market_environment'):
            env = market_context['market_environment']
            lines.append("\n🌍 Ambiente de Mercado:")
            
            if 'vix' in env:
                lines.append(f"   VIX: {format_percent(env['vix'])}")
            if 'dollar_index' in env:
                lines.append(f"   DXY: {format_price(env['dollar_index'])}")
            if 'market_breadth' in env:
                lines.append(f"   Market Breadth: {format_percent(env['market_breadth'])}")
        
        # Multi-Timeframe Trends
        if market_context and market_context.get('mtf_trends'):
            lines.append("\n📈 Tendências Multi-Timeframe:")
            for tf, trend in market_context['mtf_trends'].items():
                if isinstance(trend, dict):
                    lines.append(f"   {tf.upper()}: {trend.get('tendencia', 'N/A')}")
                else:
                    lines.append(f"   {tf.upper()}: {trend}")
        
        lines.append("="*80)
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório de contexto: {e}")
        return "Erro ao gerar relatório de contexto de mercado"


def generate_performance_report(stats_data):
    """
    Gera relatório de performance/estatísticas.
    
    Args:
        stats_data: Dicionário com estatísticas
        
    Returns:
        String formatada com relatório
    """
    try:
        lines = []
        lines.append("="*60)
        lines.append("📊 RELATÓRIO DE PERFORMANCE")
        lines.append("="*60)
        
        if 'total_signals' in stats_data:
            lines.append(f"Total de Sinais: {format_quantity(stats_data['total_signals'])}")
        
        if 'win_rate' in stats_data:
            lines.append(f"Taxa de Acerto: {format_percent(stats_data['win_rate'])}")
        
        if 'avg_profit' in stats_data:
            lines.append(f"Lucro Médio: {format_delta(stats_data['avg_profit'])}")
        
        if 'max_drawdown' in stats_data:
            lines.append(f"Drawdown Máximo: {format_percent(stats_data['max_drawdown'])}")
        
        if 'sharpe_ratio' in stats_data:
            lines.append(f"Sharpe Ratio: {format_scientific(stats_data['sharpe_ratio'], decimals=2)}")
        
        lines.append("="*60)
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório de performance: {e}")
        return "Erro ao gerar relatório de performance"