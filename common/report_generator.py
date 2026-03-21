# report_generator.py v2.0.1 - COM DETECÇÃO DE DISCREPÂNCIAS
"""
Gerador de relatórios com validação robusta e detecção de erros.

🔹 MELHORIAS v2.0.1:
  ✅ Detecta discrepâncias volume_total vs (buy + sell)
  ✅ Loga origem dos dados para debug
  ✅ Warnings detalhados com valores exatos
  ✅ Validação automática de consistência
  ✅ Todas as correções da v2.0.0 mantidas
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from decimal import Decimal, ROUND_HALF_UP

# 🔹 Importa utilitários de formatação
from common.format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific
)

logger = logging.getLogger(__name__)


def _decimal_round(value: float, decimals: int = 8) -> float:
    """Arredonda usando Decimal para evitar erros de float."""
    try:
        d = Decimal(str(value))
        quantize_str = '0.' + '0' * decimals
        return float(d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))
    except Exception:
        return round(value, decimals)


def _validate_volume_consistency(
    volume_total: float,
    volume_compra: float,
    volume_venda: float,
    event_type: str = "UNKNOWN",
    timestamp: str = "UNKNOWN"
) -> bool:
    """
    Valida consistência de volumes e loga discrepâncias.
    
    Args:
        volume_total: Volume total reportado
        volume_compra: Volume de compra
        volume_venda: Volume de venda
        event_type: Tipo do evento (para log)
        timestamp: Timestamp do evento (para log)
        
    Returns:
        True se consistente, False se há discrepância
    """
    try:
        # Arredondar para evitar erros de float
        vt = _decimal_round(volume_total, decimals=8)
        vc = _decimal_round(volume_compra, decimals=8)
        vv = _decimal_round(volume_venda, decimals=8)
        
        # Calcular soma esperada
        expected_total = _decimal_round(vc + vv, decimals=8)
        
        # Verificar discrepância (tolerância: 0.001 BTC)
        discrepancy = abs(vt - expected_total)
        
        if discrepancy > 0.001:
            logger.error(
                f"🔴 DISCREPÂNCIA DE VOLUME DETECTADA!\n"
                f"   Evento: {event_type}\n"
                f"   Timestamp: {timestamp}\n"
                f"   volume_total: {vt:.8f} BTC\n"
                f"   volume_compra: {vc:.8f} BTC\n"
                f"   volume_venda: {vv:.8f} BTC\n"
                f"   Soma (buy+sell): {expected_total:.8f} BTC\n"
                f"   DIFERENÇA: {discrepancy:.8f} BTC\n"
                f"   ---\n"
                f"   ⚠️ AÇÃO: Verificar origem de 'volume_total' em market_analyzer.py ou event_memory.py"
            )
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao validar consistência de volumes: {e}")
        return False


def generate_ai_analysis_report(
    event_data: Dict[str, Any], 
    ml_features: Dict[str, Any], 
    orderbook_data: Dict[str, Any], 
    historical_vp: Dict[str, Any], 
    multi_tf: Dict[str, Any]
) -> str:
    """
    Gera relatório estruturado de análise institucional.
    
    🔹 v2.0.1:
      - Detecta e loga discrepâncias de volume
      - Validação automática de consistência
      - Warnings detalhados para debug
      - Todas as validações da v2.0.0
    
    Args:
        event_data: Dados do evento
        ml_features: Features de ML
        orderbook_data: Dados do orderbook
        historical_vp: Volume Profile histórico
        multi_tf: Multi-timeframe data
        
    Returns:
        String com relatório formatado
    """
    try:
        # --- Extração segura de dados ---
        event_type = event_data.get("tipo_evento", "UNKNOWN")
        timestamp = event_data.get("timestamp", "UNKNOWN")
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

        # 🆕 VALIDAÇÃO DE CONSISTÊNCIA DE VOLUMES
        volume_is_consistent = _validate_volume_consistency(
            volume_total=volume_total,
            volume_compra=volume_compra,
            volume_venda=volume_venda,
            event_type=event_type,
            timestamp=timestamp
        )
        
        # 🆕 CORRIGIR volume_total se inconsistente
        if not volume_is_consistent and (volume_compra > 0 or volume_venda > 0):
            corrected_total = _decimal_round(volume_compra + volume_venda, decimals=8)
            logger.warning(
                f"✅ AUTO-CORREÇÃO: volume_total {volume_total:.8f} → {corrected_total:.8f} BTC"
            )
            volume_total = corrected_total

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
        proximos_hvns = hvns_acima[:6]

        # --- Interpretação com validação ---
        interpretation_lines = []

        # ========================================
        # 🆕 1) ORDER FLOW COM VALIDAÇÃO
        # ========================================
        delta_fmt = format_delta(delta)
        
        interpretation_lines.append(
            f"- **Order Flow:** Δ = {delta_fmt} ({'positivo' if delta >= 0 else 'negativo'})"
        )
        
        # 🆕 VALIDA VOLUMES ANTES DE COMPARAR
        has_valid_volumes = (volume_compra > 0 or volume_venda > 0)
        
        if has_valid_volumes:
            vol_compra_fmt = format_large_number(volume_compra)
            vol_venda_fmt = format_large_number(volume_venda)
            
            # 🆕 ADICIONA WARNING SE FOI CORRIGIDO
            consistency_note = ""
            if not volume_is_consistent:
                consistency_note = " ⚠️ (volume_total corrigido automaticamente)"
            
            # Só compara se volumes são diferentes
            if volume_compra != volume_venda:
                interpretation_lines.append(
                    f"  Buy volume ({vol_compra_fmt}) "
                    f"{'>' if volume_compra > volume_venda else '<'} "
                    f"Sell volume ({vol_venda_fmt}){consistency_note}."
                )
            else:
                interpretation_lines.append(
                    f"  Buy volume = Sell volume ({vol_compra_fmt}){consistency_note}."
                )
        else:
            # 🆕 VOLUMES INDISPONÍVEIS
            interpretation_lines.append(
                f"  ⚠️ Volumes individuais indisponíveis (buy={volume_compra}, sell={volume_venda})."
            )
        
        # 🆕 BUY/SELL RATIO COM VALIDAÇÃO
        if has_valid_volumes and buy_sell_ratio > 0:
            bs_ratio_fmt = f"{buy_sell_ratio:.2f}" if buy_sell_ratio < 10 else format_large_number(buy_sell_ratio)
            interpretation_lines.append(
                f"  Razão Buy/Sell = {bs_ratio_fmt}, corroborando pressão "
                f"{'compradora' if buy_sell_ratio > 1 else 'vendedora'}."
            )
        elif not has_valid_volumes and buy_sell_ratio > 0:
            # 🆕 CONTRADIÇÃO DETECTADA
            bs_ratio_fmt = f"{buy_sell_ratio:.2f}"
            interpretation_lines.append(
                f"  ⚠️ Razão Buy/Sell = {bs_ratio_fmt} (calculado via net flow, "
                f"volumes individuais indisponíveis)."
            )
        else:
            interpretation_lines.append(
                f"  Razão Buy/Sell: Indisponível."
            )

        # 🆕 PERCENTUAIS AGRESSIVOS (NÃO MULTIPLICAR POR 100)
        if aggressive_buy_pct > 0 or aggressive_sell_pct > 0:
            buy_pct_fmt = format_percent(aggressive_buy_pct)
            sell_pct_fmt = format_percent(aggressive_sell_pct)
            interpretation_lines.append(
                f"  Fluxo agressivo mostra {buy_pct_fmt} buy vs {sell_pct_fmt} sell."
            )

        # ========================================
        # 🆕 2) LIQUIDEZ COM VALIDAÇÃO DE ORDERBOOK
        # ========================================
        spread_metrics = orderbook_data.get("spread_metrics", {})
        bid_depth = spread_metrics.get("bid_depth_usd", 0)
        ask_depth = spread_metrics.get("ask_depth_usd", 0)
        imbalance_ob = orderbook_data.get("imbalance", 0)
        
        # 🆕 VALIDA ORDERBOOK
        is_orderbook_valid = (bid_depth > 0 and ask_depth > 0)
        
        if is_orderbook_valid:
            bid_depth_fmt = format_large_number(bid_depth)
            ask_depth_fmt = format_large_number(ask_depth)
            imbalance_fmt = format_delta(imbalance_ob)
            
            interpretation_lines.append(
                f"- **Liquidez:** Profundidade do livro: bids = ${bid_depth_fmt}, asks = ${ask_depth_fmt}. "
                f"Imbalance = {imbalance_fmt} → "
                f"{'oferta mais profunda' if imbalance_ob < 0 else 'demanda mais profunda'}."
            )
        else:
            # 🆕 ORDERBOOK INDISPONÍVEL
            interpretation_lines.append(
                f"- **Liquidez:** ⚠️ Orderbook indisponível (bids=${bid_depth}, asks=${ask_depth}). "
                f"Dados de profundidade não confiáveis."
            )

        # ========================================
        # 🆕 3) ZONA COM VALIDAÇÃO
        # ========================================
        # 🆕 VALIDA VALUE AREA
        is_value_area_valid = (val_daily > 0 and vah_daily > 0)
        
        if is_value_area_valid:
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
        else:
            # 🆕 VALUE AREA INDISPONÍVEL
            price_fmt = format_price(price_close)
            interpretation_lines.append(
                f"- **Zona:** ⚠️ ⚠️ VALUE AREA ZERADA: VAL=${val_daily}, VAH=${vah_daily}. "
                f"Preço atual: ${price_fmt}. "
                f"VERIFICAR: dynamic_volume_profile.py"
            )
        
        if proximos_hvns:
            hvns_str = ", ".join([f"${format_price(h)}" for h in proximos_hvns])
            interpretation_lines.append(f"  Próximos HVNs acima: {hvns_str}.")
        else:
            interpretation_lines.append("  Nenhum HVN estrutural próximo acima.")

        # ========================================
        # 🆕 4) MICROESTRUTURA / ML COM WARNINGS
        # ========================================
        ml_lines = []
        
        # Flow imbalance
        flow_imb_fmt = format_scientific(flow_imbalance, decimals=3)
        ml_lines.append(
            f"  - flow_imbalance = {flow_imb_fmt} → "
            f"{'desvio positivo' if flow_imbalance > 0 else 'pressão vendedora'} no fluxo."
        )
        
        # Order book slope
        ob_slope_fmt = format_scientific(order_book_slope, decimals=4)
        
        # 🆕 WARNING se orderbook válido mas slope zero
        if order_book_slope == 0 and is_orderbook_valid:
            ml_lines.append(
                f"  - order_book_slope = {ob_slope_fmt} ⚠️ ⚠️ (QUEBRADO! Orderbook tem dados mas slope=0)"
            )
            logger.warning(
                f"🔴 INCONSISTÊNCIA: order_book_slope=0 mas orderbook válido "
                f"(bids=${bid_depth}, asks=${ask_depth}). "
                f"VERIFICAR: orderbook_analyzer.py ou ml_features.py"
            )
        else:
            ml_lines.append(
                f"  - order_book_slope = {ob_slope_fmt} → "
                f"{'maior densidade ofertada' if order_book_slope < 0 else 'maior densidade demandada'}."
            )
        
        # Liquidity gradient
        liq_grad_fmt = format_scientific(liquidity_gradient, decimals=2)
        ml_lines.append(f"  - liquidity_gradient = {liq_grad_fmt}")
        
        # 🆕 Volume SMA ratio COM VALIDAÇÃO
        if volume_sma_ratio > 0:
            # Se está em escala 0-5 (ex: 1.5 = 150%), converte
            if volume_sma_ratio <= 10:
                vol_sma_display = format_percent(volume_sma_ratio * 100)
            else:
                # Já está em percentual (ex: 150%)
                vol_sma_display = format_percent(volume_sma_ratio)
            
            # 🆕 WARNING se extremo
            if volume_sma_ratio > 5.0 or (volume_sma_ratio <= 10 and volume_sma_ratio > 3.0):
                vol_sma_display += " ⚠️ (extremo!)"
            
            ml_lines.append(
                f"  - volume_sma_ratio = {vol_sma_display} → "
                f"volume {'elevado' if volume_sma_ratio > 1.5 else 'normal/baixo'} vs média."
            )
        else:
            ml_lines.append(f"  - volume_sma_ratio = Indisponível")
        
        # Momentum score
        momentum_fmt = format_scientific(momentum_score, decimals=5)
        ml_lines.append(
            f"  - momentum_score = {momentum_fmt} → "
            f"viés de {'alta' if momentum_score > 0 else 'baixa'}."
        )
        
        # 🆕 Tick rule sum COM WARNING
        tick_fmt = format_scientific(tick_rule_sum, decimals=3)
        
        # 🆕 WARNING se tick_rule zero mas delta significativo
        if tick_rule_sum == 0 and abs(delta) > 100:
            ml_lines.append(
                f"  - tick_rule_sum = {tick_fmt} ⚠️ (pode estar quebrado - delta={delta_fmt})"
            )
        else:
            ml_lines.append(
                f"  - tick_rule_sum = {tick_fmt} → "
                f"{'predomínio de trades com tick up' if tick_rule_sum > 0 else 'predomínio de trades com tick down'}."
            )

        interpretation_lines.append("- **Microestrutura/ML:**")
        interpretation_lines.extend(ml_lines)

        # ========================================
        # 5) FORÇA DOMINANTE
        # ========================================
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

        # ========================================
        # 6) EXPECTATIVA
        # ========================================
        curto_prazo = (
            "Teste das zonas de HVN acima." if delta > 0 
            else "Pressão vendedora testando suportes."
        )
        
        if is_value_area_valid and poc_daily > 0:
            poc_fmt = format_price(poc_daily)
            medio_prazo = (
                f"Sustentação acima do POC diário (${poc_fmt}) favorece viés de alta." 
                if price_close > poc_daily 
                else f"Romper POC (${poc_fmt}) com volume será necessário para reversão."
            )
        else:
            medio_prazo = "⚠️ POC indisponível - expectativa limitada sem referências estruturais."

        # ========================================
        # 7) CONTEXTO MULTI-TIMEFRAME
        # ========================================
        mtf_section = ""
        if multi_tf:
            mtf_lines = ["- **Contexto Multi-Timeframe:**"]
            for tf, data in multi_tf.items():
                if isinstance(data, dict):
                    tendencia = data.get('tendencia', 'N/A')
                    mtf_lines.append(f"  - {tf.upper()}: {tendencia}")
                else:
                    mtf_lines.append(f"  - {tf.upper()}: {data}")
            
            if len(mtf_lines) > 1:
                mtf_section = "\n".join(mtf_lines) + "\n"

        # ========================================
        # MONTAGEM FINAL
        # ========================================
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


def generate_simple_event_summary(event_data: Dict[str, Any]) -> str:
    """
    Gera sumário simples e formatado de um evento.
    
    Args:
        event_data: Dicionário com dados do evento
        
    Returns:
        String formatada com sumário
    """
    try:
        tipo = event_data.get('tipo_evento', 'EVENTO')
        resultado = event_data.get('resultado_da_batalha', 'N/A')
        timestamp = event_data.get('timestamp', 'N/A')
        
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
        logger.error(f"Erro ao gerar sumário: {e}")
        return f"Erro ao gerar sumário: {str(e)}"


def generate_market_context_report(
    market_context: Dict[str, Any], 
    historical_vp: Dict[str, Any]
) -> str:
    """
    Gera relatório do contexto de mercado.
    
    Args:
        market_context: Contexto de mercado
        historical_vp: Volume Profile histórico
        
    Returns:
        String formatada
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
        return "Erro ao gerar relatório de contexto"


def generate_performance_report(stats_data: Dict[str, Any]) -> str:
    """
    Gera relatório de performance.
    
    Args:
        stats_data: Estatísticas
        
    Returns:
        String formatada
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