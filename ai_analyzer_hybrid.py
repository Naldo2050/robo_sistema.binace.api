import logging
from datetime import datetime

class AIAnalyzer:
    def __init__(self, headless=True, user_data_dir="./hybrid_data"):
        self.enabled = True
        self.use_advanced_analysis = True
        logging.info("🧠 IA Analyzer HÍBRIDA inicializada - Análise avançada ativada")

    def _format_contexto_macro(self, event_data: dict) -> list:
        context_lines = []
        contexto = event_data.get("contexto_macro", {})
        if not contexto or not any(contexto.values()):
            return context_lines

        context_lines.append("🌐 CONTEXTO DE MERCADO:")
        
        mtf = contexto.get("mtf_trends", {})
        if mtf:
            trends = [f"{tf.upper()}:{data.get('tendencia', 'N/A')[0]}" for tf, data in mtf.items()]
            context_lines.append(f"   → Tendência (15m,1H,4H): {', '.join(trends)}")

        derivatives = contexto.get("derivatives", {}).get(event_data.get("ativo"), {})
        if derivatives:
            funding = derivatives.get('funding_rate_percent', 0)
            ls_ratio = derivatives.get('long_short_ratio', 0)
            long_liq = derivatives.get('longs_usd', 0)
            short_liq = derivatives.get('shorts_usd', 0)
            context_lines.append(f"   → Futuros: Funding {funding:.4f}% | L/S {ls_ratio:.2f} | Liq L ${long_liq:,.0f} S ${short_liq:,.0f}")

        vp = event_data.get("historical_vp", {}).get("daily", {})
        if vp and vp.get('poc', 0) > 0:
            preco_atual = event_data.get('preco_fechamento', 0)
            poc, vah, val = vp.get('poc'), vp.get('vah'), vp.get('val')
            pos = "dentro da Value Area"
            if preco_atual > vah: pos = "acima da Value Area (Premium)"
            elif preco_atual < val: pos = "abaixo da Value Area (Discount)"
            context_lines.append(f"   → Posição no Dia: Preço {pos} | POC @ ${poc:,.0f}")
        
        context_lines.append("")
        return context_lines

    def analyze_event(self, event_data: dict) -> str:
        try:
            tipo_evento = event_data.get("tipo_evento", "N/A")
            ativo = event_data.get("ativo", "N/A")
            delta = event_data.get("delta", 0); volume = event_data.get("volume_total", 0)
            preco = event_data.get("preco_fechamento", 0); volume_compra = event_data.get("volume_compra", 0)
            volume_venda = event_data.get("volume_venda", 0); indice_absorcao = event_data.get("indice_absorcao", 0)

            contexto = event_data.get("contexto_macro", {})
            atr15m = contexto.get("mtf_trends", {}).get("15m", {}).get("atr", 0)
            if atr15m == 0:
                atr15m = preco * 0.002

            analysis = [
                f"🎯 ANÁLISE PROFISSIONAL - {tipo_evento}",
                f"💎 {ativo} | ${preco:,.2f} | Volume: {volume:,.2f}",
                "━" * 65
            ]

            context_lines = self._format_contexto_macro(event_data)
            if context_lines:
                analysis.extend(context_lines)

            force_index = self._calculate_force_index(delta, volume)
            volume_analysis = self._analyze_volume_profile(volume_compra, volume_venda, volume)
            microstructure = self._analyze_microstructure(delta, indice_absorcao, volume)

            if "Absorção" in tipo_evento:
                direction = "COMPRADORA" if delta > 0 else "VENDEDORA"
                strength = self._calculate_strength(abs(delta), volume, indice_absorcao)
                analysis.extend([
                    "📊 INTERPRETAÇÃO TÉCNICA:",
                    f"   → Absorção detectada com força {direction}",
                    f"   → Índice de absorção: {indice_absorcao:.2f} ({self._interpret_absorption_index(indice_absorcao)})",
                    f"   → Force Index: {force_index:.2f}",
                    "", "⚡ ANÁLISE DE FLUXO:",
                    f"   → Delta: {delta:,.0f} ({self._interpret_delta(delta, volume)})",
                    f"   → Ratio C/V: {volume_compra/volume_venda:.2f}" if volume_venda > 0 else "   → Ratio C/V: ∞",
                    f"   → Volume Profile: {volume_analysis}",
                    "", "🎯 FORÇA DOMINANTE:",
                    f"   → Pressão {direction} com intensidade {strength}",
                    f"   → {microstructure}",
                    "", "📈 EXPECTATIVA DE PREÇO:",
                    self._generate_price_forecast(preco, direction, atr15m),
                    "", "🚀 ESTRATÉGIA RECOMENDADA:",
                    self._generate_strategy_absorption(preco, direction, atr15m)
                ])
            elif "Exaustão" in tipo_evento:
                exhaustion_type = self._classify_exhaustion(delta, volume)
                reversal_probability = self._calculate_reversal_probability(volume, delta)
                analysis.extend([
                    "📊 INTERPRETAÇÃO TÉCNICA:",
                    f"   → Exaustão de volume detectada ({exhaustion_type})",
                    f"   → Volume: {volume:,.0f} (Elevado)",
                    f"   → Probabilidade de reversão: {reversal_probability:.1f}%",
                    "", "⚡ ANÁLISE DE MOMENTUM:",
                    f"   → Delta terminal: {delta:,.0f}",
                    f"   → {volume_analysis}",
                    f"   → Indicadores de clímax: {self._detect_climax_indicators(volume, delta)}",
                    "", "🎯 FORÇA DOMINANTE:",
                    "   → Enfraquecimento da tendência atual",
                    f"   → {microstructure}",
                    "", "📈 EXPECTATIVA DE PREÇO:",
                    "   → Consolidação ou reversão esperada",
                    f"   → Níveis críticos: {self._calculate_support_resistance(preco, atr15m)}",
                    "", "🚀 ESTRATÉGIA RECOMENDADA:",
                    self._generate_exhaustion_strategy(reversal_probability)
                ])
            elif "Liquidez" in tipo_evento:
                liquidity_impact = self._analyze_liquidity_impact(volume, delta)
                analysis.extend([
                    "📊 INTERPRETAÇÃO TÉCNICA:",
                    "   → Fluxo significativo de liquidez detectado",
                    f"   → Impacto no book: {liquidity_impact}",
                    f"   → Volume de impacto: {volume:,.0f}",
                    "", "⚡ ANÁLISE DE LIQUIDEZ:",
                    f"   → {volume_analysis}",
                    "   → Reorganização dos níveis de S/R em andamento",
                    "", "🎯 FORÇA DOMINANTE:",
                    "   → Redistribuição de liquidez ativa",
                    f"   → {microstructure}",
                    "", "📈 EXPECTATIVA DE PREÇO:",
                    "   → Volatilidade aumentada no curto prazo",
                    "   → Novos níveis de equilíbrio em formação",
                    "", "🚀 ESTRATÉGIA RECOMENDADA:",
                    self._generate_liquidity_strategy(volume)
                ])

            if event_data.get("contexto_sma"):
                analysis.append(f"📍 CONTEXTO TÉCNICO: {event_data.get('contexto_sma')}")

            analysis.extend([
                "", "⚠️  GESTÃO DE RISCO:",
                self._generate_risk_management(atr15m),
                "", f"🕐 Análise gerada: {datetime.now().strftime('%H:%M:%S')}",
                "━" * 65
            ])
            return "\n".join(analysis)
        except Exception as e:
            logging.error(f"❌ Erro na análise híbrida: {e}")
            return f"Erro na análise híbrida avançada: {str(e)}"

    def _generate_price_forecast(self, preco, direction, atr):
        if direction == "COMPRADORA":
            target = preco + (1.5 * atr)
            return f"   → Pressão para ALTA. Alvo sugerido (1.5x ATR): ${target:,.2f}"
        else:
            target = preco - (1.5 * atr)
            return f"   → Pressão para BAIXA. Alvo sugerido (1.5x ATR): ${target:,.2f}"

    def _generate_strategy_absorption(self, preco, direction, atr):
        if direction == "COMPRADORA":
            stop_loss = preco - (1.0 * atr)
            return f"   → Entrada COMPRADORA em pullbacks. Stop sugerido (1x ATR): ${stop_loss:,.2f}"
        else:
            stop_loss = preco + (1.0 * atr)
            return f"   → Entrada VENDEDORA em bounces. Stop sugerido (1x ATR): ${stop_loss:,.2f}"

    def _generate_risk_management(self, atr):
        return f"   → Volatilidade (ATR 15m): ${atr:,.2f}. Ajuste o tamanho da posição de acordo."
    
    def _calculate_support_resistance(self, preco, atr):
        support = preco - (0.5 * atr)
        resistance = preco + (0.5 * atr)
        return f"S: ${support:,.2f} | R: ${resistance:,.2f}"

    def _generate_exhaustion_strategy(self, probability):
        if probability > 70: return "   → Preparar reversão. Aguardar sinal de confirmação."
        elif probability > 50: return "   → Reduzir posições na direção atual. Monitorar."
        else: return "   → Possível pausa. Manter posições com stop ajustado."

    def _generate_liquidity_strategy(self, volume):
        if volume > 100000:
            return "   → Aguardar estabilização. Evitar entradas imediatas."
        return "   → Monitorar breakouts dos novos níveis."
        
    def _calculate_force_index(self, delta, volume):
        if volume == 0: return 0
        return (delta / volume) * 100

    def _analyze_volume_profile(self, volume_compra, volume_venda, volume_total):
        if volume_total == 0: return "Sem volume"
        buy_percentage = (volume_compra / volume_total) * 100
        sell_percentage = (volume_venda / volume_total) * 100
        if buy_percentage > 60: return f"Dominância compradora ({buy_percentage:.1f}%)"
        elif sell_percentage > 60: return f"Dominância vendedora ({sell_percentage:.1f}%)"
        return f"Equilibrado (C:{buy_percentage:.1f}% V:{sell_percentage:.1f}%)"

    def _analyze_microstructure(self, delta, indice_absorcao, volume):
        if abs(delta) > volume * 0.1: return "Microestrutura direcional forte"
        elif indice_absorcao > 2: return "Absorção significativa detectada"
        return "Microestrutura neutra"

    def _calculate_strength(self, abs_delta, volume, indice_absorcao):
        delta_ratio = abs_delta / volume if volume > 0 else 0
        if delta_ratio > 0.15 and indice_absorcao > 2: return "MUITO FORTE"
        elif delta_ratio > 0.1 or indice_absorcao > 1.5: return "FORTE"
        elif delta_ratio > 0.05: return "MODERADA"
        return "FRACA"

    def _interpret_absorption_index(self, indice):
        if indice > 3: return "Absorção extrema"
        elif indice > 2: return "Absorção forte"
        elif indice > 1: return "Absorção moderada"
        return "Absorção fraca"

    def _interpret_delta(self, delta, volume):
        if volume == 0: return "sem volume"
        ratio = abs(delta) / volume
        if ratio > 0.2: return "desequilíbrio extremo"
        elif ratio > 0.1: return "desequilíbrio significativo"
        elif ratio > 0.05: return "desequilíbrio moderado"
        return "relativamente equilibrado"

    def _classify_exhaustion(self, delta, volume):
        if volume > 50000:
            if abs(delta) < volume * 0.05: return "Exaustão por distribuição"
            else: return "Exaustão climática"
        return "Exaustão por baixo interesse"

    def _calculate_reversal_probability(self, volume, delta):
        volume_factor = min(volume / 10000, 5)
        delta_factor = abs(delta) / volume if volume > 0 else 0
        probability = (volume_factor * 10) + (delta_factor * 30) + 20
        return min(probability, 95)

    def _detect_climax_indicators(self, volume, delta):
        indicators = []
        if volume > 100000: indicators.append("Volume extremo")
        if abs(delta) > volume * 0.2: indicators.append("Delta extremo")
        if not indicators: indicators.append("Volume/Delta elevado")
        return ", ".join(indicators)
    
    def _analyze_liquidity_impact(self, volume, delta):
        impact_ratio = abs(delta) / volume if volume > 0 else 0
        if impact_ratio > 0.2: return "Alto impacto"
        elif impact_ratio > 0.1: return "Impacto moderado"
        return "Baixo impacto"

    def close(self):
        pass

    def __del__(self):
        self.close()