import logging
from datetime import datetime

# Thresholds de sinal (se não existir no config, usa defaults)
try:
    import config
    MIN_SIGNAL_VOLUME_BTC = getattr(config, "MIN_SIGNAL_VOLUME_BTC", 1.0)
    MIN_SIGNAL_TPS = getattr(config, "MIN_SIGNAL_TPS", 2.0)
except Exception:
    MIN_SIGNAL_VOLUME_BTC = 1.0
    MIN_SIGNAL_TPS = 2.0


class AIAnalyzer:
    def __init__(self, headless=True, user_data_dir="./hybrid_data"):
        self.enabled = True
        self.use_advanced_analysis = True
        logging.info("🧠 IA Analyzer HÍBRIDA inicializada - Análise avançada ativada")

    # ----------------------------
    # FORMATADORES DE CONTEXTO
    # ----------------------------
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

            # Regime por timeframe (se existir)
            regimes = []
            for tf in ["15m", "1h", "4h"]:
                r = mtf.get(tf, {}).get("regime")
                if r:
                    regimes.append(f"{tf.upper()}:{r}")
            if regimes:
                context_lines.append(f"   → Regime: {', '.join(regimes)}")

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

    def _vp_density_note(self, hvn_near: int, lvn_near: int) -> str:
        try:
            if hvn_near is None or lvn_near is None:
                return ""
            if lvn_near >= hvn_near + 5:
                return "   → Ambiente: Mais LVNs (vazios). Propenso a deslocamentos/impulsos."
            if hvn_near >= lvn_near + 5:
                return "   → Ambiente: Mais HVNs (ímãs). Propenso a mean reversion/atrito."
            return "   → Ambiente: Misto (ímãs e vazios próximos)."
        except Exception:
            return ""

    def _format_vp_features(self, event_data: dict) -> list:
        lines = []
        vp_feat = event_data.get("vp_features", {})
        if not vp_feat:
            return lines

        preco_atual = event_data.get("preco_fechamento", None)

        dist_poc = vp_feat.get("distance_to_poc")
        nearest_hvn = vp_feat.get("nearest_hvn")
        dist_hvn = vp_feat.get("dist_to_nearest_hvn")
        nearest_lvn = vp_feat.get("nearest_lvn")
        dist_lvn = vp_feat.get("dist_to_nearest_lvn")
        hvn_near = vp_feat.get("hvns_within_0_5pct")
        lvn_near = vp_feat.get("lvns_within_0_5pct")
        in_single = vp_feat.get("in_single_print_zone", False)

        lines.append("🧲 VP (níveis críticos):")
        if dist_poc is not None:
            bias_poc = "acima" if dist_poc > 0 else "abaixo" if dist_poc < 0 else "no"
            lines.append(f"   → Preço {bias_poc} do POC por ${abs(dist_poc):,.2f}")

        if nearest_hvn is not None and (dist_hvn is not None or preco_atual is not None):
            if preco_atual is not None:
                side_hvn = "acima" if nearest_hvn > preco_atual else "abaixo" if nearest_hvn < preco_atual else "no"
                diff_hvn = abs(nearest_hvn - preco_atual)
            else:
                side_hvn = "acima" if (dist_hvn or 0) < 0 else "abaixo" if (dist_hvn or 0) > 0 else "no"
                diff_hvn = abs(dist_hvn or 0)
            lines.append(f"   → HVN mais próximo: ${nearest_hvn:,.2f} ({diff_hvn:,.2f} {side_hvn})")

        if nearest_lvn is not None and (dist_lvn is not None or preco_atual is not None):
            if preco_atual is not None:
                side_lvn = "acima" if nearest_lvn > preco_atual else "abaixo" if nearest_lvn < preco_atual else "no"
                diff_lvn = abs(nearest_lvn - preco_atual)
            else:
                side_lvn = "acima" if (dist_lvn or 0) < 0 else "abaixo" if (dist_lvn or 0) > 0 else "no"
                diff_lvn = abs(dist_lvn or 0)
            lines.append(f"   → LVN mais próximo: ${nearest_lvn:,.2f} ({diff_lvn:,.2f} {side_lvn})")

        if hvn_near is not None and lvn_near is not None:
            lines.append(f"   → Densidade (±0.5%): HVN {hvn_near} | LVN {lvn_near}")
            note = self._vp_density_note(hvn_near, lvn_near)
            if note:
                lines.append(note)

        if in_single:
            lines.append("   → Zona de single print: propensa a deslocamentos rápidos (preenchimento/escape)")

        lines.append("")
        return lines

    def _format_flow_features(self, event_data: dict) -> list:
        lines = []
        flux = event_data.get("fluxo_continuo") or event_data.get("flow_metrics") or {}
        if not flux:
            return lines

        cvd = flux.get("cvd")
        whale_delta = flux.get("whale_delta")
        bursts = flux.get("bursts", {})
        sector = flux.get("sector_flow", {})

        retail_delta = sector.get("retail", {}).get("delta")
        mid_delta = sector.get("mid", {}).get("delta")
        whale_bucket_delta = sector.get("whale", {}).get("delta")

        lines.append("👥 FLUXO (CVD/Whales/Bursts):")
        if cvd is not None:
            lines.append(f"   → CVD: {cvd:,.2f}")
        if whale_delta is not None:
            lines.append(f"   → Whales Δ (threshold): {whale_delta:,.2f}")
        if any(v is not None for v in [retail_delta, mid_delta, whale_bucket_delta]):
            lines.append(
                f"   → Buckets Δ: Retail {retail_delta if retail_delta is not None else 0:,.2f} | "
                f"Mid {mid_delta if mid_delta is not None else 0:,.2f} | "
                f"Whale {whale_bucket_delta if whale_bucket_delta is not None else 0:,.2f}"
            )

        if bursts:
            bcount = bursts.get("count", 0)
            bmax = bursts.get("max_burst_volume", 0.0)
            if bcount > 0:
                lines.append(f"   → Bursts: {bcount} rajadas (máx {bmax:.2f} BTC/200ms)")
        lines.append("")
        return lines

    def _format_orderbook_micro(self, event_data: dict) -> list:
        lines = []
        lifecycle = event_data.get("order_lifecycle")
        if not lifecycle:
            return lines
        avg = lifecycle.get("avg_order_lifetime_ms")
        spoof = lifecycle.get("spoofing_detected")
        layer = lifecycle.get("layering_detected")
        short_cnt = lifecycle.get("short_ttl_orders")
        lines.append("📘 Microestrutura do Book:")
        if avg is not None and avg > 0:
            lines.append(f"   → TTL médio: {avg:,.0f} ms | curtas: {short_cnt if short_cnt is not None else 0}")
        else:
            lines.append(f"   → TTL médio: N/A | curtas: {short_cnt if short_cnt is not None else 0}")
        if spoof is not None or layer is not None:
            lines.append(f"   → Spoofing: {bool(spoof)} | Layering: {bool(layer)}")
            if spoof or layer:
                lines.append("   → Alerta: risco maior de fake flow. Exigir confirmação (bursts/execuções reais).")
        lines.append("")
        return lines

    # ----------------------------
    # LÓGICA DE DIREÇÃO/ALINHAMENTO
    # ----------------------------
    def _get_absorption_direction(self, event_data: dict, delta: float) -> str:
        result = (event_data.get("resultado_da_batalha") or "").lower()
        if "absorção de compra" in result:
            return "COMPRADORA"
        if "absorção de venda" in result:
            return "VENDEDORA"
        return "COMPRADORA" if delta > 0 else "VENDEDORA"

    def _mtf_alignment(self, contexto: dict, direction: str) -> str:
        try:
            mtf = contexto.get("mtf_trends", {})
            macro = []
            for tf in ["1h", "4h"]:
                t = mtf.get(tf, {}).get("tendencia")
                if t:
                    macro.append(t.lower())
            if len(macro) < 1:
                return "misto"

            want_up = (direction == "COMPRADORA")
            macro_up_votes = sum(1 for t in macro if t.startswith("a"))  # Alta
            macro_down_votes = sum(1 for t in macro if t.startswith("b"))  # Baixa

            if want_up and macro_up_votes == len(macro):
                return "a favor"
            if (not want_up) and macro_down_votes == len(macro):
                return "a favor"
            if want_up and macro_down_votes == len(macro):
                return "contra"
            if (not want_up) and macro_up_votes == len(macro):
                return "contra"
            return "misto"
        except Exception:
            return "misto"

    def _regime_flags(self, contexto: dict) -> list:
        flags = []
        mtf = contexto.get("mtf_trends", {})
        reg15 = mtf.get("15m", {}).get("regime")
        reg1h = mtf.get("1h", {}).get("regime")
        reg4h = mtf.get("4h", {}).get("regime")
        regimes = [r for r in [reg15, reg1h, reg4h] if r]
        if not regimes:
            return flags
        if any(r.lower().startswith("manip") for r in regimes):
            flags.append("⚠️ Regime de possível manipulação (alta vol + baixo volume). Exigir confirmação.")
        if any(r.lower().startswith("instit") for r in regimes):
            flags.append("✅ Regime institucional detectado (alta vol + alto volume). Movimentos tendem a ser mais críveis.")
        return flags

    def _no_signal_reasons(self, event_data, volume, tipo_evento) -> list:
        reasons = []
        if tipo_evento and "teste de conexão" in tipo_evento.lower():
            return []  # ignora gating no teste
        if not event_data.get("is_signal", False):
            reasons.append("is_signal=False")
        res = (event_data.get("resultado_da_batalha") or "").lower()
        if "sem absorção" in res and "Absorção" in tipo_evento:
            reasons.append("resultado: Sem Absorção")
        tps = event_data.get("trades_per_second", 0) or 0
        if volume is not None and volume < MIN_SIGNAL_VOLUME_BTC:
            reasons.append(f"volume baixo (< {MIN_SIGNAL_VOLUME_BTC} BTC)")
        if tps < MIN_SIGNAL_TPS:
            reasons.append(f"baixa atividade (TPS < {MIN_SIGNAL_TPS})")
        return reasons

    def _compute_target_multiplier(self, alignment: str, regime_flags: list) -> float:
        # Base
        target_mul = 1.5
        # Contra-tendência reduz alvo
        if alignment == "contra":
            target_mul = 1.0
        # Manipulação -> manter curto
        if any("manip" in f.lower() for f in regime_flags):
            target_mul = min(target_mul, 1.0)
        # Institucional a favor -> ampliar alvo
        if any("instit" in f.lower() for f in regime_flags) and alignment == "a favor":
            target_mul = 1.8
        return target_mul

    def _compute_confidence(self, event_data: dict, direction: str, alignment: str, regime_flags: list) -> int:
        try:
            flux = event_data.get("fluxo_continuo") or event_data.get("flow_metrics") or {}
            bursts = flux.get("bursts", {}) or {}
            bcount = bursts.get("count", 0) or 0

            sector = flux.get("sector_flow", {}) or {}
            whale_bucket_delta = (sector.get("whale", {}) or {}).get("delta", 0) or 0

            score = 50
            if alignment == "a favor":
                score += 15
            elif alignment == "contra":
                score -= 15

            if any("instit" in f.lower() for f in regime_flags):
                score += 10
            if any("manip" in f.lower() for f in regime_flags):
                score -= 10

            if direction == "COMPRADORA":
                score += 10 if whale_bucket_delta > 0 else (-5 if whale_bucket_delta < 0 else 0)
            else:
                score += 10 if whale_bucket_delta < 0 else (-5 if whale_bucket_delta > 0 else 0)

            if bcount > 0:
                score += 8

            # VP ambiente
            vp_feat = event_data.get("vp_features", {}) or {}
            hvn_near = vp_feat.get("hvns_within_0_5pct")
            lvn_near = vp_feat.get("lvns_within_0_5pct")
            if hvn_near is not None and lvn_near is not None:
                if lvn_near >= hvn_near + 5:
                    score += 4  # vazio favorece deslocamentos
                elif hvn_near >= lvn_near + 5:
                    score -= 2  # imãs reduzem convicção de deslocamento

            # Clamps
            score = max(5, min(95, score))
            return int(round(score))
        except Exception:
            return 50

    # ----------------------------
    # PIPELINE PRINCIPAL
    # ----------------------------
    def analyze_event(self, event_data: dict) -> str:
        try:
            tipo_evento = event_data.get("tipo_evento", "N/A")
            ativo = event_data.get("ativo", "N/A")
            delta = event_data.get("delta", 0)
            volume = event_data.get("volume_total", 0)
            preco = event_data.get("preco_fechamento", 0)
            volume_compra = event_data.get("volume_compra", 0)
            volume_venda = event_data.get("volume_venda", 0)
            indice_absorcao = event_data.get("indice_absorcao", 0)

            # Atalho: teste de conexão → resposta curta sem gating
            if tipo_evento and "teste de conexão" in tipo_evento.lower():
                return "\n".join([
                    f"🎯 ANÁLISE PROFISSIONAL - {tipo_evento}",
                    f"💎 {ativo} | ${preco:,.2f} | Volume: {volume:,.2f}",
                    "━" * 65,
                    "✅ Módulo de IA operacional.",
                    f"🕐 Análise gerada: {datetime.now().strftime('%H:%M:%S')}",
                    "━" * 65
                ])

            contexto = event_data.get("contexto_macro", {})
            atr15m = contexto.get("mtf_trends", {}).get("15m", {}).get("atr", 0)
            if atr15m == 0:
                atr15m = preco * 0.002

            analysis = [
                f"🎯 ANÁLISE PROFISSIONAL - {tipo_evento}",
                f"💎 {ativo} | ${preco:,.2f} | Volume: {volume:,.2f}",
                "━" * 65
            ]

            # Contexto Macro
            context_lines = self._format_contexto_macro(event_data)
            if context_lines:
                analysis.extend(context_lines)

            # Métricas base
            force_index = self._calculate_force_index(delta, volume)
            volume_analysis = self._analyze_volume_profile(volume_compra, volume_venda, volume)
            microstructure = self._analyze_microstructure(delta, indice_absorcao, volume)

            # VP & Flow Features
            vp_lines = self._format_vp_features(event_data)
            flow_lines = self._format_flow_features(event_data)
            ob_lines = self._format_orderbook_micro(event_data)

            # Regime/Alinhamento
            direction_for_alignment = self._get_absorption_direction(event_data, delta)
            alignment = self._mtf_alignment(contexto, direction_for_alignment)
            regime_flags = self._regime_flags(contexto)

            # Gating de "sem sinal" (sem absorção, volume/tps baixos, is_signal False)
            reasons = self._no_signal_reasons(event_data, volume, tipo_evento)
            if ("Absorção" in tipo_evento and reasons) or (volume is not None and volume < MIN_SIGNAL_VOLUME_BTC) or ((event_data.get("trades_per_second", 0) or 0) < MIN_SIGNAL_TPS):
                analysis.extend([
                    "ℹ️ MODO INFORMATIVO (Sem Sinal Operacional):",
                    f"   → Motivos: {', '.join(reasons) if reasons else 'critérios de qualidade não atendidos'}",
                ])
                if vp_lines:
                    analysis.extend(vp_lines)
                analysis.extend([
                    "⚡ FLUXO/ESTRUTURA:",
                    f"   → Delta: {delta:,.0f} ({self._interpret_delta(delta, volume)})",
                    f"   → {volume_analysis}",
                ])
                if flow_lines:
                    analysis.extend(flow_lines)
                if ob_lines:
                    analysis.extend(ob_lines)
                for f in regime_flags:
                    analysis.append(f)

                # Confiança do contexto
                ctx_conf = self._compute_confidence(event_data, direction_for_alignment, alignment, regime_flags)
                analysis.append(f"📊 Confiança do contexto: {ctx_conf}/100")

                analysis.extend([
                    "", "⚠️  GESTÃO DE RISCO:",
                    self._generate_risk_management(atr15m),
                    "", f"🕐 Análise gerada: {datetime.now().strftime('%H:%M:%S')}",
                    "━" * 65
                ])
                return "\n".join(analysis)

            # Casos com sinal
            if "Absorção" in tipo_evento:
                direction = self._get_absorption_direction(event_data, delta)
                strength = self._calculate_strength(abs(delta), volume, indice_absorcao)

                analysis.extend([
                    "📊 INTERPRETAÇÃO TÉCNICA:",
                    f"   → Absorção detectada com força {direction}",
                    f"   → Índice de absorção: {indice_absorcao:.2f} ({self._interpret_absorption_index(indice_absorcao)})",
                    f"   → Force Index: {force_index:.2f}",
                    f"   → Alinhamento MTF: {alignment}",
                ])

                if vp_lines:
                    analysis.extend(vp_lines)
                analysis.extend([
                    "⚡ ANÁLISE DE FLUXO:",
                    f"   → Delta: {delta:,.0f} ({self._interpret_delta(delta, volume)})",
                    f"   → Ratio C/V: {volume_compra/volume_venda:.2f}" if volume_venda > 0 else "   → Ratio C/V: ∞",
                    f"   → Volume Profile: {volume_analysis}",
                ])
                if flow_lines:
                    analysis.extend(flow_lines)
                if ob_lines:
                    analysis.extend(ob_lines)

                # Alvo dinâmico
                target_mul = self._compute_target_multiplier(alignment, regime_flags)

                if direction == "COMPRADORA":
                    target = preco + (target_mul * atr15m)
                    price_line = f"   → Pressão para ALTA. Alvo sugerido ({target_mul:.1f}x ATR): ${target:,.2f}"
                else:
                    target = preco - (target_mul * atr15m)
                    price_line = f"   → Pressão para BAIXA. Alvo sugerido ({target_mul:.1f}x ATR): ${target:,.2f}"

                analysis.extend([
                    "🎯 FORÇA DOMINANTE:",
                    f"   → Pressão {direction} com intensidade {strength}",
                    f"   → {microstructure}",
                    "", "📈 EXPECTATIVA DE PREÇO:",
                    price_line,
                    "", "🚀 ESTRATÉGIA RECOMENDADA:",
                    self._strategy_with_context(preco, direction, atr15m, alignment, regime_flags)
                ])

                # Confiança do setup
                setup_conf = self._compute_confidence(event_data, direction, alignment, regime_flags)
                analysis.append(f"📊 Confiança do setup: {setup_conf}/100")

            elif "Exaustão" in tipo_evento:
                exhaustion_type = self._classify_exhaustion(delta, volume)
                reversal_probability = self._calculate_reversal_probability(volume, delta)
                analysis.extend([
                    "📊 INTERPRETAÇÃO TÉCNICA:",
                    f"   → Exaustão de volume detectada ({exhaustion_type})",
                    f"   → Volume: {volume:,.0f} (Elevado)",
                    f"   → Probabilidade de reversão: {reversal_probability:.1f}%",
                    f"   → Alinhamento MTF: {alignment}",
                ])
                if vp_lines:
                    analysis.extend(vp_lines)

                analysis.extend([
                    "⚡ ANÁLISE DE MOMENTUM:",
                    f"   → Delta terminal: {delta:,.0f}",
                    f"   → {volume_analysis}",
                    f"   → Indicadores de clímax: {self._detect_climax_indicators(volume, delta)}",
                ])
                if flow_lines:
                    analysis.extend(flow_lines)
                if ob_lines:
                    analysis.extend(ob_lines)

                analysis.extend([
                    "🎯 FORÇA DOMINANTE:",
                    "   → Enfraquecimento da tendência atual",
                    f"   → {microstructure}",
                    "", "📈 EXPECTATIVA DE PREÇO:",
                    "   → Consolidação ou reversão esperada",
                    f"   → Níveis críticos: {self._calculate_support_resistance(preco, atr15m)}",
                    "", "🚀 ESTRATÉGIA RECOMENDADA:",
                    self._generate_exhaustion_strategy(reversal_probability)
                ])

                # Confiança do contexto (exaustão)
                ex_conf = self._compute_confidence(event_data, "VENDEDORA" if delta < 0 else "COMPRADORA", alignment, regime_flags)
                analysis.append(f"📊 Confiança do contexto: {ex_conf}/100")

            elif "Liquidez" in tipo_evento:
                liquidity_impact = self._analyze_liquidity_impact(volume, delta)
                analysis.extend([
                    "📊 INTERPRETAÇÃO TÉCNICA:",
                    "   → Fluxo significativo de liquidez detectado",
                    f"   → Impacto no book: {liquidity_impact}",
                    f"   → Volume de impacto: {volume:,.0f}",
                ])
                if vp_lines:
                    analysis.extend(vp_lines)
                analysis.extend([
                    "⚡ ANÁLISE DE LIQUIDEZ:",
                    f"   → {volume_analysis}",
                    "   → Reorganização dos níveis de S/R em andamento",
                ])
                if flow_lines:
                    analysis.extend(flow_lines)
                if ob_lines:
                    analysis.extend(ob_lines)
                analysis.extend([
                    "🎯 FORÇA DOMINANTE:",
                    "   → Redistribuição de liquidez ativa",
                    f"   → {microstructure}",
                    "", "📈 EXPECTATIVA DE PREÇO:",
                    "   → Volatilidade aumentada no curto prazo",
                    "   → Novos níveis de equilíbrio em formação",
                    "", "🚀 ESTRATÉGIA RECOMENDADA:",
                    self._generate_liquidity_strategy(volume)
                ])

                liq_conf = self._compute_confidence(event_data, "COMPRADORA" if delta > 0 else "VENDEDORA", alignment, regime_flags)
                analysis.append(f"📊 Confiança do contexto: {liq_conf}/100")

            # Regime flags e SMA extra
            for f in regime_flags:
                analysis.append(f)
            if event_data.get("contexto_sma"):
                analysis.append(f"📍 CONTEXTO TÉCNICO: {event_data.get('contexto_sma')}")

            # Encerramento
            analysis.extend([
                "", "⚠️  GESTÃO DE RISCO:",
                self._generate_risk_management(atr15m),
                "", f"🕐 Análise gerada: {datetime.now().strftime('%H:%M:%S')}",
                "━" * 65
            ])
            return "\n".join(analysis)
        except Exception as e:
            logging.error(f"❌ Erro na análise híbrida: {e}", exc_info=True)
            return f"Erro na análise híbrida avançada: {str(e)}"

    # ----------------------------
    # ESTRATÉGIAS E UTILITÁRIOS
    # ----------------------------
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
        
    # ----------------------------
    # MÉTRICAS/INTERPRETAÇÃO
    # ----------------------------
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
        if volume <= 0:
            return "Microestrutura neutra"
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
        if volume > 0 and abs(delta) > volume * 0.2: indicators.append("Delta extremo")
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