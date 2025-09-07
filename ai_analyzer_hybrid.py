# ai_analyzer_hybrid.py

import logging
import time
import requests
from datetime import datetime

class AIAnalyzer:
    def __init__(self, headless=True, user_data_dir="./hybrid_data"):
        self.enabled = True
        self.use_advanced_analysis = True
        logging.info("🧠 IA Analyzer HÍBRIDA inicializada - Análise avançada ativada")
    
    def analyze_event(self, event_data: dict) -> str:
        """Análise híbrida avançada baseada em algoritmos profissionais."""
        try:
            tipo_evento = event_data.get("tipo_evento", "N/A")
            ativo = event_data.get("ativo", "N/A")
            delta = event_data.get("delta", 0)
            volume = event_data.get("volume_total", 0)
            preco = event_data.get("preco_fechamento", 0)
            volume_compra = event_data.get("volume_compra", 0)
            volume_venda = event_data.get("volume_venda", 0)
            indice_absorcao = event_data.get("indice_absorcao", 0)
            
            # ANÁLISE PROFISSIONAL AVANÇADA
            analysis = []
            
            # Cabeçalho profissional
            analysis.append(f"🎯 ANÁLISE PROFISSIONAL - {tipo_evento}")
            analysis.append(f"💎 {ativo} | ${preco:.2f} | Volume: {volume:,.0f}")
            analysis.append("━" * 65)
            
            # Análise de Force Index
            force_index = self._calculate_force_index(delta, volume)
            
            # Análise de Volume Profile
            volume_analysis = self._analyze_volume_profile(volume_compra, volume_venda, volume)
            
            # Análise de Market Microstructure
            microstructure = self._analyze_microstructure(delta, indice_absorcao, volume)
            
            if "Absorção" in tipo_evento:
                direction = "COMPRADORA" if delta > 0 else "VENDEDORA"
                strength = self._calculate_strength(abs(delta), volume, indice_absorcao)
                
                analysis.extend([
                    f"📊 INTERPRETAÇÃO TÉCNICA:",
                    f"   → Absorção detectada com força {direction}",
                    f"   → Índice de absorção: {indice_absorcao:.2f} ({self._interpret_absorption_index(indice_absorcao)})",
                    f"   → Force Index: {force_index:.2f}",
                    "",
                    f"⚡ ANÁLISE DE FLUXO:",
                    f"   → Delta: {delta:,.0f} ({self._interpret_delta(delta, volume)})",
                    f"   → Ratio C/V: {volume_compra/volume_venda:.2f}" if volume_venda > 0 else "   → Ratio C/V: ∞",
                    f"   → Volume Profile: {volume_analysis}",
                    "",
                    f"🎯 FORÇA DOMINANTE:",
                    f"   → Pressão {direction} com intensidade {strength}",
                    f"   → {microstructure}",
                    "",
                    f"📈 EXPECTATIVA DE PREÇO:",
                    self._generate_price_forecast(delta, volume, preco, direction),
                    "",
                    f"🚀 ESTRATÉGIA RECOMENDADA:",
                    self._generate_strategy(delta, volume, indice_absorcao, direction)
                ])
            
            elif "Exaustão" in tipo_evento:
                exhaustion_type = self._classify_exhaustion(delta, volume)
                reversal_probability = self._calculate_reversal_probability(volume, delta)
                
                analysis.extend([
                    f"📊 INTERPRETAÇÃO TÉCNICA:",
                    f"   → Exaustão de volume detectada ({exhaustion_type})",
                    f"   → Volume: {volume:,.0f} (Elevado)",
                    f"   → Probabilidade de reversão: {reversal_probability:.1f}%",
                    "",
                    f"⚡ ANÁLISE DE MOMENTUM:",
                    f"   → Delta terminal: {delta:,.0f}",
                    f"   → {volume_analysis}",
                    f"   → Indicadores de clímax: {self._detect_climax_indicators(volume, delta)}",
                    "",
                    f"🎯 FORÇA DOMINANTE:",
                    f"   → Enfraquecimento da tendência atual",
                    f"   → {microstructure}",
                    "",
                    f"📈 EXPECTATIVA DE PREÇO:",
                    f"   → Consolidação ou reversão esperada",
                    f"   → Níveis críticos: {self._calculate_support_resistance(preco, volume)}",
                    "",
                    f"🚀 ESTRATÉGIA RECOMENDADA:",
                    self._generate_exhaustion_strategy(volume, delta, reversal_probability)
                ])
            
            elif "Liquidez" in tipo_evento:
                liquidity_impact = self._analyze_liquidity_impact(volume, delta)
                
                analysis.extend([
                    f"📊 INTERPRETAÇÃO TÉCNICA:",
                    f"   → Fluxo significativo de liquidez detectado",
                    f"   → Impacto no book: {liquidity_impact}",
                    f"   → Volume de impacto: {volume:,.0f}",
                    "",
                    f"⚡ ANÁLISE DE LIQUIDEZ:",
                    f"   → {volume_analysis}",
                    f"   → Reorganização dos níveis de S/R em andamento",
                    "",
                    f"🎯 FORÇA DOMINANTE:",
                    f"   → Redistribuição de liquidez ativa",
                    f"   → {microstructure}",
                    "",
                    f"📈 EXPECTATIVA DE PREÇO:",
                    f"   → Volatilidade aumentada no curto prazo",
                    f"   → Novos níveis de equilíbrio em formação",
                    "",
                    f"🚀 ESTRATÉGIA RECOMENDADA:",
                    self._generate_liquidity_strategy(volume, delta)
                ])
            
            # Adiciona contexto SMA se disponível
            if event_data.get("contexto_sma"):
                analysis.append(f"📍 CONTEXTO TÉCNICO: {event_data.get('contexto_sma')}")
            
            # Risk Management
            analysis.extend([
                "",
                "⚠️  GESTÃO DE RISCO:",
                self._generate_risk_management(volume, delta, preco),
                "",
                f"🕐 Análise gerada: {datetime.now().strftime('%H:%M:%S')}",
                "━" * 65
            ])
            
            return "\n".join(analysis)
            
        except Exception as e:
            logging.error(f"❌ Erro na análise híbrida: {e}")
            return f"Erro na análise híbrida avançada: {str(e)}"
    
    def _calculate_force_index(self, delta, volume):
        """Calcula Force Index personalizado."""
        if volume == 0:
            return 0
        return (delta / volume) * 100
    
    def _analyze_volume_profile(self, volume_compra, volume_venda, volume_total):
        """Analisa o perfil de volume."""
        if volume_total == 0:
            return "Sem volume"
        
        buy_percentage = (volume_compra / volume_total) * 100
        sell_percentage = (volume_venda / volume_total) * 100
        
        if buy_percentage > 60:
            return f"Dominância compradora ({buy_percentage:.1f}%)"
        elif sell_percentage > 60:
            return f"Dominância vendedora ({sell_percentage:.1f}%)"
        else:
            return f"Equilibrado (C:{buy_percentage:.1f}% V:{sell_percentage:.1f}%)"
    
    def _analyze_microstructure(self, delta, indice_absorcao, volume):
        """Analisa microestrutura do mercado."""
        if abs(delta) > volume * 0.1:
            return "Microestrutura direcional forte"
        elif indice_absorcao > 2:
            return "Absorção significativa detectada"
        else:
            return "Microestrutura neutra"
    
    def _calculate_strength(self, abs_delta, volume, indice_absorcao):
        """Calcula força do movimento."""
        delta_ratio = abs_delta / volume if volume > 0 else 0
        
        if delta_ratio > 0.15 and indice_absorcao > 2:
            return "MUITO FORTE"
        elif delta_ratio > 0.1 or indice_absorcao > 1.5:
            return "FORTE"
        elif delta_ratio > 0.05:
            return "MODERADA"
        else:
            return "FRACA"
    
    def _interpret_absorption_index(self, indice):
        """Interpreta índice de absorção."""
        if indice > 3:
            return "Absorção extrema"
        elif indice > 2:
            return "Absorção forte"
        elif indice > 1:
            return "Absorção moderada"
        else:
            return "Absorção fraca"
    
    def _interpret_delta(self, delta, volume):
        """Interpreta o delta."""
        if volume == 0:
            return "sem volume"
        ratio = abs(delta) / volume
        if ratio > 0.2:
            return "desequilíbrio extremo"
        elif ratio > 0.1:
            return "desequilíbrio significativo"
        elif ratio > 0.05:
            return "desequilíbrio moderado"
        else:
            return "relativamente equilibrado"
    
    def _generate_price_forecast(self, delta, volume, preco, direction):
        """Gera previsão de preço."""
        strength = "forte" if abs(delta) > volume * 0.1 else "moderada"
        
        if direction == "COMPRADORA":
            return f"   → Pressão {strength} para ALTA. Target: ${preco * 1.002:.2f} - ${preco * 1.005:.2f}"
        else:
            return f"   → Pressão {strength} para BAIXA. Target: ${preco * 0.998:.2f} - ${preco * 0.995:.2f}"
    
    def _generate_strategy(self, delta, volume, indice, direction):
        """Gera estratégia de trading."""
        if abs(delta) > volume * 0.15 and indice > 2:
            if direction == "COMPRADORA":
                return "   → LONG agressivo em pullbacks. Stop: 0.2%. Target: 0.5%"
            else:
                return "   → SHORT agressivo em bounces. Stop: 0.2%. Target: 0.5%"
        elif abs(delta) > volume * 0.1:
            return f"   → Entrada {direction.lower()} em rompimentos. Stop: 0.3%"
        else:
            return "   → Aguardar confirmação. Posição reduzida."
    
    def _classify_exhaustion(self, delta, volume):
        """Classifica tipo de exaustão."""
        if volume > 50000:  # Volume alto
            if abs(delta) < volume * 0.05:
                return "Exaustão por distribuição"
            else:
                return "Exaustão climática"
        else:
            return "Exaustão por baixo interesse"
    
    def _calculate_reversal_probability(self, volume, delta):
        """Calcula probabilidade de reversão."""
        volume_factor = min(volume / 10000, 5)  # Normaliza volume
        delta_factor = abs(delta) / volume if volume > 0 else 0
        
        probability = (volume_factor * 10) + (delta_factor * 30) + 20
        return min(probability, 95)
    
    def _detect_climax_indicators(self, volume, delta):
        """Detecta indicadores de clímax."""
        indicators = []
        if volume > 100000:
            indicators.append("Volume extremo")
        if abs(delta) > volume * 0.2:
            indicators.append("Delta extremo")
        if not indicators:
            indicators.append("Volume/Delta elevado")
        return ", ".join(indicators)
    
    def _calculate_support_resistance(self, preco, volume):
        """Calcula níveis de suporte/resistência."""
        volatility = 0.002 if volume > 50000 else 0.001
        support = preco * (1 - volatility)
        resistance = preco * (1 + volatility)
        return f"S: ${support:.2f} | R: ${resistance:.2f}"
    
    def _generate_exhaustion_strategy(self, volume, delta, probability):
        """Gera estratégia para exaustão."""
        if probability > 70:
            return "   → Preparar reversão. Aguardar sinal de confirmação."
        elif probability > 50:
            return "   → Reduzir posições na direção atual. Monitorar."
        else:
            return "   → Possível pausa. Manter posições com stop ajustado."
    
    def _analyze_liquidity_impact(self, volume, delta):
        """Analisa impacto na liquidez."""
        impact_ratio = abs(delta) / volume if volume > 0 else 0
        
        if impact_ratio > 0.2:
            return "Alto impacto"
        elif impact_ratio > 0.1:
            return "Impacto moderado"
        else:
            return "Baixo impacto"
    
    def _generate_liquidity_strategy(self, volume, delta):
        """Gera estratégia para eventos de liquidez."""
        if volume > 100000:
            return "   → Aguardar estabilização. Evitar entradas imediatas."
        else:
            return "   → Monitorar breakouts dos novos níveis."
    
    def _generate_risk_management(self, volume, delta, preco):
        """Gera recomendações de gestão de risco."""
        volatility = "alta" if volume > 100000 else "normal"
        stop_pct = "0.3%" if volume > 100000 else "0.2%"
        
        return f"   → Volatilidade {volatility}. Stop sugerido: {stop_pct}. Position sizing: conservador."
    
    def close(self):
        """Método para compatibilidade."""
        pass
    
    def __del__(self):
        """Destructor para compatibilidade."""
        pass