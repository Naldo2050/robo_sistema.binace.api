# ai_analyzer_qwen.py
import logging
import os
import random
import time
from typing import Any, Dict

# Tentativa de importar OpenAI (modo compatível)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI não instalado. Para usar API real: pip install openai")

# Tentativa de importar DashScope (modo nativo)
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logging.warning("DashScope não instalado. Para usar API real: pip install dashscope")

class AIAnalyzer:
    def __init__(self, headless: bool = True, user_data_dir: str = "./qwen_data"):
        self.enabled = True
        self.use_advanced_analysis = True
        self.use_mock = True  # Inicia com simulação até testar API
        self.api_mode = None  # 'openai', 'dashscope', ou 'mock'
        self.client = None
        
        # Configura chave API
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            self.api_key = "sk-718563fc96564790af405699dd0c6e85"  # Fallback
            logging.warning("⚠️ Usando chave API hardcoded. Configure DASHSCOPE_API_KEY no ambiente.")
        
        # Tenta inicializar a API
        self._initialize_api()
        
        logging.info("🧠 IA Analyzer Qwen inicializada - Análise avançada ativada")

    def _initialize_api(self):
        """Inicializa a API preferencial disponível."""
        
        # Prioridade 1: OpenAI compatível (mais estável)
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
                )
                logging.info("🔧 OpenAI client configurado (modo compatível)")
                return
            except Exception as e:
                logging.warning(f"Erro ao configurar OpenAI client: {e}")
        
        # Prioridade 2: DashScope nativo
        if DASHSCOPE_AVAILABLE:
            try:
                dashscope.api_key = self.api_key
                dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
                logging.info("🔧 DashScope configurado (modo nativo)")
                return
            except Exception as e:
                logging.warning(f"Erro ao configurar DashScope: {e}")
        
        logging.info("🎭 Nenhuma API disponível - usando apenas simulação")

    def _create_prompt(self, event_data: Dict[str, Any]) -> str:
        """Cria o prompt otimizado para análise de eventos de mercado."""
        tipo_evento = event_data.get("tipo_evento", "N/A")
        ativo = event_data.get("ativo", "N/A")
        descricao = event_data.get("descricao", "Sem descrição.")
        delta = float(event_data.get("delta", 0.0) or 0.0)
        volume_total = float(event_data.get("volume_total", 0.0) or 0.0)
        preco_fechamento = float(event_data.get("preco_fechamento", 0.0) or 0.0)
        volume_compra = float(event_data.get("volume_compra", 0.0) or 0.0)
        volume_venda = float(event_data.get("volume_venda", 0.0) or 0.0)

        contexto_extra = ""
        if event_data.get("contexto_sma"):
            contexto_extra += f"- **Contexto SMA:** {event_data.get('contexto_sma')}\n"
        if event_data.get("indice_absorcao") is not None:
            try:
                contexto_extra += f"- **Índice de Absorção:** {float(event_data.get('indice_absorcao')):.2f}\n"
            except Exception:
                contexto_extra += f"- **Índice de Absorção:** {event_data.get('indice_absorcao')}\n"

        prompt = f"""Analise este evento de mercado como um especialista em order flow:

**DADOS DO EVENTO:**
- Ativo: {ativo}
- Tipo: {tipo_evento}
- Descrição: {descricao}
- Delta: {delta:.2f}
- Volume Total: {volume_total:.0f}
- Volume Compra: {volume_compra:.0f}
- Volume Venda: {volume_venda:.0f}
- Preço Fechamento: ${preco_fechamento:.2f}
{contexto_extra}

**ANÁLISE SOLICITADA:**
Forneça uma análise concisa (máximo 150 palavras) respondendo:

1. **Interpretação:** O que este evento revela sobre o fluxo de ordens?
2. **Força Dominante:** Compradores ou vendedores estão no controle?
3. **Expectativa:** Qual movimento é mais provável no curto prazo?
4. **Ação:** Recomendação prática para este cenário.

Seja direto e objetivo."""
        return prompt

    def _call_openai_compatible(self, prompt: str) -> str:
        """Chama a API no modo OpenAI compatível."""
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "Você é um analista especialista em order flow e trading algorítmico."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.3,
                top_p=0.8
            )
            
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            return ""
            
        except Exception as e:
            logging.error(f"Erro na API OpenAI compatível: {e}")
            return ""

    def _call_dashscope_native(self, prompt: str) -> str:
        """Chama a API no modo DashScope nativo."""
        try:
            response = Generation.call(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "Você é um analista especialista em order flow e trading algorítmico."},
                    {"role": "user", "content": prompt}
                ],
                result_format="message",
                max_tokens=512,
                temperature=0.3,
                top_p=0.8
            )
            
            if hasattr(response, 'status_code') and response.status_code == 200:
                if hasattr(response, 'output') and response.output:
                    choices = response.output.get('choices', [])
                    if choices and len(choices) > 0:
                        message = choices[0].get('message', {})
                        return message.get('content', '').strip()
            
            logging.error(f"Erro DashScope: status={getattr(response, 'status_code', 'N/A')}")
            return ""
            
        except Exception as e:
            logging.error(f"Erro na API DashScope: {e}")
            return ""

    def _generate_mock_analysis(self, event_data: Dict[str, Any]) -> str:
        """Gera uma análise simulada baseada nos dados do evento."""
        tipo_evento = event_data.get("tipo_evento", "N/A")
        ativo = event_data.get("ativo", "BTCUSDT")
        delta = float(event_data.get("delta", 0.0) or 0.0)
        volume_compra = float(event_data.get("volume_compra", 0.0) or 0.0)
        volume_venda = float(event_data.get("volume_venda", 0.0) or 0.0)
        volume_total = float(event_data.get("volume_total", 0.0) or 0.0)
        
        templates = {
            "Demanda Forte": [
                f"**🟢 Interpretação:** Fluxo agressivo de compras no {ativo}. Delta {delta:.1f} confirma absorção eficiente da oferta pelos compradores.\n\n**Força Dominante:** Bulls controlam com {(volume_compra/max(volume_total,1)*100):.1f}% do volume total.\n\n**Expectativa:** Momentum altista sustentado se continuar o fluxo.\n\n**Ação:** Posições compradas em pullbacks. Stop loss ajustado nos suportes.",
                
                f"**🔥 Interpretação:** Pressão compradora institucional detectada. Delta {delta:.1f} indica desequilíbrio favorável aos bulls.\n\n**Força Dominante:** Smart money acumulando posições.\n\n**Expectativa:** Rompimento de resistências próximas provável.\n\n**Ação:** Aguarde confirmação de continuidade. Risk management rigoroso."
            ],
            
            "Pressão de Venda": [
                f"**🔴 Interpretação:** Distribuição agressiva identificada. Delta negativo de {delta:.1f} revela pressão vendedora institucional.\n\n**Força Dominante:** Bears dominam o tape com {volume_venda:.0f} em volume.\n\n**Expectativa:** Teste de suportes inferiores iminente.\n\n**Ação:** Cautela em longs. Considere proteções ou posições short.",
                
                f"**⚠️ Interpretação:** Absorção fraca dos compradores. Delta {delta:.1f} sinaliza capitulação.\n\n**Força Dominante:** Vendedores controlam o order flow.\n\n**Expectativa:** Continuação baixista até estabilização.\n\n**Ação:** Aguarde sinais de reversão antes de comprar."
            ],
            
            "Alerta de Liquidez": [
                f"**⚡ Interpretação:** Desequilíbrio crítico de liquidez no {ativo}. Paredes significativas podem gerar breakout explosivo.\n\n**Força Dominante:** Assimetria na liquidez criando tensão.\n\n**Expectativa:** Movimento violento na direção do rompimento.\n\n**Ação:** Posição reduzida até confirmação de direção. Prepare-se para volatilidade.",
                
                f"**🎯 Interpretação:** Market makers posicionando grandes volumes. Concentração anômala de ordens detectada.\n\n**Força Dominante:** Institucionais preparando movimento.\n\n**Expectativa:** Breakout violento quando absorvidas as paredes.\n\n**Ação:** Evite contra-tendência. Siga a direção confirmada."
            ],
            
            "Absorção": [
                f"**📊 Interpretação:** Smart money processando {volume_total:.0f} em volume sem impacto no preço. Acumulação/distribuição ativa.\n\n**Força Dominante:** Equilíbrio temporário entre forças.\n\n**Expectativa:** Consolidação até definição clara.\n\n**Ação:** Posições neutras. Aguarde rompimento confirmado.",
                
                f"**🟡 Interpretação:** Absorção institucional no {ativo}. Delta {delta:.1f} sugere processo de acumulação.\n\n**Força Dominante:** Grandes players ativos.\n\n**Expectativa:** Movimento direcional pós-absorção.\n\n**Ação:** Monitor contínuo. Siga o smart money."
            ]
        }
        
        event_templates = templates.get(tipo_evento, templates["Absorção"])
        analysis = random.choice(event_templates)
        
        # Simula tempo de processamento
        time.sleep(random.uniform(1.0, 2.5))
        return analysis

    def test_connection(self) -> bool:
        """Testa conectividade com as APIs disponíveis."""
        
        # Teste 1: OpenAI compatível
        if OPENAI_AVAILABLE and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": "Responda: OK"}],
                    max_tokens=10
                )
                if response.choices and response.choices[0].message.content:
                    logging.info("✅ API OpenAI compatível funcionando")
                    self.api_mode = 'openai'
                    self.use_mock = False
                    return True
            except Exception as e:
                logging.debug(f"OpenAI compatível falhou: {e}")
        
        # Teste 2: DashScope nativo
        if DASHSCOPE_AVAILABLE:
            try:
                response = Generation.call(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": "Responda: OK"}],
                    result_format="message",
                    max_tokens=10
                )
                if hasattr(response, 'status_code') and response.status_code == 200:
                    logging.info("✅ API DashScope nativa funcionando")
                    self.api_mode = 'dashscope'
                    self.use_mock = False
                    return True
            except Exception as e:
                logging.debug(f"DashScope nativo falhou: {e}")
        
        # Fallback para simulação
        logging.info("⚠️ APIs indisponíveis - usando simulação")
        self.api_mode = 'mock'
        self.use_mock = True
        return False

    def analyze_event(self, event_data: Dict[str, Any]) -> str:
        """Analisa um evento de mercado."""
        if not self.enabled:
            return "Analisador de IA desabilitado."

        # Se ainda não testou, testa primeiro
        if self.api_mode is None:
            self.test_connection()

        prompt = self._create_prompt(event_data)
        logging.info(f"🤖 Analisando evento: {event_data.get('tipo_evento', 'N/A')}")

        analysis = ""
        
        # Tenta API real primeiro
        if self.api_mode == 'openai' and self.client:
            analysis = self._call_openai_compatible(prompt)
        elif self.api_mode == 'dashscope':
            analysis = self._call_dashscope_native(prompt)
        
        # Se API falhou ou não há API, usa simulação
        if not analysis or self.api_mode == 'mock':
            if self.api_mode != 'mock':
                logging.warning("🔄 API falhou, usando simulação como fallback")
            analysis = self._generate_mock_analysis(event_data)
            logging.info("🎭 Análise simulada gerada")
        else:
            logging.info("✅ Análise real da API gerada")

        return analysis

    def close(self):
        """Limpa recursos."""
        self.client = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
    print("🧪 Teste do AIAnalyzer Qwen\n")

    # Inicializa analisador
    analyzer = AIAnalyzer()
    
    print("=== TESTE DE CONECTIVIDADE ===")
    connection_ok = analyzer.test_connection()
    print(f"Modo ativo: {analyzer.api_mode}")
    print(f"Usando simulação: {analyzer.use_mock}\n")

    # Testes com diferentes tipos de eventos
    test_events = [
        {
            "tipo_evento": "Demanda Forte",
            "ativo": "BTCUSDT",
            "descricao": "Pressão compradora detectada",
            "delta": 145.8,
            "volume_total": 2850,
            "preco_fechamento": 67500.0,
            "volume_compra": 1800,
            "volume_venda": 1050,
            "contexto_sma": "acima da SMA200",
            "indice_absorcao": 0.78
        },
        {
            "tipo_evento": "Alerta de Liquidez",
            "ativo": "ETHUSDT",
            "descricao": "Parede significativa",
            "delta": -32.4,
            "volume_total": 1650,
            "preco_fechamento": 3450.0,
            "volume_compra": 650,
            "volume_venda": 1000
        }
    ]

    for i, event in enumerate(test_events, 1):
        print(f"=== TESTE {i}: {event['tipo_evento']} ===")
        result = analyzer.analyze_event(event)
        print(f"{result}\n")
        print("-" * 70)

    analyzer.close()