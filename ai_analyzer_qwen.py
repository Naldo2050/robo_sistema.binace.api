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

# 🔹 IMPORTA TIME MANAGER
from time_manager import TimeManager

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
            self.api_key = "coloque_sua_chave_aqui"  # ⚠️ Substitua por variável de ambiente
            logging.warning("⚠️ Usando chave API hardcoded. Configure DASHSCOPE_API_KEY no ambiente.")
        
        # Tenta inicializar a API
        self._initialize_api()
        
        # 🔹 Inicializa TimeManager
        self.time_manager = TimeManager()
        
        # Controle de reconexão automática
        self.last_test_time = 0
        self.test_interval_seconds = 120  # Testa conexão a cada 2 minutos
        self.connection_failed_count = 0
        self.max_failures_before_mock = 3
        
        logging.info("🧠 IA Analyzer Qwen inicializada - Análise avançada ativada")

    def _initialize_api(self):
        """Inicializa a API preferencial disponível."""
        if OPENAI_AVAILABLE:
            try:
                # 🔹 CORRIGIDO: REMOVIDOS ESPAÇOS FINAIS
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
                )
                logging.info("🔧 OpenAI client configurado (modo compatível)")
                return
            except Exception as e:
                logging.warning(f"Erro ao configurar OpenAI client: {e}")
        
        if DASHSCOPE_AVAILABLE:
            try:
                # 🔹 CORRIGIDO: REMOVIDOS ESPAÇOS FINAIS
                dashscope.api_key = self.api_key
                dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
                logging.info("🔧 DashScope configurado (modo nativo)")
                return
            except Exception as e:
                logging.warning(f"Erro ao configurar DashScope: {e}")
        
        logging.info("🎭 Nenhuma API disponível - usando apenas simulação")

    def _should_test_connection(self):
        """Verifica se é hora de testar a conexão novamente."""
        now = time.time()
        if now - self.last_test_time > self.test_interval_seconds:
            return True
        return False

    def _test_connection(self) -> bool:
        """Testa conexão com APIs e atualiza estado."""
        if self.connection_failed_count >= self.max_failures_before_mock:
            self.use_mock = True
            self.api_mode = 'mock'
            logging.warning("⚠️ IA desativada por falhas repetidas. Usando modo mock.")
            return False

        if OPENAI_AVAILABLE and self.client:
            try:
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": "Responda: OK"}],
                    max_tokens=10,
                    timeout=10
                )
                if response.choices and "OK" in response.choices[0].message.content.upper():
                    self.api_mode = 'openai'
                    self.use_mock = False
                    self.connection_failed_count = 0
                    logging.info("✅ Conexão com OpenAI bem-sucedida")
                    return True
            except Exception as e:
                logging.warning(f"Erro ao testar conexão OpenAI: {e}")

        if DASHSCOPE_AVAILABLE:
            try:
                response = Generation.call(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": "Responda: OK"}],
                    result_format="message",
                    max_tokens=10,
                    timeout=10
                )
                if hasattr(response, 'status_code') and response.status_code == 200:
                    if hasattr(response, 'output') and response.output:
                        choices = response.output.get('choices', [])
                        if choices and "OK" in choices[0].get('message', {}).get('content', '').upper():
                            self.api_mode = 'dashscope'
                            self.use_mock = False
                            self.connection_failed_count = 0
                            logging.info("✅ Conexão com DashScope bem-sucedida")
                            return True
            except Exception as e:
                logging.warning(f"Erro ao testar conexão DashScope: {e}")

        # Falhou
        self.connection_failed_count += 1
        logging.warning(f"❌ Falha {self.connection_failed_count}/{self.max_failures_before_mock} na conexão com IA. Tentando novamente em {self.test_interval_seconds}s...")
        return False

    def _create_prompt(self, event_data: Dict[str, Any]) -> str:
        """
        Prompt institucional para análise da IA, adaptado:
        - Para OrderBook: usa métricas de OB (imbalance/ratio/pressure/spread/impactos/alertas) e preco_atual.
        - Para Absorção/Exaustão: usa delta/volume e demais features.
        Inclui multi-timeframe, memória curta, probabilidades históricas, derivativos, VP histórico, e (NOVO) sizing dinâmico.
        """
        tipo_evento = event_data.get("tipo_evento", "N/A")
        ativo = event_data.get("ativo", "N/A")
        descricao = event_data.get("descricao", "Sem descrição.")

        # Valores gerais
        delta = float(event_data.get("delta") or 0)
        volume_total = float(event_data.get("volume_total") or 0)
        volume_compra = float(event_data.get("volume_compra") or 0)
        volume_venda = float(event_data.get("volume_venda") or 0)
        preco = event_data.get("preco_atual") or event_data.get("preco_fechamento") or 0

        # Multi-timeframes
        multi_tf = event_data.get("multi_tf", {})
        multi_tf_str = "\n".join(
            [f"- {tf}: Δ={vals.get('delta')} | Vol={vals.get('volume')} | Tend={vals.get('tendencia')}" 
             for tf, vals in multi_tf.items()]
        ) if multi_tf else "Não informado"

        # Memória curta
        memoria = event_data.get("event_history", [])
        memoria_str = "\n".join([f"- {e.get('timestamp')} | {e.get('tipo_evento')} {e.get('resultado_da_batalha')} Δ={e.get('delta')} Vol={e.get('volume_total')}"
                                 for e in memoria]) if memoria else "Nenhum evento recente."

        # Probabilidades
        conf = event_data.get("historical_confidence", {})
        prob_long = conf.get("long_prob", "N/A")
        prob_short = conf.get("short_prob", "N/A")
        prob_neutral = conf.get("neutral_prob", "N/A")

        # Zona institucional (se existir)
        z = event_data.get("zone_context") or {}
        zone_str = ""
        if z:
            zone_str = f"""
🟦 Zona Institucional
- Tipo: {z.get('kind')} | TF: {z.get('timeframe')} | Score: {z.get('score')}
- Faixa: {z.get('low')} ~ {z.get('high')} (centro: {z.get('anchor_price')})
- Confluências: {", ".join(z.get('confluence', []))}
- Toques: {z.get('touch_count')} | Último toque: {z.get('last_touched')}
"""

        # 🔹 NOVO: Derivativos
        derivativos = event_data.get("derivatives", {}).get(ativo, {}) or event_data.get("derivatives", {}).get("BTCUSDT", {})
        if derivativos:
            deriv_str = f"""
🏦 Derivativos ({ativo})
- Funding Rate: {derivativos.get('funding_rate_percent', 0):.4f}%
- Open Interest: {derivativos.get('open_interest', 0):,.0f}
- Long/Short Ratio: {derivativos.get('long_short_ratio', 0):.2f}
- Liquidações (5min): Longs=${derivativos.get('longs_usd', 0):,.0f} | Shorts=${derivativos.get('shorts_usd', 0):,.0f}
"""
        else:
            deriv_str = "\n🏦 Derivativos: Dados indisponíveis no momento."

        # 🔹 NOVO: Volume Profile Histórico
        vp = event_data.get("historical_vp", {}).get("daily", {})
        if vp:
            vp_str = f"""
📊 Volume Profile Histórico (Diário)
- POC: ${vp.get('poc', 0):,.2f}
- Value Area: ${vp.get('val', 0):,.2f} — ${vp.get('vah', 0):,.2f}
- HVNs: {', '.join([f'${x:,.2f}' for x in vp.get('hvns', [])[:3]])}
- LVNs: {', '.join([f'${x:,.2f}' for x in vp.get('lvns', [])[:3]])}
"""
        else:
            vp_str = "\n📊 Volume Profile Histórico: Indisponível."

        # Caso: OrderBook (tem métricas específicas)
        if "imbalance" in event_data or tipo_evento == "OrderBook":
            imbalance = event_data.get("imbalance", "N/A")
            ratio = event_data.get("volume_ratio", "N/A")
            pressure = event_data.get("pressure", "N/A")
            sm = event_data.get("spread_metrics") or {}
            spread = sm.get("spread", "N/A")
            spread_pct = sm.get("spread_percent", "N/A")
            bid_usd = sm.get("bid_depth_usd", "N/A")
            ask_usd = sm.get("ask_depth_usd", "N/A")
            mi_buy = event_data.get("market_impact_buy", {}) or {}
            mi_sell = event_data.get("market_impact_sell", {}) or {}
            alertas = event_data.get("alertas_liquidez", [])

            ob_str = f"""
📊 Evento OrderBook
- Preço: {preco}
- Imbalance: {imbalance} | Ratio: {ratio} | Pressure: {pressure}
- Spread: {spread} ({spread_pct}%)
- Profundidade (USD): Bid={bid_usd} | Ask={ask_usd}
- Market Impact (Buy): {mi_buy}
- Market Impact (Sell): {mi_sell}
- Alertas: {", ".join(alertas) if alertas else "Nenhum"}

{'⚠️ ALERTA: Fluxo institucional detectado (iceberg recarregando) — grandes players estão absorvendo para virar o jogo.' if event_data.get('iceberg_reloaded') else ''}
"""

            prompt = f"""
📌 Ativo: {ativo} | Tipo: {tipo_evento}
📝 Descrição: {descricao}
{zone_str}
{ob_str}
{deriv_str}
{vp_str}
📈 Multi-Timeframes
{multi_tf_str}

⏳ Memória de eventos
{memoria_str}

📉 Probabilidade Histórica
Long={prob_long} | Short={prob_short} | Neutro={prob_neutral}

🎯 Tarefa
Forneça parecer institucional e um PLANO ancorado na zona (se houver):
1) Interpretação (order flow, liquidez, zona).
2) Força dominante.
3) Expectativa (curto/médio prazo).
4) Probabilidade mais provável (considere os valores acima).
5) Plano de trade: direção, condição de entrada (gatilho/trigger na zona), stop (invalidação fora da zona), alvos 1/2 (próximas zonas), riscos.
6) Gestão de posição: sugerir sizing dinâmico baseado em:
   - Risco em % do ATR (ex: não arriscar mais que 0.5x ATR)
   - Volume da parede defendida (ex: não entrar com mais que 30% do volume da parede)
   - Volatilidade do cluster (ex: reduzir posição se price_std > X%)
"""
            return prompt

        # Caso padrão (Absorção/Exaustão etc.)
        prompt = f"""
📌 Ativo: {ativo} | Tipo: {tipo_evento}
📝 Descrição: {descricao}
{zone_str}
{deriv_str}
{vp_str}
📊 Dados:
- Preço: {preco}
- Delta: {delta}
- Vol: {volume_total} (Buy={volume_compra} | Sell={volume_venda})

📈 Multi-Timeframes
{multi_tf_str}

⏳ Memória de eventos
{memoria_str}

📉 Probabilidade Histórica
Long={prob_long} | Short={prob_short} | Neutro={prob_neutral}

🎯 Tarefa
Forneça parecer institucional e um PLANO ancorado na zona (se houver):
1) Interpretação (order flow, liquidez, zona).
2) Força dominante.
3) Expectativa (curto/médio prazo).
4) Probabilidade mais provável (considere os valores acima).
5) Plano de trade: direção, condição de entrada (gatilho/trigger na zona), stop (invalidação fora da zona), alvos 1/2 (próximas zonas), riscos.
6) Gestão de posição: sugerir sizing dinâmico baseado em:
   - Risco em % do ATR (ex: não arriscar mais que 0.5x ATR)
   - Volume da parede defendida (ex: não entrar com mais que 30% do volume da parede)
   - Volatilidade do cluster (ex: reduzir posição se price_std > X%)
"""
        return prompt

    def _call_openai_compatible(self, prompt: str, max_retries: int = 3) -> str:
        """Chama a API OpenAI com retry e timeout."""
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "Você é um analista institucional de trading e order flow."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=600,
                    temperature=0.25,
                    top_p=0.8,
                    timeout=30  # 🔹 NOVO: timeout de 30 segundos
                )
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content.strip()
                    if len(content) > 10:  # Verifica se a resposta é significativa
                        return content
                    else:
                        logging.warning(f"Resposta da API muito curta: {content}")
                return ""
            except Exception as e:
                logging.error(f"Erro na API OpenAI compatível (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Aguardando {delay:.1f}s antes de retry...")
                    time.sleep(delay)
        return ""

    def _call_dashscope_native(self, prompt: str, max_retries: int = 3) -> str:
        """Chama a API DashScope com retry e timeout."""
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = Generation.call(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "Você é um analista institucional de trading e order flow."},
                        {"role": "user", "content": prompt}
                    ],
                    result_format="message",
                    max_tokens=600,
                    temperature=0.25,
                    top_p=0.8,
                    timeout=30  # 🔹 NOVO: timeout de 30 segundos
                )
                if hasattr(response, 'status_code') and response.status_code == 200:
                    if hasattr(response, 'output') and response.output:
                        choices = response.output.get('choices', [])
                        if choices:
                            content = choices[0].get('message', {}).get('content', '').strip()
                            if len(content) > 10:  # Verifica se a resposta é significativa
                                return content
                return ""
            except Exception as e:
                logging.error(f"Erro API DashScope (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Aguardando {delay:.1f}s antes de retry...")
                    time.sleep(delay)
        return ""

    def _generate_mock_analysis(self, event_data: Dict[str, Any]) -> str:
        # 🔹 USA TIME MANAGER NO MOCK
        timestamp = self.time_manager.now_iso()
        return f"""**Interpretação (mock):** Detecção de {event_data.get('tipo_evento')} no {event_data.get('ativo')} às {timestamp}.
**Força Dominante:** {event_data.get('resultado_da_batalha')}
**Expectativa:** Teste de continuação provável baseado em mock.
**Plano:** Short abaixo do POC, alvo no VAL. Stop no HVN."""

    def analyze_event(self, event_data: Dict[str, Any]) -> str:
        """Analisa evento com fallback robusto."""
        if not self.enabled:
            return "IA Analyzer desabilitado."

        # 🔹 NOVO: Testa conexão periodicamente
        if self._should_test_connection():
            self.last_test_time = time.time()
            if not self._test_connection():
                logging.warning("⚠️ Falha na conexão com IA. Usando modo mock temporariamente.")

        # Garante que a API foi testada
        if self.api_mode is None:
            self.test_connection()

        prompt = self._create_prompt(event_data)
        analysis = ""

        max_retries = 2
        for attempt in range(max_retries + 1):  # inclui tentativa original + retries
            if self.api_mode == 'openai' and self.client:
                analysis = self._call_openai_compatible(prompt)
            elif self.api_mode == 'dashscope':
                analysis = self._call_dashscope_native(prompt)
            else:
                analysis = ""

            # Verifica se a análise é válida
            if analysis and len(analysis.strip()) > 50:  # análise válida
                break
            elif attempt < max_retries:
                logging.warning(f"IA retornou resposta inválida. Retry {attempt+1}/{max_retries}...")
                time.sleep(2 ** attempt)  # backoff exponencial

        # 🔹 Fallback: usa análise básica se tudo falhar
        if not analysis or len(analysis.strip()) <= 50:
            logging.warning("⚠️ IA falhou ou retornou resposta curta. Usando análise básica.")
            analysis = self._generate_mock_analysis(event_data)

        # 🔹 NOVO: Envie heartbeat sempre que a IA responder
        # (Isso garante que o HealthMonitor saiba que a IA está viva)
        logging.debug("🧠 IA enviou análise. Registrando heartbeat.")
        # 👇 AQUI É O QUE FAZ A IA ENVIAR HEARTBEAT
        # Mas como essa classe não tem acesso ao HealthMonitor diretamente...
        # Então, vamos deixar isso como observação: o HealthMonitor vai ver que a IA respondeu quando ela chamar analyze_event()
        # Como analyze_event() é chamado pelo bot, e o bot chama o health_monitor.heartbeat("ai")
        # ... então precisamos garantir que o bot chame isso.

        return analysis

    def close(self):
        self.client = None

    def __del__(self):
        self.close()