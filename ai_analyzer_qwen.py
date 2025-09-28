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
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except Exception:
    DASHSCOPE_AVAILABLE = False
    logging.warning("DashScope não instalado. Para usar API real: pip install dashscope")

from time_manager import TimeManager


def _extract_dashscope_text(resp) -> str:
    """Extrai texto de respostas do DashScope em formatos variados."""
    try:
        # resp.output pode ser objeto ou dict
        output = getattr(resp, "output", None)
        if output is None and isinstance(resp, dict):
            output = resp.get("output")

        if output is None:
            return ""

        # choices pode ser list em obj/dict
        choices = getattr(output, "choices", None)
        if choices is None and isinstance(output, dict):
            choices = output.get("choices")

        if not choices:
            return ""

        choice0 = choices[0]
        message = getattr(choice0, "message", None)
        if message is None and isinstance(choice0, dict):
            message = choice0.get("message")

        # message.content costuma ser uma lista de pedaços {'text':...}
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")

        if isinstance(content, list) and content:
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    return str(part["text"]).strip()

        # fallback: alguns formatos trazem 'message' como string
        if isinstance(message, str):
            return message.strip()

        # fallback final: alguns retornam direto em message_content
        message_content = getattr(choice0, "message_content", None)
        if message_content is None and isinstance(choice0, dict):
            message_content = choice0.get("message_content")
        if isinstance(message_content, list) and message_content:
            for part in message_content:
                if isinstance(part, dict) and part.get("text"):
                    return str(part["text"]).strip()

        return ""
    except Exception:
        return ""


class AIAnalyzer:
    def __init__(self):
        self.client = None
        self.enabled = False
        self.mode = None  # "openai" | "dashscope" | None
        self.time_manager = TimeManager()

        self.last_test_time = 0
        self.test_interval_seconds = 120
        self.connection_failed_count = 0
        self.max_failures_before_mock = 3

        logging.info("🧠 IA Analyzer Qwen inicializada - Análise avançada ativada")

        # ✅ auto-inicializa
        try:
            self._initialize_api()
        except Exception as e:
            logging.warning(f"Falha na inicialização de provedores de IA: {e}. Usando mock.")
            self.mode = None
            self.enabled = True  # mock ligado

    def _initialize_api(self):
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()  # usa OPENAI_API_KEY/OPENAI_BASE_URL se existirem
                self.mode = "openai"
                self.enabled = True
                logging.info("🔧 OpenAI client configurado (modo compatível)")
                return
            except Exception as e:
                logging.warning(f"OpenAI indisponível: {e}")

        if DASHSCOPE_AVAILABLE:
            try:
                self.mode = "dashscope"
                self.enabled = True
                logging.info("🔧 DashScope (Qwen) configurado (modo nativo)")
                return
            except Exception as e:
                logging.warning(f"DashScope indisponível: {e}")

        self.mode = None
        self.enabled = True  # mock
        logging.info("🔧 Modo MOCK ativado (sem provedores externos).")

    def _should_test_connection(self) -> bool:
        now = time.time()
        return (now - self.last_test_time) >= self.test_interval_seconds

    def _test_connection(self) -> bool:
        if self.mode is None and not self.client:
            try:
                self._initialize_api()
            except Exception:
                pass

        prompt = "Ping curto. Responda com 'OK'."
        try:
            if self.mode == "openai":
                r = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "Diagnóstico curto. Responda apenas 'OK'."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=3,
                    temperature=0.0,
                    timeout=10,
                )
                content = r.choices[0].message.content.strip().upper()
                return content.startswith("OK")

            elif self.mode == "dashscope":
                r = Generation.call(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "Diagnóstico curto. Responda apenas 'OK'."},
                        {"role": "user", "content": prompt},
                    ],
                    result_format="message",
                    max_tokens=3,
                    temperature=0.0,
                    timeout=10,
                )
                content = _extract_dashscope_text(r).upper()
                return content.startswith("OK")

            else:
                return True  # mock
        except Exception as e:
            self.connection_failed_count += 1
            logging.warning(f"Falha no ping da IA ({self.connection_failed_count}): {e}")
            return False

    def _create_prompt(self, event_data: Dict[str, Any]) -> str:
        tipo_evento = event_data.get("tipo_evento", "N/A")
        ativo = event_data.get("ativo", "N/A")
        descricao = event_data.get("descricao", "Sem descrição.")

        delta = float(event_data.get("delta") or 0)
        volume_total = float(event_data.get("volume_total") or 0)
        volume_compra = float(event_data.get("volume_compra") or 0)
        volume_venda = float(event_data.get("volume_venda") or 0)
        preco = event_data.get("preco_atual") or event_data.get("preco_fechamento") or 0

        multi_tf = event_data.get("multi_tf", {})
        multi_tf_str = "\n".join(f"- {tf}: {v}" for tf, v in multi_tf.items()) if multi_tf else "Indisponível."

        memoria = event_data.get("event_history", [])
        memoria_str = (
            "\n".join(
                [f"- {e.get('timestamp')} | {e.get('tipo_evento')} | {e.get('resultado_da_batalha')} Δ={e.get('delta')} Vol={e.get('volume_total')}" for e in memoria]
            )
            if memoria
            else "Nenhum evento recente."
        )

        conf = event_data.get("historical_confidence", {})
        prob_long = conf.get("long_prob", "Indisponível")
        prob_short = conf.get("short_prob", "Indisponível")
        prob_neutral = conf.get("neutral_prob", "Indisponível")

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

        derivativos = event_data.get("derivatives", {}).get(ativo, {}) or event_data.get("derivatives", {}).get("BTCUSDT", {})
        if derivativos:
            deriv_str = f"""
🏦 Derivativos ({ativo})
- Funding Rate: {derivativos.get('funding_rate_percent', 0):.4f}%
- OI (USD): {derivativos.get('open_interest_usd', 0):,.0f}
- Long/Short Ratio: {derivativos.get('long_short_ratio', 0):.2f}
- Liquidações (5min): Longs=${derivativos.get('longs_usd', 0):,.0f} | Shorts=${derivativos.get('shorts_usd', 0):,.0f}
"""
        else:
            deriv_str = "\n🏦 Derivativos: Dados indisponíveis no momento."

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

        if "imbalance" in event_data or tipo_evento == "OrderBook":
            imbalance = event_data.get("imbalance", "Indisponível")
            ratio = event_data.get("volume_ratio", "Indisponível")
            pressure = event_data.get("pressure", "Indisponível")
            sm = event_data.get("spread_metrics") or {}
            spread = sm.get("spread", "Indisponível")
            spread_pct = sm.get("spread_percent", "Indisponível")
            bid_usd = sm.get("bid_depth_usd", "Indisponível")
            ask_usd = sm.get("ask_depth_usd", "Indisponível")
            mi_buy = event_data.get("market_impact_buy", {}) or {}
            mi_sell = event_data.get("market_impact_sell", {}) or {}
            alertas = event_data.get("alertas_liquidez", [])

            mi_lines = ""
            try:
                if mi_buy:
                    mi_lines += f"\n- Market Impact (Buy): {mi_buy}"
                if mi_sell:
                    mi_lines += f"\n- Market Impact (Sell): {mi_sell}"
            except Exception:
                mi_lines = ""

            ob_str = f"""
📊 Evento OrderBook
- Preço: {preco}
- Imbalance: {imbalance} | Ratio: {ratio} | Pressure: {pressure}
- Spread: {spread} ({spread_pct}%)
- Profundidade (USD): Bid={bid_usd} | Ask={ask_usd}{mi_lines}
- Alertas: {", ".join(alertas) if alertas else "Nenhum"}

{'⚠️ ALERTA: Fluxo institucional detectado (iceberg recarregando) — possível absorção/defesa de nível.' if event_data.get('iceberg_reloaded') else ''}
"""

            return f"""
🧠 **Análise Institucional – {ativo} | {tipo_evento}**

📝 Descrição: {descricao}
{ob_str}
{zone_str}{deriv_str}{vp_str}

📈 Multi-Timeframes
{multi_tf_str}

⏳ Memória de eventos
{memoria_str}

📉 Probabilidade Histórica
Long={prob_long} | Short={prob_short} | Neutro={prob_neutral}

🎯 Tarefa
NÃO INVENTE números. Se um campo acima estiver 'Indisponível' ou ausente, responda explicitamente 'Indisponível' e não estime.
Forneça parecer institucional e um PLANO ancorado na zona (se houver):
1) Interpretação (order flow, liquidez, zona).
2) Força dominante.
3) Expectativa (curto/médio prazo).
4) Probabilidade mais provável (considere os valores acima).
5) Plano de trade: direção, condição de entrada (gatilho/trigger), invalidação (fora da zona), alvos 1/2 (próximas zonas), riscos.
6) Gestão de posição: sugerir sizing dinâmico baseado em:
   - Risco em % do ATR (ex: não arriscar mais que 0.5x ATR)
   - Volume da parede defendida (ex: não entrar com mais que 30% do volume da parede)
   - Volatilidade do cluster (ex: reduzir posição se price_std > X%)
"""

        vol_line = f"- Vol: {volume_total}"
        if (volume_compra > 0) or (volume_venda > 0):
            vol_line += f" (Buy={volume_compra} | Sell={volume_venda})"

        return f"""
🧠 **Análise Institucional – {ativo} | {tipo_evento}**

📝 Descrição: {descricao}

- Preço: {preco}
- Delta: {delta}
{vol_line}

📈 Multi-Timeframes
{multi_tf_str}

⏳ Memória de eventos
{memoria_str}

📉 Probabilidade Histórica
Long={prob_long} | Short={prob_short} | Neutro={prob_neutral}

🎯 Tarefa
NÃO INVENTE números. Se um campo acima estiver 'Indisponível' ou ausente, responda explicitamente 'Indisponível' e não estime.
Forneça parecer institucional e um PLANO ancorado na zona (se houver):
1) Interpretação (order flow, liquidez, zona).
2) Força dominante.
3) Expectativa (curto/médio prazo).
4) Probabilidade mais provável (considere os valores acima).
5) Plano de trade: direção, condição de entrada (gatilho/trigger), invalidação (fora da zona), alvos 1/2 (próximas zonas), riscos.
6) Gestão de posição: sugerir sizing dinâmico baseado em:
   - Risco em % do ATR (ex: não arriscar mais que 0.5x ATR)
   - Volume da parede defendida (ex: não entrar com mais que 30% do volume da parede)
   - Volatilidade do cluster (ex: reduzir posição se price_std > X%)
"""

    def _call_openai_compatible(self, prompt: str, max_retries: int = 3) -> str:
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "Você é um analista institucional de trading e order flow. REGRAS: 1) Use SOMENTE números e métricas explicitamente fornecidos no prompt; 2) Se um dado não for fornecido, escreva 'Indisponível' e NÃO estime; 3) Não invente bps, market impact, spread ou volumes; 4) Se livro de ofertas e fita (delta) divergirem, explique o motivo; 5) Seja sucinto e objetivo; 6) Não dê conselho financeiro."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=600,
                    temperature=0.25,
                    timeout=30,
                )
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content.strip()
                    if len(content) > 10:
                        return content
                logging.warning("Resposta OpenAI curta/indisponível.")
                return ""
            except Exception as e:
                logging.error(f"Erro na API OpenAI (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Aguardando {delay:.1f}s antes de retry...")
                    time.sleep(delay)
        return ""

    def _call_dashscope(self, prompt: str, max_retries: int = 3) -> str:
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = Generation.call(
                    model="qwen-plus",
                    messages=[
                        {"role": "system", "content": "Você é um analista institucional de trading e order flow. REGRAS: 1) Use SOMENTE números e métricas explicitamente fornecidos no prompt; 2) Se um dado não for fornecido, escreva 'Indisponível' e NÃO estime; 3) Não invente bps, market impact, spread ou volumes; 4) Se livro de ofertas e fita (delta) divergirem, explique o motivo; 5) Seja sucinto e objetivo; 6) Não dê conselho financeiro."},
                        {"role": "user", "content": prompt},
                    ],
                    result_format="message",
                    max_tokens=600,
                    temperature=0.25,
                    timeout=30,
                )
                content = _extract_dashscope_text(response).strip()
                if len(content) > 10:
                    return content
                logging.warning("Resposta DashScope curta/indisponível.")
                return ""
            except Exception as e:
                logging.error(f"Erro API DashScope (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Aguardando {delay:.1f}s antes de retry...")
                    time.sleep(delay)
        return ""

    def _generate_mock_analysis(self, event_data: Dict[str, Any]) -> str:
        timestamp = self.time_manager.now_iso()
        return f"""**Interpretação (mock):** Detecção de {event_data.get('tipo_evento')} no {event_data.get('ativo')} às {timestamp}.
**Força Dominante:** {event_data.get('resultado_da_batalha')}
**Expectativa:** Teste de continuação provável baseado em mock.
**Plano:** Short abaixo do POC, alvo no VAL. Stop no HVN."""

    def analyze_event(self, event_data: Dict[str, Any]) -> str:
        if not self.enabled:
            try:
                self._initialize_api()
            except Exception:
                pass
            if not self.enabled:
                logging.warning("IA não inicializada; retornando análise mock.")
                return self._generate_mock_analysis(event_data)

        if self._should_test_connection():
            self.last_test_time = time.time()
            if not self._test_connection():
                logging.warning("⚠️ Falha na conexão com IA. Usando modo mock temporariamente.")
                if self.connection_failed_count >= self.max_failures_before_mock:
                    return self._generate_mock_analysis(event_data)

        try:
            prompt = self._create_prompt(event_data)
        except Exception as e:
            logging.error(f"Erro ao criar prompt: {e}", exc_info=True)
            return self._generate_mock_analysis(event_data)

        try:
            if self.mode == "openai":
                analysis = self._call_openai_compatible(prompt)
            elif self.mode == "dashscope":
                analysis = self._call_dashscope(prompt)
            else:
                analysis = self._generate_mock_analysis(event_data)
        except Exception as e:
            logging.error(f"Erro na chamada de IA: {e}", exc_info=True)
            analysis = self._generate_mock_analysis(event_data)

        if not analysis:
            analysis = self._generate_mock_analysis(event_data)
        return analysis

    def close(self):
        self.client = None

    def __del__(self):
        self.close()
