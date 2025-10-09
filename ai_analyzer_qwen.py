# ai_analyzer_qwen.py v2.0.1 - CORRIGIDO
"""
AI Analyzer para eventos de mercado com validação de dados.

🔹 CORREÇÕES v2.0.1:
  ✅ Corrige extração de orderbook (pega do lugar certo)
  ✅ Valida orderbook_data ANTES de formatar
  ✅ Detecta contradições corretamente (volumes vs ratio)
  ✅ Logs mais claros sobre fonte dos dados
  ✅ Fallback para múltiplos caminhos de dados
  ✅ System prompt melhorado
"""

import logging
import os
import random
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific
)

try:
    import config as app_config
except Exception:
    app_config = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from dashscope import Generation
    import dashscope
    DASHSCOPE_AVAILABLE = True
except Exception:
    DASHSCOPE_AVAILABLE = False

from time_manager import TimeManager

load_dotenv()


def _extract_dashscope_text(resp) -> str:
    """Extrai texto de respostas do DashScope."""
    try:
        output = getattr(resp, "output", None)
        if output is None and isinstance(resp, dict):
            output = resp.get("output")
        if not output:
            return ""
        choices = getattr(output, "choices", None)
        if choices is None and isinstance(output, dict):
            choices = output.get("choices")
        if not choices:
            return ""
        choice0 = choices[0]
        message = getattr(choice0, "message", None)
        if message is None and isinstance(choice0, dict):
            message = choice0.get("message")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, list) and content:
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    return str(part["text"]).strip()
        if isinstance(message, str):
            return message.strip()
        return ""
    except Exception:
        return ""


class AIAnalyzer:
    """Analisador de IA com validação robusta de dados."""
    
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self.enabled = False
        self.mode: Optional[str] = None
        self.time_manager = TimeManager()

        self.model_name = (
            getattr(app_config, "QWEN_MODEL", None)
            or os.getenv("QWEN_MODEL")
            or "qwen-plus"
        )

        self.last_test_time = 0
        self.test_interval_seconds = 120
        self.connection_failed_count = 0
        self.max_failures_before_mock = 3

        logging.info("🧠 IA Analyzer Qwen v2.0.1 inicializada - Validação robusta ativada")
        try:
            self._initialize_api()
        except Exception as e:
            logging.warning(f"Falha ao inicializar provedores de IA: {e}. Usando mock.")
            self.mode = None
            self.enabled = True

    def _initialize_api(self):
        """Inicializa provedores de IA."""
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()
                self.mode = "openai"
                self.enabled = True
                logging.info("🔧 OpenAI client configurado (modo compatível)")
                return
            except Exception as e:
                logging.warning(f"OpenAI indisponível: {e}")

        token = os.getenv("DASHSCOPE_API_KEY")
        if DASHSCOPE_AVAILABLE and token:
            try:
                dashscope.api_key = token
                self.mode = "dashscope"
                self.enabled = True
                logging.info("🔧 DashScope configurado (modo nativo)")
                return
            except Exception as e:
                logging.warning(f"DashScope indisponível: {e}")

        self.mode = None
        self.enabled = True
        logging.info("🔧 Modo MOCK ativado (sem provedores externos).")

    def _should_test_connection(self) -> bool:
        """Verifica se deve testar conexão."""
        now = time.time()
        return (now - self.last_test_time) >= self.test_interval_seconds

    def _test_connection(self) -> bool:
        """Testa conexão com IA."""
        if self.mode is None and not self.client:
            try:
                self._initialize_api()
            except Exception:
                pass

        prompt = "Ping curto. Responda com 'OK'."
        try:
            if self.mode == "openai":
                r = self.client.chat.completions.create(
                    model=self.model_name,
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
                    model=self.model_name,
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
                return True
        except Exception as e:
            self.connection_failed_count += 1
            logging.warning(f"Falha no ping da IA ({self.connection_failed_count}): {e}")
            return False

    # ====================================================================
    # 🆕 EXTRAÇÃO DE DADOS CORRIGIDA
    # ====================================================================
    
    def _extract_orderbook_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai dados de orderbook de múltiplas fontes possíveis.
        
        🔹 CORRIGIDO v2.0.1:
          - Tenta múltiplos caminhos
          - Valida dados extraídos
          - Retorna dict vazio se inválido
        
        Fontes testadas (em ordem):
          1. event_data['orderbook_data']
          2. event_data['spread_metrics']
          3. event_data['contextual_snapshot']['orderbook_data']
          4. event_data['contextual']['orderbook_data']
        """
        # Tenta múltiplos caminhos
        candidates = [
            event_data.get('orderbook_data'),
            event_data.get('spread_metrics'),
            (event_data.get('contextual_snapshot') or {}).get('orderbook_data'),
            (event_data.get('contextual') or {}).get('orderbook_data'),
        ]
        
        for i, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                continue
            
            # Valida que tem dados úteis
            has_depth = (
                candidate.get('bid_depth_usd') is not None or
                candidate.get('ask_depth_usd') is not None
            )
            
            if has_depth:
                bid_usd = float(candidate.get('bid_depth_usd', 0) or 0)
                ask_usd = float(candidate.get('ask_depth_usd', 0) or 0)
                
                # Valida que não é zero
                if bid_usd > 0 and ask_usd > 0:
                    logging.debug(f"✅ Orderbook extraído da fonte #{i}: bid=${bid_usd:,.0f}, ask=${ask_usd:,.0f}")
                    return candidate
                else:
                    logging.debug(f"⚠️ Fonte #{i} tem dados zerados (bid=${bid_usd}, ask=${ask_usd})")
        
        logging.warning("⚠️ Nenhuma fonte de orderbook válida encontrada")
        return {}

    # ====================================================================
    # PROMPT BUILDER COM VALIDAÇÃO ROBUSTA
    # ====================================================================
    
    def _create_prompt(self, event_data: Dict[str, Any]) -> str:
        """
        Cria prompt para IA com validação de dados.
        
        🔹 CORRIGIDO v2.0.1:
          - Extrai orderbook corretamente
          - Valida ANTES de formatar
          - Detecta contradições de volumes
        """
        # Campos básicos
        tipo_evento = event_data.get("tipo_evento", "N/A")
        ativo = event_data.get("ativo") or event_data.get("symbol") or "N/A"
        descricao = event_data.get("descricao", "Sem descrição.")
        
        # 🆕 EXTRAÇÃO DE ORDERBOOK CORRIGIDA
        ob_data = self._extract_orderbook_data(event_data)
        
        # Valida orderbook ANTES de usar
        bid_usd_raw = float(ob_data.get('bid_depth_usd', 0) or 0)
        ask_usd_raw = float(ob_data.get('ask_depth_usd', 0) or 0)
        is_orderbook_valid = (bid_usd_raw > 0 and ask_usd_raw > 0)
        
        if not is_orderbook_valid:
            logging.warning(
                f"⚠️ Orderbook INVÁLIDO para prompt: bid=${bid_usd_raw}, ask=${ask_usd_raw}"
            )
        
        # Delta e volumes
        delta_raw = event_data.get("delta")
        volume_total_raw = event_data.get("volume_total")
        volume_compra_raw = event_data.get("volume_compra")
        volume_venda_raw = event_data.get("volume_venda")
        
        delta = float(delta_raw) if delta_raw is not None else None
        volume_total = float(volume_total_raw) if volume_total_raw is not None else None
        volume_compra = float(volume_compra_raw) if volume_compra_raw is not None else None
        volume_venda = float(volume_venda_raw) if volume_venda_raw is not None else None
        
        # 🆕 VALIDA CONSISTÊNCIA DELTA vs VOLUMES
        if delta is not None and abs(delta) > 1.0:
            if (volume_compra == 0 and volume_venda == 0) or volume_total == 0:
                logging.warning(
                    f"⚠️ Inconsistência: delta={delta:.2f} mas volumes zerados. "
                    f"Marcando volumes como indisponíveis."
                )
                volume_compra = None
                volume_venda = None
                volume_total = None
        
        preco = (
            event_data.get("preco_atual")
            or event_data.get("preco_fechamento")
            or (event_data.get("ohlc", {}) or {}).get("close")
            or 0
        )

        # Multi TF
        multi_tf = (
            event_data.get("multi_tf")
            or event_data.get("contextual_snapshot", {}).get("multi_tf")
            or event_data.get("contextual", {}).get("multi_tf")
            or {}
        )
        multi_tf_str = (
            "\n".join(f"- {tf}: {v}" for tf, v in multi_tf.items()) if multi_tf else "Indisponível."
        )

        # Memória de eventos
        memoria = event_data.get("event_history", [])
        memoria_str = ""
        if memoria:
            mem_lines = []
            for e in memoria:
                mem_delta = format_delta(e.get('delta', 0))
                mem_vol = format_large_number(e.get('volume_total', 0))
                mem_lines.append(
                    f"   - {e.get('timestamp')} | {e.get('tipo_evento')} "
                    f"{e.get('resultado_da_batalha')} (Δ={mem_delta}, Vol={mem_vol})"
                )
            memoria_str = "\n".join(mem_lines)
        else:
            memoria_str = "   Nenhum evento recente."

        # Probabilidade histórica
        conf = event_data.get("historical_confidence", {})
        prob_long = conf.get("long_prob", "Indisponível")
        prob_short = conf.get("short_prob", "Indisponível")
        prob_neutral = conf.get("neutral_prob", "Indisponível")

        # Volume Profile
        vp = (
            event_data.get("historical_vp", {}).get("daily", {})
            or event_data.get("contextual_snapshot", {}).get("historical_vp", {}).get("daily", {})
            or {}
        )
        vp_str = ""
        if vp:
            poc_fmt = format_price(vp.get('poc', 0))
            val_fmt = format_price(vp.get('val', 0))
            vah_fmt = format_price(vp.get('vah', 0))
            vp_str = f"""
📊 Volume Profile (Diário)
   POC: ${poc_fmt} | VAL: ${val_fmt} | VAH: ${vah_fmt}
"""

        # 🆕 ORDER FLOW COM VALIDAÇÃO DE CONTRADIÇÕES
        flow = (
            event_data.get("fluxo_continuo")
            or event_data.get("flow_metrics")
            or (event_data.get("contextual_snapshot") or {}).get("flow_metrics")
            or (event_data.get("contextual") or {}).get("flow_metrics")
            or {}
        )
        
        order_flow_str = ""
        if isinstance(flow, dict) and flow:
            of = flow.get("order_flow", {})
            if isinstance(of, dict) and of:
                try:
                    buy_vol = of.get("buy_volume", 0)
                    sell_vol = of.get("sell_volume", 0)
                    bsr = of.get("buy_sell_ratio")
                    
                    has_volumes = (buy_vol > 0 or sell_vol > 0)
                    
                    # 🆕 DETECTA CONTRADIÇÃO
                    if not has_volumes and bsr is not None and bsr > 0:
                        logging.warning(
                            f"⚠️ CONTRADIÇÃO: buy/sell volumes zero mas ratio={bsr}. "
                            f"Marcando ratio como indisponível."
                        )
                        bsr = None
                    
                    flow_lines = []
                    
                    # Net flows
                    nf1 = of.get("net_flow_1m")
                    if nf1 is not None:
                        flow_lines.append(f"   Net Flow 1m: {format_delta(nf1)}")
                    
                    # Flow imbalance
                    fi = of.get("flow_imbalance")
                    if fi is not None:
                        flow_lines.append(f"   Flow Imbalance: {format_scientific(fi, 4)}")
                    
                    # Ratio (apenas se válido)
                    if bsr is not None:
                        flow_lines.append(f"   Buy/Sell Ratio: {format_scientific(bsr, 2)}")
                    
                    if flow_lines:
                        order_flow_str = "\n🚰 Fluxo de Ordens\n" + "\n".join(flow_lines) + "\n"
                
                except Exception as e:
                    logging.error(f"Erro ao processar order_flow: {e}")

        # 🆕 ML FEATURES
        ml = event_data.get("ml_features") or event_data.get("ml") or {}
        ml_str = ""
        if isinstance(ml, dict) and ml:
            try:
                mf = ml.get("microstructure", {}) or {}
                
                tick_rule = mf.get("tick_rule_sum")
                flow_imb = mf.get("flow_imbalance")
                
                ml_lines = []
                if tick_rule is not None:
                    ml_lines.append(f"   Tick Rule Sum: {format_scientific(tick_rule, 3)}")
                if flow_imb is not None:
                    ml_lines.append(f"   Flow Imbalance: {format_scientific(flow_imb, 4)}")
                
                if ml_lines:
                    ml_str = "\n📐 ML Features\n" + "\n".join(ml_lines) + "\n"
            except Exception:
                pass

        # 🆕 EVENTO ORDERBOOK COM VALIDAÇÃO
        if tipo_evento == "OrderBook" or "imbalance" in event_data:
            if not is_orderbook_valid:
                # ❌ ORDERBOOK INVÁLIDO
                ob_str = f"""
📊 Evento OrderBook - ⚠️ DADOS INDISPONÍVEIS

🔴 ATENÇÃO: Orderbook zerado ou inválido
   Bid Depth: ${bid_usd_raw:,.2f}
   Ask Depth: ${ask_usd_raw:,.2f}

⚠️ Análise de livro INDISPONÍVEL
   Use APENAS métricas de fluxo se disponíveis:
   - net_flow (delta acumulado)
   - flow_imbalance (proporção buy/sell)
   - tick_rule_sum (upticks vs downticks)
"""
            else:
                # ✅ ORDERBOOK VÁLIDO
                imbalance = ob_data.get("imbalance", 0)
                mid = ob_data.get("mid", 0)
                spread_pct = ob_data.get("spread_percent", 0)
                
                ob_str = f"""
📊 Evento OrderBook ✅

   Preço Mid: {format_price(mid)}
   Spread: {format_percent(spread_pct)}
   
   Profundidade (USD):
   • Bids: {format_large_number(bid_usd_raw)}
   • Asks: {format_large_number(ask_usd_raw)}
   
   Imbalance: {format_scientific(imbalance, 4)}
"""

            return f"""
🧠 **Análise Institucional – {ativo} | {tipo_evento}**

📝 Descrição: {descricao}
{ob_str}{ml_str}{vp_str}{order_flow_str}

📈 Multi-Timeframes
{multi_tf_str}

⏳ Memória de eventos
{memoria_str}

📉 Probabilidade Histórica
   Long={prob_long} | Short={prob_short} | Neutro={prob_neutral}

🎯 Tarefa
CRÍTICO: Se dados estiverem marcados como "Indisponível" ou "⚠️", NÃO os use.

{"🔴 ORDERBOOK INDISPONÍVEL - Use APENAS métricas de fluxo (net_flow, flow_imbalance, tick_rule)" if not is_orderbook_valid else ""}

Forneça análise institucional cobrindo:
1) Interpretação (use APENAS dados disponíveis)
2) Força dominante
3) Expectativa (curto/médio prazo)
4) Probabilidade (considere valores históricos)
5) Plano de trade SE houver dados suficientes
6) Gestão de posição (se aplicável)

Se dados críticos faltarem, seja explícito sobre limitações.
"""

        # PROMPT PADRÃO
        vol_line = (
            "Indisponível" if volume_total is None 
            else f"{format_large_number(volume_total)}"
        )
        delta_line = f"{format_delta(delta)}" if delta is not None else "Indisponível"

        return f"""
🧠 **Análise Institucional – {ativo} | {tipo_evento}**

📝 Descrição: {descricao}

   Preço: {format_price(preco)}
   Delta: {delta_line}
   Volume: {vol_line}
{ml_str}{vp_str}{order_flow_str}

📈 Multi-Timeframes
{multi_tf_str}

⏳ Memória de eventos
{memoria_str}

📉 Probabilidade Histórica
   Long={prob_long} | Short={prob_short} | Neutro={prob_neutral}

🎯 Tarefa
Use APENAS dados explicitamente fornecidos.
Se marcado como "Indisponível", NÃO use na análise.

Forneça análise institucional:
1) Interpretação
2) Força dominante
3) Expectativa
4) Probabilidade
5) Plano de trade (se dados suficientes)
"""

    # ====================================================================
    # CALLERS
    # ====================================================================
    
    def _call_openai_compatible(self, prompt: str, max_retries: int = 3) -> str:
        """Chama OpenAI API com retry."""
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é analista institucional de order flow. "
                                "REGRAS:\n"
                                "1) Use SOMENTE dados fornecidos explicitamente\n"
                                "2) Se marcado 'Indisponível' ou '⚠️', NÃO use\n"
                                "3) Orderbook zerado? Use fluxo (net_flow, tick_rule)\n"
                                "4) Contradições? Ignore dado contraditório\n"
                                "5) Seja sucinto e objetivo"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=700,
                    temperature=0.25,
                    timeout=30,
                )
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content.strip()
                    if len(content) > 10:
                        return content
                return ""
            except Exception as e:
                logging.error(f"Erro OpenAI (tentativa {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        return ""

    def _call_dashscope(self, prompt: str, max_retries: int = 3) -> str:
        """Chama DashScope API com retry."""
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = Generation.call(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é analista institucional de order flow. "
                                "REGRAS:\n"
                                "1) Use SOMENTE dados fornecidos explicitamente\n"
                                "2) Se marcado 'Indisponível' ou '⚠️', NÃO use\n"
                                "3) Orderbook zerado? Use fluxo (net_flow, tick_rule)\n"
                                "4) Contradições? Ignore dado contraditório\n"
                                "5) Seja sucinto e objetivo"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    result_format="message",
                    max_tokens=700,
                    temperature=0.25,
                    timeout=30,
                )
                content = _extract_dashscope_text(response).strip()
                if len(content) > 10:
                    return content
                return ""
            except Exception as e:
                logging.error(f"Erro DashScope (tentativa {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        return ""

    def _generate_mock_analysis(self, event_data: Dict[str, Any]) -> str:
        """Gera análise mock quando IA indisponível."""
        timestamp = self.time_manager.now_iso()
        mock_price = format_price(event_data.get('preco_fechamento', 0))
        mock_delta = format_delta(event_data.get('delta', 0))
        
        return (
            f"**Interpretação (mock):** {event_data.get('tipo_evento')} em "
            f"{event_data.get('ativo')} às {timestamp}.\n"
            f"Preço: ${mock_price} | Delta: {mock_delta}\n"
            f"**Força:** {event_data.get('resultado_da_batalha')}\n"
            f"**Expectativa:** Monitorar reação (dados limitados - modo mock)."
        )

    # ====================================================================
    # INTERFACE PÚBLICA
    # ====================================================================
    
    def analyze_event(self, event_data: Dict[str, Any]) -> str:
        """Analisa evento e retorna análise da IA."""
        if not self.enabled:
            try:
                self._initialize_api()
            except Exception:
                pass
        
        if not self.enabled:
            return self._generate_mock_analysis(event_data)

        if self._should_test_connection():
            self.last_test_time = time.time()
            if not self._test_connection():
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
        """Fecha conexão com IA."""
        self.client = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass