# ai_analyzer_qwen.py v2.3.0 - COM SUPORTE GROQCLOUD + STRUCTURED OUTPUT
"""
AI Analyzer para eventos de mercado com validação de dados.

🔹 NOVIDADES v2.3.0:
  ✅ Structured Output focado em regiões de entrada:
     - sentiment, confidence, action
     - entry_zone, invalidation_zone, region_type
  ✅ Prompts ajustados para foco em:
     - suporte / resistência
     - absorção / exaustão
     - falta de demanda / oferta
     - pontos claros de entrada e zonas de defesa/invalidação

🔹 NOVIDADES v2.2.0:
  ✅ Suporte a Structured Output (JSON Mode) para Groq/OpenAI (se Pydantic disponível)
  ✅ Templates de prompt com Jinja2 (se disponível), com fallback para f-strings
  ✅ Cliente assíncrono AsyncOpenAI para Groq/OpenAI (usado internamente via asyncio.run)

🔹 NOVIDADES v2.1.0:
  ✅ Suporte completo ao GroqCloud (PRIORIDADE 1)
  ✅ Fallback inteligente: Groq → OpenAI → DashScope → Mock
  ✅ Validação automática de chave Groq
  ✅ Logs detalhados de qual provedor está ativo

🔹 CORREÇÕES v2.0.2 (mantidas):
  ✅ Método analyze() adicionado para compatibilidade com main.py
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
import asyncio
from typing import Any, Dict, Optional, Literal

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

# OpenAI / Groq (cliente síncrono + assíncrono)
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
    ASYNC_OPENAI_AVAILABLE = True
except Exception:
    try:
        from openai import OpenAI  # type: ignore
        OPENAI_AVAILABLE = True
    except Exception:
        OPENAI_AVAILABLE = False
    ASYNC_OPENAI_AVAILABLE = False
    AsyncOpenAI = None  # type: ignore

# DashScope
try:
    from dashscope import Generation
    import dashscope
    DASHSCOPE_AVAILABLE = True
except Exception:
    DASHSCOPE_AVAILABLE = False

# Jinja2 para templates (opcional)
try:
    from jinja2 import Environment, BaseLoader
    JINJA_AVAILABLE = True
except Exception:
    JINJA_AVAILABLE = False
    Environment = None  # type: ignore
    BaseLoader = object  # type: ignore

# Pydantic para structured output (opcional)
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except Exception:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore

from time_manager import TimeManager

load_dotenv()

# ========================
# Structured Output Schema
# ========================

if PYDANTIC_AVAILABLE:
    class AITradeAnalysis(BaseModel):
        """
        Esquema estruturado para a resposta da IA.

        Focado em:
        - direção (sentiment)
        - força (confidence)
        - ação sugerida (action)
        - região de entrada / zona de defesa (entry_zone)
        - zona de invalidação (invalidation_zone)
        - tipo de região (region_type: suporte/resistência/absorção/exaustão/etc.)
        """
        sentiment: Literal["bullish", "bearish", "neutral"]
        confidence: float  # 0.0–1.0
        action: Literal["buy", "sell", "hold", "flat", "wait", "avoid"]
        rationale: str
        entry_zone: Optional[str] = None
        invalidation_zone: Optional[str] = None
        region_type: Optional[str] = None
else:
    AITradeAnalysis = None  # type: ignore


# ========================
# Jinja2 Templates
# ========================

if JINJA_AVAILABLE:
    _jinja_env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
else:
    _jinja_env = None

ORDERBOOK_TEMPLATE = """
🧠 **Análise Institucional – {{ ativo }} | {{ tipo_evento }}**

📝 Descrição: {{ descricao }}
{{ ob_str }}{{ ml_str }}{{ vp_str }}{{ order_flow_str }}

📈 Multi-Timeframes
{{ multi_tf_str }}

⏳ Memória de eventos
{{ memoria_str }}

📉 Probabilidade Histórica
   Long={{ prob_long }} | Short={{ prob_short }} | Neutro={{ prob_neutral }}

🎯 Tarefa
CRÍTICO: Se dados estiverem marcados como "Indisponível" ou "⚠️", NÃO os use.

{% if not is_orderbook_valid %}
🔴 ORDERBOOK INDISPONÍVEL - Use APENAS métricas de fluxo (net_flow, flow_imbalance, tick_rule)
{% endif %}

Foque em:
1) Identificar regiões importantes:
   - suportes/resistências relevantes
   - áreas de absorção (defesa) e exaustão (fraqueza)
   - buracos de liquidez (falta de demanda/oferta)
2) Sugerir, SE HOUVER clareza, uma região aproximada de entrada (entry_zone) e uma zona de invalidação (invalidation_zone).
3) Se o cenário não estiver claro, recomende aguardar (sem forçar trade).

Se dados críticos faltarem, seja explícito sobre limitações.
"""

DEFAULT_TEMPLATE = """
🧠 **Análise Institucional – {{ ativo }} | {{ tipo_evento }}**

📝 Descrição: {{ descricao }}

   Preço: {{ preco_fmt }}
   Delta: {{ delta_line }}
   Volume: {{ vol_line }}
{{ ml_str }}{{ vp_str }}{{ order_flow_str }}

📈 Multi-Timeframes
{{ multi_tf_str }}

⏳ Memória de eventos
{{ memoria_str }}

📉 Probabilidade Histórica
   Long={{ prob_long }} | Short={{ prob_short }} | Neutro={{ prob_neutral }}

🎯 Tarefa
Use APENAS dados explicitamente fornecidos.
Se marcado como "Indisponível", NÃO use na análise.

Foque em:
1) Força ou fraqueza do movimento (sentimento).
2) Presença de região de defesa (suporte/absorção) ou oferta (resistência/exaustão).
3) Se houver cenários claros, descreva:
   - região aproximada de entrada (entry_zone)
   - zona de invalidação (invalidation_zone)
4) Se não houver entrada clara, recomende aguardar (wait/avoid) e explique o porquê.
"""


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
    """Analisador de IA com validação robusta de dados e suporte GroqCloud + structured output."""
    
    def __init__(self):
        self.client: Optional[Any] = None        # cliente síncrono
        self.client_async: Optional[Any] = None  # cliente assíncrono (AsyncOpenAI)
        self.enabled = False
        self.mode: Optional[str] = None
        self.time_manager = TimeManager()

        # Modelo padrão (será sobrescrito se Groq estiver ativo)
        self.model_name = (
            getattr(app_config, "QWEN_MODEL", None)
            or os.getenv("QWEN_MODEL")
            or "qwen-plus"
        )

        self.last_test_time = 0
        self.test_interval_seconds = 120
        self.connection_failed_count = 0
        self.max_failures_before_mock = 3

        logging.info("🧠 IA Analyzer v2.3.0 inicializada - GroqCloud + Structured Output focado em regiões de entrada")
        try:
            self._initialize_api()
        except Exception as e:
            logging.warning(f"Falha ao inicializar provedores de IA: {e}. Usando mock.")
            self.mode = None
            self.enabled = True

    def _initialize_api(self):
        """
        Inicializa provedores de IA com ordem de prioridade.
        
        ✅ PRIORIDADE:
          1. GroqCloud (rápido e barato)
          2. OpenAI (se configurado)
          3. DashScope (fallback)
          4. Mock (última opção)
        """
        
        # ============================================================
        # 🚀 PRIORIDADE 1: GROQCLOUD
        # ============================================================
        groq_key = os.getenv("GROQ_API_KEY") or getattr(app_config, "GROQ_API_KEY", None)
        
        if OPENAI_AVAILABLE and groq_key:
            # Valida formato da chave
            if not groq_key.startswith("gsk_"):
                logging.warning(
                    f"⚠️ GROQ_API_KEY suspeita (não começa com 'gsk_'). "
                    f"Tentando mesmo assim..."
                )
            
            try:
                # Cliente síncrono OpenAI-compatível apontando para Groq
                self.client = OpenAI(
                    api_key=groq_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                # Cliente assíncrono, se disponível
                if ASYNC_OPENAI_AVAILABLE and AsyncOpenAI is not None:
                    self.client_async = AsyncOpenAI(
                        api_key=groq_key,
                        base_url="https://api.groq.com/openai/v1"
                    )
                
                # Sobrescreve model_name com modelo Groq
                self.model_name = (
                    getattr(app_config, "GROQ_MODEL", None)
                    or os.getenv("GROQ_MODEL")
                    or "llama-3.1-70b-versatile"
                )
                
                self.mode = "groq"
                self.enabled = True
                
                logging.info(
                    f"🚀 GroqCloud ATIVO | Modelo: {self.model_name} | "
                    f"Chave: {groq_key[:10]}...{groq_key[-4:]}"
                )
                return
                
            except Exception as e:
                logging.warning(f"⚠️ Groq falhou na inicialização: {e}")
        else:
            if not OPENAI_AVAILABLE:
                logging.debug("OpenAI lib não disponível (necessária para Groq)")
            if not groq_key:
                logging.debug("GROQ_API_KEY não encontrada")
        
        # ============================================================
        # PRIORIDADE 2: OPENAI (compatibilidade original)
        # ============================================================
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()  # Usa OPENAI_API_KEY
                if ASYNC_OPENAI_AVAILABLE and AsyncOpenAI is not None:
                    self.client_async = AsyncOpenAI()
                self.mode = "openai"
                self.enabled = True
                logging.info("🔧 OpenAI client configurado (modo compatível)")
                return
            except Exception as e:
                logging.warning(f"OpenAI indisponível: {e}")

        # ============================================================
        # PRIORIDADE 3: DASHSCOPE (fallback)
        # ============================================================
        token = os.getenv("DASHSCOPE_API_KEY") or getattr(app_config, "DASHSCOPE_API_KEY", None)
        
        if DASHSCOPE_AVAILABLE and token:
            try:
                dashscope.api_key = token
                self.mode = "dashscope"
                self.enabled = True
                logging.info("🔧 DashScope configurado (modo nativo - fallback)")
                return
            except Exception as e:
                logging.warning(f"DashScope indisponível: {e}")

        # ============================================================
        # FALLBACK FINAL: MOCK
        # ============================================================
        self.mode = None
        self.enabled = True
        logging.info("🔧 Modo MOCK ativado (sem provedores externos).")

    def _should_test_connection(self) -> bool:
        """Verifica se deve testar conexão."""
        now = time.time()
        return (now - self.last_test_time) >= self.test_interval_seconds

    def _test_connection(self) -> bool:
        """Testa conexão com IA (ping curto)."""
        if self.mode is None and not self.client:
            try:
                self._initialize_api()
            except Exception:
                pass

        prompt = "Ping curto. Responda com 'OK'."
        try:
            if self.mode == "openai" or self.mode == "groq":
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
                success = content.startswith("OK")
                
                if success and self.mode == "groq":
                    logging.debug("✅ Groq ping OK")
                
                return success
                
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
                return True  # Mock sempre OK
                
        except Exception as e:
            self.connection_failed_count += 1
            logging.warning(
                f"Falha no ping da IA [{self.mode}] ({self.connection_failed_count}): {e}"
            )
            return False

    # ====================================================================
    # 🆕 EXTRAÇÃO DE DADOS CORRIGIDA (mantido de v2.0.2)
    # ====================================================================
    
    def _extract_orderbook_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai dados de orderbook de múltiplas fontes possíveis.
        """
        candidates = [
            event_data.get('orderbook_data'),
            event_data.get('spread_metrics'),
            (event_data.get('contextual_snapshot') or {}).get('orderbook_data'),
            (event_data.get('contextual') or {}).get('orderbook_data'),
        ]
        
        for i, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                continue
            
            has_depth = (
                candidate.get('bid_depth_usd') is not None or
                candidate.get('ask_depth_usd') is not None
            )
            
            if has_depth:
                bid_usd = float(candidate.get('bid_depth_usd', 0) or 0)
                ask_usd = float(candidate.get('ask_depth_usd', 0) or 0)
                
                if bid_usd > 0 and ask_usd > 0:
                    logging.debug(f"✅ Orderbook extraído da fonte #{i}: bid=${bid_usd:,.0f}, ask=${ask_usd:,.0f}")
                    return candidate
                else:
                    logging.debug(f"⚠️ Fonte #{i} tem dados zerados (bid=${bid_usd}, ask=${ask_usd})")
        
        logging.warning("⚠️ Nenhuma fonte de orderbook válida encontrada")
        return {}

    # ====================================================================
    # PROMPT BUILDER COM VALIDAÇÃO ROBUSTA (usando Jinja2 se disponível)
    # ====================================================================
    
    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Renderiza template com Jinja2 se disponível, senão usa concatenação simples."""
        if template_name == "orderbook":
            tmpl_str = ORDERBOOK_TEMPLATE
        else:
            tmpl_str = DEFAULT_TEMPLATE

        if JINJA_AVAILABLE and _jinja_env is not None:
            try:
                tmpl = _jinja_env.from_string(tmpl_str)
                return tmpl.render(**context)
            except Exception as e:
                logging.error(f"Erro ao renderizar template Jinja2: {e}")

        # Fallback simples sem Jinja2
        if template_name == "orderbook":
            return (
                f"🧠 **Análise Institucional – {context['ativo']} | {context['tipo_evento']}**\n\n"
                f"📝 Descrição: {context['descricao']}\n"
                f"{context['ob_str']}{context['ml_str']}{context['vp_str']}{context['order_flow_str']}\n\n"
                f"📈 Multi-Timeframes\n{context['multi_tf_str']}\n\n"
                f"⏳ Memória de eventos\n{context['memoria_str']}\n\n"
                f"📉 Probabilidade Histórica\n"
                f"   Long={context['prob_long']} | Short={context['prob_short']} | Neutro={context['prob_neutral']}\n\n"
                "🎯 Tarefa\n"
                "CRÍTICO: Se dados estiverem marcados como \"Indisponível\" ou \"⚠️\", NÃO os use.\n\n"
                + ("🔴 ORDERBOOK INDISPONÍVEL - Use APENAS métricas de fluxo (net_flow, flow_imbalance, tick_rule)\n\n"
                   if not context.get("is_orderbook_valid", True) else "")
                + "Foque em identificar regiões importantes (suporte/resistência, absorção, exaustão, buracos de liquidez)\n"
                "e em sugerir, SE HOUVER clareza, uma região de entrada e zona de invalidação.\n"
                "Se dados críticos faltarem, seja explícito sobre limitações.\n"
            )
        else:
            return (
                f"🧠 **Análise Institucional – {context['ativo']} | {context['tipo_evento']}**\n\n"
                f"📝 Descrição: {context['descricao']}\n\n"
                f"   Preço: {context['preco_fmt']}\n"
                f"   Delta: {context['delta_line']}\n"
                f"   Volume: {context['vol_line']}\n"
                f"{context['ml_str']}{context['vp_str']}{context['order_flow_str']}\n\n"
                "📈 Multi-Timeframes\n"
                f"{context['multi_tf_str']}\n\n"
                "⏳ Memória de eventos\n"
                f"{context['memoria_str']}\n\n"
                "📉 Probabilidade Histórica\n"
                f"   Long={context['prob_long']} | Short={context['prob_short']} | Neutro={context['prob_neutral']}\n\n"
                "🎯 Tarefa\n"
                "Use APENAS dados explicitamente fornecidos.\n"
                "Se marcado como \"Indisponível\", NÃO use na análise.\n\n"
                "Foque em:\n"
                "1) Força ou fraqueza do movimento (sentimento).\n"
                "2) Presença de região de defesa (suporte/absorção) ou oferta (resistência/exaustão).\n"
                "3) Se houver cenários claros, descreva:\n"
                "   - região aproximada de entrada (entry_zone)\n"
                "   - zona de invalidação (invalidation_zone)\n"
                "4) Se não houver entrada clara, recomende aguardar (wait/avoid) e explique o porquê.\n"
            )

    def _create_prompt(self, event_data: Dict[str, Any]) -> str:
        """
        Cria prompt para IA com validação de dados.
        """
        tipo_evento = event_data.get("tipo_evento", "N/A")
        ativo = event_data.get("ativo") or event_data.get("symbol") or "N/A"
        descricao = event_data.get("descricao", "Sem descrição.")
        
        # Orderbook
        ob_data = self._extract_orderbook_data(event_data)
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
        
        # Consistência delta vs volumes
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

        # Order flow
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

        # ML FEATURES
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

        # OrderBook específico
        if tipo_evento == "OrderBook" or "imbalance" in event_data:
            if not is_orderbook_valid:
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
            context = {
                "ativo": ativo,
                "tipo_evento": tipo_evento,
                "descricao": descricao,
                "ob_str": ob_str,
                "ml_str": ml_str,
                "vp_str": vp_str,
                "order_flow_str": order_flow_str,
                "multi_tf_str": multi_tf_str,
                "memoria_str": memoria_str,
                "prob_long": prob_long,
                "prob_short": prob_short,
                "prob_neutral": prob_neutral,
                "is_orderbook_valid": is_orderbook_valid,
            }
            return self._render_template("orderbook", context)

        # Prompt padrão
        vol_line = (
            "Indisponível" if volume_total is None 
            else f"{format_large_number(volume_total)}"
        )
        delta_line = f"{format_delta(delta)}" if delta is not None else "Indisponível"
        preco_fmt = format_price(preco)

        context = {
            "ativo": ativo,
            "tipo_evento": tipo_evento,
            "descricao": descricao,
            "preco_fmt": preco_fmt,
            "delta_line": delta_line,
            "vol_line": vol_line,
            "ml_str": ml_str,
            "vp_str": vp_str,
            "order_flow_str": order_flow_str,
            "multi_tf_str": multi_tf_str,
            "memoria_str": memoria_str,
            "prob_long": prob_long,
            "prob_short": prob_short,
            "prob_neutral": prob_neutral,
        }
        return self._render_template("default", context)

    # ====================================================================
    # CALLERS (SYNC + ASYNC / STRUCTURED)
    # ====================================================================

    async def _a_call_openai_text(self, prompt: str) -> str:
        """Versão assíncrona simples (texto livre) para OpenAI/Groq."""
        if not self.client_async:
            raise RuntimeError("Cliente assíncrono não inicializado")
        try:
            response = await self.client_async.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é analista institucional de fluxo, suporte/resistência e regiões de defesa.\n"
                            "REGRAS:\n"
                            "1) Use SOMENTE dados fornecidos explicitamente.\n"
                            "2) Se marcado 'Indisponível' ou '⚠️', NÃO use.\n"
                            "3) Orderbook zerado? Use fluxo (net_flow, flow_imbalance, tick_rule).\n"
                            "4) Contradições? Ignore dado contraditório.\n"
                            "5) Foque em identificar REGIÕES IMPORTANTES:\n"
                            "   - Suportes e resistências relevantes\n"
                            "   - Regiões de absorção (defesa) e exaustão (fraqueza)\n"
                            "   - Falta de demanda/oferta (breaks, buracos de liquidez)\n"
                            "6) Só sugira ENTRADA (compra/venda) quando houver uma região CLARA e bem defendida,\n"
                            "   descrevendo preço aproximado da entrada e uma zona de invalidação.\n"
                            "7) Se o cenário não estiver claro, prefira recomendar 'aguardar' e explique o porquê.\n"
                            "8) Seja sucinto, objetivo e profissional."
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
                return content
            return ""
        except Exception as e:
            logging.error(f"Erro assíncrono {self.mode} (texto): {e}")
            return ""

    async def _a_call_openai_structured(self, prompt: str) -> tuple[str, Optional[AITradeAnalysis]]:
        """
        Versão assíncrona usando JSON Mode + Pydantic.
        Retorna (raw_json_string, objeto_validado_ou_None).
        """
        if not self.client_async:
            raise RuntimeError("Cliente assíncrono não inicializado")
        if not (PYDANTIC_AVAILABLE and AITradeAnalysis):
            text = await self._a_call_openai_text(prompt)
            return text, None
        try:
            response = await self.client_async.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é analista institucional de fluxo, suporte/resistência e regiões de defesa.\n"
                            "Responda APENAS em JSON com os campos:\n"
                            "sentiment (bullish/bearish/neutral),\n"
                            "confidence (0-1),\n"
                            "action (buy/sell/hold/flat/wait/avoid),\n"
                            "rationale (string),\n"
                            "entry_zone (string ou null),\n"
                            "invalidation_zone (string ou null),\n"
                            "region_type (string ou null, ex: 'suporte', 'resistência', 'absorção', 'exaustão').\n"
                            "Regras:\n"
                            "- Só indique entry_zone se houver uma região CLARA de entrada (defesa institucional, suporte/resistência relevante, absorção/exaustão bem formadas).\n"
                            "- Se o cenário não estiver claro, use action='wait' ou 'avoid' e entry_zone=null.\n"
                            "- Não inclua texto fora do JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.25,
                timeout=30,
                response_format={"type": "json_object"},
            )
            if not response.choices:
                return "", None
            content = response.choices[0].message.content.strip()
            if not content:
                return "", None
            try:
                if hasattr(AITradeAnalysis, "model_validate_json"):
                    obj = AITradeAnalysis.model_validate_json(content)  # type: ignore[attr-defined]
                else:
                    obj = AITradeAnalysis.parse_raw(content)           # type: ignore[attr-defined]
                return content, obj
            except Exception as e:
                logging.warning(f"Falha ao parsear JSON structured: {e}. Usando apenas texto.")
                return content, None
        except Exception as e:
            logging.error(f"Erro JSON Mode {self.mode}: {e}. Fallback texto.")
            text = await self._a_call_openai_text(prompt)
            return text, None

    def _call_openai_compatible(self, prompt: str, max_retries: int = 3) -> str:
        """
        Chama cliente OpenAI-compatível de forma síncrona (texto livre).
        Mantido para fallback e para ping.
        """
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é analista institucional de fluxo, suporte/resistência e regiões de defesa.\n"
                                "REGRAS:\n"
                                "1) Use SOMENTE dados fornecidos explicitamente.\n"
                                "2) Se marcado 'Indisponível' ou '⚠️', NÃO use.\n"
                                "3) Orderbook zerado? Use fluxo (net_flow, flow_imbalance, tick_rule).\n"
                                "4) Contradições? Ignore dado contraditório.\n"
                                "5) Foque em identificar REGIÕES IMPORTANTES:\n"
                                "   - Suportes e resistências relevantes\n"
                                "   - Regiões de absorção (defesa) e exaustão (fraqueza)\n"
                                "   - Falta de demanda/oferta (breaks, buracos de liquidez)\n"
                                "6) Só sugira ENTRADA (compra/venda) quando houver uma região CLARA e bem defendida,\n"
                                "   descrevendo preço aproximado da entrada e uma zona de invalidação.\n"
                                "7) Se o cenário não estiver claro, prefira recomendar 'aguardar' e explique o porquê.\n"
                                "8) Seja sucinto, objetivo e profissional."
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
                        if self.mode == "groq":
                            logging.debug(f"✅ Groq respondeu ({len(content)} chars)")
                        return content
                return ""
            except Exception as e:
                logging.error(
                    f"Erro {self.mode.upper()} (tentativa {attempt+1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
        return ""

    def _call_dashscope(self, prompt: str, max_retries: int = 3) -> str:
        """Chama DashScope API com retry (texto livre)."""
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = Generation.call(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é analista institucional de fluxo, suporte/resistência e regiões de defesa.\n"
                                "REGRAS:\n"
                                "1) Use SOMENTE dados fornecidos explicitamente.\n"
                                "2) Se marcado 'Indisponível' ou '⚠️', NÃO use.\n"
                                "3) Orderbook zerado? Use fluxo (net_flow, flow_imbalance, tick_rule).\n"
                                "4) Contradições? Ignore dado contraditório.\n"
                                "5) Foque em identificar REGIÕES IMPORTANTES:\n"
                                "   - Suportes e resistências relevantes\n"
                                "   - Regiões de absorção (defesa) e exaustão (fraqueza)\n"
                                "   - Falta de demanda/oferta (breaks, buracos de liquidez)\n"
                                "6) Só sugira ENTRADA (compra/venda) quando houver uma região CLARA e bem defendida,\n"
                                "   descrevendo preço aproximado da entrada e uma zona de invalidação.\n"
                                "7) Se o cenário não estiver claro, prefira recomendar 'aguardar' e explique o porquê.\n"
                                "8) Seja sucinto, objetivo e profissional."
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

    def _call_model(self, prompt: str, event_data: Dict[str, Any]) -> tuple[str, Optional[Any]]:
        """
        Chama o provedor atual e retorna (raw_response, structured_or_None).
        """
        # Preferir JSON Mode + AsyncOpenAI se disponível
        if self.mode in ("openai", "groq") and self.client_async and PYDANTIC_AVAILABLE and AITradeAnalysis:
            try:
                raw, structured = asyncio.run(self._a_call_openai_structured(prompt))
                if raw:
                    return raw, structured
            except Exception as e:
                logging.error(f"Erro ao usar AsyncOpenAI structured: {e}")
                # Fallback para cliente síncrono
                try:
                    text = self._call_openai_compatible(prompt)
                    return text, None
                except Exception as e2:
                    logging.error(f"Erro fallback sync OpenAI: {e2}")
                    return self._generate_mock_analysis(event_data), None

        # Sem JSON Mode ou sem Async: usar cliente síncrono
        if self.mode in ("openai", "groq") and self.client:
            text = self._call_openai_compatible(prompt)
            return text, None
        elif self.mode == "dashscope":
            text = self._call_dashscope(prompt)
            return text, None
        else:
            text = self._generate_mock_analysis(event_data)
            return text, None

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
    # NÚCLEO DE ANÁLISE (compartilhado entre analyze_event e analyze)
    # ====================================================================

    def _analyze_internal(self, event_data: Dict[str, Any]) -> tuple[str, Optional[Any]]:
        """
        Núcleo de análise: constrói prompt, chama modelo e retorna (texto, structured).
        """
        if not self.enabled:
            try:
                self._initialize_api()
            except Exception:
                pass
        
        if not self.enabled:
            return self._generate_mock_analysis(event_data), None

        if self._should_test_connection():
            self.last_test_time = time.time()
            if not self._test_connection():
                if self.connection_failed_count >= self.max_failures_before_mock:
                    return self._generate_mock_analysis(event_data), None

        try:
            prompt = self._create_prompt(event_data)
        except Exception as e:
            logging.error(f"Erro ao criar prompt: {e}", exc_info=True)
            return self._generate_mock_analysis(event_data), None

        try:
            raw, structured = self._call_model(prompt, event_data)
        except Exception as e:
            logging.error(f"Erro na chamada de IA: {e}", exc_info=True)
            raw, structured = self._generate_mock_analysis(event_data), None

        if not raw:
            raw = self._generate_mock_analysis(event_data)
        return raw, structured

    # ====================================================================
    # INTERFACE PÚBLICA
    # ====================================================================
    
    def analyze_event(self, event_data: Dict[str, Any]) -> str:
        """
        Analisa evento e retorna análise da IA (string).
        Mantido para compatibilidade com código legado.
        """
        try:
            analysis_text, _ = self._analyze_internal(event_data)
            return analysis_text
        except Exception as e:
            logging.error(f"Erro em analyze_event(): {e}", exc_info=True)
            return self._generate_mock_analysis(event_data)

    def analyze(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa evento e retorna resultado estruturado.

        Retorno:
            {
              "raw_response": str,
              "structured": dict | None,  # se JSON Mode + Pydantic disponíveis
              "tipo_evento": str,
              "ativo": str,
              "timestamp": str,
              "success": bool,
              "mode": str,
              "model": str,
              "error": str (opcional)
            }
        """
        try:
            analysis_text, structured = self._analyze_internal(event_data)
            
            tipo_evento = event_data.get("tipo_evento", "N/A")
            ativo = event_data.get("ativo") or event_data.get("symbol") or "N/A"
            
            logging.info(f"✅ IA [{self.mode or 'mock'}] analisou: {tipo_evento} - {ativo}")
            
            return {
                "raw_response": analysis_text,
                "structured": (
                    structured.model_dump() if (PYDANTIC_AVAILABLE and structured is not None) else None
                ),
                "tipo_evento": tipo_evento,
                "ativo": ativo,
                "timestamp": self.time_manager.now_iso(),
                "success": True,
                "mode": self.mode or "mock",
                "model": self.model_name,
            }
            
        except Exception as e:
            logging.error(f"❌ Erro em analyze(): {e}", exc_info=True)
            return {
                "raw_response": f"❌ Erro ao analisar evento: {str(e)}",
                "structured": None,
                "tipo_evento": event_data.get("tipo_evento", "N/A"),
                "ativo": event_data.get("ativo") or event_data.get("symbol") or "N/A",
                "timestamp": self.time_manager.now_iso(),
                "success": False,
                "error": str(e),
                "mode": self.mode or "mock",
                "model": self.model_name,
            }

    def close(self):
        """Fecha conexão com IA."""
        if self.mode == "groq":
            logging.info("🔌 Desconectando GroqCloud...")
        self.client = None
        self.client_async = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ====================================================================
# 🧪 TESTE DE VALIDAÇÃO (executar diretamente)
# ====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTANDO AI_ANALYZER v2.3.0 (GroqCloud + Structured Output focado em regiões)")
    print("=" * 70)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    analyzer = AIAnalyzer()
    
    print(f"\n✅ Modo ativo: {analyzer.mode or 'MOCK'}")
    print(f"✅ Modelo: {analyzer.model_name}")
    print(f"✅ Enabled: {analyzer.enabled}")
    
    if analyzer.mode:
        print("\n🔍 Testando conexão...")
        if analyzer._test_connection():
            print("✅ Conexão OK!")
        else:
            print("❌ Falha na conexão")
    
    print("\n📝 Testando análise...")
    mock_event = {
        "tipo_evento": "Absorção",
        "ativo": "BTCUSDT",
        "delta": -15.5,
        "volume_total": 125.3,
        "preco_fechamento": 95000,
        "resultado_da_batalha": "Vendedores",
    }
    
    result = analyzer.analyze(mock_event)
    
    print(f"\n📊 Resultado:")
    print(f"  Success: {result['success']}")
    print(f"  Modo: {result.get('mode', 'N/A')}")
    print(f"  Modelo: {result.get('model', 'N/A')}")
    print(f"  Structured: {result.get('structured')}")
    print(f"  Resposta ({len(result['raw_response'])} chars):")
    print(f"  {result['raw_response'][:300]}...")
    
    print("\n" + "=" * 70)
    print("✅ TESTE CONCLUÍDO")
    print("=" * 70 + "\n")