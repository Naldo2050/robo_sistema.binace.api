# ai_analyzer_qwen.py v2.4.0 + PATCH 2 - COM INTELIGÊNCIA QUANTITATIVA COMO BASE PRINCIPAL
"""
AI Analyzer para eventos de mercado com validação de dados.

🔹 NOVIDADES v2.3.0:
  ✅ Modo texto livre para GroqCloud (sem JSON estruturado)
     - resposta em texto natural
     - foco em análise qualitativa
     - compatibilidade total com modelos Groq
  ✅ Prompts ajustados para foco em:
     - suporte / resistência
     - absorção / exaustão
     - falta de demanda / oferta
     - pontos claros de entrada e zonas de defesa/invalidação

🔹 NOVIDADES v2.2.0:
  ✅ Templates de prompt com Jinja2 (se disponível), com fallback para f-strings
  ✅ Cliente assíncrono AsyncOpenAI para Groq/OpenAI (usado internamente via asyncio.run)

🔹 NOVIDADES v2.1.0:
   ✅ Suporte completo ao GroqCloud (PRIORIDADE 1)
   ✅ Fallback inteligente: Groq → OpenAI → Mock (DashScope desabilitado)
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

🔹 NOVIDADE v2.3.x (ESTA VERSÃO):
  ✅ Integração com HealthMonitor via heartbeat periódico:
     - AIAnalyzer pode receber um HealthMonitor externo
     - Envia heartbeat("ai") a cada 30s enquanto ativo
     - Fecha a thread de heartbeat no close()

🔹 PATCH 2 - IMPLEMENTADO:
  ✅ Impede fallback automático para provider errado
  ✅ Se provider=groq, tenta modelos alternativos da Groq apenas
  ✅ Se nenhum modelo Groq funcionar, vai para MOCK (não OpenAI automaticamente)
  ✅ Só troca para outro provider se explicitamente configurado em provider_fallbacks
  ✅ Fallback automático DESABILITADO por padrão
"""

import logging
import os
import random
import time
import asyncio
import threading
import json
from typing import Any, Dict, Optional, Literal, TYPE_CHECKING

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

# Import opcional do HealthMonitor (para type hint e uso de heartbeat)
if TYPE_CHECKING:
    # Import usado apenas para type checking (Pylance, mypy etc.)
    from health_monitor import HealthMonitor

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
    Environment = None  # type: ignore[assignment, misc]
    BaseLoader = None  # type: ignore[assignment, misc]

# Pydantic for structured output (optional)
if TYPE_CHECKING:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
else:
    try:
        from pydantic import BaseModel
        PYDANTIC_AVAILABLE = True
    except Exception:
        PYDANTIC_AVAILABLE = False
        # Use a dummy class for runtime when Pydantic is not available
        class BaseModel:  # type: ignore[no-redef]
            pass

from time_manager import TimeManager
from orderbook_core.structured_logging import StructuredLogger

# Regime Detector integration
# Sempre use type: ignore pois o módulo pode não existir
try:
    from src.analysis.regime_detector import RegimeDetector  # type: ignore[import-not-found,reportAttributeAccessIssue]
    REGIME_DETECTOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Regime detector não disponível: {e}")
    REGIME_DETECTOR_AVAILABLE = False

load_dotenv()



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
    # Define as None when Pydantic is not available
    AITradeAnalysis = None  # type: ignore[assignment,misc]


# ========================
# Jinja2 Templates
# ========================

if JINJA_AVAILABLE:
    try:
        # BaseLoader() é a classe base do Jinja2, não uma instância
        _jinja_env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)  # type: ignore[arg-type]
    except Exception:
        _jinja_env = None
else:
    _jinja_env = None

# ========================
# SYSTEM PROMPT (v2.4.0)
# ========================

SYSTEM_PROMPT = """Você é analista institucional de fluxo, suporte/resistência e regiões de defesa.

🔹 HORIZONTE DE ANÁLISE: Focado em entradas rápidas de 5-15 minutos (scalp), com validação em horizontes maiores se disponível.

🔹 OBJETIVO: Identificar regiões de entrada claras para trades curtos, priorizando defesa institucional (absorção) e pontos de invalidação técnicos. Não force trades em ruído.

═══════════════════════════════════════════════════════
🧠 REGRA FUNDAMENTAL: INTELIGÊNCIA QUANTITATIVA É A BASE
═══════════════════════════════════════════════════════

O modelo XGBoost fornece probabilidades matemáticas (prob_up, prob_down) e um action_bias (compra/venda/aguardar).

✅ USE A INTELIGÊNCIA QUANTITATIVA COMO BASE PRINCIPAL:
   - O action_bias deve guiar sua decisão inicial
   - A confiança do modelo (confidence_score) indica força da previsão

⚠️ SÓ VÁ CONTRA O VIÉS MATEMÁTICO SE:
   - Houver absorção MASSIVA no orderbook contrária ao viés
   - Whale activity significativa na direção oposta
   - CVD/Net Flow em forte divergência
   - Evidência EXTREMA de exaustão/reversão no fluxo

Se não houver evidência MUITO FORTE, SIGA o action_bias do modelo.

═══════════════════════════════════════════════════════

REGRAS GERAIS:
1) Use SOMENTE dados fornecidos explicitamente.
2) Se marcado 'Indisponível' ou '⚠️', NÃO use.
3) Orderbook zerado? Use fluxo (net_flow, flow_imbalance, tick_rule).
4) Contradições? Ignore dado contraditório.
5) Foque em identificar REGIÕES IMPORTANTES:
   - Suportes e resistências relevantes (use VP: POC/VAH/VAL)
   - Regiões de absorção (defesa) e exaustão (fraqueza) via fluxo/whales
   - Falta de demanda/oferta (breaks, buracos de liquidez)
6) Só sugira ENTRADA quando houver região CLARA e bem defendida,
   descrevendo preço aproximado e zona de invalidação.
7) Em contexto de RUÍDO (range estreito, fluxo misto, confiança baixa):
   - Reduza sinais; prefira action='wait' ou 'avoid'.
8) Incentive action='wait' se confiança do modelo < 50% ou edge não claro.
9) Se o cenário não estiver claro, prefira recomendar 'aguardar'.
10) Seja sucinto, objetivo e profissional.

11) Responda SEMPRE e APENAS em português do Brasil.
12) NÃO utilize inglês em nenhuma parte da resposta.
13) NÃO use tags <think> nem mostre seu raciocínio passo a passo; entregue apenas a análise final em português.
14) GARANTA que toda sua comunicação seja em português brasileiro, sem mistura de idiomas.

Responda sempre e apenas em português do Brasil.
Não utilize inglês em nenhuma parte da resposta.
Não use tags <think> nem mostre seu raciocínio passo a passo; entregue apenas a análise final em português.
"""



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


def _is_model_decommissioned_error(err: Exception) -> bool:
    """
    Detecta o erro retornado pela API (via SDK OpenAI compatível) quando o modelo foi descontinuado.
    Fazemos por string + tentativa de ler 'body' quando existir.
    """
    msg = str(err)
    if "model_decommissioned" in msg:
        return True

    body = getattr(err, "body", None)
    if isinstance(body, dict):
        code = (body.get("error") or {}).get("code")
        if code == "model_decommissioned":
            return True

    return False


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


def _dedupe_keep_order(items):
    out = []
    for x in items:
        if x and x not in out:
            out.append(x)
    return out


def _models_from_cfg(cfg: dict) -> list[str]:
    primary = cfg.get("model", "qwen/qwen3-32b")
    fallbacks = cfg.get("model_fallbacks", [])
    if not isinstance(fallbacks, list):
        fallbacks = []
    return _dedupe_keep_order([primary, *fallbacks])


class AIAnalyzer:
    """Analisador de IA com validação robusta de dados e suporte GroqCloud + structured output + heartbeat."""
    
    def __init__(
        self,
        health_monitor: Optional["HealthMonitor"] = None,
        module_name: str = "ai",
    ):
        """
        health_monitor: instância de HealthMonitor (opcional).
        module_name: nome usado para registrar heartbeat (default: 'ai').

        Compatível com chamadas antigas: AIAnalyzer() continua funcionando.
        """
        self.client: Optional[Any] = None        # cliente síncrono
        self.client_async: Optional[Any] = None  # cliente assíncrono (AsyncOpenAI)
        self.enabled = False
        self.mode: Optional[str] = None
        self.time_manager = TimeManager()

        # Logger estruturado interno da IA
        self.slog = StructuredLogger("ai_analyzer", "AI")

        # Integração com HealthMonitor
        self.health_monitor = health_monitor
        self.module_name = module_name
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None

        # Carrega configuração do config.json
        self.config = {}
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            logging.warning(f"Erro ao carregar config.json: {e}")

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

        logging.info(
            "🧠 IA Analyzer v2.4.0 + PATCH 2 inicializada - GroqCloud (modo texto livre)"
        )
        try:
            self._initialize_api()
        except Exception as e:
            logging.warning(f"Falha ao inicializar provedores de IA: {e}. Usando mock.")
            self.mode = None
            self.enabled = True

        # Se existir HealthMonitor, envia um primeiro heartbeat e inicia thread periódica
        if self.health_monitor is not None:
            try:
                self.health_monitor.heartbeat(self.module_name)
            except Exception:
                pass

            self._hb_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"AIAnalyzerHeartbeat-{self.module_name}",
                daemon=True,
            )
            self._hb_thread.start()
            logging.info("🫀 Heartbeat do módulo '%s' iniciado (AIAnalyzer).", self.module_name)

    # ---------------- HEARTBEAT LOOP ----------------
    def _heartbeat_loop(self):
        """
        Envia heartbeat periódico para o HealthMonitor enquanto o AIAnalyzer estiver ativo.
        Se o processo travar de verdade, esta thread também pára, e o HealthMonitor detecta.
        """
        # Intervalo padrão de heartbeat: 30s (pode ser ajustado lendo do config.py)
        interval = getattr(app_config, "HEALTH_CHECK_INTERVAL", 30)
        interval = max(5, min(interval, 60))  # mantém entre 5s e 60s para segurança

        while not self._hb_stop.is_set():
            try:
                if self.health_monitor is not None:
                    self.health_monitor.heartbeat(self.module_name)
            except Exception:
                # Nunca deixe exceção matar a thread de heartbeat
                pass
            # Espera até o próximo ciclo ou parada
            for _ in range(int(interval * 10)):  # subdivide em passos de 0.1s para reação rápida ao stop
                if self._hb_stop.is_set():
                    break
                time.sleep(0.1)

    # ------------------------------------------------

    def _initialize_api(self):
        """
        Inicializa provedores de IA com PATCH 2 - Fallback Controlado.
        
        ✅ NOVA LÓGICA (Patch 2):
          - Se provider=groq, tenta apenas modelos alternativos da Groq
          - Se nenhum modelo Groq funcionar, vai para MOCK (não OpenAI automaticamente)
          - Só troca para outro provider se explicitamente configurado em provider_fallbacks
          - Fallback automático DESABILITADO por padrão
        """
        
        # Carrega configuração do config.json
        ai_cfg = self.config.get("ai", {})
        provider = ai_cfg.get("provider", "groq")
        provider_fallbacks = ai_cfg.get("provider_fallbacks", [])
        
        # Lista para tracking de providers testados
        providers_tested = []
        
        # ============================================================
        # 🚀 PROVIDER ESPECÍFICO: GROQ
        # ============================================================
        if provider == "groq":
            providers_tested.append("groq")
            groq_key = os.getenv("GROQ_API_KEY") or getattr(app_config, "GROQ_API_KEY", None)
            
            if OPENAI_AVAILABLE and groq_key:
                groq_cfg = ai_cfg.get("groq", {})
                groq_base_url = groq_cfg.get("base_url", "https://api.groq.com/openai/v1")
                
                # Valida formato da chave
                if not groq_key.startswith("gsk_"):
                    logging.warning(
                        f"⚠️ GROQ_API_KEY suspeita (não começa com 'gsk_'). "
                        f"Tentando mesmo assim..."
                    )
                
                self.base_url = groq_base_url
                # Cliente síncrono OpenAI-compatível apontando para Groq
                self.client = OpenAI(
                    api_key=groq_key,
                    base_url=self.base_url
                )
                # Cliente assíncrono, se disponível
                if ASYNC_OPENAI_AVAILABLE and AsyncOpenAI is not None:
                    self.client_async = AsyncOpenAI(
                        api_key=groq_key,
                        base_url=self.base_url
                    )
                logging.info("🔧 Groq client configurado | base_url=%s", self.base_url)

                models = _models_from_cfg(groq_cfg)
                self._groq_model_candidates = models
                if not models:
                    logging.warning("Nenhum modelo Groq configurado em ai.groq.model")
                else:
                    last_err = None
                    selected = None
                    for m in models:
                        try:
                            self.client.chat.completions.create(
                                model=m,
                                messages=[{"role": "user", "content": "ping"}],
                                temperature=0,
                                max_tokens=1,
                                timeout=10,
                            )
                            selected = m
                            break
                        except Exception as e:
                            last_err = e
                            logging.warning(f"⚠️ Ping falhou no modelo Groq {m}: {e}")
                    if not selected:
                        logging.error(f"❌ Groq sem modelo válido. Último erro: {last_err}")
                        # PATCH 2: Se Groq falhar, NÃO fazer fallback automático
                        # Vai para mock ou tentar fallbacks explicitamente configurados
                    else:
                        self.model_name = selected
                        self.mode = "groq"
                        self.enabled = True
                        logging.info(
                            f"🚀 GroqCloud ATIVO | Modelo final: {self.model_name} | "
                            f"Chave: {groq_key[:10]}...{groq_key[-4:]}"
                        )
                        try:
                            self.slog.info(
                                "ai_provider_selected",
                                provider="groq",
                                model=self.model_name,
                            )
                        except Exception:
                            pass
                        return
            
            # Se chegou aqui, Groq falhou
            # PATCH 2: Só tenta fallbacks se explicitamente configurados
            if not provider_fallbacks:
                logging.info("🔧 Groq falhou e nenhum fallback configurado. Ativando modo MOCK.")
                self._activate_mock_mode()
                return
            else:
                logging.info(f"🔄 Groq falhou, tentando fallbacks configurados: {provider_fallbacks}")
        
        # ============================================================
        # FALLBACKS EXPLICITAMENTE CONFIGURADOS
        # ============================================================
        # Só chega aqui se:
        # 1. Provider não é groq, OU
        # 2. Groq falhou E tem fallbacks configurados
        
        for fallback_provider in provider_fallbacks:
            if fallback_provider in providers_tested:
                continue  # Já testou este provider
                
            if fallback_provider == "openai":
                providers_tested.append("openai")
                if self._try_initialize_openai():
                    return
                    
            elif fallback_provider == "dashscope":
                providers_tested.append("dashscope")
                if self._try_initialize_dashscope():
                    return
        
        # ============================================================
        # PROVIDER PADRÃO: OPENAI (só se não for groq ou groq com fallback)
        # ============================================================
        if provider != "groq" and "openai" not in providers_tested:
            if self._try_initialize_openai():
                return
        
        # ============================================================
        # FALLBACK FINAL: MOCK
        # ============================================================
        self._activate_mock_mode()
        
    def _try_initialize_openai(self) -> bool:
        """Tenta inicializar OpenAI. Retorna True se succeeded."""
        try:
            self.client = OpenAI()  # Usa OPENAI_API_KEY
            if ASYNC_OPENAI_AVAILABLE and AsyncOpenAI is not None:
                self.client_async = AsyncOpenAI()
            self.mode = "openai"
            self.enabled = True
            logging.info("🔧 OpenAI client configurado (fallback)")
            try:
                self.slog.info(
                    "ai_provider_selected",
                    provider="openai",
                    model=self.model_name,
                )
            except Exception:
                pass
            return True
        except Exception as e:
            logging.warning(f"OpenAI indisponível: {e}")
            return False
    
    def _try_initialize_dashscope(self) -> bool:
        """Tenta inicializar DashScope. Retorna True se succeeded."""
        token = os.getenv("DASHSCOPE_API_KEY") or getattr(app_config, "DASHSCOPE_API_KEY", None)
        
        if DASHSCOPE_AVAILABLE and token:
            try:
                import dashscope
                dashscope.api_key = token
                self.mode = "dashscope"
                self.enabled = True
                logging.info("🔧 DashScope configurado (fallback)")
                try:
                    self.slog.info(
                        "ai_provider_selected",
                        provider="dashscope",
                        model=self.model_name,
                    )
                except Exception:
                    pass
                return True
            except Exception as e:
                logging.warning(f"DashScope indisponível: {e}")
        
        return False
    
    def _activate_mock_mode(self):
        """Ativa modo mock."""
        self.mode = None
        self.enabled = True
        logging.info("🔧 Modo MOCK ativado (sem provedores externos).")
        try:
            self.slog.warning(
                "ai_provider_selected",
                provider="mock",
                model=self.model_name,
            )
        except Exception:
            pass

    async def _select_working_groq_model(self) -> str | None:
        """Seleciona um modelo Groq válido testando cada candidato."""
        if not self._groq_model_candidates:
            return None

        last_err = None
        for candidate in self._groq_model_candidates:
            try:
                # Temporariamente define o modelo para teste
                original_model = self.model_name
                self.model_name = candidate

                # Testa conexão (ping)
                if await self._ping_once_async():
                    # Sucesso: mantém o modelo selecionado
                    logging.info(f"✅ Modelo Groq válido encontrado: {candidate}")
                    return candidate
                else:
                    logging.warning(f"⚠️ Ping falhou para modelo {candidate}")

            except Exception as e:
                last_err = e
                if _is_model_decommissioned_error(e):
                    logging.warning(f"⚠️ Modelo Groq descontinuado: {candidate} | tentando próximo fallback...")
                    continue
                else:
                    logging.warning(f"⚠️ Falha ao testar modelo Groq {candidate}: {e}")
                    continue
            finally:
                # Restaura modelo original se falhou
                if 'original_model' in locals():
                    self.model_name = original_model

        logging.error(f"❌ Nenhum modelo Groq funcionou. Último erro: {last_err}")
        return None

    async def _ping_once_async(self) -> bool:
        """Versão assíncrona do ping para um modelo."""
        try:
            if self.client_async is None:
                return False
            # Tenta uma chamada simples
            response = await self.client_async.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Diagnóstico curto. Responda apenas 'OK'."},
                    {"role": "user", "content": "Ping curto. Responda com 'OK'."},
                ],
                max_tokens=3,
                temperature=0.0,
                timeout=10,
            )
            content = response.choices[0].message.content.strip().upper()
            return content.startswith("OK")
        except Exception:
            return False

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
        ok = True
        try:
            if self.mode == "openai" or self.mode == "groq":
                if self.client is None:
                    return False
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
                
                ok = success
                
            elif self.mode == "dashscope":
                r = Generation.call(
                    model=self.model_name,
                    messages=[  # type: ignore[arg-type]
                        {"role": "system", "content": "Diagnóstico curto. Responda apenas 'OK'."},
                        {"role": "user", "content": prompt},
                    ],
                    result_format="message",
                    max_tokens=3,
                    temperature=0.0,
                    timeout=10,
                )
                content = _extract_dashscope_text(r).upper()
                ok = content.startswith("OK")
            else:
                ok = True  # Mock sempre OK
                
        except Exception as e:
            self.connection_failed_count += 1
            logging.warning(
                f"Falha no ping da IA [{self.mode}] ({self.connection_failed_count}): {e}"
            )
            ok = False

        # log estruturado
        try:
            if ok:
                self.slog.info(
                    "ai_ping_ok",
                    mode=self.mode or "mock",
                )
            else:
                self.slog.warning(
                    "ai_ping_failed",
                    mode=self.mode or "mock",
                    failures=self.connection_failed_count,
                )
        except Exception:
            pass

        return ok

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
        Cria prompt para IA.
        [MODIFICADO] Verifica se existe 'ai_payload' estruturado antes de usar lógica legada.
        """
        ai_payload = event_data.get("ai_payload")
        if ai_payload and isinstance(ai_payload, dict):
            try:
                logging.debug("Usando ai_payload estruturado para montar o prompt")
                return self._build_structured_prompt(ai_payload)
            except Exception as e:
                logging.error(
                    f"Erro ao construir prompt estruturado: {e}. "
                    f"Usando fallback legado.",
                    exc_info=True,
                )
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

    def _build_structured_prompt(self, payload: Dict[str, Any]) -> str:
        """
        Constrói o prompt usando o payload padronizado (ai_payload_builder).
        Se algum campo estiver faltando, faz fallback para 'N/A' ou valores simples.
        """
        meta = payload.get("signal_metadata", {}) or {}
        price = payload.get("price_context", {}) or {}
        flow = payload.get("flow_context", {}) or {}
        ob = payload.get("orderbook_context", {}) or {}
        macro = payload.get("macro_context", {}) or {}
        ml = payload.get("ml_features", {}) or {}
        hist = payload.get("historical_stats", {}) or {}

        symbol = payload.get("symbol") or meta.get("symbol") or "N/A"
        timestamp = payload.get("timestamp") or meta.get("timestamp") or "N/A"

        # Preço atual e OHLC
        ohlc = price.get("ohlc", {}) or {}
        current_price = format_price(price.get("current_price"))
        open_p = format_price(ohlc.get("open"))
        high_p = format_price(ohlc.get("high"))
        low_p = format_price(ohlc.get("low"))
        close_p = format_price(ohlc.get("close"))

        vp = price.get("volume_profile_daily", {}) or {}
        poc = format_price(vp.get("poc"))
        vah = format_price(vp.get("vah"))
        val = format_price(vp.get("val"))

        lines = []

        # Cabeçalho
        lines.append(f"Análise Institucional – {symbol}")
        lines.append(f"{timestamp} | Tipo: {meta.get('type', 'N/A')}")
        lines.append(f"Descrição: {meta.get('description', 'Sem descrição')}")
        lines.append(f"Resultado da Batalha: {meta.get('battle_result', 'N/A')}")
        lines.append("")

        # Preço
        lines.append("CONTEXTO DE PREÇO")
        lines.append(f"  • Preço Atual: {current_price}")
        lines.append(f"  • OHLC: O:{open_p} H:{high_p} L:{low_p} C:{close_p}")
        lines.append(f"  • VP Diário: POC {poc} | VAH {vah} | VAL {val}")
        lines.append("")

        # Fluxo básico (adapte conforme chaves reais do seu builder)
        net_flow = flow.get("net_flow")
        cvd_acc = flow.get("cvd_accumulated")
        if net_flow is not None or cvd_acc is not None:
            lines.append("CONTEXTO DE FLUXO")
            if net_flow is not None:
                lines.append(f"  • Net Flow (janela): {format_delta(net_flow)}")
            if cvd_acc is not None:
                lines.append(f"  • CVD acumulado: {format_delta(cvd_acc)}")
            lines.append("")

        # Orderbook (simplificado, adapte conforme builder)
        bid_usd = ob.get("bid_depth_usd")
        ask_usd = ob.get("ask_depth_usd")
        imbalance = ob.get("imbalance")
        if bid_usd is not None or ask_usd is not None:
            lines.append("ORDERBOOK / LIQUIDEZ")
            lines.append(
                f"  • Bids: {format_large_number(bid_usd)} | "
                f"Asks: {format_large_number(ask_usd)}"
            )
            if imbalance is not None:
                lines.append(f"  • Imbalance: {format_delta(imbalance)}")
            lines.append("")

        # Macro (simplificado)
        if macro:
            lines.append("MACRO / REGIME")
            session = macro.get("session") or macro.get("session_name")
            if session:
                lines.append(f"  • Sessão: {session}")
            trends = macro.get("multi_timeframe_trends") or {}
            if trends:
                lines.append("  • Tendências multi-timeframe:")
                for tf, tr in trends.items():
                    val = tr.get("tendencia") if isinstance(tr, dict) else tr
                    lines.append(f"    - {tf}: {val}")
            lines.append("")

        # Histórico (simplificado)
        if hist:
            lp = hist.get("long_prob")
            sp = hist.get("short_prob")
            np_ = hist.get("neutral_prob")
            lines.append("ESTATÍSTICA HISTÓRICA")
            lines.append(
                f"  • Probabilidades: Long={lp} | Short={sp} | Neutro={np_}"
            )
            lines.append("")

        # ============================================================
        # 🆕 INTELIGÊNCIA QUANTITATIVA (quant_model + ml_str)
        # ============================================================
        quant = payload.get("quant_model", {}) or {}
        ml_str_raw = payload.get("ml_str", "") or ""

        if quant:
            prob_up = quant.get("model_probability_up")
            prob_down = quant.get("model_probability_down")
            sentiment_model = quant.get("model_sentiment", "Indefinido")
            action_bias = quant.get("action_bias", "aguardar")
            confidence_model = quant.get("confidence_score", 0.0)
            features_used = quant.get("features_used", 0)
            total_features = quant.get("total_features", 0)

            lines.append("═══════════════════════════════════════════════════════")
            lines.append("🧠 INTELIGÊNCIA QUANTITATIVA (XGBoost) – USO OBRIGATÓRIO")
            lines.append("═══════════════════════════════════════════════════════")
            if prob_up is not None:
                lines.append(f"  📈 Probabilidade de Alta: {prob_up * 100:.1f}%")
            if prob_down is not None:
                lines.append(f"  📉 Probabilidade de Baixa: {prob_down * 100:.1f}%")
            lines.append(f"  🎯 Viés Matemático: {sentiment_model}")
            lines.append(f"  🔒 Action Bias (viés sugerido): {action_bias.upper()}")
            lines.append(f"  📊 Confiança do Modelo: {confidence_model * 100:.1f}%")
            lines.append(f"  🔍 Features usadas: {features_used}/{total_features}")
            lines.append("")
            lines.append("⚠️ REGRA CRÍTICA:")
            lines.append("   Esta inteligência quantitativa é sua BASE PRINCIPAL de decisão.")
            lines.append("   Só vá CONTRA o action_bias se o fluxo ou orderbook mostrarem")
            lines.append("   evidência MUITO FORTE (ex: absorção massiva, whale dump) na")
            lines.append("   direção oposta. Caso contrário, SIGA o viés matemático.")
            lines.append("")
        elif ml_str_raw:
            # Fallback: usa ml_str formatado se quant_model não existir
            lines.append(ml_str_raw.strip())
            lines.append("")

        # Instruções
        lines.append("═══════════════════════════════════════════════════════")
        lines.append("📋 TAREFA DA IA")
        lines.append("═══════════════════════════════════════════════════════")
        lines.append("")
        lines.append("1) USE A INTELIGÊNCIA QUANTITATIVA COMO BASE PRINCIPAL.")
        lines.append("   O viés matemático (action_bias) deve guiar sua decisão inicial.")
        lines.append("")
        lines.append("2) CONFIRME OU INVALIDE com dados de fluxo e orderbook:")
        lines.append("   - CVD/Net Flow alinhados com o viés? → CONFIRMA")
        lines.append("   - Orderbook com absorção forte? → Pode indicar defesa/reversão")
        lines.append("   - Whale activity oposta ao viés? → Sinal de alerta")
        lines.append("")
        lines.append("3) SÓ CONTRARIE O VIÉS QUANTITATIVO se houver evidência MUITO FORTE:")
        lines.append("   - Absorção massiva contrária")
        lines.append("   - Whale dump/pump significativo")
        lines.append("   - Divergência grave fluxo vs preço")
        lines.append("")
        lines.append("4) Defina região de entrada e zona de invalidação (se houver setup).")
        lines.append("")
        lines.append("5) Se dados forem conflitantes ou confiança quantitativa baixa (<50%),")
        lines.append("   recomende aguardar com action='wait' ou 'avoid'.")

        return "\n".join(lines)

    # ====================================================================
    # CALLERS (SYNC + ASYNC)
    # ====================================================================

    async def _a_call_openai_text(self, prompt: str) -> str:
        """Versão assíncrona simples (texto livre) para OpenAI/Groq com fallbacks."""
        if not self.client_async:
            raise RuntimeError("Cliente assíncrono não inicializado")

        # Usar lista centralizada de candidatos
        models_to_try = self._groq_model_candidates if self.mode == "groq" else [self.model_name]

        for model in models_to_try:
            try:
                response = await self.client_async.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=700,
                    temperature=0.25,
                    timeout=30,
                )
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content.strip()
                    # Sucesso: atualizar modelos se foi fallback
                    if model != self.model_name:
                        logging.info(f"🔄 Modelo trocado de {self.model_name} para {model} devido a decommissioned")
                        self.model_name = model
                        self.groq_model = model
                    return content
            except Exception as e:
                if _is_model_decommissioned_error(e):
                    logging.warning(f"Modelo {model} decommissioned. Tentando próximo...")
                    continue
                else:
                    logging.error(f"Erro com modelo {model}: {e}. Tentando próximo...")
                    continue

        # Todos falharam
        logging.error(f"Todos os modelos falharam para texto.")
        return ""



    def _call_openai_compatible(self, prompt: str, max_retries: int = 3) -> str:
        """
        Chama cliente OpenAI-compatível de forma síncrona (texto livre).
        Mantido para fallback e para ping.
        """
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(  # type: ignore[union-attr]
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
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
                    f"Erro {(self.mode or 'unknown').upper()} (tentativa {attempt+1}/{max_retries}): {e}"
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
                    messages=[  # type: ignore[arg-type]
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
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
        Chama o provedor atual e retorna (raw_response, None).
        Sempre usa modo texto livre.
        """
        # Sempre usar texto livre (modo padrão)
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

        # Refresca heartbeat no final de uma análise bem-sucedida (se monitor disponível)
        try:
            if self.health_monitor is not None:
                self.health_monitor.heartbeat(self.module_name)
        except Exception:
            pass

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
            
            try:
                self.slog.info(
                    "ai_analyze_ok",
                    mode=self.mode or "mock",
                    model=self.model_name,
                    tipo_evento=tipo_evento,
                    ativo=ativo,
                )
            except Exception:
                pass
            
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
            try:
                self.slog.error(
                    "ai_analyze_error",
                    error=str(e),
                    mode=self.mode or "mock",
                    model=self.model_name,
                    tipo_evento=event_data.get("tipo_evento", "N/A"),
                    ativo=event_data.get("ativo") or event_data.get("symbol") or "N/A",
                )
            except Exception:
                pass
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
        """Fecha conexão com IA e encerra heartbeat, se houver."""
        # Para heartbeat
        try:
            self._hb_stop.set()
            if self._hb_thread is not None and self._hb_thread.is_alive():
                self._hb_thread.join(timeout=5)
        except Exception:
            pass

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
    print("🧪 TESTANDO AI_ANALYZER v2.4.0 + PATCH 2 (GroqCloud - modo texto livre)")
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
