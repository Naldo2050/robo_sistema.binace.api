# ai_analyzer_qwen.py v2.5.1 - COM INTELIGÊNCIA QUANTITATIVA COMO BASE PRINCIPAL
"""
AI Analyzer para eventos de mercado com validação de dados.

🔹 NOVIDADES v2.5.1:
  ✅ Todos os erros Pylance corrigidos
  ✅ Type hints corrigidos e compatíveis
  ✅ Imports com fallbacks adequados
  ✅ Tratamento correto de tipos opcionais

🔹 NOVIDADES v2.5.0:
  ✅ Correções de bugs e melhorias de código
  ✅ Imports reorganizados e limpos
  ✅ Inicialização de variáveis corrigida
  ✅ Tratamento de erros aprimorado
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

# Payload Compressor (deep compression)
_optimize_deep_for_ai: Optional[Callable[..., Dict[str, Any]]] = None
_SectionCache: Optional[type] = None
_COMPRESSED_SYSTEM_PROMPT: Optional[str] = None
try:
    from src.utils.ai_payload_optimizer import (
        optimize_deep_for_ai as _optimize_deep,
        SectionCache as _SectionCacheClass,
        SYSTEM_PROMPT_COMPRESSED as _COMPRESSED_PROMPT,
    )
    _optimize_deep_for_ai = _optimize_deep
    _SectionCache = _SectionCacheClass
    _COMPRESSED_SYSTEM_PROMPT = _COMPRESSED_PROMPT
except ImportError:
    logging.info("ai_payload_optimizer deep compression not available")

# ========================
# CARREGAR .env
# ========================

load_dotenv()

# ========================
# IMPORTS LOCAIS COM FALLBACK
# ========================

try:
    import config as app_config
except Exception:
    app_config = None  # type: ignore[assignment]

# Format utils - com fallbacks inline
try:
    from format_utils import (
        format_delta as _format_delta,
        format_large_number as _format_large_number,
        format_percent as _format_percent,
        format_price as _format_price,
        format_quantity as _format_quantity,
        format_scientific as _format_scientific,
        format_time_seconds as _format_time_seconds,
    )
    
    def format_price(val: Any) -> str:
        return _format_price(val)
    
    def format_quantity(val: Any) -> str:
        return _format_quantity(val)
    
    def format_percent(val: Any, decimals: int = 2) -> str:
        return _format_percent(val, decimals)
    
    def format_large_number(val: Any) -> str:
        return _format_large_number(val)
    
    def format_delta(val: Any) -> str:
        return _format_delta(val)
    
    def format_scientific(val: Any, decimals: int = 4) -> str:
        return _format_scientific(val, decimals)
    
    def format_time_seconds(val: Any) -> str:
        return _format_time_seconds(val)

except ImportError:
    def format_price(val: Any) -> str:
        try:
            return f"{float(val):,.2f}" if val is not None else "N/A"
        except (ValueError, TypeError):
            return "N/A"
    
    def format_quantity(val: Any) -> str:
        try:
            return f"{float(val):,.4f}" if val is not None else "N/A"
        except (ValueError, TypeError):
            return "N/A"
    
    def format_percent(val: Any, decimals: int = 2) -> str:
        try:
            return f"{float(val) * 100:.{decimals}f}%" if val is not None else "N/A"
        except (ValueError, TypeError):
            return "N/A"
    
    def format_large_number(val: Any) -> str:
        try:
            v = float(val) if val is not None else 0
            if abs(v) >= 1_000_000:
                return f"{v/1_000_000:.2f}M"
            elif abs(v) >= 1_000:
                return f"{v/1_000:.2f}K"
            return f"{v:.2f}"
        except (ValueError, TypeError):
            return "N/A"
    
    def format_delta(val: Any) -> str:
        try:
            v = float(val) if val is not None else 0
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.2f}"
        except (ValueError, TypeError):
            return "N/A"
    
    def format_scientific(val: Any, decimals: int = 4) -> str:
        try:
            return f"{float(val):.{decimals}e}" if val is not None else "N/A"
        except (ValueError, TypeError):
            return "N/A"
    
    def format_time_seconds(val: Any) -> str:
        try:
            return f"{float(val):.2f}s" if val is not None else "N/A"
        except (ValueError, TypeError):
            return "N/A"

# TimeManager - com fallback
_TimeManager: Optional[type] = None
try:
    from time_manager import TimeManager as _TimeManagerImported
    _TimeManager = _TimeManagerImported
except ImportError:
    pass

if _TimeManager is not None:
    TimeManager = _TimeManager
else:
    class TimeManager:
        """Fallback TimeManager."""
        def now_iso(self) -> str:
            return datetime.now(timezone.utc).isoformat()
        
        def now_timestamp(self) -> float:
            return time.time()

# ========================
# TYPE CHECKING IMPORTS
# ========================

if TYPE_CHECKING:
    from health_monitor import HealthMonitor

# ========================
# IMPORTS OPCIONAIS COM TIPAGEM CORRETA
# ========================

# OpenAI / Groq
_OpenAI: Optional[type] = None
_AsyncOpenAI: Optional[type] = None
OPENAI_AVAILABLE = False
ASYNC_OPENAI_AVAILABLE = False

try:
    from openai import AsyncOpenAI as _AsyncOpenAIClass
    from openai import OpenAI as _OpenAIClass
    _OpenAI = _OpenAIClass
    _AsyncOpenAI = _AsyncOpenAIClass
    OPENAI_AVAILABLE = True
    ASYNC_OPENAI_AVAILABLE = True
except ImportError:
    try:
        from openai import OpenAI as _OpenAIClass  # type: ignore[import-not-found,no-redef]
        _OpenAI = _OpenAIClass
        OPENAI_AVAILABLE = True
    except ImportError:
        pass

# DashScope
_Generation: Optional[type] = None
_dashscope: Optional[Any] = None
DASHSCOPE_AVAILABLE = False

try:
    import dashscope as _dashscope_module
    from dashscope import Generation as _GenerationClass
    _Generation = _GenerationClass
    _dashscope = _dashscope_module
    DASHSCOPE_AVAILABLE = True
except ImportError:
    pass

# Jinja2
_jinja_env: Optional[Any] = None
JINJA_AVAILABLE = False

try:
    from jinja2 import BaseLoader, Environment
    JINJA_AVAILABLE = True
    try:
        _jinja_env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    except Exception:
        pass
except ImportError:
    pass

# Pydantic
PYDANTIC_AVAILABLE = False
try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    pass

# ========================
# IMPORTS DO PROJETO (com fallback)
# ========================

# StructuredLogger
_StructuredLogger: Optional[type] = None
try:
    from orderbook_core.structured_logging import StructuredLogger as _StructuredLoggerClass
    _StructuredLogger = _StructuredLoggerClass
except ImportError:
    pass


class FallbackStructuredLogger:
    """Fallback logger quando StructuredLogger não disponível."""
    
    def __init__(self, name: str, prefix: str = "") -> None:
        self._logger = logging.getLogger(name)
        self._prefix = prefix

    def info(self, event: str, **kwargs: Any) -> None:
        self._logger.info(f"[{self._prefix}] {event}: {kwargs}")

    def warning(self, event: str, **kwargs: Any) -> None:
        self._logger.warning(f"[{self._prefix}] {event}: {kwargs}")

    def error(self, event: str, **kwargs: Any) -> None:
        self._logger.error(f"[{self._prefix}] {event}: {kwargs}")


def _create_structured_logger(name: str, prefix: str) -> Any:
    """Cria StructuredLogger ou fallback."""
    if _StructuredLogger is not None:
        return _StructuredLogger(name, prefix)
    return FallbackStructuredLogger(name, prefix)


# LLM Payload Guardrail
_ensure_safe_llm_payload: Callable[[Any], Any]
try:
    from market_orchestrator.ai.llm_payload_guardrail import ensure_safe_llm_payload as _ensure_safe
    _ensure_safe_llm_payload = _ensure_safe
except ImportError:
    def _fallback_ensure_safe(payload: Any) -> Any:
        return payload
    _ensure_safe_llm_payload = _fallback_ensure_safe

# AI Payload Builder
_get_llm_payload_config: Callable[[], Dict[str, Any]]
try:
    from market_orchestrator.ai.ai_payload_builder import get_llm_payload_config as _get_config
    _get_llm_payload_config = _get_config
except ImportError:
    def _get_llm_payload_config() -> Dict[str, Any]:
        return {}

# Payload Metrics Aggregator
_append_metric_line: Callable[..., None]
_summarize_metrics: Callable[..., Dict[str, Any]]

try:
    from market_orchestrator.ai.payload_metrics_aggregator import (
        append_metric_line as _append_line,
        summarize_metrics as _summarize,
    )
    _append_metric_line = _append_line
    _summarize_metrics = _summarize
except ImportError:
    def _append_metric_line(*args: Any, **kwargs: Any) -> None:
        pass

    def _summarize_metrics(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {}

# ========================
# VARIÁVEIS GLOBAIS
# ========================

_PAYLOAD_METRICS_CALLS: int = 0
_PAYLOAD_METRICS_LAST_TS: float = 0.0

# ========================
# FUNÇÕES AUXILIARES
# ========================


def _evaluate_payload_tripwires(
    summary: Dict[str, Any], tripwires: Dict[str, Any]
) -> List[str]:
    """Avalia tripwires do payload e retorna lista de violações."""
    violations: List[str] = []
    if not isinstance(summary, dict) or not isinstance(tripwires, dict):
        return violations

    def exceeds(key_summary: str, key_thresh: str) -> bool:
        val = summary.get(key_summary)
        thresh = tripwires.get(key_thresh)
        return val is not None and thresh is not None and val > thresh

    if exceeds("fallback_rate", "fallback_rate_max"):
        violations.append("fallback_rate")
    if exceeds("abort_rate", "abort_rate_max"):
        violations.append("abort_rate")
    if exceeds("guardrail_block_rate", "guardrail_block_rate_max"):
        violations.append("guardrail_block_rate")
    if exceeds("bytes_p95", "bytes_p95_max"):
        violations.append("bytes_p95")

    cache_min = tripwires.get("cache_hit_rate_min")
    if isinstance(cache_min, dict):
        cache_hit_rate = summary.get("cache_hit_rate")
        if isinstance(cache_hit_rate, dict):
            for section, min_val in cache_min.items():
                try:
                    val = cache_hit_rate.get(section)
                    if val is not None and min_val is not None and val < float(min_val):
                        violations.append(f"cache_hit_rate.{section}")
                except (ValueError, TypeError):
                    continue

    return violations


def _log_payload_tripwires(summary: Dict[str, Any]) -> None:
    """Loga violações de tripwires do payload."""
    cfg = _get_llm_payload_config()
    tripwires = cfg.get("tripwires") if isinstance(cfg, dict) else {}
    violations = _evaluate_payload_tripwires(summary, tripwires or {})
    if violations:
        logging.warning(
            "PAYLOAD_TRIPWIRE_TRIGGERED %s",
            json.dumps({"violations": violations, "summary": summary}, ensure_ascii=False),
        )


def _is_model_decommissioned_error(err: Exception) -> bool:
    """Detecta se o erro indica que o modelo foi descontinuado."""
    msg = str(err).lower()
    if "model_decommissioned" in msg or "decommissioned" in msg:
        return True

    body = getattr(err, "body", None)
    if isinstance(body, dict):
        error_info = body.get("error")
        if isinstance(error_info, dict):
            code = error_info.get("code", "")
            if code == "model_decommissioned":
                return True

    return False


def _extract_dashscope_text(resp: Any) -> str:
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

        if isinstance(content, str):
            return content.strip()

        return ""
    except Exception:
        return ""


def _dedupe_keep_order(items: List[str]) -> List[str]:
    """Remove duplicatas mantendo a ordem."""
    out: List[str] = []
    for x in items:
        if x and x not in out:
            out.append(x)
    return out


def _models_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    """Extrai lista de modelos da configuração."""
    primary = cfg.get("model", "qwen/qwen3-32b")
    fallbacks = cfg.get("model_fallbacks", [])
    if not isinstance(fallbacks, list):
        fallbacks = []
    return _dedupe_keep_order([primary, *fallbacks])


# ========================
# CLASSE DE ANÁLISE
# ========================


class AITradeAnalysis:
    """Esquema estruturado para a resposta da IA."""

    def __init__(self, **kwargs: Any) -> None:
        self.sentiment: Optional[str] = kwargs.get("sentiment")
        self.confidence: Optional[float] = kwargs.get("confidence")
        self.action: Optional[str] = kwargs.get("action")
        self.rationale: Optional[str] = kwargs.get("rationale")
        self.entry_zone: Optional[str] = kwargs.get("entry_zone")
        self.invalidation_zone: Optional[str] = kwargs.get("invalidation_zone")
        self.region_type: Optional[str] = kwargs.get("region_type")

    def model_dump(self) -> Dict[str, Any]:
        """Retorna os dados como dicionário."""
        return {
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "action": self.action,
            "rationale": self.rationale,
            "entry_zone": self.entry_zone,
            "invalidation_zone": self.invalidation_zone,
            "region_type": self.region_type,
        }


# ========================
# SYSTEM PROMPTS
# ========================

SYSTEM_PROMPT_LEGACY = """Você é analista institucional de fluxo, suporte/resistência e regiões de defesa.

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

Responda sempre e apenas em português do Brasil.
Não utilize inglês em nenhuma parte da resposta.
Não use tags <think> nem mostre seu raciocínio passo a passo; entregue apenas a análise final em português.
"""

SYSTEM_PROMPT = """Voce e um analista institucional de fluxo, microestrutura e suporte/resistencia.

REGRAS OBRIGATORIAS (CRITICAS):
1. Responda APENAS em portugues do Brasil
2. NAO use ingles em NENHUMA parte
3. NAO mostre raciocinio, pensamentos ou analise passo a passo
4. NAO comece com "Okay", "Let me", "First", "Looking at" ou similares
5. Responda DIRETAMENTE com o JSON, sem texto antes ou depois

Horizonte: entradas rapidas de 5-15 minutos (scalp).

Use o vies quantitativo (action_bias + confidence_score) como base.
So contrarie o vies se houver evidencia MUITO forte no fluxo/orderbook.
Em duvida/ruido, prefira action="wait" ou action="avoid".

FORMATO DE SAIDA (OBRIGATORIO):
Responda SOMENTE com um JSON valido, sem markdown, sem texto extra:
{"sentiment":"bullish|bearish|neutral","confidence":0.0-1.0,"action":"buy|sell|hold|flat|wait|avoid","rationale":"texto curto PT-BR","entry_zone":"preco ou null","invalidation_zone":"preco ou null","region_type":"tipo ou null"}
"""

# ========================
# TEMPLATES
# ========================

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


# ========================
# TIPO PARA MENSAGENS
# ========================

ChatMessage = Dict[str, str]


# ========================
# CLASSE PRINCIPAL
# ========================


class AIAnalyzer:
    """
    Analisador de IA com validação robusta de dados e suporte a múltiplos provedores.
    
    Suporta:
    - GroqCloud (prioridade)
    - OpenAI
    - DashScope
    - Mock (fallback)
    """

    def __init__(
        self,
        health_monitor: Optional["HealthMonitor"] = None,
        module_name: str = "ai",
    ) -> None:
        """
        Inicializa o AIAnalyzer.

        Args:
            health_monitor: Instância de HealthMonitor (opcional).
            module_name: Nome usado para registrar heartbeat (default: 'ai').
        """
        # Clientes de API
        self.client: Optional[Any] = None
        self.client_async: Optional[Any] = None
        
        # Estado
        self.enabled: bool = False
        self.mode: Optional[str] = None
        self.base_url: Optional[str] = None
        self.model_name: str = "qwen-plus"
        
        # Lista de modelos candidatos para fallback
        self._groq_model_candidates: List[str] = []
        
        # Gerenciador de tempo
        self.time_manager: Any = TimeManager()

        # Logger estruturado
        self.slog: Any = _create_structured_logger("ai_analyzer", "AI")

        # Integração com HealthMonitor
        self.health_monitor = health_monitor
        self.module_name = module_name
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None

        # Configuração
        self.config: Dict[str, Any] = {}
        self._load_config()

        # Payload optimizer (deep compression)
        self._section_cache = _SectionCache() if _SectionCache is not None else None
        ai_cfg = self.config.get("ai", {})
        self._compression_enabled = (
            ai_cfg.get("payload_compression", True)
            if isinstance(ai_cfg, dict)
            else True
        )
        if self._section_cache and self._compression_enabled:
            logging.info(
                "🗜️ Payload compression ACTIVE (estimated ~70%% token savings)"
            )

        # Modelo padrão
        self.model_name = (
            getattr(app_config, "QWEN_MODEL", None) if app_config else None
        ) or os.getenv("QWEN_MODEL") or "qwen-plus"

        # Controle de conexão
        self.last_test_time: float = 0.0
        self.test_interval_seconds: int = 120
        self.connection_failed_count: int = 0
        self.max_failures_before_mock: int = 3

        logging.info(
            "🧠 IA Analyzer v2.5.1 inicializada - GroqCloud (modo texto livre)"
        )
        
        # Inicializa API
        try:
            self._initialize_api()
        except Exception as e:
            logging.warning(f"Falha ao inicializar provedores de IA: {e}. Usando mock.")
            self.mode = None
            self.enabled = True

        # Inicia heartbeat se HealthMonitor disponível
        self._start_heartbeat()

    def _load_config(self) -> None:
        """Carrega configuração do config.json."""
        try:
            config_path = Path("config.json")
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            else:
                logging.debug("config.json não encontrado, usando configuração padrão")
        except Exception as e:
            logging.warning(f"Erro ao carregar config.json: {e}")
            self.config = {}

    def _start_heartbeat(self) -> None:
        """Inicia thread de heartbeat se HealthMonitor disponível."""
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
            logging.info(
                "🫀 Heartbeat do módulo '%s' iniciado (AIAnalyzer).", self.module_name
            )

    def _heartbeat_loop(self) -> None:
        """Envia heartbeat periódico para o HealthMonitor."""
        interval = 30
        if app_config is not None:
            interval = getattr(app_config, "HEALTH_CHECK_INTERVAL", 30)
        interval = max(5, min(interval, 60))

        while not self._hb_stop.is_set():
            try:
                if self.health_monitor is not None:
                    self.health_monitor.heartbeat(self.module_name)
            except Exception:
                pass
            
            for _ in range(int(interval * 10)):
                if self._hb_stop.is_set():
                    break
                time.sleep(0.1)

    def _initialize_api(self) -> None:
        """Inicializa provedores de IA com fallback controlado."""
        ai_cfg = self.config.get("ai", {})
        if not isinstance(ai_cfg, dict):
            ai_cfg = {}
            
        provider = ai_cfg.get("provider", "groq")
        provider_fallbacks = ai_cfg.get("provider_fallbacks", [])
        if not isinstance(provider_fallbacks, list):
            provider_fallbacks = []

        providers_tested: List[str] = []

        # PROVIDER: GROQ
        if provider == "groq":
            providers_tested.append("groq")
            if self._try_initialize_groq(ai_cfg):
                return

            if not provider_fallbacks:
                logging.info(
                    "🔧 Groq falhou e nenhum fallback configurado. Ativando modo MOCK."
                )
                self._activate_mock_mode()
                return
            else:
                logging.info(
                    f"🔄 Groq falhou, tentando fallbacks configurados: {provider_fallbacks}"
                )

        # FALLBACKS
        for fallback_provider in provider_fallbacks:
            if fallback_provider in providers_tested:
                continue

            if fallback_provider == "openai":
                providers_tested.append("openai")
                if self._try_initialize_openai():
                    return

            elif fallback_provider == "dashscope":
                providers_tested.append("dashscope")
                if self._try_initialize_dashscope():
                    return

        # PROVIDER PADRÃO: OPENAI
        if provider != "groq" and "openai" not in providers_tested:
            if self._try_initialize_openai():
                return

        # FALLBACK FINAL: MOCK
        self._activate_mock_mode()

    def _try_initialize_groq(self, ai_cfg: Dict[str, Any]) -> bool:
        """Tenta inicializar Groq. Retorna True se sucesso."""
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key is None and app_config is not None:
            groq_key = getattr(app_config, "GROQ_API_KEY", None)

        if not OPENAI_AVAILABLE or not groq_key or _OpenAI is None:
            logging.warning("Groq indisponível: SDK ou chave não encontrada")
            return False

        groq_cfg = ai_cfg.get("groq", {})
        if not isinstance(groq_cfg, dict):
            groq_cfg = {}
            
        groq_base_url = groq_cfg.get("base_url", "https://api.groq.com/openai/v1")

        if not groq_key.startswith("gsk_"):
            logging.warning(
                "⚠️ GROQ_API_KEY suspeita (não começa com 'gsk_'). Tentando mesmo assim..."
            )

        self.base_url = groq_base_url

        # Cliente síncrono
        try:
            self.client = _OpenAI(api_key=groq_key, base_url=self.base_url)
        except Exception as e:
            logging.error(f"Erro ao criar cliente Groq síncrono: {e}")
            return False

        # Cliente assíncrono
        if ASYNC_OPENAI_AVAILABLE and _AsyncOpenAI is not None:
            try:
                self.client_async = _AsyncOpenAI(api_key=groq_key, base_url=self.base_url)
            except Exception as e:
                logging.warning(f"Erro ao criar cliente Groq assíncrono: {e}")

        logging.info("🔧 Groq client configurado | base_url=%s", self.base_url)

        # Lista de modelos para testar
        models = _models_from_cfg(groq_cfg)
        self._groq_model_candidates = models

        if not models:
            logging.warning("Nenhum modelo Groq configurado em ai.groq.model")
            return False

        # Testa cada modelo
        last_err: Optional[Exception] = None
        selected: Optional[str] = None

        for m in models:
            try:
                if self.client is None:
                    break
                messages: List[ChatMessage] = [{"role": "user", "content": "ping"}]
                self.client.chat.completions.create(
                    model=m,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=0,
                    max_tokens=1,
                    timeout=10,
                )
                selected = m
                break
            except Exception as e:
                last_err = e
                if _is_model_decommissioned_error(e):
                    logging.warning(f"⚠️ Modelo Groq descontinuado: {m}")
                else:
                    logging.warning(f"⚠️ Ping falhou no modelo Groq {m}: {e}")

        if not selected:
            logging.error(f"❌ Groq sem modelo válido. Último erro: {last_err}")
            return False

        self.model_name = selected
        self.mode = "groq"
        self.enabled = True
        
        logging.info(
            f"🚀 GroqCloud ATIVO | Modelo: {self.model_name} | "
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
            
        return True

    def _try_initialize_openai(self) -> bool:
        """Tenta inicializar OpenAI. Retorna True se sucesso."""
        if not OPENAI_AVAILABLE or _OpenAI is None:
            logging.warning("OpenAI SDK não disponível")
            return False
            
        try:
            self.client = _OpenAI()
            if ASYNC_OPENAI_AVAILABLE and _AsyncOpenAI is not None:
                self.client_async = _AsyncOpenAI()
            self.mode = "openai"
            self.enabled = True
            logging.info("🔧 OpenAI client configurado")
            
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
        """Tenta inicializar DashScope. Retorna True se sucesso."""
        if not DASHSCOPE_AVAILABLE or _dashscope is None:
            logging.warning("DashScope SDK não disponível")
            return False
            
        token = os.getenv("DASHSCOPE_API_KEY")
        if token is None and app_config is not None:
            token = getattr(app_config, "DASHSCOPE_API_KEY", None)

        if not token:
            logging.warning("DashScope: chave não encontrada")
            return False

        try:
            _dashscope.api_key = token
            self.mode = "dashscope"
            self.enabled = True
            logging.info("🔧 DashScope configurado")
            
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

    def _activate_mock_mode(self) -> None:
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

        prompt = "Ping. Responda APENAS: OK"
        ok = False  # Inicializa como False
        
        try:
            if self.mode in ("openai", "groq") and self.client is not None:
                messages: List[ChatMessage] = [
                    {"role": "system", "content": "Responda APENAS com a palavra OK, nada mais."},
                    {"role": "user", "content": prompt},
                ]
                r = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=5,  # Reduzido para forçar resposta curta
                    temperature=0.0,
                    timeout=10,
                )
                content = (r.choices[0].message.content or "").strip().upper()
                # Aceita variações: "OK", "OK.", "OK!", etc.
                ok = "OK" in content and len(content) < 10

                if ok and self.mode == "groq":
                    logging.debug("✅ Groq ping OK")
                elif not ok:
                    # CORREÇÃO: Incrementar failures quando resposta inválida
                    self.connection_failed_count += 1
                    logging.warning(
                        f"⚠️ Ping respondeu mas conteúdo inesperado: '{content[:50]}'"
                    )

            elif self.mode == "dashscope" and _Generation is not None:
                messages: List[ChatMessage] = [
                    {"role": "system", "content": "Responda APENAS com a palavra OK."},
                    {"role": "user", "content": prompt},
                ]
                r = _Generation.call(
                    model=self.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    result_format="message",
                    max_tokens=5,
                    temperature=0.0,
                    timeout=10,
                )
                content = _extract_dashscope_text(r).upper()
                ok = "OK" in content and len(content) < 10
                if not ok:
                    self.connection_failed_count += 1
            else:
                ok = True  # Mock sempre OK

        except Exception as e:
            self.connection_failed_count += 1
            logging.warning(
                f"Falha no ping da IA [{self.mode}] ({self.connection_failed_count}): {e}"
            )
            ok = False

        # Log estruturado
        try:
            if ok:
                self.slog.info("ai_ping_ok", mode=self.mode or "mock")
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
    # EXTRAÇÃO DE DADOS
    # ====================================================================

    def _extract_orderbook_data(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai dados de orderbook de múltiplas fontes possíveis."""
        candidates = [
            event_data.get("orderbook_data"),
            event_data.get("spread_metrics"),
            (event_data.get("contextual_snapshot") or {}).get("orderbook_data"),
            (event_data.get("contextual") or {}).get("orderbook_data"),
        ]

        for i, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                continue

            has_depth = (
                candidate.get("bid_depth_usd") is not None
                or candidate.get("ask_depth_usd") is not None
            )

            if has_depth:
                bid_usd = float(candidate.get("bid_depth_usd", 0) or 0)
                ask_usd = float(candidate.get("ask_depth_usd", 0) or 0)

                if bid_usd > 0 and ask_usd > 0:
                    logging.debug(
                        f"✅ Orderbook extraído da fonte #{i}: bid=${bid_usd:,.0f}, ask=${ask_usd:,.0f}"
                    )
                    return candidate
                else:
                    logging.debug(
                        f"⚠️ Fonte #{i} tem dados zerados (bid=${bid_usd}, ask=${ask_usd})"
                    )

        logging.warning("⚠️ Nenhuma fonte de orderbook válida encontrada")
        return {}

    # ====================================================================
    # PROMPT BUILDER
    # ====================================================================

    def _get_system_prompt(self) -> str:
        """System prompt: compacto se compressão ativa, senão legado."""
        ai_cfg = self.config.get("ai", {})
        if not isinstance(ai_cfg, dict):
            ai_cfg = {}

        # Se compressão profunda ativa, usar prompt compacto com dicionário de chaves
        if (
            getattr(self, '_compression_enabled', False)
            and _COMPRESSED_SYSTEM_PROMPT is not None
        ):
            return _COMPRESSED_SYSTEM_PROMPT

        # Fallback para prompts originais
        if ai_cfg.get("prompt_style") == "legacy":
            return SYSTEM_PROMPT_LEGACY
        return SYSTEM_PROMPT

    def _render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Renderiza template com Jinja2 se disponível."""
        tmpl_str = ORDERBOOK_TEMPLATE if template_name == "orderbook" else DEFAULT_TEMPLATE

        if JINJA_AVAILABLE and _jinja_env is not None:
            try:
                tmpl = _jinja_env.from_string(tmpl_str)
                return tmpl.render(**context)
            except Exception as e:
                logging.error(f"Erro ao renderizar template Jinja2: {e}")

        return self._render_template_fallback(template_name, context)

    def _render_template_fallback(
        self, template_name: str, context: Dict[str, Any]
    ) -> str:
        """Renderiza template sem Jinja2."""
        if template_name == "orderbook":
            orderbook_warning = ""
            if not context.get("is_orderbook_valid", True):
                orderbook_warning = (
                    "\n🔴 ORDERBOOK INDISPONÍVEL - "
                    "Use APENAS métricas de fluxo (net_flow, flow_imbalance, tick_rule)\n"
                )

            return (
                f"🧠 **Análise Institucional – {context['ativo']} | {context['tipo_evento']}**\n\n"
                f"📝 Descrição: {context['descricao']}\n"
                f"{context['ob_str']}{context['ml_str']}{context['vp_str']}{context['order_flow_str']}\n\n"
                f"📈 Multi-Timeframes\n{context['multi_tf_str']}\n\n"
                f"⏳ Memória de eventos\n{context['memoria_str']}\n\n"
                f"📉 Probabilidade Histórica\n"
                f"   Long={context['prob_long']} | Short={context['prob_short']} | Neutro={context['prob_neutral']}\n\n"
                "🎯 Tarefa\n"
                'CRÍTICO: Se dados estiverem marcados como "Indisponível" ou "⚠️", NÃO os use.\n'
                f"{orderbook_warning}"
                "Foque em identificar regiões importantes e sugerir entrada/invalidação se houver clareza.\n"
            )
        else:
            return (
                f"🧠 **Análise Institucional – {context['ativo']} | {context['tipo_evento']}**\n\n"
                f"📝 Descrição: {context['descricao']}\n\n"
                f"   Preço: {context['preco_fmt']}\n"
                f"   Delta: {context['delta_line']}\n"
                f"   Volume: {context['vol_line']}\n"
                f"{context['ml_str']}{context['vp_str']}{context['order_flow_str']}\n\n"
                f"📈 Multi-Timeframes\n{context['multi_tf_str']}\n\n"
                f"⏳ Memória de eventos\n{context['memoria_str']}\n\n"
                f"📉 Probabilidade Histórica\n"
                f"   Long={context['prob_long']} | Short={context['prob_short']} | Neutro={context['prob_neutral']}\n\n"
                "🎯 Tarefa\nUse APENAS dados explicitamente fornecidos.\n"
            )

    def _create_prompt(self, event_data: Dict[str, Any]) -> str:
        """
        Cria prompt para IA.
        
        Prioridade:
        1. Compressão profunda (se habilitada) — reduz ~70% dos tokens
        2. Prompt estruturado (ai_payload do builder)
        3. Prompt legado (fallback)
        """
        # ========================================
        # MODO 1: COMPRESSÃO PROFUNDA (prioridade)
        # ========================================
        if (
            getattr(self, '_compression_enabled', False)
            and _optimize_deep_for_ai is not None
        ):
            try:
                _deep_fn = _optimize_deep_for_ai

                # ============================================================
                # EXTRAIR raw_event EXPLICITAMENTE
                # ============================================================
                # event_data tem esta estrutura:
                #   event_data = {
                #     "tipo_evento": "...",
                #     "symbol": "...",
                #     "raw_event": {          ← ESTE contém TUDO (60KB+)
                #       "raw_event": {...},   ← dados brutos aninhados
                #       "contextual_snapshot": {...},
                #       "fluxo_continuo": {...},
                #       "multi_tf": {...},
                #       "orderbook_data": {...},
                #       etc.
                #     },
                #     "ai_payload": {         ← ESTE é filtrado (3KB) - NÃO USAR
                #       "_v": 2,
                #       "quant_model": {...},
                #       etc.
                #     }
                #   }
                #
                # O compressor PRECISA do raw_event, NÃO do ai_payload.
                
                source_for_compression = None
                compression_source_name = "unknown"
                use_ai_payload_direct = False
                
                # ============================================================
                # DECISÃO DE SOURCE PARA COMPRESSÃO
                # ============================================================
                # Prioridade:
                #   1. ai_payload v2 DIRETO (já otimizado, ~3KB → skip rebuild)
                #   2. raw_event (dados brutos ~60KB → compressão completa)
                #   3. fallback (evento sem ai_payload)
                # ============================================================
                
                ai_p = event_data.get("ai_payload")
                
                # Prioridade 1: Se ai_payload v2 existe, usar DIRETO
                # Ele já foi comprimido por build_ai_input() → compress_payload()
                # Não precisa rebuild → recompress (ciclo inútil)
                if isinstance(ai_p, dict) and ai_p.get("_v") == 2 and len(ai_p) > 3:
                    use_ai_payload_direct = True
                    compression_source_name = "ai_payload_v2_direct"
                    logging.debug(
                        "COMPRESSION_SOURCE using ai_payload v2 direct, keys=%d, bytes=%d",
                        len(ai_p),
                        len(json.dumps(ai_p, ensure_ascii=False).encode("utf-8")),
                    )
                else:
                    # Prioridade 2: raw_event disponível → compressão completa
                    raw_evt = event_data.get("raw_event")
                    if isinstance(raw_evt, dict) and len(raw_evt) > 5:
                        source_for_compression = dict(raw_evt)
                        compression_source_name = "raw_event"
                        for meta_key in ("tipo_evento", "symbol", "janela_numero", "epoch_ms", "data_context"):
                            if meta_key not in source_for_compression and meta_key in event_data:
                                source_for_compression[meta_key] = event_data[meta_key]
                        
                        logging.debug(
                            "COMPRESSION_SOURCE raw_event found, keys=%d, has_contextual=%s, has_multi_tf=%s",
                            len(source_for_compression),
                            "contextual_snapshot" in source_for_compression,
                            "multi_tf" in source_for_compression,
                        )
                    else:
                        # Prioridade 3: fallback — ai_payload não-v2 com rebuild
                        if isinstance(ai_p, dict) and len(ai_p) > 3:
                            source_for_compression = self._rebuild_from_ai_payload(ai_p, event_data)
                            compression_source_name = "rebuilt_from_ai_payload"
                        else:
                            source_for_compression = {
                                k: v for k, v in event_data.items()
                                if k != "ai_payload"
                            }
                            compression_source_name = "event_data_sans_ai_payload"
                        logging.warning(
                            "COMPRESSION_SOURCE raw_event not found (type=%s), using %s",
                            type(event_data.get("raw_event")).__name__,
                            compression_source_name,
                        )

                # ============================================================
                # COMPRESSÃO ou USO DIRETO
                # ============================================================
                if use_ai_payload_direct and isinstance(ai_p, dict):
                    # ai_payload v2 já está otimizado — converter direto para
                    # formato comprimido sem passar pelo _deep_fn
                    compressed = self._v2_to_compressed_prompt(ai_p)

                    # Injetar multi-timeframe do event_data original
                    # (não está no ai_payload v2 mas é crítico para análise)
                    _raw_for_tf = event_data.get("raw_event")
                    if isinstance(_raw_for_tf, dict):
                        _mtf = _raw_for_tf.get("multi_tf")
                        if not isinstance(_mtf, dict):
                            # Tentar dentro de contextual_snapshot
                            _cs = _raw_for_tf.get("contextual_snapshot")
                            if isinstance(_cs, dict):
                                _mtf = _cs.get("multi_tf")
                        
                        if isinstance(_mtf, dict) and _mtf:
                            tf_compact = {}
                            for tf_key, tf_data in _mtf.items():
                                if isinstance(tf_data, dict):
                                    tf_compact[tf_key] = {
                                        "trend": tf_data.get("tendencia"),
                                        "rsi": tf_data.get("rsi_short"),
                                        "macd": tf_data.get("macd"),
                                        "macd_s": tf_data.get("macd_signal"),
                                        "adx": tf_data.get("adx"),
                                        "regime": tf_data.get("regime"),
                                        "mme21": tf_data.get("mme_21"),
                                        "atr": tf_data.get("atr"),
                                    }
                                    # Limpar None
                                    tf_compact[tf_key] = {
                                        k: v for k, v in tf_compact[tf_key].items()
                                        if v is not None
                                    }
                            if tf_compact:
                                compressed["tf"] = tf_compact
                                logging.debug(
                                    "MULTI_TF_INJECTED timeframes=%s",
                                    list(tf_compact.keys()),
                                )
                else:
                    compressed = _deep_fn(
                        source_for_compression,
                        section_cache=getattr(self, '_section_cache', None),
                    )

                # Verificar se compressão extraiu dados suficientes
                essential_keys = {"price", "ob", "tf", "flow"}
                found_keys = set(compressed.keys())
                missing_essential = essential_keys - found_keys
                
                if missing_essential:
                    logging.warning(
                        "COMPRESSION_INCOMPLETE missing_keys=%s found_keys=%s source=%s",
                        missing_essential,
                        list(found_keys)[:10],
                        compression_source_name,
                    )

                # Injetar quant_model do ai_payload se não veio da compressão
                if "quant" not in compressed:
                    ai_p = event_data.get("ai_payload")
                    if isinstance(ai_p, dict):
                        quant = ai_p.get("quant_model")
                        if isinstance(quant, dict) and quant:
                            try:
                                from src.utils.ai_payload_optimizer import PrecisionRounder
                                _rr = PrecisionRounder.r
                                quant_compressed = {
                                    k: v for k, v in {
                                        "pu": _rr(quant.get("model_probability_up"), "ratio"),
                                        "pd": _rr(quant.get("model_probability_down"), "ratio"),
                                        "ab": quant.get("action_bias"),
                                        "cs": _rr(quant.get("confidence_score"), "ratio"),
                                        "sent": quant.get("model_sentiment"),
                                    }.items() if v is not None
                                }
                                if quant_compressed:
                                    compressed["quant"] = quant_compressed
                            except ImportError:
                                pass

                # Injetar regime_analysis do ai_payload
                if "regime" not in compressed:
                    ai_p = event_data.get("ai_payload")
                    if isinstance(ai_p, dict):
                        regime = ai_p.get("regime_analysis")
                        if isinstance(regime, dict) and regime:
                            regime_c = {
                                k: v for k, v in {
                                    "mr": regime.get("market_regime"),
                                    "cr": regime.get("correlation_regime"),
                                    "vr": regime.get("volatility_regime"),
                                    "rc": regime.get("regime_confidence"),
                                    "pd": regime.get("primary_driver"),
                                }.items() if v is not None
                            }
                            if regime_c:
                                compressed["regime"] = regime_c

                # Injetar signal_metadata
                if "sig" not in compressed:
                    ai_p = event_data.get("ai_payload")
                    if isinstance(ai_p, dict):
                        sig = ai_p.get("signal_metadata")
                        if isinstance(sig, dict) and sig:
                            sig_c = {
                                k: v for k, v in {
                                    "type": sig.get("type"),
                                    "battle": sig.get("battle_result"),
                                    "desc": sig.get("description"),
                                }.items() if v is not None
                            }
                            if sig_c:
                                compressed["sig"] = sig_c

                prompt = json.dumps(
                    compressed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                logging.info(
                    "DEEP_COMPRESSION_ACTIVE prompt_len=%d chars (~%d tokens) source=%s keys=%s",
                    len(prompt),
                    int(len(prompt) / 3.5),
                    compression_source_name,
                    list(compressed.keys())[:15],
                )
                return prompt

            except Exception as e:
                logging.error(
                    f"Deep compression failed, falling back to structured: {e}",
                    exc_info=True,
                )
        
        # ========================================
        # MODO 2: PROMPT ESTRUTURADO (ai_payload)
        # ========================================
        ai_payload = event_data.get("ai_payload")
        if ai_payload and isinstance(ai_payload, dict):
            try:
                logging.debug("Usando ai_payload estruturado para montar o prompt")
                return self._build_structured_prompt(ai_payload)
            except Exception as e:
                logging.error(
                    f"Erro ao construir prompt estruturado: {e}. Usando fallback legado.",
                    exc_info=True,
                )
        
        # ========================================
        # MODO 3: PROMPT LEGADO (fallback final)
        # ========================================
        return self._create_legacy_prompt(event_data)

    @staticmethod
    def _rebuild_from_ai_payload(
        ai_payload: Dict[str, Any], event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reconstrói um pseudo-evento a partir do ai_payload para o compressor.
        Usado quando raw_event não está disponível no event_data.
        Mapeia as chaves do builder v2 para as chaves que o compressor espera.
        """
        rebuilt: Dict[str, Any] = {}

        # Metadados
        rebuilt["symbol"] = ai_payload.get("symbol") or event_data.get("symbol")
        rebuilt["epoch_ms"] = ai_payload.get("epoch_ms") or event_data.get("epoch_ms")
        rebuilt["tipo_evento"] = event_data.get("tipo_evento", "ANALYSIS_TRIGGER")
        rebuilt["janela_numero"] = event_data.get("janela_numero")
        rebuilt["data_context"] = event_data.get("data_context", "real_time")

        # Price → contextual_snapshot.ohlc
        price_ctx = ai_payload.get("price_context") or {}
        ohlc = price_ctx.get("ohlc") or {}
        rebuilt["contextual_snapshot"] = {
            "ohlc": ohlc,
            "poc_price": (price_ctx.get("volume_profile_daily") or {}).get("poc"),
            "volume_total": None,
            "volume_compra": None,
            "volume_venda": None,
        }

        # Flow → fluxo_continuo
        flow_ctx = ai_payload.get("flow_context") or {}
        if flow_ctx:
            rebuilt["fluxo_continuo"] = {
                "cvd": flow_ctx.get("cvd_accumulated"),
                "order_flow": {
                    "net_flow_1m": flow_ctx.get("net_flow"),
                    "flow_imbalance": flow_ctx.get("flow_imbalance"),
                    "aggressive_buy_pct": flow_ctx.get("aggressive_buyers"),
                    "aggressive_sell_pct": flow_ctx.get("aggressive_sellers"),
                },
                "tipo_absorcao": flow_ctx.get("absorption_type"),
            }

        # Orderbook
        ob_ctx = ai_payload.get("orderbook_context") or {}
        if ob_ctx:
            rebuilt["orderbook_data"] = {
                "bid_depth_usd": ob_ctx.get("bid_depth_usd"),
                "ask_depth_usd": ob_ctx.get("ask_depth_usd"),
                "imbalance": ob_ctx.get("imbalance"),
                "pressure": ob_ctx.get("market_impact_score"),
                "depth_metrics": ob_ctx.get("depth_metrics"),
            }

        # Macro → market_context + market_environment
        macro = ai_payload.get("macro_context") or {}
        if macro:
            regime = macro.get("regime") or {}
            rebuilt["market_context"] = {
                "trading_session": macro.get("session"),
                "session_phase": macro.get("phase"),
            }
            rebuilt["market_environment"] = {
                "market_structure": regime.get("structure"),
                "trend_direction": regime.get("trend"),
                "risk_sentiment": regime.get("sentiment"),
                "volatility_regime": (price_ctx.get("volatility") or {}).get("volatility_regime"),
            }
            # Correlações
            corr = macro.get("correlations") or {}
            if corr:
                rebuilt["market_environment"]["correlation_spy"] = corr.get("sp500")
                rebuilt["market_environment"]["correlation_dxy"] = corr.get("dxy")

        # Technical indicators → simular multi_tf (apenas 1h)
        tech = ai_payload.get("technical_indicators") or {}
        if tech:
            macd_data = tech.get("macd") or {}
            rebuilt["multi_tf"] = {
                "1h": {
                    "rsi_short": tech.get("rsi"),
                    "macd": macd_data.get("line"),
                    "macd_signal": macd_data.get("signal"),
                    "adx": tech.get("adx"),
                }
            }

        # Volume Profile
        vp_daily = (price_ctx.get("volume_profile_daily") or {})
        if vp_daily.get("poc") is not None:
            rebuilt["historical_vp"] = {
                "daily": {
                    "poc": vp_daily.get("poc"),
                    "vah": vp_daily.get("vah"),
                    "val": vp_daily.get("val"),
                    "hvns": vp_daily.get("hvns_nearby"),
                    "lvns": vp_daily.get("lvns_nearby"),
                    "status": "success",
                }
            }

        # Cross asset → ml_features.cross_asset
        cross = ai_payload.get("cross_asset_context") or {}
        if cross:
            rebuilt["ml_features"] = {
                "cross_asset": {
                    "btc_eth_corr_7d": (cross.get("btc_eth_correlations") or {}).get("short_term_7d"),
                    "btc_eth_corr_30d": (cross.get("btc_eth_correlations") or {}).get("long_term_30d"),
                    "btc_dxy_corr_30d": (cross.get("btc_dxy_correlations") or {}).get("medium_term_30d"),
                    "btc_dxy_corr_90d": (cross.get("btc_dxy_correlations") or {}).get("long_term_90d"),
                    "dxy_return_5d": (cross.get("dxy_momentum") or {}).get("return_5d"),
                    "dxy_return_20d": (cross.get("dxy_momentum") or {}).get("return_20d"),
                    "correlation_regime": "DECORRELATED",
                }
            }

        # Quant model
        quant = ai_payload.get("quant_model") or {}
        if quant:
            rebuilt["quant_model"] = quant

        # Signal metadata
        sig = ai_payload.get("signal_metadata") or {}
        if sig:
            rebuilt["resultado_da_batalha"] = sig.get("battle_result")

        # Regime analysis
        regime_a = ai_payload.get("regime_analysis") or {}
        if regime_a:
            rebuilt["regime_analysis"] = regime_a

        # Limpar Nones
        return {k: v for k, v in rebuilt.items() if v is not None}

    def _v2_to_compressed_prompt(self, ai_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converte ai_payload v2 diretamente para formato de prompt comprimido,
        SEM passar pelo ciclo rebuild → recompress.
        
        O ai_payload v2 já contém todos os dados necessários organizados
        por build_ai_input() + compress_payload().
        """
        compressed: Dict[str, Any] = {}
        
        # Symbol e contexto
        compressed["s"] = ai_payload.get("symbol", "BTCUSDT")
        
        # Timestamp
        epoch = ai_payload.get("epoch_ms")
        if epoch:
            compressed["t"] = epoch
        
        # Evento/macro
        macro = ai_payload.get("macro_context")
        if isinstance(macro, dict):
            compressed["ctx"] = {
                "session": macro.get("session"),
                "phase": macro.get("phase"),
            }
            regime = macro.get("regime")
            if isinstance(regime, dict):
                compressed["ctx"]["structure"] = regime.get("structure")
                compressed["ctx"]["trend"] = regime.get("trend")
        
        # Price context
        pc = ai_payload.get("price_context")
        if isinstance(pc, dict):
            price_data: Dict[str, Any] = {}
            ohlc = pc.get("ohlc")
            if isinstance(ohlc, dict):
                price_data["o"] = ohlc.get("open")
                price_data["h"] = ohlc.get("high")
                price_data["l"] = ohlc.get("low")
                price_data["c"] = ohlc.get("close")
                price_data["vwap"] = ohlc.get("vwap")
            price_data["cur"] = pc.get("current_price")
            
            # Volume profile
            vp = pc.get("volume_profile_daily")
            if isinstance(vp, dict):
                price_data["vp"] = {
                    "poc": vp.get("poc"),
                    "vah": vp.get("vah"),
                    "val": vp.get("val"),
                }
                hvns = vp.get("hvns_nearby")
                if isinstance(hvns, list) and hvns:
                    price_data["vp"]["hvn"] = hvns[:5]
                lvns = vp.get("lvns_nearby")
                if isinstance(lvns, list) and lvns:
                    price_data["vp"]["lvn"] = lvns[:5]
            
            # Price action
            pa = pc.get("price_action")
            if isinstance(pa, dict):
                price_data["pa"] = {
                    "range_pct": pa.get("candle_range_pct"),
                    "body_pct": pa.get("candle_body_pct"),
                    "close_pos": pa.get("close_position"),
                }
            
            # Volatility
            vol = pc.get("volatility")
            if isinstance(vol, dict):
                price_data["vol_regime"] = vol.get("volatility_regime")
            
            compressed["price"] = price_data
        
        # Orderbook
        ob = ai_payload.get("orderbook_context")
        if isinstance(ob, dict):
            compressed["ob"] = {
                "bid": ob.get("bid_depth_usd"),
                "ask": ob.get("ask_depth_usd"),
                "imb": ob.get("imbalance"),
                "impact": ob.get("market_impact_score"),
                "walls": ob.get("walls_detected"),
            }
            dm = ob.get("depth_metrics")
            if isinstance(dm, dict):
                compressed["ob"]["d_imb"] = dm.get("depth_imbalance")
        
        # Flow context
        fc = ai_payload.get("flow_context")
        if isinstance(fc, dict):
            compressed["flow"] = {
                "net": fc.get("net_flow"),
                "cvd": fc.get("cvd_accumulated"),
                "imb": fc.get("flow_imbalance"),
                "ag_buy": fc.get("aggressive_buyers"),
                "ag_sell": fc.get("aggressive_sellers"),
                "abs": fc.get("absorption_type"),
                "clusters": fc.get("liquidity_clusters_count"),
            }
        
        # Technical indicators
        ti = ai_payload.get("technical_indicators")
        if isinstance(ti, dict):
            compressed["ind"] = {
                "rsi": ti.get("rsi"),
                "adx": ti.get("adx"),
            }
            macd = ti.get("macd")
            if isinstance(macd, dict):
                compressed["ind"]["macd"] = macd.get("line")
                compressed["ind"]["macd_s"] = macd.get("signal")
                compressed["ind"]["macd_h"] = macd.get("histogram")
        
        # Cross-asset
        ca = ai_payload.get("cross_asset_context")
        if isinstance(ca, dict):
            cross: Dict[str, Any] = {}
            btc_eth = ca.get("btc_eth_correlations")
            if isinstance(btc_eth, dict):
                cross["eth_corr"] = btc_eth.get("short_term_7d")
            btc_dxy = ca.get("btc_dxy_correlations")
            if isinstance(btc_dxy, dict):
                cross["dxy_corr"] = btc_dxy.get("medium_term_30d")
            dxy_mom = ca.get("dxy_momentum")
            if isinstance(dxy_mom, dict):
                cross["dxy_5d"] = dxy_mom.get("return_5d")
                cross["dxy_20d"] = dxy_mom.get("return_20d")
            if cross:
                compressed["cross"] = cross
        
        # Quant model
        qm = ai_payload.get("quant_model")
        if isinstance(qm, dict):
            compressed["quant"] = {
                "p_up": qm.get("model_probability_up"),
                "p_down": qm.get("model_probability_down"),
                "conf": qm.get("confidence_score"),
                "bias": qm.get("action_bias"),
            }
        
        # Regime analysis
        ra = ai_payload.get("regime_analysis")
        if isinstance(ra, dict):
            compressed["regime"] = {
                "market": ra.get("market_regime"),
                "corr": ra.get("correlation_regime"),
                "vol": ra.get("volatility_regime"),
                "driver": ra.get("primary_driver"),
                "risk": ra.get("risk_score"),
            }
        
        # Signal metadata
        # Signal metadata
        # Multi-timeframe
        mtf = ai_payload.get("multi_tf")
        if isinstance(mtf, dict) and mtf:
            compressed["tf"] = mtf  # Já vem compactado do builder

        sm = ai_payload.get("signal_metadata")
        if isinstance(sm, dict):
            compressed["sig"] = {
                "type": sm.get("type"),
                "battle": sm.get("battle_result"),
                "desc": sm.get("description"),
            }
        
        # Limpar None values
        compressed = self._clean_none_recursive(compressed)
        
        return compressed

    @staticmethod
    def _clean_none_recursive(d: Any) -> Any:
        """Remove chaves com valor None recursivamente."""
        if isinstance(d, dict):
            return {
                k: AIAnalyzer._clean_none_recursive(v)
                for k, v in d.items()
                if v is not None
            }
        if isinstance(d, list):
            return [AIAnalyzer._clean_none_recursive(i) for i in d if i is not None]
        return d

    def _create_legacy_prompt(self, event_data: Dict[str, Any]) -> str:
        """Cria prompt usando lógica legada."""
        tipo_evento = event_data.get("tipo_evento", "N/A")
        ativo = event_data.get("ativo") or event_data.get("symbol") or "N/A"
        descricao = event_data.get("descricao", "Sem descrição.")

        ob_data = self._extract_orderbook_data(event_data)
        bid_usd_raw = float(ob_data.get("bid_depth_usd", 0) or 0)
        ask_usd_raw = float(ob_data.get("ask_depth_usd", 0) or 0)
        is_orderbook_valid = bid_usd_raw > 0 and ask_usd_raw > 0

        if not is_orderbook_valid:
            logging.warning(
                f"⚠️ Orderbook INVÁLIDO para prompt: bid=${bid_usd_raw}, ask=${ask_usd_raw}"
            )

        delta_raw = event_data.get("delta")
        volume_total_raw = event_data.get("volume_total")
        volume_compra_raw = event_data.get("volume_compra")
        volume_venda_raw = event_data.get("volume_venda")

        delta = float(delta_raw) if delta_raw is not None else None
        volume_total = float(volume_total_raw) if volume_total_raw is not None else None
        volume_compra = float(volume_compra_raw) if volume_compra_raw is not None else None
        volume_venda = float(volume_venda_raw) if volume_venda_raw is not None else None

        if delta is not None and abs(delta) > 1.0:
            if (volume_compra == 0 and volume_venda == 0) or volume_total == 0:
                logging.warning(
                    f"⚠️ Inconsistência: delta={delta:.2f} mas volumes zerados. "
                    "Marcando volumes como indisponíveis."
                )
                volume_total = None

        preco = (
            event_data.get("preco_atual")
            or event_data.get("preco_fechamento")
            or (event_data.get("ohlc") or {}).get("close")
            or 0
        )

        multi_tf = (
            event_data.get("multi_tf")
            or (event_data.get("contextual_snapshot") or {}).get("multi_tf")
            or (event_data.get("contextual") or {}).get("multi_tf")
            or {}
        )
        multi_tf_str = (
            "\n".join(f"- {tf}: {v}" for tf, v in multi_tf.items())
            if multi_tf
            else "Indisponível."
        )

        memoria = event_data.get("event_history", [])
        if memoria:
            mem_lines = []
            for e in memoria:
                mem_delta = format_delta(e.get("delta", 0))
                mem_vol = format_large_number(e.get("volume_total", 0))
                mem_lines.append(
                    f"   - {e.get('timestamp')} | {e.get('tipo_evento')} "
                    f"{e.get('resultado_da_batalha')} (Δ={mem_delta}, Vol={mem_vol})"
                )
            memoria_str = "\n".join(mem_lines)
        else:
            memoria_str = "   Nenhum evento recente."

        conf = event_data.get("historical_confidence", {})
        prob_long = conf.get("long_prob", "Indisponível")
        prob_short = conf.get("short_prob", "Indisponível")
        prob_neutral = conf.get("neutral_prob", "Indisponível")

        vp = (
            (event_data.get("historical_vp") or {}).get("daily", {})
            or (event_data.get("contextual_snapshot") or {})
            .get("historical_vp", {})
            .get("daily", {})
            or {}
        )
        vp_str = ""
        if vp:
            poc_fmt = format_price(vp.get("poc", 0))
            val_fmt = format_price(vp.get("val", 0))
            vah_fmt = format_price(vp.get("vah", 0))
            vp_str = f"""
📊 Volume Profile (Diário)
   POC: ${poc_fmt} | VAL: ${val_fmt} | VAH: ${vah_fmt}
"""

        flow = (
            event_data.get("fluxo_continuo")
            or event_data.get("flow_metrics")
            or (event_data.get("contextual_snapshot") or {}).get("flow_metrics")
            or (event_data.get("contextual") or {}).get("flow_metrics")
            or {}
        )

        order_flow_str = self._build_order_flow_string(flow)
        ml_str = self._build_ml_string(event_data)

        if tipo_evento == "OrderBook" or "imbalance" in event_data:
            ob_str = self._build_orderbook_string(
                ob_data, bid_usd_raw, ask_usd_raw, is_orderbook_valid
            )

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

        vol_line = (
            "Indisponível"
            if volume_total is None
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

    def _build_order_flow_string(self, flow: Dict[str, Any]) -> str:
        """Constrói string de order flow."""
        if not isinstance(flow, dict) or not flow:
            return ""

        of = flow.get("order_flow", {})
        if not isinstance(of, dict) or not of:
            return ""

        try:
            buy_vol = of.get("buy_volume", 0)
            sell_vol = of.get("sell_volume", 0)
            bsr = of.get("buy_sell_ratio")

            has_volumes = buy_vol > 0 or sell_vol > 0

            if not has_volumes and bsr is not None and bsr > 0:
                logging.warning(
                    f"⚠️ CONTRADIÇÃO: buy/sell volumes zero mas ratio={bsr}. "
                    "Marcando ratio como indisponível."
                )
                bsr = None

            flow_lines = []

            nf1 = of.get("net_flow_1m")
            if nf1 is not None:
                flow_lines.append(f"   Net Flow 1m: {format_delta(nf1)}")

            fi = of.get("flow_imbalance")
            if fi is not None:
                flow_lines.append(f"   Flow Imbalance: {format_scientific(fi, 4)}")

            if bsr is not None:
                flow_lines.append(f"   Buy/Sell Ratio: {format_scientific(bsr, 2)}")

            if flow_lines:
                return "\n🚰 Fluxo de Ordens\n" + "\n".join(flow_lines) + "\n"

        except Exception as e:
            logging.error(f"Erro ao processar order_flow: {e}")

        return ""

    def _build_ml_string(self, event_data: Dict[str, Any]) -> str:
        """Constrói string de ML features."""
        ml = event_data.get("ml_features") or event_data.get("ml") or {}
        if not isinstance(ml, dict) or not ml:
            return ""

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
                return "\n📐 ML Features\n" + "\n".join(ml_lines) + "\n"
        except Exception:
            pass

        return ""

    def _build_orderbook_string(
        self,
        ob_data: Dict[str, Any],
        bid_usd_raw: float,
        ask_usd_raw: float,
        is_valid: bool,
    ) -> str:
        """Constrói string do orderbook."""
        if not is_valid:
            return f"""
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

        imbalance = ob_data.get("imbalance", 0)
        mid = ob_data.get("mid", 0)
        spread_pct = ob_data.get("spread_percent", 0)

        return f"""
📊 Evento OrderBook ✅

   Preço Mid: {format_price(mid)}
   Spread: {format_percent(spread_pct)}
   
   Profundidade (USD):
   • Bids: {format_large_number(bid_usd_raw)}
   • Asks: {format_large_number(ask_usd_raw)}
   
   Imbalance: {format_scientific(imbalance, 4)}
"""

    def _build_structured_prompt(self, payload: Dict[str, Any]) -> str:
        """Prompt compacto para uso com ai_payload_builder."""
        ai_cfg = self.config.get("ai", {})
        if isinstance(ai_cfg, dict) and ai_cfg.get("prompt_style") == "legacy":
            return self._build_structured_prompt_legacy(payload)

        meta = payload.get("signal_metadata") or {}
        price = payload.get("price_context") or {}
        flow = payload.get("flow_context") or {}
        ob = payload.get("orderbook_context") or {}
        macro = payload.get("macro_context") or {}
        quant = payload.get("quant_model") or {}

        symbol = payload.get("symbol") or payload.get("ativo") or meta.get("symbol") or "N/A"
        tipo = meta.get("type") or payload.get("tipo_evento") or "N/A"
        descricao = meta.get("description") or payload.get("descricao") or ""
        batalha = meta.get("battle_result") or payload.get("resultado_da_batalha") or "N/A"
        epoch_ms = (
            payload.get("epoch_ms")
            or payload.get("timestamp_ms")
            or payload.get("timestamp")
        )

        header_parts = [f"ativo={symbol}", f"tipo={tipo}"]
        if epoch_ms is not None:
            header_parts.append(f"epoch_ms={epoch_ms}")
        if batalha != "N/A":
            header_parts.append(f"batalha={batalha}")

        lines: List[str] = [" | ".join(header_parts)]
        if descricao:
            lines.append(f"descricao={descricao}")

        ohlc = price.get("ohlc") or {}
        cur = price.get("current_price")
        if cur is not None:
            lines.append(
                f"preco={format_price(cur)} ohlc=O{format_price(ohlc.get('open'))} "
                f"H{format_price(ohlc.get('high'))} L{format_price(ohlc.get('low'))} "
                f"C{format_price(ohlc.get('close'))}"
            )

        vp = price.get("volume_profile_daily") or {}
        vp_parts = [
            "vp_diario:",
            f"poc={format_price(vp.get('poc'))}",
            f"vah={format_price(vp.get('vah'))}",
            f"val={format_price(vp.get('val'))}",
        ]
        hvns = vp.get("hvns_nearby")
        if isinstance(hvns, list) and hvns:
            vp_parts.append("hvn_near=" + ",".join(format_price(x) for x in hvns[:5]))
        lvns = vp.get("lvns_nearby")
        if isinstance(lvns, list) and lvns:
            vp_parts.append("lvn_near=" + ",".join(format_price(x) for x in lvns[:5]))
        lines.append(" ".join(vp_parts))

        flow_bits = []
        if flow.get("net_flow") is not None:
            flow_bits.append(f"net_flow={format_delta(flow.get('net_flow'))}")
        if flow.get("cvd_accumulated") is not None:
            flow_bits.append(f"cvd={format_delta(flow.get('cvd_accumulated'))}")
        if flow.get("flow_imbalance") is not None:
            flow_bits.append(f"flow_imb={format_scientific(flow.get('flow_imbalance'), 4)}")
        if flow.get("aggressive_buyers") is not None:
            flow_bits.append(f"aggr_buy={format_percent(flow.get('aggressive_buyers'), 1)}")
        if flow.get("aggressive_sellers") is not None:
            flow_bits.append(f"aggr_sell={format_percent(flow.get('aggressive_sellers'), 1)}")
        if flow.get("absorption_type") is not None:
            flow_bits.append(f"absorption={flow.get('absorption_type')}")
        if flow_bits:
            lines.append("fluxo: " + " | ".join(flow_bits))

        ob_bits = []
        if ob.get("bid_depth_usd") is not None:
            ob_bits.append(f"bid_depth_usd={format_large_number(ob.get('bid_depth_usd'))}")
        if ob.get("ask_depth_usd") is not None:
            ob_bits.append(f"ask_depth_usd={format_large_number(ob.get('ask_depth_usd'))}")
        if ob.get("imbalance") is not None:
            ob_bits.append(f"imbalance={format_scientific(ob.get('imbalance'), 4)}")
        if ob.get("pressure") is not None:
            ob_bits.append(f"pressure={format_scientific(ob.get('pressure'), 4)}")
        if ob.get("spread_percent") is not None:
            ob_bits.append(f"spread={format_percent(ob.get('spread_percent'), 4)}")
        if ob_bits:
            lines.append("orderbook: " + " | ".join(ob_bits))

        macro_bits = []
        if macro.get("session") is not None:
            macro_bits.append(f"session={macro.get('session')}")
        if macro.get("phase") is not None:
            macro_bits.append(f"phase={macro.get('phase')}")
        regime = macro.get("regime") or {}
        if regime.get("structure") is not None:
            macro_bits.append(f"structure={regime.get('structure')}")
        if regime.get("trend") is not None:
            macro_bits.append(f"trend={regime.get('trend')}")
        if regime.get("sentiment") is not None:
            macro_bits.append(f"risk={regime.get('sentiment')}")
        if macro_bits:
            lines.append("macro: " + " | ".join(macro_bits))

        quant_bits = []
        if quant.get("action_bias") is not None:
            quant_bits.append(f"action_bias={quant.get('action_bias')}")
        if quant.get("confidence_score") is not None:
            quant_bits.append(
                f"confidence_score={format_percent(quant.get('confidence_score'), 1)}"
            )
        if quant.get("model_probability_up") is not None:
            quant_bits.append(
                f"prob_up={format_percent(quant.get('model_probability_up'), 1)}"
            )
        if quant_bits:
            lines.append("quant: " + " | ".join(quant_bits))

        lines.append("SAIDA: responda apenas com JSON (schema do SYSTEM prompt).")
        return "\n".join(lines)

    def _build_structured_prompt_legacy(self, payload: Dict[str, Any]) -> str:
        """Constrói prompt legado usando payload estruturado."""
        meta = payload.get("signal_metadata") or {}
        price = payload.get("price_context") or {}
        flow = payload.get("flow_context") or {}
        ob = payload.get("orderbook_context") or {}
        macro = payload.get("macro_context") or {}
        hist = payload.get("historical_stats") or {}
        quant = payload.get("quant_model") or {}

        symbol = payload.get("symbol") or meta.get("symbol") or "N/A"
        timestamp = payload.get("timestamp") or meta.get("timestamp") or "N/A"

        ohlc = price.get("ohlc") or {}
        current_price = format_price(price.get("current_price"))
        open_p = format_price(ohlc.get("open"))
        high_p = format_price(ohlc.get("high"))
        low_p = format_price(ohlc.get("low"))
        close_p = format_price(ohlc.get("close"))

        vp = price.get("volume_profile_daily") or {}
        poc = format_price(vp.get("poc"))
        vah = format_price(vp.get("vah"))
        val = format_price(vp.get("val"))

        lines: List[str] = []

        lines.append(f"Análise Institucional – {symbol}")
        lines.append(f"{timestamp} | Tipo: {meta.get('type', 'N/A')}")
        lines.append(f"Descrição: {meta.get('description', 'Sem descrição')}")
        lines.append(f"Resultado da Batalha: {meta.get('battle_result', 'N/A')}")
        lines.append("")

        lines.append("CONTEXTO DE PREÇO")
        lines.append(f"  • Preço Atual: {current_price}")
        lines.append(f"  • OHLC: O:{open_p} H:{high_p} L:{low_p} C:{close_p}")
        lines.append(f"  • VP Diário: POC {poc} | VAH {vah} | VAL {val}")
        lines.append("")

        net_flow = flow.get("net_flow")
        cvd_acc = flow.get("cvd_accumulated")
        if net_flow is not None or cvd_acc is not None:
            lines.append("CONTEXTO DE FLUXO")
            if net_flow is not None:
                lines.append(f"  • Net Flow (janela): {format_delta(net_flow)}")
            if cvd_acc is not None:
                lines.append(f"  • CVD acumulado: {format_delta(cvd_acc)}")
            lines.append("")

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

        if macro:
            lines.append("MACRO / REGIME")
            session = macro.get("session") or macro.get("session_name")
            if session:
                lines.append(f"  • Sessão: {session}")
            trends = macro.get("multi_timeframe_trends") or {}
            if trends:
                lines.append("  • Tendências multi-timeframe:")
                for tf, tr in trends.items():
                    val_trend = tr.get("tendencia") if isinstance(tr, dict) else tr
                    lines.append(f"    - {tf}: {val_trend}")
            lines.append("")

        if hist:
            lp = hist.get("long_prob")
            sp = hist.get("short_prob")
            np_ = hist.get("neutral_prob")
            lines.append("ESTATÍSTICA HISTÓRICA")
            lines.append(f"  • Probabilidades: Long={lp} | Short={sp} | Neutro={np_}")
            lines.append("")

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
            lines.append("")

        lines.append("═══════════════════════════════════════════════════════")
        lines.append("📋 TAREFA DA IA")
        lines.append("═══════════════════════════════════════════════════════")
        lines.append("")
        lines.append("1) USE A INTELIGÊNCIA QUANTITATIVA COMO BASE PRINCIPAL.")
        lines.append("2) CONFIRME OU INVALIDE com dados de fluxo e orderbook.")
        lines.append("3) SÓ CONTRARIE O VIÉS QUANTITATIVO se houver evidência MUITO FORTE.")
        lines.append("4) Defina região de entrada e zona de invalidação (se houver setup).")
        lines.append("5) Se dados conflitantes ou confiança baixa (<50%), recomende aguardar.")

        return "\n".join(lines)

    # ====================================================================
    # MÉTRICAS E LOGS
    # ====================================================================

    def _extract_list_counts(self, payload: Dict[str, Any]) -> Dict[str, int]:
        """Percorre dict/list procurando listas relevantes para medir tamanho."""
        counts: Dict[str, int] = {}
        keys_of_interest = {
            "clusters",
            "hvn_nodes",
            "lvn_nodes",
            "event_history",
            "signals",
            "windows",
            "orders",
        }

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(v, list) and (
                        k in keys_of_interest or k.endswith("_list")
                    ):
                        counts[k] = len(v)
                    elif isinstance(v, (dict, list)):
                        _walk(v)
            elif isinstance(node, list):
                for item in node:
                    if isinstance(item, (dict, list)):
                        _walk(item)

        _walk(payload)
        return counts

    def _log_payload_metrics(
        self, payload: Dict[str, Any], event_data: Dict[str, Any]
    ) -> None:
        """Registra métricas do payload antes de enviar para o modelo."""
        global _PAYLOAD_METRICS_CALLS, _PAYLOAD_METRICS_LAST_TS

        if not isinstance(payload, dict):
            return

        try:
            payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            metrics: Dict[str, Any] = {
                "payload_bytes": payload_bytes,
                "keys_top_level": list(payload.keys()),
                "schema_version": payload.get("_v") or "v1",
                "symbol": (
                    payload.get("symbol")
                    or event_data.get("symbol")
                    or event_data.get("ativo")
                ),
                "epoch_ms": (
                    event_data.get("epoch_ms")
                    or event_data.get("timestamp_ms")
                    or event_data.get("timestamp")
                ),
                "counts": self._extract_list_counts(payload),
            }

            metrics_line = json.dumps(metrics, ensure_ascii=False)
            logging.info(metrics_line)

            try:
                logs_dir = Path("logs")
                logs_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = str(logs_dir / "payload_metrics.jsonl")
                _append_metric_line(metrics, metrics_path)

                _PAYLOAD_METRICS_CALLS += 1
                now = time.time()
                if _PAYLOAD_METRICS_CALLS >= 200 or (now - _PAYLOAD_METRICS_LAST_TS) >= 600:
                    summary = _summarize_metrics(metrics_path, 2000)
                    if summary:
                        logging.info(
                            "PAYLOAD_METRICS_SUMMARY %s",
                            json.dumps(summary, ensure_ascii=False),
                        )
                        _log_payload_tripwires(summary)
                    _PAYLOAD_METRICS_CALLS = 0
                    _PAYLOAD_METRICS_LAST_TS = now
            except Exception as file_err:
                logging.error(f"Erro ao persistir payload metrics: {file_err}", exc_info=True)
        except Exception as e:
            logging.error(f"Erro ao registrar métricas do payload: {e}", exc_info=True)

    # ====================================================================
    # CHAMADAS AO MODELO
    # ====================================================================

    @staticmethod
    def _sanitize_llm_text(text: str) -> str:
        """Remove blocos de raciocínio e lixo comum."""
        if not isinstance(text, str):
            return ""

        s = text.strip()
        if not s:
            return ""

        # Remove <think>...</think>
        try:
            s = re.sub(
                r"<think>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL
            ).strip()
        except Exception:
            pass

        s = s.replace("<think>", "").replace("</think>", "").strip()
        
        # =====================================
        # NOVO: Remove raciocínio em texto livre
        # =====================================
        # Detecta padrões comuns de raciocínio
        reasoning_patterns = [
            r"^(Okay|Ok|Alright|Let me|Let's|First|I need to|I'll|Looking at|Analyzing|Based on).*?\n\n",
            r"^(Hmm|Well|So|Now).*?\n\n",
            r"^(The user wants|The data shows|Given the).*?\n\n",
        ]
        
        for pattern in reasoning_patterns:
            try:
                # Remove apenas se houver JSON depois
                if re.search(r'\{[^}]+\}', s):
                    s = re.sub(pattern, "", s, flags=re.IGNORECASE | re.DOTALL).strip()
            except Exception:
                pass
        
        # Se começa com texto e tem JSON no meio, extrai o JSON
        if not s.startswith("{") and "{" in s:
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', s)
            if json_match:
                potential_json = json_match.group(1)
                try:
                    # Valida se é JSON válido
                    json.loads(potential_json)
                    s = potential_json
                except json.JSONDecodeError:
                    pass
        
        return s

    @staticmethod
    def _try_parse_json_dict(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort: extrai um dict JSON de uma resposta de LLM."""
        if not isinstance(text, str):
            return None
        s = text.strip()
        if not s:
            return None

        if s.startswith("```"):
            parts = s.split("```")
            if len(parts) >= 3:
                s = parts[1].strip()
                if s.lower().startswith("json"):
                    s = s[4:].lstrip()

        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            obj = json.loads(s[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _get_model_params(self) -> Dict[str, Any]:
        """Retorna parâmetros do modelo da configuração."""
        ai_cfg = self.config.get("ai", {})
        if not isinstance(ai_cfg, dict):
            ai_cfg = {}

        try:
            max_tokens = int(ai_cfg.get("max_tokens", 450) or 450)
        except (ValueError, TypeError):
            max_tokens = 450
        max_tokens = max(200, min(max_tokens, 1200))

        try:
            temperature = float(ai_cfg.get("temperature", 0.25) or 0.25)
        except (ValueError, TypeError):
            temperature = 0.25
        temperature = max(0.0, min(temperature, 1.0))

        try:
            timeout = int(ai_cfg.get("timeout", 30) or 30)
        except (ValueError, TypeError):
            timeout = 30

        return {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }

    async def _a_call_openai_text(self, prompt: str) -> str:
        """Versão assíncrona para OpenAI/Groq com fallbacks."""
        if self.client_async is None:
            raise RuntimeError("Cliente assíncrono não inicializado")

        models_to_try = (
            self._groq_model_candidates if self.mode == "groq" else [self.model_name]
        )
        params = self._get_model_params()

        for model in models_to_try:
            try:
                messages: List[ChatMessage] = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ]
                response = await self.client_async.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    timeout=params["timeout"],
                )
                if response.choices and len(response.choices) > 0:
                    content = self._sanitize_llm_text(
                        response.choices[0].message.content or ""
                    )
                    if model != self.model_name:
                        logging.info(
                            f"🔄 Modelo trocado de {self.model_name} para {model}"
                        )
                        self.model_name = model
                    return content
            except Exception as e:
                if _is_model_decommissioned_error(e):
                    logging.warning(f"Modelo {model} decommissioned. Tentando próximo...")
                    continue
                else:
                    logging.error(f"Erro com modelo {model}: {e}. Tentando próximo...")
                    continue

        logging.error("Todos os modelos falharam para texto.")
        return ""

    def _call_openai_compatible(self, prompt: str, max_retries: int = 3) -> str:
        """Chama cliente OpenAI-compatível de forma síncrona."""
        if self.client is None:
            raise RuntimeError("Cliente não inicializado")

        params = self._get_model_params()
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                messages: List[ChatMessage] = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    timeout=params["timeout"],
                )
                if response.choices and len(response.choices) > 0:
                    content = self._sanitize_llm_text(
                        response.choices[0].message.content or ""
                    )
                    if len(content) > 10:
                        if self.mode == "groq":
                            logging.debug(f"✅ Groq respondeu ({len(content)} chars)")
                        return content
                return ""
            except Exception as e:
                logging.error(
                    f"Erro {(self.mode or 'unknown').upper()} "
                    f"(tentativa {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))

        return ""

    def _call_dashscope(self, prompt: str, max_retries: int = 3) -> str:
        """Chama DashScope API com retry."""
        if _Generation is None:
            return ""

        params = self._get_model_params()
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                messages: List[ChatMessage] = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ]
                response = _Generation.call(
                    model=self.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    result_format="message",
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    timeout=params["timeout"],
                )
                content = self._sanitize_llm_text(_extract_dashscope_text(response))
                if len(content) > 10:
                    return content
                return ""
            except Exception as e:
                logging.error(f"Erro DashScope (tentativa {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))

        return ""

    def _call_model(
        self, prompt: str, event_data: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Chama o provedor atual e retorna (raw_response, structured_or_None)."""
        if self.mode in ("openai", "groq") and self.client is not None:
            text = self._sanitize_llm_text(self._call_openai_compatible(prompt))
            parsed = self._try_parse_json_dict(text)
            if parsed is not None:
                try:
                    text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    pass
            return text, parsed

        elif self.mode == "dashscope":
            text = self._sanitize_llm_text(self._call_dashscope(prompt))
            parsed = self._try_parse_json_dict(text)
            if parsed is not None:
                try:
                    text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    pass
            return text, parsed

        else:
            text = self._generate_mock_analysis(event_data)
            return text, None

    def _generate_mock_analysis(self, event_data: Dict[str, Any]) -> str:
        """Gera análise mock quando IA indisponível."""
        timestamp = self.time_manager.now_iso()
        mock_price = format_price(event_data.get("preco_fechamento", 0))
        mock_delta = format_delta(event_data.get("delta", 0))

        return (
            f"**Interpretação (mock):** {event_data.get('tipo_evento')} em "
            f"{event_data.get('ativo')} às {timestamp}.\n"
            f"Preço: ${mock_price} | Delta: {mock_delta}\n"
            f"**Força:** {event_data.get('resultado_da_batalha')}\n"
            f"**Expectativa:** Monitorar reação (dados limitados - modo mock)."
        )

    # ====================================================================
    # NÚCLEO DE ANÁLISE
    # ====================================================================

    def _analyze_internal(
        self, event_data: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Núcleo de análise: constrói prompt, chama modelo e retorna resultado."""
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

        # Verificar _v ANTES do guardrail para preservar a info de payload já comprimido
        _ai_payload_has_v2 = None
        _original_ai_payload = event_data.get("ai_payload")
        if isinstance(_original_ai_payload, dict):
            _ai_payload_has_v2 = _original_ai_payload.get("_v") == 2
        
        event_data_safe = _ensure_safe_llm_payload(event_data)
        if event_data_safe is None:
            logging.error("Análise abortada por leak de payload completo (guardrail).")
            return "analysis skipped (unsafe payload)", None
        event_data = event_data_safe
        
        # Preservar info de v2 que foi removida pelo guardrail
        if _ai_payload_has_v2:
            ai_p = event_data.get("ai_payload")
            if isinstance(ai_p, dict):
                ai_p["_v"] = 2

        try:
            prompt = self._create_prompt(event_data)
        except Exception as e:
            logging.error(f"Erro ao criar prompt: {e}", exc_info=True)
            return self._generate_mock_analysis(event_data), None

        try:
            payload_for_metrics = event_data.get("ai_payload")
            if isinstance(payload_for_metrics, dict):
                self._log_payload_metrics(payload_for_metrics, event_data)
            elif isinstance(event_data, dict):
                self._log_payload_metrics(event_data, event_data)
        except Exception as e:
            logging.error(f"Erro ao instrumentar payload: {e}", exc_info=True)

        self._log_payload_debug(event_data)

        try:
            raw, structured = self._call_model(prompt, event_data)
        except Exception as e:
            logging.error(f"Erro na chamada de IA: {e}", exc_info=True)
            raw, structured = self._generate_mock_analysis(event_data), None

        if not raw:
            raw = self._generate_mock_analysis(event_data)

        try:
            if self.health_monitor is not None:
                self.health_monitor.heartbeat(self.module_name)
        except Exception:
            pass

        return raw, structured

    def _log_payload_debug(self, event_data: Dict[str, Any]) -> None:
        """Log detalhado do payload para debug."""
        try:
            payload_for_llm = event_data.get("ai_payload")
            payload_root_name = "ai_payload"
            if not isinstance(payload_for_llm, dict):
                payload_for_llm = event_data if isinstance(event_data, dict) else {}
                payload_root_name = "event"

            payload_bytes = len(
                json.dumps(payload_for_llm, ensure_ascii=False).encode("utf-8")
            )
            keys_sample = (
                list(payload_for_llm.keys())[:20]
                if isinstance(payload_for_llm, dict)
                else []
            )

            logging.info(
                "LLM_PAYLOAD_INFO root=%s bytes=%s keys_sample=%s",
                payload_root_name,
                payload_bytes,
                keys_sample,
            )

            leak_keys = {
                "raw_event",
                "ANALYSIS_TRIGGER",
                "contextual_snapshot",
                "historical_vp",
                "observability",
            }
            if isinstance(payload_for_llm, dict) and leak_keys.intersection(
                payload_for_llm.keys()
            ):
                logging.warning(
                    "FULL_PAYLOAD_LEAK_RISK root=%s suspicious_keys=%s",
                    payload_root_name,
                    list(leak_keys.intersection(payload_for_llm.keys())),
                )

            if os.getenv("DUMP_LLM_PAYLOAD", "0") == "1":
                self._dump_llm_payload(event_data, payload_bytes)

        except Exception as e:
            logging.error(f"Erro ao logar metadados do payload: {e}", exc_info=True)

    def _dump_llm_payload(
        self, event_data: Dict[str, Any], payload_bytes: int
    ) -> None:
        """Salva payload para debug."""
        try:
            flags: Dict[str, Any] = {}
            try:
                llm_cfg = _get_llm_payload_config()
                flags = {
                    "v2_enabled": bool(llm_cfg.get("v2_enabled", True)),
                    "max_bytes": int(llm_cfg.get("max_bytes", 6144) or 6144),
                    "guardrail_hard_enabled": bool(
                        llm_cfg.get("guardrail_hard_enabled", True)
                    ),
                }
            except Exception:
                pass

            dump_obj: Dict[str, Any] = event_data if isinstance(event_data, dict) else {"event_data": event_data}
            dump_obj = dict(dump_obj)
            dump_meta: Dict[str, Any] = {
                "payload_bytes_final": payload_bytes,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "flags": flags,
            }
            dump_obj["_dump_meta"] = dump_meta

            Path("logs").mkdir(parents=True, exist_ok=True)
            Path("logs/last_llm_payload.json").write_text(
                json.dumps(dump_obj, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            ap = event_data.get("ai_payload") if isinstance(event_data, dict) else None
            mc = (ap or {}).get("macro_context") if isinstance(ap, dict) else None
            ca = (ap or {}).get("cross_asset_context") if isinstance(ap, dict) else None
            logging.info(
                "DUMP_LLM_PAYLOAD wrote=logs/last_llm_payload.json "
                "macro_type=%s cross_type=%s has_section_cache=%s",
                type(mc).__name__,
                type(ca).__name__,
                isinstance((ap or {}).get("_section_cache"), dict),
            )
        except Exception:
            logging.exception("DUMP_LLM_PAYLOAD failed")

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
              "structured": dict | None,
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

            structured_out: Optional[Dict[str, Any]] = None
            if structured is not None:
                if isinstance(structured, dict):
                    structured_out = structured
                elif hasattr(structured, "model_dump"):
                    structured_out = structured.model_dump()  # type: ignore[union-attr]

            tipo_evento = event_data.get("tipo_evento", "N/A")
            ativo = event_data.get("ativo") or event_data.get("symbol") or "N/A"

            logging.info(
                f"✅ IA [{self.mode or 'mock'}] analisou: {tipo_evento} - {ativo}"
            )

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
                "structured": structured_out,
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

    # ====================================================================
    # CLEANUP
    # ====================================================================

    def close(self) -> None:
        """Fecha conexão com IA e encerra heartbeat."""
        try:
            self._hb_stop.set()
            if self._hb_thread is not None and self._hb_thread.is_alive():
                self._hb_thread.join(timeout=5)
        except Exception:
            pass

        if self.mode == "groq":
            logging.info("🔌 Desconectando GroqCloud...")

        try:
            if self.client is not None and hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass

        try:
            if self.client_async is not None and hasattr(self.client_async, "close"):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is not None and not loop.is_closed():
                    try:
                        loop.create_task(self.client_async.close())
                    except Exception:
                        pass
                else:
                    try:
                        asyncio.run(self.client_async.close())
                    except Exception:
                        pass
        except Exception:
            pass

        self.client = None
        self.client_async = None

    async def aclose(self) -> None:
        """Fecha conexões async."""
        try:
            self._hb_stop.set()
            if self._hb_thread is not None and self._hb_thread.is_alive():
                self._hb_thread.join(timeout=5)
        except Exception:
            pass

        if self.mode == "groq":
            logging.info("🔌 Desconectando GroqCloud...")

        try:
            if self.client is not None and hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass

        try:
            if self.client_async is not None and hasattr(self.client_async, "close"):
                await self.client_async.close()
        except Exception:
            pass

        self.client = None
        self.client_async = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ====================================================================
# TESTE DE VALIDAÇÃO
# ====================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 TESTANDO AI_ANALYZER v2.5.1 (GroqCloud - modo texto livre)")
    print("=" * 70)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
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

    print("\n📊 Resultado:")
    print(f"  Success: {result['success']}")
    print(f"  Modo: {result.get('mode', 'N/A')}")
    print(f"  Modelo: {result.get('model', 'N/A')}")
    print(f"  Structured: {result.get('structured')}")
    print(f"  Resposta ({len(result['raw_response'])} chars):")
    print(f"  {result['raw_response'][:300]}...")

    analyzer.close()

    print("\n" + "=" * 70)
    print("✅ TESTE CONCLUÍDO")
    print("=" * 70 + "\n")