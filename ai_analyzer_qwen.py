# ai_analyzer_qwen.py - Analisador de IA com formatação padronizada
import logging
import os
import random
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# 🔹 IMPORTA UTILITÁRIOS DE FORMATAÇÃO
from format_utils import (
    format_price,
    format_quantity,
    format_percent,
    format_large_number,
    format_delta,
    format_time_seconds,
    format_scientific
)

# config é opcional (permite pegar tokens e modelo)
try:
    import config as app_config
except Exception:  # pragma: no cover
    app_config = None

# Tentativa de importar OpenAI (modo compatível)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI não instalado. Para usar API real: pip install openai")

# Tentativa de importar DashScope (modo nativo)
try:
    from dashscope import Generation
    import dashscope  # para setar api_key
    DASHSCOPE_AVAILABLE = True
except Exception:
    DASHSCOPE_AVAILABLE = False
    logging.warning("DashScope não instalado. Para usar API real: pip install dashscope")

from time_manager import TimeManager

load_dotenv()  # Carrega variáveis de ambiente do arquivo .env, se existir


def _extract_dashscope_text(resp) -> str:
    """Extrai texto de respostas do DashScope em formatos variados."""
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
        self.client: Optional[OpenAI] = None
        self.enabled = False
        self.mode: Optional[str] = None  # "openai" | "dashscope" | None
        self.time_manager = TimeManager()

        # Configura nome do modelo
        self.model_name = (
            getattr(app_config, "QWEN_MODEL", None)
            or os.getenv("QWEN_MODEL")
            or "qwen-plus"
        )

        self.last_test_time = 0
        self.test_interval_seconds = 120
        self.connection_failed_count = 0
        self.max_failures_before_mock = 3

        logging.info("🧠 IA Analyzer Qwen inicializada - Análise avançada ativada")
        try:
            self._initialize_api()
        except Exception as e:
            logging.warning(f"Falha ao inicializar provedores de IA: {e}. Usando mock.")
            self.mode = None
            self.enabled = True  # mock ligado

    # ---------------------------
    # Inicialização de provedores
    # ---------------------------
    def _initialize_api(self):
        # 1) OpenAI (compatível). Usa env OPENAI_* ou base compatível (ex: DashScope compat-mode)
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()
                self.mode = "openai"
                self.enabled = True
                logging.info("🔧 OpenAI client configurado (modo compatível)")
                return
            except Exception as e:
                logging.warning(f"OpenAI indisponível: {e}")

        # 2) DashScope (nativo)
        # Lê a chave *exclusivamente* da variável de ambiente
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
        elif DASHSCOPE_AVAILABLE and not token:
            logging.warning("DashScope API key não encontrada (variável DASHSCOPE_API_KEY). Mantendo modo mock.")

        # 3) Mock (sem provedores externos)
        self.mode = None
        self.enabled = True
        logging.info("🔧 Modo MOCK ativado (sem provedores externos).")

    # ---------------------------
    # Healthcheck de conexão
    # ---------------------------
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
                return True  # mock
        except Exception as e:
            self.connection_failed_count += 1
            logging.warning(f"Falha no ping da IA ({self.connection_failed_count}): {e}")
            return False

    # ---------------------------
    # Prompt builder com formatação padronizada
    # ---------------------------
    def _create_prompt(self, event_data: Dict[str, Any]) -> str:
        # Campos básicos
        tipo_evento = event_data.get("tipo_evento", "N/A")
        ativo = event_data.get("ativo") or event_data.get("symbol") or "N/A"
        descricao = event_data.get("descricao", "Sem descrição.")
        
        # 🔧 VALIDAÇÃO DE CONSISTÊNCIA DELTA vs VOLUMES
        delta_raw = event_data.get("delta")
        volume_total_raw = event_data.get("volume_total")
        volume_compra_raw = event_data.get("volume_compra")
        volume_venda_raw = event_data.get("volume_venda")
        
        # Converte valores, mantendo None se ausente
        delta = float(delta_raw) if delta_raw is not None else None
        volume_total = float(volume_total_raw) if volume_total_raw is not None else None
        volume_compra = float(volume_compra_raw) if volume_compra_raw is not None else None
        volume_venda = float(volume_venda_raw) if volume_venda_raw is not None else None
        
        # Detecta inconsistência: delta significativo mas volumes zerados
        if delta is not None and abs(delta) > 1.0:
            if (volume_compra == 0 and volume_venda == 0) or volume_total == 0:
                logging.warning(
                    f"⚠️ Inconsistência detectada: delta={delta:.2f} mas volumes zerados. "
                    f"Marcando volumes como indisponíveis."
                )
                volume_compra = None
                volume_venda = None
                volume_total = None
        
        # Se volumes individuais não batem com total, marca como indisponível
        if volume_compra is not None and volume_venda is not None and volume_total is not None:
            calc_total = volume_compra + volume_venda
            if abs(calc_total - volume_total) > 0.01:
                logging.warning(
                    f"⚠️ Volumes inconsistentes: compra({volume_compra}) + venda({volume_venda}) "
                    f"!= total({volume_total}). Marcando como indisponíveis."
                )
                volume_compra = None
                volume_venda = None
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

        # Memória de eventos (se vier anexa)
        memoria = event_data.get("event_history", [])
        memoria_str = ""
        if memoria:
            mem_lines = []
            for e in memoria:
                mem_delta = format_delta(e.get('delta', 0))
                mem_vol = format_large_number(e.get('volume_total', 0))
                mem_lines.append(
                    f"- {e.get('timestamp')} | {e.get('tipo_evento')} | "
                    f"{e.get('resultado_da_batalha')} Δ={mem_delta} Vol={mem_vol}"
                )
            memoria_str = "\n".join(mem_lines)
        else:
            memoria_str = "Nenhum evento recente."

        # Probabilidade histórica
        conf = event_data.get("historical_confidence", {})
        prob_long = conf.get("long_prob", "Indisponível")
        prob_short = conf.get("short_prob", "Indisponível")
        prob_neutral = conf.get("neutral_prob", "Indisponível")

        # Zona institucional
        z = event_data.get("zone_context") or {}
        zone_str = ""
        if z:
            conflu = z.get("confluence") or []
            if not isinstance(conflu, list):
                conflu = [str(conflu)]
            
            # 🔹 FORMATAÇÃO CORRIGIDA
            zone_low = format_price(z.get('low', 0))
            zone_high = format_price(z.get('high', 0))
            zone_anchor = format_price(z.get('anchor_price', 0))
            
            zone_str = f"""
🟦 Zona Institucional
- Tipo: {z.get('kind')} | TF: {z.get('timeframe')} | Score: {z.get('score')}
- Faixa: {zone_low} ~ {zone_high} (centro: {zone_anchor})
- Confluências: {", ".join(conflu) if conflu else "Nenhuma"}
- Toques: {z.get('touch_count')} | Último toque: {z.get('last_touched')}
"""

        # Derivativos
        deriv_map = (
            event_data.get("derivatives")
            or event_data.get("contextual_snapshot", {}).get("derivatives")
            or event_data.get("contextual", {}).get("derivatives")
            or {}
        )
        derivativos = deriv_map.get(ativo, {}) or deriv_map.get("BTCUSDT", {})
        if derivativos:
            try:
                oi_usd_val = float(derivativos.get("open_interest_usd") or 0)
            except Exception:
                oi_usd_val = 0.0
            
            # 🔹 FORMATAÇÃO CORRIGIDA
            oi_line = format_large_number(oi_usd_val) + " USD" if oi_usd_val > 0 else (
                format_quantity(derivativos.get('open_interest', 0)) + " contratos"
                if derivativos.get('open_interest') is not None
                else "Indisponível"
            )
            
            funding_rate = format_percent(derivativos.get('funding_rate_percent', 0))
            ls_ratio = format_scientific(derivativos.get('long_short_ratio', 0), decimals=2)
            longs_usd = format_large_number(derivativos.get('longs_usd', 0))
            shorts_usd = format_large_number(derivativos.get('shorts_usd', 0))
            
            deriv_str = f"""
🏦 Derivativos ({ativo})
- Funding Rate: {funding_rate}
- OI: {oi_line}
- Long/Short Ratio: {ls_ratio}
- Liquidações (5min): Longs=${longs_usd} | Shorts=${shorts_usd}
"""
        else:
            deriv_str = "\n🏦 Derivativos: Indisponível."

        # Volume Profile (Diário) - COM FORMATAÇÃO CORRIGIDA
        vp = (
            event_data.get("historical_vp", {}).get("daily", {})
            or event_data.get("contextual_snapshot", {}).get("historical_vp", {}).get("daily", {})
            or {}
        )
        if vp:
            try:
                # 🔹 FORMATAÇÃO CORRIGIDA
                current_price_for_vp = float(preco)
                hvns_list = sorted(vp.get("hvns") or [], key=lambda x: abs(x - current_price_for_vp))
                lvns_list = sorted(vp.get("lvns") or [], key=lambda x: abs(x - current_price_for_vp))

                # Formata as listas com format_price
                hvn_str = ", ".join([f"${format_price(x)}" for x in hvns_list[:12]]) or "Nenhum"
                lvn_str = ", ".join([f"${format_price(x)}" for x in lvns_list[:12]]) or "Nenhum"
                
                poc_fmt = format_price(vp.get('poc', 0))
                val_fmt = format_price(vp.get('val', 0))
                vah_fmt = format_price(vp.get('vah', 0))
            except Exception:
                hvn_str = lvn_str = "Indisponível"
                poc_fmt = val_fmt = vah_fmt = "Indisponível"
            
            vp_str = f"""
📊 Volume Profile (Diário)
- POC: ${poc_fmt} | Value Area: ${val_fmt} a ${vah_fmt}
- HVNs (próximos): {hvn_str}
- LVNs (próximos): {lvn_str}
"""
        else:
            vp_str = "\n📊 Volume Profile Histórico: Indisponível."

        # Contexto de mercado
        ctx_snap = event_data.get("contextual_snapshot", {}) or {}
        market_ctx = (
            event_data.get("market_context")
            or ctx_snap.get("market_context")
            or event_data.get("contextual", {}).get("market_context", {})
            or {}
        )
        market_env = (
            event_data.get("market_environment")
            or ctx_snap.get("market_environment")
            or event_data.get("contextual", {}).get("market_environment", {})
            or {}
        )

        market_ctx_str = ""
        if market_ctx:
            try:
                sess = market_ctx.get("trading_session", "Indisponível")
                phase = market_ctx.get("session_phase", "Indisponível")
                close_sec = market_ctx.get("time_to_session_close", None)
                close_str = format_time_seconds(close_sec * 1000) if isinstance(close_sec, (int, float)) else "Indisponível"
                dow = market_ctx.get("day_of_week", "Indisponível")
                is_holiday = market_ctx.get("is_holiday", None)
                holiday_str = "Sim" if is_holiday else ("Não" if is_holiday is not None else "Indisponível")
                hours_type = market_ctx.get("market_hours_type", "Indisponível")
                market_ctx_str = (
                    f"\n🌍 Contexto de Mercado\n- Sessão: {sess} ({phase}), fecha em {close_str}\n"
                    f"- Dia da semana: {dow} | Feriado: {holiday_str}\n- Horário de mercado: {hours_type}\n"
                )
            except Exception:
                market_ctx_str = ""

        market_env_str = ""
        if market_env:
            try:
                vol_reg = market_env.get("volatility_regime", "Indisponível")
                trend_dir = market_env.get("trend_direction", "Indisponível")
                mkt_struct = market_env.get("market_structure", "Indisponível")
                liq_env = market_env.get("liquidity_environment", "Indisponível")
                risk_sent = market_env.get("risk_sentiment", "Indisponível")

                # 🔹 FORMATAÇÃO CORRIGIDA para correlações
                def fmt_corr(v):
                    try:
                        return format_delta(float(v))
                    except Exception:
                        return "Ind"

                corr_str = (
                    f"SP500 {fmt_corr(market_env.get('correlation_spy'))}, "
                    f"DXY {fmt_corr(market_env.get('correlation_dxy'))}, "
                    f"GOLD {fmt_corr(market_env.get('correlation_gold'))}"
                )
                market_env_str = (
                    f"\n🌡 Ambiente de Mercado\n- Volatilidade: {vol_reg} | Tendência: {trend_dir} | Estrutura: {mkt_struct}\n"
                    f"- Liquidez: {liq_env} | Sentimento de risco: {risk_sent}\n- Correlações: {corr_str}\n"
                )
            except Exception:
                market_env_str = ""

        # Order book depth / spread
        ob_depth = (
            event_data.get("order_book_depth")
            or ctx_snap.get("orderbook_data", {}).get("order_book_depth")
            or event_data.get("contextual", {}).get("orderbook_data", {}).get("order_book_depth")
            or {}
        )
        depth_str = ""
        if isinstance(ob_depth, dict) and ob_depth:
            try:
                lines = []
                for lvl in ("L1", "L5", "L10", "L25"):
                    d = ob_depth.get(lvl)
                    if isinstance(d, dict) and d:
                        # 🔹 FORMATAÇÃO CORRIGIDA
                        bids = format_large_number(d.get("bids", 0))
                        asks = format_large_number(d.get("asks", 0))
                        imb = format_scientific(d.get("imbalance", 0), decimals=2)
                        lines.append(f"- {lvl}: Bid {bids}, Ask {asks}, Imb {imb}")
                
                total_ratio = ob_depth.get("total_depth_ratio")
                ratio_str = format_scientific(total_ratio, decimals=3) if total_ratio else "Indisponível"
                if lines:
                    depth_str = "\n📑 Profundidade do Livro (USD)\n" + "\n".join(lines) + f"\n- Desvio total: {ratio_str}\n"
            except Exception:
                depth_str = ""

        spread_ana = (
            event_data.get("spread_analysis")
            or ctx_snap.get("orderbook_data", {}).get("spread_metrics")
            or event_data.get("contextual", {}).get("spread_analysis")
            or {}
        )
        spread_str = ""
        if isinstance(spread_ana, dict) and spread_ana:
            try:
                # 🔹 FORMATAÇÃO CORRIGIDA
                cs = spread_ana.get("current_spread_bps") or spread_ana.get("spread_bps")
                avg1 = spread_ana.get("avg_spread_1h")
                avg24 = spread_ana.get("avg_spread_24h")
                pct = spread_ana.get("spread_percentile")
                tdur = spread_ana.get("tight_spread_duration_min")
                vol_sp = spread_ana.get("spread_volatility")
                
                spread_str = (
                    "\n📏 Spread\n"
                    f"- Atual: {format_scientific(cs, decimals=2) if cs else 'Ind'} bps\n"
                    f"- Médias: 1h {format_scientific(avg1, decimals=2) if avg1 else 'Ind'} bps, "
                    f"24h {format_scientific(avg24, decimals=2) if avg24 else 'Ind'} bps\n"
                    f"- Percentil: {format_percent(pct) if pct else 'Ind'} | "
                    f"Tight Dur: {format_quantity(tdur) if tdur else 'Ind'} min | "
                    f"Vol: {format_scientific(vol_sp, decimals=3) if vol_sp else 'Ind'}\n"
                )
            except Exception:
                spread_str = ""

        # Fluxo contínuo (order flow + participantes)
        flow = (
            event_data.get("fluxo_continuo")
            or event_data.get("flow_metrics")
            or ctx_snap.get("flow_metrics")
            or event_data.get("contextual", {}).get("flow_metrics")
            or {}
        )
        order_flow_str = ""
        participants_str = ""
        if isinstance(flow, dict) and flow:
            of = flow.get("order_flow", {})
            if isinstance(of, dict) and of:
                try:
                    nf1, nf5, nf15 = of.get("net_flow_1m"), of.get("net_flow_5m"), of.get("net_flow_15m")
                    ab, asell, pb, ps = of.get("aggressive_buy_pct"), of.get("aggressive_sell_pct"), of.get("passive_buy_pct"), of.get("passive_sell_pct")
                    bsr = of.get("buy_sell_ratio")
                    of_lines = []
                    
                    # 🔹 FORMATAÇÃO CORRIGIDA
                    if any(v is not None for v in (nf1, nf5, nf15)):
                        nf1_fmt = format_delta(nf1) if nf1 else "Ind"
                        nf5_fmt = format_delta(nf5) if nf5 else "Ind"
                        nf15_fmt = format_delta(nf15) if nf15 else "Ind"
                        of_lines.append(f"- Net Flow: 1m {nf1_fmt}, 5m {nf5_fmt}, 15m {nf15_fmt}")
                    
                    if any(v is not None for v in (ab, asell, pb, ps)):
                        ab_fmt = format_percent(ab) if ab else "Ind"
                        as_fmt = format_percent(asell) if asell else "Ind"
                        pb_fmt = format_percent(pb) if pb else "Ind"
                        ps_fmt = format_percent(ps) if ps else "Ind"
                        of_lines.append(f"- Agressivo: Buy {ab_fmt} | Sell {as_fmt} | Passivo: Buy {pb_fmt} | Sell {ps_fmt}")
                    
                    if bsr is not None:
                        bsr_str = format_scientific(float(bsr), decimals=2) if bsr else "Ind"
                        of_lines.append(f"- Razão Buy/Sell: {bsr_str}")
                    
                    if of_lines:
                        order_flow_str = "\n🚰 Fluxo de Ordens\n" + "\n".join(of_lines) + "\n"
                except Exception:
                    order_flow_str = ""

            pa = flow.get("participant_analysis", {})
            if isinstance(pa, dict) and pa:
                try:
                    lines = []
                    label_map = {"retail_flow": "Retail", "institutional_flow": "Institucional", "hft_flow": "HFT"}
                    for role in ("retail_flow", "institutional_flow", "hft_flow"):
                        info = pa.get(role)
                        if not isinstance(info, dict):
                            continue
                        
                        # 🔹 FORMATAÇÃO CORRIGIDA
                        vol_pct = format_percent(info.get("volume_pct", 0))
                        direction = info.get("direction") or "Ind"
                        avg_sz = format_quantity(info.get("avg_order_size", 0))
                        sentiment = info.get("sentiment") or info.get("activity_level") or "Ind"
                        act_level = info.get("activity_level")
                        sent_str = f"{sentiment} ({act_level})" if act_level and sentiment and act_level != sentiment else sentiment
                        
                        lines.append(
                            f"- {label_map.get(role, role)}: Vol {vol_pct}, "
                            f"Dir {direction}, Avg {avg_sz}, Sent. {sent_str}"
                        )
                    if lines:
                        participants_str = "\n👥 Participantes\n" + "\n".join(lines) + "\n"
                except Exception:
                    participants_str = ""

        # ML FEATURES (price/volume/microstructure)
        ml = (event_data.get("ml_features") or event_data.get("ml") or {})
        ml_str = ""
        if isinstance(ml, dict) and ml:
            try:
                pf = ml.get("price_features", {}) or {}
                vf = ml.get("volume_features", {}) or {}
                mf = ml.get("microstructure", {}) or {}

                p_lines = []
                for k in ("returns_1", "returns_5", "returns_15", "volatility_1", "volatility_5", "volatility_15", "momentum_score"):
                    if k in pf:
                        val = pf[k]
                        # 🔹 FORMATAÇÃO CORRIGIDA
                        if k.startswith("returns_"):
                            p_lines.append(f"{k}={format_percent(val * 100)}")
                        elif k.startswith("volatility_"):
                            p_lines.append(f"{k}={format_scientific(val, decimals=5)}")
                        else:
                            p_lines.append(f"{k}={format_scientific(val, decimals=5)}")

                v_lines = []
                for k in ("volume_sma_ratio", "volume_momentum", "buy_sell_pressure", "liquidity_gradient"):
                    if k in vf:
                        val = vf[k]
                        # 🔹 FORMATAÇÃO CORRIGIDA
                        if "pressure" in k:
                            v_lines.append(f"{k}={format_delta(val)}")
                        elif "ratio" in k:
                            v_lines.append(f"{k}={format_percent(val * 100) if val < 10 else format_percent(val)}")
                        else:
                            v_lines.append(f"{k}={format_scientific(val, decimals=2)}")

                m_lines = []
                for k in ("order_book_slope", "flow_imbalance", "tick_rule_sum", "trade_intensity"):
                    if k in mf:
                        val = mf[k]
                        # 🔹 FORMATAÇÃO CORRIGIDA
                        m_lines.append(f"{k}={format_scientific(val, decimals=3)}")

                blocks = []
                if p_lines:
                    blocks.append("• Price: " + ", ".join(p_lines))
                if v_lines:
                    blocks.append("• Volume: " + ", ".join(v_lines))
                if m_lines:
                    blocks.append("• Microstructure: " + ", ".join(m_lines))

                if blocks:
                    ml_str = "\n📐 ML Features\n" + "\n".join(blocks) + "\n"
            except Exception:
                ml_str = ""

        # Caso específico: evento de OrderBook
        if (event_data.get("tipo_evento") == "OrderBook") or ("imbalance" in event_data):
            sm = (
                event_data.get("spread_metrics")
                or ctx_snap.get("orderbook_data", {}).get("spread_metrics")
                or {}
            )
            # 🔹 FORMATAÇÃO CORRIGIDA
            imbalance = format_scientific(event_data.get("imbalance", 0), decimals=3)
            ratio = format_scientific(event_data.get("volume_ratio", 0), decimals=2)
            pressure = format_delta(event_data.get("pressure", 0))
            spread = format_price(sm.get("spread", 0))
            spread_pct = format_percent(sm.get("spread_percent", 0))
            bid_usd = format_large_number(sm.get("bid_depth_usd", 0))
            ask_usd = format_large_number(sm.get("ask_depth_usd", 0))
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
- Preço: {format_price(preco)}
- Imbalance: {imbalance} | Ratio: {ratio} | Pressure: {pressure}
- Spread: {spread} ({spread_pct})
- Profundidade (USD): Bid={bid_usd} | Ask={ask_usd}{mi_lines}
- Alertas: {", ".join(alertas) if alertas else "Nenhum"}
"""

            return f"""
🧠 **Análise Institucional – {ativo} | {tipo_evento}**

📝 Descrição: {descricao}
{ob_str}{ml_str}{zone_str}{deriv_str}{vp_str}{market_ctx_str}{market_env_str}{depth_str}{spread_str}{order_flow_str}{participants_str}

📈 Multi-Timeframes
{multi_tf_str}

⏳ Memória de eventos
{memoria_str}

📉 Probabilidade Histórica
Long={prob_long} | Short={prob_short} | Neutro={prob_neutral}

🎯 Tarefa
NÃO INVENTE números. Se um campo acima estiver 'Indisponível' ou ausente, responda explicitamente 'Indisponível' e não estime.
Forneça parecer institucional e um PLANO ancorado na zona (se houver), cobrindo:
1) Interpretação (order flow, liquidez, zona, microestrutura/ML).
2) Força dominante.
3) Expectativa (curto/médio prazo).
4) Probabilidade mais provável (considere os valores acima).
5) Plano de trade: direção, condição de entrada (gatilho), invalidação (fora da zona), alvos 1/2 (próximas zonas), riscos.
6) Gestão de posição: sizing dinâmico com base em ATR, volume de parede e volatilidade do cluster.
"""

        # Prompt padrão
# 🔹 FORMATAÇÃO CORRIGIDA
        vol_line = (
            "- Vol: Indisponível" if volume_total is None else f"- Vol: {format_large_number(volume_total)}"
        )
        if ((volume_compra or 0) > 0) or ((volume_venda or 0) > 0):
            vol_line += f" (Buy={format_large_number(volume_compra or 0)} | Sell={format_large_number(volume_venda or 0)})"

        # Linha de Delta com fallback
        delta_line = f"- Delta: {format_delta(delta)}" if delta is not None else "- Delta: Indisponível"

        return f"""
🧠 **Análise Institucional – {ativo} | {tipo_evento}**

📝 Descrição: {descricao}

- Preço: {format_price(preco) if preco else "Indisponível"}
{delta_line}
{vol_line}
{ml_str}{zone_str}{deriv_str}{vp_str}{market_ctx_str}{market_env_str}{depth_str}{spread_str}{order_flow_str}{participants_str}

📈 Multi-Timeframes
{multi_tf_str}

⏳ Memória de eventos
{memoria_str}

📉 Probabilidade Histórica
Long={prob_long} | Short={prob_short} | Neutro={prob_neutral}

🎯 Tarefa
NÃO INVENTE números. Se um campo acima estiver 'Indisponível' ou ausente, responda explicitamente 'Indisponível' e não estime.
Forneça parecer institucional e um PLANO ancorado na zona (se houver), cobrindo:
1) Interpretação (order flow, liquidez, zona, microestrutura/ML).
2) Força dominante.
3) Expectativa (curto/médio prazo).
4) Probabilidade mais provável (considere os valores acima).
5) Plano de trade: direção, condição de entrada (gatilho), invalidação (fora da zona), alvos 1/2 (próximas zonas), riscos.
6) Gestão de posição: sizing dinâmico com base em ATR, volume de parede e volatilidade do cluster.
"""

    # ---------------------------
    # Callers de provedores
    # ---------------------------
    def _call_openai_compatible(self, prompt: str, max_retries: int = 3) -> str:
        base_delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é um analista institucional de trading e order flow. "
                                "REGRAS: 1) Use SOMENTE números e métricas explicitamente fornecidos; "
                                "2) Se um dado não for fornecido, escreva 'Indisponível' e NÃO estime; "
                                "3) Não invente bps, market impact, spread ou volumes; "
                                "4) Se livro e fita (delta) divergirem, explique; "
                                "5) Seja sucinto e objetivo; 6) Não é conselho financeiro."
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
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Você é um analista institucional de trading e order flow. "
                                "REGRAS: 1) Use SOMENTE números e métricas explicitamente fornecidos; "
                                "2) Se um dado não for fornecido, escreva 'Indisponível' e NÃO estime; "
                                "3) Não invente bps, market impact, spread ou volumes; "
                                "4) Se livro e fita (delta) divergirem, explique; "
                                "5) Seja sucinto e objetivo; 6) Não é conselho financeiro."
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
                logging.warning("Resposta DashScope curta/indisponível.")
                return ""
            except Exception as e:
                logging.error(f"Erro API DashScope (tentativa {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Aguardando {delay:.1f}s antes de retry...")
                    time.sleep(delay)
        return ""

    # ---------------------------
    # Fallback mock
    # ---------------------------
    def _generate_mock_analysis(self, event_data: Dict[str, Any]) -> str:
        timestamp = self.time_manager.now_iso()
        # 🔹 FORMATAÇÃO CORRIGIDA NO MOCK
        mock_price = format_price(event_data.get('preco_fechamento', 0))
        mock_delta = format_delta(event_data.get('delta', 0))
        
        return (
            f"**Interpretação (mock):** Detecção de {event_data.get('tipo_evento')} em "
            f"{event_data.get('ativo') or event_data.get('symbol')} às {timestamp}.\n"
            f"Preço: ${mock_price} | Delta: {mock_delta}\n"
            f"**Força Dominante:** {event_data.get('resultado_da_batalha')}\n"
            f"**Expectativa:** Lateralização com viés conforme delta/fluxo recente (mock).\n"
            f"**Plano:** Operar reação na zona; stop além da borda; alvo no próximo HVN/LVN."
        )

    # ---------------------------
    # Interface pública
    # ---------------------------
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
        try:
            self.close()
        except Exception:
            pass