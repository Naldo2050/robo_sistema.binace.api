# ai_analyzer_qwen.py
import logging
import os
import random
import time
from typing import Any, Dict

# config é opcional (permite pegar a key via config.DASHSCOPE_API_KEY ou config.AI_KEYS["dashscope"])
try:
    import config as app_config
except Exception:  # pragma: no cover
    app_config = None

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
        # 1) OpenAI (compatível). Usa OPENAI_API_KEY/OPENAI_BASE_URL se existirem.
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI()
                self.mode = "openai"
                self.enabled = True
                logging.info("🔧 OpenAI client configurado (modo compatível)")
                return
            except Exception as e:
                logging.warning(f"OpenAI indisponível: {e}")

        # 2) DashScope (nativo): resolve token por env e/ou config
        token = os.getenv("DASHSCOPE_API_KEY")
        if not token and app_config is not None:
            token = getattr(app_config, "DASHSCOPE_API_KEY", None)
            if token is None:
                ai_keys = getattr(app_config, "AI_KEYS", None)
                if isinstance(ai_keys, dict):
                    token = ai_keys.get("dashscope") or ai_keys.get("DASHSCOPE_API_KEY")

        if DASHSCOPE_AVAILABLE and token:
            try:
                import dashscope  # garantir objeto raiz para setar api_key
                dashscope.api_key = token
                self.mode = "dashscope"
                self.enabled = True
                logging.info("🔧 DashScope configurado (modo nativo)")
                return
            except Exception as e:
                logging.warning(f"DashScope indisponível: {e}")
        elif DASHSCOPE_AVAILABLE and not token:
            logging.warning("DashScope API key não encontrada. Mantendo modo mock.")

        # 3) Mock (sem provedores externos)
        self.mode = None
        self.enabled = True
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

        # Derivativos: pega o bloco do ativo; fallback para BTCUSDT
        deriv_map = event_data.get("derivatives", {}) or {}
        derivativos = deriv_map.get(ativo, {}) or deriv_map.get("BTCUSDT", {})
        if derivativos:
            # Fallback: se OI em USD ausente/zero, mostra OI bruto (contratos)
            try:
                oi_usd_val = float(derivativos.get("open_interest_usd") or 0)
            except Exception:
                oi_usd_val = 0.0
            oi_val = derivativos.get("open_interest")
            if oi_usd_val and oi_usd_val > 0:
                oi_line = f"{oi_usd_val:,.0f} USD"
            else:
                oi_line = f"{oi_val:,.0f} contratos" if oi_val is not None else "Indisponível"

            deriv_str = f"""
🏦 Derivativos ({ativo})
- Funding Rate: {derivativos.get('funding_rate_percent', 0):.4f}%
- OI: {oi_line}
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

        # ------------ Contexto e ambiente de mercado ------------
        # Tenta extrair dados de contexto do evento ou do bloco contextual
        market_ctx = event_data.get("market_context") or event_data.get("contextual", {}).get("market_context", {})  # type: ignore
        market_env = event_data.get("market_environment") or event_data.get("contextual", {}).get("market_environment", {})  # type: ignore

        market_ctx_str = ""
        if isinstance(market_ctx, dict) and market_ctx:
            try:
                sess = market_ctx.get("trading_session", "Indisponível")
                phase = market_ctx.get("session_phase", "Indisponível")
                close_sec = market_ctx.get("time_to_session_close", None)
                close_str = f"{close_sec}seg" if close_sec is not None else "Indisponível"
                dow = market_ctx.get("day_of_week", "Indisponível")
                is_holiday = market_ctx.get("is_holiday", None)
                holiday_str = "Sim" if is_holiday else ("Não" if is_holiday is not None else "Indisponível")
                hours_type = market_ctx.get("market_hours_type", "Indisponível")
                market_ctx_str = f"\n🌍 Contexto de Mercado\n- Sessão: {sess} ({phase}), fecha em {close_str}\n- Dia da semana: {dow} | Feriado: {holiday_str}\n- Horário de mercado: {hours_type}\n"
            except Exception:
                market_ctx_str = ""

        market_env_str = ""
        if isinstance(market_env, dict) and market_env:
            try:
                vol_reg = market_env.get("volatility_regime", "Indisponível")
                trend_dir = market_env.get("trend_direction", "Indisponível")
                mkt_struct = market_env.get("market_structure", "Indisponível")
                liq_env = market_env.get("liquidity_environment", "Indisponível")
                risk_sent = market_env.get("risk_sentiment", "Indisponível")
                corr_spy = market_env.get("correlation_spy", None)
                corr_dxy = market_env.get("correlation_dxy", None)
                corr_gold = market_env.get("correlation_gold", None)
                # Formata correlações apenas se numéricas
                def fmt_corr(v):
                    try:
                        return f"{float(v):+0.2f}"  # ex: +0.45
                    except Exception:
                        return "Indisponível"
                corr_str = f"SP500 {fmt_corr(corr_spy)}, DXY {fmt_corr(corr_dxy)}, GOLD {fmt_corr(corr_gold)}"
                market_env_str = f"\n🌡 Ambiente de Mercado\n- Volatilidade: {vol_reg} | Tendência: {trend_dir} | Estrutura: {mkt_struct}\n- Liquidez: {liq_env} | Sentimento de risco: {risk_sent}\n- Correlações: {corr_str}\n"
            except Exception:
                market_env_str = ""

        # ------------ Ordem de livro: profundidade e spread ------------
        ob_depth = event_data.get("order_book_depth") or event_data.get("contextual", {}).get("order_book_depth", {})  # type: ignore
        depth_str = ""
        if isinstance(ob_depth, dict) and ob_depth:
            try:
                # Usar apenas alguns níveis principais se disponíveis
                lines = []
                for lvl in ("L1", "L5", "L10", "L25"):
                    d = ob_depth.get(lvl)
                    if isinstance(d, dict) and d:
                        bids = d.get("bids")
                        asks = d.get("asks")
                        imb = d.get("imbalance")
                        bid_str = f"{bids:,.0f}" if isinstance(bids, (int, float)) else "Ind"  # Indisponível
                        ask_str = f"{asks:,.0f}" if isinstance(asks, (int, float)) else "Ind"
                        imb_str = f"{imb:+0.2f}" if isinstance(imb, (int, float)) else "Ind"
                        lines.append(f"- {lvl}: Bid {bid_str}, Ask {ask_str}, Imb {imb_str}")
                total_ratio = ob_depth.get("total_depth_ratio")
                ratio_str = f"{total_ratio:+0.3f}" if isinstance(total_ratio, (int, float)) else "Indisponível"
                if lines:
                    depth_str = "\n📑 Profundidade do Livro (USD)\n" + "\n".join(lines) + f"\n- Desvio total: {ratio_str}\n"
            except Exception:
                depth_str = ""

        spread_ana = event_data.get("spread_analysis") or event_data.get("contextual", {}).get("spread_analysis", {})  # type: ignore
        spread_str = ""
        if isinstance(spread_ana, dict) and spread_ana:
            try:
                cs = spread_ana.get("current_spread_bps")
                avg1 = spread_ana.get("avg_spread_1h")
                avg24 = spread_ana.get("avg_spread_24h")
                pct = spread_ana.get("spread_percentile")
                tdur = spread_ana.get("tight_spread_duration_min")
                vol_sp = spread_ana.get("spread_volatility")
                def fmt_num(x, dec=2):
                    try:
                        return f"{float(x):0.{dec}f}"
                    except Exception:
                        return "Ind"
                spread_str = "\n📏 Spread\n" + \
                    f"- Atual: {fmt_num(cs,2)} bps\n" + \
                    f"- Médias: 1h {fmt_num(avg1,2)} bps, 24h {fmt_num(avg24,2)} bps\n" + \
                    f"- Percentil: {fmt_num(pct,1)}% | Tight Dur: {fmt_num(tdur,1)} min | Vol: {fmt_num(vol_sp,3)}\n"
            except Exception:
                spread_str = ""

        # ------------ Fluxo de ordens e participantes ------------
        # Procura métricas de fluxo no evento
        flow = event_data.get("fluxo_continuo") or event_data.get("flow_metrics") or event_data.get("contextual", {}).get("flow_metrics", {})  # type: ignore
        order_flow_str = ""
        participants_str = ""
        if isinstance(flow, dict) and flow:
            # Order flow
            of = flow.get("order_flow", {})
            if isinstance(of, dict) and of:
                try:
                    nf1 = of.get("net_flow_1m")
                    nf5 = of.get("net_flow_5m")
                    nf15 = of.get("net_flow_15m")
                    ab = of.get("aggressive_buy_pct")
                    asell = of.get("aggressive_sell_pct")
                    pb = of.get("passive_buy_pct")
                    ps = of.get("passive_sell_pct")
                    bsr = of.get("buy_sell_ratio")
                    def fmt_nf(x):
                        try:
                            return f"{float(x):+,.0f}"
                        except Exception:
                            return "Ind"
                    def fmt_pct(x):
                        try:
                            return f"{float(x):0.1f}%"
                        except Exception:
                            return "Ind"
                    of_lines = []
                    if any(v is not None for v in (nf1, nf5, nf15)):
                        of_lines.append(f"- Net Flow: 1m {fmt_nf(nf1)}, 5m {fmt_nf(nf5)}, 15m {fmt_nf(nf15)}")
                    if any(v is not None for v in (ab, asell, pb, ps)):
                        of_lines.append(f"- Agressivo: Buy {fmt_pct(ab)} | Sell {fmt_pct(asell)} | Passivo: Buy {fmt_pct(pb)} | Sell {fmt_pct(ps)}")
                    if bsr is not None:
                        try:
                            bsr_str = f"{float(bsr):0.2f}"
                        except Exception:
                            bsr_str = "Ind"
                        of_lines.append(f"- Razão Buy/Sell: {bsr_str}")
                    if of_lines:
                        order_flow_str = "\n🚰 Fluxo de Ordens\n" + "\n".join(of_lines) + "\n"
                except Exception:
                    order_flow_str = ""

            # Participant analysis
            pa = flow.get("participant_analysis", {})
            if isinstance(pa, dict) and pa:
                try:
                    lines = []
                    for role in ("retail_flow", "institutional_flow", "hft_flow"):
                        info = pa.get(role)
                        if not isinstance(info, dict):
                            continue
                        vol_pct = info.get("volume_pct")
                        direction = info.get("direction") or "Ind"
                        avg_sz = info.get("avg_order_size")
                        sentiment = info.get("sentiment") or info.get("activity_level") or "Ind"
                        act_level = info.get("activity_level") or info.get("activity" )
                        # Some mapping to friendly names
                        label_map = {
                            "retail_flow": "Retail",
                            "institutional_flow": "Institucional",
                            "hft_flow": "HFT"
                        }
                        name = label_map.get(role, role)
                        try:
                            vol_pct_str = f"{float(vol_pct):0.1f}%" if vol_pct is not None else "Ind"
                        except Exception:
                            vol_pct_str = "Ind"
                        try:
                            avg_sz_str = f"{float(avg_sz):0.2f}" if avg_sz is not None else "Ind"
                        except Exception:
                            avg_sz_str = "Ind"
                        if act_level and sentiment and act_level != sentiment:
                            sent_str = f"{sentiment} ({act_level})"
                        else:
                            sent_str = sentiment
                        lines.append(f"- {name}: Vol {vol_pct_str}, Dir {direction}, Avg {avg_sz_str}, Sent. {sent_str}")
                    if lines:
                        participants_str = "\n👥 Participantes\n" + "\n".join(lines) + "\n"
                except Exception:
                    participants_str = ""


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
{zone_str}{deriv_str}{vp_str}{market_ctx_str}{market_env_str}{depth_str}{spread_str}{order_flow_str}{participants_str}

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

{zone_str}{deriv_str}{vp_str}{market_ctx_str}{market_env_str}{depth_str}{spread_str}{order_flow_str}{participants_str}

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
