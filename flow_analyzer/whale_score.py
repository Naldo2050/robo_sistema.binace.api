"""
Whale Accumulation Score — Detector de acumulação/distribuição institucional.

Score composto de -100 (distribuição forte) a +100 (acumulação forte)
baseado em múltiplas fontes:

  1. Whale/Mid flow direction               → -30 a +30 pontos
  2. Order book depth asymmetry              → -20 a +20 pontos
  3. Absorption pattern bias                 → -25 a +25 pontos
  4. Derivatives context (OI + LSR)          → -25 a +25 pontos

Classificações:
  +50 a +100 → STRONG_ACCUMULATION
  +20 a +49  → MILD_ACCUMULATION
  -19 a +19  → NEUTRAL
  -49 a -20  → MILD_DISTRIBUTION
  -100 a -50 → STRONG_DISTRIBUTION

Uso:
    calculator = WhaleAccumulationCalculator()
    result = calculator.calculate(
        sector_flow={"mid": {"delta": -1.66}, "retail": {"delta": 2.27}},
        orderbook_data={"bid_depth_usd": 552161, "ask_depth_usd": 477276},
        absorption_data={"buyer_strength": 4.5, "seller_exhaustion": 1.0},
        derivatives_data={"BTCUSDT": {"long_short_ratio": 2.42, "open_interest": 79425}},
    )
    print(result["score"])            # ex: 28
    print(result["classification"])   # ex: "MILD_ACCUMULATION"
"""

import logging
import math
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


def _finite_or_none(value) -> Optional[float]:
    """float finito ou None. P0-FINAL-CLOSE: NaN/±Inf/str inválida nunca votam
    e nunca serializam (mesma semântica de flow_analyzer.metrics._finite_volume,
    sem importar para manter este módulo folha)."""
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _canonical_absorption_direction(label: object) -> str:
    """Direção canônica da absorção (somente evidência explicativa, P0-A2).

    Convenção canônica (absorption.py:10-16, data_handler.py:1160-1169):
    - Absorção de Compra / COMPRA / BUY => BEARISH (BUY agressivo absorvido por SELL)
    - Absorção de Venda / VENDA / SELL => BULLISH (SELL agressivo absorvido por BUY)
    - Neutra, ausente ou desconhecida => NEUTRAL

    NOTA DE NÃO-REUSO: common/signal_direction.infer_signal_side NÃO é reutilizado
    aqui de propósito. Ele mapeia "COMPRA" genérico => LONG/BULLISH e retorna
    LONG/SHORT/NEUTRAL/UNKNOWN com precedência event_type/explicit_side, enquanto
    neste contexto "COMPRA"/"BUY" significa absorção de compra => BEARISH.
    Reutilizá-lo inverteria o contrato e acoplaria flow_analyzer a common.
    Função privada local, sem threshold/peso, sem conversão em número.
    """
    text = str(label or "").upper()
    if "COMPRA" in text or "BUY" in text:
        return "BEARISH"
    if "VENDA" in text or "SELL" in text:
        return "BULLISH"
    return "NEUTRAL"


class WhaleAccumulationCalculator:
    """
    Calcula score de acumulação/distribuição de whales.
    
    Combina sinais de múltiplas fontes em um score único.
    Mantém histórico para detectar tendências de acumulação ao longo do tempo.
    """

    def __init__(self, history_window: int = 30):
        """
        Args:
            history_window: Quantos scores anteriores manter para média móvel.
        """
        self._history: deque = deque(maxlen=history_window)
        self._last_score = 0
        self._last_calc_ms = 0

    def calculate(
        self,
        sector_flow: Optional[dict] = None,
        orderbook_data: Optional[dict] = None,
        absorption_data: Optional[dict] = None,
        derivatives_data: Optional[dict] = None,
        onchain_data: Optional[dict] = None,
        cvd: Optional[float] = None,
    ) -> dict:
        """
        Calcula o Whale Accumulation Score.
        
        Args:
            sector_flow: Fluxo por setor.
                Espera: {
                    "whale": {"buy": x, "sell": x, "delta": x},  # se disponível
                    "mid": {"buy": x, "sell": x, "delta": x},
                    "retail": {"buy": x, "sell": x, "delta": x},
                }
            orderbook_data: Dados do order book.
                Espera: {"bid_depth_usd": x, "ask_depth_usd": x, "imbalance": x}
            absorption_data: Dados de absorção atual.
                Espera: {
                    "index": x, "classification": "...",
                    "buyer_strength": x, "seller_exhaustion": x,
                    "continuation_probability": x,
                }
                OU: {"current_absorption": {...}} (nested)
            derivatives_data: Dados de derivativos.
                Espera: {"BTCUSDT": {"long_short_ratio": x, "open_interest": x, "open_interest_usd": x}}
            onchain_data: Dados on-chain (se disponível).
                Espera: {"exchange_netflow": x, "whale_transactions": x, "funding_rates": {...}}
            cvd: Cumulative Volume Delta acumulado.
            
        Returns:
            Dict com score (-100 a +100), classificação e componentes.
        """
        components = {}
        score: float = 0.0

        # ═══════════════════════════════════════════
        # 1. WHALE / MID FLOW DIRECTION (-30 a +30)
        # ═══════════════════════════════════════════
        flow_score: float = 0.0
        flow_detail: dict = {}
        _sector_nonvoting = False

        if sector_flow and isinstance(sector_flow, dict):
            # Priorizar whale, fallback para mid
            whale_data = sector_flow.get("whale", {})
            mid_data = sector_flow.get("mid", {})
            retail_data = sector_flow.get("retail", {})

            # Delta do whale/mid (quem move o mercado). P0-FINAL-CLOSE:
            # NaN/±Inf/inválido não vota: vira 0.0 (mesmo que campo ausente)
            # com rastro em detail; se NENHUM delta for utilizável mas havia
            # sujeira non-finite, o componente é NON_VOTING (nunca NaN/Inf).
            _bad_fields = []

            def _sector_delta(data, name):
                if not isinstance(data, dict):
                    return 0.0
                if "delta" not in data:
                    return 0.0
                v = _finite_or_none(data.get("delta"))
                if v is None:
                    _bad_fields.append(name)
                    return 0.0
                return v

            def _delta_usable(data, name):
                return (isinstance(data, dict)
                        and data.get("delta") is not None
                        and name not in _bad_fields)

            whale_delta = _sector_delta(whale_data, "whale")
            mid_delta = _sector_delta(mid_data, "mid")
            retail_delta = _sector_delta(retail_data, "retail")
            _sector_nonvoting = bool(_bad_fields) and not (
                _delta_usable(whale_data, "whale")
                or _delta_usable(mid_data, "mid")
                or _delta_usable(retail_data, "retail")
            )
            if _bad_fields:
                flow_detail["nonfinite_ignored"] = sorted(_bad_fields)

            # Usar whale se disponível, senão mid
            primary_delta = whale_delta if whale_delta != 0 else mid_delta
            
            # Normalizar: clamp entre -30 e +30
            # Delta é em BTC, escalar por fator
            flow_score = max(-30, min(30, primary_delta * 10))

            # Divergência smart money vs retail (sinal forte)
            # Se whales compram e retail vende = acumulação silenciosa
            smart_delta = whale_delta + mid_delta
            if smart_delta > 0 and retail_delta < 0:
                flow_score = min(30, flow_score + 10)  # Bonus: smart money buying while retail sells
                flow_detail["divergence"] = "smart_accumulation"
            elif smart_delta < 0 and retail_delta > 0:
                flow_score = max(-30, flow_score - 10)  # Smart money distributing
                flow_detail["divergence"] = "smart_distribution"
            else:
                flow_detail["divergence"] = "aligned"

            flow_detail["whale_delta"] = round(whale_delta, 4)
            flow_detail["mid_delta"] = round(mid_delta, 4)
            flow_detail["retail_delta"] = round(retail_delta, 4)
            flow_detail["primary_delta"] = round(primary_delta, 4)

        # CVD como fallback/complemento. P0-FINAL-CLOSE: NaN/±Inf nunca votam
        # (antes: max/min propagavam NaN; Inf virava score extremo). Fórmula
        # para CVD finito bit-equivalente.
        if cvd is not None and flow_score == 0:
            _cvd_v = _finite_or_none(cvd)
            if _cvd_v is None:
                flow_detail["cvd_status"] = "NON_VOTING_NONFINITE"
            else:
                flow_score = max(-15, min(15, _cvd_v * 5))
                flow_detail["cvd_used"] = True

        components["flow"] = {
            "score": round(flow_score, 2),
            "max": 30,
            "detail": flow_detail,
        }
        if _sector_nonvoting:
            components["flow"]["status"] = "NON_VOTING_NONFINITE"
            components["flow"]["reason"] = "SECTOR_DELTA_NONFINITE"
        score += flow_score

        # ═══════════════════════════════════════════
        # 2. ORDER BOOK DEPTH ASYMMETRY (-20 a +20)
        # ═══════════════════════════════════════════
        depth_score: float = 0.0
        depth_detail: dict = {}
        depth_status: Optional[str] = None
        depth_reason: Optional[str] = None

        if not orderbook_data or not isinstance(orderbook_data, dict):
            depth_status = "NON_VOTING_MISSING"
            depth_reason = "ORDERBOOK_DATA_MISSING"
        else:
            raw_bid = orderbook_data.get("bid_depth_usd")
            raw_ask = orderbook_data.get("ask_depth_usd")

            if raw_bid is None or raw_ask is None:
                depth_status = "NON_VOTING_MISSING"
                depth_reason = "BID_OR_ASK_DEPTH_MISSING"
            else:
                bid_depth = _finite_or_none(raw_bid)
                ask_depth = _finite_or_none(raw_ask)

                if bid_depth is None or ask_depth is None:
                    depth_status = "NON_VOTING_INVALID_INPUT"
                    depth_reason = "BID_OR_ASK_DEPTH_NONFINITE_OR_MALFORMED"
                    depth_detail["invalid_fields"] = [
                        f for f, v in [("bid_depth_usd", bid_depth), ("ask_depth_usd", ask_depth)] if v is None
                    ]
                elif bid_depth < 0 or ask_depth < 0:
                    depth_status = "NON_VOTING_INVALID_INPUT"
                    depth_reason = "NEGATIVE_DEPTH"
                elif bid_depth == 0.0 and ask_depth == 0.0:
                    # Contrato comprovado: orderbook_core/event_factory.py emite 0/0
                    # exclusivamente em erro/indisponibilidade (fail-closed)
                    depth_status = "NON_VOTING_ZERO_DEPTH"
                    depth_reason = "ZERO_TOTAL_DEPTH"
                    depth_detail["bid_depth"] = 0.0
                    depth_detail["ask_depth"] = 0.0
                else:
                    total_depth = bid_depth + ask_depth
                    # total_depth > 0 garantido
                    depth_ratio = (bid_depth - ask_depth) / total_depth
                    depth_score = depth_ratio * 20.0  # -20 a +20

                    depth_detail["bid_depth"] = round(bid_depth, 2)
                    depth_detail["ask_depth"] = round(ask_depth, 2)
                    depth_detail["ratio"] = round(depth_ratio, 4)

                    # Depth metrics mais detalhados
                    depth_metrics = orderbook_data.get("depth_metrics", {})
                    if isinstance(depth_metrics, dict):
                        deep_imb = depth_metrics.get("depth_imbalance", 0)
                        deep_imb_v = _finite_or_none(deep_imb)
                        if deep_imb_v is not None:
                            # Confirmar com depth mais profundo
                            if (depth_ratio > 0 and deep_imb_v > 0) or (depth_ratio < 0 and deep_imb_v < 0):
                                depth_score *= 1.2  # Confirmação = boost
                                depth_detail["deep_confirmation"] = True
                            else:
                                depth_detail["deep_confirmation"] = False

                    depth_score = max(-20.0, min(20.0, depth_score))

        components["depth"] = {
            "score": round(depth_score, 2),
            "max": 20,
            "detail": depth_detail,
        }
        if depth_status:
            components["depth"]["status"] = depth_status
            components["depth"]["reason"] = depth_reason
        score += depth_score

        # ═══════════════════════════════════════════
        # 3. ABSORPTION PATTERN BIAS — P0-A2 FAIL-CLOSED NON-VOTING
        # ═══════════════════════════════════════════════════════════
        # Contrato P0-A2: contribuição numérica neutralizada (score sempre 0.0)
        # porque magnitude net*3 provou-se semanticamente assimétrica e deriva
        # da mesma fita de trades do componente flow (double counting), sem
        # magnitude alternativa canônica validada. Direção conhecida vira apenas
        # evidência explicativa (canonical_direction), NUNCA número no total.
        # Preservados para explicabilidade/compat: max=25, label, index,
        # classification, buyer/seller, net legado só p/ diagnóstico.
        abs_score: float = 0.0
        abs_detail: dict = {}

        if absorption_data and isinstance(absorption_data, dict):
            # Suportar formato nested ou flat
            abs_inner = absorption_data.get("current_absorption", absorption_data)

            if isinstance(abs_inner, dict):
                buyer_str = _finite_or_none(abs_inner.get("buyer_strength")) or 0.0
                seller_exh = _finite_or_none(abs_inner.get("seller_exhaustion")) or 0.0
                abs_index = _finite_or_none(abs_inner.get("index")) or 0.0
                classification = str(abs_inner.get("classification") or "")
                label = str(abs_inner.get("label") or "")

                # Métrica legada SOMENTE para diagnóstico/compatibilidade.
                # NÃO entra no score (fail-closed). Assimetria documentada em
                # tests/unit/test_whale_absorption_boost_p0a.py (P0-A -> P0-A2).
                net_absorption = buyer_str - seller_exh

                canonical_direction = _canonical_absorption_direction(label)

                abs_detail["buyer_strength"] = buyer_str
                abs_detail["seller_exhaustion"] = seller_exh
                abs_detail["net_absorption"] = round(net_absorption, 2)
                abs_detail["legacy_unvalidated_metric"] = True
                abs_detail["index"] = abs_index
                abs_detail["label"] = label
                abs_detail["classification"] = classification
                abs_detail["canonical_direction"] = canonical_direction

        components["absorption"] = {
            "score": round(abs_score, 2),
            "max": 25,
            "status": "NON_VOTING_UNVALIDATED_MAGNITUDE",
            "canonical_direction": _canonical_absorption_direction(
                abs_detail.get("label", "") if abs_detail else ""
            )
            if abs_detail
            else "NEUTRAL",
            "detail": abs_detail,
        }
        score += abs_score

        # ═══════════════════════════════════════════
        # 4. DERIVATIVES CONTEXT (-25 a +25)
        # ═══════════════════════════════════════════
        deriv_score: float = 0.0
        deriv_detail: dict = {}

        # 4.1 LSR (Long/Short Ratio)
        btc_deriv: dict = {}
        if derivatives_data and isinstance(derivatives_data, dict):
            raw_btc = derivatives_data.get("BTCUSDT", derivatives_data)
            if isinstance(raw_btc, dict):
                btc_deriv = raw_btc

        # Open interest informativo (sem quebrar calculate em inputs não finitos/inválidos)
        raw_oi = btc_deriv.get("open_interest")
        oi_v = _finite_or_none(raw_oi)
        raw_oi_usd = btc_deriv.get("open_interest_usd")
        oi_usd_v = _finite_or_none(raw_oi_usd)
        if oi_v is not None:
            deriv_detail["open_interest"] = oi_v
        if oi_usd_v is not None:
            deriv_detail["open_interest_usd"] = oi_usd_v

        raw_lsr = btc_deriv.get("long_short_ratio")
        if raw_lsr is None:
            lsr_valid = False
            lsr_status = "NON_VOTING_MISSING"
            lsr_reason = "LSR_MISSING"
            lsr_observed = None
            lsr_contrib = 0.0
        else:
            v_lsr = _finite_or_none(raw_lsr)
            if v_lsr is None:
                lsr_valid = False
                lsr_status = "NON_VOTING_INVALID_INPUT"
                lsr_reason = "LSR_NONFINITE_OR_MALFORMED"
                lsr_observed = None
                lsr_contrib = 0.0
            elif v_lsr <= 0:
                lsr_valid = False
                lsr_status = "NON_VOTING_INVALID_INPUT"
                lsr_reason = "LSR_NON_POSITIVE"
                lsr_observed = v_lsr
                lsr_contrib = 0.0
            else:
                lsr_valid = True
                lsr_status = "VALID"
                lsr_reason = None
                lsr_observed = round(v_lsr, 4)
                if v_lsr > 1.0:
                    lsr_score = min(20.0, (v_lsr - 1.0) * 15.0)
                else:
                    lsr_score = max(-20.0, (v_lsr - 1.0) * 20.0)
                lsr_contrib = round(lsr_score, 2)
                deriv_detail["long_short_ratio"] = lsr_observed
                deriv_detail["lsr_score"] = lsr_contrib

        # 4.2 Funding Rates
        funding = None
        if onchain_data and isinstance(onchain_data, dict):
            funding = onchain_data.get("funding_rates")
            if funding is None:
                funding = onchain_data.get("funding_rate")
        if funding is None and btc_deriv:
            funding = btc_deriv.get("funding_rates")
            if funding is None:
                funding = btc_deriv.get("funding_rate")

        if funding is None:
            funding_valid = False
            funding_status = "NON_VOTING_MISSING"
            funding_reason = "FUNDING_MISSING"
            funding_observed = None
            funding_contrib = 0.0
        elif isinstance(funding, dict):
            if not funding:
                funding_valid = False
                funding_status = "NON_VOTING_MISSING"
                funding_reason = "FUNDING_RATES_EMPTY"
                funding_observed = None
                funding_contrib = 0.0
            else:
                valid_rates = []
                for rate_val in funding.values():
                    fv = _finite_or_none(rate_val)
                    if fv is not None:
                        valid_rates.append(fv)
                if valid_rates:
                    avg_funding = sum(valid_rates) / len(valid_rates)
                    funding_score = max(-5.0, min(5.0, avg_funding * 10000.0))
                    funding_valid = True
                    funding_status = "VALID"
                    funding_reason = None
                    funding_observed = round(avg_funding, 6)
                    funding_contrib = round(funding_score, 2)
                    deriv_detail["avg_funding"] = funding_observed
                    deriv_detail["funding_score"] = funding_contrib
                else:
                    funding_valid = False
                    funding_status = "NON_VOTING_INVALID_INPUT"
                    funding_reason = "FUNDING_RATES_ALL_NONFINITE"
                    funding_observed = None
                    funding_contrib = 0.0
        else:
            fv = _finite_or_none(funding)
            if fv is not None:
                avg_funding = fv
                funding_score = max(-5.0, min(5.0, avg_funding * 10000.0))
                funding_valid = True
                funding_status = "VALID"
                funding_reason = None
                funding_observed = round(avg_funding, 6)
                funding_contrib = round(funding_score, 2)
                deriv_detail["avg_funding"] = funding_observed
                deriv_detail["funding_score"] = funding_contrib
            else:
                funding_valid = False
                funding_status = "NON_VOTING_INVALID_INPUT"
                funding_reason = "FUNDING_RATE_NONFINITE_OR_MALFORMED"
                funding_observed = None
                funding_contrib = 0.0

        # 4.3 On-chain Exchange Netflow
        if not onchain_data or not isinstance(onchain_data, dict):
            netflow_valid = False
            netflow_status = "NON_VOTING_MISSING"
            netflow_reason = "ONCHAIN_DATA_MISSING"
            netflow_observed = None
            netflow_contrib = 0.0
        else:
            req_paid = onchain_data.get("requires_paid_api") or []
            is_paid = (
                (isinstance(req_paid, (list, tuple, set)) and "exchange_netflow" in req_paid)
                or onchain_data.get("status") in ("requires_paid_api", "unavailable")
            )
            if is_paid:
                netflow_valid = False
                netflow_status = "NON_VOTING_MISSING"
                netflow_reason = "REQUIRES_PAID_API"
                netflow_observed = None
                netflow_contrib = 0.0
            else:
                raw_nf = onchain_data.get("exchange_netflow")
                if raw_nf is None or "exchange_netflow" not in onchain_data:
                    netflow_valid = False
                    netflow_status = "NON_VOTING_MISSING"
                    netflow_reason = "EXCHANGE_NETFLOW_MISSING"
                    netflow_observed = None
                    netflow_contrib = 0.0
                else:
                    nf_v = _finite_or_none(raw_nf)
                    if nf_v is None:
                        netflow_valid = False
                        netflow_status = "NON_VOTING_INVALID_INPUT"
                        netflow_reason = "NETFLOW_NONFINITE_OR_MALFORMED"
                        netflow_observed = None
                        netflow_contrib = 0.0
                    elif nf_v == 0.0:
                        # Observação zero real (fluxo líquido nulo)
                        netflow_valid = True
                        netflow_status = "VALID_ZERO_OBSERVED"
                        netflow_reason = None
                        netflow_observed = 0.0
                        netflow_contrib = 0.0
                    else:
                        nf_bonus = max(-5.0, min(5.0, -nf_v * 0.02))
                        netflow_valid = True
                        netflow_status = "VALID"
                        netflow_reason = None
                        netflow_observed = round(nf_v, 4)
                        netflow_contrib = round(nf_bonus, 2)
                        deriv_detail["exchange_netflow"] = netflow_observed
                        deriv_detail["netflow_signal"] = "accumulation" if netflow_observed < 0 else "distribution"

        # Detalhe per-input
        deriv_detail["inputs"] = {
            "lsr": {
                "observed": lsr_observed,
                "validity": lsr_status,
                "contribution": lsr_contrib,
            },
            "funding": {
                "observed": funding_observed,
                "validity": funding_status,
                "contribution": funding_contrib,
            },
            "netflow": {
                "observed": netflow_observed,
                "validity": netflow_status,
                "contribution": netflow_contrib,
            },
        }
        if lsr_reason:
            deriv_detail["inputs"]["lsr"]["reason"] = lsr_reason
        if funding_reason:
            deriv_detail["inputs"]["funding"]["reason"] = funding_reason
        if netflow_reason:
            deriv_detail["inputs"]["netflow"]["reason"] = netflow_reason

        # Total do componente derivatives
        deriv_score = max(-25.0, min(25.0, lsr_contrib + funding_contrib + netflow_contrib))
        any_deriv_valid = lsr_valid or funding_valid or netflow_valid

        components["derivatives"] = {
            "score": round(deriv_score, 2),
            "max": 25,
            "detail": deriv_detail,
        }
        if not any_deriv_valid:
            has_invalid = any(s == "NON_VOTING_INVALID_INPUT" for s in (lsr_status, funding_status, netflow_status))
            components["derivatives"]["status"] = "NON_VOTING_INVALID_INPUT" if has_invalid else "NON_VOTING_MISSING"
            components["derivatives"]["reason"] = "ALL_INPUTS_INVALID" if has_invalid else "ALL_INPUTS_MISSING"

        score += deriv_score

        # ═══════════════════════════════════════════
        # SCORE FINAL E CLASSIFICAÇÃO
        # ═══════════════════════════════════════════
        if not math.isfinite(score):
            score = 0.0
        score = max(-100, min(100, round(score)))

        if score >= 50:
            classification = "STRONG_ACCUMULATION"
        elif score >= 20:
            classification = "MILD_ACCUMULATION"
        elif score >= -19:
            classification = "NEUTRAL"
        elif score >= -49:
            classification = "MILD_DISTRIBUTION"
        else:
            classification = "STRONG_DISTRIBUTION"

        # Bias simplificado
        if score > 10:
            bias = "ACCUMULATING"
        elif score < -10:
            bias = "DISTRIBUTING"
        else:
            bias = "NEUTRAL"

        # Registrar no histórico
        now_ms = int(time.time() * 1000)
        self._history.append({"score": score, "ts": now_ms})
        self._last_score = score
        self._last_calc_ms = now_ms

        # Tendência (comparar com histórico)
        trend = self._calculate_trend()

        return {
            "score": score,
            "classification": classification,
            "bias": bias,
            "components": components,
            "trend": trend,
            "status": "success",
        }

    def _calculate_trend(self) -> dict:
        """Calcula tendência do score ao longo do tempo."""
        if len(self._history) < 3:
            return {
                "direction": "insufficient_data",
                "avg_score": self._last_score,
                "samples": len(self._history),
            }

        scores = [h["score"] for h in self._history]
        avg_score = sum(scores) / len(scores)
        recent_avg = sum(scores[-5:]) / min(5, len(scores))

        # Tendência
        if recent_avg > avg_score + 5:
            direction = "increasing_accumulation"
        elif recent_avg < avg_score - 5:
            direction = "increasing_distribution"
        else:
            direction = "stable"

        # Momentum: diferença entre último e média
        momentum = self._last_score - avg_score

        return {
            "direction": direction,
            "avg_score": round(avg_score, 1),
            "recent_avg": round(recent_avg, 1),
            "momentum": round(momentum, 1),
            "samples": len(self._history),
            "score_range": {"min": min(scores), "max": max(scores)},
        }

    def get_last_score(self) -> int:
        """Retorna último score calculado."""
        return self._last_score

    def get_history_summary(self) -> dict:
        """Retorna resumo do histórico de scores."""
        if not self._history:
            return {"status": "empty", "samples": 0}

        scores = [h["score"] for h in self._history]
        return {
            "status": "ok",
            "samples": len(scores),
            "current": scores[-1],
            "avg": round(sum(scores) / len(scores), 1),
            "min": min(scores),
            "max": max(scores),
            "std": round(
                (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5, 1
            ) if len(scores) > 1 else 0,
            "trend": self._calculate_trend()["direction"],
        }

    def reset(self) -> None:
        """Limpa histórico."""
        self._history.clear()
        self._last_score = 0