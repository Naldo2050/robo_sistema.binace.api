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
        depth_detail = {}

        if orderbook_data and isinstance(orderbook_data, dict):
            bid_depth = float(orderbook_data.get("bid_depth_usd", 0))
            ask_depth = float(orderbook_data.get("ask_depth_usd", 0))
            ob_imbalance = orderbook_data.get("imbalance", None)

            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                depth_ratio = (bid_depth - ask_depth) / total_depth
                depth_score = depth_ratio * 20  # -20 a +20

                depth_detail["bid_depth"] = round(bid_depth, 2)
                depth_detail["ask_depth"] = round(ask_depth, 2)
                depth_detail["ratio"] = round(depth_ratio, 4)

                # Depth metrics mais detalhados
                depth_metrics = orderbook_data.get("depth_metrics", {})
                if isinstance(depth_metrics, dict):
                    deep_imb = depth_metrics.get("depth_imbalance", 0)
                    if isinstance(deep_imb, (int, float)):
                        # Confirmar com depth mais profundo
                        if (depth_ratio > 0 and deep_imb > 0) or (depth_ratio < 0 and deep_imb < 0):
                            depth_score *= 1.2  # Confirmação = boost
                            depth_detail["deep_confirmation"] = True
                        else:
                            depth_detail["deep_confirmation"] = False

            depth_score = max(-20, min(20, depth_score))

        components["depth"] = {
            "score": round(depth_score, 2),
            "max": 20,
            "detail": depth_detail,
        }
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
                buyer_str = float(abs_inner.get("buyer_strength", 0))
                seller_exh = float(abs_inner.get("seller_exhaustion", 0))
                abs_index = float(abs_inner.get("index", 0))
                classification = str(abs_inner.get("classification", ""))
                label = str(abs_inner.get("label", ""))

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

        if derivatives_data and isinstance(derivatives_data, dict):
            # Buscar dados de BTCUSDT
            btc_deriv = derivatives_data.get("BTCUSDT", derivatives_data)

            if isinstance(btc_deriv, dict):
                lsr = float(btc_deriv.get("long_short_ratio", 1.0))
                oi = float(btc_deriv.get("open_interest", 0))
                oi_usd = float(btc_deriv.get("open_interest_usd", 0))

                # Long/Short Ratio
                # LSR > 2 = muito mais longs = posicionamento bullish
                # LSR < 0.5 = muito mais shorts = posicionamento bearish
                # Mas cuidado: LSR extremo pode indicar crowded trade
                if lsr > 1:
                    # Normalizar: LSR 1→0pts, LSR 2→15pts, LSR 3→20pts
                    lsr_score = min(20, (lsr - 1) * 15)
                else:
                    # LSR 1→0pts, LSR 0.5→-15pts, LSR 0.3→-20pts
                    lsr_score = max(-20, (lsr - 1) * 20)

                deriv_score += lsr_score
                deriv_detail["long_short_ratio"] = lsr
                deriv_detail["lsr_score"] = round(lsr_score, 2)

                # Funding rates (se disponível via onchain)
                if onchain_data and isinstance(onchain_data, dict):
                    funding = onchain_data.get("funding_rates", {})
                    if isinstance(funding, dict) and funding:
                        avg_funding = sum(float(v) for v in funding.values()) / len(funding)
                        # Funding positivo = longs pagam shorts = bullish positioning
                        funding_score = max(-5, min(5, avg_funding * 10000))
                        deriv_score += funding_score
                        deriv_detail["avg_funding"] = round(avg_funding, 6)
                        deriv_detail["funding_score"] = round(funding_score, 2)

                deriv_score = max(-25, min(25, deriv_score))

        # On-chain exchange netflow como bônus
        if onchain_data and isinstance(onchain_data, dict):
            netflow = onchain_data.get("exchange_netflow", 0)
            if isinstance(netflow, (int, float)) and netflow != 0:
                # Netflow negativo = saída de exchanges = acumulação
                # Netflow positivo = entrada em exchanges = distribuição
                netflow_bonus = max(-5, min(5, -netflow * 0.02))
                deriv_score = max(-25, min(25, deriv_score + netflow_bonus))
                deriv_detail["exchange_netflow"] = netflow
                deriv_detail["netflow_signal"] = "accumulation" if netflow < 0 else "distribution"

        components["derivatives"] = {
            "score": round(deriv_score, 2),
            "max": 25,
            "detail": deriv_detail,
        }
        score += deriv_score

        # ═══════════════════════════════════════════
        # SCORE FINAL E CLASSIFICAÇÃO
        # ═══════════════════════════════════════════
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