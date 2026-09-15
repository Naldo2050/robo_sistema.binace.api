# institutional/crypto_cot.py
# -*- coding: utf-8 -*-
"""
Crypto COT (Commitment of Traders) & Positioning Analyzer.
Fase P1.1 (Arquitetura Context-Only).

Interpreta a estrutura de posicionamento da Binance Futures (fonte intraday,
independente do COT oficial semanal da CFTC/CME — ver institutional/cftc_cot.py):
1. Global Long/Short Account Ratio (mercado geral Binance, todas as contas)
2. Top Trader Long/Short Account Ratio (Top 20% contas por margem, Binance)
3. Top Trader Long/Short Position Ratio (Top 20% por volume nocional, Binance)
4. Open Interest e Deltas Temporais (1h / 4h)
5. Funding Rate Canônico (fração decimal)

NOMENCLATURA: "Global" = universo Binance (não equivale a "varejo" como fato);
"Top Traders" = coorte Binance por margem/volume (não equivale a
"institucional"/"smart money" como fato). Este módulo NÃO é o COT oficial.

RESTRIÇÃO DE ARQUITETURA:
Este módulo é ESTRITAMENTE CONTEXT-ONLY.
Não gera ordens de compra/venda diretas, não altera sizing nem limites de risco.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from institutional.base import AnalysisResult, Side, Signal, SignalStrength

logger = logging.getLogger("CryptoCOT")


class PositioningRegime(str, Enum):
    """Regimes determinísticos de posicionamento de mercado."""
    NEUTRAL = "NEUTRAL"
    CROWDED_LONG = "CROWDED_LONG"
    CROWDED_SHORT = "CROWDED_SHORT"
    TOP_LONG_DIVERGENCE = "TOP_LONG_DIVERGENCE"
    TOP_SHORT_DIVERGENCE = "TOP_SHORT_DIVERGENCE"
    OI_EXPANSION = "OI_EXPANSION"
    SQUEEZE_RISK = "SQUEEZE_RISK"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class CryptoCOTAnalysis:
    """Resultado estruturado da interpretação de posicionamento."""
    symbol: str
    observed_at: float
    regime: PositioningRegime
    reasons: List[str] = field(default_factory=list)

    # Ratios brutos e normalizados
    global_account_ratio: Optional[float] = None
    top_account_ratio: Optional[float] = None
    top_position_ratio: Optional[float] = None

    # Percentuais
    global_long_pct: Optional[float] = None
    global_short_pct: Optional[float] = None
    top_long_account_pct: Optional[float] = None
    top_long_position_pct: Optional[float] = None

    # Divergências
    top_account_vs_global: Optional[float] = None
    top_position_vs_global: Optional[float] = None

    # Divergências em pontos percentuais de long-share (métrica canônica;
    # os campos *_vs_global acima são legado em diferença bruta de ratio).
    # long_share = ratio / (1 + ratio); divergence_pp = (top - global) * 100.
    top_account_vs_global_pp: Optional[float] = None
    top_position_vs_global_pp: Optional[float] = None

    # Lados/direções estruturados (evitam parse de texto livre em reasons).
    squeeze_side: Optional[str] = None  # "LONG" | "SHORT" | None
    oi_direction: Optional[str] = None  # "EXPANSION" | "CONTRACTION" | None

    # Qualidade: {"missing_fields": [...], "warnings": [...]}.
    quality: Dict[str, Any] = field(default_factory=dict)

    # Open interest e deltas
    open_interest: Optional[float] = None
    open_interest_usd: Optional[float] = None
    oi_delta_1h: Optional[float] = None
    oi_delta_4h: Optional[float] = None

    # Funding rate canônico
    funding_rate: Optional[float] = None

    # Proveniência temporal (MEDIUM-2): três instantes separados.
    # source_as_of = instante da fonte (quando disponível).
    # retrieved_at = quando a resposta foi recebida (quando disponível;
    #   NUNCA preenchido com analyzed_at).
    # observed_at  = instante da análise (analyzed_at, epoch float legado).
    source_as_of: Optional[str] = None
    retrieved_at: Optional[str] = None
    analyzed_at: Optional[str] = None

    # Freshness
    is_stale: bool = False
    is_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Exporta para dicionário serializável em JSON."""
        d = asdict(self)
        d["regime"] = self.regime.value
        return d


def _pp(top: Optional[float], glob: Optional[float]) -> Optional[float]:
    """Divergência em pontos percentuais de long-share (canônica)."""
    if top is None or glob is None:
        return None
    return round((top / (1.0 + top) - glob / (1.0 + glob)) * 100.0, 4)


class CryptoCOT:
    """
    Motor de análise de posicionamento Binance (intraday, context-only).
    Avalia divergências institucionais, crowding e risco de liquidação em cascata.
    """

    # Thresholds determinísticos documentados
    CROWDED_LONG_RATIO = 2.0     # > 66.7% contas compradas
    CROWDED_SHORT_RATIO = 0.5    # > 66.7% contas vendidas
    DIVERGENCE_THRESHOLD = 0.40  # Diferença expressiva entre Top Traders e Varejo
    OI_EXPANSION_1H = 0.05       # +5% crescimento de OI em 1h
    OI_EXPANSION_4H = 0.10       # +10% crescimento de OI em 4h
    SQUEEZE_FUNDING_THRESHOLD = 0.0003  # 0.03% (3 bps por 8h)

    def __init__(self):
        pass

    def analyze(
        self,
        positioning_data: Optional[Any],
        funding_rate: Optional[float] = None,
        symbol: str = "BTCUSDT",
    ) -> CryptoCOTAnalysis:
        """
        Interpreta dados de posicionamento e retorna regime determinístico com evidências.
        """
        now = time.time()
        analyzed_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

        if positioning_data is None:
            return CryptoCOTAnalysis(
                symbol=symbol,
                observed_at=now,
                regime=PositioningRegime.UNKNOWN,
                reasons=["Dados de posicionamento indisponíveis (None)"],
                analyzed_at=analyzed_iso,
                is_available=False,
            )

        # Suporta tanto BinancePositioningSnapshot quanto dict
        if hasattr(positioning_data, "to_dict"):
            p_dict = positioning_data.to_dict()
        elif isinstance(positioning_data, dict):
            p_dict = positioning_data
        else:
            return CryptoCOTAnalysis(
                symbol=symbol,
                observed_at=now,
                regime=PositioningRegime.UNKNOWN,
                reasons=["Tipo de dado de posicionamento inválido"],
                analyzed_at=analyzed_iso,
                is_available=False,
            )

        is_stale = bool(p_dict.get("is_stale", False))
        is_available = bool(p_dict.get("is_available", True))

        if not is_available:
            return CryptoCOTAnalysis(
                symbol=symbol,
                observed_at=now,
                regime=PositioningRegime.UNKNOWN,
                reasons=["Posicionamento marcado como indisponível pela fonte"],
                analyzed_at=analyzed_iso,
                is_available=False,
                is_stale=is_stale,
            )

        if is_stale:
            return CryptoCOTAnalysis(
                symbol=symbol,
                observed_at=now,
                regime=PositioningRegime.UNKNOWN,
                reasons=[f"Dados obsoletos (age={p_dict.get('age_seconds')}s > limite)"],
                analyzed_at=analyzed_iso,
                is_available=False,
                is_stale=True,
            )

        # Prioriza funding_rate canônico passado ou contido no dict
        fr = funding_rate if funding_rate is not None else p_dict.get("funding_rate")

        warnings: List[str] = []
        missing: List[str] = []

        def _as_ratio(v: Any, name: str) -> Optional[float]:
            if v is None or isinstance(v, bool):
                if v is None:
                    missing.append(name)
                else:
                    warnings.append(f"{name} bool rejeitado")
                    missing.append(name)
                return None
            try:
                f = float(v)
            except (ValueError, TypeError):
                warnings.append(f"{name} não-numérico rejeitado")
                missing.append(name)
                return None
            if not math.isfinite(f):
                warnings.append(f"{name} non-finite rejeitado")
                missing.append(name)
                return None
            if f < 0:
                warnings.append(f"{name} negativo rejeitado (ratio >= 0)")
                missing.append(name)
                return None
            return f

        g_ratio = _as_ratio(p_dict.get("global_account_ratio"), "global_account_ratio")
        t_acc_ratio = _as_ratio(p_dict.get("top_account_ratio"), "top_account_ratio")
        t_pos_ratio = _as_ratio(p_dict.get("top_position_ratio"), "top_position_ratio")

        # Funding: rejeita bool, non-finite e valores fora de limites plausíveis.
        if isinstance(fr, bool):
            warnings.append("funding_rate bool rejeitado")
            fr = None
        elif fr is not None:
            try:
                fr_f = float(fr)
                if not math.isfinite(fr_f):
                    warnings.append("funding_rate non-finite rejeitado")
                    fr = None
                elif abs(fr_f) > 0.05:
                    warnings.append(f"funding_rate {fr_f} fora de faixa plausível (|fr|<=0.05)")
                    fr = None
                else:
                    fr = fr_f
            except (ValueError, TypeError):
                warnings.append("funding_rate não-numérico rejeitado")
                fr = None

        # Open interest negativo é inválido (nível, não delta).
        oi = p_dict.get("open_interest")
        if oi is not None and not isinstance(oi, bool):
            try:
                oi_f = float(oi)
                if not math.isfinite(oi_f) or oi_f < 0:
                    warnings.append("open_interest inválido rejeitado")
                    oi = None
            except (ValueError, TypeError):
                warnings.append("open_interest não-numérico rejeitado")
                oi = None
        elif isinstance(oi, bool):
            warnings.append("open_interest bool rejeitado")
            oi = None

        g_long_pct = p_dict.get("global_long_account_pct")
        g_short_pct = p_dict.get("global_short_account_pct")
        t_long_acc_pct = p_dict.get("top_long_account_pct")
        t_long_pos_pct = p_dict.get("top_long_position_pct")

        oi_usd = p_dict.get("open_interest_usd")
        oi_1h = p_dict.get("oi_delta_1h")
        oi_4h = p_dict.get("oi_delta_4h")

        # Consistência ratio x percentual (alerta em quality, sem mudar regime).
        def _long_share(r: Optional[float]) -> Optional[float]:
            return (r / (1.0 + r)) if r is not None else None

        _g_share = _long_share(g_ratio)
        if _g_share is not None and g_long_pct is not None \
                and not isinstance(g_long_pct, bool):
            try:
                if abs(float(g_long_pct) - _g_share * 100.0) > 2.0:
                    warnings.append("global_long_account_pct inconsistente com global_account_ratio")
            except (ValueError, TypeError):
                pass

        # Verifica completude essencial
        if g_ratio is None or t_pos_ratio is None:
            reasons = ["Dados parciais: ausência de global_account_ratio ou top_position_ratio"]
            return CryptoCOTAnalysis(
                symbol=symbol,
                observed_at=now,
                regime=PositioningRegime.PARTIAL,
                reasons=reasons,
                global_account_ratio=g_ratio,
                top_account_ratio=t_acc_ratio,
                top_position_ratio=t_pos_ratio,
                top_account_vs_global_pp=_pp(t_acc_ratio, g_ratio),
                top_position_vs_global_pp=_pp(t_pos_ratio, g_ratio),
                open_interest=oi,
                open_interest_usd=oi_usd,
                oi_delta_1h=oi_1h,
                oi_delta_4h=oi_4h,
                funding_rate=fr,
                source_as_of=p_dict.get("source_as_of"),
                retrieved_at=p_dict.get("retrieved_at"),
                analyzed_at=analyzed_iso,
                quality={"missing_fields": sorted(set(missing)),
                         "warnings": warnings},
                is_available=True,
                is_stale=False,
            )

        # Cálculos de divergência
        top_acc_vs_global = (
            round(t_acc_ratio - g_ratio, 4) if t_acc_ratio is not None else None
        )
        top_pos_vs_global = round(t_pos_ratio - g_ratio, 4)

        reasons: List[str] = []
        regime = PositioningRegime.NEUTRAL
        squeeze_side: Optional[str] = None
        oi_direction: Optional[str] = None

        # 1. Detecção de Risco de Squeeze (Funding Extremo + Crowding)
        if fr is not None:
            if fr > self.SQUEEZE_FUNDING_THRESHOLD and g_ratio > self.CROWDED_LONG_RATIO:
                regime = PositioningRegime.SQUEEZE_RISK
                squeeze_side = "LONG"
                reasons.append(
                    f"Risco de Long Squeeze: funding elevado ({fr*100:.3f}%) com mercado geral comprado (L/S={g_ratio:.2f})"
                )
            elif fr < -self.SQUEEZE_FUNDING_THRESHOLD and g_ratio < self.CROWDED_SHORT_RATIO:
                regime = PositioningRegime.SQUEEZE_RISK
                squeeze_side = "SHORT"
                reasons.append(
                    f"Risco de Short Squeeze: funding negativo ({fr*100:.3f}%) com mercado geral vendido (L/S={g_ratio:.2f})"
                )

        # 2. Detecção de Divergência Top Traders vs mercado geral
        # (legado: diferença bruta de ratios; canônico: divergence_pp).
        if regime == PositioningRegime.NEUTRAL:
            if top_pos_vs_global > self.DIVERGENCE_THRESHOLD:
                regime = PositioningRegime.TOP_LONG_DIVERGENCE
                reasons.append(
                    f"Top Traders posicionados comprados vs mercado geral (Top Pos={t_pos_ratio:.2f}, Global={g_ratio:.2f}, Diff=+{top_pos_vs_global:.2f})"
                )
            elif top_pos_vs_global < -self.DIVERGENCE_THRESHOLD:
                regime = PositioningRegime.TOP_SHORT_DIVERGENCE
                reasons.append(
                    f"Top Traders posicionados vendidos vs mercado geral (Top Pos={t_pos_ratio:.2f}, Global={g_ratio:.2f}, Diff={top_pos_vs_global:.2f})"
                )

        # 3. Detecção de Crowding Unilateral
        if regime == PositioningRegime.NEUTRAL:
            if g_ratio >= self.CROWDED_LONG_RATIO:
                regime = PositioningRegime.CROWDED_LONG
                reasons.append(f"Mercado sobrecarregado na compra (Global L/S={g_ratio:.2f} >= {self.CROWDED_LONG_RATIO})")
            elif g_ratio <= self.CROWDED_SHORT_RATIO:
                regime = PositioningRegime.CROWDED_SHORT
                reasons.append(f"Mercado sobrecarregado na venda (Global L/S={g_ratio:.2f} <= {self.CROWDED_SHORT_RATIO})")

        # 4. Detecção de variação expressiva de Open Interest
        # (regime legado OI_EXPANSION mantido; direção em oi_direction).
        if regime == PositioningRegime.NEUTRAL:
            if oi_1h is not None and abs(oi_1h) >= self.OI_EXPANSION_1H:
                regime = PositioningRegime.OI_EXPANSION
                oi_direction = "EXPANSION" if oi_1h > 0 else "CONTRACTION"
                direction = "crescimento" if oi_1h > 0 else "queda"
                reasons.append(f"Variação expressiva de OI 1h ({direction} de {oi_1h*100:+.1f}%)")
            elif oi_4h is not None and abs(oi_4h) >= self.OI_EXPANSION_4H:
                regime = PositioningRegime.OI_EXPANSION
                oi_direction = "EXPANSION" if oi_4h > 0 else "CONTRACTION"
                direction = "crescimento" if oi_4h > 0 else "queda"
                reasons.append(f"Variação expressiva de OI 4h ({direction} de {oi_4h*100:+.1f}%)")

        if not reasons:
            reasons.append(f"Posicionamento equilibrado (Global L/S={g_ratio:.2f}, Top Pos={t_pos_ratio:.2f})")

        return CryptoCOTAnalysis(
            symbol=symbol,
            observed_at=now,
            regime=regime,
            reasons=reasons,
            global_account_ratio=g_ratio,
            top_account_ratio=t_acc_ratio,
            top_position_ratio=t_pos_ratio,
            global_long_pct=g_long_pct,
            global_short_pct=g_short_pct,
            top_long_account_pct=t_long_acc_pct,
            top_long_position_pct=t_long_pos_pct,
            top_account_vs_global=top_acc_vs_global,
            top_position_vs_global=top_pos_vs_global,
            top_account_vs_global_pp=_pp(t_acc_ratio, g_ratio),
            top_position_vs_global_pp=_pp(t_pos_ratio, g_ratio),
            squeeze_side=squeeze_side,
            oi_direction=oi_direction,
            quality={"missing_fields": sorted(set(missing)),
                     "warnings": warnings},
            open_interest=oi,
            open_interest_usd=oi_usd,
            oi_delta_1h=oi_1h,
            oi_delta_4h=oi_4h,
            funding_rate=fr,
            source_as_of=p_dict.get("source_as_of"),
            retrieved_at=p_dict.get("retrieved_at"),
            analyzed_at=analyzed_iso,
            is_stale=is_stale,
            is_available=is_available,
        )

    def to_legacy_analysis_result(self, analysis: CryptoCOTAnalysis) -> AnalysisResult:
        """
        Converte para formato AnalysisResult da camada institucional mantendo Side.UNKNOWN
        para garantir que seja estritamente CONTEXT-ONLY (sem interferência em trade).
        """
        res = AnalysisResult(source="crypto_cot", timestamp=analysis.observed_at)
        res.metrics = analysis.to_dict()
        res.confidence = 0.0  # Context-only, zero influência direcional
        return res
