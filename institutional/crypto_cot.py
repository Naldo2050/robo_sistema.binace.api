# institutional/crypto_cot.py
# -*- coding: utf-8 -*-
"""
Crypto COT (Commitment of Traders) & Positioning Analyzer.
Fase P1.1 (Arquitetura Context-Only).

Interpreta a estrutura de posicionamento institucional da Binance Futures:
1. Global Long/Short Account Ratio (Varejo + Mercado Geral)
2. Top Trader Long/Short Account Ratio (Top 20% Contas)
3. Top Trader Long/Short Position Ratio (Top 20% Volume Financeiro)
4. Open Interest e Deltas Temporais (1h / 4h)
5. Funding Rate Canônico (fração decimal)

RESTRIÇÃO DE ARQUITETURA:
Este módulo é ESTRITAMENTE CONTEXT-ONLY.
Não gera ordens de compra/venda diretas, não altera sizing nem limites de risco.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
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

    # Open interest e deltas
    open_interest: Optional[float] = None
    open_interest_usd: Optional[float] = None
    oi_delta_1h: Optional[float] = None
    oi_delta_4h: Optional[float] = None

    # Funding rate canônico
    funding_rate: Optional[float] = None

    # Freshness
    is_stale: bool = False
    is_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Exporta para dicionário serializável em JSON."""
        d = asdict(self)
        d["regime"] = self.regime.value
        return d


class CryptoCOT:
    """
    Motor de análise de posicionamento institucional (Crypto COT).
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

        if positioning_data is None:
            return CryptoCOTAnalysis(
                symbol=symbol,
                observed_at=now,
                regime=PositioningRegime.UNKNOWN,
                reasons=["Dados de posicionamento indisponíveis (None)"],
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
                is_available=False,
                is_stale=is_stale,
            )

        if is_stale:
            return CryptoCOTAnalysis(
                symbol=symbol,
                observed_at=now,
                regime=PositioningRegime.UNKNOWN,
                reasons=[f"Dados obsoletos (age={p_dict.get('age_seconds')}s > limite)"],
                is_available=False,
                is_stale=True,
            )

        g_ratio = p_dict.get("global_account_ratio")
        t_acc_ratio = p_dict.get("top_account_ratio")
        t_pos_ratio = p_dict.get("top_position_ratio")
        g_long_pct = p_dict.get("global_long_account_pct")
        g_short_pct = p_dict.get("global_short_account_pct")
        t_long_acc_pct = p_dict.get("top_long_account_pct")
        t_long_pos_pct = p_dict.get("top_long_position_pct")

        oi = p_dict.get("open_interest")
        oi_usd = p_dict.get("open_interest_usd")
        oi_1h = p_dict.get("oi_delta_1h")
        oi_4h = p_dict.get("oi_delta_4h")

        # Prioriza funding_rate canônico passado ou contido no dict
        fr = funding_rate if funding_rate is not None else p_dict.get("funding_rate")

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
                open_interest=oi,
                open_interest_usd=oi_usd,
                oi_delta_1h=oi_1h,
                oi_delta_4h=oi_4h,
                funding_rate=fr,
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

        # 1. Detecção de Risco de Squeeze (Funding Extremo + Crowding)
        if fr is not None:
            if fr > self.SQUEEZE_FUNDING_THRESHOLD and g_ratio > self.CROWDED_LONG_RATIO:
                regime = PositioningRegime.SQUEEZE_RISK
                reasons.append(
                    f"Risco de Long Squeeze: funding elevado ({fr*100:.3f}%) com varejo comprado (L/S={g_ratio:.2f})"
                )
            elif fr < -self.SQUEEZE_FUNDING_THRESHOLD and g_ratio < self.CROWDED_SHORT_RATIO:
                regime = PositioningRegime.SQUEEZE_RISK
                reasons.append(
                    f"Risco de Short Squeeze: funding negativo ({fr*100:.3f}%) com varejo vendido (L/S={g_ratio:.2f})"
                )

        # 2. Detecção de Divergência Top Traders vs Global
        if regime == PositioningRegime.NEUTRAL:
            if top_pos_vs_global > self.DIVERGENCE_THRESHOLD:
                regime = PositioningRegime.TOP_LONG_DIVERGENCE
                reasons.append(
                    f"Top Traders posicionados comprados vs Varejo (Top Pos={t_pos_ratio:.2f}, Global={g_ratio:.2f}, Diff=+{top_pos_vs_global:.2f})"
                )
            elif top_pos_vs_global < -self.DIVERGENCE_THRESHOLD:
                regime = PositioningRegime.TOP_SHORT_DIVERGENCE
                reasons.append(
                    f"Top Traders posicionados vendidos vs Varejo (Top Pos={t_pos_ratio:.2f}, Global={g_ratio:.2f}, Diff={top_pos_vs_global:.2f})"
                )

        # 3. Detecção de Crowding Unilateral
        if regime == PositioningRegime.NEUTRAL:
            if g_ratio >= self.CROWDED_LONG_RATIO:
                regime = PositioningRegime.CROWDED_LONG
                reasons.append(f"Mercado sobrecarregado na compra (Global L/S={g_ratio:.2f} >= {self.CROWDED_LONG_RATIO})")
            elif g_ratio <= self.CROWDED_SHORT_RATIO:
                regime = PositioningRegime.CROWDED_SHORT
                reasons.append(f"Mercado sobrecarregado na venda (Global L/S={g_ratio:.2f} <= {self.CROWDED_SHORT_RATIO})")

        # 4. Detecção de Expansão de Open Interest
        if regime == PositioningRegime.NEUTRAL:
            if oi_1h is not None and abs(oi_1h) >= self.OI_EXPANSION_1H:
                regime = PositioningRegime.OI_EXPANSION
                direction = "crescimento" if oi_1h > 0 else "queda"
                reasons.append(f"Variação expressiva de OI 1h ({direction} de {oi_1h*100:+.1f}%)")
            elif oi_4h is not None and abs(oi_4h) >= self.OI_EXPANSION_4H:
                regime = PositioningRegime.OI_EXPANSION
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
            open_interest=oi,
            open_interest_usd=oi_usd,
            oi_delta_1h=oi_1h,
            oi_delta_4h=oi_4h,
            funding_rate=fr,
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
