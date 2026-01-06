# flow/trade_flow_analyzer.py
# -*- coding: utf-8 -*-

"""
TradeFlowAnalyzer extraído do market_orchestrator.py.
Lógica idêntica à original, apenas movida para este módulo.
"""

from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from data_handler import (
    create_absorption_event,
    create_exhaustion_event,
)


class TradeFlowAnalyzer:
    """Analisador de fluxo de trades."""

    def __init__(self, vol_factor_exh: float, tz_output: ZoneInfo) -> None:
        self.vol_factor_exh = vol_factor_exh
        self.tz_output = tz_output

    def analyze_window(
        self,
        window_data: List[Dict[str, Any]],
        symbol: str,
        history_volumes: List[float],
        dynamic_delta_threshold: float,
        historical_profile: Optional[Dict[str, Any]] = None,
    ):
        """
        Retorna (absorption_event, exhaustion_event)

        TODO O CÓDIGO AQUI É IDÊNTICO AO ARQUIVO ORIGINAL
        (linha por linha).
        """

        if not window_data or len(window_data) < 2:
            return (
                {
                    "is_signal": False,
                    "delta": 0,
                    "volume_total": 0,
                    "preco_fechamento": 0,
                },
                {
                    "is_signal": False,
                    "delta": 0,
                    "volume_total": 0,
                    "preco_fechamento": 0,
                },
            )

        absorption_event = create_absorption_event(
            window_data,
            symbol,
            delta_threshold=dynamic_delta_threshold,
            tz_output=self.tz_output,
            historical_profile=historical_profile,
        )

        exhaustion_event = create_exhaustion_event(
            window_data,
            symbol,
            history_volumes=list(history_volumes),
            volume_factor=self.vol_factor_exh,
            tz_output=self.tz_output,
            historical_profile=historical_profile,
        )

        return absorption_event, exhaustion_event
