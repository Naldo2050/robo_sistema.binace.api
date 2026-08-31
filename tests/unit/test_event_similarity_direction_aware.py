# tests/unit/test_event_similarity_direction_aware.py
"""
Testes unitários para a busca de similaridade de eventos com cálculo de resumo
direction-aware em EventSimilaritySearch.

Cobre:
1. Preservação integral do ranking top_k estrutural de eventos similares;
2. Cálculo de historical_win_rate considerando apenas candidatos direction-compatible;
3. Cenário de lados misturados (não misturar lados opostos na estatística de win rate);
4. Amostras compatíveis insuficientes (< 2) retornando historical_win_rate = None.
"""

import pytest
from events.event_similarity import EventSimilaritySearch


def test_build_summary_direction_compatible_filtering():
    search = EventSimilaritySearch.__new__(EventSimilaritySearch)

    # 5 eventos similares estruturais: 3 SHORTs e 2 LONGs
    similar_events = [
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Compra",  # SHORT
            "outcome": {"15m": {"direction": "DOWN", "pct": -0.8}},  # WIN para SHORT
        },
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Compra",  # SHORT
            "outcome": {"15m": {"direction": "DOWN", "pct": -1.2}},  # WIN para SHORT
        },
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Compra",  # SHORT
            "outcome": {"15m": {"direction": "UP", "pct": 0.5}},  # LOSS para SHORT
        },
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Venda",  # LONG (oposto)
            "outcome": {"15m": {"direction": "UP", "pct": 1.5}},  # WIN para LONG
        },
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Venda",  # LONG (oposto)
            "outcome": {"15m": {"direction": "DOWN", "pct": -1.0}},  # LOSS para LONG
        },
    ]

    # Consulta é um SHORT
    query_short = {
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Compra",  # SHORT
    }

    summary = search._build_summary(similar_events, current_event=query_short)

    # Contagem estrutural total intacta
    assert summary["similar_count"] == 5
    assert summary["with_outcomes"] == 5
    assert summary["query_side"] == "SHORT"
    assert summary["compatible_candidates_count"] == 3
    assert summary["compatible_samples_used"] == 3

    # Entre os 3 SHORTs compatíveis: 2 DOWN (WIN) e 1 UP (LOSS)
    # Win rate = 2 / 3 * 100 = 66.7%
    assert summary["historical_win_rate"] == 66.7
    assert summary["directional_win_rate"] == 66.7


def test_build_summary_insufficient_compatible_samples():
    search = EventSimilaritySearch.__new__(EventSimilaritySearch)

    # Apenas 1 candidato compatível SHORT e 4 candidatos LONG
    similar_events = [
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Compra",  # SHORT
            "outcome": {"15m": {"direction": "DOWN", "pct": -0.8}},
        },
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Venda",  # LONG
            "outcome": {"15m": {"direction": "UP", "pct": 1.0}},
        },
    ]

    query_short = {
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Compra",  # SHORT
    }

    summary = search._build_summary(similar_events, current_event=query_short)

    assert summary["compatible_candidates_count"] == 1
    assert summary["compatible_samples_used"] == 1
    # < 2 amostras compatíveis -> None (não inventa win_rate)
    assert summary["historical_win_rate"] is None
    assert summary["directional_win_rate"] is None


def test_build_summary_unknown_query_side():
    search = EventSimilaritySearch.__new__(EventSimilaritySearch)

    similar_events = [
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Compra",
            "outcome": {"15m": {"direction": "DOWN", "pct": -0.8}},
        }
    ]

    query_unknown = {
        "tipo_evento": "CustomEvent",
        "resultado_da_batalha": "UnmappedLabel",
    }

    summary = search._build_summary(similar_events, current_event=query_unknown)

    assert summary["query_side"] == "UNKNOWN"
    assert summary["historical_win_rate"] is None
    assert summary["directional_win_rate"] is None


def test_build_summary_flat_denominator_contract():
    """
    Testa que EventSimilarity inclui FLAT no denominador de historical_win_rate
    de forma 100% idêntica ao OutcomeTracker:
    Candidatos compatíveis: 2 WIN, 1 LOSS, 1 FLAT
    historical_win_rate = 2 / (2 + 1 + 1) * 100 = 50.0%
    directional_win_rate = 2 / (2 + 1) * 100 = 66.7%
    """
    search = EventSimilaritySearch.__new__(EventSimilaritySearch)

    similar_events = [
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Compra",  # SHORT
            "outcome": {"15m": {"direction": "DOWN", "pct": -0.8}},  # WIN (1)
        },
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Compra",  # SHORT
            "outcome": {"15m": {"direction": "DOWN", "pct": -1.2}},  # WIN (2)
        },
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Compra",  # SHORT
            "outcome": {"15m": {"direction": "UP", "pct": 0.5}},  # LOSS (1)
        },
        {
            "tipo_evento": "Absorção",
            "resultado_da_batalha": "Absorção de Compra",  # SHORT
            "outcome": {"15m": {"direction": "FLAT", "pct": 0.0}},  # FLAT (1)
        },
    ]

    query_short = {
        "tipo_evento": "Absorção",
        "resultado_da_batalha": "Absorção de Compra",
    }

    summary = search._build_summary(similar_events, current_event=query_short)

    assert summary["compatible_candidates_count"] == 4
    assert summary["compatible_samples_used"] == 4

    # 2 WINS de um total de 4 (2+1+1) => 50.0%
    assert summary["historical_win_rate"] == 50.0
    # 2 WINS de um total direcional de 3 (2+1) => 66.7%
    assert summary["directional_win_rate"] == 66.7
