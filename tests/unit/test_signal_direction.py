# tests/unit/test_signal_direction.py
"""
Testes unitários para o módulo canônico common/signal_direction.py.
Cobre normalização de strings, inferência de polaridade (Long, Short, Neutral, Unknown),
precedência canônica, classificação de outcome e extração de confiança direcional.
"""

import pytest
from common.signal_direction import (
    normalize_signal_label,
    infer_signal_side,
    classify_outcome,
    get_directional_confidence,
)


# ==============================================================================
# 1. TESTES DE NORMALIZAÇÃO
# ==============================================================================

@pytest.mark.parametrize(
    "raw_input, expected",
    [
        ("Absorção de Venda", "ABSORCAO DE VENDA"),
        ("Absorcao de Venda", "ABSORCAO DE VENDA"),
        ("  absorção   de   compra  ", "ABSORCAO DE COMPRA"),
        ("Exaustão de Venda", "EXAUSTAO DE VENDA"),
        ("Exaustão de Compra", "EXAUSTAO DE COMPRA"),
        ("Demanda no Livro (Bid>Ask)", "DEMANDA NO LIVRO (BID>ASK)"),
        ("Oferta no Livro (Ask>Bid)", "OFERTA NO LIVRO (ASK>BID)"),
        ("Vencedores: Vendedores", "VENCEDORES: VENDEDORES"),
        ("Sem Absorção", "SEM ABSORCAO"),
        ("Equilíbrio", "EQUILIBRIO"),
        ("N/A", "N/A"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_signal_label(raw_input, expected):
    assert normalize_signal_label(raw_input) == expected


# ==============================================================================
# 2. TESTES DE INFERÊNCIA DE POLARIDADE (infer_signal_side)
# ==============================================================================

@pytest.mark.parametrize(
    "battle_result, event_type, explicit_side, expected_side",
    [
        # Sinais Bullish / Long
        ("Absorção de Venda", "Absorção", None, "LONG"),
        ("Absorcao de Venda", "Absorção", None, "LONG"),
        ("ABSORÇÃO DE VENDA", None, None, "LONG"),
        ("Absorção de Venda (Bullish)", "Absorção", None, "LONG"),
        ("Exaustão de Venda", "Exaustão", None, "LONG"),
        ("Exaustao de Venda", "Exaustão", None, "LONG"),
        ("Demanda no Livro (Bid>Ask)", "OrderBook", None, "LONG"),
        ("Leve Demanda no Livro", "OrderBook", None, "LONG"),
        ("Demanda Forte", "OrderBook", None, "LONG"),
        ("COMPRA", "OrderBook", None, "LONG"),
        ("BULLISH", "OrderBook", None, "LONG"),
        ("SUPPLY_EXHAUSTION", "Alerta", None, "LONG"),

        # Sinais Bearish / Short
        ("Absorção de Compra", "Absorção", None, "SHORT"),
        ("Absorcao de Compra", "Absorção", None, "SHORT"),
        ("ABSORÇÃO DE COMPRA", None, None, "SHORT"),
        ("Absorção de Compra (Bearish)", "Absorção", None, "SHORT"),
        ("Exaustão de Compra", "Exaustão", None, "SHORT"),
        ("Exaustao de Compra", "Exaustão", None, "SHORT"),
        ("Oferta no Livro (Ask>Bid)", "OrderBook", None, "SHORT"),
        ("Leve Oferta no Livro", "OrderBook", None, "SHORT"),
        ("Vencedores: Vendedores", "OrderBook", None, "SHORT"),
        ("VENDEDOR_VENCEDOR", "OrderBook", None, "SHORT"),
        ("Vendedores", "OrderBook", None, "SHORT"),
        ("VENDA", "OrderBook", None, "SHORT"),
        ("BEARISH", "OrderBook", None, "SHORT"),
        ("DEMAND_EXHAUSTION", "Alerta", None, "SHORT"),

        # Sinais Neutros
        ("Sem Absorção", "Absorção", None, "NEUTRAL"),
        ("Sem Exaustão", "Exaustão", None, "NEUTRAL"),
        ("Equilíbrio", "OrderBook", None, "NEUTRAL"),
        ("NEUTRAL", "OrderBook", None, "NEUTRAL"),
        ("N/A", "ANALYSIS_TRIGGER", None, "NEUTRAL"),
        ("NONE", "Alerta", None, "NEUTRAL"),
        ("INDISPONÍVEL", "OrderBook", None, "NEUTRAL"),
        ("Dados inválidos", "OrderBook", None, "NEUTRAL"),
        ("Janela vazia", "Absorção", None, "NEUTRAL"),
        ("Preços inválidos", "Absorção", None, "NEUTRAL"),
        ("Erro", "Absorção", None, "NEUTRAL"),
        ("EMERGÊNCIA", "OrderBook", None, "NEUTRAL"),
        ("VOLATILITY_EXPANSION", "Alerta", None, "NEUTRAL"),
        ("VOLATILITY_SQUEEZE", "Alerta", None, "NEUTRAL"),

        # Precedência: battle_result prevalece sobre explicit_side inconsistente
        ("Absorção de Venda", "Absorção", "sell", "LONG"),
        ("Absorção de Compra", "Absorção", "buy", "SHORT"),

        # Fallback de event_type quando battle_result é vazio
        (None, "Absorção de Venda", None, "LONG"),
        ("", "Absorção de Compra", None, "SHORT"),

        # Fallback de explicit_side quando battle_result e event_type são neutros/vazios
        (None, None, "buy", "LONG"),
        ("", "", "sell", "SHORT"),
        (None, None, "long", "LONG"),
        ("", "", "short", "SHORT"),

        # Sinais Desconhecidos (UNKNOWN) - nunca assume LONG
        ("StringDesconhecida", "TipoInexistente", None, "UNKNOWN"),
        (None, None, None, "UNKNOWN"),
        ("", "", "", "UNKNOWN"),
    ],
)
def test_infer_signal_side(battle_result, event_type, explicit_side, expected_side):
    assert infer_signal_side(event_type, battle_result, explicit_side) == expected_side


# ==============================================================================
# 3. TESTES DA MATRIZ DE OUTCOME (classify_outcome)
# ==============================================================================

@pytest.mark.parametrize(
    "signal_side, outcome_direction, expected_result",
    [
        # LONG
        ("LONG", "UP", "WIN"),
        ("LONG", "DOWN", "LOSS"),
        ("LONG", "FLAT", "FLAT"),
        ("LONG", "UNKNOWN", "UNKNOWN"),
        ("LONG", None, "UNKNOWN"),

        # SHORT
        ("SHORT", "DOWN", "WIN"),
        ("SHORT", "UP", "LOSS"),
        ("SHORT", "FLAT", "FLAT"),
        ("SHORT", "UNKNOWN", "UNKNOWN"),
        ("SHORT", None, "UNKNOWN"),

        # NEUTRAL / UNKNOWN
        ("NEUTRAL", "UP", "UNKNOWN"),
        ("NEUTRAL", "DOWN", "UNKNOWN"),
        ("NEUTRAL", "FLAT", "UNKNOWN"),
        ("UNKNOWN", "UP", "UNKNOWN"),
        ("UNKNOWN", "DOWN", "UNKNOWN"),
        ("UNKNOWN", "FLAT", "UNKNOWN"),
        ("INVALID", "UP", "UNKNOWN"),
    ],
)
def test_classify_outcome(signal_side, outcome_direction, expected_result):
    assert classify_outcome(signal_side, outcome_direction) == expected_result


# ==============================================================================
# 4. TESTES DE RESOLUÇÃO DE CONFIANÇA DIRECIONAL (get_directional_confidence)
# ==============================================================================

def test_get_directional_confidence():
    conf = {"long_prob": 0.15, "short_prob": 0.85, "neutral_prob": 0.0}

    # SHORT deve usar short_prob
    assert get_directional_confidence(conf, "short") == 0.85
    assert get_directional_confidence(conf, "SHORT") == 0.85

    # LONG deve usar long_prob
    assert get_directional_confidence(conf, "long") == 0.15
    assert get_directional_confidence(conf, "LONG") == 0.15

    # NEUTRAL / UNKNOWN / None deve usar fallback seguro 0.5
    assert get_directional_confidence(conf, "neutral") == 0.5
    assert get_directional_confidence(conf, "unknown") == 0.5
    assert get_directional_confidence(conf, None) == 0.5
    assert get_directional_confidence({}, "short") == 0.5
    assert get_directional_confidence(None, "long") == 0.5

    # Proteção contra NaN, Inf, valores fora da faixa [0, 1] ou inválidos
    nan_conf = {"long_prob": float("nan"), "short_prob": float("inf")}
    assert get_directional_confidence(nan_conf, "long") == 0.5
    assert get_directional_confidence(nan_conf, "short") == 0.5

    out_of_range_conf = {"long_prob": 1.5, "short_prob": -0.2}
    assert get_directional_confidence(out_of_range_conf, "long") == 0.5
    assert get_directional_confidence(out_of_range_conf, "short") == 0.5


def test_precedence_conflict_battle_result_overrides_explicit_side():
    """
    Garante que battle_result canônico tem precedência absoluta sobre explicit_side,
    impedindo que side ambíguo (taker vs maker vs absorption side) inverta a polaridade.
    """
    # battle_result='Absorção de Compra' (venda de mercado absorvida -> SHORT)
    # explicit_side='buy' (ex: agressor de compra ou lado comprador)
    # Resultado DEVE permanecer SHORT
    assert infer_signal_side(
        event_type="Absorção",
        battle_result="Absorção de Compra",
        explicit_side="buy",
    ) == "SHORT"

    # battle_result='Absorção de Venda' (compra de mercado absorvida -> LONG)
    # explicit_side='sell'
    # Resultado DEVE permanecer LONG
    assert infer_signal_side(
        event_type="Absorção",
        battle_result="Absorção de Venda",
        explicit_side="sell",
    ) == "LONG"
