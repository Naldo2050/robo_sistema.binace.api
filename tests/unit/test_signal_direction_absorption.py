# -*- coding: utf-8 -*-
"""
Teste do Bug 4: signal_direction deve ser "long" para rótulos bullish de absorção.

A expressão em market_orchestrator.py:996 (BULLISH_RESULTS) era:
    ("COMPRA", "BULLISH") -> "Absorção de Venda" (venda absorvida = bullish)
    caía no "short" por mismatch de string.

Este teste replica exatamente a expressão do código para garantir o contrato:
    signal_direction = "long" if resultado.upper() in BULLISH_RESULTS else "short"
"""
from market_orchestrator.market_orchestrator import BULLISH_RESULTS


def _signal_direction(resultado: str) -> str:
    return "long" if resultado.upper() in BULLISH_RESULTS else "short"


def test_bullish_absorcao_venda_long():
    assert _signal_direction("Absorção de Venda") == "long"


def test_bearish_absorcao_compra_short():
    assert _signal_direction("Absorção de Compra") == "short"


def test_variante_sem_acento():
    assert _signal_direction("ABSORCAO DE VENDA") == "long"


def test_exaustao_venda_long():
    assert _signal_direction("Exaustão de Venda") == "long"


def test_demanda_no_livro_long():
    assert _signal_direction("Demanda no Livro (Bid>Ask)") == "long"
    assert _signal_direction("Leve Demanda no Livro") == "long"


def test_supply_exhaustion_long():
    assert _signal_direction("SUPPLY_EXHAUSTION") == "long"


def test_legado_compra_bullish_long():
    assert _signal_direction("COMPRA") == "long"
    assert _signal_direction("BULLISH") == "long"


def test_sem_absorcao_short():
    assert _signal_direction("Sem Absorção") == "short"
    assert _signal_direction("") == "short"
