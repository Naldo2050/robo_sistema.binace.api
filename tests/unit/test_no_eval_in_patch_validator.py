# tests/unit/test_no_eval_in_patch_validator.py
"""
Regressão de segurança (SEC-4): tests/unit/test_patch_validator.py não deve
usar eval() — mesmo em teste, o padrão é inseguro e pode ser copiado para prod.

Contrato exigido:
1. O fonte de test_patch_validator.py não contém chamada a eval(.
2. ast.literal_eval cobre os formatos usados no teste ('[68425, 69153]',
   '68425') e REJEITA payload malicioso (ex: __import__/os.system),
   o que eval() executaria.
"""

import ast
import re
from pathlib import Path

import pytest

SOURCE = Path("tests/unit/test_patch_validator.py").read_text(encoding="utf-8")


def test_no_bare_eval_in_patch_validator():
    """Falha enquanto houver eval( no fonte (exclui literal_eval)."""
    bare = [
        line
        for line in SOURCE.splitlines()
        if re.search(r"(?<![\w.])eval\s*\(", line)
    ]
    assert bare == []


@pytest.mark.parametrize(
    "zone_input, expected",
    [("[68425, 69153]", [68425, 69153]), ("68425", 68425)],
)
def test_literal_eval_parses_zone_formats(zone_input, expected):
    assert ast.literal_eval(zone_input) == expected


def test_literal_eval_rejects_code_execution():
    with pytest.raises((ValueError, SyntaxError)):
        ast.literal_eval("__import__('os').system('echo pwned')")
