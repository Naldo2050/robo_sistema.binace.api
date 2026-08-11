# common/json_safe.py
"""
Sanitizador canônico de non-finite para serialização JSON (RFC 8259).

ETAPA 6 (auditoria forense macro/intermarket):
    NaN/±Inf nunca devem ser persistidos como literais JSON nem chegar ao
    prompt/LLM. JSON válido (RFC 8259) não admite NaN/Infinity.

Regras:
    - float NaN/+Inf/-Inf  -> None  (ausência semântica, vira JSON null)
    - numpy floating non-finite -> None
    - None  continua None
    - 0.0   continua 0
    - números finitos preservados (sem arredondamento)
    - strings/booleans/datetime/etc. intocados
    - NÃO fabrica 0, NÃO forward-fill, NÃO usa último valor válido
    - NÃO muta o objeto de entrada (retorna cópia para containers)
"""

from __future__ import annotations

import math
from typing import Any

try:
    import numpy as _np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - numpy é dependência padrão
    _np = None
    _HAS_NUMPY = False


def is_non_finite_number(value: Any) -> bool:
    """True se value é um número IEEE non-finite (NaN/±Inf).

    Inclui floats Python e numpy scalars (np.float64/np.floating).
    """
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, int):
        return False
    if _HAS_NUMPY and isinstance(value, _np.floating):
        try:
            return bool(not _np.isfinite(value))
        except (TypeError, ValueError):
            return False
    return False


def sanitize_json_safe(value: Any) -> Any:
    """Converte recursivamente NaN/±Inf em None; preserva todo o resto.

    Retorna cópia para dict/list/tuple; escalares finitos retornam o mesmo
    objeto. Nunca muta a entrada.
    """
    if isinstance(value, dict):
        return {k: sanitize_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_safe(v) for v in value]
    if is_non_finite_number(value):
        return None
    return value


def json_dumps_rfc8259(obj: Any, **kwargs: Any) -> str:
    """json.dumps que garante JSON RFC 8259 (lança em non-finite residual).

    Uso em fronteiras de serialização onde a entrada já deve estar sanitizada:
    falhar cedo revela um caminho de dados que pulou sanitize_json_safe.
    """
    kwargs.setdefault("ensure_ascii", False)
    kwargs["allow_nan"] = False
    return __import__("json").dumps(obj, **kwargs)
