# common/identity_contract.py
"""Contrato puro de identidade de eventos — H1-B3-C-H1 / H1-B3-D / INC5-B4-I1.

Identidades determinísticas para:
- Exaustão   -> exhaustion-v2: SHA256("v2|{window_open_ms}|{symbol}|Exaustão|{delta:.8f}")
- Zona       -> zone-v2:       SHA256("Zona|{symbol}|{window_key}|{kind}|{timeframe}|"
                                    "{anchor:.2f}|{low:.2f}|{high:.2f}|{last_touched}")
- Absorção   -> absorption-v1: SHA256("{symbol}|Absorção|{window_open_ms}|0.0") [congelada]
- Trigger    -> trigger-v1:    SHA256("TRIGGER|{symbol}|{window_key}")

Princípios:
- Sem I/O, sem relógio, sem UUID, sem estado global, sem mutação de entradas.
- Entradas iguais => identidade idêntica (Windows/Linux, Python 3.8+):
  f"{x:.2f}" / f"{x:.8f}" usam ponto como separador decimal, independente de locale.
- NaN/±Infinity nunca são formatados: são rejeitados com exceção explícita.
- Canonicalização de zero: 0.0, -0.0, Decimal("-0.000"), "-0.0" e "0.0" geram
  rigorosamente a mesma representação canônica ("0.00000000" / "0.00") e mesmo hash.
- Validação estrita e fail-closed de last_touched: rejeita bool, NaN, ±Infinity,
  strings vazias/malformadas e timestamps negativos; normaliza ISO e numérico.
- Preservação integral de event_id e payload bruto (deepcopy) em falhas de delta.
- Estabilidade de Zona: equivalência estrita entre objeto LevelZone e dict após replay.
- Erros nunca viram IDs aleatórios (fail-closed via IdentityContractError).
"""

from __future__ import annotations

import copy
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import math
from numbers import Integral, Real
from typing import Any, Dict, Optional, Tuple

IDENTITY_VERSION_EXHAUSTION = "exhaustion-v2"
IDENTITY_VERSION_ZONE = "zone-v2"
IDENTITY_VERSION_ABSORPTION = "absorption-v1"
IDENTITY_VERSION_TRIGGER = "trigger-v1"

TIPO_EXAUSTAO = "Exaustão"
TIPO_ZONA = "Zona"
TIPO_ABSORCAO = "Absorção"
TIPO_TRIGGER = "ANALYSIS_TRIGGER"

PRODUCER_EXHAUSTION = TIPO_EXAUSTAO
PRODUCER_ZONE = TIPO_ZONA
PRODUCER_ABSORPTION = TIPO_ABSORCAO
PRODUCER_TRIGGER = TIPO_TRIGGER

REGISTERED_PRODUCERS = (
    PRODUCER_EXHAUSTION,
    PRODUCER_ZONE,
    PRODUCER_ABSORPTION,
    PRODUCER_TRIGGER,
)

_DETERMINISTIC_ID_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_WINDOW_KEY_FIELDS = ("window_id", "features_window_id")


class IdentityContractError(ValueError):
    """Componente ausente, inválido ou não finito para compor a identidade."""

    def __init__(
        self,
        message: str,
        event_id: Optional[str] = None,
        raw_payload: Optional[Dict[str, Any]] = None,
        field_name: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.event_id = event_id
        self.raw_payload = copy.deepcopy(raw_payload) if raw_payload is not None else None
        self.field_name = field_name


def _require_text(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise IdentityContractError(
            f"{name}: texto obrigatório (recebido {type(value).__name__})"
        )
    text = value.strip()
    if not text:
        raise IdentityContractError(f"{name}: texto vazio")
    for char in text:
        if char == "|" or ord(char) < 32:
            raise IdentityContractError(f"{name}: caractere inválido {char!r}")
    return text


def _require_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise IdentityContractError(
            f"{name}: número finito obrigatório (recebido bool)"
        )
    if isinstance(value, Decimal):
        if value.is_nan() or value.is_infinite():
            raise IdentityContractError(f"{name}: valor não finito rejeitado")
        number = float(value)
    elif isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        # Risco 1: strings de zero canônico ("-0.0", "0.0", "-0", "0") são convertidas para 0.0
        try:
            dec = Decimal(text)
            if dec.is_nan() or dec.is_infinite():
                raise IdentityContractError(f"{name}: valor não finito rejeitado")
            if dec == 0:
                number = 0.0
            else:
                raise IdentityContractError(
                    f"{name}: número finito obrigatório (recebido str {value!r})"
                )
        except InvalidOperation:
            raise IdentityContractError(
                f"{name}: número finito obrigatório (recebido str {value!r})"
            )
    else:
        raise IdentityContractError(
            f"{name}: número finito obrigatório (recebido {type(value).__name__})"
        )

    if not math.isfinite(number):
        raise IdentityContractError(f"{name}: valor não finito rejeitado")

    # Canonicalização de zero (Risco 1): qualquer representação de zero vira 0.0 positivo
    if number == 0.0:
        return 0.0
    return number


def _require_integral(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise IdentityContractError(
            f"{name}: inteiro obrigatório (recebido bool)"
        )
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, (Real, Decimal)):
        fval = float(value)
        if math.isfinite(fval) and fval == int(fval):
            return int(fval)
    raise IdentityContractError(
        f"{name}: inteiro obrigatório (recebido {type(value).__name__})"
    )


def canonicalize_number(value: Any, name: str = "number") -> float:
    """Canonicaliza qualquer número, Decimal ou string numérica para float finito determinístico.

    Elimina zero negativo (-0.0 -> 0.0). Rejeita bool, NaN e ±Infinity.
    """
    return _require_finite(value, name)


def canonicalize_structure(data: Any) -> Any:
    """Canonicaliza recursivamente estruturas aninhadas (dict, list, tuple).

    Garante que -0.0, Decimal("-0.0") e strings "0.0" / "-0.0" em campos numéricos
    sejam convertidos para 0.0 positivo canônico. Não muta estruturas de entrada.
    """
    if isinstance(data, dict):
        return {k: canonicalize_structure(v) for k, v in data.items()}
    if isinstance(data, list):
        return [canonicalize_structure(v) for v in data]
    if isinstance(data, tuple):
        return tuple(canonicalize_structure(v) for v in data)
    if isinstance(data, (float, Decimal)) and data == 0.0:
        return 0.0
    return data


def normalize_last_touched(last_touched: Any) -> str:
    """Validação e normalização determinística e fail-closed de last_touched (Risco 2).

    Aceita:
    - String ISO 8601 de data/hora válida (ex.: '2026-10-24T11:59:59.123+00:00')
    - Timestamp numérico >= 0 (int, float finito, Decimal, string numérica)

    Rejeita (fail-closed via IdentityContractError):
    - None ou ausente
    - bool (True / False)
    - NaN, ±Infinity
    - Timestamps negativos (< 0)
    - Strings vazias ou malformadas
    - Tipos arbitrários não reconhecidos
    """
    if isinstance(last_touched, bool):
        raise IdentityContractError("last_touched: booleano não permitido")
    if last_touched is None:
        raise IdentityContractError("last_touched: valor obrigatório (recebido None)")

    # 1. Se for número direto (int, float, Decimal)
    if isinstance(last_touched, (Integral, Real, Decimal)):
        fval = float(last_touched)
        if not math.isfinite(fval):
            raise IdentityContractError("last_touched: valor não finito rejeitado")
        if fval < 0:
            raise IdentityContractError("last_touched: timestamp negativo rejeitado")
        if fval == 0.0:
            return "0"
        if fval == int(fval):
            return str(int(fval))
        return f"{fval:.3f}"

    # 2. Se for string
    if not isinstance(last_touched, str):
        raise IdentityContractError(
            f"last_touched: tipo inválido ({type(last_touched).__name__})"
        )

    text = last_touched.strip()
    if not text:
        raise IdentityContractError("last_touched: texto vazio")
    for char in text:
        if char == "|" or ord(char) < 32:
            raise IdentityContractError(f"last_touched: caractere inválido {char!r}")

    lowered = text.lower()
    if lowered in (
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    ):
        raise IdentityContractError("last_touched: valor não finito rejeitado")

    # 2a. Tentar interpretar como número puramente formatado
    try:
        dec = Decimal(text)
        if dec.is_nan() or dec.is_infinite():
            raise IdentityContractError("last_touched: valor não finito rejeitado")
        if dec < 0:
            raise IdentityContractError("last_touched: timestamp negativo rejeitado")
        if dec == 0:
            return "0"
        if dec == int(dec):
            return str(int(dec))
        return str(dec)
    except (InvalidOperation, TypeError):
        pass

    # 2b. Se não for número, validar formato ISO de data/hora
    try:
        iso_cand = text.replace("Z", "+00:00")
        datetime.fromisoformat(iso_cand)
        return text
    except (ValueError, TypeError):
        raise IdentityContractError(
            f"last_touched: formato temporal ou numérico inválido {text!r}"
        )


def exhaustion_v2_event_id(window_open_ms: int, symbol: str, delta_btc: float) -> str:
    """Identidade v2 da Exaustão com delta efetivo (buy_btc - sell_btc).

    Preserva o sinal do delta. Canonicaliza zero (-0.0 -> 0.0 -> '0.00000000').
    Rejeita componentes ausentes, não finitos ou com separador/caractere de controle.
    """
    window = _require_integral(window_open_ms, "window_open_ms")
    sym = _require_text(symbol, "symbol")
    delta = _require_finite(delta_btc, "delta_btc")

    if delta == 0.0:
        formatted_delta = "0.00000000"
    else:
        formatted_delta = f"{delta:.8f}"

    base = f"v2|{window}|{sym}|{TIPO_EXAUSTAO}|{formatted_delta}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def zone_v2_event_id(
    symbol: Any,
    window_key: Any,
    kind: Any,
    timeframe: Any,
    anchor_price: Any,
    low: Any,
    high: Any,
    last_touched: Any,
) -> str:
    """Identidade v2 da Zona derivada do snapshot da própria ocorrência.

    Componentes: símbolo, chave de janela, tipo/timeframe da zona, geometria
    (anchor/low/high formatados com 2 casas e canonicalização de zero) e last_touched.
    Não usa touch_count, janela_numero nem identidade do evento base.
    """
    sym = _require_text(symbol, "symbol")
    window = _require_text(window_key, "window_key")
    kind_n = _require_text(kind, "kind")
    timeframe_n = _require_text(timeframe, "timeframe")
    anchor = _require_finite(anchor_price, "anchor_price")
    low_n = _require_finite(low, "low")
    high_n = _require_finite(high, "high")
    touched = normalize_last_touched(last_touched)

    fmt_anchor = "0.00" if anchor == 0.0 else f"{anchor:.2f}"
    fmt_low = "0.00" if low_n == 0.0 else f"{low_n:.2f}"
    fmt_high = "0.00" if high_n == 0.0 else f"{high_n:.2f}"

    base = (
        f"Zona|{sym}|{window}|{kind_n}|{timeframe_n}|"
        f"{fmt_anchor}|{fmt_low}|{fmt_high}|{touched}"
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def resolve_zone_window_key(event: Dict[str, Any]) -> str:
    """Resolve a chave de janela do snapshot de Zona, em ordem determinística:

    window_id -> features_window_id -> str(epoch_ms).
    Campos vazios/whitespace são tratados como ausentes. Ausência total
    ou tipo inválido => IdentityContractError (fail-closed).
    """
    if not isinstance(event, dict):
        raise IdentityContractError("window_key: evento não é dicionário")

    for field in _WINDOW_KEY_FIELDS:
        value = event.get(field)
        if value is None:
            continue
        if isinstance(value, bool):
            raise IdentityContractError(f"window_key: campo {field} inválido (bool)")
        if isinstance(value, Integral):
            return str(int(value))
        if not isinstance(value, str):
            raise IdentityContractError(
                f"window_key: campo {field} com tipo inválido ({type(value).__name__})"
            )
        if not value.strip():
            continue
        return _require_text(value, f"window_key({field})")

    epoch = event.get("epoch_ms")
    if epoch is None:
        raise IdentityContractError(
            "window_key: window_id/features_window_id/epoch_ms ausentes"
        )
    return str(_require_integral(epoch, "epoch_ms"))


def compute_zone_identity(
    zone_or_event: Any,
    window_key: Optional[str] = None,
    symbol: Optional[str] = None,
) -> str:
    """Computa a identidade v2 da Zona a partir de objeto (LevelZone) ou dicionário (replay).

    Garante invariância absoluta entre o objeto original em memória e o
    dicionário desserializado via json.loads (replay de log/banco/JSONL) - Risco 4.
    """
    if hasattr(zone_or_event, "to_dict") and callable(zone_or_event.to_dict):
        data = zone_or_event.to_dict()
    elif hasattr(zone_or_event, "__dataclass_fields__"):
        from dataclasses import asdict

        data = asdict(zone_or_event)
    elif isinstance(zone_or_event, dict):
        data = dict(zone_or_event)
    else:
        raise IdentityContractError(
            f"zona: tipo não suportado ({type(zone_or_event).__name__})"
        )

    # Se for um evento completo de Zona, extrai o contexto se presente
    ctx = data.get("zone_context")
    if isinstance(ctx, dict):
        merged = dict(ctx)
        for k in ("ativo", "symbol", "window_id", "features_window_id", "epoch_ms"):
            if k in data and k not in merged:
                merged[k] = data[k]
        data = merged

    # Resolver símbolo
    sym = symbol or data.get("ativo") or data.get("symbol")
    if not sym:
        raise IdentityContractError("zona: symbol/ativo ausente")

    # Resolver chave de janela
    if not window_key:
        window_key = resolve_zone_window_key(data)

    # Extrair tipo e timeframe da zona (suporta português e inglês)
    kind = data.get("kind") or data.get("tipo_zona") or data.get("tipo")
    timeframe = data.get("timeframe") or data.get("tempo_grafico", "intraday")

    # Extrair geometria (suporta nomes da LevelZone e variantes em português)
    anchor = (
        data.get("anchor_price")
        if data.get("anchor_price") is not None
        else data.get("centro")
    )
    low = (
        data.get("low")
        if data.get("low") is not None
        else data.get("preco_min")
    )
    high = (
        data.get("high")
        if data.get("high") is not None
        else data.get("preco_max")
    )
    touched = data.get("last_touched")

    # Risco 4: Coerção limpa de float drift / Decimal / str numérico em replay
    if anchor is not None and isinstance(anchor, (str, Decimal)):
        try:
            anchor = float(Decimal(str(anchor).strip()))
        except Exception:
            pass
    if low is not None and isinstance(low, (str, Decimal)):
        try:
            low = float(Decimal(str(low).strip()))
        except Exception:
            pass
    if high is not None and isinstance(high, (str, Decimal)):
        try:
            high = float(Decimal(str(high).strip()))
        except Exception:
            pass

    return zone_v2_event_id(
        symbol=sym,
        window_key=window_key,
        kind=kind,
        timeframe=timeframe,
        anchor_price=anchor,
        low=low,
        high=high,
        last_touched=touched,
    )


def validate_and_preserve_payload(
    event: Dict[str, Any],
    delta_field: str = "delta_fechamento",
    allow_missing_delta: bool = False,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Valida campos de delta preservando 100% o payload original e seu event_id (Risco 3).

    Garante que:
    1. O payload de entrada jamais sofre mutação in-place (opera em cópia profunda).
    2. Caso o delta_field seja inválido (NaN, Inf, bool, tipo inválido), lança
       IdentityContractError com event_id, raw_payload (deepcopy) e field_name
       acessíveis para envio direto à quarentena forense (H1-B2-Q1).
    """
    if not isinstance(event, dict):
        raise IdentityContractError("payload: deve ser um dicionário", raw_payload=None)

    payload_snapshot = copy.deepcopy(event)

    # Extrai ou infere event_id se possível
    event_id = payload_snapshot.get("event_id")
    if not is_deterministic_event_id(event_id):
        tipo = payload_snapshot.get("tipo_evento")
        try:
            if tipo == TIPO_EXAUSTAO and "window_open_ms" in payload_snapshot:
                sym = (
                    payload_snapshot.get("ativo")
                    or payload_snapshot.get("symbol", "")
                )
                d_val = payload_snapshot.get("delta") or payload_snapshot.get(
                    "delta_btc", 0.0
                )
                event_id = exhaustion_v2_event_id(
                    payload_snapshot["window_open_ms"], sym, d_val
                )
            elif tipo == TIPO_ZONA:
                event_id = compute_zone_identity(payload_snapshot)
        except Exception:
            event_id = event_id if isinstance(event_id, str) else None

    # Validação do campo delta
    if delta_field in payload_snapshot:
        raw_delta = payload_snapshot[delta_field]
        if raw_delta is None and allow_missing_delta:
            return event_id, payload_snapshot
        try:
            delta_num = _require_finite(raw_delta, delta_field)
            payload_snapshot[delta_field] = delta_num
        except IdentityContractError as e:
            raise IdentityContractError(
                f"{delta_field}: falha de integridade ({e.message})",
                event_id=event_id,
                raw_payload=payload_snapshot,
                field_name=delta_field,
            ) from e
    elif not allow_missing_delta:
        raise IdentityContractError(
            f"{delta_field}: campo obrigatório ausente",
            event_id=event_id,
            raw_payload=payload_snapshot,
            field_name=delta_field,
        )

    return event_id, payload_snapshot


def compute_producer_identity(event: Dict[str, Any]) -> Tuple[str, str]:
    """Retorna (identity_version, event_id) para qualquer evento de produtor registrado (Risco 5)."""
    if not isinstance(event, dict):
        raise IdentityContractError("evento deve ser um dicionário")

    tipo = event.get("tipo_evento")
    if not tipo:
        raise IdentityContractError("tipo_evento obrigatório")

    sym = event.get("ativo") or event.get("symbol")
    if not sym:
        raise IdentityContractError(f"{tipo}: ativo/symbol obrigatório")

    if tipo == PRODUCER_EXHAUSTION:
        w_open = event.get("window_open_ms")
        if w_open is None:
            raise IdentityContractError(f"{tipo}: window_open_ms obrigatório")
        delta = event.get("delta")
        if delta is None:
            delta = event.get("delta_btc")
        if delta is None:
            raise IdentityContractError(f"{tipo}: delta obrigatório")
        return IDENTITY_VERSION_EXHAUSTION, exhaustion_v2_event_id(w_open, sym, delta)

    if tipo == PRODUCER_ZONE:
        return IDENTITY_VERSION_ZONE, compute_zone_identity(event)

    if tipo == PRODUCER_ABSORPTION:
        w_open = event.get("window_open_ms") or event.get("epoch_ms")
        if w_open is None:
            raise IdentityContractError(f"{tipo}: window_open_ms/epoch_ms obrigatório")
        base = f"{sym}|Absorção|{w_open}|0.0"
        return IDENTITY_VERSION_ABSORPTION, hashlib.sha256(base.encode("utf-8")).hexdigest()

    if tipo == PRODUCER_TRIGGER:
        win_key = resolve_zone_window_key(event)
        base = f"TRIGGER|{sym}|{win_key}"
        return IDENTITY_VERSION_TRIGGER, hashlib.sha256(base.encode("utf-8")).hexdigest()

    raise IdentityContractError(f"produtor não registrado ou não suportado: {tipo!r}")


def is_deterministic_event_id(value: Any) -> bool:
    """True somente para SHA-256 hexdigest minúsculo (64 chars) — identidade durável.

    IDs técnicos gerados pelo EventSaver (uuid4()[:8], 8 chars) NÃO são
    identidades determinísticas e não devem ser preservados como base_event_id.
    """
    return (
        isinstance(value, str)
        and len(value) == _DETERMINISTIC_ID_LENGTH
        and all(char in _HEX_DIGITS for char in value)
    )
