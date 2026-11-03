# -*- coding: utf-8 -*-
"""Testes herméticos das funções puras de identidade (H1-B3-D).

Cobre Exaustão v2, Zona v2, resolução de window_key e detecção de
identidade determinística. Sem I/O, sem relógio, sem dados operacionais.
"""

import copy
from decimal import Decimal
import hashlib
import json

import pytest

from common.identity_contract import (
    IDENTITY_VERSION_ABSORPTION,
    IDENTITY_VERSION_EXHAUSTION,
    IDENTITY_VERSION_TRIGGER,
    IDENTITY_VERSION_ZONE,
    REGISTERED_PRODUCERS,
    IdentityContractError,
    canonicalize_number,
    canonicalize_structure,
    compute_producer_identity,
    compute_zone_identity,
    exhaustion_v2_event_id,
    is_deterministic_event_id,
    normalize_last_touched,
    resolve_zone_window_key,
    validate_and_preserve_payload,
    zone_v2_event_id,
)

SYMBOL = "BTCUSDT"
WINDOW = 1700000060000
LAST_TOUCHED = "2026-10-24T11:59:59.123+00:00"


def _manual_exhaustion(window, symbol, delta):
    base = f"v2|{window}|{symbol}|Exaustão|{delta:.8f}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _manual_zone(symbol, window, kind, timeframe, anchor, low, high, touched):
    base = (
        f"Zona|{symbol}|{window}|{kind}|{timeframe}|"
        f"{anchor:.2f}|{low:.2f}|{high:.2f}|{touched}"
    )
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# =============================================================================
# EXAUSTÃO V2
# =============================================================================
def test_exhaustion_v2_formula_manual_e_determinismo():
    eid = exhaustion_v2_event_id(WINDOW, SYMBOL, 2.9)
    assert eid == _manual_exhaustion(WINDOW, SYMBOL, 2.9)
    assert eid == exhaustion_v2_event_id(WINDOW, SYMBOL, 2.9)
    assert IDENTITY_VERSION_EXHAUSTION == "exhaustion-v2"


def test_exhaustion_v2_positivo_negativo_zero_distintos():
    pos = exhaustion_v2_event_id(WINDOW, SYMBOL, 2.9)
    neg = exhaustion_v2_event_id(WINDOW, SYMBOL, -2.9)
    zero = exhaustion_v2_event_id(WINDOW, SYMBOL, 0.0)
    assert len({pos, neg, zero}) == 3
    assert neg == _manual_exhaustion(WINDOW, SYMBOL, -2.9)
    assert zero == _manual_exhaustion(WINDOW, SYMBOL, 0.0)


def test_exhaustion_v2_tiny_delta_formata_zero_mas_prefixo_impede_colisao_v1():
    tiny_pos = 2.9e-9
    tiny_neg = -2.9e-9
    assert f"{tiny_pos:.8f}" == "0.00000000"
    assert f"{tiny_neg:.8f}" == "-0.00000000"

    id_tiny_pos = exhaustion_v2_event_id(WINDOW, SYMBOL, tiny_pos)
    id_tiny_neg = exhaustion_v2_event_id(WINDOW, SYMBOL, tiny_neg)
    assert id_tiny_pos == _manual_exhaustion(WINDOW, SYMBOL, tiny_pos)
    assert id_tiny_neg == _manual_exhaustion(WINDOW, SYMBOL, tiny_neg)
    assert id_tiny_pos != id_tiny_neg

    v1_id = hashlib.sha256(
        f"{WINDOW}|{SYMBOL}|Exaustão|0.00000000".encode("utf-8")
    ).hexdigest()
    assert id_tiny_pos != v1_id
    assert id_tiny_neg != v1_id


@pytest.mark.parametrize("delta", [float("nan"), float("inf"), float("-inf")])
def test_exhaustion_v2_rejeita_nao_finito(delta):
    with pytest.raises(IdentityContractError):
        exhaustion_v2_event_id(WINDOW, SYMBOL, delta)


@pytest.mark.parametrize("window", [WINDOW + 0.5, None, True, "1700000060000", [WINDOW]])
def test_exhaustion_v2_rejeita_janela_invalida(window):
    with pytest.raises(IdentityContractError):
        exhaustion_v2_event_id(window, SYMBOL, 2.9)


@pytest.mark.parametrize("symbol", [None, "", "   ", 123, "BTC|USDT", "BTC\x00USDT"])
def test_exhaustion_v2_rejeita_symbol_invalido(symbol):
    with pytest.raises(IdentityContractError):
        exhaustion_v2_event_id(WINDOW, symbol, 2.9)


@pytest.mark.parametrize("delta", [None, "2.9", True, [1.0]])
def test_exhaustion_v2_rejeita_delta_de_tipo_invalido(delta):
    with pytest.raises(IdentityContractError):
        exhaustion_v2_event_id(WINDOW, SYMBOL, delta)


# =============================================================================
# ZONA V2
# =============================================================================
def test_zone_v2_formula_manual_e_determinismo():
    args = (SYMBOL, str(WINDOW), "POC", "daily", 100.0, 95.5, 104.5, LAST_TOUCHED)
    zid = zone_v2_event_id(*args)
    assert zid == _manual_zone(*args)
    assert zid == zone_v2_event_id(*args)
    assert IDENTITY_VERSION_ZONE == "zone-v2"


def test_zone_v2_last_touched_distinto_gera_id_distinto():
    t1 = "2026-10-24T11:59:59.123+00:00"
    t2 = "2026-10-24T12:10:05.987+00:00"
    id1 = zone_v2_event_id(SYMBOL, str(WINDOW), "POC", "daily", 100.0, 95.5, 104.5, t1)
    id2 = zone_v2_event_id(SYMBOL, str(WINDOW), "POC", "daily", 100.0, 95.5, 104.5, t2)
    assert id1 != id2


def test_zone_v2_kind_timeframe_geometria_alteram_id():
    base_id = zone_v2_event_id(SYMBOL, str(WINDOW), "POC", "daily", 100.0, 95.5, 104.5, LAST_TOUCHED)
    kind_id = zone_v2_event_id(SYMBOL, str(WINDOW), "VAL", "daily", 100.0, 95.5, 104.5, LAST_TOUCHED)
    tf_id = zone_v2_event_id(SYMBOL, str(WINDOW), "POC", "weekly", 100.0, 95.5, 104.5, LAST_TOUCHED)
    geo_id = zone_v2_event_id(SYMBOL, str(WINDOW), "POC", "daily", 100.01, 95.5, 104.5, LAST_TOUCHED)
    win_id = zone_v2_event_id(SYMBOL, "1700000120000", "POC", "daily", 100.0, 95.5, 104.5, LAST_TOUCHED)
    assert len({base_id, kind_id, tf_id, geo_id, win_id}) == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"symbol": None},
        {"symbol": ""},
        {"window_key": None},
        {"window_key": ""},
        {"kind": None},
        {"timeframe": ""},
        {"last_touched": None},
        {"last_touched": "   "},
        {"anchor_price": float("nan")},
        {"low": float("inf")},
        {"high": float("-inf")},
        {"anchor_price": "100.0"},
        {"anchor_price": True},
        {"kind": "POC|x"},
        {"symbol": "BTC|USDT"},
    ],
)
def test_zone_v2_rejeita_componentes_invalidos(overrides):
    params = {
        "symbol": SYMBOL,
        "window_key": str(WINDOW),
        "kind": "POC",
        "timeframe": "daily",
        "anchor_price": 100.0,
        "low": 95.5,
        "high": 104.5,
        "last_touched": LAST_TOUCHED,
    }
    params.update(overrides)
    with pytest.raises(IdentityContractError):
        zone_v2_event_id(
            params["symbol"],
            params["window_key"],
            params["kind"],
            params["timeframe"],
            params["anchor_price"],
            params["low"],
            params["high"],
            params["last_touched"],
        )


# =============================================================================
# RESOLUÇÃO DE WINDOW_KEY
# =============================================================================
def test_resolve_zone_window_key_prioridade():
    assert resolve_zone_window_key(
        {"window_id": "111", "features_window_id": "222", "epoch_ms": 333}
    ) == "111"
    assert resolve_zone_window_key(
        {"features_window_id": "222", "epoch_ms": 333}
    ) == "222"
    assert resolve_zone_window_key({"epoch_ms": 333}) == "333"
    assert resolve_zone_window_key({"window_id": 111, "epoch_ms": 333}) == "111"


def test_resolve_zone_window_key_campos_vazios_sao_ausentes():
    assert resolve_zone_window_key(
        {"window_id": "  ", "features_window_id": "222", "epoch_ms": 333}
    ) == "222"
    assert resolve_zone_window_key({"window_id": None, "epoch_ms": 333}) == "333"


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"window_id": "  "},
        {"epoch_ms": None},
        {"epoch_ms": 333.5},
        {"epoch_ms": True},
        {"window_id": True, "epoch_ms": 333},
        {"window_id": 3.5, "epoch_ms": 333},
    ],
)
def test_resolve_zone_window_key_rejeita_sem_origem_valida(event):
    with pytest.raises(IdentityContractError):
        resolve_zone_window_key(event)


def test_resolve_zone_window_key_nao_muta_evento():
    event = {"window_id": "111", "features_window_id": "222", "epoch_ms": 333}
    snapshot = dict(event)
    resolve_zone_window_key(event)
    assert event == snapshot


# =============================================================================
# IDENTIDADE DETERMINÍSTICA
# =============================================================================
def test_is_deterministic_event_id():
    assert is_deterministic_event_id("a" * 64)
    assert is_deterministic_event_id(hashlib.sha256(b"x").hexdigest())
    assert not is_deterministic_event_id("deadbeef")
    assert not is_deterministic_event_id("A" * 64)
    assert not is_deterministic_event_id(12345)
    assert not is_deterministic_event_id(None)
    assert not is_deterministic_event_id("")


# =============================================================================
# RISCO 1: CANONICALIZAÇÃO DE -0.0 (ZERO NEGATIVO)
# =============================================================================
def test_exhaustion_v2_canonicalizacao_zero_negativo():
    base_id = exhaustion_v2_event_id(WINDOW, SYMBOL, 0.0)

    # Todas as representações numéricas e decimais de zero devem gerar exatamente o mesmo hash
    assert exhaustion_v2_event_id(WINDOW, SYMBOL, -0.0) == base_id
    assert exhaustion_v2_event_id(WINDOW, SYMBOL, Decimal("-0.0")) == base_id
    assert exhaustion_v2_event_id(WINDOW, SYMBOL, Decimal("-0.00000000")) == base_id
    assert exhaustion_v2_event_id(WINDOW, SYMBOL, Decimal("0.0")) == base_id
    assert exhaustion_v2_event_id(WINDOW, SYMBOL, "-0.0") == base_id
    assert exhaustion_v2_event_id(WINDOW, SYMBOL, "0.0") == base_id
    assert exhaustion_v2_event_id(WINDOW, SYMBOL, 0) == base_id


def test_zone_v2_canonicalizacao_zero_negativo_geometria():
    base_id = zone_v2_event_id(
        SYMBOL, str(WINDOW), "POC", "daily", 0.0, 0.0, 0.0, LAST_TOUCHED
    )
    # Geometria com -0.0 e Decimal("-0.00")
    neg_id = zone_v2_event_id(
        SYMBOL,
        str(WINDOW),
        "POC",
        "daily",
        -0.0,
        Decimal("-0.00"),
        "-0.0",
        LAST_TOUCHED,
    )
    assert neg_id == base_id


def test_canonicalize_structure_nested():
    data = {
        "float_zero": -0.0,
        "dec_zero": Decimal("-0.000"),
        "nested": {"list": [-0.0, 1.5, 0.0]},
        "positive": 2.5,
    }
    canon = canonicalize_structure(data)
    assert canon["float_zero"] == 0.0
    assert canon["dec_zero"] == 0.0
    assert canon["nested"]["list"][0] == 0.0
    assert canon["positive"] == 2.5
    # Garante imutabilidade da entrada
    assert data["float_zero"] == -0.0


# =============================================================================
# RISCO 2: VALIDAÇÃO ESTRITA DE LAST_TOUCHED
# =============================================================================
def test_normalize_last_touched_formatos_validos():
    # ISO strings
    assert normalize_last_touched(LAST_TOUCHED) == LAST_TOUCHED
    assert normalize_last_touched("2026-10-24T12:00:00Z") == "2026-10-24T12:00:00Z"

    # Timestamps numéricos >= 0
    assert normalize_last_touched(1700000060000) == "1700000060000"
    assert normalize_last_touched(1700000060000.0) == "1700000060000"
    assert normalize_last_touched(Decimal("1700000060000")) == "1700000060000"
    assert normalize_last_touched("1700000060000") == "1700000060000"
    assert normalize_last_touched(0) == "0"
    assert normalize_last_touched(0.0) == "0"
    assert normalize_last_touched(-0.0) == "0"


@pytest.mark.parametrize(
    "invalid_touched",
    [
        None,
        True,
        False,
        "   ",
        "",
        "nan",
        "NaN",
        float("nan"),
        "inf",
        "-inf",
        float("inf"),
        float("-inf"),
        -1,
        -1000.5,
        "-1700000060000",
        "data-invalida-sem-sentido",
        "2026-99-99T99:99:99",
        [1700000060000],
        {"ts": 1700000060000},
    ],
)
def test_normalize_last_touched_rejeicoes_fail_closed(invalid_touched):
    with pytest.raises(IdentityContractError):
        normalize_last_touched(invalid_touched)


# =============================================================================
# RISCO 3: PRESERVAÇÃO INTEGRAL DE EVENT_ID E PAYLOAD EM FALHAS DE DELTA
# =============================================================================
def test_validate_and_preserve_payload_falha_preserva_event_id_e_payload():
    original_event = {
        "tipo_evento": "Exaustão",
        "ativo": SYMBOL,
        "window_open_ms": WINDOW,
        "delta": 2.9,
        "delta_fechamento": float("nan"),
        "meta": {"nested_id": "preserve_me"},
    }
    snapshot_before = copy.deepcopy(original_event)

    with pytest.raises(IdentityContractError) as exc_info:
        validate_and_preserve_payload(original_event, delta_field="delta_fechamento")

    exc = exc_info.value
    # 1. Payload original NÃO sofreu mutação in-place
    assert original_event == snapshot_before

    # 2. Atributos da exceção preservam event_id e raw_payload íntegro para quarentena
    expected_eid = exhaustion_v2_event_id(WINDOW, SYMBOL, 2.9)
    assert exc.event_id == expected_eid
    assert exc.field_name == "delta_fechamento"
    assert exc.raw_payload is not None
    assert exc.raw_payload["meta"]["nested_id"] == "preserve_me"
    assert exc.raw_payload["ativo"] == SYMBOL


def test_validate_and_preserve_payload_sucesso_normaliza_sem_mutar_original():
    original_event = {
        "tipo_evento": "Exaustão",
        "ativo": SYMBOL,
        "window_open_ms": WINDOW,
        "delta": 2.9,
        "delta_fechamento": "-0.0",
        "event_id": exhaustion_v2_event_id(WINDOW, SYMBOL, 2.9),
    }
    snapshot_before = copy.deepcopy(original_event)

    eid, clean_payload = validate_and_preserve_payload(
        original_event, delta_field="delta_fechamento"
    )

    # Original intacto
    assert original_event == snapshot_before
    # Retorno limpo e delta normalizado
    assert eid == original_event["event_id"]
    assert clean_payload["delta_fechamento"] == 0.0


# =============================================================================
# RISCO 4: ESTABILIDADE DE ZONA APÓS REPLAY
# =============================================================================
def test_compute_zone_identity_replay_invariance_objeto_e_json_loads():
    # Simula objeto de Zona
    class MockLevelZone:
        def __init__(self):
            self.kind = "POC"
            self.timeframe = "daily"
            self.anchor_price = 100.0
            self.low = 95.5
            self.high = 104.5
            self.last_touched = LAST_TOUCHED

        def to_dict(self):
            return {
                "kind": self.kind,
                "timeframe": self.timeframe,
                "anchor_price": self.anchor_price,
                "low": self.low,
                "high": self.high,
                "last_touched": self.last_touched,
                "touch_count": 5,
            }

    zone_obj = MockLevelZone()
    id_from_obj = compute_zone_identity(
        zone_obj, window_key=str(WINDOW), symbol=SYMBOL
    )

    # Simula serialização JSON -> Replay -> json.loads
    json_bytes = json.dumps(zone_obj.to_dict()).encode("utf-8")
    replayed_dict = json.loads(json_bytes.decode("utf-8"))
    id_from_replay = compute_zone_identity(
        replayed_dict, window_key=str(WINDOW), symbol=SYMBOL
    )

    # Identidade rigorosamente invariante
    assert id_from_obj == id_from_replay
    assert id_from_obj == zone_v2_event_id(
        SYMBOL, str(WINDOW), "POC", "daily", 100.0, 95.5, 104.5, LAST_TOUCHED
    )


def test_compute_zone_identity_drift_float_decimal_str():
    # Flutuante
    d_float = {
        "tipo_evento": "Zona",
        "ativo": SYMBOL,
        "window_id": str(WINDOW),
        "kind": "POC",
        "timeframe": "daily",
        "anchor_price": 100.0,
        "low": 95.5,
        "high": 104.5,
        "last_touched": LAST_TOUCHED,
    }
    # Decimal
    d_decimal = {
        "tipo_evento": "Zona",
        "ativo": SYMBOL,
        "window_id": str(WINDOW),
        "kind": "POC",
        "timeframe": "daily",
        "anchor_price": Decimal("100.000"),
        "low": Decimal("95.500"),
        "high": Decimal("104.500"),
        "last_touched": LAST_TOUCHED,
    }
    # String numérica (replay histórico de log/json/db)
    d_str = {
        "tipo_evento": "Zona",
        "ativo": SYMBOL,
        "window_id": str(WINDOW),
        "kind": "POC",
        "timeframe": "daily",
        "anchor_price": "100.00",
        "low": "95.50",
        "high": "104.50",
        "last_touched": LAST_TOUCHED,
    }

    id_float = compute_zone_identity(d_float)
    id_decimal = compute_zone_identity(d_decimal)
    id_str = compute_zone_identity(d_str)

    assert id_float == id_decimal == id_str


# =============================================================================
# RISCO 5: COBERTURA COMPLETA DE TODOS OS PRODUTORES REGISTRADOS
# =============================================================================
@pytest.mark.parametrize(
    "event,expected_version",
    [
        (
            {
                "tipo_evento": "Exaustão",
                "ativo": SYMBOL,
                "window_open_ms": WINDOW,
                "delta": 2.9,
            },
            IDENTITY_VERSION_EXHAUSTION,
        ),
        (
            {
                "tipo_evento": "Zona",
                "ativo": SYMBOL,
                "window_id": str(WINDOW),
                "zone_context": {
                    "kind": "POC",
                    "timeframe": "daily",
                    "anchor_price": 100.0,
                    "low": 95.5,
                    "high": 104.5,
                    "last_touched": LAST_TOUCHED,
                },
            },
            IDENTITY_VERSION_ZONE,
        ),
        (
            {
                "tipo_evento": "Absorção",
                "ativo": SYMBOL,
                "window_open_ms": WINDOW,
            },
            IDENTITY_VERSION_ABSORPTION,
        ),
        (
            {
                "tipo_evento": "ANALYSIS_TRIGGER",
                "symbol": SYMBOL,
                "window_id": str(WINDOW),
            },
            IDENTITY_VERSION_TRIGGER,
        ),
    ],
)
def test_compute_producer_identity_todos_produtores(event, expected_version):
    version, eid = compute_producer_identity(event)
    assert version == expected_version
    assert is_deterministic_event_id(eid)


def test_compute_producer_identity_produtor_desconhecido_rejeita():
    event = {"tipo_evento": "ProdutorDesconhecido", "ativo": SYMBOL}
    with pytest.raises(IdentityContractError) as exc_info:
        compute_producer_identity(event)
    assert "não registrado" in str(exc_info.value)
