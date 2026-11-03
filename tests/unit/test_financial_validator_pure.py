# tests/unit/test_financial_validator_pure.py
"""
Suíte de Testes Herméticos e Puros para o Validador Semântico Financeiro.

Fase 5.11 - Gate H1-B2-B1:
Cobre integralmente os 50 testes especificados (FV-01 a FV-50) sem tocar
em disco operacional, rede, banco SQLite ou variáveis globais.
"""

from __future__ import annotations

import copy
import decimal
import math
import sys
from unittest.mock import patch
import pytest

from common.financial_validator import (
    FinancialValidationResult,
    validate_financial_event_semantics,
)


class TestFinancialValidatorPure:
    """Suíte hermética de 50 testes FV-01 a FV-50."""

    # FV-01: NaN em preco_fechamento de Absorção
    def test_fv01_nan_in_preco_fechamento_absorcao(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": float("nan")}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.is_rejected is True
        assert res.status == "INVALID"
        assert res.category == "FINANCIAL_SIGNAL"
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "NAN_IN_NUMERIC_FIELD"

    # FV-02: Infinity em preco_fechamento de Exaustão
    def test_fv02_infinity_in_preco_fechamento_exaustao(self):
        event = {"tipo_evento": "Exaustão", "preco_fechamento": float("inf")}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.is_rejected is True
        assert res.status == "INVALID"
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "POS_INF_IN_NUMERIC_FIELD"

    # FV-03: Infinity negativo em preço de referência classificado
    def test_fv03_negative_infinity_in_price(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": float("-inf")}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "NEG_INF_IN_NUMERIC_FIELD"

    # FV-04: String "__NaN__" em campo financeiro
    def test_fv04_string_dunder_nan_in_price(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": "__NaN__"}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "DEGENERATE_STRING_TOKEN"

    # FV-05: String "NaN" em campo financeiro
    def test_fv05_string_nan_in_price(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": "NaN"}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "DEGENERATE_STRING_TOKEN"

    # FV-06: Booleano True como preço
    def test_fv06_boolean_true_as_price(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": True}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "BOOLEAN_IN_NUMERIC_FIELD"

    # FV-07: Preço negativo inválido
    def test_fv07_negative_price(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": -50000.0}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "NUMERIC_VALUE_MUST_BE_POSITIVE_NEGATIVE"

    # FV-08: Preço zero inválido
    def test_fv08_zero_price(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": 0.0}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "NUMERIC_VALUE_MUST_BE_POSITIVE_ZERO"

    # FV-09: Preço positivo finito válido
    def test_fv09_valid_positive_finite_price(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": 78989.5}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.is_accepted is True
        assert res.status == "VALID"
        assert res.category == "FINANCIAL_SIGNAL"
        assert res.rejected_field is None

    # FV-10: String decimal válida preservada
    def test_fv10_valid_decimal_string(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": "78989.50"}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-11: String decimal com oito casas preservada
    def test_fv11_eight_decimal_places_string(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": "0.00012345"}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-12: Notação científica válida
    def test_fv12_scientific_notation_valid(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": "1.5e4"}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-13: Decimal finito válido
    def test_fv13_decimal_finite_valid(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": decimal.Decimal("78989.5")}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-14: Decimal NaN inválido
    def test_fv14_decimal_nan_invalid(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": decimal.Decimal("NaN")}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "NAN_IN_NUMERIC_FIELD"

    # FV-15: Decimal Infinity inválido
    def test_fv15_decimal_infinity_invalid(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": decimal.Decimal("Infinity")}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "POS_INF_IN_NUMERIC_FIELD"

    # FV-16: Campo obrigatório ausente
    def test_fv16_mandatory_field_missing(self):
        event = {"tipo_evento": "Absorção", "volume_total": 100.0}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "MANDATORY_FIELD_MISSING"

    # FV-17: Campo obrigatório None
    def test_fv17_mandatory_field_none(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": None}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert res.rejection_reason == "MANDATORY_FIELD_IS_NONE"

    # FV-18: RSI analítico NaN permitido
    def test_fv18_analytical_rsi_nan_permitted(self):
        event = {
            "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 78000.0,
            "rsi": float("nan"),
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"
        assert res.category == "ANALYTICAL_TRIGGER"

    # FV-19: MACD analítico NaN permitido
    def test_fv19_analytical_macd_nan_permitted(self):
        event = {
            "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 78000.0,
            "macd": float("nan"),
            "features": {"macd_hist": float("nan")},
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-20: Delta negativo permitido
    def test_fv20_negative_delta_permitted(self):
        event = {
            "tipo_evento": "Absorção",
            "preco_fechamento": 78000.0,
            "delta": -125.5,
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-21: Delta zero permitido
    def test_fv21_zero_delta_permitted(self):
        event = {
            "tipo_evento": "Absorção",
            "preco_fechamento": 78000.0,
            "delta": 0.0,
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-22: Evento Zona com campos opcionais
    def test_fv22_zone_event_with_optional_fields(self):
        event = {
            "tipo_evento": "Zona",
            "level": 75000.0,
            "volume_total": 500.0,
            "delta": -50.0,
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"
        assert res.category == "ZONE_EVENT"

    # FV-23: Evento Alerta contextual
    def test_fv23_alert_event_contextual(self):
        event = {
            "tipo_evento": "Alerta",
            "price": 79000.0,
            "context": {"price": 79000.0, "volume": 12.0},
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"
        assert res.category == "ALERT_EVENT"

    # FV-24: Evento AI_ANALYSIS válido
    def test_fv24_ai_analysis_valid(self):
        event = {
            "tipo_evento": "AI_ANALYSIS",
            "anchor_price": 78500.0,
            "ai_result": {"direction": "BULLISH"},
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"
        assert res.category == "AI_DIAGNOSTIC"

    # FV-25: Categoria desconhecida sem contrato
    def test_fv25_unknown_category_without_contract(self):
        event = {"tipo_evento": "EVENTO_TOTALMENTE_DESCONHECIDO", "data": "abc"}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.is_unclassified is True
        assert res.status == "UNCLASSIFIED"
        assert res.category == "UNCLASSIFIED"
        assert res.requires_classification is True
        assert res.rejection_reason == "UNKNOWN_EVENT_TYPE_NO_CONTRACT"

    # FV-26: Categoria desconhecida contendo price
    def test_fv26_unknown_category_containing_price(self):
        event = {"tipo_evento": "CUSTOM_ORDER_EXECUTION", "price": 50000.0}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.is_unclassified is True
        assert res.status == "UNCLASSIFIED"
        assert res.requires_classification is True
        assert res.rejection_reason == "UNKNOWN_EVENT_TYPE_WITH_FINANCIAL_FIELDS"

    # FV-27: Evento malformado não-dict
    def test_fv27_malformed_event_not_dict(self):
        res1 = validate_financial_event_semantics(["not", "a", "dict"])
        assert res1.is_valid is False
        assert res1.status == "INVALID"
        assert res1.category == "MALFORMED"
        assert res1.rejection_reason == "EVENT_NOT_A_DICT"

        res2 = validate_financial_event_semantics("string_event")
        assert res2.is_valid is False
        assert res2.status == "INVALID"

    # FV-28: Campo financeiro contendo lista
    def test_fv28_financial_field_containing_list(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": [78000.0]}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert "UNSUPPORTED_CONTAINER_TYPE_LIST" in res.rejection_reason

    # FV-29: Campo financeiro contendo dict
    def test_fv29_financial_field_containing_dict(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": {"price": 78000.0}}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert "UNSUPPORTED_CONTAINER_TYPE_DICT" in res.rejection_reason

    # FV-30: Objeto não suportado em campo financeiro
    def test_fv30_unsupported_object_in_financial_field(self):
        class CustomObj:
            pass

        event = {"tipo_evento": "Absorção", "preco_fechamento": CustomObj()}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejected_field == "preco_fechamento"
        assert "UNSUPPORTED_NUMERIC_TYPE_CUSTOMOBJ" in res.rejection_reason
        assert "<CustomObj>" in res.rejected_value_repr

    # FV-31: Payload não é modificado
    def test_fv31_payload_is_not_modified(self):
        orig_event = {
            "tipo_evento": "Absorção",
            "preco_fechamento": "78989.50",
            "delta": -100.0,
            "nested": {"val": 10},
        }
        event_copy = copy.deepcopy(orig_event)
        validate_financial_event_semantics(orig_event)
        assert orig_event == event_copy

    # FV-32: String decimal não é arredondada
    def test_fv32_decimal_string_not_rounded(self):
        high_precision_str = "78989.1234567890123"
        event = {"tipo_evento": "Absorção", "preco_fechamento": high_precision_str}
        validate_financial_event_semantics(event)
        assert event["preco_fechamento"] == high_precision_str
        assert isinstance(event["preco_fechamento"], str)

    # FV-33: Nenhuma chamada a filesystem
    def test_fv33_no_filesystem_calls(self):
        with patch("builtins.open") as mock_open, patch("os.stat") as mock_stat:
            event = {"tipo_evento": "Absorção", "preco_fechamento": 78000.0}
            validate_financial_event_semantics(event)
            assert mock_open.call_count == 0
            assert mock_stat.call_count == 0

    # FV-34: Nenhuma conexão SQLite
    def test_fv34_no_sqlite_connections(self):
        with patch("sqlite3.connect") as mock_conn:
            event = {"tipo_evento": "Absorção", "preco_fechamento": 78000.0}
            validate_financial_event_semantics(event)
            assert mock_conn.call_count == 0

    # FV-35: Nenhum import operacional circular
    def test_fv35_no_circular_operational_imports(self):
        # Validador não deve conter imports de event_store, event_saver ou fallback_reconciler
        mod = sys.modules["common.financial_validator"]
        assert not hasattr(mod, "EventStore")
        assert not hasattr(mod, "EventSaver")
        assert not hasattr(mod, "FallbackReconciler")

    # FV-36: Nenhuma dependência de relógio ou UUID
    def test_fv36_no_clock_or_uuid_dependencies(self):
        with patch("time.time") as mock_time, patch("uuid.uuid4") as mock_uuid:
            event = {"tipo_evento": "Absorção", "preco_fechamento": 78000.0}
            validate_financial_event_semantics(event)
            assert mock_time.call_count == 0
            assert mock_uuid.call_count == 0

    # FV-37: Estrutura aninhada suportada
    def test_fv37_nested_structure_supported(self):
        event = {
            "tipo_evento": "Absorção",
            "preco_fechamento": 78000.0,
            "raw_event": {"preco_fechamento": 78000.0},
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

        # Com valor aninhado inválido
        event_bad = {
            "tipo_evento": "Absorção",
            "preco_fechamento": 78000.0,
            "raw_event": {"preco_fechamento": float("nan")},
        }
        res_bad = validate_financial_event_semantics(event_bad)
        assert res_bad.is_valid is False
        assert res_bad.rejected_field == "raw_event.preco_fechamento"

    # FV-38: Estrutura autorreferencial
    def test_fv38_self_referential_structure(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": 78000.0}
        event["self"] = event  # ciclo direto
        event["raw_event"] = event  # ciclo em raw_event
        # Não deve travar em RecursionError
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-39: Bool rejeitado antes de int
    def test_fv39_bool_rejected_before_int(self):
        event_false = {"tipo_evento": "Absorção", "preco_fechamento": False}
        res = validate_financial_event_semantics(event_false)
        assert res.is_valid is False
        assert res.rejection_reason == "BOOLEAN_IN_NUMERIC_FIELD"

    # FV-40: Resultado imutável
    def test_fv40_result_immutable(self):
        res = FinancialValidationResult(
            is_valid=True, status="VALID", category="FINANCIAL_SIGNAL"
        )
        with pytest.raises(Exception):
            res.is_valid = False  # frozen dataclass proíbe mutação

    # FV-41: Valores negativos legítimos em campos analíticos
    def test_fv41_negative_values_in_analytical_fields(self):
        event = {
            "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 78000.0,
            "delta": -999.0,
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-42: Valores zero legítimos em campos analíticos
    def test_fv42_zero_values_in_analytical_fields(self):
        event = {
            "tipo_evento": "ANALYSIS_TRIGGER",
            "preco_fechamento": 78000.0,
            "volume_total": 0.0,
            "delta": 0.0,
        }
        res = validate_financial_event_semantics(event)
        assert res.is_valid is True
        assert res.status == "VALID"

    # FV-43: Token "__Infinity__" rejeitado em preço
    def test_fv43_token_dunder_infinity_rejected(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": "__Infinity__"}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejection_reason == "DEGENERATE_STRING_TOKEN"

    # FV-44: Token "__-Infinity__" rejeitado em preço
    def test_fv44_token_dunder_neg_infinity_rejected(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": "__-Infinity__"}
        res = validate_financial_event_semantics(event)
        assert res.is_valid is False
        assert res.rejection_reason == "DEGENERATE_STRING_TOKEN"

    # FV-45: Validador não altera identidade event_id
    def test_fv45_validator_does_not_change_event_id(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": 78000.0, "event_id": "ABC_123"}
        validate_financial_event_semantics(event)
        assert event["event_id"] == "ABC_123"

    # FV-46: Validador não altera campo id legado
    def test_fv46_validator_does_not_change_legacy_id(self):
        event = {"tipo_evento": "Absorção", "preco_fechamento": 78000.0, "id": 9999}
        validate_financial_event_semantics(event)
        assert event["id"] == 9999
        assert isinstance(event["id"], int)

    # FV-47: Validador não converte Decimal em float
    def test_fv47_validator_does_not_convert_decimal_to_float(self):
        d_val = decimal.Decimal("78989.123456789")
        event = {"tipo_evento": "Absorção", "preco_fechamento": d_val}
        validate_financial_event_semantics(event)
        assert isinstance(event["preco_fechamento"], decimal.Decimal)
        assert event["preco_fechamento"] == d_val

    # FV-48: Validador não transforma NaN em None
    def test_fv48_validator_does_not_transform_nan_to_none(self):
        nan_val = float("nan")
        event = {"tipo_evento": "Absorção", "preco_fechamento": nan_val}
        validate_financial_event_semantics(event)
        assert math.isnan(event["preco_fechamento"])  # Continua NaN no evento original

    # FV-49: Resultado UNCLASSIFIED distinto de VALID
    def test_fv49_unclassified_distinct_from_valid(self):
        unclass_event = {"tipo_evento": "CUSTOM_MYSTERY_SIGNAL"}
        res = validate_financial_event_semantics(unclass_event)
        assert res.status == "UNCLASSIFIED"
        assert res.is_valid is False
        assert res.is_accepted is False
        assert res.is_unclassified is True
        assert res.requires_classification is True

    # FV-50: Nenhum efeito colateral observável
    def test_fv50_no_observable_side_effects(self):
        event1 = {"tipo_evento": "Absorção", "preco_fechamento": 78000.0}
        event2 = {"tipo_evento": "Absorção", "preco_fechamento": float("nan")}
        res1 = validate_financial_event_semantics(event1)
        res2 = validate_financial_event_semantics(event2)
        res3 = validate_financial_event_semantics(event1)
        # Determinismo e repetibilidade absoluta
        assert res1 == res3
        assert res1.is_valid is True
        assert res2.is_valid is False

    # FV-51: Instanciação direta com status="VALID" e is_valid=False deve levantar ValueError
    def test_fv51_valid_with_is_valid_false_raises_value_error(self):
        with pytest.raises(ValueError, match="status 'VALID' exige is_valid=True"):
            FinancialValidationResult(
                is_valid=False,
                status="VALID",
                category="FINANCIAL_SIGNAL",
            )

    # FV-52: Instanciação direta com status="INVALID" e is_valid=True deve levantar ValueError
    def test_fv52_invalid_with_is_valid_true_raises_value_error(self):
        with pytest.raises(ValueError, match="status 'INVALID' exige is_valid=False"):
            FinancialValidationResult(
                is_valid=True,
                status="INVALID",
                category="FINANCIAL_SIGNAL",
                rejection_reason="ERR",
            )

    # FV-53: Instanciação direta com status="UNCLASSIFIED" e requires_classification=False deve levantar ValueError
    def test_fv53_unclassified_without_requires_classification_raises_value_error(self):
        with pytest.raises(ValueError, match="status 'UNCLASSIFIED' exige requires_classification=True"):
            FinancialValidationResult(
                is_valid=False,
                status="UNCLASSIFIED",
                category="UNCLASSIFIED",
                requires_classification=False,
                rejection_reason="UNKNOWN",
            )

    # FV-54: Instanciação direta com status="VALID" e rejection_reason preenchido deve levantar ValueError
    def test_fv54_valid_with_rejection_reason_raises_value_error(self):
        with pytest.raises(ValueError, match="status 'VALID' exige rejection_reason=None"):
            FinancialValidationResult(
                is_valid=True,
                status="VALID",
                category="FINANCIAL_SIGNAL",
                rejection_reason="SOME_REASON",
            )

    # FV-55: Instanciação direta com status="INVALID" e rejection_reason None ou vazio deve levantar ValueError
    def test_fv55_invalid_without_rejection_reason_raises_value_error(self):
        with pytest.raises(ValueError, match="status 'INVALID' exige rejection_reason como string não vazia"):
            FinancialValidationResult(
                is_valid=False,
                status="INVALID",
                category="FINANCIAL_SIGNAL",
                rejection_reason=None,
            )
        with pytest.raises(ValueError, match="status 'INVALID' exige rejection_reason como string não vazia"):
            FinancialValidationResult(
                is_valid=False,
                status="INVALID",
                category="FINANCIAL_SIGNAL",
                rejection_reason="   ",
            )

    # FV-56: Instanciação direta com status desconhecido deve levantar ValueError
    def test_fv56_unknown_status_raises_value_error(self):
        with pytest.raises(ValueError, match="status desconhecido 'UNKNOWN_STATUS'"):
            FinancialValidationResult(
                is_valid=True,
                status="UNKNOWN_STATUS",
                category="CUSTOM",
            )

    # FV-57: Instanciação legítima com status="VALID" deve funcionar normalmente
    def test_fv57_legitimate_valid_instance(self):
        res = FinancialValidationResult(
            is_valid=True,
            status="VALID",
            category="FINANCIAL_SIGNAL",
            requires_classification=False,
            rejected_field=None,
            rejected_value_repr=None,
            rejection_reason=None,
        )
        assert res.is_valid is True
        assert res.status == "VALID"
        assert res.is_accepted is True
        assert res.is_rejected is False
        assert res.is_unclassified is False

    # FV-58: Instanciação legítima com status="INVALID" deve funcionar normalmente
    def test_fv58_legitimate_invalid_instance(self):
        res = FinancialValidationResult(
            is_valid=False,
            status="INVALID",
            category="FINANCIAL_SIGNAL",
            rejected_field="preco_fechamento",
            rejected_value_repr="nan",
            rejection_reason="NAN_IN_NUMERIC_FIELD",
            requires_classification=False,
        )
        assert res.is_valid is False
        assert res.status == "INVALID"
        assert res.is_rejected is True
        assert res.is_accepted is False
        assert res.is_unclassified is False
        assert res.rejection_reason == "NAN_IN_NUMERIC_FIELD"

    # FV-59: Instanciação legítima com status="UNCLASSIFIED" deve funcionar normalmente
    def test_fv59_legitimate_unclassified_instance(self):
        res = FinancialValidationResult(
            is_valid=False,
            status="UNCLASSIFIED",
            category="UNCLASSIFIED",
            requires_classification=True,
            rejected_field="tipo_evento",
            rejected_value_repr="UNKNOWN",
            rejection_reason="UNKNOWN_EVENT_TYPE_NO_CONTRACT",
        )
        assert res.is_valid is False
        assert res.status == "UNCLASSIFIED"
        assert res.is_unclassified is True
        assert res.requires_classification is True
        assert res.is_accepted is False
        assert res.is_rejected is False

    # FV-60: Regressão geral de invariantes da validação funcional
    def test_fv60_regression_full_suite_consistency(self):
        # Valida que todos os métodos retornam instâncias 100% aderentes às invariantes
        sample_valid = {"tipo_evento": "Absorção", "preco_fechamento": 50000.0}
        sample_invalid = {"tipo_evento": "Absorção", "preco_fechamento": -100.0}
        sample_unclass = {"tipo_evento": "ORDEM_LEGADA_DESCONHECIDA"}

        res_v = validate_financial_event_semantics(sample_valid)
        res_i = validate_financial_event_semantics(sample_invalid)
        res_u = validate_financial_event_semantics(sample_unclass)

        assert res_v.is_accepted is True
        assert res_i.is_rejected is True
        assert res_u.is_unclassified is True
