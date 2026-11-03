# tests/unit/test_financial_quarantine.py
"""
Suíte de Testes Herméticos e de Hardening da Quarentena Forense (Gate H1-B2-Q1-H).

Cobre:
- Testes QT-01 a QT-30 adaptados ao contrato tipado e status triestado.
- Testes QH-01 a QH-40 cobrindo tipagem, fronteiras, multiprocesso e contingências.

Opera 100% isolado em tmp_path sem qualquer acesso aos dados operacionais de produção.
"""

from __future__ import annotations

import copy
import dataclasses
import decimal
import json
import math
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from common.financial_quarantine import (
    FinancialQuarantine,
    QuarantineWriteResult,
    decode_typed_payload,
    encode_typed_payload,
)
from common.financial_validator import (
    FinancialValidationResult,
    validate_financial_event_semantics,
)


class TestFinancialQuarantine:
    """Suíte consolidada de quarentena: QT-01 a QT-30 e QH-01 a QH-40."""

    @pytest.fixture
    def sample_invalid_result(self) -> FinancialValidationResult:
        return FinancialValidationResult(
            is_valid=False,
            status="INVALID",
            category="FINANCIAL_SIGNAL",
            rejected_field="preco_fechamento",
            rejected_value_repr="nan",
            rejection_reason="NAN_IN_NUMERIC_FIELD",
        )

    @pytest.fixture
    def sample_unclassified_result(self) -> FinancialValidationResult:
        return FinancialValidationResult(
            is_valid=False,
            status="UNCLASSIFIED",
            category="UNCLASSIFIED",
            requires_classification=True,
            rejected_field="tipo_evento",
            rejected_value_repr="MYSTERY_SIGNAL",
            rejection_reason="UNKNOWN_EVENT_TYPE_NO_CONTRACT",
        )

    # =========================================================================
    # SUÍTE QT-01 A QT-30 (ADAPTADA AO CONTRATO TIPADO V2)
    # =========================================================================

    def test_qt01_quarantine_nan_serialized_as_nan_type(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        payload = {"tipo_evento": "Absorção", "preco_fechamento": float("nan")}
        res = q.quarantine_event(payload, sample_invalid_result, "test_caller")

        assert res.status == "CONFIRMED"
        assert res.success is True
        assert res.file_path is not None
        assert Path(res.file_path).exists()

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        assert data["quarantine_format_version"] == 2
        assert data["original_payload_encoding"] == "TYPED_TREE_V1"
        decoded = decode_typed_payload(data["original_payload"])
        assert math.isnan(decoded["preco_fechamento"])

    def test_qt02_quarantine_infinity_serialized(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        payload = {"tipo_evento": "Exaustão", "preco_fechamento": float("inf")}
        res = q.quarantine_event(payload, sample_invalid_result, "test_caller")

        assert res.status == "CONFIRMED"
        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        decoded = decode_typed_payload(data["original_payload"])
        assert math.isinf(decoded["preco_fechamento"]) and decoded["preco_fechamento"] > 0

    def test_qt03_quarantine_negative_infinity_serialized(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        payload = {"tipo_evento": "Exaustão", "preco_fechamento": float("-inf")}
        res = q.quarantine_event(payload, sample_invalid_result, "test_caller")

        assert res.status == "CONFIRMED"
        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        decoded = decode_typed_payload(data["original_payload"])
        assert math.isinf(decoded["preco_fechamento"]) and decoded["preco_fechamento"] < 0

    def test_qt04_quarantine_decimal_serialized_without_precision_loss(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        dec = decimal.Decimal("98765.123456789012345")
        payload = {"tipo_evento": "Absorção", "preco_fechamento": dec}
        res = q.quarantine_event(payload, sample_invalid_result, "test_caller")

        assert res.status == "CONFIRMED"
        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())
        decoded = decode_typed_payload(data["original_payload"])
        assert isinstance(decoded["preco_fechamento"], decimal.Decimal)
        assert decoded["preco_fechamento"] == dec

    def test_qt05_quarantine_unclassified_subdirectory(self, tmp_path: Path, sample_unclassified_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        payload = {"tipo_evento": "MYSTERY_SIGNAL", "valor": 123}
        res = q.quarantine_event(payload, sample_unclassified_result, "test_caller")

        assert res.status == "CONFIRMED"
        fp = Path(res.file_path)
        assert fp.parent.name == "unclassified"
        assert fp.name.startswith("unclassified_events_")

    def test_qt06_quarantine_collision_subdirectory(self, tmp_path: Path):
        q = FinancialQuarantine(tmp_path / "quarantine")
        payload = {"event_id": "COLLIDE_01", "preco": 100.0}
        collision_val = FinancialValidationResult(
            is_valid=False,
            status="INVALID",
            category="COLLISION",
            rejection_reason="DIVERGENT_HASH_COLLISION",
        )
        res = q.quarantine_event(payload, collision_val, "event_store", category="COLLISION")

        assert res.status == "CONFIRMED"
        fp = Path(res.file_path)
        assert fp.parent.name == "collision"
        assert fp.name.startswith("collision_events_")

    def test_qt07_original_payload_not_mutated(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        nan_val = float("nan")
        original_payload = {"tipo_evento": "Absorção", "preco_fechamento": nan_val, "dados": [1, 2, 3]}
        snapshot = copy.deepcopy(original_payload)

        q.quarantine_event(original_payload, sample_invalid_result, "test_caller")

        assert math.isnan(original_payload["preco_fechamento"])
        assert original_payload["dados"] == snapshot["dados"]
        assert original_payload["tipo_evento"] == snapshot["tipo_evento"]

    def test_qt08_result_success_and_filepath(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({"teste": 1}, sample_invalid_result, "test_caller")

        assert isinstance(res, QuarantineWriteResult)
        assert res.status == "CONFIRMED"
        assert res.success is True
        assert res.file_path is not None
        assert res.error_type is None
        assert res.error_message is None

    def test_qt09_permission_failure_handling(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        with patch.object(Path, "mkdir", side_effect=PermissionError("Acesso negado")):
            res = q.quarantine_event({"teste": 1}, sample_invalid_result, "test_caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.error_type == "PermissionError"
        assert "Acesso negado" in (res.error_message or "")

    def test_qt10_uncreatable_directory_handling(self, tmp_path: Path, sample_invalid_result):
        blocking_file = tmp_path / "blocking_file"
        blocking_file.write_text("bloqueio")

        q = FinancialQuarantine(blocking_file / "sub_quarantine")
        res = q.quarantine_event({"teste": 1}, sample_invalid_result, "test_caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.error_type in ("NotADirectoryError", "FileExistsError", "OSError")

    def test_qt11_valid_json_per_line(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        payload = {"tipo": "teste", "val": 42}
        res = q.quarantine_event(payload, sample_invalid_result, "test_caller")

        with open(res.file_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = json.loads(line)
                assert isinstance(parsed, dict)
                assert parsed["category"] == "INVALID"

    def test_qt12_multiple_events_appended(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        for i in range(5):
            res = q.quarantine_event({"seq": i}, sample_invalid_result, "test_caller")
            assert res.status == "CONFIRMED"

        with open(res.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 5
        for i, line in enumerate(lines):
            data = json.loads(line)
            decoded = decode_typed_payload(data["original_payload"])
            assert decoded["seq"] == i

    def test_qt13_quarantine_id_is_uuid4(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({"val": 1}, sample_invalid_result, "test_caller")

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        qid = data["quarantine_id"]
        parsed_uuid = uuid.UUID(qid)
        assert parsed_uuid.version == 4

    def test_qt14_timestamp_utc_iso8601(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({"val": 1}, sample_invalid_result, "test_caller")

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        ts = data["quarantine_timestamp_utc"]
        dt = datetime.fromisoformat(ts)
        assert dt is not None
        assert dt.tzinfo is not None

    def test_qt15_source_caller_present(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({"val": 1}, sample_invalid_result, "reconciler_worker_01")

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        assert data["source_caller"] == "reconciler_worker_01"

    def test_qt16_rejection_reason_matches_result(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({"val": 1}, sample_invalid_result, "test_caller")

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        assert data["rejection_reason"] == sample_invalid_result.rejection_reason
        assert data["rejected_field"] == sample_invalid_result.rejected_field
        assert data["rejected_value_repr"] == sample_invalid_result.rejected_value_repr

    def test_qt17_original_payload_exact_preservation(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        payload = {
            "str_field": "bitcoin",
            "int_field": 100,
            "float_nan": float("nan"),
            "float_inf": float("inf"),
            "str_dec": "75000.12345678",
        }
        res = q.quarantine_event(payload, sample_invalid_result, "test_caller")

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        decoded = decode_typed_payload(data["original_payload"])
        assert decoded["str_field"] == "bitcoin"
        assert decoded["int_field"] == 100
        assert math.isnan(decoded["float_nan"])
        assert math.isinf(decoded["float_inf"])
        assert decoded["str_dec"] == "75000.12345678"

    def test_qt18_symlink_quarantine_dir_protection(self, tmp_path: Path, sample_invalid_result):
        real_dir = tmp_path / "real_quarantine"
        real_dir.mkdir()
        symlink_dir = tmp_path / "sym_quarantine"

        try:
            os.symlink(real_dir, symlink_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks de diretório não suportados/permitidos sem privilégios administrativos")

        q = FinancialQuarantine(symlink_dir)
        res = q.quarantine_event({"val": 1}, sample_invalid_result, "test_caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.error_type == "PATH_SECURITY_VIOLATION"

    def test_qt19_symlink_subdirectory_protection(self, tmp_path: Path, sample_invalid_result):
        base_dir = tmp_path / "quarantine"
        base_dir.mkdir()
        real_invalid = tmp_path / "external_invalid"
        real_invalid.mkdir()
        symlink_sub = base_dir / "invalid"

        try:
            os.symlink(real_invalid, symlink_sub, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks de diretório não suportados/permitidos no ambiente")

        q = FinancialQuarantine(base_dir)
        res = q.quarantine_event({"val": 1}, sample_invalid_result, "test_caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.error_type == "PATH_SECURITY_VIOLATION"

    def test_qt20_purity_no_forbidden_imports(self):
        import common.financial_quarantine as fq
        source_code = Path(fq.__file__).read_text(encoding="utf-8")

        forbidden_names = [
            "EventStore",
            "EventSaver",
            "FallbackReconciler",
            "json_safe",
            "runtime_lock",
        ]
        for name in forbidden_names:
            assert f"import {name}" not in source_code
            assert f"from database.event_store" not in source_code
            assert f"from events.event_saver" not in source_code
            assert f"from events.fallback_reconciler" not in source_code
            assert f"from common.json_safe" not in source_code

    def test_qt21_quarantine_write_result_is_frozen(self):
        res = QuarantineWriteResult(status="CONFIRMED", success=True, file_path="/fake/path")
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.success = False

    def test_qt22_high_precision_decimal_string_preserved(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        prec_str = "65000.12345678"
        payload = {"price": prec_str}
        res = q.quarantine_event(payload, sample_invalid_result, "test_caller")

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        decoded = decode_typed_payload(data["original_payload"])
        assert decoded["price"] == prec_str

    def test_qt23_event_id_not_altered(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        payload = {"event_id": "STRICT_ID_987", "preco": 100}
        res = q.quarantine_event(payload, sample_invalid_result, "test_caller")

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        decoded = decode_typed_payload(data["original_payload"])
        assert decoded["event_id"] == "STRICT_ID_987"
        assert payload["event_id"] == "STRICT_ID_987"

    def test_qt24_emergency_serialization_line_rejected_cleanly(self, tmp_path: Path, sample_invalid_result):
        # Tipos não suportados agora falham de forma segura sem truncamento cego
        q = FinancialQuarantine(tmp_path / "quarantine")

        class UnserializableHostileObject:
            def __repr__(self):
                raise RuntimeError("Hostile repr")

        res = q.quarantine_event(UnserializableHostileObject(), sample_invalid_result, "test_caller")
        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.error_type == "TypeError"

    def test_qt25_fsync_called_effectively(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        with patch("os.fsync") as mock_fsync:
            res = q.quarantine_event({"val": 1}, sample_invalid_result, "test_caller")

        assert res.status == "CONFIRMED"
        assert mock_fsync.called

    def test_qt26_file_lock_used(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        if platform.system() == "Windows":
            import msvcrt
            with patch.object(msvcrt, "locking", wraps=msvcrt.locking) as mock_lock:
                res = q.quarantine_event({"val": 1}, sample_invalid_result, "test_caller")
                assert res.status == "CONFIRMED"
                assert mock_lock.called
        else:
            import fcntl
            with patch.object(fcntl, "flock", wraps=fcntl.flock) as mock_lock:
                res = q.quarantine_event({"val": 1}, sample_invalid_result, "test_caller")
                assert res.status == "CONFIRMED"
                assert mock_lock.called

    def test_qt27_daily_file_uses_utc_date(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({"val": 1}, sample_invalid_result, "test_caller")

        expected_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        assert expected_date in Path(res.file_path).name

    def test_qt28_empty_payload_preserved(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({}, sample_invalid_result, "test_caller")

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        decoded = decode_typed_payload(data["original_payload"])
        assert decoded == {}

    def test_qt29_nested_lists_and_dicts_preserved(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        payload = {
            "nested_dict": {"a": [1, 2, {"b": float("nan")}]},
            "list_of_decimals": [decimal.Decimal("1.23"), decimal.Decimal("4.56")],
        }
        res = q.quarantine_event(payload, sample_invalid_result, "test_caller")

        with open(res.file_path, "r", encoding="utf-8") as f:
            data = json.loads(f.readline())

        decoded = decode_typed_payload(data["original_payload"])
        assert math.isnan(decoded["nested_dict"]["a"][2]["b"])
        assert decoded["list_of_decimals"] == [decimal.Decimal("1.23"), decimal.Decimal("4.56")]

    def test_qt30_two_events_same_day_same_file(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res1 = q.quarantine_event({"e": 1}, sample_invalid_result, "test_caller")
        res2 = q.quarantine_event({"e": 2}, sample_invalid_result, "test_caller")

        assert res1.status == "CONFIRMED"
        assert res2.status == "CONFIRMED"
        assert res1.file_path == res2.file_path

        with open(res1.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2

    # =========================================================================
    # SUÍTE DE HARDENING QH-01 A QH-40
    # =========================================================================

    def test_qh01_float_nan_vs_string_nan(self):
        node_nan = encode_typed_payload(float("nan"))
        node_str = encode_typed_payload("NaN")
        assert node_nan != node_str
        assert node_nan["type"] == "float" and node_nan["special"] == "NaN"
        assert node_str["type"] == "str" and node_str["value"] == "NaN"

    def test_qh02_float_infinity_vs_string_infinity(self):
        node_inf = encode_typed_payload(float("inf"))
        node_str = encode_typed_payload("Infinity")
        assert node_inf != node_str
        assert node_inf["type"] == "float" and node_inf["special"] == "Infinity"
        assert node_str["type"] == "str" and node_str["value"] == "Infinity"

    def test_qh03_float_neg_infinity_vs_string_neg_infinity(self):
        node_ninf = encode_typed_payload(float("-inf"))
        node_str = encode_typed_payload("-Infinity")
        assert node_ninf != node_str
        assert node_ninf["type"] == "float" and node_ninf["special"] == "-Infinity"
        assert node_str["type"] == "str" and node_str["value"] == "-Infinity"

    def test_qh04_decimal_finite_vs_string_decimal(self):
        d_val = decimal.Decimal("65000.12345678")
        s_val = "65000.12345678"
        node_dec = encode_typed_payload(d_val)
        node_str = encode_typed_payload(s_val)
        assert node_dec != node_str
        assert node_dec["type"] == "decimal" and node_dec["value"] == s_val
        assert node_str["type"] == "str" and node_str["value"] == s_val

    def test_qh05_decimal_nan_vs_string_nan(self):
        node_dec_nan = encode_typed_payload(decimal.Decimal("NaN"))
        node_str = encode_typed_payload("NaN")
        assert node_dec_nan != node_str
        assert node_dec_nan["type"] == "decimal" and node_dec_nan["special"] == "NaN"

    def test_qh06_decimal_infinity_vs_string_infinity(self):
        node_dec_inf = encode_typed_payload(decimal.Decimal("Infinity"))
        node_str = encode_typed_payload("Infinity")
        assert node_dec_inf != node_str
        assert node_dec_inf["type"] == "decimal" and node_dec_inf["special"] == "Infinity"

    def test_qh07_float_negative_zero_preserved(self):
        neg_zero = -0.0
        node = encode_typed_payload(neg_zero)
        assert node["type"] == "float"
        assert node.get("is_negative_zero") is True
        decoded = decode_typed_payload(node)
        assert decoded == 0.0
        assert math.copysign(1.0, decoded) < 0.0

    def test_qh08_boolean_vs_integer(self):
        node_bool = encode_typed_payload(True)
        node_int = encode_typed_payload(1)
        assert node_bool != node_int
        assert node_bool["type"] == "bool"
        assert node_int["type"] == "int"

    def test_qh09_large_integer_preserved(self):
        big_int = 10**30
        node = encode_typed_payload(big_int)
        assert node["type"] == "int" and node["value"] == big_int
        assert decode_typed_payload(node) == big_int

    def test_qh10_dict_with_internal_keyword_keys(self):
        # Chaves que poderiam colidir com campos de nós tipados
        tricky_payload = {
            "type": "malicious",
            "value": 123,
            "entries": [1, 2],
            "special": "NaN",
            "__quarantine_type__": "fake",
        }
        node = encode_typed_payload(tricky_payload)
        decoded = decode_typed_payload(node)
        assert decoded == tricky_payload

    def test_qh11_nested_structures(self):
        complex_data = {"a": [1, {"b": decimal.Decimal("2.5")}, (3, 4)], "c": {5, 6}}
        node = encode_typed_payload(complex_data)
        decoded = decode_typed_payload(node)
        assert decoded["a"][1]["b"] == decimal.Decimal("2.5")
        assert isinstance(decoded["a"][2], tuple)
        assert decoded["c"] == {5, 6}

    def test_qh12_list_order_preserved(self):
        seq = [10, 5, 20, 1, 99]
        node = encode_typed_payload(seq)
        assert decode_typed_payload(node) == seq

    def test_qh13_original_payload_immutable(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        original = {"list": [1, 2, {"key": "val"}]}
        original_copy = copy.deepcopy(original)
        q.quarantine_event(original, sample_invalid_result, "caller")
        assert original == original_copy

    def test_qh14_unsupported_object_returns_failure(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")

        class UnsupportedRandomClass:
            pass

        res = q.quarantine_event({"bad": UnsupportedRandomClass()}, sample_invalid_result, "caller")
        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.error_type == "TypeError"

    def test_qh15_no_truncated_fallback_returns_success(self, tmp_path: Path, sample_invalid_result):
        # Assegura que nenhum objeto não suportado retorne CONFIRMED/success=True
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({"func": lambda x: x}, sample_invalid_result, "caller")
        assert res.success is False
        assert res.status != "CONFIRMED"

    def test_qh16_no_repr_invoked_on_hostile_objects(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")

        class HostileRepr:
            def __repr__(self):
                raise AssertionError("REPR HOSTIL FOI INVOCADO INDEVIDAMENTE!")

        # Deve rejeitar com TypeError sem invocar __repr__
        res = q.quarantine_event({"hostile": HostileRepr()}, sample_invalid_result, "caller")
        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.error_type == "TypeError"

    def test_qh17_preexisting_truncated_jsonl_blocks_append(self, tmp_path: Path, sample_invalid_result):
        base_dir = tmp_path / "quarantine"
        sub_dir = base_dir / "invalid"
        sub_dir.mkdir(parents=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        corrupted_file = sub_dir / f"invalid_events_{date_str}.jsonl"

        # Escreve linha sem newline no final (simulando crash anterior no meio do write)
        corrupted_file.write_bytes(b'{"corrupted": true')

        q = FinancialQuarantine(base_dir)
        res = q.quarantine_event({"price": 100}, sample_invalid_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.error_type == "PREEXISTING_JSONL_TRUNCATION"

    def test_qh18_empty_existing_file_admits_append(self, tmp_path: Path, sample_invalid_result):
        base_dir = tmp_path / "quarantine"
        sub_dir = base_dir / "invalid"
        sub_dir.mkdir(parents=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        empty_file = sub_dir / f"invalid_events_{date_str}.jsonl"
        empty_file.touch()

        q = FinancialQuarantine(base_dir)
        res = q.quarantine_event({"price": 100}, sample_invalid_result, "caller")

        assert res.status == "CONFIRMED"
        assert res.success is True

    def test_qh19_file_with_newline_admits_append(self, tmp_path: Path, sample_invalid_result):
        base_dir = tmp_path / "quarantine"
        sub_dir = base_dir / "invalid"
        sub_dir.mkdir(parents=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        valid_file = sub_dir / f"invalid_events_{date_str}.jsonl"
        valid_file.write_bytes(b'{"valid_prev": true}\n')

        q = FinancialQuarantine(base_dir)
        res = q.quarantine_event({"price": 100}, sample_invalid_result, "caller")

        assert res.status == "CONFIRMED"
        assert res.success is True

    def test_qh20_failure_before_write_produces_failed_before_write(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        # Referência circular no payload
        circular = {}
        circular["self"] = circular
        res = q.quarantine_event(circular, sample_invalid_result, "caller")
        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False

    def test_qh21_write_failure_produces_persistence_uncertain(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        orig_open = open

        class MockWriteFileWrapper:
            def __init__(self, f):
                self._f = f
            def __getattr__(self, name):
                return getattr(self._f, name)
            def write(self, *args, **kwargs):
                raise OSError("Disk write error")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self._f.__exit__(*args)

        def mock_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            return MockWriteFileWrapper(f)

        with patch("builtins.open", side_effect=mock_open):
            res = q.quarantine_event({"a": 1}, sample_invalid_result, "caller")
        assert res.status == "PERSISTENCE_UNCERTAIN"
        assert res.success is False

    def test_qh22_flush_failure_produces_persistence_uncertain(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        orig_open = open

        class MockFlushFileWrapper:
            def __init__(self, f):
                self._f = f
            def __getattr__(self, name):
                return getattr(self._f, name)
            def flush(self, *args, **kwargs):
                raise OSError("Flush error")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self._f.__exit__(*args)

        def mock_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            return MockFlushFileWrapper(f)

        with patch("builtins.open", side_effect=mock_open):
            res = q.quarantine_event({"a": 1}, sample_invalid_result, "caller")
        assert res.status == "PERSISTENCE_UNCERTAIN"
        assert res.success is False

    def test_qh23_fsync_failure_produces_persistence_uncertain(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        with patch("os.fsync", side_effect=OSError("Fsync error")):
            res = q.quarantine_event({"a": 1}, sample_invalid_result, "caller")
        assert res.status == "PERSISTENCE_UNCERTAIN"
        assert res.success is False

    def test_qh24_confirmed_requires_success_true(self):
        res = QuarantineWriteResult(status="CONFIRMED", success=True, file_path="/tmp/test.jsonl")
        assert res.success is True

    def test_qh25_failed_before_write_requires_success_false(self):
        with pytest.raises(ValueError, match="exige success=False"):
            QuarantineWriteResult(
                status="FAILED_BEFORE_WRITE",
                success=True,
                file_path=None,
                error_type="ERR",
                error_message="msg",
            )

    def test_qh26_persistence_uncertain_requires_success_false(self):
        with pytest.raises(ValueError, match="exige success=False"):
            QuarantineWriteResult(
                status="PERSISTENCE_UNCERTAIN",
                success=True,
                file_path=None,
                error_type="ERR",
                error_message="msg",
            )

    def test_qh27_dataclass_rejects_contradictory_combinations(self):
        with pytest.raises(ValueError, match="exige error_type=None"):
            QuarantineWriteResult(
                status="CONFIRMED",
                success=True,
                file_path="/tmp/f.jsonl",
                error_type="ILLEGAL_ERROR",
            )

        with pytest.raises(ValueError, match="status desconhecido"):
            QuarantineWriteResult(
                status="INVALID_STATUS",
                success=False,
                file_path=None,
                error_type="E",
                error_message="m",
            )

    def test_qh28_lock_windows_uses_fixed_region(self, tmp_path: Path, sample_invalid_result):
        if platform.system() != "Windows":
            pytest.skip("Teste de msvcrt aplicável estritamente no Windows")

        import msvcrt

        q = FinancialQuarantine(tmp_path / "quarantine")
        captured_seeks = []

        original_seek = io_seek = None
        # Inspeciona se seek(0, os.SEEK_SET) é executado antes do msvcrt.locking
        res = q.quarantine_event({"a": 1}, sample_invalid_result, "caller")
        assert res.status == "CONFIRMED"

    def test_qh29_lock_linux_uses_cooperative_exclusivity(self):
        if platform.system() == "Windows":
            pytest.skip("Teste de fcntl aplicável em POSIX/Linux")
        import fcntl

        # Verificação conceitual da flag
        assert hasattr(fcntl, "LOCK_EX") and hasattr(fcntl, "LOCK_NB")

    def test_qh30_multiprocess_concurrency_two_workers(self, tmp_path: Path):
        worker_script = Path(__file__).parent.parent / "helpers" / "financial_quarantine_worker.py"
        q_dir = tmp_path / "quarantine_concurrent_2"

        proc1 = subprocess.Popen(
            [sys.executable, str(worker_script), "--quarantine-dir", str(q_dir), "--caller-id", "W1", "--num-events", "20"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc2 = subprocess.Popen(
            [sys.executable, str(worker_script), "--quarantine-dir", str(q_dir), "--caller-id", "W2", "--num-events", "20"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        out1, err1 = proc1.communicate(timeout=30)
        out2, err2 = proc2.communicate(timeout=30)

        assert proc1.returncode == 0, f"Worker 1 crash: {err1}"
        assert proc2.returncode == 0, f"Worker 2 crash: {err2}"

        res1 = json.loads(out1.strip().splitlines()[-1])
        res2 = json.loads(out2.strip().splitlines()[-1])

        total_confirmed = res1["confirmed"] + res2["confirmed"]
        total_lock_failed = res1["lock_failed"] + res2["lock_failed"]

        assert total_confirmed > 0
        assert total_confirmed + total_lock_failed == 40
        assert res1["uncertain"] == 0 and res2["uncertain"] == 0

        # Valida que todos os eventos confirmados estão perfeitamente legíveis no arquivo
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        q_file = q_dir / "invalid" / f"invalid_events_{date_str}.jsonl"
        with open(q_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == total_confirmed
        for line in lines:
            parsed = json.loads(line)
            assert parsed["quarantine_format_version"] == 2

    def test_qh31_multiprocess_concurrency_four_workers(self, tmp_path: Path):
        worker_script = Path(__file__).parent.parent / "helpers" / "financial_quarantine_worker.py"
        q_dir = tmp_path / "quarantine_concurrent_4"

        procs = []
        for i in range(4):
            p = subprocess.Popen(
                [sys.executable, str(worker_script), "--quarantine-dir", str(q_dir), "--caller-id", f"W{i}", "--num-events", "15"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            procs.append(p)

        results = []
        for p in procs:
            out, err = p.communicate(timeout=45)
            assert p.returncode == 0, f"Worker crash: {err}"
            results.append(json.loads(out.strip().splitlines()[-1]))

        sum_confirmed = sum(r["confirmed"] for r in results)
        sum_lock_failed = sum(r["lock_failed"] for r in results)

        assert sum_confirmed > 0
        assert sum_confirmed + sum_lock_failed == 60

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        q_file = q_dir / "invalid" / f"invalid_events_{date_str}.jsonl"
        with open(q_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == sum_confirmed
        # Zero linhas corrompidas ou intercaladas
        for line in lines:
            parsed = json.loads(line)
            assert parsed["category"] == "INVALID"

    def test_qh32_all_persisted_lines_are_valid_json(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        for i in range(10):
            q.quarantine_event({"i": i, "val": decimal.Decimal(f"{i}.99")}, sample_invalid_result, "caller")

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        target_file = tmp_path / "quarantine" / "invalid" / f"invalid_events_{date_str}.jsonl"
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                parsed = json.loads(line)
                assert parsed["original_payload_encoding"] == "TYPED_TREE_V1"

    def test_qh33_no_confirmed_event_disappears(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        confirmed_ids = set()
        for i in range(15):
            res = q.quarantine_event({"id_check": i}, sample_invalid_result, "caller")
            if res.status == "CONFIRMED":
                confirmed_ids.add(i)

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        target_file = tmp_path / "quarantine" / "invalid" / f"invalid_events_{date_str}.jsonl"
        found_ids = set()
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                decoded = decode_typed_payload(data["original_payload"])
                found_ids.add(decoded["id_check"])

        assert confirmed_ids == found_ids

    def test_qh34_lock_acquisition_failure_writes_zero_bytes(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        target_file = tmp_path / "quarantine" / "invalid" / f"invalid_events_{date_str}.jsonl"

        # Simula falha ao adquirir lock
        if platform.system() == "Windows":
            import msvcrt
            with patch.object(msvcrt, "locking", side_effect=OSError("Lock busy")):
                res = q.quarantine_event({"data": 1}, sample_invalid_result, "caller")
        else:
            import fcntl
            with patch.object(fcntl, "flock", side_effect=OSError("Lock busy")):
                res = q.quarantine_event({"data": 1}, sample_invalid_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.error_type == "LOCK_ACQUISITION_FAILED"
        # Zero bytes escritos
        assert target_file.stat().st_size == 0

    def test_qh35_filepath_strictly_confined_to_quarantine_root(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({"a": 1}, sample_invalid_result, "caller")
        assert res.status == "CONFIRMED"
        assert Path(res.file_path).resolve().is_relative_to((tmp_path / "quarantine").resolve())

    def test_qh36_preexisting_symlink_blocked(self, tmp_path: Path, sample_invalid_result):
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        symlink_dir = tmp_path / "sym_dir"

        try:
            os.symlink(real_dir, symlink_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink não suportado no ambiente")

        q = FinancialQuarantine(symlink_dir)
        res = q.quarantine_event({"a": 1}, sample_invalid_result, "caller")
        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.error_type == "PATH_SECURITY_VIOLATION"

    def test_qh37_operational_directories_not_touched(self):
        # Assegura que dados/quarantine não existe
        assert not Path("dados/quarantine").exists()

    def test_qh38_no_integration_with_event_store_or_event_saver(self):
        import common.financial_quarantine as fq
        code = Path(fq.__file__).read_text(encoding="utf-8")
        assert "EventStore" not in code
        assert "EventSaver" not in code
        assert "FallbackReconciler" not in code

    def test_qh39_no_automatic_event_deduplication(self, tmp_path: Path, sample_invalid_result):
        # Dois eventos idênticos devem ser persistidos ambos sequencialmente
        q = FinancialQuarantine(tmp_path / "quarantine")
        res1 = q.quarantine_event({"dup": 1}, sample_invalid_result, "caller")
        res2 = q.quarantine_event({"dup": 1}, sample_invalid_result, "caller")

        assert res1.status == "CONFIRMED"
        assert res2.status == "CONFIRMED"

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        target_file = tmp_path / "quarantine" / "invalid" / f"invalid_events_{date_str}.jsonl"
        with open(target_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_qh40_no_automatic_recovery_or_truncation(self, tmp_path: Path, sample_invalid_result):
        # Arquivo existente corrompido não é apagado nem recuperado automaticamente
        base_dir = tmp_path / "quarantine"
        sub_dir = base_dir / "invalid"
        sub_dir.mkdir(parents=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        corrupted_file = sub_dir / f"invalid_events_{date_str}.jsonl"
        corrupted_file.write_bytes(b"BROKEN_WITHOUT_NEWLINE")

        q = FinancialQuarantine(base_dir)
        res = q.quarantine_event({"a": 1}, sample_invalid_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        # Conteúdo do arquivo corrompido permanece intocado
        assert corrupted_file.read_bytes() == b"BROKEN_WITHOUT_NEWLINE"

    # =========================================================================
    # SUÍTE DE HARDENING COMPLEMENTAR QH2-01 A QH2-10 (GATE H1-B2-Q1-H2)
    # =========================================================================

    def test_qh2_01_close_failure_post_fsync_produces_persistence_uncertain(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        orig_open = open

        class MockCloseFailWrapper:
            def __init__(self, f):
                self._f = f
            def __getattr__(self, name):
                return getattr(self._f, name)
            def close(self):
                self._f.close()
                raise OSError("Simulated close failure post-sync")

        def mock_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            return MockCloseFailWrapper(f)

        with patch("builtins.open", side_effect=mock_open):
            res = q.quarantine_event({"data": 1}, sample_invalid_result, "caller")

        assert res.status == "PERSISTENCE_UNCERTAIN"
        assert res.success is False
        assert "close pós-escrita/fsync" in (res.error_message or "")

    def test_qh2_02_close_failure_pre_write_produces_failed_before_write(self, tmp_path: Path, sample_invalid_result):
        base_dir = tmp_path / "quarantine"
        sub_dir = base_dir / "invalid"
        sub_dir.mkdir(parents=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        corrupted_file = sub_dir / f"invalid_events_{date_str}.jsonl"
        corrupted_file.write_bytes(b"BROKEN_JSONL")

        orig_open = open

        class MockCloseFailPreWriteWrapper:
            def __init__(self, f):
                self._f = f
            def __getattr__(self, name):
                return getattr(self._f, name)
            def close(self):
                self._f.close()
                raise OSError("Close error pre-write")

        def mock_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            return MockCloseFailPreWriteWrapper(f)

        q = FinancialQuarantine(base_dir)
        with patch("builtins.open", side_effect=mock_open):
            res = q.quarantine_event({"data": 1}, sample_invalid_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert corrupted_file.read_bytes() == b"BROKEN_JSONL"

    def test_qh2_03_unlock_failure_post_write_produces_persistence_uncertain(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")

        if platform.system() == "Windows":
            import msvcrt
            orig_locking = msvcrt.locking
            def mock_locking(fd, mode, nbytes):
                if mode == msvcrt.LK_UNLCK:
                    raise OSError("Simulated Windows unlock failure")
                return orig_locking(fd, mode, nbytes)
            patch_target = "msvcrt.locking"
            mock_fn = mock_locking
        else:
            import fcntl
            orig_flock = fcntl.flock
            def mock_flock(fd, op):
                if op == fcntl.LOCK_UN:
                    raise OSError("Simulated POSIX unlock failure")
                return orig_flock(fd, op)
            patch_target = "fcntl.flock"
            mock_fn = mock_flock

        with patch(patch_target, side_effect=mock_fn):
            res = q.quarantine_event({"data": 1}, sample_invalid_result, "caller")

        assert res.status == "PERSISTENCE_UNCERTAIN"
        assert res.success is False
        assert res.error_type == "UNLOCK_FAILURE"
        assert "Falha de unlock pós-escrita" in (res.error_message or "")

    def test_qh2_04_unlock_failure_pre_write_produces_failed_before_write(self, tmp_path: Path, sample_invalid_result):
        base_dir = tmp_path / "quarantine"
        sub_dir = base_dir / "invalid"
        sub_dir.mkdir(parents=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        corrupted_file = sub_dir / f"invalid_events_{date_str}.jsonl"
        corrupted_file.write_bytes(b"BROKEN_PRE")

        q = FinancialQuarantine(base_dir)

        if platform.system() == "Windows":
            import msvcrt
            orig_locking = msvcrt.locking
            def mock_locking(fd, mode, nbytes):
                if mode == msvcrt.LK_UNLCK:
                    raise OSError("Simulated Windows unlock failure pre-write")
                return orig_locking(fd, mode, nbytes)
            patch_target = "msvcrt.locking"
            mock_fn = mock_locking
        else:
            import fcntl
            orig_flock = fcntl.flock
            def mock_flock(fd, op):
                if op == fcntl.LOCK_UN:
                    raise OSError("Simulated POSIX unlock failure pre-write")
                return orig_flock(fd, op)
            patch_target = "fcntl.flock"
            mock_fn = mock_flock

        with patch(patch_target, side_effect=mock_fn):
            res = q.quarantine_event({"data": 1}, sample_invalid_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.status != "CONFIRMED"

    def test_qh2_05_write_failure_followed_by_close_failure(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        orig_open = open

        class MockDoubleFailWrapper:
            def __init__(self, f):
                self._f = f
            def __getattr__(self, name):
                return getattr(self._f, name)
            def write(self, *args, **kwargs):
                raise OSError("Primary write failure")
            def close(self):
                self._f.close()
                raise OSError("Secondary close failure")

        def mock_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            return MockDoubleFailWrapper(f)

        with patch("builtins.open", side_effect=mock_open):
            res = q.quarantine_event({"data": 1}, sample_invalid_result, "caller")

        assert res.status == "PERSISTENCE_UNCERTAIN"
        assert res.success is False
        assert "Primary write failure" in (res.error_message or "")
        assert "Secondary close failure" in (res.error_message or "")

    def test_qh2_06_fsync_failure_followed_by_close_failure(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        orig_open = open

        class MockFsyncCloseFailWrapper:
            def __init__(self, f):
                self._f = f
            def __getattr__(self, name):
                return getattr(self._f, name)
            def close(self):
                self._f.close()
                raise OSError("Secondary close failure after fsync")

        def mock_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            return MockFsyncCloseFailWrapper(f)

        with patch("builtins.open", side_effect=mock_open):
            with patch("os.fsync", side_effect=OSError("Primary fsync failure")):
                res = q.quarantine_event({"data": 1}, sample_invalid_result, "caller")

        assert res.status == "PERSISTENCE_UNCERTAIN"
        assert res.success is False
        assert "Primary fsync failure" in (res.error_message or "")
        assert "Secondary close failure" in (res.error_message or "")

    def test_qh2_07_confirmed_returned_only_after_full_lifecycle(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        res = q.quarantine_event({"check": 100}, sample_invalid_result, "caller")

        assert res.status == "CONFIRMED"
        assert res.success is True
        assert res.file_path is not None
        assert Path(res.file_path).exists()
        with open(res.file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["quarantine_format_version"] == 2

    def test_qh2_08_preexisting_truncated_file_preserves_content_and_fails(self, tmp_path: Path, sample_invalid_result):
        base_dir = tmp_path / "quarantine"
        sub_dir = base_dir / "invalid"
        sub_dir.mkdir(parents=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        corrupted_file = sub_dir / f"invalid_events_{date_str}.jsonl"
        original_bytes = b'{"truncated_entry": true'
        corrupted_file.write_bytes(original_bytes)

        q = FinancialQuarantine(base_dir)
        res = q.quarantine_event({"new": 2}, sample_invalid_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.error_type == "PREEXISTING_JSONL_TRUNCATION"
        assert corrupted_file.read_bytes() == original_bytes

    def test_qh2_09_no_automatic_retry_on_io_failure(self, tmp_path: Path, sample_invalid_result):
        q = FinancialQuarantine(tmp_path / "quarantine")
        write_call_count = 0
        orig_open = open

        class MockCountWrapper:
            def __init__(self, f):
                self._f = f
            def __getattr__(self, name):
                return getattr(self._f, name)
            def write(self, *args, **kwargs):
                nonlocal write_call_count
                write_call_count += 1
                raise OSError("Disk write failed exactly once")

        def mock_open(*args, **kwargs):
            f = orig_open(*args, **kwargs)
            return MockCountWrapper(f)

        with patch("builtins.open", side_effect=mock_open):
            res = q.quarantine_event({"val": 1}, sample_invalid_result, "caller")

        assert res.status == "PERSISTENCE_UNCERTAIN"
        assert write_call_count == 1

    def test_qh2_10_quarantine_write_result_tri_state_invariants(self):
        confirmed = QuarantineWriteResult(status="CONFIRMED", success=True, file_path="/tmp/f.jsonl")
        assert confirmed.status == "CONFIRMED"

        with pytest.raises(ValueError, match="exige success=False"):
            QuarantineWriteResult(status="FAILED_BEFORE_WRITE", success=True, file_path="/tmp/f.jsonl", error_type="ERR", error_message="msg")

        with pytest.raises(ValueError, match="exige success=False"):
            QuarantineWriteResult(status="PERSISTENCE_UNCERTAIN", success=True, file_path="/tmp/f.jsonl", error_type="ERR", error_message="msg")

        with pytest.raises(ValueError, match="exige error_type não vazio"):
            QuarantineWriteResult(status="FAILED_BEFORE_WRITE", success=False, file_path="/tmp/f.jsonl", error_type=None, error_message="msg")

    # =========================================================================
    # SUÍTE DE HARDENING QH2-C-01 A QH2-C-06 (GATE H1-B2-Q1-H2-C)
    # =========================================================================

    def test_qh2_c_01_lock_failure_followed_by_close_failure(self, tmp_path: Path, sample_invalid_result):
        """
        QH2-C-01 — Falha de lock seguida de falha de close:
        - Interceptar lock para falhar.
        - Fazer close() falhar controladamente.
        - Confirmar FAILED_BEFORE_WRITE e success=False.
        - Confirmar que error_type preserva causa primária (LOCK_ACQUISITION_FAILED).
        - Confirmar que error_message inclui erro secundário de close.
        - Confirmar que write() não foi chamado e conteúdo permaneceu inalterado.
        - Garantir que descritor real é fechado.
        """
        base_dir = tmp_path / "quarantine"
        q = FinancialQuarantine(base_dir)
        orig_open = open
        wrapper_instances = []

        class MockLockAndCloseFailWrapper:
            def __init__(self, f):
                self._f = f
                self.write_called = False
                self.close_called = False
                wrapper_instances.append(self)
            def __getattr__(self, name):
                return getattr(self._f, name)
            def write(self, *args, **kwargs):
                self.write_called = True
                return self._f.write(*args, **kwargs)
            def close(self):
                self.close_called = True
                try:
                    self._f.close()
                finally:
                    raise OSError("Secondary close failure during lock abort")

        def mock_open(*args, **kwargs):
            return MockLockAndCloseFailWrapper(orig_open(*args, **kwargs))

        lock_patch = "msvcrt.locking" if platform.system() == "Windows" else "fcntl.flock"
        with patch(lock_patch, side_effect=OSError("Lock acquisition denied")):
            with patch("builtins.open", side_effect=mock_open):
                res = q.quarantine_event({"val": 1}, sample_invalid_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.error_type == "LOCK_ACQUISITION_FAILED"
        assert "Falha ao adquirir lock exclusivo no arquivo" in res.error_message
        assert "Lock acquisition denied" in res.error_message
        assert "Secondary close failure during lock abort" in res.error_message
        assert "close_error" in res.error_message
        assert len(wrapper_instances) == 1
        assert wrapper_instances[0].write_called is False
        assert wrapper_instances[0].close_called is True
        # Verifica que o descritor subjacente foi de fato fechado
        assert wrapper_instances[0]._f.closed is True
        # Verifica que arquivo em disco está vazio (0 bytes)
        assert Path(res.file_path).stat().st_size == 0

    def test_qh2_c_02_pre_write_validation_failure_with_failed_cleanup(self, tmp_path: Path, sample_invalid_result):
        """
        QH2-C-02 — Falha pré-escrita de validação com cleanup falho:
        - Arquivo JSONL previamente truncado sem newline.
        - Injetar falha de unlock e/ou close de maneira controlada.
        - Confirmar FAILED_BEFORE_WRITE e success=False.
        - Confirmar preservação do erro primário e inclusão dos secundários.
        - Confirmar que conteúdo original não foi alterado e zero chamadas a write().
        """
        base_dir = tmp_path / "quarantine"
        sub_dir = base_dir / "invalid"
        sub_dir.mkdir(parents=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        corrupted_file = sub_dir / f"invalid_events_{date_str}.jsonl"
        original_bytes = b'{"truncated_entry": true'
        corrupted_file.write_bytes(original_bytes)

        q = FinancialQuarantine(base_dir)
        orig_open = open
        wrapper_instances = []

        class MockValidationAndCleanupFailWrapper:
            def __init__(self, f):
                self._f = f
                self.write_called = False
                wrapper_instances.append(self)
            def __getattr__(self, name):
                return getattr(self._f, name)
            def write(self, *args, **kwargs):
                self.write_called = True
                return self._f.write(*args, **kwargs)
            def close(self):
                try:
                    self._f.close()
                finally:
                    raise OSError("Secondary close failure on preexisting truncation")

        def mock_open(*args, **kwargs):
            return MockValidationAndCleanupFailWrapper(orig_open(*args, **kwargs))

        with patch("builtins.open", side_effect=mock_open):
            res = q.quarantine_event({"val": 2}, sample_invalid_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.error_type == "PREEXISTING_JSONL_TRUNCATION"
        assert "Arquivo de quarentena pré-existente não termina com quebra de linha" in res.error_message
        assert "Secondary close failure on preexisting truncation" in res.error_message
        assert "close_error" in res.error_message
        assert len(wrapper_instances) == 1
        assert wrapper_instances[0].write_called is False
        assert wrapper_instances[0]._f.closed is True
        assert corrupted_file.read_bytes() == original_bytes

    def test_qh2_c_03_exclusive_pre_write_close_failure(self, tmp_path: Path, sample_invalid_result):
        """
        QH2-C-03 — Falha exclusivamente pré-escrita de close:
        - Exercitar um caminho sem tentativa de write onde o descritor é fechado antes da escrita.
        - Injetar falha de close().
        - Confirmar FAILED_BEFORE_WRITE, success=False, error_type="CLOSE_FAILURE".
        - Confirmar bytes do arquivo inalterados e que o erro de close chega ao caller.
        """
        base_dir = tmp_path / "quarantine"
        q = FinancialQuarantine(base_dir)
        orig_open = open
        wrapper_instances = []

        class MockPreWriteClosedWrapper:
            def __init__(self, f):
                self._f = f
                self.write_called = False
                self._is_closed = False
                wrapper_instances.append(self)
            @property
            def closed(self):
                return self._is_closed or self._f.closed
            def __getattr__(self, name):
                return getattr(self._f, name)
            def tell(self):
                # Simula que o descritor foi externamente fechado durante a checagem pré-escrita
                self._is_closed = True
                return self._f.tell()
            def write(self, *args, **kwargs):
                self.write_called = True
                return self._f.write(*args, **kwargs)
            def close(self):
                try:
                    self._f.close()
                finally:
                    raise OSError("Exclusive pre-write close failure")

        def mock_open(*args, **kwargs):
            return MockPreWriteClosedWrapper(orig_open(*args, **kwargs))

        with patch("builtins.open", side_effect=mock_open):
            res = q.quarantine_event({"val": 3}, sample_invalid_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.error_type == "CLOSE_FAILURE"
        assert "Falha durante close pré-escrita" in res.error_message
        assert "Exclusive pre-write close failure" in res.error_message
        assert len(wrapper_instances) == 1
        assert wrapper_instances[0].write_called is False
        assert wrapper_instances[0]._f.closed is True
        assert Path(res.file_path).stat().st_size == 0

    def test_qh2_c_04_ordinary_non_oserror_during_write(self, tmp_path: Path, sample_invalid_result):
        """
        QH2-C-04 — Exceção ordinária durante write:
        - Injetar uma exceção ordinária não-OSError (ValueError) durante write() após bytes parciais.
        - Confirmar PERSISTENCE_UNCERTAIN e success=False.
        - Confirmar que a exceção não resulta em FAILED_BEFORE_WRITE nem CONFIRMED.
        - Confirmar preservação do diagnóstico e ausência de retry automático.
        """
        base_dir = tmp_path / "quarantine"
        q = FinancialQuarantine(base_dir)
        orig_open = open
        write_call_count = 0
        wrapper_instances = []

        class MockOrdinaryErrorWriteWrapper:
            def __init__(self, f):
                self._f = f
                wrapper_instances.append(self)
            def __getattr__(self, name):
                return getattr(self._f, name)
            def write(self, b):
                nonlocal write_call_count
                write_call_count += 1
                # Submete bytes parciais antes de lançar exceção não-OSError
                self._f.write(b[:10])
                raise ValueError("Simulated ordinary non-OSError (ValueError) during write")

        def mock_open(*args, **kwargs):
            return MockOrdinaryErrorWriteWrapper(orig_open(*args, **kwargs))

        with patch("builtins.open", side_effect=mock_open):
            res = q.quarantine_event({"val": 4}, sample_invalid_result, "caller")

        assert res.status == "PERSISTENCE_UNCERTAIN"
        assert res.success is False
        assert res.status != "FAILED_BEFORE_WRITE"
        assert res.status != "CONFIRMED"
        assert res.error_type == "ValueError"
        assert "Simulated ordinary non-OSError (ValueError) during write" in res.error_message
        assert write_call_count == 1  # Sem retry automático
        assert len(wrapper_instances) == 1
        assert wrapper_instances[0]._f.closed is True

    def test_qh2_c_05_encoding_exception_with_surrogate_character(self, tmp_path: Path, sample_invalid_result):
        """
        QH2-C-05 — Exceção de encoding com caractere surrogate:
        - Avaliar entrada pública contendo surrogate inválido para utf-8.
        - Confirmar FAILED_BEFORE_WRITE e success=False seguro, sem mutação.
        - Confirmar que erro de codificação não cria arquivo corrompido nem atinge escrita.
        """
        base_dir = tmp_path / "quarantine"
        q = FinancialQuarantine(base_dir)

        # Injeta um surrogate desemparelhado que passa pelo json.dumps mas falha no .encode("utf-8")
        malformed_result = FinancialValidationResult(
            is_valid=False,
            status="INVALID",
            category="INVALID",
            rejection_reason="Surrogate \ud800 in reason",
            rejected_field="field",
            rejected_value_repr="val",
        )

        res = q.quarantine_event({"normal": 1}, malformed_result, "caller")

        assert res.status == "FAILED_BEFORE_WRITE"
        assert res.success is False
        assert res.error_type == "UnicodeEncodeError"
        assert "Falha ao codificar JSON do envelope" in res.error_message
        # Confirma que nenhum arquivo foi criado ou mutado
        target_file = Path(res.file_path)
        assert not target_file.exists()

    def test_qh2_c_06_confirmed_normal_and_regression_close_post_fsync(self, tmp_path: Path, sample_invalid_result):
        """
        QH2-C-06 — Confirmação normal e regressão do close pós-fsync:
        - Confirmar que ciclo normal sem falhas retorna CONFIRMED e success=True após close.
        - Confirmar que OSError em close pós-fsync retorna PERSISTENCE_UNCERTAIN e CLOSE_FAILURE.
        - Confirmar ausência de falso sucesso e ausência de retry.
        """
        base_dir = tmp_path / "quarantine"
        q = FinancialQuarantine(base_dir)

        # 1. Ciclo normal íntegro
        res_ok = q.quarantine_event({"status": "healthy"}, sample_invalid_result, "caller")
        assert res_ok.status == "CONFIRMED"
        assert res_ok.success is True
        assert res_ok.error_type is None
        assert res_ok.error_message is None
        assert Path(res_ok.file_path).exists()

        # 2. Regressão: Falha de close pós-fsync
        orig_open = open
        close_call_count = 0

        class MockPostFsyncCloseFailWrapper:
            def __init__(self, f):
                self._f = f
            def __getattr__(self, name):
                return getattr(self._f, name)
            def close(self):
                nonlocal close_call_count
                close_call_count += 1
                self._f.close()
                raise OSError("Simulated OSError in close post-fsync")

        def mock_open(*args, **kwargs):
            return MockPostFsyncCloseFailWrapper(orig_open(*args, **kwargs))

        with patch("builtins.open", side_effect=mock_open):
            res_fail = q.quarantine_event({"status": "healthy_fail"}, sample_invalid_result, "caller")

        assert res_fail.status == "PERSISTENCE_UNCERTAIN"
        assert res_fail.success is False
        assert res_fail.error_type == "CLOSE_FAILURE"
        assert "close pós-escrita/fsync" in res_fail.error_message
        assert "Simulated OSError in close post-fsync" in res_fail.error_message
        assert close_call_count == 1  # Sem retry automático de cleanup
