# tests/unit/test_quality_summary_latency.py
# -*- coding: utf-8 -*-
"""
Fix 1 — quality_summary: POOR não penalizava confiança.

Casos obrigatórios:
  - latency_category="OK"      -> confidence_cap == 1.0, reliable == 1
  - latency_category="DEGR"    -> confidence_cap == 0.7, reliable == 0
  - latency_category="POOR"    -> confidence_cap == 0.4, reliable == 0
  - latency_category="UNKNOWN" -> confidence_cap == 0.3, reliable == 0
  - latency_category=""        -> confidence_cap == 0.3, reliable == 0
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from market_orchestrator.ai.payload_sections.quality_summary import (
    build_quality_summary,
)


def _summary_for(lat_cat: str) -> dict:
    payload = {
        "qual": {"lat": lat_cat, "ms": 7312, "liq": "NORMAL"},
        "ctx": {},
    }
    return build_quality_summary(payload)


@pytest.mark.parametrize(
    "lat_cat,expected_cap,expected_reliable",
    [
        ("OK", 1.0, 1),
        ("DEGR", 0.7, 0),
        ("POOR", 0.4, 0),
        ("UNKNOWN", 0.3, 0),
        ("", 0.3, 0),
    ],
)
def test_latency_caps(lat_cat, expected_cap, expected_reliable):
    summary = _summary_for(lat_cat)
    assert summary["confidence_cap"] == expected_cap
    assert summary["reliable"] == expected_reliable
