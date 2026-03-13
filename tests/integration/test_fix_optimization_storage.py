from fix_optimization import clean_event, remove_enriched_snapshot, simplify_historical_vp


def test_clean_event_removes_large_fields_recursively():
    evt = {
        "tipo_evento": "ANALYSIS_TRIGGER",
        "observability": {"a": 1},
        "nested": {
            "enriched_snapshot": {"big": "x" * 1000},
            "vp": {"volume_nodes": [1, 2, 3], "single_prints": [4, 5]},
        },
        "list": [{"observability": {"b": 2}}],
    }

    cleaned = clean_event(evt)
    assert "observability" not in cleaned
    assert "enriched_snapshot" not in cleaned["nested"]
    assert "volume_nodes" not in cleaned["nested"]["vp"]
    assert "single_prints" not in cleaned["nested"]["vp"]
    assert "observability" not in cleaned["list"][0]


def test_simplify_historical_vp_keeps_only_expected_fields():
    evt = {
        "historical_vp": {
            "daily": {
                "poc": 1,
                "vah": 2,
                "val": 3,
                "hvns_nearby": [1, 2],
                "lvns_nearby": [3, 4],
                "extra": "drop",
                "volume_nodes": [1],
            },
            "weekly": {"poc": 10, "vah": 20, "val": 30, "status": "ok", "extra": "drop"},
            "monthly": {"poc": 100, "vah": 200, "val": 300, "status": "ok", "single_prints": [1]},
        }
    }

    simplified = simplify_historical_vp(evt)
    hvp = simplified["historical_vp"]
    assert set(hvp["daily"].keys()) <= {"poc", "vah", "val", "hvns_nearby", "lvns_nearby", "status"}
    assert set(hvp["weekly"].keys()) <= {"poc", "vah", "val", "status"}
    assert set(hvp["monthly"].keys()) <= {"poc", "vah", "val", "status"}


def test_remove_enriched_snapshot_removes_anywhere():
    evt = {"a": {"enriched_snapshot": {"x": 1}}, "enriched_snapshot": {"y": 2}}
    out = remove_enriched_snapshot(evt)
    assert "enriched_snapshot" not in out
    assert "enriched_snapshot" not in out["a"]

