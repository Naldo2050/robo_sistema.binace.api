# diagnose_optimization.py — proxy de compatibilidade
# Modulo movido para scripts/diagnostics/diagnose_optimization.py
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "diagnose_optimization",
    Path(__file__).parent / "scripts" / "diagnostics" / "diagnose_optimization.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

has_key = _mod.has_key
iter_events = _mod.iter_events
main = _mod.main
FORBIDDEN = _mod.FORBIDDEN
TARGET_TYPES = _mod.TARGET_TYPES
