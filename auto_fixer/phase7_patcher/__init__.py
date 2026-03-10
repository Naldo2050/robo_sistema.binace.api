"""
Fase 7 - Patch Generator.
Gera e aplica patches automaticamente.
"""

from .patch_generator import PatchGenerator, Patch
from .patch_validator import PatchValidator
from .patch_applier import PatchApplier, RollbackManager

__all__ = [
    "PatchGenerator",
    "Patch",
    "PatchValidator",
    "PatchApplier",
    "RollbackManager",
]
