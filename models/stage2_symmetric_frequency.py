"""Backward-compatible import shim.

RAPD-DSwin keeps the dual-source symmetric SSP implementation in
``frequency_decomposition.py``. Older Stage-2 files still import this module
name, so this shim preserves compatibility without duplicating code.
"""

from .frequency_decomposition import (
    Stage2SymmetricFrequencyNet,
    SymmetricFrequencyReliabilityScreen,
)

__all__ = [
    "Stage2SymmetricFrequencyNet",
    "SymmetricFrequencyReliabilityScreen",
]
