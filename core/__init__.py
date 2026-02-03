"""
FMM Core Module

This module contains the core data structures and base classes for the
Fast Multipole Method implementation.
"""

from .particle import Particle
from .cell import Cell, CellType
from .tree import Tree, TreeConfig
from .expansion import Expansion, MultipoleExpansion, LocalExpansion
from .operators import Operator, P2M, M2M, M2L, L2L, L2P, P2P, M2P, P2L
from .fmm import FMM, StandardFMM, KernelIndependentFMM
from .kernel_independent import (
    KernelIndependentExpansion,
    KernelIndependentP2M,
    KernelIndependentM2L,
    ChebyshevInterpolant,
    SVDM2LCompressor,
    ACACompressor
)

__all__ = [
    'Particle',
    'Cell',
    'CellType',
    'Tree',
    'TreeConfig',
    'Expansion',
    'MultipoleExpansion',
    'LocalExpansion',
    'Operator',
    'P2M',
    'M2M',
    'M2L',
    'L2L',
    'L2P',
    'P2P',
    'M2P',
    'P2L',
    'FMM',
    'StandardFMM',
    'KernelIndependentFMM',
    'KernelIndependentExpansion',
    'KernelIndependentP2M',
    'KernelIndependentM2L',
    'ChebyshevInterpolant',
    'SVDM2LCompressor',
    'ACACompressor',
]
