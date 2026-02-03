"""
Fast Multipole Method (FMM) Implementation

A comprehensive implementation of the Fast Multipole Method for
solving N-body problems with O(N) complexity.

Based on the PhD thesis research covering:
- Chapter 2: Standard FMM with kernel-specific expansions
- Chapter 3: High-frequency FMM for Helmholtz
- Chapter 4: Kernel-independent FMM with ACA/SVD compression

This package includes:
- Core FMM operators (P2M, M2M, M2L, L2L, L2P, P2P, M2P, P2L)
- Kernel-specific implementations (Laplace, Helmholtz)
- High-frequency Helmholtz with spherical cubature
- Kernel-independent FMM using Chebyshev interpolation
- Adaptive Cross Approximation (ACA) compression
- Dual Tree Traversal (DTT) for adaptive trees
- UVWXY-lists for non-uniform distributions
- Both 2D and 3D support
"""

from fmm.core import (
    Particle,
    Cell,
    CellType,
    Tree,
    TreeConfig,
    FMM,
    StandardFMM,
    KernelIndependentFMM,
    MultipoleExpansion,
    LocalExpansion,
    M2P,
    P2L,
)
from fmm.kernels import (
    Kernel,
    LaplaceKernel,
    HelmholtzKernel,
    YukawaKernel,
    CoulombKernel,
    StokesKernel,
    create_kernel,
)

__version__ = '0.2.0'

__all__ = [
    # Core classes
    'Particle',
    'Cell',
    'CellType',
    'Tree',
    'TreeConfig',
    'FMM',
    'StandardFMM',
    'KernelIndependentFMM',
    'MultipoleExpansion',
    'LocalExpansion',
    'M2P',
    'P2L',
    # Kernels
    'Kernel',
    'LaplaceKernel',
    'HelmholtzKernel',
    'YukawaKernel',
    'CoulombKernel',
    'StokesKernel',
    'create_kernel',
]
