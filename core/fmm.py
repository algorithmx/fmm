"""
Main FMM Module

Implements the Fast Multipole Method algorithm that orchestrates
all operators in the upward and downward passes.

Includes high-frequency FMM support with diagonal operators and
directional decomposition for extreme high-frequency problems.
"""

from typing import List, Dict, Optional, Callable, Union
import numpy as np
from abc import ABC, abstractmethod

from .tree import Tree, TreeConfig
from .cell import Cell
from .particle import Particle
from .expansion import MultipoleExpansion, LocalExpansion
from .operators import (
    P2M, M2M, M2L, L2L, L2P, P2P,
    HighFrequencyM2L, HighFrequencyM2M, HighFrequencyL2L, Cubature
)
from .kernel_independent import (
    KernelIndependentExpansion,
    KernelIndependentP2M,
    KernelIndependentM2L,
    FFTBasedM2L,
    AdaptiveM2L
)
from .directional_fmm import DirectionalFMM, AdaptiveDirectionalFMM


class FMM(ABC):
    """
    Abstract base class for Fast Multipole Method implementations.

    Provides the core framework for FMM with specific implementations
    for kernel-specific and kernel-independent variants.
    """

    def __init__(self, particles: List[Particle], kernel_func: Callable,
                 config: Optional[TreeConfig] = None):
        """
        Initialize FMM.

        Args:
            particles: List of particles
            kernel_func: Kernel function G(x, y) computing potential
            config: Tree configuration (optional)
        """
        if config is None:
            config = TreeConfig()

        self.particles = particles
        self.kernel_func = kernel_func
        self.config = config

        # Build tree
        self.tree = Tree(particles, config)

        # Storage for expansions
        self.multipole_expansions: Dict[int, MultipoleExpansion] = {}
        self.local_expansions: Dict[int, LocalExpansion] = {}

        # Initialize operators
        self.p2m = P2M(config.expansion_order, config.dimension)
        self.m2m = M2M(config.expansion_order, config.dimension)
        self.m2l = M2L(config.expansion_order, config.dimension)
        self.l2l = L2L(config.expansion_order, config.dimension)
        self.l2p = L2P(config.expansion_order, config.dimension)
        self.p2p = P2P(kernel_func, config.dimension)

    def compute(self) -> np.ndarray:
        """
        Compute all particle potentials using FMM.

        Returns:
            Array of potentials for each particle
        """
        # Reset all particle potentials
        for particle in self.particles:
            particle.reset_potential()

        # Upward pass: build multipole expansions
        self._upward_pass()

        # Downward pass: distribute to local expansions
        self._downward_pass()

        # Direct near-field computation (P2P)
        self._direct_pass()

        # Return potentials
        return np.array([p.potential for p in self.particles])

    def _upward_pass(self):
        """
        Upward pass: Build multipole expansions from leaves to root.

        1. P2M: Convert particles to multipole expansions at leaves
        2. M2M: Aggregate multipole expansions up the tree
        """
        # Process leaves bottom-up
        for level in range(self.tree.get_max_level(), 0, -1):
            cells = self.tree.get_cells_at_level(level)

            for cell in cells:
                if cell.is_leaf:
                    # P2M: particles to multipole
                    expansion = self.p2m.apply(cell, self.kernel_func)
                    self.multipole_expansions[id(cell)] = expansion
                else:
                    # M2M: aggregate children's multipole expansions
                    if cell.children:
                        # Combine all children's expansions
                        for child in cell.children:
                            if id(child) in self.multipole_expansions:
                                child_exp = self.multipole_expansions[id(child)]
                                # Translate to parent
                                parent_exp = self.m2m.apply(
                                    child.center,
                                    cell.center,
                                    child_exp
                                )

                                if id(cell) in self.multipole_expansions:
                                    # Add to existing parent expansion
                                    self.multipole_expansions[id(cell)].add(parent_exp)
                                else:
                                    self.multipole_expansions[id(cell)] = parent_exp

    def _downward_pass(self):
        """
        Downward pass: Distribute multipole expansions to local expansions.

        1. M2L: Convert multipole expansions to local expansions
        2. L2L: Propagate local expansions down the tree
        """
        # Process top-down
        for level in range(1, self.tree.get_max_level() + 1):
            cells = self.tree.get_cells_at_level(level)

            for cell in cells:
                # Initialize local expansion
                local_exp = LocalExpansion(
                    center=cell.center,
                    order=self.config.expansion_order,
                    dimension=self.config.dimension
                )

                # M2L: contributions from well-separated cells
                interaction_list = self.tree.get_interaction_list(cell)
                for source_cell in interaction_list:
                    if id(source_cell) in self.multipole_expansions:
                        source_exp = self.multipole_expansions[id(source_cell)]
                        # Convert to local expansion
                        contrib_exp = self.m2l.apply(source_exp, cell.center)
                        local_exp.add(contrib_exp)

                self.local_expansions[id(cell)] = local_exp

                # L2L: propagate to children
                if not cell.is_leaf and cell.children:
                    for child in cell.children:
                        child_local = self.l2l.apply(
                            cell.center,
                            child.center,
                            local_exp
                        )

                        if id(child) in self.local_expansions:
                            self.local_expansions[id(child)].add(child_local)
                        else:
                            self.local_expansions[id(child)] = child_local

    def _direct_pass(self):
        """
        Direct pass: Compute near-field interactions directly.

        P2P: Direct particle-to-particle computation for adjacent cells.
        """
        leaves = self.tree.leaves

        for leaf in leaves:
            # Get adjacent leaves
            neighbors = self.tree.get_near_field_neighbors(leaf)

            # P2P with neighbors
            for neighbor in neighbors + [leaf]:
                self.p2p.apply(neighbor.particles, leaf.particles)

            # L2P: convert local expansion to particle potentials
            if id(leaf) in self.local_expansions:
                self.l2p.apply(self.local_expansions[id(leaf)], leaf.particles)

    def get_error_estimate(self, reference: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Estimate FMM error compared to direct computation.

        Args:
            reference: Reference potentials from direct computation (optional)

        Returns:
            Dictionary with error metrics
        """
        if reference is None:
            # Compute reference using direct method
            reference = self._direct_compute()

        fmm_result = np.array([p.potential for p in self.particles])

        # Compute error metrics
        abs_error = np.abs(fmm_result - reference)
        rel_error = abs_error / (np.abs(reference) + 1e-14)

        return {
            'max_absolute_error': np.max(abs_error),
            'mean_absolute_error': np.mean(abs_error),
            'max_relative_error': np.max(rel_error),
            'mean_relative_error': np.mean(rel_error),
            'l2_error': np.linalg.norm(fmm_result - reference) / np.linalg.norm(reference)
        }

    def _direct_compute(self) -> np.ndarray:
        """Compute potentials directly (O(N^2)) for validation."""
        potentials = np.zeros(len(self.particles))

        for i, target in enumerate(self.particles):
            for j, source in enumerate(self.particles):
                if i != j:
                    value = self.kernel_func(source.position, target.position)
                    potentials[i] += source.charge * value

        return potentials


class StandardFMM(FMM):
    """
    Standard Fast Multipole Method for Laplace kernel.

    Implements Algorithm 4 (Upward Pass) and Algorithm 5 (Downward Pass)
    from Chapter 2 of the thesis.

    Operators: P2M → M2M → M2L → L2L → L2P

    Complexity: O(N log N) for N particles

    Standard FMM implementation using kernel-specific analytical expansions.
    """

    def __init__(self, particles: List[Particle], kernel_func: Callable,
                 config: Optional[TreeConfig] = None):
        super().__init__(particles, kernel_func, config)


class KernelIndependentFMM(FMM):
    """
    Kernel-Independent FMM using Chebyshev interpolation.

    Implements Chapter 4 methodology:
    - Chebyshev node interpolation (avoids Runge phenomenon)
    - SVD/ACA compression for low-rank M2L
    - FFT acceleration for translation-invariant kernels

    Complexity: O(N log N) for general kernels

    This approach works for any kernel function without requiring
    analytical expansions.
    """

    def __init__(self, particles: List[Particle], kernel_func: Callable,
                 config: Optional[TreeConfig] = None,
                 compression_tolerance: float = 1e-6):
        """
        Initialize kernel-independent FMM.

        Args:
            particles: List of particles
            kernel_func: Kernel function G(x, y)
            config: Tree configuration
            compression_tolerance: SVD compression tolerance
        """
        if config is None:
            config = TreeConfig()

        super().__init__(particles, kernel_func, config)

        # Use kernel-independent operators
        self.ki_p2m = KernelIndependentP2M(
            config.expansion_order,
            config.dimension
        )
        self.ki_m2l = KernelIndependentM2L(
            config.expansion_order,
            config.dimension,
            compression_tolerance
        )

        # Storage for kernel-independent expansions
        self.ki_expansions: Dict[int, KernelIndependentExpansion] = {}

    def _upward_pass(self):
        """Upward pass using kernel-independent P2M."""
        for level in range(self.tree.get_max_level(), 0, -1):
            cells = self.tree.get_cells_at_level(level)

            for cell in cells:
                if cell.is_leaf:
                    # Kernel-independent P2M
                    expansion = self.ki_p2m.apply(cell, self.kernel_func)
                    self.ki_expansions[id(cell)] = expansion

    def _downward_pass(self):
        """Downward pass using kernel-independent M2L."""
        leaves = self.tree.leaves

        for leaf in leaves:
            # Initialize local expansion
            local_exp = KernelIndependentExpansion(
                center=leaf.center,
                size=leaf.size,
                order=self.config.expansion_order,
                dimension=self.config.dimension
            )

            # M2L from interaction list
            interaction_list = self.tree.get_interaction_list(leaf)
            for source_cell in interaction_list:
                if id(source_cell) in self.ki_expansions:
                    source_exp = self.ki_expansions[id(source_cell)]
                    contrib_exp = self.ki_m2l.apply(
                        source_exp,
                        leaf.center,
                        leaf.size,
                        self.kernel_func
                    )
                    local_exp.add(contrib_exp)

            self.ki_expansions[id(leaf)] = local_exp

    def _direct_pass(self):
        """Direct pass with kernel-independent L2P."""
        leaves = self.tree.leaves

        for leaf in leaves:
            # P2P with neighbors
            neighbors = self.tree.get_near_field_neighbors(leaf)
            for neighbor in neighbors + [leaf]:
                self.p2p.apply(neighbor.particles, leaf.particles)

            # L2P: evaluate local expansion at particle positions
            if id(leaf) in self.ki_expansions:
                expansion = self.ki_expansions[id(leaf)]
                positions = np.array([p.position for p in leaf.particles])
                potentials = expansion.evaluate(positions)

                for particle, potential in zip(leaf.particles, potentials):
                    particle.potential += potential


# ============================================================================
# High-Frequency FMM Implementations
# ============================================================================

class HighFrequencyFMM(StandardFMM):
    """
    High-Frequency FMM for Helmholtz kernel.

    Implements Chapter 3 methodology with diagonal M2L operators:
    - P2M: exp(-iκ⟨ŷ,λ⟩) phase factor
    - M2M: exp(-iκ⟨Δctr,λ⟩) translation
    - M2L: Diagonal operator T_L(λ) - O(Q) complexity
    - L2L: exp(+iκ⟨Δctr,λ⟩) translation
    - L2P: exp(+iκ⟨x-c,λ⟩) phase factor (adjoint of P2M)

    Adaptive order: L ≈ κw + (1.8d₀)^(2/3) × (κw)^(1/3)

    Complexity: O(N log N + NQ) where Q = O((κw)^(d-1)/d)

    From Chapter 3:
    For high-frequency Helmholtz equation (∇² + κ²)u = f, we use:
    - Diagonal M2L operators: O(Q) instead of O(Q²)
    - Spherical interpolation for M2M/L2L between different cubature grids
    - Adaptive order selection based on local frequency parameter κw

    Key optimizations:
    1. Diagonal M2L: (M2L * q)(λ_p) = T_L(t, λ_p) * q(λ_p)
    2. Fast spherical interpolation: O(L² log L) instead of O(L⁴)
    3. Adaptive order: L ∼ κw for optimal accuracy
    """

    def __init__(self, particles: List[Particle], kernel_func: Callable,
                 wavenumber: float, config: Optional[TreeConfig] = None):
        """
        Initialize high-frequency FMM.

        Args:
            particles: List of particles
            kernel_func: Helmholtz kernel function
            wavenumber: Wavenumber κ
            config: Tree configuration
        """
        if config is None:
            config = TreeConfig()

        # Enable high-frequency features
        config.use_high_frequency_m2l = True
        config.use_spherical_interpolation = True

        super().__init__(particles, kernel_func, config)

        self.wavenumber = wavenumber
        self.hf_m2l = None
        self.hf_m2m = None
        self.hf_l2l = None

        # Storage for high-frequency expansions (cubature coefficients)
        self.hf_expansions: Dict[int, np.ndarray] = {}

        # Determine if we should use high-frequency operators
        kw = self._compute_kw()
        if kw > 1.0:  # Threshold for using high-frequency operators
            self._initialize_high_frequency_operators(kw)

    def _compute_kw(self) -> float:
        """Compute frequency parameter κw."""
        if not self.particles:
            return 0.0

        positions = np.array([p.position for p in self.particles])
        bbox_size = np.max(positions, axis=0) - np.min(positions, axis=0)
        w = np.max(bbox_size)

        return self.wavenumber * w

    def _initialize_high_frequency_operators(self, kw: float):
        """
        Initialize high-frequency operators based on frequency parameter.

        From Chapter 3, the adaptive order formula is:
        L ≈ κw + (1.8d₀)^(2/3) * (κw)^(1/3)

        where:
        - κw is the frequency parameter
        - d₀ is the number of requested digits of accuracy

        This formula balances accuracy and efficiency for high-frequency problems.
        """
        if self.config.adaptive_order:
            # Chapter 3 formula: L ≈ κw + (1.8d₀)^(2/3) * (κw)^(1/3)
            d0 = self.config.accuracy_digits
            accuracy_term = (1.8 * d0) ** (2/3) * (kw ** (1/3))
            order = max(self.config.min_order,
                       min(int(kw + accuracy_term), self.config.max_order))
        else:
            order = self.config.expansion_order

        # Initialize high-frequency operators
        self.hf_m2l = HighFrequencyM2L(order, self.wavenumber, self.config.dimension)
        self.hf_m2m = HighFrequencyM2M(order, self.wavenumber, self.config.dimension)
        self.hf_l2l = HighFrequencyL2L(order, self.wavenumber, self.config.dimension)

        # Update cell orders
        for cell in self.tree.leaves:
            if cell.cubature_order is None:
                cell.cubature_order = order

    def _upward_pass(self):
        """
        Upward pass with high-frequency operators.

        Uses high-frequency M2M with spherical interpolation.
        """
        if self.hf_m2m is None:
            # Fall back to standard FMM
            super()._upward_pass()
            return

        # Process leaves bottom-up
        for level in range(self.tree.get_max_level(), 0, -1):
            cells = self.tree.get_cells_at_level(level)

            for cell in cells:
                if cell.is_leaf:
                    # P2M: Convert particles to cubature coefficients
                    # For high-frequency FMM, we sample on cubature nodes
                    self._p2m_high_frequency(cell)
                else:
                    # High-frequency M2M with spherical interpolation
                    self._m2m_high_frequency(cell)

    def _p2m_high_frequency(self, cell: Cell):
        """
        High-frequency P2M operator.

        From Chapter 2, Definition 2.2.7:
        P2M[s] · q (λ) = Σ_{y∈s∩Y} e^{-iκ⟨ŷ, λ⟩} q(y)

        where ŷ = y - ctr(s) is the position relative to cell center.

        Note: Uses NEGATIVE phase factor exp(-iκ⟨y-c, λ⟩).
        This is the adjoint of L2P which uses POSITIVE phase factor.
        """
        if cell.cubature_order is None:
            cell.cubature_order = self.config.expansion_order

        cubature = Cubature(cell.cubature_order)

        # Compute coefficients at cubature nodes
        coefficients = np.zeros(cubature.num_nodes, dtype=np.complex128)

        for q, lam in enumerate(cubature.nodes):
            # Sample contribution from all particles in this cell
            for particle in cell.particles:
                # Phase factor: e^{-iκ⟨y-c, λ⟩} (NEGATIVE exponent per Definition 2.2.7)
                dy = particle.position - cell.center
                phase = np.exp(-1j * self.wavenumber * np.dot(dy, lam))

                # Add charge contribution
                coefficients[q] += particle.charge * phase

        # Store coefficients
        self.hf_expansions[id(cell)] = coefficients

    def _m2m_high_frequency(self, cell: Cell):
        """
        High-frequency M2M operator using spherical interpolation.

        Translates coefficients from child cubature grids to parent grid.
        Uses Chapter 3 adaptive order formula for per-cell order selection.
        """
        if not cell.children:
            return

        # Determine parent order using Chapter 3 formula
        if self.config.adaptive_order:
            kw_parent = self._compute_kw_for_cell(cell)
            d0 = self.config.accuracy_digits
            accuracy_term = (1.8 * d0) ** (2/3) * (kw_parent ** (1/3))
            parent_order = max(self.config.min_order,
                             min(int(kw_parent + accuracy_term), self.config.max_order))
        else:
            parent_order = self.config.expansion_order

        cell.cubature_order = parent_order
        cubature = Cubature(parent_order)
        parent_coefficients = np.zeros(cubature.num_nodes, dtype=np.complex128)

        # Aggregate and translate from children
        for child in cell.children:
            if id(child) in self.hf_expansions:
                child_coefficients = self.hf_expansions[id(child)]
                child_order = child.cubature_order if child.cubature_order else parent_order

                # Translate using spherical interpolation
                translated = self.hf_m2m.apply(
                    child_coefficients,
                    child.center,
                    cell.center,
                    child_order,
                    parent_order
                )

                parent_coefficients += translated

        self.hf_expansions[id(cell)] = parent_coefficients

    def _compute_kw_for_cell(self, cell: Cell) -> float:
        """Compute local frequency parameter for a cell."""
        return self.wavenumber * cell.size

    def _downward_pass(self):
        """
        Downward pass with high-frequency M2L operators.

        Uses diagonal M2L operators for O(Q) complexity.
        """
        if self.hf_m2l is None:
            # Fall back to standard FMM
            super()._downward_pass()
            return

        # Process top-down
        for level in range(1, self.tree.get_max_level() + 1):
            cells = self.tree.get_cells_at_level(level)

            for cell in cells:
                # Initialize local expansion
                cell_order = cell.cubature_order if cell.cubature_order else self.config.expansion_order
                local_coeffs = np.zeros(Cubature(cell_order).num_nodes, dtype=np.complex128)

                # M2L: contributions from well-separated cells
                interaction_list = self.tree.get_interaction_list(cell)
                for source_cell in interaction_list:
                    if id(source_cell) in self.hf_expansions:
                        source_coeffs = self.hf_expansions[id(source_cell)]

                        # Apply diagonal M2L operator
                        contrib = self.hf_m2l.apply(
                            source_coeffs,
                            cell.center,
                            source_cell.center
                        )

                        local_coeffs += contrib

                self.hf_expansions[id(cell)] = local_coeffs

                # L2L: propagate to children
                if not cell.is_leaf and cell.children:
                    for child in cell.children:
                        child_order = child.cubature_order if child.cubature_order else cell_order
                        child_local = self.hf_l2l.apply(
                            local_coeffs,
                            cell.center,
                            child.center,
                            cell_order,
                            child_order
                        )

                        child_id = id(child)
                        if child_id in self.hf_expansions:
                            self.hf_expansions[child_id] += child_local
                        else:
                            self.hf_expansions[child_id] = child_local

    def _direct_pass(self):
        """
        Direct pass with high-frequency L2P.

        Evaluates local expansion at particle positions.
        """
        leaves = self.tree.leaves

        for leaf in leaves:
            # Get adjacent leaves
            neighbors = self.tree.get_near_field_neighbors(leaf)

            # P2P with neighbors
            for neighbor in neighbors + [leaf]:
                self.p2p.apply(neighbor.particles, leaf.particles)

            # L2P: evaluate local expansion at particle positions
            if id(leaf) in self.hf_expansions:
                self._l2p_high_frequency(leaf)

    def _l2p_high_frequency(self, cell: Cell):
        """
        High-frequency L2P operator.

        Evaluates local expansion coefficients at particle positions.

        From Chapter 2, Section 2.2.5:
        L2P is the adjoint of P2M operator.

        Note: Uses POSITIVE phase factor exp(+iκ⟨x-c, λ⟩).
        This is the adjoint of P2M which uses NEGATIVE phase factor exp(-iκ⟨y-c, λ⟩).
        The adjoint relationship ensures: ⟨P2M(q), p⟩ = ⟨q, L2P(p)⟩
        """
        coefficients = self.hf_expansions[id(cell)]
        cell_order = cell.cubature_order if cell.cubature_order else self.config.expansion_order
        cubature = Cubature(cell_order)

        for particle in cell.particles:
            # Interpolate from cubature nodes to particle position
            dx = particle.position - cell.center
            potential = 0.0

            for q, (lam, w) in enumerate(zip(cubature.nodes, cubature.weights)):
                # Phase factor: e^{+iκ⟨x-c, λ⟩} (POSITIVE exponent, adjoint of P2M)
                phase = np.exp(1j * self.wavenumber * np.dot(dx, lam))
                potential += coefficients[q] * phase * w

            particle.potential += potential.real


class KernelIndependentFFT(FMM):
    """
    Kernel-independent FMM with FFT-based M2L operators.

    Uses FFT-based M2L for translation-invariant kernels and
    adaptive method selection for general kernels.
    """

    def __init__(self, particles: List[Particle], kernel_func: Callable,
                 config: Optional[TreeConfig] = None,
                 compression_tolerance: float = 1e-6):
        """
        Initialize kernel-independent FMM with FFT acceleration.

        Args:
            particles: List of particles
            kernel_func: Kernel function
            config: Tree configuration
            compression_tolerance: Tolerance for compression
        """
        if config is None:
            config = TreeConfig()

        super().__init__(particles, kernel_func, config)

        # Use adaptive M2L with FFT support
        self.adaptive_m2l = AdaptiveM2L(
            config.expansion_order,
            config.dimension,
            compression_tolerance
        )

        # Storage for kernel-independent expansions
        self.ki_expansions: Dict[int, KernelIndependentExpansion] = {}

    def _upward_pass(self):
        """Upward pass using kernel-independent P2M."""
        for level in range(self.tree.get_max_level(), 0, -1):
            cells = self.tree.get_cells_at_level(level)

            for cell in cells:
                if cell.is_leaf:
                    # Kernel-independent P2M
                    ki_p2m = KernelIndependentP2M(
                        self.config.expansion_order,
                        self.config.dimension
                    )
                    expansion = ki_p2m.apply(cell, self.kernel_func)
                    self.ki_expansions[id(cell)] = expansion

    def _downward_pass(self):
        """Downward pass using FFT-accelerated M2L."""
        leaves = self.tree.leaves

        for leaf in leaves:
            # Initialize local expansion
            local_exp = KernelIndependentExpansion(
                center=leaf.center,
                size=leaf.size,
                order=self.config.expansion_order,
                dimension=self.config.dimension
            )

            # M2L from interaction list
            interaction_list = self.tree.get_interaction_list(leaf)
            for source_cell in interaction_list:
                if id(source_cell) in self.ki_expansions:
                    source_exp = self.ki_expansions[id(source_cell)]

                    # Use adaptive M2L (automatically chooses FFT/ACA/SVD)
                    source_coeffs = source_exp._coefficients
                    source_bounds = source_exp.bounds
                    target_bounds = local_exp.bounds

                    target_coeffs = self.adaptive_m2l.apply(
                        source_coeffs,
                        self.kernel_func,
                        source_bounds,
                        target_bounds
                    )

                    # Add to local expansion
                    local_exp._coefficients += target_coeffs

            self.ki_expansions[id(leaf)] = local_exp

    def _direct_pass(self):
        """Direct pass with kernel-independent L2P."""
        leaves = self.tree.leaves

        for leaf in leaves:
            # P2P with neighbors
            neighbors = self.tree.get_near_field_neighbors(leaf)
            for neighbor in neighbors + [leaf]:
                self.p2p.apply(neighbor.particles, leaf.particles)

            # L2P: evaluate local expansion at particle positions
            if id(leaf) in self.ki_expansions:
                expansion = self.ki_expansions[id(leaf)]
                positions = np.array([p.position for p in leaf.particles])
                potentials = expansion.evaluate(positions)

                for particle, potential in zip(leaf.particles, potentials):
                    particle.potential += potential


class HybridFMM(FMM):
    """
    Hybrid FMM that automatically selects the best algorithm.

    Selects between:
    - Standard FMM for low-frequency problems
    - High-frequency FMM for moderate κw (1 < κw < 10)
    - Directional FMM for extreme high-frequency (κw >= 10)
    """

    def __init__(self, particles: List[Particle], kernel_func: Callable,
                 wavenumber: float = 0.0, config: Optional[TreeConfig] = None):
        """
        Initialize hybrid FMM.

        Args:
            particles: List of particles
            kernel_func: Kernel function
            wavenumber: Wavenumber (0 for Laplace)
            config: Tree configuration
        """
        if config is None:
            config = TreeConfig()

        self.particles = particles
        self.kernel_func = kernel_func
        self.wavenumber = wavenumber
        self.config = config

        # Determine which FMM variant to use
        self._fmm_variant = self._select_variant()

        # Initialize appropriate FMM
        self._initialize_fmm()

    def _select_variant(self) -> str:
        """Select appropriate FMM variant based on problem parameters."""
        if self.wavenumber == 0:
            return 'standard'

        # Compute κw
        positions = np.array([p.position for p in self.particles])
        bbox_size = np.max(positions, axis=0) - np.min(positions, axis=0)
        w = np.max(bbox_size)
        kw = self.wavenumber * w

        if kw < 1.0:
            return 'standard'
        elif kw < self.config.directional_threshold_kw:
            return 'high_frequency'
        else:
            return 'directional'

    def _initialize_fmm(self):
        """Initialize the selected FMM variant."""
        if self._fmm_variant == 'standard':
            self._fmm = StandardFMM(self.particles, self.kernel_func, self.config)
        elif self._fmm_variant == 'high_frequency':
            self._fmm = HighFrequencyFMM(
                self.particles, self.kernel_func, self.wavenumber, self.config
            )
        elif self._fmm_variant == 'directional':
            self._fmm = AdaptiveDirectionalFMM(
                self.particles, self.kernel_func, self.wavenumber,
                self.config.directional_threshold_kw
            )
        else:
            # Default to standard
            self._fmm = StandardFMM(self.particles, self.kernel_func, self.config)

    def compute(self) -> np.ndarray:
        """Compute potentials using the selected variant."""
        return self._fmm.compute()

    def get_error_estimate(self, reference: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Get error estimate from the selected variant."""
        return self._fmm.get_error_estimate(reference)

    @property
    def variant(self) -> str:
        """Return the FMM variant being used."""
        return self._fmm_variant
