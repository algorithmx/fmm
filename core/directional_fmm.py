"""
Directional FMM Module

Implements directional Fast Multipole Method for extreme high-frequency problems.

From Chapter 4, Section 4.2 and Chapter 7:
For κw >> 10 (extreme high-frequency), the standard FMM becomes inefficient
due to the large number of terms required. Directional FMM decomposes the
problem into directional wedges to achieve O(N log N) complexity.

Key idea:
G(x,y) ≈ e^{iκ⟨x,u⟩} K_u(x,y) e^{-iκ⟨y,u⟩}

where K_u(x,y) is a low-frequency kernel in direction u.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum

from .cell import Cell
from .particle import Particle
from .expansion import Expansion


class DirectionalWedge(Enum):
    """Directional wedges for decomposition."""
    POS_X = (1, 0, 0)
    NEG_X = (-1, 0, 0)
    POS_Y = (0, 1, 0)
    NEG_Y = (0, -1, 0)
    POS_Z = (0, 0, 1)
    NEG_Z = (0, 0, -1)


@dataclass
class DirectionalCell:
    """
    Extended cell with direction sets for directional FMM.

    From Chapter 7, Section 7.2:
    Directional cells store information about which directions
    (wedges) are active for efficient high-frequency computation.
    """
    base_cell: Cell
    direction_set: List[np.ndarray] = field(default_factory=list)
    wedge_expansions: Dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        return self.base_cell.center

    @property
    def size(self) -> float:
        return self.base_cell.size

    @property
    def level(self) -> int:
        return self.base_cell.level

    @property
    def is_leaf(self) -> bool:
        return self.base_cell.is_leaf


class DirectionalExpansion(Expansion):
    """
    Wedge-based expansion for directional FMM.

    From Chapter 7, Section 7.3:
    Instead of storing full spherical harmonics expansion,
    we store directional coefficients for each active wedge.

    For direction u:
    Φ_u(x) = e^{iκ⟨x,u⟩} * K_u(x)

    where K_u(x) is a low-frequency local expansion.
    """

    def __init__(self, center: np.ndarray, direction: np.ndarray,
                 order: int, wavenumber: float, dimension: int = 3):
        """
        Initialize directional expansion.

        Args:
            center: Center of the cell
            direction: Unit vector u for this directional expansion
            order: Expansion order (can be lower than full FMM)
            wavenumber: Helmholtz wavenumber κ
            dimension: Spatial dimension (must be 3)
        """
        super().__init__(order, dimension)
        self.center = np.asarray(center, dtype=np.float64)
        self.direction = direction / np.linalg.norm(direction)
        self.wavenumber = wavenumber

        # Low-frequency expansion coefficients
        # These vary slowly compared to the oscillatory factor
        self._coefficients = np.zeros(order + 1, dtype=np.complex128)

    @property
    def num_coefficients(self) -> int:
        return len(self._coefficients)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate directional expansion at given points.

        Φ_u(x) = e^{iκ⟨x,u⟩} * Σ_n L_n * |x-c|^n

        Args:
            points: Array of points (N x 3)

        Returns:
            Array of potential values
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        result = np.zeros(points.shape[0], dtype=np.complex128)

        # Compute relative positions
        dx = points - self.center

        for i, point in enumerate(points):
            # Oscillatory phase factor: e^{iκ⟨x-c,u⟩}
            phase = np.exp(1j * self.wavenumber * np.dot(dx[i], self.direction))

            # Low-frequency local expansion
            r = np.linalg.norm(dx[i])
            local_sum = 0.0
            for n in range(self.order + 1):
                local_sum += self._coefficients[n] * (r ** n)

            result[i] = phase * local_sum

        return result.real

    def zero(self):
        """Reset all coefficients to zero."""
        self._coefficients.fill(0.0)

    def add(self, other: 'DirectionalExpansion'):
        """Add another directional expansion."""
        if self._coefficients is not None and other._coefficients is not None:
            self._coefficients += other._coefficients

    def get_coefficient(self, n: int) -> complex:
        """Get coefficient at order n."""
        if 0 <= n < len(self._coefficients):
            return self._coefficients[n]
        return 0.0

    def set_coefficient(self, n: int, value: complex):
        """Set coefficient at order n."""
        if 0 <= n < len(self._coefficients):
            self._coefficients[n] = value


class DirectionalFMM:
    """
    Directional Fast Multipole Method for extreme high-frequency problems.

    From Chapter 7:
    When κw >> 10, standard FMM requires too many terms for accuracy.
    Directional FMM achieves O(N log N) complexity by:

    1. Decomposing the problem into directional wedges
    2. Using directional expansions with lower order
    3. Combining results from all directions

    Algorithm:
    - Partition sphere into D directions
    - For each direction u:
        * Apply phase modulation: sources → e^{-iκ⟨y,u⟩} * sources
        * Run standard FMM with modulated sources
        * Apply inverse phase: e^{iκ⟨x,u⟩} * result
    - Combine results from all directions
    """

    def __init__(self, particles: List[Particle], kernel_func: callable,
                 wavenumber: float, config=None):
        """
        Initialize directional FMM.

        Args:
            particles: List of particles
            kernel_func: Helmholtz kernel function
            wavenumber: Wavenumber κ
            config: Optional configuration (if None, uses defaults)
        """
        self.particles = particles
        self.kernel_func = kernel_func
        self.wavenumber = wavenumber

        # Determine number of directions based on κw
        # More directions needed for higher frequency
        kw = self._compute_kw()
        self.num_directions = self._compute_num_directions(kw)

        # Generate directions (uniform on sphere)
        self.directions = self._generate_directions(self.num_directions)

        # Storage for directional expansions
        self.directional_expansions: Dict[int, Dict[int, DirectionalExpansion]] = {}

    def _compute_kw(self) -> float:
        """Compute κw product (frequency parameter)."""
        if not self.particles:
            return 0.0

        positions = np.array([p.position for p in self.particles])
        bbox_size = np.max(positions, axis=0) - np.min(positions, axis=0)
        w = np.max(bbox_size)

        return self.wavenumber * w

    def _compute_num_directions(self, kw: float) -> int:
        """
        Compute required number of directions based on κw.

        From Chapter 7: D ≈ (κw)^{(d-1)/d} for d dimensions
        For 3D: D ≈ (κw)^{2/3}
        """
        if kw < 10:
            return 6  # Minimum (coordinate directions)
        else:
            # Scale with frequency
            return int(6 * (kw / 10) ** (2/3))

    def _generate_directions(self, num_directions: int) -> List[np.ndarray]:
        """
        Generate uniformly distributed directions on sphere.

        Uses Fibonacci sphere algorithm for uniform distribution.

        Args:
            num_directions: Number of directions to generate

        Returns:
            List of unit direction vectors
        """
        directions = []

        # Golden angle
        phi = np.pi * (3 - np.sqrt(5))

        for i in range(num_directions):
            y = 1 - (i / float(num_directions - 1)) * 2
            radius = np.sqrt(1 - y * y)

            theta = phi * i

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            direction = np.array([x, y, z])
            directions.append(direction / np.linalg.norm(direction))

        return directions

    def _modulate_charges(self, direction: np.ndarray) -> np.ndarray:
        """
        Apply phase modulation to source charges.

        q_u(y) = q(y) * e^{-iκ⟨y,u⟩}

        Args:
            direction: Direction vector u

        Returns:
            Modulated charges
        """
        modulated = np.zeros(len(self.particles), dtype=np.complex128)

        for i, particle in enumerate(self.particles):
            phase = np.exp(-1j * self.wavenumber * np.dot(particle.position, direction))
            modulated[i] = particle.charge * phase

        return modulated

    def _demodulate_potentials(self, potentials: np.ndarray,
                              points: np.ndarray,
                              direction: np.ndarray) -> np.ndarray:
        """
        Apply inverse phase modulation to potentials.

        Φ(x) = e^{iκ⟨x,u⟩} * Φ_u(x)

        Args:
            potentials: Modulated potentials
            points: Target points
            direction: Direction vector u

        Returns:
            Demodulated potentials
        """
        demodulated = np.zeros_like(potentials, dtype=np.complex128)

        for i, point in enumerate(points):
            phase = np.exp(1j * self.wavenumber * np.dot(point, direction))
            demodulated[i] = phase * potentials[i]

        return demodulated

    def compute(self) -> np.ndarray:
        """
        Compute potentials using directional FMM.

        Returns:
            Array of potentials for each particle

        Algorithm:
        1. For each direction u:
           a. Modulate source charges: q_u = q * e^{-iκ⟨y,u⟩}
           b. Run standard FMM with modulated charges
           c. Demodulate result: Φ = e^{iκ⟨x,u⟩} * Φ_u
        2. Combine results from all directions
        """
        total_potential = np.zeros(len(self.particles), dtype=np.complex128)

        # For each direction
        for direction in self.directions:
            # Modulate charges
            modulated_charges = self._modulate_charges(direction)

            # Run standard FMM with modulated charges
            # (In practice, this would call StandardFMM with modified particles)
            directional_potential = self._run_standard_fmm(
                modulated_charges, direction
            )

            # Demodulate and accumulate
            points = np.array([p.position for p in self.particles])
            demodulated = self._demodulate_potentials(
                directional_potential, points, direction
            )

            total_potential += demodulated

        # Take real part (physical potential)
        return total_potential.real

    def _run_standard_fmm(self, charges: np.ndarray,
                         direction: np.ndarray) -> np.ndarray:
        """
        Run standard FMM with directionally modulated charges.

        This is a placeholder - in practice, this would integrate
        with the existing StandardFMM class.

        Args:
            charges: Modulated charges
            direction: Direction vector

        Returns:
            Potentials from this direction
        """
        # Placeholder: Direct computation for now
        # In practice, would call StandardFMM with modified particles
        potentials = np.zeros(len(charges), dtype=np.complex128)

        positions = np.array([p.position for p in self.particles])

        for i, (pos_i, q_i) in enumerate(zip(positions, charges)):
            for j, (pos_j, _) in enumerate(zip(positions, charges)):
                if i != j:
                    value = self.kernel_func(pos_j, pos_i)
                    potentials[i] += q_j * value

        return potentials

    def get_complexity(self) -> Dict[str, float]:
        """
        Get theoretical complexity analysis.

        Returns:
            Dictionary with complexity metrics
        """
        kw = self._compute_kw()
        N = len(self.particles)

        # Directional FMM complexity: O(N log N) with D directions
        directional_ops = N * np.log(N) * self.num_directions

        # Standard FMM complexity at high frequency: O(N (κw)^2)
        standard_ops = N * (kw ** 2)

        # Speedup factor
        speedup = standard_ops / directional_ops if directional_ops > 0 else float('inf')

        return {
            'kw': kw,
            'num_directions': self.num_directions,
            'directional_ops': directional_ops,
            'standard_ops': standard_ops,
            'speedup': speedup
        }


class AdaptiveDirectionalFMM(DirectionalFMM):
    """
    Adaptive directional FMM that automatically switches between
    standard and directional based on κw threshold.

    From Chapter 7:
    - Use standard FMM for κw < 10
    - Use directional FMM for κw >= 10
    """

    def __init__(self, particles: List[Particle], kernel_func: callable,
                 wavenumber: float, threshold_kw: float = 10.0):
        """
        Initialize adaptive directional FMM.

        Args:
            particles: List of particles
            kernel_func: Helmholtz kernel function
            wavenumber: Wavenumber κ
            threshold_kw: Threshold for switching to directional mode
        """
        super().__init__(particles, kernel_func, wavenumber)
        self.threshold_kw = threshold_kw
        self.use_directional = self._compute_kw() >= threshold_kw

    def compute(self) -> np.ndarray:
        """
        Compute potentials using adaptive method.

        Returns:
            Array of potentials for each particle
        """
        if not self.use_directional:
            # Use standard FMM
            return self._run_standard_fmm_full()
        else:
            # Use directional FMM
            return super().compute()

    def _run_standard_fmm_full(self) -> np.ndarray:
        """Run standard FMM without directional decomposition."""
        charges = np.array([p.charge for p in self.particles])
        positions = np.array([p.position for p in self.particles])

        potentials = np.zeros(len(charges), dtype=np.complex128)

        for i, (pos_i, q_i) in enumerate(zip(positions, charges)):
            for j, (pos_j, q_j) in enumerate(zip(positions, charges)):
                if i != j:
                    value = self.kernel_func(pos_j, pos_i)
                    potentials[i] += q_j * value

        return potentials.real
