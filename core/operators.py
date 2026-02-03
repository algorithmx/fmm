"""
Operators Module

Implements the six FMM operators: P2M, M2M, M2L, L2L, L2P, P2P.
Also includes M2P and P2L operators for adaptive FMM.
Includes high-frequency FMM operators for Helmholtz kernel.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
import numpy as np
import math
from functools import lru_cache

from .cell import Cell
from .particle import Particle
from .expansion import MultipoleExpansion, LocalExpansion


# ============================================================================
# Gant Coefficient Implementation
# ============================================================================

def _wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """
    Compute Wigner 3-j symbol using Racah formula.

    From Chapter 2, the Wigner 3-j symbol is defined as:
    (j1 j2 j3; m1 m2 m3) with selection rules:
    - Triangle condition: |j1-j2| <= j3 <= j1+j2
    - Sum rule: m1 + m2 + m3 = 0
    - Individual bounds: |mi| <= ji

    Args:
        j1, j2, j3: Angular momentum quantum numbers (degrees)
        m1, m2, m3: Magnetic quantum numbers (orders)

    Returns:
        Value of the Wigner 3-j symbol (0 if selection rules not satisfied)
    """
    # Selection rules
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0

    # Use sympy for accurate computation if available
    try:
        from sympy.physics.wigner import wigner_3j
        result = float(wigner_3j(j1, j2, j3, m1, m2, m3))
        return result
    except ImportError:
        # Fallback: implement via direct integration of spherical harmonics
        return _wigner_3j_integral(j1, j2, j3, m1, m2, m3)


def _wigner_3j_integral(j1: int, j2: int, j3: int,
                        m1: int, m2: int, m3: int) -> float:
    """
    Compute Wigner 3-j symbol via direct integration of spherical harmonics.

    Uses the relation:
    ∫ Y_{j1,m1} Y_{j2,m2} Y_{j3,m3}^* dΩ =
        sqrt((2j1+1)(2j2+1)(2j3+1)/(4π)) * (j1 j2 j3; 0 0 0) * (j1 j2 j3; m1 m2 m3)

    This is a fallback when sympy is not available.
    """
    from scipy.special import sph_harm
    from scipy.integrate import simpson

    # Simple numerical integration over sphere
    n_theta = 50
    n_phi = 100

    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

    # Compute spherical harmonics product
    Y1 = sph_harm(m1, j1, Phi, Theta)
    Y2 = sph_harm(m2, j2, Phi, Theta)
    Y3_conj = np.conj(sph_harm(m3, j3, Phi, Theta))

    integrand = Y1 * Y2 * Y3_conj * np.sin(Theta)

    # 2D Simpson integration
    integral = simpson(simpson(integrand, phi, axis=1), theta, axis=0)

    # Convert to Wigner 3-j symbol
    # The integral is related to Gaunt coefficient, which relates to 3-j symbol
    return integral


def compute_gaunt_coefficient(l1: int, l2: int, l3: int,
                             m1: int, m2: int, m3: int) -> complex:
    """
    Compute Gaunt coefficient G(l1,l2,l3; m1,m2,m3).

    From Chapter 2, the Gaunt coefficient is:
    G(l1,l2,l3; m1,m2,m3) = ∫ Y_{l1,m1} Y_{l2,m2} Y_{l3,m3}^* dΩ

    Relation to Wigner 3-j symbols:
    G = sqrt((2l1+1)(2l2+1)(2l3+1)/(4π)) * (-1)^m3 * (l1 l2 l3; m1 m2 -m3) * (l1 l2 l3; 0 0 0)

    Selection rules:
    - Triangle: |l1-l2| <= l3 <= l1+l2
    - m1 + m2 + m3 = 0
    - |mi| <= li

    Args:
        l1, l2, l3: Degrees of spherical harmonics
        m1, m2, m3: Orders of spherical harmonics

    Returns:
        Complex Gaunt coefficient value
    """
    # Selection rule: m1 + m2 + m3 = 0
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
        return 0.0
    if l3 < abs(l1 - l2) or l3 > l1 + l2:
        return 0.0

    # Check parity: l1 + l2 + l3 must be even (from (l1 l2 l3; 0 0 0) selection rule)
    if (l1 + l2 + l3) % 2 != 0:
        return 0.0

    try:
        from sympy.physics.wigner import wigner_3j

        # Compute Wigner 3-j symbols
        wigner_m = wigner_3j(l1, l2, l3, m1, m2, -m3)
        wigner_0 = wigner_3j(l1, l2, l3, 0, 0, 0)

        # Convert to Gaunt coefficient
        prefactor = np.sqrt((2*l1 + 1) * (2*l2 + 1) * (2*l3 + 1) / (4*np.pi))
        phase = (-1) ** m3

        return complex(prefactor * phase * float(wigner_m * wigner_0))
    except ImportError:
        # Fallback to numerical integration
        from scipy.special import sph_harm
        from scipy.integrate import simpson

        n_theta, n_phi = 60, 120
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        Theta, Phi = np.meshgrid(theta, phi, indexing='ij')

        Y1 = sph_harm(m1, l1, Phi, Theta)
        Y2 = sph_harm(m2, l2, Phi, Theta)
        Y3_conj = np.conj(sph_harm(m3, l3, Phi, Theta))

        integrand = Y1 * Y2 * Y3_conj * np.sin(Theta)
        integral = simpson(simpson(integrand, phi, axis=1), theta, axis=0)

        return complex(integral)


class Operator(ABC):
    """
    Abstract base class for FMM operators.

    All operators implement a common interface for applying
    the operation to cells or particles.
    """

    def __init__(self, order: int, dimension: int = 2):
        """
        Initialize the operator.

        Args:
            order: Truncation order for expansions
            dimension: Spatial dimension (2 or 3)
        """
        self.order = order
        self.dimension = dimension

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Apply the operator."""
        pass


class P2M(Operator):
    """
    Particles-to-Multipole operator.

    Implements Chapter 2, Definition 2.2.2 for 3D Laplace kernel:
    M_{n,m} = q * Y_{n,m}^*(θ,φ) * r^n

    For 2D: M_n = q * r^n * e^{-inθ}

    Converts particle charges into a multipole expansion at the leaf cell level.
    """

    def __init__(self, order: int = 4, dimension: int = 2):
        super().__init__(order, dimension)

    def apply(self, cell: Cell, kernel_func: callable) -> MultipoleExpansion:
        """
        Apply P2M operator to create multipole expansion from particles.

        Args:
            cell: Leaf cell containing particles
            kernel_func: The kernel function G(x, y)

        Returns:
            Multipole expansion for the cell
        """
        if not cell.is_leaf:
            raise ValueError("P2M can only be applied to leaf cells")

        expansion = MultipoleExpansion(
            center=cell.center,
            order=self.order,
            dimension=self.dimension
        )

        # Compute multipole coefficients from particles
        for particle in cell.particles:
            self._add_particle_contribution(expansion, particle, kernel_func)

        return expansion

        # Complexity: O(Np²) where N = particles, p = order

    def _add_particle_contribution(self, expansion: MultipoleExpansion,
                                   particle: Particle, kernel_func: callable):
        """
        Add a single particle's contribution to the multipole expansion.

        For 2D Laplace kernel (Chapter 2, Eq. 2.12):
        G(x,y) = -1/(2π) * log|x-y| = Re[∑_{n=0}^∞ M_n / (z-z0)^(n+1)]
        where M_n = q * (z - z0)^n (no factorial for standard Laplace)

        For 3D Laplace kernel (Chapter 2, Eq. 2.12-2.13):
        1/|x-y| = ∑_{n=0}^∞ ∑_{m=-n}^n M_{n,m} * Y_{n,m}(θ,φ) / r^{n+1}

        where r, theta are polar/spherical coordinates relative to expansion center.
        """
        dx = particle.position - expansion.center
        r = np.linalg.norm(dx)

        if r < 1e-14:
            return  # Skip self-contribution

        if self.dimension == 2:
            # 2D: Use complex representation
            # For Laplace: M_n = q * z^n (standard formulation, no factorial)
            theta = np.arctan2(dx[1], dx[0])
            z = r * np.exp(1j * theta)

            for n in range(self.order + 1):
                coeff = expansion.get_coefficient(n)
                # Standard 2D Laplace multipole coefficient (Chapter 2, Eq. 2.12)
                new_coeff = coeff + particle.charge * (z ** n)
                expansion.set_coefficient(n, new_coeff)
        else:
            # 3D: Use spherical harmonics with proper normalization
            theta = np.arccos(np.clip(dx[2] / (r + 1e-14), -1, 1))
            phi = np.arctan2(dx[1], dx[0])

            for n in range(self.order + 1):
                for m in range(-n, n + 1):
                    coeff = expansion.get_coefficient(n, m)
                    Y_nm = expansion._spherical_harmonic(n, m, theta, phi)
                    # M_{n,m} = q * Y_{n,m}^* * r^n (Chapter 2, Eq. 2.13)
                    # Using complex conjugate for proper inner product
                    new_coeff = coeff + particle.charge * np.conj(Y_nm) * (r ** n)
                    expansion.set_coefficient(n, new_coeff, m)


class M2M(Operator):
    """
    Multipole-to-Multipole translation operator.

    Implements Chapter 2, Definition 2.2.3 for 3D:
    Uses Gaunt coefficients for translation between multipole expansions.

    Complexity: O(p⁴) per operation where p = expansion order

    Translates multipole expansions from child cells to parent cells.
    """

    def __init__(self, order: int = 4, dimension: int = 2):
        super().__init__(order, dimension)
        # Precompute Gaunt coefficients for 3D
        if dimension == 3:
            self._gaunt_cache = self._precompute_gaunt_coefficients()

    def apply(self, source_center: np.ndarray, target_center: np.ndarray,
              source_expansion: MultipoleExpansion) -> MultipoleExpansion:
        """
        Apply M2M operator to translate multipole expansion.

        Args:
            source_center: Center of source cell
            target_center: Center of target (parent) cell
            source_expansion: Multipole expansion from child

        Returns:
            Translated multipole expansion at parent level
        """
        target_expansion = MultipoleExpansion(
            center=target_center,
            order=self.order,
            dimension=self.dimension
        )

        # Translation vector
        dx = source_center - target_center
        r = np.linalg.norm(dx)

        if self.dimension == 2:
            theta = np.arctan2(dx[1], dx[0])
            z = r * np.exp(1j * theta)

            # Translate each coefficient
            for n in range(self.order + 1):
                translated_coeff = 0j
                for k in range(n + 1):
                    source_coeff = source_expansion.get_coefficient(k)
                    # Binomial coefficient for translation
                    binom = math.factorial(n) / (
                        math.factorial(k) * math.factorial(n - k)
                    )
                    translated_coeff += binom * source_coeff * (z ** (n - k))

                target_expansion.set_coefficient(n, translated_coeff)
        else:
            # 3D translation using spherical harmonics addition theorem
            theta = np.arccos(np.clip(dx[2] / (r + 1e-14), -1, 1))
            phi = np.arctan2(dx[1], dx[0])

            for n in range(self.order + 1):
                for m in range(-n, n + 1):
                    translated_coeff = 0j
                    for k in range(self.order + 1):
                        for l in range(-k, k + 1):
                            source_coeff = source_expansion.get_coefficient(k, l)
                            # Translation coefficient using Gaunt coefficients
                            coeff = self._translation_coefficient(n, m, k, l, r, theta, phi)
                            translated_coeff += source_coeff * coeff

                    target_expansion.set_coefficient(n, translated_coeff, m)

        return target_expansion

        # Complexity: O(p⁴) for 3D (nested loops over k,l,n,m with Gaunt coupling)

    def _translation_coefficient(self, n: int, m: int, k: int, l: int,
                                 r: float, theta: float, phi: float) -> complex:
        """
        Compute translation coefficient for 3D M2M using Gaunt coefficients.

        From Chapter 2, the M2M translation coefficient involves:
        T_{k,l;n,m} = r^{n-k} * Y_{n-k,m-l}(θ,φ) * G(k, n-k, n, l, m-l, m)

        for n >= k, combining child multipole (k,l) to parent (n,m).
        """
        if n < k:
            return 0.0

        coupled_degree = n - k
        coupled_order = m - l

        # Selection rule: |m-l| <= n-k
        if abs(coupled_order) > coupled_degree:
            return 0.0

        # Distance factor: r^(n-k)
        dist_factor = r ** coupled_degree

        # Spherical harmonic for translation direction
        from scipy.special import sph_harm
        Y_coupled = sph_harm(coupled_order, coupled_degree, phi, theta)

        # Gaunt coefficient for coupling
        gaunt_key = (k, coupled_degree, n, l, coupled_order, m)
        gaunt_coeff = self._gaunt_cache.get(gaunt_key, 0.0)

        return dist_factor * gaunt_coeff * Y_coupled

    def _precompute_gaunt_coefficients(self) -> dict:
        """
        Precompute Gaunt coefficients for M2M translation.

        For M2M, we need coefficients of the form:
        G(k, n-k, n, l, m-l, m)

        Returns:
            Dictionary with precomputed Gaunt coefficients
        """
        gaunt_cache = {}

        for k in range(self.order + 1):
            for n in range(k, self.order + 1):  # n >= k for M2M
                l2 = n - k  # Middle degree
                l3 = n      # Total degree

                for l in range(-k, k + 1):
                    for m in range(-n, n + 1):
                        m2 = m - l  # Middle order

                        if abs(m2) > l2:
                            continue

                        # Parity rule
                        if (k + l2 + l3) % 2 != 0:
                            continue

                        key = (k, l2, l3, l, m2, m)
                        gaunt_cache[key] = compute_gaunt_coefficient(k, l2, l3, l, m2, m)

        return gaunt_cache


class M2L(Operator):
    """
    Multipole-to-Local conversion operator.

    3D: Chapter 2, Definition 2.2.4 using outgoing solid harmonics
    2D: Standard complex analysis formulation

    From Chapter 2, Section 2.2.4.3:
    - 2D: L_n = ∑_{k=0}^p M_k / z^(k+n+1) * (-1)^k / (k+n)!
    - 3D: Uses Legendre/Gegenbauer coupling with Gaunt coefficients

    Complexity: O(p⁴) for 3D, O(p²) for 2D

    Converts source multipole expansions to target local expansions.
    This is the key operator for FMM's efficiency.
    """

    def __init__(self, order: int = 4, dimension: int = 2):
        super().__init__(order, dimension)
        # Precompute and cache Gaunt coefficients for 3D
        if dimension == 3:
            self._gaunt_cache = self._precompute_gaunt_coefficients()
            self._spherical_harmonic_cache = {}

    def apply(self, source_expansion: MultipoleExpansion,
              target_center: np.ndarray) -> LocalExpansion:
        """
        Apply M2L operator to convert multipole to local expansion.

        Args:
            source_expansion: Multipole expansion from source cell
            target_center: Center of target cell

        Returns:
            Local expansion at target cell
        """
        target_expansion = LocalExpansion(
            center=target_center,
            order=self.order,
            dimension=self.dimension
        )

        # Translation vector
        dx = target_center - source_expansion.center
        r = np.linalg.norm(dx)

        if r < 1e-14:
            # Same cell, no contribution
            return target_expansion

        if self.dimension == 2:
            self._apply_2d(source_expansion, target_expansion, dx, r)
        else:
            self._apply_3d(source_expansion, target_expansion, dx, r, theta_phi=(dx, r))

        return target_expansion

    def _apply_2d(self, source_expansion: MultipoleExpansion,
                  target_expansion: LocalExpansion, dx: np.ndarray, r: float):
        """
        Apply 2D M2L translation.

        From Chapter 2, Eq. 2.12:
        L_n = ∑_{k=0}^p M_k / z^(k+n+1) * (-1)^k / k!

        where z is the complex translation vector.
        """
        theta = np.arctan2(dx[1], dx[0])
        z = r * np.exp(1j * theta)

        # M2L translation using multipole-to-local conversion
        for n in range(self.order + 1):
            local_coeff = 0j
            for k in range(self.order + 1):
                source_coeff = source_expansion.get_coefficient(k)

                # M2L coefficient: (-1)^k / z^(k+n+1)
                # Note: The factorial in denominator is for Laplace kernels
                coeff = source_coeff * ((-1) ** k) / (z ** (k + n + 1))
                local_coeff += coeff

            target_expansion.set_coefficient(n, local_coeff)

        # Complexity: O(p²) for 2D (double sum over k,n)

    def _apply_3d(self, source_expansion: MultipoleExpansion,
                  target_expansion: LocalExpansion, dx: np.ndarray, r: float,
                  theta_phi=None):
        """
        Apply 3D M2L translation using proper spherical harmonics coupling.

        From Chapter 2, Eq. 2.13:
        L_{n,m} = ∑_{k=0}^p ∑_{l=-k}^k M_{k,l} * T_{k+l,n+m}(y_c - x_c)

        where T involves Gaunt coefficients and spherical harmonics.
        """
        theta = np.arccos(np.clip(dx[2] / (r + 1e-14), -1, 1))
        phi = np.arctan2(dx[1], dx[0])

        for n in range(self.order + 1):
            for m in range(-n, n + 1):
                local_coeff = 0j
                for k in range(self.order + 1):
                    for l in range(-k, k + 1):
                        source_coeff = source_expansion.get_coefficient(k, l)

                        # Compute M2L translation coefficient using Gaunt coefficients
                        # T_{k,l;n,m} involves coupling between (k,l) and (n,m)
                        coeff = self._m2l_coefficient_3d(k, l, n, m, r, theta, phi)
                        local_coeff += source_coeff * coeff

                target_expansion.set_coefficient(n, local_coeff, m)

        # Complexity: O(p⁴) for 3D (quadruple sum with Gaunt coupling)

    def _m2l_coefficient_3d(self, k: int, l: int, n: int, m: int,
                            r: float, theta: float, phi: float) -> complex:
        """
        Compute 3D M2L translation coefficient using proper Gaunt coefficients.

        From Chapter 2, Section 2.2.4.3:
        T_{k,l;n,m} = (-1)^{k+n} / r^{k+n+1} * Y_{k+n,l+m}(θ,φ) * G(k,n,l,m)

        where G is the Gaunt coefficient (integral of three spherical harmonics):
        G(l1,l2,l3; m1,m2,m3) = ∫ Y_{l1,m1} Y_{l2,m2} Y_{l3,m3}^* dΩ

        Args:
            k: Degree of source multipole coefficient
            l: Order of source multipole coefficient
            n: Degree of target local coefficient
            m: Order of target local coefficient
            r: Distance between cell centers
            theta: Polar angle of translation vector
            phi: Azimuthal angle of translation vector

        Returns:
            Complex M2L translation coefficient
        """
        from scipy.special import sph_harm

        # Coupled degree and order for the translation
        coupled_degree = k + n
        coupled_order = l + m

        # Selection rule: |l+m| <= k+n
        if abs(coupled_order) > coupled_degree:
            return 0.0

        # Distance factor: (-1)^(k+n) / r^(k+n+1)
        dist_factor = ((-1) ** (k + n)) / (r ** (k + n + 1))

        # Spherical harmonic for the translation direction
        Y_coupled = self._get_spherical_harmonic(coupled_degree, coupled_order, theta, phi)

        # Gaunt coefficient G(k, n, l, m) = G(k, n, k+n, l, m, l+m)
        # This couples (k,l) multipole with (n,m) local through Y_{k+n, l+m}
        gaunt_key = (k, n, coupled_degree, l, m, coupled_order)
        gaunt_coeff = self._gaunt_cache.get(gaunt_key, 0.0)

        return dist_factor * gaunt_coeff * Y_coupled

    def _get_spherical_harmonic(self, n: int, m: int, theta: float, phi: float) -> complex:
        """Get or compute spherical harmonic with caching."""
        key = (n, m, theta, phi)
        if key not in self._spherical_harmonic_cache:
            from scipy.special import sph_harm
            self._spherical_harmonic_cache[key] = sph_harm(m, n, phi, theta)
        return self._spherical_harmonic_cache[key]

    def _precompute_gaunt_coefficients(self) -> dict:
        """
        Precompute Gaunt coefficients for efficient M2L translation.

        From Chapter 2, the Gaunt coefficient is:
        G(l1,l2,l3; m1,m2,m3) = ∫ Y_{l1,m1} Y_{l2,m2} Y_{l3,m3}^* dΩ

        This precomputes all Gaunt coefficients needed for M2L translation
        up to the expansion order.

        Returns:
            Dictionary with keys (l1, l2, l3, m1, m2, m3) and complex values
        """
        gaunt_cache = {}

        # Precompute for all combinations needed for M2L translation
        # For M2L, we need: G(k, n, k+n, l, m, l+m)
        # where k, n range from 0 to order, and l = -k..k, m = -n..n
        for k in range(self.order + 1):
            for n in range(self.order + 1):
                l3 = k + n  # Coupled degree
                if l3 > 2 * self.order:  # Limit coupling degree
                    continue

                for l in range(-k, k + 1):
                    for m in range(-n, n + 1):
                        l3_m = l + m  # Coupled order

                        # Selection rule: |l+m| <= k+n
                        if abs(l3_m) > l3:
                            continue

                        # Parity rule: k + n + (k+n) must be even
                        if (k + n + l3) % 2 != 0:
                            continue

                        key = (k, n, l3, l, m, l3_m)
                        gaunt_cache[key] = compute_gaunt_coefficient(k, n, l3, l, m, l3_m)

        return gaunt_cache


class L2L(Operator):
    """
    Local-to-Local translation operator.

    Implements Chapter 2, Definition 2.2.5 for 3D:
    Translates local expansion from parent to child cell.

    Complexity: O(p⁴) per operation where p = expansion order

    Translates local expansions from parent cells to child cells.
    """

    def __init__(self, order: int = 4, dimension: int = 2):
        super().__init__(order, dimension)
        # Precompute Gaunt coefficients for 3D
        if dimension == 3:
            self._gaunt_cache = self._precompute_gaunt_coefficients()

    def apply(self, source_center: np.ndarray, target_center: np.ndarray,
              source_expansion: LocalExpansion) -> LocalExpansion:
        """
        Apply L2L operator to translate local expansion.

        Args:
            source_center: Center of source (parent) cell
            target_center: Center of target (child) cell
            source_expansion: Local expansion from parent

        Returns:
            Translated local expansion at child level
        """
        target_expansion = LocalExpansion(
            center=target_center,
            order=self.order,
            dimension=self.dimension
        )

        # Translation vector
        dx = target_center - source_center
        r = np.linalg.norm(dx)

        if self.dimension == 2:
            theta = np.arctan2(dx[1], dx[0])
            z = r * np.exp(1j * theta)

            # Translate each coefficient
            for n in range(self.order + 1):
                translated_coeff = 0j
                for k in range(n, self.order + 1):
                    source_coeff = source_expansion.get_coefficient(k)
                    # Translation coefficient
                    binom = math.factorial(k) / (
                        math.factorial(n) * math.factorial(k - n)
                    )
                    translated_coeff += binom * source_coeff * (z ** (k - n))

                target_expansion.set_coefficient(n, translated_coeff)
        else:
            # 3D translation
            theta = np.arccos(np.clip(dx[2] / (r + 1e-14), -1, 1))
            phi = np.arctan2(dx[1], dx[0])

            for n in range(self.order + 1):
                for m in range(-n, n + 1):
                    translated_coeff = 0j
                    for k in range(n, self.order + 1):
                        for l in range(-k, k + 1):
                            source_coeff = source_expansion.get_coefficient(k, l)
                            coeff = self._translation_coefficient(k, l, n, m, r, theta, phi)
                            translated_coeff += source_coeff * coeff

                    target_expansion.set_coefficient(n, translated_coeff, m)

        return target_expansion

        # Complexity: O(p⁴) for 3D

    def _translation_coefficient(self, k: int, l: int, n: int, m: int,
                                 r: float, theta: float, phi: float) -> complex:
        """
        Compute translation coefficient for 3D L2L using Gaunt coefficients.

        From Chapter 2, the L2L translation coefficient involves:
        T_{k,l;n,m} = r^{k-n} * Y_{k-n,l-m}(θ,φ) * G(n, k-n, k, m, l-m, l)

        for k >= n, combining parent local (k,l) to child (n,m).
        """
        if k < n:
            return 0.0

        coupled_degree = k - n
        coupled_order = l - m

        # Selection rule: |l-m| <= k-n
        if abs(coupled_order) > coupled_degree:
            return 0.0

        # Distance factor: r^(k-n)
        dist_factor = r ** coupled_degree

        # Spherical harmonic for translation direction
        from scipy.special import sph_harm
        Y_coupled = sph_harm(coupled_order, coupled_degree, phi, theta)

        # Gaunt coefficient for coupling
        gaunt_key = (n, coupled_degree, k, m, coupled_order, l)
        gaunt_coeff = self._gaunt_cache.get(gaunt_key, 0.0)

        return dist_factor * gaunt_coeff * Y_coupled

    def _precompute_gaunt_coefficients(self) -> dict:
        """
        Precompute Gaunt coefficients for L2L translation.

        For L2L, we need coefficients of the form:
        G(n, k-n, k, m, l-m, l)

        Returns:
            Dictionary with precomputed Gaunt coefficients
        """
        gaunt_cache = {}

        for n in range(self.order + 1):
            for k in range(n, self.order + 1):  # k >= n for L2L
                l2 = k - n  # Middle degree
                l3 = k      # Total degree

                for m in range(-n, n + 1):
                    for l in range(-k, k + 1):
                        m2 = l - m  # Middle order

                        if abs(m2) > l2:
                            continue

                        # Parity rule
                        if (n + l2 + l3) % 2 != 0:
                            continue

                        key = (n, l2, l3, m, m2, l)
                        gaunt_cache[key] = compute_gaunt_coefficient(n, l2, l3, m, m2, l)

        return gaunt_cache


class L2P(Operator):
    """
    Local-to-Particles operator.

    Implements Chapter 2, Definition 2.2.6 for 3D:
    Φ(x) = Σ I_{n,m}(x-c) * L_{n,m} using regular solid harmonics

    Complexity: O(Np²) where N = number of target particles

    Converts local expansion to particle potentials.
    """

    def __init__(self, order: int = 4, dimension: int = 2):
        super().__init__(order, dimension)

    def apply(self, expansion: LocalExpansion, particles: List[Particle]):
        """
        Apply L2P operator to compute particle potentials.

        Args:
            expansion: Local expansion to evaluate
            particles: Target particles to add potentials to
        """
        positions = np.array([p.position for p in particles])
        potentials = expansion.evaluate(positions)

        for particle, potential in zip(particles, potentials):
            particle.potential += potential

        # Complexity: O(Np²) where N = target particles


class P2P(Operator):
    """
    Particles-to-Particles operator.

    Direct computation of near-field interactions.

    Complexity: O(N²) where N = near-field particles
    """

    def __init__(self, kernel_func: callable, dimension: int = 2):
        super().__init__(order=0, dimension=dimension)
        self.kernel_func = kernel_func

    def apply(self, source_particles: List[Particle],
              target_particles: List[Particle]):
        """
        Apply P2P operator for direct near-field computation.

        Args:
            source_particles: Source particles
            target_particles: Target particles
        """
        for source in source_particles:
            for target in target_particles:
                if source is not target:
                    # Compute kernel contribution
                    value = self.kernel_func(source.position, target.position)
                    target.potential += source.charge * value

        # Complexity: O(N²) where N = near-field particles


class M2P(Operator):
    """
    Multipole-to-Particles operator.

    Directly evaluates multipole expansion at particle positions.

    From Chapter 2, Section 2.2.4.2:
    M2P is used when a leaf cell is well-separated from a source cell.
    This bypasses the intermediate local expansion step.
    """

    def __init__(self, order: int = 4, dimension: int = 2):
        super().__init__(order, dimension)

    def apply(self, multipole_expansion: MultipoleExpansion, particles: List[Particle]):
        """
        Apply M2P operator to compute particle potentials from multipole expansion.

        Args:
            multipole_expansion: Source multipole expansion
            particles: Target particles to add potentials to
        """
        positions = np.array([p.position for p in particles])
        potentials = multipole_expansion.evaluate(positions)

        for particle, potential in zip(particles, potentials):
            particle.potential += potential


class P2L(Operator):
    """
    Particles-to-Local operator.

    Converts particle charges directly to local expansion coefficients.

    From Chapter 2, Section 2.2.4.2:
    P2L is used when particles are well-separated from the target cell center.
    This is the adjoint of L2P operator.
    """

    def __init__(self, order: int = 4, dimension: int = 2):
        super().__init__(order, dimension)

    def apply(self, particles: List[Particle], local_expansion: LocalExpansion,
              kernel_func: callable = None):
        """
        Apply P2L operator to create local expansion from particles.

        This is essentially the transpose of L2P:
        Instead of evaluating expansion at particles, we accumulate
        particle contributions to expansion coefficients.

        Args:
            particles: Source particles
            local_expansion: Target local expansion to add to
            kernel_func: Optional kernel function (for non-standard kernels)
        """
        for particle in particles:
            dx = particle.position - local_expansion.center
            r = np.linalg.norm(dx)

            if r < 1e-14:
                continue

            if self.dimension == 2:
                # 2D: Use complex representation
                theta = np.arctan2(dx[1], dx[0])
                z = r * np.exp(1j * theta)

                for n in range(self.order + 1):
                    coeff = local_expansion.get_coefficient(n)
                    # Local expansion: L_n += q * z^n / n! (for Laplace)
                    new_coeff = coeff + particle.charge * (z ** n)
                    local_expansion.set_coefficient(n, new_coeff)
            else:
                # 3D: Use spherical harmonics
                theta = np.arccos(np.clip(dx[2] / (r + 1e-14), -1, 1))
                phi = np.arctan2(dx[1], dx[0])

                for n in range(self.order + 1):
                    for m in range(-n, n + 1):
                        coeff = local_expansion.get_coefficient(n, m)
                        Y_nm = local_expansion._spherical_harmonic(n, m, theta, phi)
                        # L_{n,m} += q * Y_{n,m} * r^n
                        new_coeff = coeff + particle.charge * Y_nm * (r ** n)
                        local_expansion.set_coefficient(n, new_coeff, m)


# ============================================================================
# High-Frequency FMM Operators
# ============================================================================

class Cubature:
    """
    Spherical cubature rules for high-frequency FMM.

    From Chapter 3, Section 3.1.1:
    Provides quadrature nodes and weights on the unit sphere S²
    for numerical integration of oscillatory kernels.
    """

    def __init__(self, order: int):
        """
        Initialize cubature rule.

        Args:
            order: Cubature order L (number of θ and φ points)
        """
        self.order = order
        self.nodes, self.weights = self._generate_cubature()

    def _generate_cubature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate cubature nodes and weights.

        Uses Gauss-Legendre quadrature for θ and trapezoidal rule for φ.

        Returns:
            Tuple of (nodes, weights) where:
            - nodes: Array of shape (Q, 3) with unit vectors λ_q
            - weights: Array of shape (Q,) with quadrature weights w_q
        """
        # Gauss-Legendre nodes for cos(θ)
        from scipy.special import roots_legendre
        cos_theta, theta_weights = roots_legendre(self.order)

        # Trapezoidal rule for φ
        phi = 2 * np.pi * np.arange(2 * self.order) / (2 * self.order)
        phi_weights = (2 * np.pi / (2 * self.order)) * np.ones(2 * self.order)

        # Create tensor product grid
        nodes = []
        weights = []

        for ct, w_theta in zip(cos_theta, theta_weights):
            theta = np.arccos(ct)
            sin_theta = np.sin(theta)

            for ph, w_phi in zip(phi, phi_weights):
                # Unit vector λ = (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))
                lam = np.array([
                    sin_theta * np.cos(ph),
                    sin_theta * np.sin(ph),
                    ct
                ])
                nodes.append(lam)
                weights.append(w_theta * w_phi * sin_theta)  # Include sin(θ) from spherical measure

        return np.array(nodes), np.array(weights)

    @property
    def num_nodes(self) -> int:
        """Return the number of cubature nodes."""
        return len(self.weights)


class HighFrequencyM2L(Operator):
    """
    High-frequency M2L operator using diagonal translation operators.

    From Chapter 3, Section 3.1.2:
    For high-frequency Helmholtz kernel G_κ(x,y) = e^{iκ|x-y|} / (4π|x-y|),
    the M2L operator becomes diagonal in the cubature basis:

    (M2L * q)(λ_p) = T_L(t, λ_p) * q(λ_p)

    where T_L(t, λ) = Σ_{l=0}^L (2l+1) * h_l^(1)(κ|t|) * P_l(⟨t/|t|, λ⟩)

    This reduces complexity from O(Q²) to O(Q) per M2L operation.
    """

    def __init__(self, order: int, wavenumber: float, dimension: int = 3):
        """
        Initialize high-frequency M2L operator.

        Args:
            order: Truncation order L
            wavenumber: Helmholtz wavenumber κ
            dimension: Spatial dimension (must be 3 for spherical harmonics)
        """
        super().__init__(order, dimension)
        self.wavenumber = wavenumber

        if dimension != 3:
            raise ValueError("High-frequency M2L requires 3D for spherical harmonics")

        # Create cubature rule
        self.cubature = Cubature(order)

        # Cache for diagonal operators
        self._operator_cache: Dict[Tuple[float, float, float], np.ndarray] = {}

    def _compute_diagonal_operator(self, translation_vector: np.ndarray) -> np.ndarray:
        """
        Compute diagonal M2L operator T_L(t, λ_q).

        From Chapter 3, Eq. 3.8:
        T_L(t, λ_q) = Σ_{l=0}^L (2l+1) * h_l^(1)(κ|t|) * P_l(⟨t/|t|, λ_q⟩)

        where h_l^(1) is the spherical Hankel function of the first kind.

        Args:
            translation_vector: Vector t from source to target cell center

        Returns:
            Array of diagonal values T_L(t, λ_q) for each cubature node λ_q
        """
        t = np.linalg.norm(translation_vector)
        if t < 1e-14:
            return np.zeros(self.cubature.num_nodes)

        # Direction of translation
        t_hat = translation_vector / t

        # Compute diagonal operator for each cubature node
        diagonal = np.zeros(self.cubature.num_nodes, dtype=np.complex128)

        from scipy.special import spherical_jn, spherical_yn

        for q, lam in enumerate(self.cubature.nodes):
            # Dot product between translation direction and cubature node
            cos_gamma = np.dot(t_hat, lam)

            # Sum over spherical harmonics degrees
            T = 0.0
            for l in range(self.order + 1):
                # Spherical Hankel function: h_l^(1)(x) = j_l(x) + i * y_l(x)
                j_l = spherical_jn(l, self.wavenumber * t)
                y_l = spherical_yn(l, self.wavenumber * t)
                h_l = j_l + 1j * y_l

                # Legendre polynomial P_l(cos(γ))
                from scipy.special import eval_legendre
                P_l = eval_legendre(l, cos_gamma)

                # Add contribution: (2l+1) * h_l^(1)(κ|t|) * P_l(⟨t/|t|, λ⟩)
                T += (2 * l + 1) * h_l * P_l

            diagonal[q] = T

        return diagonal

    def apply(self, source_coefficients: np.ndarray,
              target_center: np.ndarray,
              source_center: np.ndarray) -> np.ndarray:
        """
        Apply high-frequency M2L operator.

        Args:
            source_coefficients: Source coefficients q(λ_q) at cubature nodes
            target_center: Center of target cell
            source_center: Center of source cell

        Returns:
            Target coefficients (M2L * q)(λ_p) at cubature nodes
        """
        translation_vector = target_center - source_center

        # Check cache
        cache_key = (round(translation_vector[0], 6),
                     round(translation_vector[1], 6),
                     round(translation_vector[2], 6))
        if cache_key in self._operator_cache:
            diagonal = self._operator_cache[cache_key]
        else:
            diagonal = self._compute_diagonal_operator(translation_vector)
            self._operator_cache[cache_key] = diagonal

        # Element-wise multiplication: (M2L * q)(λ_p) = T_L(λ_p) * q(λ_p)
        # This is O(Q) instead of O(Q²)!
        target_coefficients = diagonal * source_coefficients

        return target_coefficients


class SphericalInterpolator:
    """
    Spherical interpolation between different cubature grids.

    From Chapter 3, Section 3.1.3:
    For high-frequency FMM with adaptive order, we need to interpolate
    between different cubature grids at different tree levels.

    Interpolation formula:
    I_S²(µ, λ) = Σ_{l=0}^L (2l+1)/(4π) * P_l(⟨µ, λ⟩)

    where µ is a target direction and λ is a source direction.
    """

    def __init__(self, source_order: int, target_order: int):
        """
        Initialize spherical interpolator.

        Args:
            source_order: Source cubature order
            target_order: Target cubature order
        """
        self.source_order = source_order
        self.target_order = target_order

        self.source_cubature = Cubature(source_order)
        self.target_cubature = Cubature(target_order)

        # Precompute interpolation matrix
        self._interpolation_matrix = self._compute_interpolation_matrix()

    def _compute_interpolation_matrix(self) -> np.ndarray:
        """
        Compute spherical interpolation matrix.

        Returns:
            Matrix I of shape (Q_target, Q_source) such that
            f_target = I @ f_source
        """
        Q_source = self.source_cubature.num_nodes
        Q_target = self.target_cubature.num_nodes

        # Maximum degree for interpolation
        L_max = min(self.source_order, self.target_order)

        # Build interpolation matrix
        I = np.zeros((Q_target, Q_source), dtype=np.complex128)

        from scipy.special import eval_legendre

        for p, mu in enumerate(self.target_cubature.nodes):
            for q, lam in enumerate(self.source_cubature.nodes):
                # Dot product between directions
                cos_gamma = np.dot(mu, lam)

                # Compute interpolation kernel
                # I_S²(µ, λ) = Σ_{l=0}^L_max (2l+1)/(4π) * P_l(⟨µ, λ⟩)
                value = 0.0
                for l in range(L_max + 1):
                    P_l = eval_legendre(l, cos_gamma)
                    value += (2 * l + 1) / (4 * np.pi) * P_l

                I[p, q] = value

        return I

    def interpolate(self, source_coefficients: np.ndarray,
                   translation_vector: np.ndarray,
                   wavenumber: float) -> np.ndarray:
        """
        Interpolate coefficients from source to target grid with translation phase.

        From Chapter 2, Definition 2.2.10 (L2L):
        L2L applies positive phase factor: exp(+iκ⟨ctr(t')-ctr(t), λ⟩)

        Args:
            source_coefficients: Source coefficients p(λ_q)
            translation_vector: Translation vector dx = target_center - source_center
            wavenumber: Helmholtz wavenumber κ

        Returns:
            Interpolated coefficients at target nodes with phase factor
        """
        # Apply interpolation matrix
        target_coefficients = self._interpolation_matrix @ source_coefficients

        # Apply translation phase factor: exp(+iκ⟨dx, λ⟩) for L2L
        for p, lam in enumerate(self.target_cubature.nodes):
            phase = np.exp(1j * wavenumber * np.dot(translation_vector, lam))
            target_coefficients[p] *= phase

        return target_coefficients

    def interpolate_for_m2m(self, source_coefficients: np.ndarray,
                            translation_vector: np.ndarray,
                            wavenumber: float) -> np.ndarray:
        """
        Interpolate coefficients from source to target grid with M2M phase factor.

        From Chapter 2, Definition 2.2.8 (M2M):
        M2M applies NEGATIVE phase factor: exp(-iκ⟨ctr(s)-ctr(s'), λ⟩)

        Note the difference from L2L:
        - M2M: exp(-iκ⟨ctr(s)-ctr(s'), λ⟩) = exp(-iκ⟨dx, λ⟩)
        - L2L: exp(+iκ⟨ctr(t')-ctr(t), λ⟩) = exp(+iκ⟨dx, λ⟩)

        Args:
            source_coefficients: Source coefficients q(λ_q)
            translation_vector: Translation vector dx = target_center - source_center
                                (ctr(s') - ctr(s) where s' is parent, s is child)
            wavenumber: Helmholtz wavenumber κ

        Returns:
            Interpolated coefficients at target nodes with M2M phase factor
        """
        # Apply interpolation matrix
        target_coefficients = self._interpolation_matrix @ source_coefficients

        # Apply translation phase factor: exp(-iκ⟨dx, λ⟩) for M2M
        # Note the NEGATIVE sign in the exponent
        for p, lam in enumerate(self.target_cubature.nodes):
            phase = np.exp(-1j * wavenumber * np.dot(translation_vector, lam))
            target_coefficients[p] *= phase

        return target_coefficients


class HighFrequencyM2M(Operator):
    """
    High-frequency M2M operator using spherical interpolation.

    From Chapter 3, Section 3.1.3:
    M2M for high-frequency FMM requires interpolation between different
    cubature grids when cells at different levels have different orders.
    """

    def __init__(self, order: int, wavenumber: float, dimension: int = 3):
        """
        Initialize high-frequency M2M operator.

        Args:
            order: Maximum expansion order
            wavenumber: Helmholtz wavenumber κ
            dimension: Spatial dimension (must be 3)
        """
        super().__init__(order, dimension)
        self.wavenumber = wavenumber

        if dimension != 3:
            raise ValueError("High-frequency M2M requires 3D for spherical harmonics")

        # Cache interpolators for different order pairs
        self._interpolator_cache: Dict[Tuple[int, int], SphericalInterpolator] = {}

    def _get_interpolator(self, source_order: int, target_order: int) -> SphericalInterpolator:
        """Get or create interpolator for given order pair."""
        key = (source_order, target_order)
        if key not in self._interpolator_cache:
            self._interpolator_cache[key] = SphericalInterpolator(source_order, target_order)
        return self._interpolator_cache[key]

    def apply(self, source_coefficients: np.ndarray,
              source_center: np.ndarray,
              target_center: np.ndarray,
              source_order: int,
              target_order: int) -> np.ndarray:
        """
        Apply high-frequency M2M operator.

        From Chapter 2, Definition 2.2.8:
        M2M[q](λ) = exp(-iκ⟨ctr(s)-ctr(s'), λ⟩) * q(λ)

        where s is the child cell and s' is the parent cell.

        Args:
            source_coefficients: Source coefficients q(λ_q) at cubature nodes
            source_center: Center of source (child) cell
            target_center: Center of target (parent) cell
            source_order: Source cubature order
            target_order: Target cubature order

        Returns:
            Translated coefficients at target cubature nodes
        """
        translation_vector = target_center - source_center

        # Get interpolator for this order pair
        interpolator = self._get_interpolator(source_order, target_order)

        # Interpolate with M2M translation phase (NEGATIVE exponent)
        target_coefficients = interpolator.interpolate_for_m2m(
            source_coefficients,
            translation_vector,
            self.wavenumber
        )

        return target_coefficients


class HighFrequencyL2L(Operator):
    """
    High-frequency L2L operator using spherical interpolation.

    From Chapter 3, Section 3.1.3:
    L2L for high-frequency FMM requires interpolation between different
    cubature grids when cells at different levels have different orders.
    """

    def __init__(self, order: int, wavenumber: float, dimension: int = 3):
        """
        Initialize high-frequency L2L operator.

        Args:
            order: Maximum expansion order
            wavenumber: Helmholtz wavenumber κ
            dimension: Spatial dimension (must be 3)
        """
        super().__init__(order, dimension)
        self.wavenumber = wavenumber

        if dimension != 3:
            raise ValueError("High-frequency L2L requires 3D for spherical harmonics")

        # Cache interpolators for different order pairs
        self._interpolator_cache: Dict[Tuple[int, int], SphericalInterpolator] = {}

    def _get_interpolator(self, source_order: int, target_order: int) -> SphericalInterpolator:
        """Get or create interpolator for given order pair."""
        key = (source_order, target_order)
        if key not in self._interpolator_cache:
            self._interpolator_cache[key] = SphericalInterpolator(source_order, target_order)
        return self._interpolator_cache[key]

    def apply(self, source_coefficients: np.ndarray,
              source_center: np.ndarray,
              target_center: np.ndarray,
              source_order: int,
              target_order: int) -> np.ndarray:
        """
        Apply high-frequency L2L operator.

        From Chapter 2, Definition 2.2.10:
        L2L[p](λ) = exp(+iκ⟨ctr(t')-ctr(t), λ⟩) * p(λ)

        where t is the parent cell and t' is the child cell.

        Args:
            source_coefficients: Source coefficients p(λ_q) at cubature nodes
            source_center: Center of source (parent) cell
            target_center: Center of target (child) cell
            source_order: Source cubature order
            target_order: Target cubature order

        Returns:
            Translated coefficients at target cubature nodes
        """
        translation_vector = target_center - source_center

        # Get interpolator for this order pair
        interpolator = self._get_interpolator(source_order, target_order)

        # Interpolate with L2L translation phase (POSITIVE exponent)
        target_coefficients = interpolator.interpolate(
            source_coefficients,
            translation_vector,
            self.wavenumber
        )

        return target_coefficients
