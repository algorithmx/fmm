"""
FMM Kernels Module

Common kernel functions used in FMM applications.
"""

import numpy as np
from typing import Callable
from abc import ABC, abstractmethod


class Kernel(ABC):
    """Abstract base class for kernel functions."""

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate kernel G(x, y).

        Args:
            x: Source point coordinates
            y: Target point coordinates

        Returns:
            Kernel value
        """
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of kernel with respect to target point.

        Args:
            x: Source point coordinates
            y: Target point coordinates

        Returns:
            Gradient vector
        """
        pass


class LaplaceKernel(Kernel):
    """
    Laplace kernel (Green's function for Laplace equation).

    G(x, y) = -1/(2*pi) * log(|x - y|)    in 2D
    G(x, y) = 1/(4*pi*|x - y|)            in 3D
    """

    def __init__(self, dimension: int = 2):
        """
        Initialize Laplace kernel.

        Args:
            dimension: Spatial dimension (2 or 3)
        """
        if dimension not in [2, 3]:
            raise ValueError("Dimension must be 2 or 3")
        self.dimension = dimension

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate Laplace kernel."""
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return 0.0  # Self-interaction

        if self.dimension == 2:
            return -np.log(r) / (2 * np.pi)
        else:
            return 1.0 / (4 * np.pi * r)

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of Laplace kernel."""
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return np.zeros_like(x)

        grad = np.zeros_like(x)
        if self.dimension == 2:
            grad = -(x - y) / (2 * np.pi * r ** 2)
        else:
            grad = -(x - y) / (4 * np.pi * r ** 3)

        return grad


class HelmholtzKernel(Kernel):
    """
    Helmholtz kernel (Green's function for Helmholtz equation).

    From Chapter 3: High-Frequency FMM for Helmholtz

    G(x, y) = i/4 * H_0^(1)(k*|x-y|)    in 2D
    G(x, y) = exp(i*k*|x-y|)/(4*pi*|x-y|) in 3D

    where H_0^(1) is the Hankel function of the first kind.

    For high-frequency regime (κw >> 1), the truncation order is:
    L ≈ κw + (1.8d₀)²/³(κw)¹/³

    Uses spherical cubature for discretization on the unit sphere.
    """

    def __init__(self, wavenumber: float, dimension: int = 2,
                 use_high_frequency: bool = True, cell_diameter: float = 1.0):
        """
        Initialize Helmholtz kernel.

        Args:
            wavenumber: Wavenumber k (κ)
            dimension: Spatial dimension (2 or 3)
            use_high_frequency: Enable high-frequency formulation
            cell_diameter: Typical cell diameter d₀ for truncation order calculation
        """
        if dimension not in [2, 3]:
            raise ValueError("Dimension must be 2 or 3")
        self.dimension = dimension
        self.k = wavenumber
        self.use_high_frequency = use_high_frequency
        self.cell_diameter = cell_diameter

        # High-frequency parameter
        self.kw = wavenumber * cell_diameter

        # Compute adaptive truncation order for high-frequency
        if use_high_frequency:
            self.adaptive_order = self._compute_adaptive_order()

        # Initialize spherical cubature for 3D
        if dimension == 3:
            self._init_spherical_cubature()

    def _compute_adaptive_order(self) -> int:
        """
        Compute adaptive truncation order for high-frequency Helmholtz.

        From Chapter 3:
        L ≈ κw + (1.8d₀)²/³(κw)¹/³

        where:
        - κw = k * cell_diameter is the high-frequency parameter
        - d₀ is the cell diameter
        """
        kw = self.kw
        # Truncation order formula
        L = int(np.ceil(kw + (1.8 * self.cell_diameter)**(2/3) * kw**(1/3)))
        # Add safety margin
        return max(L, 10)  # Minimum order of 10 for stability

    def _init_spherical_cubature(self):
        """
        Initialize spherical cubature nodes and weights for 3D.

        Uses product Gauss-Legendre quadrature on the sphere.
        From Chapter 3: Spherical cubature rules.
        """
        from scipy.special import roots_legendre

        # Number of quadrature points per dimension
        self.n_theta = 16  # Polar angle points
        self.n_phi = 32    # Azimuthal angle points

        # Get Gauss-Legendre nodes and weights for theta
        theta_nodes, theta_weights = roots_legendre(self.n_theta)

        # Map from [-1, 1] to [0, π]
        self.sphere_theta = np.arccos(theta_nodes)
        self.sphere_theta_weights = theta_weights

        # Equally spaced points in phi
        self.sphere_phi = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        self.sphere_phi_weights = np.full(self.n_phi, 2*np.pi / self.n_phi)

        # Create combined weight array
        self._sphere_weights = np.outer(
            self.sphere_theta_weights,
            self.sphere_phi_weights
        ).ravel()

    def get_spherical_cubature_points(self) -> tuple:
        """
        Get spherical cubature points and weights for 3D.

        Returns:
            Tuple of (theta, phi, weights) arrays
        """
        if self.dimension != 3:
            raise ValueError("Spherical cubature only available for 3D")

        # Create meshgrid for all points
        theta_grid, phi_grid = np.meshgrid(
            self.sphere_theta,
            self.sphere_phi,
            indexing='ij'
        )

        return theta_grid.ravel(), phi_grid.ravel(), self._sphere_weights

    def __call__(self, x: np.ndarray, y: np.ndarray) -> complex:
        """
        Evaluate Helmholtz kernel (full complex value).

        For 2D: G = i/4 * H_0^(1)(kr)
        For 3D: G = exp(ikr) / (4*pi*r)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return 0.0  # Self-interaction

        kr = self.k * r

        if self.dimension == 2:
            # 2D Helmholtz: i/4 * H_0^(1)(kr)
            # H_0^(1)(kr) = J_0(kr) + i*Y_0(kr)
            from scipy.special import jv, yv
            j0 = jv(0, kr)   # Bessel function of first kind
            y0 = yv(0, kr)   # Bessel function of second kind

            # i/4 * (J0 + i*Y0) = i*J0/4 - Y0/4
            return -0.25 * y0 + 0.25j * j0
        else:
            # 3D Helmholtz: exp(ikr) / (4*pi*r)
            # = cos(kr)/(4*pi*r) + i*sin(kr)/(4*pi*r)
            return (np.cos(kr) + 1j * np.sin(kr)) / (4 * np.pi * r)

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of Helmholtz kernel.

        Returns complex gradient vector.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return np.zeros_like(x, dtype=complex)

        direction = (x - y) / r
        kr = self.k * r

        if self.dimension == 2:
            # Gradient of i/4 * H_0^(1)(kr)
            # d/dr H_0^(1)(kr) = -k * H_1^(1)(kr)
            from scipy.special import jv, yv
            j1 = jv(1, kr)   # Bessel J_1
            y1 = yv(1, kr)   # Bessel Y_1

            # H_1^(1) = J_1 + i*Y_1
            # Gradient = -i*k/4 * H_1^(1) * direction/r
            h1 = j1 + 1j * y1
            grad_factor = -1j * self.k / 4 * h1 / r
            return grad_factor * direction
        else:
            # Gradient of exp(ikr) / (4*pi*r)
            # d/dr [exp(ikr)/(4*pi*r)] = exp(ikr) * (ik*r - 1) / (4*pi*r²)
            cos_kr = np.cos(kr)
            sin_kr = np.sin(kr)
            exp_kr = cos_kr + 1j * sin_kr

            factor = exp_kr * (1j * self.k * r - 1) / (4 * np.pi * r ** 2)
            return factor * direction

    def get_recommended_expansion_order(self) -> int:
        """
        Get recommended expansion order based on frequency regime.

        From Chapter 3:
        - Low frequency (κw << 1): Fixed order (e.g., 10-15)
        - High frequency (κw >> 1): Adaptive order
        """
        if self.use_high_frequency and self.kw > 1.0:
            return self.adaptive_order
        else:
            # Standard order for low-frequency regime
            return max(10, int(np.ceil(self.k * 2)))

    def gegenbauer_expansion(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                             max_order: int) -> np.ndarray:
        """
        Compute Gegenbauer expansion for high-frequency Helmholtz.

        From Chapter 3: Uses Gegenbauer addition theorem for directional treatment.

        exp(ik·(x-y)) = 4π * Σ_{n=0}^∞ i^n * j_n(kr) * Σ_{m=-n}^n Y_{n,m}(θ_k) * Y_{n,m}^*(θ_{x-y})

        This enables diagonal M2L operators in the high-frequency formulation.
        """
        from scipy.special import spherical_jn, sph_harm

        result = np.zeros_like(r, dtype=complex)

        for n in range(max_order + 1):
            # Spherical Bessel function j_n(kr)
            jn = spherical_jn(n, self.k * r)

            # Sum over azimuthal orders
            for m in range(-n, n + 1):
                # This is a placeholder for the full directional expansion
                # Full implementation would integrate over incoming directions
                result += (1j ** n) * jn

        return result


class YukawaKernel(Kernel):
    """
    Yukawa (screened Coulomb) kernel.

    G(x, y) = exp(-kappa*|x-y|) / |x-y|

    Used in plasma physics and molecular dynamics.
    """

    def __init__(self, kappa: float):
        """
        Initialize Yukawa kernel.

        Args:
            kappa: Screening parameter
        """
        self.kappa = kappa

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate Yukawa kernel."""
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return 0.0  # Self-interaction

        return np.exp(-self.kappa * r) / r

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of Yukawa kernel."""
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return np.zeros_like(x)

        direction = (x - y) / r
        factor = -(1 + self.kappa * r) * np.exp(-self.kappa * r) / (r ** 2)
        return direction * factor


class CoulombKernel(Kernel):
    """
    Coulomb kernel (electrostatic potential).

    G(x, y) = 1 / |x - y|

    Similar to 3D Laplace kernel but simpler constant factor.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Evaluate Coulomb kernel."""
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return 0.0  # Self-interaction

        return 1.0 / r

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient of Coulomb kernel."""
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return np.zeros_like(x)

        return -(x - y) / (r ** 3)


class StokesKernel(Kernel):
    """
    Stokeslet kernel (fluid dynamics).

    G(x, y) = I/|x-y| + (x-y)(x-y)^T/|x-y|^3

    Used for Stokes flow problems.
    """

    def __init__(self, viscosity: float = 1.0):
        """
        Initialize Stokes kernel.

        Args:
            viscosity: Fluid viscosity
        """
        self.viscosity = viscosity

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate Stokes kernel (returns scalar for simplicity).

        For full implementation, this would return a matrix.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return 0.0  # Self-interaction

        return 1.0 / (8 * np.pi * self.viscosity * r)

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient (simplified)."""
        x = np.asarray(x)
        y = np.asarray(y)
        r = np.linalg.norm(x - y)

        if r < 1e-14:
            return np.zeros_like(x)

        return -(x - y) / (8 * np.pi * self.viscosity * r ** 3)


def create_kernel(name: str, **kwargs) -> Kernel:
    """
    Factory function to create kernel instances.

    Args:
        name: Kernel type name ('laplace', 'helmholtz', 'yukawa', 'coulomb', 'stokes')
        **kwargs: Kernel-specific parameters

    Returns:
        Kernel instance
    """
    name = name.lower()

    if name == 'laplace':
        dimension = kwargs.get('dimension', 2)
        return LaplaceKernel(dimension)
    elif name == 'helmholtz':
        wavenumber = kwargs.get('wavenumber', 1.0)
        dimension = kwargs.get('dimension', 2)
        return HelmholtzKernel(wavenumber, dimension)
    elif name == 'yukawa':
        kappa = kwargs.get('kappa', 1.0)
        return YukawaKernel(kappa)
    elif name == 'coulomb':
        return CoulombKernel()
    elif name == 'stokes':
        viscosity = kwargs.get('viscosity', 1.0)
        return StokesKernel(viscosity)
    else:
        raise ValueError(f"Unknown kernel type: {name}")


__all__ = [
    'Kernel',
    'LaplaceKernel',
    'HelmholtzKernel',
    'YukawaKernel',
    'CoulombKernel',
    'StokesKernel',
    'create_kernel',
]
