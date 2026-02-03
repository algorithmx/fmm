"""
Expansion Module

Defines the mathematical expansions used in FMM.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class Expansion(ABC):
    """
    Abstract base class for FMM expansions.

    An expansion represents an approximation of field contributions
    using a series expansion (multipole or local).
    """

    def __init__(self, order: int, dimension: int = 2):
        """
        Initialize the expansion.

        Args:
            order: Truncation order of the expansion
            dimension: Spatial dimension (2 or 3)
        """
        self.order = order
        self.dimension = dimension
        self._coefficients: Optional[np.ndarray] = None

    @property
    @abstractmethod
    def num_coefficients(self) -> int:
        """Return the number of coefficients in the expansion."""
        pass

    @abstractmethod
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the expansion at given points.

        Args:
            points: Array of points to evaluate at (N x dimension)

        Returns:
            Array of potential values at each point
        """
        pass

    @abstractmethod
    def zero(self):
        """Reset all coefficients to zero."""
        pass

    def _vectorized_spherical_harmonics(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Compute all spherical harmonics for all points at once.

        Uses scipy.special.sph_harm for vectorized computation.

        Args:
            theta: Polar angles (num_points,)
            phi: Azimuthal angles (num_points,)

        Returns:
            Y: Array of shape (num_points, p+1, 2p+1) containing Y_{n,m}
               Indexing: Y[i, n, m+p] = Y_{n,m}(theta[i], phi[i])
        """
        from scipy.special import sph_harm

        num_points = len(theta)
        p = self.order

        # Initialize output array
        Y = np.zeros((num_points, p + 1, 2 * p + 1), dtype=np.complex128)

        # Vectorized computation for all (n,m) combinations
        for n in range(p + 1):
            for m in range(-n, n + 1):
                Y[:, n, m + p] = sph_harm(m, n, phi, theta)

        return Y

    def add(self, other: 'Expansion'):
        """Add another expansion to this one."""
        if self._coefficients is not None and other._coefficients is not None:
            self._coefficients += other._coefficients

    def scale(self, factor: float):
        """Scale all coefficients by a factor."""
        if self._coefficients is not None:
            self._coefficients *= factor


class MultipoleExpansion(Expansion):
    """
    Multipole expansion representing far-field contributions from source particles.

    Valid OUTSIDE the cell that created it.
    """

    def __init__(self, center: np.ndarray, order: int, dimension: int = 2):
        """
        Initialize multipole expansion.

        Args:
            center: Center of the cell this expansion represents
            order: Truncation order of the expansion
            dimension: Spatial dimension
        """
        super().__init__(order, dimension)
        self.center = np.asarray(center, dtype=np.float64)
        self._initialize_coefficients()

    def _initialize_coefficients(self):
        """Initialize coefficient array."""
        if self.dimension == 2:
            # For 2D: (order + 1) complex coefficients
            # Using real representation: 2 * (order + 1) for [real, imag] pairs
            self._coefficients = np.zeros(2 * (self.order + 1), dtype=np.float64)
        else:
            # For 3D: Spherical harmonics up to order p
            # Number of coefficients: (p + 1)^2
            self._coefficients = np.zeros((self.order + 1) ** 2, dtype=np.complex128)

    @property
    def num_coefficients(self) -> int:
        """Return the number of coefficients."""
        return len(self._coefficients)

    def get_coefficient(self, n: int, m: Optional[int] = None) -> complex:
        """
        Get a coefficient from the expansion.

        For 2D: n is the order index
        For 3D: n is the degree, m is the order

        3D Indexing: idx = n(n+1) + m
            Packs (n,m) pairs into flat array:
            - n=0: [0,0] → idx=0
            - n=1: [-1,1] → idx=[1,2,3]
            - n=2: [-2,2] → idx=[4,5,6,7,8]
            - n=p: [-(p),p] → idx=[p², (p+1)²-1]

        Returns:
            Complex coefficient value
        """
        if self.dimension == 2:
            if n < 0 or n > self.order:
                return 0.0
            real_part = self._coefficients[2 * n]
            imag_part = self._coefficients[2 * n + 1]
            return complex(real_part, imag_part)
        else:
            # 3D: map (n, m) to flat index
            if m is None:
                m = 0
            # Boundary check: m must be in [-n, n]
            if m < -n or m > n:
                return 0.0
            idx = n * (n + 1) + m
            return self._coefficients[idx]

    def set_coefficient(self, n: int, value: complex, m: Optional[int] = None):
        """Set a coefficient in the expansion."""
        if self.dimension == 2:
            if 0 <= n <= self.order:
                self._coefficients[2 * n] = value.real
                self._coefficients[2 * n + 1] = value.imag
        else:
            if m is None:
                m = 0
            # Boundary check: m must be in [-n, n]
            if -n <= m <= n:
                idx = n * (n + 1) + m
                self._coefficients[idx] = value

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the multipole expansion at given points.

        Uses the standard multipole expansion formula:
        phi(r) = sum_{n=0}^p (M_n / |r - center|^{n+1}) * P_n(cos(theta))

        Args:
            points: Array of points (N x dimension)

        Returns:
            Array of potential values
        """
        if self.dimension == 2:
            return self._evaluate_2d(points)
        else:
            return self._evaluate_3d(points)

    def _evaluate_2d(self, points: np.ndarray) -> np.ndarray:
        """Evaluate 2D multipole expansion."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        result = np.zeros(points.shape[0], dtype=np.float64)

        # Compute relative positions
        dx = points[:, 0] - self.center[0]
        dy = points[:, 1] - self.center[1]
        r2 = dx**2 + dy**2 + 1e-14  # Avoid division by zero

        # Convert to polar coordinates
        r = np.sqrt(r2)
        theta = np.arctan2(dy, dx)

        # Evaluate multipole expansion
        for n in range(self.order + 1):
            coeff = self.get_coefficient(n)
            # Multipole term: coeff / r^(n+1) * exp(-i*n*theta)
            magnitude = np.abs(coeff) / (r ** (n + 1))
            phase = np.angle(coeff) - n * theta
            result += magnitude * np.cos(phase)

        return result

    def _evaluate_3d(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate 3D multipole expansion using spherical harmonics.

        Vectorized implementation using scipy.special.sph_harm.

        From Chapter 2, Eq. 2.12-2.13:
        Φ(x) = ∑_{n=0}^p ∑_{m=-n}^n M_{n,m} * Y_{n,m}(θ,φ) / r^{n+1}

        Complexity: O(Np²) with vectorization
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        num_points = points.shape[0]
        result = np.zeros(num_points, dtype=np.float64)

        # Compute relative positions
        dx = points[:, 0] - self.center[0]
        dy = points[:, 1] - self.center[1]
        dz = points[:, 2] - self.center[2]
        r2 = dx**2 + dy**2 + dz**2 + 1e-14

        r = np.sqrt(r2)

        # Convert to spherical coordinates
        theta = np.arccos(np.clip(dz / r, -1, 1))
        phi = np.arctan2(dy, dx)

        # Compute all spherical harmonics at once (vectorized)
        Y_all = self._vectorized_spherical_harmonics(theta, phi)

        # Evaluate expansion
        for n in range(self.order + 1):
            r_power = r ** (n + 1)
            for m in range(-n, n + 1):
                coeff = self.get_coefficient(n, m)
                Y_nm = Y_all[:, n, m + self.order]
                result += (coeff * Y_nm / r_power).real

        return result

    def _spherical_harmonic(self, n: int, m: int, theta: float, phi: float) -> complex:
        """
        Compute spherical harmonic Y_{n,m}(theta, phi).

        From Chapter 2, Eq. 2.12-2.13:
        Y_{n,m}(θ,φ) = sqrt((2n+1)/(4π) * (n-m)!/(n+m)!) * P_n^m(cos(θ)) * e^(imφ)

        Uses proper Condon-Shortley phase (-1)^m for m > 0.

        Args:
            n: Degree of the spherical harmonic
            m: Order of the spherical harmonic (-n ≤ m ≤ n)
            theta: Polar angle (0 ≤ theta ≤ π)
            phi: Azimuthal angle (0 ≤ phi < 2π)

        Returns:
            Complex spherical harmonic value
        """
        from scipy.special import lpmv
        import scipy.special as sp

        if abs(m) > n:
            return 0.0

        # Compute the normalization factor
        # N_{n,m} = sqrt((2n+1)/(4π) * (n-|m|)!/(n+|m|)!)
        abs_m = abs(m)
        log_norm = 0.5 * (np.log(2*n + 1) - np.log(4*np.pi) +
                         sp.gammaln(n - abs_m + 1) - sp.gammaln(n + abs_m + 1))
        normalization = np.exp(log_norm)

        # Associated Legendre polynomial P_n^|m|(cos(theta))
        cos_theta = np.cos(theta)
        p_nm = lpmv(abs_m, n, cos_theta)

        # Condon-Shortley phase: (-1)^m for m > 0
        phase_factor = 1.0
        if m > 0:
            phase_factor = (-1) ** m

        # Complex exponential e^(imφ)
        exp_imphi = np.exp(1j * m * phi)

        # Full complex spherical harmonic
        Y_nm = normalization * p_nm * phase_factor * exp_imphi

        return Y_nm

    def zero(self):
        """Reset all coefficients to zero."""
        if self._coefficients is not None:
            self._coefficients.fill(0.0)


class LocalExpansion(Expansion):
    """
    Local expansion representing far-field contributions at a target cell.

    Valid INSIDE the cell it represents.
    """

    def __init__(self, center: np.ndarray, order: int, dimension: int = 2):
        """
        Initialize local expansion.

        Args:
            center: Center of the cell this expansion represents
            order: Truncation order of the expansion
            dimension: Spatial dimension
        """
        super().__init__(order, dimension)
        self.center = np.asarray(center, dtype=np.float64)
        self._initialize_coefficients()

    def _initialize_coefficients(self):
        """Initialize coefficient array."""
        if self.dimension == 2:
            self._coefficients = np.zeros(2 * (self.order + 1), dtype=np.float64)
        else:
            self._coefficients = np.zeros((self.order + 1) ** 2, dtype=np.complex128)

    @property
    def num_coefficients(self) -> int:
        """Return the number of coefficients."""
        return len(self._coefficients)

    def get_coefficient(self, n: int, m: Optional[int] = None) -> complex:
        """
        Get a coefficient from the expansion.

        For 2D: n is the order index
        For 3D: n is the degree, m is the order

        3D Indexing: idx = n(n+1) + m
            Packs (n,m) pairs into flat array:
            - n=0: [0,0] → idx=0
            - n=1: [-1,1] → idx=[1,2,3]
            - n=2: [-2,2] → idx=[4,5,6,7,8]
            - n=p: [-(p),p] → idx=[p², (p+1)²-1]

        Returns:
            Complex coefficient value
        """
        if self.dimension == 2:
            if n < 0 or n > self.order:
                return 0.0
            real_part = self._coefficients[2 * n]
            imag_part = self._coefficients[2 * n + 1]
            return complex(real_part, imag_part)
        else:
            if m is None:
                m = 0
            # Boundary check: m must be in [-n, n]
            if m < -n or m > n:
                return 0.0
            idx = n * (n + 1) + m
            return self._coefficients[idx]

    def set_coefficient(self, n: int, value: complex, m: Optional[int] = None):
        """Set a coefficient in the expansion."""
        if self.dimension == 2:
            if 0 <= n <= self.order:
                self._coefficients[2 * n] = value.real
                self._coefficients[2 * n + 1] = value.imag
        else:
            if m is None:
                m = 0
            # Boundary check: m must be in [-n, n]
            if -n <= m <= n:
                idx = n * (n + 1) + m
                self._coefficients[idx] = value

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the local expansion at given points.

        Uses the standard local expansion formula:
        phi(r) = sum_{n=0}^p L_n * |r - center|^n * P_n(cos(theta))

        Args:
            points: Array of points (N x dimension)

        Returns:
            Array of potential values
        """
        if self.dimension == 2:
            return self._evaluate_2d(points)
        else:
            return self._evaluate_3d(points)

    def _evaluate_2d(self, points: np.ndarray) -> np.ndarray:
        """Evaluate 2D local expansion."""
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        result = np.zeros(points.shape[0], dtype=np.float64)

        # Compute relative positions
        dx = points[:, 0] - self.center[0]
        dy = points[:, 1] - self.center[1]
        r2 = dx**2 + dy**2 + 1e-14

        r = np.sqrt(r2)
        theta = np.arctan2(dy, dx)

        # Evaluate local expansion
        for n in range(self.order + 1):
            coeff = self.get_coefficient(n)
            # Local term: coeff * r^n * exp(i*n*theta)
            magnitude = np.abs(coeff) * (r ** n)
            phase = np.angle(coeff) + n * theta
            result += magnitude * np.cos(phase)

        return result

    def _evaluate_3d(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate 3D local expansion using spherical harmonics.

        Vectorized implementation using scipy.special.sph_harm.

        Local expansion formula:
        Φ(x) = ∑_{n=0}^p ∑_{m=-n}^n L_{n,m} * Y_{n,m}(θ,φ) * r^n

        Complexity: O(Np²) with vectorization
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        num_points = points.shape[0]
        result = np.zeros(num_points, dtype=np.float64)

        # Compute relative positions
        dx = points[:, 0] - self.center[0]
        dy = points[:, 1] - self.center[1]
        dz = points[:, 2] - self.center[2]
        r2 = dx**2 + dy**2 + dz**2 + 1e-14

        r = np.sqrt(r2)

        # Convert to spherical coordinates
        theta = np.arccos(np.clip(dz / r, -1, 1))
        phi = np.arctan2(dy, dx)

        # Compute all spherical harmonics at once (vectorized)
        Y_all = self._vectorized_spherical_harmonics(theta, phi)

        # Evaluate expansion
        for n in range(self.order + 1):
            r_power = r ** n
            for m in range(-n, n + 1):
                coeff = self.get_coefficient(n, m)
                Y_nm = Y_all[:, n, m + self.order]
                result += (coeff * Y_nm * r_power).real

        return result

    def _spherical_harmonic(self, n: int, m: int, theta: float, phi: float) -> complex:
        """
        Compute spherical harmonic Y_{n,m}(theta, phi).

        From Chapter 2, Eq. 2.12-2.13:
        Y_{n,m}(θ,φ) = sqrt((2n+1)/(4π) * (n-m)!/(n+m)!) * P_n^m(cos(θ)) * e^(imφ)

        Uses proper Condon-Shortley phase (-1)^m for m > 0.
        """
        from scipy.special import lpmv
        import scipy.special as sp

        if abs(m) > n:
            return 0.0

        # Compute the normalization factor
        abs_m = abs(m)
        log_norm = 0.5 * (np.log(2*n + 1) - np.log(4*np.pi) +
                         sp.gammaln(n - abs_m + 1) - sp.gammaln(n + abs_m + 1))
        normalization = np.exp(log_norm)

        # Associated Legendre polynomial P_n^|m|(cos(theta))
        cos_theta = np.cos(theta)
        p_nm = lpmv(abs_m, n, cos_theta)

        # Condon-Shortley phase: (-1)^m for m > 0
        phase_factor = 1.0
        if m > 0:
            phase_factor = (-1) ** m

        # Complex exponential e^(imφ)
        exp_imphi = np.exp(1j * m * phi)

        # Full complex spherical harmonic
        Y_nm = normalization * p_nm * phase_factor * exp_imphi

        return Y_nm

    def zero(self):
        """Reset all coefficients to zero."""
        if self._coefficients is not None:
            self._coefficients.fill(0.0)
