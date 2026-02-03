"""
Kernel-Independent FMM Module

Implements kernel-independent FMM using Chebyshev interpolation
and SVD compression techniques.

Includes FFT-based M2L for translation-invariant kernels.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy.fft import fftn, ifftn

from .cell import Cell
from .particle import Particle
from .expansion import Expansion, MultipoleExpansion, LocalExpansion


class ChebyshevInterpolant:
    """
    Chebyshev polynomial interpolation for kernel-independent FMM.

    Uses Chebyshev nodes for better stability and convergence.
    """

    def __init__(self, order: int, dimension: int = 2):
        """
        Initialize Chebyshev interpolant.

        Args:
            order: Number of Chebyshev points per dimension
            dimension: Spatial dimension (2 or 3)
        """
        self.order = order
        self.dimension = dimension

        # Generate Chebyshev nodes on [-1, 1]
        # Chebyshev nodes: cos((2k-1)π/(2n)) for k = 1,...,n
        k = np.arange(1, order + 1)
        self.nodes_1d = np.cos((2 * k - 1) * np.pi / (2 * order))

        # Create multi-dimensional grid
        self._create_grid()

    def _create_grid(self):
        """Create multi-dimensional Chebyshev grid."""
        if self.dimension == 2:
            # 2D grid
            x, y = np.meshgrid(self.nodes_1d, self.nodes_1d, indexing='ij')
            self.nodes = np.column_stack([x.ravel(), y.ravel()])
            self.num_nodes = self.order ** 2
        else:
            # 3D grid
            x, y, z = np.meshgrid(self.nodes_1d, self.nodes_1d, self.nodes_1d, indexing='ij')
            self.nodes = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            self.num_nodes = self.order ** 3

    def interpolate(self, values: np.ndarray, points: np.ndarray,
                    source_bounds: Tuple[np.ndarray, np.ndarray],
                    target_bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Interpolate values from Chebyshev grid to target points.

        Args:
            values: Function values at Chebyshev nodes
            points: Target points to evaluate at
            source_bounds: (min, max) bounds of source domain
            target_bounds: (min, max) bounds of target domain

        Returns:
            Interpolated values at target points
        """
        # Map points to reference domain [-1, 1]^d
        source_min, source_max = source_bounds
        target_min, target_max = target_bounds

        # Normalize target points to [-1, 1]
        normalized_points = np.zeros_like(points)
        for d in range(self.dimension):
            center = (target_min[d] + target_max[d]) / 2.0
            half_size = (target_max[d] - target_min[d]) / 2.0
            normalized_points[:, d] = (points[:, d] - center) / half_size

        # Perform Chebyshev interpolation
        result = np.zeros(len(points))
        for i, point in enumerate(normalized_points):
            result[i] = self._evaluate_chebyshev(values, point)

        return result

    def _evaluate_chebyshev(self, coefficients: np.ndarray, point: np.ndarray) -> float:
        """Evaluate Chebyshev series at a point using barycentric formula."""
        if self.dimension == 2:
            return self._evaluate_chebyshev_2d(coefficients, point)
        else:
            return self._evaluate_chebyshev_3d(coefficients, point)

    def _evaluate_chebyshev_2d(self, coefficients: np.ndarray, point: np.ndarray) -> float:
        """Evaluate 2D Chebyshev interpolant."""
        x, y = point

        # Compute Chebyshev polynomials at point
        tx = np.cos(np.arange(self.order) * np.arccos(x))
        ty = np.cos(np.arange(self.order) * np.arccos(y))

        # Sum over coefficients
        result = 0.0
        idx = 0
        for i in range(self.order):
            for j in range(self.order):
                result += coefficients[idx] * tx[i] * ty[j]
                idx += 1

        return result

    def _evaluate_chebyshev_3d(self, coefficients: np.ndarray, point: np.ndarray) -> float:
        """Evaluate 3D Chebyshev interpolant."""
        x, y, z = point

        # Compute Chebyshev polynomials at point
        tx = np.cos(np.arange(self.order) * np.arccos(x))
        ty = np.cos(np.arange(self.order) * np.arccos(y))
        tz = np.cos(np.arange(self.order) * np.arccos(z))

        # Sum over coefficients
        result = 0.0
        idx = 0
        for i in range(self.order):
            for j in range(self.order):
                for k in range(self.order):
                    result += coefficients[idx] * tx[i] * ty[j] * tz[k]
                    idx += 1

        return result


class SVDM2LCompressor:
    """
    SVD-based compression for M2L operator.

    Compresses the M2L translation operator using singular value decomposition
    for improved efficiency in kernel-independent FMM.

    From Chapter 4: SVD provides optimal low-rank approximation but is
    computationally expensive: O(#K³ + #L²#K + #L#K²).
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize SVD compressor.

        Args:
            tolerance: Truncation tolerance for SVD
        """
        self.tolerance = tolerance
        self.compressed_operators = {}

    def compress(self, interaction_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compress interaction matrix using SVD.

        Args:
            interaction_matrix: M2L interaction matrix (num_targets x num_sources)

        Returns:
            Tuple of (U, S, Vt) SVD components with truncated rank
        """
        # Perform SVD
        U, S, Vt = np.linalg.svd(interaction_matrix, full_matrices=False)

        # Determine truncation rank based on tolerance
        total_energy = np.sum(S ** 2)
        cumulative_energy = np.cumsum(S ** 2)
        rank = np.searchsorted(cumulative_energy / total_energy, 1.0 - self.tolerance) + 1
        rank = min(rank, len(S))

        # Truncate to rank
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vt_trunc = Vt[:rank, :]

        return U_trunc, S_trunc, Vt_trunc

    def get_rank(self, S: np.ndarray) -> int:
        """Get effective rank based on tolerance."""
        return len(S)


class ACACompressor:
    """
    Adaptive Cross Approximation (ACA) compressor for M2L operator.

    From Chapter 4: ACA provides efficient low-rank approximation with
    complexity O(r²(#K + #L)), where r is the numerical rank.

    ACA is particularly efficient for kernel-generated matrices because:
    1. Only accesses matrix elements adaptively
    2. No full SVD computation required
    3. Works well for smooth, low-rank kernels

    Implements the partially pivoted ACA algorithm.
    """

    def __init__(self, tolerance: float = 1e-6, max_rank: int = 50):
        """
        Initialize ACA compressor.

        Args:
            tolerance: Stopping tolerance for approximation
            max_rank: Maximum rank to compute
        """
        self.tolerance = tolerance
        self.max_rank = max_rank

    def compress(self, interaction_matrix: np.ndarray,
                 matrix_accessor: callable = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compress interaction matrix using Adaptive Cross Approximation.

        From Chapter 4:
        A ≈ U * V^T where U and V are low-rank factors

        Algorithm:
        1. Find maximum pivot in row/column
        2. Extract rank-1 component
        3. Update residual matrix
        4. Repeat until convergence or max_rank

        Args:
            interaction_matrix: M2L interaction matrix (num_targets x num_sources)
            matrix_accessor: Optional function to access matrix elements on-the-fly

        Returns:
            Tuple of (U, V) where U is (num_targets x rank) and V is (num_sources x rank)
        """
        m, n = interaction_matrix.shape

        # Initialize low-rank factors
        U = np.zeros((m, self.max_rank))
        V = np.zeros((n, self.max_rank))

        # Initialize residual (copy of original matrix or accessor)
        if matrix_accessor is None:
            R = interaction_matrix.copy()
        else:
            R = None  # Use accessor instead

        # Track computed pivots
        computed_rows = set()
        computed_cols = set()

        for k in range(self.max_rank):
            # Find pivot: maximum element in residual
            if R is not None:
                # Full residual matrix available
                if k == 0:
                    # First iteration: find global maximum
                    idx = np.argmax(np.abs(R))
                    i, j = idx // n, idx % n
                else:
                    # Subsequent iterations: find row/column maximum
                    i = np.argmax(np.max(np.abs(R[:, :]), axis=1))
                    j = np.argmax(np.abs(R[i, :]))
            else:
                # Use accessor to compute pivot (simplified - needs row/col access)
                # This is a placeholder for full on-the-fly ACA
                if k == 0:
                    row_0 = self._get_row(interaction_matrix, 0, matrix_accessor)
                    j = np.argmax(np.abs(row_0))
                    i = 0
                else:
                    row_i = self._get_row(interaction_matrix, i, matrix_accessor)
                    j = np.argmax(np.abs(row_i))

            # Extract pivot value
            if R is not None:
                pivot = R[i, j]
            else:
                pivot = self._get_element(interaction_matrix, i, j, matrix_accessor)

            if abs(pivot) < 1e-14:
                break

            # Extract column i and row j (normalized)
            if R is not None:
                col_i = R[:, i].copy()
                row_j = R[j, :].copy()
            else:
                col_i = self._get_col(interaction_matrix, i, matrix_accessor)
                row_j = self._get_row(interaction_matrix, j, matrix_accessor)

            # Store rank-1 component
            U[:, k] = col_i
            V[:, k] = row_j / pivot

            # Update residual: R = R - u_k * v_k^T
            if R is not None:
                R -= np.outer(U[:, k], V[:, k])

            # Check convergence
            # Estimate approximation error using Frobenius norm of latest component
            component_norm = np.linalg.norm(U[:, k]) * np.linalg.norm(V[:, k])
            if component_norm < self.tolerance:
                # Trim to current rank
                U = U[:, :k+1]
                V = V[:, :k+1]
                break

        # Return U and V such that A ≈ U @ V.T
        return U, V

    def _get_row(self, matrix: np.ndarray, i: int, accessor: callable) -> np.ndarray:
        """Get a row from the matrix (with caching)."""
        if accessor is None:
            return matrix[i, :]
        else:
            return accessor.row(i)

    def _get_col(self, matrix: np.ndarray, j: int, accessor: callable) -> np.ndarray:
        """Get a column from the matrix (with caching)."""
        if accessor is None:
            return matrix[:, j]
        else:
            return accessor.col(j)

    def _get_element(self, matrix: np.ndarray, i: int, j: int, accessor: callable) -> float:
        """Get a single element from the matrix."""
        if accessor is None:
            return matrix[i, j]
        else:
            return accessor.get(i, j)

    def get_effective_rank(self, U: np.ndarray, V: np.ndarray) -> int:
        """Get the effective rank of the approximation."""
        return U.shape[1]


class KernelIndependentExpansion(Expansion):
    """
    Kernel-independent expansion using interpolation.

    Represents the field contribution using an interpolant rather than
    analytical multipole/local expansions.
    """

    def __init__(self, center: np.ndarray, size: float, order: int,
                 dimension: int = 2):
        """
        Initialize kernel-independent expansion.

        Args:
            center: Center of the cell
            size: Size of the cell
            order: Number of interpolation points per dimension
            dimension: Spatial dimension
        """
        super().__init__(order, dimension)
        self.center = np.asarray(center, dtype=np.float64)
        self.size = size
        self.interpolant = ChebyshevInterpolant(order, dimension)

        # Initialize coefficient array
        self._coefficients = np.zeros(self.interpolant.num_nodes, dtype=np.float64)

        # Compute bounds
        half_size = size / 2.0
        self.bounds = (
            self.center - half_size,
            self.center + half_size
        )

    @property
    def num_coefficients(self) -> int:
        """Return the number of interpolation coefficients."""
        return len(self._coefficients)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the expansion at given points."""
        return self.interpolant.interpolate(
            self._coefficients,
            points,
            self.bounds,
            self.bounds
        )

    def zero(self):
        """Reset all coefficients to zero."""
        self._coefficients.fill(0.0)

    def set_values(self, values: np.ndarray):
        """Set the interpolation values."""
        self._coefficients = values.copy()


class KernelIndependentP2M:
    """
    Kernel-independent P2M operator using Chebyshev interpolation.
    """

    def __init__(self, order: int = 4, dimension: int = 2):
        self.order = order
        self.dimension = dimension

    def apply(self, cell: Cell, kernel_func: callable) -> KernelIndependentExpansion:
        """
        Apply kernel-independent P2M operator.

        Args:
            cell: Leaf cell containing particles
            kernel_func: Kernel function G(x, y)

        Returns:
            Kernel-independent expansion
        """
        expansion = KernelIndependentExpansion(
            center=cell.center,
            size=cell.size,
            order=self.order,
            dimension=self.dimension
        )

        # Evaluate kernel at Chebyshev nodes
        nodes = expansion.interpolant.nodes

        # Map nodes to physical space
        min_bound, max_bound = expansion.bounds
        physical_nodes = np.zeros_like(nodes)
        for d in range(self.dimension):
            center_d = (min_bound[d] + max_bound[d]) / 2.0
            half_size_d = (max_bound[d] - min_bound[d]) / 2.0
            physical_nodes[:, d] = center_d + half_size_d * nodes[:, d]

        # Compute potential at each node
        for i, node in enumerate(physical_nodes):
            potential = 0.0
            for particle in cell.particles:
                value = kernel_func(particle.position, node)
                potential += particle.charge * value
            expansion._coefficients[i] = potential

        return expansion


class KernelIndependentM2L:
    """
    Kernel-independent M2L operator using SVD or ACA compression.

    From Chapter 4:
    - SVD: Optimal but expensive O(#K³ + #L²#K + #L#K²)
    - ACA: Adaptive Cross Approximation O(r²(#K + #L))
    """

    def __init__(self, order: int = 4, dimension: int = 2,
                 compression_tolerance: float = 1e-6,
                 use_aca: bool = False):
        """
        Initialize kernel-independent M2L operator.

        Args:
            order: Number of interpolation points per dimension
            dimension: Spatial dimension (2 or 3)
            compression_tolerance: Tolerance for compression
            use_aca: Use ACA instead of SVD (default: False)
        """
        self.order = order
        self.dimension = dimension
        self.use_aca = use_aca

        if use_aca:
            self.compressor = ACACompressor(compression_tolerance)
        else:
            self.compressor = SVDM2LCompressor(compression_tolerance)

        self._cached_operators = {}

    def apply(self, source_expansion: KernelIndependentExpansion,
              target_center: np.ndarray, target_size: float,
              kernel_func: callable) -> KernelIndependentExpansion:
        """
        Apply kernel-independent M2L operator.

        Args:
            source_expansion: Source expansion
            target_center: Center of target cell
            target_size: Size of target cell
            kernel_func: Kernel function

        Returns:
            Local expansion at target
        """
        target_expansion = KernelIndependentExpansion(
            center=target_center,
            size=target_size,
            order=self.order,
            dimension=self.dimension
        )

        # Check if we have a cached operator for this configuration
        cache_key = self._get_cache_key(source_expansion, target_expansion)
        if cache_key in self._cached_operators:
            if self.use_aca:
                U, V = self._cached_operators[cache_key]
                # A ≈ U @ V.T, so: result = U @ (V.T @ source_coeffs)
                target_expansion._coefficients = U @ (V.T @ source_expansion._coefficients)
            else:
                U, S, Vt = self._cached_operators[cache_key]
                target_expansion._coefficients = U @ (S * (Vt @ source_expansion._coefficients))
            return target_expansion

        # Build interaction matrix
        source_nodes = source_expansion.interpolant.nodes
        target_nodes = target_expansion.interpolant.nodes

        # Map nodes to physical space
        source_min, source_max = source_expansion.bounds
        target_min, target_max = target_expansion.bounds

        physical_source = self._map_to_physical(source_nodes, source_min, source_max)
        physical_target = self._map_to_physical(target_nodes, target_min, target_max)

        # Build interaction matrix
        interaction_matrix = np.zeros((len(physical_target), len(physical_source)))
        for i, t_node in enumerate(physical_target):
            for j, s_node in enumerate(physical_source):
                interaction_matrix[i, j] = kernel_func(s_node, t_node)

        # Compress using selected method
        if self.use_aca:
            U, V = self.compressor.compress(interaction_matrix)
            self._cached_operators[cache_key] = (U, V)
            target_expansion._coefficients = U @ (V.T @ source_expansion._coefficients)
        else:
            U, S, Vt = self.compressor.compress(interaction_matrix)
            self._cached_operators[cache_key] = (U, S, Vt)
            target_expansion._coefficients = U @ (S * (Vt @ source_expansion._coefficients))

        return target_expansion

    def _map_to_physical(self, nodes: np.ndarray, min_bound: np.ndarray,
                        max_bound: np.ndarray) -> np.ndarray:
        """Map normalized nodes to physical space."""
        physical = np.zeros_like(nodes)
        for d in range(self.dimension):
            center_d = (min_bound[d] + max_bound[d]) / 2.0
            half_size_d = (max_bound[d] - min_bound[d]) / 2.0
            physical[:, d] = center_d + half_size_d * nodes[:, d]
        return physical

    def _get_cache_key(self, source: KernelIndependentExpansion,
                      target: KernelIndependentExpansion) -> Tuple:
        """Generate cache key for operator lookup."""
        dx = target.center - source.center
        dist = np.linalg.norm(dx)
        return (dist, source.size, target.size)


class FFTBasedM2L:
    """
    FFT-based M2L operator for translation-invariant kernels.

    From Chapter 4, Section 4.1.4:
    For translation-invariant kernels G(x,y) = K(x-y), the M2L operator
    has Toeplitz structure that can be exploited using FFT.

    Algorithm:
    1. Embed Toeplitz matrix in block-circulant matrix
    2. Diagonalize via FFT
    3. Apply: FFT → multiply by diagonal → IFFT
    4. Extract relevant portion

    Complexity: O(d L^d log L) vs O(L^2d) for direct method.

    This is particularly efficient for kernels that depend only on x-y.
    """

    def __init__(self, order: int, dimension: int = 2):
        """
        Initialize FFT-based M2L operator.

        Args:
            order: Number of interpolation points per dimension
            dimension: Spatial dimension (2 or 3)
        """
        self.order = order
        self.dimension = dimension

        # Cache for FFT kernels
        self._kernel_cache: Dict[Tuple[float, ...], np.ndarray] = {}

    def _is_translation_invariant(self, kernel_func: callable) -> bool:
        """
        Check if kernel is translation-invariant.

        A kernel is translation-invariant if G(x,y) = K(x-y) depends only
        on the difference x-y, not on absolute positions.

        Args:
            kernel_func: Kernel function G(x, y)

        Returns:
            True if kernel appears translation-invariant
        """
        # Test with random points
        np.random.seed(42)
        x1 = np.random.rand(self.dimension)
        y1 = np.random.rand(self.dimension)
        dx = x1 - y1

        x2 = np.random.rand(self.dimension)
        y2 = x2 - dx  # Same difference

        val1 = kernel_func(x1, y1)
        val2 = kernel_func(x2, y2)

        # Check if values are approximately equal
        return abs(val1 - val2) < 1e-10

    def _compute_fft_kernel(self, kernel_func: callable,
                           source_bounds: Tuple[np.ndarray, np.ndarray],
                           target_bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute FFT kernel for translation-invariant M2L.

        From Chapter 4, Section 4.1.4:
        For translation-invariant kernels K(x-y), the M2L operator has Toeplitz
        structure. We embed the Toeplitz matrix in a circulant matrix and use FFT.

        Algorithm:
        1. Generate Chebyshev interpolation nodes for source and target grids
        2. Construct first column of Toeplitz matrix by evaluating kernel at
           positions: target_node[i] - source_node[0] for all i
        3. Embed in (2n-1) circulant matrix
        4. Take FFT to get diagonal kernel in Fourier domain

        Args:
            kernel_func: Translation-invariant kernel function K(x-y)
            source_bounds: (min, max) bounds of source domain
            target_bounds: (min, max) bounds of target domain

        Returns:
            FFT kernel array (diagonal in Fourier domain)
        """
        n = self.order
        fft_size = 2 * n - 1

        # Generate Chebyshev nodes on [-1, 1]
        k = np.arange(1, n + 1)
        nodes_1d = np.cos((2 * k - 1) * np.pi / (2 * n))

        # Create multi-dimensional node grids
        if self.dimension == 2:
            x, y = np.meshgrid(nodes_1d, nodes_1d, indexing='ij')
            source_nodes = np.column_stack([x.ravel(), y.ravel()])
            target_nodes = np.column_stack([x.ravel(), y.ravel()])
        else:
            x, y, z = np.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing='ij')
            source_nodes = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            target_nodes = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        # Map nodes to physical coordinates
        source_min, source_max = source_bounds
        target_min, target_max = target_bounds

        physical_source = np.zeros_like(source_nodes)
        physical_target = np.zeros_like(target_nodes)

        for d in range(self.dimension):
            # Map from [-1, 1] to physical bounds
            center_s = (source_min[d] + source_max[d]) / 2.0
            half_s = (source_max[d] - source_min[d]) / 2.0
            physical_source[:, d] = center_s + half_s * source_nodes[:, d]

            center_t = (target_min[d] + target_max[d]) / 2.0
            half_t = (target_max[d] - target_min[d]) / 2.0
            physical_target[:, d] = center_t + half_t * target_nodes[:, d]

        # Construct first column of Toeplitz matrix
        # Column[i] = kernel(target_node[i] - source_node[0])
        # This gives us the first column of the Toeplitz matrix
        num_nodes = len(physical_source)
        first_column = np.zeros(num_nodes, dtype=np.float64)

        # Reference source node (first node)
        source_ref = physical_source[0]

        for i in range(num_nodes):
            # Evaluate kernel at target_node[i] - source_ref
            offset = physical_target[i] - source_ref
            first_column[i] = kernel_func(np.zeros(self.dimension), offset)

        # Reshape first column to grid format
        first_column_grid = first_column.reshape((n,) * self.dimension)

        # Embed in circulant matrix of size (2n-1)^d
        # Place first column at the beginning of each dimension
        if self.dimension == 2:
            circulant_kernel = np.zeros((fft_size, fft_size), dtype=np.float64)
            circulant_kernel[:n, :n] = first_column_grid
        else:
            circulant_kernel = np.zeros((fft_size, fft_size, fft_size), dtype=np.float64)
            circulant_kernel[:n, :n, :n] = first_column_grid

        # Take FFT to get diagonal kernel in Fourier domain
        fft_kernel = fftn(circulant_kernel)

        return fft_kernel

    def apply(self, source_coefficients: np.ndarray,
              kernel_func: callable,
              source_bounds: Tuple[np.ndarray, np.ndarray],
              target_bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Apply FFT-based M2L operator.

        Args:
            source_coefficients: Source expansion coefficients
            kernel_func: Kernel function (should be translation-invariant)
            source_bounds: (min, max) bounds of source domain
            target_bounds: (min, max) bounds of target domain

        Returns:
            Target coefficients after M2L translation
        """
        # Check if kernel is translation-invariant
        if not self._is_translation_invariant(kernel_func):
            raise ValueError(
                "FFT-based M2L requires translation-invariant kernel. "
                "Use standard KernelIndependentM2L instead."
            )

        # Get or compute FFT kernel
        cache_key = tuple(np.round(target_bounds[0], 6)) + tuple(np.round(target_bounds[1], 6))
        if cache_key not in self._kernel_cache:
            self._kernel_cache[cache_key] = self._compute_fft_kernel(
                kernel_func, source_bounds, target_bounds
            )
        fft_kernel = self._kernel_cache[cache_key]

        # Reshape source coefficients to grid
        n = self.order
        if self.dimension == 2:
            source_grid = source_coefficients.reshape(n, n)
        else:
            source_grid = source_coefficients.reshape(n, n, n)

        # Pad source grid to FFT size
        fft_size = 2 * n - 1
        if self.dimension == 2:
            padded = np.zeros((fft_size, fft_size), dtype=np.complex128)
            padded[:n, :n] = source_grid
        else:
            padded = np.zeros((fft_size, fft_size, fft_size), dtype=np.complex128)
            padded[:n, :n, :n] = source_grid

        # Apply FFT
        source_fft = fftn(padded)

        # Multiply by kernel in Fourier domain (diagonal operation!)
        result_fft = fft_kernel * source_fft

        # Inverse FFT
        result_padded = ifftn(result_fft)

        # Extract relevant portion
        if self.dimension == 2:
            result_grid = result_padded[:n, :n]
        else:
            result_grid = result_padded[:n, :n, :n]

        # Flatten back to 1D
        target_coefficients = result_grid.ravel()

        return target_coefficients


class AdaptiveM2L:
    """
    Adaptive M2L operator that automatically chooses the best method.

    Selects between:
    - FFT-based M2L for translation-invariant kernels (fastest)
    - SVD-based M2L for general smooth kernels
    - ACA for low-rank kernels
    """

    def __init__(self, order: int, dimension: int = 2,
                 tolerance: float = 1e-6):
        """
        Initialize adaptive M2L operator.

        Args:
            order: Number of interpolation points per dimension
            dimension: Spatial dimension
            tolerance: Compression tolerance
        """
        self.order = order
        self.dimension = dimension
        self.tolerance = tolerance

        # Initialize different M2L methods
        self.fft_m2l = FFTBasedM2L(order, dimension)
        self.svd_m2l = SVDM2LCompressor(tolerance)
        self.aca_m2l = ACACompressor(tolerance)

    def apply(self, source_coefficients: np.ndarray,
              kernel_func: callable,
              source_bounds: Tuple[np.ndarray, np.ndarray],
              target_bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Apply adaptive M2L operator.

        Automatically selects the best method based on kernel properties.

        Args:
            source_coefficients: Source expansion coefficients
            kernel_func: Kernel function
            source_bounds: (min, max) bounds of source domain
            target_bounds: (min, max) bounds of target domain

        Returns:
            Target coefficients after M2L translation
        """
        # Check if kernel is translation-invariant
        if self.fft_m2l._is_translation_invariant(kernel_func):
            # Use FFT-based method (fastest)
            return self.fft_m2l.apply(
                source_coefficients, kernel_func, source_bounds, target_bounds
            )

        # For general kernels, use ACA (faster than full SVD)
        # Build interaction matrix and compress with ACA
        n_source = len(source_coefficients)
        n_target = n_source  # Assuming same order

        # Build interaction matrix (simplified - in practice would be cached)
        interaction_matrix = self._build_interaction_matrix(
            kernel_func, source_bounds, target_bounds
        )

        # Compress using ACA
        U, V = self.aca_m2l.compress(interaction_matrix)

        # Apply low-rank approximation
        target_coefficients = U @ (V.T @ source_coefficients)

        return target_coefficients

    def _build_interaction_matrix(self, kernel_func: callable,
                                  source_bounds: Tuple[np.ndarray, np.ndarray],
                                  target_bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Build interaction matrix for M2L."""
        # Simplified implementation
        n = self.order
        size = n ** self.dimension

        # Generate Chebyshev nodes
        k = np.arange(1, n + 1)
        nodes_1d = np.cos((2 * k - 1) * np.pi / (2 * n))

        if self.dimension == 2:
            x, y = np.meshgrid(nodes_1d, nodes_1d, indexing='ij')
            nodes = np.column_stack([x.ravel(), y.ravel()])
        else:
            x, y, z = np.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing='ij')
            nodes = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        # Map to physical space
        source_min, source_max = source_bounds
        target_min, target_max = target_bounds

        physical_source = np.zeros_like(nodes)
        physical_target = np.zeros_like(nodes)

        for d in range(self.dimension):
            center_s = (source_min[d] + source_max[d]) / 2.0
            half_s = (source_max[d] - source_min[d]) / 2.0
            physical_source[:, d] = center_s + half_s * nodes[:, d]

            center_t = (target_min[d] + target_max[d]) / 2.0
            half_t = (target_max[d] - target_min[d]) / 2.0
            physical_target[:, d] = center_t + half_t * nodes[:, d]

        # Build interaction matrix
        interaction_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                interaction_matrix[i, j] = kernel_func(physical_source[j], physical_target[i])

        return interaction_matrix
