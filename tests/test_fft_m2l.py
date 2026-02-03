"""
Tests for FFT-Based M2L

Tests the FFT-based M2L operator for kernel-independent FMM.
Verifies FFT-M2L vs SVD-M2L comparison.
"""

import pytest
import numpy as np
from fmm.core.kernel_independent import FFTBasedM2L, SVDM2LCompressor, ACACompressor


def laplace_kernel_3d(x, y):
    """3D Laplace kernel: G(x,y) = 1 / |x-y|"""
    r = np.linalg.norm(x - y)
    if r < 1e-14:
        return 0.0
    return 1.0 / r


def helmholtz_kernel_3d(x, y, kappa=10.0):
    """3D Helmholtz kernel: G(x,y) = e^{iκ|x-y|} / (4π|x-y|)"""
    r = np.linalg.norm(x - y)
    if r < 1e-14:
        return 0.0
    return np.exp(1j * kappa * r) / (4 * np.pi * r)


class TestFFTBasedM2L:
    """Test suite for FFT-based M2L operator."""

    @pytest.fixture
    def fft_m2l_2d(self):
        """FFT-based M2L for 2D."""
        return FFTBasedM2L(order=4, dimension=2)

    @pytest.fixture
    def fft_m2l_3d(self):
        """FFT-based M2L for 3D."""
        return FFTBasedM2L(order=4, dimension=3)

    def test_translation_invariant_detection(self, fft_m2l_3d):
        """Test detection of translation-invariant kernels."""
        # Laplace is translation-invariant
        assert fft_m2l_3d._is_translation_invariant(laplace_kernel_3d)

        # Create a non-translation-invariant kernel
        def non_translation_invariant(x, y):
            return np.dot(x, y) / (np.linalg.norm(x - y) + 1e-14)

        assert not fft_m2l_3d._is_translation_invariant(non_translation_invariant)

    def test_fft_kernel_computation_2d(self, fft_m2l_2d):
        """Test FFT kernel computation for 2D."""
        source_min = np.array([0.0, 0.0])
        source_max = np.array([1.0, 1.0])
        source_bounds = (source_min, source_max)

        target_min = np.array([1.5, 1.5])
        target_max = np.array([2.5, 2.5])
        target_bounds = (target_min, target_max)

        fft_kernel = fft_m2l_2d._compute_fft_kernel(
            laplace_kernel_3d, source_bounds, target_bounds
        )

        # Check shape
        fft_size = 2 * fft_m2l_2d.order - 1
        assert fft_kernel.shape == (fft_size, fft_size)

    def test_fft_kernel_computation_3d(self, fft_m2l_3d):
        """Test FFT kernel computation for 3D."""
        source_min = np.array([0.0, 0.0, 0.0])
        source_max = np.array([1.0, 1.0, 1.0])
        source_bounds = (source_min, source_max)

        target_min = np.array([1.5, 1.5, 1.5])
        target_max = np.array([2.5, 2.5, 2.5])
        target_bounds = (target_min, target_max)

        fft_kernel = fft_m2l_3d._compute_fft_kernel(
            laplace_kernel_3d, source_bounds, target_bounds
        )

        # Check shape
        fft_size = 2 * fft_m2l_3d.order - 1
        assert fft_kernel.shape == (fft_size, fft_size, fft_size)

    def test_fft_m2l_apply_2d(self, fft_m2l_2d):
        """Test FFT-based M2L application in 2D."""
        # Create source coefficients
        n = fft_m2l_2d.order
        source_coeffs = np.random.randn(n * n)

        source_min = np.array([0.0, 0.0])
        source_max = np.array([1.0, 1.0])
        source_bounds = (source_min, source_max)

        target_min = np.array([1.5, 1.5])
        target_max = np.array([2.5, 2.5])
        target_bounds = (target_min, target_max)

        target_coeffs = fft_m2l_2d.apply(
            source_coeffs, laplace_kernel_3d, source_bounds, target_bounds
        )

        # Check shape
        assert target_coeffs.shape == source_coeffs.shape

        # Check that result is different from input
        assert not np.allclose(target_coeffs, source_coeffs)

    def test_fft_m2l_apply_3d(self, fft_m2l_3d):
        """Test FFT-based M2L application in 3D."""
        # Create source coefficients
        n = fft_m2l_3d.order
        source_coeffs = np.random.randn(n * n * n)

        source_min = np.array([0.0, 0.0, 0.0])
        source_max = np.array([1.0, 1.0, 1.0])
        source_bounds = (source_min, source_max)

        target_min = np.array([1.5, 1.5, 1.5])
        target_max = np.array([2.5, 2.5, 2.5])
        target_bounds = (target_min, target_max)

        target_coeffs = fft_m2l_3d.apply(
            source_coeffs, laplace_kernel_3d, source_bounds, target_bounds
        )

        # Check shape
        assert target_coeffs.shape == source_coeffs.shape

    def test_non_translation_invariant_raises_error(self, fft_m2l_3d):
        """Test that non-translation-invariant kernel raises error."""
        def non_inv_kernel(x, y):
            return np.dot(x, y)

        source_coeffs = np.ones(fft_m2l_3d.order ** 3)
        source_min = np.array([0.0, 0.0, 0.0])
        source_max = np.array([1.0, 1.0, 1.0])
        source_bounds = (source_min, source_max)

        target_min = np.array([1.5, 1.5, 1.5])
        target_max = np.array([2.5, 2.5, 2.5])
        target_bounds = (target_min, target_max)

        with pytest.raises(ValueError, match="translation-invariant"):
            fft_m2l_3d.apply(
                source_coeffs, non_inv_kernel, source_bounds, target_bounds
            )

    def test_kernel_caching(self, fft_m2l_3d):
        """Test that FFT kernels are cached."""
        source_min = np.array([0.0, 0.0, 0.0])
        source_max = np.array([1.0, 1.0, 1.0])
        source_bounds = (source_min, source_max)

        target_min = np.array([1.5, 1.5, 1.5])
        target_max = np.array([2.5, 2.5, 2.5])
        target_bounds = (target_min, target_max)

        # First call computes and caches
        _ = fft_m2l_3d._compute_fft_kernel(
            laplace_kernel_3d, source_bounds, target_bounds
        )

        # Check cache
        cache_key = tuple(np.round(target_min, 6)) + tuple(np.round(target_max, 6))
        assert cache_key in fft_m2l_3d._kernel_cache


class TestFFTvsSVD:
    """Compare FFT-based M2L with SVD-based M2L."""

    @pytest.fixture
    def interaction_matrix(self):
        """Create a test interaction matrix."""
        n = 8
        # Create a low-rank matrix
        np.random.seed(42)
        U = np.random.randn(n, 3)
        V = np.random.randn(n, 3)
        A = U @ V.T
        return A

    def test_svd_compression(self, interaction_matrix):
        """Test SVD compression."""
        compressor = SVDM2LCompressor(tolerance=1e-6)

        U, S, Vt = compressor.compress(interaction_matrix)

        # Check shapes
        rank = len(S)
        assert U.shape[1] == rank
        assert Vt.shape[0] == rank

        # Check reconstruction error
        reconstructed = U @ np.diag(S) @ Vt
        relative_error = np.linalg.norm(reconstructed - interaction_matrix) / \
                        np.linalg.norm(interaction_matrix)

        assert relative_error < 1e-3  # Should be very accurate

    def test_aca_compression(self, interaction_matrix):
        """Test ACA compression."""
        compressor = ACACompressor(tolerance=1e-6, max_rank=10)

        U, V = compressor.compress(interaction_matrix)

        # Check shapes
        rank = U.shape[1]
        assert V.shape[1] == rank
        assert U.shape[0] == interaction_matrix.shape[0]
        assert V.shape[0] == interaction_matrix.shape[1]

        # Check reconstruction error
        reconstructed = U @ V.T
        relative_error = np.linalg.norm(reconstructed - interaction_matrix) / \
                        np.linalg.norm(interaction_matrix)

        # ACA should give reasonable approximation
        assert relative_error < 0.5

    def test_fft_vs_svd_accuracy(self):
        """Compare FFT and SVD methods on translation-invariant kernel."""
        order = 4
        fft_m2l = FFTBasedM2L(order, dimension=2)
        svd_compressor = SVDM2LCompressor(tolerance=1e-6)

        # Create source coefficients
        source_coeffs = np.random.randn(order * order)

        source_min = np.array([0.0, 0.0])
        source_max = np.array([1.0, 1.0])
        source_bounds = (source_min, source_max)

        target_min = np.array([1.5, 1.5])
        target_max = np.array([2.5, 2.5])
        target_bounds = (target_min, target_max)

        # FFT-based M2L
        target_fft = fft_m2l.apply(
            source_coeffs, laplace_kernel_3d, source_bounds, target_bounds
        )

        # Build interaction matrix and use SVD
        n = order
        interaction_matrix = np.zeros((n * n, n * n))

        # Generate Chebyshev nodes
        k = np.arange(1, n + 1)
        nodes_1d = np.cos((2 * k - 1) * np.pi / (2 * n))
        x, y = np.meshgrid(nodes_1d, nodes_1d, indexing='ij')
        nodes = np.column_stack([x.ravel(), y.ravel()])

        source_nodes = source_min + (source_max - source_min) * (nodes + 1) / 2
        target_nodes = target_min + (target_max - target_min) * (nodes + 1) / 2

        for i, t_node in enumerate(target_nodes):
            for j, s_node in enumerate(source_nodes):
                interaction_matrix[i, j] = laplace_kernel_3d(s_node, t_node)

        U, S, Vt = svd_compressor.compress(interaction_matrix)
        target_svd = U @ (S * (Vt @ source_coeffs))

        # Compare results
        # Note: FFT and SVD won't be exactly equal due to different approximations
        # but should be in the same ballpark for smooth kernels
        correlation = np.corrcoef(target_fft, target_svd)[0, 1]

        # Should have reasonable correlation
        assert correlation > 0.5


class TestComplexityScaling:
    """Test complexity scaling of different M2L methods."""

    def test_fft_complexity(self):
        """Test FFT complexity O(L^d log L)."""
        import time

        orders = [4, 8, 12]
        times = []

        for order in orders:
            fft_m2l = FFTBasedM2L(order, dimension=2)
            source_coeffs = np.random.randn(order * order)

            source_min = np.array([0.0, 0.0])
            source_max = np.array([1.0, 1.0])
            source_bounds = (source_min, source_max)

            target_min = np.array([1.5, 1.5])
            target_max = np.array([2.5, 2.5])
            target_bounds = (target_min, target_max)

            start = time.time()
            for _ in range(10):
                fft_m2l.apply(source_coeffs, laplace_kernel_3d,
                            source_bounds, target_bounds)
            elapsed = time.time() - start
            times.append(elapsed / 10)

        # Check that time scales roughly as L^d log L
        # For 2D: L² log L
        for i in range(len(orders) - 1):
            ratio = times[i+1] / times[i]
            L_ratio = (orders[i+1] / orders[i]) ** 2 * \
                     np.log(orders[i+1]) / np.log(orders[i])

            # Should be within factor of 4
            assert ratio < 4 * L_ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
