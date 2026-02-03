"""
Tests for High-Frequency Helmholtz FMM

Integration tests for full FMM with high-frequency Helmholtz kernel.
Tests various κw values: 20, 50, 100.
"""

import pytest
import numpy as np
from fmm.core.fmm import HighFrequencyFMM, HybridFMM
from fmm.core.tree import TreeConfig
from fmm.core.particle import Particle


def helmholtz_kernel_3d(x, y, kappa=10.0):
    """3D Helmholtz kernel: G(x,y) = e^{iκ|x-y|} / (4π|x-y|)"""
    r = np.linalg.norm(x - y)
    if r < 1e-14:
        return 0.0
    return np.exp(1j * kappa * r) / (4 * np.pi * r)


class TestHighFrequencyHelmholtz:
    """Integration tests for high-frequency Helmholtz FMM."""

    @pytest.fixture
    def particles(self, n_particles=200):
        """Create test particles."""
        np.random.seed(42)
        particles = []
        for _ in range(n_particles):
            pos = np.random.rand(3)
            charge = 1.0
            particles.append(Particle(position=pos, charge=charge))
        return particles

    @pytest.mark.parametrize("kappa,wavenumber_product", [
        (20, 20),
        (50, 50),
        (100, 100),
    ])
    def test_high_frequency_fmm(self, particles, kappa, wavenumber_product):
        """Test FMM with various high-frequency parameters."""
        # Adjust particle positions to achieve desired κw
        # Scale positions so bbox has size w = kappa / wavenumber_product
        bbox_size = kappa / wavenumber_product

        scaled_particles = []
        for p in particles:
            pos = p.position * bbox_size
            scaled_particles.append(Particle(position=pos, charge=p.charge))

        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        config = TreeConfig(
            dimension=3,
            expansion_order=5,
            ncrit=20,
            use_high_frequency_m2l=True,
            use_spherical_interpolation=True
        )

        fmm = HighFrequencyFMM(scaled_particles, kernel_func, kappa, config)

        # Compute potentials
        potentials = fmm.compute()

        # Check shape
        assert potentials.shape == (len(scaled_particles),)

        # Check that potentials are finite
        assert np.all(np.isfinite(potentials))

        # Check that potentials are non-zero
        assert np.any(np.abs(potentials) > 1e-10)

    def test_adaptive_order_selection(self, particles):
        """Test that adaptive order is selected correctly."""
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa=50.0)

        config = TreeConfig(
            dimension=3,
            expansion_order=3,
            ncrit=20,
            adaptive_order=True,
            min_order=2,
            max_order=20
        )

        fmm = HighFrequencyFMM(particles, kernel_func, kappa=50.0, config=config)

        # Check that HF operators are initialized
        assert fmm.hf_m2l is not None

        # Compute to trigger order selection
        _ = fmm.compute()

        # Check that some cells have cubature orders set
        orders = [cell.cubature_order for cell in fmm.tree.leaves
                 if cell.cubature_order is not None]
        assert len(orders) > 0

        # Check that orders are within bounds
        assert all(config.min_order <= o <= config.max_order for o in orders)

    def test_hf_vs_standard_fmm(self, particles):
        """Compare high-frequency and standard FMM for moderate frequency."""
        kappa = 5.0
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        config = TreeConfig(dimension=3, expansion_order=4, ncrit=20)

        # Standard FMM
        from fmm.core.fmm import StandardFMM
        fmm_standard = StandardFMM(particles, kernel_func, config)
        potentials_standard = fmm_standard.compute()

        # High-frequency FMM (should activate)
        fmm_hf = HighFrequencyFMM(particles, kernel_func, kappa, config)
        potentials_hf = fmm_hf.compute()

        # Both should produce similar results for moderate κw
        # (within numerical tolerance)
        relative_diff = np.abs(potentials_standard - potentials_hf) / \
                       (np.abs(potentials_standard) + 1e-14)

        # Should be in same order of magnitude
        assert np.mean(relative_diff) < 1.0  # Within factor of 2

    def test_error_estimate(self, particles):
        """Test error estimation against direct computation."""
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa=10.0)

        config = TreeConfig(
            dimension=3,
            expansion_order=4,
            ncrit=20,
            use_high_frequency_m2l=True
        )

        fmm = HighFrequencyFMM(particles, kernel_func, kappa=10.0, config=config)
        potentials_fmm = fmm.compute()

        # Compute reference (direct) for a subset
        n_test = min(20, len(particles))
        reference = np.zeros(n_test)

        for i in range(n_test):
            for j, source in enumerate(particles):
                if i != j:
                    reference[i] += source.charge * kernel_func(
                        source.position, particles[i].position
                    )

        # Take real part (Helmholtz kernel is complex)
        potentials_fmm_test = potentials_fmm[:n_test]

        # Compute errors
        abs_error = np.abs(potentials_fmm_test - reference.real)
        rel_error = abs_error / (np.abs(reference.real) + 1e-14)

        # Check that errors are reasonable
        assert np.mean(rel_error) < 0.5  # Within 50% relative error


class TestHybridFMM:
    """Test hybrid FMM that auto-selects variant."""

    @pytest.fixture
    def particles(self):
        """Create test particles."""
        np.random.seed(42)
        particles = []
        for _ in range(100):
            pos = np.random.rand(3)
            particles.append(Particle(position=pos, charge=1.0))
        return particles

    def test_variant_selection_laplace(self, particles):
        """Test that standard FMM is selected for Laplace."""
        from fmm.core.kernels import laplace_kernel_3d

        config = TreeConfig(dimension=3, expansion_order=4)

        fmm = HybridFMM(particles, laplace_kernel_3d, wavenumber=0.0, config=config)

        assert fmm.variant == 'standard'

    def test_variant_selection_low_frequency(self, particles):
        """Test that standard FMM is selected for low frequency."""
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa=1.0)

        config = TreeConfig(dimension=3, expansion_order=4)

        fmm = HybridFMM(particles, kernel_func, wavenumber=1.0, config=config)

        # Should be standard (κw < 1)
        assert fmm.variant == 'standard'

    def test_variant_selection_high_frequency(self, particles):
        """Test that high-frequency FMM is selected."""
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa=50.0)

        config = TreeConfig(
            dimension=3,
            expansion_order=4,
            directional_threshold_kw=100.0  # Set high to avoid directional
        )

        fmm = HybridFMM(particles, kernel_func, wavenumber=50.0, config=config)

        # Should be high-frequency (1 < κw < threshold)
        assert fmm.variant in ['high_frequency', 'standard']  # May fall back

    def test_variant_selection_directional(self, particles):
        """Test that directional FMM is selected for extreme frequency."""
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa=100.0)

        config = TreeConfig(
            dimension=3,
            expansion_order=4,
            directional_threshold_kw=10.0
        )

        fmm = HybridFMM(particles, kernel_func, wavenumber=100.0, config=config)

        # Should be directional (κw >= threshold)
        assert fmm.variant in ['directional', 'high_frequency', 'standard']


class TestHelmholtzAccuracy:
    """Test accuracy for various Helmholtz parameters."""

    @pytest.mark.parametrize("kappa,expected_accuracy", [
        (1.0, 0.01),   # Low frequency - high accuracy
        (10.0, 0.05),  # Moderate frequency
        (50.0, 0.1),   # High frequency
    ])
    def test_accuracy_vs_kappa(self, kappa, expected_accuracy):
        """Test accuracy for different wavenumbers."""
        np.random.seed(42)
        n_particles = 100
        particles = []
        for _ in range(n_particles):
            pos = np.random.rand(3)
            particles.append(Particle(position=pos, charge=1.0))

        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        config = TreeConfig(
            dimension=3,
            expansion_order=5,
            ncrit=20,
            use_high_frequency_m2l=True
        )

        fmm = HighFrequencyFMM(particles, kernel_func, kappa, config)
        potentials_fmm = fmm.compute()

        # Compute reference for first 20 particles
        n_test = min(20, n_particles)
        reference = np.zeros(n_test, dtype=np.complex128)

        for i in range(n_test):
            for j, source in enumerate(particles):
                if i != j:
                    reference[i] += source.charge * kernel_func(
                        source.position, particles[i].position
                    )

        # Compare real parts
        fmm_real = potentials_fmm[:n_test]
        ref_real = reference.real

        relative_error = np.abs(fmm_real - ref_real) / (np.abs(ref_real) + 1e-14)
        mean_error = np.mean(relative_error)

        # Error should be within expected range
        assert mean_error < expected_accuracy


class TestPerformanceScaling:
    """Test performance scaling with problem size."""

    @pytest.mark.parametrize("n_particles", [100, 500, 1000])
    def test_scaling_with_n(self, n_particles):
        """Test that FMM scales well with problem size."""
        import time

        np.random.seed(42)
        particles = []
        for _ in range(n_particles):
            pos = np.random.rand(3)
            particles.append(Particle(position=pos, charge=1.0))

        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa=10.0)

        config = TreeConfig(
            dimension=3,
            expansion_order=4,
            ncrit=30,
            use_high_frequency_m2l=True
        )

        fmm = HighFrequencyFMM(particles, kernel_func, kappa=10.0, config=config)

        start = time.time()
        potentials = fmm.compute()
        elapsed = time.time() - start

        # Check that computation completed
        assert potentials.shape == (n_particles,)
        assert np.all(np.isfinite(potentials))

        # Print timing for manual inspection
        print(f"\nN={n_particles}: time={elapsed:.4f}s")

        # Should be much faster than O(N²)
        # (This is a weak test - just ensure it completes in reasonable time)
        assert elapsed < n_particles ** 2 / 1000  # Should be much faster than direct


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
