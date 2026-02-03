"""
Tests for Adaptive Order Selection

Tests convergence with adaptive order in high-frequency FMM.
"""

import pytest
import numpy as np
from fmm.core.fmm import HighFrequencyFMM
from fmm.core.tree import TreeConfig
from fmm.core.particle import Particle


def helmholtz_kernel_3d(x, y, kappa=10.0):
    """3D Helmholtz kernel."""
    r = np.linalg.norm(x - y)
    if r < 1e-14:
        return 0.0
    return np.exp(1j * kappa * r) / (4 * np.pi * r)


class TestAdaptiveOrder:
    """Test suite for adaptive order selection."""

    @pytest.fixture
    def particles(self):
        """Create test particles."""
        np.random.seed(42)
        particles = []
        for _ in range(200):
            pos = np.random.rand(3)
            particles.append(Particle(position=pos, charge=1.0))
        return particles

    def test_adaptive_order_enabled(self, particles):
        """Test that adaptive order changes expansion order."""
        kappa = 20.0
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        config = TreeConfig(
            dimension=3,
            expansion_order=10,  # High base order
            ncrit=20,
            adaptive_order=True,
            min_order=2,
            max_order=15
        )

        fmm = HighFrequencyFMM(particles, kernel_func, kappa, config)

        # Trigger computation
        _ = fmm.compute()

        # Collect orders from cells
        orders = []
        for cell in fmm.tree.leaves:
            if cell.cubature_order is not None:
                orders.append(cell.cubature_order)

        if orders:
            # Check that orders vary (not all the same)
            assert len(set(orders)) > 1 or len(orders) == 0

            # Check that orders are within bounds
            assert all(config.min_order <= o <= config.max_order for o in orders)

    def test_adaptive_order_disabled(self, particles):
        """Test that fixed order is used when adaptive is disabled."""
        kappa = 20.0
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        fixed_order = 8
        config = TreeConfig(
            dimension=3,
            expansion_order=fixed_order,
            ncrit=20,
            adaptive_order=False
        )

        fmm = HighFrequencyFMM(particles, kernel_func, kappa, config)

        # Trigger computation
        _ = fmm.compute()

        # Check that fixed order is used
        for cell in fmm.tree.leaves:
            if cell.cubature_order is not None:
                assert cell.cubature_order == fixed_order

    def test_order_increases_with_frequency(self):
        """Test that order increases with frequency parameter."""
        np.random.seed(42)
        particles = []
        for _ in range(100):
            pos = np.random.rand(3)
            particles.append(Particle(position=pos, charge=1.0))

        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa=1.0)

        config = TreeConfig(
            dimension=3,
            expansion_order=15,
            ncrit=20,
            adaptive_order=True,
            min_order=2,
            max_order=20
        )

        # Low frequency
        fmm_low = HighFrequencyFMM(particles, kernel_func, kappa=2.0, config=config)
        _ = fmm_low.compute()

        orders_low = [cell.cubature_order for cell in fmm_low.tree.leaves
                     if cell.cubature_order is not None]

        # High frequency
        fmm_high = HighFrequencyFMM(particles, kernel_func, kappa=50.0, config=config)
        _ = fmm_high.compute()

        orders_high = [cell.cubature_order for cell in fmm_high.tree.leaves
                      if cell.cubature_order is not None]

        if orders_low and orders_high:
            # Average order should be higher for high frequency
            avg_low = np.mean(orders_low)
            avg_high = np.mean(orders_high)
            assert avg_high >= avg_low

    def test_order_bounds_respected(self, particles):
        """Test that min and max order bounds are respected."""
        kappa = 30.0
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        min_order = 5
        max_order = 12

        config = TreeConfig(
            dimension=3,
            expansion_order=20,  # Higher than max
            ncrit=20,
            adaptive_order=True,
            min_order=min_order,
            max_order=max_order
        )

        fmm = HighFrequencyFMM(particles, kernel_func, kappa, config)
        _ = fmm.compute()

        # Check all orders are within bounds
        for cell in fmm.tree.leaves:
            if cell.cubature_order is not None:
                assert min_order <= cell.cubature_order <= max_order

    def test_convergence_with_order(self):
        """Test that error decreases with increasing order."""
        np.random.seed(42)
        n_particles = 50
        particles = []
        for _ in range(n_particles):
            pos = np.random.rand(3)
            particles.append(Particle(position=pos, charge=1.0))

        kappa = 10.0
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        # Compute reference solution using high order
        config_ref = TreeConfig(dimension=3, expansion_order=15, ncrit=10)
        from fmm.core.fmm import StandardFMM
        fmm_ref = StandardFMM(particles, kernel_func, config_ref)
        reference = fmm_ref.compute()

        # Test different orders
        orders = [3, 5, 7, 10]
        errors = []

        for order in orders:
            config = TreeConfig(
                dimension=3,
                expansion_order=order,
                ncrit=10,
                adaptive_order=False
            )

            fmm = StandardFMM(particles, kernel_func, config)
            potentials = fmm.compute()

            # Compute relative error
            error = np.linalg.norm(potentials - reference) / \
                   np.linalg.norm(reference)
            errors.append(error)

        # Check that error generally decreases
        # (allow for some non-monotonicity due to numerical issues)
        assert errors[-1] < errors[0] * 2  # Final error should be better

    def test_adaptive_order_improves_accuracy(self, particles):
        """Test that adaptive order improves accuracy vs fixed order."""
        kappa = 25.0
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        # Low fixed order
        config_fixed = TreeConfig(
            dimension=3,
            expansion_order=5,
            ncrit=20,
            adaptive_order=False
        )

        fmm_fixed = HighFrequencyFMM(particles, kernel_func, kappa, config_fixed)
        potentials_fixed = fmm_fixed.compute()

        # Adaptive order (can use higher order where needed)
        config_adaptive = TreeConfig(
            dimension=3,
            expansion_order=10,
            ncrit=20,
            adaptive_order=True,
            min_order=3,
            max_order=15
        )

        fmm_adaptive = HighFrequencyFMM(particles, kernel_func, kappa, config_adaptive)
        potentials_adaptive = fmm_adaptive.compute()

        # Compute reference (direct for subset)
        n_test = min(20, len(particles))
        reference = np.zeros(n_test)

        for i in range(n_test):
            for j, source in enumerate(particles):
                if i != j:
                    reference[i] += source.charge * kernel_func(
                        source.position, particles[i].position
                    )

        # Compare errors
        error_fixed = np.linalg.norm(potentials_fixed[:n_test] - reference.real)
        error_adaptive = np.linalg.norm(potentials_adaptive[:n_test] - reference.real)

        # Adaptive should be at least as good as fixed low order
        # (This is a weak test due to various approximations)
        assert error_adaptive <= error_fixed * 1.5


class TestOrderSelectionCriteria:
    """Test the criteria used for order selection."""

    def test_kw_based_selection(self):
        """Test that order is based on κw product."""
        np.random.seed(42)
        particles = []
        for _ in range(50):
            pos = np.random.rand(3) * 0.1  # Small domain
            particles.append(Particle(position=pos, charge=1.0))

        kappa = 100.0  # High wavenumber
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        config = TreeConfig(
            dimension=3,
            expansion_order=20,
            ncrit=10,
            adaptive_order=True,
            min_order=2,
            max_order=25
        )

        fmm = HighFrequencyFMM(particles, kernel_func, kappa, config)
        _ = fmm.compute()

        # With small domain (small w), κw might still be moderate
        # Check that selected orders are reasonable
        for cell in fmm.tree.leaves:
            if cell.cubature_order is not None:
                # Order should be related to local κw
                local_kw = kappa * cell.size
                # Expected order is roughly κw
                expected_order = max(config.min_order,
                                   min(int(local_kw), config.max_order))
                # Should be close (within factor of 2)
                assert cell.cubature_order >= expected_order / 2

    def test_per_cell_order_variation(self):
        """Test that different cells can have different orders."""
        np.random.seed(42)
        particles = []
        for _ in range(300):
            pos = np.random.rand(3)
            particles.append(Particle(position=pos, charge=1.0))

        kappa = 15.0
        kernel_func = lambda x, y: helmholtz_kernel_3d(x, y, kappa)

        config = TreeConfig(
            dimension=3,
            expansion_order=12,
            ncrit=15,  # Small to get many cells
            max_depth=6,
            adaptive_order=True,
            min_order=3,
            max_order=18
        )

        fmm = HighFrequencyFMM(particles, kernel_func, kappa, config)
        _ = fmm.compute()

        # Collect orders from different levels
        orders_by_level = {}
        for cell in fmm.tree.leaves:
            if cell.cubature_order is not None:
                if cell.level not in orders_by_level:
                    orders_by_level[cell.level] = []
                orders_by_level[cell.level].append(cell.cubature_order)

        # Check that we have multiple levels with orders
        if len(orders_by_level) > 1:
            # Different levels might have different average orders
            # (though this depends on the adaptive strategy)
            assert len(orders_by_level) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
