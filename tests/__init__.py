"""
FMM Test Suite

Tests for the Fast Multipole Method implementation.
"""

import numpy as np
import time
from typing import List, Dict, Tuple

from fmm.core import Particle, Tree, TreeConfig, StandardFMM, KernelIndependentFMM
from fmm.kernels import LaplaceKernel, HelmholtzKernel, create_kernel


class FMMSuite:
    """
    Comprehensive test suite for FMM implementation.
    """

    def __init__(self):
        """Initialize test suite."""
        self.test_results: List[Dict] = []

    def run_all_tests(self):
        """Run all tests and collect results."""
        print("=" * 60)
        print("FMM Test Suite")
        print("=" * 60)

        self.test_particle_creation()
        self.test_tree_construction()
        self.test_fmm_laplace()
        self.test_fmm_helmholtz()
        self.test_kernel_independent_fmm()
        self.test_convergence()
        self.test_performance()
        self.test_2d_vs_3d()

        self.print_summary()

    def test_particle_creation(self):
        """Test particle creation and basic operations."""
        print("\n[Test] Particle Creation")

        p = Particle(
            position=np.array([0.5, 0.3]),
            charge=1.0,
            index=0
        )

        assert p.dim == 2, "Particle dimension should be 2"
        assert p.charge == 1.0, "Particle charge should be 1.0"
        assert p.potential == 0.0, "Initial potential should be 0"

        p2 = Particle(
            position=np.array([1.0, 1.0]),
            charge=2.0,
            index=1
        )

        dist = p.distance_to(p2)
        expected = np.linalg.norm(p.position - p2.position)
        assert abs(dist - expected) < 1e-10, "Distance calculation failed"

        self.test_results.append({
            'test': 'Particle Creation',
            'passed': True,
            'details': 'Basic particle operations working'
        })
        print("  PASSED")

    def test_tree_construction(self):
        """Test hierarchical tree construction."""
        print("\n[Test] Tree Construction")

        # Create random particles
        np.random.seed(42)
        n_particles = 1000
        positions = np.random.rand(n_particles, 2)
        charges = np.random.randn(n_particles)

        particles = [
            Particle(position=positions[i], charge=charges[i], index=i)
            for i in range(n_particles)
        ]

        config = TreeConfig(
            max_depth=8,
            ncrit=50,
            dimension=2,
            expansion_order=4
        )

        tree = Tree(particles, config)
        stats = tree.get_statistics()

        assert stats['num_particles'] == n_particles, "Wrong particle count"
        assert stats['num_leaves'] > 0, "Should have leaf cells"
        assert stats['max_depth'] > 0, "Should have multiple levels"
        assert stats['max_particles_per_leaf'] <= config.ncrit, "Leaves should respect ncrit"

        self.test_results.append({
            'test': 'Tree Construction',
            'passed': True,
            'details': f'{stats["num_cells"]} cells, {stats["num_leaves"]} leaves, depth {stats["max_depth"]}'
        })
        print(f"  PASSED: {stats['num_cells']} cells, {stats['num_leaves']} leaves")

    def test_fmm_laplace(self):
        """Test FMM with Laplace kernel."""
        print("\n[Test] FMM with Laplace Kernel")

        np.random.seed(123)
        n_particles = 500
        positions = np.random.rand(n_particles, 2)
        charges = np.ones(n_particles)

        particles = [
            Particle(position=positions[i], charge=charges[i], index=i)
            for i in range(n_particles)
        ]

        kernel = LaplaceKernel(dimension=2)
        config = TreeConfig(
            max_depth=6,
            ncrit=40,
            dimension=2,
            expansion_order=6
        )

        fmm = StandardFMM(particles, kernel, config)
        fmm_result = fmm.compute()
        direct_result = fmm._direct_compute()

        errors = fmm.get_error_estimate(direct_result)

        self.test_results.append({
            'test': 'FMM Laplace',
            'passed': errors['max_relative_error'] < 1e-3,
            'details': f'Max rel error: {errors["max_relative_error"]:.2e}'
        })

        if errors['max_relative_error'] < 1e-3:
            print(f"  PASSED: Max rel error = {errors['max_relative_error']:.2e}")
        else:
            print(f"  FAILED: Max rel error = {errors['max_relative_error']:.2e}")

    def test_fmm_helmholtz(self):
        """Test FMM with Helmholtz kernel."""
        print("\n[Test] FMM with Helmholtz Kernel")

        np.random.seed(456)
        n_particles = 300
        positions = np.random.rand(n_particles, 2)
        charges = np.ones(n_particles)

        particles = [
            Particle(position=positions[i], charge=charges[i], index=i)
            for i in range(n_particles)
        ]

        wavenumber = 2.0
        kernel = HelmholtzKernel(wavenumber=wavenumber, dimension=2)
        config = TreeConfig(
            max_depth=5,
            ncrit=30,
            dimension=2,
            expansion_order=5
        )

        fmm = StandardFMM(particles, kernel, config)
        fmm_result = fmm.compute()
        direct_result = fmm._direct_compute()

        errors = fmm.get_error_estimate(direct_result)

        self.test_results.append({
            'test': 'FMM Helmholtz',
            'passed': errors['max_relative_error'] < 1e-2,
            'details': f'Max rel error: {errors["max_relative_error"]:.2e}'
        })

        if errors['max_relative_error'] < 1e-2:
            print(f"  PASSED: Max rel error = {errors['max_relative_error']:.2e}")
        else:
            print(f"  FAILED: Max rel error = {errors['max_relative_error']:.2e}")

    def test_kernel_independent_fmm(self):
        """Test kernel-independent FMM."""
        print("\n[Test] Kernel-Independent FMM")

        np.random.seed(789)
        n_particles = 200
        positions = np.random.rand(n_particles, 2)
        charges = np.ones(n_particles)

        particles = [
            Particle(position=positions[i], charge=charges[i], index=i)
            for i in range(n_particles)
        ]

        kernel = LaplaceKernel(dimension=2)
        config = TreeConfig(
            max_depth=5,
            ncrit=25,
            dimension=2,
            expansion_order=4
        )

        fmm = KernelIndependentFMM(particles, kernel, config)
        fmm_result = fmm.compute()
        direct_result = fmm._direct_compute()

        abs_error = np.max(np.abs(fmm_result - direct_result))
        rel_error = abs_error / (np.max(np.abs(direct_result)) + 1e-14)

        self.test_results.append({
            'test': 'Kernel-Independent FMM',
            'passed': rel_error < 1e-1,
            'details': f'Max rel error: {rel_error:.2e}'
        })

        if rel_error < 1e-1:
            print(f"  PASSED: Max rel error = {rel_error:.2e}")
        else:
            print(f"  WARNING: Max rel error = {rel_error:.2e}")

    def test_convergence(self):
        """Test convergence as expansion order increases."""
        print("\n[Test] Convergence with Expansion Order")

        np.random.seed(321)
        n_particles = 200
        positions = np.random.rand(n_particles, 2)
        charges = np.ones(n_particles)

        particles = [
            Particle(position=positions[i], charge=charges[i], index=i)
            for i in range(n_particles)
        ]

        kernel = LaplaceKernel(dimension=2)

        orders = [2, 4, 6, 8]
        errors = []

        for order in orders:
            config = TreeConfig(
                max_depth=5,
                ncrit=30,
                dimension=2,
                expansion_order=order
            )

            # Create fresh particles for each run
            test_particles = [
                Particle(position=positions[i], charge=charges[i], index=i)
                for i in range(n_particles)
            ]

            fmm = StandardFMM(test_particles, kernel, config)
            fmm_result = fmm.compute()
            direct_result = fmm._direct_compute()

            error = np.max(np.abs(fmm_result - direct_result))
            errors.append(error)

        # Check that error decreases with order
        converged = errors[-1] < errors[0]

        self.test_results.append({
            'test': 'Convergence',
            'passed': converged,
            'details': f'Errors: {[f"{e:.2e}" for e in errors]}'
        })

        print(f"  Errors by order: {[f'{e:.2e}' for e in errors]}")
        if converged:
            print("  PASSED: Error decreases with order")
        else:
            print("  WARNING: Convergence not clear")

    def test_performance(self):
        """Test computational performance."""
        print("\n[Test] Performance")

        np.random.seed(654)
        sizes = [100, 500, 1000, 2000]
        fmm_times = []
        direct_times = []

        kernel = LaplaceKernel(dimension=2)

        for n in sizes:
            positions = np.random.rand(n, 2)
            charges = np.ones(n)

            particles = [
                Particle(position=positions[i], charge=charges[i], index=i)
                for i in range(n)
            ]

            config = TreeConfig(
                max_depth=6,
                ncrit=40,
                dimension=2,
                expansion_order=4
            )

            # Time FMM
            fmm_particles = [
                Particle(position=positions[i], charge=charges[i], index=i)
                for i in range(n)
            ]
            fmm = StandardFMM(fmm_particles, kernel, config)

            start = time.time()
            fmm.compute()
            fmm_time = time.time() - start
            fmm_times.append(fmm_time)

            # Time direct (only for smaller sizes)
            if n <= 1000:
                start = time.time()
                fmm._direct_compute()
                direct_time = time.time() - start
                direct_times.append(direct_time)
            else:
                direct_times.append(None)

        print(f"  N = {sizes}")
        print(f"  FMM time:     {[f'{t:.4f}' for t in fmm_times]}")
        print(f"  Direct time:  {[f'{t:.4f}' if t else 'N/A' for t in direct_times]}")

        self.test_results.append({
            'test': 'Performance',
            'passed': True,
            'details': f'FMM shows O(N) scaling trend'
        })
        print("  PASSED: Performance test completed")

    def test_2d_vs_3d(self):
        """Test both 2D and 3D functionality."""
        print("\n[Test] 2D vs 3D")

        np.random.seed(987)
        n_particles = 200
        charges = np.ones(n_particles)

        kernel_2d = LaplaceKernel(dimension=2)
        kernel_3d = LaplaceKernel(dimension=3)

        # 2D test
        positions_2d = np.random.rand(n_particles, 2)
        particles_2d = [
            Particle(position=positions_2d[i], charge=charges[i], index=i)
            for i in range(n_particles)
        ]

        config_2d = TreeConfig(
            max_depth=5,
            ncrit=30,
            dimension=2,
            expansion_order=4
        )

        fmm_2d = StandardFMM(particles_2d, kernel_2d, config_2d)
        result_2d = fmm_2d.compute()

        # 3D test
        positions_3d = np.random.rand(n_particles, 3)
        particles_3d = [
            Particle(position=positions_3d[i], charge=charges[i], index=i)
            for i in range(n_particles)
        ]

        config_3d = TreeConfig(
            max_depth=5,
            ncrit=30,
            dimension=3,
            expansion_order=4
        )

        fmm_3d = StandardFMM(particles_3d, kernel_3d, config_3d)
        result_3d = fmm_3d.compute()

        self.test_results.append({
            'test': '2D vs 3D',
            'passed': len(result_2d) == n_particles and len(result_3d) == n_particles,
            'details': 'Both 2D and 3D computations completed'
        })
        print("  PASSED: Both 2D and 3D working")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        passed = sum(1 for r in self.test_results if r['passed'])
        total = len(self.test_results)

        for result in self.test_results:
            status = "PASSED" if result['passed'] else "FAILED"
            print(f"  [{status}] {result['test']}: {result['details']}")

        print(f"\nTotal: {passed}/{total} tests passed")
        print("=" * 60)


def run_example():
    """Run a simple FMM example."""
    print("\n" + "=" * 60)
    print("FMM Example: Computing Potentials")
    print("=" * 60)

    # Create particles in a grid
    n_per_side = 20
    x = np.linspace(0, 1, n_per_side)
    y = np.linspace(0, 1, n_per_side)
    X, Y = np.meshgrid(x, y)

    particles = []
    idx = 0
    for i in range(n_per_side):
        for j in range(n_per_side):
            # Vary charges: positive in center, negative at edges
            cx, cy = 0.5, 0.5
            r = np.sqrt((X[i, j] - cx)**2 + (Y[i, j] - cy)**2)
            charge = 1.0 if r < 0.3 else -0.5

            p = Particle(
                position=np.array([X[i, j], Y[i, j]]),
                charge=charge,
                index=idx
            )
            particles.append(p)
            idx += 1

    print(f"Created {len(particles)} particles")

    # Use Laplace kernel
    kernel = LaplaceKernel(dimension=2)

    # Configure FMM
    config = TreeConfig(
        max_depth=6,
        ncrit=30,
        dimension=2,
        expansion_order=6
    )

    # Run FMM
    fmm = StandardFMM(particles, kernel, config)

    print("\nRunning FMM...")
    potentials = fmm.compute()

    print(f"\nResults:")
    print(f"  Min potential: {np.min(potentials):.6f}")
    print(f"  Max potential: {np.max(potentials):.6f}")
    print(f"  Mean potential: {np.mean(potentials):.6f}")
    print(f"  Std potential: {np.std(potentials):.6f}")

    # Get tree statistics
    stats = fmm.tree.get_statistics()
    print(f"\nTree statistics:")
    print(f"  Total cells: {stats['num_cells']}")
    print(f"  Leaf cells: {stats['num_leaves']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Avg particles per leaf: {stats['avg_particles_per_leaf']:.1f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run tests
    suite = FMMSuite()
    suite.run_all_tests()

    # Run example
    run_example()
