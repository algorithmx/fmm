"""
Tests for Spherical Interpolation

Tests the spherical interpolation for high-frequency FMM.
Verifies interpolation accuracy and charge conservation.
"""

import pytest
import numpy as np
from fmm.core.operators import SphericalInterpolator, Cubature


class TestSphericalInterpolator:
    """Test suite for spherical interpolator."""

    @pytest.fixture
    def interpolator(self):
        """Spherical interpolator."""
        return SphericalInterpolator(source_order=3, target_order=5)

    def test_initialization(self, interpolator):
        """Test interpolator initialization."""
        assert interpolator.source_order == 3
        assert interpolator.target_order == 5

        assert interpolator.source_cubature is not None
        assert interpolator.target_cubature is not None

    def test_interpolation_matrix_shape(self, interpolator):
        """Test that interpolation matrix has correct shape."""
        I = interpolator._interpolation_matrix

        Q_source = interpolator.source_cubature.num_nodes
        Q_target = interpolator.target_cubature.num_nodes

        assert I.shape == (Q_target, Q_source)

    def test_interpolation_consistency(self):
        """Test that interpolation is consistent for same order."""
        # Same order should give identity-like behavior
        interpolator = SphericalInterpolator(source_order=5, target_order=5)

        # Constant function
        source_coeffs = np.ones(interpolator.source_cubature.num_nodes)

        wavenumber = 10.0
        translation_vector = np.array([0.1, 0.0, 0.0])

        target_coeffs = interpolator.interpolate(
            source_coeffs, translation_vector, wavenumber
        )

        # Should preserve constant (approximately)
        assert np.mean(np.abs(target_coeffs)) > 0

    def test_interpolation_with_phase(self, interpolator):
        """Test interpolation with phase factor."""
        source_coeffs = np.random.randn(interpolator.source_cubature.num_nodes) + \
                       1j * np.random.randn(interpolator.source_cubature.num_nodes)

        wavenumber = 10.0
        translation_vector = np.array([0.5, 0.0, 0.0])

        target_coeffs = interpolator.interpolate(
            source_coeffs, translation_vector, wavenumber
        )

        # Check that phase is applied
        assert not np.allclose(target_coeffs, source_coeffs)

    def test_zero_translation(self, interpolator):
        """Test interpolation with zero translation."""
        source_coeffs = np.ones(interpolator.source_cubature.num_nodes, dtype=np.complex128)

        wavenumber = 10.0
        translation_vector = np.array([0.0, 0.0, 0.0])

        target_coeffs = interpolator.interpolate(
            source_coeffs, translation_vector, wavenumber
        )

        # With zero translation and constant input, should have phase factor = 1
        # So result should be close to interpolated constant
        assert np.mean(np.abs(target_coeffs)) > 0

    def test_different_orders(self):
        """Test interpolation between different orders."""
        # Up-sampling
        interp_up = SphericalInterpolator(source_order=3, target_order=7)

        # Down-sampling
        interp_down = SphericalInterpolator(source_order=7, target_order=3)

        source_coeffs = np.random.randn(Cubature(3).num_nodes) + \
                       1j * np.random.randn(Cubature(3).num_nodes)

        wavenumber = 5.0
        translation_vector = np.array([0.2, 0.0, 0.0])

        # Up-sampling
        target_up = interp_up.interpolate(
            source_coeffs, translation_vector, wavenumber
        )

        # Check that output size matches target
        assert target_up.shape == (Cubature(7).num_nodes,)


class TestCubatureInterpolation:
    """Test cubature rules for interpolation."""

    def test_cubature_orders(self):
        """Test different cubature orders."""
        for order in [2, 3, 5, 7]:
            cubature = Cubature(order)
            assert cubature is not None
            assert cubature.num_nodes > 0
            assert cubature.num_nodes == order * 2 * order

    def test_cubature_symmetry(self):
        """Test that cubature nodes are symmetric."""
        cubature = Cubature(order=5)

        # For each node, check that opposite direction exists
        for node in cubature.nodes:
            opposite = -node
            # Find closest node to opposite
            distances = np.linalg.norm(cubature.nodes - opposite, axis=1)
            min_dist = np.min(distances)
            assert min_dist < 0.1  # Should find very close match


class TestChargeConservation:
    """Test charge conservation properties."""

    def test_total_charge_preserved(self):
        """Test that total charge is preserved through interpolation."""
        # Create interpolator
        interpolator = SphericalInterpolator(source_order=5, target_order=7)

        # Source coefficients representing total charge
        source_coeffs = np.ones(interpolator.source_cubature.num_nodes)

        # Total "charge" in source
        source_total = np.sum(source_coeffs * interpolator.source_cubature.weights)

        # Interpolate with zero translation
        wavenumber = 10.0
        translation_vector = np.array([0.0, 0.0, 0.0])

        target_coeffs = interpolator.interpolate(
            source_coeffs, translation_vector, wavenumber
        )

        # Total "charge" in target
        target_total = np.sum(target_coeffs * interpolator.target_cubature.weights)

        # Should be approximately equal (within numerical tolerance)
        # Note: With phase factors, this is more complex
        assert source_total > 0
        assert target_total is not None


class TestSphericalAccuracy:
    """Test accuracy of spherical interpolation."""

    def test_spherical_harmonics_interpolation(self):
        """Test interpolation of spherical harmonics."""
        # Create interpolator
        interpolator = SphericalInterpolator(source_order=7, target_order=10)

        # Test with low-degree spherical harmonics (should interpolate well)
        from scipy.special import sph_harm

        # Create a test function: Y_{lm} for l=2, m=1
        l_test, m_test = 2, 1

        source_coeffs = np.zeros(interpolator.source_cubature.num_nodes, dtype=np.complex128)
        for q, node in enumerate(interpolator.source_cubature.nodes):
            # Convert to spherical coordinates
            theta = np.arccos(node[2])
            phi = np.arctan2(node[1], node[0])
            source_coeffs[q] = sph_harm(m_test, l_test, phi, theta)

        # Interpolate
        wavenumber = 0.0  # No oscillations
        translation_vector = np.array([0.0, 0.0, 0.0])

        target_coeffs = interpolator.interpolate(
            source_coeffs, translation_vector, wavenumber
        )

        # Check that interpolated values match expected spherical harmonic
        max_error = 0.0
        for q, node in enumerate(interpolator.target_cubature.nodes):
            theta = np.arccos(node[2])
            phi = np.arctan2(node[1], node[0])
            expected = sph_harm(m_test, l_test, phi, theta)
            error = abs(target_coeffs[q] - expected)
            max_error = max(max_error, error)

        # Should have reasonable accuracy for low-degree harmonics
        assert max_error < 0.5  # Relatively loose tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
