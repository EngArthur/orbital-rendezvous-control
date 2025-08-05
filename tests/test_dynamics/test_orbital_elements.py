"""
Unit tests for orbital elements module.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import pytest
import numpy as np
from src.dynamics.orbital_elements import (
    OrbitalElements, cartesian_to_orbital_elements, 
    orbital_elements_to_cartesian, propagate_orbital_elements_mean_motion
)
from src.utils.constants import EARTH_MU, EARTH_RADIUS


class TestOrbitalElements:
    """Test cases for OrbitalElements class."""
    
    def test_orbital_elements_creation(self):
        """Test creation of orbital elements."""
        # ISS-like orbit
        elements = OrbitalElements(
            a=EARTH_RADIUS + 400e3,  # 400 km altitude
            e=0.0001,
            i=np.radians(51.6),
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        assert elements.a > EARTH_RADIUS
        assert 0 <= elements.e < 1
        assert 0 <= elements.i <= np.pi
        assert elements.period > 0
        assert elements.mean_motion > 0
    
    def test_orbital_elements_validation(self):
        """Test validation of orbital elements."""
        # Test negative semi-major axis
        with pytest.raises(ValueError):
            OrbitalElements(-1000, 0.1, 0.1, 0.0, 0.0, 0.0)
        
        # Test invalid eccentricity
        with pytest.raises(ValueError):
            OrbitalElements(7000e3, 1.1, 0.1, 0.0, 0.0, 0.0)
        
        # Test invalid inclination
        with pytest.raises(ValueError):
            OrbitalElements(7000e3, 0.1, -0.1, 0.0, 0.0, 0.0)
    
    def test_orbital_properties(self):
        """Test orbital property calculations."""
        # Circular orbit
        elements = OrbitalElements(
            a=7000e3,
            e=0.0,
            i=0.0,
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        # Test radius for circular orbit
        assert abs(elements.radius() - elements.a) < 1e-6
        
        # Test velocity for circular orbit
        v_expected = np.sqrt(EARTH_MU / elements.a)
        assert abs(elements.velocity_magnitude() - v_expected) < 1e-6


class TestCoordinateConversions:
    """Test cases for coordinate conversions."""
    
    def test_cartesian_to_orbital_round_trip(self):
        """Test round-trip conversion: orbital -> cartesian -> orbital."""
        # Original orbital elements
        original = OrbitalElements(
            a=7000e3,
            e=0.1,
            i=np.radians(30),
            omega_cap=np.radians(45),
            omega=np.radians(60),
            f=np.radians(90)
        )
        
        # Convert to Cartesian
        r_vec, v_vec = orbital_elements_to_cartesian(original)
        
        # Convert back to orbital elements
        converted = cartesian_to_orbital_elements(r_vec, v_vec)
        
        # Check if elements match (within tolerance)
        tolerance = 1e-10
        assert abs(converted.a - original.a) < tolerance
        assert abs(converted.e - original.e) < tolerance
        assert abs(converted.i - original.i) < tolerance
        assert abs(converted.omega_cap - original.omega_cap) < tolerance
        assert abs(converted.omega - original.omega) < tolerance
        assert abs(converted.f - original.f) < tolerance
    
    def test_circular_equatorial_orbit(self):
        """Test conversion for circular equatorial orbit."""
        elements = OrbitalElements(
            a=7000e3,
            e=0.0,
            i=0.0,
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        r_vec, v_vec = orbital_elements_to_cartesian(elements)
        
        # For circular equatorial orbit at f=0, position should be [a, 0, 0]
        expected_r = np.array([elements.a, 0.0, 0.0])
        np.testing.assert_allclose(r_vec, expected_r, rtol=1e-10)
        
        # Velocity should be [0, v_circular, 0]
        v_circular = np.sqrt(EARTH_MU / elements.a)
        expected_v = np.array([0.0, v_circular, 0.0])
        np.testing.assert_allclose(v_vec, expected_v, rtol=1e-10)


class TestOrbitalPropagation:
    """Test cases for orbital propagation."""
    
    def test_mean_motion_propagation(self):
        """Test mean motion propagation."""
        elements = OrbitalElements(
            a=7000e3,
            e=0.1,
            i=np.radians(30),
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        # Propagate for one period
        period = elements.period
        propagated = propagate_orbital_elements_mean_motion(elements, period)
        
        # True anomaly should return to approximately the same value
        # (within numerical precision)
        assert abs(propagated.f - elements.f) < 1e-6
        
        # Other elements should remain unchanged
        assert abs(propagated.a - elements.a) < 1e-12
        assert abs(propagated.e - elements.e) < 1e-12
        assert abs(propagated.i - elements.i) < 1e-12
    
    def test_short_time_propagation(self):
        """Test propagation for short time intervals."""
        elements = OrbitalElements(
            a=7000e3,
            e=0.0,
            i=0.0,
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        # Propagate for 1 minute
        dt = 60.0
        propagated = propagate_orbital_elements_mean_motion(elements, dt)
        
        # Change in true anomaly should match mean motion
        expected_delta_f = elements.mean_motion * dt
        actual_delta_f = propagated.f - elements.f
        
        assert abs(actual_delta_f - expected_delta_f) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])

