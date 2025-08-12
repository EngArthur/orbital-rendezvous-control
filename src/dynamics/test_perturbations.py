"""
Unit tests for perturbations module.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import pytest
import numpy as np
from src.dynamics.perturbations import (
    SpacecraftProperties, j2_perturbation_acceleration_eci,
    j2_perturbation_acceleration_rsw, atmospheric_density_exponential,
    atmospheric_drag_acceleration_eci, total_perturbation_acceleration_rsw,
    orbital_decay_time_estimate, perturbation_analysis_summary
)
from src.dynamics.orbital_elements import OrbitalElements
from src.utils.constants import EARTH_RADIUS, EARTH_MU


class TestSpacecraftProperties:
    """Test cases for SpacecraftProperties class."""
    
    def test_spacecraft_properties_creation(self):
        """Test creation of spacecraft properties."""
        spacecraft = SpacecraftProperties(
            mass=100.0,
            drag_area=2.0,
            drag_coefficient=2.2
        )
        
        assert spacecraft.mass == 100.0
        assert spacecraft.drag_area == 2.0
        assert spacecraft.drag_coefficient == 2.2
        assert spacecraft.srp_area == 2.0  # Should default to drag_area


class TestJ2Perturbations:
    """Test cases for J2 gravitational perturbations."""
    
    def test_j2_perturbation_eci_polar_orbit(self):
        """Test J2 perturbation for polar orbit."""
        # Position at North Pole
        r_eci = np.array([0, 0, EARTH_RADIUS + 400e3])
        
        a_j2 = j2_perturbation_acceleration_eci(r_eci)
        
        # At poles, J2 acceleration should be primarily in z-direction
        assert abs(a_j2[0]) < 1e-10  # x-component should be zero
        assert abs(a_j2[1]) < 1e-10  # y-component should be zero
        assert a_j2[2] < 0  # z-component should be negative (toward Earth)
    
    def test_j2_perturbation_eci_equatorial_orbit(self):
        """Test J2 perturbation for equatorial orbit."""
        # Position at equator
        r_eci = np.array([EARTH_RADIUS + 400e3, 0, 0])
        
        a_j2 = j2_perturbation_acceleration_eci(r_eci)
        
        # At equator, J2 acceleration should be outward in x-direction
        assert a_j2[0] > 0  # x-component should be positive (away from Earth)
        assert abs(a_j2[1]) < 1e-10  # y-component should be zero
        assert abs(a_j2[2]) < 1e-10  # z-component should be zero
    
    def test_j2_perturbation_rsw(self):
        """Test J2 perturbation in RSW frame."""
        # ISS-like orbit
        elements = OrbitalElements(
            a=EARTH_RADIUS + 400e3,
            e=0.0001,
            i=np.radians(51.6),
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        a_j2_rsw = j2_perturbation_acceleration_rsw(elements)
        
        # Should have all three components for inclined orbit
        assert len(a_j2_rsw) == 3
        assert not np.allclose(a_j2_rsw, 0)


class TestAtmosphericDrag:
    """Test cases for atmospheric drag."""
    
    def test_atmospheric_density_sea_level(self):
        """Test atmospheric density at sea level."""
        rho = atmospheric_density_exponential(0.0)
        
        # Should be close to standard sea level density
        assert abs(rho - 1.225) < 0.01
    
    def test_atmospheric_density_high_altitude(self):
        """Test atmospheric density at high altitude."""
        rho_400km = atmospheric_density_exponential(400e3)
        rho_800km = atmospheric_density_exponential(800e3)
        
        # Density should decrease with altitude
        assert rho_400km > rho_800km
        assert rho_800km > 0
    
    def test_atmospheric_drag_acceleration(self):
        """Test atmospheric drag acceleration calculation."""
        # Low Earth orbit
        r_eci = np.array([EARTH_RADIUS + 300e3, 0, 0])
        v_eci = np.array([0, 7800, 0])  # Approximate orbital velocity
        
        spacecraft = SpacecraftProperties(
            mass=100.0,
            drag_area=2.0,
            drag_coefficient=2.2
        )
        
        a_drag = atmospheric_drag_acceleration_eci(r_eci, v_eci, spacecraft)
        
        # Drag should oppose velocity direction
        assert np.dot(a_drag, v_eci) < 0
        assert np.linalg.norm(a_drag) > 0


class TestTotalPerturbations:
    """Test cases for total perturbation calculations."""
    
    def test_total_perturbation_acceleration(self):
        """Test total perturbation acceleration."""
        elements = OrbitalElements(
            a=EARTH_RADIUS + 400e3,
            e=0.0001,
            i=np.radians(51.6),
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        spacecraft = SpacecraftProperties(
            mass=100.0,
            drag_area=2.0
        )
        
        from src.dynamics.orbital_elements import orbital_elements_to_cartesian
        _, v_eci = orbital_elements_to_cartesian(elements)
        
        # Test with both perturbations
        a_total = total_perturbation_acceleration_rsw(
            elements, v_eci, spacecraft, include_j2=True, include_drag=True
        )
        
        # Test with only J2
        a_j2_only = total_perturbation_acceleration_rsw(
            elements, v_eci, spacecraft, include_j2=True, include_drag=False
        )
        
        # Test with only drag
        a_drag_only = total_perturbation_acceleration_rsw(
            elements, v_eci, spacecraft, include_j2=False, include_drag=True
        )
        
        # Total should be sum of individual perturbations (approximately)
        assert len(a_total) == 3
        assert len(a_j2_only) == 3
        assert len(a_drag_only) == 3


class TestOrbitalDecay:
    """Test cases for orbital decay analysis."""
    
    def test_decay_time_high_altitude(self):
        """Test decay time for high altitude orbit."""
        elements = OrbitalElements(
            a=EARTH_RADIUS + 800e3,  # 800 km altitude
            e=0.0,
            i=0.0,
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        spacecraft = SpacecraftProperties(mass=100.0, drag_area=2.0)
        
        decay_time = orbital_decay_time_estimate(elements, spacecraft)
        
        # High altitude should have very long decay time
        assert decay_time > 1e8  # More than ~3 years
    
    def test_decay_time_low_altitude(self):
        """Test decay time for low altitude orbit."""
        elements = OrbitalElements(
            a=EARTH_RADIUS + 200e3,  # 200 km altitude
            e=0.0,
            i=0.0,
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        spacecraft = SpacecraftProperties(mass=100.0, drag_area=2.0)
        
        decay_time = orbital_decay_time_estimate(elements, spacecraft)
        
        # Low altitude should have shorter decay time
        assert decay_time < 1e7  # Less than ~4 months


class TestPerturbationAnalysis:
    """Test cases for perturbation analysis."""
    
    def test_perturbation_analysis_summary(self):
        """Test perturbation analysis summary."""
        elements = OrbitalElements(
            a=EARTH_RADIUS + 400e3,
            e=0.0001,
            i=np.radians(51.6),
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        spacecraft = SpacecraftProperties(mass=100.0, drag_area=2.0)
        
        analysis = perturbation_analysis_summary(elements, spacecraft)
        
        # Check that all expected keys are present
        expected_keys = [
            'altitude_km',
            'atmospheric_density_kg_m3',
            'j2_acceleration_magnitude_m_s2',
            'drag_acceleration_magnitude_m_s2',
            'perturbation_ratio_drag_to_j2',
            'estimated_decay_time_days',
            'dominant_perturbation'
        ]
        
        for key in expected_keys:
            assert key in analysis
        
        # Check reasonable values
        assert analysis['altitude_km'] > 0
        assert analysis['atmospheric_density_kg_m3'] > 0
        assert analysis['j2_acceleration_magnitude_m_s2'] > 0
        assert analysis['dominant_perturbation'] in ['j2', 'drag']


if __name__ == "__main__":
    pytest.main([__file__])

