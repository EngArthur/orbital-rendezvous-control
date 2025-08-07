"""
Orbital Perturbation Models

This module implements various orbital perturbation models including
J2 gravitational perturbations and atmospheric drag effects.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass

from ..utils.constants import (
    EARTH_MU, EARTH_RADIUS, EARTH_J2, EARTH_ROTATION_RATE,
    EARTH_ATMOSPHERE_SCALE_HEIGHT, EARTH_ATMOSPHERE_DENSITY_SEA_LEVEL
)
from .orbital_elements import OrbitalElements


@dataclass
class SpacecraftProperties:
    """
    Spacecraft physical properties for perturbation calculations.
    
    Attributes:
        mass: Spacecraft mass [kg]
        drag_area: Cross-sectional area for drag [m²]
        drag_coefficient: Drag coefficient [-]
        srp_area: Solar radiation pressure area [m²]
        srp_coefficient: Solar radiation pressure coefficient [-]
    """
    mass: float
    drag_area: float
    drag_coefficient: float = 2.2
    srp_area: float = None
    srp_coefficient: float = 1.8
    
    def __post_init__(self):
        """Set default SRP area if not provided."""
        if self.srp_area is None:
            self.srp_area = self.drag_area


def j2_perturbation_acceleration_eci(r_eci: np.ndarray) -> np.ndarray:
    """
    Calculate J2 gravitational perturbation acceleration in ECI frame.
    
    Args:
        r_eci: Position vector in ECI frame [m] (3x1)
    
    Returns:
        J2 perturbation acceleration in ECI frame [m/s²] (3x1)
    """
    if r_eci.shape != (3,):
        raise ValueError("Position vector must be 3D")
    
    x, y, z = r_eci
    r = np.linalg.norm(r_eci)
    
    if r <= EARTH_RADIUS:
        raise ValueError("Position inside Earth")
    
    # J2 perturbation acceleration components
    factor = -1.5 * EARTH_J2 * EARTH_MU * EARTH_RADIUS**2 / r**5
    
    ax = factor * x * (5 * z**2 / r**2 - 1)
    ay = factor * y * (5 * z**2 / r**2 - 1)
    az = factor * z * (5 * z**2 / r**2 - 3)
    
    return np.array([ax, ay, az])


def j2_perturbation_acceleration_rsw(elements: OrbitalElements) -> np.ndarray:
    """
    Calculate J2 perturbation acceleration in RSW frame.
    
    Args:
        elements: Orbital elements
    
    Returns:
        J2 perturbation acceleration in RSW frame [m/s²] (3x1)
        [radial, along-track, cross-track]
    """
    # Orbital parameters
    a, e, i, omega_cap, omega, f = (
        elements.a, elements.e, elements.i,
        elements.omega_cap, elements.omega, elements.f
    )
    
    r = elements.radius()
    n = elements.mean_motion
    
    # Trigonometric functions
    cos_f = np.cos(f)
    sin_f = np.sin(f)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    cos_u = np.cos(omega + f)  # Argument of latitude
    sin_u = np.sin(omega + f)
    
    # J2 perturbation factor
    factor = -1.5 * EARTH_J2 * EARTH_MU * EARTH_RADIUS**2 / r**4
    
    # RSW components
    a_r = factor * (1 - 3 * sin_i**2 * sin_u**2)
    
    a_s = factor * sin_i**2 * np.sin(2 * u)
    
    a_w = factor * sin_i * np.cos(i) * np.sin(2 * u)
    
    return np.array([a_r, a_s, a_w])


def atmospheric_density_exponential(altitude: float) -> float:
    """
    Calculate atmospheric density using exponential model.
    
    Args:
        altitude: Altitude above Earth surface [m]
    
    Returns:
        Atmospheric density [kg/m³]
    """
    if altitude < 0:
        return EARTH_ATMOSPHERE_DENSITY_SEA_LEVEL
    
    return EARTH_ATMOSPHERE_DENSITY_SEA_LEVEL * np.exp(-altitude / EARTH_ATMOSPHERE_SCALE_HEIGHT)


def atmospheric_drag_acceleration_eci(r_eci: np.ndarray, v_eci: np.ndarray,
                                     spacecraft: SpacecraftProperties) -> np.ndarray:
    """
    Calculate atmospheric drag acceleration in ECI frame.
    
    Args:
        r_eci: Position vector in ECI frame [m] (3x1)
        v_eci: Velocity vector in ECI frame [m/s] (3x1)
        spacecraft: Spacecraft properties
    
    Returns:
        Drag acceleration in ECI frame [m/s²] (3x1)
    """
    if r_eci.shape != (3,) or v_eci.shape != (3,):
        raise ValueError("Position and velocity vectors must be 3D")
    
    # Calculate altitude
    r = np.linalg.norm(r_eci)
    altitude = r - EARTH_RADIUS
    
    if altitude < 0:
        return np.zeros(3)  # No drag below surface
    
    # Atmospheric density
    rho = atmospheric_density_exponential(altitude)
    
    if rho < 1e-15:  # Negligible density
        return np.zeros(3)
    
    # Relative velocity (accounting for Earth rotation)
    omega_earth = np.array([0, 0, EARTH_ROTATION_RATE])
    v_rel = v_eci - np.cross(omega_earth, r_eci)
    v_rel_mag = np.linalg.norm(v_rel)
    
    if v_rel_mag < 1e-6:  # Negligible relative velocity
        return np.zeros(3)
    
    # Drag acceleration
    drag_factor = -0.5 * rho * spacecraft.drag_coefficient * spacecraft.drag_area / spacecraft.mass
    a_drag = drag_factor * v_rel_mag * v_rel
    
    return a_drag


def atmospheric_drag_acceleration_rsw(elements: OrbitalElements, v_eci: np.ndarray,
                                    spacecraft: SpacecraftProperties) -> np.ndarray:
    """
    Calculate atmospheric drag acceleration in RSW frame.
    
    Args:
        elements: Orbital elements
        v_eci: Velocity vector in ECI frame [m/s] (3x1)
        spacecraft: Spacecraft properties
    
    Returns:
        Drag acceleration in RSW frame [m/s²] (3x1)
    """
    from .orbital_elements import orbital_elements_to_cartesian, eci_to_rsw_matrix
    
    # Get position in ECI
    r_eci, _ = orbital_elements_to_cartesian(elements)
    
    # Calculate drag in ECI
    a_drag_eci = atmospheric_drag_acceleration_eci(r_eci, v_eci, spacecraft)
    
    # Transform to RSW frame
    T_eci_to_rsw = eci_to_rsw_matrix(elements)
    a_drag_rsw = T_eci_to_rsw @ a_drag_eci
    
    return a_drag_rsw


def total_perturbation_acceleration_rsw(elements: OrbitalElements, v_eci: np.ndarray,
                                      spacecraft: SpacecraftProperties,
                                      include_j2: bool = True,
                                      include_drag: bool = True) -> np.ndarray:
    """
    Calculate total perturbation acceleration in RSW frame.
    
    Args:
        elements: Orbital elements
        v_eci: Velocity vector in ECI frame [m/s] (3x1)
        spacecraft: Spacecraft properties
        include_j2: Include J2 perturbations
        include_drag: Include atmospheric drag
    
    Returns:
        Total perturbation acceleration in RSW frame [m/s²] (3x1)
    """
    a_total = np.zeros(3)
    
    if include_j2:
        a_total += j2_perturbation_acceleration_rsw(elements)
    
    if include_drag:
        a_total += atmospheric_drag_acceleration_rsw(elements, v_eci, spacecraft)
    
    return a_total


def propagate_orbital_elements_perturbed(elements: OrbitalElements, 
                                       v_eci: np.ndarray,
                                       spacecraft: SpacecraftProperties,
                                       delta_t: float,
                                       include_j2: bool = True,
                                       include_drag: bool = True) -> OrbitalElements:
    """
    Propagate orbital elements including perturbations using Gauss' equations.
    
    Args:
        elements: Initial orbital elements
        v_eci: Velocity vector in ECI frame [m/s] (3x1)
        spacecraft: Spacecraft properties
        delta_t: Time step [s]
        include_j2: Include J2 perturbations
        include_drag: Include atmospheric drag
    
    Returns:
        Propagated orbital elements
    """
    from .orbital_elements import orbital_elements_rates_gauss, propagate_orbital_elements_mean_motion
    
    # Calculate perturbation acceleration
    a_pert = total_perturbation_acceleration_rsw(
        elements, v_eci, spacecraft, include_j2, include_drag
    )
    
    # Calculate rates using Gauss' equations
    element_rates = orbital_elements_rates_gauss(elements, a_pert)
    
    # Propagate using first-order integration
    new_a = elements.a + element_rates[0] * delta_t
    new_e = elements.e + element_rates[1] * delta_t
    new_i = elements.i + element_rates[2] * delta_t
    new_omega_cap = elements.omega_cap + element_rates[3] * delta_t
    new_omega = elements.omega + element_rates[4] * delta_t
    new_f = elements.f + element_rates[5] * delta_t
    
    # Ensure valid ranges
    new_e = max(0.0, min(new_e, 0.99999))
    new_i = max(0.0, min(new_i, np.pi))
    
    # Normalize angles
    from ..utils.math_utils import wrap_to_2pi
    new_omega_cap = wrap_to_2pi(new_omega_cap)
    new_omega = wrap_to_2pi(new_omega)
    new_f = wrap_to_2pi(new_f)
    
    return OrbitalElements(
        new_a, new_e, new_i, new_omega_cap, new_omega, new_f, elements.mu
    )


def orbital_decay_time_estimate(elements: OrbitalElements, 
                               spacecraft: SpacecraftProperties) -> float:
    """
    Estimate orbital decay time due to atmospheric drag.
    
    Args:
        elements: Orbital elements
        spacecraft: Spacecraft properties
    
    Returns:
        Estimated decay time [s]
    """
    # Simplified decay time estimation
    from .orbital_elements import orbital_elements_to_cartesian
    
    r_eci, v_eci = orbital_elements_to_cartesian(elements)
    altitude = np.linalg.norm(r_eci) - EARTH_RADIUS
    
    if altitude > 500e3:  # Above 500 km, very long decay time
        return 1e10  # Essentially infinite
    
    # Atmospheric density
    rho = atmospheric_density_exponential(altitude)
    
    if rho < 1e-15:
        return 1e10
    
    # Ballistic coefficient
    beta = spacecraft.mass / (spacecraft.drag_coefficient * spacecraft.drag_area)
    
    # Simplified decay time (very approximate)
    # Based on exponential atmosphere and circular orbit assumption
    scale_height_orbital = EARTH_ATMOSPHERE_SCALE_HEIGHT
    decay_time = (2 * beta * scale_height_orbital) / (rho * np.linalg.norm(v_eci))
    
    return decay_time


def perturbation_analysis_summary(elements: OrbitalElements,
                                spacecraft: SpacecraftProperties) -> dict:
    """
    Provide summary analysis of perturbation effects.
    
    Args:
        elements: Orbital elements
        spacecraft: Spacecraft properties
    
    Returns:
        Dictionary with perturbation analysis
    """
    from .orbital_elements import orbital_elements_to_cartesian
    
    r_eci, v_eci = orbital_elements_to_cartesian(elements)
    altitude = np.linalg.norm(r_eci) - EARTH_RADIUS
    
    # J2 perturbation magnitude
    a_j2 = j2_perturbation_acceleration_rsw(elements)
    j2_magnitude = np.linalg.norm(a_j2)
    
    # Drag perturbation magnitude
    a_drag = atmospheric_drag_acceleration_rsw(elements, v_eci, spacecraft)
    drag_magnitude = np.linalg.norm(a_drag)
    
    # Atmospheric density
    rho = atmospheric_density_exponential(altitude)
    
    # Decay time estimate
    decay_time = orbital_decay_time_estimate(elements, spacecraft)
    
    return {
        'altitude_km': altitude / 1000,
        'atmospheric_density_kg_m3': rho,
        'j2_acceleration_magnitude_m_s2': j2_magnitude,
        'drag_acceleration_magnitude_m_s2': drag_magnitude,
        'perturbation_ratio_drag_to_j2': drag_magnitude / j2_magnitude if j2_magnitude > 0 else 0,
        'estimated_decay_time_days': decay_time / 86400,
        'dominant_perturbation': 'drag' if drag_magnitude > j2_magnitude else 'j2'
    }

