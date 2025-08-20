"""Orbital Perturbation Models

This module implements various orbital perturbation models including
J2 gravitational perturbations and atmospheric drag effects.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import required modules
from dynamics.orbital_elements import (OrbitalElements,
                                       orbital_elements_to_cartesian)
from utils.constants import EARTH_MU, EARTH_RADIUS

# Define constants directly (to avoid import issues)
EARTH_J2 = 1.08262668e-3  # Earth's J2 coefficient (dimensionless)
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s
EARTH_ATMOSPHERE_SCALE_HEIGHT = 8400.0  # m
EARTH_ATMOSPHERE_DENSITY_SEA_LEVEL = 1.225  # kg/m³


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


def eci_to_rsw_transformation_matrix(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """
    Calculate transformation matrix from ECI to RSW frame using position and velocity vectors.
    
    Args:
        r_vec: Position vector in ECI frame [m]
        v_vec: Velocity vector in ECI frame [m/s]
        
    Returns:
        3x3 transformation matrix from ECI to RSW
    """
    # Normalize position vector (radial direction)
    r_hat = r_vec / np.linalg.norm(r_vec)
    
    # Angular momentum vector (cross-track direction)
    h_vec = np.cross(r_vec, v_vec)
    w_hat = h_vec / np.linalg.norm(h_vec)
    
    # Along-track direction (completes right-handed system)
    s_hat = np.cross(w_hat, r_hat)
    
    # RSW transformation matrix (each row is a unit vector)
    R_eci_to_rsw = np.array([
        r_hat,  # Radial
        s_hat,  # Along-track (S)
        w_hat   # Cross-track (W)
    ])
    
    return R_eci_to_rsw


def calculate_orbital_properties(elements: OrbitalElements) -> dict:
    """
    Calculate orbital properties directly from elements to avoid dependency issues.
    
    Args:
        elements: Orbital elements
        
    Returns:
        Dictionary with calculated properties
    """
    a, e, mu = elements.a, elements.e, elements.mu
    
    # Calculate properties directly
    h = np.sqrt(mu * a * (1 - e**2))  # Angular momentum magnitude
    n = np.sqrt(mu / a**3)  # Mean motion
    period = 2 * np.pi / n  # Orbital period
    
    return {
        'angular_momentum_magnitude': h,
        'mean_motion': n,
        'period': period
    }


def orbital_elements_rates_gauss_local(elements: OrbitalElements, 
                                     perturbation_acceleration: np.ndarray) -> np.ndarray:
    """
    Calculate rates of change of orbital elements using Gauss' variational equations.
    
    Args:
        elements: Current orbital elements
        perturbation_acceleration: Perturbation acceleration in RSW frame [m/s²] (3x1)
                                  [radial, along-track, cross-track]
    
    Returns:
        Rates of orbital elements [da/dt, de/dt, di/dt, dΩ/dt, dω/dt, df/dt]
    """
    if perturbation_acceleration.shape != (3,):
        raise ValueError("Perturbation acceleration must be 3D vector")
    
    a, e, i, omega_cap, omega, f = (
        elements.a, elements.e, elements.i,
        elements.omega_cap, elements.omega, elements.f
    )
    
    # Perturbation components
    a_r, a_s, a_w = perturbation_acceleration
    
    # Calculate orbital properties directly (bulletproof approach)
    props = calculate_orbital_properties(elements)
    h = props['angular_momentum_magnitude']
    n = props['mean_motion']
    
    # Current radius (calculate directly)
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    p = a * (1 - e**2)  # Semi-latus rectum
    
    # Trigonometric functions
    cos_f = np.cos(f)
    sin_f = np.sin(f)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    
    # Handle singularities for circular and equatorial orbits
    if e < 1e-8:  # Nearly circular orbit
        # For circular orbits, some terms become undefined
        # Use simplified equations
        da_dt = 2 * a**2 / h * a_s
        de_dt = 1 / h * (p * sin_f * a_r + (p + r) * cos_f * a_s)
        di_dt = r * cos_f / h * a_w
        domega_cap_dt = r * sin_f / (h * sin_i) * a_w if sin_i > 1e-8 else 0.0
        domega_dt = -cos_i * domega_cap_dt if sin_i > 1e-8 else 0.0
        df_dt = h / r**2
    else:
        # Standard Gauss' variational equations
        da_dt = 2 * a**2 / h * (e * sin_f * a_r + p / r * a_s)
        
        de_dt = 1 / h * (p * sin_f * a_r + ((p + r) * cos_f + r * e) * a_s)
        
        di_dt = r * cos_f / h * a_w
        
        if sin_i > 1e-8:  # Avoid division by zero for equatorial orbits
            domega_cap_dt = r * sin_f / (h * sin_i) * a_w
        else:
            domega_cap_dt = 0.0
        
        domega_dt = (1 / (h * e) * (-p * cos_f * a_r + (p + r) * sin_f * a_s) - 
                     cos_i * domega_cap_dt)
        
        df_dt = (h / r**2 + 1 / (h * e) * (p * cos_f * a_r - (p + r) * sin_f * a_s))
    
    return np.array([da_dt, de_dt, di_dt, domega_cap_dt, domega_dt, df_dt])


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
    
    # Calculate radius directly
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    
    # Trigonometric functions
    cos_f = np.cos(f)
    sin_f = np.sin(f)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    
    # Argument of latitude
    u = omega + f
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    
    # J2 perturbation factor
    factor = -1.5 * EARTH_J2 * EARTH_MU * EARTH_RADIUS**2 / r**4
    
    # RSW components
    a_r = factor * (1 - 3 * sin_i**2 * sin_u**2)
    a_s = factor * sin_i**2 * np.sin(2 * u)
    a_w = factor * sin_i * cos_i * np.sin(2 * u)
    
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
    # Get position and velocity in ECI
    r_eci, v_eci_calc = orbital_elements_to_cartesian(elements)
    
    # Calculate drag in ECI
    a_drag_eci = atmospheric_drag_acceleration_eci(r_eci, v_eci, spacecraft)
    
    # Transform to RSW frame using our auxiliary function
    T_eci_to_rsw = eci_to_rsw_transformation_matrix(r_eci, v_eci_calc)
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
    from utils.math_utils import wrap_to_2pi

    # Calculate perturbation acceleration
    a_pert = total_perturbation_acceleration_rsw(
        elements, v_eci, spacecraft, include_j2, include_drag
    )
    
    # Calculate rates using Gauss' equations (local implementation)
    element_rates = orbital_elements_rates_gauss_local(elements, a_pert)
    
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
