"""
Orbital Elements and Coordinate Transformations

This module implements orbital elements representation and conversions between
different coordinate systems used in orbital mechanics. Based on the theory
presented in Okasha & Newman (2014).

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import os
import sys
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

# Add src to path for imports
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..', '..', 'src')
sys.path.insert(0, src_dir)

# Import with absolute paths
from utils.constants import (EARTH_MU, PI, TOLERANCE_ANGLE, TOLERANCE_POSITION,
                             TWO_PI)
from utils.math_utils import (normalize_angle, rotation_matrix_313,
                              solve_kepler_equation, wrap_to_2pi, wrap_to_pi)


@dataclass
class OrbitalElements:
    """
    Classical orbital elements representation.
    
    Attributes:
        a: Semi-major axis [m]
        e: Eccentricity [-]
        i: Inclination [rad]
        omega_cap: Right ascension of ascending node (RAAN) [rad]
        omega: Argument of periapsis [rad]
        f: True anomaly [rad]
        mu: Gravitational parameter [m³/s²]
    """
    a: float
    e: float
    i: float
    omega_cap: float
    omega: float
    f: float
    mu: float = EARTH_MU
    
    def __post_init__(self):
        """Validate orbital elements after initialization."""
        if self.a <= 0:
            raise ValueError("Semi-major axis must be positive")
        if not (0 <= self.e < 1):
            raise ValueError("Eccentricity must be in range [0, 1)")
        if not (0 <= self.i <= PI):
            raise ValueError("Inclination must be in range [0, π]")
        
        # Normalize angles
        self.omega_cap = wrap_to_2pi(self.omega_cap)
        self.omega = wrap_to_2pi(self.omega)
        self.f = wrap_to_2pi(self.f)
    
    @property
    def period(self) -> float:
        """Orbital period [s]."""
        return 2 * PI * np.sqrt(self.a**3 / self.mu)
    
    @property
    def mean_motion(self) -> float:
        """Mean motion [rad/s]."""
        return np.sqrt(self.mu / self.a**3)
    
    @property
    def angular_momentum(self) -> float:
        """Specific angular momentum [m²/s]."""
        return np.sqrt(self.mu * self.a * (1 - self.e**2))
    
    @property
    def energy(self) -> float:
        """Specific orbital energy [m²/s²]."""
        return -self.mu / (2 * self.a)
    
    def radius(self) -> float:
        """Current radius [m]."""
        return self.a * (1 - self.e**2) / (1 + self.e * np.cos(self.f))
    
    def velocity_magnitude(self) -> float:
        """Current velocity magnitude [m/s]."""
        r = self.radius()
        return np.sqrt(self.mu * (2/r - 1/self.a))
    
    def eccentric_anomaly(self) -> float:
        """Eccentric anomaly [rad]."""
        cos_E = (self.e + np.cos(self.f)) / (1 + self.e * np.cos(self.f))
        sin_E = np.sqrt(1 - self.e**2) * np.sin(self.f) / (1 + self.e * np.cos(self.f))
        return np.arctan2(sin_E, cos_E)
    
    def mean_anomaly(self) -> float:
        """Mean anomaly [rad]."""
        E = self.eccentric_anomaly()
        return E - self.e * np.sin(E)


def orbital_elements_to_cartesian(elements: OrbitalElements) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert orbital elements to Cartesian coordinates.
    
    Args:
        elements: Orbital elements
        
    Returns:
        Tuple of (position [m], velocity [m/s]) vectors in ECI frame
    """
    # Current radius and velocity magnitude
    r = elements.radius()
    h = elements.angular_momentum
    
    # Position in perifocal frame
    r_pqw = np.array([
        r * np.cos(elements.f),
        r * np.sin(elements.f),
        0.0
    ])
    
    # Velocity in perifocal frame
    v_pqw = np.array([
        -elements.mu / h * np.sin(elements.f),
        elements.mu / h * (elements.e + np.cos(elements.f)),
        0.0
    ])
    
    # Rotation matrix from perifocal to ECI
    R_pqw_to_eci = rotation_matrix_313(elements.omega_cap, elements.i, elements.omega)
    
    # Transform to ECI frame
    r_eci = R_pqw_to_eci @ r_pqw
    v_eci = R_pqw_to_eci @ v_pqw
    
    return r_eci, v_eci


def cartesian_to_orbital_elements(r_vec: np.ndarray, v_vec: np.ndarray, 
                                mu: float = EARTH_MU) -> OrbitalElements:
    """
    Convert Cartesian coordinates to orbital elements.
    
    Args:
        r_vec: Position vector [m]
        v_vec: Velocity vector [m/s]
        mu: Gravitational parameter [m³/s²]
        
    Returns:
        Orbital elements
    """
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    # Angular momentum vector
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    # Node vector
    k_hat = np.array([0, 0, 1])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)
    
    # Eccentricity vector
    e_vec = ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)
    
    # Specific energy
    energy = v**2/2 - mu/r
    
    # Semi-major axis
    if abs(energy) > TOLERANCE_POSITION:
        a = -mu / (2 * energy)
    else:
        a = np.inf  # Parabolic orbit
    
    # Inclination
    i = np.arccos(np.clip(h_vec[2] / h, -1, 1))
    
    # Right ascension of ascending node
    if n > TOLERANCE_POSITION:
        omega_cap = np.arccos(np.clip(n_vec[0] / n, -1, 1))
        if n_vec[1] < 0:
            omega_cap = TWO_PI - omega_cap
    else:
        omega_cap = 0.0  # Equatorial orbit
    
    # Argument of periapsis
    if n > TOLERANCE_POSITION and e > TOLERANCE_POSITION:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
        if e_vec[2] < 0:
            omega = TWO_PI - omega
    else:
        omega = 0.0  # Circular or equatorial orbit
    
    # True anomaly
    if e > TOLERANCE_POSITION:
        f = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
        if np.dot(r_vec, v_vec) < 0:
            f = TWO_PI - f
    else:
        # Circular orbit - use argument of latitude
        if n > TOLERANCE_POSITION:
            f = np.arccos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1, 1))
            if r_vec[2] < 0:
                f = TWO_PI - f
        else:
            f = np.arctan2(r_vec[1], r_vec[0])
            f = wrap_to_2pi(f)
    
    return OrbitalElements(a, e, i, omega_cap, omega, f, mu)


def propagate_orbital_elements_mean_motion(elements: OrbitalElements, 
                                         delta_t: float) -> OrbitalElements:
    """
    Propagate orbital elements using mean motion (Keplerian motion).
    
    Args:
        elements: Initial orbital elements
        delta_t: Time step [s]
    
    Returns:
        Propagated orbital elements
    """
    # Calculate initial mean anomaly
    M0 = elements.mean_anomaly()
    
    # Propagate mean anomaly
    n = elements.mean_motion
    M = M0 + n * delta_t
    M = wrap_to_2pi(M)
    
    # Solve for new eccentric anomaly
    E = solve_kepler_equation(M, elements.e)
    
    # Calculate new true anomaly
    cos_f = (np.cos(E) - elements.e) / (1 - elements.e * np.cos(E))
    sin_f = np.sqrt(1 - elements.e**2) * np.sin(E) / (1 - elements.e * np.cos(E))
    f_new = np.arctan2(sin_f, cos_f)
    f_new = wrap_to_2pi(f_new)
    
    # Create new orbital elements (only true anomaly changes in Keplerian motion)
    return OrbitalElements(
        elements.a, elements.e, elements.i,
        elements.omega_cap, elements.omega, f_new, elements.mu
    )


def propagate_orbital_elements_perturbed(elements: OrbitalElements, 
                                       delta_t: float,
                                       perturbation_accelerations: np.ndarray) -> OrbitalElements:
    """
    Propagate orbital elements with perturbations using Gauss variational equations.
    
    Args:
        elements: Initial orbital elements
        delta_t: Time step [s]
        perturbation_accelerations: Perturbation accelerations in RSW frame [m/s²]
        
    Returns:
        Propagated orbital elements with perturbations
    """
    # Current orbital parameters
    r = elements.radius()
    h = elements.angular_momentum
    n = elements.mean_motion
    
    # Perturbation accelerations in RSW frame
    a_r, a_s, a_w = perturbation_accelerations
    
    # Gauss variational equations (rates of change)
    da_dt = 2 * elements.a**2 / h * (elements.e * np.sin(elements.f) * a_r + 
                                     (1 + elements.e * np.cos(elements.f)) * a_s)
    
    de_dt = 1 / h * (np.sin(elements.f) * a_r + 
                     (np.cos(elements.f) + np.cos(elements.f + elements.omega)) * a_s)
    
    di_dt = r * np.cos(elements.f + elements.omega) / h * a_w
    
    domega_cap_dt = r * np.sin(elements.f + elements.omega) / (h * np.sin(elements.i)) * a_w
    
    domega_dt = 1 / (elements.e * h) * (-np.cos(elements.f) * a_r + 
                                        np.sin(elements.f) * a_s) - \
                np.cos(elements.i) * domega_cap_dt
    
    df_dt = h / r**2 + 1 / (elements.e * h) * (np.cos(elements.f) * a_r - 
                                               np.sin(elements.f) * a_s)
    
    # Integrate using Euler method
    new_a = elements.a + da_dt * delta_t
    new_e = elements.e + de_dt * delta_t
    new_i = elements.i + di_dt * delta_t
    new_omega_cap = elements.omega_cap + domega_cap_dt * delta_t
    new_omega = elements.omega + domega_dt * delta_t
    new_f = elements.f + df_dt * delta_t
    
    # Normalize angles
    new_omega_cap = wrap_to_2pi(new_omega_cap)
    new_omega = wrap_to_2pi(new_omega)
    new_f = wrap_to_2pi(new_f)
    
    # Ensure physical constraints
    new_e = max(0.0, min(new_e, 0.99))  # Keep eccentricity in valid range
    new_i = max(0.0, min(new_i, PI))    # Keep inclination in valid range
    
    return OrbitalElements(new_a, new_e, new_i, new_omega_cap, new_omega, new_f, elements.mu)

