"""
Orbital Elements and Coordinate Transformations

This module implements orbital elements representation and conversions between
different coordinate systems used in orbital mechanics. Based on the theory
presented in Okasha & Newman (2014).

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass

from ..utils.constants import EARTH_MU, PI, TWO_PI, TOLERANCE_ANGLE, TOLERANCE_POSITION
from ..utils.math_utils import (
    normalize_angle, wrap_to_2pi, wrap_to_pi, 
    solve_kepler_equation, rotation_matrix_313
)


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
    omega_cap: float  # RAAN (Ω)
    omega: float      # Argument of periapsis (ω)
    f: float          # True anomaly
    mu: float = EARTH_MU
    
    def __post_init__(self):
        """Validate orbital elements after initialization."""
        self._validate_elements()
        self._normalize_angles()
    
    def _validate_elements(self):
        """Validate orbital element ranges."""
        if self.a <= 0:
            raise ValueError("Semi-major axis must be positive")
        if not (0 <= self.e < 1):
            raise ValueError("Eccentricity must be in range [0, 1)")
        if not (0 <= self.i <= PI):
            raise ValueError("Inclination must be in range [0, π]")
        if self.mu <= 0:
            raise ValueError("Gravitational parameter must be positive")
    
    def _normalize_angles(self):
        """Normalize angular elements to proper ranges."""
        self.omega_cap = wrap_to_2pi(self.omega_cap)
        self.omega = wrap_to_2pi(self.omega)
        self.f = wrap_to_2pi(self.f)
    
    @property
    def period(self) -> float:
        """Orbital period [s]."""
        return TWO_PI * np.sqrt(self.a**3 / self.mu)
    
    @property
    def mean_motion(self) -> float:
        """Mean motion [rad/s]."""
        return np.sqrt(self.mu / self.a**3)
    
    @property
    def specific_energy(self) -> float:
        """Specific orbital energy [m²/s²]."""
        return -self.mu / (2 * self.a)
    
    @property
    def angular_momentum_magnitude(self) -> float:
        """Specific angular momentum magnitude [m²/s]."""
        return np.sqrt(self.mu * self.a * (1 - self.e**2))
    
    def mean_anomaly(self) -> float:
        """Calculate mean anomaly from true anomaly [rad]."""
        E = self.eccentric_anomaly()
        return E - self.e * np.sin(E)
    
    def eccentric_anomaly(self) -> float:
        """Calculate eccentric anomaly from true anomaly [rad]."""
        cos_f = np.cos(self.f)
        sin_f = np.sin(self.f)
        
        cos_E = (self.e + cos_f) / (1 + self.e * cos_f)
        sin_E = np.sqrt(1 - self.e**2) * sin_f / (1 + self.e * cos_f)
        
        return np.arctan2(sin_E, cos_E)
    
    def radius(self) -> float:
        """Calculate orbital radius at current true anomaly [m]."""
        return self.a * (1 - self.e**2) / (1 + self.e * np.cos(self.f))
    
    def velocity_magnitude(self) -> float:
        """Calculate velocity magnitude at current position [m/s]."""
        r = self.radius()
        return np.sqrt(self.mu * (2/r - 1/self.a))


def cartesian_to_orbital_elements(r_vec: np.ndarray, v_vec: np.ndarray, 
                                mu: float = EARTH_MU) -> OrbitalElements:
    """
    Convert Cartesian state vectors to orbital elements.
    
    Args:
        r_vec: Position vector in ECI frame [m] (3x1)
        v_vec: Velocity vector in ECI frame [m/s] (3x1)
        mu: Gravitational parameter [m³/s²]
    
    Returns:
        Orbital elements
    """
    if r_vec.shape != (3,) or v_vec.shape != (3,):
        raise ValueError("Position and velocity vectors must be 3D")
    
    # Magnitudes
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    
    if r < TOLERANCE_POSITION:
        raise ValueError("Position magnitude too small")
    
    # Angular momentum vector
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    
    if h < TOLERANCE_POSITION:
        raise ValueError("Angular momentum too small (rectilinear motion)")
    
    # Node vector
    k_vec = np.array([0, 0, 1])
    n_vec = np.cross(k_vec, h_vec)
    n = np.linalg.norm(n_vec)
    
    # Eccentricity vector
    e_vec = ((v**2 - mu/r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)
    
    # Specific energy
    energy = v**2/2 - mu/r
    
    # Semi-major axis
    if abs(energy) < TOLERANCE_POSITION:
        raise ValueError("Parabolic orbit not supported")
    a = -mu / (2 * energy)
    
    # Inclination
    i = np.arccos(np.clip(h_vec[2] / h, -1, 1))
    
    # Right ascension of ascending node (RAAN)
    if n < TOLERANCE_POSITION:  # Equatorial orbit
        omega_cap = 0.0
    else:
        omega_cap = np.arccos(np.clip(n_vec[0] / n, -1, 1))
        if n_vec[1] < 0:
            omega_cap = TWO_PI - omega_cap
    
    # Argument of periapsis
    if n < TOLERANCE_POSITION:  # Equatorial orbit
        if i < TOLERANCE_ANGLE:  # Equatorial prograde
            omega = np.arccos(np.clip(e_vec[0] / e, -1, 1))
            if e_vec[1] < 0:
                omega = TWO_PI - omega
        else:  # Equatorial retrograde
            omega = np.arccos(np.clip(-e_vec[0] / e, -1, 1))
            if e_vec[1] > 0:
                omega = TWO_PI - omega
    else:
        if e < TOLERANCE_POSITION:  # Circular orbit
            omega = 0.0
        else:
            omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
            if e_vec[2] < 0:
                omega = TWO_PI - omega
    
    # True anomaly
    if e < TOLERANCE_POSITION:  # Circular orbit
        if n < TOLERANCE_POSITION:  # Equatorial circular
            f = np.arccos(np.clip(r_vec[0] / r, -1, 1))
            if r_vec[1] < 0:
                f = TWO_PI - f
        else:
            f = np.arccos(np.clip(np.dot(n_vec, r_vec) / (n * r), -1, 1))
            if r_vec[2] < 0:
                f = TWO_PI - f
    else:
        f = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
        if np.dot(r_vec, v_vec) < 0:
            f = TWO_PI - f
    
    return OrbitalElements(a, e, i, omega_cap, omega, f, mu)


def orbital_elements_to_cartesian(elements: OrbitalElements) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert orbital elements to Cartesian state vectors.
    
    Args:
        elements: Orbital elements
    
    Returns:
        Tuple of (position_vector, velocity_vector) in ECI frame
    """
    # Calculate position and velocity in perifocal frame
    r = elements.radius()
    h = elements.angular_momentum_magnitude
    
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
    
    # Transformation matrix from perifocal to ECI
    R_pqw_to_eci = rotation_matrix_313(
        elements.omega_cap, 
        elements.i, 
        elements.omega
    ).T
    
    # Transform to ECI frame
    r_eci = R_pqw_to_eci @ r_pqw
    v_eci = R_pqw_to_eci @ v_pqw
    
    return r_eci, v_eci


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


def orbital_elements_rates_gauss(elements: OrbitalElements, 
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
    
    # Orbital parameters
    r = elements.radius()
    h = elements.angular_momentum_magnitude
    n = elements.mean_motion
    p = a * (1 - e**2)  # Semi-latus rectum
    
    # Trigonometric functions
    cos_f = np.cos(f)
    sin_f = np.sin(f)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    
    # Gauss' variational equations
    da_dt = 2 * a**2 / h * (e * sin_f * a_r + p / r * a_s)
    
    de_dt = 1 / h * (p * sin_f * a_r + ((p + r) * cos_f + r * e) * a_s)
    
    di_dt = r * cos_f / h * a_w
    
    domega_cap_dt = r * sin_f / (h * sin_i) * a_w
    
    domega_dt = (1 / (h * e) * (-p * cos_f * a_r + (p + r) * sin_f * a_s) - 
                 cos_i * domega_cap_dt)
    
    df_dt = (h / r**2 + 1 / (h * e) * (p * cos_f * a_r - (p + r) * sin_f * a_s))
    
    return np.array([da_dt, de_dt, di_dt, domega_cap_dt, domega_dt, df_dt])


def rsw_to_eci_matrix(elements: OrbitalElements) -> np.ndarray:
    """
    Calculate transformation matrix from RSW (radial-along track-cross track) 
    to ECI frame.
    
    Args:
        elements: Orbital elements
    
    Returns:
        Transformation matrix [3x3]
    """
    # Get position and velocity in ECI
    r_eci, v_eci = orbital_elements_to_cartesian(elements)
    
    # Radial unit vector
    r_hat = r_eci / np.linalg.norm(r_eci)
    
    # Cross-track unit vector (normal to orbital plane)
    h_vec = np.cross(r_eci, v_eci)
    w_hat = h_vec / np.linalg.norm(h_vec)
    
    # Along-track unit vector
    s_hat = np.cross(w_hat, r_hat)
    
    # Transformation matrix (columns are unit vectors)
    return np.column_stack([r_hat, s_hat, w_hat])


def eci_to_rsw_matrix(elements: OrbitalElements) -> np.ndarray:
    """
    Calculate transformation matrix from ECI to RSW frame.
    
    Args:
        elements: Orbital elements
    
    Returns:
        Transformation matrix [3x3]
    """
    return rsw_to_eci_matrix(elements).T

