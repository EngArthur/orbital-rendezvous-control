"""
Relative Motion Dynamics

This module implements relative motion dynamics between spacecraft,
including LVLH frame dynamics and coupled translational-rotational motion.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .orbital_elements import OrbitalElements, orbital_elements_to_cartesian
from .attitude_dynamics import AttitudeState, SpacecraftInertia
from ..utils.constants import EARTH_MU
from ..utils.math_utils import skew_symmetric, quaternion_to_rotation_matrix


@dataclass
class RelativeState:
    """
    Relative state between chaser and target spacecraft.
    
    Attributes:
        position: Relative position in LVLH frame [m] (3x1)
        velocity: Relative velocity in LVLH frame [m/s] (3x1)
        time: Time [s]
    """
    position: np.ndarray
    velocity: np.ndarray
    time: float = 0.0
    
    def __post_init__(self):
        """Validate relative state."""
        if self.position.shape != (3,) or self.velocity.shape != (3,):
            raise ValueError("Position and velocity must be 3D vectors")
    
    @property
    def range(self) -> float:
        """Range between spacecraft [m]."""
        return np.linalg.norm(self.position)
    
    @property
    def range_rate(self) -> float:
        """Range rate [m/s]."""
        if self.range < 1e-6:
            return 0.0
        return np.dot(self.position, self.velocity) / self.range


@dataclass
class CoupledState:
    """
    Coupled translational and rotational state.
    
    Attributes:
        relative_state: Relative translational state
        chaser_attitude: Chaser attitude state
        target_attitude: Target attitude state (optional)
    """
    relative_state: RelativeState
    chaser_attitude: AttitudeState
    target_attitude: Optional[AttitudeState] = None


def lvlh_frame_angular_velocity(elements: OrbitalElements) -> np.ndarray:
    """
    Calculate LVLH frame angular velocity.
    
    Args:
        elements: Target orbital elements
    
    Returns:
        LVLH angular velocity [rad/s] (3x1)
    """
    # For circular orbits, LVLH frame rotates with mean motion
    n = elements.mean_motion
    
    # LVLH angular velocity in LVLH frame
    # For general orbits, this includes orbital rate variations
    r = elements.radius()
    h = elements.angular_momentum_magnitude
    
    # Angular velocity magnitude
    omega_magnitude = h / r**2
    
    # LVLH angular velocity vector (pointing along orbit normal)
    omega_lvlh = np.array([0, 0, omega_magnitude])
    
    return omega_lvlh


def lvlh_frame_angular_acceleration(elements: OrbitalElements,
                                  element_rates: np.ndarray) -> np.ndarray:
    """
    Calculate LVLH frame angular acceleration.
    
    Args:
        elements: Target orbital elements
        element_rates: Rates of orbital elements [da/dt, de/dt, di/dt, dΩ/dt, dω/dt, df/dt]
    
    Returns:
        LVLH angular acceleration [rad/s²] (3x1)
    """
    # Simplified calculation for circular orbits
    # Full implementation would require detailed orbital mechanics
    
    n = elements.mean_motion
    r = elements.radius()
    
    # Rate of change of orbital rate
    da_dt = element_rates[0]
    dn_dt = -1.5 * n * da_dt / elements.a
    
    # Angular acceleration (simplified)
    alpha_lvlh = np.array([0, 0, dn_dt])
    
    return alpha_lvlh


def clohessy_wiltshire_dynamics(relative_state: RelativeState,
                              target_elements: OrbitalElements,
                              control_acceleration: np.ndarray = None) -> np.ndarray:
    """
    Clohessy-Wiltshire relative motion dynamics.
    
    Args:
        relative_state: Current relative state
        target_elements: Target orbital elements
        control_acceleration: Control acceleration in LVLH frame [m/s²] (3x1)
    
    Returns:
        State derivative [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    """
    if control_acceleration is None:
        control_acceleration = np.zeros(3)
    
    if control_acceleration.shape != (3,):
        raise ValueError("Control acceleration must be 3D vector")
    
    x, y, z = relative_state.position
    vx, vy, vz = relative_state.velocity
    
    n = target_elements.mean_motion  # Mean motion
    
    # Clohessy-Wiltshire equations
    ax = 3 * n**2 * x + 2 * n * vy + control_acceleration[0]
    ay = -2 * n * vx + control_acceleration[1]
    az = -n**2 * z + control_acceleration[2]
    
    return np.array([vx, vy, vz, ax, ay, az])


def nonlinear_relative_dynamics(relative_state: RelativeState,
                               target_elements: OrbitalElements,
                               control_acceleration: np.ndarray = None,
                               perturbation_acceleration: np.ndarray = None) -> np.ndarray:
    """
    Nonlinear relative motion dynamics.
    
    Args:
        relative_state: Current relative state
        target_elements: Target orbital elements
        control_acceleration: Control acceleration in LVLH frame [m/s²] (3x1)
        perturbation_acceleration: Perturbation acceleration in LVLH frame [m/s²] (3x1)
    
    Returns:
        State derivative [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    """
    if control_acceleration is None:
        control_acceleration = np.zeros(3)
    if perturbation_acceleration is None:
        perturbation_acceleration = np.zeros(3)
    
    x, y, z = relative_state.position
    vx, vy, vz = relative_state.velocity
    
    # Target position and velocity
    r_target, v_target = orbital_elements_to_cartesian(target_elements)
    r_target_mag = np.linalg.norm(r_target)
    
    # Chaser position in ECI frame (approximate)
    from .orbital_elements import rsw_to_eci_matrix
    T_rsw_to_eci = rsw_to_eci_matrix(target_elements)
    r_relative_eci = T_rsw_to_eci @ relative_state.position
    r_chaser = r_target + r_relative_eci
    r_chaser_mag = np.linalg.norm(r_chaser)
    
    # Gravitational accelerations
    mu = EARTH_MU
    a_target = -mu * r_target / r_target_mag**3
    a_chaser = -mu * r_chaser / r_chaser_mag**3
    
    # Relative acceleration in ECI
    a_rel_eci = a_chaser - a_target
    
    # Transform to LVLH frame
    from .orbital_elements import eci_to_rsw_matrix
    T_eci_to_rsw = eci_to_rsw_matrix(target_elements)
    a_rel_lvlh = T_eci_to_rsw @ a_rel_eci
    
    # LVLH frame rotation effects
    omega_lvlh = lvlh_frame_angular_velocity(target_elements)
    
    # Coriolis and centrifugal accelerations
    a_coriolis = -2 * np.cross(omega_lvlh, relative_state.velocity)
    a_centrifugal = -np.cross(omega_lvlh, np.cross(omega_lvlh, relative_state.position))
    
    # Total acceleration
    a_total = (a_rel_lvlh + a_coriolis + a_centrifugal + 
               control_acceleration + perturbation_acceleration)
    
    return np.array([vx, vy, vz, a_total[0], a_total[1], a_total[2]])


def coupled_translational_rotational_dynamics(coupled_state: CoupledState,
                                             target_elements: OrbitalElements,
                                             chaser_inertia: SpacecraftInertia,
                                             control_force: np.ndarray = None,
                                             control_torque: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coupled translational and rotational dynamics.
    
    Args:
        coupled_state: Current coupled state
        target_elements: Target orbital elements
        chaser_inertia: Chaser spacecraft inertia
        control_force: Control force in chaser body frame [N] (3x1)
        control_torque: Control torque in chaser body frame [N⋅m] (3x1)
    
    Returns:
        Tuple of (translational_derivative, rotational_derivative)
    """
    if control_force is None:
        control_force = np.zeros(3)
    if control_torque is None:
        control_torque = np.zeros(3)
    
    # Extract states
    rel_state = coupled_state.relative_state
    chaser_att = coupled_state.chaser_attitude
    
    # Transform control force from body to LVLH frame
    R_body_to_lvlh = quaternion_to_rotation_matrix(chaser_att.quaternion)
    control_acceleration_lvlh = R_body_to_lvlh @ control_force  # Assuming unit mass
    
    # Translational dynamics
    trans_derivative = nonlinear_relative_dynamics(
        rel_state, target_elements, control_acceleration_lvlh
    )
    
    # Rotational dynamics
    from .attitude_dynamics import quaternion_kinematics, euler_equations, gravity_gradient_torque
    
    # Get chaser position in ECI for gravity gradient calculation
    r_target, _ = orbital_elements_to_cartesian(target_elements)
    from .orbital_elements import rsw_to_eci_matrix
    T_rsw_to_eci = rsw_to_eci_matrix(target_elements)
    r_relative_eci = T_rsw_to_eci @ rel_state.position
    r_chaser_eci = r_target + r_relative_eci
    
    # External torques
    gravity_torque = gravity_gradient_torque(chaser_att.quaternion, r_chaser_eci, chaser_inertia)
    total_torque = gravity_torque + control_torque
    
    # Quaternion kinematics
    q_dot = quaternion_kinematics(chaser_att.quaternion, chaser_att.angular_velocity)
    
    # Angular dynamics
    omega_dot = euler_equations(chaser_att.angular_velocity, chaser_inertia, total_torque)
    
    rot_derivative = np.concatenate([q_dot, omega_dot])
    
    return trans_derivative, rot_derivative


def propagate_relative_state(relative_state: RelativeState,
                           target_elements: OrbitalElements,
                           delta_t: float,
                           control_acceleration: np.ndarray = None,
                           use_nonlinear: bool = False) -> RelativeState:
    """
    Propagate relative state using numerical integration.
    
    Args:
        relative_state: Current relative state
        target_elements: Target orbital elements
        delta_t: Time step [s]
        control_acceleration: Control acceleration [m/s²] (3x1)
        use_nonlinear: Use nonlinear dynamics instead of Clohessy-Wiltshire
    
    Returns:
        Propagated relative state
    """
    if use_nonlinear:
        dynamics_func = nonlinear_relative_dynamics
    else:
        dynamics_func = clohessy_wiltshire_dynamics
    
    # Current state vector
    y0 = np.concatenate([relative_state.position, relative_state.velocity])
    
    # Dynamics function
    y_dot = dynamics_func(relative_state, target_elements, control_acceleration)
    
    # Euler integration (can be upgraded to RK4)
    y_new = y0 + y_dot * delta_t
    
    return RelativeState(
        y_new[0:3], y_new[3:6], relative_state.time + delta_t
    )


def relative_state_from_absolute(chaser_elements: OrbitalElements,
                               target_elements: OrbitalElements) -> RelativeState:
    """
    Calculate relative state from absolute orbital elements.
    
    Args:
        chaser_elements: Chaser orbital elements
        target_elements: Target orbital elements
    
    Returns:
        Relative state in target LVLH frame
    """
    # Get Cartesian states
    r_chaser, v_chaser = orbital_elements_to_cartesian(chaser_elements)
    r_target, v_target = orbital_elements_to_cartesian(target_elements)
    
    # Relative vectors in ECI
    r_rel_eci = r_chaser - r_target
    v_rel_eci = v_chaser - v_target
    
    # Transform to LVLH frame
    from .orbital_elements import eci_to_rsw_matrix
    T_eci_to_lvlh = eci_to_rsw_matrix(target_elements)
    
    r_rel_lvlh = T_eci_to_lvlh @ r_rel_eci
    v_rel_lvlh = T_eci_to_lvlh @ v_rel_eci
    
    # Account for LVLH frame rotation
    omega_lvlh = lvlh_frame_angular_velocity(target_elements)
    v_rel_lvlh -= np.cross(omega_lvlh, r_rel_lvlh)
    
    return RelativeState(r_rel_lvlh, v_rel_lvlh)


def absolute_elements_from_relative(relative_state: RelativeState,
                                  target_elements: OrbitalElements) -> OrbitalElements:
    """
    Calculate chaser absolute orbital elements from relative state.
    
    Args:
        relative_state: Relative state in target LVLH frame
        target_elements: Target orbital elements
    
    Returns:
        Chaser orbital elements
    """
    # Get target Cartesian state
    r_target, v_target = orbital_elements_to_cartesian(target_elements)
    
    # Transform relative state to ECI
    from .orbital_elements import rsw_to_eci_matrix
    T_lvlh_to_eci = rsw_to_eci_matrix(target_elements)
    
    # Account for LVLH frame rotation
    omega_lvlh = lvlh_frame_angular_velocity(target_elements)
    v_rel_corrected = relative_state.velocity + np.cross(omega_lvlh, relative_state.position)
    
    r_rel_eci = T_lvlh_to_eci @ relative_state.position
    v_rel_eci = T_lvlh_to_eci @ v_rel_corrected
    
    # Chaser absolute state
    r_chaser = r_target + r_rel_eci
    v_chaser = v_target + v_rel_eci
    
    # Convert to orbital elements
    from .orbital_elements import cartesian_to_orbital_elements
    return cartesian_to_orbital_elements(r_chaser, v_chaser)


def relative_motion_stm(target_elements: OrbitalElements, 
                       delta_t: float) -> np.ndarray:
    """
    Calculate state transition matrix for Clohessy-Wiltshire dynamics.
    
    Args:
        target_elements: Target orbital elements
        delta_t: Time interval [s]
    
    Returns:
        State transition matrix [6x6]
    """
    n = target_elements.mean_motion
    nt = n * delta_t
    
    cos_nt = np.cos(nt)
    sin_nt = np.sin(nt)
    
    # Clohessy-Wiltshire state transition matrix
    stm = np.zeros((6, 6))
    
    # Position-position block
    stm[0, 0] = 4 - 3*cos_nt
    stm[0, 1] = 0
    stm[0, 2] = 0
    stm[1, 0] = 6*(sin_nt - nt)
    stm[1, 1] = 1
    stm[1, 2] = 0
    stm[2, 0] = 0
    stm[2, 1] = 0
    stm[2, 2] = cos_nt
    
    # Position-velocity block
    stm[0, 3] = sin_nt/n
    stm[0, 4] = 2*(1 - cos_nt)/n
    stm[0, 5] = 0
    stm[1, 3] = 2*(cos_nt - 1)/n
    stm[1, 4] = (4*sin_nt - 3*nt)/n
    stm[1, 5] = 0
    stm[2, 3] = 0
    stm[2, 4] = 0
    stm[2, 5] = sin_nt/n
    
    # Velocity-position block
    stm[3, 0] = 3*n*sin_nt
    stm[3, 1] = 0
    stm[3, 2] = 0
    stm[4, 0] = 6*n*(cos_nt - 1)
    stm[4, 1] = 0
    stm[4, 2] = 0
    stm[5, 0] = 0
    stm[5, 1] = 0
    stm[5, 2] = -n*sin_nt
    
    # Velocity-velocity block
    stm[3, 3] = cos_nt
    stm[3, 4] = 2*sin_nt
    stm[3, 5] = 0
    stm[4, 3] = -2*sin_nt
    stm[4, 4] = 4*cos_nt - 3
    stm[4, 5] = 0
    stm[5, 3] = 0
    stm[5, 4] = 0
    stm[5, 5] = cos_nt
    
    return stm

