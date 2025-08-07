"""
Attitude Dynamics and Kinematics

This module implements spacecraft attitude dynamics using quaternions,
including attitude kinematics, Euler's equations, and gravity gradient torques.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass

from ..utils.constants import EARTH_MU
from ..utils.math_utils import (
    quaternion_normalize, quaternion_multiply, quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion, skew_symmetric
)


@dataclass
class SpacecraftInertia:
    """
    Spacecraft inertia properties.
    
    Attributes:
        Ixx, Iyy, Izz: Principal moments of inertia [kg⋅m²]
        Ixy, Ixz, Iyz: Products of inertia [kg⋅m²]
    """
    Ixx: float
    Iyy: float
    Izz: float
    Ixy: float = 0.0
    Ixz: float = 0.0
    Iyz: float = 0.0
    
    @property
    def inertia_matrix(self) -> np.ndarray:
        """Inertia matrix [3x3]."""
        return np.array([
            [self.Ixx, -self.Ixy, -self.Ixz],
            [-self.Ixy, self.Iyy, -self.Iyz],
            [-self.Ixz, -self.Iyz, self.Izz]
        ])
    
    @property
    def principal_moments(self) -> np.ndarray:
        """Principal moments of inertia [3x1]."""
        return np.array([self.Ixx, self.Iyy, self.Izz])
    
    def is_principal_axes(self, tolerance: float = 1e-6) -> bool:
        """Check if inertia is aligned with principal axes."""
        return (abs(self.Ixy) < tolerance and 
                abs(self.Ixz) < tolerance and 
                abs(self.Iyz) < tolerance)


@dataclass
class AttitudeState:
    """
    Complete attitude state representation.
    
    Attributes:
        quaternion: Attitude quaternion [q0, q1, q2, q3] (scalar first)
        angular_velocity: Angular velocity [rad/s] (3x1)
        time: Time [s]
    """
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    time: float = 0.0
    
    def __post_init__(self):
        """Validate and normalize attitude state."""
        if self.quaternion.shape != (4,):
            raise ValueError("Quaternion must be 4-element vector")
        if self.angular_velocity.shape != (3,):
            raise ValueError("Angular velocity must be 3-element vector")
        
        self.quaternion = quaternion_normalize(self.quaternion)
    
    @property
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix from quaternion [3x3]."""
        return quaternion_to_rotation_matrix(self.quaternion)
    
    def copy(self) -> 'AttitudeState':
        """Create a copy of the attitude state."""
        return AttitudeState(
            self.quaternion.copy(),
            self.angular_velocity.copy(),
            self.time
        )


def quaternion_kinematics(quaternion: np.ndarray, angular_velocity: np.ndarray) -> np.ndarray:
    """
    Calculate quaternion time derivative from angular velocity.
    
    Args:
        quaternion: Current quaternion [q0, q1, q2, q3]
        angular_velocity: Angular velocity [rad/s] (3x1)
    
    Returns:
        Quaternion derivative [dq0/dt, dq1/dt, dq2/dt, dq3/dt]
    """
    if quaternion.shape != (4,) or angular_velocity.shape != (3,):
        raise ValueError("Invalid input dimensions")
    
    q0, q1, q2, q3 = quaternion
    wx, wy, wz = angular_velocity
    
    # Quaternion kinematic matrix
    omega_matrix = 0.5 * np.array([
        [-q1, -q2, -q3],
        [ q0, -q3,  q2],
        [ q3,  q0, -q1],
        [-q2,  q1,  q0]
    ])
    
    return omega_matrix @ angular_velocity


def euler_equations(angular_velocity: np.ndarray, 
                   inertia: SpacecraftInertia,
                   external_torque: np.ndarray) -> np.ndarray:
    """
    Calculate angular acceleration using Euler's equations.
    
    Args:
        angular_velocity: Angular velocity [rad/s] (3x1)
        inertia: Spacecraft inertia properties
        external_torque: External torque [N⋅m] (3x1)
    
    Returns:
        Angular acceleration [rad/s²] (3x1)
    """
    if angular_velocity.shape != (3,) or external_torque.shape != (3,):
        raise ValueError("Angular velocity and torque must be 3D vectors")
    
    I = inertia.inertia_matrix
    omega = angular_velocity
    
    # Euler's equation: I * omega_dot + omega × (I * omega) = T_external
    gyroscopic_torque = np.cross(omega, I @ omega)
    angular_acceleration = np.linalg.solve(I, external_torque - gyroscopic_torque)
    
    return angular_acceleration


def gravity_gradient_torque(quaternion: np.ndarray, 
                          position_eci: np.ndarray,
                          inertia: SpacecraftInertia) -> np.ndarray:
    """
    Calculate gravity gradient torque.
    
    Args:
        quaternion: Attitude quaternion [q0, q1, q2, q3]
        position_eci: Position vector in ECI frame [m] (3x1)
        inertia: Spacecraft inertia properties
    
    Returns:
        Gravity gradient torque in body frame [N⋅m] (3x1)
    """
    if quaternion.shape != (4,) or position_eci.shape != (3,):
        raise ValueError("Invalid input dimensions")
    
    r = np.linalg.norm(position_eci)
    if r == 0:
        return np.zeros(3)
    
    # Unit vector from Earth center to spacecraft in ECI
    r_hat_eci = position_eci / r
    
    # Transform to body frame
    R_eci_to_body = quaternion_to_rotation_matrix(quaternion).T
    r_hat_body = R_eci_to_body @ r_hat_eci
    
    # Gravity gradient torque
    mu_over_r3 = EARTH_MU / r**3
    I = inertia.inertia_matrix
    
    torque = 3 * mu_over_r3 * np.cross(r_hat_body, I @ r_hat_body)
    
    return torque


def magnetic_dipole_torque(quaternion: np.ndarray,
                         magnetic_field_eci: np.ndarray,
                         magnetic_dipole_body: np.ndarray) -> np.ndarray:
    """
    Calculate magnetic dipole torque.
    
    Args:
        quaternion: Attitude quaternion [q0, q1, q2, q3]
        magnetic_field_eci: Magnetic field vector in ECI frame [T] (3x1)
        magnetic_dipole_body: Magnetic dipole moment in body frame [A⋅m²] (3x1)
    
    Returns:
        Magnetic torque in body frame [N⋅m] (3x1)
    """
    if (quaternion.shape != (4,) or 
        magnetic_field_eci.shape != (3,) or 
        magnetic_dipole_body.shape != (3,)):
        raise ValueError("Invalid input dimensions")
    
    # Transform magnetic field to body frame
    R_eci_to_body = quaternion_to_rotation_matrix(quaternion).T
    B_body = R_eci_to_body @ magnetic_field_eci
    
    # Magnetic torque: T = m × B
    torque = np.cross(magnetic_dipole_body, B_body)
    
    return torque


def solar_radiation_pressure_torque(quaternion: np.ndarray,
                                   sun_vector_eci: np.ndarray,
                                   surface_properties: dict) -> np.ndarray:
    """
    Calculate solar radiation pressure torque.
    
    Args:
        quaternion: Attitude quaternion [q0, q1, q2, q3]
        sun_vector_eci: Unit vector to Sun in ECI frame (3x1)
        surface_properties: Dictionary with surface area, reflectivity, and center of pressure
    
    Returns:
        SRP torque in body frame [N⋅m] (3x1)
    """
    # Simplified SRP torque calculation
    # In practice, this requires detailed surface modeling
    
    if quaternion.shape != (4,) or sun_vector_eci.shape != (3,):
        raise ValueError("Invalid input dimensions")
    
    # Transform sun vector to body frame
    R_eci_to_body = quaternion_to_rotation_matrix(quaternion).T
    sun_body = R_eci_to_body @ sun_vector_eci
    
    # Simplified calculation (placeholder for detailed implementation)
    # Real implementation would require surface panel modeling
    area = surface_properties.get('area', 1.0)
    reflectivity = surface_properties.get('reflectivity', 0.3)
    cp_offset = surface_properties.get('center_of_pressure_offset', np.zeros(3))
    
    # Solar pressure constant
    P_solar = 4.56e-6  # N/m² at 1 AU
    
    # Force magnitude (simplified)
    force_magnitude = P_solar * area * (1 + reflectivity) * max(0, np.dot(sun_body, np.array([1, 0, 0])))
    
    # Torque = r × F (simplified)
    force_body = force_magnitude * np.array([1, 0, 0])  # Simplified direction
    torque = np.cross(cp_offset, force_body)
    
    return torque


def propagate_attitude_state(state: AttitudeState,
                           inertia: SpacecraftInertia,
                           external_torque: np.ndarray,
                           delta_t: float,
                           method: str = 'rk4') -> AttitudeState:
    """
    Propagate attitude state using numerical integration.
    
    Args:
        state: Current attitude state
        inertia: Spacecraft inertia properties
        external_torque: External torque [N⋅m] (3x1)
        delta_t: Time step [s]
        method: Integration method ('euler', 'rk4')
    
    Returns:
        Propagated attitude state
    """
    if external_torque.shape != (3,):
        raise ValueError("External torque must be 3D vector")
    
    def attitude_dynamics(t: float, y: np.ndarray) -> np.ndarray:
        """Attitude dynamics function for integration."""
        q = y[0:4]
        omega = y[4:7]
        
        # Normalize quaternion
        q = quaternion_normalize(q)
        
        # Quaternion kinematics
        q_dot = quaternion_kinematics(q, omega)
        
        # Angular dynamics
        omega_dot = euler_equations(omega, inertia, external_torque)
        
        return np.concatenate([q_dot, omega_dot])
    
    # Current state vector
    y0 = np.concatenate([state.quaternion, state.angular_velocity])
    
    if method.lower() == 'euler':
        # Euler integration
        y_dot = attitude_dynamics(state.time, y0)
        y_new = y0 + y_dot * delta_t
    
    elif method.lower() == 'rk4':
        # Runge-Kutta 4th order
        k1 = attitude_dynamics(state.time, y0)
        k2 = attitude_dynamics(state.time + delta_t/2, y0 + k1*delta_t/2)
        k3 = attitude_dynamics(state.time + delta_t/2, y0 + k2*delta_t/2)
        k4 = attitude_dynamics(state.time + delta_t, y0 + k3*delta_t)
        
        y_new = y0 + (k1 + 2*k2 + 2*k3 + k4) * delta_t / 6
    
    else:
        raise ValueError(f"Unknown integration method: {method}")
    
    # Extract new state
    q_new = quaternion_normalize(y_new[0:4])
    omega_new = y_new[4:7]
    
    return AttitudeState(q_new, omega_new, state.time + delta_t)


def attitude_error_quaternion(q_desired: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
    """
    Calculate attitude error quaternion.
    
    Args:
        q_desired: Desired quaternion [q0, q1, q2, q3]
        q_actual: Actual quaternion [q0, q1, q2, q3]
    
    Returns:
        Error quaternion [q0, q1, q2, q3]
    """
    if q_desired.shape != (4,) or q_actual.shape != (4,):
        raise ValueError("Quaternions must be 4-element vectors")
    
    # Normalize inputs
    q_desired = quaternion_normalize(q_desired)
    q_actual = quaternion_normalize(q_actual)
    
    # Error quaternion: q_error = q_desired * q_actual^(-1)
    # For unit quaternions, q^(-1) = q_conjugate
    q_actual_conj = np.array([q_actual[0], -q_actual[1], -q_actual[2], -q_actual[3]])
    
    q_error = quaternion_multiply(q_desired, q_actual_conj)
    
    # Ensure scalar part is positive (shortest rotation)
    if q_error[0] < 0:
        q_error = -q_error
    
    return q_error


def attitude_error_angle_axis(q_error: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Convert error quaternion to angle-axis representation.
    
    Args:
        q_error: Error quaternion [q0, q1, q2, q3]
    
    Returns:
        Tuple of (error_angle [rad], error_axis [3x1])
    """
    if q_error.shape != (4,):
        raise ValueError("Error quaternion must be 4-element vector")
    
    q_error = quaternion_normalize(q_error)
    q0, q1, q2, q3 = q_error
    
    # Error angle
    error_angle = 2 * np.arccos(abs(q0))
    
    # Error axis
    sin_half_angle = np.sqrt(q1**2 + q2**2 + q3**2)
    
    if sin_half_angle < 1e-6:  # Small angle
        error_axis = np.array([1, 0, 0])  # Arbitrary axis
    else:
        error_axis = np.array([q1, q2, q3]) / sin_half_angle
    
    return error_angle, error_axis


def quaternion_from_euler_angles(roll: float, pitch: float, yaw: float,
                                sequence: str = 'ZYX') -> np.ndarray:
    """
    Create quaternion from Euler angles.
    
    Args:
        roll: Roll angle [rad]
        pitch: Pitch angle [rad]
        yaw: Yaw angle [rad]
        sequence: Rotation sequence ('ZYX', 'XYZ', etc.)
    
    Returns:
        Quaternion [q0, q1, q2, q3]
    """
    from ..utils.math_utils import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z
    
    if sequence.upper() == 'ZYX':  # Yaw-Pitch-Roll
        R = rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)
    elif sequence.upper() == 'XYZ':
        R = rotation_matrix_x(roll) @ rotation_matrix_y(pitch) @ rotation_matrix_z(yaw)
    else:
        raise ValueError(f"Unsupported rotation sequence: {sequence}")
    
    return rotation_matrix_to_quaternion(R)


def euler_angles_from_quaternion(quaternion: np.ndarray, 
                                sequence: str = 'ZYX') -> Tuple[float, float, float]:
    """
    Extract Euler angles from quaternion.
    
    Args:
        quaternion: Quaternion [q0, q1, q2, q3]
        sequence: Rotation sequence ('ZYX', 'XYZ', etc.)
    
    Returns:
        Tuple of (roll, pitch, yaw) [rad]
    """
    R = quaternion_to_rotation_matrix(quaternion)
    
    if sequence.upper() == 'ZYX':  # Yaw-Pitch-Roll
        # Extract angles from rotation matrix
        pitch = np.arcsin(-R[2, 0])
        
        if abs(np.cos(pitch)) > 1e-6:  # Not at singularity
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:  # Gimbal lock
            roll = 0.0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
    
    elif sequence.upper() == 'XYZ':
        yaw = np.arcsin(R[0, 1])
        
        if abs(np.cos(yaw)) > 1e-6:
            roll = np.arctan2(-R[1, 2], R[2, 2])
            pitch = np.arctan2(-R[0, 2], R[0, 0])
        else:
            roll = 0.0
            pitch = np.arctan2(R[1, 0], R[1, 1])
    
    else:
        raise ValueError(f"Unsupported rotation sequence: {sequence}")
    
    return roll, pitch, yaw

