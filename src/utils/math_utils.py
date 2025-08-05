"""
Mathematical Utilities for Orbital Mechanics

This module provides mathematical functions and utilities commonly used
in orbital mechanics and spacecraft dynamics calculations.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import Union, Tuple
from .constants import PI, TWO_PI, TOLERANCE_ANGLE


def normalize_angle(angle: float, center: float = 0.0) -> float:
    """
    Normalize angle to be within [center-π, center+π].
    
    Args:
        angle: Angle to normalize [rad]
        center: Center of the normalized range [rad]
    
    Returns:
        Normalized angle [rad]
    """
    normalized = angle - center
    normalized = normalized - TWO_PI * np.floor((normalized + PI) / TWO_PI)
    return normalized + center


def wrap_to_2pi(angle: float) -> float:
    """
    Wrap angle to [0, 2π] range.
    
    Args:
        angle: Input angle [rad]
    
    Returns:
        Wrapped angle [rad]
    """
    return angle - TWO_PI * np.floor(angle / TWO_PI)


def wrap_to_pi(angle: float) -> float:
    """
    Wrap angle to [-π, π] range.
    
    Args:
        angle: Input angle [rad]
    
    Returns:
        Wrapped angle [rad]
    """
    return normalize_angle(angle, 0.0)


def skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from 3D vector.
    
    Args:
        vector: 3D vector [3x1]
    
    Returns:
        Skew-symmetric matrix [3x3]
    """
    if vector.shape != (3,):
        raise ValueError("Input must be a 3D vector")
    
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])


def rotation_matrix_x(angle: float) -> np.ndarray:
    """
    Create rotation matrix about X-axis.
    
    Args:
        angle: Rotation angle [rad]
    
    Returns:
        Rotation matrix [3x3]
    """
    c = np.cos(angle)
    s = np.sin(angle)
    
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle: float) -> np.ndarray:
    """
    Create rotation matrix about Y-axis.
    
    Args:
        angle: Rotation angle [rad]
    
    Returns:
        Rotation matrix [3x3]
    """
    c = np.cos(angle)
    s = np.sin(angle)
    
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle: float) -> np.ndarray:
    """
    Create rotation matrix about Z-axis.
    
    Args:
        angle: Rotation angle [rad]
    
    Returns:
        Rotation matrix [3x3]
    """
    c = np.cos(angle)
    s = np.sin(angle)
    
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rotation_matrix_313(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Create rotation matrix using 3-1-3 Euler angle sequence.
    
    Args:
        phi: First rotation about Z-axis [rad]
        theta: Second rotation about X-axis [rad]
        psi: Third rotation about Z-axis [rad]
    
    Returns:
        Rotation matrix [3x3]
    """
    return rotation_matrix_z(psi) @ rotation_matrix_x(theta) @ rotation_matrix_z(phi)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion to unit length.
    
    Args:
        q: Quaternion [q0, q1, q2, q3] where q0 is scalar part
    
    Returns:
        Normalized quaternion
    """
    if q.shape != (4,):
        raise ValueError("Quaternion must be 4-element vector")
    
    norm = np.linalg.norm(q)
    if norm < TOLERANCE_ANGLE:
        raise ValueError("Quaternion norm too small")
    
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion [q0, q1, q2, q3]
        q2: Second quaternion [q0, q1, q2, q3]
    
    Returns:
        Product quaternion
    """
    if q1.shape != (4,) or q2.shape != (4,):
        raise ValueError("Quaternions must be 4-element vectors")
    
    q0_1, q1_1, q2_1, q3_1 = q1
    q0_2, q1_2, q2_2, q3_2 = q2
    
    return np.array([
        q0_1*q0_2 - q1_1*q1_2 - q2_1*q2_2 - q3_1*q3_2,
        q0_1*q1_2 + q1_1*q0_2 + q2_1*q3_2 - q3_1*q2_2,
        q0_1*q2_2 - q1_1*q3_2 + q2_1*q0_2 + q3_1*q1_2,
        q0_1*q3_2 + q1_1*q2_2 - q2_1*q1_2 + q3_1*q0_2
    ])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion [q0, q1, q2, q3]
    
    Returns:
        Rotation matrix [3x3]
    """
    q = quaternion_normalize(q)
    q0, q1, q2, q3 = q
    
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3), 1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2+q2**2)]
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: Rotation matrix [3x3]
    
    Returns:
        Quaternion [q0, q1, q2, q3]
    """
    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")
    
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * q0
        q0 = 0.25 * s
        q1 = (R[2, 1] - R[1, 2]) / s
        q2 = (R[0, 2] - R[2, 0]) / s
        q3 = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * q1
        q0 = (R[2, 1] - R[1, 2]) / s
        q1 = 0.25 * s
        q2 = (R[0, 1] + R[1, 0]) / s
        q3 = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * q2
        q0 = (R[0, 2] - R[2, 0]) / s
        q1 = (R[0, 1] + R[1, 0]) / s
        q2 = 0.25 * s
        q3 = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * q3
        q0 = (R[1, 0] - R[0, 1]) / s
        q1 = (R[0, 2] + R[2, 0]) / s
        q2 = (R[1, 2] + R[2, 1]) / s
        q3 = 0.25 * s
    
    return np.array([q0, q1, q2, q3])


def solve_kepler_equation(mean_anomaly: float, eccentricity: float, 
                         tolerance: float = 1e-12, max_iterations: int = 100) -> float:
    """
    Solve Kepler's equation for eccentric anomaly using Newton-Raphson method.
    
    Args:
        mean_anomaly: Mean anomaly [rad]
        eccentricity: Orbital eccentricity
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
    
    Returns:
        Eccentric anomaly [rad]
    """
    # Initial guess
    E = mean_anomaly if eccentricity < 0.8 else PI
    
    for _ in range(max_iterations):
        f = E - eccentricity * np.sin(E) - mean_anomaly
        df = 1 - eccentricity * np.cos(E)
        
        delta_E = -f / df
        E += delta_E
        
        if abs(delta_E) < tolerance:
            return E
    
    raise RuntimeError(f"Kepler equation did not converge after {max_iterations} iterations")

