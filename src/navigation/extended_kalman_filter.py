"""
Extended Kalman Filter for Relative Navigation

This module implements an Extended Kalman Filter (EKF) for relative spacecraft
navigation using multiple sensor types including LIDAR, star trackers, and IMUs.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from ..dynamics.relative_motion import RelativeState, relative_motion_stm
from ..dynamics.orbital_elements import OrbitalElements
from ..dynamics.attitude_dynamics import AttitudeState
from ..utils.math_utils import quaternion_to_rotation_matrix


class SensorType(Enum):
    """Enumeration of sensor types."""
    LIDAR = "lidar"
    STAR_TRACKER = "star_tracker"
    GYROSCOPE = "gyroscope"
    ACCELEROMETER = "accelerometer"
    GPS = "gps"


@dataclass
class EKFState:
    """
    Extended Kalman Filter state representation.
    
    State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    - Position and velocity in LVLH frame
    - Attitude quaternion (chaser relative to LVLH)
    - Angular velocity in body frame
    """
    position: np.ndarray          # Position in LVLH frame [m] (3x1)
    velocity: np.ndarray          # Velocity in LVLH frame [m/s] (3x1)
    quaternion: np.ndarray        # Attitude quaternion [qw, qx, qy, qz] (4x1)
    angular_velocity: np.ndarray  # Angular velocity in body frame [rad/s] (3x1)
    covariance: np.ndarray        # State covariance matrix (13x13)
    time: float = 0.0
    
    def __post_init__(self):
        """Validate EKF state dimensions."""
        if self.position.shape != (3,):
            raise ValueError("Position must be 3D vector")
        if self.velocity.shape != (3,):
            raise ValueError("Velocity must be 3D vector")
        if self.quaternion.shape != (4,):
            raise ValueError("Quaternion must be 4D vector")
        if self.angular_velocity.shape != (3,):
            raise ValueError("Angular velocity must be 3D vector")
        if self.covariance.shape != (13, 13):
            raise ValueError("Covariance must be 13x13 matrix")
    
    @property
    def state_vector(self) -> np.ndarray:
        """Get state as vector [13x1]."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.quaternion,
            self.angular_velocity
        ])
    
    @classmethod
    def from_state_vector(cls, x: np.ndarray, P: np.ndarray, time: float = 0.0) -> 'EKFState':
        """Create EKF state from state vector."""
        if x.shape != (13,):
            raise ValueError("State vector must be 13D")
        
        return cls(
            position=x[0:3],
            velocity=x[3:6],
            quaternion=x[6:10],
            angular_velocity=x[10:13],
            covariance=P,
            time=time
        )


@dataclass
class ProcessNoise:
    """Process noise parameters for EKF."""
    position_noise: float = 1e-6      # Position process noise [m²/s³]
    velocity_noise: float = 1e-8      # Velocity process noise [m²/s⁵]
    attitude_noise: float = 1e-10     # Attitude process noise [rad²/s]
    angular_vel_noise: float = 1e-12  # Angular velocity process noise [rad²/s³]


@dataclass
class Measurement:
    """Sensor measurement data."""
    sensor_type: SensorType
    data: np.ndarray
    covariance: np.ndarray
    time: float
    sensor_id: str = "default"


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for relative spacecraft navigation.
    
    This implementation handles coupled translational and rotational motion
    with multiple sensor types for robust navigation.
    """
    
    def __init__(self, initial_state: EKFState, process_noise: ProcessNoise):
        """
        Initialize Extended Kalman Filter.
        
        Args:
            initial_state: Initial state estimate
            process_noise: Process noise parameters
        """
        self.state = initial_state
        self.process_noise = process_noise
        self.measurement_history: List[Measurement] = []
        
        # Innovation statistics for filter monitoring
        self.innovation_history: List[np.ndarray] = []
        self.innovation_covariance_history: List[np.ndarray] = []
    
    def predict(self, target_elements: OrbitalElements, delta_t: float,
                control_acceleration: Optional[np.ndarray] = None,
                control_torque: Optional[np.ndarray] = None) -> None:
        """
        Prediction step of the EKF.
        
        Args:
            target_elements: Target orbital elements
            delta_t: Time step [s]
            control_acceleration: Control acceleration in LVLH frame [m/s²] (3x1)
            control_torque: Control torque in body frame [N⋅m] (3x1)
        """
        if control_acceleration is None:
            control_acceleration = np.zeros(3)
        if control_torque is None:
            control_torque = np.zeros(3)
        
        # Current state
        x = self.state.state_vector
        P = self.state.covariance
        
        # Propagate state using nonlinear dynamics
        x_pred = self._propagate_state(x, target_elements, delta_t, 
                                     control_acceleration, control_torque)
        
        # Compute state transition matrix (Jacobian)
        F = self._compute_state_transition_matrix(x, target_elements, delta_t)
        
        # Process noise matrix
        Q = self._compute_process_noise_matrix(delta_t)
        
        # Propagate covariance
        P_pred = F @ P @ F.T + Q
        
        # Update state
        self.state = EKFState.from_state_vector(x_pred, P_pred, 
                                               self.state.time + delta_t)
    
    def update(self, measurement: Measurement, target_elements: OrbitalElements) -> None:
        """
        Update step of the EKF.
        
        Args:
            measurement: Sensor measurement
            target_elements: Target orbital elements
        """
        # Current state
        x = self.state.state_vector
        P = self.state.covariance
        
        # Predicted measurement
        h_pred = self._measurement_model(x, measurement.sensor_type, target_elements)
        
        # Measurement Jacobian
        H = self._compute_measurement_jacobian(x, measurement.sensor_type, target_elements)
        
        # Innovation
        y = measurement.data - h_pred
        
        # Innovation covariance
        S = H @ P @ H.T + measurement.covariance
        
        # Kalman gain
        K = P @ H.T @ np.linalg.inv(S)
        
        # State update
        x_updated = x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(13) - K @ H
        P_updated = I_KH @ P @ I_KH.T + K @ measurement.covariance @ K.T
        
        # Normalize quaternion
        x_updated[6:10] = x_updated[6:10] / np.linalg.norm(x_updated[6:10])
        
        # Update state
        self.state = EKFState.from_state_vector(x_updated, P_updated, measurement.time)
        
        # Store measurement and innovation for analysis
        self.measurement_history.append(measurement)
        self.innovation_history.append(y)
        self.innovation_covariance_history.append(S)
    
    def _propagate_state(self, x: np.ndarray, target_elements: OrbitalElements,
                        delta_t: float, control_accel: np.ndarray,
                        control_torque: np.ndarray) -> np.ndarray:
        """Propagate state using nonlinear dynamics."""
        # Extract state components
        pos = x[0:3]
        vel = x[3:6]
        quat = x[6:10]
        omega = x[10:13]
        
        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)
        
        # Relative motion dynamics (simplified Clohessy-Wiltshire)
        n = target_elements.mean_motion
        
        # Translational dynamics with control
        pos_new = pos + vel * delta_t
        
        # Clohessy-Wiltshire acceleration
        accel_cw = np.array([
            3 * n**2 * pos[0] + 2 * n * vel[1],
            -2 * n * vel[0],
            -n**2 * pos[2]
        ])
        
        vel_new = vel + (accel_cw + control_accel) * delta_t
        
        # Attitude dynamics
        from ..dynamics.attitude_dynamics import quaternion_kinematics
        
        # Quaternion kinematics
        q_dot = quaternion_kinematics(quat, omega)
        quat_new = quat + q_dot * delta_t
        quat_new = quat_new / np.linalg.norm(quat_new)
        
        # Angular dynamics (simplified - no external torques except control)
        omega_new = omega + control_torque * delta_t  # Assuming unit inertia
        
        return np.concatenate([pos_new, vel_new, quat_new, omega_new])
    
    def _compute_state_transition_matrix(self, x: np.ndarray, 
                                       target_elements: OrbitalElements,
                                       delta_t: float) -> np.ndarray:
        """Compute state transition matrix (Jacobian of dynamics)."""
        n = target_elements.mean_motion
        
        # Initialize Jacobian
        F = np.eye(13)
        
        # Position derivatives
        F[0:3, 3:6] = np.eye(3) * delta_t  # ∂pos/∂vel
        
        # Velocity derivatives (Clohessy-Wiltshire)
        F[3, 0] = 3 * n**2 * delta_t      # ∂vx/∂x
        F[3, 4] = 2 * n * delta_t         # ∂vx/∂vy
        F[4, 3] = -2 * n * delta_t        # ∂vy/∂vx
        F[5, 2] = -n**2 * delta_t         # ∂vz/∂z
        
        # Attitude dynamics (linearized quaternion kinematics)
        quat = x[6:10]
        omega = x[10:13]
        
        # ∂quat/∂quat (quaternion kinematics Jacobian)
        F[6:10, 6:10] = self._quaternion_kinematics_jacobian_q(quat, omega, delta_t)
        
        # ∂quat/∂omega (quaternion kinematics Jacobian)
        F[6:10, 10:13] = self._quaternion_kinematics_jacobian_omega(quat, delta_t)
        
        # Angular velocity dynamics (identity for simplified model)
        F[10:13, 10:13] = np.eye(3)
        
        return F
    
    def _quaternion_kinematics_jacobian_q(self, quat: np.ndarray, 
                                        omega: np.ndarray, delta_t: float) -> np.ndarray:
        """Compute Jacobian of quaternion kinematics w.r.t. quaternion."""
        qw, qx, qy, qz = quat
        wx, wy, wz = omega
        
        # Quaternion kinematics matrix
        Omega = 0.5 * np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw]
        ])
        
        # Jacobian of quaternion update
        dq_dq = np.eye(4) + Omega @ np.array([[wx], [wy], [wz]]) @ np.array([[0, -1, 0, 0]]) * delta_t
        
        # Simplified: identity + small perturbation
        return np.eye(4) + 0.5 * delta_t * np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
    
    def _quaternion_kinematics_jacobian_omega(self, quat: np.ndarray, 
                                            delta_t: float) -> np.ndarray:
        """Compute Jacobian of quaternion kinematics w.r.t. angular velocity."""
        qw, qx, qy, qz = quat
        
        return 0.5 * delta_t * np.array([
            [-qx, -qy, -qz],
            [ qw, -qz,  qy],
            [ qz,  qw, -qx],
            [-qy,  qx,  qw]
        ])
    
    def _compute_process_noise_matrix(self, delta_t: float) -> np.ndarray:
        """Compute process noise covariance matrix."""
        Q = np.zeros((13, 13))
        
        # Position noise (integrated from acceleration noise)
        Q[0:3, 0:3] = np.eye(3) * self.process_noise.position_noise * delta_t**3 / 3
        Q[0:3, 3:6] = np.eye(3) * self.process_noise.position_noise * delta_t**2 / 2
        Q[3:6, 0:3] = np.eye(3) * self.process_noise.position_noise * delta_t**2 / 2
        Q[3:6, 3:6] = np.eye(3) * self.process_noise.velocity_noise * delta_t
        
        # Attitude noise
        Q[6:10, 6:10] = np.eye(4) * self.process_noise.attitude_noise * delta_t
        
        # Angular velocity noise
        Q[10:13, 10:13] = np.eye(3) * self.process_noise.angular_vel_noise * delta_t
        
        return Q
    
    def _measurement_model(self, x: np.ndarray, sensor_type: SensorType,
                          target_elements: OrbitalElements) -> np.ndarray:
        """Compute predicted measurement based on state."""
        pos = x[0:3]
        vel = x[3:6]
        quat = x[6:10]
        omega = x[10:13]
        
        if sensor_type == SensorType.LIDAR:
            # LIDAR measures range and range-rate
            range_val = np.linalg.norm(pos)
            range_rate = np.dot(pos, vel) / range_val if range_val > 1e-6 else 0.0
            return np.array([range_val, range_rate])
        
        elif sensor_type == SensorType.STAR_TRACKER:
            # Star tracker measures attitude (quaternion)
            return quat
        
        elif sensor_type == SensorType.GYROSCOPE:
            # Gyroscope measures angular velocity
            return omega
        
        elif sensor_type == SensorType.ACCELEROMETER:
            # Accelerometer measures specific force in body frame
            # Transform gravitational acceleration to body frame
            R_lvlh_to_body = quaternion_to_rotation_matrix(quat).T
            
            # Simplified: assume only gravitational acceleration
            n = target_elements.mean_motion
            accel_lvlh = np.array([3 * n**2 * pos[0], 0, -n**2 * pos[2]])
            accel_body = R_lvlh_to_body @ accel_lvlh
            
            return accel_body
        
        elif sensor_type == SensorType.GPS:
            # GPS measures position in LVLH frame
            return pos
        
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    def _compute_measurement_jacobian(self, x: np.ndarray, sensor_type: SensorType,
                                    target_elements: OrbitalElements) -> np.ndarray:
        """Compute measurement Jacobian matrix."""
        pos = x[0:3]
        vel = x[3:6]
        quat = x[6:10]
        
        if sensor_type == SensorType.LIDAR:
            # LIDAR Jacobian: [∂range/∂x, ∂range_rate/∂x]
            range_val = np.linalg.norm(pos)
            
            if range_val < 1e-6:
                H = np.zeros((2, 13))
            else:
                # Range Jacobian
                drange_dpos = pos / range_val
                
                # Range rate Jacobian
                drange_rate_dpos = (vel * range_val - pos * np.dot(pos, vel) / range_val) / range_val**2
                drange_rate_dvel = pos / range_val
                
                H = np.zeros((2, 13))
                H[0, 0:3] = drange_dpos
                H[1, 0:3] = drange_rate_dpos
                H[1, 3:6] = drange_rate_dvel
            
            return H
        
        elif sensor_type == SensorType.STAR_TRACKER:
            # Star tracker Jacobian: ∂quat/∂x
            H = np.zeros((4, 13))
            H[0:4, 6:10] = np.eye(4)
            return H
        
        elif sensor_type == SensorType.GYROSCOPE:
            # Gyroscope Jacobian: ∂omega/∂x
            H = np.zeros((3, 13))
            H[0:3, 10:13] = np.eye(3)
            return H
        
        elif sensor_type == SensorType.ACCELEROMETER:
            # Accelerometer Jacobian: ∂accel_body/∂x
            H = np.zeros((3, 13))
            
            # Simplified: assume linear relationship
            # In practice, this requires careful derivation of rotation matrix derivatives
            R_lvlh_to_body = quaternion_to_rotation_matrix(quat).T
            n = target_elements.mean_motion
            
            # ∂accel_body/∂pos
            daccel_lvlh_dpos = np.array([[3 * n**2, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, -n**2]])
            H[0:3, 0:3] = R_lvlh_to_body @ daccel_lvlh_dpos
            
            # ∂accel_body/∂quat (rotation matrix derivatives - complex)
            # Simplified implementation
            H[0:3, 6:10] = np.zeros((3, 4))
            
            return H
        
        elif sensor_type == SensorType.GPS:
            # GPS Jacobian: ∂pos/∂x
            H = np.zeros((3, 13))
            H[0:3, 0:3] = np.eye(3)
            return H
        
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    def get_innovation_statistics(self) -> Dict[str, float]:
        """Compute innovation statistics for filter monitoring."""
        if not self.innovation_history:
            return {}
        
        # Recent innovations (last 10 measurements)
        recent_innovations = self.innovation_history[-10:]
        recent_covariances = self.innovation_covariance_history[-10:]
        
        # Normalized innovation squared (NIS)
        nis_values = []
        for y, S in zip(recent_innovations, recent_covariances):
            try:
                nis = y.T @ np.linalg.inv(S) @ y
                nis_values.append(nis)
            except np.linalg.LinAlgError:
                continue
        
        if not nis_values:
            return {}
        
        return {
            'mean_nis': np.mean(nis_values),
            'std_nis': np.std(nis_values),
            'innovation_norm': np.linalg.norm(recent_innovations[-1]) if recent_innovations else 0.0,
            'num_measurements': len(self.measurement_history)
        }
    
    def reset_covariance(self, new_covariance: np.ndarray) -> None:
        """Reset filter covariance (for reinitialization)."""
        if new_covariance.shape != (13, 13):
            raise ValueError("Covariance must be 13x13 matrix")
        
        self.state.covariance = new_covariance.copy()
    
    def get_position_uncertainty(self) -> float:
        """Get 3-sigma position uncertainty."""
        pos_cov = self.state.covariance[0:3, 0:3]
        return 3 * np.sqrt(np.trace(pos_cov))
    
    def get_velocity_uncertainty(self) -> float:
        """Get 3-sigma velocity uncertainty."""
        vel_cov = self.state.covariance[3:6, 3:6]
        return 3 * np.sqrt(np.trace(vel_cov))
    
    def get_attitude_uncertainty(self) -> float:
        """Get 3-sigma attitude uncertainty (angle)."""
        # Simplified: use quaternion covariance trace
        att_cov = self.state.covariance[6:10, 6:10]
        return 3 * np.sqrt(np.trace(att_cov))


def create_initial_ekf_state(relative_state: RelativeState,
                           attitude_state: AttitudeState,
                           initial_covariance: Optional[np.ndarray] = None) -> EKFState:
    """
    Create initial EKF state from relative and attitude states.
    
    Args:
        relative_state: Initial relative state
        attitude_state: Initial attitude state
        initial_covariance: Initial covariance matrix (13x13)
    
    Returns:
        Initial EKF state
    """
    if initial_covariance is None:
        # Default initial covariance
        initial_covariance = np.diag([
            100.0, 100.0, 100.0,      # Position uncertainty [m²]
            1.0, 1.0, 1.0,            # Velocity uncertainty [m²/s²]
            0.01, 0.01, 0.01, 0.01,   # Attitude uncertainty [rad²]
            0.001, 0.001, 0.001       # Angular velocity uncertainty [rad²/s²]
        ])
    
    return EKFState(
        position=relative_state.position.copy(),
        velocity=relative_state.velocity.copy(),
        quaternion=attitude_state.quaternion.copy(),
        angular_velocity=attitude_state.angular_velocity.copy(),
        covariance=initial_covariance,
        time=relative_state.time
    )

