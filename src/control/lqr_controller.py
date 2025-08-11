"""
Linear Quadratic Regulator (LQR) Controllers

This module implements LQR controllers for both translational and rotational
spacecraft control during orbital rendezvous operations.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import scipy.linalg

from ..dynamics.relative_motion import RelativeState
from ..dynamics.attitude_dynamics import AttitudeState
from ..dynamics.orbital_elements import OrbitalElements


@dataclass
class LQRWeights:
    """LQR cost function weights."""
    Q_position: float = 1.0      # Position error weight
    Q_velocity: float = 1.0      # Velocity error weight
    Q_attitude: float = 1.0      # Attitude error weight
    Q_angular_vel: float = 1.0   # Angular velocity error weight
    R_force: float = 1.0         # Control force weight
    R_torque: float = 1.0        # Control torque weight


@dataclass
class ControlLimits:
    """Control system limits."""
    max_force: float = 10.0      # Maximum force per axis [N]
    max_torque: float = 1.0      # Maximum torque per axis [N⋅m]
    max_force_rate: float = 1.0  # Maximum force rate [N/s]
    max_torque_rate: float = 0.1 # Maximum torque rate [N⋅m/s]


class TranslationalLQRController:
    """
    LQR controller for translational motion in LVLH frame.
    
    Uses Clohessy-Wiltshire dynamics for relative motion control.
    State vector: [x, y, z, vx, vy, vz]
    Control vector: [fx, fy, fz]
    """
    
    def __init__(self, weights: LQRWeights, limits: ControlLimits, 
                 spacecraft_mass: float = 10.0):
        """
        Initialize translational LQR controller.
        
        Args:
            weights: LQR cost function weights
            limits: Control system limits
            spacecraft_mass: Spacecraft mass [kg]
        """
        self.weights = weights
        self.limits = limits
        self.mass = spacecraft_mass
        
        # LQR gains (computed when target orbit is set)
        self.K_trans = None
        self.target_orbit = None
        
        # Control history for rate limiting
        self.previous_force = np.zeros(3)
        self.previous_time = 0.0
    
    def set_target_orbit(self, target_elements: OrbitalElements) -> None:
        """
        Set target orbit and compute LQR gains.
        
        Args:
            target_elements: Target orbital elements
        """
        self.target_orbit = target_elements
        n = target_elements.mean_motion  # Mean motion [rad/s]
        
        # Clohessy-Wiltshire state matrix (6x6)
        A = np.array([
            [0,  0,  0,  1,  0,  0],
            [0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0,  0,  1],
            [3*n**2, 0,  0,  0,  2*n, 0],
            [0,  0,  0, -2*n, 0,  0],
            [0,  0, -n**2, 0,  0,  0]
        ])
        
        # Control input matrix (6x3)
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) / self.mass  # Force to acceleration
        
        # Cost matrices
        Q = np.diag([
            self.weights.Q_position, self.weights.Q_position, self.weights.Q_position,
            self.weights.Q_velocity, self.weights.Q_velocity, self.weights.Q_velocity
        ])
        
        R = np.diag([
            self.weights.R_force, self.weights.R_force, self.weights.R_force
        ])
        
        # Solve Riccati equation
        try:
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            self.K_trans = np.linalg.inv(R) @ B.T @ P
        except Exception as e:
            raise RuntimeError(f"Failed to solve LQR problem: {e}")
    
    def compute_control(self, current_state: RelativeState, 
                       desired_state: RelativeState,
                       current_time: float) -> np.ndarray:
        """
        Compute LQR control force.
        
        Args:
            current_state: Current relative state
            desired_state: Desired relative state
            current_time: Current time [s]
        
        Returns:
            Control force vector [N]
        """
        if self.K_trans is None:
            raise RuntimeError("Target orbit not set. Call set_target_orbit() first.")
        
        # State error
        state_error = np.array([
            current_state.position[0] - desired_state.position[0],
            current_state.position[1] - desired_state.position[1],
            current_state.position[2] - desired_state.position[2],
            current_state.velocity[0] - desired_state.velocity[0],
            current_state.velocity[1] - desired_state.velocity[1],
            current_state.velocity[2] - desired_state.velocity[2]
        ])
        
        # LQR control law: u = -K * x_error
        force_command = -self.K_trans @ state_error
        
        # Apply control limits
        force_command = self._apply_control_limits(force_command, current_time)
        
        return force_command
    
    def _apply_control_limits(self, force_command: np.ndarray, 
                            current_time: float) -> np.ndarray:
        """Apply control limits and rate limits."""
        # Magnitude limits
        for i in range(3):
            force_command[i] = np.clip(force_command[i], 
                                     -self.limits.max_force, 
                                      self.limits.max_force)
        
        # Rate limits
        if current_time > self.previous_time:
            dt = current_time - self.previous_time
            max_change = self.limits.max_force_rate * dt
            
            for i in range(3):
                change = force_command[i] - self.previous_force[i]
                if abs(change) > max_change:
                    force_command[i] = self.previous_force[i] + np.sign(change) * max_change
        
        # Update history
        self.previous_force = force_command.copy()
        self.previous_time = current_time
        
        return force_command
    
    def get_control_gains(self) -> Optional[np.ndarray]:
        """Get LQR control gains matrix."""
        return self.K_trans.copy() if self.K_trans is not None else None
    
    def analyze_stability(self) -> Dict[str, float]:
        """
        Analyze closed-loop stability.
        
        Returns:
            Dictionary with stability metrics
        """
        if self.K_trans is None or self.target_orbit is None:
            return {}
        
        n = self.target_orbit.mean_motion
        
        # Closed-loop system matrix
        A = np.array([
            [0,  0,  0,  1,  0,  0],
            [0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0,  0,  1],
            [3*n**2, 0,  0,  0,  2*n, 0],
            [0,  0,  0, -2*n, 0,  0],
            [0,  0, -n**2, 0,  0,  0]
        ])
        
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) / self.mass
        
        A_cl = A - B @ self.K_trans
        
        # Eigenvalue analysis
        eigenvals = np.linalg.eigvals(A_cl)
        
        return {
            'max_real_part': np.max(np.real(eigenvals)),
            'stability_margin': -np.max(np.real(eigenvals)),
            'damping_ratio': np.min(np.abs(np.real(eigenvals) / np.abs(eigenvals))),
            'natural_frequency': np.mean(np.abs(eigenvals))
        }


class AttitudeLQRController:
    """
    LQR controller for attitude control using quaternion representation.
    
    Uses linearized quaternion dynamics for attitude control.
    State vector: [q1, q2, q3, wx, wy, wz] (quaternion vector part + angular velocity)
    Control vector: [Mx, My, Mz] (torque)
    """
    
    def __init__(self, weights: LQRWeights, limits: ControlLimits,
                 spacecraft_inertia: np.ndarray):
        """
        Initialize attitude LQR controller.
        
        Args:
            weights: LQR cost function weights
            limits: Control system limits
            spacecraft_inertia: Spacecraft inertia matrix [kg⋅m²]
        """
        self.weights = weights
        self.limits = limits
        self.inertia = spacecraft_inertia
        self.inertia_inv = np.linalg.inv(spacecraft_inertia)
        
        # LQR gains
        self.K_att = None
        
        # Control history for rate limiting
        self.previous_torque = np.zeros(3)
        self.previous_time = 0.0
        
        # Compute LQR gains
        self._compute_lqr_gains()
    
    def _compute_lqr_gains(self) -> None:
        """Compute LQR gains for attitude control."""
        # Linearized attitude dynamics (6x6)
        # State: [q1, q2, q3, wx, wy, wz]
        A = np.zeros((6, 6))
        
        # Quaternion kinematics (linearized around identity)
        A[0:3, 3:6] = 0.5 * np.eye(3)
        
        # Angular dynamics: J*w_dot = -w x (J*w) + M
        # Linearized around zero angular velocity: w_dot = J^-1 * M
        A[3:6, 3:6] = np.zeros((3, 3))  # No gyroscopic terms at zero angular velocity
        
        # Control input matrix (6x3)
        B = np.zeros((6, 3))
        B[3:6, 0:3] = self.inertia_inv  # Torque to angular acceleration
        
        # Cost matrices
        Q = np.diag([
            self.weights.Q_attitude, self.weights.Q_attitude, self.weights.Q_attitude,
            self.weights.Q_angular_vel, self.weights.Q_angular_vel, self.weights.Q_angular_vel
        ])
        
        R = np.diag([
            self.weights.R_torque, self.weights.R_torque, self.weights.R_torque
        ])
        
        # Solve Riccati equation
        try:
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            self.K_att = np.linalg.inv(R) @ B.T @ P
        except Exception as e:
            raise RuntimeError(f"Failed to solve attitude LQR problem: {e}")
    
    def compute_control(self, current_attitude: AttitudeState,
                       desired_attitude: AttitudeState,
                       current_time: float) -> np.ndarray:
        """
        Compute LQR attitude control torque.
        
        Args:
            current_attitude: Current attitude state
            desired_attitude: Desired attitude state
            current_time: Current time [s]
        
        Returns:
            Control torque vector [N⋅m]
        """
        # Quaternion error (using vector part for small angles)
        q_current = current_attitude.quaternion
        q_desired = desired_attitude.quaternion
        
        # Ensure quaternions have same sign (shortest rotation)
        if np.dot(q_current, q_desired) < 0:
            q_desired = -q_desired
        
        # Quaternion error (vector part)
        q_error = self._quaternion_error(q_current, q_desired)
        
        # Angular velocity error
        w_error = current_attitude.angular_velocity - desired_attitude.angular_velocity
        
        # State error vector
        state_error = np.concatenate([q_error[1:4], w_error])  # [q1, q2, q3, wx, wy, wz]
        
        # LQR control law: u = -K * x_error
        torque_command = -self.K_att @ state_error
        
        # Apply control limits
        torque_command = self._apply_control_limits(torque_command, current_time)
        
        return torque_command
    
    def _quaternion_error(self, q_current: np.ndarray, q_desired: np.ndarray) -> np.ndarray:
        """
        Compute quaternion error.
        
        Args:
            q_current: Current quaternion [w, x, y, z]
            q_desired: Desired quaternion [w, x, y, z]
        
        Returns:
            Quaternion error [w, x, y, z]
        """
        # Quaternion multiplication: q_error = q_desired^-1 * q_current
        q_desired_conj = np.array([q_desired[0], -q_desired[1], -q_desired[2], -q_desired[3]])
        
        # Quaternion multiplication
        q_error = np.array([
            q_desired_conj[0]*q_current[0] - q_desired_conj[1]*q_current[1] - q_desired_conj[2]*q_current[2] - q_desired_conj[3]*q_current[3],
            q_desired_conj[0]*q_current[1] + q_desired_conj[1]*q_current[0] + q_desired_conj[2]*q_current[3] - q_desired_conj[3]*q_current[2],
            q_desired_conj[0]*q_current[2] - q_desired_conj[1]*q_current[3] + q_desired_conj[2]*q_current[0] + q_desired_conj[3]*q_current[1],
            q_desired_conj[0]*q_current[3] + q_desired_conj[1]*q_current[2] - q_desired_conj[2]*q_current[1] + q_desired_conj[3]*q_current[0]
        ])
        
        return q_error
    
    def _apply_control_limits(self, torque_command: np.ndarray,
                            current_time: float) -> np.ndarray:
        """Apply control limits and rate limits."""
        # Magnitude limits
        for i in range(3):
            torque_command[i] = np.clip(torque_command[i],
                                      -self.limits.max_torque,
                                       self.limits.max_torque)
        
        # Rate limits
        if current_time > self.previous_time:
            dt = current_time - self.previous_time
            max_change = self.limits.max_torque_rate * dt
            
            for i in range(3):
                change = torque_command[i] - self.previous_torque[i]
                if abs(change) > max_change:
                    torque_command[i] = self.previous_torque[i] + np.sign(change) * max_change
        
        # Update history
        self.previous_torque = torque_command.copy()
        self.previous_time = current_time
        
        return torque_command
    
    def get_control_gains(self) -> np.ndarray:
        """Get LQR control gains matrix."""
        return self.K_att.copy()
    
    def analyze_stability(self) -> Dict[str, float]:
        """
        Analyze closed-loop stability.
        
        Returns:
            Dictionary with stability metrics
        """
        # Closed-loop system matrix
        A = np.zeros((6, 6))
        A[0:3, 3:6] = 0.5 * np.eye(3)
        A[3:6, 3:6] = np.zeros((3, 3))
        
        B = np.zeros((6, 3))
        B[3:6, 0:3] = self.inertia_inv
        
        A_cl = A - B @ self.K_att
        
        # Eigenvalue analysis
        eigenvals = np.linalg.eigvals(A_cl)
        
        return {
            'max_real_part': np.max(np.real(eigenvals)),
            'stability_margin': -np.max(np.real(eigenvals)),
            'damping_ratio': np.min(np.abs(np.real(eigenvals) / np.abs(eigenvals))),
            'natural_frequency': np.mean(np.abs(eigenvals))
        }


class CoupledLQRController:
    """
    Coupled LQR controller for simultaneous translational and rotational control.
    
    Combines translational and attitude control with coupling terms.
    """
    
    def __init__(self, weights: LQRWeights, limits: ControlLimits,
                 spacecraft_mass: float, spacecraft_inertia: np.ndarray,
                 coupling_strength: float = 0.1):
        """
        Initialize coupled LQR controller.
        
        Args:
            weights: LQR cost function weights
            limits: Control system limits
            spacecraft_mass: Spacecraft mass [kg]
            spacecraft_inertia: Spacecraft inertia matrix [kg⋅m²]
            coupling_strength: Coupling strength between translation and rotation
        """
        self.weights = weights
        self.limits = limits
        self.mass = spacecraft_mass
        self.inertia = spacecraft_inertia
        self.inertia_inv = np.linalg.inv(spacecraft_inertia)
        self.coupling_strength = coupling_strength
        
        # Individual controllers
        self.trans_controller = TranslationalLQRController(weights, limits, spacecraft_mass)
        self.att_controller = AttitudeLQRController(weights, limits, spacecraft_inertia)
        
        # Coupled gains (computed when target orbit is set)
        self.K_coupled = None
        self.target_orbit = None
    
    def set_target_orbit(self, target_elements: OrbitalElements) -> None:
        """
        Set target orbit and compute coupled LQR gains.
        
        Args:
            target_elements: Target orbital elements
        """
        self.target_orbit = target_elements
        self.trans_controller.set_target_orbit(target_elements)
        
        n = target_elements.mean_motion
        
        # Combined state matrix (12x12)
        # State: [x, y, z, vx, vy, vz, q1, q2, q3, wx, wy, wz]
        A = np.zeros((12, 12))
        
        # Translational dynamics (Clohessy-Wiltshire)
        A[0:6, 0:6] = np.array([
            [0,  0,  0,  1,  0,  0],
            [0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0,  0,  1],
            [3*n**2, 0,  0,  0,  2*n, 0],
            [0,  0,  0, -2*n, 0,  0],
            [0,  0, -n**2, 0,  0,  0]
        ])
        
        # Attitude dynamics
        A[6:9, 9:12] = 0.5 * np.eye(3)  # Quaternion kinematics
        
        # Coupling terms (simplified)
        A[3:6, 6:9] = self.coupling_strength * np.eye(3)  # Attitude affects translation
        A[9:12, 0:3] = self.coupling_strength * np.eye(3)  # Position affects attitude
        
        # Control input matrix (12x6)
        B = np.zeros((12, 6))
        B[3:6, 0:3] = np.eye(3) / self.mass      # Force to acceleration
        B[9:12, 3:6] = self.inertia_inv          # Torque to angular acceleration
        
        # Cost matrices
        Q = np.diag([
            self.weights.Q_position, self.weights.Q_position, self.weights.Q_position,
            self.weights.Q_velocity, self.weights.Q_velocity, self.weights.Q_velocity,
            self.weights.Q_attitude, self.weights.Q_attitude, self.weights.Q_attitude,
            self.weights.Q_angular_vel, self.weights.Q_angular_vel, self.weights.Q_angular_vel
        ])
        
        R = np.diag([
            self.weights.R_force, self.weights.R_force, self.weights.R_force,
            self.weights.R_torque, self.weights.R_torque, self.weights.R_torque
        ])
        
        # Solve Riccati equation
        try:
            P = scipy.linalg.solve_continuous_are(A, B, Q, R)
            self.K_coupled = np.linalg.inv(R) @ B.T @ P
        except Exception as e:
            raise RuntimeError(f"Failed to solve coupled LQR problem: {e}")
    
    def compute_control(self, current_relative: RelativeState,
                       desired_relative: RelativeState,
                       current_attitude: AttitudeState,
                       desired_attitude: AttitudeState,
                       current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coupled LQR control.
        
        Args:
            current_relative: Current relative state
            desired_relative: Desired relative state
            current_attitude: Current attitude state
            desired_attitude: Desired attitude state
            current_time: Current time [s]
        
        Returns:
            Tuple of (force_command, torque_command)
        """
        if self.K_coupled is None:
            # Fall back to individual controllers
            force = self.trans_controller.compute_control(
                current_relative, desired_relative, current_time)
            torque = self.att_controller.compute_control(
                current_attitude, desired_attitude, current_time)
            return force, torque
        
        # Combined state error
        q_error = self.att_controller._quaternion_error(
            current_attitude.quaternion, desired_attitude.quaternion)
        
        state_error = np.concatenate([
            current_relative.position - desired_relative.position,
            current_relative.velocity - desired_relative.velocity,
            q_error[1:4],  # Quaternion vector part
            current_attitude.angular_velocity - desired_attitude.angular_velocity
        ])
        
        # Coupled LQR control law
        control_command = -self.K_coupled @ state_error
        
        # Split into force and torque
        force_command = control_command[0:3]
        torque_command = control_command[3:6]
        
        # Apply individual limits
        force_command = self.trans_controller._apply_control_limits(force_command, current_time)
        torque_command = self.att_controller._apply_control_limits(torque_command, current_time)
        
        return force_command, torque_command
    
    def get_control_gains(self) -> Optional[np.ndarray]:
        """Get coupled LQR control gains matrix."""
        return self.K_coupled.copy() if self.K_coupled is not None else None
    
    def analyze_stability(self) -> Dict[str, float]:
        """
        Analyze coupled system stability.
        
        Returns:
            Dictionary with stability metrics
        """
        if self.K_coupled is None or self.target_orbit is None:
            return {}
        
        n = self.target_orbit.mean_motion
        
        # Combined system matrix
        A = np.zeros((12, 12))
        A[0:6, 0:6] = np.array([
            [0,  0,  0,  1,  0,  0],
            [0,  0,  0,  0,  1,  0],
            [0,  0,  0,  0,  0,  1],
            [3*n**2, 0,  0,  0,  2*n, 0],
            [0,  0,  0, -2*n, 0,  0],
            [0,  0, -n**2, 0,  0,  0]
        ])
        A[6:9, 9:12] = 0.5 * np.eye(3)
        A[3:6, 6:9] = self.coupling_strength * np.eye(3)
        A[9:12, 0:3] = self.coupling_strength * np.eye(3)
        
        B = np.zeros((12, 6))
        B[3:6, 0:3] = np.eye(3) / self.mass
        B[9:12, 3:6] = self.inertia_inv
        
        A_cl = A - B @ self.K_coupled
        
        # Eigenvalue analysis
        eigenvals = np.linalg.eigvals(A_cl)
        
        return {
            'max_real_part': np.max(np.real(eigenvals)),
            'stability_margin': -np.max(np.real(eigenvals)),
            'damping_ratio': np.min(np.abs(np.real(eigenvals) / np.abs(eigenvals))),
            'natural_frequency': np.mean(np.abs(eigenvals)),
            'coupling_effect': self.coupling_strength
        }


def create_default_lqr_weights() -> LQRWeights:
    """Create default LQR weights for typical rendezvous mission."""
    return LQRWeights(
        Q_position=1.0,
        Q_velocity=1.0,
        Q_attitude=10.0,      # Higher weight on attitude precision
        Q_angular_vel=1.0,
        R_force=0.1,          # Lower weight allows more aggressive control
        R_torque=0.1
    )


def create_conservative_lqr_weights() -> LQRWeights:
    """Create conservative LQR weights for fuel-efficient operation."""
    return LQRWeights(
        Q_position=1.0,
        Q_velocity=1.0,
        Q_attitude=1.0,
        Q_angular_vel=1.0,
        R_force=10.0,         # Higher weight penalizes control effort
        R_torque=10.0
    )


def create_aggressive_lqr_weights() -> LQRWeights:
    """Create aggressive LQR weights for fast response."""
    return LQRWeights(
        Q_position=10.0,      # High precision requirements
        Q_velocity=10.0,
        Q_attitude=100.0,
        Q_angular_vel=10.0,
        R_force=0.01,         # Very low control penalty
        R_torque=0.01
    )


def create_default_control_limits() -> ControlLimits:
    """Create default control limits for typical spacecraft."""
    return ControlLimits(
        max_force=10.0,       # 10 N per axis
        max_torque=1.0,       # 1 N⋅m per axis
        max_force_rate=1.0,   # 1 N/s rate limit
        max_torque_rate=0.1   # 0.1 N⋅m/s rate limit
    )


def analyze_lqr_performance(controller, simulation_results: Dict) -> Dict[str, float]:
    """
    Analyze LQR controller performance from simulation results.
    
    Args:
        controller: LQR controller instance
        simulation_results: Dictionary with simulation data
    
    Returns:
        Performance metrics dictionary
    """
    if 'position_errors' not in simulation_results:
        return {}
    
    position_errors = np.array(simulation_results['position_errors'])
    velocity_errors = np.array(simulation_results.get('velocity_errors', []))
    control_forces = np.array(simulation_results.get('control_forces', []))
    control_torques = np.array(simulation_results.get('control_torques', []))
    
    metrics = {
        'position_rms_error': np.sqrt(np.mean(np.sum(position_errors**2, axis=1))),
        'position_max_error': np.max(np.linalg.norm(position_errors, axis=1)),
        'final_position_error': np.linalg.norm(position_errors[-1]) if len(position_errors) > 0 else 0,
    }
    
    if len(velocity_errors) > 0:
        metrics.update({
            'velocity_rms_error': np.sqrt(np.mean(np.sum(velocity_errors**2, axis=1))),
            'velocity_max_error': np.max(np.linalg.norm(velocity_errors, axis=1)),
            'final_velocity_error': np.linalg.norm(velocity_errors[-1])
        })
    
    if len(control_forces) > 0:
        metrics.update({
            'force_rms': np.sqrt(np.mean(np.sum(control_forces**2, axis=1))),
            'force_max': np.max(np.linalg.norm(control_forces, axis=1)),
            'total_delta_v': np.sum(np.linalg.norm(control_forces, axis=1)) * simulation_results.get('dt', 1.0) / simulation_results.get('mass', 10.0)
        })
    
    if len(control_torques) > 0:
        metrics.update({
            'torque_rms': np.sqrt(np.mean(np.sum(control_torques**2, axis=1))),
            'torque_max': np.max(np.linalg.norm(control_torques, axis=1))
        })
    
    # Stability metrics
    stability = controller.analyze_stability()
    metrics.update({f'stability_{k}': v for k, v in stability.items()})
    
    return metrics

