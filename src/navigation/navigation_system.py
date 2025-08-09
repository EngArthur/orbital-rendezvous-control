"""
Integrated Navigation System

This module implements a complete navigation system that integrates the
Extended Kalman Filter with sensor models for autonomous spacecraft navigation.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from .extended_kalman_filter import ExtendedKalmanFilter, EKFState, ProcessNoise, create_initial_ekf_state
from .sensor_models import SensorSuite, Measurement, SensorType
from ..dynamics.relative_motion import RelativeState
from ..dynamics.attitude_dynamics import AttitudeState
from ..dynamics.orbital_elements import OrbitalElements


@dataclass
class NavigationConfiguration:
    """Configuration for navigation system."""
    ekf_process_noise: ProcessNoise
    max_measurement_age: float = 1.0        # Maximum age for measurements [s]
    innovation_threshold: float = 5.0       # Innovation threshold for outlier detection
    covariance_reset_threshold: float = 100.0  # Threshold for covariance reset
    enable_outlier_detection: bool = True   # Enable measurement outlier detection
    enable_adaptive_tuning: bool = False    # Enable adaptive filter tuning


@dataclass
class NavigationPerformance:
    """Navigation system performance metrics."""
    position_error_rms: float
    velocity_error_rms: float
    attitude_error_rms: float
    filter_consistency: float
    measurement_rejection_rate: float
    computation_time_avg: float


class NavigationSystem:
    """
    Integrated navigation system for spacecraft rendezvous.
    
    Combines Extended Kalman Filter with multiple sensor types for
    robust and accurate relative navigation.
    """
    
    def __init__(self, config: NavigationConfiguration, 
                 sensor_suite: SensorSuite,
                 initial_state: EKFState):
        """
        Initialize navigation system.
        
        Args:
            config: Navigation system configuration
            sensor_suite: Sensor suite for measurements
            initial_state: Initial EKF state estimate
        """
        self.config = config
        self.sensor_suite = sensor_suite
        self.ekf = ExtendedKalmanFilter(initial_state, config.ekf_process_noise)
        
        # Performance tracking
        self.measurement_count = 0
        self.rejected_measurement_count = 0
        self.computation_times: List[float] = []
        
        # State history for analysis
        self.state_history: List[EKFState] = [initial_state]
        self.true_state_history: List[Dict] = []
        self.measurement_history: List[Measurement] = []
        
        # Filter monitoring
        self.innovation_norms: List[float] = []
        self.covariance_traces: List[float] = []
    
    def update(self, true_state: Dict, target_elements: OrbitalElements, 
               current_time: float, control_input: Optional[Dict] = None) -> EKFState:
        """
        Update navigation system with new measurements.
        
        Args:
            true_state: True spacecraft state (for simulation)
            target_elements: Target orbital elements
            current_time: Current simulation time [s]
            control_input: Control inputs (acceleration and torque)
        
        Returns:
            Updated EKF state estimate
        """
        start_time = time.time()
        
        # Extract control inputs
        if control_input is None:
            control_acceleration = np.zeros(3)
            control_torque = np.zeros(3)
        else:
            control_acceleration = control_input.get('acceleration', np.zeros(3))
            control_torque = control_input.get('torque', np.zeros(3))
        
        # Time step
        dt = current_time - self.ekf.state.time
        
        # Prediction step
        if dt > 0:
            self.ekf.predict(target_elements, dt, control_acceleration, control_torque)
        
        # Generate measurements
        measurements = self.sensor_suite.generate_measurements(true_state, current_time)
        
        # Process measurements
        for measurement in measurements:
            if self._is_measurement_valid(measurement, current_time):
                if self._detect_outlier(measurement, target_elements):
                    self.rejected_measurement_count += 1
                    continue
                
                # Update with measurement
                self.ekf.update(measurement, target_elements)
                self.measurement_count += 1
                self.measurement_history.append(measurement)
        
        # Monitor filter performance
        self._monitor_filter_performance()
        
        # Adaptive tuning (if enabled)
        if self.config.enable_adaptive_tuning:
            self._adaptive_tuning()
        
        # Store state history
        self.state_history.append(self.ekf.state)
        self.true_state_history.append(true_state.copy())
        
        # Record computation time
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        return self.ekf.state
    
    def _is_measurement_valid(self, measurement: Measurement, current_time: float) -> bool:
        """Check if measurement is valid for processing."""
        # Check measurement age
        age = current_time - measurement.time
        if age > self.config.max_measurement_age:
            return False
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(measurement.data)):
            return False
        
        # Check covariance matrix
        if not np.all(np.isfinite(measurement.covariance)):
            return False
        
        # Check positive definiteness of covariance
        try:
            np.linalg.cholesky(measurement.covariance)
        except np.linalg.LinAlgError:
            return False
        
        return True
    
    def _detect_outlier(self, measurement: Measurement, 
                       target_elements: OrbitalElements) -> bool:
        """Detect measurement outliers using innovation test."""
        if not self.config.enable_outlier_detection:
            return False
        
        # Predict measurement
        x = self.ekf.state.state_vector
        h_pred = self.ekf._measurement_model(x, measurement.sensor_type, target_elements)
        
        # Innovation
        innovation = measurement.data - h_pred
        
        # Innovation covariance
        H = self.ekf._compute_measurement_jacobian(x, measurement.sensor_type, target_elements)
        P = self.ekf.state.covariance
        S = H @ P @ H.T + measurement.covariance
        
        # Normalized innovation squared (NIS)
        try:
            nis = innovation.T @ np.linalg.inv(S) @ innovation
            
            # Chi-squared test
            dof = len(measurement.data)  # Degrees of freedom
            threshold = self.config.innovation_threshold * dof
            
            return nis > threshold
        
        except np.linalg.LinAlgError:
            # If matrix inversion fails, accept measurement
            return False
    
    def _monitor_filter_performance(self) -> None:
        """Monitor filter performance and consistency."""
        # Innovation statistics
        innovation_stats = self.ekf.get_innovation_statistics()
        if 'innovation_norm' in innovation_stats:
            self.innovation_norms.append(innovation_stats['innovation_norm'])
        
        # Covariance trace
        P_trace = np.trace(self.ekf.state.covariance)
        self.covariance_traces.append(P_trace)
        
        # Check for covariance explosion
        if P_trace > self.config.covariance_reset_threshold:
            self._reset_covariance()
    
    def _reset_covariance(self) -> None:
        """Reset filter covariance when it becomes too large."""
        # Create new covariance matrix with reasonable values
        new_covariance = np.diag([
            100.0, 100.0, 100.0,      # Position uncertainty [m²]
            1.0, 1.0, 1.0,            # Velocity uncertainty [m²/s²]
            0.01, 0.01, 0.01, 0.01,   # Attitude uncertainty [rad²]
            0.001, 0.001, 0.001       # Angular velocity uncertainty [rad²/s²]
        ])
        
        self.ekf.reset_covariance(new_covariance)
    
    def _adaptive_tuning(self) -> None:
        """Adaptive tuning of filter parameters based on performance."""
        # Simple adaptive tuning based on innovation statistics
        if len(self.innovation_norms) < 10:
            return
        
        recent_innovations = self.innovation_norms[-10:]
        avg_innovation = np.mean(recent_innovations)
        
        # If innovations are consistently large, increase process noise
        if avg_innovation > 2.0:
            self.ekf.process_noise.position_noise *= 1.1
            self.ekf.process_noise.velocity_noise *= 1.1
            self.ekf.process_noise.attitude_noise *= 1.1
            self.ekf.process_noise.angular_vel_noise *= 1.1
        
        # If innovations are consistently small, decrease process noise
        elif avg_innovation < 0.5:
            self.ekf.process_noise.position_noise *= 0.9
            self.ekf.process_noise.velocity_noise *= 0.9
            self.ekf.process_noise.attitude_noise *= 0.9
            self.ekf.process_noise.angular_vel_noise *= 0.9
    
    def get_current_estimate(self) -> Tuple[RelativeState, AttitudeState]:
        """
        Get current navigation estimate as relative and attitude states.
        
        Returns:
            Tuple of (relative_state, attitude_state)
        """
        state = self.ekf.state
        
        relative_state = RelativeState(
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            time=state.time
        )
        
        attitude_state = AttitudeState(
            quaternion=state.quaternion.copy(),
            angular_velocity=state.angular_velocity.copy(),
            time=state.time
        )
        
        return relative_state, attitude_state
    
    def get_uncertainty_estimates(self) -> Dict[str, float]:
        """
        Get current uncertainty estimates.
        
        Returns:
            Dictionary with uncertainty metrics
        """
        return {
            'position_uncertainty_3sigma': self.ekf.get_position_uncertainty(),
            'velocity_uncertainty_3sigma': self.ekf.get_velocity_uncertainty(),
            'attitude_uncertainty_3sigma': self.ekf.get_attitude_uncertainty(),
            'covariance_trace': np.trace(self.ekf.state.covariance)
        }
    
    def analyze_performance(self) -> NavigationPerformance:
        """
        Analyze navigation system performance.
        
        Returns:
            Performance metrics
        """
        if len(self.state_history) < 2 or len(self.true_state_history) < 2:
            return NavigationPerformance(0, 0, 0, 0, 0, 0)
        
        # Compute errors
        position_errors = []
        velocity_errors = []
        attitude_errors = []
        
        for i, (est_state, true_state) in enumerate(zip(self.state_history[1:], 
                                                       self.true_state_history[1:])):
            # Position error
            pos_error = np.linalg.norm(est_state.position - true_state['relative_state'].position)
            position_errors.append(pos_error)
            
            # Velocity error
            vel_error = np.linalg.norm(est_state.velocity - true_state['relative_state'].velocity)
            velocity_errors.append(vel_error)
            
            # Attitude error (simplified)
            from ..dynamics.attitude_dynamics import attitude_error_quaternion
            q_error = attitude_error_quaternion(est_state.quaternion, 
                                              true_state['attitude_state'].quaternion)
            att_error = 2 * np.arccos(np.abs(q_error[0]))  # Angle error
            attitude_errors.append(att_error)
        
        # RMS errors
        position_error_rms = np.sqrt(np.mean(np.array(position_errors)**2))
        velocity_error_rms = np.sqrt(np.mean(np.array(velocity_errors)**2))
        attitude_error_rms = np.sqrt(np.mean(np.array(attitude_errors)**2))
        
        # Filter consistency (simplified)
        filter_consistency = np.mean(self.innovation_norms) if self.innovation_norms else 0.0
        
        # Measurement rejection rate
        total_measurements = self.measurement_count + self.rejected_measurement_count
        rejection_rate = (self.rejected_measurement_count / total_measurements 
                         if total_measurements > 0 else 0.0)
        
        # Average computation time
        avg_computation_time = np.mean(self.computation_times) if self.computation_times else 0.0
        
        return NavigationPerformance(
            position_error_rms=position_error_rms,
            velocity_error_rms=velocity_error_rms,
            attitude_error_rms=attitude_error_rms,
            filter_consistency=filter_consistency,
            measurement_rejection_rate=rejection_rate,
            computation_time_avg=avg_computation_time
        )
    
    def get_sensor_status(self) -> Dict[str, Dict]:
        """Get status of all sensors in the suite."""
        return self.sensor_suite.get_sensor_status()
    
    def enable_sensor(self, sensor_id: str) -> None:
        """Enable specific sensor."""
        self.sensor_suite.enable_sensor(sensor_id)
    
    def disable_sensor(self, sensor_id: str) -> None:
        """Disable specific sensor."""
        self.sensor_suite.disable_sensor(sensor_id)
    
    def reset_navigation(self, new_initial_state: EKFState) -> None:
        """Reset navigation system with new initial state."""
        self.ekf = ExtendedKalmanFilter(new_initial_state, self.config.ekf_process_noise)
        
        # Clear history
        self.state_history = [new_initial_state]
        self.true_state_history = []
        self.measurement_history = []
        self.innovation_norms = []
        self.covariance_traces = []
        
        # Reset counters
        self.measurement_count = 0
        self.rejected_measurement_count = 0
        self.computation_times = []
    
    def save_navigation_data(self, filename: str) -> None:
        """Save navigation data for post-processing analysis."""
        import pickle
        
        data = {
            'state_history': self.state_history,
            'true_state_history': self.true_state_history,
            'measurement_history': self.measurement_history,
            'innovation_norms': self.innovation_norms,
            'covariance_traces': self.covariance_traces,
            'performance': self.analyze_performance()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_navigation_data(self, filename: str) -> Dict:
        """Load navigation data from file."""
        import pickle
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        return data


def create_default_navigation_system(initial_relative_state: RelativeState,
                                    initial_attitude_state: AttitudeState,
                                    sensor_suite: Optional[SensorSuite] = None) -> NavigationSystem:
    """
    Create a navigation system with default configuration.
    
    Args:
        initial_relative_state: Initial relative state estimate
        initial_attitude_state: Initial attitude state estimate
        sensor_suite: Sensor suite (creates default if None)
    
    Returns:
        Configured navigation system
    """
    # Default process noise
    process_noise = ProcessNoise(
        position_noise=1e-6,
        velocity_noise=1e-8,
        attitude_noise=1e-10,
        angular_vel_noise=1e-12
    )
    
    # Default configuration
    config = NavigationConfiguration(
        ekf_process_noise=process_noise,
        max_measurement_age=1.0,
        innovation_threshold=5.0,
        covariance_reset_threshold=100.0,
        enable_outlier_detection=True,
        enable_adaptive_tuning=False
    )
    
    # Create sensor suite if not provided
    if sensor_suite is None:
        from .sensor_models import create_typical_sensor_suite
        sensor_suite = create_typical_sensor_suite()
    
    # Create initial EKF state
    initial_ekf_state = create_initial_ekf_state(initial_relative_state, initial_attitude_state)
    
    return NavigationSystem(config, sensor_suite, initial_ekf_state)


def run_navigation_simulation(navigation_system: NavigationSystem,
                            true_state_sequence: List[Dict],
                            target_elements: OrbitalElements,
                            time_sequence: List[float],
                            control_sequence: Optional[List[Dict]] = None) -> NavigationPerformance:
    """
    Run complete navigation simulation.
    
    Args:
        navigation_system: Navigation system to simulate
        true_state_sequence: Sequence of true states
        target_elements: Target orbital elements
        time_sequence: Time sequence [s]
        control_sequence: Control input sequence
    
    Returns:
        Navigation performance metrics
    """
    if control_sequence is None:
        control_sequence = [None] * len(true_state_sequence)
    
    if len(true_state_sequence) != len(time_sequence):
        raise ValueError("State and time sequences must have same length")
    
    if len(control_sequence) != len(true_state_sequence):
        raise ValueError("Control sequence must match state sequence length")
    
    # Run simulation
    for true_state, current_time, control_input in zip(true_state_sequence, 
                                                      time_sequence, 
                                                      control_sequence):
        navigation_system.update(true_state, target_elements, current_time, control_input)
    
    return navigation_system.analyze_performance()

