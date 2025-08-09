"""
Unit tests for Extended Kalman Filter module.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import pytest
import numpy as np
from src.navigation.extended_kalman_filter import (
    EKFState, ProcessNoise, ExtendedKalmanFilter, SensorType, Measurement,
    create_initial_ekf_state
)
from src.dynamics.relative_motion import RelativeState
from src.dynamics.attitude_dynamics import AttitudeState
from src.dynamics.orbital_elements import OrbitalElements
from src.utils.constants import EARTH_RADIUS


class TestEKFState:
    """Test cases for EKF state representation."""
    
    def test_ekf_state_creation(self):
        """Test creation of EKF state."""
        position = np.array([100.0, 200.0, 300.0])
        velocity = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        angular_velocity = np.array([0.1, 0.2, 0.3])
        covariance = np.eye(13)
        
        state = EKFState(position, velocity, quaternion, angular_velocity, covariance)
        
        assert np.allclose(state.position, position)
        assert np.allclose(state.velocity, velocity)
        assert np.allclose(state.quaternion, quaternion)
        assert np.allclose(state.angular_velocity, angular_velocity)
        assert np.allclose(state.covariance, covariance)
    
    def test_ekf_state_validation(self):
        """Test EKF state validation."""
        # Wrong position dimension
        with pytest.raises(ValueError):
            EKFState(
                position=np.array([1.0, 2.0]),  # Should be 3D
                velocity=np.array([1.0, 2.0, 3.0]),
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.1, 0.2, 0.3]),
                covariance=np.eye(13)
            )
        
        # Wrong covariance dimension
        with pytest.raises(ValueError):
            EKFState(
                position=np.array([1.0, 2.0, 3.0]),
                velocity=np.array([1.0, 2.0, 3.0]),
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.1, 0.2, 0.3]),
                covariance=np.eye(10)  # Should be 13x13
            )
    
    def test_state_vector_property(self):
        """Test state vector property."""
        position = np.array([100.0, 200.0, 300.0])
        velocity = np.array([1.0, 2.0, 3.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        angular_velocity = np.array([0.1, 0.2, 0.3])
        covariance = np.eye(13)
        
        state = EKFState(position, velocity, quaternion, angular_velocity, covariance)
        state_vector = state.state_vector
        
        assert len(state_vector) == 13
        assert np.allclose(state_vector[0:3], position)
        assert np.allclose(state_vector[3:6], velocity)
        assert np.allclose(state_vector[6:10], quaternion)
        assert np.allclose(state_vector[10:13], angular_velocity)
    
    def test_from_state_vector(self):
        """Test creation from state vector."""
        x = np.array([100, 200, 300, 1, 2, 3, 1, 0, 0, 0, 0.1, 0.2, 0.3])
        P = np.eye(13)
        
        state = EKFState.from_state_vector(x, P)
        
        assert np.allclose(state.position, x[0:3])
        assert np.allclose(state.velocity, x[3:6])
        assert np.allclose(state.quaternion, x[6:10])
        assert np.allclose(state.angular_velocity, x[10:13])


class TestProcessNoise:
    """Test cases for process noise parameters."""
    
    def test_process_noise_defaults(self):
        """Test default process noise values."""
        noise = ProcessNoise()
        
        assert noise.position_noise > 0
        assert noise.velocity_noise > 0
        assert noise.attitude_noise > 0
        assert noise.angular_vel_noise > 0
    
    def test_process_noise_custom(self):
        """Test custom process noise values."""
        noise = ProcessNoise(
            position_noise=1e-5,
            velocity_noise=1e-7,
            attitude_noise=1e-9,
            angular_vel_noise=1e-11
        )
        
        assert noise.position_noise == 1e-5
        assert noise.velocity_noise == 1e-7
        assert noise.attitude_noise == 1e-9
        assert noise.angular_vel_noise == 1e-11


class TestExtendedKalmanFilter:
    """Test cases for Extended Kalman Filter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initial state
        position = np.array([1000.0, 0.0, 0.0])
        velocity = np.array([0.0, 1.0, 0.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        angular_velocity = np.array([0.0, 0.0, 0.1])
        covariance = np.eye(13) * 0.1
        
        self.initial_state = EKFState(position, velocity, quaternion, 
                                    angular_velocity, covariance)
        
        # Process noise
        self.process_noise = ProcessNoise()
        
        # Target elements
        self.target_elements = OrbitalElements(
            a=EARTH_RADIUS + 400e3,
            e=0.0001,
            i=np.radians(51.6),
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        # Create EKF
        self.ekf = ExtendedKalmanFilter(self.initial_state, self.process_noise)
    
    def test_ekf_initialization(self):
        """Test EKF initialization."""
        assert np.allclose(self.ekf.state.position, self.initial_state.position)
        assert np.allclose(self.ekf.state.velocity, self.initial_state.velocity)
        assert np.allclose(self.ekf.state.quaternion, self.initial_state.quaternion)
        assert np.allclose(self.ekf.state.angular_velocity, self.initial_state.angular_velocity)
    
    def test_prediction_step(self):
        """Test EKF prediction step."""
        initial_position = self.ekf.state.position.copy()
        initial_time = self.ekf.state.time
        
        # Predict forward
        dt = 10.0  # 10 seconds
        self.ekf.predict(self.target_elements, dt)
        
        # State should have changed
        assert not np.allclose(self.ekf.state.position, initial_position)
        assert self.ekf.state.time == initial_time + dt
        
        # Quaternion should remain normalized
        q_norm = np.linalg.norm(self.ekf.state.quaternion)
        assert abs(q_norm - 1.0) < 1e-10
    
    def test_measurement_model_lidar(self):
        """Test LIDAR measurement model."""
        x = self.ekf.state.state_vector
        
        h = self.ekf._measurement_model(x, SensorType.LIDAR, self.target_elements)
        
        # Should return range and range-rate
        assert len(h) == 2
        assert h[0] > 0  # Range should be positive
    
    def test_measurement_model_star_tracker(self):
        """Test star tracker measurement model."""
        x = self.ekf.state.state_vector
        
        h = self.ekf._measurement_model(x, SensorType.STAR_TRACKER, self.target_elements)
        
        # Should return quaternion
        assert len(h) == 4
        assert abs(np.linalg.norm(h) - 1.0) < 1e-10  # Should be normalized
    
    def test_measurement_model_gyroscope(self):
        """Test gyroscope measurement model."""
        x = self.ekf.state.state_vector
        
        h = self.ekf._measurement_model(x, SensorType.GYROSCOPE, self.target_elements)
        
        # Should return angular velocity
        assert len(h) == 3
        assert np.allclose(h, x[10:13])
    
    def test_measurement_jacobian_lidar(self):
        """Test LIDAR measurement Jacobian."""
        x = self.ekf.state.state_vector
        
        H = self.ekf._compute_measurement_jacobian(x, SensorType.LIDAR, self.target_elements)
        
        # Should be 2x13 matrix
        assert H.shape == (2, 13)
        
        # Should have non-zero elements for position and velocity
        assert not np.allclose(H[0, 0:3], 0)  # Range depends on position
        assert not np.allclose(H[1, 0:6], 0)  # Range-rate depends on pos and vel
    
    def test_measurement_jacobian_star_tracker(self):
        """Test star tracker measurement Jacobian."""
        x = self.ekf.state.state_vector
        
        H = self.ekf._compute_measurement_jacobian(x, SensorType.STAR_TRACKER, self.target_elements)
        
        # Should be 4x13 matrix
        assert H.shape == (4, 13)
        
        # Should be identity for quaternion elements
        assert np.allclose(H[0:4, 6:10], np.eye(4))
    
    def test_update_step_lidar(self):
        """Test EKF update step with LIDAR measurement."""
        # Create LIDAR measurement
        measurement_data = np.array([1000.0, 1.0])  # Range and range-rate
        measurement_cov = np.diag([1.0, 0.01])      # Measurement covariance
        
        measurement = Measurement(
            sensor_type=SensorType.LIDAR,
            data=measurement_data,
            covariance=measurement_cov,
            time=0.0
        )
        
        initial_covariance_trace = np.trace(self.ekf.state.covariance)
        
        # Update with measurement
        self.ekf.update(measurement, self.target_elements)
        
        # Covariance should decrease (information gain)
        final_covariance_trace = np.trace(self.ekf.state.covariance)
        assert final_covariance_trace < initial_covariance_trace
    
    def test_update_step_star_tracker(self):
        """Test EKF update step with star tracker measurement."""
        # Create star tracker measurement
        measurement_data = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        measurement_cov = np.eye(4) * 1e-6                 # High accuracy
        
        measurement = Measurement(
            sensor_type=SensorType.STAR_TRACKER,
            data=measurement_data,
            covariance=measurement_cov,
            time=0.0
        )
        
        initial_attitude = self.ekf.state.quaternion.copy()
        
        # Update with measurement
        self.ekf.update(measurement, self.target_elements)
        
        # Attitude should be closer to measurement
        final_attitude = self.ekf.state.quaternion
        assert np.linalg.norm(final_attitude - measurement_data) < \
               np.linalg.norm(initial_attitude - measurement_data)
    
    def test_process_noise_matrix(self):
        """Test process noise matrix computation."""
        dt = 1.0
        Q = self.ekf._compute_process_noise_matrix(dt)
        
        # Should be 13x13 matrix
        assert Q.shape == (13, 13)
        
        # Should be positive semi-definite
        eigenvals = np.linalg.eigvals(Q)
        assert np.all(eigenvals >= -1e-10)  # Allow small numerical errors
    
    def test_state_transition_matrix(self):
        """Test state transition matrix computation."""
        x = self.ekf.state.state_vector
        dt = 1.0
        
        F = self.ekf._compute_state_transition_matrix(x, self.target_elements, dt)
        
        # Should be 13x13 matrix
        assert F.shape == (13, 13)
        
        # Diagonal elements should be close to 1 for small dt
        diag_elements = np.diag(F)
        assert np.all(diag_elements > 0.5)  # Should be positive
    
    def test_innovation_statistics(self):
        """Test innovation statistics computation."""
        # Initially should be empty
        stats = self.ekf.get_innovation_statistics()
        assert len(stats) == 0
        
        # Add some measurements
        measurement = Measurement(
            sensor_type=SensorType.LIDAR,
            data=np.array([1000.0, 1.0]),
            covariance=np.diag([1.0, 0.01]),
            time=0.0
        )
        
        self.ekf.update(measurement, self.target_elements)
        
        # Should have statistics now
        stats = self.ekf.get_innovation_statistics()
        assert 'mean_nis' in stats
        assert 'innovation_norm' in stats
    
    def test_uncertainty_estimates(self):
        """Test uncertainty estimate methods."""
        pos_uncertainty = self.ekf.get_position_uncertainty()
        vel_uncertainty = self.ekf.get_velocity_uncertainty()
        att_uncertainty = self.ekf.get_attitude_uncertainty()
        
        assert pos_uncertainty > 0
        assert vel_uncertainty > 0
        assert att_uncertainty > 0
    
    def test_covariance_reset(self):
        """Test covariance reset functionality."""
        # Set large covariance
        large_cov = np.eye(13) * 1000
        self.ekf.reset_covariance(large_cov)
        
        assert np.allclose(self.ekf.state.covariance, large_cov)
        
        # Reset to smaller covariance
        small_cov = np.eye(13) * 0.1
        self.ekf.reset_covariance(small_cov)
        
        assert np.allclose(self.ekf.state.covariance, small_cov)


class TestCreateInitialEKFState:
    """Test cases for initial EKF state creation."""
    
    def test_create_initial_state_default_covariance(self):
        """Test creation with default covariance."""
        relative_state = RelativeState(
            position=np.array([100.0, 200.0, 300.0]),
            velocity=np.array([1.0, 2.0, 3.0])
        )
        
        attitude_state = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.1, 0.2, 0.3])
        )
        
        ekf_state = create_initial_ekf_state(relative_state, attitude_state)
        
        assert np.allclose(ekf_state.position, relative_state.position)
        assert np.allclose(ekf_state.velocity, relative_state.velocity)
        assert np.allclose(ekf_state.quaternion, attitude_state.quaternion)
        assert np.allclose(ekf_state.angular_velocity, attitude_state.angular_velocity)
        assert ekf_state.covariance.shape == (13, 13)
    
    def test_create_initial_state_custom_covariance(self):
        """Test creation with custom covariance."""
        relative_state = RelativeState(
            position=np.array([100.0, 200.0, 300.0]),
            velocity=np.array([1.0, 2.0, 3.0])
        )
        
        attitude_state = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.1, 0.2, 0.3])
        )
        
        custom_cov = np.eye(13) * 0.5
        ekf_state = create_initial_ekf_state(relative_state, attitude_state, custom_cov)
        
        assert np.allclose(ekf_state.covariance, custom_cov)


if __name__ == "__main__":
    pytest.main([__file__])

