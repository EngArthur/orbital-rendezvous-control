"""
Unit tests for sensor models module.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import pytest
import numpy as np
from src.navigation.sensor_models import (
    SensorConfiguration, LidarSensorModel, StarTrackerSensorModel,
    GyroscopeSensorModel, AccelerometerSensorModel, GPSSensorModel,
    SensorSuite, create_typical_sensor_suite, create_high_accuracy_sensor_suite,
    analyze_sensor_performance
)
from src.navigation.extended_kalman_filter import SensorType
from src.dynamics.relative_motion import RelativeState
from src.dynamics.attitude_dynamics import AttitudeState
from src.dynamics.orbital_elements import OrbitalElements
from src.utils.constants import EARTH_RADIUS


class TestSensorConfiguration:
    """Test cases for sensor configuration."""
    
    def test_sensor_config_creation(self):
        """Test sensor configuration creation."""
        config = SensorConfiguration(
            update_rate=10.0,
            noise_std=np.array([0.1, 0.01]),
            bias=np.array([0.05, 0.001])
        )
        
        assert config.update_rate == 10.0
        assert np.allclose(config.noise_std, [0.1, 0.01])
        assert np.allclose(config.bias, [0.05, 0.001])
        assert config.scale_factor == 1.0
        assert config.enabled == True


class TestLidarSensorModel:
    """Test cases for LIDAR sensor model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SensorConfiguration(
            update_rate=10.0,
            noise_std=np.array([0.1, 0.01]),
            bias=np.array([0.05, 0.001])
        )
        self.lidar = LidarSensorModel(self.config)
        
        # True state
        self.true_state = {
            'relative_state': RelativeState(
                position=np.array([1000.0, 0.0, 0.0]),
                velocity=np.array([0.0, 1.0, 0.0])
            ),
            'attitude_state': AttitudeState(
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.1])
            )
        }
    
    def test_lidar_config_validation(self):
        """Test LIDAR configuration validation."""
        # Wrong noise dimension
        with pytest.raises(ValueError):
            config = SensorConfiguration(
                update_rate=10.0,
                noise_std=np.array([0.1]),  # Should be 2D
                bias=np.array([0.05, 0.001])
            )
            LidarSensorModel(config)
        
        # Wrong bias dimension
        with pytest.raises(ValueError):
            config = SensorConfiguration(
                update_rate=10.0,
                noise_std=np.array([0.1, 0.01]),
                bias=np.array([0.05])  # Should be 2D
            )
            LidarSensorModel(config)
    
    def test_lidar_measurement_generation(self):
        """Test LIDAR measurement generation."""
        measurement = self.lidar.generate_measurement(self.true_state, 0.1)
        
        assert measurement is not None
        assert measurement.sensor_type == SensorType.LIDAR
        assert len(measurement.data) == 2  # Range and range-rate
        assert measurement.data[0] > 0     # Range should be positive
        assert measurement.covariance.shape == (2, 2)
    
    def test_lidar_measurement_timing(self):
        """Test LIDAR measurement timing."""
        # First measurement should be available
        measurement1 = self.lidar.generate_measurement(self.true_state, 0.0)
        assert measurement1 is not None
        
        # Second measurement too soon should be None
        measurement2 = self.lidar.generate_measurement(self.true_state, 0.05)
        assert measurement2 is None
        
        # Third measurement after sufficient time should be available
        measurement3 = self.lidar.generate_measurement(self.true_state, 0.15)
        assert measurement3 is not None
    
    def test_lidar_disabled_sensor(self):
        """Test disabled LIDAR sensor."""
        self.lidar.config.enabled = False
        measurement = self.lidar.generate_measurement(self.true_state, 0.1)
        assert measurement is None


class TestStarTrackerSensorModel:
    """Test cases for star tracker sensor model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SensorConfiguration(
            update_rate=1.0,
            noise_std=np.array([1e-5, 1e-5, 1e-5]),
            bias=np.array([1e-6, 1e-6, 1e-6])
        )
        self.star_tracker = StarTrackerSensorModel(self.config)
        
        # True state
        self.true_state = {
            'relative_state': RelativeState(
                position=np.array([1000.0, 0.0, 0.0]),
                velocity=np.array([0.0, 1.0, 0.0])
            ),
            'attitude_state': AttitudeState(
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.1])
            )
        }
    
    def test_star_tracker_measurement_generation(self):
        """Test star tracker measurement generation."""
        measurement = self.star_tracker.generate_measurement(self.true_state, 1.0)
        
        assert measurement is not None
        assert measurement.sensor_type == SensorType.STAR_TRACKER
        assert len(measurement.data) == 4  # Quaternion
        
        # Quaternion should be normalized
        q_norm = np.linalg.norm(measurement.data)
        assert abs(q_norm - 1.0) < 1e-6
    
    def test_star_tracker_config_validation(self):
        """Test star tracker configuration validation."""
        # Wrong noise dimension
        with pytest.raises(ValueError):
            config = SensorConfiguration(
                update_rate=1.0,
                noise_std=np.array([1e-5, 1e-5]),  # Should be 3D
                bias=np.array([1e-6, 1e-6, 1e-6])
            )
            StarTrackerSensorModel(config)


class TestGyroscopeSensorModel:
    """Test cases for gyroscope sensor model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SensorConfiguration(
            update_rate=100.0,
            noise_std=np.array([1e-4, 1e-4, 1e-4]),
            bias=np.array([1e-5, 1e-5, 1e-5])
        )
        self.gyroscope = GyroscopeSensorModel(self.config)
        
        # True state
        self.true_state = {
            'relative_state': RelativeState(
                position=np.array([1000.0, 0.0, 0.0]),
                velocity=np.array([0.0, 1.0, 0.0])
            ),
            'attitude_state': AttitudeState(
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.1, 0.2, 0.3])
            )
        }
    
    def test_gyroscope_measurement_generation(self):
        """Test gyroscope measurement generation."""
        measurement = self.gyroscope.generate_measurement(self.true_state, 0.01)
        
        assert measurement is not None
        assert measurement.sensor_type == SensorType.GYROSCOPE
        assert len(measurement.data) == 3  # Angular velocity
        assert measurement.covariance.shape == (3, 3)
    
    def test_gyroscope_high_rate(self):
        """Test gyroscope high update rate."""
        # Should get measurements at high rate
        times = np.arange(0, 0.1, 0.01)  # 100 Hz
        measurements = []
        
        for t in times:
            meas = self.gyroscope.generate_measurement(self.true_state, t)
            if meas is not None:
                measurements.append(meas)
        
        # Should get most measurements (allowing for timing discretization)
        assert len(measurements) >= 8


class TestAccelerometerSensorModel:
    """Test cases for accelerometer sensor model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SensorConfiguration(
            update_rate=100.0,
            noise_std=np.array([1e-4, 1e-4, 1e-4]),
            bias=np.array([1e-5, 1e-5, 1e-5])
        )
        self.accelerometer = AccelerometerSensorModel(self.config)
        
        # Target elements
        self.target_elements = OrbitalElements(
            a=EARTH_RADIUS + 400e3,
            e=0.0001,
            i=np.radians(51.6),
            omega_cap=0.0,
            omega=0.0,
            f=0.0
        )
        
        # True state
        self.true_state = {
            'relative_state': RelativeState(
                position=np.array([1000.0, 0.0, 0.0]),
                velocity=np.array([0.0, 1.0, 0.0])
            ),
            'attitude_state': AttitudeState(
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.1, 0.2, 0.3])
            ),
            'target_elements': self.target_elements
        }
    
    def test_accelerometer_measurement_generation(self):
        """Test accelerometer measurement generation."""
        measurement = self.accelerometer.generate_measurement(self.true_state, 0.01)
        
        assert measurement is not None
        assert measurement.sensor_type == SensorType.ACCELEROMETER
        assert len(measurement.data) == 3  # Specific force
        assert measurement.covariance.shape == (3, 3)


class TestGPSSensorModel:
    """Test cases for GPS sensor model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SensorConfiguration(
            update_rate=1.0,
            noise_std=np.array([5.0, 5.0, 5.0]),
            bias=np.array([1.0, 1.0, 1.0])
        )
        self.gps = GPSSensorModel(self.config)
        
        # True state
        self.true_state = {
            'relative_state': RelativeState(
                position=np.array([1000.0, 500.0, 200.0]),
                velocity=np.array([0.0, 1.0, 0.0])
            ),
            'attitude_state': AttitudeState(
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.1, 0.2, 0.3])
            )
        }
    
    def test_gps_measurement_generation(self):
        """Test GPS measurement generation."""
        measurement = self.gps.generate_measurement(self.true_state, 1.0)
        
        assert measurement is not None
        assert measurement.sensor_type == SensorType.GPS
        assert len(measurement.data) == 3  # Position
        assert measurement.covariance.shape == (3, 3)
        
        # Should be close to true position (within noise bounds)
        true_pos = self.true_state['relative_state'].position
        pos_error = np.linalg.norm(measurement.data - true_pos)
        assert pos_error < 50.0  # Should be within reasonable bounds


class TestSensorSuite:
    """Test cases for sensor suite."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.suite = SensorSuite()
        
        # Add LIDAR sensor
        lidar_config = SensorConfiguration(
            update_rate=10.0,
            noise_std=np.array([0.1, 0.01]),
            bias=np.array([0.05, 0.001])
        )
        self.suite.add_sensor(LidarSensorModel(lidar_config, "lidar_1"))
        
        # Add star tracker
        star_config = SensorConfiguration(
            update_rate=1.0,
            noise_std=np.array([1e-5, 1e-5, 1e-5]),
            bias=np.array([1e-6, 1e-6, 1e-6])
        )
        self.suite.add_sensor(StarTrackerSensorModel(star_config, "star_tracker_1"))
        
        # True state
        self.true_state = {
            'relative_state': RelativeState(
                position=np.array([1000.0, 0.0, 0.0]),
                velocity=np.array([0.0, 1.0, 0.0])
            ),
            'attitude_state': AttitudeState(
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.1])
            )
        }
    
    def test_sensor_suite_add_remove(self):
        """Test adding and removing sensors."""
        initial_count = len(self.suite.sensors)
        
        # Add GPS sensor
        gps_config = SensorConfiguration(
            update_rate=1.0,
            noise_std=np.array([5.0, 5.0, 5.0]),
            bias=np.array([1.0, 1.0, 1.0])
        )
        self.suite.add_sensor(GPSSensorModel(gps_config, "gps_1"))
        
        assert len(self.suite.sensors) == initial_count + 1
        assert "gps_1" in self.suite.sensors
        
        # Remove sensor
        self.suite.remove_sensor("gps_1")
        assert len(self.suite.sensors) == initial_count
        assert "gps_1" not in self.suite.sensors
    
    def test_sensor_suite_generate_measurements(self):
        """Test measurement generation from suite."""
        measurements = self.suite.generate_measurements(self.true_state, 1.0)
        
        # Should get measurements from both sensors
        sensor_types = [m.sensor_type for m in measurements]
        assert SensorType.LIDAR in sensor_types
        assert SensorType.STAR_TRACKER in sensor_types
    
    def test_sensor_suite_enable_disable(self):
        """Test enabling and disabling sensors."""
        # Disable LIDAR
        self.suite.disable_sensor("lidar_1")
        measurements = self.suite.generate_measurements(self.true_state, 1.0)
        
        sensor_types = [m.sensor_type for m in measurements]
        assert SensorType.LIDAR not in sensor_types
        assert SensorType.STAR_TRACKER in sensor_types
        
        # Re-enable LIDAR
        self.suite.enable_sensor("lidar_1")
        measurements = self.suite.generate_measurements(self.true_state, 2.0)
        
        sensor_types = [m.sensor_type for m in measurements]
        assert SensorType.LIDAR in sensor_types
    
    def test_sensor_suite_status(self):
        """Test sensor status reporting."""
        status = self.suite.get_sensor_status()
        
        assert "lidar_1" in status
        assert "star_tracker_1" in status
        
        assert "enabled" in status["lidar_1"]
        assert "update_rate" in status["lidar_1"]
        assert "measurement_count" in status["lidar_1"]
    
    def test_sensor_suite_measurements_by_type(self):
        """Test getting measurements by type."""
        # Generate some measurements
        self.suite.generate_measurements(self.true_state, 1.0)
        self.suite.generate_measurements(self.true_state, 2.0)
        
        lidar_measurements = self.suite.get_measurements_by_type(SensorType.LIDAR)
        star_measurements = self.suite.get_measurements_by_type(SensorType.STAR_TRACKER)
        
        assert len(lidar_measurements) > 0
        assert len(star_measurements) > 0
        
        # Check types
        for m in lidar_measurements:
            assert m.sensor_type == SensorType.LIDAR
        for m in star_measurements:
            assert m.sensor_type == SensorType.STAR_TRACKER


class TestSensorSuiteCreation:
    """Test cases for sensor suite creation functions."""
    
    def test_create_typical_sensor_suite(self):
        """Test creation of typical sensor suite."""
        suite = create_typical_sensor_suite()
        
        # Should have multiple sensors
        assert len(suite.sensors) >= 4
        
        # Check for expected sensor types
        sensor_ids = list(suite.sensors.keys())
        assert any("lidar" in sid for sid in sensor_ids)
        assert any("star_tracker" in sid for sid in sensor_ids)
        assert any("gyro" in sid for sid in sensor_ids)
        assert any("accel" in sid for sid in sensor_ids)
    
    def test_create_high_accuracy_sensor_suite(self):
        """Test creation of high accuracy sensor suite."""
        suite = create_high_accuracy_sensor_suite()
        
        # Should have sensors
        assert len(suite.sensors) >= 3
        
        # Check that it's different from typical suite
        typical_suite = create_typical_sensor_suite()
        assert len(suite.sensors) != len(typical_suite.sensors)


class TestSensorPerformanceAnalysis:
    """Test cases for sensor performance analysis."""
    
    def test_analyze_sensor_performance_empty(self):
        """Test performance analysis with empty data."""
        measurements = []
        true_values = []
        
        results = analyze_sensor_performance(measurements, true_values)
        assert len(results) == 0
    
    def test_analyze_sensor_performance_mismatched_length(self):
        """Test performance analysis with mismatched lengths."""
        from src.navigation.extended_kalman_filter import Measurement
        
        measurements = [
            Measurement(SensorType.LIDAR, np.array([1000.0, 1.0]), 
                       np.eye(2), 0.0)
        ]
        true_values = []
        
        with pytest.raises(ValueError):
            analyze_sensor_performance(measurements, true_values)
    
    def test_analyze_sensor_performance_valid(self):
        """Test performance analysis with valid data."""
        from src.navigation.extended_kalman_filter import Measurement
        
        # Create measurements and true values
        measurements = [
            Measurement(SensorType.LIDAR, np.array([1000.1, 1.01]), 
                       np.eye(2), 0.0),
            Measurement(SensorType.LIDAR, np.array([999.9, 0.99]), 
                       np.eye(2), 1.0),
            Measurement(SensorType.GPS, np.array([100.1, 200.2, 300.3]), 
                       np.eye(3), 0.0)
        ]
        
        true_values = [
            np.array([1000.0, 1.0]),
            np.array([1000.0, 1.0]),
            np.array([100.0, 200.0, 300.0])
        ]
        
        results = analyze_sensor_performance(measurements, true_values)
        
        # Should have results for both sensor types
        assert 'lidar' in results
        assert 'gps' in results
        
        # Check structure of results
        lidar_results = results['lidar']
        assert 'mean_error' in lidar_results
        assert 'std_error' in lidar_results
        assert 'rms_error' in lidar_results
        assert 'max_error' in lidar_results
        assert 'num_measurements' in lidar_results
        
        assert lidar_results['num_measurements'] == 2


if __name__ == "__main__":
    pytest.main([__file__])

