"""
Sensor Models for Spacecraft Navigation

This module implements realistic sensor models for spacecraft navigation,
including LIDAR, star trackers, gyroscopes, accelerometers, and GPS.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .extended_kalman_filter import SensorType, Measurement
from ..dynamics.relative_motion import RelativeState
from ..dynamics.attitude_dynamics import AttitudeState
from ..dynamics.orbital_elements import OrbitalElements
from ..utils.math_utils import quaternion_to_rotation_matrix


@dataclass
class SensorConfiguration:
    """Base configuration for sensors."""
    update_rate: float          # Update rate [Hz]
    noise_std: np.ndarray      # Measurement noise standard deviation
    bias: np.ndarray           # Sensor bias
    scale_factor: float = 1.0  # Scale factor error
    enabled: bool = True       # Sensor enabled flag


class SensorModel(ABC):
    """Abstract base class for sensor models."""
    
    def __init__(self, config: SensorConfiguration, sensor_id: str = "default"):
        """
        Initialize sensor model.
        
        Args:
            config: Sensor configuration
            sensor_id: Unique sensor identifier
        """
        self.config = config
        self.sensor_id = sensor_id
        self.last_measurement_time = 0.0
        self.measurement_count = 0
    
    @abstractmethod
    def generate_measurement(self, true_state: Dict, time: float) -> Optional[Measurement]:
        """
        Generate sensor measurement from true state.
        
        Args:
            true_state: Dictionary containing true spacecraft state
            time: Current time [s]
        
        Returns:
            Measurement object or None if no measurement available
        """
        pass
    
    def is_measurement_available(self, time: float) -> bool:
        """Check if measurement is available at given time."""
        if not self.config.enabled:
            return False
        
        dt = time - self.last_measurement_time
        return dt >= (1.0 / self.config.update_rate)
    
    def _add_noise(self, measurement: np.ndarray) -> np.ndarray:
        """Add noise to measurement."""
        noise = np.random.normal(0, self.config.noise_std)
        return measurement + noise
    
    def _apply_bias_and_scale(self, measurement: np.ndarray) -> np.ndarray:
        """Apply bias and scale factor errors."""
        return self.config.scale_factor * measurement + self.config.bias


class LidarSensorModel(SensorModel):
    """
    LIDAR sensor model for range and range-rate measurements.
    
    Measures distance and relative velocity to target spacecraft.
    """
    
    def __init__(self, config: SensorConfiguration, sensor_id: str = "lidar"):
        """Initialize LIDAR sensor model."""
        super().__init__(config, sensor_id)
        
        # Validate configuration
        if config.noise_std.shape != (2,):
            raise ValueError("LIDAR noise_std must be 2D: [range_std, range_rate_std]")
        if config.bias.shape != (2,):
            raise ValueError("LIDAR bias must be 2D: [range_bias, range_rate_bias]")
    
    def generate_measurement(self, true_state: Dict, time: float) -> Optional[Measurement]:
        """Generate LIDAR measurement."""
        if not self.is_measurement_available(time):
            return None
        
        relative_state = true_state['relative_state']
        
        # True range and range rate
        range_true = np.linalg.norm(relative_state.position)
        
        if range_true < 1e-6:
            range_rate_true = 0.0
        else:
            range_rate_true = np.dot(relative_state.position, relative_state.velocity) / range_true
        
        # Perfect measurement
        measurement_true = np.array([range_true, range_rate_true])
        
        # Add errors
        measurement_noisy = self._add_noise(measurement_true)
        measurement_final = self._apply_bias_and_scale(measurement_noisy)
        
        # Measurement covariance
        R = np.diag(self.config.noise_std**2)
        
        # Update timing
        self.last_measurement_time = time
        self.measurement_count += 1
        
        return Measurement(
            sensor_type=SensorType.LIDAR,
            data=measurement_final,
            covariance=R,
            time=time,
            sensor_id=self.sensor_id
        )


class StarTrackerSensorModel(SensorModel):
    """
    Star tracker sensor model for attitude measurements.
    
    Provides high-accuracy attitude measurements as quaternions.
    """
    
    def __init__(self, config: SensorConfiguration, sensor_id: str = "star_tracker"):
        """Initialize star tracker sensor model."""
        super().__init__(config, sensor_id)
        
        # Validate configuration
        if config.noise_std.shape != (3,):
            raise ValueError("Star tracker noise_std must be 3D: [roll_std, pitch_std, yaw_std]")
        if config.bias.shape != (3,):
            raise ValueError("Star tracker bias must be 3D: [roll_bias, pitch_bias, yaw_bias]")
    
    def generate_measurement(self, true_state: Dict, time: float) -> Optional[Measurement]:
        """Generate star tracker measurement."""
        if not self.is_measurement_available(time):
            return None
        
        attitude_state = true_state['attitude_state']
        
        # True quaternion
        q_true = attitude_state.quaternion.copy()
        
        # Convert to Euler angles for noise addition
        from ..dynamics.attitude_dynamics import euler_angles_from_quaternion, quaternion_from_euler_angles
        
        roll, pitch, yaw = euler_angles_from_quaternion(q_true)
        euler_true = np.array([roll, pitch, yaw])
        
        # Add noise and bias
        euler_noisy = self._add_noise(euler_true)
        euler_final = self._apply_bias_and_scale(euler_noisy)
        
        # Convert back to quaternion
        q_measured = quaternion_from_euler_angles(*euler_final)
        
        # Measurement covariance (in quaternion space - simplified)
        # In practice, this would be more complex
        R = np.diag([1e-6, 1e-6, 1e-6, 1e-6])  # Small quaternion covariance
        
        # Update timing
        self.last_measurement_time = time
        self.measurement_count += 1
        
        return Measurement(
            sensor_type=SensorType.STAR_TRACKER,
            data=q_measured,
            covariance=R,
            time=time,
            sensor_id=self.sensor_id
        )


class GyroscopeSensorModel(SensorModel):
    """
    Gyroscope sensor model for angular velocity measurements.
    
    Measures angular velocity in body frame with bias and noise.
    """
    
    def __init__(self, config: SensorConfiguration, sensor_id: str = "gyroscope"):
        """Initialize gyroscope sensor model."""
        super().__init__(config, sensor_id)
        
        # Validate configuration
        if config.noise_std.shape != (3,):
            raise ValueError("Gyroscope noise_std must be 3D")
        if config.bias.shape != (3,):
            raise ValueError("Gyroscope bias must be 3D")
        
        # Gyroscope-specific parameters
        self.bias_instability = 1e-6  # Bias instability [rad/s]
        self.random_walk = 1e-8       # Angular random walk [rad/s/√Hz]
    
    def generate_measurement(self, true_state: Dict, time: float) -> Optional[Measurement]:
        """Generate gyroscope measurement."""
        if not self.is_measurement_available(time):
            return None
        
        attitude_state = true_state['attitude_state']
        
        # True angular velocity
        omega_true = attitude_state.angular_velocity.copy()
        
        # Add noise and bias
        omega_noisy = self._add_noise(omega_true)
        omega_final = self._apply_bias_and_scale(omega_noisy)
        
        # Measurement covariance
        R = np.diag(self.config.noise_std**2)
        
        # Update timing
        self.last_measurement_time = time
        self.measurement_count += 1
        
        return Measurement(
            sensor_type=SensorType.GYROSCOPE,
            data=omega_final,
            covariance=R,
            time=time,
            sensor_id=self.sensor_id
        )


class AccelerometerSensorModel(SensorModel):
    """
    Accelerometer sensor model for specific force measurements.
    
    Measures specific force (acceleration minus gravity) in body frame.
    """
    
    def __init__(self, config: SensorConfiguration, sensor_id: str = "accelerometer"):
        """Initialize accelerometer sensor model."""
        super().__init__(config, sensor_id)
        
        # Validate configuration
        if config.noise_std.shape != (3,):
            raise ValueError("Accelerometer noise_std must be 3D")
        if config.bias.shape != (3,):
            raise ValueError("Accelerometer bias must be 3D")
    
    def generate_measurement(self, true_state: Dict, time: float) -> Optional[Measurement]:
        """Generate accelerometer measurement."""
        if not self.is_measurement_available(time):
            return None
        
        relative_state = true_state['relative_state']
        attitude_state = true_state['attitude_state']
        target_elements = true_state['target_elements']
        
        # Compute specific force in LVLH frame
        pos = relative_state.position
        n = target_elements.mean_motion
        
        # Gravitational acceleration in LVLH frame (Clohessy-Wiltshire)
        accel_gravity_lvlh = np.array([
            3 * n**2 * pos[0],
            0.0,
            -n**2 * pos[2]
        ])
        
        # Transform to body frame
        R_lvlh_to_body = quaternion_to_rotation_matrix(attitude_state.quaternion).T
        accel_gravity_body = R_lvlh_to_body @ accel_gravity_lvlh
        
        # Specific force (what accelerometer measures)
        specific_force_true = accel_gravity_body
        
        # Add noise and bias
        specific_force_noisy = self._add_noise(specific_force_true)
        specific_force_final = self._apply_bias_and_scale(specific_force_noisy)
        
        # Measurement covariance
        R = np.diag(self.config.noise_std**2)
        
        # Update timing
        self.last_measurement_time = time
        self.measurement_count += 1
        
        return Measurement(
            sensor_type=SensorType.ACCELEROMETER,
            data=specific_force_final,
            covariance=R,
            time=time,
            sensor_id=self.sensor_id
        )


class GPSSensorModel(SensorModel):
    """
    GPS sensor model for position measurements.
    
    Provides position measurements in LVLH frame (simplified model).
    """
    
    def __init__(self, config: SensorConfiguration, sensor_id: str = "gps"):
        """Initialize GPS sensor model."""
        super().__init__(config, sensor_id)
        
        # Validate configuration
        if config.noise_std.shape != (3,):
            raise ValueError("GPS noise_std must be 3D")
        if config.bias.shape != (3,):
            raise ValueError("GPS bias must be 3D")
    
    def generate_measurement(self, true_state: Dict, time: float) -> Optional[Measurement]:
        """Generate GPS measurement."""
        if not self.is_measurement_available(time):
            return None
        
        relative_state = true_state['relative_state']
        
        # True position
        position_true = relative_state.position.copy()
        
        # Add noise and bias
        position_noisy = self._add_noise(position_true)
        position_final = self._apply_bias_and_scale(position_noisy)
        
        # Measurement covariance
        R = np.diag(self.config.noise_std**2)
        
        # Update timing
        self.last_measurement_time = time
        self.measurement_count += 1
        
        return Measurement(
            sensor_type=SensorType.GPS,
            data=position_final,
            covariance=R,
            time=time,
            sensor_id=self.sensor_id
        )


class SensorSuite:
    """
    Collection of sensors for spacecraft navigation.
    
    Manages multiple sensors and provides unified interface for measurements.
    """
    
    def __init__(self):
        """Initialize empty sensor suite."""
        self.sensors: Dict[str, SensorModel] = {}
        self.measurement_history: List[Measurement] = []
    
    def add_sensor(self, sensor: SensorModel) -> None:
        """Add sensor to the suite."""
        self.sensors[sensor.sensor_id] = sensor
    
    def remove_sensor(self, sensor_id: str) -> None:
        """Remove sensor from the suite."""
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
    
    def generate_measurements(self, true_state: Dict, time: float) -> List[Measurement]:
        """Generate measurements from all available sensors."""
        measurements = []
        
        for sensor in self.sensors.values():
            measurement = sensor.generate_measurement(true_state, time)
            if measurement is not None:
                measurements.append(measurement)
                self.measurement_history.append(measurement)
        
        return measurements
    
    def get_sensor_status(self) -> Dict[str, Dict]:
        """Get status of all sensors."""
        status = {}
        
        for sensor_id, sensor in self.sensors.items():
            status[sensor_id] = {
                'enabled': sensor.config.enabled,
                'update_rate': sensor.config.update_rate,
                'measurement_count': sensor.measurement_count,
                'last_measurement_time': sensor.last_measurement_time
            }
        
        return status
    
    def enable_sensor(self, sensor_id: str) -> None:
        """Enable specific sensor."""
        if sensor_id in self.sensors:
            self.sensors[sensor_id].config.enabled = True
    
    def disable_sensor(self, sensor_id: str) -> None:
        """Disable specific sensor."""
        if sensor_id in self.sensors:
            self.sensors[sensor_id].config.enabled = False
    
    def get_measurements_by_type(self, sensor_type: SensorType) -> List[Measurement]:
        """Get all measurements of specific type."""
        return [m for m in self.measurement_history if m.sensor_type == sensor_type]
    
    def clear_measurement_history(self) -> None:
        """Clear measurement history."""
        self.measurement_history.clear()


def create_typical_sensor_suite() -> SensorSuite:
    """
    Create a typical sensor suite for spacecraft navigation.
    
    Returns:
        Configured sensor suite with realistic parameters
    """
    suite = SensorSuite()
    
    # LIDAR sensor
    lidar_config = SensorConfiguration(
        update_rate=10.0,  # 10 Hz
        noise_std=np.array([0.1, 0.01]),  # 10 cm range, 1 cm/s range-rate
        bias=np.array([0.05, 0.001])      # 5 cm range bias, 1 mm/s range-rate bias
    )
    suite.add_sensor(LidarSensorModel(lidar_config, "lidar_primary"))
    
    # Star tracker
    star_tracker_config = SensorConfiguration(
        update_rate=1.0,  # 1 Hz
        noise_std=np.array([1e-5, 1e-5, 1e-5]),  # 2 arcsec (1σ)
        bias=np.array([1e-6, 1e-6, 1e-6])        # Small bias
    )
    suite.add_sensor(StarTrackerSensorModel(star_tracker_config, "star_tracker_primary"))
    
    # Gyroscope
    gyro_config = SensorConfiguration(
        update_rate=100.0,  # 100 Hz
        noise_std=np.array([1e-4, 1e-4, 1e-4]),  # 0.01 deg/s (1σ)
        bias=np.array([1e-5, 1e-5, 1e-5])        # 0.001 deg/s bias
    )
    suite.add_sensor(GyroscopeSensorModel(gyro_config, "gyro_primary"))
    
    # Accelerometer
    accel_config = SensorConfiguration(
        update_rate=100.0,  # 100 Hz
        noise_std=np.array([1e-4, 1e-4, 1e-4]),  # 0.1 mg (1σ)
        bias=np.array([1e-5, 1e-5, 1e-5])        # 0.01 mg bias
    )
    suite.add_sensor(AccelerometerSensorModel(accel_config, "accel_primary"))
    
    # GPS (if available)
    gps_config = SensorConfiguration(
        update_rate=1.0,  # 1 Hz
        noise_std=np.array([5.0, 5.0, 5.0]),     # 5 m (1σ)
        bias=np.array([1.0, 1.0, 1.0])           # 1 m bias
    )
    suite.add_sensor(GPSSensorModel(gps_config, "gps_primary"))
    
    return suite


def create_high_accuracy_sensor_suite() -> SensorSuite:
    """
    Create a high-accuracy sensor suite for precision navigation.
    
    Returns:
        High-accuracy sensor suite
    """
    suite = SensorSuite()
    
    # High-accuracy LIDAR
    lidar_config = SensorConfiguration(
        update_rate=20.0,  # 20 Hz
        noise_std=np.array([0.01, 0.001]),  # 1 cm range, 1 mm/s range-rate
        bias=np.array([0.005, 0.0001])      # 5 mm range bias, 0.1 mm/s range-rate bias
    )
    suite.add_sensor(LidarSensorModel(lidar_config, "lidar_high_accuracy"))
    
    # High-accuracy star tracker
    star_tracker_config = SensorConfiguration(
        update_rate=2.0,  # 2 Hz
        noise_std=np.array([5e-6, 5e-6, 5e-6]),  # 1 arcsec (1σ)
        bias=np.array([1e-7, 1e-7, 1e-7])        # Very small bias
    )
    suite.add_sensor(StarTrackerSensorModel(star_tracker_config, "star_tracker_high_accuracy"))
    
    # High-accuracy gyroscope
    gyro_config = SensorConfiguration(
        update_rate=200.0,  # 200 Hz
        noise_std=np.array([5e-5, 5e-5, 5e-5]),  # 0.003 deg/s (1σ)
        bias=np.array([1e-6, 1e-6, 1e-6])        # 0.0001 deg/s bias
    )
    suite.add_sensor(GyroscopeSensorModel(gyro_config, "gyro_high_accuracy"))
    
    return suite


def analyze_sensor_performance(measurements: List[Measurement], 
                             true_values: List[np.ndarray]) -> Dict[str, float]:
    """
    Analyze sensor performance metrics.
    
    Args:
        measurements: List of sensor measurements
        true_values: List of true values corresponding to measurements
    
    Returns:
        Dictionary with performance metrics
    """
    if len(measurements) != len(true_values):
        raise ValueError("Number of measurements and true values must match")
    
    if not measurements:
        return {}
    
    # Group by sensor type
    sensor_types = set(m.sensor_type for m in measurements)
    results = {}
    
    for sensor_type in sensor_types:
        # Filter measurements for this sensor type
        type_measurements = [m for m in measurements if m.sensor_type == sensor_type]
        type_true_values = [true_values[i] for i, m in enumerate(measurements) 
                          if m.sensor_type == sensor_type]
        
        if not type_measurements:
            continue
        
        # Compute errors
        errors = []
        for meas, true_val in zip(type_measurements, type_true_values):
            error = meas.data - true_val
            errors.append(error)
        
        errors = np.array(errors)
        
        # Performance metrics
        results[sensor_type.value] = {
            'mean_error': np.mean(errors, axis=0).tolist(),
            'std_error': np.std(errors, axis=0).tolist(),
            'rms_error': np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            'max_error': np.max(np.abs(errors), axis=0).tolist(),
            'num_measurements': len(type_measurements)
        }
    
    return results

