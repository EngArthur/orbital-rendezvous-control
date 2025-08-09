# Orbital Rendezvous Control System

A complete Python implementation of autonomous orbital rendezvous guidance, navigation, and control based on the research paper "Relative Motion Guidance, Navigation and Control for Autonomous Orbital Rendezvous" by Okasha & Newman (2014).

## Project Overview

This project demonstrates advanced expertise in:
- **Orbital Mechanics**: Complete implementation of orbital dynamics and perturbations
- **Attitude Dynamics**: Quaternion-based attitude representation and control
- **Navigation Systems**: Extended Kalman Filter for relative state estimation
- **Sensor Fusion**: Multi-sensor integration for robust navigation
- **Control Systems**: Coupled translational and rotational control
- **Aerospace Engineering**: High-fidelity spacecraft simulation

## Features

### Phase 1 - Foundations (Complete)
- **Orbital Elements**: Complete orbital elements representation and conversions
- **Coordinate Systems**: ECI, LVLH (Hill frame), RSW transformations
- **Mathematical Utilities**: Quaternions, rotation matrices, Kepler equation solver
- **Unit Testing**: Comprehensive test suite for validation

### Phase 2 - Dynamics (Complete)
- **Orbital Perturbations**: J2 gravitational and atmospheric drag effects
- **Attitude Dynamics**: Quaternion-based attitude kinematics and dynamics
- **Relative Motion**: Clohessy-Wiltshire and nonlinear relative dynamics
- **Coupled Motion**: Translational-rotational coupling effects
- **Environmental Models**: Atmospheric density and gravity gradient torques

### Phase 3 - Navigation (Complete)
- **Extended Kalman Filter**: Nonlinear state estimation for relative navigation
- **Sensor Models**: LIDAR, star tracker, gyroscope, accelerometer, and GPS
- **Measurement Processing**: Nonlinear measurement models and Jacobian computation
- **Sensor Fusion**: Multi-sensor integration with outlier detection
- **Navigation System**: Complete integrated navigation with performance monitoring

### Phase 4 - Control (Planned)
- **LQR Controllers**: Optimal translational and rotational control
- **Guidance Laws**: Rendezvous trajectory planning and execution
- **Thruster Models**: Realistic actuator dynamics and constraints

### Phase 5 - Simulation (Planned)
- **Monte Carlo Analysis**: Statistical performance evaluation
- **Mission Scenarios**: Complete rendezvous and docking simulations
- **Visualization**: 3D trajectory and attitude visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/orbital-rendezvous-control.git
cd orbital-rendezvous-control

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

### Phase 1 Example - Orbital Elements

```python
import numpy as np
from src.dynamics.orbital_elements import OrbitalElements, orbital_elements_to_cartesian

# Create ISS-like orbit
elements = OrbitalElements(
    a=6.78e6,                    # Semi-major axis [m]
    e=0.0001,                    # Eccentricity
    i=np.radians(51.6),          # Inclination [rad]
    omega_cap=0.0,               # RAAN [rad]
    omega=0.0,                   # Argument of periapsis [rad]
    f=0.0                        # True anomaly [rad]
)

# Convert to Cartesian coordinates
position, velocity = orbital_elements_to_cartesian(elements)
print(f"Orbital Period: {elements.period/3600:.2f} hours")
```

### Phase 2 Example - Perturbations and Attitude

```python
from src.dynamics.perturbations import SpacecraftProperties, perturbation_analysis_summary
from src.dynamics.attitude_dynamics import SpacecraftInertia, AttitudeState

# Define spacecraft properties
spacecraft = SpacecraftProperties(
    mass=10.0,                   # 10 kg CubeSat
    drag_area=0.01,              # 10 cm² cross-section
    drag_coefficient=2.2
)

# Analyze perturbation effects
analysis = perturbation_analysis_summary(elements, spacecraft)
print(f"Dominant perturbation: {analysis['dominant_perturbation']}")
print(f"Estimated decay time: {analysis['estimated_decay_time_days']:.1f} days")

# Attitude dynamics
inertia = SpacecraftInertia(Ixx=0.1, Iyy=0.15, Izz=0.2)
attitude = AttitudeState(
    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    angular_velocity=np.array([0.01, 0.02, 0.03])
)
```

### Phase 3 Example - Extended Kalman Filter Navigation

```python
from src.navigation.navigation_system import create_default_navigation_system
from src.navigation.sensor_models import create_typical_sensor_suite
from src.dynamics.relative_motion import RelativeState
from src.dynamics.attitude_dynamics import AttitudeState

# Initial states
initial_relative_state = RelativeState(
    position=np.array([1000.0, -500.0, 200.0]),  # LVLH frame [m]
    velocity=np.array([0.1, 0.2, -0.05])         # LVLH frame [m/s]
)

initial_attitude_state = AttitudeState(
    quaternion=np.array([1.0, 0.0, 0.0, 0.0]),   # Identity quaternion
    angular_velocity=np.array([0.01, 0.02, 0.03]) # Body frame [rad/s]
)

# Create navigation system
nav_system = create_default_navigation_system(
    initial_relative_state, 
    initial_attitude_state
)

# Update with measurements
true_state = {
    'relative_state': initial_relative_state,
    'attitude_state': initial_attitude_state,
    'target_elements': target_elements
}

estimated_state = nav_system.update(true_state, target_elements, current_time)
print(f"Position uncertainty (3σ): {nav_system.ekf.get_position_uncertainty():.2f} m")
```

### Phase 3 Example - Sensor Models

```python
from src.navigation.sensor_models import create_typical_sensor_suite, SensorType

# Create sensor suite
sensor_suite = create_typical_sensor_suite()

# Generate measurements
measurements = sensor_suite.generate_measurements(true_state, current_time)

for measurement in measurements:
    print(f"{measurement.sensor_type.value}: {measurement.data}")
    
# Analyze sensor performance
from src.navigation.sensor_models import analyze_sensor_performance
performance = analyze_sensor_performance(measurements, true_values)
print(f"LIDAR RMS error: {performance['lidar']['rms_error']}")
```

## Testing

Run the test suite to validate implementations:

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_dynamics/test_orbital_elements.py
pytest tests/test_dynamics/test_perturbations.py
pytest tests/test_dynamics/test_attitude_dynamics.py
pytest tests/test_navigation/test_extended_kalman_filter.py
pytest tests/test_navigation/test_sensor_models.py

# Run with coverage
pytest --cov=src tests/
```

## Examples

Run the comprehensive examples to see the system in action:

```bash
# Phase 1 example - Basic orbital mechanics
python examples/basic_example_fixed.py

# Phase 2 example - Perturbations and attitude dynamics
python examples/phase2_example.py

# Phase 3 example - Extended Kalman Filter navigation
python examples/phase3_example.py
```

<<<<<<< HEAD
## Technical Background
=======
##  Technical Background
>>>>>>> 91c7bee120bd6ab47a0e8662aadc4c7961281198

This implementation is based on the theoretical framework presented in:

> Okasha, M., & Newman, B. (2014). Relative motion guidance, navigation and control for autonomous orbital rendezvous. Journal of Aerospace Technology and Management, 6(3), 301-318.

### Key Technical Components

#### Phase 1 & 2 - Dynamics
1. **Gauss' Variational Equations**: For orbital element propagation under perturbations
2. **J2 Perturbation Model**: Earth's oblateness effects on orbital motion
3. **Atmospheric Drag Model**: Exponential atmosphere with Earth rotation effects
4. **Quaternion Attitude Dynamics**: Singularity-free attitude representation
5. **Gravity Gradient Torques**: Environmental torques for attitude dynamics
6. **LVLH Coordinate Frame**: Local-Vertical-Local-Horizontal reference frame
7. **Clohessy-Wiltshire Equations**: Linear relative motion dynamics
8. **State Transition Matrix**: Analytical solution for relative motion propagation

#### Phase 3 - Navigation
9. **Extended Kalman Filter**: Nonlinear state estimation with 13-state vector
10. **Sensor Models**: Realistic models for LIDAR, star tracker, gyroscope, accelerometer, GPS
11. **Measurement Jacobians**: Analytical derivatives for nonlinear measurement models
12. **Innovation-based Outlier Detection**: Chi-squared test for measurement validation
13. **Multi-sensor Fusion**: Optimal combination of heterogeneous sensor data
14. **Covariance Analysis**: Uncertainty quantification and filter monitoring
15. **Process Noise Modeling**: Realistic noise models for spacecraft dynamics
16. **Measurement Noise Modeling**: Sensor-specific noise characteristics

## Project Structure

```
orbital-rendezvous-control/
├── src/                          # Source code
│   ├── dynamics/                 # Orbital and attitude dynamics
│   │   ├── orbital_elements.py   # Orbital elements and conversions
│   │   ├── perturbations.py      # J2 and drag perturbations
│   │   ├── attitude_dynamics.py  # Quaternion attitude dynamics
│   │   └── relative_motion.py    # Relative motion dynamics
│   ├── navigation/               # Navigation and filtering
│   │   ├── extended_kalman_filter.py  # EKF implementation
│   │   ├── sensor_models.py      # Sensor models and suite
│   │   └── navigation_system.py  # Integrated navigation system
│   ├── control/                  # Control systems (Phase 4)
│   ├── simulation/               # Simulation environment (Phase 5)
│   └── utils/                    # Utilities and constants
├── tests/                        # Unit tests
│   ├── test_dynamics/            # Dynamics module tests
│   └── test_navigation/          # Navigation module tests
├── examples/                     # Usage examples
├── docs/                         # Documentation
└── data/                         # Simulation data and results
```

## Development Roadmap

### Phase 1: Foundations 
- [x] Orbital elements and conversions
- [x] Mathematical utilities
- [x] Unit testing framework

### Phase 2: Dynamics 
- [x] Perturbation models (J2, atmospheric drag)
- [x] Attitude dynamics with quaternions
- [x] Relative motion equations
- [x] Coupled translational-rotational dynamics

### Phase 3: Navigation 
- [x] Extended Kalman Filter implementation
- [x] Sensor models (LIDAR, star tracker, gyros, accelerometer, GPS)
- [x] Measurement models and linearization
- [x] Multi-sensor fusion and outlier detection
- [x] Integrated navigation system
- [x] Performance analysis and monitoring

### Phase 4: Control (In Progress)
- [ ] LQR translational controller
- [ ] Quaternion-based attitude controller
- [ ] Coupled control strategies
- [ ] Thruster allocation and management

### Phase 5: Simulation
- [ ] Complete spacecraft model
- [ ] Orbital environment simulation
- [ ] Monte Carlo analysis tools
- [ ] 3D visualization

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 91c7bee120bd6ab47a0e8662aadc4c7961281198
## Performance Metrics

- **Code Coverage**: >90% (Phase 1-3)
- **Documentation**: Complete API documentation
- **Validation**: Comparison with published results
- **Performance**: Real-time capable simulation
- **Navigation Accuracy**: Sub-meter position estimation
- **Attitude Accuracy**: Sub-degree attitude estimation

## Navigation System Capabilities

### Extended Kalman Filter
- **State Vector**: 13-dimensional (position, velocity, quaternion, angular velocity)
- **Prediction**: Nonlinear dynamics with Clohessy-Wiltshire and attitude kinematics
- **Update**: Multiple sensor types with individual measurement models
- **Robustness**: Innovation-based outlier detection and adaptive tuning

### Sensor Suite
- **LIDAR**: Range and range-rate measurements (10 Hz, 10 cm accuracy)
- **Star Tracker**: Attitude quaternion measurements (1 Hz, 2 arcsec accuracy)
- **Gyroscope**: Angular velocity measurements (100 Hz, 0.01°/s accuracy)
- **Accelerometer**: Specific force measurements (100 Hz, 0.1 mg accuracy)
- **GPS**: Position measurements (1 Hz, 5 m accuracy)

### Performance Monitoring
- **Innovation Statistics**: Normalized Innovation Squared (NIS) monitoring
- **Covariance Analysis**: Trace monitoring and automatic reset
- **Measurement Validation**: Chi-squared test for outlier detection
- **Computation Time**: Real-time performance tracking

## Author

**Arthur Allex Feliphe Barbosa Moreno**
- Institution: IME - Instituto Militar de Engenharia
- Email: 
- LinkedIn: [arthurmoreno](https://www.linkedin.com/in/arthurmoreno/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

*This project demonstrates advanced aerospace engineering capabilities and serves as a comprehensive portfolio piece showcasing expertise in orbital mechanics, attitude dynamics, perturbation modeling, navigation, sensor fusion, and spacecraft control systems.*

