# Orbital Rendezvous Control System

A comprehensive Python implementation of autonomous orbital rendezvous guidance, navigation, and control systems based on the research paper "Relative Motion Guidance, Navigation and Control for Autonomous Orbital Rendezvous" by Okasha & Newman (2014).

**Author:** Arthur Allex Feliphe Barbosa Moreno  
**Institution:** IME - Instituto Militar de Engenharia - 2025

## Project Overview

This project implements a complete orbital rendezvous system with high-fidelity models for spacecraft dynamics, navigation, and control. The implementation follows aerospace industry standards and includes realistic sensor models, actuator dynamics, and environmental perturbations.

### Current Implementation Status

 **Phase 1: Orbital Mechanics Foundation** (Complete)
- Orbital elements representation and conversions
- Cartesian to orbital elements transformations
- Keplerian orbit propagation
- Mathematical utilities (quaternions, rotations, Kepler solver)

 **Phase 2: Dynamics and Perturbations** (Complete)
- J2 gravitational perturbation modeling
- Atmospheric drag with exponential density model
- Quaternion-based attitude dynamics
- Gravity gradient torque calculations
- Clohessy-Wiltshire and nonlinear relative motion

 **Phase 3: Navigation Systems** (Complete)
- Extended Kalman Filter for 13-state estimation
- Multi-sensor fusion (LIDAR, star tracker, gyroscope, accelerometer, GPS)
- Innovation-based outlier detection
- Real-time performance monitoring

 **Phase 4: Control Systems** (Complete)
- LQR controllers for translational and rotational control
- Coupled translational-rotational control
- Adaptive guidance laws with trajectory generation
- Realistic actuator models (thrusters and reaction wheels)
- Control allocation and actuator management

 **Phase 5: Complete Simulation** (In Development)
- Monte Carlo simulation framework
- 3D visualization and animation
- Mission planning and analysis tools
- Performance optimization

##  Features

### Orbital Mechanics
- **Orbital Elements**: Complete representation with validation
- **Coordinate Systems**: ECI, LVLH, RSW transformations
- **Orbit Propagation**: Keplerian and perturbed motion
- **Perturbations**: J2, atmospheric drag, solar radiation pressure

### Attitude Dynamics
- **Quaternion Representation**: Singularity-free attitude representation
- **Euler Equations**: Full nonlinear rotational dynamics
- **Environmental Torques**: Gravity gradient, magnetic, aerodynamic
- **Attitude Propagation**: RK4 integration with quaternion normalization

### Relative Motion
- **Clohessy-Wiltshire**: Linear relative motion model
- **Nonlinear Dynamics**: High-fidelity relative motion
- **State Transition Matrix**: Analytical solution for linear case
- **Coupled Motion**: Translational-rotational coupling effects

### Navigation Systems
- **Extended Kalman Filter**: 13-state estimation (position, velocity, attitude, angular velocity)
- **Sensor Models**: Realistic noise and bias models
  - LIDAR: Range and range-rate measurements
  - Star Tracker: Quaternion attitude measurements
  - Gyroscope: Angular velocity measurements
  - Accelerometer: Specific force measurements
  - GPS: Position measurements
- **Multi-Rate Fusion**: Different sensor update rates
- **Outlier Detection**: Chi-squared test for measurement validation

### Control Systems
- **LQR Controllers**: Optimal linear quadratic regulators
  - Translational LQR for position/velocity control
  - Attitude LQR for quaternion/angular velocity control
  - Coupled LQR for integrated 6-DOF control
- **Guidance Laws**: Trajectory generation and following
  - Linear guidance with waypoint following
  - Nonlinear guidance with optimal control
  - Adaptive guidance with performance monitoring
- **Actuator Models**: Realistic thruster and reaction wheel dynamics
  - Thruster models with propellant consumption
  - Reaction wheel models with momentum saturation
  - Control allocation for redundant actuators

##  Installation

### Requirements
- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- pytest (for testing)

### Setup
```bash
git clone https://github.com/EngArthur/orbital-rendezvous-control.git
cd orbital-rendezvous-control
pip install -r requirements.txt
```

##  Quick Start

### Basic Orbital Elements Example
```python
from src.dynamics.orbital_elements import OrbitalElements
from src.utils.constants import EARTH_MU
import numpy as np

# Create ISS-like orbit
orbit = OrbitalElements(
    a=6.78e6,           # Semi-major axis [m]
    e=0.0003,           # Eccentricity
    i=np.radians(51.6), # Inclination [rad]
    omega_cap=0.0,      # RAAN [rad]
    omega=0.0,          # Argument of periapsis [rad]
    f=0.0,              # True anomaly [rad]
    mu=EARTH_MU
)

print(f"Orbital period: {orbit.period/3600:.2f} hours")
print(f"Mean motion: {orbit.mean_motion*180/np.pi:.4f} deg/s")
```

### Navigation System Example
```python
from src.navigation.extended_kalman_filter import ExtendedKalmanFilter
from src.navigation.sensor_models import LIDARModel, StarTrackerModel

# Create EKF for relative navigation
ekf = ExtendedKalmanFilter()

# Add sensors
lidar = LIDARModel(range_noise=0.1, rate_noise=0.01)
star_tracker = StarTrackerModel(attitude_noise=np.radians(2/3600))  # 2 arcsec

# Process measurements
ekf.predict(dt=1.0)
ekf.update_lidar(lidar_measurement, measurement_time)
ekf.update_star_tracker(attitude_measurement, measurement_time)
```

### Control System Example
```python
from src.control.lqr_controller import CoupledLQRController
from src.control.guidance_laws import AdaptiveGuidanceLaw
from src.control.actuator_models import create_hybrid_actuator_suite

# Create coupled 6-DOF controller
controller = CoupledLQRController(weights, limits, mass, inertia)
controller.set_target_orbit(target_orbit)

# Create adaptive guidance law
guidance = AdaptiveGuidanceLaw(target_orbit, constraints)

# Create actuator suite
actuators = create_hybrid_actuator_suite()

# Compute control
force_cmd, torque_cmd = controller.compute_control(
    current_state, desired_state, current_attitude, desired_attitude, time
)

# Apply through actuators
actual_force, actual_torque = actuators.allocate_control(
    force_cmd, torque_cmd, time
)
```

##  Examples

### Phase 1: Orbital Mechanics
```bash
python examples/basic_example_fixed.py
```
Demonstrates orbital elements, coordinate transformations, and orbit propagation.

### Phase 2: Dynamics and Perturbations
```bash
python examples/phase2_example.py
```
Shows perturbation effects, attitude dynamics, and relative motion modeling.

### Phase 3: Navigation Systems
```bash
python examples/phase3_example.py
```
Illustrates Extended Kalman Filter operation with multi-sensor fusion.

### Phase 4: Control Systems
```bash
python examples/phase4_example.py
```
Complete rendezvous simulation with guidance, navigation, and control.

##  Testing

Run the complete test suite:
```bash
python -m pytest tests/ -v
```

Run specific test modules:
```bash
python -m pytest tests/test_dynamics/ -v
python -m pytest tests/test_navigation/ -v
python -m pytest tests/test_control/ -v
```

##  Performance Metrics

The system includes comprehensive performance analysis:

### Navigation Performance
- Position estimation accuracy: < 1m RMS
- Velocity estimation accuracy: < 0.01 m/s RMS
- Attitude estimation accuracy: < 0.1° RMS
- Filter convergence time: < 300s

### Control Performance
- Position tracking accuracy: < 5m RMS
- Velocity tracking accuracy: < 0.05 m/s RMS
- Fuel consumption: < 10 m/s total ΔV
- Control bandwidth: > 0.1 Hz

### Guidance Performance
- Trajectory optimization: Minimum fuel consumption
- Constraint satisfaction: 100% compliance
- Adaptive performance: < 10% tracking error

##  Technical Details

### Mathematical Models

#### Orbital Dynamics
The system uses Gauss variational equations for perturbed orbital motion:

```
da/dt = (2a²/h) * [e*sin(f)*F_r + (p/r)*F_t]
de/dt = (1/h) * [p*sin(f)*F_r + ((p+r)*cos(f) + r*e)*F_t]
```

Where F_r and F_t are radial and tangential perturbation forces.

#### Attitude Dynamics
Quaternion-based attitude propagation using Euler's equations:

```
dq/dt = (1/2) * Ω(ω) * q
dω/dt = I⁻¹ * [τ - ω × (I*ω)]
```

#### Relative Motion
Clohessy-Wiltshire equations for linear relative motion:

```
ẍ - 2nẏ - 3n²x = F_x/m
ÿ + 2nẋ = F_y/m
z̈ + n²z = F_z/m
```

### Control Algorithms

#### LQR Control
Linear Quadratic Regulator with cost function:

```
J = ∫[x^T*Q*x + u^T*R*u]dt
```

Optimal gain: K = R⁻¹*B^T*P

#### Extended Kalman Filter
Nonlinear state estimation with linearized measurement model:

```
x̂(k+1|k) = f(x̂(k|k), u(k))
P(k+1|k) = F*P(k|k)*F^T + Q
```

##  References

1. Okasha, M., & Newman, B. (2014). "Relative Motion Guidance, Navigation and Control for Autonomous Orbital Rendezvous." Journal of Aerospace Technology and Management, 6(3), 301-318.

2. Clohessy, W. H., & Wiltshire, R. S. (1960). "Terminal guidance system for satellite rendezvous." Journal of the Aerospace Sciences, 27(9), 653-658.

3. Vallado, D. A. (2013). "Fundamentals of Astrodynamics and Applications" (4th ed.). Microcosm Press.

4. Wie, B. (2008). "Space Vehicle Dynamics and Control" (2nd ed.). AIAA Education Series.

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Based on the research by Okasha & Newman (2014)
- Inspired by real spacecraft rendezvous missions
- Developed for educational and research purposes

##  Contact

**Arthur Allex Feliphe Barbosa Moreno**  
Instituto Militar de Engenharia (IME)  
Email:   
LinkedIn: [https://www.linkedin.com/in/arthurmoreno/]

---

*This project demonstrates advanced aerospace engineering concepts and serves as a comprehensive reference for orbital rendezvous system design and implementation.*

