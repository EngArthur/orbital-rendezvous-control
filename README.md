# Orbital Rendezvous Control System

A complete Python implementation of autonomous orbital rendezvous guidance, navigation, and control based on the research paper "Relative Motion Guidance, Navigation and Control for Autonomous Orbital Rendezvous" by Okasha & Newman (2014).

## Project Overview

This project demonstrates advanced expertise in:
- **Orbital Mechanics**: Complete implementation of orbital dynamics and perturbations
- **Attitude Dynamics**: Quaternion-based attitude representation and control
- **Navigation Systems**: Extended Kalman Filter for relative state estimation
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

### Phase 3 - Navigation (Planned)
- **Extended Kalman Filter**: Relative navigation with sensor fusion
- **Sensor Models**: LIDAR, star tracker, gyroscope, and accelerometer models
- **Measurement Processing**: Nonlinear measurement models and linearization

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

### Phase 2 Example - Relative Motion

```python
from src.dynamics.relative_motion import RelativeState, propagate_relative_state

# Initial relative state (1 km behind target)
relative_state = RelativeState(
    position=np.array([0.0, -1000.0, 0.0]),  # LVLH frame [m]
    velocity=np.array([0.0, 0.0, 0.0])       # LVLH frame [m/s]
)

# Propagate using Clohessy-Wiltshire dynamics
new_state = propagate_relative_state(relative_state, target_elements, 600.0)
print(f"New range: {new_state.range:.1f} m")
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

# Run with coverage
pytest --cov=src tests/
```

<<<<<<< HEAD
##  Examples

Run the comprehensive examples to see the system in action:

```bash
# Phase 1 example - Basic orbital mechanics
python examples/basic_example_fixed.py

# Phase 2 example - Perturbations and attitude dynamics
python examples/phase2_example.py
```

##  Technical Background

This implementation is based on the theoretical framework presented in:

> Okasha, M., & Newman, B. (2014). Relative motion guidance, navigation and control for autonomous orbital rendezvous. Journal of Aerospace Technology and Management, 6(3), 301-318.

### Key Technical Components

1. **Gauss' Variational Equations**: For orbital element propagation under perturbations
2. **J2 Perturbation Model**: Earth's oblateness effects on orbital motion
3. **Atmospheric Drag Model**: Exponential atmosphere with Earth rotation effects
4. **Quaternion Attitude Dynamics**: Singularity-free attitude representation
5. **Gravity Gradient Torques**: Environmental torques for attitude dynamics
6. **LVLH Coordinate Frame**: Local-Vertical-Local-Horizontal reference frame
7. **Clohessy-Wiltshire Equations**: Linear relative motion dynamics
8. **State Transition Matrix**: Analytical solution for relative motion propagation

## Project Structure

```
orbital-rendezvous-control/
├── src/                          # Source code
│   ├── dynamics/                 # Orbital and attitude dynamics
│   │   ├── orbital_elements.py   # Orbital elements and conversions
│   │   ├── perturbations.py      # J2 and drag perturbations
│   │   ├── attitude_dynamics.py  # Quaternion attitude dynamics
│   │   └── relative_motion.py    # Relative motion dynamics
│   ├── navigation/               # Navigation and filtering (Phase 3)
│   ├── control/                  # Control systems (Phase 4)
│   ├── simulation/               # Simulation environment (Phase 5)
│   └── utils/                    # Utilities and constants
├── tests/                        # Unit tests
│   └── test_dynamics/            # Dynamics module tests
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

### Phase 3: Navigation (In Progress)
- [ ] Extended Kalman Filter implementation
- [ ] Sensor models (LIDAR, star tracker, gyros)
- [ ] Measurement models and linearization

### Phase 4: Control
- [ ] LQR translational controller
- [ ] Quaternion-based attitude controller
- [ ] Coupled control strategies

### Phase 5: Simulation
- [ ] Complete spacecraft model
- [ ] Orbital environment simulation
- [ ] Monte Carlo analysis tools

<<<<<<< HEAD
## 📊 Performance Metrics

- **Code Coverage**: >85% (Phase 1-2)
- **Documentation**: Complete API documentation
- **Validation**: Comparison with published results
- **Performance**: Real-time capable simulation

## Author
=======
## Author
>>>>>>> 8fd08889de1a4586ba4d0213b2080e8cf7e21736

**Arthur Allex Feliphe Barbosa Moreno**
- Institution: IME - Instituto Militar de Engenharia
- Email: 
- LinkedIn: [arthurmoreno](https://www.linkedin.com/in/arthurmoreno/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

<<<<<<< HEAD
---

*This project demonstrates advanced aerospace engineering capabilities and serves as a comprehensive portfolio piece showcasing expertise in orbital mechanics, attitude dynamics, perturbation modeling, and spacecraft control systems.*
=======
## Performance Metrics

- **Code Coverage**: Target >90%
- **Documentation**: Complete API documentation
- **Validation**: Comparison with published results
- **Performance**: Real-time capable simulation


>>>>>>> 8fd08889de1a4586ba4d0213b2080e8cf7e21736

