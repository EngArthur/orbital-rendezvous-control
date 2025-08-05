# Orbital Rendezvous Control System

A complete Python implementation of autonomous orbital rendezvous guidance, navigation, and control based on the research paper "Relative Motion Guidance, Navigation and Control for Autonomous Orbital Rendezvous" by Okasha & Newman (2014).

## 🚀 Project Overview

This project demonstrates advanced expertise in:
- **Orbital Mechanics**: Complete implementation of orbital dynamics and perturbations
- **Navigation Systems**: Extended Kalman Filter for relative state estimation
- **Control Systems**: Coupled translational and rotational control
- **Aerospace Engineering**: High-fidelity spacecraft simulation

## 📋 Features

### ✅ Implemented
- **Orbital Elements**: Complete orbital elements representation and conversions
- **Coordinate Systems**: ECI, LVLH (Hill frame), RSW transformations
- **Mathematical Utilities**: Quaternions, rotation matrices, Kepler equation solver
- **Unit Testing**: Comprehensive test suite for validation

### 🚧 In Development
- **Perturbation Models**: J2 gravitational and atmospheric drag effects
- **Attitude Dynamics**: Quaternion-based attitude representation and control
- **Extended Kalman Filter**: Relative navigation with sensor fusion
- **Control Systems**: LQR-based translational and rotational controllers
- **Simulation Environment**: Complete orbital rendezvous scenarios

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/orbital-rendezvous-control.git
cd orbital-rendezvous-control

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 📖 Usage

### Basic Orbital Elements Example

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
print(f"Position: {position}")
print(f"Velocity: {velocity}")
print(f"Orbital Period: {elements.period/3600:.2f} hours")
```

### Coordinate Transformations

```python
from src.dynamics.orbital_elements import cartesian_to_orbital_elements

# Convert Cartesian state back to orbital elements
recovered_elements = cartesian_to_orbital_elements(position, velocity)
print(f"Semi-major axis: {recovered_elements.a/1000:.1f} km")
print(f"Eccentricity: {recovered_elements.e:.6f}")
```

## 🧪 Testing

Run the test suite to validate implementations:

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_dynamics/test_orbital_elements.py

# Run with coverage
pytest --cov=src tests/
```

## 📚 Technical Background

This implementation is based on the theoretical framework presented in:

> Okasha, M., & Newman, B. (2014). Relative motion guidance, navigation and control for autonomous orbital rendezvous. Journal of Aerospace Technology and Management, 6(3), 301-318.

### Key Technical Components

1. **Gauss' Variational Equations**: For orbital element propagation under perturbations
2. **LVLH Coordinate Frame**: Local-Vertical-Local-Horizontal reference frame
3. **Extended Kalman Filter**: For relative state estimation using LIDAR and star tracker
4. **Coupled Control**: Simultaneous translational and rotational control

## 🏗️ Project Structure

```
orbital-rendezvous-control/
├── src/                          # Source code
│   ├── dynamics/                 # Orbital and attitude dynamics
│   ├── navigation/               # Navigation and filtering
│   ├── control/                  # Control systems
│   ├── simulation/               # Simulation environment
│   └── utils/                    # Utilities and constants
├── tests/                        # Unit tests
├── examples/                     # Usage examples
├── docs/                         # Documentation
└── data/                         # Simulation data and results
```

## 🎯 Development Roadmap

### Phase 1: Foundations ✅
- [x] Orbital elements and conversions
- [x] Mathematical utilities
- [x] Unit testing framework

### Phase 2: Dynamics (In Progress)
- [ ] Perturbation models (J2, atmospheric drag)
- [ ] Attitude dynamics with quaternions
- [ ] Relative motion equations

### Phase 3: Navigation
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

## 👨‍💻 Author

**Arthur Allex Feliphe Barbosa Moreno**
- Institution: IME - Instituto Militar de Engenharia
- Email: arthur.moreno@ime.eb.br
- LinkedIn: [arthurmoreno](https://www.linkedin.com/in/arthurmoreno/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📊 Performance Metrics

- **Code Coverage**: Target >90%
- **Documentation**: Complete API documentation
- **Validation**: Comparison with published results
- **Performance**: Real-time capable simulation

---

*This project demonstrates advanced aerospace engineering capabilities and serves as a comprehensive portfolio piece showcasing expertise in orbital mechanics, navigation, and control systems.*

