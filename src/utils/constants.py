"""
Physical and Mathematical Constants for Orbital Mechanics

This module contains fundamental constants used throughout the orbital
rendezvous control system implementation.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np

# Earth Physical Constants
EARTH_MU = 3.986004418e14  # Earth gravitational parameter [m³/s²]
EARTH_RADIUS = 6.3781363e6  # Earth mean radius [m]
EARTH_J2 = 1.08262668e-3   # Earth J2 coefficient (oblateness)
EARTH_ROTATION_RATE = 7.2921159e-5  # Earth rotation rate [rad/s]

# Atmospheric Model Constants
EARTH_ATMOSPHERE_SCALE_HEIGHT = 8500.0  # Scale height [m]
EARTH_ATMOSPHERE_DENSITY_SEA_LEVEL = 1.225  # Sea level density [kg/m³]

# Mathematical Constants
PI = np.pi
TWO_PI = 2.0 * np.pi
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

# Numerical Tolerances
TOLERANCE_POSITION = 1e-6  # Position tolerance [m]
TOLERANCE_VELOCITY = 1e-9  # Velocity tolerance [m/s]
TOLERANCE_ANGLE = 1e-12    # Angular tolerance [rad]
TOLERANCE_TIME = 1e-6      # Time tolerance [s]

# Integration Parameters
DEFAULT_INTEGRATION_STEP = 1.0  # Default integration step [s]
MAX_INTEGRATION_STEP = 60.0     # Maximum integration step [s]
MIN_INTEGRATION_STEP = 0.01     # Minimum integration step [s]

# Orbital Element Limits
MAX_ECCENTRICITY = 0.99999  # Maximum allowed eccentricity
MIN_SEMI_MAJOR_AXIS = EARTH_RADIUS + 200e3  # Minimum altitude 200 km
MAX_SEMI_MAJOR_AXIS = EARTH_RADIUS + 50000e3  # Maximum altitude 50,000 km

# Control System Parameters
MAX_THRUST_ACCELERATION = 10.0  # Maximum thrust acceleration [m/s²]
MAX_ANGULAR_ACCELERATION = 0.1  # Maximum angular acceleration [rad/s²]

# Sensor Parameters
LIDAR_MAX_RANGE = 10000.0  # Maximum LIDAR range [m]
LIDAR_ACCURACY = 0.1       # LIDAR accuracy [m]
STAR_TRACKER_ACCURACY = 1e-5  # Star tracker accuracy [rad]
GYRO_BIAS_STABILITY = 1e-6    # Gyro bias stability [rad/s]

# Simulation Parameters
DEFAULT_SIMULATION_TIME = 86400.0  # Default simulation time [s] (1 day)
DEFAULT_TIME_STEP = 1.0            # Default time step [s]

