"""
Unit tests for attitude dynamics module.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import pytest
import numpy as np
from src.dynamics.attitude_dynamics import (
    SpacecraftInertia, AttitudeState, quaternion_kinematics,
    euler_equations, gravity_gradient_torque, propagate_attitude_state,
    attitude_error_quaternion, quaternion_from_euler_angles,
    euler_angles_from_quaternion
)
from src.utils.constants import EARTH_RADIUS


class TestSpacecraftInertia:
    """Test cases for SpacecraftInertia class."""
    
    def test_inertia_creation(self):
        """Test creation of spacecraft inertia."""
        inertia = SpacecraftInertia(
            Ixx=100.0,
            Iyy=150.0,
            Izz=200.0
        )
        
        assert inertia.Ixx == 100.0
        assert inertia.Iyy == 150.0
        assert inertia.Izz == 200.0
        assert inertia.Ixy == 0.0  # Default value
    
    def test_inertia_matrix(self):
        """Test inertia matrix calculation."""
        inertia = SpacecraftInertia(
            Ixx=100.0, Iyy=150.0, Izz=200.0,
            Ixy=10.0, Ixz=20.0, Iyz=30.0
        )
        
        I = inertia.inertia_matrix
        
        assert I[0, 0] == 100.0
        assert I[1, 1] == 150.0
        assert I[2, 2] == 200.0
        assert I[0, 1] == -10.0
        assert I[0, 2] == -20.0
        assert I[1, 2] == -30.0
    
    def test_principal_axes_check(self):
        """Test principal axes alignment check."""
        # Principal axes aligned
        inertia1 = SpacecraftInertia(Ixx=100.0, Iyy=150.0, Izz=200.0)
        assert inertia1.is_principal_axes()
        
        # Not aligned
        inertia2 = SpacecraftInertia(
            Ixx=100.0, Iyy=150.0, Izz=200.0, Ixy=10.0
        )
        assert not inertia2.is_principal_axes()


class TestAttitudeState:
    """Test cases for AttitudeState class."""
    
    def test_attitude_state_creation(self):
        """Test creation of attitude state."""
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        omega = np.array([0.1, 0.2, 0.3])
        
        state = AttitudeState(q, omega)
        
        assert np.allclose(state.quaternion, q)
        assert np.allclose(state.angular_velocity, omega)
        assert state.time == 0.0
    
    def test_quaternion_normalization(self):
        """Test automatic quaternion normalization."""
        q = np.array([2.0, 0.0, 0.0, 0.0])  # Non-unit quaternion
        omega = np.array([0.0, 0.0, 0.0])
        
        state = AttitudeState(q, omega)
        
        # Should be normalized
        assert abs(np.linalg.norm(state.quaternion) - 1.0) < 1e-10
        assert abs(state.quaternion[0] - 1.0) < 1e-10
    
    def test_rotation_matrix_property(self):
        """Test rotation matrix property."""
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        omega = np.array([0.0, 0.0, 0.0])
        
        state = AttitudeState(q, omega)
        R = state.rotation_matrix
        
        # Should be identity matrix
        assert np.allclose(R, np.eye(3))


class TestQuaternionKinematics:
    """Test cases for quaternion kinematics."""
    
    def test_quaternion_kinematics_zero_angular_velocity(self):
        """Test quaternion kinematics with zero angular velocity."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.0, 0.0, 0.0])
        
        q_dot = quaternion_kinematics(q, omega)
        
        # Derivative should be zero
        assert np.allclose(q_dot, np.zeros(4))
    
    def test_quaternion_kinematics_pure_rotation(self):
        """Test quaternion kinematics with pure rotation about z-axis."""
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
        omega = np.array([0.0, 0.0, 1.0])  # 1 rad/s about z
        
        q_dot = quaternion_kinematics(q, omega)
        
        # Expected derivative for pure z rotation
        expected = np.array([0.0, 0.0, 0.0, 0.5])
        assert np.allclose(q_dot, expected)


class TestEulerEquations:
    """Test cases for Euler's equations."""
    
    def test_euler_equations_no_torque(self):
        """Test Euler's equations with no external torque."""
        omega = np.array([0.1, 0.2, 0.3])
        inertia = SpacecraftInertia(Ixx=100.0, Iyy=150.0, Izz=200.0)
        torque = np.array([0.0, 0.0, 0.0])
        
        omega_dot = euler_equations(omega, inertia, torque)
        
        # Should have gyroscopic coupling effects
        assert not np.allclose(omega_dot, np.zeros(3))
    
    def test_euler_equations_principal_axes(self):
        """Test Euler's equations for principal axes."""
        omega = np.array([1.0, 0.0, 0.0])  # Pure rotation about x
        inertia = SpacecraftInertia(Ixx=100.0, Iyy=150.0, Izz=200.0)
        torque = np.array([0.0, 0.0, 0.0])
        
        omega_dot = euler_equations(omega, inertia, torque)
        
        # For principal axes with pure rotation, should be zero
        assert abs(omega_dot[0]) < 1e-10
        assert omega_dot[1] == 0.0  # No coupling for pure x rotation
        assert omega_dot[2] == 0.0


class TestGravityGradientTorque:
    """Test cases for gravity gradient torque."""
    
    def test_gravity_gradient_torque_aligned(self):
        """Test gravity gradient torque for aligned spacecraft."""
        q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        r_eci = np.array([EARTH_RADIUS + 400e3, 0.0, 0.0])  # Radial position
        
        # Spacecraft with different principal moments
        inertia = SpacecraftInertia(Ixx=100.0, Iyy=150.0, Izz=200.0)
        
        torque = gravity_gradient_torque(q, r_eci, inertia)
        
        # Should have torque due to inertia differences
        assert len(torque) == 3
    
    def test_gravity_gradient_torque_zero_position(self):
        """Test gravity gradient torque at zero position."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        r_eci = np.array([0.0, 0.0, 0.0])
        inertia = SpacecraftInertia(Ixx=100.0, Iyy=150.0, Izz=200.0)
        
        torque = gravity_gradient_torque(q, r_eci, inertia)
        
        # Should be zero at center of Earth
        assert np.allclose(torque, np.zeros(3))


class TestAttitudePropagation:
    """Test cases for attitude propagation."""
    
    def test_attitude_propagation_no_torque(self):
        """Test attitude propagation with no external torque."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([0.1, 0.0, 0.0])  # Small rotation about x
        state0 = AttitudeState(q0, omega0)
        
        inertia = SpacecraftInertia(Ixx=100.0, Iyy=100.0, Izz=100.0)  # Sphere
        torque = np.array([0.0, 0.0, 0.0])
        
        # Propagate for small time step
        state1 = propagate_attitude_state(state0, inertia, torque, 0.1)
        
        # Quaternion should remain normalized
        assert abs(np.linalg.norm(state1.quaternion) - 1.0) < 1e-10
        
        # For spherical inertia, angular velocity should remain constant
        assert np.allclose(state1.angular_velocity, omega0, atol=1e-6)
    
    def test_attitude_propagation_methods(self):
        """Test different integration methods."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        omega0 = np.array([0.1, 0.2, 0.3])
        state0 = AttitudeState(q0, omega0)
        
        inertia = SpacecraftInertia(Ixx=100.0, Iyy=150.0, Izz=200.0)
        torque = np.array([1.0, 2.0, 3.0])
        
        # Test both integration methods
        state_euler = propagate_attitude_state(state0, inertia, torque, 0.01, 'euler')
        state_rk4 = propagate_attitude_state(state0, inertia, torque, 0.01, 'rk4')
        
        # Both should produce normalized quaternions
        assert abs(np.linalg.norm(state_euler.quaternion) - 1.0) < 1e-10
        assert abs(np.linalg.norm(state_rk4.quaternion) - 1.0) < 1e-10
        
        # RK4 should be more accurate (but hard to test without analytical solution)
        assert not np.allclose(state_euler.quaternion, state_rk4.quaternion)


class TestAttitudeErrors:
    """Test cases for attitude error calculations."""
    
    def test_attitude_error_identity(self):
        """Test attitude error for identical quaternions."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([1.0, 0.0, 0.0, 0.0])
        
        q_error = attitude_error_quaternion(q1, q2)
        
        # Error should be identity quaternion
        assert np.allclose(q_error, np.array([1.0, 0.0, 0.0, 0.0]))
    
    def test_attitude_error_opposite(self):
        """Test attitude error for opposite quaternions."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([-1.0, 0.0, 0.0, 0.0])  # Same rotation, opposite sign
        
        q_error = attitude_error_quaternion(q1, q2)
        
        # Should still be identity (quaternions are double cover)
        assert np.allclose(abs(q_error), np.array([1.0, 0.0, 0.0, 0.0]))


class TestEulerAngles:
    """Test cases for Euler angle conversions."""
    
    def test_euler_angles_round_trip(self):
        """Test round-trip conversion: Euler -> quaternion -> Euler."""
        roll = np.radians(30)
        pitch = np.radians(45)
        yaw = np.radians(60)
        
        # Convert to quaternion
        q = quaternion_from_euler_angles(roll, pitch, yaw)
        
        # Convert back to Euler angles
        roll2, pitch2, yaw2 = euler_angles_from_quaternion(q)
        
        # Should match original angles
        assert abs(roll - roll2) < 1e-10
        assert abs(pitch - pitch2) < 1e-10
        assert abs(yaw - yaw2) < 1e-10
    
    def test_euler_angles_identity(self):
        """Test Euler angles for identity quaternion."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        
        roll, pitch, yaw = euler_angles_from_quaternion(q)
        
        # Should all be zero
        assert abs(roll) < 1e-10
        assert abs(pitch) < 1e-10
        assert abs(yaw) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])

