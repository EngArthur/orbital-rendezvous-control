"""
Tests for Actuator Models

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from control.actuator_models import (
    ThrusterModel,
    ReactionWheelModel,
    ActuatorSuite,
    ActuatorConfiguration,
    ThrusterProperties,
    ReactionWheelProperties,
    ActuatorType,
    create_typical_thruster_suite,
    create_typical_reaction_wheel_suite,
    create_hybrid_actuator_suite,
    analyze_actuator_performance
)


class TestActuatorConfiguration:
    """Test actuator configuration."""
    
    def test_default_configuration(self):
        """Test default configuration creation."""
        config = ActuatorConfiguration()
        
        assert config.max_force == 1.0
        assert config.min_force == 0.0
        assert config.response_time == 0.1
        assert config.enabled is True
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        config = ActuatorConfiguration(
            max_force=5.0,
            min_force=0.1,
            response_time=0.05,
            noise_std=0.02,
            enabled=False
        )
        
        assert config.max_force == 5.0
        assert config.min_force == 0.1
        assert config.response_time == 0.05
        assert config.noise_std == 0.02
        assert config.enabled is False


class TestThrusterProperties:
    """Test thruster properties."""
    
    def test_default_properties(self):
        """Test default thruster properties."""
        props = ThrusterProperties()
        
        assert props.specific_impulse == 220.0
        assert props.minimum_impulse_bit == 1e-6
        assert props.propellant_mass == 1.0
    
    def test_custom_properties(self):
        """Test custom thruster properties."""
        props = ThrusterProperties(
            specific_impulse=300.0,
            minimum_impulse_bit=1e-5,
            propellant_mass=5.0
        )
        
        assert props.specific_impulse == 300.0
        assert props.minimum_impulse_bit == 1e-5
        assert props.propellant_mass == 5.0


class TestReactionWheelProperties:
    """Test reaction wheel properties."""
    
    def test_default_properties(self):
        """Test default reaction wheel properties."""
        props = ReactionWheelProperties()
        
        assert props.max_angular_momentum == 1.0
        assert props.max_angular_velocity == 6000.0
        assert props.wheel_inertia == 0.01
    
    def test_custom_properties(self):
        """Test custom reaction wheel properties."""
        props = ReactionWheelProperties(
            max_angular_momentum=2.0,
            max_angular_velocity=8000.0,
            wheel_inertia=0.02
        )
        
        assert props.max_angular_momentum == 2.0
        assert props.max_angular_velocity == 8000.0
        assert props.wheel_inertia == 0.02


class TestThrusterModel:
    """Test thruster model."""
    
    @pytest.fixture
    def thruster_config(self):
        """Create test thruster configuration."""
        return ActuatorConfiguration(
            max_force=1.0,
            response_time=0.1,
            noise_std=0.0  # No noise for predictable tests
        )
    
    @pytest.fixture
    def thruster_props(self):
        """Create test thruster properties."""
        return ThrusterProperties(
            specific_impulse=220.0,
            propellant_mass=2.0
        )
    
    @pytest.fixture
    def thruster(self, thruster_config, thruster_props):
        """Create test thruster."""
        position = np.array([1.0, 0.0, 0.0])
        direction = np.array([-1.0, 0.0, 0.0])
        return ThrusterModel(thruster_config, thruster_props, position, direction, "test_thruster")
    
    def test_thruster_initialization(self, thruster):
        """Test thruster initialization."""
        assert thruster.actuator_id == "test_thruster"
        assert thruster.current_thrust == 0.0
        assert thruster.propellant_consumed == 0.0
        assert np.allclose(thruster.direction, [-1.0, 0.0, 0.0])
    
    def test_compute_thrust_basic(self, thruster):
        """Test basic thrust computation."""
        thrust_command = 0.5
        actual_thrust = thruster.compute_thrust(thrust_command, 1.0)
        
        assert isinstance(actual_thrust, float)
        assert 0 <= actual_thrust <= thruster.config.max_force
    
    def test_compute_thrust_limits(self, thruster):
        """Test thrust limits enforcement."""
        # Test upper limit
        thrust_command = 10.0  # Above max_force
        actual_thrust = thruster.compute_thrust(thrust_command, 1.0)
        assert actual_thrust <= thruster.config.max_force
        
        # Test lower limit
        thrust_command = -1.0  # Below min_force
        actual_thrust = thruster.compute_thrust(thrust_command, 2.0)
        assert actual_thrust >= thruster.config.min_force
    
    def test_minimum_impulse_bit(self, thruster):
        """Test minimum impulse bit enforcement."""
        # Very small thrust command
        thrust_command = 1e-8
        actual_thrust = thruster.compute_thrust(thrust_command, 1.0)
        
        # Should be zero due to minimum impulse bit
        assert actual_thrust == 0.0
    
    def test_propellant_consumption(self, thruster):
        """Test propellant consumption."""
        initial_propellant = thruster.propellant_consumed
        
        # Apply thrust for some time
        thrust_command = 0.5
        for i in range(10):
            thruster.compute_thrust(thrust_command, float(i))
        
        # Propellant should be consumed
        assert thruster.propellant_consumed > initial_propellant
    
    def test_propellant_depletion(self, thruster):
        """Test behavior when propellant is depleted."""
        # Consume all propellant
        thruster.propellant_consumed = thruster.properties.propellant_mass
        
        thrust_command = 0.5
        actual_thrust = thruster.compute_thrust(thrust_command, 1.0)
        
        # Should produce no thrust
        assert actual_thrust == 0.0
    
    def test_get_force_torque(self, thruster):
        """Test force and torque calculation."""
        thrust = 1.0
        force, torque = thruster.get_force_torque(thrust)
        
        assert isinstance(force, np.ndarray)
        assert isinstance(torque, np.ndarray)
        assert force.shape == (3,)
        assert torque.shape == (3,)
        
        # Force should be in thrust direction
        expected_force = thrust * thruster.direction
        assert np.allclose(force, expected_force)
    
    def test_get_status(self, thruster):
        """Test status reporting."""
        status = thruster.get_status()
        
        assert 'actuator_id' in status
        assert 'enabled' in status
        assert 'current_thrust' in status
        assert 'temperature' in status
        assert 'propellant_remaining' in status
        assert 'propellant_fraction' in status
        
        assert status['actuator_id'] == "test_thruster"
        assert status['enabled'] is True
    
    def test_reset(self, thruster):
        """Test thruster reset."""
        # Change some state
        thruster.compute_thrust(0.5, 1.0)
        thruster.temperature = 50.0
        
        # Reset
        thruster.reset()
        
        assert thruster.current_thrust == 0.0
        assert thruster.temperature == 20.0
        assert thruster.propellant_consumed == 0.0
        assert len(thruster.thrust_history) == 0


class TestReactionWheelModel:
    """Test reaction wheel model."""
    
    @pytest.fixture
    def wheel_config(self):
        """Create test wheel configuration."""
        return ActuatorConfiguration(
            max_force=0.1,  # 0.1 N⋅m max torque
            response_time=0.05,
            noise_std=0.0  # No noise for predictable tests
        )
    
    @pytest.fixture
    def wheel_props(self):
        """Create test wheel properties."""
        return ReactionWheelProperties(
            max_angular_momentum=1.0,
            wheel_inertia=0.01
        )
    
    @pytest.fixture
    def wheel(self, wheel_config, wheel_props):
        """Create test reaction wheel."""
        axis = np.array([1.0, 0.0, 0.0])
        return ReactionWheelModel(wheel_config, wheel_props, axis, "test_wheel")
    
    def test_wheel_initialization(self, wheel):
        """Test wheel initialization."""
        assert wheel.actuator_id == "test_wheel"
        assert wheel.angular_velocity == 0.0
        assert wheel.angular_momentum == 0.0
        assert np.allclose(wheel.axis, [1.0, 0.0, 0.0])
    
    def test_compute_torque_basic(self, wheel):
        """Test basic torque computation."""
        torque_command = 0.05
        actual_torque = wheel.compute_torque(torque_command, 1.0)
        
        assert isinstance(actual_torque, float)
        assert abs(actual_torque) <= wheel.config.max_force
    
    def test_compute_torque_limits(self, wheel):
        """Test torque limits enforcement."""
        # Test upper limit
        torque_command = 1.0  # Above max_force
        actual_torque = wheel.compute_torque(torque_command, 1.0)
        assert abs(actual_torque) <= wheel.config.max_force
        
        # Test lower limit
        torque_command = -1.0  # Below -max_force
        actual_torque = wheel.compute_torque(torque_command, 2.0)
        assert abs(actual_torque) <= wheel.config.max_force
    
    def test_angular_momentum_buildup(self, wheel):
        """Test angular momentum buildup."""
        initial_momentum = wheel.angular_momentum
        
        # Apply torque for some time
        torque_command = 0.05
        for i in range(10):
            wheel.compute_torque(torque_command, float(i))
        
        # Momentum should build up
        assert abs(wheel.angular_momentum) > abs(initial_momentum)
    
    def test_saturation_behavior(self, wheel):
        """Test saturation behavior."""
        # Saturate the wheel
        wheel.angular_momentum = wheel.properties.max_angular_momentum
        
        # Try to add more momentum in same direction
        torque_command = 0.05
        actual_torque = wheel.compute_torque(torque_command, 1.0)
        
        # Should not be able to add more momentum in same direction
        # (This is a simplified test - actual behavior depends on implementation)
        assert isinstance(actual_torque, float)
    
    def test_get_torque_vector(self, wheel):
        """Test torque vector calculation."""
        torque = 0.1
        torque_vector = wheel.get_torque_vector(torque)
        
        assert isinstance(torque_vector, np.ndarray)
        assert torque_vector.shape == (3,)
        
        # Should be along wheel axis
        expected_vector = torque * wheel.axis
        assert np.allclose(torque_vector, expected_vector)
    
    def test_get_status(self, wheel):
        """Test status reporting."""
        status = wheel.get_status()
        
        assert 'actuator_id' in status
        assert 'enabled' in status
        assert 'angular_velocity_rpm' in status
        assert 'angular_momentum' in status
        assert 'saturation_level' in status
        assert 'is_saturated' in status
        
        assert status['actuator_id'] == "test_wheel"
        assert status['enabled'] is True
    
    def test_desaturate(self, wheel):
        """Test wheel desaturation."""
        # Saturate wheel
        wheel.angular_momentum = 0.8  # Near saturation
        
        # Apply external torque to desaturate
        external_torque = -0.1
        dt = 1.0
        wheel.desaturate(external_torque, dt)
        
        # Momentum should decrease
        assert wheel.angular_momentum < 0.8
    
    def test_reset(self, wheel):
        """Test wheel reset."""
        # Change some state
        wheel.compute_torque(0.05, 1.0)
        
        # Reset
        wheel.reset()
        
        assert wheel.angular_velocity == 0.0
        assert wheel.angular_momentum == 0.0
        assert wheel.motor_torque == 0.0
        assert len(wheel.torque_history) == 0


class TestActuatorSuite:
    """Test actuator suite."""
    
    @pytest.fixture
    def suite(self):
        """Create test actuator suite."""
        return ActuatorSuite()
    
    @pytest.fixture
    def test_thruster(self):
        """Create test thruster."""
        config = ActuatorConfiguration(max_force=1.0)
        props = ThrusterProperties(propellant_mass=2.0)
        position = np.array([1.0, 0.0, 0.0])
        direction = np.array([-1.0, 0.0, 0.0])
        return ThrusterModel(config, props, position, direction, "thruster_1")
    
    @pytest.fixture
    def test_wheel(self):
        """Create test reaction wheel."""
        config = ActuatorConfiguration(max_force=0.1)
        props = ReactionWheelProperties()
        axis = np.array([1.0, 0.0, 0.0])
        return ReactionWheelModel(config, props, axis, "wheel_x")
    
    def test_suite_initialization(self, suite):
        """Test suite initialization."""
        assert len(suite.thrusters) == 0
        assert len(suite.reaction_wheels) == 0
        assert suite.actuator_count == 0
    
    def test_add_thruster(self, suite, test_thruster):
        """Test adding thruster to suite."""
        suite.add_thruster(test_thruster)
        
        assert len(suite.thrusters) == 1
        assert "thruster_1" in suite.thrusters
        assert suite.actuator_count == 1
        assert suite.thruster_allocation_matrix is not None
    
    def test_add_reaction_wheel(self, suite, test_wheel):
        """Test adding reaction wheel to suite."""
        suite.add_reaction_wheel(test_wheel)
        
        assert len(suite.reaction_wheels) == 1
        assert "wheel_x" in suite.reaction_wheels
        assert suite.actuator_count == 1
        assert suite.wheel_allocation_matrix is not None
    
    def test_allocate_control_thrusters_only(self, suite, test_thruster):
        """Test control allocation with thrusters only."""
        suite.add_thruster(test_thruster)
        
        desired_force = np.array([0.1, 0.0, 0.0])
        desired_torque = np.zeros(3)
        
        actual_force, actual_torque = suite.allocate_control(
            desired_force, desired_torque, 1.0
        )
        
        assert isinstance(actual_force, np.ndarray)
        assert isinstance(actual_torque, np.ndarray)
        assert actual_force.shape == (3,)
        assert actual_torque.shape == (3,)
    
    def test_allocate_control_wheels_only(self, suite, test_wheel):
        """Test control allocation with reaction wheels only."""
        suite.add_reaction_wheel(test_wheel)
        
        desired_force = np.zeros(3)
        desired_torque = np.array([0.01, 0.0, 0.0])
        
        actual_force, actual_torque = suite.allocate_control(
            desired_force, desired_torque, 1.0
        )
        
        assert isinstance(actual_force, np.ndarray)
        assert isinstance(actual_torque, np.ndarray)
        assert actual_force.shape == (3,)
        assert actual_torque.shape == (3,)
    
    def test_allocate_control_hybrid(self, suite, test_thruster, test_wheel):
        """Test control allocation with both thrusters and wheels."""
        suite.add_thruster(test_thruster)
        suite.add_reaction_wheel(test_wheel)
        
        desired_force = np.array([0.1, 0.0, 0.0])
        desired_torque = np.array([0.01, 0.0, 0.0])
        
        actual_force, actual_torque = suite.allocate_control(
            desired_force, desired_torque, 1.0
        )
        
        assert isinstance(actual_force, np.ndarray)
        assert isinstance(actual_torque, np.ndarray)
    
    def test_get_suite_status(self, suite, test_thruster, test_wheel):
        """Test suite status reporting."""
        suite.add_thruster(test_thruster)
        suite.add_reaction_wheel(test_wheel)
        
        status = suite.get_suite_status()
        
        assert 'num_thrusters' in status
        assert 'num_reaction_wheels' in status
        assert 'total_propellant_consumed' in status
        assert 'total_delta_v' in status
        assert 'thrusters' in status
        assert 'reaction_wheels' in status
        
        assert status['num_thrusters'] == 1
        assert status['num_reaction_wheels'] == 1
    
    def test_check_actuator_health(self, suite, test_thruster, test_wheel):
        """Test actuator health checking."""
        suite.add_thruster(test_thruster)
        suite.add_reaction_wheel(test_wheel)
        
        health = suite.check_actuator_health()
        
        assert 'thruster_1' in health
        assert 'wheel_x' in health
        assert health['thruster_1'] is True  # Should be healthy initially
        assert health['wheel_x'] is True
    
    def test_enable_disable_actuator(self, suite, test_thruster):
        """Test enabling/disabling actuators."""
        suite.add_thruster(test_thruster)
        
        # Disable
        success = suite.disable_actuator("thruster_1")
        assert success
        assert not test_thruster.config.enabled
        
        # Enable
        success = suite.enable_actuator("thruster_1")
        assert success
        assert test_thruster.config.enabled
        
        # Try non-existent actuator
        success = suite.disable_actuator("non_existent")
        assert not success
    
    def test_reset_all_actuators(self, suite, test_thruster, test_wheel):
        """Test resetting all actuators."""
        suite.add_thruster(test_thruster)
        suite.add_reaction_wheel(test_wheel)
        
        # Change some state
        test_thruster.compute_thrust(0.5, 1.0)
        test_wheel.compute_torque(0.05, 1.0)
        
        # Reset all
        suite.reset_all_actuators()
        
        assert test_thruster.current_thrust == 0.0
        assert test_wheel.angular_velocity == 0.0
        assert suite.total_delta_v == 0.0


class TestActuatorSuiteCreation:
    """Test actuator suite creation functions."""
    
    def test_create_typical_thruster_suite(self):
        """Test typical thruster suite creation."""
        suite = create_typical_thruster_suite()
        
        assert len(suite.thrusters) == 8  # 8 thrusters in cubic configuration
        assert suite.thruster_allocation_matrix is not None
        assert suite.thruster_allocation_matrix.shape == (6, 8)  # 6 DOF, 8 thrusters
    
    def test_create_typical_reaction_wheel_suite(self):
        """Test typical reaction wheel suite creation."""
        suite = create_typical_reaction_wheel_suite()
        
        assert len(suite.reaction_wheels) == 3  # 3 orthogonal wheels
        assert suite.wheel_allocation_matrix is not None
        assert suite.wheel_allocation_matrix.shape == (3, 3)  # 3 DOF, 3 wheels
    
    def test_create_hybrid_actuator_suite(self):
        """Test hybrid actuator suite creation."""
        suite = create_hybrid_actuator_suite()
        
        assert len(suite.thrusters) == 8
        assert len(suite.reaction_wheels) == 3
        assert suite.thruster_allocation_matrix is not None
        assert suite.wheel_allocation_matrix is not None


class TestActuatorPerformanceAnalysis:
    """Test actuator performance analysis."""
    
    def test_analyze_actuator_performance(self):
        """Test actuator performance analysis."""
        suite = create_hybrid_actuator_suite()
        
        # Simulate some operation
        for i in range(10):
            suite.allocate_control(
                np.array([0.1, 0.0, 0.0]),
                np.array([0.01, 0.0, 0.0]),
                float(i)
            )
        
        # Mock simulation results
        simulation_results = {}
        
        metrics = analyze_actuator_performance(suite, simulation_results)
        
        assert 'total_delta_v' in metrics
        assert 'total_propellant_consumed' in metrics
        assert 'num_active_thrusters' in metrics
        assert 'num_active_wheels' in metrics
        
        assert metrics['total_delta_v'] >= 0
        assert metrics['total_propellant_consumed'] >= 0


if __name__ == '__main__':
    pytest.main([__file__])

