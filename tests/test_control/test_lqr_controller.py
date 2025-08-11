"""
Tests for LQR Controllers

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from control.lqr_controller import (
    TranslationalLQRController,
    AttitudeLQRController,
    CoupledLQRController,
    LQRWeights,
    ControlLimits,
    create_default_lqr_weights,
    create_default_control_limits,
    analyze_lqr_performance
)
from dynamics.relative_motion import RelativeState
from dynamics.attitude_dynamics import AttitudeState
from dynamics.orbital_elements import OrbitalElements
from utils.constants import EARTH_MU


class TestLQRWeights:
    """Test LQR weights configuration."""
    
    def test_default_weights_creation(self):
        """Test creation of default LQR weights."""
        weights = create_default_lqr_weights()
        
        assert isinstance(weights, LQRWeights)
        assert weights.Q_position > 0
        assert weights.Q_velocity > 0
        assert weights.Q_attitude > 0
        assert weights.Q_angular_vel > 0
        assert weights.R_force > 0
        assert weights.R_torque > 0
    
    def test_weights_modification(self):
        """Test modification of LQR weights."""
        weights = LQRWeights(
            Q_position=2.0,
            Q_velocity=3.0,
            R_force=0.5
        )
        
        assert weights.Q_position == 2.0
        assert weights.Q_velocity == 3.0
        assert weights.R_force == 0.5


class TestControlLimits:
    """Test control limits configuration."""
    
    def test_default_limits_creation(self):
        """Test creation of default control limits."""
        limits = create_default_control_limits()
        
        assert isinstance(limits, ControlLimits)
        assert limits.max_force > 0
        assert limits.max_torque > 0
        assert limits.max_force_rate > 0
        assert limits.max_torque_rate > 0
    
    def test_limits_modification(self):
        """Test modification of control limits."""
        limits = ControlLimits(
            max_force=5.0,
            max_torque=2.0
        )
        
        assert limits.max_force == 5.0
        assert limits.max_torque == 2.0


class TestTranslationalLQRController:
    """Test translational LQR controller."""
    
    @pytest.fixture
    def controller(self):
        """Create test controller."""
        weights = create_default_lqr_weights()
        limits = create_default_control_limits()
        return TranslationalLQRController(weights, limits, spacecraft_mass=10.0)
    
    @pytest.fixture
    def target_orbit(self):
        """Create test target orbit."""
        return OrbitalElements(
            a=7000e3, e=0.0, i=np.radians(51.6),
            omega_cap=0.0, omega=0.0, f=0.0, mu=EARTH_MU
        )
    
    def test_controller_initialization(self, controller):
        """Test controller initialization."""
        assert controller.mass == 10.0
        assert controller.K_trans is None
        assert controller.target_orbit is None
    
    def test_set_target_orbit(self, controller, target_orbit):
        """Test setting target orbit."""
        controller.set_target_orbit(target_orbit)
        
        assert controller.target_orbit is not None
        assert controller.K_trans is not None
        assert controller.K_trans.shape == (3, 6)  # 3 controls, 6 states
    
    def test_compute_control_without_orbit(self, controller):
        """Test control computation without target orbit."""
        current_state = RelativeState(
            position=np.array([1.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            time=0.0
        )
        desired_state = RelativeState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            time=0.0
        )
        
        with pytest.raises(RuntimeError):
            controller.compute_control(current_state, desired_state, 0.0)
    
    def test_compute_control_with_orbit(self, controller, target_orbit):
        """Test control computation with target orbit."""
        controller.set_target_orbit(target_orbit)
        
        current_state = RelativeState(
            position=np.array([10.0, 5.0, 2.0]),
            velocity=np.array([0.1, -0.05, 0.02]),
            time=0.0
        )
        desired_state = RelativeState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            time=0.0
        )
        
        force = controller.compute_control(current_state, desired_state, 0.0)
        
        assert isinstance(force, np.ndarray)
        assert force.shape == (3,)
        assert np.all(np.abs(force) <= controller.limits.max_force)
    
    def test_control_limits(self, controller, target_orbit):
        """Test control limits enforcement."""
        controller.set_target_orbit(target_orbit)
        
        # Large error to trigger limits
        current_state = RelativeState(
            position=np.array([1000.0, 1000.0, 1000.0]),
            velocity=np.array([10.0, 10.0, 10.0]),
            time=0.0
        )
        desired_state = RelativeState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            time=0.0
        )
        
        force = controller.compute_control(current_state, desired_state, 0.0)
        
        # Should be limited
        assert np.all(np.abs(force) <= controller.limits.max_force)
    
    def test_rate_limits(self, controller, target_orbit):
        """Test control rate limits."""
        controller.set_target_orbit(target_orbit)
        
        current_state = RelativeState(
            position=np.array([10.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        desired_state = RelativeState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            time=0.0
        )
        
        # First control command
        force1 = controller.compute_control(current_state, desired_state, 0.0)
        
        # Second control command (should be rate limited)
        force2 = controller.compute_control(current_state, desired_state, 0.1)
        
        # Rate should be limited
        rate = np.linalg.norm(force2 - force1) / 0.1
        assert rate <= controller.limits.max_force_rate * 1.1  # Small tolerance
    
    def test_stability_analysis(self, controller, target_orbit):
        """Test stability analysis."""
        controller.set_target_orbit(target_orbit)
        
        stability = controller.analyze_stability()
        
        assert 'max_real_part' in stability
        assert 'stability_margin' in stability
        assert stability['max_real_part'] < 0  # Stable system
        assert stability['stability_margin'] > 0


class TestAttitudeLQRController:
    """Test attitude LQR controller."""
    
    @pytest.fixture
    def controller(self):
        """Create test controller."""
        weights = create_default_lqr_weights()
        limits = create_default_control_limits()
        inertia = np.diag([1.0, 2.0, 3.0])  # kg⋅m²
        return AttitudeLQRController(weights, limits, inertia)
    
    def test_controller_initialization(self, controller):
        """Test controller initialization."""
        assert controller.K_att is not None
        assert controller.K_att.shape == (3, 6)  # 3 torques, 6 states
    
    def test_compute_control(self, controller):
        """Test attitude control computation."""
        current_attitude = AttitudeState(
            quaternion=np.array([0.9, 0.1, 0.1, 0.1]),
            angular_velocity=np.array([0.01, -0.02, 0.005]),
            time=0.0
        )
        desired_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        torque = controller.compute_control(current_attitude, desired_attitude, 0.0)
        
        assert isinstance(torque, np.ndarray)
        assert torque.shape == (3,)
        assert np.all(np.abs(torque) <= controller.limits.max_torque)
    
    def test_quaternion_error_computation(self, controller):
        """Test quaternion error computation."""
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([0.707, 0.707, 0.0, 0.0])  # 90° rotation about x
        
        q_error = controller._quaternion_error(q1, q2)
        
        assert isinstance(q_error, np.ndarray)
        assert q_error.shape == (4,)
        assert abs(np.linalg.norm(q_error) - 1.0) < 1e-10  # Unit quaternion
    
    def test_control_limits(self, controller):
        """Test torque limits enforcement."""
        # Large attitude error
        current_attitude = AttitudeState(
            quaternion=np.array([0.0, 1.0, 0.0, 0.0]),  # 180° rotation
            angular_velocity=np.array([1.0, 1.0, 1.0]),
            time=0.0
        )
        desired_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        torque = controller.compute_control(current_attitude, desired_attitude, 0.0)
        
        # Should be limited
        assert np.all(np.abs(torque) <= controller.limits.max_torque)
    
    def test_stability_analysis(self, controller):
        """Test stability analysis."""
        stability = controller.analyze_stability()
        
        assert 'max_real_part' in stability
        assert 'stability_margin' in stability
        assert stability['max_real_part'] < 0  # Stable system


class TestCoupledLQRController:
    """Test coupled LQR controller."""
    
    @pytest.fixture
    def controller(self):
        """Create test controller."""
        weights = create_default_lqr_weights()
        limits = create_default_control_limits()
        inertia = np.diag([1.0, 2.0, 3.0])
        return CoupledLQRController(weights, limits, 10.0, inertia, coupling_strength=0.1)
    
    @pytest.fixture
    def target_orbit(self):
        """Create test target orbit."""
        return OrbitalElements(
            a=7000e3, e=0.0, i=np.radians(51.6),
            omega_cap=0.0, omega=0.0, f=0.0, mu=EARTH_MU
        )
    
    def test_controller_initialization(self, controller):
        """Test controller initialization."""
        assert controller.coupling_strength == 0.1
        assert controller.K_coupled is None
        assert hasattr(controller, 'trans_controller')
        assert hasattr(controller, 'att_controller')
    
    def test_set_target_orbit(self, controller, target_orbit):
        """Test setting target orbit."""
        controller.set_target_orbit(target_orbit)
        
        assert controller.target_orbit is not None
        assert controller.K_coupled is not None
        assert controller.K_coupled.shape == (6, 12)  # 6 controls, 12 states
    
    def test_compute_coupled_control(self, controller, target_orbit):
        """Test coupled control computation."""
        controller.set_target_orbit(target_orbit)
        
        current_relative = RelativeState(
            position=np.array([10.0, 5.0, 2.0]),
            velocity=np.array([0.1, -0.05, 0.02]),
            time=0.0
        )
        desired_relative = RelativeState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            time=0.0
        )
        current_attitude = AttitudeState(
            quaternion=np.array([0.9, 0.1, 0.1, 0.1]),
            angular_velocity=np.array([0.01, -0.02, 0.005]),
            time=0.0
        )
        desired_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        force, torque = controller.compute_control(
            current_relative, desired_relative,
            current_attitude, desired_attitude, 0.0
        )
        
        assert isinstance(force, np.ndarray)
        assert isinstance(torque, np.ndarray)
        assert force.shape == (3,)
        assert torque.shape == (3,)
        assert np.all(np.abs(force) <= controller.limits.max_force)
        assert np.all(np.abs(torque) <= controller.limits.max_torque)
    
    def test_fallback_to_individual_controllers(self, controller):
        """Test fallback when coupled gains not available."""
        current_relative = RelativeState(
            position=np.array([1.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        desired_relative = RelativeState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            time=0.0
        )
        current_attitude = AttitudeState(
            quaternion=np.array([0.9, 0.1, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        desired_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        # Should use individual controllers when coupled gains not available
        force, torque = controller.compute_control(
            current_relative, desired_relative,
            current_attitude, desired_attitude, 0.0
        )
        
        assert isinstance(force, np.ndarray)
        assert isinstance(torque, np.ndarray)
    
    def test_stability_analysis(self, controller, target_orbit):
        """Test coupled stability analysis."""
        controller.set_target_orbit(target_orbit)
        
        stability = controller.analyze_stability()
        
        assert 'max_real_part' in stability
        assert 'stability_margin' in stability
        assert 'coupling_effect' in stability
        assert stability['coupling_effect'] == 0.1


class TestLQRPerformanceAnalysis:
    """Test LQR performance analysis functions."""
    
    def test_analyze_lqr_performance_empty_results(self):
        """Test performance analysis with empty results."""
        controller = TranslationalLQRController(
            create_default_lqr_weights(),
            create_default_control_limits(),
            10.0
        )
        
        results = analyze_lqr_performance(controller, {})
        
        assert isinstance(results, dict)
        assert len(results) == 0
    
    def test_analyze_lqr_performance_with_data(self):
        """Test performance analysis with simulation data."""
        weights = create_default_lqr_weights()
        limits = create_default_control_limits()
        controller = TranslationalLQRController(weights, limits, 10.0)
        
        # Set target orbit for stability analysis
        target_orbit = OrbitalElements(
            a=7000e3, e=0.0, i=np.radians(51.6),
            omega_cap=0.0, omega=0.0, f=0.0, mu=EARTH_MU
        )
        controller.set_target_orbit(target_orbit)
        
        # Mock simulation results
        simulation_results = {
            'position_errors': np.random.randn(100, 3) * 0.1,
            'velocity_errors': np.random.randn(100, 3) * 0.01,
            'control_forces': np.random.randn(100, 3) * 0.5,
            'dt': 1.0,
            'mass': 10.0
        }
        
        metrics = analyze_lqr_performance(controller, simulation_results)
        
        assert 'position_rms_error' in metrics
        assert 'velocity_rms_error' in metrics
        assert 'force_rms' in metrics
        assert 'total_delta_v' in metrics
        assert 'stability_max_real_part' in metrics
        
        assert metrics['position_rms_error'] > 0
        assert metrics['velocity_rms_error'] > 0
        assert metrics['force_rms'] > 0
        assert metrics['total_delta_v'] > 0


if __name__ == '__main__':
    pytest.main([__file__])

