"""
Tests for Guidance Laws

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from control.guidance_laws import (
    LinearGuidanceLaw,
    NonlinearGuidanceLaw,
    AdaptiveGuidanceLaw,
    GuidanceWaypoint,
    GuidanceConstraints,
    GuidancePhase,
    create_approach_trajectory,
    create_station_keeping_trajectory,
    analyze_guidance_performance
)
from dynamics.relative_motion import RelativeState
from dynamics.attitude_dynamics import AttitudeState
from dynamics.orbital_elements import OrbitalElements
from utils.constants import EARTH_MU


class TestGuidanceWaypoint:
    """Test guidance waypoint functionality."""
    
    def test_waypoint_creation(self):
        """Test waypoint creation."""
        waypoint = GuidanceWaypoint(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.array([-0.1, 0.0, 0.0]),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=100.0,
            phase=GuidancePhase.APPROACH
        )
        
        assert np.allclose(waypoint.position, [100.0, 0.0, 0.0])
        assert np.allclose(waypoint.velocity, [-0.1, 0.0, 0.0])
        assert waypoint.time == 100.0
        assert waypoint.phase == GuidancePhase.APPROACH
    
    def test_waypoint_with_constraints(self):
        """Test waypoint with additional constraints."""
        constraints = {'max_approach_angle': 30.0}
        
        waypoint = GuidanceWaypoint(
            position=np.array([50.0, 0.0, 0.0]),
            velocity=np.array([-0.05, 0.0, 0.0]),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=50.0,
            phase=GuidancePhase.PROXIMITY,
            constraints=constraints
        )
        
        assert waypoint.constraints['max_approach_angle'] == 30.0


class TestGuidanceConstraints:
    """Test guidance constraints."""
    
    def test_default_constraints(self):
        """Test default constraints creation."""
        constraints = GuidanceConstraints()
        
        assert constraints.max_velocity > 0
        assert constraints.max_acceleration > 0
        assert constraints.min_range > 0
        assert constraints.approach_corridor_angle > 0
    
    def test_custom_constraints(self):
        """Test custom constraints."""
        constraints = GuidanceConstraints(
            max_velocity=0.5,
            max_acceleration=0.05,
            min_range=5.0,
            station_keeping_box=np.array([10.0, 5.0, 5.0])
        )
        
        assert constraints.max_velocity == 0.5
        assert constraints.max_acceleration == 0.05
        assert constraints.min_range == 5.0
        assert np.allclose(constraints.station_keeping_box, [10.0, 5.0, 5.0])


class TestLinearGuidanceLaw:
    """Test linear guidance law."""
    
    @pytest.fixture
    def target_orbit(self):
        """Create test target orbit."""
        return OrbitalElements(
            a=7000e3, e=0.0, i=np.radians(51.6),
            omega_cap=0.0, omega=0.0, f=0.0, mu=EARTH_MU
        )
    
    @pytest.fixture
    def constraints(self):
        """Create test constraints."""
        return GuidanceConstraints(
            max_velocity=1.0,
            max_acceleration=0.1,
            min_range=10.0
        )
    
    @pytest.fixture
    def guidance_law(self, target_orbit, constraints):
        """Create test guidance law."""
        return LinearGuidanceLaw(target_orbit, constraints)
    
    def test_initialization(self, guidance_law, target_orbit):
        """Test guidance law initialization."""
        assert guidance_law.target_orbit == target_orbit
        assert guidance_law.mean_motion == target_orbit.mean_motion
        assert len(guidance_law.waypoints) == 0
    
    def test_add_waypoint(self, guidance_law):
        """Test adding waypoints."""
        waypoint = GuidanceWaypoint(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.array([-0.1, 0.0, 0.0]),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=100.0,
            phase=GuidancePhase.APPROACH
        )
        
        guidance_law.add_waypoint(waypoint)
        
        assert len(guidance_law.waypoints) == 1
        assert guidance_law.waypoints[0] == waypoint
    
    def test_waypoint_sorting(self, guidance_law):
        """Test waypoints are sorted by time."""
        waypoint1 = GuidanceWaypoint(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=200.0,
            phase=GuidancePhase.APPROACH
        )
        
        waypoint2 = GuidanceWaypoint(
            position=np.array([50.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=100.0,
            phase=GuidancePhase.PROXIMITY
        )
        
        guidance_law.add_waypoint(waypoint1)
        guidance_law.add_waypoint(waypoint2)
        
        assert guidance_law.waypoints[0].time == 100.0
        assert guidance_law.waypoints[1].time == 200.0
    
    def test_generate_trajectory_no_waypoints(self, guidance_law):
        """Test trajectory generation with no waypoints."""
        initial_state = RelativeState(
            position=np.array([200.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        initial_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        success = guidance_law.generate_trajectory(
            initial_state, initial_attitude, (0.0, 100.0), 1.0
        )
        
        assert not success
    
    def test_generate_trajectory_with_waypoints(self, guidance_law):
        """Test trajectory generation with waypoints."""
        # Add waypoints
        waypoint1 = GuidanceWaypoint(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.array([-0.1, 0.0, 0.0]),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=50.0,
            phase=GuidancePhase.APPROACH
        )
        
        waypoint2 = GuidanceWaypoint(
            position=np.array([10.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=100.0,
            phase=GuidancePhase.DOCKING
        )
        
        guidance_law.add_waypoint(waypoint1)
        guidance_law.add_waypoint(waypoint2)
        
        initial_state = RelativeState(
            position=np.array([200.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        initial_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        success = guidance_law.generate_trajectory(
            initial_state, initial_attitude, (0.0, 120.0), 1.0
        )
        
        assert success
        assert len(guidance_law.trajectory_states) > 0
        assert len(guidance_law.trajectory_attitudes) > 0
        assert len(guidance_law.trajectory_time) > 0
    
    def test_get_reference_state(self, guidance_law):
        """Test getting reference state."""
        # Generate trajectory first
        waypoint = GuidanceWaypoint(
            position=np.array([50.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=50.0,
            phase=GuidancePhase.APPROACH
        )
        guidance_law.add_waypoint(waypoint)
        
        initial_state = RelativeState(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        initial_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        guidance_law.generate_trajectory(
            initial_state, initial_attitude, (0.0, 100.0), 1.0
        )
        
        ref_rel, ref_att = guidance_law.get_reference_state(25.0)
        
        assert isinstance(ref_rel, RelativeState)
        assert isinstance(ref_att, AttitudeState)
    
    def test_get_reference_state_no_trajectory(self, guidance_law):
        """Test getting reference state without trajectory."""
        with pytest.raises(RuntimeError):
            guidance_law.get_reference_state(10.0)
    
    def test_validate_trajectory(self, guidance_law):
        """Test trajectory validation."""
        # Generate simple trajectory
        waypoint = GuidanceWaypoint(
            position=np.array([20.0, 0.0, 0.0]),  # Above min_range
            velocity=np.array([-0.05, 0.0, 0.0]),  # Below max_velocity
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=50.0,
            phase=GuidancePhase.APPROACH
        )
        guidance_law.add_waypoint(waypoint)
        
        initial_state = RelativeState(
            position=np.array([50.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        initial_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        guidance_law.generate_trajectory(
            initial_state, initial_attitude, (0.0, 100.0), 1.0
        )
        
        validation = guidance_law.validate_trajectory()
        
        assert 'trajectory_exists' in validation
        assert validation['trajectory_exists']
        assert 'velocity_constraint' in validation
        assert 'range_constraint' in validation
        assert 'acceleration_constraint' in validation


class TestNonlinearGuidanceLaw:
    """Test nonlinear guidance law."""
    
    @pytest.fixture
    def target_orbit(self):
        """Create test target orbit."""
        return OrbitalElements(
            a=7000e3, e=0.0, i=np.radians(51.6),
            omega_cap=0.0, omega=0.0, f=0.0, mu=EARTH_MU
        )
    
    @pytest.fixture
    def constraints(self):
        """Create test constraints."""
        return GuidanceConstraints()
    
    @pytest.fixture
    def guidance_law(self, target_orbit, constraints):
        """Create test guidance law."""
        return NonlinearGuidanceLaw(target_orbit, constraints)
    
    def test_initialization(self, guidance_law):
        """Test nonlinear guidance law initialization."""
        assert len(guidance_law.waypoints) == 0
        assert len(guidance_law.control_history) == 0
    
    def test_generate_trajectory(self, guidance_law):
        """Test nonlinear trajectory generation."""
        waypoint = GuidanceWaypoint(
            position=np.array([50.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=50.0,
            phase=GuidancePhase.APPROACH
        )
        guidance_law.add_waypoint(waypoint)
        
        initial_state = RelativeState(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        initial_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        success = guidance_law.generate_trajectory(
            initial_state, initial_attitude, (0.0, 100.0), 1.0
        )
        
        assert success
        assert len(guidance_law.trajectory_states) > 0
        assert len(guidance_law.control_history) > 0
    
    def test_get_control_estimate(self, guidance_law):
        """Test getting control estimate."""
        waypoint = GuidanceWaypoint(
            position=np.array([50.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=50.0,
            phase=GuidancePhase.APPROACH
        )
        guidance_law.add_waypoint(waypoint)
        
        initial_state = RelativeState(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        initial_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        guidance_law.generate_trajectory(
            initial_state, initial_attitude, (0.0, 100.0), 1.0
        )
        
        control = guidance_law.get_control_estimate(25.0)
        
        assert isinstance(control, np.ndarray)
        assert control.shape == (3,)


class TestAdaptiveGuidanceLaw:
    """Test adaptive guidance law."""
    
    @pytest.fixture
    def target_orbit(self):
        """Create test target orbit."""
        return OrbitalElements(
            a=7000e3, e=0.0, i=np.radians(51.6),
            omega_cap=0.0, omega=0.0, f=0.0, mu=EARTH_MU
        )
    
    @pytest.fixture
    def constraints(self):
        """Create test constraints."""
        return GuidanceConstraints()
    
    @pytest.fixture
    def guidance_law(self, target_orbit, constraints):
        """Create test guidance law."""
        return AdaptiveGuidanceLaw(target_orbit, constraints)
    
    def test_initialization(self, guidance_law):
        """Test adaptive guidance law initialization."""
        assert guidance_law.current_mode == 'linear'
        assert guidance_law.adaptation_enabled
        assert len(guidance_law.tracking_errors) == 0
    
    def test_generate_trajectory(self, guidance_law):
        """Test adaptive trajectory generation."""
        waypoint = GuidanceWaypoint(
            position=np.array([50.0, 0.0, 0.0]),
            velocity=np.array([-0.01, 0.0, 0.0]),  # Low velocity to satisfy constraints
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=50.0,
            phase=GuidancePhase.APPROACH
        )
        guidance_law.add_waypoint(waypoint)
        
        initial_state = RelativeState(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        initial_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        success = guidance_law.generate_trajectory(
            initial_state, initial_attitude, (0.0, 100.0), 1.0
        )
        
        assert success
    
    def test_update_performance(self, guidance_law):
        """Test performance update and adaptation."""
        # Generate trajectory first
        waypoint = GuidanceWaypoint(
            position=np.array([50.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            attitude=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=50.0,
            phase=GuidancePhase.APPROACH
        )
        guidance_law.add_waypoint(waypoint)
        
        initial_state = RelativeState(
            position=np.array([100.0, 0.0, 0.0]),
            velocity=np.zeros(3),
            time=0.0
        )
        initial_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=0.0
        )
        
        guidance_law.generate_trajectory(
            initial_state, initial_attitude, (0.0, 100.0), 1.0
        )
        
        # Simulate poor tracking performance
        current_state = RelativeState(
            position=np.array([75.0, 10.0, 5.0]),  # Large error
            velocity=np.array([0.1, 0.1, 0.1]),
            time=25.0
        )
        current_attitude = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            time=25.0
        )
        
        # Update performance multiple times to trigger adaptation
        for i in range(15):
            guidance_law.update_performance(current_state, current_attitude, 25.0 + i)
        
        assert len(guidance_law.tracking_errors) > 0
    
    def test_get_performance_metrics(self, guidance_law):
        """Test getting performance metrics."""
        # Add some tracking errors
        guidance_law.tracking_errors = [0.1, 0.2, 0.15, 0.3, 0.25]
        
        metrics = guidance_law.get_performance_metrics()
        
        assert 'current_error' in metrics
        assert 'mean_error' in metrics
        assert 'max_error' in metrics
        assert 'error_std' in metrics
        assert metrics['current_error'] == 0.25
        assert metrics['max_error'] == 0.3


class TestTrajectoryCreationFunctions:
    """Test trajectory creation utility functions."""
    
    @pytest.fixture
    def target_orbit(self):
        """Create test target orbit."""
        return OrbitalElements(
            a=7000e3, e=0.0, i=np.radians(51.6),
            omega_cap=0.0, omega=0.0, f=0.0, mu=EARTH_MU
        )
    
    @pytest.fixture
    def constraints(self):
        """Create test constraints."""
        return GuidanceConstraints()
    
    def test_create_approach_trajectory(self, target_orbit, constraints):
        """Test approach trajectory creation."""
        waypoints = create_approach_trajectory(
            initial_range=1000.0,
            final_range=50.0,
            approach_velocity=0.1,
            target_orbit=target_orbit,
            constraints=constraints
        )
        
        assert len(waypoints) == 5
        assert waypoints[0].time == 0.0
        assert waypoints[-1].time > waypoints[0].time
        
        # Check ranges decrease
        ranges = [np.linalg.norm(wp.position) for wp in waypoints]
        assert ranges == sorted(ranges, reverse=True)
        
        # Check phases
        assert waypoints[0].phase == GuidancePhase.APPROACH
        assert waypoints[-1].phase == GuidancePhase.DOCKING
    
    def test_create_station_keeping_trajectory(self):
        """Test station keeping trajectory creation."""
        center_position = np.array([100.0, 0.0, 0.0])
        box_size = np.array([10.0, 5.0, 5.0])
        orbit_period = 5400.0  # 90 minutes
        
        waypoints = create_station_keeping_trajectory(
            center_position=center_position,
            box_size=box_size,
            orbit_period=orbit_period,
            num_orbits=1
        )
        
        assert len(waypoints) == 8  # 8 points per orbit
        assert waypoints[0].time == 0.0
        assert waypoints[-1].time == orbit_period
        
        # Check all waypoints are in station keeping phase
        for wp in waypoints:
            assert wp.phase == GuidancePhase.STATION_KEEPING
        
        # Check positions are within box
        for wp in waypoints:
            relative_pos = wp.position - center_position
            assert abs(relative_pos[1]) <= box_size[1] / 2
            assert abs(relative_pos[2]) <= box_size[2] / 2


class TestGuidancePerformanceAnalysis:
    """Test guidance performance analysis."""
    
    def test_analyze_guidance_performance_empty_results(self):
        """Test performance analysis with empty results."""
        target_orbit = OrbitalElements(
            a=7000e3, e=0.0, i=np.radians(51.6),
            omega_cap=0.0, omega=0.0, f=0.0, mu=EARTH_MU
        )
        constraints = GuidanceConstraints()
        guidance_law = LinearGuidanceLaw(target_orbit, constraints)
        
        results = analyze_guidance_performance(guidance_law, {})
        
        assert isinstance(results, dict)
        assert len(results) == 0
    
    def test_analyze_guidance_performance_with_data(self):
        """Test performance analysis with simulation data."""
        target_orbit = OrbitalElements(
            a=7000e3, e=0.0, i=np.radians(51.6),
            omega_cap=0.0, omega=0.0, f=0.0, mu=EARTH_MU
        )
        constraints = GuidanceConstraints()
        guidance_law = LinearGuidanceLaw(target_orbit, constraints)
        
        # Mock reference and actual states
        n_points = 100
        reference_states = []
        actual_states = []
        
        for i in range(n_points):
            ref_state = RelativeState(
                position=np.array([100.0 - i, 0.0, 0.0]),
                velocity=np.array([-0.1, 0.0, 0.0]),
                time=float(i)
            )
            
            # Add some error to actual state
            actual_state = RelativeState(
                position=ref_state.position + np.random.randn(3) * 0.1,
                velocity=ref_state.velocity + np.random.randn(3) * 0.01,
                time=float(i)
            )
            
            reference_states.append(ref_state)
            actual_states.append(actual_state)
        
        simulation_results = {
            'reference_states': reference_states,
            'actual_states': actual_states
        }
        
        metrics = analyze_guidance_performance(guidance_law, simulation_results)
        
        assert 'position_rms_error' in metrics
        assert 'velocity_rms_error' in metrics
        assert 'position_max_error' in metrics
        assert 'velocity_max_error' in metrics
        assert 'trajectory_length' in metrics
        
        assert metrics['position_rms_error'] > 0
        assert metrics['velocity_rms_error'] > 0
        assert metrics['trajectory_length'] == n_points


if __name__ == '__main__':
    pytest.main([__file__])

