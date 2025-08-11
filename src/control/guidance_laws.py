"""
Guidance Laws for Orbital Rendezvous

This module implements various guidance laws for autonomous orbital rendezvous
operations, including trajectory planning and reference generation.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

from ..dynamics.relative_motion import RelativeState, relative_motion_stm
from ..dynamics.attitude_dynamics import AttitudeState, quaternion_from_euler_angles
from ..dynamics.orbital_elements import OrbitalElements


class GuidancePhase(Enum):
    """Rendezvous guidance phases."""
    APPROACH = "approach"
    PROXIMITY = "proximity"
    DOCKING = "docking"
    STATION_KEEPING = "station_keeping"
    DEPARTURE = "departure"


@dataclass
class GuidanceWaypoint:
    """Waypoint for guidance trajectory."""
    position: np.ndarray        # Position in LVLH frame [m]
    velocity: np.ndarray        # Velocity in LVLH frame [m/s]
    attitude: np.ndarray        # Quaternion [w, x, y, z]
    angular_velocity: np.ndarray # Angular velocity [rad/s]
    time: float                 # Time to reach waypoint [s]
    phase: GuidancePhase        # Guidance phase
    constraints: Dict = None    # Additional constraints


@dataclass
class GuidanceConstraints:
    """Constraints for guidance trajectory generation."""
    max_velocity: float = 1.0           # Maximum approach velocity [m/s]
    max_acceleration: float = 0.1       # Maximum acceleration [m/s²]
    min_range: float = 10.0             # Minimum safe range [m]
    approach_corridor_angle: float = 30.0  # Approach corridor half-angle [deg]
    docking_alignment_tolerance: float = 5.0  # Docking alignment tolerance [deg]
    station_keeping_box: np.ndarray = None  # Station keeping box [±x, ±y, ±z] [m]


class LinearGuidanceLaw:
    """
    Linear guidance law using Clohessy-Wiltshire dynamics.
    
    Generates smooth trajectories between waypoints using the
    state transition matrix solution.
    """
    
    def __init__(self, target_orbit: OrbitalElements, constraints: GuidanceConstraints):
        """
        Initialize linear guidance law.
        
        Args:
            target_orbit: Target orbital elements
            constraints: Guidance constraints
        """
        self.target_orbit = target_orbit
        self.constraints = constraints
        self.mean_motion = target_orbit.mean_motion
        
        # Trajectory storage
        self.waypoints: List[GuidanceWaypoint] = []
        self.trajectory_time: List[float] = []
        self.trajectory_states: List[RelativeState] = []
        self.trajectory_attitudes: List[AttitudeState] = []
    
    def add_waypoint(self, waypoint: GuidanceWaypoint) -> None:
        """Add waypoint to guidance trajectory."""
        self.waypoints.append(waypoint)
        self.waypoints.sort(key=lambda w: w.time)  # Sort by time
    
    def clear_waypoints(self) -> None:
        """Clear all waypoints."""
        self.waypoints.clear()
        self.trajectory_time.clear()
        self.trajectory_states.clear()
        self.trajectory_attitudes.clear()
    
    def generate_trajectory(self, initial_state: RelativeState,
                          initial_attitude: AttitudeState,
                          time_span: Tuple[float, float],
                          time_step: float = 1.0) -> bool:
        """
        Generate complete guidance trajectory.
        
        Args:
            initial_state: Initial relative state
            initial_attitude: Initial attitude state
            time_span: (start_time, end_time) [s]
            time_step: Time step for trajectory [s]
        
        Returns:
            True if trajectory generated successfully
        """
        if len(self.waypoints) == 0:
            warnings.warn("No waypoints defined for trajectory generation")
            return False
        
        start_time, end_time = time_span
        times = np.arange(start_time, end_time + time_step, time_step)
        
        self.trajectory_time = times.tolist()
        self.trajectory_states = []
        self.trajectory_attitudes = []
        
        # Generate trajectory segments between waypoints
        current_state = initial_state
        current_attitude = initial_attitude
        current_time = start_time
        
        waypoint_idx = 0
        
        for t in times:
            # Find current waypoint segment
            while (waypoint_idx < len(self.waypoints) - 1 and 
                   t >= self.waypoints[waypoint_idx + 1].time):
                waypoint_idx += 1
            
            if waypoint_idx < len(self.waypoints):
                # Interpolate to waypoint
                target_waypoint = self.waypoints[waypoint_idx]
                
                if waypoint_idx == 0:
                    # First segment: from initial state to first waypoint
                    start_pos = initial_state.position
                    start_vel = initial_state.velocity
                    start_att = initial_attitude.quaternion
                    start_angvel = initial_attitude.angular_velocity
                    segment_start_time = start_time
                else:
                    # Subsequent segments: from previous waypoint
                    prev_waypoint = self.waypoints[waypoint_idx - 1]
                    start_pos = prev_waypoint.position
                    start_vel = prev_waypoint.velocity
                    start_att = prev_waypoint.attitude
                    start_angvel = prev_waypoint.angular_velocity
                    segment_start_time = prev_waypoint.time
                
                # Time within current segment
                segment_time = t - segment_start_time
                total_segment_time = target_waypoint.time - segment_start_time
                
                if total_segment_time > 0:
                    # Generate state using Clohessy-Wiltshire STM
                    rel_state = self._generate_cw_trajectory(
                        start_pos, start_vel, target_waypoint.position, 
                        target_waypoint.velocity, segment_time, total_segment_time
                    )
                    
                    # Interpolate attitude
                    att_state = self._interpolate_attitude(
                        start_att, start_angvel, target_waypoint.attitude,
                        target_waypoint.angular_velocity, segment_time, total_segment_time
                    )
                else:
                    # At waypoint
                    rel_state = RelativeState(
                        position=target_waypoint.position.copy(),
                        velocity=target_waypoint.velocity.copy(),
                        time=t
                    )
                    att_state = AttitudeState(
                        quaternion=target_waypoint.attitude.copy(),
                        angular_velocity=target_waypoint.angular_velocity.copy(),
                        time=t
                    )
            else:
                # Beyond last waypoint - maintain final state
                final_waypoint = self.waypoints[-1]
                rel_state = RelativeState(
                    position=final_waypoint.position.copy(),
                    velocity=final_waypoint.velocity.copy(),
                    time=t
                )
                att_state = AttitudeState(
                    quaternion=final_waypoint.attitude.copy(),
                    angular_velocity=final_waypoint.angular_velocity.copy(),
                    time=t
                )
            
            self.trajectory_states.append(rel_state)
            self.trajectory_attitudes.append(att_state)
        
        return True
    
    def _generate_cw_trajectory(self, start_pos: np.ndarray, start_vel: np.ndarray,
                               end_pos: np.ndarray, end_vel: np.ndarray,
                               current_time: float, total_time: float) -> RelativeState:
        """Generate trajectory using Clohessy-Wiltshire dynamics."""
        if total_time <= 0:
            return RelativeState(position=start_pos, velocity=start_vel, time=current_time)
        
        # Use STM to propagate from start to current time
        stm = relative_motion_stm(self.mean_motion, current_time)
        
        # Initial state vector
        x0 = np.concatenate([start_pos, start_vel])
        
        # Propagate using STM
        x_current = stm @ x0
        
        # Apply boundary condition correction for smooth trajectory
        if current_time < total_time:
            # Blend with target state based on time ratio
            alpha = current_time / total_time
            # Use smooth interpolation (cubic)
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            
            target_state = np.concatenate([end_pos, end_vel])
            x_current = (1 - alpha_smooth) * x_current + alpha_smooth * target_state
        
        return RelativeState(
            position=x_current[0:3],
            velocity=x_current[3:6],
            time=current_time
        )
    
    def _interpolate_attitude(self, start_quat: np.ndarray, start_angvel: np.ndarray,
                            end_quat: np.ndarray, end_angvel: np.ndarray,
                            current_time: float, total_time: float) -> AttitudeState:
        """Interpolate attitude using SLERP."""
        if total_time <= 0:
            return AttitudeState(quaternion=start_quat, angular_velocity=start_angvel, time=current_time)
        
        # Time ratio
        t = current_time / total_time
        t = np.clip(t, 0.0, 1.0)
        
        # SLERP for quaternion
        dot_product = np.dot(start_quat, end_quat)
        
        # Ensure shortest path
        if dot_product < 0:
            end_quat = -end_quat
            dot_product = -dot_product
        
        # Clamp dot product to avoid numerical issues
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        if dot_product > 0.9995:
            # Linear interpolation for very close quaternions
            result_quat = (1 - t) * start_quat + t * end_quat
            result_quat = result_quat / np.linalg.norm(result_quat)
        else:
            # Spherical linear interpolation
            theta = np.arccos(abs(dot_product))
            sin_theta = np.sin(theta)
            
            w1 = np.sin((1 - t) * theta) / sin_theta
            w2 = np.sin(t * theta) / sin_theta
            
            result_quat = w1 * start_quat + w2 * end_quat
        
        # Linear interpolation for angular velocity
        result_angvel = (1 - t) * start_angvel + t * end_angvel
        
        return AttitudeState(
            quaternion=result_quat,
            angular_velocity=result_angvel,
            time=current_time
        )
    
    def get_reference_state(self, time: float) -> Tuple[RelativeState, AttitudeState]:
        """
        Get reference state at specific time.
        
        Args:
            time: Query time [s]
        
        Returns:
            Tuple of (relative_state, attitude_state)
        """
        if len(self.trajectory_states) == 0:
            raise RuntimeError("No trajectory generated. Call generate_trajectory() first.")
        
        # Find closest time index
        time_array = np.array(self.trajectory_time)
        idx = np.argmin(np.abs(time_array - time))
        
        return self.trajectory_states[idx], self.trajectory_attitudes[idx]
    
    def validate_trajectory(self) -> Dict[str, bool]:
        """
        Validate generated trajectory against constraints.
        
        Returns:
            Dictionary with validation results
        """
        if len(self.trajectory_states) == 0:
            return {'trajectory_exists': False}
        
        results = {'trajectory_exists': True}
        
        # Check velocity constraints
        max_vel = 0.0
        for state in self.trajectory_states:
            vel_mag = np.linalg.norm(state.velocity)
            max_vel = max(max_vel, vel_mag)
        
        results['velocity_constraint'] = max_vel <= self.constraints.max_velocity
        results['max_velocity_achieved'] = max_vel
        
        # Check minimum range constraint
        min_range = float('inf')
        for state in self.trajectory_states:
            range_val = np.linalg.norm(state.position)
            min_range = min(min_range, range_val)
        
        results['range_constraint'] = min_range >= self.constraints.min_range
        results['min_range_achieved'] = min_range
        
        # Check acceleration constraints (approximate)
        max_accel = 0.0
        if len(self.trajectory_states) > 1:
            dt = self.trajectory_time[1] - self.trajectory_time[0]
            for i in range(1, len(self.trajectory_states)):
                dv = self.trajectory_states[i].velocity - self.trajectory_states[i-1].velocity
                accel_mag = np.linalg.norm(dv) / dt
                max_accel = max(max_accel, accel_mag)
        
        results['acceleration_constraint'] = max_accel <= self.constraints.max_acceleration
        results['max_acceleration_achieved'] = max_accel
        
        return results


class NonlinearGuidanceLaw:
    """
    Nonlinear guidance law using numerical optimization.
    
    Generates optimal trajectories considering nonlinear dynamics
    and complex constraints.
    """
    
    def __init__(self, target_orbit: OrbitalElements, constraints: GuidanceConstraints):
        """
        Initialize nonlinear guidance law.
        
        Args:
            target_orbit: Target orbital elements
            constraints: Guidance constraints
        """
        self.target_orbit = target_orbit
        self.constraints = constraints
        self.mean_motion = target_orbit.mean_motion
        
        # Trajectory storage
        self.waypoints: List[GuidanceWaypoint] = []
        self.trajectory_time: List[float] = []
        self.trajectory_states: List[RelativeState] = []
        self.trajectory_attitudes: List[AttitudeState] = []
        self.control_history: List[np.ndarray] = []
    
    def add_waypoint(self, waypoint: GuidanceWaypoint) -> None:
        """Add waypoint to guidance trajectory."""
        self.waypoints.append(waypoint)
        self.waypoints.sort(key=lambda w: w.time)
    
    def generate_trajectory(self, initial_state: RelativeState,
                          initial_attitude: AttitudeState,
                          time_span: Tuple[float, float],
                          time_step: float = 1.0) -> bool:
        """
        Generate optimal trajectory using nonlinear optimization.
        
        Args:
            initial_state: Initial relative state
            initial_attitude: Initial attitude state
            time_span: (start_time, end_time) [s]
            time_step: Time step for trajectory [s]
        
        Returns:
            True if trajectory generated successfully
        """
        # For now, implement a simplified version using multiple shooting
        # In a full implementation, this would use scipy.optimize or similar
        
        start_time, end_time = time_span
        times = np.arange(start_time, end_time + time_step, time_step)
        
        self.trajectory_time = times.tolist()
        self.trajectory_states = []
        self.trajectory_attitudes = []
        self.control_history = []
        
        # Simplified implementation: use linear guidance as initial guess
        # then apply nonlinear corrections
        linear_guidance = LinearGuidanceLaw(self.target_orbit, self.constraints)
        linear_guidance.waypoints = self.waypoints.copy()
        
        if not linear_guidance.generate_trajectory(initial_state, initial_attitude, time_span, time_step):
            return False
        
        # Apply nonlinear corrections
        for i, t in enumerate(times):
            linear_rel, linear_att = linear_guidance.get_reference_state(t)
            
            # Apply nonlinear corrections (simplified)
            corrected_rel = self._apply_nonlinear_corrections(linear_rel, t)
            corrected_att = linear_att  # Simplified: no attitude corrections
            
            self.trajectory_states.append(corrected_rel)
            self.trajectory_attitudes.append(corrected_att)
            
            # Estimate required control (for analysis)
            if i > 0:
                dt = times[i] - times[i-1]
                dv = corrected_rel.velocity - self.trajectory_states[i-1].velocity
                control_accel = dv / dt
                self.control_history.append(control_accel)
            else:
                self.control_history.append(np.zeros(3))
        
        return True
    
    def _apply_nonlinear_corrections(self, linear_state: RelativeState, time: float) -> RelativeState:
        """Apply nonlinear corrections to linear trajectory."""
        # Simplified nonlinear corrections
        # In practice, this would involve solving differential equations
        
        # Example: add small perturbations based on nonlinear terms
        n = self.mean_motion
        x, y, z = linear_state.position
        vx, vy, vz = linear_state.velocity
        
        # Nonlinear corrections (simplified)
        # These would come from higher-order terms in the dynamics
        pos_correction = np.array([
            -0.001 * x * (x**2 + y**2) / 1000**2,  # Small nonlinear position correction
            -0.001 * y * (x**2 + y**2) / 1000**2,
            0.0
        ])
        
        vel_correction = np.array([
            0.0001 * vx * np.linalg.norm(linear_state.position) / 1000,
            0.0001 * vy * np.linalg.norm(linear_state.position) / 1000,
            0.0
        ])
        
        corrected_position = linear_state.position + pos_correction
        corrected_velocity = linear_state.velocity + vel_correction
        
        return RelativeState(
            position=corrected_position,
            velocity=corrected_velocity,
            time=time
        )
    
    def get_reference_state(self, time: float) -> Tuple[RelativeState, AttitudeState]:
        """Get reference state at specific time."""
        if len(self.trajectory_states) == 0:
            raise RuntimeError("No trajectory generated. Call generate_trajectory() first.")
        
        time_array = np.array(self.trajectory_time)
        idx = np.argmin(np.abs(time_array - time))
        
        return self.trajectory_states[idx], self.trajectory_attitudes[idx]
    
    def get_control_estimate(self, time: float) -> np.ndarray:
        """Get estimated control acceleration at specific time."""
        if len(self.control_history) == 0:
            return np.zeros(3)
        
        time_array = np.array(self.trajectory_time)
        idx = np.argmin(np.abs(time_array - time))
        idx = min(idx, len(self.control_history) - 1)
        
        return self.control_history[idx]


class AdaptiveGuidanceLaw:
    """
    Adaptive guidance law that adjusts trajectory based on current performance.
    
    Combines linear and nonlinear guidance with real-time adaptation.
    """
    
    def __init__(self, target_orbit: OrbitalElements, constraints: GuidanceConstraints):
        """
        Initialize adaptive guidance law.
        
        Args:
            target_orbit: Target orbital elements
            constraints: Guidance constraints
        """
        self.target_orbit = target_orbit
        self.constraints = constraints
        
        # Guidance modes
        self.linear_guidance = LinearGuidanceLaw(target_orbit, constraints)
        self.nonlinear_guidance = NonlinearGuidanceLaw(target_orbit, constraints)
        
        # Adaptation parameters
        self.adaptation_enabled = True
        self.performance_threshold = 0.1  # Switch threshold
        self.current_mode = 'linear'      # Current guidance mode
        
        # Performance tracking
        self.tracking_errors: List[float] = []
        self.adaptation_history: List[str] = []
    
    def add_waypoint(self, waypoint: GuidanceWaypoint) -> None:
        """Add waypoint to both guidance laws."""
        self.linear_guidance.add_waypoint(waypoint)
        self.nonlinear_guidance.add_waypoint(waypoint)
    
    def generate_trajectory(self, initial_state: RelativeState,
                          initial_attitude: AttitudeState,
                          time_span: Tuple[float, float],
                          time_step: float = 1.0) -> bool:
        """Generate adaptive trajectory."""
        # Start with linear guidance
        success = self.linear_guidance.generate_trajectory(
            initial_state, initial_attitude, time_span, time_step)
        
        if not success:
            return False
        
        # Validate linear trajectory
        validation = self.linear_guidance.validate_trajectory()
        
        # Switch to nonlinear if needed
        if (not validation.get('velocity_constraint', True) or
            not validation.get('acceleration_constraint', True)):
            
            print("Switching to nonlinear guidance due to constraint violations")
            self.current_mode = 'nonlinear'
            success = self.nonlinear_guidance.generate_trajectory(
                initial_state, initial_attitude, time_span, time_step)
        
        return success
    
    def get_reference_state(self, time: float) -> Tuple[RelativeState, AttitudeState]:
        """Get reference state from current guidance mode."""
        if self.current_mode == 'linear':
            return self.linear_guidance.get_reference_state(time)
        else:
            return self.nonlinear_guidance.get_reference_state(time)
    
    def update_performance(self, current_state: RelativeState,
                         current_attitude: AttitudeState,
                         time: float) -> None:
        """Update performance metrics and adapt if necessary."""
        if not self.adaptation_enabled:
            return
        
        # Get reference state
        ref_rel, ref_att = self.get_reference_state(time)
        
        # Compute tracking error
        pos_error = np.linalg.norm(current_state.position - ref_rel.position)
        vel_error = np.linalg.norm(current_state.velocity - ref_rel.velocity)
        total_error = pos_error + vel_error
        
        self.tracking_errors.append(total_error)
        
        # Adapt if performance is poor
        if len(self.tracking_errors) > 10:  # Need some history
            recent_error = np.mean(self.tracking_errors[-10:])
            
            if (recent_error > self.performance_threshold and 
                self.current_mode == 'linear'):
                
                print(f"Adapting to nonlinear guidance due to tracking error: {recent_error:.3f}")
                self.current_mode = 'nonlinear'
                self.adaptation_history.append(f"t={time:.1f}: linear->nonlinear")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if len(self.tracking_errors) == 0:
            return {}
        
        return {
            'current_error': self.tracking_errors[-1],
            'mean_error': np.mean(self.tracking_errors),
            'max_error': np.max(self.tracking_errors),
            'error_std': np.std(self.tracking_errors),
            'num_adaptations': len(self.adaptation_history)
        }


def create_approach_trajectory(initial_range: float, final_range: float,
                             approach_velocity: float, target_orbit: OrbitalElements,
                             constraints: GuidanceConstraints) -> List[GuidanceWaypoint]:
    """
    Create standard approach trajectory waypoints.
    
    Args:
        initial_range: Initial range from target [m]
        final_range: Final range from target [m]
        approach_velocity: Approach velocity [m/s]
        target_orbit: Target orbital elements
        constraints: Guidance constraints
    
    Returns:
        List of guidance waypoints
    """
    waypoints = []
    
    # Approach phase waypoints
    ranges = np.linspace(initial_range, final_range, 5)
    
    for i, r in enumerate(ranges):
        # Position on approach corridor (radial direction)
        position = np.array([r, 0.0, 0.0])
        
        # Velocity towards target
        if i < len(ranges) - 1:
            velocity = np.array([-approach_velocity, 0.0, 0.0])
        else:
            velocity = np.array([0.0, 0.0, 0.0])  # Stop at final waypoint
        
        # Attitude aligned with target
        attitude = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        angular_velocity = np.zeros(3)
        
        # Time based on constant velocity
        if i == 0:
            time = 0.0
        else:
            time = (initial_range - r) / approach_velocity
        
        # Phase determination
        if r > 100.0:
            phase = GuidancePhase.APPROACH
        elif r > 20.0:
            phase = GuidancePhase.PROXIMITY
        else:
            phase = GuidancePhase.DOCKING
        
        waypoint = GuidanceWaypoint(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity,
            time=time,
            phase=phase
        )
        
        waypoints.append(waypoint)
    
    return waypoints


def create_station_keeping_trajectory(center_position: np.ndarray,
                                    box_size: np.ndarray,
                                    orbit_period: float,
                                    num_orbits: int = 1) -> List[GuidanceWaypoint]:
    """
    Create station keeping trajectory waypoints.
    
    Args:
        center_position: Center position for station keeping [m]
        box_size: Station keeping box size [±x, ±y, ±z] [m]
        orbit_period: Orbital period [s]
        num_orbits: Number of orbits to generate
    
    Returns:
        List of guidance waypoints
    """
    waypoints = []
    
    # Create circular pattern within station keeping box
    num_points = 8 * num_orbits
    times = np.linspace(0, num_orbits * orbit_period, num_points)
    
    for i, t in enumerate(times):
        # Circular motion in Y-Z plane
        angle = 2 * np.pi * t / orbit_period
        
        position = center_position + np.array([
            0.0,
            0.5 * box_size[1] * np.cos(angle),
            0.5 * box_size[2] * np.sin(angle)
        ])
        
        # Velocity for circular motion
        omega = 2 * np.pi / orbit_period
        velocity = np.array([
            0.0,
            -0.5 * box_size[1] * omega * np.sin(angle),
             0.5 * box_size[2] * omega * np.cos(angle)
        ])
        
        # Maintain attitude
        attitude = np.array([1.0, 0.0, 0.0, 0.0])
        angular_velocity = np.zeros(3)
        
        waypoint = GuidanceWaypoint(
            position=position,
            velocity=velocity,
            attitude=attitude,
            angular_velocity=angular_velocity,
            time=t,
            phase=GuidancePhase.STATION_KEEPING
        )
        
        waypoints.append(waypoint)
    
    return waypoints


def analyze_guidance_performance(guidance_law, simulation_results: Dict) -> Dict[str, float]:
    """
    Analyze guidance law performance.
    
    Args:
        guidance_law: Guidance law instance
        simulation_results: Simulation results dictionary
    
    Returns:
        Performance metrics
    """
    if 'reference_states' not in simulation_results or 'actual_states' not in simulation_results:
        return {}
    
    ref_states = simulation_results['reference_states']
    actual_states = simulation_results['actual_states']
    
    if len(ref_states) != len(actual_states):
        return {}
    
    # Compute tracking errors
    position_errors = []
    velocity_errors = []
    
    for ref, actual in zip(ref_states, actual_states):
        pos_error = np.linalg.norm(actual.position - ref.position)
        vel_error = np.linalg.norm(actual.velocity - ref.velocity)
        position_errors.append(pos_error)
        velocity_errors.append(vel_error)
    
    position_errors = np.array(position_errors)
    velocity_errors = np.array(velocity_errors)
    
    # Trajectory validation
    validation = guidance_law.validate_trajectory() if hasattr(guidance_law, 'validate_trajectory') else {}
    
    metrics = {
        'position_rms_error': np.sqrt(np.mean(position_errors**2)),
        'position_max_error': np.max(position_errors),
        'position_final_error': position_errors[-1],
        'velocity_rms_error': np.sqrt(np.mean(velocity_errors**2)),
        'velocity_max_error': np.max(velocity_errors),
        'velocity_final_error': velocity_errors[-1],
        'trajectory_length': len(ref_states)
    }
    
    # Add validation results
    metrics.update(validation)
    
    # Add adaptive guidance metrics if available
    if hasattr(guidance_law, 'get_performance_metrics'):
        adaptive_metrics = guidance_law.get_performance_metrics()
        metrics.update({f'adaptive_{k}': v for k, v in adaptive_metrics.items()})
    
    return metrics

