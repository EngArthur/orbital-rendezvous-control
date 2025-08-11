"""
Phase 4 Example: Complete Control System Demonstration

This example demonstrates the complete control system implementation including:
- LQR controllers for translational and rotational control
- Guidance laws for trajectory generation
- Actuator models with realistic dynamics
- Integrated control system simulation

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from control.lqr_controller import (
    CoupledLQRController,
    create_default_lqr_weights,
    create_default_control_limits,
    analyze_lqr_performance
)
from control.guidance_laws import (
    AdaptiveGuidanceLaw,
    GuidanceConstraints,
    create_approach_trajectory,
    analyze_guidance_performance
)
from control.actuator_models import (
    create_hybrid_actuator_suite,
    analyze_actuator_performance
)
from dynamics.relative_motion import RelativeState
from dynamics.attitude_dynamics import AttitudeState
from dynamics.orbital_elements import OrbitalElements
from utils.constants import EARTH_MU


def create_iss_orbit():
    """Create ISS-like target orbit."""
    return OrbitalElements(
        a=6.78e6,           # Semi-major axis [m] (408 km altitude)
        e=0.0003,           # Eccentricity
        i=np.radians(51.6), # Inclination [rad]
        omega_cap=0.0,      # RAAN [rad]
        omega=0.0,          # Argument of periapsis [rad]
        f=0.0,              # True anomaly [rad]
        mu=EARTH_MU
    )


def run_rendezvous_simulation():
    """Run complete rendezvous simulation with control system."""
    print("=== Phase 4: Control Systems Demonstration ===\n")
    
    # 1. Setup orbital environment
    print("1. Setting up orbital environment...")
    target_orbit = create_iss_orbit()
    print(f"   Target orbit altitude: {(target_orbit.a - 6.371e6)/1000:.1f} km")
    print(f"   Orbital period: {target_orbit.period/3600:.2f} hours")
    
    # 2. Create control system components
    print("\n2. Creating control system components...")
    
    # LQR Controller
    weights = create_default_lqr_weights()
    limits = create_default_control_limits()
    spacecraft_mass = 500.0  # kg
    spacecraft_inertia = np.diag([100.0, 150.0, 200.0])  # kg⋅m²
    
    controller = CoupledLQRController(
        weights, limits, spacecraft_mass, spacecraft_inertia, coupling_strength=0.1
    )
    controller.set_target_orbit(target_orbit)
    print("   ✓ Coupled LQR controller created")
    
    # Guidance Law
    constraints = GuidanceConstraints(
        max_velocity=0.5,      # m/s
        max_acceleration=0.01, # m/s²
        min_range=10.0         # m
    )
    
    guidance_law = AdaptiveGuidanceLaw(target_orbit, constraints)
    
    # Create approach trajectory
    waypoints = create_approach_trajectory(
        initial_range=1000.0,  # m
        final_range=50.0,      # m
        approach_velocity=0.1, # m/s
        target_orbit=target_orbit,
        constraints=constraints
    )
    
    for waypoint in waypoints:
        guidance_law.add_waypoint(waypoint)
    
    print("   ✓ Adaptive guidance law with approach trajectory created")
    
    # Actuator Suite
    actuator_suite = create_hybrid_actuator_suite()
    print("   ✓ Hybrid actuator suite created (8 thrusters + 3 reaction wheels)")
    
    # 3. Initial conditions
    print("\n3. Setting initial conditions...")
    initial_relative_state = RelativeState(
        position=np.array([1200.0, 100.0, 50.0]),  # m
        velocity=np.array([-0.05, 0.01, 0.005]),   # m/s
        time=0.0
    )
    
    initial_attitude_state = AttitudeState(
        quaternion=np.array([0.95, 0.1, 0.05, 0.3]),  # Slight misalignment
        angular_velocity=np.array([0.001, -0.002, 0.0005]),  # rad/s
        time=0.0
    )
    # Normalize quaternion
    initial_attitude_state.quaternion /= np.linalg.norm(initial_attitude_state.quaternion)
    
    print(f"   Initial range: {np.linalg.norm(initial_relative_state.position):.1f} m")
    print(f"   Initial velocity: {np.linalg.norm(initial_relative_state.velocity):.3f} m/s")
    
    # 4. Generate reference trajectory
    print("\n4. Generating reference trajectory...")
    simulation_time = 12000.0  # seconds (3.33 hours)
    time_step = 10.0           # seconds
    
    success = guidance_law.generate_trajectory(
        initial_relative_state,
        initial_attitude_state,
        (0.0, simulation_time),
        time_step
    )
    
    if success:
        print("   ✓ Reference trajectory generated successfully")
        validation = guidance_law.validate_trajectory()
        print(f"   Trajectory validation: {validation}")
    else:
        print("   ✗ Failed to generate reference trajectory")
        return
    
    # 5. Run closed-loop simulation
    print("\n5. Running closed-loop simulation...")
    
    times = np.arange(0.0, simulation_time + time_step, time_step)
    n_steps = len(times)
    
    # Storage arrays
    relative_states = []
    attitude_states = []
    reference_relative_states = []
    reference_attitude_states = []
    control_forces = []
    control_torques = []
    position_errors = []
    velocity_errors = []
    attitude_errors = []
    
    # Initial states
    current_relative = initial_relative_state
    current_attitude = initial_attitude_state
    
    print(f"   Simulating {n_steps} time steps...")
    
    for i, t in enumerate(times):
        # Get reference states
        ref_relative, ref_attitude = guidance_law.get_reference_state(t)
        
        # Compute control
        force_command, torque_command = controller.compute_control(
            current_relative, ref_relative,
            current_attitude, ref_attitude, t
        )
        
        # Apply actuator dynamics
        actual_force, actual_torque = actuator_suite.allocate_control(
            force_command, torque_command, t
        )
        
        # Simple integration (Euler method for demonstration)
        if i < n_steps - 1:
            dt = times[i+1] - times[i]
            
            # Translational dynamics (simplified)
            acceleration = actual_force / spacecraft_mass
            new_velocity = current_relative.velocity + acceleration * dt
            new_position = current_relative.position + current_relative.velocity * dt
            
            current_relative = RelativeState(
                position=new_position,
                velocity=new_velocity,
                time=t + dt
            )
            
            # Rotational dynamics (simplified)
            angular_acceleration = np.linalg.solve(spacecraft_inertia, actual_torque)
            new_angular_velocity = current_attitude.angular_velocity + angular_acceleration * dt
            
            # Quaternion integration (simplified)
            omega_norm = np.linalg.norm(new_angular_velocity)
            if omega_norm > 1e-12:
                axis = new_angular_velocity / omega_norm
                angle = omega_norm * dt
                dq = np.array([
                    np.cos(angle/2),
                    axis[0] * np.sin(angle/2),
                    axis[1] * np.sin(angle/2),
                    axis[2] * np.sin(angle/2)
                ])
                
                # Quaternion multiplication
                q = current_attitude.quaternion
                new_quaternion = np.array([
                    dq[0]*q[0] - dq[1]*q[1] - dq[2]*q[2] - dq[3]*q[3],
                    dq[0]*q[1] + dq[1]*q[0] + dq[2]*q[3] - dq[3]*q[2],
                    dq[0]*q[2] - dq[1]*q[3] + dq[2]*q[0] + dq[3]*q[1],
                    dq[0]*q[3] + dq[1]*q[2] - dq[2]*q[1] + dq[3]*q[0]
                ])
                new_quaternion /= np.linalg.norm(new_quaternion)
            else:
                new_quaternion = current_attitude.quaternion
            
            current_attitude = AttitudeState(
                quaternion=new_quaternion,
                angular_velocity=new_angular_velocity,
                time=t + dt
            )
        
        # Update adaptive guidance performance
        guidance_law.update_performance(current_relative, current_attitude, t)
        
        # Store data
        relative_states.append(current_relative)
        attitude_states.append(current_attitude)
        reference_relative_states.append(ref_relative)
        reference_attitude_states.append(ref_attitude)
        control_forces.append(force_command)
        control_torques.append(torque_command)
        
        # Compute errors
        pos_error = np.linalg.norm(current_relative.position - ref_relative.position)
        vel_error = np.linalg.norm(current_relative.velocity - ref_relative.velocity)
        
        # Attitude error (simplified)
        q_error = controller.att_controller._quaternion_error(
            current_attitude.quaternion, ref_attitude.quaternion
        )
        att_error = 2 * np.arccos(np.clip(abs(q_error[0]), 0, 1))  # Rotation angle
        
        position_errors.append(pos_error)
        velocity_errors.append(vel_error)
        attitude_errors.append(att_error)
        
        # Progress indicator
        if i % (n_steps // 10) == 0:
            print(f"   Progress: {100*i/n_steps:.0f}% - Range: {np.linalg.norm(current_relative.position):.1f} m")
    
    print("   ✓ Simulation completed")
    
    # 6. Performance analysis
    print("\n6. Analyzing performance...")
    
    # LQR Performance
    simulation_results = {
        'position_errors': np.array(position_errors).reshape(-1, 1),
        'velocity_errors': np.array(velocity_errors).reshape(-1, 1),
        'control_forces': np.array(control_forces),
        'control_torques': np.array(control_torques),
        'dt': time_step,
        'mass': spacecraft_mass
    }
    
    lqr_metrics = analyze_lqr_performance(controller, simulation_results)
    print("   LQR Controller Performance:")
    print(f"     Position RMS error: {lqr_metrics.get('position_rms_error', 0):.2f} m")
    print(f"     Velocity RMS error: {lqr_metrics.get('velocity_rms_error', 0):.4f} m/s")
    print(f"     Total ΔV: {lqr_metrics.get('total_delta_v', 0):.3f} m/s")
    print(f"     Stability margin: {lqr_metrics.get('stability_stability_margin', 0):.3f}")
    
    # Guidance Performance
    guidance_results = {
        'reference_states': reference_relative_states,
        'actual_states': relative_states
    }
    
    guidance_metrics = analyze_guidance_performance(guidance_law, guidance_results)
    print("   Guidance Law Performance:")
    print(f"     Position RMS error: {guidance_metrics.get('position_rms_error', 0):.2f} m")
    print(f"     Final position error: {guidance_metrics.get('position_final_error', 0):.2f} m")
    print(f"     Trajectory length: {guidance_metrics.get('trajectory_length', 0)} points")
    
    adaptive_metrics = guidance_law.get_performance_metrics()
    if adaptive_metrics:
        print(f"     Adaptations: {adaptive_metrics.get('num_adaptations', 0)}")
        print(f"     Mean tracking error: {adaptive_metrics.get('mean_error', 0):.3f}")
    
    # Actuator Performance
    actuator_metrics = analyze_actuator_performance(actuator_suite, {})
    print("   Actuator Suite Performance:")
    print(f"     Total ΔV: {actuator_metrics.get('total_delta_v', 0):.3f} m/s")
    print(f"     Propellant consumed: {actuator_metrics.get('total_propellant_consumed', 0):.3f} kg")
    print(f"     Average power: {actuator_metrics.get('average_power_consumption', 0):.1f} W")
    
    # 7. Create plots
    print("\n7. Creating visualization plots...")
    
    # Convert to arrays for plotting
    times_array = np.array(times)
    positions = np.array([state.position for state in relative_states])
    velocities = np.array([state.velocity for state in relative_states])
    ref_positions = np.array([state.position for state in reference_relative_states])
    forces = np.array(control_forces)
    torques = np.array(control_torques)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Phase 4: Complete Control System Performance', fontsize=16)
    
    # Position tracking
    axes[0, 0].plot(times_array/3600, positions[:, 0], 'b-', label='Actual X')
    axes[0, 0].plot(times_array/3600, ref_positions[:, 0], 'r--', label='Reference X')
    axes[0, 0].set_xlabel('Time [hours]')
    axes[0, 0].set_ylabel('X Position [m]')
    axes[0, 0].set_title('Radial Position Tracking')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Range vs time
    ranges = np.linalg.norm(positions, axis=1)
    axes[0, 1].plot(times_array/3600, ranges, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time [hours]')
    axes[0, 1].set_ylabel('Range [m]')
    axes[0, 1].set_title('Range to Target')
    axes[0, 1].grid(True)
    
    # Velocity tracking
    axes[1, 0].plot(times_array/3600, velocities[:, 0], 'b-', label='Actual Vx')
    axes[1, 0].plot(times_array/3600, [state.velocity[0] for state in reference_relative_states], 'r--', label='Reference Vx')
    axes[1, 0].set_xlabel('Time [hours]')
    axes[1, 0].set_ylabel('X Velocity [m/s]')
    axes[1, 0].set_title('Radial Velocity Tracking')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Control forces
    axes[1, 1].plot(times_array/3600, forces[:, 0], 'r-', label='Fx')
    axes[1, 1].plot(times_array/3600, forces[:, 1], 'g-', label='Fy')
    axes[1, 1].plot(times_array/3600, forces[:, 2], 'b-', label='Fz')
    axes[1, 1].set_xlabel('Time [hours]')
    axes[1, 1].set_ylabel('Control Force [N]')
    axes[1, 1].set_title('Control Forces')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Position errors
    axes[2, 0].semilogy(times_array/3600, position_errors, 'r-', linewidth=2)
    axes[2, 0].set_xlabel('Time [hours]')
    axes[2, 0].set_ylabel('Position Error [m]')
    axes[2, 0].set_title('Position Tracking Error')
    axes[2, 0].grid(True)
    
    # 3D trajectory
    axes[2, 1].plot(positions[:, 0], positions[:, 1], 'b-', label='Actual')
    axes[2, 1].plot(ref_positions[:, 0], ref_positions[:, 1], 'r--', label='Reference')
    axes[2, 1].plot([0], [0], 'ko', markersize=8, label='Target')
    axes[2, 1].set_xlabel('X [m]')
    axes[2, 1].set_ylabel('Y [m]')
    axes[2, 1].set_title('Trajectory (X-Y Plane)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    axes[2, 1].axis('equal')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = 'phase4_control_system_performance.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"   ✓ Performance plots saved as '{plot_filename}'")
    
    # 8. Summary
    print("\n8. Mission Summary:")
    print(f"   Initial range: {np.linalg.norm(initial_relative_state.position):.1f} m")
    print(f"   Final range: {ranges[-1]:.1f} m")
    print(f"   Range reduction: {(np.linalg.norm(initial_relative_state.position) - ranges[-1]):.1f} m")
    print(f"   Final position error: {position_errors[-1]:.2f} m")
    print(f"   Final velocity error: {velocity_errors[-1]:.4f} m/s")
    print(f"   Total fuel consumption: {actuator_metrics.get('total_propellant_consumed', 0):.3f} kg")
    print(f"   Mission duration: {simulation_time/3600:.2f} hours")
    
    # Success criteria
    success_criteria = {
        'final_range': ranges[-1] < 100.0,  # Within 100m
        'final_position_error': position_errors[-1] < 10.0,  # Within 10m error
        'final_velocity_error': velocity_errors[-1] < 0.1,   # Within 0.1 m/s error
        'fuel_consumption': actuator_metrics.get('total_propellant_consumed', 0) < 5.0  # Less than 5kg
    }
    
    print("\n9. Success Criteria:")
    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {criterion}: {status}")
    
    overall_success = all(success_criteria.values())
    print(f"\n   Overall Mission Status: {'✓ SUCCESS' if overall_success else '✗ FAILURE'}")
    
    plt.show()
    
    return {
        'times': times_array,
        'positions': positions,
        'velocities': velocities,
        'reference_positions': ref_positions,
        'control_forces': forces,
        'control_torques': torques,
        'position_errors': position_errors,
        'velocity_errors': velocity_errors,
        'attitude_errors': attitude_errors,
        'lqr_metrics': lqr_metrics,
        'guidance_metrics': guidance_metrics,
        'actuator_metrics': actuator_metrics,
        'success_criteria': success_criteria,
        'overall_success': overall_success
    }


if __name__ == '__main__':
    # Run the complete control system demonstration
    results = run_rendezvous_simulation()
    
    print("\n" + "="*60)
    print("Phase 4 Control Systems demonstration completed!")
    print("This example showcased:")
    print("• Coupled LQR controllers for 6-DOF control")
    print("• Adaptive guidance laws with trajectory generation")
    print("• Realistic actuator models with dynamics")
    print("• Integrated closed-loop simulation")
    print("• Comprehensive performance analysis")
    print("="*60)

