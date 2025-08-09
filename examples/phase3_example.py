"""
Phase 3 Example: Extended Kalman Filter Navigation

This example demonstrates the Extended Kalman Filter implementation for
spacecraft relative navigation using multiple sensor types.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.navigation.extended_kalman_filter import (
    EKFState, ProcessNoise, ExtendedKalmanFilter, SensorType, create_initial_ekf_state
)
from src.navigation.sensor_models import (
    create_typical_sensor_suite, create_high_accuracy_sensor_suite,
    analyze_sensor_performance
)
from src.navigation.navigation_system import (
    NavigationSystem, NavigationConfiguration, create_default_navigation_system,
    run_navigation_simulation
)
from src.dynamics.relative_motion import RelativeState
from src.dynamics.attitude_dynamics import AttitudeState, quaternion_from_euler_angles
from src.dynamics.orbital_elements import OrbitalElements
from src.utils.constants import EARTH_RADIUS


def demonstrate_ekf_basics():
    """Demonstrate basic EKF functionality."""
    print("=== EXTENDED KALMAN FILTER BASICS ===\n")
    
    # Create initial states
    relative_state = RelativeState(
        position=np.array([1000.0, -500.0, 200.0]),  # 1 km radial, 500 m behind, 200 m up
        velocity=np.array([0.1, 0.2, -0.05])         # Small relative velocity
    )
    
    attitude_state = AttitudeState(
        quaternion=quaternion_from_euler_angles(np.radians(5), np.radians(10), np.radians(15)),
        angular_velocity=np.array([0.01, 0.02, 0.03])  # Small angular rates
    )
    
    print("1. Initial State:")
    print(f"   Position (LVLH): [{relative_state.position[0]:.1f}, {relative_state.position[1]:.1f}, {relative_state.position[2]:.1f}] m")
    print(f"   Velocity (LVLH): [{relative_state.velocity[0]:.3f}, {relative_state.velocity[1]:.3f}, {relative_state.velocity[2]:.3f}] m/s")
    print(f"   Range: {relative_state.range:.1f} m")
    print(f"   Quaternion: [{attitude_state.quaternion[0]:.3f}, {attitude_state.quaternion[1]:.3f}, {attitude_state.quaternion[2]:.3f}, {attitude_state.quaternion[3]:.3f}]")
    
    # Create EKF state
    ekf_state = create_initial_ekf_state(relative_state, attitude_state)
    
    # Process noise
    process_noise = ProcessNoise(
        position_noise=1e-6,
        velocity_noise=1e-8,
        attitude_noise=1e-10,
        angular_vel_noise=1e-12
    )
    
    # Create EKF
    ekf = ExtendedKalmanFilter(ekf_state, process_noise)
    
    print(f"\n2. EKF Initialization:")
    print(f"   State vector dimension: {len(ekf.state.state_vector)}")
    print(f"   Covariance matrix size: {ekf.state.covariance.shape}")
    print(f"   Initial position uncertainty (3σ): {ekf.get_position_uncertainty():.2f} m")
    print(f"   Initial velocity uncertainty (3σ): {ekf.get_velocity_uncertainty():.3f} m/s")
    print(f"   Initial attitude uncertainty (3σ): {np.degrees(ekf.get_attitude_uncertainty()):.2f}°")
    
    # Target orbit
    target_elements = OrbitalElements(
        a=EARTH_RADIUS + 400e3,
        e=0.0001,
        i=np.radians(51.6),
        omega_cap=0.0,
        omega=0.0,
        f=0.0
    )
    
    print(f"\n3. Target Orbit:")
    print(f"   Altitude: {(target_elements.a - EARTH_RADIUS)/1000:.1f} km")
    print(f"   Period: {target_elements.period/3600:.2f} hours")
    print(f"   Mean motion: {np.degrees(target_elements.mean_motion)*3600:.2f}°/hour")
    
    # Prediction step
    print(f"\n4. EKF Prediction Step:")
    initial_pos = ekf.state.position.copy()
    initial_time = ekf.state.time
    
    dt = 60.0  # 1 minute
    ekf.predict(target_elements, dt)
    
    print(f"   Time step: {dt:.1f} s")
    print(f"   Position change: [{ekf.state.position[0]-initial_pos[0]:.3f}, {ekf.state.position[1]-initial_pos[1]:.3f}, {ekf.state.position[2]-initial_pos[2]:.3f}] m")
    print(f"   New time: {ekf.state.time:.1f} s")
    print(f"   Covariance trace: {np.trace(ekf.state.covariance):.2e}")


def demonstrate_sensor_models():
    """Demonstrate sensor model functionality."""
    print("\n\n=== SENSOR MODELS DEMONSTRATION ===\n")
    
    # Create sensor suite
    sensor_suite = create_typical_sensor_suite()
    
    print("1. Typical Sensor Suite:")
    status = sensor_suite.get_sensor_status()
    for sensor_id, sensor_status in status.items():
        print(f"   {sensor_id}:")
        print(f"     Update rate: {sensor_status['update_rate']:.1f} Hz")
        print(f"     Enabled: {sensor_status['enabled']}")
    
    # True state for measurements
    relative_state = RelativeState(
        position=np.array([1000.0, -500.0, 200.0]),
        velocity=np.array([0.1, 0.2, -0.05])
    )
    
    attitude_state = AttitudeState(
        quaternion=quaternion_from_euler_angles(np.radians(5), np.radians(10), np.radians(15)),
        angular_velocity=np.array([0.01, 0.02, 0.03])
    )
    
    target_elements = OrbitalElements(
        a=EARTH_RADIUS + 400e3, e=0.0001, i=np.radians(51.6),
        omega_cap=0.0, omega=0.0, f=0.0
    )
    
    true_state = {
        'relative_state': relative_state,
        'attitude_state': attitude_state,
        'target_elements': target_elements
    }
    
    print(f"\n2. Sensor Measurements at t=1.0s:")
    measurements = sensor_suite.generate_measurements(true_state, 1.0)
    
    for measurement in measurements:
        print(f"   {measurement.sensor_type.value}:")
        print(f"     Data: {measurement.data}")
        print(f"     Covariance diagonal: {np.diag(measurement.covariance)}")
        print(f"     Sensor ID: {measurement.sensor_id}")
    
    # High accuracy suite comparison
    print(f"\n3. High Accuracy vs. Typical Sensor Suite:")
    high_acc_suite = create_high_accuracy_sensor_suite()
    
    print(f"   Typical suite sensors: {len(sensor_suite.sensors)}")
    print(f"   High accuracy sensors: {len(high_acc_suite.sensors)}")
    
    # Generate measurements from both
    typical_measurements = sensor_suite.generate_measurements(true_state, 2.0)
    high_acc_measurements = high_acc_suite.generate_measurements(true_state, 2.0)
    
    print(f"   Typical measurements: {len(typical_measurements)}")
    print(f"   High accuracy measurements: {len(high_acc_measurements)}")


def demonstrate_navigation_system():
    """Demonstrate integrated navigation system."""
    print("\n\n=== INTEGRATED NAVIGATION SYSTEM ===\n")
    
    # Initial states
    initial_relative_state = RelativeState(
        position=np.array([1000.0, -500.0, 200.0]),
        velocity=np.array([0.1, 0.2, -0.05])
    )
    
    initial_attitude_state = AttitudeState(
        quaternion=quaternion_from_euler_angles(np.radians(5), np.radians(10), np.radians(15)),
        angular_velocity=np.array([0.01, 0.02, 0.03])
    )
    
    # Create navigation system
    nav_system = create_default_navigation_system(
        initial_relative_state, 
        initial_attitude_state
    )
    
    print("1. Navigation System Configuration:")
    print(f"   Process noise - Position: {nav_system.config.ekf_process_noise.position_noise:.2e}")
    print(f"   Process noise - Velocity: {nav_system.config.ekf_process_noise.velocity_noise:.2e}")
    print(f"   Process noise - Attitude: {nav_system.config.ekf_process_noise.attitude_noise:.2e}")
    print(f"   Innovation threshold: {nav_system.config.innovation_threshold:.1f}")
    print(f"   Outlier detection: {nav_system.config.enable_outlier_detection}")
    
    # Target orbit
    target_elements = OrbitalElements(
        a=EARTH_RADIUS + 400e3, e=0.0001, i=np.radians(51.6),
        omega_cap=0.0, omega=0.0, f=0.0
    )
    
    print(f"\n2. Navigation Update Sequence:")
    
    # Simulate navigation updates
    for i in range(5):
        current_time = i * 10.0  # 10 second intervals
        
        # Simulate true state evolution (simplified)
        true_relative_state = RelativeState(
            position=initial_relative_state.position + initial_relative_state.velocity * current_time,
            velocity=initial_relative_state.velocity
        )
        
        true_attitude_state = AttitudeState(
            quaternion=initial_attitude_state.quaternion,  # Simplified
            angular_velocity=initial_attitude_state.angular_velocity
        )
        
        true_state = {
            'relative_state': true_relative_state,
            'attitude_state': true_attitude_state,
            'target_elements': target_elements
        }
        
        # Update navigation
        estimated_state = nav_system.update(true_state, target_elements, current_time)
        
        # Get current estimates
        est_relative, est_attitude = nav_system.get_current_estimate()
        uncertainties = nav_system.get_uncertainty_estimates()
        
        print(f"   t = {current_time:4.1f}s:")
        print(f"     Est. position: [{est_relative.position[0]:7.1f}, {est_relative.position[1]:7.1f}, {est_relative.position[2]:7.1f}] m")
        print(f"     True position: [{true_relative_state.position[0]:7.1f}, {true_relative_state.position[1]:7.1f}, {true_relative_state.position[2]:7.1f}] m")
        print(f"     Position error: {np.linalg.norm(est_relative.position - true_relative_state.position):.2f} m")
        print(f"     Position uncertainty (3σ): {uncertainties['position_uncertainty_3sigma']:.2f} m")
    
    # Performance analysis
    print(f"\n3. Navigation Performance Analysis:")
    performance = nav_system.analyze_performance()
    
    print(f"   Position RMS error: {performance.position_error_rms:.2f} m")
    print(f"   Velocity RMS error: {performance.velocity_error_rms:.4f} m/s")
    print(f"   Attitude RMS error: {np.degrees(performance.attitude_error_rms):.2f}°")
    print(f"   Filter consistency: {performance.filter_consistency:.2f}")
    print(f"   Measurement rejection rate: {performance.measurement_rejection_rate*100:.1f}%")
    print(f"   Average computation time: {performance.computation_time_avg*1000:.2f} ms")
    
    # Sensor status
    print(f"\n4. Sensor Status:")
    sensor_status = nav_system.get_sensor_status()
    for sensor_id, status in sensor_status.items():
        print(f"   {sensor_id}: {status['measurement_count']} measurements")


def demonstrate_simulation_scenario():
    """Demonstrate complete navigation simulation scenario."""
    print("\n\n=== COMPLETE NAVIGATION SIMULATION ===\n")
    
    # Scenario: Chaser approaching target for rendezvous
    print("Scenario: Spacecraft approaching target for rendezvous")
    print("Initial separation: 2 km, closing at 0.5 m/s")
    
    # Initial states
    initial_relative_state = RelativeState(
        position=np.array([2000.0, 0.0, 0.0]),  # 2 km radial separation
        velocity=np.array([-0.5, 0.0, 0.0])    # Closing at 0.5 m/s
    )
    
    initial_attitude_state = AttitudeState(
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # Aligned
        angular_velocity=np.array([0.0, 0.0, 0.0])   # No rotation
    )
    
    # Create navigation system with high accuracy sensors
    sensor_suite = create_high_accuracy_sensor_suite()
    nav_system = create_default_navigation_system(
        initial_relative_state, 
        initial_attitude_state,
        sensor_suite
    )
    
    # Target orbit
    target_elements = OrbitalElements(
        a=EARTH_RADIUS + 400e3, e=0.0001, i=np.radians(51.6),
        omega_cap=0.0, omega=0.0, f=0.0
    )
    
    # Simulation parameters
    simulation_time = 600.0  # 10 minutes
    time_step = 10.0         # 10 seconds
    time_sequence = np.arange(0, simulation_time + time_step, time_step)
    
    # Generate true state sequence (simplified dynamics)
    true_state_sequence = []
    for t in time_sequence:
        # Simple linear motion for demonstration
        true_pos = initial_relative_state.position + initial_relative_state.velocity * t
        true_vel = initial_relative_state.velocity
        
        true_relative_state = RelativeState(position=true_pos, velocity=true_vel, time=t)
        true_attitude_state = AttitudeState(
            quaternion=initial_attitude_state.quaternion,
            angular_velocity=initial_attitude_state.angular_velocity,
            time=t
        )
        
        true_state_sequence.append({
            'relative_state': true_relative_state,
            'attitude_state': true_attitude_state,
            'target_elements': target_elements
        })
    
    print(f"\n1. Simulation Parameters:")
    print(f"   Duration: {simulation_time:.0f} seconds ({simulation_time/60:.1f} minutes)")
    print(f"   Time step: {time_step:.1f} seconds")
    print(f"   Number of steps: {len(time_sequence)}")
    
    # Run simulation
    print(f"\n2. Running Navigation Simulation...")
    performance = run_navigation_simulation(
        nav_system, true_state_sequence, target_elements, time_sequence
    )
    
    print(f"\n3. Final Performance Results:")
    print(f"   Position RMS error: {performance.position_error_rms:.2f} m")
    print(f"   Velocity RMS error: {performance.velocity_error_rms:.4f} m/s")
    print(f"   Attitude RMS error: {np.degrees(performance.attitude_error_rms):.2f}°")
    print(f"   Filter consistency: {performance.filter_consistency:.2f}")
    print(f"   Measurement rejection rate: {performance.measurement_rejection_rate*100:.1f}%")
    print(f"   Average computation time: {performance.computation_time_avg*1000:.2f} ms")
    
    # Final state comparison
    final_true_state = true_state_sequence[-1]
    final_est_relative, final_est_attitude = nav_system.get_current_estimate()
    
    print(f"\n4. Final State Comparison:")
    print(f"   True final position: [{final_true_state['relative_state'].position[0]:.1f}, {final_true_state['relative_state'].position[1]:.1f}, {final_true_state['relative_state'].position[2]:.1f}] m")
    print(f"   Est. final position: [{final_est_relative.position[0]:.1f}, {final_est_relative.position[1]:.1f}, {final_est_relative.position[2]:.1f}] m")
    
    final_range_true = np.linalg.norm(final_true_state['relative_state'].position)
    final_range_est = np.linalg.norm(final_est_relative.position)
    
    print(f"   True final range: {final_range_true:.1f} m")
    print(f"   Est. final range: {final_range_est:.1f} m")
    print(f"   Range error: {abs(final_range_est - final_range_true):.1f} m")
    
    # Uncertainty estimates
    uncertainties = nav_system.get_uncertainty_estimates()
    print(f"\n5. Final Uncertainty Estimates:")
    print(f"   Position uncertainty (3σ): {uncertainties['position_uncertainty_3sigma']:.2f} m")
    print(f"   Velocity uncertainty (3σ): {uncertainties['velocity_uncertainty_3sigma']:.4f} m/s")
    print(f"   Attitude uncertainty (3σ): {np.degrees(uncertainties['attitude_uncertainty_3sigma']):.2f}°")


def main():
    """Main demonstration function."""
    print("=== ORBITAL RENDEZVOUS CONTROL SYSTEM - PHASE 3 DEMO ===")
    print("Extended Kalman Filter Navigation System")
    
    try:
        demonstrate_ekf_basics()
        demonstrate_sensor_models()
        demonstrate_navigation_system()
        demonstrate_simulation_scenario()
        
        print("\n\n=== PHASE 3 DEMONSTRATION COMPLETED SUCCESSFULLY! ===")
        print("\nPhase 3 Features Demonstrated:")
        print("✓ Extended Kalman Filter implementation")
        print("✓ Multi-sensor fusion (LIDAR, star tracker, gyro, accelerometer, GPS)")
        print("✓ Nonlinear measurement models and Jacobians")
        print("✓ Innovation-based outlier detection")
        print("✓ Integrated navigation system")
        print("✓ Performance analysis and monitoring")
        print("✓ Complete rendezvous simulation")
        print("✓ Uncertainty quantification")
        print("\nNext: Phase 4 will add control systems for autonomous rendezvous!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

