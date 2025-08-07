"""
Phase 2 Example: Perturbations and Attitude Dynamics

This example demonstrates the new capabilities of Phase 2, including
orbital perturbations, attitude dynamics, and coupled motion.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dynamics.orbital_elements import OrbitalElements, orbital_elements_to_cartesian
from src.dynamics.perturbations import (
    SpacecraftProperties, j2_perturbation_acceleration_rsw,
    atmospheric_drag_acceleration_rsw, total_perturbation_acceleration_rsw,
    propagate_orbital_elements_perturbed, perturbation_analysis_summary
)
from src.dynamics.attitude_dynamics import (
    SpacecraftInertia, AttitudeState, quaternion_from_euler_angles,
    euler_angles_from_quaternion, propagate_attitude_state,
    gravity_gradient_torque
)
from src.dynamics.relative_motion import (
    RelativeState, clohessy_wiltshire_dynamics, propagate_relative_state,
    relative_motion_stm
)
from src.utils.constants import EARTH_RADIUS


def demonstrate_perturbations():
    """Demonstrate orbital perturbation effects."""
    print("=== ORBITAL PERTURBATIONS ANALYSIS ===\n")
    
    # Create ISS-like orbit
    elements = OrbitalElements(
        a=EARTH_RADIUS + 400e3,      # 400 km altitude
        e=0.0001,                    # Nearly circular
        i=np.radians(51.6),          # ISS inclination
        omega_cap=np.radians(45),    # RAAN
        omega=np.radians(30),        # Argument of periapsis
        f=np.radians(0)              # True anomaly
    )
    
    # Spacecraft properties (CubeSat-like)
    spacecraft = SpacecraftProperties(
        mass=10.0,                   # 10 kg CubeSat
        drag_area=0.01,              # 10 cm² cross-section
        drag_coefficient=2.2
    )
    
    print("1. Orbital Parameters:")
    print(f"   Altitude: {(elements.a - EARTH_RADIUS)/1000:.1f} km")
    print(f"   Inclination: {np.degrees(elements.i):.1f}°")
    print(f"   Period: {elements.period/3600:.2f} hours")
    
    # Get velocity for drag calculations
    _, v_eci = orbital_elements_to_cartesian(elements)
    
    # Calculate individual perturbations
    a_j2 = j2_perturbation_acceleration_rsw(elements)
    a_drag = atmospheric_drag_acceleration_rsw(elements, v_eci, spacecraft)
    a_total = total_perturbation_acceleration_rsw(elements, v_eci, spacecraft)
    
    print("\n2. Perturbation Accelerations (RSW frame):")
    print(f"   J2 perturbation:   [{a_j2[0]:.2e}, {a_j2[1]:.2e}, {a_j2[2]:.2e}] m/s²")
    print(f"   Drag perturbation: [{a_drag[0]:.2e}, {a_drag[1]:.2e}, {a_drag[2]:.2e}] m/s²")
    print(f"   Total perturbation:[{a_total[0]:.2e}, {a_total[1]:.2e}, {a_total[2]:.2e}] m/s²")
    
    print(f"\n   J2 magnitude:   {np.linalg.norm(a_j2):.2e} m/s²")
    print(f"   Drag magnitude: {np.linalg.norm(a_drag):.2e} m/s²")
    print(f"   Ratio (Drag/J2): {np.linalg.norm(a_drag)/np.linalg.norm(a_j2):.2f}")
    
    # Perturbation analysis
    analysis = perturbation_analysis_summary(elements, spacecraft)
    print(f"\n3. Perturbation Analysis:")
    print(f"   Atmospheric density: {analysis['atmospheric_density_kg_m3']:.2e} kg/m³")
    print(f"   Dominant perturbation: {analysis['dominant_perturbation'].upper()}")
    print(f"   Estimated decay time: {analysis['estimated_decay_time_days']:.1f} days")
    
    # Propagate with perturbations
    print("\n4. Orbital Evolution with Perturbations:")
    current_elements = elements
    time_step = 3600.0  # 1 hour
    
    for i in range(5):  # 5 hours
        _, v_eci = orbital_elements_to_cartesian(current_elements)
        new_elements = propagate_orbital_elements_perturbed(
            current_elements, v_eci, spacecraft, time_step
        )
        
        delta_a = (new_elements.a - elements.a) / 1000  # km
        delta_e = new_elements.e - elements.e
        
        print(f"   After {i+1:2d} hours: Δa = {delta_a:+.3f} km, Δe = {delta_e:+.6f}")
        current_elements = new_elements


def demonstrate_attitude_dynamics():
    """Demonstrate attitude dynamics and control."""
    print("\n\n=== ATTITUDE DYNAMICS DEMONSTRATION ===\n")
    
    # Spacecraft inertia (small satellite)
    inertia = SpacecraftInertia(
        Ixx=0.1,    # kg⋅m²
        Iyy=0.15,   # kg⋅m²
        Izz=0.2     # kg⋅m²
    )
    
    print("1. Spacecraft Inertia Properties:")
    print(f"   Ixx = {inertia.Ixx:.3f} kg⋅m²")
    print(f"   Iyy = {inertia.Iyy:.3f} kg⋅m²")
    print(f"   Izz = {inertia.Izz:.3f} kg⋅m²")
    print(f"   Principal axes aligned: {inertia.is_principal_axes()}")
    
    # Initial attitude state
    initial_euler = [np.radians(10), np.radians(20), np.radians(30)]  # Roll, pitch, yaw
    q0 = quaternion_from_euler_angles(*initial_euler)
    omega0 = np.array([0.01, 0.02, 0.03])  # rad/s
    
    attitude_state = AttitudeState(q0, omega0)
    
    print(f"\n2. Initial Attitude State:")
    print(f"   Euler angles: [{np.degrees(initial_euler[0]):.1f}°, {np.degrees(initial_euler[1]):.1f}°, {np.degrees(initial_euler[2]):.1f}°]")
    print(f"   Quaternion: [{q0[0]:.3f}, {q0[1]:.3f}, {q0[2]:.3f}, {q0[3]:.3f}]")
    print(f"   Angular velocity: [{omega0[0]:.3f}, {omega0[1]:.3f}, {omega0[2]:.3f}] rad/s")
    
    # Simulate attitude motion with gravity gradient torque
    print(f"\n3. Attitude Evolution (with gravity gradient torque):")
    
    # Orbital position for gravity gradient calculation
    orbital_elements = OrbitalElements(
        a=EARTH_RADIUS + 400e3, e=0.0, i=0.0, omega_cap=0.0, omega=0.0, f=0.0
    )
    r_eci, _ = orbital_elements_to_cartesian(orbital_elements)
    
    current_state = attitude_state
    dt = 10.0  # 10 seconds
    
    for i in range(6):  # 1 minute total
        # Calculate gravity gradient torque
        gg_torque = gravity_gradient_torque(current_state.quaternion, r_eci, inertia)
        
        # Propagate attitude
        new_state = propagate_attitude_state(current_state, inertia, gg_torque, dt, 'rk4')
        
        # Convert to Euler angles for display
        roll, pitch, yaw = euler_angles_from_quaternion(new_state.quaternion)
        
        print(f"   t = {(i+1)*dt:3.0f}s: Euler = [{np.degrees(roll):6.2f}°, {np.degrees(pitch):6.2f}°, {np.degrees(yaw):6.2f}°]")
        print(f"            ω = [{new_state.angular_velocity[0]:6.4f}, {new_state.angular_velocity[1]:6.4f}, {new_state.angular_velocity[2]:6.4f}] rad/s")
        
        current_state = new_state


def demonstrate_relative_motion():
    """Demonstrate relative motion dynamics."""
    print("\n\n=== RELATIVE MOTION DYNAMICS ===\n")
    
    # Target orbit (ISS-like)
    target_elements = OrbitalElements(
        a=EARTH_RADIUS + 400e3,
        e=0.0001,
        i=np.radians(51.6),
        omega_cap=0.0,
        omega=0.0,
        f=0.0
    )
    
    print("1. Target Orbital Parameters:")
    print(f"   Altitude: {(target_elements.a - EARTH_RADIUS)/1000:.1f} km")
    print(f"   Period: {target_elements.period/3600:.2f} hours")
    print(f"   Mean motion: {np.degrees(target_elements.mean_motion)*3600:.2f}°/hour")
    
    # Initial relative state (chaser 1 km behind target)
    relative_state = RelativeState(
        position=np.array([0.0, -1000.0, 0.0]),  # 1 km behind in along-track
        velocity=np.array([0.0, 0.0, 0.0])       # Initially at rest relative to target
    )
    
    print(f"\n2. Initial Relative State:")
    print(f"   Position (LVLH): [{relative_state.position[0]:.1f}, {relative_state.position[1]:.1f}, {relative_state.position[2]:.1f}] m")
    print(f"   Velocity (LVLH): [{relative_state.velocity[0]:.3f}, {relative_state.velocity[1]:.3f}, {relative_state.velocity[2]:.3f}] m/s")
    print(f"   Range: {relative_state.range:.1f} m")
    
    # Propagate relative motion using Clohessy-Wiltshire dynamics
    print(f"\n3. Relative Motion Evolution (Clohessy-Wiltshire):")
    
    current_state = relative_state
    dt = 600.0  # 10 minutes
    
    for i in range(6):  # 1 hour total
        new_state = propagate_relative_state(current_state, target_elements, dt)
        
        print(f"   t = {(i+1)*dt/60:3.0f}min: pos = [{new_state.position[0]:7.1f}, {new_state.position[1]:7.1f}, {new_state.position[2]:7.1f}] m")
        print(f"              vel = [{new_state.velocity[0]:6.3f}, {new_state.velocity[1]:6.3f}, {new_state.velocity[2]:6.3f}] m/s")
        print(f"              range = {new_state.range:.1f} m")
        
        current_state = new_state
    
    # Demonstrate state transition matrix
    print(f"\n4. State Transition Matrix (30 minutes):")
    stm = relative_motion_stm(target_elements, 1800.0)  # 30 minutes
    
    print("   STM diagonal elements:")
    for i in range(6):
        print(f"   STM[{i},{i}] = {stm[i,i]:8.4f}")


def main():
    """Main demonstration function."""
    print("=== ORBITAL RENDEZVOUS CONTROL SYSTEM - PHASE 2 DEMO ===")
    
    try:
        demonstrate_perturbations()
        demonstrate_attitude_dynamics()
        demonstrate_relative_motion()
        
        print("\n\n=== PHASE 2 DEMONSTRATION COMPLETED SUCCESSFULLY! ===")
        print("\nPhase 2 Features Demonstrated:")
        print("✓ J2 gravitational perturbations")
        print("✓ Atmospheric drag modeling")
        print("✓ Orbital decay analysis")
        print("✓ Quaternion-based attitude dynamics")
        print("✓ Gravity gradient torques")
        print("✓ Attitude propagation with RK4 integration")
        print("✓ Clohessy-Wiltshire relative motion")
        print("✓ State transition matrix calculations")
        print("\nNext: Phase 3 will add Extended Kalman Filter navigation!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

