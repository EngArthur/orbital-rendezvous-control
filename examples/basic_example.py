"""
Basic Example: Orbital Elements and Conversions

This example demonstrates the basic usage of the orbital elements module,
including creation, conversions, and propagation.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import os
import sys

import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import direto dos módulos (sem usar __init__.py)
from dynamics.orbital_elements import (OrbitalElements,
                                       cartesian_to_orbital_elements,
                                       orbital_elements_to_cartesian,
                                       propagate_orbital_elements_mean_motion)
from utils.constants import EARTH_MU, EARTH_RADIUS


def main():
    """Main example function."""
    print("=== Orbital Rendezvous Control System - Basic Example ===\n")
    
    # Create ISS-like orbital elements
    print("1. Creating ISS-like orbital elements:")
    elements = OrbitalElements(
        a=EARTH_RADIUS + 400e3,      # 400 km altitude
        e=0.0001,                    # Nearly circular
        i=np.radians(51.6),          # ISS inclination
        omega_cap=np.radians(45),    # RAAN
        omega=np.radians(30),        # Argument of periapsis
        f=np.radians(0),             # True anomaly
        mu=EARTH_MU                  # Gravitational parameter
    )
    
    print(f"  Semi-major axis: {elements.a/1000:.1f} km")
    print(f"  Eccentricity: {elements.e:.6f}")
    print(f"  Inclination: {np.degrees(elements.i):.1f}°")
    print(f"  Orbital period: {elements.period/3600:.2f} hours")
    print(f"  Mean motion: {np.degrees(elements.mean_motion)*3600:.2f}°/hour")
    
    # Convert to Cartesian coordinates
    print("\n2. Converting to Cartesian coordinates:")
    r_vec, v_vec = orbital_elements_to_cartesian(elements)
    
    print(f"  Position [km]: [{r_vec[0]/1000:.1f}, {r_vec[1]/1000:.1f}, {r_vec[2]/1000:.1f}]")
    print(f"  Velocity [m/s]: [{v_vec[0]:.1f}, {v_vec[1]:.1f}, {v_vec[2]:.1f}]")
    print(f"  Altitude: {(np.linalg.norm(r_vec) - EARTH_RADIUS)/1000:.1f} km")
    print(f"  Speed: {np.linalg.norm(v_vec):.1f} m/s")
    
    # Convert back to orbital elements
    print("\n3. Converting back to orbital elements:")
    recovered = cartesian_to_orbital_elements(r_vec, v_vec, EARTH_MU)
    
    print(f"  Semi-major axis: {recovered.a/1000:.1f} km")
    print(f"  Eccentricity: {recovered.e:.6f}")
    print(f"  Inclination: {np.degrees(recovered.i):.1f}°")
    
    # Check conversion accuracy
    print("\n4. Conversion accuracy:")
    print(f"  Δa: {abs(recovered.a - elements.a):.2e} m")
    print(f"  Δe: {abs(recovered.e - elements.e):.2e}")
    print(f"  Δi: {abs(recovered.i - elements.i):.2e} rad")
    
    # Propagate orbit for different time intervals
    print("\n5. Orbital propagation:")
    time_intervals = [60, 3600, elements.period/4, elements.period]
    
    for dt in time_intervals:
        propagated = propagate_orbital_elements_mean_motion(elements, dt)
        delta_f = np.degrees(propagated.f - elements.f)
        
        if dt < 3600:
            time_str = f"{dt:.0f} seconds"
        elif dt < 86400:
            time_str = f"{dt/3600:.1f} hours"
        else:
            time_str = f"{dt/86400:.1f} days"
            
        print(f"  After {time_str}: Δf = {delta_f:.1f}°")
    
    # Demonstrate orbital properties
    print("\n6. Orbital properties at different true anomalies:")
    true_anomalies = [0, 90, 180, 270]
    
    for f_deg in true_anomalies:
        test_elements = OrbitalElements(
            elements.a, elements.e, elements.i,
            elements.omega_cap, elements.omega, np.radians(f_deg),
            mu=EARTH_MU
        )
        
        r = test_elements.radius()
        v = test_elements.velocity_magnitude()
        altitude = (r - EARTH_RADIUS) / 1000
        
        print(f"  f = {f_deg:3d}°: r = {r/1000:.1f} km, v = {v:.1f} m/s, alt = {altitude:.1f} km")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
