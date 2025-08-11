"""
Actuator Models for Spacecraft Control

This module implements realistic models for spacecraft actuators including
thrusters, reaction wheels, and magnetic torquers.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


class ActuatorType(Enum):
    """Types of spacecraft actuators."""
    THRUSTER = "thruster"
    REACTION_WHEEL = "reaction_wheel"
    MAGNETIC_TORQUER = "magnetic_torquer"
    CONTROL_MOMENT_GYRO = "cmg"


@dataclass
class ActuatorConfiguration:
    """Configuration parameters for actuators."""
    max_force: float = 1.0          # Maximum force/torque [N or N⋅m]
    min_force: float = 0.0          # Minimum force/torque [N or N⋅m]
    response_time: float = 0.1      # Response time constant [s]
    noise_std: float = 0.01         # Noise standard deviation
    bias: float = 0.0               # Constant bias
    efficiency: float = 1.0         # Efficiency factor (0-1)
    power_consumption: float = 10.0 # Power consumption [W]
    enabled: bool = True            # Actuator enabled flag


@dataclass
class ThrusterProperties:
    """Specific properties for thrusters."""
    specific_impulse: float = 220.0     # Specific impulse [s]
    minimum_impulse_bit: float = 1e-6  # Minimum impulse bit [N⋅s]
    duty_cycle_limit: float = 1.0      # Maximum duty cycle (0-1)
    thermal_time_constant: float = 60.0 # Thermal time constant [s]
    propellant_mass: float = 1.0       # Available propellant [kg]


@dataclass
class ReactionWheelProperties:
    """Specific properties for reaction wheels."""
    max_angular_momentum: float = 1.0   # Maximum angular momentum [N⋅m⋅s]
    max_angular_velocity: float = 6000.0 # Maximum angular velocity [rpm]
    wheel_inertia: float = 0.01         # Wheel moment of inertia [kg⋅m²]
    friction_coefficient: float = 1e-6  # Friction coefficient [N⋅m⋅s/rad]
    back_emf_constant: float = 0.01     # Back EMF constant [V⋅s/rad]


class ThrusterModel:
    """
    Model for spacecraft thrusters.
    
    Includes realistic effects like minimum impulse bit, thermal effects,
    and propellant consumption.
    """
    
    def __init__(self, config: ActuatorConfiguration, 
                 properties: ThrusterProperties,
                 position: np.ndarray, direction: np.ndarray,
                 actuator_id: str = "thruster"):
        """
        Initialize thruster model.
        
        Args:
            config: Actuator configuration
            properties: Thruster-specific properties
            position: Thruster position in body frame [m]
            direction: Thrust direction unit vector in body frame
            actuator_id: Unique actuator identifier
        """
        self.config = config
        self.properties = properties
        self.position = position / np.linalg.norm(position) if np.linalg.norm(position) > 0 else position
        self.direction = direction / np.linalg.norm(direction)
        self.actuator_id = actuator_id
        
        # State variables
        self.current_thrust = 0.0
        self.temperature = 20.0  # Temperature [°C]
        self.propellant_consumed = 0.0  # Consumed propellant [kg]
        self.total_impulse = 0.0  # Total impulse delivered [N⋅s]
        self.duty_cycle_history = []
        self.last_update_time = 0.0
        
        # Performance tracking
        self.thrust_history = []
        self.command_history = []
        self.efficiency_history = []
    
    def compute_thrust(self, thrust_command: float, current_time: float) -> float:
        """
        Compute actual thrust output.
        
        Args:
            thrust_command: Commanded thrust [N]
            current_time: Current time [s]
        
        Returns:
            Actual thrust output [N]
        """
        dt = current_time - self.last_update_time if current_time > self.last_update_time else 0.01
        self.last_update_time = current_time
        
        # Apply limits
        thrust_command = np.clip(thrust_command, self.config.min_force, self.config.max_force)
        
        # Check propellant availability
        if self.propellant_consumed >= self.properties.propellant_mass:
            warnings.warn(f"Thruster {self.actuator_id} out of propellant")
            return 0.0
        
        # Apply minimum impulse bit
        if abs(thrust_command) < self.properties.minimum_impulse_bit / dt:
            thrust_command = 0.0
        
        # First-order response dynamics
        tau = self.config.response_time
        alpha = dt / (tau + dt)
        self.current_thrust = (1 - alpha) * self.current_thrust + alpha * thrust_command
        
        # Apply efficiency and bias
        actual_thrust = self.current_thrust * self.config.efficiency + self.config.bias
        
        # Add noise
        if self.config.noise_std > 0:
            noise = np.random.normal(0, self.config.noise_std)
            actual_thrust += noise
        
        # Thermal effects (simplified)
        self._update_thermal_state(abs(actual_thrust), dt)
        thermal_factor = 1.0 - 0.1 * max(0, (self.temperature - 50) / 100)  # Degrade at high temp
        actual_thrust *= thermal_factor
        
        # Update propellant consumption
        if actual_thrust > 0:
            mass_flow_rate = actual_thrust / (self.properties.specific_impulse * 9.81)
            self.propellant_consumed += mass_flow_rate * dt
        
        # Update total impulse
        self.total_impulse += abs(actual_thrust) * dt
        
        # Store history
        self.thrust_history.append(actual_thrust)
        self.command_history.append(thrust_command)
        self.efficiency_history.append(thermal_factor * self.config.efficiency)
        
        return actual_thrust
    
    def _update_thermal_state(self, thrust_magnitude: float, dt: float) -> None:
        """Update thruster thermal state."""
        # Heat generation proportional to thrust
        heat_generation = thrust_magnitude * 0.1  # Simplified heat model
        
        # Thermal dynamics (first-order cooling)
        tau_thermal = self.properties.thermal_time_constant
        ambient_temp = 20.0
        
        # Temperature update
        alpha_thermal = dt / (tau_thermal + dt)
        equilibrium_temp = ambient_temp + heat_generation * 10  # Simplified
        self.temperature = (1 - alpha_thermal) * self.temperature + alpha_thermal * equilibrium_temp
    
    def get_force_torque(self, thrust: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get force and torque vectors in body frame.
        
        Args:
            thrust: Thrust magnitude [N]
        
        Returns:
            Tuple of (force_vector, torque_vector)
        """
        force = thrust * self.direction
        torque = np.cross(self.position, force)
        
        return force, torque
    
    def get_status(self) -> Dict:
        """Get thruster status information."""
        return {
            'actuator_id': self.actuator_id,
            'enabled': self.config.enabled,
            'current_thrust': self.current_thrust,
            'temperature': self.temperature,
            'propellant_remaining': self.properties.propellant_mass - self.propellant_consumed,
            'propellant_fraction': (self.properties.propellant_mass - self.propellant_consumed) / self.properties.propellant_mass,
            'total_impulse': self.total_impulse,
            'efficiency': self.efficiency_history[-1] if self.efficiency_history else self.config.efficiency
        }
    
    def reset(self) -> None:
        """Reset thruster to initial state."""
        self.current_thrust = 0.0
        self.temperature = 20.0
        self.propellant_consumed = 0.0
        self.total_impulse = 0.0
        self.thrust_history.clear()
        self.command_history.clear()
        self.efficiency_history.clear()


class ReactionWheelModel:
    """
    Model for reaction wheels.
    
    Includes saturation, friction, and back-EMF effects.
    """
    
    def __init__(self, config: ActuatorConfiguration,
                 properties: ReactionWheelProperties,
                 axis: np.ndarray, actuator_id: str = "reaction_wheel"):
        """
        Initialize reaction wheel model.
        
        Args:
            config: Actuator configuration
            properties: Reaction wheel properties
            axis: Wheel spin axis unit vector in body frame
            actuator_id: Unique actuator identifier
        """
        self.config = config
        self.properties = properties
        self.axis = axis / np.linalg.norm(axis)
        self.actuator_id = actuator_id
        
        # State variables
        self.angular_velocity = 0.0  # Wheel angular velocity [rad/s]
        self.angular_momentum = 0.0  # Stored angular momentum [N⋅m⋅s]
        self.motor_torque = 0.0      # Current motor torque [N⋅m]
        self.last_update_time = 0.0
        
        # Performance tracking
        self.torque_history = []
        self.speed_history = []
        self.momentum_history = []
    
    def compute_torque(self, torque_command: float, current_time: float) -> float:
        """
        Compute actual torque output.
        
        Args:
            torque_command: Commanded torque [N⋅m]
            current_time: Current time [s]
        
        Returns:
            Actual torque output [N⋅m]
        """
        dt = current_time - self.last_update_time if current_time > self.last_update_time else 0.01
        self.last_update_time = current_time
        
        # Apply limits
        torque_command = np.clip(torque_command, -self.config.max_force, self.config.max_force)
        
        # Check saturation
        max_momentum = self.properties.max_angular_momentum
        if abs(self.angular_momentum) >= max_momentum:
            # Saturated - can only provide torque to desaturate
            if np.sign(torque_command) == np.sign(self.angular_momentum):
                torque_command = 0.0  # Cannot increase momentum further
        
        # First-order response dynamics
        tau = self.config.response_time
        alpha = dt / (tau + dt)
        self.motor_torque = (1 - alpha) * self.motor_torque + alpha * torque_command
        
        # Friction torque
        friction_torque = -self.properties.friction_coefficient * self.angular_velocity
        
        # Net torque on wheel
        net_wheel_torque = self.motor_torque + friction_torque
        
        # Update wheel state
        wheel_acceleration = net_wheel_torque / self.properties.wheel_inertia
        self.angular_velocity += wheel_acceleration * dt
        
        # Limit wheel speed
        max_speed = self.properties.max_angular_velocity * 2 * np.pi / 60  # Convert rpm to rad/s
        self.angular_velocity = np.clip(self.angular_velocity, -max_speed, max_speed)
        
        # Update angular momentum
        self.angular_momentum = self.properties.wheel_inertia * self.angular_velocity
        
        # Reaction torque on spacecraft (Newton's 3rd law)
        spacecraft_torque = -self.motor_torque
        
        # Apply efficiency and bias
        actual_torque = spacecraft_torque * self.config.efficiency + self.config.bias
        
        # Add noise
        if self.config.noise_std > 0:
            noise = np.random.normal(0, self.config.noise_std)
            actual_torque += noise
        
        # Store history
        self.torque_history.append(actual_torque)
        self.speed_history.append(self.angular_velocity)
        self.momentum_history.append(self.angular_momentum)
        
        return actual_torque
    
    def get_torque_vector(self, torque: float) -> np.ndarray:
        """
        Get torque vector in body frame.
        
        Args:
            torque: Torque magnitude [N⋅m]
        
        Returns:
            Torque vector in body frame
        """
        return torque * self.axis
    
    def get_status(self) -> Dict:
        """Get reaction wheel status information."""
        max_momentum = self.properties.max_angular_momentum
        max_speed_rpm = self.properties.max_angular_velocity
        current_speed_rpm = self.angular_velocity * 60 / (2 * np.pi)
        
        return {
            'actuator_id': self.actuator_id,
            'enabled': self.config.enabled,
            'angular_velocity_rpm': current_speed_rpm,
            'angular_momentum': self.angular_momentum,
            'saturation_level': abs(self.angular_momentum) / max_momentum,
            'speed_fraction': abs(current_speed_rpm) / max_speed_rpm,
            'motor_torque': self.motor_torque,
            'is_saturated': abs(self.angular_momentum) >= 0.95 * max_momentum
        }
    
    def desaturate(self, external_torque: float, dt: float) -> None:
        """
        Desaturate wheel using external torque.
        
        Args:
            external_torque: External torque for desaturation [N⋅m]
            dt: Time step [s]
        """
        # Apply external torque to reduce wheel momentum
        momentum_change = -external_torque * dt
        self.angular_momentum += momentum_change
        self.angular_velocity = self.angular_momentum / self.properties.wheel_inertia
    
    def reset(self) -> None:
        """Reset reaction wheel to initial state."""
        self.angular_velocity = 0.0
        self.angular_momentum = 0.0
        self.motor_torque = 0.0
        self.torque_history.clear()
        self.speed_history.clear()
        self.momentum_history.clear()


class ActuatorSuite:
    """
    Suite of actuators for spacecraft control.
    
    Manages multiple actuators and provides unified interface.
    """
    
    def __init__(self):
        """Initialize actuator suite."""
        self.thrusters: Dict[str, ThrusterModel] = {}
        self.reaction_wheels: Dict[str, ReactionWheelModel] = {}
        self.actuator_count = 0
        
        # Control allocation
        self.thruster_allocation_matrix = None
        self.wheel_allocation_matrix = None
        
        # Performance tracking
        self.total_delta_v = 0.0
        self.total_propellant_consumed = 0.0
        self.power_consumption_history = []
    
    def add_thruster(self, thruster: ThrusterModel) -> None:
        """Add thruster to suite."""
        self.thrusters[thruster.actuator_id] = thruster
        self.actuator_count += 1
        self._update_allocation_matrices()
    
    def add_reaction_wheel(self, wheel: ReactionWheelModel) -> None:
        """Add reaction wheel to suite."""
        self.reaction_wheels[wheel.actuator_id] = wheel
        self.actuator_count += 1
        self._update_allocation_matrices()
    
    def _update_allocation_matrices(self) -> None:
        """Update control allocation matrices."""
        # Thruster allocation matrix (6 x N_thrusters)
        if self.thrusters:
            n_thrusters = len(self.thrusters)
            self.thruster_allocation_matrix = np.zeros((6, n_thrusters))
            
            for i, thruster in enumerate(self.thrusters.values()):
                force, torque = thruster.get_force_torque(1.0)  # Unit thrust
                self.thruster_allocation_matrix[0:3, i] = force
                self.thruster_allocation_matrix[3:6, i] = torque
        
        # Reaction wheel allocation matrix (3 x N_wheels)
        if self.reaction_wheels:
            n_wheels = len(self.reaction_wheels)
            self.wheel_allocation_matrix = np.zeros((3, n_wheels))
            
            for i, wheel in enumerate(self.reaction_wheels.values()):
                torque_vector = wheel.get_torque_vector(1.0)  # Unit torque
                self.wheel_allocation_matrix[:, i] = torque_vector
    
    def allocate_control(self, desired_force: np.ndarray, 
                        desired_torque: np.ndarray,
                        current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Allocate control commands to actuators.
        
        Args:
            desired_force: Desired force vector [N]
            desired_torque: Desired torque vector [N⋅m]
            current_time: Current time [s]
        
        Returns:
            Tuple of (actual_force, actual_torque)
        """
        actual_force = np.zeros(3)
        actual_torque = np.zeros(3)
        
        # Allocate force to thrusters
        if self.thrusters and self.thruster_allocation_matrix is not None:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            
            # Pseudo-inverse allocation (simplified)
            A = self.thruster_allocation_matrix
            try:
                thruster_commands = np.linalg.pinv(A) @ desired_wrench
                
                # Apply thruster commands
                for i, (thruster_id, thruster) in enumerate(self.thrusters.items()):
                    if thruster.config.enabled:
                        actual_thrust = thruster.compute_thrust(thruster_commands[i], current_time)
                        force, torque = thruster.get_force_torque(actual_thrust)
                        actual_force += force
                        actual_torque += torque
                        
            except np.linalg.LinAlgError:
                warnings.warn("Thruster allocation matrix is singular")
        
        # Allocate remaining torque to reaction wheels
        remaining_torque = desired_torque - actual_torque
        
        if self.reaction_wheels and self.wheel_allocation_matrix is not None:
            try:
                wheel_commands = np.linalg.pinv(self.wheel_allocation_matrix) @ remaining_torque
                
                # Apply wheel commands
                for i, (wheel_id, wheel) in enumerate(self.reaction_wheels.items()):
                    if wheel.config.enabled:
                        actual_wheel_torque = wheel.compute_torque(wheel_commands[i], current_time)
                        torque_vector = wheel.get_torque_vector(actual_wheel_torque)
                        actual_torque += torque_vector
                        
            except np.linalg.LinAlgError:
                warnings.warn("Reaction wheel allocation matrix is singular")
        
        # Update performance metrics
        self._update_performance_metrics(current_time)
        
        return actual_force, actual_torque
    
    def _update_performance_metrics(self, current_time: float) -> None:
        """Update suite performance metrics."""
        # Total propellant consumption
        total_propellant = sum(t.propellant_consumed for t in self.thrusters.values())
        self.total_propellant_consumed = total_propellant
        
        # Total delta-V (approximate)
        if self.thrusters:
            total_impulse = sum(t.total_impulse for t in self.thrusters.values())
            # Assuming constant mass (simplified)
            spacecraft_mass = 100.0  # kg (should be parameter)
            self.total_delta_v = total_impulse / spacecraft_mass
        
        # Power consumption
        total_power = 0.0
        for thruster in self.thrusters.values():
            if thruster.config.enabled and thruster.current_thrust > 0:
                total_power += thruster.config.power_consumption
        
        for wheel in self.reaction_wheels.values():
            if wheel.config.enabled:
                total_power += wheel.config.power_consumption
        
        self.power_consumption_history.append(total_power)
    
    def get_suite_status(self) -> Dict:
        """Get status of entire actuator suite."""
        status = {
            'num_thrusters': len(self.thrusters),
            'num_reaction_wheels': len(self.reaction_wheels),
            'total_propellant_consumed': self.total_propellant_consumed,
            'total_delta_v': self.total_delta_v,
            'current_power_consumption': self.power_consumption_history[-1] if self.power_consumption_history else 0.0,
            'thrusters': {},
            'reaction_wheels': {}
        }
        
        # Individual actuator status
        for thruster_id, thruster in self.thrusters.items():
            status['thrusters'][thruster_id] = thruster.get_status()
        
        for wheel_id, wheel in self.reaction_wheels.items():
            status['reaction_wheels'][wheel_id] = wheel.get_status()
        
        return status
    
    def check_actuator_health(self) -> Dict[str, bool]:
        """Check health status of all actuators."""
        health = {}
        
        for thruster_id, thruster in self.thrusters.items():
            status = thruster.get_status()
            health[thruster_id] = (
                status['enabled'] and
                status['propellant_fraction'] > 0.05 and  # At least 5% propellant
                status['temperature'] < 100.0  # Not overheated
            )
        
        for wheel_id, wheel in self.reaction_wheels.items():
            status = wheel.get_status()
            health[wheel_id] = (
                status['enabled'] and
                not status['is_saturated']  # Not saturated
            )
        
        return health
    
    def enable_actuator(self, actuator_id: str) -> bool:
        """Enable specific actuator."""
        if actuator_id in self.thrusters:
            self.thrusters[actuator_id].config.enabled = True
            return True
        elif actuator_id in self.reaction_wheels:
            self.reaction_wheels[actuator_id].config.enabled = True
            return True
        return False
    
    def disable_actuator(self, actuator_id: str) -> bool:
        """Disable specific actuator."""
        if actuator_id in self.thrusters:
            self.thrusters[actuator_id].config.enabled = False
            return True
        elif actuator_id in self.reaction_wheels:
            self.reaction_wheels[actuator_id].config.enabled = False
            return True
        return False
    
    def reset_all_actuators(self) -> None:
        """Reset all actuators to initial state."""
        for thruster in self.thrusters.values():
            thruster.reset()
        
        for wheel in self.reaction_wheels.values():
            wheel.reset()
        
        self.total_delta_v = 0.0
        self.total_propellant_consumed = 0.0
        self.power_consumption_history.clear()


def create_typical_thruster_suite() -> ActuatorSuite:
    """Create typical thruster configuration for small spacecraft."""
    suite = ActuatorSuite()
    
    # Thruster configuration
    thruster_config = ActuatorConfiguration(
        max_force=1.0,
        min_force=0.0,
        response_time=0.1,
        noise_std=0.01,
        efficiency=0.9,
        power_consumption=50.0
    )
    
    thruster_props = ThrusterProperties(
        specific_impulse=220.0,
        minimum_impulse_bit=1e-6,
        propellant_mass=2.0
    )
    
    # 8 thrusters in cubic configuration
    positions = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([1.0, -1.0, -1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, -1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0])
    ]
    
    directions = [
        np.array([-1.0, -1.0, -1.0]),
        np.array([-1.0, -1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([1.0, -1.0, -1.0]),
        np.array([1.0, -1.0, 1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, 1.0, 1.0])
    ]
    
    for i, (pos, direction) in enumerate(zip(positions, directions)):
        thruster = ThrusterModel(
            thruster_config, thruster_props, pos, direction, f"thruster_{i+1}"
        )
        suite.add_thruster(thruster)
    
    return suite


def create_typical_reaction_wheel_suite() -> ActuatorSuite:
    """Create typical reaction wheel configuration."""
    suite = ActuatorSuite()
    
    # Reaction wheel configuration
    wheel_config = ActuatorConfiguration(
        max_force=0.1,  # 0.1 N⋅m max torque
        response_time=0.05,
        noise_std=0.001,
        efficiency=0.95,
        power_consumption=20.0
    )
    
    wheel_props = ReactionWheelProperties(
        max_angular_momentum=1.0,
        max_angular_velocity=6000.0,
        wheel_inertia=0.01,
        friction_coefficient=1e-6
    )
    
    # 3 orthogonal reaction wheels
    axes = [
        np.array([1.0, 0.0, 0.0]),  # X-axis
        np.array([0.0, 1.0, 0.0]),  # Y-axis
        np.array([0.0, 0.0, 1.0])   # Z-axis
    ]
    
    for i, axis in enumerate(axes):
        wheel = ReactionWheelModel(
            wheel_config, wheel_props, axis, f"reaction_wheel_{['x', 'y', 'z'][i]}"
        )
        suite.add_reaction_wheel(wheel)
    
    return suite


def create_hybrid_actuator_suite() -> ActuatorSuite:
    """Create hybrid actuator suite with both thrusters and reaction wheels."""
    suite = ActuatorSuite()
    
    # Add thrusters
    thruster_suite = create_typical_thruster_suite()
    for thruster in thruster_suite.thrusters.values():
        suite.add_thruster(thruster)
    
    # Add reaction wheels
    wheel_suite = create_typical_reaction_wheel_suite()
    for wheel in wheel_suite.reaction_wheels.values():
        suite.add_reaction_wheel(wheel)
    
    return suite


def analyze_actuator_performance(suite: ActuatorSuite, 
                               simulation_results: Dict) -> Dict[str, float]:
    """
    Analyze actuator suite performance.
    
    Args:
        suite: Actuator suite
        simulation_results: Simulation results
    
    Returns:
        Performance metrics
    """
    status = suite.get_suite_status()
    
    metrics = {
        'total_delta_v': status['total_delta_v'],
        'total_propellant_consumed': status['total_propellant_consumed'],
        'average_power_consumption': np.mean(suite.power_consumption_history) if suite.power_consumption_history else 0.0,
        'peak_power_consumption': np.max(suite.power_consumption_history) if suite.power_consumption_history else 0.0,
        'num_active_thrusters': sum(1 for t in status['thrusters'].values() if t['enabled']),
        'num_active_wheels': sum(1 for w in status['reaction_wheels'].values() if w['enabled'])
    }
    
    # Thruster-specific metrics
    if status['thrusters']:
        propellant_fractions = [t['propellant_fraction'] for t in status['thrusters'].values()]
        temperatures = [t['temperature'] for t in status['thrusters'].values()]
        
        metrics.update({
            'min_propellant_fraction': np.min(propellant_fractions),
            'mean_propellant_fraction': np.mean(propellant_fractions),
            'max_thruster_temperature': np.max(temperatures),
            'mean_thruster_temperature': np.mean(temperatures)
        })
    
    # Reaction wheel specific metrics
    if status['reaction_wheels']:
        saturation_levels = [w['saturation_level'] for w in status['reaction_wheels'].values()]
        
        metrics.update({
            'max_wheel_saturation': np.max(saturation_levels),
            'mean_wheel_saturation': np.mean(saturation_levels),
            'num_saturated_wheels': sum(1 for w in status['reaction_wheels'].values() if w['is_saturated'])
        })
    
    return metrics

