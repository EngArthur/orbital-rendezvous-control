"""
Monte Carlo Simulation Framework

This module provides comprehensive Monte Carlo simulation capabilities for
orbital rendezvous missions, including uncertainty propagation, statistical
analysis, and performance assessment.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import json
import time
from pathlib import Path

@dataclass
class UncertaintyModel:
    """Model for system uncertainties and dispersions."""
    
    # Initial state uncertainties (3-sigma values)
    position_uncertainty: np.ndarray = field(default_factory=lambda: np.array([10.0, 5.0, 5.0]))  # m
    velocity_uncertainty: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.005, 0.005]))  # m/s
    attitude_uncertainty: float = np.radians(1.0)  # rad (3-sigma)
    angular_velocity_uncertainty: np.ndarray = field(default_factory=lambda: np.array([0.001, 0.001, 0.001]))  # rad/s
    
    # Sensor uncertainties (1-sigma values)
    sensor_uncertainties: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'lidar': {'range_noise': 0.1, 'rate_noise': 0.01},  # m, m/s
        'star_tracker': {'attitude_noise': np.radians(2/3600)},  # rad (2 arcsec)
        'gyroscope': {'bias_stability': 0.01, 'noise': 0.001},  # rad/s
        'accelerometer': {'bias_stability': 1e-4, 'noise': 1e-5},  # m/s²
        'gps': {'position_noise': 1.0, 'velocity_noise': 0.1}  # m, m/s
    })
    
    # Actuator uncertainties
    actuator_uncertainties: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'thrusters': {
            'thrust_uncertainty': 0.05,  # 5% thrust uncertainty
            'alignment_error': np.radians(0.5),  # rad
            'minimum_impulse_uncertainty': 0.1  # 10% uncertainty
        },
        'reaction_wheels': {
            'torque_uncertainty': 0.03,  # 3% torque uncertainty
            'momentum_uncertainty': 0.02,  # 2% momentum uncertainty
            'friction_uncertainty': 0.1   # 10% friction uncertainty
        }
    })


@dataclass
class SimulationConfiguration:
    """Configuration for Monte Carlo simulation."""
    
    # Simulation parameters
    num_runs: int = 1000
    max_workers: int = 4
    random_seed: Optional[int] = None
    
    # Time parameters
    simulation_duration: float = 14400.0  # seconds (4 hours)
    time_step: float = 10.0  # seconds
    
    # Analysis parameters
    success_criteria: Dict[str, float] = field(default_factory=lambda: {
        'final_position_error': 10.0,  # m
        'final_velocity_error': 0.1,   # m/s
        'final_attitude_error': np.radians(5.0),  # rad
        'max_control_effort': 100.0,   # N or N⋅m
        'total_delta_v': 50.0,         # m/s
        'mission_duration': 18000.0    # seconds (5 hours)
    })
    
    # Output parameters
    save_individual_runs: bool = False
    save_statistics: bool = True
    output_directory: str = "monte_carlo_results"
    
    # Uncertainty model
    uncertainty_model: UncertaintyModel = field(default_factory=UncertaintyModel)


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    
    run_id: int
    success: bool
    
    # Final states
    final_position_error: float
    final_velocity_error: float
    final_attitude_error: float
    
    # Performance metrics
    total_delta_v: float
    max_control_effort: float
    mission_duration: float
    fuel_consumption: float
    
    # Error statistics
    position_rms_error: float = 0.0
    velocity_rms_error: float = 0.0
    attitude_rms_error: float = 0.0
    
    # Failure mode (if applicable)
    failure_mode: Optional[str] = None
    failure_time: Optional[float] = None


@dataclass
class MonteCarloStatistics:
    """Statistical results from Monte Carlo simulation."""
    
    # Basic statistics
    num_runs: int
    num_successful: int
    success_rate: float
    
    # Performance statistics
    final_position_error_stats: Dict[str, float]
    final_velocity_error_stats: Dict[str, float]
    final_attitude_error_stats: Dict[str, float]
    total_delta_v_stats: Dict[str, float]
    fuel_consumption_stats: Dict[str, float]
    
    # Percentile data
    percentiles: Dict[str, np.ndarray]
    
    # Failure analysis
    failure_modes: Dict[str, int] = field(default_factory=dict)
    failure_times: List[float] = field(default_factory=list)


class MonteCarloSimulator:
    """Monte Carlo simulation framework for orbital rendezvous."""
    
    def __init__(self, config: SimulationConfiguration):
        """Initialize Monte Carlo simulator."""
        self.config = config
        self.results: List[SimulationResult] = []
        self.statistics: Optional[MonteCarloStatistics] = None
        
        # Create output directory
        self.output_path = Path(config.output_directory)
        self.output_path.mkdir(exist_ok=True)
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def run_single_simulation(self, run_id: int) -> SimulationResult:
        """Run a single Monte Carlo simulation."""
        try:
            # Generate dispersed initial conditions
            uncertainty = self.config.uncertainty_model
            
            # Base initial conditions
            base_position = np.array([1000.0, 100.0, 50.0])  # m
            base_velocity = np.array([-0.1, 0.01, 0.005])    # m/s
            
            # Generate dispersions (3-sigma)
            position_dispersion = np.random.normal(0, uncertainty.position_uncertainty / 3)
            velocity_dispersion = np.random.normal(0, uncertainty.velocity_uncertainty / 3)
            
            # Apply dispersions
            initial_position = base_position + position_dispersion
            initial_velocity = base_velocity + velocity_dispersion
            
            # Simplified simulation loop
            times = np.arange(0.0, self.config.simulation_duration + self.config.time_step, self.config.time_step)
            
            current_position = initial_position.copy()
            current_velocity = initial_velocity.copy()
            
            total_delta_v = 0.0
            max_control_effort = 0.0
            
            positions = [current_position.copy()]
            velocities = [current_velocity.copy()]
            
            # Simple control loop
            for i, t in enumerate(times[1:], 1):
                dt = times[i] - times[i-1]
                
                # Simple proportional control
                control_force = -0.01 * current_position - 0.1 * current_velocity
                
                # Add actuator uncertainties
                thrust_uncertainty = 1 + np.random.normal(0, uncertainty.actuator_uncertainties['thrusters']['thrust_uncertainty'])
                control_force *= thrust_uncertainty
                
                # Update metrics
                control_magnitude = np.linalg.norm(control_force)
                total_delta_v += control_magnitude * dt / 500.0  # Assuming 500kg mass
                max_control_effort = max(max_control_effort, control_magnitude)
                
                # Simple integration
                acceleration = control_force / 500.0  # 500kg mass
                current_velocity += acceleration * dt
                current_position += current_velocity * dt
                
                positions.append(current_position.copy())
                velocities.append(current_velocity.copy())
            
            # Compute final errors
            final_position_error = np.linalg.norm(current_position)
            final_velocity_error = np.linalg.norm(current_velocity)
            final_attitude_error = np.random.normal(0, 0.1)  # Simplified
            
            # Compute RMS errors
            positions_array = np.array(positions)
            velocities_array = np.array(velocities)
            
            position_rms_error = np.sqrt(np.mean(np.sum(positions_array**2, axis=1)))
            velocity_rms_error = np.sqrt(np.mean(np.sum(velocities_array**2, axis=1)))
            attitude_rms_error = abs(final_attitude_error)
            
            # Check success criteria
            success = (
                final_position_error <= self.config.success_criteria['final_position_error'] and
                final_velocity_error <= self.config.success_criteria['final_velocity_error'] and
                abs(final_attitude_error) <= self.config.success_criteria['final_attitude_error'] and
                max_control_effort <= self.config.success_criteria['max_control_effort'] and
                total_delta_v <= self.config.success_criteria['total_delta_v']
            )
            
            # Create result
            result = SimulationResult(
                run_id=run_id,
                success=success,
                final_position_error=final_position_error,
                final_velocity_error=final_velocity_error,
                final_attitude_error=abs(final_attitude_error),
                total_delta_v=total_delta_v,
                max_control_effort=max_control_effort,
                mission_duration=self.config.simulation_duration,
                fuel_consumption=total_delta_v * 500.0 / 220.0 / 9.81,  # Simplified fuel calculation
                position_rms_error=position_rms_error,
                velocity_rms_error=velocity_rms_error,
                attitude_rms_error=attitude_rms_error
            )
            
            return result
            
        except Exception as e:
            # Handle simulation failure
            return SimulationResult(
                run_id=run_id,
                success=False,
                final_position_error=np.inf,
                final_velocity_error=np.inf,
                final_attitude_error=np.inf,
                total_delta_v=np.inf,
                max_control_effort=np.inf,
                mission_duration=self.config.simulation_duration,
                fuel_consumption=np.inf,
                failure_mode=str(type(e).__name__),
                failure_time=0.0
            )
    
    def run_monte_carlo(self) -> MonteCarloStatistics:
        """Run complete Monte Carlo simulation."""
        print(f"Starting Monte Carlo simulation with {self.config.num_runs} runs...")
        start_time = time.time()
        
        # Run simulations
        self.results = []
        for i in range(self.config.num_runs):
            result = self.run_single_simulation(i)
            self.results.append(result)
            
            # Progress update
            if (i + 1) % max(1, self.config.num_runs // 10) == 0:
                progress = (i + 1) / self.config.num_runs * 100
                print(f"Progress: {progress:.0f}%")
        
        elapsed_time = time.time() - start_time
        print(f"Monte Carlo simulation completed in {elapsed_time:.1f} seconds")
        
        # Compute statistics
        self.statistics = self._compute_statistics()
        
        # Save results
        if self.config.save_statistics:
            self._save_results()
        
        return self.statistics
    
    def _compute_statistics(self) -> MonteCarloStatistics:
        """Compute statistical analysis of results."""
        successful_results = [r for r in self.results if r.success]
        
        # Basic statistics
        num_successful = len(successful_results)
        success_rate = num_successful / len(self.results)
        
        if num_successful == 0:
            # Handle case with no successful runs
            return MonteCarloStatistics(
                num_runs=len(self.results),
                num_successful=0,
                success_rate=0.0,
                final_position_error_stats={'mean': np.inf, 'std': np.inf, 'min': np.inf, 'max': np.inf},
                final_velocity_error_stats={'mean': np.inf, 'std': np.inf, 'min': np.inf, 'max': np.inf},
                final_attitude_error_stats={'mean': np.inf, 'std': np.inf, 'min': np.inf, 'max': np.inf},
                total_delta_v_stats={'mean': np.inf, 'std': np.inf, 'min': np.inf, 'max': np.inf},
                fuel_consumption_stats={'mean': np.inf, 'std': np.inf, 'min': np.inf, 'max': np.inf},
                percentiles={}
            )
        
        # Extract data arrays
        position_errors = np.array([r.final_position_error for r in successful_results])
        velocity_errors = np.array([r.final_velocity_error for r in successful_results])
        attitude_errors = np.array([r.final_attitude_error for r in successful_results])
        delta_vs = np.array([r.total_delta_v for r in successful_results])
        fuel_consumptions = np.array([r.fuel_consumption for r in successful_results])
        
        # Compute statistics
        def compute_stats(data):
            return {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'median': np.median(data)
            }
        
        # Percentiles
        percentiles_values = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentiles = {
            'position_error': np.percentile(position_errors, percentiles_values),
            'velocity_error': np.percentile(velocity_errors, percentiles_values),
            'attitude_error': np.percentile(attitude_errors, percentiles_values),
            'delta_v': np.percentile(delta_vs, percentiles_values),
            'fuel_consumption': np.percentile(fuel_consumptions, percentiles_values),
            'percentile_values': percentiles_values
        }
        
        # Failure analysis
        failed_results = [r for r in self.results if not r.success]
        failure_modes = {}
        failure_times = []
        
        for result in failed_results:
            if result.failure_mode:
                failure_modes[result.failure_mode] = failure_modes.get(result.failure_mode, 0) + 1
            if result.failure_time is not None:
                failure_times.append(result.failure_time)
        
        return MonteCarloStatistics(
            num_runs=len(self.results),
            num_successful=num_successful,
            success_rate=success_rate,
            final_position_error_stats=compute_stats(position_errors),
            final_velocity_error_stats=compute_stats(velocity_errors),
            final_attitude_error_stats=compute_stats(attitude_errors),
            total_delta_v_stats=compute_stats(delta_vs),
            fuel_consumption_stats=compute_stats(fuel_consumptions),
            percentiles=percentiles,
            failure_modes=failure_modes,
            failure_times=failure_times
        )
    
    def _save_results(self):
        """Save simulation results and statistics."""
        # Save configuration
        config_file = self.output_path / "simulation_config.json"
        with open(config_file, 'w') as f:
            config_dict = {
                'num_runs': self.config.num_runs,
                'simulation_duration': self.config.simulation_duration,
                'time_step': self.config.time_step,
                'success_criteria': self.config.success_criteria
            }
            json.dump(config_dict, f, indent=2)
        
        # Save statistics
        stats_file = self.output_path / "monte_carlo_statistics.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump(self.statistics, f)
        
        print(f"Results saved to {self.output_path}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive simulation report."""
        if self.statistics is None:
            return "No statistics available. Run simulation first."
        
        stats = self.statistics
        
        report = f"""
Monte Carlo Simulation Report
============================

Simulation Configuration:
- Number of runs: {stats.num_runs}
- Successful runs: {stats.num_successful}
- Success rate: {stats.success_rate:.1%}

Performance Statistics (Successful Runs Only):
----------------------------------------------

Final Position Error:
- Mean: {stats.final_position_error_stats['mean']:.2f} m
- Std:  {stats.final_position_error_stats['std']:.2f} m
- Min:  {stats.final_position_error_stats['min']:.2f} m
- Max:  {stats.final_position_error_stats['max']:.2f} m

Final Velocity Error:
- Mean: {stats.final_velocity_error_stats['mean']:.4f} m/s
- Std:  {stats.final_velocity_error_stats['std']:.4f} m/s
- Min:  {stats.final_velocity_error_stats['min']:.4f} m/s
- Max:  {stats.final_velocity_error_stats['max']:.4f} m/s

Total Delta-V:
- Mean: {stats.total_delta_v_stats['mean']:.2f} m/s
- Std:  {stats.total_delta_v_stats['std']:.2f} m/s
- Min:  {stats.total_delta_v_stats['min']:.2f} m/s
- Max:  {stats.total_delta_v_stats['max']:.2f} m/s

Fuel Consumption:
- Mean: {stats.fuel_consumption_stats['mean']:.2f} kg
- Std:  {stats.fuel_consumption_stats['std']:.2f} kg
- Min:  {stats.fuel_consumption_stats['min']:.2f} kg
- Max:  {stats.fuel_consumption_stats['max']:.2f} kg
"""
        
        if stats.failure_modes:
            report += "\nFailure Analysis:\n"
            report += "----------------\n"
            for mode, count in stats.failure_modes.items():
                percentage = count / stats.num_runs * 100
                report += f"  {mode}: {count} runs ({percentage:.1f}%)\n"
        
        return report
    
    def plot_results(self, save_plots: bool = True):
        """Generate comprehensive plots of Monte Carlo results."""
        if self.statistics is None:
            print("No statistics available. Run simulation first.")
            return
        
        successful_results = [r for r in self.results if r.success]
        
        if len(successful_results) == 0:
            print("No successful runs to plot.")
            return
        
        # Extract data
        position_errors = [r.final_position_error for r in successful_results]
        velocity_errors = [r.final_velocity_error for r in successful_results]
        delta_vs = [r.total_delta_v for r in successful_results]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Monte Carlo Results ({len(successful_results)} successful runs)', fontsize=16)
        
        # Position error histogram
        axes[0, 0].hist(position_errors, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.config.success_criteria['final_position_error'], 
                          color='red', linestyle='--', label='Requirement')
        axes[0, 0].set_xlabel('Final Position Error [m]')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Final Position Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Velocity error histogram
        axes[0, 1].hist(velocity_errors, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.config.success_criteria['final_velocity_error'], 
                          color='red', linestyle='--', label='Requirement')
        axes[0, 1].set_xlabel('Final Velocity Error [m/s]')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Final Velocity Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Delta-V histogram
        axes[1, 0].hist(delta_vs, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(self.config.success_criteria['total_delta_v'], 
                          color='red', linestyle='--', label='Requirement')
        axes[1, 0].set_xlabel('Total Delta-V [m/s]')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Total Delta-V Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Success rate pie chart
        success_count = len(successful_results)
        failure_count = len(self.results) - success_count
        
        axes[1, 1].pie([success_count, failure_count], 
                      labels=[f'Success ({success_count})', f'Failure ({failure_count})'],
                      autopct='%1.1f%%', startangle=90,
                      colors=['lightgreen', 'lightcoral'])
        axes[1, 1].set_title('Mission Success Rate')
        
        plt.tight_layout()
        
        if save_plots:
            plot_file = self.output_path / "monte_carlo_results.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {plot_file}")
        
        plt.show()


def create_default_simulation_config() -> SimulationConfiguration:
    """Create default simulation configuration."""
    return SimulationConfiguration(
        num_runs=100,  # Reduced for example
        max_workers=2,
        simulation_duration=14400.0,  # 4 hours
        time_step=10.0
    )


if __name__ == '__main__':
    # Example usage
    config = create_default_simulation_config()
    simulator = MonteCarloSimulator(config)
    
    print("Running Monte Carlo simulation example...")
    statistics = simulator.run_monte_carlo()
    
    print("\nSimulation Report:")
    print(simulator.generate_report())
    
    print("\nGenerating plots...")
    simulator.plot_results()
    
    print("Monte Carlo simulation example completed!")

