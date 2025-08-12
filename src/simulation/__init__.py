"""
Simulation Framework Module

This module provides comprehensive simulation capabilities for orbital
rendezvous missions, including Monte Carlo analysis, 3D visualization,
and performance assessment.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

from .monte_carlo import (
    MonteCarloSimulator,
    SimulationConfiguration,
    UncertaintyModel,
    SimulationResult,
    MonteCarloStatistics,
    create_default_simulation_config
)

from .visualization import (
    TrajectoryVisualizer,
    TrajectoryData,
    create_3d_animation,
    generate_sample_trajectory
)

__all__ = [
    # Monte Carlo simulation
    'MonteCarloSimulator',
    'SimulationConfiguration',
    'UncertaintyModel',
    'SimulationResult',
    'MonteCarloStatistics',
    'create_default_simulation_config',
    
    # Visualization
    'TrajectoryVisualizer',
    'TrajectoryData',
    'create_3d_animation',
    'generate_sample_trajectory'
]

__version__ = '1.0.0'
__author__ = 'Arthur Allex Feliphe Barbosa Moreno'
__institution__ = 'IME - Instituto Militar de Engenharia'

