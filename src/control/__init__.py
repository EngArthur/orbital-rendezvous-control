"""
Control Systems Module

This module provides comprehensive control systems for orbital rendezvous,
including LQR controllers, guidance laws, and actuator models.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

from .lqr_controller import (
    TranslationalLQRController,
    AttitudeLQRController,
    CoupledLQRController,
    LQRWeights,
    ControlLimits,
    create_default_lqr_weights,
    create_conservative_lqr_weights,
    create_aggressive_lqr_weights,
    create_default_control_limits,
    analyze_lqr_performance
)

from .guidance_laws import (
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

from .actuator_models import (
    ThrusterModel,
    ReactionWheelModel,
    ActuatorSuite,
    ActuatorConfiguration,
    ThrusterProperties,
    ReactionWheelProperties,
    ActuatorType,
    create_typical_thruster_suite,
    create_typical_reaction_wheel_suite,
    create_hybrid_actuator_suite,
    analyze_actuator_performance
)

__all__ = [
    # LQR Controllers
    'TranslationalLQRController',
    'AttitudeLQRController', 
    'CoupledLQRController',
    'LQRWeights',
    'ControlLimits',
    'create_default_lqr_weights',
    'create_conservative_lqr_weights',
    'create_aggressive_lqr_weights',
    'create_default_control_limits',
    'analyze_lqr_performance',
    
    # Guidance Laws
    'LinearGuidanceLaw',
    'NonlinearGuidanceLaw',
    'AdaptiveGuidanceLaw',
    'GuidanceWaypoint',
    'GuidanceConstraints',
    'GuidancePhase',
    'create_approach_trajectory',
    'create_station_keeping_trajectory',
    'analyze_guidance_performance',
    
    # Actuator Models
    'ThrusterModel',
    'ReactionWheelModel',
    'ActuatorSuite',
    'ActuatorConfiguration',
    'ThrusterProperties',
    'ReactionWheelProperties',
    'ActuatorType',
    'create_typical_thruster_suite',
    'create_typical_reaction_wheel_suite',
    'create_hybrid_actuator_suite',
    'analyze_actuator_performance'
]

