"""
3D Visualization and Animation Module

This module provides comprehensive 3D visualization capabilities for
orbital rendezvous missions, including trajectory plotting, real-time
animation, and interactive visualization tools.

Author: Arthur Allex Feliphe Barbosa Moreno
Institution: IME - Instituto Militar de Engenharia - 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class TrajectoryData:
    """Container for trajectory data."""
    
    time: np.ndarray
    position: np.ndarray  # Shape: (N, 3)
    velocity: np.ndarray  # Shape: (N, 3)
    attitude: Optional[np.ndarray] = None  # Shape: (N, 4) quaternions
    control: Optional[np.ndarray] = None   # Shape: (N, 6) force + torque
    
    def __post_init__(self):
        """Validate trajectory data."""
        if self.position.shape[0] != len(self.time):
            raise ValueError("Position and time arrays must have same length")
        if self.velocity.shape[0] != len(self.time):
            raise ValueError("Velocity and time arrays must have same length")


class TrajectoryVisualizer:
    """3D trajectory visualization and animation."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """Initialize visualizer."""
        self.figsize = figsize
        self.fig = None
        self.axes = None
        
    def plot_trajectory_3d(self, 
                          chaser_data: TrajectoryData,
                          target_data: Optional[TrajectoryData] = None,
                          show_velocity_vectors: bool = False,
                          show_control_vectors: bool = False,
                          save_path: Optional[str] = None) -> plt.Figure:
        """Plot 3D trajectory with optional velocity and control vectors."""
        
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot chaser trajectory
        self.ax.plot(chaser_data.position[:, 0], 
                    chaser_data.position[:, 1], 
                    chaser_data.position[:, 2],
                    'b-', linewidth=2, label='Chaser trajectory')
        
        # Mark start and end points
        self.ax.scatter([chaser_data.position[0, 0]], 
                       [chaser_data.position[0, 1]], 
                       [chaser_data.position[0, 2]],
                       color='green', s=100, label='Start', marker='o')
        
        self.ax.scatter([chaser_data.position[-1, 0]], 
                       [chaser_data.position[-1, 1]], 
                       [chaser_data.position[-1, 2]],
                       color='orange', s=100, label='End', marker='s')
        
        # Plot target (if provided)
        if target_data is not None:
            self.ax.plot(target_data.position[:, 0], 
                        target_data.position[:, 1], 
                        target_data.position[:, 2],
                        'r--', linewidth=2, label='Target trajectory')
        else:
            # Static target at origin
            self.ax.scatter([0], [0], [0], color='red', s=200, 
                           label='Target', marker='*')
        
        # Add velocity vectors (every 10th point)
        if show_velocity_vectors and chaser_data.velocity is not None:
            step = max(1, len(chaser_data.time) // 20)
            for i in range(0, len(chaser_data.time), step):
                pos = chaser_data.position[i]
                vel = chaser_data.velocity[i] * 100  # Scale for visibility
                self.ax.quiver(pos[0], pos[1], pos[2],
                              vel[0], vel[1], vel[2],
                              color='cyan', alpha=0.6, arrow_length_ratio=0.1)
        
        # Add control vectors (every 20th point)
        if show_control_vectors and chaser_data.control is not None:
            step = max(1, len(chaser_data.time) // 10)
            for i in range(0, len(chaser_data.time), step):
                pos = chaser_data.position[i]
                force = chaser_data.control[i, :3] * 10  # Scale for visibility
                self.ax.quiver(pos[0], pos[1], pos[2],
                              force[0], force[1], force[2],
                              color='magenta', alpha=0.8, arrow_length_ratio=0.1)
        
        # Set labels and title
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_zlabel('Z [m]')
        self.ax.set_title('3D Orbital Rendezvous Trajectory')
        self.ax.legend()
        
        # Set equal aspect ratio
        self._set_equal_aspect_3d()
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D trajectory plot saved to {save_path}")
        
        return self.fig
    
    def plot_trajectory_projections(self,
                                   chaser_data: TrajectoryData,
                                   target_data: Optional[TrajectoryData] = None,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot trajectory projections on XY, XZ, and YZ planes."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Trajectory Projections', fontsize=16)
        
        # XY projection
        axes[0, 0].plot(chaser_data.position[:, 0], chaser_data.position[:, 1], 
                       'b-', linewidth=2, label='Chaser')
        axes[0, 0].scatter([chaser_data.position[0, 0]], [chaser_data.position[0, 1]], 
                          color='green', s=50, label='Start')
        axes[0, 0].scatter([chaser_data.position[-1, 0]], [chaser_data.position[-1, 1]], 
                          color='orange', s=50, label='End')
        if target_data is not None:
            axes[0, 0].plot(target_data.position[:, 0], target_data.position[:, 1], 
                           'r--', linewidth=2, label='Target')
        else:
            axes[0, 0].scatter([0], [0], color='red', s=100, label='Target')
        axes[0, 0].set_xlabel('X [m]')
        axes[0, 0].set_ylabel('Y [m]')
        axes[0, 0].set_title('XY Projection')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        
        # XZ projection
        axes[0, 1].plot(chaser_data.position[:, 0], chaser_data.position[:, 2], 
                       'b-', linewidth=2, label='Chaser')
        axes[0, 1].scatter([chaser_data.position[0, 0]], [chaser_data.position[0, 2]], 
                          color='green', s=50, label='Start')
        axes[0, 1].scatter([chaser_data.position[-1, 0]], [chaser_data.position[-1, 2]], 
                          color='orange', s=50, label='End')
        if target_data is not None:
            axes[0, 1].plot(target_data.position[:, 0], target_data.position[:, 2], 
                           'r--', linewidth=2, label='Target')
        else:
            axes[0, 1].scatter([0], [0], color='red', s=100, label='Target')
        axes[0, 1].set_xlabel('X [m]')
        axes[0, 1].set_ylabel('Z [m]')
        axes[0, 1].set_title('XZ Projection')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axis('equal')
        
        # YZ projection
        axes[1, 0].plot(chaser_data.position[:, 1], chaser_data.position[:, 2], 
                       'b-', linewidth=2, label='Chaser')
        axes[1, 0].scatter([chaser_data.position[0, 1]], [chaser_data.position[0, 2]], 
                          color='green', s=50, label='Start')
        axes[1, 0].scatter([chaser_data.position[-1, 1]], [chaser_data.position[-1, 2]], 
                          color='orange', s=50, label='End')
        if target_data is not None:
            axes[1, 0].plot(target_data.position[:, 1], target_data.position[:, 2], 
                           'r--', linewidth=2, label='Target')
        else:
            axes[1, 0].scatter([0], [0], color='red', s=100, label='Target')
        axes[1, 0].set_xlabel('Y [m]')
        axes[1, 0].set_ylabel('Z [m]')
        axes[1, 0].set_title('YZ Projection')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # Distance vs time
        distances = np.linalg.norm(chaser_data.position, axis=1)
        axes[1, 1].plot(chaser_data.time / 3600, distances, 'b-', linewidth=2)
        axes[1, 1].set_xlabel('Time [hours]')
        axes[1, 1].set_ylabel('Distance to Target [m]')
        axes[1, 1].set_title('Distance vs Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory projections saved to {save_path}")
        
        return fig
    
    def create_animation(self,
                        chaser_data: TrajectoryData,
                        target_data: Optional[TrajectoryData] = None,
                        animation_speed: float = 1.0,
                        trail_length: int = 50,
                        save_path: Optional[str] = None) -> animation.FuncAnimation:
        """Create animated 3D trajectory visualization."""
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize empty plots
        chaser_line, = ax.plot([], [], [], 'b-', linewidth=2, label='Chaser')
        chaser_point, = ax.plot([], [], [], 'bo', markersize=8)
        chaser_trail, = ax.plot([], [], [], 'b-', alpha=0.5, linewidth=1)
        
        target_point, = ax.plot([0], [0], [0], 'r*', markersize=15, label='Target')
        
        if target_data is not None:
            target_line, = ax.plot([], [], [], 'r--', linewidth=2, label='Target')
            target_trail, = ax.plot([], [], [], 'r-', alpha=0.5, linewidth=1)
        
        # Set up the plot
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Animated Orbital Rendezvous')
        ax.legend()
        
        # Set axis limits
        all_positions = chaser_data.position
        if target_data is not None:
            all_positions = np.vstack([all_positions, target_data.position])
        
        margin = 0.1
        x_range = [all_positions[:, 0].min(), all_positions[:, 0].max()]
        y_range = [all_positions[:, 1].min(), all_positions[:, 1].max()]
        z_range = [all_positions[:, 2].min(), all_positions[:, 2].max()]
        
        x_margin = (x_range[1] - x_range[0]) * margin
        y_margin = (y_range[1] - y_range[0]) * margin
        z_margin = (z_range[1] - z_range[0]) * margin
        
        ax.set_xlim(x_range[0] - x_margin, x_range[1] + x_margin)
        ax.set_ylim(y_range[0] - y_margin, y_range[1] + y_margin)
        ax.set_zlim(z_range[0] - z_margin, z_range[1] + z_margin)
        
        # Animation function
        def animate(frame):
            # Calculate actual frame index based on speed
            actual_frame = int(frame * animation_speed) % len(chaser_data.time)
            
            # Update chaser
            chaser_pos = chaser_data.position[actual_frame]
            chaser_point.set_data([chaser_pos[0]], [chaser_pos[1]])
            chaser_point.set_3d_properties([chaser_pos[2]])
            
            # Update chaser trail
            trail_start = max(0, actual_frame - trail_length)
            trail_pos = chaser_data.position[trail_start:actual_frame+1]
            if len(trail_pos) > 1:
                chaser_trail.set_data(trail_pos[:, 0], trail_pos[:, 1])
                chaser_trail.set_3d_properties(trail_pos[:, 2])
            
            # Update target (if moving)
            if target_data is not None:
                target_pos = target_data.position[actual_frame]
                target_point.set_data([target_pos[0]], [target_pos[1]])
                target_point.set_3d_properties([target_pos[2]])
                
                # Update target trail
                target_trail_pos = target_data.position[trail_start:actual_frame+1]
                if len(target_trail_pos) > 1:
                    target_trail.set_data(target_trail_pos[:, 0], target_trail_pos[:, 1])
                    target_trail.set_3d_properties(target_trail_pos[:, 2])
            
            # Update title with time
            current_time = chaser_data.time[actual_frame]
            ax.set_title(f'Animated Orbital Rendezvous - Time: {current_time/3600:.2f} hours')
            
            return chaser_point, chaser_trail, target_point
        
        # Create animation
        frames = int(len(chaser_data.time) / animation_speed)
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=50, blit=False, repeat=True)
        
        # Save animation if requested
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20)
            print(f"Animation saved to {save_path}")
        
        return anim
    
    def plot_performance_metrics(self,
                               chaser_data: TrajectoryData,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot performance metrics over time."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Performance Metrics', fontsize=16)
        
        time_hours = chaser_data.time / 3600
        
        # Position components
        axes[0, 0].plot(time_hours, chaser_data.position[:, 0], 'r-', label='X')
        axes[0, 0].plot(time_hours, chaser_data.position[:, 1], 'g-', label='Y')
        axes[0, 0].plot(time_hours, chaser_data.position[:, 2], 'b-', label='Z')
        axes[0, 0].set_xlabel('Time [hours]')
        axes[0, 0].set_ylabel('Position [m]')
        axes[0, 0].set_title('Position Components')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Velocity components
        axes[0, 1].plot(time_hours, chaser_data.velocity[:, 0], 'r-', label='Vx')
        axes[0, 1].plot(time_hours, chaser_data.velocity[:, 1], 'g-', label='Vy')
        axes[0, 1].plot(time_hours, chaser_data.velocity[:, 2], 'b-', label='Vz')
        axes[0, 1].set_xlabel('Time [hours]')
        axes[0, 1].set_ylabel('Velocity [m/s]')
        axes[0, 1].set_title('Velocity Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distance to target
        distances = np.linalg.norm(chaser_data.position, axis=1)
        axes[0, 2].plot(time_hours, distances, 'k-', linewidth=2)
        axes[0, 2].set_xlabel('Time [hours]')
        axes[0, 2].set_ylabel('Distance [m]')
        axes[0, 2].set_title('Distance to Target')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Velocity magnitude
        velocity_magnitudes = np.linalg.norm(chaser_data.velocity, axis=1)
        axes[1, 0].plot(time_hours, velocity_magnitudes, 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Time [hours]')
        axes[1, 0].set_ylabel('Speed [m/s]')
        axes[1, 0].set_title('Speed')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Control effort (if available)
        if chaser_data.control is not None:
            force_magnitudes = np.linalg.norm(chaser_data.control[:, :3], axis=1)
            axes[1, 1].plot(time_hours, force_magnitudes, 'orange', linewidth=2)
            axes[1, 1].set_xlabel('Time [hours]')
            axes[1, 1].set_ylabel('Control Force [N]')
            axes[1, 1].set_title('Control Effort')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Control data\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Control Effort')
        
        # Cumulative delta-V
        if chaser_data.control is not None:
            dt = np.diff(chaser_data.time)
            dt = np.append(dt, dt[-1])  # Extend to match array length
            force_magnitudes = np.linalg.norm(chaser_data.control[:, :3], axis=1)
            delta_v_increments = force_magnitudes * dt / 500.0  # Assuming 500kg mass
            cumulative_delta_v = np.cumsum(delta_v_increments)
            axes[1, 2].plot(time_hours, cumulative_delta_v, 'brown', linewidth=2)
            axes[1, 2].set_xlabel('Time [hours]')
            axes[1, 2].set_ylabel('Cumulative ΔV [m/s]')
            axes[1, 2].set_title('Fuel Consumption')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Control data\nnot available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Fuel Consumption')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance metrics plot saved to {save_path}")
        
        return fig
    
    def _set_equal_aspect_3d(self):
        """Set equal aspect ratio for 3D plot."""
        if self.ax is None:
            return
            
        # Get current axis limits
        x_limits = self.ax.get_xlim3d()
        y_limits = self.ax.get_ylim3d()
        z_limits = self.ax.get_zlim3d()
        
        # Calculate ranges
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        
        # Find maximum range
        max_range = max(x_range, y_range, z_range)
        
        # Calculate centers
        x_center = (x_limits[0] + x_limits[1]) / 2
        y_center = (y_limits[0] + y_limits[1]) / 2
        z_center = (z_limits[0] + z_limits[1]) / 2
        
        # Set equal limits
        self.ax.set_xlim3d(x_center - max_range/2, x_center + max_range/2)
        self.ax.set_ylim3d(y_center - max_range/2, y_center + max_range/2)
        self.ax.set_zlim3d(z_center - max_range/2, z_center + max_range/2)


def create_3d_animation(chaser_trajectory: np.ndarray,
                       time_array: np.ndarray,
                       target_trajectory: Optional[np.ndarray] = None,
                       save_path: Optional[str] = None) -> animation.FuncAnimation:
    """Convenience function to create 3D animation from trajectory arrays."""
    
    # Create TrajectoryData objects
    chaser_data = TrajectoryData(
        time=time_array,
        position=chaser_trajectory,
        velocity=np.gradient(chaser_trajectory, axis=0)  # Approximate velocity
    )
    
    target_data = None
    if target_trajectory is not None:
        target_data = TrajectoryData(
            time=time_array,
            position=target_trajectory,
            velocity=np.gradient(target_trajectory, axis=0)
        )
    
    # Create visualizer and animation
    visualizer = TrajectoryVisualizer()
    return visualizer.create_animation(chaser_data, target_data, save_path=save_path)


def generate_sample_trajectory(duration: float = 3600.0, 
                             time_step: float = 10.0) -> TrajectoryData:
    """Generate sample trajectory for demonstration purposes."""
    
    time = np.arange(0, duration + time_step, time_step)
    
    # Generate spiral approach trajectory
    x = 1000 * np.exp(-time/1800) * np.cos(time/600)
    y = 1000 * np.exp(-time/1800) * np.sin(time/600)
    z = 50 * np.sin(time/300) * np.exp(-time/3600)
    
    position = np.column_stack([x, y, z])
    
    # Calculate velocity by differentiation
    velocity = np.gradient(position, axis=0) / time_step
    
    # Generate sample control (proportional control)
    control_force = -0.01 * position - 0.1 * velocity
    control_torque = np.random.normal(0, 0.1, (len(time), 3))  # Random torque
    control = np.column_stack([control_force, control_torque])
    
    return TrajectoryData(
        time=time,
        position=position,
        velocity=velocity,
        control=control
    )


if __name__ == '__main__':
    # Example usage
    print("Generating sample trajectory...")
    trajectory = generate_sample_trajectory(duration=3600.0)
    
    print("Creating visualizations...")
    visualizer = TrajectoryVisualizer()
    
    # 3D plot
    fig1 = visualizer.plot_trajectory_3d(trajectory, show_velocity_vectors=True)
    
    # Projections
    fig2 = visualizer.plot_trajectory_projections(trajectory)
    
    # Performance metrics
    fig3 = visualizer.plot_performance_metrics(trajectory)
    
    # Animation
    print("Creating animation...")
    anim = visualizer.create_animation(trajectory, animation_speed=2.0)
    
    plt.show()
    
    print("Visualization example completed!")

