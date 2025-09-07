#!/usr/bin/env python3
"""
Trajectory Generation Module
Provides extensible trajectory generators for different flight patterns
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from obstacles import ObstacleEnvironment, CircleObstacle, SquareObstacle, ObstacleVisualProps

class TrajectoryType(Enum):
    HOVER = "hover"
    CIRCLE = "circle"
    FIGURE8 = "figure8"
    LINE = "line"
    SPIRAL = "spiral"
    LANDING = "landing"
    STEP = "step"
    ZIGZAG = "zigzag"
    WAYPOINTS = "waypoints"
    ACCELERATION = "acceleration"
    CONSTRAINED = "constrained"
    RAPID_ZIGZAG = "rapid_zigzag"
    
    # Complex composite trajectories
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    INSPECTION_PATROL = "inspection_patrol"
    AERIAL_DANCE = "aerial_dance"
    SLALOM_COURSE = "slalom_course"
    MULTI_OBSTACLE_NAV = "multi_obstacle_nav"

class TrajectoryGenerator(ABC):
    """Abstract base class for trajectory generators"""
    
    def __init__(self, nstates: int = 12):
        self.nstates = nstates
    
    @abstractmethod
    def generate(self, duration: float, control_freq: float, **kwargs) -> np.ndarray:
        """Generate trajectory
        
        Args:
            duration: Trajectory duration in seconds
            control_freq: Control frequency in Hz
            **kwargs: Trajectory-specific parameters
        
        Returns:
            X_ref: Reference trajectory array of shape (nstates, N)
        """
        pass

class HoverTrajectoryGenerator(TrajectoryGenerator):
    """Hover at a fixed position"""
    
    def generate(self, duration: float, control_freq: float, 
                 position: List[float] = [0, 0, 1], 
                 yaw: float = 0, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
        X_ref[0, :] = position[0]  # x
        X_ref[1, :] = position[1]  # y
        X_ref[2, :] = position[2]  # z
        X_ref[5, :] = yaw          # yaw
        
        return X_ref

class CircleTrajectoryGenerator(TrajectoryGenerator):
    """Circular trajectory in horizontal plane"""
    
    def generate(self, duration: float, control_freq: float,
                 radius: float = 1.0, 
                 center: List[float] = [0, 0, 1],
                 omega: Optional[float] = None, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        if omega is None:
            omega = 2*np.pi/duration
        
        X_ref[0, :] = center[0] + radius * np.cos(omega * t)
        X_ref[1, :] = center[1] + radius * np.sin(omega * t)
        X_ref[2, :] = center[2]
        
        # Velocities
        X_ref[6, :] = -radius * omega * np.sin(omega * t)
        X_ref[7, :] = radius * omega * np.cos(omega * t)
        X_ref[8, :] = 0
        
        return X_ref

class Figure8TrajectoryGenerator(TrajectoryGenerator):
    """Figure-8 trajectory pattern"""
    
    def generate(self, duration: float, control_freq: float,
                 scale: float = 1.0,
                 center: List[float] = [0, 0, 1],
                 omega: Optional[float] = None, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        if omega is None:
            omega = 2*np.pi/duration
        
        X_ref[0, :] = center[0] + scale * np.sin(omega * t)
        X_ref[1, :] = center[1] + scale * np.sin(2*omega * t)/2
        X_ref[2, :] = center[2]
        
        # Velocities
        X_ref[6, :] = scale * omega * np.cos(omega * t)
        X_ref[7, :] = scale * omega * np.cos(2*omega * t)
        X_ref[8, :] = 0
        
        return X_ref

class LineTrajectoryGenerator(TrajectoryGenerator):
    """Straight line trajectory between two points"""
    
    def generate(self, duration: float, control_freq: float,
                 start_pos: List[float] = [0, 0, 1],
                 end_pos: List[float] = [2, 0, 1], **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
        for i in range(3):
            X_ref[i, :] = np.linspace(start_pos[i], end_pos[i], N)
        
        # Constant velocity
        vel = np.array(end_pos) - np.array(start_pos)
        vel = vel / duration
        X_ref[6:9, :] = vel[:, np.newaxis]
        
        return X_ref

class SpiralTrajectoryGenerator(TrajectoryGenerator):
    """Spiral trajectory with expanding radius"""
    
    def generate(self, duration: float, control_freq: float,
                 radius_max: float = 1.0,
                 center: List[float] = [0, 0, 1],
                 height_gain: float = 0.5,
                 omega: Optional[float] = None, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        if omega is None:
            omega = 4*np.pi/duration
        
        r_t = radius_max * t / duration
        
        X_ref[0, :] = center[0] + r_t * np.cos(omega * t)
        X_ref[1, :] = center[1] + r_t * np.sin(omega * t)
        X_ref[2, :] = center[2] + height_gain * t / duration
        
        return X_ref

class LandingTrajectoryGenerator(TrajectoryGenerator):
    """Landing trajectory from height to ground"""
    
    def generate(self, duration: float, control_freq: float,
                 start_height: float = 2.0,
                 land_pos: List[float] = [0, 0, 0], **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
        X_ref[0, :] = land_pos[0]
        X_ref[1, :] = land_pos[1]
        X_ref[2, :] = np.linspace(start_height, land_pos[2], N)
        
        # Descending velocity
        X_ref[8, :] = -(start_height - land_pos[2]) / duration
        
        return X_ref

class StepTrajectoryGenerator(TrajectoryGenerator):
    """Step response trajectory with sudden position changes"""
    
    def generate(self, duration: float, control_freq: float,
                 step_size: float = 1.0,
                 step_duration: float = 2.0, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
        step_points = int(step_duration * control_freq)
        
        # Initial position
        X_ref[0:3, :] = np.array([[0], [0], [1.0]])
        
        # Add steps every step_duration seconds
        for i in range(1, int(duration / step_duration)):
            start_idx = i * step_points
            if start_idx < N:
                X_ref[0, start_idx:] += step_size * ((-1) ** i)
                X_ref[1, start_idx:] += step_size * 0.5 * ((-1) ** (i+1))
        
        return X_ref

class ZigzagTrajectoryGenerator(TrajectoryGenerator):
    """Zigzag trajectory with sharp direction changes"""
    
    def generate(self, duration: float, control_freq: float,
                 amplitude: float = 1.0,
                 frequency: float = 0.5, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        # Create zigzag pattern
        X_ref[0, :] = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        X_ref[1, :] = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + np.pi/2))
        X_ref[2, :] = 1.0  # Constant height
        
        return X_ref

class WaypointsTrajectoryGenerator(TrajectoryGenerator):
    """Multi-waypoint trajectory"""
    
    def generate(self, duration: float, control_freq: float,
                 waypoints: List[List[float]] = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], 
                 **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
        waypoints = np.array(waypoints)
        n_waypoints = len(waypoints)
        
        # Create smooth trajectory through waypoints
        for i in range(n_waypoints - 1):
            start_idx = int(i * N / (n_waypoints - 1))
            end_idx = int((i + 1) * N / (n_waypoints - 1))
            
            if end_idx <= N:
                for j in range(3):  # x, y, z
                    X_ref[j, start_idx:end_idx] = np.linspace(
                        waypoints[i, j], waypoints[i+1, j], 
                        end_idx - start_idx
                    )
        
        return X_ref

class AccelerationTrajectoryGenerator(TrajectoryGenerator):
    """High acceleration trajectory for testing control limits"""
    
    def generate(self, duration: float, control_freq: float,
                 max_acceleration: float = 2.0, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        # Sinusoidal trajectory with high acceleration
        freq = 2.0  # High frequency for high acceleration
        X_ref[0, :] = (max_acceleration / (2 * np.pi * freq)**2) * np.sin(2 * np.pi * freq * t)
        X_ref[1, :] = (max_acceleration / (2 * np.pi * freq)**2) * np.cos(2 * np.pi * freq * t)
        X_ref[2, :] = 1.0 + 0.5 * np.sin(np.pi * freq * t)
        
        # Add velocities (derivatives)
        X_ref[6, :] = (max_acceleration / (2 * np.pi * freq)) * np.cos(2 * np.pi * freq * t)
        X_ref[7, :] = -(max_acceleration / (2 * np.pi * freq)) * np.sin(2 * np.pi * freq * t)
        X_ref[8, :] = 0.5 * np.pi * freq * np.cos(np.pi * freq * t)
        
        return X_ref

class ConstrainedTrajectoryGenerator(TrajectoryGenerator):
    """Constrained trajectory for obstacle avoidance testing"""
    
    def generate(self, duration: float, control_freq: float,
                 obstacle_center: List[float] = [0.5, 0.5, 1.0],
                 obstacle_radius: float = 0.5, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        # Create trajectory that goes around obstacle
        center_x, center_y, center_z = obstacle_center
        radius = obstacle_radius + 0.3  # Safety margin
        
        # Circular path around obstacle
        X_ref[0, :] = center_x + radius * np.cos(2 * np.pi * t / duration)
        X_ref[1, :] = center_y + radius * np.sin(2 * np.pi * t / duration)
        X_ref[2, :] = center_z
        
        return X_ref

class RapidZigzagTrajectoryGenerator(TrajectoryGenerator):
    """Rapid zigzag trajectory with controlled maximum velocity"""
    
    def generate(self, duration: float, control_freq: float,
                 amplitude: float = 0.3,
                 segment_duration: float = 2.0,
                 height: float = 1.5,
                 forward_speed: float = 1.0, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        # Number of segments
        n_segments = int(duration / segment_duration)
        
        # Calculate bounds
        x_constraint = 4.0
        y_constraint = 1.0
        
        # Create waypoints
        waypoints_x = []
        waypoints_y = []
        
        mid_segment = n_segments // 2
        
        for i in range(n_segments + 1):
            if i <= mid_segment:
                progress = i / mid_segment if mid_segment > 0 else 0
                x = progress * x_constraint
            else:
                progress = (i - mid_segment) / (n_segments - mid_segment) if (n_segments - mid_segment) > 0 else 0
                x = x_constraint * (1 - progress)
            
            # Create Y oscillations
            base_oscillation = amplitude * ((-1) ** i)
            intensity_factor = 1.5 if abs(i - mid_segment) <= 2 else 1.0
            variation = 0.3 * np.sin(i * 1.3) + 0.2 * np.cos(i * 2.1)
            y = base_oscillation * intensity_factor + variation * amplitude * 0.4
            y = np.clip(y, -y_constraint, y_constraint)
            
            waypoints_x.append(x)
            waypoints_y.append(y)
        
        # Interpolate between waypoints with velocity limiting
        max_allowed_velocity = 8.0
        
        for i in range(len(t)):
            segment_idx = min(int(t[i] / segment_duration), n_segments - 1)
            segment_progress = (t[i] % segment_duration) / segment_duration
            
            smooth_factor = 1.5
            transition = 0.5 * (1 + np.tanh(smooth_factor * (segment_progress - 0.5)))
            
            # Interpolate position
            if segment_idx < len(waypoints_x) - 1:
                X_ref[0, i] = waypoints_x[segment_idx] + \
                              (waypoints_x[segment_idx + 1] - waypoints_x[segment_idx]) * transition
                X_ref[1, i] = waypoints_y[segment_idx] + \
                              (waypoints_y[segment_idx + 1] - waypoints_y[segment_idx]) * transition
            else:
                X_ref[0, i] = waypoints_x[-1]
                X_ref[1, i] = waypoints_y[-1]
            
            X_ref[2, i] = height + 0.05 * np.sin(2 * t[i])
            
            # Calculate and limit velocities
            if i > 0:
                dt_local = t[i] - t[i-1]
                vx_raw = (X_ref[0, i] - X_ref[0, i-1]) / dt_local
                vy_raw = (X_ref[1, i] - X_ref[1, i-1]) / dt_local
                vz_raw = (X_ref[2, i] - X_ref[2, i-1]) / dt_local
                
                v_magnitude = np.sqrt(vx_raw**2 + vy_raw**2 + vz_raw**2)
                if v_magnitude > max_allowed_velocity:
                    scale_factor = max_allowed_velocity / v_magnitude
                    X_ref[6, i] = vx_raw * scale_factor
                    X_ref[7, i] = vy_raw * scale_factor
                    X_ref[8, i] = vz_raw * scale_factor
                else:
                    X_ref[6, i] = vx_raw
                    X_ref[7, i] = vy_raw
                    X_ref[8, i] = vz_raw
        
        return X_ref

class CompositeTrajectoryGenerator(TrajectoryGenerator):
    """Base class for composite trajectories made from multiple segments"""
    
    def __init__(self, nstates: int = 12):
        super().__init__(nstates)
        self.segments = []
        self.segment_info = []  # Store metadata about each segment
    
    def add_segment(self, generator: TrajectoryGenerator, duration: float, 
                   name: str = "", **kwargs):
        """Add a trajectory segment"""
        self.segments.append({
            'generator': generator,
            'duration': duration,
            'kwargs': kwargs,
            'name': name
        })
    
    def generate(self, duration: float, control_freq: float, **kwargs) -> np.ndarray:
        """Generate composite trajectory by concatenating segments"""
        if not self.segments:
            raise ValueError("No trajectory segments defined")
        
        # Calculate total duration of all segments
        total_segment_duration = sum(seg['duration'] for seg in self.segments)
        
        # Scale segment durations to fit requested total duration
        scale_factor = duration / total_segment_duration
        
        all_trajectories = []
        segment_boundaries = [0]  # Track where each segment starts/ends
        
        for segment in self.segments:
            scaled_duration = segment['duration'] * scale_factor
            segment_trajectory = segment['generator'].generate(
                scaled_duration, control_freq, **segment['kwargs']
            )
            all_trajectories.append(segment_trajectory)
            segment_boundaries.append(segment_boundaries[-1] + segment_trajectory.shape[1])
        
        # Concatenate all trajectories
        X_ref = np.concatenate(all_trajectories, axis=1)
        
        # Store segment information for visualization
        self.segment_info = []
        for i, segment in enumerate(self.segments):
            self.segment_info.append({
                'name': segment['name'],
                'start_idx': segment_boundaries[i],
                'end_idx': segment_boundaries[i + 1],
                'duration': segment['duration'] * scale_factor
            })
        
        return X_ref

class ObstacleAvoidanceTrajectoryGenerator(CompositeTrajectoryGenerator):
    """Composite trajectory for avoiding obstacles using arc + line segments"""
    
    def __init__(self, nstates: int = 12):
        super().__init__(nstates)
        self.obstacle_env = None
    
    def generate(self, duration: float, control_freq: float,
                 start_pos: List[float] = [0, 0, 1.5],
                 end_pos: List[float] = [4, 0, 1.5],
                 obstacles: Optional[List] = None,
                 safety_margin: float = 0.4, **kwargs) -> np.ndarray:
        
        # Create default obstacle environment if none provided
        if obstacles is None:
            self.obstacle_env = self._create_default_obstacle_environment()
        else:
            self.obstacle_env = obstacles
        
        # Clear existing segments
        self.segments = []
        
        # Generate path around obstacles
        waypoints = self._plan_obstacle_avoiding_path(start_pos, end_pos, safety_margin)
        
        # Create trajectory segments based on waypoints
        segment_duration = duration / max(1, len(waypoints) - 1)
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            # Always use straight lines for waypoint-based navigation
            # The waypoints are already planned to avoid obstacles
            line_gen = LineTrajectoryGenerator(self.nstates)
            self.add_segment(
                line_gen, segment_duration,
                name=f"Segment_{i}",
                start_pos=start, end_pos=end
            )
        
        return super().generate(duration, control_freq, **kwargs)
    
    def _create_default_obstacle_environment(self) -> ObstacleEnvironment:
        """Create default obstacle environment"""
        env = ObstacleEnvironment()
        
        # Add a circular obstacle
        circle = CircleObstacle(
            center=[2.0, 0.5, 1.0],
            radius=0.6,
            height_range=(0.5, 2.5),
            visual_props=ObstacleVisualProps(color='red', alpha=0.7)
        )
        env.add_obstacle(circle)
        
        # Add a square obstacle
        square = SquareObstacle(
            center=[1.5, -1.0, 1.0],
            side_length=0.8,
            height_range=(0.5, 2.5),
            visual_props=ObstacleVisualProps(color='blue', alpha=0.7)
        )
        env.add_obstacle(square)
        
        return env
    
    def _plan_obstacle_avoiding_path(self, start_pos: List[float], 
                                   end_pos: List[float], 
                                   safety_margin: float) -> List[List[float]]:
        """Plan waypoints that avoid obstacles using simple but effective strategy"""
        waypoints = [start_pos]
        
        # Check if direct path is clear
        if self._is_path_clear(start_pos, end_pos, safety_margin):
            waypoints.append(end_pos)
            return waypoints
        
        # Simple strategy: go up and around obstacles
        # Calculate safe intermediate points
        
        # First waypoint: go up to clear obstacles
        mid_x = (start_pos[0] + end_pos[0]) / 2
        safe_y = start_pos[1] + 2.0  # Go well above obstacles
        safe_z = max(start_pos[2], end_pos[2]) + 0.5  # Go higher
        waypoint1 = [mid_x, safe_y, safe_z]
        
        # Check if this waypoint is safe, if not adjust
        while (self.obstacle_env.get_minimum_distance_to_obstacles(np.array(waypoint1)) < safety_margin):
            safe_y += 0.5  # Move further away
            safe_z += 0.2  # Go even higher
            waypoint1 = [mid_x, safe_y, safe_z]
            
            # Safety limit
            if safe_y > start_pos[1] + 4.0:
                break
        
        waypoints.append(waypoint1)
        
        # Second waypoint: position to descend to target
        descent_x = end_pos[0] - 0.5  # Position before target
        descent_y = waypoint1[1]      # Keep same Y
        descent_z = waypoint1[2]      # Keep same Z
        waypoint2 = [descent_x, descent_y, descent_z]
        
        # Ensure descent waypoint is also safe
        while (self.obstacle_env.get_minimum_distance_to_obstacles(np.array(waypoint2)) < safety_margin):
            descent_y += 0.3
            waypoint2 = [descent_x, descent_y, descent_z]
            
            # Safety limit
            if descent_y > waypoint1[1] + 2.0:
                break
        
        waypoints.append(waypoint2)
        waypoints.append(end_pos)
        
        return waypoints
    
    def _is_path_clear(self, start: List[float], end: List[float], 
                      safety_margin: float, resolution: int = 20) -> bool:
        """Check if path between two points is clear of obstacles"""
        if self.obstacle_env is None:
            return True
        
        # Sample points along the path
        for i in range(resolution + 1):
            t = i / resolution
            point = [start[j] + t * (end[j] - start[j]) for j in range(3)]
            
            # Check distance to all obstacles
            min_distance = self.obstacle_env.get_minimum_distance_to_obstacles(np.array(point))
            if min_distance < safety_margin:
                return False
        
        return True
    
    def _find_next_safe_waypoint(self, current: List[float], target: List[float], 
                               safety_margin: float) -> Optional[List[float]]:
        """Find next safe waypoint towards target"""
        # Try different directions to bypass obstacles
        directions = [
            [0, 1.2, 0],    # Go up in Y (larger step)
            [0, -1.2, 0],   # Go down in Y  
            [1, 0, 0],      # Go right in X
            [-1, 0, 0],     # Go left in X
            [0, 0, 0.8],    # Go up in Z (larger step)
            [0, 1.2, 0.5],  # Go up and right
            [0, -1.2, 0.5], # Go up and left
            [0.8, 0.8, 0],  # Diagonal movement
            [-0.8, 0.8, 0], # Diagonal movement
            [0.8, -0.8, 0], # Diagonal movement
            [-0.8, -0.8, 0],# Diagonal movement
        ]
        
        step_size = 1.0
        
        for direction in directions:
            candidate = [
                current[0] + direction[0] * step_size,
                current[1] + direction[1] * step_size,
                current[2] + direction[2] * step_size
            ]
            
            # Check if this position is safe
            if self.obstacle_env.get_minimum_distance_to_obstacles(np.array(candidate)) >= safety_margin:
                # Check if we can make progress towards target or it's a reasonable detour
                current_dist = np.linalg.norm(np.array(target) - np.array(current))
                candidate_dist = np.linalg.norm(np.array(target) - np.array(candidate))
                
                # Accept if we're getting closer or if it's a reasonable detour (less than 1.5x current distance)
                if candidate_dist <= current_dist * 1.5:
                    return candidate
        
        return None
    
    def _find_fallback_waypoint(self, current: List[float], target: List[float], 
                              safety_margin: float) -> List[float]:
        """Fallback waypoint when no good path is found"""
        # Go up and towards target
        mid_x = (current[0] + target[0]) / 2
        mid_y = current[1] + 1.5  # Go up to avoid obstacles
        mid_z = max(current[2], target[2]) + 0.5  # Go higher
        
        return [mid_x, mid_y, mid_z]
    
    def _should_use_curved_segment(self, start: List[float], end: List[float]) -> bool:
        """Determine if segment should be curved for obstacle avoidance"""
        if self.obstacle_env is None:
            return False
        
        # Use straight line if path is clear
        safety_margin = 0.3
        if self._is_path_clear(start, end, safety_margin):
            return False
        
        # Use curved segment if there are obstacles in the way
        mid_point = [(start[i] + end[i]) / 2 for i in range(3)]
        return self.obstacle_env.get_minimum_distance_to_obstacles(np.array(mid_point)) < 1.0
    
    def _calculate_arc_parameters(self, start: List[float], 
                                end: List[float]) -> Tuple[List[float], float]:
        """Calculate center and radius for circular arc that avoids obstacles"""
        if self.obstacle_env is None:
            center = [(start[i] + end[i]) / 2 for i in range(3)]
            radius = np.linalg.norm(np.array(end) - np.array(start)) / 2
            return center, max(radius, 0.5)
        
        # Find a safe center point that avoids obstacles
        safety_margin = 0.4
        base_center = [(start[i] + end[i]) / 2 for i in range(3)]
        base_radius = np.linalg.norm(np.array(end) - np.array(start)) / 2
        
        # Try different offsets to find a safe arc
        offsets = [
            [0, 0.8, 0],      # Offset in +Y
            [0, -0.8, 0],     # Offset in -Y
            [0.8, 0, 0],      # Offset in +X
            [-0.8, 0, 0],     # Offset in -X
            [0, 0, 0.5],      # Offset in +Z
            [0, 0.8, 0.3],    # Offset in +Y and +Z
            [0, -0.8, 0.3],   # Offset in -Y and +Z
        ]
        
        for offset in offsets:
            center = [base_center[i] + offset[i] for i in range(3)]
            
            # Check if this center creates a safe arc
            if self._is_arc_safe(start, end, center, base_radius, safety_margin):
                return center, max(base_radius, 0.5)
        
        # Fallback: use base center with larger radius
        return base_center, max(base_radius * 1.5, 0.8)
    
    def _is_arc_safe(self, start: List[float], end: List[float], center: List[float], 
                    radius: float, safety_margin: float, resolution: int = 20) -> bool:
        """Check if circular arc is safe from obstacles"""
        # Sample points along the arc
        start_vec = np.array(start) - np.array(center)
        end_vec = np.array(end) - np.array(center)
        
        # Calculate angles
        start_angle = np.arctan2(start_vec[1], start_vec[0])
        end_angle = np.arctan2(end_vec[1], end_vec[0])
        
        # Handle angle wrapping
        if end_angle < start_angle:
            end_angle += 2 * np.pi
        
        for i in range(resolution + 1):
            t = i / resolution
            angle = start_angle + t * (end_angle - start_angle)
            
            arc_point = [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
                start[2] + t * (end[2] - start[2])  # Linear interpolation in Z
            ]
            
            # Check distance to obstacles
            min_distance = self.obstacle_env.get_minimum_distance_to_obstacles(np.array(arc_point))
            if min_distance < safety_margin:
                return False
        
        return True

class InspectionPatrolTrajectoryGenerator(CompositeTrajectoryGenerator):
    """Composite trajectory for inspection missions with multiple patterns"""
    
    def generate(self, duration: float, control_freq: float,
                 inspection_points: List[List[float]] = None,
                 circle_radius: float = 0.8,
                 hover_duration_ratio: float = 0.2, **kwargs) -> np.ndarray:
        
        if inspection_points is None:
            inspection_points = [
                [1, 1, 1.5], [3, 2, 1.8], [2, -1, 1.2], [0, 0, 2.0]
            ]
        
        # Clear existing segments
        self.segments = []
        
        n_points = len(inspection_points)
        segment_duration = duration / (n_points * 3)  # 3 actions per point
        
        for i, point in enumerate(inspection_points):
            # 1. Navigate to inspection point
            if i == 0:
                start_pos = [0, 0, 1.0]
            else:
                start_pos = inspection_points[i-1]
            
            line_gen = LineTrajectoryGenerator(self.nstates)
            self.add_segment(
                line_gen, segment_duration,
                name=f"Navigate_to_{i}",
                start_pos=start_pos, end_pos=point
            )
            
            # 2. Circle around inspection point
            circle_gen = CircleTrajectoryGenerator(self.nstates)
            self.add_segment(
                circle_gen, segment_duration,
                name=f"Inspect_{i}",
                radius=circle_radius, center=point
            )
            
            # 3. Brief hover for detailed inspection
            hover_gen = HoverTrajectoryGenerator(self.nstates)
            self.add_segment(
                hover_gen, segment_duration * hover_duration_ratio,
                name=f"Hover_{i}",
                position=point
            )
        
        return super().generate(duration, control_freq, **kwargs)

class AerialDanceTrajectoryGenerator(CompositeTrajectoryGenerator):
    """Composite trajectory combining artistic flight patterns"""
    
    def generate(self, duration: float, control_freq: float,
                 center_pos: List[float] = [0, 0, 1.5],
                 scale_factor: float = 1.0, **kwargs) -> np.ndarray:
        
        # Clear existing segments
        self.segments = []
        
        # Segment 1: Figure-8 opening
        figure8_gen = Figure8TrajectoryGenerator(self.nstates)
        self.add_segment(
            figure8_gen, duration * 0.3,
            name="Opening_Figure8",
            scale=scale_factor, center=center_pos
        )
        
        # Segment 2: Ascending spiral
        spiral_center = center_pos.copy()
        spiral_gen = SpiralTrajectoryGenerator(self.nstates)
        self.add_segment(
            spiral_gen, duration * 0.25,
            name="Ascending_Spiral",
            radius_max=scale_factor * 1.2, center=spiral_center,
            height_gain=0.8
        )
        
        # Segment 3: High-altitude circle
        high_center = center_pos.copy()
        high_center[2] += 0.8
        circle_gen = CircleTrajectoryGenerator(self.nstates)
        self.add_segment(
            circle_gen, duration * 0.2,
            name="High_Circle",
            radius=scale_factor * 1.5, center=high_center
        )
        
        # Segment 4: Descending figure-8
        figure8_gen2 = Figure8TrajectoryGenerator(self.nstates)
        self.add_segment(
            figure8_gen2, duration * 0.25,
            name="Descending_Figure8",
            scale=scale_factor * 0.8, center=center_pos
        )
        
        return super().generate(duration, control_freq, **kwargs)

class SlalomCourseTrajectoryGenerator(CompositeTrajectoryGenerator):
    """Composite trajectory for slalom course navigation"""
    
    def generate(self, duration: float, control_freq: float,
                 course_length: float = 6.0,
                 obstacle_spacing: float = 1.5,
                 zigzag_amplitude: float = 0.8, **kwargs) -> np.ndarray:
        
        # Clear existing segments
        self.segments = []
        
        # Generate slalom course with alternating turns
        n_obstacles = int(course_length / obstacle_spacing)
        segment_duration = duration / n_obstacles
        
        current_pos = [0, 0, 1.2]
        
        for i in range(n_obstacles):
            # Calculate next position with zigzag pattern
            next_x = current_pos[0] + obstacle_spacing
            next_y = zigzag_amplitude * ((-1) ** i)
            next_z = 1.2 + 0.1 * np.sin(i * 0.5)  # Slight altitude variation
            next_pos = [next_x, next_y, next_z]
            
            # Use curved path for sharp turns, straight for gentle ones
            if abs(next_y - current_pos[1]) > zigzag_amplitude * 0.7:
                # Sharp turn - use circular arc
                arc_center = [(current_pos[0] + next_pos[0]) / 2,
                             (current_pos[1] + next_pos[1]) / 2,
                             (current_pos[2] + next_pos[2]) / 2]
                arc_radius = np.linalg.norm(np.array(next_pos) - np.array(current_pos)) / 2
                
                circle_gen = CircleTrajectoryGenerator(self.nstates)
                self.add_segment(
                    circle_gen, segment_duration,
                    name=f"Turn_{i}",
                    radius=max(arc_radius, 0.3), center=arc_center
                )
            else:
                # Gentle turn - use straight line
                line_gen = LineTrajectoryGenerator(self.nstates)
                self.add_segment(
                    line_gen, segment_duration,
                    name=f"Straight_{i}",
                    start_pos=current_pos, end_pos=next_pos
                )
            
            current_pos = next_pos
        
        return super().generate(duration, control_freq, **kwargs)

class MultiObstacleNavTrajectoryGenerator(CompositeTrajectoryGenerator):
    """Advanced composite trajectory for complex multi-obstacle navigation"""
    
    def generate(self, duration: float, control_freq: float,
                 start_pos: List[float] = [0, 0, 1.0],
                 target_pos: List[float] = [5, 3, 1.5],
                 intermediate_targets: List[List[float]] = None, **kwargs) -> np.ndarray:
        
        if intermediate_targets is None:
            intermediate_targets = [
                [1.5, 0.5, 1.2], [3.0, 1.5, 1.8], [4.0, 2.5, 1.3]
            ]
        
        # Clear existing segments
        self.segments = []
        
        # Create complex navigation path
        all_waypoints = [start_pos] + intermediate_targets + [target_pos]
        n_segments = len(all_waypoints) - 1
        
        for i in range(n_segments):
            current = all_waypoints[i]
            next_point = all_waypoints[i + 1]
            segment_duration = duration / n_segments
            
            # Vary trajectory type based on segment characteristics
            distance = np.linalg.norm(np.array(next_point) - np.array(current))
            height_change = abs(next_point[2] - current[2])
            
            if height_change > 0.3:
                # Significant height change - use spiral
                spiral_gen = SpiralTrajectoryGenerator(self.nstates)
                self.add_segment(
                    spiral_gen, segment_duration,
                    name=f"Climb_{i}",
                    radius_max=distance / 4, center=current,
                    height_gain=height_change
                )
            elif distance > 2.0:
                # Long distance - use waypoint navigation with curve
                mid_point = [(current[j] + next_point[j]) / 2 for j in range(3)]
                mid_point[1] += 0.5  # Add lateral offset for obstacle avoidance
                
                waypoints_gen = WaypointsTrajectoryGenerator(self.nstates)
                self.add_segment(
                    waypoints_gen, segment_duration,
                    name=f"Navigate_{i}",
                    waypoints=[current, mid_point, next_point]
                )
            else:
                # Short distance - direct line
                line_gen = LineTrajectoryGenerator(self.nstates)
                self.add_segment(
                    line_gen, segment_duration,
                    name=f"Direct_{i}",
                    start_pos=current, end_pos=next_point
                )
        
        return super().generate(duration, control_freq, **kwargs)

class TrajectoryFactory:
    """Factory class for creating trajectory generators"""
    
    _generators = {
        TrajectoryType.HOVER: HoverTrajectoryGenerator,
        TrajectoryType.CIRCLE: CircleTrajectoryGenerator,
        TrajectoryType.FIGURE8: Figure8TrajectoryGenerator,
        TrajectoryType.LINE: LineTrajectoryGenerator,
        TrajectoryType.SPIRAL: SpiralTrajectoryGenerator,
        TrajectoryType.LANDING: LandingTrajectoryGenerator,
        TrajectoryType.STEP: StepTrajectoryGenerator,
        TrajectoryType.ZIGZAG: ZigzagTrajectoryGenerator,
        TrajectoryType.WAYPOINTS: WaypointsTrajectoryGenerator,
        TrajectoryType.ACCELERATION: AccelerationTrajectoryGenerator,
        TrajectoryType.CONSTRAINED: ConstrainedTrajectoryGenerator,
        TrajectoryType.RAPID_ZIGZAG: RapidZigzagTrajectoryGenerator,
        
        # Complex composite trajectories
        TrajectoryType.OBSTACLE_AVOIDANCE: ObstacleAvoidanceTrajectoryGenerator,
        TrajectoryType.INSPECTION_PATROL: InspectionPatrolTrajectoryGenerator,
        TrajectoryType.AERIAL_DANCE: AerialDanceTrajectoryGenerator,
        TrajectoryType.SLALOM_COURSE: SlalomCourseTrajectoryGenerator,
        TrajectoryType.MULTI_OBSTACLE_NAV: MultiObstacleNavTrajectoryGenerator,
    }
    
    @classmethod
    def create_generator(cls, traj_type: TrajectoryType, nstates: int = 12) -> TrajectoryGenerator:
        """Create a trajectory generator instance
        
        Args:
            traj_type: Type of trajectory to generate
            nstates: Number of state variables
        
        Returns:
            TrajectoryGenerator instance
        """
        if traj_type not in cls._generators:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
        
        return cls._generators[traj_type](nstates)
    
    @classmethod
    def generate_trajectory(cls, traj_type: TrajectoryType, 
                           duration: float, 
                           control_freq: float,
                           nstates: int = 12,
                           **kwargs) -> np.ndarray:
        """Generate trajectory directly without creating generator instance
        
        Args:
            traj_type: Type of trajectory to generate
            duration: Trajectory duration in seconds
            control_freq: Control frequency in Hz
            nstates: Number of state variables
            **kwargs: Trajectory-specific parameters
        
        Returns:
            X_ref: Reference trajectory array of shape (nstates, N)
        """
        generator = cls.create_generator(traj_type, nstates)
        return generator.generate(duration, control_freq, **kwargs)
    
    @classmethod
    def register_generator(cls, traj_type: TrajectoryType, 
                          generator_class: type):
        """Register a new trajectory generator
        
        Args:
            traj_type: Trajectory type enum value
            generator_class: Generator class inheriting from TrajectoryGenerator
        """
        cls._generators[traj_type] = generator_class
    
    @classmethod
    def get_available_types(cls) -> List[TrajectoryType]:
        """Get list of available trajectory types"""
        return list(cls._generators.keys())

def create_trajectory(traj_type: Union[str, TrajectoryType], 
                     duration: float, 
                     control_freq: float,
                     nstates: int = 12,
                     **kwargs) -> np.ndarray:
    """Convenience function to create trajectories
    
    Args:
        traj_type: Type of trajectory (string or enum)
        duration: Trajectory duration in seconds
        control_freq: Control frequency in Hz
        nstates: Number of state variables
        **kwargs: Trajectory-specific parameters
    
    Returns:
        X_ref: Reference trajectory array of shape (nstates, N)
    """
    if isinstance(traj_type, str):
        traj_type = TrajectoryType(traj_type)
    
    return TrajectoryFactory.generate_trajectory(
        traj_type, duration, control_freq, nstates, **kwargs
    )