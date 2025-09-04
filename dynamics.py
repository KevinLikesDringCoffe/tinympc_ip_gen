#!/usr/bin/env python3
"""
Dynamics Model Module for Quadcopter Systems
Provides extensible dynamics models for different quadcopter platforms
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class NoiseModel:
    """Realistic noise model for quadcopter simulation"""
    # Position noise (m) - GPS/optical flow uncertainty
    position_std: float = 0.005  
    
    # Velocity noise (m/s) - Velocity estimation uncertainty
    velocity_std: float = 0.02
    
    # Angle noise (rad) - IMU measurement noise
    angle_std: float = 0.002  # ~0.1 degrees
    
    # Angular velocity noise (rad/s) - Gyroscope noise
    angular_velocity_std: float = 0.01  # ~0.6 deg/s
    
    # Process noise scaling factors
    position_process_noise: float = 0.0001
    velocity_process_noise: float = 0.001
    angle_process_noise: float = 0.0001
    angular_velocity_process_noise: float = 0.005
    
    # Actuator noise - thrust variations
    thrust_noise_std: float = 0.02  # 2% of nominal thrust
    
    def get_state_noise_std(self, dt: float) -> np.ndarray:
        """Get noise standard deviations for all state variables, scaled by timestep"""
        sqrt_dt = np.sqrt(dt)
        
        # State vector: [x, y, z, phi_x, phi_y, phi_z, vx, vy, vz, wx, wy, wz]
        noise_std = np.array([
            self.position_process_noise * sqrt_dt,  # x
            self.position_process_noise * sqrt_dt,  # y
            self.position_process_noise * sqrt_dt * 1.5,  # z
            self.angle_process_noise * sqrt_dt,     # roll
            self.angle_process_noise * sqrt_dt,     # pitch
            self.angle_process_noise * sqrt_dt * 2, # yaw
            self.velocity_process_noise * sqrt_dt,  # vx
            self.velocity_process_noise * sqrt_dt,  # vy
            self.velocity_process_noise * sqrt_dt * 1.2,  # vz
            self.angular_velocity_process_noise * sqrt_dt,  # wx
            self.angular_velocity_process_noise * sqrt_dt,  # wy
            self.angular_velocity_process_noise * sqrt_dt * 1.5,  # wz
        ])
        
        return noise_std
    
    def get_measurement_noise_std(self) -> np.ndarray:
        """Get measurement noise standard deviations"""
        return np.array([
            self.position_std, self.position_std, self.position_std * 1.5,  # position
            self.angle_std, self.angle_std, self.angle_std * 2,  # angles
            self.velocity_std, self.velocity_std, self.velocity_std * 1.2,  # velocity
            self.angular_velocity_std, self.angular_velocity_std, self.angular_velocity_std * 1.5  # angular velocity
        ])
    
    def get_initial_state_noise_std(self) -> np.ndarray:
        """Get initial state uncertainty"""
        return np.array([
            0.05, 0.05, 0.1,  # Initial position uncertainty (m)
            0.05, 0.05, 0.1,  # Initial angle uncertainty (rad) ~3-6 degrees
            0.01, 0.01, 0.02,  # Initial velocity uncertainty (m/s)
            0.02, 0.02, 0.05   # Initial angular velocity uncertainty (rad/s)
        ])

@dataclass
class QuadcopterParams:
    """Base parameters for quadcopter systems"""
    mass: float = 0.036  # kg
    gravity: float = 9.81  # m/s^2
    arm_length: float = 0.046  # m
    thrust_to_torque: float = 0.005964552
    Ixx: float = 1.43e-5  # kg*m^2
    Iyy: float = 1.43e-5  # kg*m^2
    Izz: float = 2.89e-5  # kg*m^2
    
    # Additional scaling factors for extensibility
    thrust_scale: float = 1.0
    torque_scale: float = 1.0
    drag_coefficient: float = 0.1
    angular_damping: float = 0.5

@dataclass
class CrazyflieParams(QuadcopterParams):
    """Specific parameters for Crazyflie platform"""
    mass: float = 0.036
    arm_length: float = 0.046
    thrust_to_torque: float = 0.005964552
    Ixx: float = 1.43e-5
    Iyy: float = 1.43e-5
    Izz: float = 2.89e-5

class DynamicsModel(ABC):
    """Abstract base class for quadcopter dynamics models"""
    
    def __init__(self, params: QuadcopterParams, noise_model: Optional[NoiseModel] = None):
        self.params = params
        self.noise_model = noise_model or NoiseModel()
        self.nstates = 12  # [x, y, z, phi_x, phi_y, phi_z, vx, vy, vz, wx, wy, wz]
        self.ninputs = 4   # [u1, u2, u3, u4]
    
    @abstractmethod
    def generate_system_matrices(self, control_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate discrete-time system matrices A and B"""
        pass
    
    @abstractmethod
    def generate_cost_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate LQR cost matrices Q and R"""
        pass
    
    @abstractmethod
    def generate_constraints(self) -> Dict:
        """Generate constraint parameters"""
        pass

class LinearizedQuadcopterDynamics(DynamicsModel):
    """Linearized quadcopter dynamics model around hover condition"""
    
    def __init__(self, params: Optional[QuadcopterParams] = None, 
                 noise_model: Optional[NoiseModel] = None):
        if params is None:
            params = CrazyflieParams()  # Default to Crazyflie
        super().__init__(params, noise_model)
        self._gravity_disturbance = None
    
    def generate_system_matrices(self, control_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate discrete-time system matrices A and B"""
        dt = 1.0 / control_freq
        g = self.params.gravity
        
        # Continuous-time system matrix
        A_cont = np.zeros((12, 12))
        
        # Position dynamics: x_dot = v + gravity_coupling
        A_cont[0, 6] = 1.0    # dx/dt = vx
        A_cont[1, 7] = 1.0    # dy/dt = vy  
        A_cont[2, 8] = 1.0    # dz/dt = vz
        A_cont[0, 4] = g      # dx/dt += g*phi_y (gravity coupling)
        A_cont[1, 3] = -g     # dy/dt += -g*phi_x (gravity coupling)
        
        # Attitude dynamics: phi_dot = omega
        A_cont[3, 9] = 1.0    # dphi_x/dt = wx
        A_cont[4, 10] = 1.0   # dphi_y/dt = wy
        A_cont[5, 11] = 1.0   # dphi_z/dt = wz
        
        # Velocity dynamics with damping
        drag_coeff = self.params.drag_coefficient
        A_cont[6, 6] = -drag_coeff   # dvx/dt = -drag*vx
        A_cont[7, 7] = -drag_coeff   # dvy/dt = -drag*vy
        A_cont[8, 8] = -drag_coeff   # dvz/dt = -drag*vz
        
        ang_damping = self.params.angular_damping
        A_cont[9, 9] = -ang_damping    # dwx/dt = -damping*wx
        A_cont[10, 10] = -ang_damping  # dwy/dt = -damping*wy
        A_cont[11, 11] = -ang_damping  # dwz/dt = -damping*wz
        
        # Discretize
        A = np.eye(12) + A_cont * dt
        
        # Add gravity as a constant disturbance
        gravity_effect = np.zeros(12)
        gravity_effect[8] = -g * dt  # dvz due to gravity
        self._gravity_disturbance = gravity_effect
        
        # Control matrix B
        B = np.zeros((12, 4))
        
        # Thrust affects vertical acceleration
        thrust_gain = self.params.thrust_scale / self.params.mass
        B[8, :] = thrust_gain
        
        # Moments affect angular accelerations
        arm = 0.707 * self.params.arm_length
        
        # Roll moment: tau_x = arm * (u3 + u4 - u1 - u2) / 4
        roll_gain = arm / (4 * self.params.Ixx) * 100 * self.params.torque_scale
        B[9, 0] = -roll_gain
        B[9, 1] = -roll_gain
        B[9, 2] = roll_gain
        B[9, 3] = roll_gain
        
        # Pitch moment: tau_y = arm * (u1 + u4 - u2 - u3) / 4
        pitch_gain = arm / (4 * self.params.Iyy) * 100 * self.params.torque_scale
        B[10, 0] = pitch_gain
        B[10, 1] = -pitch_gain
        B[10, 2] = -pitch_gain
        B[10, 3] = pitch_gain
        
        # Yaw moment: tau_z = k * (u1 + u3 - u2 - u4) / 4
        yaw_gain = self.params.thrust_to_torque / (4 * self.params.Izz) * 100 * self.params.torque_scale
        B[11, 0] = yaw_gain
        B[11, 1] = -yaw_gain
        B[11, 2] = yaw_gain
        B[11, 3] = -yaw_gain
        
        # Discretize
        B = B * dt
        
        return A, B
    
    def generate_cost_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate LQR cost matrices Q and R"""
        # State weights: [x, y, z, phi_x, phi_y, phi_z, vx, vy, vz, wx, wy, wz]
        q_diag = np.array([
            100.0,   # x position
            100.0,   # y position
            400.0,   # z position (higher weight)
            4.0,     # roll angle
            4.0,     # pitch angle
            1111.0,  # yaw angle (very high weight)
            4.0,     # x velocity
            4.0,     # y velocity
            100.0,   # z velocity (higher weight for gravity compensation)
            2.0,     # roll rate
            2.0,     # pitch rate
            25.0     # yaw rate
        ])
        
        Q = np.diag(q_diag)
        R = np.diag([144.0] * 4)  # Control weights
        
        return Q, R
    
    def generate_constraints(self) -> Dict:
        """Generate constraint parameters (box constraints)"""
        # Scale constraints based on platform size/capability
        thrust_limit = 0.5 * self.params.thrust_scale
        position_limit = 5.0 * max(1.0, self.params.mass / 0.036)  # Scale with mass
        velocity_limit = 3.0 * max(1.0, self.params.thrust_scale)  # Scale with thrust capability
        
        constraints = {
            'u_min': np.array([-thrust_limit] * 4),
            'u_max': np.array([thrust_limit] * 4),
            'x_min': np.array([-position_limit, -position_limit, 0, 
                              -0.5, -0.5, -np.pi, 
                              -velocity_limit, -velocity_limit, -velocity_limit, 
                              -2*np.pi, -2*np.pi, -2*np.pi]),
            'x_max': np.array([position_limit, position_limit, position_limit, 
                              0.5, 0.5, np.pi, 
                              velocity_limit, velocity_limit, velocity_limit, 
                              2*np.pi, 2*np.pi, 2*np.pi])
        }
        return constraints
    
    @property
    def gravity_disturbance(self) -> np.ndarray:
        """Get the gravity disturbance vector"""
        return self._gravity_disturbance if self._gravity_disturbance is not None else np.zeros(12)

class ScalableQuadcopterDynamics(LinearizedQuadcopterDynamics):
    """Scalable quadcopter dynamics for different sizes of platforms"""
    
    def __init__(self, params: Optional[QuadcopterParams] = None, 
                 noise_model: Optional[NoiseModel] = None):
        super().__init__(params, noise_model)
    
    @classmethod
    def create_scaled_crazyflie(cls, scale_factor: float = 1.0) -> 'ScalableQuadcopterDynamics':
        """Create a scaled version of Crazyflie dynamics
        
        Args:
            scale_factor: Scaling factor (1.0 = original Crazyflie, 2.0 = 2x larger, etc.)
        """
        params = CrazyflieParams()
        
        # Scale physical parameters appropriately
        params.mass = params.mass * (scale_factor ** 3)  # Mass scales with volume
        params.arm_length = params.arm_length * scale_factor  # Linear dimension
        params.Ixx = params.Ixx * (scale_factor ** 5)  # Moment of inertia scales with mass * length^2
        params.Iyy = params.Iyy * (scale_factor ** 5)
        params.Izz = params.Izz * (scale_factor ** 5)
        
        # Thrust and torque capabilities scale with motor size
        params.thrust_scale = scale_factor ** 2  # Thrust scales with rotor disk area
        params.torque_scale = scale_factor ** 3  # Torque scales with thrust * arm_length
        
        # Adjust damping for larger platforms
        params.drag_coefficient = params.drag_coefficient / scale_factor  # Larger platforms have less relative drag
        params.angular_damping = params.angular_damping / scale_factor
        
        return cls(params)

def create_dynamics_model(platform: str = "crazyflie", 
                         scale_factor: float = 1.0,
                         custom_params: Optional[QuadcopterParams] = None,
                         noise_model: Optional[NoiseModel] = None) -> DynamicsModel:
    """Factory function to create dynamics models
    
    Args:
        platform: Platform type ("crazyflie", "custom", "scaled_crazyflie")
        scale_factor: Scaling factor for scaled platforms
        custom_params: Custom parameters for "custom" platform
        noise_model: Custom noise model
    
    Returns:
        DynamicsModel instance
    """
    if platform == "crazyflie":
        return LinearizedQuadcopterDynamics(CrazyflieParams(), noise_model)
    elif platform == "scaled_crazyflie":
        return ScalableQuadcopterDynamics.create_scaled_crazyflie(scale_factor)
    elif platform == "custom" and custom_params is not None:
        return LinearizedQuadcopterDynamics(custom_params, noise_model)
    else:
        raise ValueError(f"Unknown platform: {platform}")