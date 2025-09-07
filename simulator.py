#!/usr/bin/env python3
"""
Simulator Module for MPC Controllers
Provides extensible simulation framework for different control strategies
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import sys
import os

# Try to import tinympc, but don't fail if it's not available
try:
    import tinympc
    TINYMPC_AVAILABLE = True
except ImportError:
    TINYMPC_AVAILABLE = False
    print("WARNING: TinyMPC software library not available. Only hardware solver will be supported.")

from dynamics import DynamicsModel, NoiseModel

def check_available_solvers():
    """Check which MPC solvers are available"""
    available_solvers = []
    
    # Check software solver
    if TINYMPC_AVAILABLE:
        available_solvers.append("software")
    
    # Check hardware solver
    try:
        # Add parent directory to Python path to import hw_interface
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from hw_interface import MPCSolver
        # Try to check if PYNQ is available
        try:
            from pynq import Overlay
            available_solvers.append("hardware")
        except ImportError:
            pass  # PYNQ not available, hardware solver not available
    except ImportError:
        pass  # hw_interface not available
    
    return available_solvers

def get_default_solver_type():
    """Get the default solver type based on what's available"""
    available = check_available_solvers()
    
    if "software" in available:
        return "software"
    elif "hardware" in available:
        return "hardware"
    else:
        return None

class ControlMode(Enum):
    """MPC Control Mode"""
    TRACKING = "tracking"     # Traditional trajectory tracking
    REGULATOR = "regulator"   # Regulator mode - each step tracks back to origin

class MPCSimulator(ABC):
    """Abstract base class for MPC simulators"""
    
    def __init__(self, dynamics_model: DynamicsModel, 
                 X_ref: np.ndarray,
                 horizon: int = 50,
                 control_mode: ControlMode = ControlMode.TRACKING):
        self.dynamics_model = dynamics_model
        self.X_ref = X_ref
        self.horizon = horizon
        self.control_mode = control_mode
        
        # Generate system matrices
        self.dt = 1.0 / 50.0  # Default timestep, will be updated
        self.A, self.B = dynamics_model.generate_system_matrices(50.0)
        self.Q, self.R = dynamics_model.generate_cost_matrices()
        self.constraints = dynamics_model.generate_constraints()
        
        # History tracking
        self.x_history = []
        self.u_history = []
        self.cost_history = []
        
    def set_control_frequency(self, control_freq: float):
        """Update control frequency and regenerate system matrices"""
        self.dt = 1.0 / control_freq
        self.A, self.B = self.dynamics_model.generate_system_matrices(control_freq)
    
    @abstractmethod
    def simulate(self, steps: int = 200, 
                 initial_state: Optional[np.ndarray] = None,
                 verbose: bool = True):
        """Run simulation"""
        pass
    
    def reset(self):
        """Reset simulation state"""
        self.x_history = []
        self.u_history = []
        self.cost_history = []
    
    def get_results(self) -> Dict:
        """Get simulation results"""
        x_history = np.array(self.x_history)
        u_history = np.array(self.u_history) if self.u_history else np.array([])
        
        results = {
            'x_history': x_history,
            'u_history': u_history,
            'cost_history': self.cost_history,
            'dynamics_model': self.dynamics_model,
            'constraints': self.constraints
        }
        
        if len(x_history) > 0:
            # Calculate performance metrics
            results.update(self._calculate_performance_metrics(x_history))
        
        return results
    
    def _calculate_performance_metrics(self, x_history: np.ndarray) -> Dict:
        """Calculate performance metrics"""
        metrics = {}
        
        # Position tracking error
        position_errors = []
        position_error_components = []
        
        for i in range(len(x_history)):
            if i < self.X_ref.shape[1]:
                ref_pos = self.X_ref[:3, i]
            else:
                ref_pos = self.X_ref[:3, -1]
            
            actual_pos = x_history[i, :3]
            error = np.linalg.norm(actual_pos - ref_pos)
            position_errors.append(error)
            
            error_components = actual_pos - ref_pos
            position_error_components.append(error_components)
        
        position_error_components = np.array(position_error_components)
        
        metrics['final_position_error'] = position_errors[-1] if position_errors else 0.0
        metrics['mean_position_error'] = np.mean(position_errors)
        metrics['max_position_error'] = np.max(position_errors)
        metrics['rmse_position_error'] = np.sqrt(np.mean(position_error_components**2))
        
        if self.cost_history:
            metrics['average_cost'] = np.mean(self.cost_history)
        
        if self.u_history:
            u_history = np.array(self.u_history)
            metrics['max_control_input'] = np.max(np.abs(u_history))
            
            # Check constraint violations
            u_min = self.constraints['u_min']
            u_max = self.constraints['u_max']
            violations = np.sum((u_history < u_min) | (u_history > u_max))
            metrics['constraint_violations'] = violations
        
        return metrics

class TinyMPCSimulator(MPCSimulator):
    """MPC simulator using TinyMPC solver (software or hardware)"""
    
    # Hardware configuration constants
    HARDWARE_CONFIG = {
        'overlay_path': 'tinympc_design.bit',
        'ip_name': 'tinympc_hls_0',
        'clock_frequency_mhz': 250
    }
    
    def __init__(self, dynamics_model: DynamicsModel, 
                 X_ref: np.ndarray,
                 horizon: int = 50,
                 control_mode: ControlMode = ControlMode.TRACKING,
                 solver_type: str = "auto",
                 bitstream_path: str = None):
        super().__init__(dynamics_model, X_ref, horizon, control_mode)
        
        # Store bitstream path for hardware solver
        self.bitstream_path = bitstream_path
        
        # Auto-select solver type if requested
        if solver_type == "auto":
            available_solvers = check_available_solvers()
            if not available_solvers:
                raise RuntimeError(
                    "No MPC solvers are available. Please install either:\n"
                    "1. TinyMPC library for software solver, or\n"
                    "2. PYNQ library with hw_interface for hardware solver"
                )
            
            # Prefer software solver if available (usually faster for development)
            if "software" in available_solvers:
                solver_type = "software"
                print(f"Auto-selected software solver (available: {available_solvers})")
            else:
                solver_type = "hardware"
                print(f"Auto-selected hardware solver (available: {available_solvers})")
        
        # Manual selection with fallback
        elif solver_type == "software" and not TINYMPC_AVAILABLE:
            print("WARNING: TinyMPC software library not available, switching to hardware solver")
            solver_type = "hardware"
        
        self.solver_type = solver_type
        self._setup_solver()
    
    def _setup_solver(self):
        """Setup MPC solver based on solver type"""
        if self.solver_type == "hardware":
            try:
                # Add parent directory to Python path to import hw_interface
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                # Try to import and use hardware solver
                from hw_interface import MPCSolver
                
                # Use custom bitstream path if provided, otherwise use default
                overlay_path = self.bitstream_path if self.bitstream_path else self.HARDWARE_CONFIG['overlay_path']
                
                self.mpc = MPCSolver(
                    driver_or_path=overlay_path,
                    ip_name=self.HARDWARE_CONFIG['ip_name'],
                    nx=12, nu=4, n_horizon=self.horizon,
                    clock_frequency_mhz=self.HARDWARE_CONFIG['clock_frequency_mhz']
                )
                
                print(f"Hardware MPC solver initialized successfully")
                print(f"  Bitstream: {overlay_path}")
                print(f"  IP core: {self.HARDWARE_CONFIG['ip_name']}")
                print(f"  Horizon: {self.horizon}")
                
                if self.bitstream_path:
                    print(f"  Using custom bitstream path: {self.bitstream_path}")
                
            except ImportError as e:
                print(f"WARNING: Hardware solver not available: {e}")
                if TINYMPC_AVAILABLE:
                    print("  Falling back to software solver...")
                    self.solver_type = "software"
                    self._setup_software_solver()
                else:
                    print("  Cannot fall back to software solver: TinyMPC library not available")
                    raise RuntimeError(
                        "Neither hardware nor software MPC solver is available. "
                        "Please either:\n"
                        "1. Install PYNQ library and hw_interface for hardware solver, or\n"
                        "2. Install TinyMPC library for software solver"
                    )
            except Exception as e:
                print(f"WARNING: Hardware solver initialization failed: {e}")
                if TINYMPC_AVAILABLE:
                    print("  Falling back to software solver...")
                    self.solver_type = "software"
                    self._setup_software_solver()
                else:
                    print("  Cannot fall back to software solver: TinyMPC library not available")
                    raise RuntimeError(
                        "Neither hardware nor software MPC solver is available. "
                        "Please either:\n"
                        "1. Install PYNQ library and provide hardware bitstream for hardware solver, or\n"
                        "2. Install TinyMPC library for software solver"
                    )
        else:
            self._setup_software_solver()
    
    def _setup_software_solver(self):
        """Setup software TinyMPC solver"""
        if not TINYMPC_AVAILABLE:
            raise ImportError(
                "TinyMPC software library is not available. "
                "Please install TinyMPC or use hardware solver instead. "
                "To use hardware solver, set solver_type='hardware'."
            )
        
        # Setup software TinyMPC solver
        self.mpc = tinympc.TinyMPC()
        self.mpc.setup(self.A, self.B, self.Q, self.R, self.horizon)
        
        # Set bounds
        if 'u_min' in self.constraints and 'u_max' in self.constraints:
            self.mpc.u_min = self.constraints['u_min']
            self.mpc.u_max = self.constraints['u_max']
        
        if 'x_min' in self.constraints and 'x_max' in self.constraints:
            self.mpc.x_min = self.constraints['x_min']
            self.mpc.x_max = self.constraints['x_max']
    
    def set_control_frequency(self, control_freq: float):
        """Update control frequency and regenerate system matrices"""
        super().set_control_frequency(control_freq)
        
        # Recreate solver with new matrices
        if self.solver_type == "software":
            if not TINYMPC_AVAILABLE:
                print("WARNING: Cannot update control frequency: TinyMPC software library not available")
                return
                
            # Recreate software MPC solver with new matrices
            self.mpc = tinympc.TinyMPC()
            self.mpc.setup(self.A, self.B, self.Q, self.R, self.horizon)
            
            # Reset bounds
            if 'u_min' in self.constraints and 'u_max' in self.constraints:
                self.mpc.u_min = self.constraints['u_min']
                self.mpc.u_max = self.constraints['u_max']
            
            if 'x_min' in self.constraints and 'x_max' in self.constraints:
                self.mpc.x_min = self.constraints['x_min']
                self.mpc.x_max = self.constraints['x_max']
        else:
            # Hardware solver matrices are hardcoded in hardware
            # No need to recreate, but we can update clock frequency if needed
            try:
                if hasattr(self.mpc, 'set_clock_frequency'):
                    # Optionally adjust clock frequency based on control frequency
                    # Keep default for now
                    pass
            except:
                pass
    
    def simulate(self, steps: int = 200, 
                 initial_state: Optional[np.ndarray] = None,
                 verbose: bool = True):
        """Run simulation"""
        if initial_state is None:
            # Use realistic initial state noise
            initial_noise_std = self.dynamics_model.noise_model.get_initial_state_noise_std()
            state = self.X_ref[:, 0] + np.random.normal(0, initial_noise_std, 12)
        else:
            state = initial_state.copy()
        
        # Allow Z=0 operation for 2D trajectory testing
        # state[2] = max(state[2], 0.1)  # Keep above ground - commented out
        self.x_current = state
        self.x_history = [self.x_current.copy()]
        self.u_history = []
        self.cost_history = []
        
        for step in range(steps):
            # Get reference trajectory for the horizon
            if self.control_mode == ControlMode.REGULATOR:
                X_ref_horizon, U_ref_horizon = self._generate_regulator_references(step)
            else:
                X_ref_horizon, U_ref_horizon = self._generate_tracking_references(step)
            
            # Solve MPC problem
            try:
                if self.solver_type == "hardware":
                    # Hardware solver expects direct solve call with parameters
                    solution = self.mpc.solve(self.x_current, X_ref_horizon.T, U_ref_horizon.T)
                    if solution is not None and 'controls' in solution:
                        u_control = solution['controls'][0] if len(solution['controls']) > 0 else np.zeros(self.B.shape[1])
                    else:
                        u_control = np.zeros(self.B.shape[1])
                        if verbose and step < 10:
                            print(f"Warning: Hardware MPC solver failed at step {step}, using zero control")
                else:
                    # Software solver uses set/solve pattern
                    self.mpc.set_x0(self.x_current)
                    self.mpc.set_x_ref(X_ref_horizon)
                    self.mpc.set_u_ref(U_ref_horizon)
                    
                    solution = self.mpc.solve()
                    if solution is not None and 'controls' in solution:
                        u_control = solution['controls'].flatten()
                    else:
                        u_control = np.zeros(self.B.shape[1])
                        if verbose and step < 10:
                            print(f"Warning: Software MPC solver failed at step {step}, using zero control")
                            
            except Exception as e:
                if verbose and step < 10:
                    print(f"Warning: MPC solver error at step {step}: {e}")
                u_control = np.zeros(self.B.shape[1])
            
            # Compute cost
            if self.control_mode == ControlMode.REGULATOR:
                ref_point = self._get_target_reference_point(step)
                x_error = self.x_current - ref_point
            else:
                ref_start_idx = min(step, self.X_ref.shape[1] - 1)
                x_error = self.x_current - self.X_ref[:, ref_start_idx]
            
            cost = x_error.T @ self.Q @ x_error + u_control.T @ self.R @ u_control
            self.cost_history.append(cost)
            
            # Add noise and simulate forward
            u_noisy = self._add_actuator_noise(u_control)
            self.x_current = self._simulate_forward(self.x_current, u_noisy)
            
            # Apply state constraints
            self.x_current = np.clip(self.x_current, 
                                   self.constraints['x_min'], 
                                   self.constraints['x_max'])
            
            # Record
            self.x_history.append(self.x_current.copy())
            self.u_history.append(u_control.copy())
        
        if verbose:
            self._print_results()
    
    def _generate_tracking_references(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate references for traditional tracking mode"""
        ref_start_idx = min(step, self.X_ref.shape[1] - 1)
        
        # Create reference trajectory for the horizon
        X_ref_horizon = np.zeros((self.A.shape[0], self.horizon))
        U_ref_horizon = np.zeros((self.B.shape[1], self.horizon - 1))
        
        for i in range(self.horizon):
            ref_idx = min(ref_start_idx + i, self.X_ref.shape[1] - 1)
            X_ref_horizon[:, i] = self.X_ref[:, ref_idx]
        
        # Zero reference for control inputs (hover equilibrium)
        for i in range(self.horizon - 1):
            U_ref_horizon[:, i] = np.zeros(self.B.shape[1])
        
        return X_ref_horizon, U_ref_horizon
    
    def _generate_regulator_references(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate references for regulator mode"""
        ref_start_idx = min(step, self.X_ref.shape[1] - 1)
        
        X_ref_horizon = np.zeros((self.A.shape[0], self.horizon))
        U_ref_horizon = np.zeros((self.B.shape[1], self.horizon - 1))
        
        # Target state for this step
        target_state = self.X_ref[:, min(ref_start_idx, self.X_ref.shape[1] - 1)]
        
        # Generate trajectory from current state to target
        for i in range(self.horizon):
            alpha = min(i / (self.horizon - 1), 1.0) if self.horizon > 1 else 1.0
            X_ref_horizon[:, i] = (1 - alpha) * self.x_current + alpha * target_state
        
        # Zero reference for control inputs
        for i in range(self.horizon - 1):
            U_ref_horizon[:, i] = np.zeros(self.B.shape[1])
        
        return X_ref_horizon, U_ref_horizon
    
    def _get_target_reference_point(self, step: int) -> np.ndarray:
        """Get the target reference point for current step"""
        ref_idx = min(step, self.X_ref.shape[1] - 1)
        return self.X_ref[:, ref_idx]
    
    def _add_actuator_noise(self, u_control: np.ndarray) -> np.ndarray:
        """Add actuator noise to control inputs"""
        actuator_noise = np.random.normal(0, self.dynamics_model.noise_model.thrust_noise_std, 
                                        len(u_control))
        return u_control * (1 + actuator_noise)
    
    def _simulate_forward(self, x_current: np.ndarray, u_control: np.ndarray) -> np.ndarray:
        """Simulate system forward one time step"""
        # Add process noise
        process_noise_std = self.dynamics_model.noise_model.get_state_noise_std(self.dt)
        process_noise = np.random.normal(0, process_noise_std, len(x_current))
        
        # Add gravity disturbance
        gravity_disturbance = self.dynamics_model.gravity_disturbance
        
        # Forward simulation
        x_next = self.A @ x_current + self.B @ u_control + process_noise + gravity_disturbance
        
        return x_next
    
    def _print_results(self):
        """Print simulation results"""
        metrics = self._calculate_performance_metrics(np.array(self.x_history))
        
        print("Simulation Results:")
        print(f"  Final position error: {metrics['final_position_error']:.4f} m")
        print(f"  Average cost: {metrics['average_cost']:.3f}")
        print(f"  Max control input: {metrics['max_control_input']:.3f}")
        
        if metrics['constraint_violations'] == 0:
            print("  No constraint violations")
        else:
            print(f"  ✗ {metrics['constraint_violations']} constraint violations")
        
        # Generate plots
        self.plot_results()
        
        print(f"  Plot saved as 'simulation_results.png'")
        print(f"  Mean position error: {metrics['mean_position_error']:.4f} m")
        print(f"  RMSE position error: {metrics['rmse_position_error']:.4f} m")
        print(f"  Max position error: {metrics['max_position_error']:.4f} m")
    
    def plot_results(self, save_filename: str = 'simulation_results.png'):
        """Plot simulation results"""
        x_history = np.array(self.x_history)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time arrays
        t_sim = np.arange(len(x_history)) * self.dt
        t_ref = np.arange(min(len(x_history), self.X_ref.shape[1])) * self.dt
        
        # Plot 1: XY trajectory comparison
        ax1.plot(self.X_ref[0, :len(t_ref)], self.X_ref[1, :len(t_ref)], 
                'r--', linewidth=2, label='Reference', alpha=0.8)
        ax1.plot(x_history[:, 0], x_history[:, 1], 
                'b-', linewidth=2, label='Actual')
        ax1.scatter(x_history[0, 0], x_history[0, 1], 
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(x_history[-1, 0], x_history[-1, 1], 
                   c='red', s=100, marker='s', label='End', zorder=5)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Trajectory Tracking (XY View)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Position error over time
        position_errors = []
        position_error_components = []
        
        for i in range(len(x_history)):
            if i < self.X_ref.shape[1]:
                ref_pos = self.X_ref[:3, i]
            else:
                ref_pos = self.X_ref[:3, -1]
            
            actual_pos = x_history[i, :3]
            error = np.linalg.norm(actual_pos - ref_pos)
            position_errors.append(error)
            
            error_components = actual_pos - ref_pos
            position_error_components.append(error_components)
        
        ax2.plot(t_sim, position_errors, 'b-', linewidth=2, label='Position Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Tracking Error vs Time')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(position_errors)
        max_error = np.max(position_errors)
        
        ax2.axhline(y=mean_error, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_error:.3f} m')
        ax2.axhline(y=max_error, color='red', linestyle='--', alpha=0.7, 
                   label=f'Max: {max_error:.3f} m')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory instead of showing it
        print(f"  Plot saved as '{save_filename}'")

def create_simulator(dynamics_model: DynamicsModel,
                    X_ref: np.ndarray,
                    horizon: int = 50,
                    control_mode: ControlMode = ControlMode.TRACKING,
                    solver_type: str = "tinympc",
                    mpc_solver_type: str = "auto",
                    bitstream_path: str = None) -> MPCSimulator:
    """Factory function to create MPC simulators
    
    Args:
        dynamics_model: Dynamics model instance
        X_ref: Reference trajectory
        horizon: MPC horizon
        control_mode: Control mode (tracking or regulator)
        solver_type: Type of solver to use (legacy parameter, kept for compatibility)
        mpc_solver_type: MPC solver type ("auto", "software" or "hardware")
        bitstream_path: Path to FPGA bitstream file for hardware solver
    
    Returns:
        MPCSimulator instance
    """
    if solver_type == "tinympc":
        return TinyMPCSimulator(dynamics_model, X_ref, horizon, control_mode, mpc_solver_type, bitstream_path)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

# Legacy compatibility functions for existing code
class SimpleMPCSimulator(TinyMPCSimulator):
    """Legacy compatibility class"""
    
    def __init__(self, problem: Dict, solver_type: str = "auto", bitstream_path: str = None):
        # Extract parameters from problem dictionary (legacy format)
        dynamics_model = problem.get('dynamics_model')
        if dynamics_model is None:
            # Create dynamics model from legacy problem format
            from dynamics import LinearizedQuadcopterDynamics, CrazyflieParams
            params = problem.get('params', CrazyflieParams())
            noise_model = problem.get('noise_model')
            dynamics_model = LinearizedQuadcopterDynamics(params, noise_model)
            
            # Set gravity disturbance if available
            gravity_disturbance = problem['system'].get('gravity_disturbance', np.zeros(12))
            dynamics_model._gravity_disturbance = gravity_disturbance
        
        X_ref = problem['trajectory']['X_ref']
        horizon = problem['horizon']
        control_mode = problem.get('control_mode', ControlMode.TRACKING)
        
        super().__init__(dynamics_model, X_ref, horizon, control_mode, solver_type, bitstream_path)
        
        # Set control frequency from problem
        control_freq = problem['system']['control_freq']
        self.set_control_frequency(control_freq)

class RegulatorMPCSimulator(SimpleMPCSimulator):
    """Legacy compatibility class for regulator mode"""
    
    def __init__(self, problem: Dict, solver_type: str = "auto", bitstream_path: str = None):
        # Force regulator mode
        problem['control_mode'] = ControlMode.REGULATOR
        super().__init__(problem, solver_type, bitstream_path)