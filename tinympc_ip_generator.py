#!/usr/bin/env python3
"""
Consolidated TinyMPC HLS IP Core Generator
Contains all necessary components in a single file
"""

import numpy as np
import os
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# ============================================================================
# QUADCOPTER DYNAMICS COMPONENTS (from dynamics.py)
# ============================================================================

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

class LinearizedQuadcopterDynamics:
    """Linearized quadcopter dynamics model around hover condition"""
    
    def __init__(self, params: Optional[QuadcopterParams] = None):
        if params is None:
            params = CrazyflieParams()  # Default to Crazyflie
        self.params = params
        self.nstates = 12  # [x, y, z, phi_x, phi_y, phi_z, vx, vy, vz, wx, wy, wz]
        self.ninputs = 4   # [u1, u2, u3, u4]
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

def create_dynamics_model(platform: str = "crazyflie") -> LinearizedQuadcopterDynamics:
    """Factory function to create dynamics models"""
    if platform == "crazyflie":
        return LinearizedQuadcopterDynamics(CrazyflieParams())
    else:
        raise ValueError(f"Unknown platform: {platform}")

# ============================================================================
# TINYMPC REFERENCE SOLVER (from tinympcref.py)
# ============================================================================

class tinympcref:
    def __init__(self):
        self.nx = 0 # number of states
        self.nu = 0 # number of control inputs
        self.N = 0 # number of knotpoints in the horizon
        self.A = [] # state transition matrix
        self.B = [] # control matrix
        self.Q = [] # state cost matrix (diagonal)
        self.R = [] # input cost matrix (digaonal)
        self.rho = 0
        self.x_min = [] # lower bounds on state
        self.x_max = [] # upper bounds on state
        self.u_min = [] # lower bounds on input
        self.u_max = [] # upper bounds on input

        self.iter = 0
        self.solved = 0
        
        self.primal_residual_state = 0.0
        self.dual_residual_state = 0.0
        self.primal_residual_input = 0.0
        self.dual_residual_input = 0.0
        
    def setup(self, A, B, Q, R, N, rho=1.0,
              x_min=None, x_max=None, u_min=None, u_max=None, verbose=False, **settings):
        self.rho = rho
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] == B.shape[0]
        assert Q.shape[0] == Q.shape[1]
        assert A.shape[0] == Q.shape[0]
        assert R.shape[0] == R.shape[1]
        assert B.shape[1] == R.shape[0]

        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.nx = A.shape[0]
        self.nu = B.shape[1]

        assert N > 1
        self.N = N

        self.x = np.zeros((self.N, self.nx))
        self.u = np.zeros((self.N-1, self.nu))

        self.q = np.zeros((self.N, self.nx))
        self.r = np.zeros((self.N-1, self.nu))

        self.p = np.zeros((self.N, self.nx))
        self.d = np.zeros((self.N-1, self.nu))

        self.v = np.zeros((self.N, self.nx))
        self.vnew = np.zeros((self.N, self.nx))
        self.z = np.zeros((self.N-1, self.nu))
        self.znew = np.zeros((self.N-1, self.nu))

        self.g = np.zeros((self.N, self.nx))
        self.y = np.zeros((self.N-1, self.nu))

        self.Q = (self.Q + self.rho * np.eye(self.nx)).diagonal()
        self.R = (self.R + self.rho * np.eye(self.nu)).diagonal()

        if x_min is not None:
            self.x_min = np.tile(x_min, (self.N, 1))
        else:
            self.x_min = -np.inf * np.ones((self.N, self.nx))

        if x_max is not None:
            self.x_max = np.tile(x_max, (self.N, 1))
        else:
            self.x_max = np.inf * np.ones((self.N, self.nx))
            
        if u_min is not None:
            self.u_min = np.tile(u_min, (self.N-1, 1))
        else:
            self.u_min = -np.inf * np.ones((self.N-1, self.nu))

        if u_max is not None:
            self.u_max = np.tile(u_max, (self.N-1, 1))
        else:
            self.u_max = np.inf * np.ones((self.N-1, self.nu))

        self.Xref = np.zeros((self.N, self.nx))
        self.Uref = np.zeros((self.N-1, self.nu))

        self.Qu = np.zeros(self.nu)

        Q1 = np.diag(self.Q) + rho * np.eye(self.nx)
        R1 = np.diag(self.R) + rho * np.eye(self.nu)

        if(verbose):
            print('A = ', A)
            print('B = ', B)
            print('Q = ', self.Q)
            print('R = ', self.R)
            print('rho = ', rho)

        Ktp1 = np.zeros((self.nu, self.nx))
        Ptp1 = rho * np.eye(self.nx)
        Kinf = np.zeros((self.nu, self.nx))
        Pinf = np.zeros((self.nx, self.nx))

        for i in range(1000):
            Kinf = np.linalg.inv(R1 + B.T @ Ptp1 @ B) @ (B.T @ Ptp1 @ A)
            Pinf = Q1 + A.T @ Ptp1 @ (A - B @ Kinf)

            if np.max(np.abs(Kinf - Ktp1)) < 1e-5:
                if(verbose):
                    print('Kinf converged after %d iterations' % i)
                break

            Ktp1 = Kinf
            Ptp1 = Pinf

        Quu_inv = np.linalg.inv(R1 + B.T @ Pinf @ B)
        AmBKt = (A - B @ Kinf).T

        self.Kinf = Kinf
        self.Pinf = Pinf
        self.Quu_inv = Quu_inv
        self.AmBKt = AmBKt

        if(verbose):
            print('Kinf:', Kinf)
            print('Pinf:', Pinf)
            print('Quu_inv:', Quu_inv)
            print('AmBKt:', AmBKt)
        
        if 'abs_pri_tol' in settings:
            self.abs_pri_tol = settings.pop('abs_pri_tol')
        else:
            self.abs_pri_tol = 1e-3
        if 'abs_dua_tol' in settings:
            self.abs_dua_tol = settings.pop('abs_dua_tol')
        else:
            self.abs_dua_tol = 1e-3
        if 'max_iter' in settings:
            self.max_iter = settings.pop('max_iter')
        else:
            self.max_iter = 100
        if 'check_termination' in settings:
            self.check_termination = settings.pop('check_termination')
        else:
            self.check_termination = 10
        if 'en_state_bound' in settings:
            self.en_state_bound = 1 if settings.pop('en_state_bound') else 0
        if 'en_input_bound' in settings:
            self.en_input_bound = 1 if settings.pop('en_input_bound') else 0
        
    def set_x0(self, x0):
        assert len(x0) == self.nx
        self.x[0] = x0

    def set_x_ref(self, x_ref):
        self.Xref = x_ref
    
    def set_u_ref(self, u_ref):
        self.Uref = u_ref

    def _forward_pass(self):
        for i in range(self.N-1):
            self.u[i] = -self.Kinf @ self.x[i] - self.d[i]
            self.x[i+1] = self.A @ self.x[i] + self.B @ self.u[i]

    def _update_slack(self):
        self.znew = self.u + self.y
        self.vnew = self.x + self.g

        # box constraints on input
        self.znew = np.minimum(self.u_max, np.maximum(self.u_min, self.znew))

        self.vnew = np.minimum(self.x_max, np.maximum(self.x_min, self.vnew))
    
    def _update_dual(self):
        self.y = self.y + self.u - self.znew
        self.g = self.g + self.x - self.vnew

    def _update_linear_cost(self):
        for i in range(self.N-1):
            self.r[i] = -self.Uref[i] * self.R

        for i in range(self.N):
            self.q[i] = -self.Xref[i] * self.Q

        self.r -= self.rho * (self.znew - self.y)
        self.q -= self.rho * (self.vnew - self.g)

        # self.p[self.N-1] = -self.Xref[self.N-1].T @ self.Pinf
        self.p[self.N-1] = -self.Pinf.T @ self.Xref[self.N-1]

        # print('Xref:', self.Xref)
        self.p[self.N-1] -= self.rho * (self.vnew[self.N-1] - self.g[self.N-1])

    def _backward_pass(self):
        for i in range(self.N-2, -1, -1):
            self.d[i] = self.Quu_inv @ (self.B.T @ self.p[i+1] + self.r[i])
            self.p[i] = self.q[i] + self.AmBKt @ self.p[i+1] - self.Kinf.T @ self.r[i]

    def _check_termination(self):
        if self.check_termination == 0:
            return False

        if self.iter % self.check_termination == 0:
            self.primal_residual_state = np.max(np.abs(self.x - self.vnew))
            self.dual_residual_state = np.max(np.abs(self.v - self.vnew)) * self.rho
            self.primal_residual_input = np.max(np.abs(self.u - self.znew))
            self.dual_residual_input = np.max(np.abs(self.z - self.znew)) * self.rho

            if (self.primal_residual_state < self.abs_pri_tol and
                self.primal_residual_input < self.abs_pri_tol and
                self.dual_residual_state < self.abs_dua_tol and
                self.dual_residual_input < self.abs_dua_tol):
                return True
        return False

    def solve(self):
        self.iter = 0
        self.solved = 0

        for i in range(self.max_iter):
            self._forward_pass()
            self._update_slack()

            self._update_dual()
            self._update_linear_cost()

            self.iter += 1
            if self._check_termination():
                self.solved = 1
                break
            
            self.v = self.vnew
            self.z = self.znew

            self._backward_pass()
        
        self.x = self.vnew
        self.u = self.znew

# ============================================================================
# HLS FUNCTION GENERATOR (from hls_generator_v2.py)
# ============================================================================

class BaseFunctionGenerator:
    """Base class for HLS function generation"""
    
    def __init__(self, solver, precision='float'):
        self.solver = solver
        self.precision = precision
        self.nx = solver.nx
        self.nu = solver.nu
        self.N = solver.N
        
        self.data_type = 'float' if precision == 'float' else 'double'
        self.suffix = 'f' if precision == 'float' else ''
        self.tolerance = '1e-3f' if precision == 'float' else '1e-6'
        
    def _format_value(self, value):
        """Format numerical value with appropriate precision"""
        if self.precision == 'float':
            return f"{value:.10f}{self.suffix}"
        else:
            return f"{value:.15f}{self.suffix}"
    
    def _get_optimized_expr(self, matrix, row, col, var_name):
        """Generate optimized expression eliminating special values"""
        value = matrix[row, col]
        
        if abs(value) < 1e-10:
            return None
        elif abs(value - 1.0) < 1e-10:
            return var_name
        elif abs(value + 1.0) < 1e-10:
            return f"-{var_name}"
        else:
            return f"{self._format_value(value)} * {var_name}"

class TinyMPCFunctionGenerator:
    """Generator for individual TinyMPC functions"""
    
    def __init__(self, function_name, N=5, control_freq=100.0, precision='float'):
        """Initialize for specific function"""
        dynamics = create_dynamics_model("crazyflie")
        A, B = dynamics.generate_system_matrices(control_freq)
        Q, R = dynamics.generate_cost_matrices()
        
        self.solver = tinympcref()
        self.solver.setup(A, B, Q, R, N, rho=1.0)
        
        self.function_name = function_name
        self.precision = precision
        self.data_type = 'float' if precision == 'float' else 'double'
        self.suffix = 'f' if precision == 'float' else ''
        self.tolerance = '1e-3f' if precision == 'float' else '1e-6'
        
        self.base_gen = BaseFunctionGenerator(self.solver, precision)
    
    def generate_implementation(self):
        """Generate implementation for specific function"""
        if self.function_name == 'forward_pass':
            return self._generate_forward_pass()
        elif self.function_name == 'backward_pass':
            return self._generate_backward_pass()
        elif self.function_name == 'update_slack':
            return self._generate_update_slack()
        elif self.function_name == 'update_dual':
            return self._generate_update_dual()
        elif self.function_name == 'update_linear_cost':
            return self._generate_update_linear_cost()
        else:
            return f'#include "{self.function_name}.h"\n\n// Implementation placeholder\n'
    
    def _generate_forward_pass(self):
        """Generate forward pass implementation"""
        code = f'#include "tinympc_solver.h"\n\n'
        code += """void forward_pass(
    data_t x[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t d[N_MINUS_1][NU]
) {
    forward_loop: for (int k = 0; k < N_MINUS_1; k++) {
#pragma HLS PIPELINE
        
        // Control: u[k] = -Kinf @ x[k] - d[k]
"""
        
        # Generate control computations
        for i in range(self.solver.nu):
            code += f"        u[k][{i}] = -("
            terms = []
            for j in range(self.solver.nx):
                expr = self.base_gen._get_optimized_expr(self.solver.Kinf, i, j, f"x[k][{j}]")
                if expr is not None:
                    terms.append(expr)
            code += " + ".join(terms) if terms else f"0.0{self.suffix}"
            code += f") - d[k][{i}];\n"
        
        code += "\n        // State: x[k+1] = A @ x[k] + B @ u[k]\n"
        
        # Generate state updates
        for i in range(self.solver.nx):
            code += f"        x[k+1][{i}] = "
            terms = []
            
            # A @ x[k] terms
            for j in range(self.solver.nx):
                expr = self.base_gen._get_optimized_expr(self.solver.A, i, j, f"x[k][{j}]")
                if expr is not None:
                    terms.append(expr)
            
            # B @ u[k] terms
            for j in range(self.solver.nu):
                expr = self.base_gen._get_optimized_expr(self.solver.B, i, j, f"u[k][{j}]")
                if expr is not None:
                    terms.append(expr)
            
            code += " + ".join(terms) if terms else f"0.0{self.suffix}"
            code += ";\n"
        
        code += "    }\n}"
        return code
    
    def _generate_backward_pass(self):
        """Generate backward pass implementation"""
        code = f'#include "tinympc_solver.h"\n\n'
        code += f"""void backward_pass(
    data_t d[N_MINUS_1][NU],
    data_t p[N][NX],
    data_t q[N][NX],
    data_t r[N_MINUS_1][NU]
) {{
    backward_loop: for (int k = N-2; k >= 0; k--) {{
#pragma HLS PIPELINE
        
        // Compute d[k] = Quu_inv @ (B.T @ p[k+1] + r[k])
"""
        
        # Generate d[k] computation: expand Quu_inv @ (B.T @ p[k+1] + r[k])
        for j in range(self.solver.nu):
            code += f"        d[k][{j}] = "
            terms = []
            
            # For each d[k][j], compute sum over l of Quu_inv[j,l] * (sum_i B.T[l,i] * p[k+1][i] + r[k][l])
            for l in range(self.solver.nu):
                if abs(self.solver.Quu_inv[j, l]) > 1e-10:
                    # Compute B.T @ p[k+1] for column l
                    bt_terms = []
                    for i in range(self.solver.nx):
                        bt_expr = self.base_gen._get_optimized_expr(self.solver.B.T, l, i, f"p[k+1][{i}]")
                        if bt_expr is not None:
                            bt_terms.append(bt_expr)
                    
                    # Combine (B.T @ p[k+1])[l] + r[k][l]
                    if bt_terms:
                        inner_expr = f"({' + '.join(bt_terms)} + r[k][{l}])"
                    else:
                        inner_expr = f"r[k][{l}]"
                    
                    # Multiply by Quu_inv[j,l]
                    quu_expr = self.base_gen._get_optimized_expr(self.solver.Quu_inv, j, l, inner_expr)
                    if quu_expr is not None:
                        terms.append(quu_expr)
            
            code += " + ".join(terms) if terms else self.base_gen._format_value(0.0)
            code += ";\n"
        
        code += "\n        // Compute p[k] = q[k] + AmBKt @ p[k+1] - Kinf.T @ r[k]\n"
        
        # Generate p[k] computation
        for i in range(self.solver.nx):
            code += f"        p[k][{i}] = q[k][{i}]"
            
            # Add AmBKt @ p[k+1] terms
            for j in range(self.solver.nx):
                expr = self.base_gen._get_optimized_expr(self.solver.AmBKt, i, j, f"p[k+1][{j}]")
                if expr is not None:
                    code += f" + {expr}"
            
            # Subtract Kinf.T @ r[k] terms  
            for j in range(self.solver.nu):
                expr = self.base_gen._get_optimized_expr(self.solver.Kinf.T, i, j, f"r[k][{j}]")
                if expr is not None:
                    code += f" - {expr}"
            
            code += ";\n"
        
        code += "    }\n}"
        return code
    
    def _generate_update_slack(self):
        """Generate update slack implementation"""
        return f'#include "tinympc_solver.h"\n#include <algorithm>\n\nvoid update_slack(\n    data_t znew[N_MINUS_1][NU],\n    data_t vnew[N][NX],\n    data_t u[N_MINUS_1][NU],\n    data_t x[N][NX],\n    data_t y[N_MINUS_1][NU],\n    data_t g[N][NX],\n    data_t u_min[N_MINUS_1][NU],\n    data_t u_max[N_MINUS_1][NU],\n    data_t x_min[N][NX],\n    data_t x_max[N][NX]\n) {{\n    // Update input slack variables\n    update_u_slack: for (int k = 0; k < N_MINUS_1; k++) {{\n#pragma HLS PIPELINE\n        for (int j = 0; j < NU; j++) {{\n#pragma HLS UNROLL\n            data_t temp_u = u[k][j] + y[k][j];\n            znew[k][j] = (temp_u > u_max[k][j]) ? u_max[k][j] : \n                        (temp_u < u_min[k][j]) ? u_min[k][j] : temp_u;\n        }}\n    }}\n    \n    // Update state slack variables\n    update_x_slack: for (int k = 0; k < N; k++) {{\n#pragma HLS PIPELINE\n        for (int i = 0; i < NX; i++) {{\n#pragma HLS UNROLL\n            data_t temp_x = x[k][i] + g[k][i];\n            vnew[k][i] = (temp_x > x_max[k][i]) ? x_max[k][i] : \n                        (temp_x < x_min[k][i]) ? x_min[k][i] : temp_x;\n        }}\n    }}\n}}'
    
    def _generate_update_dual(self):
        """Generate update dual implementation"""
        return f'#include "tinympc_solver.h"\n\nvoid update_dual(\n    data_t y[N_MINUS_1][NU],\n    data_t g[N][NX],\n    data_t u[N_MINUS_1][NU],\n    data_t x[N][NX],\n    data_t znew[N_MINUS_1][NU],\n    data_t vnew[N][NX]\n) {{\n    // Update input dual variables\n    update_y_dual: for (int k = 0; k < N_MINUS_1; k++) {{\n#pragma HLS PIPELINE\n        for (int j = 0; j < NU; j++) {{\n#pragma HLS UNROLL\n            y[k][j] = y[k][j] + u[k][j] - znew[k][j];\n        }}\n    }}\n    \n    // Update state dual variables\n    update_g_dual: for (int k = 0; k < N; k++) {{\n#pragma HLS PIPELINE\n        for (int i = 0; i < NX; i++) {{\n#pragma HLS UNROLL\n            g[k][i] = g[k][i] + x[k][i] - vnew[k][i];\n        }}\n    }}\n}}'
    
    def _generate_update_linear_cost(self):
        """Generate update linear cost implementation with hardcoded parameters"""
        code = f'#include "tinympc_solver.h"\n\n'
        
        # Get hardcoded values
        rho_val = self.base_gen._format_value(self.solver.rho)
        
        code += f"""void update_linear_cost(
    data_t r[N_MINUS_1][NU],
    data_t q[N][NX],
    data_t p[N][NX],
    data_t Uref[N_MINUS_1][NU],
    data_t Xref[N][NX],
    data_t znew[N_MINUS_1][NU],
    data_t vnew[N][NX],
    data_t y[N_MINUS_1][NU],
    data_t g[N][NX]
) {{
    // Update input linear cost with hardcoded R values
    update_r_cost: for (int k = 0; k < N_MINUS_1; k++) {{
#pragma HLS PIPELINE
"""
        
        # Generate unrolled R computations
        for j in range(self.solver.nu):
            r_val = self.base_gen._format_value(self.solver.R[j])
            code += f"        r[k][{j}] = -Uref[k][{j}] * {r_val} - {rho_val} * (znew[k][{j}] - y[k][{j}]);\n"
        
        code += """    }
    
    // Update state linear cost with hardcoded Q values
    update_q_cost: for (int k = 0; k < N_MINUS_1; k++) {
#pragma HLS PIPELINE
"""
        
        # Generate unrolled Q computations
        for i in range(self.solver.nx):
            q_val = self.base_gen._format_value(self.solver.Q[i])
            code += f"        q[k][{i}] = -Xref[k][{i}] * {q_val} - {rho_val} * (vnew[k][{i}] - g[k][{i}]);\n"
        
        code += """    }
    
    // Terminal state cost (k = N-1) with hardcoded Q values
"""
        
        for i in range(self.solver.nx):
            q_val = self.base_gen._format_value(self.solver.Q[i])
            code += f"    q[N-1][{i}] = -Xref[N-1][{i}] * {q_val} - {rho_val} * (vnew[N-1][{i}] - g[N-1][{i}]);\n"
        
        code += "\n    // Terminal costate: p[N-1] = -Pinf.T @ Xref[N-1] - rho * (vnew[N-1] - g[N-1])\n"
        
        # Generate hardcoded Pinf.T @ Xref[N-1] multiplication  
        for i in range(self.solver.nx):
            code += f"    p[N-1][{i}] = "
            terms = []
            
            # Add -Pinf.T @ Xref[N-1] terms
            for j in range(self.solver.nx):
                expr = self.base_gen._get_optimized_expr(-self.solver.Pinf.T, i, j, f"Xref[N-1][{j}]")
                if expr is not None:
                    terms.append(expr)
            
            # Add -rho * (vnew[N-1][i] - g[N-1][i]) term with hardcoded rho
            terms.append(f"-{rho_val} * (vnew[N-1][{i}] - g[N-1][{i}])")
            
            code += " + ".join(terms) if terms else f"-{rho_val} * (vnew[N-1][{i}] - g[N-1][{i}])"
            code += ";\n"
        
        code += "}"
        return code

# ============================================================================
# MAIN IP GENERATOR CLASS
# ============================================================================

class TinyMPCIPGenerator:
    """Generator for complete TinyMPC HLS IP core"""
    
    def __init__(self, N=5, control_freq=100.0, precision='float', max_iter=10):
        """Initialize IP core generator"""
        dynamics = create_dynamics_model("crazyflie")
        A, B = dynamics.generate_system_matrices(control_freq)
        Q, R = dynamics.generate_cost_matrices()
        
        # Get constraints from dynamics
        self.constraints = dynamics.generate_constraints()
        
        self.solver = tinympcref()
        self.solver.setup(A, B, Q, R, N, rho=1.0, 
                x_min=self.constraints['x_min'], 
                x_max=self.constraints['x_max'], 
                u_min=self.constraints['u_min'], 
                u_max=self.constraints['u_max'], 
                max_iter=max_iter, 
                check_termination=10)
        
        self.N = N
        self.max_iter = max_iter
        self.precision = precision
        self.data_type = 'float' if precision == 'float' else 'double'
        self.suffix = 'f' if precision == 'float' else ''
        self.tolerance = '1e-3f' if precision == 'float' else '1e-6'
        
    def _format_value(self, value):
        """Format numerical value with appropriate precision"""
        if self.precision == 'float':
            return f"{value:.10f}{self.suffix}"
        else:
            return f"{value:.15f}{self.suffix}"
    
    def _get_optimized_expr(self, matrix, row, col, var_name):
        """Generate optimized expression eliminating special values"""
        value = matrix[row, col]
        
        if abs(value) < 1e-10:
            return None
        elif abs(value - 1.0) < 1e-10:
            return var_name
        elif abs(value + 1.0) < 1e-10:
            return f"-{var_name}"
        else:
            return f"{self._format_value(value)} * {var_name}"
    
    def generate_header(self):
        """Generate main IP core header"""
        return f"""#ifndef TINYMPC_SOLVER_H
#define TINYMPC_SOLVER_H

#define NX {self.solver.nx}
#define NU {self.solver.nu}
#define N {self.solver.N}
#define N_MINUS_1 {self.solver.N-1}
#define MAX_ITER {self.max_iter}

// Memory layout constants
#define X0_OFFSET 0
#define XREF_OFFSET {self.solver.nx}
#define UREF_OFFSET {self.solver.nx + self.solver.N * self.solver.nx}
#define X_OUT_OFFSET {self.solver.nx + self.solver.N * self.solver.nx + (self.solver.N-1) * self.solver.nu}
#define U_OUT_OFFSET {self.solver.nx + 2 * self.solver.N * self.solver.nx + (self.solver.N-1) * self.solver.nu}

#define MAIN_MEMORY_SIZE {self.solver.nx + 2 * self.solver.N * self.solver.nx + 2 * (self.solver.N-1) * self.solver.nu}

typedef {self.data_type} data_t;

// Main IP core interface
void tinympc_solver(
    data_t main_memory[MAIN_MEMORY_SIZE],
    int max_iter,
    int check_termination_iter
);

// Sub-function declarations
void forward_pass(
    data_t x[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t d[N_MINUS_1][NU]
);

void backward_pass(
    data_t d[N_MINUS_1][NU],
    data_t p[N][NX],
    data_t q[N][NX],
    data_t r[N_MINUS_1][NU]
);

void update_slack(
    data_t znew[N_MINUS_1][NU],
    data_t vnew[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t x[N][NX],
    data_t y[N_MINUS_1][NU],
    data_t g[N][NX],
    data_t u_min[N_MINUS_1][NU],
    data_t u_max[N_MINUS_1][NU],
    data_t x_min[N][NX],
    data_t x_max[N][NX]
);

void update_dual(
    data_t y[N_MINUS_1][NU],
    data_t g[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t x[N][NX],
    data_t znew[N_MINUS_1][NU],
    data_t vnew[N][NX]
);

void update_linear_cost(
    data_t r[N_MINUS_1][NU],
    data_t q[N][NX],
    data_t p[N][NX],
    data_t Uref[N_MINUS_1][NU],
    data_t Xref[N][NX],
    data_t znew[N_MINUS_1][NU],
    data_t vnew[N][NX],
    data_t y[N_MINUS_1][NU],
    data_t g[N][NX]
);

bool check_termination(
    data_t x[N][NX],
    data_t vnew[N][NX],
    data_t v[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t znew[N_MINUS_1][NU],
    data_t z[N_MINUS_1][NU]
);

#endif"""

    def generate_main_solver(self):
        """Generate main TinyMPC solver implementation"""
        # Generate constraint arrays with dynamic size based on actual N
        u_min_vals = ", ".join([self._format_value(v) for v in self.constraints['u_min']])
        u_max_vals = ", ".join([self._format_value(v) for v in self.constraints['u_max']])
        x_min_vals = ", ".join([self._format_value(v) for v in self.constraints['x_min']])
        x_max_vals = ", ".join([self._format_value(v) for v in self.constraints['x_max']])
        
        # Generate U_MIN and U_MAX arrays with correct N_MINUS_1 size
        u_min_rows = []
        u_max_rows = []
        for k in range(self.N - 1):
            u_min_rows.append(f"    {{{u_min_vals}}}")
            u_max_rows.append(f"    {{{u_max_vals}}}")
        
        # Generate X_MIN and X_MAX arrays with correct N size
        x_min_rows = []
        x_max_rows = []
        for k in range(self.N):
            x_min_rows.append(f"    {{{x_min_vals}}}")
            x_max_rows.append(f"    {{{x_max_vals}}}")
        
        return f"""#include "tinympc_solver.h"
#include <string.h>

// Hardcoded constraints as ROM (constant arrays) with dynamic size N={self.N}
const data_t U_MIN[N_MINUS_1][NU] = {{
{",\n".join(u_min_rows)}
}};
#pragma HLS ARRAY_PARTITION variable=U_MIN complete dim=2

const data_t U_MAX[N_MINUS_1][NU] = {{
{",\n".join(u_max_rows)}
}};
#pragma HLS ARRAY_PARTITION variable=U_MAX complete dim=2

const data_t X_MIN[N][NX] = {{
{",\n".join(x_min_rows)}
}};
#pragma HLS ARRAY_PARTITION variable=X_MIN complete dim=2

const data_t X_MAX[N][NX] = {{
{",\n".join(x_max_rows)}
}};
#pragma HLS ARRAY_PARTITION variable=X_MAX complete dim=2

void tinympc_solver(
    data_t main_memory[MAIN_MEMORY_SIZE],
    int max_iter,
    int check_termination_iter
) {{
#pragma HLS INTERFACE m_axi port=main_memory bundle=gmem
#pragma HLS INTERFACE s_axilite port=max_iter
#pragma HLS INTERFACE s_axilite port=check_termination_iter
#pragma HLS INTERFACE s_axilite port=return

    // Local workspace variables with partition pragmas
    data_t x[N][NX];
    #pragma HLS ARRAY_PARTITION variable=x complete dim=2
    data_t u[N_MINUS_1][NU];
    #pragma HLS ARRAY_PARTITION variable=u complete dim=2
    data_t d[N_MINUS_1][NU];
    #pragma HLS ARRAY_PARTITION variable=d complete dim=2
    data_t vnew[N][NX];
    #pragma HLS ARRAY_PARTITION variable=vnew complete dim=2
    data_t znew[N_MINUS_1][NU];
    #pragma HLS ARRAY_PARTITION variable=znew complete dim=2
    data_t v[N][NX];
    #pragma HLS ARRAY_PARTITION variable=v complete dim=2
    data_t z[N_MINUS_1][NU];
    #pragma HLS ARRAY_PARTITION variable=z complete dim=2
    data_t y[N_MINUS_1][NU];
    #pragma HLS ARRAY_PARTITION variable=y complete dim=2
    data_t g[N][NX];
    #pragma HLS ARRAY_PARTITION variable=g complete dim=2
    data_t p[N][NX];
    #pragma HLS ARRAY_PARTITION variable=p complete dim=2
    data_t q[N][NX];
    #pragma HLS ARRAY_PARTITION variable=q complete dim=2
    data_t r[N_MINUS_1][NU];
    #pragma HLS ARRAY_PARTITION variable=r complete dim=2
    data_t Xref[N][NX];
    #pragma HLS ARRAY_PARTITION variable=Xref complete dim=2
    data_t Uref[N_MINUS_1][NU];
    #pragma HLS ARRAY_PARTITION variable=Uref complete dim=2
    
    // Load input data from main memory
    load_inputs: for (int i = 0; i < NX; i++) {{
        x[0][i] = main_memory[X0_OFFSET + i];  // Initial state
    }}
    
    load_xref: for (int k = 0; k < N; k++) {{
        for (int i = 0; i < NX; i++) {{
            Xref[k][i] = main_memory[XREF_OFFSET + k*NX + i];
        }}
    }}
    
    load_uref: for (int k = 0; k < N_MINUS_1; k++) {{
        for (int j = 0; j < NU; j++) {{
            Uref[k][j] = main_memory[UREF_OFFSET + k*NU + j];
        }}
    }}
    
    // Initialize ALL workspace variables to prevent non-deterministic behavior
    init_workspace: for (int k = 0; k < N_MINUS_1; k++) {{
        for (int j = 0; j < NU; j++) {{
            z[k][j] = {self._format_value(0.0)};
            znew[k][j] = {self._format_value(0.0)};
            y[k][j] = {self._format_value(0.0)};
            u[k][j] = {self._format_value(0.0)};
            d[k][j] = {self._format_value(0.0)};
            r[k][j] = {self._format_value(0.0)};
        }}
    }}
    
    init_state_workspace: for (int k = 0; k < N; k++) {{
        for (int i = 0; i < NX; i++) {{
            v[k][i] = {self._format_value(0.0)};
            vnew[k][i] = {self._format_value(0.0)};
            g[k][i] = {self._format_value(0.0)};
            p[k][i] = {self._format_value(0.0)};
            q[k][i] = {self._format_value(0.0)};
            if (k > 0) {{
                x[k][i] = {self._format_value(0.0)};
            }}
        }}
    }}
    
    // ADMM iterations
    admm_loop: for (int iter = 0; iter < max_iter; iter++) {{
#pragma HLS PIPELINE off
        
        // Step 1: Forward pass
        forward_pass(x, u, d);
        
        // Step 2: Update slack variables (using global ROM constraints)
        update_slack(znew, vnew, u, x, y, g, 
                    (data_t(*)[NU])U_MIN, (data_t(*)[NU])U_MAX,
                    (data_t(*)[NX])X_MIN, (data_t(*)[NX])X_MAX);
        
        // Step 3: Update dual variables
        update_dual(y, g, u, x, znew, vnew);
        
        // Step 4: Update linear cost terms
        update_linear_cost(r, q, p, Uref, Xref, znew, vnew, y, g);
        
        // Step 5: Check termination (optional)
        if ((iter + 1) % check_termination_iter == 0) {{
            if (check_termination(x, vnew, v, u, znew, z)) {{
                break;
            }}
        }}
        
        // Update workspace for next iteration
        copy_workspace: for (int k = 0; k < N; k++) {{
            for (int i = 0; i < NX; i++) {{
                if (k < N_MINUS_1) {{
                    for (int j = 0; j < NU; j++) {{
                        z[k][j] = znew[k][j];
                    }}
                }}
                v[k][i] = vnew[k][i];
            }}
        }}
        
        // Step 6: Backward pass
        backward_pass(d, p, q, r);
    }}
    
    // Store results back to main memory
    store_x_out: for (int k = 0; k < N; k++) {{
        for (int i = 0; i < NX; i++) {{
            main_memory[X_OUT_OFFSET + k*NX + i] = vnew[k][i];
        }}
    }}
    
    store_u_out: for (int k = 0; k < N_MINUS_1; k++) {{
        for (int j = 0; j < NU; j++) {{
            main_memory[U_OUT_OFFSET + k*NU + j] = znew[k][j];
        }}
    }}
}}"""

    def generate_test_data(self):
        """Generate test data using tinympcref.py reference"""
        test_cases = []
        results = []
        
        # Test case 1: Hover stability
        x0_hover = np.array([0.01, 0.01, 0.05] + [0.0]*(self.solver.nx-3), dtype=np.float32)
        xref_hover = np.zeros((self.N, self.solver.nx), dtype=np.float32)
        uref_hover = np.zeros((self.N-1, self.solver.nu), dtype=np.float32)
        uref_hover[:, 0] = 0.5  # Hover thrust
        
        test_cases.append({
            'name': 'hover_stability',
            'x0': x0_hover,
            'xref': xref_hover,
            'uref': uref_hover
        })
        
        # Test case 2: Position tracking  
        x0_track = np.array([0.01, 0.01, 0.05] + [0.0]*(self.solver.nx-3), dtype=np.float32)
        xref_track = np.zeros((self.N, self.solver.nx), dtype=np.float32)
        xref_track[:, :3] = 1.0  # Move to (1,1,1)
        uref_track = np.zeros((self.N-1, self.solver.nu), dtype=np.float32)
        uref_track[:, 0] = 0.5
        
        test_cases.append({
            'name': 'position_tracking', 
            'x0': x0_track,
            'xref': xref_track,
            'uref': uref_track
        })
        
        # Test case 3: Disturbance rejection
        # np.random.seed(42)  # Fixed seed for deterministic results
        x0_disturb = np.random.randn(self.solver.nx).astype(np.float32) * 0.05
        xref_disturb = np.zeros((self.N, self.solver.nx), dtype=np.float32)
        uref_disturb = np.zeros((self.N-1, self.solver.nu), dtype=np.float32)
        uref_disturb[:, 0] = 0.5
        
        test_cases.append({
            'name': 'disturbance_rejection',
            'x0': x0_disturb, 
            'xref': xref_disturb,
            'uref': uref_disturb
        })
        
        # Generate reference solutions
        for case in test_cases:
            # Setup solver with test case data
            self.solver.set_x0(case['x0'])
            self.solver.set_x_ref(case['xref'])
            self.solver.set_u_ref(case['uref'])
            
            # Run solver with termination checking enabled
            self.solver.solve()
            
            # Collect results including actual iteration count
            results.append({
                'x_out': self.solver.vnew.copy(),
                'u_out': self.solver.znew.copy(),
                'final_iter': self.solver.iter  # Actual iterations run
            })
        
        return self._generate_test_data_header(test_cases, results)
    
    def _generate_test_data_header(self, test_cases, results):
        """Generate test_data.h file"""
        header_parts = []
        header_parts.append("#ifndef TEST_DATA_H")
        header_parts.append("#define TEST_DATA_H")
        header_parts.append("")
        header_parts.append('#include "tinympc_solver.h"')
        header_parts.append("")
        header_parts.append("#define NUM_TESTS 3")
        header_parts.append("")
        
        # Test initial states
        header_parts.append(f"{self.data_type} test_x0[NUM_TESTS][NX] = {{")
        for i, case in enumerate(test_cases):
            values = ", ".join([self._format_value(case['x0'][j]) for j in range(self.solver.nx)])
            header_parts.append(f"    {{{values}}},  // {case['name']}")
        header_parts.append("};")
        header_parts.append("")
        
        # Test reference trajectories
        header_parts.append(f"{self.data_type} test_xref[NUM_TESTS][N][NX] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.N):
                values = ", ".join([self._format_value(case['xref'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test reference inputs
        header_parts.append(f"{self.data_type} test_uref[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.N-1):
                values = ", ".join([self._format_value(case['uref'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Expected state outputs
        header_parts.append(f"{self.data_type} expected_x_out[NUM_TESTS][N][NX] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.N):
                values = ", ".join([self._format_value(result['x_out'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Expected control outputs
        header_parts.append(f"{self.data_type} expected_u_out[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.N-1):
                values = ", ".join([self._format_value(result['u_out'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Expected iteration counts
        header_parts.append("int expected_iterations[NUM_TESTS] = {")
        for i, result in enumerate(results):
            header_parts.append(f"    {result['final_iter']},  // {test_cases[i]['name']}")
        header_parts.append("};")
        header_parts.append("")
        header_parts.append("#endif")
        
        return "\n".join(header_parts)

    def generate_subfunctions(self):
        """Generate all sub-function implementations"""
        functions = ['forward_pass', 'backward_pass', 'update_slack', 'update_dual', 'update_linear_cost']
        implementations = []
        
        for func_name in functions:
            gen = TinyMPCFunctionGenerator(func_name, self.N, precision=self.precision)
            impl = gen.generate_implementation()
            # Remove the include line since we'll include everything in one file
            impl = impl.split('\n', 1)[1] if impl.startswith('#include') else impl
            implementations.append(f"// {func_name.upper()} IMPLEMENTATION\n{impl}\n")
        
        # Add check_termination function implementation
        check_term_impl = self._generate_check_termination()
        implementations.append(f"// CHECK_TERMINATION IMPLEMENTATION\n{check_term_impl}\n")
        
        return "\n".join(implementations)

    def _generate_check_termination(self):
        """Generate check_termination function implementation"""
        # Get rho value from solver
        rho_val = self.solver.rho if hasattr(self.solver, 'rho') else 1.0
        
        return f"""bool check_termination(
    data_t x[N][NX],
    data_t vnew[N][NX], 
    data_t v[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t znew[N_MINUS_1][NU],
    data_t z[N_MINUS_1][NU]
) {{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=x complete dim=2
#pragma HLS ARRAY_PARTITION variable=vnew complete dim=2  
#pragma HLS ARRAY_PARTITION variable=v complete dim=2
#pragma HLS ARRAY_PARTITION variable=u complete dim=2
#pragma HLS ARRAY_PARTITION variable=znew complete dim=2
#pragma HLS ARRAY_PARTITION variable=z complete dim=2

    // Tolerance values (matching tinympcref.py defaults)
    const data_t abs_pri_tol = {self._format_value(1e-3)};
    const data_t abs_dua_tol = {self._format_value(1e-3)};
    const data_t rho = {self._format_value(rho_val)};
    
    // Compute primal residual for states (max abs difference between x and vnew)
    data_t primal_residual_state = {self._format_value(0.0)};
    compute_primal_state: for (int k = 0; k < N; k++) {{
        for (int i = 0; i < NX; i++) {{
            data_t diff = x[k][i] - vnew[k][i];
            if (diff < 0) diff = -diff;  // abs()
            if (diff > primal_residual_state) primal_residual_state = diff;
        }}
    }}
    
    // Compute dual residual for states (max abs difference between v and vnew) * rho
    data_t dual_residual_state = {self._format_value(0.0)};
    compute_dual_state: for (int k = 0; k < N; k++) {{
        for (int i = 0; i < NX; i++) {{
            data_t diff = v[k][i] - vnew[k][i];
            if (diff < 0) diff = -diff;  // abs()
            if (diff > dual_residual_state) dual_residual_state = diff;
        }}
    }}
    dual_residual_state = dual_residual_state * rho;
    
    // Compute primal residual for inputs (max abs difference between u and znew)
    data_t primal_residual_input = {self._format_value(0.0)};
    compute_primal_input: for (int k = 0; k < N_MINUS_1; k++) {{
        for (int j = 0; j < NU; j++) {{
            data_t diff = u[k][j] - znew[k][j];
            if (diff < 0) diff = -diff;  // abs()
            if (diff > primal_residual_input) primal_residual_input = diff;
        }}
    }}
    
    // Compute dual residual for inputs (max abs difference between z and znew) * rho  
    data_t dual_residual_input = {self._format_value(0.0)};
    compute_dual_input: for (int k = 0; k < N_MINUS_1; k++) {{
        for (int j = 0; j < NU; j++) {{
            data_t diff = z[k][j] - znew[k][j];
            if (diff < 0) diff = -diff;  // abs()
            if (diff > dual_residual_input) dual_residual_input = diff;
        }}
    }}
    dual_residual_input = dual_residual_input * rho;
    
    // Check convergence criteria (all four residuals must be below their tolerances)
    bool converged = (primal_residual_state < abs_pri_tol) && 
                     (primal_residual_input < abs_pri_tol) &&
                     (dual_residual_state < abs_dua_tol) && 
                     (dual_residual_input < abs_dua_tol);
    
    return converged;
}}"""

    def generate_testbench(self):
        """Generate comprehensive IP core testbench with reference validation"""
        return f"""#include <iostream>
#include <cmath>
#include <cstdlib>
#include <memory>
#include "tinympc_solver.h"
#include "test_data.h"

const {self.data_type} TOLERANCE = {self.tolerance};

bool validate_arrays(data_t* actual, data_t* expected, int size, const char* name) {{
    {self.data_type} max_error = 0.0{self.suffix};
    for (int i = 0; i < size; i++) {{
        {self.data_type} error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
    }}
    
    bool passed = (max_error <= TOLERANCE);
    std::cout << "  " << name << " - " << (passed ? "PASSED" : "FAILED")
              << " (max error: " << max_error << ")" << std::endl;
    return passed;
}}

void print_array(const char* name, data_t* arr, int size) {{
    std::cout << "  " << name << ": [";
    for (int i = 0; i < std::min(size, 5); i++) {{
        std::cout << arr[i];
        if (i < std::min(size, 5) - 1) std::cout << ", ";
    }}
    if (size > 5) std::cout << "...";
    std::cout << "]" << std::endl;
}}

int main() {{
    std::cout << "TinyMPC IP Core Validation with Reference Data" << std::endl;
    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;
    std::cout << "Memory size: " << MAIN_MEMORY_SIZE << " elements" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed_tests = 0;
    
    // Allocate main memory dynamically to avoid stack overflow
    data_t* main_memory = (data_t*)malloc(MAIN_MEMORY_SIZE * sizeof(data_t));
    if (main_memory == nullptr) {{
        std::cerr << "ERROR: Failed to allocate memory for main_memory array!" << std::endl;
        return -1;
    }}
    
    for (int test = 0; test < NUM_TESTS; test++) {{
        std::cout << "\\nTest " << test << ": ";
        if (test == 0) std::cout << "hover_stability";
        else if (test == 1) std::cout << "position_tracking";
        else std::cout << "disturbance_rejection";
        std::cout << std::endl;
        
        // Initialize main memory with zeros
        for (int i = 0; i < MAIN_MEMORY_SIZE; i++) {{
            main_memory[i] = 0.0{self.suffix};
        }}
        
        // Load test case data into main memory
        // Load x0
        for (int i = 0; i < NX; i++) {{
            main_memory[X0_OFFSET + i] = test_x0[test][i];
        }}
        
        // Load xref
        for (int k = 0; k < N; k++) {{
            for (int i = 0; i < NX; i++) {{
                main_memory[XREF_OFFSET + k*NX + i] = test_xref[test][k][i];
            }}
        }}
        
        // Load uref
        for (int k = 0; k < N_MINUS_1; k++) {{
            for (int j = 0; j < NU; j++) {{
                main_memory[UREF_OFFSET + k*NU + j] = test_uref[test][k][j];
            }}
        }}
        
        // Call IP core
        tinympc_solver(main_memory, MAX_ITER, 10);
        
        // Validate results against reference
        bool test_passed = true;
        
        // Extract and validate x_out
        data_t* x_out = (data_t*)malloc(N * NX * sizeof(data_t));
        if (x_out == nullptr) {{
            std::cerr << "ERROR: Failed to allocate memory for x_out!" << std::endl;
            free(main_memory);
            return -1;
        }}
        
        for (int k = 0; k < N; k++) {{
            for (int i = 0; i < NX; i++) {{
                x_out[k*NX + i] = main_memory[X_OUT_OFFSET + k*NX + i];
            }}
        }}
        test_passed &= validate_arrays(x_out, (data_t*)expected_x_out[test], N*NX, "x_out");
        
        // Extract and validate u_out
        data_t* u_out = (data_t*)malloc(N_MINUS_1 * NU * sizeof(data_t));
        if (u_out == nullptr) {{
            std::cerr << "ERROR: Failed to allocate memory for u_out!" << std::endl;
            free(main_memory);
            free(x_out);
            return -1;
        }}
        
        for (int k = 0; k < N_MINUS_1; k++) {{
            for (int j = 0; j < NU; j++) {{
                u_out[k*NU + j] = main_memory[U_OUT_OFFSET + k*NU + j];
            }}
        }}
        test_passed &= validate_arrays(u_out, (data_t*)expected_u_out[test], N_MINUS_1*NU, "u_out");
        
        // Free temporary arrays
        free(x_out);
        free(u_out);
        
        // Display summary
        std::cout << "  Expected iterations: " << expected_iterations[test] << std::endl;
        std::cout << "  Overall: " << (test_passed ? "PASSED" : "FAILED") << std::endl;
        
        if (test_passed) passed_tests++;
    }}
    
    // Free main memory
    free(main_memory);
    
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Final Result: " << passed_tests << "/" << NUM_TESTS << " tests passed" << std::endl;
    
    return (passed_tests == NUM_TESTS) ? 0 : 1;
}}"""

    def generate_tcl(self):
        """Generate Vitis HLS TCL script for IP core"""
        return f"""# TinyMPC IP Core HLS Project ({self.precision})
open_project -reset tinympc_ip_{self.precision}
add_files tinympc_solver.cpp
add_files -tb testbench.cpp
set_top tinympc_solver

open_solution -reset "solution1"
set_part {{xc7z020clg400-1}}
create_clock -period 10

# Optimization directives
config_interface -m_axi_addr64=false
config_interface -m_axi_auto_max_ports=false

# Run C simulation
csim_design -clean

# Optional: Run synthesis and implementation
# csynth_design
# cosim_design
# export_design -format ip_catalog

exit"""

    def generate_all(self, output_dir="tinympc_ip"):
        """Generate complete IP core project"""
        os.makedirs(output_dir, exist_ok=True)
        
        files = {
            "tinympc_solver.h": self.generate_header(),
            "tinympc_solver.cpp": self.generate_main_solver() + "\n\n" + self.generate_subfunctions(),
            "test_data.h": self.generate_test_data(),
            "testbench.cpp": self.generate_testbench(),
            "csim.tcl": self.generate_tcl()
        }
        
        for filename, content in files.items():
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
                
        print(f"Generated TinyMPC IP Core ({self.precision}) in {output_dir}/")
        print(f"System: {self.solver.nx} states, {self.solver.nu} inputs, N={self.solver.N}")
        print(f"Memory size: {self.solver.nx + 2 * self.solver.N * self.solver.nx + 2 * (self.solver.N-1) * self.solver.nu} elements")
        print(f"Usage: cd {output_dir} && vitis_hls -f csim.tcl")


def main():
    parser = argparse.ArgumentParser(description='Generate TinyMPC IP Core')
    parser.add_argument('--N', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--freq', type=float, default=100.0, help='Control frequency (Hz)')
    parser.add_argument('--precision', choices=['float', 'double'], default='float', help='Data precision')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum ADMM iterations')
    parser.add_argument('--output', default='tinympc_ip', help='Output directory')
    
    args = parser.parse_args()
    
    generator = TinyMPCIPGenerator(N=args.N, control_freq=args.freq, 
                                  precision=args.precision, max_iter=args.max_iter)
    generator.generate_all(args.output)


if __name__ == "__main__":
    main()