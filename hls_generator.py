#!/usr/bin/env python3
"""
HLS Function Generator Module for TinyMPC IP Core
Provides HLS function generation utilities for individual TinyMPC solver components
"""

import numpy as np
from dynamics import create_dynamics_model
from tinympcref import tinympcref


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
        return f'#include "tinympc_solver.h"\n\nvoid update_slack(\n    data_t znew[N_MINUS_1][NU],\n    data_t vnew[N][NX],\n    data_t u[N_MINUS_1][NU],\n    data_t x[N][NX],\n    data_t y[N_MINUS_1][NU],\n    data_t g[N][NX],\n    data_t u_min[N_MINUS_1][NU],\n    data_t u_max[N_MINUS_1][NU],\n    data_t x_min[N][NX],\n    data_t x_max[N][NX]\n) {{\n    // Update input slack variables\n    update_u_slack: for (int k = 0; k < N_MINUS_1; k++) {{\n#pragma HLS PIPELINE\n        for (int j = 0; j < NU; j++) {{\n#pragma HLS UNROLL\n            data_t temp_u = u[k][j] + y[k][j];\n            znew[k][j] = (temp_u > u_max[k][j]) ? u_max[k][j] : \n                        (temp_u < u_min[k][j]) ? u_min[k][j] : temp_u;\n        }}\n    }}\n    \n    // Update state slack variables\n    update_x_slack: for (int k = 0; k < N; k++) {{\n#pragma HLS PIPELINE\n        for (int i = 0; i < NX; i++) {{\n#pragma HLS UNROLL\n            data_t temp_x = x[k][i] + g[k][i];\n            vnew[k][i] = (temp_x > x_max[k][i]) ? x_max[k][i] : \n                        (temp_x < x_min[k][i]) ? x_min[k][i] : temp_x;\n        }}\n    }}\n}}'
    
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