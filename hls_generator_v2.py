#!/usr/bin/env python3
"""
Enhanced TinyMPC HLS Code Generator
Supports individual function generation with proper test data
"""

import numpy as np
import os
import argparse
from tinympcref import tinympcref as TinyMPCReference
from dynamics import create_dynamics_model


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
        
        self.solver = TinyMPCReference()
        self.solver.setup(A, B, Q, R, N, rho=1.0)
        
        self.function_name = function_name
        self.precision = precision
        self.data_type = 'float' if precision == 'float' else 'double'
        self.suffix = 'f' if precision == 'float' else ''
        self.tolerance = '1e-3f' if precision == 'float' else '1e-6'
        
        self.base_gen = BaseFunctionGenerator(self.solver, precision)
    
    def generate_header(self):
        """Generate header for specific function"""
        headers = {
            'forward_pass': f"""#ifndef FORWARD_PASS_H
#define FORWARD_PASS_H

#define NX {self.solver.nx}
#define NU {self.solver.nu}
#define N {self.solver.N}
#define N_MINUS_1 {self.solver.N-1}

typedef {self.data_type} data_t;

void forward_pass(
    data_t x[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t d[N_MINUS_1][NU]
);

#endif""",
            
            'backward_pass': f"""#ifndef BACKWARD_PASS_H
#define BACKWARD_PASS_H

#define NX {self.solver.nx}
#define NU {self.solver.nu}
#define N {self.solver.N}
#define N_MINUS_1 {self.solver.N-1}

typedef {self.data_type} data_t;

void backward_pass(
    data_t d[N_MINUS_1][NU],
    data_t p[N][NX],
    data_t q[N][NX],
    data_t r[N_MINUS_1][NU]
);

#endif""",

            'update_slack': f"""#ifndef UPDATE_SLACK_H
#define UPDATE_SLACK_H

#define NX {self.solver.nx}
#define NU {self.solver.nu}
#define N {self.solver.N}
#define N_MINUS_1 {self.solver.N-1}

typedef {self.data_type} data_t;

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

#endif""",
            
            'update_dual': f"""#ifndef UPDATE_DUAL_H
#define UPDATE_DUAL_H

#define NX {self.solver.nx}
#define NU {self.solver.nu}
#define N {self.solver.N}
#define N_MINUS_1 {self.solver.N-1}

typedef {self.data_type} data_t;

void update_dual(
    data_t y[N_MINUS_1][NU],
    data_t g[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t x[N][NX],
    data_t znew[N_MINUS_1][NU],
    data_t vnew[N][NX]
);

#endif""",

            'update_linear_cost': f"""#ifndef UPDATE_LINEAR_COST_H
#define UPDATE_LINEAR_COST_H

#define NX {self.solver.nx}
#define NU {self.solver.nu}
#define N {self.solver.N}
#define N_MINUS_1 {self.solver.N-1}

typedef {self.data_type} data_t;

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

#endif""",

            'check_termination': f"""#ifndef CHECK_TERMINATION_H
#define CHECK_TERMINATION_H

#define NX {self.solver.nx}
#define NU {self.solver.nu}
#define N {self.solver.N}
#define N_MINUS_1 {self.solver.N-1}

typedef {self.data_type} data_t;

bool check_termination(
    data_t x[N][NX],
    data_t vnew[N][NX],
    data_t v[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t znew[N_MINUS_1][NU],
    data_t z[N_MINUS_1][NU]
);

#endif"""
        }
        
        return headers.get(self.function_name, f"""#ifndef {self.function_name.upper()}_H
#define {self.function_name.upper()}_H

#define NX {self.solver.nx}
#define NU {self.solver.nu}
#define N {self.solver.N}
#define N_MINUS_1 {self.solver.N-1}

typedef {self.data_type} data_t;

#endif""")
    
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
        elif self.function_name == 'check_termination':
            return self._generate_check_termination()
        else:
            return f'#include "{self.function_name}.h"\n\n// Implementation placeholder\n'
    
    def _generate_forward_pass(self):
        """Generate forward pass implementation"""
        code = f'#include "{self.function_name}.h"\n\n'
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
        code = f'#include "{self.function_name}.h"\n\n'
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
        return f'#include "{self.function_name}.h"\n#include <algorithm>\n\nvoid update_slack(\n    data_t znew[N_MINUS_1][NU],\n    data_t vnew[N][NX],\n    data_t u[N_MINUS_1][NU],\n    data_t x[N][NX],\n    data_t y[N_MINUS_1][NU],\n    data_t g[N][NX],\n    data_t u_min[N_MINUS_1][NU],\n    data_t u_max[N_MINUS_1][NU],\n    data_t x_min[N][NX],\n    data_t x_max[N][NX]\n) {{\n    // Update input slack variables\n    update_u_slack: for (int k = 0; k < N_MINUS_1; k++) {{\n#pragma HLS PIPELINE\n        for (int j = 0; j < NU; j++) {{\n#pragma HLS UNROLL\n            data_t temp_u = u[k][j] + y[k][j];\n            znew[k][j] = (temp_u > u_max[k][j]) ? u_max[k][j] : \n                        (temp_u < u_min[k][j]) ? u_min[k][j] : temp_u;\n        }}\n    }}\n    \n    // Update state slack variables\n    update_x_slack: for (int k = 0; k < N; k++) {{\n#pragma HLS PIPELINE\n        for (int i = 0; i < NX; i++) {{\n#pragma HLS UNROLL\n            data_t temp_x = x[k][i] + g[k][i];\n            vnew[k][i] = (temp_x > x_max[k][i]) ? x_max[k][i] : \n                        (temp_x < x_min[k][i]) ? x_min[k][i] : temp_x;\n        }}\n    }}\n}}'
    
    def _generate_update_dual(self):
        """Generate update dual implementation"""
        return f'#include "{self.function_name}.h"\n\nvoid update_dual(\n    data_t y[N_MINUS_1][NU],\n    data_t g[N][NX],\n    data_t u[N_MINUS_1][NU],\n    data_t x[N][NX],\n    data_t znew[N_MINUS_1][NU],\n    data_t vnew[N][NX]\n) {{\n    // Update input dual variables\n    update_y_dual: for (int k = 0; k < N_MINUS_1; k++) {{\n#pragma HLS PIPELINE\n        for (int j = 0; j < NU; j++) {{\n#pragma HLS UNROLL\n            y[k][j] = y[k][j] + u[k][j] - znew[k][j];\n        }}\n    }}\n    \n    // Update state dual variables\n    update_g_dual: for (int k = 0; k < N; k++) {{\n#pragma HLS PIPELINE\n        for (int i = 0; i < NX; i++) {{\n#pragma HLS UNROLL\n            g[k][i] = g[k][i] + x[k][i] - vnew[k][i];\n        }}\n    }}\n}}'
    
    def _generate_update_linear_cost(self):
        """Generate update linear cost implementation with hardcoded parameters"""
        code = f'#include "{self.function_name}.h"\n\n'
        
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
    
    def _generate_check_termination(self):
        """Generate check termination implementation with hardcoded parameters"""
        # Get hardcoded values
        rho_val = self.base_gen._format_value(self.solver.rho)
        abs_pri_tol = self.base_gen._format_value(self.solver.abs_pri_tol)
        abs_dua_tol = self.base_gen._format_value(self.solver.abs_dua_tol)
        
        return f'''#include "{self.function_name}.h"
#include <cmath>

bool check_termination(
    data_t x[N][NX],
    data_t vnew[N][NX],
    data_t v[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t znew[N_MINUS_1][NU],
    data_t z[N_MINUS_1][NU]
) {{
    data_t primal_residual_state = {self.base_gen._format_value(0.0)};
    data_t dual_residual_state = {self.base_gen._format_value(0.0)};
    data_t primal_residual_input = {self.base_gen._format_value(0.0)};
    data_t dual_residual_input = {self.base_gen._format_value(0.0)};
    
    // Compute state residuals
    state_residuals: for (int k = 0; k < N; k++) {{
#pragma HLS PIPELINE
        for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
            // primal_residual_state = max(abs(x - vnew))
            data_t pri_res = x[k][i] - vnew[k][i];
            pri_res = (pri_res < {self.base_gen._format_value(0.0)}) ? -pri_res : pri_res;
            primal_residual_state = (pri_res > primal_residual_state) ? pri_res : primal_residual_state;
            
            // dual_residual_state = max(abs(v - vnew))
            data_t dua_res = v[k][i] - vnew[k][i];
            dua_res = (dua_res < {self.base_gen._format_value(0.0)}) ? -dua_res : dua_res;
            dual_residual_state = (dua_res > dual_residual_state) ? dua_res : dual_residual_state;
        }}
    }}
    
    // Compute input residuals
    input_residuals: for (int k = 0; k < N_MINUS_1; k++) {{
#pragma HLS PIPELINE
        for (int j = 0; j < NU; j++) {{
#pragma HLS UNROLL
            // primal_residual_input = max(abs(u - znew))
            data_t pri_res = u[k][j] - znew[k][j];
            pri_res = (pri_res < {self.base_gen._format_value(0.0)}) ? -pri_res : pri_res;
            primal_residual_input = (pri_res > primal_residual_input) ? pri_res : primal_residual_input;
            
            // dual_residual_input = max(abs(z - znew))
            data_t dua_res = z[k][j] - znew[k][j];
            dua_res = (dua_res < {self.base_gen._format_value(0.0)}) ? -dua_res : dua_res;
            dual_residual_input = (dua_res > dual_residual_input) ? dua_res : dual_residual_input;
        }}
    }}
    
    // Apply hardcoded rho scaling to dual residuals
    dual_residual_state *= {rho_val};
    dual_residual_input *= {rho_val};
    
    // Check convergence with hardcoded tolerances
    return (primal_residual_state < {abs_pri_tol} && 
            primal_residual_input < {abs_pri_tol} &&
            dual_residual_state < {abs_dua_tol} && 
            dual_residual_input < {abs_dua_tol});
}}'''
    
    def generate_test_data(self):
        """Generate test data for specific function"""
        dtype = np.float32 if self.precision == 'float' else np.float64
        
        if self.function_name == 'forward_pass':
            return self._generate_forward_pass_test_data(dtype)
        elif self.function_name == 'backward_pass':
            return self._generate_backward_pass_test_data(dtype)
        elif self.function_name == 'update_slack':
            return self._generate_update_slack_test_data(dtype)
        elif self.function_name == 'update_dual':
            return self._generate_update_dual_test_data(dtype)
        elif self.function_name == 'update_linear_cost':
            return self._generate_update_linear_cost_test_data(dtype)
        elif self.function_name == 'check_termination':
            return self._generate_check_termination_test_data(dtype)
        else:
            return self._generate_simple_test_data(dtype)
    
    def _generate_forward_pass_test_data(self, dtype):
        """Generate test data for forward pass"""
        test_cases = [
            {
                'name': 'hover_stability',
                'x0': np.zeros(self.solver.nx, dtype=dtype),
                'd': np.zeros((self.solver.N-1, self.solver.nu), dtype=dtype)
            },
            {
                'name': 'position_tracking',
                'x0': np.array([0.1, 0.1, 0.05] + [0.0]*(self.solver.nx-3), dtype=dtype),
                'd': np.zeros((self.solver.N-1, self.solver.nu), dtype=dtype)
            },
            {
                'name': 'disturbance_rejection',
                'x0': np.random.randn(self.solver.nx).astype(dtype) * 0.02,
                'd': np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.01
            }
        ]
        
        results = []
        for case in test_cases:
            self.solver.set_x0(case['x0'])
            self.solver.d = case['d']
            self.solver._forward_pass()
            results.append({'x': self.solver.x.copy(), 'u': self.solver.u.copy()})
        
        return self._generate_forward_pass_test_header(test_cases, results)
    
    def _generate_backward_pass_test_data(self, dtype):
        """Generate test data for backward pass"""
        test_cases = []
        results = []
        
        for i, name in enumerate(['case1', 'case2', 'case3']):
            # Generate random test inputs for backward pass
            q = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.1
            r = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.05
            p_init = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.02
            
            # Run reference backward pass computation
            d_ref = np.zeros((self.solver.N-1, self.solver.nu), dtype=dtype)
            p_ref = p_init.copy()
            
            # Simulate backward pass manually
            for k in range(self.solver.N-2, -1, -1):
                # d[k] = Quu_inv @ (B.T @ p[k+1] + r[k])
                bt_p = self.solver.B.T @ p_ref[k+1] + r[k]
                d_ref[k] = self.solver.Quu_inv @ bt_p
                
                # p[k] = q[k] + AmBKt @ p[k+1] - Kinf.T @ r[k]
                p_ref[k] = q[k] + self.solver.AmBKt @ p_ref[k+1] - self.solver.Kinf.T @ r[k]
            
            test_cases.append({
                'name': name,
                'q': q,
                'r': r,
                'p_initial': p_init
            })
            
            results.append({
                'd': d_ref,
                'p': p_ref
            })
        
        return self._generate_backward_pass_test_header(test_cases, results)
    
    def _generate_forward_pass_test_header(self, test_cases, results):
        """Generate forward pass test data header"""
        header_parts = []
        header_parts.append("#ifndef TEST_DATA_H")
        header_parts.append("#define TEST_DATA_H")
        header_parts.append("")
        header_parts.append(f'#include "{self.function_name}.h"')
        header_parts.append("")
        header_parts.append("#define NUM_TESTS 3")
        header_parts.append("")
        
        # Test initial states
        header_parts.append(f"{self.data_type} test_x0[NUM_TESTS][NX] = {{")
        for i, case in enumerate(test_cases):
            values = ", ".join([self.base_gen._format_value(case['x0'][j]) for j in range(self.solver.nx)])
            header_parts.append(f"    {{{values}}},  // {case['name']}")
        header_parts.append("};")
        header_parts.append("")
        
        # Test disturbances
        header_parts.append(f"{self.data_type} test_d[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(case['d'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Expected outputs
        header_parts.append(f"{self.data_type} expected_x[NUM_TESTS][N][NX] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(result['x'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        header_parts.append(f"{self.data_type} expected_u[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(result['u'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        header_parts.append("#endif")
        
        return "\n".join(header_parts)
    
    def _generate_backward_pass_test_header(self, test_cases, results):
        """Generate backward pass test data header"""
        header_parts = []
        header_parts.append("#ifndef TEST_DATA_H")
        header_parts.append("#define TEST_DATA_H")
        header_parts.append("")
        header_parts.append(f'#include "{self.function_name}.h"')
        header_parts.append("")
        header_parts.append("#define NUM_TESTS 3")
        header_parts.append("")
        
        # Test q arrays
        header_parts.append(f"{self.data_type} test_q[NUM_TESTS][N][NX] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(case['q'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test r arrays  
        header_parts.append(f"{self.data_type} test_r[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(case['r'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test p initial arrays
        header_parts.append(f"{self.data_type} test_p_initial[NUM_TESTS][N][NX] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(case['p_initial'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Expected outputs
        header_parts.append(f"{self.data_type} expected_d[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(result['d'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        header_parts.append(f"{self.data_type} expected_p[NUM_TESTS][N][NX] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(result['p'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        header_parts.append("#endif")
        
        return "\n".join(header_parts)
    
    def _generate_update_slack_test_header(self, test_cases, results):
        """Generate update_slack test data header"""
        header_parts = []
        header_parts.append("#ifndef TEST_DATA_H")
        header_parts.append("#define TEST_DATA_H")
        header_parts.append("")
        header_parts.append(f'#include "{self.function_name}.h"')
        header_parts.append("")
        header_parts.append("#define NUM_TESTS 3")
        header_parts.append("")
        
        # Test u arrays
        header_parts.append(f"{self.data_type} test_u[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(case['u'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test y arrays
        header_parts.append(f"{self.data_type} test_y[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(case['y'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test v arrays
        header_parts.append(f"{self.data_type} test_v[NUM_TESTS][N][NX] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(case['v'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test g arrays
        header_parts.append(f"{self.data_type} test_g[NUM_TESTS][N][NX] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(case['g'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test u_min arrays
        header_parts.append(f"{self.data_type} test_u_min[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(case['u_min'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test u_max arrays
        header_parts.append(f"{self.data_type} test_u_max[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(case['u_max'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test x_min arrays
        header_parts.append(f"{self.data_type} test_x_min[NUM_TESTS][N][NX] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(case['x_min'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Test x_max arrays
        header_parts.append(f"{self.data_type} test_x_max[NUM_TESTS][N][NX] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(case['x_max'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Expected znew arrays
        header_parts.append(f"{self.data_type} expected_znew[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(result['znew'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Expected vnew arrays
        header_parts.append(f"{self.data_type} expected_vnew[NUM_TESTS][N][NX] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(result['vnew'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        header_parts.append("#endif")
        
        return "\n".join(header_parts)
    
    def _generate_update_dual_test_header(self, test_cases, results):
        """Generate update_dual test data header"""
        header_parts = []
        header_parts.append("#ifndef TEST_DATA_H")
        header_parts.append("#define TEST_DATA_H")
        header_parts.append("")
        header_parts.append(f'#include "{self.function_name}.h"')
        header_parts.append("")
        header_parts.append("#define NUM_TESTS 3")
        header_parts.append("")
        
        # Test input arrays
        for var_name in ['u', 'znew', 'y_initial']:
            header_parts.append(f"{self.data_type} test_{var_name}[NUM_TESTS][N_MINUS_1][NU] = {{")
            for i, case in enumerate(test_cases):
                header_parts.append(f"    {{  // {case['name']}")
                for k in range(self.solver.N-1):
                    values = ", ".join([self.base_gen._format_value(case[var_name][k,j]) for j in range(self.solver.nu)])
                    header_parts.append(f"        {{{values}}},")
                header_parts.append("    },")
            header_parts.append("};")
            header_parts.append("")
        
        for var_name in ['x', 'vnew', 'g_initial']:
            header_parts.append(f"{self.data_type} test_{var_name}[NUM_TESTS][N][NX] = {{")
            for i, case in enumerate(test_cases):
                header_parts.append(f"    {{  // {case['name']}")
                for k in range(self.solver.N):
                    values = ", ".join([self.base_gen._format_value(case[var_name][k,j]) for j in range(self.solver.nx)])
                    header_parts.append(f"        {{{values}}},")
                header_parts.append("    },")
            header_parts.append("};")
            header_parts.append("")
        
        # Expected outputs
        header_parts.append(f"{self.data_type} expected_y[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.solver.N-1):
                values = ", ".join([self.base_gen._format_value(result['y_expected'][k,j]) for j in range(self.solver.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        header_parts.append(f"{self.data_type} expected_g[NUM_TESTS][N][NX] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.solver.N):
                values = ", ".join([self.base_gen._format_value(result['g_expected'][k,j]) for j in range(self.solver.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        header_parts.append("#endif")
        
        return "\n".join(header_parts)
    
    def _generate_update_linear_cost_test_header(self, test_cases, results):
        """Generate update_linear_cost test data header"""
        header_parts = []
        header_parts.append("#ifndef TEST_DATA_H")
        header_parts.append("#define TEST_DATA_H")
        header_parts.append("")
        header_parts.append(f'#include "{self.function_name}.h"')
        header_parts.append("")
        header_parts.append("#define NUM_TESTS 3")
        header_parts.append("")
        
        # Test input arrays
        for var_name, dims in [('Uref', '[N_MINUS_1][NU]'), ('Xref', '[N][NX]'), ('znew', '[N_MINUS_1][NU]'), 
                              ('vnew', '[N][NX]'), ('y', '[N_MINUS_1][NU]'), ('g', '[N][NX]')]:
            header_parts.append(f"{self.data_type} test_{var_name}[NUM_TESTS]{dims} = {{")
            for i, case in enumerate(test_cases):
                header_parts.append(f"    {{  // {case['name']}")
                if var_name in ['Uref', 'znew', 'y']:
                    for k in range(self.solver.N-1):
                        values = ", ".join([self.base_gen._format_value(case[var_name][k,j]) for j in range(self.solver.nu)])
                        header_parts.append(f"        {{{values}}},")
                else:  # Xref, vnew, g
                    for k in range(self.solver.N):
                        values = ", ".join([self.base_gen._format_value(case[var_name][k,j]) for j in range(self.solver.nx)])
                        header_parts.append(f"        {{{values}}},")
                header_parts.append("    },")
            header_parts.append("};")
            header_parts.append("")
        
        # Expected outputs
        for var_name, dims in [('r', '[N_MINUS_1][NU]'), ('q', '[N][NX]'), ('p', '[N][NX]')]:
            header_parts.append(f"{self.data_type} expected_{var_name}[NUM_TESTS]{dims} = {{")
            for i, result in enumerate(results):
                header_parts.append(f"    {{  // {test_cases[i]['name']}")
                if var_name == 'r':
                    for k in range(self.solver.N-1):
                        values = ", ".join([self.base_gen._format_value(result[var_name][k,j]) for j in range(self.solver.nu)])
                        header_parts.append(f"        {{{values}}},")
                else:  # q, p
                    for k in range(self.solver.N):
                        values = ", ".join([self.base_gen._format_value(result[var_name][k,j]) for j in range(self.solver.nx)])
                        header_parts.append(f"        {{{values}}},")
                header_parts.append("    },")
            header_parts.append("};")
            header_parts.append("")
        header_parts.append("#endif")
        
        return "\n".join(header_parts)
    
    def _generate_check_termination_test_header(self, test_cases, results):
        """Generate check_termination test data header"""
        header_parts = []
        header_parts.append("#ifndef TEST_DATA_H")
        header_parts.append("#define TEST_DATA_H")
        header_parts.append("")
        header_parts.append(f'#include "{self.function_name}.h"')
        header_parts.append("")
        header_parts.append("#define NUM_TESTS 3")
        header_parts.append("")
        
        # Test input arrays
        for var_name, dims in [('x', '[N][NX]'), ('vnew', '[N][NX]'), ('v', '[N][NX]'), 
                              ('u', '[N_MINUS_1][NU]'), ('znew', '[N_MINUS_1][NU]'), ('z', '[N_MINUS_1][NU]')]:
            header_parts.append(f"{self.data_type} test_{var_name}[NUM_TESTS]{dims} = {{")
            for i, case in enumerate(test_cases):
                header_parts.append(f"    {{  // {case['name']}")
                if var_name in ['u', 'znew', 'z']:
                    for k in range(self.solver.N-1):
                        values = ", ".join([self.base_gen._format_value(case[var_name][k,j]) for j in range(self.solver.nu)])
                        header_parts.append(f"        {{{values}}},")
                else:  # x, vnew, v
                    for k in range(self.solver.N):
                        values = ", ".join([self.base_gen._format_value(case[var_name][k,j]) for j in range(self.solver.nx)])
                        header_parts.append(f"        {{{values}}},")
                header_parts.append("    },")
            header_parts.append("};")
            header_parts.append("")
        
        # Expected convergence results
        header_parts.append("bool expected_converged[NUM_TESTS] = {")
        for i, result in enumerate(results):
            header_parts.append(f"    {str(result['converged']).lower()},  // {test_cases[i]['name']}")
        header_parts.append("};")
        header_parts.append("")
        header_parts.append("#endif")
        
        return "\n".join(header_parts)
    
    def _generate_update_slack_test_data(self, dtype):
        """Generate test data for update_slack"""
        test_cases = []
        results = []
        
        for i, name in enumerate(['bounded_case', 'unbounded_case', 'mixed_case']):
            # Generate test inputs
            u = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.1
            y = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.05
            v = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.1  
            g = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.05
            
            # Generate constraint bounds
            u_min = np.full((self.solver.N-1, self.solver.nu), -2.0, dtype=dtype)
            u_max = np.full((self.solver.N-1, self.solver.nu), 2.0, dtype=dtype)
            x_min = np.full((self.solver.N, self.solver.nx), -10.0, dtype=dtype)
            x_max = np.full((self.solver.N, self.solver.nx), 10.0, dtype=dtype)
            
            # Compute expected outputs using reference implementation
            znew = np.zeros_like(u)
            vnew = np.zeros_like(v)
            
            # Update slack variables (box projection)
            for k in range(self.solver.N-1):
                for j in range(self.solver.nu):
                    temp_u = u[k,j] + y[k,j]
                    znew[k,j] = max(u_min[k,j], min(u_max[k,j], temp_u))
            
            for k in range(self.solver.N):
                for j in range(self.solver.nx):
                    temp_x = v[k,j] + g[k,j]
                    vnew[k,j] = max(x_min[k,j], min(x_max[k,j], temp_x))
            
            test_cases.append({
                'name': name, 'u': u, 'y': y, 'v': v, 'g': g,
                'u_min': u_min, 'u_max': u_max, 'x_min': x_min, 'x_max': x_max
            })
            results.append({'znew': znew, 'vnew': vnew})
        
        return self._generate_update_slack_test_header(test_cases, results)
    
    def _generate_update_dual_test_data(self, dtype):
        """Generate test data for update_dual"""
        test_cases = []
        results = []
        
        for i, name in enumerate(['case1', 'case2', 'case3']):
            # Generate test inputs
            u = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.1
            znew = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.05
            y_initial = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.03
            x = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.1
            vnew = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.05
            g_initial = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.03
            
            # Compute expected outputs (dual update: y = y + rho*(u - znew), g = g + rho*(x - vnew))
            y_expected = y_initial + self.solver.rho * (u - znew)
            g_expected = g_initial + self.solver.rho * (x - vnew)
            
            test_cases.append({
                'name': name, 'u': u, 'znew': znew, 'y_initial': y_initial, 
                'x': x, 'vnew': vnew, 'g_initial': g_initial
            })
            results.append({'y_expected': y_expected, 'g_expected': g_expected})
        
        return self._generate_update_dual_test_header(test_cases, results)
    
    def _generate_update_linear_cost_test_data(self, dtype):
        """Generate test data for update_linear_cost"""
        test_cases = []
        results = []
        
        for i, name in enumerate(['tracking_case', 'regulation_case', 'disturbance_case']):
            # Generate test inputs
            if name == 'tracking_case':
                Uref = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.1
                Xref = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.1
            else:
                Uref = np.zeros((self.solver.N-1, self.solver.nu), dtype=dtype)
                Xref = np.zeros((self.solver.N, self.solver.nx), dtype=dtype)
            
            znew = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.05
            vnew = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.05
            y = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.02
            g = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.02
            
            # Compute expected outputs
            r = np.zeros_like(znew)
            q = np.zeros_like(vnew)
            p = np.zeros_like(vnew)
            
            for k in range(self.solver.N-1):
                for j in range(self.solver.nu):
                    r[k,j] = -Uref[k,j] * self.solver.R[j] - self.solver.rho * (znew[k,j] - y[k,j])
            
            for k in range(self.solver.N-1):
                for j in range(self.solver.nx):
                    q[k,j] = -Xref[k,j] * self.solver.Q[j] - self.solver.rho * (vnew[k,j] - g[k,j])
            
            # Terminal cost
            for j in range(self.solver.nx):
                q[self.solver.N-1,j] = -Xref[self.solver.N-1,j] * self.solver.Q[j] - self.solver.rho * (vnew[self.solver.N-1,j] - g[self.solver.N-1,j])
                p[self.solver.N-1,j] = -self.solver.Pinf.T[j,:] @ Xref[self.solver.N-1,:] - self.solver.rho * (vnew[self.solver.N-1,j] - g[self.solver.N-1,j])
            
            test_cases.append({'name': name, 'Uref': Uref, 'Xref': Xref, 'znew': znew, 'vnew': vnew, 'y': y, 'g': g})
            results.append({'r': r, 'q': q, 'p': p})
        
        return self._generate_update_linear_cost_test_header(test_cases, results)
    
    def _generate_check_termination_test_data(self, dtype):
        """Generate test data for check_termination"""
        test_cases = []
        results = []
        
        for i, name in enumerate(['converged_case', 'not_converged_case', 'boundary_case']):
            if name == 'converged_case':
                # Small residuals that should converge
                x = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.0001
                vnew = x + np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.0001
                v = vnew + np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.0001
                u = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.0001
                znew = u + np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.0001
                z = znew + np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.0001
                expected = True
            else:
                # Large residuals that should not converge
                scale = 0.1 if name == 'not_converged_case' else 0.002
                x = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * scale
                vnew = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * scale
                v = np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * scale
                u = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * scale
                znew = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * scale
                z = np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * scale
                expected = False if name == 'not_converged_case' else True
            
            test_cases.append({'name': name, 'x': x, 'vnew': vnew, 'v': v, 'u': u, 'znew': znew, 'z': z})
            results.append({'converged': expected})
        
        return self._generate_check_termination_test_header(test_cases, results)

    def _generate_simple_test_data(self, dtype):
        """Generate simple test data for other functions"""
        return f"""#ifndef TEST_DATA_H
#define TEST_DATA_H

#include "{self.function_name}.h"

#define NUM_TESTS 1

// Placeholder test data - needs to be customized for each function
{self.data_type} test_data[1] = {{1.0{self.suffix}}};

#endif"""
    
    def generate_testbench(self):
        """Generate testbench for specific function"""
        if self.function_name == 'forward_pass':
            return self._generate_forward_pass_testbench()
        elif self.function_name == 'backward_pass':
            return self._generate_backward_pass_testbench()
        elif self.function_name == 'update_slack':
            return self._generate_update_slack_testbench()
        elif self.function_name == 'update_dual':
            return self._generate_update_dual_testbench()
        elif self.function_name == 'update_linear_cost':
            return self._generate_update_linear_cost_testbench()
        elif self.function_name == 'check_termination':
            return self._generate_check_termination_testbench()
        else:
            return self._generate_simple_testbench()
    
    def _generate_forward_pass_testbench(self):
        """Generate testbench for forward pass"""
        return f"""#include <iostream>
#include <cmath>
#include "{self.function_name}.h"
#include "test_data.h"

const {self.data_type} TOLERANCE = {self.tolerance};

bool validate_arrays(data_t* actual, data_t* expected, int size, const char* name) {{
    {self.data_type} max_error = 0.0{self.suffix};
    for (int i = 0; i < size; i++) {{
        {self.data_type} error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
    }}
    
    bool passed = (max_error <= TOLERANCE);
    std::cout << name << " - " << (passed ? "PASSED" : "FAILED")
              << " (max error: " << max_error << ")" << std::endl;
    return passed;
}}

int main() {{
    std::cout << "Forward Pass Function Validation" << std::endl;
    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    const char* names[NUM_TESTS] = {{"hover_stability", "position_tracking", "disturbance_rejection"}};
    
    for (int test = 0; test < NUM_TESTS; test++) {{
        std::cout << "\\nTest " << (test + 1) << ": " << names[test] << std::endl;
        
        data_t x[N][NX], u[N_MINUS_1][NU];
        
        // Initialize x[0]
        for (int i = 0; i < NX; i++) {{
            x[0][i] = test_x0[test][i];
        }}
        
        // Run forward pass
        forward_pass(x, u, test_d[test]);
        
        // Validate results
        bool x_ok = validate_arrays((data_t*)x, (data_t*)expected_x[test], N * NX, "States");
        bool u_ok = validate_arrays((data_t*)u, (data_t*)expected_u[test], N_MINUS_1 * NU, "Controls");
        
        if (x_ok && u_ok) passed++;
    }}
    
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Result: " << passed << "/" << NUM_TESTS << " tests passed" << std::endl;
    
    return (passed == NUM_TESTS) ? 0 : 1;
}}"""
    
    def _generate_backward_pass_testbench(self):
        """Generate testbench for backward pass"""
        return f"""#include <iostream>
#include <cmath>
#include "{self.function_name}.h"
#include "test_data.h"

const {self.data_type} TOLERANCE = {self.tolerance};

bool validate_arrays(data_t* actual, data_t* expected, int size, const char* name) {{
    {self.data_type} max_error = 0.0{self.suffix};
    for (int i = 0; i < size; i++) {{
        {self.data_type} error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
    }}
    
    bool passed = (max_error <= TOLERANCE);
    std::cout << name << " - " << (passed ? "PASSED" : "FAILED")
              << " (max error: " << max_error << ")" << std::endl;
    return passed;
}}

int main() {{
    std::cout << "Backward Pass Function Validation" << std::endl;
    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    const char* names[NUM_TESTS] = {{"case1", "case2", "case3"}};
    
    for (int test = 0; test < NUM_TESTS; test++) {{
        std::cout << "\\nTest " << (test + 1) << ": " << names[test] << std::endl;
        
        data_t d[N_MINUS_1][NU], p[N][NX];
        
        // Copy initial p values
        for (int k = 0; k < N; k++) {{
            for (int i = 0; i < NX; i++) {{
                p[k][i] = test_p_initial[test][k][i];
            }}
        }}
        
        // Run backward pass
        backward_pass(d, p, test_q[test], test_r[test]);
        
        // Validate results
        bool d_ok = validate_arrays((data_t*)d, (data_t*)expected_d[test], N_MINUS_1 * NU, "d values");
        bool p_ok = validate_arrays((data_t*)p, (data_t*)expected_p[test], N * NX, "p values");
        
        if (d_ok && p_ok) passed++;
    }}
    
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Result: " << passed << "/" << NUM_TESTS << " tests passed" << std::endl;
    
    return (passed == NUM_TESTS) ? 0 : 1;
}}"""
    
    def _generate_update_slack_testbench(self):
        """Generate testbench for update_slack"""
        return f"""#include <iostream>
#include <cmath>
#include "{self.function_name}.h"
#include "test_data.h"

const {self.data_type} TOLERANCE = {self.tolerance};

bool validate_arrays(data_t* actual, data_t* expected, int size, const char* name) {{
    {self.data_type} max_error = 0.0{self.suffix};
    for (int i = 0; i < size; i++) {{
        {self.data_type} error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
    }}
    
    bool passed = (max_error <= TOLERANCE);
    std::cout << name << " - " << (passed ? "PASSED" : "FAILED")
              << " (max error: " << max_error << ")" << std::endl;
    return passed;
}}

int main() {{
    std::cout << "Update Slack Function Validation" << std::endl;
    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    
    for (int test = 0; test < NUM_TESTS; test++) {{
        data_t u[N_MINUS_1][NU], y[N_MINUS_1][NU];
        data_t v[N][NX], g[N][NX];
        data_t znew[N_MINUS_1][NU], vnew[N][NX];
        data_t u_min[N_MINUS_1][NU], u_max[N_MINUS_1][NU];
        data_t x_min[N][NX], x_max[N][NX];
        
        // Copy test inputs
        for (int k = 0; k < N_MINUS_1; k++) {{
            for (int j = 0; j < NU; j++) {{
                u[k][j] = test_u[test][k][j];
                y[k][j] = test_y[test][k][j];
                u_min[k][j] = test_u_min[test][k][j];
                u_max[k][j] = test_u_max[test][k][j];
            }}
        }}
        for (int k = 0; k < N; k++) {{
            for (int i = 0; i < NX; i++) {{
                v[k][i] = test_v[test][k][i];
                g[k][i] = test_g[test][k][i];
                x_min[k][i] = test_x_min[test][k][i];
                x_max[k][i] = test_x_max[test][k][i];
            }}
        }}
        
        // Call function under test
        update_slack(znew, vnew, u, v, y, g, u_min, u_max, x_min, x_max);
        
        // Validate outputs
        std::cout << "Test " << test << ":" << std::endl;
        bool test_passed = true;
        test_passed &= validate_arrays((data_t*)znew, (data_t*)expected_znew[test], N_MINUS_1*NU, "  znew");
        test_passed &= validate_arrays((data_t*)vnew, (data_t*)expected_vnew[test], N*NX, "  vnew");
        
        if (test_passed) passed++;
    }}
    
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Result: " << passed << "/" << NUM_TESTS << " tests passed" << std::endl;
    
    return (passed == NUM_TESTS) ? 0 : 1;
}}"""
    
    def _generate_update_dual_testbench(self):
        """Generate testbench for update_dual"""
        return f"""#include <iostream>
#include <cmath>
#include "{self.function_name}.h"
#include "test_data.h"

const {self.data_type} TOLERANCE = {self.tolerance};

bool validate_arrays(data_t* actual, data_t* expected, int size, const char* name) {{
    {self.data_type} max_error = 0.0{self.suffix};
    for (int i = 0; i < size; i++) {{
        {self.data_type} error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
    }}
    
    bool passed = (max_error <= TOLERANCE);
    std::cout << name << " - " << (passed ? "PASSED" : "FAILED")
              << " (max error: " << max_error << ")" << std::endl;
    return passed;
}}

int main() {{
    std::cout << "Update Dual Function Validation" << std::endl;
    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    
    for (int test = 0; test < NUM_TESTS; test++) {{
        data_t u[N_MINUS_1][NU], znew[N_MINUS_1][NU], y[N_MINUS_1][NU];
        data_t x[N][NX], vnew[N][NX], g[N][NX];
        
        // Copy test inputs
        for (int k = 0; k < N_MINUS_1; k++) {{
            for (int j = 0; j < NU; j++) {{
                u[k][j] = test_u[test][k][j];
                znew[k][j] = test_znew[test][k][j];
                y[k][j] = test_y_initial[test][k][j];  // Start with initial y values
            }}
        }}
        for (int k = 0; k < N; k++) {{
            for (int i = 0; i < NX; i++) {{
                x[k][i] = test_x[test][k][i];
                vnew[k][i] = test_vnew[test][k][i];
                g[k][i] = test_g_initial[test][k][i];  // Start with initial g values
            }}
        }}
        
        // Call function under test (y and g are modified in place)
        update_dual(y, g, u, x, znew, vnew);
        
        // Validate outputs
        std::cout << "Test " << test << ":" << std::endl;
        bool test_passed = true;
        test_passed &= validate_arrays((data_t*)y, (data_t*)expected_y[test], N_MINUS_1*NU, "  y");
        test_passed &= validate_arrays((data_t*)g, (data_t*)expected_g[test], N*NX, "  g");
        
        if (test_passed) passed++;
    }}
    
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Result: " << passed << "/" << NUM_TESTS << " tests passed" << std::endl;
    
    return (passed == NUM_TESTS) ? 0 : 1;
}}"""
    
    def _generate_update_linear_cost_testbench(self):
        """Generate testbench for update_linear_cost"""
        return f"""#include <iostream>
#include <cmath>
#include "{self.function_name}.h"
#include "test_data.h"

const {self.data_type} TOLERANCE = {self.tolerance};

bool validate_arrays(data_t* actual, data_t* expected, int size, const char* name) {{
    {self.data_type} max_error = 0.0{self.suffix};
    for (int i = 0; i < size; i++) {{
        {self.data_type} error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
    }}
    
    bool passed = (max_error <= TOLERANCE);
    std::cout << name << " - " << (passed ? "PASSED" : "FAILED")
              << " (max error: " << max_error << ")" << std::endl;
    return passed;
}}

int main() {{
    std::cout << "Update Linear Cost Function Validation" << std::endl;
    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    
    for (int test = 0; test < NUM_TESTS; test++) {{
        data_t r[N_MINUS_1][NU], q[N][NX], p[N][NX];
        data_t Uref[N_MINUS_1][NU], Xref[N][NX];
        data_t znew[N_MINUS_1][NU], vnew[N][NX];
        data_t y[N_MINUS_1][NU], g[N][NX];
        
        // Copy test inputs
        for (int k = 0; k < N_MINUS_1; k++) {{
            for (int j = 0; j < NU; j++) {{
                Uref[k][j] = test_Uref[test][k][j];
                znew[k][j] = test_znew[test][k][j];
                y[k][j] = test_y[test][k][j];
            }}
        }}
        for (int k = 0; k < N; k++) {{
            for (int i = 0; i < NX; i++) {{
                Xref[k][i] = test_Xref[test][k][i];
                vnew[k][i] = test_vnew[test][k][i];
                g[k][i] = test_g[test][k][i];
            }}
        }}
        
        // Call function under test
        update_linear_cost(r, q, p, Uref, Xref, znew, vnew, y, g);
        
        // Validate outputs
        std::cout << "Test " << test << ":" << std::endl;
        bool test_passed = true;
        test_passed &= validate_arrays((data_t*)r, (data_t*)expected_r[test], N_MINUS_1*NU, "  r");
        test_passed &= validate_arrays((data_t*)q, (data_t*)expected_q[test], N*NX, "  q");
        test_passed &= validate_arrays((data_t*)p, (data_t*)expected_p[test], N*NX, "  p");
        
        if (test_passed) passed++;
    }}
    
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Result: " << passed << "/" << NUM_TESTS << " tests passed" << std::endl;
    
    return (passed == NUM_TESTS) ? 0 : 1;
}}"""
    
    def _generate_check_termination_testbench(self):
        """Generate testbench for check_termination"""
        return f"""#include <iostream>
#include <cmath>
#include "{self.function_name}.h"
#include "test_data.h"

int main() {{
    std::cout << "Check Termination Function Validation" << std::endl;
    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    
    for (int test = 0; test < NUM_TESTS; test++) {{
        data_t x[N][NX], vnew[N][NX], v[N][NX];
        data_t u[N_MINUS_1][NU], znew[N_MINUS_1][NU], z[N_MINUS_1][NU];
        
        // Copy test inputs
        for (int k = 0; k < N_MINUS_1; k++) {{
            for (int j = 0; j < NU; j++) {{
                u[k][j] = test_u[test][k][j];
                znew[k][j] = test_znew[test][k][j];
                z[k][j] = test_z[test][k][j];
            }}
        }}
        for (int k = 0; k < N; k++) {{
            for (int i = 0; i < NX; i++) {{
                x[k][i] = test_x[test][k][i];
                vnew[k][i] = test_vnew[test][k][i];
                v[k][i] = test_v[test][k][i];
            }}
        }}
        
        // Call function under test
        bool converged = check_termination(x, vnew, v, u, znew, z);
        
        // Validate output
        bool test_passed = (converged == expected_converged[test]);
        std::cout << "Test " << test << ": " 
                  << (test_passed ? "PASSED" : "FAILED")
                  << " (expected: " << (expected_converged[test] ? "true" : "false")
                  << ", actual: " << (converged ? "true" : "false") << ")" << std::endl;
        
        if (test_passed) passed++;
    }}
    
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Result: " << passed << "/" << NUM_TESTS << " tests passed" << std::endl;
    
    return (passed == NUM_TESTS) ? 0 : 1;
}}"""
    
    def _generate_simple_testbench(self):
        """Generate simple testbench for other functions"""
        return f"""#include <iostream>
#include "{self.function_name}.h"
#include "test_data.h"

int main() {{
    std::cout << "{self.function_name.title().replace('_', ' ')} Function Validation" << std::endl;
    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;
    std::cout << "========================================" << std::endl;
    
    // TODO: Implement specific test logic for {self.function_name}
    std::cout << "Test implementation needed for {self.function_name}" << std::endl;
    
    return 0;
}}"""
    
    def generate_tcl(self):
        """Generate Vitis HLS script"""
        return f"""# TinyMPC {self.function_name} HLS ({self.precision})
open_project -reset {self.function_name}_{self.precision}
add_files {self.function_name}.cpp
add_files -tb testbench.cpp
set_top {self.function_name}
open_solution -reset "solution1"
set_part {{xc7z020clg400-1}}
create_clock -period 10
csim_design
exit"""
    
    def generate_all(self, output_dir=None):
        """Generate complete HLS project for specific function"""
        if output_dir is None:
            output_dir = f"hls_{self.function_name}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        files = {
            f"{self.function_name}.h": self.generate_header(),
            f"{self.function_name}.cpp": self.generate_implementation(),
            "test_data.h": self.generate_test_data(),
            "testbench.cpp": self.generate_testbench(),
            "csim.tcl": self.generate_tcl()
        }
        
        for filename, content in files.items():
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write(content)
        
        print(f"Generated {self.function_name} HLS ({self.precision}) in {output_dir}/")
        print(f"System: {self.solver.nx} states, {self.solver.nu} inputs, N={self.solver.N}")
        print(f"Usage: cd {output_dir} && vitis_hls -f csim.tcl")


def main():
    parser = argparse.ArgumentParser(description='Generate TinyMPC Function HLS')
    parser.add_argument('--N', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--freq', type=float, default=100.0, help='Control frequency (Hz)')
    parser.add_argument('--precision', choices=['float', 'double'], default='float', help='Data precision')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--function', 
                       choices=['forward_pass', 'backward_pass', 'update_slack', 'update_dual', 
                               'update_linear_cost', 'check_termination'],
                       default='forward_pass',
                       help='Function to generate')
    
    args = parser.parse_args()
    
    generator = TinyMPCFunctionGenerator(args.function, N=args.N, control_freq=args.freq, 
                                        precision=args.precision)
    generator.generate_all(args.output)


if __name__ == "__main__":
    main()