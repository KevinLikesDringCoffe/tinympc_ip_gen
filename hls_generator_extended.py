#!/usr/bin/env python3
"""
Extended TinyMPC HLS Code Generator for Crazyflie Quadcopter
Supports all TinyMPC ADMM functions: forward_pass, update_slack, update_dual, 
update_linear_cost, check_termination, backward_pass
"""

import numpy as np
import os
import argparse
from tinympcref import tinympcref as TinyMPCReference
from dynamics import create_dynamics_model


class BaseFunctionGenerator:
    """Base class for HLS function generation with common utilities"""
    
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
    
    def _generate_loop_bounds_check(self, bound_var, dimension):
        """Generate bounds checking for arrays"""
        return f"if ({bound_var} >= 0 && {bound_var} < {dimension})"


class ForwardPassGenerator(BaseFunctionGenerator):
    """Generate forward pass HLS implementation"""
    
    def generate_header_content(self):
        """Generate header declarations for forward pass"""
        return f"""void forward_pass(
    data_t x[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t d[N_MINUS_1][NU]
);"""
    
    def generate_implementation(self):
        """Generate optimized forward pass implementation"""
        code = """void forward_pass(
    data_t x[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t d[N_MINUS_1][NU]
) {
    forward_loop: for (int k = 0; k < N_MINUS_1; k++) {
#pragma HLS PIPELINE
        
        // Control: u[k] = -Kinf @ x[k] - d[k]
"""
        
        # Generate control computations
        for i in range(self.nu):
            code += f"        u[k][{i}] = -("
            terms = []
            for j in range(self.nx):
                expr = self._get_optimized_expr(self.solver.Kinf, i, j, f"x[k][{j}]")
                if expr is not None:
                    terms.append(expr)
            code += " + ".join(terms) if terms else f"0.0{self.suffix}"
            code += f") - d[k][{i}];\n"
        
        code += "\n        // State: x[k+1] = A @ x[k] + B @ u[k]\n"
        
        # Generate state updates
        for i in range(self.nx):
            code += f"        x[k+1][{i}] = "
            terms = []
            
            # A @ x[k] terms
            for j in range(self.nx):
                expr = self._get_optimized_expr(self.solver.A, i, j, f"x[k][{j}]")
                if expr is not None:
                    terms.append(expr)
            
            # B @ u[k] terms
            for j in range(self.nu):
                expr = self._get_optimized_expr(self.solver.B, i, j, f"u[k][{j}]")
                if expr is not None:
                    terms.append(expr)
            
            code += " + ".join(terms) if terms else f"0.0{self.suffix}"
            code += ";\n"
        
        code += "    }\n}"
        return code


class UpdateSlackGenerator(BaseFunctionGenerator):
    """Generate update slack variables HLS implementation"""
    
    def generate_header_content(self):
        """Generate header declarations for update slack"""
        return f"""void update_slack(
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
);"""
    
    def generate_implementation(self):
        """Generate optimized update slack implementation"""
        return f"""void update_slack(
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
) {{
    // Update input slack variables
    update_u_slack: for (int k = 0; k < N_MINUS_1; k++) {{
#pragma HLS PIPELINE
        for (int j = 0; j < NU; j++) {{
#pragma HLS UNROLL
            data_t temp_u = u[k][j] + y[k][j];
            znew[k][j] = (temp_u > u_max[k][j]) ? u_max[k][j] : 
                        (temp_u < u_min[k][j]) ? u_min[k][j] : temp_u;
        }}
    }}
    
    // Update state slack variables
    update_x_slack: for (int k = 0; k < N; k++) {{
#pragma HLS PIPELINE
        for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
            data_t temp_x = x[k][i] + g[k][i];
            vnew[k][i] = (temp_x > x_max[k][i]) ? x_max[k][i] : 
                        (temp_x < x_min[k][i]) ? x_min[k][i] : temp_x;
        }}
    }}
}}"""


class UpdateDualGenerator(BaseFunctionGenerator):
    """Generate update dual variables HLS implementation"""
    
    def generate_header_content(self):
        """Generate header declarations for update dual"""
        return f"""void update_dual(
    data_t y[N_MINUS_1][NU],
    data_t g[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t x[N][NX],
    data_t znew[N_MINUS_1][NU],
    data_t vnew[N][NX]
);"""
    
    def generate_implementation(self):
        """Generate optimized update dual implementation"""
        return f"""void update_dual(
    data_t y[N_MINUS_1][NU],
    data_t g[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t x[N][NX],
    data_t znew[N_MINUS_1][NU],
    data_t vnew[N][NX]
) {{
    // Update input dual variables
    update_y_dual: for (int k = 0; k < N_MINUS_1; k++) {{
#pragma HLS PIPELINE
        for (int j = 0; j < NU; j++) {{
#pragma HLS UNROLL
            y[k][j] = y[k][j] + u[k][j] - znew[k][j];
        }}
    }}
    
    // Update state dual variables
    update_g_dual: for (int k = 0; k < N; k++) {{
#pragma HLS PIPELINE
        for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
            g[k][i] = g[k][i] + x[k][i] - vnew[k][i];
        }}
    }}
}}"""


class UpdateLinearCostGenerator(BaseFunctionGenerator):
    """Generate update linear cost HLS implementation"""
    
    def generate_header_content(self):
        """Generate header declarations for update linear cost"""
        return f"""void update_linear_cost(
    data_t r[N_MINUS_1][NU],
    data_t q[N][NX],
    data_t p[N][NX],
    data_t Uref[N_MINUS_1][NU],
    data_t Xref[N][NX],
    data_t Q_diag[NX],
    data_t R_diag[NU],
    data_t rho,
    data_t znew[N_MINUS_1][NU],
    data_t vnew[N][NX],
    data_t y[N_MINUS_1][NU],
    data_t g[N][NX]
);"""
    
    def generate_implementation(self):
        """Generate optimized update linear cost implementation with hardcoded Pinf"""
        code = f"""void update_linear_cost(
    data_t r[N_MINUS_1][NU],
    data_t q[N][NX],
    data_t p[N][NX],
    data_t Uref[N_MINUS_1][NU],
    data_t Xref[N][NX],
    data_t Q_diag[NX],
    data_t R_diag[NU],
    data_t rho,
    data_t znew[N_MINUS_1][NU],
    data_t vnew[N][NX],
    data_t y[N_MINUS_1][NU],
    data_t g[N][NX]
) {{
    // Update input linear cost
    update_r_cost: for (int k = 0; k < N_MINUS_1; k++) {{
#pragma HLS PIPELINE
        for (int j = 0; j < NU; j++) {{
#pragma HLS UNROLL
            r[k][j] = -Uref[k][j] * R_diag[j] - rho * (znew[k][j] - y[k][j]);
        }}
    }}
    
    // Update state linear cost
    update_q_cost: for (int k = 0; k < N_MINUS_1; k++) {{
#pragma HLS PIPELINE
        for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
            q[k][i] = -Xref[k][i] * Q_diag[i] - rho * (vnew[k][i] - g[k][i]);
        }}
    }}
    
    // Terminal state cost (k = N-1)
    for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
        q[N-1][i] = -Xref[N-1][i] * Q_diag[i] - rho * (vnew[N-1][i] - g[N-1][i]);
    }}
    
    // Terminal costate: p[N-1] = -Pinf.T @ Xref[N-1] - rho * (vnew[N-1] - g[N-1])
    // Hardcode Pinf.T matrix multiplication
"""
        
        # Generate hardcoded Pinf.T @ Xref[N-1] multiplication
        for i in range(self.nx):
            code += f"    p[N-1][{i}] = "
            terms = []
            
            # Add -Pinf.T @ Xref[N-1] terms
            for j in range(self.nx):
                expr = self._get_optimized_expr(-self.solver.Pinf.T, i, j, f"Xref[N-1][{j}]")
                if expr is not None:
                    terms.append(expr)
            
            # Add -rho * (vnew[N-1][i] - g[N-1][i]) term
            terms.append(f"-rho * (vnew[N-1][{i}] - g[N-1][{i}])")
            
            code += " + ".join(terms) if terms else f"-rho * (vnew[N-1][{i}] - g[N-1][{i}])"
            code += ";\n"
        
        code += "}"
        return code


class CheckTerminationGenerator(BaseFunctionGenerator):
    """Generate check termination HLS implementation"""
    
    def generate_header_content(self):
        """Generate header declarations for check termination"""
        return f"""bool check_termination(
    data_t x[N][NX],
    data_t vnew[N][NX],
    data_t v[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t znew[N_MINUS_1][NU],
    data_t z[N_MINUS_1][NU],
    data_t rho,
    data_t abs_pri_tol,
    data_t abs_dua_tol
);"""
    
    def generate_implementation(self):
        """Generate optimized check termination implementation"""
        return f"""bool check_termination(
    data_t x[N][NX],
    data_t vnew[N][NX],
    data_t v[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t znew[N_MINUS_1][NU],
    data_t z[N_MINUS_1][NU],
    data_t rho,
    data_t abs_pri_tol,
    data_t abs_dua_tol
) {{
    data_t primal_residual_state = {self._format_value(0.0)};
    data_t dual_residual_state = {self._format_value(0.0)};
    data_t primal_residual_input = {self._format_value(0.0)};
    data_t dual_residual_input = {self._format_value(0.0)};
    
    // Compute state residuals
    state_residuals: for (int k = 0; k < N; k++) {{
#pragma HLS PIPELINE
        for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
            data_t pri_res = x[k][i] - vnew[k][i];
            data_t dua_res = v[k][i] - vnew[k][i];
            
            pri_res = (pri_res < {self._format_value(0.0)}) ? -pri_res : pri_res;
            dua_res = (dua_res < {self._format_value(0.0)}) ? -dua_res : dua_res;
            
            primal_residual_state = (pri_res > primal_residual_state) ? pri_res : primal_residual_state;
            dual_residual_state = (dua_res > dual_residual_state) ? dua_res : dual_residual_state;
        }}
    }}
    
    // Compute input residuals
    input_residuals: for (int k = 0; k < N_MINUS_1; k++) {{
#pragma HLS PIPELINE
        for (int j = 0; j < NU; j++) {{
#pragma HLS UNROLL
            data_t pri_res = u[k][j] - znew[k][j];
            data_t dua_res = z[k][j] - znew[k][j];
            
            pri_res = (pri_res < {self._format_value(0.0)}) ? -pri_res : pri_res;
            dua_res = (dua_res < {self._format_value(0.0)}) ? -dua_res : dua_res;
            
            primal_residual_input = (pri_res > primal_residual_input) ? pri_res : primal_residual_input;
            dual_residual_input = (dua_res > dual_residual_input) ? dua_res : dual_residual_input;
        }}
    }}
    
    dual_residual_state *= rho;
    dual_residual_input *= rho;
    
    return (primal_residual_state < abs_pri_tol && 
            primal_residual_input < abs_pri_tol &&
            dual_residual_state < abs_dua_tol && 
            dual_residual_input < abs_dua_tol);
}}"""


class BackwardPassGenerator(BaseFunctionGenerator):
    """Generate backward pass HLS implementation"""
    
    def generate_header_content(self):
        """Generate header declarations for backward pass"""
        return f"""void backward_pass(
    data_t d[N_MINUS_1][NU],
    data_t p[N][NX],
    data_t q[N][NX],
    data_t r[N_MINUS_1][NU]
);"""
    
    def generate_implementation(self):
        """Generate optimized backward pass implementation with hardcoded matrices"""
        code = f"""void backward_pass(
    data_t d[N_MINUS_1][NU],
    data_t p[N][NX],
    data_t q[N][NX],
    data_t r[N_MINUS_1][NU]
) {{
    backward_loop: for (int k = N-2; k >= 0; k--) {{
#pragma HLS PIPELINE
        
        // Compute d[k] = Quu_inv @ (B.T @ p[k+1] + r[k])
        // Hardcode all matrix multiplications
"""
        
        # Generate d[k] computation: expand Quu_inv @ (B.T @ p[k+1] + r[k])
        for j in range(self.nu):
            code += f"        d[k][{j}] = "
            terms = []
            
            # For each d[k][j], compute sum over l of Quu_inv[j,l] * (sum_i B.T[l,i] * p[k+1][i] + r[k][l])
            for l in range(self.nu):
                if abs(self.solver.Quu_inv[j, l]) > 1e-10:
                    # Compute B.T @ p[k+1] for column l
                    bt_terms = []
                    for i in range(self.nx):
                        bt_expr = self._get_optimized_expr(self.solver.B.T, l, i, f"p[k+1][{i}]")
                        if bt_expr is not None:
                            bt_terms.append(bt_expr)
                    
                    # Combine (B.T @ p[k+1])[l] + r[k][l]
                    if bt_terms:
                        inner_expr = f"({' + '.join(bt_terms)} + r[k][{l}])"
                    else:
                        inner_expr = f"r[k][{l}]"
                    
                    # Multiply by Quu_inv[j,l]
                    quu_expr = self._get_optimized_expr(self.solver.Quu_inv, j, l, inner_expr)
                    if quu_expr is not None:
                        terms.append(quu_expr)
            
            code += " + ".join(terms) if terms else f"{self._format_value(0.0)}"
            code += ";\n"
        
        code += "\n        // Compute p[k] = q[k] + AmBKt @ p[k+1] - Kinf.T @ r[k]\n"
        
        # Generate p[k] computation with hardcoded AmBKt and Kinf.T
        for i in range(self.nx):
            code += f"        p[k][{i}] = q[k][{i}]"
            
            # Add AmBKt @ p[k+1] terms
            for j in range(self.nx):
                expr = self._get_optimized_expr(self.solver.AmBKt, i, j, f"p[k+1][{j}]")
                if expr is not None:
                    code += f" + {expr}"
            
            # Subtract Kinf.T @ r[k] terms  
            for j in range(self.nu):
                expr = self._get_optimized_expr(self.solver.Kinf.T, i, j, f"r[k][{j}]")
                if expr is not None:
                    code += f" - {expr}"
            
            code += ";\n"
        
        code += "    }\n}"
        return code


class TinyMPCHLSGenerator:
    """Unified HLS generator for complete TinyMPC solver"""
    
    def __init__(self, N=5, control_freq=100.0, precision='float', functions=None):
        """Initialize with Crazyflie dynamics"""
        dynamics = create_dynamics_model("crazyflie")
        A, B = dynamics.generate_system_matrices(control_freq)
        Q, R = dynamics.generate_cost_matrices()
        
        self.solver = TinyMPCReference()
        self.solver.setup(A, B, Q, R, N, rho=1.0)
        
        self.precision = precision
        self.data_type = 'float' if precision == 'float' else 'double'
        self.suffix = 'f' if precision == 'float' else ''
        
        # Initialize all function generators
        self.functions = functions or ['forward_pass', 'update_slack', 'update_dual', 
                                     'update_linear_cost', 'check_termination', 'backward_pass']
        
        self.generators = {
            'forward_pass': ForwardPassGenerator(self.solver, precision),
            'update_slack': UpdateSlackGenerator(self.solver, precision),
            'update_dual': UpdateDualGenerator(self.solver, precision),
            'update_linear_cost': UpdateLinearCostGenerator(self.solver, precision),
            'check_termination': CheckTerminationGenerator(self.solver, precision),
            'backward_pass': BackwardPassGenerator(self.solver, precision)
        }
    
    def generate_header(self):
        """Generate complete header file"""
        header = f"""#ifndef TINYMPC_SOLVER_H
#define TINYMPC_SOLVER_H

#define NX {self.solver.nx}
#define NU {self.solver.nu}
#define N {self.solver.N}
#define N_MINUS_1 {self.solver.N-1}

typedef {self.data_type} data_t;

"""
        
        # Add function declarations
        for func_name in self.functions:
            if func_name in self.generators:
                header += self.generators[func_name].generate_header_content() + "\n\n"
        
        header += "#endif"
        return header
    
    def generate_implementation(self):
        """Generate complete implementation file"""
        impl = '#include "tinympc_solver.h"\n#include <algorithm>\n\n'
        
        for func_name in self.functions:
            if func_name in self.generators:
                impl += self.generators[func_name].generate_implementation() + "\n\n"
        
        return impl
    
    def _format_value(self, value):
        """Format numerical value with appropriate precision"""
        if self.precision == 'float':
            return f"{value:.10f}{self.suffix}"
        else:
            return f"{value:.15f}{self.suffix}"
    
    def generate_test_data(self):
        """Generate test data based on selected functions"""
        dtype = np.float32 if self.precision == 'float' else np.float64
        
        # Create test scenarios
        test_cases = [
            {
                'name': 'hover_stability',
                'x0': np.zeros(self.solver.nx, dtype=dtype),
                'Xref': np.zeros((self.solver.N, self.solver.nx), dtype=dtype),
                'Uref': np.zeros((self.solver.N-1, self.solver.nu), dtype=dtype)
            },
            {
                'name': 'position_tracking', 
                'x0': np.array([0.1, 0.1, 0.05] + [0.0]*(self.solver.nx-3), dtype=dtype),
                'Xref': np.zeros((self.solver.N, self.solver.nx), dtype=dtype),
                'Uref': np.zeros((self.solver.N-1, self.solver.nu), dtype=dtype)
            },
            {
                'name': 'random_case',
                'x0': np.random.randn(self.solver.nx).astype(dtype) * 0.02,
                'Xref': np.random.randn(self.solver.N, self.solver.nx).astype(dtype) * 0.01,
                'Uref': np.random.randn(self.solver.N-1, self.solver.nu).astype(dtype) * 0.005
            }
        ]
        
        # Generate test data based on which functions are selected
        results = []
        for case in test_cases:
            self.solver.set_x0(case['x0'])
            self.solver.set_x_ref(case['Xref']) 
            self.solver.set_u_ref(case['Uref'])
            
            # Initialize solver state for intermediate function testing
            if any(func in ['update_slack', 'update_dual', 'update_linear_cost', 'backward_pass'] for func in self.functions):
                # Run a few ADMM iterations to get realistic intermediate states
                self.solver.solve()
            else:
                # Just do forward pass
                self.solver._forward_pass()
            
            # Collect all possible outputs that any function might need
            result = {
                'x': self.solver.x.copy(),
                'u': self.solver.u.copy(),
            }
            
            # Add intermediate variables if needed
            if hasattr(self.solver, 'znew'):
                result['znew'] = self.solver.znew.copy()
            if hasattr(self.solver, 'vnew'):
                result['vnew'] = self.solver.vnew.copy()
            if hasattr(self.solver, 'y'):
                result['y'] = self.solver.y.copy()
            if hasattr(self.solver, 'g'):
                result['g'] = self.solver.g.copy()
            if hasattr(self.solver, 'r'):
                result['r'] = self.solver.r.copy()
            if hasattr(self.solver, 'q'):
                result['q'] = self.solver.q.copy()
            if hasattr(self.solver, 'p'):
                result['p'] = self.solver.p.copy()
            if hasattr(self.solver, 'd'):
                result['d'] = self.solver.d.copy()
            
            results.append(result)
        
        return self._generate_test_data_header(test_cases, results)
    
    def _generate_test_data_header(self, test_cases, results):
        """Generate test data header file"""
        header_parts = []
        header_parts.append("#ifndef TEST_DATA_H")
        header_parts.append("#define TEST_DATA_H")
        header_parts.append("")
        header_parts.append('#include "tinympc_solver.h"')
        header_parts.append("")
        header_parts.append("#define NUM_TESTS 3")
        header_parts.append("")
        
        # Helper function to format array values
        def format_array_values(arr, num_elements=None):
            if num_elements is None:
                values = [self._format_value(val) for val in arr.flatten()]
            else:
                values = [self._format_value(arr.flatten()[i] if i < arr.size else 0.0) 
                         for i in range(num_elements)]
            return ", ".join(values)
        
        # Generate all test data arrays
        arrays_to_generate = [
            ('test_x0', [case['x0'] for case in test_cases], f'[NUM_TESTS][NX]'),
            ('test_Xref', [case['Xref'] for case in test_cases], f'[NUM_TESTS][N][NX]'),
            ('test_Uref', [case['Uref'] for case in test_cases], f'[NUM_TESTS][N_MINUS_1][NU]'),
            ('expected_x', [result['x'] for result in results], f'[NUM_TESTS][N][NX]'),
            ('expected_u', [result['u'] for result in results], f'[NUM_TESTS][N_MINUS_1][NU]'),
        ]
        
        for array_name, test_data, dimensions in arrays_to_generate:
            header_parts.append(f"{self.data_type} {array_name}{dimensions} = {{")
            
            for i, data in enumerate(test_data):
                header_parts.append(f"    {{  // {test_cases[i]['name']}")
                
                if len(data.shape) == 1:
                    # 1D array
                    values = format_array_values(data)
                    header_parts.append(f"        {{{values}}}")
                elif len(data.shape) == 2:
                    # 2D array
                    for j in range(data.shape[0]):
                        values = format_array_values(data[j])
                        header_parts.append(f"        {{{values}}},")
                
                header_parts.append("    },")
            
            header_parts.append("};")
            header_parts.append("")
        
        header_parts.append("#endif")
        return "\n".join(header_parts)
    
    def generate_testbench(self):
        """Generate comprehensive testbench"""
        return f"""#include <iostream>
#include <cmath>
#include "tinympc_solver.h"
#include "test_data.h"

const {self.data_type} TOLERANCE = {self.generators['forward_pass'].tolerance};

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
    std::cout << "TinyMPC Complete Solver Validation" << std::endl;
    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;
    std::cout << "Functions: forward_pass update_slack update_dual update_linear_cost check_termination backward_pass" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int total_tests = 0;
    int passed_tests = 0;
    
    const char* names[NUM_TESTS] = {{"hover_stability", "position_tracking", "full_trajectory"}};
    
    for (int test = 0; test < NUM_TESTS; test++) {{
        std::cout << "\\nTest " << (test + 1) << ": " << names[test] << std::endl;
        
        // Initialize test data
        data_t x[N][NX], u[N_MINUS_1][NU];
        data_t znew[N_MINUS_1][NU], vnew[N][NX];
        data_t y[N_MINUS_1][NU], g[N][NX];
        data_t r[N_MINUS_1][NU], q[N][NX], p[N][NX], d[N_MINUS_1][NU];
        
        // Copy initial conditions
        for (int i = 0; i < NX; i++) {{
            x[0][i] = test_x0[test][i];
        }}
        
        // Test individual functions
        // Test forward_pass if available
            forward_pass(x, u, d);
            total_tests++;
            if (validate_arrays((data_t*)u, (data_t*)expected_u[test], 
                              N_MINUS_1 * NU, "Forward Pass")) {{
                passed_tests++;
            }}
        }}
        
        // Add more function tests as needed...
    }}
    
    std::cout << "\\n========================================" << std::endl;
    std::cout << "Result: " << passed_tests << "/" << total_tests << " tests passed" << std::endl;
    
    return (passed_tests == total_tests) ? 0 : 1;
}}"""
    
    def generate_tcl(self):
        """Generate Vitis HLS script"""
        return f"""# TinyMPC Complete Solver HLS ({self.precision})
open_project -reset tinympc_solver_{self.precision}
add_files tinympc_solver.cpp
add_files -tb testbench.cpp
set_top {self.functions[0] if self.functions else 'forward_pass'}
open_solution -reset "solution1"
set_part {{xc7z020clg400-1}}
create_clock -period 10
csim_design
exit"""
    
    def generate_all(self, output_dir="hls_tinympc_complete"):
        """Generate complete HLS project"""
        os.makedirs(output_dir, exist_ok=True)
        
        files = {
            "tinympc_solver.h": self.generate_header(),
            "tinympc_solver.cpp": self.generate_implementation(),
            "test_data.h": self.generate_test_data(),
            "testbench.cpp": self.generate_testbench(),
            "csim.tcl": self.generate_tcl()
        }
        
        for filename, content in files.items():
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write(content)
        
        print(f"Generated TinyMPC Complete Solver HLS ({self.precision}) in {output_dir}/")
        print(f"System: {self.solver.nx} states, {self.solver.nu} inputs, N={self.solver.N}")
        print(f"Functions: {', '.join(self.functions)}")
        print(f"Usage: cd {output_dir} && vitis_hls -f csim.tcl")


def main():
    parser = argparse.ArgumentParser(description='Generate TinyMPC Complete Solver HLS')
    parser.add_argument('--N', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--freq', type=float, default=100.0, help='Control frequency (Hz)')
    parser.add_argument('--precision', choices=['float', 'double'], default='float', help='Data precision')
    parser.add_argument('--output', default='hls_tinympc_complete', help='Output directory')
    parser.add_argument('--functions', nargs='*', 
                       choices=['forward_pass', 'update_slack', 'update_dual', 
                               'update_linear_cost', 'check_termination', 'backward_pass'],
                       help='Functions to generate (default: all)')
    
    args = parser.parse_args()
    
    generator = TinyMPCHLSGenerator(N=args.N, control_freq=args.freq, 
                                   precision=args.precision, functions=args.functions)
    generator.generate_all(args.output)


if __name__ == "__main__":
    main()