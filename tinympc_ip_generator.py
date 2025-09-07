#!/usr/bin/env python3
"""
TinyMPC HLS IP Core Generator
Modularized generator for TinyMPC hardware acceleration IP cores
"""

import numpy as np
import os
import argparse
from typing import Dict

# Import existing modules
from dynamics import create_dynamics_model
from tinympcref import tinympcref
from hls_generator import TinyMPCFunctionGenerator


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
        self.control_freq = control_freq
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
        
        u_min_str = ",\n".join(u_min_rows)
        u_max_str = ",\n".join(u_max_rows)
        x_min_str = ",\n".join(x_min_rows)
        x_max_str = ",\n".join(x_max_rows)
        
        return f"""#include "tinympc_solver.h"
#include <string.h>

// Hardcoded constraints as ROM (constant arrays) with dynamic size N={self.N}
const data_t U_MIN[N_MINUS_1][NU] = {{
{u_min_str}
}};

const data_t U_MAX[N_MINUS_1][NU] = {{
{u_max_str}
}};

const data_t X_MIN[N][NX] = {{
{x_min_str}
}};

const data_t X_MAX[N][NX] = {{
{x_max_str}
}};

void tinympc_solver(
    data_t main_memory[MAIN_MEMORY_SIZE],
    int max_iter,
    int check_termination_iter
) {{
#pragma HLS INTERFACE m_axi port=main_memory bundle=gmem depth={self.solver.nx + 2 * self.solver.N * self.solver.nx + 2 * (self.solver.N-1) * self.solver.nu}
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
#pragma HLS PIPELINE II=1
        x[0][i] = main_memory[X0_OFFSET + i];  // Initial state
    }}
    
    load_xref: for (int k = 0; k < N; k++) {{
        for (int i = 0; i < NX; i++) {{
#pragma HLS PIPELINE II=1
            Xref[k][i] = main_memory[XREF_OFFSET + k*NX + i];
        }}
    }}
    
    load_uref: for (int k = 0; k < N_MINUS_1; k++) {{
        for (int j = 0; j < NU; j++) {{
#pragma HLS PIPELINE II=1
            Uref[k][j] = main_memory[UREF_OFFSET + k*NU + j];
        }}
    }}
    
    // Initialize ALL workspace variables to prevent non-deterministic behavior
    init_workspace: for (int k = 0; k < N_MINUS_1; k++) {{
#pragma HLS PIPELINE II=1
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
#pragma HLS PIPELINE II=1
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
        """Generate test data using tinympcref reference"""
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
    
    return 0;
}}"""

    def generate_tcl(self):
        """Generate Vitis HLS TCL script for IP core"""
        freq_str = f"{int(self.control_freq)}Hz" if self.control_freq == int(self.control_freq) else f"{self.control_freq}Hz"
        project_name = f"tinympc_N{self.N}_{freq_str}_{self.precision}"
        
        return f"""# TinyMPC IP Core HLS Project (N={self.N}, {freq_str}, {self.precision})
open_project -reset {project_name}
add_files tinympc_solver.cpp
add_files -tb testbench.cpp
set_top tinympc_solver

open_solution -reset "solution1"
set_part {{xc7z020clg400-1}}
create_clock -period 10

# Optimization directives
# config_interface -m_axi_addr64=false
# config_interface -m_axi_auto_max_ports=false

# Run C simulation
csim_design -clean

# Optional: Run synthesis and implementation
csynth_design
# cosim_design
export_design -format ip_catalog

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
    parser.add_argument('--output', default=None, help='Output directory (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Auto-generate output directory name if not specified
    if args.output is None:
        freq_str = f"{int(args.freq)}Hz" if args.freq == int(args.freq) else f"{args.freq}Hz"
        output_dir = f"tinympcproj_N{args.N}_{freq_str}_{args.precision}"
    else:
        output_dir = args.output
    
    generator = TinyMPCIPGenerator(N=args.N, control_freq=args.freq, 
                                  precision=args.precision, max_iter=args.max_iter)
    generator.generate_all(output_dir)


if __name__ == "__main__":
    main()