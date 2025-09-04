#!/usr/bin/env python3
"""
TinyMPC HLS Code Generator for Crazyflie Quadcopter
"""

import numpy as np
import os
import argparse
from tinympcref import tinympcref as TinyMPCReference
from dynamics import create_dynamics_model


class CrazyflieTinyMPCGenerator:
    """HLS code generator for Crazyflie quadcopter"""
    
    def __init__(self, N=5, control_freq=100.0, precision='float'):
        """Initialize with Crazyflie dynamics"""
        dynamics = create_dynamics_model("crazyflie")
        A, B = dynamics.generate_system_matrices(control_freq)
        Q, R = dynamics.generate_cost_matrices()
        
        self.solver = TinyMPCReference()
        self.solver.setup(A, B, Q, R, N, rho=1.0)
        self.A = self.solver.A
        self.B = self.solver.B
        self.Kinf = self.solver.Kinf
        
        self.nx = self.solver.nx
        self.nu = self.solver.nu
        self.N = self.solver.N
        self.precision = precision
        
        self.data_type = 'float' if precision == 'float' else 'double'
        self.suffix = 'f' if precision == 'float' else ''
        self.tolerance = '1e-3f' if precision == 'float' else '1e-6'
        
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
            if self.precision == 'float':
                return f"{value:.10f}{self.suffix} * {var_name}"
            else:
                return f"{value:.15f}{self.suffix} * {var_name}"
    
    def generate_header(self):
        """Generate header file"""
        return f"""#ifndef FORWARD_PASS_H
#define FORWARD_PASS_H

#define NX {self.nx}
#define NU {self.nu}
#define N {self.N}
#define N_MINUS_1 {self.N-1}

typedef {self.data_type} data_t;

void forward_pass(
    data_t x[N][NX],
    data_t u[N_MINUS_1][NU],
    data_t d[N_MINUS_1][NU]
);

#endif"""
    
    def generate_forward_pass(self):
        """Generate optimized forward pass implementation"""
        code = """#include "forward_pass.h"

void forward_pass(
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
                expr = self._get_optimized_expr(self.Kinf, i, j, f"x[k][{j}]")
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
                expr = self._get_optimized_expr(self.A, i, j, f"x[k][{j}]")
                if expr is not None:
                    terms.append(expr)
            
            # B @ u[k] terms
            for j in range(self.nu):
                expr = self._get_optimized_expr(self.B, i, j, f"u[k][{j}]")
                if expr is not None:
                    terms.append(expr)
            
            code += " + ".join(terms) if terms else f"0.0{self.suffix}"
            code += ";\n"
        
        code += "    }\n}"
        return code
    
    def generate_test_data(self):
        """Generate test data"""
        dtype = np.float32 if self.precision == 'float' else np.float64
        
        test_cases = [
            {
                'name': 'hover_stability',
                'x0': np.zeros(self.nx, dtype=dtype),
                'd': np.zeros((self.N-1, self.nu), dtype=dtype)
            },
            {
                'name': 'position_tracking',
                'x0': np.array([0.1, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype),
                'd': np.zeros((self.N-1, self.nu), dtype=dtype)
            },
            {
                'name': 'disturbance_rejection',
                'x0': np.random.randn(self.nx).astype(dtype) * 0.02,
                'd': np.random.randn(self.N-1, self.nu).astype(dtype) * 0.01
            }
        ]
        
        # Generate reference outputs
        results = []
        for case in test_cases:
            self.solver.set_x0(case['x0'])
            self.solver.d = case['d']
            self.solver._forward_pass()
            results.append({'x': self.solver.x.copy(), 'u': self.solver.u.copy()})
        
        # Generate header - build it step by step
        header_parts = []
        header_parts.append("#ifndef TEST_DATA_H")
        header_parts.append("#define TEST_DATA_H")
        header_parts.append("")
        header_parts.append('#include "forward_pass.h"')
        header_parts.append("")
        header_parts.append("#define NUM_TESTS 3")
        header_parts.append("")
        
        # Test initial states
        header_parts.append(f"{self.data_type} test_x0[NUM_TESTS][NX] = {{")
        for i, case in enumerate(test_cases):
            if self.precision == 'float':
                values = ", ".join([f"{case['x0'][j]:.10f}{self.suffix}" for j in range(self.nx)])
            else:
                values = ", ".join([f"{case['x0'][j]:.15f}{self.suffix}" for j in range(self.nx)])
            header_parts.append(f"    {{{values}}},  // {case['name']}")
        header_parts.append("};")
        header_parts.append("")
        
        # Test disturbances
        header_parts.append(f"{self.data_type} test_d[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, case in enumerate(test_cases):
            header_parts.append(f"    {{  // {case['name']}")
            for k in range(self.N-1):
                if self.precision == 'float':
                    values = ", ".join([f"{case['d'][k,j]:.10f}{self.suffix}" for j in range(self.nu)])
                else:
                    values = ", ".join([f"{case['d'][k,j]:.15f}{self.suffix}" for j in range(self.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Expected states
        header_parts.append(f"{self.data_type} expected_x[NUM_TESTS][N][NX] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.N):
                if self.precision == 'float':
                    values = ", ".join([f"{result['x'][k,j]:.10f}{self.suffix}" for j in range(self.nx)])
                else:
                    values = ", ".join([f"{result['x'][k,j]:.15f}{self.suffix}" for j in range(self.nx)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        
        # Expected controls
        header_parts.append(f"{self.data_type} expected_u[NUM_TESTS][N_MINUS_1][NU] = {{")
        for i, result in enumerate(results):
            header_parts.append(f"    {{  // {test_cases[i]['name']}")
            for k in range(self.N-1):
                if self.precision == 'float':
                    values = ", ".join([f"{result['u'][k,j]:.10f}{self.suffix}" for j in range(self.nu)])
                else:
                    values = ", ".join([f"{result['u'][k,j]:.15f}{self.suffix}" for j in range(self.nu)])
                header_parts.append(f"        {{{values}}},")
            header_parts.append("    },")
        header_parts.append("};")
        header_parts.append("")
        header_parts.append("#endif")
        
        return "\n".join(header_parts)
    
    def generate_testbench(self):
        """Generate validation testbench"""
        zero_val = f"0.0{self.suffix}"
        
        testbench_parts = []
        testbench_parts.append("#include <iostream>")
        testbench_parts.append("#include <cmath>")
        testbench_parts.append('#include "forward_pass.h"')
        testbench_parts.append('#include "test_data.h"')
        testbench_parts.append("")
        testbench_parts.append(f"const {self.data_type} TOLERANCE = {self.tolerance};")
        testbench_parts.append("")
        
        # Validation functions
        testbench_parts.append("bool validate_trajectory(data_t actual[N][NX], data_t expected[N][NX], const char* name) {")
        testbench_parts.append(f"    data_t max_error = {zero_val};")
        testbench_parts.append("    for (int i = 0; i < N; i++) {")
        testbench_parts.append("        for (int j = 0; j < NX; j++) {")
        testbench_parts.append("            data_t error = std::abs(actual[i][j] - expected[i][j]);")
        testbench_parts.append("            max_error = std::max(max_error, error);")
        testbench_parts.append("        }")
        testbench_parts.append("    }")
        testbench_parts.append("    ")
        testbench_parts.append("    bool passed = (max_error <= TOLERANCE);")
        testbench_parts.append('    std::cout << name << " - " << (passed ? "PASSED" : "FAILED")')
        testbench_parts.append('              << " (max error: " << max_error << ")" << std::endl;')
        testbench_parts.append("    return passed;")
        testbench_parts.append("}")
        testbench_parts.append("")
        
        testbench_parts.append("bool validate_control(data_t actual[N_MINUS_1][NU], data_t expected[N_MINUS_1][NU], const char* name) {")
        testbench_parts.append(f"    data_t max_error = {zero_val};")
        testbench_parts.append("    for (int i = 0; i < N_MINUS_1; i++) {")
        testbench_parts.append("        for (int j = 0; j < NU; j++) {")
        testbench_parts.append("            data_t error = std::abs(actual[i][j] - expected[i][j]);")
        testbench_parts.append("            max_error = std::max(max_error, error);")
        testbench_parts.append("        }")
        testbench_parts.append("    }")
        testbench_parts.append("    ")
        testbench_parts.append("    bool passed = (max_error <= TOLERANCE);")
        testbench_parts.append('    std::cout << name << " - " << (passed ? "PASSED" : "FAILED")')
        testbench_parts.append('              << " (max error: " << max_error << ")" << std::endl;')
        testbench_parts.append("    return passed;")
        testbench_parts.append("}")
        testbench_parts.append("")
        
        # Main function
        testbench_parts.append("int main() {")
        testbench_parts.append('    std::cout << "Crazyflie TinyMPC Forward Pass Validation" << std::endl;')
        testbench_parts.append('    std::cout << "System: " << NX << " states, " << NU << " inputs, N=" << N << std::endl;')
        testbench_parts.append('    std::cout << "========================================" << std::endl;')
        testbench_parts.append("    ")
        testbench_parts.append("    int passed = 0;")
        testbench_parts.append('    const char* names[NUM_TESTS] = {"hover_stability", "position_tracking", "disturbance_rejection"};')
        testbench_parts.append("    ")
        testbench_parts.append("    for (int test = 0; test < NUM_TESTS; test++) {")
        testbench_parts.append('        std::cout << "\\nTest " << (test + 1) << ": " << names[test] << std::endl;')
        testbench_parts.append("        ")
        testbench_parts.append("        data_t x[N][NX], u[N_MINUS_1][NU];")
        testbench_parts.append("        ")
        testbench_parts.append("        // Initialize")
        testbench_parts.append("        for (int i = 0; i < NX; i++) {")
        testbench_parts.append("            x[0][i] = test_x0[test][i];")
        testbench_parts.append("        }")
        testbench_parts.append("        ")
        testbench_parts.append("        // Run HLS")
        testbench_parts.append("        forward_pass(x, u, test_d[test]);")
        testbench_parts.append("        ")
        testbench_parts.append("        // Validate")
        testbench_parts.append('        bool x_ok = validate_trajectory(x, expected_x[test], "States");')
        testbench_parts.append('        bool u_ok = validate_control(u, expected_u[test], "Controls");')
        testbench_parts.append("        ")
        testbench_parts.append("        if (x_ok && u_ok) passed++;")
        testbench_parts.append("    }")
        testbench_parts.append("    ")
        testbench_parts.append('    std::cout << "\\n========================================" << std::endl;')
        testbench_parts.append('    std::cout << "Result: " << passed << "/" << NUM_TESTS << " tests passed" << std::endl;')
        testbench_parts.append("    ")
        testbench_parts.append("    return (passed == NUM_TESTS) ? 0 : 1;")
        testbench_parts.append("}")
        
        return "\n".join(testbench_parts)
    
    def generate_tcl(self):
        """Generate Vitis HLS script"""
        return f"""# Crazyflie TinyMPC Forward Pass HLS ({self.precision})
open_project -reset crazyflie_tinympc_{self.precision}
add_files forward_pass.cpp
add_files -tb testbench.cpp
set_top forward_pass
open_solution -reset "solution1"
set_part {{xc7z020clg400-1}}
create_clock -period 10
csim_design
exit"""
    
    def generate_all(self, output_dir="hls_crazyflie"):
        """Generate complete HLS project"""
        os.makedirs(output_dir, exist_ok=True)
        
        files = {
            "forward_pass.h": self.generate_header(),
            "forward_pass.cpp": self.generate_forward_pass(),
            "test_data.h": self.generate_test_data(),
            "testbench.cpp": self.generate_testbench(),
            "csim.tcl": self.generate_tcl()
        }
        
        for filename, content in files.items():
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write(content)
        
        print(f"Generated Crazyflie TinyMPC HLS ({self.precision}) in {output_dir}/")
        print(f"System: {self.nx} states, {self.nu} inputs, N={self.N}")
        print(f"Sparsity: A={((abs(self.A) < 1e-10).sum() / self.A.size * 100):.1f}% zeros")
        print(f"Usage: cd {output_dir} && vitis_hls -f csim.tcl")


def main():
    parser = argparse.ArgumentParser(description='Generate Crazyflie TinyMPC HLS')
    parser.add_argument('--N', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--freq', type=float, default=100.0, help='Control frequency (Hz)')
    parser.add_argument('--precision', choices=['float', 'double'], default='float', help='Data precision')
    parser.add_argument('--output', default='hls_crazyflie', help='Output directory')
    
    args = parser.parse_args()
    
    generator = CrazyflieTinyMPCGenerator(N=args.N, control_freq=args.freq, precision=args.precision)
    generator.generate_all(args.output)


if __name__ == "__main__":
    main()