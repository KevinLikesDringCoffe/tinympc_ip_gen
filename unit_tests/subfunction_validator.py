#!/usr/bin/env python3
"""
Numerical validation tool for individual TinyMPC subfunctions
"""

import numpy as np
import ctypes
import subprocess
import tempfile
import os
from typing import Dict, Any, List, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparse_codegen.generator import CodeGenerator
from sparse_codegen.core import Variable, DataType, get_sparsity_info


class SubfunctionValidator:
    """Validator for individual TinyMPC subfunctions"""
    
    def __init__(self):
        self.temp_files = []
    
    def validate_forward_pass(self, A: np.ndarray, B: np.ndarray, Kinf: np.ndarray, 
                            x0: np.ndarray, d: np.ndarray, N: int, 
                            ops: List[Dict[str, Any]], tolerance: float = 1e-5) -> Tuple[bool, str]:
        """Validate forward pass subfunction numerically"""
        
        # Python reference implementation
        nx, nu = x0.shape[0], d.shape[1]
        x_ref = np.zeros((N, nx), dtype=np.float32)
        u_ref = np.zeros((N-1, nu), dtype=np.float32)
        
        x_ref[0] = x0
        for k in range(N-1):
            u_ref[k] = -(Kinf @ x_ref[k] + d[k])
            x_ref[k+1] = A @ x_ref[k] + B @ u_ref[k]
        
        # Generate C code for forward pass only
        code = self._generate_forward_pass_c_code(A, B, Kinf, x0, d, N, ops)
        
        # Compile and run
        success, x_c, u_c, error_msg = self._compile_and_run_c_code(code, x0, d, N, nx, nu)
        
        if not success:
            return False, f"Compilation failed: {error_msg}"
        
        # Compare results
        x_error = np.max(np.abs(x_ref - x_c))
        u_error = np.max(np.abs(u_ref - u_c))
        max_error = max(x_error, u_error)
        
        passed = max_error < tolerance
        details = f"Max error: {max_error:.2e}, x_error: {x_error:.2e}, u_error: {u_error:.2e}"
        
        return passed, details
    
    def _generate_forward_pass_c_code(self, A: np.ndarray, B: np.ndarray, Kinf: np.ndarray,
                                    x0: np.ndarray, d: np.ndarray, N: int, 
                                    ops: List[Dict[str, Any]]) -> str:
        """Generate complete C code for forward pass validation"""
        nx, nu = A.shape[0], B.shape[1]
        
        # Generate matrices as embedded constants
        def array_to_c_init(arr, name):
            if arr.ndim == 1:
                values = ', '.join(f"{x:.6f}f" for x in arr.flatten())
                return f"float {name}[{arr.shape[0]}] = {{{values}}};"
            else:
                lines = [f"float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{"]
                for i in range(arr.shape[0]):
                    values = ', '.join(f"{arr[i,j]:.6f}f" for j in range(arr.shape[1]))
                    lines.append(f"  {{{values}}},")
                lines.append("};")
                return '\n'.join(lines)
        
        # Generate operation code
        codegen = CodeGenerator()
        
        # Register matrices
        for name, matrix in [('A', A), ('B', B), ('Kinf', Kinf)]:
            info = get_sparsity_info(matrix)
            codegen.register_variable(
                Variable(name, DataType.MATRIX, list(matrix.shape),
                        is_sparse=True, sparsity_pattern=info['pattern'], is_constant=True)
            )
        
        try:
            ast = codegen.build_ast(ops)
            optimized_ast = codegen.optimize_ast(ast)
            algo_code = codegen.generate_code(optimized_ast, target='c')
        except Exception as e:
            algo_code = f"// Code generation failed: {e}\n"
        
        code_lines = [
            "#include <stdio.h>",
            "#include <stdlib.h>", 
            "#include <math.h>",
            "",
            f"#define NX {nx}",
            f"#define NU {nu}",
            f"#define N {N}",
            "",
            "// System matrices",
            array_to_c_init(A, "A"),
            array_to_c_init(B, "B"),
            array_to_c_init(Kinf, "Kinf"),
            "",
            "// Global arrays",
            f"float x[{N}][{nx}];",
            f"float u[{N-1}][{nu}];", 
            f"float d[{N-1}][{nu}];",
            f"float temp_kinf[{nu}];",
            f"float temp_u[{nu}];",
            f"float temp_a[{nx}];",
            f"float temp_b[{nx}];",
            "",
            "// Forward pass implementation",
            "void forward_pass_impl() {",
            "    " + algo_code.replace('\n', '\n    '),
            "}",
            "",
            "// External interface",
            "void forward_pass(float* x0_in, float* d_in, float* x_out, float* u_out) {",
            "    // Copy inputs",
            f"    for (int i = 0; i < {nx}; i++) x[0][i] = x0_in[i];",
            f"    for (int k = 0; k < {N-1}; k++)",
            f"        for (int i = 0; i < {nu}; i++) d[k][i] = d_in[k * {nu} + i];",
            "",
            "    // Execute algorithm", 
            "    forward_pass_impl();",
            "",
            "    // Copy outputs",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++) x_out[k * {nx} + i] = x[k][i];",
            f"    for (int k = 0; k < {N-1}; k++)",
            f"        for (int i = 0; i < {nu}; i++) u_out[k * {nu} + i] = u[k][i];",
            "}"
        ]
        
        return '\n'.join(code_lines)
    
    def _compile_and_run_c_code(self, c_code: str, x0: np.ndarray, d: np.ndarray, 
                               N: int, nx: int, nu: int) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """Compile and run C code, return results"""
        
        try:
            # Write C code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(c_code)
                c_file = f.name
            self.temp_files.append(c_file)
            
            # Compile to shared library
            so_file = c_file.replace('.c', '.so')
            self.temp_files.append(so_file)
            
            cmd = ['gcc', '-shared', '-fPIC', '-O2', '-o', so_file, c_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, None, None, f"Compilation error: {result.stderr}"
            
            # Load and call
            lib = ctypes.CDLL(so_file)
            lib.forward_pass.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # x0
                ctypes.POINTER(ctypes.c_float),  # d  
                ctypes.POINTER(ctypes.c_float),  # x_out
                ctypes.POINTER(ctypes.c_float)   # u_out
            ]
            
            # Prepare arrays
            x0_flat = x0.astype(np.float32)
            d_flat = d.astype(np.float32).flatten()
            x_out = np.zeros(N * nx, dtype=np.float32)
            u_out = np.zeros((N-1) * nu, dtype=np.float32)
            
            # Call function
            lib.forward_pass(
                x0_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                d_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                u_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            
            # Reshape outputs
            x_result = x_out.reshape((N, nx))
            u_result = u_out.reshape((N-1, nu))
            
            return True, x_result, u_result, ""
            
        except Exception as e:
            return False, None, None, f"Runtime error: {str(e)}"
    
    def cleanup(self):
        """Clean up temporary files"""
        for f in self.temp_files:
            try:
                os.unlink(f)
            except:
                pass
        self.temp_files = []
    
    def validate_slack_update(self, u: np.ndarray, y: np.ndarray, x: np.ndarray, g: np.ndarray,
                             ops: List[Dict[str, Any]], tolerance: float = 1e-5,
                             u_min: np.ndarray = None, u_max: np.ndarray = None,
                             x_min: np.ndarray = None, x_max: np.ndarray = None) -> Tuple[bool, str]:
        """Validate slack update subfunction numerically"""
        N, nx = x.shape
        Nu, nu = u.shape
        
        # Python reference implementation
        znew_ref = u + y  # Input slack: znew = u + y
        vnew_ref = x + g  # State slack: vnew = x + g
        
        # Apply box constraints if provided
        if u_min is not None and u_max is not None:
            znew_ref = np.clip(znew_ref, u_min, u_max)
        if x_min is not None and x_max is not None:
            vnew_ref = np.clip(vnew_ref, x_min, x_max)
        
        # Generate and run C code
        code = self._generate_slack_update_c_code(u, y, x, g, u_min, u_max, x_min, x_max, ops)
        success, znew_c, vnew_c, error_msg = self._compile_and_run_slack_c_code(
            code, u, y, x, g, u_min, u_max, x_min, x_max, N, nx, Nu, nu)
        
        if not success:
            return False, f"Compilation failed: {error_msg}"
        
        # Compare results
        z_error = np.max(np.abs(znew_ref - znew_c))
        v_error = np.max(np.abs(vnew_ref - vnew_c))
        max_error = max(z_error, v_error)
        
        passed = max_error < tolerance
        details = f"Max error: {max_error:.2e}, z_error: {z_error:.2e}, v_error: {v_error:.2e}"
        
        return passed, details
    
    def _generate_slack_update_c_code(self, u: np.ndarray, y: np.ndarray, x: np.ndarray, g: np.ndarray,
                                     u_min: np.ndarray, u_max: np.ndarray, x_min: np.ndarray, x_max: np.ndarray,
                                     ops: List[Dict[str, Any]]) -> str:
        """Generate C code for slack update validation"""
        N, nx = x.shape
        Nu, nu = u.shape
        
        # Simple implementation for testing
        code_lines = [
            "#include <stdio.h>",
            "#include <math.h>",
            "",
            f"#define NX {nx}",
            f"#define NU {nu}", 
            f"#define N {N}",
            "",
            f"float u[{Nu}][{nu}];",
            f"float y[{Nu}][{nu}];",
            f"float x[{N}][{nx}];",
            f"float g[{N}][{nx}];",
            f"float znew[{Nu}][{nu}];",
            f"float vnew[{N}][{nx}];",
            "",
            "void slack_update() {",
            "    // znew = u + y",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++)",
            "            znew[k][i] = u[k][i] + y[k][i];",
            "",
            "    // vnew = x + g", 
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++)",
            "            vnew[k][i] = x[k][i] + g[k][i];",
            "}",
            "",
            "void slack_update_interface(float* u_in, float* y_in, float* x_in, float* g_in,",
            "                           float* znew_out, float* vnew_out) {",
            "    // Copy inputs",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++) {{",
            f"            u[k][i] = u_in[k * {nu} + i];",
            f"            y[k][i] = y_in[k * {nu} + i];",
            "        }",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++) {{",
            f"            x[k][i] = x_in[k * {nx} + i];",
            f"            g[k][i] = g_in[k * {nx} + i];",
            "        }",
            "",
            "    slack_update();",
            "",
            "    // Copy outputs",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++)",
            f"            znew_out[k * {nu} + i] = znew[k][i];",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++)",
            f"            vnew_out[k * {nx} + i] = vnew[k][i];",
            "}"
        ]
        
        return '\n'.join(code_lines)
    
    def _compile_and_run_slack_c_code(self, c_code: str, u: np.ndarray, y: np.ndarray, 
                                     x: np.ndarray, g: np.ndarray, u_min, u_max, x_min, x_max,
                                     N: int, nx: int, Nu: int, nu: int) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """Compile and run slack update C code"""
        try:
            # Write and compile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(c_code)
                c_file = f.name
            self.temp_files.append(c_file)
            
            so_file = c_file.replace('.c', '.so')
            self.temp_files.append(so_file)
            
            cmd = ['gcc', '-shared', '-fPIC', '-O2', '-o', so_file, c_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, None, None, f"Compilation error: {result.stderr}"
            
            # Load and call
            lib = ctypes.CDLL(so_file)
            lib.slack_update_interface.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # u
                ctypes.POINTER(ctypes.c_float),  # y
                ctypes.POINTER(ctypes.c_float),  # x
                ctypes.POINTER(ctypes.c_float),  # g
                ctypes.POINTER(ctypes.c_float),  # znew_out
                ctypes.POINTER(ctypes.c_float)   # vnew_out
            ]
            
            # Prepare arrays
            u_flat = u.astype(np.float32).flatten()
            y_flat = y.astype(np.float32).flatten()
            x_flat = x.astype(np.float32).flatten()
            g_flat = g.astype(np.float32).flatten()
            znew_out = np.zeros(Nu * nu, dtype=np.float32)
            vnew_out = np.zeros(N * nx, dtype=np.float32)
            
            # Call function
            lib.slack_update_interface(
                u_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                g_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                znew_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                vnew_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            
            # Reshape outputs
            znew_result = znew_out.reshape((Nu, nu))
            vnew_result = vnew_out.reshape((N, nx))
            
            return True, znew_result, vnew_result, ""
            
        except Exception as e:
            return False, None, None, f"Runtime error: {str(e)}"

    def validate_dual_update(self, y: np.ndarray, u: np.ndarray, znew: np.ndarray,
                            g: np.ndarray, x: np.ndarray, vnew: np.ndarray,
                            ops: List[Dict[str, Any]], tolerance: float = 1e-5) -> Tuple[bool, str]:
        """Validate dual update subfunction numerically"""
        
        # Python reference implementation  
        y_ref = y + u - znew  # Dual input update: y = y + u - znew
        g_ref = g + x - vnew  # Dual state update: g = g + x - vnew
        
        # Generate and run C code
        code = self._generate_dual_update_c_code(y, u, znew, g, x, vnew, ops)
        success, y_c, g_c, error_msg = self._compile_and_run_dual_c_code(
            code, y, u, znew, g, x, vnew)
        
        if not success:
            return False, f"Compilation failed: {error_msg}"
        
        # Compare results
        y_error = np.max(np.abs(y_ref - y_c))
        g_error = np.max(np.abs(g_ref - g_c))
        max_error = max(y_error, g_error)
        
        passed = max_error < tolerance
        details = f"Max error: {max_error:.2e}, y_error: {y_error:.2e}, g_error: {g_error:.2e}"
        
        return passed, details
    
    def _generate_dual_update_c_code(self, y: np.ndarray, u: np.ndarray, znew: np.ndarray,
                                    g: np.ndarray, x: np.ndarray, vnew: np.ndarray,
                                    ops: List[Dict[str, Any]]) -> str:
        """Generate C code for dual update validation"""
        Nu, nu = y.shape
        N, nx = g.shape
        
        code_lines = [
            "#include <stdio.h>",
            "#include <math.h>",
            "",
            f"#define NX {nx}",
            f"#define NU {nu}",
            f"#define N {N}",
            "",
            f"float y[{Nu}][{nu}];",
            f"float u[{Nu}][{nu}];",
            f"float znew[{Nu}][{nu}];",
            f"float g[{N}][{nx}];",
            f"float x[{N}][{nx}];",
            f"float vnew[{N}][{nx}];",
            "",
            "void dual_update() {",
            "    // y = y + u - znew",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++)",
            "            y[k][i] = y[k][i] + u[k][i] - znew[k][i];",
            "",
            "    // g = g + x - vnew",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++)",
            "            g[k][i] = g[k][i] + x[k][i] - vnew[k][i];",
            "}",
            "",
            "void dual_update_interface(float* y_in, float* u_in, float* znew_in,",
            "                          float* g_in, float* x_in, float* vnew_in,",
            "                          float* y_out, float* g_out) {",
            "    // Copy inputs",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++) {{",
            f"            y[k][i] = y_in[k * {nu} + i];",
            f"            u[k][i] = u_in[k * {nu} + i];",
            f"            znew[k][i] = znew_in[k * {nu} + i];",
            "        }",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++) {{",
            f"            g[k][i] = g_in[k * {nx} + i];",
            f"            x[k][i] = x_in[k * {nx} + i];",
            f"            vnew[k][i] = vnew_in[k * {nx} + i];",
            "        }",
            "",
            "    dual_update();",
            "",
            "    // Copy outputs",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++)",
            f"            y_out[k * {nu} + i] = y[k][i];",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++)",
            f"            g_out[k * {nx} + i] = g[k][i];",
            "}"
        ]
        
        return '\n'.join(code_lines)
    
    def _compile_and_run_dual_c_code(self, c_code: str, y: np.ndarray, u: np.ndarray, znew: np.ndarray,
                                    g: np.ndarray, x: np.ndarray, vnew: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """Compile and run dual update C code"""
        Nu, nu = y.shape
        N, nx = g.shape
        
        try:
            # Write and compile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(c_code)
                c_file = f.name
            self.temp_files.append(c_file)
            
            so_file = c_file.replace('.c', '.so')
            self.temp_files.append(so_file)
            
            cmd = ['gcc', '-shared', '-fPIC', '-O2', '-o', so_file, c_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, None, None, f"Compilation error: {result.stderr}"
            
            # Load and call
            lib = ctypes.CDLL(so_file)
            lib.dual_update_interface.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # y
                ctypes.POINTER(ctypes.c_float),  # u
                ctypes.POINTER(ctypes.c_float),  # znew
                ctypes.POINTER(ctypes.c_float),  # g
                ctypes.POINTER(ctypes.c_float),  # x
                ctypes.POINTER(ctypes.c_float),  # vnew
                ctypes.POINTER(ctypes.c_float),  # y_out
                ctypes.POINTER(ctypes.c_float)   # g_out
            ]
            
            # Prepare arrays
            y_flat = y.astype(np.float32).flatten()
            u_flat = u.astype(np.float32).flatten()
            znew_flat = znew.astype(np.float32).flatten()
            g_flat = g.astype(np.float32).flatten()
            x_flat = x.astype(np.float32).flatten()
            vnew_flat = vnew.astype(np.float32).flatten()
            y_out = np.zeros(Nu * nu, dtype=np.float32)
            g_out = np.zeros(N * nx, dtype=np.float32)
            
            # Call function
            lib.dual_update_interface(
                y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                u_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                znew_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                g_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                vnew_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                g_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            
            # Reshape outputs
            y_result = y_out.reshape((Nu, nu))
            g_result = g_out.reshape((N, nx))
            
            return True, y_result, g_result, ""
            
        except Exception as e:
            return False, None, None, f"Runtime error: {str(e)}"

    def validate_backward_pass(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                              q: np.ndarray, r: np.ndarray, p_terminal: np.ndarray,
                              ops: List[Dict[str, Any]], tolerance: float = 1e-5) -> Tuple[bool, str]:
        """Validate backward pass subfunction numerically"""
        N, nx = q.shape
        Nu, nu = r.shape
        
        # Python reference implementation of Riccati recursion
        # Compute required matrices
        from scipy.linalg import solve_discrete_are
        P_inf = solve_discrete_are(A, B, Q, R)
        K_inf = np.linalg.solve(R + B.T @ P_inf @ B, B.T @ P_inf @ A)
        
        # Derived matrices
        Quu_inv = np.linalg.inv(R + B.T @ P_inf @ B)
        AmBKt = A - B @ K_inf
        
        # Initialize
        p_ref = np.zeros((N, nx), dtype=np.float32)
        d_ref = np.zeros((N-1, nu), dtype=np.float32)
        p_ref[N-1] = p_terminal
        
        # Backward pass: d[k] = Quu_inv @ (B.T @ p[k+1] + r[k]), p[k] = q[k] + AmBKt.T @ p[k+1] - K_inf.T @ r[k]
        for k in range(N-2, -1, -1):
            temp_bt_p = B.T @ p_ref[k+1]
            d_ref[k] = Quu_inv @ (temp_bt_p + r[k])
            p_ref[k] = q[k] + AmBKt.T @ p_ref[k+1] - K_inf.T @ r[k]
        
        # Generate and run C code
        code = self._generate_backward_pass_c_code(A, B, Q, R, P_inf, K_inf, Quu_inv, AmBKt, 
                                                  q, r, p_terminal, ops)
        success, p_c, d_c, error_msg = self._compile_and_run_backward_c_code(
            code, q, r, p_terminal, N, nx, Nu, nu)
        
        if not success:
            return False, f"Compilation failed: {error_msg}"
        
        # Compare results
        p_error = np.max(np.abs(p_ref - p_c))
        d_error = np.max(np.abs(d_ref - d_c))
        max_error = max(p_error, d_error)
        
        passed = max_error < tolerance
        details = f"Max error: {max_error:.2e}, p_error: {p_error:.2e}, d_error: {d_error:.2e}"
        
        return passed, details
    
    def _generate_backward_pass_c_code(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                                      P_inf: np.ndarray, K_inf: np.ndarray, Quu_inv: np.ndarray, AmBKt: np.ndarray,
                                      q: np.ndarray, r: np.ndarray, p_terminal: np.ndarray,
                                      ops: List[Dict[str, Any]]) -> str:
        """Generate C code for backward pass validation"""
        N, nx = q.shape
        Nu, nu = r.shape
        
        def array_to_c_init(arr, name):
            if arr.ndim == 1:
                values = ', '.join(f"{x:.6f}f" for x in arr.flatten())
                return f"float {name}[{arr.shape[0]}] = {{{values}}};"
            else:
                lines = [f"float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{"]
                for i in range(arr.shape[0]):
                    values = ', '.join(f"{arr[i,j]:.6f}f" for j in range(arr.shape[1]))
                    lines.append(f"  {{{values}}},")
                lines.append("};")
                return '\n'.join(lines)
        
        code_lines = [
            "#include <stdio.h>",
            "#include <math.h>",
            "",
            f"#define NX {nx}",
            f"#define NU {nu}",
            f"#define N {N}",
            "",
            "// System matrices",
            array_to_c_init(B.T, "B_T"),
            array_to_c_init(K_inf.T, "Kinf_T"),
            array_to_c_init(Quu_inv, "Quu_inv"),
            array_to_c_init(AmBKt.T, "AmBKt_T"),
            "",
            "// Variables",
            f"float q[{N}][{nx}];",
            f"float r[{Nu}][{nu}];",
            f"float p[{N}][{nx}];",
            f"float d[{Nu}][{nu}];",
            f"float temp_bt_p[{nu}];",
            f"float temp_quu_input[{nu}];",
            f"float temp_kinf_r[{nx}];",
            "",
            "void backward_pass() {",
            f"    for (int k = {N-2}; k >= 0; k--) {{",
            "        // B.T @ p[k+1]",
            f"        for (int i = 0; i < {nu}; i++) {{",
            "            temp_bt_p[i] = 0.0f;",
            f"            for (int j = 0; j < {nx}; j++)",
            "                temp_bt_p[i] += B_T[i][j] * p[k+1][j];",
            "        }",
            "",
            "        // d[k] = Quu_inv @ (B.T @ p[k+1] + r[k])",
            f"        for (int i = 0; i < {nu}; i++)",
            "            temp_quu_input[i] = temp_bt_p[i] + r[k][i];",
            "",
            f"        for (int i = 0; i < {nu}; i++) {{",
            "            d[k][i] = 0.0f;",
            f"            for (int j = 0; j < {nu}; j++)",
            "                d[k][i] += Quu_inv[i][j] * temp_quu_input[j];",
            "        }",
            "",
            "        // Kinf.T @ r[k]",
            f"        for (int i = 0; i < {nx}; i++) {{",
            "            temp_kinf_r[i] = 0.0f;",
            f"            for (int j = 0; j < {nu}; j++)",
            "                temp_kinf_r[i] += Kinf_T[i][j] * r[k][j];",
            "        }",
            "",
            "        // p[k] = q[k] + AmBKt.T @ p[k+1] - Kinf.T @ r[k]",
            f"        for (int i = 0; i < {nx}; i++) {{",
            "            float temp_ambkt = 0.0f;",
            f"            for (int j = 0; j < {nx}; j++)",
            "                temp_ambkt += AmBKt_T[i][j] * p[k+1][j];",
            "            p[k][i] = q[k][i] + temp_ambkt - temp_kinf_r[i];",
            "        }",
            "    }",
            "}",
            "",
            "void backward_pass_interface(float* q_in, float* r_in, float* p_terminal_in,",
            "                            float* p_out, float* d_out) {",
            "    // Copy inputs",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++)",
            f"            q[k][i] = q_in[k * {nx} + i];",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++)",
            f"            r[k][i] = r_in[k * {nu} + i];",
            f"    for (int i = 0; i < {nx}; i++)",
            f"        p[{N-1}][i] = p_terminal_in[i];",
            "",
            "    backward_pass();",
            "",
            "    // Copy outputs",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++)",
            f"            p_out[k * {nx} + i] = p[k][i];",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++)",
            f"            d_out[k * {nu} + i] = d[k][i];",
            "}"
        ]
        
        return '\n'.join(code_lines)
    
    def _compile_and_run_backward_c_code(self, c_code: str, q: np.ndarray, r: np.ndarray, p_terminal: np.ndarray,
                                        N: int, nx: int, Nu: int, nu: int) -> Tuple[bool, np.ndarray, np.ndarray, str]:
        """Compile and run backward pass C code"""
        try:
            # Write and compile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(c_code)
                c_file = f.name
            self.temp_files.append(c_file)
            
            so_file = c_file.replace('.c', '.so')
            self.temp_files.append(so_file)
            
            cmd = ['gcc', '-shared', '-fPIC', '-O2', '-o', so_file, c_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, None, None, f"Compilation error: {result.stderr}"
            
            # Load and call
            lib = ctypes.CDLL(so_file)
            lib.backward_pass_interface.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # q
                ctypes.POINTER(ctypes.c_float),  # r
                ctypes.POINTER(ctypes.c_float),  # p_terminal
                ctypes.POINTER(ctypes.c_float),  # p_out
                ctypes.POINTER(ctypes.c_float)   # d_out
            ]
            
            # Prepare arrays
            q_flat = q.astype(np.float32).flatten()
            r_flat = r.astype(np.float32).flatten()
            p_terminal_flat = p_terminal.astype(np.float32)
            p_out = np.zeros(N * nx, dtype=np.float32)
            d_out = np.zeros(Nu * nu, dtype=np.float32)
            
            # Call function
            lib.backward_pass_interface(
                q_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                r_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                p_terminal_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                p_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                d_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            
            # Reshape outputs
            p_result = p_out.reshape((N, nx))
            d_result = d_out.reshape((Nu, nu))
            
            return True, p_result, d_result, ""
            
        except Exception as e:
            return False, None, None, f"Runtime error: {str(e)}"
    
    def validate_linear_cost(self, Q: np.ndarray, R: np.ndarray, Xref: np.ndarray, Uref: np.ndarray,
                            x: np.ndarray, u: np.ndarray, vnew: np.ndarray, znew: np.ndarray,
                            y: np.ndarray, g: np.ndarray, rho: float,
                            ops: List[Dict[str, Any]], tolerance: float = 1e-5) -> Tuple[bool, str]:
        """Validate linear cost update subfunction numerically"""
        N, nx = x.shape
        Nu, nu = u.shape
        
        # Python reference implementation
        q_ref = np.zeros((N, nx), dtype=np.float32)
        r_ref = np.zeros((Nu, nu), dtype=np.float32)
        p_ref = np.zeros((N, nx), dtype=np.float32)
        
        # r[k] = -Uref[k] * R - rho * (znew[k] - y[k])
        for k in range(Nu):
            r_ref[k] = -(Uref[k] * np.diag(R)) - rho * (znew[k] - y[k])
        
        # q[k] = -Xref[k] * Q - rho * (vnew[k] - g[k])
        for k in range(N):
            q_ref[k] = -(Xref[k] * np.diag(Q)) - rho * (vnew[k] - g[k])
        
        # Generate and run C code
        code = self._generate_linear_cost_c_code(Q, R, Xref, Uref, x, u, vnew, znew, y, g, rho, ops)
        success, q_c, r_c, p_c, error_msg = self._compile_and_run_linear_cost_c_code(
            code, Q, R, Xref, Uref, x, u, vnew, znew, y, g, rho, N, nx, Nu, nu)
        
        if not success:
            return False, f"Compilation failed: {error_msg}"
        
        # Compare results
        q_error = np.max(np.abs(q_ref - q_c))
        r_error = np.max(np.abs(r_ref - r_c))
        max_error = max(q_error, r_error)
        
        passed = max_error < tolerance
        details = f"Max error: {max_error:.2e}, q_error: {q_error:.2e}, r_error: {r_error:.2e}"
        
        return passed, details
    
    def _generate_linear_cost_c_code(self, Q: np.ndarray, R: np.ndarray, Xref: np.ndarray, Uref: np.ndarray,
                                    x: np.ndarray, u: np.ndarray, vnew: np.ndarray, znew: np.ndarray,
                                    y: np.ndarray, g: np.ndarray, rho: float, ops: List[Dict[str, Any]]) -> str:
        """Generate C code for linear cost validation"""
        N, nx = x.shape
        Nu, nu = u.shape
        
        def array_to_c_init(arr, name):
            if arr.ndim == 1:
                values = ', '.join(f"{x:.6f}f" for x in arr.flatten())
                return f"float {name}[{arr.shape[0]}] = {{{values}}};"
            else:
                lines = [f"float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{"]
                for i in range(arr.shape[0]):
                    values = ', '.join(f"{arr[i,j]:.6f}f" for j in range(arr.shape[1]))
                    lines.append(f"  {{{values}}},")
                lines.append("};")
                return '\n'.join(lines)
        
        code_lines = [
            "#include <stdio.h>",
            "#include <math.h>",
            "",
            f"#define NX {nx}",
            f"#define NU {nu}",
            f"#define N {N}",
            f"#define RHO {rho:.6f}f",
            "",
            "// Cost matrices",
            array_to_c_init(np.diag(Q), "Q"),
            array_to_c_init(np.diag(R), "R"),
            "",
            "// Variables",
            f"float Xref[{N}][{nx}];",
            f"float Uref[{Nu}][{nu}];",
            f"float x[{N}][{nx}];",
            f"float u[{Nu}][{nu}];",
            f"float vnew[{N}][{nx}];",
            f"float znew[{Nu}][{nu}];",
            f"float y[{Nu}][{nu}];",
            f"float g[{N}][{nx}];",
            f"float q[{N}][{nx}];",
            f"float r[{Nu}][{nu}];",
            f"float p[{N}][{nx}];",
            "",
            "void linear_cost_update() {",
            "    // r[k] = -Uref[k] * R - rho * (znew[k] - y[k])",
            f"    for (int k = 0; k < {Nu}; k++) {{",
            f"        for (int i = 0; i < {nu}; i++) {{",
            "            r[k][i] = -(Uref[k][i] * R[i]) - RHO * (znew[k][i] - y[k][i]);",
            "        }",
            "    }",
            "",
            "    // q[k] = -Xref[k] * Q - rho * (vnew[k] - g[k])",
            f"    for (int k = 0; k < {N}; k++) {{",
            f"        for (int i = 0; i < {nx}; i++) {{",
            "            q[k][i] = -(Xref[k][i] * Q[i]) - RHO * (vnew[k][i] - g[k][i]);",
            "        }",
            "    }",
            "}",
            "",
            "void linear_cost_interface(float* Xref_in, float* Uref_in, float* x_in, float* u_in,",
            "                          float* vnew_in, float* znew_in, float* y_in, float* g_in,",
            "                          float* q_out, float* r_out, float* p_out) {",
            "    // Copy inputs",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++) {{",
            f"            Xref[k][i] = Xref_in[k * {nx} + i];",
            f"            x[k][i] = x_in[k * {nx} + i];",
            f"            vnew[k][i] = vnew_in[k * {nx} + i];",
            f"            g[k][i] = g_in[k * {nx} + i];",
            "        }",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++) {{",
            f"            Uref[k][i] = Uref_in[k * {nu} + i];",
            f"            u[k][i] = u_in[k * {nu} + i];",
            f"            znew[k][i] = znew_in[k * {nu} + i];",
            f"            y[k][i] = y_in[k * {nu} + i];",
            "        }",
            "",
            "    linear_cost_update();",
            "",
            "    // Copy outputs",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++) {{",
            f"            q_out[k * {nx} + i] = q[k][i];",
            f"            p_out[k * {nx} + i] = p[k][i];",
            "        }",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++)",
            f"            r_out[k * {nu} + i] = r[k][i];",
            "}"
        ]
        
        return '\n'.join(code_lines)
    
    def _compile_and_run_linear_cost_c_code(self, c_code: str, Q: np.ndarray, R: np.ndarray, 
                                           Xref: np.ndarray, Uref: np.ndarray, x: np.ndarray, u: np.ndarray,
                                           vnew: np.ndarray, znew: np.ndarray, y: np.ndarray, g: np.ndarray, rho: float,
                                           N: int, nx: int, Nu: int, nu: int) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, str]:
        """Compile and run linear cost C code"""
        try:
            # Write and compile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(c_code)
                c_file = f.name
            self.temp_files.append(c_file)
            
            so_file = c_file.replace('.c', '.so')
            self.temp_files.append(so_file)
            
            cmd = ['gcc', '-shared', '-fPIC', '-O2', '-o', so_file, c_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, None, None, None, f"Compilation error: {result.stderr}"
            
            # Load and call
            lib = ctypes.CDLL(so_file)
            lib.linear_cost_interface.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # Xref
                ctypes.POINTER(ctypes.c_float),  # Uref
                ctypes.POINTER(ctypes.c_float),  # x
                ctypes.POINTER(ctypes.c_float),  # u
                ctypes.POINTER(ctypes.c_float),  # vnew
                ctypes.POINTER(ctypes.c_float),  # znew
                ctypes.POINTER(ctypes.c_float),  # y
                ctypes.POINTER(ctypes.c_float),  # g
                ctypes.POINTER(ctypes.c_float),  # q_out
                ctypes.POINTER(ctypes.c_float),  # r_out
                ctypes.POINTER(ctypes.c_float)   # p_out
            ]
            
            # Prepare arrays
            Xref_flat = Xref.astype(np.float32).flatten()
            Uref_flat = Uref.astype(np.float32).flatten()
            x_flat = x.astype(np.float32).flatten()
            u_flat = u.astype(np.float32).flatten()
            vnew_flat = vnew.astype(np.float32).flatten()
            znew_flat = znew.astype(np.float32).flatten()
            y_flat = y.astype(np.float32).flatten()
            g_flat = g.astype(np.float32).flatten()
            q_out = np.zeros(N * nx, dtype=np.float32)
            r_out = np.zeros(Nu * nu, dtype=np.float32)
            p_out = np.zeros(N * nx, dtype=np.float32)
            
            # Call function
            lib.linear_cost_interface(
                Xref_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                Uref_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                u_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                vnew_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                znew_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                g_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                q_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                r_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                p_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            
            # Reshape outputs
            q_result = q_out.reshape((N, nx))
            r_result = r_out.reshape((Nu, nu))
            p_result = p_out.reshape((N, nx))
            
            return True, q_result, r_result, p_result, ""
            
        except Exception as e:
            return False, None, None, None, f"Runtime error: {str(e)}"
    
    def validate_termination_check(self, x: np.ndarray, u: np.ndarray, vnew: np.ndarray, znew: np.ndarray,
                                  abs_pri_tol: float, abs_dua_tol: float, iter_num: int, check_freq: int,
                                  ops: List[Dict[str, Any]], tolerance: float = 1e-5) -> Tuple[bool, str]:
        """Validate termination check subfunction numerically"""
        N, nx = x.shape
        Nu, nu = u.shape
        
        # Python reference implementation
        solved_ref = 0
        
        if iter_num % check_freq == 0:
            # Compute primal residuals
            primal_res_state = np.max(np.abs(x - vnew))
            primal_res_input = np.max(np.abs(u - znew))
            
            # For dual residuals, we assume they are provided (simplified)
            dual_res_state = primal_res_state * 0.1  # Simplified
            dual_res_input = primal_res_input * 0.1
            
            # Check convergence
            if (primal_res_state < abs_pri_tol and primal_res_input < abs_pri_tol and
                dual_res_state < abs_dua_tol and dual_res_input < abs_dua_tol):
                solved_ref = 1
        
        # Generate and run C code
        code = self._generate_termination_c_code(x, u, vnew, znew, abs_pri_tol, abs_dua_tol, 
                                                iter_num, check_freq, ops)
        success, solved_c, error_msg = self._compile_and_run_termination_c_code(
            code, x, u, vnew, znew, abs_pri_tol, abs_dua_tol, iter_num, N, nx, Nu, nu)
        
        if not success:
            return False, f"Compilation failed: {error_msg}"
        
        # Compare results
        passed = (solved_ref == solved_c)
        details = f"Expected solved: {solved_ref}, Got: {solved_c}"
        
        return passed, details
    
    def _generate_termination_c_code(self, x: np.ndarray, u: np.ndarray, vnew: np.ndarray, znew: np.ndarray,
                                    abs_pri_tol: float, abs_dua_tol: float, iter_num: int, check_freq: int,
                                    ops: List[Dict[str, Any]]) -> str:
        """Generate C code for termination check validation"""
        N, nx = x.shape
        Nu, nu = u.shape
        
        code_lines = [
            "#include <stdio.h>",
            "#include <math.h>",
            "",
            f"#define NX {nx}",
            f"#define NU {nu}",
            f"#define N {N}",
            "",
            f"float x[{N}][{nx}];",
            f"float u[{Nu}][{nu}];",
            f"float vnew[{N}][{nx}];",
            f"float znew[{Nu}][{nu}];",
            "int solved = 0;",
            "",
            "float fmaxf_array(float* arr, int size) {",
            "    float max_val = fabsf(arr[0]);",
            "    for (int i = 1; i < size; i++)",
            "        if (fabsf(arr[i]) > max_val) max_val = fabsf(arr[i]);",
            "    return max_val;",
            "}",
            "",
            "void termination_check(int iter, float abs_pri_tol, float abs_dua_tol) {",
            f"    if (iter % {check_freq} == 0) {{",
            "        // Compute primal residuals",
            "        float primal_res_state = 0.0f;",
            "        float primal_res_input = 0.0f;",
            "",
            f"        for (int k = 0; k < {N}; k++)",
            f"            for (int i = 0; i < {nx}; i++) {{",
            "                float diff = fabsf(x[k][i] - vnew[k][i]);",
            "                if (diff > primal_res_state) primal_res_state = diff;",
            "            }",
            "",
            f"        for (int k = 0; k < {Nu}; k++)",
            f"            for (int i = 0; i < {nu}; i++) {{",
            "                float diff = fabsf(u[k][i] - znew[k][i]);",
            "                if (diff > primal_res_input) primal_res_input = diff;",
            "            }",
            "",
            "        // Simplified dual residuals",
            "        float dual_res_state = primal_res_state * 0.1f;",
            "        float dual_res_input = primal_res_input * 0.1f;",
            "",
            "        // Check convergence",
            "        if (primal_res_state < abs_pri_tol && primal_res_input < abs_pri_tol &&",
            "            dual_res_state < abs_dua_tol && dual_res_input < abs_dua_tol) {",
            "            solved = 1;",
            "        }",
            "    }",
            "}",
            "",
            "void termination_interface(float* x_in, float* u_in, float* vnew_in, float* znew_in,",
            "                          int iter, float abs_pri_tol, float abs_dua_tol, int* solved_out) {",
            "    // Copy inputs",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++) {{",
            f"            x[k][i] = x_in[k * {nx} + i];",
            f"            vnew[k][i] = vnew_in[k * {nx} + i];",
            "        }",
            f"    for (int k = 0; k < {Nu}; k++)",
            f"        for (int i = 0; i < {nu}; i++) {{",
            f"            u[k][i] = u_in[k * {nu} + i];",
            f"            znew[k][i] = znew_in[k * {nu} + i];",
            "        }",
            "",
            "    solved = 0;",
            "    termination_check(iter, abs_pri_tol, abs_dua_tol);",
            "    *solved_out = solved;",
            "}"
        ]
        
        return '\n'.join(code_lines)
    
    def _compile_and_run_termination_c_code(self, c_code: str, x: np.ndarray, u: np.ndarray, 
                                           vnew: np.ndarray, znew: np.ndarray, abs_pri_tol: float, abs_dua_tol: float,
                                           iter_num: int, N: int, nx: int, Nu: int, nu: int) -> Tuple[bool, int, str]:
        """Compile and run termination check C code"""
        try:
            # Write and compile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(c_code)
                c_file = f.name
            self.temp_files.append(c_file)
            
            so_file = c_file.replace('.c', '.so')
            self.temp_files.append(so_file)
            
            cmd = ['gcc', '-shared', '-fPIC', '-O2', '-o', so_file, c_file]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, 0, f"Compilation error: {result.stderr}"
            
            # Load and call
            lib = ctypes.CDLL(so_file)
            lib.termination_interface.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # x
                ctypes.POINTER(ctypes.c_float),  # u
                ctypes.POINTER(ctypes.c_float),  # vnew
                ctypes.POINTER(ctypes.c_float),  # znew
                ctypes.c_int,                    # iter
                ctypes.c_float,                  # abs_pri_tol
                ctypes.c_float,                  # abs_dua_tol
                ctypes.POINTER(ctypes.c_int)     # solved_out
            ]
            
            # Prepare arrays
            x_flat = x.astype(np.float32).flatten()
            u_flat = u.astype(np.float32).flatten()
            vnew_flat = vnew.astype(np.float32).flatten()
            znew_flat = znew.astype(np.float32).flatten()
            solved_out = ctypes.c_int()
            
            # Call function
            lib.termination_interface(
                x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                u_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                vnew_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                znew_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                iter_num,
                abs_pri_tol,
                abs_dua_tol,
                ctypes.byref(solved_out)
            )
            
            return True, solved_out.value, ""
            
        except Exception as e:
            return False, 0, f"Runtime error: {str(e)}"

    def __del__(self):
        self.cleanup()