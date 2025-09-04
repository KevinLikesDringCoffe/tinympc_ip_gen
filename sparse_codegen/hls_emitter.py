"""
HLS C++ Code Emitter for TinyMPC
Generates optimized HLS C++ code with proper pragmas and memory layout
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .core import Variable, DataType, get_sparsity_info
from .ast_nodes import *


class HLSCodeEmitter:
    """Generate optimized HLS C++ code for TinyMPC algorithm steps"""
    
    def __init__(self, nx: int, nu: int, N: int, target_clock_mhz: int = 250):
        self.nx = nx
        self.nu = nu 
        self.N = N
        self.target_clock_mhz = target_clock_mhz
        self.clock_period_ns = 1000 / target_clock_mhz
        
        # Array configuration for separate BRAM allocation
        self.separate_arrays = True
        
        # Keep legacy workspace for backward compatibility (deprecated)
        self.workspace_width = max(nx, nu, 1)
        self.workspace_depth = self._calculate_workspace_depth()
        self.workspace_layout = self._calculate_workspace_layout()
        
    def _calculate_workspace_depth(self) -> int:
        """Calculate required workspace depth based on problem dimensions"""
        # State variables: x(N), v(N), vnew(N), g(N)  
        state_vars = 4 * self.N * self.nx
        
        # Input variables: u(N-1), z(N-1), znew(N-1), y(N-1), d(N-1)
        input_vars = 5 * (self.N - 1) * self.nu
        
        # Cost variables: q(N), r(N-1), p(N)
        cost_vars = (2 * self.N + self.N - 1) * max(self.nx, self.nu)
        
        # References: Xref(N), Uref(N-1)
        ref_vars = (self.N * self.nx + (self.N - 1) * self.nu)
        
        total_elements = state_vars + input_vars + cost_vars + ref_vars
        return int(total_elements / self.workspace_width) + 100  # Add buffer
        
    def _calculate_workspace_layout(self) -> Dict[str, Dict[str, int]]:
        """Calculate memory layout for workspace variables"""
        layout = {}
        offset = 0
        
        # State variables
        for var in ['x', 'v', 'vnew', 'g']:
            layout[var] = {'offset': offset, 'size': self.N * self.nx}
            offset += self.N * self.nx // self.workspace_width + 1
            
        # Input variables  
        for var in ['u', 'z', 'znew', 'y', 'd']:
            layout[var] = {'offset': offset, 'size': (self.N - 1) * self.nu}
            offset += (self.N - 1) * self.nu // self.workspace_width + 1
            
        # Cost variables
        for var in ['q', 'p']:
            layout[var] = {'offset': offset, 'size': self.N * self.nx}
            offset += self.N * self.nx // self.workspace_width + 1
            
        layout['r'] = {'offset': offset, 'size': (self.N - 1) * self.nu}
        offset += (self.N - 1) * self.nu // self.workspace_width + 1
        
        # Reference variables
        layout['Xref'] = {'offset': offset, 'size': self.N * self.nx}
        offset += self.N * self.nx // self.workspace_width + 1
        
        layout['Uref'] = {'offset': offset, 'size': (self.N - 1) * self.nu}
        offset += (self.N - 1) * self.nu // self.workspace_width + 1
        
        return layout
        
    def generate_header(self, function_name: str) -> str:
        """Generate HLS function header with separate BRAM arrays"""
        header = f"""#ifndef TINYMPC_{function_name.upper()}_H
#define TINYMPC_{function_name.upper()}_H

#include <ap_fixed.h>
#include <hls_stream.h>

// Problem dimensions
#define NX {self.nx}
#define NU {self.nu}
#define N {self.N}

// Data types
typedef float data_t;
typedef ap_fixed<32,16> fixed_t;

// Separate BRAM arrays for each vector type
typedef struct {{
    // State trajectory and variables
    data_t x[N][NX];
    data_t v[N][NX];
    data_t vnew[N][NX];
    data_t g[N][NX];
    
    // Input trajectory and variables
    data_t u[N-1][NU];
    data_t z[N-1][NU];
    data_t znew[N-1][NU];
    data_t y[N-1][NU];
    data_t d[N-1][NU];
    
    // Cost variables
    data_t q[N][NX];
    data_t p[N][NX];
    data_t r[N-1][NU];
    
    // Reference trajectories
    data_t xref[N][NX];
    data_t uref[N-1][NU];
}} tinympc_workspace_t;

#endif
"""
        return header
        
    def generate_forward_pass(self, matrices: Dict[str, np.ndarray]) -> str:
        """Generate optimized forward pass function with separate arrays"""
        A = matrices['A']
        B = matrices['B']
        Kinf = matrices['Kinf']
        
        # Analyze sparsity
        A_sparsity = get_sparsity_info(A)
        B_sparsity = get_sparsity_info(B) 
        K_sparsity = get_sparsity_info(Kinf)
        
        code = """
void tinympc_forward_pass(tinympc_workspace_t& ws) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ws.x complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.u complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.d complete dim=2

    // Forward simulation with embedded system matrices
    forward_loop: for (int k = 0; k < N-1; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=100
        
        // u[k] = -Kinf @ x[k] - d[k]
"""
        
        # Generate sparse matrix multiplication for -Kinf @ x[k]
        for i in range(self.nu):
            nonzero_terms = []
            for j in range(self.nx):
                if abs(Kinf[i, j]) > 1e-10:
                    nonzero_terms.append(f"{float(-Kinf[i, j]):.6f}f * ws.x[k][{j}]")
            
            if nonzero_terms:
                code += f"        ws.u[k][{i}] = {' + '.join(nonzero_terms)} - ws.d[k][{i}];\n"
            else:
                code += f"        ws.u[k][{i}] = -ws.d[k][{i}];\n"
                
        code += "\n        // x[k+1] = A @ x[k] + B @ u[k]\n"
        
        # Generate sparse matrix multiplication for A @ x[k] + B @ u[k]
        for i in range(self.nx):
            terms = []
            
            # A @ x[k] terms
            for j in range(self.nx):
                if abs(A[i, j]) > 1e-10:
                    terms.append(f"{float(A[i, j]):.6f}f * ws.x[k][{j}]")
                    
            # B @ u[k] terms  
            for j in range(self.nu):
                if abs(B[i, j]) > 1e-10:
                    terms.append(f"{float(B[i, j]):.6f}f * ws.u[k][{j}]")
                    
            if terms:
                code += f"        ws.x[k+1][{i}] = {' + '.join(terms)};\n"
            else:
                code += f"        ws.x[k+1][{i}] = 0.0f;\n"
                
        code += "    }\n}\n"
        return code
        
    def generate_slack_update(self, bounds: Dict[str, np.ndarray]) -> str:
        """Generate slack variable update function with separate arrays"""
        code = """
void tinympc_update_slack(tinympc_workspace_t& ws) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ws.u complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.y complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.znew complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.x complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.g complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.vnew complete dim=2

    // Update input slack variables
    input_slack_loop: for (int k = 0; k < N-1; k++) {
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NU; i++) {
#pragma HLS UNROLL
            data_t temp = ws.u[k][i] + ws.y[k][i];
"""
        
        # Add input bounds if they exist
        if 'u_min' in bounds and 'u_max' in bounds:
            u_min = bounds['u_min']
            u_max = bounds['u_max']
            code += f"            temp = fmaxf({float(u_min[0]):.6f}f, fminf({float(u_max[0]):.6f}f, temp));\n"
        
        code += """            ws.znew[k][i] = temp;
        }
    }
    
    // Update state slack variables
    state_slack_loop: for (int k = 0; k < N; k++) {
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NX; i++) {
#pragma HLS UNROLL
            data_t temp = ws.x[k][i] + ws.g[k][i];
"""
        
        # Add state bounds if they exist
        if 'x_min' in bounds and 'x_max' in bounds:
            x_min = bounds['x_min']
            x_max = bounds['x_max']
            code += f"            temp = fmaxf({float(x_min[0]):.6f}f, fminf({float(x_max[0]):.6f}f, temp));\n"
            
        code += """            ws.vnew[k][i] = temp;
        }
    }
}
"""
        return code
        
    def generate_dual_update(self) -> str:
        """Generate dual variable update function with separate arrays"""
        return """
void tinympc_update_dual(tinympc_workspace_t& ws) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ws.y complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.u complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.znew complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.g complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.x complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.vnew complete dim=2

    // Update input dual variables
    input_dual_loop: for (int k = 0; k < N-1; k++) {
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NU; i++) {
#pragma HLS UNROLL
            ws.y[k][i] = ws.y[k][i] + ws.u[k][i] - ws.znew[k][i];
        }
    }
    
    // Update state dual variables
    state_dual_loop: for (int k = 0; k < N; k++) {
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NX; i++) {
#pragma HLS UNROLL
            ws.g[k][i] = ws.g[k][i] + ws.x[k][i] - ws.vnew[k][i];
        }
    }
}
"""
        
    def generate_linear_cost(self, matrices: Dict[str, np.ndarray]) -> str:
        """Generate linear cost update function with separate arrays"""
        Q_diag = np.diag(matrices['Q']) if matrices['Q'].ndim == 2 else matrices['Q']
        R_diag = np.diag(matrices['R']) if matrices['R'].ndim == 2 else matrices['R']
        Pinf = matrices['Pinf']
        rho = matrices.get('rho', 1.0)
        
        code = f"""
void tinympc_update_linear_cost(tinympc_workspace_t& ws) {{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ws.r complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.uref complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.znew complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.y complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.q complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.xref complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.vnew complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.g complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.p complete dim=2

    // Update input cost
    input_cost_loop: for (int k = 0; k < N-1; k++) {{
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NU; i++) {{
#pragma HLS UNROLL
            ws.r[k][i] = -ws.uref[k][i] * {float(R_diag[0]):.6f}f - {float(rho):.6f}f * (ws.znew[k][i] - ws.y[k][i]);
        }}
    }}
    
    // Update state cost
    state_cost_loop: for (int k = 0; k < N; k++) {{
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
            ws.q[k][i] = -ws.xref[k][i] * {float(Q_diag[0]):.6f}f - {float(rho):.6f}f * (ws.vnew[k][i] - ws.g[k][i]);
        }}
    }}
    
    // Terminal cost with Pinf matrix
"""
        
        # Generate Pinf multiplication for terminal cost
        for i in range(self.nx):
            terms = []
            for j in range(self.nx):
                if abs(Pinf[i, j]) > 1e-10:
                    terms.append(f"{float(-Pinf[i, j]):.6f}f * ws.xref[N-1][{j}]")
            
            if terms:
                code += f"    ws.p[N-1][{i}] = {' + '.join(terms)} - {float(rho):.6f}f * (ws.vnew[N-1][{i}] - ws.g[N-1][{i}]);\n"
            else:
                code += f"    ws.p[N-1][{i}] = -{float(rho):.6f}f * (ws.vnew[N-1][{i}] - ws.g[N-1][{i}]);\n"
                
        code += "}\n"
        return code
        
    def generate_backward_pass(self, matrices: Dict[str, np.ndarray]) -> str:
        """Generate backward Riccati recursion function with separate arrays"""
        Quu_inv = matrices['Quu_inv']
        AmBKt = matrices['AmBKt']
        B = matrices['B']
        Kinf = matrices['Kinf']
        
        code = """
void tinympc_backward_pass(tinympc_workspace_t& ws) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ws.d complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.p complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.r complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.q complete dim=2

    // Backward Riccati recursion
    backward_loop: for (int k = N-2; k >= 0; k--) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=100
        
        // d[k] = Quu_inv @ (B.T @ p[k+1] + r[k])
"""
        
        # Generate Quu_inv @ (B.T @ p[k+1] + r[k])
        for i in range(self.nu):
            # First compute B.T @ p[k+1]
            bt_p_terms = []
            for j in range(self.nx):
                if abs(B[j, i]) > 1e-10:  # B.T[i,j] = B[j,i]
                    bt_p_terms.append(f"{float(B[j, i]):.6f}f * ws.p[k+1][{j}]")
                    
            bt_p_expr = ' + '.join(bt_p_terms) if bt_p_terms else "0.0f"
            
            # Then multiply by Quu_inv
            quu_terms = []
            for j in range(self.nu):
                if abs(Quu_inv[i, j]) > 1e-10:
                    if j == i and bt_p_expr == "0.0f":
                        quu_terms.append(f"{float(Quu_inv[i, j]):.6f}f * ws.r[k][{j}]")
                    else:
                        quu_terms.append(f"{float(Quu_inv[i, j]):.6f}f * (({bt_p_expr}) + ws.r[k][{j}])")
                        
            if quu_terms:
                code += f"        ws.d[k][{i}] = {' + '.join(quu_terms)};\n"
            else:
                code += f"        ws.d[k][{i}] = 0.0f;\n"
                
        code += "\n        // p[k] = q[k] + AmBKt @ p[k+1] - Kinf.T @ r[k]\n"
        
        # Generate p[k] update
        for i in range(self.nx):
            terms = [f"ws.q[k][{i}]"]
            
            # AmBKt @ p[k+1] terms
            for j in range(self.nx):
                if abs(AmBKt[i, j]) > 1e-10:
                    terms.append(f"{float(AmBKt[i, j]):.6f}f * ws.p[k+1][{j}]")
                    
            # -Kinf.T @ r[k] terms
            for j in range(self.nu):
                if abs(Kinf[j, i]) > 1e-10:  # Kinf.T[i,j] = Kinf[j,i]
                    terms.append(f"{float(-Kinf[j, i]):.6f}f * ws.r[k][{j}]")
                    
            code += f"        ws.p[k][{i}] = {' + '.join(terms)};\n"
            
        code += "    }\n}\n"
        return code
        
    def generate_termination_check(self, tolerances: Dict[str, float]) -> str:
        """Generate termination check function with separate arrays"""
        abs_pri_tol = tolerances.get('abs_pri_tol', 1e-3)
        abs_dua_tol = tolerances.get('abs_dua_tol', 1e-3)
        rho = tolerances.get('rho', 1.0)
        
        return f"""
int tinympc_check_termination(tinympc_workspace_t& ws) {{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=ws.x complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.vnew complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.v complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.u complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.znew complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.z complete dim=2

    data_t pri_res_state = 0.0f;
    data_t pri_res_input = 0.0f;
    data_t dua_res_state = 0.0f;
    data_t dua_res_input = 0.0f;
    
    // Compute state residuals
    state_residual_loop: for (int k = 0; k < N; k++) {{
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
            data_t pri_diff = fabsf(ws.x[k][i] - ws.vnew[k][i]);
            data_t dua_diff = fabsf(ws.v[k][i] - ws.vnew[k][i]) * {float(rho):.6f}f;
            
            pri_res_state = fmaxf(pri_res_state, pri_diff);
            dua_res_state = fmaxf(dua_res_state, dua_diff);
        }}
    }}
    
    // Compute input residuals
    input_residual_loop: for (int k = 0; k < N-1; k++) {{
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NU; i++) {{
#pragma HLS UNROLL
            data_t pri_diff = fabsf(ws.u[k][i] - ws.znew[k][i]);
            data_t dua_diff = fabsf(ws.z[k][i] - ws.znew[k][i]) * {float(rho):.6f}f;
            
            pri_res_input = fmaxf(pri_res_input, pri_diff);
            dua_res_input = fmaxf(dua_res_input, dua_diff);
        }}
    }}
    
    // Check convergence
    return (pri_res_state < {float(abs_pri_tol):.6f}f && 
            pri_res_input < {float(abs_pri_tol):.6f}f &&
            dua_res_state < {float(abs_dua_tol):.6f}f && 
            dua_res_input < {float(abs_dua_tol):.6f}f) ? 1 : 0;
}}
"""
        
    def generate_main_solver(self, max_iter: int = 100, check_termination: int = 25) -> str:
        """Generate main TinyMPC solver function with separate BRAMs"""
        return f"""
void tinympc_solver(
    tinympc_workspace_t& ws,
    int max_iter = {max_iter},
    int check_termination_iter = {check_termination}
) {{
#pragma HLS INTERFACE m_axi port=ws bundle=main_memory
#pragma HLS ARRAY_PARTITION variable=ws.x complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.u complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.v complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.vnew complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.z complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.znew complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.g complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.y complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.d complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.q complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.p complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.r complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.xref complete dim=2
#pragma HLS ARRAY_PARTITION variable=ws.uref complete dim=2
#pragma HLS TOP

    // Main ADMM loop
    admm_loop: for (int iter = 0; iter < max_iter; iter++) {{
#pragma HLS LOOP_TRIPCOUNT min=10 max=100
        
        // Execute algorithm steps
        tinympc_forward_pass(ws);
        tinympc_update_slack(ws);
        tinympc_update_dual(ws);
        tinympc_update_linear_cost(ws);
        
        // Check termination periodically
        if (check_termination_iter > 0 && (iter % check_termination_iter == 0)) {{
            if (tinympc_check_termination(ws)) {{
                break;  // Converged
            }}
        }}
        
        // Update slack variables for next iteration
        copy_slack_loop: for (int k = 0; k < N; k++) {{
#pragma HLS PIPELINE II=1
            for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
                ws.v[k][i] = ws.vnew[k][i];
            }}
        }}
        
        copy_input_loop: for (int k = 0; k < N-1; k++) {{
#pragma HLS PIPELINE II=1
            for (int i = 0; i < NU; i++) {{
#pragma HLS UNROLL
                ws.z[k][i] = ws.znew[k][i];
            }}
        }}
        
        // Backward pass
        tinympc_backward_pass(ws);
    }}
    
    // Copy final results
    final_copy_x: for (int k = 0; k < N; k++) {{
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NX; i++) {{
#pragma HLS UNROLL
            ws.x[k][i] = ws.vnew[k][i];
        }}
    }}
    
    final_copy_u: for (int k = 0; k < N-1; k++) {{
#pragma HLS PIPELINE II=1
        for (int i = 0; i < NU; i++) {{
#pragma HLS UNROLL
            ws.u[k][i] = ws.znew[k][i];
        }}
    }}
}}
"""

    def generate_complete_solver(self, matrices: Dict[str, np.ndarray], 
                                bounds: Dict[str, np.ndarray], 
                                tolerances: Dict[str, float],
                                max_iter: int = 100,
                                check_termination: int = 25) -> Dict[str, str]:
        """Generate complete HLS solver with all functions"""
        
        header = self.generate_header("solver")
        forward_pass = self.generate_forward_pass(matrices)
        slack_update = self.generate_slack_update(bounds)
        dual_update = self.generate_dual_update()
        linear_cost = self.generate_linear_cost(matrices)
        backward_pass = self.generate_backward_pass(matrices)
        termination = self.generate_termination_check(tolerances)
        main_solver = self.generate_main_solver(max_iter, check_termination)
        
        # Generate header file content
        header_content = header + f"""
// Function prototypes
void tinympc_solver(tinympc_workspace_t& ws, int max_iter = {max_iter}, int check_termination_iter = {check_termination});

#endif
"""
        
        # Generate source file content  
        source_content = f"""#include "tinympc_solver.h"


{forward_pass}


{slack_update}


{dual_update}


{linear_cost}


{backward_pass}


{termination}


{main_solver}

"""
        
        return {
            'header': header_content,
            'source': source_content
        }