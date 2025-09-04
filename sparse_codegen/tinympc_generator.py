"""
Complete TinyMPC solver generator
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .core import Variable, DataType, get_sparsity_info, IndexExpr
from .generator import CodeGenerator
from .ast_nodes import *
from .subfunction_generators import (
    ForwardPassGenerator, SlackUpdateGenerator, DualUpdateGenerator, 
    LinearCostGenerator, BackwardPassGenerator, TerminationCheckGenerator
)


@dataclass
class TinyMPCConfig:
    """Configuration for TinyMPC solver generation"""
    unroll: bool = False
    inline: bool = False
    target: str = 'c'
    max_iter: int = 100
    check_termination: int = 25
    enable_state_bounds: bool = False
    enable_input_bounds: bool = False
    rho: float = 1.0


class TinyMPCGenerator:
    """Complete TinyMPC solver code generator"""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, N: int,
                 x_min: Optional[np.ndarray] = None, x_max: Optional[np.ndarray] = None,
                 u_min: Optional[np.ndarray] = None, u_max: Optional[np.ndarray] = None,
                 rho: float = 1.0):
        """
        Initialize TinyMPC generator
        
        Args:
            A: State transition matrix (nx x nx)
            B: Input matrix (nx x nu) 
            Q: State cost matrix (nx x nx, diagonal)
            R: Input cost matrix (nu x nu, diagonal)
            N: Horizon length
            x_min, x_max: State bounds (optional)
            u_min, u_max: Input bounds (optional)
            rho: ADMM penalty parameter
        """
        self.A = A.astype(np.float32)
        self.B = B.astype(np.float32)
        self.Q = Q.astype(np.float32)
        self.R = R.astype(np.float32)
        self.N = N
        self.rho = rho
        
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        
        # Compute derived matrices
        self._compute_derived_matrices()
        
        # Handle bounds
        self.has_state_bounds = x_min is not None and x_max is not None
        self.has_input_bounds = u_min is not None and u_max is not None
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        
        # Initialize code generator
        self.codegen = CodeGenerator()
        self._register_matrices()
        
        # Initialize subfunction generators
        self.forward_gen = ForwardPassGenerator(self.nx, self.nu, N)
        self.slack_gen = SlackUpdateGenerator(self.nx, self.nu, N, self.has_input_bounds, self.has_state_bounds)
        self.dual_gen = DualUpdateGenerator(self.nx, self.nu, N)
        self.cost_gen = LinearCostGenerator(self.nx, self.nu, N)
        self.backward_gen = BackwardPassGenerator(self.nx, self.nu, N)
        self.termination_gen = TerminationCheckGenerator(self.nx, self.nu, N)
    
    def _compute_derived_matrices(self):
        """Compute Kinf, Pinf, Quu_inv, AmBKt matrices"""
        # Add rho to cost matrices
        Q1 = self.Q + self.rho * np.eye(self.nx)
        R1 = self.R + self.rho * np.eye(self.nu)
        
        # Iterative computation of Kinf and Pinf
        Ktp1 = np.zeros((self.nu, self.nx))
        Ptp1 = self.rho * np.eye(self.nx)
        
        for i in range(1000):
            Kinf = np.linalg.inv(R1 + self.B.T @ Ptp1 @ self.B) @ (self.B.T @ Ptp1 @ self.A)
            Pinf = Q1 + self.A.T @ Ptp1 @ (self.A - self.B @ Kinf)
            
            if np.max(np.abs(Kinf - Ktp1)) < 1e-5:
                break
            
            Ktp1 = Kinf.copy()
            Ptp1 = Pinf.copy()
        
        self.Kinf = Kinf.astype(np.float32)
        self.Pinf = Pinf.astype(np.float32)
        self.Quu_inv = np.linalg.inv(R1 + self.B.T @ Pinf @ self.B).astype(np.float32)
        self.AmBKt = (self.A - self.B @ Kinf).T.astype(np.float32)
        
        # Diagonal matrices for element-wise operations
        self.Q_diag = np.diag(Q1).astype(np.float32)
        self.R_diag = np.diag(R1).astype(np.float32)
    
    def _register_matrices(self):
        """Register all matrices with sparsity information"""
        matrices = {
            'A': self.A,
            'B': self.B, 
            'Kinf': self.Kinf,
            'Pinf': self.Pinf,
            'Pinf_T': self.Pinf.T,
            'Quu_inv': self.Quu_inv,
            'AmBKt': self.AmBKt,
            'Kinf_T': self.Kinf.T,
            'B_T': self.B.T,
            'Q': self.Q_diag,
            'R': self.R_diag
        }
        
        for name, matrix in matrices.items():
            info = get_sparsity_info(matrix)
            self.codegen.register_variable(
                Variable(name, DataType.MATRIX, list(matrix.shape),
                        is_sparse=True, sparsity_pattern=info['pattern'], is_constant=True)
            )
    
    def generate(self, config: Optional[TinyMPCConfig] = None) -> str:
        """Generate complete TinyMPC solver C code with sparse optimization"""
        if config is None:
            config = TinyMPCConfig()
        
        # Generate header
        header = self._generate_header(config)
        
        # Generate subfunction declarations
        subfunc_declarations = self._generate_subfunction_declarations()
        
        # Generate sparse subfunctions (constants embedded)
        subfunctions = self._generate_subfunctions(config)
        
        # Generate variable declarations
        declarations = self._generate_declarations(config)
        
        # Generate solve function
        solve_body = self._generate_solve_function(config)
        
        return header + "\n" + subfunc_declarations + "\n" + subfunctions + "\n" + declarations + "\n" + solve_body
    
    def _generate_header(self, config: TinyMPCConfig) -> str:
        """Generate header with includes and defines"""
        lines = [
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <math.h>",
            "",
            f"// TinyMPC Sparse Solver - nx={self.nx}, nu={self.nu}, N={self.N}",
            f"// Generated with embedded constants and zero-skipping optimization",
            "",
            f"#define NX {self.nx}",
            f"#define NU {self.nu}",
            f"#define N {self.N}",
            f"#define MAX_ITER {config.max_iter}",
            f"#define CHECK_FREQ {config.check_termination}",
            f"#define RHO {config.rho:.6f}f",
            "",
            "// Tolerance settings",
            "#define ABS_PRI_TOL 1e-3f",
            "#define ABS_DUA_TOL 1e-3f",
            ""
        ]
        return '\n'.join(lines)
    
    def _generate_declarations(self, config: TinyMPCConfig) -> str:
        """Generate variable declarations"""
        lines = [
            "// Problem variables",
            f"float x[N][NX];",
            f"float u[N-1][NU];",
            f"float Xref[N][NX];",
            f"float Uref[N-1][NU];",
            "",
            "// ADMM variables", 
            f"float v[N][NX];",
            f"float vnew[N][NX];",
            f"float z[N-1][NU];",
            f"float znew[N-1][NU];",
            f"float g[N][NX];",
            f"float y[N-1][NU];",
            "",
            "// Cost variables",
            f"float q[N][NX];",
            f"float r[N-1][NU];",
            f"float p[N][NX];",
            f"float d[N-1][NU];",
            "",
            "// Algorithm state",
            "int iter;",
            "int solved;",
            "float primal_res_state;",
            "float dual_res_state;", 
            "float primal_res_input;",
            "float dual_res_input;",
            ""
        ]
        
        if self.has_state_bounds:
            lines.extend([
                f"float x_min[N][NX];",
                f"float x_max[N][NX];",
                ""
            ])
        
        if self.has_input_bounds:
            lines.extend([
                f"float u_min[N-1][NU];",
                f"float u_max[N-1][NU];",
                ""
            ])
        
        return '\n'.join(lines)
    
    def _generate_matrix_constants(self) -> str:
        """Generate matrix constant definitions"""
        lines = ["\n// Matrix constants"]
        
        # A matrix
        lines.append(f"const float A[NX][NX] = {{")
        for i in range(self.nx):
            row = ", ".join(f"{self.A[i,j]:.6f}f" for j in range(self.nx))
            lines.append(f"    {{{row}}},")
        lines[-1] = lines[-1].rstrip(',')  # Remove last comma
        lines.append("};")
        
        # B matrix
        lines.append(f"const float B[NX][NU] = {{")
        for i in range(self.nx):
            row = ", ".join(f"{self.B[i,j]:.6f}f" for j in range(self.nu))
            lines.append(f"    {{{row}}},")
        lines[-1] = lines[-1].rstrip(',')
        lines.append("};")
        
        # Kinf matrix
        lines.append(f"const float Kinf[NU][NX] = {{")
        for i in range(self.nu):
            row = ", ".join(f"{self.Kinf[i,j]:.6f}f" for j in range(self.nx))
            lines.append(f"    {{{row}}},")
        lines[-1] = lines[-1].rstrip(',')
        lines.append("};")
        
        # Pinf matrix
        lines.append(f"const float Pinf[NX][NX] = {{")
        for i in range(self.nx):
            row = ", ".join(f"{self.Pinf[i,j]:.6f}f" for j in range(self.nx))
            lines.append(f"    {{{row}}},")
        lines[-1] = lines[-1].rstrip(',')
        lines.append("};")
        
        # Quu_inv matrix
        lines.append(f"const float Quu_inv[NU][NU] = {{")
        for i in range(self.nu):
            row = ", ".join(f"{self.Quu_inv[i,j]:.6f}f" for j in range(self.nu))
            lines.append(f"    {{{row}}},")
        lines[-1] = lines[-1].rstrip(',')
        lines.append("};")
        
        # AmBKt matrix
        lines.append(f"const float AmBKt[NX][NX] = {{")
        for i in range(self.nx):
            row = ", ".join(f"{self.AmBKt[i,j]:.6f}f" for j in range(self.nx))
            lines.append(f"    {{{row}}},")
        lines[-1] = lines[-1].rstrip(',')
        lines.append("};")
        
        # Q and R diagonal matrices
        lines.append(f"const float Q[NX] = {{{', '.join(f'{self.Q_diag[i]:.6f}f' for i in range(self.nx))}}};")
        lines.append(f"const float R[NU] = {{{', '.join(f'{self.R_diag[i]:.6f}f' for i in range(self.nu))}}};")
        
        lines.append("")
        return '\n'.join(lines)
    
    def _generate_subfunction_declarations(self) -> str:
        """Generate forward declarations for subfunctions"""
        lines = [
            "\n// Function declarations",
            "void forward_pass(float x[N][NX], float u[N-1][NU], float d[N-1][NU]);",
            "void update_slack(float vnew[N][NX], float znew[N-1][NU], float x[N][NX], float u[N-1][NU], float g[N][NX], float y[N-1][NU]);",
            "void update_dual(float g[N][NX], float y[N-1][NU], float x[N][NX], float u[N-1][NU], float vnew[N][NX], float znew[N-1][NU]);",
            "void update_linear_cost(float q[N][NX], float r[N-1][NU], float p[N][NX], float Xref[N][NX], float Uref[N-1][NU], float vnew[N][NX], float znew[N-1][NU], float g[N][NX], float y[N-1][NU]);",
            "void backward_pass(float p[N][NX], float d[N-1][NU], float q[N][NX], float r[N-1][NU]);",
            "int check_termination(float x[N][NX], float u[N-1][NU], float vnew[N][NX], float znew[N-1][NU], float v[N][NX], float z[N-1][NU], int iter);",
            ""
        ]
        return '\n'.join(lines)
    
    def _generate_subfunctions(self, config: TinyMPCConfig) -> str:
        """Generate all subfunction implementations"""
        functions = []
        
        # Forward pass function
        functions.append(self._generate_forward_pass_function())
        
        # Slack update function
        functions.append(self._generate_slack_update_function())
        
        # Dual update function
        functions.append(self._generate_dual_update_function())
        
        # Linear cost function
        functions.append(self._generate_linear_cost_function())
        
        # Backward pass function
        functions.append(self._generate_backward_pass_function())
        
        # Termination check function
        functions.append(self._generate_termination_check_function())
        
        return '\n\n'.join(functions)
    
    def _generate_forward_pass_function(self) -> str:
        """Generate sparse forward pass function implementation"""
        lines = [
            "void forward_pass(float x[N][NX], float u[N-1][NU], float d[N-1][NU]) {",
            "    for (int k = 0; k < N-1; k++) {",
            "        // u[k] = -Kinf * x[k] - d[k] (sparse)",
        ]
        
        # Generate sparse Kinf multiplication 
        from .core import get_sparsity_info
        kinf_sparsity = get_sparsity_info(self.Kinf)
        
        for i in range(self.nu):
            lines.append(f"        u[k][{i}] = -d[k][{i}];")
            for (row, col), value in kinf_sparsity['pattern'].items():
                if row == i:
                    lines.append(f"        u[k][{i}] -= {value:.6f}f * x[k][{col}];")
        
        lines.extend([
            "        ",
            "        // x[k+1] = A * x[k] + B * u[k] (sparse)",
        ])
        
        # Generate sparse A and B multiplication
        a_sparsity = get_sparsity_info(self.A)
        b_sparsity = get_sparsity_info(self.B)
        
        for i in range(self.nx):
            lines.append(f"        x[k+1][{i}] = 0.0f;")
            
            # Sparse A multiplication
            for (row, col), value in a_sparsity['pattern'].items():
                if row == i:
                    if abs(value - 1.0) < 1e-10:
                        lines.append(f"        x[k+1][{i}] += x[k][{col}];")
                    else:
                        lines.append(f"        x[k+1][{i}] += {value:.6f}f * x[k][{col}];")
            
            # Sparse B multiplication  
            for (row, col), value in b_sparsity['pattern'].items():
                if row == i:
                    if abs(value - 1.0) < 1e-10:
                        lines.append(f"        x[k+1][{i}] += u[k][{col}];")
                    else:
                        lines.append(f"        x[k+1][{i}] += {value:.6f}f * u[k][{col}];")
        
        lines.extend([
            "    }",
            "}"
        ])
        
        return '\n'.join(lines)
    
    def _generate_slack_update_function(self) -> str:
        """Generate slack update function implementation"""
        lines = [
            "void update_slack(float vnew[N][NX], float znew[N-1][NU], float x[N][NX], float u[N-1][NU], float g[N][NX], float y[N-1][NU]) {",
            "    // znew = u + y",
            "    for (int k = 0; k < N-1; k++) {",
            "        for (int i = 0; i < NU; i++) {",
            "            znew[k][i] = u[k][i] + y[k][i];",
            "        }",
            "    }",
            "    ",
            "    // vnew = x + g", 
            "    for (int k = 0; k < N; k++) {",
            "        for (int i = 0; i < NX; i++) {",
            "            vnew[k][i] = x[k][i] + g[k][i];",
            "        }",
            "    }",
            "}"
        ]
        return '\n'.join(lines)
    
    def _generate_dual_update_function(self) -> str:
        """Generate dual update function implementation"""
        lines = [
            "void update_dual(float g[N][NX], float y[N-1][NU], float x[N][NX], float u[N-1][NU], float vnew[N][NX], float znew[N-1][NU]) {",
            "    // y = y + u - znew",
            "    for (int k = 0; k < N-1; k++) {",
            "        for (int i = 0; i < NU; i++) {",
            "            y[k][i] = y[k][i] + u[k][i] - znew[k][i];",
            "        }",
            "    }",
            "    ",
            "    // g = g + x - vnew",
            "    for (int k = 0; k < N; k++) {",
            "        for (int i = 0; i < NX; i++) {",
            "            g[k][i] = g[k][i] + x[k][i] - vnew[k][i];",
            "        }",
            "    }",
            "}"
        ]
        return '\n'.join(lines)
    
    def _generate_linear_cost_function(self) -> str:
        """Generate sparse linear cost update function implementation"""
        lines = [
            "void update_linear_cost(float q[N][NX], float r[N-1][NU], float p[N][NX], float Xref[N][NX], float Uref[N-1][NU], float vnew[N][NX], float znew[N-1][NU], float g[N][NX], float y[N-1][NU]) {",
            "    // r[i] = -Uref[i] * R - rho * (znew - y) (sparse)",
        ]
        
        from .core import get_sparsity_info
        
        # Generate sparse R multiplication for all inputs
        lines.append("    for (int k = 0; k < N-1; k++) {")
        for i in range(self.nu):
            r_val = self.R_diag[i]
            lines.append(f"        r[k][{i}] = -Uref[k][{i}] * {r_val:.6f}f - RHO * (znew[k][{i}] - y[k][{i}]);")
        lines.append("    }")
        
        lines.append("    ")
        lines.append("    // q[i] = -Xref[i] * Q - rho * (vnew - g) (sparse)")
        
        # Generate sparse Q multiplication for all states
        lines.append("    for (int k = 0; k < N; k++) {")
        for i in range(self.nx):
            q_val = self.Q_diag[i]
            lines.append(f"        q[k][{i}] = -Xref[k][{i}] * {q_val:.6f}f - RHO * (vnew[k][{i}] - g[k][{i}]);")
        lines.append("    }")
        
        lines.append("    ")
        lines.append("    // Terminal cost: p[N-1] = -Pinf.T @ Xref[N-1] - rho * (vnew[N-1] - g[N-1]) (sparse)")
        
        # Generate sparse Pinf.T multiplication
        pinf_sparsity = get_sparsity_info(self.Pinf)
        for i in range(self.nx):
            lines.append(f"    p[N-1][{i}] = -RHO * (vnew[N-1][{i}] - g[N-1][{i}]);")
            for (row, col), value in pinf_sparsity['pattern'].items():
                if col == i:  # Pinf.T[i][row] = Pinf[row][i]
                    if abs(value - 1.0) < 1e-10:
                        lines.append(f"    p[N-1][{i}] -= Xref[N-1][{row}];")
                    else:
                        lines.append(f"    p[N-1][{i}] -= {value:.6f}f * Xref[N-1][{row}];")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def _generate_backward_pass_function(self) -> str:
        """Generate sparse backward pass function implementation"""
        lines = [
            "void backward_pass(float p[N][NX], float d[N-1][NU], float q[N][NX], float r[N-1][NU]) {",
            "    for (int k = N-2; k >= 0; k--) {",
            "        // Compute B.T @ p[k+1] + r[k] (sparse)",
            "        float bt_p_plus_r[NU];",
        ]
        
        from .core import get_sparsity_info
        b_sparsity = get_sparsity_info(self.B)
        quu_inv_sparsity = get_sparsity_info(self.Quu_inv)
        ambkt_sparsity = get_sparsity_info(self.AmBKt)
        kinf_sparsity = get_sparsity_info(self.Kinf)
        
        # Sparse B.T @ p computation
        for i in range(self.nu):
            lines.append(f"        bt_p_plus_r[{i}] = r[k][{i}];")
            for (row, col), value in b_sparsity['pattern'].items():
                if col == i:  # B.T[i][row] = B[row][i]
                    if abs(value - 1.0) < 1e-10:
                        lines.append(f"        bt_p_plus_r[{i}] += p[k+1][{row}];")
                    else:
                        lines.append(f"        bt_p_plus_r[{i}] += {value:.6f}f * p[k+1][{row}];")
        
        lines.append("        ")
        lines.append("        // d[k] = Quu_inv @ bt_p_plus_r (sparse)")
        
        # Sparse Quu_inv multiplication
        for i in range(self.nu):
            lines.append(f"        d[k][{i}] = 0.0f;")
            for (row, col), value in quu_inv_sparsity['pattern'].items():
                if row == i:
                    if abs(value - 1.0) < 1e-10:
                        lines.append(f"        d[k][{i}] += bt_p_plus_r[{col}];")
                    else:
                        lines.append(f"        d[k][{i}] += {value:.6f}f * bt_p_plus_r[{col}];")
        
        lines.append("        ")
        lines.append("        // p[k] = q[k] + AmBKt @ p[k+1] - Kinf.T @ r[k] (sparse)")
        
        # Sparse AmBKt and Kinf.T computation
        for i in range(self.nx):
            lines.append(f"        p[k][{i}] = q[k][{i}];")
            lines.append("            ")
            
            # Sparse AmBKt @ p[k+1]
            for (row, col), value in ambkt_sparsity['pattern'].items():
                if row == i:
                    if abs(value - 1.0) < 1e-10:
                        lines.append(f"        p[k][{i}] += p[k+1][{col}];")
                    else:
                        lines.append(f"        p[k][{i}] += {value:.6f}f * p[k+1][{col}];")
            
            lines.append("            ")
            
            # Sparse - Kinf.T @ r[k]  
            for (row, col), value in kinf_sparsity['pattern'].items():
                if col == i:  # Kinf.T[i][row] = Kinf[row][i]
                    if abs(value - 1.0) < 1e-10:
                        lines.append(f"        p[k][{i}] -= r[k][{row}];")
                    else:
                        lines.append(f"        p[k][{i}] -= {value:.6f}f * r[k][{row}];")
        
        lines.extend([
            "    }",
            "}"
        ])
        
        return '\n'.join(lines)
    
    def _generate_termination_check_function(self) -> str:
        """Generate termination check function implementation"""
        lines = [
            "int check_termination(float x[N][NX], float u[N-1][NU], float vnew[N][NX], float znew[N-1][NU], float v[N][NX], float z[N-1][NU], int iter) {",
            "    if (CHECK_FREQ == 0 || iter % CHECK_FREQ != 0) {",
            "        return 0;",
            "    }",
            "    ",
            "    float primal_res_state = 0.0f;",
            "    float dual_res_state = 0.0f;",
            "    float primal_res_input = 0.0f;",
            "    float dual_res_input = 0.0f;",
            "    ",
            "    // Compute primal and dual residuals",
            "    for (int k = 0; k < N; k++) {",
            "        for (int i = 0; i < NX; i++) {",
            "            float pri_diff = fabsf(x[k][i] - vnew[k][i]);",
            "            float dual_diff = fabsf(v[k][i] - vnew[k][i]) * RHO;",
            "            if (pri_diff > primal_res_state) primal_res_state = pri_diff;",
            "            if (dual_diff > dual_res_state) dual_res_state = dual_diff;",
            "        }",
            "    }",
            "    ",
            "    for (int k = 0; k < N-1; k++) {",
            "        for (int i = 0; i < NU; i++) {",
            "            float pri_diff = fabsf(u[k][i] - znew[k][i]);",
            "            float dual_diff = fabsf(z[k][i] - znew[k][i]) * RHO;",
            "            if (pri_diff > primal_res_input) primal_res_input = pri_diff;",
            "            if (dual_diff > dual_res_input) dual_res_input = dual_diff;",
            "        }",
            "    }",
            "    ",
            "    return (primal_res_state < ABS_PRI_TOL && primal_res_input < ABS_PRI_TOL &&",
            "            dual_res_state < ABS_DUA_TOL && dual_res_input < ABS_DUA_TOL);",
            "}"
        ]
        return '\n'.join(lines)
    
    def _generate_solve_function(self, config: TinyMPCConfig) -> str:
        """Generate main solve function"""
        lines = [
            "void tinympc_solve(const float* x0, const float* xref, const float* uref, float* x_out, float* u_out, int* status) {",
            "    // Copy initial state",
            "    for (int i = 0; i < NX; i++) {",
            "        x[0][i] = x0[i];",
            "    }",
            "",
            "    // Copy reference trajectories", 
            "    for (int k = 0; k < N; k++) {",
            "        for (int i = 0; i < NX; i++) {",
            "            Xref[k][i] = xref[k * NX + i];",
            "        }",
            "    }",
            "    for (int k = 0; k < N-1; k++) {",
            "        for (int i = 0; i < NU; i++) {",
            "            Uref[k][i] = uref[k * NU + i];",
            "        }",
            "    }",
            "",
            "    // Initialize variables",
            "    iter = 0;",
            "    solved = 0;",
            "    ",
            "    // Initialize ADMM variables to zero",
            "    for (int k = 0; k < N; k++) {",
            "        for (int i = 0; i < NX; i++) {",
            "            if (k > 0) x[k][i] = 0.0f;  // x[1:N] = 0 (x[0] is set from x0)",
            "            q[k][i] = 0.0f;",
            "            p[k][i] = 0.0f;",
            "            v[k][i] = 0.0f;",
            "            vnew[k][i] = 0.0f;",
            "            g[k][i] = 0.0f;",
            "        }",
            "    }",
            "    for (int k = 0; k < N-1; k++) {",
            "        for (int i = 0; i < NU; i++) {",
            "            u[k][i] = 0.0f;",
            "            r[k][i] = 0.0f;",
            "            d[k][i] = 0.0f;",
            "            z[k][i] = 0.0f;",
            "            znew[k][i] = 0.0f;",
            "            y[k][i] = 0.0f;",
            "        }",
            "    }",
            "",
            "    // ADMM main loop",
            "    for (int i = 0; i < MAX_ITER; i++) {",
            "        // Forward pass",
            "        forward_pass(x, u, d);",
            "",
            "        // Update slack variables",
            "        update_slack(vnew, znew, x, u, g, y);",
            "",
            "        // Update dual variables",
            "        update_dual(g, y, x, u, vnew, znew);",
            "",
            "        // Update linear cost",
            "        update_linear_cost(q, r, p, Xref, Uref, vnew, znew, g, y);",
            "",
            "        // Update iteration counter",
            "        iter++;",
            "",
            "        // Check termination",
            "        if (check_termination(x, u, vnew, znew, v, z, iter)) {",
            "            solved = 1;",
            "            break;",
            "        }",
            "",
            "        // Update slack variable copies",
            "        for (int k = 0; k < N; k++) {",
            "            for (int j = 0; j < NX; j++) {",
            "                v[k][j] = vnew[k][j];",
            "            }",
            "        }",
            "        for (int k = 0; k < N-1; k++) {",
            "            for (int j = 0; j < NU; j++) {",
            "                z[k][j] = znew[k][j];",
            "            }",
            "        }",
            "",
            "        // Backward pass",
            "        backward_pass(p, d, q, r);",
            "    }",
            "",
            "    // Final update",
            "    for (int k = 0; k < N; k++) {",
            "        for (int i = 0; i < NX; i++) {", 
            "            x[k][i] = vnew[k][i];",
            "        }",
            "    }",
            "    for (int k = 0; k < N-1; k++) {",
            "        for (int i = 0; i < NU; i++) {",
            "            u[k][i] = znew[k][i];",
            "        }",
            "    }",
            "",
            "    // Copy outputs",
            "    for (int k = 0; k < N; k++) {",
            "        for (int i = 0; i < NX; i++) {",
            "            x_out[k * NX + i] = x[k][i];", 
            "        }",
            "    }",
            "    for (int k = 0; k < N-1; k++) {",
            "        for (int i = 0; i < NU; i++) {",
            "            u_out[k * NU + i] = u[k][i];",
            "        }",
            "    }",
            "",
            "    *status = solved;",
            "}"
        ]
        
        return '\n'.join(lines)
    
    
    def generate_embedded_constants(self, config: Optional[TinyMPCConfig] = None) -> str:
        """Generate solver with embedded constants for maximum optimization"""
        if config is None:
            config = TinyMPCConfig()
        
        # This will embed all matrix values directly in the code
        config.inline = True
        return self.generate(config)
    
    def generate_array_constants(self, config: Optional[TinyMPCConfig] = None) -> str:
        """Generate solver with array-based constants"""
        if config is None:
            config = TinyMPCConfig()
        
        header = self._generate_header(config)
        constants = self._generate_constant_arrays()
        declarations = self._generate_declarations(config)
        solve_body = self._generate_solve_function(config)
        
        return header + "\n" + constants + "\n" + subfunc_declarations + "\n" + subfunctions + "\n" + declarations + "\n" + solve_body
    
    def _generate_constant_arrays(self) -> str:
        """Generate constant array definitions"""
        lines = ["// Constant matrices"]
        
        # A matrix
        lines.append(f"const float A_data[NX][NX] = {{")
        for i in range(self.nx):
            row = ", ".join(f"{self.A[i,j]:.6f}f" for j in range(self.nx))
            lines.append(f"    {{{row}}},")
        lines.append("};")
        lines.append("")
        
        # B matrix
        lines.append(f"const float B_data[NX][NU] = {{")
        for i in range(self.nx):
            row = ", ".join(f"{self.B[i,j]:.6f}f" for j in range(self.nu))
            lines.append(f"    {{{row}}},")
        lines.append("};")
        lines.append("")
        
        # Kinf matrix
        lines.append(f"const float Kinf_data[NU][NX] = {{")
        for i in range(self.nu):
            row = ", ".join(f"{self.Kinf[i,j]:.6f}f" for j in range(self.nx))
            lines.append(f"    {{{row}}},")
        lines.append("};")
        lines.append("")
        
        # Other matrices...
        matrices = {
            'Quu_inv_data': self.Quu_inv,
            'AmBKt_data': self.AmBKt,
            'Q_data': self.Q_diag,
            'R_data': self.R_diag
        }
        
        for name, matrix in matrices.items():
            if matrix.ndim == 1:
                values = ", ".join(f"{val:.6f}f" for val in matrix)
                lines.append(f"const float {name}[{matrix.shape[0]}] = {{{values}}};")
            else:
                lines.append(f"const float {name}[{matrix.shape[0]}][{matrix.shape[1]}] = {{")
                for i in range(matrix.shape[0]):
                    row = ", ".join(f"{matrix[i,j]:.6f}f" for j in range(matrix.shape[1]))
                    lines.append(f"    {{{row}}},")
                lines.append("};")
            lines.append("")
        
        return '\n'.join(lines)
    
    def get_problem_info(self) -> Dict[str, Any]:
        """Get problem information for validation"""
        return {
            'nx': self.nx,
            'nu': self.nu,
            'N': self.N,
            'A': self.A,
            'B': self.B,
            'Q': self.Q,
            'R': self.R,
            'Kinf': self.Kinf,
            'Pinf': self.Pinf,
            'Quu_inv': self.Quu_inv,
            'AmBKt': self.AmBKt,
            'rho': self.rho,
            'has_state_bounds': self.has_state_bounds,
            'has_input_bounds': self.has_input_bounds
        }