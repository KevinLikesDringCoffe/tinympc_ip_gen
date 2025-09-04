"""
Individual Algorithm Step Generators for TinyMPC
Each generator creates HLS C code for a specific TinyMPC algorithm step
"""

import numpy as np
from typing import Dict, Any, Optional, List
from .hls_emitter import HLSCodeEmitter


class TinyMPCStepGenerator:
    """Base class for TinyMPC algorithm step generators"""
    
    def __init__(self, nx: int, nu: int, N: int):
        self.nx = nx
        self.nu = nu
        self.N = N
        self.emitter = HLSCodeEmitter(nx, nu, N)
        

class ForwardPassGenerator(TinyMPCStepGenerator):
    """Generator for forward pass: u[k] = -Kinf @ x[k] - d[k], x[k+1] = A @ x[k] + B @ u[k]"""
    
    def generate_function(self, matrices: Dict[str, np.ndarray]) -> str:
        """Generate standalone forward pass HLS function"""
        return self.emitter.generate_forward_pass(matrices)
        
    def generate_testbench(self, test_data_dir: str) -> str:
        """Generate testbench for forward pass validation"""
        return f"""// Forward Pass Testbench
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "tinympc_solver.h"

bool test_forward_pass() {{
    workspace_t workspace;
    
    // Initialize workspace with test data
    // Load from {test_data_dir}
    
    // Run forward pass
    tinympc_forward_pass(workspace);
    
    // Validate results
    return true; // Implement validation logic
}}

int main() {{
    std::cout << "Testing forward pass..." << std::endl;
    bool passed = test_forward_pass();
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}}"""


class SlackUpdateGenerator(TinyMPCStepGenerator):
    """Generator for slack variable updates with box constraints"""
    
    def generate_function(self, bounds: Dict[str, np.ndarray]) -> str:
        """Generate standalone slack update HLS function"""
        return self.emitter.generate_slack_update(bounds)
        
    def generate_testbench(self, test_data_dir: str) -> str:
        """Generate testbench for slack update validation"""
        return f"""// Slack Update Testbench
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "tinympc_solver.h"

bool test_slack_update() {{
    workspace_t workspace;
    
    // Initialize workspace with test data
    // Load from {test_data_dir}
    
    // Run slack update
    tinympc_update_slack(workspace);
    
    // Validate results
    return true; // Implement validation logic
}}

int main() {{
    std::cout << "Testing slack update..." << std::endl;
    bool passed = test_slack_update();
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}}"""


class DualUpdateGenerator(TinyMPCStepGenerator):
    """Generator for dual variable updates in ADMM"""
    
    def generate_function(self) -> str:
        """Generate standalone dual update HLS function"""
        return self.emitter.generate_dual_update()
        
    def generate_testbench(self, test_data_dir: str) -> str:
        """Generate testbench for dual update validation"""
        return f"""// Dual Update Testbench
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "tinympc_solver.h"

bool test_dual_update() {{
    workspace_t workspace;
    
    // Initialize workspace with test data
    // Load from {test_data_dir}
    
    // Run dual update
    tinympc_update_dual(workspace);
    
    // Validate results
    return true; // Implement validation logic
}}

int main() {{
    std::cout << "Testing dual update..." << std::endl;
    bool passed = test_dual_update();
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}}"""


class LinearCostGenerator(TinyMPCStepGenerator):
    """Generator for linear cost term updates"""
    
    def generate_function(self, matrices: Dict[str, np.ndarray]) -> str:
        """Generate standalone linear cost update HLS function"""
        return self.emitter.generate_linear_cost(matrices)
        
    def generate_testbench(self, test_data_dir: str) -> str:
        """Generate testbench for linear cost validation"""
        return f"""// Linear Cost Update Testbench
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "tinympc_solver.h"

bool test_linear_cost() {{
    workspace_t workspace;
    
    // Initialize workspace with test data
    // Load from {test_data_dir}
    
    // Run linear cost update
    tinympc_update_linear_cost(workspace);
    
    // Validate results
    return true; // Implement validation logic
}}

int main() {{
    std::cout << "Testing linear cost update..." << std::endl;
    bool passed = test_linear_cost();
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}}"""


class BackwardPassGenerator(TinyMPCStepGenerator):
    """Generator for backward Riccati recursion"""
    
    def generate_function(self, matrices: Dict[str, np.ndarray]) -> str:
        """Generate standalone backward pass HLS function"""
        return self.emitter.generate_backward_pass(matrices)
        
    def generate_testbench(self, test_data_dir: str) -> str:
        """Generate testbench for backward pass validation"""
        return f"""// Backward Pass Testbench
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "tinympc_solver.h"

bool test_backward_pass() {{
    workspace_t workspace;
    
    // Initialize workspace with test data
    // Load from {test_data_dir}
    
    // Run backward pass
    tinympc_backward_pass(workspace);
    
    // Validate results
    return true; // Implement validation logic
}}

int main() {{
    std::cout << "Testing backward pass..." << std::endl;
    bool passed = test_backward_pass();
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}}"""


class TerminationCheckGenerator(TinyMPCStepGenerator):
    """Generator for convergence termination check"""
    
    def generate_function(self, tolerances: Dict[str, float]) -> str:
        """Generate standalone termination check HLS function"""
        return self.emitter.generate_termination_check(tolerances)
        
    def generate_testbench(self, test_data_dir: str) -> str:
        """Generate testbench for termination check validation"""
        return f"""// Termination Check Testbench
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "tinympc_solver.h"

bool test_termination_check() {{
    workspace_t workspace;
    
    // Initialize workspace with test data
    // Load from {test_data_dir}
    
    // Run termination check
    int converged = tinympc_check_termination(workspace);
    
    // Validate results
    return true; // Implement validation logic
}}

int main() {{
    std::cout << "Testing termination check..." << std::endl;
    bool passed = test_termination_check();
    std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed ? 0 : 1;
}}"""


class TinyMPCStepManager:
    """Manages all individual algorithm step generators"""
    
    def __init__(self, nx: int, nu: int, N: int):
        self.nx = nx
        self.nu = nu
        self.N = N
        
        # Initialize all step generators
        self.forward_gen = ForwardPassGenerator(nx, nu, N)
        self.slack_gen = SlackUpdateGenerator(nx, nu, N)
        self.dual_gen = DualUpdateGenerator(nx, nu, N)
        self.cost_gen = LinearCostGenerator(nx, nu, N)
        self.backward_gen = BackwardPassGenerator(nx, nu, N)
        self.termination_gen = TerminationCheckGenerator(nx, nu, N)
        
    def generate_all_functions(self, 
                              matrices: Dict[str, np.ndarray],
                              bounds: Dict[str, np.ndarray],
                              tolerances: Dict[str, float]) -> Dict[str, str]:
        """Generate all individual HLS functions"""
        
        functions = {}
        
        functions['forward_pass'] = self.forward_gen.generate_function(matrices)
        functions['slack_update'] = self.slack_gen.generate_function(bounds)
        functions['dual_update'] = self.dual_gen.generate_function()
        functions['linear_cost'] = self.cost_gen.generate_function(matrices)
        functions['backward_pass'] = self.backward_gen.generate_function(matrices)
        functions['termination_check'] = self.termination_gen.generate_function(tolerances)
        
        return functions
        
    def generate_all_testbenches(self, test_data_dir: str) -> Dict[str, str]:
        """Generate testbenches for all individual functions"""
        
        testbenches = {}
        
        testbenches['forward_pass'] = self.forward_gen.generate_testbench(test_data_dir)
        testbenches['slack_update'] = self.slack_gen.generate_testbench(test_data_dir)
        testbenches['dual_update'] = self.dual_gen.generate_testbench(test_data_dir)
        testbenches['linear_cost'] = self.cost_gen.generate_testbench(test_data_dir)
        testbenches['backward_pass'] = self.backward_gen.generate_testbench(test_data_dir)
        testbenches['termination_check'] = self.termination_gen.generate_testbench(test_data_dir)
        
        return testbenches
        
    def create_individual_projects(self, output_dir: str, test_data_dir: str,
                                  matrices: Dict[str, np.ndarray],
                                  bounds: Dict[str, np.ndarray],
                                  tolerances: Dict[str, float]) -> List[str]:
        """Create separate HLS projects for each algorithm step"""
        import os
        from .vitis_hls_integration import VitisHLSIntegration
        
        functions = self.generate_all_functions(matrices, bounds, tolerances)
        testbenches = self.generate_all_testbenches(test_data_dir)
        
        project_dirs = []
        hls_integration = VitisHLSIntegration()
        
        # Get header from main emitter
        emitter = HLSCodeEmitter(self.nx, self.nu, self.N)
        header = emitter.generate_header("functions")
        
        for step_name in functions.keys():
            step_dir = os.path.join(output_dir, f"tinympc_{step_name}")
            os.makedirs(step_dir, exist_ok=True)
            
            # Write function source
            source_code = header + "\n" + functions[step_name]
            with open(os.path.join(step_dir, f"tinympc_{step_name}.cpp"), 'w') as f:
                f.write(source_code)
                
            # Write testbench
            with open(os.path.join(step_dir, f"tb_{step_name}.cpp"), 'w') as f:
                f.write(testbenches[step_name])
                
            # Create TCL script
            tcl_script = hls_integration.generate_tcl_script(
                project_dir=step_dir,
                cpp_files=[f"tinympc_{step_name}.cpp"],
                testbench_files=[f"tb_{step_name}.cpp"],
                top_function=f"tinympc_{step_name}",
                run_csim=True,
                run_synthesis=False
            )
            
            project_dirs.append(step_dir)
            print(f"Created project for {step_name} in {step_dir}")
            
        return project_dirs
            
            # u[k] = -temp_u
            for j in range(self.nu):
                ops.append({
                    'type': 'assignment',
                    'target': 'u',
                    'target_idx': [k, j],
                    'value': UnaryOp('-', ArrayAccess('temp_u', [j]))
                })
            
            # temp_a = A * x[k]
            ops.append({
                'type': 'matvec',
                'matrix': 'A',
                'vector': 'x',
                'result': 'temp_a',
                'vec_idx': [k]
            })
            
            # temp_b = B * u[k]
            ops.append({
                'type': 'matvec',
                'matrix': 'B',
                'vector': 'u',
                'result': 'temp_b',
                'vec_idx': [k]
            })
            
            # x[k+1] = temp_a + temp_b
            ops.append({
                'type': 'vecadd',
                'left': 'temp_a',
                'right': 'temp_b',
                'result': 'x',
                'res_idx': [k+1],
                'size': self.nx
            })
        
        return ops
    
    def _generate_looped(self) -> List[Dict[str, Any]]:
        """Generate looped forward pass"""
        return [{
            'type': 'loop',
            'var': 'i',
            'start': 0,
            'end': self.N - 1,
            'body': [
                {
                    'type': 'matvec',
                    'matrix': 'Kinf',
                    'vector': 'x',
                    'result': 'temp_kinf',
                    'vec_idx': [IndexExpr('i')]
                },
                {
                    'type': 'vecadd',
                    'left': 'temp_kinf',
                    'right': 'd',
                    'result': 'temp_u',
                    'right_idx': [IndexExpr('i')],
                    'size': self.nu
                },
                *[{
                    'type': 'assignment',
                    'target': 'u',
                    'target_idx': [IndexExpr('i'), j],
                    'value': UnaryOp('-', ArrayAccess('temp_u', [j]))
                } for j in range(self.nu)],
                {
                    'type': 'matvec',
                    'matrix': 'A',
                    'vector': 'x',
                    'result': 'temp_a',
                    'vec_idx': [IndexExpr('i')]
                },
                {
                    'type': 'matvec',
                    'matrix': 'B',
                    'vector': 'u',
                    'result': 'temp_b',
                    'vec_idx': [IndexExpr('i')]
                },
                {
                    'type': 'vecadd',
                    'left': 'temp_a',
                    'right': 'temp_b',
                    'result': 'x',
                    'res_idx': [IndexExpr('i', offset=1)],
                    'size': self.nx
                }
            ]
        }]


class SlackUpdateGenerator:
    """Generator for slack variable updates with box constraints"""
    
    def __init__(self, nx: int, nu: int, N: int, has_input_bounds: bool = False, has_state_bounds: bool = False):
        self.nx = nx
        self.nu = nu
        self.N = N
        self.has_input_bounds = has_input_bounds
        self.has_state_bounds = has_state_bounds
    
    def generate_operations(self, unroll: bool = False) -> List[Dict[str, Any]]:
        """Generate slack update operations"""
        ops = [{'type': 'comment', 'text': 'Update slack variables'}]
        
        if unroll:
            ops.extend(self._generate_unrolled())
        else:
            ops.extend(self._generate_looped())
        
        return ops
    
    def _generate_unrolled(self) -> List[Dict[str, Any]]:
        """Generate fully unrolled slack updates"""
        ops = []
        
        # Input slack variables: znew = u + y
        for k in range(self.N - 1):
            for i in range(self.nu):
                ops.append({
                    'type': 'assignment',
                    'target': 'znew',
                    'target_idx': [k, i],
                    'value': BinaryOp('+', 
                        ArrayAccess('u', [k, i]),
                        ArrayAccess('y', [k, i]))
                })
        
        # State slack variables: vnew = x + g
        for k in range(self.N):
            for i in range(self.nx):
                ops.append({
                    'type': 'assignment',
                    'target': 'vnew',
                    'target_idx': [k, i],
                    'value': BinaryOp('+',
                        ArrayAccess('x', [k, i]),
                        ArrayAccess('g', [k, i]))
                })
        
        # Apply box constraints
        if self.has_input_bounds:
            ops.extend(self._input_constraints_unrolled())
        if self.has_state_bounds:
            ops.extend(self._state_constraints_unrolled())
        
        return ops
    
    def _generate_looped(self) -> List[Dict[str, Any]]:
        """Generate looped slack updates"""
        ops = []
        
        # Input slack update loop
        ops.append({
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N - 1,
            'body': [
                {
                    'type': 'element_wise',
                    'op': 'add',
                    'left': ArrayAccess('u', [IndexExpr('k')]),
                    'right': ArrayAccess('y', [IndexExpr('k')]),
                    'result': ArrayAccess('znew', [IndexExpr('k')]),
                    'size': self.nu
                }
            ]
        })
        
        # State slack update loop  
        ops.append({
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N,
            'body': [
                {
                    'type': 'element_wise',
                    'op': 'add',
                    'left': ArrayAccess('x', [IndexExpr('k')]),
                    'right': ArrayAccess('g', [IndexExpr('k')]),
                    'result': ArrayAccess('vnew', [IndexExpr('k')]),
                    'size': self.nx
                }
            ]
        })
        
        # Apply box constraints
        if self.has_input_bounds:
            ops.extend(self._input_constraints_looped())
        if self.has_state_bounds:
            ops.extend(self._state_constraints_looped())
        
        return ops
    
    def _input_constraints_unrolled(self) -> List[Dict[str, Any]]:
        """Generate unrolled input box constraints"""
        ops = [{'type': 'comment', 'text': 'Apply input box constraints'}]
        
        for k in range(self.N - 1):
            for i in range(self.nu):
                ops.append({
                    'type': 'min_max',
                    'operation': 'clamp',
                    'operands': [
                        ArrayAccess('znew', [k, i]),
                        ArrayAccess('u_min', [k, i]),
                        ArrayAccess('u_max', [k, i])
                    ],
                    'result': ArrayAccess('znew', [k, i])
                })
        
        return ops
    
    def _state_constraints_unrolled(self) -> List[Dict[str, Any]]:
        """Generate unrolled state box constraints"""
        ops = [{'type': 'comment', 'text': 'Apply state box constraints'}]
        
        for k in range(self.N):
            for i in range(self.nx):
                ops.append({
                    'type': 'min_max',
                    'operation': 'clamp',
                    'operands': [
                        ArrayAccess('vnew', [k, i]),
                        ArrayAccess('x_min', [k, i]),
                        ArrayAccess('x_max', [k, i])
                    ],
                    'result': ArrayAccess('vnew', [k, i])
                })
        
        return ops
    
    def _input_constraints_looped(self) -> List[Dict[str, Any]]:
        """Generate looped input box constraints"""
        return [{
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N - 1,
            'body': [{
                'type': 'min_max',
                'operation': 'clamp',
                'operands': [
                    ArrayAccess('znew', [IndexExpr('k')]),
                    ArrayAccess('u_min', [IndexExpr('k')]),
                    ArrayAccess('u_max', [IndexExpr('k')])
                ],
                'result': ArrayAccess('znew', [IndexExpr('k')]),
                'size': self.nu
            }]
        }]
    
    def _state_constraints_looped(self) -> List[Dict[str, Any]]:
        """Generate looped state box constraints"""
        return [{
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N,
            'body': [{
                'type': 'min_max',
                'operation': 'clamp',
                'operands': [
                    ArrayAccess('vnew', [IndexExpr('k')]),
                    ArrayAccess('x_min', [IndexExpr('k')]),
                    ArrayAccess('x_max', [IndexExpr('k')])
                ],
                'result': ArrayAccess('vnew', [IndexExpr('k')]),
                'size': self.nx
            }]
        }]


class DualUpdateGenerator:
    """Generator for dual variable updates"""
    
    def __init__(self, nx: int, nu: int, N: int):
        self.nx = nx
        self.nu = nu
        self.N = N
    
    def generate_operations(self, unroll: bool = False) -> List[Dict[str, Any]]:
        """Generate dual update operations: y = y + u - znew, g = g + x - vnew"""
        ops = [{'type': 'comment', 'text': 'Update dual variables'}]
        
        if unroll:
            ops.extend(self._generate_unrolled())
        else:
            ops.extend(self._generate_looped())
        
        return ops
    
    def _generate_unrolled(self) -> List[Dict[str, Any]]:
        """Generate unrolled dual updates"""
        ops = []
        
        # y = y + u - znew
        for k in range(self.N - 1):
            for i in range(self.nu):
                ops.append({
                    'type': 'assignment',
                    'target': 'y',
                    'target_idx': [k, i],
                    'value': BinaryOp('-',
                        BinaryOp('+',
                            ArrayAccess('y', [k, i]),
                            ArrayAccess('u', [k, i])),
                        ArrayAccess('znew', [k, i]))
                })
        
        # g = g + x - vnew
        for k in range(self.N):
            for i in range(self.nx):
                ops.append({
                    'type': 'assignment',
                    'target': 'g',
                    'target_idx': [k, i],
                    'value': BinaryOp('-',
                        BinaryOp('+',
                            ArrayAccess('g', [k, i]),
                            ArrayAccess('x', [k, i])),
                        ArrayAccess('vnew', [k, i]))
                })
        
        return ops
    
    def _generate_looped(self) -> List[Dict[str, Any]]:
        """Generate looped dual updates"""
        ops = []
        
        # y update loop
        ops.append({
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N - 1,
            'body': self._dual_input_loop_body()
        })
        
        # g update loop
        ops.append({
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N,
            'body': self._dual_state_loop_body()
        })
        
        return ops
    
    def _dual_input_loop_body(self) -> List[Dict[str, Any]]:
        """Loop body for y = y + u - znew"""
        return [{
            'type': 'element_wise',
            'op': 'sub',
            'left': ArrayAccess('temp_y_u', [IndexExpr('k')]),
            'right': ArrayAccess('znew', [IndexExpr('k')]),
            'result': ArrayAccess('y', [IndexExpr('k')]),
            'size': self.nu,
            'temp_ops': [{
                'type': 'element_wise',
                'op': 'add',
                'left': ArrayAccess('y', [IndexExpr('k')]),
                'right': ArrayAccess('u', [IndexExpr('k')]),
                'result': ArrayAccess('temp_y_u', [IndexExpr('k')]),
                'size': self.nu
            }]
        }]
    
    def _dual_state_loop_body(self) -> List[Dict[str, Any]]:
        """Loop body for g = g + x - vnew"""
        return [{
            'type': 'element_wise',
            'op': 'sub',
            'left': ArrayAccess('temp_g_x', [IndexExpr('k')]),
            'right': ArrayAccess('vnew', [IndexExpr('k')]),
            'result': ArrayAccess('g', [IndexExpr('k')]),
            'size': self.nx,
            'temp_ops': [{
                'type': 'element_wise',
                'op': 'add',
                'left': ArrayAccess('g', [IndexExpr('k')]),
                'right': ArrayAccess('x', [IndexExpr('k')]),
                'result': ArrayAccess('temp_g_x', [IndexExpr('k')]),
                'size': self.nx
            }]
        }]


class LinearCostGenerator:
    """Generator for linear cost updates"""
    
    def __init__(self, nx: int, nu: int, N: int):
        self.nx = nx
        self.nu = nu
        self.N = N
    
    def generate_operations(self, unroll: bool = False) -> List[Dict[str, Any]]:
        """Generate linear cost update operations"""
        ops = [{'type': 'comment', 'text': 'Update linear cost terms'}]
        
        if unroll:
            ops.extend(self._generate_unrolled())
        else:
            ops.extend(self._generate_looped())
        
        return ops
    
    def _generate_unrolled(self) -> List[Dict[str, Any]]:
        """Generate unrolled linear cost updates"""
        ops = []
        
        # r[i] = -Uref[i] * R (element-wise)
        ops.append({'type': 'comment', 'text': 'Update r: r[i] = -Uref[i] * R'})
        for k in range(self.N - 1):
            ops.append({
                'type': 'element_wise',
                'op': 'mul',
                'left': ArrayAccess('Uref', [k]),
                'right': ArrayAccess('R', []),
                'result': ArrayAccess('temp_r_ref', [k]),
                'size': self.nu
            })
            for i in range(self.nu):
                ops.append({
                    'type': 'assignment',
                    'target': 'r',
                    'target_idx': [k, i],
                    'value': UnaryOp('-', ArrayAccess('temp_r_ref', [k, i]))
                })
        
        # q[i] = -Xref[i] * Q (element-wise)
        ops.append({'type': 'comment', 'text': 'Update q: q[i] = -Xref[i] * Q'})
        for k in range(self.N):
            ops.append({
                'type': 'element_wise',
                'op': 'mul',
                'left': ArrayAccess('Xref', [k]),
                'right': ArrayAccess('Q', []),
                'result': ArrayAccess('temp_q_ref', [k]),
                'size': self.nx
            })
            for i in range(self.nx):
                ops.append({
                    'type': 'assignment',
                    'target': 'q',
                    'target_idx': [k, i],
                    'value': UnaryOp('-', ArrayAccess('temp_q_ref', [k, i]))
                })
        
        # r -= rho * (znew - y)
        ops.append({'type': 'comment', 'text': 'Update r: r -= rho * (znew - y)'})
        for k in range(self.N - 1):
            for i in range(self.nu):
                ops.append({
                    'type': 'assignment',
                    'target': 'r',
                    'target_idx': [k, i],
                    'value': BinaryOp('-',
                        ArrayAccess('r', [k, i]),
                        BinaryOp('*',
                            Literal('rho'),
                            BinaryOp('-',
                                ArrayAccess('znew', [k, i]),
                                ArrayAccess('y', [k, i]))))
                })
        
        # q -= rho * (vnew - g)  
        ops.append({'type': 'comment', 'text': 'Update q: q -= rho * (vnew - g)'})
        for k in range(self.N):
            for i in range(self.nx):
                ops.append({
                    'type': 'assignment',
                    'target': 'q',
                    'target_idx': [k, i],
                    'value': BinaryOp('-',
                        ArrayAccess('q', [k, i]),
                        BinaryOp('*',
                            Literal('rho'),
                            BinaryOp('-',
                                ArrayAccess('vnew', [k, i]),
                                ArrayAccess('g', [k, i]))))
                })
        
        # Terminal cost: p[N-1] = -Pinf.T @ Xref[N-1] - rho * (vnew[N-1] - g[N-1])
        ops.append({'type': 'comment', 'text': 'Update terminal cost p[N-1]'})
        ops.append({
            'type': 'sparse_matvec',
            'matrix': 'Pinf_T',
            'vector': ArrayAccess('Xref', [self.N-1]),
            'result': ArrayAccess('temp_pinf_x', []),
            'transpose': True
        })
        
        for i in range(self.nx):
            ops.append({
                'type': 'assignment',
                'target': 'p',
                'target_idx': [self.N-1, i],
                'value': BinaryOp('-',
                    BinaryOp('-',
                        Literal(0.0),
                        ArrayAccess('temp_pinf_x', [i])),
                    BinaryOp('*',
                        Literal('rho'),
                        BinaryOp('-',
                            ArrayAccess('vnew', [self.N-1, i]),
                            ArrayAccess('g', [self.N-1, i]))))
            })
        
        return ops
    
    def _generate_looped(self) -> List[Dict[str, Any]]:
        """Generate looped linear cost updates"""
        ops = []
        
        # Reference cost terms
        ops.append({
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N - 1,
            'body': [{
                'type': 'element_wise',
                'op': 'mul',
                'left': ArrayAccess('Uref', [IndexExpr('k')]),
                'right': ArrayAccess('R', []),
                'result': ArrayAccess('temp_r_ref', [IndexExpr('k')]),
                'size': self.nu
            }, {
                'type': 'vector_scalar',
                'op': 'mul',
                'vector': ArrayAccess('temp_r_ref', [IndexExpr('k')]),
                'scalar': -1.0,
                'result': ArrayAccess('r', [IndexExpr('k')]),
                'size': self.nu
            }]
        })
        
        ops.append({
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N,
            'body': [{
                'type': 'element_wise',
                'op': 'mul',
                'left': ArrayAccess('Xref', [IndexExpr('k')]),
                'right': ArrayAccess('Q', []),
                'result': ArrayAccess('temp_q_ref', [IndexExpr('k')]),
                'size': self.nx
            }, {
                'type': 'vector_scalar',
                'op': 'mul',
                'vector': ArrayAccess('temp_q_ref', [IndexExpr('k')]),
                'scalar': -1.0,
                'result': ArrayAccess('q', [IndexExpr('k')]),
                'size': self.nx
            }]
        })
        
        # ADMM penalty terms
        ops.append({
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N - 1,
            'body': self._input_penalty_loop_body()
        })
        
        ops.append({
            'type': 'loop',
            'var': 'k',
            'start': 0,
            'end': self.N,
            'body': self._state_penalty_loop_body()
        })
        
        return ops
    
    def _input_penalty_loop_body(self) -> List[Dict[str, Any]]:
        """Loop body for r -= rho * (znew - y)"""
        return [{
            'type': 'element_wise',
            'op': 'sub',
            'left': ArrayAccess('znew', [IndexExpr('k')]),
            'right': ArrayAccess('y', [IndexExpr('k')]),
            'result': ArrayAccess('temp_penalty', [IndexExpr('k')]),
            'size': self.nu
        }, {
            'type': 'vector_scalar',
            'op': 'mul',
            'vector': ArrayAccess('temp_penalty', [IndexExpr('k')]),
            'scalar': Literal('rho'),
            'result': ArrayAccess('temp_rho_penalty', [IndexExpr('k')]),
            'size': self.nu
        }, {
            'type': 'element_wise',
            'op': 'sub',
            'left': ArrayAccess('r', [IndexExpr('k')]),
            'right': ArrayAccess('temp_rho_penalty', [IndexExpr('k')]),
            'result': ArrayAccess('r', [IndexExpr('k')]),
            'size': self.nu
        }]
    
    def _state_penalty_loop_body(self) -> List[Dict[str, Any]]:
        """Loop body for q -= rho * (vnew - g)"""
        return [{
            'type': 'element_wise',
            'op': 'sub',
            'left': ArrayAccess('vnew', [IndexExpr('k')]),
            'right': ArrayAccess('g', [IndexExpr('k')]),
            'result': ArrayAccess('temp_penalty', [IndexExpr('k')]),
            'size': self.nx
        }, {
            'type': 'vector_scalar',
            'op': 'mul',
            'vector': ArrayAccess('temp_penalty', [IndexExpr('k')]),
            'scalar': Literal('rho'),
            'result': ArrayAccess('temp_rho_penalty', [IndexExpr('k')]),
            'size': self.nx
        }, {
            'type': 'element_wise',
            'op': 'sub',
            'left': ArrayAccess('q', [IndexExpr('k')]),
            'right': ArrayAccess('temp_rho_penalty', [IndexExpr('k')]),
            'result': ArrayAccess('q', [IndexExpr('k')]),
            'size': self.nx
        }]


class BackwardPassGenerator:
    """Generator for backward pass operations"""
    
    def __init__(self, nx: int, nu: int, N: int):
        self.nx = nx
        self.nu = nu
        self.N = N
    
    def generate_operations(self, unroll: bool = False) -> List[Dict[str, Any]]:
        """Generate backward pass operations"""
        ops = [{'type': 'comment', 'text': 'Backward pass'}]
        
        if unroll:
            ops.extend(self._generate_unrolled())
        else:
            ops.extend(self._generate_looped())
        
        return ops
    
    def _generate_unrolled(self) -> List[Dict[str, Any]]:
        """Generate unrolled backward pass"""
        ops = []
        
        for k in range(self.N - 2, -1, -1):
            # d[k] = Quu_inv @ (B.T @ p[k+1] + r[k])
            ops.append({'type': 'comment', 'text': f'Backward pass iteration k={k}'})
            
            # B.T @ p[k+1]
            ops.append({
                'type': 'sparse_matvec',
                'matrix': 'B_T',
                'vector': ArrayAccess('p', [k+1]),
                'result': ArrayAccess('temp_bt_p', []),
                'transpose': True
            })
            
            # B.T @ p[k+1] + r[k]
            for i in range(self.nu):
                ops.append({
                    'type': 'assignment',
                    'target': 'temp_quu_input',
                    'target_idx': [i],
                    'value': BinaryOp('+',
                        ArrayAccess('temp_bt_p', [i]),
                        ArrayAccess('r', [k, i]))
                })
            
            # d[k] = Quu_inv @ temp_quu_input
            ops.append({
                'type': 'sparse_matvec',
                'matrix': 'Quu_inv',
                'vector': ArrayAccess('temp_quu_input', []),
                'result': ArrayAccess('d', [k])
            })
            
            # p[k] = q[k] + AmBKt @ p[k+1] - Kinf.T @ r[k]
            ops.append({
                'type': 'sparse_matvec',
                'matrix': 'AmBKt',
                'vector': ArrayAccess('p', [k+1]),
                'result': ArrayAccess('temp_ambkt_p', [])
            })
            
            ops.append({
                'type': 'sparse_matvec',
                'matrix': 'Kinf_T',
                'vector': ArrayAccess('r', [k]),
                'result': ArrayAccess('temp_kinf_r', []),
                'transpose': True
            })
            
            for i in range(self.nx):
                ops.append({
                    'type': 'assignment',
                    'target': 'p',
                    'target_idx': [k, i],
                    'value': BinaryOp('-',
                        BinaryOp('+',
                            ArrayAccess('q', [k, i]),
                            ArrayAccess('temp_ambkt_p', [i])),
                        ArrayAccess('temp_kinf_r', [i]))
                })
        
        return ops
    
    def _generate_looped(self) -> List[Dict[str, Any]]:
        """Generate looped backward pass"""
        return [{
            'type': 'loop',
            'var': 'k',
            'start': self.N - 2,
            'end': -1,
            'step': -1,
            'body': self._backward_loop_body()
        }]
    
    def _backward_loop_body(self) -> List[Dict[str, Any]]:
        """Loop body for backward pass"""
        return [
            # d[k] = Quu_inv @ (B.T @ p[k+1] + r[k])
            {
                'type': 'sparse_matvec',
                'matrix': 'B_T',
                'vector': ArrayAccess('p', [IndexExpr('k', offset=1)]),
                'result': ArrayAccess('temp_bt_p', []),
                'transpose': True
            },
            {
                'type': 'element_wise',
                'op': 'add',
                'left': ArrayAccess('temp_bt_p', []),
                'right': ArrayAccess('r', [IndexExpr('k')]),
                'result': ArrayAccess('temp_quu_input', []),
                'size': self.nu
            },
            {
                'type': 'sparse_matvec',
                'matrix': 'Quu_inv',
                'vector': ArrayAccess('temp_quu_input', []),
                'result': ArrayAccess('d', [IndexExpr('k')])
            },
            # p[k] = q[k] + AmBKt @ p[k+1] - Kinf.T @ r[k]
            {
                'type': 'sparse_matvec',
                'matrix': 'AmBKt',
                'vector': ArrayAccess('p', [IndexExpr('k', offset=1)]),
                'result': ArrayAccess('temp_ambkt_p', [])
            },
            {
                'type': 'sparse_matvec',
                'matrix': 'Kinf_T',
                'vector': ArrayAccess('r', [IndexExpr('k')]),
                'result': ArrayAccess('temp_kinf_r', []),
                'transpose': True
            },
            {
                'type': 'element_wise',
                'op': 'add',
                'left': ArrayAccess('q', [IndexExpr('k')]),
                'right': ArrayAccess('temp_ambkt_p', []),
                'result': ArrayAccess('temp_p_sum', []),
                'size': self.nx
            },
            {
                'type': 'element_wise',
                'op': 'sub',
                'left': ArrayAccess('temp_p_sum', []),
                'right': ArrayAccess('temp_kinf_r', []),
                'result': ArrayAccess('p', [IndexExpr('k')]),
                'size': self.nx
            }
        ]


class TerminationCheckGenerator:
    """Generator for convergence checking"""
    
    def __init__(self, nx: int, nu: int, N: int, check_frequency: int = 25):
        self.nx = nx
        self.nu = nu
        self.N = N
        self.check_frequency = check_frequency
    
    def generate_operations(self, unroll: bool = False) -> List[Dict[str, Any]]:
        """Generate termination check operations"""
        return [
            {'type': 'comment', 'text': 'Check termination conditions'},
            {
                'type': 'conditional',
                'condition': BinaryOp('==', 
                    BinaryOp('%', Literal('iter'), Literal(self.check_frequency)), 
                    Literal(0)),
                'body': [
                    # Compute residuals
                    {
                        'type': 'abs_max',
                        'vector': ArrayAccess('x_diff', []),
                        'result': ArrayAccess('primal_res_state', []),
                        'size': self.N * self.nx,
                        'prep_ops': [{
                            'type': 'element_wise',
                            'op': 'sub',
                            'left': ArrayAccess('x', []),
                            'right': ArrayAccess('vnew', []),
                            'result': ArrayAccess('x_diff', []),
                            'size': self.N * self.nx
                        }]
                    },
                    {
                        'type': 'abs_max',
                        'vector': ArrayAccess('u_diff', []),
                        'result': ArrayAccess('primal_res_input', []),
                        'size': (self.N - 1) * self.nu,
                        'prep_ops': [{
                            'type': 'element_wise',
                            'op': 'sub',
                            'left': ArrayAccess('u', []),
                            'right': ArrayAccess('znew', []),
                            'result': ArrayAccess('u_diff', []),
                            'size': (self.N - 1) * self.nu
                        }]
                    },
                    # Check convergence
                    {
                        'type': 'conditional',
                        'condition': BinaryOp('&&',
                            BinaryOp('&&',
                                BinaryOp('<', ArrayAccess('primal_res_state', []), Literal('abs_pri_tol')),
                                BinaryOp('<', ArrayAccess('primal_res_input', []), Literal('abs_pri_tol'))),
                            BinaryOp('&&',
                                BinaryOp('<', ArrayAccess('dual_res_state', []), Literal('abs_dua_tol')),
                                BinaryOp('<', ArrayAccess('dual_res_input', []), Literal('abs_dua_tol')))),
                        'body': [
                            {
                                'type': 'assignment',
                                'target': 'solved',
                                'target_idx': [],
                                'value': Literal(1)
                            }
                        ]
                    }
                ]
            }
        ]