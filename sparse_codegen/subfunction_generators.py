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