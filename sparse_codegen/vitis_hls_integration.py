"""
Vitis HLS Integration Tools
Automated C simulation and synthesis for TinyMPC
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class VitisHLSIntegration:
    """Manage Vitis HLS project creation and testing"""
    
    def __init__(self, project_name: str = "tinympc_hls", 
                 xilinx_path: str = "/tools/Xilinx/Vitis_HLS/2023.2"):
        self.project_name = project_name
        self.xilinx_path = xilinx_path
        self.settings_script = os.path.join(xilinx_path, "settings64.sh")
        
    def setup_environment(self) -> Dict[str, str]:
        """Setup Xilinx environment variables"""
        env = os.environ.copy()
        
        # Source Xilinx settings if available
        if os.path.exists(self.settings_script):
            # Get environment from sourced script
            cmd = f"bash -c 'source {self.settings_script} && env'"
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env[key] = value
                print(f"Xilinx environment loaded from {self.settings_script}")
            except Exception as e:
                print(f"Warning: Could not source Xilinx settings: {e}")
        else:
            print(f"Warning: Xilinx settings not found at {self.settings_script}")
            
        return env
        
    def generate_tcl_script(self, project_dir: str, 
                           cpp_files: List[str],
                           testbench_files: List[str],
                           top_function: str = "tinympc_solver",
                           target_part: str = "xc7z020clg400-1",
                           clock_period: float = 10.0,
                           run_csim: bool = True,
                           run_synthesis: bool = False,
                           run_cosim: bool = False) -> str:
        """Generate TCL script for HLS operations"""
        
        tcl_content = f"""# TinyMPC HLS Project Script
# Generated automatically

# Create and open project
open_project -reset {self.project_name}
set_top {top_function}

# Add source files
"""
        
        for cpp_file in cpp_files:
            tcl_content += f"add_files {cpp_file}\n"
            
        # Add testbench files
        for tb_file in testbench_files:
            tcl_content += f"add_files -tb {tb_file}\n"
            
        tcl_content += f"""
# Create solution
open_solution -reset "solution1" -flow_target vivado
set_part {target_part}
create_clock -period {clock_period} -name default

# Configure optimization directives
config_compile -name_max_length 50
config_interface -m_axi_offset slave
config_rtl -reset all

"""
        
        if run_csim:
            tcl_content += """
# Run C simulation
puts "Running C simulation..."
csim_design -clean

"""
        
        if run_synthesis:
            tcl_content += """
# Run synthesis  
puts "Running synthesis..."
csynth_design

"""
        
        if run_cosim:
            tcl_content += """
# Run co-simulation
puts "Running RTL co-simulation..."
cosim_design -trace_level all

"""
        
        tcl_content += """
# Close project
close_project
exit
"""
        
        tcl_path = os.path.join(project_dir, "run_hls.tcl")
        with open(tcl_path, 'w') as f:
            f.write(tcl_content)
            
        return tcl_path
        
    def generate_testbench(self, test_data_dir: str, 
                          nx: int, nu: int, N: int,
                          output_dir: str) -> str:
        """Generate C++ testbench for HLS validation"""
        
        testbench_code = f"""// TinyMPC HLS Testbench
// Generated automatically

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include "tinympc_solver.h"

using namespace std;

// Test tolerances
const float ABS_TOL = 1e-3f;
const float REL_TOL = 1e-2f;

// Utility functions
bool load_array_1d(const string& filename, vector<float>& data, int size) {{
    ifstream file(filename);
    if (!file.is_open()) {{
        cout << "Error: Cannot open file " << filename << endl;
        return false;
    }}
    
    data.resize(size);
    for (int i = 0; i < size; i++) {{
        file >> data[i];
    }}
    file.close();
    return true;
}}

bool load_array_2d(const string& filename, vector<vector<float>>& data, int rows, int cols) {{
    ifstream file(filename);
    if (!file.is_open()) {{
        cout << "Error: Cannot open file " << filename << endl;
        return false;
    }}
    
    data.resize(rows);
    for (int i = 0; i < rows; i++) {{
        data[i].resize(cols);
        for (int j = 0; j < cols; j++) {{
            file >> data[i][j];
        }}
    }}
    file.close();
    return true;
}}

bool compare_arrays(const vector<float>& a, const vector<float>& b, 
                   const string& name, float abs_tol = ABS_TOL, float rel_tol = REL_TOL) {{
    if (a.size() != b.size()) {{
        cout << "Error: Size mismatch for " << name << endl;
        return false;
    }}
    
    bool passed = true;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    
    for (size_t i = 0; i < a.size(); i++) {{
        float abs_error = abs(a[i] - b[i]);
        float rel_error = abs_error / max(abs(a[i]), abs(b[i]) + 1e-8f);
        
        max_abs_error = max(max_abs_error, abs_error);
        max_rel_error = max(max_rel_error, rel_error);
        
        if (abs_error > abs_tol && rel_error > rel_tol) {{
            cout << "Error at index " << i << ": expected " << a[i] 
                 << ", got " << b[i] << " (abs_err=" << abs_error 
                 << ", rel_err=" << rel_error << ")" << endl;
            passed = false;
        }}
    }}
    
    cout << name << ": max_abs_error=" << max_abs_error 
         << ", max_rel_error=" << max_rel_error;
    if (passed) {{
        cout << " [PASSED]" << endl;
    }} else {{
        cout << " [FAILED]" << endl;
    }}
    
    return passed;
}}

void workspace_to_arrays(workspace_t& workspace, 
                        vector<vector<float>>& x_out,
                        vector<vector<float>>& u_out) {{
    // Extract x and u from workspace
    for (int k = 0; k < N; k++) {{
        for (int i = 0; i < NX; i++) {{
            x_out[k][i] = X(k, i);
        }}
    }}
    
    for (int k = 0; k < N-1; k++) {{
        for (int i = 0; i < NU; i++) {{
            u_out[k][i] = U(k, i);
        }}
    }}
}}

void arrays_to_workspace(workspace_t& workspace,
                        const vector<float>& x0,
                        const vector<vector<float>>& Xref,
                        const vector<vector<float>>& Uref) {{
    // Set initial state
    for (int i = 0; i < NX; i++) {{
        X(0, i) = x0[i];
    }}
    
    // Set reference trajectories
    for (int k = 0; k < N; k++) {{
        for (int i = 0; i < NX; i++) {{
            XREF(k, i) = Xref[k][i];
        }}
    }}
    
    for (int k = 0; k < N-1; k++) {{
        for (int i = 0; i < NU; i++) {{
            UREF(k, i) = Uref[k][i];
        }}
    }}
}}

int main() {{
    cout << "TinyMPC HLS Testbench" << endl;
    cout << "Problem size: nx=" << NX << ", nu=" << NU << ", N=" << N << endl;
    
    // Test data directory
    string test_dir = "../test_data";
    
    // Test all available test cases
    vector<string> test_cases;
    for (int i = 0; i < 10; i++) {{  // Check up to 10 test cases
        string test_case_dir = test_dir + "/test_" + to_string(i);
        ifstream check(test_case_dir + "/x0_input.txt");
        if (check.good()) {{
            test_cases.push_back(test_case_dir);
        }}
        check.close();
    }}
    
    if (test_cases.empty()) {{
        cout << "Error: No test cases found in " << test_dir << endl;
        return 1;
    }}
    
    cout << "Found " << test_cases.size() << " test cases" << endl;
    
    int passed_tests = 0;
    int total_tests = test_cases.size();
    
    for (size_t test_idx = 0; test_idx < test_cases.size(); test_idx++) {{
        cout << "\\n--- Test Case " << test_idx << " ---" << endl;
        string test_case_dir = test_cases[test_idx];
        
        // Load test inputs
        vector<float> x0;
        vector<vector<float>> Xref, Uref;
        
        if (!load_array_1d(test_case_dir + "/x0_input.txt", x0, NX) ||
            !load_array_2d(test_case_dir + "/Xref_input.txt", Xref, N, NX) ||
            !load_array_2d(test_case_dir + "/Uref_input.txt", Uref, N-1, NU)) {{
            cout << "Error loading test inputs for test " << test_idx << endl;
            continue;
        }}
        
        // Load golden outputs
        vector<vector<float>> x_golden, u_golden;
        if (!load_array_2d(test_case_dir + "/x_golden_output.txt", x_golden, N, NX) ||
            !load_array_2d(test_case_dir + "/u_golden_output.txt", u_golden, N-1, NU)) {{
            cout << "Error loading golden outputs for test " << test_idx << endl;
            continue;
        }}
        
        // Initialize workspace
        workspace_t workspace;
        
        // Clear workspace
        for (int i = 0; i < WORKSPACE_DEPTH; i++) {{
            for (int j = 0; j < WORKSPACE_WIDTH; j++) {{
                workspace[i][j] = 0.0f;
            }}
        }}
        
        // Set test inputs
        arrays_to_workspace(workspace, x0, Xref, Uref);
        
        // Run HLS function
        cout << "Running HLS solver..." << endl;
        tinympc_solver(workspace);
        
        // Extract results
        vector<vector<float>> x_result(N, vector<float>(NX));
        vector<vector<float>> u_result(N-1, vector<float>(NU));
        workspace_to_arrays(workspace, x_result, u_result);
        
        // Compare results
        bool test_passed = true;
        
        // Flatten arrays for comparison
        vector<float> x_golden_flat, x_result_flat;
        vector<float> u_golden_flat, u_result_flat;
        
        for (int k = 0; k < N; k++) {{
            for (int i = 0; i < NX; i++) {{
                x_golden_flat.push_back(x_golden[k][i]);
                x_result_flat.push_back(x_result[k][i]);
            }}
        }}
        
        for (int k = 0; k < N-1; k++) {{
            for (int i = 0; i < NU; i++) {{
                u_golden_flat.push_back(u_golden[k][i]);
                u_result_flat.push_back(u_result[k][i]);
            }}
        }}
        
        // Validate results
        bool x_passed = compare_arrays(x_golden_flat, x_result_flat, "State trajectory X");
        bool u_passed = compare_arrays(u_golden_flat, u_result_flat, "Control trajectory U");
        
        test_passed = x_passed && u_passed;
        
        if (test_passed) {{
            cout << "Test " << test_idx << " PASSED" << endl;
            passed_tests++;
        }} else {{
            cout << "Test " << test_idx << " FAILED" << endl;
        }}
    }}
    
    cout << "\\n=== Test Summary ===" << endl;
    cout << "Passed: " << passed_tests << "/" << total_tests << endl;
    cout << "Success rate: " << fixed << setprecision(1) 
         << (100.0 * passed_tests / total_tests) << "%" << endl;
    
    if (passed_tests == total_tests) {{
        cout << "All tests PASSED!" << endl;
        return 0;
    }} else {{
        cout << "Some tests FAILED!" << endl;
        return 1;
    }}
}}"""
        
        testbench_path = os.path.join(output_dir, "testbench.cpp")
        with open(testbench_path, 'w') as f:
            f.write(testbench_code)
            
        return testbench_path
        
    def run_hls_flow(self, project_dir: str, tcl_script: str,
                    run_csim: bool = True,
                    run_synthesis: bool = False,
                    run_cosim: bool = False,
                    timeout: int = 600) -> Tuple[bool, str]:
        """Execute HLS flow using Vitis HLS"""
        
        env = self.setup_environment()
        
        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(project_dir)
        
        try:
            # Run Vitis HLS with TCL script
            cmd = ["vitis_hls", "-f", os.path.basename(tcl_script)]
            
            print(f"Running: {' '.join(cmd)}")
            print(f"Working directory: {project_dir}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=timeout,
                env=env
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            # Save output log
            with open("hls_output.log", "w") as f:
                f.write(output)
                
            if success:
                print("HLS flow completed successfully")
            else:
                print(f"HLS flow failed with return code {result.returncode}")
                
            return success, output
            
        except subprocess.TimeoutExpired:
            print(f"HLS flow timed out after {timeout} seconds")
            return False, "Timeout"
        except Exception as e:
            print(f"Error running HLS flow: {e}")
            return False, str(e)
        finally:
            os.chdir(original_cwd)
            
    def create_hls_project(self, output_dir: str, 
                          hls_code: str,
                          test_data_dir: str,
                          nx: int, nu: int, N: int,
                          header_code: Optional[str] = None,
                          run_csim: bool = True,
                          run_synthesis: bool = False) -> Tuple[bool, str]:
        """Create complete HLS project and run flow"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Write HLS source code
        solver_path = os.path.join(output_dir, "tinympc_solver.cpp")
        with open(solver_path, 'w') as f:
            f.write(hls_code)
            
        # Write header file if provided
        if header_code:
            header_path = os.path.join(output_dir, "tinympc_solver.h")
            with open(header_path, 'w') as f:
                f.write(header_code)
            
        # Generate testbench
        testbench_path = self.generate_testbench(test_data_dir, nx, nu, N, output_dir)
        
        # Create TCL script
        tcl_script = self.generate_tcl_script(
            project_dir=output_dir,
            cpp_files=["tinympc_solver.cpp"],
            testbench_files=["testbench.cpp"],
            run_csim=run_csim,
            run_synthesis=run_synthesis
        )
        
        print(f"HLS project created in {output_dir}")
        print(f"Files generated:")
        print(f"  - {solver_path}")
        print(f"  - {testbench_path}")
        print(f"  - {tcl_script}")
        
        # Run HLS flow
        if run_csim or run_synthesis:
            success, output = self.run_hls_flow(output_dir, tcl_script, run_csim, run_synthesis)
            return success, output
        else:
            return True, "Project created successfully"


def create_example_hls_project():
    """Create example HLS project for testing"""
    from .hls_emitter import HLSCodeEmitter
    from .test_data_generator import TestDataGenerator
    
    # System configuration
    nx, nu, N = 4, 2, 10
    
    # Generate test data
    print("Generating test data...")
    generator = TestDataGenerator(nx, nu, N)
    system = generator.generate_random_system()
    test_cases = generator.generate_test_trajectories(num_tests=2)
    
    test_data_dir = "example_test_data"
    generator.save_test_data(test_data_dir, system, test_cases)
    
    # Generate HLS code
    print("Generating HLS code...")
    emitter = HLSCodeEmitter(nx, nu, N)
    
    matrices = system.copy()
    matrices.update({
        'Kinf': np.random.randn(nu, nx).astype(np.float32),
        'Pinf': np.random.randn(nx, nx).astype(np.float32),
        'Quu_inv': np.random.randn(nu, nu).astype(np.float32),
        'AmBKt': np.random.randn(nx, nx).astype(np.float32),
        'rho': 1.0
    })
    
    bounds = {}
    tolerances = {'abs_pri_tol': 1e-3, 'abs_dua_tol': 1e-3, 'rho': 1.0}
    
    hls_code = emitter.generate_complete_solver(matrices, bounds, tolerances)
    
    # Create HLS project
    print("Creating HLS project...")
    hls_integration = VitisHLSIntegration()
    success, output = hls_integration.create_hls_project(
        output_dir="example_hls_project",
        hls_code=hls_code,
        test_data_dir=os.path.abspath(test_data_dir),
        nx=nx, nu=nu, N=N,
        run_csim=True,
        run_synthesis=False
    )
    
    print(f"HLS project creation: {'SUCCESS' if success else 'FAILED'}")
    if not success:
        print("Output:", output)


if __name__ == "__main__":
    create_example_hls_project()