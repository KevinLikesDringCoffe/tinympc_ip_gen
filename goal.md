# TinyMPC HLS Code Generator

## Project Architecture

### Directory Structure
```
codegen/
├── goal.md                    # This project overview document
├── tinympcref.py             # Python reference implementation
├── sparse_codegen/           # HLS code generation modules
│   ├── __init__.py
│   ├── core.py               # Core data structures and utilities
│   ├── generator.py          # Base code generator
│   ├── tinympc_generator.py  # Main TinyMPC solver generator
│   ├── subfunction_generators.py  # Individual algorithm step generators
│   ├── emitters.py           # Code emission utilities
│   ├── ast_nodes.py          # AST node definitions
│   └── code_validator.py     # Generated code validation
└── unittest/                 # Unit testing framework
    ├── run_all_tests.py      # Test suite runner
    ├── subfunction_validator.py  # Component validation
    └── test_*.py             # Individual function tests
```

## TinyMPC Algorithm Overview

TinyMPC is a lightweight Model Predictive Control (MPC) solver optimized for embedded systems. The complete algorithm is implemented in `tinympcref.py` as a Python reference.

### Core Parameters
- **System Matrices**: `A` (state transition), `B` (control matrix)  
- **Cost Matrices**: `Q` (state cost), `R` (input cost)
- **Optimization Settings**: `N` (prediction horizon), `rho` (ADMM penalty factor)
- **Constraints**: `x_min/x_max` (state bounds), `u_min/u_max` (input bounds)

### Problem Formulation
Given initial state `x0`, reference trajectory `xref`, and reference input `uref`, TinyMPC solves for optimal control sequence `u` that:
- Minimizes quadratic cost over N-step horizon
- Tracks reference trajectory `xref`  
- Satisfies box constraints on states and inputs

### Pre-computed Matrices
The solver performs offline pre-computation of key matrices for runtime efficiency:
- **Kinf**: Infinite horizon gain matrix
- **Pinf**: Infinite horizon Riccati matrix  
- **Quu_inv**: Inverse of input Hessian
- **AmBKt**: Combined system matrix (A - B*K)^T

## Project Objectives

This project generates customized HLS IP cores for specific MPC problems. By exploiting the sparsity and fixed nature of system dynamics, matrix-vector operations are optimized through:

1. **Matrix Hardcoding**: Sparse system matrices embedded directly in hardware logic
2. **Custom IP Generation**: Problem-specific optimizations based on (A,B,Q,R,N,rho) parameters
3. **Efficient Memory Layout**: Optimized BRAM usage for workspace variables

### Hardware Interface Specification
The generated IP core exposes the following interface:
- **main_memory**: Unified memory interface containing:
  - Input: `x0` (initial state), `xref` (reference trajectory), `uref` (reference input)
  - Output: `x_out` (state trajectory), `u_out` (optimal control sequence)
- **max_iter**: Maximum ADMM iterations 
- **check_termination_iter**: Termination check frequency

## HLS Implementation Requirements

### Design Principles
The HLS implementation balances code readability with hardware performance through:

1. **Modular Architecture**: Each TinyMPC algorithm step implemented as separate HLS function:
   - `forward_pass`: Forward simulation with control constraints
   - `update_slack`: Slack variable updates for constraint handling
   - `update_dual`: Dual variable updates (ADMM)
   - `update_linear_cost`: Linear cost term updates
   - `backward_pass`: Backward Riccati recursion
   - `check_termination`: Primal/dual residual convergence check

2. **Memory Architecture**: 
   - Separate BRAM allocation for each algorithm variable
   - Workspace loaded from `main_memory` at computation start
   - Results written back to `main_memory` at completion

3. **Testing and Debugging Framework**:
   - Individual C testbenches for each HLS function
   - Automated golden reference generation from Python implementation
   - Comprehensive validation against reference outputs

### Vitis HLS Integration

#### Environment Setup
```bash
# Source Xilinx tools environment
source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
# or appropriate Xilinx installation path
```

#### C Simulation Workflow
Each generated HLS function includes corresponding test infrastructure:

1. **TCL Script Template** (`csim.tcl`):
```tcl
# Open project and add files
open_project -reset tinympc_hls
add_files tinympc_solver.cpp
add_files -tb testbench.cpp
set_top tinympc_solver

# Configure solution
open_solution -reset "solution1"
set_part {xc7z020clg400-1}  # or target FPGA part
create_clock -period 10

# Run C simulation
csim_design -clean
```

2. **Automated Test Execution**:
```bash
# Generate test data from Python reference  
python generate_test_data.py

# Run HLS C simulation
vitis_hls -f csim.tcl

# Validate results
python validate_csim_results.py
```

3. **Validation Flow**:
   - Python reference generates golden outputs
   - HLS testbench compares against golden data
   - Automated pass/fail reporting with detailed error analysis

### Performance Targets
- **Clock Frequency**: 100-250 MHz (depending on target FPGA)
- **Latency**: Sub-millisecond solver execution  
- **Resource Utilization**: Optimized BRAM and DSP usage
- **Throughput**: Support for 100Hz+ control loop rates