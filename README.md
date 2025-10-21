# TinyMPC FPGA IP Generator

Automated HLS code generation and FPGA deployment framework for TinyMPC (Model Predictive Control) optimization on Ultra96 boards.

## Overview

This project generates custom hardware accelerators for specific MPC problems by exploiting sparsity patterns in system matrices. Each generated bitstream is optimized for a particular control problem configuration (horizon N, control frequency, dynamics model).

**Key Features:**
- Automated HLS C++ code generation from Python reference implementation
- Problem-specific optimization through matrix hardcoding
- Drop-in hardware driver compatible with software solver API
- Support for 100Hz+ control loop rates with <5ms solve time

## Quick Start

### Prerequisites

**For bitstream generation:**
- Xilinx Vitis HLS 2021.2+
- Xilinx Vivado 2021.2+
- Python 3.7+ with numpy

**For FPGA deployment:**
- Ultra96 board with PYNQ Linux
- PYNQ Python package

### Build Bitstream

```bash
# Generate bitstream for N=5 horizon at 100Hz control frequency
make N=5 FREQ=100

# Output: bitstream/tinympcproj_N5_100Hz_float.bit
#         bitstream/tinympcproj_N5_100Hz_float.hwh
```

### Deploy on Ultra96

```python
from driver import tinympc_hw

# Load bitstream and create solver
solver = tinympc_hw("bitstream/tinympcproj_N5_100Hz_float.bit")

# Configure solver
solver.setup(max_iter=100, check_termination=10)

# Set problem data
solver.set_x0(x0)          # Initial state [nx]
solver.set_x_ref(xref)     # Reference trajectory [N, nx]
solver.set_u_ref(uref)     # Reference inputs [N-1, nu]

# Solve and get results
solver.solve()
print(f"Solve time: {solver.solve_time:.2f}ms")
print(f"Optimal control: {solver.u}")
```

## Project Structure

```
.
├── Makefile                      # Build automation
├── tinympc_ip_generator.py       # Main HLS code generator
├── hls_generator.py              # HLS generation utilities
├── tinympcref.py                 # Python reference solver
├── dynamics.py                   # System dynamics models
├── impl/                         # Vivado implementation scripts
│   ├── workflow.tcl              # Bitstream generation flow
│   └── ultra96_bd.tcl            # Block design for Ultra96
├── driver/                       # Hardware driver for PYNQ
│   ├── tinympc_hw.py             # High-level driver API
│   └── tinympc_driver.py         # Low-level hardware interface
├── notebooks/                    # Example notebooks
│   ├── hardware_performance_test.ipynb
│   ├── hw_sw_comparison.ipynb
│   └── simple_tracking_demo.ipynb
└── bitstream/                    # Generated bitstream files (created by make)
```

## Build Stages

The build process consists of three stages:

### 1. Code Generation
```bash
make N=5 FREQ=100 generate
```
Generates HLS C++ code in `tinympcproj_N5_100Hz_float/`:
- `tinympc_solver.cpp/h` - HLS implementation with hardcoded matrices
- `testbench.cpp` - C simulation testbench
- `test_data.h` - Golden reference data

### 2. HLS Synthesis
```bash
make N=5 FREQ=100 hls
```
Runs Vitis HLS to:
- Perform C simulation validation
- Synthesize RTL
- Package as Vivado IP core

### 3. Bitstream Generation
```bash
make N=5 FREQ=100 bitstream
```
Runs Vivado to:
- Integrate HLS IP into Ultra96 block design
- Implement and generate bitstream
- Extract `.bit` and `.hwh` files to `bitstream/`

## Configuration Parameters

- **N**: MPC horizon length (default: 40)
- **FREQ**: Control frequency in Hz (default: 100)
- **Precision**: Float (32-bit) or double (64-bit)

Example:
```bash
make N=10 FREQ=200    # 10-step horizon, 200Hz control
```

## Hardware Driver API

The driver provides a drop-in replacement for the software solver:

```python
# Replace software solver:
# from tinympcref import tinympcref as Solver

# With hardware solver:
from driver import tinympc_hw as Solver

solver = Solver("bitstream/tinympcproj_N5_100Hz_float.bit")
# Identical API from here...
```

**Key Methods:**
- `setup(**kwargs)` - Configure solver parameters
- `set_x0(x0)` - Set initial state
- `set_x_ref(xref)` - Set reference trajectory
- `set_u_ref(uref)` - Set reference inputs
- `solve()` - Synchronous solve
- `solve_async()` - Asynchronous solve

**Properties:**
- `x` - Optimal state trajectory
- `u` - Optimal control inputs
- `solve_time` - Hardware execution time (ms)
- `iter` - Number of iterations
- `solved` - Convergence flag

## Examples

See `notebooks/` for detailed examples:
- **simple_tracking_demo.ipynb** - Basic trajectory tracking
- **hw_sw_comparison.ipynb** - Hardware vs software comparison
- **hardware_performance_test.ipynb** - Performance benchmarking

## Clean Build

```bash
make clean        # Remove build artifacts
make distclean    # Remove all generated files including HLS code
```

## System Requirements

**Build Environment:**
- Xilinx Vitis HLS 2021.2 or later
- Xilinx Vivado 2021.2 or later
- Python 3.7+ with numpy

**Target Platform:**
- Ultra96 or compatible Zynq UltraScale+ board
- PYNQ Linux 2.6+
- Minimum 512MB RAM

## Algorithm Overview

TinyMPC uses ADMM (Alternating Direction Method of Multipliers) to solve:

```
minimize   Σ ||x[k] - xref[k]||²_Q + ||u[k] - uref[k]||²_R
subject to x[k+1] = A*x[k] + B*u[k]
           x_min ≤ x[k] ≤ x_max
           u_min ≤ u[k] ≤ u_max
```

The hardware implementation decomposes the solver into modular functions:
- `forward_pass` - Forward simulation with constraints
- `backward_pass` - Backward Riccati recursion
- `update_slack` - Slack variable updates
- `update_dual` - Dual variable updates
- `check_termination` - Convergence check

## Performance

Typical performance for quadrotor control (nx=12, nu=4, N=5):
- **Hardware solve time:** 1-5ms
- **Control loop rate:** 100-200Hz
- **Speedup vs software:** 10-50x (problem-dependent)

## License

[Include your license information here]

## Citation

If you use this work, please cite:
[Include your paper citation here]
