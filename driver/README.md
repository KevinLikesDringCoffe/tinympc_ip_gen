# TinyMPC Hardware Driver for Ultra96

A high-performance hardware driver for TinyMPC optimization on Ultra96 FPGA boards. Provides a drop-in replacement for the software `tinympcref` with identical API but FPGA acceleration.

## Features

- **Drop-in Replacement**: Compatible with `tinympcref.py` interface
- **FPGA Acceleration**: Hardware-accelerated optimization on Ultra96  
- **Dual Execution Modes**: Both synchronous and asynchronous solving
- **Auto Configuration**: Automatically detects parameters from bitstream filename
- **Memory Management**: Efficient physical memory allocation and management
- **Performance Monitoring**: Built-in timing and throughput measurement

## Installation

### Prerequisites

- Ultra96 board with PYNQ Linux
- PYNQ Python package
- Generated TinyMPC bitstream files

### Setup

1. Copy the driver directory to your Ultra96:
   ```bash
   scp -r driver/ xilinx@<ultra96-ip>:~/
   ```

2. On Ultra96, install any missing dependencies:
   ```bash
   pip install numpy
   ```

## Quick Start

### Basic Usage

```python
from driver import tinympc_hw

# Load bitstream and create solver  
solver = tinympc_hw("bitstream/tinympcproj_N5_100Hz_float.bit")

# Setup (only algorithm parameters are used)
solver.setup(max_iter=100, check_termination=10)

# Set problem data
solver.set_x0([0.1, 0.1, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Initial state
solver.set_x_ref(np.zeros((5, 12)))  # Reference trajectory  
solver.set_u_ref(np.ones((4, 4)) * 0.5)  # Reference inputs

# Solve optimization
solver.solve()

# Access results
print(f"Solved in {solver.solve_time:.2f}ms")
print(f"Optimal trajectory: {solver.x}")
print(f"Optimal controls: {solver.u}")
```

### Async Usage

```python
# Start optimization asynchronously
solver.solve_async()

# Do other work while optimization runs
time.sleep(1.0)  # Simulate other computation

# Wait for completion and get results
if solver.wait(timeout=5.0):
    results = solver.get_results()
    print(f"Async solve completed: {results['solve_time']:.2f}ms")
```

### Drop-in Replacement

```python
# Replace this:
# from tinympcref import tinympcref as Solver

# With this:
from driver import tinympc_hw as Solver

# Rest of code remains identical!
solver = Solver("bitstream/tinympcproj_N5_100Hz_float.bit") 
solver.setup(A, B, Q, R, N)  # A,B,Q,R ignored but compatible
solver.set_x0(x0)
solver.solve()
```

## API Reference

### Main Class: `tinympc_hw`

#### Constructor
```python
tinympc_hw(bitstream_path=None, hwh_path=None)
```

#### Key Methods

**Hardware Setup**
- `load_bitstream(bitstream_path, hwh_path=None)` - Load FPGA bitstream
- `setup(**kwargs)` - Configure solver parameters
- `cleanup()` - Release hardware resources

**Problem Setup** (compatible with tinympcref)
- `set_x0(x0)` - Set initial state
- `set_x_ref(xref)` - Set reference trajectory  
- `set_u_ref(uref)` - Set reference inputs

**Solving**
- `solve(timeout=10.0)` - Synchronous solve
- `solve_async()` - Start async solve
- `wait(timeout=10.0)` - Wait for async completion
- `is_done()` - Check if async solve is complete
- `get_results()` - Get async solve results

**Status & Info**
- `get_info()` - Get solver information
- `print_stats()` - Print solver statistics

#### Properties (compatible with tinympcref)
- `x` - Optimal state trajectory (N, nx)
- `u` - Optimal control inputs (N-1, nu)  
- `iter` - Number of iterations completed
- `solved` - Convergence flag (1 if converged)
- `solve_time` - Hardware execution time (ms)

## Memory Layout

The driver automatically manages memory layout according to the hardware IP specification:

```
Memory Buffer Layout:
[0:nx]                    - x0 (initial state)
[nx:nx+N*nx]              - xref (reference trajectory)  
[nx+N*nx:...]             - uref (reference inputs)
[...:...+N*nx]           - x_out (output states)
[...:...+(N-1)*nu]       - u_out (output controls)
```

## Parameter Auto-Detection

The driver automatically detects system parameters from bitstream filename:

- **Pattern**: `tinympcproj_N<N>_<freq>Hz_float.bit`
- **Example**: `tinympcproj_N5_100Hz_float.bit` → N=5, freq=100Hz
- **System**: Assumes quadrotor dynamics (nx=12, nu=4)

## Examples

### Basic Usage
```bash
cd driver/examples
python basic_usage.py
```

### Async Execution
```bash
python async_example.py
```

### Performance Benchmark
```bash
python benchmark.py
```

## Performance Notes

- **Hardware Time**: Pure FPGA computation time
- **Total Time**: Includes memory transfers and Python overhead
- **Typical Performance**: 1-5ms hardware time for N=5 problems
- **Async Benefit**: ~90% hardware utilization with proper pipelining

## Troubleshooting

### Common Issues

1. **Bitstream not found**
   - Ensure bitstream files are generated: `make N=5 FREQ=100`
   - Check file paths are correct

2. **Parameter detection failed**
   - Use standard filename format: `tinympcproj_N<N>_<freq>Hz_float.bit`
   - Or call `setup()` with explicit parameters

3. **Memory allocation failed**
   - Ensure sufficient CMA memory available
   - Try smaller problem sizes

4. **PYNQ not found**
   - Install PYNQ: `pip install pynq`
   - Ensure running on PYNQ-supported board

### Debug Mode

Enable verbose logging for debugging:
```python
solver.setup(verbose=True)
```

This enables detailed logging of:
- Memory layout and transfers
- Hardware register operations  
- Timing information
- Error conditions

## Hardware Requirements

- **Board**: Ultra96 or compatible PYNQ board
- **FPGA**: Zynq UltraScale+ (xczu3eg)
- **Memory**: Minimum 512MB RAM
- **OS**: PYNQ Linux 2.6+

## Bitstream Compatibility

The driver works with bitstreams generated by the TinyMPC build system:

- **Supported**: Float precision (32-bit)
- **Interface**: AXI-Lite control + AXI Master memory
- **IP Name**: Must contain 'tinympc' in name
- **Parameters**: N, nx, nu hardcoded in bitstream

## License

This driver is part of the TinyMPC project. See main project for license terms.