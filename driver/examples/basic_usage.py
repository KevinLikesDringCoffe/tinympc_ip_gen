#!/usr/bin/env python3
"""
Basic Usage Example for TinyMPC Hardware Driver
Demonstrates the simple drop-in replacement for tinympcref
"""

import sys
import os
import numpy as np

# Add driver to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinympc_hw import tinympc_hw

def main():
    print("TinyMPC Hardware Driver - Basic Usage Example")
    print("=" * 50)
    
    # Path to bitstream files (adjust as needed)
    bitstream_path = "../../bitstream/tinympcproj_N5_100Hz_float.bit"
    hwh_path = "../../bitstream/tinympcproj_N5_100Hz_float.hwh"
    
    # Check if bitstream files exist
    if not os.path.exists(bitstream_path):
        print(f"ERROR: Bitstream file not found: {bitstream_path}")
        print("Please generate bitstream first using: make N=5 FREQ=100")
        return
    
    try:
        # Create hardware solver (similar to tinympcref)
        print("1. Creating hardware solver...")
        solver = tinympc_hw()
        
        # Load bitstream to FPGA
        print("2. Loading bitstream to FPGA...")
        solver.load_bitstream(bitstream_path, hwh_path)
        
        # Setup solver parameters (only algorithm params are used)
        print("3. Setting up solver parameters...")
        solver.setup(
            max_iter=100,           # Maximum ADMM iterations
            check_termination=10,   # Check convergence every 10 iterations
            verbose=True           # Enable verbose logging
        )
        
        # Print solver information
        solver.print_stats()
        
        # Create test data
        print("4. Creating test data...")
        
        # Initial state: small perturbation from hover
        x0 = np.zeros(solver.nx)
        x0[:3] = [0.1, 0.1, 0.05]  # Small position perturbation
        print(f"   Initial state: {x0}")
        
        # Reference trajectory: hover at origin
        xref = np.zeros((solver.N, solver.nx))
        print(f"   Reference trajectory: shape {xref.shape}")
        
        # Reference inputs: hover thrust
        uref = np.zeros((solver.N-1, solver.nu))
        uref[:, 0] = 0.5  # Hover thrust for quadrotor
        print(f"   Reference inputs: shape {uref.shape}")
        
        # Set problem data
        print("5. Setting problem data...")
        solver.set_x0(x0)
        solver.set_x_ref(xref)
        solver.set_u_ref(uref)
        
        # Solve optimization problem
        print("6. Solving optimization problem...")
        success = solver.solve(timeout=5.0)
        
        if success:
            print(f"\n✓ Optimization successful!")
            print(f"   Solve time: {solver.solve_time:.2f} ms")
            print(f"   Iterations: {solver.iter}")
            print(f"   Converged: {'Yes' if solver.solved else 'No'}")
            
            # Print results
            print(f"\n7. Results:")
            print(f"   Final state: {solver.x[-1]}")
            print(f"   First control: {solver.u[0]}")
            print(f"   Control trajectory shape: {solver.u.shape}")
            print(f"   State trajectory shape: {solver.x.shape}")
            
            # Compute some metrics
            position_error = np.linalg.norm(solver.x[-1, :3])  # Final position error
            control_effort = np.mean(np.linalg.norm(solver.u, axis=1))
            
            print(f"\n   Performance metrics:")
            print(f"   Final position error: {position_error:.4f}")
            print(f"   Average control effort: {control_effort:.4f}")
            
        else:
            print("\n✗ Optimization failed or timed out")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup (automatically handled by context manager or destructor)
        if 'solver' in locals():
            solver.cleanup()
        print("\n8. Cleanup completed")

def demonstrate_compatibility():
    """
    Demonstrate drop-in compatibility with tinympcref interface
    """
    print("\n" + "=" * 50)
    print("Compatibility Demonstration")
    print("=" * 50)
    
    # This code would work identically with tinympcref
    
    bitstream_path = "../../bitstream/tinympcproj_N5_100Hz_float.bit"
    
    if not os.path.exists(bitstream_path):
        print("Skipping compatibility demo - bitstream not found")
        return
    
    # Option 1: Replace tinympcref import
    # from tinympcref import tinympcref as solver_class
    solver_class = tinympc_hw  # Use hardware instead
    
    # Option 2: Use with context manager
    with solver_class(bitstream_path) as solver:
        # Setup (matrices are ignored for hardware)
        solver.setup(max_iter=50, verbose=False)
        
        # Set problem
        x0 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * (solver.nx-6))
        solver.set_x0(x0)
        solver.set_x_ref(np.zeros((solver.N, solver.nx)))
        solver.set_u_ref(np.ones((solver.N-1, solver.nu)) * 0.5)
        
        # Solve
        solver.solve()
        
        # Access results (same as tinympcref)
        print(f"Hardware result - converged: {solver.solved}")
        print(f"Hardware result - iterations: {solver.iter}")
        print(f"Hardware result - solve time: {solver.solve_time:.2f} ms")

if __name__ == "__main__":
    main()
    demonstrate_compatibility()