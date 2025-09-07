#!/usr/bin/env python3
"""
Async Usage Example for TinyMPC Hardware Driver
Demonstrates non-blocking optimization execution
"""

import sys
import os
import numpy as np
import time

# Add driver to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinympc_hw import tinympc_hw

def simulate_other_work(duration=2.0):
    """Simulate doing other work while optimization runs"""
    print(f"   Doing other work for {duration:.1f}s...")
    start_time = time.time()
    work_count = 0
    
    while time.time() - start_time < duration:
        # Simulate some computation
        _ = np.random.randn(100, 100) @ np.random.randn(100, 100)
        work_count += 1
        time.sleep(0.1)
    
    print(f"   Completed {work_count} work units")

def main():
    print("TinyMPC Hardware Driver - Async Usage Example")
    print("=" * 55)
    
    # Path to bitstream files
    bitstream_path = "../../bitstream/tinympcproj_N5_100Hz_float.bit"
    
    if not os.path.exists(bitstream_path):
        print(f"ERROR: Bitstream file not found: {bitstream_path}")
        print("Please generate bitstream first using: make N=5 FREQ=100")
        return
    
    try:
        # Setup hardware solver
        print("1. Setting up hardware solver...")
        solver = tinympc_hw(bitstream_path)
        solver.setup(max_iter=100, check_termination=10, verbose=True)
        
        print(f"   System: {solver.nx} states, {solver.nu} inputs, N={solver.N}")
        
        # Prepare multiple test problems
        print("\n2. Preparing test problems...")
        
        test_problems = []
        
        # Problem 1: Hover stabilization
        x0_1 = np.zeros(solver.nx)
        x0_1[:3] = [0.2, 0.1, 0.1]  # Larger perturbation
        test_problems.append({
            'name': 'hover_stabilization',
            'x0': x0_1,
            'xref': np.zeros((solver.N, solver.nx)),
            'uref': np.ones((solver.N-1, solver.nu)) * 0.5
        })
        
        # Problem 2: Position tracking
        x0_2 = np.zeros(solver.nx)
        x0_2[:3] = [0.0, 0.0, 0.0]
        xref_2 = np.zeros((solver.N, solver.nx))
        xref_2[:, :3] = [1.0, 1.0, 1.0]  # Move to (1,1,1)
        test_problems.append({
            'name': 'position_tracking',
            'x0': x0_2,
            'xref': xref_2,
            'uref': np.ones((solver.N-1, solver.nu)) * 0.6
        })
        
        # Problem 3: Random disturbance
        x0_3 = np.random.randn(solver.nx) * 0.1
        test_problems.append({
            'name': 'random_disturbance',
            'x0': x0_3,
            'xref': np.zeros((solver.N, solver.nx)),
            'uref': np.ones((solver.N-1, solver.nu)) * 0.5
        })
        
        print(f"   Created {len(test_problems)} test problems")
        
        # Demonstrate async solving
        print("\n3. Async solving demonstration...")
        
        results = []
        
        for i, problem in enumerate(test_problems):
            print(f"\n--- Problem {i+1}: {problem['name']} ---")
            
            # Set problem data
            solver.set_x0(problem['x0'])
            solver.set_x_ref(problem['xref'])
            solver.set_u_ref(problem['uref'])
            
            # Start async solve
            print("   Starting async optimization...")
            start_time = time.time()
            success = solver.solve_async()
            
            if not success:
                print("   ✗ Failed to start async solve")
                continue
            
            # Do other work while optimization runs
            simulate_other_work(duration=1.5)
            
            # Check if done (non-blocking)
            if solver.is_done():
                print("   ✓ Optimization completed while doing other work!")
            else:
                print("   ⏳ Optimization still running, waiting for completion...")
                
            # Wait for completion with timeout
            completed = solver.wait(timeout=10.0)
            total_time = time.time() - start_time
            
            if completed:
                try:
                    result = solver.get_results()
                    results.append(result)
                    
                    print(f"   ✓ Completed successfully in {total_time:.2f}s total")
                    print(f"     Hardware time: {result['solve_time']:.2f} ms")
                    print(f"     Iterations: {result['iter']}")
                    print(f"     Converged: {'Yes' if result['solved'] else 'No'}")
                    print(f"     Final position: {result['x'][-1, :3]}")
                    
                except Exception as e:
                    print(f"   ✗ Failed to get results: {e}")
            else:
                print(f"   ✗ Timeout after {total_time:.2f}s")
        
        # Summary
        print(f"\n4. Summary of {len(results)} successful solves:")
        print("   Problem              | Time (ms) | Converged | Final Pos Error")
        print("   " + "-" * 60)
        
        for i, (problem, result) in enumerate(zip(test_problems[:len(results)], results)):
            pos_error = np.linalg.norm(result['x'][-1, :3])
            conv_str = "Yes" if result['solved'] else "No"
            print(f"   {problem['name']:<20} | {result['solve_time']:>8.2f} | {conv_str:>9} | {pos_error:>13.4f}")
        
        # Demonstrate batch processing
        print(f"\n5. Batch processing demonstration...")
        batch_solve_multiple_problems(solver, test_problems[:2])
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'solver' in locals():
            solver.cleanup()
        print("\n6. Cleanup completed")

def batch_solve_multiple_problems(solver, problems):
    """
    Demonstrate solving multiple problems in sequence with async
    """
    print("   Batch solving multiple problems...")
    
    batch_start = time.time()
    batch_results = []
    
    for i, problem in enumerate(problems):
        # Setup problem
        solver.set_x0(problem['x0'])
        solver.set_x_ref(problem['xref'])
        solver.set_u_ref(problem['uref'])
        
        # Quick async solve
        solver.solve_async()
        
        # Brief work simulation
        time.sleep(0.5)  # Simulate 0.5s of other work
        
        # Get result
        if solver.wait(timeout=5.0):
            result = solver.get_results()
            batch_results.append(result)
            print(f"     Problem {i+1}: {result['solve_time']:.1f}ms")
        else:
            print(f"     Problem {i+1}: TIMEOUT")
    
    batch_time = time.time() - batch_start
    avg_hw_time = np.mean([r['solve_time'] for r in batch_results])
    
    print(f"   Batch completed in {batch_time:.2f}s total")
    print(f"   Average hardware time: {avg_hw_time:.1f}ms")
    print(f"   Efficiency: {avg_hw_time*len(batch_results)/1000/batch_time*100:.1f}% hardware utilization")

if __name__ == "__main__":
    main()