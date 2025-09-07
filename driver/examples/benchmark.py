#!/usr/bin/env python3
"""
Performance Benchmark for TinyMPC Hardware Driver
Compares hardware vs software performance and measures throughput
"""

import sys
import os
import numpy as np
import time

# Add driver to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinympc_hw import tinympc_hw

# Try to import software reference for comparison
try:
    sys.path.append("../../")
    from tinympcref import tinympcref
    HAS_SOFTWARE = True
except ImportError:
    HAS_SOFTWARE = False
    print("Warning: tinympcref not available, skipping software comparison")

def create_test_problems(N, nx, nu, num_problems=10):
    """Create a set of test problems for benchmarking"""
    problems = []
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(num_problems):
        # Random initial states (small perturbations)
        x0 = np.random.randn(nx) * 0.1
        
        # Random reference trajectories
        xref = np.random.randn(N, nx) * 0.2
        
        # Random reference inputs with hover bias
        uref = np.random.randn(N-1, nu) * 0.1
        if nu >= 1:
            uref[:, 0] += 0.5  # Add hover thrust bias
        
        problems.append({
            'name': f'random_problem_{i+1}',
            'x0': x0.astype(np.float32),
            'xref': xref.astype(np.float32),  
            'uref': uref.astype(np.float32)
        })
    
    return problems

def benchmark_hardware_solver(solver, problems, iterations=1):
    """Benchmark hardware solver performance"""
    print(f"Benchmarking hardware solver ({len(problems)} problems, {iterations} iterations)...")
    
    times = []
    results = []
    
    for iteration in range(iterations):
        iteration_times = []
        
        for i, problem in enumerate(problems):
            # Setup problem
            solver.set_x0(problem['x0'])
            solver.set_x_ref(problem['xref'])  
            solver.set_u_ref(problem['uref'])
            
            # Time the solve
            start_time = time.time()
            success = solver.solve(timeout=10.0)
            end_time = time.time()
            
            if success:
                solve_time = solver.solve_time  # Hardware time
                total_time = (end_time - start_time) * 1000  # Total time including overhead
                iteration_times.append({
                    'hw_time': solve_time,
                    'total_time': total_time,
                    'success': True
                })
            else:
                iteration_times.append({
                    'hw_time': 0,
                    'total_time': 0,
                    'success': False
                })
                print(f"   Problem {i+1} failed")
        
        times.append(iteration_times)
        
        # Store results from first iteration
        if iteration == 0:
            for problem_times in iteration_times:
                if problem_times['success']:
                    results.append({
                        'x': solver.x.copy(),
                        'u': solver.u.copy(),
                        'solved': solver.solved,
                        'iter': solver.iter
                    })
    
    return times, results

def benchmark_software_solver(problems, N, nx, nu, iterations=1):
    """Benchmark software solver performance (if available)"""
    if not HAS_SOFTWARE:
        return None, None
    
    print(f"Benchmarking software solver ({len(problems)} problems, {iterations} iterations)...")
    
    # Create software solver with dummy system matrices
    solver = tinympcref()
    
    # Create dummy system (quadrotor-like)
    A = np.eye(nx) + np.random.randn(nx, nx) * 0.01
    B = np.random.randn(nx, nu) * 0.1
    Q = np.eye(nx) * 0.1
    R = np.eye(nu) * 0.1
    
    solver.setup(A, B, Q, R, N, max_iter=100, check_termination=10)
    
    times = []
    results = []
    
    for iteration in range(iterations):
        iteration_times = []
        
        for i, problem in enumerate(problems):
            # Setup problem
            solver.set_x0(problem['x0'])
            solver.set_x_ref(problem['xref'])
            solver.set_u_ref(problem['uref'])
            
            # Time the solve
            start_time = time.time()
            solver.solve()
            end_time = time.time()
            
            solve_time = (end_time - start_time) * 1000
            iteration_times.append({
                'sw_time': solve_time,
                'success': solver.solved == 1
            })
        
        times.append(iteration_times)
        
        # Store results from first iteration
        if iteration == 0:
            for _ in range(len(problems)):
                results.append({
                    'x': solver.x.copy() if hasattr(solver, 'x') else None,
                    'u': solver.u.copy() if hasattr(solver, 'u') else None,
                    'solved': solver.solved,
                    'iter': solver.iter
                })
    
    return times, results

def benchmark_async_throughput(solver, problems):
    """Benchmark async throughput performance"""
    print(f"Benchmarking async throughput ({len(problems)} problems)...")
    
    start_time = time.time()
    completed_count = 0
    
    for problem in problems:
        # Setup problem
        solver.set_x0(problem['x0'])
        solver.set_x_ref(problem['xref'])
        solver.set_u_ref(problem['uref'])
        
        # Start async solve
        if solver.solve_async():
            # Brief wait to simulate other work
            time.sleep(0.1)  # 100ms of "other work"
            
            # Wait for completion
            if solver.wait(timeout=5.0):
                completed_count += 1
            
    total_time = time.time() - start_time
    throughput = completed_count / total_time
    
    return {
        'total_time': total_time,
        'completed': completed_count, 
        'throughput': throughput
    }

def print_benchmark_results(hw_times, sw_times=None, hw_results=None, sw_results=None):
    """Print formatted benchmark results"""
    
    # Hardware statistics
    if hw_times:
        hw_hw_times = []
        hw_total_times = []
        hw_success_count = 0
        
        for iteration in hw_times:
            for problem in iteration:
                if problem['success']:
                    hw_hw_times.append(problem['hw_time'])
                    hw_total_times.append(problem['total_time'])
                    hw_success_count += 1
        
        print(f"\n=== Hardware Solver Results ===")
        print(f"Success rate: {hw_success_count}/{len(hw_times)*len(hw_times[0])} ({hw_success_count/(len(hw_times)*len(hw_times[0]))*100:.1f}%)")
        if hw_hw_times:
            print(f"Hardware time: {np.mean(hw_hw_times):.2f} ± {np.std(hw_hw_times):.2f} ms")
            print(f"Total time:    {np.mean(hw_total_times):.2f} ± {np.std(hw_total_times):.2f} ms") 
            print(f"Overhead:      {np.mean(hw_total_times) - np.mean(hw_hw_times):.2f} ms")
            print(f"Min/Max HW:    {np.min(hw_hw_times):.2f} / {np.max(hw_hw_times):.2f} ms")
    
    # Software statistics
    if sw_times:
        sw_times_flat = []
        sw_success_count = 0
        
        for iteration in sw_times:
            for problem in iteration:
                if problem['success']:
                    sw_times_flat.append(problem['sw_time'])
                    sw_success_count += 1
        
        print(f"\n=== Software Solver Results ===")
        print(f"Success rate: {sw_success_count}/{len(sw_times)*len(sw_times[0])} ({sw_success_count/(len(sw_times)*len(sw_times[0]))*100:.1f}%)")
        if sw_times_flat:
            print(f"Software time: {np.mean(sw_times_flat):.2f} ± {np.std(sw_times_flat):.2f} ms")
            print(f"Min/Max SW:    {np.min(sw_times_flat):.2f} / {np.max(sw_times_flat):.2f} ms")
    
    # Comparison
    if hw_times and sw_times and hw_hw_times and sw_times_flat:
        speedup = np.mean(sw_times_flat) / np.mean(hw_hw_times)
        print(f"\n=== Performance Comparison ===")
        print(f"Speedup (SW/HW): {speedup:.2f}x")
        print(f"Hardware is {speedup:.1f}x faster than software")
    
    # Result quality comparison (if both available)
    if hw_results and sw_results and len(hw_results) == len(sw_results):
        print(f"\n=== Result Quality Comparison ===")
        differences = []
        
        for hw_res, sw_res in zip(hw_results[:3], sw_results[:3]):  # Compare first 3 results
            if hw_res and sw_res and hw_res.get('x') is not None and sw_res.get('x') is not None:
                # Compare final states
                diff = np.linalg.norm(hw_res['x'][-1] - sw_res['x'][-1])
                differences.append(diff)
        
        if differences:
            print(f"Final state differences: {np.mean(differences):.6f} ± {np.std(differences):.6f}")
            print(f"Max difference: {np.max(differences):.6f}")

def main():
    print("TinyMPC Hardware Driver - Performance Benchmark")
    print("=" * 60)
    
    # Configuration
    bitstream_path = "../../bitstream/tinympcproj_N5_100Hz_float.bit"
    num_test_problems = 20
    num_iterations = 3
    
    if not os.path.exists(bitstream_path):
        print(f"ERROR: Bitstream file not found: {bitstream_path}")
        print("Please generate bitstream first using: make N=5 FREQ=100")
        return
    
    try:
        # Setup hardware solver
        print("1. Setting up hardware solver...")
        hw_solver = tinympc_hw(bitstream_path)
        hw_solver.setup(max_iter=100, check_termination=10, verbose=False)
        
        N, nx, nu = hw_solver.N, hw_solver.nx, hw_solver.nu
        print(f"   System: N={N}, nx={nx}, nu={nu}")
        
        # Create test problems
        print(f"2. Creating {num_test_problems} test problems...")
        problems = create_test_problems(N, nx, nu, num_test_problems)
        
        # Benchmark hardware solver
        print(f"3. Running hardware benchmark...")
        hw_times, hw_results = benchmark_hardware_solver(hw_solver, problems, num_iterations)
        
        # Benchmark software solver (if available)
        sw_times, sw_results = None, None
        if HAS_SOFTWARE:
            print(f"4. Running software benchmark...")
            sw_times, sw_results = benchmark_software_solver(problems, N, nx, nu, num_iterations)
        else:
            print("4. Skipping software benchmark (tinympcref not available)")
        
        # Benchmark async throughput
        print(f"5. Running async throughput benchmark...")
        async_results = benchmark_async_throughput(hw_solver, problems[:10])
        
        # Print results
        print_benchmark_results(hw_times, sw_times, hw_results, sw_results)
        
        print(f"\n=== Async Throughput Results ===")
        print(f"Total time: {async_results['total_time']:.2f} s")
        print(f"Completed: {async_results['completed']}/10 problems")
        print(f"Throughput: {async_results['throughput']:.2f} problems/second")
        
        print(f"\n=== System Information ===")
        info = hw_solver.get_info()
        print(f"Driver type: {info['type']}")
        print(f"Control frequency: {info['control_freq']} Hz")
        print(f"Memory size: {info['memory_info']['memory_size']} elements")
        print(f"Memory bandwidth estimate: {info['memory_info']['memory_size'] * 4 / (np.mean([t['hw_time'] for iteration in hw_times for t in iteration if t['success']]) / 1000) / 1e6:.1f} MB/s")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'hw_solver' in locals():
            hw_solver.cleanup()
        print("\n6. Benchmark completed")

if __name__ == "__main__":
    main()