"""
Test Data Generator for TinyMPC HLS Testing
Generates golden reference data and test vectors
"""

import numpy as np
import os
from typing import Dict, Any, List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tinympcref import TinyMPCRef


class TestDataGenerator:
    """Generate test data for HLS validation"""
    
    def __init__(self, nx: int, nu: int, N: int):
        self.nx = nx
        self.nu = nu
        self.N = N
        
    def generate_random_system(self, sparsity: float = 0.3, seed: int = 42) -> Dict[str, np.ndarray]:
        """Generate random stable system matrices with specified sparsity"""
        np.random.seed(seed)
        
        # Generate stable A matrix
        A = np.random.randn(self.nx, self.nx) * 0.1
        A = A + np.eye(self.nx) * 0.9  # Make diagonally dominant for stability
        
        # Apply sparsity
        mask = np.random.random((self.nx, self.nx)) < sparsity
        A = A * mask
        
        # Ensure eigenvalues are stable
        eigenvals = np.linalg.eigvals(A)
        if np.max(np.abs(eigenvals)) >= 1.0:
            A = A * 0.8  # Scale to ensure stability
            
        # Generate B matrix
        B = np.random.randn(self.nx, self.nu) * 0.5
        mask = np.random.random((self.nx, self.nu)) < sparsity
        B = B * mask
        
        # Generate diagonal cost matrices
        Q = np.diag(np.random.uniform(0.1, 2.0, self.nx))
        R = np.diag(np.random.uniform(0.1, 2.0, self.nu))
        
        return {
            'A': A.astype(np.float32),
            'B': B.astype(np.float32),
            'Q': Q.astype(np.float32),
            'R': R.astype(np.float32)
        }
    
    def generate_quadrotor_system(self) -> Dict[str, np.ndarray]:
        """Generate realistic quadrotor system matrices"""
        if self.nx != 12 or self.nu != 4:
            raise ValueError("Quadrotor system requires nx=12, nu=4")
            
        dt = 0.02  # 50 Hz control rate
        g = 9.81
        
        # State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        # Control: [thrust, roll_rate, pitch_rate, yaw_rate]
        
        A = np.eye(12)
        A[0, 3] = dt   # x += vx * dt
        A[1, 4] = dt   # y += vy * dt
        A[2, 5] = dt   # z += vz * dt
        A[6, 9] = dt   # phi += p * dt
        A[7, 10] = dt  # theta += q * dt
        A[8, 11] = dt  # psi += r * dt
        
        B = np.zeros((12, 4))
        B[5, 0] = dt   # vz += thrust * dt
        B[9, 1] = dt   # p += roll_rate * dt
        B[10, 2] = dt  # q += pitch_rate * dt
        B[11, 3] = dt  # r += yaw_rate * dt
        
        # Cost matrices - penalize position and velocity more
        Q = np.diag([10, 10, 10, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        R = np.diag([0.1, 0.1, 0.1, 0.1])
        
        return {
            'A': A.astype(np.float32),
            'B': B.astype(np.float32), 
            'Q': Q.astype(np.float32),
            'R': R.astype(np.float32)
        }
    
    def generate_test_trajectories(self, num_tests: int = 5) -> List[Dict[str, np.ndarray]]:
        """Generate multiple test cases with different trajectories"""
        test_cases = []
        
        for i in range(num_tests):
            # Random initial state
            x0 = np.random.randn(self.nx) * 0.5
            
            # Generate reference trajectory
            if i == 0:
                # Constant reference (hovering)
                Xref = np.zeros((self.N, self.nx))
                Uref = np.zeros((self.N - 1, self.nu))
            elif i == 1:
                # Linear trajectory
                Xref = np.linspace(x0, np.random.randn(self.nx), self.N)
                Uref = np.random.randn(self.N - 1, self.nu) * 0.1
            else:
                # Sinusoidal trajectory
                t = np.linspace(0, 2*np.pi, self.N)
                Xref = np.zeros((self.N, self.nx))
                for j in range(min(3, self.nx)):  # First 3 states follow sine wave
                    Xref[:, j] = np.sin(t + j * np.pi/3)
                Uref = np.random.randn(self.N - 1, self.nu) * 0.1
                
            test_cases.append({
                'x0': x0.astype(np.float32),
                'Xref': Xref.astype(np.float32),
                'Uref': Uref.astype(np.float32),
                'test_name': f'trajectory_{i}'
            })
            
        return test_cases
    
    def generate_golden_reference(self, system: Dict[str, np.ndarray], 
                                 test_case: Dict[str, np.ndarray],
                                 rho: float = 1.0,
                                 max_iter: int = 100) -> Dict[str, Any]:
        """Generate golden reference using Python implementation"""
        
        # Create and setup solver
        solver = TinyMPCRef()
        solver.setup(
            A=system['A'], B=system['B'], 
            Q=system['Q'], R=system['R'],
            N=self.N, rho=rho,
            max_iter=max_iter
        )
        
        # Set test case
        solver.set_x0(test_case['x0'])
        solver.set_x_ref(test_case['Xref'])
        solver.set_u_ref(test_case['Uref'])
        
        # Collect step-by-step results
        step_results = {}
        
        # Initial state
        step_results['initial'] = {
            'x': solver.x.copy(),
            'u': solver.u.copy(),
            'iteration': 0
        }
        
        # Run solver with step tracking
        solver.iter = 0
        solver.solved = 0
        
        for iteration in range(max_iter):
            # Forward pass
            forward_result = solver.forward_pass_step()
            
            # Slack update
            slack_result = solver.slack_update_step()
            
            # Dual update  
            dual_result = solver.dual_update_step()
            
            # Linear cost update
            cost_result = solver.linear_cost_step()
            
            # Check termination
            term_result = solver.termination_check_step()
            
            # Store iteration results
            step_results[f'iter_{iteration}'] = {
                'forward_pass': forward_result,
                'slack_update': slack_result,
                'dual_update': dual_result,
                'linear_cost': cost_result,
                'termination': term_result,
                'iteration': iteration + 1
            }
            
            if term_result['terminated']:
                solver.solved = 1
                break
                
            # Update for next iteration
            solver.v = solver.vnew.copy()
            solver.z = solver.znew.copy()
            
            # Backward pass
            backward_result = solver.backward_pass_step()
            step_results[f'iter_{iteration}']['backward_pass'] = backward_result
            
        # Final results
        final_result = {
            'x_optimal': solver.x.copy(),
            'u_optimal': solver.u.copy(),
            'converged': solver.solved,
            'iterations': solver.iter,
            'algorithm_data': solver.get_algorithm_data(),
            'step_by_step': step_results
        }
        
        return final_result
    
    def save_test_data(self, output_dir: str, system: Dict[str, np.ndarray],
                      test_cases: List[Dict[str, np.ndarray]],
                      rho: float = 1.0, max_iter: int = 100):
        """Save test data in format suitable for HLS testbench"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save system matrices
        np.save(os.path.join(output_dir, 'system_A.npy'), system['A'])
        np.save(os.path.join(output_dir, 'system_B.npy'), system['B'])
        np.save(os.path.join(output_dir, 'system_Q.npy'), system['Q'])
        np.save(os.path.join(output_dir, 'system_R.npy'), system['R'])
        
        # Generate and save golden reference for each test case
        for i, test_case in enumerate(test_cases):
            print(f"Generating golden reference for test case {i}...")
            
            golden = self.generate_golden_reference(system, test_case, rho, max_iter)
            
            test_dir = os.path.join(output_dir, f"test_{i}")
            os.makedirs(test_dir, exist_ok=True)
            
            # Save input data
            np.save(os.path.join(test_dir, 'x0.npy'), test_case['x0'])
            np.save(os.path.join(test_dir, 'Xref.npy'), test_case['Xref'])
            np.save(os.path.join(test_dir, 'Uref.npy'), test_case['Uref'])
            
            # Save golden outputs
            np.save(os.path.join(test_dir, 'x_optimal.npy'), golden['x_optimal'])
            np.save(os.path.join(test_dir, 'u_optimal.npy'), golden['u_optimal'])
            
            # Save algorithm data for detailed validation
            np.savez(os.path.join(test_dir, 'algorithm_data.npz'), **golden['algorithm_data'])
            
            # Save in text format for C++ testbench
            self._save_as_text(test_dir, test_case, golden)
            
        print(f"Test data saved to {output_dir}")
        
    def _save_as_text(self, test_dir: str, test_case: Dict[str, np.ndarray], 
                     golden: Dict[str, Any]):
        """Save test data in text format for C++ testbench"""
        
        def save_array_txt(filename: str, array: np.ndarray):
            with open(os.path.join(test_dir, filename), 'w') as f:
                if array.ndim == 1:
                    for val in array:
                        f.write(f"{val:.6f}\n")
                else:
                    for row in array:
                        for val in row:
                            f.write(f"{val:.6f} ")
                        f.write("\n")
        
        # Input files
        save_array_txt('x0_input.txt', test_case['x0'])
        save_array_txt('Xref_input.txt', test_case['Xref'])
        save_array_txt('Uref_input.txt', test_case['Uref'])
        
        # Golden output files
        save_array_txt('x_golden_output.txt', golden['x_optimal'])
        save_array_txt('u_golden_output.txt', golden['u_optimal'])
        
        # Test metadata
        with open(os.path.join(test_dir, 'test_info.txt'), 'w') as f:
            f.write(f"nx={self.nx}\n")
            f.write(f"nu={self.nu}\n")
            f.write(f"N={self.N}\n")
            f.write(f"converged={golden['converged']}\n")
            f.write(f"iterations={golden['iterations']}\n")


def generate_example_data():
    """Generate example test data for common system configurations"""
    
    # Generate test data for different system sizes
    configs = [
        {'nx': 4, 'nu': 2, 'N': 10},   # Small system
        {'nx': 6, 'nu': 3, 'N': 15},   # Medium system
        {'nx': 12, 'nu': 4, 'N': 20},  # Quadrotor system
    ]
    
    for config in configs:
        print(f"Generating data for nx={config['nx']}, nu={config['nu']}, N={config['N']}")
        
        generator = TestDataGenerator(config['nx'], config['nu'], config['N'])
        
        # Generate system
        if config['nx'] == 12 and config['nu'] == 4:
            system = generator.generate_quadrotor_system()
        else:
            system = generator.generate_random_system()
            
        # Generate test cases
        test_cases = generator.generate_test_trajectories(num_tests=3)
        
        # Save data
        output_dir = f"test_data_nx{config['nx']}_nu{config['nu']}_N{config['N']}"
        generator.save_test_data(output_dir, system, test_cases)


if __name__ == "__main__":
    generate_example_data()