#!/usr/bin/env python3
"""
Unit test for Backward Pass subfunction
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparse_codegen.subfunction_generators import BackwardPassGenerator


class TestBackwardPass(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters"""
        self.nx = 2
        self.nu = 1
        self.N = 5
    
    def test_backward_pass_structure(self):
        """Test backward pass operation structure"""
        gen = BackwardPassGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=False)
        self.assertGreater(len(ops), 0)
        
        # Should have comment
        self.assertTrue(any(op.get('type') == 'comment' for op in ops))
        
        # Should have backward loop
        loops = [op for op in ops if op.get('type') == 'loop']
        self.assertEqual(len(loops), 1)
        
        # Check loop direction (backward)
        loop = loops[0]
        self.assertEqual(loop['start'], self.N - 2)
        self.assertEqual(loop['end'], -1)
        self.assertEqual(loop['step'], -1)
    
    def test_unrolled_backward_pass(self):
        """Test unrolled backward pass"""
        gen = BackwardPassGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=True)
        self.assertGreater(len(ops), 0)
        
        # Should have operations for each backward step
        comments = [op for op in ops if op.get('type') == 'comment']
        iteration_comments = [c for c in comments if 'iteration k=' in c.get('text', '')]
        
        # Should have N-2 iterations (k = N-2 down to 0)
        self.assertEqual(len(iteration_comments), self.N - 1)
    
    def test_backward_operations(self):
        """Test that backward pass has correct operations"""
        gen = BackwardPassGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=True)
        
        # Count matrix-vector operations (sparse_matvec)
        matvec_ops = [op for op in ops if op.get('type') in ['matvec', 'sparse_matvec']]
        
        # Each iteration should have:
        # 1. B.T @ p[k+1] 
        # 2. Quu_inv @ temp_quu_input (d[k] computation)
        # 3. AmBKt @ p[k+1]
        # 4. Kinf.T @ r[k]
        # Total: 4 * (N-1) matrix operations
        expected_matvec = 4 * (self.N - 1)
        self.assertEqual(len(matvec_ops), expected_matvec)
        
        # Check matrix names used
        matrix_names = [op.get('matrix') for op in matvec_ops]
        required_matrices = ['B_T', 'Quu_inv', 'AmBKt', 'Kinf_T']
        
        for matrix in required_matrices:
            self.assertIn(matrix, matrix_names)
    
    def test_assignment_structure(self):
        """Test assignment operations in backward pass"""
        gen = BackwardPassGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=True)
        assignments = [op for op in ops if op.get('type') == 'assignment']
        
        # Each iteration updates:
        # - temp_quu_input (nu elements)
        # - p[k] (nx elements)  
        # Total per iteration: nu + nx
        expected_assignments = (self.nu + self.nx) * (self.N - 1)
        self.assertGreaterEqual(len(assignments), expected_assignments)
        
        # Check target variables
        targets = [op.get('target') for op in assignments]
        self.assertIn('temp_quu_input', targets)
        self.assertIn('p', targets)
    
    def test_numerical_correctness(self):
        """Test backward pass numerical accuracy"""
        from subfunction_validator import SubfunctionValidator
        
        gen = BackwardPassGenerator(self.nx, self.nu, self.N)
        validator = SubfunctionValidator()
        
        # Generate test system matrices
        A = np.array([[1.0, 0.1], [0.0, 0.95]], dtype=np.float32)
        B = np.array([[0.5], [0.5]], dtype=np.float32)
        Q = np.diag([1.0, 0.1]).astype(np.float32)
        R = np.array([[0.1]], dtype=np.float32)
        
        # Generate test data
        q = np.random.rand(self.N, self.nx).astype(np.float32) * 0.1
        r = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        p_terminal = np.random.rand(self.nx).astype(np.float32) * 0.1
        
        # Test both modes
        for unroll in [False, True]:
            with self.subTest(unroll=unroll):
                ops = gen.generate_operations(unroll=unroll)
                
                # Numerical validation
                passed, details = validator.validate_backward_pass(
                    A, B, Q, R, q, r, p_terminal, ops=ops, tolerance=1e-4
                )
                
                self.assertTrue(passed, f"Backward pass numerical validation failed with unroll={unroll}: {details}")
        
        validator.cleanup()
    
    def test_loop_body_structure(self):
        """Test structure of backward pass loop body"""
        gen = BackwardPassGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=False)
        
        # Find the loop
        loop_op = next(op for op in ops if op.get('type') == 'loop')
        body = loop_op['body']
        
        self.assertGreater(len(body), 0)
        
        # Check that body has required operations
        body_types = [op.get('type') for op in body]
        self.assertIn('sparse_matvec', body_types)
        self.assertIn('element_wise', body_types)


if __name__ == '__main__':
    unittest.main()