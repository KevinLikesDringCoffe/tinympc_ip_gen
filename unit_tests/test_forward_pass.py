#!/usr/bin/env python3
"""
Unit test for Forward Pass subfunction
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparse_codegen.subfunction_generators import ForwardPassGenerator
from sparse_codegen.code_validator import validate_code
from sparse_codegen import TinyMPCGenerator, TinyMPCConfig


class TestForwardPass(unittest.TestCase):
    
    def setUp(self):
        """Set up test system"""
        self.A = np.array([[1.0, 0.1], [0.0, 0.95]], dtype=np.float32)
        self.B = np.array([[0.5], [0.5]], dtype=np.float32)
        self.Q = np.diag([1.0, 0.1]).astype(np.float32)
        self.R = np.array([[0.1]], dtype=np.float32)
        self.N = 5
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
        
        # Compute Kinf
        generator = TinyMPCGenerator(self.A, self.B, self.Q, self.R, self.N)
        self.Kinf = generator.Kinf
    
    def test_forward_pass_structure(self):
        """Test forward pass operation structure"""
        gen = ForwardPassGenerator(self.nx, self.nu, self.N)
        
        # Test unrolled
        ops_unrolled = gen.generate_operations(unroll=True)
        self.assertGreater(len(ops_unrolled), 0)
        self.assertTrue(any('Forward pass iteration' in str(op) for op in ops_unrolled))
        
        # Test looped
        ops_looped = gen.generate_operations(unroll=False)
        self.assertGreater(len(ops_looped), 0)
        self.assertTrue(any(op.get('type') == 'loop' for op in ops_looped))
    
    def test_forward_pass_validation(self):
        """Test forward pass numerical accuracy"""
        from subfunction_validator import SubfunctionValidator
        
        gen = ForwardPassGenerator(self.nx, self.nu, self.N)
        validator = SubfunctionValidator()
        
        # Generate test data
        x0 = np.random.rand(self.nx).astype(np.float32)
        d = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        
        # Test both modes 
        for unroll in [False, True]:
            with self.subTest(unroll=unroll):
                ops = gen.generate_operations(unroll=unroll)
                
                # Numerical validation
                passed, details = validator.validate_forward_pass(
                    self.A, self.B, self.Kinf, x0, d, self.N, ops, tolerance=1e-5
                )
                
                self.assertTrue(passed, f"Forward pass numerical validation failed with unroll={unroll}: {details}")
        
        validator.cleanup()
    
    def test_different_system_sizes(self):
        """Test forward pass with different system sizes"""
        sizes = [(1, 1, 3), (2, 1, 4), (3, 2, 3)]
        
        for nx, nu, N in sizes:
            with self.subTest(nx=nx, nu=nu, N=N):
                # Generate random stable system
                A = np.random.rand(nx, nx).astype(np.float32) * 0.1
                np.fill_diagonal(A, np.random.rand(nx) * 0.3 + 0.5)
                
                B = np.random.rand(nx, nu).astype(np.float32) * 0.5
                Q = np.eye(nx, dtype=np.float32)
                R = np.eye(nu, dtype=np.float32) * 0.1
                
                # Compute Kinf
                generator = TinyMPCGenerator(A, B, Q, R, N)
                
                # Test forward pass
                gen = ForwardPassGenerator(nx, nu, N)
                ops = gen.generate_operations(unroll=False)
                
                self.assertGreater(len(ops), 0)
                
                # Check for required operations
                op_types = [op.get('type') for op in ops]
                self.assertIn('loop', op_types)
                self.assertIn('comment', op_types)


if __name__ == '__main__':
    unittest.main()