#!/usr/bin/env python3
"""
Unit test for Slack Update subfunction
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparse_codegen.subfunction_generators import SlackUpdateGenerator


class TestSlackUpdate(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters"""
        self.nx = 2
        self.nu = 1
        self.N = 4
    
    def test_slack_update_structure(self):
        """Test slack update operation structure"""
        gen = SlackUpdateGenerator(self.nx, self.nu, self.N, 
                                 has_input_bounds=False, has_state_bounds=False)
        
        # Test without constraints
        ops = gen.generate_operations(unroll=False)
        self.assertGreater(len(ops), 0)
        
        # Should have comment
        self.assertTrue(any(op.get('type') == 'comment' for op in ops))
        
        # Check operation types
        op_types = [op.get('type') for op in ops]
        self.assertIn('loop', op_types)
    
    def test_slack_update_with_constraints(self):
        """Test slack update with box constraints"""
        gen = SlackUpdateGenerator(self.nx, self.nu, self.N,
                                 has_input_bounds=True, has_state_bounds=True)
        
        ops = gen.generate_operations(unroll=True)
        self.assertGreater(len(ops), 0)
        
        # Should include constraint operations
        has_constraints = any('constraint' in str(op).lower() for op in ops)
        self.assertTrue(has_constraints or len(ops) > 10)  # More ops with constraints
    
    def test_unrolled_vs_looped(self):
        """Test difference between unrolled and looped implementations"""
        gen = SlackUpdateGenerator(self.nx, self.nu, self.N)
        
        ops_unrolled = gen.generate_operations(unroll=True)
        ops_looped = gen.generate_operations(unroll=False)
        
        # Unrolled should have more operations
        self.assertGreater(len(ops_unrolled), len(ops_looped))
        
        # Looped should have loop operations
        loop_types = [op.get('type') for op in ops_looped]
        self.assertIn('loop', loop_types)
    
    def test_algorithm_correctness(self):
        """Test slack update algorithm numerical correctness"""
        from subfunction_validator import SubfunctionValidator
        
        nx, nu, N = 2, 1, 3
        gen = SlackUpdateGenerator(nx, nu, N)
        validator = SubfunctionValidator()
        
        # Generate test data
        u = np.random.rand(N-1, nu).astype(np.float32) * 0.1
        y = np.random.rand(N-1, nu).astype(np.float32) * 0.1
        x = np.random.rand(N, nx).astype(np.float32) * 0.5
        g = np.random.rand(N, nx).astype(np.float32) * 0.1
        
        # Test both modes
        for unroll in [False, True]:
            with self.subTest(unroll=unroll):
                ops = gen.generate_operations(unroll=unroll)
                
                # Numerical validation
                passed, details = validator.validate_slack_update(
                    u, y, x, g, ops=ops, tolerance=1e-5
                )
                
                self.assertTrue(passed, f"Slack update numerical validation failed with unroll={unroll}: {details}")
        
        validator.cleanup()


if __name__ == '__main__':
    unittest.main()