#!/usr/bin/env python3
"""
Unit test for Termination Check subfunction
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparse_codegen.subfunction_generators import TerminationCheckGenerator


class TestTerminationCheck(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters"""
        self.nx = 2
        self.nu = 1 
        self.N = 5
        self.check_freq = 25
    
    def test_termination_structure(self):
        """Test termination check operation structure"""
        gen = TerminationCheckGenerator(self.nx, self.nu, self.N, self.check_freq)
        
        ops = gen.generate_operations()
        self.assertGreater(len(ops), 0)
        
        # Should have comment
        self.assertTrue(any(op.get('type') == 'comment' for op in ops))
        
        # Should have conditional check
        conditionals = [op for op in ops if op.get('type') == 'conditional']
        self.assertGreater(len(conditionals), 0)
    
    def test_residual_computation(self):
        """Test residual computation operations"""
        gen = TerminationCheckGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations()
        
        # Should compute both primal and dual residuals
        # Look for abs_max operations
        residual_ops = []
        
        def find_abs_max_ops(ops_list):
            for op in ops_list:
                if op.get('type') == 'abs_max':
                    residual_ops.append(op)
                elif op.get('type') == 'conditional' and 'body' in op:
                    find_abs_max_ops(op['body'])
        
        find_abs_max_ops(ops)
        
        # Should have residual computations for state and input
        self.assertGreaterEqual(len(residual_ops), 2)
    
    def test_convergence_condition(self):
        """Test convergence condition structure"""
        gen = TerminationCheckGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations()
        
        # Find the main conditional
        main_conditional = None
        for op in ops:
            if op.get('type') == 'conditional':
                main_conditional = op
                break
        
        self.assertIsNotNone(main_conditional)
        
        # Should have nested conditionals for convergence check
        body = main_conditional.get('body', [])
        inner_conditionals = [op for op in body if op.get('type') == 'conditional']
        
        self.assertGreater(len(inner_conditionals), 0)
    
    def test_frequency_check(self):
        """Test that termination check respects frequency parameter"""
        gen = TerminationCheckGenerator(self.nx, self.nu, self.N, check_frequency=10)
        
        ops = gen.generate_operations()
        
        # The outer conditional should check iter % check_frequency == 0
        main_conditional = next(op for op in ops if op.get('type') == 'conditional')
        
        # Check condition structure
        condition = main_conditional.get('condition')
        self.assertIsNotNone(condition)
        
        # Should involve modulo operation
        condition_str = str(condition)
        self.assertTrue('%' in condition_str or 'iter' in condition_str)
    
    def test_solved_flag_setting(self):
        """Test that solved flag is set correctly"""
        gen = TerminationCheckGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations()
        
        # Find assignment to 'solved' variable
        solved_assignments = []
        
        def find_solved_assignments(ops_list):
            for op in ops_list:
                if (op.get('type') == 'assignment' and 
                    op.get('target') == 'solved'):
                    solved_assignments.append(op)
                elif op.get('type') == 'conditional' and 'body' in op:
                    find_solved_assignments(op['body'])
        
        find_solved_assignments(ops)
        
        self.assertGreater(len(solved_assignments), 0)
        
        # Check that solved is set to 1
        for assignment in solved_assignments:
            value = assignment.get('value')
            self.assertEqual(str(value), '1')
    
    def test_numerical_correctness(self):
        """Test termination check numerical accuracy"""
        from subfunction_validator import SubfunctionValidator
        
        gen = TerminationCheckGenerator(self.nx, self.nu, self.N, check_frequency=1)  # Check every iteration
        validator = SubfunctionValidator()
        
        # Generate test data - small residuals to test convergence
        x = np.random.rand(self.N, self.nx).astype(np.float32) * 0.001
        u = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.001
        vnew = x + np.random.rand(self.N, self.nx).astype(np.float32) * 0.0001  # Small difference
        znew = u + np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.0001
        
        # Test convergence case
        abs_pri_tol = 1e-3
        abs_dua_tol = 1e-4
        iter_num = 0  # Will trigger check since frequency=1
        
        ops = gen.generate_operations(unroll=False)
        
        # Numerical validation
        passed, details = validator.validate_termination_check(
            x, u, vnew, znew, abs_pri_tol, abs_dua_tol, iter_num, 1, ops=ops, tolerance=1e-5
        )
        
        self.assertTrue(passed, f"Termination check numerical validation failed: {details}")
        
        validator.cleanup()


if __name__ == '__main__':
    unittest.main()