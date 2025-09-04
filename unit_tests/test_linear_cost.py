#!/usr/bin/env python3
"""
Unit test for Linear Cost Update subfunction
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparse_codegen.subfunction_generators import LinearCostGenerator


class TestLinearCost(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters"""
        self.nx = 3
        self.nu = 2
        self.N = 4
    
    def test_cost_update_structure(self):
        """Test linear cost update operation structure"""
        gen = LinearCostGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=False)
        self.assertGreater(len(ops), 0)
        
        # Should have comment
        self.assertTrue(any(op.get('type') == 'comment' for op in ops))
        
        # Should have multiple loops for different cost terms
        loops = [op for op in ops if op.get('type') == 'loop']
        self.assertGreater(len(loops), 0)
    
    def test_cost_components(self):
        """Test that all cost components are updated"""
        gen = LinearCostGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=True)
        
        # Should update r, q, and terminal p
        assignments = [op for op in ops if op.get('type') == 'assignment']
        targets = [op.get('target') for op in assignments]
        
        # Check that r and q are updated
        self.assertIn('r', targets)
        self.assertIn('q', targets)
        self.assertIn('p', targets)  # Terminal cost
    
    def test_numerical_correctness(self):
        """Test linear cost numerical accuracy"""
        from subfunction_validator import SubfunctionValidator
        
        gen = LinearCostGenerator(self.nx, self.nu, self.N)
        validator = SubfunctionValidator()
        
        # Generate test data
        Q = np.diag([1.0, 0.1, 0.2]).astype(np.float32)
        R = np.diag([0.1, 0.05]).astype(np.float32)
        Xref = np.random.rand(self.N, self.nx).astype(np.float32) * 0.1
        Uref = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        x = np.random.rand(self.N, self.nx).astype(np.float32) * 0.5
        u = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        vnew = np.random.rand(self.N, self.nx).astype(np.float32) * 0.5
        znew = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        y = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        g = np.random.rand(self.N, self.nx).astype(np.float32) * 0.1
        rho = 1.0
        
        # Test both modes
        for unroll in [False, True]:
            with self.subTest(unroll=unroll):
                ops = gen.generate_operations(unroll=unroll)
                
                # Numerical validation
                passed, details = validator.validate_linear_cost(
                    Q, R, Xref, Uref, x, u, vnew, znew, y, g, rho, ops=ops, tolerance=1e-4
                )
                
                self.assertTrue(passed, f"Linear cost numerical validation failed with unroll={unroll}: {details}")
        
        validator.cleanup()
    
    def test_reference_cost_terms(self):
        """Test reference cost term generation"""
        gen = LinearCostGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=True)
        
        # Count element_wise operations that use reference terms
        r_element_wise = 0
        q_element_wise = 0
        
        for op in ops:
            if op.get('type') == 'element_wise':
                # Check left operand for Uref/Xref
                left = op.get('left')
                if left and hasattr(left, 'array') and left.array == 'Uref':
                    r_element_wise += 1
                elif left and hasattr(left, 'array') and left.array == 'Xref':
                    q_element_wise += 1
        
        # Should have element_wise operations for reference terms
        expected_r_ref = self.N - 1  # One per time step
        expected_q_ref = self.N      # One per time step
        
        self.assertGreaterEqual(r_element_wise, expected_r_ref)
        self.assertGreaterEqual(q_element_wise, expected_q_ref)
    
    def test_admm_penalty_terms(self):
        """Test ADMM penalty term generation"""
        gen = LinearCostGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=True)
        
        # Count penalty term updates
        penalty_updates = 0
        
        for op in ops:
            if op.get('type') == 'assignment':
                value_str = str(op.get('value', ''))
                # Look for rho penalty terms
                if 'rho' in value_str.lower() and ('znew' in value_str or 'vnew' in value_str):
                    penalty_updates += 1
        
        # Should have penalty updates for all variables
        expected_penalty = (self.N - 1) * self.nu + self.N * self.nx
        self.assertGreaterEqual(penalty_updates, expected_penalty)
    
    def test_terminal_cost(self):
        """Test terminal cost p[N-1] computation"""
        gen = LinearCostGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=True)
        
        # Should have Pinf matrix-vector operation (sparse_matvec)
        matvec_ops = [op for op in ops if op.get('type') in ['matvec', 'sparse_matvec']]
        pinf_ops = [op for op in matvec_ops if op.get('matrix') == 'Pinf_T']
        
        self.assertGreater(len(pinf_ops), 0, "Should have Pinf transpose operation")
        
        # Should update p[N-1]
        p_terminal_updates = 0
        for op in ops:
            if (op.get('type') == 'assignment' and 
                op.get('target') == 'p' and 
                op.get('target_idx') and 
                op.get('target_idx')[0] == self.N - 1):
                p_terminal_updates += 1
        
        self.assertEqual(p_terminal_updates, self.nx)


if __name__ == '__main__':
    unittest.main()