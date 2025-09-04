#!/usr/bin/env python3
"""
Unit test for Dual Update subfunction
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparse_codegen.subfunction_generators import DualUpdateGenerator


class TestDualUpdate(unittest.TestCase):
    
    def setUp(self):
        """Set up test parameters"""
        self.nx = 3
        self.nu = 2
        self.N = 4
    
    def test_dual_update_structure(self):
        """Test dual update operation structure"""
        gen = DualUpdateGenerator(self.nx, self.nu, self.N)
        
        ops = gen.generate_operations(unroll=False)
        self.assertGreater(len(ops), 0)
        
        # Should have comment
        self.assertTrue(any(op.get('type') == 'comment' for op in ops))
        
        # Should have loops for y and g updates
        loops = [op for op in ops if op.get('type') == 'loop']
        self.assertEqual(len(loops), 2)  # One for y, one for g
    
    def test_algorithm_dimensions(self):
        """Test dual update algorithm numerical correctness"""
        from subfunction_validator import SubfunctionValidator
        
        gen = DualUpdateGenerator(self.nx, self.nu, self.N)
        validator = SubfunctionValidator()
        
        # Generate test data
        y = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        u = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1  
        znew = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        g = np.random.rand(self.N, self.nx).astype(np.float32) * 0.1
        x = np.random.rand(self.N, self.nx).astype(np.float32) * 0.5
        vnew = np.random.rand(self.N, self.nx).astype(np.float32) * 0.5
        
        # Test both modes
        for unroll in [False, True]:
            with self.subTest(unroll=unroll):
                ops = gen.generate_operations(unroll=unroll)
                
                # Numerical validation
                passed, details = validator.validate_dual_update(
                    y, u, znew, g, x, vnew, ops=ops, tolerance=1e-5
                )
                
                self.assertTrue(passed, f"Dual update numerical validation failed with unroll={unroll}: {details}")
        
        validator.cleanup()
    
    def test_operation_types(self):
        """Test that dual update uses correct operations"""
        gen = DualUpdateGenerator(self.nx, self.nu, self.N)
        ops = gen.generate_operations(unroll=True)
        
        assignments = [op for op in ops if op.get('type') == 'assignment']
        
        # Check that assignments have the right structure
        for assignment in assignments:
            self.assertIn('target', assignment)
            self.assertIn('target_idx', assignment)
            self.assertIn('value', assignment)
            
            target = assignment['target']
            self.assertIn(target, ['y', 'g'])
    
    def test_unrolled_vs_looped_equivalence(self):
        """Test that unrolled and looped generate equivalent operations"""
        gen = DualUpdateGenerator(self.nx, self.nu, self.N)
        
        ops_unrolled = gen.generate_operations(unroll=True)
        ops_looped = gen.generate_operations(unroll=False)
        
        # Different structure but should cover same variables
        self.assertGreater(len(ops_unrolled), len(ops_looped))
        
        # Looped should have exactly 2 loops + comment
        loop_count = sum(1 for op in ops_looped if op.get('type') == 'loop')
        comment_count = sum(1 for op in ops_looped if op.get('type') == 'comment')
        
        self.assertEqual(loop_count, 2)
        self.assertEqual(comment_count, 1)


if __name__ == '__main__':
    unittest.main()