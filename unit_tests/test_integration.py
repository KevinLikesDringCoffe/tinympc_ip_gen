#!/usr/bin/env python3
"""
Integration test for all TinyMPC subfunctions
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sparse_codegen import TinyMPCGenerator, TinyMPCConfig
from sparse_codegen.subfunction_generators import (
    ForwardPassGenerator, SlackUpdateGenerator, DualUpdateGenerator,
    LinearCostGenerator, BackwardPassGenerator, TerminationCheckGenerator
)


class TestTinyMPCIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test system"""
        self.A = np.array([[1.0, 0.1], [0.0, 0.95]], dtype=np.float32)
        self.B = np.array([[0.5], [0.5]], dtype=np.float32)
        self.Q = np.diag([1.0, 0.1]).astype(np.float32)
        self.R = np.array([[0.1]], dtype=np.float32)
        self.N = 4
        self.nx = self.A.shape[0]
        self.nu = self.B.shape[1]
    
    def test_all_subfunctions_generate_ops(self):
        """Test that all subfunctions can generate operations"""
        generators = [
            ('Forward', ForwardPassGenerator(self.nx, self.nu, self.N)),
            ('Slack', SlackUpdateGenerator(self.nx, self.nu, self.N)),
            ('Dual', DualUpdateGenerator(self.nx, self.nu, self.N)),
            ('Cost', LinearCostGenerator(self.nx, self.nu, self.N)),
            ('Backward', BackwardPassGenerator(self.nx, self.nu, self.N)),
            ('Termination', TerminationCheckGenerator(self.nx, self.nu, self.N))
        ]
        
        for name, gen in generators:
            with self.subTest(subfunction=name):
                ops = gen.generate_operations(unroll=False)
                self.assertGreater(len(ops), 0, f"{name} generator produced no operations")
                
                # Should have at least a comment
                has_comment = any(op.get('type') == 'comment' for op in ops)
                self.assertTrue(has_comment, f"{name} generator has no comments")
    
    def test_complete_solver_generation(self):
        """Test complete solver generation"""
        generator = TinyMPCGenerator(self.A, self.B, self.Q, self.R, self.N)
        
        configs = [
            TinyMPCConfig(unroll=False, inline=False, target='c'),
            TinyMPCConfig(unroll=True, inline=False, target='c'),
            TinyMPCConfig(unroll=False, inline=True, target='hls')
        ]
        
        for i, config in enumerate(configs):
            with self.subTest(config=i):
                code = generator.generate(config)
                
                self.assertGreater(len(code), 500)
                
                # Check for required components
                components = [
                    '#include',
                    'tinympc_solve',
                    'Forward pass',
                    'temp_kinf',
                    'temp_u'
                ]
                
                for comp in components:
                    self.assertIn(comp, code, f"Missing component: {comp}")
    
    def test_matrix_dimensions_consistency(self):
        """Test that generated operations respect matrix dimensions"""
        generator = TinyMPCGenerator(self.A, self.B, self.Q, self.R, self.N)
        
        # Check derived matrix dimensions
        self.assertEqual(generator.Kinf.shape, (self.nu, self.nx))
        self.assertEqual(generator.Pinf.shape, (self.nx, self.nx))
        self.assertEqual(generator.Quu_inv.shape, (self.nu, self.nu))
        self.assertEqual(generator.AmBKt.shape, (self.nx, self.nx))
        
        # Check diagonal matrices
        self.assertEqual(generator.Q_diag.shape, (self.nx,))
        self.assertEqual(generator.R_diag.shape, (self.nu,))
    
    def test_variable_declarations(self):
        """Test that all required variables are declared"""
        generator = TinyMPCGenerator(self.A, self.B, self.Q, self.R, self.N)
        config = TinyMPCConfig()
        
        declarations = generator._generate_declarations(config)
        
        # Check for required arrays
        required_vars = [
            'float x[N][NX]',
            'float u[N-1][NU]',
            'float v[N][NX]',
            'float vnew[N][NX]',
            'float z[N-1][NU]',
            'float znew[N-1][NU]',
            'float g[N][NX]',
            'float y[N-1][NU]',
            'float q[N][NX]',
            'float r[N-1][NU]',
            'float p[N][NX]',
            'float d[N-1][NU]',
            'float Q[NX]',
            'float R[NU]',
            'temp_kinf',
            'temp_u',
            'temp_a',
            'temp_b'
        ]
        
        for var in required_vars:
            self.assertIn(var, declarations, f"Missing variable declaration: {var}")
    
    def test_compilation_readiness(self):
        """Test that generated code is ready for compilation"""
        generator = TinyMPCGenerator(self.A, self.B, self.Q, self.R, self.N)
        config = TinyMPCConfig(target='c')
        
        code = generator.generate(config)
        
        # Basic syntax checks
        self.assertEqual(code.count('{'), code.count('}'), "Unmatched braces")
        
        # Check for required C elements
        self.assertIn('#include <stdio.h>', code)
        self.assertIn('#include <math.h>', code)
        
        # Check for defines
        defines = ['#define NX', '#define NU', '#define N']
        for define in defines:
            self.assertIn(define, code)
        
        # No Python syntax should remain
        python_syntax = ['import ', 'def ', 'class ', 'ArrayAccess(array=']
        for syntax in python_syntax:
            self.assertNotIn(syntax, code, f"Found Python syntax: {syntax}")
    
    def test_complete_solver_numerical_validation(self):
        """Test complete TinyMPC solver code generation"""
        generator = TinyMPCGenerator(self.A, self.B, self.Q, self.R, self.N)
        
        # Test different configurations
        configs = [
            TinyMPCConfig(unroll=False, inline=False, target='c'),
            TinyMPCConfig(unroll=True, inline=False, target='c')
        ]
        
        for i, config in enumerate(configs):
            with self.subTest(config=i):
                code = generator.generate(config)
                
                # Basic validation - code should compile without errors
                self.assertGreater(len(code), 1000)
                self.assertIn('tinympc_solve', code)
                self.assertIn('Forward pass', code)
                
                # Check for no Python syntax
                python_syntax = ['ArrayAccess(', 'BinaryOp(', 'UnaryOp(']
                for syntax in python_syntax:
                    self.assertNotIn(syntax, code, f"Found Python AST syntax: {syntax}")
    
    def test_subfunction_integration_validation(self):
        """Test that all subfunctions work together numerically"""
        from subfunction_validator import SubfunctionValidator
        from sparse_codegen.subfunction_generators import (
            ForwardPassGenerator, SlackUpdateGenerator, DualUpdateGenerator
        )
        
        validator = SubfunctionValidator()
        
        # Generate test data
        x0 = np.random.rand(self.nx).astype(np.float32)
        d = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        
        # Generate Kinf
        generator = TinyMPCGenerator(self.A, self.B, self.Q, self.R, self.N)
        
        # Test forward pass
        fp_gen = ForwardPassGenerator(self.nx, self.nu, self.N)
        fp_ops = fp_gen.generate_operations(unroll=False)
        
        passed, details = validator.validate_forward_pass(
            self.A, self.B, generator.Kinf, x0, d, self.N, fp_ops, tolerance=1e-5
        )
        
        self.assertTrue(passed, f"Forward pass integration validation failed: {details}")
        
        # Test slack and dual updates with random data
        u = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        y = np.random.rand(self.N-1, self.nu).astype(np.float32) * 0.1
        x = np.random.rand(self.N, self.nx).astype(np.float32) * 0.5
        g = np.random.rand(self.N, self.nx).astype(np.float32) * 0.1
        
        # Slack update
        slack_gen = SlackUpdateGenerator(self.nx, self.nu, self.N)
        slack_ops = slack_gen.generate_operations(unroll=False)
        
        passed, details = validator.validate_slack_update(u, y, x, g, slack_ops, tolerance=1e-5)
        self.assertTrue(passed, f"Slack update integration validation failed: {details}")
        
        # Dual update
        vnew = x + g
        znew = u + y
        
        dual_gen = DualUpdateGenerator(self.nx, self.nu, self.N)
        dual_ops = dual_gen.generate_operations(unroll=False)
        
        passed, details = validator.validate_dual_update(y, u, znew, g, x, vnew, dual_ops, tolerance=1e-5)
        self.assertTrue(passed, f"Dual update integration validation failed: {details}")
        
        validator.cleanup()


if __name__ == '__main__':
    unittest.main()