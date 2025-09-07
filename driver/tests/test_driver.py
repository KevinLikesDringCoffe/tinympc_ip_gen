#!/usr/bin/env python3
"""
Unit tests for TinyMPC Hardware Driver
"""

import sys
import os
import numpy as np
import unittest
from unittest.mock import Mock, MagicMock, patch

# Add driver to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager class"""
    
    def setUp(self):
        from memory_manager import MemoryManager
        self.mm = MemoryManager()
    
    def test_parameter_inference(self):
        """Test parameter inference from filename"""
        test_cases = [
            ("tinympcproj_N5_100Hz_float.bit", (5, 12, 4, 100.0)),
            ("tinympcproj_N10_50Hz_float.bit", (10, 12, 4, 50.0)),
            ("tinympcproj_N20_20Hz_float.bit", (20, 12, 4, 20.0)),
            ("invalid_filename.bit", (None, None, None, None))
        ]
        
        for filename, expected in test_cases:
            with self.subTest(filename=filename):
                result = self.mm.infer_parameters_from_bitstream_path(filename)
                self.assertEqual(result, expected)
    
    def test_memory_layout_setup(self):
        """Test memory layout calculation"""
        N, nx, nu = 5, 6, 4
        self.mm.setup_memory_layout(N, nx, nu)
        
        # Check parameters
        self.assertEqual(self.mm.N, N)
        self.assertEqual(self.mm.nx, nx)
        self.assertEqual(self.mm.nu, nu)
        
        # Check offsets
        self.assertEqual(self.mm.X0_OFFSET, 0)
        self.assertEqual(self.mm.XREF_OFFSET, nx)
        self.assertEqual(self.mm.UREF_OFFSET, nx + N * nx)
        
        # Check memory size
        expected_size = nx + 2 * N * nx + 2 * (N-1) * nu
        self.assertEqual(self.mm.memory_size, expected_size)
    
    def test_input_validation(self):
        """Test input data validation"""
        self.mm.setup_memory_layout(5, 6, 4)
        
        # Valid inputs
        x0 = np.zeros(6)
        xref = np.zeros((5, 6))
        uref = np.zeros((4, 4))
        
        # Should not raise
        self.mm.validate_input_data(x0, xref, uref)
        
        # Invalid inputs
        with self.assertRaises(ValueError):
            self.mm.validate_input_data(np.zeros(5))  # Wrong x0 size
        
        with self.assertRaises(ValueError):
            self.mm.validate_input_data(xref=np.zeros((4, 6)))  # Wrong xref shape
    
    def test_data_packing_unpacking(self):
        """Test data packing and unpacking"""
        N, nx, nu = 3, 4, 2
        self.mm.setup_memory_layout(N, nx, nu)
        
        # Create test data
        x0 = np.array([1, 2, 3, 4], dtype=np.float32)
        xref = np.ones((3, 4), dtype=np.float32) * 2
        uref = np.ones((2, 2), dtype=np.float32) * 3
        
        # Create memory buffer
        memory_buffer = np.zeros(self.mm.memory_size, dtype=np.float32)
        
        # Pack data
        self.mm.pack_input_data(memory_buffer, x0, xref, uref)
        
        # Check packed data
        np.testing.assert_array_equal(
            memory_buffer[self.mm.X0_OFFSET:self.mm.X0_OFFSET+nx], x0
        )
        
        # Pack dummy output data
        x_out_dummy = np.ones((N, nx), dtype=np.float32) * 10
        u_out_dummy = np.ones((N-1, nu), dtype=np.float32) * 20
        
        memory_buffer[self.mm.X_OUT_OFFSET:self.mm.X_OUT_OFFSET+N*nx] = x_out_dummy.flatten()
        memory_buffer[self.mm.U_OUT_OFFSET:self.mm.U_OUT_OFFSET+(N-1)*nu] = u_out_dummy.flatten()
        
        # Unpack results
        x_out, u_out = self.mm.unpack_output_data(memory_buffer)
        
        np.testing.assert_array_equal(x_out, x_out_dummy)
        np.testing.assert_array_equal(u_out, u_out_dummy)
    
    def test_test_data_creation(self):
        """Test test data generation"""
        self.mm.setup_memory_layout(5, 12, 4)
        
        for test_type in ['hover', 'tracking', 'random']:
            x0, xref, uref = self.mm.create_test_data(test_type)
            
            # Check shapes
            self.assertEqual(x0.shape, (12,))
            self.assertEqual(xref.shape, (5, 12))
            self.assertEqual(uref.shape, (4, 4))
            
            # Check data types
            self.assertEqual(x0.dtype, np.float32)
            self.assertEqual(xref.dtype, np.float32)
            self.assertEqual(uref.dtype, np.float32)

class TestHardwareInterface(unittest.TestCase):
    """Test HardwareInterface class (mocked)"""
    
    def setUp(self):
        # Mock PYNQ imports
        self.pynq_mock = Mock()
        self.overlay_mock = Mock()
        self.ip_mock = Mock()
        self.allocate_mock = Mock()
        
        sys.modules['pynq'] = self.pynq_mock
        self.pynq_mock.Overlay = Mock(return_value=self.overlay_mock)
        self.pynq_mock.allocate = self.allocate_mock
        
        from hw_interface import HardwareInterface
        self.hw = HardwareInterface()
    
    def test_overlay_loading(self):
        """Test overlay loading"""
        # Setup mock
        self.overlay_mock.ip_dict = {'tinympc_solver_0': 'mock_ip'}
        setattr(self.overlay_mock, 'tinympc_solver_0', self.ip_mock)
        
        # Test loading
        self.hw.load_overlay("test.bit", "test.hwh")
        
        # Verify calls
        self.pynq_mock.Overlay.assert_called_once_with("test.bit")
        self.assertTrue(self.hw.is_loaded)
    
    def test_memory_setup(self):
        """Test memory allocation"""
        # Setup mock
        mock_buffer = np.zeros(100, dtype=np.float32)
        mock_buffer.physical_address = 0x10000000
        self.allocate_mock.return_value = mock_buffer
        
        # Test setup
        self.hw.setup_memory(100, 6, 4, 5)
        
        # Verify
        self.allocate_mock.assert_called_once()
        self.assertEqual(self.hw.memory_size, 100)
        self.assertEqual(self.hw.nx, 6)
        self.assertEqual(self.hw.nu, 4)
        self.assertEqual(self.hw.N, 5)
    
    def test_register_operations(self):
        """Test register read/write operations"""
        self.hw.is_loaded = True
        self.hw.ip = self.ip_mock
        self.hw.memory_buffer = Mock()
        self.hw.memory_buffer.physical_address = 0x10000000
        
        # Test writing control registers
        self.hw.write_control_registers(100, 10)
        
        # Verify register writes
        self.ip_mock.write.assert_called()
        self.assertGreaterEqual(self.ip_mock.write.call_count, 3)  # At least 3 registers
    
    def tearDown(self):
        # Clean up mocked modules
        if 'pynq' in sys.modules:
            del sys.modules['pynq']

class TestTinyMPCHW(unittest.TestCase):
    """Test main tinympc_hw class (mocked)"""
    
    def setUp(self):
        # Mock all PYNQ dependencies
        self.setup_mocks()
        
        from tinympc_hw import tinympc_hw
        self.solver_class = tinympc_hw
    
    def setup_mocks(self):
        """Setup all necessary mocks"""
        # Mock PYNQ
        self.pynq_mock = Mock()
        self.overlay_mock = Mock()
        self.ip_mock = Mock()
        
        sys.modules['pynq'] = self.pynq_mock
        self.pynq_mock.Overlay = Mock(return_value=self.overlay_mock)
        
        mock_buffer = np.zeros(100, dtype=np.float32)
        mock_buffer.physical_address = 0x10000000
        self.pynq_mock.allocate = Mock(return_value=mock_buffer)
        
        # Setup overlay mock
        self.overlay_mock.ip_dict = {'tinympc_solver_0': 'mock_ip'}
        setattr(self.overlay_mock, 'tinympc_solver_0', self.ip_mock)
    
    def test_initialization(self):
        """Test solver initialization"""
        solver = self.solver_class()
        
        self.assertFalse(solver.loaded)
        self.assertEqual(solver.nx, 0)
        self.assertEqual(solver.nu, 0)
        self.assertEqual(solver.N, 0)
    
    def test_bitstream_loading(self):
        """Test bitstream loading with parameter inference"""
        solver = self.solver_class()
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            solver.load_bitstream("tinympcproj_N5_100Hz_float.bit")
        
        self.assertTrue(solver.loaded)
        self.assertEqual(solver.N, 5)
        self.assertEqual(solver.nx, 12)  # Quadrotor states
        self.assertEqual(solver.nu, 4)   # Quadrotor inputs
    
    def test_setup_compatibility(self):
        """Test setup method compatibility with tinympcref"""
        solver = self.solver_class()
        solver.loaded = True
        solver.N, solver.nx, solver.nu = 5, 6, 4
        
        # Should accept tinympcref parameters but ignore system matrices
        A = np.random.randn(6, 6)
        B = np.random.randn(6, 4)
        Q = np.random.randn(6, 6)
        R = np.random.randn(4, 4)
        
        # Should not raise error
        solver.setup(A, B, Q, R, N=5, max_iter=50, verbose=True)
        
        # Algorithm parameters should be stored
        self.assertEqual(solver.max_iter, 50)
        self.assertTrue(solver.verbose)
    
    def test_problem_setup(self):
        """Test problem data setting"""
        solver = self.solver_class()
        solver.memory_manager.setup_memory_layout(5, 6, 4)
        
        # Set problem data
        x0 = np.zeros(6)
        xref = np.zeros((5, 6))
        uref = np.zeros((4, 4))
        
        solver.set_x0(x0)
        solver.set_x_ref(xref)
        solver.set_u_ref(uref)
        
        # Verify data is stored
        np.testing.assert_array_equal(solver._x0, x0)
        np.testing.assert_array_equal(solver._xref, xref)
        np.testing.assert_array_equal(solver._uref, uref)
    
    def test_get_info(self):
        """Test info retrieval"""
        solver = self.solver_class()
        solver.loaded = True
        solver.N, solver.nx, solver.nu = 5, 6, 4
        solver.control_freq = 100.0
        
        info = solver.get_info()
        
        self.assertEqual(info['type'], 'hardware')
        self.assertTrue(info['loaded'])
        self.assertEqual(info['N'], 5)
        self.assertEqual(info['nx'], 6)
        self.assertEqual(info['nu'], 4)
        self.assertEqual(info['control_freq'], 100.0)
    
    def tearDown(self):
        # Clean up mocked modules
        modules_to_remove = ['pynq', 'hw_interface', 'memory_manager', 'tinympc_hw']
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

def run_tests():
    """Run all tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestMemoryManager))
    suite.addTest(unittest.makeSuite(TestHardwareInterface))
    suite.addTest(unittest.makeSuite(TestTinyMPCHW))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("TinyMPC Hardware Driver - Unit Tests")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)