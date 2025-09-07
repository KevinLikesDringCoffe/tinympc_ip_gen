"""
TinyMPC Hardware Driver - Main Interface
Provides tinympcref-compatible interface for hardware acceleration on Ultra96
"""

import numpy as np
import time
import logging
import os
from threading import Thread, Event
try:
    from .hw_interface import HardwareInterface
    from .memory_manager import MemoryManager
except ImportError:
    from hw_interface import HardwareInterface
    from memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class tinympc_hw:
    """
    TinyMPC Hardware Driver with tinympcref-compatible interface
    
    This class provides the same interface as tinympcref.py but executes
    the optimization on FPGA hardware instead of CPU software.
    """
    
    def __init__(self, bitstream_path=None, hwh_path=None):
        """
        Initialize TinyMPC hardware driver
        
        Args:
            bitstream_path: Path to .bit file (can be set later)
            hwh_path: Path to .hwh file (optional, auto-detected if None)
        """
        # Hardware interface components
        self.hw_interface = HardwareInterface()
        self.memory_manager = MemoryManager()
        
        # System parameters (will be inferred from bitstream)
        self.nx = 0
        self.nu = 0
        self.N = 0
        self.control_freq = 0.0
        self.loaded = False
        
        # Algorithm parameters (compatible with tinympcref)
        self.max_iter = 100
        self.check_termination = 10
        self.verbose = False
        
        # State variables (compatible with tinympcref)
        self.x = None           # Optimal state trajectory (N, nx)
        self.u = None           # Optimal control inputs (N-1, nu)
        self.iter = 0           # Actual iterations completed
        self.solved = 0         # Convergence flag (1 if converged, 0 otherwise)
        self.solve_time = 0.0   # Hardware execution time (ms)
        
        # Residual tracking (for compatibility)
        self.primal_residual_state = 0.0
        self.dual_residual_state = 0.0
        self.primal_residual_input = 0.0
        self.dual_residual_input = 0.0
        
        # Input data storage
        self._x0 = None
        self._xref = None
        self._uref = None
        
        # Async execution support
        self._async_thread = None
        self._async_event = Event()
        self._async_result = None
        self._async_error = None
        
        # Auto-load bitstream if provided
        if bitstream_path is not None:
            self.load_bitstream(bitstream_path, hwh_path)
    
    def load_bitstream(self, bitstream_path, hwh_path=None):
        """
        Load bitstream to FPGA and setup hardware interface
        
        Args:
            bitstream_path: Path to .bit file
            hwh_path: Path to .hwh file (optional)
        """
        try:
            # Load overlay
            self.hw_interface.load_overlay(bitstream_path, hwh_path)
            
            # Try to infer parameters from bitstream filename
            N, nx, nu, freq = self.memory_manager.infer_parameters_from_bitstream_path(bitstream_path)
            
            if N is None or nx is None or nu is None:
                raise RuntimeError(
                    "Could not infer parameters from bitstream filename. "
                    "Please use a filename like 'tinympcproj_N5_100Hz_float.bit' "
                    "or call setup() with explicit parameters."
                )
            
            # Setup memory layout
            self.memory_manager.setup_memory_layout(N, nx, nu)
            
            # Store parameters
            self.N = N
            self.nx = nx
            self.nu = nu
            self.control_freq = freq
            
            # Allocate hardware memory
            self.hw_interface.setup_memory(
                self.memory_manager.memory_size, nx, nu, N
            )
            
            # Initialize result arrays
            self.x = np.zeros((self.N, self.nx), dtype=np.float32)
            self.u = np.zeros((self.N-1, self.nu), dtype=np.float32)
            
            self.loaded = True
            logger.info(f"Successfully loaded bitstream: N={N}, nx={nx}, nu={nu}, freq={freq}Hz")
            
        except Exception as e:
            logger.error(f"Failed to load bitstream: {e}")
            raise
    
    def setup(self, A=None, B=None, Q=None, R=None, N=None, rho=1.0,
              x_min=None, x_max=None, u_min=None, u_max=None, 
              max_iter=100, check_termination=10, verbose=False, **settings):
        """
        Setup solver parameters (compatible with tinympcref interface)
        
        Note: A, B, Q, R matrices are ignored as they are hardcoded in hardware.
        Only algorithm parameters (max_iter, check_termination) are used.
        
        Args:
            A, B, Q, R: System matrices (ignored, for compatibility only)
            N: Horizon length (ignored, inferred from bitstream)  
            rho: ADMM penalty parameter (ignored, hardcoded in hardware)
            x_min, x_max, u_min, u_max: Constraints (ignored, hardcoded in hardware)
            max_iter: Maximum ADMM iterations
            check_termination: Check termination every N iterations
            verbose: Enable verbose logging
            **settings: Additional settings (ignored)
        """
        if not self.loaded:
            logger.warning("Bitstream not loaded. Call load_bitstream() first.")
            return
        
        # Store algorithm parameters
        self.max_iter = max_iter
        self.check_termination = check_termination
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
            self.memory_manager.print_memory_layout()
        
        # Warn about ignored parameters
        if A is not None or B is not None or Q is not None or R is not None:
            logger.warning("System matrices A, B, Q, R are ignored (hardcoded in hardware)")
        
        if N is not None and N != self.N:
            logger.warning(f"Specified N={N} ignored, using hardware N={self.N}")
        
        logger.info(f"Setup complete: max_iter={max_iter}, check_termination={check_termination}")
    
    def set_x0(self, x0):
        """
        Set initial state
        
        Args:
            x0: Initial state vector (nx,)
        """
        x0 = np.asarray(x0, dtype=np.float32)
        self.memory_manager.validate_input_data(x0=x0)
        self._x0 = x0.copy()
        
        if self.verbose:
            logger.debug(f"Set x0: {x0}")
    
    def set_x_ref(self, x_ref):
        """
        Set reference state trajectory
        
        Args:
            x_ref: Reference state trajectory (N, nx)
        """
        x_ref = np.asarray(x_ref, dtype=np.float32)
        self.memory_manager.validate_input_data(xref=x_ref)
        self._xref = x_ref.copy()
        
        if self.verbose:
            logger.debug(f"Set xref: shape {x_ref.shape}")
    
    def set_u_ref(self, u_ref):
        """
        Set reference control input trajectory
        
        Args:
            u_ref: Reference control input trajectory (N-1, nu)
        """
        u_ref = np.asarray(u_ref, dtype=np.float32)
        self.memory_manager.validate_input_data(uref=u_ref)
        self._uref = u_ref.copy()
        
        if self.verbose:
            logger.debug(f"Set uref: shape {u_ref.shape}")
    
    def solve(self, timeout=10.0):
        """
        Solve optimization problem synchronously (blocking)
        
        Args:
            timeout: Maximum time to wait for completion (seconds)
            
        Returns:
            bool: True if solved successfully, False if failed/timeout
        """
        if not self.loaded:
            raise RuntimeError("Bitstream not loaded. Call load_bitstream() first.")
        
        if self._x0 is None:
            raise RuntimeError("Initial state not set. Call set_x0() first.")
        
        try:
            start_time = time.time()
            
            # Pack input data into memory
            self.memory_manager.pack_input_data(
                self.hw_interface.memory_buffer, 
                self._x0, self._xref, self._uref
            )
            
            # Write control registers
            self.hw_interface.write_control_registers(
                self.max_iter, self.check_termination
            )
            
            # Start computation
            computation_start = time.time()
            self.hw_interface.start_computation()
            
            # Wait for completion
            if not self.hw_interface.wait_for_completion(timeout):
                logger.error(f"Hardware computation timeout after {timeout}s")
                self.solved = 0
                return False
            
            computation_end = time.time()
            self.solve_time = (computation_end - computation_start) * 1000  # Convert to ms
            
            # Read results
            self.x, self.u = self.memory_manager.unpack_output_data(
                self.hw_interface.memory_buffer
            )
            
            # Mark as solved (hardware doesn't provide convergence info)
            self.solved = 1
            self.iter = self.max_iter  # Assume max iterations (conservative)
            
            total_time = (time.time() - start_time) * 1000
            
            if self.verbose:
                logger.info(f"Solved in {self.solve_time:.2f}ms (total: {total_time:.2f}ms)")
                logger.debug(f"Final state: {self.x[-1]}")
                logger.debug(f"First control: {self.u[0]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Solve failed: {e}")
            self.solved = 0
            return False
    
    def solve_async(self):
        """
        Start optimization asynchronously (non-blocking)
        
        Returns:
            bool: True if started successfully
        """
        if not self.loaded:
            raise RuntimeError("Bitstream not loaded. Call load_bitstream() first.")
        
        if self._async_thread is not None and self._async_thread.is_alive():
            logger.warning("Async solve already in progress")
            return False
        
        # Reset async state
        self._async_event.clear()
        self._async_result = None
        self._async_error = None
        
        # Start async thread
        self._async_thread = Thread(target=self._async_solve_worker)
        self._async_thread.start()
        
        if self.verbose:
            logger.debug("Started async solve")
        
        return True
    
    def _async_solve_worker(self):
        """Worker function for async solve"""
        try:
            result = self.solve()
            self._async_result = result
        except Exception as e:
            self._async_error = e
            self._async_result = False
        finally:
            self._async_event.set()
    
    def is_done(self):
        """
        Check if async computation is complete
        
        Returns:
            bool: True if computation is done (or no async solve running)
        """
        if self._async_thread is None:
            return True
        
        return self._async_event.is_set()
    
    def wait(self, timeout=10.0):
        """
        Wait for async computation to complete
        
        Args:
            timeout: Maximum time to wait (seconds)
            
        Returns:
            bool: True if completed, False if timeout
        """
        if self._async_thread is None:
            return True
        
        completed = self._async_event.wait(timeout)
        
        if completed and self._async_error is not None:
            raise self._async_error
        
        return completed
    
    def get_results(self):
        """
        Get results from async computation
        
        Returns:
            dict: Results dictionary with 'x', 'u', 'solved', 'iter', 'solve_time'
            
        Raises:
            RuntimeError: If async computation not complete or failed
        """
        if not self.is_done():
            raise RuntimeError("Async computation not complete. Call wait() first.")
        
        if self._async_error is not None:
            raise self._async_error
        
        if self._async_result is False:
            raise RuntimeError("Async computation failed")
        
        return {
            'x': self.x.copy(),
            'u': self.u.copy(),
            'solved': self.solved,
            'iter': self.iter,
            'solve_time': self.solve_time
        }
    
    def reset(self):
        """Reset solver state"""
        self.iter = 0
        self.solved = 0
        self.solve_time = 0.0
        
        self.primal_residual_state = 0.0
        self.dual_residual_state = 0.0
        self.primal_residual_input = 0.0
        self.dual_residual_input = 0.0
        
        if self.hw_interface.is_loaded:
            self.hw_interface.reset_ip()
        
        if self.verbose:
            logger.debug("Reset solver state")
    
    def get_info(self):
        """
        Get solver information
        
        Returns:
            dict: Solver information
        """
        return {
            'type': 'hardware',
            'loaded': self.loaded,
            'nx': self.nx,
            'nu': self.nu,
            'N': self.N,
            'control_freq': self.control_freq,
            'max_iter': self.max_iter,
            'check_termination': self.check_termination,
            'memory_info': self.memory_manager.get_memory_info() if self.loaded else None
        }
    
    def print_stats(self):
        """Print solver statistics"""
        print(f"\n=== TinyMPC Hardware Solver Stats ===")
        print(f"System: {self.nx} states, {self.nu} inputs, N={self.N}")
        print(f"Control frequency: {self.control_freq} Hz")
        print(f"Loaded: {self.loaded}")
        print(f"Last solve: {self.solve_time:.2f}ms, {self.iter} iterations")
        print(f"Converged: {'Yes' if self.solved else 'No'}")
        if self.loaded:
            memory_info = self.memory_manager.get_memory_info()
            print(f"Memory size: {memory_info['memory_size']} elements")
        print(f"=====================================\n")
    
    def cleanup(self):
        """Clean up resources"""
        # Stop any running async computation
        if self._async_thread is not None and self._async_thread.is_alive():
            logger.info("Waiting for async computation to complete...")
            self.wait(timeout=5.0)
        
        # Clean up hardware interface
        self.hw_interface.cleanup()
        self.loaded = False
        
        logger.info("Cleaned up hardware resources")
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.cleanup()

# Alias for compatibility
TinyMPCHW = tinympc_hw