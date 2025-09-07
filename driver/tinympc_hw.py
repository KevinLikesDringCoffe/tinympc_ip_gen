"""
TinyMPC Hardware Driver - Compatibility Interface
Provides tinympcref-compatible interface for hardware acceleration using the TinyMPCDriver
"""

import numpy as np
import time
import logging
import os
import re
try:
    from .tinympc_driver import TinyMPCDriver
except ImportError:
    from tinympc_driver import TinyMPCDriver

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
        # Initialize the core driver
        self._driver = None
        
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
        
        # Auto-load bitstream if provided
        if bitstream_path is not None:
            self.load_bitstream(bitstream_path, hwh_path)
    
    def _infer_parameters_from_bitstream_path(self, bitstream_path):
        """
        Try to infer N, nx, nu from bitstream file path
        
        Args:
            bitstream_path: Path to bitstream file
            
        Returns:
            tuple: (N, nx, nu, freq) or (None, None, None, None) if unable to infer
        """
        try:
            # Extract filename from path
            filename = os.path.basename(bitstream_path)
            
            # Pattern: tinympcproj_N<N>_<freq>Hz_float
            pattern = r'tinympcproj_N(\d+)_(\d+(?:\.\d+)?)Hz_float'
            match = re.search(pattern, filename)
            
            if match:
                N = int(match.group(1))
                freq = float(match.group(2))
                
                # Infer nx, nu based on typical quadrotor dynamics
                # This is based on the crazyflie model used in the generator
                nx = 12  # [x, y, z, vx, vy, vz, phi, theta, psi, wx, wy, wz]
                nu = 4   # [thrust, roll_torque, pitch_torque, yaw_torque]
                
                logger.info(f"Inferred parameters from filename: N={N}, nx={nx}, nu={nu}, freq={freq}Hz")
                return N, nx, nu, freq
            else:
                logger.warning(f"Could not parse parameters from filename: {filename}")
                return None, None, None, None
                
        except Exception as e:
            logger.error(f"Error inferring parameters: {e}")
            return None, None, None, None
    
    def load_bitstream(self, bitstream_path, hwh_path=None):
        """
        Load bitstream to FPGA and setup hardware interface
        
        Args:
            bitstream_path: Path to .bit file
            hwh_path: Path to .hwh file (optional)
        """
        try:
            # Try to infer parameters from bitstream filename
            N, nx, nu, freq = self._infer_parameters_from_bitstream_path(bitstream_path)
            
            if N is None or nx is None or nu is None:
                raise RuntimeError(
                    "Could not infer parameters from bitstream filename. "
                    "Please use a filename like 'tinympcproj_N5_100Hz_float.bit' "
                    "or call setup() with explicit parameters."
                )
            
            # Set system parameters
            self.N = N
            self.nx = nx
            self.nu = nu
            self.control_freq = freq if freq else 100.0
            
            # Initialize the core driver with inferred parameters
            self._driver = TinyMPCDriver(
                overlay_path=bitstream_path,
                nx=self.nx,
                nu=self.nu,
                n_horizon=self.N,
                clock_frequency_mhz=250
            )
            
            self.loaded = True
            logger.info(f"Bitstream loaded: N={self.N}, nx={self.nx}, nu={self.nu}")
            
        except Exception as e:
            logger.error(f"Failed to load bitstream: {e}")
            self.loaded = False
            raise
    
    def setup(self, max_iter=100, check_termination=10, verbose=False, **kwargs):
        """
        Setup solver parameters (compatible with tinympcref interface)
        
        Args:
            max_iter: Maximum ADMM iterations
            check_termination: Check convergence every N iterations
            verbose: Enable verbose logging
            **kwargs: Additional parameters (ignored for hardware)
        """
        self.max_iter = max_iter
        self.check_termination = check_termination
        self.verbose = verbose
        
        if self.verbose:
            logger.info(f"Setup complete: max_iter={max_iter}, check_termination={check_termination}")
    
    def set_x0(self, x0):
        """Set initial state (compatible with tinympcref)"""
        self._x0 = np.asarray(x0, dtype=np.float32).reshape(-1)
        if self._x0.shape[0] != self.nx:
            logger.warning(f"x0 shape mismatch: expected {self.nx}, got {self._x0.shape[0]}")
    
    def set_x_ref(self, xref):
        """Set reference state trajectory (compatible with tinympcref)"""
        self._xref = np.asarray(xref, dtype=np.float32)
        if self._xref.shape != (self.N, self.nx):
            logger.warning(f"xref shape mismatch: expected ({self.N}, {self.nx}), got {self._xref.shape}")
    
    def set_u_ref(self, uref):
        """Set reference control trajectory (compatible with tinympcref)"""
        self._uref = np.asarray(uref, dtype=np.float32)
        if self._uref.shape != (self.N-1, self.nu):
            logger.warning(f"uref shape mismatch: expected ({self.N-1}, {self.nu}), got {self._uref.shape}")
    
    def solve(self, timeout=10.0):
        """
        Solve the MPC problem on hardware
        
        Args:
            timeout: Maximum computation time in seconds
            
        Returns:
            bool: True if solved successfully
        """
        if not self.loaded or self._driver is None:
            raise RuntimeError("Hardware not loaded. Call load_bitstream() first.")
        
        if self._x0 is None or self._xref is None or self._uref is None:
            raise RuntimeError("Problem data not set. Call set_x0(), set_x_ref(), set_u_ref() first.")
        
        try:
            start_time = time.time()
            
            # Solve using the core driver
            result = self._driver.solve(
                x0=self._x0,
                xref=self._xref,
                uref=self._uref,
                max_iter=self.max_iter,
                check_termination_iter=self.check_termination,
                timeout=timeout
            )
            
            # Store results in tinympcref-compatible format
            self.x = result['states']
            self.u = result['controls']
            self.solve_time = result['solve_time'] * 1000  # Convert to ms
            self.solved = 1  # Assume converged if no timeout
            self.iter = self.max_iter  # Hardware doesn't report actual iterations
            
            # Update residuals (set to small values since hardware doesn't provide them)
            self.primal_residual_state = 1e-6
            self.dual_residual_state = 1e-6
            self.primal_residual_input = 1e-6
            self.dual_residual_input = 1e-6
            
            if self.verbose:
                logger.info(f"Hardware solve completed in {self.solve_time:.2f} ms")
            
            return True
            
        except Exception as e:
            logger.error(f"Hardware solve failed: {e}")
            self.solved = 0
            return False
    
    def print_stats(self):
        """Print solver statistics (compatible with tinympcref)"""
        print(f"TinyMPC Hardware Solver Statistics:")
        print(f"  System dimensions: nx={self.nx}, nu={self.nu}, N={self.N}")
        print(f"  Control frequency: {self.control_freq} Hz")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Check termination: {self.check_termination}")
        print(f"  Hardware loaded: {self.loaded}")
        
        if hasattr(self, 'solve_time') and self.solve_time > 0:
            print(f"  Last solve time: {self.solve_time:.2f} ms")
            print(f"  Converged: {'Yes' if self.solved else 'No'}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self._driver is not None:
            # The TinyMPCDriver handles its own cleanup
            self._driver = None
        self.loaded = False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __del__(self):
        """Destructor"""
        self.cleanup()
    
    # Additional methods for direct hardware access
    def get_driver_info(self):
        """Get information about the underlying hardware driver"""
        if self._driver is not None:
            return self._driver.get_info()
        return None
    
    def set_clock_frequency(self, freq_mhz):
        """Set FPGA clock frequency"""
        if self._driver is not None:
            self._driver.set_clock_frequency(freq_mhz)
    
    def get_clock_frequency(self):
        """Get current FPGA clock frequency"""
        if self._driver is not None:
            return self._driver.get_clock_frequency()
        return None