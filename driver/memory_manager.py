"""
Memory Management for TinyMPC Hardware Driver
Handles memory layout and data organization for IP core communication
"""

import numpy as np
import re
import os
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory layout and data organization for TinyMPC IP"""
    
    def __init__(self):
        self.nx = 0
        self.nu = 0
        self.N = 0
        self.memory_size = 0
        self.precision = 'float'
        
        # Memory layout offsets
        self.X0_OFFSET = 0
        self.XREF_OFFSET = 0
        self.UREF_OFFSET = 0
        self.X_OUT_OFFSET = 0
        self.U_OUT_OFFSET = 0
        
    def infer_parameters_from_bitstream_path(self, bitstream_path):
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
    
    def setup_memory_layout(self, N, nx, nu, precision='float'):
        """
        Setup memory layout based on IP core specification
        
        Args:
            N: Prediction horizon
            nx: Number of states
            nu: Number of inputs
            precision: Data precision ('float' or 'double')
        """
        self.N = N
        self.nx = nx
        self.nu = nu
        self.precision = precision
        
        # Calculate memory offsets (matching tinympc_solver.h)
        self.X0_OFFSET = 0
        self.XREF_OFFSET = nx
        self.UREF_OFFSET = nx + N * nx
        self.X_OUT_OFFSET = nx + N * nx + (N-1) * nu
        self.U_OUT_OFFSET = nx + 2 * N * nx + (N-1) * nu
        
        # Total memory size
        self.memory_size = nx + 2 * N * nx + 2 * (N-1) * nu
        
        logger.info(f"Memory layout setup: N={N}, nx={nx}, nu={nu}")
        logger.info(f"Memory offsets: X0={self.X0_OFFSET}, XREF={self.XREF_OFFSET}, "
                   f"UREF={self.UREF_OFFSET}, X_OUT={self.X_OUT_OFFSET}, U_OUT={self.U_OUT_OFFSET}")
        logger.info(f"Total memory size: {self.memory_size} elements ({self.memory_size*4} bytes)")
    
    def validate_input_data(self, x0=None, xref=None, uref=None):
        """
        Validate input data dimensions
        
        Args:
            x0: Initial state vector
            xref: Reference state trajectory
            uref: Reference input trajectory
            
        Raises:
            ValueError: If data dimensions are incorrect
        """
        if x0 is not None:
            x0 = np.asarray(x0)
            if x0.shape != (self.nx,):
                raise ValueError(f"x0 must have shape ({self.nx},), got {x0.shape}")
        
        if xref is not None:
            xref = np.asarray(xref)
            if xref.shape != (self.N, self.nx):
                raise ValueError(f"xref must have shape ({self.N}, {self.nx}), got {xref.shape}")
        
        if uref is not None:
            uref = np.asarray(uref)
            if uref.shape != (self.N-1, self.nu):
                raise ValueError(f"uref must have shape ({self.N-1}, {self.nu}), got {uref.shape}")
    
    def pack_input_data(self, memory_buffer, x0=None, xref=None, uref=None):
        """
        Pack input data into memory buffer according to IP layout
        
        Args:
            memory_buffer: Target memory buffer
            x0: Initial state vector (nx,)
            xref: Reference state trajectory (N, nx)
            uref: Reference input trajectory (N-1, nu)
        """
        if len(memory_buffer) < self.memory_size:
            raise ValueError(f"Memory buffer too small: {len(memory_buffer)} < {self.memory_size}")
        
        # Validate input data
        self.validate_input_data(x0, xref, uref)
        
        # Pack x0 (initial state)
        if x0 is not None:
            x0 = np.asarray(x0, dtype=np.float32)
            memory_buffer[self.X0_OFFSET:self.X0_OFFSET+self.nx] = x0
            logger.debug(f"Packed x0: {x0}")
        
        # Pack xref (reference trajectory)
        if xref is not None:
            xref = np.asarray(xref, dtype=np.float32)
            xref_flat = xref.flatten()
            end_idx = self.XREF_OFFSET + len(xref_flat)
            memory_buffer[self.XREF_OFFSET:end_idx] = xref_flat
            logger.debug(f"Packed xref: shape {xref.shape}")
        
        # Pack uref (reference inputs)
        if uref is not None:
            uref = np.asarray(uref, dtype=np.float32)
            uref_flat = uref.flatten()
            end_idx = self.UREF_OFFSET + len(uref_flat)
            memory_buffer[self.UREF_OFFSET:end_idx] = uref_flat
            logger.debug(f"Packed uref: shape {uref.shape}")
    
    def unpack_output_data(self, memory_buffer):
        """
        Unpack output data from memory buffer
        
        Args:
            memory_buffer: Source memory buffer
            
        Returns:
            tuple: (x_out, u_out) where
                x_out: Optimal state trajectory (N, nx)
                u_out: Optimal control inputs (N-1, nu)
        """
        if len(memory_buffer) < self.memory_size:
            raise ValueError(f"Memory buffer too small: {len(memory_buffer)} < {self.memory_size}")
        
        # Unpack x_out (output state trajectory)
        x_out_flat = memory_buffer[self.X_OUT_OFFSET:self.X_OUT_OFFSET + self.N * self.nx]
        x_out = x_out_flat.reshape((self.N, self.nx)).copy()
        
        # Unpack u_out (output control inputs)
        u_out_flat = memory_buffer[self.U_OUT_OFFSET:self.U_OUT_OFFSET + (self.N-1) * self.nu]
        u_out = u_out_flat.reshape((self.N-1, self.nu)).copy()
        
        logger.debug(f"Unpacked results: x_out {x_out.shape}, u_out {u_out.shape}")
        return x_out, u_out
    
    def create_test_data(self, test_type='hover'):
        """
        Create test data for validation
        
        Args:
            test_type: Type of test data ('hover', 'tracking', 'random')
            
        Returns:
            tuple: (x0, xref, uref) test data
        """
        if test_type == 'hover':
            # Small perturbation from hover
            x0 = np.zeros(self.nx, dtype=np.float32)
            x0[:3] = [0.01, 0.01, 0.05]  # Small position perturbation
            
            # Zero reference trajectory (hover at origin)
            xref = np.zeros((self.N, self.nx), dtype=np.float32)
            
            # Hover thrust reference
            uref = np.zeros((self.N-1, self.nu), dtype=np.float32)
            if self.nu >= 1:
                uref[:, 0] = 0.5  # Hover thrust
            
        elif test_type == 'tracking':
            # Start with small perturbation
            x0 = np.zeros(self.nx, dtype=np.float32)
            x0[:3] = [0.01, 0.01, 0.05]
            
            # Move to (1,1,1) position
            xref = np.zeros((self.N, self.nx), dtype=np.float32)
            xref[:, :3] = 1.0
            
            # Hover thrust reference
            uref = np.zeros((self.N-1, self.nu), dtype=np.float32)
            if self.nu >= 1:
                uref[:, 0] = 0.5
            
        elif test_type == 'random':
            # Random initial state (small)
            x0 = np.random.randn(self.nx).astype(np.float32) * 0.1
            
            # Random reference trajectory
            xref = np.random.randn(self.N, self.nx).astype(np.float32) * 0.5
            
            # Random reference inputs
            uref = np.random.randn(self.N-1, self.nu).astype(np.float32) * 0.3
            if self.nu >= 1:
                uref[:, 0] += 0.5  # Add hover bias to thrust
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return x0, xref, uref
    
    def print_memory_layout(self):
        """Print memory layout information for debugging"""
        print(f"\n=== TinyMPC Memory Layout ===")
        print(f"Parameters: N={self.N}, nx={self.nx}, nu={self.nu}")
        print(f"Total memory size: {self.memory_size} elements ({self.memory_size*4} bytes)")
        print(f"\nMemory sections:")
        print(f"  X0 (initial state):        [{self.X0_OFFSET:4d}:{self.X0_OFFSET+self.nx:4d}]  ({self.nx} elements)")
        print(f"  XREF (reference states):   [{self.XREF_OFFSET:4d}:{self.XREF_OFFSET+self.N*self.nx:4d}]  ({self.N*self.nx} elements)")
        print(f"  UREF (reference inputs):   [{self.UREF_OFFSET:4d}:{self.UREF_OFFSET+(self.N-1)*self.nu:4d}]  ({(self.N-1)*self.nu} elements)")
        print(f"  X_OUT (output states):     [{self.X_OUT_OFFSET:4d}:{self.X_OUT_OFFSET+self.N*self.nx:4d}]  ({self.N*self.nx} elements)")
        print(f"  U_OUT (output inputs):     [{self.U_OUT_OFFSET:4d}:{self.U_OUT_OFFSET+(self.N-1)*self.nu:4d}]  ({(self.N-1)*self.nu} elements)")
        print(f"===============================\n")
    
    def get_memory_info(self):
        """
        Get memory layout information as dictionary
        
        Returns:
            dict: Memory layout information
        """
        return {
            'N': self.N,
            'nx': self.nx,  
            'nu': self.nu,
            'memory_size': self.memory_size,
            'precision': self.precision,
            'offsets': {
                'x0': self.X0_OFFSET,
                'xref': self.XREF_OFFSET,
                'uref': self.UREF_OFFSET,
                'x_out': self.X_OUT_OFFSET,
                'u_out': self.U_OUT_OFFSET
            },
            'sizes': {
                'x0': self.nx,
                'xref': self.N * self.nx,
                'uref': (self.N-1) * self.nu,
                'x_out': self.N * self.nx,
                'u_out': (self.N-1) * self.nu
            }
        }