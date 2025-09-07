"""
Hardware Interface Layer for TinyMPC Ultra96 Driver
Provides low-level hardware access using PYNQ
"""

import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class HardwareInterface:
    """Low-level hardware interface for TinyMPC IP core"""
    
    def __init__(self):
        self.overlay = None
        self.ip = None
        self.is_loaded = False
        self.memory_buffer = None
        self.memory_size = 0
        
        # Control register addresses (AXI-Lite interface)
        self.CTRL_REG = 0x00        # Control register
        self.STATUS_REG = 0x04      # Status register  
        self.MAX_ITER_REG = 0x10    # max_iter parameter
        self.CHECK_TERM_REG = 0x18  # check_termination_iter parameter
        self.MEMORY_ADDR_REG = 0x20 # main_memory base address
        
        # Control bits
        self.CTRL_START = 0x01      # Start computation
        self.CTRL_RESET = 0x02      # Reset IP
        self.STATUS_DONE = 0x02     # Computation done
        self.STATUS_IDLE = 0x04     # IP idle
        
    def load_overlay(self, bitstream_path, hwh_path=None):
        """
        Load FPGA overlay from bitstream file
        
        Args:
            bitstream_path: Path to .bit file
            hwh_path: Path to .hwh file (optional, auto-detected if None)
        """
        try:
            from pynq import Overlay
        except ImportError:
            raise ImportError("PYNQ library not found. Please install PYNQ or run on PYNQ-supported hardware.")
        
        try:
            if hwh_path is None:
                # Try to find corresponding .hwh file
                hwh_path = bitstream_path.replace('.bit', '.hwh')
            
            logger.info(f"Loading overlay from {bitstream_path}")
            self.overlay = Overlay(bitstream_path)
            
            # Find TinyMPC IP core
            # Look for IP with 'tinympc' in the name
            tinympc_ips = [ip for ip in self.overlay.ip_dict.keys() 
                          if 'tinympc' in ip.lower()]
            
            if not tinympc_ips:
                raise RuntimeError("No TinyMPC IP found in overlay")
            
            if len(tinympc_ips) > 1:
                logger.warning(f"Multiple TinyMPC IPs found: {tinympc_ips}, using first one")
            
            ip_name = tinympc_ips[0]
            self.ip = getattr(self.overlay, ip_name)
            
            logger.info(f"Found TinyMPC IP: {ip_name}")
            self.is_loaded = True
            
            # Reset IP after loading
            self.reset_ip()
            
        except Exception as e:
            logger.error(f"Failed to load overlay: {e}")
            raise
    
    def extract_parameters_from_overlay(self):
        """
        Extract N, nx, nu parameters from overlay metadata
        
        Returns:
            tuple: (N, nx, nu) parameters
        """
        # Try to extract from IP name or metadata
        # This is a best-effort approach since parameters are hardcoded
        
        # Method 1: Try to parse from IP name or overlay name
        if hasattr(self.overlay, 'metadata'):
            metadata = self.overlay.metadata
            # Look for parameters in metadata
            pass
        
        # Method 2: Try to infer from memory interface
        if self.ip and hasattr(self.ip, 'mmio'):
            # Could read some configuration registers if available
            pass
        
        # Method 3: Default values or user-specified
        # For now, return None to indicate auto-detection failed
        logger.warning("Parameter auto-detection not implemented, user must specify")
        return None, None, None
    
    def setup_memory(self, memory_size, nx, nu, N):
        """
        Allocate and setup physical memory for IP communication
        
        Args:
            memory_size: Total memory size needed
            nx: Number of states
            nu: Number of inputs  
            N: Horizon length
        """
        if self.memory_buffer is not None:
            # Free existing buffer
            del self.memory_buffer
        
        # Allocate physically contiguous memory
        try:
            from pynq import allocate
        except ImportError:
            raise ImportError("PYNQ library not found. Please install PYNQ or run on PYNQ-supported hardware.")
        
        self.memory_buffer = allocate(shape=(memory_size,), dtype=np.float32)
        self.memory_size = memory_size
        
        # Clear memory
        self.memory_buffer[:] = 0.0
        
        # Store parameters for offset calculations
        self.nx = nx
        self.nu = nu  
        self.N = N
        
        # Calculate memory layout offsets
        self.X0_OFFSET = 0
        self.XREF_OFFSET = nx
        self.UREF_OFFSET = nx + N * nx
        self.X_OUT_OFFSET = nx + N * nx + (N-1) * nu
        self.U_OUT_OFFSET = nx + 2 * N * nx + (N-1) * nu
        
        logger.info(f"Allocated {memory_size} float32 elements ({memory_size*4} bytes)")
        
    def write_data_to_memory(self, x0=None, xref=None, uref=None):
        """
        Write input data to memory buffer
        
        Args:
            x0: Initial state (nx,)
            xref: Reference trajectory (N, nx)
            uref: Reference inputs (N-1, nu)
        """
        if self.memory_buffer is None:
            raise RuntimeError("Memory not allocated")
        
        # Write x0 (initial state)
        if x0 is not None:
            if len(x0) != self.nx:
                raise ValueError(f"x0 must have length {self.nx}")
            self.memory_buffer[self.X0_OFFSET:self.X0_OFFSET+self.nx] = x0.astype(np.float32)
        
        # Write xref (reference trajectory)
        if xref is not None:
            if xref.shape != (self.N, self.nx):
                raise ValueError(f"xref must have shape ({self.N}, {self.nx})")
            xref_flat = xref.flatten().astype(np.float32)
            self.memory_buffer[self.XREF_OFFSET:self.XREF_OFFSET+len(xref_flat)] = xref_flat
        
        # Write uref (reference inputs)
        if uref is not None:
            if uref.shape != (self.N-1, self.nu):
                raise ValueError(f"uref must have shape ({self.N-1}, {self.nu})")
            uref_flat = uref.flatten().astype(np.float32)
            self.memory_buffer[self.UREF_OFFSET:self.UREF_OFFSET+len(uref_flat)] = uref_flat
    
    def read_results_from_memory(self):
        """
        Read computation results from memory buffer
        
        Returns:
            tuple: (x_out, u_out) where
                x_out: Optimal state trajectory (N, nx)
                u_out: Optimal control inputs (N-1, nu)
        """
        if self.memory_buffer is None:
            raise RuntimeError("Memory not allocated")
        
        # Read x_out (output state trajectory)
        x_out_flat = self.memory_buffer[self.X_OUT_OFFSET:self.X_OUT_OFFSET+self.N*self.nx]
        x_out = x_out_flat.reshape((self.N, self.nx))
        
        # Read u_out (output control inputs)
        u_out_flat = self.memory_buffer[self.U_OUT_OFFSET:self.U_OUT_OFFSET+(self.N-1)*self.nu]
        u_out = u_out_flat.reshape((self.N-1, self.nu))
        
        return x_out.copy(), u_out.copy()
    
    def write_control_registers(self, max_iter, check_termination_iter):
        """
        Write control parameters to IP registers
        
        Args:
            max_iter: Maximum ADMM iterations
            check_termination_iter: Check termination every N iterations
        """
        if not self.is_loaded:
            raise RuntimeError("Overlay not loaded")
        
        # Write memory base address (physical address of buffer)
        memory_phys_addr = self.memory_buffer.physical_address
        self.ip.write(self.MEMORY_ADDR_REG, memory_phys_addr)
        
        # Write algorithm parameters
        self.ip.write(self.MAX_ITER_REG, max_iter)
        self.ip.write(self.CHECK_TERM_REG, check_termination_iter)
        
        logger.debug(f"Written control registers: max_iter={max_iter}, "
                    f"check_term={check_termination_iter}, "
                    f"memory_addr=0x{memory_phys_addr:x}")
    
    def start_computation(self):
        """Start IP computation"""
        if not self.is_loaded:
            raise RuntimeError("Overlay not loaded")
        
        # Write start bit to control register
        self.ip.write(self.CTRL_REG, self.CTRL_START)
        logger.debug("Started IP computation")
    
    def is_computation_done(self):
        """
        Check if computation is complete
        
        Returns:
            bool: True if computation is done
        """
        if not self.is_loaded:
            return False
        
        status = self.ip.read(self.STATUS_REG)
        return (status & self.STATUS_DONE) != 0
    
    def wait_for_completion(self, timeout=10.0):
        """
        Wait for computation to complete
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            bool: True if completed within timeout, False if timed out
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_computation_done():
                return True
            time.sleep(0.001)  # 1ms polling interval
        
        logger.error(f"Computation timeout after {timeout}s")
        return False
    
    def reset_ip(self):
        """Reset the IP core"""
        if not self.is_loaded:
            return
        
        # Write reset bit
        self.ip.write(self.CTRL_REG, self.CTRL_RESET)
        time.sleep(0.001)  # Short delay
        
        # Clear control register
        self.ip.write(self.CTRL_REG, 0)
        logger.debug("Reset IP core")
    
    def get_execution_time(self):
        """
        Get hardware execution time (if supported by IP)
        
        Returns:
            float: Execution time in milliseconds, or 0 if not supported
        """
        # This would require additional hardware counters in the IP
        # For now, return 0 to indicate not supported
        return 0.0
    
    def cleanup(self):
        """Clean up resources"""
        if self.memory_buffer is not None:
            del self.memory_buffer
            self.memory_buffer = None
        
        self.overlay = None
        self.ip = None
        self.is_loaded = False
        logger.info("Hardware interface cleaned up")
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        self.cleanup()