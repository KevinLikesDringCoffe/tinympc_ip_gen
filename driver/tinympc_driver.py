"""
PYNQ Driver for TinyMPC Custom IP

This module provides low-level hardware interface for TinyMPC IP core
including memory management, register control, and data transfer.

This driver implementation matches the register interface and control flow
from run_hls_ip() in pynq_tinympc_test.py for maximum compatibility with
the HLS-generated IP core.

Version 2.0.0 - Restructured to match run_hls_ip register layout:
- Control register (0x00): Combined status/control bits
- Memory address split into low/high 32-bit registers (0x10/0x14)
- Max iterations parameter at 0x1C
- Polling-based completion detection with proper bit checking
"""

import numpy as np
import time
import warnings

# Import PYNQ - required for hardware acceleration
try:
    from pynq import Overlay, allocate
    from pynq.ps import Clocks
except ImportError as e:
    raise ImportError(
        "PYNQ library is required for hardware MPC solver. "
        "Please install PYNQ or use software MPC solver instead. "
        f"Original error: {e}"
    ) from e


class TinyMPCDriver:
    """
    Low-level PYNQ driver for TinyMPC custom IP.
    
    This driver handles:
    - IP core initialization and control
    - Memory allocation and data transfer
    - Hardware register access
    - Interrupt handling
    """
    
    # HLS IP register offsets (matches run_hls_ip from pynq_tinympc_test.py)
    CTRL_REG_OFFSET = 0x00      # Control register (bit 0: start, bit 1: done, bit 2: idle, bit 3: ready)
    GIER_REG_OFFSET = 0x04      # Global interrupt enable
    IP_IER_REG_OFFSET = 0x08    # IP interrupt enable  
    IP_ISR_REG_OFFSET = 0x0C    # IP interrupt status
    MEM_ADDR_LOW_OFFSET = 0x10  # main_memory_V (lower 32 bits of pointer)
    MEM_ADDR_HIGH_OFFSET = 0x14 # main_memory_V (upper 32 bits of pointer) 
    MAX_ITER_REG_OFFSET = 0x1C  # max_iter_V (32-bit integer)
    CHECK_TERMINATION_ITER_REG_OFFSET = 0x20  # check_termination_iter_V (32-bit integer)
    
    # Control register bits
    CTRL_START_BIT = 0x01       # Start computation
    CTRL_DONE_BIT = 0x02        # Computation done
    CTRL_IDLE_BIT = 0x04        # IP core idle
    CTRL_READY_BIT = 0x08       # Ready for new computation
    
    def __init__(self, overlay_path=None, ip_name="tinympc_hls_wrapper_0", nx=12, nu=4, n_horizon=10, clock_frequency_mhz=250):
        """
        Initialize TinyMPC driver.
        
        Args:
            overlay_path (str): Path to the bitstream overlay file (.bit)
            ip_name (str): Name of the TinyMPC IP in the overlay
            nx (int): Number of states
            nu (int): Number of controls
            n_horizon (int): MPC horizon length
            clock_frequency_mhz (int): FPGA clock frequency in MHz (default: 250)
        """
        # Configurable hardware constants
        self.NX = nx
        self.NU = nu  
        self.N_HORIZON = n_horizon
        self.clock_frequency_mhz = clock_frequency_mhz
        
        # Calculate memory layout offsets based on problem dimensions
        self.MEM_X0_OFFSET = 0
        self.MEM_XREF_OFFSET = self.MEM_X0_OFFSET + self.NX
        self.MEM_UREF_OFFSET = self.MEM_XREF_OFFSET + (self.N_HORIZON * self.NX)
        self.MEM_U_OUT_OFFSET = self.MEM_UREF_OFFSET + ((self.N_HORIZON - 1) * self.NU)
        self.MEM_X_OUT_OFFSET = self.MEM_U_OUT_OFFSET + ((self.N_HORIZON - 1) * self.NU)
        
        # Total memory size calculation
        self.TOTAL_MEM_SIZE = self.MEM_X_OUT_OFFSET + (self.N_HORIZON * self.NX)
        
        self.overlay = None
        self.ip_core = None
        self.memory_buffer = None
        self.ip_name = ip_name
        
        if overlay_path:
            self.load_overlay(overlay_path)
            
    def load_overlay(self, overlay_path):
        """
        Load FPGA overlay and initialize IP core.
        
        Args:
            overlay_path (str): Path to bitstream overlay file
        """
        try:
            print(f"Loading overlay from: {overlay_path}")
            self.overlay = Overlay(overlay_path)
            
            # Set clock frequency
            self._set_clock_frequency()
            
            # Get reference to TinyMPC IP core
            if hasattr(self.overlay, self.ip_name):
                self.ip_core = getattr(self.overlay, self.ip_name)
                print(f"Found TinyMPC IP core: {self.ip_name}")
            else:
                available_ips = [name for name in dir(self.overlay) if not name.startswith('_')]
                raise RuntimeError(f"TinyMPC IP '{self.ip_name}' not found in overlay. "
                                 f"Available IPs: {available_ips}")
            
            # Allocate contiguous memory buffer for data transfer
            self._allocate_memory()
            
            # Initialize IP core
            self._initialize_ip()
            
            print("TinyMPC driver initialized successfully")
            
        except Exception as e:
            print(f"Error loading overlay: {e}")
            raise
            
    def _set_clock_frequency(self):
        """Set FPGA clock frequency for optimal performance."""
        try:
            current_freq = Clocks.fclk0_mhz
            if current_freq != self.clock_frequency_mhz:
                print(f"Setting FCLK0 frequency from {current_freq} MHz to {self.clock_frequency_mhz} MHz")
                Clocks.fclk0_mhz = self.clock_frequency_mhz
                print(f"Clock frequency set to {Clocks.fclk0_mhz} MHz")
            else:
                print(f"FCLK0 frequency already set to {current_freq} MHz")
        except Exception as e:
            print(f"Warning: Could not set clock frequency: {e}")
            print(f"Continuing with current clock frequency")
            
    def _allocate_memory(self):
        """Allocate contiguous memory buffer for IP communication."""
        try:
            # Allocate physically contiguous memory buffer
            self.memory_buffer = allocate(shape=(self.TOTAL_MEM_SIZE,), dtype=np.float32, cacheable=False)
            print(f"Allocated {self.TOTAL_MEM_SIZE} float32 memory buffer at 0x{self.memory_buffer.physical_address:08x}")
            
            # Clear memory buffer
            self.memory_buffer[:] = 0.0
            
        except Exception as e:
            print(f"Error allocating memory: {e}")
            raise
            
    def _initialize_ip(self):
        """Initialize IP core registers (matches run_hls_ip implementation)."""
        try:
            # Clear control register first
            if hasattr(self.ip_core, 'write'):
                self.ip_core.write(self.CTRL_REG_OFFSET, 0x00)
            else:
                # Alternative register access method
                self.ip_core.register_map.ctrl = 0x00
            
            # Set memory address (split into 32-bit high/low parts)
            phys_addr = self.memory_buffer.physical_address
            addr_low = phys_addr & 0xFFFFFFFF
            addr_high = (phys_addr >> 32) & 0xFFFFFFFF
            
            if hasattr(self.ip_core, 'write'):
                self.ip_core.write(self.MEM_ADDR_LOW_OFFSET, addr_low)
                self.ip_core.write(self.MEM_ADDR_HIGH_OFFSET, addr_high)
            else:
                # Alternative register access method
                self.ip_core.register_map.main_memory_V = addr_low
                self.ip_core.register_map.main_memory_V_1 = addr_high
                
            print(f"IP core initialized with memory address: 0x{phys_addr:016x}")
            print(f"  Lower 32 bits: 0x{addr_low:08x}")
            print(f"  Upper 32 bits: 0x{addr_high:08x}")
            
        except Exception as e:
            print(f"Error initializing IP core: {e}")
            raise
            
    def reset(self):
        """Reset the IP core by clearing control register."""
        try:
            if hasattr(self.ip_core, 'write'):
                self.ip_core.write(self.CTRL_REG_OFFSET, 0x00)
            else:
                self.ip_core.register_map.ctrl = 0x00
                
        except Exception as e:
            print(f"Warning: Could not reset IP core: {e}")
            
    def is_ready(self):
        """Check if IP core is ready for computation (matches run_hls_ip)."""
        try:
            if hasattr(self.ip_core, 'read'):
                ctrl_reg = self.ip_core.read(self.CTRL_REG_OFFSET)
            else:
                ctrl_reg = self.ip_core.register_map.ctrl
                
            return bool(ctrl_reg & self.CTRL_READY_BIT)
        except:
            return True  # Assume ready if can't read status
            
    def is_done(self):
        """Check if computation is complete (matches run_hls_ip)."""
        try:
            if hasattr(self.ip_core, 'read'):
                ctrl_reg = self.ip_core.read(self.CTRL_REG_OFFSET)
            else:
                ctrl_reg = self.ip_core.register_map.ctrl
                
            return bool(ctrl_reg & self.CTRL_DONE_BIT)
        except:
            return True  # Assume done if can't read status
            
    def is_idle(self):
        """Check if IP core is idle."""
        try:
            if hasattr(self.ip_core, 'read'):
                ctrl_reg = self.ip_core.read(self.CTRL_REG_OFFSET)
            else:
                ctrl_reg = self.ip_core.register_map.ctrl
                
            return bool(ctrl_reg & self.CTRL_IDLE_BIT)
        except:
            return True  # Assume idle if can't read status
            
    def get_control_status(self):
        """Get detailed control register status for debugging."""
        try:
            if hasattr(self.ip_core, 'read'):
                ctrl_reg = self.ip_core.read(self.CTRL_REG_OFFSET)
            else:
                ctrl_reg = self.ip_core.register_map.ctrl
                
            return {
                'raw_value': f"0x{ctrl_reg:08x}",
                'ready': bool(ctrl_reg & self.CTRL_READY_BIT),
                'idle': bool(ctrl_reg & self.CTRL_IDLE_BIT),
                'done': bool(ctrl_reg & self.CTRL_DONE_BIT),
                'start': bool(ctrl_reg & self.CTRL_START_BIT)
            }
        except Exception as e:
            return {'error': str(e)}
            
    def set_max_iterations(self, max_iter):
        """
        Set maximum iterations parameter.
        
        Args:
            max_iter (int): Maximum number of ADMM iterations
        """
        try:
            if hasattr(self.ip_core, 'write'):
                self.ip_core.write(self.MAX_ITER_REG_OFFSET, int(max_iter))
            else:
                self.ip_core.register_map.max_iter = int(max_iter)
                
        except Exception as e:
            print(f"Warning: Could not set max_iter: {e}")
            
    def set_check_termination_iter(self, check_termination_iter):
        """
        Set check termination iterations parameter.
        
        Args:
            check_termination_iter (int): Number of iterations between termination checks
        """
        try:
            if hasattr(self.ip_core, 'write'):
                self.ip_core.write(self.CHECK_TERMINATION_ITER_REG_OFFSET, int(check_termination_iter))
            else:
                self.ip_core.register_map.check_termination_iter = int(check_termination_iter)
                
        except Exception as e:
            print(f"Warning: Could not set check_termination_iter: {e}")
            
    def set_clock_frequency(self, frequency_mhz):
        """
        Set FPGA clock frequency.
        
        Args:
            frequency_mhz (int): Clock frequency in MHz
        """
        self.clock_frequency_mhz = frequency_mhz
        try:
            current_freq = Clocks.fclk0_mhz
            print(f"Changing FCLK0 frequency from {current_freq} MHz to {frequency_mhz} MHz")
            Clocks.fclk0_mhz = frequency_mhz
            print(f"Clock frequency updated to {Clocks.fclk0_mhz} MHz")
        except Exception as e:
            print(f"Warning: Could not set clock frequency: {e}")
            
    def get_clock_frequency(self):
        """
        Get current FPGA clock frequency.
        
        Returns:
            int: Current clock frequency in MHz
        """
        try:
            return Clocks.fclk0_mhz
        except:
            return self.clock_frequency_mhz  # Return stored value if can't read actual
            
    def load_input_data(self, x0, xref, uref):
        """
        Load input data into memory buffer.
        
        Args:
            x0 (np.array): Initial state [nx]
            xref (np.array): Reference state trajectory [N, nx]  
            uref (np.array): Reference control trajectory [N-1, nu]
        """
        if self.memory_buffer is None:
            raise RuntimeError("Memory buffer not allocated")
            
        # Validate input dimensions
        x0 = np.asarray(x0, dtype=np.float32).flatten()
        xref = np.asarray(xref, dtype=np.float32)
        uref = np.asarray(uref, dtype=np.float32)
        
        if x0.shape[0] != self.NX:
            raise ValueError(f"x0 must have shape ({self.NX},), got {x0.shape}")
        if xref.shape != (self.N_HORIZON, self.NX):
            raise ValueError(f"xref must have shape ({self.N_HORIZON}, {self.NX}), got {xref.shape}")
        if uref.shape != (self.N_HORIZON - 1, self.NU):
            raise ValueError(f"uref must have shape ({self.N_HORIZON - 1}, {self.NU}), got {uref.shape}")
            
        # Load data into memory buffer
        # Initial state
        self.memory_buffer[self.MEM_X0_OFFSET:self.MEM_X0_OFFSET + self.NX] = x0
        
        # Reference trajectory (flatten row-wise)
        xref_flat = xref.flatten()
        self.memory_buffer[self.MEM_XREF_OFFSET:self.MEM_XREF_OFFSET + len(xref_flat)] = xref_flat
        
        # Reference controls (flatten row-wise)
        uref_flat = uref.flatten()
        self.memory_buffer[self.MEM_UREF_OFFSET:self.MEM_UREF_OFFSET + len(uref_flat)] = uref_flat
        
        # Flush cache to ensure data is written to memory
        self.memory_buffer.flush()
        
    def get_output_data(self):
        """
        Retrieve output data from memory buffer.
        
        Returns:
            tuple: (states, controls)
                - states (np.array): Optimal state trajectory [N, nx]
                - controls (np.array): Optimal control trajectory [N-1, nu]
        """
        if self.memory_buffer is None:
            raise RuntimeError("Memory buffer not allocated")
            
        # Invalidate cache to ensure fresh data is read
        self.memory_buffer.invalidate()
        
        # Extract output states
        x_out_start = self.MEM_X_OUT_OFFSET
        x_out_end = x_out_start + (self.N_HORIZON * self.NX)
        x_out_flat = self.memory_buffer[x_out_start:x_out_end]
        x_out = x_out_flat.reshape((self.N_HORIZON, self.NX))
        
        # Extract output controls
        u_out_start = self.MEM_U_OUT_OFFSET
        u_out_end = u_out_start + ((self.N_HORIZON - 1) * self.NU)
        u_out_flat = self.memory_buffer[u_out_start:u_out_end]
        u_out = u_out_flat.reshape((self.N_HORIZON - 1, self.NU))
        
        return x_out.copy(), u_out.copy()
        
    def start_computation(self):
        """Start MPC computation on hardware (matches run_hls_ip)."""
        try:
            # Check initial status for debugging
            status = self.get_control_status()
            print(f"Before start - Control: {status['raw_value']} (ready: {status['ready']}, idle: {status['idle']})")
            
            # Start the IP by writing to control register
            if hasattr(self.ip_core, 'write'):
                self.ip_core.write(self.CTRL_REG_OFFSET, self.CTRL_START_BIT)
            else:
                self.ip_core.register_map.ctrl = self.CTRL_START_BIT
                
            print("IP started...")
                
        except Exception as e:
            print(f"Error starting computation: {e}")
            raise
            
    def wait_for_completion(self, timeout=10.0):
        """
        Wait for computation to complete (matches run_hls_ip polling logic).
        
        Args:
            timeout (float): Maximum wait time in seconds
            
        Returns:
            bool: True if completed successfully, False if timeout
        """
        timeout_counter = int(timeout * 1000000)  # Convert to microsecond counter
        poll_count = 0
        ctrl_reg = 0  # Initialize to avoid undefined variable
        
        while timeout_counter > 0:
            try:
                ctrl_reg = self.ip_core.read(self.CTRL_REG_OFFSET) if hasattr(self.ip_core, 'read') else self.ip_core.register_map.ctrl
                poll_count += 1
                
                if ctrl_reg & self.CTRL_DONE_BIT:  # Check done bit
                    print(f"IP completed after {poll_count} polls")
                    
                    # Clear the done bit by writing to control register (matches run_hls_ip)
                    if hasattr(self.ip_core, 'write'):
                        self.ip_core.write(self.CTRL_REG_OFFSET, 0x00)
                    else:
                        self.ip_core.register_map.ctrl = 0x00
                        
                    return True
                    
                timeout_counter -= 1
                
            except Exception as e:
                print(f"Error reading control register: {e}")
                return False
            
        print(f"Timeout after {poll_count} polls, last ctrl_reg: 0x{ctrl_reg:08x}")
        return False
        
    def solve(self, x0, xref, uref, max_iter=100, check_termination_iter=None, timeout=10.0):
        """
        High-level solve function combining all steps (matches run_hls_ip workflow).
        
        Args:
            x0 (np.array): Initial state [nx]
            xref (np.array): Reference state trajectory [N, nx]
            uref (np.array): Reference control trajectory [N-1, nu]
            max_iter (int): Maximum ADMM iterations
            check_termination_iter (int, optional): Number of iterations between termination checks.
                                                   If None, defaults to max_iter (no early termination)
            timeout (float): Computation timeout in seconds
            
        Returns:
            dict: Solution containing 'states' and 'controls'
        """
        start_time = time.time()
        
        # Set default check_termination_iter to max_iter if not specified (no early termination)
        if check_termination_iter is None:
            check_termination_iter = max_iter
        
        try:
            # Step 1: Set parameters
            self.set_max_iterations(max_iter)
            self.set_check_termination_iter(check_termination_iter)
            print(f"Parameters written: max_iter={max_iter}, check_termination_iter={check_termination_iter}")
            
            # Step 2: Load input data to memory buffer
            self.load_input_data(x0, xref, uref)
            
            # Step 3: Start computation
            self.start_computation()
            
            # Step 4: Wait for completion with polling
            if not self.wait_for_completion(timeout):
                raise RuntimeError(f"Hardware execution timeout after {timeout}s")
                
            # Step 5: Retrieve results
            states, controls = self.get_output_data()
            
            execution_time = time.time() - start_time
            print(f"Hardware execution completed in {execution_time:.4f} seconds")
            
            return {
                'states': states,
                'controls': controls,
                'states_all': states,  # For compatibility with TinyMPC
                'controls_all': controls,
                'solve_time': execution_time,
                'max_iter': max_iter,
                'check_termination_iter': check_termination_iter
            }
            
        except Exception as e:
            print(f"Error in solve: {e}")
            # Try alternative register layout if the first one fails (matches run_hls_ip)
            try:
                print("Trying alternative register layout...")
                
                # Some HLS versions use different offsets
                phys_addr = self.memory_buffer.physical_address
                if hasattr(self.ip_core, 'write'):
                    self.ip_core.write(0x10, phys_addr & 0xFFFFFFFF)
                    self.ip_core.write(0x14, (phys_addr >> 32) & 0xFFFFFFFF)
                    self.ip_core.write(0x18, max_iter)  # Try offset 0x18 instead of 0x1C
                    self.ip_core.write(0x20, check_termination_iter)  # Try offset 0x20 for check_termination_iter
                    
                    self.ip_core.write(0x00, 0x01)
                    
                    timeout_counter = int(timeout * 1000000)
                    while timeout_counter > 0:
                        ctrl_reg = self.ip_core.read(0x00)
                        if ctrl_reg & 0x02:
                            break
                        timeout_counter -= 1
                        
                    if timeout_counter == 0:
                        raise RuntimeError("Hardware execution timeout")
                        
                    self.ip_core.write(0x00, 0x00)
                    
                    # Retrieve results
                    states, controls = self.get_output_data()
                    execution_time = time.time() - start_time
                    
                    return {
                        'states': states,
                        'controls': controls,
                        'states_all': states,
                        'controls_all': controls,
                        'solve_time': execution_time,
                        'max_iter': max_iter,
                        'check_termination_iter': check_termination_iter
                    }
                    
                else:
                    raise RuntimeError("Alternative register access not supported")
                    
            except Exception as e2:
                raise RuntimeError(f"Both register layouts failed: {e}, {e2}")
        
    def __del__(self):
        """Cleanup resources."""
        if self.memory_buffer is not None:
            self.memory_buffer.freebuffer()
            
    def get_info(self):
        """Get driver and hardware information."""
        info = {
            'driver_version': '2.0.0',  # Updated version to reflect run_hls_ip compatibility
            'nx': self.NX,
            'nu': self.NU,
            'horizon': self.N_HORIZON,
            'memory_size': self.TOTAL_MEM_SIZE,
            'overlay_loaded': self.overlay is not None,
            'ip_core_found': self.ip_core is not None,
            'memory_allocated': self.memory_buffer is not None,
            'register_layout': 'HLS IP AXI4-Lite (matches run_hls_ip)',
            'control_register': f"0x{self.CTRL_REG_OFFSET:02x}",
            'memory_addr_low': f"0x{self.MEM_ADDR_LOW_OFFSET:02x}",
            'memory_addr_high': f"0x{self.MEM_ADDR_HIGH_OFFSET:02x}",
            'max_iter_register': f"0x{self.MAX_ITER_REG_OFFSET:02x}",
            'check_termination_iter_register': f"0x{self.CHECK_TERMINATION_ITER_REG_OFFSET:02x}",
            'clock_frequency_mhz': self.get_clock_frequency(),
            'target_clock_frequency_mhz': self.clock_frequency_mhz,
        }
        
        if self.memory_buffer is not None:
            info['memory_address'] = f"0x{self.memory_buffer.physical_address:016x}"
            info['memory_address_low'] = f"0x{self.memory_buffer.physical_address & 0xFFFFFFFF:08x}"
            info['memory_address_high'] = f"0x{(self.memory_buffer.physical_address >> 32) & 0xFFFFFFFF:08x}"
            
        # Add current control status if IP is available
        if self.ip_core is not None:
            info['control_status'] = self.get_control_status()
            
        return info