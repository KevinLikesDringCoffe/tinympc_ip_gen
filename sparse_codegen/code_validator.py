"""
Unified C code validator with compilation and wrapper functionality
"""

import ctypes
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import tempfile
import subprocess
import os
import time


@dataclass
class ValidationResult:
    """Result of code validation"""
    passed: bool
    max_error: float
    avg_error: float
    c_output: np.ndarray
    python_output: np.ndarray
    c_time: float
    python_time: float
    error_details: Optional[str] = None


class CodeValidator:
    """All-in-one C code validator with compilation and testing"""
    
    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance
        self.lib = None
        self.lib_path = None
    
    def validate(self, c_code: str, A: np.ndarray, B: np.ndarray, Kinf: np.ndarray, 
                N: int, seed: int = 42) -> ValidationResult:
        """
        Validate generated C code against Python reference
        
        Args:
            c_code: Generated C code fragment
            A: State matrix (nx x nx)
            B: Input matrix (nx x nu)
            Kinf: Feedback gain (nu x nx)
            N: Horizon length
            seed: Random seed for test data
        
        Returns:
            ValidationResult with comparison details
        """
        # Generate test data
        nx, nu = A.shape[0], B.shape[1]
        test_data = self._generate_test_data(nx, nu, N, seed)
        
        # Wrap and compile C code
        full_c_code = self._wrap_c_code(c_code, nx, nu, N)
        if not self._compile(full_c_code):
            return ValidationResult(
                passed=False, max_error=float('inf'), avg_error=float('inf'),
                c_output=np.array([]), python_output=np.array([]),
                c_time=0, python_time=0, error_details="Compilation failed"
            )
        
        # Execute C code
        c_output, c_time = self._execute_c_code(test_data, nx, nu, N)
        if c_output is None:
            self._cleanup()
            return ValidationResult(
                passed=False, max_error=float('inf'), avg_error=float('inf'),
                c_output=np.array([]), python_output=np.array([]),
                c_time=0, python_time=0, error_details="C execution failed"
            )
        
        # Execute Python reference
        python_output, python_time = self._execute_python_code(test_data, A, B, Kinf)
        
        # Compare results
        max_error, avg_error = self._compare_outputs(c_output, python_output)
        passed = max_error < self.tolerance
        
        # Cleanup
        self._cleanup()
        
        return ValidationResult(
            passed=passed, max_error=max_error, avg_error=avg_error,
            c_output=c_output, python_output=python_output,
            c_time=c_time, python_time=python_time,
            error_details=None if passed else f"Max error {max_error:.2e} exceeds tolerance {self.tolerance:.2e}"
        )
    
    def _generate_test_data(self, nx: int, nu: int, N: int, seed: int) -> Dict[str, np.ndarray]:
        """Generate random test data"""
        np.random.seed(seed)
        return {
            'x0': np.random.randn(nx).astype(np.float32),
            'xref': np.random.randn(N, nx).astype(np.float32),
            'd': np.random.randn(N-1, nu).astype(np.float32),
            'u': np.zeros((N-1, nu), dtype=np.float32),
            'x': np.zeros((N, nx), dtype=np.float32)
        }
    
    def _wrap_c_code(self, c_code: str, nx: int, nu: int, N: int) -> str:
        """Wrap generated C code with headers and interface functions"""
        lines = [
            "#include <stdio.h>",
            "#include <stdlib.h>", 
            "#include <math.h>",
            "",
            f"#define NX {nx}",
            f"#define NU {nu}",
            f"#define N {N}",
            "",
            "// Global arrays",
            f"float x[{N}][{nx}];",
            f"float u[{N-1}][{nu}];", 
            f"float d[{N-1}][{nu}];",
            f"float xref[{N}][{nx}];",
            "",
            "// Algorithm implementation",
            "void forward_pass_impl() {",
            self._extract_algorithm_body(c_code),
            "}",
            "",
            "// External interface",
            "void forward_pass(float* x0, float* xref_in, float* d_in, float* x_out, float* u_out) {",
            "    // Copy inputs",
            f"    for (int i = 0; i < {nx}; i++) x[0][i] = x0[i];",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++) xref[k][i] = xref_in[k * {nx} + i];",
            f"    for (int k = 0; k < {N-1}; k++)",
            f"        for (int i = 0; i < {nu}; i++) d[k][i] = d_in[k * {nu} + i];",
            "",
            "    // Execute algorithm", 
            "    forward_pass_impl();",
            "",
            "    // Copy outputs",
            f"    for (int k = 0; k < {N}; k++)",
            f"        for (int i = 0; i < {nx}; i++) x_out[k * {nx} + i] = x[k][i];",
            f"    for (int k = 0; k < {N-1}; k++)",
            f"        for (int i = 0; i < {nu}; i++) u_out[k * {nu} + i] = u[k][i];",
            "}"
        ]
        return '\n'.join(lines)
    
    def _extract_algorithm_body(self, code: str) -> str:
        """Extract algorithm body from generated code"""
        lines = code.split('\n')
        body_lines = []
        temp_vars = []
        
        for line in lines:
            if line.startswith('//') or not line.strip():
                continue
            if line.strip().startswith('float temp_'):
                temp_vars.append('    ' + line.strip())
                continue
            if line.strip().startswith('float ') or line.strip().startswith('#'):
                continue
            if line.strip():
                body_lines.append('    ' + line)
        
        result = []
        if temp_vars:
            result.extend(temp_vars)
            result.append('')
        result.extend(body_lines)
        return '\n'.join(result) if result else '    // Empty'
    
    def _compile(self, c_code: str) -> bool:
        """Compile C code using GCC"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(c_code)
                c_file = f.name
            
            self.lib_path = tempfile.mktemp(suffix='.so')
            cmd = ['gcc', '-shared', '-fPIC', '-O2', '-o', self.lib_path, c_file, '-lm']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            os.unlink(c_file)
            
            if result.returncode == 0:
                self.lib = ctypes.CDLL(self.lib_path)
                return True
            else:
                return False
        except Exception:
            return False
    
    def _execute_c_code(self, test_data: Dict[str, np.ndarray], nx: int, nu: int, N: int) -> Tuple[Optional[np.ndarray], float]:
        """Execute compiled C code"""
        try:
            # Prepare arrays
            x_out = np.zeros((N, nx), dtype=np.float32)
            u_out = np.zeros((N-1, nu), dtype=np.float32)
            
            # Flatten input arrays for C function
            x0_flat = test_data['x0'].astype(np.float32)
            xref_flat = test_data['xref'].astype(np.float32).flatten()
            d_flat = test_data['d'].astype(np.float32).flatten()
            x_out_flat = x_out.flatten()
            u_out_flat = u_out.flatten()
            
            # Convert to ctypes pointers
            ptrs = [arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for arr in [
                x0_flat, xref_flat, d_flat, x_out_flat, u_out_flat
            ]]
            
            # Get and call function
            func = getattr(self.lib, 'forward_pass')
            func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 5
            func.restype = None
            
            start_time = time.perf_counter()
            func(*ptrs)
            end_time = time.perf_counter()
            
            return np.concatenate([x_out_flat, u_out_flat]), end_time - start_time
            
        except Exception as e:
            print(f"C execution error: {e}")
            return None, 0
    
    def _execute_python_code(self, test_data: Dict[str, np.ndarray], A: np.ndarray, 
                           B: np.ndarray, Kinf: np.ndarray) -> Tuple[np.ndarray, float]:
        """Execute Python reference implementation"""
        start_time = time.perf_counter()
        
        x0 = test_data['x0'].astype(np.float32)
        d = test_data['d'].astype(np.float32)
        N = d.shape[0] + 1
        
        # Forward pass algorithm
        x = np.zeros((N, A.shape[0]), dtype=np.float32)
        u = np.zeros((N-1, B.shape[1]), dtype=np.float32)
        x[0] = x0
        
        for i in range(N - 1):
            u[i] = -Kinf @ x[i] - d[i]
            x[i+1] = A @ x[i] + B @ u[i]
        
        end_time = time.perf_counter()
        return np.concatenate([x.flatten(), u.flatten()]), end_time - start_time
    
    def _compare_outputs(self, c_output: np.ndarray, python_output: np.ndarray) -> Tuple[float, float]:
        """Compare C and Python outputs"""
        if c_output.shape != python_output.shape:
            min_size = min(c_output.size, python_output.size)
            c_output = c_output.flatten()[:min_size]
            python_output = python_output.flatten()[:min_size]
        
        errors = np.abs(c_output - python_output)
        return float(np.max(errors)), float(np.mean(errors))
    
    def _cleanup(self):
        """Clean up temporary files"""
        if self.lib_path and os.path.exists(self.lib_path):
            try:
                os.unlink(self.lib_path)
            except:
                pass


def validate_code(A: np.ndarray, B: np.ndarray, Kinf: np.ndarray, N: int,
                 c_code: str, tolerance: float = 1e-4, seed: int = 42) -> bool:
    """
    Simple validation function
    
    Args:
        A: State matrix (nx x nx)
        B: Input matrix (nx x nu) 
        Kinf: Feedback gain (nu x nx)
        N: Horizon length
        c_code: Generated C code
        tolerance: Error tolerance
        seed: Random seed
    
    Returns:
        True if validation passed
    """
    validator = CodeValidator(tolerance)
    result = validator.validate(c_code, A, B, Kinf, N, seed)
    
    if result.passed:
        speedup = result.python_time / result.c_time if result.c_time > 0 else 0
        print(f"PASSED: Max error {result.max_error:.2e}, Speedup {speedup:.1f}x")
    else:
        print(f"FAILED: Max error {result.max_error:.2e}")
        if result.error_details:
            print(f"  Details: {result.error_details}")
    
    return result.passed