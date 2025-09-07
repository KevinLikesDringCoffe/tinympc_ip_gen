"""
TinyMPC Hardware Driver for Ultra96
Provides tinympcref-compatible interface for FPGA acceleration
"""

from .tinympc_driver import TinyMPCDriver
from .tinympc_hw import tinympc_hw

__version__ = "2.0.0"
__author__ = "TinyMPC Hardware Driver"

__all__ = [
    'TinyMPCDriver',    # Core hardware driver
    'tinympc_hw',       # Compatibility wrapper
]