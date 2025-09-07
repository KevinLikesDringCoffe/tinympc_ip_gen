"""
TinyMPC Hardware Driver for Ultra96
Provides tinympcref-compatible interface for FPGA acceleration
"""

from .tinympc_hw import tinympc_hw, TinyMPCHW
from .memory_manager import MemoryManager
from .hw_interface import HardwareInterface

__version__ = "1.0.0"
__author__ = "TinyMPC Hardware Driver"

__all__ = [
    'tinympc_hw',
    'TinyMPCHW', 
    'MemoryManager',
    'HardwareInterface'
]