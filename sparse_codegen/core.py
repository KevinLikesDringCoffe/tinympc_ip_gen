from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np


class DataType(Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"


@dataclass
class Variable:
    name: str
    dtype: DataType
    dims: List[int]
    is_sparse: bool = False
    sparsity_pattern: Optional[Dict[Tuple[int, int], float]] = None
    is_constant: bool = False
    
    def __repr__(self):
        return f"Variable({self.name}, {self.dtype.value}, dims={self.dims}, sparse={self.is_sparse})"


@dataclass  
class IndexExpr:
    base: str
    offset: int = 0
    scale: int = 1
    
    def __str__(self):
        if self.offset == 0 and self.scale == 1:
            return self.base
        elif self.scale == 1:
            if self.offset > 0:
                return f"{self.base}+{self.offset}"
            else:
                return f"{self.base}{self.offset}"
        else:
            if self.offset == 0:
                return f"{self.scale}*{self.base}"
            elif self.offset > 0:
                return f"{self.scale}*{self.base}+{self.offset}"
            else:
                return f"{self.scale}*{self.base}{self.offset}"
    
    def __repr__(self):
        return str(self)
    
    def substitute(self, value: int) -> int:
        return self.scale * value + self.offset
    
    @staticmethod
    def constant(value: int) -> 'IndexExpr':
        return IndexExpr("", offset=value, scale=0)


def extract_sparsity_pattern(matrix: np.ndarray, threshold: float = 1e-10) -> Dict[Tuple[int, int], float]:
    pattern = {}
    
    if matrix.ndim == 1:
        # Handle 1D arrays (diagonal matrices)
        for i in range(matrix.shape[0]):
            if abs(matrix[i]) > threshold:
                pattern[(i, 0)] = float(matrix[i])
    else:
        # Handle 2D matrices
        rows, cols = matrix.shape
        for i in range(rows):
            for j in range(cols):
                if abs(matrix[i, j]) > threshold:
                    pattern[(i, j)] = float(matrix[i, j])
    return pattern


def get_sparsity_info(matrix: np.ndarray, threshold: float = 1e-10) -> Dict[str, Any]:
    total_elements = matrix.size
    nonzero_elements = np.sum(np.abs(matrix) > threshold)
    sparsity_ratio = 1.0 - (nonzero_elements / total_elements)
    
    return {
        'total_elements': total_elements,
        'nonzero_elements': nonzero_elements,
        'sparsity_ratio': sparsity_ratio,
        'shape': matrix.shape,
        'pattern': extract_sparsity_pattern(matrix, threshold)
    }