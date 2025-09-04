"""Essential AST nodes for forward pass generation"""

from dataclasses import dataclass
from typing import List, Optional, Union, Any
from abc import ABC, abstractmethod
from .core import Variable, IndexExpr


class ASTNode(ABC):
    @abstractmethod
    def accept(self, visitor):
        pass


@dataclass
class BinaryOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_unary_op(self)


@dataclass
class ArrayAccess(ASTNode):
    array: str
    indices: List[Union[int, IndexExpr, str]]
    
    def accept(self, visitor):
        return visitor.visit_array_access(self)


@dataclass
class Assignment(ASTNode):
    target: ASTNode
    value: ASTNode
    accumulate: bool = False
    
    def accept(self, visitor):
        return visitor.visit_assignment(self)


@dataclass
class Loop(ASTNode):
    var: str
    start: Union[int, str]
    end: Union[int, str]
    body: List[ASTNode]
    step: int = 1
    
    def accept(self, visitor):
        return visitor.visit_loop(self)


@dataclass
class SparseMatVec(ASTNode):
    matrix: Variable
    vector: ASTNode
    result: ASTNode
    row_idx: Optional[IndexExpr] = None
    transpose: bool = False
    
    def accept(self, visitor):
        return visitor.visit_sparse_matvec(self)


@dataclass
class VectorOp(ASTNode):
    op: str
    vectors: List[ASTNode]
    result: ASTNode
    size: Optional[int] = None
    
    def accept(self, visitor):
        return visitor.visit_vector_op(self)


@dataclass
class Literal(ASTNode):
    value: Union[int, float, str]
    
    def accept(self, visitor):
        return visitor.visit_literal(self)
    
    def __str__(self):
        if isinstance(self.value, float):
            return f"{self.value:.6f}f"
        else:
            return str(self.value)


@dataclass
class Comment(ASTNode):
    text: str
    
    def accept(self, visitor):
        return visitor.visit_comment(self)


@dataclass
class ElementWiseOp(ASTNode):
    op: str  # 'mul', 'add', 'sub'
    left: ASTNode
    right: ASTNode
    result: ASTNode
    size: int
    
    def accept(self, visitor):
        return visitor.visit_element_wise_op(self)


@dataclass
class MinMaxOp(ASTNode):
    operation: str  # 'min', 'max', 'clamp'
    operands: List[ASTNode]
    result: ASTNode
    size: Optional[int] = None
    
    def accept(self, visitor):
        return visitor.visit_min_max_op(self)


@dataclass
class VectorScalarOp(ASTNode):
    op: str  # 'mul', 'add', 'sub'
    vector: ASTNode
    scalar: Union[float, ASTNode]
    result: ASTNode
    size: int
    
    def accept(self, visitor):
        return visitor.visit_vector_scalar_op(self)


@dataclass
class AbsMaxOp(ASTNode):
    vector: ASTNode
    result: ASTNode
    size: int
    
    def accept(self, visitor):
        return visitor.visit_abs_max_op(self)


@dataclass
class ConditionalBlock(ASTNode):
    condition: ASTNode
    body: List[ASTNode]
    else_body: Optional[List[ASTNode]] = None
    
    def accept(self, visitor):
        return visitor.visit_conditional_block(self)


class ASTVisitor(ABC):
    @abstractmethod
    def visit_binary_op(self, node: BinaryOp): pass
    
    @abstractmethod
    def visit_unary_op(self, node: UnaryOp): pass
    
    @abstractmethod
    def visit_array_access(self, node: ArrayAccess): pass
    
    @abstractmethod
    def visit_assignment(self, node: Assignment): pass
    
    @abstractmethod
    def visit_loop(self, node: Loop): pass
    
    @abstractmethod
    def visit_sparse_matvec(self, node: SparseMatVec): pass
    
    @abstractmethod
    def visit_vector_op(self, node: VectorOp): pass
    
    @abstractmethod
    def visit_literal(self, node: Literal): pass
    
    @abstractmethod
    def visit_comment(self, node: Comment): pass
    
    @abstractmethod
    def visit_element_wise_op(self, node: ElementWiseOp): pass
    
    @abstractmethod
    def visit_min_max_op(self, node: MinMaxOp): pass
    
    @abstractmethod
    def visit_vector_scalar_op(self, node: VectorScalarOp): pass
    
    @abstractmethod
    def visit_abs_max_op(self, node: AbsMaxOp): pass
    
    @abstractmethod
    def visit_conditional_block(self, node: ConditionalBlock): pass