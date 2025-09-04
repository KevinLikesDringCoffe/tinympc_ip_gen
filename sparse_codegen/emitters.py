"""Essential code emitters"""

from typing import List, Union
from .ast_nodes import *
from .core import IndexExpr


class CCodeEmitter(ASTVisitor):
    def __init__(self):
        self.indent_level = 0
        self.code_lines = []
        
    def emit(self, ast: List[ASTNode]) -> str:
        self.code_lines = []
        self.indent_level = 0
        for node in ast:
            node.accept(self)
        return '\n'.join(self.code_lines)
    
    def _indent(self) -> str:
        return '    ' * self.indent_level
    
    def _add_line(self, line: str):
        if line:
            self.code_lines.append(self._indent() + line)
        else:
            self.code_lines.append('')
    
    def visit_binary_op(self, node: BinaryOp) -> str:
        left = self._evaluate_node(node.left)
        right = self._evaluate_node(node.right)
        return f"({left} {node.op} {right})"
    
    def visit_unary_op(self, node: UnaryOp) -> str:
        operand = self._evaluate_node(node.operand)
        return f"{node.op}({operand})"
    
    def visit_array_access(self, node: ArrayAccess) -> str:
        if not node.indices:
            return node.array
        indices = ''.join(f"[{str(idx)}]" for idx in node.indices)
        return f"{node.array}{indices}"
    
    def visit_assignment(self, node: Assignment):
        target = self._evaluate_node(node.target)
        value = self._evaluate_node(node.value)
        self._add_line(f"{target} = {value};")
    
    def visit_loop(self, node: Loop):
        self._add_line(f"for (int {node.var} = {node.start}; {node.var} < {node.end}; {node.var} += {node.step}) {{")
        self.indent_level += 1
        for body_node in node.body:
            body_node.accept(self)
        self.indent_level -= 1
        self._add_line("}")
    
    def visit_sparse_matvec(self, node: SparseMatVec):
        matrix = node.matrix
        if not matrix.sparsity_pattern:
            return
        
        rows_dict = {}
        for (i, j), value in matrix.sparsity_pattern.items():
            if i not in rows_dict:
                rows_dict[i] = []
            rows_dict[i].append((j, value))
        
        for row_idx in sorted(rows_dict.keys()):
            elements = rows_dict[row_idx]
            
            if isinstance(node.result, ArrayAccess) and node.result.indices:
                # If result has indices, append row_idx after them
                indices_str = ''.join(f"[{str(idx)}]" for idx in node.result.indices)
                result_access = f"{node.result.array}{indices_str}[{row_idx}]"
            else:
                result_access = f"{self._evaluate_node(node.result)}[{row_idx}]"
            
            terms = []
            for col_idx, value in elements:
                if isinstance(node.vector, ArrayAccess) and node.vector.indices:
                    # Properly format with all indices
                    indices_str = ''.join(f"[{str(idx)}]" for idx in node.vector.indices)
                    vector_access = f"{node.vector.array}{indices_str}[{col_idx}]"
                else:
                    vector_access = f"{self._evaluate_node(node.vector)}[{col_idx}]"
                
                if abs(value - 1.0) < 1e-10:
                    terms.append(vector_access)
                elif abs(value + 1.0) < 1e-10:
                    terms.append(f"-{vector_access}")
                else:
                    terms.append(f"{value:.6f}f * {vector_access}")
            
            if terms:
                self._add_line(f"{result_access} = {' + '.join(terms)};")
    
    def visit_vector_op(self, node: VectorOp):
        if node.size:
            for i in range(node.size):
                result_access = f"{self._evaluate_node(node.result)}[{i}]"
                operands = [f"{self._evaluate_node(vec)}[{i}]" for vec in node.vectors]
                
                if node.op == 'add':
                    self._add_line(f"{result_access} = {' + '.join(operands)};")
                elif node.op == 'sub':
                    self._add_line(f"{result_access} = {operands[0]} - {operands[1]};")
    
    def visit_literal(self, node: Literal) -> str:
        return str(node)
    
    def visit_comment(self, node: Comment):
        self._add_line(f"// {node.text}")
    
    def visit_element_wise_op(self, node: ElementWiseOp):
        """Element-wise operations between vectors/arrays"""
        for i in range(node.size):
            left_access = f"{self._evaluate_node(node.left)}[{i}]"
            right_access = f"{self._evaluate_node(node.right)}[{i}]"
            result_access = f"{self._evaluate_node(node.result)}[{i}]"
            
            if node.op == 'mul':
                self._add_line(f"{result_access} = {left_access} * {right_access};")
            elif node.op == 'add':
                self._add_line(f"{result_access} = {left_access} + {right_access};")
            elif node.op == 'sub':
                self._add_line(f"{result_access} = {left_access} - {right_access};")
    
    def visit_min_max_op(self, node: MinMaxOp):
        """Min/max operations for box constraints"""
        if node.size:
            for i in range(node.size):
                result_access = f"{self._evaluate_node(node.result)}[{i}]"
                operands = [f"{self._evaluate_node(op)}[{i}]" for op in node.operands]
                
                if node.operation == 'clamp':
                    # clamp(value, min, max) = min(max, max(min, value))
                    self._add_line(f"{result_access} = fminf({operands[2]}, fmaxf({operands[1]}, {operands[0]}));")
                elif node.operation == 'min':
                    expr = operands[0]
                    for op in operands[1:]:
                        expr = f"fminf({expr}, {op})"
                    self._add_line(f"{result_access} = {expr};")
                elif node.operation == 'max':
                    expr = operands[0]
                    for op in operands[1:]:
                        expr = f"fmaxf({expr}, {op})"
                    self._add_line(f"{result_access} = {expr};")
        else:
            # Scalar operations
            result_access = self._evaluate_node(node.result)
            operands = [self._evaluate_node(op) for op in node.operands]
            
            if node.operation == 'clamp':
                self._add_line(f"{result_access} = fminf({operands[2]}, fmaxf({operands[1]}, {operands[0]}));")
    
    def visit_vector_scalar_op(self, node: VectorScalarOp):
        """Vector-scalar operations"""
        scalar_val = node.scalar if isinstance(node.scalar, (int, float)) else self._evaluate_node(node.scalar)
        
        for i in range(node.size):
            vector_access = f"{self._evaluate_node(node.vector)}[{i}]"
            result_access = f"{self._evaluate_node(node.result)}[{i}]"
            
            if node.op == 'mul':
                self._add_line(f"{result_access} = {scalar_val} * {vector_access};")
            elif node.op == 'add':
                self._add_line(f"{result_access} = {vector_access} + {scalar_val};")
            elif node.op == 'sub':
                self._add_line(f"{result_access} = {vector_access} - {scalar_val};")
    
    def visit_abs_max_op(self, node: AbsMaxOp):
        """Compute max absolute value of vector"""
        result_access = self._evaluate_node(node.result)
        self._add_line(f"{result_access} = 0.0f;")
        
        for i in range(node.size):
            vector_access = f"{self._evaluate_node(node.vector)}[{i}]"
            self._add_line(f"{result_access} = fmaxf({result_access}, fabsf({vector_access}));")
    
    def visit_conditional_block(self, node: ConditionalBlock):
        """Conditional blocks (if/else)"""
        condition = self._evaluate_node(node.condition)
        self._add_line(f"if ({condition}) {{")
        self.indent_level += 1
        
        for body_node in node.body:
            body_node.accept(self)
        
        if node.else_body:
            self.indent_level -= 1
            self._add_line("} else {")
            self.indent_level += 1
            for else_node in node.else_body:
                else_node.accept(self)
        
        self.indent_level -= 1
        self._add_line("}")
    
    def _evaluate_node(self, node: ASTNode) -> str:
        if isinstance(node, Literal):
            return str(node)
        elif isinstance(node, ArrayAccess):
            return self.visit_array_access(node)
        elif isinstance(node, BinaryOp):
            return self.visit_binary_op(node)
        elif isinstance(node, UnaryOp):
            return self.visit_unary_op(node)
        else:
            return ""


class HLSCodeEmitter(CCodeEmitter):
    def visit_loop(self, node: Loop):
        self._add_line("#pragma HLS PIPELINE II=1")
        super().visit_loop(node)
    
    def visit_sparse_matvec(self, node: SparseMatVec):
        self._add_line("#pragma HLS INLINE")
        super().visit_sparse_matvec(node)