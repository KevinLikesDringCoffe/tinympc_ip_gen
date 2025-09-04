from typing import Dict, List, Optional, Union, Any
from .core import Variable, IndexExpr, DataType
from .ast_nodes import *
import copy


class CodeGenerator:
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.loop_depth = 0
        self.unroll_mode = False
        self.current_indices = {}
        self.temp_var_counter = 0
        
    def register_variable(self, var: Variable):
        self.variables[var.name] = var
        
    def get_temp_var_name(self) -> str:
        name = f"temp_{self.temp_var_counter}"
        self.temp_var_counter += 1
        return name
        
    def build_ast(self, operations: List[Dict]) -> List[ASTNode]:
        ast_nodes = []
        for op in operations:
            node = self._build_operation(op)
            if node:
                ast_nodes.append(node)
        return ast_nodes
    
    def _build_operation(self, op: Dict) -> Optional[ASTNode]:
        op_type = op.get('type')
        
        if op_type == 'matvec':
            return self._build_matvec(op)
        elif op_type == 'vecadd':
            return self._build_vecadd(op)
        elif op_type == 'vecsub':
            return self._build_vecsub(op)
        elif op_type == 'loop':
            return self._build_loop(op)
        elif op_type == 'assignment':
            return self._build_assignment(op)
        elif op_type == 'comment':
            return Comment(op['text'])
        elif op_type == 'unary_op':
            return UnaryOp(op['op'], self._build_value_node(op['operand']))
        elif op_type == 'element_wise':
            return self._build_element_wise(op)
        elif op_type == 'min_max':
            return self._build_min_max(op)
        elif op_type == 'vector_scalar':
            return self._build_vector_scalar(op)
        elif op_type == 'abs_max':
            return self._build_abs_max(op)
        elif op_type == 'conditional':
            return self._build_conditional(op)
        elif op_type == 'sparse_matvec':
            return self._build_sparse_matvec(op)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    def _build_matvec(self, op: Dict) -> ASTNode:
        matrix_name = op['matrix']
        matrix = self.variables.get(matrix_name)
        
        vector_node = self._build_access_node(op['vector'], op.get('vec_idx'))
        result_node = self._build_access_node(op['result'], op.get('res_idx'))
        
        return SparseMatVec(
            matrix=matrix,
            vector=vector_node,
            result=result_node,
            row_idx=op.get('row_idx'),
            transpose=op.get('transpose', False)
        )
    
    def _build_vecadd(self, op: Dict) -> ASTNode:
        left = self._build_access_node(op['left'], op.get('left_idx'))
        right = self._build_access_node(op['right'], op.get('right_idx'))
        result = self._build_access_node(op['result'], op.get('res_idx'))
        
        return VectorOp(
            op='add',
            vectors=[left, right],
            result=result,
            size=op.get('size')
        )
    
    def _build_vecsub(self, op: Dict) -> ASTNode:
        left = self._build_access_node(op['left'], op.get('left_idx'))
        right = self._build_access_node(op['right'], op.get('right_idx'))
        result = self._build_access_node(op['result'], op.get('res_idx'))
        
        return VectorOp(
            op='sub',
            vectors=[left, right],
            result=result,
            size=op.get('size')
        )
    
    def _build_loop(self, op: Dict) -> ASTNode:
        body_ast = []
        for body_op in op['body']:
            node = self._build_operation(body_op)
            if node:
                body_ast.append(node)
        
        return Loop(
            var=op['var'],
            start=op['start'],
            end=op['end'],
            body=body_ast,
            step=op.get('step', 1)
        )
    
    def _build_assignment(self, op: Dict) -> ASTNode:
        target = self._build_access_node(op['target'], op.get('target_idx'))
        value = self._build_value_node(op['value'])
        
        return Assignment(
            target=target,
            value=value,
            accumulate=op.get('accumulate', False)
        )
    
    def _build_access_node(self, name: str, idx: Optional[Union[int, IndexExpr, List]] = None) -> ArrayAccess:
        if idx is None:
            return ArrayAccess(name, [])
        elif isinstance(idx, list):
            return ArrayAccess(name, idx)
        else:
            return ArrayAccess(name, [idx])
    
    def _build_value_node(self, value: Any) -> ASTNode:
        if isinstance(value, (int, float)):
            return Literal(value)
        elif isinstance(value, str):
            return ArrayAccess(value, [])
        elif isinstance(value, dict):
            return self._build_operation(value)
        elif isinstance(value, ASTNode):
            return value
        else:
            return Literal(value)
    
    def optimize_ast(self, ast: List[ASTNode]) -> List[ASTNode]:
        optimized = []
        for node in ast:
            optimized_node = self._optimize_node(node)
            if isinstance(optimized_node, list):
                optimized.extend(optimized_node)
            else:
                optimized.append(optimized_node)
        return optimized
    
    def _optimize_node(self, node: ASTNode) -> Union[ASTNode, List[ASTNode]]:
        if isinstance(node, Loop) and self.unroll_mode:
            return self._unroll_loop(node)
        elif isinstance(node, SparseMatVec):
            return self._optimize_sparse(node)
        else:
            return node
    
    def _unroll_loop(self, loop: Loop) -> List[ASTNode]:
        unrolled = []
        
        if isinstance(loop.start, int) and isinstance(loop.end, int):
            for i in range(loop.start, loop.end, loop.step):
                unrolled.append(Comment(f"Iteration {i}"))
                for body_node in loop.body:
                    unrolled_node = self._substitute_index(body_node, loop.var, i)
                    unrolled.append(unrolled_node)
        else:
            return [loop]
        
        return unrolled
    
    def _substitute_index(self, node: ASTNode, var: str, value: int) -> ASTNode:
        node_copy = copy.deepcopy(node)
        
        if isinstance(node_copy, ArrayAccess):
            new_indices = []
            for idx in node_copy.indices:
                if isinstance(idx, IndexExpr) and idx.base == var:
                    new_indices.append(idx.substitute(value))
                elif isinstance(idx, str) and idx == var:
                    new_indices.append(value)
                else:
                    new_indices.append(idx)
            node_copy.indices = new_indices
        elif isinstance(node_copy, SparseMatVec):
            if node_copy.row_idx and node_copy.row_idx.base == var:
                node_copy.row_idx = IndexExpr.constant(node_copy.row_idx.substitute(value))
            node_copy.vector = self._substitute_index(node_copy.vector, var, value)
            node_copy.result = self._substitute_index(node_copy.result, var, value)
        elif isinstance(node_copy, VectorOp):
            node_copy.vectors = [self._substitute_index(v, var, value) for v in node_copy.vectors]
            node_copy.result = self._substitute_index(node_copy.result, var, value)
        elif isinstance(node_copy, Assignment):
            node_copy.target = self._substitute_index(node_copy.target, var, value)
            node_copy.value = self._substitute_index(node_copy.value, var, value)
        
        return node_copy
    
    def _optimize_sparse(self, node: SparseMatVec) -> SparseMatVec:
        return node
    
    def _build_element_wise(self, op: Dict) -> ASTNode:
        """Build element-wise operation node"""
        left = self._build_access_node(op['left'], op.get('left_idx'))
        right = self._build_access_node(op['right'], op.get('right_idx'))
        result = self._build_access_node(op['result'], op.get('res_idx'))
        
        return ElementWiseOp(
            op=op['op'],
            left=left,
            right=right,
            result=result,
            size=op['size']
        )
    
    def _build_min_max(self, op: Dict) -> ASTNode:
        """Build min/max operation node"""
        operands = [self._build_access_node(operand) if isinstance(operand, str) 
                   else self._build_value_node(operand) for operand in op['operands']]
        result = self._build_access_node(op['result'], op.get('res_idx'))
        
        return MinMaxOp(
            operation=op['operation'],
            operands=operands,
            result=result,
            size=op.get('size')
        )
    
    def _build_vector_scalar(self, op: Dict) -> ASTNode:
        """Build vector-scalar operation node"""
        vector = self._build_access_node(op['vector'], op.get('vec_idx'))
        result = self._build_access_node(op['result'], op.get('res_idx'))
        
        return VectorScalarOp(
            op=op['op'],
            vector=vector,
            scalar=op['scalar'],
            result=result,
            size=op['size']
        )
    
    def _build_abs_max(self, op: Dict) -> ASTNode:
        """Build absolute maximum operation node"""
        vector = self._build_access_node(op['vector'], op.get('vec_idx'))
        result = self._build_access_node(op['result'], op.get('res_idx'))
        
        return AbsMaxOp(
            vector=vector,
            result=result,
            size=op['size']
        )
    
    def _build_conditional(self, op: Dict) -> ASTNode:
        """Build conditional block node"""
        condition = self._build_value_node(op['condition'])
        body = [self._build_operation(body_op) for body_op in op['body']]
        body = [node for node in body if node is not None]
        
        else_body = None
        if 'else_body' in op:
            else_body = [self._build_operation(else_op) for else_op in op['else_body']]
            else_body = [node for node in else_body if node is not None]
        
        return ConditionalBlock(
            condition=condition,
            body=body,
            else_body=else_body
        )
    
    def _build_sparse_matvec(self, op: Dict) -> ASTNode:
        """Build sparse matrix-vector operation (alias for matvec)"""
        return self._build_matvec(op)
    
    def generate_code(self, ast: List[ASTNode], target: str = 'c') -> str:
        if target == 'c':
            from .emitters import CCodeEmitter
            return CCodeEmitter().emit(ast)
        elif target == 'hls':
            from .emitters import HLSCodeEmitter
            return HLSCodeEmitter().emit(ast)
        else:
            raise ValueError(f"Unsupported target: {target}")