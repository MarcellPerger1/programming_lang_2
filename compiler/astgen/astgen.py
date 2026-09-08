from __future__ import annotations

import inspect
import sys
from typing import Callable, overload, TypeVar, TypeAlias

from util import flatten_force, is_strict_subclass, get_mro
from .ast_nodes import *
from .eval_literal import eval_number, eval_string
from .errors import AstParseError
from ..common import region_union, RegionUnionArgT, HasRegion, StrRegion
from ..cst.cst_node import AnyNode, Node, Leaf, SizedNode
from ..cst.cst_nodes import *
from ..cst.cstgen import CstGen


AutowalkerT: TypeAlias = Callable[['AstGen', AnyNode], AstNode]
CT = TypeVar('CT', bound=AutowalkerT)

_AUTOWALK_EXPR_DICT: dict[type[AnyNode], AutowalkerT] = {}


@overload
def _register_autowalk_expr(node_type: type[AnyNode], /) -> Callable[[CT], CT]: ...
# noinspection overloads
@overload  # (Pycharm's compatibility detection is not working properly)
def _register_autowalk_expr(cls: CT, /) -> CT: ...


def _register_autowalk_expr(node_type: type[AnyNode] | None = None, /):
    def decor(fn):
        _AUTOWALK_EXPR_DICT[
            node_type if node_type is not None
            else _detect_autowalk_type_from_annot(fn)
        ] = fn
        return fn
    if node_type is None or isinstance(node_type, type):
        return decor
    f, node_type = node_type, None
    return decor(f)


# A lot of coverage ignoring below in the error cases - we don't test this
# function specifically (if the autowalking works, this must have worked)
def _detect_autowalk_type_from_annot(fn):
    # Warning: Introspection black magic below! May contain demons and/or dragons.
    try:
        sig = inspect.signature(fn, eval_str=True, globals=globals())
    except Exception as e:  # pragma: no cover
        raise TypeError("Unable to detect node_type (cannot resolve annotations)") from e
    try:
        bound = sig.bind(0, 1)  # simulate call w/ 2 args
    except TypeError as e:  # pragma: no cover
        raise TypeError("Unable to detect node_type (signature may be incompatible)") from e
    arg2_name: str = tuple(bound.arguments)[1]  # get name it's bound to
    param = sig.parameters[arg2_name]  # lookup the param by name
    if param.kind not in (param.POSITIONAL_ONLY,
                          param.POSITIONAL_OR_KEYWORD):  # pragma: no cover
        raise TypeError("Unable to detect node_type (cannot find second positional arg)")
    if not is_strict_subclass(param.annotation, (Node, Leaf, SizedNode)):  # pragma: no cover
        raise TypeError("Unable to detect node_type (annotation is not a node type)")
    return param.annotation


# The job of this is to check that the syntax is valid and the semantics
# COULD be valid
# e.g. reject `2 + b = q;`, reject `1 + q;` (cannot represent expr in smt)
# e.g. accept `abc.d[2] = 3` even if there is no `abc`
class AstGen:
    def __init__(self, cst: CstGen):
        self.cst = cst
        self.src = self.cst.src
        self.result: AstProgramNode | None = None

    def parse(self) -> AstProgramNode:
        if self.result:
            return self.result
        result = self.result = self._walk_program(self.cst.parse())
        return result

    def _walk_program(self, root: ProgramNode):
        return AstProgramNode(root.region, self._walk_block(root.statements))

    def _walk_smt(self, smt: AnyNode) -> list[AstNode]:
        if isinstance(smt, NopNode):
            return []
        elif isinstance(smt, DeclNode):
            return self._walk_var_decl(smt)
        elif isinstance(smt, RepeatBlock):
            return [AstRepeat(smt.region, self._walk_expr(smt.count),
                              self._walk_block(smt.block))]
        elif isinstance(smt, WhileBlock):
            return [AstWhile(smt.region, self._walk_expr(smt.cond),
                             self._walk_block(smt.block))]
        elif isinstance(smt, ConditionalBlock):
            return self._walk_conditional(smt)
        elif isinstance(smt, AssignNode):  # Simple assignment
            # TODO: maybe separate SetAttr, SetItem, SetVar nodes?
            return [AstAssign(smt.region, self._walk_assign_left(smt.target),
                              self._walk_expr(smt.source))]
        elif isinstance(smt, AssignOpNode):  # Other (aug.) assignment
            return [AstAugAssign(
                smt.region, smt.name, self._walk_assign_left(smt.target),
                self._walk_expr(smt.source))]
        elif isinstance(smt, DefineNode):
            # (Check arg types later during name resolution)
            return [AstDefine(
                smt.region, self._walk_ident(smt.ident),
                [self._walk_arg_decl(a) for a in smt.args_decl.decls],
                self._walk_block(smt.block))]
        elif isinstance(smt, CallNode):
            # Check that it transforms into smt-intrinsic and
            # not expr-intrinsic in codegen/typecheck
            return [self._walk_call(smt)]
        else:
            raise self.err(
                f"Expected statement, not {smt.name!r} expression. Hint: "
                f"expressions have no side-effect so are not allowed at "
                f"the root level.", smt.region)

    def _walk_arg_decl(self, a: ArgDeclNode):
        return AstDefineParam(a.region, self._walk_ident(a.type), self._walk_ident(a.ident))

    def _walk_var_decl(self, smt: DeclNode):
        scope = (VarDeclScope.LET if isinstance(smt.decl_scope, DeclScope_Let)
                 else VarDeclScope.GLOBAL)
        tp = (VarDeclType.LIST if isinstance(smt.decl_type, DeclType_List)
              else VarDeclType.VARIABLE)
        # Add the region from the keywords to first decl (to make single-var
        # decls a more sensible .region that includes the `let` keyword as well)
        extra_region_first = region_union(smt.decl_scope, smt.decl_type)
        return [self._walk_single_decl(d, scope, tp,
                                       extra_region_first if i == 0 else None)
                for i, d in enumerate(smt.decl_list.decls)]

    def _walk_single_decl(self, d: DeclItemNode, scope: VarDeclScope,
                          tp: VarDeclType, extra_region: StrRegion | None):
        return AstDeclNode(
            region_union(d.ident, d.value, extra_region),
            scope, tp, self._walk_ident(d.ident),
            None if d.value is None else self._walk_expr(d.value))

    def _walk_assign_left(self, lhs: AnyNode) -> AstNode:
        if isinstance(lhs, IdentNode):
            return self._walk_ident(lhs)
        if isinstance(lhs, GetitemNode):
            # We're getting the item so there could be anything on the LHS
            # from the POV of the syntax analysis so use regular method
            return self._walk_getitem(lhs)
        if isinstance(lhs, GetattrNode):  # ditto
            return self._walk_getattr(lhs)
        raise self.err(f"Cannot assign to {lhs.name!r} expr", lhs)

    def _walk_conditional(self, smt: ConditionalBlock):
        # Build up else/elseif parts inner-first
        else_cst = smt.else_block  # New var to workaround Pycharm narrowing bug
        else_block = (None if isinstance(else_cst, NullElseBlock)
                      else self._walk_block(else_cst.block))
        for elseif in reversed(smt.elseif_blocks):  # Last = innermost
            # region is current elseif to end
            else_block = [AstIf(elseif.region | smt.else_block.region,
                                self._walk_expr(elseif.cond),
                                self._walk_block(elseif.block), else_block)]
        return [AstIf(smt.region, self._walk_expr(smt.if_block.cond),
                      self._walk_block(smt.if_block.block), else_block)]

    def _walk_block(self, nodes: list[AnyNode] | BlockNode) -> list[AstNode]:
        nodes = nodes.statements if isinstance(nodes, BlockNode) else nodes
        return flatten_force(map(self._walk_smt, nodes))

    def _walk_expr(self, expr: AnyNode) -> AstNode:
        return self.autowalk_expr(expr)

    @_register_autowalk_expr  # Don't need to pass the type due to black magic
    def _walk_number(self, node: NumberNode) -> AstNumber:
        return AstNumber(node.region, eval_number(self.node_str(node)))

    @_register_autowalk_expr
    def _walk_string(self, node: StringNode) -> AstString:
        return AstString(node.region, self._eval_string(node))

    @_register_autowalk_expr
    def _walk_autocat(self, node: AutocatNode) -> AstString:
        return AstString(node.region, ''.join(map(self._eval_string, node.parts)))

    @_register_autowalk_expr
    def _walk_ident(self, ident: IdentNode) -> AstIdent:
        return AstIdent(ident.region, self.node_str(ident, intern=True))

    @_register_autowalk_expr
    def _walk_attr_name(self, attr_name: AttrNameNode) -> AstAttrName:
        return AstAttrName(
            attr_name.region, self.node_str(attr_name, intern=True))

    @_register_autowalk_expr
    def _walk_list_literal(self, ls: ListNode):
        # For now it is UB to use this anywhere but in variable decls
        #  (until proper syntax desugar-ing for literal arrays is done).
        # Anyway, for now the codegen/typechecker/desugar step will check this
        return AstListLiteral(ls.region, [self._walk_expr(i) for i in ls.items])

    @_register_autowalk_expr
    def _walk_getattr(self, node: GetattrNode) -> AstAttribute:
        return AstAttribute(node.region, self._walk_expr(node.target),
                            self._walk_attr_name(node.attr))

    @_register_autowalk_expr
    def _walk_getitem(self, node: GetitemNode) -> AstItem:
        return AstItem(node.region, self._walk_expr(node.target),
                       self._walk_expr(node.item))

    @_register_autowalk_expr
    def _walk_paren(self, node: ParenNode) -> AstNode:
        return self._walk_expr(node.contents)

    @_register_autowalk_expr
    def _walk_call(self, node: CallNode) -> AstCall:
        return AstCall(node.region, self._walk_expr(node.target),
                       [self._walk_expr(a) for a in node.arglist.args])

    @_register_autowalk_expr
    def _walk_unary_op(self, node: UnaryOpNode) -> AstUnaryOp:
        op = node.name.removesuffix('(unary)')
        return AstUnaryOp(node.region, op, self._walk_expr(node.operand))

    @_register_autowalk_expr
    def _walk_binary_op(self, node: BinOpNode) -> AstBinOp:
        return AstBinOp(node.region, node.name, self._walk_expr(node.left),
                        self._walk_expr(node.right))

    def _eval_string(self, node: StringNode):
        return eval_string(self.node_str(node), node.region, self.src)

    def node_str(self, node: HasRegion, intern=False):
        s = node.region.resolve(self.src)
        if intern:
            return sys.intern(s)
        return s

    def err(self, msg: str, loc: RegionUnionArgT):
        return AstParseError(msg, region_union(loc), self.src)

    def autowalk_expr(self, node: AnyNode):
        return self._lookup_autowalk_fn(type(node))(self, node)

    @classmethod
    def _lookup_autowalk_fn(cls, t: type[AnyNode]):
        assert t is not Leaf and t is not Node and issubclass(t, AnyNode)
        if value := _AUTOWALK_EXPR_DICT.get(t):
            return value
        for supertype in get_mro(t):  # See if any supertypes have the autowalker
            if not issubclass(supertype, AnyNode):
                continue  # Skip - could be a mixin
            if value := _AUTOWALK_EXPR_DICT.get(supertype):
                _AUTOWALK_EXPR_DICT[t] = value  # Add it as cache
                return value
        raise LookupError(f"No such autowalk-er declared ({t})")  # pragma: no cover
