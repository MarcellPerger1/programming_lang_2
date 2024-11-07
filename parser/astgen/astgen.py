from __future__ import annotations

import inspect
import sys
from typing import Callable, overload, TypeVar, TypeAlias

from util import flatten_force, is_strict_subclass
from .ast_node import *  # TODO: add __all__
from .errors import LocatedAstError
from ..common import region_union, RegionUnionArgT
from ..cst.base_node import AnyNode, Node, Leaf
from ..cst.named_node import NamedLeafCls, NamedNodeCls, NamedSizedNodeCls
from ..cst.nodes import *
from ..cst.treegen import CstGen

# Final syntax lowering/codegen: (???)
# AST -> blocks, intrinsics, functions -> blocks & intrinsics -> intrinsics
#     |                                 |
#  ops to intrinsics             resolve idents

ALLOWED_IN_SMT = (  # Note: use with isinstance
    CallNode,
    NopNode,
    ConditionalBlock,
    WhileBlock,
    RepeatBlock,
    DefineNode,
    LetNode,
    GlobalNode,
    AssignOpNode,
)


AutowalkerT: TypeAlias = Callable[['AstGen', AnyNode], AstNode]
CT = TypeVar('CT', bound=AutowalkerT)

_AUTOWALK_EXPR_DICT: dict[type[AnyNode], AutowalkerT] = {}


@overload
def _register_autowalk_expr(node_type: type[AnyNode], /) -> Callable[[CT], CT]: ...
@overload
def _register_autowalk_expr(cls: CT, /) -> CT: ...


def _register_autowalk_expr(node_type: type[AnyNode] = None, /):
    def decor(fn):
        _AUTOWALK_EXPR_DICT[
            node_type if node_type is not None
            else _detect_autowalk_type_from_annot(fn)
        ] = fn
        return fn
    if node_type is None or isinstance(node_type, type):
        return decor
    return decor(node_type)


def _detect_autowalk_type_from_annot(fn):
    # Warning: Introspection black magic below! May contain demons and/or dragons.
    try:
        sig = inspect.signature(fn, eval_str=True, globals=globals())
    except Exception as e:
        raise TypeError("Cannot automatically determine node_type") from e
    try:
        bound = sig.bind(0, 1)  # simulate call w/ 2 args
    except TypeError as e:
        raise TypeError("Cannot automatically determine node_type") from e
    arg2_name: str = (*bound.arguments,)[1]  # get name it's bound to
    param = sig.parameters[arg2_name]  # lookup the param by name
    if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
        raise TypeError("Cannot automatically determine node_type")
    if not is_strict_subclass(param.annotation, (
            NamedLeafCls, NamedNodeCls, NamedSizedNodeCls)):
        raise TypeError("Cannot automatically determine node_type")
    return param.annotation


# The job of this is to check that the syntax is valid and the semantics
# COULD be valid
# e.g. reject `2 + b = q;`, reject `1 + q;` (cannot represent expr in smt)
# e.g. accept `abc.d[2] = 3` even if there is no `abc`
class AstGen:
    root: AnyNode

    def __init__(self, cst: CstGen):
        self.cst = cst
        self.src = self.cst.src
        self.result: AstNode | None = None

    def walk(self):
        if not self.result:
            self.root = self.cst.parse()
            self.result = self._walk_program(self.root)
        return self.result

    def _walk_program(self, root: AnyNode):
        assert isinstance(root, ProgramNode)
        return AstProgramNode(root.region, self._walk_block(root.statements))

    def _walk_smt(self, smt: AnyNode) -> list[AstNode]:
        if not isinstance(smt, ALLOWED_IN_SMT):
            raise self.err(
                f"Expected statement, not {smt.name!r} expression. Hint: "
                f"expressions have no side-effect so are not allowed at "
                f"the root level.", smt.region)
        if isinstance(smt, NopNode):
            return []
        elif isinstance(smt, (LetNode, GlobalNode)):
            decl_tp = VarDeclType.LET if isinstance(smt, LetNode) else VarDeclType.GLOBAL
            decls = [(self._walk_ident(d.ident),
                      None if d.value is None else self._walk_expr(d.value))
                     for d in smt.decls]
            return [AstDeclNode(smt.region, decl_tp, decls)]
        elif isinstance(smt, RepeatBlock):
            return [AstRepeat(smt.region, self._walk_expr(smt.count),
                              self._walk_block(smt.block.statements))]
        elif isinstance(smt, WhileBlock):
            return [AstWhile(smt.region, self._walk_expr(smt.cond),
                             self._walk_block(smt.block))]
        elif isinstance(smt, ConditionalBlock):
            # Build up else/elseif parts inner-first
            node = (None if isinstance(smt.else_block, NullElseBlock)
                    else self._walk_block(smt.else_block.block))
            for elseif in smt.elseif_blocks:
                # region is current elseif to end
                node = AstIf(elseif.region | smt.else_block.region,
                             self._walk_expr(elseif.cond),
                             self._walk_block(elseif.block), node)
            return [AstIf(smt.region, self._walk_expr(smt.if_block.cond),
                          self._walk_block(smt.if_block.block), node)]
        elif isinstance(smt, AssignNode):  # Simple assignment
            return [AstAssign(smt.region, self._walk_assign_left(smt.target),
                              self._walk_expr(smt.source))]
        elif isinstance(smt, AssignOpNode):  # Other (aug.) assignment
            return [AstAugAssign(
                smt.region, smt.name, self._walk_assign_left(smt.target),
                self._walk_expr(smt.target))]
        elif isinstance(smt, DefineNode):
            # Check arg types during name resolution
            decls = [(self._walk_ident(d.type), self._walk_ident(d.ident))
                     for d in smt.args_decl.decls]
            return [AstDefine(smt.region, self._walk_ident(smt.ident),
                              decls, self._walk_block(smt.block))]
        elif isinstance(smt, CallNode):
            # Check that it transforms into smt-intrinsic and
            # not expr-intrinsic in codegen/typecheck
            return [self._walk_call(smt)]
        assert 0, f"Unhandled node class {type(smt).__name__} in _walk_smt"

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

    def _walk_block(self, nodes: list[AnyNode] | BlockNode) -> list[AstNode]:
        if isinstance(nodes, BlockNode):
            nodes = nodes.statements
        return flatten_force(map(self._walk_smt, nodes))

    def _walk_expr(self, expr: AnyNode) -> AstNode:
        return self.autowalk_expr(expr)

    @_register_autowalk_expr  # Don't need to pass the type due to black magic
    def _walk_number(self, node: NumberNode) -> AstNumber:
        return AstNumber(node.region)

    @_register_autowalk_expr
    def _walk_string(self, node: StringNode) -> AstString:
        return AstString(node.region)

    @_register_autowalk_expr
    def _walk_autocat(self, node: AutocatNode) -> AstString:
        # TODO: value=... attr!
        return AstString(node.region)

    @_register_autowalk_expr
    def _walk_ident(self, ident: IdentNode) -> AstIdent:
        return AstIdent(ident.region, self.node_str(ident, intern=True))

    @_register_autowalk_expr
    def _walk_attr_name(self, attr_name: AttrNameNode) -> AstAttrName:
        return AstAttrName(
            attr_name.region, self.node_str(attr_name, intern=True))

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

    def node_str(self, node: HasRegion, intern=False):
        s = node.region.resolve(self.src)
        if intern:
            return sys.intern(s)
        return s

    def err(self, msg: str, loc: RegionUnionArgT):
        return LocatedAstError(msg, region_union(loc), self.src)

    def autowalk_expr(self, node: AnyNode):
        return self._lookup_autowalk_fn(type(node))(self, node)

    @classmethod
    def _lookup_autowalk_fn(cls, t: type[AnyNode]):
        assert t != Leaf and t != Node and issubclass(t, AnyNode)
        if value := _AUTOWALK_EXPR_DICT.get(t):
            return value
        for supertype in t.mro():  # See if any supertypes have the autowalker
            if not issubclass(supertype, AnyNode):
                continue  # Skip - could be a mixin
            if value := _AUTOWALK_EXPR_DICT.get(supertype):
                return value
        raise LookupError(f"No such autowalk-er declared ({t})")
