from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from .name_resolver import FuncInfo, Scope, NameResolver
from .types import TypeInfo, ValType, BoolType, ListType, VoidType, FunctionType
from ..astgen.ast_nodes import *
from ..common import BaseLocatedError, region_union, RegionUnionArgT


class TypecheckError(BaseLocatedError):
    """Errors raised by the typechecker"""


NodeTypecheckFn: TypeAlias = 'Callable[[Typechecker, AstNode], TypeInfo | None]'
NodeTypecheckFnStrict: TypeAlias = 'Callable[[Typechecker, AstNode], TypeInfo]'

_typecheck_dispatch: dict[type[AstNode], NodeTypecheckFnStrict] = {}


@dataclass
class TypeMetadata:
    type: TypeInfo


class Typechecker:
    _curr_scope: Scope

    def __init__(self, name_resolver: NameResolver):
        self.resolver = name_resolver
        self.src = self.resolver.src
        self.is_ok: bool | None = None
        self.typed_ast: AstProgramNode[TypeMetadata] | None = None

    def _init(self):
        self.resolver.run()
        self.orig_ast = self.resolver.ast
        self.top_scope = self.resolver.top_scope
        self._curr_scope = self.top_scope

    def run(self) -> AstProgramNode[TypeMetadata]:
        if self.typed_ast is not None:
            return self.typed_ast
        self._init()
        self._typecheck(self.orig_ast)
        # TODO: tests for the output types
        self.typed_ast = self.orig_ast  # should now have the types
        return self.typed_ast

    def _node_typechecker(self, tp=None):
        if tp is None:
            assert callable(self)
            tp = self  # Called as decor in this class

        def decor(fn: NodeTypecheckFn):
            @functools.wraps(fn)
            def new_fn(self_inner: Typechecker, n: AstNode) -> TypeInfo:
                n_type = fn(self_inner, n) or VoidType()  # return None = void
                n.meta = TypeMetadata(n_type)
                return n_type
            _typecheck_dispatch[tp] = new_fn
            return new_fn
        return decor

    def _typecheck(self, n: AstNode):
        try:
            fn = _typecheck_dispatch[type(n)]
        except KeyError:
            fn = type(self)._typecheck_node_fallback
        return fn(self, n)

    def _typecheck_node_fallback(self, n: AstNode):
        raise NotImplementedError(f"No typechecker function for node "
                                  f"type {type(n).__name__}")

    @_node_typechecker(AstProgramNode)
    def _typecheck_program(self, n: AstProgramNode):
        self._typecheck_block(n.statements)

    def _typecheck_block(self, block: list[AstNode]):
        for smt in block:
            self.expect_type(self._typecheck(smt), VoidType(), smt)

    @_node_typechecker(AstDeclNode)
    def _typecheck_decl(self, n: AstDeclNode):
        if not n.value:  # Nothing to check
            return
        expect = self._resolve_scope(n.scope).declared[n.ident.id].tp_info
        self.expect_type(self._typecheck(n.value), expect, n)

    @_node_typechecker(AstRepeat)
    def _typecheck_repeat(self, n: AstRepeat):
        # For now, we don't differentiate between number/string (as sc doesn't)
        self.expect_type(self._typecheck(n.count), ValType(), n.count)
        self._typecheck_block(n.body)

    @_node_typechecker(AstIf)
    def _typecheck_if(self, n: AstIf):
        self.expect_type(self._typecheck(n.cond), BoolType(), n.cond)
        self._typecheck_block(n.if_body)
        if n.else_body is not None:
            self._typecheck_block(n.else_body)

    @_node_typechecker(AstWhile)
    def _typecheck_while(self, n: AstWhile):
        self.expect_type(self._typecheck(n.cond), BoolType(), n.cond)
        self._typecheck_block(n.body)

    @_node_typechecker(AstAssign)
    def _typecheck_assign(self, n: AstAssign):  # super tempted to call this _typecheck_ass
        if isinstance(n.target, AstIdent):
            target_tp = self._curr_scope.used[n.target.id].tp_info
        elif isinstance(n.target, AstItem):  # ls[i] = v
            target_tp = self._typecheck(n.target)  # Also checks that `ls` is a list
        elif isinstance(n.target, AstAttribute):
            raise self.err("Setting attributes is currently unsupported", n.target)
        else:
            assert 0, "Unknown simple-assignment type"
        if target_tp == ListType():
            raise self.err("Cannot assign directly to list", n)
        self.expect_type(self._typecheck(n.source), target_tp, n)

    @_node_typechecker(AstAugAssign)
    def _typecheck_aug_assign(self, n: AstAugAssign):
        # TODO: change this when desugaring is implemented
        #  (for now only +=, only on variables)
        if n.op != '+=':
            raise self.err(f"The '{n.op}' operator is not implemented", n)
        if not isinstance(n.target, AstIdent):
            raise self.err(f"The '+=' operator is only implemented for variables", n)
        target_tp = self._curr_scope.used[n.target.id].tp_info
        if target_tp != ValType():
            raise self.err(f"Cannot apply += to {target_tp}", n)
        self.expect_type(self._typecheck(n.source), ValType(), n.source)

    @_node_typechecker(AstDefine)
    def _typecheck_define(self, n: AstDefine):
        # Don't really need to check much here - type is generated from the
        # syntax so must be correct. Set _curr_scope and check body
        func_info = self._curr_scope.declared[n.ident.id]
        assert isinstance(func_info, FuncInfo)
        old_scope = self._curr_scope
        self._curr_scope = func_info.subscope
        try:
            self._typecheck_block(n.body)
        finally:
            self._curr_scope = old_scope

    @_node_typechecker(AstNumber)
    def _typecheck_number(self, _n: AstNumber):
        return ValType()

    @_node_typechecker(AstString)
    def _typecheck_string(self, _n: AstString):
        return ValType()

    @_node_typechecker(AstListLiteral)
    def _typecheck_list(self, n: AstListLiteral):
        for item in n.items:
            if self._typecheck(item) != ValType():
                raise self.err("Can only have ValType()s in list", item)
        return ListType()

    @_node_typechecker(AstIdent)
    def _typecheck_ident(self, n: AstIdent):
        return self._curr_scope.used[n.id].tp_info

    @_node_typechecker(AstAttrName)
    def _typecheck_attr_name(self, _n: AstAttrName):
        assert 0, "AstAttrName has no type, cannot be checked on its own"

    @_node_typechecker(AstAttribute)
    def _typecheck_attribute(self, n: AstAttribute):
        # TODO: implement this properly, with better types and stuff
        raise self.err("Attributes are not implemented yet", n)

    @_node_typechecker(AstItem)
    def _typecheck_item(self, n: AstItem):
        # TODO: this will require different intrinsics for string vs list getitem
        container_tp = self._typecheck(n.obj)
        if container_tp not in (ListType(), ValType()):
            raise self.err(f"Cannot get item of {container_tp}", n)
        self.expect_type(self._typecheck(n.index), ValType(), n.index)

    @_node_typechecker(AstCall)
    def _typecheck_call(self, n: AstCall):
        called_tp = self._typecheck(n.obj)
        if not isinstance(called_tp, FunctionType):
            raise self.err(f"Cannot call {called_tp}", n.obj)
        if len(called_tp.arg_types) != len(n.args):
            if n.args and len(n.args) > len(called_tp.arg_types):
                region = n.args[-1].region  # Highlight extraneous arg
            else:
                region = n.region
            raise self.err(f"Incorrect number of arguments, expected "
                           f"{len(called_tp.arg_types)}, got {len(n.args)}",
                           region)
        for decl_t, arg_node in zip(called_tp.arg_types, n.args):
            self.expect_type(self._typecheck(arg_node), decl_t, arg_node)
        return called_tp.ret_type

    _BINARY_OP_TYPES = dict.fromkeys([
        *'+-*/%', '**', '..', '==', '!=', '<', '>', '<=', '>='
    ], ValType()) | dict.fromkeys([
        '&&', '||'
    ], BoolType())

    _UNARY_OP_TYPES = dict.fromkeys([
        *'+-'
    ], ValType()) | dict.fromkeys([
        '!'
    ], BoolType())

    # TODO: allow casting bool to val? - auto-cast or explicit?
    @_node_typechecker(AstBinOp)
    def _typecheck_bin_op(self, n: AstBinOp):
        expect_tp = self._BINARY_OP_TYPES[n.op]
        self.expect_type(self._typecheck(n.left), expect_tp, n.left)
        self.expect_type(self._typecheck(n.right), expect_tp, n.right)

    @_node_typechecker(AstUnaryOp)
    def _typecheck_unary_op(self, n: AstUnaryOp):
        expect_tp = self._UNARY_OP_TYPES[n.op]
        self.expect_type(self._typecheck(n.operand), expect_tp, n.operand)

    def _resolve_scope(self, scope_tp: VarDeclScope):
        return self.top_scope if scope_tp == VarDeclScope.GLOBAL else self._curr_scope

    def err(self, msg: str, loc: RegionUnionArgT):
        return TypecheckError(msg, region_union(loc), self.src)

    def expect_type(self, actual: TypeInfo, exp: TypeInfo, loc: RegionUnionArgT):
        if exp != actual:
            raise self.err(f"Expected type {exp}, got type {actual}", loc)
