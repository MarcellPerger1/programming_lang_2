from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

from util.recursive_eq import recursive_eq
from ..astgen.ast_nodes import *
from ..astgen.astgen import AstGen
from ..astgen.filtered_walker import FilteredWalker
from ..common import BaseLocatedError, StrRegion, region_union, RegionUnionArgT


@dataclass
class TypeInfo:
    def __post_init__(self):
        assert type(self) != TypeInfo, "Cannot instantiate TypeInfo directly,use a subclass"


@dataclass
class ValType(TypeInfo):
    pass


@dataclass
class BoolType(TypeInfo):
    pass


@dataclass
class ListType(TypeInfo):
    pass


@dataclass
class VoidType(TypeInfo):
    """The ``void`` type - represents 'there must not be a value here'.

    For example, this is the return type of function that don't return anything
    (e.g. all regular user-defined scratch functions).
    """


@dataclass
class FunctionType(TypeInfo):
    arg_types: list[TypeInfo]
    ret_type: TypeInfo


@dataclass
class NameInfo:
    decl_scope: Scope
    ident: str
    tp_info: TypeInfo
    # node: AstNode  # <-- Why do we need this?
    is_param: bool = field(default=False, kw_only=True)


@dataclass
class FuncInfo(NameInfo):
    tp_info: FunctionType  # Overrides types (doesn't change order)
    params_info: list[ParamInfo]
    # Can't just pass default_factory=Scope as it is only defined below
    subscope: Scope = field(default_factory=lambda: Scope())

    @classmethod
    def from_param_info(
            cls, decl_scope: Scope, ident: str, params_info: list[ParamInfo],
            ret_type: TypeInfo, subscope: Scope = None):
        subscope = subscope or Scope()
        tp_info = FunctionType([p.tp for p in params_info], ret_type)
        return cls(decl_scope, ident, tp_info, params_info, subscope)


@dataclass
class ParamInfo:
    name: str
    tp: TypeInfo


@dataclass
class Scope:
    declared: dict[str, NameInfo] = field(default_factory=dict)
    used: dict[str, NameInfo] = field(default_factory=dict)
    """Add references to outer scopes' variables that we use.
    (so type codegen/type-checker knows what each AstIdent refers to)"""


Scope.__eq__ = recursive_eq(Scope.__eq__)


class NameResolutionError(BaseLocatedError):
    pass


# The reason `let` isn't used is because we don't want to imply similarity
#   between parameters as local variables (where none exists in Scratch).
#   Also, we might want to use `let` later as a modifier to bind it to
#     an actual local var.
# Don't need to `sys.intern` these manually as Python automatically does
#   this for literals.
PARAM_TYPES = {'number', 'string', 'val', 'bool'}


# Variables:
#  - We can prevent usages before the variable is declared in 2 ways:
#    - Based on time: very sensible, like JS, but requires too many runtime features
#    - Based on location: somewhat makes sense except for inner functions -
#       they may be called later so should be able to access any variables.
#  - Or we can just ignore it (e.g. `var` in JS) and pretend everything was
#     declared at the top (but not assigned to - i.e. hoist `var foo;` to top).
# To minimise accidental errors, option 1.2 is best
#  (errors shouldn't pass silently, and that method requires no special runtime)
class NameResolver:
    def __init__(self, astgen: AstGen):
        self.astgen = astgen
        self.src = self.astgen.src
        self.top_scope: Scope | None = None

    def _init(self):
        self.ast = self.astgen.parse()
        self.top_scope = Scope()

    def run(self):
        if self.top_scope:
            return self.top_scope
        self._init()
        self.run_on_new_scope(self.ast.statements, curr_scope=self.top_scope)
        return self.top_scope

    def run_on_new_scope(self, block: list[AstNode], parent_scopes: list[Scope] = None,
                         curr_scope: Scope = None):
        def enter_ident(n: AstIdent):
            for s in scope_stack[::-1]:  # Inefficient, creates a copy!
                if info := s.declared.get(n.id):
                    curr_scope.used[n.id] = info
                    return
            raise self.err(f"Name '{n.id}' is not defined", n.region)

        def enter_decl(n: AstDeclNode):
            # Need semi-special logic here to prevent walking it walking
            # the AstIdent that is currently being declared.
            AstNode.walk_obj(n.value, walker)  # Don't walk `n.ident`
            # Do this after walking (that is when the name is bound)
            ident = n.ident.id
            target_scope = curr_scope if n.scope == VarDeclScope.LET else self.top_scope
            if ident in target_scope.declared:
                raise self.err("Variable already declared", n.region)
            target_scope.declared[ident] = NameInfo(target_scope, ident, (
                ValType() if n.type == VarDeclType.VARIABLE else ListType()))
            return True

        def enter_fn_decl(fn: AstDefine):
            ident = fn.ident.id
            if ident in curr_scope.declared:
                raise self.err("Function already declared", fn.ident.region)
            subscope = Scope()
            params: list[ParamInfo] = []
            for tp_node, name_node in fn.params:
                if tp_node.id not in PARAM_TYPES:
                    raise self.err("Unknown parameter type", tp_node.region)
                if (name := name_node.id) in subscope.declared:
                    raise self.err("There is already a parameter of this name",
                                   name_node.region)
                tp = BoolType() if tp_node.id == 'bool' else ValType()
                subscope.declared[name] = NameInfo(subscope, name, tp, is_param=True)
                params.append(ParamInfo(name, tp))
            curr_scope.declared[ident] = info = FuncInfo.from_param_info(
                curr_scope, ident, params,
                ret_type=VoidType(), subscope=subscope)
            inner_funcs.append((info, fn))  # Store funcs for later walking
            # Skip walking body, only walk inner after collecting all declared
            #  variables in outer scope so function can use all variables
            #  declared in outer scope - even the ones declared below it)
            return True

        curr_scope = curr_scope or Scope()
        scope_stack = parent_scopes or []
        scope_stack.append(curr_scope)
        inner_funcs: list[tuple[FuncInfo, AstDefine]] = []
        # Walk self
        walker = (FilteredWalker()
                  .register_enter(AstIdent, enter_ident)
                  .register_enter(AstDeclNode, enter_decl)
                  .register_enter(AstDefine, enter_fn_decl))
        walker.walk(block)
        # Walk sub-functions
        for fn_info, fn_decl in inner_funcs:
            fn_info.subscope = self.run_on_new_scope(
                fn_decl.body, scope_stack, fn_info.subscope)
        return scope_stack.pop()  # Remove current scope from stack & return it

    def err(self, msg: str, region: StrRegion):
        return NameResolutionError(msg, region, self.src)


class TypecheckError(BaseLocatedError):
    """Errors raised by the typechecker"""


NodeTypecheckFn: TypeAlias = 'Callable[[Typechecker, AstNode], TypeInfo | None]'

_typecheck_dispatch: dict[type[AstNode], NodeTypecheckFn] = {}


class Typechecker:
    _curr_scope: Scope

    def __init__(self, name_resolver: NameResolver):
        self.resolver = name_resolver
        self.src = self.resolver.src
        self.is_ok: bool | None = None

    def _init(self):
        self.resolver.run()
        self.ast = self.resolver.ast
        self.top_scope = self.resolver.top_scope
        self._curr_scope = self.top_scope

    def run(self):
        if self.is_ok is None:
            return self.is_ok
        self._init()
        self._typecheck(self.ast)
        self.is_ok = True  # didn't raise any errors
        return self.is_ok

    def _node_typechecker(self, tp=None):
        if tp is None:
            assert callable(self)
            tp = self  # Called as decor in this class

        def decor(fn: NodeTypecheckFn):
            _typecheck_dispatch[tp] = fn
            return fn
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
            if (tp := self._typecheck(smt)) is not None:
                self.expect_type(tp, VoidType(), smt)

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

    def _resolve_scope(self, scope_tp: VarDeclScope):
        return self.top_scope if scope_tp == VarDeclScope.GLOBAL else self._curr_scope

    def err(self, msg: str, loc: RegionUnionArgT):
        return TypecheckError(msg, region_union(loc), self.src)

    def expect_type(self, actual: TypeInfo, exp: TypeInfo, loc: RegionUnionArgT):
        if exp != actual:
            # TODO: maybe better type formatting
            raise self.err(f"Expected type {exp}, got type {actual}", loc)
