from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from parser.astgen.ast_node import (
    AstNode, walk_ast, WalkableT, WalkerCallType, AstIdent, AstDeclNode,
    AstDefine, VarDeclType, VarDeclScope)
from parser.astgen.astgen import AstGen
from parser.common import BaseLocatedError, StrRegion
from util import flatten_force

WT = TypeVar('WT', bound=WalkableT)
VT = TypeVar('VT')

SpecificCbT = Callable[[WT], bool | None]
SpecificCbsDict = dict[type[WT] | type, list[Callable[[WT], bool | None]]]
BothCbT = Callable[[WT, WalkerCallType], bool | None]
BothCbsDict = dict[type[WT] | type, list[Callable[[WT, WalkerCallType], bool | None]]]


class _WalkerRegistry:
    def __init__(self, enter_cbs: SpecificCbsDict = (),
                 exit_cbs: SpecificCbsDict = (),
                 both_sbc: BothCbsDict = ()):
        self.enter_cbs: SpecificCbsDict = dict(enter_cbs)  # Copy them,
        self.exit_cbs: SpecificCbsDict = dict(exit_cbs)    # also converts default () -> {}
        self.both_cbs: BothCbsDict = dict(both_sbc)

    def copy(self):
        return _WalkerRegistry(self.enter_cbs, self.exit_cbs, self.both_cbs)

    def register_both(self, t: type[WT], fn: Callable[[WT, WalkerCallType], bool | None]):
        self.both_cbs.setdefault(t, []).append(fn)
        return self

    def register_enter(self, t: type[WT], fn: Callable[[WT], bool | None]):
        self.enter_cbs.setdefault(t, []).append(fn)
        return self

    def register_exit(self, t: type[WT], fn: Callable[[WT], bool | None]):
        self.exit_cbs.setdefault(t, []).append(fn)
        return self

    def __call__(self, *args, **kwargs):
        return self

    def on_enter(self, *tps: type[WT] | type):
        """Decorator version of register_enter."""
        def decor(fn: SpecificCbT):
            for t in tps:
                self.register_enter(t, fn)
            return fn
        return decor

    def on_exit(self, *tps: type[WT] | type):
        """Decorator version of register_exit."""
        def decor(fn: SpecificCbT):
            for t in tps:
                self.register_exit(t, fn)
            return fn
        return decor

    def on_both(self, *tps: type[WT] | type):
        """Decorator version of register_both."""
        def decor(fn: BothCbT):
            for t in tps:
                self.register_both(t, fn)
            return fn
        return decor


# TODO: This does not belong in this module!
class FilteredWalker(_WalkerRegistry):
    def __init__(self):
        cls_reg = self.class_registry()
        super().__init__(cls_reg.enter_cbs, cls_reg.exit_cbs, cls_reg.both_cbs)

    @classmethod
    def class_registry(cls) -> _WalkerRegistry:
        return _WalkerRegistry()

    @classmethod
    def create_cls_registry(cls, fn=None):
        """Create a class-level registry that can be added to using decorators.

        This can be used in two ways (at the top of your class)::

            # MUST be this name
            class_registry = FilteredWalker.create_cls_registry()

        or::

            @classmethod
            @FilteredWalker.create_cls_registry
            def class_registry(cls):  # MUST be this name
                pass

        and when registering methods::

            @class_registry.on_enter(AstDefine)
            def enter_define(self, ...):
                ...

        The restrictions on name are because we have no other way of detecting
         it (without metaclass dark magic) as we can't refer to the class while
         its namespace is being evaluated
        """
        if fn is not None and (parent := fn(cls)) is not None:
            return _WalkerRegistry.copy(parent)
        return _WalkerRegistry()

    def __call__(self, o: WalkableT, call_type: WalkerCallType):
        result = None
        # Call more specific ones first
        specific_cbs = self.enter_cbs if call_type == WalkerCallType.PRE else self.exit_cbs
        for fn in self._get_funcs(specific_cbs, type(o)):
            result = fn(o) or result
        for fn in self._get_funcs(self.both_cbs, type(o)):
            result = fn(o, call_type) or result
        return result

    @classmethod
    def _get_funcs(cls, mapping: dict[type[WT] | type, list[VT]], tp: type[WT]) -> list[VT]:
        """Also looks at superclasses/MRO"""
        return flatten_force(mapping.get(sub, []) for sub in tp.mro())


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
            for tp, param in fn.params:
                if tp.id not in PARAM_TYPES:
                    raise self.err("Unknown parameter type", tp.region)
                if param.id in subscope.declared:
                    raise self.err("There is already a parameter of this name", param.region)
                tp = BoolType() if param.id == 'bool' else ValType()
                subscope.declared[param.id] = NameInfo(subscope, param.id, tp, is_param=True)
                params.append(ParamInfo(param.id, tp))
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
        walk_ast(block, walker)
        # Walk sub-functions
        for fn_info, fn_decl in inner_funcs:
            fn_info.subscope = self.run_on_new_scope(
                fn_decl.body, scope_stack, fn_info.subscope)
        return scope_stack.pop()  # Remove current scope from stack & return it

    def err(self, msg: str, region: StrRegion):
        return NameResolutionError(msg, region, self.src)


class Typechecker:
    def __init__(self, name_resolver: NameResolver):
        self.resolver = name_resolver
        self.src = self.resolver.src
        self.is_ok: bool | None = None

    def _init(self):
        self.resolver.run()
        self.ast = self.resolver.ast
        self.top_scope = self.resolver.top_scope

    def run(self):
        if self.is_ok is None:
            return self.is_ok
        self._typecheck()
        self.is_ok = True
        return self.is_ok

    def _typecheck(self):
        walker = FilteredWalker()

        self.ast.walk(walker)
        ...


